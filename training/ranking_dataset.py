import json
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple


class RankingDataset(Dataset):
    """
    Dataset for training the full CartComplete ranking model.

    Each sample is a (cart, positive_item, negative_item) triplet.
    The ranker learns via BPR loss:
        score(cart, positive) > score(cart, negative)

    Negative types (from preprocessing/negative_sampling.py):
        hard negatives     — items from same category, never added
        same-menu negatives — items from same restaurant, not added
        random negatives   — random items from catalog

    Data source:
        outputs/train_pairs_instacart.parquet
        outputs/items_instacart.parquet
        outputs/text_embeddings_instacart.npy

    Each parquet row has:
        cart        : JSON string of ordered product_id list
        positive    : int product_id  — item actually added
        negatives   : JSON list of int product_ids — sampled negatives
        hour        : int  0-23
        dow         : int  0-6
        meal_period : int  0-3  (breakfast/lunch/dinner/late-night)
        restaurant_id : int

    Explicit cross features computed per (cart, candidate) pair:
        pmi_score           — from PMI matrix
        co_occurrence_score — raw co-occurrence count normalised
        category_gap_score  — 1 if category not in cart else 0
        price_ratio         — candidate_price / mean_cart_price
        novelty_score       — 1 - popularity_rank (less popular = more novel)
    """

    N_FOOD_GROUPS  = 5
    N_MEAL_PERIODS = 5

    def __init__(
        self,
        pairs_path:      str,
        items_path:      str,
        text_embs_path:  str,
        pid2idx_path:    str,
        pmi_path:        str,            # outputs/pmi_matrix_instacart.npz
        max_cart_len:    int  = 50,
        n_negatives:     int  = 1,       # negatives per positive to sample
        max_samples:     Optional[int] = None,
    ):
        super().__init__()

        self.max_cart_len = max_cart_len
        self.n_negatives  = n_negatives

        # ── load pairs ────────────────────────────────────────────────────────
        print(f"  Loading ranking pairs from {pairs_path}...")
        pairs = pd.read_parquet(pairs_path)
        pairs['cart']      = pairs['cart'].apply(json.loads)
        pairs['negatives'] = pairs['negatives'].apply(json.loads)

        if max_samples is not None:
            pairs = pairs.head(max_samples)

        self.pairs = pairs.reset_index(drop=True)
        print(f"  Pairs loaded: {len(self.pairs):,}")

        # ── load items ────────────────────────────────────────────────────────
        print(f"  Loading items from {items_path}...")
        items = pd.read_parquet(items_path)
        self.items = items.set_index('product_id')

        # ── load text embeddings ──────────────────────────────────────────────
        print(f"  Loading text embeddings from {text_embs_path}...")
        self.text_embs    = np.load(text_embs_path)   # (n_items, 384)
        self.text_emb_dim = self.text_embs.shape[1]
        print(f"  Text embeddings shape: {self.text_embs.shape}")

        # ── load pid2idx ──────────────────────────────────────────────────────
        with open(pid2idx_path) as f:
            pid2idx_raw = json.load(f)
        self.pid2idx = {int(k): int(v) for k, v in pid2idx_raw.items()}
        self.n_items = len(self.pid2idx)

        # ── load PMI matrix ───────────────────────────────────────────────────
        print(f"  Loading PMI matrix from {pmi_path}...")
        self.pmi_matrix = load_npz(pmi_path)
        self.pmi_max    = 5.0
        print(f"  PMI matrix loaded.")

        # ── food group map ────────────────────────────────────────────────────
        self.fg2idx = {
            'main': 0, 'side': 1, 'drink': 2,
            'snack': 3, 'dessert': 4,
        }

        print(f"  RankingDataset ready: {len(self.pairs):,} pairs | "
              f"{self.n_items:,} items")

    # ── item feature builder ──────────────────────────────────────────────────

    def _get_item_features(self, product_id: int) -> Dict[str, torch.Tensor]:
        """
        Build feature tensors for a single product_id.
        Identical structure to RetrievalDataset._get_item_features()
        so both datasets produce compatible tensors.

        Returns:
            item_idx    : ()     long   — embedding table index
            text_emb    : (384,) float  — MiniLM embedding
            price       : (1,)   float  — normalised price
            food_group  : ()     long   — food group index 0-4
            popularity  : (1,)   float  — normalised popularity rank
        """
        idx = self.pid2idx.get(product_id, 0)

        if idx > 0 and idx < len(self.text_embs):
            text_emb = torch.tensor(
                self.text_embs[idx], dtype=torch.float32
            )
        else:
            text_emb = torch.zeros(self.text_emb_dim, dtype=torch.float32)

        if product_id in self.items.index:
            row        = self.items.loc[product_id]
            price      = float(row.get('price', 4.0))
            fg_str     = str(row.get('food_group', 'side'))
            popularity = float(row.get('popularity_rank', 0.5))
        else:
            price      = 4.0
            fg_str     = 'side'
            popularity = 0.5

        food_group = self.fg2idx.get(fg_str, 1)

        return {
            'item_idx':   torch.tensor(idx,         dtype=torch.long),
            'text_emb':   text_emb,
            'price':      torch.tensor([price / 20.0], dtype=torch.float32),
            'food_group': torch.tensor(food_group,  dtype=torch.long),
            'popularity': torch.tensor([popularity], dtype=torch.float32),
        }

    # ── explicit cross features ───────────────────────────────────────────────

    def _get_cross_features(
        self,
        cart_pids:    list,
        candidate_pid: int,
    ) -> torch.Tensor:
        """
        Compute explicit cross features between a cart and one candidate.

        Features (5-dim):
            [0] pmi_score           — max PMI between candidate and any cart item
            [1] co_occurrence_score — mean co-occurrence normalised 0-1
            [2] category_gap_score  — 1 if candidate category not in cart
            [3] price_ratio         — candidate_price / mean_cart_price
            [4] novelty_score       — 1 - candidate_popularity_rank

        Returns: (5,) float tensor
        """
        cand_idx = self.pid2idx.get(candidate_pid, 0)

        # PMI score — max over cart items
        pmi_scores = []
        for cart_pid in cart_pids:
            cart_idx = self.pid2idx.get(int(cart_pid), 0)
            if cart_idx > 0 and cand_idx > 0:
                try:
                    pmi_val = self.pmi_matrix[cart_idx, cand_idx]
                    pmi_scores.append(float(pmi_val))
                except Exception:
                    pmi_scores.append(0.0)
            else:
                pmi_scores.append(0.0)

        pmi_score = max(pmi_scores) / self.pmi_max if pmi_scores else 0.0
        pmi_score = float(np.clip(pmi_score, 0.0, 1.0))

        # Co-occurrence score — mean normalised
        co_score = float(np.clip(np.mean(pmi_scores) / self.pmi_max, 0.0, 1.0))

        # Category gap score
        if candidate_pid in self.items.index:
            cand_fg = str(self.items.loc[candidate_pid].get('food_group', 'side'))
        else:
            cand_fg = 'side'

        cart_fgs = set()
        for pid in cart_pids:
            if int(pid) in self.items.index:
                cart_fgs.add(
                    str(self.items.loc[int(pid)].get('food_group', 'side'))
                )

        category_gap = 1.0 if cand_fg not in cart_fgs else 0.0

        # Price ratio — candidate / mean cart price
        if candidate_pid in self.items.index:
            cand_price = float(
                self.items.loc[candidate_pid].get('price', 4.0)
            )
        else:
            cand_price = 4.0

        cart_prices = []
        for pid in cart_pids:
            if int(pid) in self.items.index:
                cart_prices.append(
                    float(self.items.loc[int(pid)].get('price', 4.0))
                )
        mean_cart_price = np.mean(cart_prices) if cart_prices else 4.0
        price_ratio = float(
            np.clip(cand_price / (mean_cart_price + 1e-6), 0.0, 5.0) / 5.0
        )

        # Novelty score — 1 - popularity (less popular = more novel)
        if candidate_pid in self.items.index:
            pop = float(
                self.items.loc[candidate_pid].get('popularity_rank', 0.5)
            )
        else:
            pop = 0.5
        novelty = 1.0 - pop

        return torch.tensor(
            [pmi_score, co_score, category_gap, price_ratio, novelty],
            dtype=torch.float32
        )   # (5,)

    # ── cart encoder ──────────────────────────────────────────────────────────

    def _encode_cart(self, cart_pids: list) -> Dict[str, torch.Tensor]:
        """
        Encode an ordered list of cart product_ids into padded tensors.

        Returns dict of padded tensors each shape (max_cart_len, *)
        plus cart_mask (max_cart_len,).
        """
        N = self.max_cart_len

        cart_item_idxs   = torch.zeros(N, dtype=torch.long)
        cart_text_embs   = torch.zeros(N, self.text_emb_dim)
        cart_prices      = torch.zeros(N, 1)
        cart_food_groups = torch.zeros(N, dtype=torch.long)
        cart_popularity  = torch.zeros(N, 1)
        cart_mask        = torch.zeros(N, dtype=torch.long)

        # Keep most recent items if cart is too long
        cart_pids_trunc = cart_pids[-N:]

        for i, pid in enumerate(cart_pids_trunc):
            feat = self._get_item_features(int(pid))
            cart_item_idxs[i]   = feat['item_idx']
            cart_text_embs[i]   = feat['text_emb']
            cart_prices[i]      = feat['price']
            cart_food_groups[i] = feat['food_group']
            cart_popularity[i]  = feat['popularity']
            cart_mask[i]        = 1

        return {
            'cart_item_idxs':   cart_item_idxs,    # (N,)
            'cart_text_embs':   cart_text_embs,    # (N, 384)
            'cart_prices':      cart_prices,        # (N, 1)
            'cart_food_groups': cart_food_groups,   # (N,)
            'cart_popularity':  cart_popularity,    # (N, 1)
            'cart_mask':        cart_mask,          # (N,)
        }

    # ── dataset methods ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns one BPR training triplet:

        Cart tensors (shared for pos and neg):
            cart_item_idxs   : (N,)
            cart_text_embs   : (N, 384)
            cart_prices      : (N, 1)
            cart_food_groups : (N,)
            cart_popularity  : (N, 1)
            cart_mask        : (N,)

        Positive candidate:
            pos_item_idx     : ()
            pos_text_emb     : (384,)
            pos_price        : (1,)
            pos_food_group   : ()
            pos_popularity   : (1,)
            pos_cross_feats  : (5,)   explicit cross features

        Negative candidate:
            neg_item_idx     : ()
            neg_text_emb     : (384,)
            neg_price        : (1,)
            neg_food_group   : ()
            neg_popularity   : (1,)
            neg_cross_feats  : (5,)   explicit cross features

        Context:
            hour             : ()
            dow              : ()
            meal_period      : ()
            restaurant_id    : ()
        """
        row = self.pairs.iloc[idx]

        cart_pids = row['cart']
        pos_pid   = int(row['positive'])
        neg_list  = row['negatives']

        hour = int(row.get('hour', 12))
        dow  = int(row.get('dow', 0))

        meal_raw = row.get('meal_period', 'lunch')

        meal_map = {
            'breakfast':  1,
            'lunch':      2,
            'afternoon':  3,
            'dinner':     4,
            'late_night': 5,
        }

        if isinstance(meal_raw, str):
            meal_period = meal_map.get(
                meal_raw.lower(),
                2
            )
        else:
            meal_period = int(meal_raw)

        rest_id = int(row.get('restaurant_id', 0))


        # Sample one negative from the available negatives list
        if len(neg_list) > 0:
            neg_pid = int(
                neg_list[np.random.randint(0, len(neg_list))]
            )
        else:
            # Fallback — random item from catalog
            neg_pid = int(
                np.random.choice(list(self.pid2idx.keys()))
            )

        # ── encode cart ───────────────────────────────────────────────────────
        cart_tensors = self._encode_cart(cart_pids)

        # ── encode positive ───────────────────────────────────────────────────
        pos_feat = self._get_item_features(pos_pid)
        pos_cross = self._get_cross_features(cart_pids, pos_pid)

        # ── encode negative ───────────────────────────────────────────────────
        neg_feat = self._get_item_features(neg_pid)
        neg_cross = self._get_cross_features(cart_pids, neg_pid)

        return {
            # ── cart ──────────────────────────────────────────────────────────
            **cart_tensors,

            # ── positive ──────────────────────────────────────────────────────
            'pos_item_idx':    pos_feat['item_idx'],       # ()
            'pos_text_emb':    pos_feat['text_emb'],       # (384,)
            'pos_price':       pos_feat['price'],          # (1,)
            'pos_food_group':  pos_feat['food_group'],     # ()
            'pos_popularity':  pos_feat['popularity'],     # (1,)
            'pos_cross_feats': pos_cross,                  # (5,)

            # ── negative ──────────────────────────────────────────────────────
            'neg_item_idx':    neg_feat['item_idx'],       # ()
            'neg_text_emb':    neg_feat['text_emb'],       # (384,)
            'neg_price':       neg_feat['price'],          # (1,)
            'neg_food_group':  neg_feat['food_group'],     # ()
            'neg_popularity':  neg_feat['popularity'],     # (1,)
            'neg_cross_feats': neg_cross,                  # (5,)

            # ── context ───────────────────────────────────────────────────────
            'hour':          torch.tensor(hour,        dtype=torch.long),
            'dow':           torch.tensor(dow,         dtype=torch.long),
            'meal_period':   torch.tensor(meal_period, dtype=torch.long),
            'restaurant_id': torch.tensor(rest_id,     dtype=torch.long),
        }


def build_ranking_dataloaders(
    train_pairs_path: str,
    val_pairs_path:   str,
    items_path:       str,
    text_embs_path:   str,
    pid2idx_path:     str,
    pmi_path:         str,
    max_cart_len:     int = 50,
    n_negatives:      int = 1,
    batch_size:       int = 256,
    num_workers:      int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples:   Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to build train and val dataloaders.
    Called from training/train_ranker.py.

    Returns (train_loader, val_loader)
    """
    train_ds = RankingDataset(
        pairs_path     = train_pairs_path,
        items_path     = items_path,
        text_embs_path = text_embs_path,
        pid2idx_path   = pid2idx_path,
        pmi_path       = pmi_path,
        max_cart_len   = max_cart_len,
        n_negatives    = n_negatives,
        max_samples    = max_train_samples,
    )

    val_ds = RankingDataset(
        pairs_path     = val_pairs_path,
        items_path     = items_path,
        text_embs_path = text_embs_path,
        pid2idx_path   = pid2idx_path,
        pmi_path       = pmi_path,
        max_cart_len   = max_cart_len,
        n_negatives    = n_negatives,
        max_samples    = max_val_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    print(f"\n  Train batches : {len(train_loader):,}")
    print(f"  Val batches   : {len(val_loader):,}")

    return train_loader, val_loader


# ─────────────────────────────────────────────
# USAGE IN training/train_ranker.py
# ─────────────────────────────────────────────
#
# from training.ranking_dataset import build_ranking_dataloaders
#
# train_loader, val_loader = build_ranking_dataloaders(
#     train_pairs_path = "outputs/train_pairs_instacart.parquet",
#     val_pairs_path   = "outputs/val_pairs_instacart.parquet",
#     items_path       = "outputs/items_instacart.parquet",
#     text_embs_path   = "outputs/text_embeddings_instacart.npy",
#     pid2idx_path     = "outputs/pid2idx_instacart.json",
#     pmi_path         = "outputs/pmi_matrix_instacart.npz",
#     max_cart_len     = 50,
#     n_negatives      = 1,
#     batch_size       = 256,
#     num_workers      = 4,
# )
#
# for batch in train_loader:
#     cart_mask      = batch['cart_mask']         # (B, N)
#     pos_cross      = batch['pos_cross_feats']   # (B, 5)
#     neg_cross      = batch['neg_cross_feats']   # (B, 5)
#     meal_period    = batch['meal_period']        # (B,)
#     ...
# ─────────────────────────────────────────────


# ── sanity check ─────────────────────────────
if __name__ == "__main__":
    import os

    BASE      = "artifacts"
    PAIRS     = f"{BASE}/processed/train_pairs_instacart.parquet"
    ITEMS     = f"{BASE}/processed/items_instacart.parquet"
    TEXT_EMBS = f"{BASE}/embeddings/text_embeddings_instacart.npy"
    PID2IDX   = f"{BASE}/mappings/pid2idx_instacart.json"
    PMI       = f"{BASE}/graphs/pmi_matrix_instacart.npz"

    required = [PAIRS, ITEMS, TEXT_EMBS, PID2IDX, PMI]
    if not all(os.path.exists(p) for p in required):
        print("Artifact files not found — run preprocessing first.")
        for p in required:
            print(f"  {'✓' if os.path.exists(p) else '✕'} {p}")
    else:
        ds = RankingDataset(
            pairs_path     = PAIRS,
            items_path     = ITEMS,
            text_embs_path = TEXT_EMBS,
            pid2idx_path   = PID2IDX,
            pmi_path       = PMI,
            max_cart_len   = 50,
            max_samples    = 100,
        )

        sample = ds[0]
        print("\nSample keys and shapes:")
        for k, v in sample.items():
            shape = v.shape if hasattr(v, 'shape') else v
            print(f"  {k:<22} : {shape}")

        loader = DataLoader(ds, batch_size=8, shuffle=False)
        batch  = next(iter(loader))
        print("\nBatch shapes:")
        for k, v in batch.items():
            print(f"  {k:<22} : {v.shape}")