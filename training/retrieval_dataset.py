# training/retrieval_dataset.py
# ─────────────────────────────────────────────
# PyTorch Dataset for training the Two-Tower
# retrieval model.
#
# Each sample is a (cart, positive_item) pair.
# The two-tower model learns to place cart vectors
# close to their positive item vectors in the
# shared retrieval space via InfoNCE contrastive loss.
#
# Negatives are handled inside InBatchNegativeMiner
# in two_tower.py — no explicit negatives needed here.
#
# Data source:
#   outputs/train_pairs_instacart.parquet
#   outputs/items_instacart.parquet
#   outputs/text_embeddings_instacart.npy
#
# Each parquet row has:
#   cart       : JSON string of product_id list
#   positive   : int product_id
#   cart_size  : int
#   cart_total : float
#   hour       : int
#   dow        : int
# ─────────────────────────────────────────────

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple


class RetrievalDataset(Dataset):
    """
    Dataset for Two-Tower retrieval model training.

    Produces one sample per (cart, positive_item) pair.
    Each sample contains:
        cart_item_features   : (N, item_feat_dim)  — encoded cart items
        cart_mask            : (N,)                — 1=real 0=pad
        positive_item_feat   : (item_feat_dim,)    — positive candidate

    item_feat_dim is the concatenation of:
        item_id one-hot index    → looked up as integer, embedded in model
        text_embedding           → (384,) MiniLM
        price                    → (1,) scalar
        food_group_onehot        → (5,)
        popularity               → (1,) scalar

    The model handles embedding lookups internally.
    This dataset provides raw IDs and feature arrays.
    """

    N_FOOD_GROUPS = 5

    def __init__(
        self,
        pairs_path:      str,          # path to train/val/test_pairs_instacart.parquet
        items_path:      str,          # path to items_instacart.parquet
        text_embs_path:  str,          # path to text_embeddings_instacart.npy
        pid2idx_path:    str,          # path to pid2idx_instacart.json
        max_cart_len:    int  = 50,    # pad/truncate cart to this length
        max_samples:     Optional[int] = None,   # for fast debugging
    ):
        super().__init__()

        self.max_cart_len = max_cart_len

        # ── load pairs ────────────────────────────────────────────────────────
        print(f"  Loading pairs from {pairs_path}...")
        pairs = pd.read_parquet(pairs_path)

        # Decode cart from JSON string back to list
        pairs['cart']      = pairs['cart'].apply(json.loads)
        pairs['negatives'] = pairs['negatives'].apply(json.loads)

        if max_samples is not None:
            pairs = pairs.head(max_samples)

        self.pairs = pairs.reset_index(drop=True)
        print(f"  Pairs loaded: {len(self.pairs):,}")

        # ── load item metadata ────────────────────────────────────────────────
        print(f"  Loading items from {items_path}...")
        items = pd.read_parquet(items_path)
        self.items = items.set_index('product_id')

        # ── load text embeddings ──────────────────────────────────────────────
        print(f"  Loading text embeddings from {text_embs_path}...")
        self.text_embs = np.load(text_embs_path)   # (n_items, 384)
        print(f"  Text embeddings shape: {self.text_embs.shape}")

        # ── load product id → integer index map ───────────────────────────────
        with open(pid2idx_path) as f:
            pid2idx_raw = json.load(f)
        # Keys are stored as strings in JSON
        self.pid2idx = {int(k): int(v) for k, v in pid2idx_raw.items()}

        # ── food group → index map ────────────────────────────────────────────
        self.fg2idx = {
            'main':    0,
            'side':    1,
            'drink':   2,
            'snack':   3,
            'dessert': 4,
        }

        # ── item count for embedding table size ───────────────────────────────
        self.n_items = len(self.pid2idx)
        self.text_emb_dim = self.text_embs.shape[1]

        print(f"  Dataset ready: {len(self.pairs):,} pairs | "
              f"{self.n_items:,} items | "
              f"text_emb_dim={self.text_emb_dim}")

    def _get_item_features(self, product_id: int) -> Dict[str, torch.Tensor]:
        """
        Build feature dict for a single product_id.

        Returns dict with:
            item_idx    : int scalar tensor  — for embedding table lookup
            text_emb    : (384,) float       — MiniLM semantic embedding
            price       : (1,)  float        — normalized synthetic price
            food_group  : int scalar tensor  — food group index 0-4
            popularity  : (1,)  float        — normalized popularity rank
        """
        idx = self.pid2idx.get(product_id, 0)   # 0 = unknown/padding

        # Text embedding
        if idx > 0 and idx < len(self.text_embs):
            text_emb = torch.tensor(
                self.text_embs[idx], dtype=torch.float32
            )
        else:
            text_emb = torch.zeros(self.text_emb_dim, dtype=torch.float32)

        # Item metadata from items_instacart.parquet
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
            'item_idx':   torch.tensor(idx,        dtype=torch.long),
            'text_emb':   text_emb,
            'price':      torch.tensor([price / 20.0], dtype=torch.float32),
            'food_group': torch.tensor(food_group, dtype=torch.long),
            'popularity': torch.tensor([popularity], dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns one training sample:

        cart_item_idxs   : (max_cart_len,)          item indices for embedding
        cart_text_embs   : (max_cart_len, 384)       text embeddings
        cart_prices      : (max_cart_len, 1)         price scalars
        cart_food_groups : (max_cart_len,)           food group indices
        cart_popularity  : (max_cart_len, 1)         popularity scores
        cart_mask        : (max_cart_len,)            1=real 0=pad

        pos_item_idx     : ()                        positive item index
        pos_text_emb     : (384,)                    positive text emb
        pos_price        : (1,)
        pos_food_group   : ()
        pos_popularity   : (1,)

        hour             : ()                        order context
        dow              : ()
        """
        row = self.pairs.iloc[idx]

        cart_pids  = row['cart']        # list of product_ids
        pos_pid    = int(row['positive'])
        hour       = int(row.get('hour', 12))
        dow        = int(row.get('dow', 0))

        # ── encode cart items ─────────────────────────────────────────────────
        N = self.max_cart_len

        cart_item_idxs   = torch.zeros(N, dtype=torch.long)
        cart_text_embs   = torch.zeros(N, self.text_emb_dim)
        cart_prices      = torch.zeros(N, 1)
        cart_food_groups = torch.zeros(N, dtype=torch.long)
        cart_popularity  = torch.zeros(N, 1)
        cart_mask        = torch.zeros(N, dtype=torch.long)

        # Truncate if cart is longer than max_cart_len
        cart_pids_trunc = cart_pids[-N:]   # keep most recent items

        for i, pid in enumerate(cart_pids_trunc):
            feat = self._get_item_features(int(pid))
            cart_item_idxs[i]   = feat['item_idx']
            cart_text_embs[i]   = feat['text_emb']
            cart_prices[i]      = feat['price']
            cart_food_groups[i] = feat['food_group']
            cart_popularity[i]  = feat['popularity']
            cart_mask[i]        = 1

        # ── encode positive item ──────────────────────────────────────────────
        pos_feat = self._get_item_features(pos_pid)

        return {
            # cart
            'cart_item_idxs':   cart_item_idxs,    # (N,)
            'cart_text_embs':   cart_text_embs,    # (N, 384)
            'cart_prices':      cart_prices,        # (N, 1)
            'cart_food_groups': cart_food_groups,   # (N,)
            'cart_popularity':  cart_popularity,    # (N, 1)
            'cart_mask':        cart_mask,          # (N,)
            # positive candidate
            'pos_item_idx':     pos_feat['item_idx'],
            'pos_text_emb':     pos_feat['text_emb'],
            'pos_price':        pos_feat['price'],
            'pos_food_group':   pos_feat['food_group'],
            'pos_popularity':   pos_feat['popularity'],
            # context
            'hour':             torch.tensor(hour, dtype=torch.long),
            'dow':              torch.tensor(dow,  dtype=torch.long),
        }


class RetrievalItemDataset(Dataset):
    """
    Dataset for encoding ALL items in the catalog.
    Used by training/build_faiss_index.py to build
    the FAISS index after retrieval training.

    Returns one item per sample with all its features.
    The TwoTowerModel.encode_all_items() iterates this
    and stores the resulting vectors in the FAISS index.
    """

    def __init__(
        self,
        items_path:     str,
        text_embs_path: str,
        pid2idx_path:   str,
    ):
        super().__init__()

        items = pd.read_parquet(items_path)
        self.items = items.reset_index(drop=True)

        self.text_embs = np.load(text_embs_path)

        with open(pid2idx_path) as f:
            pid2idx_raw = json.load(f)
        self.pid2idx = {int(k): int(v) for k, v in pid2idx_raw.items()}

        self.fg2idx = {
            'main': 0, 'side': 1, 'drink': 2,
            'snack': 3, 'dessert': 4,
        }

        self.text_emb_dim = self.text_embs.shape[1]
        print(f"  ItemDataset: {len(self.items):,} items")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.items.iloc[idx]
        pid = int(row['product_id'])
        emb_idx = self.pid2idx.get(pid, 0)

        if emb_idx > 0 and emb_idx < len(self.text_embs):
            text_emb = torch.tensor(
                self.text_embs[emb_idx], dtype=torch.float32
            )
        else:
            text_emb = torch.zeros(self.text_emb_dim)

        price      = float(row.get('price', 4.0)) / 20.0
        fg_str     = str(row.get('food_group', 'side'))
        food_group = self.fg2idx.get(fg_str, 1)
        popularity = float(row.get('popularity_rank', 0.5))

        return {
            'product_id':  torch.tensor(pid,       dtype=torch.long),
            'item_idx':    torch.tensor(emb_idx,   dtype=torch.long),
            'text_emb':    text_emb,
            'price':       torch.tensor([price],   dtype=torch.float32),
            'food_group':  torch.tensor(food_group,dtype=torch.long),
            'popularity':  torch.tensor([popularity], dtype=torch.float32),
        }


def build_retrieval_dataloaders(
    train_pairs_path: str,
    val_pairs_path:   str,
    items_path:       str,
    text_embs_path:   str,
    pid2idx_path:     str,
    max_cart_len:     int = 50,
    batch_size:       int = 512,
    num_workers:      int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples:   Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to build train and val dataloaders.
    Called from training/train_retrieval.py.

    Returns (train_loader, val_loader)
    """
    train_ds = RetrievalDataset(
        pairs_path     = train_pairs_path,
        items_path     = items_path,
        text_embs_path = text_embs_path,
        pid2idx_path   = pid2idx_path,
        max_cart_len   = max_cart_len,
        max_samples    = max_train_samples,
    )

    val_ds = RetrievalDataset(
        pairs_path     = val_pairs_path,
        items_path     = items_path,
        text_embs_path = text_embs_path,
        pid2idx_path   = pid2idx_path,
        max_cart_len   = max_cart_len,
        max_samples    = max_val_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,    # InfoNCE needs full batches
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
# USAGE IN training/train_retrieval.py
# ─────────────────────────────────────────────
#
# from training.retrieval_dataset import build_retrieval_dataloaders
#
# train_loader, val_loader = build_retrieval_dataloaders(
#     train_pairs_path = "outputs/train_pairs_instacart.parquet",
#     val_pairs_path   = "outputs/val_pairs_instacart.parquet",
#     items_path       = "outputs/items_instacart.parquet",
#     text_embs_path   = "outputs/text_embeddings_instacart.npy",
#     pid2idx_path     = "outputs/pid2idx_instacart.json",
#     max_cart_len     = 50,
#     batch_size       = 512,
#     num_workers      = 4,
# )
#
# for batch in train_loader:
#     cart_embs = batch['cart_text_embs']     # (B, N, 384)
#     cart_mask = batch['cart_mask']          # (B, N)
#     pos_emb   = batch['pos_text_emb']       # (B, 384)
#     ...
# ─────────────────────────────────────────────


# ── sanity check ─────────────────────────────
if __name__ == "__main__":
    import os

    # Paths — adjust to your actual artifact paths
    BASE = "artifacts"
    PAIRS      = f"{BASE}/processed/train_pairs_instacart.parquet"
    ITEMS      = f"{BASE}/processed/items_instacart.parquet"
    TEXT_EMBS  = f"{BASE}/embeddings/text_embeddings_instacart.npy"
    PID2IDX    = f"{BASE}/mappings/pid2idx_instacart.json"

    if not all(os.path.exists(p) for p in [PAIRS, ITEMS, TEXT_EMBS, PID2IDX]):
        print("Artifact files not found — run preprocessing first.")
        print("Expected paths:")
        for p in [PAIRS, ITEMS, TEXT_EMBS, PID2IDX]:
            print(f"  {'✓' if os.path.exists(p) else '✕'} {p}")
    else:
        ds = RetrievalDataset(
            pairs_path     = PAIRS,
            items_path     = ITEMS,
            text_embs_path = TEXT_EMBS,
            pid2idx_path   = PID2IDX,
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