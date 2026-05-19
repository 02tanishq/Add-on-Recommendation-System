import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import faiss
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ── recommendation result dataclass ──────────────────────────────────────────

@dataclass
class AddOnRecommendation:
    """
    Single add-on recommendation result.

    Returned by the recommend() function for each
    candidate item surfaced to the user.
    """
    product_id:    int
    product_name:  str
    food_group:    str
    price:         float
    score:         float              # ranker logit
    rank:          int                # 1 = top recommendation
    pmi_score:     float = 0.0        # PMI with cart items
    novelty_score: float = 0.0        # 1 - popularity
    reason:        str   = ""         # human-readable explanation


@dataclass
class RecommendationResult:
    """
    Full result object returned by the recommender.

    Contains top-K add-on recommendations plus
    metadata about the request and retrieval.
    """
    cart_items:       List[str]                  # product names in cart
    recommendations:  List[AddOnRecommendation]  # ranked add-ons
    retrieval_time_ms: float = 0.0
    ranking_time_ms:   float = 0.0
    total_time_ms:     float = 0.0
    n_candidates:      int   = 0
    context:           Dict  = field(default_factory=dict)


# ── feature builder ───────────────────────────────────────────────────────────

class FeatureBuilder:
    """
    Builds all feature tensors needed for retrieval and ranking
    from raw product_ids and context signals.

    Loaded once at startup and reused across all requests.
    """

    def __init__(
        self,
        items_path:     str,
        text_embs_path: str,
        pid2idx_path:   str,
        pmi_path:       str,
        device:         torch.device,
    ):
        # ── item metadata ─────────────────────────────────────────────────────
        items       = pd.read_parquet(items_path)
        self.items  = items.set_index('product_id')

        # ── text embeddings ───────────────────────────────────────────────────
        self.text_embs    = np.load(text_embs_path)
        self.text_emb_dim = self.text_embs.shape[1]

        # ── pid → idx mapping ─────────────────────────────────────────────────
        with open(pid2idx_path) as f:
            raw = json.load(f)
        self.pid2idx = {int(k): int(v) for k, v in raw.items()}

        # ── PMI matrix ────────────────────────────────────────────────────────
        from scipy.sparse import load_npz
        self.pmi_matrix = load_npz(pmi_path)
        self.pmi_max    = 5.0

        # ── food group map ────────────────────────────────────────────────────
        self.fg2idx = {
            'main': 0, 'side': 1, 'drink': 2,
            'snack': 3, 'dessert': 4,
        }
        self.idx2fg = {v: k for k, v in self.fg2idx.items()}

        self.device = device

    def get_item_meta(self, product_id: int) -> Dict:
        """Returns raw metadata dict for a product_id."""
        if product_id in self.items.index:
            row = self.items.loc[product_id]
            return {
                'name':       str(row.get('product_name', f'Item {product_id}')),
                'food_group': str(row.get('food_group', 'side')),
                'price':      float(row.get('price', 4.0)),
                'popularity': float(row.get('popularity_rank', 0.5)),
            }
        return {
            'name':       f'Item {product_id}',
            'food_group': 'side',
            'price':      4.0,
            'popularity': 0.5,
        }

    def build_item_tensors(
        self,
        product_ids: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """
        Builds feature tensors for a batch of product_ids.

        Returns dict with keys matching AddOnRecSys.forward() signature.
        """
        B = len(product_ids)

        item_idxs   = np.zeros(B,  dtype=np.int64)
        text_embs   = np.zeros((B, self.text_emb_dim), dtype=np.float32)
        prices      = np.zeros((B, 1), dtype=np.float32)
        food_groups = np.zeros(B,  dtype=np.int64)
        popularity  = np.zeros((B, 1), dtype=np.float32)

        for i, pid in enumerate(product_ids):
            idx            = self.pid2idx.get(int(pid), 0)
            item_idxs[i]   = idx

            if idx > 0 and idx < len(self.text_embs):
                text_embs[i] = self.text_embs[idx]

            meta           = self.get_item_meta(int(pid))
            prices[i]      = meta['price'] / 20.0
            food_groups[i] = self.fg2idx.get(meta['food_group'], 1)
            popularity[i]  = meta['popularity']

        return {
            'item_idxs':   torch.tensor(item_idxs,   device=self.device),
            'text_embs':   torch.tensor(text_embs,   device=self.device),
            'prices':      torch.tensor(prices,       device=self.device),
            'food_groups': torch.tensor(food_groups, device=self.device),
            'popularity':  torch.tensor(popularity,  device=self.device),
        }

    def build_cart_tensors(
        self,
        cart_pids:   List[int],
        max_cart_len: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """
        Builds padded cart tensors from a list of product_ids.

        Returns dict matching the cart_* inputs of AddOnRecSys.forward().
        """
        N = max_cart_len

        cart_item_idxs   = np.zeros(N,  dtype=np.int64)
        cart_text_embs   = np.zeros((N, self.text_emb_dim), dtype=np.float32)
        cart_prices      = np.zeros((N, 1), dtype=np.float32)
        cart_food_groups = np.zeros(N,  dtype=np.int64)
        cart_popularity  = np.zeros((N, 1), dtype=np.float32)
        cart_mask        = np.zeros(N,  dtype=np.int64)

        cart_pids_trunc = cart_pids[-N:]

        for i, pid in enumerate(cart_pids_trunc):
            idx                  = self.pid2idx.get(int(pid), 0)
            cart_item_idxs[i]    = idx

            if idx > 0 and idx < len(self.text_embs):
                cart_text_embs[i] = self.text_embs[idx]

            meta                 = self.get_item_meta(int(pid))
            cart_prices[i]       = meta['price'] / 20.0
            cart_food_groups[i]  = self.fg2idx.get(meta['food_group'], 1)
            cart_popularity[i]   = meta['popularity']
            cart_mask[i]         = 1

        return {
            'cart_item_idxs':   torch.tensor(
                cart_item_idxs[None],   device=self.device
            ),   # (1, N)
            'cart_text_embs':   torch.tensor(
                cart_text_embs[None],   device=self.device
            ),   # (1, N, 384)
            'cart_prices':      torch.tensor(
                cart_prices[None],      device=self.device
            ),   # (1, N, 1)
            'cart_food_groups': torch.tensor(
                cart_food_groups[None], device=self.device
            ),   # (1, N)
            'cart_popularity':  torch.tensor(
                cart_popularity[None],  device=self.device
            ),   # (1, N, 1)
            'cart_mask':        torch.tensor(
                cart_mask[None],        device=self.device
            ),   # (1, N)
        }

    def build_cross_features(
        self,
        cart_pids:    List[int],
        cand_pids:    np.ndarray,
    ) -> torch.Tensor:
        """
        Computes explicit cross features between cart and all candidates.

        Features (5-dim per candidate):
            [0] pmi_score
            [1] co_occurrence_score
            [2] category_gap_score
            [3] price_ratio
            [4] novelty_score

        Returns: (n_cands, 5)
        """
        n_cands     = len(cand_pids)
        feats       = np.zeros((n_cands, 5), dtype=np.float32)

        # Cart-level stats
        cart_prices = []
        cart_fgs    = set()
        for pid in cart_pids:
            meta = self.get_item_meta(int(pid))
            cart_prices.append(meta['price'])
            cart_fgs.add(meta['food_group'])
        mean_cart_price = np.mean(cart_prices) if cart_prices else 4.0

        for i, cand_pid in enumerate(cand_pids):
            cand_idx  = self.pid2idx.get(int(cand_pid), 0)
            cand_meta = self.get_item_meta(int(cand_pid))

            # PMI + co-occurrence
            pmi_vals = []
            for cpid in cart_pids:
                cidx = self.pid2idx.get(int(cpid), 0)
                if cidx > 0 and cand_idx > 0:
                    try:
                        pmi_vals.append(
                            float(self.pmi_matrix[cidx, cand_idx])
                        )
                    except Exception:
                        pmi_vals.append(0.0)
                else:
                    pmi_vals.append(0.0)

            pmi_score = float(np.clip(
                max(pmi_vals) / self.pmi_max if pmi_vals else 0.0,
                0.0, 1.0
            ))
            co_score  = float(np.clip(
                np.mean(pmi_vals) / self.pmi_max if pmi_vals else 0.0,
                0.0, 1.0
            ))

            cat_gap     = 1.0 if cand_meta['food_group'] not in cart_fgs else 0.0
            price_ratio = float(np.clip(
                cand_meta['price'] / (mean_cart_price + 1e-6) / 5.0,
                0.0, 1.0
            ))
            novelty     = 1.0 - cand_meta['popularity']

            feats[i] = [pmi_score, co_score, cat_gap, price_ratio, novelty]

        return torch.tensor(feats, device=self.device)


# ── explainer ─────────────────────────────────────────────────────────────────

class ReasonGenerator:
    """
    Generates simple human-readable reasons for each recommendation.

    Rules-based on the explicit cross features — keeps it
    interpretable without needing a separate model.

    Used in the LinkedIn demo output.
    """

    TEMPLATES = {
        'pmi':       "frequently ordered together with items in your cart",
        'gap_drink': "adds a drink to complete your meal",
        'gap_side':  "adds a side to complement your main",
        'gap_dessert': "adds a sweet finish to your order",
        'gap_snack': "adds a snack to your order",
        'novelty':   "a popular item you haven't tried yet",
        'price':     "great value addition to your cart",
        'default':   "pairs well with your current order",
    }

    def generate(
        self,
        cand_meta:    Dict,
        cross_feats:  np.ndarray,   # (5,)
        cart_fgs:     set,
    ) -> str:
        pmi_score  = cross_feats[0]
        cat_gap    = cross_feats[2]
        novelty    = cross_feats[4]
        food_group = cand_meta['food_group']

        if pmi_score > 0.5:
            return self.TEMPLATES['pmi']
        if cat_gap > 0.5:
            key = f'gap_{food_group}'
            return self.TEMPLATES.get(key, self.TEMPLATES['default'])
        if novelty > 0.7:
            return self.TEMPLATES['novelty']
        return self.TEMPLATES['default']


# ── main recommender ──────────────────────────────────────────────────────────

class AddOnRecommender:
    """
    CartComplete Add-On Recommendation Engine.

    Full inference pipeline:
        Stage 1 — FAISS ANN retrieval → ~100 candidates
        Stage 2 — Full AddOnRecSys ranker → scored candidates
        Stage 3 — Sort + top-K + reason generation

    Loaded once, reused for all requests (stateless per request).

    Usage
    ─────
    recommender = AddOnRecommender.from_artifacts(
        artifacts_dir = "artifacts",
        checkpoint    = "artifacts/checkpoints/full_model.pt",
    )

    result = recommender.recommend(
        cart_product_ids = [196, 25133, 38928],
        top_k            = 5,
        hour             = 19,
        dow              = 5,
        meal_period      = 2,
        restaurant_id    = 1,
    )

    for rec in result.recommendations:
        print(f"{rec.rank}. {rec.product_name} — {rec.reason}")
    """

    def __init__(
        self,
        model:          'AddOnRecSys',
        two_tower:      'TwoTowerModel',
        feature_builder: FeatureBuilder,
        faiss_index:    faiss.Index,
        product_ids:    np.ndarray,
        device:         torch.device,
        max_cart_len:   int   = 50,
        retrieval_k:    int   = 100,
    ):
        self.model           = model
        self.two_tower       = two_tower
        self.fb              = feature_builder
        self.index           = faiss_index
        self.product_ids     = product_ids
        self.device          = device
        self.max_cart_len    = max_cart_len
        self.retrieval_k     = retrieval_k
        self.reason_gen      = ReasonGenerator()

        self.model.eval()
        self.two_tower.eval()

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir:  str,
        checkpoint:     str,
        retrieval_ckpt: str,
        device:         Optional[torch.device] = None,
        retrieval_k:    int   = 100,
        max_cart_len:   int   = 50,
        # model config
        text_emb_dim:   int   = 384,
        d_model:        int   = 64,
        n_food_groups:  int   = 5,
        n_cross_features: int = 8,
        num_restaurants: int  = 5000,
        backbone_dim:   int   = 256,
    ) -> 'AddOnRecommender':
        """
        Convenience constructor — loads everything from artifacts dir.

        Parameters
        ----------
        artifacts_dir   : root artifacts directory
        checkpoint      : path to full_model.pt
        retrieval_ckpt  : path to retrieval_tower.pt
        """
        from models.Add_on_RecSys import AddOnRecSys
        from models.two_tower import TwoTowerModel

        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )

        print(f"\n{'='*55}")
        print(f"  Loading CartComplete Add-On Recommender")
        print(f"  Device : {device}")
        print(f"{'='*55}")

        # ── paths ─────────────────────────────────────────────────────────────
        items_path     = 'outputs/items_instacart.parquet'
        text_embs_path = 'outputs/text_embeddings_instacart.npy'
        pid2idx_path   = 'outputs/pid2idx_instacart.json'
        pmi_path       = 'outputs/pmi_matrix_instacart.npz'
        index_path     = os.path.join(
            artifacts_dir, 'indexes',     'item_index.faiss'
        )
        meta_path      = os.path.join(
            artifacts_dir, 'indexes',     'item_index_meta.json'
        )

        # ── feature builder ───────────────────────────────────────────────────
        print("\n  Building feature builder...")
        fb = FeatureBuilder(
            items_path     = items_path,
            text_embs_path = text_embs_path,
            pid2idx_path   = pid2idx_path,
            pmi_path       = pmi_path,
            device         = device,
        )

        # ── n_items ───────────────────────────────────────────────────────────
        n_items = len(fb.pid2idx)

        # ── load full ranker ──────────────────────────────────────────────────
        print(f"  Loading AddOnRecSys from {checkpoint}...")
        model = AddOnRecSys(
            num_items        = n_items,
            num_categories   = n_food_groups,
            text_emb_dim     = text_emb_dim,
            d_model          = d_model,
            n_cross_features = n_cross_features,
            n_biz_features   = 4,
            num_restaurants  = num_restaurants,
            backbone_dim     = backbone_dim,
            max_cart_len     = max_cart_len,
        ).to(device)

        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        print(f"  Ranker loaded  (epoch {ckpt.get('epoch','?')})")

        # ── load two-tower ────────────────────────────────────────────────────
        print(f"  Loading TwoTowerModel from {retrieval_ckpt}...")
        two_tower = TwoTowerModel(
            d_in    = 384,
            d_tower = 128,
            d_out   = 64,
            dropout = 0.1,
        ).to(device)

        ret_ckpt = torch.load(retrieval_ckpt, map_location=device)
        two_tower.load_state_dict(ret_ckpt['model'])
        two_tower.eval()
        print(f"  Two-tower loaded")

        # ── load FAISS index ──────────────────────────────────────────────────
        print(f"  Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)

        with open(meta_path) as f:
            meta = json.load(f)
        product_ids = np.array(meta['product_ids'], dtype=np.int64)
        print(f"  FAISS index loaded: {index.ntotal:,} items")

        print(f"\n  Recommender ready.\n{'='*55}\n")

        return cls(
            model           = model,
            two_tower       = two_tower,
            feature_builder = fb,
            faiss_index     = index,
            product_ids     = product_ids,
            device          = device,
            max_cart_len    = max_cart_len,
            retrieval_k     = retrieval_k,
        )

    @torch.no_grad()
    def recommend(
        self,
        cart_product_ids: List[int],
        top_k:            int  = 5,
        hour:             int  = 12,
        dow:              int  = 0,
        meal_period:      int  = 1,
        restaurant_id:    int  = 0,
        filter_in_cart:   bool = True,
    ) -> RecommendationResult:
        """
        Generate top-K add-on recommendations for a given cart.

        Parameters
        ----------
        cart_product_ids : ordered list of product_ids in the cart
                           (order matters — most recent = last)
        top_k            : number of recommendations to return
        hour             : current hour 0-23
        dow              : day of week 0-6
        meal_period      : 0=breakfast 1=lunch 2=dinner 3=late-night
        restaurant_id    : restaurant integer ID
        filter_in_cart   : remove items already in cart from results

        Returns
        -------
        RecommendationResult with ranked AddOnRecommendation list
        """
        total_start = time.time()

        # ── stage 1: retrieval ────────────────────────────────────────────────
        ret_start = time.time()

        cart_tensors = self.fb.build_cart_tensors(
            cart_pids    = cart_product_ids,
            max_cart_len = self.max_cart_len,
        )

        # Build cart item embeddings via shared item encoder
        _N  = cart_tensors['cart_item_idxs'].size(1)
        _BN = 1 * _N
        _cart_item_embs = self.model.item_encoder(
            item_id  = cart_tensors['cart_item_idxs'].view(_BN),
            category = cart_tensors['cart_food_groups'].view(_BN),
            price    = cart_tensors['cart_prices'].squeeze(-1).view(_BN),
            text_emb = cart_tensors['cart_text_embs'].view(_BN, -1),
        ).view(1, _N, self.model.d_model)

        cart_emb = self.two_tower.encode_cart( 
            cart_embs = cart_tensors['cart_text_embs'],
            cart_mask = cart_tensors['cart_mask'],
        )   # (1, d_out)

        cart_emb_np = nn.functional.normalize(
            cart_emb, dim=-1
        ).cpu().numpy().astype(np.float32)

        _, faiss_indices = self.index.search(
            cart_emb_np, self.retrieval_k
        )   # (1, retrieval_k)

        cand_pids = self.product_ids[faiss_indices[0]]   # (retrieval_k,)

        # Filter items already in cart
        if filter_in_cart:
            cart_set  = set(cart_product_ids)
            cand_pids = np.array([
                pid for pid in cand_pids
                if pid not in cart_set
            ])

        ret_time_ms = (time.time() - ret_start) * 1000

        # ── stage 2: ranking ──────────────────────────────────────────────────
        rank_start = time.time()

        n_cands     = len(cand_pids)
        cand_feats  = self.fb.build_item_tensors(cand_pids)
        cross_feats = self.fb.build_cross_features(
            cart_pids  = cart_product_ids,
            cand_pids  = cand_pids,
        )   # (n_cands, 5)

        # Expand cart tensors to (n_cands, N, *)
        def expand(key, n):
            t = cart_tensors[key]
            if t.dim() == 2:
                return t.expand(n, -1)
            return t.expand(n, -1, -1)

        ctx = torch.tensor

        _dummy_baskets = torch.zeros(
            n_cands, 1, self.model.d_model, device=self.device
        )
        _dummy_ufeats  = torch.zeros(n_cands, 8, device=self.device)

        scores_out = self.model(
            cart_item_ids    = expand('cart_item_idxs',   n_cands),
            cart_categories  = expand('cart_food_groups', n_cands),
            cart_prices      = expand('cart_prices',      n_cands).squeeze(-1),
            cart_text_embs   = expand('cart_text_embs',   n_cands),
            cart_mask        = expand('cart_mask',        n_cands),
            cand_item_id     = cand_feats['item_idxs'],
            cand_category    = cand_feats['food_groups'],
            cand_price       = cand_feats['prices'].squeeze(-1),
            cand_text_emb    = cand_feats['text_embs'],
            cand_food_group  = cand_feats['food_groups'],
            cand_popularity  = cand_feats['popularity'].squeeze(-1),
            past_basket_embs = _dummy_baskets,
            user_features    = _dummy_ufeats,
            basket_mask      = None,
            hour             = torch.full(
                (n_cands,), hour, dtype=torch.long, device=self.device
            ),
            day_of_week      = torch.full(
                (n_cands,), dow,  dtype=torch.long, device=self.device
            ),
            meal_slot        = torch.full(
                (n_cands,), meal_period,   dtype=torch.long, device=self.device
            ),
            restaurant_id    = torch.full(
                (n_cands,), restaurant_id, dtype=torch.long, device=self.device
            ),
            pmi_score        = cross_feats[:, 0],
            co_occur_score   = cross_feats[:, 1],
            cart_total       = cart_tensors['cart_prices'].squeeze(-1).sum(dim=1).expand(n_cands),
            cart_size        = cart_tensors['cart_mask'].sum(dim=1).float().expand(n_cands),
            return_logit     = True,
        )
        scores = torch.sigmoid(
            scores_out['add_logit'].squeeze(-1) * 40
        ).cpu().numpy()   # (n_cands,)

        rank_time_ms  = (time.time() - rank_start) * 1000
        total_time_ms = (time.time() - total_start) * 1000

        # ── stage 3: sort + build results ─────────────────────────────────────
        sorted_idx = np.argsort(scores)[::-1]   # descending

        cross_feats_np = cross_feats.cpu().numpy()

        cart_names = [
            self.fb.get_item_meta(pid)['name']
            for pid in cart_product_ids
        ]

        # Food groups in cart for reason generation
        cart_fgs = {
            self.fb.get_item_meta(pid)['food_group']
            for pid in cart_product_ids
        }

        recommendations = []
        for rank, idx in enumerate(sorted_idx[:top_k], start=1):
            pid       = int(cand_pids[idx])
            meta      = self.fb.get_item_meta(pid)
            cf        = cross_feats_np[idx]
            reason    = self.reason_gen.generate(meta, cf, cart_fgs)

            recommendations.append(AddOnRecommendation(
                product_id    = pid,
                product_name  = meta['name'],
                food_group    = meta['food_group'],
                price         = meta['price'],
                score         = float(scores[idx]),
                rank          = rank,
                pmi_score     = float(cf[0]),
                novelty_score = float(cf[4]),
                reason        = reason,
            ))

        return RecommendationResult(
            cart_items        = cart_names,
            recommendations   = recommendations,
            retrieval_time_ms = ret_time_ms,
            ranking_time_ms   = rank_time_ms,
            total_time_ms     = total_time_ms,
            n_candidates      = n_cands,
            context           = {
                'hour':          hour,
                'dow':           dow,
                'meal_period':   meal_period,
                'restaurant_id': restaurant_id,
            },
        )

    def recommend_batch(
        self,
        carts:         List[List[int]],
        top_k:         int = 5,
        hour:          int = 12,
        dow:           int = 0,
        meal_period:   int = 1,
        restaurant_id: int = 0,
    ) -> List[RecommendationResult]:
        """
        Batch recommendation for multiple carts.
        Processes each cart independently — no batch FAISS yet.
        """
        return [
            self.recommend(
                cart_product_ids = cart,
                top_k            = top_k,
                hour             = hour,
                dow              = dow,
                meal_period      = meal_period,
                restaurant_id    = restaurant_id,
            )
            for cart in carts
        ]


# ── pretty printer ────────────────────────────────────────────────────────────

def print_result(result: RecommendationResult):
    """Prints a RecommendationResult in a clean readable format."""

    print(f"\n{'='*55}")
    print(f"  CART")
    print(f"{'─'*55}")
    for i, name in enumerate(result.cart_items, 1):
        print(f"    {i}. {name}")

    print(f"\n{'─'*55}")
    print(f"  TOP-{len(result.recommendations)} ADD-ON RECOMMENDATIONS")
    print(f"{'─'*55}")

    for rec in result.recommendations:
        print(
            f"  {rec.rank}. {rec.product_name:<30} "
            f"${rec.price:>5.2f}  "
            f"[{rec.food_group}]"
        )
        print(f"     → {rec.reason}")
        print(
            f"     PMI: {rec.pmi_score:.3f}  "
            f"Novelty: {rec.novelty_score:.3f}  "
            f"Score: {rec.score:.10f}"
        )

    print(f"\n{'─'*55}")
    print(
        f"  Retrieval : {result.retrieval_time_ms:.1f}ms  |  "
        f"Ranking : {result.ranking_time_ms:.1f}ms  |  "
        f"Total : {result.total_time_ms:.1f}ms"
    )
    print(f"  Candidates evaluated : {result.n_candidates}")
    print(f"{'='*55}\n")


# ── entry point / demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CartComplete Add-On Recommender — Demo"
    )

    parser.add_argument(
        '--artifacts_dir', default="artifacts"
    )
    parser.add_argument(
        '--checkpoint',
        default="artifacts/checkpoints/full_model.pt"
    )
    parser.add_argument(
        '--retrieval_ckpt',
        default="artifacts/checkpoints/retrieval_tower.pt"
    )
    parser.add_argument(
        '--cart', nargs='+', type=int,
        default=[196, 25133, 38928],
        help="Product IDs in cart"
    )
    parser.add_argument('--top_k',         type=int, default=5)
    parser.add_argument('--hour',          type=int, default=19)
    parser.add_argument('--dow',           type=int, default=5)
    parser.add_argument('--meal_period',   type=int, default=2)
    parser.add_argument('--restaurant_id', type=int, default=0)

    args = parser.parse_args()

    # ── load recommender ──────────────────────────────────────────────────────
    recommender = AddOnRecommender.from_artifacts(
        artifacts_dir  = args.artifacts_dir,
        checkpoint     = args.checkpoint,
        retrieval_ckpt = args.retrieval_ckpt,
    )

    # ── run recommendation ────────────────────────────────────────────────────
    result = recommender.recommend(
        cart_product_ids = args.cart,
        top_k            = args.top_k,
        hour             = args.hour,
        dow              = args.dow,
        meal_period      = args.meal_period,
        restaurant_id    = args.restaurant_id,
    )

    # ── print results ─────────────────────────────────────────────────────────
    print_result(result)