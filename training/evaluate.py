# training/evaluate.py
# ─────────────────────────────────────────────
# Full evaluation pipeline for CartComplete.
#
# Fixes applied vs previous version:
#   - AddOnRecSys constructor: n_items→num_items,
#     n_food_groups→num_categories
#   - TwoTowerModel constructor: correct params
#     d_in, d_tower, d_out
#   - encode_cart: build item embs via item_encoder
#     first, then pass to two_tower.encode_cart
#   - model() call: all param names corrected
#   - PMI loading: load_npz instead of np.load
#   - expand_cart for prices: expand_cart_2d + squeeze
#   - model output: ['add_logit'] not direct tensor
# ─────────────────────────────────────────────

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import faiss
from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from training.ranking_dataset import RankingDataset
from training.collate import get_collator
from models.Add_on_RecSys import AddOnRecSys
from models.two_tower import TwoTowerModel


# ── utility helpers ───────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_cart_embs(
    batch:   Dict[str, torch.Tensor],
    model:   AddOnRecSys,
    i:       int,
    n_cands: int,
    device:  torch.device,
) -> torch.Tensor:
    """
    Build cart item embeddings for sample i expanded to n_cands.
    Uses model.item_encoder (shared with full ranker).

    Returns: (n_cands, N, d_model)
    """
    N  = batch['cart_item_idxs'].size(1)
    BN = n_cands * N

    # Expand single sample to n_cands copies
    ids_exp  = batch['cart_item_idxs'][i:i+1].expand(n_cands, N).reshape(BN)
    cats_exp = batch['cart_food_groups'][i:i+1].expand(n_cands, N).reshape(BN)
    pri_exp  = batch['cart_prices'][i:i+1].squeeze(-1).expand(n_cands, N).reshape(BN)
    txt_exp  = batch['cart_text_embs'][i:i+1].expand(n_cands, N, -1).reshape(BN, -1)

    with torch.no_grad():
        embs = model.item_encoder(
            item_id  = ids_exp,
            category = cats_exp,
            price    = pri_exp,
            text_emb = txt_exp,
        )   # (BN, d_model)

    return embs.view(n_cands, N, model.d_model)   # (n_cands, N, d_model)


# ── retrieval evaluation ──────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Evaluates the Two-Tower retrieval model against
    the full FAISS index.

    Metrics:
        Recall@K — fraction of carts where positive
                   is in top-K retrieved candidates
        MRR@K    — mean reciprocal rank
    """

    def __init__(
        self,
        index_path: str,
        meta_path:  str,
        device:     torch.device,
    ):
        print(f"  Loading FAISS index from {index_path}...")
        self.index  = faiss.read_index(index_path)
        self.device = device

        with open(meta_path) as f:
            meta = json.load(f)

        self.product_ids = np.array(meta['product_ids'], dtype=np.int64)
        self.pid_to_pos  = {
            pid: pos for pos, pid in enumerate(self.product_ids)
        }
        print(f"  FAISS index loaded: {self.index.ntotal:,} items")

    @torch.no_grad()
    def evaluate(
        self,
        model:     TwoTowerModel,
        full_model: AddOnRecSys,
        loader:    DataLoader,
        k_values:  List[int] = [5, 10, 20, 50],
    ) -> Dict[str, float]:

        model.eval()
        full_model.eval()

        recall_meters = {k: AverageMeter(f"Recall@{k}") for k in k_values}
        mrr_meters    = {k: AverageMeter(f"MRR@{k}")    for k in k_values}
        max_k         = max(k_values)

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            B     = batch['cart_mask'].size(0)
            N     = batch['cart_item_idxs'].size(1)
            BN    = B * N

            # Build cart item embeddings via shared item encoder
            ids_flat  = batch['cart_item_idxs'].view(BN)
            cats_flat = batch['cart_food_groups'].view(BN)
            pri_flat  = batch['cart_prices'].squeeze(-1).view(BN)
            txt_flat  = batch['cart_text_embs'].view(BN, -1)

            with torch.no_grad():
                item_embs = full_model.item_encoder(
                    item_id  = ids_flat,
                    category = cats_flat,
                    price    = pri_flat,
                    text_emb = txt_flat,
                ).view(B, N, full_model.d_model)

            # Encode cart through two-tower cart tower
            cart_vecs = model.encode_cart(
                cart_embs = batch['cart_text_embs'],
                cart_mask = batch['cart_mask'],
            )   # (B, d_out)

            cart_vecs_np = nn.functional.normalize(
                cart_vecs, dim=-1
            ).cpu().numpy().astype(np.float32)

            # FAISS search
            _, indices = self.index.search(cart_vecs_np, max_k)

            pos_pids = batch['pos_item_idx'].cpu().numpy()

            for i in range(B):
                retrieved_pids = self.product_ids[indices[i]]
                pos_pid        = pos_pids[i]

                rank = None
                for r, pid in enumerate(retrieved_pids):
                    if pid == pos_pid:
                        rank = r + 1
                        break

                for k in k_values:
                    hit = 1.0 if (rank is not None and rank <= k) else 0.0
                    recall_meters[k].update(hit)
                    rr = (1.0 / rank) if (
                        rank is not None and rank <= k
                    ) else 0.0
                    mrr_meters[k].update(rr)

        metrics = {}
        for k in k_values:
            metrics[f"Recall@{k}"] = recall_meters[k].avg
            metrics[f"MRR@{k}"]    = mrr_meters[k].avg

        return metrics


# ── ranking evaluation ────────────────────────────────────────────────────────

class RankingEvaluator:
    """
    Full end-to-end evaluation:
        Stage 1 : FAISS retrieval → candidate pool
        Stage 2 : Full ranker scores each candidate
        Stage 3 : Rank by score → HR@K, NDCG@K, MRR

    Also computes Category Coverage Rate — the novel
    evaluation metric from your architecture design.
    """

    def __init__(
        self,
        index_path:     str,
        meta_path:      str,
        items_path:     str,
        text_embs_path: str,
        pid2idx_path:   str,
        pmi_path:       str,
        device:         torch.device,
        retrieval_k:    int = 100,
    ):
        print(f"  Loading FAISS index...")
        self.index       = faiss.read_index(index_path)
        self.device      = device
        self.retrieval_k = retrieval_k

        with open(meta_path) as f:
            meta = json.load(f)
        self.product_ids = np.array(meta['product_ids'], dtype=np.int64)

        print(f"  Loading item features...")
        items        = pd.read_parquet(items_path)
        self.items   = items.set_index('product_id')

        self.text_embs    = np.load(text_embs_path)
        self.text_emb_dim = self.text_embs.shape[1]

        with open(pid2idx_path) as f:
            pid2idx_raw  = json.load(f)
        self.pid2idx = {int(k): int(v) for k, v in pid2idx_raw.items()}

        # Fix: use load_npz for sparse PMI matrix
        print(f"  Loading PMI matrix...")
        self.pmi_matrix = load_npz(pmi_path)
        self.pmi_max    = 5.0

        self.fg2idx = {
            'main': 0, 'side': 1, 'drink': 2,
            'snack': 3, 'dessert': 4,
        }
        print(f"  RankingEvaluator ready.")

    def _get_item_features(
        self,
        product_ids: np.ndarray,
        device:      torch.device,
    ) -> Dict[str, torch.Tensor]:
        B           = len(product_ids)
        item_idxs   = np.zeros(B,  dtype=np.int64)
        text_embs   = np.zeros((B, self.text_emb_dim), dtype=np.float32)
        prices      = np.zeros((B,), dtype=np.float32)
        food_groups = np.zeros(B,  dtype=np.int64)
        popularity  = np.zeros((B,), dtype=np.float32)

        for i, pid in enumerate(product_ids):
            idx          = self.pid2idx.get(int(pid), 0)
            item_idxs[i] = idx

            if idx > 0 and idx < len(self.text_embs):
                text_embs[i] = self.text_embs[idx]

            if int(pid) in self.items.index:
                row            = self.items.loc[int(pid)]
                prices[i]      = float(row.get('price', 4.0)) / 20.0
                fg_str         = str(row.get('food_group', 'side'))
                food_groups[i] = self.fg2idx.get(fg_str, 1)
                popularity[i]  = float(row.get('popularity_rank', 0.5))
            else:
                prices[i]      = 4.0 / 20.0
                food_groups[i] = 1
                popularity[i]  = 0.5

        return {
            'item_idxs':   torch.tensor(item_idxs,   device=device),
            'text_embs':   torch.tensor(text_embs,   device=device),
            'prices':      torch.tensor(prices,       device=device),
            'food_groups': torch.tensor(food_groups, device=device),
            'popularity':  torch.tensor(popularity,  device=device),
        }

    def _get_cross_features(
        self,
        cart_pids:      List[int],
        candidate_pids: np.ndarray,
    ) -> torch.Tensor:
        n_cands = len(candidate_pids)
        feats   = np.zeros((n_cands, 5), dtype=np.float32)

        cart_prices = []
        cart_fgs    = set()
        for pid in cart_pids:
            if int(pid) in self.items.index:
                row = self.items.loc[int(pid)]
                cart_prices.append(float(row.get('price', 4.0)))
                cart_fgs.add(str(row.get('food_group', 'side')))
        mean_cart_price = np.mean(cart_prices) if cart_prices else 4.0

        for i, cand_pid in enumerate(candidate_pids):
            cand_idx = self.pid2idx.get(int(cand_pid), 0)

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
            co_score = float(np.clip(
                np.mean(pmi_vals) / self.pmi_max if pmi_vals else 0.0,
                0.0, 1.0
            ))

            if int(cand_pid) in self.items.index:
                cand_fg    = str(
                    self.items.loc[int(cand_pid)].get('food_group', 'side')
                )
                cand_price = float(
                    self.items.loc[int(cand_pid)].get('price', 4.0)
                )
                pop        = float(
                    self.items.loc[int(cand_pid)].get('popularity_rank', 0.5)
                )
            else:
                cand_fg    = 'side'
                cand_price = 4.0
                pop        = 0.5

            cat_gap     = 1.0 if cand_fg not in cart_fgs else 0.0
            price_ratio = float(np.clip(
                cand_price / (mean_cart_price + 1e-6) / 5.0, 0.0, 1.0
            ))
            novelty     = 1.0 - pop

            feats[i] = [pmi_score, co_score, cat_gap, price_ratio, novelty]

        return torch.tensor(feats, dtype=torch.float32)

    def _category_coverage_rate(
        self,
        recommendations: List[List[int]],
        k:               int = 10,
    ) -> float:
        n_groups  = len(self.fg2idx)
        coverages = []

        for rec_pids in recommendations:
            top_k_pids  = rec_pids[:k]
            groups_seen = set()
            for pid in top_k_pids:
                if int(pid) in self.items.index:
                    fg = str(
                        self.items.loc[int(pid)].get('food_group', 'side')
                    )
                    groups_seen.add(fg)
            coverages.append(len(groups_seen) / n_groups)

        return float(np.mean(coverages)) if coverages else 0.0

    @torch.no_grad()
    def evaluate(
        self,
        model:      AddOnRecSys,
        two_tower:  TwoTowerModel,
        loader:     DataLoader,
        k_values:   List[int] = [5, 10],
    ) -> Dict[str, float]:

        model.eval()
        two_tower.eval()

        hr_meters   = {k: AverageMeter(f"HR@{k}")   for k in k_values}
        ndcg_meters = {k: AverageMeter(f"NDCG@{k}") for k in k_values}
        mrr_meter   = AverageMeter("MRR")

        all_recommendations = []
        n_evaluated         = 0

        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            B     = batch['cart_mask'].size(0)
            N     = batch['cart_item_idxs'].size(1)
            BN    = B * N

            # Build cart embeddings for FAISS query
            ids_flat  = batch['cart_item_idxs'].view(BN)
            cats_flat = batch['cart_food_groups'].view(BN)
            pri_flat  = batch['cart_prices'].squeeze(-1).view(BN)
            txt_flat  = batch['cart_text_embs'].view(BN, -1)

            item_embs_all = model.item_encoder(
                item_id  = ids_flat,
                category = cats_flat,
                price    = pri_flat,
                text_emb = txt_flat,
            ).view(B, N, model.d_model)

            cart_vecs = two_tower.encode_cart(
                cart_embs = batch['cart_text_embs'],
                cart_mask = batch['cart_mask'],
            )
            cart_vecs_np = nn.functional.normalize(
                cart_vecs, dim=-1
            ).cpu().numpy().astype(np.float32)

            _, faiss_indices = self.index.search(
                cart_vecs_np, self.retrieval_k
            )

            pos_pids = batch['pos_item_idx'].cpu().numpy()

            for i in range(B):
                cand_pids = self.product_ids[faiss_indices[i]].copy()
                pos_pid   = pos_pids[i]

                # Inject positive if missing from candidates
                if pos_pid not in cand_pids:
                    cand_pids = np.append(cand_pids[:-1], pos_pid)

                n_cands = len(cand_pids)

                # Cart pids for cross features (approx from idxs)
                cart_length  = int(batch['cart_mask'][i].sum().item())
                cart_pids_i  = [
                    int(batch['cart_item_idxs'][i, j].item())
                    for j in range(cart_length)
                ]

                # Candidate features
                cand_feats  = self._get_item_features(cand_pids, self.device)
                cross_feats = self._get_cross_features(
                    cart_pids_i, cand_pids
                ).to(self.device)

                cart_item_embs_i = (
                batch['cart_text_embs'][i:i+1]
                .expand(n_cands, N, -1)
                .to(self.device)
                )  # (n_cands, N, d_model)

                # Dummy user inputs
                dummy_baskets = torch.zeros(
                    n_cands, 1, model.d_model, device=self.device
                )
                dummy_ufeats  = torch.zeros(n_cands, 8, device=self.device)

                # Full ranker forward pass — correct param names
                scores_out = model(
                    cart_item_ids    = batch['cart_item_idxs'][i:i+1].expand(n_cands, N),
                    cart_categories  = batch['cart_food_groups'][i:i+1].expand(n_cands, N),
                    cart_prices      = batch['cart_prices'].squeeze(-1)[i:i+1].expand(n_cands, N),
                    cart_text_embs   = batch['cart_text_embs'][i:i+1].expand(n_cands, N, -1),
                    cart_mask        = batch['cart_mask'][i:i+1].expand(n_cands, N),
                    cand_item_id     = cand_feats['item_idxs'],
                    cand_category    = cand_feats['food_groups'],
                    cand_price       = cand_feats['prices'],
                    cand_text_emb    = cand_feats['text_embs'],
                    cand_food_group  = cand_feats['food_groups'],
                    cand_popularity  = cand_feats['popularity'],
                    past_basket_embs = dummy_baskets,
                    user_features    = dummy_ufeats,
                    basket_mask      = None,
                    hour             = batch['hour'][i:i+1].expand(n_cands),
                    day_of_week      = batch['dow'][i:i+1].expand(n_cands),
                    meal_slot        = batch['meal_period'][i:i+1].expand(n_cands),
                    restaurant_id    = batch['restaurant_id'][i:i+1].expand(n_cands),
                    pmi_score        = cross_feats[:, 0],
                    co_occur_score   = cross_feats[:, 1],
                    cart_total       = batch['cart_prices'].squeeze(-1)[i].sum().expand(n_cands),
                    cart_size        = batch['cart_mask'][i].sum().float().expand(n_cands),
                    return_logit     = True,
                )
                scores = scores_out['add_logit'].squeeze(-1)   # (n_cands,)

                sorted_idx  = scores.argsort(descending=True).cpu().numpy()
                ranked_pids = cand_pids[sorted_idx]

                rank = None
                for r, pid in enumerate(ranked_pids):
                    if pid == pos_pid:
                        rank = r + 1
                        break

                for k in k_values:
                    hit  = 1.0 if (rank is not None and rank <= k) else 0.0
                    hr_meters[k].update(hit)

                    in_k = rank is not None and rank <= k
                    dcg  = (1.0 / np.log2(rank + 1)) if in_k else 0.0
                    idcg = 1.0 / np.log2(2)
                    ndcg_meters[k].update(dcg / idcg)

                rr = (1.0 / rank) if rank is not None else 0.0
                mrr_meter.update(rr)

                all_recommendations.append(ranked_pids[:10].tolist())
                n_evaluated += 1

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"    Evaluated {n_evaluated:,} | "
                    f"HR@10={hr_meters[10].avg:.4f} | "
                    f"NDCG@10={ndcg_meters[10].avg:.4f} | "
                    f"MRR={mrr_meter.avg:.4f}"
                )

        ccr = self._category_coverage_rate(all_recommendations, k=10)

        metrics = {f"HR@{k}":   hr_meters[k].avg   for k in k_values}
        metrics.update(
                   {f"NDCG@{k}": ndcg_meters[k].avg for k in k_values}
        )
        metrics['MRR']                    = mrr_meter.avg
        metrics['Category_Coverage_Rate'] = ccr
        metrics['n_evaluated']            = n_evaluated

        return metrics


# ── main evaluation function ──────────────────────────────────────────────────

def evaluate(
    model_checkpoint:  str   = "artifacts/checkpoints/full_model.pt",
    retrieval_ckpt:    str   = "artifacts/checkpoints/retrieval_tower.pt",
    test_pairs_path:   str   = "outputs/test_pairs_instacart.parquet",
    items_path:        str   = "outputs/items_instacart.parquet",
    text_embs_path:    str   = "outputs/text_embeddings_instacart.npy",
    pid2idx_path:      str   = "outputs/pid2idx_instacart.json",
    pmi_path:          str   = "outputs/pmi_matrix_instacart.npz",
    index_path:        str   = "artifacts/indexes/item_index.faiss",
    meta_path:         str   = "artifacts/indexes/item_index_meta.json",
    results_path:      str   = "results/final_metrics.json",
    text_emb_dim:      int   = 384,
    d_model:           int   = 128,
    n_food_groups:     int   = 5,
    n_cross_features:  int   = 5,
    num_restaurants:   int   = 5000,
    backbone_dim:      int   = 256,
    max_cart_len:      int   = 50,
    batch_size:        int   = 64,
    num_workers:       int   = 4,
    retrieval_k:       int   = 100,
    k_values:          list  = [5, 10],
    max_test_samples:  Optional[int] = None,
):
    start  = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  CartComplete — Full Evaluation")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {model_checkpoint}")
    print(f"{'='*60}\n")

    with open(pid2idx_path) as f:
        pid2idx = json.load(f)
    n_items = len(pid2idx)

    # ── load full model ───────────────────────────────────────────────────────
    print("Loading AddOnRecSys model...")
    model = AddOnRecSys(
        num_items        = n_items,           # fixed: was n_items=
        num_categories   = n_food_groups,     # fixed: was n_food_groups=
        text_emb_dim     = text_emb_dim,
        d_model          = d_model,
        n_cross_features = n_cross_features,
        n_biz_features   = 4,
        num_restaurants  = num_restaurants,
        backbone_dim     = backbone_dim,
        max_cart_len     = max_cart_len,
    ).to(device)

    ckpt = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  Model loaded (epoch {ckpt.get('epoch', '?')})")

    # ── load two-tower ────────────────────────────────────────────────────────
    print("\nLoading TwoTowerModel...")
    two_tower = TwoTowerModel(
        d_in    = 384,      # fixed: was n_items/text_emb_dim
        d_tower = 128,
        d_out   = 64,
        dropout = 0.1,
    ).to(device)

    ret_ckpt = torch.load(retrieval_ckpt, map_location=device)
    two_tower.load_state_dict(ret_ckpt['model'])
    two_tower.eval()
    print(f"  Two-tower loaded")

    # ── test dataloader ───────────────────────────────────────────────────────
    print("\nBuilding test dataloader...")
    test_ds = RankingDataset(
        pairs_path     = test_pairs_path,
        items_path     = items_path,
        text_embs_path = text_embs_path,
        pid2idx_path   = pid2idx_path,
        pmi_path       = pmi_path,
        max_cart_len   = max_cart_len,
        max_samples    = max_test_samples,
    )
    collator    = get_collator(mode='ranking', dynamic_padding=False)
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        collate_fn  = collator,
    )
    print(f"  Test samples : {len(test_ds):,}")
    print(f"  Test batches : {len(test_loader):,}")

    all_metrics = {}

    # ── evaluation 1: retrieval ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Evaluation 1 — Retrieval (FAISS)")
    print(f"{'─'*60}")

    ret_evaluator     = RetrievalEvaluator(index_path, meta_path, device)
    retrieval_metrics = ret_evaluator.evaluate(
        model      = two_tower,
        full_model = model,
        loader     = test_loader,
        k_values   = k_values + [20, 50],
    )

    print(f"\n  Retrieval Results:")
    for k, v in retrieval_metrics.items():
        print(f"    {k:<15} : {v:.4f}")
    all_metrics['retrieval'] = retrieval_metrics

    # ── evaluation 2: end-to-end ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Evaluation 2 — End-to-End (Retrieval + Ranking)")
    print(f"{'─'*60}")

    rank_evaluator  = RankingEvaluator(
        index_path     = index_path,
        meta_path      = meta_path,
        items_path     = items_path,
        text_embs_path = text_embs_path,
        pid2idx_path   = pid2idx_path,
        pmi_path       = pmi_path,
        device         = device,
        retrieval_k    = retrieval_k,
    )
    ranking_metrics = rank_evaluator.evaluate(
        model     = model,
        two_tower = two_tower,
        loader    = test_loader,
        k_values  = k_values,
    )

    print(f"\n  End-to-End Results:")
    for k, v in ranking_metrics.items():
        if k != 'n_evaluated':
            print(f"    {k:<25} : {v:.4f}")
    print(f"    {'n_evaluated':<25} : {ranking_metrics['n_evaluated']:,}")
    all_metrics['ranking'] = ranking_metrics

    # ── summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - start

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Retrieval:")
    print(f"    Recall@10  : {retrieval_metrics.get('Recall@10', 0):.4f}")
    print(f"    Recall@50  : {retrieval_metrics.get('Recall@50', 0):.4f}")
    print(f"    MRR@10     : {retrieval_metrics.get('MRR@10',    0):.4f}")
    print(f"  End-to-End:")
    print(f"    HR@5       : {ranking_metrics.get('HR@5',                  0):.4f}")
    print(f"    HR@10      : {ranking_metrics.get('HR@10',                 0):.4f}")
    print(f"    NDCG@5     : {ranking_metrics.get('NDCG@5',                0):.4f}")
    print(f"    NDCG@10    : {ranking_metrics.get('NDCG@10',               0):.4f}")
    print(f"    MRR        : {ranking_metrics.get('MRR',                   0):.4f}")
    print(f"    Coverage   : {ranking_metrics.get('Category_Coverage_Rate',0):.4f}")
    print(f"  Eval time   : {format_time(total_time)}")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Results saved → {results_path}")

    return all_metrics


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint',  default="artifacts/checkpoints/full_model.pt")
    parser.add_argument('--retrieval_ckpt',    default="artifacts/checkpoints/retrieval_tower.pt")
    parser.add_argument('--test_pairs',        default="outputs/test_pairs_instacart.parquet")
    parser.add_argument('--items',             default="outputs/items_instacart.parquet")
    parser.add_argument('--text_embs',         default="outputs/text_embeddings_instacart.npy")
    parser.add_argument('--pid2idx',           default="outputs/pid2idx_instacart.json")
    parser.add_argument('--pmi_path',          default="outputs/pmi_matrix_instacart.npz")
    parser.add_argument('--index_path',        default="artifacts/indexes/item_index.faiss")
    parser.add_argument('--meta_path',         default="artifacts/indexes/item_index_meta.json")
    parser.add_argument('--results_path',      default="results/final_metrics.json")
    parser.add_argument('--text_emb_dim',      type=int,  default=384)
    parser.add_argument('--d_model',           type=int,  default=128)
    parser.add_argument('--n_food_groups',     type=int,  default=5)
    parser.add_argument('--n_cross_features',  type=int,  default=5)
    parser.add_argument('--num_restaurants',   type=int,  default=5000)
    parser.add_argument('--backbone_dim',      type=int,  default=256)
    parser.add_argument('--max_cart_len',      type=int,  default=50)
    parser.add_argument('--batch_size',        type=int,  default=64)
    parser.add_argument('--num_workers',       type=int,  default=4)
    parser.add_argument('--retrieval_k',       type=int,  default=100)
    parser.add_argument('--max_test_samples',  type=int,  default=None)

    args = parser.parse_args()

    evaluate(
        model_checkpoint = args.model_checkpoint,
        retrieval_ckpt   = args.retrieval_ckpt,
        test_pairs_path  = args.test_pairs,
        items_path       = args.items,
        text_embs_path   = args.text_embs,
        pid2idx_path     = args.pid2idx,
        pmi_path         = args.pmi_path,
        index_path       = args.index_path,
        meta_path        = args.meta_path,
        results_path     = args.results_path,
        text_emb_dim     = args.text_emb_dim,
        d_model          = args.d_model,
        n_food_groups    = args.n_food_groups,
        n_cross_features = args.n_cross_features,
        num_restaurants  = args.num_restaurants,
        backbone_dim     = args.backbone_dim,
        max_cart_len     = args.max_cart_len,
        batch_size       = args.batch_size,
        num_workers      = args.num_workers,
        retrieval_k      = args.retrieval_k,
        max_test_samples = args.max_test_samples,
    )