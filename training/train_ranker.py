# training/train_ranker.py
# ─────────────────────────────────────────────
# Full training loop for CartComplete Add-On Ranker.
#
# Fixes applied vs previous version:
#   - cart_item_idxs  (not cart_item_ids)
#   - cart_food_groups (not cart_categories)
#   - pos_item_idx    (not cand_item_idxs)
#   - pos_text_emb    (not cand_text_embs)
#   - pos_price       (not cand_prices)
#   - criterion() now passes cart_food_groups + cart_mask
#   - validate() indentation fixed
#   - loss_dict key = 'total' — matches CartCompleteLoss
# ─────────────────────────────────────────────

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, Optional, Tuple

from training.ranking_dataset import build_ranking_dataloaders
from training.collate import get_collator
from training.losses import CartCompleteLoss
from models.Add_on_RecSys import AddOnRecSys


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


# ── ranking metrics ───────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ranking_metrics(
    model:      AddOnRecSys,
    val_loader: DataLoader,
    device:     torch.device,
    k_values:   list = [5, 10],
) -> Dict[str, float]:
    """
    HR@K, NDCG@K, MRR on validation set.
    Batch-level ranking: each positive ranked
    against all negatives in the same batch.
    """
    model.eval()

    hr_meters   = {k: AverageMeter(f"HR@{k}")   for k in k_values}
    ndcg_meters = {k: AverageMeter(f"NDCG@{k}") for k in k_values}
    mrr_meter   = AverageMeter("MRR")

    for batch in val_loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        B      = batch['cart_mask'].size(0)
        dummy_baskets = torch.zeros(B, 1, model.d_model, device=device)
        dummy_ufeats  = torch.zeros(B, 8, device=device)

        # ── score positives ───────────────────────────────────────────────────
        pos_out = model(
            cart_item_ids    = batch['cart_item_idxs'],
            cart_categories  = batch['cart_food_groups'],
            cart_prices      = batch['cart_prices'].squeeze(-1),
            cart_text_embs   = batch['cart_text_embs'],
            cart_mask        = batch['cart_mask'],
            cand_item_id     = batch['pos_item_idx'],
            cand_category    = batch['pos_food_group'],
            cand_price       = batch['pos_price'].squeeze(-1),
            cand_text_emb    = batch['pos_text_emb'],
            cand_food_group  = batch['pos_food_group'],
            cand_popularity  = batch['pos_popularity'].squeeze(-1),
            past_basket_embs = dummy_baskets,
            user_features    = dummy_ufeats,
            basket_mask      = None,
            hour             = batch['hour'],
            day_of_week      = batch['dow'],
            meal_slot        = batch['meal_period'],
            restaurant_id    = batch['restaurant_id'],
            cart_total       = batch['cart_prices'].squeeze(-1).sum(dim=1),
            cart_size        = batch['cart_mask'].sum(dim=1).float(),
            return_logit     = True,
        )
        pos_logits = pos_out['add_logit'].squeeze(-1)   # (B,)

        # ── score negatives ───────────────────────────────────────────────────
        neg_out = model(
            cart_item_ids    = batch['cart_item_idxs'],
            cart_categories  = batch['cart_food_groups'],
            cart_prices      = batch['cart_prices'].squeeze(-1),
            cart_text_embs   = batch['cart_text_embs'],
            cart_mask        = batch['cart_mask'],
            cand_item_id     = batch['neg_item_idx'],
            cand_category    = batch['neg_food_group'],
            cand_price       = batch['neg_price'].squeeze(-1),
            cand_text_emb    = batch['neg_text_emb'],
            cand_food_group  = batch['neg_food_group'],
            cand_popularity  = batch['neg_popularity'].squeeze(-1),
            past_basket_embs = dummy_baskets,
            user_features    = dummy_ufeats,
            basket_mask      = None,
            hour             = batch['hour'],
            day_of_week      = batch['dow'],
            meal_slot        = batch['meal_period'],
            restaurant_id    = batch['restaurant_id'],
            cart_total       = batch['cart_prices'].squeeze(-1).sum(dim=1),
            cart_size        = batch['cart_mask'].sum(dim=1).float(),
            return_logit     = True,
        )
        neg_logits = neg_out['add_logit'].squeeze(-1)   # (B,)

        # ── batch-level ranking ───────────────────────────────────────────────
        # (B, B+1): col 0 = positive, cols 1..B = all batch negatives
        score_matrix = torch.cat([
            pos_logits.unsqueeze(1),
            neg_logits.unsqueeze(0).expand(B, B),
        ], dim=1)   # (B, B+1)

        sorted_idx = score_matrix.argsort(dim=1, descending=True)
        pos_ranks  = (sorted_idx == 0).nonzero(as_tuple=False)
        pos_ranks  = pos_ranks[pos_ranks[:, 0].argsort()][:, 1] + 1  # 1-indexed

        for k in k_values:
            hits = (pos_ranks <= k).float()
            hr_meters[k].update(hits.mean().item(), n=B)

        for k in k_values:
            in_top_k = (pos_ranks <= k).float()
            dcg      = in_top_k / torch.log2(pos_ranks.float() + 1)
            idcg     = torch.ones_like(dcg) / torch.log2(
                torch.tensor(2.0, device=device)
            )
            ndcg = (dcg / idcg).clamp(0, 1)
            ndcg_meters[k].update(ndcg.mean().item(), n=B)

        rr = 1.0 / pos_ranks.float()
        mrr_meter.update(rr.mean().item(), n=B)

    metrics = {f"HR@{k}":   hr_meters[k].avg   for k in k_values}
    metrics.update({f"NDCG@{k}": ndcg_meters[k].avg for k in k_values})
    metrics['MRR'] = mrr_meter.avg
    return metrics


# ── one training epoch ────────────────────────────────────────────────────────

def train_one_epoch(
    model:     AddOnRecSys,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: CartCompleteLoss,
    device:    torch.device,
    epoch:     int,
    log_every: int = 50,
) -> Dict[str, float]:

    model.train()

    meters = {
        'total':    AverageMeter('Total'),
        'bpr':      AverageMeter('BPR'),
        'coverage': AverageMeter('Coverage'),
        'hard_neg': AverageMeter('HardNeg'),
    }

    start = time.time()

    for step, batch in enumerate(loader):

        batch = {k: v.to(device) for k, v in batch.items()}
        B     = batch['cart_mask'].size(0)

        # Dummy user inputs — real user histories loaded separately
        # when user_histories.pkl is integrated in a later phase
        dummy_baskets = torch.zeros(B, 1, model.d_model, device=device)
        dummy_ufeats  = torch.zeros(B, 8, device=device)

        # Shared cart args — avoid repeating in pos and neg calls
        cart_kwargs = dict(
            cart_item_ids    = batch['cart_item_idxs'],
            cart_categories  = batch['cart_food_groups'],
            cart_prices      = batch['cart_prices'].squeeze(-1),
            cart_text_embs   = batch['cart_text_embs'],
            cart_mask        = batch['cart_mask'],
            past_basket_embs = dummy_baskets,
            user_features    = dummy_ufeats,
            basket_mask      = None,
            hour             = batch['hour'],
            day_of_week      = batch['dow'],
            meal_slot        = batch['meal_period'],
            restaurant_id    = batch['restaurant_id'],
            cart_total       = batch['cart_prices'].squeeze(-1).sum(dim=1),
            cart_size        = batch['cart_mask'].sum(dim=1).float(),
            return_logit     = True,
        )

        # ── positive forward pass ─────────────────────────────────────────────
        pos_out = model(
            **cart_kwargs,
            cand_item_id    = batch['pos_item_idx'],
            cand_category   = batch['pos_food_group'],
            cand_price      = batch['pos_price'].squeeze(-1),
            cand_text_emb   = batch['pos_text_emb'],
            cand_food_group = batch['pos_food_group'],
            cand_popularity = batch['pos_popularity'].squeeze(-1),
        )
        pos_logits = pos_out['add_logit'].squeeze(-1)   # (B,)

        # ── negative forward pass ─────────────────────────────────────────────
        neg_out = model(
            **cart_kwargs,
            cand_item_id    = batch['neg_item_idx'],
            cand_category   = batch['neg_food_group'],
            cand_price      = batch['neg_price'].squeeze(-1),
            cand_text_emb   = batch['neg_text_emb'],
            cand_food_group = batch['neg_food_group'],
            cand_popularity = batch['neg_popularity'].squeeze(-1),
        )
        neg_logits = neg_out['add_logit'].squeeze(-1)   # (B,)

        # ── loss ──────────────────────────────────────────────────────────────
        # CartCompleteLoss.forward() signature:
        #   pos_scores, neg_scores, pos_food_groups,
        #   cart_food_groups, cart_mask, all_scores
        total_loss, loss_dict = criterion(
            pos_scores       = pos_logits,
            neg_scores       = neg_logits,
            pos_food_groups  = batch['pos_food_group'],
            cart_food_groups = batch['cart_food_groups'],
            cart_mask        = batch['cart_mask'],
            all_scores       = neg_logits,
        )

        # ── backward ──────────────────────────────────────────────────────────
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── update meters ─────────────────────────────────────────────────────
        for key in meters:
            if key in loss_dict:
                meters[key].update(loss_dict[key], n=B)

        # ── logging ───────────────────────────────────────────────────────────
        if (step + 1) % log_every == 0:
            elapsed = time.time() - start
            lr      = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:02d} | "
                f"Step {step+1:04d}/{len(loader):04d} | "
                f"Loss {meters['total'].avg:.4f} | "
                f"BPR {meters['bpr'].avg:.4f} | "
                f"Cov {meters['coverage'].avg:.4f} | "
                f"LR {lr:.2e} | "
                f"Time {format_time(elapsed)}"
            )

    return {'train_' + k: v.avg for k, v in meters.items()}


# ── validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:     AddOnRecSys,
    loader:    DataLoader,
    criterion: CartCompleteLoss,
    device:    torch.device,
    k_values:  list = [5, 10],
) -> Dict[str, float]:

    model.eval()
    loss_meter = AverageMeter('Val Loss')

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B     = batch['cart_mask'].size(0)

        dummy_baskets = torch.zeros(B, 1, model.d_model, device=device)
        dummy_ufeats  = torch.zeros(B, 8, device=device)

        cart_kwargs = dict(
            cart_item_ids    = batch['cart_item_idxs'],
            cart_categories  = batch['cart_food_groups'],
            cart_prices      = batch['cart_prices'].squeeze(-1),
            cart_text_embs   = batch['cart_text_embs'],
            cart_mask        = batch['cart_mask'],
            past_basket_embs = dummy_baskets,
            user_features    = dummy_ufeats,
            basket_mask      = None,
            hour             = batch['hour'],
            day_of_week      = batch['dow'],
            meal_slot        = batch['meal_period'],
            restaurant_id    = batch['restaurant_id'],
            cart_total       = batch['cart_prices'].squeeze(-1).sum(dim=1),
            cart_size        = batch['cart_mask'].sum(dim=1).float(),
            return_logit     = True,
        )

        pos_out = model(
            **cart_kwargs,
            cand_item_id    = batch['pos_item_idx'],
            cand_category   = batch['pos_food_group'],
            cand_price      = batch['pos_price'].squeeze(-1),
            cand_text_emb   = batch['pos_text_emb'],
            cand_food_group = batch['pos_food_group'],
            cand_popularity = batch['pos_popularity'].squeeze(-1),
        )
        pos_logits = pos_out['add_logit'].squeeze(-1)   # (B,)

        neg_out = model(
            **cart_kwargs,
            cand_item_id    = batch['neg_item_idx'],
            cand_category   = batch['neg_food_group'],
            cand_price      = batch['neg_price'].squeeze(-1),
            cand_text_emb   = batch['neg_text_emb'],
            cand_food_group = batch['neg_food_group'],
            cand_popularity = batch['neg_popularity'].squeeze(-1),
        )
        neg_logits = neg_out['add_logit'].squeeze(-1)   # (B,)

        _, loss_dict = criterion(
            pos_scores       = pos_logits,
            neg_scores       = neg_logits,
            pos_food_groups  = batch['pos_food_group'],
            cart_food_groups = batch['cart_food_groups'],
            cart_mask        = batch['cart_mask'],
            all_scores       = neg_logits,
        )
        loss_meter.update(loss_dict['total'], n=B)

    rank_metrics = compute_ranking_metrics(model, loader, device, k_values)

    return {'val_loss': loss_meter.avg, **rank_metrics}


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model:     AddOnRecSys,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    metrics:   Dict,
    path:      str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':     epoch + 1,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics':   metrics,
    }, path)
    print(f"  Checkpoint saved (epoch {epoch+1}) → {path}")


def load_checkpoint(
    model:     AddOnRecSys,
    optimizer: Optional[torch.optim.Optimizer],
    path:      str,
    device:    torch.device,
) -> Tuple[int, Dict]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f"  Checkpoint loaded ← {path}")
    return ckpt['epoch'], ckpt['metrics']


# ── main training function ────────────────────────────────────────────────────

def train_ranker(
    train_pairs_path:    str   = "outputs/train_pairs_instacart.parquet",
    val_pairs_path:      str   = "outputs/val_pairs_instacart.parquet",
    items_path:          str   = "outputs/items_instacart.parquet",
    text_embs_path:      str   = "outputs/text_embeddings_instacart.npy",
    pid2idx_path:        str   = "outputs/pid2idx_instacart.json",
    pmi_path:            str   = "outputs/pmi_matrix_instacart.npz",
    retrieval_ckpt_path: str   = "artifacts/checkpoints/retrieval_tower.pt",
    checkpoint_dir:      str   = "artifacts/checkpoints",
    text_emb_dim:        int   = 384,
    d_model:             int   = 64,
    n_food_groups:       int   = 5,
    n_cross_features:    int   = 8,
    max_cart_len:        int   = 50,
    num_restaurants:     int   = 5000,
    backbone_dim:        int   = 256,
    epochs:              int   = 8,
    batch_size:          int   = 32,
    lr:                  float = 1e-4,
    weight_decay:        float = 1e-2,
    warmup_pct:          float = 0.1,
    lambda_coverage:     float = 0.1,
    bpr_margin:          float = 0.0,
    use_hard_negatives:  bool  = True,
    num_workers:         int   = 0,
    log_every:           int   = 50,
    freeze_retrieval:    bool  = True,
    freeze_epochs:       int   = 5,
    resume_from:         Optional[str] = None,
    max_train_samples:   Optional[int] = None,
    max_val_samples:     Optional[int] = None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  CartComplete — Add-On Ranker Training")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"  LR         : {lr}")
    print(f"{'='*60}\n")

    # ── dataloaders ───────────────────────────────────────────────────────────
    collator = get_collator(mode='ranking', dynamic_padding=False)

    train_loader, val_loader = build_ranking_dataloaders(
        train_pairs_path  = train_pairs_path,
        val_pairs_path    = val_pairs_path,
        items_path        = items_path,
        text_embs_path    = text_embs_path,
        pid2idx_path      = pid2idx_path,
        pmi_path          = pmi_path,
        max_cart_len      = max_cart_len,
        n_negatives       = 1,
        batch_size        = batch_size,
        num_workers       = num_workers,
        max_train_samples = max_train_samples,
        max_val_samples   = max_val_samples,
    )

    train_loader.collate_fn = collator
    val_loader.collate_fn   = collator

    # ── model ─────────────────────────────────────────────────────────────────
    with open(pid2idx_path) as f:
        pid2idx = json.load(f)
    n_items = len(pid2idx)

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
        lambda_coverage  = lambda_coverage,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {total_params:,}")

    # ── load pretrained retrieval tower ───────────────────────────────────────
    if os.path.exists(retrieval_ckpt_path):
        print(f"\nLoading pretrained retrieval tower...")
        ckpt       = torch.load(retrieval_ckpt_path, map_location=device)
        model_dict = model.state_dict()
        pretrained = {
            k: v for k, v in ckpt['model'].items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(pretrained):,} / {len(model_dict):,} tensors")
    else:
        print(f"\nNo retrieval checkpoint found — training from scratch.")

    # ── freeze retrieval tower stage 1 ────────────────────────────────────────
    if freeze_retrieval:
        print(f"\nFreezing retrieval tower for first {freeze_epochs} epochs...")
        for name, param in model.named_parameters():
            if 'cart_encoder' in name or 'item_encoder' in name:
                param.requires_grad = False

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    total_steps = epochs * len(train_loader)
    scheduler   = OneCycleLR(
        optimizer,
        max_lr          = lr,
        total_steps     = total_steps,
        pct_start       = warmup_pct,
        anneal_strategy = 'cos',
    )

    # ── loss ──────────────────────────────────────────────────────────────────
    criterion = CartCompleteLoss(
        lambda_coverage    = lambda_coverage,
        bpr_margin         = bpr_margin,
        use_hard_negatives = use_hard_negatives,
        n_food_groups      = n_food_groups,
    )

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_ndcg10 = 0.0
    history     = []

    if resume_from and os.path.exists(resume_from):
        start_epoch, prev = load_checkpoint(model, optimizer, resume_from, device)
        best_ndcg10 = prev.get('NDCG@10', 0.0)
        start_epoch += 1

    # ── training loop ─────────────────────────────────────────────────────────
    train_start = time.time()

    for epoch in range(start_epoch, epochs):
        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch+1}/{epochs}")
        print(f"{'─'*60}")

        # Unfreeze for end-to-end fine-tuning after freeze_epochs
        if freeze_retrieval and epoch == freeze_epochs:
            print(f"\n  Unfreezing all parameters...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = AdamW(
                model.parameters(),
                lr=lr * 0.1, weight_decay=weight_decay,
            )
            remaining = (epochs - epoch) * len(train_loader)
            scheduler = OneCycleLR(
                optimizer,
                max_lr          = lr * 0.1,
                total_steps     = remaining,
                pct_start       = 0.05,
                anneal_strategy = 'cos',
            )

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, epoch + 1, log_every,
        )

        print(f"\n  Validating...")
        val_metrics = validate(
            model, val_loader, criterion, device, k_values=[5, 10]
        )

        epoch_metrics = {'epoch': epoch + 1, **train_metrics, **val_metrics}
        history.append(epoch_metrics)

        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    Train Loss : {train_metrics['train_total']:.4f}")
        print(f"    Train BPR  : {train_metrics['train_bpr']:.4f}")
        print(f"    Train Cov  : {train_metrics['train_coverage']:.4f}")
        print(f"    Val Loss   : {val_metrics['val_loss']:.4f}")
        print(f"    HR@5       : {val_metrics.get('HR@5',    0):.4f}")
        print(f"    HR@10      : {val_metrics.get('HR@10',   0):.4f}")
        print(f"    NDCG@5     : {val_metrics.get('NDCG@5',  0):.4f}")
        print(f"    NDCG@10    : {val_metrics.get('NDCG@10', 0):.4f}")
        print(f"    MRR        : {val_metrics.get('MRR',     0):.4f}")

        # Save best checkpoint
        ndcg10 = val_metrics.get('NDCG@10', 0.0)
        if ndcg10 > best_ndcg10:
            best_ndcg10 = ndcg10
            for name in ['full_model.pt', 'best_checkpoint.pt']:
                save_checkpoint(
                    model, optimizer, epoch, epoch_metrics,
                    os.path.join(checkpoint_dir, name),
                )
            print(f"  ✓ New best NDCG@10: {best_ndcg10:.4f}")

        # Save latest
        save_checkpoint(
            model, optimizer, epoch, epoch_metrics,
            os.path.join(checkpoint_dir, 'ranker_latest.pt'),
        )

    # ── save history ──────────────────────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_path = os.path.join(checkpoint_dir, 'ranker_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - train_start
    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"  Total time   : {format_time(total_time)}")
    print(f"  Best NDCG@10 : {best_ndcg10:.4f}")
    print(f"  Best model   → {checkpoint_dir}/full_model.pt")
    print(f"{'='*60}\n")

    return history


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_pairs',      default="outputs/train_pairs_instacart.parquet")
    parser.add_argument('--val_pairs',        default="outputs/val_pairs_instacart.parquet")
    parser.add_argument('--items',            default="outputs/items_instacart.parquet")
    parser.add_argument('--text_embs',        default="outputs/text_embeddings_instacart.npy")
    parser.add_argument('--pid2idx',          default="outputs/pid2idx_instacart.json")
    parser.add_argument('--pmi_path',         default="outputs/pmi_matrix_instacart.npz")
    parser.add_argument('--retrieval_ckpt',   default="artifacts/checkpoints/retrieval_tower.pt")
    parser.add_argument('--checkpoint_dir',   default="artifacts/checkpoints")
    parser.add_argument('--text_emb_dim',     type=int,   default=384)
    parser.add_argument('--d_model',          type=int,   default=64)
    parser.add_argument('--n_food_groups',    type=int,   default=5)
    parser.add_argument('--n_cross_features', type=int,   default=8)
    parser.add_argument('--num_restaurants',  type=int,   default=5000)
    parser.add_argument('--backbone_dim',     type=int,   default=256)
    parser.add_argument('--max_cart_len',     type=int,   default=50)
    parser.add_argument('--epochs',           type=int,   default=8)
    parser.add_argument('--batch_size',       type=int,   default=32)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--weight_decay',     type=float, default=1e-2)
    parser.add_argument('--warmup_pct',       type=float, default=0.1)
    parser.add_argument('--lambda_coverage',  type=float, default=0.1)
    parser.add_argument('--bpr_margin',       type=float, default=0.0)
    parser.add_argument('--freeze_epochs',    type=int,   default=5)
    parser.add_argument('--no_freeze',        action='store_true')
    parser.add_argument('--no_hard_neg',      action='store_true')
    parser.add_argument('--num_workers',      type=int,   default=0)
    parser.add_argument('--log_every',        type=int,   default=50)
    parser.add_argument('--resume_from',      type=str,   default=None)
    parser.add_argument('--max_train_samples',type=int,   default=None)
    parser.add_argument('--max_val_samples',  type=int,   default=None)

    args = parser.parse_args()

    train_ranker(
        train_pairs_path    = args.train_pairs,
        val_pairs_path      = args.val_pairs,
        items_path          = args.items,
        text_embs_path      = args.text_embs,
        pid2idx_path        = args.pid2idx,
        pmi_path            = args.pmi_path,
        retrieval_ckpt_path = args.retrieval_ckpt,
        checkpoint_dir      = args.checkpoint_dir,
        text_emb_dim        = args.text_emb_dim,
        d_model             = args.d_model,
        n_food_groups       = args.n_food_groups,
        n_cross_features    = args.n_cross_features,
        num_restaurants     = args.num_restaurants,
        backbone_dim        = args.backbone_dim,
        max_cart_len        = args.max_cart_len,
        epochs              = args.epochs,
        batch_size          = args.batch_size,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        warmup_pct          = args.warmup_pct,
        lambda_coverage     = args.lambda_coverage,
        bpr_margin          = args.bpr_margin,
        freeze_retrieval    = not args.no_freeze,
        freeze_epochs       = args.freeze_epochs,
        use_hard_negatives  = not args.no_hard_neg,
        num_workers         = args.num_workers,
        log_every           = args.log_every,
        resume_from         = args.resume_from,
        max_train_samples   = args.max_train_samples,
        max_val_samples     = args.max_val_samples,
    )