import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, Optional, Tuple
from training.retrieval_dataset import build_retrieval_dataloaders
from training.collate import get_collator
from training.losses import InfoNCELoss
from models.two_tower import TwoTowerModel


# ── utility helpers ───────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks a running average of a scalar metric."""

    def __init__(self, name: str):
        self.name  = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

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


# ── recall@K evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def compute_recall_at_k(
    model:       TwoTowerModel,
    val_loader:  DataLoader,
    device:      torch.device,
    k_values:    list = [1, 5, 10, 20],
) -> Dict[str, float]:
    """
    Compute Recall@K for the retrieval model on the validation set.

    Strategy:
        For each batch, compute the similarity between every
        cart vector and every item vector in the batch.
        A hit@K = 1 if the positive item is in the top-K
        most similar items for that cart.

    Note: This is an approximation of full-catalog recall
    since we only compare within the batch. Full evaluation
    happens in training/evaluate.py after FAISS index is built.
    """
    model.eval()

    recall_meters = {k: AverageMeter(f"Recall@{k}") for k in k_values}
    max_k = max(k_values)

    for batch in val_loader:
        # Move batch to device
        batch = {key: val.to(device) for key, val in batch.items()}

        # Encode carts
        cart_embs = model.encode_cart(
            cart_embs=batch['cart_text_embs'],
            cart_mask=batch['cart_mask'],
        )

        # Encode positive items
        item_embs = model.encode_candidate(
            batch['pos_text_emb']
        )

        # Normalise for cosine similarity
        cart_embs = nn.functional.normalize(cart_embs, dim=-1)
        item_embs = nn.functional.normalize(item_embs, dim=-1)

        # Similarity matrix (B, B) — diagonal = positive pairs
        sim = torch.matmul(cart_embs, item_embs.T)   # (B, B)
        B   = sim.size(0)

        # Get top-K indices for each cart
        _, top_k_indices = sim.topk(
            min(max_k, B), dim=1, largest=True
        )   # (B, K)

        # Ground truth: positive is always at diagonal index
        labels = torch.arange(B, device=device)   # (B,)

        for k in k_values:
            k_capped = min(k, B)
            top_k    = top_k_indices[:, :k_capped]   # (B, k)
            hits     = (
                top_k == labels.unsqueeze(1)
            ).any(dim=1).float()   # (B,)
            recall_meters[k].update(hits.mean().item(), n=B)

    return {f"Recall@{k}": recall_meters[k].avg for k in k_values}


# ── one training epoch ────────────────────────────────────────────────────────

def train_one_epoch(
    model:       TwoTowerModel,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    scheduler:   torch.optim.lr_scheduler._LRScheduler,
    criterion:   InfoNCELoss,
    device:      torch.device,
    epoch:       int,
    log_every:   int = 100,
) -> Dict[str, float]:
    """
    Runs one full training epoch for the two-tower retrieval model.

    Returns dict of averaged metrics for this epoch.
    """
    model.train()

    loss_meter = AverageMeter("Loss")
    start_time = time.time()

    for step, batch in enumerate(loader):
        batch = {key: val.to(device) for key, val in batch.items()}

        # ── encode cart tower ─────────────────────────────────────────────────
        cart_embs = model.encode_cart(
            cart_embs=batch['cart_text_embs'],
            cart_mask=batch['cart_mask'],
        ) # (B, d)

        # ── encode item tower ─────────────────────────────────────────────────
        item_embs = model.encode_candidate(
            batch['pos_text_emb']
        ) # (B, d)

        # ── L2 normalise both towers ──────────────────────────────────────────
        cart_embs = nn.functional.normalize(cart_embs, dim=-1)
        item_embs = nn.functional.normalize(item_embs, dim=-1)

        # ── InfoNCE loss ──────────────────────────────────────────────────────
        loss = criterion(cart_embs, item_embs)

        # ── backward pass ─────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — stabilises early training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item(), n=cart_embs.size(0))

        if (step + 1) % log_every == 0:
            elapsed = time.time() - start_time
            lr      = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:02d} | "
                f"Step {step+1:05d}/{len(loader):05d} | "
                f"Loss {loss_meter.avg:.4f} | "
                f"LR {lr:.2e} | "
                f"Time {format_time(elapsed)}"
            )

    return {
        'train_loss': loss_meter.avg,
        'lr':         scheduler.get_last_lr()[0],
    }

# ── validation epoch ──────────────────────────────────────────────────────────

# @torch.no_grad()
# def validate(
#     model:      TwoTowerModel,
#     loader:     DataLoader,
#     criterion:  InfoNCELoss,
#     device:     torch.device,
#     k_values:   list = [1, 5, 10, 20],
# ) -> Dict[str, float]:
#     """
#     Full validation — InfoNCE loss + Recall@K metrics.
#     """
#     model.eval()

#     loss_meter = AverageMeter("Val Loss")

#     for batch in loader:
#         batch = {key: val.to(device) for key, val in batch.items()}

#         cart_embs = model.encode_cart(
#             cart_embs=batch['cart_text_embs'],
#             cart_mask=batch['cart_mask'],
#         )

#         # Encode positive items
#         item_embs = model.encode_candidate(
#             batch['pos_text_emb']
#         )

#         cart_embs = nn.functional.normalize(cart_embs, dim=-1)
#         item_embs = nn.functional.normalize(item_embs, dim=-1)

#         loss = criterion(cart_embs, item_embs)
#         loss_meter.update(loss.item(), n=cart_embs.size(0))

#     # Recall@K metrics
#     recall_metrics = {
#         "Recall@1": 0,
#         "Recall@5": 0,
#         "Recall@10": 0,
#         "Recall@20": 0,
#     }

#     return {
#         'val_loss': loss_meter.avg,
#         **recall_metrics,
#     }


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model:      TwoTowerModel,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    metrics:    Dict,
    path:       str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':      epoch,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'metrics':    metrics,
    }, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(
    model:     TwoTowerModel,
    optimizer: Optional[torch.optim.Optimizer],
    path:      str,
    device:    torch.device,
) -> Tuple[int, Dict]:
    ckpt      = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f"  Checkpoint loaded ← {path}")
    return ckpt['epoch'], ckpt['metrics']


# ── main training function ────────────────────────────────────────────────────

def train_retrieval(
    # ── paths ──────────────────────────────────────────────────────────────
    train_pairs_path:  str = "outputs/train_pairs_instacart.parquet",
    val_pairs_path:    str = "outputs/val_pairs_instacart.parquet",
    items_path:        str = "outputs/items_instacart.parquet",
    text_embs_path:    str = "outputs/text_embeddings_instacart.npy",
    pid2idx_path:      str = "outputs/pid2idx_instacart.json",
    checkpoint_dir:    str = "artifacts/checkpoints",
    # ── model config ───────────────────────────────────────────────────────
    text_emb_dim:      int   = 384,
    d_model:           int   = 128,
    n_food_groups:     int   = 5,
    max_cart_len:      int   = 50,
    # ── training config ────────────────────────────────────────────────────
    epochs:            int   = 1,
    batch_size:        int   = 16,
    lr:                float = 3e-4,
    weight_decay:      float = 1e-2,
    temperature:       float = 0.07,
    warmup_pct:        float = 0.1,
    num_workers:       int   = 0,
    log_every:         int   = 100,
    resume_from:       Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples:   Optional[int] = None,
):
    """
    Full training loop for the CartComplete Two-Tower retrieval model.

    Pipeline:
        1. Build dataloaders
        2. Initialise TwoTowerModel
        3. Train with InfoNCE loss for `epochs` epochs
        4. Validate with InfoNCE loss + Recall@K
        5. Save best checkpoint based on Recall@10
        6. Save training history to JSON

    Output artifacts:
        artifacts/checkpoints/retrieval_tower.pt   — best model
        artifacts/checkpoints/retrieval_history.json
    """

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  CartComplete — Two-Tower Retrieval Training")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {epochs}")
    print(f"  Batch   : {batch_size}")
    print(f"  LR      : {lr}")
    print(f"{'='*60}\n")

    # ── dataloaders ───────────────────────────────────────────────────────────
    print("Building dataloaders...")
    collator = get_collator(mode='retrieval', dynamic_padding=False)

    train_loader, val_loader = build_retrieval_dataloaders(
        train_pairs_path  = train_pairs_path,
        val_pairs_path    = val_pairs_path,
        items_path        = items_path,
        text_embs_path    = text_embs_path,
        pid2idx_path      = pid2idx_path,
        max_cart_len      = max_cart_len,
        batch_size        = batch_size,
        num_workers       = num_workers,
        max_train_samples = max_train_samples,
        max_val_samples   = max_val_samples,
    )

    # Override collate_fn with our collator
    train_loader.collate_fn = collator
    val_loader.collate_fn   = collator

    # ── model ─────────────────────────────────────────────────────────────────
    print("\nInitialising TwoTowerModel...")

    # Get n_items from pid2idx
    with open(pid2idx_path) as f:
        pid2idx = json.load(f)
    n_items = len(pid2idx)

    model = TwoTowerModel(
        d_in=384,
        d_tower=128,
        d_out=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        temperature=0.07,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total_params:,}")

    # ── optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = weight_decay,
    )

    total_steps = epochs * len(train_loader)

    scheduler = OneCycleLR(
        optimizer,
        max_lr          = lr,
        total_steps     = total_steps,
        pct_start       = warmup_pct,
        anneal_strategy = 'cos',
    )

    # ── loss ──────────────────────────────────────────────────────────────────
    criterion = InfoNCELoss(
        temperature = temperature,
        symmetrise  = True,
    )

    # ── optionally resume ─────────────────────────────────────────────────────
    start_epoch   = 0
    best_recall10 = 0.0
    history       = []

    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from {resume_from}...")
        start_epoch, prev_metrics = load_checkpoint(
            model, optimizer, resume_from, device
        )
        best_recall10 = prev_metrics.get('Recall@10', 0.0)
        start_epoch  += 1

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training from epoch {start_epoch + 1}...\n")
    train_start = time.time()

    for epoch in range(start_epoch, epochs):
        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch+1}/{epochs}")
        print(f"{'─'*60}")

        # Train
        train_metrics = train_one_epoch(
            model     = model,
            loader    = train_loader,
            optimizer = optimizer,
            scheduler = scheduler,
            criterion = criterion,
            device    = device,
            epoch     = epoch + 1,
            log_every = log_every,
        )

        history.append({
            "epoch": epoch + 1,
            **train_metrics,
        })

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=train_metrics,
            path=os.path.join(
            checkpoint_dir,
                "retrieval_tower.pt"
            ),
        )

        # # Validate
        # print(f"\n  Validating...")
        # val_metrics = validate(
        #     model     = model,
        #     loader    = val_loader,
        #     criterion = criterion,
        #     device    = device,
        #     k_values  = [1, 5, 10, 20],
        # )

        # Merge metrics
        # epoch_metrics = {
        #     'epoch': epoch + 1,
        #     **train_metrics,
        #     **val_metrics,
        # }
        # history.append(epoch_metrics)

        # # Print epoch summary
        # print(f"\n  Epoch {epoch+1} Summary:")
        # print(f"    Train Loss  : {train_metrics['train_loss']:.4f}")
        # print(f"    Val Loss    : {val_metrics['val_loss']:.4f}")
        # print(f"    Recall@1    : {val_metrics.get('Recall@1',  0):.4f}")
        # print(f"    Recall@5    : {val_metrics.get('Recall@5',  0):.4f}")
        # print(f"    Recall@10   : {val_metrics.get('Recall@10', 0):.4f}")
        # print(f"    Recall@20   : {val_metrics.get('Recall@20', 0):.4f}")

        # # Save best checkpoint
        # recall10 = val_metrics.get('Recall@10', 0.0)
        # if recall10 > best_recall10:
        #     best_recall10 = recall10
        #     save_checkpoint(
        #         model     = model,
        #         optimizer = optimizer,
        #         epoch     = epoch,
        #         metrics   = epoch_metrics,
        #         path      = os.path.join(
        #             checkpoint_dir, 'retrieval_tower.pt'
        #         ),
        #     )
        #     print(f"  ✓ New best Recall@10: {best_recall10:.4f}")

        # # Save latest checkpoint (for resuming)
        # save_checkpoint(
        #     model     = model,
        #     optimizer = optimizer,
        #     epoch     = epoch,
        #     metrics   = epoch_metrics,
        #     path      = os.path.join(
        #         checkpoint_dir, 'retrieval_latest.pt'
        #     ),
        # )

    # ── save history ──────────────────────────────────────────────────────────
    history_path = os.path.join(checkpoint_dir, 'retrieval_history.json')
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history saved → {history_path}")

    total_time = time.time() - train_start
    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"  Total time     : {format_time(total_time)}")
    print(f"  Best Recall@10 : {best_recall10:.4f}")
    print(f"  Best model     → {checkpoint_dir}/retrieval_tower.pt")
    print(f"{'='*60}\n")

    return history


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train CartComplete Two-Tower Retrieval Model"
    )

    # Paths
    parser.add_argument('--train_pairs',  default="outputs/train_pairs_instacart.parquet")
    parser.add_argument('--val_pairs',    default="outputs/val_pairs_instacart.parquet")
    parser.add_argument('--items',        default="outputs/items_instacart.parquet")
    parser.add_argument('--text_embs',    default="outputs/text_embeddings_instacart.npy")
    parser.add_argument('--pid2idx',      default="outputs/pid2idx_instacart.json")
    parser.add_argument('--checkpoint_dir', default="artifacts/checkpoints")

    # Model
    parser.add_argument('--text_emb_dim', type=int,   default=384)
    parser.add_argument('--d_model',      type=int,   default=128)
    parser.add_argument('--max_cart_len', type=int,   default=50)

    # Training
    parser.add_argument('--epochs',       type=int,   default=1)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--temperature',  type=float, default=0.07)
    parser.add_argument('--warmup_pct',   type=float, default=0.1)
    parser.add_argument('--num_workers',  type=int,   default=0)
    parser.add_argument('--log_every',    type=int,   default=100)
    parser.add_argument('--resume_from',  type=str,   default=None)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples',   type=int, default=None)

    args = parser.parse_args()

    train_retrieval(
        train_pairs_path  = args.train_pairs,
        val_pairs_path    = args.val_pairs,
        items_path        = args.items,
        text_embs_path    = args.text_embs,
        pid2idx_path      = args.pid2idx,
        checkpoint_dir    = args.checkpoint_dir,
        text_emb_dim      = args.text_emb_dim,
        d_model           = args.d_model,
        max_cart_len      = args.max_cart_len,
        epochs            = args.epochs,
        batch_size        = args.batch_size,
        lr                = args.lr,
        weight_decay      = args.weight_decay,
        temperature       = args.temperature,
        warmup_pct        = args.warmup_pct,
        num_workers       = args.num_workers,
        log_every         = args.log_every,
        resume_from       = args.resume_from,
        max_train_samples = args.max_train_samples,
        max_val_samples   = args.max_val_samples,
    )