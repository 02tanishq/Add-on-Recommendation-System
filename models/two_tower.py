

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ── cart tower ────────────────────────────────────────────────────────────────

class CartTower(nn.Module):
    """
    Encodes the current cart into a single query vector
    for FAISS nearest-neighbor retrieval.

    Uses a lightweight Set Transformer (2 SAB layers)
    NOT the full SetTransformerEncoder from set_transformer.py —
    that is used in the expensive ranking stage.
    This tower must be fast: it runs at retrieval time.

    Architecture:
        cart item embeddings (B, N, d_in)
            → input projection (d_in → d_tower)
            → 2 × SAB (self-attention blocks)
            → mean + max pooling
            → output projection → L2 norm
            → cart_vec (B, d_out)
    """

    def __init__(
        self,
        d_in:     int,           # input item embedding dim
        d_tower:  int  = 128,    # internal tower dim
        d_out:    int  = 64,     # output retrieval vector dim
        n_heads:  int  = 4,
        n_layers: int  = 2,
        dropout:  float = 0.1,
    ):
        super().__init__()

        self.d_out = d_out

        # Project item embeddings to tower dim
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_tower),
            nn.LayerNorm(d_tower),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Lightweight self-attention blocks
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model        = d_tower,
                nhead          = n_heads,
                dim_feedforward= d_tower * 4,
                dropout        = dropout,
                activation     = 'gelu',
                batch_first    = True,
                norm_first     = True,    # pre-norm — more stable
            )
            for _ in range(n_layers)
        ])

        # Pooling projection: mean + max concat → d_out
        self.pool_proj = nn.Sequential(
            nn.Linear(d_tower * 2, d_tower),
            nn.LayerNorm(d_tower),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_tower, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(
        self,
        cart_embs: torch.Tensor,
        cart_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        cart_embs : (B, N, d_in)
        cart_mask : (B, N)  1=real  0=padding

        Returns   : (B, d_out)  L2-normalized cart vector
        """
        x = self.input_proj(cart_embs)   # (B, N, d_tower)

        # PyTorch TransformerEncoderLayer expects True=ignore (padding)
        src_key_padding_mask = None
        if cart_mask is not None:
            src_key_padding_mask = (cart_mask == 0)   # (B, N) True=pad

        for layer in self.attn_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        # Mean + max pooling
        if cart_mask is not None:
            mask_f    = cart_mask.float().unsqueeze(-1)        # (B, N, 1)
            lengths   = mask_f.sum(1).clamp(min=1)            # (B, 1)
            mean_pool = (x * mask_f).sum(1) / lengths         # (B, d_tower)
            x_masked  = x.masked_fill(mask_f == 0, -1e9)
            max_pool  = x_masked.max(1).values                # (B, d_tower)
        else:
            mean_pool = x.mean(1)
            max_pool  = x.max(1).values

        pooled   = torch.cat([mean_pool, max_pool], dim=-1)   # (B, 2*d_tower)
        cart_vec = self.pool_proj(pooled)                     # (B, d_out)

        # L2 normalize — dot product == cosine similarity in FAISS
        return F.normalize(cart_vec, p=2, dim=-1)


# ── candidate tower ───────────────────────────────────────────────────────────

class CandidateTower(nn.Module):
    """
    Encodes a single candidate item into a key vector
    that lives in the same space as the cart vector.

    This is simpler than the cart tower — no attention needed,
    just a deep MLP over item features.

    At offline time: encode ALL items → store in FAISS index.
    At online time:  just do ANN lookup, no re-encoding needed.

    Architecture:
        candidate features (B, d_in)
            → 3-layer MLP with residuals
            → L2 norm
            → candidate_vec (B, d_out)
    """

    def __init__(
        self,
        d_in:    int,
        d_tower: int   = 128,
        d_out:   int   = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_out = d_out

        # Layer 1
        self.fc1   = nn.Linear(d_in, d_tower)
        self.norm1 = nn.LayerNorm(d_tower)

        # Layer 2 + residual
        self.fc2   = nn.Linear(d_tower, d_tower)
        self.norm2 = nn.LayerNorm(d_tower)

        # Layer 3 → output
        self.fc3   = nn.Linear(d_tower, d_out)
        self.norm3 = nn.LayerNorm(d_out)

        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, candidate_emb: torch.Tensor) -> torch.Tensor:
        """
        candidate_emb : (B, d_in)
        Returns       : (B, d_out)  L2-normalized candidate vector
        """
        # Layer 1
        x  = self.dropout(self.act(self.norm1(self.fc1(candidate_emb))))

        # Layer 2 + residual
        h  = self.dropout(self.act(self.norm2(self.fc2(x))))
        x  = x + h    # residual — dims match, no projection needed

        # Layer 3 → output
        x  = self.norm3(self.fc3(x))

        return F.normalize(x, p=2, dim=-1)   # (B, d_out)


# ── in-batch negative miner ───────────────────────────────────────────────────

class InBatchNegativeMiner(nn.Module):
    """
    Mines hard negatives from the current batch.

    Instead of random negatives from the catalog,
    uses other items IN THE SAME BATCH as negatives.
    These are harder because they were all retrieved
    as plausible candidates for some cart in the batch.

    This is the standard approach in two-tower training
    (Google, 2019 — Sampling-Bias-Corrected Neural Modeling).

    For a batch of B (cart, positive) pairs:
        Similarity matrix : (B, B)
        Diagonal          : sim(cart_i, positive_i) — positives
        Off-diagonal      : sim(cart_i, positive_j) — in-batch negatives

    Returns InfoNCE loss over this similarity matrix.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        cart_vecs:      torch.Tensor,   # (B, d_out)  L2-normalized
        candidate_vecs: torch.Tensor,   # (B, d_out)  L2-normalized
    ) -> torch.Tensor:
        """
        Returns InfoNCE contrastive loss scalar.
        """
        # Similarity matrix: (B, B)
        # sim[i, j] = dot(cart_i, candidate_j)
        sim = torch.matmul(cart_vecs, candidate_vecs.T) / self.temperature

        # Labels: diagonal is positive (cart_i, candidate_i)
        labels = torch.arange(sim.size(0), device=sim.device)

        # Symmetric loss: cart→candidate and candidate→cart
        loss_cart2cand = F.cross_entropy(sim,   labels)
        loss_cand2cart = F.cross_entropy(sim.T, labels)

        return (loss_cart2cand + loss_cand2cart) / 2


# ── main two tower model ──────────────────────────────────────────────────────

class TwoTowerModel(nn.Module):
    """
    Full Two-Tower Retrieval Model.

    Combines CartTower + CandidateTower with
    in-batch negative contrastive training.

    Training
    ────────
    Input: (cart_embs, positive_item_emb) pairs
    Loss : InfoNCE over in-batch negatives
    → minimises sim(cart, negative)
    → maximises sim(cart, positive)

    After training:
    ────────────────
    1. Run all items through CandidateTower
    2. Store vectors in FAISS index
    3. At serving: encode cart → query FAISS → top-100 candidates

    The two towers share the same output space (d_out)
    enforced by L2 normalisation in both towers.
    """

    def __init__(
        self,
        d_in:        int,           # input item feature dim
        d_tower:     int  = 128,    # internal tower width
        d_out:       int  = 64,     # shared retrieval space dim
        n_heads:     int  = 4,
        n_layers:    int  = 2,
        dropout:     float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.cart_tower = CartTower(
            d_in    = d_in,
            d_tower = d_tower,
            d_out   = d_out,
            n_heads = n_heads,
            n_layers= n_layers,
            dropout = dropout,
        )

        self.candidate_tower = CandidateTower(
            d_in    = d_in,
            d_tower = d_tower,
            d_out   = d_out,
            dropout = dropout,
        )

        self.neg_miner = InBatchNegativeMiner(temperature=temperature)

        self.d_out = d_out

    def encode_cart(
        self,
        cart_embs: torch.Tensor,
        cart_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode cart → query vector.
        Called at online serving time.

        cart_embs : (B, N, d_in)
        cart_mask : (B, N)
        Returns   : (B, d_out)
        """
        return self.cart_tower(cart_embs, cart_mask)

    def encode_candidate(self, candidate_emb: torch.Tensor) -> torch.Tensor:
        """
        Encode one item → key vector.
        Called offline to build FAISS index.

        candidate_emb : (B, d_in)
        Returns       : (B, d_out)
        """
        return self.candidate_tower(candidate_emb)

    def forward(
        self,
        cart_embs:     torch.Tensor,
        candidate_emb: torch.Tensor,
        cart_mask:     Optional[torch.Tensor] = None,
        return_loss:   bool = True,
    ) -> dict:
        """
        Parameters
        ──────────
        cart_embs     : (B, N, d_in)  — cart item feature vectors
        candidate_emb : (B, d_in)     — positive candidate per cart
        cart_mask     : (B, N)        — 1=real  0=padding
        return_loss   : bool          — compute InfoNCE loss

        Returns dict with:
            cart_vec      : (B, d_out)
            candidate_vec : (B, d_out)
            similarity    : (B,)        dot product per pair
            loss          : scalar      InfoNCE (if return_loss=True)
        """
        cart_vec      = self.cart_tower(cart_embs, cart_mask)
        candidate_vec = self.candidate_tower(candidate_emb)

        # Per-pair similarity (diagonal of full sim matrix)
        similarity = (cart_vec * candidate_vec).sum(dim=-1)   # (B,)

        out = {
            'cart_vec':      cart_vec,
            'candidate_vec': candidate_vec,
            'similarity':    similarity,
        }

        if return_loss:
            out['loss'] = self.neg_miner(cart_vec, candidate_vec)

        return out

    @torch.no_grad()
    def encode_all_items(
        self,
        item_embs:  torch.Tensor,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """
        Encode entire item catalog in batches.
        Called offline by training/build_faiss_index.py

        item_embs  : (N_items, d_in)
        batch_size : items per forward pass

        Returns    : (N_items, d_out)  — store in FAISS
        """
        self.eval()
        all_vecs = []
        for i in range(0, len(item_embs), batch_size):
            batch = item_embs[i: i + batch_size]
            vec   = self.candidate_tower(batch)
            all_vecs.append(vec.cpu())
        return torch.cat(all_vecs, dim=0)   # (N_items, d_out)


# ── similarity utilities ──────────────────────────────────────────────────────

def cosine_similarity_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute full cosine similarity matrix between two sets of vectors.

    a : (M, d)
    b : (N, d)
    Returns : (M, N)
    """
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return torch.matmul(a, b.T)


def top_k_retrieval(
    cart_vec:    torch.Tensor,
    item_vecs:   torch.Tensor,
    k:           int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact top-K retrieval without FAISS.
    Use for evaluation / small catalogs.
    For production use FAISS via build_faiss_index.py.

    cart_vec  : (B, d_out)
    item_vecs : (N_items, d_out)

    Returns:
        scores  : (B, K)  similarity scores
        indices : (B, K)  item indices
    """
    sim     = cosine_similarity_matrix(cart_vec, item_vecs)   # (B, N)
    scores, indices = sim.topk(k, dim=-1, largest=True)
    return scores, indices





# ── sanity check ─────────────────────────────
if __name__ == "__main__":
    B, N, d_in = 8, 6, 256

    model = TwoTowerModel(
        d_in        = d_in,
        d_tower     = 128,
        d_out       = 64,
        n_heads     = 4,
        n_layers    = 2,
        dropout     = 0.1,
        temperature = 0.07,
    )

    cart_embs  = torch.randn(B, N, d_in)
    cand_emb   = torch.randn(B, d_in)
    cart_mask  = torch.ones(B, N, dtype=torch.long)
    cart_mask[:, -2:] = 0   # last 2 positions padding

    out = model(cart_embs, cand_emb, cart_mask, return_loss=True)
    print(f"cart_vec      : {out['cart_vec'].shape}")       # (8, 64)
    print(f"candidate_vec : {out['candidate_vec'].shape}")  # (8, 64)
    print(f"similarity    : {out['similarity'].shape}")     # (8,)
    print(f"InfoNCE loss  : {out['loss'].item():.4f}")

    # Offline index building
    all_items = torch.randn(1000, d_in)
    all_vecs  = model.encode_all_items(all_items, batch_size=256)
    print(f"All item vecs : {all_vecs.shape}")              # (1000, 64)

    # Exact top-K retrieval (no FAISS)
    cart_q           = out['cart_vec']
    scores, indices  = top_k_retrieval(cart_q, all_vecs, k=10)
    print(f"Top-10 scores   : {scores.shape}")              # (8, 10)
    print(f"Top-10 indices  : {indices.shape}")             # (8, 10)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total:,}")