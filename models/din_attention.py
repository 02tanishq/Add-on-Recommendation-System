
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DINAttention(nn.Module):
    """
    Target-aware attention where the candidate item
    is the query and cart items are keys and values.

    For each cart item i, computes an attention weight:
        w_i = MLP([e_cart_i ; e_cand ; e_cart_i - e_cand ; e_cart_i * e_cand])

    The four-way interaction vector captures:
        e_cart_i              — what this cart item is
        e_cand                — what the candidate is
        e_cart_i - e_cand     — how different they are
        e_cart_i * e_cand     — element-wise similarity

    This is the exact formulation from the DIN paper.
    Weights are then used to compute a weighted sum
    of cart embeddings → cross_repr.

    cross_repr tells the ranker:
        "given this candidate, which cart items
         matter most and how strongly do they
         signal that the candidate is needed?"
    """

    def __init__(
        self,
        d_model:    int,
        hidden_dim: int   = 64,
        dropout:    float = 0.1,
    ):
        super().__init__()

        # Attention MLP — input is 4 × d_model
        # [e_i ; e_c ; e_i - e_c ; e_i * e_c]
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Optional: project cross_repr to d_model after weighted sum
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Explicit cross feature — element-wise product of
        # candidate and attention-weighted cart repr
        # Gives the ranker a direct complementarity signal
        self.cross_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.d_model = d_model


    def forward(
        self,
        candidate_emb: torch.Tensor,
        cart_embs:     torch.Tensor,
        cart_mask:     Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ──────────
        candidate_emb : (B, d_model)
        cart_embs     : (B, N, d_model)
        cart_mask     : (B, N)  1=real  0=padding

        Returns
        ───────
        cross_repr   : (B, d_model)
            Attention-weighted cart representation
            conditioned on the candidate.

        attn_weights : (B, N)
            Normalized attention weights per cart item.
            Used directly for explainability heatmap.
            attn_weights[b, i] = how much cart item i
            drove the recommendation of candidate b.
        """

        B, N, d = cart_embs.shape

        # Expand candidate to match cart sequence length
        # (B, d_model) → (B, N, d_model)
        cand_exp = candidate_emb.unsqueeze(1).expand(B, N, d)

        # Build four-way interaction vector per cart item
        diff    = cart_embs - cand_exp                # (B, N, d)
        product = cart_embs * cand_exp                # (B, N, d)

        # Concatenate: [e_cart ; e_cand ; diff ; product]
        interaction = torch.cat(
            [cart_embs, cand_exp, diff, product], dim=-1
        )   # (B, N, 4*d)

        # Attention scores — one scalar per cart item
        attn_scores = self.attn_mlp(interaction).squeeze(-1)   # (B, N)

        # Mask out padding positions before softmax
        if cart_mask is not None:
            # cart_mask: 1=real 0=pad → mask pad positions with -inf
            pad_mask    = (cart_mask == 0)              # (B, N) True=pad
            attn_scores = attn_scores.masked_fill(pad_mask, -1e9)

        # Normalize to probability distribution
        attn_weights = F.softmax(attn_scores, dim=-1)   # (B, N)

        # Weighted sum of cart embeddings
        # (B, N, 1) × (B, N, d) → sum over N → (B, d)
        weighted_sum = (attn_weights.unsqueeze(-1) * cart_embs).sum(dim=1)
        cross_repr   = self.output_proj(weighted_sum)   # (B, d_model)

        # Explicit cross feature: candidate ⊗ cross_repr
        # This is the MTGR cross feature — element-wise product
        # gives the ranker a direct complementarity signal
        explicit_cross = torch.cat(
            [cross_repr, candidate_emb * cross_repr], dim=-1
        )   # (B, 2*d_model)
        cross_repr = self.cross_proj(explicit_cross)    # (B, d_model)

        return cross_repr, attn_weights


class MultiHeadDINAttention(nn.Module):
    """
    Multi-head version of DIN attention.

    Runs H independent DIN attention heads in parallel,
    each learning a different complementarity pattern:

        Head 1 → cuisine compatibility signal
        Head 2 → price range compatibility
        Head 3 → meal completion signal
        Head 4 → co-purchase pattern

    Outputs are concatenated and projected back to d_model.

    Use this instead of DINAttention when d_model >= 128
    and you want richer complementarity representations.
    For d_model = 64 single head is sufficient.
    """

    def __init__(
        self,
        d_model:    int,
        num_heads:  int   = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head    = d_model // num_heads

        # One DIN head per attention head
        self.heads = nn.ModuleList([
            DINAttention(self.d_head, hidden_dim=self.d_head * 2, dropout=dropout)
            for _ in range(num_heads)
        ])

        # Project concatenated heads back to d_model
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.d_model = d_model

    def forward(
        self,
        candidate_emb: torch.Tensor,
        cart_embs:     torch.Tensor,
        cart_mask:     Optional[torch.Tensor] = None,
    ):
        """
        Returns
        ───────
        cross_repr   : (B, d_model)
        attn_weights : (B, num_heads, N)
            One attention distribution per head.
            Average across heads for the heatmap.
        """
        B, N, d = cart_embs.shape

        # Split embeddings into num_heads chunks along d_model
        cand_chunks = candidate_emb.split(self.d_head, dim=-1)   # H × (B, d_head)
        cart_chunks = cart_embs.split(self.d_head, dim=-1)       # H × (B, N, d_head)

        head_outputs = []
        head_weights = []

        for i, head in enumerate(self.heads):
            h_cross, h_attn = head(
                cand_chunks[i], cart_chunks[i], cart_mask
            )
            head_outputs.append(h_cross)    # (B, d_head)
            head_weights.append(h_attn)     # (B, N)

        # Concat all head outputs → (B, d_model)
        cross_repr   = self.output_proj(
            torch.cat(head_outputs, dim=-1)
        )

        # Stack attention weights → (B, num_heads, N)
        attn_weights = torch.stack(head_weights, dim=1)

        return cross_repr, attn_weights




# ── sanity check ─────────────────────────────
if __name__ == "__main__":
    B, N, d = 8, 6, 64

    din = DINAttention(d_model=d, hidden_dim=64, dropout=0.1)

    cand = torch.randn(B, d)
    cart = torch.randn(B, N, d)
    mask = torch.ones(B, N, dtype=torch.long)
    mask[:, -2:] = 0   # last 2 positions padding

    cross_repr, attn_weights = din(cand, cart, mask)
    print(f"cross_repr   : {cross_repr.shape}")    # (8, 64)
    print(f"attn_weights : {attn_weights.shape}")  # (8, 6)

    # Multi-head version
    mdin = MultiHeadDINAttention(d_model=128, num_heads=4, dropout=0.1)
    cand2 = torch.randn(B, 128)
    cart2 = torch.randn(B, N, 128)
    cross2, weights2 = mdin(cand2, cart2, mask)
    print(f"MultiHead cross_repr   : {cross2.shape}")    # (8, 128)
    print(f"MultiHead attn_weights : {weights2.shape}")  # (8, 4, 6)

    total = sum(p.numel() for p in din.parameters())
    print(f"DIN parameters : {total:,}")