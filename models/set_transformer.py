import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


# ── multi-head self attention ─────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention.
    Permutation-equivariant — each cart item attends to every
    other item and updates its representation accordingly.
    Captures pairwise signals like:
        burger ↔ cola       (drink affinity)
        pizza  ↔ garlic bread (cuisine match)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.scale     = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,   # (B, n) 1=real 0=pad
    ) -> torch.Tensor:

        B, n, _ = x.shape

        def split(t):
            return t.view(B, n, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # (B,H,n,n)

        if mask is not None:
            key_mask = (mask == 0).unsqueeze(1).unsqueeze(2)          # (B,1,1,n)
            scores   = scores.masked_fill(key_mask, -1e9)

        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out  = torch.matmul(attn, V)                                  # (B,H,n,d_k)
        out  = out.transpose(1, 2).contiguous().view(B, n, -1)
        return self.W_o(out)                                          # (B,n,d_model)


# ── self attention block ──────────────────────────────────────────────────────

class SAB(nn.Module):
    """
    Self-Attention Block:
        x → MHA(x,x) → Add & Norm → FFN → Add & Norm

    One SAB = one round of every cart item attending to
    every other cart item. Stacking 2 SABs captures
    second-order basket interactions efficiently.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ffn(x))
        return x


# ── mean + max pooling ────────────────────────────────────────────────────────

class MeanMaxPooling(nn.Module):
    """
    Permutation-invariant pooling over the cart sequence.

    Mean pool → average basket signal
    Max pool  → strongest / most salient item signal

    concat → (B, 2*d_model) → Linear → (B, d_model)

    Why both?
    Mean alone misses dominant items.
    Max alone misses basket-level composition.
    Together they give a richer fixed-size basket summary.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if mask is not None:
            mask_f   = mask.float().unsqueeze(-1)              # (B,n,1)
            lengths  = mask_f.sum(dim=1).clamp(min=1)         # (B,1)
            mean_vec = (x * mask_f).sum(dim=1) / lengths      # (B,d)
            max_vec  = x.masked_fill(
                mask_f == 0, -1e9
            ).max(dim=1).values                                # (B,d)
        else:
            mean_vec = x.mean(dim=1)
            max_vec  = x.max(dim=1).values

        return self.proj(
            torch.cat([mean_vec, max_vec], dim=-1)
        )                                                      # (B,d_model)


# ── SASRec sequential encoder ─────────────────────────────────────────────────

class SASRecEncoder(nn.Module):
    """
    Simplified SASRec — Self-Attentive Sequential Recommendation.

    Treats the ordered cart sequence as a temporal signal:
        item_1 → item_2 → ... → item_n

    Uses causal (left-to-right) self-attention so each position
    only attends to previous items — preserving order dynamics.

    Captures patterns like:
        "user always adds cola AFTER burger"
        "last item added was spicy → next is likely a drink"

    Output: last non-padding hidden state → seq_repr (B, d_model)
    """

    def __init__(
        self,
        d_model:    int,
        num_heads:  int,
        num_layers: int   = 2,
        max_len:    int   = 50,
        dropout:    float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Learnable positional embeddings — position 0 = first item added
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Causal transformer layers
        self.layers = nn.ModuleList([
            SAB(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _causal_mask(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular mask to enforce causality.
        Position i cannot attend to position j > i.
        Returns (n, n) bool mask — True = masked out.
        """
        return torch.triu(
            torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1
        )

    def forward(
        self,
        x:           torch.Tensor,              # (B, n, d_model)  projected embeddings
        padding_mask: Optional[torch.Tensor] = None,  # (B, n) 1=real 0=pad
    ) -> torch.Tensor:
        """
        Returns
        -------
        seq_repr : (B, d_model)
            Representation of the last real item in the sequence,
            capturing recency and order dynamics.
        """
        B, n, _ = x.shape
        device  = x.device

        # Add positional embeddings
        positions = torch.arange(n, device=device).unsqueeze(0)  # (1,n)
        x = self.dropout(x + self.pos_emb(positions))            # (B,n,d)

        # Causal mask — (n, n) broadcast over batch and heads
        causal = self._causal_mask(n, device)

        for layer in self.layers:
            # Manually apply causal masking inside SAB
            # Reuse SAB but pass causal attention via padding mask
            # We compute causal scores by masking attention directly
            x = self._causal_sab(layer, x, causal, padding_mask)

        x = self.norm(x)   # (B, n, d_model)

        # Extract last non-padding position per sequence
        if padding_mask is not None:
            lengths   = padding_mask.sum(dim=1) - 1          # (B,) last real idx
            lengths   = lengths.clamp(min=0)
            batch_idx = torch.arange(B, device=device)
            seq_repr  = x[batch_idx, lengths]                # (B, d_model)
        else:
            seq_repr = x[:, -1, :]                           # (B, d_model)

        return seq_repr

    def _causal_sab(
        self,
        sab:          SAB,
        x:            torch.Tensor,
        causal_mask:  torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Runs one SAB layer with causal masking injected into attention.
        """
        B, n, _ = x.shape
        H       = sab.attn.num_heads
        d_k     = sab.attn.d_k
        scale   = sab.attn.scale

        Q = sab.attn.W_q(x).view(B, n, H, d_k).transpose(1, 2)
        K = sab.attn.W_k(x).view(B, n, H, d_k).transpose(1, 2)
        V = sab.attn.W_v(x).view(B, n, H, d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B,H,n,n)

        # Apply causal mask
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), -1e9
        )

        # Apply padding mask on keys
        if padding_mask is not None:
            key_mask = (padding_mask == 0).unsqueeze(1).unsqueeze(2)
            scores   = scores.masked_fill(key_mask, -1e9)

        attn = sab.attn.attn_drop(F.softmax(scores, dim=-1))
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, n, -1)
        attn_out = sab.attn.W_o(out)

        # Residual + norm + FFN
        x = sab.norm1(x + attn_out)
        x = sab.norm2(x + sab.ffn(x))
        return x


# ── cart fusion layer ─────────────────────────────────────────────────────────

class CartFusionLayer(nn.Module):
    """
    Fuses SASRec (sequential) and Set Transformer (set) representations.

        [seq_repr ; set_repr]  →  Linear → LayerNorm → GELU  →  basket_repr

    seq_repr captures ORDER DYNAMICS  (what was added when)
    set_repr captures BASKET SEMANTICS (what is in the cart together)

    Together they give a basket_repr that knows both WHAT is in
    the cart and IN WHAT ORDER it was assembled — critical for
    food add-on recommendation where recency matters
    (last item added = strongest signal for next add-on).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        seq_repr: torch.Tensor,   # (B, d_model)  from SASRec
        set_repr: torch.Tensor,   # (B, d_model)  from Set Transformer
    ) -> torch.Tensor:            # (B, d_model)  basket_repr
        return self.fusion(
            torch.cat([seq_repr, set_repr], dim=-1)
        )


# ── main set transformer encoder ─────────────────────────────────────────────

class SetTransformerEncoder(nn.Module):
    """
    Cart Encoder — Final Merged Architecture.

    Encodes a variable-length ordered basket of item embeddings
    into a single basket_repr by running TWO parallel encoders:

        ┌─────────────────────────────────────────────┐
        │  SASRec (sequential)                        │
        │  causal self-attention over ordered items   │
        │  → seq_repr  (order dynamics + recency)     │
        └─────────────────────────────────────────────┘
                           +
        ┌─────────────────────────────────────────────┐
        │  Set Transformer (set)                      │
        │  SAB stack + mean/max pooling               │
        │  → set_repr  (basket semantics)             │
        └─────────────────────────────────────────────┘
                           ↓
                    CartFusionLayer
                           ↓
                     basket_repr  (B, d_model)
                           ↓
                    → GatedFusion (Stage 9)

    Input  : (B, n, input_dim)
    Output : (B, d_model)  — basket_repr
    """

    def __init__(
        self,
        input_dim:       int,
        d_model:         int   = 128,
        num_heads:       int   = 4,
        num_sab_layers:  int   = 2,
        num_sasrec_layers: int = 2,
        max_seq_len:     int   = 50,
        dropout:         float = 0.1,
    ):
        """
        Parameters
        ──────────
        input_dim         : dim of each item embedding
                            (item + text + category + graph + price)
        d_model           : internal and output dimension
        num_heads         : attention heads
        num_sab_layers    : SAB blocks in Set Transformer branch
        num_sasrec_layers : causal SAB blocks in SASRec branch
        max_seq_len       : max cart length for positional embeddings
        dropout           : dropout rate
        """
        super().__init__()

        self.d_model = d_model

        # Shared input projection — both branches start from same space
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Set Transformer branch ────────────────────────────────────────────
        self.sab_layers = nn.ModuleList([
            SAB(d_model, num_heads, dropout)
            for _ in range(num_sab_layers)
        ])
        self.pooling = MeanMaxPooling(d_model, dropout)

        # ── SASRec branch ─────────────────────────────────────────────────────
        self.sasrec = SASRecEncoder(
            d_model    = d_model,
            num_heads  = num_heads,
            num_layers = num_sasrec_layers,
            max_len    = max_seq_len,
            dropout    = dropout,
        )

        # ── Cart Fusion Layer ─────────────────────────────────────────────────
        self.cart_fusion = CartFusionLayer(d_model, dropout)

        # ── final norm ────────────────────────────────────────────────────────
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        item_embeddings: torch.Tensor,
        padding_mask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ──────────
        item_embeddings : (B, n, input_dim)
            Ordered sequence of cart item embeddings.
            Order matters — first item = first added to cart.
        padding_mask    : (B, n)  1=real item  0=padding
                          Pass None if all carts are same length.

        Returns
        ───────
        basket_repr : (B, d_model)
        """

        # Shared projection
        x = self.input_proj(item_embeddings)       # (B, n, d_model)

        # ── Set Transformer branch ────────────────────────────────────────────
        set_x = x
        for sab in self.sab_layers:
            set_x = sab(set_x, padding_mask)       # (B, n, d_model)
        set_repr = self.pooling(set_x, padding_mask)  # (B, d_model)

        # ── SASRec branch ─────────────────────────────────────────────────────
        seq_repr = self.sasrec(x, padding_mask)    # (B, d_model)

        # ── Cart Fusion ───────────────────────────────────────────────────────
        basket_repr = self.cart_fusion(seq_repr, set_repr)   # (B, d_model)
        basket_repr = self.output_norm(basket_repr)

        return basket_repr


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    B         = 8
    n         = 6
    input_dim = 256    # item + text + category + graph + price dims

    model = SetTransformerEncoder(
        input_dim         = input_dim,
        d_model           = 128,
        num_heads         = 4,
        num_sab_layers    = 2,
        num_sasrec_layers = 2,
        max_seq_len       = 50,
        dropout           = 0.1,
    )

    cart_emb = torch.randn(B, n, input_dim)
    mask     = torch.ones(B, n, dtype=torch.long)
    mask[:, -2:] = 0   # last 2 positions are padding

    basket_repr = model(cart_emb, padding_mask=mask)
    print(f"basket_repr shape : {basket_repr.shape}")   # → (8, 128)

    # Variable cart sizes
    sizes    = [2, 3, 4, 5, 6, 1, 3, 2]
    var_mask = torch.zeros(B, n, dtype=torch.long)
    for i, s in enumerate(sizes):
        var_mask[i, :s] = 1

    var_repr = model(cart_emb, padding_mask=var_mask)
    print(f"Variable cart size shape : {var_repr.shape}")   # → (8, 128)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total:,}")