import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ── cross network v2 layer ────────────────────────────────────────────────────

class CrossLayerV2(nn.Module):
    """
    Single DCN-V2 Cross Layer.

    DCN-V1 used a rank-1 (vector) cross interaction — cheap but limited.
    DCN-V2 replaces it with a full weight matrix W ∈ R^{d×d}:

        x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l

    This captures ALL pairwise feature interactions between the
    input x_0 and the current representation x_l, which is critical
    for learning subtle food compatibility signals such as:

        "pizza + garlic bread"  (cuisine match)
        "spicy curry + lassi"   (complementary flavour)
        "kids meal + juice"     (demographic context)

    Two variants:
        'full'   — W is d×d  (expressive, use for small d)
        'low_rank' — W ≈ U·V^T where U,V ∈ R^{d×r}  (efficient for large d)
    """

    def __init__(
        self,
        d: int,
        variant: str  = 'low_rank',   # 'full' | 'low_rank'
        rank:    int  = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.variant = variant

        if variant == 'full':
            self.W = nn.Linear(d, d, bias=True)

        elif variant == 'low_rank':
            # W ≈ U · V^T  reduces O(d²) → O(d·r)
            self.U = nn.Linear(d, rank, bias=False)
            self.V = nn.Linear(rank, d, bias=True)

        else:
            raise ValueError(f"variant must be 'full' or 'low_rank', got {variant}")

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x0: torch.Tensor,   # (B, d)  — original input, held fixed
        xl: torch.Tensor,   # (B, d)  — current layer representation
    ) -> torch.Tensor:
        """
        Returns x_{l+1} : (B, d)
        """
        if self.variant == 'full':
            interaction = self.W(xl)                   # (B, d)
        else:
            interaction = self.V(self.U(xl))           # (B, d)

        # Element-wise product with x0 + residual
        out = x0 * interaction + xl                    # (B, d)
        return self.dropout(out)


# ── cross network v2 stack ────────────────────────────────────────────────────

class CrossNetworkV2(nn.Module):
    """
    Stack of CrossLayerV2 blocks.

    After L layers the network has implicitly modelled all
    feature interactions up to degree L+1 without enumerating them.
    This is what makes DCN-V2 so parameter-efficient for the
    food add-on ranking problem where interactions between
    100+ features matter.
    """

    def __init__(
        self,
        d:        int,
        num_layers: int  = 3,
        variant:  str   = 'low_rank',
        rank:     int   = 32,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossLayerV2(d, variant, rank, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x  : (B, d)
        out: (B, d)   — cross-interaction enriched representation
        """
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


# ── deep network (MLP with residuals) ────────────────────────────────────────

class DeepNetwork(nn.Module):
    """
    Standard deep MLP with residual connections.

    Matches your architecture doc:
        512 → 256 → 128 → 64

    Residual shortcuts are added whenever consecutive hidden dims
    are equal so gradients flow cleanly through the deep stack.
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: List[int] = [512, 256, 128, 64],
        dropout:     float     = 0.1,
        activation:  str       = 'gelu',
    ):
        super().__init__()

        act_fn = nn.GELU() if activation == 'gelu' else nn.ReLU()

        self.blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()

        in_dim = input_dim
        for out_dim in hidden_dims:
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                act_fn,
                nn.Dropout(dropout),
            )
            self.blocks.append(block)

            # Residual projection if dims differ
            if in_dim != out_dim:
                self.residuals.append(nn.Linear(in_dim, out_dim, bias=False))
            else:
                self.residuals.append(nn.Identity())

            in_dim = out_dim

        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, input_dim)
        out : (B, hidden_dims[-1])
        """
        for block, res in zip(self.blocks, self.residuals):
            x = block(x) + res(x)
        return x


# ── stacked mixture of experts (optional upgrade) ─────────────────────────────

class MoELayer(nn.Module):
    """
    Mixture of Experts layer — optional drop-in upgrade for the
    deep branch when you need more capacity without full width.

    E experts each produce a d_out vector; a gating network
    computes soft weights over experts using the input.

    Particularly useful for the food domain where user segments
    (vegan / non-veg / keto / …) benefit from specialised experts.
    """

    def __init__(
        self,
        d_in:      int,
        d_out:     int,
        num_experts: int  = 4,
        dropout:   float  = 0.1,
    ):
        super().__init__()

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.LayerNorm(d_out),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(d_in, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.gate(x)                          # (B, E)
        expert_outs = torch.stack(
            [e(x) for e in self.experts], dim=1
        )                                             # (B, E, d_out)
        out = (gates.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, d_out)
        return out


# ── parallel combination: cross + deep ───────────────────────────────────────

class ParallelCrossDeep(nn.Module):
    """
    Runs CrossNetworkV2 and DeepNetwork in PARALLEL on the same input
    then concatenates their outputs.

    This is the standard DCN-V2 'parallel' variant:

        x ──▶ CrossNet ──▶ x_cross ──┐
                                      ├──▶ concat ──▶ head
        x ──▶ DeepNet  ──▶ x_deep  ──┘

    The cross branch captures explicit high-order interactions;
    the deep branch captures implicit non-linear patterns.
    Together they are strictly more expressive than either alone.
    """

    def __init__(
        self,
        input_dim:        int,
        cross_layers:     int        = 3,
        cross_variant:    str        = 'low_rank',
        cross_rank:       int        = 32,
        deep_hidden_dims: List[int]  = [512, 256, 128, 64],
        dropout:          float      = 0.1,
    ):
        super().__init__()

        self.cross_net = CrossNetworkV2(
            d          = input_dim,
            num_layers = cross_layers,
            variant    = cross_variant,
            rank       = cross_rank,
            dropout    = dropout,
        )

        self.deep_net = DeepNetwork(
            input_dim   = input_dim,
            hidden_dims = deep_hidden_dims,
            dropout     = dropout,
        )

        self.output_dim = input_dim + self.deep_net.output_dim

    def forward(self, x: torch.Tensor):
        x_cross = self.cross_net(x)                         # (B, input_dim)
        x_deep  = self.deep_net(x)                         # (B, deep_out)
        return torch.cat([x_cross, x_deep], dim=-1)        # (B, input_dim + deep_out)


# ── main DCN-V2 ranking backbone ──────────────────────────────────────────────

class DCNV2Ranker(nn.Module):
    """
    Full DCN-V2 Ranking Backbone for CartComplete add-on recommendation.

    Input features (Stage 4 concatenation from your architecture doc)
    ─────────────────────────────────────────────────────────────────
        cart_repr        (B, d_model)   ← SetTransformer + SASRec fusion
        cross_repr       (B, d_model)   ← DIN cross attention output
        candidate_repr   (B, d_model)   ← multimodal item encoder output
        user_repr        (B, d_model)   ← user encoder output
        context_repr     (B, d_model)   ← context encoder output
        cross_features   (B, n_cross)   ← explicit cross features
        business_features(B, n_biz)     ← margin, stock, prep efficiency

    Pipeline
    ────────
    concat all inputs
        → input_proj  (project to backbone_dim)
        → BatchNorm   (stabilise diverse feature scales)
        → ParallelCrossDeep (cross + deep branches)
        → output_head (MLP → sigmoid → P(add item))

    Output
    ──────
    logit / probability : (B, 1)  — P(user adds candidate to cart)
    """

    def __init__(
        self,
        # ── input dims ────────────────────────────────────────────────────────
        d_model:           int        = 128,   # dim of each repr vector
        n_cross_features:  int        = 16,    # explicit cross feature count
        n_biz_features:    int        = 4,     # business feature count
        # ── backbone ──────────────────────────────────────────────────────────
        backbone_dim:      int        = 256,   # projected input dim
        cross_layers:      int        = 3,
        cross_variant:     str        = 'low_rank',
        cross_rank:        int        = 32,
        deep_hidden_dims:  List[int]  = [512, 256, 128, 64],
        # ── output head ───────────────────────────────────────────────────────
        head_hidden_dims:  List[int]  = [64, 32],
        dropout:           float      = 0.1,
        use_moe:           bool       = False,
        num_experts:       int        = 4,
    ):
        super().__init__()

        self.d_model = d_model

        # ── compute total raw input dim ───────────────────────────────────────
        # 5 repr vectors (cart, cross, candidate, user, context) + cross_features + biz
        self.raw_input_dim = (5 * d_model) + n_cross_features + n_biz_features

        # ── input projection + normalisation ─────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(self.raw_input_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Feature-scale normalisation — important because cross_features
        # (PMI scores, co-occurrence counts) live on very different scales
        # from the unit-norm repr vectors
        self.input_bn = nn.BatchNorm1d(backbone_dim)

        # ── optional MoE upgrade ──────────────────────────────────────────────
        if use_moe:
            self.moe = MoELayer(backbone_dim, backbone_dim, num_experts, dropout)
        else:
            self.moe = nn.Identity()

        # ── parallel cross + deep backbone ───────────────────────────────────
        self.backbone = ParallelCrossDeep(
            input_dim        = backbone_dim,
            cross_layers     = cross_layers,
            cross_variant    = cross_variant,
            cross_rank       = cross_rank,
            deep_hidden_dims = deep_hidden_dims,
            dropout          = dropout,
        )

        # ── output head ───────────────────────────────────────────────────────
        head_input_dim = self.backbone.output_dim
        head_layers    = []
        in_dim         = head_input_dim

        for h in head_hidden_dims:
            head_layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        head_layers.append(nn.Linear(in_dim, 1))   # logit
        self.output_head = nn.Sequential(*head_layers)

        self._init_weights()

    def _init_weights(self):
        """He init for linear layers, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        cart_repr:          torch.Tensor,            # (B, d_model)
        cross_repr:         torch.Tensor,            # (B, d_model)
        candidate_repr:     torch.Tensor,            # (B, d_model)
        user_repr:          torch.Tensor,            # (B, d_model)
        context_repr:       torch.Tensor,            # (B, d_model)
        cross_features:     torch.Tensor,            # (B, n_cross_features)
        business_features:  torch.Tensor,            # (B, n_biz_features)
        return_logit:       bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        All repr tensors come from their respective encoders.
        cross_features   : cuisine_compat, co-occurrence, PMI, price_compat,
                           novelty, popularity_bias, category_gap
        business_features: estimated_margin, attach_freq, stock_avail, prep_eff

        Returns
        -------
        prob  : (B, 1)   P(user adds candidate) — sigmoid applied
        logit : (B, 1)   raw logit — returned when return_logit=True
        """

        # ── stage 4: feature concatenation ───────────────────────────────────
        x = torch.cat([
            cart_repr,
            cross_repr,
            candidate_repr,
            user_repr,
            context_repr,
            cross_features,
            business_features,
        ], dim=-1)                                         # (B, raw_input_dim)

        # ── project + normalise ───────────────────────────────────────────────
        x = self.input_proj(x)                            # (B, backbone_dim)
        x = self.input_bn(x)                              # (B, backbone_dim)

        # ── optional MoE ─────────────────────────────────────────────────────
        x = self.moe(x)                                   # (B, backbone_dim)

        # ── parallel cross + deep ─────────────────────────────────────────────
        x = self.backbone(x)                              # (B, backbone_dim + deep_out)

        # ── output head → P(add item) ─────────────────────────────────────────
        logit = self.output_head(x)                       # (B, 1)

        if return_logit:
            return logit

        return torch.sigmoid(logit)                       # (B, 1)

    def predict_score(
        self,
        cart_repr:         torch.Tensor,
        cross_repr:        torch.Tensor,
        candidate_repr:    torch.Tensor,
        user_repr:         torch.Tensor,
        context_repr:      torch.Tensor,
        cross_features:    torch.Tensor,
        business_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference-time helper.
        Returns raw float score (B, 1) without sigmoid
        for use in Stage 5 re-ranking formula:

            FinalScore = CTR × Margin × DiversityBoost × NoveltyBoost
        """
        with torch.no_grad():
            return self.forward(
                cart_repr, cross_repr, candidate_repr,
                user_repr, context_repr,
                cross_features, business_features,
                return_logit=True,
            )


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    B        = 8
    d_model  = 128
    n_cross  = 16    # cuisine_compat, pmi, co-occur, price_compat, novelty …
    n_biz    = 4     # margin, attach_freq, stock, prep_eff

    model = DCNV2Ranker(
        d_model           = d_model,
        n_cross_features  = n_cross,
        n_biz_features    = n_biz,
        backbone_dim      = 256,
        cross_layers      = 3,
        cross_variant     = 'low_rank',
        cross_rank        = 32,
        deep_hidden_dims  = [512, 256, 128, 64],
        head_hidden_dims  = [64, 32],
        dropout           = 0.1,
        use_moe           = False,
    )

    # Dummy inputs — shapes match your architecture doc Stage 4
    cart_repr      = torch.randn(B, d_model)
    cross_repr     = torch.randn(B, d_model)
    candidate_repr = torch.randn(B, d_model)
    user_repr      = torch.randn(B, d_model)
    context_repr   = torch.randn(B, d_model)
    cross_feats    = torch.randn(B, n_cross)
    biz_feats      = torch.randn(B, n_biz)

    prob  = model(
        cart_repr, cross_repr, candidate_repr,
        user_repr, context_repr,
        cross_feats, biz_feats,
    )
    print(f"P(add item) shape : {prob.shape}")           # → (8, 1)
    print(f"P(add item) range : [{prob.min():.3f}, {prob.max():.3f}]")

    logit = model.predict_score(
        cart_repr, cross_repr, candidate_repr,
        user_repr, context_repr,
        cross_feats, biz_feats,
    )
    print(f"logit shape       : {logit.shape}")          # → (8, 1)

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters  : {total:,}")