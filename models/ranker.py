

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from models.dcn_v2 import DCNV2Ranker


# ── coverage head ─────────────────────────────────────────────────────────────

class CategoryCoverageHead(nn.Module):
    """
    Auxiliary head that predicts which food group
    categories are missing from the current cart.

    This is the novel training contribution of CartComplete.

    Input  : cart_repr (B, d_model) from SetTransformerEncoder
    Output : coverage_logits (B, 5) — one logit per food group
             [main, side, drink, snack, dessert]

    Training target:
        For each food group g:
            target[g] = 1 if group g is NOT in current cart
            target[g] = 0 if group g IS already in cart

    Loss: BCE averaged over 5 food groups
    Effect: Forces the Set Transformer to learn food group
            structure — what constitutes a complete meal.
            The ranker then implicitly upweights candidates
            that fill missing food groups.

    This is why CartComplete recommends a drink when
    the cart has no drink — not because drinks are popular
    but because the coverage head learned meal completion.
    """

    FOOD_GROUPS = ['main', 'side', 'drink', 'snack', 'dessert']
    N_GROUPS    = 5

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.N_GROUPS),
            # No sigmoid here — BCEWithLogitsLoss handles it
        )

    def forward(self, cart_repr: torch.Tensor) -> torch.Tensor:
        """
        cart_repr : (B, d_model)
        Returns   : (B, 5) logits — one per food group
        """
        return self.head(cart_repr)   # (B, 5)

    def predict_missing(self, cart_repr: torch.Tensor) -> Dict[str, float]:
        """
        Inference helper — returns dict of missing food group
        probabilities for a single cart (B=1).

        Use in explainer.py to generate natural language:
        "Recommended because your cart has no drink yet."
        """
        with torch.no_grad():
            logits = self.forward(cart_repr)        # (B, 5)
            probs  = torch.sigmoid(logits)          # (B, 5)

        result = {}
        for i, group in enumerate(self.FOOD_GROUPS):
            result[group] = probs[0, i].item()
        return result   # {main: 0.1, side: 0.3, drink: 0.9, ...}


# ── score calibration ─────────────────────────────────────────────────────────

class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration of ranking scores.

    Learned temperature T scales the logit before sigmoid:
        P(add) = sigmoid(logit / T)

    T > 1 → smoother probabilities (less confident)
    T < 1 → sharper probabilities (more confident)

    Calibration is important for the final score formula:
        FinalScore = CTR × Margin × DiversityBoost × NoveltyBoost

    Without calibration CTR scores are poorly scaled
    relative to the other multiplicative terms.

    Train T on val set after main model is trained.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)


# ── business score layer ──────────────────────────────────────────────────────

class BusinessScoreLayer(nn.Module):
    """
    Computes the final business-aware recommendation score.

    Beyond P(add item), production systems need to account for:
        margin_score    — restaurant profit on this item
        novelty_boost   — prefer items user hasn't tried before
        diversity_boost — prefer items from underrepresented categories

    Final formula from your architecture doc:
        FinalScore = CTR × Margin × DiversityBoost × NoveltyBoost

    All terms are clamped to [0, 1] before multiplication
    to prevent score explosion.

    This is NOT trained — it is a deterministic post-processing
    layer applied after the ranker produces CTR scores.
    """

    def __init__(
        self,
        margin_weight:    float = 0.3,
        novelty_weight:   float = 0.2,
        diversity_weight: float = 0.2,
    ):
        super().__init__()
        self.margin_w    = margin_weight
        self.novelty_w   = novelty_weight
        self.diversity_w = diversity_weight

    def forward(
        self,
        ctr_prob:        torch.Tensor,   # (B,)  from ranker sigmoid
        margin_score:    torch.Tensor,   # (B,)  normalized 0-1
        novelty_score:   torch.Tensor,   # (B,)  1 = never ordered before
        diversity_score: torch.Tensor,   # (B,)  1 = fills missing category
    ) -> torch.Tensor:
        """
        Returns final_score : (B,) in [0, 1]
        """
        # Clamp all inputs to [0.01, 1] to avoid zero products
        ctr  = ctr_prob.clamp(0.01, 1.0)
        mg   = margin_score.clamp(0.01, 1.0)
        nov  = novelty_score.clamp(0.01, 1.0)
        div  = diversity_score.clamp(0.01, 1.0)

        # Weighted geometric mean
        # Pure multiplication would collapse to near-zero
        # for items with low margin or low novelty
        score = (
            ctr ** (1 - self.margin_w - self.novelty_w - self.diversity_w)
            * mg  ** self.margin_w
            * nov ** self.novelty_w
            * div ** self.diversity_w
        )
        return score.clamp(0.0, 1.0)


# ── main ranker ───────────────────────────────────────────────────────────────

# class CartCompleteRanker(nn.Module):
#     """
#     Full ranking module for CartComplete.

#     Wraps DCNV2Ranker (from dcn_v2.py) and adds:
#         1. CategoryCoverageHead  — auxiliary coverage loss
#         2. TemperatureScaling    — probability calibration
#         3. BusinessScoreLayer    — final score formula

#     This is the last model component before the API response.
#     Every candidate from the retrieval stage (Stage 1) passes
#     through this module to get its final ranking score.

#     Forward pass returns:
#         add_prob         : (B, 1)   P(user adds this item)
#         coverage_logits  : (B, 5)   food group missing logits
#         final_score      : (B,)     business-aware ranking score

#     Training uses:
#         L = L_bpr + 0.1 × L_coverage
#     """

#     def __init__(
#         self,
#         d_model:          int        = 128,
#         n_cross_features: int        = 16,
#         n_biz_features:   int        = 4,
#         backbone_dim:     int        = 256,
#         cross_layers:     int        = 3,
#         cross_variant:    str        = 'low_rank',
#         cross_rank:       int        = 32,
#         deep_hidden_dims: list       = [512, 256, 128, 64],
#         head_hidden_dims: list       = [64, 32],
#         dropout:          float      = 0.1,
#         use_moe:          bool       = False,
#         num_experts:      int        = 4,
#         margin_weight:    float      = 0.3,
#         novelty_weight:   float      = 0.2,
#         diversity_weight: float      = 0.2,
#     ):
#         super().__init__()

#         # ── main ranking backbone ─────────────────────────────────────────────
#         self.dcnv2 = DCNV2Ranker(
#             d_model           = d_model,
#             n_cross_features  = n_cross_features,
#             n_biz_features    = n_biz_features,
#             backbone_dim      = backbone_dim,
#             cross_layers      = cross_layers,
#             cross_variant     = cross_variant,
#             cross_rank        = cross_rank,
#             deep_hidden_dims  = deep_hidden_dims,
#             head_hidden_dims  = head_hidden_dims,
#             dropout           = dropout,
#             use_moe           = use_moe,
#             num_experts       = num_experts,
#         )

#         # ── auxiliary coverage head ───────────────────────────────────────────
#         # Takes cart_repr directly — shares no weights with dcnv2
#         self.coverage_head = CategoryCoverageHead(d_model, dropout)

#         # ── temperature calibration ───────────────────────────────────────────
#         self.temp_scaling = TemperatureScaling()

#         # ── business score layer ──────────────────────────────────────────────
#         self.biz_scorer = BusinessScoreLayer(
#             margin_weight    = margin_weight,
#             novelty_weight   = novelty_weight,
#             diversity_weight = diversity_weight,
#         )

#         self.d_model = d_model

#     def forward(
#         self,
#         # ── from GatedFusion ─────────────────────────────────────────────────
#         cart_repr:          torch.Tensor,   # (B, d_model)
#         cross_repr:         torch.Tensor,   # (B, d_model)
#         candidate_repr:     torch.Tensor,   # (B, d_model)
#         user_repr:          torch.Tensor,   # (B, d_model)
#         context_repr:       torch.Tensor,   # (B, d_model)
#         fused_repr:         torch.Tensor,   # (B, d_model) from GatedFusion
#         # ── explicit features ─────────────────────────────────────────────────
#         cross_features:     torch.Tensor,   # (B, n_cross_features)
#         biz_features:       torch.Tensor,   # (B, n_biz_features)
#         # ── business score inputs ─────────────────────────────────────────────
#         margin_score:       Optional[torch.Tensor] = None,   # (B,)
#         novelty_score:      Optional[torch.Tensor] = None,   # (B,)
#         diversity_score:    Optional[torch.Tensor] = None,   # (B,)
#         # ── mode flags ───────────────────────────────────────────────────────
#         return_logit:       bool = False,
#         compute_final_score:bool = False,
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Parameters
#         ──────────
#         cart_repr      : (B, d_model) — from SetTransformerEncoder
#                          Used by coverage head to predict missing groups.
#         fused_repr     : (B, d_model) — from GatedFusion
#                          Replaces cart_repr as input to dcnv2 since it
#                          fuses all five signals already.
#         cross_features : (B, n_cross) — PMI, cuisine compat, price ratio,
#                          category gap, co-occurrence, novelty, popularity
#         biz_features   : (B, n_biz)  — margin, attach_freq, stock, prep_eff

#         Returns dict with:
#             add_prob        : (B, 1)   P(user adds this item) — sigmoid
#             add_logit       : (B, 1)   raw logit (if return_logit=True)
#             coverage_logits : (B, 5)   food group missing logits
#             final_score     : (B,)     business-aware score (if requested)
#         """

#         # ── main ranking forward ──────────────────────────────────────────────
#         # Pass fused_repr as cart_repr to DCNV2 since it already
#         # incorporates all five signals from GatedFusion
#         raw_logit = self.dcnv2(
#             cart_repr         = fused_repr,        # fused signal
#             cross_repr        = cross_repr,
#             candidate_repr    = candidate_repr,
#             user_repr         = user_repr,
#             context_repr      = context_repr,
#             cross_features    = cross_features,
#             business_features = biz_features,
#             return_logit      = True,
#         )   # (B, 1)

#         # Temperature calibration
#         calibrated_logit = self.temp_scaling(raw_logit)   # (B, 1)
#         add_prob         = torch.sigmoid(calibrated_logit) # (B, 1)

#         # ── coverage head ─────────────────────────────────────────────────────
#         # Uses cart_repr (not fused_repr) — we want the coverage head
#         # to learn from the cart state alone, not from the candidate signal
#         coverage_logits = self.coverage_head(cart_repr)    # (B, 5)

#         # ── build output dict ─────────────────────────────────────────────────
#         out = {
#             'add_prob':        add_prob,
#             'coverage_logits': coverage_logits,
#         }

#         if return_logit:
#             out['add_logit'] = calibrated_logit

#         # ── final business-aware score ────────────────────────────────────────
#         if compute_final_score:
#             B      = add_prob.size(0)
#             device = add_prob.device

#             mg  = margin_score    if margin_score    is not None else torch.ones(B, device=device)
#             nov = novelty_score   if novelty_score   is not None else torch.ones(B, device=device)
#             div = diversity_score if diversity_score is not None else torch.ones(B, device=device)

#             out['final_score'] = self.biz_scorer(
#                 add_prob.squeeze(-1), mg, nov, div
#             )   # (B,)

#         return out


# # ── combined training loss ────────────────────────────────────────────────────

# class CartCompleteLoss(nn.Module):
#     """
#     Combined training loss for CartComplete.

#     L = L_bpr + lambda_cov × L_coverage

#     L_bpr      : BPR pairwise ranking loss
#                  added item scores higher than hard negatives
#                  Operates on add_prob (B,) or add_logit (B,)

#     L_coverage : BCE over food group missing predictions
#                  Forces cart encoder to learn meal structure
#                  lambda = 0.1 keeps it as a soft regularizer

#     Both losses are computed per batch and averaged.
#     """

#     def __init__(self, lambda_coverage: float = 0.1):
#         super().__init__()
#         self.lambda_cov = lambda_coverage
#         self.bce        = nn.BCEWithLogitsLoss()

#     def bpr_loss(
#         self,
#         pos_logits: torch.Tensor,   # (B,)  logits for positive items
#         neg_logits: torch.Tensor,   # (B, K) logits for K negatives per positive
#     ) -> torch.Tensor:
#         """
#         BPR: maximize margin between positive and negative scores.
#         L_bpr = -mean( log σ(pos - neg) )

#         neg_logits may have K negatives per positive.
#         We average over all K negatives per positive.
#         """
#         # pos_logits : (B,) → (B, 1) → broadcast over K negatives
#         pos = pos_logits.unsqueeze(-1)                      # (B, 1)
#         diff = pos - neg_logits                             # (B, K)
#         loss = -F.logsigmoid(diff).mean()
#         return loss

#     def coverage_loss(
#         self,
#         coverage_logits: torch.Tensor,   # (B, 5)
#         cart_has_group:  torch.Tensor,   # (B, 5)  1=group present 0=missing
#     ) -> torch.Tensor:
#         """
#         BCE loss for food group coverage prediction.

#         Target: 1 if food group is MISSING from cart (needs to be filled)
#                 0 if food group is PRESENT in cart (already covered)

#         Note: cart_has_group has 1=present so target = 1 - cart_has_group
#         """
#         missing_target = 1.0 - cart_has_group.float()      # (B, 5)
#         return self.bce(coverage_logits, missing_target)

#     def forward(
#         self,
#         pos_logits:      torch.Tensor,   # (B,)
#         neg_logits:      torch.Tensor,   # (B, K)
#         coverage_logits: torch.Tensor,   # (B, 5)
#         cart_has_group:  torch.Tensor,   # (B, 5)  1=group present
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Returns dict with individual and total losses.
#         Log all three in W&B during training.
#         """
#         l_bpr  = self.bpr_loss(pos_logits, neg_logits)
#         l_cov  = self.coverage_loss(coverage_logits, cart_has_group)
#         l_total = l_bpr + self.lambda_cov * l_cov

#         return {
#             'loss':          l_total,
#             'loss_bpr':      l_bpr,
#             'loss_coverage': l_cov,
#         }





# ── sanity check ─────────────────────────────
if __name__ == "__main__":
    B, d, n_cross, n_biz = 8, 128, 16, 4

    ranker = CartCompleteRanker(
        d_model          = d,
        n_cross_features = n_cross,
        n_biz_features   = n_biz,
        backbone_dim     = 256,
        cross_layers     = 3,
        dropout          = 0.1,
    )

    cart_repr      = torch.randn(B, d)
    cross_repr     = torch.randn(B, d)
    candidate_repr = torch.randn(B, d)
    user_repr      = torch.randn(B, d)
    context_repr   = torch.randn(B, d)
    fused_repr     = torch.randn(B, d)
    cross_feats    = torch.randn(B, n_cross)
    biz_feats      = torch.randn(B, n_biz)

    out = ranker(
        cart_repr       = cart_repr,
        cross_repr      = cross_repr,
        candidate_repr  = candidate_repr,
        user_repr       = user_repr,
        context_repr    = context_repr,
        fused_repr      = fused_repr,
        cross_features  = cross_feats,
        biz_features    = biz_feats,
        return_logit    = True,
        compute_final_score = True,
        margin_score    = torch.rand(B),
        novelty_score   = torch.rand(B),
        diversity_score = torch.rand(B),
    )

    print(f"add_prob        : {out['add_prob'].shape}")         # (8, 1)
    print(f"add_logit       : {out['add_logit'].shape}")        # (8, 1)
    print(f"coverage_logits : {out['coverage_logits'].shape}")  # (8, 5)
    print(f"final_score     : {out['final_score'].shape}")      # (8,)

    # Coverage head — missing food group prediction
    missing = ranker.coverage_head.predict_missing(cart_repr[:1])
    print(f"\nMissing food groups (single cart):")
    for group, prob in missing.items():
        print(f"  {group:<10}: {prob:.4f}")

    # Loss computation
    loss_fn  = CartCompleteLoss(lambda_coverage=0.1)
    K        = 4   # hard negatives per positive
    losses   = loss_fn(
        pos_logits      = out['add_logit'].squeeze(-1)[:B//2],
        neg_logits      = torch.randn(B//2, K),
        coverage_logits = out['coverage_logits'][:B//2],
        cart_has_group  = torch.randint(0, 2, (B//2, 5)).float(),
    )
    print(f"\nLoss total    : {losses['loss'].item():.4f}")
    print(f"Loss BPR      : {losses['loss_bpr'].item():.4f}")
    print(f"Loss coverage : {losses['loss_coverage'].item():.4f}")

    total = sum(p.numel() for p in ranker.parameters())
    print(f"\nTotal parameters : {total:,}")

# ── main ranker ───────────────────────────────────────────────────────────────

class CartCompleteRanker(nn.Module):
    def __init__(
        self,
        d_model:          int        = 128,
        n_cross_features: int        = 16,
        n_biz_features:   int        = 4,
        backbone_dim:     int        = 256,
        cross_layers:     int        = 3,
        cross_variant:    str        = "low_rank",
        cross_rank:       int        = 32,
        deep_hidden_dims: list       = [512, 256, 128, 64],
        head_hidden_dims: list       = [64, 32],
        dropout:          float      = 0.1,
        use_moe:          bool       = False,
        num_experts:      int        = 4,
        margin_weight:    float      = 0.3,
        novelty_weight:   float      = 0.2,
        diversity_weight: float      = 0.2,
    ):
        super().__init__()
        from models.dcn_v2 import DCNV2Ranker

        self.dcnv2 = DCNV2Ranker(
            d_model           = d_model,
            n_cross_features  = n_cross_features,
            n_biz_features    = n_biz_features,
            backbone_dim      = backbone_dim,
            cross_layers      = cross_layers,
            cross_variant     = cross_variant,
            cross_rank        = cross_rank,
            deep_hidden_dims  = deep_hidden_dims,
            head_hidden_dims  = head_hidden_dims,
            dropout           = dropout,
            use_moe           = use_moe,
            num_experts       = num_experts,
        )
        self.coverage_head = CategoryCoverageHead(d_model, dropout)
        self.temp_scaling  = TemperatureScaling()
        self.biz_scorer    = BusinessScoreLayer(
            margin_weight    = margin_weight,
            novelty_weight   = novelty_weight,
            diversity_weight = diversity_weight,
        )
        self.d_model = d_model

    def forward(
        self,
        cart_repr:          torch.Tensor,
        cross_repr:         torch.Tensor,
        candidate_repr:     torch.Tensor,
        user_repr:          torch.Tensor,
        context_repr:       torch.Tensor,
        fused_repr:         torch.Tensor,
        cross_features:     torch.Tensor,
        biz_features:       torch.Tensor,
        margin_score:       torch.Tensor  = None,
        novelty_score:      torch.Tensor  = None,
        diversity_score:    torch.Tensor  = None,
        return_logit:       bool = False,
        compute_final_score:bool = False,
    ):
        raw_logit = self.dcnv2(
            cart_repr         = fused_repr,
            cross_repr        = cross_repr,
            candidate_repr    = candidate_repr,
            user_repr         = user_repr,
            context_repr      = context_repr,
            cross_features    = cross_features,
            business_features = biz_features,
            return_logit      = True,
        )
        calibrated_logit = self.temp_scaling(raw_logit)
        add_prob         = torch.sigmoid(calibrated_logit)
        coverage_logits  = self.coverage_head(cart_repr)

        out = {
            "add_prob":        add_prob,
            "coverage_logits": coverage_logits,
        }
        if return_logit:
            out["add_logit"] = calibrated_logit

        if compute_final_score:
            B      = add_prob.size(0)
            device = add_prob.device
            mg  = margin_score    if margin_score    is not None else torch.ones(B, device=device)
            nov = novelty_score   if novelty_score   is not None else torch.ones(B, device=device)
            div = diversity_score if diversity_score is not None else torch.ones(B, device=device)
            out["final_score"] = self.biz_scorer(
                add_prob.squeeze(-1), mg, nov, div
            )
        return out
