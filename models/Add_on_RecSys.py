import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from models.item_encoder    import ItemEncoder
from models.set_transformer import SetTransformerEncoder
from models.user_encoder    import SmartUserEncoder
from models.context_encoder import ContextEncoder
from models.din_attention   import DINAttention, MultiHeadDINAttention
from models.gated_fusion    import GatedFusion
from models.ranker          import CartCompleteRanker
from training.losses        import CartCompleteLoss


# ── explicit cross feature extractor ─────────────────────────────────────────

class ExplicitCrossFeatures(nn.Module):
    """
    Computes handcrafted cross features between
    the cart and the candidate item.

    These are the MTGR explicit cross features —
    the paper showed that dropping them causes
    performance collapse that scaling cannot recover.

    Features computed:
        category_gap      — 1 if candidate fills a missing food group
        price_ratio       — candidate_price / avg_cart_item_price
        price_add_pct     — candidate_price / cart_total
        cuisine_compat    — precomputed from PMI (passed in as input)
        pmi_score         — item-item PMI from co-occurrence graph
        co_occur_score    — raw co-occurrence count (normalized)
        novelty_score     — 1 if user never ordered this item before
        popularity_bias   — global popularity rank of candidate (normalized)

    All scalars are concat → (B, n_cross) and passed to ranker.
    n_cross = 8 base features + any additional ones you add.

    Note: pmi_score and co_occur_score are looked up from
    the precomputed PMI matrix in outputs/
    and passed in as tensors — not computed here.
    """

    N_FEATURES = 8

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        # Light projection to normalize the raw feature values
        self.norm = nn.Sequential(
            nn.Linear(self.N_FEATURES, self.N_FEATURES),
            nn.LayerNorm(self.N_FEATURES),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        # cart state features
        cart_has_main:    torch.Tensor,   # (B,) float 0/1
        cart_has_side:    torch.Tensor,   # (B,) float 0/1
        cart_has_drink:   torch.Tensor,   # (B,) float 0/1
        cart_has_snack:   torch.Tensor,   # (B,) float 0/1
        cart_has_dessert: torch.Tensor,   # (B,) float 0/1
        cart_total:       torch.Tensor,   # (B,) float — sum of cart prices
        cart_size:        torch.Tensor,   # (B,) float — n items in cart
        # candidate features
        cand_food_group:  torch.Tensor,   # (B,) int — food group index
        cand_price:       torch.Tensor,   # (B,) float
        cand_popularity:  torch.Tensor,   # (B,) float 0-1 normalized rank
        # precomputed graph features
        pmi_score:        torch.Tensor,   # (B,) float — from PMI matrix
        co_occur_score:   torch.Tensor,   # (B,) float — normalized
        # user-candidate features
        novelty_score:    torch.Tensor,   # (B,) float 1=never ordered
        cuisine_compat:   torch.Tensor,   # (B,) float 0-1 compatibility
    ) -> torch.Tensor:
        """
        Returns cross_features : (B, N_FEATURES)
        """

        # category_gap: does candidate fill a missing food group?
        # Map food group index to corresponding has_* flag
        cart_coverage = torch.stack([
            cart_has_main,
            cart_has_side,
            cart_has_drink,
            cart_has_snack,
            cart_has_dessert,
        ], dim=1)   # (B, 5)

        # One-hot of candidate food group → (B, 5)
        # clamp to [0, 4] in case of unknown group index
        group_idx     = cand_food_group.clamp(0, 4)
        cand_onehot   = F.one_hot(group_idx, num_classes=5).float()

        # Gap = 1 if candidate's group not covered in cart
        group_covered = (cart_coverage * cand_onehot).sum(dim=1)   # (B,)
        category_gap  = 1.0 - group_covered                        # (B,)

        # price features
        avg_cart_price = cart_total / cart_size.clamp(min=1)        # (B,)
        price_ratio    = cand_price / avg_cart_price.clamp(min=0.01)
        price_add_pct  = cand_price / cart_total.clamp(min=0.01)

        # Clamp ratios to reasonable range
        price_ratio    = price_ratio.clamp(0.0, 5.0) / 5.0         # normalize
        price_add_pct  = price_add_pct.clamp(0.0, 1.0)

        # Stack all features → (B, N_FEATURES)
        raw = torch.stack([
            category_gap,                           # 1
            price_ratio,                            # 2
            price_add_pct,                          # 3
            cuisine_compat,                         # 4
            pmi_score.clamp(0.0, 10.0) / 10.0,    # 5  normalize PMI
            co_occur_score,                         # 6
            novelty_score,                          # 7
            cand_popularity,                        # 8
        ], dim=1)   # (B, 8)

        return self.norm(raw)   # (B, N_FEATURES)


# ── main model ────────────────────────────────────────────────────────────────

class AddOnRecSys(nn.Module):
    """
    CartComplete — Full Add-On Recommendation System.

    Named AddOnRecSys to match your file: addon_recsys.py

    Single forward() call runs the full pipeline:
        item encoding → cart encoding → candidate encoding
        → user encoding → context encoding
        → DIN cross attention → explicit cross features
        → gated fusion → DCN-V2 ranker
        → P(add item) + coverage logits + attention weights

    All components are initialized here and their
    hyperparameters are controlled via configs/model.yaml.
    """

    def __init__(
        self,
        # ── item encoder ──────────────────────────────────────────────────────
        num_items:          int,
        num_categories:     int         = 5,
        text_emb_dim:       int         = 384,    # MiniLM output
        # ── shared d_model ────────────────────────────────────────────────────
        d_model:            int         = 128,
        # ── cart encoder ──────────────────────────────────────────────────────
        n_heads:            int         = 4,
        n_sab_layers:       int         = 2,
        n_sasrec_layers:    int         = 2,
        max_cart_len:       int         = 50,
        # ── user encoder ──────────────────────────────────────────────────────
        n_user_features:    int         = 8,
        # ── context encoder ───────────────────────────────────────────────────
        num_restaurants:    int         = 10000,
        num_cuisine_types:  int         = 32,
        # ── DIN attention ─────────────────────────────────────────────────────
        din_hidden_dim:     int         = 64,
        use_multihead_din:  bool        = False,
        din_heads:          int         = 4,
        # ── ranker ────────────────────────────────────────────────────────────
        n_cross_features:   int         = 8,
        n_biz_features:     int         = 4,
        backbone_dim:       int         = 256,
        cross_layers:       int         = 3,
        cross_variant:      str         = 'low_rank',
        cross_rank:         int         = 32,
        deep_hidden_dims:   list        = [512, 256, 128, 64],
        head_hidden_dims:   list        = [64, 32],
        use_moe:            bool        = False,
        num_experts:        int         = 4,
        # ── loss ──────────────────────────────────────────────────────────────
        lambda_coverage:    float       = 0.1,
        # ── general ───────────────────────────────────────────────────────────
        dropout:            float       = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # ── item encoder (shared for cart items and candidate) ────────────────
        # item_emb dim: d_model
        # category_emb: d_model // 4
        # price_emb:    d_model // 4
        # text_proj:    d_model
        # fused output: d_model
        self.item_encoder = ItemEncoder(
            num_items      = num_items,
            num_categories = num_categories,
            text_emb_dim   = text_emb_dim,
            d_model        = d_model,
            dropout        = dropout,
        )

        # Item encoder output dim fed into cart encoder
        # SetTransformerEncoder expects input_dim = d_model
        item_out_dim = d_model

        # ── cart encoder: Set Transformer + SASRec ────────────────────────────
        self.cart_encoder = SetTransformerEncoder(
            input_dim          = item_out_dim,
            d_model            = d_model,
            num_heads          = n_heads,
            num_sab_layers     = n_sab_layers,
            num_sasrec_layers  = n_sasrec_layers,
            max_seq_len        = max_cart_len,
            dropout            = dropout,
        )

        # ── user encoder ──────────────────────────────────────────────────────
        self.user_encoder = SmartUserEncoder(
            d_model          = d_model,
            n_user_features  = n_user_features,
            dropout          = dropout,
        )

        # ── context encoder ───────────────────────────────────────────────────
        self.context_encoder = ContextEncoder(
            d_model            = d_model,
            num_restaurants    = num_restaurants,
            num_cuisine_types  = num_cuisine_types,
            dropout            = dropout,
        )

        # ── DIN cross attention ───────────────────────────────────────────────
        if use_multihead_din:
            self.din = MultiHeadDINAttention(
                d_model   = d_model,
                num_heads = din_heads,
                dropout   = dropout,
            )
        else:
            self.din = DINAttention(
                d_model    = d_model,
                hidden_dim = din_hidden_dim,
                dropout    = dropout,
            )
        self.use_multihead_din = use_multihead_din

        # ── explicit cross features ───────────────────────────────────────────
        self.cross_feat_extractor = ExplicitCrossFeatures(dropout=dropout)

        # ── gated fusion ──────────────────────────────────────────────────────
        self.gated_fusion = GatedFusion(
            d_model = d_model,
            dropout = dropout,
        )

        # ── ranker ────────────────────────────────────────────────────────────
        self.ranker = CartCompleteRanker(
            d_model          = d_model,
            n_cross_features = n_cross_features,
            n_biz_features   = n_biz_features,
            backbone_dim     = backbone_dim,
            cross_layers     = cross_layers,
            cross_variant    = cross_variant,
            cross_rank       = cross_rank,
            deep_hidden_dims = deep_hidden_dims,
            head_hidden_dims = head_hidden_dims,
            use_moe          = use_moe,
            num_experts      = num_experts,
            dropout          = dropout,
        )

        # ── loss function ─────────────────────────────────────────────────────
        self.loss_fn = CartCompleteLoss(lambda_coverage=lambda_coverage)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def encode_cart(
        self,
        cart_item_ids:  torch.Tensor,             # (B, N)
        cart_categories:torch.Tensor,             # (B, N)
        cart_prices:    torch.Tensor,             # (B, N) float
        cart_text_embs: torch.Tensor,             # (B, N, text_emb_dim)
        cart_mask:      Optional[torch.Tensor],   # (B, N) 1=real 0=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all cart items and produce:
            cart_item_embs : (B, N, d_model) — per-item vectors
            basket_repr    : (B, d_model)    — full cart summary

        Separated so DIN attention can use cart_item_embs
        and the rest of the pipeline uses basket_repr.
        """
        B, N = cart_item_ids.shape

        # Encode each cart item independently
        # Reshape to (B*N, ...) → ItemEncoder → reshape back
        ids_flat = cart_item_ids.reshape(B * N)
        cats_flat  = cart_categories.reshape(B * N)
        price_flat = cart_prices.reshape(B * N)
        text_flat  = cart_text_embs.reshape(B * N, -1)

        item_embs_flat = self.item_encoder(
            item_id   = ids_flat,
            category  = cats_flat,
            price     = price_flat,
            text_emb  = text_flat,
        )   # (B*N, d_model)

        cart_item_embs = item_embs_flat.view(B, N, self.d_model)  # (B, N, d_model)

        # Encode full cart → basket_repr
        basket_repr = self.cart_encoder(
            item_embeddings = cart_item_embs,
            padding_mask    = cart_mask,
        )   # (B, d_model)

        return cart_item_embs, basket_repr

    def encode_candidate(
        self,
        cand_item_id:  torch.Tensor,   # (B,)
        cand_category: torch.Tensor,   # (B,)
        cand_price:    torch.Tensor,   # (B,) float
        cand_text_emb: torch.Tensor,   # (B, text_emb_dim)
    ) -> torch.Tensor:
        """
        Encode candidate item → candidate_repr (B, d_model)
        """
        return self.item_encoder(
            item_id  = cand_item_id,
            category = cand_category,
            price    = cand_price,
            text_emb = cand_text_emb,
        )   # (B, d_model)

    def forward(
        self,
        # ── cart inputs ───────────────────────────────────────────────────────
        cart_item_ids:    torch.Tensor,             # (B, N)
        cart_categories:  torch.Tensor,             # (B, N)
        cart_prices:      torch.Tensor,             # (B, N) float
        cart_text_embs:   torch.Tensor,             # (B, N, text_emb_dim)
        cart_mask:        Optional[torch.Tensor],   # (B, N) 1=real 0=pad
        # ── candidate inputs ──────────────────────────────────────────────────
        cand_item_id:     torch.Tensor,             # (B,)
        cand_category:    torch.Tensor,             # (B,)
        cand_price:       torch.Tensor,             # (B,) float
        cand_text_emb:    torch.Tensor,             # (B, text_emb_dim)
        cand_food_group:  torch.Tensor,             # (B,) int 0-4
        cand_popularity:  torch.Tensor,             # (B,) float
        # ── user inputs ───────────────────────────────────────────────────────
        past_basket_embs: torch.Tensor,             # (B, H, d_model)
        user_features:    torch.Tensor,             # (B, n_user_features)
        basket_mask:      Optional[torch.Tensor],   # (B, H) 1=real 0=pad
        # ── context inputs ────────────────────────────────────────────────────
        hour:             torch.Tensor,             # (B,) int 0-23
        day_of_week:      torch.Tensor,             # (B,) int 0-6
        month:            Optional[torch.Tensor] = None,
        is_weekend:       Optional[torch.Tensor] = None,
        meal_slot:        Optional[torch.Tensor] = None,
        weather_type:     Optional[torch.Tensor] = None,
        temperature:      Optional[torch.Tensor] = None,
        humidity:         Optional[torch.Tensor] = None,
        festival_id:      Optional[torch.Tensor] = None,
        days_to_event:    Optional[torch.Tensor] = None,
        is_holiday:       Optional[torch.Tensor] = None,
        restaurant_id:    Optional[torch.Tensor] = None,
        cuisine_type:     Optional[torch.Tensor] = None,
        avg_prep_time:    Optional[torch.Tensor] = None,
        avg_rating:       Optional[torch.Tensor] = None,
        # ── explicit cross feature inputs ─────────────────────────────────────
        cart_has_main:    Optional[torch.Tensor] = None,   # (B,)
        cart_has_side:    Optional[torch.Tensor] = None,
        cart_has_drink:   Optional[torch.Tensor] = None,
        cart_has_snack:   Optional[torch.Tensor] = None,
        cart_has_dessert: Optional[torch.Tensor] = None,
        cart_total:       Optional[torch.Tensor] = None,   # (B,) float
        cart_size:        Optional[torch.Tensor] = None,   # (B,) float
        pmi_score:        Optional[torch.Tensor] = None,   # (B,)
        co_occur_score:   Optional[torch.Tensor] = None,   # (B,)
        novelty_score:    Optional[torch.Tensor] = None,   # (B,)
        cuisine_compat:   Optional[torch.Tensor] = None,   # (B,)
        # ── business features ─────────────────────────────────────────────────
        biz_features:     Optional[torch.Tensor] = None,   # (B, n_biz)
        margin_score:     Optional[torch.Tensor] = None,   # (B,)
        diversity_score:  Optional[torch.Tensor] = None,   # (B,)
        # ── mode flags ────────────────────────────────────────────────────────
        return_logit:           bool = False,
        compute_final_score:    bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full end-to-end forward pass.

        Returns dict:
            add_prob        : (B, 1)   P(user adds candidate)
            coverage_logits : (B, 5)   food group missing logits
            attn_weights    : (B, N) or (B, n_heads, N)
            add_logit       : (B, 1)   if return_logit=True
            final_score     : (B,)     if compute_final_score=True
        """
        B  = cart_item_ids.size(0)
        device = cart_item_ids.device

        def _ones(shape): return torch.ones(shape, device=device)
        def _zeros(shape): return torch.zeros(shape, device=device)

        # ── Stage 1+2 : encode cart ───────────────────────────────────────────
        cart_item_embs, basket_repr = self.encode_cart(
            cart_item_ids, cart_categories,
            cart_prices, cart_text_embs, cart_mask,
        )
        # cart_item_embs : (B, N, d_model)
        # basket_repr    : (B, d_model)

        # ── Stage 3 : encode candidate ────────────────────────────────────────
        candidate_repr = self.encode_candidate(
            cand_item_id, cand_category, cand_price, cand_text_emb
        )   # (B, d_model)

        # ── Stage 4 : encode user ─────────────────────────────────────────────
        user_repr = self.user_encoder(
            past_basket_embs, user_features, basket_mask
        )   # (B, d_model)

        # ── Stage 5 : encode context ──────────────────────────────────────────
        context_repr = self.context_encoder(
            hour          = hour,
            day_of_week   = day_of_week,
            month         = month,
            is_weekend    = is_weekend,
            meal_slot     = meal_slot,
            weather_type  = weather_type,
            temperature   = temperature,
            humidity      = humidity,
            festival_id   = festival_id,
            days_to_event = days_to_event,
            is_holiday    = is_holiday,
            restaurant_id = restaurant_id,
            cuisine_type  = cuisine_type,
            avg_prep_time = avg_prep_time,
            avg_rating    = avg_rating,
        )   # (B, d_model)

        # ── Stage 6 : DIN cross attention ─────────────────────────────────────
        cross_repr, attn_weights = self.din(
            candidate_emb = candidate_repr,
            cart_embs     = cart_item_embs,
            cart_mask     = cart_mask,
        )
        # cross_repr   : (B, d_model)
        # attn_weights : (B, N) or (B, n_heads, N)

        # ── Stage 7 : explicit cross features ─────────────────────────────────
        # Safe defaults for optional inputs
        N = cart_item_ids.size(1)
        cross_features = self.cross_feat_extractor(
            cart_has_main    = cart_has_main    if cart_has_main    is not None else _zeros(B),
            cart_has_side    = cart_has_side    if cart_has_side    is not None else _zeros(B),
            cart_has_drink   = cart_has_drink   if cart_has_drink   is not None else _zeros(B),
            cart_has_snack   = cart_has_snack   if cart_has_snack   is not None else _zeros(B),
            cart_has_dessert = cart_has_dessert if cart_has_dessert is not None else _zeros(B),
            cart_total       = cart_total       if cart_total       is not None else _ones(B) * 10.0,
            cart_size        = cart_size        if cart_size        is not None else _ones(B),
            cand_food_group  = cand_food_group,
            cand_price       = cand_price,
            cand_popularity  = cand_popularity,
            pmi_score        = pmi_score        if pmi_score        is not None else _zeros(B),
            co_occur_score   = co_occur_score   if co_occur_score   is not None else _zeros(B),
            novelty_score    = novelty_score    if novelty_score    is not None else _ones(B),
            cuisine_compat   = cuisine_compat   if cuisine_compat   is not None else _ones(B) * 0.5,
        )   # (B, n_cross_features)

        # ── Stage 8 : gated fusion ────────────────────────────────────────────
        fused_repr = self.gated_fusion(
            basket_repr    = basket_repr,
            cross_repr     = cross_repr,
            candidate_repr = candidate_repr,
            user_repr      = user_repr,
            context_repr   = context_repr,
        )   # (B, d_model)

        # ── Stage 9+10 : rank + coverage ──────────────────────────────────────
        biz = biz_features if biz_features is not None \
              else torch.zeros(B, 4, device=device)

        ranker_out = self.ranker(
            cart_repr           = basket_repr,
            cross_repr          = cross_repr,
            candidate_repr      = candidate_repr,
            user_repr           = user_repr,
            context_repr        = context_repr,
            fused_repr          = fused_repr,
            cross_features      = cross_features,
            biz_features        = biz,
            margin_score        = margin_score,
            novelty_score       = novelty_score,
            diversity_score     = diversity_score,
            return_logit        = return_logit,
            compute_final_score = compute_final_score,
        )

        # ── build output dict ─────────────────────────────────────────────────
        out = {
            'add_prob':        ranker_out['add_prob'],
            'coverage_logits': ranker_out['coverage_logits'],
            'attn_weights':    attn_weights,
            'basket_repr':     basket_repr,
            'fused_repr':      fused_repr,
        }
        if return_logit:
            out['add_logit'] = ranker_out['add_logit']
        if compute_final_score and 'final_score' in ranker_out:
            out['final_score'] = ranker_out['final_score']

        return out

    def compute_loss(
        self,
        pos_logits:      torch.Tensor,   # (B,)   logit for positive item
        neg_logits:      torch.Tensor,   # (B, K) logits for K negatives
        coverage_logits: torch.Tensor,   # (B, 5)
        cart_has_group:  torch.Tensor,   # (B, 5) 1=group present in cart
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience wrapper around CartCompleteLoss.
        Call this in training/train_ranker.py.

        Returns dict:
            loss          : total loss scalar
            loss_bpr      : BPR component
            loss_coverage : coverage BCE component
        """
        return self.loss_fn(
            pos_logits      = pos_logits,
            neg_logits      = neg_logits,
            coverage_logits = coverage_logits,
            cart_has_group  = cart_has_group,
        )

    @torch.no_grad()
    def score_candidates(
        self,
        candidate_ids:   torch.Tensor,   # (K,)  candidate item indices
        candidate_embs:  torch.Tensor,   # (K, d_in)
        basket_repr:     torch.Tensor,   # (1, d_model)  encoded cart
        cart_item_embs:  torch.Tensor,   # (1, N, d_model)
        user_repr:       torch.Tensor,   # (1, d_model)
        context_repr:    torch.Tensor,   # (1, d_model)
        cart_mask:       Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score K candidates efficiently at inference time.
        Expands cart/user/context to match K candidates.

        Returns scores : (K,) — use argsort to get top-K.
        Called by inference/ranking.py
        """
        self.eval()
        K = candidate_ids.size(0)

        # Expand single cart/user/context to K copies
        basket_exp  = basket_repr.expand(K, -1)         # (K, d_model)
        cart_exp    = cart_item_embs.expand(K, -1, -1)  # (K, N, d_model)
        user_exp    = user_repr.expand(K, -1)           # (K, d_model)
        ctx_exp     = context_repr.expand(K, -1)        # (K, d_model)
        mask_exp    = cart_mask.expand(K, -1) if cart_mask is not None else None

        # Encode candidates
        cand_repr, attn_weights = self.din(
            candidate_emb = candidate_embs,
            cart_embs     = cart_exp,
            cart_mask     = mask_exp,
        )

        fused = self.gated_fusion(
            basket_repr    = basket_exp,
            cross_repr     = cand_repr,
            candidate_repr = candidate_embs,
            user_repr      = user_exp,
            context_repr   = ctx_exp,
        )

        ranker_out = self.ranker(
            cart_repr      = basket_exp,
            cross_repr     = cand_repr,
            candidate_repr = candidate_embs,
            user_repr      = user_exp,
            context_repr   = ctx_exp,
            fused_repr     = fused,
            cross_features = torch.zeros(K, 8, device=basket_repr.device),
            biz_features   = torch.zeros(K, 4, device=basket_repr.device),
            return_logit   = True,
        )

        return ranker_out['add_logit'].squeeze(-1)   # (K,)


# ─────────────────────────────────────────────
# USAGE IN training/train_ranker.py
# ─────────────────────────────────────────────
#
# from models.addon_recsys import AddOnRecSys
#
# model = AddOnRecSys(
#     num_items        = 38000,
#     num_categories   = 5,
#     text_emb_dim     = 384,
#     d_model          = 128,
#     n_heads          = 4,
#     n_sab_layers     = 2,
#     n_sasrec_layers  = 2,
#     max_cart_len     = 50,
#     n_user_features  = 8,
#     num_restaurants  = 5000,
#     dropout          = 0.1,
# )
#
# out = model(**batch)
# losses = model.compute_loss(
#     pos_logits      = out['add_logit'].squeeze(-1)[pos_idx],
#     neg_logits      = neg_logits,
#     coverage_logits = out['coverage_logits'],
#     cart_has_group  = batch['cart_has_group'],
# )
# losses['loss'].backward()
# ─────────────────────────────────────────────


# ── sanity check ─────────────────────────────
if __name__ == "__main__":
    B, N, H = 4, 6, 10
    d_model  = 128
    text_dim = 384
    K_neg    = 4

    model = AddOnRecSys(
        num_items       = 1000,
        num_categories  = 5,
        text_emb_dim    = text_dim,
        d_model         = d_model,
        n_heads         = 4,
        n_sab_layers    = 2,
        n_sasrec_layers = 2,
        max_cart_len    = 50,
        n_user_features = 8,
        num_restaurants = 500,
        dropout         = 0.1,
    )

    out = model(
        # cart
        cart_item_ids   = torch.randint(1, 1000, (B, N)),
        cart_categories = torch.randint(0, 5,    (B, N)),
        cart_prices     = torch.rand(B, N) * 10,
        cart_text_embs  = torch.randn(B, N, text_dim),
        cart_mask       = torch.ones(B, N, dtype=torch.long),
        # candidate
        cand_item_id    = torch.randint(1, 1000, (B,)),
        cand_category   = torch.randint(0, 5,    (B,)),
        cand_price      = torch.rand(B) * 10,
        cand_text_emb   = torch.randn(B, text_dim),
        cand_food_group = torch.randint(0, 5,    (B,)),
        cand_popularity = torch.rand(B),
        # user
        past_basket_embs= torch.randn(B, H, d_model),
        user_features   = torch.randn(B, 8),
        basket_mask     = torch.ones(B, H, dtype=torch.long),
        # context
        hour            = torch.randint(0, 24, (B,)),
        day_of_week     = torch.randint(0, 7,  (B,)),
        return_logit    = True,
        compute_final_score = True,
        margin_score    = torch.rand(B),
        novelty_score   = torch.rand(B),
        diversity_score = torch.rand(B),
    )

    print(f"add_prob        : {out['add_prob'].shape}")
    print(f"coverage_logits : {out['coverage_logits'].shape}")
    print(f"attn_weights    : {out['attn_weights'].shape}")
    print(f"add_logit       : {out['add_logit'].shape}")
    print(f"final_score     : {out['final_score'].shape}")

    # Loss
    B2 = B // 2
    losses = model.compute_loss(
        pos_logits      = out['add_logit'].squeeze(-1)[:B2],
        neg_logits      = torch.randn(B2, K_neg),
        coverage_logits = out['coverage_logits'][:B2],
        cart_has_group  = torch.randint(0, 2, (B2, 5)).float(),
    )
    print(f"\nloss          : {losses['loss'].item():.4f}")
    print(f"loss_bpr      : {losses['loss_bpr'].item():.4f}")
    print(f"loss_coverage : {losses['loss_coverage'].item():.4f}")

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters : {total:,}")