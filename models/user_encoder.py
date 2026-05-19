
import torch
import torch.nn as nn
from typing import Optional


class UserEncoder(nn.Module):
    """
    Encodes user history into user_repr using:

        1. Mean pooling over past basket embeddings
           → captures long-term average preference

        2. Precomputed user feature vector
           → exploration score, category affinity,
             addon history rate, avg basket value,
             avg basket size

        3. Fusion of both signals → user_repr

    Why mean pooling and not GRU?
    ─────────────────────────────
    The current cart already tells us what the user
    wants RIGHT NOW. The user encoder only needs to
    tell us WHO this user is in general — their
    long-term food preferences, price sensitivity,
    and exploration tendency. Mean pooling over past
    baskets captures this cleanly without the risk
    of GRU overfitting on sparse histories
    (Instacart users have ~4-10 orders on average).
    """

    def __init__(
        self,
        d_model:       int,
        n_user_features: int  = 8,
        # user features from users_instacart.parquet:
        # exploration_score, affinity_main, affinity_side,
        # affinity_drink, affinity_snack, affinity_dessert,
        # addon_history_rate, avg_basket_value
        dropout:       float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Project raw user stats to d_model // 2
        user_feat_dim = d_model // 2
        self.user_feat_proj = nn.Sequential(
            nn.Linear(n_user_features, user_feat_dim),
            nn.LayerNorm(user_feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Basket history pooling projection
        # Mean-pooled basket embs are already d_model
        # Just normalize + dropout before fusion
        self.basket_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Fusion: basket_repr (d_model) + user_feat (d_model//2)
        fused_dim = d_model + user_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )


    def mean_pool_baskets(
        self,
        past_basket_embs: torch.Tensor,
        basket_mask:      Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Mean pool over H past basket embeddings.

        past_basket_embs : (B, H, d_model)
        basket_mask      : (B, H)  1=real  0=padding

        Returns : (B, d_model)
        """
        if basket_mask is not None:
            mask_f   = basket_mask.float().unsqueeze(-1)      # (B, H, 1)
            lengths  = mask_f.sum(dim=1).clamp(min=1)        # (B, 1)
            pooled   = (past_basket_embs * mask_f).sum(dim=1) / lengths
        else:
            pooled = past_basket_embs.mean(dim=1)

        return pooled   # (B, d_model)


    def forward(
        self,
        past_basket_embs:  torch.Tensor,
        user_features:     torch.Tensor,
        basket_mask:       Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ──────────
        past_basket_embs : (B, H, d_model)
            Embeddings of H most recent past baskets.
            Each basket embedding = mean of item embeddings
            in that basket, precomputed during preprocessing.
            H = max history length (e.g. 10), padded with zeros.

        user_features : (B, n_user_features)
            Precomputed per-user statistics loaded from
            users_instacart.parquet:
                [0] exploration_score      float 0-1
                [1] affinity_main          float 0-1
                [2] affinity_side          float 0-1
                [3] affinity_drink         float 0-1
                [4] affinity_snack         float 0-1
                [5] affinity_dessert       float 0-1
                [6] addon_history_rate     float 0-1
                [7] avg_basket_value       float (normalized)

        basket_mask : (B, H)  1=real basket  0=padding
            Pass None if all users have same history length.

        Returns
        ───────
        user_repr : (B, d_model)
        """

        # 1. Pool past basket embeddings → long-term preference
        basket_pooled = self.mean_pool_baskets(
            past_basket_embs, basket_mask
        )                                           # (B, d_model)
        basket_repr   = self.basket_proj(basket_pooled)

        # 2. Project user feature stats
        user_feat_repr = self.user_feat_proj(user_features)   # (B, d_model//2)

        # 3. Fuse both signals
        fused     = torch.cat([basket_repr, user_feat_repr], dim=-1)
        user_repr = self.fusion(fused)              # (B, d_model)

        return user_repr


class ColdStartUserEncoder(nn.Module):
    """
    Fallback encoder for new users with zero order history.

    When a user has no past baskets (new user cold start),
    we cannot mean pool — there is nothing to pool.
    Instead we encode only from user_features (demographic
    and contextual signals available at signup or inferred):

        user_features → MLP → user_repr

    In cartcomplete.py:
        if user has history → UserEncoder
        else               → ColdStartUserEncoder

    Both output (B, d_model) so the rest of the pipeline
    does not need to know which was used.
    """

    def __init__(
        self,
        d_model:         int,
        n_user_features: int   = 8,
        dropout:         float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_user_features, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, user_features: torch.Tensor) -> torch.Tensor:
        # user_features : (B, n_user_features)
        return self.encoder(user_features)   # (B, d_model)


class SmartUserEncoder(nn.Module):
    """
    Wrapper that automatically routes between
    UserEncoder and ColdStartUserEncoder per sample
    within the same batch.

    Handles mixed batches where some users have history
    and some do not — common in production systems.

    In practice on Instacart most users have history,
    so cold start is rare but must be handled cleanly.
    """

    def __init__(
        self,
        d_model:         int,
        n_user_features: int   = 8,
        dropout:         float = 0.1,
    ):
        super().__init__()

        self.warm_encoder = UserEncoder(
            d_model, n_user_features, dropout
        )
        self.cold_encoder = ColdStartUserEncoder(
            d_model, n_user_features, dropout
        )
        self.d_model = d_model

    def forward(
        self,
        past_basket_embs: torch.Tensor,
        user_features:    torch.Tensor,
        basket_mask:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ──────────
        past_basket_embs : (B, H, d_model)
        user_features    : (B, n_user_features)
        basket_mask      : (B, H)  1=real  0=padding

        Returns
        ───────
        user_repr : (B, d_model)
        """
        B = past_basket_embs.size(0)

        # Identify which users have at least one real past basket
        if basket_mask is not None:
            has_history = basket_mask.sum(dim=1) > 0   # (B,) bool
        else:
            has_history = torch.ones(B, dtype=torch.bool,
                                     device=past_basket_embs.device)

        user_repr = torch.zeros(
            B, self.d_model, device=past_basket_embs.device
        )

        # Warm users — have history
        warm_idx = has_history.nonzero(as_tuple=True)[0]
        if len(warm_idx) > 0:
            warm_repr = self.warm_encoder(
                past_basket_embs[warm_idx],
                user_features[warm_idx],
                basket_mask[warm_idx] if basket_mask is not None else None,
            )
            user_repr[warm_idx] = warm_repr

        # Cold users — no history
        cold_idx = (~has_history).nonzero(as_tuple=True)[0]
        if len(cold_idx) > 0:
            cold_repr = self.cold_encoder(user_features[cold_idx])
            user_repr[cold_idx] = cold_repr

        return user_repr   # (B, d_model)





# ── sanity check ─────────────────────────────
if __name__ == "__main__":
    B, H, d = 8, 10, 64
    n_feats  = 8

    encoder = SmartUserEncoder(d_model=d, n_user_features=n_feats)

    past_baskets = torch.randn(B, H, d)
    user_feats   = torch.randn(B, n_feats)

    # Mixed batch: first 6 users have history, last 2 are cold start
    mask = torch.ones(B, H, dtype=torch.long)
    mask[6:, :] = 0   # cold start users

    user_repr = encoder(past_baskets, user_feats, mask)
    print(f"user_repr shape : {user_repr.shape}")   # (8, 64)

    total = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters : {total:,}")