import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ── BPR ranking loss ──────────────────────────────────────────────────────────

class BPRLoss(nn.Module):
    """
    Bayesian Personalised Ranking Loss.

    Core idea:
        The score of a positive item (actually added to cart)
        should be higher than the score of a negative item
        (not added) by a safe margin.

        L_BPR = -log(sigmoid(score_pos - score_neg))

    Negative types in our system (from ranking_dataset.py):
        hard negatives      — same category, never added
        same-menu negatives — same restaurant, not added
        random negatives    — random catalog items

    Why BPR over BCE?
        BCE treats each item independently.
        BPR directly optimises the RELATIVE ordering of items
        which is exactly what a ranker needs — we care whether
        the positive item ranks above the negative, not the
        absolute probability values.

    Margin variant:
        L_BPR = -log(sigmoid(score_pos - score_neg - margin))
        Adds a safety margin so the model doesn't just barely
        separate positives from negatives.
    """

    def __init__(self, margin: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.margin    = margin
        self.reduction = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,   # (B,) or (B, 1)
        neg_scores: torch.Tensor,   # (B,) or (B, 1)
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pos_scores : raw logits for positive items  (before sigmoid)
        neg_scores : raw logits for negative items  (before sigmoid)

        Returns
        -------
        loss : scalar  (or (B,) if reduction='none')
        """
        pos_scores = pos_scores.squeeze(-1)   # (B,)
        neg_scores = neg_scores.squeeze(-1)   # (B,)

        # Score difference — positive should be higher
        diff = pos_scores - neg_scores - self.margin   # (B,)

        # BPR loss = -log(sigmoid(diff))
        # Numerically stable via log_sigmoid
        loss = -F.logsigmoid(diff)                     # (B,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss   # (B,)


# ── hard negative BPR loss ────────────────────────────────────────────────────

class HardNegativeBPRLoss(nn.Module):
    """
    BPR Loss with hard negative mining inside the batch.

    Standard BPR uses pre-sampled negatives from the dataset.
    This variant additionally mines the HARDEST negatives
    from within the current batch:

        For each sample i, the hard negative is the negative
        item j in the batch with the HIGHEST score — i.e. the
        one the model is most confused about.

    Loss is a weighted combination:
        L = alpha * L_BPR(sampled_neg) + (1-alpha) * L_BPR(hard_neg)

    Hard negatives accelerate training on ambiguous items
    like: "burger" when the positive is "cola" but "fries"
    also scores very high — exactly where the model needs
    to learn finer distinctions.
    """

    def __init__(
        self,
        margin:    float = 0.0,
        alpha:     float = 0.5,    # weight for sampled negatives
        reduction: str   = 'mean',
    ):
        super().__init__()
        self.bpr       = BPRLoss(margin=margin, reduction=reduction)
        self.alpha     = alpha
        self.reduction = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,   # (B,)
        neg_scores: torch.Tensor,   # (B,)  pre-sampled negatives
        all_scores: Optional[torch.Tensor] = None,  # (B,)  all candidate scores
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total_loss    : scalar
        hard_neg_loss : scalar  (for logging)
        """
        # Standard BPR with sampled negatives
        sampled_loss = self.bpr(pos_scores, neg_scores)

        # Hard negative mining within batch
        if all_scores is not None:
            B = pos_scores.size(0)
            # Mask out positions where the item IS the positive
            # by cloning and filling diagonal with -inf
            score_matrix = all_scores.unsqueeze(0).expand(B, -1).clone()
            score_matrix.fill_diagonal_(-1e9)

            # Hardest negative per sample = highest scoring non-positive
            hard_neg_scores, _ = score_matrix.max(dim=1)   # (B,)
            hard_neg_loss = self.bpr(pos_scores, hard_neg_scores)
        else:
            hard_neg_loss = torch.tensor(0.0, device=pos_scores.device)

        total = (
            self.alpha * sampled_loss +
            (1 - self.alpha) * hard_neg_loss
        )

        return total, hard_neg_loss


# ── coverage loss ─────────────────────────────────────────────────────────────

class CoverageLoss(nn.Module):
    """
    Coverage / Diversity Loss.

    Encourages the model to recommend items from food groups
    that are MISSING from the current cart — improving diversity
    and completing the meal rather than just reinforcing what
    is already there.

    Example:
        Cart   = [burger, fries]           → main + side present
        Missing = drink, dessert, snack
        Coverage loss pushes the model to score drink/dessert/snack
        candidates higher

    Implementation:
        For each sample, compute which food groups are absent
        from the cart. The coverage loss penalises the model
        when the positive item's food group IS already in the
        cart (low coverage) and rewards when it is missing
        (high coverage).

        L_cov = BCE(P(missing_group | cart), is_missing_label)

    This acts as a soft diversity regulariser — the model
    learns to balance between highly compatible items AND
    items that fill gaps in the basket.
    """

    N_FOOD_GROUPS = 5   # main, side, drink, snack, dessert

    def __init__(
        self,
        n_food_groups: int   = 5,
        reduction:     str   = 'mean',
    ):
        super().__init__()
        self.n_food_groups = n_food_groups
        self.reduction     = reduction

    def _get_missing_groups(
        self,
        cart_food_groups: torch.Tensor,   # (B, N)  long
        cart_mask:        torch.Tensor,   # (B, N)  long  1=real 0=pad
    ) -> torch.Tensor:
        """
        For each sample in the batch compute a binary vector
        of length n_food_groups indicating which groups are
        ABSENT from the cart.

        Returns: (B, n_food_groups)  float  1=missing 0=present
        """
        B = cart_food_groups.size(0)
        present = torch.zeros(
            B, self.n_food_groups,
            device=cart_food_groups.device
        )

        # Only consider real (non-padding) cart items
        for g in range(self.n_food_groups):
            group_mask = (cart_food_groups == g).float()   # (B, N)
            group_mask = group_mask * cart_mask.float()    # zero out padding
            present[:, g] = (group_mask.sum(dim=1) > 0).float()

        missing = 1.0 - present   # (B, n_food_groups)
        return missing

    def forward(
        self,
        pos_food_groups:  torch.Tensor,   # (B,)    long  food group of positive
        cart_food_groups: torch.Tensor,   # (B, N)  long  food groups in cart
        cart_mask:        torch.Tensor,   # (B, N)  long
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pos_food_groups  : food group index of the positive candidate
        cart_food_groups : food group indices of cart items
        cart_mask        : 1=real cart item, 0=padding

        Returns
        -------
        coverage_loss : scalar
        """
        B = pos_food_groups.size(0)

        # Which groups are missing from each cart?  (B, n_food_groups)
        missing = self._get_missing_groups(cart_food_groups, cart_mask)

        # One-hot encode the positive item's food group
        pos_onehot = F.one_hot(
            pos_food_groups, num_classes=self.n_food_groups
        ).float()   # (B, n_food_groups)

        # Coverage target — 1 if positive fills a missing group
        # This is element-wise: did we recommend from a missing group?
        coverage_target = (pos_onehot * missing).sum(dim=1)   # (B,)
        coverage_target = coverage_target.clamp(0, 1)

        # We want coverage_target to be high — so we maximise it
        # Equivalently minimise (1 - coverage_target)
        loss = 1.0 - coverage_target   # (B,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ── InfoNCE contrastive loss (retrieval training) ─────────────────────────────

class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent Contrastive Loss for Two-Tower retrieval training.

    For a batch of (cart, positive_item) pairs, treats all OTHER
    positive items in the batch as negatives (in-batch negatives).

    L_InfoNCE = -log(
        exp(sim(cart_i, item_i) / τ)
        ─────────────────────────────────────────────
        Σ_j exp(sim(cart_i, item_j) / τ)
    )

    Where:
        sim = cosine similarity
        τ   = temperature (lower = sharper distribution)

    Why in-batch negatives work well here:
        With batch size 512, each cart has 511 negatives —
        many of which are genuinely hard (other food items
        from the same restaurant) — without any extra sampling.

    Symmetrised version:
        Also computes loss from item→cart direction and averages.
        This doubles the effective training signal per batch.
    """

    def __init__(self, temperature: float = 0.07, symmetrise: bool = True):
        super().__init__()
        self.temperature  = temperature
        self.symmetrise   = symmetrise

    def forward(
        self,
        cart_embs: torch.Tensor,   # (B, d)  L2-normalised cart vectors
        item_embs: torch.Tensor,   # (B, d)  L2-normalised item vectors
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cart_embs : L2-normalised cart representations from cart tower
        item_embs : L2-normalised item representations from item tower
        Both must be normalised before passing in.

        Returns
        -------
        loss : scalar
        """
        B = cart_embs.size(0)

        # Cosine similarity matrix  (B, B)
        # Since both are L2-normalised, dot product = cosine sim
        sim_matrix = torch.matmul(
            cart_embs, item_embs.T
        ) / self.temperature   # (B, B)

        # Labels — diagonal is the positive pair
        labels = torch.arange(B, device=cart_embs.device)

        # Cart → Item direction
        loss_c2i = F.cross_entropy(sim_matrix, labels)

        if self.symmetrise:
            # Item → Cart direction
            loss_i2c = F.cross_entropy(sim_matrix.T, labels)
            return (loss_c2i + loss_i2c) / 2.0

        return loss_c2i


# ── combined CartComplete training loss ───────────────────────────────────────

class CartCompleteLoss(nn.Module):
    """
    Combined training loss for the full CartComplete ranker.

    Final Loss (from architecture doc):
        L = L_BPR + λ_cov × L_coverage

    Where:
        L_BPR       — pairwise ranking loss (positive > negative)
        L_coverage  — diversity regulariser (fill missing food groups)
        λ_cov       — coverage weight (default 0.1 from architecture doc)

    Optionally includes hard negative mining:
        L = L_BPR_hard + λ_cov × L_coverage

    All individual loss values are returned for logging/monitoring.
    """

    def __init__(
        self,
        lambda_coverage:     float = 0.1,    # from architecture doc
        bpr_margin:          float = 0.0,
        use_hard_negatives:  bool  = True,
        hard_neg_alpha:      float = 0.5,
        n_food_groups:       int   = 5,
    ):
        super().__init__()

        self.lambda_coverage = lambda_coverage

        if use_hard_negatives:
            self.bpr_loss = HardNegativeBPRLoss(
                margin    = bpr_margin,
                alpha     = hard_neg_alpha,
                reduction = 'mean',
            )
        else:
            self.bpr_loss = BPRLoss(
                margin    = bpr_margin,
                reduction = 'mean',
            )

        self.coverage_loss = CoverageLoss(
            n_food_groups = n_food_groups,
            reduction     = 'mean',
        )

        self.use_hard_negatives = use_hard_negatives

    def forward(
        self,
        pos_scores:       torch.Tensor,            # (B,)  ranker logits positive
        neg_scores:       torch.Tensor,            # (B,)  ranker logits negative
        pos_food_groups:  torch.Tensor,            # (B,)  food group of positive
        cart_food_groups: torch.Tensor,            # (B,N) food groups in cart
        cart_mask:        torch.Tensor,            # (B,N) padding mask
        all_scores:       Optional[torch.Tensor] = None,  # (B,) for hard mining
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss : scalar — backprop through this
        loss_dict  : dict   — individual components for logging
        """

        # ── BPR loss ──────────────────────────────────────────────────────────
        if self.use_hard_negatives:
            bpr, hard_neg_loss = self.bpr_loss(
                pos_scores, neg_scores, all_scores
            )
        else:
            bpr = self.bpr_loss(pos_scores, neg_scores)
            hard_neg_loss = torch.tensor(0.0, device=pos_scores.device)

        # ── coverage loss ─────────────────────────────────────────────────────
        cov = self.coverage_loss(
            pos_food_groups, cart_food_groups, cart_mask
        )

        # ── total loss ────────────────────────────────────────────────────────
        total = bpr + self.lambda_coverage * cov

        loss_dict = {
            'total':        total.item(),
            'bpr':          bpr.item(),
            'coverage':     cov.item(),
            'hard_neg':     hard_neg_loss.item(),
            'lambda_cov':   self.lambda_coverage,
        }

        return total, loss_dict


# ── sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B = 16
    N = 10   # max cart len

    # ── BPR Loss ──────────────────────────────────────────────────────────────
    bpr = BPRLoss(margin=0.1)
    pos = torch.randn(B)
    neg = torch.randn(B)
    print(f"BPR Loss          : {bpr(pos, neg).item():.4f}")

    # ── Coverage Loss ─────────────────────────────────────────────────────────
    cov_loss       = CoverageLoss(n_food_groups=5)
    pos_fg         = torch.randint(0, 5, (B,))
    cart_fg        = torch.randint(0, 5, (B, N))
    cart_mask      = torch.ones(B, N, dtype=torch.long)
    cart_mask[:, -3:] = 0

    print(f"Coverage Loss     : {cov_loss(pos_fg, cart_fg, cart_mask).item():.4f}")

    # ── InfoNCE Loss ──────────────────────────────────────────────────────────
    infonce    = InfoNCELoss(temperature=0.07)
    cart_embs  = F.normalize(torch.randn(B, 128), dim=-1)
    item_embs  = F.normalize(torch.randn(B, 128), dim=-1)
    print(f"InfoNCE Loss      : {infonce(cart_embs, item_embs).item():.4f}")

    # ── CartComplete Combined Loss ─────────────────────────────────────────────
    cc_loss = CartCompleteLoss(
        lambda_coverage    = 0.1,
        bpr_margin         = 0.0,
        use_hard_negatives = True,
        hard_neg_alpha     = 0.5,
        n_food_groups      = 5,
    )

    total, loss_dict = cc_loss(
        pos_scores       = pos,
        neg_scores       = neg,
        pos_food_groups  = pos_fg,
        cart_food_groups = cart_fg,
        cart_mask        = cart_mask,
        all_scores       = torch.randn(B),
    )

    print(f"\nCartComplete Loss breakdown:")
    for k, v in loss_dict.items():
        print(f"  {k:<15} : {v:.4f}")