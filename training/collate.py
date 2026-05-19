import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any


# ── constants ─────────────────────────────────────────────────────────────────

CART_KEYS_1D = [
    'cart_item_idxs',    # (N,)  long
    'cart_food_groups',  # (N,)  long
    'cart_mask',         # (N,)  long
]

CART_KEYS_2D = [
    'cart_text_embs',    # (N, 384)  float
    'cart_prices',       # (N, 1)    float
    'cart_popularity',   # (N, 1)    float
]

ITEM_SCALAR_KEYS_LONG = [
    'item_idx',
    'food_group',
]

ITEM_SCALAR_KEYS_FLOAT = [
    'text_emb',      # (384,)
    'price',         # (1,)
    'popularity',    # (1,)
]


# ── retrieval collate ─────────────────────────────────────────────────────────

class RetrievalCollator:
    """
    Collator for RetrievalDataset batches.

    All cart tensors are already padded to max_cart_len inside
    the dataset — so collation is a simple stack, not a pad.

    Handles:
        cart_*          : (B, N, *)   already padded
        pos_*           : (B, *)      positive item features
        hour / dow      : (B,)        context scalars
    """

    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        batch : list of dicts from RetrievalDataset.__getitem__

        Returns
        -------
        collated dict — all tensors have batch dimension B prepended
        """
        collated = {}

        # ── cart tensors — already fixed length, just stack ───────────────────
        for key in CART_KEYS_1D + CART_KEYS_2D:
            if key in batch[0]:
                collated[key] = torch.stack(
                    [sample[key] for sample in batch], dim=0
                )

        # ── positive item features ────────────────────────────────────────────
        for key in ['pos_item_idx', 'pos_food_group']:
            if key in batch[0]:
                collated[key] = torch.stack(
                    [sample[key] for sample in batch], dim=0
                )

        for key in ['pos_text_emb', 'pos_price', 'pos_popularity']:
            if key in batch[0]:
                collated[key] = torch.stack(
                    [sample[key] for sample in batch], dim=0
                )

        # ── context ───────────────────────────────────────────────────────────
        for key in ['hour', 'dow']:
            if key in batch[0]:
                collated[key] = torch.stack(
                    [sample[key] for sample in batch], dim=0
                )

        return collated


# ── ranking collate ───────────────────────────────────────────────────────────

class RankingCollator:
    """
    Collator for RankingDataset batches.

    Handles everything RetrievalCollator does plus:
        neg_*           : (B, *)      negative item features
        pos_cross_feats : (B, 5)      explicit cross features — positive
        neg_cross_feats : (B, 5)      explicit cross features — negative
        meal_period     : (B,)        context
        restaurant_id   : (B,)        context

    Cart tensors are already padded inside the dataset.
    Cross features are already fixed size (5,) — just stack.
    """

    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        batch : list of dicts from RankingDataset.__getitem__

        Returns
        -------
        collated dict with all tensors batched
        """
        collated = {}

        # ── cart tensors ──────────────────────────────────────────────────────
        for key in CART_KEYS_1D + CART_KEYS_2D:
            if key in batch[0]:
                collated[key] = torch.stack(
                    [sample[key] for sample in batch], dim=0
                )

        # ── positive item ─────────────────────────────────────────────────────
        for key in [
            'pos_item_idx',
            'pos_food_group',
            'pos_text_emb',
            'pos_price',
            'pos_popularity',
            'pos_cross_feats',
        ]:
            if key in batch[0]:
                collated[key] = torch.stack(
                    [sample[key] for sample in batch], dim=0
                )

        # ── negative item ─────────────────────────────────────────────────────
        for key in [
            'neg_item_idx',
            'neg_food_group',
            'neg_text_emb',
            'neg_price',
            'neg_popularity',
            'neg_cross_feats',
        ]:
            if key in batch[0]:
                collated[key] = torch.stack(
                    [sample[key] for sample in batch], dim=0
                )

        # ── context ───────────────────────────────────────────────────────────
        for key in ['hour', 'dow', 'meal_period', 'restaurant_id']:
            if key in batch[0]:
                collated[key] = torch.stack(
                    [sample[key] for sample in batch], dim=0
                )

        return collated


# ── dynamic padding collator (variable length carts) ─────────────────────────

class DynamicPaddingCollator:
    """
    Alternative collator for variable-length cart sequences.

    Use this instead of RankingCollator / RetrievalCollator when
    you want to pad each batch to its OWN longest sequence rather
    than the global max_cart_len set in the dataset.

    Benefits:
        - Shorter batches run faster
        - Less wasted compute on padding tokens
        - Memory efficient for skewed cart size distributions

    Trade-off:
        - Slightly more complex collation logic
        - Cannot use fixed-shape assumptions downstream

    Usage:
        Pass mode='retrieval' or mode='ranking' at init.
    """

    def __init__(self, mode: str = 'ranking'):
        assert mode in ('retrieval', 'ranking'), \
            "mode must be 'retrieval' or 'ranking'"
        self.mode = mode

    def _pad_cart(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Dynamically pad cart tensors to the longest cart in this batch.
        Builds cart_mask from actual sequence lengths.
        """
        # Get actual cart lengths from cart_mask
        # cart_mask was built at dataset level with global max_cart_len
        # Here we re-derive the true length and re-pad to batch max
        cart_lengths = [
            int(sample['cart_mask'].sum().item())
            for sample in batch
        ]
        batch_max_len = max(cart_lengths)

        B = len(batch)
        text_dim = batch[0]['cart_text_embs'].shape[-1]

        # Allocate output tensors
        cart_item_idxs   = torch.zeros(B, batch_max_len, dtype=torch.long)
        cart_text_embs   = torch.zeros(B, batch_max_len, text_dim)
        cart_prices      = torch.zeros(B, batch_max_len, 1)
        cart_food_groups = torch.zeros(B, batch_max_len, dtype=torch.long)
        cart_popularity  = torch.zeros(B, batch_max_len, 1)
        cart_mask        = torch.zeros(B, batch_max_len, dtype=torch.long)

        for i, (sample, length) in enumerate(zip(batch, cart_lengths)):
            L = min(length, batch_max_len)
            cart_item_idxs[i,   :L] = sample['cart_item_idxs'][:L]
            cart_text_embs[i,   :L] = sample['cart_text_embs'][:L]
            cart_prices[i,      :L] = sample['cart_prices'][:L]
            cart_food_groups[i, :L] = sample['cart_food_groups'][:L]
            cart_popularity[i,  :L] = sample['cart_popularity'][:L]
            cart_mask[i,        :L] = 1

        return {
            'cart_item_idxs':   cart_item_idxs,
            'cart_text_embs':   cart_text_embs,
            'cart_prices':      cart_prices,
            'cart_food_groups': cart_food_groups,
            'cart_popularity':  cart_popularity,
            'cart_mask':        cart_mask,
        }

    def __call__(
        self,
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:

        collated = self._pad_cart(batch)

        # ── positive item ─────────────────────────────────────────────────────
        for key in [
            'pos_item_idx', 'pos_food_group',
            'pos_text_emb', 'pos_price',
            'pos_popularity', 'pos_cross_feats',
        ]:
            if key in batch[0]:
                collated[key] = torch.stack(
                    [s[key] for s in batch], dim=0
                )

        # ── negative item (ranking mode only) ─────────────────────────────────
        if self.mode == 'ranking':
            for key in [
                'neg_item_idx', 'neg_food_group',
                'neg_text_emb', 'neg_price',
                'neg_popularity', 'neg_cross_feats',
            ]:
                if key in batch[0]:
                    collated[key] = torch.stack(
                        [s[key] for s in batch], dim=0
                    )

        # ── context ───────────────────────────────────────────────────────────
        context_keys = ['hour', 'dow']
        if self.mode == 'ranking':
            context_keys += ['meal_period', 'restaurant_id']

        for key in context_keys:
            if key in batch[0]:
                collated[key] = torch.stack(
                    [s[key] for s in batch], dim=0
                )

        return collated


# ── collator factory ──────────────────────────────────────────────────────────

def get_collator(
    mode:           str  = 'ranking',
    dynamic_padding: bool = False,
) -> Any:
    """
    Factory function — returns the right collator based on config.

    Parameters
    ----------
    mode            : 'retrieval' or 'ranking'
    dynamic_padding : True  → DynamicPaddingCollator (memory efficient)
                      False → RankingCollator / RetrievalCollator (simpler)

    Usage in train scripts
    ----------------------
    from training.collate import get_collator

    collator = get_collator(mode='ranking', dynamic_padding=False)
    loader   = DataLoader(dataset, batch_size=256, collate_fn=collator)
    """
    if dynamic_padding:
        return DynamicPaddingCollator(mode=mode)

    if mode == 'retrieval':
        return RetrievalCollator()
    elif mode == 'ranking':
        return RankingCollator()
    else:
        raise ValueError(f"mode must be 'retrieval' or 'ranking', got {mode}")


# ── sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── mock batch to test collators without real data ────────────────────────
    def make_mock_ranking_sample(
        cart_len: int = 4,
        text_dim: int = 384,
        max_cart: int = 10,
    ) -> Dict[str, torch.Tensor]:
        N = max_cart
        mask = torch.zeros(N, dtype=torch.long)
        mask[:cart_len] = 1

        return {
            # cart
            'cart_item_idxs':   torch.randint(0, 1000, (N,)),
            'cart_text_embs':   torch.randn(N, text_dim),
            'cart_prices':      torch.rand(N, 1),
            'cart_food_groups': torch.randint(0, 5, (N,)),
            'cart_popularity':  torch.rand(N, 1),
            'cart_mask':        mask,
            # positive
            'pos_item_idx':    torch.tensor(42,  dtype=torch.long),
            'pos_text_emb':    torch.randn(text_dim),
            'pos_price':       torch.rand(1),
            'pos_food_group':  torch.tensor(2,   dtype=torch.long),
            'pos_popularity':  torch.rand(1),
            'pos_cross_feats': torch.rand(5),
            # negative
            'neg_item_idx':    torch.tensor(99,  dtype=torch.long),
            'neg_text_emb':    torch.randn(text_dim),
            'neg_price':       torch.rand(1),
            'neg_food_group':  torch.tensor(3,   dtype=torch.long),
            'neg_popularity':  torch.rand(1),
            'neg_cross_feats': torch.rand(5),
            # context
            'hour':          torch.tensor(14, dtype=torch.long),
            'dow':           torch.tensor(2,  dtype=torch.long),
            'meal_period':   torch.tensor(1,  dtype=torch.long),
            'restaurant_id': torch.tensor(7,  dtype=torch.long),
        }

    # Build a mock batch of 8 samples with variable cart lengths
    cart_lengths = [2, 4, 3, 6, 1, 5, 4, 3]
    mock_batch   = [
        make_mock_ranking_sample(cart_len=l)
        for l in cart_lengths
    ]

    # ── test RankingCollator ──────────────────────────────────────────────────
    ranking_collator = get_collator(mode='ranking', dynamic_padding=False)
    batch_fixed      = ranking_collator(mock_batch)

    print("RankingCollator output shapes:")
    for k, v in batch_fixed.items():
        print(f"  {k:<22} : {v.shape}")

    # ── test DynamicPaddingCollator ───────────────────────────────────────────
    dynamic_collator = get_collator(mode='ranking', dynamic_padding=True)
    batch_dynamic    = dynamic_collator(mock_batch)

    print("\nDynamicPaddingCollator output shapes:")
    for k, v in batch_dynamic.items():
        print(f"  {k:<22} : {v.shape}")

    print(f"\nFixed cart dim  : {batch_fixed['cart_mask'].shape[1]}")
    print(f"Dynamic cart dim: {batch_dynamic['cart_mask'].shape[1]}"
          f"  (= max cart len in batch = {max(cart_lengths)})")