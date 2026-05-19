

import torch
import torch.nn as nn


class ItemEncoder(nn.Module):

    def __init__(
        self,
        num_items,          # total number of unique products
        num_categories,     # number of food groups (5: main/side/drink/snack/dessert)
        text_emb_dim,       # MiniLM output dim = 384
        d_model     = 64,   # output dimension for all embeddings
        price_bins  = 20,   # number of price buckets
        dropout     = 0.1,
    ):
        super().__init__()

        # Item ID embedding — learns collaborative signal
        self.item_emb = nn.Embedding(
            num_items + 1, d_model, padding_idx=0
        )

        # Food group embedding — main/side/drink/snack/dessert
        self.category_emb = nn.Embedding(
            num_categories + 1, d_model // 4, padding_idx=0
        )

        # Price is a continuous float — bucket it then embed
        # Buckets: 0.5–2, 2–4, 4–6, ... up to price_bins
        self.price_emb = nn.Embedding(
            price_bins + 1, d_model // 4
        )

        # Project MiniLM 384-dim text vector down to d_model
        self.text_proj = nn.Sequential(
            nn.Linear(text_emb_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final fusion: concat all four → project to d_model
        # Sizes: d_model + d_model + d_model//4 + d_model//4
        fused_dim = d_model + d_model + d_model // 4 + d_model // 4
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.price_bins = price_bins
        self.d_model    = d_model


    def bucketize_price(self, price):
        # price: (B,) float tensor
        # Returns (B,) int tensor of bucket indices
        # Bucket boundaries: evenly spaced 0 to 20
        boundaries = torch.linspace(0, 20, self.price_bins - 1).to(price.device)
        return torch.bucketize(price, boundaries).clamp(0, self.price_bins)


    def forward(self, item_id, category, price, text_emb):
        # item_id  : (B,)
        # category : (B,)
        # price    : (B,) float
        # text_emb : (B, 384)

        id_vec    = self.item_emb(item_id)                  # (B, d_model)
        cat_vec   = self.category_emb(category)             # (B, d_model//4)
        price_idx = self.bucketize_price(price)             # (B,)
        price_vec = self.price_emb(price_idx)               # (B, d_model//4)
        text_vec  = self.text_proj(text_emb)                # (B, d_model)

        fused = torch.cat(
            [id_vec, text_vec, cat_vec, price_vec], dim=-1  # (B, fused_dim)
        )
        return self.fusion(fused)                           # (B, d_model)