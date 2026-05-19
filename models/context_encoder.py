import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


# ── constants ─────────────────────────────────────────────────────────────────

HOURS_IN_DAY    = 24
DAYS_IN_WEEK    = 7
MONTHS_IN_YEAR  = 12


# ── positional / cyclic helpers ───────────────────────────────────────────────

class CyclicEncoding(nn.Module):
    """
    Encodes a cyclic integer feature (hour, weekday, month) as
    [sin(2π·x/period), cos(2π·x/period)] — preserves circular continuity.

    e.g. hour 23 and hour 0 are close, not far apart.
    """

    def __init__(self, period: int):
        super().__init__()
        self.period = period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B,) integer tensor in [0, period)
        Returns (B, 2)
        """
        angle = 2.0 * math.pi * x.float() / self.period
        return torch.stack([angle.sin(), angle.cos()], dim=-1)  # (B, 2)


class LearnedEmbedding(nn.Module):
    """Thin wrapper around nn.Embedding with LayerNorm."""

    def __init__(self, num_embeddings: int, d_out: int):
        super().__init__()
        self.emb  = nn.Embedding(num_embeddings, d_out, padding_idx=0)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.norm(self.emb(idx))


# ── branch encoders ───────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Encodes time-based signals:
        hour_of_day   → cyclic sin/cos  (2-dim)
        day_of_week   → cyclic sin/cos  (2-dim)
        month         → cyclic sin/cos  (2-dim)
        is_weekend    → scalar flag     (1-dim)
        meal_slot     → learned emb     (breakfast / lunch / dinner / late-night)

    Total raw dim → projected to d_out.
    """

    NUM_MEAL_SLOTS = 5  # 0=pad, 1=breakfast, 2=lunch, 3=dinner, 4=late-night

    def __init__(self, d_out: int, dropout: float = 0.1):
        super().__init__()

        self.hour_enc    = CyclicEncoding(HOURS_IN_DAY)
        self.day_enc     = CyclicEncoding(DAYS_IN_WEEK)
        self.month_enc   = CyclicEncoding(MONTHS_IN_YEAR)
        self.meal_emb    = LearnedEmbedding(self.NUM_MEAL_SLOTS, 8)

        raw_dim = 2 + 2 + 2 + 1 + 8   # 15

        self.proj = nn.Sequential(
            nn.Linear(raw_dim, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hour:       torch.Tensor,   # (B,) int  0-23
        day_of_week: torch.Tensor,  # (B,) int  0-6
        month:      torch.Tensor,   # (B,) int  0-11
        is_weekend: torch.Tensor,   # (B,) float 0/1
        meal_slot:  torch.Tensor,   # (B,) int  1-4
    ) -> torch.Tensor:

        h   = self.hour_enc(hour)           # (B, 2)
        d   = self.day_enc(day_of_week)     # (B, 2)
        mo  = self.month_enc(month)         # (B, 2)
        iw  = is_weekend.float().unsqueeze(-1)  # (B, 1)
        meal_slot = meal_slot.clamp(1, self.NUM_MEAL_SLOTS - 1)
        ms  = self.meal_emb(meal_slot)      # (B, 8)

        raw = torch.cat([h, d, mo, iw, ms], dim=-1)   # (B, 15)
        return self.proj(raw)                          # (B, d_out)


class WeatherEncoder(nn.Module):
    """
    Encodes weather context:
        weather_type  → learned embedding  (sunny/rainy/cloudy/snowy/…)
        temperature   → scalar (normalised)
        humidity      → scalar (normalised)

    Rainy weather → comfort food bias.
    Hot weather   → cold drinks / ice cream bias.
    """

    NUM_WEATHER_TYPES = 9  # 0=pad,1=sunny,2=cloudy,3=rainy,4=stormy,
                           # 5=snowy,6=foggy,7=windy,8=heatwave

    def __init__(self, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.weather_emb = LearnedEmbedding(self.NUM_WEATHER_TYPES, 16)

        raw_dim = 16 + 1 + 1   # 18

        self.proj = nn.Sequential(
            nn.Linear(raw_dim, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        weather_type: torch.Tensor,   # (B,) int
        temperature:  torch.Tensor,   # (B,) float  normalised ~N(0,1)
        humidity:     torch.Tensor,   # (B,) float  0-1
    ) -> torch.Tensor:

        we  = self.weather_emb(weather_type)        # (B, 16)
        tmp = temperature.float().unsqueeze(-1)     # (B, 1)
        hum = humidity.float().unsqueeze(-1)        # (B, 1)

        raw = torch.cat([we, tmp, hum], dim=-1)     # (B, 18)
        return self.proj(raw)                       # (B, d_out)


class FestivalEncoder(nn.Module):
    """
    Encodes special-day / festival context:
        festival_id   → learned embedding  (Diwali, Eid, Christmas, …)
        days_to_event → scalar  (negative = past, 0 = today, positive = upcoming)
        is_holiday    → binary flag

    Captures demand spikes around festivals & public holidays.
    """

    NUM_FESTIVALS = 32   # 0 = no festival / padding

    def __init__(self, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.festival_emb = LearnedEmbedding(self.NUM_FESTIVALS, 16)

        raw_dim = 16 + 1 + 1   # 18

        self.proj = nn.Sequential(
            nn.Linear(raw_dim, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        festival_id:   torch.Tensor,   # (B,) int
        days_to_event: torch.Tensor,   # (B,) float
        is_holiday:    torch.Tensor,   # (B,) float 0/1
    ) -> torch.Tensor:

        fe  = self.festival_emb(festival_id)           # (B, 16)
        dte = days_to_event.float().unsqueeze(-1)      # (B, 1)
        ih  = is_holiday.float().unsqueeze(-1)         # (B, 1)

        raw = torch.cat([fe, dte, ih], dim=-1)         # (B, 18)
        return self.proj(raw)                          # (B, d_out)


class RestaurantContextEncoder(nn.Module):
    """
    Encodes restaurant-level context:
        restaurant_id   → learned embedding
        cuisine_type    → learned embedding
        avg_prep_time   → scalar
        avg_rating      → scalar

    Keeps context_repr restaurant-aware so the ranker can
    surface items that make sense for this specific kitchen.
    """

    def __init__(
        self,
        num_restaurants: int,
        num_cuisine_types: int,
        d_out: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.restaurant_emb  = LearnedEmbedding(num_restaurants + 1, 32)
        self.cuisine_type_emb = LearnedEmbedding(num_cuisine_types + 1, 16)

        raw_dim = 32 + 16 + 1 + 1   # 50

        self.proj = nn.Sequential(
            nn.Linear(raw_dim, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        restaurant_id:  torch.Tensor,   # (B,) int
        cuisine_type:   torch.Tensor,   # (B,) int
        avg_prep_time:  torch.Tensor,   # (B,) float  normalised
        avg_rating:     torch.Tensor,   # (B,) float  0-5 normalised
    ) -> torch.Tensor:

        re  = self.restaurant_emb(restaurant_id)     # (B, 32)
        ct  = self.cuisine_type_emb(cuisine_type)    # (B, 16)
        apt = avg_prep_time.float().unsqueeze(-1)    # (B, 1)
        ar  = avg_rating.float().unsqueeze(-1)       # (B, 1)

        raw = torch.cat([re, ct, apt, ar], dim=-1)   # (B, 50)
        return self.proj(raw)                        # (B, d_out)


# ── gated context fusion ──────────────────────────────────────────────────────

class ContextGatedFusion(nn.Module):
    """
    Fuses temporal, weather, festival, and restaurant branches
    with learned soft gates — same idea as in user_encoder.py.

    Softmax gates ensure contributions sum to 1 and the model
    can learn to down-weight irrelevant signals (e.g. weather
    when indoors delivery is the norm).
    """

    def __init__(self, num_branches: int, d: int):
        super().__init__()
        self.gate_fc = nn.Linear(num_branches * d, num_branches)
        self.num_branches = num_branches
        self.d = d

    def forward(self, *branches: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(branches, dim=1)              # (B, K, d)
        concat  = stacked.view(stacked.size(0), -1)         # (B, K*d)
        gates   = torch.softmax(self.gate_fc(concat), dim=-1)  # (B, K)
        fused   = (gates.unsqueeze(-1) * stacked).sum(dim=1)   # (B, d)
        return fused


# ── main context encoder ──────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    """
    Encodes ALL session-level context signals into a single context_repr.

    Signals (all optional — zeroed out if absent)
    ─────────────────────────────────────────────
    Temporal   : hour_of_day, day_of_week, month, is_weekend, meal_slot
    Weather    : weather_type, temperature, humidity
    Festival   : festival_id, days_to_event, is_holiday
    Restaurant : restaurant_id, cuisine_type, avg_prep_time, avg_rating

                        ↓  per-branch MLP projection
                        ↓  ContextGatedFusion
                     context_repr  (B, d_model)

    The output feeds Stage 4 feature concatenation alongside
    cart_repr, user_repr, candidate_repr, and cross_repr.
    """

    def __init__(
        self,
        d_model:           int  = 128,
        num_restaurants:   int  = 10_000,
        num_cuisine_types: int  = 32,
        dropout:           float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # ── branch encoders ───────────────────────────────────────────────────
        self.temporal_enc    = TemporalEncoder(d_model, dropout)
        self.weather_enc     = WeatherEncoder(d_model, dropout)
        self.festival_enc    = FestivalEncoder(d_model, dropout)
        self.restaurant_enc  = RestaurantContextEncoder(
            num_restaurants, num_cuisine_types, d_model, dropout
        )

        # ── fusion ────────────────────────────────────────────────────────────
        self.fusion = ContextGatedFusion(num_branches=4, d=d_model)

        # ── output projection ─────────────────────────────────────────────────
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        # ── temporal ──────────────────────────────────────────────────────────
        hour:           Optional[torch.Tensor] = None,   # (B,) int 0-23
        day_of_week:    Optional[torch.Tensor] = None,   # (B,) int 0-6
        month:          Optional[torch.Tensor] = None,   # (B,) int 0-11
        is_weekend:     Optional[torch.Tensor] = None,   # (B,) float
        meal_slot:      Optional[torch.Tensor] = None,   # (B,) int 1-4
        # ── weather ───────────────────────────────────────────────────────────
        weather_type:   Optional[torch.Tensor] = None,   # (B,) int
        temperature:    Optional[torch.Tensor] = None,   # (B,) float
        humidity:       Optional[torch.Tensor] = None,   # (B,) float
        # ── festival ─────────────────────────────────────────────────────────
        festival_id:    Optional[torch.Tensor] = None,   # (B,) int
        days_to_event:  Optional[torch.Tensor] = None,   # (B,) float
        is_holiday:     Optional[torch.Tensor] = None,   # (B,) float
        # ── restaurant ───────────────────────────────────────────────────────
        restaurant_id:  Optional[torch.Tensor] = None,   # (B,) int
        cuisine_type:   Optional[torch.Tensor] = None,   # (B,) int
        avg_prep_time:  Optional[torch.Tensor] = None,   # (B,) float
        avg_rating:     Optional[torch.Tensor] = None,   # (B,) float
    ) -> torch.Tensor:
        """
        Returns
        -------
        context_repr : (B, d_model)
        """

        # Infer B and device from whatever is available
        ref = next(
            t for t in [
                hour, day_of_week, weather_type, festival_id, restaurant_id
            ] if t is not None
        )
        B      = ref.size(0)
        device = next(self.parameters()).device

        def _zeros() -> torch.Tensor:
            return torch.zeros(B, self.d_model, device=device)

        # ── temporal branch ───────────────────────────────────────────────────
        temporal_inputs = [hour, day_of_week, month, is_weekend, meal_slot]
        if all(t is not None for t in temporal_inputs):
            h_temporal = self.temporal_enc(
                hour, day_of_week, month, is_weekend, meal_slot
            )
        else:
            # Provide safe defaults for missing temporal fields
            hour        = hour        if hour        is not None else torch.zeros(B, dtype=torch.long, device=device)
            day_of_week = day_of_week if day_of_week is not None else torch.zeros(B, dtype=torch.long, device=device)
            month       = month       if month       is not None else torch.zeros(B, dtype=torch.long, device=device)
            is_weekend  = is_weekend  if is_weekend  is not None else torch.zeros(B, device=device)
            meal_slot   = meal_slot   if meal_slot   is not None else torch.ones(B, dtype=torch.long, device=device)
            h_temporal  = self.temporal_enc(hour, day_of_week, month, is_weekend, meal_slot)

        # ── weather branch ────────────────────────────────────────────────────
        weather_inputs = [weather_type, temperature, humidity]
        if all(t is not None for t in weather_inputs):
            h_weather = self.weather_enc(weather_type, temperature, humidity)
        else:
            h_weather = _zeros()

        # ── festival branch ───────────────────────────────────────────────────
        festival_inputs = [festival_id, days_to_event, is_holiday]
        if all(t is not None for t in festival_inputs):
            h_festival = self.festival_enc(festival_id, days_to_event, is_holiday)
        else:
            h_festival = _zeros()

        # ── restaurant branch ─────────────────────────────────────────────────
        restaurant_inputs = [restaurant_id, cuisine_type, avg_prep_time, avg_rating]
        if all(t is not None for t in restaurant_inputs):
            h_restaurant = self.restaurant_enc(
                restaurant_id, cuisine_type, avg_prep_time, avg_rating
            )
        else:
            h_restaurant = _zeros()

        # ── gated fusion ──────────────────────────────────────────────────────
        fused = self.fusion(h_temporal, h_weather, h_festival, h_restaurant)

        # ── final projection ──────────────────────────────────────────────────
        context_repr = self.output_proj(fused)
        return context_repr


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    B = 8

    model = ContextEncoder(
        d_model           = 128,
        num_restaurants   = 5000,
        num_cuisine_types = 32,
        dropout           = 0.1,
    )

    out = model(
        # temporal
        hour         = torch.randint(0, 24, (B,)),
        day_of_week  = torch.randint(0, 7,  (B,)),
        month        = torch.randint(0, 12, (B,)),
        is_weekend   = torch.randint(0, 2,  (B,)).float(),
        meal_slot    = torch.randint(1, 5,  (B,)),
        # weather
        weather_type = torch.randint(1, 9,  (B,)),
        temperature  = torch.randn(B),
        humidity     = torch.rand(B),
        # festival
        festival_id  = torch.randint(0, 32, (B,)),
        days_to_event= torch.randint(-3, 8, (B,)).float(),
        is_holiday   = torch.randint(0, 2,  (B,)).float(),
        # restaurant
        restaurant_id = torch.randint(1, 5001, (B,)),
        cuisine_type  = torch.randint(1, 33,   (B,)),
        avg_prep_time = torch.rand(B),
        avg_rating    = torch.rand(B),
    )
    print(f"context_repr shape (full): {out.shape}")   # → (8, 128)

    # Partial inputs — only temporal + restaurant available
    out_partial = model(
        hour          = torch.randint(0, 24,   (B,)),
        day_of_week   = torch.randint(0, 7,    (B,)),
        restaurant_id = torch.randint(1, 5001, (B,)),
        cuisine_type  = torch.randint(1, 33,   (B,)),
        avg_prep_time = torch.rand(B),
        avg_rating    = torch.rand(B),
    )
    print(f"context_repr shape (partial): {out_partial.shape}")  # → (8, 128)