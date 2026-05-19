
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# ── single signal gate ────────────────────────────────────────────────────────

class SignalGate(nn.Module):
    """
    Computes a soft gate vector for one input signal.

    Given the full concatenated context of all signals,
    learns a d_model-dimensional gate for this specific signal.
    Gate values ∈ (0, 1) per dimension — allows selective
    suppression of irrelevant feature dimensions, not just
    suppression of the whole signal at once.

    This is a feature-level gate, not a scalar gate.
    Strictly more expressive than a single scalar weight.
    """

    def __init__(self, d_model: int, n_signals: int, dropout: float = 0.1):
        super().__init__()
        # Input: all signals concatenated → (B, n_signals * d_model)
        # Output: gate vector → (B, d_model)
        self.gate_fc = nn.Sequential(
            nn.Linear(n_signals * d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),    # gate values in (0, 1)
        )

    def forward(self, all_signals_concat: torch.Tensor) -> torch.Tensor:
        # all_signals_concat : (B, n_signals * d_model)
        # Returns : (B, d_model) gate vector
        return self.gate_fc(all_signals_concat)


# ── expert projection per signal ──────────────────────────────────────────────

class SignalProjection(nn.Module):
    """
    Projects each input signal through its own dedicated
    transformation before gating.

    Each signal type (cart, cross, candidate, user, context)
    has different statistical properties and semantic meaning.
    Giving each its own projection lets the model normalise
    and transform each signal independently before fusion —
    similar to how MTGR uses Group-Layer Normalization (GLN)
    to normalize different token types separately.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)   # (B, d_model)


# ── main gated fusion ─────────────────────────────────────────────────────────

class GatedFusion(nn.Module):
    """
    Multi-signal gated fusion for CartComplete Stage 8.

    Architecture:
        ┌──────────────┐
        │ basket_repr  │──▶ SignalProjection ──▶ proj_basket ──┐
        │ cross_repr   │──▶ SignalProjection ──▶ proj_cross  ──┤
        │ candidate_r  │──▶ SignalProjection ──▶ proj_cand   ──┤──▶ concat ──▶ gate_context
        │ user_repr    │──▶ SignalProjection ──▶ proj_user   ──┤
        │ context_repr │──▶ SignalProjection ──▶ proj_ctx    ──┘
        └──────────────┘
                │
                ▼ gate_context : (B, 5 * d_model)
                │
                ├──▶ SignalGate(basket)    ──▶ gate_b  × proj_basket ──┐
                ├──▶ SignalGate(cross)     ──▶ gate_c  × proj_cross  ──┤
                ├──▶ SignalGate(candidate) ──▶ gate_ca × proj_cand   ──┼──▶ sum ──▶ fusion MLP ──▶ fused_repr
                ├──▶ SignalGate(user)      ──▶ gate_u  × proj_user   ──┤
                └──▶ SignalGate(context)   ──▶ gate_x  × proj_ctx    ──┘

    Each gate is a (B, d_model) vector — not a scalar.
    Allows per-dimension selective suppression.

    Final weighted sum + residual of original signals
    is passed through a fusion MLP → fused_repr.
    """

    SIGNAL_NAMES = ['basket', 'cross', 'candidate', 'user', 'context']

    def __init__(
        self,
        d_model:   int,
        dropout:   float = 0.1,
    ):
        super().__init__()

        self.d_model   = d_model
        self.n_signals = len(self.SIGNAL_NAMES)

        # ── per-signal projections (MTGR GLN-inspired) ───────────────────────
        self.projections = nn.ModuleDict({
            name: SignalProjection(d_model, dropout)
            for name in self.SIGNAL_NAMES
        })

        # ── per-signal gates ─────────────────────────────────────────────────
        self.gates = nn.ModuleDict({
            name: SignalGate(d_model, self.n_signals, dropout)
            for name in self.SIGNAL_NAMES
        })

        # ── fusion MLP after gated sum ────────────────────────────────────────
        # Input: gated_sum (d_model) + residual mean of all signals (d_model)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        basket_repr:    torch.Tensor,   # (B, d_model) — cart encoding
        cross_repr:     torch.Tensor,   # (B, d_model) — DIN cross attention
        candidate_repr: torch.Tensor,   # (B, d_model) — item encoder
        user_repr:      torch.Tensor,   # (B, d_model) — user encoder
        context_repr:   torch.Tensor,   # (B, d_model) — context encoder
    ) -> torch.Tensor:
        """
        Parameters
        ──────────
        basket_repr    : (B, d_model)  from SetTransformerEncoder
        cross_repr     : (B, d_model)  from DINAttention
        candidate_repr : (B, d_model)  from ItemEncoder
        user_repr      : (B, d_model)  from SmartUserEncoder
        context_repr   : (B, d_model)  from ContextEncoder

        Returns
        ───────
        fused_repr : (B, d_model)
            Single fused vector fed into DCNV2Ranker Stage 9.
        """

        signals = {
            'basket':    basket_repr,
            'cross':     cross_repr,
            'candidate': candidate_repr,
            'user':      user_repr,
            'context':   context_repr,
        }

        # ── step 1: project each signal independently ─────────────────────────
        projected = {
            name: self.projections[name](sig)
            for name, sig in signals.items()
        }   # each: (B, d_model)

        # ── step 2: build gate context — concat all projected signals ─────────
        gate_context = torch.cat(
            [projected[name] for name in self.SIGNAL_NAMES], dim=-1
        )   # (B, n_signals * d_model)

        # ── step 3: compute per-signal feature-level gates ────────────────────
        gated_signals = {}
        for name in self.SIGNAL_NAMES:
            gate_vec = self.gates[name](gate_context)     # (B, d_model) ∈ (0,1)
            gated_signals[name] = gate_vec * projected[name]

        # ── step 4: sum gated signals ─────────────────────────────────────────
        gated_sum = sum(gated_signals.values())           # (B, d_model)

        # ── step 5: residual — unweighted mean of all original signals ────────
        # Prevents information loss if gates are too aggressive
        residual = torch.stack(
            [signals[name] for name in self.SIGNAL_NAMES], dim=1
        ).mean(dim=1)                                     # (B, d_model)

        # ── step 6: fusion MLP over [gated_sum ; residual] ───────────────────
        fused_repr = self.fusion_mlp(
            torch.cat([gated_sum, residual], dim=-1)
        )                                                 # (B, d_model)

        return fused_repr


# ── gate inspection utility ───────────────────────────────────────────────────

class GatedFusionWithInspection(GatedFusion):
    """
    Extended version of GatedFusion that also returns
    per-signal gate activations for analysis.

    Use during notebooks/ablation_study.ipynb to visualise
    which signals the model relies on most in different contexts.

    Example insight:
        late-night orders  → context gate activation ↑
        repeat users       → user gate activation ↑
        cold-start users   → candidate gate activation ↑
    """

    def forward_with_gates(
        self,
        basket_repr:    torch.Tensor,
        cross_repr:     torch.Tensor,
        candidate_repr: torch.Tensor,
        user_repr:      torch.Tensor,
        context_repr:   torch.Tensor,
    ):
        """
        Returns
        ───────
        fused_repr   : (B, d_model)
        gate_summary : dict  signal_name → mean gate activation (scalar)
                       Use for logging and analysis.
        """
        signals = {
            'basket':    basket_repr,
            'cross':     cross_repr,
            'candidate': candidate_repr,
            'user':      user_repr,
            'context':   context_repr,
        }

        projected = {
            name: self.projections[name](sig)
            for name, sig in signals.items()
        }

        gate_context = torch.cat(
            [projected[name] for name in self.SIGNAL_NAMES], dim=-1
        )

        gated_signals = {}
        gate_activations = {}
        for name in self.SIGNAL_NAMES:
            gate_vec = self.gates[name](gate_context)
            gated_signals[name]    = gate_vec * projected[name]
            gate_activations[name] = gate_vec.mean().item()

        gated_sum = sum(gated_signals.values())

        residual = torch.stack(
            [signals[name] for name in self.SIGNAL_NAMES], dim=1
        ).mean(dim=1)

        fused_repr = self.fusion_mlp(
            torch.cat([gated_sum, residual], dim=-1)
        )

        return fused_repr, gate_activations





# ── sanity check ─────────────────────────────
if __name__ == "__main__":
    B, d = 8, 128

    model = GatedFusion(d_model=d, dropout=0.1)

    basket    = torch.randn(B, d)
    cross     = torch.randn(B, d)
    candidate = torch.randn(B, d)
    user      = torch.randn(B, d)
    context   = torch.randn(B, d)

    fused = model(basket, cross, candidate, user, context)
    print(f"fused_repr shape : {fused.shape}")   # (8, 128)

    # Inspection version
    inspector = GatedFusionWithInspection(d_model=d, dropout=0.1)
    fused2, gate_vals = inspector.forward_with_gates(
        basket, cross, candidate, user, context
    )
    print(f"fused_repr shape (inspection) : {fused2.shape}")
    print("Gate activations per signal:")
    for name, val in gate_vals.items():
        print(f"  {name:<12} : {val:.4f}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total:,}")