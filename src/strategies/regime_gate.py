"""
src/strategies/regime_gate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cramér-Rao Selective Entry Filter

Theory
------
Cramér-Rao lower bound:
    Var(μ̂) ≥ σ²/T
    → Minimum variance of the drift estimator given T observations.
    → Only enter when the t-statistic exceeds the noise floor:
          t = |μ̂| · √T / σ  >  snr_min

This is the information-theoretic justification for selectivity:
when we cannot distinguish the drift from zero, the expected profit
is below the Cramér-Rao noise floor and entry is irrational.

Combined gate (AND of all conditions):
    1. Hurst H > hurst_min        — persistent (predictable) regime
    2. purity > purity_min        — Lindblad coherence (not in chaotic transition)
    3. |μ̂|√T / σ > snr_min       — drift exceeds Cramér-Rao noise floor

Relationship to win-rate
------------------------
Let p = P(TP hit before SL). For GBM with drift μ, volatility σ,
TP barrier b, SL barrier a (a < 0 < b), the first-passage probability:

    p = [exp(2μ·x₀/σ²) - exp(2μa/σ²)] / [exp(2μb/σ²) - exp(2μa/σ²)]

This is monotone in |μ/σ| (the Sharpe ratio per unit time).
The CR filter enforces that μ/σ is statistically above the noise floor,
which is a necessary condition for p > 0.5.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CramerRaoResult:
    """Diagnostic result from CramerRaoFilter.check()."""
    allow_entry: bool
    snr:         float   # t-statistic |μ̂|√T / σ
    hurst:       float   # Hurst exponent estimate
    purity:      float   # Lindblad purity Tr(ρ²)
    hurst_ok:    bool
    purity_ok:   bool
    snr_ok:      bool
    reason:      str     # "pass" or pipe-joined failure reasons


# ─────────────────────────────────────────────────────────────────────────────
# CramerRaoFilter
# ─────────────────────────────────────────────────────────────────────────────

class CramerRaoFilter:
    """
    Selective entry gate based on the Cramér-Rao lower bound and regime state.

    Gate passes iff ALL conditions hold:

        H > hurst_min
            Market is in a persistent (H > 0.5) or mildly persistent regime.
            H < hurst_min → mean-reverting noise, no predictable drift.

        purity > purity_min
            Lindblad decoherence purity Tr(ρ²) above threshold.
            Low purity → market in chaotic regime transition → abstain.

        |μ̂|·√T / σ > snr_min
            t-statistic of observed drift exceeds noise floor.
            This is directly the Cramér-Rao condition: signal > noise bound.

    Args:
        hurst_min   : minimum Hurst exponent (default 0.52)
        purity_min  : minimum Lindblad purity (default 0.50)
        snr_min     : minimum t-statistic     (default 0.80)
    """

    def __init__(
        self,
        hurst_min:  float = 0.52,
        purity_min: float = 0.50,
        snr_min:    float = 0.80,
    ) -> None:
        self.hurst_min  = hurst_min
        self.purity_min = purity_min
        self.snr_min    = snr_min

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def check(
        self,
        log_returns: np.ndarray,
        hurst:       float = 0.5,
        purity:      float = 1.0,
    ) -> CramerRaoResult:
        """
        Evaluate the Cramér-Rao gate for a potential entry.

        Args:
            log_returns : 1-D array of recent log-returns  [T]
            hurst       : Hurst exponent estimate (from HurstEstimator)
            purity      : Lindblad purity score Tr(ρ²) in [0, 1]

        Returns:
            CramerRaoResult — .allow_entry is True iff all conditions pass.
        """
        T = len(log_returns)

        # Insufficient history → abstain unconditionally
        if T < 10:
            return CramerRaoResult(
                allow_entry=False, snr=0.0, hurst=hurst, purity=purity,
                hurst_ok=False, purity_ok=False, snr_ok=False,
                reason="insufficient_data",
            )

        mu_hat    = float(np.mean(log_returns))
        # ddof=1: unbiased sample std — correct denominator for t-statistic
        # t = μ̂ / (s/√T),  s = std(ddof=1)
        # np.std(ddof=0) underestimates σ by √((T-1)/T), inflating SNR for small T
        sigma_hat = float(np.std(log_returns, ddof=1)) + 1e-8
        snr       = abs(mu_hat) * math.sqrt(T) / sigma_hat  # t-statistic

        # Note on Hurst interpretation:
        #   H > 0.55 → persistent (trending)  → momentum signal is reliable
        #   H < 0.45 → anti-persistent (mean-reverting) → also predictable but
        #              in the OPPOSITE direction. This filter blocks both H < hurst_min
        #              regimes intentionally: the swing strategy is trend-following and
        #              should not enter in pure mean-reversion regimes.
        #   0.45 < H < 0.55 → near-random-walk → CR noise floor dominates → abstain
        hurst_ok  = hurst  > self.hurst_min
        purity_ok = purity > self.purity_min
        snr_ok    = snr    > self.snr_min

        allow = hurst_ok and purity_ok and snr_ok

        if allow:
            reason = "pass"
        else:
            parts = []
            if not hurst_ok:
                parts.append(f"H={hurst:.3f}<{self.hurst_min}")
            if not purity_ok:
                parts.append(f"purity={purity:.3f}<{self.purity_min}")
            if not snr_ok:
                parts.append(f"snr={snr:.3f}<{self.snr_min}")
            reason = "|".join(parts)

        return CramerRaoResult(
            allow_entry=allow,
            snr=snr,
            hurst=hurst,
            purity=purity,
            hurst_ok=hurst_ok,
            purity_ok=purity_ok,
            snr_ok=snr_ok,
            reason=reason,
        )

    def to_dict(self) -> dict:
        """Return filter parameters as a plain dict (for TUI logging)."""
        return {
            "cr_hurst_min":  self.hurst_min,
            "cr_purity_min": self.purity_min,
            "cr_snr_min":    self.snr_min,
        }

    def __repr__(self) -> str:
        return (
            f"CramerRaoFilter("
            f"hurst_min={self.hurst_min}, "
            f"purity_min={self.purity_min}, "
            f"snr_min={self.snr_min})"
        )
