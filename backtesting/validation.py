"""
Two-gate validation framework for strategy alpha verification.

Gate 1: Engineering sanity check (win rate > BEP).
Gate 2: Bootstrap hypothesis test — p < 0.05 on mean R-multiple > 0,
        minimum 250 trades.

Breakeven derivation (from configs):
  TP = 3.0 * ATR,  SL = 1.0 * ATR
  BEP = SL / (SL + TP) = 1 / (1 + 3) = 25.0%
  With Bybit fees (round-trip 0.075% * eff_leverage 5x = 0.375%/trade):
    BEP_adj ~ 25.4%
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BREAKEVEN_RATE = 0.254  # 25.4% — fee-adjusted breakeven win rate

# ---------------------------------------------------------------------------
# Gate 1 — Engineering sanity check (Schwertz)
# ---------------------------------------------------------------------------


def gate_1_system_check(wins: int, total_trades: int) -> bool:
    """Return True (PASS) only if raw win rate > breakeven.

    Args:
        wins: Number of winning trades.
        total_trades: Total number of trades executed.

    Returns:
        True if win_rate > BREAKEVEN_RATE, False otherwise.
        Returns False if total_trades == 0 (no data = no pass).
    """
    if total_trades <= 0:
        return False
    win_rate = wins / total_trades
    return win_rate > BREAKEVEN_RATE


# ---------------------------------------------------------------------------
# Gate 2 — Bootstrap alpha validation (Viktor's spec)
# ---------------------------------------------------------------------------

MIN_TRADES = 250
DEFAULT_N_BOOTSTRAP = 10_000


@dataclass
class BootstrapResult:
    """Container for bootstrap hypothesis test output."""
    p_value: float
    observed_mean: float
    n_trades: int
    n_bootstrap: int
    ci_lower: float   # 95% CI lower bound (percentile method)
    ci_upper: float   # 95% CI upper bound
    passed: bool       # p < alpha AND n >= MIN_TRADES


def gate_2_bootstrap_validation(
    r_multiples: list[float] | np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = 0.05,
    min_trades: int = MIN_TRADES,
    seed: int | None = 42,
) -> BootstrapResult:
    """Non-parametric bootstrap test: H0: E[R] <= 0  vs  H1: E[R] > 0.

    Method
    ------
    1. Compute observed sample mean x_bar.
    2. Center the data under H0: r_centered = r - x_bar  (so mean = 0).
    3. Draw n_bootstrap resamples (with replacement) from r_centered,
       compute each resample mean.
    4. p-value = fraction of bootstrap means >= x_bar.
       (How often does a mean-zero world produce a mean as extreme
       as what we observed?)

    This is the standard shift-based bootstrap test for one-sided
    hypotheses on the mean (Efron & Tibshirani, 1993, Ch. 16).

    Parameters
    ----------
    r_multiples : array-like
        Realized R-multiples per trade (e.g., +2.1, -1.0, +0.5).
    n_bootstrap : int
        Number of bootstrap resamples (default 10,000).
    alpha : float
        Significance level (default 0.05).
    min_trades : int
        Minimum trade count to run the test (default 250).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    BootstrapResult
        p_value, observed_mean, CI, and pass/fail verdict.
    """
    r = np.asarray(r_multiples, dtype=np.float64)
    n = len(r)

    # Insufficient data → automatic fail
    if n < min_trades:
        return BootstrapResult(
            p_value=1.0,
            observed_mean=float(np.mean(r)) if n > 0 else 0.0,
            n_trades=n,
            n_bootstrap=n_bootstrap,
            ci_lower=np.nan,
            ci_upper=np.nan,
            passed=False,
        )

    rng = np.random.default_rng(seed)
    observed_mean = float(np.mean(r))

    # --- Center under H0: shift data so mean = 0 ---
    r_centered = r - observed_mean

    # --- Vectorized bootstrap resampling ---
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means_h0 = r_centered[idx].mean(axis=1)

    # --- One-sided p-value (right tail) ---
    p_value = float(np.mean(boot_means_h0 >= observed_mean))

    # --- 95% CI via percentile method (un-centered) ---
    boot_means_raw = r[idx].mean(axis=1)
    ci_lower = float(np.percentile(boot_means_raw, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means_raw, 100 * (1 - alpha / 2)))

    return BootstrapResult(
        p_value=p_value,
        observed_mean=observed_mean,
        n_trades=n,
        n_bootstrap=n_bootstrap,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        passed=(p_value < alpha) and (n >= min_trades),
    )
