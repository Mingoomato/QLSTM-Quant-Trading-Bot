"""
src/features/normalization.py
─────────────────────────────────────────────────────────────────────────────
Non-parametric normalization utilities — A/B-testable against rolling z-score.

Background
──────────
Rolling z-scores assume the feature distribution is Gaussian and stationary
within the look-back window.  Structural microstructure signals (funding rate,
OI delta, CVD, liquidation proxy) are heavy-tailed and regime-switching, so
z-score normalization can produce extreme outliers and stale denominators.

Rank-Gaussian (Inverse Normal Transform, INT) maps any distribution to N(0,1):
  1.  Compute the rank of each observation within the rolling window.
  2.  Convert ranks to quantiles in (0, 1) — blom/van der Waerden correction
      avoids -∞ / +∞ at the boundaries.
  3.  Apply scipy.stats.norm.ppf (probit / Φ⁻¹) to obtain standard normal
      scores.

Properties
──────────
  • Distribution-free: works on fat tails, multimodal, regime-switching series.
  • Bounded dynamic range: output ∈ (−4, +4) for any input distribution.
  • Preserves rank ordering (monotone transform) — no information loss.
  • Rolling window keeps it adaptive (same hyper-parameter as z-score window).

References
──────────
  • Blom, G. (1958). Statistical Estimates and Transformed Beta Variables.
    Wiley.  (blom correction: (rank − 3/8) / (n + 1/4))
  • Van der Waerden (1952). Order tests for two-sample problem.
    (correction: rank / (n + 1))
  • Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Wiley. Ch. 8 — fractional differentiation & feature engineering.
  • Ruf, J. & Wang, W. (2020). Neural networks for option pricing and hedging.
    arxiv:1901.08943  (INT pre-processing for finance NN inputs)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm


# ─────────────────────────────────────────────────────────────────────────────
# Correction flavours for rank → quantile mapping
# ─────────────────────────────────────────────────────────────────────────────
_CORRECTIONS = {
    # (rank - a) / (n + b)
    "blom":        (3 / 8, 1 / 4),   # Blom (1958) — recommended default
    "van_der_waerden": (0, 1),        # rank / (n + 1) — classic
    "hazen":       (0.5, 0),          # (rank - 0.5) / n
    "tukey":       (1 / 3, 1 / 3),   # Tukey (1962)
}


def _rank_to_quantile(
    ranks: np.ndarray,
    n: int,
    correction: str = "blom",
) -> np.ndarray:
    """Map integer ranks [1..n] to open-unit-interval quantiles using the
    specified plotting-position correction."""
    a, b = _CORRECTIONS[correction]
    return (ranks - a) / (n + b)


# ─────────────────────────────────────────────────────────────────────────────
# Core scalar function (vectorised over a single window)
# ─────────────────────────────────────────────────────────────────────────────

def _int_transform(
    window: np.ndarray,
    correction: str = "blom",
    clip_sigma: float = 4.0,
) -> float:
    """Inverse-Normal Transform for the *last* value of a rolling window.

    Parameters
    ----------
    window : 1-D array, shape (w,)
        The rolling window values.  The last element is the current observation.
    correction : str
        Plotting-position correction — see _CORRECTIONS.
    clip_sigma : float
        Hard clip on the probit output to prevent ±∞ bleed-through.

    Returns
    -------
    float
        Standard-normal score for window[-1].
    """
    n = len(window)
    if n < 2:
        return 0.0

    # argsort argsort = rank (0-based), shift to 1-based
    ranks = np.argsort(np.argsort(window)) + 1  # shape (n,)
    q = _rank_to_quantile(ranks, n, correction)
    # probit of the last element
    score = float(_norm.ppf(q[-1]))
    return float(np.clip(score, -clip_sigma, clip_sigma))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def normalize_rank_gaussian(
    series: pd.Series,
    window: int = 96,
    correction: str = "blom",
    clip_sigma: float = 4.0,
    min_periods: int | None = None,
) -> pd.Series:
    """Non-parametric Inverse-Normal Transform (rank-Gaussian normalization).

    Drop-in replacement for rolling z-score.  Maps any univariate series to
    approximate N(0, 1) using only rank information within a rolling window —
    no distributional assumption required.

    Parameters
    ----------
    series : pd.Series
        Raw feature values (e.g. funding_rate, oi_change, cvd_trend …).
    window : int, default 96
        Look-back length (candles).  Same semantic as z-score window.
        For 1h candles: 96 = 4 days.
    correction : {'blom', 'van_der_waerden', 'hazen', 'tukey'}
        Plotting-position correction for rank → quantile mapping.
        'blom' is the recommended default (minimal expected squared error).
    clip_sigma : float, default 4.0
        Hard clip on probit output.  Prevents infinite values when all
        observations in the window are identical (quantile = 0 or 1).
    min_periods : int or None
        Minimum number of non-NaN observations to compute a result.
        Defaults to max(2, window // 4).

    Returns
    -------
    pd.Series
        Rank-Gaussian normalized values, same index as input.
        NaN where the window is too short (controlled by min_periods).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> s = pd.Series(np.random.exponential(1, 200))  # right-skewed
    >>> z  = (s - s.rolling(96).mean()) / s.rolling(96).std()   # z-score
    >>> rg = normalize_rank_gaussian(s, window=96)               # INT
    >>> # rg is ~N(0,1) by construction; z is NOT when s is exponential
    """
    if min_periods is None:
        min_periods = max(2, window // 4)

    result = series.rolling(window=window, min_periods=min_periods).apply(
        lambda w: _int_transform(w, correction=correction, clip_sigma=clip_sigma),
        raw=True,
    )
    return result.rename(f"{series.name}_rg" if series.name else "rg")


def normalize_zscore_rolling(
    series: pd.Series,
    window: int = 96,
    min_periods: int | None = None,
    clip_sigma: float = 4.0,
) -> pd.Series:
    """Rolling z-score — the *baseline* for A/B testing.

    Included here so callers can import both normalizers from one module and
    swap them with a single flag.

    Parameters
    ----------
    series : pd.Series
        Raw feature values.
    window : int
        Look-back window (same as normalize_rank_gaussian).
    min_periods : int or None
        Defaults to max(2, window // 4).
    clip_sigma : float
        Hard clip to match the INT output range.

    Returns
    -------
    pd.Series
        Z-score normalized values, clipped to ±clip_sigma.
    """
    if min_periods is None:
        min_periods = max(2, window // 4)

    mu = series.rolling(window=window, min_periods=min_periods).mean()
    sigma = series.rolling(window=window, min_periods=min_periods).std(ddof=1)
    z = (series - mu) / sigma.replace(0, np.nan)
    return z.clip(-clip_sigma, clip_sigma).rename(
        f"{series.name}_z" if series.name else "z"
    )


def normalize_feature(
    series: pd.Series,
    method: str = "rank_gaussian",
    window: int = 96,
    **kwargs,
) -> pd.Series:
    """Dispatcher for A/B testing.

    Parameters
    ----------
    series : pd.Series
    method : {'rank_gaussian', 'zscore'}
        'rank_gaussian' → normalize_rank_gaussian()
        'zscore'        → normalize_zscore_rolling()
    window : int
    **kwargs
        Forwarded to the selected normalizer.

    Returns
    -------
    pd.Series
    """
    if method == "rank_gaussian":
        return normalize_rank_gaussian(series, window=window, **kwargs)
    elif method == "zscore":
        return normalize_zscore_rolling(series, window=window, **kwargs)
    else:
        raise ValueError(f"Unknown normalization method: {method!r}. "
                         f"Choose 'rank_gaussian' or 'zscore'.")


# ─────────────────────────────────────────────────────────────────────────────
# Batch helper — normalize a DataFrame of structural features
# ─────────────────────────────────────────────────────────────────────────────

# 13-dim structural feature columns (from MEMORY.md spec)
STRUCTURAL_FEATURE_COLS = [
    "fr_z", "fr_trend",
    "oi_change_z", "oi_price_div",
    "liq_long_z", "liq_short_z",
    "cvd_trend_z", "cvd_price_div", "taker_ratio_z",
    "ema200_dev", "ema200_slope", "vol_regime", "vol_change",
]


def normalize_structural_features(
    df: pd.DataFrame,
    method: str = "rank_gaussian",
    window: int = 96,
    cols: list[str] | None = None,
    suffix: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Apply normalization to each structural feature column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns in `cols`.
    method : {'rank_gaussian', 'zscore'}
        Normalization method to apply.
    window : int
        Rolling window (candles).
    cols : list[str] or None
        Columns to normalize.  Defaults to STRUCTURAL_FEATURE_COLS.
    suffix : bool
        If True, add '_rg' or '_z' suffix to output columns.
        If False (default), overwrite the input columns in-place copy.
    **kwargs
        Forwarded to the normalizer.

    Returns
    -------
    pd.DataFrame
        New DataFrame (original not mutated).
    """
    if cols is None:
        cols = [c for c in STRUCTURAL_FEATURE_COLS if c in df.columns]

    out = df.copy()
    for col in cols:
        normalized = normalize_feature(out[col], method=method, window=window, **kwargs)
        if suffix:
            out[normalized.name] = normalized
        else:
            out[col] = normalized.values  # keep original column name
    return out
