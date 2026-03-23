"""
DEPRECATED — features_v3.py
────────────────────────────────────────────────────────────────────────────
Moved here 2026-03-23 as part of structural-feature pivot.

Reason for deprecation:
  OOS 2026 Q1 validation showed WR=26.4% (noise-level) for QLSTM models
  trained on statistical features (Hurst, entropy, MACD, RSI, etc.).
  These signals are self-destructive in live markets.  The active pipeline
  now uses 13-dim structural/mechanism features (features_structural.py).

Do NOT use in new code.  No active pipeline file imports from here.
Retained for historical reference only.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.features_v2 import build_features_v2, compute_true_atr


# ── V3 extra feature helpers ──────────────────────────────────────────────────

def _hurst_rs(log_rets: np.ndarray, n_scales: int = 4) -> float:
    """Hurst exponent via R/S analysis (pure NumPy)."""
    n = len(log_rets)
    if n < 8:
        return 0.5

    min_scale = max(4, n // (2 ** n_scales))
    log_ns, log_rs = [], []

    for k in range(n_scales):
        s = max(4, min_scale * (2 ** k))
        if s > n // 2:
            break
        n_chunks = n // s
        if n_chunks < 1:
            continue

        rs_vals = []
        for c in range(n_chunks):
            chunk = log_rets[c * s: (c + 1) * s]
            y = chunk - chunk.mean()
            cumsum = np.cumsum(y)
            R = cumsum.max() - cumsum.min()
            S = chunk.std() + 1e-8
            rs_vals.append(R / S)

        if rs_vals:
            rs_mean = float(np.mean(rs_vals))
            log_ns.append(math.log(s))
            log_rs.append(math.log(max(rs_mean, 1e-8)))

    if len(log_ns) < 2:
        return 0.5

    ln = np.array(log_ns)
    lr = np.array(log_rs)
    ln_c = ln - ln.mean()
    lr_c = lr - lr.mean()
    denom = (ln_c ** 2).sum()
    if denom < 1e-10:
        return 0.5

    H = float((ln_c * lr_c).sum() / denom)
    return float(np.clip(H, 0.05, 0.95))


def _lag1_autocorrelation(log_rets: np.ndarray) -> float:
    """Lag-1 autocorrelation γ(1)."""
    if len(log_rets) < 4:
        return 0.0
    r = log_rets - log_rets.mean()
    cov1 = float(np.mean(r[1:] * r[:-1]))
    var = float(np.var(r)) + 1e-10
    return float(np.clip(cov1 / var, -0.9, 0.9))


def _price_entropy(closes: np.ndarray, n_bins: int = 10) -> float:
    """Shannon entropy of the price-change distribution (normalised 0–1)."""
    window = closes[-20:] if len(closes) >= 20 else closes
    if len(window) < 3:
        return 0.5

    returns = np.diff(np.log(np.where(window > 0, window, 1e-8)))
    if len(returns) < 2:
        return 0.5

    counts, _ = np.histogram(returns, bins=n_bins)
    total = counts.sum()
    if total == 0:
        return 0.5

    probs = counts / total
    eps = 1e-10
    H = -np.sum(probs[probs > 0] * np.log(probs[probs > 0] + eps))
    H_max = math.log(n_bins)
    return float(np.clip(H / (H_max + eps), 0.0, 1.0))


# ── V3 per-bar feature builder ────────────────────────────────────────────────

def build_features_v3(df: pd.DataFrame) -> np.ndarray:
    """
    Build a 30-dim V3 feature vector for the LAST bar in df.
    DEPRECATED — see module docstring.
    """
    base = build_features_v2(df)

    closes = df["close"].values.astype(float)
    n = len(closes)
    eps = 1e-8

    if n >= 2:
        log_rets = np.diff(np.log(np.where(closes > 0, closes, eps)))
    else:
        log_rets = np.array([0.0])

    H = _hurst_rs(log_rets, n_scales=4)
    entropy = _price_entropy(closes, n_bins=8)
    purity_proxy = float(np.clip(1.0 - 2.0 * entropy, -1.0, 1.0))

    extra = np.array([
        float(np.clip(H, -np.pi, np.pi)),
        float(np.clip(purity_proxy, -np.pi, np.pi)),
    ], dtype=np.float32)

    return np.concatenate([base, extra])


# ── Disk-cached V3 feature matrix builder ────────────────────────────────────

def generate_and_cache_features_v3(
    df_clean: pd.DataFrame,
    cache_path: str,
    warmup: int = 120,
    lookback: int = 30,
    verbose: bool = True,
) -> np.ndarray:
    """Build [N, 30] V3 feature matrix with disk cache. DEPRECATED."""
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if len(cached) == len(df_clean):
            if verbose:
                print(f"[features_v3] Loaded cache: {cache_path}")
            return cached
        if verbose:
            print(
                f"[features_v3] Cache size mismatch "
                f"({len(cached)} vs {len(df_clean)}), rebuilding..."
            )

    if verbose:
        print(f"[features_v3] Building {len(df_clean)} V3 feature vectors...")

    n = len(df_clean)
    feature_list = []

    iterator = (
        tqdm(range(n), desc="Building V3 Features", ascii=True)
        if verbose else range(n)
    )
    for i in iterator:
        if i < warmup:
            feature_list.append(np.zeros(17, dtype=np.float32))
            continue
        window = df_clean.iloc[max(0, i - lookback + 1): i + 1]
        feature_list.append(build_features_v3(window))

    all_features = np.array(feature_list, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.save(cache_path, all_features)
    if verbose:
        print(f"[features_v3] Saved cache: {cache_path}")
    return all_features
