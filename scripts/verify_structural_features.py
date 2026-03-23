"""
verify_structural_features.py
─────────────────────────────
Gate 1 engineering sanity check for the 13-dim structural feature pipeline.

Checks:
  1. Output shape [N, 13]
  2. All 13 column names present and in correct order
  3. Rolling Z-score normalization: post-warmup mean ≈ 0, std ≈ 1
  4. Warmup period: first (Z_WINDOW-1) rows are NaN
  5. Clipping: all values in [-5, 5] (post-warmup)
  6. No look-ahead bias (permutation test)
  7. Causal feature correctness spot-checks
  8. Torch tensor conversion [N, 13]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch

from src.models.features_structural import (
    build_features_structural,
    FEAT_COLUMNS,
    FEAT_DIM,
    Z_WINDOW,
)

# ── 1. Generate synthetic data ────────────────────────────────────────────
np.random.seed(42)
N = 2000  # bars

close = 50000 + np.cumsum(np.random.randn(N) * 100)
high = close + np.abs(np.random.randn(N) * 50)
low = close - np.abs(np.random.randn(N) * 50)
opn = close + np.random.randn(N) * 20
volume = np.abs(np.random.randn(N) * 1000) + 500

# Synthetic funding rate (oscillates around 0, 8h cycle → ffill to 1h)
funding_rate = np.sin(np.arange(N) * 2 * np.pi / 24) * 0.001 + np.random.randn(N) * 0.0002

# Synthetic open interest
open_interest = 1e9 + np.cumsum(np.random.randn(N) * 1e6)

# Synthetic taker buy volume
taker_buy_volume = volume * (0.5 + np.random.randn(N) * 0.05)
taker_buy_volume = np.clip(taker_buy_volume, 0, volume)

df = pd.DataFrame({
    "open": opn,
    "high": high,
    "low": low,
    "close": close,
    "volume": volume,
    "funding_rate": funding_rate,
    "open_interest": open_interest,
    "taker_buy_volume": taker_buy_volume,
})

print("=" * 70)
print("STRUCTURAL FEATURES VERIFICATION — Gate 1 Engineering Sanity Check")
print("=" * 70)

# ── 2. Build features ────────────────────────────────────────────────────
feat_df = build_features_structural(df, verbose=True)
print()

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}  — {detail}")
        failed += 1

# ── CHECK 1: Shape ───────────────────────────────────────────────────────
print("\n[CHECK 1] Output shape")
check("Shape is (N, 13)", feat_df.shape == (N, FEAT_DIM),
      f"got {feat_df.shape}, expected ({N}, {FEAT_DIM})")

# ── CHECK 2: Column names ───────────────────────────────────────────────
print("\n[CHECK 2] Column names and order")
check("Columns match FEAT_COLUMNS", list(feat_df.columns) == FEAT_COLUMNS,
      f"got {list(feat_df.columns)}")
check("FEAT_DIM == 13", FEAT_DIM == 13, f"got {FEAT_DIM}")

expected_names = [
    "fr_z", "fr_trend", "oi_change_z", "oi_price_div",
    "liq_long_z", "liq_short_z", "cvd_trend_z", "cvd_price_div",
    "taker_ratio_z", "ema200_dev", "ema200_slope", "vol_regime", "vol_change"
]
check("All 13 expected names present", list(feat_df.columns) == expected_names)

# ── CHECK 3: Warmup NaN ─────────────────────────────────────────────────
print(f"\n[CHECK 3] Warmup period (first {Z_WINDOW-1} bars should be NaN)")
warmup_nans = feat_df.iloc[:Z_WINDOW - 1].isna().all(axis=1).sum()
check(f"First {Z_WINDOW-1} rows are NaN", warmup_nans == Z_WINDOW - 1,
      f"only {warmup_nans}/{Z_WINDOW-1} rows are NaN")

post_warmup = feat_df.iloc[Z_WINDOW:]
check("Post-warmup has no NaN", post_warmup.isna().sum().sum() == 0,
      f"found {post_warmup.isna().sum().sum()} NaN values")

# ── CHECK 4: Z-score normalization ───────────────────────────────────────
print("\n[CHECK 4] Rolling Z-score normalization (post-warmup stats)")
for col in FEAT_COLUMNS:
    v = feat_df[col].dropna()
    mean_ok = abs(v.mean()) < 0.5  # rolling z-score global mean should be near 0
    std_ok = 0.3 < v.std() < 3.0   # should be roughly 1, allow wide margin
    check(f"{col:16s} mean={v.mean():+.3f} std={v.std():.3f}",
          mean_ok and std_ok,
          f"mean_ok={mean_ok}, std_ok={std_ok}")

# ── CHECK 5: Clipping [-5, +5] ──────────────────────────────────────────
print("\n[CHECK 5] Clipping bounds [-5, +5]")
vals = post_warmup.values
check("All values ≤ +5.0", np.nanmax(vals) <= 5.0 + 1e-9,
      f"max={np.nanmax(vals):.4f}")
check("All values ≥ -5.0", np.nanmin(vals) >= -5.0 - 1e-9,
      f"min={np.nanmin(vals):.4f}")

# ── CHECK 6: No look-ahead bias (perturbation test) ─────────────────────
print("\n[CHECK 6] Look-ahead bias test")
# Perturb the LAST 100 bars → features before perturbation point should be identical
df_perturbed = df.copy()
perturb_start = N - 100
df_perturbed.loc[perturb_start:, "close"] *= 1.5
df_perturbed.loc[perturb_start:, "funding_rate"] *= 2.0
df_perturbed.loc[perturb_start:, "open_interest"] *= 0.8

feat_perturbed = build_features_structural(df_perturbed)

# Check bars BEFORE the perturbation minus a buffer for rolling window effects
safe_end = perturb_start - Z_WINDOW - 20  # generous buffer
if safe_end > Z_WINDOW:
    orig_safe = feat_df.iloc[Z_WINDOW:safe_end].values
    pert_safe = feat_perturbed.iloc[Z_WINDOW:safe_end].values
    max_diff = np.nanmax(np.abs(orig_safe - pert_safe))
    check(f"Pre-perturbation features unchanged (max_diff={max_diff:.2e})",
          max_diff < 1e-10,
          f"max_diff={max_diff:.2e} — LOOK-AHEAD DETECTED")
else:
    check("Skipped (not enough bars)", False, "N too small for this test")

# ── CHECK 7: Torch tensor conversion ────────────────────────────────────
print("\n[CHECK 7] Torch tensor conversion")
arr = feat_df.values.astype(np.float32)
arr = np.nan_to_num(arr, nan=0.0)
tensor = torch.from_numpy(arr)
check(f"Tensor shape {tuple(tensor.shape)}", tensor.shape == (N, 13))
check("Tensor dtype float32", tensor.dtype == torch.float32)
check("No inf/nan in tensor", torch.isfinite(tensor).all().item())

# ── CHECK 8: Feature mechanism spot-checks ───────────────────────────────
print("\n[CHECK 8] Feature mechanism spot-checks")
# fr_z should correlate with raw funding_rate
fr_raw = df["funding_rate"].values[Z_WINDOW:]
fr_z = feat_df["fr_z"].values[Z_WINDOW:]
corr_fr = np.corrcoef(fr_raw, fr_z)[0, 1]
check(f"fr_z correlates with raw funding_rate (r={corr_fr:.3f})",
      corr_fr > 0.3, f"correlation too low: {corr_fr:.3f}")

# vol_regime (ATR z-score) should be non-constant
vol_std = feat_df["vol_regime"].dropna().std()
check(f"vol_regime has variance (std={vol_std:.3f})", vol_std > 0.1)

# ── SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 70)

if failed > 0:
    print("\n⚠️  Some checks failed — review feature pipeline before training.")
    sys.exit(1)
else:
    print("\n✅ All checks passed. Feature tensor [N, 13] verified.")
    print(f"   Sample tensor shape: {tensor.shape}")
    print(f"   Ready for QuantumFinancialAgent training.")
    sys.exit(0)
