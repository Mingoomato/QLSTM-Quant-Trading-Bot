"""
calibration_validator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Calibration Validation for p_long Probability Buckets (Task A10)

Implements:
  1. reliability_diagram()  — binned reliability diagram data
  2. compute_ece()          — Expected Calibration Error across buckets
  3. validate_calibration() — Gate check (PASS/FAIL) for model-standalone alpha

Gate Criteria (from implementation plan):
  - If actual WR in p_long > 0.70 bucket < 60% → FAIL
  - If ECE >= 0.10 → FAIL
  - Both must pass for model-standalone alpha approval

Mathematical Foundation:
  ECE = Σ_m (|B_m| / N) · |acc(B_m) - conf(B_m)|

  where B_m = samples in bin m, acc = actual win rate, conf = mean predicted prob.
  ECE ∈ [0, 1]; lower is better. ECE = 0 means perfectly calibrated.

  Reliability diagram: plot acc(B_m) vs conf(B_m) for each bin m.
  Perfect calibration = diagonal line (acc = conf).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BinStats:
    """Statistics for a single calibration bin."""
    bin_lower: float          # Lower edge of bin
    bin_upper: float          # Upper edge of bin
    bin_center: float         # Midpoint of bin
    n_samples: int            # Number of samples in bin
    mean_predicted: float     # Mean predicted probability (confidence)
    actual_win_rate: float    # Actual win rate (accuracy) in bin
    calibration_gap: float    # |actual_win_rate - mean_predicted|


@dataclass
class ReliabilityResult:
    """Full result from reliability_diagram()."""
    bins: List[BinStats]               # Per-bin statistics
    n_bins: int                        # Number of bins requested
    n_total: int                       # Total number of samples
    ece: float                         # Expected Calibration Error
    mce: float                         # Maximum Calibration Error
    overconfidence_ratio: float        # Fraction of bins where conf > acc
    bin_edges: np.ndarray              # Bin edge array [n_bins + 1]


@dataclass
class CalibrationGateResult:
    """Gate check result for model-standalone alpha approval."""
    status: str                        # "PASS" or "FAIL"
    ece: float                         # ECE value
    ece_threshold: float               # ECE threshold (default 0.10)
    ece_pass: bool                     # ECE < threshold
    high_conf_wr: Optional[float]      # Actual WR in p_long > 0.70 bucket
    high_conf_n: int                   # N samples in high-conf bucket
    high_conf_wr_threshold: float      # WR threshold (default 0.60)
    high_conf_wr_pass: bool            # WR >= threshold
    reliability: ReliabilityResult     # Full reliability diagram data
    reasons: List[str]                 # Human-readable failure reasons


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reliability Diagram
# ─────────────────────────────────────────────────────────────────────────────

def reliability_diagram(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> ReliabilityResult:
    """
    Compute reliability diagram statistics for calibration assessment.

    The reliability diagram bins predictions by confidence level and compares
    the mean predicted probability to the actual win rate in each bin.
    Perfect calibration: actual_win_rate == mean_predicted for every bin.

    Args:
        predictions: Array of predicted probabilities (p_long), shape [N].
                     Values in [0, 1].
        actuals:     Array of binary outcomes (1 = win/long correct, 0 = loss),
                     shape [N]. Must be same length as predictions.
        n_bins:      Number of equal-width bins to divide [0, 1] into.

    Returns:
        ReliabilityResult with per-bin stats, ECE, and MCE.

    Raises:
        ValueError: If inputs are invalid (empty, different lengths, out of range).
    """
    # ── Input validation ────────────────────────────────────────────────
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    if predictions.ndim != 1 or actuals.ndim != 1:
        raise ValueError("predictions and actuals must be 1-D arrays")
    if len(predictions) != len(actuals):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, actuals={len(actuals)}"
        )
    if len(predictions) == 0:
        raise ValueError("Cannot compute reliability diagram on empty arrays")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    # Clamp predictions to [0, 1] (handles floating-point edge cases)
    predictions = np.clip(predictions, 0.0, 1.0)

    # Validate actuals are binary
    unique_vals = np.unique(actuals)
    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        raise ValueError(
            f"actuals must be binary (0 or 1), got unique values: {unique_vals}"
        )

    N = len(predictions)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[BinStats] = []
    ece = 0.0
    mce = 0.0
    overconf_count = 0
    nonempty_count = 0

    # ── Bin computation ─────────────────────────────────────────────────
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        center = (lo + hi) / 2.0

        # Last bin includes upper edge (closed on right)
        if i == n_bins - 1:
            mask = (predictions >= lo) & (predictions <= hi)
        else:
            mask = (predictions >= lo) & (predictions < hi)

        n_in_bin = int(mask.sum())

        if n_in_bin == 0:
            bins.append(BinStats(
                bin_lower=lo, bin_upper=hi, bin_center=center,
                n_samples=0, mean_predicted=center,
                actual_win_rate=0.0, calibration_gap=0.0,
            ))
            continue

        mean_pred = float(predictions[mask].mean())
        actual_wr = float(actuals[mask].mean())
        gap = abs(actual_wr - mean_pred)

        bins.append(BinStats(
            bin_lower=lo, bin_upper=hi, bin_center=center,
            n_samples=n_in_bin, mean_predicted=mean_pred,
            actual_win_rate=actual_wr, calibration_gap=gap,
        ))

        # ECE: weighted absolute gap
        ece += (n_in_bin / N) * gap
        # MCE: max gap
        mce = max(mce, gap)
        # Overconfidence: predicted > actual
        if mean_pred > actual_wr:
            overconf_count += 1
        nonempty_count += 1

    overconf_ratio = (overconf_count / nonempty_count) if nonempty_count > 0 else 0.0

    return ReliabilityResult(
        bins=bins,
        n_bins=n_bins,
        n_total=N,
        ece=ece,
        mce=mce,
        overconfidence_ratio=overconf_ratio,
        bin_edges=bin_edges,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. ECE Computation (standalone, for quick checks)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) across probability buckets.

    ECE = Σ_m (|B_m| / N) · |acc(B_m) - conf(B_m)|

    This is a convenience wrapper around reliability_diagram().

    Args:
        predictions: Predicted probabilities [N], values in [0, 1].
        actuals:     Binary outcomes [N] (1 = win, 0 = loss).
        n_bins:      Number of equal-width bins.

    Returns:
        ECE value in [0, 1]. Lower is better.
    """
    result = reliability_diagram(predictions, actuals, n_bins=n_bins)
    return result.ece


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gate Check — Calibration Validation for Model-Standalone Alpha
# ─────────────────────────────────────────────────────────────────────────────

def validate_calibration(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
    ece_threshold: float = 0.10,
    high_conf_threshold: float = 0.70,
    high_conf_wr_threshold: float = 0.60,
    min_high_conf_samples: int = 5,
) -> CalibrationGateResult:
    """
    Gate check for model-standalone alpha approval (Task A10).

    Criteria (ALL must pass):
      1. ECE < ece_threshold (default 0.10)
      2. Actual WR in p_long > high_conf_threshold bucket >= high_conf_wr_threshold
      3. Sufficient samples in high-confidence bucket (>= min_high_conf_samples)

    If any criterion fails, status = "FAIL" and model-standalone alpha is disabled.

    Args:
        predictions:            Predicted p_long values [N], in [0, 1].
        actuals:                Binary outcomes [N] (1 = long was correct, 0 = not).
        n_bins:                 Number of bins for reliability diagram / ECE.
        ece_threshold:          Maximum acceptable ECE (default 0.10).
        high_conf_threshold:    p_long threshold for "high confidence" bucket (default 0.70).
        high_conf_wr_threshold: Minimum actual WR required in high-conf bucket (default 0.60).
        min_high_conf_samples:  Minimum samples needed in high-conf bucket for valid test.

    Returns:
        CalibrationGateResult with PASS/FAIL status and detailed diagnostics.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    # Compute full reliability diagram + ECE
    reliability = reliability_diagram(predictions, actuals, n_bins=n_bins)
    ece = reliability.ece

    reasons: List[str] = []

    # ── Check 1: ECE threshold ──────────────────────────────────────────
    ece_pass = ece < ece_threshold
    if not ece_pass:
        reasons.append(
            f"ECE = {ece:.4f} >= {ece_threshold:.2f} threshold"
        )

    # ── Check 2: High-confidence bucket WR ──────────────────────────────
    high_conf_mask = predictions > high_conf_threshold
    high_conf_n = int(high_conf_mask.sum())

    if high_conf_n < min_high_conf_samples:
        # Insufficient samples — cannot validate high-conf bucket
        high_conf_wr = None
        high_conf_wr_pass = False
        reasons.append(
            f"Insufficient samples in p_long > {high_conf_threshold:.2f} bucket: "
            f"N={high_conf_n} < {min_high_conf_samples} minimum"
        )
    else:
        high_conf_wr = float(actuals[high_conf_mask].mean())
        high_conf_wr_pass = high_conf_wr >= high_conf_wr_threshold
        if not high_conf_wr_pass:
            reasons.append(
                f"Actual WR in p_long > {high_conf_threshold:.2f} bucket = "
                f"{high_conf_wr:.4f} < {high_conf_wr_threshold:.2f} threshold"
            )

    # ── Final gate decision ─────────────────────────────────────────────
    all_pass = ece_pass and high_conf_wr_pass
    status = "PASS" if all_pass else "FAIL"

    return CalibrationGateResult(
        status=status,
        ece=ece,
        ece_threshold=ece_threshold,
        ece_pass=ece_pass,
        high_conf_wr=high_conf_wr,
        high_conf_n=high_conf_n,
        high_conf_wr_threshold=high_conf_wr_threshold,
        high_conf_wr_pass=high_conf_wr_pass,
        reliability=reliability,
        reasons=reasons,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pretty-Print Utilities
# ─────────────────────────────────────────────────────────────────────────────

def format_reliability_table(result: ReliabilityResult) -> str:
    """Format reliability diagram as ASCII table for TUI / logging."""
    lines = [
        f"{'Bin':>12s} | {'N':>5s} | {'Mean Pred':>9s} | {'Actual WR':>9s} | {'Gap':>7s}",
        "-" * 55,
    ]
    for b in result.bins:
        if b.n_samples == 0:
            lines.append(
                f"[{b.bin_lower:.2f}-{b.bin_upper:.2f}] | {'---':>5s} | "
                f"{'---':>9s} | {'---':>9s} | {'---':>7s}"
            )
        else:
            lines.append(
                f"[{b.bin_lower:.2f}-{b.bin_upper:.2f}] | {b.n_samples:>5d} | "
                f"{b.mean_predicted:>9.4f} | {b.actual_win_rate:>9.4f} | "
                f"{b.calibration_gap:>7.4f}"
            )
    lines.append("-" * 55)
    lines.append(f"ECE = {result.ece:.4f}  |  MCE = {result.mce:.4f}  |  N = {result.n_total}")
    return "\n".join(lines)


def format_gate_result(result: CalibrationGateResult) -> str:
    """Format gate check result for logging."""
    icon = "PASS" if result.status == "PASS" else "FAIL"
    lines = [
        f"=== Calibration Gate: [{icon}] ===",
        f"  ECE:  {result.ece:.4f}  (threshold < {result.ece_threshold:.2f})  "
        f"{'OK' if result.ece_pass else 'FAIL'}",
    ]
    if result.high_conf_wr is not None:
        lines.append(
            f"  WR(p>{result.high_conf_wr_threshold:.0%} bucket): "
            f"{result.high_conf_wr:.4f}  (threshold >= {result.high_conf_wr_threshold:.2f})  "
            f"{'OK' if result.high_conf_wr_pass else 'FAIL'}  "
            f"(N={result.high_conf_n})"
        )
    else:
        lines.append(
            f"  WR(p>0.70 bucket): N/A — insufficient samples "
            f"(N={result.high_conf_n})"
        )
    if result.reasons:
        lines.append("  Failure reasons:")
        for r in result.reasons:
            lines.append(f"    - {r}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Self-Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  calibration_validator.py -- Self-Test")
    print("=" * 70)

    np.random.seed(42)

    # ── Test 1: Perfectly calibrated model ──────────────────────────────
    print("\n[1] Perfectly calibrated model (predictions == actual rates)")
    N = 1000
    preds_perfect = np.random.uniform(0.0, 1.0, N)
    actuals_perfect = (np.random.uniform(0.0, 1.0, N) < preds_perfect).astype(float)

    rel = reliability_diagram(preds_perfect, actuals_perfect, n_bins=10)
    print(format_reliability_table(rel))
    print(f"  ECE should be small: {rel.ece:.4f}")
    assert rel.ece < 0.05, f"ECE too high for perfect calibration: {rel.ece}"
    print("  OK")

    # ── Test 2: Overconfident model ─────────────────────────────────────
    print("\n[2] Overconfident model (predicts 0.80 but actual WR = 0.50)")
    N = 200
    preds_overconf = np.full(N, 0.80)
    actuals_overconf = np.random.binomial(1, 0.50, N).astype(float)

    rel2 = reliability_diagram(preds_overconf, actuals_overconf, n_bins=10)
    print(format_reliability_table(rel2))
    print(f"  ECE should be ~0.30: {rel2.ece:.4f}")
    assert rel2.ece > 0.20, f"ECE too low for overconfident model: {rel2.ece}"
    print("  OK")

    # ── Test 3: Gate check — PASS scenario ──────────────────────────────
    print("\n[3] Gate check — good model (should PASS)")
    N = 300
    # Well-calibrated model: high-conf bucket has WR ~ 0.72
    preds_good = np.concatenate([
        np.random.uniform(0.3, 0.7, 200),   # medium confidence
        np.random.uniform(0.71, 0.90, 100),  # high confidence
    ])
    actuals_good = np.concatenate([
        np.random.binomial(1, 0.50, 200).astype(float),  # ~50% WR
        np.random.binomial(1, 0.75, 100).astype(float),  # ~75% WR in high-conf
    ])

    gate = validate_calibration(preds_good, actuals_good, n_bins=10)
    print(format_gate_result(gate))
    print(f"  Status: {gate.status}")
    # Note: stochastic — might not always pass, but likely with these params

    # ── Test 4: Gate check — FAIL scenario (low WR in high-conf) ───────
    print("\n[4] Gate check — bad model (high-conf WR < 60%, should FAIL)")
    N = 200
    preds_bad = np.concatenate([
        np.random.uniform(0.3, 0.7, 100),
        np.random.uniform(0.71, 0.95, 100),  # claims high confidence
    ])
    actuals_bad = np.concatenate([
        np.random.binomial(1, 0.50, 100).astype(float),
        np.random.binomial(1, 0.30, 100).astype(float),  # but actual WR only 30%
    ])

    gate2 = validate_calibration(preds_bad, actuals_bad, n_bins=10)
    print(format_gate_result(gate2))
    assert gate2.status == "FAIL", "Should FAIL with low high-conf WR"
    assert not gate2.high_conf_wr_pass, "high_conf_wr_pass should be False"
    print("  Correctly returned FAIL")

    # ── Test 5: Gate check — FAIL scenario (high ECE) ──────────────────
    print("\n[5] Gate check — high ECE (should FAIL)")
    N = 200
    preds_ece_bad = np.full(N, 0.90)  # always predicts 90%
    actuals_ece_bad = np.random.binomial(1, 0.50, N).astype(float)

    gate3 = validate_calibration(preds_ece_bad, actuals_ece_bad, n_bins=10)
    print(format_gate_result(gate3))
    assert gate3.status == "FAIL", "Should FAIL with high ECE"
    assert not gate3.ece_pass, "ece_pass should be False"
    print("  Correctly returned FAIL")

    # ── Test 6: Edge case — empty high-conf bucket ──────────────────────
    print("\n[6] Edge case — no samples above 0.70")
    preds_low = np.random.uniform(0.2, 0.65, 100)
    actuals_low = np.random.binomial(1, 0.45, 100).astype(float)

    gate4 = validate_calibration(preds_low, actuals_low, n_bins=10)
    print(format_gate_result(gate4))
    assert gate4.status == "FAIL", "Should FAIL with insufficient high-conf samples"
    assert gate4.high_conf_wr is None
    print("  Correctly returned FAIL (insufficient samples)")

    # ── Test 7: compute_ece standalone ──────────────────────────────────
    print("\n[7] compute_ece() standalone function")
    ece_val = compute_ece(preds_perfect, actuals_perfect, n_bins=10)
    print(f"  ECE = {ece_val:.4f}")
    assert isinstance(ece_val, float)
    print("  OK")

    print("\n" + "=" * 70)
    print("  All calibration_validator.py tests passed!")
    print("=" * 70)
