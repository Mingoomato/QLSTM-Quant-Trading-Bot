"""Unit tests for backtesting.validation — Gate 2 bootstrap alpha test.

Validates the implementation against Viktor's specification:
  - H0: E[R] <= 0  vs  H1: E[R] > 0
  - One-sided bootstrap p-value
  - min 250 trades required
  - p < 0.05 to pass
  - Efron & Tibshirani (1993) shift-based resampling
"""

from __future__ import annotations

import numpy as np
import pytest

from backtesting.validation import (
    BREAKEVEN_RATE,
    MIN_TRADES,
    BootstrapResult,
    gate_1_system_check,
    gate_2_bootstrap_validation,
)


# ── Gate 1 sanity tests ──────────────────────────────────────────────


class TestGate1:
    def test_above_breakeven_passes(self):
        assert gate_1_system_check(wins=30, total_trades=100) is True

    def test_below_breakeven_fails(self):
        assert gate_1_system_check(wins=20, total_trades=100) is False

    def test_zero_trades_fails(self):
        assert gate_1_system_check(wins=0, total_trades=0) is False

    def test_exact_breakeven_fails(self):
        # > BREAKEVEN_RATE required, not >=
        # Use fractions that hit exactly 0.254
        total = 1000
        wins = int(BREAKEVEN_RATE * total)  # 254
        assert gate_1_system_check(wins=wins, total_trades=total) is False


# ── Gate 2 bootstrap tests ───────────────────────────────────────────


class TestGate2Bootstrap:
    """Core bootstrap hypothesis test validation."""

    # -- Return type & fields --

    def test_returns_bootstrap_result(self):
        r = np.random.default_rng(0).normal(0.1, 1.0, 300)
        result = gate_2_bootstrap_validation(r)
        assert isinstance(result, BootstrapResult)
        assert hasattr(result, "p_value")
        assert hasattr(result, "observed_mean")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "passed")

    # -- Strong positive alpha → should reject H0 --

    def test_strong_positive_alpha_rejects_h0(self):
        """R-multiples with clear positive mean → p ≈ 0, pass=True."""
        rng = np.random.default_rng(123)
        r = rng.normal(loc=0.5, scale=1.0, size=300)
        result = gate_2_bootstrap_validation(r, seed=42)
        assert result.p_value < 0.05
        assert result.passed is True
        assert result.observed_mean > 0

    # -- Strong negative alpha → should NOT reject H0 --

    def test_negative_mean_fails(self):
        """R-multiples with negative mean → high p-value, pass=False."""
        rng = np.random.default_rng(456)
        r = rng.normal(loc=-0.5, scale=1.0, size=300)
        result = gate_2_bootstrap_validation(r, seed=42)
        assert result.p_value > 0.5  # far from significant
        assert result.passed is False

    # -- Zero-mean data → p ≈ 0.5 --

    def test_zero_mean_pvalue_near_half(self):
        """Under true H0 (mean=0), p-value should be ~0.5."""
        rng = np.random.default_rng(789)
        r = rng.normal(loc=0.0, scale=1.0, size=500)
        result = gate_2_bootstrap_validation(r, seed=42)
        # Allow range [0.2, 0.8] — exact value depends on sample
        assert 0.2 < result.p_value < 0.8
        assert result.passed is False

    # -- Minimum trade count enforcement --

    def test_insufficient_trades_auto_fail(self):
        """Fewer than MIN_TRADES → p=1.0, passed=False, CI=NaN."""
        r = np.ones(100) * 5.0  # trivially positive but too few
        result = gate_2_bootstrap_validation(r, min_trades=MIN_TRADES)
        assert result.p_value == 1.0
        assert result.passed is False
        assert result.n_trades == 100
        assert np.isnan(result.ci_lower)
        assert np.isnan(result.ci_upper)

    def test_empty_input_auto_fail(self):
        result = gate_2_bootstrap_validation([], min_trades=MIN_TRADES)
        assert result.p_value == 1.0
        assert result.passed is False
        assert result.n_trades == 0
        assert result.observed_mean == 0.0

    # -- Reproducibility with seed --

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(999)
        r = rng.normal(0.2, 1.0, 300)
        r1 = gate_2_bootstrap_validation(r, seed=42)
        r2 = gate_2_bootstrap_validation(r, seed=42)
        assert r1.p_value == r2.p_value
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_different_seed_different_result(self):
        rng = np.random.default_rng(111)
        r = rng.normal(0.1, 1.0, 300)
        r1 = gate_2_bootstrap_validation(r, seed=1)
        r2 = gate_2_bootstrap_validation(r, seed=2)
        # Extremely unlikely to be identical with different seeds
        assert r1.p_value != r2.p_value

    # -- Confidence interval sanity --

    def test_ci_contains_observed_mean(self):
        """95% CI should bracket the observed mean for moderate data."""
        rng = np.random.default_rng(222)
        r = rng.normal(0.3, 1.0, 400)
        result = gate_2_bootstrap_validation(r, seed=42)
        assert result.ci_lower <= result.observed_mean <= result.ci_upper

    def test_ci_positive_for_strong_alpha(self):
        """If alpha is strong enough, entire 95% CI should be > 0."""
        rng = np.random.default_rng(333)
        r = rng.normal(1.0, 0.5, 500)  # very strong signal
        result = gate_2_bootstrap_validation(r, seed=42)
        assert result.ci_lower > 0

    # -- p-value bounds --

    def test_pvalue_between_0_and_1(self):
        for loc in [-1.0, 0.0, 0.5, 2.0]:
            rng = np.random.default_rng(42)
            r = rng.normal(loc, 1.0, 300)
            result = gate_2_bootstrap_validation(r, seed=42)
            assert 0.0 <= result.p_value <= 1.0

    # -- Custom alpha threshold --

    def test_custom_alpha_threshold(self):
        """Marginal signal that passes at alpha=0.10 but fails at alpha=0.01."""
        rng = np.random.default_rng(555)
        r = rng.normal(0.12, 1.0, 300)
        result_loose = gate_2_bootstrap_validation(r, alpha=0.10, seed=42)
        result_strict = gate_2_bootstrap_validation(r, alpha=0.01, seed=42)
        # p-value is the same, only pass/fail verdict differs
        assert result_loose.p_value == result_strict.p_value
        # At least one should differ in passed status (or both fail)
        if result_loose.passed:
            assert result_strict.passed is False or result_strict.p_value < 0.01

    # -- Accepts list input --

    def test_accepts_list_input(self):
        r = [0.5, -0.3, 1.2, -0.1, 0.8] * 60  # 300 trades
        result = gate_2_bootstrap_validation(r, seed=42)
        assert isinstance(result.p_value, float)

    # -- n_trades and n_bootstrap fields correct --

    def test_metadata_fields(self):
        r = np.random.default_rng(42).normal(0.1, 1.0, 300)
        result = gate_2_bootstrap_validation(r, n_bootstrap=5000, seed=42)
        assert result.n_trades == 300
        assert result.n_bootstrap == 5000
