# Feature Validation Framework Specification

**Author:** Viktor, Quant Researcher CTO
**Date:** 2026-03-23
**Status:** MANDATORY — No feature enters production without passing all gates.

---

## 1. Motivation

The 28-dim V3 model collapsed OOS (WR=26.4%, ROI=-54%) because statistical features
(especially Hurst R/S) were non-stationary and overfitted to training regimes. This
specification establishes a rigorous, single-factor validation framework using
Combinatorially Purged Cross-Validation (CPCV) to prevent recurrence.

**References:**
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 7–12.
- Lopez de Prado, M. & Lewis, M. (2019). "Detection of False Investment Strategies Using Unsupervised Learning Methods." *Quantitative Finance*, 19(9).
- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*, 40(5), pp. 94-107.

---

## 2. Validation Gates (Sequential, All Must Pass)

### Gate 0: Stationarity Pre-Screen
**Purpose:** Reject features that are structurally non-stationary before any backtest.

| Test | Threshold | Action on Fail |
|------|-----------|----------------|
| ADF (Augmented Dickey-Fuller) | p < 0.01 on raw feature | Reject or difference |
| KPSS (stationarity null) | p > 0.05 | Reject |
| Rolling mean/var ratio | max/min < 3.0 over 180-day windows | Reject |
| Alpha half-life (OLS on lagged values) | t_half > 180 bars (180 hours at 1h) | Reject |

**Rationale:** Hurst R/S failed because it is I(1) on short windows. ADF+KPSS double-test
catches both unit-root and trend-stationarity failures.

### Gate 1: Single-Factor Profitability (Engineering Kill-Switch)
**Purpose:** Verify that a feature, in isolation, generates positive PnL above breakeven.

| Metric | Threshold | Derivation |
|--------|-----------|------------|
| Win Rate (WR) | > 25.4% (BEP) | TP=3.0×ATR, SL=1.0×ATR, fees=0.375%/trade |
| Total PnL | > 0 after fees | Net of 0.075% round-trip × 5x eff. leverage |
| Trade count | ≥ 100 | Minimum for statistical power |

**Method:** `scripts/backtest_structural.py --signals <single_feature> --days 365`

Gate 1 is NOT a statistical proof of alpha. It is a binary engineering filter: if a feature
cannot produce positive PnL in isolation under realistic costs, it has no business in a model.

### Gate 2: Statistical Significance via CPCV
**Purpose:** Prove the feature's edge is not due to overfitting or data snooping.

#### 2.1 Combinatorially Purged Cross-Validation (CPCV)

Standard k-fold CV leaks information through:
1. **Serial correlation:** Train/test splits share autocorrelated observations.
2. **Label leakage:** Labels computed from future data overlap with training windows.

CPCV fixes both:

```
Given N groups, choose k test groups from C(N, k) combinations.
For each combination:
  1. Define test set T = union of k groups
  2. Define embargo zone E = {t : |t - boundary(T)| < h}
     where h = max label horizon (96 bars for our system)
  3. Define purge zone P = {t ∈ train : label(t) overlaps any t' ∈ T}
  4. Train set = all data \ (T ∪ E ∪ P)
  5. Train model, evaluate on T
```

**Parameters for our system:**
- N = 6 groups (each ~2 months of 1h data over 365 days)
- k = 1 (leave-one-group-out) → C(6,1) = 6 folds
- h = 96 bars (embargo = 96 hours = 4 days)
- Label horizon = TP/SL hit time, max 96 bars

This yields 6 OOS performance samples. Each fold's train set is purged of any observation
whose label window overlaps the test fold, plus a 96-bar embargo buffer.

#### 2.2 Statistical Tests on CPCV Results

| Test | Threshold | Formula |
|------|-----------|---------|
| Bootstrap 95% CI on WR | Lower bound > 25.4% (BEP) | 10,000 bootstrap resamples of per-trade R-multiples |
| Deflated Sharpe Ratio (DSR) | DSR > 0 at 95% confidence | DSR = (SR̂ - SR₀) / σ̂_SR × deflation(N_trials) |
| Binomial test | p < 0.05 | H₀: WR = BEP; H₁: WR > BEP |

**Deflated Sharpe Ratio** accounts for multiple testing:
```
SR₀ = E[max(SR₁,...,SR_N)] under H₀ (all strategies are noise)
    ≈ √(2 log N) × σ_SR    (Bonferroni-like correction)

DSR = Φ[ (SR̂ - SR₀) / σ̂_SR × √T ]
```
where N = number of feature candidates tested. If we test 13 features individually,
N=13 and the Sharpe hurdle rises accordingly.

### Gate 3: Regime Robustness (Rolling OOS)
**Purpose:** Verify the feature works across different market regimes.

| Test | Method | Pass Criterion |
|------|--------|----------------|
| Rolling 90-day OOS | Walk-forward: train on [0,t], test on [t, t+90d] | WR > BEP in ≥ 4/6 windows |
| Regime-conditional WR | Split by vol_regime (low/mid/high) | WR > BEP in ≥ 2/3 regimes |
| Drawdown recovery | Max consecutive losses | < 8 (current circuit breaker) |
| Signal decay test | t-stat of feature→return on [t, t+Δ] for Δ=1..30 days | t-stat > 2.0 for Δ ≤ 14 days |

### Gate 4: Causal Attribution (Post-Training)
**Purpose:** Verify the model actually uses the feature, not a spurious correlate.

| Method | Tool | Pass Criterion |
|--------|------|----------------|
| Integrated Gradients | `tools/analyze_residuals.py` | Feature attribution ≥ 5% of total gradient magnitude |
| Permutation importance | Shuffle single feature, measure PnL drop | Δ_PnL < -5% (feature matters) |
| SHAP interaction | SHAP values on test set | No single interaction > 50% of main effect (no proxy) |

---

## 3. Feature Candidate Lifecycle

```
Proposed Feature
       │
       ▼
  [Gate 0: Stationarity]──FAIL──→ Reject / Transform
       │ PASS
       ▼
  [Gate 1: Single-Factor PnL]──FAIL──→ Reject
       │ PASS
       ▼
  [Gate 2: CPCV + DSR]──FAIL──→ Reject (overfitting)
       │ PASS
       ▼
  [Gate 3: Regime Robustness]──FAIL──→ Conditional (regime-gated only)
       │ PASS
       ▼
  [Gate 4: Causal Attribution]──FAIL──→ Reject (spurious)
       │ PASS
       ▼
  APPROVED for Production
```

---

## 4. Hard Limits

| Parameter | Limit | Rationale |
|-----------|-------|-----------|
| Max feature count | 15 | Curse of dimensionality: with T≈4000 1h bars/year and TP=3×ATR trade frequency ~200/year, effective sample-to-feature ratio must stay > 13:1 (see §5 post-mortem) |
| Max correlated features | |ρ| < 0.7 pairwise | Multicollinearity inflates variance |
| Min alpha half-life | 180 bars (7.5 days at 1h) | Features decaying faster are arbitraged away |
| Min trade count per fold | 15 | Below this, WR CI is too wide for inference |

---

## 5. Implementation Checklist

- [ ] `tools/run_cpcv_validation.py` — CPCV engine with purging + embargo
- [ ] `tools/analyze_residuals.py` — Residual analysis + attribution (see separate script)
- [ ] `tools/compute_deflated_sharpe.py` — DSR calculator with multiple-testing correction
- [ ] Update `scripts/backtest_structural.py` to output per-fold metrics
- [ ] Add Gate 0 stationarity checks to feature builder pipeline

---

## 6. Appendix: Why CPCV Over Standard K-Fold

Standard k-fold on time series:
- Train: [0, 0.8T], Test: [0.8T, T] → single split, high variance
- Expanding window: train grows monotonically → recency bias
- Purged k-fold (without combinatorial): only k folds → low statistical power

CPCV with N=6, k=1 gives 6 independent OOS evaluations with proper purging.
With N=6, k=2: C(6,2)=15 folds — more power but higher computation.
Start with k=1; upgrade to k=2 if ambiguous results.

**Key insight from Lopez de Prado:** The probability of backtest overfitting (PBO)
can be computed from CPCV results:
```
PBO = fraction of CPCV combinations where OOS Sharpe < 0
```
Target: PBO < 0.10 (less than 10% of folds show negative OOS Sharpe).
