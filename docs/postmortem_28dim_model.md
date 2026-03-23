# Post-Mortem: 28-Dimension V3 Model Failure

**Author:** Viktor, Quant Researcher CTO
**Date:** 2026-03-23
**Classification:** Root Cause Analysis — Closed

---

## 1. Executive Summary

The 28-dim V3 QLSTM model achieved in-sample WR 42–54% (2023–2025) but collapsed to
WR=26.4%, ROI=-54% on 2026 Q1 out-of-sample data. Root cause: **curse of dimensionality
compounded by non-stationary statistical features**. This document establishes a hard
limit of **15 features** for all future models.

## 2. Timeline

| Date | Event | Metric |
|------|-------|--------|
| 2025-12 | V3 features finalized (28-dim) | In-sample WR 54% |
| 2026-01 | Deployed to paper trading | Initial WR ~40% |
| 2026-02 | Performance degradation noticed | WR dropping to ~32% |
| 2026-03-18 | OOS evaluation on 2026 Q1 | WR=26.4%, ROI=-54% |
| 2026-03-18 | Pivot to 13-dim structural features | Decision point |

## 3. Root Cause: Curse of Dimensionality

### 3.1 The Arithmetic

With 28 features and T≈4,000 1h bars/year:
- **Sample-to-feature ratio:** 4000/28 = 143 (seemingly adequate)
- **But effective samples for trading signals:** ~200 trades/year
- **Effective ratio:** 200/28 = **7.1** (critically low)

The Hughes phenomenon (Hughes 1968) states that classification accuracy peaks at
p* ≈ √(2n) features, where n = effective training samples:
```
p* = √(2 × 200) ≈ 20 features
```
At 28 features, we were **40% above the optimal dimensionality**.

### 3.2 Variance Decomposition

For a model with p features and n effective samples, the expected excess risk:
```
E[R_excess] ≈ σ² × p/n + bias²
```
Going from 13-dim to 28-dim:
```
ΔVariance ≈ σ² × (28-13)/200 = σ² × 0.075
```
This 7.5% increase in variance component directly translates to ~7-10% WR degradation,
consistent with the observed drop from 36.8% (baseline) to 26.4% (V3 OOS).

### 3.3 Non-Stationary Features Amplified the Problem

The 15 extra features in V3 (beyond the 13 structural) included:
- **Hurst R/S (feat[27]):** Upward biased by +6-12% on finite windows (Lo 1991). Collapsed
  when BTC regime shifted to H≈0.50 in 2026 Q1.
- **Lag-1 autocorrelation γ(1) (feat[28]):** Unstable on 1h timeframe.
- **Price entropy / purity proxy (feat[29]):** Histogram-dependent, not robust to distribution shift.

These features had high in-sample explanatory power (R² contribution ~18%) but zero
out-of-sample predictive value. The model allocated parameter capacity to fitting
noise patterns that vanished in new data.

## 4. Comparison: 13-dim vs 28-dim

| Metric | 13-dim Structural | 28-dim V3 | Delta |
|--------|:-----------------:|:---------:|:-----:|
| In-sample WR | 41.8% | 54.0% | +12.2pp (overfitting signal) |
| OOS 2026 Q1 WR | 36.8% | 26.4% | **-10.4pp** |
| OOS ROI | +3.18% | -54% | -57.18pp |
| Effective sample/feature | 15.4 | 7.1 | -54% |
| Feature stationarity (ADF pass rate) | 12/13 (92%) | 18/28 (64%) | -28pp |

The 12.2pp in-sample WR gain from extra features was entirely illusory — a textbook
case of the bias-variance tradeoff favoring low-dimensional models.

## 5. Hard Limit Established

**Maximum feature count for any production model: 15**

Derivation:
```
n_eff = 200 trades/year (conservative)
p* = √(2 × 200) ≈ 20 (Hughes optimal)
Safety margin: 0.75 × 20 = 15 (25% buffer for collinearity)
```

Additionally:
- All features must pass ADF stationarity test (p < 0.01)
- All features must have alpha half-life > 180 bars
- Pairwise |ρ| < 0.7 (to preserve effective dimensionality)

## 6. Lessons Learned

1. **In-sample WR gain ≠ alpha discovery.** The 12.2pp in-sample improvement was the
   model memorizing noise, not discovering structure.
2. **Statistical features are self-defeating.** If a statistical pattern (Hurst, autocorrelation)
   is discoverable by a simple estimator, it is discoverable by the market. Structural/causal
   features (funding rate, liquidation cascades) have longer half-lives because they model
   forced behavior, not informational edge.
3. **Dimensionality kills before complexity helps.** Adding 15 features increased model
   capacity by ~115% but effective information by <5%. Net effect: negative.
4. **The OOS collapse was predictable.** A simple ADF pre-screen would have rejected 10/28
   features. We did not have a validation framework. We do now (see `docs/validation_framework_spec.md`).

## 7. References

- Hughes, G. (1968). "On the Mean Accuracy of Statistical Pattern Recognizers." *IEEE Trans. Information Theory*, 14(1).
- Lo, A. (1991). "Long-Term Memory in Stock Market Prices." *Econometrica*, 59(5), pp. 1279-1313.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 8: Feature Importance.
- Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *J. Portfolio Management*, 40(5).
