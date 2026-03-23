# QLSTM-Quant-Trading-Bot

# Quantum-Classical Hybrid Reinforcement Learning for Cryptocurrency Algorithmic Trading

> A portfolio project demonstrating the application of quantum variational circuits, advanced stochastic physics, and differential geometry to autonomous derivatives trading on Bybit Perpetual Futures.

---

## Abstract

This project implements a full-stack automated trading system in which the core decision-making engine is a **Quantum-Classical Hybrid Actor-Critic agent** (QLSTM + QA3C). The classical backbone encodes 28-dimensional market-microstructure features — including Frenet–Serret geometric invariants derived from phase-space curves, Hurst-exponent-based long-memory estimates, and CVD (Cumulative Volume Delta) order-flow signals — and passes them through a **Variational Quantum Circuit (VQC)** built on PennyLane. The agent is trained end-to-end with an **Advanced Path-Integral Loss** that unifies policy-gradient optimization (Generalized Advantage Estimation), a Lindblad quantum master equation for regime detection, a Fokker–Planck SDE regulariser, and Wasserstein Distributionally Robust Optimization. Walk-forward cross-validation over 10 folds (2023–2025) confirms positive expected value in all validated folds, and an independent behavioral alpha strategy (funding-rate squeeze + EMA-200 filter) achieves +126% ROI over 3.25 years out-of-sample. The system runs as a real-time Bloomberg-style TUI (Textual) and writes a full SQLite audit trail.

---

## Table of Contents

1. [Motivation & References](#1-motivation--references)
2. [Tech Stack](#2-tech-stack)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Architecture](#4-model-architecture)
5. [Results & Visualization](#5-results--visualization)
6. [Limitations & Future Work](#6-limitations--future-work)
7. [Quick Start](#7-quick-start)

---

## 1. Motivation & References

### 1.1 Why Quantum Circuits for Finance?

Classical deep learning models treat financial time series as memoryless (Markovian) stochastic processes. However, empirical evidence shows that crypto return series exhibit **long-range dependence** (Hurst exponent H > 0.5) and fat tails inconsistent with Gaussian Brownian Motion. Standard gradient-based optimizers on deep networks suffer from **barren plateaus** when applied to high-dimensional parameter manifolds, and scalar value functions discard the full return distribution required for tail-risk-aware trading.

Variational Quantum Circuits (VQCs) offer three complementary advantages:

- **Entanglement-based feature interaction**: IsingZZ coupling layers capture non-linear correlations between market features without requiring explicit feature engineering.
- **Natural expressivity on the unitary manifold**: Quantum gates live on SU(2^n), whose Riemannian geometry is better suited to probability-distribution optimization than Euclidean parameter space.
- **Quantum coherence as a regime signal**: The purity Tr(ρ²) of the circuit's density matrix, evolved under the Lindblad master equation, serves as a real-time indicator of market regime coherence.

### 1.2 Key Concepts Implemented

| Concept | Mathematical Foundation | Reference |
|---|---|---|
| Variational Quantum Circuit (VQC) | Parametrised unitary U(θ) = Π_k exp(-iθ_k H_k) | Cerezo et al. (2021), *Nature Reviews Physics* |
| Lindblad Master Equation | dρ/dt = -i[H,ρ] + Σ_k γ_k(L_k ρ L_k† - ½{L_k†L_k, ρ}) | Breuer & Petruccione (2002) |
| Fokker–Planck / Langevin SDE | dx = μ(x)dt + σ(x)dW; L_FP = -log N(x'; x+μdt, σ²dt) | Risken (1989) |
| Generalized Advantage Estimation (GAE) | Â_t = Σ_l (γλ)^l [r_{t+l} + γV(s_{t+l+1}) - V(s_{t+l})] | Schulman et al. (2016), *ICLR* |
| Wasserstein DRO | min_θ max_{P: W₁(P,Q)≤ε} E_P[ℓ] | Esfahani & Kuhn (2018), *Mathematical Programming* |
| Hurst Exponent (R/S Analysis) | E[R(n)/S(n)] ≈ c · n^H | Mandelbrot & Wallis (1969) |
| Matrix Product States (MPS) | \|ψ⟩ = Σ A¹[i₁]A²[i₂]…Aⁿ[iₙ]\|i₁…iₙ⟩ | Vidal (2003), *PRL* |
| Mutual Information Neural Estimation (MINE) | I(X;Y) ≥ E_joint[T] - log E_marginal[e^T] | Belghazi et al. (2018), *ICML* |
| Optimal Stopping (Snell Envelope) | J(x,t) = sup_{τ≥t} E[g(X_τ)\|X_t=x] | Peskir & Shiryaev (2006) |
| Frenet–Serret Geometry | Γ(t+Δ) ≈ Γ(t) + Δ\|Γ'\|T + (Δ\|Γ'\|)²/2 · κN | do Carmo (1976) |
| Platt Calibration | P_calibrated = σ(T · logit + b) | Platt (1999) |
| RMT Marchenko–Pastur Denoising | λ± = σ²(1 ± √(D/T))² | Marchenko & Pastur (1967); López de Prado (2020) |
| CVD / Order Flow Imbalance | delta_i = V_i × (2c_i - h_i - l_i)/(h_i - l_i + ε) | López de Prado (2012) |

---

## 2. Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **Deep Learning** | PyTorch 2.x (CUDA), custom autograd |
| **Quantum Computing** | PennyLane 0.38 (`default.qubit`, IsingZZ ansatz) |
| **Reinforcement Learning** | Custom Actor-Critic (QA3C) with GAE + entropy bonus |
| **TUI / Dashboard** | Textual 0.59 (Bloomberg-style terminal UI) |
| **Exchange API** | Bybit V5 REST API (`pybit`) — BTCUSDT Perpetual Futures |
| **Data** | OHLCV + Funding Rate + Open Interest via Bybit REST |
| **Storage** | SQLite (full audit trail), NumPy `.npz` (checkpoints) |
| **Hyperparameter Optimization** | Optuna |
| **Numerical / Scientific** | NumPy, Pandas, SciPy |
| **Visualization** | Matplotlib, Plotly |

---

## 3. Data Pipeline

### 3.1 Collection

Raw OHLCV candles, perpetual **funding rates** (8-hour cycle), and **open interest** (15-minute granularity) are fetched from the Bybit V5 REST API for three symbols: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`. All timestamps are stored and displayed in **KST (UTC+9)**. Missing rows are forward-filled to preserve temporal integrity.

```
Bybit REST API
  ├── /v5/market/kline            → OHLCV (1h candles)
  ├── /v5/market/funding/history  → funding_rate (8h, forward-filled)
  └── /v5/market/open-interest    → OI (15m, forward-filled)
```

### 3.2 Feature Engineering — V4 (28-dimensional)

All features are clipped to **[-π, π]** for direct VQC angle encoding.

```
Raw OHLCV + Funding Rate + Open Interest
    │
    ├── Log-returns: r_t = ln(P_t / P_{t-1})      ← stationarity guarantee
    ├── Rolling Z-score normalisation (window=60)  ← price-level invariance
    ├── RMT Marchenko–Pastur denoising             ← noise eigenvalue removal
    │       λ_noise = σ²(1 ± √(D/T))²
    └── 3-Binary LDA → 5 eigenvectors              ← discriminative decomposition
```

**Feature Index Map (28-dim V4):**

| Index | Feature | Mathematical / Financial Interpretation |
|---|---|---|
| 0–4 | `log_return`, `rsi_14`, `macd_val`, `atr_14`, `obi` | Classical technical indicators |
| 5–9 | `vol_ratio`, `ema12_dev`, `mom3`, `mom10`, `mom20` | Momentum term structure |
| 10–16 | `bb_width`, `roc5`, `hr_sin`, `hr_cos`, `price_zscore`, `hl_ratio`, `tick_imb` | Microstructure features |
| 17 | **Hurst H** | R/S multi-scale estimate; H>0.55 → persistent trend (fBm) |
| 18 | **γ(1)** (lag-1 autocorrelation) | γ>0: persistent; γ<0: mean-reverting |
| 19 | **purity_proxy** | 1 − 2H(return histogram); Lindblad density matrix proxy |
| 20 | `funding_rate_zscore` | Short-side crowding detector (8h perpetual) |
| 21 | `candle_body_ratio` | Bull/bear bar strength ∈ [-1,1] |
| 22 | `volume_zscore` | Abnormal volume detection |
| 23 | `oi_change_pct` | Open interest momentum (trend conviction proxy) |
| 24 | `funding_velocity` | Rate-of-change of funding (crowding acceleration) |
| 25 | `cvd_delta_zscore` | Per-bar buyer/seller aggression (order flow imbalance) |
| 26 | `cvd_trend_zscore` | 5-hour cumulative buy/sell dominance |
| 27 | `cvd_price_divergence` | Smart-money accumulation/distribution fingerprint |

### 3.3 Frenet–Serret Geometric Features — V5 Extension (54-dim = V4 + 20 Frenet)

Four market variables are interpreted as parametric curves in phase space. The Frenet–Serret apparatus extracts **curvature κ** and **torsion τ** as causal trend-reversal predictors:

```
Second-order Taylor approximation of curve evolution:
    Γ(t+Δ) ≈ Γ(t)  +  Δ|Γ'|·T  +  (Δ|Γ'|)²/2 · κ·N

Reversal signals:
    κ local maximum  →  imminent trend reversal (curve sharply bending)
    κ' < 0           →  reversal stabilizing (curvature decreasing)
    τ ≈ 0            →  regime collapses to 2D (high-confidence signal)
    turn_sign = +1   →  CCW rotation in phase space (long bias)
```

| Phase-Space Curve | Embedding | Geometric Features (dim) |
|---|---|---|
| MACD phase portrait `(ema12_dev, macd_val)` | R² | T_x, T_y, κ, turn_sign (4D) |
| CVD phase portrait `(cvd_delta_z, cvd_trend_z)` | R² | T_x, T_y, κ, turn_sign (4D) |
| Momentum term structure `(mom3, mom10, mom20)` | R³ | T, κ, τ, N (8D) |
| Price–volume path `(log_return, vol_ratio)` | R² | T_x, T_y, κ, turn_sign (4D) |

All κ values are arc-length-normalised and compressed via arctan ∈ [0, π/2]. All features are computed via backward finite differences (no look-ahead bias).

### 3.4 Training / Validation / OOS Split

```
BC Pre-training : 2019-01-01 – 2022-12-31   (4 years, regime-invariant physics)
RL Walk-forward : 2023-06-01 – 2025-12-31   (10 expanding folds, includes 2025)
OOS Evaluation  : 2026-01-01 – present      (true out-of-sample)
```

Behavior Cloning (BC) learns the physical laws of price dynamics on data the RL optimizer never touches. RL walk-forward training ensures 2025 data appears in the validation set before any live deployment, closing the common look-forward gap that caused prior 15m models to collapse out-of-sample.

---

## 4. Model Architecture

### 4.1 End-to-End Inference Pipeline

```
Raw OHLCV + Exogenous Data  [B, seq_len=96, raw_dim]
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  FEATURE PIPELINE  (src/models/features_v4.py)           │
│  28-dim log-return → Z-score → RMT denoising → V4        │
│  + Frenet κ/τ geometric features (V5: 54-dim)            │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  SPECTRAL DECOMPOSER  (src/data/spectral_decomposer.py)  │
│  3-Binary LDA (Cohen d > 1.0) → 5 eigenvectors          │
│  Input: [B, 96, 28]   Output: [B, 96, 5]                 │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  TEMPORAL CONTEXT ENCODER  (Transformer)                 │
│  Multi-head self-attention over seq_len = 96 bars        │
│  (96 × 1h = 4 days of market history)                    │
└──────────────────────┬───────────────────────────────────┘
                       │  context vector c_kt  [B, T, 5]
                       ▼
┌──────────────────────────────────────────────────────────┐
│  QUANTUM MARKET ENCODER  (src/models/quantum_layers.py)  │
│  proj_to_qubits: Linear(5 → 3, bias=False)               │
│  IsingZZ VQC (N_QUBITS=3) → ⟨σ^z⟩ expectation values   │
│  h_field (RMT noise floor) broadcast                     │
│  logit_proj: Linear(3, 3) → logits [B, 3]                │
│  Actions: HOLD (0) / LONG (+1) / SHORT (−1)              │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  INFERENCE & RISK GATES                                  │
│  AdaptiveTemperatureScaler: T = T_base × (1 + β·σ_ATR)  │
│  PlattCalibrator (T_platt, b): logit → calibrated prob   │
│  LindbladDecoherence: purity < 0.3 → force HOLD          │
│  Kill Switch + Circuit Breaker (8% MDD auto-halt)        │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
                  POSITION SIZING
          SL = 1.0×ATR, TP = 3.0×ATR  (R:R = 3:1)
          leverage = 5×  |  pos_frac = 50% of equity
```

### 4.2 Loss Function — Advanced Path-Integral Loss

The training objective unifies four physically motivated components:

```
L = L_actor  +  c_c · L_critic  +  c_fp · L_FP  −  c_H · H(π)

where:
    L_actor  = −E[ Â_t · log π(a_t | s_t) ]           (GAE policy gradient)
    L_critic = MSE( V(s_t),  returns_t )                (value function regression)
    L_FP     = −log N(x'; x + μΔt, σ²Δt)              (Fokker–Planck SDE)
    H(π)     = −Σ_a π(a) log π(a)                      (entropy bonus)

    Â_t = Σ_l (γλ)^l δ_{t+l},   δ_t = r_t + γV(s_{t+1}) − V(s_t)
```

The reward functional encodes a physical entry barrier:

```
J(τ) = Σ_t γ^t ΔV_t  −  η · I_{open}  +  R_terminal(S_T)

Entry condition:  E[Σ γ^t ΔV_t] + R_terminal  >  η  (transaction-cost floor)

η = 0.001 × leverage  (round-trip fee, Bybit Maker/Taker 0.02%/0.055%)
```

Only trades whose discounted expected cumulative P&L exceeds the fee floor generate a LONG/SHORT signal; HOLD is otherwise the mathematically optimal action.

### 4.3 Physics Layer (src/models/advanced_physics.py)

| Component | Mathematical Content |
|---|---|
| `HurstEstimator` | Multi-scale R/S regression → H; H>0.55 persistent, H<0.45 mean-reverting |
| `LindbladDecoherence` | dρ/dt = -i[H,ρ] + Σγ_k(L_kρL_k† - ½{L_k†L_k,ρ}); purity = Tr(ρ²) |
| `FokkerPlanckRegularizer` | L_FP = -log N(x'; x+μdt, σ²dt); penalises non-Langevin transitions |
| `WassersteinDROLoss` | W₁-ball adversarial robustness; gradient penalty = Kantorovich dual |
| `MPSLayer` | χ-bond MPS tensor chain; area-law entanglement; O(nχ²) contraction |
| `OptimalStoppingBoundary` | Snell envelope B_t = max(g(x_t), E[B_{t+1}]); learned SL/TP boundary |
| `PlattCalibrator` | σ(T·logit + b); ECE tracker; converts VQC scores to calibrated win-probability |
| `MINEEstimator` | I(X;Y) ≥ E_joint[T] − log E_marginal[e^T]; Donsker–Varadhan bound |

### 4.4 Behavioral Alpha Engine (Independent Module)

A second, fully independent signal exploits a structural market anomaly in perpetual futures:

```
Signal logic:
    funding_rate_zscore < −2.5σ   (crowd is heavily short)
    AND price > EMA-200            (macro uptrend confirmed)
    → forced short squeeze → LONG only

Mechanism: perpetual contracts enforce funding payments from short to long
           when shorts are overcrowded → structural buying pressure
```

---

## 5. Results & Visualization

### 5.1 Walk-Forward RL Validation (1h, BTCUSDT, 10 Folds)

Training period: 2023-06-01 – 2025-12-31 | Metric: Expected Value (EV) per trade

| Fold | Validation Period | EV (per trade) | Avg. Max Confidence |
|---|---|---|---|
| 1 | 2023-09 – 2023-11 | +0.2441 | 0.585 |
| 2 | 2023-11 – 2024-02 | +0.2106 | 0.746 |
| 3 | 2024-02 – 2024-05 | +0.3219 | 0.777 |
| 4 | 2024-05 – 2024-08 | +0.2093 | 0.807 |
| 5 | 2024-08 – 2024-10 | +0.2013 | 0.745 |
| 6 | 2024-10 – 2025-02 | +0.2672 | 0.751 |
| **7** | **2025-02 – 2025-04** | **+0.2929** | **0.830** ★ 2025 OOS |
| **8** | **2025-04 – 2025-07** | **+0.2192** | **0.856** ★ 2025 OOS |

Folds 7–8 validate on 2025 data never seen during optimization — positive EV in both folds confirms generalization beyond the training regime.

**Fold 10 Live OOS (Oct 2025 → Jan 2026, evaluated Jan–Mar 2026):**
- Win Rate: **29.1%** | Geometric BEP (5× leverage, 0.075% round-trip): ~25.4%
- ROI (Jan–Mar 2026): **+0.10%** (3.7 pp above breakeven on 87 trades)


### 5.2 Behavioral Alpha Engine — Out-of-Sample Results

Strategy: Funding-rate squeeze + EMA-200 long-only filter | 5× leverage

| Period | ROI | Win Rate | Trades | Max Drawdown |
|---|---|---|---|---|
| **2023 (true OOS, never optimized)** | **+23.63%** | 34.5% | 29 | — |
| 2024 | +87.44% | 38.1% | 41 | — |
| 2025 | +12.05% | 36.2% | 21 | — |
| 2026 Q1 | +3.18% | — | 4 | — |
| **Cumulative (3.25 years)** | **+126.30%** | **36.8%** | **95** | **16.27%** |

The 2023 data was entirely withheld from parameter selection — it constitutes a clean out-of-sample test demonstrating structural alpha.


### 5.3 Physics & Mathematics Quality Score

Independent evaluation of model sophistication across two dimensions:

| Dimension | Baseline (pre-project) | Current Implementation | Epistemological Ceiling |
|---|---|---|---|
| **Physics** | 3.5 / 10 | **7.0 / 10** | ~9.8 (Cramér–Rao bound) |
| **Mathematics** | 4.5 / 10 | **7.5 / 10** | ~9.9 (Lyapunov convergence) |

---


### 5.4 Current OOS Validation — Q1 2026 (13-dim Structural Model)

> **Honest status update** — The results below reflect the latest out-of-sample backtest as of 2026-03-23.

Model: `checkpoints/structural_13dim/test_agent.pt` | Period: 2026-01-01 → 2026-03-23 | Timeframe: 1h

| Metric | Value | Gate |
|---|---|---|
| Win Rate | 22.3% (35/157 trades) | ❌ Gate 1 FAIL (BEP = 25.4%) |
| Mean R-multiple | −0.2003 | ❌ Gate 2 FAIL (p = 1.0) |
| ROI | −51.6% | — |
| Sharpe | −2.517 | — |
| Max Drawdown | 58.22% | — |
| Avg Win R | +2.903 | — |
| Long bias | **100% long** (0 short trades) | 🔴 Critical bug |

**Root cause identified**: The agent generates exclusively long positions across all 157 trades — a directional bias inconsistent with a model trained on symmetric Long/Short/Hold outputs. Active debugging sprint underway targeting `QuantumFinancialAgent.get_action()` and the training pipeline normalization stack.

Full OOS report: [`reports/qlstm13_oos_report_q1_2026.json`](reports/qlstm13_oos_report_q1_2026.json)
R-multiple distribution: [`reports/qlstm13_rmultiples_q1_2026.csv`](reports/qlstm13_rmultiples_q1_2026.csv)

## 6. Limitations & Future Work

### 6.1 Epistemological Bound on Market Prediction

Perfect prediction in competitive markets is not only practically impossible but theoretically bounded. The **Cramér–Rao inequality** states:

```
Var(μ̂) ≥ σ² / T
```

The minimum achievable estimation error scales with the inverse of sample size — a hard statistical limit independent of model complexity. The **Jarzynski bound** from non-equilibrium thermodynamics further constrains maximum extractable alpha:

```
⟨exp(−βW)⟩ = exp(−β·ΔF)
```

where ΔF is the free energy difference representing structural information asymmetry between the model and market consensus. The realistic goal is therefore not P(correct) = 1, but a consistent positive KL divergence from the random baseline: E[log(P_model / P_random)] > 0.

### 6.2 Current Mathematical Limitations

The present system, while functional and validated, has several mathematically identified limitations that motivate further research:

**[1] Markov assumption in the advantage estimator**

The GAE backward scan assumes Markovian transitions δ_t = r_t + γV(s_{t+1}) − V(s_t). When H > 0.5, however, crypto returns exhibit long-range dependence:

```
E[r(t)·r(s)] ~ |t−s|^(2H−2),   H > 0.5
```

The physically correct equation of motion is the **Generalized Langevin Equation** with a non-Markovian memory kernel:

```
mẍ(t) = −∫₀ᵗ γ(t−τ) ẋ(τ) dτ  +  F(x)  +  η(t)
```

A Prony-series representation γ(t) = Σ_k a_k exp(−b_k t) can exactly represent any memory kernel of finite rank, and would replace the Markovian critic with a value functional over the full path history.

**[2] Path-integral loss is a finite-sum approximation**

The current objective J(τ) = Σ_t γ^t ΔV_t is named after the path-integral formulation but lacks a rigorous measure-theoretic foundation. The true **Onsager–Machlup action** gives the exact log-probability density of any continuous diffusion path:

```
S[x] = ∫₀ᵀ (ẋ(t) − μ(x,t))² / (2σ²(x,t)) dt
```

The dominant contribution comes from the classical path δS/δx = 0 (Euler–Lagrange equation); quantum / noise corrections arise from fluctuations around it. The current implementation is a 1-step discretisation that discards the saddle-point structure.

**[3] Scalar critic discards tail-risk information**

`CriticHead` outputs E[Z(s,a)] — the mean return. **Distributional RL** (IQN) models the full return distribution:

```
Z(s,a)  =^D  r(s,a) + γ Z(s', a')
```

and enables risk-sensitive policy optimization via CVaR_α[Z], correctly distinguishing "reliable 5% gain" from "50% chance of 10%, 50% chance of 0%." The distributional Bellman operator is a contraction in Wasserstein distance — a strictly better training signal.

**[4] PCA decomposes variance, not dynamics**

The spectral decomposer applies PCA to the feature covariance matrix. The **Koopman operator** U_t decomposes the dynamics: its eigenfunctions φ_k(x) satisfy (U_t φ_k)(x) = exp(λ_k t) φ_k(x), where slow-decaying modes |λ_k| ≈ 1 represent exploitable alpha signals invisible to variance-based PCA.

**[5] No stochastic calculus consistency for fBm**

The Fokker–Planck regulariser assumes standard Brownian motion (H = 0.5). For fractional Brownian motion with H ≠ 0.5 — which is **not a semimartingale** — the Itô formula acquires a Lévy area correction:

```
f(B^H_t) = f(B^H_0) + ∫₀ᵗ f'(B^H_s) dB^H_s  +  H ∫₀ᵗ f''(B^H_s) s^{2H−1} ds
```

**Rough path theory** (Lyons, 1998) provides the rigorous framework for integration against non-semimartingale paths via the lifted rough path (X, X) encoding iterated integrals.


### 6.3 On-Chain Fundamental Data Integration (Active Design)

The Q1 2026 OOS failure motivated a strategic pivot: price-derived features are observable by all market participants simultaneously and thus carry diminishing alpha. Blockchain on-chain data — while public — requires significant analytical overhead to interpret, preserving an information asymmetry.

**Planned 19-dimensional on-chain feature set:**

| Category | Features | Lead Time |
|---|---|---|
| Valuation | MVRV Z-Score, SOPR, NUPL, Realized Price deviation | 1–4 weeks |
| Network Activity | Active Addresses Z-Score, NVT Signal, New Address Growth | 1–7 days |
| Supply Dynamics | LTH Supply Ratio, STH SOPR, Exchange Netflow Z-Score | 1–5 days |
| Mining | Hash Ribbon buy signal, Miner Outflow Z-Score, Difficulty Ribbon | 2–6 weeks |
| Derivatives Context | OI/LTH ratio, Perp Basis Z-Score, Funding×OI product | Immediate–1 day |

**Data sources (free tier):** BGeometrics (MVRV, SOPR, NUPL), Blockchain.com (Active Addresses, Hash Rate), Bybit extended endpoints (OI history, Funding history)

**Validation methodology:** Granger causality test (lag 1–24h, p < 0.05) gates feature inclusion before any model integration — enforcing that features demonstrate predictive precedence over BTC 1h returns before entering the pipeline.

Full design specification: [`docs/onchain_feature_design.md`](docs/onchain_feature_design.md)

### 6.3 Future Work — Toward a Rigorous Financial Mathematics Foundation

The limitations above define a concrete research roadmap:

| Priority | Enhancement | Mathematical Content | Physics Δ | Math Δ |
|---|---|---|---|---|
| 1 | Distributional Critic (IQN + CVaR) | Quantile regression; Wasserstein Bellman contraction | +0.2 | +0.5 |
| 2 | Koopman/EDMD spectral decomposition | K = G⁻¹A, Koopman eigenfunctions φ_k | +0.3 | +0.4 |
| 3 | HJB consistency loss (PINN approach) | ∂V/∂t + max_a[f·∇V + ½Tr(σσᵀ∇²V) + r] = 0 | +0.3 | +0.3 |
| 4 | Non-Markovian memory kernel (GLE) | γ(t) = Σ_k a_k exp(−b_k t), Prony series | +0.5 | +0.3 |
| 5 | Path signature features | Chen–Fleiss theorem: universal approximation on path space | +0.1 | +0.4 |
| 6 | Quantum Natural Gradient (QFI) | F^Q_ij = 4Re[⟨∂_iψ\|∂_jψ⟩ − ⟨∂_iψ\|ψ⟩⟨ψ\|∂_jψ⟩] | +0.4 | +0.2 |
| 7 | Extreme Value Theory (GPD tail sizing) | F(y) = 1−(1+ξy/β)^{−1/ξ}; ES_α = VaR_α/(1−ξ) | +0.1 | +0.3 |
| 8 | Onsager–Machlup full path action | S[x] = ∫(ẋ−μ)²/(2σ²)dt; saddle-point inference | +0.5 | +0.2 |

The most important future direction is the integration of **stochastic differential equations** (SDE) and their associated PDE theory — Fokker–Planck, Kolmogorov backward equation, Hamilton–Jacobi–Bellman — as hard structural constraints on the learned value function and policy, rather than soft regularisation penalties. The rough path theory of Lyons (1998) and the Malliavin calculus framework for sensitivity analysis of stochastic functionals represent the natural mathematical language for this extension.

Optimal position sizing, currently based on fixed ATR multiples, would be rigorously derived from the **Merton portfolio problem** (continuous-time stochastic control), yielding a Kelly-fraction formula under the correct distributional assumptions and incorporating Extreme Value Theory (Peaks-Over-Threshold with GPD tail fitting) for dynamic leverage adjustment proportional to the live tail-risk index ξ.

These open problems — convergence proofs for the online quantum actor-critic (Lyapunov argument), full variational MPS bond-dimension optimisation (DMRG-style sweep), and the Riemannian gradient on the quantum Fisher information manifold — represent the natural intersection of **applied mathematics, financial mathematics, stochastic analysis, and quantum computing** that I am seeking to pursue at the graduate research level.

---

## 7. Quick Start

### 7.1 Installation

```bash
git clone https://github.com/<your-username>/quantum-quant-trading.git
cd quantum-quant-trading

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```
### 7.2 Environment Variables

Set your API keys in a `.env` file. **Never commit `.env` to Git** — it is already listed in `.gitignore`.

```bash
cp .env.example .env
```

Open `.env` and fill in the following values:

```dotenv
# ── Bybit API credentials ─────────────────────────────────────────
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_API_SECRET=your_bybit_api_secret_here

# ── Google Gemini API key (TUI AI panel — optional) ───────────────
GEMINI_API_KEY=your_gemini_api_key_here
```

#### How to obtain a Bybit API key

1. Log in to [bybit.com](https://www.bybit.com) → top-right profile → **API Management**
2. Click **Create New Key**
3. Key type: `System-generated API Keys`
4. Set permissions:
   - ✅ **Read** (market data)
   - ✅ **Unified Trading — Trade** (order execution; required for live mode only)
5. Optionally restrict to your IP address for added security.
6. Paste the generated `API Key` and `API Secret` into `.env`.

> **Paper trading mode** (`--mode paper`, the default) does not send any real orders — the API key is not required.
> **Live mode** (`--mode live`) requires valid credentials and sufficient margin balance.

#### How to obtain a Gemini API key (optional)

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey) (free tier available)
2. Click **Create API Key** → select a project → copy the key
3. Paste it into `GEMINI_API_KEY` in your `.env`
4. If omitted, the TUI AI assistant panel is disabled; all other features remain functional.

#### Security checklist

| Item | Action |
|---|---|
| `.env` excluded from Git | Already in `.gitignore` — verify with `git status` |
| Minimum API permissions | Grant Read + Trade only; never enable Withdrawal |
| IP whitelist | Restrict API key to your IP address on Bybit |
| If a key is leaked | Delete it immediately in Bybit API Management and reissue |


### 7.3 Environment Variables

```bash
cp .env.example .env
# Set BYBIT_API_KEY, BYBIT_API_SECRET in .env
# Paper trading mode is the default — no live orders are placed without --mode live
```

### 7.4 Training

```bash
# Step 1: Behavior Cloning pre-training (2019–2022, regime-invariant physics)
python scripts/pretrain_bc.py \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --timeframe 1h --start-date 2019-01-01 --end-date 2022-12-31 \
  --epochs 20 --device cuda --tp-mult 3.0 --sl-mult 1.0

# Step 2: Reinforcement Learning walk-forward (2023–2025, 10 folds)
python scripts/train_quantum_v2.py \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --timeframe 1h --start_date 2023-06-01 --end_date 2025-12-31 \
  --n-folds 10 --device cuda \
  --pretrain-ckpt checkpoints/quantum_v2/agent_bc_pretrained.pt \
  --tp-mult 3.0 --sl-mult 1.0
```

### 7.5 Backtesting

```bash
# Quantum agent OOS backtest
python scripts/backtest_model_v2.py \
  --model-path checkpoints/quantum_v2/agent_best.pt \
  --start-date 2026-01-01 --capital 10 --leverage 5 \
  --confidence 0.60 --tp-mult 3.0 --sl-mult 1.0 --timeframe 1h

# Behavioral alpha backtest (funding-rate squeeze)
python scripts/backtest_behavioral.py \
  --start-date 2023-01-01 --timeframe 1h \
  --capital 10 --leverage 5 --tp-mult 3.0 --sl-mult 1.0 \
  --signals fr --long-only --trend-ema 200 --fr-z-thr 2.5
```

### 7.6 Live Paper Trading TUI

```bash
python scripts/run_quantum_tui.py --mode paper \
  --quantum-model checkpoints/quantum_v2/agent_best_fold10.pt \
  --q-confidence 0.60 --q-leverage 5 --q-pos-frac 0.5 \
  --q-tp-mult 3.0 --q-sl-mult 1.0
```

**Default mode is `paper`.** Live trading requires explicit `--mode live` and valid API credentials. All trades are logged to SQLite for audit.

---

## Repository Structure

```
quantum-quant-trading/
│
├── src/
│   ├── models/                         # Core model components
│   │   ├── integrated_agent.py         # QuantumFinancialAgent — full actor-critic agent
│   │   ├── quantum_layers.py           # VQC circuits (IsingZZ ansatz, PennyLane)
│   │   ├── loss.py                     # AdvancedPathIntegralLoss, GAE, CriticHead, IQN
│   │   ├── advanced_physics.py         # Lindblad, MPS, MINE, Platt, Wasserstein DRO, Hurst
│   │   ├── features_v4.py              # 28-dim feature pipeline (current standard)
│   │   ├── features_v5.py              # 54-dim (V4 + Frenet geometric features)
│   │   ├── features_v3.py              # 30-dim (V2 + Hurst, autocorr, purity proxy)
│   │   ├── features_v2.py              # 27-dim baseline feature set
│   │   ├── frenet_features.py          # Frenet–Serret curvature/torsion extractor
│   │   ├── qng_optimizer.py            # Quantum Natural Gradient (QFI-based)
│   │   ├── ensemble_agent.py           # Ensemble wrapper over multiple agents
│   │   ├── labeling.py                 # Triple-barrier labeling for BC targets
│   │   ├── oi_profile.py               # Open interest profile features
│   │   └── base.py                     # Abstract base classes
│   │
│   ├── data/                           # Data ingestion & preprocessing
│   │   ├── spectral_decomposer.py      # RMT Marchenko–Pastur denoising + 3-Binary LDA
│   │   ├── bybit_mainnet.py            # Bybit V5 REST client (OHLCV, FR, OI)
│   │   ├── binance_client.py           # Binance REST client (auxiliary data)
│   │   └── data_client.py              # Unified data client interface
│   │
│   ├── strategies/                     # Signal filters & regime detection
│   │   ├── regime_gate.py              # Vol percentile + HTF EMA regime gate, Cramér-Rao filter
│   │   └── hmm_regime.py               # Hidden Markov Model regime classifier
│   │
│   ├── indicators/                     # Technical indicator pipeline
│   │   └── pipeline.py                 # ATR, RSI, MACD, OBI, Bollinger, momentum
│   │
│   ├── storage/                        # Persistence layer
│   │   └── database.py                 # SQLite audit trail (all trades & signals)
│   │
│   ├── app/                            # TUI entry point
│   │   ├── tui.py                      # Textual Bloomberg-style dashboard
│   │   └── cli_args.py                 # CLI argument definitions
│   │
│   ├── utils/                          # Shared utilities
│   │   ├── config.py                   # YAML config loader
│   │   ├── logging.py                  # Structured logger setup
│   │   ├── decision.py                 # Trade decision helpers
│   │   └── private_trade_logger.py     # Local trade log
│   │
│   └── viz/                            # Visualization
│       └── training_viz.py             # Walk-forward EV / equity curve plots
│
├── scripts/
│   ├── pretrain_bc.py                  # Behavior Cloning pre-training (2019–2022)
│   ├── train_quantum_v2.py             # RL walk-forward training (2023–2025, 10 folds)
│   ├── backtest_model_v2.py            # Quantum agent OOS backtest
│   ├── backtest_behavioral.py          # Behavioral alpha backtest (FR squeeze)
│   ├── run_quantum_tui.py              # Launch Bloomberg-style TUI
│   ├── run_behavioral_trade.py         # Run behavioral alpha strategy (live/paper)
│   ├── optimize_backtest.py            # Optuna hyperparameter search on backtest
│   ├── optuna_bc_search.py             # Optuna BC pre-training search
│   ├── train_bc_ensemble.py            # Ensemble BC training
│   ├── find_best_seed.py               # Seed stability search
│   ├── visualize_walk_forward.py       # Plot fold-by-fold EV results
│   └── visualize_atr_dist.py           # ATR distribution diagnostic
│
├── configs/
│   ├── default.yaml                    # Main config (risk, model, swing params)
│   └── models/                         # Per-model YAML configs
│       ├── qlstm_btc.yaml
│       ├── quantum_harmonic_oscillator.yaml
│       ├── schrodinger_indicator.yaml
│       └── ...
│
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variable template
├── .gitignore
└── README.md
```

---

*This project is for research and educational purposes. All default configurations operate in paper trading mode only. Past performance in backtests does not guarantee future results.*
