# QLSTM-Quant-Trading-Bot

> [한국어 README](README_KR.md)

# Quantum-Classical Hybrid Reinforcement Learning for Cryptocurrency Algorithmic Trading

> A portfolio project demonstrating the application of quantum variational circuits, advanced stochastic physics, and structural market-microstructure mechanisms to autonomous derivatives trading on Bybit Perpetual Futures.

---

## Abstract

This project implements a full-stack automated trading system in which the core decision-making engine is a **Quantum-Classical Hybrid Actor-Critic agent** (QLSTM + QA3C). The classical backbone encodes market-microstructure features and passes them through a **Variational Quantum Circuit (VQC)** built on PennyLane. The agent is trained end-to-end with an **Advanced Path-Integral Loss** unifying policy-gradient optimization (Generalized Advantage Estimation), Lindblad quantum master equation regime detection, a Fokker–Planck SDE regulariser, and Wasserstein Distributionally Robust Optimization.

**Research status (2026-03-24):** Walk-forward cross-validation (2023–2025) confirmed positive EV across all folds. However, Q1 2026 true OOS evaluation revealed a critical directional bias (100% LONG) traced to a zero-initialization bug in the EDMD spectral decomposer. An independent behavioral alpha strategy (funding-rate squeeze + EMA-200) achieves **+126% ROI (3.25 years OOS)**. Current work pivots to structural 13-dim mechanism-based features while the QLSTM bug is resolved.

---

## Table of Contents

1. [Motivation & References](#1-motivation--references)
2. [Tech Stack](#2-tech-stack)
3. [Architecture Evolution](#3-architecture-evolution)
4. [Pipeline A — Quantum QLSTM (V4/V5)](#4-pipeline-a--quantum-qlstm-v4v5)
5. [Pipeline B — Structural Mechanism Features (Current)](#5-pipeline-b--structural-mechanism-features-current)
6. [Results & Validation](#6-results--validation)
7. [Honest Limitations & Active Bugs](#7-honest-limitations--active-bugs)
8. [Quick Start](#8-quick-start)

---

## 1. Motivation & References

### 1.1 Why Quantum Circuits for Finance?

Classical deep learning models treat financial time series as memoryless (Markovian) stochastic processes. However, empirical evidence shows that crypto return series exhibit **long-range dependence** (Hurst exponent H > 0.5) and fat tails inconsistent with Gaussian Brownian Motion. Standard gradient-based optimizers suffer from **barren plateaus** when applied to high-dimensional parameter manifolds, and scalar value functions discard the full return distribution required for tail-risk-aware trading.

Variational Quantum Circuits (VQCs) offer three complementary advantages:

- **Entanglement-based feature interaction**: IsingZZ coupling layers capture non-linear correlations without explicit feature engineering.
- **Natural expressivity on the unitary manifold**: Quantum gates live on SU(2^n), whose Riemannian geometry is better suited to probability-distribution optimization than Euclidean space.
- **Quantum coherence as a regime signal**: The purity Tr(ρ²) of the circuit's density matrix, evolved under the Lindblad master equation, serves as a real-time market regime indicator.

### 1.2 Key Concepts Implemented

| Concept | Mathematical Foundation | Reference |
|---|---|---|
| Variational Quantum Circuit (VQC) | Parametrised unitary U(θ) = Π_k exp(-iθ_k H_k) | Cerezo et al. (2021), *Nature Reviews Physics* |
| Lindblad Master Equation | dρ/dt = -i[H,ρ] + Σ_k γ_k(L_k ρ L_k† - ½{L_k†L_k, ρ}) | Breuer & Petruccione (2002) |
| Fokker–Planck / Langevin SDE | dx = μ(x)dt + σ(x)dW; L_FP = -log N(x'; x+μdt, σ²dt) | Risken (1989) |
| Generalized Advantage Estimation (GAE) | Â_t = Σ_l (γλ)^l [r_{t+l} + γV(s_{t+l+1}) - V(s_{t+l})] | Schulman et al. (2016), *ICLR* |
| Wasserstein DRO | min_θ max_{P: W₁(P,Q)≤ε} E_P[ℓ] | Esfahani & Kuhn (2018) |
| Hurst Exponent (R/S Analysis) | E[R(n)/S(n)] ≈ c · n^H | Mandelbrot & Wallis (1969) |
| Matrix Product States (MPS) | \|ψ⟩ = Σ A¹[i₁]A²[i₂]…Aⁿ[iₙ]\|i₁…iₙ⟩ | Vidal (2003), *PRL* |
| Mutual Information Neural Estimation (MINE) | I(X;Y) ≥ E_joint[T] - log E_marginal[e^T] | Belghazi et al. (2018), *ICML* |
| Optimal Stopping (Snell Envelope) | J(x,t) = sup_{τ≥t} E[g(X_τ)\|X_t=x] | Peskir & Shiryaev (2006) |
| Frenet–Serret Geometry | Γ(t+Δ) ≈ Γ(t) + Δ\|Γ'\|T + (Δ\|Γ'\|)²/2 · κN | do Carmo (1976) |
| Platt Calibration | P_calibrated = σ(T · logit + b) | Platt (1999) |
| RMT Marchenko–Pastur Denoising | λ± = σ²(1 ± √(D/T))² | Marchenko & Pastur (1967) |

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
| **Validation** | Walk-forward cross-validation, bootstrap p-value (Gate 1/2) |
| **Numerical / Scientific** | NumPy, Pandas, SciPy |
| **Visualization** | Matplotlib, Plotly |

---

## 3. Architecture Evolution

This project documents the full development lifecycle including failures, pivots, and their root causes — an intentional choice for scientific transparency.

```
VERSION HISTORY
═══════════════════════════════════════════════════════════════════
V1  (2023-2025)  │ 15m timeframe │ 28-dim QLSTM + VQC
                 │ In-sample: WR 42–54%, EV +0.06~+0.12
                 │ OOS (2026 Q1): WR 26.4%, ROI −54%  ❌ FAILED
                 │ Cause: 15m timeframe — statistical patterns
                 │        self-destruct at inference time
                 │
V2  (2026-02)    │ 1h timeframe  │ 28-dim V4 + Frenet V5 + EDMD
                 │ Walk-forward (10 folds, 2023-2025): EV > 0 all
                 │ OOS (2026 Q1): WR 22.29%, ROI −51.6%  ❌ FAILED
                 │ Cause: EDMD _koop_vecs=zeros → input-invariant
                 │        model → 100% LONG bias (bug confirmed)
                 │
BASELINE         │ FR_LONG + EMA200 behavioral alpha
(parallel)       │ OOS (2023-2026): WR 36.8%, ROI +126%  ✓ ALPHA
                 │
V3 (CURRENT)     │ 1h timeframe  │ 13-dim structural mechanism
                 │ Pivot from statistical→structural features
                 │ Gate 1/2 validation pending after QLSTM fix
═══════════════════════════════════════════════════════════════════
```

### 3.1 Why the Pivot?

Statistical pattern features (RSI, MACD, Bollinger, Hurst) share a fundamental epistemological problem: **if a statistical pattern is reliably predictive, arbitrageurs eliminate it**. The moment a pattern enters widespread use, its signal-to-noise ratio collapses.

Structural mechanism features (funding rate, open interest, liquidation cascades, CVD) encode **the reason traders are forced to act**, not just the pattern of their actions. A funding rate squeeze forces short sellers to close positions regardless of market conviction — this is a causal mechanism, not a correlation.

---

## 4. Pipeline A — Quantum QLSTM (V4/V5)

### 4.1 Feature Engineering

```
Raw OHLCV + Funding Rate + Open Interest  [B, seq_len=96, raw_dim]
    │
    ├── Log-returns: r_t = ln(P_t / P_{t-1})      ← stationarity
    ├── Rolling Z-score normalisation (window=60)  ← price invariance
    ├── RMT Marchenko–Pastur denoising             ← noise removal
    │       λ_noise = σ²(1 ± √(D/T))²
    └── EDMD Koopman eigenvectors (5-dim)          ← predictable modes
```

**V4 Feature Index Map (28-dimensional):**

| Index | Feature | Interpretation |
|---|---|---|
| 0–4 | `log_return`, `rsi_14`, `macd_val`, `atr_14`, `obi` | Classical technicals |
| 5–9 | `vol_ratio`, `ema12_dev`, `mom3`, `mom10`, `mom20` | Momentum structure |
| 10–16 | `bb_width`, `roc5`, `hr_sin`, `hr_cos`, `price_zscore`, `hl_ratio`, `tick_imb` | Microstructure |
| 17 | **Hurst H** | R/S multi-scale; H>0.55 → persistent, H<0.45 → reverting |
| 18 | **γ(1)** lag-1 autocorrelation | γ>0: persistent; γ<0: mean-reverting |
| 19 | **purity_proxy** | 1−2H(return histogram); Lindblad density proxy |
| 20 | `funding_rate_zscore` | Short-side crowding detector |
| 21 | `candle_body_ratio` | Bull/bear bar strength ∈ [−1,1] |
| 22 | `volume_zscore` | Abnormal volume |
| 23 | `oi_change_pct` | OI momentum (trend conviction) |
| 24 | `funding_velocity` | Rate-of-change of funding crowding |
| 25 | `cvd_delta_zscore` | Per-bar buyer/seller aggression |
| 26 | `cvd_trend_zscore` | 5h cumulative buy/sell dominance |
| 27 | `cvd_price_divergence` | Smart-money accumulation fingerprint |

**V5 Extension — Frenet–Serret Geometric Features (54-dim = V4 + 20):**

Four market variables interpreted as parametric curves in phase space. Curvature κ and torsion τ act as causal trend-reversal predictors:

```
Second-order Taylor approximation:
    Γ(t+Δ) ≈ Γ(t)  +  Δ|Γ'|·T  +  (Δ|Γ'|)²/2 · κ·N

Reversal signals:
    κ local maximum  →  imminent trend reversal
    τ ≈ 0            →  regime collapses to 2D (high-confidence signal)
```

| Phase-Space Curve | Geometric Features |
|---|---|
| MACD phase portrait `(ema12_dev, macd_val)` | T_x, T_y, κ, turn_sign (4D) |
| CVD phase portrait `(cvd_delta_z, cvd_trend_z)` | T_x, T_y, κ, turn_sign (4D) |
| Momentum term structure `(mom3, mom10, mom20)` | T, κ, τ, N (8D) |
| Price–volume path `(log_return, vol_ratio)` | T_x, T_y, κ, turn_sign (4D) |

### 4.2 End-to-End Inference Pipeline

```
Features [B, 96, 28]
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  SPECTRAL DECOMPOSER  (src/data/spectral_decomposer.py)  │
│  EDMD Koopman eigenvectors → 5 slow modes               │
│  Input: [B, 96, 28]   Output: [B, 96, 5]                │
│  ⚠ KNOWN BUG: _koop_vecs=zeros (fix in progress)        │
└──────────────────────┬───────────────────────────────────┘
                       │  context vector c_kt  [B, T, 5]
                       ▼
┌──────────────────────────────────────────────────────────┐
│  TEMPORAL CONTEXT ENCODER  (Transformer)                 │
│  Multi-head self-attention over seq_len = 96 bars        │
│  (96 × 1h = 4 days of market history)                    │
└──────────────────────┬───────────────────────────────────┘
                       │
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
       leverage = 10×  |  pos_frac = 50%  |  eff = 5×
```

### 4.3 Loss Function — Advanced Path-Integral Loss

```
L = L_actor  +  c_c · L_critic  +  c_fp · L_FP  −  c_H · H(π)

    L_actor  = −E[ Â_t · log π(a_t | s_t) ]          (GAE policy gradient)
    L_critic = MSE( V(s_t),  returns_t )               (value function)
    L_FP     = −log N(x'; x + μΔt, σ²Δt)             (Fokker–Planck SDE)
    H(π)     = −Σ_a π(a) log π(a)                     (entropy bonus)

    Â_t = Σ_l (γλ)^l δ_{t+l},   δ_t = r_t + γV(s_{t+1}) − V(s_t)

Physical entry barrier:
    J(τ) = Σ_t γ^t ΔV_t  −  η · I_{open}  +  R_terminal(S_T)
    η = 0.001 × leverage  (Bybit round-trip fee: Maker 0.02% / Taker 0.055%)
```

### 4.4 Physics Layer (src/models/advanced_physics.py)

| Component | Mathematical Content |
|---|---|
| `HurstEstimator` | Multi-scale R/S → H; H>0.55 persistent, H<0.45 mean-reverting |
| `LindbladDecoherence` | dρ/dt = -i[H,ρ] + Σγ_k(L_kρL_k† - ½{L_k†L_k,ρ}); purity = Tr(ρ²) |
| `FokkerPlanckRegularizer` | L_FP = -log N(x'; x+μdt, σ²dt) |
| `WassersteinDROLoss` | W₁-ball adversarial robustness; Kantorovich dual gradient penalty |
| `MPSLayer` | χ-bond MPS tensor chain; area-law entanglement |
| `OptimalStoppingBoundary` | Snell envelope B_t = max(g(x_t), E[B_{t+1}]); learned SL/TP |
| `PlattCalibrator` | σ(T·logit + b); ECE tracker; VQC → calibrated win-probability |
| `MINEEstimator` | I(X;Y) ≥ E_joint[T] − log E_marginal[e^T]; Donsker–Varadhan |

---

## 5. Pipeline B — Structural Mechanism Features (Current)

### 5.1 Design Philosophy

> Statistical patterns describe *what* the market does. Structural mechanisms explain *why* participants are forced to act.

Features are selected for **causal mechanism**, not statistical correlation:

| Group | Features (13-dim) | Mechanism |
|---|---|---|
| **FR** | `fr_z`, `fr_trend` | Funding rate forces position liquidation via perpetual mechanics |
| **OI** | `oi_change_z`, `oi_price_div` | OI accumulation/divergence → crowded position detection |
| **Liq** | `liq_long_z`, `liq_short_z` | Estimated forced liquidation pressure (wick × volume proxy) |
| **CVD** | `cvd_trend_z`, `cvd_price_div`, `taker_ratio_z` | Actual order-flow aggression (taker buy/sell imbalance) |
| **Regime** | `ema200_dev`, `ema200_slope`, `vol_regime`, `vol_change` | Trend structure + volatility regime |

**Explicitly banned features:** RSI, MACD, Bollinger Bands, Hurst, Frenet-Serret, PCA/LDA output values — all statistical correlations that arbitrageurs can eliminate.

### 5.2 Behavioral Alpha Engine (Independent Baseline)

A structural signal exploiting the perpetual futures funding mechanism:

```
Signal: funding_rate_zscore < −2.5σ  (crowd is heavily short)
        AND price > EMA-200           (macro uptrend confirmed)
→ Forced short squeeze → LONG entry only

Mechanism: perpetual contracts enforce funding payments from shorts to longs
           when shorts are overcrowded → structural buying pressure regardless
           of the individual trader's view
```

```
Backtest:  python scripts/backtest_behavioral.py \
               --signals fr --long-only --trend-ema 200 --fr-z-thr 2.5
```

### 5.3 Pipeline Architecture (Current)

```
Raw OHLCV + Funding Rate + Open Interest  (Bybit REST, 1h candles)
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│  STRUCTURAL FEATURE BUILDER  (src/models/features_structural.py) │
│  13-dim mechanism-based features (no statistical patterns) │
│  FR / OI / Liquidation / CVD / Regime                    │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────┐
│  RULE-BASED SIGNAL LAYER                                  │
│  Baseline: FR_LONG + EMA200 filter                       │
│  Next: structural features → RL agent (in development)   │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────┐
│  GATE 1/2 VALIDATION  (backtesting/validation.py)        │
│  Gate 1: WR > 25.4% (geometric BEP at 5× eff leverage)  │
│  Gate 2: bootstrap p < 0.05 on r-multiple distribution  │
│  No capital allocation until BOTH gates pass             │
└───────────────────────────────────────────────────────────┘
```

---

## 6. Results & Validation

### 6.1 Walk-Forward RL Validation — Pipeline A (1h, BTCUSDT, 10 Folds)

Period: 2023-06-01 – 2025-12-31 | Metric: Expected Value per trade

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

Folds 7–8 validate on 2025 data never seen during optimization.

### 6.2 True OOS — Q1 2026 (Pipeline A — 13-dim QLSTM)

> **Honest status update** as of 2026-03-23

| Metric | Value | Gate |
|---|---|---|
| Win Rate | 22.29% (35/157 trades) | ❌ Gate 1 FAIL (BEP = 25.4%) |
| Mean R-multiple | −0.2003 | ❌ Gate 2 FAIL (p = 1.0) |
| ROI | −51.6% | — |
| Sharpe | −2.517 | — |
| Max Drawdown | 58.22% | — |
| Long bias | **100% LONG** (0 short trades) | 🔴 Critical bug |
| Kelly fraction | −6.90% | 🔴 No allocation |

**Root cause confirmed** (see [`scripts/debug_long_bias.py`](scripts/debug_long_bias.py)):
- Primary: `_koop_vecs = [[0,0,0,0,0]]` in EDMD → all VQC angles fixed → input-invariant model
- Secondary: `logit_proj` bias +0.352 LONG-SHORT gap (accumulated during training)
- Tertiary: `F.relu(advantages)` in loss → only reinforces positive-advantage actions → self-reinforcing loop

Full OOS report: [`reports/qlstm13_oos_report_q1_2026.json`](reports/qlstm13_oos_report_q1_2026.json)
R-multiple distribution: [`reports/qlstm13_rmultiples_q1_2026.csv`](reports/qlstm13_rmultiples_q1_2026.csv)

### 6.3 Behavioral Alpha — Pipeline B Baseline

Strategy: Funding-rate squeeze + EMA-200 long-only | 5× effective leverage

| Period | ROI | Win Rate | Trades | Max Drawdown |
|---|---|---|---|---|
| **2023 (true OOS — never optimized)** | **+23.63%** | 34.5% | 29 | — |
| 2024 | +87.44% | 38.1% | 41 | — |
| 2025 | +12.05% | 36.2% | 21 | — |
| 2026 Q1 | +3.18% | — | 4 | — |
| **Cumulative (3.25 years)** | **+126.30%** | **36.8%** | **95** | **16.27%** |

2023 data was entirely withheld from parameter selection — constitutes a clean OOS test of structural alpha.

### 6.4 Physics & Mathematics Quality Score

Independent evaluation (self-assessed against literature benchmarks):

| Dimension | Baseline | Current | Epistemological Ceiling |
|---|---|---|---|
| **Physics** | 3.5 / 10 | **7.0 / 10** | ~9.8 (Cramér–Rao bound) |
| **Mathematics** | 4.5 / 10 | **7.5 / 10** | ~9.9 (Lyapunov convergence) |

---

## 7. Honest Limitations & Active Bugs

### 7.1 Active Bug List (as of 2026-03-24)

| # | Bug | File | Status |
|---|---|---|---|
| 1 | `_koop_vecs = zeros` → input-invariant model | `src/data/spectral_decomposer.py` | 🔴 Fix pending |
| 2 | `logit_proj` bias +0.352 LONG-SHORT | `src/models/integrated_agent.py` | 🔴 Fix pending |
| 3 | `F.relu(advantages)` → self-reinforcing loop | `src/models/loss.py` | 🔴 Fix pending |

Regression test: [`tests/test_agent_short_signal.py`](tests/test_agent_short_signal.py)

### 7.2 Epistemological Bound on Market Prediction

Perfect prediction is theoretically bounded. The **Cramér–Rao inequality**:

```
Var(μ̂) ≥ σ² / T
```

Minimum estimation error scales inversely with sample size — a hard statistical limit independent of model complexity. The **Jarzynski bound** further constrains maximum extractable alpha:

```
⟨exp(−βW)⟩ = exp(−β·ΔF)
```

The realistic goal is a consistent positive KL divergence from random baseline: `E[log(P_model / P_random)] > 0`.

### 7.3 Identified Mathematical Gaps (Roadmap)

| Gap | Current State | Target Implementation |
|---|---|---|
| Markov assumption in GAE | δ_t = r_t + γV(s_{t+1}) − V(s_t) ignores long-range dependence | GLE memory kernel γ(t) = Σ a_k exp(−b_k t) |
| Path-integral loss approximation | J(τ) = Σ γ^t ΔV_t (no saddle-point structure) | Onsager–Machlup S[x] = ∫(ẋ−μ)²/(2σ²) dt |
| Scalar critic discards tail risk | CriticHead → E[Z(s,a)] only | IQN distributional RL → CVaR_α[Z] |
| PCA instead of Koopman dynamics | Linear variance decomposition | EDMD K = G⁻¹A (after fixing zero-init bug) |
| Barren plateaus in VQC | Random parameter init | Quantum Natural Gradient (QFI metric) |

---

## 8. Quick Start

### Setup

* All of the dependecies can be struggling to set up, I used al most 6 years-old Laptop for computing environment, so requirement.txt's contents can crash to the other upper version of CUDA services, Torch, and other libraries (I used CUDA 12.6, GeForce MX-450 Laptop GPU for the training) *

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # fill in BYBIT_API_KEY, BYBIT_API_SECRET
```

### Behavioral Alpha Backtest (Validated)

```bash
python scripts/backtest_behavioral.py \
    --signals fr --long-only --trend-ema 200 --fr-z-thr 2.5
```

### QLSTM Pipeline (After Bug Fix)

```bash
# 1. Delete old checkpoint
rm data/quantum_checkpoint.npz

# 2. Retrain (1h, 365 days)
python scripts/train_quantum_v2.py \
    --symbol BTCUSDT --timeframe 1h --days 365 --epochs 30 --device cuda

# 3. Validate (Gate 1/2)
python scripts/backtest_structural.py --days 90

# 4. TUI (paper mode only — default)
python scripts/run_quantum_tui.py --mode paper
```

### Structural Backtest

```bash
python scripts/backtest_structural.py --days 90 --symbol BTCUSDT
```

### Run Tests

```bash
pytest -q  # includes test_agent_short_signal.py (input-invariance regression)
```

---

## Project Structure

```
.
├── src/
│   ├── data/           # Bybit client, EDMD spectral decomposer
│   ├── models/         # QLSTM, VQC, advanced physics, features V4/V5
│   ├── features/       # Structural 13-dim feature builder
│   ├── strategies/     # Regime gate, HMM regime
│   ├── risk/           # Kill switch, circuit breaker, EV tracker
│   ├── storage/        # SQLite audit trail
│   └── app/            # Textual TUI
├── scripts/
│   ├── backtest_behavioral.py   # FR_LONG + EMA200 backtest
│   ├── backtest_structural.py   # Structural 13-dim backtest
│   ├── train_quantum_v2.py      # QLSTM training
│   ├── debug_long_bias.py       # Root-cause analysis tool
│   └── train_synthetic_bias_test.py  # Input-invariance test
├── backtesting/
│   ├── validation.py   # Gate 1 (WR > BEP) + Gate 2 (bootstrap p-value)
│   └── runner.py
├── reports/
│   ├── qlstm13_oos_report_q1_2026.json   # Q1 2026 OOS results
│   ├── qlstm13_rmultiples_q1_2026.csv    # R-multiple distribution
│   ├── liq_cascade_dynamics.png          # Liquidation cascade analysis
│   ├── liq_alpha_halflife.png            # Alpha decay analysis
│   └── structural_trades_*.csv           # Structural signal trade logs
├── docs/
│   ├── onchain_feature_design.md         # Next-phase on-chain feature spec
│   ├── postmortem_28dim_model.md         # 28-dim model failure analysis
│   ├── postmortems/QLSTM_v3_failure.md  # V3 model postmortem
│   └── validation_framework_spec.md     # Gate 1/2 specification
├── tests/
│   └── test_agent_short_signal.py       # CI regression: signal bias
└── configs/
    └── default.yaml
```

---

[한국어 README](README_KR.md)
