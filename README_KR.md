# QLSTM-Quant-Trading-Bot

> [English README](README.md)

# 암호화폐 자동매매를 위한 Quantum-Classical Hybrid Reinforcement Learning

Bybit Perpetual Futures에서 VQC(Variational Quantum Circuit), 확률론적 물리학, 구조적 시장 메커니즘을 활용한 자율 트레이딩 시스템입니다.

---

## Abstract

이 프로젝트의 핵심 의사결정 엔진은 **Quantum-Classical Hybrid Actor-Critic agent** (QLSTM + QA3C)입니다. Classical backbone이 시장 미시구조 feature를 추출하고, PennyLane 기반의 **VQC(Variational Quantum Circuit)**를 통과시킵니다. Generalized Advantage Estimation(GAE), Lindblad quantum master equation 기반 regime 감지, Fokker-Planck SDE regularizer, Wasserstein DRO를 통합한 **Advanced Path-Integral Loss**로 end-to-end 학습합니다.

**현재 상태 (2026-03-24):** Walk-forward cross-validation(2023~2025) 전체 fold에서 positive EV 확인. 그러나 Q1 2026 진짜 OOS 평가에서 critical한 directional bias(100% LONG)가 발견됐고, EDMD spectral decomposer의 zero-initialization bug가 원인으로 확인됐습니다. 한편 독립적으로 운용한 behavioral alpha 전략(funding-rate squeeze + EMA-200)은 **+126% ROI(3.25년 OOS)**를 기록했습니다. 현재는 QLSTM bug 수정과 병행해, structural 13-dim mechanism-based feature 기반 pipeline으로 pivot 중입니다.

---

## Table of Contents

1. [왜 Quantum Circuit인가](#1-왜-quantum-circuit인가)
2. [Tech Stack](#2-tech-stack)
3. [Architecture 변천사](#3-architecture-변천사)
4. [Pipeline A — Quantum QLSTM (V4/V5)](#4-pipeline-a--quantum-qlstm-v4v5)
5. [Pipeline B — Structural Mechanism Features (현재)](#5-pipeline-b--structural-mechanism-features-현재)
6. [Results & Validation](#6-results--validation)
7. [한계 및 Active Bugs](#7-한계-및-active-bugs)
8. [Quick Start](#8-quick-start)

---

## 1. 왜 Quantum Circuit인가

### 1.1 Classical Deep Learning의 한계

대부분의 금융 딥러닝 모델은 수익률 시계열을 Markovian stochastic process로 취급합니다. 하지만 실증 데이터를 보면 암호화폐 수익률은 **long-range dependence** (Hurst exponent H > 0.5)를 가지고, 꼬리가 Gaussian Brownian Motion과 전혀 맞지 않습니다. 고차원 parameter manifold에 대한 gradient-based optimizer는 **barren plateau** 문제에 빠지고, scalar value function은 tail-risk-aware 트레이딩에 필요한 return distribution 전체를 버립니다.

VQC는 세 가지 보완적인 장점이 있습니다:

- **Entanglement 기반 feature interaction**: IsingZZ coupling layer가 명시적 feature engineering 없이 비선형 상관관계를 포착합니다.
- **Unitary manifold에서의 자연스러운 expressivity**: Quantum gate는 SU(2^n) 위에 있고, 이 Riemannian geometry는 Euclidean space보다 확률분포 최적화에 적합합니다.
- **Quantum coherence를 regime signal로 활용**: Lindblad master equation으로 진화시킨 circuit의 density matrix purity Tr(ρ²)를 실시간 시장 regime indicator로 사용합니다.

### 1.2 구현한 핵심 개념

| 개념 | 수학적 기반 | 참고 |
|---|---|---|
| Variational Quantum Circuit (VQC) | U(θ) = Π_k exp(-iθ_k H_k) | Cerezo et al. (2021), *Nature Reviews Physics* |
| Lindblad Master Equation | dρ/dt = -i[H,ρ] + Σ_k γ_k(L_k ρ L_k† - ½{L_k†L_k, ρ}) | Breuer & Petruccione (2002) |
| Fokker-Planck / Langevin SDE | dx = μ(x)dt + σ(x)dW; L_FP = -log N(x'; x+μdt, σ²dt) | Risken (1989) |
| GAE (Generalized Advantage Estimation) | Â_t = Σ_l (γλ)^l [r_{t+l} + γV(s_{t+l+1}) - V(s_{t+l})] | Schulman et al. (2016), *ICLR* |
| Wasserstein DRO | min_θ max_{P: W₁(P,Q)≤ε} E_P[ℓ] | Esfahani & Kuhn (2018) |
| Hurst Exponent (R/S Analysis) | E[R(n)/S(n)] ≈ c · n^H | Mandelbrot & Wallis (1969) |
| Matrix Product States (MPS) | \|ψ⟩ = Σ A¹[i₁]A²[i₂]...Aⁿ[iₙ]\|i₁...iₙ⟩ | Vidal (2003), *PRL* |
| MINE (Mutual Information Neural Estimation) | I(X;Y) ≥ E_joint[T] - log E_marginal[e^T] | Belghazi et al. (2018), *ICML* |
| Optimal Stopping (Snell Envelope) | J(x,t) = sup_{τ≥t} E[g(X_τ)\|X_t=x] | Peskir & Shiryaev (2006) |
| Frenet-Serret Geometry | Γ(t+Δ) ≈ Γ(t) + Δ\|Γ'\|T + (Δ\|Γ'\|)²/2 · κN | do Carmo (1976) |
| Platt Calibration | P_calibrated = σ(T · logit + b) | Platt (1999) |
| RMT Marchenko-Pastur Denoising | λ± = σ²(1 ± √(D/T))² | Marchenko & Pastur (1967) |

---

## 2. Tech Stack

| 분류 | 도구 |
|---|---|
| **Language** | Python 3.11 |
| **Deep Learning** | PyTorch 2.x (CUDA), custom autograd |
| **Quantum Computing** | PennyLane 0.38 (`default.qubit`, IsingZZ ansatz) |
| **Reinforcement Learning** | Custom Actor-Critic (QA3C), GAE + entropy bonus |
| **TUI / Dashboard** | Textual 0.59 (Bloomberg-style terminal UI) |
| **Exchange API** | Bybit V5 REST API (`pybit`) — BTCUSDT Perpetual Futures |
| **Data** | OHLCV + Funding Rate + Open Interest via Bybit REST |
| **Storage** | SQLite (full audit trail), NumPy `.npz` (checkpoint) |
| **Hyperparameter Tuning** | Optuna |
| **Validation** | Walk-forward cross-validation, bootstrap p-value (Gate 1/2) |
| **Numerical** | NumPy, Pandas, SciPy |
| **Visualization** | Matplotlib, Plotly |

---

## 3. Architecture 변천사

실패와 pivot 과정을 그대로 기록했습니다. 과학적 투명성을 위한 의도적인 선택입니다.

```
VERSION HISTORY
═══════════════════════════════════════════════════════════════════
V1  (2023-2025)  │ 15m timeframe │ 28-dim QLSTM + VQC
                 │ In-sample: WR 42-54%, EV +0.06~+0.12
                 │ OOS (2026 Q1): WR 26.4%, ROI -54%  FAIL
                 │ 원인: 15m timeframe - 통계 패턴은
                 │        inference 시점에 자기파괴
                 │
V2  (2026-02)    │ 1h timeframe  │ 28-dim V4 + Frenet V5 + EDMD
                 │ Walk-forward (10 folds, 2023-2025): EV > 0 all
                 │ OOS (2026 Q1): WR 22.29%, ROI -51.6%  FAIL
                 │ 원인: EDMD _koop_vecs=zeros → input-invariant
                 │        model → 100% LONG bias (bug 확인)
                 │
BASELINE         │ FR_LONG + EMA200 behavioral alpha
(독립 운용)      │ OOS (2023-2026): WR 36.8%, ROI +126%  ALPHA
                 │
V3 (현재)        │ 1h timeframe  │ 13-dim structural mechanism
                 │ 통계 feature → structural feature pivot
                 │ QLSTM bug 수정 후 Gate 1/2 validation 예정
═══════════════════════════════════════════════════════════════════
```

### 3.1 왜 Pivot했나?

통계 패턴 feature(RSI, MACD, Bollinger, Hurst)에는 근본적인 인식론적 문제가 있습니다: **통계 패턴이 안정적으로 예측 가능하다면, arbitrageur들이 그것을 없애버립니다.** 패턴이 널리 쓰이는 순간 signal-to-noise ratio가 무너집니다.

Structural mechanism feature(funding rate, open interest, liquidation cascade, CVD)는 **트레이더가 행동할 수밖에 없는 이유**를 인코딩합니다. 그들의 행동 패턴이 아니라요. Funding rate squeeze는 시장 확신과 무관하게 구조적으로 short 포지션을 강제 청산시킵니다. 이건 상관관계가 아니라 인과 메커니즘입니다.

---

## 4. Pipeline A — Quantum QLSTM (V4/V5)

### 4.1 Feature Engineering

```
Raw OHLCV + Funding Rate + Open Interest  [B, seq_len=96, raw_dim]
    │
    ├── Log-returns: r_t = ln(P_t / P_{t-1})      ← stationarity 확보
    ├── Rolling Z-score normalisation (window=60)  ← price invariance
    ├── RMT Marchenko-Pastur denoising             ← noise 제거
    │       λ_noise = σ²(1 ± √(D/T))²
    └── EDMD Koopman eigenvectors (5-dim)          ← 예측 가능한 mode 추출
```

**V4 Feature Index Map (28-dimensional):**

| Index | Feature | 해석 |
|---|---|---|
| 0-4 | `log_return`, `rsi_14`, `macd_val`, `atr_14`, `obi` | Classical technical |
| 5-9 | `vol_ratio`, `ema12_dev`, `mom3`, `mom10`, `mom20` | Momentum 구조 |
| 10-16 | `bb_width`, `roc5`, `hr_sin`, `hr_cos`, `price_zscore`, `hl_ratio`, `tick_imb` | Microstructure |
| 17 | **Hurst H** | R/S multi-scale; H>0.55 → persistent, H<0.45 → mean-reverting |
| 18 | **γ(1)** lag-1 autocorrelation | γ>0: persistent; γ<0: mean-reverting |
| 19 | **purity_proxy** | 1-2H(return histogram); Lindblad density proxy |
| 20 | `funding_rate_zscore` | Short side 과밀도 감지 |
| 21 | `candle_body_ratio` | Bull/bear bar 강도 ∈ [-1,1] |
| 22 | `volume_zscore` | 비정상 거래량 |
| 23 | `oi_change_pct` | OI momentum (trend conviction) |
| 24 | `funding_velocity` | Funding crowding 변화율 |
| 25 | `cvd_delta_zscore` | 봉당 buyer/seller aggression |
| 26 | `cvd_trend_zscore` | 5시간 누적 매수/매도 우위 |
| 27 | `cvd_price_divergence` | 스마트머니 accumulation fingerprint |

**V5 Extension — Frenet-Serret Geometric Features (54-dim = V4 28 + Frenet 20):**

시장 변수 4개를 phase space상의 parametric curve로 해석합니다. Curvature κ와 torsion τ가 trend reversal의 인과적 예측 신호로 작동합니다:

```
Second-order Taylor approximation:
    Γ(t+Δ) ≈ Γ(t)  +  Δ|Γ'|·T  +  (Δ|Γ'|)²/2 · κ·N

Reversal signal:
    κ 극대값  →  trend reversal 임박
    τ ≈ 0    →  regime이 2D로 축소 (high-confidence signal)
```

| Phase-Space Curve | Geometric Feature |
|---|---|
| MACD phase portrait `(ema12_dev, macd_val)` | T_x, T_y, κ, turn_sign (4D) |
| CVD phase portrait `(cvd_delta_z, cvd_trend_z)` | T_x, T_y, κ, turn_sign (4D) |
| Momentum term structure `(mom3, mom10, mom20)` | T, κ, τ, N (8D) |
| Price-volume path `(log_return, vol_ratio)` | T_x, T_y, κ, turn_sign (4D) |

### 4.2 End-to-End Inference Pipeline

```
Features [B, 96, 28]
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  SPECTRAL DECOMPOSER  (src/data/spectral_decomposer.py)  │
│  EDMD Koopman eigenvectors → 5개 slow mode               │
│  Input: [B, 96, 28]   Output: [B, 96, 5]                │
│  BUG: _koop_vecs=zeros (수정 중)                         │
└──────────────────────┬───────────────────────────────────┘
                       │  context vector c_kt  [B, T, 5]
                       ▼
┌──────────────────────────────────────────────────────────┐
│  TEMPORAL CONTEXT ENCODER  (Transformer)                 │
│  Multi-head self-attention, seq_len = 96 bars            │
│  (96 × 1h = 4일치 시장 history)                          │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  QUANTUM MARKET ENCODER  (src/models/quantum_layers.py)  │
│  proj_to_qubits: Linear(5 → 3, bias=False)               │
│  IsingZZ VQC (N_QUBITS=3) → <σ^z> expectation value    │
│  h_field (RMT noise floor) broadcast                     │
│  logit_proj: Linear(3, 3) → logits [B, 3]                │
│  Action: HOLD (0) / LONG (+1) / SHORT (-1)               │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  INFERENCE & RISK GATES                                  │
│  AdaptiveTemperatureScaler: T = T_base × (1 + β·σ_ATR)  │
│  PlattCalibrator (T_platt, b): logit → calibrated prob   │
│  LindbladDecoherence: purity < 0.3 → HOLD 강제           │
│  Kill Switch + Circuit Breaker (MDD 8% 자동 정지)        │
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
    L_critic = MSE( V(s_t), returns_t )                (value function)
    L_FP     = −log N(x'; x + μΔt, σ²Δt)             (Fokker-Planck SDE)
    H(π)     = −Σ_a π(a) log π(a)                     (entropy bonus)

    Â_t = Σ_l (γλ)^l δ_{t+l},   δ_t = r_t + γV(s_{t+1}) − V(s_t)

Physical entry barrier:
    J(τ) = Σ_t γ^t ΔV_t  −  η · I_{open}  +  R_terminal(S_T)
    η = 0.001 × leverage  (Bybit round-trip fee: Maker 0.02% / Taker 0.055%)
```

### 4.4 Physics Layer (src/models/advanced_physics.py)

| 컴포넌트 | 수학적 내용 |
|---|---|
| `HurstEstimator` | Multi-scale R/S → H; H>0.55 persistent, H<0.45 mean-reverting |
| `LindbladDecoherence` | dρ/dt = -i[H,ρ] + Σγ_k(L_kρL_k† - ½{L_k†L_k,ρ}); purity = Tr(ρ²) |
| `FokkerPlanckRegularizer` | L_FP = -log N(x'; x+μdt, σ²dt) |
| `WassersteinDROLoss` | W₁-ball adversarial robustness; Kantorovich dual gradient penalty |
| `MPSLayer` | χ-bond MPS tensor chain; area-law entanglement |
| `OptimalStoppingBoundary` | Snell envelope B_t = max(g(x_t), E[B_{t+1}]); learned SL/TP |
| `PlattCalibrator` | σ(T·logit + b); ECE tracker; VQC → calibrated win-probability |
| `MINEEstimator` | I(X;Y) ≥ E_joint[T] − log E_marginal[e^T]; Donsker-Varadhan |

---

## 5. Pipeline B — Structural Mechanism Features (현재)

### 5.1 설계 철학

> 통계 패턴은 시장이 *무엇을* 하는지 묘사합니다. 구조적 메커니즘은 참여자들이 *왜* 행동할 수밖에 없는지를 설명합니다.

Feature 선택 기준은 통계적 상관관계가 아니라 **인과적 메커니즘**입니다:

| 그룹 | Feature (13-dim) | 메커니즘 |
|---|---|---|
| **FR** | `fr_z`, `fr_trend` | Funding rate가 perpetual 메커니즘을 통해 포지션 강제 청산 |
| **OI** | `oi_change_z`, `oi_price_div` | OI 누적/이탈 → 쏠린 포지션 감지 |
| **Liq** | `liq_long_z`, `liq_short_z` | 강제청산 압력 추정 (wick × volume proxy) |
| **CVD** | `cvd_trend_z`, `cvd_price_div`, `taker_ratio_z` | 실제 order flow aggression (taker 매수/매도 불균형) |
| **Regime** | `ema200_dev`, `ema200_slope`, `vol_regime`, `vol_change` | Trend 구조 + volatility regime |

**사용 금지 feature:** RSI, MACD, Bollinger Bands, Hurst, Frenet-Serret, PCA/LDA 결과값 — 전부 arbitrageur가 제거 가능한 통계적 상관관계.

### 5.2 Behavioral Alpha Engine (독립 Baseline)

Perpetual futures funding 메커니즘을 이용하는 structural signal:

```
Signal: funding_rate_zscore < -2.5σ  (crowd가 heavy short)
        AND price > EMA-200           (macro uptrend 확인)
→ Forced short squeeze → LONG 진입만

메커니즘: perpetual contract는 short이 과밀도일 때
          shorts → longs로 funding payment 강제 → 개별 trader의 확신과
          무관하게 구조적 buying pressure 발생
```

```bash
python scripts/backtest_behavioral.py \
    --signals fr --long-only --trend-ema 200 --fr-z-thr 2.5
```

### 5.3 Pipeline Architecture (현재)

```
Raw OHLCV + Funding Rate + Open Interest  (Bybit REST, 1h candle)
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│  STRUCTURAL FEATURE BUILDER  (src/models/features_structural.py) │
│  13-dim mechanism-based feature (통계 패턴 없음)          │
│  FR / OI / Liquidation / CVD / Regime                    │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────┐
│  RULE-BASED SIGNAL LAYER                                  │
│  Baseline: FR_LONG + EMA200 filter                       │
│  Next: structural feature → RL agent (개발 중)           │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────┐
│  GATE 1/2 VALIDATION  (backtesting/validation.py)        │
│  Gate 1: WR > 25.4% (기하평균 BEP, 5× eff leverage)     │
│  Gate 2: bootstrap p < 0.05 on r-multiple distribution  │
│  두 Gate 모두 통과 전까지 실자금 배분 없음               │
└───────────────────────────────────────────────────────────┘
```

---

## 6. Results & Validation

### 6.1 Walk-Forward RL Validation — Pipeline A (1h, BTCUSDT, 10 Fold)

Period: 2023-06-01 ~ 2025-12-31 | Metric: Expected Value per trade

| Fold | Validation Period | EV (per trade) | Avg. Max Confidence |
|---|---|---|---|
| 1 | 2023-09 ~ 2023-11 | +0.2441 | 0.585 |
| 2 | 2023-11 ~ 2024-02 | +0.2106 | 0.746 |
| 3 | 2024-02 ~ 2024-05 | +0.3219 | 0.777 |
| 4 | 2024-05 ~ 2024-08 | +0.2093 | 0.807 |
| 5 | 2024-08 ~ 2024-10 | +0.2013 | 0.745 |
| 6 | 2024-10 ~ 2025-02 | +0.2672 | 0.751 |
| **7** | **2025-02 ~ 2025-04** | **+0.2929** | **0.830** ★ 2025 OOS |
| **8** | **2025-04 ~ 2025-07** | **+0.2192** | **0.856** ★ 2025 OOS |

Fold 7-8은 optimization 과정에서 전혀 보지 않은 2025 데이터로 검증.

### 6.2 True OOS — Q1 2026 (Pipeline A — 13-dim QLSTM)

> **솔직한 현황** (2026-03-23 기준)

| Metric | 값 | Gate |
|---|---|---|
| Win Rate | 22.29% (35/157 trades) | Gate 1 FAIL (BEP = 25.4%) |
| Mean R-multiple | -0.2003 | Gate 2 FAIL (p = 1.0) |
| ROI | -51.6% | — |
| Sharpe | -2.517 | — |
| Max Drawdown | 58.22% | — |
| Long bias | **100% LONG** (SHORT 0건) | Critical bug |
| Kelly fraction | -6.90% | 자금 배분 불가 |

**Root cause 확인** ([`scripts/debug_long_bias.py`](scripts/debug_long_bias.py) 참고):
- Primary: `_koop_vecs = [[0,0,0,0,0]]` → VQC angle이 모두 고정 → input-invariant model
- Secondary: `logit_proj` bias +0.352 LONG-SHORT gap (training 중 축적)
- Tertiary: loss의 `F.relu(advantages)` → positive-advantage action만 강화 → self-reinforcing loop

Full OOS report: [`reports/qlstm13_oos_report_q1_2026.json`](reports/qlstm13_oos_report_q1_2026.json)
R-multiple distribution: [`reports/qlstm13_rmultiples_q1_2026.csv`](reports/qlstm13_rmultiples_q1_2026.csv)

### 6.3 Behavioral Alpha — Pipeline B Baseline

Strategy: Funding-rate squeeze + EMA-200 long-only | 5× effective leverage

| Period | ROI | Win Rate | Trades | Max Drawdown |
|---|---|---|---|---|
| **2023 (true OOS — parameter selection 전혀 없음)** | **+23.63%** | 34.5% | 29 | — |
| 2024 | +87.44% | 38.1% | 41 | — |
| 2025 | +12.05% | 36.2% | 21 | — |
| 2026 Q1 | +3.18% | — | 4 | — |
| **누적 (3.25년)** | **+126.30%** | **36.8%** | **95** | **16.27%** |

2023 데이터는 parameter 선택 과정에서 완전히 격리됐습니다. structural alpha의 깨끗한 OOS 검증입니다.

### 6.4 Physics & Mathematics Quality Score

문헌 기준 자체 평가:

| 차원 | Baseline | 현재 | 인식론적 상한 |
|---|---|---|---|
| **Physics** | 3.5 / 10 | **7.0 / 10** | ~9.8 (Cramér-Rao bound) |
| **Mathematics** | 4.5 / 10 | **7.5 / 10** | ~9.9 (Lyapunov convergence) |

---

## 7. 한계 및 Active Bugs

### 7.1 Active Bug List (2026-03-24 기준)

| # | Bug | 파일 | 상태 |
|---|---|---|---|
| 1 | `_koop_vecs = zeros` → input-invariant model | `src/data/spectral_decomposer.py` | 수정 대기 중 |
| 2 | `logit_proj` bias +0.352 LONG-SHORT | `src/models/integrated_agent.py` | 수정 대기 중 |
| 3 | `F.relu(advantages)` → self-reinforcing loop | `src/models/loss.py` | 수정 대기 중 |

Regression test: [`tests/test_agent_short_signal.py`](tests/test_agent_short_signal.py)

### 7.2 시장 예측의 인식론적 한계

완벽한 예측은 이론적으로 불가능합니다. **Cramér-Rao inequality:**

```
Var(μ̂) ≥ σ² / T
```

최소 추정 오차는 샘플 크기에 반비례합니다 — 모델 복잡도와 무관한 통계적 하한선입니다. **Jarzynski bound**는 추출 가능한 최대 alpha에 추가 제약을 부과합니다:

```
<exp(-βW)> = exp(-β·ΔF)
```

현실적인 목표는 random baseline 대비 일관된 양의 KL divergence입니다: `E[log(P_model / P_random)] > 0`.

### 7.3 식별된 수학적 Gap (Roadmap)

| Gap | 현재 상태 | 목표 구현 |
|---|---|---|
| Markov assumption in GAE | δ_t가 long-range dependence 무시 | GLE memory kernel γ(t) = Σ a_k exp(-b_k t) |
| Path-integral loss 근사 | J(τ) = Σ γ^t ΔV_t (saddle-point 구조 없음) | Onsager-Machlup S[x] = ∫(ẋ-μ)²/(2σ²) dt |
| Scalar critic이 tail risk 버림 | CriticHead → E[Z(s,a)] only | IQN distributional RL → CVaR_α[Z] |
| Koopman dynamics 대신 PCA | Linear variance decomposition | EDMD K = G⁻¹A (zero-init bug 수정 후) |
| VQC barren plateau | Random parameter init | Quantum Natural Gradient (QFI metric) |

---

## 8. Quick Start

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # BYBIT_API_KEY, BYBIT_API_SECRET 입력
```

### Behavioral Alpha Backtest (검증 완료)

```bash
python scripts/backtest_behavioral.py \
    --signals fr --long-only --trend-ema 200 --fr-z-thr 2.5
```

### QLSTM Pipeline (Bug Fix 후)

```bash
# 1. 기존 checkpoint 삭제
rm data/quantum_checkpoint.npz

# 2. Retrain (1h, 365일)
python scripts/train_quantum_v2.py \
    --symbol BTCUSDT --timeframe 1h --days 365 --epochs 30 --device cuda

# 3. Validate (Gate 1/2)
python scripts/backtest_structural.py --days 90

# 4. TUI (paper mode 기본값)
python scripts/run_quantum_tui.py --mode paper
```

### Structural Backtest

```bash
python scripts/backtest_structural.py --days 90 --symbol BTCUSDT
```

### 테스트 실행

```bash
pytest -q  # test_agent_short_signal.py 포함 (input-invariance regression)
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
│   ├── backtest_behavioral.py        # FR_LONG + EMA200 backtest
│   ├── backtest_structural.py        # Structural 13-dim backtest
│   ├── train_quantum_v2.py           # QLSTM training
│   ├── debug_long_bias.py            # Root-cause analysis tool
│   └── train_synthetic_bias_test.py  # Input-invariance test
├── backtesting/
│   ├── validation.py   # Gate 1 (WR > BEP) + Gate 2 (bootstrap p-value)
│   └── runner.py
├── reports/
│   ├── qlstm13_oos_report_q1_2026.json   # Q1 2026 OOS 결과
│   ├── qlstm13_rmultiples_q1_2026.csv    # R-multiple distribution
│   ├── liq_cascade_dynamics.png          # Liquidation cascade 분석
│   ├── liq_alpha_halflife.png            # Alpha decay 분석
│   └── structural_trades_*.csv           # Structural signal 거래 기록
├── docs/
│   ├── onchain_feature_design.md         # Next-phase on-chain feature spec
│   ├── postmortem_28dim_model.md         # 28-dim 모델 실패 분석
│   ├── postmortems/QLSTM_v3_failure.md  # V3 모델 postmortem
│   └── validation_framework_spec.md     # Gate 1/2 specification
├── tests/
│   └── test_agent_short_signal.py       # CI regression: signal bias
└── configs/
    └── default.yaml
```
