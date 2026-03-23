# QLSTM-Quant-Trading-Bot

# 암호화폐 알고리즘 트레이딩을 위한 양자-고전 하이브리드 강화학습 시스템

> Bybit 무기한 선물 자율 트레이딩에 양자 변분 회로, 고급 확률 물리학, 구조적 시장 미시구조 메커니즘을 적용한 포트폴리오 프로젝트.

---

## 초록

이 프로젝트는 **양자-고전 하이브리드 액터-크리틱 에이전트** (QLSTM + QA3C)를 핵심 의사결정 엔진으로 하는 풀스택 자동화 트레이딩 시스템입니다. 고전 백본이 시장 미시구조 피처를 인코딩하고, PennyLane 기반의 **변분 양자 회로(VQC)**에 입력합니다. 에이전트는 정책 경사 최적화(일반화 이점 추정, GAE), Lindblad 양자 마스터 방정식 레짐 감지, Fokker–Planck SDE 정규화기, Wasserstein 분포적 강건 최적화를 결합한 **고급 경로 적분 손실**로 엔드-투-엔드 학습됩니다.

**연구 현황 (2026-03-24):** 워크-포워드 교차 검증(2023–2025)에서 전 폴드 양의 기대값을 확인했습니다. 그러나 2026년 Q1 진정 OOS 평가에서 EDMD 스펙트럼 분해기의 영벡터 초기화 버그로 인한 치명적 방향 편향(100% LONG)이 발견되었습니다. 독립적인 행동 알파 전략(펀딩레이트 스퀴즈 + EMA-200)은 **3.25년 OOS 기간 동안 +126% ROI**를 달성했습니다. QLSTM 버그 수정과 병행하여 구조적 13차원 메커니즘 기반 피처로의 피벗을 진행 중입니다.

---

## 목차

1. [동기 및 참고문헌](#1-동기-및-참고문헌)
2. [기술 스택](#2-기술-스택)
3. [아키텍처 진화](#3-아키텍처-진화)
4. [파이프라인 A — 양자 QLSTM (V4/V5)](#4-파이프라인-a--양자-qlstm-v4v5)
5. [파이프라인 B — 구조적 메커니즘 피처 (현재)](#5-파이프라인-b--구조적-메커니즘-피처-현재)
6. [결과 및 검증](#6-결과-및-검증)
7. [한계 및 활성 버그](#7-한계-및-활성-버그)
8. [빠른 시작](#8-빠른-시작)

---

## 1. 동기 및 참고문헌

### 1.1 왜 양자 회로인가?

고전 딥러닝 모델은 금융 시계열을 기억이 없는(마르코프) 확률 과정으로 취급합니다. 그러나 암호화폐 수익률은 **장기 의존성** (허스트 지수 H > 0.5)과 가우시안 브라운 운동과 일치하지 않는 두꺼운 꼬리를 보입니다. 고차원 파라미터 매니폴드에서 표준 경사 기반 최적화는 **불모 고원(barren plateau)** 문제를 겪으며, 스칼라 가치 함수는 꼬리 위험 인식 트레이딩에 필요한 전체 수익률 분포를 버립니다.

변분 양자 회로(VQC)는 세 가지 상호 보완적 장점을 제공합니다:

- **얽힘 기반 피처 상호작용**: IsingZZ 결합 레이어가 명시적 피처 엔지니어링 없이 시장 피처 간 비선형 상관관계를 포착합니다.
- **유니터리 매니폴드에서의 자연스러운 표현력**: 양자 게이트는 SU(2^n) 위에 존재하며, 이는 유클리드 파라미터 공간보다 확률 분포 최적화에 더 적합한 리만 기하를 갖습니다.
- **레짐 신호로서의 양자 결맞음**: Lindblad 마스터 방정식으로 진화하는 회로 밀도 행렬의 순도 Tr(ρ²)가 실시간 시장 레짐 지표로 작동합니다.

### 1.2 구현된 핵심 개념

| 개념 | 수학적 기반 | 참고문헌 |
|---|---|---|
| 변분 양자 회로 (VQC) | 파라미터화된 유니터리 U(θ) = Π_k exp(-iθ_k H_k) | Cerezo et al. (2021), *Nature Reviews Physics* |
| Lindblad 마스터 방정식 | dρ/dt = -i[H,ρ] + Σ_k γ_k(L_k ρ L_k† - ½{L_k†L_k, ρ}) | Breuer & Petruccione (2002) |
| Fokker–Planck / Langevin SDE | dx = μ(x)dt + σ(x)dW; L_FP = -log N(x'; x+μdt, σ²dt) | Risken (1989) |
| 일반화 이점 추정 (GAE) | Â_t = Σ_l (γλ)^l [r_{t+l} + γV(s_{t+l+1}) - V(s_{t+l})] | Schulman et al. (2016), *ICLR* |
| Wasserstein DRO | min_θ max_{P: W₁(P,Q)≤ε} E_P[ℓ] | Esfahani & Kuhn (2018) |
| 허스트 지수 (R/S 분석) | E[R(n)/S(n)] ≈ c · n^H | Mandelbrot & Wallis (1969) |
| 행렬 곱 상태 (MPS) | \|ψ⟩ = Σ A¹[i₁]A²[i₂]…Aⁿ[iₙ]\|i₁…iₙ⟩ | Vidal (2003), *PRL* |
| 상호 정보 신경 추정 (MINE) | I(X;Y) ≥ E_joint[T] - log E_marginal[e^T] | Belghazi et al. (2018), *ICML* |
| 최적 정지 (Snell 포락선) | J(x,t) = sup_{τ≥t} E[g(X_τ)\|X_t=x] | Peskir & Shiryaev (2006) |
| Frenet–Serret 기하학 | Γ(t+Δ) ≈ Γ(t) + Δ\|Γ'\|T + (Δ\|Γ'\|)²/2 · κN | do Carmo (1976) |
| Platt 보정 | P_calibrated = σ(T · logit + b) | Platt (1999) |
| RMT Marchenko–Pastur 잡음 제거 | λ± = σ²(1 ± √(D/T))² | Marchenko & Pastur (1967) |

---

## 2. 기술 스택

| 분류 | 도구 |
|---|---|
| **언어** | Python 3.11 |
| **딥러닝** | PyTorch 2.x (CUDA), 커스텀 autograd |
| **양자 컴퓨팅** | PennyLane 0.38 (`default.qubit`, IsingZZ 안사츠) |
| **강화학습** | 커스텀 액터-크리틱 (QA3C), GAE + 엔트로피 보너스 |
| **TUI / 대시보드** | Textual 0.59 (블룸버그 스타일 터미널 UI) |
| **거래소 API** | Bybit V5 REST API (`pybit`) — BTCUSDT 무기한 선물 |
| **데이터** | Bybit REST를 통한 OHLCV + 펀딩레이트 + 미결제약정 |
| **저장소** | SQLite (전체 감사 추적), NumPy `.npz` (체크포인트) |
| **하이퍼파라미터 최적화** | Optuna |
| **검증** | 워크-포워드 교차 검증, 부트스트랩 p-값 (Gate 1/2) |
| **수치 / 과학** | NumPy, Pandas, SciPy |
| **시각화** | Matplotlib, Plotly |

---

## 3. 아키텍처 진화

이 프로젝트는 실패, 피벗, 근본 원인 분석을 포함한 전체 개발 생애주기를 문서화합니다 — 과학적 투명성을 위한 의도적 선택입니다.

```
버전 이력
═══════════════════════════════════════════════════════════════════
V1  (2023-2025)  │ 15분봉 │ 28-dim QLSTM + VQC
                 │ 인샘플: WR 42–54%, EV +0.06~+0.12
                 │ OOS (2026 Q1): WR 26.4%, ROI −54%  ❌ 실패
                 │ 원인: 15분봉 — 통계 패턴이 추론 시점에 자기파괴
                 │
V2  (2026-02)    │ 1시간봉 │ 28-dim V4 + Frenet V5 + EDMD
                 │ 워크-포워드 (10폴드, 2023-2025): 전 폴드 EV > 0
                 │ OOS (2026 Q1): WR 22.29%, ROI −51.6%  ❌ 실패
                 │ 원인: EDMD _koop_vecs=0 → 입력 불변 모델
                 │        → 100% LONG 편향 (버그 확인됨)
                 │
기준선           │ FR_LONG + EMA200 행동 알파
(병렬 운용)      │ OOS (2023-2026): WR 36.8%, ROI +126%  ✓ 알파 확인
                 │
V3 (현재)        │ 1시간봉 │ 13-dim 구조적 메커니즘 피처
                 │ 통계→구조 피처로 피벗
                 │ QLSTM 버그 수정 후 Gate 1/2 검증 예정
═══════════════════════════════════════════════════════════════════
```

### 3.1 왜 피벗했는가?

통계적 패턴 피처(RSI, MACD, 볼린저, 허스트)는 근본적인 인식론적 문제를 공유합니다: **통계 패턴이 신뢰할 수 있게 예측 가능하다면, 차익 거래자들이 그것을 제거합니다.** 패턴이 광범위하게 사용되는 순간, 신호 대 잡음비가 붕괴됩니다.

구조적 메커니즘 피처(펀딩레이트, 미결제약정, 청산 연쇄, CVD)는 **트레이더들이 행동하도록 강제받는 이유**를 인코딩합니다, 단순히 행동 패턴이 아니라. 펀딩레이트 스퀴즈는 숏 포지션 보유자들을 시장 확신과 무관하게 강제로 청산시킵니다 — 이것은 상관관계가 아닌 인과 메커니즘입니다.

---

## 4. 파이프라인 A — 양자 QLSTM (V4/V5)

### 4.1 피처 엔지니어링

```
원시 OHLCV + 펀딩레이트 + 미결제약정  [B, seq_len=96, raw_dim]
    │
    ├── 로그 수익률: r_t = ln(P_t / P_{t-1})      ← 정상성 보장
    ├── 롤링 Z점수 정규화 (window=60)              ← 가격 수준 불변성
    ├── RMT Marchenko–Pastur 잡음 제거             ← 잡음 고유값 제거
    │       λ_noise = σ²(1 ± √(D/T))²
    └── EDMD 쿠프만 고유벡터 (5차원)               ← 예측 가능한 모드 추출
```

**V4 피처 인덱스 맵 (28차원):**

| 인덱스 | 피처 | 해석 |
|---|---|---|
| 0–4 | `log_return`, `rsi_14`, `macd_val`, `atr_14`, `obi` | 고전 기술 지표 |
| 5–9 | `vol_ratio`, `ema12_dev`, `mom3`, `mom10`, `mom20` | 모멘텀 기간 구조 |
| 10–16 | `bb_width`, `roc5`, `hr_sin`, `hr_cos`, `price_zscore`, `hl_ratio`, `tick_imb` | 미시구조 |
| 17 | **허스트 H** | R/S 다중 스케일; H>0.55 → 지속성, H<0.45 → 평균 회귀 |
| 18 | **γ(1)** 시차-1 자기상관 | γ>0: 지속성; γ<0: 평균 회귀 |
| 19 | **purity_proxy** | 1−2H(수익률 히스토그램); Lindblad 밀도 행렬 프록시 |
| 20 | `funding_rate_zscore` | 숏 포지션 쏠림 감지기 |
| 21 | `candle_body_ratio` | 강세/약세 봉 강도 ∈ [−1,1] |
| 22 | `volume_zscore` | 비정상 거래량 감지 |
| 23 | `oi_change_pct` | OI 모멘텀 (추세 확신 프록시) |
| 24 | `funding_velocity` | 펀딩 쏠림 가속도 |
| 25 | `cvd_delta_zscore` | 봉별 매수/매도 공격성 (주문 흐름 불균형) |
| 26 | `cvd_trend_zscore` | 5시간 누적 매수/매도 지배력 |
| 27 | `cvd_price_divergence` | 스마트머니 축적/분산 지문 |

**V5 확장 — Frenet–Serret 기하 피처 (54차원 = V4 + 20):**

4가지 시장 변수를 위상 공간의 파라메트릭 곡선으로 해석합니다. 곡률 κ와 비틀림 τ가 인과적 추세 전환 예측 변수로 작동합니다:

```
2차 테일러 근사:
    Γ(t+Δ) ≈ Γ(t)  +  Δ|Γ'|·T  +  (Δ|Γ'|)²/2 · κ·N

전환 신호:
    κ 극대값  →  추세 전환 임박 (곡선이 급격히 꺾이는 순간)
    τ ≈ 0     →  레짐이 2D로 붕괴 (고신뢰 신호)
```

| 위상 공간 곡선 | 기하 피처 |
|---|---|
| MACD 위상 초상 `(ema12_dev, macd_val)` | T_x, T_y, κ, turn_sign (4차원) |
| CVD 위상 초상 `(cvd_delta_z, cvd_trend_z)` | T_x, T_y, κ, turn_sign (4차원) |
| 모멘텀 기간 구조 `(mom3, mom10, mom20)` | T, κ, τ, N (8차원) |
| 가격-거래량 경로 `(log_return, vol_ratio)` | T_x, T_y, κ, turn_sign (4차원) |

### 4.2 엔드-투-엔드 추론 파이프라인

```
피처 [B, 96, 28]
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  스펙트럼 분해기  (src/data/spectral_decomposer.py)       │
│  EDMD 쿠프만 고유벡터 → 5개 느린 모드                    │
│  입력: [B, 96, 28]   출력: [B, 96, 5]                    │
│  ⚠ 알려진 버그: _koop_vecs=0 (수정 진행 중)              │
└──────────────────────┬───────────────────────────────────┘
                       │  컨텍스트 벡터 c_kt  [B, T, 5]
                       ▼
┌──────────────────────────────────────────────────────────┐
│  시계열 컨텍스트 인코더  (Transformer)                    │
│  seq_len = 96봉에 걸친 멀티헤드 자기 어텐션              │
│  (96 × 1h = 4일치 시장 이력)                             │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  양자 시장 인코더  (src/models/quantum_layers.py)         │
│  proj_to_qubits: Linear(5 → 3, bias=False)               │
│  IsingZZ VQC (N_QUBITS=3) → ⟨σ^z⟩ 기댓값               │
│  h_field (RMT 잡음 바닥) 브로드캐스트                    │
│  logit_proj: Linear(3, 3) → 로짓 [B, 3]                  │
│  행동: HOLD (0) / LONG (+1) / SHORT (−1)                 │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  추론 및 위험 게이트                                      │
│  적응형 온도 스케일러: T = T_base × (1 + β·σ_ATR)       │
│  Platt 보정기 (T_platt, b): 로짓 → 보정된 확률           │
│  Lindblad 결어긋남: 순도 < 0.3 → HOLD 강제               │
│  킬 스위치 + 회로 차단기 (MDD 8% 자동 정지)             │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
               포지션 사이징
       SL = 1.0×ATR, TP = 3.0×ATR  (R:R = 3:1)
       레버리지 = 10×  |  포지션 비중 = 50%  |  실효 = 5×
```

### 4.3 손실 함수 — 고급 경로 적분 손실

```
L = L_actor  +  c_c · L_critic  +  c_fp · L_FP  −  c_H · H(π)

    L_actor  = −E[ Â_t · log π(a_t | s_t) ]          (GAE 정책 경사)
    L_critic = MSE( V(s_t),  returns_t )               (가치 함수 회귀)
    L_FP     = −log N(x'; x + μΔt, σ²Δt)             (Fokker–Planck SDE)
    H(π)     = −Σ_a π(a) log π(a)                     (엔트로피 보너스)

    Â_t = Σ_l (γλ)^l δ_{t+l},   δ_t = r_t + γV(s_{t+1}) − V(s_t)

물리적 진입 장벽:
    J(τ) = Σ_t γ^t ΔV_t  −  η · I_{open}  +  R_terminal(S_T)
    η = 0.001 × 레버리지  (Bybit 왕복 수수료: Maker 0.02% / Taker 0.055%)
```

### 4.4 물리 레이어 (src/models/advanced_physics.py)

| 컴포넌트 | 수학적 내용 |
|---|---|
| `HurstEstimator` | 다중 스케일 R/S → H; H>0.55 지속성, H<0.45 평균 회귀 |
| `LindbladDecoherence` | dρ/dt = -i[H,ρ] + Σγ_k(L_kρL_k† - ½{L_k†L_k,ρ}); 순도 = Tr(ρ²) |
| `FokkerPlanckRegularizer` | L_FP = -log N(x'; x+μdt, σ²dt) |
| `WassersteinDROLoss` | W₁-공 적대적 강건성; Kantorovich 쌍대 경사 패널티 |
| `MPSLayer` | χ-결합 MPS 텐서 체인; 면적 법칙 얽힘 |
| `OptimalStoppingBoundary` | Snell 포락선 B_t = max(g(x_t), E[B_{t+1}]); 학습된 SL/TP |
| `PlattCalibrator` | σ(T·logit + b); ECE 추적기; VQC → 보정된 승률 확률 |
| `MINEEstimator` | I(X;Y) ≥ E_joint[T] − log E_marginal[e^T]; Donsker–Varadhan |

---

## 5. 파이프라인 B — 구조적 메커니즘 피처 (현재)

### 5.1 설계 철학

> 통계 패턴은 시장이 *무엇을* 하는지 묘사합니다. 구조적 메커니즘은 참여자들이 *왜* 행동하도록 강제받는지 설명합니다.

피처는 통계적 상관관계가 아닌 **인과 메커니즘**으로 선택됩니다:

| 그룹 | 피처 (13차원) | 메커니즘 |
|---|---|---|
| **FR** | `fr_z`, `fr_trend` | 펀딩레이트가 무기한 선물 메커니즘을 통해 포지션 강제 청산 |
| **OI** | `oi_change_z`, `oi_price_div` | OI 누적/괴리 → 쏠린 포지션 감지 |
| **Liq** | `liq_long_z`, `liq_short_z` | 추정 강제 청산 압력 (wick × 거래량 프록시) |
| **CVD** | `cvd_trend_z`, `cvd_price_div`, `taker_ratio_z` | 실제 주문 흐름 공격성 (테이커 매수/매도 불균형) |
| **레짐** | `ema200_dev`, `ema200_slope`, `vol_regime`, `vol_change` | 추세 구조 + 변동성 레짐 |

**명시적 금지 피처:** RSI, MACD, 볼린저 밴드, 허스트, Frenet-Serret, PCA/LDA 출력값 — 모두 차익 거래자가 제거할 수 있는 통계적 상관관계.

### 5.2 행동 알파 엔진 (독립 기준선)

무기한 선물 펀딩 메커니즘을 활용하는 구조적 신호:

```
신호: funding_rate_zscore < −2.5σ  (군중이 강하게 숏 포지션)
      AND price > EMA-200           (거시 상승 추세 확인)
→ 숏 스퀴즈 강제 발생 → LONG 진입만

메커니즘: 숏이 과밀할 때 무기한 계약은 숏에서 롱으로 펀딩 지급 강제
          → 개별 트레이더의 견해와 무관한 구조적 매수 압력 발생
```

```
백테스트: python scripts/backtest_behavioral.py \
               --signals fr --long-only --trend-ema 200 --fr-z-thr 2.5
```

### 5.3 파이프라인 아키텍처 (현재)

```
원시 OHLCV + 펀딩레이트 + 미결제약정  (Bybit REST, 1시간봉)
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  구조적 피처 빌더  (src/models/features_structural.py)      │
│  13차원 메커니즘 기반 피처 (통계 패턴 없음)               │
│  FR / OI / 청산 / CVD / 레짐                              │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│  규칙 기반 신호 레이어                                     │
│  기준선: FR_LONG + EMA200 필터                            │
│  다음: 구조적 피처 → RL 에이전트 (개발 중)               │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│  Gate 1/2 검증  (backtesting/validation.py)               │
│  Gate 1: WR > 25.4% (5× 실효 레버리지 기하 BEP)         │
│  Gate 2: 부트스트랩 p < 0.05 (R-배수 분포 기준)         │
│  두 게이트 모두 통과 전까지 자본 배분 없음               │
└────────────────────────────────────────────────────────────┘
```

---

## 6. 결과 및 검증

### 6.1 워크-포워드 RL 검증 — 파이프라인 A (1시간봉, BTCUSDT, 10폴드)

기간: 2023-06-01 – 2025-12-31 | 지표: 트레이드당 기대값 (EV)

| 폴드 | 검증 기간 | EV (트레이드당) | 평균 최대 신뢰도 |
|---|---|---|---|
| 1 | 2023-09 – 2023-11 | +0.2441 | 0.585 |
| 2 | 2023-11 – 2024-02 | +0.2106 | 0.746 |
| 3 | 2024-02 – 2024-05 | +0.3219 | 0.777 |
| 4 | 2024-05 – 2024-08 | +0.2093 | 0.807 |
| 5 | 2024-08 – 2024-10 | +0.2013 | 0.745 |
| 6 | 2024-10 – 2025-02 | +0.2672 | 0.751 |
| **7** | **2025-02 – 2025-04** | **+0.2929** | **0.830** ★ 2025 OOS |
| **8** | **2025-04 – 2025-07** | **+0.2192** | **0.856** ★ 2025 OOS |

폴드 7–8은 최적화 과정에서 전혀 보지 않은 2025년 데이터로 검증됩니다.

### 6.2 진정 OOS — 2026 Q1 (파이프라인 A — 13차원 QLSTM)

> **정직한 현황 업데이트** (2026-03-23 기준)

| 지표 | 값 | Gate |
|---|---|---|
| 승률 | 22.29% (35/157 트레이드) | ❌ Gate 1 실패 (BEP = 25.4%) |
| 평균 R-배수 | −0.2003 | ❌ Gate 2 실패 (p = 1.0) |
| ROI | −51.6% | — |
| Sharpe | −2.517 | — |
| 최대 낙폭 | 58.22% | — |
| 방향 편향 | **100% LONG** (숏 트레이드 0건) | 🔴 치명적 버그 |
| 켈리 비율 | −6.90% | 🔴 자본 배분 불가 |

**근본 원인 확인** ([`scripts/debug_long_bias.py`](scripts/debug_long_bias.py) 참조):
- 1차: `_koop_vecs = [[0,0,0,0,0]]` → 모든 VQC 각도 고정 → 입력 불변 모델
- 2차: `logit_proj` 편향 +0.352 LONG-SHORT 차이 (학습 중 누적)
- 3차: 손실 함수의 `F.relu(advantages)` → 양의 이점 행동만 강화 → 자기 강화 루프

전체 OOS 리포트: [`reports/qlstm13_oos_report_q1_2026.json`](reports/qlstm13_oos_report_q1_2026.json)
R-배수 분포: [`reports/qlstm13_rmultiples_q1_2026.csv`](reports/qlstm13_rmultiples_q1_2026.csv)

### 6.3 행동 알파 — 파이프라인 B 기준선

전략: 펀딩레이트 스퀴즈 + EMA-200 롱 온리 | 5× 실효 레버리지

| 기간 | ROI | 승률 | 트레이드 수 | 최대 낙폭 |
|---|---|---|---|---|
| **2023 (진정 OOS — 파라미터 최적화 미사용)** | **+23.63%** | 34.5% | 29 | — |
| 2024 | +87.44% | 38.1% | 41 | — |
| 2025 | +12.05% | 36.2% | 21 | — |
| 2026 Q1 | +3.18% | — | 4 | — |
| **누적 (3.25년)** | **+126.30%** | **36.8%** | **95** | **16.27%** |

2023년 데이터는 파라미터 선택에서 완전히 제외되었으며 — 구조적 알파의 순수 OOS 테스트를 구성합니다.

### 6.4 물리학 및 수학 품질 점수

문헌 벤치마크 대비 자체 평가:

| 차원 | 기준선 | 현재 구현 | 인식론적 한계 |
|---|---|---|---|
| **물리학** | 3.5 / 10 | **7.0 / 10** | ~9.8 (Cramér–Rao 한계) |
| **수학** | 4.5 / 10 | **7.5 / 10** | ~9.9 (Lyapunov 수렴) |

---

## 7. 한계 및 활성 버그

### 7.1 활성 버그 목록 (2026-03-24 기준)

| # | 버그 | 파일 | 상태 |
|---|---|---|---|
| 1 | `_koop_vecs = zeros` → 입력 불변 모델 | `src/data/spectral_decomposer.py` | 🔴 수정 대기 |
| 2 | `logit_proj` 편향 +0.352 LONG-SHORT | `src/models/integrated_agent.py` | 🔴 수정 대기 |
| 3 | `F.relu(advantages)` → 자기 강화 루프 | `src/models/loss.py` | 🔴 수정 대기 |

회귀 테스트: [`tests/test_agent_short_signal.py`](tests/test_agent_short_signal.py)

### 7.2 시장 예측의 인식론적 한계

완벽한 예측은 이론적으로 한계가 있습니다. **Cramér–Rao 부등식**:

```
Var(μ̂) ≥ σ² / T
```

최소 추정 오차는 표본 크기의 역수에 비례하여 스케일됩니다 — 모델 복잡성과 무관한 강한 통계적 한계. **Jarzynski 한계**는 추출 가능한 최대 알파를 추가로 제약합니다:

```
⟨exp(−βW)⟩ = exp(−β·ΔF)
```

현실적인 목표는 P(정답) = 1이 아니라, 무작위 기준선으로부터의 일관된 양의 KL 발산입니다: `E[log(P_model / P_random)] > 0`.

### 7.3 확인된 수학적 격차 (로드맵)

| 격차 | 현재 상태 | 목표 구현 |
|---|---|---|
| GAE의 마르코프 가정 | δ_t = r_t + γV(s_{t+1}) − V(s_t)가 장기 의존성 무시 | GLE 메모리 커널 γ(t) = Σ a_k exp(−b_k t) |
| 경로 적분 손실 근사 | J(τ) = Σ γ^t ΔV_t (안장점 구조 없음) | Onsager–Machlup S[x] = ∫(ẋ−μ)²/(2σ²) dt |
| 스칼라 크리틱이 꼬리 위험 버림 | CriticHead → E[Z(s,a)]만 출력 | IQN 분포적 RL → CVaR_α[Z] |
| PCA 대신 쿠프만 동역학 필요 | 선형 분산 분해 | EDMD K = G⁻¹A (영벡터 초기화 버그 수정 후) |
| VQC의 불모 고원 | 무작위 파라미터 초기화 | 양자 자연 경사법 (QFI 계량) |

---

## 8. 빠른 시작

### 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # BYBIT_API_KEY, BYBIT_API_SECRET 입력
```

### 행동 알파 백테스트 (검증됨)

```bash
python scripts/backtest_behavioral.py \
    --signals fr --long-only --trend-ema 200 --fr-z-thr 2.5
```

### QLSTM 파이프라인 (버그 수정 후)

```bash
# 1. 기존 체크포인트 삭제
rm data/quantum_checkpoint.npz

# 2. 재학습 (1시간봉, 365일)
python scripts/train_quantum_v2.py \
    --symbol BTCUSDT --timeframe 1h --days 365 --epochs 30 --device cuda

# 3. Gate 1/2 검증
python scripts/backtest_structural.py --days 90

# 4. TUI (기본값: 페이퍼 트레이딩)
python scripts/run_quantum_tui.py --mode paper
```

### 구조적 백테스트

```bash
python scripts/backtest_structural.py --days 90 --symbol BTCUSDT
```

### 테스트 실행

```bash
pytest -q  # test_agent_short_signal.py (입력 불변성 회귀 테스트) 포함
```

---

## 프로젝트 구조

```
.
├── src/
│   ├── data/           # Bybit 클라이언트, EDMD 스펙트럼 분해기
│   ├── models/         # QLSTM, VQC, 고급 물리학, 피처 V4/V5
│   ├── features/       # 구조적 13차원 피처 빌더
│   ├── strategies/     # 레짐 게이트, HMM 레짐
│   ├── risk/           # 킬 스위치, 회로 차단기, EV 추적기
│   ├── storage/        # SQLite 감사 추적
│   └── app/            # Textual TUI
├── scripts/
│   ├── backtest_behavioral.py   # FR_LONG + EMA200 백테스트
│   ├── backtest_structural.py   # 구조적 13차원 백테스트
│   ├── train_quantum_v2.py      # QLSTM 학습
│   ├── debug_long_bias.py       # 100% LONG 편향 근본 원인 분석 도구
│   └── train_synthetic_bias_test.py  # 입력 불변성 테스트
├── backtesting/
│   ├── validation.py   # Gate 1 (WR > BEP) + Gate 2 (부트스트랩 p-값)
│   └── runner.py
├── reports/
│   ├── qlstm13_oos_report_q1_2026.json   # 2026 Q1 OOS 결과
│   ├── qlstm13_rmultiples_q1_2026.csv    # R-배수 분포
│   ├── liq_cascade_dynamics.png          # 청산 연쇄 분석
│   ├── liq_alpha_halflife.png            # 알파 감쇠 분석
│   └── structural_trades_*.csv           # 구조적 신호 트레이드 로그
├── docs/
│   ├── onchain_feature_design.md         # 다음 단계 온체인 피처 설계서
│   ├── postmortem_28dim_model.md         # 28차원 모델 실패 분석
│   ├── postmortems/QLSTM_v3_failure.md  # V3 모델 포스트모텀
│   └── validation_framework_spec.md     # Gate 1/2 명세서
├── tests/
│   └── test_agent_short_signal.py       # CI 회귀: 신호 편향 테스트
└── configs/
    └── default.yaml
```

---

> **참고:** `Testnet_API.md`와 `.env`는 이 저장소에서 제외되어 있습니다. 모든 API 키는 환경 변수에서만 불러옵니다. 페이퍼 트레이딩이 기본 모드이며 — 라이브 트레이딩은 명시적 `--mode live` 플래그가 필요합니다.

---

[English README](README.md)
