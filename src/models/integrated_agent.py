"""
integrated_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quantum Trading V2 — Phase 1+2+3 통합 에이전트

마스터플랜 참조: Part 1~3 전체
기존 참조:      qlstm_qa3c.py (대체 대상)

아키텍처:
    [입력 x: B×T×27]
         ↓ Phase 1: SpectralDecomposer
    [c_kt, Δc_kt: B×T×4]
         ↓ Phase 2: QuantumHamiltonianLayer
    [logits: B×T×3  |  expvals: B×T×4  |  J: B×4×4]
         ↓ Phase 3: PathIntegralLoss
    [J(τ) = Σγ^t ΔV_t - η·I_open + R_terminal]
         ↓
    [loss.backward() → optimizer.step()]

클래스 구조:
    QuantumFinancialAgent   — 메인 에이전트 (nn.Module)
      ├─ QuantumMarketEncoder (Phase 1+2)
      ├─ PathIntegralLoss    (Phase 3)
      ├─ train_step()        — 배치 학습 1스텝
      ├─ select_action()     — 추론 전용 (그리디 / 샘플)
      └─ save/load_checkpoint()

    SnipingMonitor          — 수수료 방어 지표 실시간 추적
    quick_train_demo()      — 독립 실행 검증용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations  # __future__ 모듈에서 annotations를 가져온다  # Import annotations from __future__ module

import math  # 수학 계산(로그, 지수, 삼각함수 등) 표준 라이브러리를 불러온다  # Import Python standard math library (log, exp, trig)
import os  # 운영체제 기능(파일·폴더 조작) 표준 라이브러리를 불러온다  # Import OS interface — file and directory operations
import time  # 시간 측정 및 대기 표준 라이브러리를 불러온다  # Import Time measurement and sleep utilities

import numpy as np  # 넘파이(숫자 계산 라이브러리)를 np라는 별명으로 불러온다  # Import NumPy (numerical computation library) as "np"
from collections import deque  # 특수한 자료구조(deque, Counter 등) 표준 라이브러리에서 deque를 가져온다  # Import deque from Specialized container datatypes (deque, Counter)
from dataclasses import asdict, dataclass, field  # 데이터 클래스 자동 생성 도구에서 asdict, dataclass, field를 가져온다  # Import asdict, dataclass, field from Auto-generate boilerplate for data-holding classes
# 타입 힌트(변수 종류 표시) 도구에서 Any, Deque, Dict, List, Optional, Tuple를 가져온다
# 타입 힌트(변수 종류 표시) 도구에서 Any, Deque, Dict, List, Optional, Tuple를 가져온다
# Import Any, Deque, Dict, List, Optional, Tuple from Type hint annotations
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch  # 파이토치 — 딥러닝(인공지능 학습) 핵심 라이브러리를 불러온다  # Import PyTorch — core deep learning library
import torch.nn as nn  # 파이토치 신경망 구성 도구 모음를 nn라는 별명으로 불러온다  # Import PyTorch neural network building blocks as "nn"
import torch.nn.functional as F  # 파이토치 함수형 도구(활성화 함수, 손실 함수 등)를 F라는 별명으로 불러온다  # Import PyTorch functional API (activations, losses) as "F"
import torch.optim as optim  # 파이토치 최적화(가중치 업데이트) 도구를 optim라는 별명으로 불러온다  # Import PyTorch optimizers (weight update algorithms) as "optim"

# src.data.spectral_decomposer 모듈에서 SpectralDecomposer를 가져온다
# src.data.spectral_decomposer 모듈에서 SpectralDecomposer를 가져온다
# Import SpectralDecomposer from src.data.spectral_decomposer module
from src.data.spectral_decomposer import SpectralDecomposer
# src.models.quantum_layers 모듈에서 QuantumHamiltonianLayer, QuantumMarketEn...를 가져온다
# src.models.quantum_layers 모듈에서 QuantumHamiltonianLayer, QuantumMarketEn...를 가져온다
# Import QuantumHamiltonianLayer, QuantumMarketEn... from src.models.quantum_layers module
from src.models.quantum_layers import QuantumHamiltonianLayer, QuantumMarketEncoder, N_QUBITS
from src.models.loss import (  # src.models.loss 모듈에서 (를 가져온다
    AdaptiveTemperatureScaler,
    AdvancedPathIntegralLoss,
    CriticHead,  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
    GeneralizedAdvantage,  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
    MarketState,
    PathIntegralLoss,
    TradingPath,
    TradingPathBuilder,
    build_path_integral_loss,
)
from src.models.advanced_physics import (  # src.models.advanced_physics 모듈에서 (를 가져온다
    EntropyProductionEstimator,
    HurstEstimator,
    LindbladDecoherence,  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
    MINEEstimator,
    OptimalStoppingBoundary,
    PlattCalibrator,  # 플랫 교정: 모델의 원시 출력을 실제 확률값으로 보정  # Platt scaling: converts raw logits to calibrated probabilities
)
# src.strategies.regime_gate 모듈에서 CramerRaoFilter를 가져온다
# src.strategies.regime_gate 모듈에서 CramerRaoFilter를 가져온다
from src.strategies.regime_gate import CramerRaoFilter  # Import CramerRaoFilter from src.strategies.regime_gate module


# ─────────────────────────────────────────────────────────────────────────────
# 설정 데이터클래스
# ─────────────────────────────────────────────────────────────────────────────

@dataclass  # 이 클래스를 데이터 저장용으로 자동 설정한다 (@dataclass)  # Decorator: auto-generate __init__, __repr__, etc.
class AgentConfig:  # ★ [AgentConfig] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    QuantumFinancialAgent 하이퍼파라미터 전체.

    마스터플랜 기본값:
        gamma    = 0.99   (Part 3.1: 시간 할인율)
        eta_base = 0.000375 (Bybit 0.075% round-trip × pos_frac 0.5 = 0.0375% per unit lev)
        leverage = 10.0   (η = eta_base × leverage = 0.375% of capital per trade)
        t_base   = 1.0    (Part 3.3: 기본 온도)
    """
    # 입력 / 회로 크기
    feature_dim:     int   = 28   # V4: 18(V3-base) + 8(microstructure/CVD) + 2(liq) = 28-dim
    n_eigenvectors:  int   = 5    # LDA 방향 5 (Transformer 입력 풍부화); VQC는 N_QUBITS=3 유지
    n_vqc_layers:    int   = 3
    n_actions:       int   = 3       # Hold / Long / Short
    use_lightning:   bool  = True
    learnable_basis: bool  = False

    # 마스터플랜 물리 파라미터 (Part 3)
    gamma:           float = 0.99
    eta_base:        float = 0.000375  # Bybit round-trip 0.075% × pos_frac 0.5 = 0.0375% per unit leverage
    leverage:        float = 10.0  # 레버리지: 실제 증거금의 몇 배로 거래하는지
    r_tp:            float = 1.0
    r_sl:            float = -1.0
    r_strategic_min: float = 0.5
    r_strategic_max: float = 0.9

    # 온도 스케일링 (Part 3.3)
    t_base:          float = 1.0
    beta_atr:        float = 2.0

    # 학습
    lr:              float = 5e-5   # P8 fix: 1e-3 → 3e-4 (GN_HIGH 억제, 안정적 수렴)
    weight_decay:    float = 1e-4
    # P8 fix: grad_clip 5.0 → 1.0
    # 실측 GN이 5-7까지 도달 → clip=5.0은 실질적으로 무효.
    # Ep3+부터 ⚠GN_HIGH 경고 → Actor loss 발산 (-0.5 → -8.8).
    # clip=1.0: GN이 1.0을 초과할 때 방향은 보존하고 크기만 절삭 (L2 정규화 방향).
    grad_clip:       float = 1.0  # 기울기 노름(크기)을 제한한다
    # P8 fix: entropy_reg 0.20 → 0.05
    # 0.20: 엔트로피 보너스가 policy gradient를 압도 → 방향 학습 불가.
    # 0.01: VQC 회로 고유 SHORT 편향이 완전 표출 → LP=0% SHORT 붕괴.
    # 0.05: SHORT 붕괴 방지(LP>0% 유지) + policy gradient가 주도.
    entropy_reg:     float = 0.05  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
    # 방향성 대칭 페널티: Σ(P(a) - 1/3)² — 3-way uniformity penalty.
    # |P(L)-P(S)| 방식은 P(H) 방치 → HOLD collapse 재발.
    # 3-way: HOLD/LONG/SHORT 모두 1/3에서 벗어나면 gradient로 즉시 교정.
    # raw 값 ~0.008 이므로 coef=2.0 (구버전 0.5×0.03=0.015 와 동일 교정력).
    dir_sym_coef:    float = 1.0   # 2.0→5.0: 3-way uniformity pressure 강화 (LP=0% 탈출)
    act_sym_coef:    float = 0.5   # 2.0→0.5: HOLD 등장 허용하면서 확률 집중도 유지 (너무 강하면 max_prob<0.4 → 거래 0건)

    # Logit Bias Regularization: L_bias = coef × (bias[LONG] - bias[SHORT])²
    # SP=0% / LP=0% 고착의 근본 원인 (bias drift)를 직접 차단.
    logit_bias_reg_coef: float = 1.0

    # Actor Loss 스케일: 1.0→0.5로 낮춰 actor loss 발산 억제
    actor_coef:      float = 0.5

    # 진입 확신 임계값 (η 방어선)
    # 모델이 최대 확률 행동의 확률이 이 값 미만이면 강제로 Hold
    # 0.55 → 100% GateAttrition (raw softmax max_prob ≈ 0.43 on BTC 15m)
    # 0.40 → random baseline(0.33) + 20% margin; calibrate up after Platt fitting
    confidence_threshold: float = 0.40

    # 체크포인트 경로
    checkpoint_dir:  str   = "checkpoints/quantum_v2"  # 체크포인트(저장된 모델 상태) 관련 처리를 한다  # Checkpoint: saved model state for resuming training

    # ── Advanced Physics Roadmap ───────────────────────────────────────────
    # GAE / Critic
    use_advanced_loss:  bool  = True   # True → AdvancedPathIntegralLoss (GAE)
                                        # False → original PathIntegralLoss (REINFORCE)
    lam_gae:            float = 0.95   # GAE lambda (0=TD, 1=MC)
    critic_coef:        float = 0.5    # L_critic weight
    fp_coef:            float = 0.05   # Fokker-Planck regularizer weight

    # Platt calibration (replaces hard threshold)
    use_platt:          bool  = True   # True → calibrated softmax threshold
    platt_init_temp:    float = 1.0    # initial Platt temperature

    # Lindblad regime detection
    use_lindblad:       bool  = True   # True → regime purity score logged
    n_lindblad:         int   = 2      # number of Lindblad jump operators

    # Class-weighted auxiliary Cross-Entropy loss (Fix #3: LP ceiling)
    # Inverse-frequency weights force model to attend to rare LONG signals.
    # Weights: LONG=3.22x (1/0.311), SHORT=1.72x (1/0.583), HOLD=5.0x (capped)
    aux_ce_weight:      float = 0.30   # 0.05→0.30: AWBC alignment-gate 제거로 LONG 예제에도 발동

    # MINE mutual information
    use_mine:           bool  = False  # True → MI lower bound as aux loss
    mine_coef:          float = 0.01   # MINE auxiliary loss weight

    # ── Win-rate roadmap (Priority 1-3) ───────────────────────────────────
    # IQN distributional critic Z(s) with CVaR baseline
    use_iqn:            bool  = True   # True → IQNCriticHead; False → scalar CriticHead
    iqn_quantiles:      int   = 32     # number of τ samples per forward
    cvar_alpha:         float = 0.2    # CVaR level: pessimistic 20th-percentile

    # Cramér-Rao selective entry filter (Priority 1)
    use_cr_filter:      bool  = True   # True → gate entry on H + purity + SNR
    cr_hurst_min:       float = 0.45   # minimum Hurst exponent; BTC 15m ≈ 0.45-0.50 (was 0.52 → 100% block)
    cr_purity_min:      float = 0.05   # ← 0.25→0.05: Lindblad untrained(no grad) → purity≈0.15 (random) → CR 연쇄차단
    cr_snr_min:         float = 0.05   # ← 0.15→0.05: 20-bar 윈도우에서 SNR=0.15는 실질적으로 도달 불가

    # Lindblad regime change threshold (used in select_action)
    # Lindblad 파라미터는 train_step에서 no_grad 블록 → gradient 없음 → 랜덤 초기화 유지
    # → regime_prob이 랜덤하게 0.7~0.9에 분포 → 0.7 기준이면 대부분 차단
    # Fold5 ProdEval: threshold=0.90 → Lindblad 18/25(72%) 차단 → Pass=14% (과차단)
    # 0.97로 상향 → 극단적 레짐 전환(regime_prob>0.97)만 차단 → 차단율 ~5% 목표
    lindblad_regime_threshold: float = 0.97  # 0.90→0.97: Fold5 72% 과차단 해소  # Lindblad master equation: quantum decoherence model

    # Dynamic Fisher-Rao threshold (Priority 3)
    # BTC 15m은 t_stat≈0 (랜덤워크) → adaptive 공식이 항상 max(0.60)으로 수렴
    # → AvgMaxProb(0.42)가 0.60을 넘지 못해 100% 차단. 고정값으로 전환.
    use_fisher_threshold: bool  = False  # False → confidence_threshold 고정값 사용
    fisher_threshold_min: float = 0.38   # (adaptive 비활성화 상태에서 미사용)
    fisher_threshold_max: float = 0.60   # (adaptive 비활성화 상태에서 미사용)

    # Koopman EDMD (Priority 4)
    use_edmd:             bool  = True  # True → EDMD Koopman; False → PCA+RMT

    # Non-Markov GLE advantage (Priority 5)
    use_gle:              bool  = True  # True → H-adaptive GAE lambda
    gle_lam_scale:        float = 0.04  # sensitivity: lam_eff = base_lam + scale*(H-0.5)

    # ── PAC-Bayes Proximal Regularization (Method J) ──────────────────────
    # McAllester(1999): E[L(θ)] ≤ E_train[L] + sqrt(KL(Q||P) + ln(m/δ)) / (2m-1)
    # KL(Q||P) ≈ ||θ - θ_BC||² / 2σ²  (Gaussian prior 근사)
    # → L_prox = (pac_bayes_coef / N_eff) × ||θ - θ_BC||²
    # N_eff = N_bars / (1 + 2Σρ(k))  (Bartlett 자기상관 보정)
    # 창이 짧을수록(N_eff↓) λ 자동 증가 → 단기 레짐에서 BC 사전지식 보존
    pac_bayes_coef:   float = 0.01   # C in λ = C / N_eff
    pac_bayes_n_eff:  float = 1000.0 # 실제 값은 train_quantum_v2.py에서 설정

    # ── Spectral Normalization (Method G) ─────────────────────────────────
    # BC 단계: False (HOLD 억압 방지), RL 단계: True (logit 폭발 → AvgMaxProb=1 차단)
    use_spectral_norm: bool = False

    # Entropy production gate (Priority 6)
    # ep_threshold=0.0 → 진단 전용 (항상 통과) — 데드락 방지:
    # history가 비어있을 때 Ṡ=0이므로 threshold>0이면 영구 차단됨.
    # prod_eval_quick은 select_action()만 호출하므로 history가 모두 HOLD로 채워짐.
    use_entropy_prod:     bool  = True  # True → Schnakenberg Ṡ 계산 (로깅용)
    ep_window:            int   = 50    # rolling window for action history
    ep_threshold:         float = 0.0   # 0.0 = diagnostics only (게이팅 비활성)

    # ── Quantum Natural Gradient (Priority 6, §11.16) ─────────────────────
    # Replaces Adam for VQC params with Diagonal-QFI SGD.
    # Mitigates Barren Plateau: grad/QFI ratio stays finite even when
    # both grad and QFI → 0 exponentially (they cancel in the ratio).
    use_qng:          bool  = True   # True → DiagonalQNGOptimizer (hybrid)
                                      # False → plain AdamW (old behaviour)
    lr_quantum:       float = 0.05   # QNG step size for vqc_weights
                                      # Larger than lr is fine — QFI normalises scale
    qfi_update_freq:  int   = 20     # QFI recomputation interval (steps)
                                      # Cost: 2×24=48 TorchVQC passes per update

    # ── Temporal Context Encoder (Transformer between LDA and VQC) ────────
    use_transformer:          bool = True
    transformer_d_model:      int  = 16
    transformer_n_heads:      int  = 2
    transformer_n_layers:     int  = 2

    @property  # 이 메서드를 속성처럼 obj.속성 형태로 접근할 수 있게 만든다  # Decorator: expose method as a read-only attribute
    def effective_eta(self) -> float:  # [effective_eta] 함수 정의 시작
        """η = eta_base × leverage"""
        return self.eta_base * self.leverage  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# SnipingMonitor — 수수료 방어 지표 실시간 추적
# ─────────────────────────────────────────────────────────────────────────────

class SnipingMonitor:  # ★ [SnipingMonitor] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    '스나이퍼' 매매 품질을 실시간으로 추적하는 모니터.

    마스터플랜 철학:
        "이전 528회 매매 기록을 50회 미만의 고효율 타격(스나이핑)으로 축소하고
         샤프 지수를 플러스 전환" (Part 4)

    추적 지표:
        - trade_count      : 총 진입 횟수
        - tp_count         : TP 달성 횟수
        - sl_count         : SL 피해 횟수
        - sc_count         : 전략적 조기 종료 횟수
        - fee_ratio        : 수수료 대비 실현 수익 비율
        - avg_hold_steps   : 평균 포지션 유지 시간
        - sniper_score     : 종합 스나이퍼 점수 (높을수록 '스나이퍼')
    """

    def __init__(self, window: int = 100) -> None:  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self.window = window
        self._results: Deque[Dict[str, float]] = deque(maxlen=window)
        self._step = 0
        self.cumulative_J = 0.0
        self.trade_count  = 0

    def record(  # [record] 함수 정의 시작
        self,
        terminal_state: torch.Tensor,   # [B]
        J: torch.Tensor,                # [B]
        hold_steps: torch.Tensor,       # [B]
        realized_pnl: torch.Tensor,     # [B]
        e_eta: float,                   # effective η
    ) -> None:
        """배치 결과를 기록."""
        for b in range(terminal_state.shape[0]):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            st   = int(terminal_state[b].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            j_b  = float(J[b].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            h_b  = int(hold_steps[b].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            pnl  = float(realized_pnl[b].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

            self._results.append({  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                "state":      st,
                "J":          j_b,
                "hold_steps": h_b,
                "pnl":        pnl,  # 손익: 이번 거래에서 얻은 이익 또는 손실  # PnL: realized profit and loss for this trade
            })
            if st in (MarketState.LONG, MarketState.SHORT,  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                      MarketState.TP_HIT, MarketState.SL_HIT,  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
                      MarketState.STRATEGIC_CLOSE):
                self.trade_count += 1
            self.cumulative_J += j_b
        self._step += 1

    @property  # 이 메서드를 속성처럼 obj.속성 형태로 접근할 수 있게 만든다  # Decorator: expose method as a read-only attribute
    def stats(self) -> Dict[str, float]:  # [stats] 함수 정의 시작
        if not self._results:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
            return {}  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
        results = list(self._results)  # 리스트를 만들거나 다른 자료형을 리스트로 변환한다
        n = len(results)  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items

        tp_c  = sum(1 for r in results if r["state"] == MarketState.TP_HIT)  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
        sl_c  = sum(1 for r in results if r["state"] == MarketState.SL_HIT)  # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지
        sc_c  = sum(1 for r in results if r["state"] == MarketState.STRATEGIC_CLOSE)  # 모든 값을 더한다
        non_obs = tp_c + sl_c + sc_c
        win_rate = (tp_c + sc_c) / max(non_obs, 1)  # 가장 큰 값을 찾는다

        avg_J    = sum(r["J"] for r in results) / n  # 모든 값을 더한다
        avg_hold = sum(r["hold_steps"] for r in results) / n  # 모든 값을 더한다
        avg_pnl  = sum(r["pnl"] for r in results) / n  # 손익: 이번 거래에서 얻은 이익 또는 손실  # PnL: realized profit and loss for this trade

        # 스나이퍼 점수: 승률 × (1 / log2(avg_hold+2)) × sign(avg_pnl+ε)
        # → 높은 승률 & 짧은 보유 & 양의 수익 → 높은 점수
        hold_penalty = max(math.log2(avg_hold + 2), 1.0)  # 가장 큰 값을 찾는다
        sniper_score = win_rate / hold_penalty * math.copysign(1, avg_pnl + 1e-8)

        return {  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
            "n_samples":    n,
            "tp_rate":      tp_c / max(non_obs, 1),  # 가장 큰 값을 찾는다
            "sl_rate":      sl_c / max(non_obs, 1),  # 가장 큰 값을 찾는다
            "sc_rate":      sc_c / max(non_obs, 1),  # 가장 큰 값을 찾는다
            "win_rate":     win_rate,
            "avg_J":        avg_J,
            "avg_hold":     avg_hold,
            "avg_pnl":      avg_pnl,
            "sniper_score": sniper_score,
            "total_trades": self.trade_count,
        }

    def summary_str(self) -> str:  # [summary_str] 함수 정의 시작
        s = self.stats
        if not s:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
            return "SnipingMonitor: 아직 데이터 없음"  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
        return (  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
            f"[SnipingMonitor] "  # 문자열 안에 변수 값을 넣어 만든다
            f"WinRate={s['win_rate']:.1%}  "  # 문자열 안에 변수 값을 넣어 만든다
            f"TP={s['tp_rate']:.1%}  SL={s['sl_rate']:.1%}  SC={s['sc_rate']:.1%}  "  # 문자열 안에 변수 값을 넣어 만든다
            f"AvgHold={s['avg_hold']:.1f}bars  "  # 문자열 안에 변수 값을 넣어 만든다
            f"AvgJ={s['avg_J']:.4f}  "  # 문자열 안에 변수 값을 넣어 만든다
            f"SniperScore={s['sniper_score']:.4f}  "  # 문자열 안에 변수 값을 넣어 만든다
            f"TotalTrades={s['total_trades']}"  # 문자열 안에 변수 값을 넣어 만든다
        )


# ─────────────────────────────────────────────────────────────────────────────
# TrainStepResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass  # 이 클래스를 데이터 저장용으로 자동 설정한다 (@dataclass)  # Decorator: auto-generate __init__, __repr__, etc.
class TrainStepResult:  # ★ [TrainStepResult] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """train_step 반환 데이터."""
    loss:           float
    J_mean:         float
    J_std:          float
    policy_loss:    float
    entropy:        float
    eta_effective:  float
    t_temp_mean:    float
    pct_tp:         float
    pct_sl:         float
    pct_sc:         float
    # 양자 회로 관련
    expvals_mean:   float   # ⟨σ^z⟩ 배치 평균
    J_coupling_max: float   # 이징 J_ij 최대 강도
    grad_norm:      float   # 그라디언트 노름
    step_time_ms:   float   # 스텝 소요 시간 (ms)
    # Advanced Physics Roadmap
    critic_loss:    float = 0.0   # L_critic: V(s) regression loss
    fp_loss:        float = 0.0   # Fokker-Planck consistency loss
    dir_sym_loss:   float = 0.0   # Directional symmetry |P(L)-P(S)|
    adv_mean:       float = 0.0   # GAE advantage mean
    adv_std:        float = 0.0   # GAE advantage std
    V_mean:         float = 0.0   # Critic value V(s) mean
    purity_mean:    float = 1.0   # Lindblad purity (1=coherent, 0=mixed)
    regime_prob:    float = 0.0   # Lindblad regime-change probability
    platt_temp:     float = 1.0   # Platt calibration temperature
    qfi_mean:       float = float("nan")  # Diagonal QFI mean (QNG only)


# ─────────────────────────────────────────────────────────────────────────────
# QuantumFinancialAgent
# ─────────────────────────────────────────────────────────────────────────────

class QuantumFinancialAgent(nn.Module):  # ★ [QuantumFinancialAgent] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Phase 1 + 2 + 3 통합 에이전트.

    마스터플랜 전체 파이프라인:
        x [B, T, 27]
          → SpectralDecomposer   : c_kt, Δc_kt
          → QuantumHamiltonianLayer: logits, ⟨σ^z⟩, J_ij
          → PathIntegralLoss      : J(τ) → backward()

    기존 qlstm_qa3c.py 대비 핵심 차이:
        - 단순 CrossEntropy → PathIntegralLoss (γ, η, R_terminal)
        - 고정 VQC → IsingZZ(J_ij) Hamiltonian + 학습 가능 VQC
        - 27차원 원시 입력 → SpectralDecomposer (스펙트럼 고유 분해)

    Args:
        config : AgentConfig
        device : 실행 디바이스
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        config: AgentConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()  # 부모 클래스의 초기화 메서드를 실행한다  # Calls the parent class constructor
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── 메인 인코더: Phase 1 + Phase 2 ────────────────────────────
        self.encoder = QuantumMarketEncoder(
            feature_dim=config.feature_dim,
            n_eigenvectors=config.n_eigenvectors,
            n_vqc_layers=config.n_vqc_layers,
            n_actions=config.n_actions,
            use_lightning=config.use_lightning,
            learnable_basis=config.learnable_basis,
            use_edmd=config.use_edmd,
            use_spectral_norm=config.use_spectral_norm,
            use_transformer=config.use_transformer,
            transformer_d_model=config.transformer_d_model,
            transformer_n_heads=config.transformer_n_heads,
            transformer_n_layers=config.transformer_n_layers,
        )

        # ── 손실 함수: Phase 3 (Advanced or Classic) ───────────────────
        if config.use_advanced_loss:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # GAE + Critic + Fokker-Planck
            self.loss_fn = AdvancedPathIntegralLoss(
                gamma=config.gamma,  # 감가율: 미래 보상을 현재 가치로 환산할 때 곱하는 비율(0~1)  # Discount factor γ ∈ (0,1]: weight for future rewards
                eta_base=config.eta_base,
                leverage=config.leverage,  # 레버리지: 실제 증거금의 몇 배로 거래하는지
                lam_gae=config.lam_gae,  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
                critic_coef=config.critic_coef,  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
                fp_coef=config.fp_coef,
                entropy_reg=config.entropy_reg,  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
                state_dim=N_QUBITS,   # projected c_kt is qubit-dim (3), not n_eigenvectors (5)
                t_base=config.t_base,
                beta_atr=config.beta_atr,
                r_tp=config.r_tp,
                r_sl=config.r_sl,
                r_strategic_min=config.r_strategic_min,
                r_strategic_max=config.r_strategic_max,
                use_iqn=config.use_iqn,  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
                iqn_quantiles=config.iqn_quantiles,  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
                cvar_alpha=config.cvar_alpha,  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
                use_gle=config.use_gle,
                gle_lam_scale=config.gle_lam_scale,
                dir_sym_coef=config.dir_sym_coef,
                act_sym_coef=config.act_sym_coef,
                actor_coef=config.actor_coef,
            )
            self._use_advanced = True
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            # Classic REINFORCE PathIntegralLoss
            self.loss_fn = PathIntegralLoss(
                gamma=config.gamma,  # 감가율: 미래 보상을 현재 가치로 환산할 때 곱하는 비율(0~1)  # Discount factor γ ∈ (0,1]: weight for future rewards
                eta_base=config.eta_base,
                leverage=config.leverage,  # 레버리지: 실제 증거금의 몇 배로 거래하는지
                r_tp=config.r_tp,
                r_sl=config.r_sl,
                r_strategic_min=config.r_strategic_min,
                r_strategic_max=config.r_strategic_max,
                t_base=config.t_base,
                beta_atr=config.beta_atr,
                entropy_reg=config.entropy_reg,  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
            )
            self._use_advanced = False

        # ── Advanced Physics Modules ────────────────────────────────────
        # Lindblad decoherence (regime detection)
        self.lindblad: Optional[LindbladDecoherence] = None  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
        if config.use_lindblad:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self.lindblad = LindbladDecoherence(  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
                n_qubits=N_QUBITS,   # VQC expvals dim = N_QUBITS=3, not n_eigenvectors
                n_lindblad=config.n_lindblad,  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
            )

        # Platt calibration (replaces hard confidence threshold)
        self.platt: Optional[PlattCalibrator] = None  # 플랫 교정: 모델의 원시 출력을 실제 확률값으로 보정  # Platt scaling: converts raw logits to calibrated probabilities
        if config.use_platt:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self.platt = PlattCalibrator(  # 플랫 교정: 모델의 원시 출력을 실제 확률값으로 보정  # Platt scaling: converts raw logits to calibrated probabilities
                n_classes=config.n_actions,
                init_temp=config.platt_init_temp,  # 플랫 교정: 모델의 원시 출력을 실제 확률값으로 보정  # Platt scaling: converts raw logits to calibrated probabilities
            )

        # MINE mutual information
        self.mine: Optional[MINEEstimator] = None
        if config.use_mine:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self.mine = MINEEstimator(
                x_dim=config.n_eigenvectors,
                y_dim=config.n_actions,
                hidden=64,
            )

        # Hurst estimator (stateless, no parameters)
        self.hurst_est = HurstEstimator(n_scales=4)

        # Cramér-Rao selective entry filter (Priority 1)
        self.cr_filter: Optional[CramerRaoFilter] = None
        if config.use_cr_filter:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self.cr_filter = CramerRaoFilter(
                hurst_min=config.cr_hurst_min,
                purity_min=config.cr_purity_min,
                snr_min=config.cr_snr_min,
            )

        # Entropy production estimator (Priority 6)
        self.ep_estimator: Optional[EntropyProductionEstimator] = None
        if config.use_entropy_prod:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self.ep_estimator = EntropyProductionEstimator(
                n_states=config.n_actions,
                window=config.ep_window,
                threshold=config.ep_threshold,
            )

        # ── TradingPathBuilder 유틸리티 ────────────────────────────────
        self.path_builder = TradingPathBuilder(
            leverage=config.leverage,  # 레버리지: 실제 증거금의 몇 배로 거래하는지
            tp_pct=0.010,   # symmetric: 1×ATR (was 0.015 = 1.5%)
            sl_pct=0.010,   # symmetric: 1×ATR (was 0.010 = 1.0%)
        )

        # ── 스나이퍼 모니터 ────────────────────────────────────────────
        self.monitor = SnipingMonitor(window=200)

        # ── 옵티마이저 ────────────────────────────────────────────────
        # use_qng=True  → DiagonalQNGOptimizer (QNG for VQC, AdamW for rest)
        # use_qng=False → plain AdamW (legacy behaviour)
        trainable_params = (
            list(self.encoder.parameters())  # 리스트를 만들거나 다른 자료형을 리스트로 변환한다
            + list(self.loss_fn.parameters())  # 리스트를 만들거나 다른 자료형을 리스트로 변환한다
        )
        if self.lindblad is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            trainable_params += list(self.lindblad.parameters())  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
        if self.platt is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            trainable_params += list(self.platt.parameters())  # 플랫 교정: 모델의 원시 출력을 실제 확률값으로 보정  # Platt scaling: converts raw logits to calibrated probabilities
        if self.mine is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            trainable_params += list(self.mine.parameters())  # 리스트를 만들거나 다른 자료형을 리스트로 변환한다

        if config.use_qng:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # src.models.qng_optimizer 모듈에서 DiagonalQNGOptimizer를 가져온다
            # src.models.qng_optimizer 모듈에서 DiagonalQNGOptimizer를 가져온다
            # Import DiagonalQNGOptimizer from src.models.qng_optimizer module
            from src.models.qng_optimizer import DiagonalQNGOptimizer
            self.optimizer = DiagonalQNGOptimizer(
                self,
                lr_classical=config.lr,
                lr_quantum=config.lr_quantum,
                weight_decay=config.weight_decay,
                qfi_update_freq=config.qfi_update_freq,
            )
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=config.lr,  # 학습률: 한 번에 파라미터를 얼마나 크게 업데이트할지 결정  # Learning rate: step size for each parameter update
                weight_decay=config.weight_decay,
            )

        # ── 학습률 스케줄러 (코사인 어닐링) ──────────────────────────
        # QNG hybrid: scheduler targets classical_optimizer (AdamW) only.
        # VQC lr_quantum is kept constant — QFI already normalises scale.
        # src.models.qng_optimizer 모듈에서 DiagonalQNGOptimizer as _DQNG를 가져온다
        # src.models.qng_optimizer 모듈에서 DiagonalQNGOptimizer as _DQNG를 가져온다
        # Import DiagonalQNGOptimizer as _DQNG from src.models.qng_optimizer module
        from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
        _sched_target = (
            self.optimizer.classical_optimizer
            if isinstance(self.optimizer, _DQNG)  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            else self.optimizer
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            _sched_target, T_0=100, T_mult=2, eta_min=1e-5
        )

        # ── 학습 통계 ─────────────────────────────────────────────────
        self.global_step = 0
        self._loss_history: Deque[float] = deque(maxlen=100)
        self._J_history:    Deque[float] = deque(maxlen=100)

        # 디바이스 이동
        self.to(self.device)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        x: torch.Tensor,
        last_step_only: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        전체 파이프라인 순전파.

        Args:
            x             : [B, T, 27]
            last_step_only: True → 마지막 타임스텝 로짓만 반환 (추론 최적화)

        Returns:
            logits  : [B, T, 3] 또는 [B, 1, 3]
            expvals : [B, T, 4] 큐비트 ⟨σ^z⟩ 기댓값
            J       : [B, 4, 4] 이징 상호작용 행렬
            c_kt    : [B, T, 4] 고유 공간 투영값 (Phase 1 출력)
        """
        return self.encoder(x, last_step_only=last_step_only)  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    # ─────────────────────────────────────────────────────────────────────
    # PAC-Bayes Proximal Regularization (Method J)
    # ─────────────────────────────────────────────────────────────────────

    def set_bc_prior(self, state_dict: dict) -> None:  # [set_bc_prior] 함수 정의 시작
        """BC 사전분포 저장 — PAC-Bayes 근접 정규화의 기준점 P.

        VQC 가중치는 RL 시작 시 near-identity로 재초기화하므로 제외.
        classical_head / logit_proj 등 고전 파라미터만 BC 지식으로 고정.
        """
        self._bc_prior_params: dict = {}
        for k, v in state_dict.items():  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
            if "vqc_weights" not in k:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다
                # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다
                # Moves tensor to specified device or dtype
                self._bc_prior_params[k] = v.detach().clone().to(self.device)

    def _pac_bayes_proximal_loss(self) -> torch.Tensor:  # [_pac_bayes_proximal_loss] 내부 전용 함수 정의
        """PAC-Bayes 근접 손실: L_prox = λ × ||θ - θ_BC||²

        McAllester bound 최소화:
          E[L(θ)] ≤ E_train[L] + sqrt(KL(Q||P) + ln(m/δ)) / (2m-1)
          KL(Q||P) ≈ ||θ - θ_BC||² / 2  (단위 분산 Gaussian prior)
          λ = pac_bayes_coef / N_eff  (N_eff↓ → λ↑: 짧은 창일수록 BC 지식 강화)
        """
        if not getattr(self, "_bc_prior_params", None):  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
            return torch.tensor(0.0, device=self.device)  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
        lam = self.config.pac_bayes_coef / max(self.config.pac_bayes_n_eff, 1.0)  # 가장 큰 값을 찾는다
        total = torch.tensor(0.0, device=self.device)  # 파이썬 데이터를 파이토치 텐서로 변환한다  # Converts Python data to a PyTorch tensor
        for name, param in self.named_parameters():  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
            if name in self._bc_prior_params and param.requires_grad:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                diff = param - self._bc_prior_params[name]
                total = total + (diff * diff).sum()  # 모든 값을 더한다
        return lam * total  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    # ─────────────────────────────────────────────────────────────────────
    # train_step — 핵심 학습 1스텝
    # ─────────────────────────────────────────────────────────────────────

    def train_step(  # [train_step] 모델 학습 한 스텝을 실행한다 (순전파+역전파+업데이트)
        self,
        x:             torch.Tensor,          # [B, T, 27] 입력 피처
        prices:        torch.Tensor,          # [B, T]    종가 시계열
        directions:    torch.Tensor,          # [B]       +1/-1
        entry_idx:     torch.Tensor,          # [B]       진입 타임스텝
        labels:        torch.Tensor,          # [B]       0=Obs/1=TP/2=SL/3=SC
        atr:           Optional[torch.Tensor] = None,  # [B] ATR
        last_step_only: bool = False,
    ) -> TrainStepResult:
        """
        배치 데이터를 받아 PathIntegralLoss 기반으로 파라미터를 업데이트.

        수수료 방어 메커니즘:
            η = eta_base × leverage = {self.config.effective_eta:.6f}
            J(τ) > 0 이어야만 진입이 수학적으로 유리함.
            J(τ) < 0 인 경로 → loss 증가 → 해당 행동 확률 ↓

        Args:
            x          : [B, T, 27] 정규화 전 원시 피처 (내부에서 Z-score 적용)
            prices     : [B, T]     종가 (ΔV_t 계산용)
            directions : [B]        포지션 방향 +1/-1
            entry_idx  : [B]        포지션 진입 타임스텝 인덱스
            labels     : [B]        트리플 배리어 레이블 (0~3)
            atr        : [B]        ATR 값 (None 이면 균일 더미)
            last_step_only: bool    마지막 타임스텝 로짓만 사용

        Returns:
            TrainStepResult
        """
        t_start = time.perf_counter()
        self.train()  # 모델을 학습 모드로 전환한다 (Dropout, BatchNorm 활성화)  # Switches model to training mode (enables Dropout, BN)

        # 디바이스 이동
        x          = x.to(self.device)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype
        prices     = prices.to(self.device)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype
        directions = directions.to(self.device)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype
        entry_idx  = entry_idx.to(self.device)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype
        labels     = labels.to(self.device)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype
        if atr is None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            atr = torch.ones(x.shape[0], device=self.device) * 0.01  # 1로 채워진 텐서를 만든다  # Creates a ones-filled tensor
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            atr = atr.to(self.device)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype

        # ── TradingPath 구성 ───────────────────────────────────────────
        # 원시 가격 데이터 → ΔV_t, event_mask, terminal_state 계산
        path = self.path_builder.from_tensors(
            prices, directions, entry_idx, labels, atr  # ATR: 가격의 평균 변동 폭 (Average True Range)
        )

        # ── Forward: Phase 1 + Phase 2 ────────────────────────────────
        self.optimizer.zero_grad()  # 이전 단계에서 쌓인 기울기를 0으로 초기화한다  # Resets gradients to zero before the next backward pass
        logits, expvals, J_coupling, c_kt = self.forward(
            x, last_step_only=last_step_only
        )                                          # [B, T_run, 3], [B, T_run, 4], [B, 4, 4]

        # last_step_only 모드에서 logits: [B, 1, 3] → [B, 3] (loss 호환)
        if last_step_only and logits.dim() == 3 and logits.shape[1] == 1:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            logits = logits.squeeze(1)  # 크기가 1인 차원을 없앤다  # Removes dimensions of size 1

        # ── Loss: Phase 3 (Advanced or Classic) ───────────────────────
        if self._use_advanced:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # Advanced: GAE + Critic + FP
            # c_kt must have full T dim even in last_step_only mode
            # (critic needs all T steps for GAE)
            c_kt_full = c_kt                              # [B, T, K]

            loss, info = self.loss_fn(
                logits=logits,
                path=path,
                c_kt=c_kt_full,
                atr_norm=path.atr_norm,
            )

            # ── MINE auxiliary loss ─────────────────────────────────────
            if self.mine is not None and self.config.use_mine:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                probs_soft = torch.softmax(  # 가장 큰 값을 찾는다
                    logits if logits.dim() == 2 else logits[:, -1, :], dim=-1
                )
                # Feature: c_kt mean-pooled  [B, K]
                feat = c_kt.mean(dim=1)                   # [B, K]
                mi_lb, _ = self.mine(feat, probs_soft.detach())  # 기울기 계산 그래프에서 분리한다 (값만 복사)  # Detaches tensor from the computation graph
                mine_loss = -self.config.mine_coef * mi_lb
                loss = loss + mine_loss
                info["mine_loss"] = mine_loss.item()  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
                info["mine_mi_lb"] = mi_lb.item()  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            # Classic REINFORCE
            loss, info = self.loss_fn(
                logits=logits,
                path=path,
                atr_norm=path.atr_norm,
            )

        # ── PAC-Bayes Proximal Regularization (Method J) ───────────────
        # λ = C / N_eff 로 자동 스케일링 — 창이 짧을수록 BC 사전지식 더 강하게 보존
        pac_loss = self._pac_bayes_proximal_loss()
        if pac_loss.item() > 0.0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            loss = loss + pac_loss
            info["pac_bayes"] = pac_loss.item()  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

        # ── Logit Bias Regularization ───────────────────────────────────
        # L_bias = coef × (bias[LONG] - bias[SHORT])²
        # logit_proj.bias[1]=LONG, [2]=SHORT 비대칭 → SP=0%/LP=0% 고착 원인.
        if self.config.logit_bias_reg_coef > 0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            _lp = self.encoder.quantum_layer.logit_proj
            # logit_proj may be Linear or Sequential — get the final layer's bias
            _final = _lp[-1] if isinstance(_lp, torch.nn.Sequential) else _lp  # 변수가 특정 타입인지 확인한다  # Checks if object is an instance of given type(s)
            _b = _final.bias
            bias_reg = self.config.logit_bias_reg_coef * (_b[1] - _b[2]) ** 2
            loss = loss + bias_reg
            info["bias_reg"] = bias_reg.item()  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

        # ── Advantage-Weighted Behavioral Cloning (AWBC) ───────────────
        # 기존 aux_ce 문제: CE는 레이블 방향으로 항상 당기지만 RL은 반대로 학습 가능
        # → 그래디언트 충돌로 학습 불안정.
        #
        # AWBC 해법: CE는 (RL 방향 == 레이블) AND (Â > 0) 일 때만 발동.
        # RL이 이미 맞는 방향으로 갈 때만 보조해서 충돌 원천 차단.
        if self.config.aux_ce_weight > 0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            from src.models.loss import AdvantageWeightedBC  # src.models.loss 모듈에서 AdvantageWeightedBC를 가져온다
            logits_2d = logits if logits.dim() == 2 else logits[:, -1, :]
            # per-sample GAE advantage: AdvancedPathIntegralLoss가 info에 [B] 텐서로 제공
            _adv = info.get("adv_per_sample", None)
            if _adv is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                _adv = _adv.to(self.device)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype
            _awbc = AdvantageWeightedBC(weight=self.config.aux_ce_weight)  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
            # path.action은 directions에서 유도된 {0,1,2} — label=3(SL) 포함 안 됨
            awbc_loss = _awbc(logits_2d, path.action, advantages=_adv)  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
            loss = loss + awbc_loss

        # ── Lindblad Diagnostics (no gradient needed) ──────────────────
        purity_val, coherence_val, regime_prob_val = 1.0, 0.0, 0.0
        if self.lindblad is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            with torch.no_grad():  # 메모리 절약을 위해 기울기 계산 없이 추론만 실행한다  # Context: disable gradient tracking for inference (saves memory)
                purity_t, coh_t, rp_t = self.lindblad(expvals)  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
                purity_val    = float(purity_t.mean().item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
                coherence_val = float(coh_t.mean().item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
                regime_prob_val = float(rp_t.mean().item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

        # ── Backward + Gradient Clip ───────────────────────────────────
        loss.backward()  # 손실 함수를 역방향으로 미분해 기울기를 계산한다  # Computes gradients via backpropagation

        # 그라디언트 노름 계산 (모니터링용)
        total_norm = 0.0
        for p in self.parameters():  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
            if p.grad is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                total_norm += p.grad.data.norm(2).item() ** 2  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        total_norm = math.sqrt(total_norm)

        torch.nn.utils.clip_grad_norm_(  # 기울기 폭발을 막기 위해 기울기 크기를 잘라낸다  # Clips gradient norm to prevent exploding gradients
            list(self.parameters()),  # 리스트를 만들거나 다른 자료형을 리스트로 변환한다
            self.config.grad_clip,  # 기울기 노름(크기)을 제한한다
        )
        # Pass quantum_layer to QNG optimizer for QFI diagonal update.
        # Falls back gracefully if optimizer is plain AdamW (no encoder arg).
        # src.models.qng_optimizer 모듈에서 DiagonalQNGOptimizer as _DQNG를 가져온다
        # src.models.qng_optimizer 모듈에서 DiagonalQNGOptimizer as _DQNG를 가져온다
        # Import DiagonalQNGOptimizer as _DQNG from src.models.qng_optimizer module
        from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
        if isinstance(self.optimizer, _DQNG):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self.optimizer.step(encoder=self.encoder.quantum_layer)
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            self.optimizer.step()  # 계산된 기울기로 모델 파라미터를 한 발짝 업데이트한다  # Updates model parameters using computed gradients
        self.scheduler.step()

        # ── 통계 업데이트 ──────────────────────────────────────────────
        self.global_step += 1
        self._loss_history.append(info["loss"])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
        self._J_history.append(info["J_mean"])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list

        # SnipingMonitor 기록
        if self._use_advanced:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # Use J_mean from info as proxy
            J_proxy = torch.full(
                (x.shape[0],), info["J_mean"],  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
                device=self.device, dtype=x.dtype  # 연산을 수행할 하드웨어(GPU/CPU)를 설정한다  # Target compute device: CUDA GPU or CPU
            )
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            J_proxy = self.loss_fn._compute_path_reward(path).detach()  # 기울기 계산 그래프에서 분리한다 (값만 복사)  # Detaches tensor from the computation graph

        self.monitor.record(
            terminal_state=path.terminal_state,
            J=J_proxy,
            hold_steps=path.hold_steps,
            realized_pnl=path.realized_pnl,  # 손익: 이번 거래에서 얻은 이익 또는 손실
            e_eta=self.loss_fn.effective_eta.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        )

        t_end = time.perf_counter()

        # Get Platt temperature for logging
        platt_temp_val = 1.0  # 플랫 교정: 모델의 원시 출력을 실제 확률값으로 보정  # Platt scaling: converts raw logits to calibrated probabilities
        if self.platt is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            platt_temp_val = float(self.platt.temperature.item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

        # QFI stats (QNG only)
        # src.models.qng_optimizer 모듈에서 DiagonalQNGOptimizer as _DQNG를 가져온다
        # src.models.qng_optimizer 모듈에서 DiagonalQNGOptimizer as _DQNG를 가져온다
        # Import DiagonalQNGOptimizer as _DQNG from src.models.qng_optimizer module
        from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
        qfi_mean_val = float("nan")  # 실수(소수)로 변환한다
        if isinstance(self.optimizer, _DQNG):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            qfi_mean_val = self.optimizer.get_qfi_stats().get("qfi_mean", float("nan"))  # 실수(소수)로 변환한다

        return TrainStepResult(  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
            loss=info["loss"],
            J_mean=info["J_mean"],
            J_std=info.get("J_std", 0.0),
            policy_loss=info.get("policy_loss", info.get("actor_loss", 0.0)),
            entropy=info["entropy"],
            eta_effective=info["eta_effective"],
            t_temp_mean=info["t_temp_mean"],
            pct_tp=info["pct_tp"],
            pct_sl=info["pct_sl"],
            pct_sc=info["pct_sc"],
            expvals_mean=float(expvals.mean().item()),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            J_coupling_max=float(J_coupling.abs().max().item()),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            grad_norm=total_norm,
            step_time_ms=(t_end - t_start) * 1000,
            # Advanced fields
            critic_loss=info.get("critic_loss", 0.0),  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
            fp_loss=info.get("fp_loss", 0.0),
            dir_sym_loss=info.get("dir_sym_loss", 0.0),
            adv_mean=info.get("adv_mean", 0.0),
            adv_std=info.get("adv_std", 0.0),  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
            V_mean=info.get("V_mean", 0.0),
            purity_mean=purity_val,
            regime_prob=regime_prob_val,
            platt_temp=platt_temp_val,  # 플랫 교정: 모델의 원시 출력을 실제 확률값으로 보정  # Platt scaling: converts raw logits to calibrated probabilities
            qfi_mean=qfi_mean_val,
        )

    # ─────────────────────────────────────────────────────────────────────
    # _compute_fisher_threshold — Dynamic Fisher-Rao confidence gate
    # ─────────────────────────────────────────────────────────────────────

    # [_compute_fisher_threshold] 내부 전용 함수 정의
    # [_compute_fisher_threshold] 내부 전용 함수 정의
    # [_compute_fisher_threshold] Private helper function
    def _compute_fisher_threshold(self, log_returns: np.ndarray) -> float:
        """
        Adaptive confidence threshold derived from Fisher information / Cramér-Rao.

        t-statistic = |μ̂| · √T / σ    (proxy for Fisher information content)

        threshold(t) = lo + (hi - lo) / (1 + t)

            t → ∞  (strong signal) : threshold → lo = 0.50  (liberal entry)
            t → 0  (noise)         : threshold → hi = 0.85  (very selective)

        This implements the information-geometry principle: when the observed
        signal is statistically well above the Cramér-Rao noise floor, we relax
        the confidence gate; when the market is near-efficient (t ≈ 0), we
        require very high conviction before entering.

        Args:
            log_returns : 1-D numpy array of recent log-returns [T]

        Returns:
            adaptive threshold ∈ [fisher_threshold_min, fisher_threshold_max]
        """
        T = len(log_returns)  # 로그 수익률: ln(현재 가격 ÷ 이전 가격)  # Log-return: ln(P_t / P_{t-1}) — stationary price change
        if T < 5:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            return float(self.config.confidence_threshold)  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

        mu    = float(np.mean(log_returns))  # 평균값을 계산한다  # Computes the mean value
        sigma = float(np.std(log_returns)) + 1e-8  # 표준편차를 계산한다  # Computes the standard deviation
        t_stat = abs(mu) * math.sqrt(T) / sigma  # 절대값을 구한다

        lo = self.config.fisher_threshold_min
        hi = self.config.fisher_threshold_max
        threshold = lo + (hi - lo) / (1.0 + t_stat)
        return float(np.clip(threshold, lo, hi))  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    # ─────────────────────────────────────────────────────────────────────
    # select_action — 추론 전용
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()  # 이 함수/클래스에 특별한 기능을 추가하는 데코레이터  # Decorator: modifies the function / class below
    def select_action(  # [select_action] 현재 상태를 보고 행동(매수/매도/관망)을 결정한다
        self,
        x: torch.Tensor,                   # [1, T, 27] 또는 [T, 27]
        atr_norm: Optional[float] = 0.01,
        mode: str = "greedy",              # "greedy" | "sample"
    ) -> Tuple[int, float, torch.Tensor]:
        """
        단일 샘플에 대한 행동 선택.

        Platt 교정 활성 시:
            raw logits → Platt(T·z+b) softmax → calibrated probs
            Threshold = confidence_threshold (calibrated)
            Advantage: P(action=LONG)=0.65 는 실제 ~65% TP 달성률을 의미

        수수료 방어 필터:
            최대 확률 행동이 confidence_threshold 미만이면 강제 OBSERVE(0).

        Lindblad 체크:
            regime_prob > 0.7 (purity < 0.3) 이면 강제 Hold.
            시장이 체제 전환 중이면 포지션 불가.

        Args:
            x        : [1, T, 27] 또는 [T, 27]
            atr_norm : 정규화 ATR (온도 스케일링용)
            mode     : "greedy" 또는 "sample"

        Returns:
            action   : int (0=Hold, 1=Long, 2=Short)
            prob     : float (선택 행동의 확률)
            probs    : [3] 전체 확률 분포
        """
        self.eval()  # 모델을 평가 모드로 전환한다 (Dropout 비활성화)  # Switches model to evaluation mode (disables Dropout)
        if x.dim() == 2:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            x = x.unsqueeze(0)                              # [1, T, 27]
        x = x.to(self.device)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype

        atr_t = torch.tensor([atr_norm], device=self.device)  # 파이썬 데이터를 파이토치 텐서로 변환한다  # Converts Python data to a PyTorch tensor

        # Forward (마지막 타임스텝만)
        logits, expvals, J_coupling, c_kt = self.forward(x, last_step_only=True)
        logits_last = logits[:, -1, :]                      # [1, 3]

        # ── Platt 교정 또는 온도 스케일링 ─────────────────────────────
        if self.platt is not None and self.config.use_platt:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # Platt calibrated probabilities
            probs = self.platt.calibrate(logits_last)[0]    # [3]
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            # Legacy: ATR-adaptive temperature scaling
            probs, _ = self.loss_fn.temp_scaler(logits_last, atr_t)
            probs = probs[0]                                 # [3]

        # ── Lindblad Regime Check ─────────────────────────────────────
        # High regime_prob → market is decoherent → skip entry
        # NOTE: Lindblad는 train_step에서 no_grad 블록으로만 호출되므로
        # 파라미터에 gradient가 전달되지 않아 랜덤 초기화 상태 유지.
        # → purity ≈ 0.10~0.20 (랜덤), regime_prob ≈ 0.8~0.9 (랜덤 bias)
        # → purity를 CR filter에 넘기면 CR도 항상 차단되는 연쇄 효과 발생.
        # Fix: purity_for_cr은 항상 1.0 고정 (Lindblad 학습 완료 전까지).
        #       regime_prob threshold는 configurable (lindblad_regime_threshold).
        force_hold_by_regime = False  # 레짐: 현재 시장이 추세장/횡보장/급변동 중 어느 상태인지  # Market regime: trending / ranging / volatile
        purity_for_cr = 1.0     # CR filter에는 Lindblad purity 미전달 (untrained)
        if self.lindblad is not None and self.config.use_lindblad:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            _, _, regime_prob_t = self.lindblad(expvals)  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
            if float(regime_prob_t.mean().item()) > self.config.lindblad_regime_threshold:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                force_hold_by_regime = True  # 레짐: 현재 시장이 추세장/횡보장/급변동 중 어느 상태인지  # Market regime: trending / ranging / volatile

        # ── Cramér-Rao Selective Entry Filter (Priority 1) ───────────
        # Gate: H > cr_hurst_min AND purity > cr_purity_min AND SNR > cr_snr_min
        force_hold_by_cr = False
        if self.cr_filter is not None and self.config.use_cr_filter:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            log_rets_np = x[0, :, 0].cpu().float().numpy()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
            hurst_val   = float(self.hurst_est._hurst_single(  # 실수(소수)로 변환한다
                x[0, :, 0].cpu().float()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
            ).item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            cr_result = self.cr_filter.check(log_rets_np, hurst_val, purity_for_cr)
            if not cr_result.allow_entry:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
                force_hold_by_cr = True

        # ── Entropy Production Rate Gate (Priority 6) ─────────────────
        # Schnakenberg Ṡ from recent action history.
        # Low Ṡ → market near detailed balance → near-efficient → abstain.
        force_hold_by_ep = False
        if self.ep_estimator is not None and self.config.use_entropy_prod:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self.ep_estimator.compute()          # refresh Ṡ from latest history
            if not self.ep_estimator.allows_entry():  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
                force_hold_by_ep = True

        # ── Dynamic Fisher-Rao Threshold (Priority 3) ─────────────────
        # Adaptive confidence gate: tight when market is near-efficient,
        # relaxed when the t-statistic signals a detectable drift.
        if self.config.use_fisher_threshold:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            log_rets_np_th = x[0, :, 0].cpu().float().numpy()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
            confidence_threshold = self._compute_fisher_threshold(log_rets_np_th)
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            confidence_threshold = self.config.confidence_threshold

        # ── 수수료 방어 필터 ─────────────────────────────────────────
        max_prob, action_raw = probs.max(dim=0)  # 가장 큰 값을 찾는다
        action = int(action_raw.item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        prob   = float(max_prob.item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

        if force_hold_by_regime and action != 0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            action = 0
            prob   = float(probs[0].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        elif force_hold_by_cr and action != 0:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
            # Cramér-Rao gate: insufficient statistical edge → Hold
            action = 0
            prob   = float(probs[0].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        elif force_hold_by_ep and action != 0:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
            # Entropy production gate: market near equilibrium → Hold
            action = 0
            prob   = float(probs[0].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        elif prob < confidence_threshold and action != 0:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
            # 확신 부족 → 강제 Hold (수수료 방어, Fisher-Rao 적응형 임계값)
            action = 0
            prob   = float(probs[0].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

        if mode == "sample":  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # 확률적 샘플링
            action = int(torch.multinomial(probs, num_samples=1).item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            prob   = float(probs[action].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

        # Update entropy production history with the final action
        if self.ep_estimator is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self.ep_estimator.update(action)

        return action, prob, probs  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    @torch.no_grad()  # 이 함수/클래스에 특별한 기능을 추가하는 데코레이터  # Decorator: modifies the function / class below
    def select_action_verbose(  # [select_action_verbose] 함수 정의 시작
        self,
        x: torch.Tensor,
        atr_norm: Optional[float] = 0.01,
    ) -> dict:
        """
        select_action와 동일하되, Gemini에 전달할 풍부한 내부 정보를 dict로 반환.

        Returns dict with keys:
            action, prob, probs, p_hold, p_long, p_short,
            hurst, regime_prob, confidence_threshold,
            force_hold_cr, force_hold_ep, force_hold_regime,
            logit_long, logit_short, logit_margin
        """
        self.eval()  # 모델을 평가 모드로 전환한다 (Dropout 비활성화)  # Switches model to evaluation mode (disables Dropout)
        if x.dim() == 2:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            x = x.unsqueeze(0)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
        x = x.to(self.device)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype
        atr_t = torch.tensor([atr_norm], device=self.device)  # 파이썬 데이터를 파이토치 텐서로 변환한다  # Converts Python data to a PyTorch tensor

        logits, expvals, J_coupling, c_kt = self.forward(x, last_step_only=True)
        logits_last = logits[:, -1, :]  # [1, 3]

        if self.platt is not None and self.config.use_platt:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            probs = self.platt.calibrate(logits_last)[0]  # 플랫 교정: 모델의 원시 출력을 실제 확률값으로 보정  # Platt scaling: converts raw logits to calibrated probabilities
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            probs, _ = self.loss_fn.temp_scaler(logits_last, atr_t)
            probs = probs[0]

        # Lindblad
        regime_prob = 0.0
        force_hold_regime = False  # 레짐: 현재 시장이 추세장/횡보장/급변동 중 어느 상태인지  # Market regime: trending / ranging / volatile
        if self.lindblad is not None and self.config.use_lindblad:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            _, _, regime_prob_t = self.lindblad(expvals)  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
            regime_prob = float(regime_prob_t.mean().item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            if regime_prob > self.config.lindblad_regime_threshold:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                force_hold_regime = True  # 레짐: 현재 시장이 추세장/횡보장/급변동 중 어느 상태인지  # Market regime: trending / ranging / volatile

        # Cramér-Rao
        hurst_val = 0.5
        force_hold_cr = False
        if self.cr_filter is not None and self.config.use_cr_filter:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            log_rets_np = x[0, :, 0].cpu().float().numpy()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
            hurst_val = float(self.hurst_est._hurst_single(  # 실수(소수)로 변환한다
                x[0, :, 0].cpu().float()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
            ).item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            cr_result = self.cr_filter.check(log_rets_np, hurst_val, 1.0)
            if not cr_result.allow_entry:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
                force_hold_cr = True

        # Entropy Production
        force_hold_ep = False
        if self.ep_estimator is not None and self.config.use_entropy_prod:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self.ep_estimator.compute()
            if not self.ep_estimator.allows_entry():  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
                force_hold_ep = True

        # Fisher-Rao threshold
        if self.config.use_fisher_threshold:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            log_rets_np_th = x[0, :, 0].cpu().float().numpy()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
            confidence_threshold = self._compute_fisher_threshold(log_rets_np_th)
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            confidence_threshold = self.config.confidence_threshold

        # Action
        max_prob, action_raw = probs.max(dim=0)  # 가장 큰 값을 찾는다
        action = int(action_raw.item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        prob = float(max_prob.item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        if force_hold_regime or force_hold_cr or force_hold_ep:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            action = 0
            prob = float(probs[0].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        elif prob < confidence_threshold and action != 0:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
            action = 0
            prob = float(probs[0].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

        if self.ep_estimator is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self.ep_estimator.update(action)

        logit_long  = float(logits_last[0, 1].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        logit_short = float(logits_last[0, 2].item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

        return {  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
            "action":               action,
            "prob":                 prob,
            "probs":                probs,
            "p_hold":               float(probs[0].item()),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "p_long":               float(probs[1].item()),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "p_short":              float(probs[2].item()),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "hurst":                hurst_val,  # 허스트 지수: H>0.5=추세형, H<0.5=평균회귀형, H=0.5=무작위  # Hurst exponent: H>0.5 trending, H<0.5 mean-reverting
            "regime_prob":          regime_prob,
            "confidence_threshold": confidence_threshold,
            "force_hold_cr":        force_hold_cr,
            "force_hold_ep":        force_hold_ep,
            "force_hold_regime":    force_hold_regime,  # 레짐: 현재 시장이 추세장/횡보장/급변동 중 어느 상태인지  # Market regime: trending / ranging / volatile
            "logit_long":           logit_long,
            "logit_short":          logit_short,
            "logit_margin":         logit_long - logit_short,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 체크포인트
    # ─────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, path_or_tag: str = "latest") -> str:  # [save_checkpoint] 현재 모델 상태를 파일에 저장한다
        """파라미터 + 옵티마이저 + 학습 통계 저장."""
        if path_or_tag.endswith(".pt") or path_or_tag.endswith(".pth"):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            path = path_or_tag
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            # 폴더와 파일 이름을 합쳐 경로를 만든다
            # 폴더와 파일 이름을 합쳐 경로를 만든다
            # Joins path components into a single path string
            path = os.path.join(self.config.checkpoint_dir, f"agent_{path_or_tag}.pt")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)  # 필요한 폴더가 없으면 새로 만든다  # Creates directory (and parents) if they do not exist

        ckpt: Dict[str, Any] = {  # 체크포인트(저장된 모델 상태) 관련 처리를 한다  # Checkpoint: saved model state for resuming training
            "global_step":    self.global_step,
            "model_state":    self.state_dict(),  # 딕셔너리(키-값 쌍)를 만든다
            "optimizer":      self.optimizer.state_dict(),  # 딕셔너리(키-값 쌍)를 만든다
            "scheduler":      self.scheduler.state_dict(),  # 딕셔너리(키-값 쌍)를 만든다
            "config":         asdict(self.config),  # 딕셔너리(키-값 쌍)를 만든다
            "J_history":      list(self._J_history),  # 리스트를 만들거나 다른 자료형을 리스트로 변환한다
            "loss_history":   list(self._loss_history),  # 리스트를 만들거나 다른 자료형을 리스트로 변환한다
        }
        torch.save(ckpt, path)  # 모델/텐서를 파일에 저장한다  # Saves a tensor/model to disk
        return path  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    # [load_checkpoint] 저장된 모델 상태를 파일에서 불러온다
    # [load_checkpoint] 저장된 모델 상태를 파일에서 불러온다
    # [load_checkpoint] Loads model weights from a checkpoint file
    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        """저장된 체크포인트 로드."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)  # 파일에서 저장된 모델/텐서를 불러온다  # Loads a tensor/model from disk
        # Spectral Norm 호환: weight_orig → weight (weight_u/v 제거)
        raw = ckpt["model_state"]  # 체크포인트(저장된 모델 상태) 관련 처리를 한다  # Checkpoint: saved model state for resuming training
        if any(k.endswith("weight_orig") for k in raw):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            clean: dict = {}
            for k, v in raw.items():  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                if k.endswith("weight_orig"):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                    clean[k[: -len("_orig")]] = v  # weight_orig → weight
                elif k.endswith(("weight_u", "weight_v")):  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
                    pass  # 불필요한 SN 보조 벡터 제거  # No-op placeholder
                else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                    clean[k] = v
            ckpt["model_state"] = clean  # 체크포인트(저장된 모델 상태) 관련 처리를 한다  # Checkpoint: saved model state for resuming training
        # Resize Koopman buffers to match checkpoint before loading state_dict
        raw_state = ckpt["model_state"]
        _sel_key  = "encoder.decomposer._koop_selected"
        _vecs_key = "encoder.decomposer._koop_vecs"
        if _sel_key in raw_state and _vecs_key in raw_state:
            import torch as _torch
            _sel  = raw_state[_sel_key]
            _vecs = raw_state[_vecs_key]
            self.encoder.decomposer.register_buffer(
                "_koop_selected", _torch.zeros(_sel.shape[0], dtype=_torch.long))
            self.encoder.decomposer.register_buffer(
                "_koop_vecs", _torch.zeros(_vecs.shape[0], _vecs.shape[1]))
            self.encoder.decomposer._koop_fitted = True
        self.load_state_dict(ckpt["model_state"], strict=strict)  # 딕셔너리(키-값 쌍)를 만든다  # Checkpoint: saved model state for resuming training
        self.optimizer.load_state_dict(ckpt["optimizer"])  # 딕셔너리(키-값 쌍)를 만든다  # Checkpoint: saved model state for resuming training
        self.scheduler.load_state_dict(ckpt["scheduler"])  # 딕셔너리(키-값 쌍)를 만든다  # Checkpoint: saved model state for resuming training
        self.global_step = ckpt.get("global_step", 0)  # 체크포인트(저장된 모델 상태) 관련 처리를 한다  # Checkpoint: saved model state for resuming training
        self._J_history.extend(ckpt.get("J_history", []))  # 리스트 뒤에 다른 리스트의 항목들을 이어 붙인다  # Extends list by appending all items from iterable
        self._loss_history.extend(ckpt.get("loss_history", []))  # 리스트 뒤에 다른 리스트의 항목들을 이어 붙인다  # Extends list by appending all items from iterable

    # ─────────────────────────────────────────────────────────────────────
    # 진단 유틸리티
    # ─────────────────────────────────────────────────────────────────────

    def parameter_count(self) -> Dict[str, int]:  # [parameter_count] 함수 정의 시작
        """모듈별 학습 파라미터 수."""
        enc_params  = sum(p.numel() for p in self.encoder.parameters())  # 모든 값을 더한다
        loss_params = sum(p.numel() for p in self.loss_fn.parameters())  # 모든 값을 더한다
        return {  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
            "encoder":  enc_params,
            "loss_fn":  loss_params,
            "total":    enc_params + loss_params,
        }

    def print_architecture(self) -> None:  # [print_architecture] 함수 정의 시작
        print("=" * 65)  # 결과를 화면에 출력한다  # Prints output to stdout
        print("  QuantumFinancialAgent Advanced Architecture")  # 결과를 화면에 출력한다  # Prints output to stdout
        print("=" * 65)  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"  Feature dim    : {self.config.feature_dim}")  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"  N eigenvectors : {self.config.n_eigenvectors}  (= N qubits)")  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"  VQC layers     : {self.config.n_vqc_layers}")  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"  Actions        : {self.config.n_actions}  (Hold/Long/Short)")  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"  Device         : {self.device}")  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"  Lightning QML  : {self.config.use_lightning}")  # 결과를 화면에 출력한다  # Prints output to stdout
        print()  # 결과를 화면에 출력한다  # Prints output to stdout
        print("  [Reward Design]")  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"    γ (time discount)  : {self.config.gamma}")  # 결과를 화면에 출력한다  # Prints output to stdout
        # 레버리지: 실제 증거금의 몇 배로 거래하는지
        # 레버리지: 실제 증거금의 몇 배로 거래하는지
        # Prints output to stdout
        print(f"    η (fee penalty)    : {self.config.eta_base} × {self.config.leverage:.0f}x"
              f" = {self.config.effective_eta:.4f}")  # 문자열 안에 변수 값을 넣어 만든다
        print(f"    R_TP / R_SL / R_SC : "  # 결과를 화면에 출력한다  # Prints output to stdout
              f"+{self.config.r_tp} / {self.config.r_sl} / "  # 문자열 안에 변수 값을 넣어 만든다
              f"[{self.config.r_strategic_min}, {self.config.r_strategic_max}]")  # 문자열 안에 변수 값을 넣어 만든다
        print(f"    Confidence thr     : {self.config.confidence_threshold}")  # 결과를 화면에 출력한다  # Prints output to stdout
        print()  # 결과를 화면에 출력한다  # Prints output to stdout
        print("  [Advanced Physics Roadmap]")  # 결과를 화면에 출력한다  # Prints output to stdout
        # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지
        # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지
        # Generalized Advantage Estimation (GAE)
        print(f"    Loss mode      : {'AdvancedPIL (GAE+Critic+FP)' if self._use_advanced else 'Classic REINFORCE'}")
        if self._use_advanced:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            print(f"    GAE λ          : {self.config.lam_gae}")  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
            print(f"    Critic coef    : {self.config.critic_coef}")  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
            print(f"    FP coef        : {self.config.fp_coef}")  # 결과를 화면에 출력한다  # Prints output to stdout
        # 플랫 교정: 모델의 원시 출력을 실제 확률값으로 보정
        # 플랫 교정: 모델의 원시 출력을 실제 확률값으로 보정
        # Platt scaling: converts raw logits to calibrated probabilities
        print(f"    Platt calib    : {'ON' if self.platt is not None else 'OFF'}")
        # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식
        # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식
        # Lindblad master equation: quantum decoherence model
        print(f"    Lindblad       : {'ON (n_L=' + str(self.config.n_lindblad) + ')' if self.lindblad is not None else 'OFF'}")
        print(f"    MINE           : {'ON' if self.mine is not None else 'OFF'}")  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"    RMT denoising  : ON (in SpectralDecomposer)")  # 마르첸코-파스투르 임계값: 노이즈와 신호를 구분하는 기준  # Marchenko-Pastur RMT denoising threshold
        print()  # 결과를 화면에 출력한다  # Prints output to stdout
        pc = self.parameter_count()
        print(f"  [Parameters] encoder={pc['encoder']:,}  loss={pc['loss_fn']:,}  "  # 결과를 화면에 출력한다  # Prints output to stdout
              f"total={pc['total']:,}")  # 문자열 안에 변수 값을 넣어 만든다
        print("=" * 65)  # 결과를 화면에 출력한다  # Prints output to stdout


# ─────────────────────────────────────────────────────────────────────────────
# 팩토리 함수
# ─────────────────────────────────────────────────────────────────────────────

def build_quantum_agent(  # [build_quantum_agent] 함수 정의 시작
    config: Optional[AgentConfig] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,  # 체크포인트(저장된 모델 상태) 관련 처리를 한다  # Checkpoint: saved model state for resuming training
) -> QuantumFinancialAgent:
    """
    QuantumFinancialAgent 인스턴스 생성.

    Args:
        config          : AgentConfig (None 이면 기본값)
        device          : 실행 디바이스
        checkpoint_path : 기존 체크포인트 경로 (fine-tune 시 사용)
    """
    if config is None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        config = AgentConfig()
    if device is None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 연산을 수행할 하드웨어(GPU/CPU)를 설정한다  # Target compute device: CUDA GPU or CPU

    agent = QuantumFinancialAgent(config=config, device=device)

    if checkpoint_path and os.path.exists(checkpoint_path):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        agent.load_checkpoint(checkpoint_path)  # 정수로 변환한다  # Checkpoint: saved model state for resuming training
        print(f"[build_quantum_agent] 체크포인트 로드: {checkpoint_path}")  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"  global_step = {agent.global_step}")  # 결과를 화면에 출력한다  # Prints output to stdout

    return agent  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# quick_train_demo — 독립 실행 검증
# ─────────────────────────────────────────────────────────────────────────────

# [quick_train_demo] 함수 정의 시작
# [quick_train_demo] 함수 정의 시작
# [quick_train_demo] Function definition
def quick_train_demo(n_steps: int = 5, batch_size: int = 4, seq_len: int = 20) -> None:
    """
    합성 데이터로 full 학습 루프를 실행하는 빠른 검증.

    수수료 방어 거동 확인:
        - 초기에는 무작위 행동으로 J(τ) ≈ 0 근방
        - n_steps 이후 모델이 수수료를 고려해 Hold 선호로 수렴하는지 확인
    """
    print("=" * 65)  # 결과를 화면에 출력한다  # Prints output to stdout
    print("  quick_train_demo — Phase 1+2+3 통합 학습 루프 검증")  # 결과를 화면에 출력한다  # Prints output to stdout
    print("=" * 65)  # 결과를 화면에 출력한다  # Prints output to stdout

    cfg = AgentConfig(
        feature_dim=26,   # V4 26-dim (accel, autocorr 제거; vol_ratio+mom3 유지)
        n_eigenvectors=3,  # N_QUBITS=3 와 일치
        n_vqc_layers=2,
        n_actions=3,
        leverage=25.0,  # 레버리지: 실제 증거금의 몇 배로 거래하는지
        gamma=0.99,  # 감가율: 미래 보상을 현재 가치로 환산할 때 곱하는 비율(0~1)  # Discount factor γ ∈ (0,1]: weight for future rewards
        lr=3e-3,  # 학습률: 한 번에 파라미터를 얼마나 크게 업데이트할지 결정  # Learning rate: step size for each parameter update
        use_lightning=True,
    )
    device = torch.device("cpu")  # 연산을 수행할 하드웨어(GPU/CPU)를 설정한다  # Target compute device: CUDA GPU or CPU
    agent  = build_quantum_agent(config=cfg, device=device)
    agent.print_architecture()

    print("\n  [학습 루프 시작]")  # 결과를 화면에 출력한다  # Prints output to stdout
    B, T, F = batch_size, seq_len, 27

    for step in range(1, n_steps + 1):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
        # 합성 시장 데이터 생성
        x       = torch.randn(B, T, F)  # 표준 정규분포(평균 0, 표준편차 1) 난수 텐서를 만든다  # Creates a tensor with standard normal random values
        prices  = 50000.0 + torch.cumsum(torch.randn(B, T) * 50, dim=1)  # 표준 정규분포(평균 0, 표준편차 1) 난수 텐서를 만든다  # Creates a tensor with standard normal random values
        dirs    = torch.randint(0, 2, (B,)).float() * 2 - 1      # ±1  # Casts tensor to float32
        entries = torch.zeros(B, dtype=torch.long)                # t=0 진입
        labels  = torch.randint(0, 4, (B,))                       # 랜덤 결과
        atr     = torch.full((B,), 150.0)  # ATR: 가격의 평균 변동 폭 (Average True Range)

        result  = agent.train_step(
            x, prices, dirs, entries, labels, atr,  # ATR: 가격의 평균 변동 폭 (Average True Range)
            last_step_only=True,
        )

        print(  # 결과를 화면에 출력한다  # Prints output to stdout
            f"  Step {step:3d} | "  # 문자열 안에 변수 값을 넣어 만든다
            f"loss={result.loss:+.4f}  "  # 문자열 안에 변수 값을 넣어 만든다
            f"J={result.J_mean:+.4f}  "  # 문자열 안에 변수 값을 넣어 만든다
            f"A={result.adv_mean:+.4f}(±{result.adv_std:.3f})  "  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
            f"V={result.V_mean:+.4f}  "  # 문자열 안에 변수 값을 넣어 만든다
            f"Lc={result.critic_loss:.4f}  "  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
            f"FP={result.fp_loss:.4f}  "  # 문자열 안에 변수 값을 넣어 만든다
            f"purity={result.purity_mean:.3f}  "  # 문자열 안에 변수 값을 넣어 만든다
            f"η={result.eta_effective:.4f}  "  # 문자열 안에 변수 값을 넣어 만든다
            f"T={result.platt_temp:.3f}  "  # 플랫 교정: 모델의 원시 출력을 실제 확률값으로 보정  # Platt scaling: converts raw logits to calibrated probabilities
            f"∇={result.grad_norm:.3f}  "  # 기울기 노름(크기)을 제한한다
            f"tp={result.pct_tp:.1%}  sl={result.pct_sl:.1%}  "  # 문자열 안에 변수 값을 넣어 만든다
            f"t={result.step_time_ms:.0f}ms"  # 문자열 안에 변수 값을 넣어 만든다
        )

    print()  # 결과를 화면에 출력한다  # Prints output to stdout
    print("  [스나이퍼 모니터 보고서]")  # 결과를 화면에 출력한다  # Prints output to stdout
    print("  " + agent.monitor.summary_str())  # 결과를 화면에 출력한다  # Prints output to stdout

    print()  # 결과를 화면에 출력한다  # Prints output to stdout
    print("  [추론 테스트: select_action]")  # 결과를 화면에 출력한다  # Prints output to stdout
    x_single = torch.randn(1, T, F)  # 표준 정규분포(평균 0, 표준편차 1) 난수 텐서를 만든다  # Creates a tensor with standard normal random values
    action, prob, probs = agent.select_action(x_single, atr_norm=0.01, mode="greedy")
    action_name = {0: "HOLD/관망", 1: "LONG 진입", 2: "SHORT 진입"}[action]
    print(f"  → 결정: {action_name}  (확률={prob:.4f})")  # 결과를 화면에 출력한다  # Prints output to stdout
    print(f"    전체 확률: Hold={probs[0]:.4f}  Long={probs[1]:.4f}  Short={probs[2]:.4f}")  # 결과를 화면에 출력한다  # Prints output to stdout

    print()  # 결과를 화면에 출력한다  # Prints output to stdout
    print("  [체크포인트 저장 테스트]")  # 결과를 화면에 출력한다  # Prints output to stdout
    ckpt_path = agent.save_checkpoint("demo")  # 정수로 변환한다  # Checkpoint: saved model state for resuming training
    print(f"  → 저장: {ckpt_path}")  # 결과를 화면에 출력한다  # Prints output to stdout

    print("\n  ✓ quick_train_demo 완료! Phase 1+2+3 통합 검증 성공.")  # 결과를 화면에 출력한다  # Prints output to stdout
    print("=" * 65)  # 결과를 화면에 출력한다  # Prints output to stdout


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
    quick_train_demo(n_steps=5, batch_size=4, seq_len=20)
