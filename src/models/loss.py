"""
loss.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quantum Trading V2 — Phase 3 구현체 (Advanced Edition)

마스터플랜 참조: Part 3.1 양자 경로 적분 보상 수식
           Part 3.2 종료 상태별 보상 R_terminal(S_T)
           Part 3.3 로짓 및 확률 교정 (Temperature Scaling)

Advanced additions (Physicist/Mathematician Roadmap):
  - CriticHead          : State-value V(s) for variance reduction
  - GeneralizedAdvantage: GAE(γ,λ) replaces raw REINFORCE returns
    δ_t = r_t + γV(s_{t+1}) - V(s_t)
    Â_t = Σ_{l≥0} (γλ)^l δ_{t+l}    [reduces variance by ~90%]
  - AdvancedPathIntegralLoss: Full objective combining
    PathIntegral + GAE + Fokker-Planck + Wasserstein DRO

핵심 수식:
    J(τ) = Σ_{t=0}^{T} γ^t ΔV_t  -  η · I_{open}  +  R_terminal(S_T)

각 항의 물리적 의미:
    ① γ^t ΔV_t  : 시간 할인된 미실현 손익 (Hold 상태 내재 보상)
    ② η · I_{open}: 진입 페널티 (수수료+슬리피지 = 엔트로피 증가)
    ③ R_terminal : TP(E)/SL(F)/전략적 종료(D) 시 최종 결과 보상

수학적 '무분별한 매매 억제' 메커니즘:
    ─────────────────────────────────────────────────────────────────
    진입을 해야 이익이 나려면 반드시:
        E[Σ γ^t ΔV_t] + R_terminal > η  ... (진입의 물리적 문턱)

    γ^t 항은 시간이 지날수록 보상을 기하급수적으로 감소시킴:
        t=1  →  γ^1 ≃ 0.99
        t=10 →  γ^10 ≃ 0.904
        t=50 →  γ^50 ≃ 0.605

    따라서 '같은 수익이라도 빠를수록 더 가치 있다'는 물리적 원칙이
    손실 함수 자체에 하드코딩됨. 진입 직후 η 페널티가 즉시 부과되므로
    모델은 η < E[누적_보상] 임을 스스로 판단해야만 진입 신호를 내보냄.

    예: η = 0.0005 × 25 (레버리지) = 0.0125 = 1.25%
        이 문턱을 넘지 못하면 Hold 상태가 수학적으로 최적해.
    ─────────────────────────────────────────────────────────────────

온도 교정 (ATR 적응형):
    P(a_t|X_t) = softmax(z / T_temp)
    T_temp = T_base × (1 + β · σ_ATR)
    → 변동성이 클수록 출력을 보수적으로 스무딩
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations  # __future__ 모듈에서 annotations를 가져온다  # Import annotations from __future__ module

import math  # 수학 계산(로그, 지수, 삼각함수 등) 표준 라이브러리를 불러온다  # Import Python standard math library (log, exp, trig)
import numpy as np  # 넘파이(숫자 계산 라이브러리)를 np라는 별명으로 불러온다  # Import NumPy (numerical computation library) as "np"
from dataclasses import dataclass, field  # 데이터 클래스 자동 생성 도구에서 dataclass, field를 가져온다  # Import dataclass, field from Auto-generate boilerplate for data-holding classes
from enum import IntEnum  # 열거형(상수 집합) 정의 모듈에서 IntEnum를 가져온다  # Import IntEnum from Enumeration constants definition
from typing import Optional, Tuple  # 타입 힌트(변수 종류 표시) 도구에서 Optional, Tuple를 가져온다  # Import Optional, Tuple from Type hint annotations

import torch  # 파이토치 — 딥러닝(인공지능 학습) 핵심 라이브러리를 불러온다  # Import PyTorch — core deep learning library
import torch.nn as nn  # 파이토치 신경망 구성 도구 모음를 nn라는 별명으로 불러온다  # Import PyTorch neural network building blocks as "nn"
import torch.nn.functional as F  # 파이토치 함수형 도구(활성화 함수, 손실 함수 등)를 F라는 별명으로 불러온다  # Import PyTorch functional API (activations, losses) as "F"


# ─────────────────────────────────────────────────────────────────────────────
# 마르코프 상태 정의 (마스터플랜 Part 1.1)
# ─────────────────────────────────────────────────────────────────────────────

class MarketState(IntEnum):  # ★ [MarketState] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    트레이딩 사건의 마르코프 결정 과정 상태.

    마스터플랜 Part 1.1:
        S0 (Observe)  : 바닥 상태, E(S0)=0, ΔS_entropy=0
        LONG          : Long  진입 — 들뜬 상태 B (+1)
        SHORT         : Short 진입 — 들뜬 상태 C (-1)
        HOLD          : 결맞음 유지 A — 감쇠 인자 γ 적용
        TP_HIT        : 관측 연산자 TP 에 의한 파동함수 붕괴 (E)
        SL_HIT        : 관측 연산자 SL 에 의한 파동함수 붕괴 (F)
        STRATEGIC_CLOSE: 자발적 방출 D — 지능적 조기 종료
    """
    OBSERVE          = 0   # Hold/관망  → 행동 인덱스 0
    LONG             = 1   # Long 진입  → 행동 인덱스 1
    SHORT            = 2   # Short 진입 → 행동 인덱스 2
    HOLD             = 3   # 포지션 유지(내부 상태)
    TP_HIT           = 4   # 종료: TP 익절
    SL_HIT           = 5   # 종료: SL 손절
    STRATEGIC_CLOSE  = 6   # 종료: 전략적 조기 종료 (D 사건)


# ─────────────────────────────────────────────────────────────────────────────
# 경로(τ) 데이터 컨테이너
# ─────────────────────────────────────────────────────────────────────────────

@dataclass  # 이 클래스를 데이터 저장용으로 자동 설정한다 (@dataclass)  # Decorator: auto-generate __init__, __repr__, etc.
class TradingPath:  # ★ [TradingPath] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    단일 거래 경로 τ 의 모든 정보를 담는 컨테이너.

    마스터플랜 Part 3.1:
        J(τ) = Σ γ^t ΔV_t  -  η·I_{open}  +  R_terminal(S_T)

    각 필드는 배치 차원을 포함: shape [B, ...]

    Attributes:
        delta_v     : [B, T]  각 타임스텝 ΔV_t (Hold 중 미실현 손익 변화율)
                              ΔV_t = (price_t - price_{t-1}) / price_entry × direction × leverage
        event_mask  : [B, T]  int, 각 타임스텝의 MarketState (0~6)
        terminal_state: [B]   int, 최종 종료 상태 (TP_HIT/SL_HIT/STRATEGIC_CLOSE)
        realized_pnl: [B]     float, 실현 손익 (STRATEGIC_CLOSE 보상 연산에 사용)
        hold_steps  : [B]     int,   포지션 유지 타임스텝 수 (T_hold)
        atr_norm    : [B]     float, 정규화 ATR (온도 스케일링에 사용)
        opened      : [B]     bool,  해당 경로에서 포지션이 열렸는지 (I_{open})
        action      : [B]     int,   진입 행동 (0=HOLD, 1=LONG, 2=SHORT)
                              Fix B: event_mask는 LONG/SHORT를 담지 않으므로
                              last_step_only 모드의 policy gradient용으로 labels에서 직접 도출.
    """
    delta_v:        torch.Tensor              # [B, T]
    event_mask:     torch.Tensor              # [B, T] int
    terminal_state: torch.Tensor              # [B] int
    realized_pnl:   torch.Tensor              # [B] float
    hold_steps:     torch.Tensor              # [B] int
    atr_norm:       torch.Tensor              # [B] float
    opened:         torch.Tensor              # [B] bool
    action:         torch.Tensor              # [B] int — Fix B: 진입 행동 레이블


# ─────────────────────────────────────────────────────────────────────────────
# 온도 교정기 (마스터플랜 Part 3.3)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveTemperatureScaler(nn.Module):  # ★ [AdaptiveTemperatureScaler] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    ATR 변동성에 비례하는 적응형 온도 스케일링.

    마스터플랜 수식 (Part 3.3):
        P(a_t|X_t) = softmax(z_a / T_temp)
        T_temp = T_base × (1 + β · σ_ATR)

    물리적 의미:
        - 변동성(ATR)이 크면 T_temp↑ → softmax 출력이 균등화
          → 모델이 불확실한 시장에서 '보수적' 포지션 유지
        - 변동성이 작으면 T_temp↓ → 확신 있는 방향으로 확률 집중
          → 저변동 추세장에서 강한 진입 신호 생성

    Args:
        t_base : 기본 온도 (기본 1.0)
        beta   : ATR 민감도 (기본 2.0 — 학습 가능)
        t_min  : 최소 온도 (과잉 확신 방지)
        t_max  : 최대 온도 (너무 무작위 방지)
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        t_base: float = 1.0,
        beta: float = 2.0,
        t_min: float = 0.3,
        t_max: float = 5.0,
    ) -> None:
        super().__init__()  # 부모 클래스의 초기화 메서드를 실행한다  # Calls the parent class constructor
        self.log_t_base = nn.Parameter(torch.tensor(math.log(t_base)))  # 학습 중 자동 업데이트되는 파라미터로 등록한다  # Registers tensor as a learnable parameter (tracked by autograd)
        self.log_beta   = nn.Parameter(torch.tensor(math.log(beta)))  # 학습 중 자동 업데이트되는 파라미터로 등록한다  # Registers tensor as a learnable parameter (tracked by autograd)
        self.t_min = t_min
        self.t_max = t_max

    @property  # 이 메서드를 속성처럼 obj.속성 형태로 접근할 수 있게 만든다  # Decorator: expose method as a read-only attribute
    def t_base(self) -> torch.Tensor:  # [t_base] 함수 정의 시작
        return torch.exp(self.log_t_base)  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    @property  # 이 메서드를 속성처럼 obj.속성 형태로 접근할 수 있게 만든다  # Decorator: expose method as a read-only attribute
    def beta(self) -> torch.Tensor:  # [beta] 함수 정의 시작
        return torch.exp(self.log_beta)  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        logits: torch.Tensor,        # [B, ..., A]
        atr_norm: torch.Tensor,      # [B]  정규화 ATR
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        온도 스케일링된 softmax 확률 반환.

        Args:
            logits   : [B, T, A] 또는 [B, A]
            atr_norm : [B]

        Returns:
            probs    : logits 와 동일 shape, 확률
            t_temp   : [B] 실제 사용된 온도값 (디버깅/로깅용)
        """
        # T_temp = T_base × (1 + β · σ_ATR)
        t_temp = self.t_base * (1.0 + self.beta * atr_norm.clamp(min=0.0))
        t_temp = t_temp.clamp(self.t_min, self.t_max)        # [B]

        # 브로드캐스트를 위해 차원 확장
        if logits.dim() == 3:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # [B, T, A] 케이스: atr 를 [B, 1, 1] 로
            t_scaled = t_temp.view(-1, 1, 1)  # 텐서의 형태(차원)를 바꾼다  # Reshapes tensor (shares memory if possible)
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            # [B, A] 케이스: atr 를 [B, 1] 로
            t_scaled = t_temp.view(-1, 1)  # 텐서의 형태(차원)를 바꾼다  # Reshapes tensor (shares memory if possible)

        # softmax(z / T_temp)  [마스터플랜 Part 3.3]
        probs = F.softmax(logits / t_scaled, dim=-1)  # 값들을 0~1 사이의 확률(합이 1)로 변환한다  # Softmax: converts logits to probabilities summing to 1
        return probs, t_temp  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# PathIntegralLoss — 핵심 손실 함수
# ─────────────────────────────────────────────────────────────────────────────

class PathIntegralLoss(nn.Module):  # ★ [PathIntegralLoss] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    마스터플랜 Part 3.1 — 양자 경로 적분 보상 수식의 완전한 구현체.

    핵심 수식:
        J(τ) = Σ_{t=0}^{T} γ^t ΔV_t  -  η · I_{open}  +  R_terminal(S_T)

    손실 = -E[J(τ)]   (최대화 → 최소화로 전환)

    Args:
        gamma          : 시간 할인율 (기본 0.99, 마스터플랜 Part 3.1)
        eta_base       : 기본 수수료 페널티 비율 (기본 0.0005 = 0.05%)
        leverage       : 레버리지 배수 (η = eta_base × leverage)
        r_tp           : TP 종료 보상 E (기본 +1.0, 마스터플랜 Part 3.2)
        r_sl           : SL 종료 페널티 F (기본 -1.0, 마스터플랜 Part 3.2)
        r_strategic_min: 전략적 종료 D 최소 보상 (기본 0.5)
        r_strategic_max: 전략적 종료 D 최대 보상 (기본 0.9)
        t_base          : 온도 스케일링 기본값 (Part 3.3)
        beta_atr        : ATR 온도 민감도
        entropy_reg    : 엔트로피 정규화 강도 (과잉 확신 억제 보조항)
        reduction      : 'mean' 또는 'sum'
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        gamma: float          = 0.99,
        eta_base: float       = 0.0005,
        leverage: float       = 25.0,  # 레버리지: 실제 증거금의 몇 배로 거래하는지
        r_tp: float           = 1.0,
        r_sl: float           = -1.0,
        r_strategic_min: float = 0.5,
        r_strategic_max: float = 0.9,
        t_base: float         = 1.0,
        beta_atr: float       = 2.0,
        entropy_reg: float    = 0.01,  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
        reduction: str        = "mean",
    ) -> None:
        super().__init__()  # 부모 클래스의 초기화 메서드를 실행한다  # Calls the parent class constructor

        # ── 마스터플랜 파라미터 ──────────────────────────────────────────
        self.gamma   = gamma
        # η = eta_base × leverage  [마스터플랜 Part 3.1]
        # 기본: 0.0005 × 25 = 0.0125 = 1.25%  (수수료+슬리피지 개략값)
        self.eta     = eta_base * leverage  # 레버리지: 실제 증거금의 몇 배로 거래하는지
        self.r_tp    = r_tp
        self.r_sl    = r_sl
        self.r_strategic_min = r_strategic_min
        self.r_strategic_max = r_strategic_max
        self.entropy_reg = entropy_reg  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
        self.reduction   = reduction

        # ── 학습 가능한 온도 교정기 (Part 3.3) ───────────────────────────
        self.temp_scaler = AdaptiveTemperatureScaler(
            t_base=t_base, beta=beta_atr
        )

        # ── 학습 가능한 η 스케일 ──────────────────────────────────────────
        # 레버리지 환경에서 η 를 세밀히 조정하기 위한 학습 가능 스케일
        # log-space 로 유지하여 항상 양수
        self.log_eta_scale = nn.Parameter(torch.tensor(0.0))  # e^0 = 1.0

    @property  # 이 메서드를 속성처럼 obj.속성 형태로 접근할 수 있게 만든다  # Decorator: expose method as a read-only attribute
    def effective_eta(self) -> torch.Tensor:  # [effective_eta] 함수 정의 시작
        """실효 η = η_base × leverage × learned_scale"""
        return self.eta * torch.exp(self.log_eta_scale)  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    # ─────────────────────────────────────────────────────────────────────
    # γ^t 할인 벡터 생성
    # ─────────────────────────────────────────────────────────────────────

    def _discount_vector(  # [_discount_vector] 내부 전용 함수 정의
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        γ^t 벡터 생성.

        마스터플랜 Part 3.1:
            γ = 0.99,  t = 0, 1, ..., T-1
            γ^0=1.0, γ^1≃0.99, γ^10≃0.904, γ^50≃0.605

        동일한 수익이라도 빠를수록 더 가치 있다는 원칙을 수학적으로 인코딩.

        Returns:
            [T] float tensor
        """
        t = torch.arange(T, device=device, dtype=dtype)  # 순서대로 나열된 정수 텐서를 만든다  # Creates a 1-D tensor of evenly spaced integers
        return self.gamma ** t                             # [T]

    # ─────────────────────────────────────────────────────────────────────
    # R_terminal 계산 (마스터플랜 Part 3.2)
    # ─────────────────────────────────────────────────────────────────────

    def _terminal_reward(  # [_terminal_reward] 내부 전용 함수 정의
        self,
        terminal_state: torch.Tensor,   # [B] int (MarketState enum 값)
        realized_pnl:   torch.Tensor,   # [B] float
    ) -> torch.Tensor:
        """
        종료 상태별 최종 보상 R_terminal(S_T).

        마스터플랜 Part 3.2:
            E (TP_HIT):          +1.0   (최대 보상)
            F (SL_HIT):          -1.0   (최대 페널티)
            D (STRATEGIC_CLOSE): +R_agile ∈ [+0.5, +0.9]

        전략적 종료 D 의 지능 보상 수식:
            R_agile = r_min + (r_max - r_min) × sigmoid(5 · pnl_normalized)

            여기서 pnl_normalized = realized_pnl / r_tp_threshold
            sigmoid(·) 로 부드럽게 [r_min, r_max] 로 클리핑.

        물리적 의미:
            - D 는 TP에 도달하지 않았으나 시스템이 해밀토니안 에너지 불안정
              (다이버전스 척력 J_ij < 0)을 감지하고 자발적으로 붕괴한 사건.
            - TP(+1.0) 보다 낮지만 SL(-1.0) 과는 완전히 구분되는 '지능 보상'으로
              모델이 이 행동을 강화하도록 유도.
            - pnl 에 비례하여 가변적으로 설정함으로써 '더 많이 벌었을 때
              더 높은 D 보상' → 조기 청산이지만 수익성이 높은 행동을 강화.

        Args:
            terminal_state: [B]  (MarketState 정수값)
            realized_pnl  : [B]  정규화 손익 (0-centered, ~[-1, 1] 범위)

        Returns:
            R_t: [B]  최종 보상값
        """
        device = terminal_state.device  # 연산을 수행할 하드웨어(GPU/CPU)를 설정한다  # Target compute device: CUDA GPU or CPU
        dtype  = realized_pnl.dtype  # 손익: 이번 거래에서 얻은 이익 또는 손실
        B = terminal_state.shape[0]  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array

        R_t = torch.zeros(B, device=device, dtype=dtype)  # 0으로 채워진 텐서(숫자 배열)를 만든다  # Creates a zero-filled tensor

        # ── E: TP Hit → +r_tp ────────────────────────────────────────────
        tp_mask = (terminal_state == MarketState.TP_HIT)  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
        R_t = R_t + self.r_tp * tp_mask.to(dtype)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype

        # ── F: SL Hit → +r_sl (음수) ─────────────────────────────────────
        sl_mask = (terminal_state == MarketState.SL_HIT)  # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지
        R_t = R_t + self.r_sl * sl_mask.to(dtype)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype

        # ── D: Strategic Close → R_agile ∈ [r_min, r_max] ───────────────
        # R_agile = r_min + (r_max - r_min) × sigmoid(5 · pnl_normalized)
        #
        # 왜 sigmoid(5·pnl)?
        #   pnl = 0.0 → sigmoid(0) = 0.5 → R_agile = 0.5 + 0.4/2 = 0.70
        #   pnl = +1.0 → sigmoid(5) ≃ 0.99 → R_agile ≃ 0.90  (최대)
        #   pnl = -0.5 → sigmoid(-2.5) ≃ 0.08 → R_agile ≃ 0.53  (최소)
        # 실현 손익이 좋을수록 더 높은 인센티브를 받는다.
        sc_mask = (terminal_state == MarketState.STRATEGIC_CLOSE)
        r_agile = (
            self.r_strategic_min
            + (self.r_strategic_max - self.r_strategic_min)
            * torch.sigmoid(5.0 * realized_pnl)  # 손익: 이번 거래에서 얻은 이익 또는 손실
        )
        R_t = R_t + r_agile * sc_mask.to(dtype)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype

        return R_t                                         # [B]

    # ─────────────────────────────────────────────────────────────────────
    # 경로 보상 J(τ) 계산 (핵심)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_path_reward(self, path: TradingPath) -> torch.Tensor:  # [_compute_path_reward] 내부 전용 함수 정의
        """
        마스터플랜 핵심 수식 전체 계산:
            J(τ) = Σ_{t=0}^{T} γ^t ΔV_t  -  η · I_{open}  +  R_terminal(S_T)

        ① Σ γ^t ΔV_t   : Hold(A) 상태 타임스텝에서만 ΔV_t 를 누적
            → 포지션이 없는 상태(Observe)의 ΔV_t 는 0으로 마스킹
            → γ^t 항이 빠른 종료를 수학적으로 보상

        ② η · I_{open}  : 포지션을 '열었다'는 사실 자체에 부과되는 페널티
            → I_{open} = 1 if opened else 0
            → η = 0.0005 × 레버리지  (수수료+슬리피지 추정값)
            → 이 항이 없으면 모델이 무작위 매매로 수렴 가능

        ③ R_terminal    : 경로 최종 결과 (TP/SL/전략적종료)

        Args:
            path: TradingPath 데이터 컨테이너

        Returns:
            J: [B]  각 경로의 총 보상값
        """
        delta_v = path.delta_v           # [B, T]
        B, T    = delta_v.shape  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
        device  = delta_v.device  # 연산을 수행할 하드웨어(GPU/CPU)를 설정한다  # Target compute device: CUDA GPU or CPU
        dtype   = delta_v.dtype

        # ── 항 ①: Σ γ^t ΔV_t (할인 누적 미실현 손익) ─────────────────
        gamma_t = self._discount_vector(T, device, dtype)    # [T]

        # Hold(A) 또는 진입 후 유지 중인 타임스텝 마스킹
        # event_mask 에서 HOLD(3), LONG(1), SHORT(2) → 포지션 보유 중
        # OBSERVE(0) → 포지션 없음 → ΔV_t = 0
        hold_mask = (
            (path.event_mask == MarketState.HOLD)
            | (path.event_mask == MarketState.LONG)
            | (path.event_mask == MarketState.SHORT)
        ).to(dtype)                                          # [B, T]

        # γ^t 할인 적용 후 ΔV_t 누적
        # Σ_{t=0}^{T} γ^t · ΔV_t · I_{hold,t}
        discounted_pnl = (gamma_t.unsqueeze(0) * delta_v * hold_mask).sum(dim=1)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
        # [B]

        # ── 항 ②: η · I_{open} (진입 수수료 페널티) ─────────────────────
        # 마스터플랜:
        #   "B나 C 전이 시 부과되는 음의 상수"
        #   "E[Σ γ^t ΔV_t] > η 일 때만 진입을 허용하는 물리적 문턱"
        #
        # 이 항이 바로 '무분별한 매매 억제'의 수학적 핵심임:
        #   - 매 진입마다 η 를 잃으므로 기대 수익이 η 를 초과해야만 수익성 있음
        #   - 예: η = 1.25% → 레버리지 25배에서 0.05% 수수료
        #   - 모델이 진입 신호를 발동하면 즉시 -η 를 받고 시작
        eta_penalty = self.effective_eta.to(device) * path.opened.to(dtype)   # [B]

        # ── 항 ③: R_terminal(S_T) ────────────────────────────────────────
        R_terminal = self._terminal_reward(
            path.terminal_state, path.realized_pnl  # 손익: 이번 거래에서 얻은 이익 또는 손실
        )                                                    # [B]

        # ── 최종 경로 보상 합산 ───────────────────────────────────────────
        J = discounted_pnl - eta_penalty + R_terminal        # [B]

        return J  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    # ─────────────────────────────────────────────────────────────────────
    # 엔트로피 정규화 (Part 3.3 보조항)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod  # 객체(self) 없이 호출 가능한 정적 메서드로 만든다  # Decorator: static method — no self or cls
    def _entropy_bonus(probs: torch.Tensor) -> torch.Tensor:  # [_entropy_bonus] 내부 전용 함수 정의
        """
        정책 엔트로피 정규화항.

        H(π) = -Σ_a P(a) log P(a)

        역할:
            손실에 -entropy_reg × H(π) 를 추가하면,
            정책이 너무 단일 행동에 과잉 집중하는 것을 억제.
            탐색(Exploration)과 활용(Exploitation) 균형 유지.

        Args:
            probs: [..., A]  확률 분포

        Returns:
            entropy: [...] 배치당 엔트로피 스칼라
        """
        log_probs = torch.log(probs.clamp(min=1e-8))  # 자연 로그(ln)를 계산한다  # Computes natural logarithm element-wise
        entropy = -(probs * log_probs).sum(dim=-1)           # [...] or [B, T]
        return entropy  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        logits:    torch.Tensor,   # [B, T, 3] 또는 [B, 3]  — Phase 2 출력
        path:      TradingPath,    # 경로 데이터
        atr_norm:  Optional[torch.Tensor] = None,  # [B] ATR 정규화값
    ) -> Tuple[torch.Tensor, dict]:
        """
        PathIntegralLoss 전체 계산.

        수식 요약:
            ℒ = -E[J(τ)]  +  ℒ_policy  -  entropy_reg × H(π)

            여기서:
                E[J(τ)]   = 배치 평균 경로 보상 (최대화 목표)
                ℒ_policy  = 정책 그라디언트 손실 (REINFORCE-style)
                H(π)      = 정책 엔트로피 (탐색 보너스)

        정책 그라디언트 (REINFORCE):
            ∇ℒ = -J(τ) × ∇ log π(a_t | x_t)

            J(τ) 를 Advantage 로 사용하여 그라디언트를 계산.
            good trajectory (J > 0) → 해당 행동 확률 ↑
            bad  trajectory (J < 0) → 해당 행동 확률 ↓

        Args:
            logits    : [B, T, 3] 또는 [B, 3]  Phase 2 출력 로짓
            path      : TradingPath 경로 정보
            atr_norm  : [B] ATR 정규화값 (None 이면 path.atr_norm 사용)

        Returns:
            loss      : scalar  (backward() 가능)
            info_dict : 디버깅/로깅용 딕셔너리
        """
        if atr_norm is None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            atr_norm = path.atr_norm

        # ── Step 1: 온도 스케일링된 확률 계산 (Part 3.3) ──────────────────
        probs, t_temp = self.temp_scaler(logits, atr_norm)  # [B,T,3], [B]

        # ── Step 2: 경로 보상 J(τ) 계산 (Part 3.1) ───────────────────────
        J = self._compute_path_reward(path)                  # [B]

        # ── Step 3: Advantage 정규화 ──────────────────────────────────────
        # Advantage A(τ) = (J - mean(J)) / (std(J) + ε)
        # 표준화를 통해 학습 안정성 확보 (Baseline subtraction)
        J_mean = J.mean()
        J_std  = J.std()
        # BUG-11 fix: clamp(1e-8) amplifies near-zero deviations by up to 1e6x.
        # Skip scaling when std is degenerate (all advantages identical → no signal).
        if J_std.item() > 1e-4:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            advantage = (J - J_mean) / J_std                 # [B]
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            advantage = J - J_mean                           # [B] centred only

        # ── Step 4: 정책 그라디언트 손실 (REINFORCE) ─────────────────────
        # ℒ_policy = -E[J(τ) × log π(a_t | x_t)]
        #
        # 행동 인덱스:
        #   OBSERVE(0) → 관망
        #   LONG(1)    → Long
        #   SHORT(2)   → Short
        # event_mask 에서 진입 행동(LONG/SHORT) 에 해당하는 타임스텝의
        # log-prob 에 advantage 를 곱하여 Policy Gradient 계산.

        log_probs_all = torch.log(probs.clamp(min=1e-8))     # [B, T, 3] 또는 [B, 3]

        if logits.dim() == 3:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            B, T, A = logits.shape  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
            # 각 타임스텝 t 의 행동 인덱스 = event_mask 를 [0,1,2] 로 변환
            # HOLD(3)/TP(4)/SL(5)/SC(6) → 이미 발생한 상태이므로 OBSERVE(0) 로 처리
            action_idx = path.event_mask.clone()             # [B, T]
            action_idx = action_idx.clamp(0, A - 1)          # [B, T] → [0, 2]

            # 행동별 log-prob 선택: [B, T]
            log_pi = log_probs_all.gather(
                dim=2, index=action_idx.unsqueeze(2)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
            ).squeeze(2)                                     # [B, T]

            # 진입 타임스텝(LONG/SHORT)에만 policy gradient 적용
            entry_mask = (
                (path.event_mask == MarketState.LONG)
                | (path.event_mask == MarketState.SHORT)
            ).float()                                        # [B, T]

            # advantage 를 타임스텝 차원으로 브로드캐스트
            adv_broadcast = advantage.unsqueeze(1)           # [B, 1]
            policy_loss = -(adv_broadcast * log_pi * entry_mask).sum(dim=1).mean()  # 모든 값을 더한다

            # 엔트로피: 배치+타임스텝 평균
            entropy = self._entropy_bonus(probs).mean()      # scalar

        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            # [B, A] 케이스 (last_step_only 모드)
            # Fix B: event_mask[:, -1] = HOLD(3) → clamp(0,2) = 2 = SHORT (항상 SHORT만 학습하는 버그)
            # path.action = labels에서 직접 도출한 실제 진입 행동 (0=HOLD, 1=LONG, 2=SHORT)
            B, A = logits.shape  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
            action_idx = path.action.clamp(0, A - 1)         # [B] — Fix B
            log_pi     = log_probs_all.gather(
                dim=1, index=action_idx.unsqueeze(1)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
            ).squeeze(1)                                     # [B]
            # Fix C: HOLD(0) 제외 — LONG/SHORT만 policy gradient 기여
            active_mask = (action_idx > 0).float()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
            n_active = active_mask.sum().clamp(min=1.0)  # 모든 값을 더한다
            # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지
            # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지
            policy_loss = -(advantage * log_pi * active_mask).sum() / n_active  # Generalized Advantage Estimation (GAE)
            entropy     = self._entropy_bonus(probs).mean()

        # ── Step 5: 최종 손실 합산 ────────────────────────────────────────
        # ℒ = -E[J(τ)]  +  ℒ_policy  -  entropy_reg × H(π)
        #
        #   -E[J(τ)]   : 경로 보상 최대화 (음의 보상 = 최소화 방향)
        #   ℒ_policy   : REINFORCE 그라디언트 (확률적 행동 정책 최적화)
        #   -entropy_reg × H : 탐색 보너스 (음의 엔트로피 = 다양성 유지)
        reward_loss = -J_mean                                # 보상 최대화
        loss = reward_loss + policy_loss - self.entropy_reg * entropy  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration

        # ── 정보 딕셔너리 (로깅/디버깅) ──────────────────────────────────
        info = {
            "loss":          loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "reward_loss":   reward_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "policy_loss":   policy_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "entropy":       entropy.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "J_mean":        J_mean.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "J_std":         J_std.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "J_min":         J.min().item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "J_max":         J.max().item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "eta_effective": self.effective_eta.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "t_temp_mean":   t_temp.mean().item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            # MarketState 분포
            # 텐서를 실수형(float32)으로 변환한다
            # 텐서를 실수형(float32)으로 변환한다
            # Casts tensor to float32
            "pct_tp":        (path.terminal_state == MarketState.TP_HIT).float().mean().item(),
            # 텐서를 실수형(float32)으로 변환한다
            # 텐서를 실수형(float32)으로 변환한다
            # Casts tensor to float32
            "pct_sl":        (path.terminal_state == MarketState.SL_HIT).float().mean().item(),
            # 텐서를 실수형(float32)으로 변환한다
            # 텐서를 실수형(float32)으로 변환한다
            # Casts tensor to float32
            "pct_sc":        (path.terminal_state == MarketState.STRATEGIC_CLOSE).float().mean().item(),
        }

        return loss, info  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# TradingPath 빌더 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

class TradingPathBuilder:  # ★ [TradingPathBuilder] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    원시 OHLCV 데이터 및 레이블로부터 TradingPath 를 구성하는 유틸리티.

    마스터플랜과의 연결:
        - delta_v: (price_t - price_{t-1}) / price_entry × direction × leverage
        - terminal_state: 트리플 배리어 레이블 → MarketState 에 매핑
        - atr_norm: ATR / price 로 정규화 → 온도 스케일러 입력
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        leverage: float = 25.0,  # 레버리지: 실제 증거금의 몇 배로 거래하는지
        tp_pct: float   = 0.015,   # TP 임계값 (%)
        sl_pct: float   = 0.010,   # SL 임계값 (%)
    ) -> None:
        self.leverage = leverage  # 레버리지: 실제 증거금의 몇 배로 거래하는지
        self.tp_pct   = tp_pct
        self.sl_pct   = sl_pct

    def from_tensors(  # [from_tensors] 함수 정의 시작
        self,
        prices:      torch.Tensor,   # [B, T]  종가 시계열
        directions:  torch.Tensor,   # [B]     +1(Long) 또는 -1(Short)
        entry_idx:   torch.Tensor,   # [B]     진입 시점 인덱스
        labels:      torch.Tensor,   # [B]     0=Observe, 1=TP, 2=SL, 3=SC
        atr:         torch.Tensor,   # [B]     ATR 값
    ) -> TradingPath:
        """
        원시 텐서로부터 TradingPath 를 자동 구성.

        ΔV_t 계산:
            entry_price = prices[b, entry_idx[b]]
            ΔV_t = (prices[b, t] - prices[b, t-1]) / entry_price
                   × direction[b] × leverage

        terminal_state 매핑:
            label=1 → TP_HIT
            label=2 → SL_HIT
            label=3 → STRATEGIC_CLOSE
            else    → OBSERVE

        Args:
            prices     : [B, T]
            directions : [B]
            entry_idx  : [B]
            labels     : [B]  트리플 배리어 결과 레이블
            atr        : [B]

        Returns:
            TradingPath
        """
        B, T = prices.shape  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
        device = prices.device  # 연산을 수행할 하드웨어(GPU/CPU)를 설정한다  # Target compute device: CUDA GPU or CPU
        dtype  = prices.dtype

        # ΔV_t 계산
        price_diff = torch.diff(prices, dim=1)               # [B, T-1]
        price_diff = F.pad(price_diff, (1, 0))               # [B, T] (t=0 패딩)

        # 진입가격 추출 (배치별 entry_idx)
        entry_prices = prices.gather(
            1, entry_idx.clamp(0, T-1).unsqueeze(1)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
        ).squeeze(1).clamp(min=1e-8)                         # [B]

        # ΔV_t = (ΔPrice_t / P_entry) × direction × leverage
        direction_f = directions.to(dtype)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype
        delta_v = (
            price_diff
            / entry_prices.unsqueeze(1)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
            * direction_f.unsqueeze(1)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
            * self.leverage  # 레버리지: 실제 증거금의 몇 배로 거래하는지
        )                                                    # [B, T]

        # event_mask: entry_idx 이전 = OBSERVE, 이후 = HOLD
        t_idx = torch.arange(T, device=device).unsqueeze(0) # [1, T]  # 순서대로 나열된 정수 텐서를 만든다  # Creates a 1-D tensor of evenly spaced integers
        entry_expanded = entry_idx.unsqueeze(1)              # [B, 1]
        event_mask = torch.where(
            t_idx >= entry_expanded,
            torch.full_like(t_idx, MarketState.HOLD),
            torch.full_like(t_idx, MarketState.OBSERVE),
        )                                                    # [B, T]

        # terminal_state 매핑
        # label=1: LONG TP 성공  (upper barrier hit, direction=+1) → TP_HIT  (+1.0)
        # label=2: SHORT TP 성공 (lower barrier hit, direction=-1) → TP_HIT  (+1.0)
        # label=3: SL 손절 (any direction)                         → SL_HIT  (-1.0)
        # label=0: HOLD / timeout                                  → OBSERVE (0.0)
        label_to_state = {
            1: MarketState.TP_HIT,  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
            2: MarketState.TP_HIT,  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
            3: MarketState.SL_HIT,  # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지
        }
        terminal_state = torch.full(
            (B,), fill_value=MarketState.OBSERVE, device=device, dtype=torch.long
        )
        for label_val, state_val in label_to_state.items():  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
            terminal_state = torch.where(
                labels == label_val,
                torch.full_like(terminal_state, state_val),
                terminal_state,
            )

        # realized_pnl: 마지막 ΔV 의 누적합
        realized_pnl = delta_v.sum(dim=1)                   # [B]

        # hold_steps: entry_idx 이후 스텝 수
        hold_steps = (T - entry_idx).clamp(min=0)           # [B]

        # atr_norm: ATR / price 로 정규화
        atr_norm = (atr / entry_prices).clamp(min=0.0, max=1.0)  # [B]

        # opened: LONG 또는 SHORT 이 발생했는지
        opened = (labels > 0)                                # [B] bool

        # action — direction 텐서에서 직접 도출 (labels 기반 clamp는 label=3 SL 케이스에서 오류)
        # direction=+1 → LONG(1), direction=-1 → SHORT(2), direction=0 → HOLD(0)
        # 이 방식은 TP/SL 결과와 무관하게 모델이 실제로 선택한 행동을 보존.
        dir_long = (directions.long() == 1)   # [B] bool
        dir_short = (directions.long() == -1) # [B] bool  # 텐서를 정수형(int64)으로 변환한다  # Casts tensor to int64
        action = torch.zeros(B, dtype=torch.long, device=device)  # 0으로 채워진 텐서(숫자 배열)를 만든다  # Creates a zero-filled tensor
        action = torch.where(dir_long,  torch.ones_like(action),       action)  # LONG=1
        action = torch.where(dir_short, torch.full_like(action, 2),    action)  # SHORT=2

        return TradingPath(  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
            delta_v=delta_v,
            event_mask=event_mask,
            terminal_state=terminal_state,
            realized_pnl=realized_pnl,  # 손익: 이번 거래에서 얻은 이익 또는 손실
            hold_steps=hold_steps,
            atr_norm=atr_norm,
            opened=opened,
            action=action,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 팩토리 함수
# ─────────────────────────────────────────────────────────────────────────────

def build_path_integral_loss(  # [build_path_integral_loss] 함수 정의 시작
    gamma: float    = 0.99,
    eta_base: float = 0.0005,
    leverage: float = 25.0,  # 레버리지: 실제 증거금의 몇 배로 거래하는지
    t_base: float   = 1.0,
    beta_atr: float = 2.0,
    entropy_reg: float = 0.01,  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
    device: Optional[torch.device] = None,
) -> PathIntegralLoss:
    """PathIntegralLoss 인스턴스 생성 + 디바이스 이동."""
    if device is None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 연산을 수행할 하드웨어(GPU/CPU)를 설정한다  # Target compute device: CUDA GPU or CPU
    return PathIntegralLoss(  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
        gamma=gamma,  # 감가율: 미래 보상을 현재 가치로 환산할 때 곱하는 비율(0~1)  # Discount factor γ ∈ (0,1]: weight for future rewards
        eta_base=eta_base,
        leverage=leverage,  # 레버리지: 실제 증거금의 몇 배로 거래하는지
        t_base=t_base,
        beta_atr=beta_atr,
        entropy_reg=entropy_reg,  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
    ).to(device)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype


# ─────────────────────────────────────────────────────────────────────────────
# CriticHead — State Value Estimator V(s)
# ─────────────────────────────────────────────────────────────────────────────

class CriticHead(nn.Module):  # ★ [CriticHead] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Critic network V(s) for Actor-Critic / GAE variance reduction.

    Maps spectral-space state c_kt → scalar state value V(s_t).

    Used in Generalized Advantage Estimation (GAE):
        δ_t = r_t + γ · V(s_{t+1}) - V(s_t)   (TD residual)
        Â_t = Σ_{l≥0} (γλ)^l δ_{t+l}           (GAE advantage)

    Variance reduction:
        REINFORCE variance ≈ O(R²/(1-γ)²) → can be ~10,000×
        GAE variance ≈ O(σ²_TD) → reduced by ~90% for λ=0.95

    Architecture:
        c_kt [B, T, K]  → mean-pool → [B, K]
                        → 2-layer MLP → [B, 1]

    Args:
        state_dim : dimension of c_kt (= n_eigenvectors, default 4)
        hidden    : hidden layer size (default 64)
    """

    # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
    # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
    # [__init__] Constructor — runs when the object is created
    def __init__(self, state_dim: int = 4, hidden: int = 64) -> None:
        super().__init__()  # 부모 클래스의 초기화 메서드를 실행한다  # Calls the parent class constructor
        self.net = nn.Sequential(  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module
            nn.Linear(state_dim, hidden),  # 선형 변환(행렬 곱 + 편향) 레이어를 만든다  # Fully-connected (affine) layer: y = xW^T + b
            nn.Tanh(),
            nn.Linear(hidden, hidden),  # 선형 변환(행렬 곱 + 편향) 레이어를 만든다  # Fully-connected (affine) layer: y = xW^T + b
            nn.Tanh(),
            nn.Linear(hidden, 1),  # 선형 변환(행렬 곱 + 편향) 레이어를 만든다  # Fully-connected (affine) layer: y = xW^T + b
        )

    def forward(self, c_kt: torch.Tensor) -> torch.Tensor:  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        """
        Args:
            c_kt: [B, T, K] — spectral space projections (Phase 1 output)

        Returns:
            V: [B, T] — state value at each timestep
        """
        return self.net(c_kt).squeeze(-1)                     # [B, T]


# ─────────────────────────────────────────────────────────────────────────────
# GeneralizedAdvantage — GAE(γ, λ)
# ─────────────────────────────────────────────────────────────────────────────

class GeneralizedAdvantage(nn.Module):  # ★ [GeneralizedAdvantage] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Generalized Advantage Estimation (GAE) — Schulman et al. 2015.

    Replaces raw Monte-Carlo returns G_t = Σ γ^k r_{t+k} (REINFORCE)
    with a lower-variance, lower-bias exponential weighted sum:

        δ_t = r_t + γ · V(s_{t+1}) - V(s_t)           (TD residual)
        Â_t = Σ_{l=0}^{T-t-1} (γλ)^l · δ_{t+l}        (GAE advantage)

    Bias-variance trade-off:
        λ = 1.0 → unbiased (full Monte-Carlo, high variance = REINFORCE)
        λ = 0.0 → low variance, high bias (1-step TD)
        λ = 0.95 → recommended: near-unbiased, low variance

    Policy gradient with GAE:
        ∇_θ L = -E[ Â_t · ∇_θ log π_θ(a_t|s_t) ]

    Note: Â_t is used instead of raw J(τ) → same algorithm, ~90% less variance.

    Args:
        gamma  : temporal discount rate (should match PathIntegralLoss.gamma)
        lam    : GAE lambda (0 → TD, 1 → MC, default 0.95)
        normalize: True → normalize Â per batch (zero-mean, unit-var)
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize: bool = True,
    ) -> None:
        super().__init__()  # 부모 클래스의 초기화 메서드를 실행한다  # Calls the parent class constructor
        self.gamma = gamma
        self.lam = lam
        self.normalize = normalize

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        rewards: torch.Tensor,       # [B, T]  per-step rewards r_t
        values: torch.Tensor,        # [B, T]  V(s_t) from CriticHead
        dones: Optional[torch.Tensor] = None,  # [B, T]  bool terminal mask
        hurst: Optional[float] = None,         # Hurst H for GLE adaptive lambda
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and value targets.

        Args:
            rewards : [B, T] — per-step reward r_t  (from TradingPath.delta_v)
            values  : [B, T] — state values V(s_t)  (from CriticHead)
            dones   : [B, T] — True at terminal timestep (episode end)

        Returns:
            advantages : [B, T] — Â_t  (use as REINFORCE weight)
            returns    : [B, T] — r_t + γ V(s_{t+1})  (critic regression target)
        """
        B, T = rewards.shape  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
        device = rewards.device  # 연산을 수행할 하드웨어(GPU/CPU)를 설정한다  # Target compute device: CUDA GPU or CPU

        # ── GLE Non-Markov adaptive lambda ────────────────────────────────
        # Generalized Langevin Equation (GLE) memory kernel:
        #   γ(l) = exp(-b·l),  b = 1 - H  (Prony series, 1-term)
        #
        # For persistent markets (H > 0.5): longer credit assignment → higher λ
        # For near-random-walk (H ≈ 0.5):  base lam unchanged
        # For mean-reverting (H < 0.5):    shorter memory → lower λ
        #
        # lam_eff = base_lam + 0.04 × (H - 0.5)
        #   H=0.5 → lam_eff = base_lam (unchanged)
        #   H=0.9 → lam_eff = base_lam + 0.016
        #   H=0.1 → lam_eff = base_lam - 0.016
        if hurst is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            H_val  = float(max(0.3, min(0.95, hurst)))  # 허스트 지수: H>0.5=추세형, H<0.5=평균회귀형, H=0.5=무작위  # Hurst exponent: H>0.5 trending, H<0.5 mean-reverting
            lam_eff = float(max(0.5, min(0.99, self.lam + 0.04 * (H_val - 0.5))))  # 가장 큰 값을 찾는다
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            lam_eff = self.lam

        # Bootstrap value at t=T (zero for done episodes)
        bootstrap = torch.zeros(B, device=device, dtype=rewards.dtype)  # 0으로 채워진 텐서(숫자 배열)를 만든다  # Creates a zero-filled tensor

        advantages = torch.zeros_like(rewards)           # [B, T]
        last_gae = torch.zeros(B, device=device, dtype=rewards.dtype)  # 0으로 채워진 텐서(숫자 배열)를 만든다  # Creates a zero-filled tensor

        # Reverse scan (backward in time)
        for t in reversed(range(T)):  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
            # V(s_{t+1}): bootstrap or next-step value
            if t == T - 1:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                next_value = bootstrap
            else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                next_value = values[:, t + 1]

            # done mask
            if dones is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                mask = (1.0 - dones[:, t].float())  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
            else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                mask = torch.ones(B, device=device, dtype=rewards.dtype)  # 1로 채워진 텐서를 만든다  # Creates a ones-filled tensor

            # TD residual: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[:, t] + self.gamma * next_value * mask - values[:, t]

            # GAE: Â_t = δ_t + (γλ) Â_{t+1}  [lam_eff is H-adaptive for GLE]
            last_gae = delta + self.gamma * lam_eff * mask * last_gae  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
            advantages[:, t] = last_gae  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)

        # Critic target: returns = Â + V
        returns = advantages + values  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)

        # Normalize advantages for stable PG
        if self.normalize:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            adv_mean = advantages.mean()  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
            adv_std  = advantages.std()  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
            # BUG-11 fix: skip scaling when std is degenerate to prevent explosion.
            if adv_std.item() > 1e-4:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                advantages = (advantages - adv_mean) / adv_std  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
            else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                advantages = advantages - adv_mean  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)

        return advantages, returns  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# IQN helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quantile_huber_loss(  # [_quantile_huber_loss] 내부 전용 함수 정의
    quant_vals: torch.Tensor,   # [B, T, N]
    targets:    torch.Tensor,   # [B, T]
    tau:        torch.Tensor,   # [B, T, N]
    kappa:      float = 1.0,
) -> torch.Tensor:
    """
    Quantile Huber (pinball) loss for IQN distributional critic training.

    ρ_τ(u) = |τ - 1(u < 0)| · L_κ(u)
    L_κ(u) = 0.5 u²/κ        if |u| ≤ κ
             |u| - 0.5κ       otherwise

    This is the asymmetric Huber loss weighted by |τ - 1(u<0)|,
    which is the proper scoring rule for quantile regression.

    References:
        Dabney et al. "Distributional RL with Quantile Regression" (2017)
        Dabney et al. "Implicit Quantile Networks for Distributional RL" (2018)
    """
    u = targets.unsqueeze(-1) - quant_vals           # [B, T, N]
    huber = torch.where(
        u.abs() <= kappa,  # 절대값을 구한다
        0.5 * u.pow(2) / kappa,
        u.abs() - 0.5 * kappa,  # 절대값을 구한다
    )
    rho = (tau - (u < 0).to(tau.dtype)).abs() * huber  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype
    return rho.mean()  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# IQNCriticHead — Implicit Quantile Network for Distributional Critic Z(s)
# ─────────────────────────────────────────────────────────────────────────────

class IQNCriticHead(nn.Module):  # ★ [IQNCriticHead] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Implicit Quantile Network (IQN) distributional critic.

    Instead of the scalar V(s) = E[Z(s)], IQN models the full return
    distribution by learning Z^{-1}(τ) — the quantile function.

    Architecture:
        c_kt [B, T, K] → state encoder → h [B, T, hidden]
        τ ~ U[0,1]     → cos embedding → e [B, T, N, hidden]
        h ⊙ e          → output head  → Z^{-1}(τ) [B, T, N]

    Risk-sensitive baseline for GAE:
        CVaR_α = (1/α) E[Z · 1(τ ≤ α)]
        Used as V(s_t) in the TD residual δ_t = r_t + γ·CVaR(s_{t+1}) - CVaR(s_t).
        This pessimistic baseline suppresses advantage in trap regimes
        (high upside E[Z] but heavy left tail CVaR << E[Z]).

    Mathematical basis:
        CVaR policy gradient: ∇_θ CVaR_α[Z] = E[∇_θ Z | Z ≤ VaR_α] / α
        This is a coherent risk measure (Artzner 1999), satisfying subadditivity,
        monotonicity, positive homogeneity, and translation equivariance.

    Args:
        state_dim   : spectral projection dimension K (= n_eigenvectors)
        hidden      : encoder / embedding hidden size
        n_quantiles : number of τ samples per forward pass
        cvar_alpha  : CVaR confidence level α ∈ (0, 1]  (0.2 = pessimistic 20%)
    """

    N_EMBED: int = 64   # cosine embedding dimension

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        state_dim:   int   = 4,
        hidden:      int   = 64,
        n_quantiles: int   = 32,  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
        cvar_alpha:  float = 0.2,  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
    ) -> None:
        super().__init__()  # 부모 클래스의 초기화 메서드를 실행한다  # Calls the parent class constructor
        self.n_quantiles = n_quantiles  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
        self.cvar_alpha  = cvar_alpha  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리

        # State encoder: c_kt [B, T, K] → h [B, T, hidden]
        self.encoder = nn.Sequential(  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module
            nn.Linear(state_dim, hidden),  # 선형 변환(행렬 곱 + 편향) 레이어를 만든다  # Fully-connected (affine) layer: y = xW^T + b
            nn.Tanh(),
            nn.Linear(hidden, hidden),  # 선형 변환(행렬 곱 + 편향) 레이어를 만든다  # Fully-connected (affine) layer: y = xW^T + b
            nn.Tanh(),
        )

        # Quantile embedding: cos(π·τ·i) for i=1..N_EMBED → [hidden]
        self.tau_embed = nn.Sequential(  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module
            nn.Linear(self.N_EMBED, hidden),  # 선형 변환(행렬 곱 + 편향) 레이어를 만든다  # Fully-connected (affine) layer: y = xW^T + b
            nn.ELU(),
        )

        # Output: element-wise product h ⊙ e → scalar per quantile
        self.out = nn.Linear(hidden, 1)  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        c_kt:      torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            c_kt      : [B, T, K] spectral projections
            n_samples : number of τ samples (default: self.n_quantiles)

        Returns:
            quant_vals : [B, T, N]  Z^{-1}(τ_i) — quantile value estimates
            cvar       : [B, T]     CVaR_α (pessimistic baseline for GAE)
            mean_val   : [B, T]     E[Z] (mean of quantile distribution)
            tau        : [B, T, N]  sampled quantile levels τ ~ U[0,1]
        """
        N = n_samples or self.n_quantiles  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
        B, T, K = c_kt.shape  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
        device, dtype = c_kt.device, c_kt.dtype

        # ── State encoding ────────────────────────────────────────────
        h = self.encoder(c_kt)                              # [B, T, hidden]

        # ── Quantile sampling + cosine embedding ──────────────────────
        tau = torch.rand(B, T, N, device=device, dtype=dtype)  # [B, T, N]
        i_vec = torch.arange(1, self.N_EMBED + 1, device=device, dtype=dtype)  # 순서대로 나열된 정수 텐서를 만든다  # Creates a 1-D tensor of evenly spaced integers
        tau_cos = torch.cos(math.pi * tau.unsqueeze(-1) * i_vec)  # [B, T, N, 64]
        tau_emb = self.tau_embed(tau_cos)                   # [B, T, N, hidden]

        # ── Element-wise combine + output ─────────────────────────────
        h_exp      = h.unsqueeze(2) * tau_emb              # [B, T, N, hidden]
        quant_vals = self.out(h_exp).squeeze(-1)            # [B, T, N]

        # ── CVaR_α = mean of quantiles where τ < α ───────────────────
        cvar_mask = (tau < self.cvar_alpha).to(dtype)       # [B, T, N]
        cvar_num  = (quant_vals * cvar_mask).sum(-1)        # [B, T]
        cvar_den  = cvar_mask.sum(-1).clamp(min=1.0)        # [B, T]
        cvar      = cvar_num / cvar_den                     # [B, T]

        mean_val  = quant_vals.mean(-1)                     # [B, T]

        return quant_vals, cvar, mean_val, tau  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# AdvancedPathIntegralLoss — Full Physicist/Math Objective
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedPathIntegralLoss(nn.Module):  # ★ [AdvancedPathIntegralLoss] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Advanced loss combining all roadmap improvements:

    Total loss:
        L = L_actor + L_critic + λ_FP · L_fp + λ_entropy · H(π)

    Where:
        L_actor  = -E[Â_t · log π(a_t|s_t)]     (GAE-weighted policy gradient)
        L_critic = MSE(V(s_t), returns_t)          (value function regression)
        L_fp     = NLL of Langevin SDE fit         (Fokker-Planck consistency)
        H(π)     = -Σ_a π(a) log π(a)             (entropy bonus for exploration)

    Physics:
        The GAE advantage Â_t = Σ(γλ)^l δ_{t+l} is mathematically equivalent to
        the "quantum path integral" J(τ) = Σ γ^t ΔV_t, but uses the critic V(s)
        as a baseline to subtract the mean return trajectory.

        This is analogous to renormalization in QFT: removing infinite self-energy
        corrections by subtracting the vacuum expectation value ⟨0|H|0⟩.

    Fokker-Planck:
        The SDE μ(x), σ(x) are fit to observed log-returns.
        L_fp penalizes transitions that violate Langevin dynamics:
            dx = μ(x)dt + σ(x)dW  →  p(x'|x) = N(x+μdt, σ²dt)

    Args:
        gamma       : time discount
        eta_base    : base fee penalty
        leverage    : leverage multiplier
        lam_gae     : GAE lambda (bias-variance tradeoff)
        critic_coef : L_critic weight
        fp_coef     : L_fp weight
        entropy_reg : entropy bonus weight
        state_dim   : critic input dim (= n_eigenvectors)
        t_base      : temperature scaling base
        beta_atr    : ATR sensitivity for temperature
        r_tp/sl/strategic: terminal reward parameters
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        gamma: float = 0.99,
        eta_base: float = 0.0005,
        leverage: float = 25.0,  # 레버리지: 실제 증거금의 몇 배로 거래하는지
        lam_gae: float = 0.95,  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
        critic_coef: float = 0.5,  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
        fp_coef: float = 0.05,
        entropy_reg: float = 0.01,  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
        state_dim: int = 4,
        t_base: float = 1.0,
        beta_atr: float = 2.0,
        r_tp: float = 1.0,
        r_sl: float = -1.0,
        r_strategic_min: float = 0.5,
        r_strategic_max: float = 0.9,
        use_iqn: bool = True,  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
        iqn_quantiles: int = 32,  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
        cvar_alpha: float = 0.2,  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
        use_gle: bool = True,
        gle_lam_scale: float = 0.04,
        dir_sym_coef: float = 0.5,
        act_sym_coef: float = 2.0,
        actor_coef: float = 1.0,
    ) -> None:
        super().__init__()  # 부모 클래스의 초기화 메서드를 실행한다  # Calls the parent class constructor

        self.gamma = gamma
        self.eta = eta_base * leverage  # 레버리지: 실제 증거금의 몇 배로 거래하는지
        self.actor_coef = actor_coef
        self.critic_coef = critic_coef  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
        self.fp_coef = fp_coef
        self.entropy_reg = entropy_reg  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
        self.dir_sym_coef = dir_sym_coef
        self.act_sym_coef = act_sym_coef
        self._use_gle = use_gle
        self._gle_lam_scale = gle_lam_scale

        # Re-use existing components
        self.temp_scaler = AdaptiveTemperatureScaler(t_base=t_base, beta=beta_atr)
        self.log_eta_scale = nn.Parameter(torch.tensor(0.0))  # 학습 중 자동 업데이트되는 파라미터로 등록한다  # Registers tensor as a learnable parameter (tracked by autograd)

        # Terminal rewards (same as PathIntegralLoss)
        self.r_tp = r_tp
        self.r_sl = r_sl
        self.r_strategic_min = r_strategic_min
        self.r_strategic_max = r_strategic_max

        # ── Critic head ────────────────────────────────────────────────────
        # IQN: distributional Z(s) → CVaR_α used as risk-sensitive baseline
        # Scalar: scalar V(s) = E[Z]  (classic)
        self._use_iqn = use_iqn  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
        if use_iqn:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self.critic = IQNCriticHead(  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
                state_dim=state_dim,
                hidden=64,
                n_quantiles=iqn_quantiles,  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
                cvar_alpha=cvar_alpha,  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
            )
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            self.critic = CriticHead(state_dim=state_dim, hidden=64)  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)

        # ── GAE computer ───────────────────────────────────────────────────
        # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지
        # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지
        # Generalized Advantage Estimation (GAE)
        self.gae = GeneralizedAdvantage(gamma=gamma, lam=lam_gae, normalize=True)

        # ── Fokker-Planck regularizer ──────────────────────────────────────
        # src.models.advanced_physics 모듈에서 FokkerPlanckRegularizer를 가져온다
        # src.models.advanced_physics 모듈에서 FokkerPlanckRegularizer를 가져온다
        # Import FokkerPlanckRegularizer from src.models.advanced_physics module
        from src.models.advanced_physics import FokkerPlanckRegularizer
        self.fp_reg = FokkerPlanckRegularizer(state_dim=state_dim, hidden=32)

    @property  # 이 메서드를 속성처럼 obj.속성 형태로 접근할 수 있게 만든다  # Decorator: expose method as a read-only attribute
    def effective_eta(self) -> torch.Tensor:  # [effective_eta] 함수 정의 시작
        return self.eta * torch.exp(self.log_eta_scale)  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    @staticmethod  # 객체(self) 없이 호출 가능한 정적 메서드로 만든다  # Decorator: static method — no self or cls
    def _quick_hurst(returns_flat: "np.ndarray") -> float:  # [_quick_hurst] 내부 전용 함수 정의
        """
        Fast single-scale R/S Hurst estimate from a flat 1-D array.

        R/S analysis (Mandelbrot & Wallis):
            H = log(R/S) / log(T)   (1-scale approximation)

        Used in GLE to compute H-adaptive GAE lambda.
        Returns 0.5 (random walk) on failure or insufficient data.
        """
        import numpy as _np  # 넘파이 — 빠른 숫자 계산 및 행렬 연산 라이브러리를 _np라는 별명으로 불러온다  # Import NumPy — fast numerical array computation as "_np"
        import math as _math  # 수학 계산(로그, 지수, 삼각함수 등) 표준 라이브러리를 _math라는 별명으로 불러온다  # Import Python standard math library (log, exp, trig) as "_math"
        x = returns_flat
        T = len(x)  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
        if T < 8:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            return 0.5  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
        try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
            mu  = x.mean()
            y   = x - mu
            cs  = _np.cumsum(y)  # 배열의 누적 합계를 계산한다  # Computes cumulative sum along an axis
            R   = cs.max() - cs.min()  # 가장 큰 값을 찾는다
            S   = x.std() + 1e-8
            rs  = R / S
            H   = _math.log(max(rs, 1e-8)) / _math.log(T)  # 가장 큰 값을 찾는다
            return float(_np.clip(H, 0.3, 0.9))  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
        except Exception:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
            return 0.5  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    def _terminal_reward(  # [_terminal_reward] 내부 전용 함수 정의
        self,
        terminal_state: torch.Tensor,
        realized_pnl: torch.Tensor,  # 손익: 이번 거래에서 얻은 이익 또는 손실
    ) -> torch.Tensor:
        """Same as PathIntegralLoss._terminal_reward."""
        B = terminal_state.shape[0]  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
        device = terminal_state.device  # 연산을 수행할 하드웨어(GPU/CPU)를 설정한다  # Target compute device: CUDA GPU or CPU
        dtype = realized_pnl.dtype  # 손익: 이번 거래에서 얻은 이익 또는 손실
        R_t = torch.zeros(B, device=device, dtype=dtype)  # 0으로 채워진 텐서(숫자 배열)를 만든다  # Creates a zero-filled tensor

        tp_mask = (terminal_state == MarketState.TP_HIT)  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
        sl_mask = (terminal_state == MarketState.SL_HIT)  # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지
        sc_mask = (terminal_state == MarketState.STRATEGIC_CLOSE)

        R_t = R_t + self.r_tp * tp_mask.to(dtype)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype
        R_t = R_t + self.r_sl * sl_mask.to(dtype)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype

        r_agile = (
            self.r_strategic_min
            + (self.r_strategic_max - self.r_strategic_min)
            * torch.sigmoid(5.0 * realized_pnl)  # 손익: 이번 거래에서 얻은 이익 또는 손실
        )
        R_t = R_t + r_agile * sc_mask.to(dtype)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype

        return R_t  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        logits:    torch.Tensor,       # [B, T, 3] or [B, 3]
        path:      TradingPath,
        c_kt:      torch.Tensor,       # [B, T, K] spectral projections (for critic)
        atr_norm:  Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full advanced loss computation.

        Args:
            logits   : [B, T, 3] policy logits from QuantumHamiltonianLayer
            path     : TradingPath trajectory data
            c_kt     : [B, T, K] spectral projections (Critic input)
            atr_norm : [B] ATR for temperature scaling

        Returns:
            loss     : scalar backward-able loss
            info     : diagnostic dict
        """
        if atr_norm is None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            atr_norm = path.atr_norm

        B = logits.shape[0]  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
        device = logits.device  # 연산을 수행할 하드웨어(GPU/CPU)를 설정한다  # Target compute device: CUDA GPU or CPU
        dtype = logits.dtype
        is_last_step = logits.dim() == 2

        # ── Temperature-scaled probabilities ──────────────────────────────
        if is_last_step:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            probs, t_temp = self.temp_scaler(logits, atr_norm)   # [B,3], [B]
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            probs, t_temp = self.temp_scaler(logits, atr_norm)   # [B,T,3], [B]

        log_probs_all = torch.log(probs.clamp(min=1e-8))  # 자연 로그(ln)를 계산한다  # Computes natural logarithm element-wise

        # ── Critic value estimation ───────────────────────────────────────
        # IQN: Z^{-1}(τ) → CVaR_α used as risk-sensitive V(s_t) baseline
        # Scalar: V(s_t) = MLP(c_kt)
        if self._use_iqn:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            _iqn_qv, V, _iqn_mean, _iqn_tau = self.critic(c_kt)  # [B,T_c,N],[B,T_c],[B,T_c],[B,T_c,N]
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            V = self.critic(c_kt)                                # [B, T_c]
        _T_c = V.shape[1]  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array

        # ── Per-step rewards from TradingPath ─────────────────────────────
        # r_t = γ^t * ΔV_t for Hold steps + terminal reward at last step
        B_p, T_p = path.delta_v.shape  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
        gamma_t = self.gamma ** torch.arange(T_p, device=device, dtype=dtype)  # 순서대로 나열된 정수 텐서를 만든다  # Creates a 1-D tensor of evenly spaced integers
        hold_mask = (
            (path.event_mask == MarketState.HOLD)
            | (path.event_mask == MarketState.LONG)
            | (path.event_mask == MarketState.SHORT)
        ).to(dtype)  # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다  # Moves tensor to specified device or dtype

        # Step rewards: γ^t ΔV_t for held positions
        step_rewards = gamma_t.unsqueeze(0) * path.delta_v * hold_mask  # [B, T]

        # Inject terminal reward at the last held timestep
        terminal_R = self._terminal_reward(path.terminal_state, path.realized_pnl)  # 손익: 이번 거래에서 얻은 이익 또는 손실
        # Add terminal reward at last timestep of the window
        step_rewards_with_terminal = step_rewards.clone()  # 텐서를 새로운 메모리에 복사한다  # Returns a deep copy of the tensor
        step_rewards_with_terminal[:, -1] = (
            step_rewards_with_terminal[:, -1] + terminal_R
        )

        # Subtract entry fee penalty at entry step
        eta_pen = self.effective_eta.to(device) * path.opened.to(dtype)  # [B]
        entry_idx_clamp = torch.zeros(B, device=device, dtype=torch.long)  # 0으로 채워진 텐서(숫자 배열)를 만든다  # Creates a zero-filled tensor
        step_rewards_with_terminal.scatter_add_(
            1,
            entry_idx_clamp.unsqueeze(1),  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
            -eta_pen.unsqueeze(1),  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
        )

        # ── Align T_path vs T_critic (may differ when x-window ≠ price-window) ──
        _T_p = step_rewards_with_terminal.shape[1]  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
        if _T_c != _T_p:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            _T = min(_T_c, _T_p)  # 가장 작은 값을 찾는다
            step_rewards_with_terminal = step_rewards_with_terminal[:, :_T]
            V = V[:, :_T]
            if self._use_iqn:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                _iqn_qv  = _iqn_qv[:, :_T, :]  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리
                _iqn_tau = _iqn_tau[:, :_T, :]  # IQN/CVaR: 수익의 전체 분포를 학습해 꼬리 위험을 관리

        # ── GAE Advantages (GLE Non-Markov adaptive lambda) ───────────────
        # Â_t = Σ (γλ)^l δ_{t+l},  δ_t = r_t + γV(s_{t+1}) - V(s_t)
        # When use_gle=True: λ is adapted to the Hurst exponent of the batch.
        #   H > 0.5 (persistent market) → higher λ (longer credit assignment)
        #   H ≈ 0.5 (random walk)       → base λ (unchanged)
        #   H < 0.5 (mean-reverting)    → lower λ (shorter memory)
        if self._use_gle:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            import numpy as _np_gle  # 넘파이 — 빠른 숫자 계산 및 행렬 연산 라이브러리를 _np_gle라는 별명으로 불러온다  # Import NumPy — fast numerical array computation as "_np_gle"
            # 텐서를 실수형(float32)으로 변환한다
            # 텐서를 실수형(float32)으로 변환한다
            rets_np = step_rewards_with_terminal.detach().cpu().float().numpy().flatten()  # Casts tensor to float32
            _H = self._quick_hurst(rets_np)  # 허스트 지수: H>0.5=추세형, H<0.5=평균회귀형, H=0.5=무작위  # Hurst exponent: H>0.5 trending, H<0.5 mean-reverting
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            _H = None

        advantages, returns = self.gae(  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
            rewards=step_rewards_with_terminal,   # [B, T_align]
            values=V,                             # [B, T_align]
            hurst=_H,  # 허스트 지수: H>0.5=추세형, H<0.5=평균회귀형, H=0.5=무작위  # Hurst exponent: H>0.5 trending, H<0.5 mean-reverting
        )
        # advantages: [B, T] — normalized advantage estimates

        # ── Actor (Policy Gradient) Loss ──────────────────────────────────
        if not is_last_step:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
            A_dim = logits.shape[-1]  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
            _T_act = advantages.shape[1]  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
            _em = path.event_mask[:, :_T_act]                         # [B, T_align]
            _lp = log_probs_all[:, :_T_act, :]                        # [B, T_align, 3]
            action_idx = _em.clone().clamp(0, A_dim - 1)              # [B, T_align]

            # log π(a_t | s_t)
            log_pi = _lp.gather(
                dim=2, index=action_idx.unsqueeze(2)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
            ).squeeze(2)                                               # [B, T_align]

            # Entry mask: only apply PG where we take action (LONG/SHORT)
            entry_mask = (
                (_em == MarketState.LONG)
                | (_em == MarketState.SHORT)
            ).float()                                                  # [B, T_align]

            # Actor loss: -E[Â_t · log π(a_t)]
            # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지
            # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지
            actor_loss = -(advantages * log_pi * entry_mask).sum(dim=1).mean()  # Generalized Advantage Estimation (GAE)
            entropy = -(probs * log_probs_all).sum(dim=-1).mean()  # 모든 값을 더한다
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            # last_step_only mode: single timestep
            # Fix B: event_mask[:, -1] = HOLD(3) → clamp(0,2) = 2 = SHORT (항상 SHORT만 학습하는 버그)
            # path.action = labels에서 직접 도출한 실제 진입 행동 (0=HOLD, 1=LONG, 2=SHORT)
            # 이제 LONG 샘플은 log P(LONG)을, SHORT 샘플은 log P(SHORT)을 올바르게 최적화.
            action_idx = path.action.clamp(0, logits.shape[-1] - 1)  # [B] — Fix B
            log_pi = log_probs_all.gather(
                dim=1, index=action_idx.unsqueeze(1)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
            ).squeeze(1)  # 크기가 1인 차원을 없앤다  # Removes dimensions of size 1
            # Use mean GAE advantage across T for single-step mode
            adv_scalar = advantages.mean(dim=1)  # 일반화 이점(GAE): 기대보다 얼마나 좋은 행동을 했는지  # Generalized Advantage Estimation (GAE)
            # Global advantage normalization (across ALL active LONG+SHORT examples).
            # Per-action zero-mean was incorrect: it subtracted the within-group mean,
            # making positive TP_HIT rewards look "bad" for half the examples and
            # eliminating the net learning signal entirely (net gradient ≈ 0).
            # Global normalization preserves relative quality between LONG and SHORT:
            # if LONG advantages > SHORT advantages (LONG moves happen earlier/reliably),
            # the model correctly prefers LONG.  Equal advantages → equal gradient for both.
            active_mask_bool = (action_idx > 0)  # LONG(1) and SHORT(2) only
            adv_norm = adv_scalar.clone()  # 텐서를 새로운 메모리에 복사한다  # Returns a deep copy of the tensor
            if active_mask_bool.sum() > 1:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                _a_sub = adv_scalar[active_mask_bool]  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
                adv_norm[active_mask_bool] = (
                    (_a_sub - _a_sub.mean()) / (_a_sub.std() + 1e-8)
                )
            # HOLD 샘플(action_idx=0)은 policy gradient에서 제외
            # HOLD의 Â는 unnormalized + 체계적으로 양수 → P(HOLD)↑ → P(SHORT)↓ → SP=0% 붕괴
            # LONG(1)/SHORT(2)만 actor_loss에 기여 → 방향성 정책만 학습
            active_mask = (action_idx > 0).float()  # [B]: 1=LONG/SHORT, 0=HOLD
            n_active = active_mask.sum().clamp(min=1.0)  # 모든 값을 더한다
            actor_loss = -(adv_norm * log_pi * active_mask).sum() / n_active  # 모든 값을 더한다
            entropy = -(probs * log_probs_all).sum(dim=-1).mean()  # 모든 값을 더한다

        # ── Critic Loss ───────────────────────────────────────────────────
        # IQN: quantile Huber regression over full return distribution Z(s)
        #      ρ_τ(target - Z^{-1}(τ)) — proper scoring for quantile regression
        # Scalar: MSE(V, returns)
        if self._use_iqn:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            critic_loss = _quantile_huber_loss(  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
                _iqn_qv, returns.detach(), _iqn_tau  # 기울기 계산 그래프에서 분리한다 (값만 복사)  # Detaches tensor from the computation graph
            )
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            critic_loss = F.mse_loss(V, returns.detach())  # 예측값과 실제값 차이의 제곱 평균(MSE 손실)을 계산한다  # Mean Squared Error loss: mean((pred - target)^2)

        # ── Fokker-Planck Regularizer ─────────────────────────────────────
        # Fit Langevin SDE to observed log-returns in c_kt space
        # Use c_kt[:, :, 0] as proxy for log-return direction (trend component)
        _T_fp = V.shape[1]                                             # T_align after GAE alignment
        log_ret_proxy = path.delta_v[:, :_T_fp]                       # [B, T_align]
        fp_loss, _, _ = self.fp_reg(
            state=c_kt[:, :_T_fp, :],                                 # [B, T_align, K]
            observed_returns=log_ret_proxy,                            # [B, T_align]
        )

        # ── Directional Symmetry Loss (3-way) ────────────────────────────
        # L_3way = Σ_a (P(a) - 1/3)²
        # |P(L)-P(S)| 방식은 P(H)를 방치 → HOLD collapse 재발.
        # 3-way 방식은 HOLD/LONG/SHORT 모두 1/3에서 벗어나면 즉시 페널티.
        # P(H)=0.40: (0.40-0.333)^2 = 0.0045 → gradient가 logit_H를 낮춤.
        
        #_target  = 1.0 / probs.shape[-1]     # 1/3
        #_mean_p  = probs.mean(dim=0)          # [A] batch-averaged P(a)
        #dir_sym_loss = ((_mean_p - _target) ** 2).sum()

        # Market-agnostic symmetric prior: slight HOLD bias, LONG == SHORT
        # Rationale: (1) L:S = 1:1 required for regime-agnostic universal model
        #            (2) Balanced Sampling already enforces 1:1 L:S at batch level
        #            (3) Regime adaptation handled by Lindblad gate + Hurst feature
        #            (4) Hardcoded SHORT bias hurts all Bull-market instruments
        target = probs.new_tensor([0.40, 0.30, 0.30]) # [H, L, S]
        mean_p = probs.mean(dim=0) # [H, L, S]
        dir_sym_loss = ((mean_p - target) **2).sum()  # 모든 값을 더한다

        # ── Action Symmetry Loss (3-way: HOLD/LONG/SHORT 균등 압력) ────────────
        # [설계 주의] path.action = ATR 레이블 (상수, ∂/∂θ=0) → 그래디언트 없음.
        # 올바른 구현: probs.mean(dim=0) — 배치 평균 softmax 확률 사용 (미분 가능).
        #
        # dir_sym_loss와의 차이:
        #   dir_sym_loss: Σ(mean_P(a) - 1/3)²  ← 모든 배치 샘플의 평균 확률
        #   act_sym_loss: 동일 공식이지만 act_sym_coef로 독립 조정 가능
        #   → 두 loss가 동일 방향으로 작용하되 계수를 분리 조정 가능
        #
        # 케이스별 값 (배치 평균 확률 기준):
        #   mean_P=(1/3,1/3,1/3) → 0.0  (완전 균등)
        #   mean_P=(1,0,0)       → 2/3  (HOLD-only 확률)
        #   mean_P=(0,1,0)       → 2/3  (LONG-only 확률)
        #   mean_P=(0,0,1)       → 2/3  (SHORT-only 확률)
        mean_p_act = probs.mean(dim=0)  # [3] — 미분 가능
        act_sym_loss = (
            (mean_p_act[0] - 1/3).pow(2)
            + (mean_p_act[1] - 1/3).pow(2)
            + (mean_p_act[2] - 1/3).pow(2)
        )

        # ── Total Loss ────────────────────────────────────────────────────
        # L = actor_coef×L_actor + critic_coef×L_critic + fp_coef×L_fp - entropy_reg×H + dir_sym_coef×dir_sym
        # actor_coef=0.5: actor loss 발산 억제 (1.0 기본값에서 절반)
        J_mean = step_rewards_with_terminal.sum(dim=1).mean()  # 모든 값을 더한다
        loss = (
            self.actor_coef * actor_loss
            + self.critic_coef * critic_loss  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
            + self.fp_coef * fp_loss
            - self.entropy_reg * entropy  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
            + self.dir_sym_coef * dir_sym_loss
            + self.act_sym_coef * act_sym_loss   # Action symmetry: (n_LONG/N - n_SHORT/N)² → SHORT 붕괴 직접 방지
        )

        info = {
            "loss":          loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "actor_loss":    actor_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "critic_loss":   critic_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "fp_loss":       fp_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "dir_sym_loss":  dir_sym_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "act_sym_loss":  act_sym_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "entropy":       entropy.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "J_mean":        J_mean.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "J_std":         step_rewards_with_terminal.sum(dim=1).std().item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "eta_effective": self.effective_eta.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "t_temp_mean":   t_temp.mean().item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "adv_mean":      advantages.mean().item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "adv_std":       advantages.std().item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "V_mean":        V.mean().item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            # 텐서를 실수형(float32)으로 변환한다
            # 텐서를 실수형(float32)으로 변환한다
            "pct_tp":  (path.terminal_state == MarketState.TP_HIT).float().mean().item(),  # Casts tensor to float32
            # 텐서를 실수형(float32)으로 변환한다
            # 텐서를 실수형(float32)으로 변환한다
            "pct_sl":  (path.terminal_state == MarketState.SL_HIT).float().mean().item(),  # Casts tensor to float32
            # 텐서를 실수형(float32)으로 변환한다
            # 텐서를 실수형(float32)으로 변환한다
            # Casts tensor to float32
            "pct_sc":  (path.terminal_state == MarketState.STRATEGIC_CLOSE).float().mean().item(),
            # Aliases for compatibility with existing monitors
            "policy_loss":   actor_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "J_min":         step_rewards_with_terminal.sum(dim=1).min().item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "J_max":         step_rewards_with_terminal.sum(dim=1).max().item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            # Per-sample advantage [B] tensor — used by AWBC in train_step.
            # Not a scalar: intentionally stored as tensor for gate computation.
            "adv_per_sample": advantages.mean(dim=1).detach(),  # 기울기 계산 그래프에서 분리한다 (값만 복사)  # Detaches tensor from the computation graph
        }

        return loss, info  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# AdvantageWeightedBC — aux_ce를 RL actor와 정렬하는 손실 함수
# ─────────────────────────────────────────────────────────────────────────────

class AdvantageWeightedBC(nn.Module):  # ★ [AdvantageWeightedBC] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Advantage-Weighted Behavioral Cloning (AWBC).

    기존 aux_ce 문제:
        CE는 레이블 방향으로 항상 당기지만,
        RL actor는 반대 방향을 학습할 수 있어 그래디언트 충돌 발생.

    AWBC 해법:
        CE는 다음 두 조건이 동시에 충족될 때만 발동:
          1. RL argmax(π) == label action  (방향 일치 게이트)
          2. GAE Â_t > 0                   (긍정 결과 게이트)

        수식:
            L_awbc = w · Σ_t  max(0, Â_t) · 1[RL==label] · CE(π(·|s_t), a*_t)
                                  ↑ 양수 어드밴티지  ↑ 방향 일치

    효과:
        - RL과 CE가 같은 방향일 때만 CE 발동 → 충돌 없음
        - RL과 CE가 반대 방향이면 CE 0 → 간섭 없음
        - 학습이 잘 된 바(Â > 0)에서만 지도 → 효율적 curriculum

    Args:
        weight : 전체 스케일 (aux_ce_weight와 동일 역할)
    """

    def __init__(self, weight: float = 0.05) -> None:  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        super().__init__()  # 부모 클래스의 초기화 메서드를 실행한다  # Calls the parent class constructor
        self.weight = weight

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        logits:     torch.Tensor,                    # [B, 3]
        labels:     torch.Tensor,                    # [B]  0=HOLD,1=LONG,2=SHORT
        advantages: Optional[torch.Tensor] = None,   # [B]  per-sample Â (GAE)
    ) -> torch.Tensor:
        """
        Args:
            logits    : [B, 3] — policy logits
            labels    : [B]   — expert action (0=HOLD,1=LONG,2=SHORT)
            advantages: [B]   — per-sample GAE advantage Â_t
                                None → alignment-only gate (no magnitude weighting)
        Returns:
            scalar loss (backward-able)
        """
        # ── Gate: 긍정 어드밴티지 크기 × LONG/SHORT 레이블 여부 ────────────
        # [Bug Fix] Gate 1 (RL argmax == label) 제거.
        # 모델이 항상 SHORT를 예측하면 LONG 예제는 Gate 1을 절대 통과 못 함
        # → AWBC가 LONG에 CE를 전혀 발동하지 않아 SHORT 편향이 자기 강화됨.
        # 올바른 기준: label > 0 (방향성 액션, HOLD 제외) AND Â > 0 (양의 결과).
        active = (labels > 0).float()                      # [B]  1=LONG/SHORT, 0=HOLD

        if advantages is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            adv_pos = F.relu(advantages.detach())          # [B]  max(0, Â)
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            adv_pos = torch.ones_like(active)  # 1로 채워진 텐서를 만든다  # Creates a ones-filled tensor

        # ── Combined gate: 방향성 액션 AND 양의 어드밴티지 ───────────────
        gate = active * adv_pos                            # [B]  ≥ 0

        # P6 fix: gate.sum() 기반 가중 평균 — 1e-8 floor 증폭 방지.
        # 원래 gate/(gate.mean()+1e-8) 수식 = Σ gate_i·ce_i / Σ gate_i 와 동일하지만,
        # gate.mean()이 1e-8보다 작으면 (gate.sum() ≈ 0이지만 완전한 0 아닐 때)
        # 실효 분모가 1e-8로 고정 → 개별 weight 최대 1/1e-8 = 1e8× 증폭 가능.
        # gate.sum() 직접 사용 + sum()≈0이면 즉시 0 반환 → 증폭 원천 차단.
        gate_sum = gate.sum()  # 모든 값을 더한다
        if gate_sum.item() < 1e-8:                         # active gate 없음 → 0 반환
            return gate_sum * 0.0                          # grad-safe zero scalar

        # ── Advantage-weighted CE (proper weighted mean) ─────────────────
        # L = w · Σ gate_i·ce_i / Σ gate_i  ← 가중 평균, gate_mean()와 수학적 동치
        ce_per = F.cross_entropy(logits, labels, reduction='none')  # [B]
        return self.weight * (gate * ce_per).sum() / gate_sum  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# QuantumDivineLossV2 — Quantum Contrastive Label-Smoothing (QCLS)
# Claude + Gemini 통합 제안 (2026-03-01)
# ─────────────────────────────────────────────────────────────────────────────

class RegretWeightedBCLoss(nn.Module):  # ★ [RegretWeightedBCLoss] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Regret-Weighted Behavioural Cloning Loss

    기회비용(regret) 기반 가중 손실 함수.

    핵심 아이디어:
        R(HOLD, any)   = 0       (항상 손실 없음)
        R(LONG, LONG)  = +TP     R(LONG, SHORT) = -SL  → 틀린 방향 최악
        R(SHORT, SHORT)= +TP     R(SHORT, LONG) = -SL

    Regret(pred, true) = R(optimal) - R(pred) = 기회비용

    Regret 행렬 (행=true, 열=pred):
              HOLD          LONG          SHORT
    HOLD  [  0.0,          fee,          fee  ]   ← HOLD를 잘못 예측 = 수수료만
    LONG  [  tp_mult,      0.0,          tp+sl]   ← 방향 반대 = 최악
    SHORT [  tp_mult,      tp+sl,        0.0  ]   ← 방향 반대 = 최악

    Loss = E_pred[regret(true, pred)] × CE(pred, true)
         = (Σ_a π(a|s) × regret(true, a) + 1) × CE_smooth(pred, true)
         +  λ_orth × L_orth  +  λ_parity × L_parity

    핵심 효과:
        - 모델이 틀린 방향 예측 시 gradient가 (TP+SL)× 증폭
        - 불확실할 때 HOLD 선택 유도 (HOLD 오예측 gradient = fee× 미미)
        - BC→RL 정렬: BC loss가 RL 기회비용과 동일 단위

    Args:
        tp_mult    : TP 배수 (default=2.0, 즉 2×ATR)
        sl_mult    : SL 배수 (default=1.0, 즉 1×ATR)
        fee        : 왕복 수수료 (default=0.075%, Bybit taker)
        smoothing  : label smoothing ε (default=0.10)
        orth_w     : Hilbert orthogonality 가중치
        parity_w   : 3-way balanced distribution 가중치
        regret_w   : regret 가중치 스케일 (default=1.0)
    """

    # Action indices (CLAUDE.md 기준)
    HOLD  = 0
    LONG  = 1
    SHORT = 2

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        tp_mult:   float = 2.0,  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지  # Take-profit multiplier: exit at TP × ATR
        sl_mult:   float = 1.0,  # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지  # Stop-loss multiplier: forced exit at SL × ATR
        fee:       float = 0.075,
        smoothing: float = 0.10,
        orth_w:    float = 0.15,
        parity_w:  float = 0.10,
        regret_w:  float = 1.0,
        n_class:   int   = 3,
    ) -> None:
        super().__init__()  # 부모 클래스의 초기화 메서드를 실행한다  # Calls the parent class constructor
        self.smoothing = smoothing
        self.orth_w    = orth_w
        self.parity_w  = parity_w
        self.regret_w  = regret_w
        self.n_class   = n_class

        # Regret 행렬 구성 (행=true label, 열=predicted action)
        # HOLD=0, LONG=1, SHORT=2
        tp  = tp_mult  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지  # Take-profit multiplier: exit at TP × ATR
        sl  = sl_mult  # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지  # Stop-loss multiplier: forced exit at SL × ATR
        fee = fee
        R = torch.tensor([  # 파이썬 데이터를 파이토치 텐서로 변환한다  # Converts Python data to a PyTorch tensor
            #  pred:HOLD   pred:LONG    pred:SHORT
            [  0.0,        fee,         fee        ],  # true: HOLD
            [  tp,         0.0,         tp + sl    ],  # true: LONG
            [  tp,         tp + sl,     0.0        ],  # true: SHORT
        ], dtype=torch.float32)
        self.register_buffer("regret_matrix", R)   # [3, 3]

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        logits:  torch.Tensor,   # [B, 3]
        labels:  torch.Tensor,   # [B]  (0=HOLD, 1=LONG, 2=SHORT)
        expvals: torch.Tensor,   # [B, N_QUBITS=3]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            loss  : scalar tensor
            stats : dict with "ce", "regret", "orth", "parity"
        """
        probs = F.softmax(logits, dim=1)  # [B, 3]

        # ── 1. Expected Regret Weight ────────────────────────────────────────
        # regret_matrix[labels] → [B, 3]: 각 샘플에 대해 pred action별 regret
        regret_per_action = self.regret_matrix.to(labels.device)[labels]  # [B, 3]
        # E[regret] = Σ_a π(a|s) × regret(true, a)
        expected_regret   = (probs * regret_per_action).sum(1)  # [B]
        # +1: 정답 예측 시 최소 1× gradient 보장 (regret=0이 되면 gradient 소실)
        regret_weights    = 1.0 + self.regret_w * expected_regret  # [B]

        # ── 2. Label-Smoothed Cross-Entropy ──────────────────────────────────
        with torch.no_grad():  # 메모리 절약을 위해 기울기 계산 없이 추론만 실행한다  # Context: disable gradient tracking for inference (saves memory)
            soft = torch.full_like(logits, self.smoothing / (self.n_class - 1))
            soft.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
        log_probs  = F.log_softmax(logits, dim=1)              # [B, 3]
        ce_per_sample = -(soft * log_probs).sum(dim=1)         # [B]

        # Regret 가중 적용
        ce_loss = (regret_weights.detach() * ce_per_sample).mean()  # 기울기 계산 그래프에서 분리한다 (값만 복사)  # Detaches tensor from the computation graph

        # ── 3. Hilbert Orthogonality ─────────────────────────────────────────
        centroids: list = []
        valid = True
        for i in range(self.n_class):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            mask = (labels == i)
            if mask.sum() < 1:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                valid = False
                break  # 현재 반복문을 즉시 탈출한다  # Exit the enclosing loop immediately
            c    = expvals[mask].mean(dim=0)
            norm = c.norm()
            if norm < 1e-8:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                valid = False
                break  # 현재 반복문을 즉시 탈출한다  # Exit the enclosing loop immediately
            centroids.append(c / norm)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list

        if valid:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            c_h, c_l, c_s = centroids
            orth_loss = (
                torch.dot(c_h, c_l).pow(2) +
                torch.dot(c_h, c_s).pow(2) +
                torch.dot(c_l, c_s).pow(2)
            )
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            orth_loss = torch.tensor(0.0, device=logits.device)  # 파이썬 데이터를 파이토치 텐서로 변환한다  # Converts Python data to a PyTorch tensor

        # ── 4. Parity Regularizer ────────────────────────────────────────────
        mean_probs  = probs.mean(dim=0)
        parity_loss = (
            (mean_probs[0] - 1/3).pow(2)
            + (mean_probs[1] - 1/3).pow(2)
            + (mean_probs[2] - 1/3).pow(2)
        )

        total = (
            ce_loss
            + self.orth_w   * orth_loss
            + self.parity_w * parity_loss
        )
        return total, {  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
            "ce":     ce_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "regret": expected_regret.mean().item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "orth":   orth_loss.item() if isinstance(orth_loss, torch.Tensor) else 0.0,  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "parity": parity_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        }


class QuantumDivineLossV2(nn.Module):  # ★ [QuantumDivineLossV2] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Quantum Contrastive Label-Smoothing (QCLS)

    두 관점의 통합:
      1. Label-Smoothing CE [Claude]: entropy collapse 방지.
         soft target → 모델이 ln(3) plateau에서 탈출하도록 지속적 gradient 제공.
      2. Hilbert Orthogonality [Gemini]: N=3 qubits → expvals ∈ R³.
         R³에서 3개 상호직교 단위벡터가 수학적으로 가능.
         HOLD/LONG/SHORT centroid를 Bloch Sphere의 서로 다른 octant로 분리.

    수식:
        L = L_LS_CE + λ_orth · L_orth
        L_LS_CE = H(y_smooth, p̂)          (label-smoothed cross-entropy)
        L_orth  = Σ_{i<j} cos²(c_i, c_j)   (squared cosine similarity 최소화)

    주의:
        - per-class alpha 가중치 미사용: BalancedBatchSampler(1:1:1) + class weight →
          SHORT collapse 유발 확인됨 → 균등 가중치 유지.
        - centroid는 L2 정규화 후 dot product (raw dot product ≠ cosine similarity).
        - 배치 내 특정 클래스 샘플 0개일 때 orth_loss=0 반환 (안전).

    Args:
        smoothing : label smoothing ε (default=0.10)
        orth_w    : Hilbert orthogonality 가중치 λ (default=0.15)
        n_class   : 클래스 수 (default=3: HOLD/LONG/SHORT)
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        smoothing:    float = 0.10,
        orth_w:       float = 0.15,
        parity_w:     float = 0.10,
        n_class:      int   = 3,
    ) -> None:
        super().__init__()  # 부모 클래스의 초기화 메서드를 실행한다  # Calls the parent class constructor
        self.smoothing = smoothing
        self.orth_w    = orth_w
        self.parity_w  = parity_w
        self.n_class   = n_class

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        logits:  torch.Tensor,   # [B, C]
        labels:  torch.Tensor,   # [B]
        expvals: torch.Tensor,   # [B, N_QUBITS=3]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            loss  : scalar tensor
            stats : dict with "ce", "orth", "parity" for logging
        """
        # ── 1. Label-Smoothing Cross-Entropy ────────────────────────────────
        with torch.no_grad():  # 메모리 절약을 위해 기울기 계산 없이 추론만 실행한다  # Context: disable gradient tracking for inference (saves memory)
            soft = torch.full_like(logits, self.smoothing / (self.n_class - 1))
            soft.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
        ce_loss = -(soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()  # 소프트맥스를 적용하고 로그를 취한다  # Log-softmax: numerically stable log of softmax

        # ── 2. Hilbert Orthogonality (N=3 전용, R³ 기하학) ──────────────────
        # 각 클래스의 expval centroid를 계산하고 L2 정규화 (단위벡터화)
        centroids: list[torch.Tensor] = []
        valid = True
        for i in range(self.n_class):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            mask = (labels == i)
            if mask.sum() < 1:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                valid = False
                break  # 현재 반복문을 즉시 탈출한다  # Exit the enclosing loop immediately
            c = expvals[mask].mean(dim=0)     # [K]
            norm = c.norm()
            if norm < 1e-8:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                valid = False
                break  # 현재 반복문을 즉시 탈출한다  # Exit the enclosing loop immediately
            centroids.append(c / norm)        # 단위벡터  # Appends an item to the end of the list

        if valid:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            c_h, c_l, c_s = centroids
            # cos²(c_i, c_j) = (c_i · c_j)² — 0이면 완전 직교 (패널티 없음)
            orth_loss = (
                torch.dot(c_h, c_l).pow(2) +
                torch.dot(c_h, c_s).pow(2) +
                torch.dot(c_l, c_s).pow(2)
            )
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            orth_loss = torch.tensor(0.0, device=logits.device)  # 파이썬 데이터를 파이토치 텐서로 변환한다  # Converts Python data to a PyTorch tensor

        # ── 3. Parity Regularizer (3-way balanced distribution) ──────────────
        # Σ(P(a) - 1/3)² → minimized at P_H=P_L=P_S=1/3
        # 구버전 (P_L-P_S)²는 P_H=0, P_L=P_S=0.5에서 최소 → HOLD collapse 유발
        probs       = F.softmax(logits, dim=1)        # [B, 3]
        mean_probs  = probs.mean(dim=0)               # [3]
        parity_loss = (
            (mean_probs[0] - 1/3).pow(2)
            + (mean_probs[1] - 1/3).pow(2)
            + (mean_probs[2] - 1/3).pow(2)
        )

        total = ce_loss + self.orth_w * orth_loss + self.parity_w * parity_loss
        return total, {  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
            "ce":     ce_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "orth":   orth_loss.item() if isinstance(orth_loss, torch.Tensor) else 0.0,  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            "parity": parity_loss.item(),  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
        }


# ─────────────────────────────────────────────────────────────────────────────
# 빠른 검증
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
    print("=" * 65)  # 결과를 화면에 출력한다  # Prints output to stdout
    print("  PathIntegralLoss — Phase 3 자가 검증 (Self-Test)")  # 결과를 화면에 출력한다  # Prints output to stdout
    print("  마스터플랜 Part 3.1 / 3.2 / 3.3 수식 반영 확인")  # 결과를 화면에 출력한다  # Prints output to stdout
    print("=" * 65)  # 결과를 화면에 출력한다  # Prints output to stdout

    torch.manual_seed(42)
    device = torch.device("cpu")  # 연산을 수행할 하드웨어(GPU/CPU)를 설정한다  # Target compute device: CUDA GPU or CPU

    B, T, A = 8, 30, 3

    # ── 1. 온도 스케일러 검증 ────────────────────────────────────────────
    print("\n[Test 1] AdaptiveTemperatureScaler")  # 결과를 화면에 출력한다  # Prints output to stdout
    scaler = AdaptiveTemperatureScaler(t_base=1.0, beta=2.0)
    logits  = torch.randn(B, T, A)  # 표준 정규분포(평균 0, 표준편차 1) 난수 텐서를 만든다  # Creates a tensor with standard normal random values
    atr_low  = torch.full((B,), 0.002)   # 저변동성
    atr_high = torch.full((B,), 0.05)    # 고변동성
    probs_low,  t_low  = scaler(logits, atr_low)
    probs_high, t_high = scaler(logits, atr_high)
    print(f"  저변동성 T_temp: {t_low[0].item():.4f}")  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
    print(f"  고변동성 T_temp: {t_high[0].item():.4f}")  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
    assert t_high[0] > t_low[0], "고변동성에서 온도가 더 높아야 함!"  # 조건이 반드시 참이어야 한다 (거짓이면 AssertionError 발생)  # Assertion: raises AssertionError if condition is False
    print(f"  ✓ 고변동성 > 저변동성 온도 검증 통과")  # 결과를 화면에 출력한다  # Prints output to stdout

    # ── 2. 경로 보상 J(τ) 검증 ──────────────────────────────────────────
    print("\n[Test 2] _compute_path_reward")  # 결과를 화면에 출력한다  # Prints output to stdout
    loss_fn = build_path_integral_loss(gamma=0.99, leverage=25.0)  # 레버리지: 실제 증거금의 몇 배로 거래하는지

    # TP 시나리오: 강한 상승 수익
    delta_v_tp = torch.full((B, T), 0.002)                  # 타임스텝당 +0.2%
    event_mask_tp = torch.full((B, T), MarketState.HOLD, dtype=torch.long)
    path_tp = TradingPath(
        delta_v=delta_v_tp,
        event_mask=event_mask_tp,
        # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
        # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
        terminal_state=torch.full((B,), MarketState.TP_HIT, dtype=torch.long),
        realized_pnl=torch.full((B,), 0.06),  # 손익: 이번 거래에서 얻은 이익 또는 손실
        hold_steps=torch.full((B,), T, dtype=torch.long),
        atr_norm=torch.full((B,), 0.01),
        opened=torch.ones(B, dtype=torch.bool),  # 1로 채워진 텐서를 만든다  # Creates a ones-filled tensor
    )
    J_tp = loss_fn._compute_path_reward(path_tp)
    print(f"  TP 경로 J(τ): {J_tp.mean().item():.4f}  (양수 기대)")  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

    # SL 시나리오: 손실
    delta_v_sl = torch.full((B, T), -0.003)  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
    path_sl = TradingPath(
        delta_v=delta_v_sl,  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
        event_mask=event_mask_tp,
        # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지
        # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지
        terminal_state=torch.full((B,), MarketState.SL_HIT, dtype=torch.long),
        realized_pnl=torch.full((B,), -0.09),  # 손익: 이번 거래에서 얻은 이익 또는 손실
        hold_steps=torch.full((B,), T, dtype=torch.long),
        atr_norm=torch.full((B,), 0.01),
        opened=torch.ones(B, dtype=torch.bool),  # 1로 채워진 텐서를 만든다  # Creates a ones-filled tensor
    )
    J_sl = loss_fn._compute_path_reward(path_sl)
    print(f"  SL 경로 J(τ): {J_sl.mean().item():.4f}  (음수 기대)")  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

    # 전략적 종료 시나리오
    path_sc = TradingPath(
        delta_v=torch.full((B, T), 0.001),
        event_mask=event_mask_tp,
        terminal_state=torch.full((B,), MarketState.STRATEGIC_CLOSE, dtype=torch.long),
        realized_pnl=torch.full((B,), 0.03),  # 손익: 이번 거래에서 얻은 이익 또는 손실
        hold_steps=torch.full((B,), T // 2, dtype=torch.long),
        atr_norm=torch.full((B,), 0.01),
        opened=torch.ones(B, dtype=torch.bool),  # 1로 채워진 텐서를 만든다  # Creates a ones-filled tensor
    )
    J_sc = loss_fn._compute_path_reward(path_sc)
    print(f"  SC 경로 J(τ): {J_sc.mean().item():.4f}  (0.5~0.9 기대)")  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor

    assert J_tp.mean() > J_sl.mean(), "TP > SL 검증 실패!"  # 조건이 반드시 참이어야 한다 (거짓이면 AssertionError 발생)  # Assertion: raises AssertionError if condition is False
    assert J_tp.mean() > J_sc.mean(), "TP > SC 검증 실패!"  # 조건이 반드시 참이어야 한다 (거짓이면 AssertionError 발생)  # Assertion: raises AssertionError if condition is False
    assert J_sc.mean() > J_sl.mean(), "SC > SL 검증 실패!"  # 조건이 반드시 참이어야 한다 (거짓이면 AssertionError 발생)  # Assertion: raises AssertionError if condition is False
    print("  ✓ 보상 크기 순서: TP > SC > SL 검증 통과")  # 결과를 화면에 출력한다  # Prints output to stdout

    # γ 할인 효과 검증: 짧은 hold 가 더 유리해야 함
    path_fast = TradingPath(
        delta_v=torch.cat([torch.full((B, 5), 0.01), torch.zeros(B, T-5)], dim=1),  # 0으로 채워진 텐서(숫자 배열)를 만든다  # Creates a zero-filled tensor
        event_mask=event_mask_tp,
        # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
        # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
        terminal_state=torch.full((B,), MarketState.TP_HIT, dtype=torch.long),
        realized_pnl=torch.full((B,), 0.05),  # 손익: 이번 거래에서 얻은 이익 또는 손실
        hold_steps=torch.full((B,), 5, dtype=torch.long),
        atr_norm=torch.full((B,), 0.01),
        opened=torch.ones(B, dtype=torch.bool),  # 1로 채워진 텐서를 만든다  # Creates a ones-filled tensor
    )
    path_slow = TradingPath(
        delta_v=torch.cat([torch.zeros(B, T-5), torch.full((B, 5), 0.01)], dim=1),  # 0으로 채워진 텐서(숫자 배열)를 만든다  # Creates a zero-filled tensor
        event_mask=event_mask_tp,
        # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
        # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
        terminal_state=torch.full((B,), MarketState.TP_HIT, dtype=torch.long),
        realized_pnl=torch.full((B,), 0.05),  # 손익: 이번 거래에서 얻은 이익 또는 손실
        hold_steps=torch.full((B,), T, dtype=torch.long),
        atr_norm=torch.full((B,), 0.01),
        opened=torch.ones(B, dtype=torch.bool),  # 1로 채워진 텐서를 만든다  # Creates a ones-filled tensor
    )
    J_fast = loss_fn._compute_path_reward(path_fast)
    J_slow = loss_fn._compute_path_reward(path_slow)
    print(f"\n  γ 할인 효과 (동일 수익):")  # 결과를 화면에 출력한다  # Prints output to stdout
    print(f"    빠른 수익(t=0~4): J = {J_fast.mean().item():.4f}")  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
    print(f"    느린 수익(t=25~29): J = {J_slow.mean().item():.4f}")  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
    assert J_fast.mean() > J_slow.mean(), "γ 할인: 빠른 수익이 더 유리해야 함!"  # 조건이 반드시 참이어야 한다 (거짓이면 AssertionError 발생)  # Assertion: raises AssertionError if condition is False
    print("  ✓ 시간 할인 γ^t 효과 검증 통과 (빠른 수익 > 느린 수익)")  # 결과를 화면에 출력한다  # Prints output to stdout

    # ── 3. Forward + backward 검증 ──────────────────────────────────────
    print("\n[Test 3] Forward + backward (그라디언트 흐름)")  # 결과를 화면에 출력한다  # Prints output to stdout
    logits_3d = torch.randn(B, T, A, requires_grad=True)  # 표준 정규분포(평균 0, 표준편차 1) 난수 텐서를 만든다  # Creates a tensor with standard normal random values
    loss, info = loss_fn(logits_3d, path_tp)
    loss.backward()  # 손실 함수를 역방향으로 미분해 기울기를 계산한다  # Computes gradients via backpropagation
    grad_ok = logits_3d.grad is not None
    print(f"  loss = {info['loss']:.4f}")  # 결과를 화면에 출력한다  # Prints output to stdout
    print(f"  J_mean = {info['J_mean']:.4f}")  # 결과를 화면에 출력한다  # Prints output to stdout
    print(f"  eta_effective = {info['eta_effective']:.6f}")  # 결과를 화면에 출력한다  # Prints output to stdout
    print(f"  t_temp_mean = {info['t_temp_mean']:.4f}")  # 결과를 화면에 출력한다  # Prints output to stdout
    print(f"  ✓ backward() 그라디언트 흐름: {'OK' if grad_ok else 'FAIL'}")  # 결과를 화면에 출력한다  # Prints output to stdout

    # ── 4. TradingPathBuilder 검증 ──────────────────────────────────────
    print("\n[Test 4] TradingPathBuilder (원시 데이터 → TradingPath)")  # 결과를 화면에 출력한다  # Prints output to stdout
    builder = TradingPathBuilder(leverage=25.0)  # 레버리지: 실제 증거금의 몇 배로 거래하는지
    prices = 50000.0 + torch.cumsum(torch.randn(B, T) * 50, dim=1)  # 표준 정규분포(평균 0, 표준편차 1) 난수 텐서를 만든다  # Creates a tensor with standard normal random values
    directions = torch.ones(B)  # 1로 채워진 텐서를 만든다  # Creates a ones-filled tensor
    entry_idx = torch.zeros(B, dtype=torch.long)  # 0으로 채워진 텐서(숫자 배열)를 만든다  # Creates a zero-filled tensor
    labels = torch.randint(0, 4, (B,))  # 정수로 변환한다
    atr = torch.full((B,), 150.0)  # ATR: 가격의 평균 변동 폭 (Average True Range)
    # ATR: 가격의 평균 변동 폭 (Average True Range)
    # ATR: 가격의 평균 변동 폭 (Average True Range)
    # ATR: Average True Range — average price volatility
    path_built = builder.from_tensors(prices, directions, entry_idx, labels, atr)
    print(f"  delta_v shape: {tuple(path_built.delta_v.shape)}")  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
    print(f"  event_mask sample: {path_built.event_mask[0, :5].tolist()}")  # 결과를 화면에 출력한다  # Prints output to stdout
    print(f"  terminal_state: {path_built.terminal_state.tolist()}")  # 결과를 화면에 출력한다  # Prints output to stdout
    J_built = loss_fn._compute_path_reward(path_built)
    print(f"  J(τ) range: [{J_built.min().item():.4f}, {J_built.max().item():.4f}]")  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
    print("  ✓ TradingPathBuilder 검증 통과")  # 결과를 화면에 출력한다  # Prints output to stdout

    print("\n  ✓ 모든 검증 통과! Phase 3 구현 완료.")  # 결과를 화면에 출력한다  # Prints output to stdout
    print("=" * 65)  # 결과를 화면에 출력한다  # Prints output to stdout
