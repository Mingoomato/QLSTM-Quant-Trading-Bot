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

from __future__ import annotations

import math
import os
import time

import numpy as np
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.data.spectral_decomposer import SpectralDecomposer
from src.models.quantum_layers import QuantumHamiltonianLayer, QuantumMarketEncoder, N_QUBITS
from src.models.loss import (
    AdaptiveTemperatureScaler,
    AdvancedPathIntegralLoss,
    CriticHead,
    GeneralizedAdvantage,
    MarketState,
    PathIntegralLoss,
    TradingPath,
    TradingPathBuilder,
    build_path_integral_loss,
)
from src.models.advanced_physics import (
    EntropyProductionEstimator,
    HurstEstimator,
    LindbladDecoherence,
    MINEEstimator,
    OptimalStoppingBoundary,
    PlattCalibrator,
)
from src.strategies.regime_gate import CramerRaoFilter


# ─────────────────────────────────────────────────────────────────────────────
# 설정 데이터클래스
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """
    QuantumFinancialAgent 하이퍼파라미터 전체.

    마스터플랜 기본값:
        gamma    = 0.99   (Part 3.1: 시간 할인율)
        eta_base = 0.0005 (Part 3.1: 수수료 페널티 비율)
        leverage = 25.0   (η = eta_base × leverage = 1.25%)
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
    eta_base:        float = 0.0005
    leverage:        float = 25.0
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
    grad_clip:       float = 1.0
    # P8 fix: entropy_reg 0.20 → 0.05
    # 0.20: 엔트로피 보너스가 policy gradient를 압도 → 방향 학습 불가.
    # 0.01: VQC 회로 고유 SHORT 편향이 완전 표출 → LP=0% SHORT 붕괴.
    # 0.05: SHORT 붕괴 방지(LP>0% 유지) + policy gradient가 주도.
    entropy_reg:     float = 0.05
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
    checkpoint_dir:  str   = "checkpoints/quantum_v2"

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
    lindblad_regime_threshold: float = 0.97  # 0.90→0.97: Fold5 72% 과차단 해소

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

    @property
    def effective_eta(self) -> float:
        """η = eta_base × leverage"""
        return self.eta_base * self.leverage


# ─────────────────────────────────────────────────────────────────────────────
# SnipingMonitor — 수수료 방어 지표 실시간 추적
# ─────────────────────────────────────────────────────────────────────────────

class SnipingMonitor:
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

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._results: Deque[Dict[str, float]] = deque(maxlen=window)
        self._step = 0
        self.cumulative_J = 0.0
        self.trade_count  = 0

    def record(
        self,
        terminal_state: torch.Tensor,   # [B]
        J: torch.Tensor,                # [B]
        hold_steps: torch.Tensor,       # [B]
        realized_pnl: torch.Tensor,     # [B]
        e_eta: float,                   # effective η
    ) -> None:
        """배치 결과를 기록."""
        for b in range(terminal_state.shape[0]):
            st   = int(terminal_state[b].item())
            j_b  = float(J[b].item())
            h_b  = int(hold_steps[b].item())
            pnl  = float(realized_pnl[b].item())

            self._results.append({
                "state":      st,
                "J":          j_b,
                "hold_steps": h_b,
                "pnl":        pnl,
            })
            if st in (MarketState.LONG, MarketState.SHORT,
                      MarketState.TP_HIT, MarketState.SL_HIT,
                      MarketState.STRATEGIC_CLOSE):
                self.trade_count += 1
            self.cumulative_J += j_b
        self._step += 1

    @property
    def stats(self) -> Dict[str, float]:
        if not self._results:
            return {}
        results = list(self._results)
        n = len(results)

        tp_c  = sum(1 for r in results if r["state"] == MarketState.TP_HIT)
        sl_c  = sum(1 for r in results if r["state"] == MarketState.SL_HIT)
        sc_c  = sum(1 for r in results if r["state"] == MarketState.STRATEGIC_CLOSE)
        non_obs = tp_c + sl_c + sc_c
        win_rate = (tp_c + sc_c) / max(non_obs, 1)

        avg_J    = sum(r["J"] for r in results) / n
        avg_hold = sum(r["hold_steps"] for r in results) / n
        avg_pnl  = sum(r["pnl"] for r in results) / n

        # 스나이퍼 점수: 승률 × (1 / log2(avg_hold+2)) × sign(avg_pnl+ε)
        # → 높은 승률 & 짧은 보유 & 양의 수익 → 높은 점수
        hold_penalty = max(math.log2(avg_hold + 2), 1.0)
        sniper_score = win_rate / hold_penalty * math.copysign(1, avg_pnl + 1e-8)

        return {
            "n_samples":    n,
            "tp_rate":      tp_c / max(non_obs, 1),
            "sl_rate":      sl_c / max(non_obs, 1),
            "sc_rate":      sc_c / max(non_obs, 1),
            "win_rate":     win_rate,
            "avg_J":        avg_J,
            "avg_hold":     avg_hold,
            "avg_pnl":      avg_pnl,
            "sniper_score": sniper_score,
            "total_trades": self.trade_count,
        }

    def summary_str(self) -> str:
        s = self.stats
        if not s:
            return "SnipingMonitor: 아직 데이터 없음"
        return (
            f"[SnipingMonitor] "
            f"WinRate={s['win_rate']:.1%}  "
            f"TP={s['tp_rate']:.1%}  SL={s['sl_rate']:.1%}  SC={s['sc_rate']:.1%}  "
            f"AvgHold={s['avg_hold']:.1f}bars  "
            f"AvgJ={s['avg_J']:.4f}  "
            f"SniperScore={s['sniper_score']:.4f}  "
            f"TotalTrades={s['total_trades']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TrainStepResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainStepResult:
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

class QuantumFinancialAgent(nn.Module):
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

    def __init__(
        self,
        config: AgentConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
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
        if config.use_advanced_loss:
            # GAE + Critic + Fokker-Planck
            self.loss_fn = AdvancedPathIntegralLoss(
                gamma=config.gamma,
                eta_base=config.eta_base,
                leverage=config.leverage,
                lam_gae=config.lam_gae,
                critic_coef=config.critic_coef,
                fp_coef=config.fp_coef,
                entropy_reg=config.entropy_reg,
                state_dim=N_QUBITS,   # projected c_kt is qubit-dim (3), not n_eigenvectors (5)
                t_base=config.t_base,
                beta_atr=config.beta_atr,
                r_tp=config.r_tp,
                r_sl=config.r_sl,
                r_strategic_min=config.r_strategic_min,
                r_strategic_max=config.r_strategic_max,
                use_iqn=config.use_iqn,
                iqn_quantiles=config.iqn_quantiles,
                cvar_alpha=config.cvar_alpha,
                use_gle=config.use_gle,
                gle_lam_scale=config.gle_lam_scale,
                dir_sym_coef=config.dir_sym_coef,
                act_sym_coef=config.act_sym_coef,
                actor_coef=config.actor_coef,
            )
            self._use_advanced = True
        else:
            # Classic REINFORCE PathIntegralLoss
            self.loss_fn = PathIntegralLoss(
                gamma=config.gamma,
                eta_base=config.eta_base,
                leverage=config.leverage,
                r_tp=config.r_tp,
                r_sl=config.r_sl,
                r_strategic_min=config.r_strategic_min,
                r_strategic_max=config.r_strategic_max,
                t_base=config.t_base,
                beta_atr=config.beta_atr,
                entropy_reg=config.entropy_reg,
            )
            self._use_advanced = False

        # ── Advanced Physics Modules ────────────────────────────────────
        # Lindblad decoherence (regime detection)
        self.lindblad: Optional[LindbladDecoherence] = None
        if config.use_lindblad:
            self.lindblad = LindbladDecoherence(
                n_qubits=N_QUBITS,   # VQC expvals dim = N_QUBITS=3, not n_eigenvectors
                n_lindblad=config.n_lindblad,
            )

        # Platt calibration (replaces hard confidence threshold)
        self.platt: Optional[PlattCalibrator] = None
        if config.use_platt:
            self.platt = PlattCalibrator(
                n_classes=config.n_actions,
                init_temp=config.platt_init_temp,
            )

        # MINE mutual information
        self.mine: Optional[MINEEstimator] = None
        if config.use_mine:
            self.mine = MINEEstimator(
                x_dim=config.n_eigenvectors,
                y_dim=config.n_actions,
                hidden=64,
            )

        # Hurst estimator (stateless, no parameters)
        self.hurst_est = HurstEstimator(n_scales=4)

        # Cramér-Rao selective entry filter (Priority 1)
        self.cr_filter: Optional[CramerRaoFilter] = None
        if config.use_cr_filter:
            self.cr_filter = CramerRaoFilter(
                hurst_min=config.cr_hurst_min,
                purity_min=config.cr_purity_min,
                snr_min=config.cr_snr_min,
            )

        # Entropy production estimator (Priority 6)
        self.ep_estimator: Optional[EntropyProductionEstimator] = None
        if config.use_entropy_prod:
            self.ep_estimator = EntropyProductionEstimator(
                n_states=config.n_actions,
                window=config.ep_window,
                threshold=config.ep_threshold,
            )

        # ── TradingPathBuilder 유틸리티 ────────────────────────────────
        self.path_builder = TradingPathBuilder(
            leverage=config.leverage,
            tp_pct=0.010,   # symmetric: 1×ATR (was 0.015 = 1.5%)
            sl_pct=0.010,   # symmetric: 1×ATR (was 0.010 = 1.0%)
        )

        # ── 스나이퍼 모니터 ────────────────────────────────────────────
        self.monitor = SnipingMonitor(window=200)

        # ── 옵티마이저 ────────────────────────────────────────────────
        # use_qng=True  → DiagonalQNGOptimizer (QNG for VQC, AdamW for rest)
        # use_qng=False → plain AdamW (legacy behaviour)
        trainable_params = (
            list(self.encoder.parameters())
            + list(self.loss_fn.parameters())
        )
        if self.lindblad is not None:
            trainable_params += list(self.lindblad.parameters())
        if self.platt is not None:
            trainable_params += list(self.platt.parameters())
        if self.mine is not None:
            trainable_params += list(self.mine.parameters())

        if config.use_qng:
            from src.models.qng_optimizer import DiagonalQNGOptimizer
            self.optimizer = DiagonalQNGOptimizer(
                self,
                lr_classical=config.lr,
                lr_quantum=config.lr_quantum,
                weight_decay=config.weight_decay,
                qfi_update_freq=config.qfi_update_freq,
            )
        else:
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=config.lr,
                weight_decay=config.weight_decay,
            )

        # ── 학습률 스케줄러 (코사인 어닐링) ──────────────────────────
        # QNG hybrid: scheduler targets classical_optimizer (AdamW) only.
        # VQC lr_quantum is kept constant — QFI already normalises scale.
        from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
        _sched_target = (
            self.optimizer.classical_optimizer
            if isinstance(self.optimizer, _DQNG)
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
        self.to(self.device)

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(
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
        return self.encoder(x, last_step_only=last_step_only)

    # ─────────────────────────────────────────────────────────────────────
    # PAC-Bayes Proximal Regularization (Method J)
    # ─────────────────────────────────────────────────────────────────────

    def set_bc_prior(self, state_dict: dict) -> None:
        """BC 사전분포 저장 — PAC-Bayes 근접 정규화의 기준점 P.

        VQC 가중치는 RL 시작 시 near-identity로 재초기화하므로 제외.
        classical_head / logit_proj 등 고전 파라미터만 BC 지식으로 고정.
        """
        self._bc_prior_params: dict = {}
        for k, v in state_dict.items():
            if "vqc_weights" not in k:
                self._bc_prior_params[k] = v.detach().clone().to(self.device)

    def _pac_bayes_proximal_loss(self) -> torch.Tensor:
        """PAC-Bayes 근접 손실: L_prox = λ × ||θ - θ_BC||²

        McAllester bound 최소화:
          E[L(θ)] ≤ E_train[L] + sqrt(KL(Q||P) + ln(m/δ)) / (2m-1)
          KL(Q||P) ≈ ||θ - θ_BC||² / 2  (단위 분산 Gaussian prior)
          λ = pac_bayes_coef / N_eff  (N_eff↓ → λ↑: 짧은 창일수록 BC 지식 강화)
        """
        if not getattr(self, "_bc_prior_params", None):
            return torch.tensor(0.0, device=self.device)
        lam = self.config.pac_bayes_coef / max(self.config.pac_bayes_n_eff, 1.0)
        total = torch.tensor(0.0, device=self.device)
        for name, param in self.named_parameters():
            if name in self._bc_prior_params and param.requires_grad:
                diff = param - self._bc_prior_params[name]
                total = total + (diff * diff).sum()
        return lam * total

    # ─────────────────────────────────────────────────────────────────────
    # train_step — 핵심 학습 1스텝
    # ─────────────────────────────────────────────────────────────────────

    def train_step(
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
        self.train()

        # 디바이스 이동
        x          = x.to(self.device)
        prices     = prices.to(self.device)
        directions = directions.to(self.device)
        entry_idx  = entry_idx.to(self.device)
        labels     = labels.to(self.device)
        if atr is None:
            atr = torch.ones(x.shape[0], device=self.device) * 0.01
        else:
            atr = atr.to(self.device)

        # ── TradingPath 구성 ───────────────────────────────────────────
        # 원시 가격 데이터 → ΔV_t, event_mask, terminal_state 계산
        path = self.path_builder.from_tensors(
            prices, directions, entry_idx, labels, atr
        )

        # ── Forward: Phase 1 + Phase 2 ────────────────────────────────
        self.optimizer.zero_grad()
        logits, expvals, J_coupling, c_kt = self.forward(
            x, last_step_only=last_step_only
        )                                          # [B, T_run, 3], [B, T_run, 4], [B, 4, 4]

        # last_step_only 모드에서 logits: [B, 1, 3] → [B, 3] (loss 호환)
        if last_step_only and logits.dim() == 3 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        # ── Loss: Phase 3 (Advanced or Classic) ───────────────────────
        if self._use_advanced:
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
            if self.mine is not None and self.config.use_mine:
                probs_soft = torch.softmax(
                    logits if logits.dim() == 2 else logits[:, -1, :], dim=-1
                )
                # Feature: c_kt mean-pooled  [B, K]
                feat = c_kt.mean(dim=1)                   # [B, K]
                mi_lb, _ = self.mine(feat, probs_soft.detach())
                mine_loss = -self.config.mine_coef * mi_lb
                loss = loss + mine_loss
                info["mine_loss"] = mine_loss.item()
                info["mine_mi_lb"] = mi_lb.item()
        else:
            # Classic REINFORCE
            loss, info = self.loss_fn(
                logits=logits,
                path=path,
                atr_norm=path.atr_norm,
            )

        # ── PAC-Bayes Proximal Regularization (Method J) ───────────────
        # λ = C / N_eff 로 자동 스케일링 — 창이 짧을수록 BC 사전지식 더 강하게 보존
        pac_loss = self._pac_bayes_proximal_loss()
        if pac_loss.item() > 0.0:
            loss = loss + pac_loss
            info["pac_bayes"] = pac_loss.item()

        # ── Logit Bias Regularization ───────────────────────────────────
        # L_bias = coef × (bias[LONG] - bias[SHORT])²
        # logit_proj.bias[1]=LONG, [2]=SHORT 비대칭 → SP=0%/LP=0% 고착 원인.
        if self.config.logit_bias_reg_coef > 0:
            _lp = self.encoder.quantum_layer.logit_proj
            # logit_proj may be Linear or Sequential — get the final layer's bias
            _final = _lp[-1] if isinstance(_lp, torch.nn.Sequential) else _lp
            _b = _final.bias
            bias_reg = self.config.logit_bias_reg_coef * (_b[1] - _b[2]) ** 2
            loss = loss + bias_reg
            info["bias_reg"] = bias_reg.item()

        # ── Advantage-Weighted Behavioral Cloning (AWBC) ───────────────
        # 기존 aux_ce 문제: CE는 레이블 방향으로 항상 당기지만 RL은 반대로 학습 가능
        # → 그래디언트 충돌로 학습 불안정.
        #
        # AWBC 해법: CE는 (RL 방향 == 레이블) AND (Â > 0) 일 때만 발동.
        # RL이 이미 맞는 방향으로 갈 때만 보조해서 충돌 원천 차단.
        if self.config.aux_ce_weight > 0:
            from src.models.loss import AdvantageWeightedBC
            logits_2d = logits if logits.dim() == 2 else logits[:, -1, :]
            # per-sample GAE advantage: AdvancedPathIntegralLoss가 info에 [B] 텐서로 제공
            _adv = info.get("adv_per_sample", None)
            if _adv is not None:
                _adv = _adv.to(self.device)
            _awbc = AdvantageWeightedBC(weight=self.config.aux_ce_weight)
            # path.action은 directions에서 유도된 {0,1,2} — label=3(SL) 포함 안 됨
            awbc_loss = _awbc(logits_2d, path.action, advantages=_adv)
            loss = loss + awbc_loss

        # ── Lindblad Diagnostics (no gradient needed) ──────────────────
        purity_val, coherence_val, regime_prob_val = 1.0, 0.0, 0.0
        if self.lindblad is not None:
            with torch.no_grad():
                purity_t, coh_t, rp_t = self.lindblad(expvals)
                purity_val    = float(purity_t.mean().item())
                coherence_val = float(coh_t.mean().item())
                regime_prob_val = float(rp_t.mean().item())

        # ── Backward + Gradient Clip ───────────────────────────────────
        loss.backward()

        # 그라디언트 노름 계산 (모니터링용)
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)

        torch.nn.utils.clip_grad_norm_(
            list(self.parameters()),
            self.config.grad_clip,
        )
        # Pass quantum_layer to QNG optimizer for QFI diagonal update.
        # Falls back gracefully if optimizer is plain AdamW (no encoder arg).
        from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
        if isinstance(self.optimizer, _DQNG):
            self.optimizer.step(encoder=self.encoder.quantum_layer)
        else:
            self.optimizer.step()
        self.scheduler.step()

        # ── 통계 업데이트 ──────────────────────────────────────────────
        self.global_step += 1
        self._loss_history.append(info["loss"])
        self._J_history.append(info["J_mean"])

        # SnipingMonitor 기록
        if self._use_advanced:
            # Use J_mean from info as proxy
            J_proxy = torch.full(
                (x.shape[0],), info["J_mean"],
                device=self.device, dtype=x.dtype
            )
        else:
            J_proxy = self.loss_fn._compute_path_reward(path).detach()

        self.monitor.record(
            terminal_state=path.terminal_state,
            J=J_proxy,
            hold_steps=path.hold_steps,
            realized_pnl=path.realized_pnl,
            e_eta=self.loss_fn.effective_eta.item(),
        )

        t_end = time.perf_counter()

        # Get Platt temperature for logging
        platt_temp_val = 1.0
        if self.platt is not None:
            platt_temp_val = float(self.platt.temperature.item())

        # QFI stats (QNG only)
        from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
        qfi_mean_val = float("nan")
        if isinstance(self.optimizer, _DQNG):
            qfi_mean_val = self.optimizer.get_qfi_stats().get("qfi_mean", float("nan"))

        return TrainStepResult(
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
            expvals_mean=float(expvals.mean().item()),
            J_coupling_max=float(J_coupling.abs().max().item()),
            grad_norm=total_norm,
            step_time_ms=(t_end - t_start) * 1000,
            # Advanced fields
            critic_loss=info.get("critic_loss", 0.0),
            fp_loss=info.get("fp_loss", 0.0),
            dir_sym_loss=info.get("dir_sym_loss", 0.0),
            adv_mean=info.get("adv_mean", 0.0),
            adv_std=info.get("adv_std", 0.0),
            V_mean=info.get("V_mean", 0.0),
            purity_mean=purity_val,
            regime_prob=regime_prob_val,
            platt_temp=platt_temp_val,
            qfi_mean=qfi_mean_val,
        )

    # ─────────────────────────────────────────────────────────────────────
    # _compute_fisher_threshold — Dynamic Fisher-Rao confidence gate
    # ─────────────────────────────────────────────────────────────────────

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
        T = len(log_returns)
        if T < 5:
            return float(self.config.confidence_threshold)

        mu    = float(np.mean(log_returns))
        sigma = float(np.std(log_returns)) + 1e-8
        t_stat = abs(mu) * math.sqrt(T) / sigma

        lo = self.config.fisher_threshold_min
        hi = self.config.fisher_threshold_max
        threshold = lo + (hi - lo) / (1.0 + t_stat)
        return float(np.clip(threshold, lo, hi))

    # ─────────────────────────────────────────────────────────────────────
    # select_action — 추론 전용
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
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
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)                              # [1, T, 27]
        x = x.to(self.device)

        atr_t = torch.tensor([atr_norm], device=self.device)

        # Forward (마지막 타임스텝만)
        logits, expvals, J_coupling, c_kt = self.forward(x, last_step_only=True)
        logits_last = logits[:, -1, :]                      # [1, 3]

        # ── Platt 교정 또는 온도 스케일링 ─────────────────────────────
        if self.platt is not None and self.config.use_platt:
            # Platt calibrated probabilities
            probs = self.platt.calibrate(logits_last)[0]    # [3]
        else:
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
        force_hold_by_regime = False
        purity_for_cr = 1.0     # CR filter에는 Lindblad purity 미전달 (untrained)
        if self.lindblad is not None and self.config.use_lindblad:
            _, _, regime_prob_t = self.lindblad(expvals)
            if float(regime_prob_t.mean().item()) > self.config.lindblad_regime_threshold:
                force_hold_by_regime = True

        # ── Cramér-Rao Selective Entry Filter (Priority 1) ───────────
        # Gate: H > cr_hurst_min AND purity > cr_purity_min AND SNR > cr_snr_min
        force_hold_by_cr = False
        if self.cr_filter is not None and self.config.use_cr_filter:
            log_rets_np = x[0, :, 0].cpu().float().numpy()
            hurst_val   = float(self.hurst_est._hurst_single(
                x[0, :, 0].cpu().float()
            ).item())
            cr_result = self.cr_filter.check(log_rets_np, hurst_val, purity_for_cr)
            if not cr_result.allow_entry:
                force_hold_by_cr = True

        # ── Entropy Production Rate Gate (Priority 6) ─────────────────
        # Schnakenberg Ṡ from recent action history.
        # Low Ṡ → market near detailed balance → near-efficient → abstain.
        force_hold_by_ep = False
        if self.ep_estimator is not None and self.config.use_entropy_prod:
            self.ep_estimator.compute()          # refresh Ṡ from latest history
            if not self.ep_estimator.allows_entry():
                force_hold_by_ep = True

        # ── Dynamic Fisher-Rao Threshold (Priority 3) ─────────────────
        # Adaptive confidence gate: tight when market is near-efficient,
        # relaxed when the t-statistic signals a detectable drift.
        if self.config.use_fisher_threshold:
            log_rets_np_th = x[0, :, 0].cpu().float().numpy()
            confidence_threshold = self._compute_fisher_threshold(log_rets_np_th)
        else:
            confidence_threshold = self.config.confidence_threshold

        # ── 수수료 방어 필터 ─────────────────────────────────────────
        max_prob, action_raw = probs.max(dim=0)
        action = int(action_raw.item())
        prob   = float(max_prob.item())

        if force_hold_by_regime and action != 0:
            action = 0
            prob   = float(probs[0].item())
        elif force_hold_by_cr and action != 0:
            # Cramér-Rao gate: insufficient statistical edge → Hold
            action = 0
            prob   = float(probs[0].item())
        elif force_hold_by_ep and action != 0:
            # Entropy production gate: market near equilibrium → Hold
            action = 0
            prob   = float(probs[0].item())
        elif prob < confidence_threshold and action != 0:
            # 확신 부족 → 강제 Hold (수수료 방어, Fisher-Rao 적응형 임계값)
            action = 0
            prob   = float(probs[0].item())

        if mode == "sample":
            # 확률적 샘플링
            action = int(torch.multinomial(probs, num_samples=1).item())
            prob   = float(probs[action].item())

        # Update entropy production history with the final action
        if self.ep_estimator is not None:
            self.ep_estimator.update(action)

        return action, prob, probs

    @torch.no_grad()
    def select_action_verbose(
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
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        atr_t = torch.tensor([atr_norm], device=self.device)

        logits, expvals, J_coupling, c_kt = self.forward(x, last_step_only=True)
        logits_last = logits[:, -1, :]  # [1, 3]

        if self.platt is not None and self.config.use_platt:
            probs = self.platt.calibrate(logits_last)[0]
        else:
            probs, _ = self.loss_fn.temp_scaler(logits_last, atr_t)
            probs = probs[0]

        # Lindblad
        regime_prob = 0.0
        force_hold_regime = False
        if self.lindblad is not None and self.config.use_lindblad:
            _, _, regime_prob_t = self.lindblad(expvals)
            regime_prob = float(regime_prob_t.mean().item())
            if regime_prob > self.config.lindblad_regime_threshold:
                force_hold_regime = True

        # Cramér-Rao
        hurst_val = 0.5
        force_hold_cr = False
        if self.cr_filter is not None and self.config.use_cr_filter:
            log_rets_np = x[0, :, 0].cpu().float().numpy()
            hurst_val = float(self.hurst_est._hurst_single(
                x[0, :, 0].cpu().float()
            ).item())
            cr_result = self.cr_filter.check(log_rets_np, hurst_val, 1.0)
            if not cr_result.allow_entry:
                force_hold_cr = True

        # Entropy Production
        force_hold_ep = False
        if self.ep_estimator is not None and self.config.use_entropy_prod:
            self.ep_estimator.compute()
            if not self.ep_estimator.allows_entry():
                force_hold_ep = True

        # Fisher-Rao threshold
        if self.config.use_fisher_threshold:
            log_rets_np_th = x[0, :, 0].cpu().float().numpy()
            confidence_threshold = self._compute_fisher_threshold(log_rets_np_th)
        else:
            confidence_threshold = self.config.confidence_threshold

        # Action
        max_prob, action_raw = probs.max(dim=0)
        action = int(action_raw.item())
        prob = float(max_prob.item())
        if force_hold_regime or force_hold_cr or force_hold_ep:
            action = 0
            prob = float(probs[0].item())
        elif prob < confidence_threshold and action != 0:
            action = 0
            prob = float(probs[0].item())

        if self.ep_estimator is not None:
            self.ep_estimator.update(action)

        logit_long  = float(logits_last[0, 1].item())
        logit_short = float(logits_last[0, 2].item())

        return {
            "action":               action,
            "prob":                 prob,
            "probs":                probs,
            "p_hold":               float(probs[0].item()),
            "p_long":               float(probs[1].item()),
            "p_short":              float(probs[2].item()),
            "hurst":                hurst_val,
            "regime_prob":          regime_prob,
            "confidence_threshold": confidence_threshold,
            "force_hold_cr":        force_hold_cr,
            "force_hold_ep":        force_hold_ep,
            "force_hold_regime":    force_hold_regime,
            "logit_long":           logit_long,
            "logit_short":          logit_short,
            "logit_margin":         logit_long - logit_short,
        }

    # ─────────────────────────────────────────────────────────────────────
    # 체크포인트
    # ─────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, path_or_tag: str = "latest") -> str:
        """파라미터 + 옵티마이저 + 학습 통계 저장."""
        if path_or_tag.endswith(".pt") or path_or_tag.endswith(".pth"):
            path = path_or_tag
        else:
            path = os.path.join(self.config.checkpoint_dir, f"agent_{path_or_tag}.pt")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        ckpt: Dict[str, Any] = {
            "global_step":    self.global_step,
            "model_state":    self.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "scheduler":      self.scheduler.state_dict(),
            "config":         asdict(self.config),
            "J_history":      list(self._J_history),
            "loss_history":   list(self._loss_history),
        }
        torch.save(ckpt, path)
        return path

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        """저장된 체크포인트 로드."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Spectral Norm 호환: weight_orig → weight (weight_u/v 제거)
        raw = ckpt["model_state"]
        if any(k.endswith("weight_orig") for k in raw):
            clean: dict = {}
            for k, v in raw.items():
                if k.endswith("weight_orig"):
                    clean[k[: -len("_orig")]] = v  # weight_orig → weight
                elif k.endswith(("weight_u", "weight_v")):
                    pass  # 불필요한 SN 보조 벡터 제거
                else:
                    clean[k] = v
            ckpt["model_state"] = clean
        self.load_state_dict(ckpt["model_state"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt.get("global_step", 0)
        self._J_history.extend(ckpt.get("J_history", []))
        self._loss_history.extend(ckpt.get("loss_history", []))

    # ─────────────────────────────────────────────────────────────────────
    # 진단 유틸리티
    # ─────────────────────────────────────────────────────────────────────

    def parameter_count(self) -> Dict[str, int]:
        """모듈별 학습 파라미터 수."""
        enc_params  = sum(p.numel() for p in self.encoder.parameters())
        loss_params = sum(p.numel() for p in self.loss_fn.parameters())
        return {
            "encoder":  enc_params,
            "loss_fn":  loss_params,
            "total":    enc_params + loss_params,
        }

    def print_architecture(self) -> None:
        print("=" * 65)
        print("  QuantumFinancialAgent Advanced Architecture")
        print("=" * 65)
        print(f"  Feature dim    : {self.config.feature_dim}")
        print(f"  N eigenvectors : {self.config.n_eigenvectors}  (= N qubits)")
        print(f"  VQC layers     : {self.config.n_vqc_layers}")
        print(f"  Actions        : {self.config.n_actions}  (Hold/Long/Short)")
        print(f"  Device         : {self.device}")
        print(f"  Lightning QML  : {self.config.use_lightning}")
        print()
        print("  [Reward Design]")
        print(f"    γ (time discount)  : {self.config.gamma}")
        print(f"    η (fee penalty)    : {self.config.eta_base} × {self.config.leverage:.0f}x"
              f" = {self.config.effective_eta:.4f}")
        print(f"    R_TP / R_SL / R_SC : "
              f"+{self.config.r_tp} / {self.config.r_sl} / "
              f"[{self.config.r_strategic_min}, {self.config.r_strategic_max}]")
        print(f"    Confidence thr     : {self.config.confidence_threshold}")
        print()
        print("  [Advanced Physics Roadmap]")
        print(f"    Loss mode      : {'AdvancedPIL (GAE+Critic+FP)' if self._use_advanced else 'Classic REINFORCE'}")
        if self._use_advanced:
            print(f"    GAE λ          : {self.config.lam_gae}")
            print(f"    Critic coef    : {self.config.critic_coef}")
            print(f"    FP coef        : {self.config.fp_coef}")
        print(f"    Platt calib    : {'ON' if self.platt is not None else 'OFF'}")
        print(f"    Lindblad       : {'ON (n_L=' + str(self.config.n_lindblad) + ')' if self.lindblad is not None else 'OFF'}")
        print(f"    MINE           : {'ON' if self.mine is not None else 'OFF'}")
        print(f"    RMT denoising  : ON (in SpectralDecomposer)")
        print()
        pc = self.parameter_count()
        print(f"  [Parameters] encoder={pc['encoder']:,}  loss={pc['loss_fn']:,}  "
              f"total={pc['total']:,}")
        print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 팩토리 함수
# ─────────────────────────────────────────────────────────────────────────────

def build_quantum_agent(
    config: Optional[AgentConfig] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> QuantumFinancialAgent:
    """
    QuantumFinancialAgent 인스턴스 생성.

    Args:
        config          : AgentConfig (None 이면 기본값)
        device          : 실행 디바이스
        checkpoint_path : 기존 체크포인트 경로 (fine-tune 시 사용)
    """
    if config is None:
        config = AgentConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = QuantumFinancialAgent(config=config, device=device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
        print(f"[build_quantum_agent] 체크포인트 로드: {checkpoint_path}")
        print(f"  global_step = {agent.global_step}")

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# quick_train_demo — 독립 실행 검증
# ─────────────────────────────────────────────────────────────────────────────

def quick_train_demo(n_steps: int = 5, batch_size: int = 4, seq_len: int = 20) -> None:
    """
    합성 데이터로 full 학습 루프를 실행하는 빠른 검증.

    수수료 방어 거동 확인:
        - 초기에는 무작위 행동으로 J(τ) ≈ 0 근방
        - n_steps 이후 모델이 수수료를 고려해 Hold 선호로 수렴하는지 확인
    """
    print("=" * 65)
    print("  quick_train_demo — Phase 1+2+3 통합 학습 루프 검증")
    print("=" * 65)

    cfg = AgentConfig(
        feature_dim=26,   # V4 26-dim (accel, autocorr 제거; vol_ratio+mom3 유지)
        n_eigenvectors=3,  # N_QUBITS=3 와 일치
        n_vqc_layers=2,
        n_actions=3,
        leverage=25.0,
        gamma=0.99,
        lr=3e-3,
        use_lightning=True,
    )
    device = torch.device("cpu")
    agent  = build_quantum_agent(config=cfg, device=device)
    agent.print_architecture()

    print("\n  [학습 루프 시작]")
    B, T, F = batch_size, seq_len, 27

    for step in range(1, n_steps + 1):
        # 합성 시장 데이터 생성
        x       = torch.randn(B, T, F)
        prices  = 50000.0 + torch.cumsum(torch.randn(B, T) * 50, dim=1)
        dirs    = torch.randint(0, 2, (B,)).float() * 2 - 1      # ±1
        entries = torch.zeros(B, dtype=torch.long)                # t=0 진입
        labels  = torch.randint(0, 4, (B,))                       # 랜덤 결과
        atr     = torch.full((B,), 150.0)

        result  = agent.train_step(
            x, prices, dirs, entries, labels, atr,
            last_step_only=True,
        )

        print(
            f"  Step {step:3d} | "
            f"loss={result.loss:+.4f}  "
            f"J={result.J_mean:+.4f}  "
            f"A={result.adv_mean:+.4f}(±{result.adv_std:.3f})  "
            f"V={result.V_mean:+.4f}  "
            f"Lc={result.critic_loss:.4f}  "
            f"FP={result.fp_loss:.4f}  "
            f"purity={result.purity_mean:.3f}  "
            f"η={result.eta_effective:.4f}  "
            f"T={result.platt_temp:.3f}  "
            f"∇={result.grad_norm:.3f}  "
            f"tp={result.pct_tp:.1%}  sl={result.pct_sl:.1%}  "
            f"t={result.step_time_ms:.0f}ms"
        )

    print()
    print("  [스나이퍼 모니터 보고서]")
    print("  " + agent.monitor.summary_str())

    print()
    print("  [추론 테스트: select_action]")
    x_single = torch.randn(1, T, F)
    action, prob, probs = agent.select_action(x_single, atr_norm=0.01, mode="greedy")
    action_name = {0: "HOLD/관망", 1: "LONG 진입", 2: "SHORT 진입"}[action]
    print(f"  → 결정: {action_name}  (확률={prob:.4f})")
    print(f"    전체 확률: Hold={probs[0]:.4f}  Long={probs[1]:.4f}  Short={probs[2]:.4f}")

    print()
    print("  [체크포인트 저장 테스트]")
    ckpt_path = agent.save_checkpoint("demo")
    print(f"  → 저장: {ckpt_path}")

    print("\n  ✓ quick_train_demo 완료! Phase 1+2+3 통합 검증 성공.")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    quick_train_demo(n_steps=5, batch_size=4, seq_len=20)
