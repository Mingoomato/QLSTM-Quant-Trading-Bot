"""
ensemble_agent.py — BC Ensemble Inference
═══════════════════════════════════════════════════════════════════════════════

5개 BC 모델의 softmax 확률 평균으로 액션 결정 (Probability Averaging).

전략:
  Hard voting 대신 확률 평균을 사용:
  - 각 모델의 softmax 출력 [3]을 평균 → avg_probs [3]
  - argmax(avg_probs) → 최종 액션
  - avg_probs[action] >= confidence_threshold → 진입

이유:
  Hard voting(다수결)은 개별 모델 정확도 > 50% 일 때만 효과적.
  예: 개별 39% → 3/5 hard voting → 실제 정확도 ~30% (수학적 열화)
  확률 평균: 노이즈 상쇄 → 단일 모델보다 안정적, 정확도 저하 없음

단일 에이전트와 동일한 select_action() 인터페이스 → backtest 코드 수정 최소화
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import pickle
from typing import List, Tuple, Optional

import numpy as np
import torch

from src.models.integrated_agent import build_quantum_agent, AgentConfig


class EnsembleAgent:
    """BC 앙상블 에이전트.

    단일 QuantumFinancialAgent와 동일한 select_action() 인터페이스를 제공해
    backtest_model_v2.py 수정 없이 사용 가능.
    """

    def __init__(
        self,
        ensemble_dir: str,
        device: torch.device,
        vote_threshold: Optional[int] = None,
        confidence_threshold: float = 0.45,
    ):
        self.device               = device
        self.confidence_threshold = confidence_threshold

        # config.json 로드
        config_path = os.path.join(ensemble_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Ensemble config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.n_models = self.cfg["n_models"]
        print(f"  [Ensemble] {self.n_models}개 모델 로드 중 "
              f"(prob_avg mode, conf_threshold={confidence_threshold}) ...")

        # 스케일러 로드
        scaler_path = self.cfg.get("scaler_path", os.path.join(ensemble_dir, "bc_scaler.pkl"))
        self.scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print(f"  [Ensemble] Scaler loaded: {scaler_path}")

        # 각 모델 로드
        self.agents = []
        feature_dim = self.cfg["feature_dim"]
        for ckpt_path in self.cfg["checkpoints"]:
            agent = self._load_agent(ckpt_path, feature_dim)
            self.agents.append(agent)
            print(f"  [Ensemble] ✅ {os.path.basename(ckpt_path)}")

        print(f"  [Ensemble] 로드 완료. prob_avg mode ({self.n_models}개 모델)")

    def _load_agent(self, ckpt_path: str, feature_dim: int):
        """단일 체크포인트에서 에이전트 복원."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        config = AgentConfig(
            feature_dim=feature_dim,
            confidence_threshold=0.0,   # 앙상블 레벨에서 필터링
        )
        agent = build_quantum_agent(config=config, device=self.device)
        agent.load_state_dict(ckpt["state_dict"])

        # LDA 가중치 복원
        lda_W = ckpt.get("lda_W")
        if lda_W is not None:
            decomposer = agent.encoder.decomposer
            decomposer._lda_W      = lda_W
            decomposer._lda_fitted = True

        agent.eval()
        return agent

    @torch.no_grad()
    def select_action(
        self,
        x_tensor: torch.Tensor,
        atr_norm: float = 0.0,
        mode: str = "greedy",
    ) -> Tuple[int, float, np.ndarray]:
        """
        N개 모델의 softmax 확률 평균으로 액션 결정.

        Returns:
            action      : 0=HOLD, 1=LONG, 2=SHORT
            confidence  : avg_probs[action]
            probs       : 평균 softmax 확률 [3]
        """
        # 스케일러 적용 (필요 시)
        x_np = x_tensor.cpu().numpy()
        if self.scaler is not None:
            B, T, F = x_np.shape
            x_flat  = x_np.reshape(-1, F)
            x_flat  = self.scaler.transform(x_flat).astype(np.float32)
            x_np    = x_flat.reshape(B, T, F)
            x_tensor = torch.from_numpy(x_np).to(self.device)

        probs_all = []
        for agent in self.agents:
            logits, _, _, _ = agent.forward(x_tensor, last_step_only=True)
            if logits.dim() == 3:
                logits = logits.squeeze(1)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # [3]
            probs_all.append(probs)

        avg_probs  = np.mean(probs_all, axis=0)   # [3] — 노이즈 상쇄
        action     = int(avg_probs.argmax())
        confidence = float(avg_probs[action])

        # confidence 임계값 미달 시 HOLD
        if action != 0 and confidence < self.confidence_threshold:
            action = 0

        return action, confidence, avg_probs

    # ─── backtest 호환: config 속성 ───────────────────────────────────────────
    @property
    def config(self):
        """backtest_model_v2.py가 agent.config.leverage 등에 접근하므로 프록시 제공."""
        return self.agents[0].config if self.agents else None

    def eval(self):
        for a in self.agents:
            a.eval()
        return self

    def to(self, device):
        self.device = device
        for a in self.agents:
            a.to(device)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────
def load_ensemble(
    ensemble_dir: str,
    device: torch.device,
    vote_threshold: Optional[int] = None,
    confidence_threshold: float = 0.45,
) -> EnsembleAgent:
    """ensemble_dir에서 EnsembleAgent 로드."""
    return EnsembleAgent(
        ensemble_dir=ensemble_dir,
        device=device,
        vote_threshold=vote_threshold,
        confidence_threshold=confidence_threshold,
    )
