"""
quantum_layers.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quantum Trading V2 — Phase 2 구현체

마스터플랜 참조: Part 2.3 양자 상태 인코딩 (Bloch Sphere)
           Part 2.4 큐비트 상호작용 엔진 (Quantum Hamiltonian)
           Part 3.3 로짓 및 확률 교정

역할:
    Phase 1(SpectralDecomposer)이 출력한 투영값 c_{k,t} 및 Δc_{k,t}를
    N-Qubit PennyLane 양자 회로에 물리적으로 주입하고:

    ① Bloch Sphere Encoding: RY(θ), RZ(φ) 게이트
    ② Ising Hamiltonian:   IsingZZ(π·J_ij) 상호작용
    ③ VQC Layer:           StronglyEntanglingLayers (학습 파라미터)
    ④ Data Re-uploading:   Pérez-Salinas 2020 — 중간 레이어 피처 재주입
    ⑤ Measurement:         ⟨σ^z⟩ 기댓값 → 3-class 로짓으로 변환

동적 큐비트 쌍:
    전역 QUBIT_PAIRS 하드코딩 완전 삭제.
    TorchVQC / QuantumHamiltonianLayer 생성자에서
    itertools.combinations(range(n_qubits), 2) 로 동적 생성.
    N=2 → 1쌍, N=4 → 6쌍. n_qubits 변경만으로 회로 전체가 재조립됨.

수정 이력:
    - Fix A: IsingCouplingEstimator T<2 붕괴 방지 + J 캐시
    - Fix B: TorchVQC Data Re-uploading (Pérez-Salinas 2020)
    - Fix C: 전역 QUBIT_PAIRS 삭제 → 동적 생성 (Gemini 지적)
    - Fix D: _apply_1q 하드코딩 '4' 제거 → n 파라미터화

디바이스: lightning.qubit (C++ 커널, CPU/GPU 가속)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations  # 파이썬의 타입 힌트를 최신 방식으로 사용할 수 있게 한다

import itertools  # 큐비트 쌍 조합을 만들기 위한 도구
import math  # 원주율(π) 같은 수학 상수를 쓰기 위한 도구
from typing import List, Optional, Tuple  # 변수의 종류를 명확히 표시하기 위한 도구

import pennylane as qml  # 양자 컴퓨터 시뮬레이션 도구 (PennyLane)
import torch  # 딥러닝 프레임워크 (PyTorch)
import torch.nn as nn  # 신경망 레이어들을 쓰기 위한 도구

# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────

N_QUBITS: int = 3          # 사용할 큐비트(양자 비트) 개수 — 3개로 고정 (너무 많으면 학습이 안 됨)
N_VQC_LAYERS: int = 3      # VQC(양자 회로)를 몇 층 쌓을지 (많을수록 표현력↑, 속도↓)
N_ACTIONS: int = 3         # 모델이 고를 수 있는 행동 수: [관망(0), 롱(+1), 숏(-1)]

# 큐비트별 물리적 레이블 (n_qubits 에 따라 앞 N개가 사용됨)
QUBIT_LABELS = [
    "Trend Spin",        # 큐비트 0: 추세 방향 (MACD, 모멘텀 관련)
    "Volatility Phase",  # 큐비트 1: 변동성 위상 (ATR, 거래량 관련)
    "Momentum Velocity", # 큐비트 2: 모멘텀 속도 (RSI, ROC 관련) — N>2 시 사용
    "Volume Mass",       # 큐비트 3: 거래량 질량 (OBV, VWAP 관련) — N>3 시 사용
]
# ※ 전역 QUBIT_PAIRS 완전 삭제 —
#   TorchVQC.__init__ 에서 itertools.combinations(range(N), 2) 로 동적 생성


# ─────────────────────────────────────────────────────────────────────────────
# 상관계수 기반 J_ij 계산
# ─────────────────────────────────────────────────────────────────────────────

class IsingCouplingEstimator(nn.Module):
    """
    N개 고유 성분 c_{k,t} 으로부터 이징 상호작용 계수 J_ij 를 계산.

    마스터플랜 수식 (Part 2.4):
        J_ij > 0 : 인력 — 동반 상승/하락 시 에너지 안정화
        J_ij < 0 : 척력 — 다이버전스 발생, 관망 신호 도출

        H(t) = -Σ_{i<j} J_ij σ_i^z σ_j^z  - Σ_i h_i σ_i^x

    구현 전략:
        ρ_ij = Pearson(c_i, c_j)
        J_ij = tanh(ρ_ij · α)   [tanh 압축으로 (-1,1) 바운딩]

    Fix A — T=1 붕괴 방지:
        T<2 이면 분산=0 → X_norm=0 → J=0 (IsingZZ 완전 비활성화 버그).
        직전 배치 J 를 _j_last 에 캐시해 재사용; 없으면 zeros 반환.
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,  # 큐비트 수 (기본값: 전역 상수 N_QUBITS)
        alpha_init: float = 1.0,   # 얽힘 강도 초기값
        use_delta_c: bool = True,  # 변화율(delta_c)도 함께 쓸지 여부
    ) -> None:
        super().__init__()  # 부모 클래스(nn.Module)의 초기화 함수를 먼저 실행한다
        self.n_qubits = n_qubits  # 큐비트 수를 인스턴스 변수로 저장
        self.use_delta_c = use_delta_c  # delta_c 사용 여부 저장

        # 학습 가능한 얽힘 강도 스케일 α (log-space로 항상 양수 보장)
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_init), dtype=torch.float32)
            # alpha_init의 로그값을 학습 파라미터로 만든다 (exp로 꺼내면 항상 양수)
        )

        # Fix A: T=1 엣지 케이스 캐시 (데이터가 1개뿐일 때 이전 값을 재사용하기 위한 저장소)
        self._j_last: Optional[torch.Tensor] = None

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)  # log_alpha를 지수함수로 변환 → 항상 양수인 α를 돌려준다

    def forward(
        self,
        c_kt: torch.Tensor,  # [B, T, K] 시간별 고유 성분 값
        delta_c_kt: Optional[torch.Tensor] = None,  # [B, T, K] 시간별 변화율 (없어도 됨)
    ) -> torch.Tensor:
        """
        Args:
            c_kt      : [B, T, K]  고유 공간 투영값  (K=n_qubits)
            delta_c_kt: [B, T, K]  위상 변화율 (선택)
        Returns:
            J: [B, K, K]  이징 상호작용 행렬 (대각=0, 범위 (-1,1))
        """
        if self.use_delta_c and delta_c_kt is not None:
            feat = c_kt + 0.5 * delta_c_kt  # 변화율의 절반을 더해 특징을 풍부하게 만든다
        else:
            feat = c_kt  # 변화율이 없으면 원래 값만 사용한다

        B, T, K = feat.shape  # B=배치 크기, T=시간 길이, K=특징 수

        # ── Fix A: T<2 방어 ───────────────────────────────────────────
        if T < 2:
            # 데이터가 1개뿐이면 상관계수를 계산할 수 없다
            if self._j_last is not None:
                return self._j_last.expand(B, -1, -1)  # 이전에 계산한 J를 배치 크기에 맞게 확장해서 돌려준다
            return torch.zeros(B, K, K, device=feat.device, dtype=feat.dtype)  # 이전 값도 없으면 0 행렬 반환

        # 피어슨 상관계수 행렬 계산
        X = feat.transpose(1, 2)                       # [B, K, T] — 시간과 특징 축을 바꾼다
        X_centered = X - X.mean(dim=-1, keepdim=True)  # [B, K, T] — 각 특징에서 평균을 뺀다 (중심화)
        std = X_centered.std(dim=-1, keepdim=True).clamp(min=1e-8)  # 표준편차 계산 (0이 되지 않게 최소값 보장)
        X_norm = X_centered / std                      # [B, K, T] — 표준편차로 나눠서 정규화한다

        rho = torch.bmm(X_norm, X_norm.transpose(1, 2)) / max(T - 1, 1)  # [B, K, K] — 피어슨 상관계수 행렬

        eye = torch.eye(K, device=feat.device, dtype=feat.dtype).unsqueeze(0)  # 단위 행렬 (대각선이 1)
        rho = rho * (1.0 - eye)  # 자기 자신과의 상관계수(대각선)를 0으로 만든다

        J = torch.tanh(rho * self.alpha)               # [B, K, K] — tanh로 -1~1 범위로 압축한 상호작용 계수

        # 캐시 업데이트 (detach: gradient graph 차단)
        self._j_last = J.detach().mean(dim=0, keepdim=True)  # 배치 평균을 저장해두어 다음 호출 시 재사용한다

        return J  # 이징 상호작용 행렬을 돌려준다


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Context Encoder (Transformer Residual Block)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalContextEncoder(nn.Module):
    """
    LDA 출력 c_kt [B, T, K] 에 시계열 컨텍스트를 추가하는 Transformer 인코더.

    설계 원칙:
    - Residual: output = c_kt + proj_out(transformer(proj_in(c_kt)))
    - proj_out 0으로 초기화 → 학습 초기 output ≈ c_kt (graceful degradation)
    - Pre-LayerNorm (norm_first=True): 깊은 Transformer 불안정 학습 방지
    - dropout=0.0: T≤20 소규모 시퀀스에서 드롭아웃 불필요
    """

    def __init__(
        self,
        in_dim:   int = 3,   # 입력 차원 (큐비트 수와 같음)
        d_model:  int = 16,  # Transformer 내부 차원 (클수록 표현력 높음)
        n_heads:  int = 2,   # 어텐션 헤드 수 (여러 시각으로 동시에 보는 수)
        n_layers: int = 2,   # Transformer 레이어 수
    ) -> None:
        super().__init__()  # 부모 클래스 초기화
        self.proj_in = nn.Linear(in_dim, d_model)  # 입력을 Transformer 크기로 변환하는 선형 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,          # Transformer 내부 차원
            nhead=n_heads,            # 어텐션 헤드 수
            dim_feedforward=d_model * 4,  # 피드포워드 레이어 크기 (d_model의 4배)
            dropout=0.0,              # 드롭아웃 비율 (0: 꺼짐)
            batch_first=True,         # 입력 첫 번째 차원이 배치 크기
            norm_first=True,          # 레이어 정규화를 앞에 적용 (학습 안정화)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)  # 여러 층의 Transformer 인코더
        self.proj_out = nn.Linear(d_model, in_dim)  # Transformer 출력을 원래 차원으로 되돌리는 선형 레이어
        nn.init.zeros_(self.proj_out.weight)  # proj_out 가중치를 0으로 초기화 (처음엔 입력을 그대로 통과시킴)
        nn.init.zeros_(self.proj_out.bias)    # proj_out 편향도 0으로 초기화

    def forward(self, c_kt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_kt: [B, T, K]
        Returns:
            [B, T, K]  residual-enhanced c_kt
        """
        h = self.proj_in(c_kt)      # [B, T, K] → [B, T, d_model] 입력을 Transformer 차원으로 변환
        h = self.transformer(h)     # Transformer 인코더로 시계열 패턴을 파악한다
        delta = self.proj_out(h)    # [B, T, d_model] → [B, T, K] 다시 원래 차원으로 변환한다
        return c_kt + delta         # 원래 입력에 Transformer가 찾아낸 패턴을 더한다 (잔차 연결)


# ─────────────────────────────────────────────────────────────────────────────
# PennyLane QNode 정의 (폴백용 — 실제 실행은 TorchVQC)
# ─────────────────────────────────────────────────────────────────────────────

def _make_device(n_qubits: int, use_lightning: bool = True) -> qml.Device:
    """lightning.gpu → lightning.qubit → default.qubit 순 폴백."""
    if use_lightning:
        try:
            return qml.device("lightning.gpu", wires=n_qubits)  # GPU 가속 양자 시뮬레이터를 먼저 시도
        except Exception:
            pass  # GPU 버전이 없으면 다음으로 넘어간다
        try:
            return qml.device("lightning.qubit", wires=n_qubits)  # CPU C++ 가속 양자 시뮬레이터를 시도
        except Exception:
            pass  # 이것도 없으면 다음으로 넘어간다
    return qml.device("default.qubit", wires=n_qubits)  # 기본 순수 파이썬 양자 시뮬레이터 사용


def _build_qnode(
    dev: qml.Device,                         # 양자 디바이스
    n_qubits: int,                           # 큐비트 수
    n_vqc_layers: int,                       # VQC 레이어 수
    qubit_pairs: List[Tuple[int, int]],      # 큐비트 쌍 목록
) -> qml.QNode:
    """
    단일 샘플 처리용 QNode (폴백용).
    Fix C: 전역 QUBIT_PAIRS 대신 qubit_pairs 인자를 클로저로 캡처.
    """
    @qml.qnode(dev, interface="torch", diff_method="adjoint")  # PyTorch와 연동되는 양자 회로 정의
    def circuit(
        theta: torch.Tensor,    # [K] RY 회전각 (블로흐 구면 위도)
        phi: torch.Tensor,      # [K] RZ 회전각 (블로흐 구면 경도)
        J_upper: torch.Tensor,  # [n_pairs] 큐비트 쌍의 이징 상호작용 세기
        weights: torch.Tensor,  # [L, N, 3] VQC 학습 파라미터
    ) -> list:
        for k in range(n_qubits):
            qml.RY(theta[k], wires=k)   # k번째 큐비트를 theta[k] 각도만큼 Y축으로 회전시킨다
            qml.RZ(phi[k], wires=k)     # k번째 큐비트를 phi[k] 각도만큼 Z축으로 회전시킨다
        for idx, (i, j) in enumerate(qubit_pairs):
            qml.IsingZZ(math.pi * J_upper[idx], wires=[i, j])  # i번째와 j번째 큐비트를 이징 상호작용으로 얽히게 만든다
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))  # 학습 파라미터로 큐비트들을 강하게 얽히게 만든다
        return [qml.expval(qml.PauliZ(k)) for k in range(n_qubits)]  # 각 큐비트를 Z축 방향으로 측정한 기댓값 반환

    return circuit  # 완성된 양자 회로 함수를 돌려준다


def _build_batched_qnode(
    dev: qml.Device,                         # 양자 디바이스
    n_qubits: int,                           # 큐비트 수
    n_vqc_layers: int,                       # VQC 레이어 수
    qubit_pairs: List[Tuple[int, int]],      # 큐비트 쌍 목록
):
    """배치 처리용 QNode. Fix C: qubit_pairs 동적 전달."""
    @qml.batch_params  # 여러 샘플을 한 번에 처리할 수 있게 한다
    @qml.qnode(dev, interface="torch", diff_method="adjoint")  # PyTorch 연동 양자 회로
    def batched_circuit(
        theta: torch.Tensor,    # [B, K] 배치별 RY 각도
        phi: torch.Tensor,      # [B, K] 배치별 RZ 각도
        J_upper: torch.Tensor,  # [B, n_pairs] 배치별 이징 상호작용 세기
        weights: torch.Tensor,  # [L, N, 3] VQC 학습 파라미터 (배치 공유)
    ) -> list:
        for k in range(n_qubits):
            qml.RY(theta[k], wires=k)   # k번째 큐비트를 Y축으로 회전
            qml.RZ(phi[k], wires=k)     # k번째 큐비트를 Z축으로 회전
        for idx, (i, j) in enumerate(qubit_pairs):
            qml.IsingZZ(math.pi * J_upper[idx], wires=[i, j])  # 큐비트 쌍을 이징 상호작용으로 얽는다
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))  # 강한 얽힘 레이어 적용
        return [qml.expval(qml.PauliZ(k)) for k in range(n_qubits)]  # Z축 기댓값 측정

    return batched_circuit  # 배치용 양자 회로 함수 반환


# ─────────────────────────────────────────────────────────────────────────────
# TorchVQC — PennyLane 완전 우회, 순수 PyTorch GPU 네이티브 양자 회로
# ─────────────────────────────────────────────────────────────────────────────

class TorchVQC(nn.Module):
    """
    Pure PyTorch GPU-native N-qubit quantum circuit (N configurable).

    Fix C — 동적 큐비트 쌍:
        __init__ 에서 itertools.combinations(range(N), 2) 로 생성.
        N=2 → [(0,1)], N=4 → [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)].
        전역 QUBIT_PAIRS 에 의존하지 않으므로 n_qubits 자유 변경 가능.

    Fix D — _apply_1q 파라미터화:
        lo = 1 << (n - qubit - 1) 으로 n 을 인자로 받아 하드코딩 제거.

    Fix B — Data Re-uploading (Pérez-Salinas 2020):
        VQC 중간 레이어(마지막 제외) 종료 후 θ×0.5, φ×0.5 재인코딩.

    회로 구조 (n_vqc_layers=L):
        |0…0⟩
         → RY(θ_k) RZ(φ_k)  [인코딩]
         → IsingZZ(π·J_ij)   [해밀토니안, nC2 쌍]
         → for l in 0..L-1:
               Rot(ω,θ,φ) × N  [VQC]
               CNOT ring        [얽힘]
               if l < L-1: RY(θ×0.5) RZ(φ×0.5)  [Re-uploading]
         → ⟨Z_k⟩ 측정
    """

    def __init__(self, n_vqc_layers: int = 2, n_qubits: int = N_QUBITS) -> None:
        super().__init__()  # 부모 클래스 초기화
        self.n_vqc_layers = n_vqc_layers  # VQC 레이어 수 저장
        self.N = n_qubits          # 큐비트 수 (인스턴스 변수)
        self.D = 2 ** n_qubits     # 상태벡터 차원: N개 큐비트는 2^N 개의 상태를 가질 수 있다
        N, D = self.N, self.D      # 편의를 위해 지역 변수로도 만든다

        # ── Fix C: 동적 큐비트 쌍 생성 ──────────────────────────────────
        # 전역 QUBIT_PAIRS 삭제 후 인스턴스 변수로 대체
        self._qubit_pairs: List[Tuple[int, int]] = list(
            itertools.combinations(range(N), 2)
            # N개의 큐비트 중 2개를 뽑는 모든 조합을 만든다 (N=3이면 (0,1),(0,2),(1,2) 총 3쌍)
        )
        n_pairs = len(self._qubit_pairs)  # 큐비트 쌍의 개수

        # ── IsingZZ parity mask [n_pairs, D] ─────────────────────────────
        # 이징 상호작용에서 각 상태가 +1인지 -1인지를 미리 계산해둔다 (속도 향상)
        ising_masks = torch.zeros(n_pairs, D, dtype=torch.float32)  # 모든 값 0으로 초기화
        for p, (qi, qj) in enumerate(self._qubit_pairs):
            # p번째 쌍 (qi, qj)에 대해 모든 상태 s를 검사한다
            for s in range(D):
                bi = (s >> (N - 1 - qi)) & 1  # s번째 상태에서 qi번째 큐비트 비트값 추출
                bj = (s >> (N - 1 - qj)) & 1  # s번째 상태에서 qj번째 큐비트 비트값 추출
                ising_masks[p, s] = 1.0 if (bi ^ bj) else -1.0  # 두 비트가 다르면 +1, 같으면 -1
        self.register_buffer("_ising_masks", ising_masks)  # GPU/CPU 이동 시 자동으로 따라다니는 버퍼에 등록

        # ── PauliZ sign masks [N, D] ──────────────────────────────────────
        # 각 큐비트의 Z측정 부호를 미리 계산해둔다 (|0⟩→+1, |1⟩→-1)
        z_signs = torch.zeros(N, D, dtype=torch.float32)  # 모든 값 0으로 초기화
        for k in range(N):
            # k번째 큐비트에 대해 모든 상태 s를 검사한다
            for s in range(D):
                z_signs[k, s] = 1.0 if not ((s >> (N - 1 - k)) & 1) else -1.0  # 해당 큐비트가 0이면 +1, 1이면 -1
        self.register_buffer("_z_signs", z_signs)  # 버퍼에 등록

        # ── CNOT matrices [L, N, D, D] ────────────────────────────────────
        # 각 레이어, 각 큐비트에 대한 CNOT 게이트 행렬을 미리 만들어둔다
        cnot_all = []
        for l in range(n_vqc_layers):
            rng = (l % (N - 1)) + 1 if N > 1 else 1  # 레이어마다 다른 간격으로 CNOT 연결 (링 토폴로지)
            layer = []
            for ctrl in range(N):
                tgt = (ctrl + rng) % N  # 제어 큐비트로부터 rng 간격 떨어진 큐비트가 타겟
                mat = torch.zeros(D, D, dtype=torch.complex64)  # D×D 복소수 행렬 초기화
                for s in range(D):
                    if (s >> (N - 1 - ctrl)) & 1:
                        # 제어 큐비트가 1이면 타겟 큐비트를 뒤집는다 (XOR)
                        mat[s ^ (1 << (N - 1 - tgt)), s] = 1.0
                    else:
                        mat[s, s] = 1.0  # 제어 큐비트가 0이면 타겟 큐비트는 그대로
                layer.append(mat)
            cnot_all.append(torch.stack(layer))         # [N, D, D] — 한 레이어의 N개 CNOT 행렬을 쌓는다
        self.register_buffer("_cnot_mats", torch.stack(cnot_all))  # [L, N, D, D] — 전체 레이어 CNOT 행렬 등록

    # ── 게이트 빌더 ───────────────────────────────────────────────────────

    @staticmethod
    def _ry(a: torch.Tensor) -> torch.Tensor:
        """[B] → [B, 2, 2] complex64  RY(a) gate — Y축 회전 게이트 행렬을 만든다"""
        c, s = torch.cos(a * 0.5), torch.sin(a * 0.5)  # cos(a/2), sin(a/2) 계산
        return torch.stack([
            torch.stack([ c, -s], dim=-1),  # RY 행렬의 첫 번째 행: [cos, -sin]
            torch.stack([ s,  c], dim=-1),  # RY 행렬의 두 번째 행: [sin,  cos]
        ], dim=-2).to(torch.complex64)  # 복소수 타입으로 변환

    @staticmethod
    def _rz(a: torch.Tensor) -> torch.Tensor:
        """[B] → [B, 2, 2] complex64  RZ(a) gate — Z축 회전 게이트 행렬을 만든다"""
        h = a.to(torch.complex64) * 0.5  # a/2를 복소수로 변환
        z = torch.zeros_like(h)           # 0 텐서 (행렬의 0 원소)
        return torch.stack([
            torch.stack([torch.exp(-1j * h), z             ], dim=-1),  # [e^{-ia/2}, 0]
            torch.stack([z,                  torch.exp(1j * h)], dim=-1),  # [0, e^{ia/2}]
        ], dim=-2)  # RZ 게이트: 대각선에 복소 위상이 들어간 행렬

    @staticmethod
    def _rot(w: torch.Tensor, B: int) -> torch.Tensor:
        """[3] scalar → [B, 2, 2]  Rot(φ,θ,ω) = RZ(ω)@RY(θ)@RZ(φ) — 3개 각도로 임의 단일 큐비트 회전을 만든다"""
        rz_phi   = TorchVQC._rz(w[0].expand(B))   # 첫 번째 파라미터로 Z축 회전
        ry_theta = TorchVQC._ry(w[1].expand(B))   # 두 번째 파라미터로 Y축 회전
        rz_omega = TorchVQC._rz(w[2].expand(B))   # 세 번째 파라미터로 Z축 회전
        return torch.bmm(rz_omega, torch.bmm(ry_theta, rz_phi))  # 세 회전을 순서대로 합성한다

    @staticmethod
    def _apply_1q(
        state: torch.Tensor,   # [B, D] complex64 — 현재 양자 상태 벡터
        gate:  torch.Tensor,   # [B, 2, 2] complex64 — 적용할 단일 큐비트 게이트
        qubit: int,            # 게이트를 적용할 큐비트 번호
        n: int,                # Fix D: 총 큐비트 수 인자화 (하드코딩 '4' 제거)
    ) -> torch.Tensor:         # [B, D] complex64 — 게이트 적용 후 양자 상태
        """Apply a single-qubit gate to qubit k of an n-qubit state vector."""
        B, D = state.shape   # 배치 크기와 상태 벡터 차원
        lo = 1 << (n - qubit - 1)  # qubit 아래쪽 비트들의 묶음 크기 (2^(n-qubit-1))
        hi = 1 << qubit            # qubit 위쪽 비트들의 묶음 크기 (2^qubit)
        s = state.reshape(B, hi, 2, lo)          # 상태를 [상위, 이 큐비트(2), 하위] 구조로 분리
        s = s.permute(0, 2, 1, 3).reshape(B, 2, hi * lo)  # 이 큐비트 차원을 앞으로 이동
        s = torch.bmm(gate, s)                   # 게이트 행렬을 상태에 곱한다 (양자 연산 적용)
        return s.reshape(B, 2, hi, lo).permute(0, 2, 1, 3).reshape(B, D)  # 원래 형태로 되돌린다

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        theta:       torch.Tensor,   # [B, N] RY 인코딩 각도
        phi_enc:     torch.Tensor,   # [B, N] RZ 인코딩 각도
        j_upper:     torch.Tensor,   # [B, n_pairs] 큐비트 쌍 상호작용 세기
        vqc_weights: torch.Tensor,   # [L, N, 3] VQC 회전각 (학습 파라미터)
    ) -> torch.Tensor:               # [B, N] ⟨Z_k⟩ ∈ [-1, 1] 각 큐비트의 Z측정 기댓값
        B  = theta.shape[0]   # 배치 크기
        N  = self.N           # 큐비트 수
        device = theta.device  # CPU인지 GPU인지

        # 초기 상태 |00…0⟩ — 모든 큐비트가 0인 상태에서 시작한다
        state = torch.zeros(B, self.D, dtype=torch.complex64, device=device)  # 전부 0으로 초기화
        state[:, 0] = 1.0 + 0j  # 첫 번째 기저 상태(|000⟩)의 확률진폭을 1로 설정한다

        # 1. Bloch sphere encoding — 시장 데이터를 큐비트에 인코딩한다
        for k in range(N):
            state = self._apply_1q(state, self._ry(theta[:, k]), k, N)    # k번째 큐비트에 RY 회전 적용
            state = self._apply_1q(state, self._rz(phi_enc[:, k]), k, N)  # k번째 큐비트에 RZ 회전 적용

        # 2. IsingZZ interaction — 큐비트 쌍 사이의 상호작용을 적용한다
        phi_iz = (math.pi * j_upper).to(torch.complex64)   # [B, n_pairs] 이징 위상각
        masks  = self._ising_masks.to(torch.complex64)      # [n_pairs, D] 미리 계산한 패리티 마스크
        for p in range(len(self._qubit_pairs)):
            phase = torch.exp(0.5j * phi_iz[:, p, None] * masks[p, None, :])  # 각 상태에 위상 곱하기
            state = state * phase  # 이징 상호작용 적용 (상태에 위상을 곱한다)

        # 3. VQC + Fix B: Data Re-uploading — 학습 가능한 양자 회로 적용
        for l in range(self.n_vqc_layers):
            for k in range(N):
                state = self._apply_1q(state, self._rot(vqc_weights[l, k], B), k, N)  # 각 큐비트에 학습된 회전 적용
            for k in range(N):
                state = torch.matmul(state, self._cnot_mats[l, k].mT)  # CNOT 게이트로 큐비트들을 얽는다
            # Re-uploading: 마지막 레이어 제외, 0.5 스케일로 데이터를 다시 주입한다
            if l < self.n_vqc_layers - 1:
                for k in range(N):
                    state = self._apply_1q(state, self._ry(theta[:, k] * 0.5), k, N)    # 절반 크기로 RY 재인코딩
                    state = self._apply_1q(state, self._rz(phi_enc[:, k] * 0.5), k, N)  # 절반 크기로 RZ 재인코딩

        # 4. ⟨Z_k⟩ = Σ_s |ψ_s|² · sign_k(s) — 각 큐비트를 Z방향으로 측정한다
        probs   = state.abs().pow(2)       # [B, D] 각 상태의 확률 (복소 진폭의 절댓값 제곱)
        expvals = probs @ self._z_signs.T  # [B, N] Z 측정 기댓값: 각 확률과 부호(+1/-1)의 내적
        return expvals.to(dtype=theta.dtype)  # 입력과 같은 데이터 타입으로 변환해서 반환


# ─────────────────────────────────────────────────────────────────────────────
# QuantumHamiltonianLayer — 핵심 nn.Module
# ─────────────────────────────────────────────────────────────────────────────

class QuantumHamiltonianLayer(nn.Module):
    """
    마스터플랜 Part 2.3 + 2.4 의 완전한 구현체.

    n_qubits 를 몇 개로 전달하든 _qubit_pairs, ising_masks, CNOT 행렬이
    자동으로 재조립됨 (Fix C).

    logit_proj = nn.Linear(n_qubits, n_actions):
        N=2 이면 Linear(2,3), N=4 이면 Linear(4,3) — 항상 3-class 출력.
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,          # 큐비트 수
        n_vqc_layers: int = N_VQC_LAYERS,  # VQC 레이어 수
        n_actions: int = N_ACTIONS,         # 출력 행동 수 (3: 관망/롱/숏)
        use_lightning: bool = True,         # PennyLane lightning 백엔드 사용 여부
        alpha_init: float = 1.0,            # 이징 결합 강도 초기값
        use_delta_c: bool = True,           # 변화율 특징 사용 여부
        use_spectral_norm: bool = False,    # 스펙트럴 정규화 사용 여부 (BC=False, RL=True)
    ) -> None:
        super().__init__()  # 부모 클래스 초기화

        self.n_qubits     = n_qubits      # 큐비트 수 저장
        self.n_vqc_layers = n_vqc_layers  # VQC 레이어 수 저장
        self.n_actions    = n_actions     # 행동 수 저장

        # 이징 상호작용 계수 추정기 (Fix A 포함) — 큐비트 쌍 사이의 결합 세기를 계산한다
        self.coupling = IsingCouplingEstimator(
            n_qubits=n_qubits,
            alpha_init=alpha_init,
            use_delta_c=use_delta_c,
        )

        # VQC 학습 파라미터: [L, N, 3] — 양자 회로의 회전각을 학습한다
        vqc_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=n_vqc_layers, n_wires=n_qubits
            # PennyLane에게 파라미터 형태를 물어본다
        )
        # near-identity init: Var[∂L/∂θ]|_{θ≈0} 이론 최대 → 배런 플래토 완화
        # 아주 작은 값으로 초기화해서 학습 초기에 기울기가 사라지는 문제를 줄인다
        self.vqc_weights = nn.Parameter(torch.randn(vqc_shape) * 0.01)

        # Fix C: TorchVQC 에 n_qubits 전달 → 동적 쌍 생성
        self._torch_vqc  = TorchVQC(n_vqc_layers, n_qubits)  # 순수 PyTorch 양자 회로 생성
        self._qubit_pairs = self._torch_vqc._qubit_pairs  # TorchVQC의 큐비트 쌍 목록 참조

        # QFI 캐시 — 양자 피셔 정보 계산용 입력값 저장소
        self._qfi_cache: tuple = ()

        # PennyLane QNode (참조/검증용, Fix C: qubit_pairs 동적 전달)
        self._dev = _make_device(n_qubits, use_lightning=use_lightning)  # 양자 디바이스 생성
        self._qnode = _build_qnode(
            self._dev, n_qubits, n_vqc_layers, self._qubit_pairs
            # 단일 샘플용 PennyLane 양자 회로 생성
        )
        try:
            self._batched_qnode = _build_batched_qnode(
                self._dev, n_qubits, n_vqc_layers, self._qubit_pairs
                # 배치 처리용 PennyLane 양자 회로 생성 시도
            )
            self._use_batched = True  # 배치 모드 사용 가능
        except Exception:
            self._batched_qnode = None  # 배치 모드 실패
            self._use_batched   = False  # 단일 샘플 모드로 대체

        # 로짓 프로젝션: Linear(n_qubits, n_actions)
        # N=2 → Linear(2,3), N=4 → Linear(4,3)
        # logit_proj + classical_head: use_spectral_norm 플래그로 조건부 적용
        # BC (use_spectral_norm=False): 제약 없이 logit 성장 → 클래스 구분력 최대화
        # RL (use_spectral_norm=True) : Lipschitz=1 → AvgMaxProb→1.0 구조적 차단
        if use_spectral_norm:
            from torch.nn.utils import spectral_norm as _sn  # 스펙트럴 정규화 도구 임포트
            self.logit_proj = nn.Sequential(
                _sn(nn.Linear(n_qubits, 16)),  # 스펙트럴 정규화 적용 선형 레이어
                nn.GELU(),                      # GELU 활성화 함수 (부드러운 ReLU)
                nn.Dropout(0.03),               # 3% 확률로 뉴런을 끄는 드롭아웃
                _sn(nn.Linear(16, n_actions))   # 스펙트럴 정규화 적용 출력 레이어
            )
            _classical = nn.Linear(n_qubits, n_actions, bias=True)  # 고전 경로 선형 레이어
            nn.init.xavier_uniform_(_classical.weight, gain=0.5)     # Xavier 초기화 (가중치)
            nn.init.zeros_(_classical.bias)                           # 편향 0으로 초기화
            self.classical_head = _sn(_classical)  # 고전 경로에도 스펙트럴 정규화 적용
        else:
            self.logit_proj = nn.Sequential(
                nn.Linear(n_qubits, 16),  # 큐비트 출력을 16차원으로 변환
                nn.GELU(),                # GELU 활성화
                nn.Dropout(0.03),         # 드롭아웃
                nn.Linear(16, n_actions)  # 16차원에서 행동 수(3)로 변환
            )
            self.classical_head = nn.Linear(n_qubits, n_actions, bias=True)  # 고전 경로 선형 레이어
            nn.init.xavier_uniform_(self.classical_head.weight, gain=0.5)    # Xavier 초기화
            nn.init.zeros_(self.classical_head.bias)                          # 편향 0으로 초기화

        # 외부 자기장 프로젝션 — 시장 전체 평균 상태를 큐비트 공간으로 변환한다
        self.h_proj = nn.Linear(n_qubits, n_qubits, bias=False)

        # Method G-B: 학습 가능한 logit 온도 스케일 α
        # use_spectral_norm=True 시: 방향(Spectral Norm) + 크기(α) 분리
        # use_spectral_norm=False 시: α=3.0 고정값 역할 (초기 수렴 가속)
        self.logit_scale = nn.Parameter(torch.ones(1) * 3.0)  # 로짓 스케일 파라미터 (초기값 3.0)

    def _run_circuit_batch(
        self,
        theta: torch.Tensor,  # [B_eff, K] RY 각도
        phi:   torch.Tensor,  # [B_eff, K] RZ 각도
        J:     torch.Tensor,  # [B_eff, K, K] 이징 상호작용 행렬
    ) -> torch.Tensor:        # [B_eff, K] 측정 기댓값
        """TorchVQC 로 배치 전체를 단일 GPU 커널로 실행."""
        # Fix C: 전역 QUBIT_PAIRS 대신 인스턴스 _qubit_pairs 사용
        j_upper = torch.stack(
            [J[:, i, j] for (i, j) in self._qubit_pairs], dim=1
            # 전체 K×K 행렬에서 큐비트 쌍에 해당하는 값들만 꺼내서 쌓는다
        )  # [B, n_pairs]

        n_cache = min(8, theta.shape[0])  # 최대 8개 샘플만 QFI 캐시에 저장한다
        self._qfi_cache = (
            theta[:n_cache].detach(),    # QFI 계산용 각도 캐시 (기울기 그래프에서 분리)
            phi[:n_cache].detach(),
            j_upper[:n_cache].detach(),
        )

        return self._torch_vqc(theta, phi, j_upper, self.vqc_weights)  # 양자 회로를 실행하고 측정값 반환

    @torch.no_grad()  # 기울기 계산을 끄고 실행 (빠른 연산)
    def compute_qfi_diagonal(self) -> Optional[torch.Tensor]:
        """Diagonal QFI via parameter-shift rule — 파라미터 이동 규칙으로 대각 양자 피셔 정보를 계산한다."""
        if not self._qfi_cache:
            return None  # 캐시가 없으면 계산 불가

        theta, phi, j_upper = self._qfi_cache  # 캐시에서 입력값 꺼내기
        n_params = self.vqc_weights.numel()    # VQC 파라미터 총 개수
        w_shape  = self.vqc_weights.shape      # VQC 파라미터 형태
        w_flat   = self.vqc_weights.detach().clone().reshape(-1)  # 1차원으로 펼친 파라미터
        qfi_diag = torch.zeros(n_params, device=self.vqc_weights.device)  # QFI 대각 원소들
        half_pi  = math.pi / 2.0  # π/2 상수

        for i in range(n_params):
            w_plus      = w_flat.clone(); w_plus[i]  += half_pi   # i번째 파라미터를 +π/2 이동
            w_minus     = w_flat.clone(); w_minus[i] -= half_pi   # i번째 파라미터를 -π/2 이동
            out_plus    = self._torch_vqc(theta, phi, j_upper, w_plus.reshape(w_shape))   # +이동 후 측정
            out_minus   = self._torch_vqc(theta, phi, j_upper, w_minus.reshape(w_shape))  # -이동 후 측정
            qfi_diag[i] = 0.5 * ((out_plus - out_minus) ** 2).mean()  # (출력 차이)²의 절반 = QFI 대각원소

        return qfi_diag.cpu()  # CPU 텐서로 변환해서 반환

    def forward(
        self,
        c_kt:       torch.Tensor,    # [B, T, K] 시간별 고유 성분
        delta_c_kt: torch.Tensor,    # [B, T, K] 시간별 변화율
        last_step_only: bool = False,  # True면 마지막 타임스텝만 처리 (추론 시 빠름)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            c_kt       : [B, T, K]
            delta_c_kt : [B, T, K]
        Returns:
            logits  : [B, T_run, 3]
            expvals : [B, T_run, K]
            J       : [B, K, K]
        """
        B, T, K = c_kt.shape  # 배치 크기, 시간 길이, 큐비트 수

        J       = self.coupling(c_kt, delta_c_kt)        # [B, K, K] 이징 상호작용 계수 계산
        theta   = math.pi * torch.sigmoid(c_kt / 3.0)   # [B, T, K] c_kt를 sigmoid로 0~1, π곱해 0~π 범위 RY 각도
        phi     = math.pi * torch.tanh(delta_c_kt / 3.0) # [B, T, K] delta를 tanh로 -1~1, π곱해 -π~π 범위 RZ 각도
        h_field = torch.tanh(self.h_proj(c_kt.mean(1)) / 3.0)  # [B, K] 시간 평균 c_kt로 외부 자기장 계산

        if last_step_only:
            theta_run, phi_run, T_run = theta[:, -1:, :], phi[:, -1:, :], 1  # 마지막 타임스텝만 사용
        else:
            theta_run, phi_run, T_run = theta, phi, T  # 전체 시간을 사용

        theta_flat  = theta_run.reshape(B * T_run, K)   # [B*T_run, K] 배치와 시간을 합쳐서 평탄화
        phi_flat    = phi_run.reshape(B * T_run, K)     # [B*T_run, K]
        J_flat      = J.unsqueeze(1).expand(-1, T_run, -1, -1).reshape(B * T_run, K, K)  # J를 시간 축으로 복사 후 평탄화

        expvals_flat = self._run_circuit_batch(theta_flat, phi_flat, J_flat)  # 양자 회로 실행
        expvals_flat = expvals_flat.to(dtype=theta_flat.dtype)  # 입력과 같은 dtype으로 맞춘다
        expvals      = expvals_flat.reshape(B, T_run, K)  # [B, T_run, K] 다시 배치와 시간으로 분리

        h_broadcast = h_field.unsqueeze(1).expand(-1, T_run, -1)  # 외부 자기장을 시간 축으로 복사

        # ── 양자 경로: VQC 보정항 δO_quantum ────────────────────────────────
        logits_quantum = self.logit_proj(expvals + h_broadcast)  # [B, T_run, 3] 양자 측정값에 자기장 더해서 로짓 계산

        # ── 고전 경로: LDA c_kt zeroth-order 해 O_classical ─────────────────
        # last_step_only=True → c_kt[:, -1:, :],  False → c_kt 전체
        c_classical = c_kt[:, -1:, :] if last_step_only else c_kt   # [B, T_run, K] 고전 경로 입력
        logits_classical = self.classical_head(c_classical)          # [B, T_run, 3] 고전 선형 변환으로 로짓 계산

        # ── 섭동 가산 + 온도 스케일: O_total = α × (O_classical + δO_quantum) ──
        # 배런 플래토 초기: logits_quantum ≈ bias → logits ≈ α × logits_classical
        # VQC 학습 후: logits_quantum이 비선형 보정을 추가
        # α (logit_scale): 방향은 Spectral Norm이 고정, 크기는 α가 자유 학습
        logits = self.logit_scale * (logits_classical + logits_quantum)  # [B, T_run, 3] 고전+양자 로짓 합산 후 스케일 적용

        return logits, expvals, J  # 로짓, 측정값, 상호작용 행렬을 돌려준다


# ─────────────────────────────────────────────────────────────────────────────
# QuantumMarketEncoder — SpectralDecomposer + QuantumHamiltonianLayer 통합
# ─────────────────────────────────────────────────────────────────────────────

class QuantumMarketEncoder(nn.Module):
    """Phase 1 + Phase 2 통합 인코더 — 시장 데이터를 양자 회로에 넣어 행동 로짓을 뽑는다."""

    def __init__(
        self,
        feature_dim:    int  = 27,          # 입력 특징 차원 수
        n_eigenvectors: int  = N_QUBITS,    # LDA/EDMD에서 뽑을 고유벡터 수
        n_vqc_layers:   int  = N_VQC_LAYERS,  # VQC 레이어 수
        n_actions:      int  = N_ACTIONS,   # 행동 수
        use_lightning:  bool = True,        # PennyLane 가속 백엔드 사용 여부
        learnable_basis: bool = False,      # 기저 벡터를 학습할지 여부
        use_edmd:       bool = True,        # EDMD(확장 DMD) 사용 여부
        use_lda:        bool = True,        # LDA(선형 판별 분석) 사용 여부
        use_spectral_norm: bool = False,    # 스펙트럴 정규화 (BC=False, RL=True)
        use_transformer: bool = True,       # Transformer 시계열 인코더 사용 여부
        transformer_d_model: int = 16,      # Transformer 내부 차원
        transformer_n_heads: int = 2,       # Transformer 어텐션 헤드 수
        transformer_n_layers: int = 2,      # Transformer 레이어 수
    ) -> None:
        super().__init__()  # 부모 클래스 초기화

        from src.data.spectral_decomposer import SpectralDecomposer  # 스펙트럴 분해기 임포트
        self.decomposer = SpectralDecomposer(
            feature_dim=feature_dim,
            n_eigenvectors=n_eigenvectors,
            learnable_basis=learnable_basis,
            use_edmd=use_edmd,
            use_lda=use_lda,
            # 시장 데이터를 LDA/EDMD로 분해해서 고유 성분 c_kt를 뽑는 모듈
        )
        # VQC는 항상 N_QUBITS(3) 고정 — n_eigenvectors와 독립
        # barren plateau: Var[∂L/∂θ] ~ 1/2^N_QUBITS (n_eigenvectors 변경 시에도 불변)
        self.quantum_layer = QuantumHamiltonianLayer(
            n_qubits=N_QUBITS,               # 큐비트 수는 항상 3으로 고정
            n_vqc_layers=n_vqc_layers,
            n_actions=n_actions,
            use_lightning=use_lightning,
            use_spectral_norm=use_spectral_norm,
        )
        self.temporal_encoder: Optional[TemporalContextEncoder] = None
        if use_transformer:
            self.temporal_encoder = TemporalContextEncoder(
                in_dim=n_eigenvectors,
                d_model=transformer_d_model,
                n_heads=transformer_n_heads,
                n_layers=transformer_n_layers,
                # 시계열 패턴을 Transformer로 포착하는 인코더
            )

        # Projection from n_eigenvectors → N_QUBITS when they differ.
        # Transformer sees n_eigenvectors-dim (richer); VQC sees N_QUBITS=3 (barren safe).
        if n_eigenvectors != N_QUBITS:
            self.proj_to_qubits: Optional[nn.Linear] = nn.Linear(
                n_eigenvectors, N_QUBITS, bias=False
                # n_eigenvectors 차원을 N_QUBITS(3) 차원으로 줄이는 선형 변환
            )
            nn.init.xavier_uniform_(self.proj_to_qubits.weight, gain=0.5)  # Xavier 초기화
        else:
            self.proj_to_qubits = None  # 차원이 같으면 변환 불필요

    def forward(
        self,
        x: torch.Tensor,               # [B, T, feature_dim] 원시 특징 입력
        last_step_only: bool = False,  # 마지막 타임스텝만 처리할지 여부
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, c_kt, delta_c_kt, _ = self.decomposer(x)  # 스펙트럴 분해로 고유 성분과 변화율 추출
        if self.temporal_encoder is not None:
            c_kt = self.temporal_encoder(c_kt)  # Transformer로 시계열 컨텍스트 강화
        # Project to qubit dim if n_eigenvectors != n_qubits
        if self.proj_to_qubits is not None:
            c_kt        = self.proj_to_qubits(c_kt)        # 고유 성분 차원을 큐비트 수에 맞게 줄인다
            delta_c_kt  = self.proj_to_qubits(delta_c_kt)  # 변화율도 같은 변환 적용
        logits, expvals, J = self.quantum_layer(
            c_kt, delta_c_kt, last_step_only=last_step_only
            # 양자 해밀토니안 레이어로 로짓 계산
        )
        return logits, expvals, J, c_kt  # 로짓, 측정값, 상호작용, 고유 성분 반환


# ─────────────────────────────────────────────────────────────────────────────
# 팩토리 함수
# ─────────────────────────────────────────────────────────────────────────────

def build_quantum_layer(
    n_qubits:     int  = N_QUBITS,
    n_vqc_layers: int  = N_VQC_LAYERS,
    n_actions:    int  = N_ACTIONS,
    use_lightning: bool = True,
    device: Optional[torch.device] = None,
) -> QuantumHamiltonianLayer:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU가 있으면 GPU, 없으면 CPU 사용
    return QuantumHamiltonianLayer(
        n_qubits=n_qubits, n_vqc_layers=n_vqc_layers,
        n_actions=n_actions, use_lightning=use_lightning,
    ).to(device)  # 생성한 레이어를 지정한 디바이스로 이동한다


def build_quantum_market_encoder(
    feature_dim:    int  = 27,
    n_eigenvectors: int  = N_QUBITS,
    n_vqc_layers:   int  = N_VQC_LAYERS,
    n_actions:      int  = N_ACTIONS,
    use_lightning:  bool = True,
    learnable_basis: bool = False,
    device: Optional[torch.device] = None,
) -> QuantumMarketEncoder:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 디바이스 자동 선택
    return QuantumMarketEncoder(
        feature_dim=feature_dim, n_eigenvectors=n_eigenvectors,
        n_vqc_layers=n_vqc_layers, n_actions=n_actions,
        use_lightning=use_lightning, learnable_basis=learnable_basis,
    ).to(device)  # 생성한 인코더를 지정 디바이스로 이동


# ─────────────────────────────────────────────────────────────────────────────
# 자가 검증 (N=2 / N=4 양쪽 테스트)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)  # 구분선 출력
    print("  QuantumHamiltonianLayer — Dynamic N-Qubit Self-Test")  # 테스트 제목
    print("=" * 65)
    torch.manual_seed(42)  # 난수 시드 고정 (재현 가능성 확보)

    for n_q in [2, 4]:  # 큐비트 2개와 4개 두 가지 경우를 테스트
        print(f"\n[N_QUBITS={n_q}]")  # 현재 테스트하는 큐비트 수 출력
        ql = QuantumHamiltonianLayer(n_qubits=n_q, n_vqc_layers=2, n_actions=3)  # 레이어 생성
        pairs = ql._qubit_pairs  # 큐비트 쌍 목록
        print(f"  qubit_pairs = {pairs}  ({len(pairs)} pairs)")   # 큐비트 쌍 출력
        print(f"  vqc_weights = {tuple(ql.vqc_weights.shape)}")   # VQC 파라미터 형태 출력
        print(f"  logit_proj  = Linear({n_q}, 3)")                 # 로짓 레이어 형태 출력

        B, T = 2, 20  # 배치 크기 2, 시간 길이 20
        c  = torch.randn(B, T, n_q)   # 임의 고유 성분 생성
        dc = torch.randn(B, T, n_q)   # 임의 변화율 생성
        logits, expvals, J = ql(c, dc, last_step_only=True)  # 마지막 타임스텝만 처리

        assert logits.shape  == (B, 1, 3),    f"logits shape wrong: {logits.shape}"    # 로짓 형태 검증
        assert expvals.shape == (B, 1, n_q),  f"expvals shape wrong: {expvals.shape}"  # 측정값 형태 검증
        assert J.shape       == (B, n_q, n_q),f"J shape wrong: {J.shape}"              # 상호작용 행렬 형태 검증

        logits.sum().backward()  # 역전파 실행 (기울기 계산)
        grad = ql.vqc_weights.grad.norm().item()  # VQC 파라미터의 기울기 크기
        print(f"  logits={tuple(logits.shape)} expvals={tuple(expvals.shape)} J={tuple(J.shape)}")  # 형태 출력
        print(f"  VQC grad norm = {grad:.4f}  (0이면 BARREN)")  # 기울기가 0이면 배런 플래토 문제

        # Fix A T=1 검증 — 데이터 1개일 때 캐시 재사용 확인
        _ = ql.coupling(c)                          # 정상 데이터로 한 번 실행해서 캐시를 채운다
        J_t1 = ql.coupling(torch.randn(B, 1, n_q))  # T=1(데이터 1개)로 테스트
        print(f"  T=1 J.abs().mean = {J_t1.abs().mean().item():.4f}  (캐시 재사용)")  # 캐시가 잘 작동하는지 확인

    print("\n  ALL CHECKS PASSED")  # 모든 검사 통과
    print("=" * 65)
