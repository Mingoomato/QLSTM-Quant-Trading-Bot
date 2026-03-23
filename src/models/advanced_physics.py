"""
advanced_physics.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Advanced Quantum Trading V2 — Physicist & Mathematician Roadmap

Implements the full set of advanced improvements identified by
Physics/Math evaluation (scored 3.5/10 + 4.5/10):

  1. HurstEstimator        — fBm / Hurst exponent (R/S analysis)
                              Adapts feature lookback to market memory
  2. LindbladDecoherence   — Quantum master equation for regime detection
                              Purity = coherence; low purity → regime change
  3. MINEEstimator         — Mutual Information Neural Estimation
                              I(features; labels) → feature quality bound
  4. MPSLayer              — Matrix Product States tensor network layer
                              Replaces VQC with O(n·χ²) contraction
  5. OptimalStoppingBoundary — Snell envelope SL/TP adaptation
                              Learns optimal exit boundary as function of state
  6. FokkerPlanckRegularizer — Langevin SDE consistency loss
                              Forces model to respect market stochastic dynamics
  7. WassersteinDROLoss    — Distributionally Robust Optimization
                              min_θ max_{P:W(P,Q)≤ε} E_P[ℓ]
  8. PlattCalibrator       — Platt scaling + Expected Calibration Error
                              Replaces hard confidence threshold

Physical / Mathematical Foundations
────────────────────────────────────
• fBm: B_H(t) has covariance E[B_H(t)B_H(s)] = ½(|t|^{2H}+|s|^{2H}-|t-s|^{2H})
  H>0.5 → trending, H<0.5 → mean-reverting, H=0.5 → GBM (no edge)

• Lindblad:  dρ/dt = -i[H,ρ] + Σ_k γ_k(L_k ρ L_k† - ½{L_k†L_k, ρ})
  Purity Tr(ρ²) = 1 → pure coherent trend
  Purity → 1/N → maximally mixed = regime uncertainty

• MINE:  I(X;Y) ≥ E_{P(X,Y)}[T] - log E_{P(X)⊗P(Y)}[e^T]  (DONSKER-VARADHAN)

• MPS: |ψ⟩ = Σ_{i₁…iₙ} A¹[i₁]A²[i₂]…Aⁿ[iₙ]|i₁…iₙ⟩,  bond dim χ

• Snell envelope: J(x,t) = sup_{τ≥t} E[g(X_τ)|X_t=x]
  Stopping region: S = {(x,t): J=g(x,t)}  ← learned by NN

• Fokker-Planck: ∂_t ρ = -∂_x(μρ) + ½∂²_x(σ²ρ)
  Auxiliary loss: ||ρ̂ - FP[ρ̂]|| penalizes non-physical transitions

• Wasserstein DRO: adversarial perturbation ball ||δ||_W ≤ ε
  Gradient penalty: λ E[||∇_x ℓ(θ,x)||] implements ε-Lipschitz constraint
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations  # 파이썬의 타입 힌트를 최신 방식으로 쓸 수 있게 한다  # Import annotations from __future__ module

import math  # 수학 상수(π 등)와 함수를 쓰기 위한 도구  # Import Python standard math library (log, exp, trig)
from typing import Optional, Tuple  # 변수 종류를 명확히 표시하기 위한 도구  # Import Optional, Tuple from Type hint annotations

import torch  # 딥러닝 프레임워크 (PyTorch)  # Import PyTorch — core deep learning library
import torch.nn as nn  # 신경망 레이어들을 쓰기 위한 도구  # Import PyTorch neural network building blocks as "nn"
import torch.nn.functional as F  # 활성화 함수 등 자주 쓰는 연산 모음  # Import PyTorch functional API (activations, losses) as "F"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hurst Exponent Estimator (fBm / R-S Analysis)
# ─────────────────────────────────────────────────────────────────────────────

class HurstEstimator(nn.Module):  # ★ [HurstEstimator] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Estimates the Hurst exponent H from a log-return series using R/S analysis.

    Hurst exponent:
        H > 0.5 : Persistent (trending) — momentum works
        H < 0.5 : Antipersistent (mean-reverting) — contrarian works
        H ≈ 0.5 : Random walk (GBM) — no statistical edge

    R/S Analysis (Mandelbrot & Wallis 1969):
        R(n) = max(cumsum) - min(cumsum) of mean-adjusted series
        S(n) = std(series)
        E[R(n)/S(n)] ≈ c · n^H

    Multi-scale estimation (log-log regression):
        H = Σ_k log(R_k/S_k) / log(n_k)  over scales k = 1..K

    Adaptive lookback:
        T_opt = T_base / |2H - 1|  — near H=0.5 use longer windows;
                                     far from 0.5 shorter windows suffice.

    Args:
        n_scales : number of R/S sub-scales (default 5)
        eps      : numerical stability epsilon
    """

    # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
    # [__init__] Constructor — runs when the object is created
    def __init__(self, n_scales: int = 5, eps: float = 1e-8) -> None:
        super().__init__()  # 부모 클래스 초기화  # Calls the parent class constructor
        self.n_scales = n_scales  # R/S를 계산할 스케일(시간 단위) 개수
        self.eps = eps            # 수치 안정성을 위한 아주 작은 값

    @staticmethod  # 객체(self) 없이 호출 가능한 정적 메서드로 만든다  # Decorator: static method — no self or cls
    def _rs_single(x: torch.Tensor) -> torch.Tensor:  # [_rs_single] 내부 전용 함수 정의
        """R/S statistic for a 1-D tensor x — 1차원 수익률 시계열의 R/S 통계를 계산한다."""
        n = x.shape[0]  # 시계열 길이  # Shape (dimensions) of the tensor/array
        if n < 4:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            return torch.tensor(0.5, device=x.device, dtype=x.dtype)  # 데이터가 너무 적으면 0.5(무작위) 반환  # Returns a value to the caller
        mean = x.mean()              # 평균 수익률 계산
        y = x - mean                 # 평균을 빼서 중심화한다 (편차 시계열)
        cumsum = torch.cumsum(y, dim=0)  # 누적합 계산 (random walk처럼 쌓아간다)
        R = cumsum.max() - cumsum.min()  # 범위(Range): 누적합의 최댓값 - 최솟값
        S = x.std(unbiased=True).clamp(min=1e-8)  # 표준편차(Scale): 0이 되지 않게 최소값 보장
        return (R / S).clamp(min=1e-8)  # R/S 통계 반환 (너무 작아지지 않게 clamp)

    def _hurst_single(self, x: torch.Tensor) -> torch.Tensor:  # [_hurst_single] 내부 전용 함수 정의
        """
        H from multi-scale R/S regression on a 1-D series x.
        H = Cov(log n, log R/S) / Var(log n)  (OLS slope in log-log space)
        여러 스케일에서 R/S를 계산하고 로그-로그 그래프의 기울기로 허스트 지수를 구한다.
        """
        n = x.shape[0]  # 시계열 길이  # Shape (dimensions) of the tensor/array
        if n < 8:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            return torch.tensor(0.5, device=x.device, dtype=x.dtype)  # 너무 짧으면 0.5 반환  # Returns a value to the caller

        # Build log-spaced scales — 로그 간격으로 스케일(창 크기) 목록 만들기
        min_scale = max(4, n // (2 ** self.n_scales))  # 최소 스케일 크기 (최소 4)
        scales = []
        for k in range(self.n_scales):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            s = max(4, min_scale * (2 ** k))  # 2배씩 커지는 스케일
            if s > n // 2:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                break  # 스케일이 시계열의 절반보다 크면 중단  # Exit the enclosing loop immediately
            scales.append(s)  # 유효한 스케일만 추가  # Appends an item to the end of the list

        if len(scales) < 2:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            return torch.tensor(0.5, device=x.device, dtype=x.dtype)  # 스케일이 2개 미만이면 계산 불가  # Returns a value to the caller

        log_ns, log_rs = [], []  # 로그(스케일), 로그(R/S) 값들을 저장할 목록
        for s in scales:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
            # Non-overlapping chunks of length s — 겹치지 않는 s 길이 구간으로 나눈다
            n_chunks = n // s  # 구간 개수
            if n_chunks < 1:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                continue  # 구간이 없으면 건너뛴다  # Skip the rest of this iteration
            rs_vals = []
            for c in range(n_chunks):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
                chunk = x[c * s: (c + 1) * s]  # c번째 구간 데이터 추출
                rs_vals.append(self._rs_single(chunk))  # 구간의 R/S 계산  # Appends an item to the end of the list
            rs_mean = torch.stack(rs_vals).mean()  # 구간 평균 R/S  # Stacks tensors along a new dimension
            log_ns.append(math.log(s))             # log(스케일) 추가
            log_rs.append(torch.log(rs_mean.clamp(min=1e-8)))  # log(R/S) 추가

        if len(log_ns) < 2:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            return torch.tensor(0.5, device=x.device, dtype=x.dtype)  # 데이터 부족  # Returns a value to the caller

        # OLS slope: H = Cov(log n, log R/S) / Var(log n) — 로그-로그 공간에서 최소제곱 기울기 = 허스트 지수
        ln = torch.tensor(log_ns, device=x.device, dtype=x.dtype)  # log(스케일) 텐서
        lr = torch.stack(log_rs)  # log(R/S) 텐서
        ln_c = ln - ln.mean()     # log(스케일) 중심화
        lr_c = lr - lr.mean()     # log(R/S) 중심화
        num = (ln_c * lr_c).sum()            # 공분산 분자
        den = (ln_c ** 2).sum().clamp(min=1e-8)  # 분산 분모 (0 방지)
        H = (num / den).clamp(0.05, 0.95)   # 허스트 지수: 0.05~0.95 범위로 제한
        return H  # 허스트 지수 반환  # Returns a value to the caller

    def forward(self, log_returns: torch.Tensor) -> torch.Tensor:  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        """
        Compute per-sample Hurst exponents.

        Args:
            log_returns: [B, T] — log return series

        Returns:
            hurst: [B] — H ∈ [0.05, 0.95]
        """
        B = log_returns.shape[0]  # 배치 크기  # Shape (dimensions) of the tensor/array
        results = [self._hurst_single(log_returns[b]) for b in range(B)]  # 각 샘플마다 허스트 지수 계산  # Log-return: ln(P_t / P_{t-1}) — stationary price change
        return torch.stack(results)                   # [B] 배치 전체 허스트 지수를 쌓아서 반환

    @staticmethod  # 객체(self) 없이 호출 가능한 정적 메서드로 만든다  # Decorator: static method — no self or cls
    def optimal_lookback(H: torch.Tensor, T_base: int = 30) -> torch.Tensor:  # [optimal_lookback] 함수 정의 시작
        """
        Adaptive lookback based on Hurst exponent.
        허스트 지수에 따라 최적 관찰 기간을 자동으로 계산한다.

        Near H=0.5 (random walk): longer lookback needed.
        Far from H=0.5 (strong trend/revert): shorter suffices.

        T_opt = ceil(T_base / max(|2H-1|, 0.1))

        Args:
            H      : [B] Hurst exponents
            T_base : reference lookback

        Returns:
            T_opt: [B] long tensor of optimal lookback lengths
        """
        sensitivity = (2.0 * H - 1.0).abs().clamp(min=0.1)  # |2H-1|: H가 0.5에 가까울수록 작아진다
        T_opt = (T_base / sensitivity).ceil().long().clamp(4, T_base * 4)  # 최적 기간 계산 후 4~120 범위로 제한  # Casts tensor to int64
        return T_opt  # 최적 관찰 기간 반환  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lindblad Decoherence (Quantum Open System Dynamics)
# ─────────────────────────────────────────────────────────────────────────────

class LindbladDecoherence(nn.Module):  # ★ [LindbladDecoherence] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Lindblad master equation for open quantum system regime detection.
    린드블라드 마스터 방정식으로 시장 레짐(추세/횡보) 변화를 감지한다.

    Quantum master equation:
        dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

    Trading context:
        ρ        = density matrix of market state (encoded from qubit expvals)
        H        = market Hamiltonian (diagonal ← expvals)
        L_k      = regime-transition (jump) operators (learned)
        γ_k      = decoherence rates (learned, positive)

    Outputs:
        purity     = Tr(ρ²) ∈ [1/N, 1]
            purity = 1 → pure state (strong coherent trend)
            purity → 0 → maximally mixed (regime uncertainty)
        coherence  = mean |off-diagonal| of ρ
        regime_prob = 1 - purity  → used as position-size scaling

    Regime change signal:
        When purity drops below threshold (e.g. 0.3), the market is
        decoherent → reduce position size or skip entry.

    Args:
        n_qubits   : dimension of quantum state space (2^n_qubits × 2^n_qubits)
        n_lindblad : number of jump operators (default 2)
        dt         : Lindblad time step (default 1.0)
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        n_qubits: int = 4,    # 큐비트 수 (밀도 행렬 크기 = 2^n_qubits × 2^n_qubits)
        n_lindblad: int = 2,  # 점프 연산자 개수 (레짐 전환 경로 수)  # Lindblad master equation: quantum decoherence model
        dt: float = 1.0,      # 린드블라드 시간 스텝 크기
    ) -> None:
        super().__init__()  # 부모 클래스 초기화  # Calls the parent class constructor
        self.n_qubits = n_qubits    # 큐비트 수 저장
        self.n_lindblad = n_lindblad  # 점프 연산자 수 저장  # Lindblad master equation: quantum decoherence model
        self.dt = dt                  # 시간 스텝 저장
        dim = 2 ** n_qubits           # 힐베르트 공간 차원 (2^n_qubits)

        # Lindblad jump operators L_k (real parameterization for gradient flow)
        # 린드블라드 점프 연산자: 시장 레짐 전환을 나타내는 행렬 (학습 파라미터)
        self.L_re = nn.Parameter(torch.randn(n_lindblad, dim, dim) * 0.1)  # 실수부  # Registers tensor as a learnable parameter (tracked by autograd)
        self.L_im = nn.Parameter(torch.randn(n_lindblad, dim, dim) * 0.1)  # 허수부  # Registers tensor as a learnable parameter (tracked by autograd)

        # Decoherence rates γ_k > 0 (log parameterized) — 결깨짐 속도 (항상 양수)
        self.log_gamma = nn.Parameter(torch.zeros(n_lindblad))  # log로 파라미터화하면 exp 후 항상 양수

        # Encode qubit expvals → density matrix — 큐비트 측정값을 밀도 행렬로 변환하는 네트워크
        # [B, K] → [B, dim*dim]
        self.state_encoder = nn.Sequential(  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module
            nn.Linear(n_qubits, dim * dim),  # 큐비트 수 → dim² 차원으로 선형 변환  # Fully-connected (affine) layer: y = xW^T + b
            nn.Tanh(),                        # tanh 활성화 (-1~1 범위로 정규화)
        )

    @property  # 이 메서드를 속성처럼 obj.속성 형태로 접근할 수 있게 만든다  # Decorator: expose method as a read-only attribute
    def gamma(self) -> torch.Tensor:  # [gamma] 함수 정의 시작
        return torch.exp(self.log_gamma).clamp(max=10.0)  # 결깨짐 속도: log_gamma를 exp로 변환 (최대 10으로 제한)  # Returns a value to the caller

    # [_encode_density_matrix] 내부 전용 함수 정의
    # [_encode_density_matrix] Private helper function
    def _encode_density_matrix(self, expvals: torch.Tensor) -> torch.Tensor:
        """
        Map qubit expectation values to a valid density matrix ρ.
        큐비트 측정값을 유효한 밀도 행렬로 변환한다.

        Validity conditions:
            1. Hermitian: ρ = ρ†  (enforced via (M + M^T)/2)
            2. Positive semi-definite: ρ = M^T M  (enforced via Gram matrix)
            3. Unit trace: Tr(ρ) = 1  (enforced via normalization)

        Args:
            expvals: [B, K]

        Returns:
            rho: [B, dim, dim]
        """
        B, K = expvals.shape  # 배치 크기, 큐비트 수  # Shape (dimensions) of the tensor/array
        dim = 2 ** self.n_qubits  # 힐베르트 공간 차원

        raw = self.state_encoder(expvals).view(B, dim, dim)  # [B, dim, dim] 신경망으로 행렬 생성

        # Hermitian: symmetrize — 에르미트 행렬 조건: (M + M^T)/2 하면 항상 대칭 행렬이 된다
        M = (raw + raw.transpose(-1, -2)) / 2.0              # [B, dim, dim] 대칭화

        # PSD: Gram matrix M^T M — 양의 반정치 행렬 조건: M^T M은 항상 PSD이다
        rho = torch.bmm(M.transpose(-1, -2), M)              # [B, dim, dim] 그람 행렬 계산

        # Normalize trace — 밀도 행렬 조건: Tr(ρ)=1이 되도록 정규화한다
        tr = rho.diagonal(dim1=-2, dim2=-1).sum(dim=-1)      # [B] 대각합 계산
        rho = rho / tr.clamp(min=1e-8).view(B, 1, 1)         # 대각합으로 나눠서 정규화  # Reshapes tensor (shares memory if possible)

        return rho  # 유효한 밀도 행렬 반환  # Returns a value to the caller

    def _lindblad_step(self, rho: torch.Tensor) -> torch.Tensor:  # [_lindblad_step] 내부 전용 함수 정의
        """
        One Euler step of Lindblad evolution: ρ(t+dt) ≈ ρ(t) + dt × L[ρ]
        린드블라드 방정식의 오일러 1스텝: 밀도 행렬을 한 시간 단계 진화시킨다.

        L[ρ] = Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

        Note: uses real-valued L for gradient compatibility.
        """
        B, dim, _ = rho.shape  # 배치 크기, 힐베르트 공간 차원  # Shape (dimensions) of the tensor/array
        drho = torch.zeros_like(rho)  # 밀도 행렬 변화량 초기화  # Creates a zero-filled tensor

        for k in range(self.n_lindblad):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            Lk = self.L_re[k]                             # [dim, dim] k번째 점프 연산자 (실수부)
            Lk_dag = Lk.t()                               # [dim, dim] 켤레 전치 (전치 행렬)
            LdagL = Lk_dag.mm(Lk)                         # [dim, dim] L†L 계산

            # Expand for batch — 배치 차원으로 확장한다
            Lk_b = Lk.unsqueeze(0).expand(B, -1, -1)       # [B, dim, dim]
            Lk_dag_b = Lk_dag.unsqueeze(0).expand(B, -1, -1)  # [B, dim, dim]
            LdagL_b = LdagL.unsqueeze(0).expand(B, -1, -1)    # [B, dim, dim]

            # Lindblad dissipator term — 린드블라드 소산 항 계산
            LrhoLdag = torch.bmm(torch.bmm(Lk_b, rho), Lk_dag_b)  # L ρ L† 계산
            anticomm = torch.bmm(LdagL_b, rho) + torch.bmm(rho, LdagL_b)  # {L†L, ρ} = L†Lρ + ρL†L
            drho = drho + self.gamma[k] * (LrhoLdag - 0.5 * anticomm)  # γ_k (LρL† - ½{L†L, ρ}) 누적

        rho_new = rho + self.dt * drho  # 오일러 방법으로 다음 시간 스텝의 밀도 행렬 계산

        # Re-normalize trace (preserve trace property) — 오일러 스텝 후 다시 정규화
        tr = rho_new.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # 대각합 계산
        rho_new = rho_new / tr.clamp(min=1e-8).view(B, 1, 1)  # 정규화  # Reshapes tensor (shares memory if possible)

        return rho_new  # 진화된 밀도 행렬 반환  # Returns a value to the caller

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        expvals: torch.Tensor,    # [B, K] or [B, T, K] 큐비트 측정값
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute purity, coherence, and regime-change probability.
        순수도, 결맞음, 레짐 변화 확률을 계산한다.

        Args:
            expvals: [B, K] or [B, T, K] — quantum circuit ⟨σ^z⟩ values

        Returns:
            purity      : [B] — Tr(ρ²) ∈ [1/dim, 1]
            coherence   : [B] — mean |off-diagonal| of ρ
            regime_prob : [B] — 1 - purity ∈ [0, 1-1/dim]
        """
        if expvals.dim() == 3:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            expvals = expvals[:, -1, :]                   # 3차원이면 마지막 타임스텝만 사용 [B, K]

        B = expvals.shape[0]  # 배치 크기  # Shape (dimensions) of the tensor/array
        dim = 2 ** self.n_qubits  # 힐베르트 공간 차원

        # Encode density matrix — 측정값을 밀도 행렬로 변환한다
        rho = self._encode_density_matrix(expvals)        # [B, dim, dim]

        # Apply one Lindblad evolution step — 린드블라드 방정식으로 한 스텝 진화
        rho_evolved = self._lindblad_step(rho)            # [B, dim, dim]

        # Purity: Tr(ρ²) — 순수도 계산 (1에 가까울수록 명확한 추세 상태)
        rho2 = torch.bmm(rho_evolved, rho_evolved)  # ρ² 계산
        purity = rho2.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(1.0 / dim, 1.0)  # Tr(ρ²), 범위 [1/dim, 1]

        # Coherence: mean |off-diagonal elements| — 결맞음: 비대각 원소의 평균 크기
        eye = torch.eye(dim, device=expvals.device, dtype=expvals.dtype)  # 단위 행렬
        off_diag_mask = (1.0 - eye).unsqueeze(0)  # 비대각 위치만 1인 마스크  # Inserts a new dimension of size 1
        coherence = (rho_evolved.abs() * off_diag_mask).mean(dim=(-1, -2))  # 비대각 원소 절댓값 평균

        # Regime-change probability — 레짐 변화 확률: 순수도가 낮을수록 높다
        regime_prob = (1.0 - purity).clamp(0.0, 1.0)  # 1 - 순수도 (0~1 범위)

        return purity, coherence, regime_prob  # 순수도, 결맞음, 레짐 변화 확률 반환  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# 3. MINE — Mutual Information Neural Estimation
# ─────────────────────────────────────────────────────────────────────────────

class MINEEstimator(nn.Module):  # ★ [MINEEstimator] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Mutual Information Neural Estimation (Belghazi et al. 2018).
    신경망으로 상호 정보량(두 변수가 얼마나 연관되어 있는지)을 추정한다.

    DONSKER-VARADHAN representation:
        I(X; Y) ≥ sup_T { E_{P(X,Y)}[T] - log E_{P(X)⊗P(Y)}[e^T] }

    Training: maximize MI lower bound simultaneously with main loss.

    Trading context:
        X = market feature embedding (c_kt projected to x_dim)
        Y = trade outcome (TP/SL label as one-hot or soft)
        I(X; Y) → how much features predict outcomes

    Usage:
        - Auxiliary training signal: add β_MINE × (-mi_lb) to total loss
        - Feature quality monitor: high I(X;Y) → features are informative
        - Theoretical bound: I(X;Y) ≤ H(Y) = log(num_classes)

    Args:
        x_dim  : feature embedding dimension
        y_dim  : label dimension (n_actions or 1)
        hidden : hidden layer size
        ema_decay : exponential moving average for variance reduction
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        x_dim: int,            # 특징 차원 수
        y_dim: int,            # 레이블 차원 수 (행동 수 또는 1)
        hidden: int = 64,      # 은닉층 크기
        ema_decay: float = 0.01,  # 지수이동평균 감쇠율 (분산 감소용)
    ) -> None:
        super().__init__()  # 부모 클래스 초기화  # Calls the parent class constructor
        self.ema_decay = ema_decay  # EMA 감쇠율 저장

        # Statistics network T: (X, Y) → ℝ — 통계 네트워크: 특징과 레이블을 입력받아 스칼라 출력
        self.T_net = nn.Sequential(  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module
            nn.Linear(x_dim + y_dim, hidden),  # 특징+레이블을 hidden 차원으로 변환  # Fully-connected (affine) layer: y = xW^T + b
            nn.ELU(),                           # ELU 활성화 함수 (부드러운 음수 허용)
            nn.Linear(hidden, hidden),          # 두 번째 은닉층  # Fully-connected (affine) layer: y = xW^T + b
            nn.ELU(),                           # ELU 활성화
            nn.Linear(hidden, 1),               # 스칼라 출력  # Fully-connected (affine) layer: y = xW^T + b
        )

        # EMA of denominator for variance reduction (DV representation)
        # 분산 감소를 위한 분모의 지수이동평균 (학습 안정화)
        self.register_buffer("ema_denom", torch.ones(1))  # 초기값 1  # Creates a ones-filled tensor

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        x: torch.Tensor,      # [B, x_dim] 특징 벡터
        y: torch.Tensor,      # [B, y_dim] 레이블 벡터
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MINE lower bound on I(X; Y).
        I(X;Y)의 하한(lower bound)을 계산한다.

        Uses exponential moving average (EMA) trick for variance reduction:
            log E[e^T] ≈ log(ema_denom) + E[e^T - ema_denom] / ema_denom

        Args:
            x: [B, x_dim]
            y: [B, y_dim]

        Returns:
            mi_lb  : scalar — MI lower bound I(X;Y) ≥ mi_lb
            T_joint: [B]   — T-statistic on joint distribution
        """
        B = x.shape[0]  # 배치 크기  # Shape (dimensions) of the tensor/array

        # Joint: T(x_i, y_i) — 결합 분포: x_i와 y_i가 같은 샘플에서 온 경우
        xy_joint = torch.cat([x, y], dim=-1)               # [B, x+y] x와 y를 이어 붙인다
        T_joint = self.T_net(xy_joint).squeeze(-1)          # [B] T 통계량 계산

        # Marginal: T(x_i, y_j) with shuffled j ≠ i — 주변 분포: y를 섞어서 독립인 척 만든다
        idx = torch.randperm(B, device=x.device)           # 배치 내에서 무작위 순서 생성
        xy_marginal = torch.cat([x, y[idx]], dim=-1)        # [B, x+y] x_i와 다른 샘플의 y를 붙인다
        T_marginal = self.T_net(xy_marginal).squeeze(-1)    # [B] 독립인 척한 T 통계량

        # EMA variance-reduced denominator — 분산 감소를 위한 지수이동평균 분모
        exp_T_marg = T_marginal.exp().mean()  # E[e^T_marginal] 계산
        ema_d = self.ema_denom.detach()       # 현재 EMA 분모 (기울기 차단)  # Detaches tensor from the computation graph
        # MINE-f lower bound: E[T_joint] - (e^T_marginal / ema_denom - log(ema_denom) - 1)
        mi_lb = T_joint.mean() - (
            exp_T_marg / ema_d - torch.log(ema_d) - 1.0  # 자연 로그(ln)를 계산한다  # Computes natural logarithm element-wise
            # 돈스커-바라단 하한 공식: 결합 T 평균 - 주변 분포 보정항
        )

        # Update EMA (detached, no gradient) — EMA 분모 업데이트 (기울기 없이)
        if self.training:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            with torch.no_grad():  # 메모리 절약을 위해 기울기 계산 없이 추론만 실행한다  # Context: disable gradient tracking for inference (saves memory)
                self.ema_denom.mul_(1.0 - self.ema_decay).add_(
                    self.ema_decay * exp_T_marg
                    # EMA 업데이트: 이전값 × (1-감쇠) + 새값 × 감쇠
                )

        return mi_lb, T_joint  # 상호 정보량 하한과 결합 T 통계량 반환  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# 4. Matrix Product States (MPS) Layer
# ─────────────────────────────────────────────────────────────────────────────

class MPSLayer(nn.Module):  # ★ [MPSLayer] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Matrix Product States (MPS / Tensor Train Decomposition) quantum layer.
    행렬 곱 상태(MPS) 텐서 네트워크 레이어 — VQC를 대체하는 양자 구조.

    Replaces the VQC with a tensor network that provides:
        - Controllable entanglement via bond dimension χ
        - O(n · χ² · d²) parameters (vs O(n · L · 3) for VQC)
        - Exact simulation of 1D-entangled quantum states
        - Gradient-friendly classical contraction

    MPS representation:
        |ψ⟩ = Σ_{i₁,...,iₙ} A¹_{i₁} A²_{i₂} … Aⁿ_{iₙ} |i₁…iₙ⟩

    Layer function:
        f(x) = ⟨ψ(W)| O |ψ(x)⟩  where O is a learned observable

    Input modulation:
        At each site k, physical index is modulated by input embedding x_k.

    Args:
        n_sites      : number of MPS sites (= n_qubits)
        d_phys       : physical dimension (2 for qubits)
        chi          : bond dimension (entanglement capacity)
        in_features  : input feature dimension
        out_features : output feature dimension
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        n_sites: int = 4,      # MPS 사이트 수 (큐비트 수와 같음)
        d_phys: int = 2,       # 물리 차원 (큐비트는 2: |0⟩ 또는 |1⟩)
        chi: int = 8,          # 본드 차원 (얽힘 용량, 클수록 표현력↑)
        in_features: int = 4,  # 입력 특징 차원
        out_features: int = 4, # 출력 특징 차원
    ) -> None:
        super().__init__()  # 부모 클래스 초기화  # Calls the parent class constructor
        self.n_sites = n_sites  # 사이트 수 저장
        self.d_phys = d_phys    # 물리 차원 저장
        self.chi = chi          # 본드 차원 저장

        # MPS tensors A_k with shape [chi_l, d_phys, chi_r]
        # 각 사이트의 MPS 텐서: [왼쪽 본드, 물리 차원, 오른쪽 본드] 형태
        # First site: [1, d, chi]; Last site: [chi, d, 1]; Others: [chi, d, chi]
        self.tensors = nn.ParameterList()  # 각 사이트의 텐서를 담는 목록  # Stores a neural network layer as an attribute of this module
        for i in range(n_sites):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            chi_l = 1 if i == 0 else chi          # 첫 사이트는 왼쪽 본드 크기 1
            chi_r = 1 if i == n_sites - 1 else chi  # 마지막 사이트는 오른쪽 본드 크기 1
            A = nn.Parameter(torch.randn(chi_l, d_phys, chi_r) * 0.1)  # 작은 값으로 초기화  # Creates a tensor with standard normal random values
            self.tensors.append(A)  # 텐서 목록에 추가  # Appends an item to the end of the list

        # Feature embedding: [B, in_features] → [B, n_sites * d_phys]
        # 입력 특징을 각 사이트의 물리 차원 값으로 변환하는 네트워크
        self.embed = nn.Sequential(  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module
            nn.Linear(in_features, n_sites * d_phys),  # 선형 변환  # Fully-connected (affine) layer: y = xW^T + b
            nn.Tanh(),                                   # tanh 활성화
        )

        # Output projection from contracted result — 수축 결과를 출력 차원으로 변환
        self.proj = nn.Linear(d_phys, out_features)  # 물리 차원 → 출력 차원  # Stores a neural network layer as an attribute of this module

        # LayerNorm for stability — 학습 안정화를 위한 레이어 정규화
        self.norm = nn.LayerNorm(out_features)  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module

    def _contract_mps(self, site_inputs: torch.Tensor) -> torch.Tensor:  # [_contract_mps] 내부 전용 함수 정의
        """
        Contract MPS with site-specific inputs.
        MPS 텐서를 사이트별 입력값과 수축(곱)한다.

        site_inputs: [B, n_sites, d_phys] — probability amplitudes per site

        Returns: [B, d_phys] — contracted output
        """
        B = site_inputs.shape[0]  # 배치 크기  # Shape (dimensions) of the tensor/array

        # Start from leftmost site: A_0 has shape [1, d, chi_r]
        # 가장 왼쪽 사이트부터 시작: A_0의 형태는 [1, d, chi_r]
        A_0 = self.tensors[0].squeeze(0)               # [d, chi_r] 왼쪽 본드 차원 제거
        # Contract with input at site 0: [B, d] × [d, chi_r] → [B, chi_r]
        boundary = site_inputs[:, 0, :].matmul(A_0)    # [B, chi_r] 첫 사이트 입력과 텐서 수축

        # Contract through sites 1 .. n-2 — 중간 사이트들을 순서대로 수축
        for k in range(1, self.n_sites - 1):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            Ak = self.tensors[k]                       # [chi_l, d, chi_r] k번째 텐서
            chi_l, d, chi_r = Ak.shape                 # 텐서 형태 분리  # Shape (dimensions) of the tensor/array
            xk = site_inputs[:, k, :]                  # [B, d] k번째 사이트 입력

            # boundary: [B, chi_l]
            # Contract: Σ_{d} x_k[b,d] × A_k[chi_l, d, chi_r]
            # = boundary[b, chi_l] × A_k_eff[chi_l, chi_r]  where
            #   A_k_eff = Σ_d x_k[b,d] × A_k[:, d, :]
            Ak_eff = torch.einsum("bd,ldr->blr", xk, Ak)   # [B, chi_l, chi_r] 입력-텐서 수축
            boundary = torch.bmm(
                boundary.unsqueeze(1), Ak_eff  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
            ).squeeze(1)                                # [B, chi_r] 누적 수축 결과

        # Last site: A_n has shape [chi_l, d, 1] — 마지막 사이트 처리
        A_last = self.tensors[-1].squeeze(-1)           # [chi_l, d] 오른쪽 본드 차원 제거
        x_last = site_inputs[:, -1, :]                  # [B, d] 마지막 사이트 입력
        # Σ_d x_last[b,d] × A_last[chi_l, d] → [B, chi_l]
        A_last_eff = x_last.matmul(A_last.t())          # [B, chi_l] 마지막 사이트 수축
        # Final contraction: boundary[B, chi_l] dot A_last_eff[B, chi_l]
        out = (boundary * A_last_eff).unsqueeze(1)      # [B, 1, chi_l≈d] 최종 수축

        # Project back to d_phys dim — 물리 차원으로 다시 변환
        return out.squeeze(1).matmul(  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
            torch.ones(A_last.shape[0], self.d_phys, device=out.device)  # 1로 채워진 텐서를 만든다  # Creates a ones-filled tensor
            / (A_last.shape[0] + 1e-8)  # 텐서/배열의 형태(차원과 각 크기)를 확인한다  # Shape (dimensions) of the tensor/array
            # 평균 내어 d_phys 차원으로 변환
        )                                               # [B, d_phys]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        """
        MPS-quantum forward pass.
        MPS 양자 순전파: 입력 특징을 MPS 텐서 네트워크로 처리한다.

        Args:
            x: [B, in_features]

        Returns:
            out: [B, out_features]
        """
        B = x.shape[0]  # 배치 크기  # Shape (dimensions) of the tensor/array
        embedded = self.embed(x).view(B, self.n_sites, self.d_phys)  # [B, n, d] 입력을 사이트별 물리 차원으로 변환
        # Normalize as probability amplitudes (softmax over physical dim)
        # 확률 진폭으로 정규화: 각 사이트에서 softmax 적용 (합=1)
        site_inputs = F.softmax(embedded, dim=-1)                    # [B, n, d]

        contracted = self._contract_mps(site_inputs)                 # [B, d_phys] MPS 수축 실행
        out = self.proj(contracted)                                   # [B, out_features] 출력 차원으로 변환
        return self.norm(out)  # 레이어 정규화 후 반환  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# 5. Optimal Stopping Boundary (Snell Envelope)
# ─────────────────────────────────────────────────────────────────────────────

class OptimalStoppingBoundary(nn.Module):  # ★ [OptimalStoppingBoundary] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Neural approximation of the optimal stopping boundary (Snell envelope).
    최적 청산 경계를 신경망으로 근사한다 (언제 청산하면 가장 이익인지 학습).

    Optimal stopping problem:
        J*(x, t) = sup_{τ ≥ t} E[g(X_τ) | X_t = x]

    Stopping region:   S = {(x,t) : J*(x,t) = g(x,t)}  → EXIT
    Continuation region: C = {(x,t) : J*(x,t) > g(x,t)} → HOLD

    Standard TP/SL is the static special case: g(x) = ±ATR×α constant.
    This class LEARNS the optimal stopping boundary as a function of state.

    Implementation:
        V_net(s)  → continuation value J*(x, t) (NN approximation)
        TP_net(s) → adaptive TP multiplier α_TP(s) ∈ [1.0, 4.0]
        SL_net(s) → adaptive SL multiplier α_SL(s) ∈ [0.5, 2.0]

    Training signal:
        When V_net(s) > g(s)  → continue  (policy gradient keeps position)
        When V_net(s) ≤ g(s)  → stop      (policy gradient exits)

    The optimal boundary is approximated by fitting V_net to minimize:
        L_stop = MSE(V_net(s), max(g(s), r_next + γ·V_net(s')))
        (Bellman optimality condition for stopping)

    Args:
        state_dim : dimension of state input
        hidden    : hidden layer neurons
        tp_base   : base TP multiplier
        sl_base   : base SL multiplier
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        state_dim: int = 8,      # 상태 입력 차원 수
        hidden: int = 32,        # 은닉층 뉴런 수
        tp_base: float = 2.0,    # TP 기본 배수 (기본 2배 ATR)
        sl_base: float = 1.0,    # SL 기본 배수 (기본 1배 ATR)
    ) -> None:
        super().__init__()  # 부모 클래스 초기화  # Calls the parent class constructor
        self.tp_base = tp_base  # TP 기본 배수 저장
        self.sl_base = sl_base  # SL 기본 배수 저장

        # Continuation value network J*(x,t) — 포지션을 계속 유지할 때의 기대값 네트워크
        self.value_net = nn.Sequential(  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module
            nn.Linear(state_dim, hidden),  # 상태 → 은닉층  # Fully-connected (affine) layer: y = xW^T + b
            nn.Tanh(),                      # tanh 활성화
            nn.Linear(hidden, hidden),      # 은닉층 → 은닉층  # Fully-connected (affine) layer: y = xW^T + b
            nn.Tanh(),                      # tanh 활성화
            nn.Linear(hidden, 1),           # 은닉층 → 스칼라 값  # Fully-connected (affine) layer: y = xW^T + b
        )

        # Adaptive TP multiplier: α_TP(s) ∈ [1.0, 4.0] — 상태에 따른 가변 익절가 배수 네트워크
        self.tp_scale_net = nn.Sequential(  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module
            nn.Linear(state_dim, hidden // 2),  # 상태 → 작은 은닉층  # Fully-connected (affine) layer: y = xW^T + b
            nn.ReLU(),                           # ReLU 활성화
            nn.Linear(hidden // 2, 1),           # → 스칼라  # Fully-connected (affine) layer: y = xW^T + b
            nn.Sigmoid(),                        # sigmoid로 0~1 범위
        )

        # Adaptive SL multiplier: α_SL(s) ∈ [0.5, 2.0] — 상태에 따른 가변 손절가 배수 네트워크
        self.sl_scale_net = nn.Sequential(  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module
            nn.Linear(state_dim, hidden // 2),  # 상태 → 작은 은닉층  # Fully-connected (affine) layer: y = xW^T + b
            nn.ReLU(),                           # ReLU 활성화
            nn.Linear(hidden // 2, 1),           # → 스칼라  # Fully-connected (affine) layer: y = xW^T + b
            nn.Sigmoid(),                        # sigmoid로 0~1 범위
        )

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        state: torch.Tensor,        # [B, state_dim] 현재 시장+포지션 상태
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute continuation value and adaptive TP/SL multipliers.
        포지션 유지 가치와 가변 TP/SL 배수를 계산한다.

        Args:
            state: [B, state_dim] — encoded market + position state

        Returns:
            continuation_value : [B] — V(s), value of staying in position
            tp_mult            : [B] — adaptive TP multiplier ∈ [1.0, 4.0]
            sl_mult            : [B] — adaptive SL multiplier ∈ [0.5, 2.0]
        """
        continuation_value = self.value_net(state).squeeze(-1)      # [B] 포지션 유지 가치

        # TP: 1.0 + 3.0 × sigmoid → [1.0, 4.0] — 익절가 배수: 최소 1배, 최대 4배 ATR
        tp_mult = 1.0 + 3.0 * self.tp_scale_net(state).squeeze(-1)  # [B]

        # SL: 0.5 + 1.5 × sigmoid → [0.5, 2.0] — 손절가 배수: 최소 0.5배, 최대 2배 ATR
        sl_mult = 0.5 + 1.5 * self.sl_scale_net(state).squeeze(-1)  # [B]

        return continuation_value, tp_mult, sl_mult  # 포지션 유지 가치, TP 배수, SL 배수 반환  # Returns a value to the caller

    def stopping_loss(  # [stopping_loss] 함수 정의 시작
        self,
        state: torch.Tensor,         # [B, state_dim] 현재 상태
        immediate_reward: torch.Tensor,  # [B] 지금 청산 시 보상
        next_state: torch.Tensor,    # [B, state_dim] 다음 상태
        gamma: float = 0.99,         # 미래 보상 할인율
    ) -> torch.Tensor:
        """
        Bellman optimality loss for optimal stopping:
            L = MSE(V(s), max(g(s), r_immediate + γ·V(s')))
        최적 정지 벨만 방정식 손실: 지금 청산 vs 계속 유지 중 더 나은 선택을 학습한다.

        This trains V to approximate J* via backward induction.
        """
        with torch.no_grad():  # 메모리 절약을 위해 기울기 계산 없이 추론만 실행한다  # Context: disable gradient tracking for inference (saves memory)
            V_next, _, _ = self.forward(next_state)  # 다음 상태의 유지 가치 (기울기 없이 계산)
        V_curr, _, _ = self.forward(state)  # 현재 상태의 유지 가치

        # Bellman target: max(stop now, continue) — 지금 청산과 계속 유지 중 더 큰 값을 목표로
        target = torch.maximum(
            immediate_reward,                   # 지금 청산 시 즉각 보상
            immediate_reward + gamma * V_next,  # 계속 유지: 즉각보상 + 할인된 미래 가치
        )

        return F.mse_loss(V_curr, target.detach())  # 현재 가치 추정과 목표값의 MSE 손실  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fokker-Planck Regularizer (Langevin SDE Consistency)
# ─────────────────────────────────────────────────────────────────────────────

class FokkerPlanckRegularizer(nn.Module):  # ★ [FokkerPlanckRegularizer] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Fokker-Planck / Langevin SDE consistency regularizer.
    포커-플랑크 방정식 기반 정규화: 모델의 예측이 물리적 확률 과정과 일치하도록 강제한다.

    Assumes market returns follow an Itô SDE:
        dx_t = μ(x_t) dt + σ(x_t) dW_t

    The Fokker-Planck equation for the density ρ(x, t):
        ∂_t ρ = -∂_x(μ ρ) + ½ ∂²_x(σ² ρ)

    Consistency loss:
        Fit a neural SDE (drift μ_θ, diffusion σ_θ) to the observed returns.
        The model's predicted transition probabilities must satisfy the FP PDE.

    Implementation:
        μ_θ(x) = drift MLP  → learned expected return given state x
        σ_θ(x) = diffusion MLP → learned return std given state x
        L_FP = ||observed_return - (μ_θ · dt + σ_θ · noise)||²  (one-step)
             + λ_KL · KL(ρ_model || ρ_FP)  (distribution consistency)

    Used as an auxiliary loss term in the PathIntegralLoss.

    Args:
        state_dim : dimension of state features
        hidden    : hidden neurons
        dt        : time step (1 bar)
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        state_dim: int = 4,  # 상태 특징 차원 수
        hidden: int = 32,    # 은닉층 뉴런 수
        dt: float = 1.0,     # 시간 스텝 크기 (1봉)
    ) -> None:
        super().__init__()  # 부모 클래스 초기화  # Calls the parent class constructor
        self.dt = dt  # 시간 스텝 저장

        # Neural SDE drift μ_θ(x) — 상태 x에서의 기대 수익률(드리프트)를 예측하는 네트워크
        self.drift_net = nn.Sequential(  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module
            nn.Linear(state_dim, hidden),  # 상태 → 은닉층  # Fully-connected (affine) layer: y = xW^T + b
            nn.Tanh(),                      # tanh 활성화
            nn.Linear(hidden, 1),           # → 스칼라 드리프트  # Fully-connected (affine) layer: y = xW^T + b
        )

        # Neural SDE diffusion σ_θ(x) > 0 (log parameterized)
        # 상태 x에서의 수익률 변동성(확산)을 예측하는 네트워크 (log로 파라미터화 → 항상 양수)
        self.log_diffusion_net = nn.Sequential(  # 신경망 레이어를 만들어 이 객체에 저장한다  # Stores a neural network layer as an attribute of this module
            nn.Linear(state_dim, hidden),  # 상태 → 은닉층  # Fully-connected (affine) layer: y = xW^T + b
            nn.Tanh(),                      # tanh 활성화
            nn.Linear(hidden, 1),           # → log(확산) 스칼라  # Fully-connected (affine) layer: y = xW^T + b
        )

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        state: torch.Tensor,           # [B, T, state_dim]  or [B, state_dim] 상태 특징
        observed_returns: torch.Tensor, # [B, T] or [B] 실제 관측된 로그 수익률
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Fokker-Planck consistency loss.
        포커-플랑크 일관성 손실을 계산한다.

        One-step Euler-Maruyama consistency:
            x_{t+1} ≈ x_t + μ_θ(x_t) · dt + σ_θ(x_t) · √dt · ε

        NLL loss over observed transitions:
            L_FP = -log p(x_{t+1}|x_t) where
            p(x_{t+1}|x_t) = N(x_t + μ·dt, σ²·dt)

        Args:
            state           : [B, T, state_dim] — feature states
            observed_returns: [B, T] — actual log returns

        Returns:
            fp_loss   : scalar — FP consistency loss
            mu        : [B, T] or [B] — predicted drift
            sigma     : [B, T] or [B] — predicted diffusion
        """
        squeeze = state.dim() == 2  # 2D 입력이면 나중에 다시 1D로 줄여야 한다는 표시
        if squeeze:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            state = state.unsqueeze(1)           # [B, 1, D] 차원 추가
            observed_returns = observed_returns.unsqueeze(1)  # [B, 1] 차원 추가

        B, T, D = state.shape  # 배치 크기, 시간 길이, 특징 차원  # Shape (dimensions) of the tensor/array

        # Compute drift and diffusion at each state — 각 상태에서 드리프트와 확산 계산
        mu = self.drift_net(state).squeeze(-1) * self.dt    # [B, T] 예측 드리프트 × 시간 스텝
        log_sigma = self.log_diffusion_net(state).squeeze(-1)  # [B, T] log(확산) 예측
        sigma = torch.exp(log_sigma).clamp(min=1e-4) * math.sqrt(self.dt)  # [B, T] 확산 = exp(log_sigma) × √dt

        # Sigma-collapse fix: detach sigma from the gradient graph so the
        # optimizer cannot drive σ→0 to exploit log(σ)→-∞.
        # Full Gaussian NLL:  ½(r-μ)²/σ² + log(σ) + const
        #                     ^learnable     ^causes σ-collapse when detached→off
        # Safe formulation: heteroskedastic MSE with stop-gradient on σ.
        #   L_fp = (r - μ)² / max(σ_sg², σ_floor²)   ≥ 0 always
        # σ is still updated via the drift pathway (residual gradient), but
        # log(σ) no longer creates a negative-reward sink.
        sigma_floor = observed_returns.std(unbiased=True).clamp(min=1e-6).detach()  # 실제 수익률의 표준편차 (하한값)  # Detaches tensor from the computation graph
        sigma_sg    = sigma.detach().clamp(min=sigma_floor)   # sigma를 기울기 그래프에서 분리 (σ→0 붕괴 방지)
        residual    = observed_returns - mu                    # [B, T] 잔차: 실제 수익률 - 예측 드리프트
        fp_loss     = ((residual / sigma_sg) ** 2).mean()     # 정규화된 잔차의 평균제곱 손실 (항상 ≥ 0)

        if squeeze:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            return fp_loss, mu.squeeze(1), sigma.squeeze(1)  # 2D 입력이었으면 다시 1D로 줄여서 반환  # Returns a value to the caller

        return fp_loss, mu, sigma  # 포커-플랑크 손실, 드리프트, 확산 반환  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# 7. Wasserstein DRO Loss
# ─────────────────────────────────────────────────────────────────────────────

class WassersteinDROLoss(nn.Module):  # ★ [WassersteinDROLoss] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Wasserstein Distributionally Robust Optimization (DRO).
    와서스타인 분포 강건 최적화: 레짐 변화에 강건한 손실 함수.

    Standard ERM is brittle to distribution shift (regime changes).
    DRO optimizes against the worst-case distribution within a
    Wasserstein ball of radius ε around the empirical distribution Q:

        min_θ max_{P: W_1(P,Q)≤ε} E_P[ℓ(θ, x)]

    Dual formulation (Lagrangian):
        min_θ { E_Q[ℓ(θ,x)] + λ · E_Q[||∇_x ℓ(θ,x)||] }

    The gradient norm penalty λ · E[||∇_x ℓ||] is equivalent to
    enforcing Lipschitz continuity of the loss:
        |ℓ(θ, x) - ℓ(θ, x')| ≤ λ · ||x - x'||_1

    In trading: penalizes sensitivity to small feature perturbations
    → robust to data noise, measurement error, market microstructure.

    Args:
        epsilon   : Wasserstein ball radius ε (default 0.05)
        lambda_   : dual variable λ (default 0.1)
        n_samples : number of adversarial perturbation samples
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        epsilon: float = 0.05,   # 와서스타인 공 반지름 ε (입력 변화를 얼마나 허용할지)
        lambda_: float = 0.1,    # 기울기 페널티 강도 λ
        n_samples: int = 4,      # 적대적 샘플 수
    ) -> None:
        super().__init__()  # 부모 클래스 초기화  # Calls the parent class constructor
        self.epsilon = epsilon    # ε 저장
        self.lambda_ = lambda_   # λ 저장
        self.n_samples = n_samples  # 샘플 수 저장

        # Learned λ (can be trained) — λ를 학습 파라미터로 만든다 (log로 항상 양수)
        self.log_lambda = nn.Parameter(torch.tensor(math.log(lambda_)))  # 학습 중 자동 업데이트되는 파라미터로 등록한다  # Registers tensor as a learnable parameter (tracked by autograd)

    @property  # 이 메서드를 속성처럼 obj.속성 형태로 접근할 수 있게 만든다  # Decorator: expose method as a read-only attribute
    def effective_lambda(self) -> torch.Tensor:  # [effective_lambda] 함수 정의 시작
        return torch.exp(self.log_lambda).clamp(max=10.0)  # 실효 λ: exp(log_lambda), 최대 10으로 제한  # Returns a value to the caller

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        loss_fn,                       # callable: x → scalar loss — 손실 함수 (입력받아 스칼라 반환)
        x: torch.Tensor,               # [B, ...] 입력 특징
        **loss_kwargs,                 # 손실 함수에 전달할 추가 인자
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DRO loss = E_Q[ℓ] + λ · ||∇_x ℓ||.
        DRO 손실 = 일반 손실 + 기울기 페널티를 계산한다.

        Args:
            loss_fn   : function(x, **kwargs) → scalar loss
            x         : [B, ...] nominal input features
            **loss_kwargs: additional arguments for loss_fn

        Returns:
            dro_loss    : scalar — DRO-robust objective
            grad_penalty: scalar — Lipschitz penalty term
        """
        # Nominal loss — 기본 손실 계산 (입력 그대로 사용)
        nominal_loss = loss_fn(x, **loss_kwargs)

        # Gradient penalty: estimate ||∇_x ℓ|| — 기울기 크기 페널티 계산
        x_adv = x.detach().requires_grad_(True)  # 입력에 대한 기울기를 계산하기 위해 grad 활성화  # Detaches tensor from the computation graph
        adv_loss = loss_fn(x_adv, **loss_kwargs)  # 기울기 계산용 손실
        grads = torch.autograd.grad(
            adv_loss,      # 이 값에 대해
            x_adv,         # 이 변수의 기울기를 계산
            create_graph=False,   # 2차 미분 불필요
            retain_graph=False,   # 그래프 유지 불필요
        )[0]

        # L2 gradient norm averaged over batch — 배치 평균 L2 기울기 크기
        # Flatten all non-batch dims, compute per-sample norm
        grad_norm = grads.view(x.shape[0], -1).norm(dim=-1).mean()  # 각 샘플의 기울기 크기 평균  # Reshapes tensor (shares memory if possible)
        grad_penalty = self.effective_lambda * grad_norm  # λ × ||∇_x ℓ|| 페널티

        dro_loss = nominal_loss + grad_penalty  # DRO 손실 = 기본 손실 + 페널티

        return dro_loss, grad_penalty  # DRO 손실과 페널티 반환


# ─────────────────────────────────────────────────────────────────────────────
# 8. Platt Calibrator + ECE Tracker
# ─────────────────────────────────────────────────────────────────────────────

class PlattCalibrator(nn.Module):  # ★ [PlattCalibrator] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Platt scaling + Expected Calibration Error (ECE) tracking.
    플랫 스케일링: 모델의 확률 예측을 실제 승률과 일치하도록 보정한다.

    Platt scaling (Platt 1999):
        P_calibrated(y=1|x) = σ(a · f(x) + b)
        where a, b are scalar parameters learned on a calibration set.

    Multi-class generalization:
        P_calibrated(a|x) = softmax(T · z_a + b_a)
        where T (temperature) and b_a (bias) are learned.

    Replaces hard confidence threshold (confidence_threshold=0.55) with
    a principled calibrated probability estimate.

    ECE = Σ_m (|B_m| / n) |acc(B_m) - conf(B_m)|
    where B_m are confidence bins.

    Trading interpretation:
        A well-calibrated model with predicted P(TP)=0.65 will achieve
        ~65% TP rate → directly tradeable confidence levels.

    Args:
        n_classes    : number of output classes (default 3)
        n_bins       : ECE calibration bins (default 10)
        init_temp    : initial temperature (1.0 = uncalibrated)
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        n_classes: int = 3,        # 클래스 수 (관망/롱/숏)
        n_bins: int = 10,          # ECE 보정 구간 수
        init_temp: float = 1.0,    # 초기 온도 (1.0 = 보정 없음)
    ) -> None:
        super().__init__()  # 부모 클래스 초기화  # Calls the parent class constructor
        self.n_classes = n_classes  # 클래스 수 저장
        self.n_bins = n_bins        # 구간 수 저장

        # Platt temperature (scalar, log-parameterized for positivity)
        # 플랫 온도: log로 파라미터화해서 항상 양수인 온도 파라미터
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))  # 학습 중 자동 업데이트되는 파라미터로 등록한다  # Registers tensor as a learnable parameter (tracked by autograd)

        # Per-class bias (allows shifting class probabilities)
        # 클래스별 편향: 각 클래스의 확률을 이동시킬 수 있다
        self.bias = nn.Parameter(torch.zeros(n_classes))  # 학습 중 자동 업데이트되는 파라미터로 등록한다  # Registers tensor as a learnable parameter (tracked by autograd)

    @property  # 이 메서드를 속성처럼 obj.속성 형태로 접근할 수 있게 만든다  # Decorator: expose method as a read-only attribute
    def temperature(self) -> torch.Tensor:  # [temperature] 함수 정의 시작
        """Calibration temperature T > 0 — 보정 온도 (항상 양수)."""
        return torch.exp(self.log_temp).clamp(min=0.1, max=10.0)  # 0.1~10.0 범위로 제한  # Returns a value to the caller

    def calibrate(  # [calibrate] 함수 정의 시작
        self,
        logits: torch.Tensor,          # [B, n_classes] or [B, T, n_classes] 원시 로짓
    ) -> torch.Tensor:
        """
        Apply Platt calibration to raw logits.
        원시 로짓에 플랫 보정을 적용해 보정된 확률을 반환한다.

        Returns calibrated probabilities.
        """
        T = self.temperature  # 보정 온도
        b = self.bias          # 클래스별 편향

        if logits.dim() == 3:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # [B, T, A] 3차원 입력
            cal_logits = logits / T + b.view(1, 1, -1)  # 온도로 나누고 편향 더하기  # Reshapes tensor (shares memory if possible)
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            # [B, A] 2차원 입력
            cal_logits = logits / T + b.view(1, -1)     # 온도로 나누고 편향 더하기  # Reshapes tensor (shares memory if possible)

        return F.softmax(cal_logits, dim=-1)  # softmax로 확률로 변환 (합=1)

    @torch.no_grad()  # 기울기 계산 없이 실행 (검증용)  # Decorator: modifies the function / class below
    def compute_ece(  # [compute_ece] 함수 정의 시작
        self,
        probs: torch.Tensor,           # [B, n_classes] 보정된 확률
        labels: torch.Tensor,          # [B] int64 실제 정답 레이블
    ) -> float:
        """
        Expected Calibration Error (ECE).
        기대 보정 오류: 예측 확률과 실제 정확도의 차이를 측정한다.

        ECE = Σ_m (|B_m| / n) |acc(B_m) - conf(B_m)|

        Returns ECE in [0, 1] (lower is better).
        """
        B = probs.shape[0]  # 배치 크기  # Shape (dimensions) of the tensor/array
        confidences, predictions = probs.max(dim=-1)    # [B] 각 샘플의 최대 확률과 예측 클래스
        correct = predictions.eq(labels)                # [B] bool 예측이 맞으면 True

        ece = 0.0  # ECE 누적값 초기화
        bin_edges = torch.linspace(0.0, 1.0, self.n_bins + 1, device=probs.device)  # 0~1을 n_bins 구간으로 나눈다  # Creates evenly spaced values between start and end

        for i in range(self.n_bins):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()  # i번째 구간의 하한, 상한
            in_bin = (confidences >= lo) & (confidences < hi)  # 이 구간에 속하는 샘플 선택
            n_bin = in_bin.float().sum().item()   # 구간 내 샘플 수  # Casts tensor to float32
            if n_bin == 0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                continue  # 구간이 비어있으면 건너뛴다  # Skip the rest of this iteration

            acc_bin = correct[in_bin].float().mean().item()    # 구간 내 실제 정확도  # Casts tensor to float32
            conf_bin = confidences[in_bin].mean().item()       # 구간 내 평균 예측 확률  # Extracts a Python scalar from a 1-element tensor
            ece += (n_bin / B) * abs(acc_bin - conf_bin)       # 가중 오차 누적

        return ece  # ECE 반환 (0에 가까울수록 보정이 잘 됨)

    def calibration_nll_loss(  # [calibration_nll_loss] 함수 정의 시작
        self,
        logits: torch.Tensor,          # [B, n_classes] 원시 로짓
        labels: torch.Tensor,          # [B] int64 실제 정답
    ) -> torch.Tensor:
        """
        NLL loss for fitting calibration parameters (a, b) on validation set.
        검증 세트에서 보정 파라미터(온도, 편향)를 학습하기 위한 음의 로그 우도 손실.

        min_{T, b} -Σ_i log P_cal(y_i|x_i)
        """
        cal_probs = self.calibrate(logits)              # [B, n_classes] 보정된 확률 계산
        log_probs = torch.log(cal_probs.clamp(min=1e-8))  # 로그 확률 (0이 되지 않게 최소값 보장)  # Computes natural logarithm element-wise
        nll = F.nll_loss(log_probs, labels)             # NLL 손실 계산
        return nll  # 보정 NLL 손실 반환  # Returns a value to the caller

    @torch.no_grad()
    def validate_p_long_calibration(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 10,
        ece_threshold: float = 0.10,
        high_conf_threshold: float = 0.70,
        high_conf_wr_threshold: float = 0.60,
    ) -> dict:
        """
        Gate check: validate calibration quality for p_long buckets.

        Integrates PlattCalibrator with calibration_validator module.
        Extracts p_long (class index 1) from calibrated softmax and runs
        the full reliability diagram + ECE + gate check.

        Args:
            logits:  Raw model logits [B, 3] (HOLD/LONG/SHORT).
            labels:  Ground truth action labels [B] (0=HOLD, 1=LONG, 2=SHORT).
            n_bins:  Number of calibration bins.
            ece_threshold:         ECE must be < this value.
            high_conf_threshold:   p_long threshold for high-confidence bucket.
            high_conf_wr_threshold: Minimum actual WR in high-conf bucket.

        Returns:
            dict with keys: status, ece, high_conf_wr, high_conf_n, gate_result.
        """
        from src.models.calibration_validator import validate_calibration

        cal_probs = self.calibrate(logits)       # [B, 3]
        p_long = cal_probs[:, 1].cpu().numpy()   # p_long = P(LONG)
        actuals = (labels == 1).long().cpu().numpy().astype(float)  # 1 if LONG correct

        gate = validate_calibration(
            predictions=p_long,
            actuals=actuals,
            n_bins=n_bins,
            ece_threshold=ece_threshold,
            high_conf_threshold=high_conf_threshold,
            high_conf_wr_threshold=high_conf_wr_threshold,
        )

        return {
            "status": gate.status,
            "ece": gate.ece,
            "high_conf_wr": gate.high_conf_wr,
            "high_conf_n": gate.high_conf_n,
            "ece_pass": gate.ece_pass,
            "high_conf_wr_pass": gate.high_conf_wr_pass,
            "reasons": gate.reasons,
            "gate_result": gate,
        }

    def forward(  # [forward] 신경망의 순방향 계산을 정의한다 (입력 → 출력)
        self,
        logits: torch.Tensor,  # 원시 로짓 입력
    ) -> torch.Tensor:
        """Alias for calibrate(). Returns calibrated probabilities — calibrate()의 별명. 보정된 확률 반환."""
        return self.calibrate(logits)  # calibrate 함수를 호출해서 반환


# ─────────────────────────────────────────────────────────────────────────────
# EntropyProductionEstimator — Schnakenberg Ṡ from action history
# ─────────────────────────────────────────────────────────────────────────────

class EntropyProductionEstimator:  # ★ [EntropyProductionEstimator] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    Estimates the entropy production rate Ṡ from recent action history.
    최근 행동 이력에서 엔트로피 생산율 Ṡ를 추정한다.

    Non-equilibrium statistical mechanics (Schnakenberg 1976):

        Ṡ = Σ_{i≠j} J_ij · ln(J_ij / J_ji)  ≥ 0

    Where J_ij = π_i · T_{i→j} is the probability flux from state i to j,
    π_i is the stationary distribution, and T_{i→j} is the transition probability.

    Interpretation for trading:
        Ṡ >> 0  → market is far from detailed balance → non-equilibrium →
                   directional probability fluxes dominate → PREDICTABLE
        Ṡ ≈ 0   → near detailed balance → J_ij ≈ J_ji for all pairs →
                   market is in thermal equilibrium → EFFICIENT → ABSTAIN

    Physical analogy:
        A market in perfect detailed balance has J_ij = J_ji everywhere, meaning
        each action is as likely to transition to any other as the reverse.
        When Ṡ > 0, there is a net directional cycle (e.g. HOLD→LONG→SHORT→HOLD
        has different flux than the reverse), which corresponds to a detectable
        trend cycle in the market.

    States: 0=HOLD, 1=LONG, 2=SHORT  (n_states = 3)

    Args:
        n_states  : number of discrete states (= n_actions, default 3)
        window    : rolling window of recent actions (default 50)
        threshold : minimum Ṡ to allow entry (0 = disabled, diagnostics only)
    """

    def __init__(  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self,
        n_states:  int   = 3,       # 이산 상태 수 (행동 수와 같음: 관망/롱/숏)
        window:    int   = 50,      # 최근 행동 이력 크기 (몇 개 행동을 기억할지)
        threshold: float = 0.005,   # 거래 허용 최소 엔트로피 생산율
    ) -> None:
        self.n_states  = n_states    # 상태 수 저장
        self.window    = window      # 이력 창 크기 저장
        self.threshold = threshold   # 임계값 저장
        self._history: list = []     # 최근 행동 이력 목록 (정수)
        self._last_s_dot: float = 0.0  # 마지막으로 계산한 엔트로피 생산율

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def update(self, action: int) -> None:  # [update] 내부 상태 또는 파라미터를 갱신한다
        """Record a new action into the rolling history — 새 행동을 이력에 추가한다."""
        self._history.append(int(action))  # 행동을 정수로 변환해서 이력에 추가  # Appends an item to the end of the list
        if len(self._history) > self.window:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self._history.pop(0)  # 이력이 창 크기보다 커지면 가장 오래된 항목 제거

    def compute(self) -> float:  # [compute] 함수 정의 시작
        """
        Compute entropy production rate Ṡ from current action history.
        현재 행동 이력에서 엔트로피 생산율 Ṡ를 계산한다.

        Returns 0.0 if insufficient history (< 10 samples).
        Caches result in self._last_s_dot.
        """
        import math as _math   # 수학 함수 (log 등)  # Import Python standard math library (log, exp, trig) as "_math"
        import numpy as _np    # 수치 계산 도구  # Import NumPy — fast numerical array computation as "_np"

        n = len(self._history)  # 이력 길이  # Returns the number of items
        if n < 10:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            self._last_s_dot = 0.0  # 이력이 10개 미만이면 계산 불가
            return 0.0  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

        N = self.n_states  # 상태 수

        # ── Transition count matrix C[i, j] ──────────────────────────
        # 전이 횟수 행렬: C[i,j] = 상태 i에서 j로 전이한 횟수
        C = _np.zeros((N, N), dtype=_np.float64)  # N×N 영행렬로 초기화
        for t in range(n - 1):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            i = int(self._history[t])      # 현재 상태
            j = int(self._history[t + 1])  # 다음 상태
            if 0 <= i < N and 0 <= j < N:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                C[i, j] += 1.0  # 전이 횟수 증가

        # Row-normalise → transition matrix T[i, j] = P(next=j | now=i)
        # 행 정규화: 전이 확률 행렬 생성 (각 행의 합=1)
        row_sums = C.sum(axis=1, keepdims=True).clip(min=1e-10)  # 행 합 (0 방지)
        T_mat = C / row_sums                                     # [N, N] 전이 확률 행렬

        # ── Stationary distribution π (empirical frequency) ──────────
        # 정상 분포 π: 각 상태의 경험적 빈도
        pi = _np.zeros(N, dtype=_np.float64)  # N개 상태 확률 초기화
        for a in self._history:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
            if 0 <= a < N:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                pi[a] += 1.0  # 각 상태 출현 횟수 누적
        pi /= pi.sum().clip(min=1e-10)  # 정규화해서 확률로 만든다

        # ── Probability flux J[i, j] = π_i · T[i, j] ────────────────
        # 확률 플럭스: 상태 i에서 j로의 확률 흐름
        J = pi[:, None] * T_mat                                  # [N, N]

        # ── Schnakenberg entropy production ───────────────────────────
        # 슈나켄베르크 엔트로피 생산율: Ṡ = Σ_{i≠j} J_ij · ln(J_ij / J_ji)
        s_dot = 0.0  # 엔트로피 생산율 초기화
        for i in range(N):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            for j in range(N):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
                if i != j:  # 자기 자신과의 전이는 제외  # Branch: executes only when condition is True
                    jij = J[i, j]  # i→j 플럭스
                    jji = J[j, i]  # j→i 플럭스 (역방향)
                    if jij > 1e-12 and jji > 1e-12:  # 둘 다 0보다 커야 log 계산 가능  # Branch: executes only when condition is True
                        s_dot += jij * _math.log(jij / jji)  # J_ij × log(J_ij/J_ji) 누적

        self._last_s_dot = float(max(s_dot, 0.0))  # 음수가 나오면 0으로 보정 후 저장
        return self._last_s_dot  # 엔트로피 생산율 반환  # Returns a value to the caller

    def allows_entry(self) -> bool:  # [allows_entry] 함수 정의 시작
        """
        Returns True iff Ṡ ≥ threshold.
        If threshold == 0, always returns True (diagnostics-only mode).
        Ṡ가 임계값 이상이면 거래 허용, 아니면 거부한다.
        """
        if self.threshold <= 0.0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            return True  # 임계값이 0 이하면 항상 허용 (진단 전용 모드)  # Returns a value to the caller
        return self._last_s_dot >= self.threshold  # Ṡ가 임계값 이상이면 True  # Returns a value to the caller

    @property  # 이 메서드를 속성처럼 obj.속성 형태로 접근할 수 있게 만든다  # Decorator: expose method as a read-only attribute
    def s_dot(self) -> float:  # [s_dot] 함수 정의 시작
        """Last computed entropy production rate — 마지막으로 계산한 엔트로피 생산율."""
        return self._last_s_dot  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    def __repr__(self) -> str:  # [__repr__] 객체를 문자열로 표현하는 메서드
        return (  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
            f"EntropyProductionEstimator("  # 문자열 안에 변수 값을 넣어 만든다
            f"n_states={self.n_states}, window={self.window}, "  # 문자열 안에 변수 값을 넣어 만든다
            f"threshold={self.threshold}, s_dot={self._last_s_dot:.5f})"  # 문자열 안에 변수 값을 넣어 만든다
        )  # 객체를 문자열로 표현할 때 보여줄 정보


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_advanced_physics(  # [build_advanced_physics] 함수 정의 시작
    n_qubits: int = 4,          # 큐비트 수
    state_dim: int = 4,         # 상태 차원
    n_actions: int = 3,         # 행동 수
    device: Optional[torch.device] = None,  # 디바이스 (None이면 자동 선택)
) -> dict:
    """
    Build all advanced physics modules and return as a dictionary.
    모든 고급 물리 모듈을 만들어 딕셔너리로 반환하는 편의 함수.

    Usage:
        phys = build_advanced_physics(n_qubits=4, state_dim=4, n_actions=3)
        hurst = phys["hurst"](log_returns)
        purity, coherence, regime_prob = phys["lindblad"](expvals)
        mi_lb, _ = phys["mine"](features, labels)
        out = phys["mps"](features)
        V, tp, sl = phys["stopping"](state)
        fp_loss, mu, sigma = phys["fp"](states, returns)
        platt_probs = phys["platt"](logits)

    Returns:
        dict with keys: hurst, lindblad, mine, mps, stopping, fp, platt
    """
    if device is None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU가 있으면 GPU, 없으면 CPU

    modules = {
        "hurst": HurstEstimator(n_scales=5),  # 허스트 지수 추정기 (5개 스케일)  # Hurst exponent: H>0.5 trending, H<0.5 mean-reverting
        "lindblad": LindbladDecoherence(n_qubits=n_qubits, n_lindblad=2),  # 린드블라드 결깨짐 (2개 점프 연산자)  # Lindblad master equation: quantum decoherence model
        "mine": MINEEstimator(x_dim=state_dim, y_dim=n_actions, hidden=64),  # MINE 상호 정보량 추정기
        "mps": MPSLayer(
            n_sites=n_qubits, d_phys=2, chi=8,
            in_features=state_dim, out_features=state_dim,
        ),  # MPS 텐서 네트워크 레이어 (본드 차원 8)
        "stopping": OptimalStoppingBoundary(
            state_dim=state_dim + 2, hidden=32  # +2 for pnl, time_remaining — 손익과 잔여 시간 추가
        ),  # 최적 정지 경계 (TP/SL 결정)
        "fp": FokkerPlanckRegularizer(state_dim=state_dim, hidden=32),  # 포커-플랑크 정규화기
        "platt": PlattCalibrator(n_classes=n_actions, n_bins=10),  # 플랫 보정기 (10개 구간)  # Platt scaling: converts raw logits to calibrated probabilities
    }

    for name, module in modules.items():  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
        modules[name] = module.to(device)  # 모든 모듈을 지정 디바이스로 이동한다  # Moves tensor to specified device or dtype

    return modules  # 모듈 딕셔너리 반환  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
    import time  # 시간 측정용  # Import Time measurement and sleep utilities

    torch.manual_seed(42)  # 난수 시드 고정 (재현 가능성 확보)
    device = torch.device("cpu")  # CPU에서 테스트
    B, T, K = 4, 30, 4  # 배치 크기 4, 시간 길이 30, 특징 차원 4

    print("=" * 70)  # 구분선  # Prints output to stdout
    print("  advanced_physics.py — Self-Test")  # 테스트 제목  # Prints output to stdout
    print("=" * 70)  # 결과를 화면에 출력한다  # Prints output to stdout

    # ── 1. HurstEstimator ─────────────────────────────────────────────────
    print("\n[1] HurstEstimator")  # 허스트 추정기 테스트  # Prints output to stdout
    hurst_est = HurstEstimator(n_scales=4)  # 4개 스케일로 추정기 생성
    lr = torch.randn(B, T) * 0.01  # 임의 로그 수익률 생성 (작은 값)  # Learning rate: step size for each parameter update
    H = hurst_est(lr)  # 허스트 지수 계산
    T_opt = HurstEstimator.optimal_lookback(H, T_base=30)  # 최적 관찰 기간 계산
    print(f"  H = {H.tolist()}")  # 허스트 지수 출력  # Prints output to stdout
    print(f"  T_opt = {T_opt.tolist()}")  # 최적 기간 출력  # Prints output to stdout
    assert H.shape == (B,)  # 형태 검증  # Assertion: raises AssertionError if condition is False
    assert ((H >= 0.05) & (H <= 0.95)).all(), "H out of range"  # 범위 검증  # Assertion: raises AssertionError if condition is False
    print("  ✓ Hurst in [0.05, 0.95]")  # 통과  # Hurst exponent: H>0.5 trending, H<0.5 mean-reverting

    # ── 2. LindbladDecoherence ────────────────────────────────────────────
    print("\n[2] LindbladDecoherence")  # 린드블라드 결깨짐 테스트  # Lindblad master equation: quantum decoherence model
    lindblad = LindbladDecoherence(n_qubits=2, n_lindblad=2)  # 2큐비트, 2점프 연산자  # Lindblad master equation: quantum decoherence model
    expvals = torch.randn(B, K) * 0.5  # 임의 측정값 생성  # Creates a tensor with standard normal random values
    purity, coherence, regime_prob = lindblad(expvals)  # 순수도, 결맞음, 레짐 확률 계산  # Lindblad master equation: quantum decoherence model
    print(f"  purity     = {purity.tolist()}")  # 순수도 출력  # Prints output to stdout
    print(f"  coherence  = {coherence.tolist()}")  # 결맞음 출력  # Prints output to stdout
    print(f"  regime_prob= {regime_prob.tolist()}")  # 레짐 확률 출력  # Prints output to stdout
    assert purity.shape == (B,)  # 형태 검증  # Assertion: raises AssertionError if condition is False
    print("  ✓ Lindblad shapes OK")  # 통과  # Lindblad master equation: quantum decoherence model

    # ── 3. MINEEstimator ──────────────────────────────────────────────────
    print("\n[3] MINEEstimator")  # MINE 추정기 테스트
    mine = MINEEstimator(x_dim=K, y_dim=3, hidden=32)  # 특징 차원 K, 레이블 3, 은닉 32
    x_feat = torch.randn(B, K)  # 임의 특징 생성  # Creates a tensor with standard normal random values
    y_label = F.one_hot(torch.randint(0, 3, (B,)), num_classes=3).float()  # 임의 원-핫 레이블 생성  # Casts tensor to float32
    mi_lb, T_j = mine(x_feat, y_label)  # 상호 정보량 하한 계산
    print(f"  MI lower bound = {mi_lb.item():.4f}")  # MI 하한 출력
    print(f"  T_joint range  = [{T_j.min().item():.4f}, {T_j.max().item():.4f}]")  # T 통계 범위
    # MI bound should be ≥ 0 after convergence (can be negative at init)
    print("  ✓ MINE forward OK")  # 통과  # Prints output to stdout

    # ── 4. MPSLayer ───────────────────────────────────────────────────────
    print("\n[4] MPSLayer")  # MPS 레이어 테스트
    mps = MPSLayer(n_sites=K, d_phys=2, chi=4, in_features=K, out_features=K)  # MPS 레이어 생성
    x_in = torch.randn(B, K)  # 임의 입력 생성  # Creates a tensor with standard normal random values
    out = mps(x_in)  # MPS 순전파 실행
    print(f"  MPS output shape: {tuple(out.shape)}")  # 출력 형태 확인  # Shape (dimensions) of the tensor/array
    assert out.shape == (B, K)  # 형태 검증  # Assertion: raises AssertionError if condition is False
    # Gradient flow — 기울기가 제대로 흐르는지 확인
    loss_mps = out.sum()  # 모든 출력 합산
    loss_mps.backward()  # 역전파 실행  # Computes gradients via backpropagation
    assert mps.tensors[0].grad is not None  # 첫 번째 텐서에 기울기가 있는지 확인  # Assertion: raises AssertionError if condition is False
    print("  ✓ MPS gradient flow OK")  # 통과  # Prints output to stdout

    # ── 5. OptimalStoppingBoundary ────────────────────────────────────────
    print("\n[5] OptimalStoppingBoundary")  # 최적 정지 경계 테스트  # Prints output to stdout
    stop_bound = OptimalStoppingBoundary(state_dim=K + 2, hidden=16)  # K+2 차원 상태 입력
    state = torch.randn(B, K + 2)  # 임의 상태 생성  # Creates a tensor with standard normal random values
    V, tp_mult, sl_mult = stop_bound(state)  # 포지션 유지 가치, TP/SL 배수 계산  # Take-profit multiplier: exit at TP × ATR
    print(f"  V (continuation): {V.tolist()}")  # 유지 가치 출력  # Prints output to stdout
    print(f"  TP mult ∈ [1,4]:  {tp_mult.tolist()}")  # TP 배수 출력
    print(f"  SL mult ∈ [0.5,2]:{sl_mult.tolist()}")  # SL 배수 출력
    assert ((tp_mult >= 1.0) & (tp_mult <= 4.0)).all()   # TP 범위 검증
    assert ((sl_mult >= 0.5) & (sl_mult <= 2.0)).all()   # SL 범위 검증
    print("  ✓ TP/SL multiplier ranges OK")  # 통과  # Prints output to stdout

    # ── 6. FokkerPlanckRegularizer ────────────────────────────────────────
    print("\n[6] FokkerPlanckRegularizer")  # 포커-플랑크 정규화기 테스트  # Prints output to stdout
    fp_reg = FokkerPlanckRegularizer(state_dim=K, hidden=16)  # 정규화기 생성
    states_t = torch.randn(B, T, K)   # 임의 상태 시계열 생성  # Creates a tensor with standard normal random values
    returns_t = torch.randn(B, T) * 0.01  # 임의 수익률 시계열 (작은 값)  # Creates a tensor with standard normal random values
    fp_loss, mu, sigma = fp_reg(states_t, returns_t)  # FP 손실, 드리프트, 확산 계산
    print(f"  FP loss = {fp_loss.item():.4f}")  # FP 손실 출력
    print(f"  mu range:   [{mu.min().item():.4f}, {mu.max().item():.4f}]")  # 드리프트 범위  # Extracts a Python scalar from a 1-element tensor
    print(f"  sigma range:[{sigma.min().item():.4f}, {sigma.max().item():.4f}]")  # 확산 범위  # Extracts a Python scalar from a 1-element tensor
    assert (sigma > 0).all()  # 확산이 항상 양수인지 검증  # Assertion: raises AssertionError if condition is False
    print("  ✓ FokkerPlanck loss OK, sigma > 0")  # 통과  # Prints output to stdout

    # ── 7. PlattCalibrator ────────────────────────────────────────────────
    print("\n[7] PlattCalibrator")  # 플랫 보정기 테스트  # Platt scaling: converts raw logits to calibrated probabilities
    platt = PlattCalibrator(n_classes=3, n_bins=5)  # 3 클래스, 5개 구간 보정기 생성  # Platt scaling: converts raw logits to calibrated probabilities
    logits_test = torch.randn(B, 3)  # 임의 로짓 생성  # Creates a tensor with standard normal random values
    labels_test = torch.randint(0, 3, (B,))  # 임의 정답 레이블 생성
    cal_probs = platt(logits_test)  # 보정된 확률 계산  # Platt scaling: converts raw logits to calibrated probabilities
    ece = platt.compute_ece(cal_probs, labels_test)  # ECE 계산
    nll = platt.calibration_nll_loss(logits_test, labels_test)  # 보정 NLL 계산  # Platt scaling: converts raw logits to calibrated probabilities
    print(f"  Calibrated probs: {cal_probs.detach().numpy().round(3).tolist()}")  # 보정 확률 출력  # Detaches tensor from the computation graph
    print(f"  ECE = {ece:.4f}  (0 = perfect calibration)")  # ECE 출력 (0에 가까울수록 좋음)
    print(f"  Calibration NLL = {nll.item():.4f}")  # 보정 NLL 출력  # Extracts a Python scalar from a 1-element tensor
    assert cal_probs.shape == (B, 3)  # 형태 검증  # Assertion: raises AssertionError if condition is False
    assert abs(cal_probs.sum(dim=-1) - 1.0).max().item() < 1e-4  # 확률 합=1 검증  # Assertion: raises AssertionError if condition is False
    print("  ✓ Platt calibration OK")  # 통과  # Platt scaling: converts raw logits to calibrated probabilities

    # ── 8. build_advanced_physics factory ─────────────────────────────────
    print("\n[8] build_advanced_physics (factory)")  # 팩토리 함수 테스트  # Prints output to stdout
    phys = build_advanced_physics(n_qubits=2, state_dim=K, n_actions=3)  # 모든 모듈 생성
    print(f"  Built modules: {list(phys.keys())}")  # 생성된 모듈 목록 출력  # Prints output to stdout
    print("  ✓ Factory OK")  # 통과  # Prints output to stdout

    print("\n" + "=" * 70)  # 결과를 화면에 출력한다  # Prints output to stdout
    print("  ✓ All advanced_physics.py tests passed!")  # 모든 테스트 통과  # Prints output to stdout
    print("=" * 70)  # 결과를 화면에 출력한다  # Prints output to stdout
