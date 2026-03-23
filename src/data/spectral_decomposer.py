"""
spectral_decomposer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quantum Trading V2 — Phase 1 구현체

마스터플랜 참조: Part 2.1 스펙트럼 고유 분해 및 파동 변환
           Part 2.2 4-Qubit 물리적 할당
           Part 2.3 양자 상태 인코딩 (Bloch Sphere)

역할:
    27차원 시장 데이터 윈도우를 스펙트럼 고유 분해하여
    4개의 물리적으로 해석 가능한 채널(Trend, Momentum, Volatility, Volume)로
    압축하고, 각 채널의 투영값 c_{k,t}와 위상 변화율 Δc_{k,t}를 추출한다.

    이 출력은 Phase 2(quantum_layers.py)에서 각 큐비트의
    블로흐 구면 좌표(θ, φ)로 직접 매핑된다.

출력 해석 (마스터플랜 Part 2.2):
    v_1 → Qubit 1 (Trend Spin)      : MACD, SMA 군집 — 거대한 방향성
    v_2 → Qubit 2 (Momentum Velocity): RSI, ROC      — 매수/매도 충돌 강도
    v_3 → Qubit 3 (Volatility Phase) : ATR, Bollinger — 변동성 파동 진폭
    v_4 → Qubit 4 (Volume Mass)      : OBV, VWAP     — 유동성의 물리적 질량
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# 파이썬의 오래된 버전과도 호환되도록 미래 기능을 미리 불러온다
from __future__ import annotations  # Import annotations from __future__ module

# 딥러닝을 위한 PyTorch 라이브러리를 불러온다
import torch  # Import PyTorch — core deep learning library
# 신경망 레이어 등 딥러닝 구성 요소를 담은 모듈을 불러온다
import torch.nn as nn  # Import PyTorch neural network building blocks as "nn"
# 숫자 계산을 빠르게 해주는 numpy 라이브러리를 불러온다
import numpy as np  # Import NumPy (numerical computation library) as "np"
# 타입 힌트(자료형 설명)에 쓰이는 도구들을 불러온다
from typing import Tuple, Optional  # Import Tuple, Optional from Type hint annotations

# ─────────────────────────────────────────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────────────────────────────────────────

# 입력 피처 차원 (V4 26-dim: 18 V3-base + 8 microstructure/CVD)
# 모델에 입력되는 특징의 개수: 26가지 시장 정보
FEATURE_DIM: int = 26

# 추출할 상위 고유 벡터 수 → 3개 큐비트에 1:1 대응 (N_QUBITS=3)
# 공분산 행렬에서 가장 중요한 방향 3개를 뽑는다 (양자 컴퓨터 큐비트 개수와 같다)
N_EIGENVECTORS: int = 3

# 수치 안정성을 위한 epsilon
# 계산 중 0으로 나누는 것을 막기 위한 아주 작은 수
EPSILON: float = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# SpectralDecomposer
# ─────────────────────────────────────────────────────────────────────────────

class SpectralDecomposer(nn.Module):  # ── Class [SpectralDecomposer]: groups related data and behaviour ──
    """
    27차원 시장 데이터를 4개의 고유 공간으로 분해하는 스펙트럼 분해기.

    마스터플랜 수식:
        X_norm = (X - μ) / σ                    [Z-score 정규화]
        Σ v_k  = λ_k v_k,  k = 1,2,3,4          [고유 분해]
        c_{k,t} = X_norm,t · v_k                [고유 공간 투영]
        Δc_{k,t} = c_{k,t} - c_{k,t-1}          [위상/각운동량 추출]

    RMT 개선 (Random Matrix Theory — Marchenko-Pastur denoising):
        λ_+ = σ²_MP (1 + √(D/T))²   — Marchenko-Pastur upper edge
        λ_- = σ²_MP (1 - √(D/T))²   — Marchenko-Pastur lower edge

        Any eigenvalue λ_k ≤ λ_+ belongs to the "noise bulk" (random matrix).
        Signal eigenvalues: λ_k > λ_+ → genuine cross-feature structure.

        Denoised covariance:
            Σ_signal = Σ_{k: λ_k > λ+} λ_k v_k v_k^T
                     + λ_mean_noise × Σ_{k: λ_k ≤ λ+} v_k v_k^T

        This eliminates spurious correlations caused by finite T < D,
        which is the root cause of rank deficiency (T=20 < D=27 → rank≤19).

    Args:
        feature_dim    : 입력 피처 차원 (기본 27)
        n_eigenvectors : 추출할 상위 고유 벡터 수 (기본 4, 큐비트 수와 동일)
        eps            : Z-score 정규화 분모 epsilon
        learnable_basis: True 이면 고유 벡터를 학습 가능한 파라미터로 고정
                         False(기본) 이면 배치마다 공분산으로 동적 계산
        use_rmt        : True(기본) 이면 Marchenko-Pastur denoising 적용
    """

    def __init__(  # [__init__] Constructor — runs when the object is created
        self,
        feature_dim: int = FEATURE_DIM,
        n_eigenvectors: int = N_EIGENVECTORS,
        eps: float = EPSILON,
        learnable_basis: bool = False,
        use_rmt: bool = True,  # Marchenko-Pastur RMT denoising threshold
        use_edmd: bool = True,
        use_lda: bool = True,
    ) -> None:
        # nn.Module의 초기화 함수를 먼저 호출한다 (PyTorch 규칙)
        super().__init__()  # Calls the parent class constructor

        # 입력 피처 차원을 저장한다
        self.feature_dim = feature_dim
        # 추출할 고유 벡터 수를 저장한다
        self.n_eigenvectors = n_eigenvectors
        # 0 나눔 방지용 작은 수를 저장한다
        self.eps = eps
        # 학습 가능한 기저를 사용할지 여부를 저장한다
        self.learnable_basis = learnable_basis
        # 마르첸코-파스투르 노이즈 제거를 사용할지 여부를 저장한다
        self.use_rmt = use_rmt  # Marchenko-Pastur RMT denoising threshold
        # Koopman EDMD 분해를 사용할지 여부를 저장한다
        self.use_edmd = use_edmd
        # Fisher LDA 판별 분석을 사용할지 여부를 저장한다
        self.use_lda = use_lda

        # ── Fisher LDA 버퍼 (fit_lda() 호출 전까지 zeros placeholder) ──────
        # 3 binary discriminant directions:
        #   w1: LONG  vs HOLD       (Qubit 1 — "매수 신호")
        #   w2: SHORT vs HOLD       (Qubit 2 — "매도 신호")
        #   w3: HOLD  vs LONG+SHORT (Qubit 3 — "대기 신호")
        # LDA 학습이 완료되었는지를 나타내는 플래그: 처음엔 False(미완료)
        self._lda_fitted: bool = False
        # LDA가 학습한 방향의 수를 저장한다
        self._lda_n_components: int = 0
        # LDA 판별 방향 행렬을 0으로 초기화한다: [특징차원, 3개방향]
        self.register_buffer("_lda_W", torch.zeros(feature_dim, 3))  # Creates a zero-filled tensor

        if learnable_basis:  # Branch: executes only when condition is True
            # 학습 가능한 고정 기저: 초기값은 무작위 직교화로 초기화
            # 학습 가능한 기저 행렬을 위한 빈 텐서를 만든다
            basis = torch.empty(feature_dim, n_eigenvectors)
            # 직교 행렬로 초기화한다 (열 벡터들이 서로 수직)
            nn.init.orthogonal_(basis)
            # 학습 가능한 파라미터로 등록한다
            self.register_parameter("basis", nn.Parameter(basis))  # Registers a tensor as a learnable parameter
        else:  # Branch: all previous conditions were False
            # 동적 계산 모드이면 기저를 None으로 설정한다
            self.basis = None  # type: ignore[assignment]

        # 배치 통계 캐시 (디버깅/설명 가능성용)
        # 마지막으로 계산된 고유값을 저장한다 (디버깅용)
        self._last_eigenvalues: Optional[torch.Tensor] = None
        # 마지막으로 계산된 고유벡터를 저장한다 (디버깅용)
        self._last_eigenvectors: Optional[torch.Tensor] = None
        # 마지막으로 계산된 RMT 임계값을 저장한다 (디버깅용)
        self._last_rmt_threshold: Optional[torch.Tensor] = None
        # 마지막 Koopman 고유값의 절댓값을 저장한다 (|λ_k|, EDMD 모드용)
        self._last_koopman_eig: Optional[torch.Tensor] = None  # |λ_k| of EDMD modes

        # ── Precomputed Koopman config buffers ────────────────────────────
        # Loaded by load_koopman_precomputed(); used in _edmd_decompose_enhanced()
        # When fitted: replaces dynamic per-batch EDMD with fixed projection
        self._koop_fitted: bool = False
        # Feature-level normalisation (applied before dictionary)
        self.register_buffer("_koop_feat_mu",  torch.zeros(feature_dim))
        self.register_buffer("_koop_feat_sig", torch.ones(feature_dim))
        # Dictionary-level normalisation (3D columns)
        self.register_buffer("_koop_psi_mu",   torch.zeros(feature_dim * 3))
        self.register_buffer("_koop_psi_sig",  torch.ones(feature_dim * 3))
        # Sparse selection indices and Koopman eigenvectors
        self.register_buffer("_koop_selected", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_koop_vecs",     torch.zeros(1, n_eigenvectors))

    # ─────────────────────────────────────────────────────────────────────
    # 내부 유틸리티
    # ─────────────────────────────────────────────────────────────────────

    def _zscore_normalize(self, x: torch.Tensor) -> torch.Tensor:  # [_zscore_normalize] Private helper function
        """
        마스터플랜 수식:  X_norm = (X - μ) / σ

        Args:
            x: [..., T, F]  (임의의 배치 차원, 시퀀스 길이 T, 피처 F)

        Returns:
            x_norm: 동일 shape, Z-score 정규화된 텐서
        """
        # TASK-4: 시간 축(T)만으로 정규화 → 피처별 독립 Z-score
        # dim=(-2,-1) 은 T×F 전체 스칼라 → 피처 간 스케일 파괴 (버그)
        # dim=-2 는 피처별로 자신의 시간 분포 기준 → LDA 방향 품질 개선
        # 시간 축(-2)을 따라 각 피처별 평균을 계산한다: [..., 1, F]
        mean = x.mean(dim=-2, keepdim=True)                # [..., 1, F]
        # 시간 축(-2)을 따라 각 피처별 표준편차를 계산하고 최솟값을 eps로 제한한다
        std  = x.std(dim=-2, keepdim=True).clamp(min=self.eps)        # [..., 1, F]

        # Z-score: X_norm = (X - μ) / σ (평균을 빼고 표준편차로 나눈다)
        return (x - mean) / std  # Returns a value to the caller

    @staticmethod  # Decorator: static method — no self or cls
    def _covariance_matrix(x_norm: torch.Tensor) -> torch.Tensor:  # [_covariance_matrix] Private helper function
        """
        공분산 행렬 Σ 계산.

        마스터플랜 수식:  Σ = (1/(T-1)) · X_norm^T · X_norm
                         → Σ v_k = λ_k v_k 를 위한 입력

        Args:
            x_norm: [B, T, F]  (Batch, Sequence, Feature)

        Returns:
            cov: [B, F, F]  에르미트(대칭) 공분산 행렬
        """
        # 배치 크기(B), 시퀀스 길이(T), 피처 차원(F)을 분리한다
        B, T, F = x_norm.shape  # Shape (dimensions) of the tensor/array
        # x_norm 을 피처 열 기준으로 정렬 (T 버전을 평균 제거 후 사용)
        # 시간 축(dim=1)으로 평균을 구해서 빼준다: 각 피처의 평균을 0으로 만든다
        x_centered = x_norm - x_norm.mean(dim=1, keepdim=True)  # [B, T, F]

        # Σ = X^T X / (T-1)
        # torch.linalg.eigh 는 대칭 행렬만 받으므로 반드시 대칭 공분산 행렬 사용
        # 행렬 곱으로 공분산 행렬을 계산한다: X^T × X → [B, F, F]
        cov = torch.bmm(x_centered.transpose(1, 2), x_centered)  # [B, F, F]
        # (T-1)로 나눠서 편향 없는 공분산 추정치를 구한다
        cov = cov / max(T - 1, 1)
        return cov  # Returns a value to the caller

    def _marchenko_pastur_denoise(  # [_marchenko_pastur_denoise] Private helper function
        self,
        cov: torch.Tensor,           # [B, F, F]
        T: int,                      # sequence length (number of samples)
    ) -> torch.Tensor:
        """
        Marchenko-Pastur denoising of the sample covariance matrix.

        Random Matrix Theory result (Marchenko & Pastur 1967):
            For a D×T random matrix with iid entries of variance σ²:
                λ_± = σ² (1 ± √(D/T))²   (Marchenko-Pastur edges)

            Eigenvalues in [λ-, λ+] are "noise" — pure finite-sample artifact.
            Eigenvalues above λ+ carry genuine signal.

        Denoising recipe:
            1. Eigendecompose Σ = V Λ V^T
            2. Noise floor λ_N = median of eigenvalues in the noise bulk
            3. Σ_denoised = Σ_{k: λ_k > λ+} λ_k v_k v_k^T
                          + λ_N × Σ_{k: λ_k ≤ λ+} v_k v_k^T

        This eliminates the rank deficiency problem (T < D) by replacing
        noise eigenvalues with a constant noise floor λ_N.

        Effect on SpectralDecomposer:
            - Before RMT: rank(Σ) ≤ min(T-1, D-1) ≤ 19 for T=20, D=27
            - After RMT:  Σ_denoised has full rank D (noise floor fills it)
            - Eigenvalues: more separated → more stable eigenvectors
            - Projections c_kt: less dominated by noise modes

        Args:
            cov: [B, F, F] — sample covariance matrix
            T  : sequence length used to compute cov

        Returns:
            cov_denoised: [B, F, F] — Marchenko-Pastur denoised covariance
        """
        # 수학 관련 기본 함수(제곱근 등)를 쓰기 위해 math 모듈을 불러온다
        import math as _math  # Import Python standard math library (log, exp, trig) as "_math"
        # 배치 크기(B)와 피처 차원(F)을 분리한다
        B, F, _ = cov.shape  # Shape (dimensions) of the tensor/array
        # 텐서가 CPU인지 GPU인지를 가져온다
        device = cov.device  # Target compute device: CUDA GPU or CPU
        # 텐서의 자료형(float32 등)을 가져온다
        dtype = cov.dtype

        # ── Eigendecompose raw covariance ─────────────────────────────────
        try:  # Try block: attempt code that might raise an exception
            # 공분산 행렬을 고유값 분해한다: 오름차순으로 고유값과 고유벡터를 반환한다
            eigenvalues_asc, eigenvectors_asc = torch.linalg.eigh(cov)
        except Exception:  # Except block: handles a raised exception
            # 분해에 실패하면 원래 공분산 행렬을 그대로 반환한다 (안전 처리)
            return cov  # Returns a value to the caller

        # 수치 오차로 음수가 된 고유값을 0으로 올려준다 (고유값은 0 이상이어야 한다)
        eigenvalues_asc = eigenvalues_asc.clamp(min=0.0)

        # ── Marchenko-Pastur threshold λ+ ─────────────────────────────────
        # σ²_MP 추정: 고유값의 중앙값을 사용한다 (극단값의 영향을 덜 받는다)
        sigma2_mp = eigenvalues_asc.median(dim=-1).values.clamp(min=self.eps)  # [B]

        # q = D / T  (피처 차원 / 시퀀스 길이의 비율)
        q = float(F) / max(T, 1)
        # λ+ = σ²(1 + √q)²: 순수 노이즈가 만드는 고유값의 최댓값(마르첸코-파스투르 상한)
        lambda_plus = sigma2_mp * (_math.sqrt(q) + 1.0) ** 2  # [B]
        # 최솟값을 eps로 제한한다
        lambda_plus = lambda_plus.clamp(min=self.eps)

        # 진단용으로 RMT 임계값을 캐시에 저장한다
        self._last_rmt_threshold = lambda_plus.detach().mean().unsqueeze(0)  # Inserts a new dimension of size 1

        # ── Separate signal and noise eigenvalues ─────────────────────────
        # signal_mask[b, k] = True: λ+보다 큰 고유값은 '진짜 신호'로 표시한다
        # lambda_plus를 [B, 1] 모양으로 확장해서 비교한다
        lp_expanded = lambda_plus.unsqueeze(-1)                # [B, 1]
        # 마르첸코-파스투르 임계값보다 큰 고유값만 '진짜 신호'로 선택한다
        signal_mask = (eigenvalues_asc > lp_expanded).to(dtype)  # [B, F]

        # 노이즈 마스크: 신호 마스크의 반대 (1 - 신호)
        noise_mask = 1.0 - signal_mask
        # 노이즈 고유값의 개수를 세고 최솟값을 1로 제한한다 (0 나눔 방지)
        noise_count = noise_mask.sum(dim=-1).clamp(min=1.0)      # [B]
        # 노이즈 고유값들의 평균을 노이즈 바닥(noise floor)으로 사용한다
        lambda_noise = (eigenvalues_asc * noise_mask).sum(dim=-1) / noise_count  # [B]
        # 노이즈 바닥의 최솟값을 eps로 제한한다
        lambda_noise = lambda_noise.clamp(min=self.eps)

        # ── Denoised eigenvalues ──────────────────────────────────────────
        # 신호 고유값은 그대로 유지하고, 노이즈 고유값은 노이즈 바닥으로 교체한다
        lambda_denoised = (
            eigenvalues_asc * signal_mask              # 진짜 신호 고유값은 그대로
            + lambda_noise.unsqueeze(-1) * noise_mask  # 노이즈 고유값은 평탄한 바닥값으로  # Inserts a new dimension of size 1
        )                                                         # [B, F]

        # ── Reconstruct denoised covariance Σ_d = V Λ_d V^T ─────────────
        # eigenvectors_asc: [B, F, F]  (열이 고유벡터)
        # 노이즈 제거된 고유값으로 대각 행렬을 만든다: [B, F, F]
        Lambda_d = torch.diag_embed(lambda_denoised)              # [B, F, F]
        # V × Λ_d × V^T 로 노이즈 제거된 공분산 행렬을 재구성한다
        cov_denoised = torch.bmm(
            torch.bmm(eigenvectors_asc, Lambda_d),
            eigenvectors_asc.transpose(-1, -2),  # Swaps two tensor dimensions
        )                                                         # [B, F, F]

        # 수치 오차로 대칭성이 깨지는 것을 방지: (A + A^T)/2로 완전히 대칭으로 만든다
        cov_denoised = (cov_denoised + cov_denoised.transpose(-1, -2)) / 2.0  # Swaps two tensor dimensions

        return cov_denoised  # Returns a value to the caller

    def _eigen_decompose(  # [_eigen_decompose] Private helper function
        self, cov: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        공분산 행렬의 고유값 분해.

        마스터플랜 수식:  Σ v_k = λ_k v_k,  k = 1,2,3,4
                         → 상위 n_eigenvectors 개의 고유 벡터 v_k 추출

        torch.linalg.eigh 는:
          - 대칭(에르미트) 행렬에 최적화된 수치 안정 알고리즘 사용
          - 고유값을 오름차순으로 반환 → 상위 K개는 마지막 K열

        Args:
            cov: [B, F, F]

        Returns:
            eigenvalues : [B, K]  (내림차순 — 지배적 성분이 앞)
            eigenvectors: [B, F, K] (각 열이 고유 벡터 v_k)
        """
        # 공분산 행렬을 고유값 분해한다: 오름차순으로 고유값, 고유벡터 반환
        eigenvalues_asc, eigenvectors_asc = torch.linalg.eigh(cov)

        # 내림차순으로 뒤집는다: 가장 중요한(큰) 고유값이 앞으로 온다
        eigenvalues  = eigenvalues_asc.flip(dims=[-1])          # [B, F]
        # 고유벡터도 같은 순서로 뒤집는다
        eigenvectors = eigenvectors_asc.flip(dims=[-1])         # [B, F, F]

        # 추출할 고유벡터의 수(K)를 가져온다
        K = self.n_eigenvectors
        # 상위 K개의 고유값만 선택한다
        top_eigenvalues  = eigenvalues[:, :K]                   # [B, K]
        # 상위 K개의 고유벡터만 선택한다
        top_eigenvectors = eigenvectors[:, :, :K]               # [B, F, K]

        # 나중에 설명이나 디버깅에 쓸 수 있도록 고유값을 캐시에 저장한다
        self._last_eigenvalues  = top_eigenvalues.detach()  # Detaches tensor from the computation graph
        # 나중에 설명이나 디버깅에 쓸 수 있도록 고유벡터를 캐시에 저장한다
        self._last_eigenvectors = top_eigenvectors.detach()  # Detaches tensor from the computation graph

        return top_eigenvalues, top_eigenvectors  # Returns a value to the caller

    # ─────────────────────────────────────────────────────────────────────
    # Fisher LDA — 지도 판별 분석
    # ─────────────────────────────────────────────────────────────────────

    def fit_lda(self, X: "np.ndarray", y: "np.ndarray") -> None:  # noqa: F821
        """
        Binary Fisher LDA — 두 개의 독립적인 이진 판별 방향 학습.

        Multi-class LDA (K-1=2 방향) 대신 의미론적으로 올바른 이진 LDA 사용:

            방향 1: LONG vs HOLD
                → 매수 진입 여부를 결정하는 피처 방향
                → SHORT 샘플 제외, LONG=1/HOLD=0 이진 분류

            방향 2: SHORT vs HOLD
                → 매도 진입 여부를 결정하는 피처 방향
                → LONG 샘플 제외, SHORT=1/HOLD=0 이진 분류

        Multi-class LDA와의 차이:
            Multi-class: "LONG vs (SHORT + HOLD)" 방향 — LONG vs SHORT 노이즈 혼재
            Binary LDA : "LONG vs HOLD" 방향 — 진입 결정에 직접 관련된 방향만 추출

        수식 (각 이진 LDA):
            S_W = Σ_{k∈{0,1}} Σ_{x∈C_k} (x-μ_k)(x-μ_k)^T   [within-class scatter]
            S_B = Σ_{k∈{0,1}} n_k (μ_k-μ)(μ_k-μ)^T          [between-class scatter]
            S_W^{-1} S_B w = λ w  → 최적 판별 방향 w 추출

        Args:
            X: [N, F] numpy float32 — 정규화된 피처
            y: [N]   numpy int     — 0=HOLD, 1=LONG, -1=SHORT 레이블
        """
        # 숫자 계산을 위해 numpy를 불러온다
        import numpy as np  # Import NumPy (numerical computation library) as "np"
        # 선형 판별 분석(LDA)을 위해 sklearn을 불러온다
        # Import LinearDiscriminantAnalysis from sklearn.discriminant_analysis module
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        # ── Direction 1: LONG vs HOLD ─────────────────────────────────────
        # SHORT 샘플 제외 (y != -1), 이진 레이블: LONG=1, HOLD=0
        # SHORT(-1)가 아닌 샘플만 선택한다: LONG과 HOLD만 남긴다
        mask1 = (y != -1)
        # 선택된 샘플의 피처를 가져온다
        X1    = X[mask1]
        # 레이블을 이진화한다: LONG이면 1, HOLD이면 0
        y1    = (y[mask1] == 1).astype(int)           # LONG=1, HOLD=0

        # LDA 모델을 만들고 LONG vs HOLD 방향을 학습한다
        lda1 = LinearDiscriminantAnalysis(n_components=1, solver="svd")
        lda1.fit(X1, y1)
        # 학습된 판별 방향 벡터를 float32로 가져온다
        w1 = lda1.scalings_[:, 0].astype(np.float32)  # [F]

        # ── Direction 2: SHORT vs HOLD ────────────────────────────────────
        # LONG 샘플 제외 (y != 1), 이진 레이블: SHORT=1, HOLD=0
        # LONG(1)이 아닌 샘플만 선택한다: SHORT와 HOLD만 남긴다
        mask2 = (y != 1)
        # 선택된 샘플의 피처를 가져온다
        X2    = X[mask2]
        # 레이블을 이진화한다: SHORT이면 1, HOLD이면 0
        y2    = (y[mask2] == -1).astype(int)          # SHORT=1, HOLD=0

        # LDA 모델을 만들고 SHORT vs HOLD 방향을 학습한다
        lda2 = LinearDiscriminantAnalysis(n_components=1, solver="svd")
        lda2.fit(X2, y2)
        # 학습된 판별 방향 벡터를 float32로 가져온다
        w2 = lda2.scalings_[:, 0].astype(np.float32)  # [F]

        # ── Direction 3: HOLD vs (LONG+SHORT) ────────────────────────────
        # 전체 샘플, 이진 레이블: HOLD=1, LONG or SHORT=0
        # "지금 아무것도 하지 말아야 할 상태인가?" 를 포착
        # Qubit 3에 할당 → w1/w2와 얽혀(entangle) 3-class 결정 완성
        # HOLD이면 1, 그 외(LONG/SHORT)이면 0으로 레이블을 만든다
        y3 = (y == 0).astype(int)                      # HOLD=1, LONG/SHORT=0

        # LDA 모델을 만들고 HOLD vs (LONG+SHORT) 방향을 학습한다
        lda3 = LinearDiscriminantAnalysis(n_components=1, solver="svd")
        lda3.fit(X, y3)
        # 학습된 판별 방향 벡터를 float32로 가져온다
        w3 = lda3.scalings_[:, 0].astype(np.float32)  # [F]

        # ── 세 방향 결합: [F, 3] ─────────────────────────────────────────
        # 각 Qubit: 1=LONG신호, 2=SHORT신호, 3=HOLD신호 → 얽힘으로 결합
        # w1, w2, w3 세 방향 벡터를 열로 쌓아서 [F, 3] 행렬을 만든다
        W = np.stack([w1, w2, w3], axis=1)             # [F, 3]
        # numpy 배열을 PyTorch 텐서로 변환하고 원래 버퍼와 같은 디바이스로 이동한다
        self._lda_W = torch.tensor(  # Converts Python data to a PyTorch tensor
            W, dtype=torch.float32
        ).to(self._lda_W.device)  # Moves tensor to specified device or dtype
        # LDA 학습 완료 플래그를 True로 설정한다
        self._lda_fitted = True
        # 학습된 방향의 수를 3으로 설정한다
        self._lda_n_components = 3

        # ── 분리도 진단 (Cohen's d) ──────────────────────────────────────
        # Cohen's d: 두 그룹의 평균 차이를 표준편차로 나눈 분리도 지표
        def _cohen_d(a, b):  # [_cohen_d] Private helper function
            return abs(a.mean() - b.mean()) / (  # Returns a value to the caller
                np.sqrt((a.std()**2 + b.std()**2) / 2.0) + 1e-8  # Computes square root element-wise
            )

        # X를 방향 w1에 투영한다 (1차원 점수로 변환)
        proj1 = X @ w1
        # LONG vs HOLD의 Cohen's d를 계산한다 (두 그룹이 모두 존재할 때만)
        d1 = _cohen_d(proj1[y == 1], proj1[y == 0]) if (y == 1).any() and (y == 0).any() else 0.0

        # X를 방향 w2에 투영한다
        proj2 = X @ w2
        # SHORT vs HOLD의 Cohen's d를 계산한다
        d2 = _cohen_d(proj2[y == -1], proj2[y == 0]) if (y == -1).any() and (y == 0).any() else 0.0

        # X를 방향 w3에 투영한다
        proj3 = X @ w3
        # LONG/SHORT(비HOLD) 샘플들의 투영값
        non_hold = proj3[y != 0]
        # HOLD 샘플들의 투영값
        hold_pts = proj3[y == 0]
        # HOLD vs 비HOLD의 Cohen's d를 계산한다
        # Returns the number of items
        d3 = _cohen_d(hold_pts, non_hold) if len(non_hold) > 0 and len(hold_pts) > 0 else 0.0

        # LDA 학습 결과를 화면에 출력한다
        print(f"  [LDA] 3 binary directions fitted (F={self.feature_dim}->K=3)")  # Prints output to stdout
        # 방향 1의 분리도와 큐비트 할당을 출력한다
        print(f"  [LDA] Dir1 (LONG  vs HOLD)    : Cohen d = {d1:.4f}  "  # Prints output to stdout
              f"{'[OK]' if d1 > 0.1 else '[WEAK]'}  <- Qubit 1")
        # 방향 2의 분리도와 큐비트 할당을 출력한다
        print(f"  [LDA] Dir2 (SHORT vs HOLD)    : Cohen d = {d2:.4f}  "  # Prints output to stdout
              f"{'[OK]' if d2 > 0.1 else '[WEAK]'}  <- Qubit 2")
        # 방향 3의 분리도와 큐비트 할당을 출력한다
        print(f"  [LDA] Dir3 (HOLD  vs L+S)     : Cohen d = {d3:.4f}  "  # Prints output to stdout
              f"{'[OK]' if d3 > 0.1 else '[WEAK]'}  <- Qubit 3")
        # 각 클래스(LONG, SHORT, HOLD)의 샘플 수를 출력한다
        # Loop: iterate over each item in the sequence
        for label, mask in [("LONG", y==1), ("SHORT", y==-1), ("HOLD", y==0)]:
            print(f"         {label}: {int(mask.sum()):,} samples")  # Prints output to stdout

    def _lda_project(self, x_norm: torch.Tensor) -> torch.Tensor:  # [_lda_project] Private helper function
        """
        학습된 LDA 방향으로 피처 투영.

        Args:
            x_norm: [B, T, F]  z-score 정규화된 피처
        Returns:
            c_lda:  [B, T, K_lda]  판별 공간 투영값
        """
        # 학습된 LDA 방향 행렬을 현재 텐서와 같은 디바이스로 이동한다
        W = self._lda_W.to(x_norm.device)          # [F, K_lda]
        # 각 시점의 피처 벡터를 LDA 방향들에 내적(dot product)한다: [B, T, K_lda]
        return torch.einsum("btf,fk->btk", x_norm, W)  # [B, T, K_lda]

    # ─────────────────────────────────────────────────────────────────────
    # Koopman EDMD 분해
    # ─────────────────────────────────────────────────────────────────────

    def load_koopman_precomputed(self, config_path_or_dict) -> None:
        """
        Load precomputed Koopman config from koopman_config.py output.

        After calling this, _edmd_decompose() will use the enhanced
        nonlinear dictionary + sparse selection + precomputed eigenvectors
        instead of fitting K dynamically per batch.

        Args:
            config_path_or_dict : str path to .npz OR dict from
                                  load_koopman_config() / precompute_koopman_config()
        """
        import numpy as np
        if isinstance(config_path_or_dict, str):
            from src.data.koopman_config import load_koopman_config
            cfg = load_koopman_config(config_path_or_dict)
        else:
            cfg = config_path_or_dict

        dev = next(self.parameters()).device if len(list(self.parameters())) > 0 \
              else self._koop_feat_mu.device

        def _t(key, dtype=torch.float32):
            return torch.tensor(np.array(cfg[key]), dtype=dtype).to(dev)

        self._koop_feat_mu  = _t("feat_mu")
        self._koop_feat_sig = _t("feat_sig")
        self._koop_psi_mu   = _t("psi_mu")
        self._koop_psi_sig  = _t("psi_sig")
        self._koop_selected = _t("selected", dtype=torch.long)
        self._koop_vecs     = _t("koopman_vecs")   # [lifted_D, n_modes]
        self._koop_fitted   = True

        lam = np.array(cfg["eigenvalues"])
        print(f"[SpectralDecomposer] Koopman config loaded | "
              f"|λ_k|={np.abs(lam).round(4).tolist()} | "
              f"sparse_terms={len(cfg['selected'])}", flush=True)

    def _edmd_decompose_enhanced(
        self,
        x_norm: "torch.Tensor",   # [B, T, F]  already z-scored by forward()
    ) -> "Tuple[torch.Tensor, torch.Tensor]":
        """
        Enhanced EDMD using precomputed Koopman config.

        Instead of fitting K per batch (dynamic, unreliable), this method:
            1. Rebuilds the nonlinear dictionary ψ(x)=[x,x²,tanh(x)] in PyTorch
            2. Applies precomputed dictionary normalisation
            3. Selects precomputed sparse columns
            4. Projects onto precomputed Koopman eigenvectors
               c_kt = Psi_selected @ koopman_vecs

        This is a pure matrix multiply — fast, GPU-compatible, differentiable.
        The Koopman eigenvectors were validated OOS via rolling walk-forward CV
        before training (no overfitting).

        Returns:
            eigenvalues  : [B, n_modes]    (stored eigenvalues, broadcast)
            eigenvectors : [B, F, n_modes] (first D rows of koopman_vecs)
        """
        B, T, F = x_norm.shape
        K = self.n_eigenvectors

        # ── Nonlinear dictionary (stable, bounded) ────────────────────────
        x_clip = x_norm.clamp(-5.0, 5.0)                   # [B, T, F]
        x2     = x_clip.pow(2).clamp(0.0, 25.0)
        x_tanh = torch.tanh(x_clip)
        Psi    = torch.cat([x_clip, x2, x_tanh], dim=-1)   # [B, T, 3F]

        # ── Dictionary normalisation ──────────────────────────────────────
        psi_mu  = self._koop_psi_mu.to(Psi.dtype)          # [3F]
        psi_sig = self._koop_psi_sig.to(Psi.dtype)         # [3F]
        Psi = (Psi - psi_mu) / psi_sig                     # [B, T, 3F]

        # ── Sparse column selection ───────────────────────────────────────
        sel   = self._koop_selected                         # [max_terms]
        Psi_s = Psi[:, :, sel]                             # [B, T, max_terms]

        # ── Project onto Koopman eigenvectors ─────────────────────────────
        vecs = self._koop_vecs.to(Psi.dtype)               # [max_terms, K]
        c_kt = Psi_s @ vecs                                 # [B, T, K]

        # Return dummy eigenvalues (real ones stored in buffer, for logging)
        dummy_eigs = torch.ones(B, K, device=x_norm.device, dtype=x_norm.dtype)
        # Return dummy eigenvectors (not used when c_kt is computed directly)
        dummy_vecs = torch.zeros(B, F, K, device=x_norm.device, dtype=x_norm.dtype)
        # Cache for diagnostics
        self._last_koopman_c_kt = c_kt.detach()

        return dummy_eigs, dummy_vecs, c_kt   # caller uses c_kt directly

    def _edmd_decompose(  # [_edmd_decompose] Private helper function
        self,
        x_norm: torch.Tensor,  # [B, T, F]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extended Dynamic Mode Decomposition (EDMD) Koopman operator.

        Theory
        ------
        The Koopman operator U_t acts on observables φ of the state:
            (U_t φ)(x) = E[ φ(x_t) | x_0 = x ]

        EDMD approximates U_t on a dictionary of observables (here: identity).
        Given pairs (X, X') of consecutive states:

            G = X^T X / T              [Gram matrix — auto-correlation]
            A = X^T X' / T             [cross-correlation matrix]
            K = G^{-1} A               [Koopman operator approximation]

        Eigendecompose K:
            K φ_k = λ_k φ_k

        Sorting criterion: |λ_k - 1| ascending
            |λ_k| ≈ 1 → quasi-periodic slow mode → predictable alpha signal
            |λ_k| << 1 → fast-decaying mode → noise
            |λ_k| > 1 → unstable (filtered by real-part projection)

        vs. PCA:
            PCA picks modes of maximum VARIANCE.
            EDMD picks modes of maximum PREDICTABILITY (|λ_k| ≈ 1).
            These are fundamentally different — in trending markets the
            dominant Koopman mode has near-unit eigenvalue but not the
            highest variance.

        Fallback: if EDMD fails (rank-deficient, numeric error), falls back
        to PCA via _eigen_decompose().

        Args:
            x_norm : [B, T, F]  Z-score normalized feature matrix

        Returns:
            top_eigenvalues  : [B, K]    real parts of top-K Koopman eigenvalues
            top_eigenvectors : [B, F, K] real parts of top-K Koopman eigenvectors
                               (unit-norm normalized, column-wise)
        """
        # 수학 함수를 쓰기 위해 math 모듈을 불러온다
        import math as _math  # Import Python standard math library (log, exp, trig) as "_math"
        # 배치 크기(B), 시퀀스 길이(T), 피처 차원(F)을 분리한다
        B, T, F = x_norm.shape  # Shape (dimensions) of the tensor/array
        # 추출할 고유벡터 수
        K = self.n_eigenvectors

        # 시퀀스가 4개보다 짧으면 EDMD 계산이 의미 없으므로 PCA로 대체한다
        if T < 4:  # Branch: executes only when condition is True
            return self._eigen_decompose(self._covariance_matrix(x_norm))  # Returns a value to the caller

        # 스냅샷 행렬 생성: 현재 상태(X)와 다음 상태(X_next)
        # 현재 시점의 데이터: 마지막 봉을 제외한 모든 봉
        X      = x_norm[:, :-1, :]   # [B, T-1, F] — current states
        # 다음 시점의 데이터: 첫 번째 봉을 제외한 모든 봉
        X_next = x_norm[:, 1:, :]    # [B, T-1, F] — next states
        # 실제 시퀀스 길이
        T1 = X.shape[1]  # Shape (dimensions) of the tensor/array

        # 그람 행렬(자기 상관): G = X^T × X / T1  [B, F, F]
        G = torch.bmm(X.transpose(1, 2), X) / max(T1, 1)  # Swaps two tensor dimensions
        # 교차 상관 행렬: A = X^T × X_next / T1  [B, F, F]
        A = torch.bmm(X.transpose(1, 2), X_next) / max(T1, 1)  # Swaps two tensor dimensions

        # RMT 노이즈 제거 옵션이 켜져 있으면 그람 행렬의 노이즈를 제거한다
        if self.use_rmt:  # Branch: executes only when condition is True
            G = self._marchenko_pastur_denoise(G, T1)  # Marchenko-Pastur RMT denoising threshold

        try:  # Try block: attempt code that might raise an exception
            # Koopman 연산자 K_mat = G^{-1} × A 를 최소자승법으로 구한다 (수치 안정적)
            K_mat = torch.linalg.lstsq(G, A, rcond=1e-4).solution  # [B, F, F]

            # K_mat의 고유값 분해: 비대칭 행렬이므로 복소수 고유값이 나올 수 있다
            eigenvalues, eigenvectors = torch.linalg.eig(K_mat)    # [B,F], [B,F,F]

        except Exception:  # Except block: handles a raised exception
            # 수치 오류 발생 시 표준 PCA로 대체한다
            return self._eigen_decompose(self._covariance_matrix(x_norm))  # Returns a value to the caller

        # 단위원(|λ|=1)에 가장 가까운 모드를 먼저 선택한다: 예측 가능한 신호를 우선
        sort_key = (eigenvalues - 1.0).abs()   # [B, F]  각 고유값의 |λ-1| 계산
        # |λ-1|이 작은 순서로 인덱스를 정렬한다 (단위원에 가까울수록 예측 가능)
        indices  = sort_key.argsort(dim=-1)    # [B, F]  오름차순 정렬

        # 상위 K개의 인덱스만 선택한다
        top_idx = indices[:, :K]               # [B, K]

        # 고유값의 실수 부분만 사용한다
        eig_r = eigenvalues.real                       # [B, F]
        # 선택된 K개 인덱스에 해당하는 고유값을 가져온다
        top_eigenvalues = eig_r.gather(1, top_idx)     # [B, K]

        # 고유벡터의 실수 부분만 사용한다
        eig_v_r = eigenvectors.real                                    # [B, F, F]
        # 인덱스를 [B, F, K] 모양으로 확장한다
        idx_exp = top_idx.unsqueeze(1).expand(B, F, K)                 # [B, F, K]
        # 선택된 K개 인덱스에 해당하는 고유벡터를 가져온다
        top_eigenvectors = eig_v_r.gather(2, idx_exp)                  # [B, F, K]

        # 각 고유벡터의 크기를 1로 정규화한다 (단위 벡터로 만든다)
        norms = top_eigenvectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # 각 열(고유벡터)을 그 크기로 나눠서 단위 벡터로 만든다
        top_eigenvectors = top_eigenvectors / norms                    # [B, F, K]

        # 진단용: 선택된 K개 모드의 |λ_k| 값을 캐시에 저장한다 (1에 가까울수록 좋다)
        top_eig_abs = eigenvalues.abs().gather(1, top_idx).detach()  # Detaches tensor from the computation graph
        self._last_koopman_eig  = top_eig_abs
        # 선택된 고유값을 캐시에 저장한다
        self._last_eigenvalues  = top_eigenvalues.detach()  # Detaches tensor from the computation graph
        # 선택된 고유벡터를 캐시에 저장한다
        self._last_eigenvectors = top_eigenvectors.detach()  # Detaches tensor from the computation graph

        return top_eigenvalues, top_eigenvectors  # Returns a value to the caller

    # ─────────────────────────────────────────────────────────────────────
    # 핵심 투영 연산
    # ─────────────────────────────────────────────────────────────────────

    def _project(  # [_project] Private helper function
        self,
        x_norm: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        정규화된 데이터를 고유 공간으로 투영.

        마스터플랜 수식:  c_{k,t} = X_norm,t · v_k
                          → c ∈ ℝ^{B × T × K}

        각 타임스텝 t의 27차원 피처 벡터를 4개의 고유 벡터에
        내적(dot product)하여 스칼라 투영값 c_{k,t}를 획득한다.

        물리적 해석:
            k=1 → Trend Spin      (추세 성분 투영값)
            k=2 → Momentum Velocity (모멘텀 성분 투영값)
            k=3 → Volatility Phase  (변동성 성분 투영값)
            k=4 → Volume Mass       (거래량 성분 투영값)

        Args:
            x_norm      : [B, T, F]
            eigenvectors: [B, F, K]  (동적 계산 모드)
                          또는 [F, K] (learnable_basis 모드)

        Returns:
            c_kt: [B, T, K]  — 각 타임스텝의 고유 공간 투영값
        """
        if self.learnable_basis:  # Branch: executes only when condition is True
            # 학습 가능한 기저 사용: 배치마다 같은 기저 행렬을 사용한다
            # x_norm: [B, T, F],  basis: [F, K]
            # → 각 시점의 피처를 학습된 기저 방향에 투영한다
            # Performs Einstein summation (generalized matrix ops)
            c_kt = torch.einsum("btf,fk->btk", x_norm, self.basis)
        else:  # Branch: all previous conditions were False
            # 동적 공분산 기저 사용: 배치마다 다른 고유 벡터를 사용한다
            # x_norm: [B, T, F],  eigenvectors: [B, F, K]
            # → 각 배치의 고유벡터 방향으로 데이터를 투영(압축)한다
            # Performs Einstein summation (generalized matrix ops)
            c_kt = torch.einsum("btf,bfk->btk", x_norm, eigenvectors)

        return c_kt  # Returns a value to the caller

    def _extract_phase(self, c_kt: torch.Tensor) -> torch.Tensor:  # [_extract_phase] Private helper function
        """
        투영값의 시간 미분으로 위상/각운동량 추출.

        마스터플랜 수식:  Δc_{k,t} = c_{k,t} - c_{k,t-1}

        물리적 의미:
            - Δc_{k,t} 는 블로흐 구면 위상각 φ_k 의 원천 데이터
            - φ_k = tanh(Δc_{k,t}) · π  (Part 2.3에서 사용)
            - 변화율이 클수록 적도면에서 위상이 급격히 회전함을 의미

        t=0 에서의 Δc 는 패딩(0)으로 처리하여
        시퀀스 길이와 shape을 보존한다.

        Args:
            c_kt: [B, T, K]

        Returns:
            delta_c_kt: [B, T, K] — 위상 변화율 (각운동량)
        """
        # 유한 차분: 현재 값 - 이전 값 (시간에 따른 변화량 계산)
        # torch.diff 는 [B, T-1, K] 를 반환하므로 first-step 을 0으로 패딩
        delta = torch.diff(c_kt, dim=1)                  # [B, T-1, K]

        # 첫 번째 타임스텝: 이전 값이 없으므로 0으로 패딩한다
        zero_pad = torch.zeros(  # Creates a zero-filled tensor
            c_kt.shape[0], 1, c_kt.shape[2],  # Shape (dimensions) of the tensor/array
            dtype=c_kt.dtype, device=c_kt.device
        )                                                 # [B, 1, K]
        # 0 패딩을 앞에 붙여서 원래 시퀀스 길이를 유지한다
        delta_c_kt = torch.cat([zero_pad, delta], dim=1) # [B, T, K]  # Concatenates tensors along a given dimension

        return delta_c_kt  # Returns a value to the caller

    # ─────────────────────────────────────────────────────────────────────
    # 블로흐 구면 인코딩 (Phase 2 preview — 수식만 적용)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod  # Decorator: static method — no self or cls
    def compute_bloch_angles(  # [compute_bloch_angles] Function definition
        c_kt: torch.Tensor,
        delta_c_kt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        투영값과 위상 변화율을 블로흐 구면 좌표로 변환.

        마스터플랜 수식 (Part 2.3):
            θ_k = π · σ(c_{k,t})             [스핀 강도, Spin Intensity]
            φ_k = tanh(Δc_{k,t}) · π         [각운동량/모멘텀, Phase Angle]

        여기서 σ 는 시그모이드 함수.

        물리적 해석:
            θ ∈ [0, π]: 북극(|0⟩, 상승) ~ 남극(|1⟩, 하락)
            φ ∈ [-π, π]: 적도면 회전 속도 (모멘텀 방향/강도)

        이 함수는 Phase 2(quantum_layers.py)의 Ry(θ), Rz(φ) 게이트에
        직접 주입되는 값을 사전 계산하는 유틸리티다.

        Args:
            c_kt      : [B, T, K]  투영값
            delta_c_kt: [B, T, K]  위상 변화율

        Returns:
            theta: [B, T, K]  ∈ [0, π]
            phi  : [B, T, K]  ∈ [-π, π]
        """
        # 원주율(π ≈ 3.14159)을 쓰기 위해 math 모듈을 불러온다
        import math  # Import Python standard math library (log, exp, trig)

        # θ_k = π · sigmoid(c_{k,t})
        # 시그모이드는 출력값이 (0, 1) 범위 → π를 곱하면 θ ∈ (0, π) 범위가 된다
        theta = math.pi * torch.sigmoid(c_kt)              # [B, T, K]

        # φ_k = tanh(Δc_{k,t}) · π
        # tanh은 출력값이 (-1, 1) 범위 → π를 곱하면 φ ∈ (-π, π) 범위가 된다
        phi = math.pi * torch.tanh(delta_c_kt)             # [B, T, K]

        return theta, phi  # Returns a value to the caller

    # ─────────────────────────────────────────────────────────────────────
    # 전체 파이프라인 Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(  # [forward] Defines the forward pass of the neural network
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        전체 스펙트럼 분해 파이프라인 실행.

        마스터플랜 Part 2.1 전체 흐름:
            X_t ∈ ℝ²⁷
             ↓
            Z-score 정규화: X_norm = (X - μ) / σ
             ↓
            공분산 행렬: Σ = X_norm^T X_norm / (T-1)
             ↓
            고유 분해: Σ v_k = λ_k v_k  (k=1,2,3,4)
             ↓
            투영: c_{k,t} = X_norm,t · v_k
             ↓
            위상 추출: Δc_{k,t} = c_{k,t} - c_{k,t-1}

        GPU 배치 처리:
            torch.linalg.eigh 는 배치 차원을 완전 지원하며,
            모든 연산이 CUDA tensor에서 그대로 작동한다.

        Args:
            x: [B, T, F] — 배치 크기 B, 시퀀스 길이 T, 피처 27

        Returns:
            x_norm     : [B, T, F]  Z-score 정규화된 입력
            c_kt       : [B, T, K]  고유 공간 투영값  (c_{k,t})
            delta_c_kt : [B, T, K]  위상 변화율        (Δc_{k,t})
            eigenvalues: [B, K]     상위 K개 고유값    (λ_k)
                                    → learnable_basis 모드에서는 None

        Raises:
            ValueError: 입력 피처 차원이 feature_dim 과 불일치할 때
        """
        # 입력이 2D([T, F])이면 배치 차원을 추가해서 [1, T, F]로 만든다
        if x.dim() == 2:  # Branch: executes only when condition is True
            # [T, F] 입력을 [1, T, F] 로 자동 확장
            x = x.unsqueeze(0)  # Inserts a new dimension of size 1

        # 입력 피처 차원이 설정값과 다르면 에러를 낸다
        if x.shape[-1] != self.feature_dim:  # Branch: executes only when condition is True
            raise ValueError(  # Raises an exception to signal an error
                f"입력 피처 차원 불일치: "
                f"expected {self.feature_dim}, got {x.shape[-1]}"  # Shape (dimensions) of the tensor/array
            )

        # ── Step 1: Z-score 정규화 ────────────────────────────────────────
        # X_norm = (X - μ) / σ      [마스터플랜 Part 2.1]
        # 각 피처의 평균을 0, 표준편차를 1로 만든다
        x_norm = self._zscore_normalize(x)          # [B, T, F]

        # ── Step 2 & 3: 스펙트럼 분해 ───────────────────────────────────────
        # 우선순위: LDA(지도) > learnable_basis > EDMD(비지도) > PCA+RMT
        if self.use_lda and self._lda_fitted:  # Branch: executes only when condition is True
            # ── Fisher LDA: 클래스 분리 최대 방향 (지도 판별 분석) ────────
            # S_W^{-1} S_B w = λ w  (between/within scatter ratio 최대화)
            # PCA: 분산 최대 방향 (거래 신호 무관)
            # LDA: LONG/SHORT/HOLD 분리 최대 방향 (직접적 거래 신호)
            # 학습된 LDA 방향으로 데이터를 투영한다: [B, T, K_lda]
            c_lda = self._lda_project(x_norm)           # [B, T, K_lda]
            # LDA 모드에서는 고유값이 없으므로 None으로 설정한다
            eigenvalues = None

            # LDA 방향 수가 필요한 수보다 적으면 EDMD로 나머지를 보충한다
            if c_lda.shape[-1] < self.n_eigenvectors:  # Branch: executes only when condition is True
                # 추가로 필요한 차원 수를 계산한다
                n_extra = self.n_eigenvectors - c_lda.shape[-1]  # Shape (dimensions) of the tensor/array
                # EDMD로 나머지 고유벡터를 구한다
                eig_vals, eig_vecs = self._edmd_decompose(x_norm)
                # 추가 차원만큼 투영한다
                c_extra = self._project(x_norm, eig_vecs[:, :, :n_extra])
                # LDA 투영값과 EDMD 투영값을 하나로 합친다
                c_kt = torch.cat([c_lda, c_extra], dim=-1)   # [B, T, K]
                # 추가로 사용한 고유값을 저장한다
                eigenvalues = eig_vals[:, :n_extra]
            else:  # Branch: all previous conditions were False
                # LDA 방향이 충분하면 필요한 수만큼만 잘라서 사용한다
                c_kt = c_lda[:, :, :self.n_eigenvectors]

        elif self.learnable_basis:  # Branch: previous condition was False, try this one
            # 학습 가능한 고정 기저 사용 (추론 속도 최적화)
            # 학습 가능한 기저 모드에서는 고유값이 없다
            eigenvalues  = None
            # _project 내부에서 self.basis를 사용하므로 None으로 넘긴다
            eigenvectors = None  # _project 내부에서 self.basis 사용
            # 학습된 기저 방향으로 데이터를 투영한다
            c_kt = self._project(x_norm, eigenvectors)  # [B, T, K]

        elif self.use_edmd:  # Branch: previous condition was False, try this one
            if self._koop_fitted:
                # ── Enhanced Koopman: precomputed nonlinear basis ─────────
                # Uses sparse dict + ridge-CV-tuned eigenvectors validated OOS
                _, _, c_kt = self._edmd_decompose_enhanced(x_norm)  # [B, T, K]
                eigenvalues = None
            else:
                # ── Fallback: dynamic linear EDMD per batch ───────────────
                eigenvalues, eigenvectors = self._edmd_decompose(x_norm)
                c_kt = self._project(x_norm, eigenvectors)  # [B, T, K]

        else:  # Branch: all previous conditions were False
            # ── PCA + RMT 경로 ───────────────────────────────────────────
            # 공분산 행렬을 계산한다
            cov = self._covariance_matrix(x_norm)
            # RMT 노이즈 제거 옵션이 켜져 있으면 노이즈를 제거한다
            if self.use_rmt:  # Branch: executes only when condition is True
                cov = self._marchenko_pastur_denoise(cov, x_norm.shape[1])  # Shape (dimensions) of the tensor/array
            # 공분산 행렬을 고유값 분해해서 주요 방향을 찾는다
            eigenvalues, eigenvectors = self._eigen_decompose(cov)
            # 선택된 고유벡터 방향으로 데이터를 투영한다
            c_kt = self._project(x_norm, eigenvectors)  # [B, T, K]

        # ── Step 5: 위상/각운동량 추출 ────────────────────────────────────
        # Δc_{k,t} = c_{k,t} - c_{k,t-1}  [마스터플랜 Part 2.1]
        # 투영값의 시간에 따른 변화율(속도)을 계산한다
        delta_c_kt = self._extract_phase(c_kt)      # [B, T, K]

        # 정규화된 입력, 투영값, 위상 변화율, 고유값을 반환한다
        return x_norm, c_kt, delta_c_kt, eigenvalues  # Returns a value to the caller

    # ─────────────────────────────────────────────────────────────────────
    # 설명 가능성 유틸리티
    # ─────────────────────────────────────────────────────────────────────

    def explained_variance_ratio(self) -> Optional[torch.Tensor]:  # [explained_variance_ratio] Function definition
        """
        상위 K개 고유값이 전체 분산에서 차지하는 비율 반환.

        PCA 분산 설명력: λ_k / Σ λ  (k = 1, ..., K)

        마스터플랜 해석:
            각 고유 벡터가 27차원 시장 데이터에서
            Trend/Momentum/Volatility/Volume 를 얼마나 잘
            '압축'하고 있는지를 정량화한다.

        Returns:
            ratio: [K,]  또는 None (캐시 미존재 시)
        """
        # 캐시된 고유값이 없으면 None을 반환한다
        if self._last_eigenvalues is None:  # Branch: executes only when condition is True
            return None  # Returns a value to the caller
        # 배치 차원을 평균내서 대표 고유값을 구한다
        ev = self._last_eigenvalues.mean(dim=0)     # [K]
        # 각 고유값이 전체 고유값 합에서 차지하는 비율을 계산한다
        ratio = ev / (ev.sum() + self.eps)
        return ratio  # Returns a value to the caller

    def qubit_labels(self) -> list[str]:  # [qubit_labels] Function definition
        """큐비트 물리적 할당 레이블 반환 (LDA 우선 모드 반영)."""
        # LDA가 학습된 경우 LDA 기반 큐비트 레이블을 반환한다
        if self.use_lda and self._lda_fitted:  # Branch: executes only when condition is True
            labels = [
                "Qubit-1 LONG  Signal (w1: LONG  vs HOLD)",
                "Qubit-2 SHORT Signal (w2: SHORT vs HOLD)",
                "Qubit-3 HOLD  Signal (w3: HOLD  vs LONG+SHORT)",
            ]
        else:  # Branch: all previous conditions were False
            # LDA가 없는 경우 물리적 해석 기반 큐비트 레이블을 반환한다
            labels = [
                "Qubit-1 Trend Spin       (v₁: MACD, SMA 군집)",
                "Qubit-2 Momentum Velocity (v₂: RSI, ROC)",
                "Qubit-3 Volatility Phase  (v₃: ATR, Bollinger)",  # ATR: Average True Range — average price volatility
            ]
        # n_eigenvectors 개수만큼만 잘라서 반환한다
        return labels[: self.n_eigenvectors]  # Returns a value to the caller

    def summary(self) -> str:  # [summary] Function definition
        """인간이 읽을 수 있는 분해 결과 요약 문자열 반환."""
        # 출력할 줄들을 담을 리스트를 만든다
        lines = ["────── SpectralDecomposer Summary ──────"]
        # 입력 피처 차원을 추가한다
        lines.append(f"  Feature dim  : {self.feature_dim}")  # Appends an item to the end of the list
        # 추출된 고유벡터 수를 추가한다
        lines.append(f"  Eigenvectors : {self.n_eigenvectors}")  # Appends an item to the end of the list
        # 기저 모드(학습 가능 vs 동적)를 추가한다
        # Appends an item to the end of the list
        lines.append(f"  Basis mode   : {'Learnable' if self.learnable_basis else 'Dynamic (per-batch covariance)'}")

        # 분산 설명력을 가져온다
        evr = self.explained_variance_ratio()
        # 분산 설명력이 있으면 각 큐비트의 설명력을 출력한다
        if evr is not None:  # Branch: executes only when condition is True
            lines.append("  Explained Variance Ratio (PCA):")  # Appends an item to the end of the list
            for i, (label, r) in enumerate(  # Loop: iterate over each item in the sequence
                zip(self.qubit_labels(), evr.tolist())  # Iterates over multiple iterables in parallel
            ):
                # 각 고유벡터(큐비트)가 전체 분산의 몇 퍼센트를 설명하는지 출력한다
                lines.append(f"    [{i+1}] {label} → {r*100:.2f}%")  # Appends an item to the end of the list

        lines.append("────────────────────────────────────────")  # Appends an item to the end of the list
        # 줄바꿈으로 연결해서 하나의 문자열로 반환한다
        return "\n".join(lines)  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# 팩토리 함수
# ─────────────────────────────────────────────────────────────────────────────

def build_spectral_decomposer(  # [build_spectral_decomposer] Function definition
    feature_dim: int = FEATURE_DIM,
    n_eigenvectors: int = N_EIGENVECTORS,
    learnable_basis: bool = False,
    use_rmt: bool = True,  # Marchenko-Pastur RMT denoising threshold
    device: Optional[torch.device] = None,
) -> SpectralDecomposer:
    """
    SpectralDecomposer 인스턴스를 생성하고 지정 디바이스로 이동.

    Args:
        feature_dim     : 입력 피처 차원 (기본 27)
        n_eigenvectors  : 추출할 고유 벡터 수 (기본 4)
        learnable_basis : 학습 가능한 고정 기저 사용 여부
        use_rmt         : Marchenko-Pastur RMT denoising 사용 여부 (기본 True)
        device          : 'cuda', 'cpu', 또는 None(자동)

    Returns:
        SpectralDecomposer 인스턴스
    """
    # 디바이스가 지정되지 않으면 GPU가 있으면 GPU, 없으면 CPU를 자동 선택한다
    if device is None:  # Branch: executes only when condition is True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Target compute device: CUDA GPU or CPU

    # SpectralDecomposer 객체를 만들고 지정된 디바이스로 이동한다
    model = SpectralDecomposer(
        feature_dim=feature_dim,
        n_eigenvectors=n_eigenvectors,
        learnable_basis=learnable_basis,
        use_rmt=use_rmt,  # Marchenko-Pastur RMT denoising threshold
    ).to(device)  # Moves tensor to specified device or dtype

    return model  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# 빠른 검증 (python -m src.data.spectral_decomposer 로 실행)
# ─────────────────────────────────────────────────────────────────────────────

# 이 파일을 직접 실행할 때만 아래 코드가 동작한다
if __name__ == "__main__":  # Branch: executes only when condition is True
    # 수학 함수를 쓰기 위해 math 모듈을 불러온다
    import math  # Import Python standard math library (log, exp, trig)

    print("=" * 60)  # Prints output to stdout
    print("  SpectralDecomposer — 빠른 자가 검증 (Self-Test)")  # Prints output to stdout
    print("  마스터플랜 Part 2.1 수식 반영 확인")  # Prints output to stdout
    print("=" * 60)  # Prints output to stdout

    # GPU가 있으면 GPU, 없으면 CPU를 사용한다
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Target compute device: CUDA GPU or CPU
    print(f"\n  실행 디바이스: {device}")  # Prints output to stdout

    # ── 1. 동적 공분산 모드 (기본) ───────────────────────────────────────
    print("\n[Mode 1] Dynamic Covariance Basis")  # Prints output to stdout
    # feature_dim=26, n_eigenvectors=3, 동적 기저 모드로 SpectralDecomposer를 만든다
    decomposer = build_spectral_decomposer(
        feature_dim=26, n_eigenvectors=3, learnable_basis=False, device=device
    )

    # 임의 배치: B=8, T=60 (60개 캔들 윈도우), F=27
    # 테스트용 임의 데이터를 만든다: 배치 8개, 60봉, 27개 특징
    x = torch.randn(8, 60, 27, device=device)  # Creates a tensor with standard normal random values
    # 전체 파이프라인을 실행한다
    x_norm, c_kt, delta_c_kt, eigenvalues = decomposer(x)

    # 입력 데이터의 모양을 출력한다
    print(f"  입력 shape      : {tuple(x.shape)}")  # Shape (dimensions) of the tensor/array
    # 정규화된 입력의 모양을 출력한다
    print(f"  x_norm shape    : {tuple(x_norm.shape)}")  # Shape (dimensions) of the tensor/array
    # 투영값의 모양을 출력한다
    print(f"  c_kt shape      : {tuple(c_kt.shape)}")  # Shape (dimensions) of the tensor/array
    # 위상 변화율의 모양을 출력한다
    print(f"  delta_c_kt shape: {tuple(delta_c_kt.shape)}")  # Shape (dimensions) of the tensor/array
    # 고유값의 모양을 출력한다
    print(f"  eigenvalues     : {tuple(eigenvalues.shape)}")  # Shape (dimensions) of the tensor/array

    # Z-score 확인 — 배치-타임 전체 평균 ≈ 0, std ≈ 1
    print(f"\n  Z-score 검증:")  # Prints output to stdout
    # 정규화 후 평균이 0에 가까운지 확인한다
    # Extracts a Python scalar from a 1-element tensor
    print(f"    mean(x_norm) ≈ {x_norm.mean().item():.4f}  (목표: ~0)")
    # 정규화 후 표준편차가 1에 가까운지 확인한다
    # Extracts a Python scalar from a 1-element tensor
    print(f"    std(x_norm)  ≈ {x_norm.std().item():.4f}   (목표: ~1)")

    # delta_c_kt 첫 타임스텝이 0(패딩)인지 확인한다
    # Assertion: raises AssertionError if condition is False
    assert (delta_c_kt[:, 0, :].abs() < 1e-6).all(), \
        "첫 타임스텝 패딩이 0이 아닙니다!"
    print("\n  ✓ Δc_{k,0} 패딩 검증 통과")  # Prints output to stdout

    # 분산 설명력을 가져온다
    evr = decomposer.explained_variance_ratio()
    print(f"\n  분산 설명력 (배치 평균):")  # Prints output to stdout
    # 각 큐비트의 분산 설명력을 퍼센트로 출력한다
    for label, r in zip(decomposer.qubit_labels(), evr.tolist()):  # Loop: iterate over each item in the sequence
        print(f"    {label} → {r*100:.2f}%")  # Prints output to stdout

    # 블로흐 구면 인코딩 확인
    # 투영값과 위상 변화율을 블로흐 구면 좌표(θ, φ)로 변환한다
    theta, phi = SpectralDecomposer.compute_bloch_angles(c_kt, delta_c_kt)
    # θ가 [0, π] 범위 안에 있는지 확인한다
    # Assertion: raises AssertionError if condition is False
    assert theta.min() >= 0 and theta.max() <= math.pi + 1e-4, \
        f"θ 범위 이탈: [{theta.min():.3f}, {theta.max():.3f}]"
    # φ가 [-π, π] 범위 안에 있는지 확인한다
    # Assertion: raises AssertionError if condition is False
    assert phi.min() >= -math.pi - 1e-4 and phi.max() <= math.pi + 1e-4, \
        f"φ 범위 이탈: [{phi.min():.3f}, {phi.max():.3f}]"
    print(f"\n  ✓ 블로흐 각도 범위 검증 통과")  # Prints output to stdout
    # θ의 실제 범위를 출력한다
    # Extracts a Python scalar from a 1-element tensor
    print(f"    θ ∈ [{theta.min().item():.3f}, {theta.max().item():.3f}]  (목표: [0, π])")
    # φ의 실제 범위를 출력한다
    # Extracts a Python scalar from a 1-element tensor
    print(f"    φ ∈ [{phi.min().item():.3f}, {phi.max().item():.3f}]  (목표: [-π, π])")

    # ── 2. 학습 가능한 기저 모드 ─────────────────────────────────────────
    print("\n[Mode 2] Learnable Basis")  # Prints output to stdout
    # 학습 가능한 기저 모드로 새 SpectralDecomposer를 만든다
    decomposer_l = build_spectral_decomposer(
        feature_dim=26, n_eigenvectors=3, learnable_basis=True, device=device
    )
    # 학습 가능한 기저 모드로 파이프라인을 실행한다
    x_norm_l, c_kt_l, delta_c_kt_l, ev_l = decomposer_l(x)
    # 학습 가능한 기저 모드에서는 고유값이 None이어야 한다는 것을 확인한다
    # Assertion: raises AssertionError if condition is False
    assert ev_l is None, "learnable_basis 모드에서 eigenvalues 는 None 이어야 함"
    # 투영값의 모양을 출력한다
    print(f"  ✓ c_kt shape: {tuple(c_kt_l.shape)}")  # Shape (dimensions) of the tensor/array
    print(f"  ✓ eigenvalues: None (as expected)")  # Prints output to stdout

    # 전체 요약을 출력한다
    print()  # Prints output to stdout
    print(decomposer.summary())  # Prints output to stdout

    print("\n  ✓ 모든 검증 통과! Phase 1 구현 완료.")  # Prints output to stdout
    print("=" * 60)  # Prints output to stdout
