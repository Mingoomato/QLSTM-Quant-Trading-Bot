"""
pretrain_bc.py — Phase 0: Behavior Cloning (BC) Pre-training
═══════════════════════════════════════════════════════════════════════════════

DeepMind 원칙 (Hassabis, AlphaGo Nature 2016):
  "Supervised learning provides a strong prior, reducing the search space
   for reinforcement learning exponentially."

수리물리학적 근거:
  RL gradient variance:  Var(∇J) ∝ Var(Â_t) × Var(∇ log π)
                                       ^^^
                         FP loss 발산으로 Var(Â_t) → ∞ (현재 상태)
                         → gradient noise > signal → 학습 불가

  CE gradient variance:  Var(∇J) ∝ Var(∇ log π) only
                         Â_t 항 없음 → gradient noise 4× 감소
                         → 신호 SNR = 3.18에서 학습 가능

알고리즘 구조 (AlphaGo와 대응):
  AlphaGo:  SL Policy Net (57% accuracy) → RL Fine-tuning → MCTS
  우리 시스템: BC Pre-training (>52% accuracy) → train_quantum_v2.py → Gates

실행 방법:
  python scripts/pretrain_bc.py --symbol BTCUSDT --timeframe 15m \\
    --days 1095 --epochs 20 --device cuda

출력:
  checkpoints/quantum_v2/agent_bc_pretrained.pt

이후 RL:
  python scripts/train_quantum_v2.py --symbol BTCUSDT --timeframe 15m \\
    --start_date 2023-01-01 --end_date 2024-01-01 --rolling-window 360 --pretrain-ckpt \\
    checkpoints/quantum_v2/agent_bc_pretrained.pt
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse  # 터미널에서 실행할 때 옵션(인자)을 받기 위한 라이브러리를 불러온다  # Import Command-line argument parsing
import os  # 파일과 폴더를 다루기 위한 운영체제 기능 라이브러리를 불러온다  # Import OS interface — file and directory operations
import sys  # 파이썬 실행 환경(경로 등)을 다루기 위한 라이브러리를 불러온다  # Import Python system and interpreter utilities
import time  # 시간을 측정하기 위한 라이브러리를 불러온다  # Import Time measurement and sleep utilities

# ── Windows CP949 콘솔에서 UTF-8 특수문자 출력 허용 ──────────────────────────
# stdout(표준 출력)이 encoding 변경을 지원하면 UTF-8로 바꾼다 (한글 출력 깨짐 방지)
if hasattr(sys.stdout, "reconfigure"):  # Branch: executes only when condition is True
    sys.stdout.reconfigure(encoding="utf-8")  # 화면 출력 인코딩을 UTF-8로 설정한다
# stderr(오류 출력)도 같은 방식으로 UTF-8로 바꾼다
if hasattr(sys.stderr, "reconfigure"):  # Branch: executes only when condition is True
    sys.stderr.reconfigure(encoding="utf-8")  # 오류 메시지 인코딩을 UTF-8로 설정한다
import numpy as np  # 숫자 배열 계산을 빠르게 할 수 있는 넘파이 라이브러리를 불러온다  # Import NumPy (numerical computation library) as "np"
import torch  # 딥러닝(인공지능 학습)을 위한 파이토치 라이브러리를 불러온다  # Import PyTorch — core deep learning library
import pandas as pd  # 표 형태의 데이터를 다루기 위한 판다스 라이브러리를 불러온다  # Import Pandas (DataFrame library) as "pd"
from datetime import datetime, timezone, timedelta  # 날짜와 시간을 다루는 도구들을 불러온다  # Import datetime, timezone, timedelta from Date and time handling

# 이 스크립트 파일의 한 칸 위 폴더(프로젝트 루트)를 파이썬 경로에 추가한다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 양자 인공지능 에이전트를 만들고 설정하는 모듈을 불러온다
# Import build_quantum_agent, AgentConfig from src.models.integrated_agent module
from src.models.integrated_agent import build_quantum_agent, AgentConfig
# 손실 함수(모델이 얼마나 틀렸는지 계산하는 함수)들을 불러온다
# Import QuantumDivineLossV2, RegretWeightedBCLos... from src.models.loss module
from src.models.loss import QuantumDivineLossV2, RegretWeightedBCLoss
# 거래소에서 데이터를 가져오는 데이터 클라이언트를 불러온다
from src.data.data_client import DataClient  # Import DataClient from src.data.data_client module
# 바이낸스 거래소에서 체결 내역 데이터를 가져오는 함수를 불러온다
# Import fetch_binance_taker_history from src.data.binance_client module
from src.data.binance_client import fetch_binance_taker_history
# V4 버전 특징(피처)을 만들고 캐시(저장)하는 함수를 불러온다
# Import generate_and_cache_features_v4 from src.models.features_v4 module
from src.models.features_v4 import generate_and_cache_features_v4
# V5 버전 특징(피처)을 만들고 캐시(저장)하는 함수를 불러온다
# Import generate_and_cache_features_v5 from src.models.features_v5 module
from src.models.features_v5 import generate_and_cache_features_v5
from src.models.features_structural import generate_and_cache_features_structural, FEAT_DIM as STRUCTURAL_FEAT_DIM
# 학습 과정을 그래프로 시각화하는 도구를 불러온다
from src.viz.training_viz import TrainingVisualizer  # Import TrainingVisualizer from src.viz.training_viz module
# 데이터에 정답 레이블(LONG/SHORT/HOLD)을 붙이는 함수들을 불러온다
from src.models.labeling import (  # Import ( from src.models.labeling module
    compute_triple_barrier_labels,       # 삼중 장벽 방식으로 레이블 계산
    compute_bidirectional_barrier_labels, # 양방향 장벽 방식으로 레이블 계산
    compute_clean_barrier_labels,         # 깨끗한 장벽 방식으로 레이블 계산
    standardize_1m_ohlcv,                 # 1분봉 데이터를 표준 형식으로 변환
)


# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN       = 96       # 관측 창 길이 (24시간, train_quantum_v2.py와 동일)
WARMUP        = 120      # 피처 워밍업 구간 (피처 첫 120바는 0-패딩)
LABEL_SMOOTH  = 0.1      # Label smoothing: 과신뢰 방지 (알파고 SL 단계에서도 사용)
ORTH_WEIGHT   = 0.05     # Hilbert Orthogonality: N=3 qubits R³ centroid 분리 강도
REGRET_W      = 1.0      # Regret 가중치 스케일 (1.0 = 기회비용 1× 적용)
# 타임프레임별 하루에 만들어지는 봉(캔들) 개수를 담은 사전(딕셔너리)
BARS_PER_DAY  = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}


# ─────────────────────────────────────────────────────────────────────────────
# 멀티심볼 데이터 로딩 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
# 심볼 하나의 데이터를 불러오고, 특징(feature)과 레이블(정답)을 돌려주는 함수
# [_load_symbol_data_for_bc] Private helper function
def _load_symbol_data_for_bc(sym: str, timeframe: str, start_str: str,
                              end_dt, dc, feat_ver: int = 4) -> tuple:
    """단일 심볼 데이터를 로드하고 피처+레이블을 반환.

    Returns:
        (features [N,26], actions [N]) or (None, None) on error.
    """
    print(f"\n  [{sym}] 데이터 로드 중 ({timeframe}) ...")  # 지금 어떤 심볼 데이터를 불러오는지 화면에 출력한다  # Prints output to stdout
    # 거래소에서 과거 봉(OHLCV) 데이터를 가져온다
    df_raw = dc.fetch_training_history(
        symbol=sym, timeframe=timeframe,  # 코인 이름과 봉 크기(1m, 15m 등) 지정
        start_date=start_str,             # 데이터 시작 날짜 지정
        end_ms=int(end_dt.timestamp() * 1000),  # 데이터 끝 시간을 밀리초로 변환해 지정
        cache_dir="data"                  # 캐시(임시 저장) 폴더 지정
    )
    # 데이터가 없거나 비어있으면 오류 메시지 출력 후 빈 값을 돌려준다
    if df_raw is None or df_raw.empty:  # Branch: executes only when condition is True
        print(f"  [{sym}] ERROR: empty dataframe — skip")  # 데이터가 없다는 경고를 출력한다  # Prints output to stdout
        return None, None  # 아무것도 없다는 뜻으로 None 두 개를 돌려준다  # Returns a value to the caller

    # 1분봉이면 표준 형식으로 변환하고, 아니면 그대로 사용한다
    df_clean = standardize_1m_ohlcv(df_raw) if timeframe == "1m" else df_raw
    # 시작 시간을 밀리초(1000분의 1초) 단위로 변환한다
    _start_ms = int(pd.Timestamp(start_str).timestamp() * 1000)
    # 끝 시간을 밀리초 단위로 변환한다
    _end_ms   = int(end_dt.timestamp() * 1000)
    # 끝 날짜를 "YYYY-MM-DD" 형식 문자열로 변환한다
    _end_str  = end_dt.strftime("%Y-%m-%d")

    # ── 펀딩비 merge ────────────────────────────────────────────────────────
    # 선물 거래에서 사용하는 펀딩 수수료 데이터를 가져온다
    df_funding = dc.fetch_funding_history(sym, _start_ms, _end_ms, cache_dir="data")
    # 펀딩비 데이터가 있으면 봉 데이터와 합친다
    if not df_funding.empty:  # Branch: executes only when condition is False
        # 밀리초 타임스탬프를 날짜시간 문자열로 변환한다
        df_funding["ts"] = (pd.to_datetime(df_funding["ts_ms"], unit="ms", utc=True)
                            .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
        df_clean = df_clean.copy()  # 원본 데이터를 변경하지 않도록 복사본을 만든다
        # 봉 데이터와 펀딩비 데이터를 시간(ts) 기준으로 합친다
        # Funding rate: periodic payment between longs and shorts
        df_clean = df_clean.merge(df_funding[["ts", "funding_rate"]], on="ts", how="left")
        # 빠진 펀딩비는 앞 값으로 채우고, 남은 빈칸은 0으로 채운다
        # Funding rate: periodic payment between longs and shorts
        df_clean["funding_rate"] = df_clean["funding_rate"].ffill().fillna(0.0)

    # ── OI merge ────────────────────────────────────────────────────────────
    # 미결제약정(OI, Open Interest) 데이터를 1시간 간격으로 가져온다
    df_oi = dc.fetch_open_interest_history(
        sym, _start_ms, _end_ms, interval="1h", cache_dir="data"
    )
    # OI 데이터가 있으면 봉 데이터와 합친다
    if not df_oi.empty:  # Branch: executes only when condition is False
        # 밀리초 타임스탬프를 날짜시간 문자열로 변환한다
        df_oi["ts"] = (pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
                       .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
        # 봉 데이터와 OI 데이터를 시간(ts) 기준으로 합친다
        # Open interest: total number of outstanding contracts
        df_clean = df_clean.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
        # 빠진 OI 값은 앞 값으로 채우고, 남은 빈칸은 0으로 채운다
        # Open interest: total number of outstanding contracts
        df_clean["open_interest"] = df_clean["open_interest"].ffill().fillna(0.0)

    # ── CVD (Binance taker) merge ───────────────────────────────────────────
    # 캐시 파일 이름을 만들 때 쓸 날짜 문자열(하이픈 없이)을 만든다
    _cs = start_str.replace("-", "")  # 시작 날짜에서 "-"를 없앤다 (예: 2023-01-01 → 20230101)
    _ce = end_dt.strftime("%Y%m%d")   # 끝 날짜도 같은 형식으로 만든다
    # 바이낸스 체결 데이터를 저장할 캐시 파일 경로를 만든다
    taker_cache = f"data/binance_taker_{sym}_{timeframe}_{_cs}_{_ce}.csv"
    try:  # Try block: attempt code that might raise an exception
        # 바이낸스에서 매수 체결 거래량 데이터를 가져온다
        df_taker = fetch_binance_taker_history(
            symbol=sym, interval=timeframe,    # 코인과 타임프레임 지정
            start_date=start_str, end_date=_end_str,  # 기간 지정
            cache_path=taker_cache, verbose=False,     # 캐시 경로, 로그 출력 안 함
        )
        # 가져온 데이터가 있으면 봉 데이터와 합친다
        if not df_taker.empty:  # Branch: executes only when condition is False
            df_clean = df_clean.merge(df_taker[["ts", "taker_buy_volume"]], on="ts", how="left")
            # 빠진 매수 거래량은 0으로 채운다
            df_clean["taker_buy_volume"] = df_clean["taker_buy_volume"].fillna(0.0)
    except Exception as _e:  # Except block: handles a raised exception
        # 데이터 가져오기가 실패하면 경고 메시지를 출력하고 계속 진행한다
        print(f"  [{sym}] CVD fetch failed ({_e}) — fallback")  # Prints output to stdout

    # ── 레이블 ──────────────────────────────────────────────────────────────
    # 15분봉이면 96봉(24시간), 아니면 60봉을 최대 보유 시간으로 설정한다
    bars_h = 96 if timeframe == "15m" else 60
    # 삼중 장벽 방식으로 각 봉에 LONG/SHORT/HOLD/불확실 레이블을 붙인다
    df_labeled_raw = compute_clean_barrier_labels(
        df_clean, alpha=4.0, beta=1.5, hold_band=1.5, hold_h=20, h=bars_h
    )
    # 레이블이 2(불확실)인 행을 제거하고 인덱스를 다시 매긴다
    df_labeled = df_labeled_raw[df_labeled_raw["label"] != 2].reset_index(drop=True)

    # ── 피처 ────────────────────────────────────────────────────────────────
    # 피처 버전이 5이면 V5(54차원) 피처를 만들고, 아니면 V4(28차원) 피처를 만든다
    if feat_ver == "structural":
        cache_file = f"data/feat_cache_{sym}_{timeframe}_{_cs}_{_ce}_structural.npy"
        all_features_raw = generate_and_cache_features_structural(df_clean, cache_file)
    elif feat_ver == 5:  # Branch: executes only when condition is True
        # V5 피처 캐시 파일 경로를 만든다
        cache_file = f"data/feat_cache_{sym}_{timeframe}_{_cs}_{_ce}_v5.npy"
        # V5 피처(54개 특징값)를 계산해서 캐시 파일에 저장하고 불러온다
        all_features_raw = generate_and_cache_features_v5(df_clean, cache_file)
    else:  # Branch: all previous conditions were False
        # V4 피처 캐시 파일 경로를 만든다
        cache_file = f"data/feat_cache_{sym}_{timeframe}_{_cs}_{_ce}_v4cvd.npy"
        # V4 피처(28개 특징값)를 계산해서 캐시 파일에 저장하고 불러온다
        all_features_raw = generate_and_cache_features_v4(df_clean, cache_file)

    # 레이블이 2(불확실)가 아닌 봉만 고르는 마스크(True/False 배열)를 만든다
    keep_mask    = df_labeled_raw["label"].values != 2
    # 마스크를 적용해서 유효한 봉의 피처만 남긴다
    all_features = all_features_raw[keep_mask]
    # 유효한 봉의 원본 레이블(+1, -1, 0)만 남긴다
    raw_labels   = df_labeled_raw["label"].values[keep_mask]
    # 원본 레이블(+1/-1/0)을 모델 행동 번호(1/2/0)로 바꾼다
    # Converts Python sequence to NumPy array
    actions      = np.array([raw_to_action(r) for r in raw_labels], dtype=np.int64)

    # 클래스별(LONG=1, SHORT=-1, HOLD=0) 레이블 개수를 센다
    _lv = {v: (raw_labels == v).sum() for v in [1, -1, 0]}
    # 버려진(불확실) 레이블 개수를 센다
    n_discard = (df_labeled_raw["label"] == 2).sum()
    # 클래스별 개수와 전체 봉 수를 화면에 출력한다
    print(f"  [{sym}] LONG={_lv[1]:,}  SHORT={_lv[-1]:,}  HOLD={_lv[0]:,}  "  # Prints output to stdout
          f"DISCARD={n_discard:,}  bars={len(df_clean):,}")  # Returns the number of items
    # 피처 배열과 행동 배열을 돌려준다
    return all_features, actions  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼: raw_label → action index
# ─────────────────────────────────────────────────────────────────────────────
# 원본 레이블 숫자를 모델이 사용하는 행동 번호로 바꿔주는 함수
def raw_to_action(raw_label: int) -> int:  # [raw_to_action] Function definition
    """Triple barrier raw label → 모델 action index.

    raw_label:  +1 = upper barrier (TP) hit   → LONG  (action=1)
                -1 = lower barrier (SL) hit   → SHORT (action=2)
                 0 = timeout / hold           → HOLD  (action=0)
    """
    # 원본 레이블이 +1(이익 실현)이면 행동 번호 1(LONG, 사도 된다)을 돌려준다
    if raw_label == 1:  # Branch: executes only when condition is True
        return 1   # LONG
    # 원본 레이블이 -1(손절)이면 행동 번호 2(SHORT, 팔도 된다)를 돌려준다
    elif raw_label == -1:  # Branch: previous condition was False, try this one
        return 2   # SHORT
    # 그 외(0, 시간 초과/대기)이면 행동 번호 0(HOLD, 기다린다)을 돌려준다
    return 0       # HOLD


# ─────────────────────────────────────────────────────────────────────────────
# 균형 배치 생성기
# ─────────────────────────────────────────────────────────────────────────────
# LONG/SHORT/HOLD 세 종류를 골고루 뽑아 학습 배치를 만드는 클래스
class BalancedBatchSampler:  # ── Class [BalancedBatchSampler]: groups related data and behaviour ──
    """LONG / SHORT / HOLD 각 클래스에서 동일한 수를 샘플링.

    수학적 근거:
      Class imbalance → 주류 클래스(HOLD≈60%)의 gradient가 소수 클래스(LONG/SHORT≈20%)를
      압도 → 모델이 항상 HOLD만 예측 → LONG/SHORT 학습 불가.
      Balanced sampling → 클래스별 gradient 기여 동일 → 진정한 3-way 학습.
    """

    # 클래스 초기화 함수: 레이블 배열과 배치 크기를 받아 설정한다
    # [__init__] Constructor — runs when the object is created
    def __init__(self, labels: np.ndarray, batch_size: int = 128):
        self.batch_size    = batch_size  # 한 번에 학습할 샘플 개수를 저장한다
        # 배치를 1:1:1 비율로 나눈다 (HOLD가 27%→33%로 올라가 기울기 균형 맞춤)
        self.n_hold  = batch_size // 3   # 배치 크기를 3으로 나눈 몫 = HOLD 샘플 수
        self.n_long  = batch_size // 3   # LONG 샘플 수
        self.n_short = batch_size - self.n_hold - self.n_long  # SHORT 샘플 수 (나머지)
        # 레이블이 0(HOLD)인 샘플들의 위치(인덱스)를 찾아 저장한다
        self.idx_hold     = np.where(labels == 0)[0]  # Returns elements chosen from two arrays by condition
        # 레이블이 1(LONG)인 샘플들의 위치를 찾아 저장한다
        self.idx_long     = np.where(labels == 1)[0]  # Returns elements chosen from two arrays by condition
        # 레이블이 2(SHORT)인 샘플들의 위치를 찾아 저장한다
        self.idx_short    = np.where(labels == 2)[0]  # Returns elements chosen from two arrays by condition
        # 각 클래스의 샘플 수와 배치당 비율을 화면에 출력한다
        print(  # Prints output to stdout
            f"  [BC Sampler] HOLD={len(self.idx_hold):,}  "  # Returns the number of items
            f"LONG={len(self.idx_long):,}  SHORT={len(self.idx_short):,}  "  # Returns the number of items
            f"per_batch HOLD={self.n_hold}/LONG={self.n_long}/SHORT={self.n_short}"
        )

    # 배치 인덱스들을 순서대로 꺼내는 반복자(이터레이터)를 만드는 함수
    def __iter__(self):  # [__iter__] Special / dunder method
        """HOLD=30%, LONG=35%, SHORT=35% 비율로 샘플 → 셔플."""
        # HOLD, LONG, SHORT 각 그룹에서 정해진 수만큼 랜덤으로 뽑아 하나로 합친다
        batch = np.concatenate([  # Concatenates arrays along an axis
            np.random.choice(self.idx_hold,  self.n_hold,  replace=True),   # HOLD에서 무작위로 뽑는다
            np.random.choice(self.idx_long,  self.n_long,  replace=True),   # LONG에서 무작위로 뽑는다
            np.random.choice(self.idx_short, self.n_short, replace=True),   # SHORT에서 무작위로 뽑는다
        ])
        np.random.shuffle(batch)  # 뽑은 샘플들의 순서를 무작위로 섞는다
        return iter(batch)  # 섞인 배치를 반복자로 돌려준다  # Returns a value to the caller

    # 에폭(전체 학습 한 바퀴)당 배치가 몇 개인지 계산하는 함수
    def n_batches(self, n_epochs_worth: int = 1) -> int:  # [n_batches] Function definition
        """에폭당 배치 수 (전체 인덱스 기준)."""
        # 전체 유효 샘플 수를 구한다
        total = len(self.idx_hold) + len(self.idx_long) + len(self.idx_short)  # Returns the number of items
        # 전체 샘플을 배치 크기로 나눈 몫을 돌려준다 (최소 1)
        return max(1, total // self.batch_size)  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# 배치 텐서 구성
# ─────────────────────────────────────────────────────────────────────────────
# 샘플 위치 목록을 받아서 학습에 쓸 입력 텐서와 정답 텐서를 만드는 함수
def build_batch(indices: np.ndarray, features: np.ndarray,  # [build_batch] Function definition
                actions: np.ndarray, device: torch.device):
    """인덱스 리스트 → (x_tensor [B, SEQ_LEN, F], label_tensor [B]) 변환."""
    x_list, y_list = [], []  # 입력 데이터와 정답 데이터를 담을 빈 리스트를 만든다
    n_feat = features.shape[1]  # 피처(특징값) 개수를 가져온다  # Shape (dimensions) of the tensor/array

    # 각 샘플 위치(인덱스)에 대해 반복한다
    for idx in indices:  # Loop: iterate over each item in the sequence
        # 슬라이딩 윈도우: [idx-SEQ_LEN+1 : idx+1] shape [SEQ_LEN, F]
        start = idx - SEQ_LEN + 1  # 윈도우(시계열 창)의 시작 위치를 계산한다
        # 시작 위치가 음수(데이터 범위 밖)이면 앞에 0으로 채운 패딩을 붙인다
        if start < 0:  # Branch: executes only when condition is True
            pad = np.zeros((-start, n_feat), dtype=np.float32)  # 부족한 만큼 0으로 채운 배열을 만든다  # Creates a zero-filled array
            win = np.vstack([pad, features[:idx + 1]])  # 패딩과 실제 데이터를 위아래로 붙인다
        else:  # Branch: all previous conditions were False
            win = features[start: idx + 1]  # 시작부터 현재 위치까지 데이터를 잘라낸다

        x_list.append(win)          # 입력 창(윈도우)을 리스트에 추가한다  # Appends an item to the end of the list
        y_list.append(int(actions[idx]))  # 해당 위치의 정답 행동을 정수로 변환해 추가한다  # Appends an item to the end of the list

    # 파이썬 리스트를 넘파이 배열로 묶은 뒤 파이토치 텐서로 변환해 GPU/CPU로 보낸다
    x = torch.from_numpy(np.array(x_list, dtype=np.float32)).to(device)  # Moves tensor to specified device or dtype
    # 정답 레이블을 정수형 텐서로 만들어 GPU/CPU로 보낸다
    y = torch.tensor(y_list, dtype=torch.long).to(device)  # Converts Python data to a PyTorch tensor
    return x, y  # 입력 텐서와 정답 텐서를 함께 돌려준다  # Returns a value to the caller


# ─────────────────────────────────────────────────────────────────────────────
# 검증 평가
# ─────────────────────────────────────────────────────────────────────────────
# 기울기 계산 없이 실행(메모리 절약)하겠다는 표시
@torch.no_grad()  # Decorator: modifies the function / class below
# 검증 데이터 전체에 대해 모델 정확도를 계산하는 함수
def evaluate_bc(agent, features: np.ndarray, actions: np.ndarray,  # [evaluate_bc] Function definition
                device: torch.device, batch_size: int = 256) -> dict:
    """검증셋 전체에 대한 정확도 + 클래스별 정밀도 계산."""
    agent.eval()  # 모델을 평가(검증) 모드로 전환한다 (드롭아웃 등 비활성화)  # Switches model to evaluation mode (disables Dropout)
    n = len(features)  # 전체 피처 데이터 개수를 가져온다  # Returns the number of items
    valid_start = WARMUP + SEQ_LEN  # 유효 데이터가 시작되는 인덱스를 계산한다

    all_preds, all_labels = [], []  # 모든 예측값과 정답을 모을 빈 리스트를 만든다

    # 유효 시작 위치부터 끝까지 배치 크기씩 반복한다
    for i in range(valid_start, n, batch_size):  # Loop: iterate a fixed number of times
        # 현재 배치의 인덱스 범위를 만든다 (끝 인덱스가 n을 넘지 않도록)
        batch_idx = np.arange(i, min(i + batch_size, n))  # Creates an array of evenly spaced integers
        # 배치 인덱스로 입력 텐서와 정답 텐서를 만든다
        x, y = build_batch(batch_idx, features, actions, device)

        # 모델에 입력을 넣어 예측 로짓(logits)을 얻는다 (마지막 시점만 사용)
        logits, _, _, _ = agent.forward(x, last_step_only=True)
        # 로짓의 차원이 3이면(배치×시간×클래스) 시간 차원을 압축해 배치×클래스로 만든다
        if logits.dim() == 3:  # Branch: executes only when condition is True
            logits = logits.squeeze(1)           # [B, 3]

        # 가장 높은 값의 클래스를 예측 결과로 선택한다
        preds = logits.argmax(dim=-1)
        # 예측 결과를 CPU 넘파이 배열로 변환해 리스트에 추가한다
        all_preds.append(preds.cpu().numpy())  # Moves tensor to CPU memory
        # 정답 레이블도 CPU 넘파이 배열로 변환해 리스트에 추가한다
        all_labels.append(y.cpu().numpy())  # Moves tensor to CPU memory

    # 모든 배치의 예측 결과를 하나의 배열로 합친다
    preds_arr  = np.concatenate(all_preds)  # Concatenates arrays along an axis
    # 모든 배치의 정답을 하나의 배열로 합친다
    labels_arr = np.concatenate(all_labels)  # Concatenates arrays along an axis
    # 전체 정확도를 계산한다 (맞은 것 / 전체)
    total_acc  = (preds_arr == labels_arr).mean()

    per_class = {}  # 클래스별 정확도를 담을 빈 사전을 만든다
    names = {0: "HOLD", 1: "LONG", 2: "SHORT"}  # 클래스 번호와 이름을 연결하는 사전
    # 각 클래스(0=HOLD, 1=LONG, 2=SHORT)에 대해 반복한다
    for cls in [0, 1, 2]:  # Loop: iterate over each item in the sequence
        # 해당 클래스의 정답인 위치만 골라내는 마스크를 만든다
        mask = (labels_arr == cls)
        # 해당 클래스 샘플이 1개 이상 있으면 클래스별 정확도를 계산한다
        if mask.sum() > 0:  # Branch: executes only when condition is True
            per_class[names[cls]] = (preds_arr[mask] == labels_arr[mask]).mean()
        else:  # Branch: all previous conditions were False
            per_class[names[cls]] = 0.0  # 해당 클래스 샘플이 없으면 정확도를 0으로 설정한다

    # Balanced Accuracy = 클래스별 정확도의 산술평균
    # 분포 불일치(covariate shift) 제거: 자연분포(HOLD=60%)에 편향되지 않음
    # 랜덤 baseline = 33.3% (3-class uniform), 모델이 진짜 학습하면 >33.3% 여야 함
    # HOLD, LONG, SHORT 세 클래스의 정확도를 더해서 3으로 나눈 균형 정확도를 계산한다
    bal_acc = (per_class["HOLD"] + per_class["LONG"] + per_class["SHORT"]) / 3.0
    # 결과를 사전 형태로 돌려준다
    return {  # Returns a value to the caller
        "acc":        total_acc,      # 자연분포 정확도 (참고용)
        "bal_acc":    bal_acc,        # Balanced Accuracy (체크포인트 기준)
        "acc_hold":   per_class["HOLD"],   # HOLD 클래스 정확도
        "acc_long":   per_class["LONG"],   # LONG 클래스 정확도
        "acc_short":  per_class["SHORT"],  # SHORT 클래스 정확도
        "n_samples":  len(labels_arr),     # 검증에 사용된 샘플 수  # Returns the number of items
        "dist_pred":  {
            names[c]: (preds_arr == c).mean() for c in [0, 1, 2]  # 예측 결과의 클래스별 비율
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# BC 학습 루프 (1 에폭)
# ─────────────────────────────────────────────────────────────────────────────
# 모델을 한 에폭(전체 데이터 한 바퀴) 동안 학습시키는 함수
def train_one_epoch(agent, sampler: BalancedBatchSampler,  # [train_one_epoch] Function definition
                    features: np.ndarray, actions: np.ndarray,
                    device: torch.device, label_smoothing: float,
                    orth_weight: float = ORTH_WEIGHT,
                    parity_weight: float = 0.10,
                    regret_weight: float = REGRET_W,
                    tp_mult: float = 2.0,  # Take-profit multiplier: exit at TP × ATR
                    sl_mult: float = 1.0) -> dict:  # Stop-loss multiplier: forced exit at SL × ATR
    agent.train()  # 모델을 학습 모드로 전환한다 (드롭아웃 등 활성화)  # Switches model to training mode (enables Dropout, BN)
    n_batches     = sampler.n_batches()  # 이번 에폭에서 처리할 배치 수를 가져온다
    total_loss    = 0.0  # 전체 손실값 누적 변수를 0으로 초기화한다
    total_ce      = 0.0  # 교차 엔트로피 손실 누적 변수를 0으로 초기화한다
    total_regret  = 0.0  # 후회(Regret) 손실 누적 변수를 0으로 초기화한다
    total_orth    = 0.0  # 직교성(Orthogonality) 손실 누적 변수를 0으로 초기화한다
    total_parity  = 0.0  # 대칭성(Parity) 손실 누적 변수를 0으로 초기화한다
    total_correct = 0    # 맞게 예측한 샘플 수를 0으로 초기화한다
    total_samples = 0    # 처리한 전체 샘플 수를 0으로 초기화한다

    # RegretWeightedBCLoss: 기회비용 기반 가중 손실
    # - 틀린 방향 예측 gradient: (TP+SL)× 증폭
    # - HOLD 오예측 gradient: fee× 미미 → 불확실 시 HOLD 유도
    # 후회(Regret) 가중 행동 복제 손실 함수를 만든다
    _loss_fn = RegretWeightedBCLoss(
        tp_mult=tp_mult,           # 이익 실현 배수 (손실 가중에 사용)  # Take-profit multiplier: exit at TP × ATR
        sl_mult=sl_mult,           # 손절 배수 (손실 가중에 사용)  # Stop-loss multiplier: forced exit at SL × ATR
        fee=0.075,                 # 거래 수수료 비율 (0.075%)
        smoothing=label_smoothing, # 레이블 스무딩 강도
        orth_w=orth_weight,        # 직교성 손실 가중치
        parity_w=parity_weight,    # 대칭성 손실 가중치
        regret_w=regret_weight,    # 후회 손실 가중치
    )

    # 배치 수만큼 반복해서 학습한다
    for _ in range(n_batches):  # Loop: iterate a fixed number of times
        # 샘플러에서 한 배치만큼의 인덱스를 꺼낸다
        indices = np.fromiter(sampler, dtype=np.int64, count=sampler.batch_size)
        # 유효 범위 클리핑: 워밍업+시퀀스 길이 이전 인덱스는 사용할 수 없으므로 제거한다
        valid_start = WARMUP + SEQ_LEN  # 유효 시작 인덱스를 계산한다
        indices = indices[indices >= valid_start]  # 유효 범위 안에 있는 인덱스만 남긴다
        # 유효한 인덱스가 하나도 없으면 이 배치를 건너뛴다
        if len(indices) == 0:  # Branch: executes only when condition is True
            continue  # 다음 배치로 넘어간다  # Skip the rest of this iteration

        # 인덱스를 사용해 입력 텐서와 정답 텐서를 만든다
        x, y = build_batch(indices, features, actions, device)

        agent.optimizer.zero_grad()  # 이전 배치의 기울기를 모두 0으로 초기화한다  # Resets gradients to zero before the next backward pass

        # 모델에 입력을 넣어 로짓, 양자 기댓값, 나머지 출력을 얻는다 (마지막 시점만)
        logits, expvals, _, _ = agent.forward(x, last_step_only=True)
        # 로짓의 차원이 3이면 시간 차원을 압축한다
        if logits.dim() == 3:  # Branch: executes only when condition is True
            logits  = logits.squeeze(1)           # [B, 3]: 배치×클래스 형태로 만든다
            expvals = expvals.squeeze(1)          # [B, N_QUBITS]: 배치×큐비트 형태로 만든다

        # 손실 함수에 예측값, 정답, 양자 기댓값을 넣어 손실과 세부 통계를 계산한다
        loss, _loss_stats = _loss_fn(logits, y, expvals)
        # 교차 엔트로피 손실을 누적한다
        total_ce     += _loss_stats["ce"]
        # 후회(Regret) 손실을 누적한다
        total_regret += _loss_stats["regret"]
        # 직교성(Orthogonality) 손실을 누적한다
        total_orth   += _loss_stats["orth"]
        # 대칭성(Parity) 손실을 누적한다
        total_parity += _loss_stats["parity"]

        loss.backward()  # 손실로부터 역전파(backprop)하여 기울기를 계산한다  # Computes gradients via backpropagation

        # 기울기 폭발 방지: 기울기의 최대 크기를 1.0으로 잘라낸다
        torch.nn.utils.clip_grad_norm_(  # Clips gradient norm to prevent exploding gradients
            [p for p in agent.parameters() if p.requires_grad],  # 학습 가능한 파라미터만
            max_norm=1.0  # 기울기 최대 노름(크기)을 1.0으로 제한한다
        )
        # 양자 자연 기울기 옵티마이저를 사용 중인지 확인하기 위해 임포트한다
        # Import DiagonalQNGOptimizer as _DQNG from src.models.qng_optimizer module
        from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
        # 옵티마이저가 양자 자연 기울기 방식이면 양자 레이어 정보를 추가로 넘긴다
        if isinstance(agent.optimizer, _DQNG):  # Branch: executes only when condition is True
            agent.optimizer.step(encoder=agent.encoder.quantum_layer)  # QNG 스텝 실행
        else:  # Branch: all previous conditions were False
            agent.optimizer.step()  # 일반 옵티마이저 스텝 실행  # Updates model parameters using computed gradients

        # 기울기 계산 없이 정확도와 손실을 누적한다
        with torch.no_grad():  # Context: disable gradient tracking for inference (saves memory)
            preds = logits.argmax(dim=-1)  # 예측 클래스를 선택한다
            total_correct += (preds == y).sum().item()  # 맞게 예측한 수를 더한다  # Extracts a Python scalar from a 1-element tensor
            total_samples += len(y)                     # 처리한 샘플 수를 더한다  # Returns the number of items
            total_loss    += loss.item()                # 손실값을 더한다  # Extracts a Python scalar from a 1-element tensor

    # 배치당 평균 손실과 정확도 등을 사전으로 돌려준다
    return {  # Returns a value to the caller
        "loss":   total_loss   / max(n_batches, 1),   # 평균 전체 손실
        "acc":    total_correct / max(total_samples, 1),  # 전체 정확도
        "ce":     total_ce     / max(n_batches, 1),   # 평균 교차 엔트로피 손실
        "regret": total_regret / max(n_batches, 1),   # 평균 후회 손실
        "orth":   total_orth   / max(n_batches, 1),   # 평균 직교성 손실
        "parity": total_parity / max(n_batches, 1),   # 평균 대칭성 손실
    }


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
# 프로그램 전체 흐름을 담당하는 메인 함수
def main():  # [main] Function definition
    # 터미널 실행 시 받을 옵션(인자)들을 정의하는 파서를 만든다
    parser = argparse.ArgumentParser(
        description="Phase 0: Behavior Cloning Pre-training (AlphaGo SL Phase)"
    )
    # 단일 심볼(코인 이름) 옵션: 기본값은 BTCUSDT
    parser.add_argument("--symbol",          default="BTCUSDT",  # Registers a CLI argument for argparse
                        help="단일 심볼 (--symbols 미지정 시 사용)")
    # 여러 심볼을 쉼표로 구분해 지정하는 옵션
    parser.add_argument("--symbols",         default=None,  # Registers a CLI argument for argparse
                        help="멀티심볼 쉼표구분 (예: BTCUSDT,ETHUSDT,SOLUSDT). "
                             "지정 시 --symbol 무시. 학습 데이터 합산.")
    # 봉 크기(타임프레임) 옵션: 기본값은 15분봉
    parser.add_argument("--timeframe",       default="15m")  # Registers a CLI argument for argparse
    # 학습에 사용할 과거 데이터 기간(일수) 옵션: 기본값은 4년(1461일)
    parser.add_argument("--days",   type=int, default=1461)  # 4년 = 2022 약세장 포함  # Registers a CLI argument for argparse
    # 학습 시작 날짜를 직접 지정하는 옵션 (지정하면 --days 무시)
    parser.add_argument("--start-date", dest="start_date", default=None,  # Registers a CLI argument for argparse
                        help="학습 시작일 YYYY-MM-DD (지정 시 --days 무시)")
    # 학습 종료 날짜를 직접 지정하는 옵션 (미지정 시 오늘)
    parser.add_argument("--end-date",   dest="end_date",   default=None,  # Registers a CLI argument for argparse
                        help="학습 종료일 YYYY-MM-DD (미지정 시 오늘)")
    # 학습 반복 횟수(에폭 수) 옵션: 기본값은 20번
    parser.add_argument("--epochs", type=int, default=20)  # Registers a CLI argument for argparse
    # 한 배치에 담을 샘플 수 옵션: 기본값은 192개 (클래스당 64개)
    # Registers a CLI argument for argparse
    parser.add_argument("--batch-size", type=int, default=192)   # 64×3 = 64 per class
    # 학습률(모델이 얼마나 빠르게 배울지) 옵션: 기본값은 0.0001
    parser.add_argument("--lr",     type=float, default=1e-4)  # Registers a CLI argument for argparse
    # 검증셋 비율 옵션: 0과 1 사이 값 (기본값 0.2 = 전체 데이터의 20%)
    parser.add_argument("--val-ratio", type=float, default=0.2,  # Registers a CLI argument for argparse
                        help="검증셋 비율 (0-1)")
    # 학습에 사용할 장치 옵션: GPU가 있으면 cuda, 없으면 cpu 자동 선택
    # Registers a CLI argument for argparse
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # 체크포인트(저장 파일)를 저장할 폴더 경로 옵션
    parser.add_argument("--checkpoint-dir", default="checkpoints/quantum_v2")  # Registers a CLI argument for argparse
    # 검증 정확도가 개선되지 않을 때 학습을 멈출 에폭 수(조기 종료) 옵션
    parser.add_argument("--patience", type=int, default=15,  # Registers a CLI argument for argparse
                        help="검증 정확도 개선 없을 때 Early Stop 에폭 수")
    # 레이블 스무딩 강도 옵션 (기본값은 상수 LABEL_SMOOTH)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTH)  # Registers a CLI argument for argparse
    # 힐베르트 직교성 가중치 옵션 (기본값은 상수 ORTH_WEIGHT)
    parser.add_argument("--orth-weight",     type=float, default=ORTH_WEIGHT,  # Registers a CLI argument for argparse
                        help="Hilbert Orthogonality 가중치 (N=3 qubits R³ centroid 분리)")
    # 패리티 정규화 가중치 옵션 (BC 단계에서는 매우 약하게 설정)
    parser.add_argument("--parity-weight",   type=float, default=0.005,  # Registers a CLI argument for argparse
                        help="Parity Regularizer 가중치 (BC에서는 극히 약하게 0.005 — RL의 act_sym_loss가 HOLD 균형 주담당)")
    # 후회(Regret) 가중치 옵션: 0이면 비활성, 1이면 기회비용 1배 적용
    parser.add_argument("--regret-weight",   type=float, default=REGRET_W,  # Registers a CLI argument for argparse
                        help="Regret 가중치 스케일. 0=비활성(기존 QCLS), 1=기회비용 1× (default), 2=강화")
    # TP(이익 실현) 배수 옵션: Regret 행렬 계산에 사용
    parser.add_argument("--tp-mult",         type=float, default=2.0,  # Registers a CLI argument for argparse
                        # ATR: Average True Range — average price volatility
                        help="TP 배수 (Regret 행렬용, default=2.0=2×ATR)")
    # SL(손절) 배수 옵션: Regret 행렬 계산에 사용
    parser.add_argument("--sl-mult",         type=float, default=1.0,  # Registers a CLI argument for argparse
                        # ATR: Average True Range — average price volatility
                        help="SL 배수 (Regret 행렬용, default=1.0=1×ATR)")
    # 피처(특징값) 버전 옵션: 4=28차원, 5=54차원(Frenet)
    def _fv_type(x):
        try: return int(x)
        except ValueError: return x
    parser.add_argument("--feat-ver", type=_fv_type, default=4, choices=[4, 5, "structural"],
                        help="Feature version: 4=V4 28-dim (default), 5=V5 54-dim, structural=13-dim")
    # 재현성을 위한 랜덤 시드 옵션 (미지정 시 매번 다른 결과)
    parser.add_argument("--seed", type=int, default=None,  # Registers a CLI argument for argparse
                        help="재현성을 위한 랜덤 시드 (미지정 시 랜덤)")
    args = parser.parse_args()  # 입력된 인자들을 파싱(분석)해서 args 객체에 저장한다  # Parses CLI arguments into an args namespace

    # ── 시드 고정 ──────────────────────────────────────────────────────────────
    # 시드(seed)가 지정된 경우에만 랜덤 고정을 실행한다
    if args.seed is not None:  # Branch: executes only when condition is True
        import random  # 파이썬 기본 랜덤 함수를 불러온다  # Import Random number generation
        random.seed(args.seed)           # 파이썬 기본 랜덤 시드를 고정한다
        np.random.seed(args.seed)        # 넘파이 랜덤 시드를 고정한다
        torch.manual_seed(args.seed)     # 파이토치 CPU 랜덤 시드를 고정한다
        # GPU가 있으면 GPU 랜덤 시드도 고정한다
        if torch.cuda.is_available():  # Branch: executes only when condition is True
            torch.cuda.manual_seed_all(args.seed)  # 모든 GPU의 랜덤 시드를 고정한다
        torch.backends.cudnn.deterministic = True   # 결정론적(재현 가능한) 연산을 사용한다
        torch.backends.cudnn.benchmark     = False  # 자동 성능 최적화를 끈다 (재현성 우선)
        print(f"  [Seed] 랜덤 시드 고정: {args.seed} (재현 가능)")  # 시드 고정 완료 메시지를 출력한다  # Prints output to stdout

    os.makedirs(args.checkpoint_dir, exist_ok=True)  # 체크포인트 저장 폴더가 없으면 만든다  # Creates directory (and parents) if they do not exist
    device = torch.device(args.device)  # 학습에 사용할 장치(GPU 또는 CPU) 객체를 만든다  # Target compute device: CUDA GPU or CPU

    # 학습 시작 안내 메시지를 화면에 출력한다
    print("═" * 72)  # Prints output to stdout
    print("  Phase 0: Behavior Cloning Pre-training")  # Prints output to stdout
    print("  DeepMind AlphaGo SL Policy Network 단계")  # Prints output to stdout
    print("  목표: val accuracy > 33% (random baseline ≈33%, BEP=20%)")  # Prints output to stdout
    print("═" * 72)  # Prints output to stdout

    # ── 1. 데이터 로드 (멀티심볼 지원) ──────────────────────────────────────
    # 거래소 데이터를 가져오는 클라이언트 객체를 만든다 (por=실시간 준비 모드)
    dc = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")
    # 종료 날짜가 지정됐으면 해당 날짜를, 아니면 지금 시각을 종료 시간으로 설정한다
    _end_dt = (
        datetime.strptime(args.end_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc, hour=23, minute=59)  # 지정된 날짜의 23시 59분으로 설정
        if args.end_date  # Branch: executes only when condition is True
        else datetime.now(timezone.utc)  # 현재 UTC 시각을 사용한다
    )
    # 시작 날짜가 지정됐으면 그대로, 아니면 종료 날짜에서 지정한 일수를 빼서 계산한다
    _start_str = (
        args.start_date
        if args.start_date  # Branch: executes only when condition is True
        else (_end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")  # 날짜를 문자열로 변환
    )
    # 화면에 표시할 날짜 정보 문자열을 만든다
    _days_info = (
        f"{_start_str} ~ {_end_dt.strftime('%Y-%m-%d')}"  # 시작~종료 날짜 형식
        if args.start_date  # Branch: executes only when condition is True
        else f"{args.days} days"  # 일수 형식
    )

    # 심볼 목록 결정: --symbols가 지정됐으면 쉼표로 나눠 목록을 만들고, 아니면 단일 심볼 사용
    if args.symbols:  # Branch: executes only when condition is True
        _symbol_list = [s.strip() for s in args.symbols.split(",") if s.strip()]  # 쉼표로 분리 후 공백 제거
    else:  # Branch: all previous conditions were False
        _symbol_list = [args.symbol]  # 단일 심볼을 리스트로 감싼다
    # 학습할 심볼, 타임프레임, 기간 정보를 화면에 출력한다
    print(f"\n  [Data] Symbols={_symbol_list}  TF={args.timeframe}  [{_days_info}]")  # Prints output to stdout

    # 심볼별 train/val 분리 후 합산 (시간 누수 방지)
    all_train_feats, all_train_acts = [], []  # 학습용 피처와 행동 리스트 초기화
    all_val_feats,   all_val_acts   = [], []  # 검증용 피처와 행동 리스트 초기화
    # 각 심볼에 대해 데이터를 불러오고 학습/검증으로 나눈다
    for _sym in _symbol_list:  # Loop: iterate over each item in the sequence
        # 해당 심볼의 피처와 행동 데이터를 불러온다
        _feats, _acts = _load_symbol_data_for_bc(
            _sym, args.timeframe, _start_str, _end_dt, dc, feat_ver=args.feat_ver
        )
        # 데이터가 없으면 이 심볼을 건너뛴다
        if _feats is None:  # Branch: executes only when condition is True
            continue  # 다음 심볼로 넘어간다  # Skip the rest of this iteration
        _N     = len(_feats)                        # 전체 샘플 수를 가져온다  # Returns the number of items
        _split = int(_N * (1 - args.val_ratio))     # 학습/검증 분리 인덱스를 계산한다
        all_train_feats.append(_feats[:_split])     # 앞부분을 학습용 피처에 추가한다  # Appends an item to the end of the list
        all_train_acts.append(_acts[:_split])       # 앞부분을 학습용 행동에 추가한다  # Appends an item to the end of the list
        all_val_feats.append(_feats[_split:])       # 뒷부분을 검증용 피처에 추가한다  # Appends an item to the end of the list
        all_val_acts.append(_acts[_split:])         # 뒷부분을 검증용 행동에 추가한다  # Appends an item to the end of the list

    # 유효한 학습 데이터가 하나도 없으면 오류 메시지를 출력하고 함수를 종료한다
    if not all_train_feats:  # Branch: executes only when condition is False
        print("  [Data] ERROR: 유효 데이터 없음"); return  # Prints output to stdout

    # ── 2~4. 멀티심볼 결과 합산 ──────────────────────────────────────────────
    # 여러 심볼의 학습용 피처 배열들을 세로로 합친다 (행을 쌓는다)
    train_feat = np.vstack(all_train_feats)
    # 여러 심볼의 학습용 행동 배열들을 하나로 합친다
    train_act  = np.concatenate(all_train_acts)  # Concatenates arrays along an axis
    # 여러 심볼의 검증용 피처 배열들을 세로로 합친다
    val_feat   = np.vstack(all_val_feats)
    # 여러 심볼의 검증용 행동 배열들을 하나로 합친다
    val_act    = np.concatenate(all_val_acts)  # Concatenates arrays along an axis

    # 클래스 분포 출력: 각 클래스(HOLD/LONG/SHORT)의 샘플 수와 비율을 보여준다
    _total_train = len(train_act)  # 전체 학습 샘플 수를 가져온다  # Returns the number of items
    # HOLD(0), LONG(1), SHORT(2) 각 클래스의 개수와 비율을 출력한다
    for cls, name in [(0,"HOLD"),(1,"LONG"),(2,"SHORT")]:  # Loop: iterate over each item in the sequence
        # Prints output to stdout
        print(f"    [Train] {name}: {(train_act==cls).sum():,} ({(train_act==cls).mean()*100:.1f}%)")
    # 학습/검증 샘플 수와 사용한 심볼 목록을 출력한다
    print(f"  [Split] Train={len(train_feat):,}  Val={len(val_feat):,}  "  # Prints output to stdout
          f"Symbols={_symbol_list}")

    # ── 4.5. 피처 스케일링 (StandardScaler) ──────────────────────────────────
    # 근거: Hurst(0~1), autocorr(−1~1) vs zscore(−π~π) → 스케일 불균형
    # 큰 피처가 VQC rotation gate를 독점 → 미세 신호(Hurst 등) 무시됨
    # train 데이터만으로 fit → val에 transform (data leakage 방지)
    from sklearn.preprocessing import StandardScaler as _SS  # 표준화 스케일러를 불러온다  # Import StandardScaler as _SS from sklearn.preprocessing module
    _scaler = _SS()  # 표준화 스케일러 객체를 만든다 (평균=0, 표준편차=1로 변환)
    # 학습 데이터에 스케일러를 맞추고(fit) 변환한다(transform)
    train_feat = _scaler.fit_transform(train_feat).astype(np.float32)
    # 검증 데이터는 학습 데이터로 맞춘 스케일러로만 변환한다 (데이터 누수 방지)
    val_feat   = _scaler.transform(val_feat).astype(np.float32)
    # 스케일러 저장 경로를 만든다
    _scaler_path = os.path.join(args.checkpoint_dir, "bc_scaler.pkl")  # Joins path components into a single path string
    import pickle  # 파이썬 객체를 파일로 저장하는 라이브러리를 불러온다  # Import pickle library
    # 스케일러 객체를 파일로 저장한다 (나중에 추론 시 같은 스케일러 사용)
    with open(_scaler_path, "wb") as _f:  # Context manager: resource is opened and auto-closed
        pickle.dump(_scaler, _f)  # 스케일러를 이진 파일로 저장한다
    print(f"  [Scaler] StandardScaler 적용 완료 (fit=train) → {_scaler_path}")  # 저장 완료 메시지 출력  # Prints output to stdout

    # ── 5. 에이전트 빌드 ──────────────────────────────────────────────────────
    # 에이전트 설정 객체를 만든다
    config = AgentConfig(
        lr=args.lr,              # 학습률 설정  # Learning rate: step size for each parameter update
        grad_clip=1.0,           # 기울기 최대값 1.0으로 제한
        entropy_reg=0.0,         # BC 단계: 엔트로피 정규화 불필요 (CE가 자체 처리)
        dir_sym_coef=0.0,        # BC 단계: 대칭 페널티 불필요 (balanced sampling이 처리)
        feature_dim=train_feat.shape[1],  # 피처 차원 수 (V4=28, V5=54)  # Shape (dimensions) of the tensor/array
        checkpoint_dir=args.checkpoint_dir,  # 체크포인트 저장 폴더  # Checkpoint: saved model state for resuming training
        confidence_threshold=0.40,           # 신호 최소 신뢰도 기준
    )
    # 설정을 바탕으로 양자 에이전트를 만들어 지정 장치(GPU/CPU)에 올린다
    agent = build_quantum_agent(config=config, device=device)
    # 에이전트 설정 정보를 화면에 출력한다
    print(f"  [Agent] feature_dim={config.feature_dim}  device={device}")  # Prints output to stdout

    # ── 5.4. Koopman precompute (Ridge CV + Sparse EDMD, runs once) ───────────
    _koop_path = os.path.join("data", f"koopman_config_{args.timeframe}.npz")
    _decomposer_pre = agent.encoder.decomposer
    if _decomposer_pre.use_edmd and not os.path.exists(_koop_path):
        print("\n  [Koopman] Precomputing config (Ridge CV + Sparse Dict) ...")
        from src.data.koopman_config import precompute_koopman_config
        precompute_koopman_config(
            X_train   = train_feat,
            n_modes   = config.n_eigenvectors,
            max_terms = min(40, config.feature_dim * 3),
            save_path = _koop_path,
            verbose   = True,
        )
    if _decomposer_pre.use_edmd and os.path.exists(_koop_path):
        _decomposer_pre.load_koopman_precomputed(_koop_path)
        print(f"  [Koopman] Loaded precomputed config → {_koop_path}")

    # ── 5.5. LDA 판별 방향 피팅 ───────────────────────────────────────────────
    # 핵심 버그 수정: fit_lda()가 호출되지 않으면 _lda_fitted=False → SpectralDecomposer가
    # EDMD/PCA 경로로 폴백 → c_kt에 클래스 정보 없음 → classical_head 무의미
    # train_quantum_v2.py에는 fit_lda() 있음, pretrain_bc.py에는 누락됐었음
    #
    # fit_lda() 입력 형식:
    #   X : [N, F] StandardScaler 적용 후 피처 (SpectralDecomposer 입력과 동일 스케일)
    #   y : {0=HOLD, 1=LONG, -1=SHORT} (actions의 2→-1 변환)
    # 에이전트의 인코더에서 주파수 분해기(decomposer)를 가져온다
    _decomposer = agent.encoder.decomposer
    # LDA(선형 판별 분석) 사용 설정이 켜져 있으면 피팅을 실행한다
    if _decomposer.use_lda:  # Branch: executes only when condition is True
        # actions: 0=HOLD, 1=LONG, 2=SHORT  →  fit_lda 기대값: 0=HOLD, 1=LONG, -1=SHORT
        # SHORT(2)를 -1로 바꿔서 LDA가 기대하는 레이블 형식으로 변환한다
        _train_labels_lda = np.where(train_act == 2, -1, train_act)  # 2→-1 변환, {0=HOLD,1=LONG,-1=SHORT}  # Returns elements chosen from two arrays by condition
        print("  [LDA] Fitting 3-binary discriminant directions on training data ...")  # LDA 피팅 시작 메시지 출력
        # 학습 피처와 변환된 레이블로 LDA 판별 방향을 피팅한다
        _decomposer.fit_lda(train_feat, _train_labels_lda)
    else:  # Branch: all previous conditions were False
        print("  [LDA] use_lda=False → EDMD/PCA 경로 사용")  # LDA 비사용 시 안내 메시지 출력

    # ── 6. 균형 배치 샘플러 ───────────────────────────────────────────────────
    # 학습 행동 레이블과 배치 크기를 이용해 균형 배치 샘플러를 만든다
    sampler = BalancedBatchSampler(train_act, batch_size=args.batch_size)

    # ── 7. LR 스케줄러: Cosine Annealing ─────────────────────────────────────
    # 수학적 근거: cosine schedule은 sharp minima를 피하고 flat minima에 수렴
    # (Keskar et al., 2017 - "On Large-Batch Training for DL")
    from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG  # 양자 자연 기울기 옵티마이저 불러오기  # Import DiagonalQNGOptimizer as _DQNG from src.models.qng_optimizer module
    # 양자 자연 기울기 옵티마이저이면 내부의 고전 옵티마이저를 스케줄러 대상으로 설정한다
    _sched_target = (
        agent.optimizer.classical_optimizer  # QNG 옵티마이저의 내부 고전 옵티마이저
        if isinstance(agent.optimizer, _DQNG)  # Branch: executes only when condition is True
        else agent.optimizer  # 일반 옵티마이저 그대로 사용
    )
    # 코사인 어닐링 학습률 스케줄러를 만든다 (에폭이 늘수록 학습률이 코사인 곡선으로 줄어든다)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        _sched_target, T_max=args.epochs, eta_min=1e-5  # T_max=총 에폭 수, eta_min=최소 학습률
    )

    # ── 8. 학습 루프 ──────────────────────────────────────────────────────────
    best_val_acc    = 0.0   # 지금까지 가장 좋은 검증 정확도를 0으로 초기화한다
    best_ckpt_path  = os.path.join(args.checkpoint_dir, "agent_bc_pretrained.pt")  # 최고 모델 저장 경로  # Joins path components into a single path string
    no_improve      = 0     # 개선이 없었던 에폭 수를 0으로 초기화한다
    t0              = time.time()  # 학습 시작 시각을 기록한다
    viz             = TrainingVisualizer(save_dir="reports/viz")  # 학습 시각화 객체를 만든다

    # 학습 진행 표 헤더를 화면에 출력한다
    print(f"\n  {'Epoch':>5}  {'Loss':>8}  {'TrAcc':>7}  "  # Prints output to stdout
          f"{'BalAcc':>7}  {'H%':>5}  {'L%':>5}  {'S%':>5}  {'LR':>8}  Info")
    print("  " + "─" * 68)  # 구분선을 출력한다  # Prints output to stdout
    print("  (BalAcc = (HOLD+LONG+SHORT acc)/3 — 분포 편향 제거, random=33.3%)")  # Prints output to stdout

    # 1번 에폭부터 마지막 에폭까지 반복해서 학습한다
    for epoch in range(1, args.epochs + 1):  # Loop: iterate a fixed number of times
        # ── Train ──────────────────────────────────────────────────────────
        # 한 에폭 동안 학습을 실행하고 결과 통계를 가져온다
        train_info = train_one_epoch(
            agent, sampler, train_feat, train_act, device,  # 에이전트, 샘플러, 데이터, 장치
            args.label_smoothing, args.orth_weight, args.parity_weight,  # 하이퍼파라미터들
            regret_weight=args.regret_weight,  # 후회 가중치
            tp_mult=args.tp_mult,              # TP 배수
            sl_mult=args.sl_mult,              # SL 배수
        )

        # ── Validate ───────────────────────────────────────────────────────
        # 검증 데이터로 모델 성능을 평가하고 결과 통계를 가져온다
        val_info = evaluate_bc(agent, val_feat, val_act, device, batch_size=512)

        # ── LR Step ────────────────────────────────────────────────────────
        scheduler.step()  # 학습률 스케줄러를 한 단계 진행한다 (학습률 업데이트)
        current_lr = scheduler.get_last_lr()[0]  # 현재 학습률 값을 가져온다

        # ── 체크포인트 ─────────────────────────────────────────────────────
        dp = val_info["dist_pred"]  # 예측 클래스 분포(HOLD/LONG/SHORT 비율)를 가져온다
        h_pred_pct = dp["HOLD"]    # 예측 결과에서 HOLD 비율을 가져온다

        # Collapse 감지 (단일 클래스 지배 → 의미없는 accuracy → 저장 거부)
        l_pred_pct = dp["LONG"]    # 예측 결과에서 LONG 비율을 가져온다
        s_pred_pct = dp["SHORT"]   # 예측 결과에서 SHORT 비율을 가져온다
        is_hold_collapse  = h_pred_pct >= 0.80   # HOLD 비율이 80% 이상이면 붕괴로 판단
        is_long_collapse  = l_pred_pct >= 0.80   # LONG 비율이 80% 이상이면 붕괴로 판단
        is_short_collapse = s_pred_pct >= 0.80   # SHORT 비율이 80% 이상이면 붕괴로 판단

        flag = ""  # 이번 에폭의 상태 메시지를 빈 문자열로 초기화한다
        # HOLD만 계속 예측하는 붕괴 상태이면 개선 없음으로 처리하고 저장하지 않는다
        if is_hold_collapse:  # Branch: executes only when condition is True
            no_improve += 1  # 개선 없음 카운터를 1 올린다
            flag = f"⚠ HOLD-collapse ({h_pred_pct:.0%}) — skip"  # 경고 메시지를 설정한다
        # LONG만 계속 예측하는 붕괴 상태이면 개선 없음으로 처리한다
        elif is_long_collapse:  # Branch: previous condition was False, try this one
            no_improve += 1  # 개선 없음 카운터를 1 올린다
            flag = f"⚠ LONG-collapse ({l_pred_pct:.0%}) — skip"  # 경고 메시지를 설정한다
        # SHORT만 계속 예측하는 붕괴 상태이면 개선 없음으로 처리한다
        elif is_short_collapse:  # Branch: previous condition was False, try this one
            no_improve += 1  # 개선 없음 카운터를 1 올린다
            flag = f"⚠ SHORT-collapse ({s_pred_pct:.0%}) — skip"  # 경고 메시지를 설정한다
        # 균형 정확도가 이전 최고보다 0.1%p 이상 향상됐으면 모델을 저장한다
        elif val_info["bal_acc"] > best_val_acc + 0.001:  # Branch: previous condition was False, try this one
            # Balanced Accuracy 기준 저장: 분포 불일치 편향 제거
            best_val_acc = val_info["bal_acc"]  # 최고 균형 정확도를 업데이트한다
            no_improve   = 0                     # 개선 없음 카운터를 0으로 초기화한다
            agent.save_checkpoint(best_ckpt_path)  # 현재 모델을 체크포인트 파일로 저장한다  # Checkpoint: saved model state for resuming training
            flag = "⭐ Best"  # 최고 기록 갱신 표시를 설정한다
        else:  # Branch: all previous conditions were False
            no_improve += 1  # 개선이 없으면 카운터를 1 올린다

        # ── 출력 ───────────────────────────────────────────────────────────
        # 이번 에폭의 학습/검증 결과를 한 줄로 화면에 출력한다
        print(  # Prints output to stdout
            f"  {epoch:5d}  {train_info['loss']:8.4f}  "  # 에폭 번호와 평균 손실
            f"{train_info['acc']:7.1%}  {val_info['bal_acc']:7.1%}  "  # 학습 정확도와 검증 균형 정확도
            f"{h_pred_pct:5.1%}  {dp['LONG']:5.1%}  {dp['SHORT']:5.1%}  "  # HOLD/LONG/SHORT 예측 비율
            f"rgt:{train_info['regret']:.3f}  "  # 후회 손실값
            f"{current_lr:8.2e}  {flag}"  # 현재 학습률과 상태 플래그
        )

        # ── 클래스별 정확도 (개선 있을 때만) ──────────────────────────────
        # 최고 기록이 갱신됐을 때만 클래스별 세부 정확도를 출력한다
        if "Best" in flag:  # Branch: executes only when condition is True
            print(  # Prints output to stdout
                f"         Per-class acc — "
                f"HOLD:{val_info['acc_hold']:.1%}  "   # HOLD 클래스 정확도
                f"LONG:{val_info['acc_long']:.1%}  "   # LONG 클래스 정확도
                f"SHORT:{val_info['acc_short']:.1%}  "  # SHORT 클래스 정확도
                f"NatAcc:{val_info['acc']:.1%}  "       # 자연분포 정확도
                f"(n={val_info['n_samples']:,})"         # 검증 샘플 수
            )

        # ── Visualization (매 epoch) ────────────────────────────────────────
        # 매 에폭마다 학습 그래프를 그려서 파일로 저장한다
        try:  # Try block: attempt code that might raise an exception
            viz_path = viz.plot_bc_epoch(
                epoch=epoch, n_epochs=args.epochs,  # 현재 에폭과 전체 에폭 수  # Number of full passes over the training dataset
                train_info=train_info, val_info=val_info,  # 학습/검증 통계
                lr=current_lr, dp=dp,  # 현재 학습률과 예측 분포  # Learning rate: step size for each parameter update
            )
            print(f"         [viz] → {viz_path}")  # 저장된 그래프 파일 경로를 출력한다  # Prints output to stdout
        except Exception as _ve:  # Except block: handles a raised exception
            # 시각화가 실패해도 학습을 멈추지 않고 경고만 출력한다
            print(f"         [viz] skipped ({_ve})")  # Prints output to stdout

        # ── Early Stop ─────────────────────────────────────────────────────
        # 지정된 patience 횟수 이상 개선이 없으면 조기 종료한다
        if no_improve >= args.patience:  # Branch: executes only when condition is True
            print(f"\n  [EarlyStop] {args.patience} epochs without improvement → stopping.")  # 조기 종료 메시지 출력  # Prints output to stdout
            break  # 학습 루프를 빠져나간다  # Exit the enclosing loop immediately

    elapsed = time.time() - t0  # 전체 학습에 걸린 시간을 계산한다
    # 학습 완료 요약 정보를 화면에 출력한다
    print("\n" + "═" * 72)  # Prints output to stdout
    print(f"  BC Pre-training 완료")  # Prints output to stdout
    print(f"  Best BalAcc       : {best_val_acc:.2%}  (Balanced Accuracy, random=33.3%)")  # Prints output to stdout
    _tp_pct = (train_act == 1).mean() * 100  # 학습 데이터에서 LONG 레이블의 비율을 계산한다
    # Prints output to stdout
    print(f"  Random baseline   : 33.3% (3-class uniform)  [LONG label rate: {_tp_pct:.1f}%]")
    print(f"  Elapsed           : {elapsed/60:.1f}min")  # 경과 시간을 분 단위로 출력한다  # Prints output to stdout
    print(f"  Checkpoint        : {best_ckpt_path}")  # 저장된 체크포인트 파일 경로를 출력한다  # Prints output to stdout

    # ── 결과 해석 ──────────────────────────────────────────────────────────
    print("\n  [분석]")  # Prints output to stdout
    # TP=6×ATR, SL=1.5×ATR 기준: BEP = SL/(TP+SL) = 1.5/7.5 = 20.0%
    # 수수료 포함 실질 BEP ≈ 23.2%
    # 3-class random baseline = 33.3% (uniform), HOLD-only ≈ 60%
    # BalAcc: 랜덤 baseline=33.3%, 의미있는 신호는 35%+
    # 균형 정확도 결과에 따라 다른 해석 메시지를 출력한다
    if best_val_acc > 0.40:  # Branch: executes only when condition is True
        print("  ✅ 40%+ — 강한 균형 신호. LONG/SHORT 모두 학습됨. RL 수렴 기대.")  # 40% 이상: 매우 좋음  # Prints output to stdout
    elif best_val_acc > 0.36:  # Branch: previous condition was False, try this one
        print("  ✅ 36%+ — 유효 신호. 3개 클래스 모두 random 이상 학습됨. RL 진행 가능.")  # 36~40%: 충분히 좋음  # Prints output to stdout
    elif best_val_acc > 0.33:  # Branch: previous condition was False, try this one
        print("  ⚠ 33~36% — 약한 신호. 일부 클래스만 학습됨. RL 수렴 불확실.")  # 33~36%: 약한 신호  # Prints output to stdout
    else:  # Branch: all previous conditions were False
        print("  ❌ <33% — 랜덤 baseline(33.3%)과 동일 또는 이하.")  # 33% 이하: 학습 실패  # Prints output to stdout
        print("     → 시드 변경 또는 피처 엔지니어링 재검토 필요.")  # 다음 조치 안내  # Prints output to stdout

    # 다음 단계로 실행할 RL 학습 명령어를 화면에 출력한다
    print("\n  [다음 단계]")  # Prints output to stdout
    print(f"  python scripts/train_quantum_v2.py --symbol {args.symbol} "  # Prints output to stdout
          f"--timeframe {args.timeframe} \\")
    print(f"    --days {args.days} --rolling-window 360 --n-folds 5 \\")  # Prints output to stdout
    print(f"    --pretrain-ckpt {best_ckpt_path}")  # 저장된 BC 체크포인트를 RL 학습의 시작점으로 사용  # Prints output to stdout
    print("═" * 72)  # 마무리 구분선을 출력한다  # Prints output to stdout


# 이 파일이 직접 실행될 때만 main() 함수를 호출한다 (다른 파일에서 import 시 실행 안 됨)
if __name__ == "__main__":  # Branch: executes only when condition is True
    main()  # 프로그램을 시작하는 메인 함수를 호출한다
