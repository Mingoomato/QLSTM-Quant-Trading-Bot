"""
train_quantum_v2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quantum Trading V2 — Phase 5: 통합 에이전트 워크포워드 학습 파이프라인 (수정본)

주요 개선 사항:
    1. Feature Caching: 27차원 피처 생성 속도 극대화 (.npy 저장/로드)
    2. Data Leakage 방지: 레이블링 전 순수 데이터(df_clean)로 피처 생성
    3. Dynamic Direction: Long(+1) / Short(-1) 양방향 동적 학습
    4. Index 안전성: Out of bounds 방지 패딩 처리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse  # 커맨드라인 인자 파싱 라이브러리를 불러온다  # Import Command-line argument parsing
import os  # 운영체제 기능(파일·폴더 조작) 표준 라이브러리를 불러온다  # Import OS interface — file and directory operations
import shutil  # shutil 라이브러리를 불러온다
import sys  # 파이썬 시스템 정보 및 실행 제어 표준 모듈를 불러온다  # Import Python system and interpreter utilities
import time  # 시간 측정 및 대기 표준 라이브러리를 불러온다  # Import Time measurement and sleep utilities
import torch  # 파이토치 — 딥러닝(인공지능 학습) 핵심 라이브러리를 불러온다  # Import PyTorch — core deep learning library
import numpy as np  # 넘파이(숫자 계산 라이브러리)를 np라는 별명으로 불러온다  # Import NumPy (numerical computation library) as "np"
import pandas as pd  # 판다스(데이터 표 처리 라이브러리)를 pd라는 별명으로 불러온다  # Import Pandas (DataFrame library) as "pd"
from datetime import datetime, timezone, timedelta  # 날짜와 시간 처리 표준 라이브러리에서 datetime, timezone, timedelta를 가져온다  # Import datetime, timezone, timedelta from Date and time handling
from tqdm import tqdm  # 진행 상황 막대(progress bar) 라이브러리에서 tqdm를 가져온다  # Import tqdm from Progress bar for loops

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows CP949 인코딩 픽스
if hasattr(sys.stdout, "reconfigure"):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
    sys.stderr.reconfigure(encoding="utf-8")


# ─── Google Drive 유틸리티 ────────────────────────────────────────────────────

def _mount_drive() -> bool:  # [_mount_drive] 내부 전용 함수 정의
    """Colab 환경에서 Google Drive를 마운트. 이미 마운트됐으면 스킵."""
    try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
        from google.colab import drive  # type: ignore
        if not os.path.ismount("/content/drive"):  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
            drive.mount("/content/drive", force_remount=False)
            print("  [Drive] Google Drive mounted at /content/drive")  # 결과를 화면에 출력한다  # Prints output to stdout
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            print("  [Drive] Google Drive already mounted")  # 결과를 화면에 출력한다  # Prints output to stdout
        return True  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
    except ImportError:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
        return False  # Colab 환경 아님 — 무시


def _sync_to_drive(local_path: str, drive_dir: str, label: str = "") -> None:  # [_sync_to_drive] 내부 전용 함수 정의
    """로컬 체크포인트를 Google Drive 디렉터리로 즉시 복사.

    Args:
        local_path: 복사할 원본 파일 경로 (로컬 런타임)
        drive_dir:  Drive 목적지 디렉터리 (예: /content/drive/MyDrive/quantum_v2)
        label:      로그 출력용 설명 (예: "Fold3 GlobalBest")
    """
    if not drive_dir:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
        return  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
    try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
        os.makedirs(drive_dir, exist_ok=True)  # 필요한 폴더가 없으면 새로 만든다  # Creates directory (and parents) if they do not exist
        dst = os.path.join(drive_dir, os.path.basename(local_path))  # 폴더와 파일 이름을 합쳐 경로를 만든다  # Joins path components into a single path string
        shutil.copy2(local_path, dst)
        tag = f"[{label}] " if label else ""  # 문자열 안에 변수 값을 넣어 만든다
        print(f"  [Drive] {tag}{os.path.basename(local_path)} → {drive_dir}")  # 결과를 화면에 출력한다  # Prints output to stdout
    except Exception as e:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
        print(f"  [Drive] ⚠ 복사 실패 ({local_path} → {drive_dir}): {e}")  # 결과를 화면에 출력한다  # Prints output to stdout

# src.models.integrated_agent 모듈에서 build_quantum_agent, AgentConfig를 가져온다
# Import build_quantum_agent, AgentConfig from src.models.integrated_agent module
from src.models.integrated_agent import build_quantum_agent, AgentConfig
from src.data.data_client import DataClient  # src.data.data_client 모듈에서 DataClient를 가져온다
# src.models.features_v4 모듈에서 generate_and_cache_features_v4를 가져온다
# Import generate_and_cache_features_v4 from src.models.features_v4 module
from src.models.features_v4 import generate_and_cache_features_v4
# src.data.binance_client 모듈에서 fetch_binance_taker_history를 가져온다
# Import fetch_binance_taker_history from src.data.binance_client module
from src.data.binance_client import fetch_binance_taker_history
from src.viz.training_viz import TrainingVisualizer  # src.viz.training_viz 모듈에서 TrainingVisualizer를 가져온다
from src.models.labeling import (  # src.models.labeling 모듈에서 (를 가져온다
    compute_triple_barrier_labels,
    compute_clean_barrier_labels,
    standardize_1m_ohlcv,
)

def compute_optimal_rolling_window(  # [compute_optimal_rolling_window] 함수 정의 시작
    df: pd.DataFrame,
    bpd: int = 96,
    min_trades: int = 200,
    trade_rate_per_bar: float = 0.03,
    n_regime_stds: float = 1.5,
    max_days: int = 360,
    verbose: bool = True,
) -> int:
    """데이터 기반 최적 Rolling Window 자동 계산.

    두 제약의 MAX를 취한다:
      1. Sample Efficiency: N_min 거래 완료에 필요한 최소 일수
         W_trades = min_trades / (trade_rate_per_bar * bpd)

      2. Regime Coherence: Markov 전이 행렬로 측정한 월간 vol 레짐의
         평균 지속기간의 n_regime_stds 배
         W_regime = n_regime_stds * E[L_regime_days]

    ChatGPT 방식과의 차이:
      - ChatGPT: vol 레짐(1일 평균) 또는 트렌드 레짐(임의 45일 추정)에
        2-3× 배수 적용 → 실데이터 기반 아님, 배수 근거 없음
      - 본 함수: 실제 데이터에서 Markov E[L] 직접 추정,
        Sample Efficiency 제약을 동시에 고려

    Returns:
        int: 최적 rolling window (일 단위)
    """
    from itertools import groupby  # 반복자를 조합하는 도구 모듈에서 groupby를 가져온다  # Import groupby from Iterator building blocks
    from collections import defaultdict  # 특수한 자료구조(deque, Counter 등) 표준 라이브러리에서 defaultdict를 가져온다  # Import defaultdict from Specialized container datatypes (deque, Counter)

    close = df["close"].values if "close" in df.columns else df.iloc[:, 4].values
    log_ret = np.log(close[1:] / close[:-1])  # 자연 로그를 계산한다  # Computes natural logarithm element-wise
    n = len(log_ret)  # 로그 수익률: ln(현재 가격 ÷ 이전 가격)  # Log-return: ln(P_t / P_{t-1}) — stationary price change

    # ── 월간 vol 레짐 (30d rolling std) ──────────────────────────────────────
    W30 = 30 * bpd
    vol30 = np.array([  # 파이썬 리스트를 넘파이 배열로 변환한다  # Converts Python sequence to NumPy array
        np.std(log_ret[max(0, i - W30):i]) if i > 0 else 0.0  # 표준편차를 계산한다  # Computes the standard deviation
        for i in range(W30, n + 1)  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
    ])
    p25, p75 = np.percentile(vol30, 25), np.percentile(vol30, 75)  # 백분위수 값을 계산한다  # Computes the q-th percentile of data
    regime30 = np.where(vol30 > p75, 2, np.where(vol30 < p25, 0, 1))  # 조건에 따라 두 값 중 하나를 골라 배열을 만든다  # Returns elements chosen from two arrays by condition

    # Markov 전이 행렬 → E[L] per state
    trans = defaultdict(int)  # 딕셔너리(키-값 쌍)를 만든다
    for i in range(len(regime30) - 1):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
        trans[(regime30[i], regime30[i + 1])] += 1

    E_L_days_per_state = []
    for src in range(3):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
        total_src = sum(trans[(src, dst)] for dst in range(3))  # 지정한 범위의 정수를 하나씩 만들어 낸다  # Generates a sequence of integers
        if total_src > 0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            p_stay = trans[(src, src)] / total_src
            e_l = 1.0 / (1.0 - p_stay + 1e-10) / bpd
            E_L_days_per_state.append(e_l)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list

    # 경험적 run-length 분포에서 median 계산
    run_lengths_days = []
    for k, g in groupby(regime30):  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
        run_lengths_days.append(sum(1 for _ in g) / bpd)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
    run_arr = np.array(run_lengths_days)  # 파이썬 리스트를 넘파이 배열로 변환한다  # Converts Python sequence to NumPy array
    median_L = float(np.median(run_arr))  # 중앙값을 계산한다  # Computes the median value
    mean_L   = float(np.mean(E_L_days_per_state)) if E_L_days_per_state else 20.0  # 평균값을 계산한다  # Computes the mean value

    # ── 두 제약 계산 ───────────────────────────────────────────────────────────
    W_trades  = min_trades / (trade_rate_per_bar * bpd)          # 일 단위
    W_regime  = n_regime_stds * mean_L                            # 일 단위  # Market regime: trending / ranging / volatile
    W_star    = max(W_trades, W_regime)  # 레짐: 현재 시장이 추세장/횡보장/급변동 중 어느 상태인지  # Market regime: trending / ranging / volatile
    W_final   = int(min(W_star, max_days))  # 가장 작은 값을 찾는다
    # 최소 30일 보장
    W_final   = max(W_final, 30)  # 가장 큰 값을 찾는다

    if verbose:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        print(f"\n  [AutoWindow] === 최적 Rolling Window 자동 계산 ===")  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"  [AutoWindow] 데이터: {len(df):,}봉 ({len(df)/bpd:.0f}일)")  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"  [AutoWindow] 월간 vol 레짐 E[L] = {mean_L:.1f}d  (Markov 추정)")  # 결과를 화면에 출력한다  # Prints output to stdout
        print(f"  [AutoWindow]   런 분포 median = {median_L:.1f}d  "  # 결과를 화면에 출력한다  # Prints output to stdout
              f"p75 = {np.percentile(run_arr,75):.1f}d")  # 백분위수 값을 계산한다  # Computes the q-th percentile of data
        print(f"  [AutoWindow] 제약 1 (Sample Efficiency): "  # 결과를 화면에 출력한다  # Prints output to stdout
              f"min_trades={min_trades} @ {trade_rate_per_bar*100:.1f}%/bar "  # 문자열 안에 변수 값을 넣어 만든다
              f"-> W_trades = {W_trades:.0f}d")  # 문자열 안에 변수 값을 넣어 만든다
        print(f"  [AutoWindow] 제약 2 (Regime Coherence): "  # 레짐: 현재 시장이 추세장/횡보장/급변동 중 어느 상태인지  # Market regime: trending / ranging / volatile
              f"{n_regime_stds}x E[L] = {W_regime:.0f}d")  # 레짐: 현재 시장이 추세장/횡보장/급변동 중 어느 상태인지  # Market regime: trending / ranging / volatile
        print(f"  [AutoWindow] W* = max({W_trades:.0f}, {W_regime:.0f}) "  # 레짐: 현재 시장이 추세장/횡보장/급변동 중 어느 상태인지  # Market regime: trending / ranging / volatile
              f"= {W_star:.0f}d -> cap {max_days}d -> {W_final}d")  # 문자열 안에 변수 값을 넣어 만든다
        print(f"  [AutoWindow] ================================================\n")  # 결과를 화면에 출력한다  # Prints output to stdout

    return W_final  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


# [walk_forward_folds] 함수 정의 시작
# [walk_forward_folds] Function definition
def walk_forward_folds(df: pd.DataFrame, n_folds: int = 5, min_train_bars: int = 1000,
                       rolling_bars: int = None):
    """Expanding Window (default) or Rolling Window walk-forward fold generation.

    Args:
        rolling_bars: if set, each fold trains on the most recent `rolling_bars` bars
                      immediately before the validation window (Rolling mode).
                      If None, uses expanding window (all history up to val start).

    Yields: (k, train_df, val_df, tr_idx, va_idx, train_abs, val_abs)
      tr_idx / va_idx  : (first_ts, last_ts) before reset — date display only.
      train_abs / val_abs : (abs_start, abs_end) row indices into the full df —
                            used by the caller to slice all_features correctly.
    """
    total     = len(df)  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
    fold_size = total // (n_folds + 1)

    def _pick_ts(df_s, pos):  # [_pick_ts] 내부 전용 함수 정의
        if "ts" in df_s.columns:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            return df_s["ts"].iloc[pos]  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
        return df_s.index[pos]  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    for k in range(1, n_folds + 1):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
        val_start = k * fold_size
        val_end   = (k + 1) * fold_size

        if rolling_bars is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # ── Rolling Window ─────────────────────────────────────────────
            # Fixed-size training window immediately before the val window.
            # Every fold trains on the same amount of data → no expanding bias.
            train_end   = val_start
            train_start = max(0, train_end - rolling_bars)  # 가장 큰 값을 찾는다
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            # ── Expanding Window (original) ────────────────────────────────
            train_start = 0
            train_end   = val_start
            if train_end < min_train_bars:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                train_end = min(min_train_bars, total - fold_size)  # 가장 작은 값을 찾는다

        train_slice = df.iloc[train_start:train_end]
        val_slice   = df.iloc[val_start:val_end]

        if len(train_slice) < min_train_bars:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # 결과를 화면에 출력한다
            # Prints output to stdout
            print(f"  [Fold {k}] Skipping — train bars {len(train_slice)} < min {min_train_bars}")
            continue  # 현재 반복의 남은 코드를 건너뛰고 다음 반복으로 간다  # Skip the rest of this iteration
        if len(val_slice) < 50:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            continue  # 현재 반복의 남은 코드를 건너뛰고 다음 반복으로 간다  # Skip the rest of this iteration

        tr_idx = (_pick_ts(train_slice, 0),  _pick_ts(train_slice, -1))
        va_idx = (_pick_ts(val_slice,   0),  _pick_ts(val_slice,   -1))

        yield (k,  # 값을 하나씩 내보내는 제너레이터 함수에서 값을 반환한다  # Generator: yields one value to the caller
               train_slice.reset_index(drop=True),
               val_slice.reset_index(drop=True),
               tr_idx, va_idx,
               (train_start, train_end),   # absolute row positions for feature slicing
               (val_start,   val_end))

# features_v3 deprecated 2026-03-23 — moved to src/models/deprecated/features_v3.py

# [prepare_training_data] 함수 정의 시작
# [prepare_training_data] Function definition
def prepare_training_data(df: pd.DataFrame, features: np.ndarray, seq_len: int = 20):
    """데이터프레임과 피처를 모델 학습용 텐서 딕셔너리로 변환."""
    n = len(df)  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
    if n < seq_len + 35:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        return None  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    prices = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    atrs = (df['high'] - df['low']).rolling(14).mean().bfill().values

    # 원시 레이블 (1: LONG TP signal, -1: SHORT TP signal, 0: HOLD)
    raw_labels   = df['label'].values
    # 방향별 세부 레이블 (evaluate_model 손익 계산용)
    long_labels  = df['long_label'].values  if 'long_label'  in df.columns else raw_labels.copy()
    short_labels = df['short_label'].values if 'short_label' in df.columns else raw_labels.copy()

    return {  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
        "features":     features,
        "prices":       prices,
        "highs":        highs,
        "lows":         lows,
        "raw_labels":   raw_labels,
        "long_labels":  long_labels,
        "short_labels": short_labels,
        "atr":          atrs,  # ATR: 가격의 평균 변동 폭 (Average True Range)
        "ts":           df['ts'].values,
        "n":            n
    }

class AdaptiveController:  # ★ [AdaptiveController] 클래스 정의 — 관련 데이터와 기능을 하나로 묶은 설계도
    """
    매 에폭 로그 지표를 읽어 entropy_reg / lindblad_threshold / lr 을 자동 조정.

    - entropy_reg  : mP 분포 불균형(max-min)이 크면 올리고, 균일하면 낮춤.
    - lindblad_thr : fold-end gate pass rate 기반으로 올리거나 낮춤.
    - lr           : WR이 N 에폭 연속 정체하면 LR을 절반으로 줄임.

    변경이 있을 때만 [AutoTune] 라인을 출력해 로그를 오염시키지 않는다.
    """
    # ── 불균형 기준 ──────────────────────────────────────────────────────
    IMBAL_HIGH     = 0.18   # imbalance > 이값 → entropy 올림 (1.25×)
    IMBAL_MID      = 0.12   # imbalance > 이값 → entropy 살짝 올림 (1.08×)
    IMBAL_LOW      = 0.05   # imbalance < 이값 → entropy 낮춤 (0.90×)
    ENTROPY_MAX    = 0.20   # 상한 올림: --entropy-reg 0.12 허용
    ENTROPY_MIN    = 0.03   # 하한 올림: 탐색 최소 보장 (0.005→0.03)

    # ── Lindblad 기준 ────────────────────────────────────────────────────
    LIND_PASS_TARGET = 0.80  # 목표 pass-rate (80 %)
    LIND_PASS_FLOOR  = 0.60  # 이 미만이면 대폭 올림
    LIND_PASS_CEIL   = 0.95  # 이 초과하면 약간 낮춤
    LIND_THR_MAX     = 0.999
    LIND_THR_MIN     = 0.50

    # ── LR 정체 기준 ─────────────────────────────────────────────────────
    LR_STALL_EPOCHS  = 4     # WR이 이 에폭 동안 개선 없으면 LR × 0.5
    LR_MIN           = 5e-5

    def __init__(self, agent):  # [__init__] 초기화 메서드 — 객체를 만들 때 가장 먼저 실행된다
        self.agent        = agent
        self._wr_history  : list[float] = []
        self._stall_count : int         = 0
        self._best_wr     : float       = -1.0

    # ------------------------------------------------------------------
    def step_epoch(self, diag: dict) -> None:  # [step_epoch] 함수 정의 시작
        """에폭 평가 직후 호출. entropy_reg / lr 자동 조정."""
        cfg = self.agent.config
        changes: list[str] = []

        # 1) entropy_reg — mP 불균형 기반
        mp_h = diag.get("mean_p_hold",  0.333)
        mp_l = diag.get("mean_p_long",  0.333)
        mp_s = diag.get("mean_p_short", 0.333)
        imbalance = max(mp_h, mp_l, mp_s) - min(mp_h, mp_l, mp_s)  # 가장 큰 값을 찾는다
        n_short_post = diag.get("short_n", 0)  # Post[S] — gate 통과 SHORT 수
        old_er = cfg.entropy_reg  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
        if imbalance > self.IMBAL_HIGH:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            cfg.entropy_reg = min(cfg.entropy_reg * 1.25, self.ENTROPY_MAX)  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
        elif imbalance > self.IMBAL_MID:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
            cfg.entropy_reg = min(cfg.entropy_reg * 1.08, self.ENTROPY_MAX)  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
        elif imbalance < self.IMBAL_LOW and n_short_post > 0:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
            # SP=0% 상태에서는 낮추지 않음 — 낮추면 VQC SHORT bias 억제가 풀려 S 소멸
            cfg.entropy_reg = max(cfg.entropy_reg * 0.90, self.ENTROPY_MIN)  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
        if abs(cfg.entropy_reg - old_er) > 1e-5:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            changes.append(  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                f"entropy_reg {old_er:.4f}→{cfg.entropy_reg:.4f}"  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
                f" (imbal={imbalance:.3f})"  # 문자열 안에 변수 값을 넣어 만든다
            )

        # 1b) Fisher threshold — 거래가 차단될 때 자동 완화
        n_long_post  = diag.get("long_n", 0)
        total_post   = n_short_post + n_long_post
        old_fish = getattr(cfg, "fisher_threshold_min", 0.38)  # 객체의 속성 값을 동적으로 가져온다  # Gets a named attribute from an object dynamically
        if total_post == 0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # LP=0 AND SP=0: 전방향 차단 → 더 적극적으로 낮춤 (-0.02/epoch)
            new_fish = max(old_fish - 0.02, 0.33)  # 가장 큰 값을 찾는다
            reason = "LP=SP=0 blocked"
        elif n_short_post == 0 and n_long_post > 0:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
            # SP=0%: 낮추면 저신뢰 LONG 폭발 → WR=base rate 고착. 현행 유지.
            # logit_bias_reg가 SHORT bias를 복원하도록 대기.
            new_fish = old_fish
            reason = ""
        elif n_long_post == 0 and n_short_post > 0:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
            # LP=0%: 낮추면 저신뢰 SHORT 폭발 → WR=base rate 고착. 현행 유지.
            new_fish = old_fish
            reason = ""
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            new_fish = old_fish
            reason = ""
        if reason and abs(new_fish - old_fish) > 1e-5:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            cfg.fisher_threshold_min = new_fish
            changes.append(f"fisher_min {old_fish:.3f}→{new_fish:.3f} ({reason})")  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list

        # 2) lr — WR 정체 기반
        wr = diag.get("win_rate", 0.0) if "win_rate" in diag else 0.0
        # win_rate 키가 없으면 long_prec/short_prec 평균으로 추정
        if wr == 0.0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            lp = diag.get("long_prec",  0.0)
            sp = diag.get("short_prec", 0.0)
            ln = diag.get("long_n",     0)
            sn = diag.get("short_n",    0)
            total = ln + sn
            wr = (lp * ln + sp * sn) / max(total, 1) / 100.0  # 가장 큰 값을 찾는다
        if wr > self._best_wr + 0.002:   # 0.2 % 이상 개선이어야 reset  # Branch: executes only when condition is True
            self._best_wr     = wr
            self._stall_count = 0
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            self._stall_count += 1
        if self._stall_count >= self.LR_STALL_EPOCHS:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            old_lr = cfg.lr
            cfg.lr = max(cfg.lr * 0.5, self.LR_MIN)  # 가장 큰 값을 찾는다
            # optimizer의 실제 lr도 조정
            for pg in self.agent.optimizer.param_groups:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                pg["lr"] = cfg.lr
            self._stall_count = 0   # reset
            if abs(cfg.lr - old_lr) > 1e-9:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                changes.append(  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                    f"lr {old_lr:.2e}→{cfg.lr:.2e}"  # 문자열 안에 변수 값을 넣어 만든다
                    f" (WR stall {self.LR_STALL_EPOCHS} ep)"  # 문자열 안에 변수 값을 넣어 만든다
                )

        if changes:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            print(f"  [AutoTune] {' | '.join(changes)}")  # 결과를 화면에 출력한다  # Prints output to stdout

    # ------------------------------------------------------------------
    def step_fold(self, gate_stats: dict) -> None:  # [step_fold] 함수 정의 시작
        """fold-end prod_eval 직후 호출. lindblad_threshold 자동 조정."""
        cfg = self.agent.config
        n_raw    = gate_stats.get("n_raw_trade", 0)
        n_lind   = gate_stats.get("gate_lindblad_blk", 0)  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
        if n_raw == 0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            return  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
        pass_rate = 1.0 - (n_lind / n_raw)
        old_thr   = getattr(cfg, "lindblad_regime_threshold", 0.90)  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
        new_thr   = old_thr
        if pass_rate < self.LIND_PASS_FLOOR:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            new_thr = min(old_thr + 0.04, self.LIND_THR_MAX)  # 가장 작은 값을 찾는다
        elif pass_rate < self.LIND_PASS_TARGET:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
            new_thr = min(old_thr + 0.01, self.LIND_THR_MAX)  # 가장 작은 값을 찾는다
        elif pass_rate > self.LIND_PASS_CEIL:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
            new_thr = max(old_thr - 0.01, self.LIND_THR_MIN)  # 가장 큰 값을 찾는다
        if abs(new_thr - old_thr) > 1e-5:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            cfg.lindblad_regime_threshold = new_thr  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
            print(  # 결과를 화면에 출력한다  # Prints output to stdout
                f"  [AutoTune] lindblad_thr {old_thr:.3f}→{new_thr:.3f}"  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
                f" (pass={pass_rate:.0%}, blk={n_lind}/{n_raw})"  # 문자열 안에 변수 값을 넣어 만든다
            )


def evaluate_model(agent, val_data, seq_len=20, batch_size=128):  # [evaluate_model] 함수 정의 시작
    """검증 셋 인퍼런스 및 성능 평가.

    Fix #1 (WinRate=0): 학습 중 평가에는 낮은 임계값(0.35)을 사용한다.
    agent.config.confidence_threshold(0.55)는 미학습 모델에서 3-class
    softmax max_prob ≈ 0.33~0.42 수준이므로 모든 샘플이 Hold로 필터링됨.
    학습 진행도를 측정하려면 낮은 임계값이 필요하다.
    """
    # 학습 중 평가 전용 임계값
    # BC fine-tune 모드: VQC 재초기화 없음 → BC AvgMaxProb(≈0.42) 유지
    # near-identity: mean_prob≈0.34 → 0.35 필요
    # fine-tune: mean_prob≈0.42 → 0.45로 올려 더 선별적 평가 가능
    EVAL_CONF_THRESHOLD = 0.45

    agent.eval()  # 모델을 평가 모드로 전환한다 (Dropout 비활성화)  # Switches model to evaluation mode (disables Dropout)
    n_samples = val_data["n"]
    indices = np.arange(seq_len + 120, n_samples - 35)  # 지정한 범위의 숫자들로 배열을 만든다  # Creates an array of evenly spaced integers

    if len(indices) == 0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        return 0.0, 0.0, {}, []  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    total_trades = 0
    tp_hits = 0  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
    sl_hits = 0  # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지
    total_pnl_pct = 0.0
    pnl_series = []
    # diagnostics — post-threshold action counts
    n_long = 0
    n_short = 0
    n_hold_filtered = 0
    n_raw_tp = 0
    n_raw_sl = 0
    n_raw_neutral = 0
    # A1: mean softmax probabilities (probability-level collapse detector)
    sum_p = np.zeros(3, dtype=np.float64)   # [Hold, Long, Short]
    n_prob_samples = 0
    # A2: pre-threshold action distribution (raw argmax before confidence gate)
    n_before = np.zeros(3, dtype=np.int64)  # [Hold, Long, Short]
    # A3: per-direction precision (n_wins / n_taken, separately for L and S)
    n_dir_wins  = np.zeros(2, dtype=np.int64)   # [Long wins, Short wins]
    n_dir_taken = np.zeros(2, dtype=np.int64)   # [Long taken, Short taken]

    eta      = agent.config.eta_base
    leverage = agent.config.leverage  # 레버리지: 실제 증거금의 몇 배로 거래하는지
    # fee_pct는 notional 기준 비율 (tp_pct/sl_pct와 동일 단위).
    # 왕복 수수료 = entry(maker) + exit(taker).
    fee_pct      = eta + eta   # round-trip: 0.0005 + 0.0005 = 0.001 (0.1% of notional)
    # 실제 ATR 기반 TP/SL: 레이블 생성 시 사용한 배리어와 동일한 공식.
    # 고정값(tp=0.020, sl=0.015)은 ATR 변동성 무시로 조용한 구간은 과대평가,
    # 격렬한 구간은 과소평가하는 측정 오류를 유발함.
    # 레이블 생성과 동일한 배리어 (alpha=4.0 / beta=1.5, R:R=2.67:1)
    # BC 레이블과 불일치하면 1.0~1.49×ATR 역행 후 TP 회복 구간을 SL로 잘못 처리
    # → LONG 포지션 집중 타격 → LONG 소멸 원인
    BARRIER_ALPHA = 4.0    # TP = alpha × ATR (레이블 alpha와 동일)
    BARRIER_BETA  = 1.5    # SL = beta  × ATR (레이블 beta와 동일, 1.0→1.5 수정)

    with torch.no_grad():  # 메모리 절약을 위해 기울기 계산 없이 추론만 실행한다  # Context: disable gradient tracking for inference (saves memory)
        for start_idx in range(0, len(indices), batch_size):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            batch_indices = indices[start_idx:start_idx + batch_size]

            x_list, raw_labels_list, long_labels_list, short_labels_list, atr_list, price_list = [], [], [], [], [], []
            for idx in batch_indices:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                x_list.append(val_data["features"][idx - seq_len + 1 : idx + 1])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                raw_labels_list.append(val_data["raw_labels"][idx])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                long_labels_list.append(val_data["long_labels"][idx])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                short_labels_list.append(val_data["short_labels"][idx])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                atr_list.append(val_data["atr"][idx])  # ATR: 가격의 평균 변동 폭 (Average True Range)
                price_list.append(val_data["prices"][idx])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list

            # 텐서를 실수형(float32)으로 변환한다
            x_val        = torch.from_numpy(np.array(x_list)).float().to(agent.device)  # Casts tensor to float32
            raw_l_val    = np.array(raw_labels_list)  # 파이썬 리스트를 넘파이 배열로 변환한다  # Converts Python sequence to NumPy array
            # Converts Python sequence to NumPy array
            long_l_val   = np.array(long_labels_list)   # +1=LONG TP, -1=LONG SL, 0=timeout
            # Converts Python sequence to NumPy array
            short_l_val  = np.array(short_labels_list)  # -1=SHORT TP, +1=SHORT SL, 0=timeout
            # 텐서를 실수형(float32)으로 변환한다
            a_val    = torch.from_numpy(np.array(atr_list)).float().to(agent.device)  # Casts tensor to float32

            logits, _, _, _ = agent.forward(x_val, last_step_only=True)
            if logits.dim() == 3:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                logits = logits.squeeze(1)  # 크기가 1인 차원을 없앤다  # Removes dimensions of size 1

            # temp_scaler를 우회 — 학습 중 평가에선 raw softmax 사용.
            # T = T_base*(1+beta*ATR) 가 crypto의 높은 ATR로 인해 T>>1이 되면
            # 모든 확률이 0.33 근처로 수렴해 threshold 필터링에 걸려 Hold처럼 보임.
            # raw softmax로 실제 모델이 방향을 학습하고 있는지 직접 측정.
            probs = torch.softmax(logits, dim=-1)  # 가장 큰 값을 찾는다
            max_probs, actions = probs.max(dim=-1)  # 가장 큰 값을 찾는다

            actions   = actions.cpu().numpy()  # 텐서를 CPU 메모리로 옮긴다  # Moves tensor to CPU memory
            max_probs = max_probs.cpu().numpy()  # 텐서를 CPU 메모리로 옮긴다  # Moves tensor to CPU memory

            conf_mask     = max_probs >= EVAL_CONF_THRESHOLD
            final_actions = np.where(conf_mask, actions, 0)  # 조건에 따라 두 값 중 하나를 골라 배열을 만든다  # Returns elements chosen from two arrays by condition

            n_hold_filtered += int(np.sum((actions != 0) & ~conf_mask))  # 배열의 합계를 구한다  # Sums array elements
            n_long           += int(np.sum(final_actions == 1))  # 배열의 합계를 구한다  # Sums array elements
            n_short          += int(np.sum(final_actions == 2))  # 배열의 합계를 구한다  # Sums array elements
            n_raw_tp         += int(np.sum(raw_l_val == 1))  # 배열의 합계를 구한다  # Sums array elements
            n_raw_sl         += int(np.sum(raw_l_val == -1))  # 배열의 합계를 구한다  # Sums array elements
            n_raw_neutral    += int(np.sum(raw_l_val == 0))  # 배열의 합계를 구한다  # Sums array elements

            # A1: accumulate softmax probability sums for mean computation
            probs_np = probs.cpu().numpy()          # [B, 3]
            sum_p           += probs_np.sum(axis=0) # element-wise sum per class  # 배열의 합계를 구한다  # Sums array elements
            n_prob_samples  += len(probs_np)  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items

            # A2: pre-threshold raw argmax distribution
            pre_actions = np.argmax(probs_np, axis=-1)   # no confidence gate
            for cls in range(3):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
                n_before[cls] += int(np.sum(pre_actions == cls))  # 배열의 합계를 구한다  # Sums array elements

            for i, act in enumerate(final_actions):  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                if act == 0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                    continue  # 현재 반복의 남은 코드를 건너뛰고 다음 반복으로 간다  # Skip the rest of this iteration

                total_trades += 1
                current_pnl = 0.0

                # 실제 ATR 기반 TP/SL — LONG과 SHORT 모두 동일한 크기
                #   TP = alpha×ATR (= 6×ATR)  for both LONG and SHORT
                #   SL = beta×ATR  (= 1.5×ATR) for both LONG and SHORT
                _atr   = float(atr_list[i])  # 실수(소수)로 변환한다
                _price = max(float(price_list[i]), 1e-8)  # 가장 큰 값을 찾는다
                _tp = BARRIER_ALPHA * _atr / _price   # TP pct for both directions
                _sl = BARRIER_BETA  * _atr / _price   # SL pct for both directions

                if act == 1:  # Long
                    n_dir_taken[0] += 1
                    rl = long_l_val[i]   # +1=LONG TP, -1=LONG SL, 0=timeout
                    if rl == 1:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        tp_hits += 1  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
                        n_dir_wins[0] += 1
                        current_pnl = (_tp - fee_pct)
                    elif rl == -1:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
                        sl_hits += 1  # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지
                        current_pnl = (-_sl - fee_pct)
                    else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                        current_pnl = -fee_pct
                elif act == 2:  # Short
                    n_dir_taken[1] += 1
                    # ATR: Average True Range — average price volatility
                    rl = short_l_val[i]  # -1=SHORT TP (7×ATR down), +1=SHORT SL (3×ATR up), 0=timeout
                    if rl == -1:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        tp_hits += 1  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
                        n_dir_wins[1] += 1
                        current_pnl = (_tp - fee_pct)
                    elif rl == 1:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
                        sl_hits += 1  # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지
                        current_pnl = (-_sl - fee_pct)
                    else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                        current_pnl = -fee_pct
                
                total_pnl_pct += current_pnl
                ts = val_data["ts"][batch_indices[i]]
                pnl_series.append({"ts": ts, "pnl_pct": current_pnl * leverage})  # 레버리지: 실제 증거금의 몇 배로 거래하는지  # Appends an item to the end of the list


    # Compute derived metrics
    mean_p = sum_p / max(1, n_prob_samples)          # [Hold, Long, Short]
    long_prec  = n_dir_wins[0] / max(1, n_dir_taken[0]) * 100  # 가장 큰 값을 찾는다
    short_prec = n_dir_wins[1] / max(1, n_dir_taken[1]) * 100  # 가장 큰 값을 찾는다

    _zero_diag = {
        "long": 0, "short": 0, "hold_filtered": 0,
        "raw_tp": 0, "raw_sl": 0, "raw_neutral": 0,
        "before_long": int(n_before[1]), "before_short": int(n_before[2]),  # 정수로 변환한다
        "before_hold": int(n_before[0]),  # 정수로 변환한다
        "mean_p_hold": round(float(mean_p[0]), 3),  # 숫자를 반올림한다
        "mean_p_long": round(float(mean_p[1]), 3),  # 숫자를 반올림한다
        "mean_p_short": round(float(mean_p[2]), 3),  # 숫자를 반올림한다
        "long_prec": 0.0, "long_n": 0,
        "short_prec": 0.0, "short_n": 0,
        "net_bal_pct": 0.0,
    }
    if total_trades == 0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        return 0.0, 0.0, _zero_diag, []  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    win_rate      = tp_hits / total_trades  # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
    ev_per_trade  = (total_pnl_pct * leverage) / total_trades   # net EV per trade
    net_bal_pct   = total_pnl_pct * leverage * 100              # cumulative return % on capital

    diag = {
        # post-threshold counts
        "long":          int(n_long),  # 정수로 변환한다
        "short":         int(n_short),  # 정수로 변환한다
        "hold_filtered": int(n_hold_filtered),  # 정수로 변환한다
        "raw_tp":        int(n_raw_tp),  # 정수로 변환한다
        "raw_sl":        int(n_raw_sl),  # 정수로 변환한다
        "raw_neutral":   int(n_raw_neutral),  # 정수로 변환한다
        # A2: pre-threshold raw argmax distribution
        "before_long":   int(n_before[1]),  # 정수로 변환한다
        "before_short":  int(n_before[2]),  # 정수로 변환한다
        "before_hold":   int(n_before[0]),  # 정수로 변환한다
        # A1: mean softmax probabilities
        "mean_p_hold":   round(float(mean_p[0]), 3),  # 숫자를 반올림한다
        "mean_p_long":   round(float(mean_p[1]), 3),  # 숫자를 반올림한다
        "mean_p_short":  round(float(mean_p[2]), 3),  # 숫자를 반올림한다
        # A3: per-direction precision
        "long_prec":     round(long_prec,  1),  # 숫자를 반올림한다
        "long_n":        int(n_dir_taken[0]),  # 정수로 변환한다
        "short_prec":    round(short_prec, 1),  # 숫자를 반올림한다
        "short_n":       int(n_dir_taken[1]),  # 정수로 변환한다
        "net_bal_pct":   round(net_bal_pct, 2),  # 숫자를 반올림한다
    }
    return ev_per_trade, win_rate, diag, pnl_series  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


# [prod_eval_quick] 함수 정의 시작
# [prod_eval_quick] Function definition
def prod_eval_quick(agent, val_data, seq_len: int = 20, n_sample: int = 50) -> dict:
    """
    Fold-end production evaluation diagnostic (Phase 2).

    Runs n_sample bars through TWO paths and compares:
      (1) Raw model   : forward() + argmax  — what the model WANTS to do
      (2) Prod gates  : select_action()     — what actually fires in deployment

    The difference (gate attrition) answers: "is the model being silenced by
    CR filter / Fisher-Rao / Lindblad / entropy-production gates?"

    Called once at fold-end (after best-checkpoint rollback), NOT every epoch.
    Diagnostic only — never drives checkpointing.
    Returns gate_stats dict for AdaptiveController.step_fold().
    """
    agent.eval()  # 모델을 평가 모드로 전환한다 (Dropout 비활성화)  # Switches model to evaluation mode (disables Dropout)
    n     = val_data["n"]
    # Sample from the latter half of the val set for OOS realism
    start = max(seq_len + 120, n // 2)  # 가장 큰 값을 찾는다
    end   = n - 35
    if end - start < n_sample:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        print("  [ProdEval] val set too small for prod eval — skipped.")  # 결과를 화면에 출력한다  # Prints output to stdout
        return  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    indices = np.sort(  # 배열을 오름차순으로 정렬한다  # Returns a sorted copy of an array
        np.random.choice(np.arange(start, end), size=n_sample, replace=False)  # 지정한 범위의 숫자들로 배열을 만든다  # Creates an array of evenly spaced integers
    )

    gated = [0, 0, 0]   # HOLD / LONG / SHORT  — select_action() (all gates)
    raw   = [0, 0, 0]   # HOLD / LONG / SHORT  — raw forward()   (no gates)

    # Per-gate diagnostic counters (P5: independent check — not elif chain)
    gate_lindblad_blk = 0  # Lindblad regime blocks  (regime_prob > 0.7)
    gate_cr_blk       = 0  # Cramér-Rao filter blocks (H/purity/SNR fail)
    gate_ep_blk       = 0  # Entropy production blocks (Ṡ < threshold)
    gate_fisher_blk   = 0  # Fisher-Rao confidence threshold blocks
    max_probs_list    = []  # raw softmax max_prob per sample

    with torch.no_grad():  # 메모리 절약을 위해 기울기 계산 없이 추론만 실행한다  # Context: disable gradient tracking for inference (saves memory)
        for idx in indices:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
            x_win = val_data["features"][idx - seq_len + 1: idx + 1]
            if len(x_win) < seq_len:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                # 0으로 채워진 배열을 만든다
                # Creates a zero-filled array
                pad   = np.zeros((seq_len - len(x_win), agent.config.feature_dim), dtype=np.float32)
                x_win = np.vstack([pad, x_win])
            x_t      = torch.from_numpy(x_win).float().unsqueeze(0).to(agent.device)  # 새로운 차원을 추가한다  # Inserts a new dimension of size 1
            price    = float(val_data["prices"][idx])  # 실수(소수)로 변환한다
            atr_norm = float(val_data["atr"][idx]) / max(price, 1e-8)  # ATR: 가격의 평균 변동 폭 (Average True Range)

            # (1) Full production path — all gates active
            act_g, _, _ = agent.select_action(x_t, atr_norm=atr_norm, mode="greedy")
            gated[act_g] += 1

            # (2) Raw model preference — bypass every gate
            logits, expvals, _, _ = agent.forward(x_t, last_step_only=True)
            if logits.dim() == 3:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                logits = logits.squeeze(1)  # 크기가 1인 차원을 없앤다  # Removes dimensions of size 1
            probs_raw = torch.softmax(logits, dim=-1)[0]  # 가장 큰 값을 찾는다
            max_prob  = float(probs_raw.max().item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            act_r     = int(probs_raw.argmax().item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
            raw[act_r] += 1
            max_probs_list.append(max_prob)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list

            # Gate-level diagnostic: all 4 gates checked INDEPENDENTLY
            # (Unlike select_action's elif chain, here each gate is counted
            # regardless of whether an earlier gate already blocked. This gives
            # the full attrition picture — which gates are actually firing.)
            if act_r != 0:   # raw model wanted to trade
                log_rets_g = x_t[0, :, 0].cpu().float().numpy()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32

                # Gate 1: Lindblad regime check
                # select_action 수정과 동일하게: threshold = lindblad_regime_threshold (0.90)
                purity_for_cr = 1.0  # CR filter에는 purity=1.0 고정 (select_action과 동일)
                try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
                    if agent.lindblad is not None and agent.config.use_lindblad:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        _, _, regime_prob_t = agent.lindblad(expvals)  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
                        # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식
                        # Lindblad master equation: quantum decoherence model
                        lind_thr = getattr(agent.config, "lindblad_regime_threshold", 0.90)
                        if float(regime_prob_t.mean().item()) > lind_thr:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                            gate_lindblad_blk += 1  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
                except Exception:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
                    pass  # 아무것도 하지 않는다 (빈 블록 자리 채우기)  # No-op placeholder

                # Gate 2: Cramér-Rao selective entry filter
                # purity_for_cr=1.0 고정 (select_action fix와 동일 — Lindblad untrained)
                try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
                    if agent.cr_filter is not None and agent.config.use_cr_filter:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        hurst_val  = float(agent.hurst_est._hurst_single(  # 실수(소수)로 변환한다
                            x_t[0, :, 0].cpu().float()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
                        ).item())  # 텐서에서 파이썬 숫자 하나를 꺼낸다  # Extracts a Python scalar from a 1-element tensor
                        cr_result = agent.cr_filter.check(log_rets_g, hurst_val, purity_for_cr)
                        if not cr_result.allow_entry:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
                            gate_cr_blk += 1
                except Exception:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
                    pass  # 아무것도 하지 않는다 (빈 블록 자리 채우기)  # No-op placeholder

                # Gate 3: Entropy production gate (ep_threshold=0.0 → should never fire)
                try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
                    if agent.ep_estimator is not None and agent.config.use_entropy_prod:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        agent.ep_estimator.compute()
                        if not agent.ep_estimator.allows_entry():  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
                            gate_ep_blk += 1
                except Exception:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
                    pass  # 아무것도 하지 않는다 (빈 블록 자리 채우기)  # No-op placeholder

                # Gate 4: Fisher-Rao confidence threshold
                try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
                    if agent.config.use_fisher_threshold:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        fish_thr = agent._compute_fisher_threshold(log_rets_g)
                    else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                        fish_thr = agent.config.confidence_threshold
                    if max_prob < fish_thr:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        gate_fisher_blk += 1
                except Exception:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
                    pass  # 아무것도 하지 않는다 (빈 블록 자리 채우기)  # No-op placeholder

    n_raw_trade   = raw[1]   + raw[2]
    n_gated_trade = gated[1] + gated[2]
    pass_rate  = n_gated_trade / n_sample * 100
    gate_atten = (1.0 - n_gated_trade / max(1, n_raw_trade)) * 100  # 가장 큰 값을 찾는다
    avg_mp     = float(np.mean(max_probs_list)) if max_probs_list else 0.0  # 평균값을 계산한다  # Computes the mean value
    fish_min   = agent.config.fisher_threshold_min
    fish_max   = agent.config.fisher_threshold_max

    print(  # 결과를 화면에 출력한다  # Prints output to stdout
        f"  [ProdEval/{n_sample}] "  # 문자열 안에 변수 값을 넣어 만든다
        f"Raw  L={raw[1]:3d} S={raw[2]:3d} H={raw[0]:3d} | "  # 문자열 안에 변수 값을 넣어 만든다
        f"Prod L={gated[1]:3d} S={gated[2]:3d} H={gated[0]:3d} | "  # 문자열 안에 변수 값을 넣어 만든다
        f"Pass={pass_rate:.0f}%  GateAttrition={gate_atten:.0f}%"  # 문자열 안에 변수 값을 넣어 만든다
    )
    _nrt = max(1, n_raw_trade)  # 가장 큰 값을 찾는다
    def _blk_tag(n, total):  # [_blk_tag] 내부 전용 함수 정의
        pct = n / total * 100
        if pct >= 80:  return "PRIMARY"  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        if pct >= 40:  return "major"  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        if pct >= 10:  return "minor"  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        return "none"  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
    print(  # 결과를 화면에 출력한다  # Prints output to stdout
        f"  [GateDiag ] AvgMaxProb={avg_mp:.3f}  "  # 문자열 안에 변수 값을 넣어 만든다
        f"FisherThr=[{fish_min:.2f},{fish_max:.2f}]  "  # 문자열 안에 변수 값을 넣어 만든다
        f"Fisher_blk={gate_fisher_blk}/{_nrt} "  # 문자열 안에 변수 값을 넣어 만든다
        f"({'PRIMARY BLOCKER' if gate_fisher_blk >= n_raw_trade * 0.8 else 'partial'})"  # 문자열 안에 변수 값을 넣어 만든다
    )
    print(  # 결과를 화면에 출력한다  # Prints output to stdout
        f"  [GateBreak] (of {n_raw_trade} raw-trade samples)  "  # 문자열 안에 변수 값을 넣어 만든다
        # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식
        # Lindblad master equation: quantum decoherence model
        f"Lindblad={gate_lindblad_blk}/{_nrt}({_blk_tag(gate_lindblad_blk,_nrt)})  "
        f"CR={gate_cr_blk}/{_nrt}({_blk_tag(gate_cr_blk,_nrt)})  "  # 문자열 안에 변수 값을 넣어 만든다
        f"EP={gate_ep_blk}/{_nrt}({_blk_tag(gate_ep_blk,_nrt)})  "  # 문자열 안에 변수 값을 넣어 만든다
        f"Fisher={gate_fisher_blk}/{_nrt}({_blk_tag(gate_fisher_blk,_nrt)})"  # 문자열 안에 변수 값을 넣어 만든다
    )

    return {  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
        "n_raw_trade":      n_raw_trade,
        "gate_lindblad_blk": gate_lindblad_blk,  # 린드블라드: 양자 결맞음 소실을 시뮬레이션하는 방정식  # Lindblad master equation: quantum decoherence model
        "gate_cr_blk":       gate_cr_blk,
        "gate_ep_blk":       gate_ep_blk,
        "gate_fisher_blk":   gate_fisher_blk,
        "pass_rate":         pass_rate / 100.0,
    }


def _load_symbol_data_for_rl(sym: str, args, dc):  # [_load_symbol_data_for_rl] 내부 전용 함수 정의
    """심볼별 데이터 로드·레이블링·피처 생성. (df_labeled, all_features) 반환."""
    import pickle  # pickle 라이브러리를 불러온다
    # src.data.binance_client 모듈에서 fetch_binance_taker_history as _ftk를 가져온다
    # Import fetch_binance_taker_history as _ftk from src.data.binance_client module
    from src.data.binance_client import fetch_binance_taker_history as _ftk
    from src.models.labeling import (  # src.models.labeling 모듈에서 (를 가져온다
        compute_clean_barrier_labels as _ccbl,
        standardize_1m_ohlcv as _std,
    )
    if getattr(args, "feat_ver", 4) == "structural":
        from src.models.features_structural import generate_and_cache_features_structural as _gcf
    elif getattr(args, "feat_ver", 4) == 5:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        from src.models.features_v5 import generate_and_cache_features_v5 as _gcf
    else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
        from src.models.features_v4 import generate_and_cache_features_v4 as _gcf

    _end_dt = datetime.now(timezone.utc)
    if args.end_date:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        _end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc, hour=23, minute=59)
    _start_str = (args.start_date if args.start_date
                  else (_end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d"))
    _start_ms = int(pd.Timestamp(_start_str).timestamp() * 1000)  # 정수로 변환한다
    _end_ms   = int(_end_dt.timestamp() * 1000)  # 정수로 변환한다

    # 결과를 화면에 출력한다
    # Prints output to stdout
    print(f"\n  [Data/{sym}] Fetching OHLCV {_start_str} ~ {_end_dt.strftime('%Y-%m-%d')} ...")
    df_raw = dc.fetch_training_history(
        symbol=sym, timeframe=args.timeframe, start_date=_start_str,
        end_ms=_end_ms, cache_dir="data"
    )
    if df_raw is None or df_raw.empty:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        print(f"  [Data/{sym}] ⚠ 빈 데이터 → 스킵")  # 결과를 화면에 출력한다  # Prints output to stdout
        return None, None  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    df_clean = _std(df_raw) if args.timeframe == "1m" else df_raw.copy()

    # Funding rate
    df_funding = dc.fetch_funding_history(sym, _start_ms, _end_ms, cache_dir="data")
    if not df_funding.empty:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
        df_funding["ts"] = (pd.to_datetime(df_funding["ts_ms"], unit="ms", utc=True)
                            .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
        # 펀딩 비율: 롱-숏 불균형 시 주기적으로 주고받는 비용
        # Funding rate: periodic payment between longs and shorts
        df_clean = df_clean.merge(df_funding[["ts", "funding_rate"]], on="ts", how="left")
        # 펀딩 비율: 롱-숏 불균형 시 주기적으로 주고받는 비용
        # Funding rate: periodic payment between longs and shorts
        df_clean["funding_rate"] = df_clean["funding_rate"].ffill().fillna(0.0)

    # Open interest
    df_oi = dc.fetch_open_interest_history(sym, _start_ms, _end_ms, interval="1h", cache_dir="data")
    if not df_oi.empty:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
        df_oi["ts"] = (pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
                       .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
        # 미결제 약정: 현재 열려있는 선물 계약의 총 수량
        # Open interest: total number of outstanding contracts
        df_clean = df_clean.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
        # 미결제 약정: 현재 열려있는 선물 계약의 총 수량
        # Open interest: total number of outstanding contracts
        df_clean["open_interest"] = df_clean["open_interest"].ffill().fillna(0.0)

    # Binance CVD
    _te = _end_dt.strftime("%Y-%m-%d")
    # 문자열 안에 변수 값을 넣어 만든다
    _tc = f"data/binance_taker_{sym}_{args.timeframe}_{_start_str.replace('-','')}_{_te.replace('-','')}.csv"
    try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
        df_taker = _ftk(symbol=sym, interval=args.timeframe,
                        start_date=_start_str, end_date=_te,
                        cache_path=_tc, verbose=False)
        if not df_taker.empty:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
            df_clean = df_clean.merge(df_taker[["ts", "taker_buy_volume"]], on="ts", how="left")
            df_clean["taker_buy_volume"] = df_clean["taker_buy_volume"].fillna(0.0)
    except Exception:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
        pass  # 아무것도 하지 않는다 (빈 블록 자리 채우기)  # No-op placeholder

    # Labels
    _bars_h = 96 if args.timeframe == "15m" else 60
    df_lbl_raw = _ccbl(df_clean, alpha=4.0, beta=1.5, hold_band=1.5, hold_h=20, h=_bars_h)
    df_labeled = df_lbl_raw[df_lbl_raw["label"] != 2].reset_index(drop=True)
    _lv = df_labeled["label"].value_counts().to_dict()  # 딕셔너리(키-값 쌍)를 만든다
    print(f"  [Data/{sym}] LONG={_lv.get(1,0)} SHORT={_lv.get(-1,0)} HOLD={_lv.get(0,0)} "  # 결과를 화면에 출력한다  # Prints output to stdout
          f"DISCARD={(df_lbl_raw['label']==2).sum()}")  # 문자열 안에 변수 값을 넣어 만든다

    # Features
    _fver  = getattr(args, "feat_ver", 4)  # 객체의 속성 값을 동적으로 가져온다  # Gets a named attribute from an object dynamically
    # 문자열 안에 변수 값을 넣어 만든다
    _vsuffix = "structural" if _fver == "structural" else ("5" if _fver == 5 else "4cvd")
    _cache = f"data/feat_cache_{sym}_{args.timeframe}_{args.days}d_v{_vsuffix}.npy"
    all_feat_raw = _gcf(df_clean, _cache)
    _keep = df_lbl_raw["label"].values != 2
    all_feat = all_feat_raw[_keep]

    # bc_scaler
    _scaler_path = os.path.join(args.checkpoint_dir, "bc_scaler.pkl")  # 폴더와 파일 이름을 합쳐 경로를 만든다  # Joins path components into a single path string
    if os.path.isfile(_scaler_path):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        with open(_scaler_path, "rb") as _f:  # 자원을 안전하게 열고, 블록이 끝나면 자동으로 닫는다  # Context manager: resource is opened and auto-closed
            _sc = pickle.load(_f)
        all_feat = _sc.transform(all_feat).astype(np.float32)

    if len(all_feat) != len(df_labeled):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        # 결과를 화면에 출력한다
        # Prints output to stdout
        print(f"  [Data/{sym}] ⚠ Feature/label mismatch ({len(all_feat)} vs {len(df_labeled)}) → 스킵")
        return None, None  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    return df_labeled, all_feat  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller


def main():  # [main] 함수 정의 시작
    parser = argparse.ArgumentParser(description="Quantum V2 Training Pipeline")
    parser.add_argument("--symbol", default="BTCUSDT")  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
    parser.add_argument("--symbols", default=None,  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
        help="멀티심볼 쉼표구분 (예: BTCUSDT,ETHUSDT,SOLUSDT). 지정 시 --symbol 무시.")
    parser.add_argument("--timeframe", default="15m")  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
    parser.add_argument("--days", type=int, default=30)  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
    parser.add_argument("--start_date", default=None, help="YYYY-MM-DD")  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
    parser.add_argument("--end_date", default=None, help="YYYY-MM-DD")  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
    parser.add_argument("--n-folds", type=int, default=5)  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
    parser.add_argument("--epochs", type=int, default=10)  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
    parser.add_argument("--batch-size", type=int, default=128)  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
    # 커맨드라인에서 이 인자를 받을 수 있게 등록한다
    # Registers a CLI argument for argparse
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints/quantum_v2")  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
    parser.add_argument("--leverage", type=float, default=10.0)  # 레버리지: 실제 증거금의 몇 배로 거래하는지  # Registers a CLI argument for argparse
    parser.add_argument("--confidence", type=float, default=0.40)  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
    parser.add_argument(  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
        "--drive-dir", default=None,
        help="Google Drive 저장 경로 (예: /content/drive/MyDrive/quantum_v2). "
             "지정 시 Best 체크포인트마다 Drive로 즉시 복사."
    )
    parser.add_argument(  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
        "--rolling-window", type=str, default=None,
        help="Rolling window 학습 크기 (일 단위 정수 or 'auto'). "
             "'auto' 시 데이터 기반 Markov E[L] + Sample Efficiency 제약으로 자동 계산. "
             "미설정 시 전통 Expanding Window 사용."
    )
    parser.add_argument(  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
        "--pretrain-ckpt", default=None,  # 체크포인트(저장된 모델 상태) 관련 처리를 한다  # Checkpoint: saved model state for resuming training
        # 체크포인트(저장된 모델 상태) 관련 처리를 한다
        # Checkpoint: saved model state for resuming training
        help="BC 사전학습 체크포인트 경로 (예: checkpoints/quantum_v2/agent_bc_pretrained.pt). "
             "지정 시 RL 학습 시작 전 모델 가중치만 로드 (optimizer/scheduler는 초기화 유지)."
    )
    parser.add_argument(  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
        "--seq-len", type=int, default=96,
        help="슬라이딩 윈도우 길이 (봉 단위). BC와 동일하게 맞춰야 함. 기본값=96 (24시간)."
    )
    def _fv_type(x):
        try: return int(x)
        except ValueError: return x
    parser.add_argument(  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
        "--feat-ver", type=_fv_type, default=4, choices=[4, 5, "structural"],
        help="Feature version: 4=V4 26-dim (default), 5=V5 48-dim, structural=13-dim"
    )
    parser.add_argument(  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
        "--seed", type=int, default=None,
        help="랜덤 시드 고정 (재현 가능, 예: --seed 6)"
    )
    parser.add_argument(  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
        "--tp-mult", type=float, default=4.0,
        # ATR: 가격의 평균 변동 폭 (Average True Range)
        # ATR: Average True Range — average price volatility
        help="시뮬레이션 보상: TP 배수 (ATR 단위). 레이블 생성 alpha=4.0과 일치시킬 것. 기본=4.0"
    )
    parser.add_argument(  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
        "--sl-mult", type=float, default=1.5,
        # ATR: 가격의 평균 변동 폭 (Average True Range)
        # ATR: Average True Range — average price volatility
        help="시뮬레이션 보상: SL 배수 (ATR 단위). 레이블 생성 beta=1.5와 일치시킬 것. 기본=1.5"
    )
    parser.add_argument(  # 커맨드라인에서 이 인자를 받을 수 있게 등록한다  # Registers a CLI argument for argparse
        "--entropy-reg", type=float, default=None,
        # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항
        # Entropy regularization: encourages policy exploration
        help="초기 entropy_reg 강제 설정 (기본: AgentConfig 기본값 0.05). HOLD 붕괴 시 0.12 권장"
    )
    args = parser.parse_args()  # 커맨드라인 인자를 파싱해서 args에 저장한다  # Parses CLI arguments into an args namespace

    # ── Seed 고정 ────────────────────────────────────────────────────────────
    if args.seed is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        import random  # 난수(무작위 숫자) 생성 표준 라이브러리를 불러온다  # Import Random number generation
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            torch.cuda.manual_seed_all(args.seed)
        print(f"  [Seed] 랜덤 시드 고정: {args.seed} (재현 가능)")  # 결과를 화면에 출력한다  # Prints output to stdout

    # ── Drive 초기화 ──────────────────────────────────────────────────────────
    if args.drive_dir:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        mounted = _mount_drive()
        if mounted:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            os.makedirs(args.drive_dir, exist_ok=True)  # 필요한 폴더가 없으면 새로 만든다  # Creates directory (and parents) if they do not exist
            print(f"  [Drive] 저장 경로: {args.drive_dir}")  # 결과를 화면에 출력한다  # Prints output to stdout
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            print("  [Drive] ⚠ Colab 환경이 아님 — --drive-dir 무시됨")  # 결과를 화면에 출력한다  # Prints output to stdout

    # ── 심볼 목록 결정 ──────────────────────────────────────────────────────
    _symbol_list = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else [args.symbol]  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
    )
    print(f"🚀 [Phase 5] Quantum V2 통합 학습 시작: {_symbol_list}  TF={args.timeframe}")  # 결과를 화면에 출력한다  # Prints output to stdout

    dc = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")

    # ── 심볼별 데이터 로드 (primary + secondary) ─────────────────────────────
    symbol_datasets: list = []   # list of (sym, df_labeled, all_features)
    for _sym in _symbol_list:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
        _df_lab, _all_feat = _load_symbol_data_for_rl(_sym, args, dc)
        if _df_lab is not None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            symbol_datasets.append((_sym, _df_lab, _all_feat))  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list

    if not symbol_datasets:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
        print("  [Data] ERROR: 유효 데이터 없음"); return  # 결과를 화면에 출력한다  # Prints output to stdout

    # Primary symbol (fold timing 기준)
    _primary_sym, df_labeled, all_features = symbol_datasets[0]
    print(f"\n  [Data] Primary={_primary_sym}  총 심볼={len(symbol_datasets)}  "  # 결과를 화면에 출력한다  # Prints output to stdout
          f"Primary bars={len(df_labeled):,}")  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items

    # ── Rolling Window 설정 ──────────────────────────────────────────────────
    _bars_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
    bpd = _bars_per_day.get(args.timeframe, 96)
    rolling_bars = None
    if args.rolling_window:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        rw_str = str(args.rolling_window).strip().lower()  # 문자열로 변환한다
        if rw_str == "auto":  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # 데이터 기반 최적 rolling window 자동 계산
            rw_days = compute_optimal_rolling_window(
                df_labeled,
                bpd=bpd,
                min_trades=200,
                trade_rate_per_bar=10 / bpd,  # 하루 최소 10거래 기준 (10/96 ≈ 0.104)
                n_regime_stds=1.5,
                max_days=360,
                verbose=True,
            )
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            rw_days = int(rw_str)  # 정수로 변환한다
        rolling_bars = rw_days * bpd
        print(f"  [Mode] ROLLING WINDOW: {rw_days}d = {rolling_bars} bars per fold")  # 결과를 화면에 출력한다  # Prints output to stdout
    else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
        print(f"  [Mode] EXPANDING WINDOW (all history up to each val start)")  # 결과를 화면에 출력한다  # Prints output to stdout

    fold_gen = walk_forward_folds(df_labeled, n_folds=args.n_folds, rolling_bars=rolling_bars)
    _fv = getattr(args, "feat_ver", 4)
    _feat_dim = 13 if _fv == "structural" else (48 if _fv == 5 else 28)
    _entropy_reg_init = getattr(args, "entropy_reg", None) or 0.05  # 엔트로피 정규화: 다양한 행동을 시도하도록 장려하는 항  # Entropy regularization: encourages policy exploration
    # 레버리지: 실제 증거금의 몇 배로 거래하는지
    # Entropy regularization: encourages policy exploration
    config = AgentConfig(leverage=args.leverage, checkpoint_dir=args.checkpoint_dir, confidence_threshold=args.confidence, feature_dim=_feat_dim, entropy_reg=_entropy_reg_init)
    agent = build_quantum_agent(config=config, device=torch.device(args.device))

    # ── Phase 0 BC 사전학습 가중치 로드 ────────────────────────────────────────
    # BC 체크포인트는 모델 가중치만 로드 (optimizer/scheduler는 RL 학습용으로 신규 초기화 유지).
    # AlphaGo 원칙: SL Policy Net 가중치 → RL Fine-tuning 출발점으로만 사용.
    if args.pretrain_ckpt:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        if not os.path.isfile(args.pretrain_ckpt):  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
            print(f"  [BC] ⚠ --pretrain-ckpt 파일 없음: {args.pretrain_ckpt} — 무시하고 계속")  # 결과를 화면에 출력한다  # Prints output to stdout
        else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
            _bc_ckpt = torch.load(args.pretrain_ckpt,  # 파일에서 저장된 모델/텐서를 불러온다  # Loads a tensor/model from disk
                                  map_location=torch.device(args.device),
                                  weights_only=False)
            # Strip Koopman buffers — size depends on precomputed config (loaded per-fold later)
            _bc_state = {k: v for k, v in _bc_ckpt["model_state"].items()
                         if not k.startswith("encoder.decomposer._koop_")}
            _missing, _unexpected = agent.load_state_dict(
                _bc_state, strict=False
            )
            if _missing:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                print(f"  [BC] ⚠ Missing keys ({len(_missing)}): {_missing[:5]}")  # 결과를 화면에 출력한다  # Prints output to stdout
            if _unexpected:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                print(f"  [BC] ⚠ Unexpected keys ({len(_unexpected)}): {_unexpected[:5]}")  # 결과를 화면에 출력한다  # Prints output to stdout
            print(  # 결과를 화면에 출력한다  # Prints output to stdout
                f"  [BC] ✓ BC 가중치 로드 완료: {args.pretrain_ckpt}\n"  # 문자열 안에 변수 값을 넣어 만든다  # Checkpoint: saved model state for resuming training
                f"       global_step={_bc_ckpt.get('global_step', 0)} | "  # 문자열 안에 변수 값을 넣어 만든다  # Checkpoint: saved model state for resuming training
                f"optimizer/scheduler는 RL 신규 초기화"  # 문자열 안에 변수 값을 넣어 만든다
            )

            # BC Fine-Tune 모드: VQC 재초기화 없이 BC 파라미터 그대로 유지
            # 이전: near-identity(randn*0.01) → BC 지식 손실, RL이 처음부터 재탐험
            # 현재: BC VQC 방향벡터 보존 → RL이 좋은 초기점에서 fine-tune
            for name, param in agent.named_parameters():  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                if "vqc_weights" in name:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                    param.requires_grad = True   # BC 동결 해제만
            print("  [BC] ✅ VQC weights BC 파라미터 유지 (fine-tune 모드, 재초기화 없음)")  # 결과를 화면에 출력한다  # Prints output to stdout

            # Pretrain 지식을 보존하면서 VQC 학습 가능하도록 LR 조정
            # Fine-tune LR: BC 파라미터 보존이 목표이므로 VQC LR 대폭 축소
            # near-identity 시: VQC가 처음부터 학습 → lr_quantum=0.03 필요
            # fine-tune 시: 이미 좋은 방향 → 큰 LR은 BC 지식을 덮어씀
            config.lr = 5e-5               # 클래식 레이어: 보수적 fine-tune
            config.eta_base = 0.0005
            if hasattr(agent.optimizer, 'lr_quantum'):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                agent.optimizer.lr_quantum = 0.005  # VQC: BC의 1/10 수준 (0.03→0.005)
            print(f"  [BC] 📉 Fine-Tune LR: 클래식={config.lr}, VQC=0.005 (BC 지식 보존)")  # 결과를 화면에 출력한다  # Prints output to stdout

            # ── PAC-Bayes J: N_eff 계산 (Bartlett 자기상관 보정) ─────────────
            # N_eff = N / (1 + 2Σρ(k))  — 금융 시계열 실효 독립 샘플 수
            # 창이 짧을수록(N_eff↓) λ = C/N_eff 자동 증가 → BC 사전지식 강화
            _close = df_labeled["close"].values
            _lr_arr = np.log(_close[1:] / _close[:-1])  # 자연 로그를 계산한다  # Computes natural logarithm element-wise
            _max_lag = min(20, len(_lr_arr) // 4)  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
            _acf_pos_sum = 0.0
            for _lag in range(1, _max_lag + 1):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
                _rho = float(np.corrcoef(_lr_arr[:-_lag], _lr_arr[_lag:])[0, 1])  # 상관계수 행렬을 계산한다  # Computes the Pearson correlation coefficient matrix
                if _rho > 0:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                    _acf_pos_sum += _rho
            _n_eff = len(_lr_arr) / max(1.0 + 2.0 * _acf_pos_sum, 1.0)  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
            config.pac_bayes_n_eff = float(_n_eff)  # 실수(소수)로 변환한다
            _lam = config.pac_bayes_coef / _n_eff
            print(f"  [PAC-Bayes] N_eff={_n_eff:.0f} (Bartlett)  λ={_lam:.2e}"  # 결과를 화면에 출력한다  # Prints output to stdout
                  f"  (C={config.pac_bayes_coef})")  # 문자열 안에 변수 값을 넣어 만든다

            # BC prior 저장 — RL 학습 중 이 기준점으로 당겨짐
            agent.set_bc_prior(agent.state_dict())  # 딕셔너리(키-값 쌍)를 만든다
            print(f"  [PAC-Bayes] BC prior 저장 완료 ({len(agent._bc_prior_params)} 텐서, VQC 제외)")  # 결과를 화면에 출력한다  # Prints output to stdout

            # ── Method G: RL 단계부터 Spectral Norm 활성화 ────────────────────
            # BC에서는 False (HOLD 억압 방지), RL에서는 True (logit 폭발 차단)
            # encoder는 이미 생성되어 있으므로 기존 Linear에 SN 훅을 사후 적용.
            try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
                # torch.nn.utils 모듈에서 spectral_norm as _sn를 가져온다
                # Import spectral_norm as _sn from torch.nn.utils module
                from torch.nn.utils import spectral_norm as _sn
                _ql = agent.encoder.quantum_layer
                # logit_proj: Sequential([Linear, GELU, Dropout, Linear])
                if isinstance(_ql.logit_proj, torch.nn.Sequential):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                    # 변수가 특정 타입인지 확인한다
                    # Checks if object is an instance of given type(s)
                    _linears = [m for m in _ql.logit_proj if isinstance(m, torch.nn.Linear)]
                    for _lin in _linears:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                        _sn(_lin)
                elif isinstance(_ql.logit_proj, torch.nn.Linear):  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
                    _sn(_ql.logit_proj)
                # classical_head
                if isinstance(_ql.classical_head, torch.nn.Linear):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                    _sn(_ql.classical_head)
                config.use_spectral_norm = True
                print("  [Method G] Spectral Norm 사후 적용 완료 (logit_proj + classical_head)")  # 결과를 화면에 출력한다  # Prints output to stdout
            except Exception as _e:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
                print(f"  [Method G] ⚠ Spectral Norm 적용 실패: {_e} — 스킵")  # 결과를 화면에 출력한다  # Prints output to stdout

    best_sniper_score = -float('inf')  # 실수(소수)로 변환한다
    auto_ctrl = AdaptiveController(agent)   # 자동 하이퍼파라미터 컨트롤러
    viz = TrainingVisualizer(save_dir="reports/viz")
    _fold_best_history: list = []   # fold-best 기록 (viz용)
    all_folds_pnl_series: list = []

    def _fmt_date(idx_val):  # [_fmt_date] 내부 전용 함수 정의
        """Index value -> KST date string (YYYY-MM-DD)."""
        try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
            ts = pd.Timestamp(idx_val)
            if ts.tzinfo is None:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                ts = ts.tz_localize("UTC")
            ts_kst = ts.tz_convert("Asia/Seoul")
            return ts_kst.strftime("%Y-%m-%d")  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller
        except Exception:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
            return str(idx_val)[:10]  # 함수의 계산 결과를 호출자에게 반환(돌려준다)  # Returns a value to the caller

    for k, train_df, val_df, tr_idx, va_idx, train_abs, val_abs in fold_gen:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
        tr_start = _fmt_date(tr_idx[0])
        tr_end   = _fmt_date(tr_idx[1])
        va_start = _fmt_date(va_idx[0])
        va_end   = _fmt_date(va_idx[1])
        mode_tag = "ROLLING" if rolling_bars else "EXPANDING"
        print(  # 결과를 화면에 출력한다  # Prints output to stdout
            f"\n[Fold {k}/{args.n_folds}] [{mode_tag}] "  # 문자열 안에 변수 값을 넣어 만든다
            f"Train: {len(train_df):,} bars  {tr_start} ~ {tr_end} | "  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
            f"Val: {len(val_df):,} bars  {va_start} ~ {va_end}"  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
        )

        # 피처 슬라이싱: absolute row indices (rolling 모드에서 train_start != 0)
        train_features = all_features[train_abs[0]:train_abs[1]]
        val_features   = all_features[val_abs[0]:val_abs[1]]

        train_data = prepare_training_data(train_df, train_features)
        val_data = prepare_training_data(val_df, val_features)

        if not train_data or not val_data: continue  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False

        # ── Secondary symbols: 동일 fold 비율로 슬라이스 ─────────────────────
        # Primary의 fold 위치를 0~1 비율로 환산해 secondary에 적용 (길이 다를 수 있음)
        _tr_frac_s = train_abs[0] / max(1, len(all_features))  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
        _tr_frac_e = train_abs[1] / max(1, len(all_features))  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
        _va_frac_s = val_abs[0]   / max(1, len(all_features))  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
        _va_frac_e = val_abs[1]   / max(1, len(all_features))  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
        _extra_train_datas: list = []
        _extra_val_datas:   list = []
        for _s2, _df2, _af2 in symbol_datasets[1:]:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
            _n2 = len(_df2)  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
            _ts2 = int(_tr_frac_s * _n2)  # 정수로 변환한다
            _te2 = int(_tr_frac_e * _n2)  # 정수로 변환한다
            _vs2 = int(_va_frac_s * _n2)  # 정수로 변환한다
            _ve2 = int(_va_frac_e * _n2)  # 정수로 변환한다
            if _te2 <= _ts2 or _ve2 <= _vs2: continue  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            _td2 = prepare_training_data(_df2.iloc[_ts2:_te2].reset_index(drop=True), _af2[_ts2:_te2])
            _vd2 = prepare_training_data(_df2.iloc[_vs2:_ve2].reset_index(drop=True), _af2[_vs2:_ve2])
            if _td2 and _vd2:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                _extra_train_datas.append((_s2, _td2))  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                _extra_val_datas.append((_s2, _vd2))  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list

        # ── Koopman precompute (once per fold — refit on new train window) ──
        _koop_path = os.path.join("data", f"koopman_config_{args.timeframe}_fold{k}.npz")
        _decomposer = agent.encoder.decomposer
        if _decomposer.use_edmd and not os.path.exists(_koop_path):
            print(f"  [Koopman] Fold {k}: precomputing config ...", flush=True)
            from src.data.koopman_config import precompute_koopman_config
            precompute_koopman_config(
                X_train   = train_features,
                n_modes   = agent.config.n_eigenvectors,
                max_terms = min(40, agent.config.feature_dim * 3),
                save_path = _koop_path,
                verbose   = False,
            )
        if _decomposer.use_edmd and os.path.exists(_koop_path):
            _decomposer.load_koopman_precomputed(_koop_path)

        # ── Fisher LDA: fold별 지도 판별 방향 학습 ──────────────────────────
        # PCA(분산 최대화) 대신 LONG/SHORT/HOLD 클래스 분리를 직접 최대화.
        # train_features: [N_train, F],  train_df['label']: 0/1/-1
        if _decomposer.use_lda:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            _train_labels = train_df["label"].values  # [N_train] numpy int
            _decomposer.fit_lda(train_features, _train_labels)

        # Optimizer + Scheduler reset at each fold:
        # Prevents LR from permanently decaying as training data grows fold-to-fold.
        # Model weights are from prior fold's BEST epoch (loaded at fold-end below).
        # This lets the model actively re-learn from each new fold's data.
        # src.models.qng_optimizer 모듈에서 DiagonalQNGOptimizer as _DQNG를 가져온다
        # Import DiagonalQNGOptimizer as _DQNG from src.models.qng_optimizer module
        from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
        for pg in agent.optimizer.param_groups:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
            pg["lr"] = agent.config.lr
        # Clear Adam momentum state — QNG: target classical_optimizer only
        _inner_opt = (
            agent.optimizer.classical_optimizer
            if isinstance(agent.optimizer, _DQNG)  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            else agent.optimizer
        )
        _inner_opt.state.clear()
        _sched_target = _inner_opt
        agent.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            _sched_target, T_0=100, T_mult=2, eta_min=1e-5
        )

        # Fix #3 (Threshold Leakage): per-fold best tracking.
        # best_sniper_score is GLOBAL — comparing val scores across different
        # market regimes (e.g. fold1=bearish, fold2=recovery) is meaningless.
        # Each fold must save its own best checkpoint independently.
        best_fold_sniper = -float('inf')  # 실수(소수)로 변환한다
        best_epoch_pnl_series = []

        # AdaptiveController: reset per fold so Fold N penalties don't bleed into Fold N+1.
        auto_ctrl = AdaptiveController(agent)

        # Early stopping: halt when EV doesn't improve for patience epochs.
        # Saves compute in later folds where EV plateaus by epoch 2-3.
        patience_epochs = 5
        no_improve_count = 0

        fold_epoch_times: list = []
        _epoch_log: list = []   # per-epoch metrics for visualization

        for epoch in range(1, args.epochs + 1):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
            epoch_start = time.time()  # 전체 데이터를 몇 번 반복해서 학습할지 결정  # Number of full passes over the training dataset
            agent.train()  # 모델을 학습 모드로 전환한다 (Dropout, BatchNorm 활성화)  # Switches model to training mode (enables Dropout, BN)
            n_samples = train_data["n"]
            seq_len = args.seq_len

            # Balanced class sampling: LONG:SHORT = 1:1 to prevent base-rate exploitation.
            # Without balancing, 58% SHORT labels cause action collapse toward SHORT.
            # h=96(15m=24시간)과 win_prices 창을 일치시킴 (ChatGPT 제안 1).
            # 기존 30 bars는 라벨 기준(96 bars)의 31% 만 커버 → 보상 불일치.
            H = 96 if args.timeframe == "15m" else 60
            valid_min  = seq_len + 120
            valid_max  = n_samples - (H + 1)
            all_valid  = np.arange(valid_min, valid_max)  # 지정한 범위의 숫자들로 배열을 만든다  # Creates an array of evenly spaced integers
            raw_lbl    = train_data["raw_labels"][all_valid]
            long_pool  = all_valid[raw_lbl == 1]
            short_pool = all_valid[raw_lbl == -1]
            hold_pool  = all_valid[raw_lbl == 0]
            n_per_dir  = min(len(long_pool), len(short_pool))  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
            sampled_long  = np.random.choice(long_pool,  n_per_dir, replace=False)
            sampled_short = np.random.choice(short_pool, n_per_dir, replace=False)
            n_hold = min(len(hold_pool), max(1, n_per_dir // 4))  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
            sampled_hold  = (np.random.choice(hold_pool, n_hold,
                                              replace=n_hold > len(hold_pool))  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
                             if len(hold_pool) > 0 else np.array([], dtype=int))  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            indices = np.random.permutation(
                np.concatenate([sampled_long, sampled_short, sampled_hold])  # 여러 배열을 하나로 이어붙인다  # Concatenates arrays along an axis
            )
            epoch_loss = 0  # 전체 데이터를 몇 번 반복해서 학습할지 결정  # Number of full passes over the training dataset
            epoch_critic_loss = 0.0  # P4: per-component GN diagnosis
            epoch_fp_loss     = 0.0  # 전체 데이터를 몇 번 반복해서 학습할지 결정  # Number of full passes over the training dataset
            epoch_dir_sym     = 0.0  # 전체 데이터를 몇 번 반복해서 학습할지 결정  # Number of full passes over the training dataset
            n_batches = 0

            # TP/SL 시뮬레이션 파라미터 (레이블 생성 alpha/beta와 일치)
            _tp_mult = args.tp_mult  # default=4.0
            _sl_mult = args.sl_mult  # default=1.5

            for start_idx in range(0, len(indices), args.batch_size):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
                batch_indices = indices[start_idx:start_idx + args.batch_size]

                # ── Step 1: 피처·가격·ATR 수집 (방향/레이블 미결정) ─────────────
                x_list, prices_list, atr_list, entry_list = [], [], [], []
                highs_list, lows_list = [], []

                for idx in batch_indices:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                    x_list.append(train_data["features"][idx - seq_len + 1 : idx + 1])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                    win_prices = train_data["prices"][idx : idx + H + 1]
                    prices_list.append(win_prices)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                    highs_list.append(train_data["highs"][idx : idx + H + 1])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                    lows_list.append(train_data["lows"][idx : idx + H + 1])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                    atr_list.append(train_data["atr"][idx])  # ATR: 가격의 평균 변동 폭 (Average True Range)
                    entry_list.append(0)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list

                x_train = torch.from_numpy(np.array(x_list)).float()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32

                # ── Step 2: No-grad forward → 모델 예측 행동 취득 ───────────────
                # 모델이 현재 상태에서 실제로 선택할 행동(HOLD/LONG/SHORT)을 얻는다.
                # 이 행동을 기반으로 미래 가격으로 실제 P&L을 시뮬레이션 → reward hacking 제거.
                with torch.no_grad():  # 메모리 절약을 위해 기울기 계산 없이 추론만 실행한다  # Context: disable gradient tracking for inference (saves memory)
                    # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다
                    # Moves tensor to specified device or dtype
                    _logits, _, _, _ = agent(x_train.to(agent.device), last_step_only=True)
                    # squeeze: [B,1,3] → [B,3] (last_step_only 모드에서 dim=3 가능)
                    if _logits.dim() == 3:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        _logits = _logits.squeeze(1)  # 크기가 1인 차원을 없앤다  # Removes dimensions of size 1
                    pred_actions = _logits.argmax(dim=-1).cpu().numpy()  # [B] int64

                # ── Step 3: TP/SL 시뮬레이션 → 실제 레이블 결정 ─────────────────
                # 모델 행동 + 실제 미래 고가/저가 → TP hit / SL hit / timeout 판정.
                # label=1: LONG TP hit → TP_HIT (+r_tp)
                # label=2: SHORT TP hit → TP_HIT (+r_tp)
                # label=3: SL hit (any dir) → SL_HIT (-r_sl, 음수 보상)
                # label=0: timeout / HOLD → OBSERVE (0.0)
                dirs_list, labels_list = [], []

                for i, idx in enumerate(batch_indices):  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                    act = int(pred_actions[i])  # 정수로 변환한다
                    if act == 0:  # HOLD 예측 → 거래 없음
                        dirs_list.append(0)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                        labels_list.append(0)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                        continue  # 현재 반복의 남은 코드를 건너뛰고 다음 반복으로 간다  # Skip the rest of this iteration

                    direction = 1 if act == 1 else -1
                    ep      = float(prices_list[i][0])          # 진입가
                    atr_val = float(atr_list[i])  # 실수(소수)로 변환한다
                    tp_dist = _tp_mult * atr_val               # TP = 4×ATR
                    sl_dist = _sl_mult * atr_val               # SL = 1.5×ATR

                    # numpy vectorised scan (Python 루프보다 ~10× 빠름)
                    _hi = highs_list[i][1:]                    # entry bar 제외
                    _lo = lows_list[i][1:]
                    if direction == 1:                         # LONG
                        tp_bars = np.where(_hi >= ep + tp_dist)[0]  # 조건에 따라 두 값 중 하나를 골라 배열을 만든다  # Returns elements chosen from two arrays by condition
                        sl_bars = np.where(_lo <= ep - sl_dist)[0]  # 조건에 따라 두 값 중 하나를 골라 배열을 만든다  # Returns elements chosen from two arrays by condition
                    else:                                      # SHORT
                        tp_bars = np.where(_lo <= ep - tp_dist)[0]  # 조건에 따라 두 값 중 하나를 골라 배열을 만든다  # Returns elements chosen from two arrays by condition
                        sl_bars = np.where(_hi >= ep + sl_dist)[0]  # 조건에 따라 두 값 중 하나를 골라 배열을 만든다  # Returns elements chosen from two arrays by condition

                    tp_bar = int(tp_bars[0]) if len(tp_bars) > 0 else H + 1  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
                    sl_bar = int(sl_bars[0]) if len(sl_bars) > 0 else H + 1  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items

                    dirs_list.append(direction)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                    if tp_bar <= sl_bar and tp_bar <= H:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        labels_list.append(1 if direction == 1 else 2)  # TP_HIT
                    elif sl_bar < tp_bar and sl_bar <= H:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
                        labels_list.append(3)                            # SL_HIT
                    else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                        labels_list.append(0)                            # timeout

                # x_train already built above
                max_p_len = max(len(p) for p in prices_list)  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
                prices_padded = np.zeros((len(prices_list), max_p_len))  # 0으로 채워진 배열을 만든다  # Creates a zero-filled array
                for i, p in enumerate(prices_list): prices_padded[i, :len(p)] = p  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence

                p_train = torch.from_numpy(prices_padded).float()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
                d_train = torch.from_numpy(np.array(dirs_list)).float()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
                e_train = torch.from_numpy(np.array(entry_list)).long()  # 텐서를 정수형(int64)으로 변환한다  # Casts tensor to int64
                l_train = torch.from_numpy(np.array(labels_list)).long()  # 텐서를 정수형(int64)으로 변환한다  # Casts tensor to int64
                a_train = torch.from_numpy(np.array(atr_list)).float()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32

                result = agent.train_step(x_train, p_train, d_train, e_train, l_train, a_train, last_step_only=True)
                epoch_loss        += result.loss
                epoch_critic_loss += result.critic_loss  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
                epoch_fp_loss     += result.fp_loss
                epoch_dir_sym     += result.dir_sym_loss
                n_batches += 1

            # ── Secondary symbol 학습 (같은 에폭, 시간 순서 독립적 배치) ────────
            for _s2, _td2 in _extra_train_datas:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                _n2     = _td2["n"]
                _vmin2  = seq_len + 120
                _vmax2  = _n2 - (H + 1)
                if _vmax2 <= _vmin2: continue  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                _av2    = np.arange(_vmin2, _vmax2)  # 지정한 범위의 숫자들로 배열을 만든다  # Creates an array of evenly spaced integers
                _rl2    = _td2["raw_labels"][_av2]
                _lp2    = _av2[_rl2 == 1];  _sp2 = _av2[_rl2 == -1]
                _npd2   = min(len(_lp2), len(_sp2))  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
                if _npd2 == 0: continue  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                _sl2    = np.random.choice(_lp2, _npd2, replace=False)
                _ss2    = np.random.choice(_sp2, _npd2, replace=False)
                _idx2   = np.random.permutation(np.concatenate([_sl2, _ss2]))  # 여러 배열을 하나로 이어붙인다  # Concatenates arrays along an axis
                for _si2 in range(0, len(_idx2), args.batch_size):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
                    _bi2 = _idx2[_si2:_si2 + args.batch_size]
                    _xl2, _pl2, _al2, _el2 = [], [], [], []
                    _hl2, _ll2 = [], []
                    for _ii in _bi2:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                        _xl2.append(_td2["features"][_ii - seq_len + 1 : _ii + 1])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                        _pl2.append(_td2["prices"][_ii : _ii + H + 1])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                        _hl2.append(_td2["highs"][_ii : _ii + H + 1])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                        _ll2.append(_td2["lows"][_ii : _ii + H + 1])  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                        # ATR: 가격의 평균 변동 폭 (Average True Range)
                        # ATR: Average True Range — average price volatility
                        _al2.append(_td2["atr"][_ii]);  _el2.append(0)
                    _xt2 = torch.from_numpy(np.array(_xl2)).float()  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
                    with torch.no_grad():  # 메모리 절약을 위해 기울기 계산 없이 추론만 실행한다  # Context: disable gradient tracking for inference (saves memory)
                        # 텐서를 지정한 장치(GPU/CPU) 또는 타입으로 옮긴다
                        # Moves tensor to specified device or dtype
                        _lg2, _, _, _ = agent(_xt2.to(agent.device), last_step_only=True)
                        if _lg2.dim() == 3: _lg2 = _lg2.squeeze(1)  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        _pa2 = _lg2.argmax(dim=-1).cpu().numpy()  # 텐서를 CPU 메모리로 옮긴다  # Moves tensor to CPU memory
                    _dl2, _lbl2 = [], []
                    for _ji, _ii in enumerate(_bi2):  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                        _act = int(_pa2[_ji])  # 정수로 변환한다
                        if _act == 0: _dl2.append(0); _lbl2.append(0); continue  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        _ep = float(_td2["prices"][_ii])  # 실수(소수)로 변환한다
                        _atr = float(_td2["atr"][_ii])  # ATR: 가격의 평균 변동 폭 (Average True Range)
                        _hh = _td2["highs"][_ii : _ii + H + 1]
                        _lh = _td2["lows"][_ii : _ii + H + 1]
                        # 이익 실현(TP) 목표: ATR의 몇 배에서 청산할지
                        # Take-profit multiplier: exit at TP × ATR
                        _tp_p = _ep * (1 + (_tp_mult * _atr / _ep) * (1 if _act == 1 else -1))
                        # 손절(SL) 기준: ATR의 몇 배에서 강제 청산할지
                        # Stop-loss multiplier: forced exit at SL × ATR
                        _sl_p = _ep * (1 - (_sl_mult * _atr / _ep) * (1 if _act == 1 else -1))
                        _res = 0
                        for _bi3 in range(len(_hh)):  # 지정한 횟수만큼 반복한다  # Loop: iterate a fixed number of times
                            if _act == 1:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                                if _hh[_bi3] >= _tp_p: _res = 1; break  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                                if _lh[_bi3] <= _sl_p: _res = 3; break  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                            else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                                if _lh[_bi3] <= _tp_p: _res = 2; break  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                                if _hh[_bi3] >= _sl_p: _res = 3; break  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        _dl2.append(1 if _act == 1 else -1)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                        _lbl2.append(_res)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                    _mpl2 = max(len(_p) for _p in _pl2)  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
                    _pp2  = np.zeros((len(_pl2), _mpl2))  # 0으로 채워진 배열을 만든다  # Creates a zero-filled array
                    for _ji, _p in enumerate(_pl2): _pp2[_ji, :len(_p)] = _p  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                    _r2 = agent.train_step(
                        _xt2,
                        torch.from_numpy(_pp2).float(),  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
                        torch.from_numpy(np.array(_dl2)).float(),  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
                        torch.from_numpy(np.array(_el2)).long(),  # 텐서를 정수형(int64)으로 변환한다  # Casts tensor to int64
                        torch.from_numpy(np.array(_lbl2)).long(),  # 텐서를 정수형(int64)으로 변환한다  # Casts tensor to int64
                        torch.from_numpy(np.array(_al2)).float(),  # 텐서를 실수형(float32)으로 변환한다  # Casts tensor to float32
                        last_step_only=True,
                    )
                    epoch_loss        += _r2.loss
                    epoch_critic_loss += _r2.critic_loss  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
                    epoch_fp_loss     += _r2.fp_loss
                    epoch_dir_sym     += _r2.dir_sym_loss
                    n_batches += 1

            avg_loss        = epoch_loss        / max(1, n_batches)  # 가장 큰 값을 찾는다
            avg_critic_loss = epoch_critic_loss / max(1, n_batches)  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
            avg_fp_loss     = epoch_fp_loss     / max(1, n_batches)  # 가장 큰 값을 찾는다
            avg_dir_sym     = epoch_dir_sym     / max(1, n_batches)  # 가장 큰 값을 찾는다

            # 그래디언트 노름 — result.grad_norm 은 clip 이전의 원시 노름 (pre-clip)
            # NOTE: .grad 를 직접 읽으면 clip_grad_norm_ 이 적용된 후라 항상 ≤clip 으로 보임.
            # P4: grad_clip=5.0 으로 상향 — 복합 loss(actor+critic+FP) 에서 합법적인
            # 그래디언트가 1.0을 초과해도 clip되지 않도록. 진짜 폭발(>5.0)만 차단.
            grad_norm = result.grad_norm  # pre-clip norm returned by train_step()

            # label distribution for this epoch (last batch only — approximate)
            # label: 0=HOLD/timeout, 1=LONG-TP, 2=SHORT-TP, 3=SL-hit
            lc = np.bincount(np.array(labels_list), minlength=4)  # 파이썬 리스트를 넘파이 배열로 변환한다  # Converts Python sequence to NumPy array

            epoch_elapsed = time.time() - epoch_start  # 전체 데이터를 몇 번 반복해서 학습할지 결정  # Number of full passes over the training dataset
            fold_epoch_times.append(epoch_elapsed)  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list

            # ETA: rolling average of last 3 epochs x remaining epochs
            avg_t = sum(fold_epoch_times[-3:]) / min(len(fold_epoch_times), 3)  # 리스트/배열/문자열의 길이(개수)를 구한다  # Returns the number of items
            remaining = args.epochs - epoch
            eta_sec = avg_t * remaining
            if eta_sec >= 3600:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"  # 문자열 안에 변수 값을 넣어 만든다
            elif eta_sec >= 60:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"  # 문자열 안에 변수 값을 넣어 만든다
            elif remaining == 0:  # 이전 조건이 거짓이고 이 조건이 참일 때 실행한다  # Branch: previous condition was False, try this one
                eta_str = "done"
            else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                eta_str = f"{int(eta_sec)}s"  # 문자열 안에 변수 값을 넣어 만든다

            val_ev, val_winrate, diag, pnl_series_epoch = evaluate_model(agent, val_data, seq_len=args.seq_len)
            # Secondary symbols val 평균 합산 (가중치: 균등)
            if _extra_val_datas:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                _ev_sum = val_ev; _wr_sum = val_winrate; _cnt = 1  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
                for _, _vd2 in _extra_val_datas:  # 시퀀스에서 항목을 하나씩 꺼내어 반복한다  # Loop: iterate over each item in the sequence
                    _ev2, _wr2, _, _ = evaluate_model(agent, _vd2, seq_len=args.seq_len)
                    _ev_sum += _ev2; _wr_sum += _wr2; _cnt += 1  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
                val_ev     = _ev_sum / _cnt  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
                val_winrate = _wr_sum / _cnt
            # Line 1: primary metrics for checkpointing
            print(  # 결과를 화면에 출력한다  # Prints output to stdout
                # 문자열 안에 변수 값을 넣어 만든다
                f"  Epoch {epoch:2d}/{args.epochs} | Loss: {avg_loss:.4f} | GN: {grad_norm:.4f} | "
                f"ETA: {eta_str} | EV/trade: {val_ev:+.4f} | WR: {val_winrate:.1%} | "  # 문자열 안에 변수 값을 넣어 만든다
                f"LP={diag['long_prec']:.0f}%(n={diag['long_n']})  "  # 문자열 안에 변수 값을 넣어 만든다
                f"SP={diag['short_prec']:.0f}%(n={diag['short_n']})"  # 문자열 안에 변수 값을 넣어 만든다
            )
            # Line 2: diagnostic breakdown
            # lc: simulated label distribution (last batch) — 0=timeout,1=LONG-TP,2=SHORT-TP,3=SL
            _lc_sum = max(1, lc.sum())  # 가장 큰 값을 찾는다
            print(  # 결과를 화면에 출력한다  # Prints output to stdout
                # 문자열 안에 변수 값을 넣어 만든다
                f"           Pre[L={diag['before_long']} S={diag['before_short']} H={diag['before_hold']}]"
                # 문자열 안에 변수 값을 넣어 만든다
                f"  Post[L={diag['long']} S={diag['short']} Hf={diag['hold_filtered']}]"
                # 문자열 안에 변수 값을 넣어 만든다
                f"  mP[H:{diag['mean_p_hold']:.2f} L:{diag['mean_p_long']:.2f} S:{diag['mean_p_short']:.2f}]"
                # 문자열 안에 변수 값을 넣어 만든다
                f"  RawLbl tp={diag['raw_tp']} sl={diag['raw_sl']} n={diag['raw_neutral']}"
                f"  SimLbl[TP={lc[1]+lc[2]} SL={lc[3]} TO={lc[0]}]({_lc_sum})"  # 문자열 안에 변수 값을 넣어 만든다
            )
            # Line 3: P4 gradient diagnosis — per-component loss breakdown
            # GN_WARNING fires when pre-clip GN > 5.0 (grad_clip 기준)
            gn_tag = " ⚠GN_HIGH" if grad_norm > 5.0 else ""
            print(  # 결과를 화면에 출력한다  # Prints output to stdout
                # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
                # Critic: estimates state-value function V(s)
                f"           [LossBreak] Actor={avg_loss - avg_critic_loss * agent.config.critic_coef - avg_fp_loss * agent.config.fp_coef:.4f}"
                # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
                # Critic: estimates state-value function V(s)
                f"  Critic={avg_critic_loss:.4f}  FP={avg_fp_loss:.4f}  DirSym={avg_dir_sym:.4f}{gn_tag}"
            )
            # Line 4: QNG / QFI diagnostics (only when use_qng=True)
            # src.models.qng_optimizer 모듈에서 DiagonalQNGOptimizer as _DQNG를 가져온다
            # Import DiagonalQNGOptimizer as _DQNG from src.models.qng_optimizer module
            from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
            if isinstance(agent.optimizer, _DQNG):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                _qfi = agent.optimizer.get_qfi_stats()
                _bp_tag = " ⚠BARREN" if grad_norm < 0.5 else ""
                print(  # 결과를 화면에 출력한다  # Prints output to stdout
                    f"           [QNG] QFI mean={_qfi['qfi_mean']:.4f}"  # 문자열 안에 변수 값을 넣어 만든다
                    f"  min={_qfi['qfi_min']:.4f}  max={_qfi['qfi_max']:.4f}"  # 문자열 안에 변수 값을 넣어 만든다
                    f"  lr_q={agent.optimizer.lr_quantum:.4f}"  # 문자열 안에 변수 값을 넣어 만든다
                    f"  steps={agent.optimizer._step_count}{_bp_tag}"  # 문자열 안에 변수 값을 넣어 만든다
                )
            val_sniper = val_ev  # alias for checkpoint comparisons below
            auto_ctrl.step_epoch(diag)     # entropy_reg / lr 자동 조정

            # ── per-epoch log (viz용) ─────────────────────────────────────
            # src.models.qng_optimizer 모듈에서 DiagonalQNGOptimizer as _DQNG2를 가져온다
            # Import DiagonalQNGOptimizer as _DQNG2 from src.models.qng_optimizer module
            from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG2
            _qfi_mean_log = 0.0
            if isinstance(agent.optimizer, _DQNG2):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                _qfi_mean_log = agent.optimizer.get_qfi_stats().get("qfi_mean", 0.0)
            _actor_loss_log = (avg_loss
                               # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
                               # Critic: estimates state-value function V(s)
                               - avg_critic_loss * agent.config.critic_coef
                               - avg_fp_loss     * agent.config.fp_coef)
            _epoch_log.append({  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                "epoch":       epoch,
                "loss":        avg_loss,
                "actor_loss":  _actor_loss_log,
                "critic_loss": avg_critic_loss,  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
                "fp_loss":     avg_fp_loss,
                "dir_sym":     avg_dir_sym,
                "ev":          val_ev,
                "wr":          val_winrate,
                "qfi_mean":    _qfi_mean_log,
                "mean_p_hold":  diag.get("mean_p_hold",  1/3),
                "mean_p_long":  diag.get("mean_p_long",  1/3),
                "mean_p_short": diag.get("mean_p_short", 1/3),
                "n_trades":    diag.get("long_n", 0) + diag.get("short_n", 0),
            })

            # P3 fix: minimum trades guard.
            # EV/trade with few samples is statistically meaningless:
            # 9 trades × leverage=25 → EV=+0.058 from 2 lucky wins beats
            # 6000 trades EV=+0.0014. Only checkpoint when sample is credible.
            MIN_TRADES_FOR_CKPT = 15  # 체크포인트(저장된 모델 상태) 관련 처리를 한다  # Checkpoint: saved model state for resuming training
            n_trades_this_epoch = diag["long_n"] + diag["short_n"]
            eligible = n_trades_this_epoch >= MIN_TRADES_FOR_CKPT  # 체크포인트(저장된 모델 상태) 관련 처리를 한다  # Checkpoint: saved model state for resuming training

            # Fix #3: compare within THIS fold only (not across folds)
            if val_sniper > best_fold_sniper and eligible:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                best_fold_sniper = val_sniper
                no_improve_count = 0
                best_epoch_pnl_series = pnl_series_epoch
                # 폴더와 파일 이름을 합쳐 경로를 만든다
                # Joins path components into a single path string
                ckpt_path = os.path.join(args.checkpoint_dir, f"agent_best_fold{k}.pt")
                agent.save_checkpoint(ckpt_path)  # 정수로 변환한다  # Checkpoint: saved model state for resuming training
                # 결과를 화면에 출력한다
                # Prints output to stdout
                print(f"  ⭐ New Best [Fold {k}] Sniper={val_sniper:.4f}  Trades={n_trades_this_epoch}  Saved: {ckpt_path}")
                # ── Visualization (fold best 업데이트 시) ─────────────────
                _best_metrics = {
                    "ev":           val_ev,
                    "wr":           val_winrate,
                    "n_trades":     n_trades_this_epoch,
                    "loss":         avg_loss,
                    "actor_loss":   _actor_loss_log,
                    "critic_loss":  avg_critic_loss,  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측  # Critic: estimates state-value function V(s)
                    "fp_loss":      avg_fp_loss,
                    "dir_sym":      avg_dir_sym,
                    "qfi_mean":     _qfi_mean_log,
                    "mean_p_long":  diag.get("mean_p_long",  1/3),
                    "mean_p_short": diag.get("mean_p_short", 1/3),
                    "avg_pnl":      0.0,   # backtest 없이는 0
                }
                _fold_best_history.append({"fold": k, **_best_metrics})  # 리스트의 맨 뒤에 항목을 추가한다  # Appends an item to the end of the list
                try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
                    viz_path = viz.plot_rl_fold_best(
                        fold=k, epoch=epoch,
                        metrics=_best_metrics,
                        fold_history=_fold_best_history,
                        epoch_log=_epoch_log,  # 전체 데이터를 몇 번 반복해서 학습할지 결정  # Number of full passes over the training dataset
                    )
                    print(f"  [viz] → {viz_path}")  # 결과를 화면에 출력한다  # Prints output to stdout
                except Exception as _ve:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
                    print(f"  [viz] skipped ({_ve})")  # 결과를 화면에 출력한다  # Prints output to stdout
                # Drive 동기화 — Fold Best
                _sync_to_drive(ckpt_path, args.drive_dir, label=f"Fold{k} Best")  # 문자열 안에 변수 값을 넣어 만든다  # Checkpoint: saved model state for resuming training
                # Also update global best → agent_best.pt for final deployment
                if val_sniper > best_sniper_score:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                    best_sniper_score = val_sniper
                    # 폴더와 파일 이름을 합쳐 경로를 만든다
                    # Joins path components into a single path string
                    global_best_path = os.path.join(args.checkpoint_dir, "agent_best.pt")
                    agent.save_checkpoint(global_best_path)  # 정수로 변환한다  # Checkpoint: saved model state for resuming training
                    # Drive 동기화 — Global Best (agent_best.pt)
                    # 문자열 안에 변수 값을 넣어 만든다
                    _sync_to_drive(global_best_path, args.drive_dir, label=f"Fold{k} GlobalBest")
            else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                if not eligible:  # 조건이 거짓일 때만 실행한다  # Branch: executes only when condition is False
                    # 결과를 화면에 출력한다
                    # Prints output to stdout
                    print(f"  [MinTrades] Ep{epoch} trades={n_trades_this_epoch} < {MIN_TRADES_FOR_CKPT} → skip ckpt")
                else:  # 앞의 모든 조건이 거짓일 때 실행한다  # Branch: all previous conditions were False
                    no_improve_count += 1
                    if no_improve_count >= patience_epochs:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
                        # 결과를 화면에 출력한다
                        # Prints output to stdout
                        print(f"  [EarlyStop] No improvement for {patience_epochs} epochs → stopping Fold {k} at Ep{epoch}")
                        break  # 현재 반복문을 즉시 탈출한다  # Exit the enclosing loop immediately

        if best_epoch_pnl_series:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            all_folds_pnl_series.extend(best_epoch_pnl_series)  # 리스트 뒤에 다른 리스트의 항목들을 이어 붙인다  # Extends list by appending all items from iterable

        # Fix #5: Roll back to best-epoch weights before next fold.
        # Without this, fold k+1 starts from epoch-20 weights (often worse
        # than the best epoch). Loading best fold k weights + resetting the
        # optimizer at the top of fold k+1 gives fold k+1 the cleanest start.
        fold_best_path = os.path.join(args.checkpoint_dir, f"agent_best_fold{k}.pt")  # 폴더와 파일 이름을 합쳐 경로를 만든다  # Joins path components into a single path string
        if os.path.exists(fold_best_path):  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
            # 파일에서 저장된 모델/텐서를 불러온다
            # Loads a tensor/model from disk
            ckpt = torch.load(fold_best_path, map_location=agent.device, weights_only=False)
            agent.load_state_dict(ckpt["model_state"], strict=True)  # 딕셔너리(키-값 쌍)를 만든다  # Checkpoint: saved model state for resuming training
            print(f"  [fold {k}] Rolled back to best-epoch weights "  # 결과를 화면에 출력한다  # Prints output to stdout
                  f"(EV={best_fold_sniper:+.4f}) → Fold {k+1} starts from this state.")  # 문자열 안에 변수 값을 넣어 만든다

            # Phase 2: production-gate diagnostic on the rolled-back best model
            # Runs select_action() (all gates) vs raw forward() on 50 val bars.
            # Diagnostic only — never affects checkpointing or training.
            try:  # 오류가 날 수 있는 코드 블록을 시도한다  # Try block: attempt code that might raise an exception
                gate_stats = prod_eval_quick(agent, val_data, seq_len=args.seq_len, n_sample=50)
                auto_ctrl.step_fold(gate_stats)   # lindblad_threshold 자동 조정
            except Exception as _pe:  # 오류가 발생했을 때 처리하는 블록  # Except block: handles a raised exception
                print(f"  [ProdEval] skipped ({_pe})")  # 결과를 화면에 출력한다  # Prints output to stdout

    print(f"\n✅ Training complete. Global best Sniper={best_sniper_score:.4f} "  # 결과를 화면에 출력한다  # Prints output to stdout
          f"→ checkpoints/quantum_v2/agent_best.pt")  # 문자열 안에 변수 값을 넣어 만든다  # Checkpoint: saved model state for resuming training

    if all_folds_pnl_series:  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
        pnl_df = pd.DataFrame(all_folds_pnl_series)
        pnl_df['ts'] = pd.to_datetime(pnl_df['ts'])
        pnl_df = pnl_df.sort_values(by="ts").reset_index(drop=True)
        pnl_df['cumulative_pnl_pct'] = pnl_df['pnl_pct'].cumsum()  # 모든 값을 더한다
        
        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)  # 필요한 폴더가 없으면 새로 만든다  # Creates directory (and parents) if they do not exist
        
        save_path = "reports/walk_forward_results.csv"
        pnl_df.to_csv(save_path, index=False)  # 비평가(Critic): 현재 상태에서 앞으로 받을 보상을 예측
        print(f"\n[Walk-Forward] 누적 손익 결과 저장 완료: {save_path}")  # 결과를 화면에 출력한다  # Prints output to stdout


if __name__ == "__main__":  # 조건이 참일 때만 실행한다  # Branch: executes only when condition is True
    main()