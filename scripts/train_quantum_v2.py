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

import argparse
import os
import shutil
import sys
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows CP949 인코딩 픽스
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ─── Google Drive 유틸리티 ────────────────────────────────────────────────────

def _mount_drive() -> bool:
    """Colab 환경에서 Google Drive를 마운트. 이미 마운트됐으면 스킵."""
    try:
        from google.colab import drive  # type: ignore
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive", force_remount=False)
            print("  [Drive] Google Drive mounted at /content/drive")
        else:
            print("  [Drive] Google Drive already mounted")
        return True
    except ImportError:
        return False  # Colab 환경 아님 — 무시


def _sync_to_drive(local_path: str, drive_dir: str, label: str = "") -> None:
    """로컬 체크포인트를 Google Drive 디렉터리로 즉시 복사.

    Args:
        local_path: 복사할 원본 파일 경로 (로컬 런타임)
        drive_dir:  Drive 목적지 디렉터리 (예: /content/drive/MyDrive/quantum_v2)
        label:      로그 출력용 설명 (예: "Fold3 GlobalBest")
    """
    if not drive_dir:
        return
    try:
        os.makedirs(drive_dir, exist_ok=True)
        dst = os.path.join(drive_dir, os.path.basename(local_path))
        shutil.copy2(local_path, dst)
        tag = f"[{label}] " if label else ""
        print(f"  [Drive] {tag}{os.path.basename(local_path)} → {drive_dir}")
    except Exception as e:
        print(f"  [Drive] ⚠ 복사 실패 ({local_path} → {drive_dir}): {e}")

from src.models.integrated_agent import build_quantum_agent, AgentConfig
from src.data.data_client import DataClient
from src.models.features_v4 import generate_and_cache_features_v4
from src.data.binance_client import fetch_binance_taker_history
from src.viz.training_viz import TrainingVisualizer
from src.models.labeling import (
    compute_triple_barrier_labels,
    compute_clean_barrier_labels,
    standardize_1m_ohlcv,
)

def compute_optimal_rolling_window(
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
    from itertools import groupby
    from collections import defaultdict

    close = df["close"].values if "close" in df.columns else df.iloc[:, 4].values
    log_ret = np.log(close[1:] / close[:-1])
    n = len(log_ret)

    # ── 월간 vol 레짐 (30d rolling std) ──────────────────────────────────────
    W30 = 30 * bpd
    vol30 = np.array([
        np.std(log_ret[max(0, i - W30):i]) if i > 0 else 0.0
        for i in range(W30, n + 1)
    ])
    p25, p75 = np.percentile(vol30, 25), np.percentile(vol30, 75)
    regime30 = np.where(vol30 > p75, 2, np.where(vol30 < p25, 0, 1))

    # Markov 전이 행렬 → E[L] per state
    trans = defaultdict(int)
    for i in range(len(regime30) - 1):
        trans[(regime30[i], regime30[i + 1])] += 1

    E_L_days_per_state = []
    for src in range(3):
        total_src = sum(trans[(src, dst)] for dst in range(3))
        if total_src > 0:
            p_stay = trans[(src, src)] / total_src
            e_l = 1.0 / (1.0 - p_stay + 1e-10) / bpd
            E_L_days_per_state.append(e_l)

    # 경험적 run-length 분포에서 median 계산
    run_lengths_days = []
    for k, g in groupby(regime30):
        run_lengths_days.append(sum(1 for _ in g) / bpd)
    run_arr = np.array(run_lengths_days)
    median_L = float(np.median(run_arr))
    mean_L   = float(np.mean(E_L_days_per_state)) if E_L_days_per_state else 20.0

    # ── 두 제약 계산 ───────────────────────────────────────────────────────────
    W_trades  = min_trades / (trade_rate_per_bar * bpd)          # 일 단위
    W_regime  = n_regime_stds * mean_L                            # 일 단위
    W_star    = max(W_trades, W_regime)
    W_final   = int(min(W_star, max_days))
    # 최소 30일 보장
    W_final   = max(W_final, 30)

    if verbose:
        print(f"\n  [AutoWindow] === 최적 Rolling Window 자동 계산 ===")
        print(f"  [AutoWindow] 데이터: {len(df):,}봉 ({len(df)/bpd:.0f}일)")
        print(f"  [AutoWindow] 월간 vol 레짐 E[L] = {mean_L:.1f}d  (Markov 추정)")
        print(f"  [AutoWindow]   런 분포 median = {median_L:.1f}d  "
              f"p75 = {np.percentile(run_arr,75):.1f}d")
        print(f"  [AutoWindow] 제약 1 (Sample Efficiency): "
              f"min_trades={min_trades} @ {trade_rate_per_bar*100:.1f}%/bar "
              f"-> W_trades = {W_trades:.0f}d")
        print(f"  [AutoWindow] 제약 2 (Regime Coherence): "
              f"{n_regime_stds}x E[L] = {W_regime:.0f}d")
        print(f"  [AutoWindow] W* = max({W_trades:.0f}, {W_regime:.0f}) "
              f"= {W_star:.0f}d -> cap {max_days}d -> {W_final}d")
        print(f"  [AutoWindow] ================================================\n")

    return W_final


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
    total     = len(df)
    fold_size = total // (n_folds + 1)

    def _pick_ts(df_s, pos):
        if "ts" in df_s.columns:
            return df_s["ts"].iloc[pos]
        return df_s.index[pos]

    for k in range(1, n_folds + 1):
        val_start = k * fold_size
        val_end   = (k + 1) * fold_size

        if rolling_bars is not None:
            # ── Rolling Window ─────────────────────────────────────────────
            # Fixed-size training window immediately before the val window.
            # Every fold trains on the same amount of data → no expanding bias.
            train_end   = val_start
            train_start = max(0, train_end - rolling_bars)
        else:
            # ── Expanding Window (original) ────────────────────────────────
            train_start = 0
            train_end   = val_start
            if train_end < min_train_bars:
                train_end = min(min_train_bars, total - fold_size)

        train_slice = df.iloc[train_start:train_end]
        val_slice   = df.iloc[val_start:val_end]

        if len(train_slice) < min_train_bars:
            print(f"  [Fold {k}] Skipping — train bars {len(train_slice)} < min {min_train_bars}")
            continue
        if len(val_slice) < 50:
            continue

        tr_idx = (_pick_ts(train_slice, 0),  _pick_ts(train_slice, -1))
        va_idx = (_pick_ts(val_slice,   0),  _pick_ts(val_slice,   -1))

        yield (k,
               train_slice.reset_index(drop=True),
               val_slice.reset_index(drop=True),
               tr_idx, va_idx,
               (train_start, train_end),   # absolute row positions for feature slicing
               (val_start,   val_end))

# generate_and_cache_features_v3 imported from src.models.features_v3
# (V3 = V2 base 27 + Hurst + autocorr + purity = 30-dim)

def prepare_training_data(df: pd.DataFrame, features: np.ndarray, seq_len: int = 20):
    """데이터프레임과 피처를 모델 학습용 텐서 딕셔너리로 변환."""
    n = len(df)
    if n < seq_len + 35:
        return None

    prices = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    atrs = (df['high'] - df['low']).rolling(14).mean().bfill().values

    # 원시 레이블 (1: LONG TP signal, -1: SHORT TP signal, 0: HOLD)
    raw_labels   = df['label'].values
    # 방향별 세부 레이블 (evaluate_model 손익 계산용)
    long_labels  = df['long_label'].values  if 'long_label'  in df.columns else raw_labels.copy()
    short_labels = df['short_label'].values if 'short_label' in df.columns else raw_labels.copy()

    return {
        "features":     features,
        "prices":       prices,
        "highs":        highs,
        "lows":         lows,
        "raw_labels":   raw_labels,
        "long_labels":  long_labels,
        "short_labels": short_labels,
        "atr":          atrs,
        "ts":           df['ts'].values,
        "n":            n
    }

class AdaptiveController:
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

    def __init__(self, agent):
        self.agent        = agent
        self._wr_history  : list[float] = []
        self._stall_count : int         = 0
        self._best_wr     : float       = -1.0

    # ------------------------------------------------------------------
    def step_epoch(self, diag: dict) -> None:
        """에폭 평가 직후 호출. entropy_reg / lr 자동 조정."""
        cfg = self.agent.config
        changes: list[str] = []

        # 1) entropy_reg — mP 불균형 기반
        mp_h = diag.get("mean_p_hold",  0.333)
        mp_l = diag.get("mean_p_long",  0.333)
        mp_s = diag.get("mean_p_short", 0.333)
        imbalance = max(mp_h, mp_l, mp_s) - min(mp_h, mp_l, mp_s)
        n_short_post = diag.get("short_n", 0)  # Post[S] — gate 통과 SHORT 수
        old_er = cfg.entropy_reg
        if imbalance > self.IMBAL_HIGH:
            cfg.entropy_reg = min(cfg.entropy_reg * 1.25, self.ENTROPY_MAX)
        elif imbalance > self.IMBAL_MID:
            cfg.entropy_reg = min(cfg.entropy_reg * 1.08, self.ENTROPY_MAX)
        elif imbalance < self.IMBAL_LOW and n_short_post > 0:
            # SP=0% 상태에서는 낮추지 않음 — 낮추면 VQC SHORT bias 억제가 풀려 S 소멸
            cfg.entropy_reg = max(cfg.entropy_reg * 0.90, self.ENTROPY_MIN)
        if abs(cfg.entropy_reg - old_er) > 1e-5:
            changes.append(
                f"entropy_reg {old_er:.4f}→{cfg.entropy_reg:.4f}"
                f" (imbal={imbalance:.3f})"
            )

        # 1b) Fisher threshold — 거래가 차단될 때 자동 완화
        n_long_post  = diag.get("long_n", 0)
        total_post   = n_short_post + n_long_post
        old_fish = getattr(cfg, "fisher_threshold_min", 0.38)
        if total_post == 0:
            # LP=0 AND SP=0: 전방향 차단 → 더 적극적으로 낮춤 (-0.02/epoch)
            new_fish = max(old_fish - 0.02, 0.33)
            reason = "LP=SP=0 blocked"
        elif n_short_post == 0 and n_long_post > 0:
            # SP=0%: 낮추면 저신뢰 LONG 폭발 → WR=base rate 고착. 현행 유지.
            # logit_bias_reg가 SHORT bias를 복원하도록 대기.
            new_fish = old_fish
            reason = ""
        elif n_long_post == 0 and n_short_post > 0:
            # LP=0%: 낮추면 저신뢰 SHORT 폭발 → WR=base rate 고착. 현행 유지.
            new_fish = old_fish
            reason = ""
        else:
            new_fish = old_fish
            reason = ""
        if reason and abs(new_fish - old_fish) > 1e-5:
            cfg.fisher_threshold_min = new_fish
            changes.append(f"fisher_min {old_fish:.3f}→{new_fish:.3f} ({reason})")

        # 2) lr — WR 정체 기반
        wr = diag.get("win_rate", 0.0) if "win_rate" in diag else 0.0
        # win_rate 키가 없으면 long_prec/short_prec 평균으로 추정
        if wr == 0.0:
            lp = diag.get("long_prec",  0.0)
            sp = diag.get("short_prec", 0.0)
            ln = diag.get("long_n",     0)
            sn = diag.get("short_n",    0)
            total = ln + sn
            wr = (lp * ln + sp * sn) / max(total, 1) / 100.0
        if wr > self._best_wr + 0.002:   # 0.2 % 이상 개선이어야 reset
            self._best_wr     = wr
            self._stall_count = 0
        else:
            self._stall_count += 1
        if self._stall_count >= self.LR_STALL_EPOCHS:
            old_lr = cfg.lr
            cfg.lr = max(cfg.lr * 0.5, self.LR_MIN)
            # optimizer의 실제 lr도 조정
            for pg in self.agent.optimizer.param_groups:
                pg["lr"] = cfg.lr
            self._stall_count = 0   # reset
            if abs(cfg.lr - old_lr) > 1e-9:
                changes.append(
                    f"lr {old_lr:.2e}→{cfg.lr:.2e}"
                    f" (WR stall {self.LR_STALL_EPOCHS} ep)"
                )

        if changes:
            print(f"  [AutoTune] {' | '.join(changes)}")

    # ------------------------------------------------------------------
    def step_fold(self, gate_stats: dict) -> None:
        """fold-end prod_eval 직후 호출. lindblad_threshold 자동 조정."""
        cfg = self.agent.config
        n_raw    = gate_stats.get("n_raw_trade", 0)
        n_lind   = gate_stats.get("gate_lindblad_blk", 0)
        if n_raw == 0:
            return
        pass_rate = 1.0 - (n_lind / n_raw)
        old_thr   = getattr(cfg, "lindblad_regime_threshold", 0.90)
        new_thr   = old_thr
        if pass_rate < self.LIND_PASS_FLOOR:
            new_thr = min(old_thr + 0.04, self.LIND_THR_MAX)
        elif pass_rate < self.LIND_PASS_TARGET:
            new_thr = min(old_thr + 0.01, self.LIND_THR_MAX)
        elif pass_rate > self.LIND_PASS_CEIL:
            new_thr = max(old_thr - 0.01, self.LIND_THR_MIN)
        if abs(new_thr - old_thr) > 1e-5:
            cfg.lindblad_regime_threshold = new_thr
            print(
                f"  [AutoTune] lindblad_thr {old_thr:.3f}→{new_thr:.3f}"
                f" (pass={pass_rate:.0%}, blk={n_lind}/{n_raw})"
            )


def evaluate_model(agent, val_data, seq_len=20, batch_size=128):
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

    agent.eval()
    n_samples = val_data["n"]
    indices = np.arange(seq_len + 120, n_samples - 35)

    if len(indices) == 0:
        return 0.0, 0.0, {}, []

    total_trades = 0
    tp_hits = 0
    sl_hits = 0
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
    leverage = agent.config.leverage
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

    with torch.no_grad():
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            x_list, raw_labels_list, long_labels_list, short_labels_list, atr_list, price_list = [], [], [], [], [], []
            for idx in batch_indices:
                x_list.append(val_data["features"][idx - seq_len + 1 : idx + 1])
                raw_labels_list.append(val_data["raw_labels"][idx])
                long_labels_list.append(val_data["long_labels"][idx])
                short_labels_list.append(val_data["short_labels"][idx])
                atr_list.append(val_data["atr"][idx])
                price_list.append(val_data["prices"][idx])

            x_val        = torch.from_numpy(np.array(x_list)).float().to(agent.device)
            raw_l_val    = np.array(raw_labels_list)
            long_l_val   = np.array(long_labels_list)   # +1=LONG TP, -1=LONG SL, 0=timeout
            short_l_val  = np.array(short_labels_list)  # -1=SHORT TP, +1=SHORT SL, 0=timeout
            a_val    = torch.from_numpy(np.array(atr_list)).float().to(agent.device)

            logits, _, _, _ = agent.forward(x_val, last_step_only=True)
            if logits.dim() == 3:
                logits = logits.squeeze(1)

            # temp_scaler를 우회 — 학습 중 평가에선 raw softmax 사용.
            # T = T_base*(1+beta*ATR) 가 crypto의 높은 ATR로 인해 T>>1이 되면
            # 모든 확률이 0.33 근처로 수렴해 threshold 필터링에 걸려 Hold처럼 보임.
            # raw softmax로 실제 모델이 방향을 학습하고 있는지 직접 측정.
            probs = torch.softmax(logits, dim=-1)
            max_probs, actions = probs.max(dim=-1)

            actions   = actions.cpu().numpy()
            max_probs = max_probs.cpu().numpy()

            conf_mask     = max_probs >= EVAL_CONF_THRESHOLD
            final_actions = np.where(conf_mask, actions, 0)

            n_hold_filtered += int(np.sum((actions != 0) & ~conf_mask))
            n_long           += int(np.sum(final_actions == 1))
            n_short          += int(np.sum(final_actions == 2))
            n_raw_tp         += int(np.sum(raw_l_val == 1))
            n_raw_sl         += int(np.sum(raw_l_val == -1))
            n_raw_neutral    += int(np.sum(raw_l_val == 0))

            # A1: accumulate softmax probability sums for mean computation
            probs_np = probs.cpu().numpy()          # [B, 3]
            sum_p           += probs_np.sum(axis=0) # element-wise sum per class
            n_prob_samples  += len(probs_np)

            # A2: pre-threshold raw argmax distribution
            pre_actions = np.argmax(probs_np, axis=-1)   # no confidence gate
            for cls in range(3):
                n_before[cls] += int(np.sum(pre_actions == cls))

            for i, act in enumerate(final_actions):
                if act == 0:
                    continue

                total_trades += 1
                current_pnl = 0.0

                # 실제 ATR 기반 TP/SL — LONG과 SHORT 모두 동일한 크기
                #   TP = alpha×ATR (= 6×ATR)  for both LONG and SHORT
                #   SL = beta×ATR  (= 1.5×ATR) for both LONG and SHORT
                _atr   = float(atr_list[i])
                _price = max(float(price_list[i]), 1e-8)
                _tp = BARRIER_ALPHA * _atr / _price   # TP pct for both directions
                _sl = BARRIER_BETA  * _atr / _price   # SL pct for both directions

                if act == 1:  # Long
                    n_dir_taken[0] += 1
                    rl = long_l_val[i]   # +1=LONG TP, -1=LONG SL, 0=timeout
                    if rl == 1:
                        tp_hits += 1
                        n_dir_wins[0] += 1
                        current_pnl = (_tp - fee_pct)
                    elif rl == -1:
                        sl_hits += 1
                        current_pnl = (-_sl - fee_pct)
                    else:
                        current_pnl = -fee_pct
                elif act == 2:  # Short
                    n_dir_taken[1] += 1
                    rl = short_l_val[i]  # -1=SHORT TP (7×ATR down), +1=SHORT SL (3×ATR up), 0=timeout
                    if rl == -1:
                        tp_hits += 1
                        n_dir_wins[1] += 1
                        current_pnl = (_tp - fee_pct)
                    elif rl == 1:
                        sl_hits += 1
                        current_pnl = (-_sl - fee_pct)
                    else:
                        current_pnl = -fee_pct
                
                total_pnl_pct += current_pnl
                ts = val_data["ts"][batch_indices[i]]
                pnl_series.append({"ts": ts, "pnl_pct": current_pnl * leverage})


    # Compute derived metrics
    mean_p = sum_p / max(1, n_prob_samples)          # [Hold, Long, Short]
    long_prec  = n_dir_wins[0] / max(1, n_dir_taken[0]) * 100
    short_prec = n_dir_wins[1] / max(1, n_dir_taken[1]) * 100

    _zero_diag = {
        "long": 0, "short": 0, "hold_filtered": 0,
        "raw_tp": 0, "raw_sl": 0, "raw_neutral": 0,
        "before_long": int(n_before[1]), "before_short": int(n_before[2]),
        "before_hold": int(n_before[0]),
        "mean_p_hold": round(float(mean_p[0]), 3),
        "mean_p_long": round(float(mean_p[1]), 3),
        "mean_p_short": round(float(mean_p[2]), 3),
        "long_prec": 0.0, "long_n": 0,
        "short_prec": 0.0, "short_n": 0,
        "net_bal_pct": 0.0,
    }
    if total_trades == 0:
        return 0.0, 0.0, _zero_diag, []

    win_rate      = tp_hits / total_trades
    ev_per_trade  = (total_pnl_pct * leverage) / total_trades   # net EV per trade
    net_bal_pct   = total_pnl_pct * leverage * 100              # cumulative return % on capital

    diag = {
        # post-threshold counts
        "long":          int(n_long),
        "short":         int(n_short),
        "hold_filtered": int(n_hold_filtered),
        "raw_tp":        int(n_raw_tp),
        "raw_sl":        int(n_raw_sl),
        "raw_neutral":   int(n_raw_neutral),
        # A2: pre-threshold raw argmax distribution
        "before_long":   int(n_before[1]),
        "before_short":  int(n_before[2]),
        "before_hold":   int(n_before[0]),
        # A1: mean softmax probabilities
        "mean_p_hold":   round(float(mean_p[0]), 3),
        "mean_p_long":   round(float(mean_p[1]), 3),
        "mean_p_short":  round(float(mean_p[2]), 3),
        # A3: per-direction precision
        "long_prec":     round(long_prec,  1),
        "long_n":        int(n_dir_taken[0]),
        "short_prec":    round(short_prec, 1),
        "short_n":       int(n_dir_taken[1]),
        "net_bal_pct":   round(net_bal_pct, 2),
    }
    return ev_per_trade, win_rate, diag, pnl_series


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
    agent.eval()
    n     = val_data["n"]
    # Sample from the latter half of the val set for OOS realism
    start = max(seq_len + 120, n // 2)
    end   = n - 35
    if end - start < n_sample:
        print("  [ProdEval] val set too small for prod eval — skipped.")
        return

    indices = np.sort(
        np.random.choice(np.arange(start, end), size=n_sample, replace=False)
    )

    gated = [0, 0, 0]   # HOLD / LONG / SHORT  — select_action() (all gates)
    raw   = [0, 0, 0]   # HOLD / LONG / SHORT  — raw forward()   (no gates)

    # Per-gate diagnostic counters (P5: independent check — not elif chain)
    gate_lindblad_blk = 0  # Lindblad regime blocks  (regime_prob > 0.7)
    gate_cr_blk       = 0  # Cramér-Rao filter blocks (H/purity/SNR fail)
    gate_ep_blk       = 0  # Entropy production blocks (Ṡ < threshold)
    gate_fisher_blk   = 0  # Fisher-Rao confidence threshold blocks
    max_probs_list    = []  # raw softmax max_prob per sample

    with torch.no_grad():
        for idx in indices:
            x_win = val_data["features"][idx - seq_len + 1: idx + 1]
            if len(x_win) < seq_len:
                pad   = np.zeros((seq_len - len(x_win), agent.config.feature_dim), dtype=np.float32)
                x_win = np.vstack([pad, x_win])
            x_t      = torch.from_numpy(x_win).float().unsqueeze(0).to(agent.device)
            price    = float(val_data["prices"][idx])
            atr_norm = float(val_data["atr"][idx]) / max(price, 1e-8)

            # (1) Full production path — all gates active
            act_g, _, _ = agent.select_action(x_t, atr_norm=atr_norm, mode="greedy")
            gated[act_g] += 1

            # (2) Raw model preference — bypass every gate
            logits, expvals, _, _ = agent.forward(x_t, last_step_only=True)
            if logits.dim() == 3:
                logits = logits.squeeze(1)
            probs_raw = torch.softmax(logits, dim=-1)[0]
            max_prob  = float(probs_raw.max().item())
            act_r     = int(probs_raw.argmax().item())
            raw[act_r] += 1
            max_probs_list.append(max_prob)

            # Gate-level diagnostic: all 4 gates checked INDEPENDENTLY
            # (Unlike select_action's elif chain, here each gate is counted
            # regardless of whether an earlier gate already blocked. This gives
            # the full attrition picture — which gates are actually firing.)
            if act_r != 0:   # raw model wanted to trade
                log_rets_g = x_t[0, :, 0].cpu().float().numpy()

                # Gate 1: Lindblad regime check
                # select_action 수정과 동일하게: threshold = lindblad_regime_threshold (0.90)
                purity_for_cr = 1.0  # CR filter에는 purity=1.0 고정 (select_action과 동일)
                try:
                    if agent.lindblad is not None and agent.config.use_lindblad:
                        _, _, regime_prob_t = agent.lindblad(expvals)
                        lind_thr = getattr(agent.config, "lindblad_regime_threshold", 0.90)
                        if float(regime_prob_t.mean().item()) > lind_thr:
                            gate_lindblad_blk += 1
                except Exception:
                    pass

                # Gate 2: Cramér-Rao selective entry filter
                # purity_for_cr=1.0 고정 (select_action fix와 동일 — Lindblad untrained)
                try:
                    if agent.cr_filter is not None and agent.config.use_cr_filter:
                        hurst_val  = float(agent.hurst_est._hurst_single(
                            x_t[0, :, 0].cpu().float()
                        ).item())
                        cr_result = agent.cr_filter.check(log_rets_g, hurst_val, purity_for_cr)
                        if not cr_result.allow_entry:
                            gate_cr_blk += 1
                except Exception:
                    pass

                # Gate 3: Entropy production gate (ep_threshold=0.0 → should never fire)
                try:
                    if agent.ep_estimator is not None and agent.config.use_entropy_prod:
                        agent.ep_estimator.compute()
                        if not agent.ep_estimator.allows_entry():
                            gate_ep_blk += 1
                except Exception:
                    pass

                # Gate 4: Fisher-Rao confidence threshold
                try:
                    if agent.config.use_fisher_threshold:
                        fish_thr = agent._compute_fisher_threshold(log_rets_g)
                    else:
                        fish_thr = agent.config.confidence_threshold
                    if max_prob < fish_thr:
                        gate_fisher_blk += 1
                except Exception:
                    pass

    n_raw_trade   = raw[1]   + raw[2]
    n_gated_trade = gated[1] + gated[2]
    pass_rate  = n_gated_trade / n_sample * 100
    gate_atten = (1.0 - n_gated_trade / max(1, n_raw_trade)) * 100
    avg_mp     = float(np.mean(max_probs_list)) if max_probs_list else 0.0
    fish_min   = agent.config.fisher_threshold_min
    fish_max   = agent.config.fisher_threshold_max

    print(
        f"  [ProdEval/{n_sample}] "
        f"Raw  L={raw[1]:3d} S={raw[2]:3d} H={raw[0]:3d} | "
        f"Prod L={gated[1]:3d} S={gated[2]:3d} H={gated[0]:3d} | "
        f"Pass={pass_rate:.0f}%  GateAttrition={gate_atten:.0f}%"
    )
    _nrt = max(1, n_raw_trade)
    def _blk_tag(n, total):
        pct = n / total * 100
        if pct >= 80:  return "PRIMARY"
        if pct >= 40:  return "major"
        if pct >= 10:  return "minor"
        return "none"
    print(
        f"  [GateDiag ] AvgMaxProb={avg_mp:.3f}  "
        f"FisherThr=[{fish_min:.2f},{fish_max:.2f}]  "
        f"Fisher_blk={gate_fisher_blk}/{_nrt} "
        f"({'PRIMARY BLOCKER' if gate_fisher_blk >= n_raw_trade * 0.8 else 'partial'})"
    )
    print(
        f"  [GateBreak] (of {n_raw_trade} raw-trade samples)  "
        f"Lindblad={gate_lindblad_blk}/{_nrt}({_blk_tag(gate_lindblad_blk,_nrt)})  "
        f"CR={gate_cr_blk}/{_nrt}({_blk_tag(gate_cr_blk,_nrt)})  "
        f"EP={gate_ep_blk}/{_nrt}({_blk_tag(gate_ep_blk,_nrt)})  "
        f"Fisher={gate_fisher_blk}/{_nrt}({_blk_tag(gate_fisher_blk,_nrt)})"
    )

    return {
        "n_raw_trade":      n_raw_trade,
        "gate_lindblad_blk": gate_lindblad_blk,
        "gate_cr_blk":       gate_cr_blk,
        "gate_ep_blk":       gate_ep_blk,
        "gate_fisher_blk":   gate_fisher_blk,
        "pass_rate":         pass_rate / 100.0,
    }


def _load_symbol_data_for_rl(sym: str, args, dc):
    """심볼별 데이터 로드·레이블링·피처 생성. (df_labeled, all_features) 반환."""
    import pickle
    from src.data.binance_client import fetch_binance_taker_history as _ftk
    from src.models.labeling import (
        compute_clean_barrier_labels as _ccbl,
        standardize_1m_ohlcv as _std,
    )
    if getattr(args, "feat_ver", 4) == 5:
        from src.models.features_v5 import generate_and_cache_features_v5 as _gcf
    else:
        from src.models.features_v4 import generate_and_cache_features_v4 as _gcf

    _end_dt = datetime.now(timezone.utc)
    if args.end_date:
        _end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc, hour=23, minute=59)
    _start_str = (args.start_date if args.start_date
                  else (_end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d"))
    _start_ms = int(pd.Timestamp(_start_str).timestamp() * 1000)
    _end_ms   = int(_end_dt.timestamp() * 1000)

    print(f"\n  [Data/{sym}] Fetching OHLCV {_start_str} ~ {_end_dt.strftime('%Y-%m-%d')} ...")
    df_raw = dc.fetch_training_history(
        symbol=sym, timeframe=args.timeframe, start_date=_start_str,
        end_ms=_end_ms, cache_dir="data"
    )
    if df_raw is None or df_raw.empty:
        print(f"  [Data/{sym}] ⚠ 빈 데이터 → 스킵")
        return None, None

    df_clean = _std(df_raw) if args.timeframe == "1m" else df_raw.copy()

    # Funding rate
    df_funding = dc.fetch_funding_history(sym, _start_ms, _end_ms, cache_dir="data")
    if not df_funding.empty:
        df_funding["ts"] = (pd.to_datetime(df_funding["ts_ms"], unit="ms", utc=True)
                            .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
        df_clean = df_clean.merge(df_funding[["ts", "funding_rate"]], on="ts", how="left")
        df_clean["funding_rate"] = df_clean["funding_rate"].ffill().fillna(0.0)

    # Open interest
    df_oi = dc.fetch_open_interest_history(sym, _start_ms, _end_ms, interval="1h", cache_dir="data")
    if not df_oi.empty:
        df_oi["ts"] = (pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
                       .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
        df_clean = df_clean.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
        df_clean["open_interest"] = df_clean["open_interest"].ffill().fillna(0.0)

    # Binance CVD
    _te = _end_dt.strftime("%Y-%m-%d")
    _tc = f"data/binance_taker_{sym}_{args.timeframe}_{_start_str.replace('-','')}_{_te.replace('-','')}.csv"
    try:
        df_taker = _ftk(symbol=sym, interval=args.timeframe,
                        start_date=_start_str, end_date=_te,
                        cache_path=_tc, verbose=False)
        if not df_taker.empty:
            df_clean = df_clean.merge(df_taker[["ts", "taker_buy_volume"]], on="ts", how="left")
            df_clean["taker_buy_volume"] = df_clean["taker_buy_volume"].fillna(0.0)
    except Exception:
        pass

    # Labels
    _bars_h = 96 if args.timeframe == "15m" else 60
    df_lbl_raw = _ccbl(df_clean, alpha=4.0, beta=1.5, hold_band=1.5, hold_h=20, h=_bars_h)
    df_labeled = df_lbl_raw[df_lbl_raw["label"] != 2].reset_index(drop=True)
    _lv = df_labeled["label"].value_counts().to_dict()
    print(f"  [Data/{sym}] LONG={_lv.get(1,0)} SHORT={_lv.get(-1,0)} HOLD={_lv.get(0,0)} "
          f"DISCARD={(df_lbl_raw['label']==2).sum()}")

    # Features
    _fver  = getattr(args, "feat_ver", 4)
    _cache = f"data/feat_cache_{sym}_{args.timeframe}_{args.days}d_v{'5' if _fver==5 else '4cvd'}.npy"
    all_feat_raw = _gcf(df_clean, _cache)
    _keep = df_lbl_raw["label"].values != 2
    all_feat = all_feat_raw[_keep]

    # bc_scaler
    _scaler_path = os.path.join(args.checkpoint_dir, "bc_scaler.pkl")
    if os.path.isfile(_scaler_path):
        with open(_scaler_path, "rb") as _f:
            _sc = pickle.load(_f)
        all_feat = _sc.transform(all_feat).astype(np.float32)

    if len(all_feat) != len(df_labeled):
        print(f"  [Data/{sym}] ⚠ Feature/label mismatch ({len(all_feat)} vs {len(df_labeled)}) → 스킵")
        return None, None

    return df_labeled, all_feat


def main():
    parser = argparse.ArgumentParser(description="Quantum V2 Training Pipeline")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--symbols", default=None,
        help="멀티심볼 쉼표구분 (예: BTCUSDT,ETHUSDT,SOLUSDT). 지정 시 --symbol 무시.")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start_date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end_date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints/quantum_v2")
    parser.add_argument("--leverage", type=float, default=25.0)
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument(
        "--drive-dir", default=None,
        help="Google Drive 저장 경로 (예: /content/drive/MyDrive/quantum_v2). "
             "지정 시 Best 체크포인트마다 Drive로 즉시 복사."
    )
    parser.add_argument(
        "--rolling-window", type=str, default=None,
        help="Rolling window 학습 크기 (일 단위 정수 or 'auto'). "
             "'auto' 시 데이터 기반 Markov E[L] + Sample Efficiency 제약으로 자동 계산. "
             "미설정 시 전통 Expanding Window 사용."
    )
    parser.add_argument(
        "--pretrain-ckpt", default=None,
        help="BC 사전학습 체크포인트 경로 (예: checkpoints/quantum_v2/agent_bc_pretrained.pt). "
             "지정 시 RL 학습 시작 전 모델 가중치만 로드 (optimizer/scheduler는 초기화 유지)."
    )
    parser.add_argument(
        "--seq-len", type=int, default=96,
        help="슬라이딩 윈도우 길이 (봉 단위). BC와 동일하게 맞춰야 함. 기본값=96 (24시간)."
    )
    parser.add_argument(
        "--feat-ver", type=int, default=4, choices=[4, 5],
        help="Feature version: 4=V4 28-dim (default), 5=V5 48-dim (Frenet, dependency-free)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="랜덤 시드 고정 (재현 가능, 예: --seed 6)"
    )
    parser.add_argument(
        "--tp-mult", type=float, default=4.0,
        help="시뮬레이션 보상: TP 배수 (ATR 단위). 레이블 생성 alpha=4.0과 일치시킬 것. 기본=4.0"
    )
    parser.add_argument(
        "--sl-mult", type=float, default=1.5,
        help="시뮬레이션 보상: SL 배수 (ATR 단위). 레이블 생성 beta=1.5와 일치시킬 것. 기본=1.5"
    )
    parser.add_argument(
        "--entropy-reg", type=float, default=None,
        help="초기 entropy_reg 강제 설정 (기본: AgentConfig 기본값 0.05). HOLD 붕괴 시 0.12 권장"
    )
    args = parser.parse_args()

    # ── Seed 고정 ────────────────────────────────────────────────────────────
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"  [Seed] 랜덤 시드 고정: {args.seed} (재현 가능)")

    # ── Drive 초기화 ──────────────────────────────────────────────────────────
    if args.drive_dir:
        mounted = _mount_drive()
        if mounted:
            os.makedirs(args.drive_dir, exist_ok=True)
            print(f"  [Drive] 저장 경로: {args.drive_dir}")
        else:
            print("  [Drive] ⚠ Colab 환경이 아님 — --drive-dir 무시됨")

    # ── 심볼 목록 결정 ──────────────────────────────────────────────────────
    _symbol_list = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else [args.symbol]
    )
    print(f"🚀 [Phase 5] Quantum V2 통합 학습 시작: {_symbol_list}  TF={args.timeframe}")

    dc = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")

    # ── 심볼별 데이터 로드 (primary + secondary) ─────────────────────────────
    symbol_datasets: list = []   # list of (sym, df_labeled, all_features)
    for _sym in _symbol_list:
        _df_lab, _all_feat = _load_symbol_data_for_rl(_sym, args, dc)
        if _df_lab is not None:
            symbol_datasets.append((_sym, _df_lab, _all_feat))

    if not symbol_datasets:
        print("  [Data] ERROR: 유효 데이터 없음"); return

    # Primary symbol (fold timing 기준)
    _primary_sym, df_labeled, all_features = symbol_datasets[0]
    print(f"\n  [Data] Primary={_primary_sym}  총 심볼={len(symbol_datasets)}  "
          f"Primary bars={len(df_labeled):,}")

    # ── Rolling Window 설정 ──────────────────────────────────────────────────
    _bars_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
    bpd = _bars_per_day.get(args.timeframe, 96)
    rolling_bars = None
    if args.rolling_window:
        rw_str = str(args.rolling_window).strip().lower()
        if rw_str == "auto":
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
        else:
            rw_days = int(rw_str)
        rolling_bars = rw_days * bpd
        print(f"  [Mode] ROLLING WINDOW: {rw_days}d = {rolling_bars} bars per fold")
    else:
        print(f"  [Mode] EXPANDING WINDOW (all history up to each val start)")

    fold_gen = walk_forward_folds(df_labeled, n_folds=args.n_folds, rolling_bars=rolling_bars)
    _feat_dim = 48 if getattr(args, "feat_ver", 4) == 5 else 28
    _entropy_reg_init = getattr(args, "entropy_reg", None) or 0.05
    config = AgentConfig(leverage=args.leverage, checkpoint_dir=args.checkpoint_dir, confidence_threshold=args.confidence, feature_dim=_feat_dim, entropy_reg=_entropy_reg_init)
    agent = build_quantum_agent(config=config, device=torch.device(args.device))

    # ── Phase 0 BC 사전학습 가중치 로드 ────────────────────────────────────────
    # BC 체크포인트는 모델 가중치만 로드 (optimizer/scheduler는 RL 학습용으로 신규 초기화 유지).
    # AlphaGo 원칙: SL Policy Net 가중치 → RL Fine-tuning 출발점으로만 사용.
    if args.pretrain_ckpt:
        if not os.path.isfile(args.pretrain_ckpt):
            print(f"  [BC] ⚠ --pretrain-ckpt 파일 없음: {args.pretrain_ckpt} — 무시하고 계속")
        else:
            _bc_ckpt = torch.load(args.pretrain_ckpt,
                                  map_location=torch.device(args.device),
                                  weights_only=False)
            _missing, _unexpected = agent.load_state_dict(
                _bc_ckpt["model_state"], strict=False
            )
            if _missing:
                print(f"  [BC] ⚠ Missing keys ({len(_missing)}): {_missing[:5]}")
            if _unexpected:
                print(f"  [BC] ⚠ Unexpected keys ({len(_unexpected)}): {_unexpected[:5]}")
            print(
                f"  [BC] ✓ BC 가중치 로드 완료: {args.pretrain_ckpt}\n"
                f"       global_step={_bc_ckpt.get('global_step', 0)} | "
                f"optimizer/scheduler는 RL 신규 초기화"
            )

            # BC Fine-Tune 모드: VQC 재초기화 없이 BC 파라미터 그대로 유지
            # 이전: near-identity(randn*0.01) → BC 지식 손실, RL이 처음부터 재탐험
            # 현재: BC VQC 방향벡터 보존 → RL이 좋은 초기점에서 fine-tune
            for name, param in agent.named_parameters():
                if "vqc_weights" in name:
                    param.requires_grad = True   # BC 동결 해제만
            print("  [BC] ✅ VQC weights BC 파라미터 유지 (fine-tune 모드, 재초기화 없음)")

            # Pretrain 지식을 보존하면서 VQC 학습 가능하도록 LR 조정
            # Fine-tune LR: BC 파라미터 보존이 목표이므로 VQC LR 대폭 축소
            # near-identity 시: VQC가 처음부터 학습 → lr_quantum=0.03 필요
            # fine-tune 시: 이미 좋은 방향 → 큰 LR은 BC 지식을 덮어씀
            config.lr = 5e-5               # 클래식 레이어: 보수적 fine-tune
            config.eta_base = 0.0005
            if hasattr(agent.optimizer, 'lr_quantum'):
                agent.optimizer.lr_quantum = 0.005  # VQC: BC의 1/10 수준 (0.03→0.005)
            print(f"  [BC] 📉 Fine-Tune LR: 클래식={config.lr}, VQC=0.005 (BC 지식 보존)")

            # ── PAC-Bayes J: N_eff 계산 (Bartlett 자기상관 보정) ─────────────
            # N_eff = N / (1 + 2Σρ(k))  — 금융 시계열 실효 독립 샘플 수
            # 창이 짧을수록(N_eff↓) λ = C/N_eff 자동 증가 → BC 사전지식 강화
            _close = df_labeled["close"].values
            _lr_arr = np.log(_close[1:] / _close[:-1])
            _max_lag = min(20, len(_lr_arr) // 4)
            _acf_pos_sum = 0.0
            for _lag in range(1, _max_lag + 1):
                _rho = float(np.corrcoef(_lr_arr[:-_lag], _lr_arr[_lag:])[0, 1])
                if _rho > 0:
                    _acf_pos_sum += _rho
            _n_eff = len(_lr_arr) / max(1.0 + 2.0 * _acf_pos_sum, 1.0)
            config.pac_bayes_n_eff = float(_n_eff)
            _lam = config.pac_bayes_coef / _n_eff
            print(f"  [PAC-Bayes] N_eff={_n_eff:.0f} (Bartlett)  λ={_lam:.2e}"
                  f"  (C={config.pac_bayes_coef})")

            # BC prior 저장 — RL 학습 중 이 기준점으로 당겨짐
            agent.set_bc_prior(agent.state_dict())
            print(f"  [PAC-Bayes] BC prior 저장 완료 ({len(agent._bc_prior_params)} 텐서, VQC 제외)")

            # ── Method G: RL 단계부터 Spectral Norm 활성화 ────────────────────
            # BC에서는 False (HOLD 억압 방지), RL에서는 True (logit 폭발 차단)
            # encoder는 이미 생성되어 있으므로 기존 Linear에 SN 훅을 사후 적용.
            try:
                from torch.nn.utils import spectral_norm as _sn
                _ql = agent.encoder.quantum_layer
                # logit_proj: Sequential([Linear, GELU, Dropout, Linear])
                if isinstance(_ql.logit_proj, torch.nn.Sequential):
                    _linears = [m for m in _ql.logit_proj if isinstance(m, torch.nn.Linear)]
                    for _lin in _linears:
                        _sn(_lin)
                elif isinstance(_ql.logit_proj, torch.nn.Linear):
                    _sn(_ql.logit_proj)
                # classical_head
                if isinstance(_ql.classical_head, torch.nn.Linear):
                    _sn(_ql.classical_head)
                config.use_spectral_norm = True
                print("  [Method G] Spectral Norm 사후 적용 완료 (logit_proj + classical_head)")
            except Exception as _e:
                print(f"  [Method G] ⚠ Spectral Norm 적용 실패: {_e} — 스킵")

    best_sniper_score = -float('inf')
    auto_ctrl = AdaptiveController(agent)   # 자동 하이퍼파라미터 컨트롤러
    viz = TrainingVisualizer(save_dir="reports/viz")
    _fold_best_history: list = []   # fold-best 기록 (viz용)
    all_folds_pnl_series: list = []

    def _fmt_date(idx_val):
        """Index value -> KST date string (YYYY-MM-DD)."""
        try:
            ts = pd.Timestamp(idx_val)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            ts_kst = ts.tz_convert("Asia/Seoul")
            return ts_kst.strftime("%Y-%m-%d")
        except Exception:
            return str(idx_val)[:10]

    for k, train_df, val_df, tr_idx, va_idx, train_abs, val_abs in fold_gen:
        tr_start = _fmt_date(tr_idx[0])
        tr_end   = _fmt_date(tr_idx[1])
        va_start = _fmt_date(va_idx[0])
        va_end   = _fmt_date(va_idx[1])
        mode_tag = "ROLLING" if rolling_bars else "EXPANDING"
        print(
            f"\n[Fold {k}/{args.n_folds}] [{mode_tag}] "
            f"Train: {len(train_df):,} bars  {tr_start} ~ {tr_end} | "
            f"Val: {len(val_df):,} bars  {va_start} ~ {va_end}"
        )

        # 피처 슬라이싱: absolute row indices (rolling 모드에서 train_start != 0)
        train_features = all_features[train_abs[0]:train_abs[1]]
        val_features   = all_features[val_abs[0]:val_abs[1]]

        train_data = prepare_training_data(train_df, train_features)
        val_data = prepare_training_data(val_df, val_features)

        if not train_data or not val_data: continue

        # ── Secondary symbols: 동일 fold 비율로 슬라이스 ─────────────────────
        # Primary의 fold 위치를 0~1 비율로 환산해 secondary에 적용 (길이 다를 수 있음)
        _tr_frac_s = train_abs[0] / max(1, len(all_features))
        _tr_frac_e = train_abs[1] / max(1, len(all_features))
        _va_frac_s = val_abs[0]   / max(1, len(all_features))
        _va_frac_e = val_abs[1]   / max(1, len(all_features))
        _extra_train_datas: list = []
        _extra_val_datas:   list = []
        for _s2, _df2, _af2 in symbol_datasets[1:]:
            _n2 = len(_df2)
            _ts2 = int(_tr_frac_s * _n2)
            _te2 = int(_tr_frac_e * _n2)
            _vs2 = int(_va_frac_s * _n2)
            _ve2 = int(_va_frac_e * _n2)
            if _te2 <= _ts2 or _ve2 <= _vs2: continue
            _td2 = prepare_training_data(_df2.iloc[_ts2:_te2].reset_index(drop=True), _af2[_ts2:_te2])
            _vd2 = prepare_training_data(_df2.iloc[_vs2:_ve2].reset_index(drop=True), _af2[_vs2:_ve2])
            if _td2 and _vd2:
                _extra_train_datas.append((_s2, _td2))
                _extra_val_datas.append((_s2, _vd2))

        # ── Fisher LDA: fold별 지도 판별 방향 학습 ──────────────────────────
        # PCA(분산 최대화) 대신 LONG/SHORT/HOLD 클래스 분리를 직접 최대화.
        # train_features: [N_train, F],  train_df['label']: 0/1/-1
        _decomposer = agent.encoder.decomposer
        if _decomposer.use_lda:
            _train_labels = train_df["label"].values  # [N_train] numpy int
            _decomposer.fit_lda(train_features, _train_labels)

        # Optimizer + Scheduler reset at each fold:
        # Prevents LR from permanently decaying as training data grows fold-to-fold.
        # Model weights are from prior fold's BEST epoch (loaded at fold-end below).
        # This lets the model actively re-learn from each new fold's data.
        from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
        for pg in agent.optimizer.param_groups:
            pg["lr"] = agent.config.lr
        # Clear Adam momentum state — QNG: target classical_optimizer only
        _inner_opt = (
            agent.optimizer.classical_optimizer
            if isinstance(agent.optimizer, _DQNG)
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
        best_fold_sniper = -float('inf')
        best_epoch_pnl_series = []

        # AdaptiveController: reset per fold so Fold N penalties don't bleed into Fold N+1.
        auto_ctrl = AdaptiveController(agent)

        # Early stopping: halt when EV doesn't improve for patience epochs.
        # Saves compute in later folds where EV plateaus by epoch 2-3.
        patience_epochs = 5
        no_improve_count = 0

        fold_epoch_times: list = []
        _epoch_log: list = []   # per-epoch metrics for visualization

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            agent.train()
            n_samples = train_data["n"]
            seq_len = args.seq_len

            # Balanced class sampling: LONG:SHORT = 1:1 to prevent base-rate exploitation.
            # Without balancing, 58% SHORT labels cause action collapse toward SHORT.
            # h=96(15m=24시간)과 win_prices 창을 일치시킴 (ChatGPT 제안 1).
            # 기존 30 bars는 라벨 기준(96 bars)의 31% 만 커버 → 보상 불일치.
            H = 96 if args.timeframe == "15m" else 60
            valid_min  = seq_len + 120
            valid_max  = n_samples - (H + 1)
            all_valid  = np.arange(valid_min, valid_max)
            raw_lbl    = train_data["raw_labels"][all_valid]
            long_pool  = all_valid[raw_lbl == 1]
            short_pool = all_valid[raw_lbl == -1]
            hold_pool  = all_valid[raw_lbl == 0]
            n_per_dir  = min(len(long_pool), len(short_pool))
            sampled_long  = np.random.choice(long_pool,  n_per_dir, replace=False)
            sampled_short = np.random.choice(short_pool, n_per_dir, replace=False)
            n_hold = min(len(hold_pool), max(1, n_per_dir // 4))
            sampled_hold  = (np.random.choice(hold_pool, n_hold,
                                              replace=n_hold > len(hold_pool))
                             if len(hold_pool) > 0 else np.array([], dtype=int))
            indices = np.random.permutation(
                np.concatenate([sampled_long, sampled_short, sampled_hold])
            )
            epoch_loss = 0
            epoch_critic_loss = 0.0  # P4: per-component GN diagnosis
            epoch_fp_loss     = 0.0
            epoch_dir_sym     = 0.0
            n_batches = 0

            # TP/SL 시뮬레이션 파라미터 (레이블 생성 alpha/beta와 일치)
            _tp_mult = args.tp_mult  # default=4.0
            _sl_mult = args.sl_mult  # default=1.5

            for start_idx in range(0, len(indices), args.batch_size):
                batch_indices = indices[start_idx:start_idx + args.batch_size]

                # ── Step 1: 피처·가격·ATR 수집 (방향/레이블 미결정) ─────────────
                x_list, prices_list, atr_list, entry_list = [], [], [], []
                highs_list, lows_list = [], []

                for idx in batch_indices:
                    x_list.append(train_data["features"][idx - seq_len + 1 : idx + 1])
                    win_prices = train_data["prices"][idx : idx + H + 1]
                    prices_list.append(win_prices)
                    highs_list.append(train_data["highs"][idx : idx + H + 1])
                    lows_list.append(train_data["lows"][idx : idx + H + 1])
                    atr_list.append(train_data["atr"][idx])
                    entry_list.append(0)

                x_train = torch.from_numpy(np.array(x_list)).float()

                # ── Step 2: No-grad forward → 모델 예측 행동 취득 ───────────────
                # 모델이 현재 상태에서 실제로 선택할 행동(HOLD/LONG/SHORT)을 얻는다.
                # 이 행동을 기반으로 미래 가격으로 실제 P&L을 시뮬레이션 → reward hacking 제거.
                with torch.no_grad():
                    _logits, _, _, _ = agent(x_train.to(agent.device), last_step_only=True)
                    # squeeze: [B,1,3] → [B,3] (last_step_only 모드에서 dim=3 가능)
                    if _logits.dim() == 3:
                        _logits = _logits.squeeze(1)
                    pred_actions = _logits.argmax(dim=-1).cpu().numpy()  # [B] int64

                # ── Step 3: TP/SL 시뮬레이션 → 실제 레이블 결정 ─────────────────
                # 모델 행동 + 실제 미래 고가/저가 → TP hit / SL hit / timeout 판정.
                # label=1: LONG TP hit → TP_HIT (+r_tp)
                # label=2: SHORT TP hit → TP_HIT (+r_tp)
                # label=3: SL hit (any dir) → SL_HIT (-r_sl, 음수 보상)
                # label=0: timeout / HOLD → OBSERVE (0.0)
                dirs_list, labels_list = [], []

                for i, idx in enumerate(batch_indices):
                    act = int(pred_actions[i])
                    if act == 0:  # HOLD 예측 → 거래 없음
                        dirs_list.append(0)
                        labels_list.append(0)
                        continue

                    direction = 1 if act == 1 else -1
                    ep      = float(prices_list[i][0])          # 진입가
                    atr_val = float(atr_list[i])
                    tp_dist = _tp_mult * atr_val               # TP = 4×ATR
                    sl_dist = _sl_mult * atr_val               # SL = 1.5×ATR

                    # numpy vectorised scan (Python 루프보다 ~10× 빠름)
                    _hi = highs_list[i][1:]                    # entry bar 제외
                    _lo = lows_list[i][1:]
                    if direction == 1:                         # LONG
                        tp_bars = np.where(_hi >= ep + tp_dist)[0]
                        sl_bars = np.where(_lo <= ep - sl_dist)[0]
                    else:                                      # SHORT
                        tp_bars = np.where(_lo <= ep - tp_dist)[0]
                        sl_bars = np.where(_hi >= ep + sl_dist)[0]

                    tp_bar = int(tp_bars[0]) if len(tp_bars) > 0 else H + 1
                    sl_bar = int(sl_bars[0]) if len(sl_bars) > 0 else H + 1

                    dirs_list.append(direction)
                    if tp_bar <= sl_bar and tp_bar <= H:
                        labels_list.append(1 if direction == 1 else 2)  # TP_HIT
                    elif sl_bar < tp_bar and sl_bar <= H:
                        labels_list.append(3)                            # SL_HIT
                    else:
                        labels_list.append(0)                            # timeout

                # x_train already built above
                max_p_len = max(len(p) for p in prices_list)
                prices_padded = np.zeros((len(prices_list), max_p_len))
                for i, p in enumerate(prices_list): prices_padded[i, :len(p)] = p

                p_train = torch.from_numpy(prices_padded).float()
                d_train = torch.from_numpy(np.array(dirs_list)).float()
                e_train = torch.from_numpy(np.array(entry_list)).long()
                l_train = torch.from_numpy(np.array(labels_list)).long()
                a_train = torch.from_numpy(np.array(atr_list)).float()

                result = agent.train_step(x_train, p_train, d_train, e_train, l_train, a_train, last_step_only=True)
                epoch_loss        += result.loss
                epoch_critic_loss += result.critic_loss
                epoch_fp_loss     += result.fp_loss
                epoch_dir_sym     += result.dir_sym_loss
                n_batches += 1

            # ── Secondary symbol 학습 (같은 에폭, 시간 순서 독립적 배치) ────────
            for _s2, _td2 in _extra_train_datas:
                _n2     = _td2["n"]
                _vmin2  = seq_len + 120
                _vmax2  = _n2 - (H + 1)
                if _vmax2 <= _vmin2: continue
                _av2    = np.arange(_vmin2, _vmax2)
                _rl2    = _td2["raw_labels"][_av2]
                _lp2    = _av2[_rl2 == 1];  _sp2 = _av2[_rl2 == -1]
                _npd2   = min(len(_lp2), len(_sp2))
                if _npd2 == 0: continue
                _sl2    = np.random.choice(_lp2, _npd2, replace=False)
                _ss2    = np.random.choice(_sp2, _npd2, replace=False)
                _idx2   = np.random.permutation(np.concatenate([_sl2, _ss2]))
                for _si2 in range(0, len(_idx2), args.batch_size):
                    _bi2 = _idx2[_si2:_si2 + args.batch_size]
                    _xl2, _pl2, _al2, _el2 = [], [], [], []
                    _hl2, _ll2 = [], []
                    for _ii in _bi2:
                        _xl2.append(_td2["features"][_ii - seq_len + 1 : _ii + 1])
                        _pl2.append(_td2["prices"][_ii : _ii + H + 1])
                        _hl2.append(_td2["highs"][_ii : _ii + H + 1])
                        _ll2.append(_td2["lows"][_ii : _ii + H + 1])
                        _al2.append(_td2["atr"][_ii]);  _el2.append(0)
                    _xt2 = torch.from_numpy(np.array(_xl2)).float()
                    with torch.no_grad():
                        _lg2, _, _, _ = agent(_xt2.to(agent.device), last_step_only=True)
                        if _lg2.dim() == 3: _lg2 = _lg2.squeeze(1)
                        _pa2 = _lg2.argmax(dim=-1).cpu().numpy()
                    _dl2, _lbl2 = [], []
                    for _ji, _ii in enumerate(_bi2):
                        _act = int(_pa2[_ji])
                        if _act == 0: _dl2.append(0); _lbl2.append(0); continue
                        _ep = float(_td2["prices"][_ii])
                        _atr = float(_td2["atr"][_ii])
                        _hh = _td2["highs"][_ii : _ii + H + 1]
                        _lh = _td2["lows"][_ii : _ii + H + 1]
                        _tp_p = _ep * (1 + (_tp_mult * _atr / _ep) * (1 if _act == 1 else -1))
                        _sl_p = _ep * (1 - (_sl_mult * _atr / _ep) * (1 if _act == 1 else -1))
                        _res = 0
                        for _bi3 in range(len(_hh)):
                            if _act == 1:
                                if _hh[_bi3] >= _tp_p: _res = 1; break
                                if _lh[_bi3] <= _sl_p: _res = 3; break
                            else:
                                if _lh[_bi3] <= _tp_p: _res = 2; break
                                if _hh[_bi3] >= _sl_p: _res = 3; break
                        _dl2.append(1 if _act == 1 else -1)
                        _lbl2.append(_res)
                    _mpl2 = max(len(_p) for _p in _pl2)
                    _pp2  = np.zeros((len(_pl2), _mpl2))
                    for _ji, _p in enumerate(_pl2): _pp2[_ji, :len(_p)] = _p
                    _r2 = agent.train_step(
                        _xt2,
                        torch.from_numpy(_pp2).float(),
                        torch.from_numpy(np.array(_dl2)).float(),
                        torch.from_numpy(np.array(_el2)).long(),
                        torch.from_numpy(np.array(_lbl2)).long(),
                        torch.from_numpy(np.array(_al2)).float(),
                        last_step_only=True,
                    )
                    epoch_loss        += _r2.loss
                    epoch_critic_loss += _r2.critic_loss
                    epoch_fp_loss     += _r2.fp_loss
                    epoch_dir_sym     += _r2.dir_sym_loss
                    n_batches += 1

            avg_loss        = epoch_loss        / max(1, n_batches)
            avg_critic_loss = epoch_critic_loss / max(1, n_batches)
            avg_fp_loss     = epoch_fp_loss     / max(1, n_batches)
            avg_dir_sym     = epoch_dir_sym     / max(1, n_batches)

            # 그래디언트 노름 — result.grad_norm 은 clip 이전의 원시 노름 (pre-clip)
            # NOTE: .grad 를 직접 읽으면 clip_grad_norm_ 이 적용된 후라 항상 ≤clip 으로 보임.
            # P4: grad_clip=5.0 으로 상향 — 복합 loss(actor+critic+FP) 에서 합법적인
            # 그래디언트가 1.0을 초과해도 clip되지 않도록. 진짜 폭발(>5.0)만 차단.
            grad_norm = result.grad_norm  # pre-clip norm returned by train_step()

            # label distribution for this epoch (last batch only — approximate)
            # label: 0=HOLD/timeout, 1=LONG-TP, 2=SHORT-TP, 3=SL-hit
            lc = np.bincount(np.array(labels_list), minlength=4)

            epoch_elapsed = time.time() - epoch_start
            fold_epoch_times.append(epoch_elapsed)

            # ETA: rolling average of last 3 epochs x remaining epochs
            avg_t = sum(fold_epoch_times[-3:]) / min(len(fold_epoch_times), 3)
            remaining = args.epochs - epoch
            eta_sec = avg_t * remaining
            if eta_sec >= 3600:
                eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60)}m"
            elif eta_sec >= 60:
                eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
            elif remaining == 0:
                eta_str = "done"
            else:
                eta_str = f"{int(eta_sec)}s"

            val_ev, val_winrate, diag, pnl_series_epoch = evaluate_model(agent, val_data, seq_len=args.seq_len)
            # Secondary symbols val 평균 합산 (가중치: 균등)
            if _extra_val_datas:
                _ev_sum = val_ev; _wr_sum = val_winrate; _cnt = 1
                for _, _vd2 in _extra_val_datas:
                    _ev2, _wr2, _, _ = evaluate_model(agent, _vd2, seq_len=args.seq_len)
                    _ev_sum += _ev2; _wr_sum += _wr2; _cnt += 1
                val_ev     = _ev_sum / _cnt
                val_winrate = _wr_sum / _cnt
            # Line 1: primary metrics for checkpointing
            print(
                f"  Epoch {epoch:2d}/{args.epochs} | Loss: {avg_loss:.4f} | GN: {grad_norm:.4f} | "
                f"ETA: {eta_str} | EV/trade: {val_ev:+.4f} | WR: {val_winrate:.1%} | "
                f"LP={diag['long_prec']:.0f}%(n={diag['long_n']})  "
                f"SP={diag['short_prec']:.0f}%(n={diag['short_n']})"
            )
            # Line 2: diagnostic breakdown
            # lc: simulated label distribution (last batch) — 0=timeout,1=LONG-TP,2=SHORT-TP,3=SL
            _lc_sum = max(1, lc.sum())
            print(
                f"           Pre[L={diag['before_long']} S={diag['before_short']} H={diag['before_hold']}]"
                f"  Post[L={diag['long']} S={diag['short']} Hf={diag['hold_filtered']}]"
                f"  mP[H:{diag['mean_p_hold']:.2f} L:{diag['mean_p_long']:.2f} S:{diag['mean_p_short']:.2f}]"
                f"  RawLbl tp={diag['raw_tp']} sl={diag['raw_sl']} n={diag['raw_neutral']}"
                f"  SimLbl[TP={lc[1]+lc[2]} SL={lc[3]} TO={lc[0]}]({_lc_sum})"
            )
            # Line 3: P4 gradient diagnosis — per-component loss breakdown
            # GN_WARNING fires when pre-clip GN > 5.0 (grad_clip 기준)
            gn_tag = " ⚠GN_HIGH" if grad_norm > 5.0 else ""
            print(
                f"           [LossBreak] Actor={avg_loss - avg_critic_loss * agent.config.critic_coef - avg_fp_loss * agent.config.fp_coef:.4f}"
                f"  Critic={avg_critic_loss:.4f}  FP={avg_fp_loss:.4f}  DirSym={avg_dir_sym:.4f}{gn_tag}"
            )
            # Line 4: QNG / QFI diagnostics (only when use_qng=True)
            from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
            if isinstance(agent.optimizer, _DQNG):
                _qfi = agent.optimizer.get_qfi_stats()
                _bp_tag = " ⚠BARREN" if grad_norm < 0.5 else ""
                print(
                    f"           [QNG] QFI mean={_qfi['qfi_mean']:.4f}"
                    f"  min={_qfi['qfi_min']:.4f}  max={_qfi['qfi_max']:.4f}"
                    f"  lr_q={agent.optimizer.lr_quantum:.4f}"
                    f"  steps={agent.optimizer._step_count}{_bp_tag}"
                )
            val_sniper = val_ev  # alias for checkpoint comparisons below
            auto_ctrl.step_epoch(diag)     # entropy_reg / lr 자동 조정

            # ── per-epoch log (viz용) ─────────────────────────────────────
            from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG2
            _qfi_mean_log = 0.0
            if isinstance(agent.optimizer, _DQNG2):
                _qfi_mean_log = agent.optimizer.get_qfi_stats().get("qfi_mean", 0.0)
            _actor_loss_log = (avg_loss
                               - avg_critic_loss * agent.config.critic_coef
                               - avg_fp_loss     * agent.config.fp_coef)
            _epoch_log.append({
                "epoch":       epoch,
                "loss":        avg_loss,
                "actor_loss":  _actor_loss_log,
                "critic_loss": avg_critic_loss,
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
            MIN_TRADES_FOR_CKPT = 15
            n_trades_this_epoch = diag["long_n"] + diag["short_n"]
            eligible = n_trades_this_epoch >= MIN_TRADES_FOR_CKPT

            # Fix #3: compare within THIS fold only (not across folds)
            if val_sniper > best_fold_sniper and eligible:
                best_fold_sniper = val_sniper
                no_improve_count = 0
                best_epoch_pnl_series = pnl_series_epoch
                ckpt_path = os.path.join(args.checkpoint_dir, f"agent_best_fold{k}.pt")
                agent.save_checkpoint(ckpt_path)
                print(f"  ⭐ New Best [Fold {k}] Sniper={val_sniper:.4f}  Trades={n_trades_this_epoch}  Saved: {ckpt_path}")
                # ── Visualization (fold best 업데이트 시) ─────────────────
                _best_metrics = {
                    "ev":           val_ev,
                    "wr":           val_winrate,
                    "n_trades":     n_trades_this_epoch,
                    "loss":         avg_loss,
                    "actor_loss":   _actor_loss_log,
                    "critic_loss":  avg_critic_loss,
                    "fp_loss":      avg_fp_loss,
                    "dir_sym":      avg_dir_sym,
                    "qfi_mean":     _qfi_mean_log,
                    "mean_p_long":  diag.get("mean_p_long",  1/3),
                    "mean_p_short": diag.get("mean_p_short", 1/3),
                    "avg_pnl":      0.0,   # backtest 없이는 0
                }
                _fold_best_history.append({"fold": k, **_best_metrics})
                try:
                    viz_path = viz.plot_rl_fold_best(
                        fold=k, epoch=epoch,
                        metrics=_best_metrics,
                        fold_history=_fold_best_history,
                        epoch_log=_epoch_log,
                    )
                    print(f"  [viz] → {viz_path}")
                except Exception as _ve:
                    print(f"  [viz] skipped ({_ve})")
                # Drive 동기화 — Fold Best
                _sync_to_drive(ckpt_path, args.drive_dir, label=f"Fold{k} Best")
                # Also update global best → agent_best.pt for final deployment
                if val_sniper > best_sniper_score:
                    best_sniper_score = val_sniper
                    global_best_path = os.path.join(args.checkpoint_dir, "agent_best.pt")
                    agent.save_checkpoint(global_best_path)
                    # Drive 동기화 — Global Best (agent_best.pt)
                    _sync_to_drive(global_best_path, args.drive_dir, label=f"Fold{k} GlobalBest")
            else:
                if not eligible:
                    print(f"  [MinTrades] Ep{epoch} trades={n_trades_this_epoch} < {MIN_TRADES_FOR_CKPT} → skip ckpt")
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience_epochs:
                        print(f"  [EarlyStop] No improvement for {patience_epochs} epochs → stopping Fold {k} at Ep{epoch}")
                        break

        if best_epoch_pnl_series:
            all_folds_pnl_series.extend(best_epoch_pnl_series)

        # Fix #5: Roll back to best-epoch weights before next fold.
        # Without this, fold k+1 starts from epoch-20 weights (often worse
        # than the best epoch). Loading best fold k weights + resetting the
        # optimizer at the top of fold k+1 gives fold k+1 the cleanest start.
        fold_best_path = os.path.join(args.checkpoint_dir, f"agent_best_fold{k}.pt")
        if os.path.exists(fold_best_path):
            ckpt = torch.load(fold_best_path, map_location=agent.device, weights_only=False)
            agent.load_state_dict(ckpt["model_state"], strict=True)
            print(f"  [fold {k}] Rolled back to best-epoch weights "
                  f"(EV={best_fold_sniper:+.4f}) → Fold {k+1} starts from this state.")

            # Phase 2: production-gate diagnostic on the rolled-back best model
            # Runs select_action() (all gates) vs raw forward() on 50 val bars.
            # Diagnostic only — never affects checkpointing or training.
            try:
                gate_stats = prod_eval_quick(agent, val_data, seq_len=args.seq_len, n_sample=50)
                auto_ctrl.step_fold(gate_stats)   # lindblad_threshold 자동 조정
            except Exception as _pe:
                print(f"  [ProdEval] skipped ({_pe})")

    print(f"\n✅ Training complete. Global best Sniper={best_sniper_score:.4f} "
          f"→ checkpoints/quantum_v2/agent_best.pt")

    if all_folds_pnl_series:
        pnl_df = pd.DataFrame(all_folds_pnl_series)
        pnl_df['ts'] = pd.to_datetime(pnl_df['ts'])
        pnl_df = pnl_df.sort_values(by="ts").reset_index(drop=True)
        pnl_df['cumulative_pnl_pct'] = pnl_df['pnl_pct'].cumsum()
        
        # Ensure reports directory exists
        os.makedirs("reports", exist_ok=True)
        
        save_path = "reports/walk_forward_results.csv"
        pnl_df.to_csv(save_path, index=False)
        print(f"\n[Walk-Forward] 누적 손익 결과 저장 완료: {save_path}")


if __name__ == "__main__":
    main()