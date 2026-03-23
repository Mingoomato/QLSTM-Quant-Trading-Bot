"""
plot_features.py
────────────────────────────────────────────────────────────────────────────
V4 피처 시각화 스크립트 — 각 차원을 개별 subplot으로 표시

사용법:
    python scripts/plot_features.py                          # 기본: 최근 30일
    python scripts/plot_features.py --days 60               # 60일치
    python scripts/plot_features.py --start 2024-01-01 --end 2024-06-01
    python scripts/plot_features.py --symbol ETHUSDT --timeframe 4h

출력:
    reports/features_v4_<symbol>_<timeframe>.png  (저장)
    화면에 바로 표시
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 프로젝트 루트를 sys.path에 추가 ─────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.bybit_mainnet import BybitMainnetClient, INTERVAL_MAP
from src.models.features_v4 import generate_and_cache_features_v4

# ── V4 피처 이름 (28개) ───────────────────────────────────────────────────────
V4_NAMES = [
    "lr0 (log-return)",        # 00
    "lr1 (1-lag return)",      # 01
    "hl (high-low ratio)",     # 02
    "vol_ratio",               # 03
    "ema12_dev",               # 04
    "macd_val",                # 05
    "rsi",                     # 06
    "atr",                     # 07
    "mom3",                    # 08
    "mom10",                   # 09
    "mom20",                   # 10
    "stoch_k",                 # 11
    "sign_change_freq",        # 12
    "trend_str",               # 13
    "hr_sin (hour sin)",       # 14
    "hr_cos (hour cos)",       # 15
    "obi (order book imb.)",   # 16
    "hurst_H",                 # 17
    "autocorr_lag1",           # 18
    "purity_proxy",            # 19
    "funding_rate_z",          # 20
    "candle_body_ratio",       # 21
    "volume_z",                # 22
    "oi_change_pct",           # 23
    "funding_velocity",        # 24
    "cvd_delta_z",             # 25
    "cvd_trend_z",             # 26
    "cvd_price_div",           # 27
]

# 그룹별 색상
GROUP_COLORS = {
    "Price":     "#4fc3f7",
    "Momentum":  "#ef5350",
    "Volatility":"#ff9800",
    "Trend":     "#66bb6a",
    "Volume":    "#ab47bc",
    "Time":      "#26c6da",
    "OrderFlow": "#ec407a",
    "Exogenous": "#ffd54f",
}

FEAT_GROUPS = [
    "Price", "Price", "Price", "Volatility",      # 00-03
    "Trend", "Trend", "Momentum", "Volatility",   # 04-07
    "Momentum", "Momentum", "Momentum", "Momentum",# 08-11
    "Time", "Trend", "Time", "Time",               # 12-15
    "OrderFlow", "Price", "Price", "Price",        # 16-19
    "Exogenous", "Price", "Volume",                # 20-22
    "Exogenous", "Exogenous",                      # 23-24
    "OrderFlow", "OrderFlow", "OrderFlow",         # 25-27
]


def _ms(dt_str: str) -> int:
    """'YYYY-MM-DD' 문자열을 UTC epoch ms로 변환."""
    dt = datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_data(symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Bybit API에서 OHLCV + 펀딩레이트 + OI를 받아 하나의 DataFrame으로 합친다."""
    client = BybitMainnetClient()

    # ── 1. OHLCV ─────────────────────────────────────────────────────────────
    interval_sec = {
        "1m": 60, "5m": 300, "15m": 900,
        "1h": 3600, "4h": 14400, "1d": 86400,
    }.get(timeframe, 3600)
    n_bars = int((end_ms - start_ms) / (interval_sec * 1000)) + 200

    print(f"[fetch] OHLCV {symbol} {timeframe}  ~{n_bars} bars ...", flush=True)
    df = client.fetch_ohlcv(symbol, timeframe, days_back=1, end_ms=end_ms, limit=n_bars)
    if df.empty:
        raise RuntimeError("OHLCV fetch returned empty DataFrame")

    # ts 열 → datetime
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    # 기간 필터
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_dt   = pd.Timestamp(end_ms,   unit="ms", tz="UTC")
    df = df[(df["ts"] >= start_dt) & (df["ts"] <= end_dt)].reset_index(drop=True)
    print(f"[fetch] OHLCV rows after filter: {len(df)}", flush=True)

    # ── 2. 펀딩레이트 ─────────────────────────────────────────────────────────
    print("[fetch] Funding rate ...", flush=True)
    try:
        df_fr = client.fetch_funding_history(symbol, start_ms, end_ms)
        if not df_fr.empty:
            df_fr["ts"] = pd.to_datetime(df_fr["ts_ms"], unit="ms", utc=True)
            df_fr = df_fr.sort_values("ts")[["ts", "funding_rate"]]
            df = pd.merge_asof(df, df_fr, on="ts", direction="backward")
            print(f"[fetch] Funding rows merged: {df['funding_rate'].notna().sum()}", flush=True)
        else:
            df["funding_rate"] = 0.0
    except Exception as e:
        print(f"[fetch] Funding fetch failed ({e}), using zeros", flush=True)
        df["funding_rate"] = 0.0

    # ── 3. OI ─────────────────────────────────────────────────────────────────
    print("[fetch] Open Interest ...", flush=True)
    try:
        df_oi = client.fetch_open_interest_history(symbol, start_ms, end_ms, interval="15min")
        if not df_oi.empty:
            df_oi["ts"] = pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
            df_oi = df_oi.sort_values("ts")[["ts", "open_interest"]]
            df = pd.merge_asof(df, df_oi, on="ts", direction="backward")
            print(f"[fetch] OI rows merged: {df['open_interest'].notna().sum()}", flush=True)
        else:
            df["open_interest"] = 0.0
    except Exception as e:
        print(f"[fetch] OI fetch failed ({e}), using zeros", flush=True)
        df["open_interest"] = 0.0

    return df


def build_features(df: pd.DataFrame, cache_path: str) -> np.ndarray:
    """V4 피처 행렬 [N, 28]을 계산한다."""
    arr = generate_and_cache_features_v4(df, cache_path, warmup=120, lookback=30, verbose=True)
    return arr  # [N, 28]


def plot_all_features(timestamps, feat_arr: np.ndarray, symbol: str, timeframe: str, out_path: str):
    """28개 피처를 7×4 그리드에 각각 개별 subplot으로 그린다."""
    n_feats = feat_arr.shape[1]  # 28
    n_cols = 4
    n_rows = (n_feats + n_cols - 1) // n_cols  # = 7

    fig = plt.figure(figsize=(24, n_rows * 3.2), facecolor="#0d1117")
    fig.suptitle(
        f"V4 Feature Time Series — {symbol} {timeframe}\n"
        f"({timestamps[0].strftime('%Y-%m-%d')} → {timestamps[-1].strftime('%Y-%m-%d')}, "
        f"{len(timestamps)} bars)",
        color="white", fontsize=15, fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.35)

    ts = np.array([t.timestamp() for t in timestamps])  # float timestamps for x-axis
    x_labels = [t.strftime("%m-%d") for t in timestamps]

    # x축 레이블용: 대략 10개 틱
    n = len(timestamps)
    tick_step = max(1, n // 10)
    tick_idx = list(range(0, n, tick_step))

    for idx in range(n_feats):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])

        name  = V4_NAMES[idx]
        group = FEAT_GROUPS[idx]
        color = GROUP_COLORS.get(group, "#aaaaaa")

        y = feat_arr[:, idx]

        # 유효값만 (warmup 구간 0 제외)
        valid_mask = (y != 0.0) | (idx in (14, 15))  # hr_sin/cos는 0이 유효값
        y_valid = y.copy()
        y_valid[~valid_mask] = np.nan

        # 배경
        ax.set_facecolor("#161b22")
        ax.spines[:].set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=7)

        # 0 기준선
        ax.axhline(0, color="#484f58", linewidth=0.6, linestyle="--", zorder=1)

        # 데이터 라인
        ax.plot(range(n), y_valid, color=color, linewidth=0.9, alpha=0.92, zorder=2)

        # 통계 밴드 (평균 ± 1σ)
        valid_vals = y_valid[~np.isnan(y_valid)]
        if len(valid_vals) > 5:
            mu  = np.mean(valid_vals)
            sig = np.std(valid_vals)
            ax.axhline(mu,       color=color, linewidth=0.5, linestyle="-",  alpha=0.4)
            ax.axhline(mu + sig, color=color, linewidth=0.4, linestyle=":",  alpha=0.3)
            ax.axhline(mu - sig, color=color, linewidth=0.4, linestyle=":",  alpha=0.3)

        # 타이틀
        ax.set_title(f"[{idx:02d}] {name}", color=color, fontsize=8.5, pad=3, fontweight="bold")

        # x축 틱 (날짜)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([x_labels[i] for i in tick_idx], rotation=30, ha="right", fontsize=6)
        ax.set_xlim(0, n - 1)

        # y축 범위
        if len(valid_vals) > 5:
            ymin, ymax = np.nanpercentile(y_valid, 1), np.nanpercentile(y_valid, 99)
            margin = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
            ax.set_ylim(ymin - margin, ymax + margin)

        # 우측 상단에 통계 표시
        if len(valid_vals) > 5:
            stats_txt = f"μ={np.mean(valid_vals):.3f}  σ={np.std(valid_vals):.3f}"
            ax.text(0.99, 0.97, stats_txt,
                    transform=ax.transAxes,
                    color="#8b949e", fontsize=6.5, ha="right", va="top")

    # 빈 subplot 숨기기
    total_cells = n_rows * n_cols
    for extra in range(n_feats, total_cells):
        row, col = divmod(extra, n_cols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[plot] Saved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Koopman EDMD — 순수 NumPy 구현 (PyTorch 의존 없음)
# ─────────────────────────────────────────────────────────────────────────────

def _build_dictionary(Xn: np.ndarray) -> np.ndarray:
    """
    Nonlinear dictionary for Kernel EDMD.

    Lifts D-dim state into 3D-dim space using STABLE nonlinear terms only:
        ψ(x) = [ x,          (D)   linear — original basis
                 x²  clipped, (D)   squared — captures regime variance
                 tanh(x) ]   (D)   bounded — saturates extremes, no explosion

    Design rule: every term must be bounded or at most O(x²) to prevent
    numerical explosion in G = Ψ^T Ψ.

    Removed: x/roll_std — caused 10^13 MSE when roll_std≈1e-8 during warmup.
    """
    x_clip = np.clip(Xn, -5.0, 5.0)          # hard clip: prevents outlier amplification
    x2     = np.clip(x_clip ** 2, 0.0, 25.0) # squared, bounded to [0, 25]
    x_tanh = np.tanh(x_clip)                  # bounded to (-1, 1)
    return np.concatenate([x_clip, x2, x_tanh], axis=1)  # [N, 3D]


def koopman_edmd(X: np.ndarray, n_modes: int = 5, reg: float = 1e-4,
                 nonlinear: bool = True):
    """
    Extended Dynamic Mode Decomposition (EDMD).

    Linear basis  (nonlinear=False):
        ψ(x) = x  →  K x_t ≈ x_{t+1}
        Problem: market is nonlinear → |λ| tops out at ~0.95

    Nonlinear / Kernel EDMD (nonlinear=True):
        ψ(x) = [x, x², tanh(x), x/σ]  →  K ψ(x_t) ≈ ψ(x_{t+1})
        Lifts state into 4D space where nonlinear dynamics become linear.
        Result: higher |λ| modes found because more structure is captured.

    Args:
        X          : [N, D] feature matrix (time-ordered)
        n_modes    : number of Koopman modes to keep
        reg        : regularisation on Gram matrix (numerical stability)
        nonlinear  : True = use nonlinear dictionary (recommended)

    Returns:
        modes       : [D, n_modes]  eigenvectors in *original* D-dim space
        eigenvalues : [n_modes]     complex eigenvalues (|λ| is key metric)
        X_new       : [N, n_modes]  data projected into Koopman basis
    """
    nonzero_mask = np.any(X != 0, axis=1)
    X_valid = X[nonzero_mask]
    N, D = X_valid.shape
    if N < 10:
        raise ValueError(f"Not enough valid rows: {N} (need >= 10)")

    X_now  = X_valid[:-1]
    X_next = X_valid[1:]
    T = len(X_now)

    mu  = X_now.mean(axis=0)
    sig = X_now.std(axis=0) + 1e-8
    Xn  = (X_now  - mu) / sig
    Xp  = (X_next - mu) / sig

    if nonlinear:
        # Lift both current and next state into the nonlinear dictionary
        Psi_now  = _build_dictionary(Xn)   # [T, 4D]
        Psi_next = _build_dictionary(Xp)   # [T, 4D]
        lifted_D = Psi_now.shape[1]
    else:
        Psi_now  = Xn
        Psi_next = Xp
        lifted_D = D

    # Koopman operator in lifted space: K = G^{-1} A
    G = (Psi_now.T  @ Psi_now)  / T + reg * np.eye(lifted_D)
    A = (Psi_now.T  @ Psi_next) / T
    K, _, _, _ = np.linalg.lstsq(G, A, rcond=1e-4)   # [4D, 4D]

    eigenvalues, eigenvectors = np.linalg.eig(K)

    # Sort by |λ - 1|: modes closest to unit circle first (most predictable)
    persistence = np.abs(np.abs(eigenvalues) - 1.0)
    sort_idx    = np.argsort(persistence)[:n_modes]

    top_eigenvalues = eigenvalues[sort_idx]
    # Project eigenvectors back to original D-dim for interpretability
    top_modes_lifted = np.real(eigenvectors[:, sort_idx])  # [4D, n_modes]
    # Take only the first D rows (linear component) for composition plot
    top_modes = top_modes_lifted[:D, :]                    # [D, n_modes]

    norms = np.linalg.norm(top_modes, axis=0, keepdims=True).clip(min=1e-8)
    top_modes /= norms

    # Project full data (N bars) into Koopman basis
    Xn_full = (X - mu) / sig
    if nonlinear:
        Psi_full = _build_dictionary(Xn_full)         # [N, 4D]
        X_new    = Psi_full @ top_modes_lifted         # [N, n_modes]
    else:
        X_new = Xn_full @ top_modes                    # [N, n_modes]

    return top_modes, top_eigenvalues, X_new


def plot_koopman_modes(
    timestamps,
    feat_arr: np.ndarray,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    symbol: str,
    timeframe: str,
    out_path: str,
):
    """
    Koopman 기저 변환 결과를 시각화한다.

    3개 패널:
      1) 원래 기저 (28-dim) vs Koopman 기저 (5-dim) 시계열 비교
      2) 각 Koopman 모드의 구성 (어떤 원래 피처가 기여하는가)
      3) 고유값 |λ_k| — 1에 가까울수록 예측 가능
    """
    n_modes = modes.shape[1]
    n_feats = modes.shape[0]

    # Koopman 투영 데이터
    nonzero_mask = np.any(feat_arr != 0, axis=1)
    mu  = feat_arr[nonzero_mask].mean(axis=0)
    sig = feat_arr[nonzero_mask].std(axis=0) + 1e-8
    X_projected = ((feat_arr - mu) / sig) @ modes   # [N, n_modes]

    n   = len(timestamps)
    x_labels = [t.strftime("%m-%d") for t in timestamps]
    tick_step = max(1, n // 10)
    tick_idx  = list(range(0, n, tick_step))

    mode_colors = ["#4fc3f7","#ef5350","#66bb6a","#ff9800","#ab47bc",
                   "#26c6da","#ffd54f","#ec407a"]

    fig = plt.figure(figsize=(24, 14), facecolor="#0d1117")
    fig.suptitle(
        f"Koopman Basis Change — {symbol} {timeframe}\n"
        f"Same data, new coordinates: Koopman eigenvector basis",
        color="white", fontsize=14, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(3, n_modes, figure=fig, hspace=0.55, wspace=0.35,
                           height_ratios=[2, 2, 1])

    # ── 행 1: Koopman 모드 시계열 ──────────────────────────────────────────
    for k in range(n_modes):
        ax = fig.add_subplot(gs[0, k])
        lam  = eigenvalues[k]
        lam_abs = abs(lam)
        color    = mode_colors[k % len(mode_colors)]

        y = X_projected[:, k]

        ax.set_facecolor("#161b22")
        ax.spines[:].set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.axhline(0, color="#484f58", lw=0.6, ls="--")

        ax.plot(range(n), y, color=color, lw=0.9, alpha=0.9)

        # |λ| 색상 코딩: 1에 가까울수록 초록, 멀수록 빨강
        lam_color = "#66bb6a" if abs(lam_abs - 1.0) < 0.05 else (
                    "#ffd54f" if abs(lam_abs - 1.0) < 0.15 else "#ef5350")

        ax.set_title(
            f"Mode {k+1}",
            color=color, fontsize=9, fontweight="bold", pad=2,
        )
        ax.text(0.5, 1.01,
                f"|λ|={lam_abs:.4f}",
                transform=ax.transAxes, ha="center", va="bottom",
                color=lam_color, fontsize=8, fontweight="bold")

        ax.set_xticks(tick_idx)
        ax.set_xticklabels([x_labels[i] for i in tick_idx], rotation=30, ha="right", fontsize=6)
        ax.set_xlim(0, n - 1)

        valid = y[np.any(feat_arr != 0, axis=1)]
        if len(valid) > 5:
            ymin = np.percentile(valid, 1); ymax = np.percentile(valid, 99)
            m = (ymax - ymin) * 0.1 or 0.1
            ax.set_ylim(ymin - m, ymax + m)
            ax.text(0.99, 0.97, f"σ={np.std(valid):.3f}",
                    transform=ax.transAxes, color="#8b949e", fontsize=7, ha="right", va="top")

    # ── 행 2: 모드 구성 — 어떤 원래 피처가 얼마나 기여하는가 ──────────────
    for k in range(n_modes):
        ax = fig.add_subplot(gs[1, k])
        color = mode_colors[k % len(mode_colors)]
        vec   = modes[:, k]   # [D] 이 모드의 고유벡터 (원래 기저에서의 좌표)

        # 절댓값 기준 상위 8개 피처만 표시
        top8_idx  = np.argsort(np.abs(vec))[::-1][:8]
        top8_vals = vec[top8_idx]
        top8_names= [V4_NAMES[i][:12] for i in top8_idx]

        bar_colors = [color if v >= 0 else "#ef5350" for v in top8_vals]
        bars = ax.barh(range(len(top8_idx)), top8_vals, color=bar_colors, alpha=0.8)

        ax.set_facecolor("#161b22")
        ax.spines[:].set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=6.5)
        ax.set_yticks(range(len(top8_idx)))
        ax.set_yticklabels(top8_names, fontsize=6.5, color="#c9d1d9")
        ax.axvline(0, color="#484f58", lw=0.8)
        ax.set_title(f"Mode {k+1} Composition\n(which original features contribute)",
                     color=color, fontsize=7.5, pad=3)
        ax.set_xlabel("Weight (eigenvector component)", color="#8b949e", fontsize=6.5)

    # ── 행 3: |λ_k| 막대 그래프 — 예측 가능성 순위 ─────────────────────────
    ax_eig = fig.add_subplot(gs[2, :])
    lam_abs_all = np.abs(eigenvalues)
    xs = np.arange(n_modes)
    bar_clrs = [
        "#66bb6a" if abs(v - 1.0) < 0.05 else
        "#ffd54f" if abs(v - 1.0) < 0.15 else "#ef5350"
        for v in lam_abs_all
    ]
    ax_eig.bar(xs, lam_abs_all, color=bar_clrs, alpha=0.85, width=0.5)
    ax_eig.axhline(1.0, color="#ffffff", lw=0.8, ls="--", alpha=0.4, label="|λ|=1.00 (perfectly preserved)")
    ax_eig.axhline(0.95, color="#ffd54f", lw=0.6, ls=":", alpha=0.5, label="|λ|=0.95 (predictability threshold)")
    ax_eig.set_facecolor("#161b22")
    ax_eig.spines[:].set_color("#30363d")
    ax_eig.tick_params(colors="#8b949e", labelsize=8)
    ax_eig.set_xticks(xs)
    ax_eig.set_xticklabels([f"Mode {k+1}" for k in range(n_modes)], color="#c9d1d9", fontsize=9)
    ax_eig.set_ylabel("|λ_k|  (closer to 1.0 = more predictable)", color="#8b949e", fontsize=8)
    ax_eig.set_title(
        "Koopman Eigenvalues |λ_k| — Quality of Basis Change\n"
        "Green: slow mode (predictable signal)  /  Yellow: borderline  /  Red: fast mode (noise)",
        color="white", fontsize=9,
    )
    ax_eig.legend(fontsize=7, labelcolor="#c9d1d9", facecolor="#161b22", edgecolor="#30363d")
    ax_eig.set_ylim(0, max(lam_abs_all) * 1.15)

    for i, v in enumerate(lam_abs_all):
        ax_eig.text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom",
                    color="#c9d1d9", fontsize=8, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[plot] Koopman saved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 1 — Ridge Cross-Validation: find optimal regularization alpha
# ─────────────────────────────────────────────────────────────────────────────

def ridge_cv_alpha(
    Psi_now:  np.ndarray,
    Psi_next: np.ndarray,
    alphas:   tuple = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0),
    n_folds:  int   = 5,
) -> float:
    """
    Find optimal ridge regularization strength via k-fold cross-validation.

    Problem without this:
        min ||Psi_next - K Psi_now||²   ← can overfit perfectly

    With ridge penalty:
        min ||Psi_next - K Psi_now||²  +  alpha * ||K||²_F
        - Large alpha: K is forced to be small → underfits
        - Small alpha: K can be large → overfits
        - Optimal alpha: lowest OOS prediction error

    CV procedure:
        Split data into n_folds chunks (time-ordered, no shuffle).
        For each fold: train on remaining, validate on held-out.
        Pick alpha with lowest average validation MSE.

    Args:
        Psi_now  : [N, lifted_D]  dictionary at time t
        Psi_next : [N, lifted_D]  dictionary at time t+1
        alphas   : candidates to search
        n_folds  : number of time-ordered folds

    Returns:
        best_alpha: float
    """
    N   = len(Psi_now)
    D   = Psi_now.shape[1]
    fold_size = N // n_folds
    best_alpha, best_mse = alphas[0], np.inf

    print(f"  [ridge-cv] Searching alpha in {alphas} with {n_folds} folds ...", flush=True)

    for alpha in alphas:
        mse_folds = []
        for k in range(n_folds):
            v0, v1 = k * fold_size, (k + 1) * fold_size
            # time-ordered: train = everything except this fold
            train = np.concatenate([Psi_now[:v0], Psi_now[v1:]], axis=0)
            ytrain = np.concatenate([Psi_next[:v0], Psi_next[v1:]], axis=0)
            val   = Psi_now[v0:v1]
            yval  = Psi_next[v0:v1]

            G = train.T @ train / len(train) + alpha * np.eye(D)
            A = train.T @ ytrain / len(train)
            K, _, _, _ = np.linalg.lstsq(G, A, rcond=1e-4)

            mse_folds.append(np.mean((val @ K - yval) ** 2))

        avg = float(np.mean(mse_folds))
        print(f"  [ridge-cv]   alpha={alpha:.0e}  OOS-MSE={avg:.6f}", flush=True)
        if avg < best_mse:
            best_mse   = avg
            best_alpha = alpha

    print(f"  [ridge-cv] Best alpha={best_alpha:.0e}  (MSE={best_mse:.6f})", flush=True)
    return best_alpha


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 2 — Sparse Dictionary: keep only informative nonlinear terms
# ─────────────────────────────────────────────────────────────────────────────

def sparse_dict_selection(
    Psi:       np.ndarray,
    Y:         np.ndarray,
    max_terms: int = 40,
) -> np.ndarray:
    """
    Select the most predictive dictionary terms using Spearman correlation.

    Problem without this:
        Full dictionary has 4D = 112 terms for D=28.
        Many nonlinear terms (x_i², tanh(x_i)) are correlated or uninformative.
        Keeping all 112 → overfitting in the K matrix.

    Solution:
        For each dictionary term ψ_j(x_t), compute its max |Spearman correlation|
        with any of the D target dimensions of x_{t+1}.
        Keep only the top `max_terms` by this score.

        Spearman (rank correlation) is used instead of Pearson because:
        - It is robust to outliers (BTC has fat tails)
        - It captures monotone nonlinear relationships

    Args:
        Psi      : [N, 4D]  full dictionary matrix
        Y        : [N, D]   next-state targets (z-scored original features)
        max_terms: maximum terms to keep

    Returns:
        selected_idx: [max_terms] sorted column indices
    """
    from scipy.stats import spearmanr

    n_terms = Psi.shape[1]
    D_target = min(Y.shape[1], 8)   # check against first 8 target dims (representative)
    scores   = np.zeros(n_terms)

    print(f"  [sparse]  Scoring {n_terms} dictionary terms ...", flush=True)
    for j in range(n_terms):
        corrs = []
        for d in range(D_target):
            r, _ = spearmanr(Psi[:, j], Y[:, d])
            corrs.append(abs(r) if not np.isnan(r) else 0.0)
        scores[j] = max(corrs)

    selected = np.argsort(scores)[::-1][:max_terms]
    selected = np.sort(selected)

    kept_linear    = int(np.sum(selected < Y.shape[1]))
    kept_nonlinear = len(selected) - kept_linear
    print(f"  [sparse]  Kept {len(selected)} / {n_terms} terms "
          f"(linear={kept_linear}, nonlinear={kept_nonlinear})", flush=True)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION 3 — Rolling Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────

def rolling_koopman_cv(
    X:             np.ndarray,
    n_modes:       int   = 5,
    train_bars:    int   = 720,   # 30 days × 24h
    test_bars:     int   = 240,   # 10 days × 24h
    step_bars:     int   = 240,
    reg:           float = 1e-4,
    max_terms:     int   = 40,
) -> list:
    """
    Rolling walk-forward Koopman cross-validation.

    Why this is the most important overfitting check:
        Even with ridge + sparse selection, we might still overfit to the
        specific time period in the data. Walk-forward CV tests whether
        the Koopman modes found on past data actually persist into the future.

    Procedure:
        Window 1:  train [0   : 720],  test [720 : 960]
        Window 2:  train [240 : 960],  test [960 : 1200]
        Window 3:  train [480 : 1200], test [1200: 1440]
        ...

    For each window, record:
        - |λ_k| of each mode (in-sample)
        - OOS prediction MSE (out-of-sample)
        - Mode stability: cosine similarity between consecutive windows' modes

    Stable |λ| + low OOS MSE = real signal
    High |λ| + high OOS MSE  = overfitting

    Returns:
        list of dicts, one per window:
            {start, end, eigenvalues, oos_mse, modes, oos_lambda}
    """
    nonzero = np.where(np.any(X != 0, axis=1))[0]
    if len(nonzero) < train_bars + test_bars:
        raise ValueError(f"Not enough valid data for rolling CV "
                         f"({len(nonzero)} bars, need {train_bars + test_bars})")

    X_valid = X[nonzero]
    N, D    = X_valid.shape
    results = []
    window  = 0

    start = 0
    while start + train_bars + test_bars <= N:
        tr_end = start + train_bars
        te_end = tr_end + test_bars

        X_train = X_valid[start:tr_end]
        X_test  = X_valid[tr_end:te_end]

        # ── Normalise on train set ────────────────────────────────────────
        mu  = X_train.mean(axis=0)
        sig = X_train.std(axis=0) + 1e-8
        Xtr = (X_train - mu) / sig
        Xte = (X_test  - mu) / sig   # apply train stats to test (no leakage)

        # ── Build dictionary on train (normalize per-column) ─────────────
        Psi_tr_now_raw  = _build_dictionary(Xtr[:-1])
        Psi_tr_next_raw = _build_dictionary(Xtr[1:])
        psi_mu  = Psi_tr_now_raw.mean(axis=0)
        psi_sig = Psi_tr_now_raw.std(axis=0).clip(min=1e-6)
        Psi_tr_now  = (Psi_tr_now_raw  - psi_mu) / psi_sig
        Psi_tr_next = (Psi_tr_next_raw - psi_mu) / psi_sig

        # ── Sparse selection (Solution 2) ──────────────────────────────────
        selected = sparse_dict_selection(Psi_tr_now, Xtr[1:], max_terms=max_terms)
        Psi_tr_now_s  = Psi_tr_now[:, selected]
        Psi_tr_next_s = Psi_tr_next[:, selected]

        # ── Ridge (Solution 1) — use provided reg (already CV-tuned) ──────
        lifted_D = Psi_tr_now_s.shape[1]
        G = Psi_tr_now_s.T @ Psi_tr_now_s / len(Psi_tr_now_s) + reg * np.eye(lifted_D)
        A = Psi_tr_now_s.T @ Psi_tr_next_s / len(Psi_tr_now_s)
        K, _, _, _ = np.linalg.lstsq(G, A, rcond=1e-4)

        eigenvalues, eigenvectors = np.linalg.eig(K)
        persistence = np.abs(np.abs(eigenvalues) - 1.0)
        top_idx     = np.argsort(persistence)[:n_modes]
        top_eigs    = eigenvalues[top_idx]
        top_vecs    = np.real(eigenvectors[:, top_idx])  # [lifted_D, n_modes]
        norms = np.linalg.norm(top_vecs, axis=0, keepdims=True).clip(min=1e-8)
        top_vecs /= norms

        # ── OOS evaluation (apply train dict stats — no leakage) ──────────
        Psi_te_now  = (_build_dictionary(Xte[:-1]) - psi_mu) / psi_sig
        Psi_te_next = (_build_dictionary(Xte[1:])  - psi_mu) / psi_sig
        Psi_te_now  = Psi_te_now[:, selected]
        Psi_te_next = Psi_te_next[:, selected]
        Y_pred      = Psi_te_now @ K
        oos_mse     = float(np.mean((Y_pred - Psi_te_next) ** 2))

        # OOS eigenvalues: fit K on test for comparison
        try:
            if len(Psi_te_now) > lifted_D:   # only fit if OOS window is large enough
                G2 = Psi_te_now.T @ Psi_te_now / len(Psi_te_now) + reg * np.eye(lifted_D)
                A2 = Psi_te_now.T @ Psi_te_next / len(Psi_te_now)
                K2, _, _, _ = np.linalg.lstsq(G2, A2, rcond=1e-4)
                eigs2, _ = np.linalg.eig(K2)
                top_eigs_oos = eigs2[np.argsort(np.abs(np.abs(eigs2) - 1.0))[:n_modes]]
            else:
                top_eigs_oos = top_eigs.copy()
        except Exception:
            top_eigs_oos = top_eigs * 0

        results.append({
            "window":      window,
            "start":       nonzero[start],
            "end":         nonzero[min(te_end - 1, N - 1)],
            "eigenvalues": top_eigs,        # in-sample |λ|
            "oos_lambda":  top_eigs_oos,    # OOS |λ|
            "oos_mse":     oos_mse,
            "modes":       top_vecs[:D, :], # first D rows = linear component
            "selected":    selected,
        })

        lam_str = " ".join(f"{abs(v):.3f}" for v in top_eigs)
        print(f"  [rolling] Window {window:02d}  "
              f"train=[{start}:{tr_end}]  test=[{tr_end}:{te_end}]  "
              f"|λ|=[{lam_str}]  OOS-MSE={oos_mse:.5f}", flush=True)

        start  += step_bars
        window += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline — combines all three solutions
# ─────────────────────────────────────────────────────────────────────────────

def koopman_full_pipeline(
    feat_arr:    np.ndarray,
    timestamps,
    n_modes:     int = 5,
    bars_per_day:int = 24,
    symbol:      str = "BTCUSDT",
    timeframe:   str = "1h",
):
    """
    Run all three anti-overfitting solutions in sequence:
        1. Ridge CV    → find optimal alpha
        2. Sparse EDMD → discard uninformative dictionary terms
        3. Rolling CV  → validate modes are real (not overfitting)

    Then plot:
        A) Final basis change (best modes on full data)
        B) Rolling validation dashboard
    """
    os.makedirs("reports", exist_ok=True)

    nonzero_mask = np.any(feat_arr != 0, axis=1)
    X_valid = feat_arr[nonzero_mask]
    N, D = X_valid.shape

    mu  = X_valid.mean(axis=0)
    sig = X_valid.std(axis=0) + 1e-8
    Xn  = (X_valid - mu) / sig

    # ── Step 1: Build full dictionary ────────────────────────────────────────
    print("\n[pipeline] Step 1: Building nonlinear dictionary ...", flush=True)
    Psi_now  = _build_dictionary(Xn[:-1])   # [N-1, 3D]
    Psi_next = _build_dictionary(Xn[1:])

    # Normalize each dictionary column to unit variance (prevents scale imbalance)
    psi_mu  = Psi_now.mean(axis=0)
    psi_sig = Psi_now.std(axis=0).clip(min=1e-6)
    Psi_now  = (Psi_now  - psi_mu) / psi_sig
    Psi_next = (Psi_next - psi_mu) / psi_sig   # apply train stats to next (no leakage)

    # ── Step 2: Sparse selection ─────────────────────────────────────────────
    print("[pipeline] Step 2: Sparse dictionary selection ...", flush=True)
    max_terms = min(40, Psi_now.shape[1])
    selected  = sparse_dict_selection(Psi_now, Xn[1:], max_terms=max_terms)
    Psi_s_now  = Psi_now[:, selected]
    Psi_s_next = Psi_next[:, selected]

    # ── Step 3: Ridge CV alpha search ────────────────────────────────────────
    print("[pipeline] Step 3: Ridge cross-validation alpha search ...", flush=True)
    best_alpha = ridge_cv_alpha(Psi_s_now, Psi_s_next)

    # ── Step 4: Final EDMD fit on full data with optimal settings ────────────
    print("[pipeline] Step 4: Final EDMD fit (full data) ...", flush=True)
    lifted_D = Psi_s_now.shape[1]
    G = Psi_s_now.T @ Psi_s_now / len(Psi_s_now) + best_alpha * np.eye(lifted_D)
    A = Psi_s_now.T @ Psi_s_next / len(Psi_s_now)
    K, _, _, _ = np.linalg.lstsq(G, A, rcond=1e-4)

    eigenvalues, eigenvectors = np.linalg.eig(K)
    persistence = np.abs(np.abs(eigenvalues) - 1.0)
    top_idx     = np.argsort(persistence)[:n_modes]
    top_eigs    = eigenvalues[top_idx]
    top_vecs    = np.real(eigenvectors[:, top_idx])
    norms = np.linalg.norm(top_vecs, axis=0, keepdims=True).clip(min=1e-8)
    top_vecs /= norms
    modes_D = top_vecs[:D, :]   # [D, n_modes] — original-space interpretation

    print(f"[pipeline] Final |λ_k|: {np.abs(top_eigs).round(4).tolist()}", flush=True)

    # Project full data into Koopman basis
    Psi_full = _build_dictionary((feat_arr - mu) / sig)[:, selected]
    X_proj   = Psi_full @ top_vecs   # [N_all, n_modes]

    # ── Plot A: basis change result ───────────────────────────────────────────
    out_a = os.path.join("reports", f"koopman_final_{symbol}_{timeframe}.png")
    plot_koopman_modes(timestamps, feat_arr, modes_D, top_eigs, symbol, timeframe, out_a)

    # ── Step 5: Rolling walk-forward validation ───────────────────────────────
    print("[pipeline] Step 5: Rolling walk-forward validation ...", flush=True)
    train_bars = min(720, N // 3)
    test_bars  = min(240, N // 6)
    step_bars  = test_bars
    rolling_results = rolling_koopman_cv(
        feat_arr, n_modes=n_modes,
        train_bars=train_bars, test_bars=test_bars, step_bars=step_bars,
        reg=best_alpha, max_terms=max_terms,
    )

    # ── Plot B: rolling validation dashboard ─────────────────────────────────
    out_b = os.path.join("reports", f"koopman_validation_{symbol}_{timeframe}.png")
    plot_rolling_validation(rolling_results, n_modes, symbol, timeframe, out_b)

    return {
        "modes": modes_D, "eigenvalues": top_eigs,
        "best_alpha": best_alpha, "selected": selected,
        "rolling": rolling_results,
    }


def plot_rolling_validation(results: list, n_modes: int,
                             symbol: str, timeframe: str, out_path: str):
    """
    Rolling walk-forward validation dashboard.

    3 rows:
      Row 1: In-sample |λ_k| per window  — should be stable if modes are real
      Row 2: OOS |λ_k| per window        — should match in-sample if no overfit
      Row 3: OOS MSE per window          — low = model generalises well

    Reading guide:
      In-sample ≈ OOS |λ|  AND  low OOS MSE  →  modes are REAL
      In-sample >> OOS |λ| OR   high OOS MSE →  OVERFITTING
    """
    if not results:
        print("[validation] No results to plot.")
        return

    n_windows  = len(results)
    win_labels = [f"W{r['window']}" for r in results]
    mode_colors = ["#4fc3f7","#ef5350","#66bb6a","#ff9800","#ab47bc"]

    fig, axes = plt.subplots(3, 1, figsize=(max(12, n_windows * 1.2), 12),
                              facecolor="#0d1117")
    fig.suptitle(
        f"Koopman Rolling Walk-Forward Validation — {symbol} {timeframe}\n"
        f"Checks whether modes are real signal or overfitting",
        color="white", fontsize=13, fontweight="bold",
    )

    xs = np.arange(n_windows)

    # ── Row 1: In-sample |λ| ─────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#161b22"); ax1.spines[:].set_color("#30363d")
    ax1.tick_params(colors="#8b949e", labelsize=8)
    for k in range(n_modes):
        ys = [abs(r["eigenvalues"][k]) if k < len(r["eigenvalues"]) else np.nan
              for r in results]
        ax1.plot(xs, ys, marker="o", ms=5, lw=1.5,
                 color=mode_colors[k % len(mode_colors)], label=f"Mode {k+1}")
    ax1.axhline(1.00, color="#ffffff", lw=0.8, ls="--", alpha=0.3, label="|λ|=1.00")
    ax1.axhline(0.95, color="#ffd54f", lw=0.7, ls=":",  alpha=0.5, label="|λ|=0.95")
    ax1.set_ylabel("In-sample |λ_k|", color="#8b949e", fontsize=9)
    ax1.set_title("In-sample Eigenvalues — stable across windows = real structure",
                  color="white", fontsize=9)
    ax1.legend(fontsize=7, labelcolor="#c9d1d9", facecolor="#161b22",
               edgecolor="#30363d", ncol=n_modes + 2)
    ax1.set_xticks(xs); ax1.set_xticklabels(win_labels, fontsize=7, color="#8b949e")
    ax1.set_ylim(0.5, 1.15)

    # ── Row 2: OOS |λ| ───────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161b22"); ax2.spines[:].set_color("#30363d")
    ax2.tick_params(colors="#8b949e", labelsize=8)
    for k in range(n_modes):
        ys_oos = [abs(r["oos_lambda"][k]) if k < len(r["oos_lambda"]) else np.nan
                  for r in results]
        ys_ins = [abs(r["eigenvalues"][k]) if k < len(r["eigenvalues"]) else np.nan
                  for r in results]
        ax2.plot(xs, ys_oos, marker="s", ms=5, lw=1.5,
                 color=mode_colors[k % len(mode_colors)],
                 label=f"Mode {k+1} OOS")
        ax2.plot(xs, ys_ins, marker="o", ms=3, lw=0.8,
                 color=mode_colors[k % len(mode_colors)],
                 alpha=0.35, ls="--")
    ax2.axhline(0.95, color="#ffd54f", lw=0.7, ls=":", alpha=0.5)
    ax2.set_ylabel("OOS |λ_k|  (solid=OOS, dashed=in-sample)", color="#8b949e", fontsize=9)
    ax2.set_title("Out-of-Sample Eigenvalues — OOS ≈ in-sample = no overfitting",
                  color="white", fontsize=9)
    ax2.legend(fontsize=7, labelcolor="#c9d1d9", facecolor="#161b22",
               edgecolor="#30363d", ncol=n_modes)
    ax2.set_xticks(xs); ax2.set_xticklabels(win_labels, fontsize=7, color="#8b949e")
    ax2.set_ylim(0.5, 1.15)

    # ── Row 3: OOS MSE ───────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#161b22"); ax3.spines[:].set_color("#30363d")
    ax3.tick_params(colors="#8b949e", labelsize=8)
    oos_mses = [r["oos_mse"] for r in results]
    bar_colors = ["#66bb6a" if v < np.median(oos_mses) else
                  "#ffd54f" if v < np.percentile(oos_mses, 75) else "#ef5350"
                  for v in oos_mses]
    ax3.bar(xs, oos_mses, color=bar_colors, alpha=0.85, width=0.6)
    ax3.axhline(np.median(oos_mses), color="#ffd54f", lw=1.0, ls="--",
                label=f"median={np.median(oos_mses):.4f}")
    ax3.set_ylabel("OOS Prediction MSE", color="#8b949e", fontsize=9)
    ax3.set_title("OOS Prediction Error — lower = model generalises to future data\n"
                  "Green: below median (good)  /  Red: above 75th percentile (bad)",
                  color="white", fontsize=9)
    ax3.legend(fontsize=8, labelcolor="#c9d1d9", facecolor="#161b22", edgecolor="#30363d")
    ax3.set_xticks(xs); ax3.set_xticklabels(win_labels, fontsize=7, color="#8b949e")

    for i, v in enumerate(oos_mses):
        ax3.text(i, v * 1.02, f"{v:.4f}", ha="center", va="bottom",
                 color="#c9d1d9", fontsize=6.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[plot] Validation saved → {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="V4 Feature visualizer — each dim in its own subplot")
    parser.add_argument("--symbol",    default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start",     default=None, help="YYYY-MM-DD (default: 30 days ago)")
    parser.add_argument("--end",       default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--days",      type=int, default=30, help="days if --start not given")
    parser.add_argument("--n-modes",   type=int, default=5,  help="Koopman modes to extract")
    parser.add_argument("--koopman-only", action="store_true", help="Generate only the Koopman plot")
    args = parser.parse_args()

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    end_ms   = _ms(args.end)   if args.end   else now_ms
    start_ms = _ms(args.start) if args.start else (end_ms - args.days * 86400 * 1000)

    # ── 데이터 취득 ───────────────────────────────────────────────────────────
    df = fetch_data(args.symbol, args.timeframe, start_ms, end_ms)
    if len(df) < 150:
        print(f"[warn] Only {len(df)} bars — features may be mostly zeros (warmup=120)")

    # ── 피처 계산 ─────────────────────────────────────────────────────────────
    cache_path = os.path.join(
        "data", f"feat_viz_{args.symbol}_{args.timeframe}.npy"
    )
    os.makedirs("data", exist_ok=True)
    feat_arr = build_features(df, cache_path)

    print(f"[info] Feature matrix: {feat_arr.shape}  (bars × dims)")

    # ── 타임스탬프 배열 ───────────────────────────────────────────────────────
    timestamps = df["ts"].dt.to_pydatetime()

    # ── 원래 기저 시각화 (28개 피처 개별 subplot) ─────────────────────────────
    if not args.koopman_only:
        out_path = os.path.join(
            "reports", f"features_v4_{args.symbol}_{args.timeframe}.png"
        )
        plot_all_features(timestamps, feat_arr, args.symbol, args.timeframe, out_path)

    # ── Koopman full pipeline (Ridge + Sparse + Rolling CV) ──────────────────
    bars_per_day = {"1h": 24, "4h": 6, "15m": 96, "1d": 1}.get(args.timeframe, 24)
    print(f"\n[koopman] Running full pipeline (n_modes={args.n_modes}) ...", flush=True)
    try:
        results = koopman_full_pipeline(
            feat_arr, timestamps,
            n_modes=args.n_modes,
            bars_per_day=bars_per_day,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[koopman] Failed: {e}")


if __name__ == "__main__":
    main()
