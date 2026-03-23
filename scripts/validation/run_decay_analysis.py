"""
run_decay_analysis.py
─────────────────────────────────────────────────────────────────────────────
Signal Decay Validation Framework — Lopez de Prado style

Measures the alpha half-life of structural microstructure signals by rolling
a 90-day OLS regression window across the full history and computing the
alpha t-statistic at each window.

Methodology
───────────
1.  Load raw OHLCV + funding + OI + taker CVD data (same pipeline as
    backtest_behavioral.py).
2.  Build V4 feature array (28-dim) to extract structural signals.
3.  For each signal and forward-return horizon H:
      a.  Align: X[i] = signal value at bar i
                 y[i] = log(close[i+H] / close[i])   (forward log-return)
      b.  Roll a window of W=90 days (W*24 bars for 1h) forward in steps
          of S=7 days.
      c.  Within each window fit OLS:  y = α + β·X + ε
            beta = OLS slope coefficient (signal's predictive power)
            t    = beta / SE(beta)   — alpha t-statistic (H0: β=0)
            p    = 2*(1 - CDF_t(|t|, df=n-2))  — two-sided p-value
4.  Output: CSV time series of (window_end, signal, horizon, n, beta, t, p)
            Console summary: half-life estimate, stable windows, decay trend

Alpha Half-Life
───────────────
Fit an exponential decay model to the rolling |beta| series:
    |beta(t)| ≈ A * exp(-λ * t)
Half-life = ln(2) / λ (in days).
A shorter half-life means the signal decays faster (less durable alpha).

Usage
─────
python scripts/validation/run_decay_analysis.py
python scripts/validation/run_decay_analysis.py --symbol BTCUSDT --timeframe 1h --days 730
python scripts/validation/run_decay_analysis.py --signals fr_z liq_long liq_short cvd_div
python scripts/validation/run_decay_analysis.py --horizons 1 4 8 24 --window 90 --step 7
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Signal definitions (V4 feature indices + direction multiplier) ────────────
# direction: +1 means signal > 0 → LONG (positive return expected)
#            -1 means signal > 0 → SHORT (negative return expected)
SIGNAL_DEFS: dict[str, dict] = {
    "fr_z": {
        "idx": 18,
        "direction": -1,   # FR > 0 → crowd is long → fade → SHORT expected
        "description": "Funding Rate z-score",
    },
    "liq_long": {
        "idx": 26,
        "direction": +1,   # Long liquidation spike → oversold bounce → LONG
        "description": "Long Liquidation z-score",
    },
    "liq_short": {
        "idx": 27,
        "direction": -1,   # Short liquidation spike → overbought fade → SHORT
        "description": "Short Liquidation z-score",
    },
    "cvd_div": {
        "idx": 25,
        "direction": +1,   # CVD accumulation divergence → LONG
        "description": "CVD-Price Divergence",
    },
    "oi_change": {
        "idx": 21,
        "direction": None,  # ambiguous — both signs analysed
        "description": "OI Change %",
    },
    "cvd_trend": {
        "idx": 24,
        "direction": +1,
        "description": "CVD Trend z-score",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ols_alpha_tstat(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (beta, t, p) via OLS regression  y = α + β·X + ε.

    beta  = OLS slope — the signal's predictive coefficient
    t     = beta / SE(beta)  — alpha t-statistic (H0: β=0)
    p     = two-sided p-value

    Masks NaN pairs.  Returns (nan, nan, nan) if n < 10.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    n = len(xm)
    if n < 10:
        return float("nan"), float("nan"), float("nan")
    slope, _intercept, _r, _p_raw, std_err = stats.linregress(xm, ym)
    slope = float(slope)
    std_err = float(std_err)
    if std_err < 1e-15:
        # degenerate: no variation in residuals
        return slope, float("nan"), float("nan")
    t = slope / std_err
    p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 2)))
    return slope, float(t), p


def _half_life_days(dates: pd.DatetimeIndex, beta_abs: np.ndarray,
                    candles_per_day: int = 24) -> float | None:
    """Fit A*exp(-lambda*t) to |beta| series.  Return half-life in days."""
    mask = np.isfinite(beta_abs) & (beta_abs > 0)
    if mask.sum() < 5:
        return None
    t_days = np.array([(d - dates[0]).days for d in dates])[mask].astype(float)
    y = beta_abs[mask]
    # log-linear OLS: ln(y) = ln(A) - lambda*t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slope, intercept, _, _, _ = stats.linregress(t_days, np.log(y + 1e-9))
    lam = -slope
    if lam <= 0:
        return None
    return float(np.log(2) / lam)


def _load_data(args) -> tuple[pd.DataFrame, np.ndarray]:
    """Load OHLCV + merge funding/OI/CVD, build V4 features.
    Returns (df_raw, feats) where feats is (N, 28) numpy array.
    """
    from src.data.data_client import DataClient
    from src.data.binance_client import fetch_binance_taker_history
    from src.models.features_v4 import generate_and_cache_features_v4

    dc = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")

    print(f"[data] Fetching {args.symbol} {args.timeframe}  {start_str} → {end_str} …")
    df_raw = dc.fetch_training_history(
        symbol=args.symbol, timeframe=args.timeframe,
        start_date=start_str, end_ms=int(end_dt.timestamp() * 1000),
        cache_dir="data",
    )
    if df_raw is None or df_raw.empty:
        raise RuntimeError("[data] No OHLCV data returned.")
    print(f"[data] {len(df_raw)} bars loaded")

    _s_ms = int(start_dt.timestamp() * 1000)
    _e_ms = int(end_dt.timestamp() * 1000)

    # ── Funding Rate ──────────────────────────────────────────────────────────
    try:
        df_fr = dc.fetch_funding_history(args.symbol, _s_ms, _e_ms, cache_dir="data")
        if not df_fr.empty:
            df_fr["ts"] = (pd.to_datetime(df_fr["ts_ms"], unit="ms", utc=True)
                           .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            df_raw = df_raw.merge(df_fr[["ts", "funding_rate"]], on="ts", how="left")
            df_raw["funding_rate"] = df_raw["funding_rate"].ffill().fillna(0.0)
            print(f"  [FR]  {(df_raw['funding_rate'] != 0).sum()}/{len(df_raw)} bars merged")
        else:
            df_raw["funding_rate"] = 0.0
    except Exception as e:
        print(f"  [FR]  skip — {e}")
        df_raw["funding_rate"] = 0.0

    # ── Open Interest ─────────────────────────────────────────────────────────
    try:
        df_oi = dc.fetch_open_interest_history(
            args.symbol, _s_ms, _e_ms, interval="1h", cache_dir="data"
        )
        if not df_oi.empty:
            df_oi["ts"] = (pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
                           .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            df_raw = df_raw.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
            df_raw["open_interest"] = df_raw["open_interest"].ffill().fillna(0.0)
            print(f"  [OI]  {(df_raw['open_interest'] != 0).sum()}/{len(df_raw)} bars merged")
        else:
            df_raw["open_interest"] = 0.0
    except Exception as e:
        print(f"  [OI]  skip — {e}")
        df_raw["open_interest"] = 0.0

    # ── Binance Taker CVD ─────────────────────────────────────────────────────
    safe_tf = args.timeframe
    cache_cvd = (f"data/binance_taker_{args.symbol}_{safe_tf}_"
                 f"{start_str.replace('-','')}_{end_str.replace('-','')}.csv")
    try:
        df_cvd = fetch_binance_taker_history(
            symbol=args.symbol, interval=safe_tf,
            start_date=start_str, end_date=end_str,
            cache_path=cache_cvd,
        )
        if df_cvd is not None and not df_cvd.empty and "taker_buy_volume" in df_cvd.columns:
            cvd_map = dict(zip(df_cvd["ts"], df_cvd["taker_buy_volume"]))
            df_raw["taker_buy_volume"] = df_raw["ts"].map(cvd_map).fillna(0.0)
            print(f"  [CVD] {int((df_raw['taker_buy_volume'] > 0).sum())}/{len(df_raw)} bars merged")
        else:
            df_raw["taker_buy_volume"] = 0.0
    except Exception as e:
        print(f"  [CVD] skip — {e}")
        df_raw["taker_buy_volume"] = 0.0

    # ── Build V4 features ─────────────────────────────────────────────────────
    cache_feat = (f"data/feat_cache_decay_{args.symbol}_{safe_tf}_"
                  f"{start_str}_{end_str}_v4.npy")
    feats = generate_and_cache_features_v4(df_raw, cache_path=cache_feat)
    if feats is None or len(feats) < 200:
        raise RuntimeError("[features] Feature build failed or insufficient data.")
    print(f"[features] {len(feats)} × {feats.shape[1]}-dim V4 features built")
    return df_raw, feats


# ── Core analysis ─────────────────────────────────────────────────────────────

def run_decay_analysis(
    df_raw: pd.DataFrame,
    feats: np.ndarray,
    signals: list[str],
    horizons: list[int],
    window_days: int,
    step_days: int,
    bars_per_day: int,
) -> pd.DataFrame:
    """Compute rolling t-stat / IC for each (signal, horizon) pair.

    Returns a DataFrame with columns:
        window_end, signal, horizon_bars, n, beta, t_stat, p_value, significant
    """
    closes = df_raw["close"].values.astype(np.float64)
    N = len(feats)

    # Parse timestamps
    if "ts" in df_raw.columns:
        timestamps = pd.to_datetime(df_raw["ts"])
    else:
        timestamps = pd.RangeIndex(N)

    window_bars = window_days * bars_per_day
    step_bars   = step_days * bars_per_day

    records: list[dict] = []

    for sig_name in signals:
        if sig_name not in SIGNAL_DEFS:
            print(f"  [warn] Unknown signal '{sig_name}' — skipping")
            continue
        sdef = SIGNAL_DEFS[sig_name]
        sig_idx = sdef["idx"]
        direction = sdef["direction"] or +1   # default +1 when ambiguous
        print(f"\n[signal] {sig_name}  ({sdef['description']})  idx={sig_idx}")

        # Extract raw signal column from features
        raw_signal = feats[:, sig_idx] * direction   # flip so positive = bullish

        for H in horizons:
            # forward log-return over H bars
            fwd_return = np.full(N, np.nan)
            for i in range(N - H):
                if closes[i] > 0 and closes[i + H] > 0:
                    fwd_return[i] = np.log(closes[i + H] / closes[i])

            # rolling window
            window_records: list[dict] = []
            start = window_bars
            while start + step_bars <= N:
                end = start
                win_start = end - window_bars
                x_win = raw_signal[win_start:end]
                y_win = fwd_return[win_start:end]

                beta, t, p = _ols_alpha_tstat(x_win, y_win)

                win_end_ts = (timestamps.iloc[end - 1]
                              if hasattr(timestamps, "iloc")
                              else end - 1)

                mask_n = np.isfinite(x_win) & np.isfinite(y_win)
                window_records.append({
                    "window_end": win_end_ts,
                    "signal": sig_name,
                    "horizon_bars": H,
                    "n": int(mask_n.sum()),
                    "beta": round(beta, 6),
                    "t_stat": round(t, 3),
                    "p_value": round(p, 5),
                    "significant": bool(p < 0.05) if np.isfinite(p) else False,
                })
                start += step_bars

            records.extend(window_records)

            # Compute half-life for this (signal, horizon)
            if window_records:
                beta_arr  = np.array([r["beta"] for r in window_records], dtype=float)
                win_dates = pd.to_datetime([r["window_end"] for r in window_records])
                hl = _half_life_days(win_dates, np.abs(beta_arr), bars_per_day)
                sig_rate  = np.mean([r["significant"] for r in window_records])
                mean_beta = float(np.nanmean(beta_arr))
                hl_str    = f"{hl:.1f}d" if hl else "∞ (no decay)"
                print(f"  H={H:>2}bars  mean_beta={mean_beta:+.6f}  "
                      f"sig_rate={sig_rate:.0%}  half-life≈{hl_str}")

    return pd.DataFrame(records)


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable decay summary table."""
    if df.empty:
        print("\n[summary] No results.")
        return

    print("\n" + "=" * 78)
    print("  SIGNAL DECAY SUMMARY")
    print("=" * 78)
    print(f"  {'Signal':<14} {'H':>5} {'Periods':>8} {'Mean β':>10} "
          f"{'Mean |t|':>9} {'Sig%':>7} {'Half-Life':>12}")
    print("-" * 78)

    for (sig, H), grp in df.groupby(["signal", "horizon_bars"]):
        n_win = len(grp)
        mean_beta = grp["beta"].mean()
        mean_abst = grp["t_stat"].abs().mean()
        sig_pct   = grp["significant"].mean() * 100

        # half-life
        beta_arr  = grp["beta"].abs().values
        win_dates = pd.to_datetime(grp["window_end"])
        hl = _half_life_days(win_dates, beta_arr)
        hl_str = f"{hl:.1f}d" if hl else "—"

        # quality indicator
        if sig_pct >= 60:
            qual = "★★★ DURABLE"
        elif sig_pct >= 40:
            qual = "★★  MODERATE"
        elif sig_pct >= 20:
            qual = "★   WEAK"
        else:
            qual = "    NOISE"

        print(f"  {sig:<14} {H:>5} {n_win:>8} {mean_beta:>+10.6f} "
              f"{mean_abst:>9.2f} {sig_pct:>6.1f}% {hl_str:>12}   {qual}")

    print("=" * 78)
    print("  β = OLS slope (y = α + β·X) — signal's predictive coefficient.")
    print("  t = β / SE(β) — alpha t-statistic (H0: β=0).")
    print("  Sig% = % of 90-day windows where |t| > 2.0 (p<0.05).")
    print("  Half-Life = days for |β| to halve (exponential decay fit).")
    print("=" * 78 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Signal Decay Validation — rolling IC/t-stat analysis"
    )
    parser.add_argument("--symbol",    default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days",      type=int, default=730,
                        help="Total history in days (default: 730 = 2 years)")
    parser.add_argument("--signals",   nargs="+",
                        default=["fr_z", "liq_long", "liq_short", "cvd_div"],
                        help="Signals to analyse")
    parser.add_argument("--horizons",  nargs="+", type=int,
                        default=[1, 4, 8, 24],
                        help="Forward-return horizons in bars (default: 1 4 8 24)")
    parser.add_argument("--window",    type=int, default=90,
                        help="Rolling window in days (default: 90)")
    parser.add_argument("--step",      type=int, default=7,
                        help="Step size in days between windows (default: 7)")
    parser.add_argument("--output",    default="reports/signal_decay.csv",
                        help="Output CSV path")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip matplotlib chart")
    args = parser.parse_args()

    # bars per day for the given timeframe
    TF_BARS: dict[str, int] = {
        "1m": 1440, "3m": 480, "5m": 288, "15m": 96,
        "30m": 48,  "1h": 24,  "4h": 6,   "1d": 1,
    }
    bars_per_day = TF_BARS.get(args.timeframe, 24)

    print(f"\n{'='*60}")
    print(f"  Signal Decay Analysis — {args.symbol} {args.timeframe}")
    print(f"  Window={args.window}d  Step={args.step}d  "
          f"History={args.days}d  Bars/day={bars_per_day}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    df_raw, feats = _load_data(args)

    # ── Run analysis ──────────────────────────────────────────────────────────
    results = run_decay_analysis(
        df_raw     = df_raw,
        feats      = feats,
        signals    = args.signals,
        horizons   = args.horizons,
        window_days= args.window,
        step_days  = args.step,
        bars_per_day = bars_per_day,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(results)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[output] Saved → {out_path}  ({len(results)} rows)")

    # ── Optional plot ─────────────────────────────────────────────────────────
    if not args.no_plot:
        _plot_decay(results, args)


def _plot_decay(df: pd.DataFrame, args) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[plot] matplotlib not available — skipping chart")
        return

    signals  = df["signal"].unique().tolist()
    horizons = sorted(df["horizon_bars"].unique().tolist())
    n_sig    = len(signals)
    n_h      = len(horizons)

    fig, axes = plt.subplots(
        n_sig, n_h,
        figsize=(5 * n_h, 3 * n_sig),
        sharex=False, sharey=False,
        squeeze=False,
    )
    fig.suptitle(
        f"Signal Decay — {args.symbol} {args.timeframe} | "
        f"{args.window}d rolling OLS β",
        fontsize=13, fontweight="bold",
    )

    for row, sig in enumerate(signals):
        for col, H in enumerate(horizons):
            ax = axes[row][col]
            sub = df[(df["signal"] == sig) & (df["horizon_bars"] == H)].copy()
            if sub.empty:
                ax.set_visible(False)
                continue

            sub["window_end"] = pd.to_datetime(sub["window_end"])
            sub = sub.sort_values("window_end")

            # OLS beta time series
            ax.bar(sub["window_end"], sub["beta"],
                   color=["#2ecc71" if v >= 0 else "#e74c3c"
                           for v in sub["beta"]],
                   alpha=0.7, width=timedelta(days=args.step * 0.9))
            ax.axhline(0, color="black", linewidth=0.8)

            # t=±2 threshold lines (approximate β where t=2 given mean n)
            mean_n = sub["n"].clip(lower=4).mean()
            mean_se = sub["beta"].abs().mean() / max(sub["t_stat"].abs().mean(), 1e-6)
            threshold = 2.0 * mean_se
            ax.axhline(+threshold, color="gray", linewidth=0.7,
                       linestyle="--", label="t=±2 approx")
            ax.axhline(-threshold, color="gray", linewidth=0.7, linestyle="--")

            ax.set_title(f"{sig}  |  H={H}bars", fontsize=9)
            ax.set_ylabel("OLS β (alpha t-stat)", fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
                     fontsize=7)

    fig.tight_layout()
    plot_path = Path(args.output).with_suffix(".png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"[plot]   Saved → {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
