# -*- coding: utf-8 -*-
"""
run_liq_event_study.py — Jose (Risk Manager)
Executes the liquidation cascade event study and outputs charts to reports/.
"""
import matplotlib
matplotlib.use("Agg")
import warnings, sys, io
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp949 UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, optimize

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ── Parameters ──────────────────────────────────────────────────────────────
WINDOW_BARS    = 60      # ±60 bars (1h TF = ±60h ≈ ±2.5 days)
SPIKE_THRESH   = 2.0     # liq_long_z > 2σ = "high"
EXTREME_THRESH = 3.0     # liq_long_z > 3σ = "extreme"
MIN_EVENT_GAP  = 24      # bars between de-clustered events
Z_WINDOW       = 50      # rolling z-score lookback

plt.style.use("dark_background")
C = dict(mean="#00d4ff", p2575="#0066cc", extreme="#ff6600",
         baseline="#888888", white="#ffffff", bg="#0a0a0a", panel="#111111")


# ── Data Loading ─────────────────────────────────────────────────────────────
def load_ohlcv(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "training_BTCUSDT_1h_20190101.csv",
                     parse_dates=["ts"], encoding="utf-8")
    df = df.rename(columns={"ts": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.set_index("timestamp").sort_index()


def load_funding(d: Path) -> pd.DataFrame:
    frames = []
    for fn in [
        "funding_BTCUSDT_1546300800000_1672531140000.csv",
        "funding_BTCUSDT_1672531200000_1735775940000.csv",
    ]:
        p = d / fn
        if p.exists():
            df = pd.read_csv(p, encoding="utf-8")
            df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
            frames.append(
                df[["timestamp", "funding_rate"]].set_index("timestamp").sort_index()
            )
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


# ── Feature Computation ───────────────────────────────────────────────────────
def rolling_zscore(s: pd.Series, w: int = Z_WINDOW) -> pd.Series:
    mu  = s.rolling(w, min_periods=w).mean()
    std = s.rolling(w, min_periods=w).std(ddof=1).replace(0, np.nan)
    return (s - mu) / std


def build_master(d: Path) -> pd.DataFrame:
    ohlcv   = load_ohlcv(d)
    funding = load_funding(d)

    m = ohlcv.copy()
    if not funding.empty:
        m = m.join(funding.resample("1h").ffill(), how="left")
        m["funding_rate"] = m["funding_rate"].ffill().fillna(0.0)
    else:
        m["funding_rate"] = 0.0

    # liq_long_z : lower-wick × volume  (long liquidation proxy)
    liq_long_raw  = (m["open"] - m["low"]).clip(lower=0) * m["volume"]
    # liq_short_z : upper-wick × volume (short liquidation proxy)
    liq_short_raw = (m["high"] - m["open"]).clip(lower=0) * m["volume"]

    # ATR
    tr = pd.concat([
        m["high"] - m["low"],
        (m["high"] - m["close"].shift()).abs(),
        (m["low"]  - m["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()

    m["liq_long_z"]  = rolling_zscore(liq_long_raw)
    m["liq_short_z"] = rolling_zscore(liq_short_raw)
    m["vol_regime"]  = rolling_zscore(atr)
    m["fr_z"]        = rolling_zscore(m["funding_rate"])
    m = m.dropna(subset=["liq_long_z"])
    return m


# ── Event Identification ──────────────────────────────────────────────────────
def identify_events(series: pd.Series, threshold: float,
                    min_gap: int = MIN_EVENT_GAP) -> pd.DatetimeIndex:
    """De-clustered local maxima above threshold."""
    candidates = series[series > threshold].sort_index()
    selected: list = []
    last_pos: int | None = None
    idx_map = {ts: i for i, ts in enumerate(series.index)}
    for ts in candidates.index:
        pos = idx_map[ts]
        if last_pos is None or (pos - last_pos) >= min_gap:
            selected.append(ts)
            last_pos = pos
        elif series[ts] > series[selected[-1]]:
            selected[-1] = ts
            last_pos = pos
    return pd.DatetimeIndex(selected)


# ── Event Matrix ─────────────────────────────────────────────────────────────
def build_event_matrix(events: pd.DatetimeIndex, price: pd.Series,
                       window: int) -> pd.DataFrame:
    """Log-return (%) relative to t=0 for each event."""
    all_idx = price.index.tolist()
    idx_map = {ts: i for i, ts in enumerate(all_idx)}
    n = len(price)
    records = []
    for ts in events:
        pos = idx_map.get(ts)
        if pos is None:
            continue
        s, e = pos - window, pos + window
        if s < 0 or e >= n:
            continue
        wp = price.iloc[s : e + 1].values
        p0 = wp[window]
        if p0 <= 0:
            continue
        records.append(np.log(wp / p0) * 100)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records, columns=list(range(-window, window + 1)))


def build_signal_matrix(events: pd.DatetimeIndex, signal: pd.Series,
                        window: int) -> pd.DataFrame:
    """Delta of a signal relative to its value at t=0."""
    all_idx = signal.index.tolist()
    idx_map = {ts: i for i, ts in enumerate(all_idx)}
    n = len(signal)
    records = []
    for ts in events:
        pos = idx_map.get(ts)
        if pos is None:
            continue
        s, e = pos - window, pos + window
        if s < 0 or e >= n:
            continue
        seg = signal.iloc[s : e + 1].values
        records.append(seg - seg[window])
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records, columns=list(range(-window, window + 1)))


# ── Chart Helpers ─────────────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors="#aaa")
    ax.spines[:].set_color("#333")


def plot_band(ax, mat: pd.DataFrame, t_axis, label: str, color: str,
              alpha_fill: float = 0.25):
    if mat.empty:
        return
    ax.fill_between(t_axis, mat.quantile(0.10), mat.quantile(0.90),
                    alpha=alpha_fill * 0.5, color=color)
    ax.fill_between(t_axis, mat.quantile(0.25), mat.quantile(0.75),
                    alpha=alpha_fill, color=color)
    ax.plot(t_axis, mat.mean().values, color=color, lw=2.5, label=label)


# ════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data …")
    master = build_master(DATA_DIR)
    print(f"  Master: {master.shape}  "
          f"{master.index.min().date()} → {master.index.max().date()}")
    print(f"  liq_long_z: mean={master.liq_long_z.mean():.3f}  "
          f"std={master.liq_long_z.std():.3f}  max={master.liq_long_z.max():.2f}")

    events_high    = identify_events(master["liq_long_z"], SPIKE_THRESH)
    events_extreme = identify_events(master["liq_long_z"], EXTREME_THRESH)
    print(f"  HIGH events    (z>{SPIKE_THRESH}):  {len(events_high)}")
    print(f"  EXTREME events (z>{EXTREME_THRESH}):  {len(events_extreme)}")

    t_axis = np.arange(-WINDOW_BARS, WINDOW_BARS + 1)

    mat_high    = build_event_matrix(events_high,    master["close"], WINDOW_BARS)
    mat_extreme = build_event_matrix(events_extreme, master["close"], WINDOW_BARS)
    mat_short_z = build_signal_matrix(events_high, master["liq_short_z"], WINDOW_BARS)

    # Vol-regime split
    hv = [ts for ts in events_high
          if ts in master.index and master.loc[ts, "vol_regime"] > 0]
    lv = [ts for ts in events_high
          if ts in master.index and master.loc[ts, "vol_regime"] <= 0]
    mat_hv = build_event_matrix(pd.DatetimeIndex(hv), master["close"], WINDOW_BARS)
    mat_lv = build_event_matrix(pd.DatetimeIndex(lv), master["close"], WINDOW_BARS)

    # ── Chart 1: Main 6-panel event study ───────────────────────────────────
    print("\nRendering Chart 1: Main event study …")
    fig = plt.figure(figsize=(18, 14), facecolor=C["bg"])
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_main   = fig.add_subplot(gs[0, :])
    ax_ext    = fig.add_subplot(gs[1, 0])
    ax_hist   = fig.add_subplot(gs[1, 1])
    ax_vol    = fig.add_subplot(gs[2, 0])
    ax_regime = fig.add_subplot(gs[2, 1])

    # Panel 1: Main trajectory
    plot_band(ax_main, mat_high, t_axis,
              f"High liq_long_z > {SPIKE_THRESH}s  (n={len(mat_high)})",
              C["mean"])
    plot_band(ax_main, mat_extreme, t_axis,
              f"Extreme liq_long_z > {EXTREME_THRESH}s  (n={len(mat_extreme)})",
              C["extreme"])
    ax_main.axvline(0, color=C["white"], lw=1.5, ls="--", alpha=0.8, label="Event t=0")
    ax_main.axhline(0, color="#555",     lw=1,   ls=":")
    ax_main.axvspan(0, WINDOW_BARS, alpha=0.04, color="white")
    ax_main.set_xlabel(f"Bars relative to liq_long_z spike  (1 bar = 1h)", color="#aaa")
    ax_main.set_ylabel("Avg Log-Return from t=0 (%)", color="#aaa")
    ax_main.set_title(
        f"BTCUSDT 1h — Avg Price Trajectory Around Long-Liquidation Spikes\n"
        f"Window: ±{WINDOW_BARS} bars  |  De-clustered (min gap={MIN_EVENT_GAP}h)",
        color="white", fontsize=12)
    ax_main.legend(fontsize=9, loc="upper left")
    ax_main.set_xlim(-WINDOW_BARS, WINDOW_BARS)
    style_ax(ax_main)

    # Panel 2: Individual extreme paths
    if not mat_extreme.empty:
        for _, row in mat_extreme.iterrows():
            ax_ext.plot(t_axis, row.values, alpha=0.10, color=C["extreme"], lw=0.7)
        ax_ext.plot(t_axis, mat_extreme.mean().values,
                    color=C["extreme"], lw=2.5, label=f"Mean (n={len(mat_extreme)})")
    ax_ext.axvline(0, color=C["white"], lw=1.5, ls="--")
    ax_ext.axhline(0, color="#555", lw=1, ls=":")
    ax_ext.set_title(f"Individual Extreme Events (z > {EXTREME_THRESH}σ)", color="white")
    ax_ext.set_xlabel("Bars", color="#aaa")
    ax_ext.set_ylabel("Log-Return (%)", color="#aaa")
    ax_ext.legend(fontsize=9)
    ax_ext.set_xlim(-WINDOW_BARS, WINDOW_BARS)
    style_ax(ax_ext)

    # Panel 3: liq_long_z histogram
    ax_hist.hist(master["liq_long_z"].clip(-3, 8),
                 bins=100, color=C["mean"], alpha=0.7, density=True)
    ax_hist.axvline(SPIKE_THRESH,   color=C["mean"],    lw=2, ls="--",
                    label=f"{SPIKE_THRESH}σ  (High)")
    ax_hist.axvline(EXTREME_THRESH, color=C["extreme"], lw=2, ls="--",
                    label=f"{EXTREME_THRESH}σ  (Extreme)")
    ax_hist.set_title("liq_long_z Distribution", color="white")
    ax_hist.set_xlabel("z-score", color="#aaa")
    ax_hist.set_ylabel("Density", color="#aaa")
    ax_hist.legend(fontsize=9)
    style_ax(ax_hist)

    # Panel 4: Vol-regime breakdown
    if not mat_hv.empty:
        ax_vol.plot(t_axis, mat_hv.mean().values,
                    color="#ff4444", lw=2, label=f"High-Vol Regime (n={len(mat_hv)})")
        ax_vol.fill_between(t_axis, mat_hv.quantile(0.25).values,
                             mat_hv.quantile(0.75).values,
                             alpha=0.2, color="#ff4444")
    if not mat_lv.empty:
        ax_vol.plot(t_axis, mat_lv.mean().values,
                    color="#44ff88", lw=2, label=f"Low-Vol Regime (n={len(mat_lv)})")
        ax_vol.fill_between(t_axis, mat_lv.quantile(0.25).values,
                             mat_lv.quantile(0.75).values,
                             alpha=0.2, color="#44ff88")
    ax_vol.axvline(0, color=C["white"], lw=1.5, ls="--")
    ax_vol.axhline(0, color="#555", lw=1, ls=":")
    ax_vol.set_title("Cascade Impact by Volatility Regime", color="white")
    ax_vol.set_xlabel("Bars", color="#aaa")
    ax_vol.set_ylabel("Avg Log-Return (%)", color="#aaa")
    ax_vol.legend(fontsize=9)
    ax_vol.set_xlim(-WINDOW_BARS, WINDOW_BARS)
    style_ax(ax_vol)

    # Panel 5: Box plot at key horizons
    horizon_list = [h for h in [4, 24, 48] if h in mat_high.columns]
    bdata   = [mat_high[h].dropna().values for h in horizon_list]
    blabels = [f"+{h}h" for h in horizon_list]
    if bdata:
        bp = ax_regime.boxplot(bdata, labels=blabels, patch_artist=True,
                                medianprops=dict(color="white", lw=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(C["p2575"])
            patch.set_alpha(0.7)
    ax_regime.axhline(0, color="#555", lw=1, ls=":")
    ax_regime.set_title("Post-Event Return Distribution at Key Horizons", color="white")
    ax_regime.set_ylabel("Log-Return from t=0 (%)", color="#aaa")
    ax_regime.set_xlabel("Horizon", color="#aaa")
    style_ax(ax_regime)

    plt.suptitle(
        "Jose — Liquidation Cascade Impact Analysis  |  BTCUSDT 1h  |  2020–2022",
        color="white", fontsize=14, y=0.99)
    out1 = REPORTS_DIR / "liq_cascade_event_study.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  ✅ Chart 1 saved → {out1}")

    # ── Statistical Table ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("LIQUIDATION CASCADE IMPACT — STATISTICAL SUMMARY")
    print("=" * 72)
    print(f"Dataset: BTCUSDT 1h  |  {master.index.min().date()} → {master.index.max().date()}")
    print(f"Total bars: {len(master):,}  ({len(master)/24/365.25:.1f} years)\n")

    for label, mat, thresh in [
        (f"High (z>{SPIKE_THRESH}σ)",    mat_high,    SPIKE_THRESH),
        (f"Extreme (z>{EXTREME_THRESH}σ)", mat_extreme, EXTREME_THRESH),
    ]:
        if mat.empty:
            print(f"{label}: no events\n")
            continue
        base_rate = len(mat) / len(master) * 100
        print(f"── {label}  n={len(mat)}  ({base_rate:.2f}% of bars)")
        print(f"  {'H':<5} {'Mean%':>8} {'Med%':>8} {'Win%':>7} {'t-stat':>8} {'p-val':>8}  Sig")
        for h in [1, 2, 4, 8, 12, 24, 36, 48, 60]:
            if h not in mat.columns:
                continue
            col = mat[h].dropna()
            t, p = stats.ttest_1samp(col, 0)
            sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"  +{h:<4} {col.mean():>+8.3f} {col.median():>+8.3f} "
                  f"{(col > 0).mean()*100:>6.1f}% {t:>+8.2f} {p:>8.4f}  {sig}")
        print()

    # ── Chart 2: Alpha Half-Life ─────────────────────────────────────────────
    print("Rendering Chart 2: Alpha half-life …")
    horizons_all = [h for h in [1,2,4,6,8,12,16,24,36,48,60]
                    if h in mat_high.columns]
    sig_strength = [abs(mat_high[h].mean()) for h in horizons_all]
    try:
        def exp_decay(t, A, tau):
            return A * np.exp(-t / tau)
        popt, _ = optimize.curve_fit(exp_decay, horizons_all, sig_strength,
                                      p0=[sig_strength[0], 24], maxfev=5000)
        tau_fit, hl = popt[1], popt[1] * np.log(2)

        fig2, ax = plt.subplots(figsize=(10, 5), facecolor=C["bg"])
        style_ax(ax)
        ax.scatter(horizons_all, sig_strength, color=C["mean"], s=60, zorder=5,
                   label="|Mean log-ret| at horizon")
        tf = np.linspace(0.5, WINDOW_BARS, 200)
        ax.plot(tf, exp_decay(tf, *popt), color=C["extreme"], lw=2,
                label=f"Exp fit: τ={tau_fit:.1f}h  →  Half-life={hl:.1f}h")
        ax.axvline(hl, color=C["extreme"], lw=1.5, ls="--", alpha=0.7,
                   label=f"Half-life = {hl:.1f}h")
        ax.set_xlabel("Horizon (bars = hours)", color="#aaa")
        ax.set_ylabel("|Mean Log-Return| (%)", color="#aaa")
        ax.set_title("Signal Decay — Alpha Half-Life of liq_long_z Spike",
                      color="white", fontsize=12)
        ax.legend(fontsize=10)
        out2 = REPORTS_DIR / "liq_alpha_halflife.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=C["bg"])
        plt.close()
        print(f"  ✅ Chart 2 saved → {out2}")
        print(f"\n  📐 Alpha Half-Life: {hl:.1f}h ({hl/24:.1f} days)")
        print(f"     Decay constant τ = {tau_fit:.1f}h")
    except Exception as ex:
        print(f"  Curve fit failed: {ex}")

    # ── Chart 3: Cascade dynamics ────────────────────────────────────────────
    print("\nRendering Chart 3: Cascade dynamics …")
    fig3, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=C["bg"])
    for ax in axes:
        style_ax(ax)

    ax = axes[0]
    if not mat_high.empty:
        ax.fill_between(t_axis, mat_high.quantile(0.25), mat_high.quantile(0.75),
                        alpha=0.3, color=C["mean"])
        ax.plot(t_axis, mat_high.mean().values, color=C["mean"], lw=2.5,
                label="Mean price log-ret %")
    ax.axvline(0, color=C["white"], lw=1.5, ls="--")
    ax.axhline(0, color="#555",     lw=1,   ls=":")
    ax.set_title("Price Trajectory After liq_long_z Spike", color="white")
    ax.set_xlabel("Bars (1 bar = 1h)", color="#aaa")
    ax.set_ylabel("Log-Return (%)", color="#aaa")
    ax.legend(fontsize=9)
    ax.set_xlim(-WINDOW_BARS, WINDOW_BARS)

    ax2 = axes[1]
    if not mat_short_z.empty:
        ax2.fill_between(t_axis, mat_short_z.quantile(0.25),
                         mat_short_z.quantile(0.75), alpha=0.3, color=C["extreme"])
        ax2.plot(t_axis, mat_short_z.mean().values, color=C["extreme"], lw=2.5,
                 label="Δliq_short_z (short-squeeze proxy)")
    ax2.axvline(0, color=C["white"], lw=1.5, ls="--")
    ax2.axhline(0, color="#555",     lw=1,   ls=":")
    ax2.set_title("liq_short_z Response Post Long-Cascade", color="white")
    ax2.set_xlabel("Bars (1 bar = 1h)", color="#aaa")
    ax2.set_ylabel("Δ liq_short_z from t=0", color="#aaa")
    ax2.legend(fontsize=9)
    ax2.set_xlim(-WINDOW_BARS, WINDOW_BARS)

    plt.suptitle("Cascade Dynamics: Long-Liq → Price → Short-Squeeze Signal",
                  color="white", fontsize=13, y=1.02)
    plt.tight_layout()
    out3 = REPORTS_DIR / "liq_cascade_dynamics.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  ✅ Chart 3 saved → {out3}")

    print("\n✅ All outputs complete.")
    print(f"   reports/liq_cascade_event_study.png")
    print(f"   reports/liq_alpha_halflife.png")
    print(f"   reports/liq_cascade_dynamics.png")


if __name__ == "__main__":
    main()
