"""
backtesting/backtest_structural.py
────────────────────────────────────────────────────────────────────────────
QLSTM 13-dim Structural Feature Backtest — Two-Gate Validation Framework

Gate 1: Engineering sanity check — WR > BEP (25.4%).
Gate 2: Bootstrap alpha validation — p < 0.05 on mean R-multiple > 0,
        minimum 250 trades (Efron & Tibshirani, 1993).

Usage:
  python -m backtesting.backtest_structural \
    --model-path checkpoints/quantum_v2/agent_best.pt \
    --start-date 2026-01-01 --end-date 2026-03-23 \
    --timeframe 1h
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch

# ── Path setup ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtesting.validation import (
    BREAKEVEN_RATE,
    BootstrapResult,
    gate_1_system_check,
    gate_2_bootstrap_validation,
)
from src.data.binance_client import fetch_binance_taker_history
from src.data.data_client import DataClient
from src.models.features_structural import (
    FEAT_COLUMNS,
    FEAT_DIM,
    FEAT_NAMES,
    build_structural_features,
)
from src.models.integrated_agent import AgentConfig, build_quantum_agent

REST_BASE = "https://api.bybit.com"

# ── Trading parameters (from CLAUDE.md §4) ──────────────────────────────
DEFAULT_TP_MULT = 3.0       # TP = 3.0 × ATR
DEFAULT_SL_MULT = 1.0       # SL = 1.0 × ATR
DEFAULT_LEVERAGE = 10.0
DEFAULT_POS_FRAC = 0.5      # 50% of equity per trade
DEFAULT_MAX_HOLD = 96       # 96 bars = 96h on 1h TF
ETA_RT = 0.00075            # Bybit round-trip fee (maker 0.02% + taker 0.055%)
SEQ_LEN = 20                # lookback window for QLSTM inference


# ── Long/Short Ratio fetch (reused from scripts/backtest_structural.py) ──
def _fetch_ls_ratio(symbol: str, start_ms: int, end_ms: int,
                    cache_dir: str = "data") -> pd.DataFrame:
    """Fetch Bybit L/S account-ratio as liquidation proxy."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"ls_ratio_{symbol}_{start_ms}_{end_ms}.csv")
    if os.path.exists(cache_path):
        try:
            df_c = pd.read_csv(cache_path)
            if not df_c.empty and "ts_ms" in df_c.columns:
                return df_c
        except Exception:
            pass

    all_records: list[dict] = []
    current_end = end_ms
    for _ in range(500):
        params = {
            "category": "linear", "symbol": symbol,
            "period": "1h", "limit": 500,
            "endTime": int(current_end),
        }
        url = f"{REST_BASE}/v5/market/account-ratio?{urlencode(params)}"
        try:
            req = Request(url, headers={"User-Agent": "TerminalQuantSuite/1.0"})
            with urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("retCode", 0) != 0:
                break
            rows = payload.get("result", {}).get("list", [])
            if not rows:
                break
            for r in rows:
                ts_ms = int(r.get("timestamp", 0))
                all_records.append({
                    "ts_ms": ts_ms,
                    "buy_ratio": float(r.get("buyRatio", 0.5)),
                    "sell_ratio": float(r.get("sellRatio", 0.5)),
                })
            oldest_ms = min(int(r.get("timestamp", current_end)) for r in rows)
            if oldest_ms <= start_ms:
                break
            current_end = oldest_ms - 1
            time.sleep(0.08)
        except Exception:
            break

    if not all_records:
        return pd.DataFrame(columns=["ts_ms", "buy_ratio", "sell_ratio"])

    df = pd.DataFrame(all_records)
    df.drop_duplicates(subset=["ts_ms"], inplace=True)
    df.sort_values("ts_ms", inplace=True)
    df = df[(df["ts_ms"] >= start_ms) & (df["ts_ms"] <= end_ms)].reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df


# ── ATR helper ────────────────────────────────────────────────────────────
def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    atr = np.zeros(n)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    atr[:period - 1] = atr[period - 1]
    return atr


# ── Sharpe / Drawdown ────────────────────────────────────────────────────
def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def _compute_sharpe(equity_curve: np.ndarray,
                    bars_per_year: float = 8760.0) -> float:
    """Annualized Sharpe ratio. 8760 = 365*24 (1h bars)."""
    if len(equity_curve) < 2:
        return 0.0
    rets = np.diff(equity_curve) / np.where(equity_curve[:-1] != 0, equity_curve[:-1], 1.0)
    mu = np.mean(rets)
    sigma = np.std(rets)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(bars_per_year))


# ═════════════════════════════════════════════════════════════════════════
#  DATA FETCH — reuses DataClient + Binance/Bybit clients
# ═════════════════════════════════════════════════════════════════════════
def _fetch_data(symbol: str, timeframe: str,
                start_date: str, end_date: str) -> pd.DataFrame | None:
    """Fetch OHLCV + funding rate + OI + taker volume + L/S ratio."""
    dc = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")

    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"[data] {symbol} {timeframe} {start_date} -> {end_date}")
    df = dc.fetch_training_history(
        symbol=symbol, timeframe=timeframe,
        start_date=start_date, end_ms=end_ms, cache_dir="data",
    )
    if df is None or df.empty:
        print("[data] ERROR: No OHLCV data.")
        return None
    print(f"[data] {len(df)} bars loaded")

    # ── Funding Rate ──
    try:
        df_fr = dc.fetch_funding_history(symbol, start_ms, end_ms, cache_dir="data")
        if not df_fr.empty:
            df_fr["ts"] = (pd.to_datetime(df_fr["ts_ms"], unit="ms", utc=True)
                           .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            df = df.merge(df_fr[["ts", "funding_rate"]], on="ts", how="left")
            df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
            print(f"  [FR] {(df['funding_rate'] != 0).sum()}/{len(df)} bars")
        else:
            df["funding_rate"] = 0.0
    except Exception as e:
        print(f"  [FR] Skip: {e}")
        df["funding_rate"] = 0.0

    # ── Open Interest ──
    try:
        df_oi = dc.fetch_open_interest_history(
            symbol, start_ms, end_ms, interval="1h", cache_dir="data")
        if not df_oi.empty:
            df_oi["ts"] = (pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
                           .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            df = df.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
            df["open_interest"] = df["open_interest"].ffill().fillna(0.0)
            print(f"  [OI] {(df['open_interest'] != 0).sum()}/{len(df)} bars")
        else:
            df["open_interest"] = 0.0
    except Exception as e:
        print(f"  [OI] Skip: {e}")
        df["open_interest"] = 0.0

    # ── Binance Taker CVD ──
    cache_cvd = (f"data/binance_taker_{symbol}_{timeframe}_"
                 f"{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv")
    try:
        df_cvd = fetch_binance_taker_history(
            symbol=symbol, interval=timeframe,
            start_date=start_date, end_date=end_date,
            cache_path=cache_cvd,
        )
        if df_cvd is not None and not df_cvd.empty and "taker_buy_volume" in df_cvd.columns:
            cvd_map = dict(zip(df_cvd["ts"], df_cvd["taker_buy_volume"]))
            df["taker_buy_volume"] = df["ts"].map(cvd_map).fillna(0.0)
            print(f"  [CVD] {int((df['taker_buy_volume'] > 0).sum())}/{len(df)} bars")
        else:
            df["taker_buy_volume"] = 0.0
    except Exception as e:
        print(f"  [CVD] Skip: {e}")
        df["taker_buy_volume"] = 0.0

    # ── Long/Short Ratio (liquidation proxy) ──
    try:
        df_ls = _fetch_ls_ratio(symbol, start_ms, end_ms, cache_dir="data")
        if not df_ls.empty:
            df_ls["ts"] = (pd.to_datetime(df_ls["ts_ms"], unit="ms", utc=True)
                           .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            ls_map_long = dict(zip(df_ls["ts"], df_ls["buy_ratio"] * 1e6))
            ls_map_short = dict(zip(df_ls["ts"], df_ls["sell_ratio"] * 1e6))
            df["liq_long_usd"] = df["ts"].map(ls_map_long).fillna(0.0)
            df["liq_short_usd"] = df["ts"].map(ls_map_short).fillna(0.0)
            print(f"  [L/S] {len(df_ls)} records merged")
        else:
            df["liq_long_usd"] = 0.0
            df["liq_short_usd"] = 0.0
    except Exception as e:
        print(f"  [L/S] Skip: {e}")
        df["liq_long_usd"] = 0.0
        df["liq_short_usd"] = 0.0

    return df


# ═════════════════════════════════════════════════════════════════════════
#  CORE: run_backtest()
# ═════════════════════════════════════════════════════════════════════════
def run_backtest(
    model_path: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    start_date: str = "2026-01-01",
    end_date: str = "2026-03-23",
    capital: float = 10.0,
    tp_mult: float = DEFAULT_TP_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
    leverage: float = DEFAULT_LEVERAGE,
    pos_frac: float = DEFAULT_POS_FRAC,
    max_hold: int = DEFAULT_MAX_HOLD,
    seq_len: int = SEQ_LEN,
    output_dir: str = "reports",
    rmultiples_csv: str | None = None,
) -> dict[str, Any]:
    """
    Run QLSTM 13-dim structural backtest with two-gate validation.

    Returns dict with: gate1_passed, gate2_result, metrics, trades_df, r_multiples.
    """
    t0 = time.time()

    # ── 1. Data fetch ───────────────────────────────────────────────────
    df_raw = _fetch_data(symbol, timeframe, start_date, end_date)
    if df_raw is None or len(df_raw) < 300:
        print("[ERROR] Insufficient data for backtest.")
        return {"gate1_passed": False, "gate2_result": None, "error": "no_data"}

    # ── 2. Build 13-dim structural features ─────────────────────────────
    print(f"\n[features] Building structural features ({FEAT_DIM}-dim) ...")
    feats = build_structural_features(df_raw, verbose=True)
    if feats is None or len(feats) < 300:
        print("[ERROR] Feature build failed.")
        return {"gate1_passed": False, "gate2_result": None, "error": "feat_build"}

    feats = feats.astype(np.float32)  # [N, 13]
    closes = df_raw["close"].values.astype(np.float64)
    highs = df_raw["high"].values.astype(np.float64)
    lows = df_raw["low"].values.astype(np.float64)
    atr14 = _compute_atr(highs, lows, closes, period=14)

    # ── 3. Load trained model ───────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[model] Loading checkpoint: {model_path}")
    print(f"[model] Device: {device}")

    # Try loading config from checkpoint first
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg = AgentConfig(**{
            k: v for k, v in ckpt["config"].items()
            if k in AgentConfig.__dataclass_fields__
        })
        print(f"[model] Config from checkpoint: feature_dim={cfg.feature_dim}")
    else:
        # Default: 13-dim structural features
        cfg = AgentConfig(feature_dim=FEAT_DIM, n_eigenvectors=5)
        print(f"[model] Default config: feature_dim={cfg.feature_dim}")

    agent = build_quantum_agent(config=cfg, device=device)
    agent.load_checkpoint(model_path, strict=False)
    agent.to(device)  # ensure all buffers (Koopman, etc.) are on correct device
    agent.eval()
    print(f"[model] Loaded. Actions: HOLD=0, LONG=1, SHORT=2")

    # ── 4. Simulation loop ──────────────────────────────────────────────
    warmup = max(250, seq_len + 10)  # Z_WINDOW=50 + EMA200 stabilization
    eff_lev = leverage * pos_frac
    equity = float(capital)
    max_equity = equity
    max_dd_pct = 0.0

    position = 0        # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_atr = 0.0
    entry_notional = 0.0
    tp_price = 0.0
    sl_price = 0.0
    hold_bars = 0

    trades: list[dict] = []
    equity_curve: list[float] = [equity]

    n_bars = len(feats)
    print(f"\n[backtest] Simulating {n_bars - warmup} bars (warmup={warmup}) ...")
    print(f"[config] TP={tp_mult}xATR  SL={sl_mult}xATR  Lev={leverage}x  PosFrac={pos_frac}")

    for i in range(warmup, n_bars):
        price = float(closes[i])
        hi = float(highs[i])
        lo = float(lows[i])
        atr = float(atr14[i]) if atr14[i] > 0 else price * 0.01

        # ── Position management (TP/SL/MaxHold) ──────────────────
        if position != 0:
            hold_bars += 1

            hit_tp = ((position == 1 and hi >= tp_price) or
                      (position == -1 and lo <= tp_price))
            hit_sl = ((position == 1 and lo <= sl_price) or
                      (position == -1 and hi >= sl_price))
            hit_max = hold_bars >= max_hold

            exit_type = None
            exit_price = price

            if hit_sl and not hit_tp:
                exit_type = "SL"
                exit_price = sl_price
            elif hit_tp:
                exit_type = "TP"
                exit_price = tp_price
            elif hit_max:
                exit_type = "MAX_HOLD"
                exit_price = price

            if exit_type:
                raw_ret = (exit_price / entry_price - 1.0) * position
                fee = ETA_RT * eff_lev
                net_ret = raw_ret * eff_lev - fee
                pnl_usd = entry_notional * net_ret
                equity = max(equity + pnl_usd, 0.001)

                trades.append({
                    "bar_idx": i,
                    "entry_i": i - hold_bars,
                    "exit_i": i,
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry_px": entry_price,
                    "exit_px": exit_price,
                    "exit_type": exit_type,
                    "pnl_pct": net_ret * 100,
                    "pnl_usd": pnl_usd,
                    "hold": hold_bars,
                    "equity": equity,
                    "entry_atr": entry_atr,
                })

                max_equity = max(max_equity, equity)
                dd = (max_equity - equity) / max_equity * 100
                max_dd_pct = max(max_dd_pct, dd)

                position = 0
                hold_bars = 0

                if equity <= 0:
                    break
                equity_curve.append(equity)
                continue

        # ── Model inference (only when flat) ──────────────────────
        if position == 0 and i >= seq_len:
            # Build input tensor [1, seq_len, feat_dim]
            feat_window = feats[i - seq_len + 1: i + 1]  # [seq_len, 13]
            # Replace NaN with 0 for inference safety
            feat_window = np.nan_to_num(feat_window, nan=0.0)
            x = torch.from_numpy(feat_window).unsqueeze(0).to(
                device=device, dtype=torch.float32
            )  # [1, seq_len, 13]

            atr_norm = atr / price if price > 0 else 0.01

            with torch.no_grad():
                action, prob, probs = agent.select_action(
                    x, atr_norm=atr_norm, mode="greedy"
                )

            # action: 0=HOLD, 1=LONG, 2=SHORT
            if action in (1, 2) and atr > 0:
                direction = 1 if action == 1 else -1
                position = direction
                entry_price = price
                entry_atr = atr
                entry_notional = equity * pos_frac
                hold_bars = 0

                if direction == 1:  # LONG
                    tp_price = entry_price + tp_mult * atr
                    sl_price = entry_price - sl_mult * atr
                else:               # SHORT
                    tp_price = entry_price - tp_mult * atr
                    sl_price = entry_price + sl_mult * atr

        equity_curve.append(equity)

    # ── Force-close open position ────────────────────────────────────
    if position != 0 and n_bars > 0:
        exit_price = float(closes[-1])
        raw_ret = (exit_price / entry_price - 1.0) * position
        fee = ETA_RT * eff_lev
        net_ret = raw_ret * eff_lev - fee
        pnl_usd = entry_notional * net_ret
        equity = max(equity + pnl_usd, 0.001)
        trades.append({
            "bar_idx": n_bars - 1,
            "entry_i": n_bars - hold_bars - 1,
            "exit_i": n_bars - 1,
            "side": "LONG" if position == 1 else "SHORT",
            "entry_px": entry_price,
            "exit_px": exit_price,
            "exit_type": "END",
            "pnl_pct": net_ret * 100,
            "pnl_usd": pnl_usd,
            "hold": hold_bars,
            "equity": equity,
            "entry_atr": entry_atr,
        })
        equity_curve.append(equity)

    # ═════════════════════════════════════════════════════════════════
    #  METRICS & R-MULTIPLES
    # ═════════════════════════════════════════════════════════════════
    df_trades = pd.DataFrame(trades)
    n_total = len(df_trades)
    eq_arr = np.array(equity_curve)

    if n_total > 0:
        n_win = int((df_trades["pnl_usd"] > 0).sum())
        n_long = int((df_trades["side"] == "LONG").sum())
        n_short = int((df_trades["side"] == "SHORT").sum())
        n_tp = int((df_trades["exit_type"] == "TP").sum())
        n_sl = int((df_trades["exit_type"] == "SL").sum())
        n_max = int((df_trades["exit_type"] == "MAX_HOLD").sum())
        wr = n_win / n_total
        roi = (equity / capital - 1.0) * 100

        # R-multiples: raw price move / (SL distance) - fee in R units
        directions = df_trades["side"].map({"LONG": 1, "SHORT": -1}).values
        sl_distances = df_trades["entry_atr"].values * sl_mult
        raw_moves = (df_trades["exit_px"].values - df_trades["entry_px"].values) * directions
        # Fee in price units: round-trip fee * entry_price (not leveraged for R calc)
        fee_price = ETA_RT * df_trades["entry_px"].values
        r_multiples = (raw_moves - fee_price) / np.where(sl_distances > 0, sl_distances, 1.0)

        avg_win_r = float(np.mean(r_multiples[r_multiples > 0])) if (r_multiples > 0).any() else 0.0
        avg_loss_r = float(np.mean(r_multiples[r_multiples <= 0])) if (r_multiples <= 0).any() else 0.0
        sharpe = _compute_sharpe(eq_arr)
        max_dd = _compute_max_drawdown(eq_arr)

        # Profit factor
        gross_profit = float(df_trades[df_trades["pnl_usd"] > 0]["pnl_usd"].sum())
        gross_loss = abs(float(df_trades[df_trades["pnl_usd"] < 0]["pnl_usd"].sum()))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        n_win = n_long = n_short = n_tp = n_sl = n_max = 0
        wr = roi = sharpe = max_dd = pf = 0.0
        r_multiples = np.array([])
        avg_win_r = avg_loss_r = 0.0

    # ═════════════════════════════════════════════════════════════════
    #  GATE 1 — Engineering sanity check
    # ═════════════════════════════════════════════════════════════════
    gate1_passed = gate_1_system_check(n_win, n_total)

    # ═════════════════════════════════════════════════════════════════
    #  GATE 2 — Bootstrap alpha validation
    # ═════════════════════════════════════════════════════════════════
    gate2_result = gate_2_bootstrap_validation(r_multiples)

    # ═════════════════════════════════════════════════════════════════
    #  REPORT
    # ═════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    bep = 1.0 / (1.0 + tp_mult / sl_mult) * 100

    print()
    print("=" * 72)
    print("  QLSTM 13-DIM STRUCTURAL BACKTEST - TWO-GATE VALIDATION")
    print("=" * 72)
    print(f"  Period          : {start_date} -> {end_date}")
    print(f"  Bars            : {n_bars} ({timeframe}), warmup={warmup}")
    print(f"  Model           : {model_path}")
    print(f"  Feature dim     : {FEAT_DIM}")
    print("-" * 72)
    print(f"  Initial Capital : ${capital:.2f}")
    print(f"  Final Equity    : ${equity:.2f}")
    print(f"  Net ROI         : {roi:+.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:.3f}")
    print(f"  Max Drawdown    : {max_dd * 100:.2f}%")
    print(f"  Profit Factor   : {pf:.3f}")
    print("-" * 72)
    print(f"  Total Trades    : {n_total}")
    print(f"    LONG / SHORT  : {n_long} / {n_short}")
    print(f"    TP / SL / MAX : {n_tp} / {n_sl} / {n_max}")
    print(f"  Win Rate        : {wr * 100:.1f}%  (BEP={bep:.1f}%)")
    print(f"  Avg Win  (R)    : {avg_win_r:+.3f}")
    print(f"  Avg Loss (R)    : {avg_loss_r:+.3f}")
    print(f"  Mean R-multiple : {gate2_result.observed_mean:+.4f}")

    # ── Gate Results ──
    print()
    print("-" * 72)
    g1_tag = "PASS" if gate1_passed else "FAIL"
    print(f"  GATE 1 (Engineering) : {g1_tag}")
    print(f"    WR={wr * 100:.1f}% vs BEP={BREAKEVEN_RATE * 100:.1f}%")

    g2_tag = "PASS" if gate2_result.passed else "FAIL"
    print(f"  GATE 2 (Alpha)       : {g2_tag}")
    print(f"    p-value={gate2_result.p_value:.4f}  (threshold=0.05)")
    print(f"    trades={gate2_result.n_trades}  (min={250})")
    print(f"    95% CI: [{gate2_result.ci_lower:+.4f}, {gate2_result.ci_upper:+.4f}]")
    print(f"    bootstrap samples={gate2_result.n_bootstrap}")
    print("-" * 72)

    verdict = "ALPHA CONFIRMED" if (gate1_passed and gate2_result.passed) else "NO ALPHA"
    print(f"  VERDICT: {verdict}")
    print("=" * 72)
    print(f"  [Elapsed] {elapsed:.1f}s")

    # ═════════════════════════════════════════════════════════════════
    #  EXPORT
    # ═════════════════════════════════════════════════════════════════
    os.makedirs(output_dir, exist_ok=True)
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Trades CSV
    trades_csv = os.path.join(output_dir, f"qlstm13_trades_{symbol}_{ts_now}.csv")
    if n_total > 0:
        df_trades.to_csv(trades_csv, index=False, encoding="utf-8")
        print(f"[export] Trades -> {trades_csv}")

    # R-multiples CSV (detailed, timestamped)
    rmult_csv = os.path.join(output_dir, f"qlstm13_rmultiples_{symbol}_{ts_now}.csv")
    if len(r_multiples) > 0:
        pd.DataFrame({
            "r_multiple": r_multiples,
            "side": df_trades["side"].values,
            "exit_type": df_trades["exit_type"].values,
        }).to_csv(rmult_csv, index=False, encoding="utf-8")
        print(f"[export] R-multiples ({len(r_multiples)} trades) -> {rmult_csv}")

    # R-multiples CSV (single column, for statistical testing)
    if rmultiples_csv is None:
        rmultiples_csv = os.path.join(output_dir, "q1_2026_rmultiples.csv")
    if len(r_multiples) > 0:
        pd.DataFrame({"r_multiple": r_multiples}).to_csv(
            rmultiples_csv, index=False, encoding="utf-8"
        )
        print(f"[export] R-multiples (stat test) -> {rmultiples_csv}")

    # Equity curve CSV
    eq_csv = os.path.join(output_dir, f"qlstm13_equity_{symbol}_{ts_now}.csv")
    pd.DataFrame({"equity": equity_curve}).to_csv(eq_csv, index=False, encoding="utf-8")
    print(f"[export] Equity curve -> {eq_csv}")

    # Performance summary JSON
    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "period": f"{start_date} -> {end_date}",
        "model_path": model_path,
        "feature_dim": FEAT_DIM,
        "n_bars": n_bars,
        "n_trades": n_total,
        "n_win": n_win,
        "win_rate": round(wr * 100, 2),
        "roi_pct": round(roi, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(pf, 3),
        "mean_r_multiple": round(float(gate2_result.observed_mean), 4),
        "avg_win_r": round(avg_win_r, 3),
        "avg_loss_r": round(avg_loss_r, 3),
        "gate1_passed": gate1_passed,
        "gate2_passed": gate2_result.passed,
        "gate2_p_value": round(gate2_result.p_value, 4),
        "gate2_ci_lower": round(gate2_result.ci_lower, 4) if not np.isnan(gate2_result.ci_lower) else None,
        "gate2_ci_upper": round(gate2_result.ci_upper, 4) if not np.isnan(gate2_result.ci_upper) else None,
        "verdict": verdict,
        "elapsed_s": round(elapsed, 1),
    }
    summary_path = os.path.join(output_dir, f"qlstm13_report_{symbol}_{ts_now}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[export] Report -> {summary_path}")

    return {
        "gate1_passed": gate1_passed,
        "gate2_result": gate2_result,
        "metrics": summary,
        "trades_df": df_trades,
        "r_multiples": r_multiples,
        "equity_curve": eq_arr,
    }


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="QLSTM 13-dim Structural Backtest - Two-Gate Validation"
    )
    p.add_argument("--model-path", required=True,
                   help="Path to trained .pt checkpoint")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--start-date", default="2026-01-01", help="OOS start (YYYY-MM-DD)")
    p.add_argument("--end-date", default="2026-03-23", help="OOS end (YYYY-MM-DD)")
    p.add_argument("--capital", type=float, default=10.0)
    p.add_argument("--tp-mult", type=float, default=DEFAULT_TP_MULT)
    p.add_argument("--sl-mult", type=float, default=DEFAULT_SL_MULT)
    p.add_argument("--leverage", type=float, default=DEFAULT_LEVERAGE)
    p.add_argument("--pos-frac", type=float, default=DEFAULT_POS_FRAC)
    p.add_argument("--max-hold", type=int, default=DEFAULT_MAX_HOLD)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--rmultiples-csv", default=None,
                   help="Path for single-column R-multiples CSV (default: reports/q1_2026_rmultiples.csv)")

    args = p.parse_args()

    result = run_backtest(
        model_path=args.model_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        capital=args.capital,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        leverage=args.leverage,
        pos_frac=args.pos_frac,
        max_hold=args.max_hold,
        seq_len=args.seq_len,
        output_dir=args.output_dir,
        rmultiples_csv=args.rmultiples_csv,
    )

    # Exit code: 0 if both gates pass, 1 otherwise
    if result.get("gate1_passed") and result.get("gate2_result") and result["gate2_result"].passed:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
