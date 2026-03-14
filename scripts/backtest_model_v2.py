"""
backtest_model_v2.py
--------------------------------------------------------------------
Quantum Trading V2 -- Phase 4: Backtest System (Improved)

Fixes applied vs original:
  #1  V2 feature builder: raw per-bar 27-dim features for SpectralDecomposer
      (no multi-period compression -- covariance over T bars is meaningful)
  #2  Proportional fee: eta * notional_at_entry (not fixed initial capital)
  #3  Current-equity position sizing (not always initial capital)
  #4  Gamma^t time decay tracked (time-discounted PnL column added)
  #5  J(tau) score tracked alongside raw PnL (gamma_pnl + R_agile bonus)
  #6  Same-bar SL/TP ordering: bar-direction heuristic (not always SL first)
  #7  True Wilder ATR: max(H-L, |H-prevC|, |L-prevC|) via EWM(span=14)
  #8  Confidence threshold default lowered to 0.55 (matches AgentConfig)
  #9  Win-rate broken out by exit type in report
  #10 Feature cache key includes end_date (prevents stale cache hits)
  #11 Maker vs taker fee distinction (entries=maker, exits=taker)
  #12 N/A: entry-candle H/L not applicable for close-price entries

Usage:
    python scripts/backtest_model_v2.py --days 30 --timeframe 15m
    python scripts/backtest_model_v2.py --start-date 2026-01-01 --end-date 2026-02-01 --visualize
    python scripts/backtest_model_v2.py --model-path checkpoints/quantum_v2/agent_latest.pt
--------------------------------------------------------------------
"""

import argparse
import math
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.models.integrated_agent import build_quantum_agent, AgentConfig
from src.data.data_client import DataClient
from src.data.bybit_mainnet import BybitMainnetClient, REST_BASE
from src.models.features_v2 import (
    compute_true_atr,
    build_features_v2,
    generate_and_cache_features_v2,
)
from src.models.features_v4 import generate_and_cache_features_v4
from src.strategies.hmm_regime import HMMRegimeDetector, RegimeState
from src.data.binance_client import fetch_binance_taker_history


# ── Metrics ───────────────────────────────────────────────────────

def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


def compute_sharpe(equity_curve: np.ndarray, periods_per_year: float = 35040) -> float:
    if len(equity_curve) < 2:
        return 0.0
    returns = np.diff(equity_curve) / np.where(equity_curve[:-1] != 0, equity_curve[:-1], 1.0)
    mu    = np.mean(returns)
    sigma = np.std(returns)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


# ── Main backtest ─────────────────────────────────────────────────

def run_backtest_v2():
    parser = argparse.ArgumentParser(description="Quantum Sniper V2 Backtest (Improved)")
    parser.add_argument("--symbol",      default="BTCUSDT")
    parser.add_argument("--timeframe",   default="15m")
    parser.add_argument("--start-date",  default=None)
    parser.add_argument("--end-date",    default=None)
    parser.add_argument("--days",        type=int,   default=30)
    parser.add_argument("--model-path",  default="checkpoints/quantum_v2/agent_latest.pt")
    parser.add_argument("--ensemble-dir", default=None,
                        help="앙상블 디렉토리 (지정 시 --model-path 무시). "
                             "예: checkpoints/quantum_v2/ensemble")
    parser.add_argument("--leverage",    type=float, default=25.0)
    # Fix #8: default lowered from 0.75 → 0.55 to match AgentConfig
    parser.add_argument("--confidence",  type=float, default=0.55,
                        help="Min action confidence to enter (was 0.75; now 0.55 = AgentConfig default)")
    parser.add_argument("--capital",     type=float, default=20.0)
    # Fix #11: separate maker/taker fees — Bybit USDT Perp 실제 수수료 기준
    # https://www.bybit.com/en/help-center/article/Trading-Fee-Structure
    # Regular user: Maker 0.02%, Taker 0.055%
    parser.add_argument("--eta-maker",   type=float, default=0.0002,
                        help="Maker fee rate (entry, limit orders) default=0.02%% (Bybit regular)")
    parser.add_argument("--eta-taker",   type=float, default=0.00055,
                        help="Taker fee rate (exit, market orders)  default=0.055%% (Bybit regular)")
    parser.add_argument("--eta",         type=float, default=None,
                        help="Legacy: single fee rate (overrides both eta-maker and eta-taker)")
    parser.add_argument("--tp",          type=float, default=None,
                        help="TP threshold fraction override (None = use ATR-based: tp_mult × ATR)")
    parser.add_argument("--sl",          type=float, default=None,
                        help="SL threshold fraction override (None = use ATR-based: sl_mult × ATR)")
    parser.add_argument("--tp-mult",     type=float, default=4.0,
                        help="TP = tp_mult × ATR / price  (학습과 동일, default=4.0)")
    parser.add_argument("--sl-mult",     type=float, default=1.0,
                        help="SL = sl_mult × ATR / price  (학습과 동일, default=1.0)")
    parser.add_argument("--pos-frac",    type=float, default=0.5,
                        help="Fraction of current equity used per trade (0.5 = 50%% of net balance, default=0.5)")
    parser.add_argument("--max-hold",    type=int,   default=96,
                        help="Max bars to hold (0=unlimited)")
    parser.add_argument("--trail-after", type=float, default=0.0,
                        help="Trailing stop activation threshold in ATR units. 0=disabled. e.g. 1.5")
    parser.add_argument("--trail-dist",  type=float, default=1.0,
                        help="Trailing stop distance from watermark in ATR units. default=1.0")
    parser.add_argument("--regime-exit-conf", type=float, default=0.0,
                        help="Regime-flip early exit: exit if model signals opposite direction "
                             "with this confidence. 0=disabled (default). e.g. 0.60")
    parser.add_argument("--gamma",       type=float, default=0.99,
                        help="Discount factor for J(tau) time-decay (Fix #4/#5)")
    parser.add_argument("--r-agile",     type=float, default=0.7,
                        help="R_agile reward scalar for strategic exits in J(tau) (Fix #5)")
    parser.add_argument("--visualize",     action="store_true")
    parser.add_argument("--output-dir",    default="reports")
    parser.add_argument("--regime-gate",   action="store_true",
                        help="Enable Hurst+vol regime gate: block entries in mean-reverting / extreme-vol regimes")
    parser.add_argument("--hurst-min",     type=float, default=0.50,
                        help="Regime gate: min Hurst exponent to allow entry (default=0.50, trending only)")
    parser.add_argument("--vol-pct-max",   type=float, default=0.85,
                        help="Regime gate: max vol percentile to allow entry (default=0.85, avoid vol spikes)")
    parser.add_argument("--no-cr-filter",  action="store_true",
                        help="Disable CramerRao entry filter entirely")
    parser.add_argument("--hmm-gate",      action="store_true",
                        help="Enable HMM 3-state regime gate (replaces Hurst threshold). "
                             "UP_TREND→LONG only, DOWN_TREND→SHORT only, CHOPPY→no entry")
    parser.add_argument("--hmm-train-days", type=int, default=90,
                        help="Days of pre-test history to fit HMM on (default=90)")
    parser.add_argument("--cr-snr-min",    type=float, default=None,
                        help="Override cr_snr_min threshold (default: use AgentConfig value)")
    parser.add_argument("--cr-diagnostics", action="store_true",
                        help="Print CR filter value distribution over first 100 bars")
    parser.add_argument("--ema-gate",      action="store_true",
                        help="EMA200 regime gate: BULL→SHORT차단, BEAR→LONG차단 (default: off)")
    parser.add_argument("--ema-band",      type=float, default=0.02,
                        help="EMA200 기준 ±band 이내는 RANGE로 양방향 허용 (default=0.02 = 2%%)")
    parser.add_argument("--hurst-adaptive", action="store_true",
                        help="Hurst 적응 물리 게이트 (hard constraint):\n"
                             "  H>0.55 (추세): 모델 신호가 최근 20봉 방향과 일치해야만 진입\n"
                             "  H<0.45 (평균회귀): 모델 신호가 최근 20봉 방향의 반대여야 진입\n"
                             "  0.45≤H≤0.55 (랜덤워크): 진입 차단")
    parser.add_argument("--hurst-trend",   type=float, default=0.55,
                        help="Hurst 추세 경계 (default=0.55, H>this → trending)")
    parser.add_argument("--hurst-mr",      type=float, default=0.45,
                        help="Hurst 평균회귀 경계 (default=0.45, H<this → mean-reverting)")
    parser.add_argument("--hurst-window",  type=int,   default=200,
                        help="Hurst 계산 윈도우 (log-return bars, default=200)")
    parser.add_argument("--hurst-dir-window", type=int, default=20,
                        help="방향성 판단 윈도우 (bars, default=20)")
    args = parser.parse_args()

    # Legacy --eta overrides both
    if args.eta is not None:
        args.eta_maker = args.eta
        args.eta_taker = args.eta

    use_atr_barrier = (args.tp is None and args.sl is None)
    tp_label = f"{args.tp_mult}×ATR" if use_atr_barrier else f"{args.tp*100:.2f}%%"
    sl_label = f"{args.sl_mult}×ATR" if use_atr_barrier else f"{args.sl*100:.2f}%%"

    print(f"[backtest] Quantum Sniper V2 (Improved): {args.symbol} ({args.timeframe})")
    print(f"[backtest] Leverage={args.leverage}x  pos_frac={args.pos_frac*100:.0f}%  "
          f"eff_leverage={args.leverage*args.pos_frac:.1f}x  "
          f"eta_maker={args.eta_maker*10000:.2f}bps({args.eta_maker*100:.4f}%)  "
          f"eta_taker={args.eta_taker*10000:.2f}bps({args.eta_taker*100:.4f}%)  "
          f"round_trip={(args.eta_maker+args.eta_taker)*100:.4f}%  "
          f"TP={tp_label}  SL={sl_label}  conf={args.confidence}")

    # ── 1. Build agent ──
    cr_snr_override = args.cr_snr_min if args.cr_snr_min is not None else 0.15
    config = AgentConfig(
        leverage=args.leverage,
        eta_base=args.eta_maker,
        confidence_threshold=args.confidence,
        checkpoint_dir=os.path.dirname(args.model_path) if args.model_path else "checkpoints/quantum_v2",
        use_cr_filter=(not args.no_cr_filter),
        cr_snr_min=cr_snr_override,
    )
    print(f"[config] CR filter: {'OFF' if args.no_cr_filter else 'ON'}  "
          f"cr_snr_min={config.cr_snr_min:.2f}  "
          f"cr_hurst_min={config.cr_hurst_min:.2f}  "
          f"cr_purity_min={config.cr_purity_min:.2f}")
    device = torch.device("cpu")  # SpectralDecomposer EDMD lstsq CUDA 불안정 → CPU 사용

    # ── 앙상블 모드 vs 단일 모델 모드 ──────────────────────────────────────
    if args.ensemble_dir and os.path.isdir(args.ensemble_dir):
        from src.models.ensemble_agent import load_ensemble
        agent = load_ensemble(
            ensemble_dir=args.ensemble_dir,
            device=device,
            confidence_threshold=args.confidence,
        )
        print(f"[model] Ensemble loaded: {args.ensemble_dir}")
    else:
        agent = build_quantum_agent(config=config, device=device)
        if os.path.exists(args.model_path):
            agent.load_checkpoint(args.model_path)
            print(f"[model] Checkpoint loaded: {args.model_path}")
        else:
            print(f"[model] WARNING: No checkpoint at {args.model_path}, using random init.")

    agent.eval()

    # ── 2. Fetch data (Bybit mainnet only) ──
    dc = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")

    # Fail-fast: confirm the underlying client is BybitMainnetClient
    if not isinstance(dc._bybit, BybitMainnetClient):
        print(f"[data] FATAL: expected BybitMainnetClient, got {type(dc._bybit)}. Aborting.")
        sys.exit(1)
    print(f"[data] Source: Bybit Mainnet  REST={REST_BASE}")

    end_dt = datetime.now(timezone.utc)
    if args.end_date:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if args.start_date:
        start_date_str = args.start_date
    else:
        start_date_str = (end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")

    end_date_str = end_dt.strftime("%Y-%m-%d")  # Fix #10

    print(f"[data] Fetching {args.symbol} {args.timeframe} from {start_date_str} to {end_date_str} ...")
    df_raw = dc.fetch_training_history(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start_date_str,
        end_ms=int(end_dt.timestamp() * 1000),
        cache_dir="data",
    )

    if df_raw is None or df_raw.empty:
        print("[data] ERROR: No data returned. Check network / date range.")
        return

    print(f"[data] Got {len(df_raw)} bars")

    # ── 2.5 HMM Regime Detector — fit on pre-test history ────────────────────
    hmm_detector: HMMRegimeDetector | None = None
    if getattr(args, "hmm_gate", False):
        try:
            _hmm_train_days = getattr(args, "hmm_train_days", 90)
            _hmm_end_dt     = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            _hmm_start_dt   = _hmm_end_dt - timedelta(days=_hmm_train_days)
            print(f"[HMM] Fetching pre-test training data "
                  f"{_hmm_start_dt.date()} ~ {_hmm_end_dt.date()} ...")
            _hmm_dc  = DataClient()
            _hmm_df  = _hmm_dc.fetch_training_history(
                symbol    = args.symbol,
                timeframe = args.timeframe,
                start_date= _hmm_start_dt.strftime("%Y-%m-%d"),
                end_ms    = int(_hmm_end_dt.timestamp() * 1000),
                cache_dir = "data",
            )
            if _hmm_df is not None and len(_hmm_df) >= 300:
                _hmm_prices   = _hmm_df["close"].values.astype(np.float64)
                _hmm_log_rets = np.diff(
                    np.log(np.where(_hmm_prices > 0, _hmm_prices, 1.0))
                )
                hmm_detector = HMMRegimeDetector()
                hmm_detector.fit(_hmm_prices, _hmm_log_rets)
                print(f"[HMM] Fitted on {len(_hmm_prices)} bars")
                print(hmm_detector.describe())
            else:
                print("[HMM] ⚠ Not enough pre-test data — HMM gate disabled")
        except Exception as _e:
            print(f"[HMM] ⚠ Fitting failed ({_e}) — HMM gate disabled")

    # ── 2.7 Fetch Binance real taker data for CVD ──────────────────────────
    _bt_taker_cache = f"data/binance_taker_{args.symbol}_{args.timeframe}_{start_date_str.replace('-','')}_{end_date_str.replace('-','')}.csv"
    try:
        df_taker = fetch_binance_taker_history(
            symbol=args.symbol, interval=args.timeframe,
            start_date=start_date_str, end_date=end_date_str,
            cache_path=_bt_taker_cache, verbose=True)
        if not df_taker.empty:
            df_raw = df_raw.merge(df_taker[["ts", "taker_buy_volume"]], on="ts", how="left")
            df_raw["taker_buy_volume"] = df_raw["taker_buy_volume"].fillna(0.0)
            print(f"  [CVD] Real taker merged: {(df_raw['taker_buy_volume']>0).sum()}/{len(df_raw)} bars")
    except Exception as _e:
        print(f"  [CVD] Binance fetch failed ({_e}) — OHLCV fallback")

    # ── 2.8 Fetch Funding Rate + Open Interest (matches training) ──────────────
    _start_ms = int(datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    _end_ms   = int(end_dt.timestamp() * 1000)
    try:
        df_funding = dc.fetch_funding_history(args.symbol, _start_ms, _end_ms, cache_dir="data")
        if not df_funding.empty:
            df_funding["ts"] = (pd.to_datetime(df_funding["ts_ms"], unit="ms", utc=True)
                                .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            df_raw = df_raw.merge(df_funding[["ts", "funding_rate"]], on="ts", how="left")
            df_raw["funding_rate"] = df_raw["funding_rate"].ffill().fillna(0.0)
            print(f"  [FR] Funding rate merged: {(df_raw['funding_rate']!=0).sum()}/{len(df_raw)} bars")
    except Exception as _e:
        print(f"  [FR] Funding rate fetch failed ({_e}) — zeros fallback")

    try:
        df_oi = dc.fetch_open_interest_history(args.symbol, _start_ms, _end_ms, interval="1h", cache_dir="data")
        if not df_oi.empty:
            df_oi["ts"] = (pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
                           .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            df_raw = df_raw.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
            df_raw["open_interest"] = df_raw["open_interest"].ffill().fillna(0.0)
            print(f"  [OI] Open interest merged: {(df_raw['open_interest']!=0).sum()}/{len(df_raw)} bars")
    except Exception as _e:
        print(f"  [OI] Open interest fetch failed ({_e}) — zeros fallback")

    # ── 3. Build V4 features (28-dim, matches training) ──
    tag = f"{args.symbol}_{args.timeframe}_{start_date_str}_{end_date_str}_v4cvd"
    cache_file = f"data/feat_cache_bt_{tag}.npy"
    features = generate_and_cache_features_v4(df_raw, cache_file)

    prices  = df_raw["close"].values.astype(float)
    highs   = df_raw["high"].values.astype(float)
    lows    = df_raw["low"].values.astype(float)
    has_open = "open" in df_raw.columns
    opens   = df_raw["open"].values.astype(float) if has_open else prices  # fallback

    # Fix #7: True Wilder ATR (replaces H-L approximation)
    atr_raw  = compute_true_atr(highs, lows, prices, period=14)
    atr_norm = atr_raw / np.where(prices > 0, prices, 1.0)

    has_ts     = "ts" in df_raw.columns
    timestamps = df_raw["ts"].values if has_ts else np.arange(len(df_raw))

    # ── 4. Simulation ──
    window_size   = 20
    warmup        = 120

    # ── Regime Gate 사전 계산 ────────────────────────────────────────────────
    # Hurst R/S 추정 + 볼 퍼센타일: 모델과 완전 분리된 외부 필터
    def _hurst_rs(log_rets: np.ndarray) -> float:
        """R/S Analysis 기반 Hurst 지수 추정 (n>=10 필요)."""
        n = len(log_rets)
        if n < 10:
            return 0.5
        mean_r  = log_rets.mean()
        cumdev  = np.cumsum(log_rets - mean_r)
        R       = cumdev.max() - cumdev.min()
        S       = log_rets.std() + 1e-10
        return float(np.log(R / S) / np.log(n))

    # ── EMA200 Regime Gate 사전 계산 ────────────────────────────────────────
    # close > EMA200*(1+band) → BULL(SHORT차단), close < EMA200*(1-band) → BEAR(LONG차단)
    _ema200 = pd.Series(prices).ewm(span=200, adjust=False).mean().values

    # 30d 롤링 일간 변동성 히스토리 (퍼센타일 계산용)
    _log_rets_all = np.diff(np.log(np.where(prices > 0, prices, 1.0)))
    _W30 = 30 * 96   # 30d worth of bars
    _vol_hist = np.array([
        _log_rets_all[max(0, j - _W30):j].std()
        for j in range(_W30, len(_log_rets_all) + 1)
    ])
    _vol_hist_sorted = np.sort(_vol_hist)

    def _vol_percentile(vol_val: float) -> float:
        idx = np.searchsorted(_vol_hist_sorted, vol_val)
        return float(idx / max(len(_vol_hist_sorted), 1))

    regime_gate    = getattr(args, "regime_gate", False)
    hurst_min      = getattr(args, "hurst_min",   0.50)
    vol_pct_max    = getattr(args, "vol_pct_max",  0.85)
    regime_blocked = 0   # counter for diagnostics

    cash     = args.capital
    position = 0          # 0=flat, 1=long, -1=short
    entry_price    = 0.0
    entry_bar      = 0
    entry_notional = 0.0  # Fix #2/#3: track notional at entry
    trade_tp       = 0.0  # per-trade ATR-based TP fraction
    trade_sl       = 0.0  # per-trade ATR-based SL fraction
    trade_atr_frac  = 0.0  # ATR/price at entry (for trailing stop)
    trail_watermark = 0.0  # ratchet-only watermark (best favorable price seen)
    trail_active    = False  # True once watermark_gain >= trail_after * ATR

    equity_history = []
    trade_records  = []
    observe_bars   = 0
    peak_equity    = args.capital

    # ── CR Diagnostics pre-scan ──────────────────────────────────────────
    if args.cr_diagnostics and config.use_cr_filter and agent.cr_filter is not None:
        n_sample  = min(100, len(df_raw) - warmup)
        snr_vals  = []
        hurst_vals = []
        purity_vals = []
        blocks = {"hurst": 0, "snr": 0, "purity": 0, "pass": 0}
        for ii in range(warmup, warmup + n_sample):
            s_idx = max(0, ii - window_size + 1)
            wf    = features[s_idx: ii + 1]
            if len(wf) < window_size:
                pad = np.zeros((window_size - len(wf), 23), dtype=np.float32)
                wf  = np.vstack([pad, wf])
            xt = torch.from_numpy(wf).float().unsqueeze(0).to(device)
            log_rets = xt[0, :, 0].cpu().float().numpy()
            h_val    = float(agent.hurst_est._hurst_single(xt[0, :, 0].cpu().float()).item())
            mu_hat   = np.mean(log_rets)
            sigma_hat = np.std(log_rets) + 1e-10
            snr_val  = abs(mu_hat) * math.sqrt(len(log_rets)) / sigma_hat
            snr_vals.append(snr_val)
            hurst_vals.append(h_val)
            h_ok = h_val   > config.cr_hurst_min
            s_ok = snr_val > config.cr_snr_min
            if not h_ok: blocks["hurst"] += 1
            if not s_ok: blocks["snr"]   += 1
            if h_ok and s_ok: blocks["pass"] += 1
        print(f"\n[CR Diagnostics] Sample {n_sample} bars post-warmup:")
        print(f"  Hurst  : min={min(hurst_vals):.3f}  mean={np.mean(hurst_vals):.3f}  "
              f"max={max(hurst_vals):.3f}  (need>{config.cr_hurst_min})")
        print(f"  SNR    : min={min(snr_vals):.3f}  mean={np.mean(snr_vals):.3f}  "
              f"max={max(snr_vals):.3f}  (need>{config.cr_snr_min})")
        print(f"  Blocks : hurst={blocks['hurst']}  snr={blocks['snr']}  "
              f"pass(H+SNR both ok)={blocks['pass']}/{n_sample}")
        print()

    t0 = time.time()
    print(f"[sim] Running {len(df_raw) - warmup} bars ...")

    for i in tqdm(range(warmup, len(df_raw)), desc="Backtesting", ascii=True):
        current_price = prices[i]

        # Build input window [1, window_size, 27]
        start_idx   = max(0, i - window_size + 1)
        window_feat = features[start_idx: i + 1]
        if len(window_feat) < window_size:
            pad         = np.zeros((window_size - len(window_feat), 23), dtype=np.float32)
            window_feat = np.vstack([pad, window_feat])
        x_tensor = torch.from_numpy(window_feat).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action, confidence, probs = agent.select_action(
                x_tensor, atr_norm=float(atr_norm[i]), mode="greedy"
            )

        # ── Flat: look for entry ──
        if position == 0:
            observe_bars += 1
            # ── Regime Gate (모델과 완전 분리, 가중치 변경 없음) ───────────────
            _regime_ok = True
            # ── Old Hurst+vol gate ─────────────────────────────────────────
            if regime_gate and action in (1, 2):
                _lr200 = _log_rets_all[max(0, i - 201):i]
                _H     = _hurst_rs(_lr200)
                _vol   = _lr200[-20:].std() if len(_lr200) >= 20 else 0.0
                _vpct  = _vol_percentile(_vol)
                if _H < hurst_min or _vpct > vol_pct_max:
                    _regime_ok = False
                    regime_blocked += 1
            # ── EMA200 Regime Gate ─────────────────────────────────────────
            if getattr(args, "ema_gate", False) and _regime_ok and action in (1, 2):
                _band   = getattr(args, "ema_band", 0.02)
                _e200   = _ema200[i]
                _cp     = prices[i]
                if _cp > _e200 * (1 + _band):       # BULL → SHORT 차단
                    if action == 2:
                        _regime_ok = False
                        regime_blocked += 1
                elif _cp < _e200 * (1 - _band):     # BEAR → LONG 차단
                    if action == 1:
                        _regime_ok = False
                        regime_blocked += 1
                # else: RANGE → 양방향 허용
            # ── HMM 3-state gate (replaces Hurst when --hmm-gate set) ──────
            if hmm_detector is not None and _regime_ok and action in (1, 2):
                _lr200 = _log_rets_all[max(0, i - 200):i]        # up to 200 log-returns
                _pr201 = prices[max(0, i - 200):i + 1]           # up to 201 prices (len = len(_lr200)+1)
                _hmm_ok, _hmm_state = hmm_detector.is_action_allowed(action, _pr201, _lr200)
                if not _hmm_ok:
                    _regime_ok = False
                    regime_blocked += 1
            # ── Hurst Adaptive Physics Gate (hard structural constraint) ────
            # 물리학적 근거:
            #   H > hurst_trend → 지속성(persistence) 구간 → 모멘텀 방향과 신호 일치 필수
            #   H < hurst_mr   → 평균회귀(anti-persistence) → 최근 방향의 반대 신호만 허용
            #   0.45 ≤ H ≤ 0.55 → 랜덤워크(Efficient Market) → edge 없음 → 진입 차단
            if getattr(args, "hurst_adaptive", False) and _regime_ok and action in (1, 2):
                _hw   = getattr(args, "hurst_window", 200)
                _dw   = getattr(args, "hurst_dir_window", 20)
                _ht   = getattr(args, "hurst_trend", 0.55)
                _hmr  = getattr(args, "hurst_mr", 0.45)
                _lr_h = _log_rets_all[max(0, i - _hw):i]
                _H    = _hurst_rs(_lr_h) if len(_lr_h) >= 10 else 0.5
                # 최근 _dw봉의 가격 방향 (양수=상승, 음수=하락)
                _dir_ret = prices[i] / prices[max(0, i - _dw)] - 1.0
                if _H > _ht:
                    # 추세 구간: 신호가 최근 방향과 일치해야 함
                    if action == 1 and _dir_ret < 0:   # LONG인데 최근 하락
                        _regime_ok = False
                        regime_blocked += 1
                    elif action == 2 and _dir_ret > 0:  # SHORT인데 최근 상승
                        _regime_ok = False
                        regime_blocked += 1
                elif _H < _hmr:
                    # 평균회귀 구간: 신호가 최근 방향의 반대여야 함
                    if action == 1 and _dir_ret > 0:   # LONG인데 최근 상승 (과매수)
                        _regime_ok = False
                        regime_blocked += 1
                    elif action == 2 and _dir_ret < 0:  # SHORT인데 최근 하락 (과매도)
                        _regime_ok = False
                        regime_blocked += 1
                else:
                    # 랜덤워크 구간 (0.45 ≤ H ≤ 0.55): edge 없음 → 전부 차단
                    _regime_ok = False
                    regime_blocked += 1
            if action in (1, 2) and confidence >= args.confidence and _regime_ok:
                position       = 1 if action == 1 else -1
                entry_price    = current_price
                entry_bar      = i

                # ATR-based TP/SL (학습 레이블과 동일한 기준)
                if use_atr_barrier:
                    _atr_frac  = float(atr_norm[i])   # ATR / price
                    trade_tp   = args.tp_mult * _atr_frac
                    trade_sl   = args.sl_mult * _atr_frac
                else:
                    trade_tp   = args.tp
                    trade_sl   = args.sl

                # Fix #2/#3: fee and sizing based on current equity × pos_frac
                # pos_frac=0.5 → 잔고 50%만 사용 (MDD 감소, 실효 레버리지 = leverage × pos_frac)
                entry_notional = cash * args.pos_frac * args.leverage
                fee_entry      = entry_notional * args.eta_maker
                cash -= fee_entry
                # Trailing stop state (reset at each new entry)
                trade_atr_frac  = trade_sl / max(args.sl_mult, 1e-8)
                trail_watermark = entry_price
                trail_active    = False
        else:
            # ── In position: check exit conditions ──
            hold_bars = i - entry_bar

            if position == 1:
                best_pct  = (highs[i]  - entry_price) / entry_price
                worst_pct = (lows[i]   - entry_price) / entry_price
            else:
                best_pct  = (entry_price - lows[i])   / entry_price
                worst_pct = (entry_price - highs[i])  / entry_price

            close_pct = position * (current_price - entry_price) / entry_price

            exit_type    = None
            exit_pnl_pct = 0.0

            # ── Watermark 갱신 (ratchet-only) ─────────────────────────────
            if position == 1:
                if highs[i] > trail_watermark:
                    trail_watermark = highs[i]
                watermark_gain = (trail_watermark - entry_price) / entry_price
            else:
                if lows[i] < trail_watermark:
                    trail_watermark = lows[i]
                watermark_gain = (entry_price - trail_watermark) / entry_price

            # Trail 활성화 체크
            if not trail_active and args.trail_after > 0:
                if watermark_gain >= args.trail_after * trade_atr_frac:
                    trail_active = True

            # Trail SL 가격 계산 (활성 시)
            trail_sl_exit_pnl = None
            if trail_active:
                _tdist = args.trail_dist * trade_atr_frac
                if position == 1:
                    _tsl_price = trail_watermark * (1 - _tdist)
                    if lows[i] <= _tsl_price:
                        trail_sl_exit_pnl = (_tsl_price - entry_price) / entry_price
                else:
                    _tsl_price = trail_watermark * (1 + _tdist)
                    if highs[i] >= _tsl_price:
                        trail_sl_exit_pnl = (entry_price - _tsl_price) / entry_price

            # Fix #6: same-bar SL+TP → use bar direction heuristic instead of always SL
            # Exit priority: both_hit → TP → TRAIL_SL → SL → MAX_HOLD → REGIME_EXIT
            both_hit = (worst_pct <= -trade_sl) and (best_pct >= trade_tp)
            if both_hit:
                # Bar direction: if close > open → bullish bar → TP hit first for longs
                bar_bullish = float(closes_scalar := current_price) >= float(opens[i])
                if position == 1:
                    exit_type    = "TP" if bar_bullish else "SL"
                    exit_pnl_pct = trade_tp if bar_bullish else -trade_sl
                else:
                    exit_type    = "TP" if not bar_bullish else "SL"
                    exit_pnl_pct = trade_tp if not bar_bullish else -trade_sl
            elif best_pct >= trade_tp:
                exit_type    = "TP"
                exit_pnl_pct = trade_tp
            elif trail_sl_exit_pnl is not None:
                exit_type    = "TRAIL_SL"
                exit_pnl_pct = trail_sl_exit_pnl
            elif worst_pct <= -trade_sl:
                exit_type    = "SL"
                exit_pnl_pct = -trade_sl
            elif args.max_hold > 0 and hold_bars >= args.max_hold:
                exit_type    = "MAX_HOLD"
                exit_pnl_pct = close_pct

            # Regime-flip early exit: same model, opposite direction signal
            if exit_type is None and args.regime_exit_conf > 0:
                opposite = 2 if position == 1 else 1  # LONG→expects SHORT(2), SHORT→expects LONG(1)
                if action == opposite and confidence >= args.regime_exit_conf:
                    exit_type    = "REGIME_EXIT"
                    exit_pnl_pct = close_pct  # exit at current bar close

            if exit_type:
                # Fix #2/#3: PnL based on entry_notional (current equity at entry)
                realized    = entry_notional * exit_pnl_pct
                # Fix #11: exits use taker fee rate
                fee_exit    = entry_notional * args.eta_taker
                net_pnl_usd = realized - fee_exit
                cash       += net_pnl_usd

                # Fix #4/#5: gamma^t discounting and J(tau) tracking
                gamma_pnl = net_pnl_usd * (args.gamma ** hold_bars)

                # J(tau) = gamma-discounted PnL + R_agile bonus for strategic exits
                r_agile   = args.r_agile if exit_type == "STRATEGIC" else 0.0
                j_tau     = gamma_pnl + r_agile

                trade_records.append({
                    "entry_bar":      entry_bar,
                    "exit_bar":       i,
                    "entry_ts":       timestamps[entry_bar] if has_ts else entry_bar,
                    "exit_ts":        timestamps[i]         if has_ts else i,
                    "side":           "LONG" if position == 1 else "SHORT",
                    "entry_price":    entry_price,
                    "exit_price":     current_price,
                    "pnl_pct":        exit_pnl_pct,
                    "pnl_usd":        net_pnl_usd,
                    "gamma_pnl_usd":  gamma_pnl,       # Fix #4
                    "j_tau":          j_tau,            # Fix #5
                    "hold_bars":      hold_bars,
                    "exit_type":      exit_type,
                    "confidence":     confidence,
                    "entry_notional": entry_notional,
                })
                position       = 0
                entry_notional = 0.0

        equity_history.append(cash)

    # ── Force-close open position at last bar ──
    if position != 0:
        final_price  = prices[-1]
        close_pct    = position * (final_price - entry_price) / entry_price
        # Fix #2/#3: use entry_notional (already set at entry)
        realized     = entry_notional * close_pct
        fee_exit     = entry_notional * args.eta_taker
        net_pnl_usd  = realized - fee_exit
        cash        += net_pnl_usd
        hold_bars    = len(df_raw) - 1 - entry_bar
        gamma_pnl    = net_pnl_usd * (args.gamma ** hold_bars)

        trade_records.append({
            "entry_bar":      entry_bar,
            "exit_bar":       len(df_raw) - 1,
            "entry_ts":       timestamps[entry_bar] if has_ts else entry_bar,
            "exit_ts":        timestamps[-1]         if has_ts else len(df_raw) - 1,
            "side":           "LONG" if position == 1 else "SHORT",
            "entry_price":    entry_price,
            "exit_price":     final_price,
            "pnl_pct":        close_pct,
            "pnl_usd":        net_pnl_usd,
            "gamma_pnl_usd":  gamma_pnl,
            "j_tau":          gamma_pnl,
            "hold_bars":      hold_bars,
            "exit_type":      "END_OF_DATA",
            "confidence":     0.0,
            "entry_notional": entry_notional,
        })
        equity_history[-1] = cash

    elapsed = time.time() - t0

    # ── 5. Compute metrics ──
    eq_arr       = np.array(equity_history)
    total_trades = len(trade_records)
    df_trades    = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

    total_pnl = cash - args.capital
    roi       = (total_pnl / args.capital) * 100
    mdd       = compute_max_drawdown(eq_arr) * 100
    sharpe    = compute_sharpe(eq_arr, periods_per_year=_bars_per_year(args.timeframe))

    tp_count    = len(df_trades[df_trades["exit_type"] == "TP"])         if total_trades else 0
    sl_count    = len(df_trades[df_trades["exit_type"] == "SL"])         if total_trades else 0
    trail_count = len(df_trades[df_trades["exit_type"] == "TRAIL_SL"])   if total_trades else 0
    sc_count    = len(df_trades[df_trades["exit_type"] == "STRATEGIC"])  if total_trades else 0
    mh_count    = len(df_trades[df_trades["exit_type"] == "MAX_HOLD"])   if total_trades else 0

    # Fix #9: break out win rate by exit type
    if total_trades:
        wins_tp    = len(df_trades[(df_trades["exit_type"] == "TP")       & (df_trades["pnl_usd"] > 0)])
        wins_trail = len(df_trades[(df_trades["exit_type"] == "TRAIL_SL") & (df_trades["pnl_usd"] > 0)])
        wins_sc    = len(df_trades[(df_trades["exit_type"] == "STRATEGIC")& (df_trades["pnl_usd"] > 0)])
        wins_mh    = len(df_trades[(df_trades["exit_type"] == "MAX_HOLD") & (df_trades["pnl_usd"] > 0)])
        wins_all   = len(df_trades[df_trades["pnl_usd"] > 0])
    else:
        wins_tp = wins_trail = wins_sc = wins_mh = wins_all = 0

    win_rate  = wins_all / max(1, total_trades) * 100
    avg_hold  = float(df_trades["hold_bars"].mean()) if total_trades else 0.0

    # Fix #4/#5: J(tau) aggregate
    j_tau_total = float(df_trades["j_tau"].sum())          if total_trades else 0.0
    gamma_roi   = float(df_trades["gamma_pnl_usd"].sum())  if total_trades else 0.0

    sniper_score = roi / max(1, total_trades)
    obs_rate     = observe_bars / max(1, len(equity_history)) * 100

    if total_trades:
        gross_win  = float(df_trades[df_trades["pnl_usd"] > 0]["pnl_usd"].sum())
        gross_loss = abs(float(df_trades[df_trades["pnl_usd"] <= 0]["pnl_usd"].sum()))
        profit_factor = gross_win / max(gross_loss, 1e-8)
    else:
        profit_factor = 0.0

    # ── 6. Print report ──
    print()
    print("=" * 68)
    print("  QUANTUM TRADING V2 [SNIPER] BACKTEST RESULT (Improved)")
    print("=" * 68)
    print(f"  Period          : {start_date_str} ~ {end_date_str}")
    print(f"  Bars            : {len(df_raw)} ({args.timeframe})")
    print(f"  Device          : {device}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print("-" * 68)
    print(f"  Initial Capital : ${args.capital:.2f}")
    print(f"  Final Equity    : ${cash:.2f}")
    print(f"  Net PnL         : ${total_pnl:+.2f} ({roi:+.2f}%)")
    print(f"  Max Drawdown    : {mdd:.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:.3f}")
    print(f"  Profit Factor   : {profit_factor:.2f}")
    print("-" * 68)
    # Fix #4/#5: J(tau) and gamma-discounted metrics
    print(f"  J(tau) Score    : {j_tau_total:+.4f}  (gamma-discounted PnL + R_agile)")
    print(f"  Gamma PnL ($)   : ${gamma_roi:+.2f}  (time-discounted, gamma={args.gamma})")
    print("-" * 68)
    print(f"  Total Trades    : {total_trades}")
    print(f"    TP / SL       : {tp_count} / {sl_count}")
    print(f"    Trail SL      : {trail_count}  ({trail_count/max(1,total_trades)*100:.1f}%)")
    print(f"    Strategic     : {sc_count}")
    print(f"    Max Hold      : {mh_count}")
    # Fix #9: win rate breakdown
    print(f"  Win Rate (all)  : {win_rate:.1f}%  ({wins_all}/{total_trades})")
    if total_trades:
        tp_win_rate    = wins_tp    / max(1, tp_count)    * 100
        trail_win_rate = wins_trail / max(1, trail_count) * 100
        sc_win_rate    = wins_sc    / max(1, sc_count)    * 100
        mh_win_rate    = wins_mh    / max(1, mh_count)    * 100
        print(f"    TP exits      : {tp_win_rate:.1f}%  ({wins_tp}/{tp_count})")
        if trail_count:
            print(f"    Trail SL      : {trail_win_rate:.1f}%  ({wins_trail}/{trail_count})")
        print(f"    Strategic     : {sc_win_rate:.1f}%  ({wins_sc}/{sc_count})")
        print(f"    Max Hold      : {mh_win_rate:.1f}%  ({wins_mh}/{mh_count})")
    print(f"  Avg Hold        : {avg_hold:.1f} bars")
    print(f"  Observe Rate    : {obs_rate:.1f}%")
    if regime_gate:
        print(f"  Regime Blocked  : {regime_blocked} signals (Hurst<{hurst_min} or vol>{vol_pct_max*100:.0f}%ile)")
    if hmm_detector is not None:
        print(f"  HMM Blocked     : {regime_blocked} signals (CHOPPY or counter-trend)")
    if getattr(args, "ema_gate", False):
        _band = getattr(args, "ema_band", 0.02)
        print(f"  EMA Gate        : {regime_blocked} signals blocked  "
              f"(BULL→no SHORT, BEAR→no LONG, band={_band*100:.0f}%)")
    if getattr(args, "hurst_adaptive", False):
        print(f"  Hurst Adaptive  : {regime_blocked} signals blocked  "
              f"(H>{getattr(args,'hurst_trend',0.55)}→trend-align, "
              f"H<{getattr(args,'hurst_mr',0.45)}→mean-rev, else→block)")
    print(f"  Sniper Score    : {sniper_score:.4f} (ROI / trades)")
    print("=" * 68)

    # ── 7. Export CSVs ──
    os.makedirs(args.output_dir, exist_ok=True)
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    eq_path = os.path.join(args.output_dir, f"equity_curve_{ts_tag}.csv")
    eq_df   = pd.DataFrame({"bar": range(warmup, warmup + len(eq_arr)), "equity": eq_arr})
    eq_df.to_csv(eq_path, index=False, encoding="utf-8")
    print(f"[export] Equity curve -> {eq_path}")

    if total_trades:
        trades_path = os.path.join(args.output_dir, f"trades_{ts_tag}.csv")
        df_trades.to_csv(trades_path, index=False, encoding="utf-8")
        print(f"[export] Trades ({total_trades}) -> {trades_path}")

    # ── 8. Visualization ──
    if args.visualize:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                 gridspec_kw={"height_ratios": [3, 1, 1]})

        ax1 = axes[0]
        ax1.plot(eq_arr, color="#00cc66", linewidth=0.8, label="Equity")
        ax1.axhline(y=args.capital, color="gray", linestyle="--", alpha=0.5, label="Initial")
        peak = np.maximum.accumulate(eq_arr)
        ax1.fill_between(range(len(eq_arr)), eq_arr, peak,
                         alpha=0.15, color="red", label="Drawdown")
        ax1.set_title(
            f"Quantum V2 Backtest: {args.symbol} {args.timeframe}  |  "
            f"ROI={roi:+.1f}%  Sharpe={sharpe:.2f}  MDD={mdd:.1f}%  "
            f"J(τ)={j_tau_total:+.2f}"
        )
        ax1.set_ylabel("Equity ($)")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        if total_trades:
            bars   = df_trades["exit_bar"].values - warmup
            pnls   = df_trades["pnl_usd"].values
            colors = ["#00cc66" if p > 0 else "#ff4444" for p in pnls]
            ax2.bar(bars, pnls, color=colors,
                    width=max(1, len(eq_arr) // 200), alpha=0.7)
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax2.set_ylabel("Trade PnL ($)")
        ax2.grid(True, alpha=0.3)

        # Fix #4: Gamma-discounted PnL subplot
        ax3 = axes[2]
        if total_trades:
            g_pnls = df_trades["gamma_pnl_usd"].values
            g_cols = ["#0099ff" if p > 0 else "#ff8800" for p in g_pnls]
            ax3.bar(bars, g_pnls, color=g_cols,
                    width=max(1, len(eq_arr) // 200), alpha=0.7)
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax3.set_ylabel(f"Gamma PnL (γ={args.gamma})")
        ax3.set_xlabel("Bar")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(args.output_dir, f"backtest_chart_{ts_tag}.png")
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"[export] Chart -> {chart_path}")


def _bars_per_year(timeframe: str) -> float:
    minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
               "1h": 60, "2h": 120, "4h": 240, "1d": 1440, "1w": 10080, "1M": 43200}
    m = minutes.get(timeframe, 15)
    return 365.25 * 24 * 60 / m


if __name__ == "__main__":
    run_backtest_v2()
