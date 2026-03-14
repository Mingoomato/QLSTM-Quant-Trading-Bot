"""
backtest_behavioral.py
--------------------------------------------------------------------
Behavioral Alpha Engine — "Exploit Dumb Money"

역사적 패턴 학습 없음. 오직 현재 봉의 시장 미시구조 신호로 진입.
대중의 집단 실수를 포착해서 반대 포지션.

신호:
  1. FR_Z      (idx 18): 펀딩레이트 극단 → 오버레버리지 군중 반대
  2. LIQ_LONG  (idx 26): 롱 청산 폭발 → 낙폭 과대, 반등 기대
  3. LIQ_SHORT (idx 27): 숏 청산 폭발 → 급등 과대, 되돌림 기대
  4. CVD_DIV   (idx 25): 가격-CVD 다이버전스 → 분배/집적 신호

사용:
  python scripts/backtest_behavioral.py --start-date 2025-01-01 --timeframe 1h
  python scripts/backtest_behavioral.py --start-date 2024-01-01 --fr-z-thr 1.5 --liq-z-thr 2.0
  python scripts/backtest_behavioral.py --signals fr,liq --start-date 2025-01-01
--------------------------------------------------------------------
"""

import argparse, os, sys, math, time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

# Quantum model (optional — loaded only with --quantum-gate)
_quantum_agent = None
_quantum_device = None


def _load_quantum_agent(model_path: str, device_str: str = "cpu"):
    """Quantum 모델 로드 (backtest 시작 시 1회)."""
    global _quantum_agent, _quantum_device
    import torch
    from src.models.integrated_agent import build_quantum_agent, AgentConfig
    device = torch.device(device_str)
    agent = build_quantum_agent(
        config=AgentConfig(feature_dim=28, n_eigenvectors=5),
        device=device,
    )
    agent.load_checkpoint(model_path, strict=False)
    agent.eval()
    _quantum_agent = agent
    _quantum_device = device
    print(f"[quantum-gate] 모델 로드: {model_path}  device={device_str}")


def _quantum_p_long(features: np.ndarray, window: int = 20, atr_norm: float = 0.01) -> float:
    """
    마지막 window봉 피처로 p_long 계산.
    features: (N, 28) 전체 피처 배열, 마지막 window봉 사용.
    Returns: p_long (float, 0~1)
    """
    import torch
    if _quantum_agent is None:
        return 0.0
    win = features[-window:]
    if len(win) < window:
        pad = np.zeros((window - len(win), features.shape[1]), dtype=np.float32)
        win = np.vstack([pad, win])
    x = torch.from_numpy(win.astype(np.float32)).unsqueeze(0).to(_quantum_device)
    with torch.no_grad():
        _, _, probs = _quantum_agent.select_action(x, atr_norm=atr_norm, mode="greedy")
    return float(probs[1].item())  # p_long


def _quantum_probs(features: np.ndarray, window: int = 20, atr_norm: float = 0.01) -> tuple[float, float, float]:
    """
    Returns: (p_hold, p_long, p_short)
    """
    import torch
    if _quantum_agent is None:
        return 0.34, 0.33, 0.33
    win = features[-window:]
    if len(win) < window:
        pad = np.zeros((window - len(win), features.shape[1]), dtype=np.float32)
        win = np.vstack([pad, win])
    x = torch.from_numpy(win.astype(np.float32)).unsqueeze(0).to(_quantum_device)
    with torch.no_grad():
        _, _, probs = _quantum_agent.select_action(x, atr_norm=atr_norm, mode="greedy")
    return float(probs[0].item()), float(probs[1].item()), float(probs[2].item())


def _quantum_full_info(features: np.ndarray, window: int = 20, atr_norm: float = 0.01) -> dict:
    """
    Gemini에 전달할 Quantum 모델의 전체 내부 정보 반환.
    Returns dict: p_hold, p_long, p_short, hurst, regime_prob,
                  confidence_threshold, force_hold_cr, force_hold_ep,
                  force_hold_regime, logit_margin, liq_long, liq_short, cvd_delta
    """
    import torch
    # V4 피처 인덱스
    IDX_CVD_DELTA = 23
    IDX_LIQ_LONG  = 26
    IDX_LIQ_SHORT = 27

    default = {
        "p_hold": 0.34, "p_long": 0.33, "p_short": 0.33,
        "hurst": 0.5, "regime_prob": 0.0, "confidence_threshold": 0.5,
        "force_hold_cr": False, "force_hold_ep": False, "force_hold_regime": False,
        "logit_margin": 0.0,
        "cvd_delta": 0.0, "liq_long": 0.0, "liq_short": 0.0,
    }
    if _quantum_agent is None:
        return default

    if features.ndim == 1:
        features = features.reshape(1, -1)

    win = features[-window:]
    if len(win) < window:
        pad = np.zeros((window - len(win), features.shape[1]), dtype=np.float32)
        win = np.vstack([pad, win])

    x = torch.from_numpy(win.astype(np.float32)).to(_quantum_device)
    try:
        info = _quantum_agent.select_action_verbose(x, atr_norm=atr_norm)
    except Exception:
        _, _, probs = _quantum_agent.select_action(x, atr_norm=atr_norm, mode="greedy")
        info = {"p_hold": float(probs[0]), "p_long": float(probs[1]), "p_short": float(probs[2]),
                "hurst": 0.5, "regime_prob": 0.0, "confidence_threshold": 0.5,
                "force_hold_cr": False, "force_hold_ep": False, "force_hold_regime": False,
                "logit_margin": 0.0}

    last_feat = features[-1]
    info["cvd_delta"] = float(last_feat[IDX_CVD_DELTA]) if len(last_feat) > IDX_CVD_DELTA else 0.0
    info["liq_long"]  = float(last_feat[IDX_LIQ_LONG])  if len(last_feat) > IDX_LIQ_LONG  else 0.0
    info["liq_short"] = float(last_feat[IDX_LIQ_SHORT]) if len(last_feat) > IDX_LIQ_SHORT else 0.0
    return info

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.data_client import DataClient
from src.data.bybit_mainnet import BybitMainnetClient, REST_BASE
from src.models.features_v2 import compute_true_atr
from src.models.features_v4 import generate_and_cache_features_v4
from src.data.binance_client import fetch_binance_taker_history
from src.models.oi_profile import precompute_poc_distances

# ── Feature indices (28-dim V4 + 1 synthetic) ────────────────────
# V2[0-15] + Hurst[16] + purity[17] + V4_extra[18-27]
IDX_FR_Z       = 18   # funding_rate_zscore
IDX_BODY       = 19   # candle_body_ratio (NOT OI)
IDX_VOL_Z      = 20   # volume_zscore
IDX_OI_PCT     = 21   # oi_change_pct
IDX_FR_VEL     = 22   # funding_velocity
IDX_CVD_DELTA  = 23   # cvd_delta_zscore
IDX_CVD_TREND  = 24   # cvd_trend_zscore
IDX_CVD_DIV    = 25   # cvd_price_divergence  (-1=price↑CVD↓, +1=price↓CVD↑)
IDX_LIQ_LONG   = 26   # liq_long_zscore  (lower_wick×vol z-score)
IDX_LIQ_SHORT  = 27   # liq_short_zscore (upper_wick×vol z-score)
IDX_OI_POC     = 28   # oi_poc_dist = (price - POC) / ATR  [synthetic, appended]


# ──────────────────────────────────────────────────────────────────
class BehavioralAlphaEngine:
    """
    각 봉에서 행동경제학 신호 계산 → 방향 + 점수 반환.
    seq_len=1: 현재 봉만, 역사 패턴 없음.
    """

    def __init__(self,
                 fr_z_thr:      float = 1.5,   # FR z-score 임계값 (절댓값)
                 liq_z_thr:     float = 1.5,   # 청산 z-score 임계값
                 cvd_div_thr:   float = 0.4,   # CVD 다이버전스 임계값
                 oi_poc_thr:    float = 1.0,   # OI POC 거리 임계값 (ATR 배수)
                 min_score:     float = 1.0,   # 최소 진입 점수 합산
                 active_signals: set  = None,  # {'fr','liq','cvd','oi_poc'} or None=all
                 ):
        self.fr_z_thr      = fr_z_thr
        self.liq_z_thr     = liq_z_thr
        self.cvd_div_thr   = cvd_div_thr
        self.oi_poc_thr    = oi_poc_thr
        self.min_score     = min_score
        self.active        = active_signals or {"fr", "liq", "cvd"}

    def score(self, feat: np.ndarray) -> tuple:
        """
        Returns: (direction: int, score: float, reasons: list)
          direction:  1=LONG, -1=SHORT, 0=NO SIGNAL
        """
        long_score  = 0.0
        short_score = 0.0
        reasons     = []

        fr_z      = float(feat[IDX_FR_Z])
        liq_long  = float(feat[IDX_LIQ_LONG])
        liq_short = float(feat[IDX_LIQ_SHORT])
        cvd_div   = float(feat[IDX_CVD_DIV])

        # ── 신호 1: 펀딩레이트 극단 (오버레버리지 군중 반대) ──────────
        # FR 매우 양수 → 모두 롱 오버레버리지 → 숏 신호
        # FR 매우 음수 → 모두 숏 오버레버리지 → 롱 신호
        if "fr" in self.active:
            if fr_z > self.fr_z_thr:
                s = min((fr_z - self.fr_z_thr) * 0.8 + 0.5, 2.0)
                short_score += s
                reasons.append(f"FR_SHORT(z={fr_z:.2f},+{s:.2f})")
            elif fr_z < -self.fr_z_thr:
                s = min((-fr_z - self.fr_z_thr) * 0.8 + 0.5, 2.0)
                long_score  += s
                reasons.append(f"FR_LONG(z={fr_z:.2f},+{s:.2f})")

        # ── 신호 2: 청산 캐스케이드 역추세 ────────────────────────────
        # 롱 청산 폭발 → 가격 낙폭 과대 → 반등 기대 → LONG
        # 숏 청산 폭발 → 가격 급등 과대 → 되돌림 기대 → SHORT
        if "liq" in self.active:
            if liq_long > self.liq_z_thr:
                s = min((liq_long - self.liq_z_thr) * 0.6 + 0.5, 2.0)
                long_score  += s
                reasons.append(f"LIQ_BOUNCE(z={liq_long:.2f},+{s:.2f})")
            if liq_short > self.liq_z_thr:
                s = min((liq_short - self.liq_z_thr) * 0.6 + 0.5, 2.0)
                short_score += s
                reasons.append(f"LIQ_FADE(z={liq_short:.2f},+{s:.2f})")

        # ── 신호 3: CVD-가격 다이버전스 (분배/집적) ───────────────────
        # cvd_div > 0 → 가격 하락중인데 CVD 매수 우세 → LONG (집적)
        # cvd_div < 0 → 가격 상승중인데 CVD 매도 우세 → SHORT (분배)
        if "cvd" in self.active:
            if cvd_div > self.cvd_div_thr:
                s = min((cvd_div - self.cvd_div_thr) * 1.5 + 0.3, 1.5)
                long_score  += s
                reasons.append(f"CVD_ACCUM(div={cvd_div:.2f},+{s:.2f})")
            elif cvd_div < -self.cvd_div_thr:
                s = min((-cvd_div - self.cvd_div_thr) * 1.5 + 0.3, 1.5)
                short_score += s
                reasons.append(f"CVD_DIST(div={cvd_div:.2f},+{s:.2f})")

        # ── 신호 4: OI POC 거리 (스마트머니 포지션 자석) ──────────────
        # poc_dist = (price - POC) / ATR  (IDX_OI_POC = 28, 합성 피처)
        # poc_dist << 0 → 가격이 POC 아래 → 위로 회귀 기대 → LONG
        # poc_dist >> 0 → 가격이 POC 위   → 아래 회귀 기대 → SHORT
        if "oi_poc" in self.active and len(feat) > IDX_OI_POC:
            poc_dist = float(feat[IDX_OI_POC])
            if poc_dist < -self.oi_poc_thr:
                s = min((-poc_dist - self.oi_poc_thr) * 0.5 + 0.4, 1.5)
                long_score  += s
                reasons.append(f"OI_POC_LONG(dist={poc_dist:.2f},+{s:.2f})")
            elif poc_dist > self.oi_poc_thr:
                s = min((poc_dist - self.oi_poc_thr) * 0.5 + 0.4, 1.5)
                short_score += s
                reasons.append(f"OI_POC_SHORT(dist={poc_dist:.2f},+{s:.2f})")

        # ── 진입 판단 ─────────────────────────────────────────────────
        if long_score >= self.min_score and long_score > short_score:
            return 1, long_score, reasons
        elif short_score >= self.min_score and short_score > long_score:
            return -1, short_score, reasons
        return 0, 0.0, []

    def apply_trend_gate(self, direction: int, price: float, ema_slow: float) -> int:
        """Regime gate: only LONG above EMA, only SHORT below EMA."""
        if ema_slow <= 0:
            return direction   # no gate if EMA not available
        if direction == 1 and price < ema_slow:
            return 0   # block LONG in downtrend
        if direction == -1 and price > ema_slow:
            return 0   # block SHORT in uptrend
        return direction


# ──────────────────────────────────────────────────────────────────
def run_backtest(args):
    t0 = time.time()

    # ── 데이터 수집 ──────────────────────────────────────────────
    dc = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")
    print(f"[data] Source: Bybit Mainnet  REST={REST_BASE}")

    end_dt = datetime.now(timezone.utc)
    if args.end_date:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    start_str = args.start_date or (end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")

    print(f"[data] Fetching {args.symbol} {args.timeframe} {start_str} → {end_str} ...")
    df_raw = dc.fetch_training_history(
        symbol=args.symbol, timeframe=args.timeframe,
        start_date=start_str, end_ms=int(end_dt.timestamp() * 1000),
        cache_dir="data",
    )
    if df_raw is None or df_raw.empty:
        print("[data] ERROR: No data."); return

    print(f"[data] Got {len(df_raw)} bars")

    # ── Funding Rate + Open Interest 병합 (backtest_model_v2.py 방식) ────
    _start_ms = int(datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    _end_ms   = int(end_dt.timestamp() * 1000)
    try:
        df_funding = dc.fetch_funding_history(args.symbol, _start_ms, _end_ms, cache_dir="data")
        if not df_funding.empty:
            df_funding["ts"] = (pd.to_datetime(df_funding["ts_ms"], unit="ms", utc=True)
                                .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            df_raw = df_raw.merge(df_funding[["ts", "funding_rate"]], on="ts", how="left")
            df_raw["funding_rate"] = df_raw["funding_rate"].ffill().fillna(0.0)
            print(f"  [FR] Merged {(df_raw['funding_rate']!=0).sum()}/{len(df_raw)} bars")
        else:
            df_raw["funding_rate"] = 0.0
    except Exception as e:
        print(f"  [FR] Skip: {e}")
        df_raw["funding_rate"] = 0.0

    try:
        df_oi = dc.fetch_open_interest_history(args.symbol, _start_ms, _end_ms, interval="1h", cache_dir="data")
        if not df_oi.empty:
            df_oi["ts"] = (pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
                           .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            df_raw = df_raw.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
            df_raw["open_interest"] = df_raw["open_interest"].ffill().fillna(0.0)
            print(f"  [OI] Merged {(df_raw['open_interest']!=0).sum()}/{len(df_raw)} bars")
        else:
            df_raw["open_interest"] = 0.0
    except Exception as e:
        print(f"  [OI] Skip: {e}")
        df_raw["open_interest"] = 0.0

    # Binance CVD — merge taker_buy_volume for real order-flow CVD
    safe_tf = args.timeframe  # e.g. "1h"
    cache_cvd = f"data/binance_taker_{args.symbol}_{safe_tf}_{start_str.replace('-','')}_{end_str.replace('-','')}.csv"
    try:
        df_cvd = fetch_binance_taker_history(
            symbol=args.symbol, interval=safe_tf,
            start_date=start_str, end_date=end_str,
            cache_path=cache_cvd,
        )
        if df_cvd is not None and not df_cvd.empty and "taker_buy_volume" in df_cvd.columns:
            # Map by timestamp string (both use "YYYY-MM-DD HH:MM:SS" UTC format)
            cvd_map = dict(zip(df_cvd["ts"], df_cvd["taker_buy_volume"]))
            if "ts" in df_raw.columns:
                df_raw["taker_buy_volume"] = df_raw["ts"].map(cvd_map).fillna(0.0)
            else:
                # Positional fallback: assume same bar count & order
                n_cvd = min(len(df_cvd), len(df_raw))
                df_raw["taker_buy_volume"] = 0.0
                df_raw.iloc[:n_cvd, df_raw.columns.get_loc("taker_buy_volume")] = \
                    df_cvd["taker_buy_volume"].values[:n_cvd]
            merged = int((df_raw["taker_buy_volume"] > 0).sum())
            print(f"  [CVD] merged taker_buy_volume: {merged}/{len(df_raw)} bars")
        else:
            df_raw["taker_buy_volume"] = 0.0
            print("  [CVD] no taker_buy_volume — using OHLCV fallback")
    except Exception as e:
        print(f"  [CVD] Skip: {e}")
        df_raw["taker_buy_volume"] = 0.0

    # ── V4 피처 빌드 (28-dim, funding+OI+CVD+liq 포함) ──────────
    # Cache suffix _v3 = includes real CVD + funding_rate + OI
    cache_path = f"data/feat_cache_behavioral_{args.symbol}_{safe_tf}_{start_str}_{end_str}_v4cvd_v3.npy"
    feats = generate_and_cache_features_v4(df_raw, cache_path=cache_path)
    if feats is None or len(feats) < 100:
        print("[features] ERROR: Not enough features."); return
    print(f"[features] Built {len(feats)} × {feats.shape[1]}-dim V4 features")

    # ATR 계산
    closes = df_raw["close"].values.astype(np.float64)
    highs  = df_raw["high"].values.astype(np.float64)
    lows   = df_raw["low"].values.astype(np.float64)
    opens  = df_raw["open"].values.astype(np.float64)
    atr14  = compute_true_atr(highs, lows, closes, period=14)

    # ── OI POC 거리 사전 계산 (oi_poc 신호 활성화 시) ──────────────
    active_set_check = set(args.signals.lower().split(",")) if args.signals else {"fr", "liq", "cvd"}
    poc_dists = None
    if "oi_poc" in active_set_check:
        oi_series = df_raw["open_interest"].values.astype(np.float64) \
                    if "open_interest" in df_raw.columns else np.zeros(len(closes))
        poc_dists = precompute_poc_distances(
            highs, lows, opens, closes, oi_series, atr14,
            window=args.oi_poc_window, n_buckets=20,
        )

    # ── EMA 200 (trend gate용) 사전 계산 ────────────────────────
    ema_period = args.trend_ema
    ema_slow = np.zeros(len(closes), dtype=np.float64)
    if ema_period > 0:
        alpha_e = 2.0 / (ema_period + 1.0)
        ema_slow[0] = closes[0]
        for _i in range(1, len(closes)):
            ema_slow[_i] = alpha_e * closes[_i] + (1 - alpha_e) * ema_slow[_i - 1]

    # ── 신호 엔진 초기화 ─────────────────────────────────────────
    active_set = set(args.signals.lower().split(",")) if args.signals else {"fr", "liq", "cvd"}
    engine = BehavioralAlphaEngine(
        fr_z_thr       = args.fr_z_thr,
        liq_z_thr      = args.liq_z_thr,
        cvd_div_thr    = args.cvd_div_thr,
        oi_poc_thr     = args.oi_poc_thr,
        min_score      = args.min_score,
        active_signals = active_set,
    )

    # ── Quantum Gate 모델 로드 ───────────────────────────────────────
    use_quantum_gate = getattr(args, "quantum_gate", False)
    q_confidence     = getattr(args, "q_confidence", 0.45)
    q_window         = getattr(args, "q_window", 20)
    if use_quantum_gate:
        mp = getattr(args, "model_path", "checkpoints/quantum_v2/agent_best_fold10.pt")
        _load_quantum_agent(mp)

    print(f"\n[config] Signals={active_set}  fr_z_thr={args.fr_z_thr}  "
          f"liq_z_thr={args.liq_z_thr}  cvd_div_thr={args.cvd_div_thr}  "
          f"min_score={args.min_score}")
    q_gate_str = f"Quantum(conf≥{q_confidence}, w={q_window})" if use_quantum_gate else "OFF"
    print(f"[config] QuantumGate={q_gate_str}")
    trend_str = f"EMA{args.trend_ema}" if args.trend_ema > 0 else "OFF"
    dir_str   = "LONG-ONLY" if args.long_only else ("SHORT-ONLY" if args.short_only else "BOTH")
    print(f"[config] TP={args.tp_mult}×ATR  SL={args.sl_mult}×ATR  "
          f"Leverage={args.leverage}x  pos_frac={args.pos_frac}  "
          f"TrendGate={trend_str}  Direction={dir_str}")

    # ── 백테스트 루프 ────────────────────────────────────────────
    warmup     = 50   # 피처 안정화 구간
    N          = len(feats)
    equity     = args.capital
    max_equity = equity
    max_dd     = 0.0
    position   = 0   # 0=flat, 1=long, -1=short
    entry_price = entry_atr = entry_notional = 0.0
    tp_price    = sl_price = 0.0
    trades      = []

    eta_maker  = 0.0002
    eta_taker  = 0.00055
    max_hold   = args.max_hold

    # 신호별 통계
    signal_stats = {}

    for i in tqdm(range(warmup, N), desc="Backtesting"):
        feat   = feats[i]
        price  = float(closes[i])
        hi     = float(highs[i])
        lo     = float(lows[i])
        atr_val = float(atr14[i])
        if atr_val <= 0 or price <= 0:
            continue

        hold_bars = 0

        # ── 포지션 종료 체크 ──────────────────────────────────────
        if position != 0:
            hold_bars = i - entry_bar
            hit_tp    = (position ==  1 and hi >= tp_price) or \
                        (position == -1 and lo <= tp_price)
            hit_sl    = (position ==  1 and lo <= sl_price) or \
                        (position == -1 and hi >= sl_price)

            exit_type = None
            exit_price = price

            if hit_tp and hit_sl:
                # 같은 봉에 둘 다 — bar 방향으로 결정
                bar_up = price >= float(closes[i-1]) if i > 0 else True
                if position == 1:
                    exit_type  = "TP" if bar_up else "SL"
                    exit_price = tp_price if bar_up else sl_price
                else:
                    exit_type  = "TP" if not bar_up else "SL"
                    exit_price = tp_price if not bar_up else sl_price
            elif hit_tp:
                exit_type  = "TP";   exit_price = tp_price
            elif hit_sl:
                exit_type  = "SL";   exit_price = sl_price
            elif hold_bars >= max_hold > 0:
                exit_type  = "MAX_HOLD"

            if exit_type:
                if position == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                fee     = entry_notional * (eta_maker + eta_taker)
                pnl_usd = entry_notional * pnl_pct - fee
                equity += pnl_usd
                max_equity = max(max_equity, equity)
                dd = (max_equity - equity) / max_equity
                max_dd = max(max_dd, dd)

                trades.append({
                    "exit_type": exit_type,
                    "direction": position,
                    "pnl_usd":   pnl_usd,
                    "pnl_pct":   pnl_pct,
                    "hold_bars": hold_bars,
                    "signal_key": entry_signal_key,
                })
                position = 0

        # ── 진입 체크 (포지션 없을 때) ──────────────────────────
        if position == 0 and equity > 0.01:
            # OI POC 거리를 feat에 합성 (IDX_OI_POC = 28)
            if poc_dists is not None:
                feat_ext = np.append(feat, poc_dists[i])
            else:
                feat_ext = feat
            direction, score, reasons = engine.score(feat_ext)

            # Trend regime gate (EMA filter)
            if args.trend_ema > 0:
                direction = engine.apply_trend_gate(direction, price, float(ema_slow[i]))

            # Direction filter
            if args.long_only  and direction == -1: direction = 0
            if args.short_only and direction ==  1: direction = 0

            # ── Quantum Gate (FR 신호 발생 시 quantum p_long 확인) ──
            if direction != 0 and use_quantum_gate:
                atr_norm_val = float(atr14[i]) / max(price, 1e-8)
                feat_window = feats[max(0, i - q_window + 1): i + 1]
                pl = _quantum_p_long(feat_window, window=q_window, atr_norm=atr_norm_val)
                if direction == 1 and pl < q_confidence:
                    direction = 0  # quantum 모델이 확신 없음 → 스킵
                elif direction == -1 and (1.0 - pl) < q_confidence:
                    direction = 0

            if direction != 0:
                margin   = equity * args.pos_frac
                notional = margin * args.leverage
                fee_in   = notional * eta_maker
                if fee_in >= equity:
                    continue

                entry_price    = price
                entry_atr      = atr_val
                entry_notional = notional
                entry_bar      = i

                tp_price = entry_price * (1 + direction * args.tp_mult * atr_val / price)
                sl_price = entry_price * (1 - direction * args.sl_mult * atr_val / price)

                equity  -= fee_in
                position = direction

                # 신호 키 (어떤 신호 조합이 발동했는지)
                entry_signal_key = "+".join(
                    r.split("(")[0] for r in reasons
                )

                # 신호별 카운터
                for r in reasons:
                    k = r.split("(")[0]
                    if k not in signal_stats:
                        signal_stats[k] = {"trades": 0, "wins": 0, "pnl": 0.0}
                    signal_stats[k]["trades"] += 1

    # ── 미결 포지션 청산 ─────────────────────────────────────────
    if position != 0:
        pnl_pct = (closes[-1] - entry_price) / entry_price * position
        fee     = entry_notional * eta_taker
        pnl_usd = entry_notional * pnl_pct - fee
        equity += pnl_usd
        trades.append({"exit_type": "OPEN", "direction": position,
                        "pnl_usd": pnl_usd, "pnl_pct": pnl_pct,
                        "hold_bars": N - entry_bar, "signal_key": entry_signal_key})

    # ── 신호별 통계 업데이트 ─────────────────────────────────────
    for t in trades:
        k = t.get("signal_key", "")
        for part in k.split("+"):
            if part in signal_stats:
                if t["pnl_usd"] > 0:
                    signal_stats[part]["wins"] += 1
                signal_stats[part]["pnl"] += t["pnl_usd"]

    # ── 결과 출력 ────────────────────────────────────────────────
    df = pd.DataFrame(trades)
    total  = len(df)
    wins   = int((df["pnl_usd"] > 0).sum()) if total else 0
    wr     = wins / total if total else 0
    roi    = (equity - args.capital) / args.capital
    pf_num = df[df["pnl_usd"] > 0]["pnl_usd"].sum() if total else 0
    pf_den = abs(df[df["pnl_usd"] < 0]["pnl_usd"].sum()) if total else 1
    pf     = pf_num / pf_den if pf_den > 0 else float("inf")

    tp_cnt = int((df["exit_type"] == "TP").sum()) if total else 0
    sl_cnt = int((df["exit_type"] == "SL").sum()) if total else 0

    print(f"\n{'='*60}")
    print(f"  BEHAVIORAL ALPHA ENGINE — BACKTEST RESULT")
    print(f"{'='*60}")
    print(f"  Period          : {start_str} ~ {end_str}")
    print(f"  Bars            : {N} ({args.timeframe})")
    print(f"  Active Signals  : {sorted(active_set)}")
    print(f"  Elapsed         : {time.time()-t0:.1f}s")
    print(f"{'-'*60}")
    print(f"  Initial Capital : ${args.capital:.2f}")
    print(f"  Final Equity    : ${equity:.2f}")
    print(f"  Net PnL         : ${equity-args.capital:.2f} ({roi*100:+.2f}%)")
    print(f"  Max Drawdown    : {max_dd*100:.2f}%")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"{'-'*60}")
    print(f"  Total Trades    : {total}")
    print(f"    TP / SL       : {tp_cnt} / {sl_cnt}")
    print(f"  Win Rate        : {wr*100:.1f}%  ({wins}/{total})")
    if total:
        print(f"  Avg Hold        : {df['hold_bars'].mean():.1f} bars")
    print(f"{'-'*60}")

    # 신호별 세부 결과
    if signal_stats:
        print(f"  SIGNAL BREAKDOWN:")
        for k, v in sorted(signal_stats.items()):
            t_cnt = v["trades"]
            w_cnt = v["wins"]
            w_r   = w_cnt / t_cnt if t_cnt else 0
            print(f"    {k:<20} trades={t_cnt:3d}  WR={w_r*100:.0f}%  PnL=${v['pnl']:+.2f}")
    print(f"{'='*60}")

    # CSV 저장
    if total:
        os.makedirs("reports", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"reports/behavioral_{ts}.csv", index=False)
        print(f"[export] {total} trades → reports/behavioral_{ts}.csv")


# ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Behavioral Alpha Backtest")
    p.add_argument("--symbol",     default="BTCUSDT")
    p.add_argument("--timeframe",  default="1h")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date",   default=None)
    p.add_argument("--days",       type=int,   default=365)
    p.add_argument("--capital",    type=float, default=10.0)
    p.add_argument("--leverage",   type=float, default=5.0)
    p.add_argument("--pos-frac",   type=float, default=0.5)
    p.add_argument("--tp-mult",    type=float, default=3.0,
                   help="TP = tp_mult × ATR")
    p.add_argument("--sl-mult",    type=float, default=1.0,
                   help="SL = sl_mult × ATR")
    p.add_argument("--max-hold",   type=int,   default=96,
                   help="최대 보유 봉 수 (0=무제한)")
    # 신호 파라미터
    p.add_argument("--signals",    default=None,
                   help="활성 신호: fr,liq,cvd (기본: 전체)")
    p.add_argument("--fr-z-thr",   type=float, default=1.5,
                   help="펀딩레이트 z-score 임계값")
    p.add_argument("--liq-z-thr",  type=float, default=1.5,
                   help="청산 z-score 임계값")
    p.add_argument("--cvd-div-thr",type=float, default=0.4,
                   help="CVD 다이버전스 임계값")
    p.add_argument("--oi-poc-thr", type=float, default=1.0,
                   help="OI POC 거리 임계값 (ATR 배수, 기본: 1.0)")
    p.add_argument("--oi-poc-window", type=int, default=100,
                   help="OI POC 롤링 윈도우 봉 수 (기본: 100봉 = 4일@1h)")
    p.add_argument("--min-score",  type=float, default=1.0,
                   help="최소 진입 점수 합산")
    p.add_argument("--long-only",  action="store_true",
                   help="LONG 신호만 진입 (SHORT 무시)")
    p.add_argument("--short-only", action="store_true",
                   help="SHORT 신호만 진입 (LONG 무시)")
    p.add_argument("--trend-ema",  type=int, default=0,
                   help="추세 레짐 게이트: EMA period (0=비활성). LONG=가격>EMA, SHORT=가격<EMA")
    # Quantum Gate
    p.add_argument("--quantum-gate",  action="store_true",
                   help="FR 신호 발동 시 quantum 모델 p_long ≥ q-confidence 추가 확인")
    p.add_argument("--model-path",  default="checkpoints/quantum_v2/agent_best_fold10.pt",
                   help="Quantum 모델 체크포인트 경로")
    p.add_argument("--q-confidence", type=float, default=0.45,
                   help="Quantum gate 최소 p_long (기본 0.45)")
    p.add_argument("--q-window",     type=int,   default=20,
                   help="Quantum inference 입력 윈도우 봉 수 (기본 20)")
    args = p.parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()
