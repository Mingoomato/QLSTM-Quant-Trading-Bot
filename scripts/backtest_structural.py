"""
backtest_structural.py
────────────────────────────────────────────────────────────────────────────
Structural Mechanism Backtest

원칙: 통계 패턴 없음. 오직 인과적 시장 메커니즘으로만 진입.
     일반화 가능한 알파만 사용.

메커니즘:
  1. FR_SQUEEZE : 펀딩레이트 극단 → 오버레버리지 청산 강제
  2. OI_DIV     : OI-가격 방향 불일치 → 역방향 포지션 누적 (스퀴즈 준비)
  3. LIQ_EXHAUST: 강제청산 폭발 후 연료 소진 → 방향 전환
  4. CVD_ABSORB : CVD-가격 다이버전스 → 스마트머니 흡수/분배

레짐 필터:
  EMA200: 추세 방향 필터 (상승추세에서 LONG, 하락추세에서 SHORT)
  vol_regime: 고변동성 레짐에서 청산 효과 증폭

사용:
  # 전체 기간 일반화 테스트 (2019-2026)
  python scripts/backtest_structural.py --start-date 2019-01-01 --timeframe 1h

  # 신호 조합 선택
  python scripts/backtest_structural.py --signals fr --start-date 2023-01-01
  python scripts/backtest_structural.py --signals fr,oi --start-date 2023-01-01
  python scripts/backtest_structural.py --signals fr,liq --long-only --start-date 2023-01-01
  python scripts/backtest_structural.py --signals fr,oi,liq,cvd --start-date 2019-01-01

  # 파라미터 조정
  python scripts/backtest_structural.py --fr-thr 2.5 --oi-div-thr 0.3 --start-date 2023-01-01
────────────────────────────────────────────────────────────────────────────
"""

import argparse, os, sys, time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_client import DataClient
REST_BASE = "https://api.bybit.com"
from src.data.binance_client import fetch_binance_taker_history
from src.models.features_structural import build_structural_features, FEAT_NAMES, FEAT_DIM


# ── Long/Short Ratio fetch (real Bybit data, replaces broken /v5/market/liquidation) ──
def fetch_ls_ratio_history(symbol: str, start_ms: int, end_ms: int,
                           cache_dir: str = "data") -> pd.DataFrame:
    """
    Fetch historical Long/Short Ratio from Bybit /v5/market/account-ratio.
    Returns DataFrame[ts_ms, buy_ratio, sell_ratio] sorted ascending.

    Used as real-data proxy for liq_long_usd / liq_short_usd:
      liq_long_usd  = buy_ratio  × 1e6  (high = many longs = liquidation concentration)
      liq_short_usd = sell_ratio × 1e6  (high = many shorts = liquidation concentration)

    NOTE: /v5/market/liquidation returns 404 — Bybit has no REST liquidation history endpoint.
          This L/S ratio is the closest available REAL structural proxy (real position imbalance data).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"ls_ratio_{symbol}_{start_ms}_{end_ms}.csv")
    if os.path.exists(cache_path):
        try:
            df_c = pd.read_csv(cache_path)
            if not df_c.empty and "ts_ms" in df_c.columns:
                print(f"  [L/S] Loaded cache: {len(df_c)} records")
                return df_c
        except Exception:
            pass

    all_records = []
    current_end = end_ms

    for _ in range(500):  # max 500 pages × 500 bars = 250k bars
        params = {
            "category": "linear",
            "symbol": symbol,
            "period": "1h",
            "limit": 500,
            "endTime": int(current_end),
        }
        url = f"{REST_BASE}/v5/market/account-ratio?{urlencode(params)}"
        try:
            req = Request(url, headers={"User-Agent": "TerminalQuantSuite/1.0"})
            with urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("retCode", 0) != 0:
                print(f"  [L/S] API error: {payload.get('retMsg')}")
                break
            rows = payload.get("result", {}).get("list", [])
            if not rows:
                break
            for r in rows:
                ts_ms = int(r.get("timestamp", 0))
                buy_r = float(r.get("buyRatio", 0.5))
                sell_r = float(r.get("sellRatio", 0.5))
                all_records.append({"ts_ms": ts_ms, "buy_ratio": buy_r, "sell_ratio": sell_r})
            oldest_ms = min(int(r.get("timestamp", current_end)) for r in rows)
            if oldest_ms <= start_ms:
                break
            current_end = oldest_ms - 1
            time.sleep(0.08)
        except Exception as e:
            print(f"  [L/S] fetch error: {e}")
            break

    if not all_records:
        return pd.DataFrame(columns=["ts_ms", "buy_ratio", "sell_ratio"])

    df = pd.DataFrame(all_records)
    df.drop_duplicates(subset=["ts_ms"], inplace=True)
    df.sort_values("ts_ms", inplace=True)
    df = df[(df["ts_ms"] >= start_ms) & (df["ts_ms"] <= end_ms)].reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    print(f"  [L/S] Fetched {len(df)} records for {symbol}")
    return df


# ── ATR helper ────────────────────────────────────────────────────────────
def _compute_atr(highs, lows, closes, period=14):
    n = len(closes)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i]  - closes[i-1]))
    atr = np.zeros(n)
    atr[period-1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    # 초반 채우기
    for i in range(period-1):
        atr[i] = atr[period-1]
    return atr


# ── Structural Signal Engine ───────────────────────────────────────────────
class StructuralSignalEngine:
    """
    구조적 메커니즘 신호 엔진.
    각 봉의 13-dim structural features를 받아 방향 + 점수 반환.

    신호 조합 (active_signals로 제어):
      'fr'  : 펀딩레이트 극단 스퀴즈
      'oi'  : OI-가격 방향 불일치 (선택적 필터 또는 독립 신호)
      'liq' : 강제청산 폭발 후 역전
      'cvd' : CVD-가격 다이버전스 (흡수/분배)
    """

    # Feature 인덱스 (FEAT_NAMES 순서)
    I_FR_Z       = 0
    I_FR_TREND   = 1
    I_OI_CHG_Z   = 2
    I_OI_DIV     = 3
    I_LIQ_LONG   = 4
    I_LIQ_SHORT  = 5
    I_CVD_DELTA  = 6
    I_CVD_TREND  = 7
    I_CVD_DIV    = 8
    I_TAKER_Z    = 9
    I_EMA200_DEV = 10
    I_EMA200_SLP = 11
    I_VOL_REG    = 12

    def __init__(self,
                 fr_thr:        float = 2.0,   # FR z-score 임계값
                 oi_div_thr:    float = 0.3,   # OI-price divergence 임계값
                 liq_thr:       float = 2.0,   # 청산 z-score 임계값
                 cvd_div_thr:   float = 0.4,   # CVD divergence 임계값
                 min_score:     float = 1.0,   # 최소 진입 점수
                 active_signals: set = None,   # {'fr','oi','liq','cvd'}
                 trend_ema:     int   = 200,   # EMA 레짐 필터 (0=비활성)
                 long_only:     bool  = False,
                 short_only:    bool  = False,
                 ):
        self.fr_thr       = fr_thr
        self.oi_div_thr   = oi_div_thr
        self.liq_thr      = liq_thr
        self.cvd_div_thr  = cvd_div_thr
        self.min_score    = min_score
        self.active       = active_signals or {"fr"}
        self.trend_ema    = trend_ema
        self.long_only    = long_only
        self.short_only   = short_only

    def score(self, feat: np.ndarray,
              price: float = 0.0,
              ema200: float = 0.0) -> tuple:
        """
        Returns: (direction: int, score: float, reasons: list)
          direction:  1=LONG, -1=SHORT, 0=NO SIGNAL
        """
        long_score  = 0.0
        short_score = 0.0
        reasons     = []

        fr_z      = float(feat[self.I_FR_Z])
        fr_trend  = float(feat[self.I_FR_TREND])
        oi_div    = float(feat[self.I_OI_DIV])
        liq_long  = float(feat[self.I_LIQ_LONG])
        liq_short = float(feat[self.I_LIQ_SHORT])
        cvd_div   = float(feat[self.I_CVD_DIV])
        taker_z   = float(feat[self.I_TAKER_Z])
        vol_reg   = float(feat[self.I_VOL_REG])
        ema_dev   = float(feat[self.I_EMA200_DEV])

        # 변동성 증폭 인자 (고변동성 레짐에서 신호 강화)
        vol_amp = 1.0 + max(vol_reg, 0.0) * 0.2

        # ── 신호 1: FR 스퀴즈 ──────────────────────────────────────────
        # FR << 0 → 숏 오버레버리지 → 강제 롱스퀴즈 → LONG
        # FR >> 0 → 롱 오버레버리지 → 강제 숏스퀴즈 → SHORT
        if "fr" in self.active:
            if fr_z < -self.fr_thr:
                base = min((-fr_z - self.fr_thr) * 0.8 + 0.5, 2.0)
                # fr_trend도 음수이면 쏠림 심화 중 → 신호 강화
                if fr_trend < -0.5:
                    base = min(base * 1.3, 2.5)
                    reasons.append(f"FR_LONG(z={fr_z:.2f},trend={fr_trend:.2f},+{base*vol_amp:.2f})")
                else:
                    reasons.append(f"FR_LONG(z={fr_z:.2f},+{base*vol_amp:.2f})")
                long_score += base * vol_amp

            elif fr_z > self.fr_thr:
                base = min((fr_z - self.fr_thr) * 0.8 + 0.5, 2.0)
                if fr_trend > 0.5:
                    base = min(base * 1.3, 2.5)
                    reasons.append(f"FR_SHORT(z={fr_z:.2f},trend={fr_trend:.2f},+{base*vol_amp:.2f})")
                else:
                    reasons.append(f"FR_SHORT(z={fr_z:.2f},+{base*vol_amp:.2f})")
                short_score += base * vol_amp

        # ── 신호 2: OI-가격 방향 불일치 ───────────────────────────────
        # oi_div > 0: OI 증가인데 가격 하락 (숏 누적) → 롱스퀴즈 준비
        # oi_div < 0: OI 증가인데 가격 상승 (롱 누적) → 숏스퀴즈 준비
        if "oi" in self.active:
            if oi_div > self.oi_div_thr:
                s = min((oi_div - self.oi_div_thr) * 1.5 + 0.4, 1.8)
                long_score += s
                reasons.append(f"OI_DIV_LONG(div={oi_div:.2f},+{s:.2f})")
            elif oi_div < -self.oi_div_thr:
                s = min((-oi_div - self.oi_div_thr) * 1.5 + 0.4, 1.8)
                short_score += s
                reasons.append(f"OI_DIV_SHORT(div={oi_div:.2f},+{s:.2f})")

        # ── 신호 3: 강제청산 폭발 후 역전 ─────────────────────────────
        # 롱 청산 폭발 → 낙폭 과대 → 매도 연료 소진 → LONG 역전
        # 숏 청산 폭발 → 급등 과대 → 매수 연료 소진 → SHORT 역전
        if "liq" in self.active:
            if liq_long > self.liq_thr:
                s = min((liq_long - self.liq_thr) * 0.6 + 0.4, 1.8)
                long_score += s
                reasons.append(f"LIQ_EXHAUST_LONG(z={liq_long:.2f},+{s:.2f})")
            if liq_short > self.liq_thr:
                s = min((liq_short - self.liq_thr) * 0.6 + 0.4, 1.8)
                short_score += s
                reasons.append(f"LIQ_EXHAUST_SHORT(z={liq_short:.2f},+{s:.2f})")

        # ── 신호 4: CVD-가격 다이버전스 (스마트머니 흡수/분배) ─────────
        # cvd_div > 0: 가격 하락인데 CVD 매수우세 → 스마트머니 흡수 → LONG
        # cvd_div < 0: 가격 상승인데 CVD 매도우세 → 스마트머니 분배 → SHORT
        if "cvd" in self.active:
            if cvd_div > self.cvd_div_thr:
                s = min((cvd_div - self.cvd_div_thr) * 1.5 + 0.3, 1.5)
                # taker 매수 확인 시 강화
                if taker_z > 0.5:
                    s = min(s * 1.2, 1.8)
                long_score += s
                reasons.append(f"CVD_ABSORB(div={cvd_div:.2f},tkr={taker_z:.2f},+{s:.2f})")
            elif cvd_div < -self.cvd_div_thr:
                s = min((-cvd_div - self.cvd_div_thr) * 1.5 + 0.3, 1.5)
                if taker_z < -0.5:
                    s = min(s * 1.2, 1.8)
                short_score += s
                reasons.append(f"CVD_DISTRIB(div={cvd_div:.2f},tkr={taker_z:.2f},+{s:.2f})")

        # ── 레짐 필터: EMA200 ─────────────────────────────────────────
        if self.trend_ema > 0 and ema200 > 0:
            if long_score  >= self.min_score and price < ema200:
                return 0, 0.0, ["EMA_BLOCK(LONG in downtrend)"]
            if short_score >= self.min_score and price > ema200:
                return 0, 0.0, ["EMA_BLOCK(SHORT in uptrend)"]

        # ── 방향 필터 ─────────────────────────────────────────────────
        if self.long_only  and short_score > long_score:
            return 0, 0.0, []
        if self.short_only and long_score  > short_score:
            return 0, 0.0, []

        # ── 진입 결정 ─────────────────────────────────────────────────
        if long_score >= self.min_score and long_score > short_score:
            return 1, long_score, reasons
        elif short_score >= self.min_score and short_score > long_score:
            return -1, short_score, reasons
        return 0, 0.0, []


# ── Main backtest ──────────────────────────────────────────────────────────
def run_backtest(args, symbol_override=None):
    t0 = time.time()

    # ── 데이터 수집 ──────────────────────────────────────────────────────
    symbol = symbol_override or args.symbol

    dc = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")
    print(f"[data] Bybit Mainnet  REST={REST_BASE}")

    end_dt = datetime.now(timezone.utc)
    if args.end_date:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    start_str = args.start_date
    end_str   = end_dt.strftime("%Y-%m-%d")

    print(f"[data] {symbol} {args.timeframe} {start_str} → {end_str}")
    df_raw = dc.fetch_training_history(
        symbol=symbol, timeframe=args.timeframe,
        start_date=start_str, end_ms=int(end_dt.timestamp() * 1000),
        cache_dir="data",
    )
    if df_raw is None or df_raw.empty:
        print("[data] ERROR: No data."); return
    print(f"[data] {len(df_raw)} bars")

    _start_ms = int(datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    _end_ms   = int(end_dt.timestamp() * 1000)

    # Funding Rate 병합
    try:
        df_fr = dc.fetch_funding_history(symbol, _start_ms, _end_ms, cache_dir="data")
        if not df_fr.empty:
            df_fr["ts"] = (pd.to_datetime(df_fr["ts_ms"], unit="ms", utc=True)
                           .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            df_raw = df_raw.merge(df_fr[["ts", "funding_rate"]], on="ts", how="left")
            df_raw["funding_rate"] = df_raw["funding_rate"].ffill().fillna(0.0)
            print(f"  [FR] {(df_raw['funding_rate']!=0).sum()}/{len(df_raw)} bars")
        else:
            df_raw["funding_rate"] = 0.0
    except Exception as e:
        print(f"  [FR] Skip: {e}")
        df_raw["funding_rate"] = 0.0

    # Open Interest 병합
    try:
        df_oi = dc.fetch_open_interest_history(
            symbol, _start_ms, _end_ms, interval="1h", cache_dir="data")
        if not df_oi.empty:
            df_oi["ts"] = (pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
                           .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            df_raw = df_raw.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
            df_raw["open_interest"] = df_raw["open_interest"].ffill().fillna(0.0)
            print(f"  [OI] {(df_raw['open_interest']!=0).sum()}/{len(df_raw)} bars")
        else:
            df_raw["open_interest"] = 0.0
    except Exception as e:
        print(f"  [OI] Skip: {e}")
        df_raw["open_interest"] = 0.0

    # Binance Taker CVD 병합
    cache_cvd = (f"data/binance_taker_{symbol}_{args.timeframe}_"
                 f"{start_str.replace('-','')}_{end_str.replace('-','')}.csv")
    try:
        df_cvd = fetch_binance_taker_history(
            symbol=symbol, interval=args.timeframe,
            start_date=start_str, end_date=end_str,
            cache_path=cache_cvd,
        )
        if df_cvd is not None and not df_cvd.empty and "taker_buy_volume" in df_cvd.columns:
            cvd_map = dict(zip(df_cvd["ts"], df_cvd["taker_buy_volume"]))
            df_raw["taker_buy_volume"] = df_raw["ts"].map(cvd_map).fillna(0.0)
            print(f"  [CVD] {int((df_raw['taker_buy_volume']>0).sum())}/{len(df_raw)} bars")
        else:
            df_raw["taker_buy_volume"] = 0.0
    except Exception as e:
        print(f"  [CVD] Skip: {e}")
        df_raw["taker_buy_volume"] = 0.0

    # Long/Short Ratio 병합 (Bybit /v5/market/account-ratio — 실제 포지션 쏠림 데이터)
    # NOTE: /v5/market/liquidation → 404 (Bybit V5에 REST 청산 기록 엔드포인트 없음)
    # 대체: account-ratio(buyRatio/sellRatio)로 포지션 집중도 측정 (청산 압력 프록시)
    #   liq_long_usd  = buyRatio  × 1e6  (롱 과집중 → 롱 강제청산 취약도)
    #   liq_short_usd = sellRatio × 1e6  (숏 과집중 → 숏 강제청산 취약도)
    try:
        df_ls = fetch_ls_ratio_history(symbol, _start_ms, _end_ms, cache_dir="data")
        if not df_ls.empty:
            df_ls["ts"] = (pd.to_datetime(df_ls["ts_ms"], unit="ms", utc=True)
                           .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
            ls_map_long  = dict(zip(df_ls["ts"], df_ls["buy_ratio"]  * 1e6))
            ls_map_short = dict(zip(df_ls["ts"], df_ls["sell_ratio"] * 1e6))
            df_raw["liq_long_usd"]  = df_raw["ts"].map(ls_map_long).fillna(0.0)
            df_raw["liq_short_usd"] = df_raw["ts"].map(ls_map_short).fillna(0.0)
            n_long  = int((df_raw["liq_long_usd"]  > 0).sum())
            n_short = int((df_raw["liq_short_usd"] > 0).sum())
            avg_buy = df_ls["buy_ratio"].mean()
            print(f"  [L/S_RATIO] long_bars={n_long}  short_bars={n_short}  "
                  f"avg_buyRatio={avg_buy:.3f}  records={len(df_ls)}")
        else:
            df_raw["liq_long_usd"]  = 0.0
            df_raw["liq_short_usd"] = 0.0
            print("  [L/S_RATIO] No data — liq features will use wick×vol proxy")
    except Exception as e:
        print(f"  [L/S_RATIO] Skip: {e}")
        df_raw["liq_long_usd"]  = 0.0
        df_raw["liq_short_usd"] = 0.0

    # ── 구조적 피처 빌드 ──────────────────────────────────────────────────
    print(f"\n[features] Building structural features ({FEAT_DIM}-dim) ...")
    feats = build_structural_features(df_raw, verbose=True)
    if feats is None or len(feats) < 100:
        print("[features] ERROR"); return

    closes = df_raw["close"].values.astype(np.float64)
    highs  = df_raw["high"].values.astype(np.float64)
    lows   = df_raw["low"].values.astype(np.float64)
    atr14  = _compute_atr(highs, lows, closes, period=14)

    # EMA200 사전 계산
    ema200 = np.zeros(len(closes))
    alpha_e = 2.0 / (args.trend_ema + 1.0)
    ema200[0] = closes[0]
    for i in range(1, len(closes)):
        ema200[i] = alpha_e * closes[i] + (1 - alpha_e) * ema200[i - 1]

    # ── 신호 엔진 초기화 ──────────────────────────────────────────────────
    active_set = set(s.strip() for s in args.signals.lower().split(","))
    engine = StructuralSignalEngine(
        fr_thr        = args.fr_thr,
        oi_div_thr    = args.oi_div_thr,
        liq_thr       = args.liq_thr,
        cvd_div_thr   = args.cvd_div_thr,
        min_score     = args.min_score,
        active_signals= active_set,
        trend_ema     = args.trend_ema,
        long_only     = args.long_only,
        short_only    = args.short_only,
    )

    dir_str = "LONG-ONLY" if args.long_only else ("SHORT-ONLY" if args.short_only else "BOTH")
    print(f"\n[config] Signals={active_set}  fr_thr={args.fr_thr}  oi_div_thr={args.oi_div_thr}")
    print(f"[config] liq_thr={args.liq_thr}  cvd_div_thr={args.cvd_div_thr}  min_score={args.min_score}")
    print(f"[config] TP={args.tp_mult}×ATR  SL={args.sl_mult}×ATR  "
          f"Leverage={args.leverage}x  TrendEMA={args.trend_ema}  Dir={dir_str}")

    # ── 백테스트 루프 ─────────────────────────────────────────────────────
    warmup      = max(200, args.trend_ema + 10)  # EMA200 안정화
    equity      = float(args.capital)
    max_equity  = equity
    max_dd      = 0.0
    position    = 0   # 0=flat, 1=long, -1=short
    entry_price = entry_atr = entry_notional = 0.0
    tp_price    = sl_price  = 0.0
    hold_bars   = 0
    trades      = []

    eta_rt  = 0.00075   # round-trip fee (maker+taker approx)
    pos_frac = args.pos_frac
    eff_lev  = args.leverage * pos_frac

    for i in tqdm(range(warmup, len(feats)), desc="Backtesting"):
        feat  = feats[i]
        price = float(closes[i])
        hi    = float(highs[i])
        lo    = float(lows[i])
        atr   = float(atr14[i]) if atr14[i] > 0 else price * 0.01
        ema_v = float(ema200[i])

        # ── 포지션 관리 ─────────────────────────────────────────────
        if position != 0:
            hold_bars += 1

            # TP/SL 체크 (봉내 가격으로)
            hit_tp = (position ==  1 and hi >= tp_price) or \
                     (position == -1 and lo <= tp_price)
            hit_sl = (position ==  1 and lo <= sl_price) or \
                     (position == -1 and hi >= sl_price)
            hit_max = hold_bars >= args.max_hold

            exit_type  = None
            exit_price = price

            if hit_sl and not hit_tp:
                exit_type  = "SL"
                exit_price = sl_price
            elif hit_tp:
                exit_type  = "TP"
                exit_price = tp_price
            elif hit_max:
                exit_type  = "MAX_HOLD"
                exit_price = price

            if exit_type:
                raw_ret = (exit_price / entry_price - 1.0) * position
                fee     = eta_rt * eff_lev
                net_ret = raw_ret * eff_lev - fee
                pnl_usd = entry_notional * net_ret
                equity  = max(equity + pnl_usd, 0.001)

                trades.append({
                    "entry_i":   i - hold_bars,
                    "exit_i":    i,
                    "side":      "LONG" if position == 1 else "SHORT",
                    "entry_px":  entry_price,
                    "exit_px":   exit_price,
                    "exit_type": exit_type,
                    "pnl_pct":   net_ret * 100,
                    "pnl_usd":   pnl_usd,
                    "hold":      hold_bars,
                    "equity":    equity,
                    "entry_atr": entry_atr,
                })

                max_equity = max(max_equity, equity)
                dd = (max_equity - equity) / max_equity * 100
                max_dd = max(max_dd, dd)

                position  = 0
                hold_bars = 0

                if equity <= 0:
                    break
                continue

        # ── 신호 탐색 ────────────────────────────────────────────────
        if position == 0:
            direction, score, reasons = engine.score(feat, price=price, ema200=ema_v)
            if direction != 0 and atr > 0:
                position      = direction
                entry_price   = price
                entry_atr     = atr
                entry_notional = equity * pos_frac
                hold_bars     = 0

                if direction == 1:   # LONG
                    tp_price = entry_price + args.tp_mult * atr
                    sl_price = entry_price - args.sl_mult * atr
                else:                # SHORT
                    tp_price = entry_price - args.tp_mult * atr
                    sl_price = entry_price + args.sl_mult * atr

    # ── 열린 포지션 강제 청산 ─────────────────────────────────────────────
    if position != 0 and len(feats) > 0:
        exit_price = float(closes[-1])
        raw_ret    = (exit_price / entry_price - 1.0) * position
        fee        = eta_rt * eff_lev
        net_ret    = raw_ret * eff_lev - fee
        pnl_usd    = entry_notional * net_ret
        equity     = max(equity + pnl_usd, 0.001)
        trades.append({
            "entry_i": len(feats) - hold_bars - 1,
            "exit_i":  len(feats) - 1,
            "side":    "LONG" if position == 1 else "SHORT",
            "entry_px":  entry_price,
            "exit_px":   exit_price,
            "exit_type": "END",
            "pnl_pct":   net_ret * 100,
            "pnl_usd":   pnl_usd,
            "hold":      hold_bars,
            "equity":    equity,
            "entry_atr": entry_atr,
        })

    # ── 결과 집계 ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    df_t = pd.DataFrame(trades)

    n_total  = len(df_t)
    n_tp     = int((df_t["exit_type"] == "TP").sum())   if n_total else 0
    n_sl     = int((df_t["exit_type"] == "SL").sum())   if n_total else 0
    n_max    = int((df_t["exit_type"] == "MAX_HOLD").sum()) if n_total else 0
    n_end    = int((df_t["exit_type"] == "END").sum())  if n_total else 0
    n_long   = int((df_t["side"] == "LONG").sum())      if n_total else 0
    n_short  = int((df_t["side"] == "SHORT").sum())     if n_total else 0
    n_win    = int((df_t["pnl_usd"] > 0).sum())         if n_total else 0
    wr       = n_win / n_total * 100 if n_total else 0.0
    roi      = (equity / args.capital - 1.0) * 100
    avg_hold = float(df_t["hold"].mean()) if n_total else 0.0
    avg_pnl  = float(df_t["pnl_usd"].mean()) if n_total else 0.0
    pf_denom = abs(df_t[df_t["pnl_usd"] < 0]["pnl_usd"].sum()) if n_total else 1
    pf       = (df_t[df_t["pnl_usd"] > 0]["pnl_usd"].sum() / pf_denom) if pf_denom > 0 else 0.0

    bep = 1.0 / (1.0 + args.tp_mult / args.sl_mult) * 100
    ev  = (wr / 100 * args.tp_mult - (1 - wr / 100) * args.sl_mult) if n_total else 0.0

    print()
    print("=" * 68)
    print("  STRUCTURAL MECHANISM BACKTEST RESULT")
    print("=" * 68)
    print(f"  Period          : {start_str} ~ {end_str}")
    print(f"  Bars            : {len(feats)} ({args.timeframe})")
    print(f"  Signals         : {active_set}")
    print(f"  Direction       : {dir_str}")
    print("-" * 68)
    print(f"  Initial Capital : ${args.capital:.2f}")
    print(f"  Final Equity    : ${equity:.2f}")
    print(f"  Net ROI         : {roi:+.2f}%")
    print(f"  Max Drawdown    : {max_dd:.2f}%")
    print(f"  Profit Factor   : {pf:.3f}")
    print("-" * 68)
    print(f"  Total Trades    : {n_total}")
    print(f"    LONG / SHORT  : {n_long} / {n_short}")
    print(f"    TP / SL       : {n_tp} / {n_sl}  (MaxHold={n_max}  END={n_end})")
    print(f"  Win Rate        : {wr:.1f}%  (BEP={bep:.1f}%)")
    print(f"  EV/trade (R)    : {ev:+.3f}")
    print(f"  Avg Hold        : {avg_hold:.1f} bars")
    print(f"  Avg PnL/trade   : ${avg_pnl:.4f}")
    print("=" * 68)

    # 연도별 수익 분석
    if n_total > 0 and "entry_i" in df_t.columns:
        print("\n  [연도별 분석]")
        df_t["year"] = df_t["entry_i"].apply(
            lambda idx: df_raw.iloc[min(int(idx), len(df_raw)-1)]["ts"][:4]
            if "ts" in df_raw.columns else "?"
        )
        yearly = df_t.groupby("year").agg(
            trades=("pnl_usd","count"),
            wins=("pnl_usd", lambda x: (x>0).sum()),
            pnl=("pnl_usd","sum")
        )
        yearly["wr"] = (yearly["wins"] / yearly["trades"] * 100).round(1)
        yearly["pnl"] = yearly["pnl"].round(4)
        for yr, row in yearly.iterrows():
            print(f"    {yr}: trades={int(row.trades):3d}  WR={row.wr:.1f}%  PnL=${row.pnl:+.4f}")

    # 신호별 통계
    if n_total > 0:
        print(f"\n  [Elapsed] {elapsed:.1f}s")

    # CSV 저장
    os.makedirs("reports", exist_ok=True)
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if n_total > 0:
        df_t.to_csv(f"reports/structural_trades_{symbol}_{ts_now}.csv", index=False)
        print(f"[export] Trades → reports/structural_trades_{symbol}_{ts_now}.csv")

    # R-multiples CSV 저장 (Gate 2 통계 검증용)
    # R-multiple = (exit_px - entry_px) * direction / (sl_mult * entry_atr)
    # 수수료 포함: fee_R = eta_rt / (sl_mult * atr / entry_px)
    if n_total > 0:
        rmult_csv = getattr(args, "rmultiples_csv", None) or f"reports/rmultiples_{symbol}_{ts_now}.csv"
        directions = df_t["side"].map({"LONG": 1, "SHORT": -1}).values
        sl_distances = df_t["entry_atr"].values * args.sl_mult
        raw_moves = (df_t["exit_px"].values - df_t["entry_px"].values) * directions
        fee_price = eta_rt * df_t["entry_px"].values  # fee on full notional (price-space)
        r_multiples = (raw_moves - fee_price) / sl_distances
        pd.DataFrame({"r_multiple": r_multiples}).to_csv(rmult_csv, index=False)
        print(f"[export] R-multiples ({len(r_multiples)} trades) → {rmult_csv}")

    return {
        "symbol":   symbol,
        "n_trades": n_total,
        "n_win":    n_win,
        "wr":       wr,
        "roi":      roi,
        "max_dd":   max_dd,
        "pf":       pf,
        "ev":       ev,
        "equity":   equity,
        "capital":  args.capital,
        "df_trades": df_t if n_total > 0 else pd.DataFrame(),
        "r_multiples": r_multiples if n_total > 0 else np.array([], dtype=np.float64),
    }


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Structural Mechanism Backtest")

    # 데이터
    p.add_argument("--symbol",     default="BTCUSDT")
    p.add_argument("--symbols",    default=None,
                   help="멀티심볼 쉼표구분 (예: BTCUSDT,ETHUSDT,SOLUSDT). 지정시 --symbol 무시")
    p.add_argument("--timeframe",  default="1h")
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date",   default=None,  help="YYYY-MM-DD (default=today)")
    p.add_argument("--capital",    type=float, default=10.0)

    # 신호 선택
    p.add_argument("--signals",    default="fr",
                   help="쉼표 구분: fr, oi, liq, cvd (예: fr,oi,liq)")
    p.add_argument("--long-only",  action="store_true")
    p.add_argument("--short-only", action="store_true")

    # 신호 파라미터
    p.add_argument("--fr-thr",       type=float, default=2.0,  help="FR z-score 임계값")
    p.add_argument("--oi-div-thr",   type=float, default=0.3,  help="OI-price divergence 임계값")
    p.add_argument("--liq-thr",      type=float, default=2.0,  help="청산 z-score 임계값")
    p.add_argument("--cvd-div-thr",  type=float, default=0.4,  help="CVD divergence 임계값")
    p.add_argument("--min-score",    type=float, default=0.5,  help="최소 진입 점수")
    p.add_argument("--trend-ema",    type=int,   default=200,  help="레짐 필터 EMA (0=비활성)")

    # 포지션/리스크
    p.add_argument("--tp-mult",    type=float, default=3.0)
    p.add_argument("--sl-mult",    type=float, default=1.0)
    p.add_argument("--leverage",   type=float, default=5.0)
    p.add_argument("--pos-frac",   type=float, default=0.5)
    p.add_argument("--max-hold",   type=int,   default=96,    help="최대 보유 봉수")
    p.add_argument("--rmultiples-csv", type=str, default=None,
                   help="R-multiples 출력 CSV 경로 (예: q1_2026_rmultiples.csv)")

    args = p.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
        results = []
        for sym in symbols:
            print(f"\n{'='*68}")
            print(f"  SYMBOL: {sym}")
            print(f"{'='*68}")
            res = run_backtest(args, symbol_override=sym)
            if res:
                results.append(res)

        if len(results) > 1:
            # 합산 통계
            total_trades = sum(r["n_trades"] for r in results)
            total_wins   = sum(r["n_win"]    for r in results)
            combined_wr  = total_wins / total_trades * 100 if total_trades else 0.0
            avg_roi      = sum(r["roi"]    for r in results) / len(results)
            avg_mdd      = sum(r["max_dd"] for r in results) / len(results)
            avg_ev       = sum(r["ev"]     for r in results) / len(results)
            trades_yr    = total_trades / 7.0   # 2019-2026 기준

            print(f"\n{'='*68}")
            print(f"  MULTI-SYMBOL COMBINED SUMMARY ({len(results)} symbols)")
            print(f"{'='*68}")
            for r in results:
                print(f"  {r['symbol']:10s}: trades={r['n_trades']:4d}  WR={r['wr']:.1f}%  "
                      f"ROI={r['roi']:+.1f}%  MDD={r['max_dd']:.1f}%  EV={r['ev']:+.3f}R")
            print(f"  {'-'*62}")
            print(f"  TOTAL     : trades={total_trades:4d}  WR={combined_wr:.1f}%  "
                  f"Avg ROI={avg_roi:+.1f}%  Avg MDD={avg_mdd:.1f}%  EV={avg_ev:+.3f}R")
            print(f"  거래빈도  : {trades_yr:.0f}회/년  ({trades_yr/12:.1f}회/월)")
            print(f"{'='*68}")
    else:
        run_backtest(args)


if __name__ == "__main__":
    main()
