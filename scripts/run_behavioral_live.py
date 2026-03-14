"""
run_behavioral_live.py
──────────────────────────────────────────────────────────────────────
Behavioral Alpha Engine — Real-Time Live Signal Runner

매 N분마다 Bybit + Binance에서 최신 봉 데이터를 가져와
FR / LIQ / CVD 신호를 실시간으로 출력합니다.

CVD 소스: Binance taker_buy_volume (실제 체결 데이터, API 키 불필요)
  - real_delta = 2 × taker_buy_volume − total_volume
  - 양수 = 순 매수압력, 음수 = 순 매도압력

사용:
  # 1h BTC (기본 — FR+EMA200 최적 세팅)
  python scripts/run_behavioral_live.py --symbol BTCUSDT --timeframe 1h

  # 15m BTC, 모든 신호 (FR+LIQ+CVD)
  python scripts/run_behavioral_live.py --symbol BTCUSDT --timeframe 15m \\
    --signals fr,liq,cvd --min-score 0.5 --interval 15

  # 5m BTC, 롱 전용 + EMA200 게이트
  python scripts/run_behavioral_live.py --symbol BTCUSDT --timeframe 5m \\
    --signals fr,liq,cvd --long-only --trend-ema 200 --min-score 0.4 --interval 5

  # 소수익 추구: ETH 15m, 낮은 임계값
  python scripts/run_behavioral_live.py --symbol ETHUSDT --timeframe 15m \\
    --signals liq,cvd --liq-z-thr 1.2 --cvd-div-thr 0.3 --min-score 0.4
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.data_client import DataClient
from src.models.features_v4 import build_features_v4
from src.models.features_v2 import compute_true_atr
from src.data.binance_client import BinancePublicClient
from src.models.oi_profile import compute_poc_dist_live

# BehavioralAlphaEngine는 backtest_behavioral에서 import
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)
from backtest_behavioral import BehavioralAlphaEngine  # noqa: E402

# ── 상수 ──────────────────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))

TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
}

# 피처 계산용 롤링 윈도우 크기 (봉 수)
LOOKBACK = 220

# V4 피처 인덱스 (backtest_behavioral.py와 동일)
IDX_FR_Z      = 18
IDX_VOL_Z     = 20
IDX_OI_PCT    = 21
IDX_FR_VEL    = 22
IDX_CVD_DELTA = 23
IDX_CVD_TREND = 24
IDX_CVD_DIV   = 25
IDX_LIQ_LONG  = 26
IDX_LIQ_SHORT = 27


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _ema_last(arr: np.ndarray, period: int) -> float:
    """배열 전체에 EMA 적용 후 마지막 값 반환."""
    if len(arr) == 0:
        return 0.0
    alpha = 2.0 / (period + 1.0)
    val = float(arr[0])
    for x in arr[1:]:
        val = alpha * float(x) + (1.0 - alpha) * val
    return val


def _bar_interval_ms(timeframe: str) -> int:
    tf_min = TF_MINUTES.get(timeframe, 60)
    return tf_min * 60 * 1000


def _seconds_to_next_close(timeframe: str) -> int:
    """현재 시각 기준 다음 봉 마감까지 남은 초."""
    tf_min = TF_MINUTES.get(timeframe, 60)
    tf_sec = tf_min * 60
    now_ts = int(time.time())
    elapsed = now_ts % tf_sec
    return tf_sec - elapsed


# ── 데이터 수집 ──────────────────────────────────────────────────────────────

def fetch_live_window(
    symbol:    str,
    timeframe: str,
    dc:        DataClient,
    binance:   BinancePublicClient,
) -> pd.DataFrame | None:
    """
    최신 LOOKBACK봉 데이터를 Bybit + Binance에서 실시간으로 수집.

    컬럼: open, high, low, close, volume, ts
         + funding_rate (Bybit, ffill)
         + open_interest (Bybit, ffill)
         + taker_buy_volume (Binance 실체결, real CVD 소스)
    """
    tf_min  = TF_MINUTES.get(timeframe, 60)
    end_dt  = datetime.now(timezone.utc)
    # 2배 버퍼 — 정렬 손실 감안
    buf_dt  = end_dt - timedelta(minutes=tf_min * LOOKBACK * 2 + 60)
    start_str = buf_dt.strftime("%Y-%m-%d")
    end_ms    = int(end_dt.timestamp() * 1000)
    _start_ms = int(buf_dt.timestamp() * 1000)

    # ── 1. Bybit OHLCV ────────────────────────────────────────────────────
    try:
        df = dc.fetch_training_history(
            symbol=symbol, timeframe=timeframe,
            start_date=start_str,
            end_ms=end_ms,
            cache_dir="data",
        )
    except Exception as e:
        print(f"  [!] Bybit OHLCV error: {e}")
        return None

    if df is None or len(df) < 50:
        print(f"  [!] 데이터 부족 ({len(df) if df is not None else 0} bars)")
        return None

    # 최신 LOOKBACK봉만 사용
    df = df.tail(LOOKBACK).copy().reset_index(drop=True)

    # ── 2. Funding Rate (Bybit, 8h 주기 → ffill) ─────────────────────────
    try:
        df_fr = dc.fetch_funding_history(symbol, _start_ms, end_ms, cache_dir="data")
        if not df_fr.empty:
            df_fr["ts"] = (
                pd.to_datetime(df_fr["ts_ms"], unit="ms", utc=True)
                .dt.tz_localize(None)
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )
            df = df.merge(df_fr[["ts", "funding_rate"]], on="ts", how="left")
            df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
        else:
            df["funding_rate"] = 0.0
    except Exception:
        df["funding_rate"] = 0.0

    # ── 3. Open Interest (Bybit, 15min 주기 → ffill) ──────────────────────
    try:
        df_oi = dc.fetch_open_interest_history(
            symbol, _start_ms, end_ms, interval="1h", cache_dir="data"
        )
        if not df_oi.empty:
            df_oi["ts"] = (
                pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
                .dt.tz_localize(None)
                .dt.strftime("%Y-%m-%d %H:%M:%S")
            )
            df = df.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
            df["open_interest"] = df["open_interest"].ffill().fillna(0.0)
        else:
            df["open_interest"] = 0.0
    except Exception:
        df["open_interest"] = 0.0

    # ── 4. Binance CVD (실체결 taker_buy_volume) ──────────────────────────
    # Binance 공개 API (API 키 불필요) — real_delta = 2×taker_buy − total_vol
    try:
        cvd_start_ms = int(
            (end_dt - timedelta(minutes=tf_min * LOOKBACK * 2 + 60))
            .timestamp() * 1000
        )
        df_cvd = binance.fetch_klines_raw(
            symbol, timeframe, cvd_start_ms, end_ms
        )
        if not df_cvd.empty and "taker_buy_volume" in df_cvd.columns:
            cvd_map = dict(zip(df_cvd["ts"], df_cvd["taker_buy_volume"]))
            df["taker_buy_volume"] = df["ts"].map(cvd_map).fillna(0.0)
            merged_n = int((df["taker_buy_volume"] > 0).sum())
            print(f"  [CVD] Binance taker {merged_n}/{len(df)} bars merged "
                  f"(avg_taker_ratio="
                  f"{(df['taker_buy_volume'] / (df['volume'] + 1e-10)).mean():.3f})")
        else:
            df["taker_buy_volume"] = 0.0
            print("  [CVD] Binance 응답 없음 — OHLCV BVC 대체")
    except Exception as e:
        print(f"  [CVD] Binance 오류 ({e}) — OHLCV BVC 대체")
        df["taker_buy_volume"] = 0.0

    return df


# ── 화면 출력 ──────────────────────────────────────────────────────────────────

def _signal_bar(direction: int, score: float) -> str:
    """방향/점수를 시각적 바로 변환."""
    if direction == 0:
        return "━━━━━━ FLAT ━━━━━━"
    bars = min(int(score * 5), 10)
    fill = "█" * bars + "░" * (10 - bars)
    if direction == 1:
        return f"▲ LONG  [{fill}] {score:.2f}"
    return f"▼ SHORT [{fill}] {score:.2f}"


def display_signal(
    symbol:    str,
    timeframe: str,
    df:        pd.DataFrame,
    feat:      np.ndarray,
    direction: int,
    score:     float,
    reasons:   list,
    ema_val:   float,
    args,
    cvd_source: str = "Binance",
) -> None:
    """포맷된 신호를 터미널에 출력."""
    now_kst   = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    price     = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else price
    price_chg = (price - prev_close) / prev_close * 100

    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    atr14  = compute_true_atr(highs, lows, closes, period=14)
    atr_val = float(atr14[-1]) if len(atr14) > 0 else 0.0

    # 피처 값 추출
    fr_z      = float(feat[IDX_FR_Z])
    cvd_delta = float(feat[IDX_CVD_DELTA])
    cvd_trend = float(feat[IDX_CVD_TREND])
    cvd_div   = float(feat[IDX_CVD_DIV])
    liq_long  = float(feat[IDX_LIQ_LONG])
    liq_short = float(feat[IDX_LIQ_SHORT])

    # 레짐 판단
    if args.trend_ema > 0 and ema_val > 0:
        regime = f"BULL (가격 > EMA{args.trend_ema})" if price > ema_val \
                 else f"BEAR (가격 < EMA{args.trend_ema})"
    else:
        regime = "N/A"

    # TP / SL 계산
    tp_str = sl_str = ""
    if direction != 0 and atr_val > 0:
        tp = price * (1 + direction * args.tp_mult * atr_val / price)
        sl = price * (1 - direction * args.sl_mult * atr_val / price)
        tp_str = f"  TP=${tp:,.2f}  SL=${sl:,.2f}"

    W = 60
    sep = "═" * W

    print(f"\n{sep}")
    print(f"  BEHAVIORAL ALPHA LIVE  [{now_kst} KST]")
    print(f"{sep}")
    print(f"  {symbol}  {timeframe}  ${price:,.4f}  ({price_chg:+.2f}%/봉)")
    print(f"  ATR(14)={atr_val:.4f}  CVD소스={cvd_source}")
    print(f"{'─'*W}")
    print(f"  ┌ 신호 지표 ─────────────────────────────────────────┐")
    fr_bar  = "+" * max(0, int(fr_z * 3)) if fr_z > 0 else "-" * max(0, int(-fr_z * 3))
    print(f"  │  FR_Z       : {fr_z:+6.3f}  {fr_bar[:20]:<20}  │")
    print(f"  │  CVD_delta  : {cvd_delta:+6.3f}  (봉 매수압력 z-score)       │")
    print(f"  │  CVD_trend  : {cvd_trend:+6.3f}  (20봉 누적 매수 우세도)      │")
    print(f"  │  CVD_div    : {cvd_div:+6.3f}  (CVD↔가격 다이버전스)        │")
    print(f"  │  Liq_LONG   : {liq_long:+6.3f}  (롱청산 프록시 z-score)       │")
    print(f"  │  Liq_SHORT  : {liq_short:+6.3f}  (숏청산 프록시 z-score)       │")
    print(f"  └───────────────────────────────────────────────────┘")
    if args.trend_ema > 0:
        print(f"  레짐: {regime}  EMA{args.trend_ema}=${ema_val:,.0f}")
    print(f"{'─'*W}")

    sig_display = _signal_bar(direction, score)
    print(f"  신호: {sig_display}")

    if reasons:
        print(f"  이유:")
        for r in reasons:
            print(f"        {r}")
    if tp_str:
        print(tp_str)

    print(f"{sep}")


# ── 메인 루프 ──────────────────────────────────────────────────────────────────

def run_live(args: argparse.Namespace) -> None:
    dc      = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")
    binance = BinancePublicClient()

    active_set = (
        set(args.signals.lower().split(","))
        if args.signals else {"fr", "liq", "cvd"}
    )
    engine = BehavioralAlphaEngine(
        fr_z_thr       = args.fr_z_thr,
        liq_z_thr      = args.liq_z_thr,
        cvd_div_thr    = args.cvd_div_thr,
        oi_poc_thr     = args.oi_poc_thr,
        min_score      = args.min_score,
        active_signals = active_set,
    )

    tf_min = TF_MINUTES.get(args.timeframe, 60)
    dir_str = ("LONG-ONLY" if args.long_only
               else ("SHORT-ONLY" if args.short_only else "양방향"))

    print(f"{'═'*60}")
    print(f"  BEHAVIORAL ALPHA LIVE RUNNER")
    print(f"{'═'*60}")
    print(f"  심볼     : {args.symbol}  {args.timeframe}")
    print(f"  신호     : {sorted(active_set)}  (min_score={args.min_score})")
    print(f"  임계값   : FR≥{args.fr_z_thr}σ  LIQ≥{args.liq_z_thr}σ  CVD_div≥{args.cvd_div_thr}")
    print(f"  레짐게이트: EMA{args.trend_ema}  방향={dir_str}")
    print(f"  갱신주기 : {args.interval}분  (봉 마감 동기화: {'ON' if args.sync_candle else 'OFF'})")
    print(f"  CVD소스  : Binance (API 키 불필요)")
    print(f"  Ctrl+C 로 종료")
    print(f"{'═'*60}\n")

    while True:
        # ── 봉 마감 동기화 (옵션) ──────────────────────────────────────────
        if args.sync_candle:
            wait_sec = _seconds_to_next_close(args.timeframe) + 3  # +3s 여유
            if wait_sec > 10:
                print(f"  [sync] 봉 마감까지 {wait_sec}초 대기 ...")
                time.sleep(wait_sec)

        t_start = time.time()

        try:
            # ── 데이터 수집 ────────────────────────────────────────────────
            print(f"  [{datetime.now(KST).strftime('%H:%M:%S')} KST] 데이터 수집 중 ...")
            df = fetch_live_window(args.symbol, args.timeframe, dc, binance)

            if df is None or len(df) < 60:
                print(f"  [!] 데이터 부족, 60초 후 재시도 ...")
                time.sleep(60)
                continue

            cvd_source = (
                "Binance taker"
                if "taker_buy_volume" in df.columns and (df["taker_buy_volume"] > 0).any()
                else "OHLCV BVC"
            )

            # ── V4 피처 계산 (마지막 봉) ──────────────────────────────────
            feat = build_features_v4(df)  # 전체 df 롤링 → 마지막 봉 피처 반환

            # ── EMA 계산 ──────────────────────────────────────────────────
            closes  = df["close"].values.astype(float)
            highs_  = df["high"].values.astype(float)
            lows_   = df["low"].values.astype(float)
            opens_  = df["open"].values.astype(float)
            ema_val = _ema_last(closes, args.trend_ema) if args.trend_ema > 0 else 0.0

            # ── OI POC 거리 계산 (oi_poc 신호 활성화 시) ─────────────────
            atr14_ = compute_true_atr(highs_, lows_, closes, period=14)
            atr_val = float(atr14_[-1]) if len(atr14_) > 0 else 0.0

            if "oi_poc" in active_set and "open_interest" in df.columns:
                oi_series = df["open_interest"].values.astype(float)
                poc_dist = compute_poc_dist_live(
                    highs_, lows_, opens_, closes, oi_series, atr_val,
                    window=args.oi_poc_window, n_buckets=20,
                )
                feat_ext = np.append(feat, [poc_dist])
            else:
                feat_ext = feat

            # ── 신호 평가 ─────────────────────────────────────────────────
            direction, score, reasons = engine.score(feat_ext)

            # 추세 게이트
            if args.trend_ema > 0:
                direction = engine.apply_trend_gate(direction, closes[-1], ema_val)

            # 방향 필터
            if args.long_only  and direction == -1:
                direction = 0
            if args.short_only and direction ==  1:
                direction = 0

            # ── 출력 ──────────────────────────────────────────────────────
            display_signal(
                args.symbol, args.timeframe, df,
                feat, direction, score, reasons, ema_val, args,
                cvd_source=cvd_source,
            )

        except KeyboardInterrupt:
            print("\n[live] 종료.")
            break
        except Exception as e:
            import traceback
            print(f"\n[!] 오류: {e}")
            traceback.print_exc()

        # ── 다음 갱신까지 대기 ────────────────────────────────────────────
        elapsed = time.time() - t_start
        if not args.sync_candle:
            sleep_sec = max(0, args.interval * 60 - elapsed)
            if sleep_sec > 0:
                next_t = datetime.now(KST) + timedelta(seconds=sleep_sec)
                print(f"  다음 갱신: {next_t.strftime('%H:%M:%S')} KST "
                      f"({sleep_sec:.0f}초 후)")
                time.sleep(sleep_sec)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Behavioral Alpha Live Signal Runner (Binance CVD)"
    )
    p.add_argument("--symbol",      default="BTCUSDT",
                   help="거래쌍 (기본: BTCUSDT)")
    p.add_argument("--timeframe",   default="1h",
                   help="봉 타임프레임 (기본: 1h). FR 신호는 1h~4h 권장")
    p.add_argument("--interval",    type=int, default=60,
                   help="갱신 주기 (분). 기본=60 (timeframe과 동일 권장)")
    p.add_argument("--sync-candle", action="store_true",
                   help="봉 마감 직후 갱신 동기화 (더 정확한 신호)")

    # 신호 파라미터
    p.add_argument("--signals",     default=None,
                   help="활성 신호: fr,liq,cvd (기본: 전체). "
                        "FR=펀딩레이트극단 LIQ=청산캐스케이드 CVD=다이버전스")
    p.add_argument("--fr-z-thr",    type=float, default=2.5,
                   help="FR z-score 임계값 (기본: 2.5σ, 높을수록 극단적 신호만)")
    p.add_argument("--liq-z-thr",   type=float, default=1.5,
                   help="청산 z-score 임계값 (기본: 1.5σ)")
    p.add_argument("--cvd-div-thr", type=float, default=0.4,
                   help="CVD 다이버전스 임계값 (기본: 0.4, 0~1)")
    p.add_argument("--min-score",   type=float, default=0.5,
                   help="최소 진입 점수 합산 (기본: 0.5)")
    p.add_argument("--oi-poc-thr",  type=float, default=1.0,
                   help="OI POC 거리 임계값 (ATR 배수, 기본: 1.0)")
    p.add_argument("--oi-poc-window", type=int, default=100,
                   help="OI POC 롤링 윈도우 봉 수 (기본: 100)")

    # 방향 / 레짐 게이트
    p.add_argument("--trend-ema",   type=int, default=200,
                   help="추세 레짐 게이트 EMA 기간 (기본: 200, 0=비활성)")
    p.add_argument("--long-only",   action="store_true",
                   help="LONG 신호만 진입")
    p.add_argument("--short-only",  action="store_true",
                   help="SHORT 신호만 진입")

    # 리스크 파라미터 (표시용)
    p.add_argument("--tp-mult",     type=float, default=3.0,
                   help="TP = tp_mult × ATR (표시용, 기본: 3.0)")
    p.add_argument("--sl-mult",     type=float, default=1.0,
                   help="SL = sl_mult × ATR (표시용, 기본: 1.0)")

    args = p.parse_args()
    run_live(args)


if __name__ == "__main__":
    main()
