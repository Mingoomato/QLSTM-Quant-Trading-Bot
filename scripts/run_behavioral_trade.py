"""
run_behavioral_trade.py
──────────────────────────────────────────────────────────────────────
Behavioral Alpha Engine — Live / Paper Trader
FR_LONG + EMA200 long-only 전략 (1h 기본).

모드:
  paper (기본): 가상 포지션 추적, 실제 주문 없음.
  live        : Bybit V5 API로 실제 시장가 주문 + TP/SL 설정.

사용:
  # Paper 모드 (기본)
  python scripts/run_behavioral_trade.py

  # Live 모드 (주의: 실제 자금)
  python scripts/run_behavioral_trade.py --mode live

  # 커스텀 설정
  python scripts/run_behavioral_trade.py --mode paper \\
    --symbol BTCUSDT --timeframe 1h --capital 100 --leverage 5 \\
    --tp-mult 3.0 --sl-mult 1.0 --confidence 0.5 \\
    --fr-z-thr 2.5 --trend-ema 200

종료 (Ctrl+C): trades.csv + equity_curve.csv → reports/ 자동 저장
──────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Optional

import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Gemini (optional) ─────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

from src.data.data_client import DataClient
from src.data.bybit_mainnet import BybitMainnetClient
from src.models.features_v4 import build_features_v4
from src.models.features_v2 import compute_true_atr
from src.data.binance_client import BinancePublicClient

# BehavioralAlphaEngine
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)
from backtest_behavioral import BehavioralAlphaEngine, _load_quantum_agent, _quantum_p_long, _quantum_probs, _quantum_full_info  # noqa: E402

KST = timezone(timedelta(hours=9))

TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
}
LOOKBACK = 220
IDX_FR_Z      = 18
IDX_CVD_DELTA = 23
IDX_CVD_TREND = 24
IDX_CVD_DIV   = 25
IDX_LIQ_LONG  = 26
IDX_LIQ_SHORT = 27

# ── Gemini Gate ───────────────────────────────────────────────────────────────

_gemini_model_instance = None

def _init_gemini(model_name: str = "auto") -> bool:
    """GEMINI_API_KEY로 Gemini 초기화. 성공 시 True."""
    global _gemini_model_instance
    if not _GEMINI_AVAILABLE:
        print("  [!] google-generativeai 미설치. pip install google-generativeai")
        return False
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API")
               or os.getenv("GOOGLE_API_KEY"))
    if not api_key:
        print("  [!] GEMINI_API, GEMINI_API_KEY, GOOGLE_API_KEY 중 하나를 환경변수에 설정하세요.")
        return False
    genai.configure(api_key=api_key)

    # 자동 모델 선택: 사용 가능한 flash 모델 중 최신 선택
    if model_name == "auto":
        candidates = []
        try:
            for m in genai.list_models():
                n = m.name  # "models/gemini-2.0-flash" 형태
                if "generateContent" in (m.supported_generation_methods or []):
                    if "flash" in n.lower() or "pro" in n.lower():
                        candidates.append(n.replace("models/", ""))
        except Exception as e:
            print(f"  [!] 모델 목록 조회 실패: {e}")
        # 최신 버전 우선 정렬 (숫자 내림차순)
        candidates.sort(reverse=True)
        print(f"  [Gemini] 사용 가능 모델: {candidates[:5]}")
        model_name = candidates[0] if candidates else "gemini-1.5-flash"

    _gemini_model_instance = genai.GenerativeModel(model_name)
    print(f"  [Gemini] {model_name} 초기화 완료")
    return True


def _parse_gemini_json(text: str) -> dict:
    """Gemini 응답에서 JSON 추출 (markdown 코드블록 제거)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _call_gemini_exit(
    side: str,           # "long" | "short"
    entry_price: float,
    current_price: float,
    unrealized_roi: float,   # % (레버리지 반영)
    tp_price: float,
    sl_price: float,
    atr: float,
    ema200: float,
    fr_z: float,
    q_hold: float,
    q_long: float,
    q_short: float,
    min_confidence: float = 0.55,
) -> tuple[bool, float, str]:
    """
    포지션 보유 중 Gemini가 익절/손절 여부 판단.
    Returns: (should_exit: bool, confidence, reason)
    """
    if _gemini_model_instance is None:
        return False, 0.0, "Gemini 비활성"

    direction = "LONG" if side == "long" else "SHORT"
    price_move_pct = (current_price - entry_price) / entry_price * 100
    if side == "short":
        price_move_pct = -price_move_pct

    q_bias = "LONG" if q_long > q_short else "SHORT" if q_short > q_long else "NEUTRAL"

    prompt = f"""You are managing an open crypto futures position. Decide whether to EXIT now or HOLD.

=== OPEN POSITION ===
- Side       : {direction}
- Entry Price: {entry_price:,.2f} USDT
- Current    : {current_price:,.2f} USDT
- Price Move : {price_move_pct:+.2f}%  (in favor if positive)
- Unrealized : {unrealized_roi:+.2f}% ROI  (with leverage)
- TP Target  : {tp_price:,.2f} USDT
- SL Level   : {sl_price:,.2f} USDT

=== MARKET CONDITIONS ===
- ATR(14)       : {atr:.2f} USDT
- EMA200        : {ema200:,.2f}  (price {'above ↑BULL' if current_price > ema200 else 'below ↓BEAR'})
- Funding Rate z: {fr_z:.3f}  ({'shorts crowded' if fr_z < -1.5 else 'longs crowded' if fr_z > 1.5 else 'neutral'})

=== QUANTUM AI MODEL ===
- p_long={q_long:.3f}  p_short={q_short:.3f}  p_hold={q_hold:.3f}  → bias: {q_bias}

=== DECISION RULES ===
- EXIT if: Quantum bias has REVERSED against your position (e.g. holding LONG but q_short dominates)
- EXIT if: Unrealized ROI > 5% and momentum is fading
- EXIT if: Market structure clearly broken (EMA flip, FR reversal)
- HOLD if: Quantum still supports your direction and ROI < TP target
- HOLD if: Small floating loss but thesis intact

Respond ONLY with valid JSON (no markdown):
{{"action": "EXIT" or "HOLD", "confidence": 0.0-1.0, "reason": "one sentence"}}"""

    try:
        resp = _gemini_model_instance.generate_content(prompt)
        data = _parse_gemini_json(resp.text)
        action = str(data.get("action", "HOLD")).upper()
        conf   = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))
        should_exit = (action == "EXIT") and (conf >= min_confidence)
        return should_exit, conf, reason
    except Exception as e:
        print(f"  [!] Gemini exit 파싱 오류: {e}")
        return False, 0.0, "파싱 오류(fail-safe hold)"


def _call_gemini_gate(
    price: float,
    atr: float,
    fr_z: float,
    ema200: float,
    regime_long: bool,
    tp_price: float,
    sl_price: float,
    score: float,
    reasons: list,
    min_confidence: float = 0.55,
) -> tuple[bool, float, str]:
    """FR_LONG 신호 승인 게이트. Returns: (approve, confidence, reason)"""
    if _gemini_model_instance is None:
        return True, 1.0, "Gemini 비활성"

    prompt = f"""You are a professional crypto futures trader evaluating a potential LONG trade.

Current Market State (BTCUSDT 1h):
- Price: {price:,.2f} USDT
- ATR(14): {atr:.2f} USDT
- Funding Rate Z-score: {fr_z:.3f}  (signal: crowd over-short, squeeze candidate)
- EMA200: {ema200:,.2f}  (price {'ABOVE' if regime_long else 'BELOW'} EMA200)
- Trade Setup: LONG @ {price:,.2f}  TP={tp_price:,.2f}  SL={sl_price:,.2f}  (R:R={((tp_price-price)/max(price-sl_price,1)):.1f})
- Signal Score: {score:.2f}  Reasons: {reasons}

Strategy: Funding rate squeeze — crowd overly short → squeeze risk. Backtest WR=36.8%, R:R=3:1, ROI=+126% over 3.25y.

Respond ONLY with valid JSON:
{{"action": "BUY" or "SKIP", "confidence": 0.0-1.0, "reason": "one sentence"}}"""

    try:
        resp = _gemini_model_instance.generate_content(prompt)
        data = _parse_gemini_json(resp.text)
        action = str(data.get("action", "SKIP")).upper()
        conf   = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))
        return (action == "BUY") and (conf >= min_confidence), conf, reason
    except Exception as e:
        print(f"  [!] Gemini gate 파싱 오류: {e}")
        return True, 0.5, f"파싱 오류(fail-open)"


def _call_gemini_signal(
    symbol: str,
    timeframe: str,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    price: float,
    atr: float,
    fr_z: float,
    ema20: float,
    ema50: float,
    ema200: float,
    tp_mult: float,
    sl_mult: float,
    current_position: Optional[str],
    min_confidence: float = 0.55,
    q_hold: float = 0.0,
    q_long: float = 0.0,
    q_short: float = 0.0,
    q_info: Optional[dict] = None,
) -> tuple[str, float, str]:
    """
    Quantum 모델의 전체 내부 정보를 포함해 Gemini가 최종 LONG/SHORT/HOLD 결정.
    Returns: (action, confidence, reason)
    """
    if _gemini_model_instance is None:
        return "HOLD", 0.0, "Gemini 비활성"

    n = min(10, len(closes))
    candles_summary = []
    for i in range(-n, 0):
        candles_summary.append(
            f"  [{i}] O={closes[i-1] if i > -n else closes[i]:.0f} "
            f"H={highs[i]:.0f} L={lows[i]:.0f} C={closes[i]:.0f} "
            f"V={volumes[i]:.0f}"
        )
    candles_str = "\n".join(candles_summary)

    tp_price  = price * (1 + tp_mult * atr / price)
    sl_long   = price * (1 - sl_mult * atr / price)
    tp_short  = price * (1 - tp_mult * atr / price)
    sl_short  = price * (1 + sl_mult * atr / price)
    rr = tp_mult / sl_mult

    position_str = f"Currently: {current_position.upper() if current_position else 'FLAT (no position)'}"

    # ── Quantum 상세 정보 ──────────────────────────────────────────────────
    q_bias = "LONG" if q_long > q_short else "SHORT" if q_short > q_long else "NEUTRAL"
    q_str = (f"p_long={q_long:.3f}  p_short={q_short:.3f}  p_hold={q_hold:.3f}  "
             f"→ Quantum bias: {q_bias}")

    # 확장 정보 (select_action_verbose)
    if q_info:
        hurst       = q_info.get("hurst", 0.5)
        regime_prob = q_info.get("regime_prob", 0.0)
        conf_thr    = q_info.get("confidence_threshold", 0.5)
        fc_cr       = q_info.get("force_hold_cr", False)
        fc_ep       = q_info.get("force_hold_ep", False)
        fc_reg      = q_info.get("force_hold_regime", False)
        logit_m     = q_info.get("logit_margin", 0.0)
        cvd_delta   = q_info.get("cvd_delta", 0.0)
        liq_long    = q_info.get("liq_long", 0.0)
        liq_short   = q_info.get("liq_short", 0.0)

        hurst_interp = ("persistent/trending (H>0.55)" if hurst > 0.55
                        else "mean-reverting (H<0.45)" if hurst < 0.45
                        else "random walk (H≈0.5)")
        gates_str = []
        if fc_cr:    gates_str.append("CR-filter BLOCKED (insufficient edge)")
        if fc_ep:    gates_str.append("Entropy-production BLOCKED (near equilibrium)")
        if fc_reg:   gates_str.append("Lindblad BLOCKED (regime shift detected)")
        gates_str = ", ".join(gates_str) if gates_str else "all gates PASSED"

        quantum_extended = f"""
=== QUANTUM AI MODEL — FULL INTERNAL STATE ===
- Probabilities   : p_long={q_long:.3f}  p_short={q_short:.3f}  p_hold={q_hold:.3f}  → bias: {q_bias}
- Logit margin    : {logit_m:+.3f}  (LONG-SHORT raw logit diff; large |value| = stronger conviction)
- Hurst exponent  : H={hurst:.3f}  → {hurst_interp}
- Regime coherence: {regime_prob:.3f}  (Lindblad purity proxy; >0.7 = decoherent/regime-shift risk)
- Adaptive gate   : conf_threshold={conf_thr:.3f}  (Fisher-Rao; lower = model sees clearer edge)
- Gate decisions  : {gates_str}
- CVD delta       : {cvd_delta:+.3f}  (cumulative volume delta; + = net buy pressure)
- Liq pressure    : long_liq={liq_long:+.3f}  short_liq={liq_short:+.3f}
  (liquidation z-scores; high liq_long = forced long exits = downward pressure)"""
    else:
        quantum_extended = f"""
=== QUANTUM AI MODEL OUTPUT ===
- {q_str}
(28-feature quantum neural net: log-returns, volatility, OBI, funding rate, CVD, liquidation pressure)"""

    prompt = f"""You are an expert crypto futures trader with access to a proprietary Quantum AI model.
Your job: synthesize ALL signals and make the final trade decision.

=== RECENT CANDLES (last {n}h) ===
{candles_str}

=== TECHNICAL INDICATORS ===
- Current Price : {price:,.2f} USDT
- ATR(14)       : {atr:.2f} USDT  ({atr/price*100:.2f}% of price)
- EMA20         : {ema20:,.2f}  (price {'above' if price > ema20 else 'below'})
- EMA50         : {ema50:,.2f}  (price {'above' if price > ema50 else 'below'})
- EMA200        : {ema200:,.2f}  (price {'above ↑BULL' if price > ema200 else 'below ↓BEAR'})
- Funding Rate z: {fr_z:.3f}  ({'shorts crowded → squeeze risk' if fr_z < -1.5 else 'longs crowded → dump risk' if fr_z > 1.5 else 'neutral'})
{quantum_extended}

=== TRADE PARAMETERS ===
- If LONG : TP={tp_price:,.2f}  SL={sl_long:,.2f}  R:R={rr:.1f}:1
- If SHORT: TP={tp_short:,.2f}  SL={sl_short:,.2f}  R:R={rr:.1f}:1
- {position_str}

=== DECISION RULES ===
1. Weight the Quantum AI output heavily — it processes signals humans cannot see.
2. If ANY quantum gate is BLOCKED → strongly lean HOLD unless technicals are overwhelming.
3. High regime_prob (>0.5) = market regime shift risk → be cautious.
4. Hurst>0.55 favors trend-following; Hurst<0.45 favors mean-reversion.
5. Large |logit_margin| = high model conviction. Near 0 = uncertain.
6. Only override Quantum if technicals (EMA trend, funding rate) strongly disagree.
7. Target 1-2 trades per day — be decisive but not reckless.

Respond ONLY with valid JSON (no markdown):
{{"action": "LONG" or "SHORT" or "HOLD", "confidence": 0.0-1.0, "reason": "one sentence"}}"""

    try:
        resp = _gemini_model_instance.generate_content(prompt)
        data = _parse_gemini_json(resp.text)
        action = str(data.get("action", "HOLD")).upper()
        conf   = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))
        if action not in ("LONG", "SHORT", "HOLD"):
            action = "HOLD"
        if conf < min_confidence:
            action = "HOLD"
        return action, conf, reason
    except Exception as e:
        print(f"  [!] Gemini signal 파싱 오류: {e}")
        return "HOLD", 0.0, f"파싱 오류: {e}"


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _ema_last(arr: np.ndarray, period: int) -> float:
    if len(arr) == 0:
        return 0.0
    alpha = 2.0 / (period + 1.0)
    val = float(arr[0])
    for x in arr[1:]:
        val = alpha * float(x) + (1.0 - alpha) * val
    return val


def _seconds_to_next_candle(timeframe: str) -> int:
    tf_sec = TF_MINUTES.get(timeframe, 60) * 60
    now_ts = int(time.time())
    return tf_sec - (now_ts % tf_sec)


def _now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def _price_str(price: float, decimals: int = 2) -> str:
    return f"{price:,.{decimals}f}"


# ── 데이터 수집 (run_behavioral_live.py와 동일 로직) ────────────────────────

def fetch_live_window(
    symbol: str,
    timeframe: str,
    dc: DataClient,
    binance: BinancePublicClient,
) -> Optional[pd.DataFrame]:
    tf_min = TF_MINUTES.get(timeframe, 60)
    end_dt = datetime.now(timezone.utc)
    buf_dt = end_dt - timedelta(minutes=tf_min * LOOKBACK * 2 + 60)
    start_str = buf_dt.strftime("%Y-%m-%d")
    end_ms = int(end_dt.timestamp() * 1000)
    _start_ms = int(buf_dt.timestamp() * 1000)

    try:
        df = dc.fetch_training_history(
            symbol=symbol, timeframe=timeframe,
            start_date=start_str, end_ms=end_ms, cache_dir="data",
        )
    except Exception as e:
        print(f"  [!] Bybit OHLCV error: {e}")
        return None
    if df is None or len(df) < 50:
        return None
    df = df.tail(LOOKBACK).copy().reset_index(drop=True)

    # Funding rate
    try:
        df_fr = dc.fetch_funding_history(symbol, _start_ms, end_ms, cache_dir="data")
        if not df_fr.empty:
            df_fr["ts"] = (
                pd.to_datetime(df_fr["ts_ms"], unit="ms", utc=True)
                .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
            )
            df = df.merge(df_fr[["ts", "funding_rate"]], on="ts", how="left")
            df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
        else:
            df["funding_rate"] = 0.0
    except Exception:
        df["funding_rate"] = 0.0

    # Open Interest
    try:
        oi_iv = "1h" if tf_min >= 60 else ("15min" if tf_min >= 15 else "5min")
        df_oi = dc.fetch_open_interest_history(
            symbol, _start_ms, end_ms, interval=oi_iv, cache_dir="data"
        )
        if not df_oi.empty:
            df_oi["ts"] = (
                pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
                .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
            )
            df = df.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
            df["open_interest"] = df["open_interest"].ffill().fillna(0.0)
        else:
            df["open_interest"] = 0.0
    except Exception:
        df["open_interest"] = 0.0

    # Binance CVD (taker volume)
    try:
        end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        start_str2 = buf_dt.strftime("%Y-%m-%d")
        df_cvd = binance.fetch_klines_with_taker(
            symbol, timeframe, start_str2, end_str, verbose=False
        )
        if not df_cvd.empty:
            df = df.merge(
                df_cvd[["ts", "taker_buy_volume", "real_delta"]], on="ts", how="left"
            )
            df["taker_buy_volume"] = df["taker_buy_volume"].ffill().fillna(0.0)
            df["real_delta"] = df["real_delta"].ffill().fillna(0.0)
        else:
            df["taker_buy_volume"] = 0.0
            df["real_delta"] = 0.0
    except Exception:
        df["taker_buy_volume"] = 0.0
        df["real_delta"] = 0.0

    return df


# ── 포지션 크기 계산 ──────────────────────────────────────────────────────────

def _calc_qty(
    capital_usdt: float,
    price: float,
    leverage: int,
    pos_frac: float = 1.0,
) -> str:
    """
    notional = capital × pos_frac × leverage
    qty (BTC) = notional / price
    Bybit 최소 주문: BTCUSDT 0.001 BTC
    """
    notional = capital_usdt * pos_frac * leverage
    qty = notional / price
    # floor (절삭) — 반올림 시 마진 초과 방지. Bybit BTCUSDT min=0.001
    qty = math.floor(qty * 1000) / 1000
    qty = max(0.001, qty)
    return f"{qty:.3f}"


# ── 보고서 저장 ───────────────────────────────────────────────────────────────

def save_reports(trades: list[dict], equity_curve: list[dict], output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    trades_path = os.path.join(output_dir, f"behavioral_trades_{ts_tag}.csv")
    equity_path = os.path.join(output_dir, f"behavioral_equity_{ts_tag}.csv")

    if trades:
        keys = list(trades[0].keys())
        with open(trades_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(trades)
        print(f"\n[SAVE] Trades → {trades_path}")

    if equity_curve:
        keys = list(equity_curve[0].keys())
        with open(equity_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(equity_curve)
        print(f"[SAVE] Equity → {equity_path}")


# ── 메인 트레이더 ─────────────────────────────────────────────────────────────

class BehavioralTrader:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.mode = args.mode.lower()
        self.live = (self.mode == "live")

        # 설정 인수
        self.symbol    = args.symbol
        self.timeframe = args.timeframe
        self.leverage  = args.leverage
        self.capital   = args.capital
        self.pos_frac   = args.pos_frac
        self.tp_mult    = args.tp_mult
        self.sl_mult    = args.sl_mult
        self.tp_roi_pct = args.tp_roi   # ROI 기준 TP (%)
        self.sl_roi_pct = args.sl_roi   # ROI 기준 SL (%)
        self.ema_period = args.trend_ema
        self.fr_z_thr   = args.fr_z_thr
        self.min_score  = args.min_score

        # 데이터 클라이언트
        self.dc      = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")
        self.binance = BinancePublicClient()

        # 신호 엔진 (long_only/trend_ema는 trader 레벨에서 처리)
        self.engine = BehavioralAlphaEngine(
            active_signals={"fr"},
            fr_z_thr=self.fr_z_thr,
            liq_z_thr=args.liq_z_thr,
            cvd_div_thr=args.cvd_div_thr,
            oi_poc_thr=args.oi_poc_thr,
        )

        # 폴링 설정
        self.interval_min = args.interval
        self.lookback     = args.lookback

        # Quantum — gemini-signal 모드에서는 항상 로드 (데이터 공급자로 사용)
        self.q_gate       = args.quantum_gate
        self.q_confidence = args.q_confidence
        self.q_window     = args.q_window
        self.model_path   = args.model_path
        if self.q_gate or args.gemini_signal:
            _load_quantum_agent(args.model_path)

        # Gemini Gate / Signal
        self.g_gate       = args.gemini_gate
        self.g_signal     = args.gemini_signal   # 주 신호 모드
        self.g_confidence = args.gemini_confidence
        self.g_model_name = args.gemini_model
        if self.g_gate or self.g_signal:
            ok = _init_gemini(self.g_model_name)
            if not ok:
                print("  [!] Gemini 초기화 실패 — Gemini 기능 비활성화")
                self.g_gate = False
                self.g_signal = False

        # 상태 머신
        self.equity          = float(self.capital)
        self.position        = None   # None = FLAT
        self.last_signal_ts  = ""    # 중복 진입 방지용 마지막 신호 봉 타임스탬프
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []

        # Live 클라이언트 (live 모드 시)
        self.bybit_exec = None
        if self.live:
            from src.data.bybit_mainnet import BybitMainnetClient
            self.bybit_exec = BybitMainnetClient()
            self._check_api_keys()
            self._sync_balance()   # 실제 계좌 잔고로 equity 동기화

        # Ctrl+C 핸들러
        signal.signal(signal.SIGINT, self._on_exit)

    def _check_api_keys(self):
        key    = os.getenv("BYBIT_API_KEY", "")
        secret = os.getenv("BYBIT_API_SECRET", "")
        if not key or not secret:
            print("\n[ERROR] BYBIT_API_KEY 또는 BYBIT_API_SECRET 이 .env에 없습니다.")
            print("  Live 모드를 사용하려면 .env에 두 변수를 설정하세요.")
            sys.exit(1)
        print(f"[OK] API 키 확인: {key[:8]}...{key[-4:]} (secret=****)")

    def _sync_balance(self):
        """Bybit 실제 USDT 잔고를 읽어 self.equity / self.capital 동기화."""
        try:
            wallet = self.bybit_exec.get_wallet_balance("UNIFIED")
            coins = wallet.get("coin", [])
            usdt_info = next((c for c in coins if c.get("coin") == "USDT"), None)
            if usdt_info is None:
                # CONTRACT 계좌 시도
                wallet2 = self.bybit_exec.get_wallet_balance("CONTRACT")
                coins2 = wallet2.get("coin", [])
                usdt_info = next((c for c in coins2 if c.get("coin") == "USDT"), None)
            if usdt_info:
                avail = float(usdt_info.get("availableToWithdraw")
                              or usdt_info.get("walletBalance", 0))
                self.equity  = avail
                self.capital = avail
                print(f"[OK] 계좌 잔고 동기화: {avail:.4f} USDT")
            else:
                print(f"[!] USDT 잔고 조회 실패 — --capital {self.capital} 사용")
        except Exception as e:
            print(f"[!] 잔고 조회 오류: {e} — --capital {self.capital} 사용")

    def _on_exit(self, *_):
        print("\n[STOP] 종료 중 — 포지션 정리 후 보고서 저장...")
        self._close_open_position(reason="MANUAL_EXIT")
        self._append_equity()
        save_reports(self.trades, self.equity_curve)
        sys.exit(0)

    def _roi_tp_sl(self, price: float, side: str) -> tuple[float, float]:
        """ROI 기준 TP/SL 가격 계산.
        tp_roi_pct=10, sl_roi_pct=5, leverage=10 →
          LONG : TP=price×1.01, SL=price×0.995
          SHORT: TP=price×0.99, SL=price×1.005
        """
        tp_move = self.tp_roi_pct / 100.0 / self.leverage
        sl_move = self.sl_roi_pct / 100.0 / self.leverage
        if side == "long":
            return price * (1 + tp_move), price * (1 - sl_move)
        else:
            return price * (1 - tp_move), price * (1 + sl_move)

    def _check_live_position_closed(self) -> bool:
        """Live 모드: Bybit에서 포지션이 사라졌으면 True 반환 후 로컬 상태 정리."""
        if not self.live or self.bybit_exec is None or self.position is None:
            return False
        try:
            positions = self.bybit_exec.get_positions(self.symbol)
            # size=0 또는 목록 없으면 청산됨
            open_pos = [p for p in positions if float(p.get("size", 0)) > 0]
            if not open_pos:
                print(f"  [BYBIT] 포지션 청산 감지 (TP/SL 도달) — 로컬 상태 초기화")
                self._sync_balance()
                self.position = None
                self.last_signal_ts = ""   # 즉시 재진입 허용
                return True
        except Exception as e:
            print(f"  [!] 포지션 조회 오류: {e}")
        return False

    def _place_order_with_retry(
        self, side: str, qty_str: str, tp_price: float, sl_price: float,
        max_retries: int = 5, delay: float = 3.0
    ) -> Optional[str]:
        """주문 실패 시 max_retries회까지 재시도. 성공 시 orderId 반환, 실패 시 None."""
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.bybit_exec.place_order(
                    symbol=self.symbol, side=side, qty=qty_str,
                    order_type="Market",
                    tp_price=f"{tp_price:.2f}",
                    sl_price=f"{sl_price:.2f}",
                )
                if resp.get("retCode", -1) == 0:
                    return resp.get("result", {}).get("orderId", "?")
                msg = resp.get("retMsg", "unknown")
                print(f"  [!] 주문 실패 (시도 {attempt}/{max_retries}): {msg}")
            except Exception as e:
                print(f"  [!] 주문 오류 (시도 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"  [RETRY] {delay:.0f}초 후 재시도...")
                time.sleep(delay)
        print(f"  [!] 주문 {max_retries}회 모두 실패 — 이번 신호 스킵")
        return None

    # ── 포지션 진입 ─────────────────────────────────────────────────────────

    def _enter_long(self, price: float, atr: float):
        # 주문 직전 최신 가격으로 갱신
        live_price = self._get_live_price()
        if live_price:
            print(f"  [PRICE] 분석가={_price_str(price)} → 현재가={_price_str(live_price)}")
            price = live_price
        tp_price, sl_price = self._roi_tp_sl(price, "long")

        if self.live and self.bybit_exec is not None:
            try:
                self.bybit_exec.set_leverage(self.symbol, self.leverage)
            except Exception as e:
                print(f"  [!] 레버리지 설정 실패: {e}")

            qty_str = _calc_qty(self.equity, price, self.leverage, self.pos_frac)
            order_id = self._place_order_with_retry("Buy", qty_str, tp_price, sl_price)
            if order_id is None:
                return
            print(f"  [BUY] 주문 체결: qty={qty_str}  orderId={order_id}")
        else:
            qty_str = _calc_qty(self.equity, price, self.leverage, self.pos_frac)
            print(f"  [PAPER BUY] qty={qty_str} @ {_price_str(price)}")

        notional = float(qty_str) * price
        fee = notional * 0.00055  # taker fee 0.055%

        self.position = {
            "side":       "long",
            "entry":      price,
            "qty":        float(qty_str),
            "notional":   notional,
            "tp":         tp_price,
            "sl":         sl_price,
            "atr":        atr,
            "entry_time": _now_kst(),
            "fee_open":   fee,
        }

        print(
            f"  TP={_price_str(tp_price)}  SL={_price_str(sl_price)}"
            f"  ATR={atr:.2f}  Notional={notional:.1f}U"
        )

    # ── 포지션 청산 ─────────────────────────────────────────────────────────

    def _close_open_position(self, reason: str, exit_price: Optional[float] = None):
        if self.position is None:
            return

        p = self.position
        if exit_price is None:
            # 현재가로 청산 (live → 실제 시장가)
            exit_price = self._get_last_price()
            if exit_price is None:
                exit_price = p["entry"]  # fallback

        if self.live and self.bybit_exec is not None:
            resp = self.bybit_exec.place_order(
                symbol=self.symbol,
                side="Sell",
                qty=f"{p['qty']:.3f}",
                order_type="Market",
                reduce_only=True,
            )
            if resp.get("retCode", -1) != 0:
                print(f"  [!] 청산 실패: {resp.get('retMsg')}")

        # LONG: profit when price rises / SHORT: profit when price falls
        direction_mult = 1.0 if p["side"] == "long" else -1.0
        pnl_pct = direction_mult * (exit_price - p["entry"]) / p["entry"] * self.leverage
        pnl_usdt = pnl_pct * self.equity * self.pos_frac
        fee_close = p["qty"] * exit_price * 0.00055
        net_pnl = pnl_usdt - p["fee_open"] - fee_close

        self.equity += net_pnl
        self.equity = max(self.equity, 0.0)

        trade = {
            "entry_time":  p["entry_time"],
            "exit_time":   _now_kst(),
            "side":        p["side"],
            "entry_price": round(p["entry"], 2),
            "exit_price":  round(exit_price, 2),
            "qty":         p["qty"],
            "tp":          round(p["tp"], 2),
            "sl":          round(p["sl"], 2),
            "reason":      reason,
            "pnl_usdt":    round(net_pnl, 4),
            "pnl_pct":     round(pnl_pct * 100, 2),
            "equity":      round(self.equity, 4),
        }
        self.trades.append(trade)

        sign = "+" if net_pnl >= 0 else ""
        print(
            f"  [{reason}] exit={_price_str(exit_price)}"
            f"  PnL={sign}{net_pnl:.2f}U ({sign}{pnl_pct*100:.1f}%)"
            f"  equity={self.equity:.2f}U"
        )
        self.position = None

    # ── SHORT 진입 ──────────────────────────────────────────────────────────

    def _enter_short(self, price: float, atr: float):
        # 주문 직전 최신 가격으로 갱신
        live_price = self._get_live_price()
        if live_price:
            print(f"  [PRICE] 분석가={_price_str(price)} → 현재가={_price_str(live_price)}")
            price = live_price
        tp_price, sl_price = self._roi_tp_sl(price, "short")

        if self.live and self.bybit_exec is not None:
            try:
                self.bybit_exec.set_leverage(self.symbol, self.leverage)
            except Exception as e:
                print(f"  [!] 레버리지 설정 실패: {e}")
            qty_str = _calc_qty(self.equity, price, self.leverage, self.pos_frac)
            order_id = self._place_order_with_retry("Sell", qty_str, tp_price, sl_price)
            if order_id is None:
                return
            print(f"  [SHORT] 주문 체결: qty={qty_str}  orderId={order_id}")
        else:
            qty_str = _calc_qty(self.equity, price, self.leverage, self.pos_frac)
            print(f"  [PAPER SHORT] qty={qty_str} @ {_price_str(price)}")

        notional = float(qty_str) * price
        fee = notional * 0.00055
        self.position = {
            "side": "short", "entry": price, "qty": float(qty_str),
            "notional": notional, "tp": tp_price, "sl": sl_price,
            "atr": atr, "entry_time": _now_kst(), "fee_open": fee,
        }
        print(f"  TP={_price_str(tp_price)}  SL={_price_str(sl_price)}  ATR={atr:.2f}  Notional={notional:.1f}U")

    # ── TP/SL 모니터링 (paper) ────────────────────────────────────────────

    def _check_exit(self, high: float, low: float):
        if self.position is None:
            return
        p = self.position
        if p["side"] == "long":
            if high >= p["tp"]:
                self._close_open_position("TP", p["tp"])
            elif low <= p["sl"]:
                self._close_open_position("SL", p["sl"])
        elif p["side"] == "short":
            if low <= p["tp"]:
                self._close_open_position("TP", p["tp"])
            elif high >= p["sl"]:
                self._close_open_position("SL", p["sl"])

    # ── 마지막 현재가 조회 ─────────────────────────────────────────────────

    def _get_live_price(self) -> Optional[float]:
        """Bybit ticker에서 실시간 최신 가격 조회 (OHLCV보다 빠름)."""
        try:
            import requests
            url = "https://api.bybit.com/v5/market/tickers"
            r = requests.get(url, params={"category": "linear", "symbol": self.symbol}, timeout=5)
            data = r.json()
            items = data.get("result", {}).get("list", [])
            if items:
                return float(items[0].get("lastPrice", 0))
        except Exception:
            pass
        # fallback: OHLCV
        try:
            from src.data.bybit_mainnet import BybitMainnetClient as _BM
            df = _BM().fetch_ohlcv(self.symbol, self.timeframe, days_back=1)
            if df is not None and not df.empty:
                return float(df["close"].iloc[-1])
        except Exception:
            pass
        return None

    def _get_last_price(self) -> Optional[float]:
        return self._get_live_price()

    # ── 지분 곡선 기록 ────────────────────────────────────────────────────

    def _append_equity(self):
        self.equity_curve.append({
            "time":   _now_kst(),
            "equity": round(self.equity, 4),
            "trades": len(self.trades),
        })

    # ── 메인 루프 ─────────────────────────────────────────────────────────

    def run(self):
        banner = "LIVE" if self.live else "PAPER"
        print("=" * 65)
        print(f"  Behavioral Alpha Trader [{banner}]")
        print(f"  {self.symbol} {self.timeframe}  |  "
              f"capital={self.capital}U  leverage={self.leverage}x")
        print(f"  TP=+{self.tp_roi_pct}% ROI  SL=-{self.sl_roi_pct}% ROI  "
              f"FR_z>{self.fr_z_thr}  EMA{self.ema_period}")
        print(f"  폴링={self.interval_min}분마다  분석={self.lookback}봉")
        if self.q_gate:
            print(f"  QuantumGate=ON  conf≥{self.q_confidence}  window={self.q_window}봉")
        if self.g_gate:
            print(f"  GeminiGate=ON  model={self.g_model_name}  conf≥{self.g_confidence}")
        if self.g_signal:
            print(f"  GeminiSignal=ON  model={self.g_model_name}  conf≥{self.g_confidence}  (주 신호 — FR 조건 무시)")
        if self.live:
            print("  *** 실제 자금이 사용됩니다. Ctrl+C로 즉시 종료 가능 ***")
        print("=" * 65)

        while True:
            print(f"\n[{_now_kst()}] 분석 중... (최근 {self.lookback}봉)")

            # 폴링 — 봉 마감 대기 없음

            # ── 데이터 수집 ────────────────────────────────────────────────
            df = fetch_live_window(self.symbol, self.timeframe, self.dc, self.binance)
            if df is None or len(df) < 50:
                print("  [!] 데이터 부족 — 스킵")
                time.sleep(self.interval_min * 60)
                continue

            # 최근 lookback봉만 분석 (마지막 봉은 현재 미완성 봉 → 제외)
            df_analysis = df.iloc[-(self.lookback + 1):-1].copy().reset_index(drop=True)
            if len(df_analysis) < 20:
                print("  [!] 분석 봉 부족 — 스킵")
                time.sleep(self.interval_min * 60)
                continue

            # 신호 봉 타임스탬프 (마지막 확정 봉)
            signal_ts = str(df_analysis["ts"].iloc[-1])

            # ── ATR 계산 ─────────────────────────────────────────────────
            highs  = df_analysis["high"].values.astype(np.float64)
            lows   = df_analysis["low"].values.astype(np.float64)
            closes = df_analysis["close"].values.astype(np.float64)
            atr14  = compute_true_atr(highs, lows, closes, period=14)
            atr    = float(atr14[-1])

            # ── EMA200 레짐 (전체 df 기준 — 더 긴 히스토리 사용) ──────────
            ema200 = _ema_last(df["close"].values.astype(np.float64), self.ema_period)
            current_price = float(closes[-1])
            current_high  = float(highs[-1])
            current_low   = float(lows[-1])
            regime_long   = current_price > ema200

            # ── V4 피처 & 신호 ────────────────────────────────────────────
            try:
                feat_arr = build_features_v4(df_analysis)
                if feat_arr is None or len(feat_arr) == 0:
                    feat = None
                elif hasattr(feat_arr, "ndim") and feat_arr.ndim == 2:
                    feat = feat_arr[-1]   # (N, 28) → (28,)
                elif hasattr(feat_arr, "ndim") and feat_arr.ndim == 1:
                    feat = feat_arr       # 이미 (28,)
                else:
                    feat = feat_arr[-1]
            except Exception as e:
                print(f"  [!] 피처 빌드 오류: {e}")
                feat = None

            if feat is None:
                print("  [!] 피처 없음 — 스킵")
                time.sleep(self.interval_min * 60)
                continue

            direction, score, reasons = self.engine.score(feat)

            # ── 현재 상태 출력 ────────────────────────────────────────────
            fr_z = float(feat[IDX_FR_Z]) if len(feat) > IDX_FR_Z else 0.0
            cvd_d = float(feat[IDX_CVD_DELTA]) if len(feat) > IDX_CVD_DELTA else 0.0

            regime_str = f"EMA{self.ema_period}={'↑BULL' if regime_long else '↓BEAR'}"
            pos_str = "FLAT"
            if self.position:
                p = self.position
                upnl = (current_price - p["entry"]) / p["entry"] * self.leverage
                pos_str = (
                    f"LONG @ {_price_str(p['entry'])}  "
                    f"uPnL={'+' if upnl>=0 else ''}{upnl*100:.1f}%  "
                    f"TP={_price_str(p['tp'])}  SL={_price_str(p['sl'])}"
                )

            print(
                f"  price={_price_str(current_price)}  ATR={atr:.1f}  "
                f"FR_z={fr_z:.2f}  CVD_d={cvd_d:.3f}"
            )
            print(f"  {regime_str}  |  Score={score:.2f}  Dir={direction}  Reasons={reasons}")
            print(f"  Position: {pos_str}")
            print(f"  Equity: {self.equity:.2f}U  |  Trades: {len(self.trades)}")

            # ── TP/SL 체크 ───────────────────────────────────────────────
            if self.live:
                self._check_live_position_closed()   # Bybit 포지션 청산 감지
            else:
                self._check_exit(current_high, current_low)

            # ── 진입 신호 체크 ────────────────────────────────────────────
            if self.position is None:

                # ══════════════════════════════════════════════════════════
                # 모드 A: Gemini Signal (주 신호 — FR 조건 무시, LONG/SHORT)
                # ══════════════════════════════════════════════════════════
                if self.g_signal and signal_ts != self.last_signal_ts:
                    ema20  = _ema_last(df["close"].values.astype(np.float64), 20)
                    ema50  = _ema_last(df["close"].values.astype(np.float64), 50)
                    volumes = df_analysis["volume"].values.astype(np.float64)
                    cur_pos = self.position["side"] if self.position else None

                    # ── Quantum 분석 (항상 실행, 데이터 공급) ────────────
                    atr_norm = atr / max(current_price, 1e-8)
                    q_info = _quantum_full_info(
                        feat_arr, window=self.q_window, atr_norm=atr_norm
                    )
                    q_hold  = q_info["p_hold"]
                    q_long  = q_info["p_long"]
                    q_short = q_info["p_short"]
                    print(
                        f"  [QUANTUM] p_long={q_long:.3f}  p_short={q_short:.3f}  "
                        f"p_hold={q_hold:.3f}  H={q_info['hurst']:.3f}  "
                        f"regime={q_info['regime_prob']:.3f}  "
                        f"logit_margin={q_info['logit_margin']:+.3f}"
                    )

                    # ── Gemini 최종 결정 (Quantum 전체 내부 정보 포함) ───
                    g_action, g_conf, g_reason = _call_gemini_signal(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        closes=closes,
                        highs=highs,
                        lows=lows,
                        volumes=volumes,
                        price=current_price,
                        atr=atr,
                        fr_z=fr_z,
                        ema20=ema20,
                        ema50=ema50,
                        ema200=ema200,
                        tp_mult=self.tp_mult,
                        sl_mult=self.sl_mult,
                        current_position=cur_pos,
                        min_confidence=self.g_confidence,
                        q_hold=q_hold,
                        q_long=q_long,
                        q_short=q_short,
                        q_info=q_info,
                    )
                    print(f"  [GEMINI] action={g_action}  conf={g_conf:.2f}  reason={g_reason}")

                    if g_action == "LONG":
                        print(f"  [SIGNAL] LONG+GeminiSignal → 진입")
                        self._enter_long(current_price, atr)
                        self.last_signal_ts = signal_ts
                    elif g_action == "SHORT":
                        print(f"  [SIGNAL] SHORT+GeminiSignal → 진입")
                        self._enter_short(current_price, atr)
                        self.last_signal_ts = signal_ts
                    else:
                        print(f"  [HOLD] Gemini: 관망")

                # ══════════════════════════════════════════════════════════
                # 모드 B: FR_LONG (기존 규칙 기반) + 선택적 Quantum/Gemini gate
                # ══════════════════════════════════════════════════════════
                elif not self.g_signal:
                    if direction == 1 and score >= self.min_score:
                        if not regime_long:
                            print(f"  [SKIP] EMA{self.ema_period} 아래 → 롱 진입 차단")
                        elif signal_ts == self.last_signal_ts:
                            print(f"  [SKIP] 같은 봉({signal_ts}) 중복 진입 방지")
                        else:
                            # ── Quantum gate (선택) ───────────────────────
                            q_ok = True
                            if self.q_gate:
                                feat_arr2 = build_features_v4(df_analysis)
                                atr_norm = atr / max(current_price, 1e-8)
                                pl = _quantum_p_long(feat_arr2, window=self.q_window, atr_norm=atr_norm)
                                print(f"  [QUANTUM] p_long={pl:.3f}  threshold={self.q_confidence}")
                                if pl < self.q_confidence:
                                    print(f"  [SKIP] Quantum gate 미달")
                                    q_ok = False
                                else:
                                    print(f"  [QUANTUM] OK  p_long={pl:.3f}")

                            # ── Gemini gate (선택) ────────────────────────
                            g_ok = True
                            if q_ok and self.g_gate:
                                tp_p = current_price * (1 + self.tp_mult * atr / current_price)
                                sl_p = current_price * (1 - self.sl_mult * atr / current_price)
                                approve, g_conf, g_reason = _call_gemini_gate(
                                    price=current_price, atr=atr, fr_z=fr_z,
                                    ema200=ema200, regime_long=regime_long,
                                    tp_price=tp_p, sl_price=sl_p, score=score,
                                    reasons=reasons, min_confidence=self.g_confidence,
                                )
                                print(f"  [GEMINI] action={'BUY' if approve else 'SKIP'}  conf={g_conf:.2f}  reason={g_reason}")
                                if not approve:
                                    print(f"  [SKIP] Gemini gate 거부")
                                    g_ok = False

                            # ── 진입 ─────────────────────────────────────
                            if q_ok and g_ok:
                                gates_str = ("+Quantum" if self.q_gate else "") + ("+Gemini" if self.g_gate else "")
                                print(f"  [SIGNAL] LONG{gates_str} (score={score:.2f}) → 진입")
                                self._enter_long(current_price, atr)
                                self.last_signal_ts = signal_ts
                    else:
                        print(f"  [FLAT] FR 신호 없음 (dir={direction}, score={score:.2f})")

            else:
                # ── 포지션 보유 중: Gemini 익절 판단 ────────────────────
                p = self.position
                direction_mult = 1.0 if p["side"] == "long" else -1.0
                unrealized_roi = direction_mult * (current_price - p["entry"]) / p["entry"] * self.leverage * 100

                print(f"  [HOLD] {p['side'].upper()} @ {_price_str(p['entry'])}"
                      f"  uROI={unrealized_roi:+.1f}%"
                      f"  TP={_price_str(p['tp'])}  SL={_price_str(p['sl'])}")

                if self.g_signal:
                    # Quantum 현재 상태
                    atr_norm = atr / max(current_price, 1e-8)
                    q_hold, q_long, q_short = _quantum_probs(
                        feat_arr, window=self.q_window, atr_norm=atr_norm
                    )
                    print(f"  [QUANTUM] p_long={q_long:.3f}  p_short={q_short:.3f}  p_hold={q_hold:.3f}")

                    should_exit, g_conf, g_reason = _call_gemini_exit(
                        side=p["side"],
                        entry_price=p["entry"],
                        current_price=current_price,
                        unrealized_roi=unrealized_roi,
                        tp_price=p["tp"],
                        sl_price=p["sl"],
                        atr=atr,
                        ema200=ema200,
                        fr_z=fr_z,
                        q_hold=q_hold,
                        q_long=q_long,
                        q_short=q_short,
                        min_confidence=self.g_confidence,
                    )
                    print(f"  [GEMINI EXIT] action={'EXIT' if should_exit else 'HOLD'}  conf={g_conf:.2f}  reason={g_reason}")

                    if should_exit:
                        self._close_open_position(reason="GEMINI_EXIT", exit_price=current_price)
                        self.last_signal_ts = ""  # 즉시 재진입 허용

            # ── 지분 기록 ────────────────────────────────────────────────
            self._append_equity()

            # 통계 요약 (10거래마다)
            if len(self.trades) > 0 and len(self.trades) % 10 == 0:
                wins = sum(1 for t in self.trades if t["pnl_usdt"] > 0)
                wr = wins / len(self.trades) * 100
                total_pnl = sum(t["pnl_usdt"] for t in self.trades)
                print(
                    f"\n  ── 중간 통계 ──────────────────────────────────"
                    f"\n  거래: {len(self.trades)}  WR: {wr:.1f}%"
                    f"  총PnL: {total_pnl:+.2f}U  자산: {self.equity:.2f}U"
                )

            # ── 다음 폴링까지 대기 ────────────────────────────────────────
            print(f"  [{_now_kst()}] {self.interval_min}분 후 재분석...")
            time.sleep(self.interval_min * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Behavioral Alpha Trader (FR_LONG)")
    p.add_argument("--mode",       default="paper",    choices=["paper", "live"],
                   help="paper=가상거래(기본) / live=실제거래")
    p.add_argument("--symbol",     default="BTCUSDT")
    p.add_argument("--timeframe",  default="1h",
                   choices=["1m","5m","15m","30m","1h","4h"])
    p.add_argument("--capital",    type=float, default=100.0,
                   help="초기 자본 (USDT, paper) / 실제 계좌잔고 기준 거래 크기 (live)")
    p.add_argument("--leverage",   type=int,   default=5)
    p.add_argument("--pos-frac",   type=float, default=1.0,
                   help="포지션 크기 = capital × pos_frac × leverage")
    p.add_argument("--tp-mult",    type=float, default=3.0, help="TP = tp_mult × ATR (ATR 모드 시)")
    p.add_argument("--sl-mult",    type=float, default=1.0, help="SL = sl_mult × ATR (ATR 모드 시)")
    p.add_argument("--tp-roi",     type=float, default=10.0,
                   help="TP 목표 ROI %%. 예: 10 → 레버리지 감안 가격 +1%% 이동. 기본=10")
    p.add_argument("--sl-roi",     type=float, default=5.0,
                   help="SL 손실 ROI %%. 예: 5 → 레버리지 감안 가격 -0.5%% 이동. 기본=5")
    p.add_argument("--fr-z-thr",   type=float, default=2.5, help="FR z-score 임계값")
    p.add_argument("--min-score",  type=float, default=0.5, help="최소 신호 점수")
    p.add_argument("--trend-ema",  type=int,   default=200, help="EMA 레짐 필터 기간")
    p.add_argument("--liq-z-thr",  type=float, default=2.0)
    p.add_argument("--cvd-div-thr",type=float, default=0.5)
    p.add_argument("--oi-poc-thr", type=float, default=1.0)
    p.add_argument("--quantum-gate",  action="store_true",
                   help="FR 신호 시 quantum 모델 p_long 추가 확인")
    p.add_argument("--model-path",  default="checkpoints/quantum_v2/agent_best_fold10.pt")
    p.add_argument("--q-confidence", type=float, default=0.45)
    p.add_argument("--q-window",     type=int,   default=20)
    # Gemini gate / signal
    p.add_argument("--gemini-gate",  action="store_true",
                   help="FR 신호 시 Gemini LLM 최종 판단 게이트 추가")
    p.add_argument("--gemini-signal", action="store_true",
                   help="Gemini를 주 신호 생성기로 사용 (FR 조건 무시, LONG/SHORT 모두 가능, 하루 1-2회 목표)")
    p.add_argument("--gemini-model", default="auto",
                   help="Gemini 모델명. 기본=auto (사용 가능한 최신 모델 자동 선택)")
    p.add_argument("--gemini-confidence", type=float, default=0.55,
                   help="Gemini 최소 confidence. 기본=0.55")
    p.add_argument("--interval",   type=int,   default=5,
                   help="폴링 간격 (분). 기본=5분마다 재분석")
    p.add_argument("--lookback",   type=int,   default=30,
                   help="분석할 최근 확정 봉 수. 기본=30봉")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.mode == "live":
        print("\n" + "!"*65)
        print("  경고: LIVE 모드 — 실제 자금으로 주문이 실행됩니다.")
        print("  계속하려면 'LIVE'를 입력하세요 (다른 입력 → 취소):")
        print("!" * 65)
        confirm = input("  > ").strip()
        if confirm != "LIVE":
            print("  취소됨. Paper 모드로 실행하려면 --mode paper (기본값)")
            sys.exit(0)

    trader = BehavioralTrader(args)
    trader.run()
