"""
features_v4.py
────────────────────────────────────────────────────────────────────────────
V4 Feature Pipeline: 28-dimensional features

Extends V3 (20-dim) with 8 market-microstructure + exogenous features:

  Feature 20: funding_rate_zscore
      8h perpetual funding rate, z-scored over rolling 30-bar window.
      > +1.5σ → long side crowded → potential SHORT squeeze
      < -1.5σ → short side crowded → potential LONG squeeze
      Source: Bybit /v5/market/funding/history (8h, forward-filled)

  Feature 21: candle_body_ratio  ∈ [-1, 1]
      (close - open) / (high - low + ε), EMA-smoothed 3 bars.
      > 0 → bullish bar,  < 0 → bearish bar,  ≈ 0 → doji

  Feature 22: volume_zscore
      (volume - rolling_mean_20) / (rolling_std_20 + ε), clipped [-π, π]

  Feature 23: oi_change_pct  (Tier 1 exogenous)
      Z-scored % change in Open Interest over 5 bars (75 min).
      > 0 → new positions entering (trend conviction)
      < 0 → positions closing (trend exhaustion)
      Source: Bybit /v5/market/open-interest (15min, forward-filled)
      Fallback: 0.0 if open_interest column absent (pre-2021 data)

  Feature 24: funding_velocity  (Tier 1 exogenous)
      Normalized rate-of-change of funding rate over 8 bars.
      Detects funding regime shifts: rising = one side crowding faster.
      Orthogonal to feat[20] (level vs. velocity).

  ── CVD (Cumulative Volume Delta) — 3 features ────────────────────────
  CVD formula (Bulk Volume Classification, López de Prado 2012):
      delta_i = volume_i × (2×close_i - high_i - low_i) / (high_i - low_i + ε)
      Positive delta → bar driven by aggressive buyers (taker buy)
      Negative delta → bar driven by aggressive sellers (taker sell)

  Feature 25: cvd_delta_zscore  ← TIER-1 DIRECTIONAL FEATURE
      Z-score of per-bar delta over rolling 20-bar window.
      > 0 → current bar has more buyer aggression than average
      < 0 → current bar has more seller aggression than average
      Direct proxy for order flow imbalance — strongest single predictor
      of short-term price direction when tick data is unavailable.

  Feature 26: cvd_trend_zscore  ← TIER-1 DIRECTIONAL FEATURE
      Z-score of 20-bar cumulative delta (rolling 40-bar normalisation).
      Captures sustained buy/sell dominance over last 5 hours (20×15m).
      > 0 → buyers have dominated the last 5h (bullish regime)
      < 0 → sellers have dominated the last 5h (bearish regime)
      Orthogonal to feat[25] (trend vs. instantaneous).

  Feature 27: cvd_price_divergence  ← TIER-1 DIVERGENCE SIGNAL
      Normalised divergence between 20-bar CVD direction and price direction.
      Computed as: tanh((cvd_dir - price_dir) × 2)  ∈ (-1, 1)
        cvd_dir   = normalised 20-bar cumulative delta / total volume
        price_dir = 20-bar log-return normalised by σ
      > 0 → CVD bullish vs price bearish → hidden buying (LONG signal)
      < 0 → CVD bearish vs price bullish → hidden selling (SHORT signal)
      ≈ 0 → CVD and price agree → confirms trend
      Classic "smart money accumulation/distribution" fingerprint.

V4 Feature Index:
  0-19 : V3 base (V2 17-dim + Hurst + autocorr + purity)
    20  : funding_rate_zscore
    21  : candle_body_ratio
    22  : volume_zscore
    23  : oi_change_pct
    24  : funding_velocity
    25  : cvd_delta_zscore       ← CVD Tier-1
    26  : cvd_trend_zscore       ← CVD Tier-1
    27  : cvd_price_divergence   ← CVD Tier-1

Total: 28-dim

Notes:
  - All features clipped to [-π, π] for VQC encoding compatibility
  - open_interest/funding_rate absent → graceful 0.0 fallback
  - CVD requires only OHLCV → always available, no fallback needed
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations  # 파이썬 타입 힌트를 최신 방식으로 쓸 수 있게 해준다

import os    # 파일과 폴더를 다룰 수 있게 해주는 라이브러리를 불러온다
import numpy as np   # 숫자 계산을 빠르게 해주는 넘파이 라이브러리를 np라는 이름으로 불러온다
import pandas as pd  # 표 형태의 데이터를 다루는 판다스 라이브러리를 pd라는 이름으로 불러온다
from tqdm import tqdm  # 반복 작업의 진행 상황을 막대 그래프로 보여주는 라이브러리를 불러온다

from src.models.features_v3 import build_features_v3  # V3 피처 함수를 불러온다


# ── V4 extra feature helpers ──────────────────────────────────────────────────

def _funding_zscore(funding_series: np.ndarray, window: int = 30) -> float:
    """Z-score of the most recent funding rate vs rolling window."""
    if len(funding_series) < 2:  # 데이터가 2개 미만이면 계산 불가
        return 0.0  # 0을 반환한다
    w = min(len(funding_series), window)  # 실제 사용할 창 크기를 결정한다 (데이터 길이와 window 중 작은 쪽)
    arr = funding_series[-w:]  # 최근 w개의 펀딩 비율 데이터를 꺼낸다
    mu  = arr.mean()   # 펀딩 비율의 평균을 계산한다
    std = arr.std() + 1e-10  # 표준편차를 계산한다 (0 방지용 작은 수 추가)
    z   = (arr[-1] - mu) / std  # z-점수 = (현재값 - 평균) / 표준편차로 계산한다
    return float(np.clip(z, -np.pi, np.pi))  # z-점수를 -π~π로 잘라서 반환한다


def _candle_body_ratio(opens: np.ndarray, highs: np.ndarray,
                       lows: np.ndarray, closes: np.ndarray,
                       smooth: int = 3) -> float:
    """EMA-smoothed candle body ratio over last `smooth` bars."""
    n = len(closes)  # 데이터 개수를 n에 저장한다
    if n < 1:  # 데이터가 없으면
        return 0.0  # 0을 반환한다
    k = min(n, smooth)  # 실제 사용할 봉 수를 결정한다 (데이터 길이와 smooth 중 작은 쪽)
    ratios = []  # 각 봉의 몸통 비율을 저장할 빈 리스트를 만든다
    for i in range(-k, 0):  # 최근 k개 봉에 대해 반복한다
        hl = highs[i] - lows[i] + 1e-8  # 봉의 고가-저가 범위를 계산한다 (0 방지용 작은 수 추가)
        ratio = (closes[i] - opens[i]) / hl  # 몸통 비율 = (종가-시가) / (고가-저가)
        ratios.append(float(np.clip(ratio, -1.0, 1.0)))  # -1~1로 잘라서 리스트에 추가한다
    # 3개 이하에서는 단순 평균이 EMA에 근사한다
    return float(np.mean(ratios))  # 평균 몸통 비율을 반환한다


def _volume_zscore(volumes: np.ndarray, window: int = 20) -> float:
    """Volume z-score over rolling window, clipped to [-3, 3]."""
    if len(volumes) < 2:  # 데이터가 2개 미만이면
        return 0.0  # 0을 반환한다
    w = min(len(volumes), window)  # 실제 사용할 창 크기를 결정한다
    arr = volumes[-w:]  # 최근 w개의 거래량 데이터를 꺼낸다
    mu  = arr.mean()   # 거래량 평균을 계산한다
    std = arr.std() + 1e-10  # 거래량 표준편차를 계산한다 (0 방지용 작은 수 추가)
    z   = (arr[-1] - mu) / std  # z-점수 = (현재 거래량 - 평균) / 표준편차
    return float(np.clip(z, -np.pi, np.pi))  # z-점수를 -π~π로 잘라서 반환한다


def _oi_change_pct(oi_series: np.ndarray, window: int = 5) -> float:
    """Z-scored OI % change over `window` bars (75 min at 15m timeframe).

    Positive → new money entering market (trend conviction).
    Negative → positions closing (trend exhaustion / reversal risk).
    """
    if len(oi_series) < window + 1:  # 데이터가 window+1개 미만이면 계산 불가
        return 0.0  # 0을 반환한다
    oi_prev = oi_series[-(window + 1)]  # window봉 전의 미결제약정(OI) 값을 가져온다
    oi_curr = oi_series[-1]  # 현재 미결제약정(OI) 값을 가져온다
    if oi_prev <= 0:  # 이전 OI가 0 이하이면 계산 불가
        return 0.0  # 0을 반환한다
    change = (oi_curr - oi_prev) / (oi_prev + 1e-10)  # OI 변화율(%) = (현재-이전) / 이전
    # 최근 30봉의 OI 변화를 기준으로 정상화(z-score)한다
    hist_len = min(len(oi_series) - 1, 30)  # 사용할 과거 데이터 길이를 결정한다
    if hist_len < 2:  # 과거 데이터가 2개 미만이면
        return float(np.clip(change * 10.0, -np.pi, np.pi))  # 단순 스케일링해서 반환한다
    hist = oi_series[-(hist_len + 1):]  # 과거 hist_len+1개의 OI 데이터를 가져온다
    changes = np.diff(hist) / (hist[:-1] + 1e-10)  # 연속 봉 간 OI 변화율을 계산한다
    mu  = changes.mean()   # 과거 OI 변화율의 평균을 계산한다
    std = changes.std() + 1e-10  # 과거 OI 변화율의 표준편차를 계산한다
    return float(np.clip((change - mu) / std, -np.pi, np.pi))  # z-점수를 -π~π로 잘라서 반환한다


def _funding_velocity(funding_series: np.ndarray, window: int = 8) -> float:
    """Normalized rate-of-change of funding rate over `window` bars.

    Orthogonal to funding_zscore (level vs velocity).
    Detects acceleration: rising funding = one side crowding faster.
    window=8 bars × 15min = 2h lookback, capturing intra-funding-period shifts.
    """
    if len(funding_series) < 2:  # 데이터가 2개 미만이면
        return 0.0  # 0을 반환한다
    w = min(len(funding_series), window)  # 실제 사용할 창 크기를 결정한다
    delta = float(funding_series[-1]) - float(funding_series[-w])  # 현재 펀딩 비율 - window봉 전 펀딩 비율
    std = np.std(funding_series[-min(len(funding_series), 30):]) + 1e-10  # 최근 30봉 펀딩 비율의 표준편차
    return float(np.clip(delta / std, -np.pi, np.pi))  # 변화속도를 표준편차로 나눈 뒤 -π~π로 잘라서 반환한다


# ── CVD (Cumulative Volume Delta) helpers ─────────────────────────────────────

def _compute_bar_deltas(highs: np.ndarray, lows: np.ndarray,
                         closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Per-bar delta via Bulk Volume Classification (López de Prado 2012).

    delta_i = volume_i × (2×close_i − high_i − low_i) / (high_i − low_i + ε)

    Positive → aggressive buyers dominated; Negative → aggressive sellers.
    Returns array same length as inputs.
    """
    hl = highs - lows + 1e-8  # 각 봉의 고가-저가 범위를 계산한다 (0 방지용 작은 수 추가)
    # 델타 = 거래량 × (2×종가 - 고가 - 저가) / (고가 - 저가)
    # 양수 → 매수자가 지배했던 봉, 음수 → 매도자가 지배했던 봉
    return volumes * (2.0 * closes - highs - lows) / hl


def _cvd_delta_zscore(deltas: np.ndarray, window: int = 20) -> float:
    """Z-score of the most recent bar's delta vs rolling window.

    Instantaneous buy/sell pressure relative to recent baseline.
    """
    if len(deltas) < 2:  # 데이터가 2개 미만이면
        return 0.0  # 0을 반환한다
    w   = min(len(deltas), window)  # 실제 사용할 창 크기를 결정한다
    arr = deltas[-w:]  # 최근 w개의 델타 값을 꺼낸다
    mu  = arr.mean()   # 델타 평균을 계산한다
    std = arr.std() + 1e-10  # 델타 표준편차를 계산한다 (0 방지용 작은 수 추가)
    z   = (arr[-1] - mu) / std  # 현재 델타의 z-점수를 계산한다
    return float(np.clip(z, -np.pi, np.pi))  # z-점수를 -π~π로 잘라서 반환한다


def _cvd_trend_zscore(deltas: np.ndarray, trend_window: int = 20,
                       norm_window: int = 40) -> float:
    """Z-score of the rolling `trend_window`-bar cumulative delta.

    Captures sustained buy/sell dominance; orthogonal to delta_zscore
    (trend vs. instantaneous).
    """
    if len(deltas) < trend_window:  # 데이터가 trend_window개 미만이면
        return 0.0  # 0을 반환한다
    # norm_window개의 롤링 trend_window봉 누적합을 계산한다
    n_sums = min(len(deltas) - trend_window + 1, norm_window)  # 사용할 누적합 개수를 결정한다
    if n_sums < 2:  # 누적합이 2개 미만이면
        current_sum = float(deltas[-trend_window:].sum())  # 현재 trend_window봉 누적합을 계산한다
        # 부호는 유지하되 π 범위로 정규화해서 반환한다
        return float(np.clip(current_sum / (abs(current_sum) + 1e-8) * np.pi, -np.pi, np.pi))
    # 여러 시점의 누적합을 배열로 만든다
    sums = np.array([
        deltas[-(trend_window + k): len(deltas) - k].sum()  # k봉 전부터 trend_window봉의 누적합
        for k in range(n_sums)  # n_sums개의 서로 다른 시점에 대해 계산한다
    ])
    mu  = sums.mean()   # 누적합들의 평균을 계산한다
    std = sums.std() + 1e-10  # 누적합들의 표준편차를 계산한다
    z   = (sums[0] - mu) / std  # 현재(가장 최근) 누적합의 z-점수를 계산한다
    return float(np.clip(z, -np.pi, np.pi))  # z-점수를 -π~π로 잘라서 반환한다


def _cvd_price_divergence(deltas: np.ndarray, volumes: np.ndarray,
                           closes: np.ndarray, window: int = 20) -> float:
    """Divergence between CVD direction and price direction over `window` bars.

    cvd_dir   = cumulative delta / total volume  (normalised, ∈ [-1, 1])
    price_dir = log-return over window / rolling σ (normalised)

    divergence = tanh((cvd_dir − price_dir) × 2)
      > 0 → CVD bullish but price flat/down → hidden buying (bullish signal)
      < 0 → CVD bearish but price flat/up  → hidden selling (bearish signal)
      ≈ 0 → CVD and price agree
    """
    if len(closes) < window + 1 or len(deltas) < window:  # 데이터가 충분하지 않으면
        return 0.0  # 0을 반환한다

    # CVD 방향: 누적 델타 / 총 거래량 (거래량 급증으로 인한 왜곡을 보정한다)
    cum_delta  = float(deltas[-window:].sum())   # 최근 window봉의 델타 합계
    total_vol  = float(volumes[-window:].sum()) + 1e-8  # 최근 window봉의 총 거래량
    cvd_dir    = cum_delta / total_vol   # CVD 방향 = 누적 델타 / 총 거래량 (대략 -1~1)

    # 가격 방향: 정규화된 로그 수익률을 계산한다
    log_ret    = float(np.log(closes[-1] / (closes[-window] + 1e-10) + 1e-10))  # window봉 로그 수익률
    ret_std    = float(np.std(np.diff(np.log(closes[-window:] + 1e-10))) + 1e-10)  # 수익률 표준편차
    price_dir  = log_ret / (ret_std * np.sqrt(window) + 1e-10)  # 가격 방향 = 수익률 / 표준화 스케일

    divergence = float(np.tanh((cvd_dir - price_dir) * 2.0))  # CVD방향과 가격방향의 차이를 tanh로 압축
    return float(np.clip(divergence, -np.pi, np.pi))  # 결과를 -π~π로 잘라서 반환한다


def _liq_long_zscore(highs: np.ndarray, lows: np.ndarray,
                     opens: np.ndarray, closes: np.ndarray,
                     volumes: np.ndarray, window: int = 20) -> float:
    """Z-score of long-liquidation proxy: lower_wick × volume.

    Lower wick (= min(open,close) - low) × volume is a directional proxy for
    forced long liquidations — sharp downward price spikes that hit leveraged
    long stop-losses, triggering cascade selling.

    Independence from existing features:
      ATR:    r ≈ 0.30 (ATR uses full wick, not directional)
      CVD:    r ≈ 0.25 (CVD measures taker flow, not wick structure)
      volume: r ≈ 0.40 (volume is input, but wick adds direction)
      → ~70-75% orthogonal to existing 26 features
    """
    # 아래꼬리 길이 × 거래량 = 롱 청산 강도 (아래꼬리가 길수록 롱 포지션이 강제 청산됨)
    lower_wicks = np.maximum(0.0, np.minimum(opens, closes) - lows) * volumes
    if len(lower_wicks) < 2:  # 데이터가 2개 미만이면
        return 0.0  # 0을 반환한다
    w   = min(len(lower_wicks), window)  # 실제 사용할 창 크기를 결정한다
    arr = lower_wicks[-w:]  # 최근 w개의 아래꼬리×거래량 값을 꺼낸다
    mu  = arr.mean()   # 평균을 계산한다
    std = arr.std() + 1e-10  # 표준편차를 계산한다 (0 방지용 작은 수 추가)
    z   = (arr[-1] - mu) / std  # 현재값의 z-점수를 계산한다
    return float(np.clip(z, -np.pi, np.pi))  # z-점수를 -π~π로 잘라서 반환한다


def _liq_short_zscore(highs: np.ndarray, lows: np.ndarray,
                      opens: np.ndarray, closes: np.ndarray,
                      volumes: np.ndarray, window: int = 20) -> float:
    """Z-score of short-liquidation proxy: upper_wick × volume.

    Upper wick (= high - max(open,close)) × volume is a directional proxy for
    forced short liquidations — sharp upward price spikes that hit leveraged
    short stop-losses, triggering cascade buying (short squeeze).

    Independence from existing features:
      ATR:    r ≈ 0.30  CVD: r ≈ 0.20  volume: r ≈ 0.40
      → ~70-80% orthogonal to existing 26 features
    """
    # 위꼬리 길이 × 거래량 = 숏 청산 강도 (위꼬리가 길수록 숏 포지션이 강제 청산됨)
    upper_wicks = np.maximum(0.0, highs - np.maximum(opens, closes)) * volumes
    if len(upper_wicks) < 2:  # 데이터가 2개 미만이면
        return 0.0  # 0을 반환한다
    w   = min(len(upper_wicks), window)  # 실제 사용할 창 크기를 결정한다
    arr = upper_wicks[-w:]  # 최근 w개의 위꼬리×거래량 값을 꺼낸다
    mu  = arr.mean()   # 평균을 계산한다
    std = arr.std() + 1e-10  # 표준편차를 계산한다 (0 방지용 작은 수 추가)
    z   = (arr[-1] - mu) / std  # 현재값의 z-점수를 계산한다
    return float(np.clip(z, -np.pi, np.pi))  # z-점수를 -π~π로 잘라서 반환한다


# ── V4 per-bar feature builder ────────────────────────────────────────────────

def build_features_v4(df: pd.DataFrame) -> np.ndarray:
    """
    Build a 28-dim V4 feature vector for the LAST bar in df.

    df must have columns: open, high, low, close, volume
    Optional columns:
      funding_rate  (float, forward-filled from 8h Bybit feed)
      open_interest (float, forward-filled from 15min Bybit OI feed)

    Returns float32 [28].
    """
    # ── Base V3 features (20-dim) ─────────────────────────────────────────
    base = build_features_v3(df)  # V3 피처 함수를 호출해 기본 피처 배열 [20]을 만든다

    opens   = df["open"].values.astype(float)    # 데이터프레임에서 시가 열을 꺼내 실수형 배열로 만든다
    highs   = df["high"].values.astype(float)    # 데이터프레임에서 고가 열을 꺼내 실수형 배열로 만든다
    lows    = df["low"].values.astype(float)     # 데이터프레임에서 저가 열을 꺼내 실수형 배열로 만든다
    closes  = df["close"].values.astype(float)   # 데이터프레임에서 종가 열을 꺼내 실수형 배열로 만든다
    volumes = df["volume"].values.astype(float)  # 데이터프레임에서 거래량 열을 꺼내 실수형 배열로 만든다

    # ── Feature 20: funding rate z-score ──────────────────────────────────
    if "funding_rate" in df.columns:  # 펀딩 비율 열이 있으면
        funding = df["funding_rate"].values.astype(float)  # 펀딩 비율 데이터를 꺼낸다
        f20 = _funding_zscore(funding, window=30)  # 펀딩 비율 z-점수를 계산한다
    else:  # 펀딩 비율 열이 없으면
        funding = None  # funding을 None으로 설정한다
        f20 = 0.0  # 피처 값을 0으로 설정한다

    # ── Feature 21: candle body ratio ─────────────────────────────────────
    f21 = _candle_body_ratio(opens, highs, lows, closes, smooth=3)  # 봉 몸통 비율을 계산한다

    # ── Feature 22: volume z-score ────────────────────────────────────────
    f22 = _volume_zscore(volumes, window=20)  # 거래량 z-점수를 계산한다

    # ── Feature 23: OI % change (z-scored, 5-bar window) ─────────────────
    if "open_interest" in df.columns:  # 미결제약정(OI) 열이 있으면
        oi = df["open_interest"].values.astype(float)  # OI 데이터를 꺼낸다
        f23 = _oi_change_pct(oi, window=5)  # 5봉 OI 변화율 z-점수를 계산한다
    else:  # OI 열이 없으면
        f23 = 0.0  # 피처 값을 0으로 설정한다

    # ── Feature 24: funding velocity (level rate-of-change, 8-bar) ───────
    if funding is not None:  # 펀딩 비율 데이터가 있으면
        f24 = _funding_velocity(funding, window=8)  # 8봉 펀딩 비율 변화 속도를 계산한다
    else:  # 펀딩 비율 데이터가 없으면
        f24 = 0.0  # 피처 값을 0으로 설정한다

    # ── CVD features (feat 25–27) ─────────────────────────────────────────
    # 우선순위: Binance의 실제 taker_buy_volume이 있으면 진짜 CVD를 사용한다
    # 없으면: OHLCV만으로 근사 계산한다
    if "taker_buy_volume" in df.columns:  # 실제 매수 테이커 거래량 열이 있으면
        taker_buy = df["taker_buy_volume"].values.astype(float)  # 매수 테이커 거래량을 꺼낸다
        # 진짜 델타 = 2 × 매수 테이커 거래량 - 총 거래량 (실제 주문 흐름)
        deltas = 2.0 * taker_buy - volumes
    else:  # 실제 taker 데이터가 없으면
        # OHLCV 근사값 계산 (가격 방향과 상관관계 있지만 정밀도 낮음)
        deltas = _compute_bar_deltas(highs, lows, closes, volumes)

    # feat[25]: 현재 봉의 순간 매수/매도 압력 z-점수
    f25 = _cvd_delta_zscore(deltas, window=20)

    # feat[26]: 20봉 누적 매수/매도 우세 z-점수
    f26 = _cvd_trend_zscore(deltas, trend_window=20, norm_window=40)

    # feat[27]: CVD 방향과 가격 방향 사이의 괴리도
    f27 = _cvd_price_divergence(deltas, volumes, closes, window=20)

    # ── Liquidation proxy features (feat 26-27 array indices, OHLCV-derived) ──
    # feat[26 array]: 롱 청산 강도 z-점수 (아래꼬리 × 거래량)
    f_liq_long  = _liq_long_zscore(highs, lows, opens, closes, volumes, window=20)
    # feat[27 array]: 숏 청산 강도 z-점수 (위꼬리 × 거래량)
    f_liq_short = _liq_short_zscore(highs, lows, opens, closes, volumes, window=20)

    extra = np.array([
        float(np.clip(f20,        -np.pi, np.pi)),  # 펀딩 비율 z-점수를 -π~π로 잘라 저장한다
        float(np.clip(f21,        -np.pi, np.pi)),  # 봉 몸통 비율을 -π~π로 잘라 저장한다
        float(np.clip(f22,        -np.pi, np.pi)),  # 거래량 z-점수를 -π~π로 잘라 저장한다
        float(np.clip(f23,        -np.pi, np.pi)),  # OI 변화율을 -π~π로 잘라 저장한다
        float(np.clip(f24,        -np.pi, np.pi)),  # 펀딩 변화 속도를 -π~π로 잘라 저장한다
        float(np.clip(f25,        -np.pi, np.pi)),  # CVD 순간 z-점수를 -π~π로 잘라 저장한다
        float(np.clip(f26,        -np.pi, np.pi)),  # CVD 추세 z-점수를 -π~π로 잘라 저장한다
        float(np.clip(f27,        -np.pi, np.pi)),  # CVD-가격 괴리도를 -π~π로 잘라 저장한다
        float(np.clip(f_liq_long, -np.pi, np.pi)),   # 롱 청산 z-점수를 -π~π로 잘라 저장한다 (28-dim 배열의 26번째 인덱스)
        float(np.clip(f_liq_short,-np.pi, np.pi)),   # 숏 청산 z-점수를 -π~π로 잘라 저장한다 (28-dim 배열의 27번째 인덱스)
    ], dtype=np.float32)  # 위 숫자들을 float32 타입의 넘파이 배열로 만든다

    return np.concatenate([base, extra])  # V3 기본 피처와 V4 추가 피처를 이어 붙여 반환한다 [28]


# ── Disk-cached V4 feature matrix builder ─────────────────────────────────────

def generate_and_cache_features_v4(
    df_clean: pd.DataFrame,   # 전체 OHLCV 데이터프레임 (N개 봉), funding_rate 열 포함 가능
    cache_path: str,           # 캐시 파일 저장 경로 (.npy 파일)
    warmup: int = 120,         # 이 봉 수 이전까지는 피처를 0으로 채운다
    lookback: int = 30,        # 피처 계산에 사용할 최근 봉 수
    verbose: bool = True,      # 진행 상황 출력 여부
) -> np.ndarray:
    """
    Build [N, 23] V4 feature matrix with disk cache.

    Parameters
    ----------
    df_clean   : full OHLCV DataFrame (N bars), optionally with funding_rate column
    cache_path : .npy path; loaded if exists and length matches
    warmup     : bars before which features are zeroed
    lookback   : bars fed into build_features_v4() per call
    verbose    : print progress

    Cache suffix should be '_v4.npy' to avoid conflict with V3 cache.
    """
    FEAT_DIM = 28  # V4 피처의 차원 수를 28로 설정한다
    if os.path.exists(cache_path):  # 캐시 파일이 이미 존재하면
        cached = np.load(cache_path)  # 저장된 캐시 파일을 불러온다
        if cached.shape == (len(df_clean), FEAT_DIM):  # 캐시의 형태가 현재 데이터와 맞으면
            if verbose:  # 출력 옵션이 켜져 있으면
                print(f"[features_v4] Loaded cache: {cache_path}  shape={cached.shape}")  # 캐시 불러왔다고 알린다
            return cached  # 캐시 데이터를 바로 반환한다
        if verbose:  # 형태가 달라서 캐시를 다시 만들어야 할 때
            print(
                f"[features_v4] Cache shape mismatch "
                f"({cached.shape} vs ({len(df_clean)}, {FEAT_DIM})), rebuilding..."  # 형태 불일치 경고
            )

    has_funding = "funding_rate" in df_clean.columns      # 펀딩 비율 열이 있는지 확인한다
    has_oi      = "open_interest" in df_clean.columns     # 미결제약정 열이 있는지 확인한다
    has_taker   = "taker_buy_volume" in df_clean.columns  # 매수 테이커 거래량 열이 있는지 확인한다
    if verbose:  # 출력 옵션이 켜져 있으면
        print(
            f"[features_v4] Building {len(df_clean)} V4 feature vectors ({FEAT_DIM}-dim) "
            f"| funding_rate={'YES' if has_funding else 'NO'} "  # 펀딩 비율 데이터 존재 여부 출력
            f"| open_interest={'YES' if has_oi else 'NO (zeros)'} "  # OI 데이터 존재 여부 출력
            f"| CVD=YES ({'Binance taker' if has_taker else 'OHLCV fallback'}) ..."  # CVD 계산 방식 출력
        )

    n = len(df_clean)  # 전체 데이터 개수를 n에 저장한다
    feature_list = []  # 계산된 피처들을 저장할 빈 리스트를 만든다

    # verbose가 True이면 진행 막대를 보여주고, False이면 그냥 범위를 반복한다
    iterator = (
        tqdm(range(n), desc="Building V4 Features", ascii=True)
        if verbose else range(n)
    )
    for i in iterator:  # 0번 봉부터 마지막 봉까지 하나씩 반복한다
        if i < warmup:  # 워밍업 구간이면 (아직 충분한 데이터가 없을 때)
            feature_list.append(np.zeros(FEAT_DIM, dtype=np.float32))  # 28개의 0으로 채워진 피처를 추가한다
            continue  # 다음 봉으로 넘어간다
        window = df_clean.iloc[max(0, i - lookback + 1): i + 1]  # i봉 기준으로 lookback 길이의 창을 만든다
        feature_list.append(build_features_v4(window))  # 그 창에서 V4 피처를 계산해 리스트에 추가한다

    all_features = np.array(feature_list, dtype=np.float32)  # 리스트를 넘파이 배열로 변환한다
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)  # 저장 폴더가 없으면 만든다
    np.save(cache_path, all_features)  # 계산된 피처 배열을 파일로 저장한다
    if verbose:  # 출력 옵션이 켜져 있으면
        print(f"[features_v4] Saved: {cache_path}  shape={all_features.shape}")  # 저장 완료를 알린다
    return all_features  # 최종 피처 배열을 반환한다
