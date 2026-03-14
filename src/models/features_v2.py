"""
features_v2.py
--------------------------------------------------------------------
Shared V2 feature pipeline for training and backtesting.

Single source of truth — both train_quantum_v2.py and
backtest_model_v2.py import from here so train/eval features
are always identical.

Key design:
  - build_features_v2(): 27-dim raw per-bar stationary features.
    Primitive statistics only; no multi-period compression so that
    SpectralDecomposer's covariance over T=window_size bars captures
    genuine cross-feature structure.
  - compute_true_atr(): Wilder ATR via max(H-L,|H-pC|,|L-pC|) EWM.
  - generate_and_cache_features_v2(): disk-cached feature matrix
    builder used by both scripts.
--------------------------------------------------------------------
"""

from __future__ import annotations  # 파이썬 타입 힌트를 최신 방식으로 쓸 수 있게 해준다

import os  # 파일과 폴더를 다룰 수 있게 해주는 라이브러리를 불러온다
import numpy as np  # 숫자 계산을 빠르게 해주는 넘파이 라이브러리를 np라는 이름으로 불러온다
import pandas as pd  # 표 형태의 데이터를 다루는 판다스 라이브러리를 pd라는 이름으로 불러온다
from tqdm import tqdm  # 반복 작업의 진행 상황을 막대 그래프로 보여주는 라이브러리를 불러온다


# ── True Wilder ATR ───────────────────────────────────────────────

def compute_true_atr(
    highs: np.ndarray,   # 각 봉의 최고가 배열
    lows: np.ndarray,    # 각 봉의 최저가 배열
    closes: np.ndarray,  # 각 봉의 종가 배열
    period: int = 14,    # ATR 계산에 사용할 봉 수 (기본값 14)
) -> np.ndarray:
    """True Wilder ATR: max(H-L, |H-prevC|, |L-prevC|) EWM(span=period).

    Replaces the simple high-low range approximation which underestimates
    ATR during gap moves.
    """
    n = len(closes)  # 종가 데이터의 개수를 n에 저장한다
    if n < 2:  # 데이터가 2개보다 적으면
        return np.zeros(n)  # 0으로 채워진 배열을 반환한다 (계산 불가)

    tr = np.zeros(n)  # n 크기의 0으로 채워진 배열을 만든다 (True Range 저장용)
    tr[0] = highs[0] - lows[0]  # 첫 번째 봉의 True Range는 고가 - 저가로 계산한다
    h  = highs[1:]   # 두 번째 봉부터의 고가 배열을 h에 저장한다
    l  = lows[1:]    # 두 번째 봉부터의 저가 배열을 l에 저장한다
    pc = closes[:-1]  # 마지막 봉을 제외한 전날 종가 배열을 pc에 저장한다
    # True Range = max(고가-저가, |고가-전일종가|, |저가-전일종가|) 중 가장 큰 값
    tr[1:] = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))

    # 지수 이동 평균(EWM)으로 부드럽게 만들어 ATR을 계산한다
    atr = pd.Series(tr).ewm(span=period, min_periods=1, adjust=False).mean().values
    return atr  # 계산된 ATR 배열을 반환한다


# ── V2 per-bar feature builder ────────────────────────────────────

def build_features_v2(df: pd.DataFrame) -> np.ndarray:
    """Build a 16-dim stationary feature vector for the LAST bar in df.

    Pruned from 27-dim: removed redundant/noisy features
    (lag-2~5, EMA26_dev, Bollinger, mom5, trade_intensity,
     amplified_return, candle_body_unsigned, accel [accel=lr0-lr1, linear redundancy]).
    vol_ratio kept: log-scale gives better Gaussian distribution → higher LDA Cohen d.

    Requires at least 2 rows of OHLCV data.  Returns float32 [16].
    """
    closes  = df["close"].values.astype(float)   # 데이터프레임에서 종가 열을 꺼내 실수형 배열로 만든다
    highs   = df["high"].values.astype(float)    # 데이터프레임에서 고가 열을 꺼내 실수형 배열로 만든다
    lows    = df["low"].values.astype(float)     # 데이터프레임에서 저가 열을 꺼내 실수형 배열로 만든다
    # volume 열이 있으면 꺼내고, 없으면 모두 1인 배열을 만든다
    volumes = df["volume"].values.astype(float) if "volume" in df.columns \
              else np.ones(len(df))
    n = len(closes)  # 종가 데이터 개수를 n에 저장한다
    eps = 1e-8  # 0으로 나누는 오류를 막기 위한 아주 작은 숫자를 eps에 저장한다

    # ── Log returns ──────────────────────────────────────────────
    if n >= 2:  # 데이터가 2개 이상이면
        # 오늘 종가를 어제 종가로 나눈 뒤 로그를 취해 수익률을 계산한다
        log_rets = np.diff(np.log(np.where(closes > 0, closes, eps)))
    else:  # 데이터가 1개뿐이면
        log_rets = np.array([0.0])  # 수익률을 0으로 설정한다

    def lr(k: int) -> float:
        # k번째 이전 봉의 로그 수익률을 반환한다. 데이터가 없으면 0을 반환한다
        return float(log_rets[-(k + 1)]) if len(log_rets) >= k + 1 else 0.0

    lr0 = lr(0)  # 가장 최근(현재) 봉의 로그 수익률을 저장한다
    lr1 = lr(1)  # 한 봉 이전의 로그 수익률을 저장한다

    # ── Intrabar log range ────────────────────────────────────────
    # 봉 안에서 고가를 저가로 나눈 뒤 로그를 취해 봉 내 가격 범위를 계산한다
    hl = float(np.log(highs[-1] / max(lows[-1], eps))) if lows[-1] > 0 else 0.0

    # ── Volume log ratio vs 20-bar mean ──────────────────────────
    v_window  = volumes[-20:] if len(volumes) >= 20 else volumes  # 최근 20개 거래량을 가져온다
    v_mean    = float(np.mean(v_window)) if len(v_window) > 0 else 1.0  # 평균 거래량을 계산한다
    # 현재 거래량을 평균 거래량으로 나눈 뒤 로그를 취해 거래량 비율을 계산한다
    vol_ratio = float(np.log(max(volumes[-1], eps) / max(v_mean, eps)))

    # ── EMA deviations (log-scale) ────────────────────────────────
    def ema_val(arr: np.ndarray, span: int) -> float:
        # 주어진 배열에 지수 이동 평균(EMA)을 계산하는 함수다
        if len(arr) == 0:  # 배열이 비어있으면
            return 0.0  # 0을 반환한다
        alpha = 2.0 / (span + 1.0)  # EMA 가중치 알파를 계산한다 (span이 클수록 알파가 작다)
        e = float(arr[0])  # 첫 번째 값으로 EMA를 초기화한다
        for x in arr[1:]:  # 두 번째 값부터 마지막 값까지 반복한다
            e = alpha * float(x) + (1.0 - alpha) * e  # 새 값과 이전 EMA를 가중 평균한다
        return e  # 최종 EMA 값을 반환한다

    ema12     = ema_val(closes, 12)   # 12봉 지수 이동 평균을 계산한다
    ema26     = ema_val(closes, 26)   # 26봉 지수 이동 평균을 계산한다
    # 현재 종가가 EMA12보다 얼마나 위/아래에 있는지 로그로 계산한다
    ema12_dev = float(np.log(max(closes[-1], eps) / max(ema12, eps)))
    # 현재 종가가 EMA26보다 얼마나 위/아래에 있는지 로그로 계산한다
    ema26_dev = float(np.log(max(closes[-1], eps) / max(ema26, eps)))
    macd_val  = ema12_dev - ema26_dev  # MACD = EMA12 편차 - EMA26 편차로 계산한다

    # ── RSI (14-bar on log returns, normalized 0-1) ───────────────
    if len(log_rets) >= 14:  # 로그 수익률이 14개 이상 있을 때
        lr14   = log_rets[-14:]  # 최근 14개의 로그 수익률을 가져온다
        gains  = np.where(lr14 > 0, lr14, 0.0)   # 양수 수익률만 남긴다 (오른 날)
        losses = np.where(lr14 < 0, -lr14, 0.0)  # 음수 수익률의 절댓값만 남긴다 (내린 날)
        ag  = float(np.mean(gains))   # 평균 상승폭을 계산한다
        al  = float(np.mean(losses))  # 평균 하락폭을 계산한다
        rs  = ag / max(al, eps)  # RS = 평균 상승폭 / 평균 하락폭으로 계산한다
        rsi = 1.0 - 1.0 / (1.0 + rs)  # RSI = 1 - 1/(1+RS) 공식으로 0~1 범위로 계산한다
    else:  # 데이터가 14개 미만이면
        rsi = 0.5  # 기본값 0.5(중립)로 설정한다

    # ── True ATR / price (proportional, 14-bar) ───────────────────
    if n >= 2:  # 데이터가 2개 이상이면
        atr_arr = compute_true_atr(highs, lows, closes, period=14)  # 14봉 True ATR을 계산한다
        atr = float(atr_arr[-1]) / max(float(closes[-1]), eps)  # ATR을 현재 종가로 나눠 비율로 만든다
    else:  # 데이터가 1개뿐이면
        atr = 0.0  # ATR을 0으로 설정한다

    # ── Momentum [3, 5, 10, 20]-bar log returns ───────────────────
    def mom(k: int) -> float:
        # k봉 전 대비 현재 종가의 로그 수익률(모멘텀)을 계산하는 함수다
        return float(np.log(max(closes[-1], eps) / max(closes[-(k + 1)], eps))) \
            if len(closes) > k else 0.0

    mom3, mom10, mom20 = mom(3), mom(10), mom(20)  # 3봉, 10봉, 20봉 모멘텀을 각각 계산한다

    # ── Stochastic %K (14-bar) ────────────────────────────────────
    if n >= 14:  # 데이터가 14개 이상이면
        lo14    = float(np.min(lows[-14:]))    # 최근 14봉 중 가장 낮은 저가를 구한다
        hi14    = float(np.max(highs[-14:]))   # 최근 14봉 중 가장 높은 고가를 구한다
        # 스토캐스틱 %K: 현재 종가가 최저-최고 범위 안에서 어디에 있는지 0~1로 나타낸다
        stoch_k = (closes[-1] - lo14) / max(hi14 - lo14, eps)
    else:  # 데이터가 14개 미만이면
        stoch_k = 0.5  # 기본값 0.5(중립)로 설정한다

    # ── Sign change frequency (10-bar) ───────────────────────────
    if len(log_rets) >= 10:  # 로그 수익률이 10개 이상 있을 때
        signs = np.sign(log_rets[-10:])  # 최근 10개 수익률의 부호(+1, 0, -1)를 구한다
        # 부호가 바뀌는 횟수를 세어 9로 나눠 0~1 범위의 방향 전환 빈도를 계산한다
        sc    = float(np.sum(np.abs(np.diff(signs)) > 0)) / 9.0
    else:  # 데이터가 10개 미만이면
        sc = 0.0  # 방향 전환 빈도를 0으로 설정한다

    # ── Trend strength: local Sharpe of log returns (10-bar) ──────
    if len(log_rets) >= 10:  # 로그 수익률이 10개 이상 있을 때
        mu10    = float(np.mean(log_rets[-10:]))  # 최근 10봉 수익률의 평균을 구한다
        sig10   = float(np.std(log_rets[-10:]))   # 최근 10봉 수익률의 표준편차를 구한다
        trend_str = mu10 / max(sig10, eps)  # 추세 강도 = 평균 / 표준편차 (샤프 비율 형태)
    else:  # 데이터가 10개 미만이면
        trend_str = 0.0  # 추세 강도를 0으로 설정한다

    # ── Time cyclical (hour sin/cos) ──────────────────────────────
    hour = 0  # 시간을 0으로 초기화한다
    if hasattr(df.index[-1], "hour"):  # 인덱스에 hour 속성이 있으면 (datetime 형식이면)
        hour = int(df.index[-1].hour)  # 마지막 봉의 시(hour)를 꺼낸다
    elif "ts" in df.columns:  # 'ts' 열이 있으면 (타임스탬프 열)
        ts_val = df["ts"].iloc[-1]  # 마지막 행의 타임스탬프 값을 가져온다
        if isinstance(ts_val, (int, float, np.integer, np.floating)):  # 숫자형이면
            # 밀리초 단위 타임스탬프를 3600000(1시간=3600초×1000ms)으로 나눠 시간을 구한다
            hour = int(ts_val // 3_600_000) % 24
    hr_sin = float(np.sin(2.0 * np.pi * hour / 24.0))  # 시간을 사인 함수로 변환해 순환성을 표현한다
    hr_cos = float(np.cos(2.0 * np.pi * hour / 24.0))  # 시간을 코사인 함수로 변환해 순환성을 표현한다

    feat = np.array([
        lr0,       # 0  현재 봉의 로그 수익률
        lr1,       # 1  한 봉 이전의 로그 수익률
        hl,        # 2  봉 내 고가/저가 로그 범위
        vol_ratio, # 3  거래량 로그 비율 vs EMA20 (로그 스케일, LDA 기여)
        ema12_dev, # 4  EMA12 편차 (추세 위치)
        macd_val,  # 5  MACD (EMA12 - EMA26)
        rsi,       # 6  RSI 14봉 (0~1)
        atr,       # 7  True ATR / 종가
        mom3,      # 8  3봉 모멘텀 (단기 방향성)
        mom10,     # 9  10봉 모멘텀
        mom20,     # 10 20봉 모멘텀
        stoch_k,   # 11 스토캐스틱 %K (14봉)
        sc,        # 12 방향 전환 빈도 (10봉)
        trend_str, # 13 추세 강도 (샤프 10봉)
        hr_sin,    # 14 시간 사인 값
        hr_cos,    # 15 시간 코사인 값
    ], dtype=np.float32)  # 위 숫자들을 float32 타입의 넘파이 배열로 만든다

    return np.clip(feat, -np.pi, np.pi)  # 모든 값을 -π ~ +π 범위로 잘라서 반환한다

# ── Disk-cached feature matrix builder ───────────────────────────

def generate_and_cache_features_v2(
    df_clean: pd.DataFrame,   # 전체 OHLCV 데이터프레임 (N개 봉)
    cache_path: str,           # 캐시 파일 저장 경로 (.npy 파일)
    warmup: int = 120,         # 이 봉 수 이전까지는 피처를 0으로 채운다 (모델 워밍업)
    lookback: int = 30,        # 피처 계산에 사용할 최근 봉 수 (EMA26 등을 위해 26 이상 필요)
    verbose: bool = True,      # 진행 상황 출력 여부
) -> np.ndarray:
    """Build [N, 27] V2 feature matrix with disk cache.

    Parameters
    ----------
    df_clean   : full OHLCV DataFrame (N bars)
    cache_path : .npy file path; loaded if exists and length matches
    warmup     : bars before which features are zeroed (model needs history)
    lookback   : bars fed into build_features_v2() per call (>= 26 for EMA26)
    verbose    : print progress
    """
    if os.path.exists(cache_path):  # 캐시 파일이 이미 존재하면
        cached = np.load(cache_path)  # 저장된 캐시 파일을 불러온다
        if len(cached) == len(df_clean):  # 캐시 길이가 현재 데이터 길이와 같으면
            if verbose:  # 출력 옵션이 켜져 있으면
                print(f"[features_v2] Loaded cache: {cache_path}")  # 캐시 불러왔다고 알린다
            return cached  # 캐시 데이터를 바로 반환한다 (새로 계산하지 않는다)
        if verbose:  # 크기가 달라서 캐시를 다시 만들어야 할 때
            print(f"[features_v2] Cache size mismatch ({len(cached)} vs {len(df_clean)}), rebuilding...")

    if verbose:  # 출력 옵션이 켜져 있으면
        print(f"[features_v2] Building {len(df_clean)} V2 feature vectors...")  # 피처 생성 시작을 알린다

    n = len(df_clean)  # 전체 데이터 개수를 n에 저장한다
    feature_list = []  # 계산된 피처들을 저장할 빈 리스트를 만든다

    # verbose가 True이면 진행 막대를 보여주고, False이면 그냥 범위를 반복한다
    iterator = tqdm(range(n), desc="Building V2 Features", ascii=True) if verbose else range(n)
    for i in iterator:  # 0번 봉부터 마지막 봉까지 하나씩 반복한다
        if i < warmup:  # 워밍업 구간이면 (아직 충분한 데이터가 없을 때)
            feature_list.append(np.zeros(16, dtype=np.float32))  # 16개의 0으로 채워진 피처를 추가한다
            continue  # 다음 봉으로 넘어간다
        window = df_clean.iloc[max(0, i - lookback + 1): i + 1]  # i봉 기준으로 lookback 길이의 창을 만든다
        feature_list.append(build_features_v2(window))  # 그 창에서 V2 피처를 계산해 리스트에 추가한다

    all_features = np.array(feature_list, dtype=np.float32)  # 리스트를 넘파이 배열로 변환한다
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)  # 저장 폴더가 없으면 만든다
    np.save(cache_path, all_features)  # 계산된 피처 배열을 파일로 저장한다
    if verbose:  # 출력 옵션이 켜져 있으면
        print(f"[features_v2] Saved cache: {cache_path}")  # 저장 완료를 알린다
    return all_features  # 최종 피처 배열을 반환한다
