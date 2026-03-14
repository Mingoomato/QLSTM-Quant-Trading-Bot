"""
features_v5.py
────────────────────────────────────────────────────────────────────────────
V5 Feature Pipeline: 48-dimensional  =  V4 (28) + Frenet-Serret (20)

선형 의존성 완전 제거 (v2 frenet):
  - 2D Normal N = rot90(T) → N 제거, turn_sign ±1 스칼라로 대체 (-6+3)
  - 3D Binormal B = T×N → B 제거 (-3)
  총 26 → 20 차원

추가된 Frenet 특징 (인덱스 28-47):
  [28-31] MACD 위상 곡선   : T_x, T_y, κ, turn_sign
  [32-35] CVD 위상 초상    : T_x, T_y, κ, turn_sign
  [36-43] 모멘텀 TS (3D)   : T_x,y,z, κ, τ, N_x,y,z
  [44-47] 가격-거래량 경로 : T_x, T_y, κ, turn_sign

예측 원리 (Frenet-Serret 기하):
  κ(t) 극대      → 추세 전환 임박  (곡선이 꺾이는 순간)
  turn_sign ±1   → 전환 방향 (CCW/CW)
  τ(t) → 0      → 3D 레짐 단순화 → 신호 신뢰도↑
  T(t) = (1,1,1)/√3 (mom) → 전 타임스케일 균일 상승

학습 시 feature_dim 변경 필요:
  AgentConfig(feature_dim=48, ...)
  SpectralDecomposer은 자동으로 48-dim 처리
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations  # 파이썬 타입 힌트를 최신 방식으로 쓸 수 있게 해준다

import os    # 파일과 폴더를 다룰 수 있게 해주는 라이브러리를 불러온다
import numpy as np   # 숫자 계산을 빠르게 해주는 넘파이 라이브러리를 np라는 이름으로 불러온다
import pandas as pd  # 표 형태의 데이터를 다루는 판다스 라이브러리를 pd라는 이름으로 불러온다
from tqdm import tqdm  # 반복 작업의 진행 상황을 막대 그래프로 보여주는 라이브러리를 불러온다

from src.models.features_v4 import build_features_v4  # V4 피처 함수를 불러온다
from src.models.frenet_features import build_frenet_features, FRENET_DIM  # Frenet 피처 함수와 차원 수를 불러온다

FEAT_DIM_V4 = 28  # V4 피처의 차원 수를 28로 설정한다
FEAT_DIM     = FEAT_DIM_V4 + FRENET_DIM   # V5 전체 피처 차원 수 = V4(28) + Frenet(FRENET_DIM), 총 54


# ── Per-bar builder ───────────────────────────────────────────────────────────

def build_features_v5(df: pd.DataFrame) -> np.ndarray:
    """
    Build 54-dim V5 feature vector for the LAST bar in df.

    Returns float32 [54]:
      [0-27]  : V4 base features
      [28-53] : Frenet-Serret geometric features
    """
    v4     = build_features_v4(df)       # V4 피처 함수를 호출해 기본 피처 배열 [28]을 만든다
    frenet = build_frenet_features(df)   # Frenet-Serret 기하 피처를 계산해 배열 [26]을 만든다
    return np.concatenate([v4, frenet]).astype(np.float32)  # 두 배열을 이어 붙여 float32로 반환한다


# ── Disk-cached matrix builder ────────────────────────────────────────────────

def generate_and_cache_features_v5(
    df_clean: pd.DataFrame,   # 전체 OHLCV 데이터프레임 (N개 봉), 선택적으로 funding_rate 등 포함 가능
    cache_path: str,           # 캐시 파일 저장 경로 (권장 접미사: '_v5.npy')
    warmup: int  = 120,        # 이 봉 수 이전까지는 피처를 0으로 채운다
    lookback: int = 30,        # 피처 계산에 사용할 최근 봉 수 (30 이상 권장)
    verbose: bool = True,      # 진행 상황 출력 여부
) -> np.ndarray:
    """
    Build [N, 54] V5 feature matrix with disk cache.

    Parameters
    ----------
    df_clean   : full OHLCV DataFrame (N bars), optionally with
                 funding_rate, open_interest, taker_buy_volume columns
    cache_path : .npy path (권장 suffix: '_v5.npy')
    warmup     : bars before which features are zeroed
    lookback   : bars fed into build_features_v5() per call (≥ 30)
    verbose    : progress output

    Cache is invalidated if shape mismatches (N or FEAT_DIM changed).
    """
    if os.path.exists(cache_path):  # 캐시 파일이 이미 존재하면
        cached = np.load(cache_path)  # 저장된 캐시 파일을 불러온다
        if cached.shape == (len(df_clean), FEAT_DIM):  # 캐시의 형태가 현재 데이터와 맞으면
            if verbose:  # 출력 옵션이 켜져 있으면
                print(f"[features_v5] Loaded cache: {cache_path}  shape={cached.shape}")  # 캐시 불러왔다고 알린다
            return cached  # 캐시 데이터를 바로 반환한다
        if verbose:  # 형태가 달라서 캐시를 다시 만들어야 할 때
            print(
                f"[features_v5] Cache shape mismatch "
                f"({cached.shape} vs ({len(df_clean)}, {FEAT_DIM})), rebuilding..."  # 형태 불일치 경고
            )

    has_funding = "funding_rate"      in df_clean.columns  # 펀딩 비율 열이 있는지 확인한다
    has_oi      = "open_interest"     in df_clean.columns  # 미결제약정 열이 있는지 확인한다
    has_taker   = "taker_buy_volume"  in df_clean.columns  # 매수 테이커 거래량 열이 있는지 확인한다
    if verbose:  # 출력 옵션이 켜져 있으면
        print(
            f"[features_v5] Building {len(df_clean)} V5 feature vectors "
            f"({FEAT_DIM}-dim = V4-28 + Frenet-20, dependency-free) "  # 피처 구성 정보를 출력한다
            f"| funding={'YES' if has_funding else 'NO'} "  # 펀딩 비율 데이터 존재 여부 출력
            f"| OI={'YES' if has_oi else 'NO'} "  # OI 데이터 존재 여부 출력
            f"| CVD={'Binance' if has_taker else 'OHLCV'} ..."  # CVD 계산 방식 출력
        )

    n            = len(df_clean)  # 전체 데이터 개수를 n에 저장한다
    feature_list = []  # 계산된 피처들을 저장할 빈 리스트를 만든다

    # verbose가 True이면 진행 막대를 보여주고, False이면 그냥 범위를 반복한다
    it = tqdm(range(n), desc="Building V5 Features", ascii=True) if verbose else range(n)
    for i in it:  # 0번 봉부터 마지막 봉까지 하나씩 반복한다
        if i < warmup:  # 워밍업 구간이면 (아직 충분한 데이터가 없을 때)
            feature_list.append(np.zeros(FEAT_DIM, dtype=np.float32))  # FEAT_DIM개의 0으로 채워진 피처를 추가한다
            continue  # 다음 봉으로 넘어간다
        window = df_clean.iloc[max(0, i - lookback + 1): i + 1]  # i봉 기준으로 lookback 길이의 창을 만든다
        feature_list.append(build_features_v5(window))  # 그 창에서 V5 피처를 계산해 리스트에 추가한다

    all_features = np.array(feature_list, dtype=np.float32)  # 리스트를 넘파이 배열로 변환한다
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)  # 저장 폴더가 없으면 만든다
    np.save(cache_path, all_features)  # 계산된 피처 배열을 파일로 저장한다
    if verbose:  # 출력 옵션이 켜져 있으면
        print(f"[features_v5] Saved: {cache_path}  shape={all_features.shape}")  # 저장 완료를 알린다
    return all_features  # 최종 피처 배열을 반환한다


# ── Feature name list (54-dim) ────────────────────────────────────────────────

from src.models.frenet_features import FRENET_FEATURE_NAMES  # Frenet 피처 이름 목록을 불러온다

# V5 전체 피처 이름 목록 (54개)
V5_FEATURE_NAMES = (
    # V4 기본 피처 이름 28개
    [
        "lr0", "lr1", "hl", "vol_ratio",          # 로그수익률(현재), 로그수익률(전봉), 고저비율, 거래량비율
        "ema12_dev", "macd_val", "rsi", "atr",     # EMA12편차, MACD, RSI, ATR
        "mom3", "mom10", "mom20", "stoch_k",       # 3봉모멘텀, 10봉모멘텀, 20봉모멘텀, 스토캐스틱K
        "sign_change_freq", "trend_str", "hr_sin", "hr_cos",  # 방향전환빈도, 추세강도, 시간사인, 시간코사인
        "hurst_H", "purity_proxy",                 # 허스트지수, 순도프록시
        "funding_rate_z", "candle_body_ratio", "volume_z", "oi_change_pct",  # 펀딩z, 봉몸통비율, 거래량z, OI변화
        "funding_velocity",                        # 펀딩변화속도
        "cvd_delta_z", "cvd_trend_z", "cvd_price_div",  # CVD순간z, CVD추세z, CVD-가격괴리도
        "liq_long_z", "liq_short_z",               # 롱청산z, 숏청산z
    ]
    # Frenet 기하 피처 이름 (FRENET_DIM개)
    + FRENET_FEATURE_NAMES  # frenet_features.py에서 가져온 피처 이름들을 이어 붙인다
)

assert len(V5_FEATURE_NAMES) == FEAT_DIM  # 피처 이름 개수가 전체 차원 수와 같은지 확인한다


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만 아래 코드를 실행한다
    rng   = np.random.default_rng(42)  # 재현 가능한 난수 생성기를 만든다 (시드=42)
    n     = 60  # 테스트용 봉 개수를 60으로 설정한다
    close = 50000.0 + np.cumsum(rng.normal(0, 200, n))  # 비트코인 가격처럼 생긴 가짜 종가를 만든다
    close = np.maximum(close, 100.0)  # 종가가 100 미만으로 떨어지지 않도록 제한한다

    df = pd.DataFrame({
        "open":   close * (1 + rng.normal(0, 0.001, n)),  # 종가에 아주 작은 변동을 더해 시가를 만든다
        "high":   close + rng.uniform(0, 400, n),          # 고가 = 종가 + 랜덤 상승폭
        "low":    np.maximum(close - rng.uniform(0, 400, n), 1.0),  # 저가 = 종가 - 랜덤 하락폭 (최소 1)
        "close":  close,   # 종가
        "volume": rng.uniform(1e6, 1e7, n),  # 100만~1000만 사이의 랜덤 거래량
    })  # 위 데이터들로 데이터프레임을 만든다

    feat = build_features_v5(df)  # V5 피처를 계산한다
    print(f"shape  : {feat.shape}  (expected: ({FEAT_DIM},))")  # 피처 형태를 출력한다
    print(f"range  : [{feat.min():.4f}, {feat.max():.4f}]")     # 피처 값의 최소-최대 범위를 출력한다
    print(f"NaN    : {np.isnan(feat).any()}  (expected: False)") # NaN 값이 있는지 확인해 출력한다
    print()  # 빈 줄을 출력한다
    print("── V4 base (0-27) ──")  # V4 기본 피처 구분선을 출력한다
    for i in range(FEAT_DIM_V4):  # V4 피처(0~27번) 각각에 대해 반복한다
        print(f"  [{i:2d}] {V5_FEATURE_NAMES[i]:<26s} = {feat[i]:+.4f}")  # 피처 이름과 값을 출력한다
    print()  # 빈 줄을 출력한다
    print(f"── Frenet (28-{FEAT_DIM-1}) ──")  # Frenet 피처 구분선을 출력한다
    for i in range(FEAT_DIM_V4, FEAT_DIM):  # Frenet 피처(28번~끝) 각각에 대해 반복한다
        print(f"  [{i:2d}] {V5_FEATURE_NAMES[i]:<30s} = {feat[i]:+.4f}")  # 피처 이름과 값을 출력한다

    assert feat.shape == (FEAT_DIM,)  # 피처 크기가 FEAT_DIM인지 확인한다
    assert not np.isnan(feat).any()   # NaN(숫자가 아닌 값)이 없는지 확인한다
    assert not np.isinf(feat).any()   # 무한대 값이 없는지 확인한다
    assert np.all(np.abs(feat) <= np.pi + 1e-5)  # 모든 값이 -π~π 범위 안에 있는지 확인한다
    print(f"\n[PASS] features_v5.py self-test OK")  # 모든 테스트 통과 메시지를 출력한다
    print(f"  V4={FEAT_DIM_V4}  Frenet={FRENET_DIM}  Total={FEAT_DIM}")  # 각 구성요소의 차원 수를 출력한다
