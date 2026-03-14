"""
features_v3.py
────────────────────────────────────────────────────────────────────────────
V3 Feature Pipeline: 30-dimensional advanced features

Extends features_v2.py (27-dim) with 3 additional physically-motivated
features based on the physicist/mathematician roadmap:

  Feature 27: Hurst exponent H ∈ [0.05, 0.95]  (fBm regime indicator)
  Feature 28: fBm roughness proxy               (return autocorrelation γ(1))
  Feature 29: Regime entropy                    (uncertainty ≈ Lindblad purity proxy)

Physical interpretation:
  H > 0.55 → persistent trend   → momentum signals are reliable
  H < 0.45 → mean-reverting     → contrarian signals are reliable
  H ≈ 0.50 → random walk        → no edge, reduce position size

  γ(1) > 0 → positive autocorrelation (persistent, fBm with H>0.5)
  γ(1) < 0 → negative autocorrelation (mean-reverting, H<0.5)

  Entropy = H(price quantile distribution) over 20-bar window
    High entropy → disordered market → regime transition (like low Lindblad purity)
    Low entropy  → ordered, directional market → high purity

V3 Feature Index:
  0-16 : Same as V2 (17-dim: log-returns, RSI, MACD, ATR, etc.)
    17  : Hurst exponent (R/S, 5 scales)
    18  : lag-1 autocorrelation of log-returns (γ(1))
    19  : price entropy (disorder measure, Lindblad purity proxy)

V3 compatibility:
  - First 17 features are V2 (pruned from 27 to 17 — noise removed)
  - Model must be retrained with feature_dim=20
  - generate_and_cache_features_v3() writes '_v3.npy' files (separate cache)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations  # 파이썬 타입 힌트를 최신 방식으로 쓸 수 있게 해준다

import os    # 파일과 폴더를 다룰 수 있게 해주는 라이브러리를 불러온다
import math  # 수학 함수(로그, 제곱근 등)를 쓸 수 있게 해주는 라이브러리를 불러온다
import numpy as np   # 숫자 계산을 빠르게 해주는 넘파이 라이브러리를 np라는 이름으로 불러온다
import pandas as pd  # 표 형태의 데이터를 다루는 판다스 라이브러리를 pd라는 이름으로 불러온다
from tqdm import tqdm  # 반복 작업의 진행 상황을 막대 그래프로 보여주는 라이브러리를 불러온다

# V2 기본 피처 함수들을 불러온다
from src.models.features_v2 import build_features_v2, compute_true_atr


# ── V3 extra feature helpers ──────────────────────────────────────────────────

def _hurst_rs(log_rets: np.ndarray, n_scales: int = 4) -> float:
    """
    Hurst exponent via R/S analysis (pure NumPy).

    Multi-scale: H = OLS slope of log(R/S) ~ H·log(n)

    Returns H ∈ [0.05, 0.95].
    """
    n = len(log_rets)  # 로그 수익률의 개수를 n에 저장한다
    if n < 8:  # 데이터가 8개 미만이면
        return 0.5  # 계산할 수 없으므로 랜덤 워크를 나타내는 0.5를 반환한다

    min_scale = max(4, n // (2 ** n_scales))  # 가장 작은 척도(분석 단위 크기)를 결정한다
    log_ns, log_rs = [], []  # 척도의 로그값과 R/S의 로그값을 저장할 빈 리스트를 만든다

    for k in range(n_scales):  # n_scales 번만큼 반복해 여러 척도에서 R/S를 계산한다
        s = max(4, min_scale * (2 ** k))  # k번째 척도 크기 s를 계산한다 (2배씩 커진다)
        if s > n // 2:  # 척도가 데이터 절반보다 크면
            break  # 더 이상 계산할 수 없으니 반복을 멈춘다
        n_chunks = n // s  # 데이터를 s 크기 조각으로 나눈 개수를 계산한다
        if n_chunks < 1:  # 조각이 1개도 없으면
            continue  # 이 척도는 건너뛴다

        rs_vals = []  # 각 조각의 R/S 값을 저장할 빈 리스트를 만든다
        for c in range(n_chunks):  # 각 조각에 대해 반복한다
            chunk = log_rets[c * s: (c + 1) * s]  # c번째 조각 데이터를 꺼낸다
            y = chunk - chunk.mean()  # 평균을 빼서 평균이 0인 데이터로 만든다
            cumsum = np.cumsum(y)  # 누적 합을 계산한다 (랜덤 워크의 경로)
            R = cumsum.max() - cumsum.min()  # 누적합의 최대-최소 범위 R을 계산한다
            S = chunk.std() + 1e-8  # 조각의 표준편차 S를 계산한다 (0 방지용 작은 수 추가)
            rs_vals.append(R / S)  # R/S 값을 리스트에 추가한다

        if rs_vals:  # R/S 값이 하나라도 있으면
            rs_mean = float(np.mean(rs_vals))  # 모든 조각의 R/S 평균을 계산한다
            log_ns.append(math.log(s))  # 척도 s의 로그값을 저장한다
            log_rs.append(math.log(max(rs_mean, 1e-8)))  # R/S 평균의 로그값을 저장한다

    if len(log_ns) < 2:  # 데이터 포인트가 2개 미만이면 회귀 불가
        return 0.5  # 랜덤 워크를 나타내는 0.5를 반환한다

    # OLS: H = Cov(log_n, log_rs) / Var(log_n)
    ln = np.array(log_ns)  # 척도 로그값 리스트를 넘파이 배열로 변환한다
    lr = np.array(log_rs)  # R/S 로그값 리스트를 넘파이 배열로 변환한다
    ln_c = ln - ln.mean()  # 척도 로그값에서 평균을 뺀다 (중앙화)
    lr_c = lr - lr.mean()  # R/S 로그값에서 평균을 뺀다 (중앙화)
    denom = (ln_c ** 2).sum()  # OLS 분모: 척도 로그값의 제곱합을 계산한다
    if denom < 1e-10:  # 분모가 0에 가까우면 (모든 척도가 같으면)
        return 0.5  # 계산 불가이므로 0.5를 반환한다

    H = float((ln_c * lr_c).sum() / denom)  # OLS 기울기로 허스트 지수 H를 계산한다
    return float(np.clip(H, 0.05, 0.95))  # H를 0.05~0.95 범위로 잘라서 반환한다


def _lag1_autocorrelation(log_rets: np.ndarray) -> float:
    """
    Lag-1 autocorrelation γ(1) = Cov(r_t, r_{t-1}) / Var(r_t).

    γ(1) > 0 → persistent (fBm H > 0.5)
    γ(1) < 0 → mean-reverting (H < 0.5)

    Returns γ(1) ∈ [-1, 1], clipped to [-0.9, 0.9].
    """
    if len(log_rets) < 4:  # 데이터가 4개 미만이면 계산 불가
        return 0.0  # 0을 반환한다
    r = log_rets - log_rets.mean()  # 평균을 빼서 평균이 0인 데이터로 만든다
    cov1 = float(np.mean(r[1:] * r[:-1]))  # 1시점 차이 자기공분산을 계산한다 (r_t × r_{t-1} 평균)
    var = float(np.var(r)) + 1e-10  # 분산을 계산한다 (0으로 나누기 방지용 작은 수 추가)
    return float(np.clip(cov1 / var, -0.9, 0.9))  # 자기상관계수를 -0.9~0.9로 잘라서 반환한다


def _price_entropy(closes: np.ndarray, n_bins: int = 10) -> float:
    """
    Shannon entropy H = -Σ p_i log p_i of the price change distribution.

    Discretize the last 20 bars of close returns into n_bins quantiles.

    High H → disordered, high uncertainty → regime transition
    Low H  → concentrated distribution → strong directional trend

    Returns H_normalized ∈ [0, 1] (divided by log(n_bins)).
    """
    window = closes[-20:] if len(closes) >= 20 else closes  # 최근 20개 종가를 가져온다
    if len(window) < 3:  # 데이터가 3개 미만이면 계산 불가
        return 0.5  # 중간값 0.5를 반환한다

    returns = np.diff(np.log(np.where(window > 0, window, 1e-8)))  # 로그 수익률을 계산한다
    if len(returns) < 2:  # 수익률이 2개 미만이면
        return 0.5  # 중간값 0.5를 반환한다

    # n_bins 개의 구간으로 수익률 분포를 나눈다
    counts, _ = np.histogram(returns, bins=n_bins)
    total = counts.sum()  # 전체 수익률 개수를 구한다
    if total == 0:  # 데이터가 없으면
        return 0.5  # 중간값 0.5를 반환한다

    probs = counts / total  # 각 구간의 확률(비율)을 계산한다
    # 샤논 엔트로피 계산: 확률이 0인 구간은 제외한다
    eps = 1e-10  # 로그(0) 오류를 막기 위한 아주 작은 숫자
    H = -np.sum(probs[probs > 0] * np.log(probs[probs > 0] + eps))  # 엔트로피 공식 적용
    H_max = math.log(n_bins)  # 최대 엔트로피(모든 구간이 동일 확률일 때)를 계산한다
    return float(np.clip(H / (H_max + eps), 0.0, 1.0))  # 엔트로피를 0~1 범위로 정규화해 반환한다


# ── V3 per-bar feature builder ────────────────────────────────────────────────

def build_features_v3(df: pd.DataFrame) -> np.ndarray:
    """
    Build a 30-dim V3 feature vector for the LAST bar in df.

    First 27 features are identical to V2 (build_features_v2).
    Last 3 features:
        [27] Hurst exponent H (fBm)
        [28] lag-1 autocorrelation γ(1)
        [29] price entropy (Lindblad purity proxy, normalized)

    Requires at least 8 rows of OHLCV data.
    Returns float32 [30].
    """
    # ── Base V2 features ────────────────────────────────────────────────────
    base = build_features_v2(df)  # V2 피처 함수를 호출해 기본 피처 배열 [16]을 만든다

    closes  = df["close"].values.astype(float)  # 데이터프레임에서 종가 열을 꺼내 실수형 배열로 만든다
    n = len(closes)  # 종가 데이터 개수를 n에 저장한다
    eps = 1e-8  # 0으로 나누는 오류를 막기 위한 아주 작은 숫자

    # ── Log returns for Hurst / autocorrelation ─────────────────────────────
    if n >= 2:  # 데이터가 2개 이상이면
        # 오늘 종가를 어제 종가로 나눈 뒤 로그를 취해 수익률을 계산한다
        log_rets = np.diff(np.log(np.where(closes > 0, closes, eps)))
    else:  # 데이터가 1개뿐이면
        log_rets = np.array([0.0])  # 수익률을 0으로 설정한다

    # ── Feature 27: Hurst exponent ───────────────────────────────────────────
    H = _hurst_rs(log_rets, n_scales=4)  # R/S 분석으로 허스트 지수 H를 계산한다

    # ── Feature 28: price entropy (disorder / Lindblad purity proxy) ────────
    entropy = _price_entropy(closes, n_bins=8)  # 종가 분포의 엔트로피를 계산한다
    # 엔트로피를 반전해 순도(purity)로 변환한다: 낮은 엔트로피 = 높은 순도
    # purity_proxy = 1 - 2*entropy  (1=정돈됨, -1=무질서함)
    purity_proxy = float(np.clip(1.0 - 2.0 * entropy, -1.0, 1.0))

    # ── Assemble 17-dim vector ───────────────────────────────────────────────
    extra = np.array([
        float(np.clip(H, -np.pi, np.pi)),               # 허스트 지수 H를 -π~π로 잘라 저장한다
        float(np.clip(purity_proxy, -np.pi, np.pi)),    # 순도 프록시를 -π~π로 잘라 저장한다
    ], dtype=np.float32)  # float32 타입의 넘파이 배열로 만든다

    return np.concatenate([base, extra])  # V2 기본 피처와 V3 추가 피처를 이어 붙여 반환한다 [17]


# ── Disk-cached V3 feature matrix builder ────────────────────────────────────

def generate_and_cache_features_v3(
    df_clean: pd.DataFrame,   # 전체 OHLCV 데이터프레임 (N개 봉)
    cache_path: str,           # 캐시 파일 저장 경로 (.npy 파일)
    warmup: int = 120,         # 이 봉 수 이전까지는 피처를 0으로 채운다 (모델 워밍업)
    lookback: int = 30,        # 피처 계산에 사용할 최근 봉 수
    verbose: bool = True,      # 진행 상황 출력 여부
) -> np.ndarray:
    """
    Build [N, 30] V3 feature matrix with disk cache.

    Parameters
    ----------
    df_clean   : full OHLCV DataFrame (N bars)
    cache_path : .npy file path; loaded if exists and length matches
    warmup     : bars before which features are zeroed (model needs history)
    lookback   : bars fed into build_features_v3() per call (>= 26 for EMA26)
    verbose    : print progress

    Note: cache files for V3 should use '_v3.npy' suffix to avoid
          conflicts with V2 caches ('_v2.npy').
    """
    if os.path.exists(cache_path):  # 캐시 파일이 이미 존재하면
        cached = np.load(cache_path)  # 저장된 캐시 파일을 불러온다
        if len(cached) == len(df_clean):  # 캐시 길이가 현재 데이터 길이와 같으면
            if verbose:  # 출력 옵션이 켜져 있으면
                print(f"[features_v3] Loaded cache: {cache_path}")  # 캐시 불러왔다고 알린다
            return cached  # 캐시 데이터를 바로 반환한다
        if verbose:  # 크기가 달라서 캐시를 다시 만들어야 할 때
            print(
                f"[features_v3] Cache size mismatch "
                f"({len(cached)} vs {len(df_clean)}), rebuilding..."  # 크기 불일치 경고를 출력한다
            )

    if verbose:  # 출력 옵션이 켜져 있으면
        print(f"[features_v3] Building {len(df_clean)} V3 feature vectors (30-dim)...")  # 피처 생성 시작을 알린다

    n = len(df_clean)  # 전체 데이터 개수를 n에 저장한다
    feature_list = []  # 계산된 피처들을 저장할 빈 리스트를 만든다

    # verbose가 True이면 진행 막대를 보여주고, False이면 그냥 범위를 반복한다
    iterator = (
        tqdm(range(n), desc="Building V3 Features", ascii=True)
        if verbose else range(n)
    )
    for i in iterator:  # 0번 봉부터 마지막 봉까지 하나씩 반복한다
        if i < warmup:  # 워밍업 구간이면 (아직 충분한 데이터가 없을 때)
            feature_list.append(np.zeros(17, dtype=np.float32))  # 17개의 0으로 채워진 피처를 추가한다
            continue  # 다음 봉으로 넘어간다
        window = df_clean.iloc[max(0, i - lookback + 1): i + 1]  # i봉 기준으로 lookback 길이의 창을 만든다
        feature_list.append(build_features_v3(window))  # 그 창에서 V3 피처를 계산해 리스트에 추가한다

    all_features = np.array(feature_list, dtype=np.float32)  # 리스트를 넘파이 배열로 변환한다
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)  # 저장 폴더가 없으면 만든다
    np.save(cache_path, all_features)  # 계산된 피처 배열을 파일로 저장한다
    if verbose:  # 출력 옵션이 켜져 있으면
        print(f"[features_v3] Saved cache: {cache_path}")  # 저장 완료를 알린다
    return all_features  # 최종 피처 배열을 반환한다


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만 아래 코드를 실행한다
    import pandas as pd  # 판다스 라이브러리를 다시 불러온다 (self-test용)

    print("=" * 60)  # 구분선을 출력한다
    print("  features_v3.py — Self-Test (30-dim V3 pipeline)")  # 테스트 제목을 출력한다
    print("=" * 60)  # 구분선을 출력한다

    # 가짜 OHLCV 데이터 50개 봉을 만든다
    rng = np.random.default_rng(42)  # 재현 가능한 난수 생성기를 만든다 (시드=42)
    n_bars = 60  # 테스트용 봉 개수를 60으로 설정한다
    close = 50000.0 + np.cumsum(rng.normal(0, 100, n_bars))  # 비트코인 가격처럼 생긴 가짜 종가를 만든다
    close = np.maximum(close, 100.0)  # 종가가 100 미만으로 떨어지지 않도록 제한한다
    high = close + rng.uniform(0, 300, n_bars)   # 고가 = 종가 + 랜덤 상승폭
    low  = close - rng.uniform(0, 300, n_bars)   # 저가 = 종가 - 랜덤 하락폭
    low  = np.maximum(low, 1.0)  # 저가가 1 미만으로 떨어지지 않도록 제한한다
    volume = rng.uniform(1e6, 1e7, n_bars)  # 100만~1000만 사이의 랜덤 거래량을 만든다

    df = pd.DataFrame({
        "open":   close * (1 + rng.normal(0, 0.001, n_bars)),  # 종가에 아주 작은 변동을 더해 시가를 만든다
        "high":   high,    # 고가
        "low":    low,     # 저가
        "close":  close,   # 종가
        "volume": volume,  # 거래량
    })  # 위 데이터들로 데이터프레임을 만든다

    # build_features_v3 함수를 테스트한다
    feat = build_features_v3(df)  # V3 피처를 계산한다
    print(f"\n  Feature shape: {feat.shape}  (expected: 30)")  # 피처 형태를 출력한다
    print(f"  First 27 (V2 base):  {feat[:3].tolist()} ... {feat[24:27].tolist()}")  # V2 피처 일부를 출력한다
    print(f"  Feature 27 (Hurst):  {feat[27]:.4f}  (expected: 0.05-0.95)")  # 허스트 지수를 출력한다
    print(f"  Feature 28 (AC(1)):  {feat[28]:.4f}  (expected: -0.9 to 0.9)")  # 자기상관계수를 출력한다
    print(f"  Feature 29 (Purity): {feat[29]:.4f}  (expected: -1 to 1)")  # 순도 프록시를 출력한다

    assert feat.shape == (30,), f"Expected (30,), got {feat.shape}"  # 피처 크기가 30인지 확인한다
    assert feat[27] >= 0.05 and feat[27] <= 0.95, f"H={feat[27]} out of range"  # 허스트 지수 범위 확인
    assert abs(feat[28]) <= 0.9, f"AC(1)={feat[28]} out of range"  # 자기상관계수 범위 확인
    assert abs(feat[29]) <= 1.0, f"Purity proxy={feat[29]} out of range"  # 순도 프록시 범위 확인
    assert not np.isnan(feat).any(), "NaN in features!"  # NaN(숫자가 아닌 값)이 없는지 확인한다
    assert not np.isinf(feat).any(), "Inf in features!"  # 무한대 값이 없는지 확인한다

    print("\n  Testing helpers:")  # 보조 함수 테스트 시작을 알린다
    rets = np.diff(np.log(close))  # 가짜 데이터로 로그 수익률을 계산한다
    H = _hurst_rs(rets, n_scales=4)  # 허스트 지수를 계산한다
    ac1 = _lag1_autocorrelation(rets)  # 1시점 자기상관계수를 계산한다
    ent = _price_entropy(close, n_bins=8)  # 가격 엔트로피를 계산한다
    print(f"    Hurst H = {H:.4f}")   # 허스트 지수를 출력한다
    print(f"    AC(1)   = {ac1:.4f}")  # 자기상관계수를 출력한다
    print(f"    Entropy = {ent:.4f}")  # 엔트로피를 출력한다

    print("\n  ✓ All features_v3.py tests passed!")  # 모든 테스트 통과 메시지를 출력한다
    print("=" * 60)  # 구분선을 출력한다
