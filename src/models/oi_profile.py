"""
oi_profile.py
─────────────────────────────────────────────────────────────────────
Open Interest Profile Builder (Python port of TradingView Pine Script)

Ref: "OI Profile [Fixed Range]" by LeviathanCapital

개념:
  가격 범위를 N개 버킷으로 나누고,
  각 봉의 OI 변화량(delta)을 해당 봉의 body 범위에 비례하여 배분.
  → Delta POC (Point of Control): OI 순변화가 가장 큰 가격 레벨

  Delta POC = 스마트머니가 가장 많이 포지션 진입/청산한 가격
            = 가격 자석 (price magnet) + 지지/저항 레벨

신호 해석:
  현재가 < POC (poc_dist < 0)
    → 가격이 주요 OI 집중 레벨 아래 → 위로 당겨질 기대 → LONG 신호
  현재가 > POC (poc_dist > 0)
    → 가격이 주요 OI 집중 레벨 위  → 아래로 되돌릴 기대 → SHORT 신호

  FR_LONG + poc_dist < -1 (가격이 POC 아래 + 군중 숏 오버레버리지)
    = 가장 강력한 LONG 컨플루언스

사용:
  from src.models.oi_profile import precompute_poc_distances

  poc_dists = precompute_poc_distances(
      highs, lows, opens, closes, oi_series, atr14,
      window=100, n_buckets=20
  )
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


# ── 핵심 프로파일 빌더 ─────────────────────────────────────────────────────────

def _build_oi_profile_fast(
    highs:     np.ndarray,
    lows:      np.ndarray,
    opens:     np.ndarray,
    closes:    np.ndarray,
    oi_series: np.ndarray,
    n_buckets: int = 20,
) -> float:
    """
    롤링 윈도우 내 OI Delta Profile을 계산하고 POC 가격을 반환.

    알고리즘 (Pine Script profileAdd 동일):
      1. 가격 범위 [range_low, range_high]를 n_buckets개로 분할
      2. 각 봉의 OI delta = oi[i] - oi[i-1]
      3. OI delta를 봉의 body 범위에 비례하여 버킷에 배분
      4. 각 버킷의 누적 delta (= OI+ - OI-)
      5. |delta|가 최대인 버킷 → POC

    Parameters
    ----------
    highs, lows, opens, closes : (window,) arrays
    oi_series                  : (window,) open interest
    n_buckets                  : 가격 버킷 수

    Returns
    -------
    poc_price : float (버킷 중간값)
    """
    n = len(oi_series)
    if n < 2:
        return float(closes[-1])

    # OI delta (봉별 변화량); 첫 봉은 0
    oi_deltas = np.empty(n, dtype=np.float64)
    oi_deltas[0] = 0.0
    oi_deltas[1:] = np.diff(oi_series)

    range_high = float(np.max(highs))
    range_low  = float(np.min(lows))
    if range_high <= range_low + 1e-10:
        return float(closes[-1])

    bucket_size = (range_high - range_low) / n_buckets
    profile_delta = np.zeros(n_buckets, dtype=np.float64)

    # 봉의 body 범위
    body_tops  = np.maximum(opens, closes)   # (n,)
    body_bots  = np.minimum(opens, closes)   # (n,)
    body_sizes = (body_tops - body_bots) + 1e-8  # 0 방지

    # 각 버킷별 overlap 계산 (vectorized over bars)
    for b in range(n_buckets):
        zone_top = range_high - b * bucket_size
        zone_bot = zone_top - bucket_size

        # body와 zone의 overlap
        overlap = np.maximum(
            0.0,
            np.minimum(body_tops, zone_top) - np.maximum(body_bots, zone_bot)
        )
        fracs = overlap / body_sizes  # (n,)

        # OI+ → 양수 delta 버킷에 가산
        pos_mask = oi_deltas > 0
        neg_mask = oi_deltas < 0

        profile_delta[b] += np.sum(oi_deltas[pos_mask] * fracs[pos_mask])
        profile_delta[b] -= np.sum((-oi_deltas[neg_mask]) * fracs[neg_mask])

    # POC = |delta|가 최대인 버킷
    poc_idx   = int(np.argmax(np.abs(profile_delta)))
    poc_price = range_high - (poc_idx + 0.5) * bucket_size

    return poc_price


def precompute_poc_distances(
    highs:     np.ndarray,
    lows:      np.ndarray,
    opens:     np.ndarray,
    closes:    np.ndarray,
    oi_series: np.ndarray,
    atr_series: np.ndarray,
    window:    int  = 100,
    n_buckets: int  = 20,
    verbose:   bool = True,
) -> np.ndarray:
    """
    전체 봉 배열에 대해 rolling OI POC 거리를 사전 계산.

    poc_dist[i] = (closes[i] - poc_price[i]) / ATR[i]
      > 0 → 가격이 POC 위 (과매수 가능)
      < 0 → 가격이 POC 아래 (반등 가능성)
      ≈ 0 → 가격이 POC 근처 (고유동성 레벨)

    클리핑: [-π, π]

    Parameters
    ----------
    highs, lows, opens, closes : (N,) float arrays (Bybit OHLCV)
    oi_series                  : (N,) float array (open_interest, Bybit)
    atr_series                 : (N,) float array (ATR-14, 단위: 가격)
    window                     : POC 계산 롤링 윈도우 (봉 수)
    n_buckets                  : OI 프로파일 가격 버킷 수

    Returns
    -------
    poc_dists : (N,) float32 array, clipped to [-π, π]
    """
    N = len(closes)
    poc_prices = np.full(N, np.nan, dtype=np.float64)
    oi_available = np.any(oi_series > 0)

    if not oi_available:
        if verbose:
            print("[oi_profile] OI 데이터 없음 → poc_dist = 0")
        return np.zeros(N, dtype=np.float32)

    iterator = (
        tqdm(range(window, N), desc="OI Profile (POC)", ascii=True)
        if verbose else range(window, N)
    )

    for i in iterator:
        s = i - window + 1
        atr = float(atr_series[i])
        if atr <= 0:
            poc_prices[i] = float(closes[i])
            continue

        poc = _build_oi_profile_fast(
            highs[s:i+1],
            lows[s:i+1],
            opens[s:i+1],
            closes[s:i+1],
            oi_series[s:i+1],
            n_buckets,
        )
        poc_prices[i] = poc

    # 거리 = (현재가 - POC) / ATR
    valid = ~np.isnan(poc_prices) & (atr_series > 0)
    poc_dists = np.zeros(N, dtype=np.float64)
    poc_dists[valid] = (closes[valid] - poc_prices[valid]) / (atr_series[valid] + 1e-10)
    poc_dists = np.clip(poc_dists, -np.pi, np.pi).astype(np.float32)

    if verbose:
        nonzero = np.sum(valid)
        print(f"[oi_profile] POC 계산 완료: {nonzero}/{N} 봉  "
              f"| poc_dist range=[{poc_dists.min():.2f}, {poc_dists.max():.2f}]  "
              f"| mean={poc_dists[valid].mean():.3f}")

    return poc_dists


# ── 단일 봉 실시간 계산 (live runner용) ──────────────────────────────────────

def compute_poc_dist_live(
    highs:     np.ndarray,
    lows:      np.ndarray,
    opens:     np.ndarray,
    closes:    np.ndarray,
    oi_series: np.ndarray,
    atr:       float,
    window:    int = 100,
    n_buckets: int = 20,
) -> float:
    """
    마지막 봉의 OI POC 거리를 실시간 계산 (live runner용).

    마지막 window봉만 사용하여 POC를 계산하고,
    마지막 가격 대비 POC 거리를 ATR로 정규화.

    Returns
    -------
    poc_dist : float, clipped to [-π, π]
    """
    n = len(closes)
    s = max(0, n - window)

    if np.all(oi_series[s:] == 0) or atr <= 0:
        return 0.0

    poc = _build_oi_profile_fast(
        highs[s:], lows[s:], opens[s:], closes[s:],
        oi_series[s:], n_buckets,
    )

    dist = (float(closes[-1]) - poc) / (atr + 1e-10)
    return float(np.clip(dist, -np.pi, np.pi))
