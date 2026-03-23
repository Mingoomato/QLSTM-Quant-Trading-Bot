"""
frenet_features.py
────────────────────────────────────────────────────────────────────────────
Frenet-Serret Geometric Feature Extractor  (20-dim, dependency-free)

컨셉: 시장의 여러 2D/3D "위상 곡선"에서 접선(T), 곡률(κ), 비틀림(τ),
      전환 방향(turn_sign)을 추출 → 곡선의 기하 구조로 직후 동향 예측.

선형 의존성 제거 (v2):
  - 2D Normal N = rot90(T) → N 제거, turn_sign ±1 스칼라로 대체
    (N은 T의 90도 회전이므로 새로운 정보 없음)
  - 3D Binormal B = T×N → B 제거
    (T,N이 주어지면 B = T×N으로 완전히 결정)

수학적 근거
  - 접촉원 근사 (2차 Taylor): Γ(t+Δ) ≈ Γ(t) + Δ|Γ'|T + (Δ|Γ'|)²/2 · κN
  - κ 극대 → 추세 전환 임박 (곡선이 꺾이는 순간)
  - turn_sign > 0 → 반시계(CCW) 회전, < 0 → 시계(CW) 회전
  - τ ≠ 0 → 3D 공간에서 나선형 = 다요인 복잡 레짐
  - Frenet 기본 정리: κ(s), τ(s)로 곡선 유일 결정

곡선 목록
  Curve 1 (2D): MACD 위상 공간        (ema12_dev, macd_val)       → 4D
  Curve 2 (2D): CVD 위상 초상         (cvd_delta_z, cvd_trend_z)  → 4D
  Curve 3 (3D): 모멘텀 텀 스트럭처    (mom3, mom10, mom20)        → 8D
  Curve 4 (2D): 가격-거래량 경로      (log_return, vol_ratio)     → 4D

Frenet Feature Index (20-dim, 모두 [-π, π] 클리핑)
  [0-3]   MACD  : T_x, T_y, κ, turn_sign
  [4-7]   CVD   : T_x, T_y, κ, turn_sign
  [8-15]  Mom   : T_x, T_y, T_z, κ, τ, N_x, N_y, N_z
  [16-19] PV    : T_x, T_y, κ, turn_sign

κ 스케일링  : arctan(κ_arc) ∈ [0, π/2]   (arc-length 정규화 후 적용)
τ 스케일링  : arctan(τ_raw) ∈ [-π/2, π/2]
turn_sign   : +1.0 (CCW / 반시계) or -1.0 (CW / 시계)

구현 원칙
  - 인과성: 후진 유한 차분 (look-ahead 없음)
  - n_traj=4 궤적점 (τ 계산에 4점 필요)
  - 최소 lookback: max(26, n_traj+22) bars
────────────────────────────────────────────────────────────────────────────
"""

# 파이썬의 오래된 버전과도 호환되도록 미래 기능을 미리 불러온다
from __future__ import annotations  # Import annotations from __future__ module

# 숫자 계산을 빠르게 해주는 numpy 라이브러리를 np라는 이름으로 불러온다
import numpy as np  # Import NumPy (numerical computation library) as "np"
# 표 형태의 데이터를 다루는 pandas 라이브러리를 pd라는 이름으로 불러온다
import pandas as pd  # Import Pandas (DataFrame library) as "pd"

# 프레네 특징 벡터의 총 크기: 20개 숫자로 구성된다
FRENET_DIM = 20


# ── EMA helper ────────────────────────────────────────────────────────────────

def _ema_series(arr: np.ndarray, span: int) -> np.ndarray:  # [_ema_series] Private helper function
    """Forward EMA series (in-place loop; n≤30 so speed is fine)."""
    # EMA 계산에 쓰이는 가중치(알파): span이 클수록 과거 값을 더 오래 기억한다
    alpha = 2.0 / (span + 1.0)
    # 결과를 담을 빈 배열을 arr와 같은 길이로 만든다
    out = np.empty(len(arr), dtype=float)  # Returns the number of items
    # 첫 번째 값은 그대로 사용한다 (이전 값이 없으므로)
    out[0] = arr[0]
    # 두 번째 값부터 마지막까지 EMA를 계산한다
    for i in range(1, len(arr)):  # Loop: iterate a fixed number of times
        # 새 EMA = 현재 값 × 알파 + 이전 EMA × (1 - 알파): 과거와 현재를 섞는다
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out  # Returns a value to the caller


# ── Frenet frame: 2D (dependency-free) ───────────────────────────────────────

def _frenet_2d(pts: np.ndarray):  # [_frenet_2d] Private helper function
    """
    2D Frenet frame from n≥3 trajectory points (causal backward diff).

    T         = unit tangent       ∈ [-1,1]²
    κ         = arctan(|dT/ds|)   ∈ [0, π/2]
    turn_sign = +1 CCW / -1 CW    ∈ {-1, +1}

    NOTE: Normal N = rot90(T) * turn_sign is NOT returned — it is linearly
    dependent on T and adds no new information.

    Returns (T[2], κ:float, turn_sign:float)
    """
    # 점이 부족할 때 기본값으로 쓸 접선 벡터: x방향(오른쪽)을 가리킨다
    T0 = np.array([1.0, 0.0], dtype=np.float32)  # Converts Python sequence to NumPy array

    # 점이 3개보다 적으면 계산 불가능하므로 기본값(접선=오른쪽, 곡률=0, 방향=CCW)을 반환한다
    if len(pts) < 3:  # Branch: executes only when condition is True
        return T0, 0.0, 1.0  # Returns a value to the caller

    # 뒤에서 3번째 점: 가장 오래된 점
    p0 = pts[-3].astype(float)
    # 뒤에서 2번째 점: 중간 점
    p1 = pts[-2].astype(float)
    # 가장 최근 점
    p2 = pts[-1].astype(float)

    # 이전 구간의 방향 벡터: 중간 점 - 오래된 점
    d_prev = p1 - p0
    # 현재 구간의 방향 벡터: 최근 점 - 중간 점
    d_curr = p2 - p1

    # 이전 구간의 길이: 0으로 나누는 것을 막기 위해 아주 작은 수(1e-8)를 더한다
    s_prev = float(np.linalg.norm(d_prev)) + 1e-8  # Computes vector or matrix norm
    # 현재 구간의 길이
    s_curr = float(np.linalg.norm(d_curr)) + 1e-8  # Computes vector or matrix norm

    # 이전 구간의 단위 접선 벡터: 방향만 남기고 길이를 1로 만든다
    T_prev = d_prev / s_prev
    # 현재 구간의 단위 접선 벡터
    T_curr = d_curr / s_curr

    # 호 길이로 정규화한 곡률: κ = |ΔT| / Δs (접선이 얼마나 빨리 방향을 바꾸는지)
    dT        = T_curr - T_prev  # 접선 벡터의 변화량
    # 두 구간의 평균 호 길이
    ds        = 0.5 * (s_prev + s_curr)
    # 곡률 원시값: 접선 변화량의 크기를 호 길이로 나눈다
    kappa_arc = float(np.linalg.norm(dT)) / (ds + 1e-8)  # Computes vector or matrix norm
    # arctan으로 변환하여 [0, π/2] 범위로 제한한다
    kappa     = float(np.arctan(kappa_arc))    # ∈ [0, π/2]

    # 외적(cross product)으로 회전 방향을 결정한다: 양수=반시계(CCW), 음수=시계(CW)
    cross      = float(d_prev[0] * d_curr[1] - d_prev[1] * d_curr[0])
    # 외적이 0 이상이면 반시계(+1), 음수이면 시계(-1)
    turn_sign  = 1.0 if cross >= 0.0 else -1.0

    # 현재 접선 벡터(float32), 곡률, 회전 방향 부호를 반환한다
    return T_curr.astype(np.float32), kappa, turn_sign  # Returns a value to the caller


# ── Frenet frame: 3D (B removed — B = T×N is nonlinearly determined) ─────────

def _frenet_3d(pts: np.ndarray):  # [_frenet_3d] Private helper function
    """
    3D Frenet frame from n≥4 trajectory points (causal backward diff).

    T   = unit tangent         ∈ [-1,1]³
    κ   = arctan(|dT/ds|)     ∈ [0, π/2]
    τ   = arctan(torsion)     ∈ [-π/2, π/2]
    N   = principal normal     ∈ [-1,1]³

    NOTE: Binormal B = T×N is NOT returned — given T and N, B is completely
    determined by B = T×N and carries no additional information.

    Returns (T[3], κ:float, τ:float, N[3])
    """
    # 기본 접선 벡터: x방향(오른쪽)
    T0 = np.array([1., 0., 0.], dtype=np.float32)  # Converts Python sequence to NumPy array
    # 기본 주법선 벡터: y방향(위)
    N0 = np.array([0., 1., 0.], dtype=np.float32)  # Converts Python sequence to NumPy array

    # 점이 3개보다 적으면 기본값(접선, 곡률=0, 비틀림=0, 주법선)을 반환한다
    if len(pts) < 3:  # Branch: executes only when condition is True
        return T0, 0.0, 0.0, N0  # Returns a value to the caller

    # 뒤에서 3번째 점
    p0 = pts[-3].astype(float)
    # 뒤에서 2번째 점
    p1 = pts[-2].astype(float)
    # 가장 최근 점
    p2 = pts[-1].astype(float)

    # 이전 구간의 방향 벡터
    d_prev = p1 - p0
    # 현재 구간의 방향 벡터
    d_curr = p2 - p1

    # 이전 구간의 길이 (0 나눔 방지)
    s_prev = float(np.linalg.norm(d_prev)) + 1e-8  # Computes vector or matrix norm
    # 현재 구간의 길이
    s_curr = float(np.linalg.norm(d_curr)) + 1e-8  # Computes vector or matrix norm

    # 이전 단위 접선 벡터
    T_prev = d_prev / s_prev
    # 현재 단위 접선 벡터
    T_curr = d_curr / s_curr

    # 곡률(κ) 계산
    # 접선 벡터의 변화량
    dT        = T_curr - T_prev
    # 두 구간의 평균 호 길이
    ds        = 0.5 * (s_prev + s_curr)
    # 호 길이 정규화 곡률 원시값
    kappa_arc = float(np.linalg.norm(dT)) / (ds + 1e-8)  # Computes vector or matrix norm
    # arctan으로 [0, π/2] 범위로 변환한다
    kappa     = float(np.arctan(kappa_arc))

    # 주법선(N) 계산: 곡률 변화 방향 (곡선이 꺾이는 방향)
    # 접선 변화량의 크기
    norm_dT = float(np.linalg.norm(dT))  # Computes vector or matrix norm
    # 크기가 충분히 크면 정규화해서 주법선으로 사용한다
    if norm_dT > 1e-7:  # Branch: executes only when condition is True
        N = (dT / norm_dT).astype(np.float32)  # 접선 변화량을 정규화해서 주법선을 구한다
    else:  # Branch: all previous conditions were False
        N = N0  # 크기가 너무 작으면 기본 주법선(y방향)을 사용한다

    # 비틀림(τ) 계산: 4번째 점이 필요하다 (t-3 시점)
    tau = 0.0  # 기본값은 비틀림 없음
    if len(pts) >= 4:  # Branch: executes only when condition is True
        # 가장 오래된 점 (뒤에서 4번째)
        p3     = pts[-4].astype(float)
        # t-3에서 t-2까지의 방향 벡터
        d_pp   = p0 - p3
        # 그 구간의 길이
        s_pp   = float(np.linalg.norm(d_pp)) + 1e-8  # Computes vector or matrix norm
        # t-3에서 t-2까지의 단위 접선 벡터
        T_pp   = d_pp / s_pp
        # 더 이전 구간의 접선 변화량
        dT_pp  = T_prev - T_pp
        # 더 이전 접선 변화량의 크기
        norm_dT_pp = float(np.linalg.norm(dT_pp))  # Computes vector or matrix norm
        # 크기가 충분하면 이전 주법선을 계산하고, 아니면 기본값 사용
        N_prev = (dT_pp / norm_dT_pp).astype(np.float32) if norm_dT_pp > 1e-7 else N0
        # 이전 이중법선(B) = 접선 × 주법선 (외적으로 수직 벡터를 만든다)
        B_prev = np.cross(T_prev, N_prev).astype(np.float32)  # Computes the cross product of two vectors
        # 이전 이중법선의 크기
        norm_Bp = float(np.linalg.norm(B_prev))  # Computes vector or matrix norm
        # 이전 이중법선을 정규화한다
        if norm_Bp > 1e-7:  # Branch: executes only when condition is True
            B_prev = B_prev / norm_Bp

        # 현재 이중법선(B) = 현재 접선 × 현재 주법선
        B_curr = np.cross(T_curr, N).astype(np.float32)  # Computes the cross product of two vectors
        # 현재 이중법선의 크기
        norm_Bc = float(np.linalg.norm(B_curr))  # Computes vector or matrix norm
        # 현재 이중법선을 정규화한다
        if norm_Bc > 1e-7:  # Branch: executes only when condition is True
            B_curr = B_curr / norm_Bc

        # 이중법선의 변화량: 곡선이 3D 공간에서 얼마나 비틀리는지를 나타낸다
        dB      = B_curr - B_prev
        # 두 구간의 평균 호 길이
        ds2     = 0.5 * (s_prev + s_curr)
        # 비틀림 원시값: -dB·N / Δs (Frenet-Serret 공식)
        tau_raw = float(-np.dot(dB, N) / (ds2 + 1e-8))  # Computes the dot product of two arrays
        # arctan으로 [-π/2, π/2] 범위로 변환한다
        tau     = float(np.arctan(tau_raw))    # ∈ [-π/2, π/2]

    # 현재 접선(float32), 곡률, 비틀림, 주법선(float32)을 반환한다
    return (  # Returns a value to the caller
        T_curr.astype(np.float32),
        kappa,
        tau,
        N.astype(np.float32),
    )


# ── Main feature builder ──────────────────────────────────────────────────────

# [build_frenet_features] Function definition
def build_frenet_features(df: pd.DataFrame, n_traj: int = 4) -> np.ndarray:
    """
    Build 20-dim Frenet-Serret feature vector (linear-dependency-free).

    Parameters
    ----------
    df     : OHLCV DataFrame (최소 max(26, n_traj+22) rows 권장)
    n_traj : 궤적점 수 (4 권장 — τ 계산에 4점 필요)

    Returns
    -------
    float32 [20], clipped to [-π, π]
    """
    # 종가(닫힘 가격) 배열을 실수형으로 가져온다
    closes  = df["close"].values.astype(float)
    # 고가(가장 높은 가격) 배열을 실수형으로 가져온다
    highs   = df["high"].values.astype(float)
    # 저가(가장 낮은 가격) 배열을 실수형으로 가져온다
    lows    = df["low"].values.astype(float)
    # 거래량 배열을 가져온다: "volume" 컬럼이 없으면 모두 1로 채운다
    volumes = (
        df["volume"].values.astype(float)
        if "volume" in df.columns else np.ones(len(df))  # Branch: executes only when condition is True
    )
    # 전체 데이터 개수
    n   = len(closes)  # Returns the number of items
    # 0으로 나누는 것을 막기 위한 아주 작은 수
    eps = 1e-8

    # 데이터가 너무 적으면 계산 불가능하므로 0으로 채운 배열을 반환한다
    if n < n_traj + 20:  # Branch: executes only when condition is True
        return np.zeros(FRENET_DIM, dtype=np.float32)  # Returns a value to the caller

    # ──────────────────────────────────────────────────────────────────────────
    # Curve 1: MACD 위상 공간  (ema12_dev, macd_val)
    # ──────────────────────────────────────────────────────────────────────────
    # 12봉 지수이동평균(단기 추세)을 계산한다
    ema12     = _ema_series(closes, 12)
    # 26봉 지수이동평균(장기 추세)을 계산한다
    ema26     = _ema_series(closes, 26)
    # 현재 가격이 단기 이동평균에서 얼마나 벗어났는지를 로그 비율로 계산한다
    ema12_dev = np.log(closes / np.maximum(ema12, eps))  # Computes natural logarithm element-wise
    # MACD 값: 단기 이동평균과 장기 이동평균의 로그 비율 (추세의 강도)
    macd_val  = np.log(ema12  / np.maximum(ema26, eps))  # Computes natural logarithm element-wise

    # 최근 n_traj개 점을 이용해 MACD 위상 곡선의 궤적을 만든다
    macd_traj = np.stack(
        [ema12_dev[-n_traj:], macd_val[-n_traj:]], axis=1
    ).astype(float)
    # MACD 곡선에서 2D 프레네 프레임(접선, 곡률, 회전방향)을 계산한다
    T_m, k_m, sign_m = _frenet_2d(macd_traj)

    # ──────────────────────────────────────────────────────────────────────────
    # Curve 2: CVD 위상 초상  (cvd_delta_z, cvd_trend_z)
    # ──────────────────────────────────────────────────────────────────────────
    # 고가와 저가의 차이: 각 봉의 가격 범위
    hl = highs - lows + eps
    # taker_buy_volume 컬럼이 있으면 실제 매수/매도 거래량 차이를 사용한다
    if "taker_buy_volume" in df.columns:  # Branch: executes only when condition is True
        taker  = df["taker_buy_volume"].values.astype(float)
        # CVD 델타: 매수 거래량 × 2 - 총 거래량 = 매수 - 매도
        deltas = 2.0 * taker - volumes
    else:  # Branch: all previous conditions were False
        # taker 데이터가 없으면 가격 위치로 매수/매도 압력을 추정한다
        deltas = volumes * (2.0 * closes - highs - lows) / hl

    # 델타의 표준편차: z-score 계산에 사용된다 (0 나눔 방지)
    d_std = float(deltas.std()) + eps

    # 최근 n_traj개 시점의 CVD 위상 좌표를 계산한다
    cvd_pts = []
    for lag in range(n_traj - 1, -1, -1):  # Loop: iterate a fixed number of times
        # lag만큼 과거의 인덱스를 계산한다
        idx = n - 1 - lag
        # 인덱스가 0보다 작으면 데이터가 없으므로 기본값(0, 0)을 넣는다
        if idx < 1:  # Branch: executes only when condition is True
            cvd_pts.append([0.0, 0.0])  # Appends an item to the end of the list
            continue  # Skip the rest of this iteration
        # 해당 시점의 델타를 z-score로 변환하고 [-π, π]로 제한한다
        delta_z = float(np.clip(deltas[idx] / d_std, -np.pi, np.pi))  # Clips values to [min, max] range
        # 최근 최대 20봉의 누적 거래량 흐름(CVD 추세)을 계산한다
        w       = min(20, idx + 1)
        # 최근 w봉의 델타 합계
        cum20   = float(deltas[idx + 1 - w: idx + 1].sum())
        # CVD 추세를 z-score로 변환하고 [-π, π]로 제한한다
        trend_z = float(np.clip(cum20 / (d_std * (w ** 0.5) + eps), -np.pi, np.pi))  # Clips values to [min, max] range
        cvd_pts.append([delta_z, trend_z])  # Appends an item to the end of the list

    # CVD 위상 좌표 배열로 변환한다
    cvd_traj = np.array(cvd_pts, dtype=float)  # Converts Python sequence to NumPy array
    # CVD 곡선에서 2D 프레네 프레임(접선, 곡률, 회전방향)을 계산한다
    T_c, k_c, sign_c = _frenet_2d(cvd_traj)

    # ──────────────────────────────────────────────────────────────────────────
    # Curve 3: 모멘텀 텀 스트럭처  (mom3, mom10, mom20)
    # ──────────────────────────────────────────────────────────────────────────
    # 최근 n_traj개 시점의 3가지 모멘텀 값을 계산한다
    mom_pts = []
    for lag in range(n_traj - 1, -1, -1):  # Loop: iterate a fixed number of times
        # lag만큼 과거의 인덱스를 계산한다
        idx = n - 1 - lag
        # 3봉 전 대비 현재 가격의 로그 수익률 (단기 모멘텀)
        # Computes natural logarithm element-wise
        m3  = float(np.log(closes[idx] / closes[max(0, idx - 3)]  + eps)) if idx >= 3  else 0.0
        # 10봉 전 대비 현재 가격의 로그 수익률 (중기 모멘텀)
        # Computes natural logarithm element-wise
        m10 = float(np.log(closes[idx] / closes[max(0, idx - 10)] + eps)) if idx >= 10 else 0.0
        # 20봉 전 대비 현재 가격의 로그 수익률 (장기 모멘텀)
        # Computes natural logarithm element-wise
        m20 = float(np.log(closes[idx] / closes[max(0, idx - 20)] + eps)) if idx >= 20 else 0.0
        mom_pts.append([m3, m10, m20])  # Appends an item to the end of the list

    # 모멘텀 좌표 배열로 변환한다
    mom_traj = np.array(mom_pts, dtype=float)  # Converts Python sequence to NumPy array
    # 모멘텀 곡선에서 3D 프레네 프레임(접선, 곡률, 비틀림, 주법선)을 계산한다
    T_mom, k_mom, tau_mom, N_mom = _frenet_3d(mom_traj)

    # ──────────────────────────────────────────────────────────────────────────
    # Curve 4: 가격-거래량 경로  (log_return, vol_ratio)
    # ──────────────────────────────────────────────────────────────────────────
    # 첫 번째 값은 0으로 패딩하고, 이후는 로그 수익률(가격 변화율)을 계산한다
    # Computes natural logarithm element-wise
    log_rets   = np.concatenate([[0.0], np.diff(np.log(np.where(closes > 0, closes, eps)))])
    # 각 시점에서 최근 20봉 평균 거래량을 계산한다
    # Converts Python sequence to NumPy array
    v_mean_arr = np.array([volumes[max(0, i - 20): i + 1].mean() for i in range(n)])
    # 현재 거래량을 20봉 평균으로 나눈 로그 비율: 거래량이 평소보다 많은지 적은지를 나타낸다
    # Computes natural logarithm element-wise
    vol_ratio  = np.log(np.maximum(volumes, eps) / np.maximum(v_mean_arr, eps))

    # 최근 n_traj개 시점의 가격-거래량 위상 좌표를 계산한다
    pv_pts = []
    for lag in range(n_traj - 1, -1, -1):  # Loop: iterate a fixed number of times
        # lag만큼 과거의 인덱스를 계산한다
        idx = n - 1 - lag
        pv_pts.append([  # Appends an item to the end of the list
            # 로그 수익률을 [-π, π] 범위로 제한한다
            float(np.clip(log_rets[idx],  -np.pi, np.pi)),  # Clips values to [min, max] range
            # 거래량 비율을 [-π, π] 범위로 제한한다
            float(np.clip(vol_ratio[idx], -np.pi, np.pi)),  # Clips values to [min, max] range
        ])

    # 가격-거래량 위상 좌표 배열로 변환한다
    pv_traj = np.array(pv_pts, dtype=float)  # Converts Python sequence to NumPy array
    # 가격-거래량 곡선에서 2D 프레네 프레임(접선, 곡률, 회전방향)을 계산한다
    T_pv, k_pv, sign_pv = _frenet_2d(pv_traj)

    # ──────────────────────────────────────────────────────────────────────────
    # 20-dim 벡터 조립 (선형 의존성 완전 제거)
    # ──────────────────────────────────────────────────────────────────────────
    # 4개 곡선에서 계산한 특징들을 하나의 20차원 배열로 합친다
    out = np.array([  # Converts Python sequence to NumPy array
        # [0-3]   MACD (4D) — N 제거, turn_sign으로 대체
        # MACD 곡선의 접선 x성분, y성분, 곡률, 회전방향
        T_m[0],   T_m[1],   k_m,     sign_m,
        # [4-7]   CVD (4D)
        # CVD 곡선의 접선 x성분, y성분, 곡률, 회전방향
        T_c[0],   T_c[1],   k_c,     sign_c,
        # [8-15]  Momentum TS (8D) — B 제거
        # 모멘텀 곡선의 접선 x,y,z성분, 곡률, 비틀림, 주법선 x,y,z성분
        T_mom[0], T_mom[1], T_mom[2],
        k_mom,    tau_mom,
        N_mom[0], N_mom[1], N_mom[2],
        # [16-19] Price-Volume (4D)
        # 가격-거래량 곡선의 접선 x성분, y성분, 곡률, 회전방향
        T_pv[0],  T_pv[1],  k_pv,    sign_pv,
    ], dtype=np.float32)

    # 모든 값을 [-π, π] 범위로 제한해서 반환한다
    return np.clip(out, -np.pi, np.pi)  # Returns a value to the caller


# ── Feature index reference ───────────────────────────────────────────────────

# 20개 특징 각각의 이름을 담은 리스트
FRENET_FEATURE_NAMES = [
    # MACD (4D)
    "frenet_macd_Tx",    "frenet_macd_Ty",    "frenet_macd_kappa",  "frenet_macd_sign",
    # CVD (4D)
    "frenet_cvd_Tx",     "frenet_cvd_Ty",     "frenet_cvd_kappa",   "frenet_cvd_sign",
    # Momentum TS (8D)
    "frenet_mom_Tx",     "frenet_mom_Ty",     "frenet_mom_Tz",
    "frenet_mom_kappa",  "frenet_mom_tau",
    "frenet_mom_Nx",     "frenet_mom_Ny",     "frenet_mom_Nz",
    # Price-Volume (4D)
    "frenet_pv_Tx",      "frenet_pv_Ty",      "frenet_pv_kappa",    "frenet_pv_sign",
]

# 특징 이름의 개수가 FRENET_DIM(20)과 같은지 확인한다: 다르면 에러를 낸다
assert len(FRENET_FEATURE_NAMES) == FRENET_DIM  # Assertion: raises AssertionError if condition is False


# ── Self-test ─────────────────────────────────────────────────────────────────

# 이 파일을 직접 실행할 때만 아래 코드가 동작한다 (라이브러리로 불러올 때는 무시된다)
if __name__ == "__main__":  # Branch: executes only when condition is True
    # 재현 가능한 난수 생성기를 만든다 (42는 씨앗값으로 항상 같은 난수를 만든다)
    rng   = np.random.default_rng(42)
    # 데이터 개수: 60봉
    n     = 60
    # 시작 가격 50000에서 정규분포 노이즈를 누적합해서 가격 경로를 만든다
    close = 50000.0 + np.cumsum(rng.normal(0, 200, n))  # Computes cumulative sum along an axis
    # 가격이 0 이하로 떨어지지 않도록 최솟값 100으로 제한한다
    close = np.maximum(close, 100.0)  # Finds the maximum value

    # 테스트용 OHLCV 데이터프레임을 만든다
    df = pd.DataFrame({
        # 시가: 종가에 아주 작은 노이즈를 더한다
        "open":   close * (1 + rng.normal(0, 0.001, n)),
        # 고가: 종가보다 0~400 높다
        "high":   close + rng.uniform(0, 400, n),
        # 저가: 종가보다 0~400 낮지만 최솟값 1 이상
        "low":    np.maximum(close - rng.uniform(0, 400, n), 1.0),  # Finds the maximum value
        # 종가
        "close":  close,
        # 거래량: 1백만~1천만 사이의 랜덤값
        "volume": rng.uniform(1e6, 1e7, n),
    })

    # 위에서 만든 데이터프레임으로 프레네 특징을 계산한다
    feat = build_frenet_features(df)
    # 특징 벡터의 모양(크기)을 출력한다
    print(f"shape : {feat.shape}   (expected: ({FRENET_DIM},))")  # Shape (dimensions) of the tensor/array
    # 특징값의 최솟값과 최댓값 범위를 출력한다
    print(f"range : [{feat.min():.4f}, {feat.max():.4f}]  (expected: [-pi, pi])")  # Prints output to stdout
    # NaN(숫자가 아닌 값)이 있는지 확인한다
    print(f"NaN   : {np.isnan(feat).any()}   (expected: False)")  # Tests element-wise for NaN
    print()  # Prints output to stdout
    # 각 특징의 이름과 값을 출력한다
    for i, name in enumerate(FRENET_FEATURE_NAMES):  # Loop: iterate over each item in the sequence
        print(f"  [{i:2d}] {name:<28s} = {feat[i]:+.4f}")  # Prints output to stdout

    # 특징 벡터의 모양이 (20,)인지 확인한다
    assert feat.shape == (FRENET_DIM,)  # Assertion: raises AssertionError if condition is False
    # NaN이 없는지 확인한다
    assert not np.isnan(feat).any()  # Assertion: raises AssertionError if condition is False
    # 무한대 값이 없는지 확인한다
    assert not np.isinf(feat).any()  # Assertion: raises AssertionError if condition is False
    # 모든 값이 [-π, π] 범위 안에 있는지 확인한다 (약간의 오차 허용)
    assert np.all(np.abs(feat) <= np.pi + 1e-5)  # Assertion: raises AssertionError if condition is False
    print("\n[PASS] frenet_features.py self-test OK")  # Prints output to stdout
