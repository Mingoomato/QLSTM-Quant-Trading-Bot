"""
features_structural.py
────────────────────────────────────────────────────────────────────────────
Structural Mechanism Features — 13-dim

원칙: 모든 피처는 인과적 시장 메커니즘을 표현한다.
     통계적 상관관계(RSI, MACD, Hurst 등) 일절 사용 안 함.

메커니즘 계층:
  Tier 1 — 강제 청산 (Forced Liquidation)
    fr_z          : 펀딩레이트 z-score  → 오버레버리지 군중 크기
    fr_trend      : 펀딩레이트 속도     → 쏠림이 심화되고 있는가
    oi_change_z   : OI 변화율 z-score  → 포지션 진입/청산 속도
    oi_price_div  : OI-가격 방향 불일치 → 역방향 포지션 누적 (스퀴즈 준비)
    liq_long_z    : 롱 강제청산 지표   → 롱 청산 폭발 → 롱 연료 소진
    liq_short_z   : 숏 강제청산 지표   → 숏 청산 폭발 → 숏 연료 소진

  Tier 2 — 실제 주문흐름 (Order Flow)
    cvd_trend_z   : 20봉 누적 CVD z   → 지속적 매수/매도 지배
    cvd_price_div : CVD-가격 방향 불일치 → 숨겨진 흡수/분배
    taker_ratio_z : Taker 매수비율 z   → 공격적 매수자 비중

  Tier 3 — 레짐 필터 (Regime / Structural Context)
    ema200_dev    : price/EMA200 - 1   → 추세 위치 (기관 참여 구조)
    ema200_slope  : EMA200 기울기 z    → 추세 강도/방향
    vol_regime    : ATR z-score        → 변동성 레짐 (청산 효과 크기)
    vol_change    : ATR 변화율 z-score → 변동성 확대/축소 속도

Feature Index:
  0: fr_z          1: fr_trend       2: oi_change_z    3: oi_price_div
  4: liq_long_z    5: liq_short_z    6: cvd_trend_z    7: cvd_price_div
  8: taker_ratio_z 9: ema200_dev    10: ema200_slope  11: vol_regime
 12: vol_change

Usage:
  from src.models.features_structural import build_features_structural
  feats_df = build_features_structural(df)  # df: OHLCV + funding_rate + open_interest + taker_buy_volume
  # Returns: pd.DataFrame with 13 columns, dtype float64

────────────────────────────────────────────────────────────────────────────
LOOK-AHEAD BIAS AUDIT  (Darvin / Codex2, 2026-03-21)
────────────────────────────────────────────────────────────────────────────
Verdict: ✅ NO LOOK-AHEAD BIAS DETECTED — all 13 features are strictly causal.

Evidence (line-by-line):

1. _rolling_zscore (core normalizer)
   Window slice: arr[i - window + 1 : i + 1]  →  indices [t-w+1 .. t]
   At bar t, only bars ≤ t are accessed. First (window-1) bars = NaN.  ✅

2. _ema — pandas ewm(adjust=False)
   Recursive EMA: S_t = α·x_t + (1-α)·S_{t-1}.  Strictly causal.  ✅

3. All raw signal computations use only current and past bars.  ✅

Signed-off: Darvin (Codex2) — B5 task complete.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.bybit_mainnet import BybitMainnetClient

# ── Constants ─────────────────────────────────────────────────────────────

# Identical lookback window for ALL rolling Z-score normalizations.
# 50 bars on 1h = ~2 days — stable enough for statistics, responsive to regime.
Z_WINDOW: int = 50

# 8-bar lag for velocity/slope calculations (~8h on 1h timeframe, matches FR cycle)
VELOCITY_LAG: int = 8

# 20-bar window for cumulative CVD and OI change lookback
CUMULATIVE_WINDOW: int = 20

# EMA span for trend structure
EMA_SPAN: int = 200

# ATR calculation window
ATR_WINDOW: int = 14

FEAT_COLUMNS: list[str] = [
    "fr_z",          # 0  - Funding rate Z-score
    "fr_trend",      # 1  - Funding rate velocity Z-score
    "oi_change_z",   # 2  - OI % change Z-score
    "oi_price_div",  # 3  - OI-price divergence Z-score
    "liq_long_z",    # 4  - Long liquidation proxy Z-score
    "liq_short_z",   # 5  - Short liquidation proxy Z-score
    "cvd_trend_z",   # 6  - Cumulative volume delta trend Z-score
    "cvd_price_div", # 7  - CVD-price divergence Z-score
    "taker_ratio_z", # 8  - Taker buy ratio Z-score
    "ema200_dev",    # 9  - Price deviation from EMA200 Z-score
    "ema200_slope",  # 10 - EMA200 slope Z-score
    "vol_regime",    # 11 - ATR Z-score (volatility regime)
    "vol_change",    # 12 - ATR rate of change Z-score (vol expansion/contraction)
]
FEAT_DIM: int = 13  # len(FEAT_COLUMNS)

# Legacy name compatibility
FEAT_NAMES = FEAT_COLUMNS

TIER_COLUMN_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "tier1_forced_liquidation": (
        "funding_rate",
        "open_interest",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ),
    "tier2_order_flow": (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ),
    "tier3_regime_context": (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ),
}

FEATURE_INPUT_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "fr_z": ("funding_rate",),
    "fr_trend": ("funding_rate",),
    "oi_change_z": ("open_interest",),
    "oi_price_div": ("open_interest", "close"),
    "liq_long_z": ("open", "low", "close", "volume"),
    "liq_short_z": ("open", "high", "close", "volume"),
    "cvd_trend_z": ("open", "high", "low", "close", "volume"),
    "cvd_price_div": ("open", "high", "low", "close", "volume"),
    "taker_ratio_z": ("volume",),
    "ema200_dev": ("close",),
    "ema200_slope": ("close",),
    "vol_regime": ("high", "low", "close"),
    "vol_change": ("high", "low", "close"),
}

FEATURE_TIER_MAP: dict[str, str] = {
    "fr_z": "tier1_forced_liquidation",
    "fr_trend": "tier1_forced_liquidation",
    "oi_change_z": "tier1_forced_liquidation",
    "oi_price_div": "tier1_forced_liquidation",
    "liq_long_z": "tier1_forced_liquidation",
    "liq_short_z": "tier1_forced_liquidation",
    "cvd_trend_z": "tier2_order_flow",
    "cvd_price_div": "tier2_order_flow",
    "taker_ratio_z": "tier2_order_flow",
    "ema200_dev": "tier3_regime_context",
    "ema200_slope": "tier3_regime_context",
    "vol_regime": "tier3_regime_context",
    "vol_change": "tier3_regime_context",
}

DEFAULT_COVERAGE_SYMBOL = "BTCUSDT"
DEFAULT_COVERAGE_START_DATE = "2021-01-01"
DEFAULT_COVERAGE_OI_INTERVAL = "1h"
DEFAULT_COVERAGE_OUTPUT_PATH = os.path.join("project_output", "feature_coverage.csv")

OPTIONAL_SOURCE_NOTES: dict[str, str] = {
    "taker_ratio_z": "Uses taker_buy_volume when present; otherwise falls back to a neutral 0.5 ratio.",
    "liq_long_z": "Uses liq_long_usd when present; otherwise uses lower-wick x volume proxy.",
    "liq_short_z": "Uses liq_short_usd when present; otherwise uses upper-wick x volume proxy.",
}


def _safe_date_str(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return value.strftime("%Y-%m-%d")


def _to_utc_timestamp(value: str | int | float | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            unit = "ms" if abs(float(value)) > 1e11 else "s"
            return pd.to_datetime(value, unit=unit, utc=True)
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def _extract_time_index(df: pd.DataFrame) -> pd.Series | None:
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        if ts.notna().any():
            return pd.Series(ts, index=df.index)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if ts.notna().any():
            return pd.Series(ts, index=df.index)
    if "ts_ms" in df.columns:
        ts = pd.to_datetime(df["ts_ms"], unit="ms", utc=True, errors="coerce")
        if ts.notna().any():
            return pd.Series(ts, index=df.index)
    if isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(df.index, utc=True, errors="coerce")
        if ts.notna().any():
            return pd.Series(ts, index=df.index)
    return None


def _column_coverage_stats(df: pd.DataFrame, ts_index: pd.Series | None, column: str) -> dict[str, Any]:
    if column not in df.columns:
        return {
            "present": False,
            "non_null_count": 0,
            "non_null_ratio": 0.0,
            "first_valid": "",
            "last_valid": "",
        }

    series = df[column]
    valid_mask = series.notna()
    non_null_count = int(valid_mask.sum())
    non_null_ratio = float(non_null_count / len(series)) if len(series) else 0.0

    first_valid = ""
    last_valid = ""
    if valid_mask.any() and ts_index is not None:
        valid_ts = ts_index.loc[valid_mask]
        if valid_ts.notna().any():
            first_valid = _safe_date_str(valid_ts.min())
            last_valid = _safe_date_str(valid_ts.max())

    return {
        "present": True,
        "non_null_count": non_null_count,
        "non_null_ratio": non_null_ratio,
        "first_valid": first_valid,
        "last_valid": last_valid,
    }


def _fetch_bybit_source_coverage(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str | None,
    probe_start_date: str,
    oi_interval: str,
    cache_dir: str,
) -> dict[str, dict[str, Any]]:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    probe_start_ts = min(pd.Timestamp(probe_start_date, tz="UTC"), start_ts)
    end_ts = pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.now(tz="UTC")
    start_ms = int(start_ts.timestamp() * 1000)
    probe_start_ms = int(probe_start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    client = BybitMainnetClient(category="linear")

    source_rows: dict[str, dict[str, Any]] = {}
    df_ohlcv = _fetch_ohlcv_range(client, symbol, timeframe, probe_start_ms, end_ms)
    ohlcv_ts = pd.to_datetime(df_ohlcv["ts_ms"], unit="ms", utc=True, errors="coerce").dropna() if not df_ohlcv.empty else pd.Series(dtype="datetime64[ns, UTC]")
    ohlcv_first = _safe_date_str(ohlcv_ts.min()) if not ohlcv_ts.empty else ""
    ohlcv_last = _safe_date_str(ohlcv_ts.max()) if not ohlcv_ts.empty else ""
    ohlcv_verified = not ohlcv_ts.empty
    ohlcv_meets = bool(ohlcv_verified and ohlcv_ts.min() <= start_ts)

    for source_name in ("open", "high", "low", "close", "volume"):
        source_rows[source_name] = {
            "verified": ohlcv_verified,
            "rows": int(len(df_ohlcv)),
            "first_valid": ohlcv_first,
            "last_valid": ohlcv_last,
            "interval": timeframe,
            "meets_start_date": ohlcv_meets,
            "status": "verified" if ohlcv_meets else ("partial" if ohlcv_verified else "fetch_failed"),
            "error": "",
        }

    fetch_specs = {
        "funding_rate": {
            "interval": "8h",
            "fetch": lambda: client.fetch_funding_history(symbol, probe_start_ms, end_ms, cache_dir=cache_dir),
        },
        "open_interest": {
            "interval": oi_interval,
            "fetch": lambda: client.fetch_open_interest_history(
                symbol, probe_start_ms, end_ms, interval=oi_interval, cache_dir=cache_dir
            ),
        },
    }

    for source_name, spec in fetch_specs.items():
        row: dict[str, Any] = {
            "verified": False,
            "rows": 0,
            "first_valid": "",
            "last_valid": "",
            "interval": spec["interval"],
            "meets_start_date": False,
            "status": "fetch_failed",
            "error": "",
        }
        try:
            df_src = spec["fetch"]()
            row["rows"] = int(len(df_src))
            if not df_src.empty and "ts_ms" in df_src.columns:
                ts = pd.to_datetime(df_src["ts_ms"], unit="ms", utc=True, errors="coerce").dropna()
                if not ts.empty:
                    first_valid = ts.min()
                    last_valid = ts.max()
                    row["verified"] = True
                    row["first_valid"] = _safe_date_str(first_valid)
                    row["last_valid"] = _safe_date_str(last_valid)
                    row["meets_start_date"] = bool(first_valid <= start_ts)
                    row["status"] = "verified" if row["meets_start_date"] else "partial"
                else:
                    row["status"] = "empty_response"
            else:
                row["status"] = "empty_response"
        except Exception as exc:
            row["error"] = str(exc)
        source_rows[source_name] = row

    return source_rows


def generate_feature_coverage_matrix(
    df: pd.DataFrame,
    symbol: str = DEFAULT_COVERAGE_SYMBOL,
    start_date: str = DEFAULT_COVERAGE_START_DATE,
    end_date: str | None = None,
    oi_interval: str = DEFAULT_COVERAGE_OI_INTERVAL,
    output_path: str = DEFAULT_COVERAGE_OUTPUT_PATH,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    Write a structural feature coverage matrix to CSV.

    The audit verifies funding-rate and open-interest history against Bybit
    back to `start_date`, then rolls those source checks up into per-feature
    and per-tier readiness rows.
    """
    ts_index = _extract_time_index(df)
    input_end = _safe_date_str(ts_index.max()) if ts_index is not None and ts_index.notna().any() else ""
    effective_end = end_date or input_end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    bybit_sources = _fetch_bybit_source_coverage(
        symbol=symbol,
        timeframe="1h",
        start_date=start_date,
        end_date=effective_end,
        probe_start_date="2019-01-01",
        oi_interval=oi_interval,
        cache_dir=cache_dir,
    )

    rows: list[dict[str, Any]] = []
    all_source_columns = sorted(
        set(TIER_COLUMN_REQUIREMENTS["tier1_forced_liquidation"])
        | set(TIER_COLUMN_REQUIREMENTS["tier2_order_flow"])
        | set(TIER_COLUMN_REQUIREMENTS["tier3_regime_context"])
        | {"taker_buy_volume"}
    )

    for source_name in all_source_columns:
        input_stats = _column_coverage_stats(df, ts_index, source_name)
        bybit_stats = bybit_sources.get(
            source_name,
            {
                "verified": False,
                "rows": "",
                "first_valid": "",
                "last_valid": "",
                "interval": "",
                "meets_start_date": False,
                "status": "input_only",
                "error": "",
            },
        )
        rows.append(
            {
                "row_type": "source",
                "name": source_name,
                "tier": "",
                "depends_on": "",
                "earliest_available_date_utc": input_stats["first_valid"] or bybit_stats.get("first_valid", ""),
                "input_present": input_stats["present"],
                "input_non_null_ratio": round(float(input_stats["non_null_ratio"]), 6),
                "input_first_valid": input_stats["first_valid"],
                "input_last_valid": input_stats["last_valid"],
                "bybit_symbol": symbol if source_name in bybit_sources else "",
                "bybit_interval": bybit_stats.get("interval", ""),
                "bybit_verified": bybit_stats.get("verified", False),
                "bybit_rows": bybit_stats.get("rows", ""),
                "bybit_first_valid": bybit_stats.get("first_valid", ""),
                "bybit_last_valid": bybit_stats.get("last_valid", ""),
                "meets_2021_01_01": bybit_stats.get("meets_start_date", False),
                "status": bybit_stats.get("status", ""),
                "notes": bybit_stats.get("error", ""),
            }
        )

    feature_rows: list[dict[str, Any]] = []
    for feature_name in FEAT_COLUMNS:
        dependencies = FEATURE_INPUT_REQUIREMENTS[feature_name]
        dep_input_stats = [_column_coverage_stats(df, ts_index, dep) for dep in dependencies]
        input_present = all(stat["present"] for stat in dep_input_stats)
        input_ratio = min((float(stat["non_null_ratio"]) for stat in dep_input_stats), default=0.0)

        feature_dates: list[pd.Timestamp] = []
        for dep in dependencies:
            dep_date = _to_utc_timestamp(bybit_sources[dep]["first_valid"]) if dep in bybit_sources else _to_utc_timestamp(
                _column_coverage_stats(df, ts_index, dep)["first_valid"]
            )
            if dep_date is not None:
                feature_dates.append(dep_date)
        earliest_feature_date = max(feature_dates) if feature_dates else None

        required_bybit_deps = [dep for dep in dependencies if dep in bybit_sources]
        bybit_verified = all(bybit_sources[dep]["verified"] for dep in required_bybit_deps) if required_bybit_deps else True
        meets_start = all(bybit_sources[dep]["meets_start_date"] for dep in required_bybit_deps) if required_bybit_deps else True

        status = "ready"
        if not input_present:
            status = "missing_input_columns"
        elif required_bybit_deps and not bybit_verified:
            status = "bybit_unverified"
        elif required_bybit_deps and not meets_start:
            status = "insufficient_history"
        elif feature_name == "taker_ratio_z" and "taker_buy_volume" not in df.columns:
            status = "fallback_proxy"

        feature_rows.append(
            {
                "row_type": "feature",
                "name": feature_name,
                "tier": FEATURE_TIER_MAP[feature_name],
                "depends_on": ",".join(dependencies),
                "earliest_available_date_utc": _safe_date_str(earliest_feature_date),
                "input_present": input_present,
                "input_non_null_ratio": round(float(input_ratio), 6),
                "input_first_valid": _safe_date_str(earliest_feature_date),
                "input_last_valid": input_end,
                "bybit_symbol": symbol if required_bybit_deps else "",
                "bybit_interval": ",".join(sorted({str(bybit_sources[dep]["interval"]) for dep in required_bybit_deps})),
                "bybit_verified": bybit_verified,
                "bybit_rows": sum(int(bybit_sources[dep]["rows"]) for dep in required_bybit_deps),
                "bybit_first_valid": _safe_date_str(earliest_feature_date) if required_bybit_deps else "",
                "bybit_last_valid": effective_end if required_bybit_deps else "",
                "meets_2021_01_01": meets_start,
                "status": status,
                "notes": OPTIONAL_SOURCE_NOTES.get(feature_name, ""),
            }
        )
    rows.extend(feature_rows)

    for tier_name, tier_columns in TIER_COLUMN_REQUIREMENTS.items():
        tier_feature_rows = [row for row in feature_rows if row["tier"] == tier_name]
        tier_dates = [_to_utc_timestamp(row["input_first_valid"]) for row in tier_feature_rows]
        tier_dates = [date for date in tier_dates if date is not None]
        earliest_tier_date = max(tier_dates) if tier_dates else None
        tier_ready = all(str(row["status"]) in {"ready", "fallback_proxy"} for row in tier_feature_rows)
        tier_meets = all(bool(row["meets_2021_01_01"]) for row in tier_feature_rows)

        rows.append(
            {
                "row_type": "tier",
                "name": tier_name,
                "tier": tier_name,
                "depends_on": ",".join(tier_columns),
                "earliest_available_date_utc": _safe_date_str(earliest_tier_date),
                "input_present": all(_column_coverage_stats(df, ts_index, col)["present"] for col in tier_columns),
                "input_non_null_ratio": round(
                    min(float(_column_coverage_stats(df, ts_index, col)["non_null_ratio"]) for col in tier_columns),
                    6,
                ),
                "input_first_valid": _safe_date_str(earliest_tier_date),
                "input_last_valid": input_end,
                "bybit_symbol": symbol,
                "bybit_interval": oi_interval,
                "bybit_verified": all(bool(row["bybit_verified"]) for row in tier_feature_rows),
                "bybit_rows": sum(int(row["bybit_rows"]) for row in tier_feature_rows if str(row["bybit_rows"]) != ""),
                "bybit_first_valid": _safe_date_str(earliest_tier_date),
                "bybit_last_valid": effective_end,
                "meets_2021_01_01": tier_meets,
                "status": "ready" if tier_ready and tier_meets else "review_required",
                "notes": "",
            }
        )

    coverage_df = pd.DataFrame(rows)
    fr_oi_ready = all(
        bool(bybit_sources[source_name]["verified"]) and bool(bybit_sources[source_name]["meets_start_date"])
        for source_name in ("funding_rate", "open_interest")
    )
    fr_oi_earliest = max(
        (
            _to_utc_timestamp(bybit_sources[source_name]["first_valid"])
            for source_name in ("funding_rate", "open_interest")
        ),
        default=None,
    )
    coverage_df = pd.concat(
        [
            coverage_df,
            pd.DataFrame(
                [
                    {
                        "row_type": "audit_summary",
                        "name": "funding_rate_and_open_interest_columns",
                        "tier": "tier1_forced_liquidation",
                        "depends_on": "funding_rate,open_interest",
                        "earliest_available_date_utc": _safe_date_str(fr_oi_earliest),
                        "input_present": all(col in df.columns for col in ("funding_rate", "open_interest")),
                        "input_non_null_ratio": round(
                            min(
                                _column_coverage_stats(df, ts_index, col)["non_null_ratio"]
                                for col in ("funding_rate", "open_interest")
                            ),
                            6,
                        ),
                        "input_first_valid": _safe_date_str(fr_oi_earliest),
                        "input_last_valid": input_end,
                        "bybit_symbol": symbol,
                        "bybit_interval": "8h,1h",
                        "bybit_verified": fr_oi_ready,
                        "bybit_rows": sum(int(bybit_sources[col]["rows"]) for col in ("funding_rate", "open_interest")),
                        "bybit_first_valid": _safe_date_str(fr_oi_earliest),
                        "bybit_last_valid": effective_end,
                        "meets_2021_01_01": fr_oi_ready,
                        "status": "ready" if fr_oi_ready else "review_required",
                        "notes": (
                            "Explicit Felix task check for Bybit funding_rate and open_interest columns "
                            f"against {start_date}."
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    coverage_df.sort_values(["row_type", "tier", "name"], inplace=True, ignore_index=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(output, index=False)
    return coverage_df


def maybe_generate_feature_coverage_matrix(
    df: pd.DataFrame,
    symbol: str = DEFAULT_COVERAGE_SYMBOL,
    start_date: str = DEFAULT_COVERAGE_START_DATE,
    end_date: str | None = None,
    oi_interval: str = DEFAULT_COVERAGE_OI_INTERVAL,
    output_path: str = DEFAULT_COVERAGE_OUTPUT_PATH,
    cache_dir: str = "data",
    mode: str = "if_missing",
) -> pd.DataFrame | None:
    output = Path(output_path)
    if mode == "off":
        return None
    if mode == "if_missing" and output.exists():
        try:
            return pd.read_csv(output)
        except Exception:
            pass
    return generate_feature_coverage_matrix(
        df=df,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        oi_interval=oi_interval,
        output_path=output_path,
        cache_dir=cache_dir,
    )


# ── Rolling helpers ───────────────────────────────────────────────────────

def _rolling_zscore(arr: np.ndarray, window: int = Z_WINDOW) -> np.ndarray:
    """
    Strictly causal rolling Z-score. Returns NaN for warmup period (first
    window-1 bars) to make warmup explicit rather than hiding it with zeros.

    Z_t = (x_t - μ_{t-w+1:t}) / (σ_{t-w+1:t} + ε)

    Parameters
    ----------
    arr : 1-D float64 array
    window : lookback window (default: Z_WINDOW = 50)

    Returns
    -------
    np.ndarray of float64, same length as arr.
    First (window - 1) elements are NaN.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        w = arr[i - window + 1: i + 1]
        mu = w.mean()
        sd = w.std(ddof=0)
        out[i] = (arr[i] - mu) / (sd + 1e-9)
    return out


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Pandas EMA — strictly causal (recursive: S_t = α·x_t + (1-α)·S_{t-1})."""
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values.astype(np.float64)


def _utc_ms(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


def _ms_to_date(ts_ms: int | float | None) -> str:
    if ts_ms is None or pd.isna(ts_ms):
        return ""
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def _extract_first_valid_ts(df: pd.DataFrame, value_column: str) -> int | None:
    if df.empty or "ts_ms" not in df.columns or value_column not in df.columns:
        return None
    values = pd.to_numeric(df[value_column], errors="coerce")
    mask = values.notna()
    if mask.any():
        return int(df.loc[mask, "ts_ms"].iloc[0])
    return None


def _bool_str(flag: bool) -> str:
    return "yes" if flag else "no"


def _fetch_ohlcv_range(
    client: BybitMainnetClient,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    years = max(1, math.ceil((end_ms - start_ms) / (86_400_000 * 365.25)))
    df = client.fetch_bulk_history(
        symbol=symbol,
        timeframe=timeframe,
        end_ms=end_ms,
        years=years,
    )
    if df.empty or "ts" not in df.columns:
        return pd.DataFrame(columns=["ts_ms", "open", "high", "low", "close", "volume"])

    out = df.copy()
    ts = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out["ts_ms"] = (ts.astype("int64") // 10**6).astype("int64")
    out = out[(out["ts_ms"] >= start_ms) & (out["ts_ms"] <= end_ms)].reset_index(drop=True)
    return out


def audit_bybit_feature_coverage(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    start_date: str = "2021-01-01",
    end_date: str | None = None,
    probe_start_date: str = "2019-01-01",
    coverage_csv_path: str = "project_output/feature_coverage.csv",
    cache_dir: str = "data",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Audit Bybit source coverage required by structural features and export a CSV.

    The report explicitly verifies that funding-rate and open-interest columns
    are available from the Bybit V5 API starting at `start_date`.
    """
    end_date = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_ms = _utc_ms(start_date)
    probe_start_ms = min(_utc_ms(probe_start_date), start_ms)
    end_ms = _utc_ms(end_date) + 86_399_999

    client = BybitMainnetClient()
    df_ohlcv = _fetch_ohlcv_range(client, symbol, timeframe, probe_start_ms, end_ms)
    df_funding = client.fetch_funding_history(symbol, probe_start_ms, end_ms, cache_dir=cache_dir)
    df_oi = client.fetch_open_interest_history(
        symbol,
        probe_start_ms,
        end_ms,
        interval="1h",
        cache_dir=cache_dir,
    )

    first_seen = {
        "open": _extract_first_valid_ts(df_ohlcv, "open"),
        "high": _extract_first_valid_ts(df_ohlcv, "high"),
        "low": _extract_first_valid_ts(df_ohlcv, "low"),
        "close": _extract_first_valid_ts(df_ohlcv, "close"),
        "volume": _extract_first_valid_ts(df_ohlcv, "volume"),
        "funding_rate": _extract_first_valid_ts(df_funding, "funding_rate"),
        "open_interest": _extract_first_valid_ts(df_oi, "open_interest"),
    }

    rows: list[dict[str, object]] = []

    for column_name, first_ts in first_seen.items():
        rows.append(
            {
                "entity_type": "source_column",
                "entity_name": column_name,
                "tier": "",
                "required_columns": column_name,
                "source_mode": "bybit_native",
                "endpoint": (
                    "/v5/market/kline"
                    if column_name in {"open", "high", "low", "close", "volume"}
                    else "/v5/market/funding/history"
                    if column_name == "funding_rate"
                    else "/v5/market/open-interest"
                ),
                "interval": timeframe if column_name != "funding_rate" else "8h",
                "earliest_available_date_utc": _ms_to_date(first_ts),
                "covers_from_2021_01_01": _bool_str(first_ts is not None and first_ts <= start_ms),
                "status": "ok" if first_ts is not None else "missing",
                "notes": "",
            }
        )

    for feature_name in FEAT_COLUMNS:
        required_cols = FEATURE_INPUT_REQUIREMENTS[feature_name]
        earliest_candidates = [first_seen.get(col) for col in required_cols]
        has_full_coverage = all(ts is not None for ts in earliest_candidates)
        feature_ts = max(ts for ts in earliest_candidates if ts is not None) if has_full_coverage else None
        source_mode = "bybit_native"
        notes = ""
        if feature_name in {"cvd_trend_z", "cvd_price_div"}:
            source_mode = "derived_from_bybit_ohlcv"
            notes = "Uses BVC fallback when taker_buy_volume is absent."
        elif feature_name == "taker_ratio_z":
            source_mode = "derived_or_optional_external"
            notes = "Defaults to neutral 0.5 when taker_buy_volume is absent."
        elif feature_name in {"liq_long_z", "liq_short_z"}:
            source_mode = "derived_from_bybit_ohlcv"
            notes = "Uses wick*volume proxy unless explicit liquidation columns are supplied."

        rows.append(
            {
                "entity_type": "feature",
                "entity_name": feature_name,
                "tier": FEATURE_TIER_MAP[feature_name],
                "required_columns": ",".join(required_cols),
                "source_mode": source_mode,
                "endpoint": "derived",
                "interval": timeframe,
                "earliest_available_date_utc": _ms_to_date(feature_ts),
                "covers_from_2021_01_01": _bool_str(feature_ts is not None and feature_ts <= start_ms),
                "status": "ok" if has_full_coverage else "partial",
                "notes": notes,
            }
        )

    for tier_name, required_cols in TIER_COLUMN_REQUIREMENTS.items():
        tier_candidates = [first_seen.get(col) for col in required_cols]
        tier_ok = all(ts is not None for ts in tier_candidates)
        tier_ts = max(ts for ts in tier_candidates if ts is not None) if tier_ok else None
        rows.append(
            {
                "entity_type": "feature_tier",
                "entity_name": tier_name,
                "tier": tier_name,
                "required_columns": ",".join(required_cols),
                "source_mode": "bybit_native_and_derived",
                "endpoint": "derived",
                "interval": timeframe,
                "earliest_available_date_utc": _ms_to_date(tier_ts),
                "covers_from_2021_01_01": _bool_str(tier_ts is not None and tier_ts <= start_ms),
                "status": "ok" if tier_ok else "partial",
                "notes": "",
            }
        )

    both_present = all(first_seen[key] is not None for key in ("funding_rate", "open_interest"))
    fr_oi_ts = max(
        ts for ts in (first_seen["funding_rate"], first_seen["open_interest"]) if ts is not None
    ) if both_present else None
    rows.append(
        {
            "entity_type": "audit_summary",
            "entity_name": "funding_rate_and_open_interest_columns",
            "tier": "tier1_forced_liquidation",
            "required_columns": "funding_rate,open_interest",
            "source_mode": "bybit_native",
            "endpoint": "/v5/market/funding/history + /v5/market/open-interest",
            "interval": "8h + 1h",
            "earliest_available_date_utc": _ms_to_date(fr_oi_ts),
            "covers_from_2021_01_01": _bool_str(
                both_present
                and first_seen["funding_rate"] <= start_ms
                and first_seen["open_interest"] <= start_ms
            ),
            "status": "ok" if both_present else "missing",
            "notes": (
                "Explicit Felix task check for Bybit funding_rate and open_interest columns. "
                f"Earliest date is probed from {probe_start_date}, coverage gate is checked at {start_date}."
            ),
        }
    )

    coverage_df = pd.DataFrame(rows)
    coverage_df.sort_values(["entity_type", "tier", "entity_name"], inplace=True, ignore_index=True)
    os.makedirs(os.path.dirname(coverage_csv_path) or ".", exist_ok=True)
    coverage_df.to_csv(coverage_csv_path, index=False)

    if verbose:
        print(
            f"[structural] Coverage audit saved to {coverage_csv_path} "
            f"for {symbol} {timeframe} ({start_date} -> {end_date})"
        )

    return coverage_df


# ── Raw signal extractors ─────────────────────────────────────────────────
# Each returns a raw 1-D float64 array. Z-scoring is applied uniformly
# in build_features_structural() with the same Z_WINDOW for all features.

def _raw_funding_rate(df: pd.DataFrame) -> np.ndarray:
    """Raw funding rate, forward-filled then zero-filled."""
    return df["funding_rate"].ffill().fillna(0.0).values.astype(np.float64)


def _raw_fr_velocity(fr: np.ndarray) -> np.ndarray:
    """FR velocity: fr[t] - fr[t - VELOCITY_LAG]. First VELOCITY_LAG bars = 0."""
    n = len(fr)
    vel = np.zeros(n, dtype=np.float64)
    for i in range(VELOCITY_LAG, n):
        vel[i] = fr[i] - fr[i - VELOCITY_LAG]
    return vel


def _raw_oi_change(df: pd.DataFrame) -> np.ndarray:
    """OI % change over CUMULATIVE_WINDOW bars."""
    oi = df["open_interest"].ffill().fillna(0.0).values.astype(np.float64)
    n = len(oi)
    chg = np.zeros(n, dtype=np.float64)
    for i in range(CUMULATIVE_WINDOW, n):
        prev = oi[i - CUMULATIVE_WINDOW]
        if prev > 1e-9:
            chg[i] = (oi[i] - prev) / prev
    return chg


def _raw_oi_price_div(df: pd.DataFrame) -> np.ndarray:
    """
    OI-price divergence: direction mismatch between OI change and price change.
    > 0: OI rising + price falling → short accumulation (long squeeze setup)
    < 0: OI falling + price rising → long unwinding (short squeeze setup)
    ≈ 0: OI and price aligned (trend reinforcement)
    """
    oi = df["open_interest"].ffill().fillna(0.0).values.astype(np.float64)
    cl = df["close"].values.astype(np.float64)
    n = len(oi)
    div = np.zeros(n, dtype=np.float64)
    w = CUMULATIVE_WINDOW
    for i in range(w, n):
        prev_oi = oi[i - w]
        oi_chg = (oi[i] - prev_oi) / (prev_oi + 1e-9) if prev_oi > 1e-9 else 0.0
        px_chg = cl[i] - cl[i - w]
        oi_dir = np.sign(oi_chg)
        px_dir = np.sign(px_chg)
        # diverge: OI↑ + price↓ = positive, OI↓ + price↑ = negative
        div[i] = np.tanh((oi_dir - px_dir) * 0.75)
    return div


def _raw_liq_long(df: pd.DataFrame) -> np.ndarray:
    """
    Long liquidation proxy: lower_wick × volume.
    Uses real liq data (liq_long_usd) if available, else wick proxy.
    """
    if "liq_long_usd" in df.columns:
        raw = df["liq_long_usd"].ffill().fillna(0.0).values.astype(np.float64)
        if raw.sum() > 0:
            return raw

    op = df["open"].values.astype(np.float64)
    cl = df["close"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)
    body_lo = np.minimum(op, cl)
    lower_wick = np.maximum(body_lo - lo, 0.0)
    return lower_wick * vol


def _raw_liq_short(df: pd.DataFrame) -> np.ndarray:
    """
    Short liquidation proxy: upper_wick × volume.
    Uses real liq data (liq_short_usd) if available, else wick proxy.
    """
    if "liq_short_usd" in df.columns:
        raw = df["liq_short_usd"].ffill().fillna(0.0).values.astype(np.float64)
        if raw.sum() > 0:
            return raw

    op = df["open"].values.astype(np.float64)
    cl = df["close"].values.astype(np.float64)
    hi = df["high"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)
    body_hi = np.maximum(op, cl)
    upper_wick = np.maximum(hi - body_hi, 0.0)
    return upper_wick * vol


def _raw_cvd_cumulative(df: pd.DataFrame) -> np.ndarray:
    """
    20-bar cumulative volume delta (taker buy - sell).
    Uses taker_buy_volume if available, else BVC approximation.
    """
    hi = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)
    cl = df["close"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64)

    if "taker_buy_volume" in df.columns:
        tbv = df["taker_buy_volume"].ffill().fillna(0.0).values.astype(np.float64)
        delta = tbv - (vol - tbv)  # buy - sell
    else:
        # Bulk Volume Classification (López de Prado 2012)
        spread = hi - lo + 1e-9
        delta = vol * (2 * cl - hi - lo) / spread

    n = len(delta)
    w = CUMULATIVE_WINDOW
    cum = np.zeros(n, dtype=np.float64)
    for i in range(w - 1, n):
        cum[i] = delta[i - w + 1: i + 1].sum()
    return cum


def _raw_cvd_price_div(df: pd.DataFrame, cum_cvd: np.ndarray) -> np.ndarray:
    """
    CVD-price divergence: direction mismatch between cumulative CVD and price.
    > 0: price falling but CVD buying → hidden accumulation (LONG signal)
    < 0: price rising but CVD selling → hidden distribution (SHORT signal)
    """
    cl = df["close"].values.astype(np.float64)
    n = len(cl)
    w = CUMULATIVE_WINDOW
    div = np.zeros(n, dtype=np.float64)
    for i in range(w, n):
        px_chg = cl[i] - cl[i - w]
        cvd_dir = np.sign(cum_cvd[i])
        px_dir = np.sign(px_chg)
        div[i] = np.tanh((cvd_dir - px_dir) * 0.75)
    return div


def _raw_taker_ratio(df: pd.DataFrame) -> np.ndarray:
    """Taker buy ratio = taker_buy_volume / total_volume. Neutral = 0.5."""
    vol = df["volume"].values.astype(np.float64)
    if "taker_buy_volume" in df.columns:
        tbv = df["taker_buy_volume"].ffill().fillna(0.0).values.astype(np.float64)
        return np.where(vol > 1e-9, tbv / vol, 0.5)
    return np.full(len(vol), 0.5, dtype=np.float64)


def _raw_ema200_dev(df: pd.DataFrame) -> np.ndarray:
    """Price deviation from EMA200: close / EMA200 - 1."""
    cl = df["close"].values.astype(np.float64)
    ema = _ema(cl, EMA_SPAN)
    return np.where(ema > 1e-9, cl / ema - 1.0, 0.0)


def _raw_ema200_slope(df: pd.DataFrame) -> np.ndarray:
    """EMA200 slope: rate of change over VELOCITY_LAG bars."""
    cl = df["close"].values.astype(np.float64)
    ema = _ema(cl, EMA_SPAN)
    n = len(ema)
    slope = np.zeros(n, dtype=np.float64)
    for i in range(VELOCITY_LAG, n):
        if ema[i - VELOCITY_LAG] > 1e-9:
            slope[i] = (ema[i] - ema[i - VELOCITY_LAG]) / ema[i - VELOCITY_LAG]
    return slope


def _raw_atr(df: pd.DataFrame) -> np.ndarray:
    """Average True Range (14-bar rolling mean of True Range)."""
    hi = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)
    cl = df["close"].values.astype(np.float64)
    n = len(cl)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i - 1]), abs(lo[i] - cl[i - 1]))
    atr = pd.Series(tr).rolling(ATR_WINDOW, min_periods=1).mean().values.astype(np.float64)
    return atr


def _raw_atr_change(atr: np.ndarray) -> np.ndarray:
    """ATR rate of change over VELOCITY_LAG bars — vol expansion/contraction speed."""
    n = len(atr)
    chg = np.zeros(n, dtype=np.float64)
    for i in range(VELOCITY_LAG, n):
        if atr[i - VELOCITY_LAG] > 1e-9:
            chg[i] = (atr[i] - atr[i - VELOCITY_LAG]) / atr[i - VELOCITY_LAG]
    return chg


# ── Main builder ──────────────────────────────────────────────────────────

def build_features_structural(df: pd.DataFrame,
                              z_window: int = Z_WINDOW,
                              verbose: bool = False,
                              coverage_csv_path: str | None = None,
                              coverage_symbol: str = "BTCUSDT",
                              coverage_timeframe: str = "1h",
                              coverage_start_date: str = "2021-01-01",
                              coverage_end_date: str | None = None,
                              coverage_cache_dir: str = "data") -> pd.DataFrame:
    """
    Build 13-dim structural mechanism features.

    All features are normalized via rolling Z-score with IDENTICAL lookback
    window (default 50 bars) for consistency. The first (z_window - 1) bars
    are NaN (warmup period — not enough history for stable Z-score).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame. Required columns: open, high, low, close, volume.
        Optional columns: funding_rate, open_interest, taker_buy_volume,
                          liq_long_usd, liq_short_usd.
    z_window : int
        Rolling Z-score lookback window (identical for all 13 features).
    verbose : bool
        Print feature statistics.
    coverage_csv_path : str | None
        Optional output path for the Bybit/source coverage audit CSV.

    Returns
    -------
    pd.DataFrame
        Shape (N, 13), dtype float64, columns = FEAT_COLUMNS.
        First (z_window - 1) rows are NaN (warmup).
        Missing input data is forward-filled before computation.
    """
    # ── Ensure optional columns exist (forward-fill + zero-fill) ──────────
    df = df.copy()
    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0
    if "open_interest" not in df.columns:
        df["open_interest"] = 0.0

    # Forward-fill all numeric columns for missing data robustness
    for col in ["open", "high", "low", "close", "volume",
                "funding_rate", "open_interest"]:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0.0)

    if "taker_buy_volume" in df.columns:
        df["taker_buy_volume"] = df["taker_buy_volume"].ffill().fillna(0.0)

    if coverage_csv_path:
        generate_feature_coverage_matrix(
            df=df,
            symbol=coverage_symbol,
            start_date=coverage_start_date,
            end_date=coverage_end_date,
            oi_interval=coverage_timeframe,
            output_path=coverage_csv_path,
            cache_dir=coverage_cache_dir,
        )

    N = len(df)

    if verbose:
        print(f"[structural] Building {N} bars × {FEAT_DIM}-dim features "
              f"(z_window={z_window}) ...")

    # ── Step 1: Compute raw signals ───────────────────────────────────────
    fr = _raw_funding_rate(df)
    fr_vel = _raw_fr_velocity(fr)
    oi_chg = _raw_oi_change(df)
    oi_px_div = _raw_oi_price_div(df)
    liq_long = _raw_liq_long(df)
    liq_short = _raw_liq_short(df)
    cum_cvd = _raw_cvd_cumulative(df)
    cvd_px_div = _raw_cvd_price_div(df, cum_cvd)
    taker_ratio = _raw_taker_ratio(df)
    ema_dev = _raw_ema200_dev(df)
    ema_slope = _raw_ema200_slope(df)
    atr = _raw_atr(df)
    atr_chg = _raw_atr_change(atr)

    # ── Step 2: Apply identical rolling Z-score to ALL 13 features ────────
    raw_signals = [
        fr,           # → fr_z
        fr_vel,       # → fr_trend
        oi_chg,       # → oi_change_z
        oi_px_div,    # → oi_price_div
        liq_long,     # → liq_long_z
        liq_short,    # → liq_short_z
        cum_cvd,      # → cvd_trend_z
        cvd_px_div,   # → cvd_price_div
        taker_ratio,  # → taker_ratio_z
        ema_dev,      # → ema200_dev
        ema_slope,    # → ema200_slope
        atr,          # → vol_regime
        atr_chg,      # → vol_change
    ]

    result = np.empty((N, FEAT_DIM), dtype=np.float64)
    for col_idx, raw in enumerate(raw_signals):
        result[:, col_idx] = _rolling_zscore(raw, z_window)

    # ── Step 3: Build DataFrame with proper column names and dtype ────────
    feat_df = pd.DataFrame(result, columns=FEAT_COLUMNS, index=df.index)
    feat_df = feat_df.astype(np.float64)

    # Clip extreme values (cap at ±5σ) but preserve NaN for warmup
    feat_df = feat_df.clip(lower=-5.0, upper=5.0)

    if verbose:
        print(f"[structural] Done. shape={feat_df.shape}, "
              f"warmup_nans={feat_df.iloc[:, 0].isna().sum()} bars")
        for col in FEAT_COLUMNS:
            v = feat_df[col].dropna()
            if len(v) > 0:
                print(f"  {col:16s}: mean={v.mean():.4f}  std={v.std():.4f}"
                      f"  min={v.min():.4f}  max={v.max():.4f}")

    return feat_df


# ── Legacy compatibility: build_structural_features (old name) ────────────

def build_structural_features(df: pd.DataFrame,
                              verbose: bool = False) -> np.ndarray:
    """
    Legacy interface — returns np.ndarray (N, 13), float32.

    Calls build_features_structural() internally, converts NaN→0
    for backward compatibility with existing consumers.
    """
    feat_df = build_features_structural(df, verbose=verbose)
    arr = feat_df.values.astype(np.float32)
    # Legacy: NaN → 0 for backward compat
    arr = np.nan_to_num(arr, nan=0.0, posinf=3.0, neginf=-3.0)
    return arr


# ── Cache helpers ─────────────────────────────────────────────────────────

def generate_and_cache_features_structural(
    df: pd.DataFrame,
    cache_file: str,
) -> np.ndarray:
    """V4/V5 interface: (df, cache_file) → np.ndarray shape (N, 13).

    Used by pretrain_bc.py / train_quantum_v2.py with --feat-ver structural.
    """
    import os
    if os.path.exists(cache_file):
        arr = np.load(cache_file)
        print(f"[structural] Loaded cache: {cache_file}  shape={arr.shape}")
        return arr
    feat_df = build_features_structural(df, verbose=False)
    arr = feat_df.values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=3.0, neginf=-3.0)
    np.save(cache_file, arr)
    print(f"[structural] Built + saved: {cache_file}  shape={arr.shape}")
    return arr


def generate_and_cache_structural(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    cache_dir: str = "data",
    force_rebuild: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Cache-first loader. Raises RuntimeError if cache missing (df must be
    provided externally via build_features_structural).
    """
    import os
    s = start_date.replace("-", "")
    e = end_date.replace("-", "")
    cache_path = os.path.join(
        cache_dir, f"feat_structural_{symbol}_{timeframe}_{s}_{e}.npy"
    )

    if not force_rebuild and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if verbose:
            print(f"[structural] Loaded cache: {cache_path}  shape={arr.shape}")
        return arr

    raise RuntimeError(
        f"[structural] Cache not found: {cache_path}\n"
        "Call build_features_structural(df) directly with your DataFrame."
    )


# ── Self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  features_structural.py -- Self-Test (13-dim)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_bars = 200  # Need enough bars for Z_WINDOW warmup

    close = 50000.0 + np.cumsum(rng.normal(0, 100, n_bars))
    close = np.maximum(close, 100.0)
    high = close + rng.uniform(0, 300, n_bars)
    low = close - rng.uniform(0, 300, n_bars)
    low = np.maximum(low, 1.0)
    volume = rng.uniform(1e6, 1e7, n_bars)

    df = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n_bars)),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "funding_rate": rng.normal(0.0001, 0.0005, n_bars),
        "open_interest": 1e9 + np.cumsum(rng.normal(0, 1e7, n_bars)),
        "taker_buy_volume": volume * rng.uniform(0.4, 0.6, n_bars),
    })

    # Test build_features_structural
    feat_df = build_features_structural(df, verbose=True)

    assert isinstance(feat_df, pd.DataFrame), "Must return DataFrame"
    assert feat_df.shape == (n_bars, 13), f"Expected ({n_bars}, 13), got {feat_df.shape}"
    assert list(feat_df.columns) == FEAT_COLUMNS, "Column names mismatch"
    assert feat_df.dtypes.unique().tolist() == [np.float64], "Must be float64"

    # Check warmup: first Z_WINDOW-1 bars should be NaN
    warmup_nans = feat_df.iloc[:Z_WINDOW - 1, 0].isna().sum()
    assert warmup_nans == Z_WINDOW - 1, (
        f"Expected {Z_WINDOW - 1} warmup NaNs, got {warmup_nans}"
    )

    # Check post-warmup: should have valid values
    valid = feat_df.iloc[Z_WINDOW:].dropna()
    assert len(valid) > 0, "No valid rows after warmup"
    assert not np.isinf(valid.values).any(), "Inf in features"

    # Check clipping: all values should be in [-5, 5] (or NaN)
    non_nan = feat_df.values[~np.isnan(feat_df.values)]
    assert non_nan.min() >= -5.0, f"Min {non_nan.min()} < -5"
    assert non_nan.max() <= 5.0, f"Max {non_nan.max()} > 5"

    # Test legacy interface
    arr = build_structural_features(df)
    assert arr.shape == (n_bars, 13), f"Legacy: expected ({n_bars}, 13)"
    assert arr.dtype == np.float32, "Legacy must be float32"
    assert not np.isnan(arr).any(), "Legacy: NaN should be replaced with 0"

    print("\n  [PASS] All features_structural.py tests passed!")
    print(f"  [PASS] Exactly {FEAT_DIM} features: {FEAT_COLUMNS}")
    print("=" * 60)
