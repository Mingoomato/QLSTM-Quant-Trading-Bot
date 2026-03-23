from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.features_structural import build_features_structural


DEFAULT_START = "2022-07-01"
DEFAULT_END = "2025-12-31"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_PROXY_Z_THRESHOLD = 2.0
DEFAULT_SPIKE_QUANTILE = 0.95
DEFAULT_PRECISION_GATE = 0.60


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate cascade proxy precision against liquidation spikes."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--proxy-z-threshold", type=float, default=DEFAULT_PROXY_Z_THRESHOLD)
    parser.add_argument("--spike-quantile", type=float, default=DEFAULT_SPIKE_QUANTILE)
    parser.add_argument("--precision-gate", type=float, default=DEFAULT_PRECISION_GATE)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="project_output")
    parser.add_argument(
        "--strict-actual",
        action="store_true",
        help="Fail instead of falling back to cached liq_heatmap labels.",
    )
    return parser.parse_args()


def _choose_training_cache(data_dir: Path, symbol: str, timeframe: str, start_date: str) -> Path:
    candidates = sorted(data_dir.glob(f"training_{symbol}_{timeframe}_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No training cache found for {symbol} {timeframe} in {data_dir}")

    start_tag = start_date.replace("-", "")
    preferred = [path for path in candidates if start_tag in path.stem]
    if preferred:
        return preferred[0]
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_market_history(data_dir: Path, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    path = _choose_training_cache(data_dir, symbol, timeframe, start_date)
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)].copy()
    if df.empty:
        raise ValueError(f"Training cache {path} has no rows inside {start_date}..{end_date}")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.attrs["source_path"] = str(path)
    return df


def _load_cached_timeseries(
    data_dir: Path,
    pattern: str,
    ts_col: str,
    value_col: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

    for path in sorted(data_dir.glob(pattern)):
        try:
            df = pd.read_csv(path, usecols=[ts_col, value_col])
        except Exception:
            continue
        if df.empty:
            continue
        df[ts_col] = pd.to_datetime(df[ts_col], unit="ms", utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col])
        df = df[(df[ts_col] >= start_ts) & (df[ts_col] <= end_ts)]
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["ts", value_col])

    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values(ts_col, inplace=True)
    merged.drop_duplicates(subset=[ts_col], keep="last", inplace=True)
    merged.rename(columns={ts_col: "ts"}, inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def _merge_funding_and_oi(df: pd.DataFrame, data_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    funding = _load_cached_timeseries(
        data_dir=data_dir,
        pattern="funding_*.csv",
        ts_col="ts_ms",
        value_col="funding_rate",
        start_date=start_date,
        end_date=end_date,
    )
    oi = _load_cached_timeseries(
        data_dir=data_dir,
        pattern="oi_*_1h_*.csv",
        ts_col="ts_ms",
        value_col="open_interest",
        start_date=start_date,
        end_date=end_date,
    )

    merged = df.copy()
    merged = merged.merge(funding, on="ts", how="left")
    merged = merged.merge(oi, on="ts", how="left")
    merged["funding_rate"] = merged.get("funding_rate", 0.0).ffill().fillna(0.0)
    merged["open_interest"] = merged.get("open_interest", 0.0).ffill().fillna(0.0)
    return merged


def _load_actual_liquidation_labels(
    data_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    frames: list[pd.DataFrame] = []

    for path in sorted(data_dir.glob(f"liquidation_{symbol}_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        required = {"ts_ms", "usd_value"}
        if not required.issubset(df.columns):
            continue
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True, errors="coerce").dt.floor("h")
        df = df.dropna(subset=["ts"])
        df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)]
        if df.empty:
            continue
        hourly = df.groupby("ts", as_index=False)["usd_value"].sum()
        hourly.rename(columns={"usd_value": "actual_liq_usd"}, inplace=True)
        frames.append(hourly)

    if not frames:
        return pd.DataFrame(columns=["ts", "actual_liq_usd"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.groupby("ts", as_index=False)["actual_liq_usd"].sum()
    merged.sort_values("ts", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def _newey_west_mean_se(values: np.ndarray, max_lag: int | None = None) -> tuple[float, int]:
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n <= 1:
        return float("nan"), 0

    mean = float(arr.mean())
    centered = arr - mean
    if max_lag is None:
        max_lag = int(math.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0))))
    max_lag = max(0, min(max_lag, n - 1))

    gamma0 = float(np.dot(centered, centered) / n)
    long_run_var = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_var += 2.0 * weight * gamma

    long_run_var = max(long_run_var, 0.0)
    se = math.sqrt(long_run_var / n)
    return float(se), max_lag


def _contingency_and_precision(
    trigger: pd.Series,
    spike: pd.Series,
) -> dict[str, Any]:
    predicted = trigger.astype(bool).to_numpy()
    actual = spike.astype(bool).to_numpy()

    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    predicted_positive = tp + fp
    actual_positive = tp + fn

    if predicted_positive == 0:
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "predicted_positive": predicted_positive,
            "actual_positive": actual_positive,
            "precision": 0.0,
            "nw_se": float("nan"),
            "nw_lag": 0,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    hit_seq = actual[predicted].astype(np.float64)
    precision = float(hit_seq.mean())
    nw_se, nw_lag = _newey_west_mean_se(hit_seq)
    ci_low = max(0.0, precision - 1.96 * nw_se) if not math.isnan(nw_se) else float("nan")
    ci_high = min(1.0, precision + 1.96 * nw_se) if not math.isnan(nw_se) else float("nan")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": predicted_positive,
        "actual_positive": actual_positive,
        "precision": precision,
        "nw_se": nw_se,
        "nw_lag": nw_lag,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _build_validation_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_dir = Path(args.data_dir)
    market = _load_market_history(data_dir, args.symbol, args.timeframe, args.start_date, args.end_date)
    merged = _merge_funding_and_oi(market, data_dir, args.start_date, args.end_date)

    feature_input = merged[
        ["ts", "open", "high", "low", "close", "volume", "funding_rate", "open_interest"]
    ].copy()
    features = build_features_structural(feature_input.set_index("ts"))
    features = features.reset_index().rename(columns={"index": "ts"})

    validation = merged.merge(features[["ts", "liq_long_z", "liq_short_z"]], on="ts", how="left")
    validation["cascade_proxy"] = validation[["liq_long_z", "liq_short_z"]].max(axis=1)

    actual = _load_actual_liquidation_labels(data_dir, args.symbol, args.start_date, args.end_date)
    metadata: dict[str, Any] = {
        "market_source": market.attrs.get("source_path", ""),
        "label_source": "",
        "label_note": "",
    }

    if not actual.empty:
        validation = validation.merge(actual, on="ts", how="left")
        validation["actual_liq_usd"] = validation["actual_liq_usd"].fillna(0.0)
        metadata["label_source"] = "actual_liquidation_cache"
        metadata["label_note"] = "Actual liquidation USD aggregated from cached Bybit liquidation records."
    else:
        if args.strict_actual:
            raise RuntimeError(
                f"No cached liquidation_{args.symbol}_*.csv files overlap the requested window."
            )
        validation["actual_liq_usd"] = validation["liq_heatmap"].astype(float)
        metadata["label_source"] = "cached_liq_heatmap_fallback"
        metadata["label_note"] = (
            "No actual liquidation cache was present, so the validator fell back to the cached "
            "hourly liq_heatmap column already stored in training history. This result is provisional."
        )

    validation.dropna(subset=["cascade_proxy", "actual_liq_usd"], inplace=True)
    validation.reset_index(drop=True, inplace=True)
    return validation, metadata


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    validation, metadata = _build_validation_frame(args)
    threshold = float(validation["actual_liq_usd"].quantile(args.spike_quantile))
    validation["actual_spike"] = validation["actual_liq_usd"] >= threshold
    validation["proxy_trigger"] = validation["cascade_proxy"] > args.proxy_z_threshold

    stats = _contingency_and_precision(validation["proxy_trigger"], validation["actual_spike"])
    keep_features = stats["precision"] >= args.precision_gate
    feature_decision = "keep" if keep_features else "remove"

    return {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "bars_evaluated": int(len(validation)),
        "proxy_trigger_z": float(args.proxy_z_threshold),
        "actual_spike_quantile": float(args.spike_quantile),
        "actual_spike_threshold_usd": float(threshold),
        "precision_gate": float(args.precision_gate),
        "precision": float(stats["precision"]),
        "precision_pct": float(stats["precision"] * 100.0),
        "newey_west_se": None if math.isnan(stats["nw_se"]) else float(stats["nw_se"]),
        "newey_west_lag": int(stats["nw_lag"]),
        "precision_ci_low": None if math.isnan(stats["ci_low"]) else float(stats["ci_low"]),
        "precision_ci_high": None if math.isnan(stats["ci_high"]) else float(stats["ci_high"]),
        "tp": int(stats["tp"]),
        "fp": int(stats["fp"]),
        "fn": int(stats["fn"]),
        "tn": int(stats["tn"]),
        "predicted_positive": int(stats["predicted_positive"]),
        "actual_positive": int(stats["actual_positive"]),
        "recommendation": feature_decision,
        "liq_long_z": feature_decision,
        "liq_short_z": feature_decision,
        **metadata,
    }


def _write_outputs(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cascade_proxy_validation.json"
    csv_path = output_dir / "cascade_proxy_validation.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    pd.DataFrame([result]).to_csv(csv_path, index=False)
    return json_path, csv_path


def main() -> int:
    args = _parse_args()
    result = _evaluate(args)
    json_path, csv_path = _write_outputs(result, Path(args.output_dir))

    ci_low = result["precision_ci_low"]
    ci_high = result["precision_ci_high"]
    ci_text = (
        "n/a"
        if ci_low is None or ci_high is None
        else f"[{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]"
    )

    print(
        f"label_source={result['label_source']}\n"
        f"bars_evaluated={result['bars_evaluated']}\n"
        f"proxy_trigger_z={result['proxy_trigger_z']:.2f}\n"
        f"actual_spike_threshold_usd={result['actual_spike_threshold_usd']:.3f}\n"
        f"contingency_table=TP:{result['tp']} FP:{result['fp']} FN:{result['fn']} TN:{result['tn']}\n"
        f"precision={result['precision_pct']:.2f}%\n"
        f"newey_west_se={result['newey_west_se'] if result['newey_west_se'] is not None else 'n/a'}\n"
        f"newey_west_lag={result['newey_west_lag']}\n"
        f"precision_95ci={ci_text}\n"
        f"recommendation={result['recommendation']} liq_long_z={result['liq_long_z']} liq_short_z={result['liq_short_z']}\n"
        f"note={result['label_note']}\n"
        f"json_output={json_path}\n"
        f"csv_output={csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
