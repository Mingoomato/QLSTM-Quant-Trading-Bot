"""
binance_client.py
─────────────────────────────────────────────────────────────────────────────
Binance PUBLIC market-data client (no API key required).

Purpose: fetch historical klines including taker_buy_base_asset_volume,
         which Bybit does NOT provide per candle.

Real CVD formula:
    real_delta = 2 × taker_buy_volume - total_volume
    (positive → buyers aggressive, negative → sellers aggressive)

Binance kline response column index:
  0  : Open time (ms)
  1  : Open
  2  : High
  3  : Low
  4  : Close
  5  : Volume (base asset)
  6  : Close time (ms)
  7  : Quote asset volume
  8  : Number of trades
  9  : Taker buy base asset volume   ← REAL CVD source
  10 : Taker buy quote asset volume
  11 : Ignore

Config (.env / environment variables):
  BINANCE_BASE_URL   Base REST URL (default: https://api.binance.com)
                     Mainland China mirror: https://api.binance.cc
                     Testnet             : https://testnet.binance.vision

  No API key needed for public market data (klines, ticker, etc).
  For account/order endpoints add: BINANCE_API_KEY, BINANCE_API_SECRET

Usage:
  client = BinancePublicClient()
  df = client.fetch_klines_with_taker("BTCUSDT", "15m",
           start="2022-01-01", end="2025-10-01")
  # df columns: ts, open, high, low, close, volume, taker_buy_volume, real_delta

Bulk history helper:
  df = fetch_binance_taker_history("BTCUSDT", "15m",
           start_date="2022-01-01", end_date="2025-10-01",
           cache_path="data/binance_taker_BTCUSDT_15m_20220101_20251001.csv")
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BINANCE_BASE_URL: str = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")

# Binance interval strings
INTERVAL_MAP = {
    "1m":  "1m",
    "3m":  "3m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "6h":  "6h",
    "12h": "12h",
    "1d":  "1d",
    "1w":  "1w",
}

_MAX_PER_REQUEST = 1000    # Binance hard limit per kline request
_RATE_LIMIT_SLEEP = 0.12   # 120ms between requests (~8 req/s, safe under 1200 weight/min)


def _ms(dt_str: str) -> int:
    """Parse 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' → UTC milliseconds."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date string: {dt_str!r}")


def _parse_kline_row(row: list) -> dict:
    """Parse one Binance kline array into a dict with taker fields."""
    open_ms  = int(row[0])
    ts_str   = datetime.utcfromtimestamp(open_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    volume   = float(row[5])
    taker_buy = float(row[9])          # taker buy base asset volume
    real_delta = 2.0 * taker_buy - volume  # positive = net buying pressure
    return {
        "ts":               ts_str,
        "open":             float(row[1]),
        "high":             float(row[2]),
        "low":              float(row[3]),
        "close":            float(row[4]),
        "volume":           volume,
        "taker_buy_volume": taker_buy,
        "real_delta":       real_delta,
    }


class BinancePublicClient:
    """
    Fetches Binance public kline data including per-candle taker volume.

    No API key required for market data endpoints.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or BINANCE_BASE_URL).rstrip("/")

    # ── Low-level request ──────────────────────────────────────────────────
    def _get(self, path: str, params: dict, retries: int = 3) -> list:
        url = f"{self.base_url}{path}?{urlencode(params)}"
        for attempt in range(retries):
            try:
                req = Request(url, headers={"User-Agent": "QuantTradingSuite/1.0"})
                with urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except HTTPError as e:
                if e.code == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"  [binance] Rate limited (429), waiting {wait}s ...")
                    time.sleep(wait)
                elif e.code == 418:
                    wait = 60
                    print(f"  [binance] IP banned (418), waiting {wait}s ...")
                    time.sleep(wait)
                else:
                    raise
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                print(f"  [binance] Request error ({exc}), retry {attempt+1}/{retries} ...")
                time.sleep(1.5)
        return []

    # ── Single window fetch ────────────────────────────────────────────────
    def fetch_klines_raw(self, symbol: str, interval: str,
                         start_ms: int, end_ms: int) -> pd.DataFrame:
        """Fetch up to 1000 klines in [start_ms, end_ms)."""
        iv = INTERVAL_MAP.get(interval, interval)
        params = {
            "symbol":    symbol.upper(),
            "interval":  iv,
            "startTime": start_ms,
            "endTime":   end_ms - 1,
            "limit":     _MAX_PER_REQUEST,
        }
        rows = self._get("/api/v3/klines", params)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([_parse_kline_row(r) for r in rows])

    # ── Bulk paginated fetch ───────────────────────────────────────────────
    def fetch_klines_with_taker(
        self,
        symbol:   str,
        interval: str,
        start:    str,
        end:      str,
        verbose:  bool = True,
    ) -> pd.DataFrame:
        """
        Fetch full historical klines with taker_buy_volume.

        Parameters
        ----------
        symbol   : e.g. 'BTCUSDT'
        interval : e.g. '15m'
        start    : 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
        end      : 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
        verbose  : print progress

        Returns
        -------
        DataFrame with columns:
          ts, open, high, low, close, volume, taker_buy_volume, real_delta
        """
        start_ms = _ms(start)
        end_ms   = _ms(end)
        now_ms   = int(time.time() * 1000)
        end_ms   = min(end_ms, now_ms)

        # Infer interval length in ms
        iv_ms_map = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000,
            "15m": 900_000, "30m": 1_800_000,
            "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
            "6h": 21_600_000, "12h": 43_200_000,
            "1d": 86_400_000, "1w": 604_800_000,
        }
        iv_ms    = iv_ms_map.get(interval, 900_000)
        total_ms = end_ms - start_ms
        est_bars = total_ms // iv_ms
        est_reqs = max(1, est_bars // _MAX_PER_REQUEST)

        if verbose:
            print(f"[binance] Fetching {symbol} {interval} "
                  f"{start} → {end}  "
                  f"(~{est_bars:,} bars, ~{est_reqs} requests) ...")

        chunks   = []
        cur_ms   = start_ms
        req_idx  = 0

        while cur_ms < end_ms:
            chunk = self.fetch_klines_raw(symbol, interval, cur_ms, end_ms)
            if chunk.empty:
                break

            chunks.append(chunk)
            last_open_ms = _ms(chunk["ts"].iloc[-1])
            cur_ms = last_open_ms + iv_ms
            req_idx += 1

            if verbose and req_idx % 10 == 0:
                pct = min(100.0, (cur_ms - start_ms) / max(total_ms, 1) * 100)
                print(f"  [binance] {req_idx} requests done  {pct:.0f}%  "
                      f"({len(chunk)} bars in last batch) ...")

            if len(chunk) < _MAX_PER_REQUEST:
                break   # Reached the end of available data

            time.sleep(_RATE_LIMIT_SLEEP)

        if not chunks:
            print("[binance] No data returned.")
            return pd.DataFrame(columns=[
                "ts", "open", "high", "low", "close",
                "volume", "taker_buy_volume", "real_delta"
            ])

        df = pd.concat(chunks, ignore_index=True)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)

        if verbose:
            print(f"[binance] Done: {len(df):,} bars  "
                  f"({df['ts'].iloc[0]} → {df['ts'].iloc[-1]})")
            taker_frac = (df["taker_buy_volume"] / (df["volume"] + 1e-10)).mean()
            print(f"  Avg taker buy ratio: {taker_frac:.3f}  "
                  f"(0.5 = balanced, >0.5 = buy-heavy)")

        return df


# ── Cached bulk history helper ─────────────────────────────────────────────

def fetch_binance_taker_history(
    symbol:     str   = "BTCUSDT",
    interval:   str   = "15m",
    start_date: str   = "2022-01-01",
    end_date:   str   = "2025-10-01",
    cache_path: Optional[str] = None,
    force:      bool  = False,
    verbose:    bool  = True,
) -> pd.DataFrame:
    """
    Fetch or load cached Binance klines with taker volume.

    The CSV cache contains: ts, taker_buy_volume, real_delta
    (only the CVD-relevant columns to keep file size small).

    Parameters
    ----------
    cache_path : if None, auto-generates path in data/ directory
    force      : re-download even if cache exists

    Returns
    -------
    DataFrame with columns: ts, taker_buy_volume, real_delta
    (merge with Bybit OHLCV on 'ts' column)
    """
    if cache_path is None:
        start_tag = start_date.replace("-", "")
        end_tag   = end_date.replace("-", "")
        cache_path = (
            f"data/binance_taker_{symbol}_{interval}_{start_tag}_{end_tag}.csv"
        )

    if not force and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, encoding="utf-8")
        if verbose:
            print(f"[binance] Loaded taker cache: {cache_path}  ({len(df):,} rows)")
        return df

    client = BinancePublicClient()
    df_full = client.fetch_klines_with_taker(
        symbol, interval, start_date, end_date, verbose=verbose
    )
    if df_full.empty:
        return df_full

    # Save only CVD-relevant columns to save disk space
    df_save = df_full[["ts", "taker_buy_volume", "real_delta"]].copy()
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    df_save.to_csv(cache_path, index=False, encoding="utf-8")
    if verbose:
        print(f"[binance] Saved taker cache: {cache_path}  ({len(df_save):,} rows)")

    return df_save


# ── Quick smoke test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("[test] Fetching 5 bars of BTCUSDT 15m from Binance ...")
    client = BinancePublicClient()
    df = client.fetch_klines_raw(
        "BTCUSDT", "15m",
        start_ms=_ms("2025-01-01"),
        end_ms=_ms("2025-01-01") + 5 * 900_000
    )
    print(df[["ts", "open", "close", "volume", "taker_buy_volume", "real_delta"]])
    print("\nReal delta > 0 means net buying pressure.")
    print("Real delta < 0 means net selling pressure.")
