"""Bybit Mainnet public market data client (REST + optional WS)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv

from src.utils.logging import setup_logger

logger = setup_logger("bybit_mainnet")


class TTLCache:
    """Simple in-memory TTL cache: {key: (value, expires_at)}."""

    def __init__(self, ttl: int = 10):
        self.ttl = ttl
        self._store: dict = {}

    def get(self, key: str):
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value, ttl: int | None = None):
        ttl = ttl if ttl is not None else self.ttl
        self._store[key] = (value, time.time() + ttl)

    def clear(self):
        self._store.clear()


REST_BASE = "https://api.bybit.com"
# Bybit V5 public WebSocket endpoints by category.
# "future" is NOT a Bybit V5 category name -- dated futures live under:
#   linear  = USDT/USDC perpetuals + USDC quarterly futures  (e.g. BTCUSDT)
#   inverse = Coin-margined perpetuals + dated quarterly      (e.g. BTCUSD, BTCUSDM25)
#   option  = BTC/ETH/SOL vanilla options                    (e.g. BTC-29MAR25-80000-C)
WS_PUBLIC = {
    "linear":  "wss://stream.bybit.com/v5/public/linear",
    "inverse": "wss://stream.bybit.com/v5/public/inverse",
    "spot":    "wss://stream.bybit.com/v5/public/spot",
    "option":  "wss://stream.bybit.com/v5/public/option",
}

# Kline interval strings accepted by Bybit REST + WS.
# Options use the same interval codes but only liquid strikes have kline data.
INTERVAL_MAP = {
    "1m":  "1",
    "3m":  "3",
    "5m":  "5",
    "15m": "15",
    "30m": "30",
    "1h":  "60",
    "2h":  "120",
    "4h":  "240",
    "1d":  "D",
    "1w":  "W",
    "1M":  "M",
}


def _to_ts_str(ms: int) -> str:
    return datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_kline_rows(rows: list[list[Any]]) -> pd.DataFrame:
    """Parse raw kline row arrays → DataFrame.

    Bybit K-line row format: [startTime, open, high, low, close, volume, turnover]
    deep_candle  = np.log1p(turnover)          — candle trade-density
    liq_heatmap  = wick_length × volume        — liquidation intensity proxy
    """
    import numpy as np  # already imported at module top; import here for safety
    if not rows:
        return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume",
                     "deep_candle", "liq_heatmap"]
        )
    parsed = []
    for r in rows:
        ts_ms = int(r[0])
        o = float(r[1])
        h = float(r[2])
        lo = float(r[3])
        c = float(r[4])
        vol = float(r[5])
        turnover = float(r[6]) if len(r) > 6 else 0.0
        # Wick length: larger of upper or lower wick
        wick = max(h - max(o, c), min(o, c) - lo)
        parsed.append(
            {
                "ts": _to_ts_str(ts_ms),
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": vol,
                "deep_candle": float(np.log1p(max(turnover, 0.0))),
                "liq_heatmap": float(wick * vol),
            }
        )
    return pd.DataFrame(parsed)


def parse_rest_kline(payload: Dict[str, Any]) -> pd.DataFrame:
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    rows = result.get("list", []) if isinstance(result, dict) else []
    return _normalize_kline_rows(list(reversed(rows)))


def parse_ws_kline_message(payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    data = payload.get("data", []) if isinstance(payload, dict) else []
    for item in data:
        rows.append(
            [
                item.get("start"),
                item.get("open"),
                item.get("high"),
                item.get("low"),
                item.get("close"),
                item.get("volume"),
                item.get("turnover", 0),  # index 6 — used by deep_candle
            ]
        )
    return _normalize_kline_rows(rows)


class BybitMainnetClient:
    """Fetches Bybit mainnet kline data. WS optional for low latency updates."""

    def __init__(self, cache_ttl: int = 10, category: str = "linear"):
        load_dotenv()
        self._cache = TTLCache(cache_ttl)
        self._last_good: Optional[pd.DataFrame] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_stop = threading.Event()
        self.category = category
        self.api_key = os.getenv("BYBIT_API_KEY", "")

    def get_data_source(self) -> str:
        return "BybitMainnet"

    def clear_cache(self):
        self._cache.clear()

    def fetch_server_time_ms(self) -> int | None:
        try:
            url = f"{REST_BASE}/v5/market/time"
            req = Request(url, headers={"User-Agent": "TerminalQuantSuite/1.0"})
            with urlopen(req, timeout=6) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("retCode", 0) != 0:
                raise ValueError(payload.get("retMsg", "Bybit error"))
            result = payload.get("result", {}) if isinstance(payload, dict) else {}
            if "timeSecond" in result:
                return int(result["timeSecond"]) * 1000
            if "timeNano" in result:
                return int(result["timeNano"]) // 1_000_000
        except Exception:
            return None
        return None

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        days_back: int = 3,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        interval = INTERVAL_MAP.get(timeframe, timeframe)
        # Default limit: if not provided, calc from days_back
        # 5m: 12/hr * 24 * days_back.
        if limit is None:
            # Conservative estimate (assuming 1m) to get enough data
            limit = max(10, days_back * 24 * 60)
        
        # Bybit limit per request is 1000 (usually 200-1000)
        # We allow fetching more via pagination
        target_limit = limit
        collected_dfs = []
        current_end = end_ms
        fetched_count = 0
        
        # Max loop to prevent infinite recursion
        max_loops = 20
        
        for _ in range(max_loops):
            if fetched_count >= target_limit:
                break
            
            req_size = min(1000, target_limit - fetched_count)
            
            cache_key = f"bybit:{symbol}:{interval}:{req_size}:{current_end or ''}"
            cached = self._cache.get(cache_key) if current_end is None and self._cache.ttl > 0 else None
            
            if cached is not None:
                new_df = cached
            else:
                try:
                    params = {
                        "category": self.category,
                        "symbol": symbol,
                        "interval": interval,
                        "limit": req_size,
                    }
                    if current_end is not None:
                        params["end"] = int(current_end)
                        
                    url = f"{REST_BASE}/v5/market/kline?{urlencode(params)}"
                    headers = {"User-Agent": "TerminalQuantSuite/1.0"}
                    if self.api_key:
                        headers["X-BAPI-API-KEY"] = self.api_key
                    
                    req = Request(url, headers=headers)
                    with urlopen(req, timeout=10) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                        
                    if payload.get("retCode", 0) != 0:
                        logger.warning(f"Bybit error: {payload.get('retMsg')}")
                        break
                        
                    new_df = parse_rest_kline(payload)
                    
                    if not new_df.empty:
                        # Cache only the latest chunk if it's "live" (end_ms=None)
                        if self._cache.ttl > 0 and current_end is None:
                            self._cache.set(cache_key, new_df, ttl=max(1, self._cache.ttl))
                except Exception as e:
                    logger.warning(f"Fetch failed: {e}")
                    break
            
            if new_df.empty:
                break
                
            # Add to collection (newest is last in DataFrame, but we want to prepend older data)
            # new_df is [Oldest ... Newest] of that chunk
            collected_dfs.append(new_df)
            fetched_count += len(new_df)
            
            # Prepare next pagination (go backwards from oldest of this chunk)
            # ts is "YYYY-mm-dd HH:MM:SS"
            # We need ms for 'end' param
            try:
                oldest_ts_str = new_df.iloc[0]["ts"]
                dt = datetime.strptime(oldest_ts_str, "%Y-%m-%d %H:%M:%S")
                # Subtract 1 interval to avoid overlap? Or just use timestamp.
                # Bybit 'end' is inclusive? No, depends.
                # Usually we pass the timestamp of (oldest - 1ms) or just oldest.
                # Safest is oldest timestamp in ms.
                # Bybit returns data ending at or before 'end'.
                # So we want (oldest_ts - interval).
                # Simplified: just use oldest_ts_ms - 1000 (1 sec)
                # But wait, interval step. 
                # Let's rely on standard practice: oldest_ts_ms - (interval_ms) ?
                # Actually, Bybit 'end' means "End time".
                # If we pass T, it returns klines up to T.
                # So we should pass T of the *previous* candle.
                # dt is the Open Time of the oldest candle.
                # So we want candles ending before that Open Time.
                # So pass dt timestamp (ms) - 1.
                oldest_ms = int(dt.timestamp() * 1000)
                current_end = oldest_ms - 1
            except Exception:
                break
                
        if not collected_dfs:
            return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume",
                     "deep_candle", "liq_heatmap"]
        )
            
        # collected_dfs are [Chunk1(New), Chunk2(Older), ...] if we just appended?
        # No, fetch 1 (limit=1000, end=None) -> [T-999...T] (Newest)
        # fetch 2 (limit=1000, end=T-999-1) -> [T-1999...T-1000] (Older)
        # So collected_dfs = [NewestChunk, OlderChunk, ...]
        # We need to reverse this list before concat to get [Oldest ... Newest]?
        # concat([Newest, Older]) -> [Newest..., Older...] -> Wrong order.
        # We want [Older..., Newest...].
        # So reverse collected_dfs.
        
        final_df = pd.concat(reversed(collected_dfs), ignore_index=True)
        
        # Sort just in case and drop dups
        final_df.sort_values("ts", inplace=True)
        final_df.drop_duplicates(subset=["ts"], inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        # Trim to exact limit if over-fetched
        if len(final_df) > target_limit:
            final_df = final_df.iloc[-target_limit:].reset_index(drop=True)
            
        self._last_good = final_df
        return final_df

    def start_kline_stream(
        self,
        symbol: str,
        timeframe: str,
        on_update: Callable[[pd.DataFrame], None],
    ) -> bool:
        try:
            import websocket  # type: ignore
        except Exception:
            return False

        if self._ws_thread and self._ws_thread.is_alive():
            return True

        interval = INTERVAL_MAP.get(timeframe, timeframe)
        topic = f"kline.{interval}.{symbol}"
        ws_base = WS_PUBLIC.get(self.category, WS_PUBLIC["linear"])

        def _run():
            def on_message(ws, message):  # noqa: ANN001
                try:
                    payload = json.loads(message)
                    if payload.get("topic") == topic:
                        df = parse_ws_kline_message(payload)
                        if not df.empty:
                            on_update(df)
                except Exception:
                    pass

            def on_open(ws):  # noqa: ANN001
                sub = {"op": "subscribe", "args": [topic]}
                ws.send(json.dumps(sub))

            ws = websocket.WebSocketApp(
                ws_base,
                on_open=on_open,
                on_message=on_message,
            )
            while not self._ws_stop.is_set():
                ws.run_forever(ping_interval=20, ping_timeout=10)
                time.sleep(1)

        self._ws_stop.clear()
        self._ws_thread = threading.Thread(target=_run, daemon=True)
        self._ws_thread.start()
        return True

    def stop_stream(self):
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_stop.set()

    def fetch_bulk_history(
        self,
        symbol: str,
        timeframe: str = "5m",
        years: int = 5,
        end_ms: Optional[int] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> pd.DataFrame:
        """Fetch large historical kline dataset with full pagination.

        Parameters
        ----------
        end_ms      : Optional upper bound in milliseconds (UTC).
                      Pagination begins from this timestamp backwards.
                      Defaults to None (latest available candle).
        on_progress : callback(fetched, total_est, status_msg)
        """
        interval = INTERVAL_MAP.get(timeframe, timeframe)
        minutes_map = {"1": 1, "5": 5, "15": 15, "30": 30, "60": 60, "D": 1440}
        mpc = minutes_map.get(interval, 5)
        total_est = int(years * 365.25 * 24 * 60 / mpc)
        max_calls = (total_est // 1000) + 10

        collected: list[pd.DataFrame] = []
        current_end: Optional[int] = end_ms  # start pagination cursor at end_ms if given
        fetched = 0

        for call_idx in range(max_calls):
            if fetched >= total_est:
                break

            req_size = min(1000, total_est - fetched)
            params: Dict[str, Any] = {
                "category": self.category,
                "symbol": symbol,
                "interval": interval,
                "limit": req_size,
            }
            if current_end is not None:
                params["end"] = int(current_end)

            try:
                url = f"{REST_BASE}/v5/market/kline?{urlencode(params)}"
                headers: Dict[str, str] = {"User-Agent": "TerminalQuantSuite/1.0"}
                if self.api_key:
                    headers["X-BAPI-API-KEY"] = self.api_key

                req = Request(url, headers=headers)
                with urlopen(req, timeout=15) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))

                if payload.get("retCode", 0) != 0:
                    logger.warning(f"Bybit bulk fetch error: {payload.get('retMsg')}")
                    break

                new_df = parse_rest_kline(payload)
                if new_df.empty:
                    break

                collected.append(new_df)
                fetched += len(new_df)

                # Move end cursor backwards
                oldest_ts_str = new_df.iloc[0]["ts"]
                dt = datetime.strptime(oldest_ts_str, "%Y-%m-%d %H:%M:%S")
                current_end = int(dt.timestamp() * 1000) - 1

                if on_progress:
                    on_progress(fetched, total_est, f"Fetched {fetched:,}/{total_est:,}")

                # Rate limit: ~10 requests/sec
                if call_idx > 0 and call_idx % 10 == 0:
                    time.sleep(1.0)
                else:
                    time.sleep(0.08)

                # Stop if API returned fewer than requested (no more data)
                if len(new_df) < req_size:
                    break

            except Exception as e:
                logger.warning(f"Bulk fetch call {call_idx} failed: {e}")
                time.sleep(2.0)
                continue

        if not collected:
            return pd.DataFrame(
            columns=["ts", "open", "high", "low", "close", "volume",
                     "deep_candle", "liq_heatmap"]
        )

        final_df = pd.concat(list(reversed(collected)), ignore_index=True)
        final_df.sort_values("ts", inplace=True)
        final_df.drop_duplicates(subset=["ts"], inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        logger.info(f"Bulk fetch complete: {len(final_df):,} candles for {symbol} {timeframe} ({years}yr)")
        return final_df

    def fetch_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        try:
            params = {
                "category": self.category,
                "symbol": symbol,
                "limit": int(limit),
            }
            url = f"{REST_BASE}/v5/market/orderbook?{urlencode(params)}"
            req = Request(url, headers={"User-Agent": "TerminalQuantSuite/1.0"})
            with urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("retCode", 0) != 0:
                raise ValueError(payload.get("retMsg", "Bybit error"))
            result = payload.get("result", {})
            return {
                "bids": result.get("b", []),
                "asks": result.get("a", []),
                "ts": result.get("ts", ""),
            }
        except Exception:
            return {"bids": [], "asks": [], "ts": ""}

    def fetch_funding_history(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        cache_dir: str = "data",
    ) -> pd.DataFrame:
        """Fetch historical funding rates for a symbol over [start_ms, end_ms].

        Bybit funds every 8 hours → ~3 records/day.
        Returns DataFrame[ts_ms (int), funding_rate (float)] sorted ascending.
        Results cached to disk as {cache_dir}/funding_{symbol}_{start_ms}_{end_ms}.csv.
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"funding_{symbol}_{start_ms}_{end_ms}.csv"
        )
        if os.path.exists(cache_path):
            try:
                df_c = pd.read_csv(cache_path)
                if not df_c.empty and "ts_ms" in df_c.columns:
                    logger.info(f"[funding] Loaded cache: {len(df_c)} records")
                    return df_c
            except Exception:
                pass

        all_records: list = []
        current_end = end_ms

        for _ in range(300):          # max 300 pages × 200 = 60 000 records (~54yr)
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": 200,
                "endTime": int(current_end),
            }
            url = f"{REST_BASE}/v5/market/funding/history?{urlencode(params)}"
            try:
                req = Request(url, headers={"User-Agent": "TerminalQuantSuite/1.0"})
                with urlopen(req, timeout=10) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("retCode", 0) != 0:
                    logger.warning(f"[funding] API error: {payload.get('retMsg')}")
                    break
                rows = payload.get("result", {}).get("list", [])
                if not rows:
                    break
                for r in rows:
                    ts_ms = int(r.get("fundingRateTimestamp", 0))
                    rate  = float(r.get("fundingRate", 0.0))
                    all_records.append({"ts_ms": ts_ms, "funding_rate": rate})
                # oldest entry in this page
                oldest_ms = min(int(r.get("fundingRateTimestamp", current_end)) for r in rows)
                if oldest_ms <= start_ms:
                    break
                current_end = oldest_ms - 1
                time.sleep(0.08)
            except Exception as e:
                logger.warning(f"[funding] fetch error: {e}")
                break

        if not all_records:
            return pd.DataFrame(columns=["ts_ms", "funding_rate"])

        df = pd.DataFrame(all_records)
        df.drop_duplicates(subset=["ts_ms"], inplace=True)
        df.sort_values("ts_ms", inplace=True)
        df = df[df["ts_ms"] >= start_ms].reset_index(drop=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"[funding] Fetched {len(df)} records for {symbol}")
        return df

    def fetch_open_interest_history(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        interval: str = "15min",
        cache_dir: str = "data",
    ) -> pd.DataFrame:
        """Fetch historical Open Interest (OI) data from Bybit V5.

        Bybit /v5/market/open-interest  intervalTime: 5min,15min,30min,1h,4h,1d
        Returns DataFrame[ts_ms (int), open_interest (float)] sorted ascending.
        Results cached to {cache_dir}/oi_{symbol}_{interval}_{start_ms}_{end_ms}.csv
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"oi_{symbol}_{interval}_{start_ms}_{end_ms}.csv"
        )
        if os.path.exists(cache_path):
            try:
                df_c = pd.read_csv(cache_path)
                if not df_c.empty and "ts_ms" in df_c.columns:
                    logger.info(f"[OI] Loaded cache: {len(df_c)} records")
                    return df_c
            except Exception:
                pass

        all_records: list = []
        current_end = end_ms

        for _ in range(500):          # max 500 pages × 200 = 100 000 bars
            params = {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": interval,
                "limit": 200,
                "endTime": int(current_end),
            }
            url = f"{REST_BASE}/v5/market/open-interest?{urlencode(params)}"
            try:
                req = Request(url, headers={"User-Agent": "TerminalQuantSuite/1.0"})
                with urlopen(req, timeout=10) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("retCode", 0) != 0:
                    logger.warning(f"[OI] API error: {payload.get('retMsg')}")
                    break
                rows = payload.get("result", {}).get("list", [])
                if not rows:
                    break
                for r in rows:
                    ts_ms = int(r.get("timestamp", 0))
                    oi    = float(r.get("openInterest", 0.0))
                    all_records.append({"ts_ms": ts_ms, "open_interest": oi})
                oldest_ms = min(int(r.get("timestamp", current_end)) for r in rows)
                if oldest_ms <= start_ms:
                    break
                current_end = oldest_ms - 1
                time.sleep(0.08)
            except Exception as e:
                logger.warning(f"[OI] fetch error: {e}")
                break

        if not all_records:
            return pd.DataFrame(columns=["ts_ms", "open_interest"])

        df = pd.DataFrame(all_records)
        df.drop_duplicates(subset=["ts_ms"], inplace=True)
        df.sort_values("ts_ms", inplace=True)
        df = df[df["ts_ms"] >= start_ms].reset_index(drop=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"[OI] Fetched {len(df)} records for {symbol}")
        return df

    def fetch_liquidation_history(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        cache_dir: str = "data",
    ) -> pd.DataFrame:
        """Fetch historical liquidation records from Bybit V5.

        Endpoint: GET /v5/market/liquidation
        side: "Sell" = long liquidated, "Buy" = short liquidated
        Returns DataFrame[ts_ms, side, price, qty, usd_value] sorted ascending.
        Note: Bybit typically retains ~30 days of liquidation history.
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"liquidation_{symbol}_{start_ms}_{end_ms}.csv"
        )
        if os.path.exists(cache_path):
            try:
                df_c = pd.read_csv(cache_path)
                if not df_c.empty and "ts_ms" in df_c.columns:
                    logger.info(f"[liq] Loaded cache: {len(df_c)} records")
                    return df_c
            except Exception:
                pass

        all_records: list = []
        cursor = ""

        for _ in range(2000):
            params = {
                "category": "linear",
                "symbol":   symbol,
                "limit":    200,
                "startTime": int(start_ms),
                "endTime":   int(end_ms),
            }
            if cursor:
                params["cursor"] = cursor
            url = f"{REST_BASE}/v5/market/liquidation?{urlencode(params)}"
            try:
                req = Request(url, headers={"User-Agent": "TerminalQuantSuite/1.0"})
                with urlopen(req, timeout=10) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("retCode", 0) != 0:
                    logger.warning(f"[liq] API error: {payload.get('retMsg')}")
                    break
                result = payload.get("result", {})
                rows   = result.get("list", [])
                if not rows:
                    break
                for r in rows:
                    ts_ms = int(r.get("updatedTime", 0))
                    side  = r.get("side", "")
                    price = float(r.get("price", 0.0))
                    qty   = float(r.get("qty", 0.0))
                    all_records.append({
                        "ts_ms":     ts_ms,
                        "side":      side,
                        "price":     price,
                        "qty":       qty,
                        "usd_value": price * qty,
                    })
                cursor = result.get("nextPageCursor", "")
                if not cursor:
                    break
                time.sleep(0.08)
            except Exception as e:
                logger.warning(f"[liq] fetch error: {e}")
                break

        if not all_records:
            return pd.DataFrame(columns=["ts_ms", "side", "price", "qty", "usd_value"])

        df = pd.DataFrame(all_records)
        df.drop_duplicates(inplace=True)
        df.sort_values("ts_ms", inplace=True)
        df = df[(df["ts_ms"] >= start_ms) & (df["ts_ms"] <= end_ms)].reset_index(drop=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"[liq] Fetched {len(df)} records for {symbol}")
        return df

    # ── Signed private endpoints ──────────────────────────────────────────

    def _sign_request(self, params: dict) -> dict:
        """Add HMAC-SHA256 signature to params for private Bybit V5 endpoints."""
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        param_str = ts + self.api_key + recv_window + urlencode(params)
        signature = hmac.new(
            api_secret.encode("utf-8"), param_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
            "User-Agent": "TerminalQuantSuite/1.0",
        }

    def _post_private(self, path: str, body: dict) -> dict:
        """POST to a private Bybit V5 endpoint with HMAC signature."""
        if not self.api_key or not os.getenv("BYBIT_API_SECRET", ""):
            raise RuntimeError("BYBIT_API_KEY / BYBIT_API_SECRET not set")
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        body_str = json.dumps(body)
        param_str = ts + self.api_key + recv_window + body_str
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        signature = hmac.new(
            api_secret.encode("utf-8"), param_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
            "User-Agent": "TerminalQuantSuite/1.0",
        }
        url = f"{REST_BASE}{path}"
        data = body_str.encode("utf-8")
        req = Request(url, data=data, headers=headers, method="POST")
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_private(self, path: str, params: dict) -> dict:
        """GET from a private Bybit V5 endpoint with HMAC signature."""
        if not self.api_key or not os.getenv("BYBIT_API_SECRET", ""):
            raise RuntimeError("BYBIT_API_KEY / BYBIT_API_SECRET not set")
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        query_str = urlencode(params)
        param_str = ts + self.api_key + recv_window + query_str
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        signature = hmac.new(
            api_secret.encode("utf-8"), param_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": recv_window,
            "User-Agent": "TerminalQuantSuite/1.0",
        }
        url = f"{REST_BASE}{path}?{query_str}"
        req = Request(url, headers=headers)
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def place_order(
        self,
        symbol: str,
        side: str,           # "Buy" | "Sell"
        qty: str,            # e.g. "0.001"
        order_type: str = "Market",
        reduce_only: bool = False,
        tp_price: Optional[str] = None,
        sl_price: Optional[str] = None,
    ) -> dict:
        """
        Place a linear perpetual market order via Bybit V5 /v5/order/create.

        Returns the full API response dict. Raises on HTTP error.
        Check retCode == 0 for success.
        """
        body: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "reduceOnly": reduce_only,
            "timeInForce": "IOC" if order_type == "Market" else "GTC",
        }
        if tp_price:
            body["takeProfit"] = tp_price
            body["tpTriggerBy"] = "LastPrice"
        if sl_price:
            body["stopLoss"] = sl_price
            body["slTriggerBy"] = "LastPrice"
        return self._post_private("/v5/order/create", body)

    def set_tp_sl(
        self,
        symbol: str,
        tp_price: Optional[str] = None,
        sl_price: Optional[str] = None,
        position_idx: int = 0,
    ) -> dict:
        """
        Set TP/SL on an existing position via /v5/position/set-tpsl.
        position_idx: 0=one-way, 1=hedge-long, 2=hedge-short
        """
        body: Dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "positionIdx": position_idx,
            "tpTriggerBy": "LastPrice",
            "slTriggerBy": "LastPrice",
        }
        if tp_price:
            body["takeProfit"] = tp_price
        if sl_price:
            body["stopLoss"] = sl_price
        return self._post_private("/v5/position/set-tpsl", body)

    def get_positions(self, symbol: str) -> list:
        """
        Return list of open positions for symbol from /v5/position/list.
        Each item: {symbol, side, size, avgPrice, unrealisedPnl, ...}
        """
        params = {"category": "linear", "symbol": symbol}
        payload = self._get_private("/v5/position/list", params)
        if payload.get("retCode", -1) != 0:
            logger.warning(f"[positions] error: {payload.get('retMsg')}")
            return []
        return payload.get("result", {}).get("list", [])

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> dict:
        """Return wallet balance info from /v5/account/wallet-balance."""
        params = {"accountType": account_type}
        payload = self._get_private("/v5/account/wallet-balance", params)
        if payload.get("retCode", -1) != 0:
            logger.warning(f"[wallet] error: {payload.get('retMsg')}")
            return {}
        items = payload.get("result", {}).get("list", [])
        return items[0] if items else {}

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel all active orders for symbol via /v5/order/cancel-all."""
        body = {"category": "linear", "symbol": symbol}
        return self._post_private("/v5/order/cancel-all", body)

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage for symbol via /v5/position/set-leverage."""
        body = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        return self._post_private("/v5/position/set-leverage", body)

    def fetch_open_interest_recent(
        self,
        symbol: str,
        interval: str = "15min",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch the most recent N bars of Open Interest (live TUI use)."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": interval,
            "limit": limit,
        }
        url = f"{REST_BASE}/v5/market/open-interest?{urlencode(params)}"
        try:
            req = Request(url, headers={"User-Agent": "TerminalQuantSuite/1.0"})
            with urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("retCode", 0) != 0:
                logger.warning(f"[OI] Recent API error: {payload.get('retMsg')}")
                return pd.DataFrame(columns=["ts_ms", "open_interest"])
            rows = payload.get("result", {}).get("list", [])
            if not rows:
                return pd.DataFrame(columns=["ts_ms", "open_interest"])
            records = [
                {"ts_ms": int(r["timestamp"]), "open_interest": float(r["openInterest"])}
                for r in rows
            ]
            df = pd.DataFrame(records)
            df.sort_values("ts_ms", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        except Exception as e:
            logger.warning(f"[OI] Recent fetch error: {e}")
            return pd.DataFrame(columns=["ts_ms", "open_interest"])
