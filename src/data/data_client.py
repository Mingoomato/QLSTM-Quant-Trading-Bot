"""Data client wrapper — Bybit Mainnet only.

Removed dead imports (bybit_testnet_private, openbb_client, mock_data
modules do not exist).  All data flows through BybitMainnetClient.
Non-mainnet paths return safe stubs so callers don't crash.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger("data_client")

from src.data.bybit_mainnet import BybitMainnetClient

_EMPTY_OHLCV_COLS = ["ts", "open", "high", "low", "close", "volume",
                     "deep_candle", "liq_heatmap"]


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_OHLCV_COLS)


class DataClient:
    def __init__(
        self,
        mode: str = "paper",
        cache_ttl: int = 30,
        bybit_env: str = "mainnet",
        bybit_category: str = "linear",
    ):
        self.mode = mode
        self._bybit: Optional[BybitMainnetClient] = None
        self._bybit_env = bybit_env
        self._bybit_category = bybit_category

        if bybit_env == "mainnet":
            self._bybit = BybitMainnetClient(cache_ttl=cache_ttl, category=bybit_category)
        else:
            logger.warning(
                f"bybit_env='{bybit_env}' requested but only 'mainnet' is supported. "
                "No Bybit client initialized."
            )

    def is_ready(self) -> bool:
        return self._bybit is not None

    def get_data_source(self) -> str:
        return self._bybit.get_data_source() if self._bybit else "unavailable"

    def clear_cache(self):
        if self._bybit:
            self._bybit.clear_cache()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        days_back: int = 3,
        end_ms: int | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if self._bybit:
            df = self._bybit.fetch_ohlcv(
                symbol, timeframe, days_back=days_back, end_ms=end_ms, limit=limit
            )
            if df is not None and len(df) > 0:
                return df
        return _empty_ohlcv()

    def fetch_server_time_ms(self) -> int | None:
        if self._bybit:
            try:
                return self._bybit.fetch_server_time_ms()
            except Exception:
                return None
        return None

    def fetch_news(self, symbol: str = "BTCUSDT", limit: int = 20) -> list[dict]:
        """News feed not available (openbb_client module absent). Returns empty list."""
        return []

    def start_kline_stream(self, symbol: str, timeframe: str, on_update) -> bool:
        if self._bybit:
            return self._bybit.start_kline_stream(symbol, timeframe, on_update)
        return False

    def stop_stream(self):
        if self._bybit:
            self._bybit.stop_stream()

    def fetch_account_snapshot(self, symbol: str | None = None) -> dict:
        """Private account data not available (bybit_testnet_private absent)."""
        return {"error": "Not enabled"}

    def start_private_stream(self, on_update) -> bool:
        """Private stream not available (bybit_testnet_private absent)."""
        return False

    def stop_private_stream(self):
        pass

    def fetch_orderbook(self, symbol: str, limit: int = 100) -> dict:
        if self._bybit:
            try:
                return self._bybit.fetch_orderbook(symbol, limit)
            except Exception:
                pass
        return {"bids": [], "asks": [], "ts": ""}

    def fetch_funding_history(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        cache_dir: str = "data",
    ) -> "pd.DataFrame":
        """Fetch historical funding rates via BybitMainnetClient.
        Returns DataFrame[ts_ms, funding_rate] or empty DataFrame on failure.
        """
        if self._bybit and isinstance(self._bybit, BybitMainnetClient):
            try:
                return self._bybit.fetch_funding_history(
                    symbol, start_ms, end_ms, cache_dir=cache_dir
                )
            except Exception as e:
                logger.warning(f"fetch_funding_history failed: {e}")
        return pd.DataFrame(columns=["ts_ms", "funding_rate"])

    def fetch_open_interest_history(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        interval: str = "15min",
        cache_dir: str = "data",
    ) -> "pd.DataFrame":
        """Fetch historical OI via BybitMainnetClient. Returns DataFrame[ts_ms, open_interest]."""
        if self._bybit and isinstance(self._bybit, BybitMainnetClient):
            try:
                return self._bybit.fetch_open_interest_history(
                    symbol, start_ms, end_ms, interval=interval, cache_dir=cache_dir
                )
            except Exception as e:
                logger.warning(f"fetch_open_interest_history failed: {e}")
        return pd.DataFrame(columns=["ts_ms", "open_interest"])

    def fetch_liquidation_history(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        cache_dir: str = "data",
    ) -> "pd.DataFrame":
        """Fetch real liquidation records via BybitMainnetClient (~30 days available).
        Returns DataFrame[ts_ms, side, price, qty, usd_value].
        side='Sell' = long liquidated, side='Buy' = short liquidated.
        """
        if self._bybit and isinstance(self._bybit, BybitMainnetClient):
            try:
                return self._bybit.fetch_liquidation_history(
                    symbol, start_ms, end_ms, cache_dir=cache_dir
                )
            except Exception as e:
                logger.warning(f"fetch_liquidation_history failed: {e}")
        return pd.DataFrame(columns=["ts_ms", "side", "price", "qty", "usd_value"])

    def fetch_open_interest_recent(
        self,
        symbol: str,
        interval: str = "15min",
        limit: int = 100,
    ) -> "pd.DataFrame":
        """Fetch recent OI bars for live TUI use."""
        if self._bybit and isinstance(self._bybit, BybitMainnetClient):
            try:
                return self._bybit.fetch_open_interest_recent(
                    symbol, interval=interval, limit=limit
                )
            except Exception as e:
                logger.warning(f"fetch_open_interest_recent failed: {e}")
        return pd.DataFrame(columns=["ts_ms", "open_interest"])

    def fetch_training_history(
        self,
        symbol: str,
        timeframe: str = "5m",
        start_date: str = "",
        end_ms: int | None = None,
        cache_tag: str | None = None,
        cache_dir: str = "data",
        cache_max_age_h: int = 24,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> pd.DataFrame:
        """Fetch historical dataset from start_date to end_ms (or now), cached to CSV.

        Parameters
        ----------
        start_date      : YYYY-MM-DD string. Defaults to 5 years ago.
        end_ms          : Optional end timestamp in milliseconds (UTC).
        cache_tag       : Optional override for the cache filename tag.
        cache_max_age_h : Hours before the cache file is considered stale.
        """
        from datetime import datetime as _dt

        if not self._bybit or not isinstance(self._bybit, BybitMainnetClient):
            raise RuntimeError(
                "fetch_training_history requires Bybit mainnet client. "
                "Create DataClient with bybit_env='mainnet'."
            )

        # Parse start_date → years back
        if start_date:
            try:
                start_dt = _dt.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Invalid start_date '{start_date}', using 5yr default")
                start_dt = _dt.utcnow() - pd.Timedelta(days=5 * 365)
        else:
            start_dt = _dt.utcnow() - pd.Timedelta(days=5 * 365)

        years = max(0.1, (_dt.utcnow() - start_dt).days / 365.25)

        file_tag = cache_tag if cache_tag else start_dt.strftime("%Y%m%d")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"training_{symbol}_{timeframe}_{file_tag}.csv"
        )

        # Serve from cache if fresh
        if os.path.exists(cache_path):
            age_h = (time.time() - os.path.getmtime(cache_path)) / 3600
            if age_h < cache_max_age_h:
                logger.info(f"Loading cached training data: {cache_path} (age={age_h:.1f}h)")
                df = pd.read_csv(cache_path, encoding="utf-8")
                logger.info(f"Loaded {len(df):,} cached candles")
                return df

        logger.info(f"Fetching {years:.1f}yr {timeframe} data for {symbol} (from {start_date})...")
        df = self._bybit.fetch_bulk_history(
            symbol, timeframe, years=years, end_ms=end_ms, on_progress=on_progress,
        )

        if not df.empty and start_date:
            df = df[df["ts"] >= start_date].reset_index(drop=True)

        if not df.empty:
            df.to_csv(cache_path, index=False, encoding="utf-8")
            logger.info(f"Cached {len(df):,} candles → {cache_path}")

        return df
