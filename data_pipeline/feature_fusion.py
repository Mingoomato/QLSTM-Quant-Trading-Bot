"""Data integrity checks for the structural feature pipeline.

Ensures on-chain data (funding rate, OI, liquidations, CVD) is fresh
before allowing trading decisions. Stale data → StaleDataException → halt.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd


class StaleDataException(Exception):
    """Raised when on-chain data is older than the allowed staleness window."""

    def __init__(self, stale_columns: list[str], age_hours: float, threshold_hours: float):
        self.stale_columns = stale_columns
        self.age_hours = age_hours
        self.threshold_hours = threshold_hours
        cols = ", ".join(stale_columns)
        super().__init__(
            f"Stale on-chain data detected — columns [{cols}] "
            f"are {age_hours:.2f}h old (threshold: {threshold_hours}h). "
            f"Trading halted."
        )


# On-chain feature groups that require freshness checks.
ONCHAIN_GROUPS: dict[str, list[str]] = {
    "funding_rate": ["fr_z", "fr_trend"],
    "open_interest": ["oi_change_z", "oi_price_div"],
    "liquidations": ["liq_long_z", "liq_short_z"],
    "cvd": ["cvd_trend_z", "cvd_price_div", "taker_ratio_z"],
}


@dataclass
class DataIntegrityModule:
    """Validates freshness of on-chain data before trading.

    Parameters
    ----------
    max_staleness_hours : float
        Maximum allowed age of the most recent on-chain data row.
        Default is 3.0 hours.
    onchain_columns : list[str] | None
        Columns to monitor. If None, uses all columns from ONCHAIN_GROUPS.
    """

    max_staleness_hours: float = 3.0
    onchain_columns: list[str] | None = None
    _resolved_columns: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.onchain_columns is not None:
            self._resolved_columns = list(self.onchain_columns)
        else:
            self._resolved_columns = [
                col for cols in ONCHAIN_GROUPS.values() for col in cols
            ]

    def check(
        self,
        df: pd.DataFrame,
        now: dt.datetime | None = None,
    ) -> None:
        """Validate that on-chain data in *df* is fresh.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a DatetimeIndex (or a ``timestamp`` column).
        now : datetime, optional
            Reference "current" time. Defaults to ``datetime.utcnow()``.

        Raises
        ------
        StaleDataException
            If the newest on-chain row is older than ``max_staleness_hours``.
        ValueError
            If required on-chain columns are missing from the DataFrame.
        """
        if now is None:
            now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

        # Ensure tz-naive comparison (CLAUDE.md §8 best practice).
        if hasattr(now, "tzinfo") and now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        # Resolve index.
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index
        elif "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"])
        else:
            raise ValueError(
                "DataFrame must have a DatetimeIndex or a 'timestamp' column."
            )

        # Strip timezone from timestamps for comparison.
        if hasattr(timestamps, "tz") and timestamps.tz is not None:
            timestamps = timestamps.tz_localize(None)

        # Check which on-chain columns are present.
        present = [c for c in self._resolved_columns if c in df.columns]
        if not present:
            raise ValueError(
                f"None of the required on-chain columns found in DataFrame. "
                f"Expected at least one of: {self._resolved_columns}"
            )

        # Find the most recent non-NaN row across on-chain columns.
        onchain_subset = df[present]
        valid_mask = onchain_subset.notna().any(axis=1)
        if not valid_mask.any():
            raise StaleDataException(
                stale_columns=present,
                age_hours=float("inf"),
                threshold_hours=self.max_staleness_hours,
            )

        valid_timestamps = timestamps[valid_mask]
        last_valid_idx = valid_timestamps[-1] if isinstance(valid_timestamps, pd.DatetimeIndex) else valid_timestamps.iloc[-1]
        if isinstance(last_valid_idx, pd.Timestamp):
            last_valid_idx = last_valid_idx.to_pydatetime().replace(tzinfo=None)

        age = now - last_valid_idx
        age_hours = age.total_seconds() / 3600.0

        if age_hours > self.max_staleness_hours:
            # Identify which columns are stale at the last valid row.
            last_row = onchain_subset.loc[valid_mask].iloc[-1]
            stale_cols = [c for c in present if pd.notna(last_row[c])]
            if not stale_cols:
                stale_cols = present
            raise StaleDataException(
                stale_columns=stale_cols,
                age_hours=age_hours,
                threshold_hours=self.max_staleness_hours,
            )
