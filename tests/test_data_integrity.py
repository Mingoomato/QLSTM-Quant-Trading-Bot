"""Tests for DataIntegrityModule and StaleDataException."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from data_pipeline.feature_fusion import (
    DataIntegrityModule,
    StaleDataException,
)


def _make_df(
    n_rows: int = 100,
    freq: str = "1h",
    end: dt.datetime | None = None,
    onchain_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Helper: build a DataFrame with DatetimeIndex and on-chain columns."""
    if end is None:
        end = dt.datetime(2026, 3, 23, 12, 0, 0)
    if onchain_cols is None:
        onchain_cols = ["fr_z", "fr_trend", "oi_change_z", "cvd_trend_z"]

    idx = pd.date_range(end=end, periods=n_rows, freq=freq)
    data = {col: np.random.randn(n_rows) for col in onchain_cols}
    data["close"] = 85000.0 + np.cumsum(np.random.randn(n_rows) * 100)
    return pd.DataFrame(data, index=idx)


class TestStaleDataException:
    """StaleDataException carries the right metadata."""

    def test_message_contains_columns(self):
        exc = StaleDataException(["fr_z", "oi_change_z"], 4.5, 3.0)
        assert "fr_z" in str(exc)
        assert "oi_change_z" in str(exc)
        assert "4.50h" in str(exc)

    def test_attributes(self):
        exc = StaleDataException(["fr_z"], 5.0, 3.0)
        assert exc.stale_columns == ["fr_z"]
        assert exc.age_hours == 5.0
        assert exc.threshold_hours == 3.0


class TestDataIntegrityModule:
    """DataIntegrityModule.check() raises on stale data, passes on fresh."""

    def test_fresh_data_passes(self):
        """Data ending 1h ago — well within 3h threshold."""
        now = dt.datetime(2026, 3, 23, 13, 0, 0)
        df = _make_df(end=dt.datetime(2026, 3, 23, 12, 0, 0))
        dim = DataIntegrityModule(max_staleness_hours=3.0)
        # Should not raise.
        dim.check(df, now=now)

    def test_stale_data_raises(self):
        """Data ending 5h ago — exceeds 3h threshold."""
        now = dt.datetime(2026, 3, 23, 17, 0, 0)
        df = _make_df(end=dt.datetime(2026, 3, 23, 12, 0, 0))
        dim = DataIntegrityModule(max_staleness_hours=3.0)
        with pytest.raises(StaleDataException) as exc_info:
            dim.check(df, now=now)
        assert exc_info.value.age_hours == pytest.approx(5.0)
        assert exc_info.value.threshold_hours == 3.0

    def test_exact_boundary_passes(self):
        """Data exactly 3h old — at boundary, should pass (not strictly >)."""
        now = dt.datetime(2026, 3, 23, 15, 0, 0)
        df = _make_df(end=dt.datetime(2026, 3, 23, 12, 0, 0))
        dim = DataIntegrityModule(max_staleness_hours=3.0)
        dim.check(df, now=now)

    def test_just_over_boundary_raises(self):
        """Data 3h + 1s old — exceeds threshold."""
        now = dt.datetime(2026, 3, 23, 15, 0, 1)
        df = _make_df(end=dt.datetime(2026, 3, 23, 12, 0, 0))
        dim = DataIntegrityModule(max_staleness_hours=3.0)
        with pytest.raises(StaleDataException):
            dim.check(df, now=now)

    def test_custom_threshold(self):
        """1h threshold — 2h-old data should raise."""
        now = dt.datetime(2026, 3, 23, 14, 0, 0)
        df = _make_df(end=dt.datetime(2026, 3, 23, 12, 0, 0))
        dim = DataIntegrityModule(max_staleness_hours=1.0)
        with pytest.raises(StaleDataException):
            dim.check(df, now=now)

    def test_missing_columns_raises_valueerror(self):
        """No on-chain columns at all → ValueError, not StaleDataException."""
        df = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2026-01-01", periods=3, freq="1h"),
        )
        dim = DataIntegrityModule()
        with pytest.raises(ValueError, match="on-chain columns"):
            dim.check(df)

    def test_all_nan_onchain_raises_stale(self):
        """On-chain columns exist but are all NaN → stale (inf age)."""
        df = _make_df()
        for col in ["fr_z", "fr_trend", "oi_change_z", "cvd_trend_z"]:
            df[col] = np.nan
        dim = DataIntegrityModule(onchain_columns=["fr_z", "fr_trend", "oi_change_z", "cvd_trend_z"])
        with pytest.raises(StaleDataException):
            dim.check(df)

    def test_timestamp_column_fallback(self):
        """DataFrame with 'timestamp' column instead of DatetimeIndex."""
        now = dt.datetime(2026, 3, 23, 13, 0, 0)
        df = _make_df(end=dt.datetime(2026, 3, 23, 12, 0, 0))
        df = df.reset_index().rename(columns={"index": "timestamp"})
        dim = DataIntegrityModule(onchain_columns=["fr_z"])
        dim.check(df, now=now)

    def test_tz_aware_now_handled(self):
        """tz-aware 'now' should not crash — stripped internally."""
        import pytz
        now = dt.datetime(2026, 3, 23, 13, 0, 0, tzinfo=pytz.UTC)
        df = _make_df(end=dt.datetime(2026, 3, 23, 12, 0, 0))
        dim = DataIntegrityModule(onchain_columns=["fr_z"])
        dim.check(df, now=now)
