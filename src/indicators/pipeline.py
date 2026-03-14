"""Compute all indicators and attach to DataFrame."""

from typing import Any, Dict

import pandas as pd

from .basic import (
    atr,
    bollinger,
    ema,
    kdj,
    ma,
    macd,
    mavol,
    parabolic_sar,
    rsi,
    stoch_rsi,
    williams_r,
    laguerre_rsi,
)


def compute_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    ind = cfg.get("indicators", {})
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Moving Averages
    for length in ind.get("ma_lengths", [7, 25, 99]):
        df[f"ma{length}"] = ma(close, length)

    # EMA
    for length in ind.get("ema_lengths", [9, 21]):
        df[f"ema{length}"] = ema(close, length)

    # Bollinger Bands
    boll_len = ind.get("boll_length", 20)
    boll_std = ind.get("boll_std", 2.0)
    up, mid, lo = bollinger(close, boll_len, boll_std)
    df["boll_up"] = up
    df["boll_mid"] = mid
    df["boll_low"] = lo

    # MACD
    dif, dea, hist = macd(
        close,
        ind.get("macd_fast", 12),
        ind.get("macd_slow", 26),
        ind.get("macd_signal", 9),
    )
    df["macd_dif"] = dif
    df["macd_dea"] = dea
    df["macd_hist"] = hist

    # RSI
    df["rsi"] = rsi(close, ind.get("rsi_length", 14))

    # Williams %R
    df["wr"] = williams_r(high, low, close, ind.get("wr_length", 14))

    # Stochastic RSI
    srsi_k, srsi_d = stoch_rsi(close, ind.get("stoch_rsi_length", 14))
    df["stoch_rsi_k"] = srsi_k
    df["stoch_rsi_d"] = srsi_d

    # KDJ
    k, d, j = kdj(high, low, close, ind.get("kdj_length", 9))
    df["kdj_k"] = k
    df["kdj_d"] = d
    df["kdj_j"] = j

    # Parabolic SAR
    df["sar"] = parabolic_sar(
        high, low,
        ind.get("sar_af", 0.02),
        ind.get("sar_max_af", 0.2),
    )

    # ATR
    atr_len = ind.get("atr_length", 14)
    df["atr"] = atr(high, low, close, atr_len)
    df["atr_pct"] = df["atr"] / close

    # Volume MA
    df["mavol"] = mavol(volume, 20)

    # Laguerre RSI (0..1)
    df["lrsi"] = laguerre_rsi(close, ind.get("lrsi_gamma", 0.5))

    return df
