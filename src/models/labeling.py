# -*- coding: utf-8 -*-
"""Triple-Barrier Labeling for 1m/10x BTC Futures.

Implements first-touch triple-barrier labeling with:
- T+1 open-entry latency fix
- Fee-aware TP/SL thresholds (net-of-roundtrip cost)
- ATR-based dynamic barriers
- Same-bar tie → SL (conservative)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Round-trip cost: 0.055% maker/taker × 2 sides
C_ROUNDTRIP = 0.0011


def standardize_1m_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, forward-fill missing 1m bars, align to minute boundaries, dedup.

    Parameters
    ----------
    df : DataFrame with at least 'ts', 'open', 'high', 'low', 'close', 'volume'

    Returns
    -------
    DataFrame reindexed to continuous 1m bars
    """
    df = df.copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
        df = df.set_index("ts")

    # Align to minute boundaries
    df.index = df.index.floor("min")
    df = df[~df.index.duplicated(keep="last")]

    # Reindex to continuous 1m
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="min")
    df = df.reindex(full_idx)

    # Forward-fill OHLC, zero-fill volume
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].ffill()
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)

    df.index.name = "ts"
    return df.reset_index()


def _compute_atr(df: pd.DataFrame, period: int = 15) -> pd.Series:
    """Compute Average True Range over `period` bars."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().values
    return pd.Series(atr, index=df.index)


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    alpha: float = 2.0,
    beta: float = 1.0,
    h: int = 15,
    c: float = C_ROUNDTRIP,
    min_rr_ratio: float = 1.5,
) -> pd.DataFrame:
    """Compute triple-barrier labels for each bar.

    Parameters
    ----------
    df          : OHLCV DataFrame (must have 'open', 'high', 'low', 'close')
    alpha       : TP multiplier applied to ATR.  Default 2.0 (was 1.5).
    beta        : SL multiplier applied to ATR.  Keep short (default 1.0).
    h           : maximum hold horizon in bars
    c           : round-trip fee as fraction of entry price
    min_rr_ratio: Minimum required TP/SL ratio (alpha/beta).
                  If alpha < beta * min_rr_ratio, alpha is auto-raised so
                  that the labelled dataset always encodes a favourable
                  risk-reward structure (no reverse RR bias). Default 1.5.

    Returns
    -------
    DataFrame with added columns: label, entry_price, tp_price, sl_price
        label: +1 (TP hit), -1 (SL hit), 0 (timeout/skip)
    """
    # ── RR ratio guard: TP must be at least min_rr_ratio × SL ──────────────
    # This prevents the reverse risk-reward bug (Avg Win < Avg Loss) by
    # ensuring that the labels themselves encode a favourable structure.
    if alpha < beta * min_rr_ratio:
        alpha_old = alpha
        alpha = beta * min_rr_ratio
        import warnings
        warnings.warn(
            f"[triple_barrier] alpha={alpha_old:.2f} < beta*min_rr_ratio="
            f"{beta*min_rr_ratio:.2f}; auto-raised to alpha={alpha:.2f}",
            RuntimeWarning, stacklevel=2,
        )
    n = len(df)
    atr = _compute_atr(df, period=15).values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    labels = np.zeros(n, dtype=int)
    entry_prices = np.full(n, np.nan)
    tp_prices = np.full(n, np.nan)
    sl_prices = np.full(n, np.nan)

    for i in range(n):
        # Need at least i+1 bar for T+1 entry
        if i + 1 >= n:
            labels[i] = 0
            continue

        entry = opens[i + 1]  # T+1 latency fix (F3)
        if np.isnan(entry) or entry <= 0 or np.isnan(atr[i]) or atr[i] <= 0:
            labels[i] = 0
            continue

        # Fee-aware barriers (F2)
        fee_offset = entry * c
        tp = entry + alpha * atr[i] + fee_offset
        sl = entry - beta * atr[i] - fee_offset

        entry_prices[i] = entry
        tp_prices[i] = tp
        sl_prices[i] = sl

        # Scan forward from i+1 through i+h (inclusive)
        hit = 0  # 0=timeout
        end = min(i + 1 + h, n)
        for j in range(i + 1, end):
            hit_tp = highs[j] >= tp
            hit_sl = lows[j] <= sl

            if hit_tp and hit_sl:
                # Same-bar tie → SL (conservative)
                hit = -1
                break
            elif hit_tp:
                hit = 1
                break
            elif hit_sl:
                hit = -1
                break

        labels[i] = hit

    result = df.copy()
    result["label"] = labels
    result["entry_price"] = entry_prices
    result["tp_price"] = tp_prices
    result["sl_price"] = sl_prices
    return result


def compute_bidirectional_barrier_labels(
    df: pd.DataFrame,
    alpha: float = 6.0,
    beta: float = 1.5,
    h: int = 96,
    c: float = C_ROUNDTRIP,
) -> pd.DataFrame:
    """Bidirectional triple-barrier labeling with symmetric R:R for LONG and SHORT.

    LONG  barriers: upper = +alpha×ATR (TP), lower = -beta×ATR  (SL)
    SHORT barriers: upper = +beta×ATR  (SL), lower = -alpha×ATR (TP)

    Both directions share the same R:R = alpha/beta (e.g. 4:1 when alpha=6, beta=1.5).

    Parameters
    ----------
    alpha : TP multiplier for both LONG and SHORT (default 6.0)
    beta  : SL multiplier for both LONG and SHORT (default 1.5)

    Returns
    -------
    DataFrame with columns:
        label       : +1 = LONG signal (price up alpha×ATR first)
                      -1 = SHORT signal (price down alpha×ATR first)
                       0 = HOLD (no alpha×ATR directional move within h bars)
        long_label  : outcome from LONG barrier scan (+1 TP, -1 SL, 0 timeout)
        short_label : outcome from SHORT barrier scan (-1 TP, +1 SL, 0 timeout)
        entry_price, tp_price, sl_price (from LONG barrier for display)
    """
    # LONG scan: upper = +alpha×ATR, lower = -beta×ATR
    long_df = compute_triple_barrier_labels(
        df, alpha=alpha, beta=beta, h=h, c=c, min_rr_ratio=0.0
    )

    # SHORT scan: upper = +beta×ATR (SHORT SL), lower = -alpha×ATR (SHORT TP)
    # Achieved by passing alpha_for_tp=beta, beta_for_sl=alpha.
    # min_rr_ratio=0.0 disables the guard that would otherwise auto-raise alpha.
    short_df = compute_triple_barrier_labels(
        df, alpha=beta, beta=alpha, h=h, c=c, min_rr_ratio=0.0
    )

    long_labels  = long_df["label"].values   # +1=LONG TP, -1=LONG SL,  0=timeout
    short_labels = short_df["label"].values  # -1=SHORT TP, +1=SHORT SL, 0=timeout

    # Combined direction signal
    final_labels = np.zeros(len(df), dtype=int)
    final_labels[long_labels == 1]   =  1   # LONG TP: strong up move (alpha×ATR)
    final_labels[short_labels == -1] = -1   # SHORT TP: strong down move (alpha×ATR)

    result = df.copy()
    result["label"]       = final_labels
    result["long_label"]  = long_labels
    result["short_label"] = short_labels
    result["entry_price"] = long_df["entry_price"]
    result["tp_price"]    = long_df["tp_price"]
    result["sl_price"]    = long_df["sl_price"]
    return result


def compute_clean_barrier_labels(
    df: pd.DataFrame,
    alpha: float = 4.0,
    beta: float = 1.5,
    hold_band: float = 1.5,
    hold_h: int = 20,
    h: int = 96,
    c: float = C_ROUNDTRIP,
) -> pd.DataFrame:
    """Clean triple-barrier labeling with explicit HOLD and ambiguous discard.

    Labels:
      +1  = LONG    : LONG TP (+alpha×ATR) hit before LONG SL (-beta×ATR)
      -1  = SHORT   : SHORT TP (-alpha×ATR) hit before SHORT SL (+beta×ATR)
       0  = HOLD    : price stays within ±hold_band×ATR for first hold_h bars
       2  = DISCARD : ambiguous zone → unprofitable, excluded from training

    Parameters
    ----------
    alpha     : TP multiplier (default 4.0×ATR). R:R = alpha/beta
    beta      : SL multiplier (default 1.5×ATR). BEP = beta/(alpha+beta) = 27.3%
    hold_band : HOLD band half-width in ATR units (default 1.5)
                Empirically: ±1.5×ATR covers bottom 13.7% of 20-bar windows
    hold_h    : bars to check for HOLD condition (default 20 = 5 hours on 15m)
    h         : max horizon for LONG/SHORT scan (default 96 = 24 hours)

    Rationale:
      HOLD = genuine short-term consolidation (first hold_h bars quiet)
      DISCARD = moved 1.5~4×ATR but no clear direction → unprofitable to trade
    """
    n = len(df)
    atr = _compute_atr(df, period=15).values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values

    labels = np.full(n, 2, dtype=int)  # default = DISCARD

    for i in range(n):
        if i + 1 >= n:
            labels[i] = 2
            continue

        entry = opens[i + 1]
        if np.isnan(entry) or entry <= 0 or np.isnan(atr[i]) or atr[i] <= 0:
            labels[i] = 2
            continue

        fee_offset = entry * c
        # LONG barriers
        long_tp = entry + alpha * atr[i] + fee_offset
        long_sl = entry - beta  * atr[i] - fee_offset
        # SHORT barriers
        short_tp = entry - alpha * atr[i] - fee_offset
        short_sl = entry + beta  * atr[i] + fee_offset
        # HOLD band (no fee offset — just price range check)
        hold_up = entry + hold_band * atr[i]
        hold_dn = entry - hold_band * atr[i]

        end = min(i + 1 + h, n)
        hold_end = min(i + 1 + hold_h, n)

        # ── LONG first-touch scan ──────────────────────────────────────────
        long_result = 0   # 0=timeout, 1=TP, -1=SL
        for j in range(i + 1, end):
            hit_tp = highs[j] >= long_tp
            hit_sl = lows[j]  <= long_sl
            if hit_tp and hit_sl:
                long_result = -1; break          # tie → SL (conservative)
            elif hit_tp:
                long_result = 1;  break
            elif hit_sl:
                long_result = -1; break

        # ── SHORT first-touch scan ─────────────────────────────────────────
        short_result = 0
        for j in range(i + 1, end):
            hit_tp = lows[j]  <= short_tp
            hit_sl = highs[j] >= short_sl
            if hit_tp and hit_sl:
                short_result = -1; break
            elif hit_tp:
                short_result = 1;  break
            elif hit_sl:
                short_result = -1; break

        # ── HOLD check: first hold_h bars within ±hold_band×ATR ───────────
        hold_ok = True
        for j in range(i + 1, hold_end):
            if highs[j] > hold_up or lows[j] < hold_dn:
                hold_ok = False
                break

        # ── Assign label (LONG/SHORT take priority over HOLD) ─────────────
        if long_result == 1 and short_result != 1:
            labels[i] = 1             # clean LONG TP
        elif short_result == 1 and long_result != 1:
            labels[i] = -1            # clean SHORT TP
        elif hold_ok:
            labels[i] = 0             # genuine consolidation (first 5h quiet)
        else:
            labels[i] = 2             # ambiguous → DISCARD

    result = df.copy()
    result["label"] = labels
    return result
