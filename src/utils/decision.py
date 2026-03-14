"""Decision and trade suggestion utilities for ultra-short horizon engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import math
import numpy as np
import pandas as pd


@dataclass
class CombinedConfidence:
    confi_long: float
    confi_short: float
    p_combined: float
    position: str
    allow_open: bool


def combined_confidence(
    quant_long: int,
    quant_short: int,
    quantum_long: int,
    quantum_short: int,
    total_models: int,
    p_win_by_group: Iterable[Tuple[str, float]],
    confi_open_threshold: float,
    p_combined_gate: float = 0.45,
) -> CombinedConfidence:
    if total_models <= 0:
        return CombinedConfidence(0.0, 0.0, 0.0, "FLAT", False)

    score_long = quant_long * 4 + quantum_long * 6
    score_short = quant_short * 4 + quantum_short * 6
    denom = total_models * 10
    confi_long = (score_long / denom) * 100.0
    confi_short = (score_short / denom) * 100.0

    position = "FLAT"
    if confi_long >= confi_open_threshold or confi_short >= confi_open_threshold:
        if confi_long > confi_short:
            position = "LONG"
        elif confi_short > confi_long:
            position = "SHORT"

    # p_combined: weighted avg of p_win_2m by group weights
    weighted = []
    for grp, p in p_win_by_group:
        w = 6.0 if grp == "quantum" else 4.0
        weighted.append((w, p))
    if not weighted:
        p_combined = 0.01
    else:
        p_combined = sum(w * p for w, p in weighted) / sum(w for w, _ in weighted)

    allow_open = p_combined >= p_combined_gate
    return CombinedConfidence(confi_long, confi_short, p_combined, position, allow_open)


def wilder_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    if df is None or len(df) < period + 1:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.iloc[:period].mean()
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr.iloc[i]) / period
    if atr != atr:
        return None
    return float(atr)


def prob_from_score(score: float, vol_1m: float, horizon: int = 2) -> float:
    if vol_1m <= 0:
        return 0.5
    z = score / (vol_1m * math.sqrt(horizon))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def select_leverage_kelly(
    entry: float,
    stop_dist: float,
    p_win: float,
    rr: float,
    lev_set: Iterable[int],
    max_loss_pct: float,
    kelly_scale: float,
) -> int:
    if entry <= 0 or stop_dist <= 0:
        return 1
    loss_frac = stop_dist / entry
    if loss_frac <= 0:
        return 1
    max_allowed = max_loss_pct / loss_frac
    b = rr
    f_star = max(0.0, (b * p_win - (1.0 - p_win)) / b) if b > 0 else 0.0
    lev_kelly = (f_star * kelly_scale) / loss_frac if loss_frac > 0 else 0.0
    cap = min(max_allowed, lev_kelly) if lev_kelly > 0 else max_allowed
    allowed = [l for l in lev_set if l <= cap]
    return max(allowed) if allowed else 1


def warmup_allow_open(is_first: bool, allow_open: bool) -> bool:
    return False if is_first else allow_open


@dataclass
class TradeSuggestion:
    model_name: str
    position: str
    entry: float | None
    sp: float | None
    tl: float | None
    lev: int | None
    holding_period: int
    enter_ts: str
    hold_until_ts: str
    loss_escape: float | None
    adv_escape: float | None
    confidence: float

    def to_pipe_row(self) -> str:
        pos = self.position
        if pos == "LONG":
            pos = "[green]LONG[/green]"
        elif pos == "SHORT":
            pos = "[red]SHORT[/red]"
        elif pos == "FLAT":
            pos = "[yellow]FLAT[/yellow]"
        if self.position == "FLAT" or self.entry is None:
            return (
                f"| {self.model_name} | {pos} | - | - | - | - | "
                f"- | - | - | - | {self.confidence:.1f}% |"
            )
        return (
            f"| {self.model_name} | {pos} | {self.entry:,.2f} | {self.sp:,.2f} | {self.tl:,.2f} | "
            f"{self.lev} | {self.enter_ts} | {self.hold_until_ts} | {self.loss_escape:,.2f} | {self.adv_escape:,.2f} | "
            f"{self.confidence:.1f}% |"
        )


def build_trade_suggestion(
    model_name: str,
    position: str,
    entry: float,
    atr_1m: float | None,
    p_win: float,
    rr: float,
    atr_k: float,
    min_stop_abs: float,
    min_stop_pct: float,
    escape_loss_mult: float,
    escape_adv_mult: float,
    lev_set: Iterable[int],
    max_loss_pct: float,
    kelly_scale: float,
    holding_period: int = 120,
    enter_ts: str = "-",
    hold_until_ts: str = "-",
) -> TradeSuggestion:
    if position == "FLAT" or entry <= 0:
        return TradeSuggestion(
            model_name,
            "FLAT",
            None,
            None,
            None,
            None,
            holding_period,
            enter_ts,
            hold_until_ts,
            None,
            None,
            100.0 * p_win,
        )

    atr_val = atr_1m if atr_1m and atr_1m > 0 else entry * 0.001
    stop_dist = max(atr_k * atr_val, min_stop_abs, min_stop_pct * entry)

    if position == "LONG":
        sp = entry - stop_dist
        tl = entry + rr * stop_dist
        loss_escape = entry - escape_loss_mult * atr_val
        adv_escape = entry + escape_adv_mult * atr_val
    else:
        sp = entry + stop_dist
        tl = entry - rr * stop_dist
        loss_escape = entry + escape_loss_mult * atr_val
        adv_escape = entry - escape_adv_mult * atr_val

    lev = 10

    return TradeSuggestion(
        model_name=model_name,
        position=position,
        entry=entry,
        sp=sp,
        tl=tl,
        lev=lev,
        holding_period=holding_period,
        enter_ts=enter_ts,
        hold_until_ts=hold_until_ts,
        loss_escape=loss_escape,
        adv_escape=adv_escape,
        confidence=100.0 * p_win,
    )


def truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def format_cell(value, kind: str) -> str:
    if kind in ("price", "min"):
        if value is None:
            return "-"
        return f"{float(value):,.2f}"
    if kind == "int":
        if value is None:
            return "-"
        return f"{int(value)}"
    if kind == "pct":
        if value is None:
            return "-"
        return f"{float(value):.1f}%"
    if kind == "text":
        return "-" if value is None else str(value)
    return "-" if value is None else str(value)


def render_trade_suggestions_table(suggestions: list[TradeSuggestion]) -> list[tuple[str, tuple[str, ...]]]:
    if not suggestions:
        return []
    combined = [s for s in suggestions if s.model_name == "COMBINED"]
    others = [s for s in suggestions if s.model_name != "COMBINED"]
    def _sort_key(s: TradeSuggestion):
        flat = 1 if s.position == "FLAT" else 0
        return (flat, -s.confidence)

    others_sorted = sorted(others, key=_sort_key)
    ordered = others_sorted + combined

    rows: list[tuple[str, tuple[str, ...]]] = []
    for s in ordered:
        model = truncate(s.model_name, 22)
        if s.position == "LONG":
            pos = "[green]LONG[/green]"
        elif s.position == "SHORT":
            pos = "[red]SHORT[/red]"
        else:
            pos = "[yellow]FLAT[/yellow]"
        rows.append(
            (
                s.model_name,
                (
                model,
                pos,
                format_cell(s.entry, "price"),
                format_cell(s.sp, "price"),
                format_cell(s.tl, "price"),
                format_cell(s.lev, "int"),
                format_cell(s.enter_ts, "text"),
                format_cell(s.hold_until_ts, "text"),
                format_cell(s.loss_escape, "min"),
                format_cell(s.adv_escape, "min"),
                format_cell(s.confidence, "pct"),
                ),
            )
        )
    return rows
