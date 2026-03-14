# -*- coding: utf-8 -*-
"""CLI argument parsing for the Terminal Quant TUI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class PeriodResult:
    seconds: Optional[int] = None


def parse_tui_args() -> argparse.Namespace:
    """Parse all TUI CLI arguments (tolerant of unknown args)."""
    p = argparse.ArgumentParser(
        description="Terminal Quant Suite",
        add_help=False,
    )
    p.add_argument("--mode", default=None, help="paper | por | live")
    p.add_argument("--equity", type=float, default=None, help="Initial equity (USD)")
    p.add_argument("--symbol", default=None)
    p.add_argument("--timeframe", default=None)
    p.add_argument("--bybit-env", dest="bybit_env", default=None)
    p.add_argument("--bybit-category", dest="bybit_category", default=None)
    p.add_argument("--lev-set", dest="lev_set", default=None)
    p.add_argument(
        "--lev-lookback-trades", dest="lev_lookback_trades", type=int, default=None
    )
    p.add_argument(
        "--lev-min-trades", dest="lev_min_trades", type=int, default=None
    )
    p.add_argument(
        "--lev-max-loss", dest="lev_max_loss", type=float, default=None
    )
    p.add_argument(
        "--lev-fee-buffer", dest="lev_fee_buffer", type=float, default=None
    )
    p.add_argument(
        "--lev-vol-cap", dest="lev_vol_cap", type=float, default=None
    )
    p.add_argument(
        "--enable-dynamic-leverage",
        dest="enable_dynamic_leverage",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--disable-dynamic-leverage",
        dest="disable_dynamic_leverage",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--enable-live-orders",
        dest="enable_live_orders",
        action="store_true",
        default=False,
    )

    # Period arguments
    p.add_argument("--period", default=None, help="Duration: 1h | 30m | 90s")
    p.add_argument("--seconds", type=int, default=None)
    p.add_argument("--minutes", type=int, default=None)
    p.add_argument("--hours", type=int, default=None)

    # ── Quantum autotrading arguments ──────────────────────────────────────
    p.add_argument(
        "--quantum-model",
        dest="quantum_model",
        default=None,
        help="Path to quantum checkpoint (checkpoints/quantum_v2/agent_best.pt)",
    )
    p.add_argument(
        "--q-confidence",
        dest="q_confidence",
        type=float,
        default=0.65,
        help="Minimum confidence to enter a position (default: 0.65)",
    )
    p.add_argument(
        "--q-leverage",
        dest="q_leverage",
        type=float,
        default=10.0,
        help="Leverage multiplier (default: 10x)",
    )
    p.add_argument(
        "--q-pos-frac",
        dest="q_pos_frac",
        type=float,
        default=0.5,
        help="Fraction of equity per trade (default: 0.5 = 50%)",
    )
    p.add_argument(
        "--q-tp-mult",
        dest="q_tp_mult",
        type=float,
        default=4.0,
        help="TP = q_tp_mult * ATR (default: 4.0)",
    )
    p.add_argument(
        "--q-sl-mult",
        dest="q_sl_mult",
        type=float,
        default=1.0,
        help="SL = q_sl_mult * ATR (default: 1.0)",
    )
    p.add_argument(
        "--q-live-train",
        dest="q_live_train",
        action="store_true",
        default=False,
        help="Enable online training after each completed trade",
    )

    # ── Dual model (Champion/Challenger) args ─────────────────────────────
    p.add_argument(
        "--q-swap-margin",
        dest="q_swap_margin",
        type=float,
        default=0.03,
        help="B must beat A's WR by this margin to trigger swap (default: 0.03 = 3%%)",
    )
    p.add_argument(
        "--q-min-eval-trades",
        dest="q_min_eval_trades",
        type=int,
        default=10,
        help="Min trades (both A and B) before any swap is evaluated (default: 10)",
    )
    p.add_argument(
        "--q-daily-loss-limit",
        dest="q_daily_loss_limit",
        type=float,
        default=0.05,
        help="Auto-halt if equity drops this fraction from session start (default: 0.05 = 5%%)",
    )
    p.add_argument(
        "--q-regime-exit-conf",
        dest="q_regime_exit_conf",
        type=float,
        default=0.0,
        help="Regime-flip early exit: exit open position when model signals opposite direction "
             "with this confidence. 0=disabled (default). e.g. 0.60",
    )
    p.add_argument(
        "--q-trail-after",
        dest="q_trail_after",
        type=float,
        default=0.0,
        help="Trailing stop activation in ATR units. 0=disabled (default). e.g. 1.5",
    )
    p.add_argument(
        "--q-trail-dist",
        dest="q_trail_dist",
        type=float,
        default=1.0,
        help="Trailing stop distance from watermark in ATR units (default: 1.0)",
    )

    args, _ = p.parse_known_args()
    return args


def parse_period_args(args: argparse.Namespace) -> PeriodResult:
    """Extract period (seconds) from parsed args."""
    seconds = None
    if getattr(args, "seconds", None):
        seconds = args.seconds
    if getattr(args, "minutes", None):
        seconds = args.minutes * 60
    if getattr(args, "hours", None):
        seconds = args.hours * 3600
    period_str = getattr(args, "period", None)
    if period_str:
        s = period_str.strip().lower()
        try:
            if s.endswith("h"):
                seconds = int(float(s[:-1]) * 3600)
            elif s.endswith("m"):
                seconds = int(float(s[:-1]) * 60)
            elif s.endswith("s"):
                seconds = int(float(s[:-1]))
            else:
                seconds = int(s)
        except ValueError:
            pass
    return PeriodResult(seconds=seconds)
