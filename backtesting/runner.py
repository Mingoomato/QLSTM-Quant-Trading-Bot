"""
backtesting/runner.py
─────────────────────────────────────────────────────────────────────
Walk-forward validation runner for QLSTM 13-dim structural model.

Executes `run_backtest` from backtesting.backtest_structural sequentially
across Viktor's decorrelated hold-out periods (CTO report 2026-03-23).

Output: Aggregated R-multiple logs for all periods, to be consumed by
        Gate 2 bootstrap hypothesis test (separate module).

Walk-Forward Folds — Viktor's Decorrelated Hold-Out Periods:
  Fold 1: Train 2023-01-01 ~ 2025-12-31, OOS 2026-01-01 ~ 2026-03-31
  Fold 2: Train 2023-04-01 ~ 2026-03-31, OOS 2026-04-01 ~ 2026-06-30
  Fold 3: Train 2023-07-01 ~ 2026-06-30, OOS 2026-07-01 ~ 2026-09-30
  Fold 4: Train 2023-10-01 ~ 2026-09-30, OOS 2026-10-01 ~ 2026-12-31
  Fold 5: Train 2024-01-01 ~ 2026-12-31, OOS 2027-01-01 ~ 2027-03-31

Each OOS window is 3 months. Adjacent OOS windows are non-overlapping.
Training windows expand forward (anchored start shifts by 3 months).
─────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest_structural import run_backtest


# ── Viktor's decorrelated hold-out periods ────────────────────────────────
@dataclass
class WalkForwardFold:
    """A single walk-forward fold with train and OOS date boundaries."""
    fold_id: int
    train_start: str   # YYYY-MM-DD
    train_end: str
    oos_start: str     # out-of-sample start
    oos_end: str       # out-of-sample end


VIKTOR_HOLDOUT_PERIODS = [
    WalkForwardFold(1, "2023-01-01", "2025-12-31", "2026-01-01", "2026-03-31"),
    WalkForwardFold(2, "2023-04-01", "2026-03-31", "2026-04-01", "2026-06-30"),
    WalkForwardFold(3, "2023-07-01", "2026-06-30", "2026-07-01", "2026-09-30"),
    WalkForwardFold(4, "2023-10-01", "2026-09-30", "2026-10-01", "2026-12-31"),
    WalkForwardFold(5, "2024-01-01", "2026-12-31", "2027-01-01", "2027-03-31"),
]


@dataclass
class FoldResult:
    """Per-fold backtest output."""
    fold_id: int
    oos_start: str
    oos_end: str
    n_trades: int
    n_wins: int
    win_rate: float
    roi: float
    max_dd: float
    r_multiples: np.ndarray
    mean_r: float
    df_trades: pd.DataFrame


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation output."""
    fold_results: list = field(default_factory=list)
    all_r_multiples: np.ndarray = field(default_factory=lambda: np.array([]))
    total_trades: int = 0
    total_wins: int = 0
    aggregate_wr: float = 0.0
    aggregate_mean_r: float = 0.0
    aggregate_median_r: float = 0.0


def run_walkforward_validation(
    model_path: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    capital: float = 10.0,
    tp_mult: float = 3.0,
    sl_mult: float = 1.0,
    leverage: float = 5.0,
    pos_frac: float = 0.5,
    max_hold: int = 96,
    seq_len: int = 96,
    export_csv: bool = True,
) -> WalkForwardResult:
    """Execute QLSTM 13-dim backtest across Viktor's decorrelated hold-out periods.

    Runs `run_backtest` sequentially on each of the 5 non-overlapping OOS
    windows. The same model checkpoint is tested across all periods to
    measure true out-of-sample generalization.

    Args:
        model_path: Path to trained QLSTM .pt checkpoint.
        symbol: Trading pair.
        timeframe: Candle timeframe.
        capital: Starting capital per fold (reset each fold).
        tp_mult: Take-profit ATR multiplier.
        sl_mult: Stop-loss ATR multiplier.
        leverage: Leverage.
        pos_frac: Position fraction.
        max_hold: Max hold bars before force-exit.
        seq_len: QLSTM sequence length.
        export_csv: If True, export aggregated R-multiples to CSV.

    Returns:
        WalkForwardResult with aggregated R-multiple logs for all periods.
    """
    folds = VIKTOR_HOLDOUT_PERIODS
    t0 = time.time()
    fold_results: list[FoldResult] = []
    all_r_multiples: list[np.ndarray] = []

    print("=" * 68)
    print("  WALK-FORWARD VALIDATION — QLSTM 13-DIM STRUCTURAL")
    print(f"  Folds: {len(folds)}  Symbol: {symbol}  Timeframe: {timeframe}")
    print(f"  Model: {model_path}")
    print(f"  TP/SL: {tp_mult}/{sl_mult}  Leverage: {leverage}x")
    print("=" * 68)

    for fold in folds:
        print(f"\n{'─' * 68}")
        print(f"  FOLD {fold.fold_id}: OOS {fold.oos_start} → {fold.oos_end}")
        print(f"{'─' * 68}")

        result = run_backtest(
            model_path=model_path,
            symbol=symbol,
            timeframe=timeframe,
            start_date=fold.oos_start,
            end_date=fold.oos_end,
            capital=capital,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            leverage=leverage,
            pos_frac=pos_frac,
            max_hold=max_hold,
            seq_len=seq_len,
            output_dir="reports",
        )

        if result is None or result.get("error"):
            err = result.get("error", "unknown") if result else "no result"
            print(f"  [FOLD {fold.fold_id}] SKIPPED — {err}")
            continue

        df_trades = result.get("trades_df", pd.DataFrame())
        metrics = result.get("metrics", {})
        r_mults = np.asarray(result.get("r_multiples", []), dtype=np.float64)

        n_trades = metrics.get("n_trades", len(df_trades))
        n_wins = metrics.get("n_win", 0)
        wr = metrics.get("win_rate", 0.0)
        roi = metrics.get("roi_pct", 0.0)
        max_dd = metrics.get("max_drawdown_pct", 0.0)
        mean_r = float(np.mean(r_mults)) if len(r_mults) > 0 else 0.0

        fold_res = FoldResult(
            fold_id=fold.fold_id,
            oos_start=fold.oos_start,
            oos_end=fold.oos_end,
            n_trades=n_trades,
            n_wins=n_wins,
            win_rate=wr,
            roi=roi,
            max_dd=max_dd,
            r_multiples=r_mults,
            mean_r=mean_r,
            df_trades=df_trades,
        )
        fold_results.append(fold_res)
        all_r_multiples.append(r_mults)

        print(f"  [FOLD {fold.fold_id}] trades={n_trades}  WR={wr:.1f}%  "
              f"ROI={roi:+.1f}%  MDD={max_dd:.1f}%  mean_R={mean_r:+.3f}")

    # ── Aggregate ────────────────────────────────────────────────────────
    if all_r_multiples:
        combined_r = np.concatenate(all_r_multiples)
    else:
        combined_r = np.array([], dtype=np.float64)

    total_trades = sum(f.n_trades for f in fold_results)
    total_wins = sum(f.n_wins for f in fold_results)
    agg_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    agg_mean_r = float(np.mean(combined_r)) if len(combined_r) > 0 else 0.0
    agg_median_r = float(np.median(combined_r)) if len(combined_r) > 0 else 0.0

    elapsed = time.time() - t0

    print(f"\n{'=' * 68}")
    print("  WALK-FORWARD AGGREGATED RESULTS")
    print(f"{'=' * 68}")
    print(f"  Folds completed   : {len(fold_results)} / {len(folds)}")
    print(f"  Total OOS trades  : {total_trades}")
    print(f"  Total wins        : {total_wins}")
    print(f"  Aggregate WR      : {agg_wr:.1f}%")
    print(f"  Mean R-multiple   : {agg_mean_r:+.4f}")
    print(f"  Median R-multiple : {agg_median_r:+.4f}")
    if len(combined_r) > 0:
        print(f"  Std R-multiple    : {float(np.std(combined_r)):.4f}")
        print(f"  Min / Max R       : {float(np.min(combined_r)):+.3f} / "
              f"{float(np.max(combined_r)):+.3f}")
    print(f"  Elapsed           : {elapsed:.1f}s")
    print(f"{'=' * 68}")

    # ── Export CSV ─────────────────────────────────────────────────────
    if export_csv and len(combined_r) > 0:
        os.makedirs("reports", exist_ok=True)

        rows = []
        for fr in fold_results:
            for idx, r in enumerate(fr.r_multiples):
                rows.append({
                    "fold": fr.fold_id,
                    "oos_start": fr.oos_start,
                    "oos_end": fr.oos_end,
                    "trade_idx": idx,
                    "r_multiple": float(r),
                })
        df_r = pd.DataFrame(rows)
        csv_path = "reports/walkforward_r_multiples.csv"
        df_r.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\n[export] R-multiples → {csv_path}  ({len(df_r)} trades)")

    wf_result = WalkForwardResult(
        fold_results=fold_results,
        all_r_multiples=combined_r,
        total_trades=total_trades,
        total_wins=total_wins,
        aggregate_wr=agg_wr,
        aggregate_mean_r=agg_mean_r,
        aggregate_median_r=agg_median_r,
    )

    return wf_result


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Walk-Forward Validation Runner (QLSTM 13-dim). "
                    "Runs all 5 of Viktor's decorrelated hold-out periods."
    )
    p.add_argument("--model-path", required=True,
                   help="Path to trained QLSTM .pt checkpoint")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--capital", type=float, default=10.0)
    p.add_argument("--tp-mult", type=float, default=3.0)
    p.add_argument("--sl-mult", type=float, default=1.0)
    p.add_argument("--leverage", type=float, default=5.0)
    p.add_argument("--pos-frac", type=float, default=0.5)
    p.add_argument("--max-hold", type=int, default=96)
    p.add_argument("--seq-len", type=int, default=96)

    args = p.parse_args()

    result = run_walkforward_validation(
        model_path=args.model_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        capital=args.capital,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        leverage=args.leverage,
        pos_frac=args.pos_frac,
        max_hold=args.max_hold,
        seq_len=args.seq_len,
    )
