"""
train_agent.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
13-dim Structural Feature QLSTM Training Pipeline

Gate 1: Engineering sanity check (model trains, loss decreases, checkpoint saves)
Gate 2: Alpha validation (p < 0.05 bootstrapped mean R-multiples, N >= 250 trades)

Usage:
    python -m training.train_agent
    python -m training.train_agent --epochs 20 --device cuda
    python -m training.train_agent --start-date 2023-01-01 --end-date 2025-12-31
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows CP949 encoding fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.models.integrated_agent import build_quantum_agent, AgentConfig
from src.data.data_client import DataClient
from src.models.features_structural import (
    build_features_structural,
    generate_and_cache_features_structural,
    FEAT_COLUMNS,
    FEAT_DIM,
)
from src.models.labeling import compute_clean_barrier_labels, standardize_1m_ohlcv
from src.data.binance_client import fetch_binance_taker_history


# ── Constants ────────────────────────────────────────────────────────────────

BARRIER_ALPHA = 4.0   # TP = alpha * ATR
BARRIER_BETA = 1.5    # SL = beta * ATR
EVAL_CONF_THRESHOLD = 0.40
DEFAULT_CHECKPOINT_DIR = "checkpoints/structural_13dim"
DEFAULT_SEQ_LEN = 96   # 96 bars on 1h = 4 days


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_training_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    cache_dir: str = "data",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fetch OHLCV + funding + OI + CVD, label, build 13-dim features.

    Returns:
        (df_labeled, features) where df_labeled has columns:
        open, high, low, close, volume, ts, label, long_label, short_label
        and features is np.ndarray shape (N, 13).
    """
    dc = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")

    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc, hour=23, minute=59
    )
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"\n  [Data] Fetching OHLCV {start_date} ~ {end_date} ...")
    df_raw = dc.fetch_training_history(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_ms=end_ms,
        cache_dir=cache_dir,
    )
    if df_raw is None or df_raw.empty:
        raise RuntimeError(f"Empty OHLCV data for {symbol}")

    df_clean = standardize_1m_ohlcv(df_raw) if timeframe == "1m" else df_raw.copy()
    print(f"  [Data] OHLCV bars: {len(df_clean):,}")

    # Funding rate
    df_funding = dc.fetch_funding_history(symbol, start_ms, end_ms, cache_dir=cache_dir)
    if not df_funding.empty:
        df_funding["ts"] = (
            pd.to_datetime(df_funding["ts_ms"], unit="ms", utc=True)
            .dt.tz_localize(None)
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        df_clean = df_clean.merge(df_funding[["ts", "funding_rate"]], on="ts", how="left")
        df_clean["funding_rate"] = df_clean["funding_rate"].ffill().fillna(0.0)
    print(f"  [Data] Funding rate: {'merged' if not df_funding.empty else 'missing (zeros)'}")

    # Open interest
    df_oi = dc.fetch_open_interest_history(
        symbol, start_ms, end_ms, interval="1h", cache_dir=cache_dir
    )
    if not df_oi.empty:
        df_oi["ts"] = (
            pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
            .dt.tz_localize(None)
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        df_clean = df_clean.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
        df_clean["open_interest"] = df_clean["open_interest"].ffill().fillna(0.0)
    print(f"  [Data] Open interest: {'merged' if not df_oi.empty else 'missing (zeros)'}")

    # Binance taker buy volume (CVD source)
    _tc = os.path.join(
        cache_dir,
        f"binance_taker_{symbol}_{timeframe}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv",
    )
    try:
        df_taker = fetch_binance_taker_history(
            symbol=symbol,
            interval=timeframe,
            start_date=start_date,
            end_date=end_date,
            cache_path=_tc,
            verbose=False,
        )
        if not df_taker.empty:
            df_clean = df_clean.merge(
                df_taker[["ts", "taker_buy_volume"]], on="ts", how="left"
            )
            df_clean["taker_buy_volume"] = df_clean["taker_buy_volume"].fillna(0.0)
            print(f"  [Data] Taker buy volume: merged")
    except Exception:
        print(f"  [Data] Taker buy volume: unavailable (fallback proxy)")

    # Triple barrier labeling
    bars_h = 60 if timeframe == "1h" else 96
    df_lbl_raw = compute_clean_barrier_labels(
        df_clean, alpha=BARRIER_ALPHA, beta=BARRIER_BETA,
        hold_band=1.5, hold_h=20, h=bars_h,
    )
    df_labeled = df_lbl_raw[df_lbl_raw["label"] != 2].reset_index(drop=True)
    _lv = df_labeled["label"].value_counts().to_dict()
    print(
        f"  [Data] Labels: LONG={_lv.get(1, 0)} SHORT={_lv.get(-1, 0)} "
        f"HOLD={_lv.get(0, 0)} DISCARD={(df_lbl_raw['label'] == 2).sum()}"
    )

    # 13-dim structural features
    _s = start_date.replace("-", "")
    _e = end_date.replace("-", "")
    cache_file = os.path.join(cache_dir, f"feat_cache_{symbol}_{timeframe}_{_s}_{_e}_structural.npy")
    all_feat_raw = generate_and_cache_features_structural(df_clean, cache_file)

    # Filter out DISCARD rows
    keep_mask = df_lbl_raw["label"].values != 2
    all_feat = all_feat_raw[keep_mask]

    if len(all_feat) != len(df_labeled):
        raise RuntimeError(
            f"Feature/label mismatch: {len(all_feat)} vs {len(df_labeled)}"
        )

    print(f"  [Data] Features shape: {all_feat.shape} (13-dim structural)")
    return df_labeled, all_feat


# ── Training Data Preparation ────────────────────────────────────────────────

def prepare_training_data(
    df: pd.DataFrame, features: np.ndarray, seq_len: int = DEFAULT_SEQ_LEN
) -> dict | None:
    """Convert DataFrame + features into dict of arrays for training loop."""
    n = len(df)
    if n < seq_len + 35:
        return None

    prices = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = (df["high"] - df["low"]).rolling(14).mean().bfill().values
    raw_labels = df["label"].values
    long_labels = df["long_label"].values if "long_label" in df.columns else raw_labels.copy()
    short_labels = df["short_label"].values if "short_label" in df.columns else raw_labels.copy()

    return {
        "features": features,
        "prices": prices,
        "highs": highs,
        "lows": lows,
        "raw_labels": raw_labels,
        "long_labels": long_labels,
        "short_labels": short_labels,
        "atr": atrs,
        "ts": df["ts"].values if "ts" in df.columns else np.arange(n),
        "n": n,
    }


# ── Walk-Forward Folds ───────────────────────────────────────────────────────

def walk_forward_folds(df, n_folds=5, min_train_bars=1000, rolling_bars=None):
    """Generate expanding or rolling window walk-forward folds.

    Yields: (fold_k, train_df, val_df, train_abs, val_abs)
    """
    total = len(df)
    fold_size = total // (n_folds + 1)

    for k in range(1, n_folds + 1):
        val_start = k * fold_size
        val_end = (k + 1) * fold_size

        if rolling_bars is not None:
            train_end = val_start
            train_start = max(0, train_end - rolling_bars)
        else:
            train_start = 0
            train_end = val_start

        if train_end < min_train_bars:
            train_end = min(min_train_bars, total - fold_size)

        train_slice = df.iloc[train_start:train_end]
        val_slice = df.iloc[val_start:val_end]

        if len(train_slice) < min_train_bars or len(val_slice) < 50:
            continue

        yield (
            k,
            train_slice.reset_index(drop=True),
            val_slice.reset_index(drop=True),
            (train_start, train_end),
            (val_start, val_end),
        )


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(agent, val_data, seq_len=DEFAULT_SEQ_LEN, batch_size=128):
    """Run validation inference and compute win rate, EV, R-multiples.

    Returns:
        (ev_per_trade, win_rate, diag_dict, r_multiples_list)
    """
    agent.eval()
    n = val_data["n"]
    indices = np.arange(seq_len + 120, n - 35)

    if len(indices) == 0:
        return 0.0, 0.0, {}, []

    total_trades = 0
    tp_hits = 0
    total_pnl_pct = 0.0
    n_long = 0
    n_short = 0
    r_multiples = []  # R-multiples for Gate 2 bootstrap test

    eta = agent.config.eta_base
    fee_pct = eta + eta  # round-trip

    with torch.no_grad():
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]

            x_list, atr_list, price_list = [], [], []
            long_l_list, short_l_list = [], []

            for idx in batch_indices:
                x_list.append(val_data["features"][idx - seq_len + 1 : idx + 1])
                atr_list.append(val_data["atr"][idx])
                price_list.append(val_data["prices"][idx])
                long_l_list.append(val_data["long_labels"][idx])
                short_l_list.append(val_data["short_labels"][idx])

            x_val = torch.from_numpy(np.array(x_list)).float().to(agent.device)

            logits, _, _, _ = agent.forward(x_val, last_step_only=True)
            if logits.dim() == 3:
                logits = logits.squeeze(1)

            probs = torch.softmax(logits, dim=-1)
            max_probs, actions = probs.max(dim=-1)
            actions = actions.cpu().numpy()
            max_probs = max_probs.cpu().numpy()

            conf_mask = max_probs >= EVAL_CONF_THRESHOLD
            final_actions = np.where(conf_mask, actions, 0)

            for i, act in enumerate(final_actions):
                if act == 0:
                    continue

                total_trades += 1
                _atr = float(atr_list[i])
                _price = max(float(price_list[i]), 1e-8)
                _tp = BARRIER_ALPHA * _atr / _price
                _sl = BARRIER_BETA * _atr / _price

                if act == 1:  # Long
                    n_long += 1
                    rl = long_l_list[i]
                    if rl == 1:
                        tp_hits += 1
                        pnl = _tp - fee_pct
                        r_mult = _tp / _sl  # R-multiple
                    elif rl == -1:
                        pnl = -_sl - fee_pct
                        r_mult = -1.0
                    else:
                        pnl = -fee_pct
                        r_mult = -fee_pct / _sl if _sl > 0 else 0.0
                elif act == 2:  # Short
                    n_short += 1
                    rl = short_l_list[i]
                    if rl == -1:
                        tp_hits += 1
                        pnl = _tp - fee_pct
                        r_mult = _tp / _sl
                    elif rl == 1:
                        pnl = -_sl - fee_pct
                        r_mult = -1.0
                    else:
                        pnl = -fee_pct
                        r_mult = -fee_pct / _sl if _sl > 0 else 0.0
                else:
                    continue

                total_pnl_pct += pnl
                r_multiples.append(r_mult)

    if total_trades == 0:
        return 0.0, 0.0, {"long": 0, "short": 0, "total": 0}, []

    leverage = agent.config.leverage
    win_rate = tp_hits / total_trades
    ev_per_trade = (total_pnl_pct * leverage) / total_trades

    diag = {
        "long": n_long,
        "short": n_short,
        "total": total_trades,
        "tp_hits": tp_hits,
        "win_rate": round(win_rate * 100, 1),
        "ev_per_trade": round(ev_per_trade, 4),
        "net_pnl_pct": round(total_pnl_pct * leverage * 100, 2),
    }
    return ev_per_trade, win_rate, diag, r_multiples


# ── Gate 2: Bootstrap Alpha Test ─────────────────────────────────────────────

def bootstrap_alpha_test(
    r_multiples: list[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap test: H0: mean(R-multiples) <= 0 vs H1: mean(R-multiples) > 0.

    Returns dict with p_value, mean_r, ci_lower, ci_upper, n_trades, gate_pass.
    """
    arr = np.array(r_multiples, dtype=np.float64)
    n = len(arr)
    if n < 10:
        return {
            "p_value": 1.0,
            "mean_r": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_trades": n,
            "gate_pass": False,
        }

    rng = np.random.RandomState(seed)
    boot_means = np.array(
        [arr[rng.randint(0, n, size=n)].mean() for _ in range(n_bootstrap)]
    )

    observed_mean = arr.mean()
    p_value = float((boot_means <= 0).sum() / n_bootstrap)
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "p_value": round(p_value, 4),
        "mean_r": round(float(observed_mean), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n_trades": n,
        "gate_pass": p_value < alpha and n >= 250,
    }


# ── Core Training Cycle ─────────────────────────────────────────────────────

def run_training_cycle(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    n_folds: int = 5,
    epochs: int = 10,
    batch_size: int = 128,
    seq_len: int = DEFAULT_SEQ_LEN,
    device: str = "cpu",
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    leverage: float = 10.0,
    seed: int | None = None,
    time_budget_s: float = 540.0,
    qfi_update_freq: int = 200,
) -> str:
    """Train the 13-dim QuantumFinancialAgent with walk-forward cross-validation.

    Returns:
        Path to the best checkpoint file (.pth).
    """
    print("=" * 65)
    print("  13-dim Structural QLSTM Training Pipeline")
    print("=" * 65)
    print(f"  Symbol     : {symbol}")
    print(f"  Timeframe  : {timeframe}")
    print(f"  Period     : {start_date} ~ {end_date}")
    print(f"  Folds      : {n_folds}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"  Seq len    : {seq_len}")
    print(f"  Device     : {device}")
    print(f"  Feature dim: {FEAT_DIM} (structural)")
    print(f"  QFI freq   : {qfi_update_freq} steps")
    print(f"  Time budget: {time_budget_s:.0f}s")
    print("=" * 65)

    t_pipeline_start = time.time()

    # Seed
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"  [Seed] Fixed: {seed}")

    # Load data
    t_data = time.time()
    df_labeled, all_features = load_training_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    print(f"  [Timer] Data loading: {time.time() - t_data:.1f}s")

    # Build agent with 13-dim feature input
    t_agent = time.time()
    config = AgentConfig(
        feature_dim=FEAT_DIM,
        leverage=leverage,
        checkpoint_dir=checkpoint_dir,
        confidence_threshold=EVAL_CONF_THRESHOLD,
        entropy_reg=0.05,
        qfi_update_freq=qfi_update_freq,
    )
    dev = torch.device(device)
    agent = build_quantum_agent(config=config, device=dev)
    agent.print_architecture()
    print(f"  [Timer] Agent build: {time.time() - t_agent:.1f}s")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Walk-forward folds
    fold_gen = walk_forward_folds(df_labeled, n_folds=n_folds)
    best_global_ev = -float("inf")
    best_checkpoint_path = ""
    all_r_multiples = []  # Accumulate across all folds for Gate 2

    bars_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
    H = 60 if timeframe == "1h" else 96  # horizon for TP/SL simulation

    _budget_exceeded = False
    for k, train_df, val_df, train_abs, val_abs in fold_gen:
        elapsed_so_far = time.time() - t_pipeline_start
        if elapsed_so_far > time_budget_s:
            print(f"\n  [Budget] {elapsed_so_far:.0f}s elapsed > {time_budget_s:.0f}s budget — stopping early")
            _budget_exceeded = True
            break
        print(f"\n{'='*65}")
        print(f"  [Fold {k}/{n_folds}] Train: {len(train_df):,} bars | Val: {len(val_df):,} bars")
        print(f"{'='*65}")

        train_features = all_features[train_abs[0] : train_abs[1]]
        val_features = all_features[val_abs[0] : val_abs[1]]

        train_data = prepare_training_data(train_df, train_features, seq_len)
        val_data = prepare_training_data(val_df, val_features, seq_len)
        if not train_data or not val_data:
            print(f"  [Fold {k}] Skipping — insufficient data")
            continue

        # LDA fit on training labels
        decomposer = agent.encoder.decomposer
        if decomposer.use_lda:
            decomposer.fit_lda(train_features, train_df["label"].values)

        # Reset optimizer state per fold
        for pg in agent.optimizer.param_groups:
            pg["lr"] = agent.config.lr
        from src.models.qng_optimizer import DiagonalQNGOptimizer as _DQNG
        _inner = (
            agent.optimizer.classical_optimizer
            if isinstance(agent.optimizer, _DQNG)
            else agent.optimizer
        )
        _inner.state.clear()
        agent.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            _inner, T_0=100, T_mult=2, eta_min=1e-5
        )

        best_fold_ev = -float("inf")

        for epoch in range(1, epochs + 1):
            elapsed_so_far = time.time() - t_pipeline_start
            if elapsed_so_far > time_budget_s:
                print(f"  [Budget] {elapsed_so_far:.0f}s elapsed — stopping fold {k} early")
                _budget_exceeded = True
                break
            epoch_start = time.time()
            agent.train()
            n_samples = train_data["n"]

            # Balanced sampling: LONG:SHORT = 1:1
            valid_min = seq_len + 120
            valid_max = n_samples - (H + 1)
            if valid_max <= valid_min:
                print(f"  [Fold {k} Ep {epoch}] Skip — valid range empty")
                continue
            all_valid = np.arange(valid_min, valid_max)
            raw_lbl = train_data["raw_labels"][all_valid]
            long_pool = all_valid[raw_lbl == 1]
            short_pool = all_valid[raw_lbl == -1]
            hold_pool = all_valid[raw_lbl == 0]
            n_per_dir = min(len(long_pool), len(short_pool))
            if n_per_dir == 0:
                print(f"  [Fold {k} Ep {epoch}] Skip — no directional samples")
                continue
            sampled_long = np.random.choice(long_pool, n_per_dir, replace=False)
            sampled_short = np.random.choice(short_pool, n_per_dir, replace=False)
            n_hold = min(len(hold_pool), max(1, n_per_dir // 4))
            sampled_hold = (
                np.random.choice(hold_pool, n_hold, replace=n_hold > len(hold_pool))
                if len(hold_pool) > 0
                else np.array([], dtype=int)
            )
            indices = np.random.permutation(
                np.concatenate([sampled_long, sampled_short, sampled_hold])
            )

            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, len(indices), batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]

                x_list, prices_list, atr_list, entry_list = [], [], [], []

                for idx in batch_idx:
                    x_list.append(train_data["features"][idx - seq_len + 1 : idx + 1])
                    win_prices = train_data["prices"][idx : idx + H + 1]
                    prices_list.append(win_prices)
                    atr_list.append(train_data["atr"][idx])
                    entry_list.append(0)

                x_train = torch.from_numpy(np.array(x_list)).float()

                # Use pre-computed labels directly (no redundant forward pass)
                # raw_labels: 1=LONG opp, -1=SHORT opp, 0=HOLD
                # long_labels/short_labels: 1=TP hit, -1=SL hit, 0=timeout
                dirs_list, labels_list = [], []
                for i, idx in enumerate(batch_idx):
                    rl = int(train_data["raw_labels"][idx])
                    if rl == 1:  # LONG opportunity
                        dirs_list.append(1)
                        ll = int(train_data["long_labels"][idx])
                        if ll == 1:
                            labels_list.append(1)   # LONG TP
                        elif ll == -1:
                            labels_list.append(3)   # SL hit
                        else:
                            labels_list.append(0)   # timeout
                    elif rl == -1:  # SHORT opportunity
                        dirs_list.append(-1)
                        sl = int(train_data["short_labels"][idx])
                        if sl == -1:
                            labels_list.append(2)   # SHORT TP
                        elif sl == 1:
                            labels_list.append(3)   # SL hit
                        else:
                            labels_list.append(0)   # timeout
                    else:  # HOLD
                        dirs_list.append(0)
                        labels_list.append(0)

                # Pad short price sequences
                max_len = max(len(p) for p in prices_list)
                padded_prices = []
                for p in prices_list:
                    if len(p) < max_len:
                        p = np.pad(p, (0, max_len - len(p)), constant_values=p[-1])
                    padded_prices.append(p)

                prices_t = torch.from_numpy(np.array(padded_prices)).float()
                dirs_t = torch.tensor(dirs_list, dtype=torch.float32)
                entry_t = torch.tensor(entry_list[: len(batch_idx)], dtype=torch.long)
                labels_t = torch.tensor(labels_list, dtype=torch.long)
                atr_t = torch.from_numpy(np.array(atr_list[: len(batch_idx)])).float()

                result = agent.train_step(
                    x=x_train,
                    prices=prices_t,
                    directions=dirs_t,
                    entry_idx=entry_t,
                    labels=labels_t,
                    atr=atr_t,
                    last_step_only=True,
                )
                epoch_loss += result.loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation
            ev, wr, diag, fold_r_mults = evaluate_model(agent, val_data, seq_len, batch_size)
            elapsed = time.time() - epoch_start

            print(
                f"  [F{k} E{epoch:02d}] Loss={avg_loss:.4f} | "
                f"WR={diag.get('win_rate', 0):.1f}% EV={ev:.4f} "
                f"L={diag.get('long', 0)} S={diag.get('short', 0)} "
                f"N={diag.get('total', 0)} | {elapsed:.1f}s"
            )

            # Save best per fold
            if ev > best_fold_ev and diag.get("total", 0) >= 10:
                best_fold_ev = ev
                fold_ckpt = os.path.join(checkpoint_dir, f"agent_best_fold{k}.pth")
                agent.save_checkpoint(fold_ckpt)
                print(f"  [F{k}] New fold-best: EV={ev:.4f} → {fold_ckpt}")

                # Accumulate R-multiples for Gate 2
                all_r_multiples.extend(fold_r_mults)

            # Save global best
            if ev > best_global_ev and diag.get("total", 0) >= 10:
                best_global_ev = ev
                best_checkpoint_path = os.path.join(checkpoint_dir, "agent_best.pth")
                agent.save_checkpoint(best_checkpoint_path)
                print(f"  [Global] New best: EV={ev:.4f} → {best_checkpoint_path}")

        if _budget_exceeded:
            break

        # Load fold-best before next fold
        fold_ckpt = os.path.join(checkpoint_dir, f"agent_best_fold{k}.pth")
        if os.path.isfile(fold_ckpt):
            agent.load_checkpoint(fold_ckpt)

    # ── Gate 1: Engineering Sanity Check ──────────────────────────────────────
    print(f"\n{'='*65}")
    print("  GATE 1: Engineering Sanity Check")
    print(f"{'='*65}")

    gate1_pass = True
    if not best_checkpoint_path or not os.path.isfile(best_checkpoint_path):
        print("  FAIL: No checkpoint saved")
        gate1_pass = False
    else:
        ckpt = torch.load(best_checkpoint_path, map_location=dev, weights_only=False)
        print(f"  Checkpoint: {best_checkpoint_path}")
        print(f"  Global step: {ckpt.get('global_step', 0)}")
        print(f"  Feature dim: {ckpt['config']['feature_dim']}")
        print(f"  PASS: Model trained and checkpoint saved")

    # ── Gate 2: Alpha Validation ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  GATE 2: Alpha Validation (Bootstrap)")
    print(f"{'='*65}")

    gate2 = bootstrap_alpha_test(all_r_multiples)
    print(f"  N trades     : {gate2['n_trades']}")
    print(f"  Mean R-mult  : {gate2['mean_r']}")
    print(f"  95% CI       : [{gate2['ci_lower']}, {gate2['ci_upper']}]")
    print(f"  p-value      : {gate2['p_value']}")
    print(f"  Gate pass    : {'PASS' if gate2['gate_pass'] else 'FAIL'}")
    if gate2['n_trades'] < 250:
        print(f"  Note: {gate2['n_trades']} trades < 250 minimum")

    t_total = time.time() - t_pipeline_start
    print(f"\n{'='*65}")
    print(f"  Training complete. Best checkpoint: {best_checkpoint_path}")
    print(f"  Gate 1: {'PASS' if gate1_pass else 'FAIL'}")
    print(f"  Gate 2: {'PASS' if gate2['gate_pass'] else 'FAIL'}")
    print(f"  Total elapsed: {t_total:.1f}s ({t_total/60:.1f}m)")
    print(f"{'='*65}")

    return best_checkpoint_path


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="13-dim Structural QLSTM Training")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--time-budget", type=float, default=540.0,
                        help="Max training wall-clock seconds (default 540, leaves 60s buffer)")
    parser.add_argument("--qfi-freq", type=int, default=200,
                        help="QFI recomputation interval (default 200; was 20, 10x cheaper)")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick sanity run: 6-month subset, 2 folds, 3 epochs",
    )
    args = parser.parse_args()

    # --quick overrides for fast validation
    if args.quick:
        args.start_date = "2025-07-01"
        args.end_date = "2025-12-31"
        args.n_folds = 2
        args.epochs = 3
        print("  [Quick mode] 6-month subset, 2 folds, 3 epochs")

    run_training_cycle(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        leverage=args.leverage,
        seed=args.seed,
        time_budget_s=args.time_budget,
        qfi_update_freq=args.qfi_freq,
    )


if __name__ == "__main__":
    main()
