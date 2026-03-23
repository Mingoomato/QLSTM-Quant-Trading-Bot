"""
profile_bottleneck.py — Identify where the 600s training timeout is spent.
Measures: data load, agent build, single forward pass, single train_step, VQC cost.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch

TIMERS = {}

def tic(label):
    TIMERS[label] = time.perf_counter()

def toc(label):
    elapsed = time.perf_counter() - TIMERS[label]
    print(f"  [{label}] {elapsed:.2f}s")
    return elapsed


def main():
    print("=" * 60)
    print("  PROFILING: Training Pipeline Bottleneck Analysis")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── Phase 1: Data Loading ────────────────────────────────────────
    print("\n--- Phase 1: Data Loading ---")
    tic("data_total")

    tic("imports")
    from src.data.data_client import DataClient
    from src.models.features_structural import (
        build_features_structural,
        generate_and_cache_features_structural,
        FEAT_COLUMNS,
        FEAT_DIM,
    )
    from src.models.labeling import compute_clean_barrier_labels, standardize_1m_ohlcv
    from src.data.binance_client import fetch_binance_taker_history
    toc("imports")

    # Use 6-month subset for profiling
    symbol = "BTCUSDT"
    timeframe = "1h"
    start_date = "2025-07-01"
    end_date = "2025-12-31"

    import pandas as pd
    from datetime import datetime, timezone

    dc = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc, hour=23, minute=59
    )
    end_ms = int(end_dt.timestamp() * 1000)

    tic("ohlcv")
    df_raw = dc.fetch_training_history(
        symbol=symbol, timeframe=timeframe,
        start_date=start_date, end_ms=end_ms, cache_dir="data",
    )
    t_ohlcv = toc("ohlcv")

    tic("funding")
    df_funding = dc.fetch_funding_history(symbol, start_ms, end_ms, cache_dir="data")
    t_funding = toc("funding")

    tic("oi")
    df_oi = dc.fetch_open_interest_history(symbol, start_ms, end_ms, interval="1h", cache_dir="data")
    t_oi = toc("oi")

    tic("taker")
    _tc = f"data/binance_taker_{symbol}_{timeframe}_profile.csv"
    try:
        df_taker = fetch_binance_taker_history(
            symbol=symbol, interval=timeframe,
            start_date=start_date, end_date=end_date,
            cache_path=_tc, verbose=False,
        )
    except Exception as e:
        print(f"  [taker] Failed: {e}")
        df_taker = pd.DataFrame()
    t_taker = toc("taker")

    t_data_total = toc("data_total")
    print(f"\n  >>> TOTAL DATA: {t_data_total:.1f}s (OHLCV={t_ohlcv:.1f} Fund={t_funding:.1f} OI={t_oi:.1f} Taker={t_taker:.1f})")
    print(f"  >>> OHLCV bars: {len(df_raw) if df_raw is not None else 0}")

    if df_raw is None or df_raw.empty:
        print("  ABORT: No data fetched")
        return

    # Merge and label
    df_clean = df_raw.copy()
    if not df_funding.empty:
        df_funding["ts"] = (
            pd.to_datetime(df_funding["ts_ms"], unit="ms", utc=True)
            .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        df_clean = df_clean.merge(df_funding[["ts", "funding_rate"]], on="ts", how="left")
        df_clean["funding_rate"] = df_clean["funding_rate"].ffill().fillna(0.0)
    if not df_oi.empty:
        df_oi["ts"] = (
            pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
            .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        df_clean = df_clean.merge(df_oi[["ts", "open_interest"]], on="ts", how="left")
        df_clean["open_interest"] = df_clean["open_interest"].ffill().fillna(0.0)
    if not df_taker.empty:
        df_clean = df_clean.merge(df_taker[["ts", "taker_buy_volume"]], on="ts", how="left")
        df_clean["taker_buy_volume"] = df_clean["taker_buy_volume"].fillna(0.0)

    tic("labeling")
    df_lbl = compute_clean_barrier_labels(df_clean, alpha=4.0, beta=1.5, hold_band=1.5, hold_h=20, h=60)
    toc("labeling")

    tic("features")
    cache_file = "data/feat_cache_profile_structural.npy"
    features = generate_and_cache_features_structural(df_clean, cache_file)
    toc("features")

    # ── Phase 2: Agent Build ─────────────────────────────────────────
    print("\n--- Phase 2: Agent Build ---")
    tic("agent_build")
    from src.models.integrated_agent import build_quantum_agent, AgentConfig
    config = AgentConfig(
        feature_dim=FEAT_DIM,
        leverage=10.0,
        checkpoint_dir="checkpoints/profile_test",
        confidence_threshold=0.40,
        entropy_reg=0.05,
    )
    dev = torch.device(device)
    agent = build_quantum_agent(config=config, device=dev)
    toc("agent_build")

    # ── Phase 3: Forward Pass Timing ─────────────────────────────────
    print("\n--- Phase 3: Forward/Train Step Timing ---")
    seq_len = 96
    batch_size = 128

    # Create dummy batch
    n_valid = min(len(features), len(df_lbl))
    if n_valid < seq_len + 200:
        print("  Not enough data for timing test")
        return

    # Build one batch
    idx_start = seq_len + 120
    batch_indices = np.arange(idx_start, min(idx_start + batch_size, n_valid - 61))
    x_list = [features[i - seq_len + 1 : i + 1] for i in batch_indices]
    x_batch = torch.from_numpy(np.array(x_list)).float().to(dev)
    print(f"  Batch shape: {x_batch.shape}")

    # Forward only
    agent.eval()
    with torch.no_grad():
        tic("forward_cold")
        logits, expvals, J, c_kt = agent.forward(x_batch, last_step_only=True)
        toc("forward_cold")

        # Warm runs
        times_fwd = []
        for i in range(5):
            t0 = time.perf_counter()
            logits, expvals, J, c_kt = agent.forward(x_batch, last_step_only=True)
            times_fwd.append(time.perf_counter() - t0)
        avg_fwd = np.mean(times_fwd)
        print(f"  [forward_warm] avg={avg_fwd:.4f}s over 5 runs")

    # Train step
    agent.train()
    H = 60
    prices_list = []
    for idx in batch_indices:
        p = df_lbl["close"].values[idx : idx + H + 1]
        if len(p) < H + 1:
            p = np.pad(p, (0, H + 1 - len(p)), constant_values=p[-1])
        prices_list.append(p)
    prices_t = torch.from_numpy(np.array(prices_list)).float()
    dirs_t = torch.ones(len(batch_indices), dtype=torch.float32)
    entry_t = torch.zeros(len(batch_indices), dtype=torch.long)
    labels_t = torch.ones(len(batch_indices), dtype=torch.long)
    atr_t = torch.ones(len(batch_indices), dtype=torch.float32) * 100.0

    tic("train_step_cold")
    result = agent.train_step(
        x=x_batch.clone(), prices=prices_t, directions=dirs_t,
        entry_idx=entry_t, labels=labels_t, atr=atr_t, last_step_only=True,
    )
    toc("train_step_cold")

    times_train = []
    for i in range(5):
        t0 = time.perf_counter()
        result = agent.train_step(
            x=x_batch.clone(), prices=prices_t, directions=dirs_t,
            entry_idx=entry_t, labels=labels_t, atr=atr_t, last_step_only=True,
        )
        times_train.append(time.perf_counter() - t0)
    avg_train = np.mean(times_train)
    print(f"  [train_step_warm] avg={avg_train:.4f}s over 5 runs")

    # ── Phase 4: Estimate Total Training Time ────────────────────────
    print("\n--- Phase 4: Full Training Time Estimate ---")
    # For full 3-year dataset (2023-2025)
    n_bars_full = 26000  # ~3 years of 1h
    n_folds = 5
    epochs = 10
    # Assume ~4000 samples per epoch (balanced long/short)
    n_samples_per_epoch = 4000
    batches_per_epoch = n_samples_per_epoch // batch_size + 1
    # Eval batches
    n_eval_samples = 2000
    eval_batches = n_eval_samples // batch_size + 1

    total_train_steps = n_folds * epochs * batches_per_epoch
    total_eval_steps = n_folds * epochs * eval_batches
    est_train_time = total_train_steps * avg_train
    est_eval_time = total_eval_steps * avg_fwd
    est_data_time = t_data_total * 2.5  # 3yr ≈ 2.5× of 6mo

    print(f"  Batches/epoch:     {batches_per_epoch}")
    print(f"  Total train steps: {total_train_steps}")
    print(f"  Total eval steps:  {total_eval_steps}")
    print(f"  Est data load:     {est_data_time:.0f}s")
    print(f"  Est training:      {est_train_time:.0f}s")
    print(f"  Est evaluation:    {est_eval_time:.0f}s")
    print(f"  Est TOTAL:         {est_data_time + est_train_time + est_eval_time:.0f}s")
    print(f"  Budget:            600s")

    if est_data_time + est_train_time + est_eval_time > 600:
        print("\n  >>> WILL TIMEOUT <<<")
        print("  Recommendations:")
        if est_data_time > 120:
            print(f"    1. Data fetch is {est_data_time:.0f}s — ensure CSV cache hits")
        if avg_train > 0.5:
            print(f"    2. train_step is {avg_train:.3f}s/batch — VQC is bottleneck")
            print(f"       Consider: reduce n_vqc_layers, or use --quick mode")
        if est_train_time > 300:
            print(f"    3. Training loop is {est_train_time:.0f}s — reduce epochs/folds")
            print(f"       Quick fix: --epochs 3 --n-folds 2")

    print("\n" + "=" * 60)
    print("  PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
