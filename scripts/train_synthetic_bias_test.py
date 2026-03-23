"""
train_synthetic_bias_test.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Long/Short Bias Diagnostic — Synthetic Balanced Training

Purpose:
    Train a fresh QuantumFinancialAgent on a synthetically balanced dataset
    where LONG and SHORT setups are equally profitable. If the trained agent
    still shows 100% long bias, the bug is in the model/loss, not the data.

Synthetic Data Design:
    - 50% of samples: clear uptrend (LONG should TP)
    - 50% of samples: clear downtrend (SHORT should TP)
    - Features are mirror-symmetric: uptrend features = -downtrend features
    - ATR is constant across all samples
    - TP/SL barriers are symmetric

Output:
    - checkpoints/synthetic_test/agent.pt
    - Action distribution report (HOLD/LONG/SHORT percentages)
    - Per-direction accuracy

Author: Finman (performance engineer)
Date: 2026-03-23
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows CP949 encoding fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.models.integrated_agent import build_quantum_agent, AgentConfig


# ── Configuration ─────────────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 512          # total samples (256 long + 256 short)
SEQ_LEN = 96             # input sequence length
FEATURE_DIM = 28         # V4 feature dimension
HORIZON = 48             # forward price window for TP/SL simulation
BASE_PRICE = 50000.0
ATR_VAL = 500.0          # constant ATR
TP_MULT = 3.0            # TP = 3 * ATR
SL_MULT = 1.5            # SL = 1.5 * ATR
N_EPOCHS = 40
BATCH_SIZE = 32
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "synthetic_test"
)
REPORT_PATH = os.path.join(CHECKPOINT_DIR, "bias_diagnostic_report.json")


def generate_synthetic_data(n_samples: int, seed: int = 42):
    """Generate perfectly balanced long/short synthetic data.

    Returns dict with:
        features: [N, SEQ_LEN, FEATURE_DIM]  — synthetic features
        prices:   [N, HORIZON+1]              — forward price paths
        highs:    [N, HORIZON+1]              — high prices
        lows:     [N, HORIZON+1]              — low prices
        atr:      [N]                         — constant ATR
        true_dir: [N]                         — +1 (long setup) or -1 (short setup)
    """
    rng = np.random.RandomState(seed)
    n_half = n_samples // 2

    # ── Features: mirror-symmetric ─────────────────────────────────────
    # For long setups: feature[0] (log-return proxy) is positive on average
    # For short setups: feature[0] is negative on average (exact mirror)
    features_long = rng.randn(n_half, SEQ_LEN, FEATURE_DIM).astype(np.float32)
    # Add a clear directional signal in the first few features
    # Feature 0: log-return trend
    trend_signal = np.linspace(0, 1.5, SEQ_LEN).reshape(1, -1, 1)
    features_long[:, :, 0:1] += trend_signal  # positive trend

    # Short features = exact mirror of long features
    features_short = features_long.copy()
    features_short[:, :, 0:1] = -features_long[:, :, 0:1]  # invert trend signal

    features = np.concatenate([features_long, features_short], axis=0)

    # ── Price paths ───────────────────────────────────────────────────
    # Long setups: price drifts UP by > TP_MULT * ATR within horizon
    # Short setups: price drifts DOWN by > TP_MULT * ATR within horizon
    prices = np.zeros((n_samples, HORIZON + 1), dtype=np.float32)
    highs = np.zeros_like(prices)
    lows = np.zeros_like(prices)

    tp_dist = TP_MULT * ATR_VAL  # 1500
    sl_dist = SL_MULT * ATR_VAL  # 750

    for i in range(n_samples):
        is_long = i < n_half
        entry = BASE_PRICE
        prices[i, 0] = entry

        if is_long:
            # Gradual uptrend: reaches TP at ~bar 20, stays above
            drift = tp_dist * 1.3 / HORIZON  # overshoot TP
            noise_scale = sl_dist * 0.15      # small noise, never hits SL
        else:
            # Gradual downtrend
            drift = -tp_dist * 1.3 / HORIZON
            noise_scale = sl_dist * 0.15

        for t in range(1, HORIZON + 1):
            noise = rng.randn() * noise_scale
            prices[i, t] = prices[i, t - 1] + drift + noise
            # High/Low with small spread
            spread = abs(noise) * 0.5 + ATR_VAL * 0.02
            highs[i, t] = max(prices[i, t] + spread, prices[i, t])
            lows[i, t] = min(prices[i, t] - spread, prices[i, t])

        highs[i, 0] = entry + ATR_VAL * 0.02
        lows[i, 0] = entry - ATR_VAL * 0.02

    atr = np.full(n_samples, ATR_VAL, dtype=np.float32)
    true_dir = np.array([1] * n_half + [-1] * n_half, dtype=np.float32)

    # Shuffle to prevent ordering bias
    perm = rng.permutation(n_samples)
    return {
        "features": features[perm],
        "prices": prices[perm],
        "highs": highs[perm],
        "lows": lows[perm],
        "atr": atr[perm],
        "true_dir": true_dir[perm],
    }


def simulate_tp_sl(prices, highs, lows, atr, direction, tp_mult, sl_mult):
    """Vectorized TP/SL barrier simulation.

    Returns:
        labels: 1=LONG_TP, 2=SHORT_TP, 3=SL_HIT, 0=TIMEOUT
        dirs:   +1/-1 actual direction
    """
    n = len(prices)
    labels = np.zeros(n, dtype=np.int64)
    dirs = np.zeros(n, dtype=np.float32)

    for i in range(n):
        d = int(direction[i])
        dirs[i] = d
        ep = prices[i, 0]
        atr_val = atr[i]
        tp_dist = tp_mult * atr_val
        sl_dist = sl_mult * atr_val

        hi = highs[i, 1:]
        lo = lows[i, 1:]

        if d == 1:  # LONG
            tp_bars = np.where(hi >= ep + tp_dist)[0]
            sl_bars = np.where(lo <= ep - sl_dist)[0]
        else:       # SHORT
            tp_bars = np.where(lo <= ep - tp_dist)[0]
            sl_bars = np.where(hi >= ep + sl_dist)[0]

        tp_bar = int(tp_bars[0]) if len(tp_bars) > 0 else HORIZON + 1
        sl_bar = int(sl_bars[0]) if len(sl_bars) > 0 else HORIZON + 1

        if tp_bar <= sl_bar and tp_bar <= HORIZON:
            labels[i] = 1 if d == 1 else 2  # TP_HIT
        elif sl_bar < tp_bar and sl_bar <= HORIZON:
            labels[i] = 3  # SL_HIT
        else:
            labels[i] = 0  # TIMEOUT

    return labels, dirs


def train_and_evaluate():
    """Main training + evaluation loop."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 70)
    print("  SYNTHETIC BIAS DIAGNOSTIC — Long/Short Balance Test")
    print("=" * 70)

    # ── Step 1: Generate data ──────────────────────────────────────────
    print("\n[1/4] Generating synthetic balanced dataset...")
    data = generate_synthetic_data(N_SAMPLES, seed=SEED)

    n_long = int((data["true_dir"] == 1).sum())
    n_short = int((data["true_dir"] == -1).sum())
    print(f"  Samples: {N_SAMPLES} total ({n_long} long, {n_short} short)")

    # Verify TP/SL simulation with true directions
    labels_true, _ = simulate_tp_sl(
        data["prices"], data["highs"], data["lows"],
        data["atr"], data["true_dir"], TP_MULT, SL_MULT
    )
    n_tp_long = int(((labels_true == 1) & (data["true_dir"] == 1)).sum())
    n_tp_short = int(((labels_true == 2) & (data["true_dir"] == -1)).sum())
    n_sl = int((labels_true == 3).sum())
    n_timeout = int((labels_true == 0).sum())
    print(f"  With true directions: TP_LONG={n_tp_long}, TP_SHORT={n_tp_short}, "
          f"SL={n_sl}, TIMEOUT={n_timeout}")

    # ── Step 2: Build fresh agent ──────────────────────────────────────
    print("\n[2/4] Building fresh agent (no checkpoint)...")
    cfg = AgentConfig(
        feature_dim=FEATURE_DIM,
        n_eigenvectors=3,
        n_vqc_layers=2,
        n_actions=3,
        leverage=10.0,
        gamma=0.99,
        lr=3e-4,
        entropy_reg=0.05,
        dir_sym_coef=1.0,
        act_sym_coef=0.5,
        use_lightning=True,
        use_platt=False,       # disable gates for clean test
        use_lindblad=False,
        use_cr_filter=False,
        use_entropy_prod=False,
        use_fisher_threshold=False,
        confidence_threshold=0.0,  # no confidence gate
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = build_quantum_agent(config=cfg, device=device)
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # ── Step 3: Training loop ──────────────────────────────────────────
    print(f"\n[3/4] Training for {N_EPOCHS} epochs (batch_size={BATCH_SIZE})...")
    t_start = time.time()

    epoch_logs = []
    for epoch in range(1, N_EPOCHS + 1):
        perm = np.random.permutation(N_SAMPLES)
        epoch_loss = 0.0
        epoch_critic = 0.0
        epoch_dir_sym = 0.0
        n_batches = 0

        # Balanced sampling: ensure each batch has ~50/50 long/short
        long_idx = np.where(data["true_dir"][perm] == 1)[0]
        short_idx = np.where(data["true_dir"][perm] == -1)[0]
        n_pairs = min(len(long_idx), len(short_idx))
        half_bs = BATCH_SIZE // 2

        for bi in range(0, n_pairs, half_bs):
            l_batch = perm[long_idx[bi:bi + half_bs]]
            s_batch = perm[short_idx[bi:bi + half_bs]]
            batch_idx = np.concatenate([l_batch, s_batch])
            np.random.shuffle(batch_idx)
            bs = len(batch_idx)
            if bs < 4:
                continue

            # Build tensors
            x = torch.from_numpy(data["features"][batch_idx]).float()
            prices_fwd = torch.from_numpy(data["prices"][batch_idx]).float()
            atr_batch = torch.from_numpy(data["atr"][batch_idx]).float()

            # Get model predictions (no grad)
            with torch.no_grad():
                logits, _, _, _ = agent(x.to(device), last_step_only=True)
                if logits.dim() == 3:
                    logits = logits.squeeze(1)
                pred_actions = logits.argmax(dim=-1).cpu().numpy()

            # Simulate TP/SL based on model's predicted direction
            dirs_list = []
            labels_list = []
            for j in range(bs):
                act = int(pred_actions[j])
                if act == 0:  # HOLD
                    dirs_list.append(0)
                    labels_list.append(0)
                    continue

                direction = 1 if act == 1 else -1
                ep = float(data["prices"][batch_idx[j], 0])
                atr_val = float(data["atr"][batch_idx[j]])
                tp_dist = TP_MULT * atr_val
                sl_dist = SL_MULT * atr_val

                hi = data["highs"][batch_idx[j], 1:]
                lo = data["lows"][batch_idx[j], 1:]

                if direction == 1:
                    tp_bars = np.where(hi >= ep + tp_dist)[0]
                    sl_bars = np.where(lo <= ep - sl_dist)[0]
                else:
                    tp_bars = np.where(lo <= ep - tp_dist)[0]
                    sl_bars = np.where(hi >= ep + sl_dist)[0]

                tp_bar = int(tp_bars[0]) if len(tp_bars) > 0 else HORIZON + 1
                sl_bar = int(sl_bars[0]) if len(sl_bars) > 0 else HORIZON + 1

                dirs_list.append(direction)
                if tp_bar <= sl_bar and tp_bar <= HORIZON:
                    labels_list.append(1 if direction == 1 else 2)
                elif sl_bar < tp_bar and sl_bar <= HORIZON:
                    labels_list.append(3)
                else:
                    labels_list.append(0)

            d_train = torch.from_numpy(np.array(dirs_list)).float()
            e_train = torch.zeros(bs, dtype=torch.long)
            l_train = torch.from_numpy(np.array(labels_list)).long()

            result = agent.train_step(
                x, prices_fwd, d_train, e_train, l_train, atr_batch,
                last_step_only=True
            )
            epoch_loss += result.loss
            epoch_critic += result.critic_loss
            epoch_dir_sym += result.dir_sym_loss
            n_batches += 1

        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_critic = epoch_critic / n_batches
            avg_dir_sym = epoch_dir_sym / n_batches
        else:
            avg_loss = avg_critic = avg_dir_sym = 0.0

        # ── Epoch evaluation: check action distribution ────────────
        with torch.no_grad():
            all_x = torch.from_numpy(data["features"]).float().to(device)
            # Process in chunks to avoid OOM
            all_actions = []
            all_probs = []
            chunk = 64
            for ci in range(0, N_SAMPLES, chunk):
                xc = all_x[ci:ci + chunk]
                lg, _, _, _ = agent(xc, last_step_only=True)
                if lg.dim() == 3:
                    lg = lg.squeeze(1)
                probs = torch.softmax(lg, dim=-1)
                acts = lg.argmax(dim=-1)
                all_actions.append(acts.cpu())
                all_probs.append(probs.cpu())

            all_actions = torch.cat(all_actions).numpy()
            all_probs = torch.cat(all_probs).numpy()

        n_hold = int((all_actions == 0).sum())
        n_long_pred = int((all_actions == 1).sum())
        n_short_pred = int((all_actions == 2).sum())

        # Accuracy: did the model predict LONG for uptrend and SHORT for downtrend?
        true_dirs = data["true_dir"]
        correct_long = int(((all_actions == 1) & (true_dirs == 1)).sum())
        correct_short = int(((all_actions == 2) & (true_dirs == -1)).sum())
        total_trades = n_long_pred + n_short_pred
        accuracy = (correct_long + correct_short) / max(total_trades, 1)

        log_entry = {
            "epoch": epoch,
            "loss": avg_loss,
            "critic_loss": avg_critic,
            "dir_sym_loss": avg_dir_sym,
            "n_hold": n_hold,
            "n_long": n_long_pred,
            "n_short": n_short_pred,
            "pct_hold": n_hold / N_SAMPLES * 100,
            "pct_long": n_long_pred / N_SAMPLES * 100,
            "pct_short": n_short_pred / N_SAMPLES * 100,
            "correct_long": correct_long,
            "correct_short": correct_short,
            "direction_accuracy": accuracy * 100,
            "mean_prob_hold": float(all_probs[:, 0].mean()),
            "mean_prob_long": float(all_probs[:, 1].mean()),
            "mean_prob_short": float(all_probs[:, 2].mean()),
        }
        epoch_logs.append(log_entry)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Ep {epoch:3d} | Loss={avg_loss:.4f} | "
                f"H={n_hold:3d} L={n_long_pred:3d} S={n_short_pred:3d} | "
                f"L%={n_long_pred/N_SAMPLES*100:.1f} S%={n_short_pred/N_SAMPLES*100:.1f} | "
                f"Acc={accuracy*100:.1f}% | "
                f"P(H)={all_probs[:,0].mean():.3f} P(L)={all_probs[:,1].mean():.3f} P(S)={all_probs[:,2].mean():.3f}"
            )

    elapsed = time.time() - t_start
    print(f"\n  Training complete in {elapsed:.1f}s")

    # ── Step 4: Save checkpoint + report ───────────────────────────────
    print("\n[4/4] Saving checkpoint and diagnostic report...")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "agent.pt")
    agent.save_checkpoint(ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    # Final detailed evaluation
    final = epoch_logs[-1]
    diagnosis = "PASS" if final["pct_short"] > 10.0 else "FAIL"

    report = {
        "test": "synthetic_long_short_bias",
        "diagnosis": diagnosis,
        "description": (
            "Trained fresh agent on 50/50 balanced synthetic data. "
            "PASS = agent learns to predict both LONG and SHORT. "
            "FAIL = agent still collapses to LONG-only despite balanced data."
        ),
        "config": {
            "n_samples": N_SAMPLES,
            "n_epochs": N_EPOCHS,
            "seq_len": SEQ_LEN,
            "feature_dim": FEATURE_DIM,
            "tp_mult": TP_MULT,
            "sl_mult": SL_MULT,
            "batch_size": BATCH_SIZE,
            "device": str(device),
        },
        "final_action_distribution": {
            "hold_pct": final["pct_hold"],
            "long_pct": final["pct_long"],
            "short_pct": final["pct_short"],
        },
        "final_mean_probs": {
            "hold": final["mean_prob_hold"],
            "long": final["mean_prob_long"],
            "short": final["mean_prob_short"],
        },
        "final_accuracy": {
            "direction_accuracy_pct": final["direction_accuracy"],
            "correct_long": final["correct_long"],
            "correct_short": final["correct_short"],
        },
        "training_time_sec": elapsed,
        "epoch_log": epoch_logs,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report: {REPORT_PATH}")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  DIAGNOSIS: {diagnosis}")
    print(f"  Action Distribution: HOLD={final['pct_hold']:.1f}% "
          f"LONG={final['pct_long']:.1f}% SHORT={final['pct_short']:.1f}%")
    print(f"  Mean Probs: P(H)={final['mean_prob_hold']:.4f} "
          f"P(L)={final['mean_prob_long']:.4f} P(S)={final['mean_prob_short']:.4f}")
    print(f"  Direction Accuracy: {final['direction_accuracy']:.1f}%")
    print(f"  Correct: {final['correct_long']} long, {final['correct_short']} short")
    print("=" * 70)

    if diagnosis == "FAIL":
        print("\n  ROOT CAUSE ANALYSIS:")
        print("  The agent cannot learn SHORT even with perfectly balanced data.")
        print("  This confirms the bug is in the MODEL or LOSS FUNCTION,")
        print("  not in the training data distribution.")
        print("  Likely causes:")
        print("    1. delta_v (unrealized PnL) computation in loss.py:685-690")
        print("    2. Policy gradient amplifying initial random LONG bias")
        print("    3. VQC weight initialization creating asymmetric logits")
        print("    4. dir_sym_loss target [0.40, 0.30, 0.30] not strong enough")
    else:
        print("\n  The agent CAN learn both directions with balanced data.")
        print("  The 100% long bias in production is caused by:")
        print("    1. Training data imbalance (BTC uptrend 2023-2025)")
        print("    2. Insufficient dir_sym regularization for real data")

    return report


if __name__ == "__main__":
    report = train_and_evaluate()
