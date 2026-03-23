"""
test_agent_short_signal.py
─────────────────────────────────────────────────────────────────────
Failing test case: synthetic 13-dim structural features that MUST
produce a SHORT signal (action=2) from the QuantumFinancialAgent.

Root cause under investigation:
    The trained agent exhibits 100% long-only bias — it never outputs
    action=2 (SHORT) even when all structural features indicate a
    bearish regime (high liq_long_z, negative fr_z, bearish CVD, etc.).

This test will FAIL until the long-only bias bug is resolved.

Feature Index (13-dim structural):
  0: fr_z          1: fr_trend       2: oi_change_z    3: oi_price_div
  4: liq_long_z    5: liq_short_z    6: cvd_trend_z    7: cvd_price_div
  8: taker_ratio_z 9: ema200_dev    10: ema200_slope  11: vol_regime
 12: vol_change
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from src.models.integrated_agent import AgentConfig, QuantumFinancialAgent


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _make_agent(device: str = "cpu") -> QuantumFinancialAgent:
    """Build a fresh 13-dim agent with ALL safety gates disabled.

    Disabling gates isolates the test to the model's raw logit output,
    removing Lindblad / CR / entropy-production / Fisher-threshold
    confounders that can force Hold regardless of model output.
    """
    cfg = AgentConfig(
        feature_dim=13,
        n_eigenvectors=3,
        n_vqc_layers=2,
        n_actions=3,
        use_lightning=True,
        learnable_basis=False,
        # ── Disable all entry gates ──
        use_platt=False,
        use_lindblad=False,
        use_cr_filter=False,
        use_entropy_prod=False,
        use_fisher_threshold=False,
        use_mine=False,
        use_iqn=False,
        use_advanced_loss=False,
        # Low confidence threshold so gate doesn't block
        confidence_threshold=0.01,
        # No directional symmetry penalty during inference
        dir_sym_coef=0.0,
        act_sym_coef=0.0,
    )
    agent = QuantumFinancialAgent(config=cfg, device=torch.device(device))
    agent.eval()
    return agent


def _build_bearish_features(seq_len: int = 96) -> torch.Tensor:
    """Synthesize a [1, T, 13] tensor representing an extreme bearish regime.

    The feature values are designed to be the mirror image of a bullish
    scenario — every structural mechanism points to SHORT:
      - Funding rate deeply negative (shorts paying longs → crowded short squeeze unlikely)
      - Liq_long_z extremely high (longs being liquidated en masse)
      - Liq_short_z near zero (no short liquidations)
      - CVD strongly negative (persistent selling)
      - EMA200 deviation negative (price well below EMA200)
      - EMA200 slope negative (downtrend)
      - Vol regime elevated (high volatility → liquidation cascades)
    """
    T = seq_len
    feat = torch.zeros(1, T, 13)

    # Tier 1: Forced Liquidation — bearish
    feat[:, :, 0] = -3.0    # fr_z: deeply negative funding rate
    feat[:, :, 1] = -2.0    # fr_trend: funding rate accelerating down
    feat[:, :, 2] = -2.5    # oi_change_z: OI dropping (positions closing)
    feat[:, :, 3] = +2.0    # oi_price_div: OI-price divergence (bearish)
    feat[:, :, 4] = +4.0    # liq_long_z: massive long liquidations
    feat[:, :, 5] = 0.0     # liq_short_z: no short liquidations

    # Tier 2: Order Flow — bearish
    feat[:, :, 6] = -3.0    # cvd_trend_z: persistent selling
    feat[:, :, 7] = -2.0    # cvd_price_div: hidden distribution
    feat[:, :, 8] = -2.5    # taker_ratio_z: aggressive sellers dominate

    # Tier 3: Regime — bearish downtrend
    feat[:, :, 9]  = -0.10  # ema200_dev: price 10% below EMA200
    feat[:, :, 10] = -3.0   # ema200_slope: strong downtrend
    feat[:, :, 11] = +2.0   # vol_regime: elevated volatility
    feat[:, :, 12] = +1.5   # vol_change: volatility expanding

    return feat


def _build_bullish_features(seq_len: int = 96) -> torch.Tensor:
    """Synthesize a [1, T, 13] tensor for an extreme bullish regime (mirror)."""
    T = seq_len
    feat = torch.zeros(1, T, 13)

    feat[:, :, 0] = +3.0    # fr_z: highly positive funding (crowded longs paying)
    feat[:, :, 1] = +2.0    # fr_trend: funding rising
    feat[:, :, 2] = +2.5    # oi_change_z: OI surging
    feat[:, :, 3] = -2.0    # oi_price_div: OI-price aligned (bullish)
    feat[:, :, 4] = 0.0     # liq_long_z: no long liquidations
    feat[:, :, 5] = +4.0    # liq_short_z: massive short liquidations (squeeze)

    feat[:, :, 6] = +3.0    # cvd_trend_z: persistent buying
    feat[:, :, 7] = +2.0    # cvd_price_div: hidden accumulation
    feat[:, :, 8] = +2.5    # taker_ratio_z: aggressive buyers

    feat[:, :, 9]  = +0.10  # ema200_dev: price 10% above EMA200
    feat[:, :, 10] = +3.0   # ema200_slope: strong uptrend
    feat[:, :, 11] = +2.0   # vol_regime: elevated vol
    feat[:, :, 12] = +1.5   # vol_change: vol expanding

    return feat


def _build_bearish_features_dynamic(feat_dim: int = 13, seq_len: int = 96) -> torch.Tensor:
    """Bearish features tensor for arbitrary feature_dim.

    For dim > 13, pads with zeros. For dim == 13, identical to
    _build_bearish_features. Used for checkpoint tests where the
    trained model may have a different feature_dim.
    """
    T = seq_len
    feat = torch.zeros(1, T, feat_dim)

    # Fill first 13 dims with bearish pattern (if feat_dim >= 13)
    n = min(feat_dim, 13)
    bearish_13 = _build_bearish_features(seq_len=T)
    feat[:, :, :n] = bearish_13[:, :, :n]

    return feat


# ────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────

class TestShortSignalGeneration:
    """Tests that the agent can produce SHORT (action=2) signals.

    These tests are expected to FAIL until the 100% long-only bias
    bug in QuantumFinancialAgent is resolved.
    """

    def test_bearish_features_produce_short(self):
        """Core failing test: extreme bearish input → action must be SHORT (2).

        If the agent is functioning correctly, an input where every
        structural feature screams 'bearish' should produce action=2.
        A result of action=1 (LONG) here proves the long-only bias.
        """
        agent = _make_agent()
        x = _build_bearish_features()

        action, prob, probs = agent.select_action(x, atr_norm=0.02, mode="greedy")

        # Diagnostic: print raw probability distribution
        print(f"\n[DIAG] Bearish input → probs: HOLD={probs[0]:.4f}, "
              f"LONG={probs[1]:.4f}, SHORT={probs[2]:.4f}")
        print(f"[DIAG] Selected action={action} (0=Hold, 1=Long, 2=Short)")

        assert action == 2, (
            f"LONG-ONLY BIAS BUG: bearish features produced action={action} "
            f"(expected SHORT=2). Probs: H={probs[0]:.4f}, L={probs[1]:.4f}, "
            f"S={probs[2]:.4f}"
        )

    def test_short_probability_exceeds_long(self):
        """P(SHORT) must be > P(LONG) on bearish features."""
        agent = _make_agent()
        x = _build_bearish_features()

        _, _, probs = agent.select_action(x, atr_norm=0.02, mode="greedy")

        p_long = float(probs[1])
        p_short = float(probs[2])

        print(f"\n[DIAG] P(LONG)={p_long:.4f}, P(SHORT)={p_short:.4f}")

        assert p_short > p_long, (
            f"LONG-ONLY BIAS BUG: P(SHORT)={p_short:.4f} <= P(LONG)={p_long:.4f} "
            f"on bearish features. The model cannot distinguish bearish from bullish."
        )

    def test_symmetry_bearish_vs_bullish(self):
        """Bearish and bullish mirror inputs must produce opposite actions.

        If both produce LONG, the model has collapsed to a single mode.
        """
        agent = _make_agent()
        x_bear = _build_bearish_features()
        x_bull = _build_bullish_features()

        act_bear, _, probs_bear = agent.select_action(x_bear, atr_norm=0.02)
        act_bull, _, probs_bull = agent.select_action(x_bull, atr_norm=0.02)

        print(f"\n[DIAG] Bearish → action={act_bear}, probs={probs_bear.tolist()}")
        print(f"[DIAG] Bullish → action={act_bull}, probs={probs_bull.tolist()}")

        # At minimum, they must not both be LONG
        assert not (act_bear == 1 and act_bull == 1), (
            f"MODE COLLAPSE: both bearish and bullish inputs produce LONG (action=1). "
            f"Bear probs: {probs_bear.tolist()}, Bull probs: {probs_bull.tolist()}"
        )

    def test_short_over_many_random_bearish_inputs(self):
        """Over 50 random bearish-biased inputs, SHORT must appear at least once.

        This tests that the model is not structurally incapable of
        outputting action=2, even with stochastic VQC initialization.
        """
        agent = _make_agent()
        actions = []

        for i in range(50):
            x = _build_bearish_features(seq_len=96)
            # Add small random noise to each sample
            noise = torch.randn_like(x) * 0.3
            x_noisy = x + noise

            act, _, _ = agent.select_action(x_noisy, atr_norm=0.02, mode="greedy")
            actions.append(act)

        short_count = actions.count(2)
        long_count = actions.count(1)
        hold_count = actions.count(0)

        print(f"\n[DIAG] 50 bearish samples → HOLD={hold_count}, "
              f"LONG={long_count}, SHORT={short_count}")

        assert short_count > 0, (
            f"STRUCTURAL SHORT BLOCKAGE: 0/{len(actions)} bearish inputs produced "
            f"SHORT. HOLD={hold_count}, LONG={long_count}. "
            f"The model is structurally incapable of outputting action=2."
        )

    def test_logit_symmetry_raw_forward(self):
        """Check raw logits (before softmax) for SHORT vs LONG bias.

        Bypasses select_action entirely to inspect the model's raw output.
        If logit[SHORT] is always << logit[LONG], the bias is in the
        network weights or architecture, not in the gating logic.
        """
        agent = _make_agent()
        x = _build_bearish_features()

        with torch.no_grad():
            logits, expvals, J, c_kt = agent.forward(x, last_step_only=True)

        logits_last = logits[0, -1, :]  # [3]
        print(f"\n[DIAG] Raw logits: HOLD={logits_last[0]:.4f}, "
              f"LONG={logits_last[1]:.4f}, SHORT={logits_last[2]:.4f}")

        # On bearish features, SHORT logit must exceed LONG logit
        assert logits_last[2] > logits_last[1], (
            f"RAW LOGIT BIAS: logit[SHORT]={logits_last[2]:.4f} <= "
            f"logit[LONG]={logits_last[1]:.4f} on extreme bearish input. "
            f"Bias exists at the network/VQC level, not just gating."
        )

    def test_loaded_checkpoint_short_signal(self):
        """If a trained checkpoint exists, verify it can produce SHORT signals.

        This test loads the actual trained checkpoint and checks if
        the trained model has the long-only bias. Skipped if no
        checkpoint exists or architecture doesn't match.
        """
        import os
        import glob as globmod
        ckpt_paths = [
            "checkpoints/structural_13dim/agent_best.pt",
            "checkpoints/structural_13dim/test_agent.pt",
            "test_agent.pt",
        ]
        # Also search for any .pt files in structural checkpoint dir
        ckpt_paths += globmod.glob("checkpoints/structural_13dim/*.pt")
        ckpt_paths += globmod.glob("checkpoints/quantum_v2/*.pt")

        ckpt_path = None
        for p in ckpt_paths:
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path is None:
            pytest.skip("No trained checkpoint found — skipping checkpoint test")

        # Load checkpoint to inspect its config
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_config = ckpt.get("config", {})

        # Build agent matching checkpoint's architecture
        cfg = AgentConfig(
            feature_dim=ckpt_config.get("feature_dim", 13),
            n_eigenvectors=ckpt_config.get("n_eigenvectors", 5),
            n_vqc_layers=ckpt_config.get("n_vqc_layers", 3),
            n_actions=3,
            # Disable all gates to isolate model output
            use_platt=False,
            use_lindblad=False,
            use_cr_filter=False,
            use_entropy_prod=False,
            use_fisher_threshold=False,
            use_mine=False,
            use_iqn=ckpt_config.get("use_iqn", False),
            use_advanced_loss=ckpt_config.get("use_advanced_loss", False),
            confidence_threshold=0.01,
        )
        agent = QuantumFinancialAgent(config=cfg, device=torch.device("cpu"))
        try:
            agent.load_checkpoint(ckpt_path)
        except RuntimeError as e:
            pytest.skip(f"Checkpoint architecture mismatch: {e}")
        agent.eval()

        feat_dim = cfg.feature_dim
        x = _build_bearish_features_dynamic(feat_dim=feat_dim)
        action, _, probs = agent.select_action(x, atr_norm=0.02)

        print(f"\n[DIAG] Checkpoint {ckpt_path} → action={action}, "
              f"probs={probs.tolist()}")

        assert action == 2, (
            f"TRAINED MODEL LONG-ONLY BIAS: checkpoint {ckpt_path} produced "
            f"action={action} on bearish features. "
            f"Probs: H={probs[0]:.4f}, L={probs[1]:.4f}, S={probs[2]:.4f}"
        )


class TestShortSignalSamplingMode:
    """Tests using stochastic sampling (mode='sample') to check if
    SHORT is even in the support of the policy distribution."""

    def test_sample_mode_can_produce_short(self):
        """In 200 stochastic samples from bearish input, SHORT must appear.

        If SHORT never appears even in sampling mode, it means
        P(SHORT) ≈ 0 — the probability mass has collapsed.
        """
        agent = _make_agent()
        x = _build_bearish_features()

        short_seen = False
        for _ in range(200):
            act, _, _ = agent.select_action(x, atr_norm=0.02, mode="sample")
            if act == 2:
                short_seen = True
                break

        assert short_seen, (
            "P(SHORT) ≈ 0: SHORT never sampled in 200 draws from bearish input. "
            "The policy distribution has collapsed — SHORT is not in the support."
        )


class TestTrainingInducedBias:
    """Tests that simulate the training process and detect if training
    itself introduces the long-only bias.

    Key hypothesis: the loss function or gradient flow during train_step
    asymmetrically suppresses SHORT logits, even when the reward signal
    is symmetric.
    """

    def test_single_train_step_preserves_short_capability(self):
        """After one train_step with balanced rewards, SHORT must still be possible.

        This isolates whether even a single gradient update kills SHORT.
        """
        agent = _make_agent()

        # Verify pre-training: bearish → SHORT works
        x_bear = _build_bearish_features()
        act_pre, _, probs_pre = agent.select_action(x_bear, atr_norm=0.02)
        p_short_pre = float(probs_pre[2])

        # One training step with synthetic data + symmetric rewards
        agent.train()
        B, T = 4, 96
        x_train = torch.randn(B, T, 13)

        # Balanced actions: 2 LONGs, 2 SHORTs (no HOLD bias)
        actions = torch.tensor([1, 2, 1, 2])
        # Symmetric rewards: LONGs and SHORTs both get +1
        rewards = torch.ones(B)

        try:
            agent.train_step(x_train, actions, rewards)
        except Exception:
            pytest.skip("train_step signature incompatible — needs investigation")

        # Post-training: bearish → SHORT must still be possible
        agent.eval()
        _, _, probs_post = agent.select_action(x_bear, atr_norm=0.02)
        p_short_post = float(probs_post[2])

        print(f"\n[DIAG] P(SHORT) pre={p_short_pre:.4f}, post={p_short_post:.4f}")

        assert p_short_post > 0.05, (
            f"TRAINING KILLS SHORT: P(SHORT) dropped from {p_short_pre:.4f} "
            f"to {p_short_post:.4f} after a single balanced train_step. "
            f"Post probs: {probs_post.tolist()}"
        )

    def test_logit_bias_after_init(self):
        """Check if the model's output bias term favors LONG over SHORT at init.

        If bias[LONG] >> bias[SHORT] at initialization, the model
        starts with a structural advantage for LONG that training reinforces.
        """
        agent = _make_agent()

        # Check if there's an explicit bias in the logit projection layer
        bias_found = False
        for name, param in agent.named_parameters():
            if "logit" in name.lower() and "bias" in name.lower():
                bias_found = True
                bias_vals = param.detach().cpu().numpy()
                print(f"\n[DIAG] {name}: {bias_vals}")
                if len(bias_vals) == 3:
                    assert abs(bias_vals[1] - bias_vals[2]) < 0.5, (
                        f"INIT BIAS ASYMMETRY: logit bias LONG={bias_vals[1]:.4f} "
                        f"vs SHORT={bias_vals[2]:.4f}. Difference "
                        f"= {abs(bias_vals[1] - bias_vals[2]):.4f} (should be < 0.5)"
                    )

        if not bias_found:
            # No explicit bias — check by running symmetric inputs
            x_zero = torch.zeros(1, 96, 13)
            with torch.no_grad():
                logits, _, _, _ = agent.forward(x_zero, last_step_only=True)
            logits_last = logits[0, -1, :]
            print(f"\n[DIAG] Zero-input logits: {logits_last.tolist()}")
            # LONG and SHORT logits should be similar on zero input
            assert abs(float(logits_last[1] - logits_last[2])) < 1.0, (
                f"ARCHITECTURAL ASYMMETRY: zero-input logits LONG={logits_last[1]:.4f} "
                f"vs SHORT={logits_last[2]:.4f}. The architecture has a built-in bias."
            )
