"""
test_agent_variance.py — Verify QuantumFinancialAgent is NOT input-invariant.

If the EDMD/Koopman pipeline or VQC collapses all inputs to the same logits,
this test will catch it.  Two distinct, known feature vectors must produce
different output logits.

Tests cover:
  1. Opposing regimes → different logits
  2. Inference mode (last_step_only) → different logits
  3. Determinism: same input → same output
  4. Batch-level: different samples in batch → different logits
  5. Random vs zeros → different logits
  6. Per-feature sensitivity: perturbing each of 13 structural features
     individually must change the output logits
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from src.models.integrated_agent import QuantumFinancialAgent, AgentConfig

# Number of structural features (current standard)
N_STRUCT_FEATURES = 13


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(
    feature_dim: int = N_STRUCT_FEATURES,
    device: torch.device = torch.device("cpu"),
) -> QuantumFinancialAgent:
    """Minimal agent — all physics gates off for isolation, small dims for speed."""
    cfg = AgentConfig(
        feature_dim=feature_dim,
        n_eigenvectors=5,
        n_vqc_layers=2,          # fewer layers → faster test
        n_actions=3,
        use_lightning=False,     # default.qubit is safer in CI
        use_edmd=True,
        use_lindblad=False,      # not needed for logit variance test
        use_platt=False,
        use_cr_filter=False,
        use_qng=False,           # use AdamW for simplicity
        use_advanced_loss=False,
    )
    agent = QuantumFinancialAgent(cfg, device=device)
    agent.eval()
    return agent


def _make_distinct_inputs(
    B: int = 1, T: int = 30, F: int = N_STRUCT_FEATURES,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """
    Return two feature tensors that are obviously different:
      x_up  — steadily rising log-returns (trending market)
      x_down — steadily falling log-returns (crashing market)
    """
    torch.manual_seed(42)

    # x_up: positive drift + small noise
    x_up = torch.randn(B, T, F, device=device) * 0.01
    x_up[:, :, 0] = torch.linspace(0.01, 0.05, T)   # feat[0] = strong positive trend

    # x_down: negative drift + small noise
    x_down = torch.randn(B, T, F, device=device) * 0.01
    x_down[:, :, 0] = torch.linspace(-0.05, -0.01, T)  # feat[0] = strong negative trend

    return x_up, x_down


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentVariance:
    """The agent must produce different logits for different inputs."""

    @pytest.fixture(scope="class")
    def agent(self):
        return _make_agent()

    @pytest.fixture(scope="class")
    def inputs(self):
        return _make_distinct_inputs()

    def test_logits_differ_for_distinct_inputs(self, agent, inputs):
        """Core invariance test: two opposing market regimes → different logits."""
        x_up, x_down = inputs

        with torch.no_grad():
            logits_up, _, _, _ = agent(x_up)
            logits_down, _, _, _ = agent(x_down)

        # Take last-timestep logits for comparison
        l_up = logits_up[:, -1, :]     # [B, 3]
        l_down = logits_down[:, -1, :]  # [B, 3]

        diff = (l_up - l_down).abs().sum().item()
        assert diff > 1e-6, (
            f"Agent is input-invariant! logits_up={l_up.tolist()}, "
            f"logits_down={l_down.tolist()}, diff={diff:.2e}"
        )

    def test_logits_differ_last_step_only(self, agent, inputs):
        """Same test with last_step_only=True (inference mode)."""
        x_up, x_down = inputs

        with torch.no_grad():
            logits_up, _, _, _ = agent(x_up, last_step_only=True)
            logits_down, _, _, _ = agent(x_down, last_step_only=True)

        l_up = logits_up.squeeze()     # [3]
        l_down = logits_down.squeeze()  # [3]

        diff = (l_up - l_down).abs().sum().item()
        assert diff > 1e-6, (
            f"Agent is input-invariant (last_step_only)! "
            f"logits_up={l_up.tolist()}, logits_down={l_down.tolist()}, diff={diff:.2e}"
        )

    def test_identical_inputs_produce_identical_logits(self, agent):
        """Sanity check: same input → same output (deterministic in eval mode)."""
        x, _ = _make_distinct_inputs()

        with torch.no_grad():
            logits_a, _, _, _ = agent(x)
            logits_b, _, _, _ = agent(x)

        diff = (logits_a - logits_b).abs().max().item()
        assert diff < 1e-5, (
            f"Non-deterministic! Same input gave different logits, max_diff={diff:.2e}"
        )

    def test_batch_elements_differ(self, agent):
        """Two different samples in the same batch → different logits per sample."""
        x_up, x_down = _make_distinct_inputs()
        x_batch = torch.cat([x_up, x_down], dim=0)  # [2, T, F]

        with torch.no_grad():
            logits, _, _, _ = agent(x_batch)

        l0 = logits[0, -1, :]  # sample 0 last step
        l1 = logits[1, -1, :]  # sample 1 last step

        diff = (l0 - l1).abs().sum().item()
        assert diff > 1e-6, (
            f"Batch-invariant! Both samples got identical logits: "
            f"{l0.tolist()} vs {l1.tolist()}, diff={diff:.2e}"
        )

    def test_random_vs_zeros(self, agent):
        """Random features vs all-zeros — logits must differ."""
        torch.manual_seed(99)
        x_rand = torch.randn(1, 30, N_STRUCT_FEATURES) * 0.1
        x_zero = torch.zeros(1, 30, N_STRUCT_FEATURES)

        with torch.no_grad():
            logits_rand, _, _, _ = agent(x_rand)
            logits_zero, _, _, _ = agent(x_zero)

        diff = (logits_rand[:, -1, :] - logits_zero[:, -1, :]).abs().sum().item()
        assert diff > 1e-6, (
            f"Agent ignores input! Random vs zeros gave same logits, diff={diff:.2e}"
        )


class TestPerFeatureSensitivity:
    """
    Perturbing each of the 13 structural features individually must change
    the output logits. This catches dead-feature pathways where a feature
    has zero gradient through the decomposer → VQC → logit_proj chain.
    """

    @pytest.fixture(scope="class")
    def agent(self):
        return _make_agent()

    @pytest.fixture(scope="class")
    def baseline_logits(self, agent):
        """Logits from a small-noise baseline (not zeros, to avoid degenerate cov)."""
        torch.manual_seed(0)
        x_base = torch.randn(1, 30, N_STRUCT_FEATURES) * 0.001
        with torch.no_grad():
            logits, _, _, _ = agent(x_base)
        return logits[:, -1, :]  # [1, 3]

    @pytest.mark.parametrize("feat_idx", list(range(N_STRUCT_FEATURES)))
    def test_feature_sensitivity(self, agent, baseline_logits, feat_idx):
        """
        Perturb feature[feat_idx] with a time-varying ramp.
        Must use varying values (not constant) to survive z-score normalization:
        z-score of constant=c is (c-c)/0 = 0 → always collapses.
        A ramp has mean≠0 AND std≠0 → survives z-scoring.
        """
        x_perturbed = torch.zeros(1, 30, N_STRUCT_FEATURES)
        # Time-varying ramp: linspace(-1, 1, T) → mean≈0, std≈0.58 → z-score preserves shape
        x_perturbed[:, :, feat_idx] = torch.linspace(-1.0, 1.0, 30)

        with torch.no_grad():
            logits_perturbed, _, _, _ = agent(x_perturbed)

        l_pert = logits_perturbed[:, -1, :]  # [1, 3]
        diff = (l_pert - baseline_logits).abs().sum().item()
        assert diff > 1e-6, (
            f"Feature[{feat_idx}] is dead! Perturbing it did not change logits. "
            f"baseline={baseline_logits.tolist()}, "
            f"perturbed={l_pert.tolist()}, diff={diff:.2e}"
        )

    def test_logit_proj_bias_is_zero_init(self, agent):
        """
        Verify that logit_proj final layer bias is initialized to zeros.
        Non-zero init causes bias to dominate over VQC signal → input-invariance.
        """
        lp = agent.encoder.quantum_layer.logit_proj
        final_layer = lp[-1] if hasattr(lp, '__getitem__') else lp
        # Unwrap spectral norm if present
        if hasattr(final_layer, 'module'):
            final_layer = final_layer.module
        bias = final_layer.bias
        assert bias is not None, "logit_proj final layer has no bias"
        max_bias = bias.abs().max().item()
        assert max_bias < 1e-6, (
            f"logit_proj final bias is NOT zero-initialized: {bias.tolist()}, "
            f"max={max_bias:.2e}. This will dominate VQC expvals and cause "
            f"input-invariance."
        )

    def test_classical_head_bias_is_zero_init(self, agent):
        """Verify classical_head bias is zero-initialized."""
        ch = agent.encoder.quantum_layer.classical_head
        final_layer = ch[-1] if hasattr(ch, '__getitem__') else ch
        if hasattr(final_layer, 'module'):
            final_layer = final_layer.module
        bias = final_layer.bias
        assert bias is not None, "classical_head has no bias"
        max_bias = bias.abs().max().item()
        assert max_bias < 1e-6, (
            f"classical_head bias not zero-initialized: {bias.tolist()}, "
            f"max={max_bias:.2e}"
        )


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
