#!/usr/bin/env python3
"""
Debug: Trace why QuantumFinancialAgent produces 100% LONG bias.

Creates synthetic short-biased inputs and traces every stage:
  features → SpectralDecomposer → c_kt → QuantumHamiltonianLayer → logits → probs → action

Expected: A short-biased input (high liq_short_z, negative cvd_trend_z)
          should produce action=2 (SHORT). If it doesn't, we identify where
          the asymmetry breaks.
"""

import sys, os, math, io
import torch
import numpy as np

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.integrated_agent import QuantumFinancialAgent, AgentConfig


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def build_synthetic_features(scenario: str, seq_len: int = 96, feat_dim: int = 13) -> torch.Tensor:
    """Build synthetic [1, T, feat_dim] tensors for specific scenarios.

    IMPORTANT: Features must have temporal variation, otherwise Z-score
    normalization (x - mean_t) / std_t will map constant signals to zero.
    We simulate a ramp-up pattern: features transition from neutral to extreme.

    13-dim structural features order (from memory):
      0: fr_z           (funding rate z-score)
      1: fr_trend        (funding rate trend)
      ...
    """
    x = torch.randn(1, seq_len, feat_dim) * 0.3  # base noise

    # Create a ramp: 0 → 1 over the sequence (simulates regime transition)
    t = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]

    if scenario == "strong_short":
        # Features ramp toward short-biased extremes
        targets = torch.tensor([[-2.5, -0.5, -1.5, 1.5, 0.2, 3.0, -2.0, -1.5, -1.5, 2.0, -0.3, 0.7, 1.0]])
        x = x + t * targets.unsqueeze(1)  # ramp from noise to target

    elif scenario == "strong_long":
        targets = torch.tensor([[2.5, 0.5, 1.5, -1.5, 3.0, 0.2, 2.0, 1.5, 1.5, -2.0, 0.3, 0.7, 1.0]])
        x = x + t * targets.unsqueeze(1)

    elif scenario == "neutral":
        pass  # just noise

    elif scenario == "negated_short":
        targets = torch.tensor([[-2.5, -0.5, -1.5, 1.5, -3.0, -0.2, -2.0, -1.5, -1.5, 2.0, -0.3, -0.7, -1.0]])
        x = x + t * targets.unsqueeze(1)

    return x


def trace_forward_pass(agent: QuantumFinancialAgent, x: torch.Tensor, label: str):
    """Trace every intermediate stage and print diagnostics."""
    print_section(f"Scenario: {label}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Input stats: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"Feature means: {[f'{x[0,:,i].mean():.3f}' for i in range(x.shape[2])]}")

    agent.eval()
    x = x.to(agent.device)

    with torch.no_grad():
        # Stage 1: SpectralDecomposer
        encoder = agent.encoder
        _, c_kt_raw, delta_c_kt, _ = encoder.decomposer(x)
        print(f"\n[Stage 1] SpectralDecomposer → c_kt")
        print(f"  c_kt shape: {tuple(c_kt_raw.shape)}")
        print(f"  c_kt last step: {c_kt_raw[0, -1, :].cpu().numpy()}")
        print(f"  c_kt mean:      {c_kt_raw.mean(dim=1)[0].cpu().numpy()}")
        print(f"  delta_c_kt last: {delta_c_kt[0, -1, :].cpu().numpy()}")

        # Stage 1.5: Temporal Encoder
        c_kt = c_kt_raw
        if encoder.temporal_encoder is not None:
            c_kt = encoder.temporal_encoder(c_kt)
            print(f"\n[Stage 1.5] TemporalContextEncoder → c_kt")
            print(f"  c_kt last step (post-transformer): {c_kt[0, -1, :].cpu().numpy()}")

        # Stage 1.6: proj_to_qubits
        c_kt_q = c_kt
        delta_c_kt_q = delta_c_kt
        if encoder.proj_to_qubits is not None:
            c_kt_q = encoder.proj_to_qubits(c_kt)
            delta_c_kt_q = encoder.proj_to_qubits(delta_c_kt)
            print(f"\n[Stage 1.6] proj_to_qubits (5→3)")
            print(f"  c_kt_q last step:      {c_kt_q[0, -1, :].cpu().numpy()}")
            print(f"  delta_c_kt_q last step: {delta_c_kt_q[0, -1, :].cpu().numpy()}")

        # Stage 2: QuantumHamiltonianLayer internals
        ql = encoder.quantum_layer
        B, T, K = c_kt_q.shape

        # Ising coupling
        J = ql.coupling(c_kt_q, delta_c_kt_q)
        print(f"\n[Stage 2a] Ising coupling J")
        print(f"  J shape: {tuple(J.shape)}")
        print(f"  J[0]: {J[0].cpu().numpy()}")

        # Encoding angles
        theta = math.pi * torch.sigmoid(c_kt_q / 3.0)
        phi = math.pi * torch.tanh(delta_c_kt_q / 3.0)
        print(f"\n[Stage 2b] Encoding angles (last timestep)")
        print(f"  theta (RY) = pi*sigmoid(c/3): {theta[0, -1, :].cpu().numpy()}")
        print(f"  phi   (RZ) = pi*tanh(dc/3):   {phi[0, -1, :].cpu().numpy()}")
        print(f"  NOTE: theta ∈ [0, π] ALWAYS (sigmoid is non-negative)")
        print(f"  NOTE: phi ∈ [-π, π] (tanh is symmetric)")

        # External field
        h_field = torch.tanh(ql.h_proj(c_kt_q.mean(1)) / 3.0)
        print(f"\n[Stage 2c] External field h_field")
        print(f"  h_field: {h_field[0].cpu().numpy()}")

        # VQC execution (last step only)
        theta_last = theta[:, -1:, :]
        phi_last = phi[:, -1:, :]
        J_flat = J.unsqueeze(1)
        theta_flat = theta_last.reshape(B, K)
        phi_flat = phi_last.reshape(B, K)
        J_flat2 = J_flat.reshape(B, K, K)
        expvals_flat = ql._run_circuit_batch(theta_flat, phi_flat, J_flat2)
        expvals = expvals_flat.reshape(B, 1, K)
        print(f"\n[Stage 2d] VQC expectation values ⟨σ^z⟩")
        print(f"  expvals: {expvals[0, 0, :].cpu().numpy()}")

        # Logit projection
        h_broadcast = h_field.unsqueeze(1).expand(-1, 1, -1)
        vqc_input = expvals + h_broadcast
        logits_quantum = ql.logit_proj(vqc_input)
        print(f"\n[Stage 2e] logit_proj(expvals + h_field)")
        print(f"  VQC input: {vqc_input[0, 0, :].cpu().numpy()}")
        print(f"  logits_quantum: {logits_quantum[0, 0, :].cpu().numpy()}")

        # Classical head
        c_classical = c_kt_q[:, -1:, :]
        logits_classical = ql.classical_head(c_classical)
        print(f"\n[Stage 2f] classical_head(c_kt)")
        print(f"  logits_classical: {logits_classical[0, 0, :].cpu().numpy()}")

        # Combined logits
        logit_scale = float(ql.logit_scale.item())
        logits = ql.logit_scale * (logits_classical + logits_quantum)
        print(f"\n[Stage 2g] Combined logits = scale * (classical + quantum)")
        print(f"  logit_scale: {logit_scale:.4f}")
        print(f"  logits: {logits[0, 0, :].cpu().numpy()}")
        print(f"  logits: HOLD={logits[0,0,0]:.4f}, LONG={logits[0,0,1]:.4f}, SHORT={logits[0,0,2]:.4f}")

        # Softmax
        probs = torch.softmax(logits[0, 0, :], dim=0)
        print(f"\n[Stage 3] Softmax probabilities")
        print(f"  P(HOLD)={probs[0]:.4f}, P(LONG)={probs[1]:.4f}, P(SHORT)={probs[2]:.4f}")
        action = int(probs.argmax().item())
        action_name = {0: "HOLD", 1: "LONG", 2: "SHORT"}[action]
        print(f"  → Selected action: {action} ({action_name})")

        # Check logit_proj bias
        _lp = ql.logit_proj
        _final = _lp[-1] if isinstance(_lp, torch.nn.Sequential) else _lp
        if hasattr(_final, 'bias') and _final.bias is not None:
            b = _final.bias.cpu().detach().numpy()
            print(f"\n[Stage 4] logit_proj final bias")
            print(f"  bias: HOLD={b[0]:.6f}, LONG={b[1]:.6f}, SHORT={b[2]:.6f}")
            print(f"  LONG-SHORT gap: {b[1]-b[2]:.6f}")

        # Check classical_head bias
        ch = ql.classical_head
        _ch_final = ch[-1] if isinstance(ch, torch.nn.Sequential) else ch
        if hasattr(_ch_final, 'bias') and _ch_final.bias is not None:
            b2 = _ch_final.bias.cpu().detach().numpy()
            print(f"\n[Stage 5] classical_head bias")
            print(f"  bias: HOLD={b2[0]:.6f}, LONG={b2[1]:.6f}, SHORT={b2[2]:.6f}")
            print(f"  LONG-SHORT gap: {b2[1]-b2[2]:.6f}")

        # Stage 6: Full select_action call
        print(f"\n[Stage 6] Full select_action() call")
        action_full, prob_full, probs_full = agent.select_action(x, atr_norm=0.01, mode="greedy")
        action_name_full = {0: "HOLD", 1: "LONG", 2: "SHORT"}[action_full]
        print(f"  action={action_full} ({action_name_full}), prob={prob_full:.4f}")
        print(f"  probs: HOLD={probs_full[0]:.4f}, LONG={probs_full[1]:.4f}, SHORT={probs_full[2]:.4f}")

        # Check which gates are blocking
        if action_full == 0 and action != 0:
            print(f"  ⚠ Raw action was {action_name} but got HOLD — a gate is blocking!")

    return logits[0, 0, :].cpu().numpy(), probs.cpu().numpy()


def check_weight_symmetry(agent: QuantumFinancialAgent):
    """Check if learned weights have inherent LONG bias."""
    print_section("Weight Symmetry Analysis")

    ql = agent.encoder.quantum_layer

    # logit_proj weights
    _lp = ql.logit_proj
    for i, layer in enumerate(_lp):
        if hasattr(layer, 'weight'):
            w = layer.weight.cpu().detach()
            print(f"\n  logit_proj[{i}] weight: shape={tuple(w.shape)}")
            if w.shape[0] == 3:  # final layer
                print(f"    W[HOLD] norm: {w[0].norm():.6f}")
                print(f"    W[LONG] norm: {w[1].norm():.6f}")
                print(f"    W[SHORT] norm: {w[2].norm():.6f}")
                print(f"    W[LONG] - W[SHORT] diff norm: {(w[1]-w[2]).norm():.6f}")
                # Check if LONG and SHORT weights are nearly identical
                cos_sim = torch.nn.functional.cosine_similarity(w[1].unsqueeze(0), w[2].unsqueeze(0))
                print(f"    cosine_sim(W[LONG], W[SHORT]): {cos_sim.item():.6f}")
        if hasattr(layer, 'bias') and layer.bias is not None:
            b = layer.bias.cpu().detach()
            print(f"    bias: {b.numpy()}")

    # classical_head weights
    ch = ql.classical_head
    if isinstance(ch, torch.nn.Sequential):
        for i, layer in enumerate(ch):
            if hasattr(layer, 'weight'):
                w = layer.weight.cpu().detach()
                if w.shape[0] == 3:
                    print(f"\n  classical_head weight: shape={tuple(w.shape)}")
                    print(f"    W[HOLD] norm: {w[0].norm():.6f}")
                    print(f"    W[LONG] norm: {w[1].norm():.6f}")
                    print(f"    W[SHORT] norm: {w[2].norm():.6f}")
    elif hasattr(ch, 'weight'):
        w = ch.weight.cpu().detach()
        print(f"\n  classical_head weight: shape={tuple(w.shape)}")
        print(f"    W[HOLD] norm: {w[0].norm():.6f}")
        print(f"    W[LONG] norm: {w[1].norm():.6f}")
        print(f"    W[SHORT] norm: {w[2].norm():.6f}")
        if hasattr(ch, 'bias') and ch.bias is not None:
            b = ch.bias.cpu().detach()
            print(f"    bias: {b.numpy()}")

    # VQC weights
    vqc_w = ql.vqc_weights.cpu().detach()
    print(f"\n  VQC weights: shape={tuple(vqc_w.shape)}, mean={vqc_w.mean():.6f}, std={vqc_w.std():.6f}")

    # logit_scale
    print(f"\n  logit_scale: {ql.logit_scale.item():.6f}")


def main():
    print_section("Long-Bias Root Cause Analysis")

    # Try loading the structural_13dim checkpoint
    ckpt_path = "checkpoints/structural_13dim/test_agent.pt"
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        # Try agent_best.pth
        ckpt_path = "checkpoints/structural_13dim/agent_best.pth"
        if not os.path.isfile(ckpt_path):
            print(f"No checkpoint found. Running with random init for structural analysis.")
            ckpt_path = None

    # Determine feature_dim from checkpoint or default
    feat_dim = 13

    # Build agent config
    config = AgentConfig(
        feature_dim=feat_dim,
        n_eigenvectors=5,
        use_advanced_loss=True,
        use_platt=True,        # Keep enabled to match checkpoint
        use_lindblad=True,     # Keep enabled to match checkpoint
        use_cr_filter=False,
        use_entropy_prod=False,
        use_fisher_threshold=False,
        confidence_threshold=0.0,  # Don't filter anything
    )

    print(f"Config: feature_dim={config.feature_dim}, n_eigenvectors={config.n_eigenvectors}")

    # Force CPU to avoid device mismatch issues
    agent = QuantumFinancialAgent(config)
    agent = agent.cpu()
    agent.device = torch.device("cpu")

    if ckpt_path is not None:
        print(f"\nLoading checkpoint: {ckpt_path}")
        try:
            agent.load_checkpoint(ckpt_path)
            print("  ✓ Checkpoint loaded successfully")
        except Exception as e:
            print(f"  ✗ Failed to load checkpoint: {e}")
            print("  Continuing with random weights...")

    # Check decomposer state
    dec = agent.encoder.decomposer
    print(f"\nSpectralDecomposer state:")
    print(f"  use_lda={dec.use_lda}, _lda_fitted={dec._lda_fitted}")
    print(f"  use_edmd={dec.use_edmd}, _koop_fitted={dec._koop_fitted}")
    print(f"  learnable_basis={dec.learnable_basis}")
    print(f"  feature_dim={dec.feature_dim}, n_eigenvectors={dec.n_eigenvectors}")
    if dec._lda_fitted and hasattr(dec, '_lda_W'):
        print(f"  LDA W shape: {tuple(dec._lda_W.shape)}")
        print(f"  LDA W norms: {[f'{dec._lda_W[:, i].norm():.4f}' for i in range(dec._lda_W.shape[1])]}")

    # Check weight symmetry
    check_weight_symmetry(agent)

    # Run scenarios
    results = {}
    for scenario in ["strong_short", "strong_long", "neutral", "negated_short"]:
        x = build_synthetic_features(scenario, seq_len=96, feat_dim=feat_dim)
        logits, probs = trace_forward_pass(agent, x, scenario)
        results[scenario] = {"logits": logits, "probs": probs}

    # Summary
    print_section("SUMMARY — Root Cause Diagnosis")
    print(f"\n{'Scenario':<20} {'HOLD':>8} {'LONG':>8} {'SHORT':>8} {'Action':>8}")
    print("-" * 60)
    for sc, r in results.items():
        p = r["probs"]
        action = int(np.argmax(p))
        action_name = {0: "HOLD", 1: "LONG", 2: "SHORT"}[action]
        print(f"{sc:<20} {p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>8.4f} {action_name:>8}")

    # Diagnosis
    print("\n--- Diagnosis ---")
    all_long = all(np.argmax(r["probs"]) == 1 for r in results.values())
    if all_long:
        print("⚠ ALL scenarios produce LONG → structural bias in model weights")
        # Check which component dominates
        for sc in ["strong_short", "strong_long"]:
            logits = results[sc]["logits"]
            print(f"  {sc}: logit gap (LONG-SHORT) = {logits[1]-logits[2]:.4f}")
    else:
        print("✓ Model can produce different actions for different inputs")
        for sc, r in results.items():
            action = int(np.argmax(r["probs"]))
            if sc == "strong_short" and action != 2:
                print(f"  ⚠ {sc} expected SHORT(2) but got {action}")
            elif sc == "strong_long" and action != 1:
                print(f"  ⚠ {sc} expected LONG(1) but got {action}")


if __name__ == "__main__":
    main()
