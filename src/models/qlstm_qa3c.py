"""QLSTM + QA3C Hybrid Quantum Reinforcement-Learning Engine.

Combines Quantum LSTM (feature extraction) with Quantum Advantage Actor-Critic
(QA3C, decision-making) for Bitcoin futures trading.  Includes physics-based
noise filtering (Schrödinger trading equation) and modified Kelly position sizing.

References
----------
- Quantum Temporal Convolutional Neural.pdf  (Eq. 23-30: QLSTM)
- 비트코인 양자 트레이딩 시스템 설계.pdf     (system design)
"""

from __future__ import annotations

import atexit
import math
import torch
import os
import time
import threading
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.logging import setup_logger

logger = setup_logger("qlstm_qa3c")

# ──────────────────────────────────────────────
# §0  Low-level quantum-circuit primitives
#     (self-contained; mirrors qtc_nn.py API)
# ──────────────────────────────────────────────

def _ry(theta: float) -> np.ndarray:
    """R_y rotation gate."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    """R_z rotation gate."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


def _cnot() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )


def _cz() -> np.ndarray:
    return np.diag(np.array([1, 1, 1, -1], dtype=complex))


def _apply_single(state: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    t = state.reshape([2] * n)
    t = np.tensordot(gate, t, axes=[[1], [q]])
    t = np.moveaxis(t, 0, q)
    return t.reshape(-1)


def _apply_two(state: np.ndarray, gate: np.ndarray, q0: int, q1: int, n: int) -> np.ndarray:
    if q0 == q1:
        return state
    lo, hi = min(q0, q1), max(q0, q1)
    t = state.reshape([2] * n)
    g4 = gate.reshape(2, 2, 2, 2)
    if q0 < q1:
        t = np.tensordot(g4, t, axes=[[2, 3], [lo, hi]])
        t = np.moveaxis(t, [0, 1], [lo, hi])
    else:
        t = np.tensordot(g4, t, axes=[[2, 3], [hi, lo]])
        t = np.moveaxis(t, [0, 1], [hi, lo])
    return t.reshape(-1)


def _expectation_z(state: np.ndarray, q: int, n: int) -> float:
    t = state.reshape([2] * n)
    p0 = float(np.sum(np.abs(np.take(t, 0, axis=q)) ** 2))
    p1 = float(np.sum(np.abs(np.take(t, 1, axis=q)) ** 2))
    return p0 - p1


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


def _minmax_to_pi(x: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx == mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn) * math.pi


# ──────────────────────────────────────────────
# §0-batch  Batched quantum-circuit primitives
#           Operate on (B, 2^n) state arrays
# ──────────────────────────────────────────────

def _minmax_to_pi_batch(X: np.ndarray) -> np.ndarray:
    """Per-row minmax→[0,π] scaling.  X shape (B, n_qubits) → (B, n_qubits)."""
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    rng = mx - mn
    rng = np.where(rng < 1e-30, 1.0, rng)  # avoid div-by-zero
    return np.where(mx == mn, 0.0, (X - mn) / rng * math.pi)


def _apply_single_batch(
    states: np.ndarray, gate: np.ndarray, q: int, n: int
) -> np.ndarray:
    """Apply a fixed 2×2 gate to qubit q across B states.

    states : (B, 2^n)
    gate   : (2, 2)
    Returns: (B, 2^n)
    """
    B = states.shape[0]
    shape = [B] + [2] * n
    t = states.reshape(shape)                     # (B, 2, 2, ..., 2)
    # tensordot on gate axis 1 ↔ qubit axis (q+1 because axis 0 is batch)
    t = np.tensordot(gate, t, axes=[[1], [q + 1]])  # (2, B, 2, ..., 2) with q-th axis consumed
    # moveaxis: axis 0 (gate output) → axis q+1
    t = np.moveaxis(t, 0, q + 1)
    return t.reshape(B, -1)


def _apply_single_batch_variable(
    states: np.ndarray, angles: np.ndarray, gate_type: str, q: int, n: int
) -> np.ndarray:
    """Apply per-element Ry or Rz rotation to qubit q.

    states : (B, 2^n)
    angles : (B,)  — rotation angle per batch element
    gate_type : 'ry' or 'rz'
    Returns: (B, 2^n)
    """
    B = states.shape[0]
    dim = 2 ** n
    shape = [B] + [2] * n

    half = angles / 2.0  # (B,)
    c = np.cos(half)
    s = np.sin(half)

    if gate_type == "ry":
        # Ry = [[c, -s], [s, c]]  — real-valued
        # Build (B, 2, 2) gates
        gates = np.zeros((B, 2, 2), dtype=complex)
        gates[:, 0, 0] = c
        gates[:, 0, 1] = -s
        gates[:, 1, 0] = s
        gates[:, 1, 1] = c
    elif gate_type == "rz":
        # Rz = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
        gates = np.zeros((B, 2, 2), dtype=complex)
        gates[:, 0, 0] = c - 1j * s
        gates[:, 1, 1] = c + 1j * s
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")

    t = states.reshape(shape)  # (B, 2, ..., 2)

    # Apply per-element gates via einsum
    # We need: out[b, i0, ..., iq, ..., in-1] = sum_j gate[b, iq_new, j] * t[b, i0, ..., j, ..., in-1]
    # Strategy: split along qubit axis, multiply, reassemble
    # Move target qubit axis to last, apply gate, move back
    t = np.moveaxis(t, q + 1, -1)            # (B, ..., 2)  target qubit now last
    out = np.einsum("bij,b...j->b...i", gates, t)  # (B, ..., 2)
    out = np.moveaxis(out, -1, q + 1)        # (B, 2, ..., 2)
    return out.reshape(B, dim)


def _apply_two_batch(
    states: np.ndarray, gate: np.ndarray, q0: int, q1: int, n: int
) -> np.ndarray:
    """Apply a fixed 4×4 two-qubit gate (e.g. CNOT) across B states.

    states : (B, 2^n)
    gate   : (4, 4) or (2, 2, 2, 2)
    Returns: (B, 2^n)
    """
    if q0 == q1:
        return states.copy()
    B = states.shape[0]
    shape = [B] + [2] * n
    t = states.reshape(shape)
    g4 = gate.reshape(2, 2, 2, 2)
    lo, hi = min(q0, q1), max(q0, q1)
    if q0 < q1:
        t = np.tensordot(g4, t, axes=[[2, 3], [lo + 1, hi + 1]])
        t = np.moveaxis(t, [0, 1], [lo + 1, hi + 1])
    else:
        t = np.tensordot(g4, t, axes=[[2, 3], [hi + 1, lo + 1]])
        t = np.moveaxis(t, [0, 1], [hi + 1, lo + 1])
    return t.reshape(B, -1)


def _expectation_z_batch(
    states: np.ndarray, q: int, n: int
) -> np.ndarray:
    """⟨Z_q⟩ for B states.  Returns (B,)."""
    B = states.shape[0]
    shape = [B] + [2] * n
    t = states.reshape(shape)
    p0 = np.sum(np.abs(np.take(t, 0, axis=q + 1)) ** 2, axis=tuple(range(1, n)))
    p1 = np.sum(np.abs(np.take(t, 1, axis=q + 1)) ** 2, axis=tuple(range(1, n)))
    return p0 - p1


def _angle_embed_batch(z: np.ndarray, n_qubits: int) -> np.ndarray:
    """Angle-embed B inputs into |0⟩^n.  z shape (B, n_qubits) → (B, 2^n)."""
    B = z.shape[0]
    dim = 2 ** n_qubits
    states = np.zeros((B, dim), dtype=complex)
    states[:, 0] = 1.0
    for i in range(n_qubits):
        states = _apply_single_batch_variable(states, z[:, i], "ry", i, n_qubits)
    return states


def _entangle_batch(states: np.ndarray, n_qubits: int) -> np.ndarray:
    """CNOT ring across B states."""
    cnot = _cnot()
    for q in range(n_qubits - 1):
        states = _apply_two_batch(states, cnot, q, q + 1, n_qubits)
    states = _apply_two_batch(states, cnot, n_qubits - 1, 0, n_qubits)
    return states


# ──────────────────────────────────────────────
# §0-gpu  GPU-Accelerated VQC via PyTorch Autograd
# ──────────────────────────────────────────────

_torch_device_cache: Optional[Any] = None
_torch_available: Optional[bool] = None


def _get_torch_device(prefer: str = "cuda") -> Any:
    """Lazy GPU detection. Returns torch.device or None if torch unavailable."""
    global _torch_device_cache, _torch_available
    if _torch_available is False:
        return None
    if _torch_device_cache is not None:
        return _torch_device_cache
    try:
        import torch
        _torch_available = True
        if prefer == "cuda" and torch.cuda.is_available():
            _torch_device_cache = torch.device("cuda")
        else:
            _torch_device_cache = torch.device("cpu")
        return _torch_device_cache
    except ImportError:
        _torch_available = False
        return None


def _vqc_forward_torch(
    x_batch: "torch.Tensor",
    theta: "torch.Tensor",
    phi: "torch.Tensor",
    n_qubits: int,
    n_layers: int,
) -> "torch.Tensor":
    """Differentiable VQC forward pass on GPU via torch tensor ops.

    Simulates the full statevector circuit using complex128 tensors.
    All operations preserve autograd graph for theta, phi, x_batch.

    Parameters
    ----------
    x_batch : (B, nq) raw inputs (NOT pre-scaled — scaling done inside)
    theta   : (L, Q) Ry rotation params, requires_grad=True
    phi     : (L, Q) Rz rotation params, requires_grad=True
    n_qubits, n_layers : circuit shape

    Returns
    -------
    (B, nq) Z-expectations, differentiable w.r.t. theta, phi, x_batch
    """
    import torch

    B = x_batch.shape[0]
    nq = n_qubits
    dim = 2 ** nq
    device = x_batch.device

    # ── minmax→[0,π] scaling (per-row) ──
    mn = x_batch.min(dim=1, keepdim=True).values
    mx = x_batch.max(dim=1, keepdim=True).values
    rng = mx - mn
    rng = torch.where(rng < 1e-30, torch.ones_like(rng), rng)
    z = torch.where(mx == mn, torch.zeros_like(x_batch),
                    (x_batch - mn) / rng * math.pi)  # (B, nq)

    # ── Gate helpers operating on (B, 2, 2, ..., 2) = (B, *[2]*nq) ──

    def _apply_ry(states_nd, angles, q):
        """Apply Ry(angles) on qubit q.  states_nd: (B, 2^nq), angles: (B,)."""
        t = states_nd.reshape([B] + [2] * nq)
        c = torch.cos(angles / 2).to(torch.complex128)  # (B,)
        s = torch.sin(angles / 2).to(torch.complex128)

        # Select |0⟩ and |1⟩ components along qubit q
        idx0 = [slice(None)] * (nq + 1)
        idx1 = [slice(None)] * (nq + 1)
        idx0[q + 1] = 0
        idx1[q + 1] = 1

        v0 = t[tuple(idx0)]  # (B, 2, ..., 2) with axis q removed
        v1 = t[tuple(idx1)]

        # Ry: |0'⟩ = c*|0⟩ - s*|1⟩, |1'⟩ = s*|0⟩ + c*|1⟩
        # Broadcast c,s to match v0 shape
        shape = [B] + [1] * (nq - 1)
        c = c.reshape(shape)
        s = s.reshape(shape)

        new_v0 = c * v0 - s * v1
        new_v1 = s * v0 + c * v1

        out = torch.stack([new_v0, new_v1], dim=q + 1)
        return out.reshape(B, dim)

    def _apply_rz(states_nd, angles, q):
        """Apply Rz(angles) on qubit q.  states_nd: (B, 2^nq), angles: (B,)."""
        t = states_nd.reshape([B] + [2] * nq)
        c = torch.cos(angles / 2)
        s = torch.sin(angles / 2)
        phase_0 = (c - 1j * s).to(torch.complex128)
        phase_1 = (c + 1j * s).to(torch.complex128)

        idx0 = [slice(None)] * (nq + 1)
        idx1 = [slice(None)] * (nq + 1)
        idx0[q + 1] = 0
        idx1[q + 1] = 1

        v0 = t[tuple(idx0)]
        v1 = t[tuple(idx1)]

        shape = [B] + [1] * (nq - 1)
        new_v0 = phase_0.reshape(shape) * v0
        new_v1 = phase_1.reshape(shape) * v1

        out = torch.stack([new_v0, new_v1], dim=q + 1)
        return out.reshape(B, dim)

    def _apply_cnot(states_nd, ctrl, targ):
        """CNOT: X on targ when ctrl=|1⟩.  Fully differentiable."""
        if ctrl == targ:
            return states_nd
        t = states_nd.reshape([B] + [2] * nq)

        # When ctrl=|0⟩: target unchanged
        # When ctrl=|1⟩: target flipped
        idx_c0 = [slice(None)] * (nq + 1)
        idx_c1 = [slice(None)] * (nq + 1)
        idx_c0[ctrl + 1] = 0
        idx_c1[ctrl + 1] = 1

        block_c0 = t[tuple(idx_c0)]  # ctrl=0, shape drops ctrl dim
        block_c1 = t[tuple(idx_c1)]  # ctrl=1

        # In block_c1, flip target qubit
        # targ axis in block_c1: if targ > ctrl, it's at position targ (shifted by -1)
        # if targ < ctrl, it's at position targ + 1 - 1 = targ
        targ_ax = targ if targ < ctrl else targ - 1
        block_c1_flipped = torch.flip(block_c1, dims=[targ_ax + 1])

        out = torch.stack([block_c0, block_c1_flipped], dim=ctrl + 1)
        return out.reshape(B, dim)

    # ── Build initial state |0⟩^n ──
    states = torch.zeros(B, dim, dtype=torch.complex128, device=device)
    states[:, 0] = 1.0

    # ── Angle embedding: Ry(z_q) on each qubit ──
    for q in range(nq):
        states = _apply_ry(states, z[:, q], q)

    # ── Variational layers ──
    for l in range(n_layers):
        # Entangling CNOT ring
        for q in range(nq - 1):
            states = _apply_cnot(states, q, q + 1)
        states = _apply_cnot(states, nq - 1, 0)

        # Parameterized rotations
        for q in range(nq):
            states = _apply_ry(states, theta[l, q].expand(B), q)
            states = _apply_rz(states, phi[l, q].expand(B), q)

        # Data re-uploading (except last layer)
        if l < n_layers - 1:
            for q in range(nq):
                states = _apply_ry(states, z[:, q] * 0.5, q)

    # ── Z-expectations: ⟨Z_q⟩ = P(|0⟩) - P(|1⟩) ──
    t = states.reshape([B] + [2] * nq)
    expectations = []
    for q in range(nq):
        idx0 = [slice(None)] * (nq + 1)
        idx1 = [slice(None)] * (nq + 1)
        idx0[q + 1] = 0
        idx1[q + 1] = 1
        p0 = torch.sum(torch.abs(t[tuple(idx0)]) ** 2,
                       dim=list(range(1, nq)))  # sum over all but batch
        p1 = torch.sum(torch.abs(t[tuple(idx1)]) ** 2,
                       dim=list(range(1, nq)))
        expectations.append(p0 - p1)  # (B,)

    return torch.stack(expectations, dim=1)  # (B, nq)


def _qlstm_step_torch(
    x_t: "torch.Tensor",
    h_prev: "torch.Tensor",
    c_prev: "torch.Tensor",
    W_proj: "torch.Tensor",
    b_proj: "torch.Tensor",
    W_gate: "torch.Tensor",
    vqc_params: "List[Tuple[torch.Tensor, torch.Tensor]]",
    n_qubits: int,
    n_layers: int,
    input_dim: int,
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """Differentiable QLSTM step via torch autograd.

    Parameters
    ----------
    x_t      : (B, input_dim) current features
    h_prev   : (B, hidden_dim)
    c_prev   : (B, hidden_dim)
    W_proj   : (n_qubits, input_dim+hidden_dim) projection weights
    b_proj   : (n_qubits,) projection bias
    W_gate   : (hidden_dim, n_qubits) back-projection weights
    vqc_params : list of 4 (theta, phi) tuples for [forget, input, output, cell]
    n_qubits, n_layers : VQC circuit shape
    input_dim : feature dimension (to slice x_t)

    Returns
    -------
    (h_new, c_new) each (B, hidden_dim), differentiable
    """
    import torch

    # Concatenate: (B, input_dim + hidden_dim)
    combined = torch.cat([x_t[:, :input_dim], h_prev], dim=1)

    # Project to qubit space: (B, n_qubits)
    u = combined @ W_proj.t() + b_proj.unsqueeze(0)

    # Layer norm (per-sample)
    u_mean = u.mean(dim=-1, keepdim=True)
    u_std = u.std(dim=-1, keepdim=True)
    u = (u - u_mean) / (u_std + 1e-8)

    # 4 VQC gate calls
    theta_f, phi_f = vqc_params[0]
    theta_i, phi_i = vqc_params[1]
    theta_o, phi_o = vqc_params[2]
    theta_g, phi_g = vqc_params[3]

    f_q = _vqc_forward_torch(u, theta_f, phi_f, n_qubits, n_layers)  # (B, nq)
    i_q = _vqc_forward_torch(u, theta_i, phi_i, n_qubits, n_layers)
    o_q = _vqc_forward_torch(u, theta_o, phi_o, n_qubits, n_layers)
    g_q = _vqc_forward_torch(u, theta_g, phi_g, n_qubits, n_layers)

    # Back-project to hidden_dim and apply gate nonlinearities
    f = torch.sigmoid(f_q @ W_gate.t())  # (B, hidden_dim)
    i = torch.sigmoid(i_q @ W_gate.t())
    o = torch.sigmoid(o_q @ W_gate.t())
    g = torch.tanh(g_q @ W_gate.t())

    # LSTM cell update
    c_new = f * c_prev + i * g
    h_new = o * torch.tanh(c_new)

    return h_new, c_new


def _qlstm_forward_sequence_torch(
    X_normed: "torch.Tensor",
    W_proj: "torch.Tensor",
    b_proj: "torch.Tensor",
    W_gate: "torch.Tensor",
    vqc_params: "List[Tuple[torch.Tensor, torch.Tensor]]",
    n_qubits: int,
    n_layers: int,
    input_dim: int,
    hidden_dim: int,
) -> "torch.Tensor":
    """Differentiable QLSTM forward over a batch of sequences.

    Parameters
    ----------
    X_normed  : (B, T, F) normalized input sequences
    Returns final hidden state (B, hidden_dim)
    """
    import torch

    B, T, F = X_normed.shape
    device = X_normed.device

    h = torch.zeros(B, hidden_dim, dtype=torch.float64, device=device)
    c = torch.zeros(B, hidden_dim, dtype=torch.float64, device=device)

    for t in range(T):
        x_t = X_normed[:, t, :]  # (B, F)
        h, c = _qlstm_step_torch(
            x_t, h, c,
            W_proj, b_proj, W_gate,
            vqc_params, n_qubits, n_layers, input_dim,
        )

    return h  # (B, hidden_dim)


def _compute_batch_grads_gpu(
    states: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    target_returns: np.ndarray,
    actor_proj: np.ndarray,
    actor_vqc_flat: np.ndarray,
    critic_proj: np.ndarray,
    critic_vqc_flat: np.ndarray,
    n_qubits_actor: int,
    n_layers_actor: int,
    n_qubits_critic: int,
    n_layers_critic: int,
    state_dim: int,
    entropy_coeff: float,
    grad_clip: float = 1.0,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    """GPU-accelerated gradient computation using torch.autograd.

    Replaces 48+ parameter-shift forward passes with a single
    forward + backward pass through the differentiable VQC.
    """
    import torch

    dev = _get_torch_device(device_str)
    if dev is None:
        raise RuntimeError("PyTorch not available")

    T = len(states)
    n_act = 3

    # ── Actor ──
    nqa = n_qubits_actor
    half_a = n_layers_actor * nqa
    a_theta_np = actor_vqc_flat[:half_a].reshape(n_layers_actor, nqa)
    a_phi_np = actor_vqc_flat[half_a:].reshape(n_layers_actor, nqa) if len(actor_vqc_flat) > half_a else np.zeros((n_layers_actor, nqa))

    # Create differentiable torch tensors
    a_proj_t = torch.tensor(actor_proj, dtype=torch.float64, device=dev, requires_grad=True)
    a_theta_t = torch.tensor(a_theta_np, dtype=torch.float64, device=dev, requires_grad=True)
    a_phi_t = torch.tensor(a_phi_np, dtype=torch.float64, device=dev, requires_grad=True)

    states_t = torch.tensor(states[:, :state_dim], dtype=torch.float64, device=dev)
    actions_t = torch.tensor(actions, dtype=torch.long, device=dev)
    advantages_t = torch.tensor(advantages, dtype=torch.float64, device=dev)
    returns_t = torch.tensor(target_returns, dtype=torch.float64, device=dev)

    # Forward: project states → VQC
    U_a = states_t @ a_proj_t.T  # (T, nqa)
    Z_a = _vqc_forward_torch(U_a, a_theta_t, a_phi_t, nqa, n_layers_actor)  # (T, nqa)

    # Softmax → policy
    logits_a = Z_a[:, :n_act]  # (T, 3)
    probs = torch.softmax(logits_a, dim=1)  # (T, 3)

    # Log-prob of chosen action
    log_probs = torch.log(probs.gather(1, actions_t.unsqueeze(1)).squeeze(1) + 1e-10)

    # Entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # (T,)

    # Actor loss: -log_prob * advantage - entropy_coeff * entropy
    actor_loss = torch.sum(-log_probs * advantages_t - entropy_coeff * entropy)

    # Backward
    actor_loss.backward()

    # Extract gradients
    sum_actor_dth = np.clip(a_theta_t.grad.cpu().numpy(), -grad_clip, grad_clip)
    sum_actor_dph = np.clip(a_phi_t.grad.cpu().numpy(), -grad_clip, grad_clip)
    sum_actor_dW = np.clip(a_proj_t.grad.cpu().numpy(), -grad_clip, grad_clip)

    actor_loss_val = float(actor_loss.detach().cpu())
    entropy_val = float(entropy.sum().detach().cpu())

    # ── Critic ──
    nqc = n_qubits_critic
    half_c = n_layers_critic * nqc
    c_theta_np = critic_vqc_flat[:half_c].reshape(n_layers_critic, nqc)
    c_phi_np = critic_vqc_flat[half_c:].reshape(n_layers_critic, nqc) if len(critic_vqc_flat) > half_c else np.zeros((n_layers_critic, nqc))

    c_proj_t = torch.tensor(critic_proj, dtype=torch.float64, device=dev, requires_grad=True)
    c_theta_t = torch.tensor(c_theta_np, dtype=torch.float64, device=dev, requires_grad=True)
    c_phi_t = torch.tensor(c_phi_np, dtype=torch.float64, device=dev, requires_grad=True)

    U_c = states_t @ c_proj_t.T  # (T, nqc)
    Z_c = _vqc_forward_torch(U_c, c_theta_t, c_phi_t, nqc, n_layers_critic)  # (T, nqc)

    # Critic loss: MSE on V(s)
    critic_loss = torch.sum((Z_c[:, 0] - returns_t) ** 2)
    critic_loss.backward()

    sum_critic_dth = np.clip(c_theta_t.grad.cpu().numpy(), -grad_clip, grad_clip)
    sum_critic_dph = np.clip(c_phi_t.grad.cpu().numpy(), -grad_clip, grad_clip)
    sum_critic_dW = np.clip(c_proj_t.grad.cpu().numpy(), -grad_clip, grad_clip)

    return {
        "actor_dth": sum_actor_dth,
        "actor_dph": sum_actor_dph,
        "actor_dW": sum_actor_dW,
        "critic_dth": sum_critic_dth,
        "critic_dph": sum_critic_dph,
        "critic_dW": sum_critic_dW,
        "actor_loss": actor_loss_val,
        "entropy": entropy_val,
    }


# ──────────────────────────────────────────────
# §0a  Process Pool for Parallel Gradient Computation
# ──────────────────────────────────────────────

_grad_pool: Optional[ProcessPoolExecutor] = None
_grad_pool_lock = threading.Lock()


def _get_grad_pool(max_workers: int = 0) -> ProcessPoolExecutor:
    """Lazy-init a module-level ProcessPoolExecutor (thread-safe)."""
    global _grad_pool
    if _grad_pool is None:
        with _grad_pool_lock:
            if _grad_pool is None:
                if max_workers <= 0:
                    max_workers = min(os.cpu_count() or 4, 8)
                _grad_pool = ProcessPoolExecutor(max_workers=max_workers)
    return _grad_pool


def _shutdown_grad_pool() -> None:
    """Clean up pool at interpreter exit."""
    global _grad_pool
    if _grad_pool is not None:
        _grad_pool.shutdown(wait=False)
        _grad_pool = None


atexit.register(_shutdown_grad_pool)


def _compute_timestep_grads_both(args: Tuple) -> Dict[str, Any]:
    """Process-safe gradient computation for one timestep (actor + critic).

    Reconstructs local VQCs from flat params — no shared state.
    Returns gradient arrays + loss + entropy as a dict.
    """
    (
        actor_proj_flat, actor_proj_shape, actor_vqc_flat,
        critic_proj_flat, critic_proj_shape, critic_vqc_flat,
        n_qubits_actor, n_layers_actor,
        n_qubits_critic, n_layers_critic,
        state, action, advantage, target_return,
        entropy_coeff, state_dim, grad_clip,
    ) = args

    # Reconstruct actor VQC
    a_vqc = VQC(n_qubits=n_qubits_actor, n_layers=n_layers_actor, seed=0)
    a_vqc.load_flat(actor_vqc_flat)
    a_proj = actor_proj_flat.reshape(actor_proj_shape)
    n_act = 3

    # --- Actor forward ---
    u_a = a_proj @ state[:state_dim]
    z_a = a_vqc.forward(u_a)
    probs = _softmax(z_a[:n_act])
    entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
    actor_loss = -math.log(probs[action] + 1e-10) * advantage - entropy_coeff * entropy

    # Analytic dL/dz
    nq_a = a_vqc.n_qubits
    dL_dz = np.zeros(nq_a)
    for k in range(n_act):
        one_hot = 1.0 if k == action else 0.0
        dL_dz[k] = advantage * (probs[k] - one_hot)
        dL_dz[k] += entropy_coeff * probs[k] * (
            math.log(probs[k] + 1e-10) + entropy
        )

    # Parameter-shift: dz/d(theta,phi)
    g_th, g_ph = a_vqc.grad_full(u_a)
    dL_dth = np.clip(np.einsum("lqk,k->lq", g_th, dL_dz), -grad_clip, grad_clip)
    dL_dph = np.clip(np.einsum("lqk,k->lq", g_ph, dL_dz), -grad_clip, grad_clip)

    # Chain rule: dL/dW_proj
    dz_du = a_vqc.input_grad(u_a)
    dL_du = dL_dz @ dz_du
    dL_dW = np.clip(np.outer(dL_du, state[:state_dim]), -grad_clip, grad_clip)

    # --- Critic forward ---
    c_vqc = VQC(n_qubits=n_qubits_critic, n_layers=n_layers_critic, seed=0)
    c_vqc.load_flat(critic_vqc_flat)
    c_proj = critic_proj_flat.reshape(critic_proj_shape)

    u_c = c_proj @ state[:state_dim]
    z_c = c_vqc.forward(u_c)
    critic_err = float(z_c[0]) - target_return

    nq_c = c_vqc.n_qubits
    dL_dz_c = np.zeros(nq_c)
    dL_dz_c[0] = critic_err

    g_th_c, g_ph_c = c_vqc.grad_full(u_c)
    dL_dth_c = np.clip(np.einsum("lqk,k->lq", g_th_c, dL_dz_c), -grad_clip, grad_clip)
    dL_dph_c = np.clip(np.einsum("lqk,k->lq", g_ph_c, dL_dz_c), -grad_clip, grad_clip)

    dz_du_c = c_vqc.input_grad(u_c)
    dL_du_c = dL_dz_c @ dz_du_c
    dL_dW_c = np.clip(np.outer(dL_du_c, state[:state_dim]), -grad_clip, grad_clip)

    return {
        "actor_dth": dL_dth,
        "actor_dph": dL_dph,
        "actor_dW": dL_dW,
        "critic_dth": dL_dth_c,
        "critic_dph": dL_dph_c,
        "critic_dW": dL_dW_c,
        "actor_loss": actor_loss,
        "entropy": entropy,
    }


def _compute_batch_grads_both(
    states: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    target_returns: np.ndarray,
    actor_proj: np.ndarray,
    actor_vqc_flat: np.ndarray,
    critic_proj: np.ndarray,
    critic_vqc_flat: np.ndarray,
    n_qubits_actor: int,
    n_layers_actor: int,
    n_qubits_critic: int,
    n_layers_critic: int,
    state_dim: int,
    entropy_coeff: float,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    """Vectorized gradient computation for ALL T timesteps at once.

    Instead of dispatching T separate process-pool tasks, this computes
    all forward passes and parameter-shift gradients using batched VQC ops.
    Returns summed gradients (same accumulation as the per-timestep loop).
    """
    T = len(states)
    n_act = 3

    # ── Actor ──
    a_vqc = VQC(n_qubits=n_qubits_actor, n_layers=n_layers_actor, seed=0)
    a_vqc.load_flat(actor_vqc_flat)
    nq_a = a_vqc.n_qubits

    # Project all states at once: (T, state_dim) @ (nq_a, state_dim).T → (T, nq_a)
    U_a = states[:, :state_dim] @ actor_proj.T

    # Batched forward for actor
    Z_a = a_vqc.forward_batch(U_a)  # (T, nq_a)

    # Compute probs, entropy, loss per timestep
    probs_all = np.zeros((T, n_act))
    entropy_all = np.zeros(T)
    actor_loss_all = np.zeros(T)
    dL_dz_all = np.zeros((T, nq_a))

    for t in range(T):
        probs = _softmax(Z_a[t, :n_act])
        probs_all[t] = probs
        ent = -float(np.sum(probs * np.log(probs + 1e-10)))
        entropy_all[t] = ent
        actor_loss_all[t] = (
            -math.log(probs[actions[t]] + 1e-10) * advantages[t]
            - entropy_coeff * ent
        )

        # Analytic dL/dz
        for k in range(n_act):
            one_hot = 1.0 if k == actions[t] else 0.0
            dL_dz_all[t, k] = advantages[t] * (probs[k] - one_hot)
            dL_dz_all[t, k] += entropy_coeff * probs[k] * (
                math.log(probs[k] + 1e-10) + ent
            )

    # Parameter-shift gradients: per-timestep grad_full_batched
    sum_actor_dth = np.zeros((n_layers_actor, nq_a))
    sum_actor_dph = np.zeros((n_layers_actor, nq_a))
    sum_actor_dW = np.zeros_like(actor_proj)

    for t in range(T):
        g_th, g_ph = a_vqc.grad_full_batched(U_a[t])
        dL_dth = np.clip(np.einsum("lqk,k->lq", g_th, dL_dz_all[t]), -grad_clip, grad_clip)
        dL_dph = np.clip(np.einsum("lqk,k->lq", g_ph, dL_dz_all[t]), -grad_clip, grad_clip)
        sum_actor_dth += dL_dth
        sum_actor_dph += dL_dph

    # Input gradients: batch all T shifted inputs at once
    # Build (T * 2 * nq_a, nq_a) shifted inputs
    eps = 0.01
    n_input = min(U_a.shape[1], nq_a)
    all_shifted = np.tile(U_a, (1, 1))  # (T, nq_a) copy
    shift_inputs = []
    for t in range(T):
        for i in range(n_input):
            x_p = U_a[t].copy(); x_p[i] += eps
            x_m = U_a[t].copy(); x_m[i] -= eps
            shift_inputs.append(x_p)
            shift_inputs.append(x_m)
    shift_inputs = np.array(shift_inputs)  # (T*2*n_input, nq_a)

    # Cap batch size to avoid memory blowup
    MAX_BATCH = 512
    if len(shift_inputs) <= MAX_BATCH:
        z_shifted = a_vqc.forward_batch(shift_inputs)
    else:
        chunks = [shift_inputs[i:i + MAX_BATCH] for i in range(0, len(shift_inputs), MAX_BATCH)]
        z_shifted = np.vstack([a_vqc.forward_batch(c) for c in chunks])

    # Extract per-timestep input grads and chain-rule for dL/dW
    idx = 0
    for t in range(T):
        dz_du = np.zeros((nq_a, n_input))
        for i in range(n_input):
            dz_du[:, i] = (z_shifted[idx] - z_shifted[idx + 1]) / (2.0 * eps)
            idx += 2
        dL_du = dL_dz_all[t] @ dz_du
        dL_dW = np.clip(np.outer(dL_du, states[t, :state_dim]), -grad_clip, grad_clip)
        sum_actor_dW += dL_dW

    # ── Critic ──
    c_vqc = VQC(n_qubits=n_qubits_critic, n_layers=n_layers_critic, seed=0)
    c_vqc.load_flat(critic_vqc_flat)
    nq_c = c_vqc.n_qubits

    U_c = states[:, :state_dim] @ critic_proj.T  # (T, nq_c)
    Z_c = c_vqc.forward_batch(U_c)  # (T, nq_c)

    dL_dz_c_all = np.zeros((T, nq_c))
    dL_dz_c_all[:, 0] = Z_c[:, 0] - target_returns

    sum_critic_dth = np.zeros((n_layers_critic, nq_c))
    sum_critic_dph = np.zeros((n_layers_critic, nq_c))
    sum_critic_dW = np.zeros_like(critic_proj)

    for t in range(T):
        g_th_c, g_ph_c = c_vqc.grad_full_batched(U_c[t])
        dL_dth_c = np.clip(np.einsum("lqk,k->lq", g_th_c, dL_dz_c_all[t]), -grad_clip, grad_clip)
        dL_dph_c = np.clip(np.einsum("lqk,k->lq", g_ph_c, dL_dz_c_all[t]), -grad_clip, grad_clip)
        sum_critic_dth += dL_dth_c
        sum_critic_dph += dL_dph_c

    # Critic input gradients
    n_input_c = min(U_c.shape[1], nq_c)
    shift_inputs_c = []
    for t in range(T):
        for i in range(n_input_c):
            x_p = U_c[t].copy(); x_p[i] += eps
            x_m = U_c[t].copy(); x_m[i] -= eps
            shift_inputs_c.append(x_p)
            shift_inputs_c.append(x_m)
    shift_inputs_c = np.array(shift_inputs_c)

    if len(shift_inputs_c) <= MAX_BATCH:
        z_shifted_c = c_vqc.forward_batch(shift_inputs_c)
    else:
        chunks = [shift_inputs_c[i:i + MAX_BATCH] for i in range(0, len(shift_inputs_c), MAX_BATCH)]
        z_shifted_c = np.vstack([c_vqc.forward_batch(c) for c in chunks])

    idx = 0
    for t in range(T):
        dz_du_c = np.zeros((nq_c, n_input_c))
        for i in range(n_input_c):
            dz_du_c[:, i] = (z_shifted_c[idx] - z_shifted_c[idx + 1]) / (2.0 * eps)
            idx += 2
        dL_du_c = dL_dz_c_all[t] @ dz_du_c
        dL_dW_c = np.clip(np.outer(dL_du_c, states[t, :state_dim]), -grad_clip, grad_clip)
        sum_critic_dW += dL_dW_c

    return {
        "actor_dth": sum_actor_dth,
        "actor_dph": sum_actor_dph,
        "actor_dW": sum_actor_dW,
        "critic_dth": sum_critic_dth,
        "critic_dph": sum_critic_dph,
        "critic_dW": sum_critic_dW,
        "actor_loss": float(np.sum(actor_loss_all)),
        "entropy": float(np.sum(entropy_all)),
    }


def _precompute_window_task(args: Tuple) -> Tuple[List[np.ndarray], List[float]]:
    """Process-safe window preprocessing: build features + QLSTM states.

    Reconstructs QLSTM from serialized params — no shared state.
    Returns (hidden_states, log_returns).
    """
    (
        window_data,  # dict of column arrays (not DataFrame — for pickling)
        qlstm_params,
        n_features, seq_len,
        input_dim, hidden_dim, n_qubits, zscore_window,
    ) = args

    # Reconstruct DataFrame
    window_df = pd.DataFrame(window_data)

    # Reconstruct QLSTM
    qlstm = QLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_qubits=n_qubits,
        seed=0,
        zscore_window=zscore_window,
    )
    qlstm.load_params(qlstm_params)

    # Build features
    feats = []
    for i in range(len(window_df)):
        feats.append(build_features(window_df.iloc[: i + 1], n_features))
    all_features = np.array(feats)

    # Build QLSTM states
    hidden_states: List[np.ndarray] = []
    log_returns: List[float] = []
    n_bars = len(all_features)
    for i in range(seq_len, n_bars):
        seq = all_features[i - seq_len + 1 : i + 1]
        _, h = qlstm.forward_sequence(seq)
        hidden_states.append(h.copy())
        log_returns.append(float(all_features[i][0]))
    return hidden_states, log_returns


def _ry(theta: float) -> np.ndarray:
    """Rotation Y gate: [[cos(th/2), -sin(th/2)], [sin(th/2), cos(th/2)]]."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(phi: float) -> np.ndarray:
    """Rotation Z gate: [[e^{-i phi/2}, 0], [0, e^{i phi/2}]]."""
    c = math.cos(phi / 2)
    s = math.sin(phi / 2)
    return np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=complex)


_CNOT_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)


def _cnot() -> np.ndarray:
    """CNOT gate (4x4 matrix)."""
    return _CNOT_MATRIX


# ──────────────────────────────────────────────
# §0b  Rolling Z-Score Normalizer
# ──────────────────────────────────────────────

@dataclass
class RollingZScoreNormalizer:
    """Rolling Z-Score normalization over the last N bars.

    Replaces fixed min-max scaling to prevent look-ahead bias and
    ensure inputs remain stationary regardless of price level.
    z_i = (x_i − μ_N) / (σ_N + ε)
    """

    window: int = 60  # N bars for rolling stats
    eps: float = 1e-8

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize a 1-D feature vector using rolling stats from itself."""
        mu = float(np.mean(x))
        sigma = float(np.std(x))
        return (x - mu) / (sigma + self.eps)

    def normalize_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Normalize a (T, F) sequence per-feature using rolling window."""
        T, F = seq.shape
        out = np.zeros_like(seq)
        w = self.window
        for f in range(F):
            col = seq[:, f]
            # Vectorized rolling mean/std via cumsum
            cs = np.cumsum(col)
            cs2 = np.cumsum(col ** 2)
            for t in range(T):
                start = max(0, t - w + 1)
                n = t - start + 1
                s = cs[t] - (cs[start - 1] if start > 0 else 0.0)
                s2 = cs2[t] - (cs2[start - 1] if start > 0 else 0.0)
                mu = s / n
                var = s2 / n - mu * mu
                sigma = np.sqrt(max(var, 0.0))
                out[t, f] = (col[t] - mu) / (sigma + self.eps)
        return out


# ──────────────────────────────────────────────
# §0c  Regime-Conditioned Normalizer
# ──────────────────────────────────────────────

class RegimeConditionedNormalizer:
    """Routes each timestep through a low-vol or high-vol normalizer.

    Maintains two RollingZScoreNormalizer instances and selects based
    on whether the current ATR is above or below the running median.
    """

    def __init__(self, window: int = 60, atr_lookback: int = 100) -> None:
        self.norm_low = RollingZScoreNormalizer(window=window)
        self.norm_high = RollingZScoreNormalizer(window=window)
        self.atr_lookback = atr_lookback
        self._atr_buffer: List[float] = []

    def get_regime(self, atr_val: float) -> int:
        """Return 0 (low-vol) or 1 (high-vol) based on median ATR."""
        self._atr_buffer.append(atr_val)
        if len(self._atr_buffer) > self.atr_lookback:
            self._atr_buffer = self._atr_buffer[-self.atr_lookback:]
        median_atr = float(np.median(self._atr_buffer))
        return 1 if atr_val >= median_atr else 0

    def normalize_sequence(
        self, seq: np.ndarray, atr_vals: np.ndarray | None = None
    ) -> np.ndarray:
        """Normalize (T, F) sequence, routing per-timestep by regime.

        If atr_vals is None, falls back to single-regime normalization.
        """
        if atr_vals is None or len(atr_vals) != seq.shape[0]:
            return self.norm_low.normalize_sequence(seq)

        T, F = seq.shape
        out = np.zeros_like(seq)

        # Build per-regime subsequences
        for t in range(T):
            regime = self.get_regime(float(atr_vals[t]))
            norm = self.norm_high if regime == 1 else self.norm_low
            # Normalize single row using rolling stats up to t
            window_start = max(0, t - norm.window + 1)
            for f in range(F):
                col_slice = seq[window_start:t + 1, f]
                mu = float(np.mean(col_slice))
                sigma = float(np.std(col_slice))
                out[t, f] = (seq[t, f] - mu) / (sigma + norm.eps)
        return out


# ──────────────────────────────────────────────
# §1  Variational Quantum Circuit (VQC)
# ──────────────────────────────────────────────

@dataclass
class VQC:
    """4-8 qubit VQC with R_y angle encoding + data re-uploading.

    Architecture per gate (v2 — optimized for BTC Futures volatility):
        |0>^n  ->  R_y(x)  ->  [CNOT-ring + R_y(θ) + R_z(φ)] x L  ->  <Z>

    Enhancements over v1:
        - 3 layers (was 2) for richer function approximation
        - R_z rotation added per layer for full Bloch sphere coverage
        - Data re-uploading: x is re-encoded between layers to avoid
          barren plateaus and improve gradient signal in noisy markets
    """

    n_qubits: int = 4
    n_layers: int = 3
    seed: int = 42
    theta: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    phi: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]  # R_z params

    def __post_init__(self) -> None:
        if self.theta is None:
            rng = np.random.default_rng(self.seed)
            self.theta = rng.standard_normal((self.n_layers, self.n_qubits)) * 0.3
        if self.phi is None:
            rng = np.random.default_rng(self.seed + 1000)
            self.phi = rng.standard_normal((self.n_layers, self.n_qubits)) * 0.3

    # ── forward ──
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return Z-expectations on all qubits.  ``x`` scaled to [0, π]."""
        z = _minmax_to_pi(x[: self.n_qubits])
        state = self._angle_embed(z)
        for l in range(self.n_layers):
            state = self._entangle(state)
            for q in range(self.n_qubits):
                state = _apply_single(state, _ry(self.theta[l, q]), q, self.n_qubits)
                state = _apply_single(state, _rz(self.phi[l, q]), q, self.n_qubits)
            # Data re-uploading: re-encode input between layers
            if l < self.n_layers - 1:
                for q in range(self.n_qubits):
                    angle = float(z[q]) if q < len(z) else 0.0
                    state = _apply_single(state, _ry(angle * 0.5), q, self.n_qubits)
        return np.array(
            [_expectation_z(state, q, self.n_qubits) for q in range(self.n_qubits)]
        )

    # ── parameter-shift gradient ──
    def grad(self, x: np.ndarray, target_qubit: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """∇_{θ,φ} ⟨Z_target⟩  via parameter-shift rule. Returns (g_theta, g_phi)."""
        g_theta = np.zeros_like(self.theta)
        g_phi = np.zeros_like(self.phi)
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                saved = self.theta[l, q]
                self.theta[l, q] = saved + math.pi / 2
                e_plus = self.forward(x)[target_qubit]
                self.theta[l, q] = saved - math.pi / 2
                e_minus = self.forward(x)[target_qubit]
                self.theta[l, q] = saved
                g_theta[l, q] = 0.5 * (e_plus - e_minus)

                saved_p = self.phi[l, q]
                self.phi[l, q] = saved_p + math.pi / 2
                e_plus_p = self.forward(x)[target_qubit]
                self.phi[l, q] = saved_p - math.pi / 2
                e_minus_p = self.forward(x)[target_qubit]
                self.phi[l, q] = saved_p
                g_phi[l, q] = 0.5 * (e_plus_p - e_minus_p)
        return g_theta, g_phi

    def grad_full(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """∇_{θ,φ} ⟨Z⟩ for ALL qubits. Returns (g_theta, g_phi) shape (L, Q, Q)."""
        nq = self.n_qubits
        g_theta = np.zeros((self.n_layers, nq, nq))
        g_phi = np.zeros((self.n_layers, nq, nq))
        for l in range(self.n_layers):
            for q in range(nq):
                saved = self.theta[l, q]
                self.theta[l, q] = saved + math.pi / 2
                z_plus = self.forward(x)
                self.theta[l, q] = saved - math.pi / 2
                z_minus = self.forward(x)
                self.theta[l, q] = saved
                g_theta[l, q, :] = 0.5 * (z_plus - z_minus)

                saved_p = self.phi[l, q]
                self.phi[l, q] = saved_p + math.pi / 2
                z_plus = self.forward(x)
                self.phi[l, q] = saved_p - math.pi / 2
                z_minus = self.forward(x)
                self.phi[l, q] = saved_p
                g_phi[l, q, :] = 0.5 * (z_plus - z_minus)
        return g_theta, g_phi

    def input_grad(self, x: np.ndarray, eps: float = 0.01) -> np.ndarray:
        """∂z/∂x via central finite differences. Returns (n_qubits, n_qubits)."""
        n = min(len(x), self.n_qubits)
        dz_dx = np.zeros((self.n_qubits, n))
        for i in range(n):
            x_p = x.copy(); x_p[i] += eps
            x_m = x.copy(); x_m[i] -= eps
            dz_dx[:, i] = (self.forward(x_p) - self.forward(x_m)) / (2.0 * eps)
        return dz_dx

    # ── batch methods ──

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Batched forward: different inputs, same params.

        X : (B, n_qubits) raw inputs
        Returns: (B, n_qubits) Z-expectations
        """
        nq = self.n_qubits
        Z = _minmax_to_pi_batch(X[:, :nq])  # (B, nq)
        states = _angle_embed_batch(Z, nq)    # (B, 2^nq)

        for l in range(self.n_layers):
            states = _entangle_batch(states, nq)
            for q in range(nq):
                states = _apply_single_batch(states, _ry(self.theta[l, q]), q, nq)
                states = _apply_single_batch(states, _rz(self.phi[l, q]), q, nq)
            if l < self.n_layers - 1:
                for q in range(nq):
                    states = _apply_single_batch_variable(
                        states, Z[:, q] * 0.5, "ry", q, nq
                    )

        return np.column_stack([
            _expectation_z_batch(states, q, nq) for q in range(nq)
        ])  # (B, nq)

    def forward_multi_params(
        self, x: np.ndarray,
        theta_batch: np.ndarray, phi_batch: np.ndarray,
    ) -> np.ndarray:
        """Same input, different params per batch element.

        x           : (n_qubits,) single input (pre-scaled via _minmax_to_pi)
        theta_batch : (B, n_layers, n_qubits)
        phi_batch   : (B, n_layers, n_qubits)
        Returns     : (B, n_qubits) Z-expectations
        """
        nq = self.n_qubits
        z = _minmax_to_pi(x[:nq])
        B = theta_batch.shape[0]

        # Tile the initial embedded state
        state0 = self._angle_embed(z)  # (2^nq,)
        states = np.tile(state0, (B, 1))  # (B, 2^nq)

        for l in range(self.n_layers):
            states = _entangle_batch(states, nq)
            for q in range(nq):
                states = _apply_single_batch_variable(
                    states, theta_batch[:, l, q], "ry", q, nq
                )
                states = _apply_single_batch_variable(
                    states, phi_batch[:, l, q], "rz", q, nq
                )
            if l < self.n_layers - 1:
                for q in range(nq):
                    angle = float(z[q]) if q < len(z) else 0.0
                    states = _apply_single_batch(
                        states, _ry(angle * 0.5), q, nq
                    )

        return np.column_stack([
            _expectation_z_batch(states, q, nq) for q in range(nq)
        ])  # (B, nq)

    def grad_full_batched(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batched parameter-shift: all shifts in one forward_multi_params call.

        Returns (g_theta, g_phi) each shape (L, Q, Q) — same as grad_full().
        """
        L, Q = self.n_layers, self.n_qubits
        n_shifts = 2 * L * Q  # theta shifts + phi shifts
        B = 2 * n_shifts       # plus and minus for each

        theta_base = np.tile(self.theta, (B, 1, 1))  # (B, L, Q)
        phi_base = np.tile(self.phi, (B, 1, 1))

        shift = math.pi / 2
        idx = 0
        # theta shifts
        for l in range(L):
            for q in range(Q):
                theta_base[idx, l, q] += shift      # plus
                theta_base[idx + 1, l, q] -= shift   # minus
                idx += 2
        # phi shifts
        for l in range(L):
            for q in range(Q):
                phi_base[idx, l, q] += shift
                phi_base[idx + 1, l, q] -= shift
                idx += 2

        # Single batched forward
        z_all = self.forward_multi_params(x, theta_base, phi_base)  # (B, Q)

        # Extract gradients
        g_theta = np.zeros((L, Q, Q))
        g_phi = np.zeros((L, Q, Q))
        idx = 0
        for l in range(L):
            for q in range(Q):
                g_theta[l, q, :] = 0.5 * (z_all[idx] - z_all[idx + 1])
                idx += 2
        for l in range(L):
            for q in range(Q):
                g_phi[l, q, :] = 0.5 * (z_all[idx] - z_all[idx + 1])
                idx += 2

        return g_theta, g_phi

    def input_grad_batched(self, x: np.ndarray, eps: float = 0.01) -> np.ndarray:
        """Batched input gradient via central differences.

        Returns (n_qubits, n_qubits) — same as input_grad().
        """
        n = min(len(x), self.n_qubits)
        # Build shifted inputs: (2*n, n_qubits)
        X_shifted = np.tile(x[:self.n_qubits].astype(float), (2 * n, 1))
        for i in range(n):
            X_shifted[2 * i, i] += eps
            X_shifted[2 * i + 1, i] -= eps

        z_all = self.forward_batch(X_shifted)  # (2*n, n_qubits)

        dz_dx = np.zeros((self.n_qubits, n))
        for i in range(n):
            dz_dx[:, i] = (z_all[2 * i] - z_all[2 * i + 1]) / (2.0 * eps)
        return dz_dx

    # ── helpers ──
    def _angle_embed(self, z: np.ndarray) -> np.ndarray:
        state = np.zeros(2 ** self.n_qubits, dtype=complex)
        state[0] = 1.0
        for i in range(self.n_qubits):
            th = float(z[i]) if i < len(z) else 0.0
            state = _apply_single(state, _ry(th), i, self.n_qubits)
        return state

    def _entangle(self, state: np.ndarray) -> np.ndarray:
        for q in range(self.n_qubits - 1):
            state = _apply_two(state, _cnot(), q, q + 1, self.n_qubits)
        state = _apply_two(state, _cnot(), self.n_qubits - 1, 0, self.n_qubits)
        return state

    @property
    def n_params(self) -> int:
        return self.n_layers * self.n_qubits * 2  # theta + phi

    def flat_params(self) -> np.ndarray:
        return np.concatenate([self.theta.ravel(), self.phi.ravel()])

    def load_flat(self, flat: np.ndarray) -> None:
        half = self.n_layers * self.n_qubits
        self.theta = flat[:half].reshape(self.n_layers, self.n_qubits)
        if len(flat) > half:
            self.phi = flat[half:].reshape(self.n_layers, self.n_qubits)
        else:
            # Backward compat: old checkpoints only have theta
            self.phi = np.zeros((self.n_layers, self.n_qubits))


# ──────────────────────────────────────────────
# §2  Quantum LSTM (QLSTM)
# ──────────────────────────────────────────────

class QLSTM:
    """Quantum LSTM with VQC per gate (Forget / Input / Output / Cell).

    Each gate owns a separate VQC.  Input ``x_t`` is concatenated with
    the previous hidden state ``h_{t-1}`` and projected to qubit-space
    before being fed into the quantum circuits.

    Parameters
    ----------
    input_dim : int   – dimensionality of each timestep feature vector
    hidden_dim : int  – size of classical hidden state
    n_qubits : int    – qubits per VQC (4-8)
    seed : int        – reproducibility
    """

    def __init__(
        self,
        input_dim: int = 27,
        hidden_dim: int = 32,
        n_qubits: int = 4,
        seed: int = 42,
        zscore_window: int = 60,
        regime_norm: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.regime_norm = regime_norm

        rng = np.random.default_rng(seed)

        # Rolling Z-Score normalizer at input
        if regime_norm:
            self.z_norm = RegimeConditionedNormalizer(window=zscore_window)
        else:
            self.z_norm = RollingZScoreNormalizer(window=zscore_window)

        # Classical projection: concat(x_t, h_{t-1}) → qubit-dim
        self.combined_dim = input_dim + hidden_dim
        self.W_proj = rng.standard_normal((n_qubits, self.combined_dim)) * 0.1
        self.b_proj = np.zeros(n_qubits)

        # Four VQC sub-circuits
        self.vqc_forget = VQC(n_qubits, n_layers=2, seed=seed)
        self.vqc_input  = VQC(n_qubits, n_layers=2, seed=seed + 1)
        self.vqc_output = VQC(n_qubits, n_layers=2, seed=seed + 2)
        self.vqc_cell   = VQC(n_qubits, n_layers=2, seed=seed + 3)

        # Back-projection: qubit-dim → hidden_dim
        self.W_gate = rng.standard_normal((hidden_dim, n_qubits)) * 0.1

        # Readout from final hidden state
        self.W_out = rng.standard_normal(hidden_dim) * 0.1
        self.b_out = 0.0

        # Training state
        self._h: np.ndarray = np.zeros(hidden_dim)
        self._c: np.ndarray = np.zeros(hidden_dim)

    def reset_state(self) -> None:
        self._h = np.zeros(self.hidden_dim)
        self._c = np.zeros(self.hidden_dim)

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean()
        std = x.std()
        return (x - mu) / (std + 1e-8)

    def step(self, x_t: np.ndarray) -> np.ndarray:
        """One QLSTM timestep.  Returns new hidden state."""
        combined = np.concatenate([x_t[:self.input_dim], self._h])
        if len(combined) < self.combined_dim:
            combined = np.pad(combined, (0, self.combined_dim - len(combined)))
        combined = combined[: self.combined_dim]

        # Project to qubit space
        u = self._layer_norm(self.W_proj @ combined + self.b_proj)

        # VQC gate outputs  ([-1, 1]^n_qubits  →  hidden_dim)
        f_q = self.vqc_forget.forward(u)
        i_q = self.vqc_input.forward(u)
        o_q = self.vqc_output.forward(u)
        g_q = self.vqc_cell.forward(u)

        # Map back to hidden_dim and apply gate non-linearities
        f = np.array([_sigmoid(v) for v in (self.W_gate @ f_q)])
        i = np.array([_sigmoid(v) for v in (self.W_gate @ i_q)])
        o = np.array([_sigmoid(v) for v in (self.W_gate @ o_q)])
        g = np.tanh(self.W_gate @ g_q)

        self._c = f * self._c + i * g
        self._h = o * np.tanh(self._c)
        return self._h

    def forward_sequence(self, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """Process full sequence (T, F).  Returns (trend_reversal_prob, hidden_state).

        The output probability is NOT a price prediction — it is the
        Trend Reversal Probability P(reversal), enabling the model to
        express uncertainty about continuation vs. reversal.
        """
        self.reset_state()
        # Apply rolling z-score normalization to the full sequence
        if X.ndim == 2 and self.regime_norm and isinstance(self.z_norm, RegimeConditionedNormalizer):
            # Feature index 10 = log-ATR (from build_features)
            atr_col = X[:, 10] if X.shape[1] > 10 else None
            X_normed = self.z_norm.normalize_sequence(X, atr_vals=atr_col)
        elif X.ndim == 2:
            X_normed = self.z_norm.normalize_sequence(X)
        else:
            X_normed = X
        for t in range(X_normed.shape[0]):
            self.step(X_normed[t])
        logit = float(self.W_out @ self._h + self.b_out)
        # Trend reversal probability ∈ [0, 1]
        # 0 = strong trend continuation, 1 = high reversal likelihood
        reversal_prob = _sigmoid(logit)
        return reversal_prob, self._h.copy()

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            "W_proj": self.W_proj.copy(),
            "b_proj": self.b_proj.copy(),
            "vqc_forget": self.vqc_forget.flat_params(),
            "vqc_input": self.vqc_input.flat_params(),
            "vqc_output": self.vqc_output.flat_params(),
            "vqc_cell": self.vqc_cell.flat_params(),
            "W_gate": self.W_gate.copy(),
            "W_out": self.W_out.copy(),
            "b_out": np.array([self.b_out]),
        }

    def load_params(self, p: Dict[str, np.ndarray]) -> None:
        self.W_proj = p["W_proj"]
        self.b_proj = p["b_proj"]
        self.vqc_forget.load_flat(p["vqc_forget"])
        self.vqc_input.load_flat(p["vqc_input"])
        self.vqc_output.load_flat(p["vqc_output"])
        self.vqc_cell.load_flat(p["vqc_cell"])
        self.W_gate = p["W_gate"]
        self.W_out = p["W_out"]
        self.b_out = float(p["b_out"][0])


# ──────────────────────────────────────────────
# §3  Quantum A3C (QA3C)
# ──────────────────────────────────────────────

# Actions: 0=Hold, 1=Long, 2=Short
ACTION_HOLD  = 0
ACTION_LONG  = 1
ACTION_SHORT = 2
ACTION_NAMES = ["HOLD", "LONG", "SHORT"]


@dataclass
class QA3CConfig:
    gamma: float = 0.99           # discount
    lr_actor: float = 0.02        # learning rate (higher for VQC gradients)
    lr_critic: float = 0.02
    entropy_coeff: float = 0.005  # reduced: allow faster convergence
    n_qubits_actor: int = 4
    n_qubits_critic: int = 4
    n_workers: int = 4
    reward_sharpe_w: float = 0.3  # Sharpe weight in reward
    reward_mdd_pen: float = 0.5   # MDD penalty weight
    reward_vol_pen: float = 0.3   # Volatility penalty weight
    action_diversity_w: float = 0.1  # Penalty for action-bias
    seed: int = 42
    spsa_c: float = 0.05            # SPSA perturbation size
    spsa_samples: int = 2           # averaged SPSA samples per gradient
    # MLP actor (replaces VQC actor for higher capacity)
    use_mlp_actor: bool = False
    mlp_hidden: int = 128
    # GAE + shaped reward
    gae_lambda: float = 0.95
    lam_drawdown: float = 0.1   # softened further: small DD should not deter entries
    lam_cost: float = 0.1       # reduced further: C_ROUNDTRIP already in net_ret
    fee_roundtrip: float = 0.0011


class QA3CAgent:
    """Quantum Advantage Actor-Critic with async workers.

    Actor VQC outputs three Z-expectations → softmax → π(a|s).
    Critic VQC outputs single Z-expectation → V(s).
    """

    def __init__(self, state_dim: int, cfg: QA3CConfig | None = None) -> None:
        self.cfg = cfg or QA3CConfig()
        self.state_dim = state_dim
        self.use_mlp_actor = self.cfg.use_mlp_actor

        rng = np.random.default_rng(self.cfg.seed)

        if self.use_mlp_actor:
            # MLP Actor: state_dim → hidden → 3 action logits (4,611 params)
            h = self.cfg.mlp_hidden
            self.actor_W1 = rng.standard_normal((h, state_dim)) * np.sqrt(2.0 / state_dim)
            self.actor_b1 = np.zeros(h)
            self.actor_W2 = rng.standard_normal((3, h)) * np.sqrt(2.0 / h)
            self.actor_b2 = np.zeros(3)
            self.actor_proj = None
            self.actor_vqc = None
        else:
            # VQC Actor: state_dim → n_qubits → 3 action logits
            nqa = max(self.cfg.n_qubits_actor, 3)
            self.actor_proj = rng.standard_normal((nqa, state_dim)) * 0.1
            self.actor_vqc = VQC(nqa, n_layers=2, seed=self.cfg.seed)
            self.actor_W1 = None
            self.actor_b1 = None
            self.actor_W2 = None
            self.actor_b2 = None

        # Critic: state_dim → n_qubits → scalar value (always VQC)
        nqc = self.cfg.n_qubits_critic
        self.critic_proj = rng.standard_normal((nqc, state_dim)) * 0.1
        self.critic_vqc = VQC(nqc, n_layers=2, seed=self.cfg.seed + 10)

        # Lock for async param updates
        self._lock = threading.Lock()

        # Momentum buffers
        if self.use_mlp_actor:
            actor_n = self.actor_W1.size + self.actor_b1.size + self.actor_W2.size + self.actor_b2.size
        else:
            actor_n = self.actor_proj.size + self.actor_vqc.n_params
        critic_n = self.critic_proj.size + self.critic_vqc.n_params
        self._actor_momentum = np.zeros(actor_n)
        self._critic_momentum = np.zeros(critic_n)
        self._momentum_beta = 0.9

        # Training stats
        self.train_steps = 0
        self.last_loss: float = 0.0
        self.last_entropy: float = 0.0

    # ── MLP actor helpers ──
    def _mlp_actor_forward(self, state: np.ndarray) -> np.ndarray:
        """MLP actor: state → logits (3,)."""
        s = state[:self.state_dim]
        h = np.tanh(self.actor_W1 @ s + self.actor_b1)
        logits = self.actor_W2 @ h + self.actor_b2
        return logits

    def _mlp_actor_grad(
        self, state: np.ndarray, dL_dlogits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Backprop through MLP actor. Returns (dW1, db1, dW2, db2)."""
        s = state[:self.state_dim]
        # Forward pass (cache intermediates)
        pre_h = self.actor_W1 @ s + self.actor_b1
        h = np.tanh(pre_h)

        # Backward: dL/dW2, dL/db2
        dW2 = np.outer(dL_dlogits, h)  # (3, hidden)
        db2 = dL_dlogits.copy()  # (3,)

        # dL/dh = W2^T @ dL_dlogits
        dL_dh = self.actor_W2.T @ dL_dlogits  # (hidden,)

        # Through tanh: dL/dpre_h = dL/dh * (1 - h^2)
        dL_dpre = dL_dh * (1.0 - h ** 2)  # (hidden,)

        # dL/dW1, dL/db1
        dW1 = np.outer(dL_dpre, s)  # (hidden, state_dim)
        db1 = dL_dpre.copy()  # (hidden,)

        return dW1, db1, dW2, db2

    # ── inference ──
    def policy(
        self,
        state: np.ndarray,
        temperature: float = 0.7,
    ) -> Tuple[np.ndarray, int]:
        """Stochastic policy for training (exploration via temperature sampling).

        Output probabilities represent the Trend Reversal / Continuation
        belief, not raw price direction:
          HOLD  = uncertain / high reversal risk
          LONG  = bullish continuation / bearish reversal
          SHORT = bearish continuation / bullish reversal

        Parameters
        ----------
        temperature : float
            Softmax temperature. T<1 sharpens, T>1 flattens distribution.
            Default T=0.7 was set after overnight training to break Hold plateau.
        """
        if self.use_mlp_actor:
            logits = self._mlp_actor_forward(state)
        else:
            u = self.actor_proj @ state[:self.state_dim]
            z = self.actor_vqc.forward(u)
            logits = z[:3]
        # Temperature-scaled stochastic sampling (multinomial equivalent in numpy)
        probs = _softmax(logits / temperature)
        action = int(np.random.choice(3, p=probs))
        return probs, action

    def value(self, state: np.ndarray) -> float:
        """V(s) scalar."""
        u = self.critic_proj @ state[:self.state_dim]
        z = self.critic_vqc.forward(u)
        return float(z[0])

    def act_with_confidence(
        self,
        state: np.ndarray,
        confidence_threshold: float = 0.6,
    ) -> Tuple[np.ndarray, int]:
        """Confidence-threshold inference for validation/backtesting.

        Unlike `policy()` (stochastic training) or `act_greedy()` (pure argmax),
        this returns HOLD unless max(P_long, P_short) exceeds the threshold.
        This prevents the model from forcing low-confidence trades during eval.

        Parameters
        ----------
        confidence_threshold : float
            Min probability for Long/Short. If below, defaults to Hold.
            Tunable: 0.60 is a reasonable starting point.
        """
        if self.use_mlp_actor:
            logits = self._mlp_actor_forward(state)
            probs = _softmax(logits)
        else:
            u = self.actor_proj @ state[:self.state_dim]
            z = self.actor_vqc.forward(u)
            probs = _softmax(z[:3])

        p_long  = float(probs[ACTION_LONG])
        p_short = float(probs[ACTION_SHORT])

        if max(p_long, p_short) >= confidence_threshold:
            action = int(np.argmax(probs))
        else:
            action = ACTION_HOLD  # not confident enough — stay flat
        return probs, action

    def act_greedy(self, state: np.ndarray) -> Tuple[np.ndarray, int]:
        """Deterministic greedy policy (argmax). Use act_with_confidence() for eval."""
        if self.use_mlp_actor:
            logits = self._mlp_actor_forward(state)
            probs = _softmax(logits)
        else:
            u = self.actor_proj @ state[:self.state_dim]
            z = self.actor_vqc.forward(u)
            probs = _softmax(z[:3])
        return probs, int(np.argmax(probs))

    def get_actor_logits(self, state: np.ndarray) -> np.ndarray:
        """Return raw logits (3,) — useful for calibration and focal loss."""
        if self.use_mlp_actor:
            return self._mlp_actor_forward(state)
        else:
            u = self.actor_proj @ state[:self.state_dim]
            z = self.actor_vqc.forward(u)
            return z[:3].copy()

    # ── SPSA helpers ──
    def _get_actor_flat(self) -> np.ndarray:
        if self.use_mlp_actor:
            return np.concatenate([
                self.actor_W1.ravel(), self.actor_b1.ravel(),
                self.actor_W2.ravel(), self.actor_b2.ravel(),
            ])
        return np.concatenate([self.actor_proj.ravel(), self.actor_vqc.flat_params()])

    def _set_actor_flat(self, flat: np.ndarray) -> None:
        if self.use_mlp_actor:
            idx = 0
            n = self.actor_W1.size
            self.actor_W1 = flat[idx:idx + n].reshape(self.actor_W1.shape); idx += n
            n = self.actor_b1.size
            self.actor_b1 = flat[idx:idx + n]; idx += n
            n = self.actor_W2.size
            self.actor_W2 = flat[idx:idx + n].reshape(self.actor_W2.shape); idx += n
            self.actor_b2 = flat[idx:]
        else:
            n_proj = self.actor_proj.size
            self.actor_proj = flat[:n_proj].reshape(self.actor_proj.shape)
            self.actor_vqc.load_flat(flat[n_proj:])

    def _get_critic_flat(self) -> np.ndarray:
        return np.concatenate([self.critic_proj.ravel(), self.critic_vqc.flat_params()])

    def _set_critic_flat(self, flat: np.ndarray) -> None:
        n_proj = self.critic_proj.size
        self.critic_proj = flat[:n_proj].reshape(self.critic_proj.shape)
        self.critic_vqc.load_flat(flat[n_proj:])

    def _actor_loss_fn(self, s: np.ndarray, a: int, advantage: float) -> float:
        u = self.actor_proj @ s[:self.state_dim]
        z = self.actor_vqc.forward(u)
        probs = _softmax(z[:3])
        log_prob = math.log(probs[a] + 1e-10)
        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        return -log_prob * advantage - self.cfg.entropy_coeff * entropy

    def _critic_loss_fn(self, s: np.ndarray, target: float) -> float:
        u = self.critic_proj @ s[:self.state_dim]
        z = self.critic_vqc.forward(u)
        return 0.5 * (float(z[0]) - target) ** 2

    # ── async update (exact chain-rule gradients, parallel batch) ──
    def accumulate_grads(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        done: bool,
    ) -> float:
        """Update ALL params via exact gradients (param-shift + chain rule).

        Default path: _compute_batch_grads_both (vectorized numpy, no IPC).
        Fallback: per-timestep _compute_timestep_grads_both via ProcessPool.
        """
        T = len(states)
        if T == 0:
            return 0.0

        # Compute values for GAE
        values = [self.value(s) for s in states]
        bootstrap = 0.0 if done else self.value(states[-1])

        # GAE(lambda) advantages and returns
        gae_lam = self.cfg.gae_lambda
        gamma = self.cfg.gamma
        gae = 0.0
        advantages_gae: List[float] = [0.0] * T
        returns: List[float] = [0.0] * T
        for t in reversed(range(T)):
            next_val = bootstrap if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * gae_lam * gae
            advantages_gae[t] = gae
            returns[t] = gae + values[t]  # GAE + V = target return

        # Learning rate with decay
        decay = 5e-4
        lr_a = self.cfg.lr_actor / (1.0 + decay * self.train_steps)
        lr_c = self.cfg.lr_critic / (1.0 + decay * self.train_steps)
        ent_c = self.cfg.entropy_coeff
        grad_clip = 1.0

        with self._lock:
            # Normalize GAE advantages
            adv_arr = np.array(advantages_gae)
            adv_std = float(np.std(adv_arr)) + 1e-8
            adv_mean = float(np.mean(adv_arr))
            norm_advantages = (adv_arr - adv_mean) / adv_std

            # Snapshot current params
            critic_vqc_flat = self.critic_vqc.flat_params()
            critic_proj_copy = self.critic_proj.copy()
            if not self.use_mlp_actor:
                actor_vqc_flat = self.actor_vqc.flat_params()
                actor_proj_copy = self.actor_proj.copy()
            else:
                actor_vqc_flat = None
                actor_proj_copy = None

        states_arr = np.array(states)
        actions_arr = np.array(actions, dtype=int)
        returns_arr = np.array(returns)

        # ── MLP Actor path: direct numpy backprop ──
        if self.use_mlp_actor:
            sum_dW1 = np.zeros_like(self.actor_W1)
            sum_db1 = np.zeros_like(self.actor_b1)
            sum_dW2 = np.zeros_like(self.actor_W2)
            sum_db2 = np.zeros_like(self.actor_b2)
            total_actor_loss = 0.0
            total_entropy = 0.0

            for t in range(T):
                logits = self._mlp_actor_forward(states[t])
                probs = _softmax(logits)
                entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
                total_entropy += entropy
                total_actor_loss += (
                    -math.log(probs[actions[t]] + 1e-10) * norm_advantages[t]
                    - ent_c * entropy
                )
                # Analytic dL/dlogits
                dL_dlogits = np.zeros(3)
                for k in range(3):
                    one_hot = 1.0 if k == actions[t] else 0.0
                    dL_dlogits[k] = norm_advantages[t] * (probs[k] - one_hot)
                    dL_dlogits[k] += ent_c * probs[k] * (
                        math.log(probs[k] + 1e-10) + entropy
                    )
                dW1, db1, dW2, db2 = self._mlp_actor_grad(states[t], dL_dlogits)
                sum_dW1 += np.clip(dW1, -grad_clip, grad_clip)
                sum_db1 += np.clip(db1, -grad_clip, grad_clip)
                sum_dW2 += np.clip(dW2, -grad_clip, grad_clip)
                sum_db2 += np.clip(db2, -grad_clip, grad_clip)

            # Critic gradients via existing batch path
            critic_result = self._compute_critic_grads(
                states_arr, returns_arr, critic_proj_copy, critic_vqc_flat, grad_clip
            )

            with self._lock:
                self.actor_W1 -= lr_a * sum_dW1
                self.actor_b1 -= lr_a * sum_db1
                self.actor_W2 -= lr_a * sum_dW2
                self.actor_b2 -= lr_a * sum_db2
                self.critic_vqc.theta -= lr_c * critic_result["critic_dth"]
                self.critic_vqc.phi -= lr_c * critic_result["critic_dph"]
                self.critic_proj -= lr_c * critic_result["critic_dW"]
                self.train_steps += T
                self.last_loss = total_actor_loss / max(T, 1)
                self.last_entropy = total_entropy / max(T, 1)
            return self.last_loss

        # ── VQC Actor path (original) ──
        result = None
        # Priority 1: GPU autograd (torch)
        try:
            result = _compute_batch_grads_gpu(
                states=states_arr,
                actions=actions_arr,
                advantages=norm_advantages,
                target_returns=returns_arr,
                actor_proj=actor_proj_copy,
                actor_vqc_flat=actor_vqc_flat,
                critic_proj=critic_proj_copy,
                critic_vqc_flat=critic_vqc_flat,
                n_qubits_actor=self.actor_vqc.n_qubits,
                n_layers_actor=self.actor_vqc.n_layers,
                n_qubits_critic=self.critic_vqc.n_qubits,
                n_layers_critic=self.critic_vqc.n_layers,
                state_dim=self.state_dim,
                entropy_coeff=ent_c,
                grad_clip=grad_clip,
            )
        except Exception:
            result = None

        # Priority 2: batched vectorized numpy gradients
        if result is None:
            try:
                result = _compute_batch_grads_both(
                    states=states_arr,
                    actions=actions_arr,
                    advantages=norm_advantages,
                    target_returns=returns_arr,
                    actor_proj=actor_proj_copy,
                    actor_vqc_flat=actor_vqc_flat,
                    critic_proj=critic_proj_copy,
                    critic_vqc_flat=critic_vqc_flat,
                    n_qubits_actor=self.actor_vqc.n_qubits,
                    n_layers_actor=self.actor_vqc.n_layers,
                    n_qubits_critic=self.critic_vqc.n_qubits,
                    n_layers_critic=self.critic_vqc.n_layers,
                    state_dim=self.state_dim,
                    entropy_coeff=ent_c,
                    grad_clip=grad_clip,
                )
            except Exception:
                result = None

        # Priority 3: per-timestep (ProcessPool or sequential)
        if result is None:
            actor_proj_flat = actor_proj_copy.ravel()
            critic_proj_flat = critic_proj_copy.ravel()
            task_args = []
            for t in range(T):
                task_args.append((
                    actor_proj_flat, self.actor_proj.shape, actor_vqc_flat,
                    critic_proj_flat, self.critic_proj.shape, critic_vqc_flat,
                    self.actor_vqc.n_qubits, self.actor_vqc.n_layers,
                    self.critic_vqc.n_qubits, self.critic_vqc.n_layers,
                    states[t], actions[t], float(norm_advantages[t]), returns[t],
                    ent_c, self.state_dim, grad_clip,
                ))

            results_list = None
            if T >= 2:
                try:
                    pool = _get_grad_pool()
                    results_list = list(pool.map(_compute_timestep_grads_both, task_args))
                except Exception:
                    results_list = None

            if results_list is None:
                results_list = [_compute_timestep_grads_both(a) for a in task_args]

            result = {
                "actor_dth": np.zeros_like(self.actor_vqc.theta),
                "actor_dph": np.zeros_like(self.actor_vqc.phi),
                "actor_dW": np.zeros_like(self.actor_proj),
                "critic_dth": np.zeros_like(self.critic_vqc.theta),
                "critic_dph": np.zeros_like(self.critic_vqc.phi),
                "critic_dW": np.zeros_like(self.critic_proj),
                "actor_loss": 0.0,
                "entropy": 0.0,
            }
            for r in results_list:
                result["actor_dth"] += r["actor_dth"]
                result["actor_dph"] += r["actor_dph"]
                result["actor_dW"] += r["actor_dW"]
                result["critic_dth"] += r["critic_dth"]
                result["critic_dph"] += r["critic_dph"]
                result["critic_dW"] += r["critic_dW"]
                result["actor_loss"] += r["actor_loss"]
                result["entropy"] += r["entropy"]

        # Apply single batch update under lock
        with self._lock:
            self.actor_vqc.theta -= lr_a * result["actor_dth"]
            self.actor_vqc.phi -= lr_a * result["actor_dph"]
            self.actor_proj -= lr_a * result["actor_dW"]
            self.critic_vqc.theta -= lr_c * result["critic_dth"]
            self.critic_vqc.phi -= lr_c * result["critic_dph"]
            self.critic_proj -= lr_c * result["critic_dW"]

            self.train_steps += T
            self.last_loss = result["actor_loss"] / max(T, 1)
            self.last_entropy = result["entropy"] / max(T, 1)

        return self.last_loss

    def _compute_critic_grads(
        self,
        states_arr: np.ndarray,
        returns_arr: np.ndarray,
        critic_proj: np.ndarray,
        critic_vqc_flat: np.ndarray,
        grad_clip: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """Compute critic-only gradients (used when MLP actor handles its own)."""
        T = len(states_arr)
        c_vqc = VQC(n_qubits=self.critic_vqc.n_qubits, n_layers=self.critic_vqc.n_layers, seed=0)
        c_vqc.load_flat(critic_vqc_flat)
        nqc = c_vqc.n_qubits

        U_c = states_arr[:, :self.state_dim] @ critic_proj.T
        Z_c = c_vqc.forward_batch(U_c)

        dL_dz_c = np.zeros((T, nqc))
        dL_dz_c[:, 0] = Z_c[:, 0] - returns_arr

        sum_dth = np.zeros((c_vqc.n_layers, nqc))
        sum_dph = np.zeros((c_vqc.n_layers, nqc))
        sum_dW = np.zeros_like(critic_proj)

        eps = 0.01
        for t in range(T):
            g_th, g_ph = c_vqc.grad_full_batched(U_c[t])
            sum_dth += np.clip(np.einsum("lqk,k->lq", g_th, dL_dz_c[t]), -grad_clip, grad_clip)
            sum_dph += np.clip(np.einsum("lqk,k->lq", g_ph, dL_dz_c[t]), -grad_clip, grad_clip)

        n_input_c = min(U_c.shape[1], nqc)
        shift_inputs = []
        for t in range(T):
            for i in range(n_input_c):
                x_p = U_c[t].copy(); x_p[i] += eps
                x_m = U_c[t].copy(); x_m[i] -= eps
                shift_inputs.append(x_p)
                shift_inputs.append(x_m)
        shift_inputs = np.array(shift_inputs)

        MAX_BATCH = 512
        if len(shift_inputs) <= MAX_BATCH:
            z_shifted = c_vqc.forward_batch(shift_inputs)
        else:
            chunks = [shift_inputs[i:i + MAX_BATCH] for i in range(0, len(shift_inputs), MAX_BATCH)]
            z_shifted = np.vstack([c_vqc.forward_batch(c) for c in chunks])

        idx = 0
        for t in range(T):
            dz_du = np.zeros((nqc, n_input_c))
            for i in range(n_input_c):
                dz_du[:, i] = (z_shifted[idx] - z_shifted[idx + 1]) / (2.0 * eps)
                idx += 2
            dL_du = dL_dz_c[t] @ dz_du
            dL_dW = np.clip(np.outer(dL_du, states_arr[t, :self.state_dim]), -grad_clip, grad_clip)
            sum_dW += dL_dW

        return {"critic_dth": sum_dth, "critic_dph": sum_dph, "critic_dW": sum_dW}

    def supervised_update(
        self,
        states: List[np.ndarray],
        labels: List[int],
        class_weights: np.ndarray | None = None,
        lr: float = 1e-3,
        gamma: float = 2.0,
        batch_size: int = 128,
    ) -> float:
        """Supervised update using focal loss through MLP actor.

        Used in Phase 1 (warm-up) of walk-forward training.
        Only applicable when use_mlp_actor=True.
        Uses mini-batch SGD for efficient learning on large datasets.

        Parameters
        ----------
        states : list of (state_dim,) hidden states
        labels : list of int labels in {-1, 0, +1}
        class_weights : (3,) from compute_class_weights
        lr : learning rate
        gamma : focal loss exponent
        batch_size : mini-batch size (default 128)

        Returns
        -------
        Average loss over all samples
        """
        if not self.use_mlp_actor:
            return 0.0

        from src.models.focal_loss import focal_loss_gradient, focal_loss_value

        T = len(states)
        if T == 0:
            return 0.0

        # Shuffle indices for mini-batch SGD
        indices = np.random.permutation(T)
        total_loss = 0.0
        grad_clip = 1.0

        for batch_start in range(0, T, batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            B = len(batch_idx)

            sum_dW1 = np.zeros_like(self.actor_W1)
            sum_db1 = np.zeros_like(self.actor_b1)
            sum_dW2 = np.zeros_like(self.actor_W2)
            sum_db2 = np.zeros_like(self.actor_b2)

            for t in batch_idx:
                logits = self._mlp_actor_forward(states[t])
                total_loss += focal_loss_value(logits, labels[t], gamma=gamma, class_weights=class_weights)

                dL_dlogits = focal_loss_gradient(logits, labels[t], gamma=gamma, class_weights=class_weights)
                dW1, db1, dW2, db2 = self._mlp_actor_grad(states[t], dL_dlogits)

                sum_dW1 += np.clip(dW1, -grad_clip, grad_clip)
                sum_db1 += np.clip(db1, -grad_clip, grad_clip)
                sum_dW2 += np.clip(dW2, -grad_clip, grad_clip)
                sum_db2 += np.clip(db2, -grad_clip, grad_clip)

            with self._lock:
                self.actor_W1 -= lr * sum_dW1 / B
                self.actor_b1 -= lr * sum_db1 / B
                self.actor_W2 -= lr * sum_dW2 / B
                self.actor_b2 -= lr * sum_db2 / B

        self.train_steps += T

        return total_loss / T

    def get_global_params(self) -> Dict[str, np.ndarray]:
        with self._lock:
            params: Dict[str, np.ndarray] = {
                "critic_proj": self.critic_proj.copy(),
                "critic_vqc": self.critic_vqc.flat_params(),
            }
            if self.use_mlp_actor:
                params["actor_W1"] = self.actor_W1.copy()
                params["actor_b1"] = self.actor_b1.copy()
                params["actor_W2"] = self.actor_W2.copy()
                params["actor_b2"] = self.actor_b2.copy()
            else:
                params["actor_proj"] = self.actor_proj.copy()
                params["actor_vqc"] = self.actor_vqc.flat_params()
            return params

    def load_global_params(self, p: Dict[str, np.ndarray]) -> None:
        with self._lock:
            if self.use_mlp_actor and "actor_W1" in p:
                self.actor_W1 = p["actor_W1"]
                self.actor_b1 = p["actor_b1"]
                self.actor_W2 = p["actor_W2"]
                self.actor_b2 = p["actor_b2"]
            elif "actor_proj" in p:
                if self.actor_proj is not None:
                    self.actor_proj = p["actor_proj"]
                if self.actor_vqc is not None:
                    self.actor_vqc.load_flat(p["actor_vqc"])
            self.critic_proj = p["critic_proj"]
            self.critic_vqc.load_flat(p["critic_vqc"])
            # Reset momentum buffers
            if self.use_mlp_actor:
                actor_n = (self.actor_W1.size + self.actor_b1.size +
                           self.actor_W2.size + self.actor_b2.size)
            else:
                actor_n = self.actor_proj.size + self.actor_vqc.n_params
            self._actor_momentum = np.zeros(actor_n)
            self._critic_momentum = np.zeros(
                self.critic_proj.size + self.critic_vqc.n_params
            )


# ──────────────────────────────────────────────
# §3b  Reward Function (Anti-Drift Enhanced)
# ──────────────────────────────────────────────

def compute_reward(
    returns: List[float],
    mdd: float,
    sharpe_w: float = 0.3,
    mdd_pen: float = 0.5,
    vol_pen: float = 0.3,
    actions: List[int] | None = None,
    diversity_w: float = 0.1,
) -> float:
    """Multi-objective reward with anti-drift penalties.

    R = mean(r) + sharpe_w·Sharpe − mdd_pen·MDD − vol_pen·σ(r) − diversity_w·ActionBias

    Components
    ----------
    MDD penalty : penalises large drawdowns to prevent one-sided bets
    Volatility penalty : discourages high-variance strategies
    Action diversity : penalises always-long / always-short bias
    """
    if not returns:
        return 0.0
    r = np.array(returns)
    mean_r = float(r.mean())
    std_r = float(r.std())
    sharpe = (mean_r / std_r) if std_r > 1e-8 else 0.0

    # Volatility penalty: penalise high-σ strategies
    vol_penalty = vol_pen * std_r

    # Action diversity penalty: penalise if >70% of actions are the same
    action_bias_penalty = 0.0
    if actions and len(actions) > 2:
        from collections import Counter
        counts = Counter(actions)
        most_common_frac = counts.most_common(1)[0][1] / len(actions)
        # Penalty kicks in above 40% concentration (catches Hold bias earlier)
        if most_common_frac > 0.4:
            action_bias_penalty = diversity_w * (most_common_frac - 0.4)

    return mean_r + sharpe_w * sharpe - mdd_pen * mdd - vol_penalty - action_bias_penalty


# ──────────────────────────────────────────────
# §3c  Async Worker (multiprocessing + thread fallback)
# ──────────────────────────────────────────────

def _run_worker_episode(args: Tuple) -> Dict[str, Any]:
    """Process-safe episode runner.  Receives serialized params, returns gradients.

    Runs a full episode loop locally — no shared state — and returns
    accumulated gradient dicts for the parent to apply.
    """
    (
        global_params_dict,
        episode_data,       # (N, state_dim) numpy array
        log_returns,        # (N,) numpy array or None
        n_steps,
        state_dim,
        cfg_dict,           # serializable QA3CConfig fields
    ) = args

    cfg = QA3CConfig(**cfg_dict)
    agent = QA3CAgent(state_dim=state_dim, cfg=cfg)
    agent.load_global_params(global_params_dict)

    T = len(episode_data)
    if T < 5:
        return {"loss": 0.0, "grads": None}

    idx = 0
    episode_returns_list: List[float] = []
    episode_actions_list: List[int] = []
    episode_peak = 1.0
    episode_equity = 1.0
    episode_mdd = 0.0

    all_grads: List[Dict[str, Any]] = []

    while idx < T - 1:
        states, actions, rewards = [], [], []
        for _ in range(n_steps):
            if idx >= T - 1:
                break
            state = episode_data[idx]
            probs, action = agent.policy(state)

            next_idx = min(idx + 1, T - 1)
            if log_returns is not None:
                log_ret = float(log_returns[next_idx])
            else:
                log_ret = float(episode_data[next_idx][0])

            if action == ACTION_LONG:
                r = log_ret
            elif action == ACTION_SHORT:
                r = -log_ret
            else:
                r = -abs(log_ret) * 0.05

            episode_equity *= (1.0 + r)
            if episode_equity > episode_peak:
                episode_peak = episode_equity
            dd = (episode_peak - episode_equity) / (episode_peak + 1e-8)
            if dd > episode_mdd:
                episode_mdd = dd

            states.append(state)
            actions.append(action)
            rewards.append(r)
            episode_returns_list.append(r)
            episode_actions_list.append(action)
            idx += 1

        if rewards:
            seg_reward = compute_reward(
                rewards,
                mdd=episode_mdd,
                sharpe_w=cfg.reward_sharpe_w,
                mdd_pen=cfg.reward_mdd_pen,
                vol_pen=cfg.reward_vol_pen,
                actions=episode_actions_list,
                diversity_w=cfg.action_diversity_w,
            )
            bonus = seg_reward / max(len(rewards), 1)
            rewards = [r + bonus for r in rewards]

        done = idx >= T - 1

        # Compute gradients locally (batched) but don't apply them
        T_seg = len(states)
        if T_seg == 0:
            continue

        ret_list: List[float] = []
        R_val = 0.0 if done else agent.value(states[-1])
        for t in reversed(range(T_seg)):
            R_val = rewards[t] + cfg.gamma * R_val
            ret_list.insert(0, R_val)

        raw_adv = []
        for t in range(T_seg):
            V = agent.value(states[t])
            raw_adv.append(ret_list[t] - V)
        adv_arr = np.array(raw_adv)
        adv_std = float(np.std(adv_arr)) + 1e-8
        adv_mean = float(np.mean(adv_arr))
        norm_adv = (adv_arr - adv_mean) / adv_std

        seg_result = _compute_batch_grads_both(
            states=np.array(states),
            actions=np.array(actions, dtype=int),
            advantages=norm_adv,
            target_returns=np.array(ret_list),
            actor_proj=agent.actor_proj.copy(),
            actor_vqc_flat=agent.actor_vqc.flat_params(),
            critic_proj=agent.critic_proj.copy(),
            critic_vqc_flat=agent.critic_vqc.flat_params(),
            n_qubits_actor=agent.actor_vqc.n_qubits,
            n_layers_actor=agent.actor_vqc.n_layers,
            n_qubits_critic=agent.critic_vqc.n_qubits,
            n_layers_critic=agent.critic_vqc.n_layers,
            state_dim=state_dim,
            entropy_coeff=cfg.entropy_coeff,
        )
        all_grads.append(seg_result)

    # Sum all segment gradients
    if not all_grads:
        return {"loss": 0.0, "grads": None}

    combined = {k: all_grads[0][k] for k in all_grads[0]}
    for g in all_grads[1:]:
        for k in combined:
            combined[k] = combined[k] + g[k] if not isinstance(combined[k], float) else combined[k] + g[k]

    return {"loss": combined["actor_loss"], "grads": combined}


class A3CWorker(threading.Thread):
    """Single async worker that runs episodes on a local copy of the agent."""

    def __init__(
        self,
        worker_id: int,
        global_agent: QA3CAgent,
        episode_data: np.ndarray,
        n_steps: int = 20,
        log_returns: np.ndarray | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.global_agent = global_agent
        # Manual deepcopy to avoid pickling the threading.Lock
        self.local_agent = QA3CAgent(state_dim=global_agent.state_dim, cfg=global_agent.cfg)
        self.local_agent.load_global_params(global_agent.get_global_params())
        self.data = episode_data
        self.log_returns = log_returns
        self.n_steps = n_steps
        self.result_loss: float = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        T = len(self.data)
        if T < 5:
            return
        idx = 0
        # Track episode-level metrics for multi-objective reward
        episode_returns: List[float] = []
        episode_actions: List[int] = []
        episode_peak = 1.0
        episode_equity = 1.0
        episode_mdd = 0.0

        while idx < T - 1 and not self._stop.is_set():
            # Sync local ← global
            self.local_agent.load_global_params(self.global_agent.get_global_params())

            states, actions, rewards = [], [], []
            for _ in range(self.n_steps):
                if idx >= T - 1:
                    break
                state = self.data[idx]
                probs, action = self.local_agent.policy(state)

                # Log-return as reward basis (stationary)
                next_idx = min(idx + 1, T - 1)
                if self.log_returns is not None:
                    log_ret = float(self.log_returns[next_idx])
                else:
                    log_ret = float(self.data[next_idx][0])

                if action == ACTION_LONG:
                    r = log_ret
                elif action == ACTION_SHORT:
                    r = -log_ret
                else:
                    # Proportional hold cost — no penalty when flat market
                    r = -abs(log_ret) * 0.05

                # Track rolling equity for MDD penalty
                episode_equity *= (1.0 + r)
                if episode_equity > episode_peak:
                    episode_peak = episode_equity
                dd = (episode_peak - episode_equity) / (episode_peak + 1e-8)
                if dd > episode_mdd:
                    episode_mdd = dd

                states.append(state)
                actions.append(action)
                rewards.append(r)
                episode_returns.append(r)
                episode_actions.append(action)
                idx += 1

            # Apply multi-objective reward adjustment per n-step segment
            if rewards:
                seg_reward = compute_reward(
                    rewards,
                    mdd=episode_mdd,
                    sharpe_w=self.global_agent.cfg.reward_sharpe_w,
                    mdd_pen=self.global_agent.cfg.reward_mdd_pen,
                    vol_pen=self.global_agent.cfg.reward_vol_pen,
                    actions=episode_actions,
                    diversity_w=self.global_agent.cfg.action_diversity_w,
                )
                # Blend segment-level multi-obj reward back into step rewards
                bonus = seg_reward / max(len(rewards), 1)
                rewards = [r + bonus for r in rewards]

            done = idx >= T - 1
            try:
                loss = self.global_agent.accumulate_grads(states, actions, rewards, done)
                self.result_loss = loss
            except Exception as e:
                logger.error(f"Training step failed: {e}")
                self._stop.set()
                return

    def stop(self) -> None:
        self._stop.set()


# ──────────────────────────────────────────────
# §4  Schrödinger Trading Equation
# ──────────────────────────────────────────────

@dataclass
class SchrodingerFilter:
    """Physics-based noise filter modelling price as quantum wave-function.

    • ψ(x) = (α/π)^{1/4} exp(-α(x−μ)²/2)   (ground-state Gaussian)
    • Probability density |ψ(x)|² gives likelihood of price at position x
    • Support/resistance → quantum potential barriers
    • Uncertainty principle ΔxΔp ≥ ℏ/2  →  suppress entries in whipsaw zones

    Anti-Drift Enhancement (v2):
        z_score, psi_squared, and delta_p are computed on **log-returns**
        to ensure price-level invariance.  Support/resistance remain in
        raw price for display purposes.
    """

    lookback: int = 60
    hbar: float = 1.0         # reduced Planck constant analogue
    whipsaw_threshold: float = 0.5  # below this → suppress entry

    def compute(
        self, closes: np.ndarray, volumes: np.ndarray | None = None,
        ofi: float = 0.0,
    ) -> Dict[str, Any]:
        """Analyse price series and return quantum-inspired metrics.

        Parameters
        ----------
        ofi : float
            Order Flow Imbalance ∈ [-1, 1].  Drives the quantum potential
            V(r, t) in the market Hamiltonian.
        """
        n = min(len(closes), self.lookback)
        window = closes[-n:]

        # ── Log-return based stationarity (Anti-Drift v2) ──
        log_ret = _log_return_series(window)
        lr = log_ret[1:]  # drop first (always 0)
        if len(lr) == 0:
            lr = np.array([0.0])

        mu_lr = float(np.mean(lr))
        sigma_lr = float(np.std(lr))
        if sigma_lr < 1e-12:
            sigma_lr = 1e-6

        # z-score on log-returns (stationary, price-level invariant)
        z_score = (lr[-1] - mu_lr) / sigma_lr

        # Wave-function probability density on log-return z-score
        alpha_lr = 1.0 / (sigma_lr ** 2)
        psi_sq = float(
            math.sqrt(alpha_lr / math.pi)
            * math.exp(-alpha_lr * (lr[-1] - mu_lr) ** 2)
        )

        # Support / resistance remain in RAW price for display
        recent = window[-20:] if len(window) >= 20 else window
        support = float(np.min(recent))
        resistance = float(np.max(recent))
        sr_range = resistance - support
        if sr_range < 1e-8:
            sr_range = float(np.std(window)) if float(np.std(window)) > 1e-8 else 1.0

        # Position within quantum potential well (raw price)
        x = float(closes[-1])
        well_position = (x - support) / sr_range  # 0 = support, 1 = resistance

        # ΔxΔp uncertainty principle on log-returns
        delta_x = sigma_lr  # return uncertainty
        # Momentum = rate of change of log-returns (2nd derivative)
        if len(lr) >= 3:
            mom_lr = np.diff(lr[-min(20, len(lr)):])
            delta_p = float(np.std(mom_lr)) if len(mom_lr) > 1 else sigma_lr
        else:
            delta_p = sigma_lr

        uncertainty_product = delta_x * delta_p
        # Whipsaw detection: relative to expected log-return uncertainty scale
        # Expected ΔxΔp ~ sigma_lr² for normal market conditions
        # Whipsaw = uncertainty much lower than expected (compressed vol)
        expected_uncertainty = sigma_lr ** 2
        is_whipsaw = (
            uncertainty_product < self.whipsaw_threshold * expected_uncertainty
            and sigma_lr < 5e-4  # also require very low absolute volatility
        )

        # ── Quantum Potential V(r, t) from OFI ──
        # Hamiltonian: H = -ℏ²/(2 m_eff) ∂²ψ/∂r² + V(r, t)
        # Effective mass: inversely proportional to volatility (liquidity proxy)
        m_eff = 1.0 / (sigma_lr + 1e-8)
        # V(r,t): OFI creates attractive (negative) or repulsive (positive) wells
        # When OFI aligns with current momentum (sign(z_score)), potential is
        # attractive (price accelerates); when opposed, it's repulsive (mean-revert).
        momentum_alignment = ofi * float(np.sign(z_score)) if abs(z_score) > 0.1 else 0.0
        quantum_potential = -ofi * sigma_lr * (1.0 + abs(z_score)) - momentum_alignment * 0.1
        potential_gradient = ofi * sigma_lr  # force = -dV/dr

        # Tunnel probability through resistance/support barrier
        # Use log-return scale for barrier height, modulated by quantum potential
        log_sr = math.log(max(resistance, 1e-8)) - math.log(max(support, 1e-8))
        if log_sr > 1e-8:
            # V(r,t) lowers effective barrier when potential pushes toward barrier
            effective_barrier = log_sr / sigma_lr * (1.0 + abs(quantum_potential))
            tunnel_prob = float(math.exp(-2.0 * effective_barrier))
        else:
            tunnel_prob = 0.5

        return {
            "psi_squared": psi_sq,
            "z_score": z_score,
            "well_position": well_position,
            "support": support,
            "resistance": resistance,
            "delta_x": delta_x,
            "delta_p": delta_p,
            "uncertainty": uncertainty_product,
            "is_whipsaw": is_whipsaw,
            "tunnel_prob": tunnel_prob,
            "quantum_potential": quantum_potential,
            "potential_gradient": potential_gradient,
            "effective_mass": m_eff,
        }


# ──────────────────────────────────────────────
# §5  Modified Kelly Criterion
# ──────────────────────────────────────────────

@dataclass
class KellyRisk:
    """Uncertainty-adaptive Kelly position sizing.

    When Schrödinger ΔxΔp uncertainty is high, automatically scales
    down from Half-Kelly to Quarter-Kelly to reduce exposure during
    unpredictable market conditions.
    """

    kelly_fraction: float = 0.5      # Half-Kelly base
    min_rr_ratio: float = 2.5        # minimum reward:risk
    max_position_pct: float = 0.25
    default_sl_pct: float = 0.03
    default_tp_pct: float = 0.075    # 2.5× SL
    uncertainty_scale_threshold: float = 0.5  # above this → reduce Kelly

    def compute(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        equity: float,
        uncertainty: float = 0.0,
        confidence: float = 1.0,
    ) -> Dict[str, float]:
        """Compute position size as fraction of equity.

        Parameters
        ----------
        uncertainty : float
            Schrödinger ΔxΔp product (higher = more uncertain market)
        confidence : float
            Model confidence [0, 1] — scales position size
        """
        p = max(0.01, min(0.99, win_rate))
        b = avg_win / max(avg_loss, 1e-8)  # odds ratio

        # Enforce minimum RR ratio
        if b < self.min_rr_ratio:
            b = self.min_rr_ratio

        # Kelly formula: f* = (p*b - (1-p)) / b
        f_full = (p * b - (1.0 - p)) / b

        # Adaptive Kelly fraction based on uncertainty
        # High uncertainty → scale toward Quarter-Kelly
        adaptive_fraction = self.kelly_fraction
        if uncertainty > 0:
            # Normalize uncertainty to [0, 1] range relative to threshold
            unc_ratio = min(uncertainty / max(self.uncertainty_scale_threshold, 1e-12), 2.0)
            # Linearly interpolate: Half-Kelly → Quarter-Kelly as uncertainty rises
            adaptive_fraction = self.kelly_fraction * max(0.5, 1.0 - 0.5 * unc_ratio)

        # Also scale by confidence
        adaptive_fraction *= max(0.3, confidence)

        f_half = adaptive_fraction * max(0.0, f_full)

        # Cap
        f_final = min(f_half, self.max_position_pct)

        return {
            "kelly_full": max(0.0, f_full),
            "kelly_half": f_half,
            "position_pct": f_final,
            "position_usd": f_final * equity,
            "rr_ratio": b,
            "win_rate": p,
            "kelly_fraction_used": adaptive_fraction,
        }


# ──────────────────────────────────────────────
# §6  Feature Engineering (Log-Return Based)
# ──────────────────────────────────────────────

def _log_return_series(prices: np.ndarray) -> np.ndarray:
    """np.log(price).diff() — the core stationarity transform."""
    log_p = np.log(np.maximum(prices, 1e-8))
    return np.diff(log_p, prepend=log_p[0])


def build_time_features(df: pd.DataFrame) -> np.ndarray:
    """Cyclical hour encoding (2 features) from the last row's timestamp.

    Returns np.ndarray of shape (2,):
        [0] hour_sin  — sin(2π × hour_utc / 24)
        [1] hour_cos  — cos(2π × hour_utc / 24)

    Intentionally minimal: 180-day datasets are too short to learn
    session-level flags without data-snooping bias.  Only the smooth
    cyclical encoding is added so the model can sense the time-of-day
    rhythm without overfitting to specific windows.
    """
    import math
    for col in ("ts", "timestamp", "datetime", "date"):
        if col in df.columns:
            ts_raw = df[col].iloc[-1]
            break
    else:
        return np.zeros(2)

    try:
        ts = pd.Timestamp(ts_raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        h = ts.hour + ts.minute / 60.0
    except Exception:
        return np.zeros(2)

    return np.array([
        math.sin(2 * math.pi * h / 24.0),
        math.cos(2 * math.pi * h / 24.0),
    ], dtype=float)


def build_features(df: pd.DataFrame, n_features: int = 27) -> np.ndarray:
    """Build feature vector using log-returns for stationarity.

    ALL price-derived features use log-return or relative ratios,
    never raw prices.  This eliminates Price Drift (가격 표류) and
    ensures the model operates on stationary distributions.
    """
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    vol = df["volume"].astype(float).values if "volume" in df.columns else np.ones(len(close))

    n = len(close)

    # ── Core: log-returns ──
    log_ret = _log_return_series(close)
    log_ret_h = _log_return_series(high)
    log_ret_l = _log_return_series(low)

    feats: list[float] = []

    # 1. Recent log returns (5) — most important features
    lr5 = log_ret[-5:] if n >= 5 else np.pad(log_ret, (5 - len(log_ret), 0))
    feats.extend(lr5.tolist())

    # 2. SMA deviation in log-return space (3)
    for w in [7, 25, 99]:
        if n >= w:
            sma_lr = float(np.mean(log_ret[-w:]))
            feats.append(log_ret[-1] - sma_lr)
        else:
            feats.append(0.0)

    # 3. RSI (computed on log-returns for stationarity) (1)
    if n >= 15:
        deltas = log_ret[-14:]
        ups = float(np.mean(np.maximum(deltas, 0)))
        downs = float(np.mean(np.maximum(-deltas, 0)))
        rsi = ups / (ups + downs + 1e-8)
        feats.append(rsi - 0.5)  # center around 0
    else:
        feats.append(0.0)

    # 4. Bollinger position via log-return z-score (1)
    if n >= 20:
        lr_window = log_ret[-20:]
        mu_lr = float(np.mean(lr_window))
        std_lr = float(np.std(lr_window))
        feats.append((log_ret[-1] - mu_lr) / (2.0 * std_lr + 1e-8))
    else:
        feats.append(0.0)

    # 5. ATR as log-range (1)
    if n >= 14:
        log_range = np.log(np.maximum(high[-14:], 1e-8)) - np.log(np.maximum(low[-14:], 1e-8))
        feats.append(float(np.mean(log_range)))
    else:
        feats.append(0.0)

    # 6. Volume log-ratio (1)
    if n >= 20:
        log_vol = np.log(np.maximum(vol, 1e-8))
        feats.append(float(log_vol[-1] - np.mean(log_vol[-20:])))
    else:
        feats.append(0.0)

    # 7. High-Low log-range (intrabar volatility) (1)
    if n >= 1:
        feats.append(float(np.log(max(high[-1], 1e-8)) - np.log(max(low[-1], 1e-8))))
    else:
        feats.append(0.0)

    # 8. MACD in log-return domain (1)
    if n >= 26:
        ema12_lr = pd.Series(log_ret).ewm(span=12).mean().iloc[-1]
        ema26_lr = pd.Series(log_ret).ewm(span=26).mean().iloc[-1]
        feats.append(float(ema12_lr - ema26_lr))
    else:
        feats.append(0.0)

    # 9. Multi-period log-return momentum (5)
    for p in [3, 5, 10, 15, 20]:
        if n >= p + 1:
            feats.append(float(np.sum(log_ret[-p:])))
        else:
            feats.append(0.0)

    # 10. Stochastic %K on log-returns (1)
    if n >= 14:
        lr14 = log_ret[-14:]
        lo14 = float(np.min(lr14))
        hi14 = float(np.max(lr14))
        feats.append((log_ret[-1] - lo14) / (hi14 - lo14 + 1e-8) - 0.5)
    else:
        feats.append(0.0)

    # 11. Trend reversal indicators (5)
    # a) Acceleration (2nd derivative of log-return)
    if n >= 3:
        accel = log_ret[-1] - log_ret[-2]
        feats.append(float(accel))
    else:
        feats.append(0.0)

    # b) Sign-change frequency (how often direction flips in last 10 bars)
    if n >= 11:
        signs = np.sign(log_ret[-10:])
        sign_changes = float(np.sum(np.abs(np.diff(signs)) > 0)) / 9.0
        feats.append(sign_changes - 0.5)  # center
    else:
        feats.append(0.0)

    # c) Trend strength (mean / std of log-returns = local Sharpe)
    if n >= 10:
        lr10 = log_ret[-10:]
        local_sharpe = float(np.mean(lr10)) / (float(np.std(lr10)) + 1e-8)
        feats.append(np.clip(local_sharpe, -3.0, 3.0) / 3.0)
    else:
        feats.append(0.0)

    # 12. Time cyclical encoding (2) — hour sin/cos from timestamp column
    feats.extend(build_time_features(df).tolist())

    # 13. deep_candle — candle trade-density: log1p(turnover)  (1)
    #     Sourced from Bybit K-line r[6] (turnover field) in bybit_mainnet.py
    if "deep_candle" in df.columns:
        feats.append(float(df["deep_candle"].iloc[-1]))
    else:
        feats.append(0.0)

    # 14. liq_heatmap — liquidation intensity proxy: wick_length × volume  (1)
    #     Formula: max(high − max(open, close), min(open, close) − low) × volume
    if "liq_heatmap" in df.columns:
        lh = float(df["liq_heatmap"].iloc[-1])
        # log-scale to prevent magnitude explosion (same philosophy as volume log-ratio)
        feats.append(float(np.log1p(max(lh, 0.0))))
    else:
        feats.append(0.0)

    # Pad / truncate to n_features
    feat_arr = np.array(feats[:n_features], dtype=float)
    if len(feat_arr) < n_features:
        feat_arr = np.pad(feat_arr, (0, n_features - len(feat_arr)))

    # NOTE: No global min-max normalization here.
    # Rolling Z-Score is applied at the QLSTM input layer instead,
    # preserving temporal locality of the normalization.
    return feat_arr


def build_sequences(
    df: pd.DataFrame, seq_len: int = 20, n_features: int = 25
) -> np.ndarray:
    """Build (seq_len, n_features) sequence from last seq_len bars."""
    if len(df) < seq_len:
        # Pad
        seqs = []
        for i in range(len(df)):
            seqs.append(build_features(df.iloc[: i + 1], n_features))
        while len(seqs) < seq_len:
            seqs.insert(0, np.zeros(n_features))
        return np.array(seqs)

    seqs = []
    for i in range(seq_len):
        end = len(df) - seq_len + i + 1
        seqs.append(build_features(df.iloc[:end], n_features))
    return np.array(seqs)


# ──────────────────────────────────────────────
# §7  Integrated Pipeline
# ──────────────────────────────────────────────

@dataclass
class QuantumSignal:
    """Output signal from the quantum pipeline."""
    action: int                # 0=HOLD, 1=LONG, 2=SHORT
    action_name: str
    confidence: float          # max policy prob
    probabilities: np.ndarray  # [p_hold, p_long, p_short]
    trend_reversal_prob: float # P(reversal) from QLSTM
    expected_return: float
    position_pct: float        # Kelly sizing
    kelly_info: Dict[str, float]
    schrodinger: Dict[str, Any]
    qlstm_hidden: np.ndarray
    value_est: float           # critic V(s)
    is_whipsaw: bool


class QuantumTradingPipeline:
    """End-to-end pipeline: OHLCV → QLSTM → QA3C → signal."""

    DEFAULT_CHECKPOINT = "data/quantum_checkpoint.npz"

    def __init__(
        self,
        n_features: int = 27,
        seq_len: int = 20,
        n_qubits: int = 4,
        hidden_dim: int = 32,
        seed: int = 42,
        auto_train_interval: int = 50,
    ) -> None:
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim

        # Components
        self.qlstm = QLSTM(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            n_qubits=n_qubits,
            seed=seed,
        )
        self.agent = QA3CAgent(
            state_dim=hidden_dim,
            cfg=QA3CConfig(n_qubits_actor=max(n_qubits, 3), n_qubits_critic=n_qubits, seed=seed),
        )
        self.schrodinger = SchrodingerFilter()
        self.kelly = KellyRisk()

        # Performance tracking
        self._trade_history: List[Dict[str, Any]] = []
        self._equity_curve: List[float] = [10000.0]
        self._max_equity: float = 10000.0

        # Training state
        self.is_training = False
        self.training_status = "Idle"
        self.stop_reason = ""
        self._workers: List[A3CWorker] = []
        self.train_epoch: int = 0
        self.train_epoch_total: int = 0
        self.train_loss: float = 0.0
        self.training_progress: float = 0.0  # 0.0 ~ 1.0
        self.training_eta: str = ""  # formatted ETA string
        self._train_phase_start: float = 0.0

        # Online learning / checkpoint
        self.auto_train_interval = auto_train_interval  # candles between auto-train
        self._candles_since_train: int = 0
        self._last_checkpoint_ts: str = ""

    def predict(
        self,
        df: pd.DataFrame,
        equity: float = 10000.0,
        greedy: bool = True,
        ofi: float = 0.0,
        min_confidence: float = 0.0,
        force_trade: bool = False,
    ) -> QuantumSignal:
        """Generate trading signal from OHLCV data.

        Parameters
        ----------
        ofi : float
            Order Flow Imbalance in [-1, 1].  Passed to SchrodingerFilter
            to compute the quantum potential V(r, t).
        min_confidence : float
            Minimum confidence (max probability) to allow a non-HOLD action.
            If the model's max(probs) < min_confidence and action != HOLD,
            the action is overridden to HOLD at the model level.
        force_trade : bool
            If True, HOLD is never returned.  When the model would output
            HOLD, it picks LONG or SHORT based on whichever has higher
            probability.  This ensures every tick produces a directional
            trade signal.
        """
        # 1. Build sequence features (log-return based)
        seq = build_sequences(df, self.seq_len, self.n_features)

        # 2. QLSTM forward pass → trend reversal probability
        reversal_prob, hidden = self.qlstm.forward_sequence(seq)

        # 3. QA3C decision (outputs trend reversal probabilities)
        if greedy:
            probs, action = self.agent.act_greedy(hidden)
        else:
            probs, action = self.agent.policy(hidden)

        # 3b. Modulate action based on reversal probability
        # High reversal prob → prefer HOLD or contra-trend
        if reversal_prob > 0.7 and action != ACTION_HOLD:
            # High reversal confidence → reduce position
            probs[0] += 0.2  # boost HOLD
            probs = probs / (probs.sum() + 1e-10)
            if reversal_prob > 0.85:
                action = ACTION_HOLD

        # 4. Schrödinger filter (with quantum potential from OFI)
        closes = df["close"].astype(float).values
        volumes = df["volume"].astype(float).values if "volume" in df.columns else None
        schro = self.schrodinger.compute(closes, volumes, ofi=ofi)

        # 5. Override action if whipsaw detected
        is_whipsaw = schro["is_whipsaw"]
        if is_whipsaw and action != ACTION_HOLD:
            action = ACTION_HOLD
            probs = np.array([0.8, 0.1, 0.1])

        # 5b. Model-level confidence gate
        if min_confidence > 0 and action != ACTION_HOLD:
            if float(np.max(probs)) < min_confidence:
                action = ACTION_HOLD

        # 5c. Force-trade: eliminate HOLD → pick best directional action
        if force_trade and action == ACTION_HOLD:
            if probs[ACTION_LONG] >= probs[ACTION_SHORT]:
                action = ACTION_LONG
            else:
                action = ACTION_SHORT

        # 6. Kelly position sizing (uncertainty-adaptive)
        win_rate = self._compute_win_rate()
        avg_win, avg_loss = self._compute_avg_win_loss()
        kelly_info = self.kelly.compute(
            win_rate, avg_win, avg_loss, equity,
            uncertainty=schro.get("uncertainty", 0.0),
            confidence=float(np.max(probs)),
        )

        # 7. Value estimate
        value_est = self.agent.value(hidden)

        # 8. Expected return estimate
        expected_return = (probs[1] - probs[2]) * kelly_info.get("rr_ratio", 2.5) * 0.01

        confidence = float(np.max(probs))

        return QuantumSignal(
            action=action,
            action_name=ACTION_NAMES[action],
            confidence=confidence,
            probabilities=probs,
            trend_reversal_prob=reversal_prob,
            expected_return=expected_return,
            position_pct=kelly_info["position_pct"],
            kelly_info=kelly_info,
            schrodinger=schro,
            qlstm_hidden=hidden,
            value_est=value_est,
            is_whipsaw=is_whipsaw,
        )

    def record_trade(self, side: str, pnl: float, equity: float) -> None:
        """Record a trade for Kelly estimation."""
        self._trade_history.append({"side": side, "pnl": pnl})
        self._equity_curve.append(equity)
        if equity > self._max_equity:
            self._max_equity = equity

    def _compute_win_rate(self) -> float:
        if not self._trade_history:
            return 0.5
        wins = sum(1 for t in self._trade_history if t["pnl"] > 0)
        return wins / len(self._trade_history)

    def _compute_avg_win_loss(self) -> Tuple[float, float]:
        wins = [t["pnl"] for t in self._trade_history if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in self._trade_history if t["pnl"] <= 0]
        avg_win = float(np.mean(wins)) if wins else 0.01
        avg_loss = float(np.mean(losses)) if losses else 0.01
        return avg_win, avg_loss

    def compute_mdd(self) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        arr = np.array(self._equity_curve)
        peak = np.maximum.accumulate(arr)
        dd = (peak - arr) / (peak + 1e-8)
        return float(np.max(dd))

    # ── Training ──

    @staticmethod
    def _sample_windows(
        df: pd.DataFrame, n_windows: int, window_size: int
    ) -> List[pd.DataFrame]:
        """Sample evenly-spaced windows from a large DataFrame."""
        n = len(df)
        if n <= window_size:
            return [df.reset_index(drop=True)]
        max_start = n - window_size
        step = max(1, max_start // n_windows)
        starts = list(range(0, max_start + 1, step))[:n_windows]
        return [df.iloc[s : s + window_size].reset_index(drop=True) for s in starts]

    @staticmethod
    def _precompute_features(
        window_df: pd.DataFrame, n_features: int
    ) -> np.ndarray:
        """Pre-compute features for all bars in a window (single pass)."""
        feats = []
        for i in range(len(window_df)):
            feats.append(build_features(window_df.iloc[: i + 1], n_features))
        return np.array(feats)

    def _build_qlstm_states(
        self, all_features: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Build QLSTM hidden states from pre-computed features."""
        hidden_states: List[np.ndarray] = []
        log_returns: List[float] = []
        n_bars = len(all_features)
        for i in range(self.seq_len, n_bars):
            seq = all_features[i - self.seq_len + 1 : i + 1]
            _, h = self.qlstm.forward_sequence(seq)
            hidden_states.append(h.copy())
            log_returns.append(float(all_features[i][0]))
        return hidden_states, log_returns

    @staticmethod
    def _format_eta(elapsed: float, progress: float) -> str:
        """Format ETA from elapsed time and progress (0-1)."""
        if progress <= 0.001:
            return "calculating..."
        remaining = elapsed / progress * (1.0 - progress)
        if remaining < 60:
            return f"{remaining:.0f}s"
        if remaining < 3600:
            return f"{int(remaining // 60)}m {int(remaining % 60)}s"
        return f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}m"

    def _update_progress(self, progress: float, phase: str, detail: str = "") -> None:
        """Update training progress bar and ETA."""
        self.training_progress = max(0.0, min(1.0, progress))
        elapsed = time.time() - self._train_phase_start
        self.training_eta = self._format_eta(elapsed, self.training_progress)
        bar_w = 15
        filled = int(bar_w * self.training_progress)
        bar = "#" * filled + "-" * (bar_w - filled)
        pct = int(self.training_progress * 100)
        self.training_status = f"{phase} [{bar}] {pct}% {detail}| ETA: {self.training_eta}"

    def start_training(
        self,
        df: pd.DataFrame,
        n_epochs: int = 5,
        fetch_history_fn: Any = None,
        n_windows: int = 10,
        window_size: int = 150,
    ) -> bool:
        """Launch async A3C training workers.

        Parameters
        ----------
        df : recent OHLCV data (fallback if no historical fetch)
        fetch_history_fn : callable() -> pd.DataFrame for bulk historical data
        n_windows : number of windows to sample from historical data
        window_size : bars per window
        """
        if self.is_training:
            return False

        if len(df) <= self.seq_len:
            self.stop_reason = f"Data too short ({len(df)} <= {self.seq_len})"
            logger.warning(self.stop_reason)
            return False

        self.is_training = True
        self.training_status = "Preparing..."
        self.stop_reason = ""
        self.train_epoch = 0
        self.train_epoch_total = n_epochs
        self.train_loss = 0.0
        self.training_progress = 0.0
        self.training_eta = ""

        df_snapshot = df.copy()

        def _train_thread_target():
            try:
                # ── Phase 1: Get training data ──
                train_df = df_snapshot
                if fetch_history_fn is not None:
                    self.training_status = "Fetching history..."
                    try:
                        hist_df = fetch_history_fn()
                        if hist_df is not None and len(hist_df) > window_size:
                            train_df = hist_df
                            logger.info(f"Using {len(train_df):,} bars of historical data")
                    except Exception as e:
                        logger.warning(f"History fetch failed, using live data: {e}")

                # ── Phase 2: Sample windows ──
                n_total_bars = len(train_df)
                if n_total_bars > window_size * 2:
                    windows = self._sample_windows(train_df, n_windows, window_size)
                    logger.info(
                        f"Sampled {len(windows)} windows of {window_size} bars "
                        f"from {n_total_bars:,} total bars"
                    )
                else:
                    if n_total_bars > 300:
                        train_df = train_df.iloc[-300:].reset_index(drop=True)
                    windows = [train_df]
                    logger.info(f"Using single window of {len(train_df)} bars")

                # ── Phase 3: Build features + QLSTM states per window ──
                self._train_phase_start = time.time()
                total_windows = len(windows)
                all_hidden: List[np.ndarray] = []
                all_lr: List[float] = []

                # Parallel window preprocessing via ProcessPoolExecutor
                self._update_progress(0.0, "Preparing", f"0/{total_windows} ")
                qlstm_params = self.qlstm.get_params()
                window_tasks = []
                for window_df in windows:
                    # Convert DataFrame to dict for cross-process pickling
                    window_data = {
                        col: window_df[col].values
                        for col in window_df.columns
                    }
                    window_tasks.append((
                        window_data, qlstm_params,
                        self.n_features, self.seq_len,
                        self.qlstm.input_dim, self.qlstm.hidden_dim,
                        self.qlstm.n_qubits, self.qlstm.z_norm.window,
                    ))

                window_results = None
                if total_windows >= 2:
                    try:
                        pool = _get_grad_pool()
                        window_results = list(pool.map(
                            _precompute_window_task, window_tasks
                        ))
                    except Exception as e:
                        logger.warning(f"Parallel window prep failed, falling back: {e}")
                        window_results = None

                if window_results is None:
                    # Sequential fallback
                    window_results = []
                    for w_idx, task in enumerate(window_tasks):
                        if not self.is_training:
                            break
                        self._update_progress(
                            w_idx / total_windows,
                            "Preparing",
                            f"{w_idx + 1}/{total_windows} ",
                        )
                        window_results.append(_precompute_window_task(task))

                for w_idx, (h_states, lr_vals) in enumerate(window_results):
                    all_hidden.extend(h_states)
                    all_lr.extend(lr_vals)
                    self._update_progress(
                        (w_idx + 1.0) / total_windows,
                        "Preparing",
                        f"{w_idx + 1}/{total_windows} ",
                    )
                    logger.info(
                        f"Window {w_idx + 1}/{total_windows}: "
                        f"{len(h_states)} states (total: {len(all_hidden)})"
                    )

                if len(all_hidden) < 2:
                    self.stop_reason = f"Insufficient data ({len(all_hidden)} < 2)"
                    logger.warning(self.stop_reason)
                    self.is_training = False
                    self.training_status = "Idle"
                    return

                episode_hidden = np.array(all_hidden)
                episode_lr = np.array(all_lr)
                logger.info(
                    f"Starting A3C training: {len(episode_hidden)} samples, "
                    f"{n_epochs} epochs"
                )

                # ── Phase 4: A3C training (multiprocessing workers) ──
                self._train_phase_start = time.time()

                # Serialize QA3CConfig for process-safe workers
                cfg_dict = {
                    f.name: getattr(self.agent.cfg, f.name)
                    for f in self.agent.cfg.__dataclass_fields__.values()
                }

                for epoch in range(n_epochs):
                    if not self.is_training:
                        break

                    n_samples = len(episode_hidden)
                    n_workers = min(self.agent.cfg.n_workers, n_samples)
                    if n_workers < 1:
                        n_workers = 1

                    chunk = n_samples // n_workers
                    global_params = self.agent.get_global_params()

                    # Build worker args
                    worker_args = []
                    for w_id in range(n_workers):
                        start = w_id * chunk
                        end = n_samples if w_id == n_workers - 1 else start + chunk
                        if end <= start:
                            continue
                        worker_args.append((
                            global_params,
                            episode_hidden[start:end],
                            episode_lr[start:end],
                            20,  # n_steps
                            self.agent.state_dim,
                            cfg_dict,
                        ))

                    # Dispatch via ProcessPoolExecutor for true parallelism
                    worker_results = None
                    if len(worker_args) >= 2:
                        try:
                            pool = _get_grad_pool()
                            worker_results = list(pool.map(
                                _run_worker_episode, worker_args
                            ))
                        except Exception as e:
                            logger.warning(f"Multiprocessing workers failed: {e}")
                            worker_results = None

                    if worker_results is None:
                        # Fallback: thread-based workers (backward compat)
                        workers = []
                        for w_id, wargs in enumerate(worker_args):
                            worker = A3CWorker(
                                worker_id=w_id,
                                global_agent=self.agent,
                                episode_data=wargs[1],
                                n_steps=20,
                                log_returns=wargs[2],
                            )
                            workers.append(worker)
                            worker.start()
                        for w in workers:
                            w.join(timeout=60)
                        self._workers = workers
                    else:
                        # Apply collected gradients from process workers
                        decay = 5e-4
                        lr_a = self.agent.cfg.lr_actor / (
                            1.0 + decay * self.agent.train_steps
                        )
                        lr_c = self.agent.cfg.lr_critic / (
                            1.0 + decay * self.agent.train_steps
                        )
                        total_loss = 0.0
                        total_ent = 0.0
                        total_T = 0
                        with self.agent._lock:
                            for wr in worker_results:
                                g = wr.get("grads")
                                if g is None:
                                    continue
                                self.agent.actor_vqc.theta -= lr_a * g["actor_dth"]
                                self.agent.actor_vqc.phi -= lr_a * g["actor_dph"]
                                self.agent.actor_proj -= lr_a * g["actor_dW"]
                                self.agent.critic_vqc.theta -= lr_c * g["critic_dth"]
                                self.agent.critic_vqc.phi -= lr_c * g["critic_dph"]
                                self.agent.critic_proj -= lr_c * g["critic_dW"]
                                total_loss += g["actor_loss"]
                                total_ent += g["entropy"]
                                total_T += 1
                            if total_T > 0:
                                self.agent.train_steps += n_samples
                                self.agent.last_loss = total_loss / max(total_T, 1)
                                self.agent.last_entropy = total_ent / max(total_T, 1)

                    self.train_epoch = epoch + 1
                    self.train_loss = self.agent.last_loss

                    self._update_progress(
                        (epoch + 1) / n_epochs,
                        "Training",
                        f"Ep {epoch + 1}/{n_epochs} ",
                    )

                self.training_status = "Completed"
                self.training_progress = 1.0
                self.training_eta = ""
                self.stop_reason = "Done"
            except Exception as e:
                self.stop_reason = f"Error: {str(e)}"
                logger.error(f"Training loop failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.training_status = "Error"
            finally:
                self.is_training = False

        t = threading.Thread(target=_train_thread_target, daemon=True)
        t.start()
        return True

    def stop_training(self) -> None:
        self.is_training = False
        for w in self._workers:
            w.stop()

    # ── Checkpoint ──
    def save_checkpoint(self, path: str | None = None) -> str:
        """Save all learned parameters + trade history to .npz."""
        import os
        path = path or self.DEFAULT_CHECKPOINT
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        qlstm_p = self.qlstm.get_params()
        qa3c_p = self.agent.get_global_params()

        save_dict: Dict[str, Any] = {}
        for k, v in qlstm_p.items():
            save_dict[f"qlstm_{k}"] = v
        for k, v in qa3c_p.items():
            save_dict[f"qa3c_{k}"] = v

        # Trade history as structured arrays
        if self._trade_history:
            save_dict["trade_pnls"] = np.array([t["pnl"] for t in self._trade_history])
            save_dict["trade_sides"] = np.array([t["side"] for t in self._trade_history])
        save_dict["equity_curve"] = np.array(self._equity_curve)
        save_dict["train_epoch"] = np.array([self.train_epoch])
        save_dict["train_steps"] = np.array([self.agent.train_steps])

        np.savez_compressed(path, **save_dict)
        from datetime import datetime, timezone, timedelta
        self._last_checkpoint_ts = datetime.now(
            timezone(timedelta(hours=9))
        ).strftime("%H:%M:%S")
        return path

    def load_checkpoint(self, path: str | None = None) -> bool:
        """Load parameters from .npz checkpoint. Returns True if loaded."""
        import os
        path = path or self.DEFAULT_CHECKPOINT
        npz_path = path if path.endswith(".npz") else path + ".npz"
        if not os.path.exists(npz_path):
            return False
        try:
            data = np.load(npz_path, allow_pickle=False)

            # ── Restore QLSTM params ──
            qlstm_p: Dict[str, np.ndarray] = {}
            for k in ["W_proj", "b_proj", "vqc_forget", "vqc_input",
                      "vqc_output", "vqc_cell", "W_gate", "W_out", "b_out"]:
                key = f"qlstm_{k}"
                if key in data:
                    qlstm_p[k] = data[key]

            # ── Auto-migrate W_proj if n_features increased (e.g. 25→27) ──
            # This lets old checkpoints be loaded without --fresh when in_features grows.
            if "W_proj" in qlstm_p:
                W = qlstm_p["W_proj"]
                expected_cols = self.qlstm.input_dim + self.qlstm.hidden_dim
                if W.shape[1] != expected_cols:
                    if W.shape[1] < expected_cols:
                        deficit = expected_cols - W.shape[1]
                        old_input_cols = W.shape[1] - self.qlstm.hidden_dim
                        padding = np.zeros((W.shape[0], deficit), dtype=W.dtype)
                        qlstm_p["W_proj"] = np.hstack([
                            W[:, :old_input_cols], padding, W[:, old_input_cols:]
                        ])
                        print(f"  [CKPT] Auto-migrated W_proj (grow): {W.shape} → "
                              f"{qlstm_p['W_proj'].shape} ({deficit} new-feature cols zero-padded)")
                    else:
                        # Truncate if somehow larger (e.g. loading 27-feat into 25-feat)
                        qlstm_p["W_proj"] = W[:, :expected_cols]
                        print(f"  [CKPT] Auto-migrated W_proj (truncate): {W.shape} → "
                              f"{qlstm_p['W_proj'].shape}")
            if qlstm_p:
                self.qlstm.load_params(qlstm_p)

            # ── Restore QA3C params (VQC actor path) ──
            qa3c_p: Dict[str, np.ndarray] = {}
            for k in ["actor_proj", "actor_vqc", "critic_proj", "critic_vqc"]:
                key = f"qa3c_{k}"
                if key in data:
                    qa3c_p[k] = data[key]
            if qa3c_p:
                self.agent.load_global_params(qa3c_p)

            # ── Restore MLP actor params (when use_mlp_actor=True) ──
            if self.agent.use_mlp_actor:
                for attr, key in [
                    ("actor_W1", "qa3c_actor_W1"),
                    ("actor_b1", "qa3c_actor_b1"),
                    ("actor_W2", "qa3c_actor_W2"),
                    ("actor_b2", "qa3c_actor_b2"),
                ]:
                    if key in data:
                        setattr(self.agent, attr, data[key].copy())

            # ── Restore critic params (always stored separately) ──
            for attr, key in [
                ("critic_proj", "qa3c_critic_proj"),
            ]:
                if key in data and hasattr(self.agent, attr):
                    setattr(self.agent, attr, data[key].copy())
            if "qa3c_critic_vqc" in data and self.agent.critic_vqc is not None:
                self.agent.critic_vqc.load_flat(data["qa3c_critic_vqc"])

            # ── Restore trade history ──
            if "trade_pnls" in data and "trade_sides" in data:
                pnls = data["trade_pnls"]
                sides = data["trade_sides"]
                self._trade_history = [
                    {"side": str(s), "pnl": float(p)}
                    for s, p in zip(sides, pnls)
                ]
            if "equity_curve" in data:
                self._equity_curve = data["equity_curve"].tolist()
                self._max_equity = max(self._equity_curve) if self._equity_curve else 10000.0
            if "train_epoch" in data:
                self.train_epoch = int(data["train_epoch"][0])
            if "train_steps" in data:
                self.agent.train_steps = int(data["train_steps"][0])

            self._last_checkpoint_ts = "loaded"
            return True
        except Exception:
            return False

    def maybe_auto_train(self, df: pd.DataFrame) -> None:
        """Increment candle counter; trigger short training if threshold reached."""
        self._candles_since_train += 1
        if (
            self.auto_train_interval > 0
            and self._candles_since_train >= self.auto_train_interval
            and not self.is_training
        ):
            self._candles_since_train = 0
            self.start_training(df, n_epochs=2)
