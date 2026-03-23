"""
qng_optimizer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hybrid Quantum Natural Gradient Optimizer (Diagonal-QFI approximation)

Motivation
----------
Adam is geometrically wrong for VQC parameters. VQC parameter space is the
unitary manifold U(2^n), not Euclidean R^n.  Adam treats equal numerical
step Δθ as equal distance everywhere — on the unitary manifold, the same
Δθ may change the quantum state a lot (near identity) or almost nothing
(near a flat region = Barren Plateau).

QNG fixes this by pre-conditioning with the Quantum Fisher Information:

    θ_{t+1} = θ_t  −  lr_q · (F^Q_diag + ε)^{-1} · ∇L

where F^Q_ii = 0.5 · E[ ||f(θ+π/2) − f(θ−π/2)||² ]

This is exact for Pauli rotation generators (Rot = Rz·Ry·Rz) used in
StronglyEntanglingLayers, and costs only 2×n_vqc_params extra forward
passes per QFI update.

Architecture
-----------
  VQC params   (vqc_weights)   → Diagonal-QFI SGD (QNG)
  Classical params (rest)      → AdamW  (unchanged)

Cost estimate (our setup)
-------------------------
  n_vqc_params = 2 layers × 4 qubits × 3 angles = 24
  QFI cost = 2 × 24 = 48 forward passes (pure PyTorch TorchVQC)
  Update every 20 steps → 2.4 extra passes/step overhead ≈ +5-15%

References
----------
  Stokes et al. (2020), "Quantum Natural Gradient", Quantum 4, 269.
  Amari (1998), "Natural Gradient Works Efficiently in Learning".
  F^Q = 4·Re[⟨∂_i ψ|∂_j ψ⟩ − ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import math
from typing import Optional, List

import torch
import torch.optim as optim


class DiagonalQNGOptimizer:
    """
    Hybrid optimizer: Diagonal-QFI QNG for VQC params + AdamW for classical.

    Args
    ----
    model            : QuantumFinancialAgent instance
    lr_classical     : AdamW LR for classical params        (default 3e-4)
    lr_quantum       : QNG step size for VQC params         (default 0.05)
                       Larger than classical LR is fine —
                       QFI normalises the update scale.
    weight_decay     : AdamW weight decay                   (default 1e-4)
    qfi_update_freq  : Steps between QFI recomputation      (default 20)
    qfi_damping      : Tikhonov ε to prevent 1/0            (default 1e-3)
    qfi_ema          : EMA factor for smoothing QFI diagonal (default 0.95)
    """

    def __init__(
        self,
        model,
        lr_classical:    float = 1e-4,
        lr_quantum:      float = 0.05,
        weight_decay:    float = 1e-4,
        qfi_update_freq: int   = 20,
        qfi_damping:     float = 1e-3,
        qfi_ema:         float = 0.95,
    ) -> None:
        self.lr_quantum      = lr_quantum
        self.qfi_update_freq = qfi_update_freq
        self.qfi_damping     = qfi_damping
        self.qfi_ema         = qfi_ema
        self._step_count     = 0

        # ── Split parameters ──────────────────────────────────────────────
        vqc_params       : List[torch.nn.Parameter] = []
        classical_params : List[torch.nn.Parameter] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "vqc_weights" in name:
                vqc_params.append(param)
            else:
                classical_params.append(param)

        self.vqc_params = vqc_params
        n_vqc = sum(p.numel() for p in vqc_params)
        n_cls = sum(p.numel() for p in classical_params)

        print(f"  [QNG] VQC params     : {n_vqc}  "
              f"(update via Diagonal-QFI every {qfi_update_freq} steps)")
        print(f"  [QNG] Classical params: {n_cls}  (AdamW, lr={lr_classical})")
        print(f"  [QNG] QFI cost/update : {2*n_vqc} forward passes")

        # QFI diagonal cache — init to 1 (identity preconditioner = plain SGD)
        self._qfi_diag: Optional[torch.Tensor] = (
            torch.ones(n_vqc) if n_vqc > 0 else None
        )
        self._qfi_initialized = False

        # ── AdamW for classical parameters ────────────────────────────────
        self.classical_optimizer = optim.AdamW(
            classical_params,
            lr=lr_classical,
            weight_decay=weight_decay,
        )

    # ── Public interface ─────────────────────────────────────────────────────

    def zero_grad(self) -> None:
        self.classical_optimizer.zero_grad()
        for p in self.vqc_params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self, encoder=None) -> None:
        """
        Apply one optimisation step.

        Args
        ----
        encoder : QuantumHamiltonianLayer  (None → skip QFI update this step)
                  Pass agent.encoder.quantum_layer from train_step.
        """
        # ── 1. Refresh QFI diagonal every qfi_update_freq steps ──────────
        if (
            encoder is not None
            and self._qfi_diag is not None
            and self._step_count % self.qfi_update_freq == 0
        ):
            new_qfi = encoder.compute_qfi_diagonal()
            if new_qfi is not None:
                new_qfi = new_qfi.cpu()
                if not self._qfi_initialized:
                    self._qfi_diag = new_qfi
                    self._qfi_initialized = True
                else:
                    # Exponential moving average — smooth out sample noise
                    # Ensure both tensors on same device (CPU)
                    self._qfi_diag = (
                        self.qfi_ema * self._qfi_diag.cpu()
                        + (1.0 - self.qfi_ema) * new_qfi.cpu()
                    )

        # ── 2. QNG update for VQC params ──────────────────────────────────
        if self.vqc_params and self._qfi_diag is not None:
            offset = 0
            for p in self.vqc_params:
                n = p.numel()
                if p.grad is None:
                    offset += n
                    continue

                qfi_slice = (
                    self._qfi_diag[offset : offset + n]
                    .reshape(p.shape)
                    .to(p.device)
                )
                # Pre-conditioned gradient: g_qng = g / (F_ii + ε)
                precond = p.grad.data / (qfi_slice + self.qfi_damping)

                with torch.no_grad():
                    p.data -= self.lr_quantum * precond

                offset += n

        # ── 3. AdamW step for classical parameters ────────────────────────
        self.classical_optimizer.step()

        self._step_count += 1

    # ── Checkpoint support ───────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "step_count":      self._step_count,
            "qfi_diag":        self._qfi_diag,
            "qfi_initialized": self._qfi_initialized,
            "lr_quantum":      self.lr_quantum,
            "classical_opt":   self.classical_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._step_count      = state.get("step_count", 0)
        self._qfi_diag        = state.get("qfi_diag", self._qfi_diag)
        self._qfi_initialized = state.get("qfi_initialized", False)
        self.lr_quantum       = state.get("lr_quantum", self.lr_quantum)
        self.classical_optimizer.load_state_dict(state["classical_opt"])

    # ── Scheduler compatibility shim ─────────────────────────────────────────

    @property
    def param_groups(self):
        """
        Delegate to classical_optimizer.param_groups so that
        CosineAnnealingWarmRestarts can adjust the classical LR.
        VQC lr_quantum is kept constant (QFI already normalises scale).
        """
        return self.classical_optimizer.param_groups

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def get_qfi_stats(self) -> dict:
        """Return QFI diagonal statistics for training log."""
        if self._qfi_diag is None or not self._qfi_initialized:
            return {"qfi_mean": float("nan"), "qfi_min": float("nan"),
                    "qfi_max": float("nan")}
        return {
            "qfi_mean": float(self._qfi_diag.mean()),
            "qfi_min":  float(self._qfi_diag.min()),
            "qfi_max":  float(self._qfi_diag.max()),
        }
