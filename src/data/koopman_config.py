"""
koopman_config.py
─────────────────────────────────────────────────────────────────────────────
Precompute Koopman EDMD configuration (Ridge CV + Sparse Dict + Validation)
on the full training dataset, then save to disk.

SpectralDecomposer loads this config at init and uses the precomputed
Koopman eigenvectors as a FIXED projection matrix instead of refitting
the Koopman operator dynamically per batch.

Pipeline:
    Training features [N, D]
        ↓ Z-score normalise
        ↓ Nonlinear dictionary  ψ(x) = [x, x², tanh(x)]  → [N, 3D]
        ↓ Dictionary normalisation (per-column unit variance)
        ↓ Sparse selection  (Spearman MI, top max_terms)
        ↓ Ridge CV alpha search (5-fold, time-ordered)
        ↓ Final Koopman operator K = G⁻¹ A
        ↓ Eigendecompose K → eigenvectors sorted by |λ-1| ascending
        → Save: feat_mu/sig, psi_mu/sig, selected, koopman_vecs, eigenvalues

During training forward pass (per batch):
    x [B, T, D]
        ↓ apply saved feat normalisation
        ↓ build dict [B, T, 3D]
        ↓ apply saved dict normalisation
        ↓ select saved columns  [B, T, max_terms]
        ↓ project: x @ koopman_vecs  → c_kt [B, T, n_modes]
    (no Koopman refitting — just a matrix multiply)
"""

from __future__ import annotations

import os
import numpy as np

KOOPMAN_CONFIG_VERSION = "1.1"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (numpy-only, called only during precompute)
# ─────────────────────────────────────────────────────────────────────────────

def _build_dict_np(Xn: np.ndarray) -> np.ndarray:
    """
    Stable nonlinear dictionary for EDMD.

    ψ(x) = [x_clip, x_clip², tanh(x_clip)]  → 3D dimensions

    Clipping prevents the x/roll_std instability that caused MSE=10^13.
    All terms bounded: x_clip ∈ [-5,5], x² ∈ [0,25], tanh ∈ (-1,1).
    """
    x_clip = np.clip(Xn, -5.0, 5.0)
    x2     = np.clip(x_clip ** 2, 0.0, 25.0)
    x_tanh = np.tanh(x_clip)
    return np.concatenate([x_clip, x2, x_tanh], axis=1)   # [N, 3D]


def _sparse_selection(
    Psi:       np.ndarray,   # [N, 3D]  normalised dictionary
    Y:         np.ndarray,   # [N, D]   next-state targets
    max_terms: int = 40,
) -> np.ndarray:
    """
    Keep only the max_terms dictionary columns most correlated with Y.
    Spearman rank correlation: robust to fat tails, captures monotone NL.
    """
    from scipy.stats import spearmanr
    n_terms  = Psi.shape[1]
    D_target = min(Y.shape[1], 8)   # check first 8 target dims (representative)
    scores   = np.zeros(n_terms)
    for j in range(n_terms):
        corrs = []
        for d in range(D_target):
            r, _ = spearmanr(Psi[:, j], Y[:, d])
            corrs.append(abs(r) if not np.isnan(r) else 0.0)
        scores[j] = max(corrs)
    return np.sort(np.argsort(scores)[::-1][:max_terms])


def _ridge_cv_alpha(
    Psi_now:  np.ndarray,
    Psi_next: np.ndarray,
    alphas:   tuple = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0),
    n_folds:  int   = 5,
    verbose:  bool  = True,
) -> float:
    """
    5-fold time-ordered CV to find optimal ridge regularisation strength.
    Returns alpha with lowest average OOS prediction MSE.
    """
    N, D = Psi_now.shape
    fold_size = N // n_folds
    best_alpha, best_mse = alphas[0], np.inf
    for alpha in alphas:
        mse_folds = []
        for k in range(n_folds):
            v0, v1 = k * fold_size, (k + 1) * fold_size
            trx = np.concatenate([Psi_now[:v0],  Psi_now[v1:]],  axis=0)
            try_ = np.concatenate([Psi_next[:v0], Psi_next[v1:]], axis=0)
            vlx  = Psi_now[v0:v1]
            vly  = Psi_next[v0:v1]
            G  = trx.T @ trx / len(trx) + alpha * np.eye(D)
            A  = trx.T @ try_ / len(trx)
            K, _, _, _ = np.linalg.lstsq(G, A, rcond=1e-4)
            mse_folds.append(float(np.mean((vlx @ K - vly) ** 2)))
        avg = float(np.mean(mse_folds))
        if verbose:
            print(f"    alpha={alpha:.0e}  OOS-MSE={avg:.6f}", flush=True)
        if avg < best_mse:
            best_mse   = avg
            best_alpha = alpha
    return best_alpha


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def precompute_koopman_config(
    X_train:   np.ndarray,
    n_modes:   int   = 5,
    max_terms: int   = 40,
    save_path: str   = "data/koopman_config.npz",
    verbose:   bool  = True,
) -> dict:
    """
    Run full Koopman precomputation on training data and save to disk.

    This runs ONCE before training (not per batch).
    The result is loaded by SpectralDecomposer.load_koopman_precomputed().

    Args:
        X_train   : [N, D]  raw training features (time-ordered)
        n_modes   : number of Koopman modes to keep (= n_eigenvectors)
        max_terms : max sparse dictionary terms to keep
        save_path : output .npz path
        verbose   : print progress

    Returns:
        config dict (same content as the saved .npz)
    """
    # ── Strip warmup zeros ────────────────────────────────────────────────────
    nonzero_mask = np.any(X_train != 0, axis=1)
    X_valid = X_train[nonzero_mask]
    N, D = X_valid.shape
    if verbose:
        print(f"[koopman_config] Valid bars={N}  features={D}", flush=True)

    # ── Step 1: Feature-level Z-score ─────────────────────────────────────────
    feat_mu  = X_valid.mean(axis=0)
    feat_sig = X_valid.std(axis=0).clip(min=1e-6)
    Xn = (X_valid - feat_mu) / feat_sig

    # ── Step 2: Nonlinear dictionary ──────────────────────────────────────────
    if verbose:
        print("[koopman_config] Step 1/4: Building nonlinear dictionary ...",
              flush=True)
    Psi_now_raw  = _build_dict_np(Xn[:-1])    # [N-1, 3D]
    Psi_next_raw = _build_dict_np(Xn[1:])

    # Dictionary-level normalisation (no leakage: use train stats only)
    psi_mu  = Psi_now_raw.mean(axis=0)
    psi_sig = Psi_now_raw.std(axis=0).clip(min=1e-6)
    Psi_now  = (Psi_now_raw  - psi_mu) / psi_sig
    Psi_next = (Psi_next_raw - psi_mu) / psi_sig

    # ── Step 3: Sparse selection ──────────────────────────────────────────────
    if verbose:
        print("[koopman_config] Step 2/4: Sparse dictionary selection ...",
              flush=True)
    max_terms = min(max_terms, Psi_now.shape[1])
    selected  = _sparse_selection(Psi_now, Xn[1:], max_terms=max_terms)
    Psi_s_now  = Psi_now[:, selected]
    Psi_s_next = Psi_next[:, selected]
    if verbose:
        print(f"[koopman_config]   Kept {len(selected)}/{Psi_now.shape[1]} terms",
              flush=True)

    # ── Step 4: Ridge CV alpha ────────────────────────────────────────────────
    if verbose:
        print("[koopman_config] Step 3/4: Ridge CV alpha search ...", flush=True)
    best_alpha = _ridge_cv_alpha(Psi_s_now, Psi_s_next, verbose=verbose)
    if verbose:
        print(f"[koopman_config]   Best alpha={best_alpha:.0e}", flush=True)

    # ── Step 5: Final Koopman operator ────────────────────────────────────────
    if verbose:
        print("[koopman_config] Step 4/4: Final Koopman eigendecomposition ...",
              flush=True)
    lifted_D = Psi_s_now.shape[1]
    G = Psi_s_now.T @ Psi_s_now / len(Psi_s_now) + best_alpha * np.eye(lifted_D)
    A = Psi_s_now.T @ Psi_s_next / len(Psi_s_now)
    K, _, _, _ = np.linalg.lstsq(G, A, rcond=1e-4)

    eigenvalues, eigenvectors = np.linalg.eig(K)
    persistence = np.abs(np.abs(eigenvalues) - 1.0)
    top_idx     = np.argsort(persistence)[:n_modes]
    top_eigs    = eigenvalues[top_idx]
    top_vecs    = np.real(eigenvectors[:, top_idx])   # [lifted_D, n_modes]
    norms = np.linalg.norm(top_vecs, axis=0, keepdims=True).clip(min=1e-8)
    top_vecs /= norms

    lam_abs = np.abs(top_eigs)
    if verbose:
        print(f"[koopman_config] |λ_k|: {lam_abs.round(4).tolist()}", flush=True)
        slow = int(np.sum(np.abs(lam_abs - 1.0) < 0.05))
        print(f"[koopman_config] Slow modes (|λ|>0.95): {slow}/{n_modes}",
              flush=True)

    config = {
        "version":       np.array([KOOPMAN_CONFIG_VERSION]),
        "n_modes":       np.array([n_modes]),
        "feature_dim":   np.array([D]),
        "feat_mu":       feat_mu.astype(np.float32),
        "feat_sig":      feat_sig.astype(np.float32),
        "psi_mu":        psi_mu.astype(np.float32),
        "psi_sig":       psi_sig.astype(np.float32),
        "selected":      selected.astype(np.int64),
        "koopman_vecs":  top_vecs.astype(np.float32),   # [lifted_D, n_modes]
        "eigenvalues":   top_eigs.real.astype(np.float32),
        "best_alpha":    np.array([best_alpha], dtype=np.float32),
    }

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez(save_path, **config)
    if verbose:
        print(f"[koopman_config] Saved → {save_path}", flush=True)

    return config


def load_koopman_config(path: str) -> dict:
    """Load precomputed Koopman config from .npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Koopman config not found: {path}")
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}
