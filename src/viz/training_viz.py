"""
src/viz/training_viz.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Visualizer — standalone module

호출 시점:
  BC pretrain : 매 epoch 종료 후   → viz.plot_bc_epoch(...)
  RL training : fold best 업데이트 → viz.plot_rl_fold_best(...)

출력: reports/viz/bc_epoch_{N:03d}.png
      reports/viz/rl_fold{K}_best.png

Red dot 의미:
  "현재 모델의 수치가 이 곡면/곡선 위 어디에 있는가"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.font_manager as _fm
import numpy as np
from scipy.special import erf

# ── English-only font (avoids CJK glyph warnings) ──────────────────────
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
_RED   = "#e74c3c"
_BLUE  = "#2980b9"
_GREEN = "#27ae60"
_GOLD  = "#f39c12"
_GRAY  = "#95a5a6"
_DARK  = "#2c3e50"

_EPS_IND = 0.015   # ε for I(x>0) approximation smoothness (≈ half a SL%)


# ─────────────────────────────────────────────────────────────────────
# I(x>0) Approximation Functions
# ─────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray, eps: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x / eps))

def _erf_approx(x: np.ndarray, eps: float) -> np.ndarray:
    return 0.5 * (1.0 + erf(x / (eps * np.sqrt(2.0))))

def _algebraic(x: np.ndarray, eps: float) -> np.ndarray:
    return 0.5 + x / (2.0 * np.sqrt(x ** 2 + eps ** 2))

def _d_sigmoid(x: np.ndarray, eps: float) -> np.ndarray:
    s = _sigmoid(x, eps)
    return s * (1 - s) / eps

def _d_erf(x: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(-(x ** 2) / (2 * eps ** 2)) / (eps * np.sqrt(2 * np.pi))

def _d_algebraic(x: np.ndarray, eps: float) -> np.ndarray:
    return (eps ** 2) / (2.0 * (x ** 2 + eps ** 2) ** 1.5)


# ─────────────────────────────────────────────────────────────────────
# Parity Loss Surface
# ─────────────────────────────────────────────────────────────────────

def _parity_surface(resolution: int = 200):
    """Return (PL_grid, PS_grid, L_parity) on the probability simplex."""
    pl = np.linspace(0, 1, resolution)
    ps = np.linspace(0, 1, resolution)
    PL, PS = np.meshgrid(pl, ps)
    PH = 1.0 - PL - PS
    valid = PH >= 0
    L = np.full_like(PL, np.nan)
    L[valid] = (
        (PH[valid] - 1/3) ** 2 +
        (PL[valid] - 1/3) ** 2 +
        (PS[valid] - 1/3) ** 2
    )
    return PL, PS, L


# ─────────────────────────────────────────────────────────────────────
# TrainingVisualizer
# ─────────────────────────────────────────────────────────────────────

class TrainingVisualizer:
    """
    Standalone visualization module for BC and RL training.

    Usage (BC):
        viz = TrainingVisualizer()
        # inside epoch loop:
        viz.plot_bc_epoch(epoch, n_epochs, train_info, val_info, lr, dp)

    Usage (RL):
        viz = TrainingVisualizer()
        # when fold best updates:
        viz.plot_rl_fold_best(fold, epoch, metrics, fold_history, epoch_log)
    """

    def __init__(self, save_dir: str = "reports/viz"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._bc_history:  List[dict] = []
        self._rl_history:  List[dict] = []   # per fold-best entry

    # ── BC ────────────────────────────────────────────────────────────

    def plot_bc_epoch(
        self,
        epoch:      int,
        n_epochs:   int,
        train_info: dict,   # {loss, acc, ce, orth, parity}
        val_info:   dict,   # {bal_acc, acc_hold, acc_long, acc_short, dist_pred}
        lr:         float,
        dp:         dict,   # {HOLD, LONG, SHORT} predicted fraction
        tp_pct:     float = 0.09,   # typical TP as fraction (6×ATR ≈ 9%)
        sl_pct:     float = 0.0225, # typical SL as fraction (1.5×ATR ≈ 2.25%)
    ) -> str:
        """Draw 6-panel BC figure. Returns saved path."""
        rec = {
            "epoch": epoch, "lr": lr,
            **{f"tr_{k}": v for k, v in train_info.items()},
            "val_bal":   val_info.get("bal_acc", 0.0),
            "val_hold":  val_info.get("acc_hold", 0.0),
            "val_long":  val_info.get("acc_long", 0.0),
            "val_short": val_info.get("acc_short", 0.0),
            "p_hold": dp.get("HOLD", 1/3),
            "p_long": dp.get("LONG", 1/3),
            "p_short": dp.get("SHORT", 1/3),
        }
        self._bc_history.append(rec)
        hist = self._bc_history

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            f"BC Pretrain — Epoch {epoch}/{n_epochs}  |  "
            f"BalAcc={val_info.get('bal_acc',0):.2%}  "
            f"CE={train_info.get('ce',0):.4f}  "
            f"Orth={train_info.get('orth',0):.4f}  "
            f"Parity={train_info.get('parity',0):.4f}",
            fontsize=12, fontweight="bold", color=_DARK
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        epochs = [h["epoch"] for h in hist]

        # ── [0,0] Loss Components ──────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(epochs, [h["tr_loss"]   for h in hist], color=_DARK,  lw=2, label="Total")
        ax.plot(epochs, [h["tr_ce"]     for h in hist], color=_BLUE,  lw=1.5, label="CE")
        ax.plot(epochs, [h["tr_orth"]   for h in hist], color=_GREEN, lw=1.5, label="Orth")
        ax.plot(epochs, [h["tr_parity"] for h in hist], color=_GOLD,  lw=1.5, label="Parity")
        ax.scatter([epoch], [rec["tr_loss"]], color=_RED, s=80, zorder=6, label="Current")
        ax.set_title("Loss Components (train)", fontsize=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

        # ── [0,1] Accuracy Curves ─────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(epochs, [h["tr_acc"]   for h in hist], color=_GRAY,  lw=1.5, ls="--", label="Train acc")
        ax.plot(epochs, [h["val_bal"]  for h in hist], color=_DARK,  lw=2,   label="Val BalAcc")
        ax.plot(epochs, [h["val_hold"] for h in hist], color=_BLUE,  lw=1.5, label="HOLD acc")
        ax.plot(epochs, [h["val_long"] for h in hist], color=_GREEN, lw=1.5, label="LONG acc")
        ax.plot(epochs, [h["val_short"]for h in hist], color=_RED,   lw=1.5, label="SHORT acc")
        ax.axhline(1/3, color=_GRAY, ls=":", lw=1, label="Random (33%)")
        ax.scatter([epoch], [rec["val_bal"]], color=_RED, s=80, zorder=6)
        ax.set_title("Accuracy over Epochs", fontsize=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)

        # ── [0,2] I(x>0) Approximation Functions ─────────────────────
        ax = fig.add_subplot(gs[0, 2])
        x_range = np.linspace(-0.12, 0.12, 600)
        eps = _EPS_IND
        ax.plot(x_range, (x_range > 0).astype(float), "k--", lw=1.2, alpha=0.5, label="I(x>0) exact")
        ax.plot(x_range, _sigmoid(x_range, eps),   color=_BLUE,  lw=2, label=f"Sigmoid (ε={eps})")
        ax.plot(x_range, _erf_approx(x_range, eps),color=_GREEN, lw=2, label="Erf/Φ(x)")
        ax.plot(x_range, _algebraic(x_range, eps), color=_RED,   lw=2, label="Algebraic ★")
        # TP / SL markers
        ax.axvline( tp_pct, color=_GREEN, ls=":", lw=1.2, alpha=0.8)
        ax.axvline(-sl_pct, color=_RED,   ls=":", lw=1.2, alpha=0.8)
        ax.text( tp_pct+0.002, 0.05, f"TP\n+{tp_pct*100:.1f}%", fontsize=7, color=_GREEN)
        ax.text(-sl_pct+0.002, 0.85, f"SL\n-{sl_pct*100:.1f}%", fontsize=7, color=_RED)
        # Red dot: breakeven (x=0) → 현재 불확실 경계
        ax.scatter([0], [_algebraic(np.array([0.0]), eps)[0]],
                   color=_RED, s=100, zorder=7, label="BEP (x=0)")
        ax.set_title("I(x>0) Approximation Functions", fontsize=10)
        ax.set_xlabel("PnL (fraction)"); ax.set_ylabel("f(x) ~ I(x>0)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # ── [1,0] Gradient of I(x>0) ─────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(x_range, _d_sigmoid(x_range, eps),   color=_BLUE,  lw=2, label="∇ Sigmoid")
        ax.plot(x_range, _d_erf(x_range, eps),       color=_GREEN, lw=2, label="∇ Erf")
        ax.plot(x_range, _d_algebraic(x_range, eps), color=_RED,   lw=2, label="∇ Algebraic ★")
        ax.axvline( tp_pct, color=_GREEN, ls=":", lw=1.2, alpha=0.8)
        ax.axvline(-sl_pct, color=_RED,   ls=":", lw=1.2, alpha=0.8)
        # Red dot at x=0: gradient at breakeven
        ax.scatter([0], [_d_algebraic(np.array([0.0]), eps)[0]],
                   color=_RED, s=100, zorder=7, label=f"BEP grad={1/(2*eps):.1f}")
        ax.set_title("Gradient Magnitude (higher = stronger signal)", fontsize=10)
        ax.set_xlabel("PnL (fraction)"); ax.set_ylabel("df/dx")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # ── [1,1] Parity Loss Surface (2D Simplex) ────────────────────
        ax = fig.add_subplot(gs[1, 1])
        PL, PS, L_par = _parity_surface(150)
        cf = ax.contourf(PL, PS, L_par, levels=20, cmap="YlOrRd_r")
        ct = ax.contour(PL, PS, L_par, levels=10, colors="gray", linewidths=0.5, alpha=0.5)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="L_parity")
        # Simplex boundary
        bx = [0, 1, 0, 0]; by = [0, 0, 1, 0]
        ax.plot(bx, by, "k-", lw=1.5)
        # Minimum marker
        ax.scatter([1/3], [1/3], color=_GREEN, s=120, marker="*",
                   zorder=8, label="Minimum (1/3, 1/3)")
        # Red dot: current model (P_LONG, P_SHORT)
        p_l_cur = rec["p_long"]
        p_s_cur = rec["p_short"]
        ax.scatter([p_l_cur], [p_s_cur], color=_RED, s=150, zorder=9,
                   label=f"Current ({p_l_cur:.2f},{p_s_cur:.2f})")
        # Trajectory
        if len(hist) > 1:
            traj_l = [h["p_long"]  for h in hist]
            traj_s = [h["p_short"] for h in hist]
            ax.plot(traj_l, traj_s, color=_DARK, lw=1, alpha=0.5, ls="--")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("P(LONG)"); ax.set_ylabel("P(SHORT)")
        ax.set_title("Parity Loss Surface (P_H = 1-P_L-P_S)", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

        # ── [1,2] Predicted Probability Distribution ──────────────────
        ax = fig.add_subplot(gs[1, 2])
        x_pos = np.arange(3)
        labels = ["HOLD", "LONG", "SHORT"]
        colors_bar = [_BLUE, _GREEN, _RED]
        # History as light bars
        if len(hist) > 1:
            h_mean = [np.mean([h["p_hold"]  for h in hist]),
                      np.mean([h["p_long"]  for h in hist]),
                      np.mean([h["p_short"] for h in hist])]
            ax.bar(x_pos, h_mean, color=[c+"55" for c in colors_bar], label="Avg (all epochs)")
        # Current as solid bars
        cur_probs = [rec["p_hold"], rec["p_long"], rec["p_short"]]
        ax.bar(x_pos, cur_probs, color=colors_bar, alpha=0.85, label="Current epoch")
        ax.axhline(1/3, color=_GRAY, ls="--", lw=1.5, label="Uniform (1/3)")
        # Red dot at current max class
        max_idx = int(np.argmax(cur_probs))
        ax.scatter([max_idx], [cur_probs[max_idx]], color=_RED, s=120, zorder=7)
        ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1); ax.set_ylabel("Predicted Probability")
        ax.set_title("Model Output Distribution (Current Epoch)", fontsize=10)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

        path = os.path.join(self.save_dir, f"bc_epoch_{epoch:03d}.png")
        fig.savefig(path, dpi=90, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    # ── RL ────────────────────────────────────────────────────────────

    def plot_rl_fold_best(
        self,
        fold:         int,
        epoch:        int,
        metrics:      dict,   # {ev, wr, loss, critic_loss, fp_loss, dir_sym, qfi_mean, ...}
        fold_history: List[dict],   # list of best metrics per fold so far
        epoch_log:    List[dict],   # per-epoch log [{epoch, loss, ev, wr, ...}, ...]
        conf_thresh:  float = 0.65,
        bep:          float = 0.232,  # break-even WR
    ) -> str:
        """Draw 6-panel RL fold-best figure. Returns saved path."""
        self._rl_history.append({"fold": fold, "epoch": epoch, **metrics})

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            f"RL Training — Fold {fold}  Epoch {epoch}  |  "
            f"EV={metrics.get('ev',0):+.4f}  "
            f"WR={metrics.get('wr',0):.2%}  "
            f"Trades={metrics.get('n_trades',0)}",
            fontsize=12, fontweight="bold", color=_DARK
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ── [0,0] EV & WR per Fold ────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        folds_seen = [h["fold"] for h in fold_history]
        evs = [h.get("ev", 0) for h in fold_history]
        wrs = [h.get("wr", 0) for h in fold_history]
        x_f = np.arange(len(folds_seen))
        w = 0.35
        bars_ev = ax.bar(x_f - w/2, evs, w, color=_BLUE, alpha=0.8, label="EV/trade")
        bars_wr = ax.bar(x_f + w/2, wrs, w, color=_GREEN, alpha=0.8, label="WR")
        ax.axhline(0,   color=_DARK, lw=1, ls="--", alpha=0.5)
        ax.axhline(bep, color=_RED,  lw=1.5, ls=":", label=f"BEP WR={bep:.1%}")
        if folds_seen:
            ax.scatter([x_f[-1] + w/2], [wrs[-1]], color=_RED, s=100, zorder=7)
            ax.scatter([x_f[-1] - w/2], [evs[-1]], color=_RED, s=100, zorder=7)
        ax.set_xticks(x_f); ax.set_xticklabels([f"F{f}" for f in folds_seen])
        ax.set_title("EV & WR per Fold (Best)", fontsize=10)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

        # ── [0,1] Loss Components over Epochs ────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        if epoch_log:
            ep_x = [e["epoch"] for e in epoch_log]
            ax.plot(ep_x, [e.get("loss",0)        for e in epoch_log], color=_DARK,  lw=2,   label="Total")
            ax.plot(ep_x, [e.get("actor_loss",0)   for e in epoch_log], color=_BLUE,  lw=1.5, label="Actor")
            ax.plot(ep_x, [e.get("critic_loss",0)  for e in epoch_log], color=_GREEN, lw=1.5, label="Critic")
            ax.plot(ep_x, [e.get("fp_loss",0)      for e in epoch_log], color=_GOLD,  lw=1.5, label="FP")
            ax.plot(ep_x, [e.get("dir_sym",0)      for e in epoch_log], color=_GRAY,  lw=1.2, label="DirSym", ls="--")
            # Red dot at current best epoch
            cur = next((e for e in epoch_log if e["epoch"] == epoch), epoch_log[-1])
            ax.scatter([epoch], [cur.get("loss",0)], color=_RED, s=100, zorder=7, label="Current Best")
        ax.set_title("Loss Components (RL Epoch)", fontsize=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # ── [0,2] QFI over Epochs (Barren Plateau Monitor) ───────────
        ax = fig.add_subplot(gs[0, 2])
        if epoch_log:
            qfi_vals = [e.get("qfi_mean", 0) for e in epoch_log]
            ax.plot(ep_x, qfi_vals, color=_BLUE, lw=2, label="QFI mean")
            ax.fill_between(ep_x, 0, qfi_vals, alpha=0.15, color=_BLUE)
            ax.axhline(0.10, color=_RED, lw=1.5, ls="--", label="Barren threshold (0.10)")
            cur_qfi = next((e for e in epoch_log if e["epoch"] == epoch), epoch_log[-1]).get("qfi_mean", 0)
            ax.scatter([epoch], [cur_qfi], color=_RED, s=100, zorder=7, label=f"Current {cur_qfi:.4f}")
            # Shade BARREN region
            ax.axhspan(0, 0.10, color=_RED, alpha=0.05)
            ax.text(ep_x[0], 0.05, "⚠ BARREN ZONE", fontsize=7, color=_RED, alpha=0.6)
        ax.set_title("QFI (Quantum Fisher Info) — Barren Plateau", fontsize=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel("QFI mean")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # ── [1,0] I(x>0) 근사 + 현재 평균 PnL ───────────────────────
        ax = fig.add_subplot(gs[1, 0])
        x_range = np.linspace(-0.12, 0.12, 600)
        eps = _EPS_IND
        ax.plot(x_range, _algebraic(x_range, eps), color=_RED,   lw=2.5, label="Algebraic (best)")
        ax.plot(x_range, _sigmoid(x_range, eps),   color=_BLUE,  lw=1.5, label="Sigmoid",  alpha=0.7)
        ax.plot(x_range, _erf_approx(x_range, eps),color=_GREEN, lw=1.5, label="Erf",      alpha=0.7)
        ax.plot(x_range, (x_range > 0).astype(float), "k--", lw=1, alpha=0.4, label="I(x>0)")
        # BEP line
        ax.axvline(0, color=_DARK, lw=1, ls=":", alpha=0.5)
        # Current avg PnL from metrics
        avg_pnl = metrics.get("avg_pnl", 0.0)
        alg_at_pnl = float(_algebraic(np.array([avg_pnl]), eps)[0])
        ax.scatter([avg_pnl], [alg_at_pnl],
                   color=_RED, s=150, zorder=8,
                   label=f"Current avg_pnl={avg_pnl*100:+.2f}%")
        ax.set_title("I(x>0) Approx + Current Model PnL", fontsize=10)
        ax.set_xlabel("PnL (fraction)"); ax.set_ylabel("f(x)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # ── [1,1] Parity Surface + Fold Trajectory ────────────────────
        ax = fig.add_subplot(gs[1, 1])
        PL, PS, L_par = _parity_surface(150)
        cf = ax.contourf(PL, PS, L_par, levels=20, cmap="YlOrRd_r")
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="L_parity")
        bx = [0, 1, 0, 0]; by = [0, 0, 1, 0]
        ax.plot(bx, by, "k-", lw=1.5)
        ax.scatter([1/3], [1/3], color=_GREEN, s=120, marker="*", zorder=8, label="Minimum")
        # Fold trajectory
        if len(self._rl_history) > 1:
            traj_l = [h.get("mean_p_long",  1/3) for h in self._rl_history]
            traj_s = [h.get("mean_p_short", 1/3) for h in self._rl_history]
            ax.plot(traj_l, traj_s, color=_DARK, lw=1.2, alpha=0.6, ls="--", label="Fold trajectory")
        # Red dot: current fold
        p_l = metrics.get("mean_p_long",  1/3)
        p_s = metrics.get("mean_p_short", 1/3)
        ax.scatter([p_l], [p_s], color=_RED, s=180, zorder=9,
                   label=f"Current ({p_l:.2f},{p_s:.2f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("P(LONG)"); ax.set_ylabel("P(SHORT)")
        ax.set_title("Parity Loss Surface + Fold Trajectory", fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

        # ── [1,2] Confidence / WR Distribution ───────────────────────
        ax = fig.add_subplot(gs[1, 2])
        if epoch_log:
            wr_vals = [e.get("wr", 0) for e in epoch_log]
            ax.plot(ep_x, wr_vals, color=_GREEN, lw=2, label="WR per epoch")
            ax.fill_between(ep_x, bep, wr_vals,
                            where=[w > bep for w in wr_vals],
                            color=_GREEN, alpha=0.2, label="WR > BEP")
            ax.fill_between(ep_x, bep, wr_vals,
                            where=[w <= bep for w in wr_vals],
                            color=_RED, alpha=0.15, label="WR < BEP")
        ax.axhline(bep, color=_RED, lw=2, ls="--", label=f"BEP={bep:.1%}")
        ax.axhline(conf_thresh, color=_BLUE, lw=1.5, ls=":", label=f"conf_thr={conf_thresh}")
        if epoch_log:
            cur_wr = next((e for e in epoch_log if e["epoch"] == epoch), epoch_log[-1]).get("wr", 0)
            ax.scatter([epoch], [cur_wr], color=_RED, s=120, zorder=7, label=f"Current WR={cur_wr:.2%}")
        ax.set_ylim(0, 1)
        ax.set_title("WR over Epochs vs BEP", fontsize=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Win Rate")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        path = os.path.join(self.save_dir, f"rl_fold{fold:02d}_best.png")
        fig.savefig(path, dpi=90, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path
