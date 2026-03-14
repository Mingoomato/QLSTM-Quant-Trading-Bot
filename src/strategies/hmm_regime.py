"""
src/strategies/hmm_regime.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HMM 3-State Regime Detector

Theory
------
3-state Gaussian HMM trained on 4 market microstructure features:

    State 0 (UP_TREND)   : positive mean return, moderate vol, persistent
    State 1 (CHOPPY)     : near-zero mean, high vol, no momentum
    State 2 (DOWN_TREND) : negative mean return, elevated vol, persistent

Gate logic (direction-aware):
    UP_TREND   → allow LONG only   (block SHORT → counter-trend)
    DOWN_TREND → allow SHORT only  (block LONG  → counter-trend)
    CHOPPY     → block all entries (no edge, fee drag kills EV)

Why HMM over simple Hurst threshold:
    - Hurst > 0.5 does not distinguish UP from DOWN trend
    - HMM uses joint distribution of return, vol, momentum, Sharpe
    - Transition matrix A encodes regime persistence (market memory)
    - Diag covariance: fewer params → stable fit on 300+ bars

Fitting
-------
    Fit on pre-test historical data (no look-ahead bias).
    Minimum 300 bars required (≈ 3 days @ 15m).
    Re-fit not required for backtesting (fit once, predict online).

Online Prediction
-----------------
    Use last `pred_window` (default 200) bars → Viterbi decode → last state.
    Viterbi uses only past data → no look-ahead bias.

Features (4-dim per bar)
--------------------------
    f[0]: rolling mean log-return × √252 (annualized, window=20)
    f[1]: rolling std log-return × √252  (annualized vol)
    f[2]: 5-bar momentum (mean of last 5 log-returns × √252)
    f[3]: pseudo-Sharpe ratio = f[0] / (f[1] + ε)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import os
import pickle
from enum import IntEnum
from typing import Tuple

import numpy as np

try:
    from hmmlearn import hmm as _hmmlib
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Regime States
# ─────────────────────────────────────────────────────────────────────────────

class RegimeState(IntEnum):
    UP_TREND   = 0   # persistent positive drift  → LONG allowed
    CHOPPY     = 1   # no clear direction          → no entry
    DOWN_TREND = 2   # persistent negative drift   → SHORT allowed

    def label(self) -> str:
        return {0: "↑ UP", 1: "≈ CHOP", 2: "↓ DOWN"}[int(self)]

    def allows_long(self)  -> bool: return self == RegimeState.UP_TREND
    def allows_short(self) -> bool: return self == RegimeState.DOWN_TREND
    def allows_entry(self) -> bool: return self != RegimeState.CHOPPY


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction  (MA-based: robust on 15m timeframe)
# ─────────────────────────────────────────────────────────────────────────────

def _build_hmm_features(prices: np.ndarray,
                        log_returns: np.ndarray) -> np.ndarray:
    """
    Build 4-dim HMM features from price + log-returns. [T, 4]

    Uses MA-based trend features — much more stable on 15m data
    than raw short-window rolling returns which are dominated by noise.

    Features:
        f[0]: (close - MA200) / MA200          trend vs long-term MA (signed %)
        f[1]: (MA50  - MA200) / MA200          MA crossover slope
        f[2]: realized_vol_100 * sqrt(35040)   annualized 100-bar vol
        f[3]: directional_frac_100             fraction of last 100 bars positive

    All clipped to [-3, 3] for fat-tail robustness.
    """
    T   = len(prices)
    feat = np.zeros((T, 4), dtype=np.float64)

    MA50  = np.zeros(T)
    MA200 = np.zeros(T)
    for i in range(T):
        MA50[i]  = prices[max(0, i-49) : i+1].mean()
        MA200[i] = prices[max(0, i-199): i+1].mean()

    _SQRT35040 = np.sqrt(35040.0)   # annualisation for 15m bars

    for i in range(T):
        ma200 = MA200[i] if MA200[i] > 0 else 1.0
        feat[i, 0] = (prices[i] - ma200) / ma200          # price vs MA200
        feat[i, 1] = (MA50[i]   - ma200) / ma200          # MA50 vs MA200

        s100 = max(0, i - 99)
        chunk = log_returns[s100 : i + 1] if i < len(log_returns) else log_returns[s100:]
        if len(chunk) >= 5:
            feat[i, 2] = chunk.std() * _SQRT35040         # annualised vol
            feat[i, 3] = float(np.sum(chunk > 0)) / len(chunk)  # directional fraction

    return np.clip(feat, -3.0, 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# HMMRegimeDetector
# ─────────────────────────────────────────────────────────────────────────────

class HMMRegimeDetector:
    """
    3-state Gaussian HMM regime detector.

    Usage
    -----
    ::

        det = HMMRegimeDetector()
        det.fit(log_returns_train)          # fit once on pre-test data

        # per bar in simulation:
        regime = det.predict(recent_log_returns)
        allowed, regime = det.is_action_allowed(action, recent_log_returns)

    Args:
        n_iter      : EM iterations for HMM fitting (default 200).
        roll_window : Feature rolling window in bars (default 20).
        pred_window : Bars of recent history for Viterbi prediction (default 200).
        min_fit_bars: Minimum bars required for fitting (default 300).
        random_state: Reproducibility seed (default 42).
    """

    def __init__(
        self,
        n_iter:       int = 200,
        roll_window:  int = 20,
        pred_window:  int = 200,
        min_fit_bars: int = 300,
        random_state: int = 42,
    ) -> None:
        if not _HMM_AVAILABLE:
            raise ImportError("hmmlearn not installed. Run: pip install hmmlearn")

        self.n_iter       = n_iter
        self.roll_window  = roll_window
        self.pred_window  = pred_window
        self.min_fit_bars = min_fit_bars
        self.random_state = random_state

        self._model:      object = None
        self._is_fitted:  bool   = False
        self._up_state:   int    = 0
        self._down_state: int    = 2
        self._choppy_state: int  = 1

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, prices: np.ndarray,
            log_returns: np.ndarray) -> "HMMRegimeDetector":
        """
        Train 3-state GaussianHMM on prices + log_returns.

        The three states are automatically identified after fitting:
            - UP_TREND   = state with highest mean (close-MA200)/MA200
            - DOWN_TREND = state with lowest  mean (close-MA200)/MA200
            - CHOPPY     = middle state

        Args:
            prices     : 1-D close price array. Must have >= min_fit_bars.
            log_returns: 1-D log-return array (len = len(prices) - 1).

        Returns:
            self (for chaining)
        """
        if len(prices) < self.min_fit_bars:
            return self   # not enough data, stay unfitted

        feat = _build_hmm_features(prices, log_returns)
        # Skip first 200 rows (MA200 warm-up period)
        feat = feat[200:]

        model = _hmmlib.GaussianHMM(
            n_components    = 3,
            covariance_type = "diag",   # fewer params, stable on 300 bars
            n_iter          = self.n_iter,
            random_state    = self.random_state,
            tol             = 1e-4,
        )
        model.fit(feat)

        # ── State identification ──────────────────────────────────────────
        # Sort states by mean of feature[0] (annualized return)
        mean_ret = model.means_[:, 0]           # [3] annualized return per state
        order    = np.argsort(mean_ret)         # ascending: [down, chop, up]

        self._down_state   = int(order[0])
        self._choppy_state = int(order[1])
        self._up_state     = int(order[2])
        self._model        = model
        self._is_fitted    = True

        return self

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, prices: np.ndarray,
                log_returns: np.ndarray) -> RegimeState:
        """
        Predict current market regime from recent prices + log-returns.

        Uses Viterbi decoding on the last `pred_window` bars.
        No look-ahead bias: uses only past data.

        Args:
            prices     : Recent close prices (len = len(log_returns) + 1).
            log_returns: Recent log-returns (at least 50 bars).

        Returns:
            RegimeState enum value.
        """
        if not self._is_fitted:
            return RegimeState.CHOPPY   # conservative fallback

        # Trim to pred_window (keep alignment: len(pr) = len(lr) + 1)
        n_lr = min(self.pred_window, len(log_returns))
        recent_lr = log_returns[-n_lr:]
        recent_pr = prices[-(n_lr + 1):]

        if len(recent_lr) < 50:
            return RegimeState.CHOPPY

        feat = _build_hmm_features(recent_pr, recent_lr)
        # Use all features — MA200 values in early rows will be approximate
        # but Viterbi still benefits from the full sequence for context

        if len(feat) < 3:
            return RegimeState.CHOPPY

        try:
            states  = self._model.predict(feat)
            current = int(states[-1])

            if current == self._up_state:
                return RegimeState.UP_TREND
            elif current == self._down_state:
                return RegimeState.DOWN_TREND
            else:
                return RegimeState.CHOPPY
        except Exception:
            return RegimeState.CHOPPY

    # ── Action gate ──────────────────────────────────────────────────────────

    def is_action_allowed(
        self, action: int,
        prices: np.ndarray, log_returns: np.ndarray,
    ) -> Tuple[bool, RegimeState]:
        """
        Direction-aware regime gate.

        Args:
            action      : 1 = LONG, 2 = SHORT
            prices      : Recent close prices (len = len(log_returns) + 1).
            log_returns : Recent log-returns for regime prediction.

        Returns:
            (allowed: bool, regime: RegimeState)

        Logic:
            UP_TREND   + LONG  → True   (with the trend)
            UP_TREND   + SHORT → False  (against the trend)
            DOWN_TREND + SHORT → True   (with the trend)
            DOWN_TREND + LONG  → False  (against the trend)
            CHOPPY     + any   → False  (no directional edge)
        """
        regime = self.predict(prices, log_returns)

        if regime == RegimeState.CHOPPY:
            return False, regime
        if action == 1 and regime == RegimeState.DOWN_TREND:
            return False, regime
        if action == 2 and regime == RegimeState.UP_TREND:
            return False, regime

        return True, regime

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def describe(self) -> str:
        """Human-readable summary of fitted states."""
        if not self._is_fitted:
            return "HMMRegimeDetector (not fitted)"

        m = self._model.means_     # [3, 4]
        lines = ["HMMRegimeDetector — 3 states:"]
        mapping = {
            self._up_state:     "UP_TREND  ",
            self._choppy_state: "CHOPPY    ",
            self._down_state:   "DOWN_TREND",
        }
        for k in range(3):
            label  = mapping.get(k, f"state{k}   ")
            dev200 = m[k, 0]   # f[0]: (close-MA200)/MA200
            cross  = m[k, 1]   # f[1]: (MA50-MA200)/MA200
            vol    = m[k, 2]   # f[2]: annualized vol
            dirfr  = m[k, 3]   # f[3]: directional fraction
            lines.append(
                f"  State {k} → {label}  "
                f"dev200={dev200:+.3f}  cross={cross:+.3f}  "
                f"vol={vol:.3f}  dir={dirfr:.2f}"
            )
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle detector to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":         self._model,
                "is_fitted":     self._is_fitted,
                "up_state":      self._up_state,
                "down_state":    self._down_state,
                "choppy_state":  self._choppy_state,
                "roll_window":   self.roll_window,
                "pred_window":   self.pred_window,
            }, f)

    @classmethod
    def load(cls, path: str) -> "HMMRegimeDetector":
        """Load detector from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        det               = cls(roll_window=data["roll_window"],
                                pred_window=data["pred_window"])
        det._model        = data["model"]
        det._is_fitted    = data["is_fitted"]
        det._up_state     = data["up_state"]
        det._down_state   = data["down_state"]
        det._choppy_state = data["choppy_state"]
        return det
