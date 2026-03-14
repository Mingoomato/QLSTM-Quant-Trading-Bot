#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launch the Terminal Quant TUI with quantum autotrading enabled.

Usage examples:
  # Paper mode (default) — quantum model inference + simulated positions
  python scripts/run_quantum_tui.py --mode paper --quantum-model checkpoints/quantum_v2/agent_best.pt

  # Por mode — live Bybit Mainnet data + quantum positions
  python scripts/run_quantum_tui.py --mode por --quantum-model checkpoints/quantum_v2/agent_best.pt

  # With online training after each completed trade
  python scripts/run_quantum_tui.py --mode por --quantum-model checkpoints/quantum_v2/agent_best.pt --q-live-train

  # Custom parameters
  python scripts/run_quantum_tui.py \\
      --mode por \\
      --quantum-model checkpoints/quantum_v2/agent_best.pt \\
      --q-confidence 0.65 \\
      --q-leverage 10 \\
      --q-pos-frac 0.5 \\
      --q-tp-mult 4.0 \\
      --q-sl-mult 1.0

Defaults (4×ATR barrier, R:R=4:1, BEP=20.0%):
  --q-confidence 0.65   (backtest 최적 임계값)
  --q-leverage   10     (10x leverage)
  --q-pos-frac   0.5    (50% equity → effective 5x → MDD ~33%)
  --q-tp-mult    4.0    (TP = 4 × ATR)
  --q-sl-mult    1.0    (SL = 1 × ATR → R:R = 4:1, BEP=20.0%)
"""

import sys
import os

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.app.tui import run

if __name__ == "__main__":
    run()
