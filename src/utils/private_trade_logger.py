"""Private trade logger for Bybit account data (no secrets)."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


def _to_kst_str(ts) -> str:
    if ts is None:
        return ""
    if isinstance(ts, (int, float)):
        if ts > 1e12:
            ts = ts / 1000.0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        s = str(ts)
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return s[:19]
            dt = dt.replace(tzinfo=timezone.utc)
    kst = dt.astimezone(timezone(timedelta(hours=9)))
    return kst.strftime("%Y-%m-%d %H:%M:%S")


class PrivateTradeLogger:
    """Derives trade entries/exits from fills and logs to CSV."""

    def __init__(self, path: str = "logs/private_trades.csv"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._seen_exec_ids: set[str] = set()
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._ensure_header()

    def _ensure_header(self):
        if os.path.exists(self.path):
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "entry_time_kst",
                "exit_time_kst",
                "symbol",
                "side",
                "qty",
                "entry_price",
                "exit_price",
                "pnl",
                "leverage",
            ])

    def _get_leverage(self, positions: List[Dict[str, Any]], symbol: str) -> str:
        for p in positions:
            if p.get("symbol") == symbol and float(p.get("size", 0) or 0) != 0:
                return str(p.get("leverage", ""))
        return ""

    def update(self, snapshot: Dict[str, Any]):
        fills = snapshot.get("fills", []) or []
        positions = snapshot.get("positions", []) or []

        for f in sorted(fills, key=lambda x: str(x.get("execTime", ""))):
            exec_id = str(f.get("execId", ""))
            if not exec_id or exec_id in self._seen_exec_ids:
                continue
            self._seen_exec_ids.add(exec_id)

            symbol = f.get("symbol", "")
            side = f.get("side", "")
            qty = float(f.get("execQty", 0) or 0)
            price = float(f.get("execPrice", 0) or 0)
            exec_time = f.get("execTime")
            exec_pnl = float(f.get("execPnl", 0) or 0)

            if not symbol or qty == 0:
                continue

            pos = self._positions.get(symbol, {
                "signed_qty": 0.0,
                "entry_price": 0.0,
                "entry_time": "",
                "side": "",
                "pnl": 0.0,
                "leverage": "",
            })

            signed_qty = pos["signed_qty"]
            delta = qty if side == "Buy" else -qty
            new_qty = signed_qty + delta

            if signed_qty == 0 and new_qty != 0:
                pos["entry_price"] = price
                pos["entry_time"] = _to_kst_str(exec_time)
                pos["side"] = "LONG" if new_qty > 0 else "SHORT"
                pos["pnl"] = 0.0
                pos["leverage"] = self._get_leverage(positions, symbol)
            else:
                if (signed_qty > 0 and delta > 0) or (signed_qty < 0 and delta < 0):
                    total_cost = abs(signed_qty) * pos["entry_price"] + abs(delta) * price
                    pos["entry_price"] = total_cost / (abs(signed_qty) + abs(delta))

            pos["signed_qty"] = new_qty
            pos["pnl"] += exec_pnl

            if signed_qty != 0 and new_qty == 0:
                self._write_trade(
                    entry_time=pos["entry_time"],
                    exit_time=_to_kst_str(exec_time),
                    symbol=symbol,
                    side=pos["side"],
                    qty=abs(signed_qty),
                    entry_price=pos["entry_price"],
                    exit_price=price,
                    pnl=pos["pnl"],
                    leverage=pos["leverage"],
                )
                pos = {
                    "signed_qty": 0.0,
                    "entry_price": 0.0,
                    "entry_time": "",
                    "side": "",
                    "pnl": 0.0,
                    "leverage": "",
                }

            self._positions[symbol] = pos

    def _write_trade(
        self,
        entry_time: str,
        exit_time: str,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        leverage: str,
    ):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                entry_time,
                exit_time,
                symbol,
                side,
                f"{qty:.8f}",
                f"{entry_price:.4f}",
                f"{exit_price:.4f}",
                f"{pnl:.4f}",
                leverage,
            ])
