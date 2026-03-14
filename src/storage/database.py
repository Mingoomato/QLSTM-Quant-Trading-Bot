"""SQLite storage for trades, positions, and equity snapshots."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, Float, Integer, String, Text, create_engine, func, text, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


class TradeRecord(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(String, nullable=False)
    symbol = Column(String, default="BTCUSDT")
    side = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    qty = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    pnl = Column(Float, default=0.0)
    equity_after = Column(Float, default=0.0)
    strategy = Column(String, default="")
    signal_reason = Column(String, default="")
    meta = Column(Text, default="{}")


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(String, nullable=False)
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    position = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    drawdown = Column(Float, default=0.0)


class PositionRecord(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(String, nullable=False)
    symbol = Column(String, default="BTCUSDT")
    side = Column(String, nullable=False)
    qty = Column(Float, nullable=False)
    avg_entry = Column(Float, nullable=False)
    current_price = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    status = Column(String, default="open")


class Storage:
    def __init__(self, db_path: str = "data/trading.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._migrate_schema()
        self.SessionLocal = sessionmaker(bind=self.engine)

    def _migrate_schema(self):
        """Add missing columns to existing tables (SQLAlchemy create_all won't ALTER)."""
        insp = inspect(self.engine)
        expected = {
            "trades": {
                "equity_after": "FLOAT DEFAULT 0.0",
                "signal_reason": "VARCHAR DEFAULT ''",
            },
        }
        with self.engine.connect() as conn:
            for table, columns in expected.items():
                if not insp.has_table(table):
                    continue
                existing = {c["name"] for c in insp.get_columns(table)}
                for col_name, col_type in columns.items():
                    if col_name not in existing:
                        conn.execute(text(
                            f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                        ))
            conn.commit()

    def record_trade(
        self,
        ts: str,
        side: str,
        price: float,
        qty: float,
        fee: float = 0.0,
        slippage: float = 0.0,
        pnl: float = 0.0,
        equity_after: float = 0.0,
        symbol: str = "BTCUSDT",
        strategy: str = "",
        signal_reason: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ):
        with self.SessionLocal() as session:
            trade = TradeRecord(
                ts=ts, symbol=symbol, side=side, price=price, qty=qty,
                fee=fee, slippage=slippage, pnl=pnl, equity_after=equity_after,
                strategy=strategy, signal_reason=signal_reason,
                meta=json.dumps(meta or {}),
            )
            session.add(trade)
            session.commit()

    def record_equity_snapshot(
        self,
        ts: str,
        equity: float,
        cash: float,
        position: float = 0.0,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        drawdown: float = 0.0,
    ):
        with self.SessionLocal() as session:
            snap = EquitySnapshot(
                ts=ts, equity=equity, cash=cash, position=position,
                unrealized_pnl=unrealized_pnl, realized_pnl=realized_pnl,
                drawdown=drawdown,
            )
            session.add(snap)
            session.commit()

    def record_position(
        self,
        ts: str,
        symbol: str,
        side: str,
        qty: float,
        avg_entry: float,
        current_price: float = 0.0,
        unrealized_pnl: float = 0.0,
        status: str = "open",
    ):
        with self.SessionLocal() as session:
            pos = PositionRecord(
                ts=ts, symbol=symbol, side=side, qty=qty, avg_entry=avg_entry,
                current_price=current_price, unrealized_pnl=unrealized_pnl,
                status=status,
            )
            session.add(pos)
            session.commit()

    def get_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.SessionLocal() as session:
            trades = session.query(TradeRecord).order_by(
                TradeRecord.id.desc()
            ).limit(limit).all()
            return [
                {
                    "id": t.id, "ts": t.ts, "symbol": t.symbol, "side": t.side,
                    "price": t.price, "qty": t.qty, "fee": t.fee, "slippage": t.slippage,
                    "pnl": t.pnl, "equity_after": t.equity_after,
                    "strategy": t.strategy, "signal_reason": t.signal_reason,
                }
                for t in reversed(trades)
            ]

    def get_equity_curve(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self.SessionLocal() as session:
            snaps = session.query(EquitySnapshot).order_by(
                EquitySnapshot.id.desc()
            ).limit(limit).all()
            return [
                {"ts": s.ts, "equity": s.equity, "drawdown": s.drawdown}
                for s in reversed(snaps)
            ]

    def get_summary(self) -> Dict[str, Any]:
        with self.SessionLocal() as session:
            total = session.query(TradeRecord).count()
            if total == 0:
                return {"total_trades": 0, "total_pnl": 0, "total_fees": 0}

            result = session.query(
                func.sum(TradeRecord.pnl),
                func.sum(TradeRecord.fee),
                func.count(TradeRecord.id),
            ).one()

            return {
                "total_trades": result[2],
                "total_pnl": round(result[0] or 0, 4),
                "total_fees": round(result[1] or 0, 4),
            }

    def export_trades_csv(self, out_path: str = "trades.csv") -> int:
        """Export all trades to CSV. Returns row count."""
        trades = self.get_trades(limit=999999)
        if not trades:
            return 0
        fieldnames = list(trades[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)
        return len(trades)
