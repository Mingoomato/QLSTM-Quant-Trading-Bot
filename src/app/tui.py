# -*- coding: utf-8 -*-
"""Terminal Quant TUI — Quantum dual-model autotrader (live data only).

Modes:
  paper  : live Bybit data + quantum inference + simulated P&L (practice)
  por    : live Bybit data + quantum inference + real Bybit orders (live trade)

Key bindings:
  b  : Start auto-trading  (por: places real orders / paper: already running)
  x  : Stop  auto-trading  (closes all open positions)
  k  : Kill switch          (emergency: close all, halt)
  q  : Quit
  1-5: Change timeframe (1m / 5m / 15m / 30m / 1h)
  i  : Toggle indicator details
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from collections import deque as _deque_q
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Static

from src.app.cli_args import parse_tui_args
from src.data.data_client import DataClient
from src.storage.database import Storage
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.private_trade_logger import PrivateTradeLogger

try:
    import numpy as _np_q
    from src.models.integrated_agent import AgentConfig, build_quantum_agent
    from src.models.features_v4 import build_features_v4
    from src.data.binance_client import fetch_binance_taker_history
    QUANTUM_AVAILABLE = True
except ImportError:
    fetch_binance_taker_history = None
    QUANTUM_AVAILABLE = False

logger = setup_logger("tui")

# ── Bybit signed REST helper ───────────────────────────────────────────────────

_BYBIT_REST = "https://api.bybit.com"


def _bybit_signed(method: str, path: str, params: dict,
                  api_key: str, api_secret: str) -> dict:
    """Execute a Bybit V5 signed REST request."""
    ts = str(int(time.time() * 1000))
    recv_window = "5000"
    if method == "POST":
        body = json.dumps(params, separators=(",", ":"))
        sign_str = ts + api_key + recv_window + body
        url = _BYBIT_REST + path
        data = body.encode("utf-8")
    else:
        qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        sign_str = ts + api_key + recv_window + qs
        url = _BYBIT_REST + path + ("?" + qs if qs else "")
        data = None
    sig = hmac.new(api_secret.encode("utf-8"),
                   sign_str.encode("utf-8"), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-SIGN": sig,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json",
    }
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Bybit API error [{method} {path}]: {e}")
        return {"retCode": -1, "retMsg": str(e)}


# ── Chart helpers ──────────────────────────────────────────────────────────────

SPARK = "▁▂▃▄▅▆▇█"
PRICE_AXIS_WIDTH = 10


def sparkline(values: list, width: int = 40) -> str:
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    mn, mx = min(sampled), max(sampled)
    rng = mx - mn if mx != mn else 1
    return "".join(SPARK[min(int((v - mn) / rng * 7), 7)] for v in sampled)


def ascii_candle_chart(df, width: int = 50, height: int = 10) -> str:
    if len(df) == 0:
        return "No data"
    n = min(len(df), width)
    recent = df.tail(n)
    opens  = recent["open"].values
    highs  = recent["high"].values
    lows   = recent["low"].values
    closes = recent["close"].values

    all_min = float(min(lows))
    all_max = float(max(highs))
    rng = all_max - all_min or 1

    def scale(v):
        return int((v - all_min) / rng * (height - 1))

    WICK, BODY = "│", "█"
    rows = [[" "] * n for _ in range(height)]
    for col in range(n):
        o, h, l, c = float(opens[col]), float(highs[col]), float(lows[col]), float(closes[col])
        for y in range(scale(l), scale(h) + 1):
            rows[height - 1 - y][col] = WICK
        char = f"[green]{BODY}[/green]" if c >= o else f"[red]{BODY}[/red]"
        for y in range(min(scale(o), scale(c)), max(scale(o), scale(c)) + 1):
            rows[height - 1 - y][col] = char

    lines = []
    for i, row in enumerate(rows):
        price_at = all_max - (i / (height - 1)) * rng if height > 1 else all_max
        lines.append(f"{price_at:>{PRICE_AXIS_WIDTH}.2f} {''.join(row)}")
    return "\n".join(lines)


def _format_ts_kst(ts) -> str:
    if ts is None:
        return ""
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    elif isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        s = str(ts)
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return s[:5]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone(timedelta(hours=9))).strftime("%H:%M")


def _format_ts_kst_full(ts) -> str:
    if ts is None:
        return "N/A"
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    elif isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        s = str(ts)
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return s[:19]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")


def _parse_ts_utc(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        if ts > 1e12:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    s = str(ts)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        dt = dt.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _timeframe_seconds(tf: str) -> int:
    return {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "4h": 14400, "1d": 86400}.get(tf, 300)


def time_axis_kst(df, width: int = 50, label_count: int = 4) -> str:
    if len(df) == 0:
        return ""
    n = min(len(df), width)
    recent = df.tail(n)
    ts_series = recent["ts"] if "ts" in recent.columns else recent.index
    step = max(1, n // (label_count - 1)) if label_count > 1 else n
    positions = list(range(0, n, step))
    if positions[-1] != n - 1:
        positions.append(n - 1)
    row = [" "] * n
    for pos in positions:
        ts = ts_series.iloc[pos] if hasattr(ts_series, "iloc") else ts_series[pos]
        label = _format_ts_kst(ts)
        if not label:
            continue
        start = max(0, min(n - len(label), pos - len(label) // 2))
        for i, ch in enumerate(label):
            row[start + i] = ch
    return " " * (PRICE_AXIS_WIDTH + 1) + "".join(row)


def orderbook_summary(ob: dict) -> list:
    def _sum(levels):
        return sum(float(lvl[1]) for lvl in levels if len(lvl) >= 2
                   and lvl[1] not in (None, ""))
    def _pct(b, a):
        tot = b + a
        return (b / tot * 100, a / tot * 100) if tot > 0 else (0.0, 0.0)

    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    if not bids and not asks:
        return ["[dim]ORDERBOOK: no data[/dim]"]
    b100, a100 = _sum(bids[:100]), _sum(asks[:100])
    b10,  a10  = _sum(bids[:10]),  _sum(asks[:10])
    p100 = _pct(b100, a100)
    p10  = _pct(b10,  a10)
    return [
        "[b]ORDERBOOK % (BUY/SELL)[/b]",
        f"  100: {p100[0]:5.1f}% / {p100[1]:5.1f}%",
        f"   10: {p10[0]:5.1f}% / {p10[1]:5.1f}%",
    ]


# ── Widgets ────────────────────────────────────────────────────────────────────

class PaneWidget(Static):
    DEFAULT_CSS = "PaneWidget { border: solid #2a3b47; padding: 0 1; overflow-y: auto; }"


class WatchlistPane(PaneWidget):
    pass


class ChartPane(PaneWidget):
    pass


class QuantPane(PaneWidget):
    pass


class StatusBar(Static):
    DEFAULT_CSS = (
        "StatusBar { height: 3; background: #0f141b; color: #cfd8dc; "
        "padding: 0 1; border-top: solid #2a3b47; }"
    )


# ── Main App ───────────────────────────────────────────────────────────────────

class TerminalQuantApp(App):
    CSS = """
    Screen { layout: vertical; background: #0b1117; color: #cfd8dc; }
    Header, Footer { background: #0f141b; color: #9fb3c8; }
    #main-area { height: 1fr; }
    #left-pane  { width: 30; }
    #center-pane { width: 1fr; }
    #right-pane { width: 34; }
    PaneWidget { border: solid #2a3b47; padding: 0 1; overflow-y: auto; }
    StatusBar  { background: #0f141b; color: #cfd8dc; border-top: solid #2a3b47; }
    """

    BINDINGS = [
        Binding("q", "quit",          "Quit",       priority=True),
        Binding("k", "kill_switch",   "Kill"),
        Binding("b", "start_trading", "StartTrade"),
        Binding("x", "stop_trading",  "StopTrade"),
        Binding("1", "set_tf('1m')",  "1m"),
        Binding("2", "set_tf('5m')",  "5m"),
        Binding("3", "set_tf('15m')", "15m"),
        Binding("4", "set_tf('30m')", "30m"),
        Binding("5", "set_tf('1h')",  "1h"),
        Binding("i", "toggle_indicators", "Indicators"),
    ]

    TITLE     = "Terminal Quant Suite"
    SUB_TITLE = "PAPER MODE"

    # ── Constructor ────────────────────────────────────────────────────────────

    def __init__(
        self,
        initial_equity: float | None = None,
        mode_override: str | None = None,
        symbol_override: str | None = None,
        timeframe_override: str | None = None,
        bybit_env: str | None = None,
        bybit_category: str | None = None,
        # Quantum params
        quantum_model_path: str | None = None,
        q_confidence: float = 0.65,
        q_leverage: float = 10.0,
        q_pos_frac: float = 0.5,
        q_tp_mult: float = 4.0,
        q_sl_mult: float = 1.0,
        q_live_train: bool = False,
        q_swap_margin: float = 0.03,
        q_min_eval_trades: int = 10,
        q_daily_loss_limit: float = 0.05,   # halt if equity drops >5% from session start
        q_regime_exit_conf: float = 0.0,    # regime-flip early exit threshold (0=disabled)
        q_trail_after: float = 0.0,         # trailing stop activation in ATR units (0=disabled)
        q_trail_dist:  float = 1.0,         # trailing stop distance from watermark in ATR units
    ):
        super().__init__()

        # Config
        config_path = "configs/default.yaml"
        if not os.path.exists(config_path):
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "..", "configs", "default.yaml",
            )
        self.cfg = load_config(config_path)
        if initial_equity  is not None: self.cfg["app"]["initial_equity"] = initial_equity
        if mode_override   is not None: self.cfg["app"]["mode"]           = mode_override
        if symbol_override is not None: self.cfg["app"]["symbol"]         = symbol_override
        if timeframe_override is not None: self.cfg["app"]["timeframe"]   = timeframe_override
        if bybit_env      is not None: self.cfg.setdefault("bybit", {})["env"]      = bybit_env
        if bybit_category is not None: self.cfg.setdefault("bybit", {})["category"] = bybit_category
        self.mode = self.cfg["app"]["mode"]

        # Bybit API keys (real orders in por mode)
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self._bybit_api_key    = os.getenv("BYBIT_API_KEY", "")
        self._bybit_api_secret = os.getenv("BYBIT_API_SECRET", "")

        # Data client — always live streaming (no TTL cache)
        bybit_cfg = self.cfg.get("bybit", {})
        self.client = DataClient(
            mode=self.mode,
            cache_ttl=0,
            bybit_env=bybit_cfg.get("env", "mainnet"),
            bybit_category=bybit_cfg.get("category", "linear"),
        )
        self.data_source = self.client.get_data_source()

        # Initial data load (chart warmup — last 2 days)
        self.df = self.client.fetch_ohlcv(
            self.cfg["app"]["symbol"],
            self.cfg["app"]["timeframe"],
            days_back=2,
        )
        self.sim_idx = max(0, len(self.df) - 1)

        # Merge Binance real taker CVD for feature consistency with training
        if fetch_binance_taker_history is not None and not self.df.empty and "ts" in self.df.columns:
            try:
                import datetime as _dt
                _sym = self.cfg["app"]["symbol"]
                _tf  = self.cfg["app"]["timeframe"]
                _now       = _dt.datetime.now(_dt.timezone.utc)
                _end_str   = _now.strftime("%Y-%m-%d")
                _start_str = (_now - _dt.timedelta(days=3)).strftime("%Y-%m-%d")
                _tc = f"data/binance_taker_{_sym}_{_tf}_{_start_str.replace('-','')}_{_end_str.replace('-','')}.csv"
                _df_t = fetch_binance_taker_history(
                    symbol=_sym, interval=_tf,
                    start_date=_start_str, end_date=_end_str,
                    cache_path=_tc, verbose=False)
                if not _df_t.empty:
                    self.df = self.df.merge(_df_t[["ts", "taker_buy_volume"]], on="ts", how="left")
                    self.df["taker_buy_volume"] = self.df["taker_buy_volume"].fillna(0.0)
            except Exception:
                pass  # silent fallback to OHLCV CVD

        # Live data state
        self._ws_latest_df              = None
        self._ws_update_count           = 0
        self._ws_last_ts                = ""
        self._last_ws_update_monotonic  = 0.0
        self._last_rest_poll            = 0.0
        self._last_live_refresh         = 0.0
        self._ws_inactive_threshold     = 15.0
        self._rest_days_back            = 1
        self._last_bar_ts_dt            = None
        self._por_initialized           = False
        self._por_last_ts               = None

        # Private account + orderbook
        self._priv_state                 = {}
        self._priv_ok                    = False
        self._priv_err                   = ""
        self._priv_last_ts               = ""
        self._priv_last_update_monotonic = 0.0
        self._last_priv_poll             = 0.0
        self._last_ob_poll               = 0.0
        self._orderbook                  = {"bids": [], "asks": [], "ts": ""}
        self._priv_trade_logger          = PrivateTradeLogger()

        # Real trading control
        self._trading_active      = False   # True = real Bybit orders (por mode only)
        self._last_pos_sync       = 0.0
        self._pos_sync_interval   = 10.0
        self._q_daily_loss_limit  = q_daily_loss_limit
        self._q_regime_exit_conf  = q_regime_exit_conf
        self._q_trail_after       = q_trail_after
        self._q_trail_dist        = q_trail_dist
        self._q_session_start_eq  = None
        # Bybit instrument lot-size rules (fetched once on startup in por mode)
        self._bybit_qty_step  = 0.001   # default for BTCUSDT
        self._bybit_min_qty   = 0.001
        # WebSocket reconnect cooldown
        self._ws_was_inactive      = False   # True when WS was inactive last tick
        self._ws_entry_cooldown    = 0       # bars to skip after WS recovery

        # Storage
        self.storage = Storage("data/tui_trades.db")

        # UI state
        self._show_indicators = False

        # ── Quantum state ──────────────────────────────────────────────────────
        self._q_agent       = None      # Champion A (frozen, trades)
        self._q_agent_b     = None      # Challenger B (learning, shadow)
        self._q_device      = None
        self._q_confidence  = q_confidence
        self._q_leverage    = q_leverage
        self._q_pos_frac    = q_pos_frac
        self._q_tp_mult     = q_tp_mult
        self._q_sl_mult     = q_sl_mult
        self._q_live_train  = q_live_train

        _init_eq = float(self.cfg["app"].get("initial_equity") or 10000.0)
        # In por mode: fetch actual Bybit equity; manual value is ignored
        if self.mode == "por" and self._bybit_api_key and self._bybit_api_secret:
            try:
                _bybit_eq = self._fetch_bybit_equity()
                if _bybit_eq is not None:
                    print(f"[por] Bybit equity loaded: ${_bybit_eq:,.2f} USDT")
                    _init_eq = _bybit_eq
            except Exception as _e:
                print(f"[por] Could not load Bybit equity ({_e}), using {_init_eq}")
        self._q_equity      = _init_eq
        self._q_pos         = None      # A's virtual position
        self._q_shadow_pos  = None      # B's shadow position
        self._q_trades: list = []
        self._q_total_trades = 0
        self._q_wins         = 0
        self._q_perf_a = {"wins": 0, "trades": 0}
        self._q_perf_b = {"wins": 0, "trades": 0}
        self._q_swap_count      = 0
        self._q_swap_margin     = q_swap_margin
        self._q_min_eval_trades = q_min_eval_trades
        self._q_next_swap_time  = None
        self._q_last_swap_info  = "No eval yet"
        self._q_feat_buf        = None
        self._q_last_action     = 0
        self._q_last_prob       = 0.0
        self._q_last_probs      = None
        self._q_status          = "Quantum: not loaded"
        self._q_last_bar_ts     = None

        # ChatGPT challenge metrics
        self._q_peak_equity   = _init_eq
        self._q_start_equity  = _init_eq
        self._q_max_drawdown  = 0.0
        self._q_start_time    = datetime.now(timezone.utc)
        self._q_pnl_list: list = []

        # Load quantum agent
        if QUANTUM_AVAILABLE and quantum_model_path:
            try:
                import copy as _copy
                import torch
                self._q_device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                cfg_q = AgentConfig(confidence_threshold=0.0)
                self._q_agent = build_quantum_agent(
                    config=cfg_q,
                    device=self._q_device,
                    checkpoint_path=quantum_model_path,
                )
                self._q_agent.eval()                         # A stays frozen
                self._q_agent_b = _copy.deepcopy(self._q_agent)
                self._q_agent_b.train()                      # B in train mode
                self._q_feat_buf  = _deque_q(maxlen=200)
                # OI 버퍼: ts_ms → open_interest 매핑 (최근 200봉)
                self._q_oi_buf    = _deque_q(maxlen=200)   # (ts_ms, oi) tuples
                self._q_oi_last_fetch_ms = 0               # 마지막 OI 갱신 시각
                # bc_scaler 로드 (BC와 동일한 피처 분포 유지)
                import os as _os, pickle as _pkl
                _scaler_path = _os.path.join(
                    _os.path.dirname(quantum_model_path), "bc_scaler.pkl"
                )
                self._q_scaler = None
                if _os.path.isfile(_scaler_path):
                    with open(_scaler_path, "rb") as _sf:
                        self._q_scaler = _pkl.load(_sf)
                    logger.info(f"bc_scaler loaded: {_scaler_path}")
                else:
                    logger.warning(f"bc_scaler not found: {_scaler_path} — raw features used")
                _now_utc = datetime.now(timezone.utc)
                self._q_next_swap_time = (
                    _now_utc + timedelta(days=1)
                ).replace(hour=0, minute=0, second=0, microsecond=0)
                self._q_status = f"Quantum A+B: loaded ({quantum_model_path})"
                logger.info(f"Quantum A+B loaded: {quantum_model_path}")
            except Exception as e:
                self._q_status = f"Quantum: load failed ({e})"
                logger.warning(f"Quantum agent load failed: {e}")

    # ── TUI lifecycle ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-area"):
            yield WatchlistPane("Loading...", id="left-pane")
            yield ChartPane("Loading...", id="center-pane")
            yield QuantPane("Loading...", id="right-pane")
        yield StatusBar("Initializing...", id="status-bar")
        yield Footer()

    def on_mount(self):
        self.client.start_kline_stream(
            self.cfg["app"]["symbol"],
            self.cfg["app"]["timeframe"],
            self._on_ws_update,
        )
        if self.mode == "por":
            self.client.start_private_stream(self._on_priv_update)
            # Fetch Bybit lot-size rules once (no auth needed)
            self._fetch_instrument_info()
        self.set_interval(self.cfg["app"]["refresh_seconds"], self.tick)
        self.refresh_all()

    def tick(self):
        self._refresh_live_data()

        # ── WebSocket reconnect detection ──────────────────────────────────────
        _ws_inactive_now = (
            time.monotonic() - self._last_ws_update_monotonic
        ) > self._ws_inactive_threshold
        if _ws_inactive_now:
            self._ws_was_inactive = True
        elif self._ws_was_inactive:
            # WS just recovered after being inactive → impose entry cooldown
            self._ws_entry_cooldown = 2
            self._ws_was_inactive   = False
            logger.info("[WS] Reconnected — entry cooldown 2 bars")

        if self.mode == "por":
            self._refresh_private_account()
            self._update_orderbook()
            # Sync with actual Bybit position every _pos_sync_interval seconds
            if self._trading_active:
                now = time.monotonic()
                if now - self._last_pos_sync >= self._pos_sync_interval:
                    self._last_pos_sync = now
                    self._sync_bybit_position()

        # Always point to last bar (live mode only)
        if self.df is not None and len(self.df) > 0:
            self.sim_idx = len(self.df) - 1
            self._quantum_step()

        self.refresh_all()

    def refresh_all(self):
        self._update_left_pane()
        self._update_chart()
        if self._q_agent is not None:
            self._update_quantum_pane()
        self._update_status()

    # ── Left pane — account + position ────────────────────────────────────────

    def _update_left_pane(self):
        pane = self.query_one("#left-pane", WatchlistPane)
        lines = []
        sym = self.cfg["app"]["symbol"]
        tf  = self.cfg["app"]["timeframe"]

        # Price
        price = None
        if self.df is not None and len(self.df) > 0:
            price = float(self.df.iloc[-1]["close"])
        lines.append(f"[b]{sym}[/b]  {tf}")
        if price:
            lines.append(f"  [b]{price:,.2f}[/b]")
        lines.append("")

        # Mode + trading state
        if self.mode == "por":
            t_state = (
                "[green]ACTIVE[/green]" if self._trading_active
                else "[dim]PAUSED[/dim]"
            )
            lines.append(f"[b]LIVE MODE[/b]  Trade: {t_state}")
            if self._trading_active:
                lines.append("  [yellow]b=stop  x=close+stop[/yellow]")
            else:
                lines.append("  [dim]b=start trading[/dim]")
        else:
            lines.append("[b]PAPER MODE[/b]  (practice)")
        lines.append("")

        # Account
        lines.append("[b]ACCOUNT[/b]")
        lines.append("-" * 26)
        if self.mode == "por" and self._priv_state.get("balances"):
            bal = self._priv_state["balances"]
            lines.append(f"  Equity:   {float(bal.get('equity', 0)):,.2f}")
            lines.append(f"  Wallet:   {float(bal.get('wallet_balance', 0)):,.2f}")
            lines.append(f"  Avail:    {float(bal.get('available_balance', 0)):,.2f}")
            lines.append(f"  Curr:     {bal.get('currency', 'USDT')}")
        else:
            eq_c = "green" if self._q_equity >= self._q_start_equity else "red"
            lines.append(
                f"  Equity:   [{eq_c}]{self._q_equity:,.2f}[/{eq_c}]"
            )
            lines.append(f"  Trades:   {self._q_total_trades}")
            wr = self._q_wins / self._q_total_trades * 100 if self._q_total_trades else 0
            wr_c = "green" if wr > 23.2 else "red"
            lines.append(f"  WinRate:  [{wr_c}]{wr:.0f}%[/{wr_c}]")
        lines.append("")

        # Position (real in por, virtual in paper)
        lines.append("[b]POSITION[/b]")
        lines.append("-" * 26)
        priv_pos = None
        if self.mode == "por":
            for p in self._priv_state.get("positions", []):
                if float(p.get("size", 0) or 0) != 0:
                    priv_pos = p
                    break

        if priv_pos:
            side  = "[green]LONG[/green]" if priv_pos.get("side") == "Buy" else "[red]SHORT[/red]"
            size  = abs(float(priv_pos.get("size", 0) or 0))
            entry = float(priv_pos.get("avgPrice", 0) or 0)
            upnl  = float(priv_pos.get("unrealisedPnl", 0) or 0)
            uc    = "green" if upnl >= 0 else "red"
            lines.append(f"  {side} {size:.4f}")
            lines.append(f"  Entry: {entry:,.2f}")
            lines.append(f"  UPnL:  [{uc}]{upnl:+,.2f}[/{uc}]")
            lev = priv_pos.get("leverage", "N/A")
            lines.append(f"  Lev:   {lev}x")
        elif self._q_pos is not None:
            p     = self._q_pos
            side_c = "green" if p["side"] == "long" else "red"
            cur_p  = price or p["entry"]
            ret_pct = (
                (cur_p - p["entry"]) / p["entry"] * 100
                if p["side"] == "long"
                else (p["entry"] - cur_p) / p["entry"] * 100
            )
            upnl = ret_pct / 100 * p["notional"]
            uc   = "green" if upnl >= 0 else "red"
            lines.append(
                f"  [{side_c}]{p['side'].upper()}[/{side_c}]"
                f"  @{p['entry']:,.0f}"
            )
            lines.append(f"  UPnL:  [{uc}]{upnl:+.1f}[/{uc}] ({ret_pct:+.2f}%)")
            lines.append(f"  TP: [green]{p['tp_price']:,.0f}[/green]")
            lines.append(f"  SL: [red]{p['sl_price']:,.0f}[/red]")
        else:
            lines.append("  [dim]Flat[/dim]")

        lines.append("")

        # Recent quantum trades
        lines.append("[b]RECENT TRADES[/b]")
        lines.append("-" * 26)
        recent = self._q_trades[-5:]
        if not recent:
            lines.append("  [dim]None yet[/dim]")
        for t in reversed(recent):
            sc = "green" if t["side"] == "long" else "red"
            ec = "green" if t["exit_type"] == "TP" else "red"
            pc = "green" if t["pnl"] >= 0 else "red"
            lines.append(
                f"  [{sc}]{t['side'][:1].upper()}[/{sc}]"
                f"[{ec}]{t['exit_type']}[/{ec}]"
                f" [{pc}]{t['pnl']:+.1f}[/{pc}]"
            )

        if self.mode == "por" and self._priv_state.get("fills"):
            lines.append("")
            lines.append("[b]RECENT FILLS[/b]")
            lines.append("-" * 26)
            for f in self._priv_state["fills"][:4]:
                side = f.get("side", "")[:3].lower()
                c    = "green" if side == "buy" else "red"
                qty  = float(f.get("execQty", 0) or 0)
                px   = float(f.get("execPrice", 0) or 0)
                lines.append(f"  [{c}]{side:3s}[/{c}] {qty:.4f} @{px:,.0f}")

        pane.update("\n".join(lines))

    # ── Center pane — live candle chart ───────────────────────────────────────

    def _update_chart(self):
        pane = self.query_one("#center-pane", ChartPane)
        if self.df is None or len(self.df) == 0:
            pane.update("[dim]Waiting for live data...[/dim]")
            return

        window = self.df.tail(50)
        row    = self.df.iloc[-1]
        sym    = self.cfg["app"]["symbol"]
        tf     = self.cfg["app"]["timeframe"]
        hi     = float(window["high"].max())
        lo     = float(window["low"].min())
        last   = float(row["close"])
        bybit_cfg = self.cfg.get("bybit", {})
        env  = bybit_cfg.get("env", "mainnet").upper()
        cat  = bybit_cfg.get("category", "linear").upper()

        lines = [
            f"[b]{sym}[/b]  TF:{tf}  Src:{self.data_source}  Env:{env}  Cat:{cat}  "
            f"Bars:{len(self.df)}  H:{hi:,.2f}  L:{lo:,.2f}  Last:[b]{last:,.2f}[/b]",
            "-" * 55,
            ascii_candle_chart(window, 50, 10),
            time_axis_kst(window, 50),
            f"  Vol: {sparkline(window['volume'].tolist(), 50)}",
            "",
        ]

        lines.extend(orderbook_summary(self._orderbook))

        # ── Signal Preview (shown before real trading starts) ─────────────────
        if not self._trading_active and self._q_agent is not None:
            # Check if halted by daily loss limit
            _daily_halted = False
            if self._q_session_start_eq and self._q_session_start_eq > 0:
                _loss_pct = (self._q_session_start_eq - self._q_equity) / self._q_session_start_eq
                if _loss_pct >= self._q_daily_loss_limit:
                    _daily_halted = True

            _header = (
                "[b]── SIGNAL PREVIEW ──[/b]  [red][b]DAILY LOSS LIMIT HIT — HALTED[/b][/red]"
                if _daily_halted
                else "[b]── SIGNAL PREVIEW ──[/b]  [dim](press [b]b[/b] to execute real orders)[/dim]"
            )
            lines += ["", _header]
            lines.append("-" * 55)

            # Daily loss meter
            if self._q_session_start_eq and self._q_session_start_eq > 0:
                _lp   = (self._q_session_start_eq - self._q_equity) / self._q_session_start_eq
                _lc   = "red" if _lp >= self._q_daily_loss_limit * 0.8 else "yellow"
                lines.append(
                    f"  Daily loss: [{_lc}]{_lp:.1%}[/{_lc}]"
                    f"  limit: {self._q_daily_loss_limit:.0%}"
                    f"  start: ${self._q_session_start_eq:,.0f}"
                )

            action  = self._q_last_action    # 0=HOLD 1=LONG 2=SHORT
            prob    = self._q_last_prob
            probs   = self._q_last_probs     # [p_hold, p_long, p_short] or None

            # Probability bar
            if probs is not None:
                try:
                    _ph, _pl, _ps = float(probs[0]), float(probs[1]), float(probs[2])
                    lines.append(
                        f"  Probs  H:[yellow]{_ph:.2f}[/yellow]"
                        f"  L:[green]{_pl:.2f}[/green]"
                        f"  S:[red]{_ps:.2f}[/red]"
                    )
                except Exception:
                    pass

            # ATR-based TP/SL
            window_atr = self.df.tail(21)
            atr_frac   = self._atr_pct(window_atr) or 0.01
            tp_pct     = self._q_tp_mult * atr_frac
            sl_pct     = self._q_sl_mult * atr_frac
            rr         = tp_pct / (sl_pct + 1e-10)

            _cur_pos   = self._q_pos is not None

            if action == 0 or prob < self._q_confidence:
                _reason = (
                    f"prob {prob:.2f} < threshold {self._q_confidence}"
                    if action != 0 else "HOLD"
                )
                lines.append(f"  Signal:  [dim]No trade — {_reason}[/dim]")
            elif rr < 3.0:
                lines.append(f"  Signal:  [dim]No trade — R:R {rr:.1f} < 3.0[/dim]")
            elif _cur_pos:
                lines.append(f"  Signal:  [dim]Waiting — position already open[/dim]")
            else:
                _side     = "LONG" if action == 1 else "SHORT"
                _sc       = "green" if _side == "LONG" else "red"
                _notional = self._q_equity * self._q_pos_frac * self._q_leverage
                _fee_in   = _notional * 0.0002
                _fee_out  = _notional * 0.00055
                _total_fee= _fee_in + _fee_out

                _tp_price = last * (1 + tp_pct) if _side == "LONG" else last * (1 - tp_pct)
                _sl_price = last * (1 - sl_pct) if _side == "LONG" else last * (1 + sl_pct)
                _win_pnl  = _notional * tp_pct - _total_fee
                _loss_pnl = -(_notional * sl_pct + _total_fee)

                lines.append(
                    f"  Signal:  [{_sc}][b]{_side}[/b][/{_sc}]"
                    f"  Conf: [b]{prob:.3f}[/b]  R:R: {rr:.1f}x"
                )
                lines.append(
                    f"  Entry:   [b]{last:,.2f}[/b]"
                    f"  Notional: ${_notional:,.0f}"
                    f"  (×{self._q_leverage:.0f} lev)"
                )
                lines.append(
                    f"  TP:  [green]{_tp_price:,.2f}[/green]"
                    f"  ({tp_pct*100:.2f}%)  "
                    f"Win PnL: [green]+${_win_pnl:,.2f}[/green]"
                )
                lines.append(
                    f"  SL:  [red]{_sl_price:,.2f}[/red]"
                    f"  ({sl_pct*100:.2f}%)  "
                    f"Loss PnL: [red]-${abs(_loss_pnl):,.2f}[/red]"
                )
                lines.append(
                    f"  Fee:  ${_total_fee:,.2f} total"
                    f"  ATR: {atr_frac*100:.3f}%"
                )
                # EV estimate based on historical WR
                _n = self._q_total_trades
                if _n >= 5:
                    _wr = self._q_wins / _n
                    _ev  = _wr * _win_pnl + (1 - _wr) * _loss_pnl
                    _evc = "green" if _ev >= 0 else "red"
                    lines.append(
                        f"  EV(WR={_wr:.0%}): [{_evc}]${_ev:+,.2f}[/{_evc}]  per trade"
                    )

        if self._show_indicators:
            lines += ["", "[b]BAR DETAIL[/b]", "-" * 55]

            def f(v, d=2):
                return f"{v:,.{d}f}" if v == v else "---"

            lines.append(f"  Close:  {f(row['close'])}")
            if "deep_candle" in row.index:
                lines.append(f"  DeepC:  {f(row.get('deep_candle', float('nan')), 3)}")
            if "liq_heatmap" in row.index:
                liq = float(row.get("liq_heatmap", 0))
                lines.append(f"  LiqMap: {liq:,.0f}")
            # ATR inline (no external indicator needed)
            atr_val = self._atr_pct(window)
            if atr_val:
                lines.append(f"  ATR%:   {atr_val:.4f}  ({atr_val*100:.2f}%)")

        pane.update("\n".join(lines))

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _update_status(self):
        bar = self.query_one("#status-bar", StatusBar)
        now_kst  = datetime.now(timezone(timedelta(hours=9)))
        now_str  = now_kst.strftime("%Y-%m-%d %H:%M")
        bar_ts   = str(self.df.iloc[-1].get("ts", "")) if self.df is not None and len(self.df) > 0 else "N/A"
        bar_str  = _format_ts_kst_full(bar_ts)
        mode_str = f"[yellow]{self.mode.upper()}[/yellow]"

        # Staleness check
        stale = ""
        if self.mode == "por":
            bar_dt = _parse_ts_utc(bar_ts)
            if bar_dt is None:
                stale = " [red]STALE[/red]"
            else:
                delta = (datetime.now(timezone.utc) - bar_dt).total_seconds()
                if delta > 2 * _timeframe_seconds(self.cfg["app"]["timeframe"]):
                    stale = " [red]STALE[/red]"

        ws_str = ""
        if self.mode == "por":
            ws_str = f" | WS:{self._ws_update_count}"

        trade_str = ""
        if self.mode == "por":
            trade_str = (
                " | [green]TRADING[/green]" if self._trading_active
                else " | [dim]PAUSED[/dim]"
            )

        q_str = ""
        if self._q_agent is not None:
            q_str = f" | Q:{self._q_status[:40]}"

        bar.update(
            f" {mode_str} | NOW:{now_str} | BAR:{bar_str}{stale}"
            f"{ws_str}{trade_str}{q_str}"
        )

    # ── Actions ────────────────────────────────────────────────────────────────

    def action_quit(self):
        self.client.stop_stream()
        self.client.stop_private_stream()
        self.exit()

    def action_kill_switch(self):
        """Emergency: stop trading + close all positions."""
        self._trading_active = False
        if self.mode == "por":
            self._close_bybit_position()
        if self._q_pos is not None:
            self._q_pos = None
        self.notify("KILL SWITCH — trading halted, positions closed", severity="error")
        self.refresh_all()

    def _validate_por_ready(self) -> tuple[bool, str]:
        """Pre-flight checks before enabling real trading.
        Returns (ok, message).
        """
        # 1. API keys present
        if not self._bybit_api_key or not self._bybit_api_secret:
            return False, "No BYBIT_API_KEY / BYBIT_API_SECRET in .env"

        # 2. API key valid (fetch server time via signed endpoint)
        try:
            r = _bybit_signed(
                "GET", "/v5/account/wallet-balance",
                {"accountType": "UNIFIED"},
                self._bybit_api_key, self._bybit_api_secret,
            )
            if r.get("retCode") != 0:
                return False, f"API auth failed: {r.get('retMsg', 'unknown')}"
        except Exception as e:
            return False, f"API connectivity error: {e}"

        # 3. Equity sufficient for minimum order
        #    Bybit BTCUSDT min qty = 0.001 BTC
        symbol  = self.cfg["app"]["symbol"]
        min_qty = 0.001   # BTC default
        price   = float(self.df.iloc[-1]["close"]) if self.df is not None and len(self.df) > 0 else 0
        if price <= 0:
            return False, "No live price available — wait for data"
        notional  = self._q_equity * self._q_pos_frac * self._q_leverage
        qty       = self._round_qty(notional / price)
        min_qty   = self._bybit_min_qty
        if qty < min_qty:
            return False, (
                f"Computed qty {qty:.4f} < min {min_qty} BTC. "
                f"Need equity > ${min_qty * price / (self._q_pos_frac * self._q_leverage):,.0f}"
            )

        return True, (
            f"OK — equity ${self._q_equity:,.2f}, "
            f"order qty ~{qty:.3f} {symbol[:3]}, "
            f"notional ~${notional:,.0f}"
        )

    def action_start_trading(self):
        """Enable autonomous trading (real orders in por, already running in paper)."""
        if self.mode == "por":
            ok, msg = self._validate_por_ready()
            if not ok:
                self.notify(f"Pre-flight FAILED: {msg}", severity="error")
                return
            self._trading_active     = True
            self._q_session_start_eq = self._q_equity   # reset daily loss baseline
            self.notify(
                f"AUTO TRADING STARTED ✓ {msg} — "
                f"daily loss limit {self._q_daily_loss_limit:.0%}",
                severity="warning",
            )
        else:
            self.notify("Paper mode: quantum model is always active", severity="information")
        self.refresh_all()

    def action_stop_trading(self):
        """Disable trading and close all open positions."""
        self._trading_active = False
        if self.mode == "por":
            ok = self._close_bybit_position()
            msg = "Positions closed" if ok else "No open positions (or API error)"
        else:
            msg = "Paper mode paused"
        self.notify(f"Trading stopped. {msg}", severity="information")
        self.refresh_all()

    def action_set_tf(self, tf: str):
        old = self.cfg["app"]["timeframe"]
        self.cfg["app"]["timeframe"] = tf
        self.notify(f"Timeframe: {old} → {tf}")
        self.client.stop_stream()
        self.client.start_kline_stream(
            self.cfg["app"]["symbol"],
            tf,
            self._on_ws_update,
        )
        self._ws_latest_df         = None
        self._ws_update_count      = 0
        self._ws_last_ts           = ""
        self._last_ws_update_monotonic = 0.0
        self._por_initialized      = False
        self._por_last_ts          = None
        self.client.clear_cache()
        self.df = self.client.fetch_ohlcv(
            self.cfg["app"]["symbol"], tf, days_back=2
        )
        self.sim_idx = max(0, len(self.df) - 1)
        self.refresh_all()

    def action_toggle_indicators(self):
        self._show_indicators = not self._show_indicators
        self.notify(f"Indicators: {'ON' if self._show_indicators else 'OFF'}")
        self.refresh_all()

    # ── Bybit real order execution ─────────────────────────────────────────────

    def _place_bybit_order(
        self,
        side: str,        # "Buy" or "Sell"
        qty: float,
        tp_price: float,
        sl_price: float,
    ) -> bool:
        """Place a market order with TP/SL on Bybit."""
        if not self._bybit_api_key or not self._bybit_api_secret:
            self.notify("No Bybit API keys — check .env", severity="error")
            return False
        symbol   = self.cfg["app"]["symbol"]
        category = self.cfg.get("bybit", {}).get("category", "linear")
        # Floor qty to Bybit step precision (never round up)
        qty = self._round_qty(qty)
        if qty < self._bybit_min_qty:
            self.notify(
                f"Order skipped: qty {qty:.4f} < min {self._bybit_min_qty} — "
                f"increase equity or reduce leverage",
                severity="error",
            )
            return False

        # Determine decimal places from step size
        _step_str = f"{self._bybit_qty_step:.10f}".rstrip("0")
        _decimals = len(_step_str.split(".")[-1]) if "." in _step_str else 0
        _qty_str  = f"{qty:.{_decimals}f}"

        params = {
            "category":    category,
            "symbol":      symbol,
            "side":        side,
            "orderType":   "Market",
            "qty":         _qty_str,
            "takeProfit":  f"{tp_price:.1f}",
            "stopLoss":    f"{sl_price:.1f}",
            "tpTriggerBy": "LastPrice",
            "slTriggerBy": "LastPrice",
            "tpslMode":    "Full",
        }
        result = _bybit_signed("POST", "/v5/order/create", params,
                               self._bybit_api_key, self._bybit_api_secret)
        if result.get("retCode") == 0:
            logger.info(
                f"Order OK: {side} {qty:.3f} {symbol} "
                f"TP={tp_price:.1f} SL={sl_price:.1f}"
            )
            self.notify(
                f"Order: {side} {qty:.3f} {symbol}  "
                f"TP:{tp_price:,.0f}  SL:{sl_price:,.0f}",
                severity="information",
            )
            return True
        err = result.get("retMsg", "Unknown")
        logger.warning(f"Order FAILED: {err}")
        self.notify(f"Order FAILED: {err[:50]}", severity="error")
        return False

    def _close_bybit_position(self) -> bool:
        """Close all open positions for the current symbol via market order."""
        if not self._bybit_api_key or not self._bybit_api_secret:
            return False
        symbol   = self.cfg["app"]["symbol"]
        category = self.cfg.get("bybit", {}).get("category", "linear")

        pos_result = _bybit_signed(
            "GET", "/v5/position/list",
            {"category": category, "symbol": symbol},
            self._bybit_api_key, self._bybit_api_secret,
        )
        if pos_result.get("retCode") != 0:
            logger.warning(f"Position query failed: {pos_result.get('retMsg')}")
            return False

        closed_any = False
        for pos in pos_result.get("result", {}).get("list", []):
            size = float(pos.get("size", 0) or 0)
            if size == 0:
                continue
            close_side = "Sell" if pos.get("side") == "Buy" else "Buy"
            close_params = {
                "category":   category,
                "symbol":     symbol,
                "side":       close_side,
                "orderType":  "Market",
                "qty":        str(size),
                "reduceOnly": True,
            }
            r = _bybit_signed("POST", "/v5/order/create", close_params,
                               self._bybit_api_key, self._bybit_api_secret)
            if r.get("retCode") == 0:
                logger.info(f"Closed: {close_side} {size} {symbol}")
                closed_any = True
            else:
                logger.warning(f"Close failed: {r.get('retMsg')}")

        return closed_any

    # ── Live data methods ──────────────────────────────────────────────────────

    def _on_ws_update(self, df):
        self._ws_latest_df = df
        self._ws_update_count += 1
        self._last_ws_update_monotonic = time.monotonic()
        try:
            if df is not None and not df.empty and "ts" in df.columns:
                self._ws_last_ts = str(df.iloc[-1]["ts"])
        except Exception:
            pass

    def _get_last_ts_dt(self, df):
        if df is None or df.empty or "ts" not in df.columns:
            return None
        return _parse_ts_utc(df.iloc[-1]["ts"])

    def _merge_df(self, df_new):
        import pandas as pd
        if df_new is None or df_new.empty:
            return
        if self.df is None or self.df.empty:
            last_new = self._get_last_ts_dt(df_new)
            if self.mode in ("por", "live") and self._is_stale_ts(last_new):
                return
            self.df = df_new
            return

        last_current = self._get_last_ts_dt(self.df)
        last_new     = self._get_last_ts_dt(df_new)
        if last_new is None:
            return
        if last_current and last_new < last_current:
            return
        if self.mode in ("por", "live") and self._is_stale_ts(last_new):
            return

        df_all = pd.concat([self.df, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["ts"], keep="last")
        df_all = df_all.sort_values("ts").reset_index(drop=True)
        self.df = df_all

    def _is_stale_ts(self, ts_dt) -> bool:
        if ts_dt is None:
            return True
        delta = (datetime.now(timezone.utc) - ts_dt).total_seconds()
        return delta > 2 * _timeframe_seconds(self.cfg["app"]["timeframe"])

    def _refresh_live_data(self):
        now = time.monotonic()
        self._apply_ws_update()
        refresh_interval = max(0.5, self.cfg["app"]["refresh_seconds"])
        if now - self._last_live_refresh < refresh_interval:
            return
        self._last_live_refresh = now
        ws_inactive = (now - self._last_ws_update_monotonic) > self._ws_inactive_threshold
        if not ws_inactive and self._ws_update_count > 0:
            return
        if now - self._last_rest_poll < max(1.0, refresh_interval):
            return
        self._last_rest_poll = now
        df = self.client.fetch_ohlcv(
            self.cfg["app"]["symbol"],
            self.cfg["app"]["timeframe"],
            days_back=self._rest_days_back,
        )
        if df is not None and len(df) > 0:
            self._merge_df(df)

    def _apply_ws_update(self):
        if self._ws_latest_df is None:
            return
        df_new = self._ws_latest_df
        self._ws_latest_df = None
        if df_new is None or df_new.empty:
            return
        self._merge_df(df_new)

    def _update_orderbook(self):
        if self.mode != "por":
            return
        now = time.monotonic()
        if now - self._last_ob_poll < 2.0:
            return
        self._last_ob_poll = now
        self._orderbook = self.client.fetch_orderbook(
            self.cfg["app"]["symbol"], limit=100
        )

    def _on_priv_update(self, payload: dict):
        self._priv_last_update_monotonic = time.monotonic()
        self._priv_ok  = True
        self._priv_last_ts = datetime.now(
            timezone(timedelta(hours=9))
        ).strftime("%H:%M:%S")
        self._priv_state["ws"] = payload

    def _refresh_private_account(self):
        if self.mode != "por":
            return
        now = time.monotonic()
        ws_inactive = (now - self._priv_last_update_monotonic) > 15.0
        if not ws_inactive and self._priv_ok and self._priv_state.get("balances"):
            return
        if now - self._last_priv_poll < 23.0:
            return
        self._last_priv_poll = now
        snapshot = self.client.fetch_account_snapshot(self.cfg["app"]["symbol"])
        if snapshot.get("error"):
            self._priv_ok  = False
            self._priv_err = "Auth failed"
            return
        self._priv_state = snapshot
        try:
            self._priv_trade_logger.update(snapshot)
        except Exception:
            pass
        self._priv_ok  = True
        self._priv_last_ts = datetime.now(
            timezone(timedelta(hours=9))
        ).strftime("%H:%M:%S")

    def _fetch_bybit_equity(self) -> float | None:
        """Fetch UNIFIED wallet totalEquity (USDT) from Bybit V5 REST.
        Returns None on failure or if API keys are not set."""
        if not self._bybit_api_key or not self._bybit_api_secret:
            return None
        try:
            result = _bybit_signed(
                "GET",
                "/v5/account/wallet-balance",
                {"accountType": "UNIFIED"},
                self._bybit_api_key,
                self._bybit_api_secret,
            )
            if result.get("retCode") != 0:
                return None
            acct_list = result.get("result", {}).get("list", [])
            if not acct_list:
                return None
            total_eq = float(acct_list[0].get("totalEquity") or 0)
            return total_eq if total_eq > 0 else None
        except Exception as e:
            logger.warning(f"_fetch_bybit_equity failed: {e}")
            return None

    def _fetch_instrument_info(self):
        """Fetch lot-size rules for the current symbol from Bybit V5.
        Populates _bybit_qty_step and _bybit_min_qty (no auth required).
        """
        symbol   = self.cfg["app"]["symbol"]
        category = self.cfg.get("bybit", {}).get("category", "linear")
        try:
            from urllib.request import urlopen
            import json as _j
            url = (
                f"https://api.bybit.com/v5/market/instruments-info"
                f"?category={category}&symbol={symbol}"
            )
            with urlopen(url, timeout=8) as resp:
                data = _j.loads(resp.read().decode("utf-8"))
            for item in data.get("result", {}).get("list", []):
                lsf = item.get("lotSizeFilter", {})
                step    = float(lsf.get("qtyStep", 0) or 0)
                min_qty = float(lsf.get("minOrderQty", 0) or 0)
                if step > 0:
                    self._bybit_qty_step = step
                if min_qty > 0:
                    self._bybit_min_qty = min_qty
                logger.info(
                    f"[instrument] {symbol} qtyStep={self._bybit_qty_step} "
                    f"minOrderQty={self._bybit_min_qty}"
                )
                return
        except Exception as e:
            logger.warning(f"_fetch_instrument_info failed: {e} — using defaults")

    def _round_qty(self, qty: float) -> float:
        """Floor qty to Bybit's qtyStep precision (never round up)."""
        import math
        step = self._bybit_qty_step
        if step <= 0:
            step = 0.001
        return math.floor(qty / step) * step

    def _sync_bybit_position(self):
        """Reconcile _q_pos with actual Bybit position.

        Key scenario: Bybit's TP or SL fires → real size = 0 but _q_pos still set.
        We detect this and record the closed trade so equity tracking stays accurate.
        """
        if not self._bybit_api_key or not self._bybit_api_secret:
            return
        symbol   = self.cfg["app"]["symbol"]
        category = self.cfg.get("bybit", {}).get("category", "linear")
        try:
            result = _bybit_signed(
                "GET", "/v5/position/list",
                {"category": category, "symbol": symbol},
                self._bybit_api_key, self._bybit_api_secret,
            )
        except Exception as e:
            logger.warning(f"_sync_bybit_position: API error {e}")
            return

        if result.get("retCode") != 0:
            return

        bybit_size = 0.0
        bybit_side = ""
        bybit_entry = 0.0
        for pos in result.get("result", {}).get("list", []):
            sz = float(pos.get("size", 0) or 0)
            if sz > 0:
                bybit_size  = sz
                bybit_side  = pos.get("side", "")
                bybit_entry = float(pos.get("avgPrice", 0) or 0)
                break

        if self._q_pos is not None and bybit_size == 0:
            # Bybit position closed (TP/SL hit) without our tick catching it
            p     = self._q_pos
            price = float(self.df.iloc[-1]["close"]) if self.df is not None else p["entry"]
            dist_tp = abs(price - p["tp_price"])
            dist_sl = abs(price - p["sl_price"])
            if dist_tp <= dist_sl:
                exit_type = "TP"
                pnl = p["notional"] * p["tp_pct"] - p["notional"] * 0.00055
            else:
                exit_type = "SL"
                pnl = -(p["notional"] * p["sl_pct"] + p["notional"] * 0.00055)

            self._q_equity       += pnl
            self._q_total_trades += 1
            if pnl > 0:
                self._q_wins += 1
            self._q_perf_a["trades"] += 1
            if pnl > 0:
                self._q_perf_a["wins"] += 1
            if self._q_equity > self._q_peak_equity:
                self._q_peak_equity = self._q_equity
            dd = (self._q_peak_equity - self._q_equity) / (self._q_peak_equity + 1e-10)
            if dd > self._q_max_drawdown:
                self._q_max_drawdown = dd
            ts = str(self.df.iloc[-1].get("ts", "")) if self.df is not None else ""
            rec = {
                "side": p["side"], "entry": p["entry"], "exit": price,
                "pnl": round(pnl, 4), "exit_type": f"{exit_type}(bybit)", "ts": ts,
            }
            self._q_trades.append(rec)
            self._q_pnl_list.append(pnl)
            self._q_pos = None
            logger.info(f"[sync] Bybit closed externally → {exit_type} pnl={pnl:.2f}")
            self.notify(
                f"Bybit {exit_type}: {p['side'].upper()} PnL {pnl:+.2f}",
                severity="information" if pnl > 0 else "warning",
            )
        elif self._q_pos is None and bybit_size > 0:
            logger.warning(
                f"[sync] Untracked Bybit position: {bybit_side} {bybit_size} @ {bybit_entry}"
            )

    def _refresh_private_account(self):
        if self.mode != "por":
            return
        now = time.monotonic()
        ws_inactive = (now - self._priv_last_update_monotonic) > 15.0
        if not ws_inactive and self._priv_ok and self._priv_state.get("balances"):
            return
        if now - self._last_priv_poll < 23.0:
            return
        self._last_priv_poll = now
        # Fetch real equity from Bybit and sync _q_equity
        eq = self._fetch_bybit_equity()
        if eq is not None:
            self._q_equity = eq
            if self._q_start_equity <= 0:          # first successful fetch
                self._q_start_equity = eq
                self._q_peak_equity  = eq
            self._priv_state = {"balances": {"equity": eq, "wallet_balance": eq}}
            self._priv_ok    = True
            self._priv_last_ts = datetime.now(
                timezone(timedelta(hours=9))
            ).strftime("%H:%M:%S")
        else:
            snapshot = self.client.fetch_account_snapshot(self.cfg["app"]["symbol"])
            if snapshot.get("error"):
                self._priv_ok  = False
                self._priv_err = "Auth failed"
                return
            self._priv_state = snapshot
            self._priv_ok  = True
            self._priv_last_ts = datetime.now(
                timezone(timedelta(hours=9))
            ).strftime("%H:%M:%S")
        try:
            self._priv_trade_logger.update(self._priv_state)
        except Exception:
            pass

    # ── ATR helper ─────────────────────────────────────────────────────────────

    def _atr_pct(self, window):
        try:
            import pandas as pd
            if window is None or len(window) < 2:
                return None
            highs  = window["high"].astype(float)
            lows   = window["low"].astype(float)
            closes = window["close"].astype(float)
            prev   = closes.shift(1)
            tr     = pd.concat([
                highs - lows,
                (highs - prev).abs(),
                (lows  - prev).abs(),
            ], axis=1).max(axis=1)
            atr   = tr.rolling(14).mean().iloc[-1]
            price = float(closes.iloc[-1])
            if price <= 0 or atr != atr:
                return None
            return float(atr / price)
        except Exception:
            return None

    # ── Quantum methods ────────────────────────────────────────────────────────

    def _quantum_step(self):
        """Run quantum model inference on current bar; manage ATR-based positions."""
        if self._q_agent is None:
            return
        if self.sim_idx < 0 or self.sim_idx >= len(self.df):
            return

        import torch

        row   = self.df.iloc[self.sim_idx]
        price = float(row["close"])
        bar_ts = str(row.get("ts", ""))

        is_new_bar = bar_ts != self._q_last_bar_ts
        self._q_last_bar_ts = bar_ts

        # ── OI 주기적 갱신 (5분마다, live/por 모드에서만) ────────────────────
        import time as _time_mod
        _now_ms = int(_time_mod.time() * 1000)
        _oi_interval_ms = 5 * 60 * 1000   # 5분
        if (is_new_bar
                and self.mode in ("por", "live")
                and _now_ms - self._q_oi_last_fetch_ms > _oi_interval_ms):
            try:
                _df_oi = self.client.fetch_open_interest_recent(
                    self.cfg["app"]["symbol"], interval="15min", limit=100
                )
                if not _df_oi.empty:
                    import pandas as _pd_oi
                    _df_oi["ts"] = (
                        _pd_oi.to_datetime(_df_oi["ts_ms"], unit="ms", utc=True)
                        .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    # self.df에 open_interest 컬럼 업데이트 (merge)
                    if "open_interest" in self.df.columns:
                        self.df = self.df.drop(columns=["open_interest"])
                    self.df = self.df.merge(
                        _df_oi[["ts", "open_interest"]], on="ts", how="left"
                    )
                    self.df["open_interest"] = (
                        self.df["open_interest"].ffill().fillna(0.0)
                    )
                    self._q_oi_last_fetch_ms = _now_ms
            except Exception:
                pass   # OI fetch 실패 시 feat[23]=0.0 fallback

        # Build rolling feature buffer
        warmup = 25
        if is_new_bar and self.sim_idx >= warmup:
            lookback = 30
            window = self.df.iloc[max(0, self.sim_idx - lookback + 1): self.sim_idx + 1]
            try:
                feat = build_features_v4(window)
                # bc_scaler 적용 (BC와 동일한 피처 분포)
                if self._q_scaler is not None:
                    import numpy as _np
                    feat = self._q_scaler.transform(
                        _np.array(feat).reshape(1, -1)
                    ).flatten().astype(_np.float32)
                self._q_feat_buf.append(feat)
            except Exception as e:
                self._q_status = f"Feat err: {str(e)[:40]}"
                return

        if len(self._q_feat_buf) < 2:
            self._q_status = f"Warmup: {len(self._q_feat_buf)} feats"
            return

        # ATR for TP/SL sizing
        window_atr = self.df.iloc[max(0, self.sim_idx - 20): self.sim_idx + 1]
        atr_frac   = self._atr_pct(window_atr) or 0.01
        tp_pct     = self._q_tp_mult * atr_frac
        sl_pct     = self._q_sl_mult * atr_frac

        # ── Check exit of existing position (every tick) ──────────────────────
        # In por+trading_active: _sync_bybit_position() handles real exits.
        # Here we only check for paper mode OR por with trading inactive
        # (so virtual tracking still works for signal preview).
        _por_real = (self.mode == "por" and self._trading_active)
        if self._q_pos is not None and not _por_real:
            p = self._q_pos
            ret = (
                (price - p["entry"]) / p["entry"]
                if p["side"] == "long"
                else (p["entry"] - price) / p["entry"]
            )

            # ── Watermark 갱신 (ratchet-only) ─────────────────────────────
            if p["side"] == "long":
                if price > p["trail_watermark"]:
                    p["trail_watermark"] = price
                wg = (p["trail_watermark"] - p["entry"]) / p["entry"]
            else:
                if price < p["trail_watermark"]:
                    p["trail_watermark"] = price
                wg = (p["entry"] - p["trail_watermark"]) / p["entry"]

            if not p["trail_active"] and self._q_trail_after > 0:
                if wg >= self._q_trail_after * p["trail_atr_frac"]:
                    p["trail_active"] = True

            # Trail SL 체크
            trail_exit_pnl = None
            if p["trail_active"]:
                _tdist = self._q_trail_dist * p["trail_atr_frac"]
                if p["side"] == "long":
                    _tsl = p["trail_watermark"] * (1 - _tdist)
                    if price <= _tsl:
                        trail_exit_pnl = (_tsl - p["entry"]) / p["entry"]
                else:
                    _tsl = p["trail_watermark"] * (1 + _tdist)
                    if price >= _tsl:
                        trail_exit_pnl = (p["entry"] - _tsl) / p["entry"]

            # Exit priority: TP → TRAIL_SL → SL
            hit_tp = ret >= p["tp_pct"]
            hit_sl = ret <= -p["sl_pct"]
            if hit_tp:
                exit_type  = "TP"
                exit_price = p["tp_price"]
                pnl = (
                    (exit_price - p["entry"]) / p["entry"] * p["notional"]
                    if p["side"] == "long"
                    else (p["entry"] - exit_price) / p["entry"] * p["notional"]
                )
            elif trail_exit_pnl is not None:
                exit_type  = "TRAIL_SL"
                exit_price = price
                pnl = trail_exit_pnl * p["notional"]
            elif hit_sl:
                exit_type  = "SL"
                exit_price = p["sl_price"]
                pnl = (
                    (exit_price - p["entry"]) / p["entry"] * p["notional"]
                    if p["side"] == "long"
                    else (p["entry"] - exit_price) / p["entry"] * p["notional"]
                )
            else:
                exit_type = None

            if exit_type is not None:
                pnl -= p["notional"] * 0.00055   # exit taker fee
                self._q_equity      += pnl
                self._q_total_trades += 1
                if pnl > 0:
                    self._q_wins += 1
                rec = {
                    "side": p["side"], "entry": p["entry"],
                    "exit": exit_price, "pnl": round(pnl, 4),
                    "exit_type": exit_type, "ts": bar_ts,
                }
                self._q_trades.append(rec)
                self._q_pnl_list.append(pnl)
                self._q_perf_a["trades"] += 1
                if pnl > 0:
                    self._q_perf_a["wins"] += 1
                if self._q_equity > self._q_peak_equity:
                    self._q_peak_equity = self._q_equity
                dd = (self._q_peak_equity - self._q_equity) / (self._q_peak_equity + 1e-10)
                if dd > self._q_max_drawdown:
                    self._q_max_drawdown = dd
                if self._q_live_train and is_new_bar:
                    self._quantum_online_train(rec)
                self._q_pos = None

        # Skip inference on intra-bar ticks
        if not is_new_bar:
            self._quantum_update_status(price)
            return

        # Decrement WS reconnect cooldown on each new bar
        if self._ws_entry_cooldown > 0:
            self._ws_entry_cooldown -= 1
            self._q_status = f"WS cooldown: {self._ws_entry_cooldown + 1} bars left"
            self._quantum_update_status(price)
            return

        # Build [1, T, 23] tensor — run on every new bar (entry AND exit decisions)
        seq     = list(self._q_feat_buf)[-60:]
        seq_arr = _np_q.array(seq, dtype=_np_q.float32)
        x = torch.tensor(seq_arr[None], dtype=torch.float32, device=self._q_device)

        try:
            action, prob, probs = self._q_agent.select_action(
                x, atr_norm=float(atr_frac)
            )
        except Exception as e:
            self._q_status = f"Infer err: {str(e)[:40]}"
            return

        self._q_last_action = action
        self._q_last_prob   = float(prob)
        self._q_last_probs  = probs

        # ── Regime-flip early exit (new bar, in position) ──────────────────────
        _por_real = (self.mode == "por" and self._trading_active)
        if (self._q_pos is not None and not _por_real
                and self._q_regime_exit_conf > 0):
            p = self._q_pos
            opposite_action = 2 if p["side"] == "long" else 1
            if action == opposite_action and float(prob) >= self._q_regime_exit_conf:
                exit_price = price  # market close of current bar
                pnl = (
                    (exit_price - p["entry"]) / p["entry"] * p["notional"]
                    if p["side"] == "long"
                    else (p["entry"] - exit_price) / p["entry"] * p["notional"]
                )
                pnl -= p["notional"] * 0.00055  # exit taker fee
                self._q_equity       += pnl
                self._q_total_trades += 1
                if pnl > 0:
                    self._q_wins += 1
                rec = {
                    "side": p["side"], "entry": p["entry"],
                    "exit": exit_price, "pnl": round(pnl, 4),
                    "exit_type": "REGIME_EXIT", "ts": bar_ts,
                }
                self._q_trades.append(rec)
                self._q_pnl_list.append(pnl)
                self._q_perf_a["trades"] += 1
                if pnl > 0:
                    self._q_perf_a["wins"] += 1
                if self._q_equity > self._q_peak_equity:
                    self._q_peak_equity = self._q_equity
                dd = (self._q_peak_equity - self._q_equity) / (self._q_peak_equity + 1e-10)
                if dd > self._q_max_drawdown:
                    self._q_max_drawdown = dd
                if self._q_live_train:
                    self._quantum_online_train(rec)
                self._q_pos = None
                self._quantum_update_status(price)
                return

        # Skip entry if already in position
        if self._q_pos is not None:
            self._quantum_update_status(price)
            return

        # ── Daily loss limit check ─────────────────────────────────────────────
        if self._trading_active and self._q_session_start_eq:
            loss_pct = (self._q_session_start_eq - self._q_equity) / self._q_session_start_eq
            if loss_pct >= self._q_daily_loss_limit:
                self._trading_active = False
                msg = (
                    f"DAILY LOSS LIMIT HIT: {loss_pct:.1%} ≥ {self._q_daily_loss_limit:.0%} "
                    f"— trading halted (start ${self._q_session_start_eq:,.0f} "
                    f"→ now ${self._q_equity:,.0f})"
                )
                logger.warning(msg)
                self.notify(msg, severity="error")
                if self.mode == "por":
                    self._close_bybit_position()
                self._q_pos = None

        # ── Entry logic ────────────────────────────────────────────────────────
        if action in (1, 2) and float(prob) >= self._q_confidence:
            rr = tp_pct / (sl_pct + 1e-10)
            if rr >= 3.0:
                side       = "long" if action == 1 else "short"
                notional   = self._q_equity * self._q_pos_frac * self._q_leverage
                if notional > 0 and self._q_equity > 0:
                    fee_entry = notional * 0.0002
                    self._q_equity -= fee_entry

                    tp_price = (
                        price * (1 + tp_pct) if side == "long"
                        else price * (1 - tp_pct)
                    )
                    sl_price = (
                        price * (1 - sl_pct) if side == "long"
                        else price * (1 + sl_pct)
                    )

                    # Paper: virtual position
                    self._q_pos = {
                        "side": side, "entry": price,
                        "tp_pct": tp_pct, "sl_pct": sl_pct,
                        "notional": notional, "confidence": float(prob),
                        "tp_price": tp_price, "sl_price": sl_price,
                        "trail_watermark": price,
                        "trail_active":    False,
                        "trail_atr_frac":  atr_frac,
                    }

                    # Por + trading active: place REAL order
                    if self.mode == "por" and self._trading_active:
                        qty = self._round_qty(notional / price)
                        self._place_bybit_order(
                            side      = "Buy" if side == "long" else "Sell",
                            qty       = qty,
                            tp_price  = tp_price,
                            sl_price  = sl_price,
                        )

        # ── Model B shadow tracking ────────────────────────────────────────────
        if self._q_agent_b is not None:
            if self._q_shadow_pos is not None:
                sp  = self._q_shadow_pos
                ret_b = (
                    (price - sp["entry"]) / sp["entry"]
                    if sp["side"] == "long"
                    else (sp["entry"] - price) / sp["entry"]
                )
                if ret_b >= sp["tp_pct"] or ret_b <= -sp["sl_pct"]:
                    self._q_perf_b["trades"] += 1
                    if ret_b >= sp["tp_pct"]:
                        self._q_perf_b["wins"] += 1
                    self._q_shadow_pos = None

            if self._q_shadow_pos is None:
                try:
                    action_b, prob_b, _ = self._q_agent_b.select_action(
                        x, atr_norm=float(atr_frac)
                    )
                except Exception:
                    action_b, prob_b = 0, 0.0
                if (action_b in (1, 2)
                        and float(prob_b) >= self._q_confidence
                        and tp_pct / (sl_pct + 1e-10) >= 3.0):
                    self._q_shadow_pos = {
                        "side":   "long" if action_b == 1 else "short",
                        "entry":  price,
                        "tp_pct": tp_pct,
                        "sl_pct": sl_pct,
                    }

        # ── Daily swap evaluation ──────────────────────────────────────────────
        if (self._q_next_swap_time is not None
                and datetime.now(timezone.utc) >= self._q_next_swap_time):
            self._quantum_eval_swap()

        self._quantum_update_status(price)

    def _quantum_update_status(self, price: float):
        act_names = ["HOLD", "LONG", "SHORT"]
        act_str   = act_names[self._q_last_action]
        wr = (
            self._q_wins / self._q_total_trades * 100
            if self._q_total_trades > 0 else 0.0
        )
        if self._q_pos is not None:
            p = self._q_pos
            upnl_pct = (
                (price - p["entry"]) / p["entry"] * 100
                if p["side"] == "long"
                else (p["entry"] - price) / p["entry"] * 100
            )
            c = "green" if upnl_pct >= 0 else "red"
            pos_str = (
                f"[{c}]{p['side'].upper()} @{p['entry']:,.0f}"
                f" ({upnl_pct:+.2f}%)[/{c}]"
            )
        else:
            pos_str = "[dim]FLAT[/dim]"
        self._q_status = (
            f"{act_str}({self._q_last_prob:.2f}) | {pos_str} | "
            f"Eq:{self._q_equity:,.0f} | WR:{wr:.0f}%({self._q_total_trades}T)"
        )

    def _quantum_eval_swap(self):
        """Daily: swap B→A if B shadow WR beats A actual WR by margin."""
        _now_utc = datetime.now(timezone.utc)
        self._q_next_swap_time = (
            _now_utc + timedelta(days=1)
        ).replace(hour=0, minute=0, second=0, microsecond=0)

        a_t = self._q_perf_a["trades"]
        b_t = self._q_perf_b["trades"]
        now_kst = datetime.now(timezone(timedelta(hours=9))).strftime("%m-%d %H:%M")

        if a_t < self._q_min_eval_trades or b_t < self._q_min_eval_trades:
            self._q_last_swap_info = (
                f"[{now_kst}] Skip: A={a_t}T B={b_t}T"
                f" (need >={self._q_min_eval_trades})"
            )
            logger.info(self._q_last_swap_info)
            return

        wr_a = self._q_perf_a["wins"] / a_t
        wr_b = self._q_perf_b["wins"] / b_t

        if wr_b > wr_a + self._q_swap_margin:
            import copy as _cp
            self._q_agent.load_state_dict(
                _cp.deepcopy(self._q_agent_b.state_dict())
            )
            self._q_agent.eval()
            self._q_swap_count += 1
            ckpt_ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
            ckpt_path = f"checkpoints/quantum_v2/champion_{ckpt_ts}.pt"
            try:
                os.makedirs("checkpoints/quantum_v2", exist_ok=True)
                self._q_agent.save_checkpoint(ckpt_path)
            except Exception:
                pass
            self._q_perf_a    = {"wins": 0, "trades": 0}
            self._q_perf_b    = {"wins": 0, "trades": 0}
            self._q_shadow_pos = None
            self._q_last_swap_info = (
                f"[{now_kst}] SWAP #{self._q_swap_count}: "
                f"B {wr_b:.1%} > A {wr_a:.1%} "
                f"(+{(wr_b-wr_a)*100:.1f}%) → saved"
            )
            logger.info(self._q_last_swap_info)
            self.notify(
                f"Quantum SWAP #{self._q_swap_count}: B→A  ({wr_b:.1%} vs {wr_a:.1%})",
                severity="warning",
            )
        else:
            self._q_last_swap_info = (
                f"[{now_kst}] No swap: "
                f"A {wr_a:.1%}({a_t}T) B {wr_b:.1%}({b_t}T) "
                f"D={(wr_b-wr_a)*100:+.1f}%"
            )
            logger.info(self._q_last_swap_info)

    def _quantum_online_train(self, completed_trade: dict):
        """Online training: trains B (challenger) only — A stays frozen."""
        if self._q_agent_b is None or len(self._q_feat_buf) < 10:
            return
        import torch
        try:
            feats   = list(self._q_feat_buf)
            seq_arr = _np_q.array(feats[-60:], dtype=_np_q.float32)
            x = torch.tensor(seq_arr[None], dtype=torch.float32, device=self._q_device)

            n      = min(len(feats), 60)
            window = self.df.iloc[max(0, self.sim_idx - n + 1): self.sim_idx + 1]
            prices_np = window["close"].astype(float).values[-len(seq_arr):]
            prices = torch.tensor(
                prices_np[None], dtype=torch.float32, device=self._q_device
            )
            directions = torch.tensor(
                [1.0 if completed_trade["side"] == "long" else -1.0],
                dtype=torch.float32, device=self._q_device,
            )
            entry_idx = torch.tensor([0], dtype=torch.long, device=self._q_device)
            label_val = 1 if completed_trade["exit_type"] == "TP" else 2
            labels    = torch.tensor([label_val], dtype=torch.long, device=self._q_device)
            atr_frac  = self._atr_pct(window) or 0.01
            atr       = torch.tensor([atr_frac], dtype=torch.float32, device=self._q_device)

            self._q_agent_b.train()
            self._q_agent_b.train_step(
                x=x, prices=prices, directions=directions,
                entry_idx=entry_idx, labels=labels, atr=atr,
                last_step_only=True,
            )
            self._q_agent_b.train()
            logger.info(f"Quantum B trained: {completed_trade['exit_type']}")
        except Exception as e:
            logger.warning(f"Quantum B train failed: {e}")

    def _update_quantum_pane(self):
        """Render quantum engine status — dual model + performance metrics."""
        import math as _math
        pane  = self.query_one("#right-pane", QuantPane)
        lines = []

        lines.append("[b]QUANTUM ENGINE  A+B[/b]")
        lines.append("-" * 30)

        # Performance metrics (ChatGPT challenge)
        n        = self._q_total_trades
        wr       = self._q_wins / n if n > 0 else 0.0
        init_eq  = self._q_start_equity
        cur_eq   = self._q_equity
        elapsed_days = max(
            1.0,
            (datetime.now(timezone.utc) - self._q_start_time).total_seconds() / 86400,
        )
        cagr = (cur_eq / init_eq) ** (365.0 / elapsed_days) - 1 if cur_eq > 0 else -1.0

        sharpe = float("nan")
        if len(self._q_pnl_list) >= 4:
            try:
                rets = [p / init_eq for p in self._q_pnl_list]
                mu   = sum(rets) / len(rets)
                var  = sum((r - mu) ** 2 for r in rets) / max(1, len(rets) - 1)
                std  = _math.sqrt(var) if var > 0 else 1e-10
                sharpe = (mu / std) * _math.sqrt(252 * n / max(1.0, elapsed_days))
            except Exception:
                pass

        p_ruin = float("nan")
        if n >= 4:
            try:
                wins_   = [p for p in self._q_pnl_list if p > 0]
                losses_ = [abs(p) for p in self._q_pnl_list if p < 0]
                avg_w   = sum(wins_) / len(wins_)   if wins_   else 0.001
                avg_l   = sum(losses_) / len(losses_) if losses_ else 0.001
                R       = avg_w / avg_l
                kelly   = wr - (1 - wr) / R
                if kelly > 0:
                    p_ruin = max(0.0, _math.exp(
                        -2 * kelly * (cur_eq / avg_l)
                    ))
                else:
                    p_ruin = 1.0
            except Exception:
                pass

        eq_c  = "green" if cur_eq >= init_eq else "red"
        cagr_c = "green" if cagr > 0 else "red"
        wr_c  = "green" if wr > 0.232 else "red"
        sh_c  = (
            "green"  if (not _math.isnan(sharpe) and sharpe > 1.0) else
            "red"    if (not _math.isnan(sharpe) and sharpe < 0)   else
            "yellow"
        )

        lines.append("[b]PERFORMANCE METRICS[/b]")
        lines.append("-" * 30)
        lines.append(f"  Equity:  [{eq_c}]{cur_eq:,.2f}[/{eq_c}]  (init {init_eq:,.0f})")
        lines.append(f"  Trades:  {n}   WR: [{wr_c}]{wr:.1%}[/{wr_c}] (BEP 23.2%)")
        if not _math.isnan(sharpe):
            lines.append(f"  Sharpe:  [{sh_c}]{sharpe:.2f}[/{sh_c}]")
        else:
            lines.append("  Sharpe:  [dim]n/a[/dim]")
        lines.append(f"  Max DD:  [red]{self._q_max_drawdown:.1%}[/red]")
        lines.append(f"  CAGR:    [{cagr_c}]{cagr:.1%}[/{cagr_c}]  ({elapsed_days:.0f}d)")
        if not _math.isnan(p_ruin):
            pr_c = "green" if p_ruin < 0.05 else ("yellow" if p_ruin < 0.20 else "red")
            lines.append(f"  P(Ruin): [{pr_c}]{p_ruin:.2%}[/{pr_c}]")
        else:
            lines.append("  P(Ruin): [dim]n/a (<4T)[/dim]")
        lines.append("  OOS:     [yellow]Partial (Fold5 val overlap)[/yellow]")
        lines.append("")

        # Dual model status
        wr_a_v = (self._q_perf_a["wins"] / self._q_perf_a["trades"]
                  if self._q_perf_a["trades"] > 0 else 0.0)
        wr_b_v = (self._q_perf_b["wins"] / self._q_perf_b["trades"]
                  if self._q_perf_b["trades"] > 0 else 0.0)
        a_clr = "green" if wr_a_v >= 0.232 else "red"
        b_clr = "green" if wr_b_v > wr_a_v + self._q_swap_margin else "yellow"

        lines.append("[b]DUAL MODEL (A Champion / B Challenger)[/b]")
        lines.append("-" * 30)
        lines.append(
            f"  A (live):   [{a_clr}]{wr_a_v:.1%}[/{a_clr}]"
            f" ({self._q_perf_a['trades']}T)  FROZEN"
        )
        lines.append(
            f"  B (shadow): [{b_clr}]{wr_b_v:.1%}[/{b_clr}]"
            f" ({self._q_perf_b['trades']}T)  TRAIN"
        )
        lines.append(f"  Swaps:   {self._q_swap_count}  "
                     f"Margin:{self._q_swap_margin:.0%}")
        if self._q_next_swap_time:
            nxt_kst = self._q_next_swap_time.astimezone(
                timezone(timedelta(hours=9))
            ).strftime("%m-%d %H:%M")
            lines.append(f"  NextEval: {nxt_kst} KST")
        lines.append(f"  Last: [dim]{self._q_last_swap_info[:28]}[/dim]")
        lines.append("")

        # Signal
        act_names  = ["HOLD", "LONG", "SHORT"]
        act_colors = ["dim",  "green", "red"]
        act   = self._q_last_action
        act_c = act_colors[act]
        lines.append("[b]SIGNAL[/b]")
        lines.append("-" * 30)
        lines.append(
            f"  [{act_c}]{act_names[act]}[/{act_c}] "
            f"p={self._q_last_prob:.3f}  buf={len(self._q_feat_buf)}bars"
        )
        if self._q_last_probs is not None:
            try:
                _p = (self._q_last_probs.cpu().numpy()
                      if hasattr(self._q_last_probs, "cpu")
                      else self._q_last_probs)
                lines.append(
                    f"  H={float(_p[0]):.2f} L={float(_p[1]):.2f} S={float(_p[2]):.2f}"
                )
            except Exception:
                pass
        lines.append(
            f"  Conf>={self._q_confidence:.2f}  Lev={self._q_leverage:.0f}x"
            f"  Frac={self._q_pos_frac:.0%}"
        )
        lines.append("")

        # Position
        lines.append("[b]POSITION[/b]")
        lines.append("-" * 30)
        if self._q_pos is not None:
            p     = self._q_pos
            side_c = "green" if p["side"] == "long" else "red"
            row_  = self.df.iloc[self.sim_idx] if self.sim_idx < len(self.df) else None
            cur_p = float(row_["close"]) if row_ is not None else p["entry"]
            upnl_pct = (
                (cur_p - p["entry"]) / p["entry"] * 100
                if p["side"] == "long"
                else (p["entry"] - cur_p) / p["entry"] * 100
            )
            upnl_abs = upnl_pct / 100 * p["notional"]
            uc = "green" if upnl_abs >= 0 else "red"
            lines.append(
                f"  [{side_c}]{p['side'].upper()}[/{side_c}]"
                f"  @{p['entry']:,.0f}  Conf:{p['confidence']:.2f}"
            )
            lines.append(
                f"  UPnL [{uc}]{upnl_abs:+.1f}[/{uc}] ({upnl_pct:+.2f}%)"
            )
            lines.append(
                f"  TP [green]{p['tp_price']:,.0f}[/green]"
                f"  SL [red]{p['sl_price']:,.0f}[/red]"
            )
        else:
            lines.append("  [dim]FLAT[/dim]")
        lines.append("")

        # Recent trades
        lines.append("[b]RECENT TRADES[/b]")
        lines.append("-" * 30)
        recent = self._q_trades[-6:]
        if not recent:
            lines.append("  [dim]None yet[/dim]")
        for t in reversed(recent):
            sc = "green" if t["side"] == "long" else "red"
            ec = "green" if t["exit_type"] == "TP" else "red"
            pc = "green" if t["pnl"] >= 0 else "red"
            lines.append(
                f"  [{sc}]{t['side'][:1].upper()}[/{sc}]"
                f"[{ec}]{t['exit_type']}[/{ec}]"
                f" [{pc}]{t['pnl']:+.1f}[/{pc}]"
                f" @{t['exit']:,.0f}"
            )

        pane.update("\n".join(lines))


# ── Entry point ────────────────────────────────────────────────────────────────

def _ask_initial_equity() -> float:
    """Prompt user for initial equity before TUI starts."""
    import argparse
    parser = argparse.ArgumentParser(description="Terminal Quant Suite")
    parser.add_argument("--equity", type=float, default=None)
    args, _ = parser.parse_known_args()

    if args.equity is not None:
        if args.equity <= 0:
            print("Error: --equity must be positive.")
            raise SystemExit(1)
        print(f"Starting with initial equity: ${args.equity:,.2f}")
        return args.equity

    default = 10_000.0
    try:
        raw = input(f"Enter initial equity in USD [{default:,.0f}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raw = ""
    if not raw:
        print(f"Using default: ${default:,.2f}")
        return default
    try:
        value = float(raw.replace(",", "").replace("$", ""))
        if value <= 0:
            print(f"Invalid. Using default: ${default:,.2f}")
            return default
        print(f"Starting with initial equity: ${value:,.2f}")
        return value
    except ValueError:
        print(f"Invalid. Using default: ${default:,.2f}")
        return default


def run():
    args = parse_tui_args()
    mode = getattr(args, "mode", None) or "paper"
    if mode == "por":
        # por mode: equity is loaded from Bybit — no manual input needed
        equity = args.equity  # may be None; __init__ will overwrite from Bybit
    else:
        equity = args.equity if args.equity is not None else _ask_initial_equity()
    TerminalQuantApp(
        initial_equity      = equity,
        mode_override       = getattr(args, "mode", None),
        symbol_override     = getattr(args, "symbol", None),
        timeframe_override  = getattr(args, "timeframe", None),
        bybit_env           = getattr(args, "bybit_env", None),
        bybit_category      = getattr(args, "bybit_category", None),
        quantum_model_path  = getattr(args, "quantum_model", None),
        q_confidence        = getattr(args, "q_confidence", 0.65),
        q_leverage          = getattr(args, "q_leverage", 10.0),
        q_pos_frac          = getattr(args, "q_pos_frac", 0.5),
        q_tp_mult           = getattr(args, "q_tp_mult", 4.0),
        q_sl_mult           = getattr(args, "q_sl_mult", 1.0),
        q_live_train        = getattr(args, "q_live_train", False),
        q_swap_margin       = getattr(args, "q_swap_margin", 0.03),
        q_min_eval_trades   = getattr(args, "q_min_eval_trades", 10),
        q_daily_loss_limit  = getattr(args, "q_daily_loss_limit", 0.05),
        q_regime_exit_conf  = getattr(args, "q_regime_exit_conf", 0.0),
        q_trail_after       = getattr(args, "q_trail_after",       0.0),
        q_trail_dist        = getattr(args, "q_trail_dist",        1.0),
    ).run()


if __name__ == "__main__":
    run()
