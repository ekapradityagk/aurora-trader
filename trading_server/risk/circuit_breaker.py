"""Aurora Trader — Circuit Breaker.

Tracks daily P&L across all positions and automatically pauses trading
when the daily loss limit (3% default) is breached. Resets at UTC midnight
and persists state in SQLite.

Design:
- Daily P&L is accumulated from position closes and tracked in SQLite
- If the loss exceeds the configured threshold, trading is paused
- A separate ``check_and_reset()`` method handles the UTC midnight reset
- Persistent state prevents loss of tracking on server restart
"""

from __future__ import annotations

import aiosqlite
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

from shared.config import load_config
from shared.logger import get_logger

logger = get_logger("trading_server.risk.circuit_breaker")

# ---------------------------------------------------------------------------
# SQLite schema for circuit breaker state
# ---------------------------------------------------------------------------

_CB_SCHEMA = """
CREATE TABLE IF NOT EXISTS circuit_breaker (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row
    date TEXT NOT NULL,                      -- ISO date (YYYY-MM-DD) in UTC
    starting_balance REAL NOT NULL DEFAULT 0.0,
    current_pnl REAL NOT NULL DEFAULT 0.0,
    is_paused INTEGER NOT NULL DEFAULT 0,   -- 0 = active, 1 = paused
    loss_limit_pct REAL NOT NULL DEFAULT 3.0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_pnl_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    pnl REAL NOT NULL,
    reason TEXT DEFAULT '',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paused_symbols (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    reason TEXT DEFAULT '',
    paused_at TEXT NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS closed_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    pnl REAL NOT NULL,
    reason TEXT DEFAULT '',
    leverage INTEGER DEFAULT 1,
    closed_at TEXT NOT NULL,
    entry_time TEXT DEFAULT '',
    strategy_name TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    price REAL NOT NULL,
    timeframe TEXT DEFAULT '',
    reason TEXT DEFAULT '',
    regime TEXT DEFAULT '',
    executed INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trailing_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entry_price REAL NOT NULL,
    current_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    leverage INTEGER DEFAULT 1,
    event_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS opportunity_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_time TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence INTEGER NOT NULL,
    primary_timeframe TEXT DEFAULT '',
    entry_notes TEXT DEFAULT '',
    price REAL DEFAULT 0,
    brewing INTEGER DEFAULT 0,
    raw_json TEXT DEFAULT ''
);
"""


class CircuitBreaker:
    """Daily P&L circuit breaker that pauses trading after a loss limit breach.

    Usage::

        cb = CircuitBreaker()
        await cb.initialize()
        await cb.record_trade_pnl("BTCUSDT", Decimal("-50.0"), "stop_loss")
        if await cb.is_paused():
            # Don't open new positions
            pass
        await cb.check_and_reset()  # call on each new candle
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        loss_limit_pct: float = 3.0,
    ) -> None:
        cfg = load_config()
        ts_cfg = cfg.data.get("trading_server", {})

        self._db_path = db_path or ts_cfg.get("database", {}).get(
            "path", "data/trading.db"
        )
        # Ensure directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Loss limit: use config's daily_loss_limit_pct, defaulting to 3%
        risk_global = cfg.risk_global
        configured_limit = float(
            risk_global.get("daily_loss_limit_pct", 5.0)
        )
        self._loss_limit_pct = configured_limit if configured_limit > 0 else loss_limit_pct

        # Max open positions (for per-symbol loss limit calculation)
        self._max_open_positions = int(
            risk_global.get("max_open_positions", 6)
        )

        self._log = logger
        self._conn: Optional[aiosqlite.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create DB connection and ensure schema exists."""
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_CB_SCHEMA)
        await self._conn.commit()
        self._log.info(
            f"CircuitBreaker initialised: db={self._db_path}, "
            f"loss_limit={self._loss_limit_pct}%"
        )
        # Ensure the singleton row exists
        await self._ensure_row()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._log.info("CircuitBreaker closed")

    async def _ensure_row(self) -> None:
        """Create the singleton circuit_breaker row if it doesn't exist."""
        today = self._utc_date()
        now = self._utc_now_str()
        await self._conn.execute("""
            INSERT OR IGNORE INTO circuit_breaker
                (id, date, starting_balance, current_pnl, is_paused,
                 loss_limit_pct, updated_at)
            VALUES (1, ?, 0.0, 0.0, 0, ?, ?)
        """, (today, self._loss_limit_pct, now))
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Daily P&L Tracking
    # ------------------------------------------------------------------

    async def record_trade_pnl(
        self,
        symbol: str,
        pnl: Decimal,
        reason: str = "trade_close",
    ) -> Dict[str, Any]:
        """Record a realised P&L from a closed trade.

        Returns:
            Dict with current daily stats: total_pnl, is_paused, etc.
        """
        today = self._utc_date()
        now = self._utc_now_str()

        # Insert into log
        await self._conn.execute("""
            INSERT INTO daily_pnl_log (date, symbol, pnl, reason, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (today, symbol, float(pnl), reason, now))

        # Update aggregate
        await self._conn.execute("""
            UPDATE circuit_breaker
            SET current_pnl = current_pnl + ?,
                updated_at = ?
            WHERE id = 1 AND date = ?
        """, (float(pnl), now, today))
        await self._conn.commit()

        # Check per-symbol loss limit — pause just this symbol
        await self._check_symbol_loss(symbol, pnl, today, now)

        # Check if we need to pause globally (catastrophic loss only)
        state = await self.get_state()
        # Global pause at -15% max drawdown (keeps account from blowing up)
        if state["loss_pct"] >= 15.0 and not state["is_paused"]:
            await self._pause()
            self._log.critical(
                f"MAX DRAWDOWN BREACHED: "
                f"PnL={state['current_pnl']:.2f} "
                f"({state['loss_pct']:.2f}% ≥ 15%). "
                f"ALL TRADING PAUSED until UTC midnight."
            )

        return state

    async def _check_symbol_loss(
        self,
        symbol: str,
        pnl: Decimal,
        today: str,
        now: str,
    ) -> None:
        """Check if a specific symbol breached its daily loss limit and pause it."""
        try:
            # Get per-symbol accumulated PnL for today
            cursor = await self._conn.execute(
                "SELECT COALESCE(SUM(pnl), 0.0) AS total FROM daily_pnl_log "
                "WHERE date = ? AND symbol = ?",
                (today, symbol),
            )
            row = await cursor.fetchone()
            symbol_pnl = float(row["total"]) if row else float(pnl)

            state = await self.get_state()
            sb = state["starting_balance"]
            if sb <= 0:
                return

            # Per-symbol limit = loss_limit_pct / max positions
            per_symbol_limit = self._loss_limit_pct / float(self._max_open_positions)
            symbol_loss_pct = (abs(symbol_pnl) / sb * 100) if symbol_pnl < 0 else 0.0

            if symbol_loss_pct >= per_symbol_limit:
                # Check if already paused
                c = await self._conn.execute(
                    "SELECT 1 FROM paused_symbols WHERE symbol = ? AND date = ?",
                    (symbol, today),
                )
                already = await c.fetchone()
                if not already:
                    await self._conn.execute(
                        "INSERT INTO paused_symbols (symbol, date, reason, paused_at) "
                        "VALUES (?, ?, ?, ?)",
                        (symbol, today,
                         f"Daily loss {symbol_loss_pct:.2f}% ≥ {per_symbol_limit:.2f}% limit",
                         now),
                    )
                    await self._conn.commit()
                    self._log.info(
                        f"{symbol} | Paused for the day: "
                        f"PnL={symbol_pnl:.2f} ({symbol_loss_pct:.2f}%) "
                        f"≥ {per_symbol_limit:.2f}% per-symbol limit"
                    )
        except Exception as exc:
            self._log.debug(f"Symbol loss check failed for {symbol}: {exc}")

    async def is_symbol_paused(self, symbol: str) -> bool:
        """Return True if this specific symbol is paused for the day."""
        if not self._conn:
            return False
        try:
            today = self._utc_date()
            c = await self._conn.execute(
                "SELECT 1 FROM paused_symbols WHERE symbol = ? AND date = ?",
                (symbol, today),
            )
            return await c.fetchone() is not None
        except Exception:
            return False

    async def pause_symbol(self, symbol: str, reason: str = "manual_pause") -> bool:
        """Pause a specific symbol without recording fake PnL.

        Directly inserts a row into paused_symbols so the per-symbol circuit
        breaker blocks new trades on this symbol for the rest of the day.

        Returns True if the symbol was newly paused, False if already paused.
        """
        if not self._conn:
            return False
        today = self._utc_date()
        now = self._utc_now_str()
        try:
            c = await self._conn.execute(
                "SELECT 1 FROM paused_symbols WHERE symbol = ? AND date = ?",
                (symbol, today),
            )
            if await c.fetchone():
                return False  # Already paused
            await self._conn.execute(
                "INSERT INTO paused_symbols (symbol, date, reason, paused_at) "
                "VALUES (?, ?, ?, ?)",
                (symbol, today, reason, now),
            )
            await self._conn.commit()
            self._log.info(
                f"{symbol} | Paused for the day: {reason}"
            )
            return True
        except Exception as exc:
            self._log.debug(f"pause_symbol failed for {symbol}: {exc}")
            return False

    async def unpause_symbol(self, symbol: str) -> None:
        """Remove a symbol from the paused list (e.g. at UTC reset)."""
        if not self._conn:
            return
        today = self._utc_date()
        await self._conn.execute(
            "DELETE FROM paused_symbols WHERE symbol = ? AND date = ?",
            (symbol, today),
        )
        await self._conn.commit()

    async def get_paused_symbols(self) -> list[str]:
        """Return list of symbols paused for the current day."""
        if not self._conn:
            return []
        try:
            today = self._utc_date()
            c = await self._conn.execute(
                "SELECT symbol FROM paused_symbols WHERE date = ?",
                (today,),
            )
            return [row["symbol"] for row in await c.fetchall()]
        except Exception:
            return []

    async def get_state(self) -> Dict[str, Any]:
        """Get the current circuit breaker state.

        Returns:
            Dict with keys: date, starting_balance, current_pnl, loss_pct,
            is_paused, loss_limit_pct.
        """
        if not self._conn:
            return {
                "date": self._utc_date(),
                "starting_balance": 0.0,
                "current_pnl": 0.0,
                "loss_pct": 0.0,
                "is_paused": False,
                "loss_limit_pct": self._loss_limit_pct,
            }

        cursor = await self._conn.execute(
            "SELECT * FROM circuit_breaker WHERE id = 1"
        )
        row = await cursor.fetchone()

        if row is None:
            return {
                "date": self._utc_date(),
                "starting_balance": 0.0,
                "current_pnl": 0.0,
                "loss_pct": 0.0,
                "is_paused": False,
                "loss_limit_pct": self._loss_limit_pct,
            }

        sb = float(row["starting_balance"])
        cp = float(row["current_pnl"])
        loss_pct = (abs(cp) / sb * 100) if sb > 0 else 0.0

        return {
            "date": row["date"],
            "starting_balance": sb,
            "current_pnl": cp,
            "loss_pct": round(loss_pct, 4),
            "is_paused": bool(row["is_paused"]),
            "loss_limit_pct": float(row["loss_limit_pct"]),
        }

    async def is_paused(self) -> bool:
        """Return True if trading is currently paused by the circuit breaker."""
        state = await self.get_state()
        return state["is_paused"]

    # ------------------------------------------------------------------
    # Reset (UTC midnight)
    # ------------------------------------------------------------------

    async def check_and_reset(self) -> bool:
        """Check if a new UTC day has started and reset the circuit breaker.

        Call this periodically (e.g. on every new 1h candle or every minute).
        Returns True if a reset occurred.
        """
        if not self._conn:
            return False

        state = await self.get_state()
        today = self._utc_date()

        if state["date"] != today:
            self._log.info(
                f"New UTC day detected: {state['date']} → {today}. "
                f"Resetting circuit breaker. "
                f"Previous day PnL: {state['current_pnl']:.2f}"
            )
            await self._reset(today)
            return True

        return False

    async def _reset(self, new_date: str) -> None:
        """Reset the circuit breaker for a new trading day.

        Uses the last known starting balance from the previous day.
        """
        state = await self.get_state()
        # The new starting balance is the old starting balance + realised PnL
        new_sb = max(state["starting_balance"] + state["current_pnl"], 100.0)
        now = self._utc_now_str()

        await self._conn.execute("""
            UPDATE circuit_breaker
            SET date = ?,
                starting_balance = ?,
                current_pnl = 0.0,
                is_paused = 0,
                updated_at = ?
            WHERE id = 1
        """, (new_date, new_sb, now))
        # Clear any per-symbol pauses from the previous day
        await self._conn.execute("DELETE FROM paused_symbols")
        await self._conn.commit()

        self._log.info(
            f"Circuit breaker reset for {new_date}: "
            f"starting_balance={new_sb:.2f}, is_paused=False"
        )

    async def set_starting_balance(self, balance: Decimal) -> None:
        """Set the daily starting balance."""
        today = self._utc_date()
        now = self._utc_now_str()
        await self._conn.execute("""
            UPDATE circuit_breaker
            SET starting_balance = ?,
                updated_at = ?
            WHERE id = 1 AND date = ?
        """, (float(balance), now, today))
        await self._conn.commit()

    async def get_daily_pnl(self) -> Decimal:
        """Get the current accumulated daily P&L."""
        state = await self.get_state()
        return Decimal(str(state["current_pnl"]))

    # ------------------------------------------------------------------
    # Closed trades persistence
    # ------------------------------------------------------------------

    async def record_closed_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str = "",
        leverage: int = 20,
        entry_time: str = "",
        strategy_name: str = "",
    ) -> None:
        """Persist a closed trade record to SQLite."""
        if not self._conn:
            return
        try:
            from datetime import datetime, timezone
            await self._conn.execute(
                "INSERT INTO closed_trades "
                "(symbol, side, entry_price, exit_price, pnl, reason, leverage, closed_at, entry_time, strategy_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, side, entry_price, exit_price, pnl, reason, leverage,
                 datetime.now(timezone.utc).isoformat(), entry_time, strategy_name),
            )
            await self._conn.commit()
        except Exception as exc:
            self._log.debug(f"Failed to persist closed trade for {symbol}: {exc}")

    async def get_recent_closed_trades(self, limit: int = 100) -> list[dict]:
        """Load recent closed trades from SQLite, newest first."""
        if not self._conn:
            return []
        try:
            cursor = await self._conn.execute(
                "SELECT * FROM closed_trades ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            self._log.debug(f"Failed to load closed trades: {exc}")
            return []

    # ------------------------------------------------------------------
    # Signal persistence
    # ------------------------------------------------------------------

    async def record_signal(
        self,
        signal_id: str,
        strategy_name: str,
        symbol: str,
        direction: str,
        confidence: float,
        price: float,
        timeframe: str = "",
        reason: str = "",
        regime: str = "",
        executed: bool = False,
    ) -> None:
        """Persist a trading signal to SQLite."""
        if not self._conn:
            return
        try:
            from datetime import datetime, timezone
            await self._conn.execute(
                "INSERT INTO signals (signal_id, strategy_name, symbol, direction, confidence, price, timeframe, reason, regime, executed, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (signal_id, strategy_name, symbol, direction, confidence, price, timeframe, reason, regime, int(executed),
                 datetime.now(timezone.utc).isoformat()),
            )
            await self._conn.commit()
        except Exception as exc:
            self._log.debug(f"Failed to persist signal: {exc}")

    async def get_recent_signals(self, limit: int = 100) -> list[dict]:
        """Load recent signals from SQLite, newest first."""
        if not self._conn:
            return []
        try:
            cursor = await self._conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            self._log.debug(f"Failed to load signals: {exc}")
            return []

    # ------------------------------------------------------------------
    # Trailing event persistence
    # ------------------------------------------------------------------

    async def record_trailing_event(
        self,
        symbol: str,
        event_type: str,
        entry_price: float,
        current_price: float,
        stop_loss: float,
        leverage: int = 1,
    ) -> None:
        """Persist a trailing stop event to SQLite."""
        if not self._conn:
            return
        try:
            from datetime import datetime, timezone
            await self._conn.execute(
                "INSERT INTO trailing_events (symbol, event_type, entry_price, current_price, stop_loss, leverage, event_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (symbol, event_type, entry_price, current_price, stop_loss, leverage,
                 datetime.now(timezone.utc).isoformat()),
            )
            await self._conn.commit()
        except Exception as exc:
            self._log.debug(f"Failed to persist trailing event: {exc}")

    async def get_recent_trailing_events(self, limit: int = 100) -> list[dict]:
        """Load recent trailing events from SQLite, newest first."""
        if not self._conn:
            return []
        try:
            cursor = await self._conn.execute(
                "SELECT * FROM trailing_events ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            self._log.debug(f"Failed to load trailing events: {exc}")
            return []

    # ------------------------------------------------------------------
    # Opportunity scan persistence
    # ------------------------------------------------------------------

    async def record_opportunity_scan(
        self,
        scan_time: str,
        symbol: str,
        direction: str,
        confidence: int,
        primary_timeframe: str = "",
        entry_notes: str = "",
        price: float = 0,
        brewing: bool = False,
        raw_json: str = "",
    ) -> None:
        """Persist an opportunity scan result to SQLite."""
        if not self._conn:
            return
        try:
            await self._conn.execute(
                "INSERT INTO opportunity_scans (scan_time, symbol, direction, confidence, primary_timeframe, entry_notes, price, brewing, raw_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (scan_time, symbol, direction, confidence, primary_timeframe, entry_notes, price, int(brewing), raw_json),
            )
            await self._conn.commit()
        except Exception as exc:
            self._log.debug(f"Failed to persist opportunity: {exc}")

    async def get_latest_opportunity_scan(self) -> list[dict]:
        """Load the most recent opportunity scan results."""
        if not self._conn:
            return []
        try:
            cursor = await self._conn.execute(
                "SELECT * FROM opportunity_scans ORDER BY id DESC LIMIT 20",
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            self._log.debug(f"Failed to load opportunities: {exc}")
            return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _pause(self) -> None:
        """Set the paused flag to True."""
        now = self._utc_now_str()
        await self._conn.execute("""
            UPDATE circuit_breaker
            SET is_paused = 1, updated_at = ?
            WHERE id = 1
        """, (now,))
        await self._conn.commit()

    async def _unpause(self) -> None:
        """Set the paused flag to False."""
        now = self._utc_now_str()
        await self._conn.execute("""
            UPDATE circuit_breaker
            SET is_paused = 0, updated_at = ?
            WHERE id = 1
        """, (now,))
        await self._conn.commit()

    @staticmethod
    def _utc_date() -> str:
        """Return today's date as YYYY-MM-DD in UTC."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _utc_now_str() -> str:
        """Return current UTC timestamp as ISO string."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
