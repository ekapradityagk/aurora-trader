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

        # Check if we need to pause
        state = await self.get_state()
        if state["loss_pct"] >= self._loss_limit_pct and not state["is_paused"]:
            await self._pause()
            self._log.critical(
                f"DAILY LOSS LIMIT BREACHED: "
                f"PnL={state['current_pnl']:.2f} "
                f"({state['loss_pct']:.2f}% ≥ {self._loss_limit_pct}%). "
                f"Trading PAUSED until UTC midnight."
            )

        return state

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
