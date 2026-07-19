"""Aurora Trader — Trade Sync Manager.

Periodically syncs trade data from Binance into local databases so the
learning server (TradeAnalyzer, ShadowAnalyzer) and circuit breaker
have data to work with.

Synced to:
  - data/winrate.db  → trade_results (for WinrateDB + ShadowAnalyzer)
  - data/trades.db   → trades (for TradeAnalyzer)
  - data/trading.db  → daily_pnl_log (for CircuitBreaker)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiosqlite

from shared.config import load_config
from shared.logger import get_logger

logger = get_logger("integration.trade_sync")

# Sync interval
SYNC_INTERVAL_SECONDS = 300  # every 5 minutes

# How far back to look for trades on first sync
INITIAL_LOOKBACK_DAYS = 7


TRADES_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    strategy TEXT DEFAULT 'mean_reversion',
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL,
    exit_price REAL,
    entry_time TEXT,
    exit_time TEXT,
    pnl REAL,
    pnl_pct REAL,
    volume REAL,
    status TEXT DEFAULT 'closed',
    source TEXT DEFAULT 'binance',
    created_at TEXT NOT NULL
);
"""


class TradeSyncManager:
    """Background task that syncs Binance trade data to local databases."""

    def __init__(self) -> None:
        self._cfg = load_config()
        self._log = logger
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._synced_ids: Set[str] = set()
        self._project_root = Path(__file__).resolve().parent.parent

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background sync loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        self._log.info(f"TradeSync started (every {SYNC_INTERVAL_SECONDS}s)")

    async def stop(self) -> None:
        """Stop the background sync loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._log.info("TradeSync stopped")

    # ------------------------------------------------------------------
    # Sync Loop
    # ------------------------------------------------------------------

    async def _sync_loop(self) -> None:
        """Continuously sync trades from Binance on a timer."""
        # Load already-synced trade IDs to avoid duplicates
        await self._load_synced_ids()

        while self._running:
            try:
                await self._sync_binance_trades()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.warning(f"Trade sync error: {exc}")

            # Wait for next interval (check every second for stop signal)
            for _ in range(SYNC_INTERVAL_SECONDS):
                if not self._running:
                    return
                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Binance Fetch
    # ------------------------------------------------------------------

    async def _sync_binance_trades(self) -> None:
        """Fetch recent Binance trades and record them locally."""
        try:
            from binance.client import Client

            key = self._cfg.exchange_api_key
            secret = self._cfg.exchange_api_secret
            if not key or not secret:
                return

            client = Client(key, secret)
            try:
                now_ms = int(time.time() * 1000)
                lookback = INITIAL_LOOKBACK_DAYS * 24 * 60 * 60 * 1000

                # 1. Futures income history for PnL + fees + funding
                income = client.futures_income_history(
                    startTime=now_ms - lookback, limit=500
                )

                new_trades: List[Dict[str, Any]] = []
                new_pnl_logs: List[Dict[str, Any]] = []

                for entry in income:
                    income_type = entry["incomeType"]
                    trade_id = f"binance_{entry['time']}_{income_type}"
                    if trade_id in self._synced_ids:
                        continue

                    symbol = entry.get("symbol", "—")
                    amount = float(entry["income"])
                    ts = int(entry["time"])
                    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                    date_str = dt.strftime("%Y-%m-%d")
                    time_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                    # Map income type to a readable reason
                    reason_map = {
                        "REALIZED_PNL": "trade_close",
                        "FUNDING_FEE": "funding",
                        "COMMISSION": "commission",
                        "TRANSFER": "transfer",
                        "INSURANCE_CLEAR": "insurance",
                    }
                    reason = reason_map.get(income_type, income_type.lower())

                    # Record in daily_pnl_log
                    new_pnl_logs.append({
                        "date": date_str,
                        "symbol": symbol,
                        "pnl": amount,
                        "reason": reason,
                        "timestamp": time_str,
                    })

                    # For REALIZED_PNL, also create a trade record
                    if income_type == "REALIZED_PNL":
                        # Determine side from PnL sign (Binance doesn't include side in income history)
                        # We'll try to get more detail from futures_account_trades
                        side = "LONG" if amount >= 0 else "SHORT"
                        version_tag = "v0.1.0"

                        new_trades.append({
                            "trade_id": f"binance_pnl_{entry['time']}",
                            "version_tag": version_tag,
                            "strategy": "mean_reversion",
                            "symbol": symbol,
                            "side": side,
                            "entry_price": 0.0,  # Not available from income history
                            "exit_price": 0.0,
                            "pnl": amount,
                            "rrr": 0.0,
                            "closed_at": time_str,
                        })

                    self._synced_ids.add(trade_id)

                client.close_connection()

                # Write to databases
                if new_pnl_logs:
                    await self._write_pnl_logs(new_pnl_logs)
                    self._log.info(f"Synced {len(new_pnl_logs)} PnL entries")

                if new_trades:
                    await self._write_trade_results(new_trades)
                    await self._write_trades_analyzer(new_trades)
                    self._log.info(f"Synced {len(new_trades)} closed trades")

                    # Update circuit breaker balance for any REALIZED_PNL
                    total_pnl = sum(t["pnl"] for t in new_trades)
                    if total_pnl != 0:
                        await self._update_circuit_breaker_pnl(new_trades)

            except Exception as exc:
                self._log.debug(f"Binance fetch failed: {exc}")
            finally:
                try:
                    client.close_connection()
                except Exception:
                    pass

        except ImportError:
            self._log.debug("binance.client not available — skipping sync")
        except Exception as exc:
            self._log.warning(f"Trade sync failed: {exc}")

    # ------------------------------------------------------------------
    # Database Writes
    # ------------------------------------------------------------------

    async def _write_pnl_logs(self, entries: List[Dict[str, Any]]) -> None:
        """Write PnL entries to trading.db daily_pnl_log."""
        db_path = self._project_root / "data" / "trading.db"
        try:
            async with aiosqlite.connect(str(db_path)) as db:
                for e in entries:
                    await db.execute(
                        """INSERT OR IGNORE INTO daily_pnl_log
                           (date, symbol, pnl, reason, timestamp)
                           VALUES (?, ?, ?, ?, ?)""",
                        (e["date"], e["symbol"], e["pnl"], e["reason"], e["timestamp"]),
                    )
                await db.commit()
        except Exception as exc:
            self._log.debug(f"Could not write PnL logs: {exc}")

    async def _write_trade_results(self, trades: List[Dict[str, Any]]) -> None:
        """Write trades to winrate.db trade_results table."""
        db_path = self._project_root / "data" / "winrate.db"
        try:
            async with aiosqlite.connect(str(db_path)) as db:
                for t in trades:
                    await db.execute(
                        """INSERT OR IGNORE INTO trade_results
                           (trade_id, version_tag, strategy, symbol, side,
                            entry_price, exit_price, pnl, rrr, closed_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (t["trade_id"], t["version_tag"], t["strategy"], t["symbol"],
                         t["side"], t["entry_price"], t["exit_price"],
                         t["pnl"], t["rrr"], t["closed_at"]),
                    )
                await db.commit()
        except Exception as exc:
            self._log.debug(f"Could not write trade results: {exc}")

    async def _write_trades_analyzer(self, trades: List[Dict[str, Any]]) -> None:
        """Write trades to trades.db (for TradeAnalyzer)."""
        db_path = self._project_root / "data" / "trades.db"
        try:
            async with aiosqlite.connect(str(db_path)) as db:
                # Ensure schema
                await db.executescript(TRADES_DB_SCHEMA)
                now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                for t in trades:
                    await db.execute(
                        """INSERT OR IGNORE INTO trades
                           (id, strategy, symbol, side, exit_price,
                            exit_time, pnl, status, source, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 'closed', 'binance', ?)""",
                        (t["trade_id"], t["strategy"], t["symbol"], t["side"],
                         t["exit_price"], t["closed_at"], t["pnl"], now),
                    )
                await db.commit()
        except Exception as exc:
            self._log.debug(f"Could not write analyzer trades: {exc}")

    async def _update_circuit_breaker_pnl(self, trades: List[Dict[str, Any]]) -> None:
        """Update circuit breaker with realized PnL from trades."""
        try:
            from trading_server.risk.circuit_breaker import CircuitBreaker

            cb = CircuitBreaker()
            await cb.initialize()
            for t in trades:
                await cb.record_trade_pnl(
                    symbol=t["symbol"],
                    pnl=Decimal(str(t["pnl"])),
                    reason=f"sync_trade_close",
                )
            await cb.close()
        except Exception as exc:
            self._log.debug(f"Could not update circuit breaker: {exc}")

    async def _load_synced_ids(self) -> None:
        """Load already-synced trade IDs from existing databases to avoid duplicates."""
        # Load from winrate.db trade_results
        db_path = self._project_root / "data" / "winrate.db"
        try:
            async with aiosqlite.connect(str(db_path)) as db:
                cursor = await db.execute("SELECT trade_id FROM trade_results")
                rows = await cursor.fetchall()
                for row in rows:
                    self._synced_ids.add(row[0])
        except Exception:
            pass

        # Load from trades.db
        db_path2 = self._project_root / "data" / "trades.db"
        try:
            async with aiosqlite.connect(str(db_path2)) as db:
                await db.executescript(TRADES_DB_SCHEMA)  # Ensure schema exists
                cursor = await db.execute("SELECT id FROM trades")
                rows = await cursor.fetchall()
                for row in rows:
                    self._synced_ids.add(row[0])
        except Exception:
            pass

        self._log.info(f"Loaded {len(self._synced_ids)} already-synced trade IDs")
