"""
Aurora Trader — Winrate Tracking Database.

Tracks trade performance per strategy version in SQLite:
  - version_winrate: aggregated stats per version
  - trade_results: individual trade records with PnL and R:R
  - daily_summary: daily aggregated metrics
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

from shared.config import load_config
from shared.logger import get_logger
from shared.models import Trade

logger = get_logger("integration.winrate_db")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = "data/trading.db"


# ---------------------------------------------------------------------------
# WinrateDB
# ---------------------------------------------------------------------------


class WinrateDB:
    """Manages winrate tracking per strategy version using SQLite.

    All write operations are async (using aiosqlite). Provides methods
    to record trades, query version stats, find the best version, and
    compare two versions side-by-side.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._cfg = load_config()
        self._log = logger

        # Resolve DB path — always use the winrate-specific path,
        # not the general integration database config.
        project_root = Path(__file__).resolve().parent.parent
        if db_path:
            self._db_path = Path(db_path)
        else:
            self._db_path = project_root / _DEFAULT_DB_PATH

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Database initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL")

            # Version winrate stats
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS version_winrate (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_tag TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    winrate REAL DEFAULT 0.0,
                    profit_factor REAL DEFAULT 0.0,
                    sharpe REAL DEFAULT 0.0,
                    avg_rr REAL DEFAULT 0.0,
                    period_start TEXT,
                    period_end TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # Individual trade results — now a VIEW pointing to closed_trades
            # Table is already created as a view during Phase 3 migration.
            # Skip table creation to avoid conflicts with the view.
            await db.execute(
                "CREATE VIEW IF NOT EXISTS trade_results AS "
                "SELECT id AS trade_id, version_tag, strategy_name AS strategy, symbol, side, "
                "       entry_price, exit_price, pnl, 0.0 AS rrr, closed_at "
                "FROM closed_trades WHERE pnl IS NOT NULL"
            )
            # Daily summary
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    version_tag TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winrate REAL DEFAULT 0.0,
                    pnl REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL
                )
                """
            )

            # Indexes
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_vw_version "
                "ON version_winrate(version_tag)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_vw_strategy "
                "ON version_winrate(strategy)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_tr_version "
                "ON closed_trades(strategy_name)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_tr_closed "
                "ON closed_trades(closed_at)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ds_date "
                "ON daily_summary(date)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ds_version "
                "ON daily_summary(version_tag)"
            )

            # Trigger: redirect INSERTs on the view to closed_trades
            await db.execute("""\
                CREATE TRIGGER IF NOT EXISTS trade_results_insert
                INSTEAD OF INSERT ON trade_results
                BEGIN
                    INSERT OR IGNORE INTO closed_trades
                        (symbol, side, entry_price, exit_price, pnl, reason, leverage, closed_at, strategy_name, version_tag)
                    VALUES (
                        NEW.symbol,
                        NEW.side,
                        NEW.entry_price,
                        NEW.exit_price,
                        NEW.pnl,
                        'winrate_db',
                        1,
                        NEW.closed_at,
                        NEW.strategy,
                        NEW.version_tag
                    );
                END
            """)

            await db.commit()

        self._log.info(f"Winrate database initialised at {self._db_path}")

    # ------------------------------------------------------------------
    # Recording trades
    # ------------------------------------------------------------------

    async def record_trade(self, trade: Trade) -> None:
        """Record a completed trade and update version winrate stats.

        This should be called by the trading server after a trade closes.

        Args:
            trade: A ``Trade`` model instance with PnL and exit price set.
        """
        if not trade.is_closed or trade.pnl is None:
            self._log.warning(
                f"Trade {trade.id} is not closed — skipping winrate update"
            )
            return

        version_tag = trade.metadata.get("version", "")

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL")

            # 1. Insert trade result
            pnl_float = float(trade.pnl)
            rrr = self._calculate_rrr(trade)

            await db.execute(
                "INSERT OR REPLACE INTO trade_results "
                "(trade_id, version_tag, strategy, symbol, side, "
                " entry_price, exit_price, pnl, rrr, closed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trade.id,
                    version_tag,
                    trade.strategy_name,
                    trade.symbol,
                    trade.side.value,
                    float(trade.entry_price),
                    float(trade.exit_price) if trade.exit_price else None,
                    pnl_float,
                    rrr,
                    trade.exit_time.isoformat() if trade.exit_time
                    else datetime.now(timezone.utc).isoformat(),
                ),
            )

            # 2. Update version_winrate
            await self._update_version_stats(db, version_tag, trade.strategy_name, pnl_float, rrr)

            # 3. Update daily summary
            today = date.today().isoformat()
            await self._update_daily_summary(db, today, version_tag, pnl_float)

            await db.commit()

        self._log.debug(
            f"Recorded trade {trade.id[:12]} for version {version_tag} "
            f"pnl={pnl_float:.2f} rrr={rrr:.2f}"
        )

    async def _update_version_stats(
        self, db: aiosqlite.Connection, version_tag: str, strategy: str,
        pnl: float, rrr: float
    ) -> None:
        """Update aggregated winrate stats for a version after a new trade."""
        now = datetime.now(timezone.utc).isoformat()

        # Check if row exists
        cursor = await db.execute(
            "SELECT id, total_trades, wins, losses, profit_factor, "
            "sharpe, avg_rr, period_start FROM version_winrate "
            "WHERE version_tag = ? AND strategy = ?",
            (version_tag, strategy),
        )
        row = await cursor.fetchone()

        if row is None:
            # First trade for this version
            is_win = 1 if pnl > 0 else 0
            winrate = 1.0 if is_win else 0.0
            await db.execute(
                "INSERT INTO version_winrate "
                "(version_tag, strategy, total_trades, wins, losses, "
                " winrate, profit_factor, sharpe, avg_rr, "
                " period_start, period_end, created_at, updated_at) "
                "VALUES (?, ?, 1, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
                (
                    version_tag,
                    strategy,
                    is_win,
                    1 - is_win,
                    winrate,
                    max(pnl, 0.0) if pnl > 0 else 0.0,  # profit_factor placeholder
                    rrr,
                    now,
                    now,
                    now,
                    now,
                ),
            )
        else:
            # Update existing row
            total = row[1] + 1
            wins = row[2] + (1 if pnl > 0 else 0)
            losses = row[3] + (1 if pnl <= 0 else 0)
            winrate = wins / total if total > 0 else 0.0

            # Running profit factor: sum of all positive PnL / sum of |negative PnL|
            prev_pf = row[4] if row[4] else 0.0
            prev_avg_rr = row[6] if row[6] else 0.0

            # We recalculate profit factor from trade_results for accuracy
            # (simple approximation: use running total)
            cursor2 = await db.execute(
                "SELECT SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), "
                "SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END), "
                "AVG(CASE WHEN rrr IS NOT NULL THEN rrr ELSE 0 END) "
                "FROM trade_results WHERE version_tag = ?",
                (version_tag,),
            )
            sum_row = await cursor2.fetchone()
            gross_profit = sum_row[0] if sum_row and sum_row[0] else 0.0
            gross_loss = sum_row[1] if sum_row and sum_row[1] else 0.0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
            avg_rr = sum_row[2] if sum_row and sum_row[2] else rrr

            # Simple sharpe: mean(PnL) / std(PnL) * sqrt(trades) approximated
            cursor3 = await db.execute(
                "SELECT AVG(pnl), SUM(pnl * pnl) FROM trade_results WHERE version_tag = ?",
                (version_tag,),
            )
            stats_row = await cursor3.fetchone()
            avg_pnl = stats_row[0] if stats_row and stats_row[0] else 0.0
            sum_sq = stats_row[1] if stats_row and stats_row[1] else 0.0
            variance = (sum_sq / total) - (avg_pnl * avg_pnl) if total > 0 else 0
            std_pnl = (variance ** 0.5) if variance > 0 else 0.0001
            sharpe = (avg_pnl / std_pnl) * (total ** 0.5) if std_pnl > 0 else 0.0

            period_start = row[7] if row[7] else now

            await db.execute(
                "UPDATE version_winrate SET total_trades=?, wins=?, losses=?, "
                "winrate=?, profit_factor=?, sharpe=?, avg_rr=?, "
                "period_end=?, updated_at=? "
                "WHERE version_tag=? AND strategy=?",
                (
                    total, wins, losses, winrate,
                    profit_factor, sharpe, avg_rr,
                    now, now, version_tag, strategy,
                ),
            )

    async def _update_daily_summary(
        self, db: aiosqlite.Connection, day: str, version_tag: str, pnl: float
    ) -> None:
        """Upsert the daily summary row for a given date+version."""
        cursor = await db.execute(
            "SELECT id, total_trades, winrate, pnl FROM daily_summary "
            "WHERE date = ? AND version_tag = ?",
            (day, version_tag),
        )
        row = await cursor.fetchone()

        if row is None:
            await db.execute(
                "INSERT INTO daily_summary "
                "(date, version_tag, total_trades, winrate, pnl, created_at) "
                "VALUES (?, ?, 1, ?, ?, ?)",
                (
                    day,
                    version_tag,
                    1.0 if pnl > 0 else 0.0,
                    pnl,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        else:
            total = row[1] + 1
            # We need win/loss count for the day
            cursor2 = await db.execute(
                "SELECT COUNT(*) FROM trade_results "
                "WHERE version_tag = ? AND closed_at LIKE ? AND pnl > 0",
                (version_tag, f"{day}%"),
            )
            day_wins = (await cursor2.fetchone())[0]
            day_winrate = day_wins / total if total > 0 else 0.0
            new_pnl = row[3] + pnl

            await db.execute(
                "UPDATE daily_summary SET total_trades=?, winrate=?, pnl=? "
                "WHERE date=? AND version_tag=?",
                (total, day_winrate, new_pnl, day, version_tag),
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_version_winrate(self, tag: str) -> Optional[Dict[str, Any]]:
        """Return winrate metrics for a specific version tag.

        Returns:
            Dict with keys: version_tag, strategy, total_trades, wins, losses,
            winrate, profit_factor, sharpe, avg_rr, period_start, period_end.
            Returns None if the tag is not found.
        """
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cursor = await db.execute(
                "SELECT version_tag, strategy, total_trades, wins, losses, "
                "winrate, profit_factor, sharpe, avg_rr, "
                "period_start, period_end "
                "FROM version_winrate WHERE version_tag = ? "
                "ORDER BY updated_at DESC LIMIT 1",
                (tag,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return {
                "version_tag": row[0],
                "strategy": row[1],
                "total_trades": row[2],
                "wins": row[3],
                "losses": row[4],
                "winrate": row[5],
                "profit_factor": row[6],
                "sharpe": row[7],
                "avg_rr": row[8],
                "period_start": row[9],
                "period_end": row[10],
            }

    async def get_best_version(self) -> Optional[Dict[str, Any]]:
        """Return the version with the highest winrate (minimum 10 trades).

        Returns:
            The best version's full stats dict, or None if no version has
            enough trades.
        """
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cursor = await db.execute(
                "SELECT version_tag, strategy, total_trades, wins, losses, "
                "winrate, profit_factor, sharpe, avg_rr, "
                "period_start, period_end "
                "FROM version_winrate "
                "WHERE total_trades >= 10 "
                "ORDER BY winrate DESC, profit_factor DESC "
                "LIMIT 1",
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return {
                "version_tag": row[0],
                "strategy": row[1],
                "total_trades": row[2],
                "wins": row[3],
                "losses": row[4],
                "winrate": row[5],
                "profit_factor": row[6],
                "sharpe": row[7],
                "avg_rr": row[8],
                "period_start": row[9],
                "period_end": row[10],
            }

    async def compare_versions(
        self, tag_a: str, tag_b: str
    ) -> Dict[str, Any]:
        """Return a side-by-side comparison of two versions.

        Args:
            tag_a: First version tag.
            tag_b: Second version tag.

        Returns:
            Dict with 'version_a', 'version_b' keys (each containing stats),
            and 'delta' with the difference.
        """
        a_stats = await self.get_version_winrate(tag_a) or {"version_tag": tag_a, "error": "not found"}
        b_stats = await self.get_version_winrate(tag_b) or {"version_tag": tag_b, "error": "not found"}

        # Compute deltas for numeric fields
        delta = {}
        for key in ["total_trades", "wins", "losses", "winrate", "profit_factor", "sharpe", "avg_rr"]:
            a_val = a_stats.get(key, 0) if isinstance(a_stats, dict) else 0
            b_val = b_stats.get(key, 0) if isinstance(b_stats, dict) else 0
            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                delta[key] = round(b_val - a_val, 4)
            else:
                delta[key] = None

        return {
            "version_a": a_stats,
            "version_b": b_stats,
            "delta": delta,
        }

    async def get_recent_trades(
        self, version_tag: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return recent trade results from closed_trades (via trade_results view)."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            if version_tag:
                cursor = await db.execute(
                    "SELECT trade_id, version_tag, strategy, symbol, side, "
                    "entry_price, exit_price, pnl, rrr, closed_at "
                    "FROM trade_results WHERE version_tag = ? "
                    "ORDER BY closed_at DESC LIMIT ?",
                    (version_tag, limit),
                )
            else:
                cursor = await db.execute(
                    "SELECT trade_id, version_tag, strategy, symbol, side, "
                    "entry_price, exit_price, pnl, rrr, closed_at "
                    "FROM trade_results ORDER BY closed_at DESC LIMIT ?",
                    (limit,),
                )
            rows = await cursor.fetchall()
            return [
                {
                    "trade_id": row[0],
                    "version_tag": row[1],
                    "strategy": row[2],
                    "symbol": row[3],
                    "side": row[4],
                    "entry_price": row[5],
                    "exit_price": row[6],
                    "pnl": row[7],
                    "rrr": row[8],
                    "closed_at": row[9],
                }
                for row in rows
            ]

    async def get_daily_summaries(
        self, version_tag: Optional[str] = None, limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Return daily summaries, optionally filtered by version."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            if version_tag:
                cursor = await db.execute(
                    "SELECT date, version_tag, total_trades, winrate, pnl "
                    "FROM daily_summary WHERE version_tag = ? "
                    "ORDER BY date DESC LIMIT ?",
                    (version_tag, limit),
                )
            else:
                cursor = await db.execute(
                    "SELECT date, version_tag, total_trades, winrate, pnl "
                    "FROM daily_summary ORDER BY date DESC LIMIT ?",
                    (limit,),
                )
            rows = await cursor.fetchall()
            return [
                {
                    "date": row[0],
                    "version_tag": row[1],
                    "total_trades": row[2],
                    "winrate": row[3],
                    "pnl": row[4],
                }
                for row in rows
            ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_rrr(trade: Trade) -> float:
        """Calculate risk-reward ratio from a closed trade.

        If the trade has explicit stop/take-profit metadata, use that.
        Otherwise infer from entry, exit, and assumed risk (entry - stop).
        """
        if trade.exit_price is None or trade.entry_price == 0:
            return 0.0

        meta = trade.metadata or {}
        stop_loss_str = meta.get("stop_loss", "")
        if stop_loss_str:
            try:
                stop_loss = float(stop_loss_str)
                entry = float(trade.entry_price)
                exit_p = float(trade.exit_price)
                if trade.side.value == "buy":
                    risk = entry - stop_loss
                    reward = exit_p - entry
                else:
                    risk = stop_loss - entry
                    reward = entry - exit_p
                if abs(risk) > 0:
                    return round(reward / risk, 4)
            except (ValueError, ZeroDivisionError):
                pass

        # Fallback: just use pnl as a ratio (unit-less)
        return abs(float(trade.pnl)) / (float(trade.entry_price) * float(trade.quantity)) if trade.quantity > 0 else 0.0
