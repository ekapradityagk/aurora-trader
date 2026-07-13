"""
Aurora Trader — Safe Rollback Mechanism.

Monitors rolling 20-trade winrate for each strategy version and
auto-triggers rollback if WR drops below 50%. Maintains a list of
"known good" versions sorted by winrate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

from shared.config import load_config
from shared.logger import get_logger

logger = get_logger("integration.rollback")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = "data/integration.db"


# ---------------------------------------------------------------------------
# RollbackManager
# ---------------------------------------------------------------------------


class RollbackManager:
    """Monitors winrate performance and executes safe rollbacks.

    Usage::

        rm = RollbackManager()
        await rm.initialize()
        await rm.auto_rollback_check("v1.0.0", "ema_crossover")
        await rm.rollback_to("v0.9.0")
        known_good = await rm.get_known_good_versions()
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        winrate_db_path: Optional[str] = None,
    ) -> None:
        self._cfg = load_config()
        self._log = logger

        project_root = Path(__file__).resolve().parent.parent

        # Resolve DB paths
        if db_path:
            self._db_path = Path(db_path)
        else:
            db_rel = self._cfg.data.get("integration", {}).get("database", {}).get("path", _DEFAULT_DB_PATH)
            self._db_path = project_root / db_rel

        if winrate_db_path:
            self._winrate_db_path = Path(winrate_db_path)
        else:
            self._winrate_db_path = project_root / "data" / "winrate.db"

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Rollback config
        rb_cfg = self._cfg.data.get("risk_management", {}).get("auto_rollback", {})
        self._min_winrate = rb_cfg.get("min_winrate", 0.50)
        self._evaluation_window = rb_cfg.get("evaluation_window", 20)
        self._max_versions_to_keep = rb_cfg.get("max_versions_to_keep", 10)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the rollback tracking table."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS rollback_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_tag TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    rolled_back_to TEXT DEFAULT '',
                    success INTEGER DEFAULT 1,
                    metadata TEXT DEFAULT '{}'
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_rb_version "
                "ON rollback_log(version_tag)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_rb_strategy "
                "ON rollback_log(strategy)"
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS known_good_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_tag TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    winrate REAL DEFAULT 0.0,
                    profit_factor REAL DEFAULT 0.0,
                    total_trades INTEGER DEFAULT 0,
                    ranked_at TEXT NOT NULL,
                    UNIQUE(version_tag, strategy)
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_strategy "
                "ON known_good_versions(strategy)"
            )
            await db.commit()
        self._log.info("Rollback manager initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def auto_rollback_check(
        self, version_tag: str, strategy_name: str
    ) -> Optional[str]:
        """Check if the current version's winrate has dropped below threshold.

        Called by the trading server after each trade close.

        Args:
            version_tag: The active version tag to check.
            strategy_name: Name of the strategy being evaluated.

        Returns:
            The tag rolled-back-to if a rollback occurred, or None.
        """
        # Get rolling winrate from the winrate database
        rolling = await self._get_rolling_winrate(version_tag, self._evaluation_window)

        if rolling is None or rolling["total_trades"] < self._evaluation_window:
            # Not enough trades to evaluate
            self._log.debug(
                f"Version '{version_tag}' ({strategy_name}): "
                f"only {rolling['total_trades'] if rolling else 0}/{self._evaluation_window} "
                f"trades — skipping rollback check"
            )
            return None

        current_wr = rolling["winrate"]
        self._log.info(
            f"Rollback check for {version_tag} ({strategy_name}): "
            f"WR={current_wr:.4f} (threshold={self._min_winrate}, "
            f"window={self._evaluation_window} trades)"
        )

        if current_wr >= self._min_winrate:
            # Performance is acceptable — add to known good versions
            await self._update_known_good(version_tag, strategy_name, rolling)
            self._log.debug(
                f"Version '{version_tag}' winrate {current_wr:.2%} ≥ {self._min_winrate:.0%} — OK"
            )
            return None

        # Winrate is below threshold — trigger rollback
        self._log.warning(
            f"ROLLBACK TRIGGERED: version '{version_tag}' ({strategy_name}) "
            f"WR={current_wr:.2%} below threshold {self._min_winrate:.0%} "
            f"over last {self._evaluation_window} trades"
        )

        # Find the best known good version to roll back to
        best_known = await self._get_best_known_good(strategy_name, exclude=version_tag)

        if best_known is None:
            self._log.error(
                f"No known-good version found to roll back to for {strategy_name}"
            )
            await self._log_rollback(
                version_tag, strategy_name,
                f"WR={current_wr:.2%} below threshold, but no fallback version available",
                rolled_back_to="",
                success=False,
            )
            return None

        # Execute the rollback
        rolled_to = await self.rollback_to(best_known["version_tag"])

        if rolled_to:
            await self._log_rollback(
                version_tag, strategy_name,
                f"Auto-rollback: WR={current_wr:.2%} below {self._min_winrate:.0%}",
                rolled_back_to=rolled_to,
                success=True,
            )
            # Send alert
            self._send_alert(
                level="WARNING",
                message=f"Auto-rollback: {strategy_name} reverted from "
                        f"'{version_tag}' to '{rolled_to}' "
                        f"(WR={current_wr:.2%})",
            )

        return rolled_to

    async def rollback_to(self, tag: str) -> Optional[str]:
        """Revert the active strategy configuration to a previous version.

        Args:
            tag: The version tag to roll back to.

        Returns:
            The tag that was rolled back to (on success), or None.
        """
        self._log.info(f"Rolling back to version '{tag}'...")

        # In a real system, this would:
        #   1. Load the parameters from the version DB
        #   2. Write them to the strategy config
        #   3. Signal the trading server to reload config
        #   4. Signal the learning server to update its recommendation
        #
        # For now, we validate the tag exists and log the action.

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL")

            # Verify the version exists in the strategy_versions table
            cursor = await db.execute(
                "SELECT version_tag, strategy_name, parameters "
                "FROM strategy_versions WHERE version_tag = ?",
                (tag,),
            )
            row = await cursor.fetchone()
            if row is None:
                self._log.error(f"Version '{tag}' not found in version database")
                return None

            self._log.info(
                f"Rolled back to '{tag}' (strategy={row[1]}, "
                f"params={row[2][:80] if row[2] else '{}'}...)"
            )

        return tag

    async def get_known_good_versions(
        self, strategy_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return known good versions sorted by winrate (descending).

        Args:
            strategy_name: Optional filter to a specific strategy.

        Returns:
            List of dicts with version_tag, strategy, winrate, profit_factor,
            total_trades, ranked_at.
        """
        async with aiosqlite.connect(str(self._db_path)) as db:
            if strategy_name:
                cursor = await db.execute(
                    "SELECT version_tag, strategy, winrate, profit_factor, "
                    "total_trades, ranked_at "
                    "FROM known_good_versions "
                    "WHERE strategy = ? "
                    "ORDER BY winrate DESC, profit_factor DESC",
                    (strategy_name,),
                )
            else:
                cursor = await db.execute(
                    "SELECT version_tag, strategy, winrate, profit_factor, "
                    "total_trades, ranked_at "
                    "FROM known_good_versions "
                    "ORDER BY winrate DESC, profit_factor DESC",
                )
            rows = await cursor.fetchall()
            return [
                {
                    "version_tag": row[0],
                    "strategy": row[1],
                    "winrate": row[2],
                    "profit_factor": row[3],
                    "total_trades": row[4],
                    "ranked_at": row[5],
                }
                for row in rows
            ]

    async def get_rollback_history(
        self, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return recent rollback events."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            cursor = await db.execute(
                "SELECT version_tag, strategy, reason, triggered_at, "
                "rolled_back_to, success, metadata "
                "FROM rollback_log ORDER BY triggered_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "version_tag": row[0],
                    "strategy": row[1],
                    "reason": row[2],
                    "triggered_at": row[3],
                    "rolled_back_to": row[4],
                    "success": bool(row[5]),
                    "metadata": json.loads(row[6]) if row[6] else {},
                }
                for row in rows
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_rolling_winrate(
        self, version_tag: str, window: int
    ) -> Optional[Dict[str, Any]]:
        """Calculate the rolling winrate over the last N trades for a version.

        Reads from the winrate database's trade_results table.
        """
        try:
            async with aiosqlite.connect(str(self._winrate_db_path)) as db:
                await db.execute("PRAGMA journal_mode=WAL")

                # Get last N trades
                cursor = await db.execute(
                    "SELECT pnl FROM trade_results "
                    "WHERE version_tag = ? AND pnl IS NOT NULL "
                    "ORDER BY closed_at DESC LIMIT ?",
                    (version_tag, window),
                )
                rows = await cursor.fetchall()

                if not rows:
                    return None

                total = len(rows)
                wins = sum(1 for r in rows if r[0] > 0)
                losses = total - wins
                winrate = wins / total if total > 0 else 0.0

                # Profit factor for this window
                gross_profit = sum(r[0] for r in rows if r[0] > 0)
                gross_loss = sum(abs(r[0]) for r in rows if r[0] < 0)
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

                return {
                    "total_trades": total,
                    "wins": wins,
                    "losses": losses,
                    "winrate": winrate,
                    "profit_factor": profit_factor,
                }
        except Exception as exc:
            self._log.error(f"Failed to query rolling winrate: {exc}")
            return None

    async def _update_known_good(
        self,
        version_tag: str,
        strategy_name: str,
        stats: Dict[str, Any],
    ) -> None:
        """Add or update a version in the known-good list."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            now = datetime.now(timezone.utc).isoformat()

            await db.execute(
                "INSERT OR REPLACE INTO known_good_versions "
                "(version_tag, strategy, winrate, profit_factor, "
                " total_trades, ranked_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    version_tag,
                    strategy_name,
                    stats.get("winrate", 0.0),
                    stats.get("profit_factor", 0.0),
                    stats.get("total_trades", 0),
                    now,
                ),
            )

            # Prune old entries beyond max
            await db.execute(
                "DELETE FROM known_good_versions WHERE id IN ("
                "SELECT id FROM known_good_versions "
                "WHERE strategy = ? "
                "ORDER BY winrate DESC "
                "LIMIT -1 OFFSET ?"
                ")",
                (strategy_name, self._max_versions_to_keep),
            )
            await db.commit()

    async def _get_best_known_good(
        self, strategy_name: str, exclude: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the best known-good version for a strategy.

        Excludes a specific version (the one that is failing).
        """
        async with aiosqlite.connect(str(self._db_path)) as db:
            if exclude:
                cursor = await db.execute(
                    "SELECT version_tag, winrate, profit_factor, total_trades "
                    "FROM known_good_versions "
                    "WHERE strategy = ? AND version_tag != ? "
                    "ORDER BY winrate DESC, profit_factor DESC LIMIT 1",
                    (strategy_name, exclude),
                )
            else:
                cursor = await db.execute(
                    "SELECT version_tag, winrate, profit_factor, total_trades "
                    "FROM known_good_versions "
                    "WHERE strategy = ? "
                    "ORDER BY winrate DESC, profit_factor DESC LIMIT 1",
                    (strategy_name,),
                )
            row = await cursor.fetchone()
            if row is None:
                return None
            return {
                "version_tag": row[0],
                "winrate": row[1],
                "profit_factor": row[2],
                "total_trades": row[3],
            }

    async def _log_rollback(
        self,
        version_tag: str,
        strategy: str,
        reason: str,
        rolled_back_to: str = "",
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a rollback event in the database."""
        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                "INSERT INTO rollback_log "
                "(version_tag, strategy, reason, triggered_at, "
                " rolled_back_to, success, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    version_tag,
                    strategy,
                    reason,
                    datetime.now(timezone.utc).isoformat(),
                    rolled_back_to,
                    1 if success else 0,
                    json.dumps(metadata or {}),
                ),
            )
            await db.commit()

    def _send_alert(self, level: str, message: str) -> None:
        """Send an alert about a rollback event.

        In production this would email, Slack, or push notify.
        For now we just log it prominently.
        """
        if level == "WARNING":
            self._log.warning(f"⚠️  ALERT: {message}")
        elif level == "CRITICAL":
            self._log.critical(f"🚨 ALERT: {message}")
        else:
            self._log.info(f"📢 ALERT: {message}")
