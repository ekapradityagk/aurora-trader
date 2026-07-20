"""
Aurora Trader — Strategy Selector.

Selects the active trading strategy based on the current market regime and
per-strategy historical performance.  Supports three core strategies:

    - mean_reversion   (best in RANGING / LOW_VOLATILITY regimes)
    - rsi_divergence   (best in VOLATILE / BREAKOUT regimes)
    - trend_follow     (best in TRENDING_BULL / TRENDING_BEAR regimes)

The selector maintains an active version tag that is consumed by the trading
server.  All selection decisions are logged to the SQLite database for audit.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

from shared.config import load_config
from shared.logger import get_logger
from shared.models import MarketRegimeType

from learning_server.regime import (
    REGIME_TO_STRATEGY,
    RegimeDetector,
    RegimeResult,
)

logger = get_logger("learning_server.strategy_selector")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_NAMES = ["mean_reversion", "rsi_divergence", "trend_follow"]

# Minimum win rate for a strategy to be considered "healthy"
MIN_STRATEGY_WINRATE = 0.40

# How far back to look for strategy performance
PERFORMANCE_LOOKBACK_TRADES = 50

# Force-switch if a strategy's WR drops below this over the window
FORCE_SWITCH_WINRATE = 0.35

# Default strategy when no regime is available
DEFAULT_STRATEGY = "mean_reversion"


# ---------------------------------------------------------------------------
# Selection Record
# ---------------------------------------------------------------------------


@dataclass
class SelectionRecord:
    """A single strategy selection decision, logged to SQLite."""

    id: str = ""
    timestamp: str = ""
    selected_strategy: str = ""
    version_tag: str = ""
    market_regime: str = ""
    regime_confidence: float = 0.0
    strategy_performance: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    previous_strategy: str = ""


# ---------------------------------------------------------------------------
# Strategy Selector
# ---------------------------------------------------------------------------


class StrategySelector:
    """Selects the optimal strategy based on market regime and performance.

    Maintains an active version tag that is written to a JSON file for the
    trading server to consume.  All decisions are logged to SQLite.

    Usage::

        selector = StrategySelector()
        decision = await selector.select()
    """

    def __init__(
        self,
        db_path: str = "data/trades.db",
        cb_db_path: str = "data/trading.db",
        state_file: str = "data/active_strategy.json",
    ) -> None:
        self._db_path = db_path
        self._cb_db_path = cb_db_path
        self._state_file = Path(state_file)
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._log = logger
        self._cfg = load_config()
        self._regime_detector = RegimeDetector()

        # Current active strategy (loaded from file on init)
        self._active_strategy: str = DEFAULT_STRATEGY
        self._active_version: str = "1.0.0"
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def select(
        self,
        symbol: str = "BTCUSDT",
        regime_result: Optional[RegimeResult] = None,
    ) -> SelectionRecord:
        """Select the best strategy for the current market conditions.

        Args:
            symbol: The trading pair to consider.
            regime_result: Optional pre-computed regime result.  If not
                           provided, the selector will query the RegimeDetector.

        Returns:
            A SelectionRecord with the chosen strategy and metadata.
        """
        previous = self._active_strategy

        # 1. Determine current regime (or use provided one)
        if regime_result is None:
            try:
                # We need OHLCV data; in a real scenario the caller passes it.
                # For now, we'll try to detect without data — if no cache,
                # we fall back to the current active strategy.
                regime_result = await self._regime_detector.detect(
                    symbol, "1h"
                )
            except Exception as exc:
                self._log.warning(
                    f"Could not detect regime: {exc}. "
                    f"Keeping current strategy: {self._active_strategy}"
                )
                return SelectionRecord(
                    id=uuid.uuid4().hex[:16],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    selected_strategy=self._active_strategy,
                    version_tag=self._active_version,
                    market_regime="unknown",
                    regime_confidence=0.0,
                    strategy_performance={},
                    reason=f"Regime detection unavailable; keeping current",
                    previous_strategy=previous,
                )

        # 2. Get per-strategy performance from recent trades
        perf = await self._get_strategy_performance()

        # 3. Determine recommended strategy from regime
        regime_strategy = REGIME_TO_STRATEGY.get(
            regime_result.regime, DEFAULT_STRATEGY
        )

        # 4. Adjust based on performance
        selected, reason = self._resolve_strategy(
            regime_strategy,
            regime_result.regime,
            regime_result.confidence,
            perf,
        )

        # 5. Update active state if changed
        if selected != self._active_strategy:
            self._log.info(
                f"Strategy switch: {self._active_strategy} → {selected} "
                f"(regime={regime_result.regime.value}, "
                f"confidence={regime_result.confidence:.2f})"
            )
            self._active_strategy = selected
            self._bump_version(selected)

        # 6. Build record
        record = SelectionRecord(
            id=uuid.uuid4().hex[:16],
            timestamp=datetime.now(timezone.utc).isoformat(),
            selected_strategy=selected,
            version_tag=self._active_version,
            market_regime=regime_result.regime.value,
            regime_confidence=regime_result.confidence,
            strategy_performance=perf,
            reason=reason,
            previous_strategy=previous,
        )

        # 7. Persist state
        self._save_state()
        await self._log_selection(record)

        self._log.info(
            f"Selected: {selected} (v{self._active_version}) | "
            f"Regime: {regime_result.regime.value} "
            f"({regime_result.confidence:.2f}) | {reason}"
        )

        return record

    def get_active_strategy(self) -> Tuple[str, str]:
        """Return the currently active (strategy_name, version_tag)."""
        return self._active_strategy, self._active_version

    # ------------------------------------------------------------------
    # Strategy resolution
    # ------------------------------------------------------------------

    def _resolve_strategy(
        self,
        regime_strategy: str,
        regime: MarketRegimeType,
        regime_confidence: float,
        perf: Dict[str, float],
    ) -> Tuple[str, str]:
        """Resolve the final strategy by combining regime recommendation
        with recent performance data.

        Returns (strategy_name, reason).
        """
        # If regime confidence is very low, prefer the current strategy
        min_conf = self._cfg.regime_detection.get("min_regime_confidence", 0.6)
        if regime_confidence < min_conf:
            # Stick with current unless it's performing very badly
            current_wr = perf.get(self._active_strategy, 0.5)
            if current_wr >= FORCE_SWITCH_WINRATE:
                return (
                    self._active_strategy,
                    f"Low regime confidence ({regime_confidence:.2f}); "
                    f"keeping {self._active_strategy} (WR={current_wr:.2f})",
                )

        # Check if the regime-recommended strategy is healthy
        rec_wr = perf.get(regime_strategy, 0.5)
        if rec_wr >= MIN_STRATEGY_WINRATE:
            return (
                regime_strategy,
                f"Regime ({regime.value}) recommends {regime_strategy} "
                f"(WR={rec_wr:.2f})",
            )

        # The recommended strategy is underperforming — try alternatives
        candidates = sorted(
            [
                (s, perf.get(s, 0.0))
                for s in STRATEGY_NAMES
                if s != regime_strategy
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        for candidate, wr in candidates:
            if wr >= MIN_STRATEGY_WINRATE:
                return (
                    candidate,
                    f"Regime ({regime.value}) suggests {regime_strategy} "
                    f"(WR={rec_wr:.2f}) but it's underperforming; "
                    f"switching to {candidate} (WR={wr:.2f})",
                )

        # All strategies are underperforming — fall back to default
        return (
            DEFAULT_STRATEGY,
            f"All strategies underperforming; falling back to "
            f"{DEFAULT_STRATEGY}",
        )

    # ------------------------------------------------------------------
    # Performance lookback
    # ------------------------------------------------------------------

    async def _get_strategy_performance(
        self,
    ) -> Dict[str, float]:
        """Get recent win rate for each strategy from all trade sources.

        Queries trades.db (legacy) and trading.db closed_trades (real-time)
        to get the most recent per-strategy win rates. Returns neutral (0.5)
        for strategies with no data.
        """
        perf: Dict[str, float] = {}
        all_trades: Dict[str, List[float]] = {s: [] for s in STRATEGY_NAMES}

        # 1. Load from trades.db (legacy TradeSync data)
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                for s in STRATEGY_NAMES:
                    cursor = await db.execute(
                        """
                        SELECT pnl, pnl_pct
                        FROM trades
                        WHERE strategy = ?
                          AND exit_price IS NOT NULL
                          AND pnl IS NOT NULL
                        ORDER BY exit_time DESC
                        LIMIT ?
                        """,
                        (s, PERFORMANCE_LOOKBACK_TRADES),
                    )
                    rows = await cursor.fetchall()
                    for r in rows:
                        all_trades[s].append(float(r["pnl"] or 0))
        except Exception as exc:
            self._log.debug(f"Could not load legacy perf: {exc}")

        # 2. Load from circuit breaker's closed_trades (REAL strategy names + prices)
        try:
            async with aiosqlite.connect(self._cb_db_path) as db:
                db.row_factory = aiosqlite.Row
                for s in STRATEGY_NAMES:
                    cursor = await db.execute(
                        """
                        SELECT pnl
                        FROM closed_trades
                        WHERE strategy_name = ?
                          AND exit_price IS NOT NULL
                          AND exit_price != 0
                        ORDER BY closed_at DESC
                        LIMIT ?
                        """,
                        (s, PERFORMANCE_LOOKBACK_TRADES),
                    )
                    rows = await cursor.fetchall()
                    for r in rows:
                        all_trades[s].append(float(r["pnl"] or 0))
                # Also grab trades from exchange_sync / auto_signal strategies
                cursor = await db.execute(
                    """
                    SELECT strategy_name, pnl
                    FROM closed_trades
                    WHERE strategy_name NOT IN ('', 'mean_reversion', 'rsi_divergence', 'trend_follow')
                      AND exit_price IS NOT NULL
                      AND exit_price != 0
                    ORDER BY closed_at DESC
                    LIMIT 50
                    """
                )
                rows = await cursor.fetchall()
                # Distribute unknown-strategy trades to the "best guess" strategy
                # based on the strategy_selector's current active strategy
                for r in rows:
                    s = r["strategy_name"] or "unknown"
                    if s not in all_trades:
                        all_trades[s] = []
                    all_trades[s].append(float(r["pnl"] or 0))
        except Exception as exc:
            self._log.debug(f"Could not load CB perf: {exc}")

        # Compute win rates
        for s in STRATEGY_NAMES:
            pnls = all_trades.get(s, [])
            if len(pnls) >= 3:
                wins = sum(1 for p in pnls if p > 0)
                perf[s] = wins / len(pnls)
                self._log.debug(f"  {s}: {wins}/{len(pnls)} wins = {perf[s]:.2f}")
            else:
                perf[s] = 0.5  # neutral — insufficient data

        # Also include any extra strategies we discovered
        for s, pnls in all_trades.items():
            if s not in STRATEGY_NAMES and len(pnls) >= 3:
                wins = sum(1 for p in pnls if p > 0)
                perf[s] = wins / len(pnls)

        return perf

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def _bump_version(self, strategy: str) -> None:
        """Increment the active version tag.

        Uses a simple major.minor.patch scheme:
            - major: incremented on strategy switch
            - minor: incremented on hyperopt-driven param update
            - patch: incremented on minor adjustments
        """
        try:
            parts = self._active_version.split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
        except (ValueError, IndexError):
            major, minor, patch = 1, 0, 0

        # Major bump on strategy switch
        new_version = f"{major + 1}.0.0"
        self._active_version = new_version
        self._log.info(f"Version bumped to {new_version} (strategy={strategy})")

    def update_params_version(self) -> None:
        """Bump the minor version (called when hyperopt updates params)."""
        try:
            parts = self._active_version.split(".")
            major = int(parts[0]) if len(parts) > 0 else 1
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
        except (ValueError, IndexError):
            major, minor, patch = 1, 0, 0

        new_version = f"{major}.{minor + 1}.{patch}"
        self._active_version = new_version
        self._log.info(f"Params version bumped to {new_version}")
        self._save_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Write the active strategy + version to a JSON file."""
        data = {
            "active_strategy": self._active_strategy,
            "active_version": self._active_version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self._state_file, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as exc:
            self._log.warning(f"Failed to save strategy state: {exc}")

    def _load_state(self) -> None:
        """Load the active strategy + version from the JSON file."""
        if not self._state_file.is_file():
            return
        try:
            with open(self._state_file) as f:
                data = json.load(f)
            self._active_strategy = data.get("active_strategy", DEFAULT_STRATEGY)
            self._active_version = data.get("active_version", "1.0.0")
            self._log.info(
                f"Loaded state: {self._active_strategy} "
                f"v{self._active_version}"
            )
        except (IOError, json.JSONDecodeError) as exc:
            self._log.warning(f"Failed to load strategy state: {exc}")

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------

    async def _log_selection(self, record: SelectionRecord) -> None:
        """Log the selection decision to the SQLite trade database.

        Creates a ``strategy_selections`` table if it doesn't exist.
        """
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS strategy_selections (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        selected_strategy TEXT NOT NULL,
                        version_tag TEXT NOT NULL,
                        market_regime TEXT,
                        regime_confidence REAL,
                        strategy_performance TEXT,
                        reason TEXT,
                        previous_strategy TEXT
                    )
                    """
                )
                await db.execute(
                    """
                    INSERT INTO strategy_selections
                        (id, timestamp, selected_strategy, version_tag,
                         market_regime, regime_confidence,
                         strategy_performance, reason, previous_strategy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.timestamp,
                        record.selected_strategy,
                        record.version_tag,
                        record.market_regime,
                        record.regime_confidence,
                        json.dumps(record.strategy_performance),
                        record.reason,
                        record.previous_strategy,
                    ),
                )
                await db.commit()
        except Exception as exc:
            self._log.warning(f"Failed to log strategy selection: {exc}")

    async def get_selection_history(
        self,
        limit: int = 20,
    ) -> List[SelectionRecord]:
        """Retrieve recent strategy selection decisions from the log."""
        records: List[SelectionRecord] = []
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT * FROM strategy_selections
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = await cursor.fetchall()
                for row in rows:
                    records.append(
                        SelectionRecord(
                            id=row["id"],
                            timestamp=row["timestamp"],
                            selected_strategy=row["selected_strategy"],
                            version_tag=row["version_tag"],
                            market_regime=row["market_regime"],
                            regime_confidence=row["regime_confidence"],
                            strategy_performance=json.loads(
                                row["strategy_performance"] or "{}"
                            ),
                            reason=row["reason"],
                            previous_strategy=row["previous_strategy"],
                        )
                    )
        except Exception as exc:
            self._log.warning(
                f"Could not load selection history: {exc}"
            )
        return records
