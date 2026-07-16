"""Aurora Trader — Pair Performance Ranker.

Tracks per-pair trading performance over rolling windows, ranks pairs
by composite score (win rate, profit factor, Sharpe), and recommends
which pairs to actively trade vs retire.

Used by the learning server to expose pair rankings via API and
eventually for weekly auto-rotation of the trading server's symbol list.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import load_config
from shared.logger import get_logger

logger = get_logger("learning_server.pair_ranker")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PairMetrics:
    """Performance metrics for a single trading pair over a rolling window."""

    symbol: str
    total_trades: int
    wins: int
    losses: int
    total_pnl: float
    win_rate: float
    avg_pnl: float
    profit_factor: float
    sharpe: float
    score: float  # composite 0-1 ranking score
    trend: str = "neutral"  # up / down / neutral (based on recent PnL trajectory)


# ---------------------------------------------------------------------------
# Pair Ranker
# ---------------------------------------------------------------------------


class PairRanker:
    """Analyze trade history and rank pairs by performance.

    Reads from the integration server's ``trade_results`` table to compute
    per-pair win rate, profit factor, Sharpe ratio, and a composite score.
    """

    def __init__(self) -> None:
        self._cfg = load_config()

        # Determine the integration DB path from config
        int_cfg = self._cfg.data.get("integration", {})
        db_path = int_cfg.get("database", {}).get("path", "data/integration.db")
        # Resolve relative to project root (assumes CWD is project root)
        self._db_path = Path(db_path)
        if not self._db_path.is_absolute():
            self._db_path = Path.cwd() / self._db_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_pair_rankings(
        self,
        window_days: int = 7,
        min_trades: int = 2,
    ) -> List[PairMetrics]:
        """Return all pairs ranked by composite performance score.

        Args:
            window_days: Look-back window in days (default 7 = weekly).
            min_trades: Minimum closed trades required to be ranked.

        Returns:
            List of ``PairMetrics`` sorted by score descending.
        """
        import aiosqlite

        lookback = datetime.now(timezone.utc) - timedelta(days=window_days)
        lookback_str = lookback.isoformat()

        if not self._db_path.exists():
            logger.warning(f"Integration DB not found at {self._db_path}")
            return []

        try:
            async with aiosqlite.connect(str(self._db_path)) as db:
                db.row_factory = aiosqlite.Row
                rows = await db.execute_fetchall(
                    """SELECT symbol, pnl, closed_at, side, strategy
                       FROM trade_results
                       WHERE pnl IS NOT NULL
                         AND closed_at >= ?
                       ORDER BY closed_at ASC""",
                    (lookback_str,),
                )
        except Exception as exc:
            logger.error(f"Failed to query trade_results: {exc}")
            return []

        if not rows:
            return []

        # Group by symbol
        by_symbol: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "pnls": [],
                "pnls_recent": [],
            }
        )

        for row in rows:
            sym: str = row["symbol"]
            pnl_val: float = row["pnl"] or 0.0
            is_win = pnl_val > 0

            by_symbol[sym]["trades"] += 1
            by_symbol[sym]["total_pnl"] += pnl_val
            by_symbol[sym]["pnls"].append(pnl_val)
            if is_win:
                by_symbol[sym]["wins"] += 1
            else:
                by_symbol[sym]["losses"] += 1

        # Compute metrics per pair and rank
        results: List[PairMetrics] = []
        for sym, data in by_symbol.items():
            if data["trades"] < min_trades:
                continue

            n = data["trades"]
            wins = data["wins"]
            losses = data["losses"]
            total_pnl = data["total_pnl"]
            pnls = data["pnls"]

            wr = wins / n if n > 0 else 0.0
            avg_pnl = total_pnl / n if n > 0 else 0.0

            # Profit factor (avoid div by zero)
            gains = sum(p for p in pnls if p > 0)
            loss_sum = abs(sum(p for p in pnls if p < 0))
            pf = gains / loss_sum if loss_sum > 0 else (gains / 0.001)

            # Sharpe-like ratio (daily-ized from trade returns)
            if len(pnls) >= 2:
                mean_p = sum(pnls) / len(pnls)
                variance = sum((p - mean_p) ** 2 for p in pnls) / len(pnls)
                std_p = math.sqrt(variance) if variance > 0 else 0.001
                sharpe = (mean_p / std_p) * math.sqrt(365)
            else:
                sharpe = 0.0

            # Composite score (weighted, clipped 0-1)
            score = (
                min(wr, 1.0) * 0.30
                + min(max(avg_pnl / 5.0, -0.5), 1.0) * 0.25
                + min(pf / 3.0, 1.0) * 0.25
                + min(max(sharpe / 2.0, -0.5), 1.0) * 0.20
            )
            score = max(0.0, min(round(score, 4), 1.0))

            # Trend direction (last 3 trades)
            trend = "neutral"
            if len(pnls) >= 3:
                recent = pnls[-3:]
                recent_avg = sum(recent) / 3
                if recent_avg > avg_pnl * 0.5:
                    trend = "up"
                elif recent_avg < avg_pnl * 0.5:
                    trend = "down"

            results.append(
                PairMetrics(
                    symbol=sym,
                    total_trades=n,
                    wins=wins,
                    losses=losses,
                    total_pnl=round(total_pnl, 2),
                    win_rate=round(wr, 4),
                    avg_pnl=round(avg_pnl, 4),
                    profit_factor=round(pf, 2),
                    sharpe=round(sharpe, 2),
                    score=round(score, 4),
                    trend=trend,
                )
            )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def get_recommended_pairs(
        self,
        window_days: int = 7,
        max_pairs: int = 6,
        min_trades: int = 2,
    ) -> List[str]:
        """Get the top N recommended pairs to actively trade."""
        rankings = await self.get_pair_rankings(
            window_days=window_days, min_trades=min_trades
        )
        return [r.symbol for r in rankings[:max_pairs]]

    async def get_retired_pairs(
        self,
        window_days: int = 14,
        min_trades: int = 3,
    ) -> List[str]:
        """Identify pairs that should be retired (worst performers)."""
        rankings = await self.get_pair_rankings(
            window_days=window_days, min_trades=min_trades
        )
        # Retire pairs with win rate < 40% or net negative PnL over threshold
        retired: List[str] = []
        for r in reversed(rankings):
            if r.win_rate < 0.40 or r.total_pnl < -10.0:
                if r.symbol not in retired:
                    retired.append(r.symbol)
            if len(retired) >= 3:
                break
        return retired
