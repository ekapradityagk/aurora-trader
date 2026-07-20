"""
Aurora Trader — Trade History Analyzer.

Reads all closed trades from the SQLite trade journal and computes
per-strategy performance metrics:
    - Win rate, profit factor, Sharpe ratio, avg R:R, max drawdown
    - Rolling 20-trade win rate per strategy
    - Identifies underperforming strategies (< 50% win rate over 20 trades)
    - Generates improvement recommendations
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite

from shared.config import load_config
from shared.logger import get_logger

logger = get_logger("learning_server.analyzer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLING_WINDOW = 20
MIN_TRADES_FOR_ANALYSIS = 5
UNDERPERFORMING_WINRATE = 0.50  # < 50% over window


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StrategyMetrics:
    """Aggregate performance metrics for a single strategy."""

    strategy_name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_risk_reward: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    rolling_win_rate: List[float] = field(default_factory=list)
    current_rolling_win_rate: float = 0.0
    is_underperforming: bool = False
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Complete analysis report for all strategies."""

    timestamp: str = ""
    total_trades: int = 0
    strategy_count: int = 0
    strategies: Dict[str, StrategyMetrics] = field(default_factory=dict)
    global_win_rate: float = 0.0
    global_profit_factor: float = 0.0
    global_sharpe: float = 0.0
    underperforming_strategies: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trade Analyzer
# ---------------------------------------------------------------------------


class TradeAnalyzer:
    """Trade history analyzer that computes per-strategy performance metrics.

    Usage::

        analyzer = TradeAnalyzer()
        report = await analyzer.analyze()
    """

    def __init__(self, db_path: str = "data/trading.db") -> None:
        self._db_path = db_path
        self._log = logger
        self._cfg = load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self) -> AnalysisReport:
        """Run full analysis of all closed trades.

        Returns:
            An AnalysisReport containing per-strategy metrics and global stats.
        """
        trades = await self._load_trades()
        if not trades:
            self._log.warning("No closed trades found for analysis")
            return AnalysisReport(timestamp=datetime.now(timezone.utc).isoformat())

        self._log.info(f"Analyzing {len(trades)} closed trades")

        # Group by strategy
        by_strategy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for t in trades:
            by_strategy[t.get("strategy_name", "unknown")].append(t)

        report = AnalysisReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_trades=len(trades),
            strategy_count=len(by_strategy),
        )

        global_pnls: List[float] = []
        global_wins = 0
        global_losses = 0

        for strategy_name, strategy_trades in by_strategy.items():
            metrics = self._compute_strategy_metrics(strategy_name, strategy_trades)
            report.strategies[strategy_name] = metrics
            global_pnls.extend(
                float(t.get("pnl", 0) or 0) for t in strategy_trades
            )
            for t in strategy_trades:
                pnl = float(t.get("pnl", 0) or 0)
                if pnl > 0:
                    global_wins += 1
                elif pnl < 0:
                    global_losses += 1

        # Global metrics
        total_win_rate = 0.0
        total_profit_factor = 0.0
        for m in report.strategies.values():
            total_win_rate += m.win_rate
            total_profit_factor += m.profit_factor

        if report.strategy_count > 0:
            report.global_win_rate = total_win_rate / report.strategy_count
            report.global_profit_factor = total_profit_factor / report.strategy_count

        if len(global_pnls) >= 2:
            report.global_sharpe = self._compute_sharpe(global_pnls)

        # Collect underperforming strategies
        report.underperforming_strategies = [
            s for s, m in report.strategies.items() if m.is_underperforming
        ]

        self._log.info(
            f"Analysis complete: {len(report.strategies)} strategies, "
            f"{len(report.underperforming_strategies)} underperforming"
        )

        return report

    async def get_strategy_metrics(
        self, strategy_name: str
    ) -> Optional[StrategyMetrics]:
        """Get metrics for a single strategy."""
        trades = await self._load_trades()
        strategy_trades = [
            t for t in trades if t.get("strategy_name") == strategy_name
        ]
        if not strategy_trades:
            return None
        return self._compute_strategy_metrics(strategy_name, strategy_trades)

    # ------------------------------------------------------------------
    # Trade loading
    # ------------------------------------------------------------------

    async def _load_trades(self) -> List[Dict[str, Any]]:
        """Load all closed trades from the single trading.db source of truth.

        Reads from the closed_trades table which now contains all trade data
        (migrated from trades.db and winrate.db).
        """
        trades: List[Dict[str, Any]] = []
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT id, symbol, side, entry_price, exit_price, pnl,
                           closed_at AS exit_time, strategy_name, leverage,
                           entry_time, reason
                    FROM closed_trades
                    WHERE pnl IS NOT NULL
                    ORDER BY closed_at ASC
                    """
                )
                rows = await cursor.fetchall()
                for row in rows:
                    d = dict(row)
                    d["entry_price"] = float(d.get("entry_price", 0) or 0)
                    d["exit_price"] = float(d.get("exit_price", 0) or 0)
                    d["pnl"] = float(d.get("pnl", 0) or 0)
                    d["pnl_pct"] = 0.0
                    trades.append(d)
        except Exception as exc:
            self._log.warning(f"Could not load trades from {self._db_path}: {exc}")

        self._log.info(f"Loaded {len(trades)} closed trades from single DB")
        return trades

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_strategy_metrics(
        self,
        strategy_name: str,
        trades: List[Dict[str, Any]],
    ) -> StrategyMetrics:
        """Compute all metrics for a single strategy from its trade list."""
        metrics = StrategyMetrics(strategy_name=strategy_name)
        metrics.total_trades = len(trades)

        if not trades:
            return metrics

        pnls: List[float] = []
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        r_values: List[float] = []  # risk-to-reward ratios

        for t in trades:
            pnl = float(t.get("pnl", 0) or 0)
            pnl_pct = float(t.get("pnl_pct", 0) or 0)
            pnls.append(pnl)

            if pnl > 0:
                wins += 1
                gross_profit += pnl
            elif pnl < 0:
                losses += 1
                gross_loss += abs(pnl)

            # Compute risk-to-reward ratio
            # R:R = |PnL| / (entry_price * stop_loss_pct)
            # Fallback: use PnL% as a proxy if R:R < 0 (unrealistic)
            entry = float(t.get("entry_price", 0) or 0)
            if entry > 0:
                # Use 1% of entry as assumed risk if no stop loss info
                assumed_risk = entry * 0.01
                rr = abs(pnl) / assumed_risk if assumed_risk > 0 else abs(pnl_pct)
            else:
                rr = abs(pnl_pct)
            if rr > 0:
                r_values.append(rr)

        metrics.wins = wins
        metrics.losses = losses
        metrics.total_pnl = round(sum(pnls), 4)

        # Win rate
        if metrics.total_trades > 0:
            metrics.win_rate = round(wins / metrics.total_trades, 4)

        # Profit factor
        if gross_loss > 0:
            metrics.profit_factor = round(gross_profit / gross_loss, 4)
        elif gross_profit > 0:
            metrics.profit_factor = float("inf")
        else:
            metrics.profit_factor = 0.0

        # Sharpe ratio
        if len(pnls) >= 2:
            metrics.sharpe_ratio = round(self._compute_sharpe(pnls), 4)

        # Average R:R
        if r_values:
            metrics.avg_risk_reward = round(sum(r_values) / len(r_values), 4)

        # Max drawdown (peak-to-trough in cumulative PnL)
        metrics.max_drawdown = round(self._compute_max_drawdown(pnls), 4)

        # Rolling 20-trade win rate
        metrics.rolling_win_rate = self._compute_rolling_win_rate(pnls)
        if metrics.rolling_win_rate:
            metrics.current_rolling_win_rate = round(metrics.rolling_win_rate[-1], 4)

        # Underperformance check
        if (
            metrics.total_trades >= ROLLING_WINDOW
            and metrics.current_rolling_win_rate < UNDERPERFORMING_WINRATE
        ):
            metrics.is_underperforming = True

        # Generate recommendations
        metrics.recommendations = self._generate_recommendations(metrics)

        return metrics

    # ------------------------------------------------------------------
    # Helper computations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sharpe(pnls: List[float], risk_free: float = 0.0) -> float:
        """Compute annualised Sharpe ratio from PnL list.

        Assumes daily trade frequency with ~252 trading days per year.
        """
        if len(pnls) < 2:
            return 0.0
        mean_ret = sum(pnls) / len(pnls)
        variance = sum((r - mean_ret) ** 2 for r in pnls) / (len(pnls) - 1)
        if variance <= 0:
            return 0.0
        std_dev = math.sqrt(variance)
        ann_factor = math.sqrt(252)
        return ((mean_ret - risk_free) / std_dev) * ann_factor

    @staticmethod
    def _compute_max_drawdown(pnls: List[float]) -> float:
        """Compute maximum drawdown from a list of PnL values.

        Uses cumulative returns: tracks the peak and computes the trough
        as a percentage drop from the peak.
        """
        if not pnls:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = (peak - cumulative) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        return max_dd

    @staticmethod
    def _compute_rolling_win_rate(pnls: List[float], window: int = ROLLING_WINDOW) -> List[float]:
        """Compute rolling win rate over the specified window size.

        A win is defined as PnL > 0.
        Returns a list of win rates for each window ending at index i.
        """
        if len(pnls) < window:
            return []

        rates: List[float] = []
        for i in range(window - 1, len(pnls)):
            window_pnls = pnls[i - window + 1: i + 1]
            wins = sum(1 for p in window_pnls if p > 0)
            rates.append(wins / window)
        return rates

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_recommendations(metrics: StrategyMetrics) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recs: List[str] = []

        if metrics.total_trades < MIN_TRADES_FOR_ANALYSIS:
            recs.append("Insufficient trade data for recommendations.")
            return recs

        if metrics.win_rate < 0.40:
            recs.append(
                "Low win rate (< 40%). Consider tightening entry filters, "
                "increasing confirmation requirements, or adjusting "
                "the strategy's risk parameters."
            )
        elif metrics.win_rate < 0.50:
            recs.append(
                "Below-average win rate. Review recent trades for "
                "common failure patterns. Consider lowering take-profit "
                "targets to capture smaller but more frequent gains."
            )

        if metrics.profit_factor < 1.0 and metrics.total_trades > 10:
            recs.append(
                "Profit factor below 1.0 — the strategy is losing money "
                "overall. Consider disabling and reviewing the parameter set."
            )
        elif metrics.profit_factor < 1.5:
            recs.append(
                "Profit factor is marginal (1.0–1.5). Small improvements "
                "in win rate or risk-to-reward could make a significant "
                "difference."
            )

        if metrics.avg_risk_reward < 1.0 and metrics.win_rate > 0.5:
            recs.append(
                "Average R:R is below 1.0 despite a decent win rate. "
                "Try increasing take-profit targets to improve "
                "risk-to-reward profile."
            )
        elif metrics.avg_risk_reward > 3.0 and metrics.win_rate < 0.3:
            recs.append(
                "High R:R but low win rate suggests the strategy is "
                "hunting for home runs. Consider adding more conservative "
                "entries to balance."
            )

        if metrics.max_drawdown > 0.20:
            recs.append(
                f"Maximum drawdown is {metrics.max_drawdown:.1%}. "
                "Consider adding a tighter stop-loss or reducing "
                "position size."
            )

        if not recs:
            recs.append("Strategy performance is acceptable. No immediate changes needed.")

        return recs
