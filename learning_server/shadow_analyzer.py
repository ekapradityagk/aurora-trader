"""Aurora Trader — Shadow Analyzer.

Post-trade behavior analysis inspired by Vibe-Trading's "Shadow Account"
feature. Profiles our trading patterns to answer:

  1. Can we open faster?  (entry timing analysis)
  2. Can we close sooner? (exit timing analysis)
  3. What biases are hurting us?  (behavioral signals)
  4. Which strategies/patterns work best?  (win rate by entry type)

Reads from the integration server's ``trade_results`` table.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.config import load_config
from shared.logger import get_logger

logger = get_logger("learning_server.shadow_analyzer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bias detection thresholds
DISPOSITION_EFFECT_WIN_HOLD_THRESHOLD_HOURS = 4   # Selling winners < 4h
DISPOSITION_EFFECT_LOSS_HOLD_THRESHOLD_HOURS = 48  # Holding losers > 48h
OVERTRADING_MAX_TRADES_PER_DAY = 10                # > 10 trades/day = overtrading
CHASE_THRESHOLD_PCT = 3.0                          # Entry after >3% move in same direction


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """A single closed trade with extended fields."""
    trade_id: str
    strategy: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    rrr: float
    closed_at: Optional[str]
    opened_at: Optional[str] = None
    entry_reason: str = ""
    exit_reason: str = ""
    holding_hours: float = 0.0


@dataclass
class StrategyProfile:
    """Performance profile for a single strategy."""
    strategy: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    profit_factor: float
    avg_rrr: float
    sharpe: float
    max_drawdown: float
    avg_holding_hours: float
    win_avg_holding_hours: float
    loss_avg_holding_hours: float
    entry_reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class BiasReport:
    """Behavioral bias indicators detected in trading data."""
    disposition_effect: Dict[str, Any] = field(default_factory=dict)
    overtrading: Dict[str, Any] = field(default_factory=dict)
    chase_entries: Dict[str, Any] = field(default_factory=dict)
    anchoring: Dict[str, Any] = field(default_factory=dict)
    overall_health: str = "unknown"


@dataclass
class ShadowReport:
    """Complete shadow analysis output."""
    timestamp: str = ""
    total_trades_analyzed: int = 0
    time_period_days: int = 0
    strategy_profiles: Dict[str, StrategyProfile] = field(default_factory=dict)
    bias_report: BiasReport = field(default_factory=BiasReport)
    entry_timing_analysis: Dict[str, Any] = field(default_factory=dict)
    exit_timing_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    top_performers: List[str] = field(default_factory=list)
    worst_performers: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shadow Analyzer
# ---------------------------------------------------------------------------


class ShadowAnalyzer:
    """Analyze closed trades to profile behavior and detect biases.

    Reads from the integration DB's ``trade_results`` table and produces
    a comprehensive ShadowReport with per-strategy profiles, bias
    detection, timing analysis, and actionable recommendations.
    """

    def __init__(self) -> None:
        self._cfg = load_config()
        self._log = logger

        # Determine DB path — point to trading.db (single source of truth)
        # where all closed trades now live (migrated from winrate.db)
        ts_cfg = self._cfg.data.get("trading_server", {})
        db_path = ts_cfg.get("database", {}).get("path", "data/trading.db")
        self._db_path = Path(db_path)
        if not self._db_path.is_absolute():
            self._db_path = Path.cwd() / self._db_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        window_days: int = 30,
        min_trades: int = 3,
    ) -> ShadowReport:
        """Run a full shadow analysis over recent trades.

        Args:
            window_days: Look-back window in days.
            min_trades: Minimum trades per strategy to include in profiles.

        Returns:
            ShadowReport with profiles, biases, and recommendations.
        """
        trades = await self._fetch_trades(window_days)
        if not trades:
            return ShadowReport(
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_trades_analyzed=0,
                time_period_days=window_days,
                recommendations=["No trades found in the analysis window."],
            )

        # Group by strategy
        by_strategy: Dict[str, List[TradeRecord]] = defaultdict(list)
        for t in trades:
            by_strategy[t.strategy].append(t)

        # Build strategy profiles
        profiles: Dict[str, StrategyProfile] = {}
        for strategy_name, strategy_trades in by_strategy.items():
            if len(strategy_trades) < min_trades:
                continue
            profiles[strategy_name] = self._profile_strategy(strategy_trades)

        # Bias detection
        bias_report = self._detect_biases(trades)

        # Timing analysis
        entry_timing = self._analyze_entry_timing(trades)
        exit_timing = self._analyze_exit_timing(trades)

        # Find top/worst performers
        performers = self._rank_performers(profiles)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            profiles, bias_report, entry_timing, exit_timing
        )

        return ShadowReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_trades_analyzed=len(trades),
            time_period_days=window_days,
            strategy_profiles=profiles,
            bias_report=bias_report,
            entry_timing_analysis=entry_timing,
            exit_timing_analysis=exit_timing,
            recommendations=recommendations,
            top_performers=performers["top"],
            worst_performers=performers["worst"],
        )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def _fetch_trades(self, window_days: int) -> List[TradeRecord]:
        """Fetch closed trades from the trading.db closed_trades table."""
        if not self._db_path.exists():
            self._log.warning(f"Trading DB not found at {self._db_path}")
            return []

        import aiosqlite

        lookback = datetime.now(timezone.utc) - timedelta(days=window_days)
        lookback_str = lookback.isoformat()

        try:
            async with aiosqlite.connect(str(self._db_path)) as db:
                db.row_factory = aiosqlite.Row
                rows = await db.execute_fetchall(
                    """SELECT id as trade_id, strategy_name as strategy, symbol, side,
                              entry_price, exit_price, pnl, 0.0 as rrr,
                              closed_at, entry_time as opened_at,
                              reason as exit_reason, reason as entry_reason
                       FROM closed_trades
                       WHERE pnl IS NOT NULL
                         AND closed_at >= ?
                       ORDER BY closed_at DESC""",
                    (lookback_str,),
                )
        except Exception as exc:
            self._log.error(f"Failed to query trades: {exc}")
            return []

        trades: List[TradeRecord] = []
        for row in rows:
            opened = row["opened_at"]
            closed = row["closed_at"]

            holding_hours = 0.0
            if opened and closed:
                try:
                    opened_dt = datetime.fromisoformat(opened)
                    closed_dt = datetime.fromisoformat(closed)
                    holding_hours = (closed_dt - opened_dt).total_seconds() / 3600
                except (ValueError, TypeError):
                    pass

            trades.append(TradeRecord(
                trade_id=row["trade_id"],
                strategy=row["strategy"],
                symbol=row["symbol"],
                side=row["side"],
                entry_price=row["entry_price"] or 0.0,
                exit_price=row["exit_price"] or 0.0,
                pnl=row["pnl"] or 0.0,
                rrr=row["rrr"] or 0.0,
                closed_at=closed,
                opened_at=opened,
                entry_reason=row["entry_reason"] or "",
                exit_reason=row["exit_reason"] or "",
                holding_hours=holding_hours,
            ))

        return trades

    # ------------------------------------------------------------------
    # Strategy profiling
    # ------------------------------------------------------------------

    def _profile_strategy(self, trades: List[TradeRecord]) -> StrategyProfile:
        """Compute detailed metrics for a single strategy."""
        n = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        n_wins = len(wins)
        n_losses = len(losses)

        wr = n_wins / n if n > 0 else 0.0
        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / n if n > 0 else 0.0

        # Profit factor
        gains = sum(t.pnl for t in wins)
        loss_sum = abs(sum(t.pnl for t in losses))
        pf = gains / loss_sum if loss_sum > 0 else (gains / 0.001 if gains > 0 else 0.0)

        # Avg RRR
        avg_rrr = sum(t.rrr for t in trades) / n if n > 0 else 0.0

        # Sharpe (trade-based)
        pnls = [t.pnl for t in trades]
        sharpe = self._compute_sharpe(pnls)

        # Max drawdown (from cumulative PnL)
        max_dd = self._compute_max_drawdown(pnls)

        # Holding time analysis
        holding_times = [t.holding_hours for t in trades if t.holding_hours > 0]
        avg_holding = sum(holding_times) / len(holding_times) if holding_times else 0.0
        win_holding = [t.holding_hours for t in wins if t.holding_hours > 0]
        loss_holding = [t.holding_hours for t in losses if t.holding_hours > 0]
        avg_win_hold = sum(win_holding) / len(win_holding) if win_holding else 0.0
        avg_loss_hold = sum(loss_holding) / len(loss_holding) if loss_holding else 0.0

        # Entry reasons distribution
        reasons: Dict[str, int] = defaultdict(int)
        for t in trades:
            if t.entry_reason:
                reasons[t.entry_reason] += 1

        return StrategyProfile(
            strategy=trades[0].strategy if trades else "unknown",
            total_trades=n,
            wins=n_wins,
            losses=n_losses,
            win_rate=round(wr, 4),
            total_pnl=round(total_pnl, 2),
            avg_pnl=round(avg_pnl, 4),
            profit_factor=round(pf, 2),
            avg_rrr=round(avg_rrr, 2),
            sharpe=round(sharpe, 2),
            max_drawdown=round(max_dd, 2),
            avg_holding_hours=round(avg_holding, 2),
            win_avg_holding_hours=round(avg_win_hold, 2),
            loss_avg_holding_hours=round(avg_loss_hold, 2),
            entry_reasons=dict(reasons),
        )

    # ------------------------------------------------------------------
    # Bias detection
    # ------------------------------------------------------------------

    def _detect_biases(self, trades: List[TradeRecord]) -> BiasReport:
        """Detect behavioral biases in trading patterns."""
        report = BiasReport()

        # 1. Disposition effect — selling winners too fast, holding losers too long
        wins = [t for t in trades if t.pnl > 0 and t.holding_hours > 0]
        losses = [t for t in trades if t.pnl <= 0 and t.holding_hours > 0]

        early_wins = sum(
            1 for t in wins if t.holding_hours < DISPOSITION_EFFECT_WIN_HOLD_THRESHOLD_HOURS
        )
        long_losses = sum(
            1 for t in losses if t.holding_hours > DISPOSITION_EFFECT_LOSS_HOLD_THRESHOLD_HOURS
        )

        disposition_score = 0.0
        if wins:
            disposition_score = early_wins / len(wins)
        if losses and not wins:
            disposition_score = long_losses / len(losses)

        has_disposition = disposition_score > 0.3

        report.disposition_effect = {
            "detected": has_disposition,
            "score": round(disposition_score, 3),
            "wins_closed_early": early_wins,
            "total_wins": len(wins),
            "losses_held_too_long": long_losses,
            "total_losses": len(losses),
            "detail": (
                f"Sold {early_wins}/{len(wins)} winning trades within "
                f"{DISPOSITION_EFFECT_WIN_HOLD_THRESHOLD_HOURS}h, "
                f"held {long_losses}/{len(losses)} losing trades beyond "
                f"{DISPOSITION_EFFECT_LOSS_HOLD_THRESHOLD_HOURS}h"
            ) if has_disposition else "No significant disposition effect detected",
        }

        # 2. Overtrading — too many trades in a short period
        trades_by_day: Dict[str, int] = defaultdict(int)
        for t in trades:
            if t.closed_at:
                day = t.closed_at[:10]  # YYYY-MM-DD
                trades_by_day[day] += 1

        high_volume_days = sum(
            1 for count in trades_by_day.values()
            if count > OVERTRADING_MAX_TRADES_PER_DAY
        )
        max_trades_in_day = max(trades_by_day.values()) if trades_by_day else 0

        has_overtrading = high_volume_days > 0
        report.overtrading = {
            "detected": has_overtrading,
            "high_volume_days": high_volume_days,
            "max_trades_in_day": max_trades_in_day,
            "threshold": OVERTRADING_MAX_TRADES_PER_DAY,
            "detail": (
                f"Found {high_volume_days} day(s) with > "
                f"{OVERTRADING_MAX_TRADES_PER_DAY} trades "
                f"(max: {max_trades_in_day}/day)"
            ) if has_overtrading else "No overtrading detected",
        }

        # 3. Chase detection — entering after large moves
        # Without price context per trade, we check if entries cluster
        # around big PnL swings (indirect signal)
        chase_pnls = [t.pnl for t in trades if abs(t.pnl or 0) > 0]
        revenge_trades = 0
        if chase_pnls:
            sorted_pnls = sorted(chase_pnls, reverse=True)
            top_10_pct = sorted_pnls[:max(1, len(sorted_pnls) // 10)]
            # Check if big wins cluster after big losses = revenge trading
            # This is a simplified heuristic
            revenge_trades = 0
            for i in range(1, len(trades)):
                if abs(trades[i - 1].pnl or 0) > 5 and trades[i].pnl > 0:
                    revenge_trades += 1

            has_chase = revenge_trades > 3
            report.chase_entries = {
                "detected": has_chase,
                "possible_revenge_trades": revenge_trades,
                "detail": (
                    f"Found {revenge_trades} possible revenge/chase trades "
                    f"(winning trade right after a big loss)"
                ) if has_chase else "No significant chase behavior detected",
            }

        # 4. Anchoring — fixating on specific price levels
        # Detect if exits cluster at round numbers
        round_exits = 0
        for t in trades:
            if t.exit_price and t.exit_price > 0:
                price_mod = t.exit_price % 1
                if price_mod < 0.01 or price_mod > 0.99:
                    round_exits += 1

        has_anchoring = round_exits > len(trades) * 0.3 if trades else False
        report.anchoring = {
            "detected": has_anchoring,
            "round_number_exits": round_exits,
            "total_closed_trades": len(trades),
            "detail": (
                f"{round_exits}/{len(trades)} trades exited at or very near "
                f"round numbers — possible anchoring bias"
            ) if has_anchoring else "No anchoring bias detected",
        }

        # Overall health assessment
        issues = sum([
            has_disposition,
            has_overtrading,
            has_chase if "detected" in report.chase_entries and report.chase_entries["detected"] else False,
            has_anchoring,
        ])
        if issues == 0:
            report.overall_health = "good"
        elif issues <= 2:
            report.overall_health = "fair"
        else:
            report.overall_health = "needs_attention"

        return report

    # ------------------------------------------------------------------
    # Timing analysis
    # ------------------------------------------------------------------

    def _analyze_entry_timing(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """Analyze entry timing — could we enter faster?"""
        if not trades:
            return {"message": "No trades to analyze"}

        # Check if we have opened_at data for entry speed analysis
        entries_with_timestamps = [t for t in trades if t.opened_at]
        if not entries_with_timestamps:
            return {
                "message": "No entry timestamps recorded yet. Enable opened_at logging.",
                "recommendation": "Record opened_at on trade open to enable entry timing analysis.",
            }

        # Entry speed = time between signal generation and actual fill
        # (We'd need signal timestamps for this — future enhancement)
        return {
            "message": f"{len(entries_with_timestamps)} trades with entry timestamps",
            "entries_recorded": len(entries_with_timestamps),
            "recommendation": "Enable signal-to-fill latency tracking for entry speed optimization.",
        }

    def _analyze_exit_timing(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """Analyze exit timing — could we close sooner/later for better results?"""
        if not trades:
            return {"message": "No trades to analyze"}

        wins = [t for t in trades if t.pnl > 0 and t.holding_hours > 0]
        losses = [t for t in trades if t.pnl <= 0 and t.holding_hours > 0]

        result: Dict[str, Any] = {}

        if wins:
            avg_win_hold = sum(t.holding_hours for t in wins) / len(wins)
            result["avg_winner_hold_hours"] = round(avg_win_hold, 2)
            # Quick winners (< 1h)
            quick_wins = sum(1 for t in wins if t.holding_hours < 1)
            result["quick_winners_under_1h"] = quick_wins
            result["quick_winner_pct"] = round(quick_wins / len(wins) * 100, 1)

        if losses:
            avg_loss_hold = sum(t.holding_hours for t in losses) / len(losses)
            result["avg_loser_hold_hours"] = round(avg_loss_hold, 2)
            # Losses that could have been cut sooner
            slow_losses = sum(1 for t in losses if t.holding_hours > 12)
            result["slow_losses_over_12h"] = slow_losses
            result["slow_loss_pct"] = round(slow_losses / len(losses) * 100, 1)

        # Comparison — are we holding winners shorter than losers? (disposition)
        if wins and losses:
            avg_win = sum(t.holding_hours for t in wins) / len(wins)
            avg_loss = sum(t.holding_hours for t in losses) / len(losses)
            result["holding_ratio_win_vs_loss"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0
            result["holding_verdict"] = (
                "You hold winners LONGER than losers ✅"
                if avg_win > avg_loss
                else "⚠️ You cut winners SHORTER than losers — classic disposition effect!"
            )

        if not result:
            result["message"] = "No trades with holding time data."

        # Recommendations based on exit timing
        recs = []
        if result.get("holding_ratio_win_vs_loss", 1) < 1:
            recs.append("Let winners run longer — your winners are held shorter than losers.")
        if result.get("slow_loss_pct", 0) > 30:
            recs.append(f"Cut losses faster — {result['slow_loss_pct']}% of losers held >12h.")
        if recs:
            result["recommendations"] = recs

        return result

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank_performers(
        self, profiles: Dict[str, StrategyProfile]
    ) -> Dict[str, List[str]]:
        """Rank strategies by composite performance."""
        if not profiles:
            return {"top": [], "worst": []}

        scored = []
        for name, p in profiles.items():
            # Composite: win rate * 0.3 + profit factor * 0.3 + sharpe * 0.2 + avg_rrr * 0.2
            composite = (
                p.win_rate * 0.3
                + min(p.profit_factor / 3, 1.0) * 0.3
                + min(max(p.sharpe / 2, 0), 1.0) * 0.2
                + min(p.avg_rrr / 3, 1.0) * 0.2
            )
            scored.append((name, composite))

        scored.sort(key=lambda x: x[1], reverse=True)

        return {
            "top": [s[0] for s in scored[:3]],
            "worst": [s[0] for s in scored[-3:]] if len(scored) >= 3 else [],
        }

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        profiles: Dict[str, StrategyProfile],
        bias_report: BiasReport,
        entry_timing: Dict[str, Any],
        exit_timing: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations from analysis."""
        recs: List[str] = []

        # Bias-based recommendations
        if bias_report.disposition_effect.get("detected"):
            recs.append(
                f"🔴 Disposition effect detected! You're cutting winners too fast "
                f"and holding losers too long. Consider trailing stops on winners "
                f"and hard stop-losses on every entry."
            )

        if bias_report.overtrading.get("detected"):
            recs.append(
                f"🔴 Overtrading detected — {bias_report.overtrading['high_volume_days']} "
                f"days with excessive trades. Quality over quantity!"
            )

        if bias_report.chase_entries.get("detected"):
            recs.append(
                f"🟡 Possible chase/ revenge trading — {bias_report.chase_entries.get('possible_revenge_trades', 0)} "
                f"trades entered right after a loss. Take a breather after a bad trade."
            )

        if bias_report.overall_health == "needs_attention":
            recs.append("Overall trading health needs attention. Review risk management rules.")

        # Performance-based recommendations
        for name, p in sorted(
            profiles.items(), key=lambda x: x[1].win_rate
        ):
            if p.total_trades >= 5 and p.win_rate < 0.4:
                recs.append(
                    f"🟡 {name} has {p.win_rate * 100:.0f}% win rate over {p.total_trades} trades — "
                    f"consider pausing or adjusting parameters."
                )
            elif p.total_trades >= 5 and p.win_rate > 0.65:
                recs.append(
                    f"✅ {name} is performing well ({p.win_rate * 100:.0f}% WR, "
                    f"PF: {p.profit_factor}) — keep it up!"
                )

        # Timing recommendations
        if exit_timing.get("recommendations"):
            recs.extend(exit_timing["recommendations"])

        if entry_timing.get("recommendation"):
            recs.append(f"ℹ️ {entry_timing['recommendation']}")

        if not recs:
            recs.append(
                "✅ No major issues detected. Keep executing your plan!"
            )

        return recs[:10]  # Cap at 10 recommendations

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _compute_sharpe(self, pnls: List[float], risk_free: float = 0.0) -> float:
        """Compute Sharpe ratio from trade PnL list."""
        if len(pnls) < 2:
            return 0.0
        mean_p = sum(pnls) / len(pnls)
        variance = sum((p - mean_p) ** 2 for p in pnls) / len(pnls)
        std_p = math.sqrt(variance) if variance > 0 else 0.001
        return (mean_p - risk_free) / std_p * math.sqrt(365)  # annualised

    def _compute_max_drawdown(self, pnls: List[float]) -> float:
        """Compute maximum drawdown from cumulative PnL series."""
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cum += p
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd
        return max_dd
