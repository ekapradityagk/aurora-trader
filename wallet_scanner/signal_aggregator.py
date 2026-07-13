"""
Aurora Trader — Wallet Scanner: Signal Aggregator.

Combines signals from exchange flow, whale tracker, and funding rate
monitors into a single weighted bias score.

Aggregation Logic:
    1. Each sub-module produces its own signals with direction + confidence.
    2. All signals are collected and categorised by symbol.
    3. Weighted aggregation produces a -10 to +10 overall score.
    4. A minimum of 2 positive (or 2 negative) signals is required to
       override the default neutral bias.
    5. Results are cached for a configurable TTL to avoid redundant work.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from shared.config import Config, load_config
from shared.logger import get_logger
from shared.models import SignalDirection

logger = get_logger("wallet_scanner.signal_aggregator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default weights for each signal category
_WEIGHTS: Dict[str, float] = {
    "exchange_flow": 0.30,  # 30% — exchange flows are reliable
    "whale": 0.35,  # 35% — whale accumulation is a strong signal
    "funding_rate": 0.20,  # 20% — funding rates are secondary
    "oi_divergence": 0.15,  # 15% — OI adds context
}

# Minimum number of signals required to override neutral
_MIN_SIGNALS_FOR_BIAS = 2

# Cache TTL in seconds
_CACHE_TTL_SEC = 300  # 5 minutes

# Score bounds
_MAX_SCORE = 10.0
_MIN_SCORE = -10.0


# ---------------------------------------------------------------------------
# Aggregated Result
# ---------------------------------------------------------------------------


@dataclass
class AggregatedSignal:
    """Combined signal result for a single symbol."""

    symbol: str
    overall_score: float  # -10 to +10
    bias: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0.0 to 1.0
    signal_count: int  # total raw signals contributing
    positive_count: int
    negative_count: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def is_bullish(self) -> bool:
        return self.bias == "bullish"

    @property
    def is_bearish(self) -> bool:
        return self.bias == "bearish"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "overall_score": round(self.overall_score, 2),
            "bias": self.bias,
            "confidence": round(self.confidence, 4),
            "signal_count": self.signal_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "details": self.details,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Signal Aggregator
# ---------------------------------------------------------------------------


class SignalAggregator:
    """Aggregates signals from all wallet scanner sub-modules.

    Receives raw signals from ExchangeFlowMonitor, WhaleTracker, and
    FundingRateMonitor, then computes a weighted -10 to +10 bias score
    per symbol.  Results are cached with a configurable TTL.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
    ) -> None:
        self._cfg = config or load_config()
        self._log = logger

        # Cache of aggregated results keyed by symbol
        self._cache: Dict[str, Tuple[AggregatedSignal, float]] = {}
        # (signal, expiry_timestamp)

        # Signal weights (configurable)
        self._weights = dict(_WEIGHTS)
        wc = self._cfg.data.get(
            "wallet_scanner", {}
        ).get("signal_weights", {})
        if wc:
            self._weights.update(wc)

        # Min signals for override
        self._min_signals = _MIN_SIGNALS_FOR_BIAS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        exchange_flow_signals: List[Dict[str, Any]],
        whale_signals: List[Dict[str, Any]],
        funding_signals: List[Dict[str, Any]],
    ) -> Dict[str, AggregatedSignal]:
        """Aggregate all signals by symbol and produce scores.

        Returns a dict mapping symbol → AggregatedSignal.
        """
        # 1. Collect all raw signals by symbol
        by_symbol: Dict[str, List[Dict[str, Any]]] = {}

        for sig_list, cat in [
            (exchange_flow_signals, "exchange_flow"),
            (whale_signals, "whale"),
            (funding_signals, "funding_rate"),
        ]:
            for sig in sig_list:
                sym = sig.get("symbol", "UNKNOWN")
                if sym not in by_symbol:
                    by_symbol[sym] = []
                sig["_category"] = cat
                by_symbol[sym].append(sig)

        # 2. Compute score per symbol
        results: Dict[str, AggregatedSignal] = {}
        now = time.time()

        for symbol, signals in by_symbol.items():
            result = self._compute_score(symbol, signals, now)
            results[symbol] = result
            self._cache[symbol] = (result, now + _CACHE_TTL_SEC)

        return results

    def get_cached(
        self, symbol: str
    ) -> Optional[AggregatedSignal]:
        """Return a cached result for *symbol* if still valid."""
        entry = self._cache.get(symbol)
        if entry is None:
            return None
        result, expiry = entry
        if time.time() > expiry:
            del self._cache[symbol]
            return None
        return result

    def get_all_cached(self) -> Dict[str, AggregatedSignal]:
        """Return all non-expired cached results."""
        now = time.time()
        valid: Dict[str, AggregatedSignal] = {}
        expired_keys: List[str] = []
        for sym, (result, expiry) in self._cache.items():
            if now > expiry:
                expired_keys.append(sym)
            else:
                valid[sym] = result
        for k in expired_keys:
            del self._cache[k]
        return valid

    def invalidate_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cached result for *symbol* (or all if *symbol* is None)."""
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update signal category weights."""
        self._weights.update(weights)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        symbol: str,
        signals: List[Dict[str, Any]],
        now: float,
    ) -> AggregatedSignal:
        """Compute the aggregated score for a single symbol from its raw
        signal list."""
        if not signals:
            return AggregatedSignal(
                symbol=symbol,
                overall_score=0.0,
                bias="neutral",
                confidence=0.0,
                signal_count=0,
                positive_count=0,
                negative_count=0,
                details={},
                timestamp=now,
            )

        # Categorise signals by polarity
        positive_signals: List[Dict[str, Any]] = []
        negative_signals: List[Dict[str, Any]] = []

        for sig in signals:
            direction = sig.get("direction", "neutral")
            confidence = sig.get("confidence", 0.5)
            sig["_weighted_confidence"] = confidence

            if direction in (
                SignalDirection.LONG.value,
                "bullish",
                "buy",
            ):
                positive_signals.append(sig)
            elif direction in (
                SignalDirection.SHORT.value,
                "bearish",
                "sell",
            ):
                negative_signals.append(sig)
            # neutral signals are counted but don't affect polarity

        # Counts
        pos_count = len(positive_signals)
        neg_count = len(negative_signals)
        total_count = len(signals)

        # 3. Check minimum signal threshold for override
        bias: str
        if pos_count >= self._min_signals and pos_count > neg_count:
            bias = "bullish"
        elif neg_count >= self._min_signals and neg_count > pos_count:
            bias = "bearish"
        else:
            bias = "neutral"

        # 4. Compute weighted score

        # Score contribution by category
        def _category_score(
            sigs: List[Dict[str, Any]],
            cat: str,
            direction_mult: float,
        ) -> float:
            cat_sigs = [s for s in sigs if s.get("_category") == cat]
            if not cat_sigs:
                return 0.0
            avg_conf = sum(
                s.get("_weighted_confidence", 0.5) for s in cat_sigs
            ) / len(cat_sigs)
            weight = self._weights.get(cat, 0.2)
            return avg_conf * weight * direction_mult * _MAX_SCORE

        bullish_score = (
            _category_score(positive_signals, "exchange_flow", 1.0)
            + _category_score(positive_signals, "whale", 1.0)
            + _category_score(positive_signals, "funding_rate", 1.0)
        )

        bearish_score = (
            _category_score(negative_signals, "exchange_flow", -1.0)
            + _category_score(negative_signals, "whale", -1.0)
            + _category_score(negative_signals, "funding_rate", -1.0)
        )

        raw_score = bullish_score + bearish_score

        # Clamp to [-10, 10]
        overall_score = max(_MIN_SCORE, min(_MAX_SCORE, raw_score))

        # Handle OI divergence signals separately (they add directional
        # nuance to funding rate signals)
        oi_signals = [
            s for s in signals if s.get("type") == "oi_divergence"
        ]
        for oi_sig in oi_signals:
            oi_dir = oi_sig.get("direction", "neutral")
            oi_conf = oi_sig.get("confidence", 0.5)
            oi_weight = self._weights.get("oi_divergence", 0.15)
            oi_adj = oi_conf * oi_weight * _MAX_SCORE
            if oi_dir in (
                SignalDirection.LONG.value,
                "bullish",
                "buy",
            ):
                overall_score = min(_MAX_SCORE, overall_score + oi_adj)
            elif oi_dir in (
                SignalDirection.SHORT.value,
                "bearish",
                "sell",
            ):
                overall_score = max(_MIN_SCORE, overall_score - oi_adj)

        # 5. Confidence = weighted average of all signal confidences
        if total_count > 0:
            total_conf = sum(
                s.get("confidence", 0.5) for s in signals
            )
            avg_conf = total_conf / total_count
        else:
            avg_conf = 0.0

        # 6. Build details
        details: Dict[str, Any] = {
            "signals_used": total_count,
            "categories_contributing": list(
                set(s.get("_category", "unknown") for s in signals)
            ),
            "raw_score": round(raw_score, 2),
            "positive_signals": [
                {
                    "type": s.get("type", ""),
                    "direction": s.get("direction", ""),
                    "confidence": s.get("confidence", 0.0),
                    "reason": s.get("reason", ""),
                }
                for s in positive_signals
            ],
            "negative_signals": [
                {
                    "type": s.get("type", ""),
                    "direction": s.get("direction", ""),
                    "confidence": s.get("confidence", 0.0),
                    "reason": s.get("reason", ""),
                }
                for s in negative_signals
            ],
        }

        result = AggregatedSignal(
            symbol=symbol,
            overall_score=round(overall_score, 2),
            bias=bias,
            confidence=round(avg_conf, 4),
            signal_count=total_count,
            positive_count=pos_count,
            negative_count=neg_count,
            details=details,
            timestamp=now,
        )

        self._log.debug(
            f"{symbol} | score={result.overall_score:+.2f} "
            f"bias={result.bias} conf={result.confidence:.2f} "
            f"(pos={pos_count} neg={neg_count} total={total_count})"
        )

        return result
