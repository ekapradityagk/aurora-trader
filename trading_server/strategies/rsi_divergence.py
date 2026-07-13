"""Aurora Trader — RSI Divergence + SMC Strategy.

Detects hidden and regular RSI divergences on 1H/4H timeframes and
checks for Smart Money Concepts (Order Block / FVG) confluence.

Requires 4-condition confluence for a valid signal:

1. RSI divergence detected (regular or hidden)
2. Order Block present at the divergence zone
3. Fair Value Gap (FVG) overlaps the entry
4. Price is at a key level (swing high/low)

Target Win Rate: 75%
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from shared.constants import INDICATOR_DEFAULTS
from shared.logger import get_logger
from shared.models import Signal, SignalDirection, TimeFrame

from trading_server.strategies.base import BaseStrategy

logger = get_logger("trading_server.strategy.rsi_divergence")


class RsiDivergenceStrategy(BaseStrategy):
    """RSI divergence + Smart Money Concepts strategy."""

    name = "rsi_divergence"

    def __init__(self) -> None:
        super().__init__()
        self._rsi_period = INDICATOR_DEFAULTS["rsi"]["period"]
        self._rsi_oversold = 25
        self._rsi_overbought = 75
        self._divergence_lookback = 30

        self._last_signals: Dict[str, Dict[str, Any]] = {}

    def configure(self, config: Dict[str, Any]) -> None:
        """Load strategy parameters."""
        super().configure(config)
        params = config.get("parameters", {})
        self._rsi_period = params.get("rsi_period", self._rsi_period)
        self._rsi_oversold = params.get("rsi_oversold", self._rsi_oversold)
        self._rsi_overbought = params.get("rsi_overbought", self._rsi_overbought)
        self._divergence_lookback = params.get(
            "divergence_lookback", self._divergence_lookback
        )

    async def execute(
        self,
        symbol: str,
        data: Dict[str, Any],
        regime: Optional[str] = None,
    ) -> Optional[Signal]:
        """Evaluate RSI divergence + SMC confluence."""
        if not self._enabled:
            return None

        # Need both 1h and 4h data
        data_1h = data.get("1h")
        data_4h = data.get("4h")

        if not data_1h or not data_4h:
            return None

        klines_1h = data_1h.get("klines", [])
        klines_4h = data_4h.get("klines", [])

        if len(klines_1h) < self._divergence_lookback + 10:
            return None
        if len(klines_4h) < self._divergence_lookback // 4 + 5:
            return None

        prices_1h = self.extract_prices(klines_1h)
        prices_4h = self.extract_prices(klines_4h)

        closes_1h = prices_1h["close"]
        highs_1h = prices_1h["high"]
        lows_1h = prices_1h["low"]
        closes_4h = prices_4h["close"]
        highs_4h = prices_4h["high"]
        lows_4h = prices_4h["low"]

        current_close = closes_1h[-1]
        current_high = highs_1h[-1]
        current_low = lows_1h[-1]

        # --- Condition 1: RSI Divergence (on 4H for higher reliability) ---
        rsi_4h = self.compute_rsi(closes_4h, self._rsi_period)
        rsi_1h = self.compute_rsi(closes_1h, self._rsi_period)

        div_result = self._detect_divergence(
            closes_4h, rsi_4h, highs_4h, lows_4h
        )
        has_divergence = div_result is not None
        div_type = div_result[0] if div_result else None
        div_direction = div_result[1] if div_result else None

        if not has_divergence:
            return None

        # --- Condition 2: Order Block (SMC) ---
        ob_result = self._find_order_block(
            closes_4h, highs_4h, lows_4h, div_direction
        )
        has_order_block = ob_result is not None

        if not has_order_block:
            return None

        # --- Condition 3: Fair Value Gap ---
        fvg_result = self._find_fvg(
            closes_4h, highs_4h, lows_4h, div_direction
        )
        has_fvg = fvg_result is not None

        # --- Condition 4: Price at swing high/low ---
        is_at_level = self._is_at_swing_level(
            current_close, highs_4h, lows_4h
        )

        # Confluence score
        conditions_met = sum([has_divergence, has_order_block, has_fvg, is_at_level])
        if conditions_met < 4:
            logger.debug(
                f"{symbol} | Only {conditions_met}/4 confluence conditions met "
                f"(div={has_divergence}, ob={has_order_block}, "
                f"fvg={has_fvg}, level={is_at_level})"
            )
            return None

        # Determine direction from divergence
        if div_direction == "bullish":
            direction = SignalDirection.LONG
            confidence = 0.70 + (4.0 / 100.0)  # base + per-condition boost
        else:
            direction = SignalDirection.SHORT
            confidence = 0.70 + (4.0 / 100.0)

        confidence = min(confidence, 0.90)

        # Build metadata
        entry_price = Decimal(str(current_close))
        atr_values = self.compute_atr(
            highs_1h, lows_1h, closes_1h, 14
        )
        current_atr = atr_values[-1] if atr_values[-1] > 0 else (current_high - current_low) * 0.01
        atr_decimal = Decimal(str(current_atr))

        if direction == SignalDirection.LONG:
            stop_loss = entry_price - atr_decimal * Decimal("2")
            take_profit = entry_price + atr_decimal * Decimal("3")
        else:
            stop_loss = entry_price + atr_decimal * Decimal("2")
            take_profit = entry_price - atr_decimal * Decimal("3")

        reason = (
            f"{div_type} {div_direction} divergence on 4H with "
            f"{conditions_met}/4 confluence conditions. "
            f"OB={'Y' if has_order_block else 'N'}, "
            f"FVG={'Y' if has_fvg else 'N'}, "
            f"SwingLevel={'Y' if is_at_level else 'N'}"
        )

        signal = Signal(
            strategy_name=self.name,
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 4),
            price=entry_price,
            timeframe=TimeFrame.H4,
            reason=reason,
            indicators={
                "rsi_4h": round(rsi_4h[-1], 2),
                "rsi_1h": round(rsi_1h[-1], 2),
                "divergence_type": div_type,
                "divergence_direction": div_direction,
                "conditions_met": conditions_met,
                "has_order_block": has_order_block,
                "has_fvg": has_fvg,
                "at_swing_level": is_at_level,
            },
            metadata={
                "entry_price": str(entry_price),
                "stop_loss": str(stop_loss),
                "take_profit": str(take_profit),
                "atr": str(atr_decimal),
            },
        )

        # Dedup
        last = self._last_signals.get(symbol)
        if last and last["direction"] == direction.value:
            return None
        self._last_signals[symbol] = {
            "direction": direction.value,
            "price": str(entry_price),
        }

        logger.info(
            f"{symbol} | {direction.value.upper()} signal | "
            f"confidence={confidence:.2f} | {reason}"
        )
        return signal

    def reset(self) -> None:
        self._last_signals.clear()

    # ------------------------------------------------------------------
    # Divergence Detection
    # ------------------------------------------------------------------

    def _detect_divergence(
        self,
        closes: List[float],
        rsi: List[float],
        highs: List[float],
        lows: List[float],
    ) -> Optional[Tuple[str, str]]:
        """Detect RSI divergence.

        Returns (type, direction) tuple where:
          type = "regular" | "hidden"
          direction = "bullish" | "bearish"

        Regular bullish: price makes lower low, RSI makes higher low
        Regular bearish: price makes higher high, RSI makes lower high
        Hidden bullish: price makes higher low, RSI makes lower low
        Hidden bearish: price makes lower high, RSI makes higher high
        """
        lookback = min(self._divergence_lookback, len(closes) - 1)
        if lookback < 20:
            return None

        # Find swing highs and lows in the lookback window
        swing_highs_idx = self._find_swing_highs(highs, lookback)
        swing_lows_idx = self._find_swing_lows(lows, lookback)

        # Check for bearish divergences (price high, RSI low)
        if len(swing_highs_idx) >= 2:
            idx2 = swing_highs_idx[-1]
            idx1 = swing_highs_idx[-2]
            price_high1, price_high2 = highs[idx1], highs[idx2]
            rsi_val1, rsi_val2 = rsi[idx1], rsi[idx2]

            # Regular bearish: higher price high + lower RSI high
            if price_high2 > price_high1 and rsi_val2 < rsi_val1 and rsi_val2 > 50:
                return ("regular", "bearish")

            # Hidden bearish: lower price high + higher RSI high
            if price_high2 < price_high1 and rsi_val2 > rsi_val1 and rsi_val1 > 50:
                return ("hidden", "bearish")

        # Check for bullish divergences (price low, RSI high)
        if len(swing_lows_idx) >= 2:
            idx2 = swing_lows_idx[-1]
            idx1 = swing_lows_idx[-2]
            price_low1, price_low2 = lows[idx1], lows[idx2]
            rsi_val1, rsi_val2 = rsi[idx1], rsi[idx2]

            # Regular bullish: lower price low + higher RSI low
            if price_low2 < price_low1 and rsi_val2 > rsi_val1 and rsi_val2 < 50:
                return ("regular", "bullish")

            # Hidden bullish: higher price low + lower RSI low
            if price_low2 > price_low1 and rsi_val2 < rsi_val1 and rsi_val1 < 50:
                return ("hidden", "bullish")

        return None

    @staticmethod
    def _find_swing_highs(data: List[float], lookback: int) -> List[int]:
        """Find swing high indices over the lookback window."""
        swings: List[int] = []
        start = max(0, len(data) - lookback)
        for i in range(start + 1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                swings.append(i)
        return swings

    @staticmethod
    def _find_swing_lows(data: List[float], lookback: int) -> List[int]:
        """Find swing low indices over the lookback window."""
        swings: List[int] = []
        start = max(0, len(data) - lookback)
        for i in range(start + 1, len(data) - 1):
            if data[i] < data[i - 1] and data[i] < data[i + 1]:
                swings.append(i)
        return swings

    # ------------------------------------------------------------------
    # Order Block Detection (SMC)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_order_block(
        closes: List[float],
        highs: List[float],
        lows: List[float],
        direction: Optional[str],
    ) -> Optional[Dict[str, float]]:
        """Find the most recent order block.

        A bullish OB is the last bearish candle before a significant upward move.
        A bearish OB is the last bullish candle before a significant downward move.
        """
        if len(closes) < 10:
            return None

        if direction == "bullish":
            # Find the last bearish candle before a 3-candle rally
            for i in range(len(closes) - 4, 3, -1):
                if (highs[i + 1] > highs[i] and
                    highs[i + 2] > highs[i + 1] and
                    closes[i] < opens_estimate(closes[i], highs[i], lows[i])):
                    return {
                        "type": "bullish_ob",
                        "high": highs[i],
                        "low": lows[i],
                        "index": i,
                    }

        elif direction == "bearish":
            # Find the last bullish candle before a 3-candle decline
            for i in range(len(closes) - 4, 3, -1):
                if (lows[i + 1] < lows[i] and
                    lows[i + 2] < lows[i + 1] and
                    closes[i] > opens_estimate(closes[i], highs[i], lows[i])):
                    return {
                        "type": "bearish_ob",
                        "high": highs[i],
                        "low": lows[i],
                        "index": i,
                    }

        return None

    # ------------------------------------------------------------------
    # Fair Value Gap Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_fvg(
        closes: List[float],
        highs: List[float],
        lows: List[float],
        direction: Optional[str],
    ) -> Optional[Dict[str, float]]:
        """Detect the most recent Fair Value Gap (FVG).

        A bullish FVG occurs when the low of candle i+2 > high of candle i.
        A bearish FVG occurs when the high of candle i+2 < low of candle i.
        """
        if len(closes) < 5:
            return None

        if direction == "bullish":
            for i in range(len(closes) - 3, 1, -1):
                if lows[i + 2] > highs[i]:
                    return {
                        "type": "bullish_fvg",
                        "upper": lows[i + 2],
                        "lower": highs[i],
                        "index": i,
                    }

        elif direction == "bearish":
            for i in range(len(closes) - 3, 1, -1):
                if highs[i + 2] < lows[i]:
                    return {
                        "type": "bearish_fvg",
                        "upper": lows[i],
                        "lower": highs[i + 2],
                        "index": i,
                    }

        return None

    # ------------------------------------------------------------------
    # Swing Level Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_at_swing_level(
        price: float,
        highs: List[float],
        lows: List[float],
        threshold_pct: float = 0.002,
    ) -> bool:
        """Check if price is near a recent swing high or low."""
        if len(highs) < 20:
            return False

        recent_highs = highs[-20:]
        recent_lows = lows[-20:]

        near_high = any(
            abs(price - h) / h < threshold_pct
            for h in recent_highs[-5:]
        )
        near_low = any(
            abs(price - l) / l < threshold_pct
            for l in recent_lows[-5:]
        )

        return near_high or near_low


# ---------------------------------------------------------------------------
# Helper: estimate open price from OHLC
# ---------------------------------------------------------------------------


def opens_estimate(close: float, high: float, low: float) -> float:
    """Estimate the open price from OHLC data."""
    return (high + low) / 2.0
