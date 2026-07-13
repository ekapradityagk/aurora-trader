"""Aurora Trader — Trend Following Strategy (Supertrend + EMA).

Used when ADX > 25 (trending market). Combines Supertrend for
direction with EMA(50/200) as a trend filter and a trailing stop
for exits.

- Entry: Supertrend flips direction + price above/below EMAs
- Exit: Trailing stop (Supertrend flip or ATR-based)
- Risk:Reward target: 3:1
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from shared.constants import INDICATOR_DEFAULTS
from shared.logger import get_logger
from shared.models import Signal, SignalDirection, TimeFrame

from trading_server.strategies.base import BaseStrategy

logger = get_logger("trading_server.strategy.trend_follow")


class TrendFollowStrategy(BaseStrategy):
    """Supertrend + EMA trend-following strategy for trending markets."""

    name = "trend_follow"

    def __init__(self) -> None:
        super().__init__()
        self._st_period = 10
        self._st_multiplier = 3.0
        self._ema_fast = 50
        self._ema_slow = 200
        self._adx_threshold = 25
        self._atr_period = 14

        self._last_signals: Dict[str, Dict[str, Any]] = {}
        self._supertrend_cache: Dict[str, Dict[str, Any]] = {}

    def configure(self, config: Dict[str, Any]) -> None:
        """Load strategy parameters."""
        super().configure(config)
        params = config.get("parameters", {})
        self._ema_fast = params.get("ema_fast", self._ema_fast)
        self._ema_slow = params.get("ema_slow", self._ema_slow)

    async def execute(
        self,
        symbol: str,
        data: Dict[str, Any],
        regime: Optional[str] = None,
    ) -> Optional[Signal]:
        """Evaluate trend-following conditions."""
        if not self._enabled:
            return None

        # Use 1h data as the primary timeframe
        tf_data = data.get("1h") or data.get("4h") or data.get("5m")
        if not tf_data:
            return None

        klines = tf_data.get("klines", [])
        if len(klines) < self._ema_slow + 10:
            return None

        prices = self.extract_prices(klines)
        closes = prices["close"]
        highs = prices["high"]
        lows = prices["low"]
        current_close = closes[-1]

        # 1. Compute ADX to confirm trending market
        adx_values = self.compute_adx(highs, lows, closes, 14)
        current_adx = adx_values[-1]

        if current_adx < self._adx_threshold:
            logger.debug(
                f"{symbol} | ADX={current_adx:.1f} < {self._adx_threshold}, "
                f"not trending — skipping trend follow"
            )
            return None

        # 2. Compute EMAs
        ema_fast_values = self.compute_ema(closes, self._ema_fast)
        ema_slow_values = self.compute_ema(closes, self._ema_slow)
        ema_fast = ema_fast_values[-1]
        ema_slow = ema_slow_values[-1]

        if ema_fast == 0 or ema_slow == 0:
            return None

        # 3. Compute Supertrend
        st_result = self.compute_supertrend(
            highs, lows, closes, self._st_period, self._st_multiplier
        )
        st_direction = st_result["direction"][-1]
        st_band = st_result["band"][-1]

        # 4. Determine entry signal
        direction: Optional[SignalDirection] = None
        confidence = 0.0
        reason = ""

        # LONG conditions: uptrend on SuperTrend, price above both EMAs
        if (
            st_direction == 1  # 1 = uptrend
            and current_close > ema_fast
            and ema_fast > ema_slow
            and current_close > st_band
        ):
            direction = SignalDirection.LONG
            # Stronger confidence with larger EMA gap
            ema_gap = (ema_fast - ema_slow) / ema_slow
            confidence = min(0.85, 0.55 + ema_gap * 10)
            reason = (
                f"Uptrend: Supertrend bullish, price {current_close:.2f} > "
                f"EMA{self._ema_fast}={ema_fast:.2f} > EMA{self._ema_slow}={ema_slow:.2f}, "
                f"ADX={current_adx:.1f}"
            )

        # SHORT conditions: downtrend on SuperTrend, price below both EMAs
        elif (
            st_direction == -1  # -1 = downtrend
            and current_close < ema_fast
            and ema_fast < ema_slow
            and current_close < st_band
        ):
            direction = SignalDirection.SHORT
            ema_gap = (ema_slow - ema_fast) / ema_fast
            confidence = min(0.85, 0.55 + ema_gap * 10)
            reason = (
                f"Downtrend: Supertrend bearish, price {current_close:.2f} < "
                f"EMA{self._ema_fast}={ema_fast:.2f} < EMA{self._ema_slow}={ema_slow:.2f}, "
                f"ADX={current_adx:.1f}"
            )

        if direction is None:
            return None

        # 5. Calculate levels
        entry_price = Decimal(str(current_close))
        atr_values = self.compute_atr(highs, lows, closes, self._atr_period)
        current_atr = atr_values[-1] if atr_values[-1] > 0 else (max(highs[-20:]) - min(lows[-20:])) * 0.01
        atr_decimal = Decimal(str(current_atr))

        # 3:1 risk:reward
        if direction == SignalDirection.LONG:
            stop_loss = entry_price - atr_decimal * Decimal("2")
            take_profit = entry_price + atr_decimal * Decimal("6")  # 3:1 R:R
        else:
            stop_loss = entry_price + atr_decimal * Decimal("2")
            take_profit = entry_price - atr_decimal * Decimal("6")  # 3:1 R:R

        # 6. Build signal
        signal = Signal(
            strategy_name=self.name,
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 4),
            price=entry_price,
            timeframe=TimeFrame.H1,
            reason=reason,
            indicators={
                "adx": round(current_adx, 2),
                "ema_fast": round(ema_fast, 2),
                "ema_slow": round(ema_slow, 2),
                "supertrend_direction": st_direction,
                "supertrend_band": round(st_band, 2),
                "atr": round(current_atr, 6),
            },
            metadata={
                "entry_price": str(entry_price),
                "stop_loss": str(stop_loss),
                "take_profit": str(take_profit),
                "rr_ratio": "3:1",
                "atr": str(atr_decimal),
                "adx": round(current_adx, 2),
                "regime": regime or "trending",
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
        self._supertrend_cache.clear()

    # ------------------------------------------------------------------
    # Supertrend Calculation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_supertrend(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 10,
        multiplier: float = 3.0,
    ) -> Dict[str, List[float]]:
        """Compute the Supertrend indicator.

        Returns:
            Dict with keys:
              - "direction": 1 (uptrend) or -1 (downtrend)
              - "band": the upper/lower band value
        """
        n = len(closes)
        direction = [1] * n
        band = [0.0] * n

        if n < period + 1:
            return {"direction": direction, "band": band}

        # True Range and ATR
        tr = [0.0] * n
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        tr[0] = highs[0] - lows[0]

        atr = [0.0] * n
        atr[period] = sum(tr[1 : period + 1]) / period  # noqa: E203
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        # Basic bands
        hl2 = [(highs[i] + lows[i]) / 2.0 for i in range(n)]

        upper_band = [0.0] * n
        lower_band = [0.0] * n

        for i in range(period, n):
            upper_band[i] = hl2[i] + multiplier * atr[i]
            lower_band[i] = hl2[i] - multiplier * atr[i]

        # Final band and direction
        for i in range(period, n):
            if closes[i] > upper_band[i - 1]:
                direction[i] = 1
            elif closes[i] < lower_band[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
                # Adjust band
                if direction[i] == 1:
                    lower_band[i] = max(lower_band[i], lower_band[i - 1])
                else:
                    upper_band[i] = min(upper_band[i], upper_band[i - 1])

            band[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

        return {"direction": direction, "band": band}
