"""Aurora Trader — Mean Reversion Strategy (BB + RSI).

Entry Logic:
- LONG: Price touches lower Bollinger Band AND RSI < 30
- SHORT: Price touches upper Bollinger Band AND RSI > 70

Filter:
- ADX(14) < 25 (ranging / non-trending market only)

Exits:
- Mid-band (1R) or opposite band (2R)
- Stop: 2 × ATR from entry

Target Win Rate: 71%
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from shared.constants import INDICATOR_DEFAULTS
from shared.logger import get_logger
from shared.models import Signal, SignalDirection, TimeFrame

from trading_server.strategies.base import BaseStrategy

logger = get_logger("trading_server.strategy.mean_reversion")


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band + RSI mean reversion strategy for ranging markets."""

    name = "mean_reversion"

    def __init__(self) -> None:
        super().__init__()
        self._bb_period = INDICATOR_DEFAULTS["bbands"]["period"]
        self._bb_std = INDICATOR_DEFAULTS["bbands"]["std_dev"]
        self._rsi_period = INDICATOR_DEFAULTS["rsi"]["period"]
        self._rsi_oversold = 30
        self._rsi_overbought = 70
        self._adx_period = 14
        self._adx_threshold = 25
        self._atr_period = 14
        self._atr_multiplier = 2.0  # stop loss multiplier

        # Track last signal per symbol to avoid duplicates
        self._last_signals: Dict[str, Dict[str, Any]] = {}

    def configure(self, config: Dict[str, Any]) -> None:
        """Load strategy parameters from configuration."""
        super().configure(config)
        params = config.get("parameters", {})
        self._bb_period = params.get("bb_period", self._bb_period)
        self._bb_std = params.get("bb_std_dev", self._bb_std)
        self._rsi_period = params.get("rsi_period", self._rsi_period)

        risk = config.get("risk", {})
        self._atr_multiplier = risk.get("stop_loss_pct", 2.0) / 100.0
        if self._atr_multiplier <= 0:
            self._atr_multiplier = 2.0

        self._log = logger

    async def execute(
        self,
        symbol: str,
        data: Dict[str, Any],
        regime: Optional[str] = None,
    ) -> Optional[Signal]:
        """Evaluate mean reversion conditions and return a signal if met."""
        if not self._enabled:
            return None

        # We need at least the 1h data for reliable BB/RSI
        tf_data = data.get("1h") or data.get("5m")
        if not tf_data:
            return None

        klines = tf_data.get("klines", [])
        if len(klines) < self._bb_period + 10:
            return None

        prices = self.extract_prices(klines)
        closes = prices["close"]
        highs = prices["high"]
        lows = prices["low"]
        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]

        # 1. Compute indicators
        rsi_values = self.compute_rsi(closes, self._rsi_period)
        current_rsi = rsi_values[-1]

        bb = self.compute_bollinger_bands(closes, self._bb_period, self._bb_std)
        current_upper = bb["upper"][-1]
        current_middle = bb["middle"][-1]
        current_lower = bb["lower"][-1]

        adx_values = self.compute_adx(highs, lows, closes, self._adx_period)
        current_adx = adx_values[-1]

        atr_values = self.compute_atr(highs, lows, closes, self._atr_period)
        current_atr = atr_values[-1] if atr_values[-1] > 0 else (current_high - current_low) * 0.01

        # Store indicators for analysis
        self._last_indicators = {
            "rsi": round(current_rsi, 2),
            "adx": round(current_adx, 2),
            "bb_upper": round(current_upper, 2),
            "bb_middle": round(current_middle, 2),
            "bb_lower": round(current_lower, 2),
            "atr": round(current_atr, 6),
            "close": round(current_close, 2),
        }

        # 2. Filter: ADX < 25 (ranging market only)
        if current_adx >= self._adx_threshold:
            msg = (f"ADX={current_adx:.1f} ≥ {self._adx_threshold}, trending — not suitable for mean reversion")
            self._last_skip_reason = msg
            logger.debug(f"{symbol} | {msg}")
            return None

        # 3. Check entry conditions
        direction: Optional[SignalDirection] = None
        confidence = 0.0
        reason = ""
        indicators: Dict[str, Any] = {
            "rsi": round(current_rsi, 2),
            "adx": round(current_adx, 2),
            "bb_upper": round(current_upper, 2),
            "bb_middle": round(current_middle, 2),
            "bb_lower": round(current_lower, 2),
            "atr": round(current_atr, 6),
        }

        # --- LONG: Price touches lower BB + RSI < 30 ---
        if current_close <= current_lower * 1.002 and current_rsi < self._rsi_oversold:
            direction = SignalDirection.LONG
            confidence = min(0.80, 0.50 + (self._rsi_oversold - current_rsi) / 100.0)
            reason = (
                f"Price at lower BB ({current_close:.2f} ≤ {current_lower:.2f}) "
                f"with RSI {current_rsi:.1f} < {self._rsi_oversold} in ranging market "
                f"(ADX={current_adx:.1f})"
            )

        # --- SHORT: Price touches upper BB + RSI > 70 ---
        elif current_close >= current_upper * 0.998 and current_rsi > self._rsi_overbought:
            direction = SignalDirection.SHORT
            confidence = min(0.80, 0.50 + (current_rsi - self._rsi_overbought) / 100.0)
            reason = (
                f"Price at upper BB ({current_close:.2f} ≥ {current_upper:.2f}) "
                f"with RSI {current_rsi:.1f} > {self._rsi_overbought} in ranging market "
                f"(ADX={current_adx:.1f})"
            )

        if direction is None:
            self._last_skip_reason = (
                f"No entry signal: RSI={current_rsi:.1f} (need <30 for LONG or >70 for SHORT), "
                f"price not at BB bands"
            )
            return None

        # 4. Calculate levels
        entry_price = Decimal(str(current_close))
        atr_decimal = Decimal(str(current_atr))
        stop_distance = atr_decimal * Decimal(str(self._atr_multiplier))

        if direction == SignalDirection.LONG:
            stop_loss = entry_price - stop_distance
            take_profit_mid = Decimal(str(current_middle))  # 1R: mid-band
            take_profit_upper = Decimal(str(current_upper))  # 2R: upper band
        else:
            stop_loss = entry_price + stop_distance
            take_profit_mid = Decimal(str(current_middle))  # 1R: mid-band
            take_profit_lower = Decimal(str(current_lower))  # 2R: lower band

        # 5. Build and return signal
        signal = Signal(
            strategy_name=self.name,
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 4),
            price=entry_price,
            timeframe=TimeFrame.H1,
            reason=reason,
            indicators=indicators,
            metadata={
                "entry_price": str(entry_price),
                "stop_loss": str(stop_loss),
                "take_profit_1r": str(take_profit_mid),
                "take_profit_2r": str(
                    take_profit_upper if direction == SignalDirection.LONG else take_profit_lower
                ),
                "atr": str(atr_decimal),
                "adx": round(current_adx, 2),
                "regime": regime or "ranging",
            },
        )

        # Deduplicate: don't fire the same signal twice in a row
        last = self._last_signals.get(symbol)
        if last and last["direction"] == direction.value:
            self._last_skip_reason = f"Duplicate {direction.value} signal suppressed (already sent)"
            logger.debug(f"{symbol} | Duplicate {direction.value} signal suppressed")
            return None

        self._last_signals[symbol] = {
            "direction": direction.value,
            "price": str(entry_price),
            "timestamp": signal.timestamp,
        }

        logger.info(
            f"{symbol} | {direction.value.upper()} signal | "
            f"confidence={confidence:.2f} | {reason}"
        )
        return signal

    def reset(self) -> None:
        """Clear per-symbol signal tracking."""
        self._last_signals.clear()
