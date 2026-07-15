"""Aurora Trader — Abstract Base Strategy.

Defines the interface that all trading strategies must implement.
Each strategy receives market data and returns a Signal indicating
the desired trading action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from shared.models import Signal, SignalDirection, TimeFrame


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement:
    - ``name`` (class attribute)
    - ``execute(symbol, data)`` → returns a Signal or None

    Optionally override:
    - ``configure(config_dict)`` for loading strategy-specific parameters
    - ``reset()`` for per-session state clearing
    """

    name: str = "base"

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._enabled: bool = True
        self._last_skip_reason: str = ""
        self._last_indicators: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_skip_reason(self) -> str:
        """Reason this strategy skipped on the last execute() call."""
        return self._last_skip_reason

    @property
    def last_indicators(self) -> Dict[str, Any]:
        """Indicator values from the last execute() call."""
        return dict(self._last_indicators)

    @abstractmethod
    async def execute(
        self,
        symbol: str,
        data: Dict[str, Any],
        regime: Optional[str] = None,
    ) -> Optional[Signal]:
        """Analyse market data and return a trading signal.

        Args:
            symbol: Trading pair symbol, e.g. ``"BTCUSDT"``.
            data: Dictionary of kline/indicator data keyed by timeframe.
                  Structure::

                      {
                          "1m": {"klines": [...], "indicators": {...}},
                          "5m": {"klines": [...], "indicators": {...}},
                          "1h": {"klines": [...], "indicators": {...}},
                          ...
                      }

            regime: Optional market regime label (e.g. "trending", "ranging").

        Returns:
            A :class:`Signal` if conditions are met, or ``None`` to skip.
        """
        ...

    def configure(self, config: Dict[str, Any]) -> None:
        """Load strategy-specific configuration.

        Subclasses can override this to extract custom parameters.
        """
        self._config = config
        self._enabled = config.get("enabled", True)

    def reset(self) -> None:
        """Reset any per-session state.

        Called at the start of each trading day or session.
        """
        pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def extract_prices(
        klines: List[Dict[str, Any]],
    ) -> Dict[str, List[float]]:
        """Extract OHLCV arrays from a list of kline dicts."""
        return {
            "open": [float(k["open"]) for k in klines],
            "high": [float(k["high"]) for k in klines],
            "low": [float(k["low"]) for k in klines],
            "close": [float(k["close"]) for k in klines],
            "volume": [float(k["volume"]) for k in klines],
        }

    @staticmethod
    def compute_rsi(
        closes: List[float], period: int = 14
    ) -> List[float]:
        """Compute Relative Strength Index values."""
        if len(closes) < period + 1:
            return [50.0] * len(closes)

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = [50.0] * period  # pad with neutral values
        for i in range(period, len(closes)):
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi_values.append(rsi)

            # Update averages for next period
            if i < len(closes) - 1:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        return rsi_values

    @staticmethod
    def compute_atr(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14,
    ) -> List[float]:
        """Compute Average True Range values."""
        if len(closes) < 2:
            return [0.0] * len(closes)

        trs = [highs[i] - lows[i] for i in range(len(closes))]
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs[i] = tr
        trs[0] = highs[0] - lows[0]

        atr_values = [0.0] * len(closes)
        atr_values[period] = sum(trs[1 : period + 1]) / period

        for i in range(period + 1, len(closes)):
            atr_values[i] = (
                atr_values[i - 1] * (period - 1) + trs[i]
            ) / period

        return atr_values

    @staticmethod
    def compute_bollinger_bands(
        closes: List[float],
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Dict[str, List[float]]:
        """Compute Bollinger Bands (upper, middle, lower)."""
        import statistics

        upper = [0.0] * len(closes)
        middle = [0.0] * len(closes)
        lower = [0.0] * len(closes)

        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1 : i + 1]
            mean = sum(window) / period
            std = statistics.stdev(window) if len(window) > 1 else 0.0
            middle[i] = mean
            upper[i] = mean + std_dev * std
            lower[i] = mean - std_dev * std

        return {"upper": upper, "middle": middle, "lower": lower}

    @staticmethod
    def compute_adx(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14,
    ) -> List[float]:
        """Compute Average Directional Index."""
        if len(closes) < period + 1:
            return [0.0] * len(closes)

        # True Range
        tr = [0.0] * len(closes)
        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        tr[0] = highs[0] - lows[0]

        # Directional Movements
        up_move = [0.0] * len(closes)
        down_move = [0.0] * len(closes)
        for i in range(1, len(closes)):
            up_move[i] = highs[i] - highs[i - 1]
            down_move[i] = lows[i - 1] - lows[i]

        # Directional Indicators
        plus_di = [0.0] * len(closes)
        minus_di = [0.0] * len(closes)

        # Use Wilder's smoothing
        atr = [0.0] * len(closes)
        plus_dm = [0.0] * len(closes)
        minus_dm = [0.0] * len(closes)

        # First ATR value
        atr[period] = sum(tr[1 : period + 1]) / period
        for i in range(1, period + 1):
            if up_move[i] > down_move[i] and up_move[i] > 0:
                plus_dm[i] = up_move[i]
            if down_move[i] > up_move[i] and down_move[i] > 0:
                minus_dm[i] = down_move[i]

        first_plus_dm = sum(plus_dm[1 : period + 1])
        first_minus_dm = sum(minus_dm[1 : period + 1])

        for i in range(period, len(closes)):
            if i > period:
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
                plus_dm[i] = (plus_dm[i - 1] * (period - 1) + (
                    up_move[i] if up_move[i] > down_move[i] and up_move[i] > 0 else 0
                )) / period
                minus_dm[i] = (minus_dm[i - 1] * (period - 1) + (
                    down_move[i] if down_move[i] > up_move[i] and down_move[i] > 0 else 0
                )) / period

            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm[i] / atr[i]
                minus_di[i] = 100 * minus_dm[i] / atr[i]

        # DX and ADX
        dx = [0.0] * len(closes)
        for i in range(period, len(closes)):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        adx_values = [0.0] * len(closes)
        adx_values[period + period - 1] = sum(dx[period : period + period]) / period  # noqa: E203

        for i in range(period + period, len(closes)):
            adx_values[i] = (
                adx_values[i - 1] * (period - 1) + dx[i]
            ) / period

        return adx_values

    @staticmethod
    def compute_ema(values: List[float], period: int) -> List[float]:
        """Compute Exponential Moving Average."""
        result = [0.0] * len(values)
        multiplier = 2.0 / (period + 1)

        if len(values) < period:
            return result

        # Start with SMA
        result[period - 1] = sum(values[:period]) / period

        for i in range(period, len(values)):
            result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1]

        return result
