"""
Aurora Trader — Market Regime Detector.

Classifies the current market state into one of three regimes:
    - TREND (strong directional movement)
    - RANGE (sideways / mean-reverting)
    - VOLATILE (wide price swings, high volatility)

Uses multiple indicators on 1H and 4H timeframes:
    - ADX(14)          — trend strength
    - Bollinger Band width — volatility compression / expansion
    - ATR ratio        — recent volatility vs longer-term average
    - EMA slope        — direction and steepness of the trend

Each classification comes with a confidence score.  Results are cached to
avoid redundant computation.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.config import load_config
from shared.logger import get_logger
from shared.models import MarketRegimeType, TimeFrame

logger = get_logger("learning_server.regime")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Thresholds for ADX
ADX_TREND_THRESHOLD = 25  # ADX > 25 indicates trending
ADX_STRONG_TREND = 40     # ADX > 40 is very strong trend

# Bollinger Band width percentiles (relative to 20-period lookback)
BB_WIDTH_RANGE = 0.15     # BB width < 15% of avg → range
BB_WIDTH_VOLATILE = 0.30  # BB width > 30% of avg → volatile

# ATR ratio thresholds
ATR_RATIO_RANGE = 0.8     # ATR ratio < 0.8 → low relative volatility
ATR_RATIO_VOLATILE = 1.5  # ATR ratio > 1.5 → high relative volatility

# EMA slope thresholds (normalised)
EMA_SLOPE_FLAT = 0.001    # |slope| < 0.1% per period → flat
EMA_SLOPE_TREND = 0.003   # |slope| > 0.3% per period → strong trend

# Cache TTL (seconds)
CACHE_TTL = 300  # 5 minutes

# Minimum number of candles required for indicator computation
MIN_CANDLES = 50


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InsufficientDataError(ValueError):
    """Raised when there is not enough price data to compute indicators."""


# ---------------------------------------------------------------------------
# Regime classification result
# ---------------------------------------------------------------------------


@dataclass
class RegimeResult:
    """Output of a single regime detection call."""

    symbol: str
    regime: MarketRegimeType
    confidence: float  # 0.0 – 1.0
    timeframe: str
    scores: Dict[str, float] = field(default_factory=dict)
    indicators: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Indicator helpers (pure functions)
# ---------------------------------------------------------------------------


def _compute_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Compute Average Directional Index (ADX).

    ADX measures trend strength on a scale of 0-100.
    Values > 25 suggest a strong trend.
    """
    if len(closes) < period + 2:
        return 0.0

    # True Range
    tr_list: List[float] = []
    # Directional Movement
    plus_dm: List[float] = []
    minus_dm: List[float] = []

    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        prev_close = closes[i - 1]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)

        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)

        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)

    # Smooth with Wilder's method (approx via EMA)
    def wilder_smooth(values: List[float], p: int) -> List[float]:
        if not values:
            return []
        smoothed = [sum(values[:p]) / p]
        for v in values[p:]:
            smoothed.append((smoothed[-1] * (p - 1) + v) / p)
        return smoothed

    atr = wilder_smooth(tr_list, period)
    plus_di_raw = wilder_smooth(plus_dm, period)
    minus_di_raw = wilder_smooth(minus_dm, period)

    # Directional Index
    di_values: List[float] = []
    for i in range(len(atr)):
        if atr[i] == 0:
            di_values.append(50.0)
            continue
        pdi = (plus_di_raw[i] / atr[i]) * 100 if i < len(plus_di_raw) else 0
        mdi = (minus_di_raw[i] / atr[i]) * 100 if i < len(minus_di_raw) else 0
        dx = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) > 0 else 0
        di_values.append(dx)

    # ADX is smoothed DX
    if len(di_values) < period:
        return 0.0
    adx_values = wilder_smooth(di_values, period)
    return adx_values[-1] if adx_values else 0.0


def _compute_bollinger_width(closes: List[float], period: int = 20, std_dev: float = 2.0) -> float:
    """Compute Bollinger Band width as a fraction of the middle band.

    Width = (upper - lower) / middle
    """
    if len(closes) < period:
        return 0.0

    recent = closes[-period:]
    mean = sum(recent) / len(recent)
    variance = sum((c - mean) ** 2 for c in recent) / len(recent)
    std = math.sqrt(variance) if variance > 0 else 0.0

    upper = mean + std_dev * std
    lower = mean - std_dev * std

    if mean == 0:
        return 0.0
    return (upper - lower) / mean


def _compute_bollinger_width_history(closes: List[float], period: int = 20) -> List[float]:
    """Compute Bollinger Band width over a rolling window for comparison."""
    widths: List[float] = []
    for i in range(period, len(closes) + 1):
        window = closes[i - period:i]
        w = _compute_bollinger_width(window, period=period)
        widths.append(w)
    return widths


def _compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Compute Average True Range (ATR)."""
    if len(closes) < period + 1:
        return 0.0

    tr_values: List[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_values.append(tr)

    return sum(tr_values[-period:]) / period if len(tr_values) >= period else 0.0


def _compute_ema(values: List[float], period: int) -> List[float]:
    """Compute Exponential Moving Average."""
    if len(values) < period:
        return []
    multiplier = 2.0 / (period + 1)
    ema = [sum(values[:period]) / period]
    for v in values[period:]:
        ema.append((v - ema[-1]) * multiplier + ema[-1])
    return ema


def _compute_ema_slope(closes: List[float], period: int = 20) -> float:
    """Compute the normalised slope of the EMA.

    Returns a value representing the rate of change as a fraction per period.
    """
    ema_values = _compute_ema(closes, period)
    if len(ema_values) < 3:
        return 0.0

    # Use linear regression over the last N EMA values for robustness
    n = min(len(ema_values), 10)
    recent_ema = ema_values[-n:]

    x_vals = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(recent_ema) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, recent_ema))
    den = sum((x - mean_x) ** 2 for x in x_vals)

    if den == 0:
        return 0.0

    slope = num / den
    # Normalise by the mean value
    if mean_y == 0:
        return 0.0
    return slope / mean_y


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------


class RegimeDetector:
    """Detects market regime (TREND / RANGE / VOLATILE) using multiple
    technical indicators on 1H and 4H data.

    Results are cached to avoid redundant computation within the TTL window.
    """

    def __init__(self, cache_dir: str = "data/regime_cache") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory_cache: Dict[str, Tuple[float, RegimeResult]] = {}
        self._log = logger
        self._cfg = load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def detect(
        self,
        symbol: str,
        timeframe: str = "1h",
        ohlcv: Optional[List[Dict[str, Any]]] = None,
    ) -> RegimeResult:
        """Classify the current market regime for *symbol*.

        Args:
            symbol: Trading pair (e.g. "BTCUSDT").
            timeframe: Data timeframe ("1h" or "4h").
            ohlcv: List of OHLCV dicts with keys open, high, low, close, volume.
                   If None the detector will try cached data or raise.

        Returns:
            RegimeResult with regime type, confidence, and supporting scores.
        """
        # Check in-memory cache
        cache_key = f"{symbol}_{timeframe}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            self._log.debug(f"Cache hit for {cache_key}")
            return cached

        # Check persistent cache
        persistent = self._load_persistent_cache(cache_key)
        if persistent is not None:
            self._in_memory_cache[cache_key] = (time.time(), persistent)
            return persistent

        if not ohlcv or len(ohlcv) < MIN_CANDLES:
            raise InsufficientDataError(
                f"Need at least {MIN_CANDLES} candles for {symbol}@{timeframe}, "
                f"got {len(ohlcv) if ohlcv else 0}"
            )

        # Compute indicators
        indicators = self._compute_indicators(ohlcv)

        # Classify regime
        result = self._classify(symbol, timeframe, indicators)

        # Cache result
        self._cache_result(cache_key, result)

        return result

    async def detect_all(
        self,
        symbol: str,
        ohlcv_1h: Optional[List[Dict[str, Any]]] = None,
        ohlcv_4h: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, RegimeResult]:
        """Run regime detection on both 1H and 4H timeframes.

        Returns a dict keyed by timeframe.
        """
        results: Dict[str, RegimeResult] = {}

        if ohlcv_1h:
            try:
                results["1h"] = await self.detect(symbol, "1h", ohlcv_1h)
            except InsufficientDataError as exc:
                self._log.warning(f"1h regime detection failed: {exc}")

        if ohlcv_4h:
            try:
                results["4h"] = await self.detect(symbol, "4h", ohlcv_4h)
            except InsufficientDataError as exc:
                self._log.warning(f"4h regime detection failed: {exc}")

        return results

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def _compute_indicators(
        self, ohlcv: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute all indicators needed for regime classification."""
        highs = [float(c["high"]) for c in ohlcv]
        lows = [float(c["low"]) for c in ohlcv]
        closes = [float(c["close"]) for c in ohlcv]

        # ADX (14)
        adx = _compute_adx(highs, lows, closes, period=14)

        # Bollinger Band width
        bb_width = _compute_bollinger_width(closes, period=20)

        # BB width history for percentile comparison
        bb_widths = _compute_bollinger_width_history(closes, period=20)
        if bb_widths:
            avg_bb_width = sum(bb_widths) / len(bb_widths)
        else:
            avg_bb_width = bb_width

        # BB width ratio (current vs average)
        bb_width_ratio = bb_width / avg_bb_width if avg_bb_width > 0 else 1.0

        # ATR
        atr_current = _compute_atr(highs, lows, closes, period=14)
        atr_long = _compute_atr(highs, lows, closes, period=50)

        # ATR ratio (recent vs longer-term)
        atr_ratio = atr_current / atr_long if atr_long > 0 else 1.0

        # EMA slope (20-period)
        ema_slope = _compute_ema_slope(closes, period=20)

        # EMA slope (50-period for trend confirmation)
        ema_slope_long = _compute_ema_slope(closes, period=50)

        # Price vs EMA(20) position
        ema_20 = _compute_ema(closes, 20)
        price_vs_ema = (closes[-1] - ema_20[-1]) / ema_20[-1] if ema_20 and ema_20[-1] != 0 else 0.0

        return {
            "adx": adx,
            "bb_width": bb_width,
            "bb_width_ratio": bb_width_ratio,
            "avg_bb_width": avg_bb_width,
            "atr_current": atr_current,
            "atr_long": atr_long,
            "atr_ratio": atr_ratio,
            "ema_slope": ema_slope,
            "ema_slope_long": ema_slope_long,
            "price_vs_ema": price_vs_ema,
            "last_close": closes[-1] if closes else 0.0,
            "last_high": highs[-1] if highs else 0.0,
            "last_low": lows[-1] if lows else 0.0,
        }

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def _classify(
        self,
        symbol: str,
        timeframe: str,
        indicators: Dict[str, Any],
    ) -> RegimeResult:
        """Classify the market regime based on computed indicators.

        Returns a RegimeResult with:
            - The most likely regime (TREND / RANGE / VOLATILE)
            - Confidence score (0.0 – 1.0)
            - Per-regime confidence scores

        The algorithm uses a weighted voting approach:
            1. ADX > 25 → trend signal
            2. BB width ratio > 1.5 or ATR ratio > 1.5 → volatile signal
            3. BB width ratio < 0.8 and ATR ratio < 0.9 and low ADX → range signal
        """
        adx = indicators["adx"]
        bb_width_ratio = indicators["bb_width_ratio"]
        atr_ratio = indicators["atr_ratio"]
        ema_slope = indicators["ema_slope"]
        ema_slope_long = indicators["ema_slope_long"]

        # ---- Component scores (each 0.0 – 1.0) ----

        # Trend score: based on ADX and EMA slope
        trend_adx = min(1.0, max(0.0, (adx - 15) / 30))  # 15→0, 45→1.0
        trend_ema = min(1.0, abs(ema_slope) / EMA_SLOPE_TREND)
        if abs(ema_slope_long) > EMA_SLOPE_FLAT:
            trend_ema = max(trend_ema, min(1.0, abs(ema_slope_long) / EMA_SLOPE_TREND))
        trend_score = (trend_adx * 0.6 + trend_ema * 0.4)

        # Range score: low ADX, BB width compressed, low ATR
        range_adx = max(0.0, min(1.0, (25 - adx) / 25))  # adx=0→1, adx=25→0
        range_bb = max(0.0, min(1.0, (1.0 - bb_width_ratio) / 0.5))  # ratio=0.5→1, ratio=1→0
        range_atr = max(0.0, min(1.0, (1.0 - atr_ratio) / 0.5))  # ratio=0.5→1, ratio=1→0
        range_ema = max(0.0, min(1.0, (EMA_SLOPE_FLAT - abs(ema_slope)) / EMA_SLOPE_FLAT))
        range_score = (range_adx * 0.3 + range_bb * 0.3 + range_atr * 0.2 + range_ema * 0.2)

        # Volatile score: high BB width, high ATR ratio
        volatile_bb = max(0.0, min(1.0, (bb_width_ratio - 1.0) / 1.0))  # ratio=1→0, ratio=2→1
        volatile_atr = max(0.0, min(1.0, (atr_ratio - 1.0) / 1.0))  # ratio=1→0, ratio=2→1
        volatile_adx = max(0.0, min(1.0, adx / 50))  # adx=50→1
        volatile_score = (volatile_bb * 0.4 + volatile_atr * 0.4 + volatile_adx * 0.2)

        # ---- Determine regime ----

        scores = {
            "trend": round(trend_score, 4),
            "range": round(range_score, 4),
            "volatile": round(volatile_score, 4),
        }

        if trend_score >= range_score and trend_score >= volatile_score:
            # Trending
            if ema_slope > 0:
                regime = MarketRegimeType.TRENDING_BULL
            else:
                regime = MarketRegimeType.TRENDING_BEAR
            confidence = trend_score
        elif volatile_score >= trend_score and volatile_score >= range_score:
            if bb_width_ratio > 2.0 and atr_ratio > 2.0:
                regime = MarketRegimeType.BREAKOUT
            else:
                regime = MarketRegimeType.VOLATILE
            confidence = volatile_score
        else:
            if bb_width_ratio < 0.5 and atr_ratio < 0.5:
                regime = MarketRegimeType.LOW_VOLATILITY
            else:
                regime = MarketRegimeType.RANGING
            confidence = range_score

        # Normalise confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return RegimeResult(
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            timeframe=timeframe,
            scores=scores,
            indicators={
                "adx": round(adx, 2),
                "bb_width_ratio": round(bb_width_ratio, 4),
                "atr_ratio": round(atr_ratio, 4),
                "ema_slope": round(ema_slope, 6),
                "ema_slope_long": round(ema_slope_long, 6),
            },
        )

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _check_cache(self, key: str) -> Optional[RegimeResult]:
        """Check in-memory cache entry for *key*.

        Returns cached result if not expired, else None.
        """
        entry = self._in_memory_cache.get(key)
        if entry is None:
            return None
        ts, result = entry
        if time.time() - ts < CACHE_TTL:
            return result
        # Expired
        del self._in_memory_cache[key]
        return None

    def _cache_result(self, key: str, result: RegimeResult) -> None:
        """Store result in both in-memory and persistent cache."""
        self._in_memory_cache[key] = (time.time(), result)
        self._save_persistent_cache(key, result)

    def _persistent_cache_path(self, key: str) -> Path:
        """Return the file path for a persistent cache entry."""
        hashed = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{hashed}.json"

    def _save_persistent_cache(self, key: str, result: RegimeResult) -> None:
        """Save a regime result to the persistent cache (JSON)."""
        path = self._persistent_cache_path(key)
        data = {
            "symbol": result.symbol,
            "regime": result.regime.value,
            "confidence": result.confidence,
            "timeframe": result.timeframe,
            "scores": result.scores,
            "indicators": result.indicators,
            "cached_at": time.time(),
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except IOError as exc:
            self._log.debug(f"Failed to write regime cache: {exc}")

    def _load_persistent_cache(self, key: str) -> Optional[RegimeResult]:
        """Load a regime result from persistent cache if not expired."""
        path = self._persistent_cache_path(key)
        if not path.is_file():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            cached_at = data.get("cached_at", 0)
            if time.time() - cached_at > CACHE_TTL:
                path.unlink(missing_ok=True)
                return None
            return RegimeResult(
                symbol=data["symbol"],
                regime=MarketRegimeType(data["regime"]),
                confidence=data["confidence"],
                timeframe=data["timeframe"],
                scores=data.get("scores", {}),
                indicators=data.get("indicators", {}),
            )
        except (IOError, json.JSONDecodeError, KeyError, ValueError) as exc:
            self._log.debug(f"Failed to load regime cache: {exc}")
            return None

    def clear_cache(self) -> None:
        """Clear both in-memory and persistent caches."""
        self._in_memory_cache.clear()
        for f in self._cache_dir.glob("*.json"):
            f.unlink(missing_ok=True)
        self._log.info("Regime cache cleared")


# ---------------------------------------------------------------------------
# Convenience mapping
# ---------------------------------------------------------------------------

REGIME_TO_STRATEGY: Dict[MarketRegimeType, str] = {
    MarketRegimeType.TRENDING_BULL: "trend_follow",
    MarketRegimeType.TRENDING_BEAR: "trend_follow",
    MarketRegimeType.RANGING: "mean_reversion",
    MarketRegimeType.VOLATILE: "rsi_divergence",
    MarketRegimeType.LOW_VOLATILITY: "mean_reversion",
    MarketRegimeType.BREAKOUT: "trend_follow",
    MarketRegimeType.UNKNOWN: "mean_reversion",
}
