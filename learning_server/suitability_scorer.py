"""Aurora Trader — Pair Suitability Scorer.

Evaluates candidate trading pairs on movement quality metrics and
recommends the best N for active trading. Runs weekly to power
auto-rotation of the trading server's symbol list.

Scoring dimensions (each 0.0–1.0, higher = better for our style):
  1. MOVEMENT QUALITY (ATR%/price) — enough volatility to profit, not crazy
  2. TREND CLARITY (ADX) — clear direction > random chop
  3. VOLUME CONSISTENCY — liquidity we can rely on
  4. FUNDING RATE — cost to hold positions
  5. CANDLE SMOOTHNESS — less wick-noise means more reliable signals

The composite suitability_score weights these dimensions and is
used by the learning server to pick the active pair set each week.
"""

from __future__ import annotations

import asyncio
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from shared.config import load_config
from shared.logger import get_logger

logger = get_logger("learning_server.suitability_scorer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target ATR%/price range (sweet spot for our style)
ATR_PCT_IDEAL_MIN = 0.5    # Below this → too dead
ATR_PCT_IDEAL_MAX = 3.0    # Above this → too wild

# ADX thresholds
ADX_MIN = 15               # Below this → too choppy
ADX_IDEAL = 25             # Sweet spot for clear direction

# Volume consistency threshold
VOLUME_CONSISTENCY_MIN = 0.3  # Coefficient of variation threshold

# Funding rate thresholds (annualised)
FUNDING_IDEAL_MAX = 0.01     # 1% annualised → cheap

# Smoothness: max acceptable body-to-range ratio variance
SMOOTHNESS_MIN_BODY_RATIO = 0.3


@dataclass
class PairSuitability:
    """Suitability score for a single trading pair."""

    symbol: str
    composite_score: float           # 0.0 – 1.0
    movement_score: float            # ATR-based
    trend_score: float               # ADX-based
    volume_score: float              # Volume consistency
    funding_score: float             # Funding rate cost
    smoothness_score: float          # Candle body/range quality

    # Raw data for reference
    atr_pct: float
    adx: float
    volume_cv: float                 # Coefficient of variation
    funding_rate_annualised: float
    avg_body_ratio: float
    avg_volume_usdt: float
    close_price: float

    regime_label: str = "unknown"


@dataclass
class SuitabilityReport:
    """Full report for all scored pairs."""

    pairs: List[PairSuitability] = field(default_factory=list)
    top_picks: List[str] = field(default_factory=list)
    scan_timestamp: str = ""
    total_scored: int = 0
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------


def _compute_adx(
    highs: List[float], lows: List[float], closes: List[float], period: int = 14
) -> float:
    """Average Directional Index — trend strength."""
    if len(closes) < period + 2:
        return 0.0

    tr_list: List[float] = []
    plus_dm: List[float] = []
    minus_dm: List[float] = []

    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)

        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)

        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)

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

    di_values: List[float] = []
    for i in range(len(atr)):
        if atr[i] == 0:
            di_values.append(50.0)
            continue
        pdi = (plus_di_raw[i] / atr[i]) * 100 if i < len(plus_di_raw) else 0
        mdi = (minus_di_raw[i] / atr[i]) * 100 if i < len(minus_di_raw) else 0
        dx = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) > 0 else 0
        di_values.append(dx)

    if len(di_values) < period:
        return 0.0
    adx_values = wilder_smooth(di_values, period)
    return adx_values[-1] if adx_values else 0.0


def _compute_atr(
    highs: List[float], lows: List[float], closes: List[float], period: int = 14
) -> float:
    """Average True Range."""
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


# ---------------------------------------------------------------------------
# Suitability Scorer
# ---------------------------------------------------------------------------


class SuitabilityScorer:
    """Score candidate pairs by movement quality for our trading style.

    Fetches 7 days of 1h OHLCV from Binance public API, computes
    per-pair metrics, and returns a ranked suitability report.
    """

    def __init__(self) -> None:
        self._cfg = load_config()
        self._log = logger
        # 7 days of 1h data = 168 candles, but we request 200 to be safe
        self._default_limit = 200

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score_universe(
        self,
        symbols: Optional[List[str]] = None,
    ) -> SuitabilityReport:
        """Score candidate pairs and produce a ranked suitability report.

        Args:
            symbols: Optional override list. Defaults to config's candidates.

        Returns:
            SuitabilityReport with per-pair scores and top picks.
        """
        if symbols is None:
            symbols = self._cfg.pair_universe_candidates

        if not symbols:
            return SuitabilityReport(
                scan_timestamp=datetime.now(timezone.utc).isoformat(),
                errors=["No candidate pairs configured"],
            )

        self._log.info(f"Scoring suitability for {len(symbols)} pairs...")

        results: List[PairSuitability] = []
        errors: List[str] = []

        # Fetch OHLCV + funding for all pairs concurrently
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:
            tasks = [self._score_single(session, sym) for sym in symbols]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for sym, outcome in zip(symbols, outcomes):
            if isinstance(outcome, BaseException):
                err_msg = f"{sym}: {outcome}"
                self._log.warning(err_msg)
                errors.append(err_msg)
            elif outcome is not None:
                results.append(outcome)

        # Sort by composite score descending
        results.sort(key=lambda r: r.composite_score, reverse=True)

        # Top picks = config's active_count
        top_n = self._cfg.pair_universe_active_count
        top_picks = [r.symbol for r in results[:top_n]]

        report = SuitabilityReport(
            pairs=results,
            top_picks=top_picks,
            scan_timestamp=datetime.now(timezone.utc).isoformat(),
            total_scored=len(results),
            errors=errors,
        )

        self._log.info(
            f"Suitability scan complete — {len(results)} scored, "
            f"top picks: {', '.join(top_picks)}"
        )
        return report

    async def _score_single(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
    ) -> Optional[PairSuitability]:
        """Score a single pair by fetching its data and computing metrics."""
        # 1. Fetch OHLCV
        ohlcv = await self._fetch_ohlcv(session, symbol)
        if not ohlcv or len(ohlcv) < 50:
            return None

        # 2. Extract columns
        highs = [float(k[2]) for k in ohlcv]
        lows = [float(k[3]) for k in ohlcv]
        closes = [float(k[4]) for k in ohlcv]
        volumes = [float(k[5]) for k in ohlcv]
        close_price = closes[-1]

        # 3. Compute metrics
        atr_value = _compute_atr(highs, lows, closes)
        atr_pct = (atr_value / close_price * 100) if close_price > 0 else 0.0

        adx = _compute_adx(highs, lows, closes)

        # Volume consistency (lower CV = more consistent)
        avg_vol = sum(volumes) / len(volumes) if volumes else 0
        vol_std = statistics.stdev(volumes) if len(volumes) > 1 else 0
        volume_cv = vol_std / avg_vol if avg_vol > 0 else 999.0

        avg_volume_usdt = avg_vol

        # Candle smoothness: ratio of body to total range
        body_ratios: List[float] = []
        for i in range(len(ohlcv)):
            o = float(ohlcv[i][1])  # open
            c = float(ohlcv[i][4])  # close
            h = float(ohlcv[i][2])  # high
            lv = float(ohlcv[i][3])  # low
            body = abs(c - o)
            rng = h - lv
            if rng > 0:
                body_ratios.append(body / rng)
        avg_body_ratio = sum(body_ratios) / len(body_ratios) if body_ratios else 0.0

        # 4. Fetch funding rate
        funding_annualised = await self._fetch_funding_rate(session, symbol)

        # 5. Compute individual scores (0.0–1.0)
        movement_score = self._score_movement(atr_pct)
        trend_score = self._score_trend(adx)
        volume_score = self._score_volume(volume_cv)
        funding_score = self._score_funding(funding_annualised)
        smoothness_score = self._score_smoothness(avg_body_ratio)

        # 6. Composite (weighted)
        composite = (
            movement_score * 0.30
            + trend_score * 0.15
            + volume_score * 0.20
            + funding_score * 0.10
            + smoothness_score * 0.25
        )

        # Determine quick regime label for reference
        regime_label = "ranging"
        if adx >= 25:
            regime_label = "trending"
        if atr_pct > 3.0:
            regime_label = "volatile"

        return PairSuitability(
            symbol=symbol,
            composite_score=round(composite, 4),
            movement_score=round(movement_score, 4),
            trend_score=round(trend_score, 4),
            volume_score=round(volume_score, 4),
            funding_score=round(funding_score, 4),
            smoothness_score=round(smoothness_score, 4),
            atr_pct=round(atr_pct, 4),
            adx=round(adx, 2),
            volume_cv=round(volume_cv, 4),
            funding_rate_annualised=round(funding_annualised, 6),
            avg_body_ratio=round(avg_body_ratio, 4),
            avg_volume_usdt=round(avg_volume_usdt, 2),
            close_price=round(close_price, 2),
            regime_label=regime_label,
        )

    # ------------------------------------------------------------------
    # Scoring functions (each maps raw metric → 0.0–1.0)
    # ------------------------------------------------------------------

    def _score_movement(self, atr_pct: float) -> float:
        """Score movement quality based on ATR%/price.

        Ideal: 0.5% – 3.0% per candle. Below 0.3% is dead, above 5% is too wild.
        """
        if atr_pct <= 0.0:
            return 0.0

        if atr_pct < ATR_PCT_IDEAL_MIN:
            # Below ideal — linear penalty
            return max(0.0, atr_pct / ATR_PCT_IDEAL_MIN * 0.7)

        if atr_pct <= ATR_PCT_IDEAL_MAX:
            # In sweet spot — score 0.7–1.0
            t = (atr_pct - ATR_PCT_IDEAL_MIN) / (ATR_PCT_IDEAL_MAX - ATR_PCT_IDEAL_MIN)
            return 0.7 + t * 0.3

        # Above ideal — gentle decay
        excess = atr_pct / ATR_PCT_IDEAL_MAX
        return max(0.0, 1.0 - (excess - 1.0) * 0.3)

    def _score_trend(self, adx: float) -> float:
        """Score trend clarity.

        ADX below 15 → too choppy. 25+ → good clarity.
        """
        if adx <= 0:
            return 0.0
        if adx < ADX_MIN:
            return max(0.0, adx / ADX_MIN * 0.5)
        if adx <= ADX_IDEAL:
            t = (adx - ADX_MIN) / (ADX_IDEAL - ADX_MIN)
            return 0.5 + t * 0.4
        # Above ideal → still fine, slow decay
        return min(1.0, 0.9 + (adx - ADX_IDEAL) * 0.005)

    def _score_volume(self, cv: float) -> float:
        """Score volume consistency (lower CV = better).

        CV < 0.3 → very consistent. CV > 1.0 → unreliable.
        """
        if cv <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - cv * 0.8))

    def _score_funding(self, annualised: float) -> float:
        """Score funding rate cost (lower = better).

        Below 1% annualised → essentially free.
        Above 10% → expensive.
        """
        if annualised <= 0:
            return 1.0  # Negative funding = getting paid = best
        if annualised <= FUNDING_IDEAL_MAX:
            return 1.0
        # Linear decay from 1% to 10%
        return max(0.0, 1.0 - (annualised - FUNDING_IDEAL_MAX) / 0.09)

    def _score_smoothness(self, avg_body_ratio: float) -> float:
        """Score candle smoothness.

        Body/range ratio > 0.5 → nice clean candles (small wicks).
        Body/range < 0.2 → wicky/spiky (noisy).
        """
        if avg_body_ratio <= 0:
            return 0.0
        return min(1.0, avg_body_ratio * 1.5)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def _fetch_ohlcv(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
    ) -> Optional[List[List[Any]]]:
        """Fetch 1h OHLCV from Binance public API."""
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval=1h&limit={self._default_limit}"
        )
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    self._log.debug(f"Binance returned {resp.status} for {symbol}")
                    return None
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self._log.debug(f"Failed to fetch OHLCV for {symbol}: {exc}")
            return None

    async def _fetch_funding_rate(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
    ) -> float:
        """Fetch latest funding rate from Binance Futures and annualise it.

        Returns annualised funding rate as a decimal (e.g. 0.05 = 5%).
        Negative = you receive funding (good for us).
        """
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return 0.0
                data = await resp.json()
                if not data or not isinstance(data, list):
                    return 0.0
                rate = float(data[0].get("fundingRate", 0))
                # Funding is every 8h on Binance Futures
                # Annualised = rate * 3 * 365
                annualised = rate * 3 * 365
                return annualised
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError, ValueError):
            return 0.0
