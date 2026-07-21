"""Aurora Trader — Opportunity Spotter.

Multi-timeframe confluence detector that identifies high-probability
entry opportunities across all active trading pairs.

Scans 15m, 1h, and 4h candles and looks for:

  - BB touch + RSI extreme → mean reversion setup
  - Trend alignment across TFs → trend follow setup
  - Price near key EMA levels → bounce/break setup
  - "Brewing" setups — getting close but not yet triggered

Each opportunity gets a confidence score (0-100) and a clear signal
direction (LONG / SHORT / BREWING_LONG / BREWING_SHORT).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import aiosqlite

from shared.config import load_config
from shared.logger import get_logger

logger = get_logger("learning_server.opportunity_spotter")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEFRAMES = {
    "15m": {"binance": "15m", "candles": 96,      "label": "15min"},
    "1h":  {"binance": "1h",  "candles": 48,      "label": "1 hour"},
    "4h":  {"binance": "4h",  "candles": 30,      "label": "4 hour"},
}

# RSI thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# BB thresholds (how close price is to a band, as fraction of band distance)
BB_TOUCH_THRESHOLD = 0.15   # Within 15% of band = "touching"
BB_BREWING_THRESHOLD = 0.30 # Within 30% of band = "brewing"

# EMA confluence thresholds
EMA_200_CONFLUENCE = 0.02   # Price within 2% of EMA200
EMA_50_CONFLUENCE = 0.015   # Price within 1.5% of EMA50

# ADX thresholds
ADX_TREND_MIN = 20

# Minimum confidence to show on dashboard
MIN_SHOW_CONFIDENCE = 30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: str
    direction: str                    # long / short / neutral
    confidence: float                 # 0.0 – 1.0
    rsi: float
    bb_position: str                  # lower / middle / upper / between
    bb_distance_pct: float            # 0=on band, 1=midpoint
    ema_position: str                 # above_200 / below_200 / near_200 / etc
    adx: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class Opportunity:
    """A complete opportunity with multi-TF confluence."""
    symbol: str
    direction: str                    # LONG / SHORT / BREWING_LONG / BREWING_SHORT
    confidence: int                   # 0-100
    primary_timeframe: str            # Best TF for entry
    entry_notes: str                  # Human-readable summary
    price: float
    timeframes: Dict[str, TimeframeSignal] = field(default_factory=dict)
    brewing: bool = False             # Not yet triggered but close


@dataclass
class ScanResult:
    """Full scan output."""
    timestamp: str = ""
    opportunities: List[Opportunity] = field(default_factory=list)
    hot_list: List[str] = field(default_factory=list)   # Confidence >= 70
    watch_list: List[str] = field(default_factory=list)  # Confidence >= 40
    total_scanned: int = 0
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Indicator helpers (standalone pure functions)
# ---------------------------------------------------------------------------


def _compute_rsi(closes: List[float], period: int = 14) -> float:
    """Relative Strength Index."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses += abs(diff)
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_bb(
    closes: List[float], period: int = 20, std_dev: float = 2.0
) -> Tuple[float, float, float, float]:
    """Bollinger Bands — returns (upper, middle, lower, width_pct)."""
    if len(closes) < period:
        return (0.0, 0.0, 0.0, 0.0)
    recent = closes[-period:]
    mean = sum(recent) / len(recent)
    variance = sum((c - mean) ** 2 for c in recent) / len(recent)
    std = math.sqrt(variance) if variance > 0 else 0.0
    upper = mean + std_dev * std
    lower = mean - std_dev * std
    width = (upper - lower) / mean if mean > 0 else 0.0
    return (upper, mean, lower, width)


def _compute_ema(values: List[float], period: int) -> Optional[float]:
    """Last EMA value."""
    if len(values) < period:
        return None
    multiplier = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = (v - ema) * multiplier + ema
    return ema


def _compute_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Average Directional Index."""
    if len(closes) < period + 2:
        return 0.0
    tr_list: List[float] = []
    plus_dm: List[float] = []
    minus_dm: List[float] = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append(tr)
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)

    def wilder(vals, p):
        if not vals:
            return []
        s = [sum(vals[:p]) / p]
        for v in vals[p:]:
            s.append((s[-1] * (p - 1) + v) / p)
        return s

    atr = wilder(tr_list, period)
    pdi = wilder(plus_dm, period)
    mdi = wilder(minus_dm, period)
    di_vals = []
    for i in range(len(atr)):
        p = (pdi[i] / atr[i]) * 100 if atr[i] > 0 else 0
        m = (mdi[i] / atr[i]) * 100 if atr[i] > 0 else 0
        dx = abs(p - m) / (p + m) * 100 if (p + m) > 0 else 0
        di_vals.append(dx)
    if len(di_vals) < period:
        return 0.0
    adx_vals = wilder(di_vals, period)
    return adx_vals[-1] if adx_vals else 0.0


# ---------------------------------------------------------------------------
# Opportunity Spotter
# ---------------------------------------------------------------------------


class OpportunitySpotter:
    """Scans active trading pairs for multi-TF entry opportunities."""

    def __init__(self, db_path: str = "data/trading.db") -> None:
        self._cfg = load_config()
        self._log = logger
        self._session: Optional[aiohttp.ClientSession] = None
        self._db_path = Path(db_path)
        # In-memory accuracy cache: {symbol: {"hits": N, "total": N, "hit_rate": float}}
        self._accuracy_cache: Dict[str, Dict[str, float]] = {}
        self._accuracy_cache_ts: float = 0
        self._accuracy_cache_ttl: float = 300  # 5 minutes

    async def _load_accuracy_stats(self) -> Dict[str, Dict[str, float]]:
        """Load past opportunity scans and compute per-symbol prediction accuracy.

        For each past scan (older than 1 hour), checks if price moved in the
        predicted direction by comparing with the latest available OHLCV data.
        Returns {symbol: {"hits": N, "total": N, "hit_rate": 0.0-1.0}}.
        """
        now = time.time()
        if now - self._accuracy_cache_ts < self._accuracy_cache_ttl and self._accuracy_cache:
            return self._accuracy_cache

        stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"hits": 0.0, "total": 0.0, "hit_rate": 0.5})

        try:
            db_path = str(self._db_path)
            if not self._db_path.is_absolute():
                db_path = str(Path.cwd() / self._db_path)

            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                # Load scans older than 1 hour (allows time for price movement)
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
                cursor = await db.execute(
                    """SELECT symbol, direction, confidence, price, scan_time, raw_json
                       FROM opportunity_scans
                       WHERE scan_time < ?
                       ORDER BY scan_time DESC
                       LIMIT 500""",
                    (cutoff,),
                )
                rows = await cursor.fetchall()

            # Group by symbol and check each prediction
            # 🚀 Deduplicate price fetches — one Binance API call per unique symbol
            price_cache: Dict[str, Optional[float]] = {}
            for row in rows:
                sym = row["symbol"]
                if sym not in price_cache:
                    price_cache[sym] = await self._fetch_current_price(sym)

            for row in rows:
                sym = row["symbol"]
                direction = row["direction"]
                scan_price = float(row["price"] or 0)
                if scan_price <= 0:
                    continue

                # Determine if this was a LONG or SHORT prediction (not brewing)
                is_long = "LONG" in (direction or "")
                is_short = "SHORT" in (direction or "")
                if not is_long and not is_short:
                    continue

                # Use cached price
                current_price = price_cache.get(sym)
                if current_price is None or current_price <= 0:
                    continue

                # Direction check: LONG = price went up, SHORT = price went down
                if is_long and current_price > scan_price * 1.001:  # 0.1% threshold
                    stats[sym]["hits"] += 1
                    stats[sym]["total"] += 1
                elif is_short and current_price < scan_price * 0.999:
                    stats[sym]["hits"] += 1
                    stats[sym]["total"] += 1
                elif abs(current_price - scan_price) / scan_price > 0.002:
                    # Price moved significantly but opposite to prediction → miss
                    stats[sym]["total"] += 1
                # else: price didn't move enough to judge → skip

            # Compute hit rates with Bayesian smoothing (start at 50%)
            for sym, s in stats.items():
                total = s["total"]
                hits = s["hits"]
                if total > 0:
                    # Beta prior: (hits + 1) / (total + 2) to avoid 0/1 extremes
                    s["hit_rate"] = round((hits + 1) / (total + 2), 4)
                else:
                    s["hit_rate"] = 0.5

            # Fallback: if no data, use neutral
            if not stats:
                self._log.debug("No past scans available for accuracy calibration")
                self._accuracy_cache = {}
                self._accuracy_cache_ts = now
                return {}

            self._accuracy_cache = dict(stats)
            self._accuracy_cache_ts = now
            self._log.debug(f"Accuracy stats loaded for {len(stats)} symbols")

        except Exception as exc:
            self._log.debug(f"Could not load accuracy stats: {exc}")
            return {}

        return self._accuracy_cache

    async def _fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch latest price for a symbol from Binance."""
        try:
            session = await self._get_session()
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get("price", 0))
        except Exception:
            pass
        return None

    def _calibrate_confidence(self, symbol: str, raw_confidence: int, stats: Dict[str, Dict[str, float]]) -> int:
        """Adjust confidence based on historical prediction accuracy for this symbol.

        Blends raw technical analysis confidence with historical hit rate:
          calibrated = raw * (1 - accuracy_weight) + (hit_rate * 100) * accuracy_weight

        The weight increases as we have more data (max 30% weight at 20+ predictions).
        """
        sym_stats = stats.get(symbol, {})
        total = sym_stats.get("total", 0)
        hit_rate = sym_stats.get("hit_rate", 0.5)

        if total < 3:
            return raw_confidence  # Not enough data to calibrate

        # Weight grows with data: 10% at 3 predictions → 30% at 20+
        accuracy_weight = min(0.30, 0.10 + total * 0.01)

        calibrated = raw_confidence * (1 - accuracy_weight) + (hit_rate * 100) * accuracy_weight
        calibrated = max(1, min(99, int(calibrated)))

        if abs(calibrated - raw_confidence) > 2:
            self._log.info(
                f"Calibrated {symbol}: {raw_confidence}% → {calibrated}% "
                f"(hit_rate={hit_rate:.2f}, n={int(total)})"
            )

        return calibrated

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def scan(self, symbols: Optional[List[str]] = None) -> ScanResult:
        """Scan active pairs for opportunities across multiple TFs.

        After computing raw technical confidence, calibrates scores based on
        historical prediction accuracy per symbol (loaded from past opportunity_scans).
        """
        if symbols is None:
            symbols = self._cfg.data.get("trading_server", {}).get("symbols", [])

        if not symbols:
            return ScanResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                errors=["No active symbols configured"],
            )

        self._log.info(f"Scanning {len(symbols)} pairs for opportunities...")

        # Load historical accuracy stats for confidence calibration
        try:
            accuracy_stats = await asyncio.wait_for(
                self._load_accuracy_stats(), timeout=45
            )
        except (asyncio.TimeoutError, Exception) as exc:
            self._log.warning(f"Accuracy calibration skipped: {exc}")
            accuracy_stats = {}
        if accuracy_stats:
            syms_with_data = [s for s in symbols if s in accuracy_stats and accuracy_stats[s]["total"] >= 3]
            if syms_with_data:
                self._log.info(f"Accuracy data available for {len(syms_with_data)} symbols")

        session = await self._get_session()
        opportunities: List[Opportunity] = []
        errors: List[str] = []

        for sym in symbols:
            try:
                opp = await self._scan_pair(session, sym)
                if opp:
                    # Calibrate confidence using historical accuracy
                    calibrated = self._calibrate_confidence(sym, opp.confidence, accuracy_stats)
                    opp.confidence = calibrated
                    opportunities.append(opp)
            except Exception as exc:
                err = f"{sym}: {exc}"
                self._log.debug(err)
                errors.append(err)

        # Sort by confidence descending
        opportunities.sort(key=lambda o: o.confidence, reverse=True)

        hot_list = [o.symbol for o in opportunities if o.confidence >= 70]
        watch_list = [o.symbol for o in opportunities if 40 <= o.confidence < 70 and not o.brewing]

        return ScanResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            opportunities=opportunities,
            hot_list=hot_list,
            watch_list=watch_list,
            total_scanned=len(symbols),
            errors=errors,
        )

    async def _scan_pair(
        self, session: aiohttp.ClientSession, symbol: str
    ) -> Optional[Opportunity]:
        """Scan a single pair across all timeframes."""
        tf_signals: Dict[str, TimeframeSignal] = {}
        all_reasons: List[str] = []
        best_direction = "neutral"
        best_confidence = 0.0
        best_tf = "1h"
        current_price = 0.0
        brewing = False

        for tf_name, tf_cfg in TIMEFRAMES.items():
            try:
                signal, price = await self._analyze_timeframe(
                    session, symbol, tf_name, tf_cfg
                )
                if signal is None:
                    continue

                tf_signals[tf_name] = signal
                current_price = price or current_price

                # Track best signal
                if signal.confidence > best_confidence:
                    best_confidence = signal.confidence
                    best_direction = signal.direction
                    best_tf = tf_name
                    all_reasons = signal.reasons

                # Brewing detection
                if signal.direction in ("long", "short") and signal.confidence < 0.5:
                    brewing = True

            except Exception as exc:
                self._log.debug(f"{symbol}@{tf_name}: {exc}")
                continue

        if not tf_signals:
            return None

        # Compute overall confidence from multi-TF agreement
        directions = [s.direction for s in tf_signals.values() if s.direction != "neutral"]
        if not directions:
            return None

        # Check for multi-TF agreement
        long_count = directions.count("long")
        short_count = directions.count("short")
        total_directional = len(directions)

        if long_count > short_count:
            overall_dir = "LONG"
            agreement = long_count / total_directional if total_directional > 0 else 0
        elif short_count > long_count:
            overall_dir = "SHORT"
            agreement = short_count / total_directional if total_directional > 0 else 0
        else:
            overall_dir = "neutral"
            agreement = 0

        if overall_dir == "neutral" or agreement < 0.3:
            return None

        # Boost confidence from multi-TF agreement
        avg_tf_conf = sum(
            s.confidence for s in tf_signals.values() if s.direction != "neutral"
        ) / max(total_directional, 1)

        overall_confidence = avg_tf_conf * 0.6 + agreement * 0.4
        overall_confidence = min(1.0, overall_confidence)

        # Brewing = close but not triggered on primary TF
        is_brewing = brewing or overall_confidence < 0.45

        # Build entry notes
        tf_detail = ", ".join(
            f"{tf}: {s.direction.upper()} ({s.reasons[0] if s.reasons else 'neutral'})"
            for tf, s in sorted(tf_signals.items())
        )

        dir_label = f"{'BREWING_' if is_brewing else ''}{overall_dir}"

        return Opportunity(
            symbol=symbol,
            direction=dir_label,
            confidence=min(99, int(overall_confidence * 100)),
            primary_timeframe=best_tf,
            entry_notes=tf_detail,
            price=current_price,
            timeframes=tf_signals,
            brewing=is_brewing,
        )

    async def _analyze_timeframe(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        tf_name: str,
        tf_cfg: Dict[str, Any],
    ) -> Tuple[Optional[TimeframeSignal], Optional[float]]:
        """Analyze a single timeframe for a symbol."""
        # Fetch OHLCV
        binance_interval = tf_cfg["binance"]
        limit = tf_cfg["candles"]
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={binance_interval}&limit={limit}"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return None, None
                raw = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, asyncio.CancelledError):
            return None, None

        if not raw or len(raw) < 30:
            return None, None

        closes = [float(k[4]) for k in raw]
        highs = [float(k[2]) for k in raw]
        lows = [float(k[3]) for k in raw]
        price = closes[-1]

        # Compute indicators
        rsi = _compute_rsi(closes)
        bb_upper, bb_mid, bb_lower, bb_width = _compute_bb(closes)
        adx = _compute_adx(highs, lows, closes)
        ema_50 = _compute_ema(closes, 50)
        ema_200 = _compute_ema(closes, 200) if len(closes) >= 200 else None

        # Determine BB position
        bb_range = bb_upper - bb_lower
        if bb_range <= 0:
            bb_position = "middle"
            bb_dist = 0.5
        else:
            # How far is price from lower band? 0 = at lower, 1 = at upper
            bb_dist = (price - bb_lower) / bb_range
            if bb_dist <= BB_TOUCH_THRESHOLD:
                bb_position = "lower_band"
            elif bb_dist >= (1 - BB_TOUCH_THRESHOLD):
                bb_position = "upper_band"
            elif bb_dist <= BB_BREWING_THRESHOLD:
                bb_position = "near_lower"
            elif bb_dist >= (1 - BB_BREWING_THRESHOLD):
                bb_position = "near_upper"
            else:
                bb_position = "between"

        # Distance from nearest band (0 = on band)
        bb_dist_from_band = min(bb_dist, 1 - bb_dist) if 0 <= bb_dist <= 1 else 0.5

        # Determine EMA position
        ema_position = "unknown"
        if ema_50:
            ema_dist = (price - ema_50) / ema_50
            if abs(ema_dist) < EMA_50_CONFLUENCE:
                ema_position = "near_50"
            elif ema_dist > 0:
                ema_position = "above_50"
            else:
                ema_position = "below_50"
        if ema_200:
            ema200_dist = (price - ema_200) / ema_200
            if abs(ema200_dist) < EMA_200_CONFLUENCE:
                ema_position = "near_200"
            elif ema200_dist < 0:
                ema_position = "below_200"

        # --- Signal detection ---

        direction = "neutral"
        confidence = 0.0
        reasons: List[str] = []

        # Mean reversion signal: BB touch + RSI extreme + low ADX
        if bb_position in ("lower_band", "near_lower") and rsi < RSI_OVERSOLD + 5:
            direction = "long"
            confidence = max(0.3, min(1.0, (RSI_OVERSOLD + 5 - rsi) / 20 + bb_dist_from_band))
            reasons.append(f"BB lower touch + RSI {rsi:.0f}")
            if adx < ADX_TREND_MIN:
                confidence = min(1.0, confidence * 1.2)
                reasons.append("low ADX (mean reversion ideal)")

        elif bb_position in ("upper_band", "near_upper") and rsi > RSI_OVERBOUGHT - 5:
            direction = "short"
            confidence = max(0.3, min(1.0, (rsi - RSI_OVERBOUGHT + 5) / 20 + bb_dist_from_band))
            reasons.append(f"BB upper touch + RSI {rsi:.0f}")
            if adx < ADX_TREND_MIN:
                confidence = min(1.0, confidence * 1.2)
                reasons.append("low ADX (mean reversion ideal)")

        # Trend follow signal: EMA bounce + ADX trending
        if adx >= ADX_TREND_MIN:
            if ema_position == "near_50" and rsi > 40 and rsi < 60:
                if direction == "neutral" or confidence < 0.5:
                    direction = "long"
                    confidence = max(0.25, min(0.7, adx / 50))
                    reasons.append(f"EMA50 bounce + ADX {adx:.0f}")
            elif ema_position == "near_200" and rsi > 40 and rsi < 60:
                if direction == "neutral" or confidence < 0.5:
                    direction = "long"
                    confidence = max(0.3, min(0.8, adx / 40))
                    reasons.append(f"EMA200 bounce + ADX {adx:.0f}")

        # Multi-TF alignment hint (from name alone)
        if adx >= ADX_TREND_MIN and rsi > 50 and bb_position in ("between", "near_upper"):
            if direction == "neutral":
                direction = "long"
                confidence = 0.2
                reasons.append(f"Mild bullish: ADX {adx:.0f}, RSI {rsi:.0f}")

        elif adx >= ADX_TREND_MIN and rsi < 50 and bb_position in ("between", "near_lower"):
            if direction == "neutral":
                direction = "short"
                confidence = 0.2
                reasons.append(f"Mild bearish: ADX {adx:.0f}, RSI {rsi:.0f}")

        # Brewing detection — getting close but not triggered
        if confidence < 0.4 and rsi > RSI_OVERSOLD and rsi < RSI_OVERSOLD + 10:
            if bb_position == "near_lower":
                if direction == "neutral":
                    direction = "long"
                    confidence = 0.2
                    reasons.append(f"Brewing: RSI {rsi:.0f} approaching oversold, near BB lower")
                    brewing = True

        if confidence < 0.4 and rsi < RSI_OVERBOUGHT and rsi > RSI_OVERBOUGHT - 10:
            if bb_position == "near_upper":
                if direction == "neutral":
                    direction = "short"
                    confidence = 0.2
                    reasons.append(f"Brewing: RSI {rsi:.0f} approaching overbought, near BB upper")

        if confidence < MIN_SHOW_CONFIDENCE / 100:
            return None, price

        signal = TimeframeSignal(
            timeframe=tf_name,
            direction=direction,
            confidence=round(confidence, 4),
            rsi=round(rsi, 2),
            bb_position=bb_position,
            bb_distance_pct=round(bb_dist_from_band, 4),
            ema_position=ema_position,
            adx=round(adx, 2),
            reasons=reasons,
        )

        return signal, price

    async def close(self) -> None:
        """Clean up HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
