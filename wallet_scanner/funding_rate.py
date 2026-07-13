"""
Aurora Trader — Wallet Scanner: Funding Rate & Open Interest Analysis.

Fetches perpetual futures funding rates and open interest data from
Binance Futures (free public endpoints) on a configurable schedule.

Interpretation:
    Negative funding  → shorts pay longs → bearish sentiment,
                        but also short-squeeze potential.
    Positive funding  → longs pay shorts → bullish sentiment,
                        but also long-squeeze potential.
    OI divergence     → rising OI + falling price = bearish divergence.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from shared.config import Config, load_config
from shared.logger import get_logger
from shared.models import SignalDirection

logger = get_logger("wallet_scanner.funding_rate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_POLL_INTERVAL_SEC = 8 * 3600  # every 8 hours (aligns with funding settlement)
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Binance Futures REST API (public, no auth needed)
_BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# Funding rate endpoint (returns last 100 8h funding rates per symbol)
_FUNDING_RATE_URL = f"{_BINANCE_FUTURES_BASE}/fapi/v1/fundingRate"
# Open interest endpoint
_OI_URL = f"{_BINANCE_FUTURES_BASE}/fapi/v1/openInterest"
# Open interest stats (historical)
_OI_STATS_URL = f"{_BINANCE_FUTURES_BASE}/futures/data/openInterestHist"
# Top trader long/short ratio (optional, from Binance data)
_LS_RATIO_URL = f"{_BINANCE_FUTURES_BASE}/futures/data/globalLongShortAccountRatio"

# Thresholds for signal generation
_FUNDING_RATE_HIGH = 0.0005  # 0.05% — extreme funding (squeeze territory)
_FUNDING_RATE_LOW = -0.0005  # -0.05%
_OI_CHANGE_THRESHOLD_PCT = 10.0  # >10% OI change = divergence risk


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FundingRecord:
    """A single funding rate data point."""

    symbol: str
    funding_rate: float  # raw rate (e.g. 0.0001 = 0.01%)
    funding_time: float  # unix seconds
    mark_price: float = 0.0
    source: str = "binance_futures"


@dataclass
class OIRecord:
    """Open interest snapshot for a symbol."""

    symbol: str
    open_interest: float  # in USDT equivalent
    timestamp: float


@dataclass
class FundingState:
    """Running state for a single symbol's funding analysis."""

    symbol: str
    current_rate: float = 0.0
    last_rate: float = 0.0
    avg_rate_24h: float = 0.0
    current_oi: float = 0.0
    previous_oi: float = 0.0
    oi_change_pct: float = 0.0
    rates_history: List[float] = field(default_factory=list)
    last_updated: float = 0.0

    @property
    def is_extreme_positive(self) -> bool:
        return self.current_rate > _FUNDING_RATE_HIGH

    @property
    def is_extreme_negative(self) -> bool:
        return self.current_rate < _FUNDING_RATE_LOW

    @property
    def is_oi_diverging(self) -> bool:
        """OI rising while rate suggests opposite direction = divergence."""
        return abs(self.oi_change_pct) > _OI_CHANGE_THRESHOLD_PCT


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Simple interval-based rate limiter."""

    def __init__(self, interval_sec: float = 1.0) -> None:
        self._interval = interval_sec
        self._last = 0.0

    async def acquire(self) -> None:
        now = time.monotonic()
        wait = self._last + self._interval - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Funding Rate Monitor
# ---------------------------------------------------------------------------


class FundingRateMonitor:
    """Monitors perpetual futures funding rates and open interest.

    Polls Binance Futures public endpoints every 8 hours.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        symbols: Optional[List[str]] = None,
    ) -> None:
        self._cfg = config or load_config()
        self._symbols = symbols or _DEFAULT_SYMBOLS
        self._log = logger

        # State per symbol
        self._states: Dict[str, FundingState] = {
            sym: FundingState(symbol=sym) for sym in self._symbols
        }

        # Recent records
        self._funding_records: List[FundingRecord] = []
        self._oi_records: List[OIRecord] = []
        self._max_records = 200

        # Rate limiter (Binance Futures: 1200 req/min, we're very polite)
        self._limiter = _RateLimiter(interval_sec=2.0)

        # Session
        self._session: Optional[aiohttp.ClientSession] = None

        # Background task
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self._task = asyncio.create_task(self._poll_loop())
        self._log.info(
            f"FundingRateMonitor started — symbols={self._symbols}"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._session:
            await self._session.close()
            self._session = None
        self._log.info("FundingRateMonitor stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self, symbol: str) -> Optional[FundingState]:
        return self._states.get(symbol)

    def get_all_states(self) -> Dict[str, FundingState]:
        return dict(self._states)

    def get_recent_funding(
        self, limit: int = 20
    ) -> List[FundingRecord]:
        return self._funding_records[-limit:]

    def get_signals(self) -> List[Dict[str, Any]]:
        """Evaluate funding rate data and produce signal dicts."""
        signals: List[Dict[str, Any]] = []

        for sym, state in self._states.items():
            # Extreme positive funding → bullish sentiment, long-squeeze risk
            if state.is_extreme_positive:
                signals.append({
                    "symbol": sym,
                    "direction": SignalDirection.LONG.value,
                    "type": "funding_positive",
                    "confidence": min(
                        abs(state.current_rate) / 0.002, 1.0
                    ),
                    "value": state.current_rate,
                    "reason": (
                        f"Funding rate {state.current_rate:+.6f} "
                        f"(avg 24h: {state.avg_rate_24h:+.6f}) — "
                        f"bullish sentiment, long-squeeze potential"
                    ),
                })

            # Extreme negative funding → bearish sentiment, short-squeeze potential
            if state.is_extreme_negative:
                signals.append({
                    "symbol": sym,
                    "direction": SignalDirection.SHORT.value,
                    "type": "funding_negative",
                    "confidence": min(
                        abs(state.current_rate) / 0.002, 1.0
                    ),
                    "value": state.current_rate,
                    "reason": (
                        f"Funding rate {state.current_rate:+.6f} "
                        f"(avg 24h: {state.avg_rate_24h:+.6f}) — "
                        f"bearish sentiment, short-squeeze potential"
                    ),
                })

            # OI divergence
            if state.is_oi_diverging:
                # Rising OI + negative funding = bearish divergence
                # Rising OI + positive funding = bullish continuation
                if state.oi_change_pct > 0 and state.current_rate < 0:
                    signals.append({
                        "symbol": sym,
                        "direction": SignalDirection.SHORT.value,
                        "type": "oi_divergence",
                        "confidence": min(
                            abs(state.oi_change_pct) / 50.0, 1.0
                        ),
                        "value": state.oi_change_pct,
                        "reason": (
                            f"OI rising {state.oi_change_pct:+.1f}% with "
                            f"negative funding {state.current_rate:+.6f} "
                            f"— bearish divergence"
                        ),
                    })
                elif state.oi_change_pct < 0 and state.current_rate > 0:
                    signals.append({
                        "symbol": sym,
                        "direction": SignalDirection.LONG.value,
                        "type": "oi_divergence",
                        "confidence": min(
                            abs(state.oi_change_pct) / 50.0, 1.0
                        ),
                        "value": state.oi_change_pct,
                        "reason": (
                            f"OI falling {state.oi_change_pct:+.1f}% with "
                            f"positive funding {state.current_rate:+.6f} "
                            f"— bullish signal"
                        ),
                    })

        return signals

    # ------------------------------------------------------------------
    # Background Poll Loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Poll funding rate data every 8 hours."""
        while self._running:
            try:
                await self._poll_funding_data()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(
                    f"Funding rate poll error: {exc}", exc_info=True
                )
            for _ in range(_POLL_INTERVAL_SEC // 30):
                if not self._running:
                    return
                await asyncio.sleep(30)

    async def _poll_funding_data(self) -> None:
        """Fetch funding rates and open interest from Binance Futures."""
        self._log.debug("Polling funding rate data...")

        tasks = [
            self._fetch_funding_rates(),
            self._fetch_open_interest(),
            self._fetch_oi_history(),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Update derived state
        now = time.time()
        for sym in self._symbols:
            state = self._states[sym]
            state.last_updated = now
            if state.rates_history:
                state.avg_rate_24h = sum(state.rates_history) / len(state.rates_history)

        self._log.debug("Funding rate poll complete")

    async def _fetch_funding_rates(self) -> None:
        """Fetch latest funding rates for all tracked symbols."""
        if not self._session:
            return
        try:
            await self._limiter.acquire()

            params = {"limit": 100}  # last 100 funding periods
            async with self._session.get(
                _FUNDING_RATE_URL, params=params
            ) as resp:
                if resp.status != 200:
                    self._log.warning(
                        f"Funding rate fetch returned {resp.status}"
                    )
                    return

                data = await resp.json()
                if not isinstance(data, list):
                    return

                # Group by symbol
                by_symbol: Dict[str, List[Dict[str, Any]]] = {}
                for entry in data:
                    sym = entry.get("symbol", "")
                    if sym in self._symbols:
                        by_symbol.setdefault(sym, []).append(entry)

                now = time.time()
                for sym in self._symbols:
                    entries = by_symbol.get(sym, [])
                    if not entries:
                        continue

                    state = self._states[sym]

                    # Latest rate
                    latest = entries[-1]
                    state.last_rate = state.current_rate
                    state.current_rate = float(latest.get("fundingRate", 0))

                    # Collect history
                    rates = [
                        float(e.get("fundingRate", 0))
                        for e in entries
                        if "fundingRate" in e
                    ]
                    state.rates_history = rates[-100:]  # keep max 100

                    # Record
                    state.last_updated = now
                    rec = FundingRecord(
                        symbol=sym,
                        funding_rate=state.current_rate,
                        funding_time=float(
                            latest.get("fundingTime", now * 1000)
                        )
                        / 1000.0,
                        mark_price=float(
                            latest.get("markPrice", 0)
                        ),
                    )
                    self._funding_records.append(rec)
                    if len(self._funding_records) > self._max_records:
                        self._funding_records.pop(0)

                    self._log.debug(
                        f"{sym} funding rate: {state.current_rate:+.6f} "
                        f"(last: {state.last_rate:+.6f})"
                    )

        except asyncio.TimeoutError:
            self._log.warning("Funding rate fetch timed out")
        except Exception as exc:
            self._log.warning(f"Funding rate error: {exc}")

    async def _fetch_open_interest(self) -> None:
        """Fetch current open interest for tracked symbols."""
        if not self._session:
            return
        try:
            await self._limiter.acquire()

            async with self._session.get(_OI_URL) as resp:
                if resp.status != 200:
                    self._log.warning(
                        f"OI fetch returned {resp.status}"
                    )
                    return

                data = await resp.json()
                if not isinstance(data, list):
                    return

                now = time.time()
                for entry in data:
                    sym = entry.get("symbol", "")
                    if sym not in self._states:
                        continue

                    oi = float(entry.get("openInterest", "0"))
                    state = self._states[sym]
                    state.previous_oi = state.current_oi
                    state.current_oi = oi

                    if state.previous_oi > 0 and state.current_oi > 0:
                        state.oi_change_pct = (
                            (state.current_oi - state.previous_oi)
                            / state.previous_oi
                            * 100.0
                        )

                    state.last_updated = now

                    rec = OIRecord(
                        symbol=sym,
                        open_interest=oi,
                        timestamp=now,
                    )
                    self._oi_records.append(rec)
                    if len(self._oi_records) > self._max_records:
                        self._oi_records.pop(0)

                    self._log.debug(
                        f"{sym} OI: {oi:.2f} USDT "
                        f"(change: {state.oi_change_pct:+.1f}%)"
                    )

        except asyncio.TimeoutError:
            self._log.warning("OI fetch timed out")
        except Exception as exc:
            self._log.warning(f"OI fetch error: {exc}")

    async def _fetch_oi_history(self) -> None:
        """Fetch historical OI stats for divergence detection.

        Uses Binance Futures open interest histogram endpoint
        (free, public).
        """
        if not self._session:
            return
        try:
            for sym in self._symbols:
                await self._limiter.acquire()

                params = {
                    "symbol": sym,
                    "period": "1h",
                    "limit": 24,  # last 24 hours
                }
                async with self._session.get(
                    _OI_STATS_URL, params=params
                ) as resp:
                    if resp.status != 200:
                        continue

                    data = await resp.json()
                    if not isinstance(data, list) or len(data) < 2:
                        continue

                    # Calculate average OI over the period
                    oi_values = [
                        float(e.get("sumOpenInterest", 0))
                        for e in data
                    ]
                    if oi_values:
                        avg_oi = sum(oi_values) / len(oi_values)
                        current_oi = oi_values[-1]
                        change = (
                            (current_oi - avg_oi) / avg_oi * 100.0
                            if avg_oi > 0
                            else 0.0
                        )

                        state = self._states[sym]
                        state.oi_change_pct = change
                        self._log.debug(
                            f"{sym} OI history: avg={avg_oi:.0f}, "
                            f"curr={current_oi:.0f} "
                            f"(change {change:+.1f}%)"
                        )

        except asyncio.TimeoutError:
            self._log.warning("OI history fetch timed out")
        except Exception as exc:
            self._log.warning(f"OI history error: {exc}")
