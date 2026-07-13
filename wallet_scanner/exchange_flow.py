"""
Aurora Trader — Wallet Scanner: Exchange Flow Monitor.

Tracks CEX (Centralised Exchange) net inflows and outflows for configured
trading symbols by polling public exchange data.  Flags large transactions
(whale-sized > $100k USDT equivalent) and maintains a rolling 24 h balance.

Interpretation:
    Large outflow from an exchange → bullish (moving to cold storage).
    Large inflow to an exchange   → bearish (preparing to sell).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import aiohttp

from shared.config import Config, load_config
from shared.logger import get_logger
from shared.models import SignalDirection

logger = get_logger("wallet_scanner.exchange_flow")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WHALE_THRESHOLD_USD_DEFAULT = 100_000.0
_ROLLING_WINDOW_HOURS = 24
_POLL_INTERVAL_SEC = 3600  # 1 hour (aligned with scanner main loop)

# Default symbols to track
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Free public endpoints used for exchange-flow estimation
# CoinGecko: no API key needed (rate-limited to ~10-30 req/min on free tier)
_COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Binance public endpoints
_BINANCE_BASE = "https://api.binance.com"
_BINANCE_TICKER_24HR = f"{_BINANCE_BASE}/api/v3/ticker/24hr"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FlowRecord:
    """A single exchange-flow event (inflow or outflow)."""

    symbol: str
    direction: str  # "inflow" | "outflow"
    amount_usd: float
    timestamp: float  # unix epoch seconds
    source: str  # which exchange / data source
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowBalance:
    """Rolling 24 h flow balance for a single symbol."""

    symbol: str
    net_flow_24h: float = 0.0  # positive = net inflow (bearish)
    total_inflow_24h: float = 0.0
    total_outflow_24h: float = 0.0
    large_tx_count_24h: int = 0
    last_updated: float = 0.0

    @property
    def is_bullish(self) -> bool:
        """Net outflow over 24 h → considered bullish."""
        return self.net_flow_24h < -self._threshold()

    @property
    def is_bearish(self) -> bool:
        """Net inflow over 24 h → considered bearish."""
        return self.net_flow_24h > self._threshold()

    @staticmethod
    def _threshold() -> float:
        return _WHALE_THRESHOLD_USD_DEFAULT * 2


# ---------------------------------------------------------------------------
# Rate-limit helper
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Sliding-window rate limiter for API calls."""

    def __init__(self, max_calls: int, window_sec: float = 60.0) -> None:
        self._max = max_calls
        self._window = window_sec
        self._timestamps: List[float] = []

    async def acquire(self) -> None:
        now = time.monotonic()
        # Prune outside window
        cutoff = now - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self._max:
            sleep_for = self._timestamps[0] + self._window - now
            if sleep_for > 0:
                logger.debug(
                    f"Rate limit reached, sleeping {sleep_for:.1f} s"
                )
                await asyncio.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


# ---------------------------------------------------------------------------
# Exchange Flow Monitor
# ---------------------------------------------------------------------------


class ExchangeFlowMonitor:
    """Monitors CEX exchange flows for configured symbols.

    Uses public Binance ticker data as a proxy for exchange-wide volume
    estimation, and CoinGecko exchange data for market-wide context.
    Maintains a rolling 24 h flow balance per symbol.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        symbols: Optional[List[str]] = None,
        whale_threshold_usd: float = _WHALE_THRESHOLD_USD_DEFAULT,
    ) -> None:
        self._cfg = config or load_config()
        self._symbols = symbols or _DEFAULT_SYMBOLS
        self._whale_threshold = whale_threshold_usd
        self._log = logger

        # Rolling balances keyed by symbol
        self._balances: Dict[str, FlowBalance] = {
            sym: FlowBalance(symbol=sym) for sym in self._symbols
        }

        # Recent flow events (ring buffer for signal generation)
        self._recent_flows: List[FlowRecord] = []
        self._max_flows = 500

        # Cached price data for USD calculations
        self._prices: Dict[str, float] = {}

        # Rate limiter: CoinGecko free tier ≈ 10 req/min
        self._cg_limiter = _RateLimiter(max_calls=8, window_sec=60.0)
        # Binance public API is generous, but be polite
        self._binance_limiter = _RateLimiter(max_calls=20, window_sec=60.0)

        # Session (lazily created)
        self._session: Optional[aiohttp.ClientSession] = None

        # Background task reference
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background polling loop."""
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self._task = asyncio.create_task(self._poll_loop())
        self._log.info(
            f"ExchangeFlowMonitor started — symbols={self._symbols}, "
            f"whale_threshold={self._whale_threshold:.0f} USD"
        )

    async def stop(self) -> None:
        """Stop the background loop and close resources."""
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
        self._log.info("ExchangeFlowMonitor stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_balance(self, symbol: str) -> Optional[FlowBalance]:
        """Return the rolling 24 h flow balance for *symbol*."""
        return self._balances.get(symbol)

    def get_all_balances(self) -> Dict[str, FlowBalance]:
        """Return all tracked balances."""
        return dict(self._balances)

    def get_recent_flows(
        self, limit: int = 50
    ) -> List[FlowRecord]:
        """Return the most recent flow records."""
        return self._recent_flows[-limit:]

    def get_signals(self) -> List[Dict[str, Any]]:
        """Evaluate current balances and produce signal dicts."""
        signals: List[Dict[str, Any]] = []
        for sym, bal in self._balances.items():
            if bal.is_bullish:
                signals.append({
                    "symbol": sym,
                    "direction": SignalDirection.LONG.value,
                    "type": "exchange_outflow",
                    "confidence": min(
                        abs(bal.net_flow_24h) / (_WHALE_THRESHOLD_USD_DEFAULT * 5),
                        1.0,
                    ),
                    "value": bal.net_flow_24h,
                    "reason": (
                        f"Net outflow ${bal.net_flow_24h:+.0f} over 24h "
                        f"({bal.large_tx_count_24h} large txs) — bullish"
                    ),
                })
            elif bal.is_bearish:
                signals.append({
                    "symbol": sym,
                    "direction": SignalDirection.SHORT.value,
                    "type": "exchange_inflow",
                    "confidence": min(
                        abs(bal.net_flow_24h) / (_WHALE_THRESHOLD_USD_DEFAULT * 5),
                        1.0,
                    ),
                    "value": bal.net_flow_24h,
                    "reason": (
                        f"Net inflow ${bal.net_flow_24h:+.0f} over 24h "
                        f"({bal.large_tx_count_24h} large txs) — bearish"
                    ),
                })
        return signals

    # ------------------------------------------------------------------
    # Background Poll Loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Poll exchange data at regular intervals."""
        while self._running:
            try:
                await self._poll_exchange_data()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(
                    f"Exchange flow poll error: {exc}", exc_info=True
                )
            # Sleep until next poll cycle
            for _ in range(_POLL_INTERVAL_SEC // 10):
                if not self._running:
                    return
                await asyncio.sleep(10)

    async def _poll_exchange_data(self) -> None:
        """Fetch ticker data and update flow estimates."""
        self._log.debug("Polling exchange flow data...")

        # 1. Fetch current prices (used for USD conversion)
        await self._fetch_prices()

        # 2. Fetch 24hr ticker stats from Binance
        await self._fetch_binance_ticker()

        # 3. Prune old flow records
        self._prune_old_records()

        self._log.debug("Exchange flow poll complete")

    async def _fetch_prices(self) -> None:
        """Fetch current prices from CoinGecko (free, no key)."""
        if not self._session:
            return
        try:
            await self._cg_limiter.acquire()

            # Map symbol -> CoinGecko coin id
            coin_ids = []
            for sym in self._symbols:
                base = sym.replace("USDT", "").lower()
                if base == "btc":
                    coin_ids.append("bitcoin")
                elif base == "eth":
                    coin_ids.append("ethereum")
                else:
                    coin_ids.append(base)

            url = f"{_COINGECKO_BASE}/simple/price"
            params = {
                "ids": ",".join(coin_ids),
                "vs_currencies": "usd",
            }
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for sym in self._symbols:
                        base = sym.replace("USDT", "").lower()
                        cg_id = {
                            "btc": "bitcoin",
                            "eth": "ethereum",
                        }.get(base, base)
                        if cg_id in data and "usd" in data[cg_id]:
                            self._prices[sym] = float(data[cg_id]["usd"])
                else:
                    self._log.warning(
                        f"CoinGecko price fetch returned {resp.status}"
                    )
        except asyncio.TimeoutError:
            self._log.warning("CoinGecko price fetch timed out")
        except Exception as exc:
            self._log.warning(f"CoinGecko price error: {exc}")

    async def _fetch_binance_ticker(self) -> None:
        """Fetch 24hr ticker from Binance and extract volume as a flow proxy.

        We use the quote volume (USDT volume) as a rough estimate of
        total exchange activity for each symbol.  A significant fraction
        of that volume implies exchange flow activity.  We estimate
        net flow as a small percentage of total volume with a random-ish
        sign based on price change direction (simplified model).
        """
        if not self._session:
            return
        try:
            await self._binance_limiter.acquire()

            params = {"symbols": str([s for s in self._symbols]).replace("'", '"')}
            async with self._session.get(
                _BINANCE_TICKER_24HR, params={"symbols": str([s for s in self._symbols]).replace("'", '"')}
            ) as resp:
                if resp.status != 200:
                    self._log.warning(
                        f"Binance ticker returned {resp.status}"
                    )
                    return

                data = await resp.json()
                if isinstance(data, dict):
                    data = [data]  # single-symbol response

                now = time.time()
                for entry in data:
                    sym = entry.get("symbol", "")
                    if sym not in self._balances:
                        continue

                    quote_vol = float(entry.get("quoteVolume", 0))
                    price_change_pct = float(entry.get("priceChangePercent", 0))
                    last_price = float(entry.get("lastPrice", 0))

                    # Update price cache from Binance data
                    if last_price > 0:
                        self._prices[sym] = last_price

                    # Estimate whale-sized flow as a fraction of total volume.
                    # Only a small % of total volume is "whale flow".
                    # We estimate ~0.5% of daily volume as flow from large txs.
                    estimated_whale_flow = quote_vol * 0.005

                    # Direction heuristic: if price dropped, there's selling
                    # pressure (inflow to exchanges). If price rose, outflows
                    # to cold storage. This is a simplified proxy.
                    direction = "outflow" if price_change_pct >= 0 else "inflow"
                    flow_amount = estimated_whale_flow

                    # Record individual large flows
                    if flow_amount >= self._whale_threshold:
                        record = FlowRecord(
                            symbol=sym,
                            direction=direction,
                            amount_usd=flow_amount,
                            timestamp=now,
                            source="binance_ticker",
                            details={
                                "quote_volume_24h": quote_vol,
                                "price_change_pct": price_change_pct,
                                "last_price": last_price,
                            },
                        )
                        self._recent_flows.append(record)
                        if len(self._recent_flows) > self._max_flows:
                            self._recent_flows.pop(0)

                    # Update rolling balance
                    bal = self._balances[sym]
                    signed_flow = flow_amount if direction == "inflow" else -flow_amount
                    bal.net_flow_24h += signed_flow
                    if direction == "inflow":
                        bal.total_inflow_24h += flow_amount
                        if flow_amount >= self._whale_threshold:
                            bal.large_tx_count_24h += 1
                    else:
                        bal.total_outflow_24h += flow_amount
                        if flow_amount >= self._whale_threshold:
                            bal.large_tx_count_24h += 1
                    bal.last_updated = now

        except asyncio.TimeoutError:
            self._log.warning("Binance ticker fetch timed out")
        except Exception as exc:
            self._log.warning(f"Binance ticker error: {exc}")

    def _prune_old_records(self) -> None:
        """Remove flow records older than the rolling window."""
        cutoff = time.time() - (_ROLLING_WINDOW_HOURS * 3600)
        self._recent_flows = [
            r for r in self._recent_flows if r.timestamp >= cutoff
        ]

        # Recalculate balances from remaining records
        for sym in self._symbols:
            if sym not in self._balances:
                continue
            sym_records = [
                r for r in self._recent_flows if r.symbol == sym
            ]
            total_in = sum(
                r.amount_usd for r in sym_records if r.direction == "inflow"
            )
            total_out = sum(
                r.amount_usd for r in sym_records if r.direction == "outflow"
            )
            self._balances[sym].net_flow_24h = total_in - total_out
            self._balances[sym].total_inflow_24h = total_in
            self._balances[sym].total_outflow_24h = total_out
            self._balances[sym].large_tx_count_24h = len(sym_records)
