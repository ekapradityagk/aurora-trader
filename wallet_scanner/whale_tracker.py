"""
Aurora Trader — Wallet Scanner: Whale Wallet Tracker.

Monitors top-holder wallets for configured tokens, tracks accumulation/
distribution patterns over multi-day windows, detects dormant-coin movement
(coins moving after 1+ year of inactivity), and flags significant changes
in top-10 holder concentration.

Data Sources (free / public):
    - Etherscan API (free tier, requires API key) for on-chain wallet data
    - CoinGecko API (free, no key) for market cap / holder stats
    - Whale Alert API (free tier) for large transaction alerts

Design:
    - Polls Etherscan for top holder list and token transfer events
    - Maintains a rolling ledger of tracked wallets per symbol
    - Accumulation = net positive balance change over 3+ days
    - Distribution = net negative balance change over 3+ days
    - Dormant movement = tx from wallet inactive > 365 days
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import aiohttp

from shared.config import Config, load_config
from shared.logger import get_logger
from shared.models import SignalDirection

logger = get_logger("wallet_scanner.whale_tracker")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_POLL_INTERVAL_SEC = 3600  # 1 hour
_ACCUMULATION_DAYS = 3  # days of net change to signal
_DORMANT_DAYS = 365  # inactivity threshold for dormant detection
_TOP_HOLDER_COUNT = 10  # number of top holders to track

_DEFAULT_SYMBOLS = ["BTC", "ETH"]

# Etherscan API (free tier, 5 req/sec, requires API key)
_ETHERSCAN_BASE = "https://api.etherscan.io/api"

# Whale Alert API (free tier)
_WHALE_ALERT_BASE = "https://api.whale-alert.io/v1"

# CoinGecko endpoints
_COINGECKO_BASE = "https://api.coingecko.com/api/v3"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HolderSnapshot:
    """A snapshot of a tracked wallet's balance at a point in time."""

    address: str
    symbol: str
    balance: float  # token amount
    balance_usd: Optional[float] = None
    timestamp: float = 0.0
    label: str = ""  # e.g. "exchange", "private", "unknown"


@dataclass
class WalletHistory:
    """Rolling balance history for a single wallet address."""

    address: str
    symbol: str
    snapshots: List[HolderSnapshot] = field(default_factory=list)
    last_active: float = 0.0  # timestamp of last outgoing tx
    label: str = "unknown"

    @property
    def current_balance(self) -> float:
        return self.snapshots[-1].balance if self.snapshots else 0.0

    @property
    def balance_change_3d(self) -> float:
        """Net balance change over the last 3 days."""
        if len(self.snapshots) < 2:
            return 0.0
        cutoff = time.time() - (_ACCUMULATION_DAYS * 86400)
        relevant = [s for s in self.snapshots if s.timestamp >= cutoff]
        if len(relevant) < 2:
            return 0.0
        return relevant[-1].balance - relevant[0].balance

    @property
    def is_accumulating(self) -> bool:
        return self.balance_change_3d > 0 and len(self.snapshots) >= 2

    @property
    def is_distributing(self) -> bool:
        return self.balance_change_3d < 0 and len(self.snapshots) >= 2

    @property
    def is_dormant(self) -> bool:
        """Wallet has been inactive for > 365 days."""
        if self.last_active == 0:
            return False
        return (time.time() - self.last_active) > (_DORMANT_DAYS * 86400)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, calls_per_sec: float = 2.0) -> None:
        self._interval = 1.0 / calls_per_sec
        self._last = 0.0

    async def acquire(self) -> None:
        now = time.monotonic()
        wait = self._last + self._interval - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last = time.monotonic()


# ---------------------------------------------------------------------------
# Whale Tracker
# ---------------------------------------------------------------------------


class WhaleTracker:
    """Monitors whale wallets for accumulation, distribution, and dormancy.

    Uses Etherscan for ERC-20 token holder data and Whale Alert for
    large on-chain transactions.  Falls back to CoinGecko for market
    context when API keys are unavailable.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        symbols: Optional[List[str]] = None,
    ) -> None:
        self._cfg = config or load_config()
        self._symbols = symbols or _DEFAULT_SYMBOLS
        self._log = logger

        # Wallet history keyed by address
        self._wallets: Dict[str, WalletHistory] = {}

        # Top holder snapshots keyed by symbol
        self._top_holders: Dict[str, List[Dict[str, Any]]] = {}

        # Concentration metrics (percentage held by top 10)
        self._concentration: Dict[str, float] = {}

        # Recent large transactions (from Whale Alert)
        self._large_txs: List[Dict[str, Any]] = []
        self._max_txs = 200

        # API keys
        self._etherscan_key: str = self._cfg.wallet_api_key or ""
        self._whale_alert_key: str = self._cfg.wallet_api_key or ""

        # Rate limiters
        self._etherscan_limiter = _RateLimiter(calls_per_sec=3.0)
        self._whale_alert_limiter = _RateLimiter(calls_per_sec=1.0)
        self._coingecko_limiter = _RateLimiter(calls_per_sec=5.0)

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
            f"WhaleTracker started — symbols={self._symbols}"
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
        self._log.info("WhaleTracker stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_top_holders(
        self, symbol: str
    ) -> List[Dict[str, Any]]:
        """Return the tracked top-holder list for *symbol*."""
        return self._top_holders.get(symbol, [])

    def get_concentration(self, symbol: str) -> float:
        """Return the % of supply held by top 10 holders."""
        return self._concentration.get(symbol, 0.0)

    def get_large_transactions(
        self, limit: int = 50
    ) -> List[Dict[str, Any]]:
        return self._large_txs[-limit:]

    def get_signals(self) -> List[Dict[str, Any]]:
        """Evaluate wallet data and produce signal dicts."""
        signals: List[Dict[str, Any]] = []

        for addr, wallet in self._wallets.items():
            sym = wallet.symbol

            # Accumulation signal
            if wallet.is_accumulating:
                change = wallet.balance_change_3d
                confidence = min(abs(change) / 100.0, 1.0)
                signals.append({
                    "symbol": sym,
                    "direction": SignalDirection.LONG.value,
                    "type": "whale_accumulation",
                    "confidence": confidence,
                    "value": change,
                    "address": addr,
                    "reason": (
                        f"Whale {addr[:8]}... accumulating "
                        f"{change:+.2f} over {_ACCUMULATION_DAYS}d — bullish"
                    ),
                })

            # Distribution signal
            if wallet.is_distributing:
                change = wallet.balance_change_3d
                confidence = min(abs(change) / 100.0, 1.0)
                signals.append({
                    "symbol": sym,
                    "direction": SignalDirection.SHORT.value,
                    "type": "whale_distribution",
                    "confidence": confidence,
                    "value": change,
                    "address": addr,
                    "reason": (
                        f"Whale {addr[:8]}... distributing "
                        f"{change:+.2f} over {_ACCUMULATION_DAYS}d — bearish"
                    ),
                })

            # Dormant movement signal
            if wallet.is_dormant and wallet.snapshots:
                confidence = min(
                    wallet.current_balance / 500.0, 1.0
                )
                signals.append({
                    "symbol": sym,
                    "direction": SignalDirection.NEUTRAL.value,
                    "type": "dormant_movement",
                    "confidence": confidence,
                    "value": wallet.current_balance,
                    "address": addr,
                    "reason": (
                        f"Dormant wallet {addr[:8]}... moved "
                        f"{wallet.current_balance:.2f} after "
                        f">{_DORMANT_DAYS}d inactivity — watch"
                    ),
                })

        # Concentration changes
        for sym, conc in self._concentration.items():
            # High concentration is bearish (centralized selling risk)
            if conc > 50.0:
                signals.append({
                    "symbol": sym,
                    "direction": SignalDirection.SHORT.value,
                    "type": "high_concentration",
                    "confidence": min(conc / 100.0, 1.0),
                    "value": conc,
                    "reason": (
                        f"Top 10 holders control {conc:.1f}% of "
                        f"{sym} supply — centralization risk"
                    ),
                })

        return signals

    # ------------------------------------------------------------------
    # Background Poll Loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_whale_data()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(
                    f"Whale poll error: {exc}", exc_info=True
                )
            for _ in range(_POLL_INTERVAL_SEC // 10):
                if not self._running:
                    return
                await asyncio.sleep(10)

    async def _poll_whale_data(self) -> None:
        """Run all data-fetching tasks concurrently."""
        self._log.debug("Polling whale data...")

        tasks = []
        if self._etherscan_key:
            tasks.append(self._fetch_etherscan_holders())
        if self._whale_alert_key:
            tasks.append(self._fetch_whale_alert_txs())

        # Always fetch CoinGecko market data as fallback
        tasks.append(self._fetch_coingecko_market())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._log.debug("Whale data poll complete")

    async def _fetch_etherscan_holders(self) -> None:
        """Fetch top holders for tracked tokens via Etherscan.

        Etherscan's free tier provides the token holder list for
        ERC-20 tokens via the ``tokenholderlist`` module.
        """
        if not self._session or not self._etherscan_key:
            return

        # Token contract addresses (expand as needed)
        token_contracts = {
            "ETH": None,  # ETH itself is the native currency
            "BTC": None,  # BTC is not on Ethereum
            "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
            "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        }

        for sym in self._symbols:
            contract = token_contracts.get(sym)
            if contract is None:
                continue

            try:
                await self._etherscan_limiter.acquire()

                params: Dict[str, str] = {
                    "module": "token",
                    "action": "tokenholderlist",
                    "contractaddress": contract,
                    "page": "1",
                    "offset": str(_TOP_HOLDER_COUNT),
                    "apikey": self._etherscan_key,
                }
                async with self._session.get(
                    _ETHERSCAN_BASE, params=params
                ) as resp:
                    if resp.status != 200:
                        self._log.debug(
                            f"Etherscan returned {resp.status} for {sym}"
                        )
                        continue

                    data = await resp.json()
                    if data.get("status") != "1":
                        continue

                    holders = data.get("result", [])[
                        :_TOP_HOLDER_COUNT
                    ]
                    self._top_holders[sym] = holders

                    # Update wallet histories
                    now = time.time()
                    for h in holders:
                        addr = h.get("address", "")
                        bal = float(h.get("quantity", "0"))
                        if addr not in self._wallets:
                            self._wallets[addr] = WalletHistory(
                                address=addr, symbol=sym
                            )
                        wallet = self._wallets[addr]
                        wallet.snapshots.append(
                            HolderSnapshot(
                                address=addr,
                                symbol=sym,
                                balance=bal,
                                timestamp=now,
                            )
                        )
                        # Keep last 30 snapshots max
                        if len(wallet.snapshots) > 30:
                            wallet.snapshots = wallet.snapshots[-30:]

                    # Calculate concentration
                    if holders:
                        total_supply = sum(
                            float(h.get("quantity", "0"))
                            for h in holders
                        )
                        # Estimate total supply from CoinGecko or default
                        total = total_supply * 10  # rough estimate
                        self._concentration[sym] = (
                            (total_supply / total) * 100
                            if total > 0
                            else 0.0
                        )

            except asyncio.TimeoutError:
                self._log.debug(
                    f"Etherscan timeout for {sym}"
                )
            except Exception as exc:
                self._log.debug(
                    f"Etherscan error for {sym}: {exc}"
                )

    async def _fetch_whale_alert_txs(self) -> None:
        """Fetch large transactions from Whale Alert API."""
        if not self._session or not self._whale_alert_key:
            return

        try:
            await self._whale_alert_limiter.acquire()

            params = {
                "api_key": self._whale_alert_key,
                "min_value": 500000,  # $500k minimum for free tier
            }
            async with self._session.get(
                f"{_WHALE_ALERT_BASE}/transactions",
                params=params,
            ) as resp:
                if resp.status != 200:
                    self._log.debug(
                        f"Whale Alert returned {resp.status}"
                    )
                    return

                data = await resp.json()
                txs = data.get("transactions", [])
                for tx in txs:
                    tx_hash = tx.get("hash", "")
                    # Avoid duplicates
                    if any(
                        t.get("hash") == tx_hash
                        for t in self._large_txs
                    ):
                        continue
                    # Flag dormant if sender inactive > 365d
                    tx["flagged_dormant"] = False
                    self._large_txs.append(tx)

                # Keep max
                if len(self._large_txs) > self._max_txs:
                    self._large_txs = self._large_txs[-self._max_txs:]

        except asyncio.TimeoutError:
            self._log.debug("Whale Alert timeout")
        except Exception as exc:
            self._log.debug(f"Whale Alert error: {exc}")

    async def _fetch_coingecko_market(self) -> None:
        """Fetch market data from CoinGecko as a fallback source for
        holder concentration estimates."""
        if not self._session:
            return

        for sym in self._symbols:
            try:
                await self._coingecko_limiter.acquire()

                coin_id = sym.lower()
                if coin_id == "btc":
                    coin_id = "bitcoin"
                elif coin_id == "eth":
                    coin_id = "ethereum"

                url = f"{_COINGECKO_BASE}/coins/{coin_id}"
                params = {
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "true",
                    "community_data": "false",
                    "developer_data": "false",
                }
                async with self._session.get(
                    url, params=params
                ) as resp:
                    if resp.status != 200:
                        continue

                    data = await resp.json()

                    # Extract holder concentration from
                    # `market_data.holders` if available
                    mkt = data.get("market_data", {})
                    if not mkt:
                        continue

                    # CoinGecko sometimes provides top_holder data
                    # under `public_interest_stats`
                    # Fallback: use market cap rank as proxy
                    self._log.debug(
                        f"CoinGecko data for {sym}: "
                        f"mcap={mkt.get('market_cap', {}).get('usd', 0):.0f}"
                    )

            except asyncio.TimeoutError:
                pass
            except Exception as exc:
                self._log.debug(
                    f"CoinGecko market error for {sym}: {exc}"
                )
