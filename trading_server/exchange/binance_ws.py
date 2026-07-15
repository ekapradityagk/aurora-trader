"""Aurora Trader — Binance WebSocket Manager.

Subscribes to real-time 1m and 5m kline streams for configured symbols.
Handles reconnection with exponential backoff and fires callbacks on
each new candle close.

Uses the python-binance library's ``BinanceSocketManager`` for reliable
WebSocket connectivity.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Set

from shared.config import load_config
from shared.constants import TIMEFRAME_BINANCE
from shared.logger import get_logger

logger = get_logger("trading_server.exchange.ws")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

KlineCallback = Callable[["str", str, Dict[str, Any]], Any]
"""Signature: callback(symbol: str, timeframe: str, kline_data: dict)"""


@dataclass
class KlineData:
    """Normalised kline data extracted from a Binance WebSocket event."""

    symbol: str
    timeframe: str
    open_time: int  # ms epoch
    close_time: int  # ms epoch
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal
    taker_buy_base: Decimal
    taker_buy_quote: Decimal
    is_final: bool  # True if this candle is closed

    @classmethod
    def from_ws_event(cls, data: Dict[str, Any]) -> "KlineData":
        k = data["k"]
        return cls(
            symbol=data["s"],
            timeframe=k["i"],
            open_time=k["t"],
            close_time=k["T"],
            open=Decimal(k["o"]),
            high=Decimal(k["h"]),
            low=Decimal(k["l"]),
            close=Decimal(k["c"]),
            volume=Decimal(k["v"]),
            quote_volume=Decimal(k["q"]),
            taker_buy_base=Decimal(k["V"]),
            taker_buy_quote=Decimal(k["Q"]),
            is_final=k["x"],
        )


# ---------------------------------------------------------------------------
# WebSocket Manager
# ---------------------------------------------------------------------------


class BinanceWebSocket:
    """Manages real-time WebSocket connections to Binance for kline streams.

    Features:
    - Subscribe to 1m and 5m klines for any number of symbols
    - Automatic reconnection with exponential backoff (up to ``max_retries``)
    - Configurable ping interval to keep the connection alive
    - Fires registered callbacks on each new candle close event

    Usage::

        ws = BinanceWebSocket(symbols=["BTCUSDT", "ETHUSDT"])
        ws.register_callback(my_callback)
        await ws.start()
        # ... runs until ws.stop() is called
        await ws.stop()
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        max_retries: int = 5,
        retry_delay_seconds: float = 2.0,
        ping_interval_seconds: int = 30,
    ) -> None:
        self._symbols: List[str] = symbols or ["BTCUSDT", "ETHUSDT"]
        self._timeframes: List[str] = timeframes or ["1m", "5m"]
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds
        self._ping_interval = ping_interval_seconds

        # Derived stream names (lowercase per Binance convention)
        self._streams: List[str] = [
            f"{sym.lower()}@kline_{TIMEFRAME_BINANCE[tf]}"
            for sym in self._symbols
            for tf in self._timeframes
        ]

        self._callbacks: List[KlineCallback] = []
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._ws: Any = None  # Binance websocket connection
        self._lock = asyncio.Lock()

        # Track subscription state
        self._subscribed_streams: Set[str] = set()

        # Config
        cfg = load_config()
        self._testnet = cfg.data.get("exchange", {}).get("testnet", True)
        self._use_futures = cfg.data.get("exchange", {}).get("use_futures", False)
        ws_cfg = cfg.data.get("trading_server", {}).get("websocket", {})
        self._max_retries = ws_cfg.get("max_retries", max_retries)
        self._retry_delay = ws_cfg.get("retry_delay_seconds", retry_delay_seconds)
        self._ping_interval = ws_cfg.get("ping_interval_seconds", ping_interval_seconds)

        self._log = get_logger("trading_server.exchange.ws")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_callback(self, callback: KlineCallback) -> None:
        """Register a function to call on each new candle close.

        The callback receives ``(symbol, timeframe, kline_data_dict)``.
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            self._log.info(f"Registered kline callback: {callback.__name__}")

    def unregister_callback(self, callback: KlineCallback) -> None:
        """Remove a previously registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            self._log.info(f"Unregistered kline callback: {callback.__name__}")

    async def start(self) -> None:
        """Connect to Binance WebSocket and begin processing streams.

        Spawns a background task that manages the connection and reconnects
        with exponential backoff on failure.
        """
        if self._running:
            self._log.warning("WebSocket already running")
            return

        self._running = True
        task = asyncio.create_task(self._run_forever())
        self._tasks.append(task)
        self._log.info(
            f"WebSocket manager started: {len(self._symbols)} symbols × "
            f"{len(self._timeframes)} timeframes = {len(self._streams)} streams"
        )

    async def stop(self) -> None:
        """Gracefully shut down the WebSocket connection and all tasks."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks.clear()
        self._log.info("WebSocket manager stopped")

    @property
    def is_connected(self) -> bool:
        """Return True if the WebSocket connection is active."""
        return self._ws is not None and self._running

    @property
    def symbols(self) -> List[str]:
        return list(self._symbols)

    @property
    def active_streams(self) -> List[str]:
        return list(self._streams)

    # ------------------------------------------------------------------
    # Internal — Connection & Reconnect Loop
    # ------------------------------------------------------------------

    async def _run_forever(self) -> None:
        """Main reconnection loop with exponential backoff."""
        attempt = 0
        while self._running:
            try:
                attempt += 1
                self._log.info(
                    f"Connecting to Binance WebSocket (attempt {attempt})..."
                )

                # We use python-binance's async client + socket manager
                from binance import AsyncClient, BinanceSocketManager

                cfg = load_config()
                api_key = cfg.exchange_api_key or None
                api_secret = cfg.exchange_api_secret or None

                if self._testnet:
                    client = await AsyncClient.create(
                        api_key=api_key or "",
                        api_secret=api_secret or "",
                        testnet=True,
                    )
                else:
                    client = await AsyncClient.create(
                        api_key=api_key or "",
                        api_secret=api_secret or "",
                    )

                bm = BinanceSocketManager(client, user_timeout=self._ping_interval)

                # Subscribe to all kline streams (futures or spot)
                if self._use_futures:
                    self._ws = bm.futures_multiplex_socket(self._streams)
                else:
                    self._ws = bm.multiplex_socket(self._streams)

                async with self._ws as stream:
                    self._log.info("WebSocket connected successfully")
                    attempt = 0  # reset backoff on successful connect
                    self._subscribed_streams = set(self._streams)

                    # Handle both old (async generator) and new (recv-based) APIs
                    try:
                        async for msg in stream:
                            if not self._running:
                                break
                            await self._handle_message(msg)
                    except AttributeError:
                        # Newer python-binance versions use recv()
                        while self._running:
                            try:
                                msg = await stream.recv()
                                if msg:
                                    await self._handle_message(msg)
                            except asyncio.TimeoutError:
                                continue

                # Clean disconnect
                await client.close_connection()
                self._ws = None

            except asyncio.CancelledError:
                self._log.info("WebSocket task cancelled")
                break
            except Exception as exc:
                self._log.error(f"WebSocket error: {exc}", exc_info=True)
                if not self._running:
                    break
                # Exponential backoff
                delay = self._retry_delay * (2 ** (attempt - 1))
                delay = min(delay, 60.0)  # cap at 60 seconds
                if attempt > self._max_retries:
                    self._log.critical(
                        f"Exceeded max retries ({self._max_retries}), "
                        f"giving up on WebSocket"
                    )
                    break
                self._log.info(f"Reconnecting in {delay:.1f}s (attempt {attempt})")
                await asyncio.sleep(delay)

    async def _handle_message(self, raw: Dict[str, Any]) -> None:
        """Process a single WebSocket message and fire callbacks on close."""
        try:
            data = raw.get("data", raw)
            event_type = data.get("e", "")
            if event_type != "kline":
                return

            kline = KlineData.from_ws_event(data)

            # Only fire callbacks on closed candles (newly finished)
            if not kline.is_final:
                return

            # Fire callbacks
            for cb in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(kline.symbol, kline.timeframe, data)
                    else:
                        cb(kline.symbol, kline.timeframe, data)
                except Exception as exc:
                    self._log.error(
                        f"Callback {cb.__name__} failed for "
                        f"{kline.symbol}@{kline.timeframe}: {exc}"
                    )

        except Exception as exc:
            self._log.error(f"Error handling WS message: {exc}", exc_info=True)
