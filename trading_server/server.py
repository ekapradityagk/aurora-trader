"""Aurora Trader — Main Trading Server.

The core execution engine of Aurora Trader. Connects to Binance via
WebSocket and REST, runs strategy analysis on each 1h candle close,
manages positions with ATR-based stop/break-even/trailing, enforces
a daily circuit breaker, and exposes HTTP endpoints for monitoring.

Design:
- Uses asyncio throughout
- WebSocket streams for 1m and 5m (real-time)
- REST polling for 15m, 1h, 4h, 1d (on candle close)
- Strategy checks triggered on 1h close
- Position lifecycle managed by RiskManager
- CircuitBreaker enforces daily loss limit
- aiohttp for HTTP API + aiohttp.web for the server
"""

from __future__ import annotations

import asyncio
import json
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import aiohttp
from aiohttp import web

from shared.config import load_config
from shared.constants import TIMEFRAMES, TIMEFRAME_BINANCE
from shared.logger import get_logger
from shared.models import (
    OrderSide,
    Position,
    PositionStatus,
    Signal,
    SignalDirection,
    TimeFrame,
    Trade,
)

from trading_server.exchange.binance_ws import BinanceWebSocket
from trading_server.exchange.binance_rest import BinanceRestClient
from trading_server.strategies.base import BaseStrategy
from trading_server.strategies.mean_reversion import MeanReversionStrategy
from trading_server.strategies.rsi_divergence import RsiDivergenceStrategy
from trading_server.strategies.trend_follow import TrendFollowStrategy
from trading_server.risk.manager import RiskManager
from trading_server.risk.circuit_breaker import CircuitBreaker

logger = get_logger("trading_server.server")

# ---------------------------------------------------------------------------
# Default symbols if not configured
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]


# ---------------------------------------------------------------------------
# Trading Server
# ---------------------------------------------------------------------------


class TradingServer:
    """Main trading server: market data → strategies → risk → execution.

    Usage::

        server = TradingServer()
        await server.start()
        # runs until Ctrl+C
        await server.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8900,
        symbols: Optional[List[str]] = None,
    ) -> None:
        cfg = load_config()
        ts_cfg = cfg.data.get("trading_server", {})

        self._host = host or ts_cfg.get("host", "127.0.0.1")
        self._port = port or ts_cfg.get("port", 8900)
        self._symbols = symbols or ts_cfg.get("symbols", DEFAULT_SYMBOLS)

        # Core components (lazy-init)
        self._ws: Optional[BinanceWebSocket] = None
        self._rest: Optional[BinanceRestClient] = None
        self._risk_mgr: Optional[RiskManager] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._strategies: Dict[str, BaseStrategy] = {}
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._running = False

        # Background tasks
        self._tasks: Set[asyncio.Task] = set()

        # --- State ---
        # Open positions keyed by symbol
        self._positions: Dict[str, Position] = {}
        # Recent signals (ring buffer)
        self._recent_signals: List[Signal] = []
        self._max_signals_kept = 100
        # Last detected market regime per symbol
        self._last_regime: Dict[str, str] = {}
        # Last candle close timestamps per timeframe (for REST polling)
        self._last_poll: Dict[str, float] = {}
        # Cached kline data per symbol per timeframe
        self._cached_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Account balance (fetched on startup from exchange)
        self._account_balance: Decimal = Decimal("0")  # filled by _update_balance()
        # Daily trade count
        self._daily_trade_count: int = 0
        # Last reset date for daily counters
        self._current_date: str = ""

        # --- Concurrency ---
        self._lock = asyncio.Lock()

        # --- Logging ---
        self._log = logger

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise all components and start the event loop."""
        if self._running:
            self._log.warning("Server already running")
            return

        self._log.info(
            f"Aurora Trader Trading Server starting — "
            f"symbols={self._symbols}"
        )

        try:
            # 1. Initialise risk components
            self._risk_mgr = RiskManager()
            self._circuit_breaker = CircuitBreaker()
            await self._circuit_breaker.initialize()
            self._log.info("Risk components initialised")

            # 2. Initialise REST client
            self._rest = BinanceRestClient()
            self._log.info("REST client initialised")

            # 3. Fetch starting balance
            await self._update_balance()
            await self._circuit_breaker.set_starting_balance(
                self._account_balance
            )
            self._log.info(
                f"Account balance: {self._account_balance:.2f} USDT"
            )

            # 4. Initialise strategies
            self._init_strategies()
            self._log.info(
                f"Strategies loaded: {list(self._strategies.keys())}"
            )

            # 5. Initialise WebSocket
            self._ws = BinanceWebSocket(
                symbols=self._symbols,
                timeframes=["1m", "5m"],
            )
            self._ws.register_callback(self._on_ws_candle)
            await self._ws.start()
            self._log.info("WebSocket connected")

            # 6. Fetch initial kline history for all timeframes
            await self._fetch_initial_klines()

            # 7. Start background tasks
            self._running = True
            self._tasks.add(
                asyncio.create_task(self._poll_higher_timeframes_loop())
            )
            self._tasks.add(
                asyncio.create_task(self._check_and_reset_loop())
            )
            self._tasks.add(
                asyncio.create_task(self._update_positions_loop())
            )
            self._log.info("Background tasks started")

            # 8. Start HTTP server
            await self._start_http_server()
            self._log.info(
                f"HTTP server listening on {self._host}:{self._port}"
            )

            self._log.info("Trading Server started successfully")

        except Exception as exc:
            self._log.critical(
                f"Failed to start server: {exc}", exc_info=True
            )
            await self.stop()
            raise

    async def stop(self) -> None:
        """Gracefully shut down the server and all components."""
        self._log.info("Trading Server shutting down...")
        self._running = False

        # Cancel background tasks
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._tasks.clear()

        # Stop HTTP server
        if self._runner:
            try:
                await self._runner.cleanup()
            except Exception as exc:
                self._log.debug(f"HTTP cleanup error: {exc}")
            self._runner = None

        # Stop WebSocket
        if self._ws:
            await self._ws.stop()

        # Close REST client
        if self._rest:
            await self._rest.close()

        # Close circuit breaker
        if self._circuit_breaker:
            await self._circuit_breaker.close()

        self._log.info("Trading Server stopped")

    # ------------------------------------------------------------------
    # Strategy Initialisation
    # ------------------------------------------------------------------

    def _init_strategies(self) -> None:
        """Load configured strategies."""
        cfg = load_config()
        strategies_cfg = cfg.data.get("strategies", {})

        strategy_registry: Dict[str, Callable[[], BaseStrategy]] = {
            "mean_reversion": lambda: MeanReversionStrategy(),
            "rsi_divergence": lambda: RsiDivergenceStrategy(),
            "trend_follow": lambda: TrendFollowStrategy(),
        }

        for name, factory in strategy_registry.items():
            strat_cfg = strategies_cfg.get(name, {})
            if strat_cfg.get("enabled", True):
                strategy = factory()
                strategy.configure(strat_cfg)
                self._strategies[name] = strategy
                self._log.info(f"  Loaded strategy: {name}")

        if not self._strategies:
            self._log.warning("No strategies enabled")

    # ------------------------------------------------------------------
    # WebSocket Callback
    # ------------------------------------------------------------------

    async def _on_ws_candle(
        self, symbol: str, timeframe: str, raw_data: Dict[str, Any]
    ) -> None:
        """Handle a closed candle from the WebSocket (1m or 5m)."""
        try:
            # Store in cache
            async with self._lock:
                if symbol not in self._cached_data:
                    self._cached_data[symbol] = {}
                if timeframe not in self._cached_data[symbol]:
                    self._cached_data[symbol][timeframe] = {
                        "klines": [],
                        "indicators": {},
                    }
                cache = self._cached_data[symbol][timeframe]

                # Extract kline
                k = raw_data.get("k", {})
                kline_entry = {
                    "open_time": k.get("t", 0),
                    "open": float(k.get("o", 0)),
                    "high": float(k.get("h", 0)),
                    "low": float(k.get("l", 0)),
                    "close": float(k.get("c", 0)),
                    "volume": float(k.get("v", 0)),
                    "close_time": k.get("T", 0),
                    "quote_volume": float(k.get("q", 0)),
                    "count": k.get("n", 0),
                    "taker_buy_volume": float(k.get("V", 0)),
                    "taker_buy_quote_volume": float(k.get("Q", 0)),
                    "is_final": k.get("x", False),
                }
                cache["klines"].append(kline_entry)

                # Keep last 500 candles max
                if len(cache["klines"]) > 500:
                    cache["klines"] = cache["klines"][-500:]

            # If this is also a 1h close (or any higher timeframe boundary),
            # trigger strategy checks. We handle 1h close below via
            # REST polling for accuracy, but we also note the time.

        except Exception as exc:
            self._log.error(
                f"Error in WS callback for {symbol}@{timeframe}: {exc}"
            )

    # ------------------------------------------------------------------
    # Initial Kline Fetch
    # ------------------------------------------------------------------

    async def _fetch_initial_klines(self) -> None:
        """Fetch historical klines for all symbols and all timeframes."""
        higher_tfs = ["15m", "1h", "4h", "1d"]

        for symbol in self._symbols:
            for tf in higher_tfs:
                try:
                    klines = await self._rest.get_klines(
                        symbol=symbol,
                        interval=tf,
                        limit=200,
                    )
                    async with self._lock:
                        if symbol not in self._cached_data:
                            self._cached_data[symbol] = {}
                        self._cached_data[symbol][tf] = {
                            "klines": klines,
                            "indicators": {},
                        }
                    self._last_poll[f"{symbol}_{tf}"] = time.time()
                    self._log.debug(
                        f"Fetched {len(klines)} {tf} klines for {symbol}"
                    )
                except Exception as exc:
                    self._log.error(
                        f"Failed to fetch initial {tf} klines for "
                        f"{symbol}: {exc}"
                    )

            # Also backfill 1m and 5m from REST so we have history
            for tf in ["1m", "5m"]:
                try:
                    klines = await self._rest.get_klines(
                        symbol=symbol,
                        interval=tf,
                        limit=100,
                    )
                    async with self._lock:
                        if symbol not in self._cached_data:
                            self._cached_data[symbol] = {}
                        self._cached_data[symbol][tf] = {
                            "klines": klines,
                            "indicators": {},
                        }
                except Exception as exc:
                    self._log.error(
                        f"Failed to fetch initial {tf} klines for "
                        f"{symbol}: {exc}"
                    )

        self._log.info("Initial kline fetch complete")

    # ------------------------------------------------------------------
    # Higher Timeframe Polling Loop
    # ------------------------------------------------------------------

    async def _poll_higher_timeframes_loop(self) -> None:
        """Periodically check and poll higher timeframes on candle close.

        For each symbol+timeframe, we check if the latest kline's
        close_time has passed (i.e. the candle has closed) and if so,
        fetch new data and potentially trigger strategy checks.
        """
        poll_intervals = {
            "15m": 15 * 60,  # every 15 min
            "1h": 3600,  # every hour
            "4h": 4 * 3600,
            "1d": 86400,
        }

        while self._running:
            try:
                for symbol in self._symbols:
                    for tf, interval in poll_intervals.items():
                        await self._check_and_poll_tf(symbol, tf, interval)

                # Wait 30 seconds between poll cycles
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(
                    f"Error in poll loop: {exc}", exc_info=True
                )
                await asyncio.sleep(60)

    async def _check_and_poll_tf(
        self, symbol: str, tf: str, interval_sec: int
    ) -> None:
        """Check if a higher timeframe needs polling and fetch it."""
        key = f"{symbol}_{tf}"
        last_poll = self._last_poll.get(key, 0.0)
        now = time.time()

        # Only poll if enough time has passed (use 80% of interval
        # to be safe)
        if now - last_poll < interval_sec * 0.8:
            return

        try:
            # Fetch latest klines
            new_klines = await self._rest.get_klines(
                symbol=symbol,
                interval=tf,
                limit=5,
            )
            if not new_klines:
                return

            latest_close_time = new_klines[-1].get("close_time", 0) / 1000.0

            # Only update if the latest candle has closed
            # (close_time < now, and we haven't seen this close before)
            if latest_close_time > now:
                # Candle still open, skip
                return

            # Apply to cache
            async with self._lock:
                if symbol not in self._cached_data:
                    self._cached_data[symbol] = {}
                if tf not in self._cached_data[symbol]:
                    self._cached_data[symbol][tf] = {
                        "klines": [],
                        "indicators": {},
                    }
                cache = self._cached_data[symbol][tf]

                # Merge new klines (dedup by open_time)
                existing_times: Set[int] = set()
                for k in cache["klines"]:
                    existing_times.add(k.get("open_time", 0))

                for k in new_klines:
                    if k.get("open_time", 0) not in existing_times:
                        cache["klines"].append(k)
                        existing_times.add(k.get("open_time", 0))

                # Keep last 500
                if len(cache["klines"]) > 500:
                    cache["klines"] = cache["klines"][-500:]

            self._last_poll[key] = now

            # If this is a 1h close, run strategy checks
            if tf == "1h":
                await self._run_strategy_checks(symbol)

            self._log.debug(
                f"Polled {tf} for {symbol}: "
                f"{len(new_klines)} new klines"
            )

        except Exception as exc:
            self._log.error(
                f"Error polling {tf} for {symbol}: {exc}"
            )

    # ------------------------------------------------------------------
    # Strategy Execution
    # ------------------------------------------------------------------

    async def _run_strategy_checks(self, symbol: str) -> None:
        """Run all enabled strategies on a symbol after a 1h close."""
        # 1. Check circuit breaker
        if await self._circuit_breaker.is_paused():
            self._log.debug(
                f"{symbol} | Circuit breaker paused — skipping strategies"
            )
            return

        # 2. Check if we already have a position for this symbol
        if symbol in self._positions and self._positions[symbol].is_open:
            self._log.debug(
                f"{symbol} | Position already open — skipping entry"
            )
            return

        # 3. Build the data dict for strategies
        data = self._build_strategy_data(symbol)
        if not data:
            return

        # 4. Determine market regime for filtering
        regime = await self._detect_regime(symbol, data)
        self._last_regime[symbol] = regime

        # 5. Run each strategy
        signals: List[Signal] = []
        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue
            try:
                signal = await strategy.execute(symbol, data, regime)
                if signal is not None:
                    signals.append(signal)
            except Exception as exc:
                self._log.error(
                    f"Strategy '{name}' error for {symbol}: {exc}",
                    exc_info=True,
                )

        if not signals:
            self._log.debug(f"{symbol} | No signals generated")
            return

        # 6. Select the best signal (highest confidence)
        signals.sort(key=lambda s: s.confidence, reverse=True)
        best = signals[0]

        # Store signals for API
        async with self._lock:
            self._recent_signals.append(best)
            if len(self._recent_signals) > self._max_signals_kept:
                self._recent_signals = self._recent_signals[
                    -self._max_signals_kept:
                ]

        self._log.info(
            f"{symbol} | Best signal: {best.direction.value.upper()} "
            f"({best.strategy_name}, confidence={best.confidence:.2f}): "
            f"{best.reason}"
        )

        # 7. Execute the signal
        await self._execute_signal(best)

    def _build_strategy_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Build the nested data structure for strategy execution."""
        cache = self._cached_data.get(symbol)
        if not cache:
            return None

        data: Dict[str, Any] = {}
        for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
            tf_data = cache.get(tf)
            if tf_data and tf_data.get("klines"):
                data[tf] = {
                    "klines": tf_data["klines"],
                    "indicators": tf_data.get("indicators", {}),
                }

        if not data:
            return None
        return data

    # ------------------------------------------------------------------
    # Signal Execution
    # ------------------------------------------------------------------

    async def _execute_signal(self, signal: Signal) -> None:
        """Execute a trading signal: calculate size, place order, track position."""
        try:
            symbol = signal.symbol
            entry_price = signal.price

            # 1. Calculate stop loss from signal metadata
            signal_meta = signal.metadata or {}
            stop_loss_str = signal_meta.get("stop_loss", "")
            atr_str = signal_meta.get("atr", "0")
            atr = Decimal(atr_str) if atr_str else Decimal("0")

            if stop_loss_str:
                stop_loss = Decimal(stop_loss_str)
            elif atr > 0:
                side = OrderSide.BUY if signal.direction == SignalDirection.LONG else OrderSide.SELL
                stop_loss = self._risk_mgr.calculate_atr_stop(
                    entry_price, atr, side
                )
            else:
                # Fallback: 2% stop
                pct = Decimal("0.02")
                if signal.direction == SignalDirection.LONG:
                    stop_loss = entry_price * (Decimal("1") - pct)
                else:
                    stop_loss = entry_price * (Decimal("1") + pct)

            # 2. Calculate position size
            size = await self._risk_mgr.calculate_position_size(
                account_balance=self._account_balance,
                entry_price=entry_price,
                stop_loss=stop_loss,
                atr=atr,
                winrate=signal.confidence,
            )

            if size <= 0:
                self._log.warning(
                    f"{symbol} | Position size is zero — skipping"
                )
                return

            # 3. Place the market order
            side_str = "BUY" if signal.direction == SignalDirection.LONG else "SELL"
            order = await self._rest.place_market_order(
                symbol=symbol,
                side=side_str,
                quantity=size,
            )

            # Extract filled price and quantity
            fills = order.get("fills", [])
            if fills:
                avg_price = Decimal(str(sum(
                    float(f["price"]) * float(f["qty"]) for f in fills
                ))) / Decimal(str(sum(float(f["qty"]) for f in fills)))
                filled_qty = Decimal(str(sum(
                    float(f["qty"]) for f in fills
                )))
            else:
                avg_price = entry_price
                filled_qty = size

            # 4. Create position record
            position = Position(
                strategy_name=signal.strategy_name,
                symbol=symbol,
                side=OrderSide.BUY if signal.direction == SignalDirection.LONG else OrderSide.SELL,
                entry_price=avg_price,
                quantity=filled_qty,
                quote_quantity=avg_price * filled_qty,
                stop_loss=stop_loss,
                take_profit=signal_meta.get("take_profit"),
                entry_time=datetime.now(timezone.utc),
                signal_id=signal.id,
                metadata={
                    "atr": str(atr),
                    "entry_price": str(avg_price),
                    "trailing_active": False,
                    "signal_reason": signal.reason,
                    "strategy": signal.strategy_name,
                },
            )

            # 5. Track it
            async with self._lock:
                self._positions[symbol] = position
                self._daily_trade_count += 1
                signal.executed = True
                signal.trade_id = position.id

            # 6. Log
            self._log.trade(
                "OPEN",
                symbol=symbol,
                side=side_str,
                entry=str(avg_price),
                qty=str(filled_qty),
                sl=str(stop_loss),
                strategy=signal.strategy_name,
                confidence=str(signal.confidence),
            )

            self._log.info(
                f"{symbol} | OPENED {side_str} position: "
                f"qty={filled_qty}, entry={avg_price:.2f}, "
                f"sl={stop_loss:.2f}"
            )

        except Exception as exc:
            self._log.error(
                f"Failed to execute signal for {signal.symbol}: {exc}",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Position Management Loop
    # ------------------------------------------------------------------

    async def _update_positions_loop(self) -> None:
        """Periodically check and update open positions."""
        while self._running:
            try:
                await self._update_positions()
                await asyncio.sleep(30)  # check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(
                    f"Error in position update loop: {exc}",
                    exc_info=True,
                )
                await asyncio.sleep(60)

    async def _update_positions(self) -> None:
        """Update all open positions: prices, stop losses, check exits."""
        async with self._lock:
            symbols_to_remove: List[str] = []

            for symbol, position in list(self._positions.items()):
                if not position.is_open:
                    continue

                try:
                    # 1. Get current price
                    current_price = await self._rest.get_ticker_price(symbol)
                    position.current_price = current_price

                    # 2. Get ATR from cache or signal metadata
                    meta = position.metadata or {}
                    atr_str = meta.get("atr", "0")
                    atr = Decimal(atr_str) if atr_str else Decimal("0")

                    # 3. Update unrealized PnL
                    if position.side == OrderSide.BUY:
                        upnl = (current_price - position.entry_price) * position.quantity
                    else:
                        upnl = (position.entry_price - current_price) * position.quantity
                    position.unrealized_pnl = upnl

                    # 4. Update stop loss (break-even / trailing)
                    if atr > 0:
                        new_sl, action = self._risk_mgr.update_stop_loss(
                            position, current_price
                        )
                        if new_sl is not None:
                            position.stop_loss = new_sl
                            if action == "trailing":
                                meta["trailing_active"] = True
                                self._log.info(
                                    f"{symbol} | Trailing stop active: "
                                    f"sl={new_sl:.2f}"
                                )
                            elif action == "break_even":
                                self._log.info(
                                    f"{symbol} | Break-even activated: "
                                    f"sl={new_sl:.2f}"
                                )

                    # 5. Check for stop loss hit
                    sl_hit = False
                    if position.stop_loss:
                        if position.side == OrderSide.BUY and current_price <= position.stop_loss:
                            sl_hit = True
                        elif position.side == OrderSide.SELL and current_price >= position.stop_loss:
                            sl_hit = True

                    if sl_hit:
                        await self._close_position(symbol, "stop_loss")
                        symbols_to_remove.append(symbol)
                        continue

                    # 6. Check for take profit hit
                    tp_hit = False
                    if position.take_profit:
                        if position.side == OrderSide.BUY and current_price >= position.take_profit:
                            tp_hit = True
                        elif position.side == OrderSide.SELL and current_price <= position.take_profit:
                            tp_hit = True

                    if tp_hit:
                        await self._close_position(symbol, "take_profit")
                        symbols_to_remove.append(symbol)
                        continue

                except Exception as exc:
                    self._log.error(
                        f"Error updating position {symbol}: {exc}"
                    )

    async def _close_position(
        self, symbol: str, reason: str
    ) -> None:
        """Close an open position by placing an opposing market order."""
        position = self._positions.get(symbol)
        if not position or not position.is_open:
            return

        try:
            side_str = "SELL" if position.side == OrderSide.BUY else "BUY"

            order = await self._rest.place_market_order(
                symbol=symbol,
                side=side_str,
                quantity=position.quantity,
            )

            # Get fill price
            fills = order.get("fills", [])
            if fills:
                exit_price = Decimal(str(sum(
                    float(f["price"]) * float(f["qty"]) for f in fills
                ))) / Decimal(str(sum(float(f["qty"]) for f in fills)))
            else:
                exit_price = position.current_price or position.entry_price

            # Calculate realised PnL
            if position.side == OrderSide.BUY:
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - exit_price) * position.quantity

            # Close position
            position.exit_price = exit_price
            position.exit_time = datetime.now(timezone.utc)
            position.exit_reason = reason
            position.realized_pnl = pnl
            position.status = PositionStatus.CLOSED

            # Record in circuit breaker
            await self._circuit_breaker.record_trade_pnl(
                symbol=symbol,
                pnl=pnl,
                reason=reason,
            )

            # Log
            self._log.trade(
                "CLOSE",
                symbol=symbol,
                side=side_str,
                entry=str(position.entry_price),
                exit=str(exit_price),
                pnl=str(pnl),
                reason=reason,
            )

            self._log.info(
                f"{symbol} | CLOSED {position.side.value.upper()} position: "
                f"pnl={pnl:.2f}, reason={reason}, "
                f"exit={exit_price:.2f}"
            )

        except Exception as exc:
            self._log.error(
                f"Failed to close position {symbol}: {exc}",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Daily Reset Loop
    # ------------------------------------------------------------------

    async def _check_and_reset_loop(self) -> None:
        """Periodically check for new UTC day and reset counters."""
        while self._running:
            try:
                reset = await self._circuit_breaker.check_and_reset()
                if reset:
                    async with self._lock:
                        self._daily_trade_count = 0
                        self._current_date = CircuitBreaker._utc_date()
                    # Re-read balance
                    await self._update_balance()
                    self._log.info("Daily counters reset")

                await asyncio.sleep(60)  # check every minute

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(f"Error in daily reset loop: {exc}")
                await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Balance / Account
    # ------------------------------------------------------------------

    async def _update_balance(self) -> None:
        """Fetch the current USDT balance."""
        try:
            balance = await self._rest.get_balance("USDT")
            if balance > 0:
                self._account_balance = balance
                self._log.info(f"Fetched live balance: {balance:.2f} USDT")
            else:
                self._log.warning(f"Balance fetch returned 0, keeping previous: {self._account_balance:.2f}")
        except Exception as exc:
            self._log.warning(f"Could not fetch balance: {exc}, using {self._account_balance:.2f}")

    # ------------------------------------------------------------------
    # Regime Detection (simple)
    # ------------------------------------------------------------------

    async def _detect_regime(
        self, symbol: str, data: Dict[str, Any]
    ) -> str:
        """Simple on-the-fly market regime detection.

        Returns "trending", "ranging", or "volatile".
        """
        try:
            tf_data = data.get("1h") or data.get("4h")
            if not tf_data:
                return "unknown"

            klines = tf_data.get("klines", [])
            if len(klines) < 20:
                return "unknown"

            closes = [float(k["close"]) for k in klines]
            highs = [float(k["high"]) for k in klines]
            lows = [float(k["low"]) for k in klines]

            # ADX for trend strength
            from trading_server.strategies.base import BaseStrategy
            adx_values = BaseStrategy.compute_adx(highs, lows, closes)
            current_adx = adx_values[-1] if adx_values else 0

            # BB width for ranging vs volatile
            bb = BaseStrategy.compute_bollinger_bands(closes)
            bb_width = (bb["upper"][-1] - bb["lower"][-1]) / bb["middle"][-1] if bb["middle"][-1] != 0 else 0

            if current_adx > 25:
                return "trending"
            elif bb_width > 0.1:
                return "volatile"
            else:
                return "ranging"

        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    # HTTP Server
    # ------------------------------------------------------------------

    async def _start_http_server(self) -> None:
        """Start the aiohttp HTTP API server."""
        self._app = web.Application()

        # Routes
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/status", self._handle_status)
        self._app.router.add_get("/positions", self._handle_positions)
        self._app.router.add_get("/signals", self._handle_signals)

        # Start
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

    async def _handle_health(
        self, request: web.Request
    ) -> web.Response:
        """Return server health status."""
        state = await self._circuit_breaker.get_state() if self._circuit_breaker else {}
        return web.json_response(
            {
                "status": "ok" if self._running else "stopped",
                "uptime_seconds": None,  # TODO: track start time
                "active_strategies": list(self._strategies.keys()),
                "ws_connected": self._ws.is_connected if self._ws else False,
                "paused": state.get("is_paused", False),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def _handle_status(
        self, request: web.Request
    ) -> web.Response:
        """Return detailed server status with daily P&L."""
        cb_state = {}
        if self._circuit_breaker:
            cb_state = await self._circuit_breaker.get_state()

        open_count = sum(
            1 for p in self._positions.values() if p.is_open
        )

        return web.json_response(
            {
                "server": {
                    "running": self._running,
                    "host": self._host,
                    "port": self._port,
                },
                "account": {
                    "balance_usdt": float(self._account_balance),
                    "daily_trade_count": self._daily_trade_count,
                },
                "circuit_breaker": cb_state,
                "positions": {
                    "open": open_count,
                    "total": len(self._positions),
                },
                "symbols": self._symbols,
                "strategies": list(self._strategies.keys()),
                "websocket": {
                    "connected": self._ws.is_connected if self._ws else False,
                    "streams": self._ws.active_streams if self._ws else [],
                },
                "regime": dict(self._last_regime),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def _handle_positions(
        self, request: web.Request
    ) -> web.Response:
        """Return all positions (open and closed)."""
        filter_param = request.query.get("filter", "all")  # open, closed, all

        positions_list = []
        for pos in self._positions.values():
            if filter_param == "open" and not pos.is_open:
                continue
            if filter_param == "closed" and pos.is_open:
                continue

            positions_list.append(pos.model_dump(mode="json"))

        return web.json_response(
            {
                "count": len(positions_list),
                "filter": filter_param,
                "positions": positions_list,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def _handle_signals(
        self, request: web.Request
    ) -> web.Response:
        """Return recent signals."""
        limit = int(request.query.get("limit", "20"))
        limit = min(limit, 100)

        signals_list = []
        for sig in self._recent_signals[-limit:]:
            signals_list.append(sig.model_dump(mode="json"))

        return web.json_response(
            {
                "count": len(signals_list),
                "signals": signals_list,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the trading server."""
    cfg = load_config()
    ts_cfg = cfg.data.get("trading_server", {})

    host = ts_cfg.get("host", "127.0.0.1")
    port = ts_cfg.get("port", 8900)
    symbols = ts_cfg.get("symbols", DEFAULT_SYMBOLS)

    server = TradingServer(host=host, port=port, symbols=symbols)

    # Handle shutdown signals
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        """Handle shutdown signals."""
        if not stop_event.is_set():
            logger.info("Shutdown signal received")
            stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows or non-UNIX may not support add_signal_handler
            pass

    try:
        await server.start()
        await stop_event.wait()
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
