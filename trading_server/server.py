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
        self._discord_webhook = ts_cfg.get("discord_webhook_url", "") or ""

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
        # Last full analysis per symbol (regime + per-strategy results)
        self._last_analysis: Dict[str, Dict[str, Any]] = {}
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

        # Trailing stop events (ring buffer for dashboard + Discord)
        self._trailing_events: List[Dict[str, Any]] = []
        self._max_trailing_events = 50

        # Closed trades history (ring buffer for dashboard)
        self._closed_trades: List[Dict[str, Any]] = []
        self._max_closed_trades = 100

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

            # 3b. Sync open positions from Binance (so we never lose track
            #     of existing positions after a restart!)
            await self._sync_positions_from_exchange()
            self._log.info(
                f"Exchange positions synced: {len([p for p in self._positions.values() if p.is_open])} open"
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

            # 6b. Run initial analysis on historical data so dashboard shows
            #     pair cards immediately instead of waiting for next 1h close
            self._log.info("Running initial strategy analysis on historical data...")
            for symbol in self._symbols:
                await self._run_strategy_checks(symbol)
            self._log.info("Initial analysis complete")

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
        strategies_result: Dict[str, Dict[str, Any]] = {}
        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                strategies_result[name] = {"enabled": False, "signal": False, "confidence": 0, "reason": "Strategy disabled"}
                continue
            try:
                signal = await strategy.execute(symbol, data, regime)
                if signal is not None:
                    signals.append(signal)
                    strategies_result[name] = {
                        "enabled": True,
                        "signal": True,
                        "direction": signal.direction.value,
                        "confidence": round(signal.confidence, 4),
                        "reason": signal.reason,
                        "indicators": signal.indicators,
                    }
                else:
                    strategies_result[name] = {
                        "enabled": True,
                        "signal": False,
                        "confidence": 0,
                        "reason": strategy.last_skip_reason or "No entry conditions met",
                        "indicators": strategy.last_indicators,
                    }
            except Exception as exc:
                self._log.error(
                    f"Strategy '{name}' error for {symbol}: {exc}",
                    exc_info=True,
                )
                strategies_result[name] = {"enabled": True, "signal": False, "confidence": 0, "reason": f"Error: {exc}"}

        # Store analysis
        async with self._lock:
            self._last_analysis[symbol] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "regime": regime,
                "strategies": strategies_result,
            }

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
    # Exchange Position Sync — fetch live positions from Binance on startup
    # ------------------------------------------------------------------

    async def _sync_positions_from_exchange(self) -> None:
        """Fetch all open positions from Binance and populate self._positions.

        This ensures positions that were opened *before* a server restart
        (or by manual trading on the exchange) are still tracked locally.
        """
        if not self._rest:
            self._log.warning("REST client not available — skipping exchange sync")
            return

        try:
            exchange_positions = await self._rest.get_open_positions()
        except Exception as exc:
            self._log.error(f"Failed to fetch open positions from exchange: {exc}")
            return

        synced = 0
        for ep in exchange_positions:
            symbol = ep.get("symbol", "")
            pos_amt = Decimal(ep.get("positionAmt", "0"))
            if pos_amt == 0:
                continue
            entry_price = Decimal(ep.get("entryPrice", "0"))
            leverage = int(float(ep.get("leverage", 20)))
            unrealized_pnl = Decimal(ep.get("unrealizedProfit", "0"))
            liquidation = ep.get("liquidationPrice")
            liq = Decimal(liquidation) if liquidation and Decimal(liquidation) != 0 else None
            mark_price = Decimal(ep.get("markPrice", "0"))
            side = OrderSide.BUY if pos_amt > 0 else OrderSide.SELL

            # Check if we already track this symbol and skip if so
            existing = self._positions.get(symbol)
            if existing and existing.is_open:
                # Update current price / unrealized PnL from exchange data
                existing.current_price = mark_price
                existing.unrealized_pnl = unrealized_pnl
                existing.liquidation_price = liq
                # Set default SL/TP if missing (e.g. from earlier sync that had no ATR)
                if existing.stop_loss is None:
                    sl_pct = Decimal("0.02")
                    existing.stop_loss = entry_price * (Decimal("1") - sl_pct) if side == OrderSide.BUY else entry_price * (Decimal("1") + sl_pct)
                # No hard TP — trailing stop handles exits
                existing.take_profit = None
                # Set default ATR in metadata if missing (needed for break-even/trailing logic)
                meta = existing.metadata or {}
                if not meta.get("atr") or meta.get("atr") == "0":
                    meta["atr"] = str(entry_price * Decimal("0.015"))  # ~1.5% of price as ATR estimate
                    existing.metadata = meta
                self._log.info(
                    f"  ↻ {symbol}: updated from exchange "
                    f"(price=${float(mark_price):.2f}, PnL={float(unrealized_pnl):+.2f}, "
                    f"sl=${float(existing.stop_loss):.2f}, tp=${float(existing.take_profit):.2f})"
                )
                continue

            # Default SL at 2%, no hard TP (trailing handles exits), ATR estimate at 1.5% of entry
            sl_pct = Decimal("0.02")
            atr_est = entry_price * Decimal("0.015")
            stop_loss = entry_price * (Decimal("1") - sl_pct) if side == OrderSide.BUY else entry_price * (Decimal("1") + sl_pct)

            # Build a Position from exchange data
            pos = Position(
                strategy_name="exchange_sync",
                symbol=symbol,
                side=side,
                status=PositionStatus.OPEN,
                entry_price=entry_price,
                current_price=mark_price,
                liquidation_price=liq,
                quantity=abs(pos_amt),
                quote_quantity=abs(pos_amt) * entry_price,
                leverage=leverage,
                margin_type="isolated",
                unrealized_pnl=unrealized_pnl,
                stop_loss=stop_loss,
                take_profit=None,
                entry_time=datetime.now(timezone.utc),
                metadata={
                    "source": "exchange_sync",
                    "exchange_data": True,
                    "atr": str(atr_est),
                },
            )
            self._positions[symbol] = pos
            synced += 1
            self._log.info(
                f"  ↻ Synced existing {symbol}: {float(pos_amt):.4f} @ ${float(entry_price):.2f} "
                f"x{leverage} (PnL={float(unrealized_pnl):+.2f})"
            )

        if synced > 0:
            self._log.info(f"✓ Synced {synced} existing position(s) from Binance exchange")
        else:
            self._log.info("No existing positions found on exchange (clean start)")

    async def _verify_positions_with_exchange(self) -> None:
        """Cross-check all tracked positions against Binance's actual open positions.

        If a position exists in self._positions but NOT on Binance (positionAmt == 0),
        it means the exchange already closed it — via SL/TP order, manual close,
        liquidation, or any other external action. We close it locally to keep state
        consistent.
        """
        if not self._rest:
            return

        try:
            exchange_positions = await self._rest.get_open_positions()
        except Exception as exc:
            self._log.warning(f"Exchange sync failed (will retry): {exc}")
            return

        # Build set of symbols that Binance says are still open
        exchange_open_symbols = set()
        for ep in exchange_positions:
            sym = ep.get("symbol", "")
            amt = Decimal(ep.get("positionAmt", "0"))
            if sym and amt != 0:
                exchange_open_symbols.add(sym)

        # Check each tracked position — if Binance no longer has it, close locally
        closed_count = 0
        for symbol, position in list(self._positions.items()):
            if not position.is_open:
                continue
            if symbol not in exchange_open_symbols:
                # Binance already closed this position
                position.exit_price = position.current_price or position.entry_price
                position.exit_time = datetime.now(timezone.utc)
                position.exit_reason = "exchange_closed"
                position.realized_pnl = position.unrealized_pnl or Decimal("0")
                position.status = PositionStatus.CLOSED
                self._log.info(
                    f"{symbol} | Detected exchange-close: position no longer on Binance "
                    f"(PnL={float(position.realized_pnl):+.2f})"
                )
                # Record in circuit breaker
                await self._circuit_breaker.record_trade_pnl(
                    symbol=symbol,
                    pnl=position.realized_pnl,
                    reason="exchange_closed",
                )

                # Record in closed trades ring buffer for dashboard
                side_str = "SELL" if position.side == OrderSide.BUY else "BUY"
                self._record_closed_trade(
                    symbol=symbol,
                    side=side_str,
                    entry_price=position.entry_price,
                    exit_price=position.exit_price or position.current_price or position.entry_price,
                    pnl=position.realized_pnl or Decimal("0"),
                    reason="exchange_closed",
                    leverage=position.leverage,
                )
                self._log.trade(
                    "CLOSE",
                    symbol=symbol,
                    side=position.side.value.upper(),
                    entry=str(position.entry_price),
                    exit=str(position.exit_price),
                    pnl=str(position.realized_pnl),
                    reason="exchange_closed",
                )
                closed_count += 1

        if closed_count:
            self._log.info(
                f"✓ Synced: {closed_count} position(s) closed on exchange "
                f"have been removed from local tracking"
            )

    def _record_closed_trade(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        exit_price: Decimal,
        pnl: Decimal,
        reason: str,
        leverage: int = 20,
    ) -> None:
        """Record a closed trade in the ring buffer for the dashboard /trades API."""
        record = {
            "symbol": symbol,
            "side": side,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "pnl": float(pnl),
            "reason": reason,
            "leverage": leverage,
            "activity_type": "futures_pnl",
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._closed_trades.append(record)
        if len(self._closed_trades) > self._max_closed_trades:
            self._closed_trades.pop(0)

    # ------------------------------------------------------------------
    # Position Management Loop
    # ------------------------------------------------------------------

    async def _update_positions_loop(self) -> None:
        """Periodically check and update open positions."""
        exchange_sync_counter = 0
        balance_sync_counter = 0
        while self._running:
            try:
                # Full exchange sync every ~5 minutes (10 × 30s)
                exchange_sync_counter += 1
                if exchange_sync_counter >= 10:
                    exchange_sync_counter = 0
                    await self._verify_positions_with_exchange()

                # Balance sync from Binance every ~2 minutes (4 × 30s)
                balance_sync_counter += 1
                if balance_sync_counter >= 4:
                    balance_sync_counter = 0
                    await self._update_balance()

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
                            old_sl = position.stop_loss
                            position.stop_loss = new_sl
                            if action == "trailing":
                                was_already = meta.get("trailing_active", False)
                                meta["trailing_active"] = True
                                self._log.info(
                                    f"{symbol} | Trailing stop active: "
                                    f"sl={new_sl:.2f}"
                                )
                                # Record event if SL actually changed (avoid duplicate spam)
                                if old_sl != new_sl:
                                    event = {
                                        "symbol": symbol,
                                        "type": "updated" if was_already else "activated",
                                        "entry_price": float(position.entry_price),
                                        "current_price": float(current_price),
                                        "stop_loss": float(new_sl),
                                        "leverage": position.leverage,
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                    }
                                    self._trailing_events.append(event)
                                    if len(self._trailing_events) > self._max_trailing_events:
                                        self._trailing_events.pop(0)
                                    # Cron-based Discord notification (handled externally)
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

            # Record in closed trades ring buffer for dashboard
            self._record_closed_trade(
                symbol=symbol,
                side=side_str,
                entry_price=position.entry_price,
                exit_price=exit_price,
                pnl=pnl,
                reason=reason,
                leverage=position.leverage,
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

    async def _send_discord_webhook(self, event: Dict[str, Any]) -> None:
        """Send a trailing stop event to Discord via webhook."""
        if not self._discord_webhook:
            return
        try:
            symbol = event["symbol"]
            ev_type = event["type"]
            entry = event["entry_price"]
            curr = event["current_price"]
            sl = event["stop_loss"]
            lev = event["leverage"]
            pnl_pct = ((curr - entry) / entry * lev * 100) if entry else 0
            sl_pct = ((sl - entry) / entry * lev * 100) if entry else 0

            title = "🔴 Trailing Activated" if ev_type == "activated" else "🔶 Trailing Updated"
            color = 16753920 if ev_type == "activated" else 16041215  # orange / yellow

            embed = {
                "title": f"{title} — {symbol}",
                "color": color,
                "fields": [
                    {"name": "Entry", "value": f"${entry:.4f}", "inline": True},
                    {"name": "Current", "value": f"${curr:.4f}", "inline": True},
                    {"name": "Trail Stop", "value": f"${sl:.4f}", "inline": True},
                    {"name": "ROI", "value": f"{pnl_pct:+.2f}%", "inline": True},
                    {"name": "Locked Profit", "value": f"{sl_pct:+.2f}%", "inline": True},
                    {"name": "Leverage", "value": f"{lev}x", "inline": True},
                ],
                "timestamp": event.get("timestamp", ""),
            }

            payload = {"embeds": [embed]}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._discord_webhook,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status not in (200, 204):
                        self._log.warning(f"Discord webhook returned {resp.status}")
        except Exception as exc:
            self._log.warning(f"Failed to send Discord webhook: {exc}")

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
        self._app.router.add_post("/api/execute", self._handle_execute)
        self._app.router.add_post("/api/reload-symbols", self._handle_reload_symbols)
        self._app.router.add_get("/api/events/trailing", self._handle_trailing_events)
        self._app.router.add_get("/api/closed-trades", self._handle_closed_trades)

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
                "analysis": {sym: {
                    "timestamp": a["timestamp"],
                    "regime": a["regime"],
                    "strategies": a["strategies"],
                } for sym, a in self._last_analysis.items()},
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

    async def _handle_execute(self, request: web.Request) -> web.Response:
        """POST /api/execute — execute a trade signal from the learning server.

        Body:
            {
                "symbol": "SOLUSDT",
                "side": "BUY" or "SELL",
                "direction": "LONG" or "SHORT",
                "confidence": 83,
                "reason": "Multi-TF confluence: BB lower + RSI 22 + low ADX",
                "price": 75.08
            }

        Safety gates:
            - Circuit breaker check (daily loss limit)
            - Max open positions check
            - Max daily trades check
            - Min balance check ($5 minimum)
        """
        if not self._running:
            return web.json_response({"error": "Server not running"}, status=503)

        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return web.json_response({"error": "Invalid JSON"}, status=400)

        symbol = body.get("symbol", "").upper()
        side_str = body.get("side", "").upper()
        direction = body.get("direction", "").upper()
        confidence = body.get("confidence", 0)
        reason = body.get("reason", "Signal from learning server")

        if not symbol or not side_str or not direction:
            return web.json_response(
                {"error": "Missing 'symbol', 'side', and 'direction'"}, status=400
            )

        if direction not in ("LONG", "SHORT"):
            return web.json_response({"error": "Direction must be LONG or SHORT"}, status=400)

        if side_str not in ("BUY", "SELL"):
            return web.json_response({"error": "Side must be BUY or SELL"}, status=400)

        # === Safety gates ===

        # 1. Circuit breaker
        if self._circuit_breaker:
            cb_state = await self._circuit_breaker.get_state()
            if cb_state.get("is_paused"):
                return web.json_response(
                    {"error": "Circuit breaker is active — trading paused",
                     "detail": cb_state.get("reason", "")},
                    status=403,
                )

        # 2. Balance check
        if self._account_balance < Decimal("5"):
            return web.json_response(
                {"error": f"Balance too low (${float(self._account_balance):.2f}) — minimum $5 required"},
                status=403,
            )

        # 3. Risk check
        if self._risk_mgr:
            open_positions = [p for p in self._positions.values() if p.is_open]
            can_open, msg = self._risk_mgr.can_open_new_position(
                open_positions, self._daily_trade_count
            )
            if not can_open:
                return web.json_response({"error": msg}, status=403)

        # 4. Check if already in a position for this symbol
        existing = self._positions.get(symbol)
        if existing and existing.is_open:
            # Same direction? → reject (already in this trade)
            is_long = direction == "LONG"
            existing_is_long = existing.side == OrderSide.BUY
            if is_long == existing_is_long:
                return web.json_response(
                    {"error": f"Already have a {direction} position for {symbol}"},
                    status=409,
                )
            # Opposite direction → close existing position first (flip)
            self._log.info(
                f"{symbol} | Flipping: closing existing "
                f"{existing.side.value.upper()} to open {direction} "
                f"(confidence={confidence})"
            )
            await self._close_position(symbol, f"flip_to_{direction.lower()}")
            # Position is now closed locally — proceed to open the new one

        # === Calculate position size ===
        balance = float(self._account_balance)
        cfg = load_config()
        lev = int(cfg.data.get("exchange", {}).get("default_leverage", 20))
        risk_pct = cfg.risk_per_trade_pct / 100.0  # e.g. 5% → 0.05
        risk_per_trade = balance * risk_pct

        price = body.get("price", 0)
        if price <= 0:
            return web.json_response({"error": "Invalid or missing price"}, status=400)

        # Minimum position value (Binance Futures minimum notional)
        min_notional = 5.0  # USDT
        min_quantity = min_notional / price

        # Calculate raw quantity FROM balance risk at 20x
        raw_qty = (risk_per_trade * lev) / price
        final_qty = max(raw_qty, min_quantity)

        # Round to step size by querying exchange info
        try:
            from binance import AsyncClient
            client = await self._rest._ensure_client()
            # Set leverage for this symbol on Binance
            try:
                await client.futures_change_leverage(
                    symbol=symbol,
                    leverage=lev,
                )
                self._log.info(f"Leverage set to {lev}x for {symbol}")
            except Exception as exc:
                self._log.warning(f"Leverage change for {symbol}: {exc}")

            info = await client.futures_exchange_info()
            step_size = 0.001
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step_size = float(f["stepSize"])
                    break

            # Round DOWN to step size
            import math
            step_dec = Decimal(str(step_size))
            qty_dec = Decimal(str(final_qty))
            rounded_qty = (qty_dec // step_dec) * step_dec
            if rounded_qty < Decimal(str(min_quantity)):
                rounded_qty = (Decimal(str(math.ceil(min_quantity / step_size))) * step_dec)
            quantity = rounded_qty
        except Exception:
            # Fallback: basic rounding
            quantity = Decimal(str(round(final_qty, 2)))
            quantity = max(quantity, Decimal("0.01"))
            quantity = min(quantity, Decimal("1000000"))

        position_value = float(quantity) * price

        # === Place market order ===
        try:
            order = await self._rest.place_market_order(
                symbol=symbol,
                side=side_str,
                quantity=quantity,
            )
        except Exception as exc:
            self._log.error(f"Failed to execute {symbol}: {exc}")
            return web.json_response(
                {"error": f"Order failed: {exc}"}, status=500
            )

        entry_price = Decimal(str(order.get("avgPrice", price)))

        # === Place stop-loss order (2% below entry for longs, above for shorts) ===
        sl_pct = Decimal("0.02")  # 2% stop loss
        sl_order_id = None
        try:
            if side_str == "BUY":
                stop_price = float(entry_price * (Decimal("1") - sl_pct))
                sl_side = "SELL"
            else:
                stop_price = float(entry_price * (Decimal("1") + sl_pct))
                sl_side = "BUY"

            # Use binance client directly for STOP_MARKET order
            sl_params = {
                "symbol": symbol,
                "side": sl_side,
                "quantity": float(quantity),
                "stopPrice": stop_price,
                "type": "STOP_MARKET",
            }
            client = await self._rest._ensure_client()
            sl_result = await client.futures_create_order(**sl_params)
            sl_order_id = str(sl_result.get("orderId", ""))
            self._log.info(
                f"🛡️ SL placed for {symbol} at ${stop_price:.2f} "
                f"(order: {sl_order_id})"
            )
        except Exception as exc:
            self._log.warning(f"Failed to place SL for {symbol}: {exc}")

        # === Take profit handled by trailing stop (tight trail at 0.5% below peak) ===
        # No hard TP placed — let winners run with trailing protection
        tp_order_id = None
        self._log.info(
            f"📈 No hard TP for {symbol} — tight trailing stop will lock profits"
        )

        # === Record position ===
        from shared.models import OrderSide as OS, PositionStatus as PS
        pos = Position(
            strategy_name="auto_signal",
            symbol=symbol,
            side=OS.BUY if side_str == "BUY" else OS.SELL,
            status=PS.OPEN,
            entry_price=Decimal(str(order.get("avgPrice", price))),
            quantity=quantity,
            quote_quantity=Decimal(str(position_value)),
            leverage=lev,
            margin_type="isolated",
            entry_time=datetime.now(timezone.utc),
            metadata={
                "confidence": confidence,
                "reason": reason,
                "order_id": str(order.get("orderId", "")),
            },
        )
        self._positions[symbol] = pos
        self._daily_trade_count += 1

        # Log
        self._log.info(
            f"🟢 AUTO-EXECUTED {direction} {symbol} "
            f"@ ${float(pos.entry_price):.2f} x {quantity:.4f} "
            f"(confidence: {confidence}%, reason: {reason})"
        )

        return web.json_response({
            "status": "executed",
            "symbol": symbol,
            "side": side_str,
            "direction": direction,
            "entry_price": float(pos.entry_price),
            "quantity": quantity,
            "position_value_usd": round(position_value, 2),
            "leverage": lev,
            "confidence": confidence,
            "reason": reason,
            "order_id": str(order.get("orderId", "")),
            "sl_order_id": sl_order_id or "",
            "tp_order_id": tp_order_id or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def _handle_reload_symbols(self, request: web.Request) -> web.Response:
        """POST /api/reload-symbols — reload symbol list from config without restart."""
        try:
            cfg = load_config()
            new_symbols = cfg.data.get("trading_server", {}).get("symbols", [])

            if not new_symbols:
                return web.json_response(
                    {"error": "No symbols in config"}, status=400
                )

            old_symbols = self._symbols.copy()
            self._symbols = new_symbols

            # Reconnect WebSocket with new symbols
            if self._ws:
                await self._ws.stop()
                self._ws = BinanceWebSocket(
                    symbols=self._symbols,
                    timeframes=["1m", "5m"],
                )
                self._ws.register_callback(self._on_ws_candle)
                await self._ws.start()

            # Re-register REST polling for new symbols
            self._last_poll = {}

            self._log.info(
                f"Symbols reloaded: {old_symbols} → {self._symbols}"
            )

            return web.json_response({
                "status": "reloaded",
                "old_symbols": old_symbols,
                "new_symbols": self._symbols,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            self._log.error(f"Symbol reload failed: {exc}")
            return web.json_response(
                {"error": f"Symbol reload failed: {exc}"}, status=500
            )

    async def _handle_trailing_events(self, request: web.Request) -> web.Response:
        """GET /api/events/trailing — return recent trailing stop events."""
        since = request.query.get("since", "")
        events = self._trailing_events
        if since:
            events = [e for e in events if e.get("timestamp", "") > since]
        return web.json_response({
            "count": len(events),
            "events": events[-20:],  # last 20 max
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


    async def _handle_closed_trades(self, request: web.Request) -> web.Response:
        """GET /api/closed-trades — return recent closed trades with PnL."""
        limit = int(request.query.get("limit", "50"))
        limit = min(limit, 100)
        trades = list(self._closed_trades)
        return web.json_response({
            "count": len(trades),
            "trades": trades[-limit:],
        })


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
