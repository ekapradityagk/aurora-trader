"""
Aurora Trader — Wallet Scanner: Main Scanner Server.

The central async scanner that:
  - Runs an HTTP server on port 8902
  - Polls on-chain data sources every hour via sub-modules
  - Aggregates signals from exchange flow, whale, and funding rate monitors
  - Produces a combined bullish/bearish bias score (-10 to +10)
  - Exposes /health, /signals, /whale-movements, /exchange-flow endpoints
  - Logs all signals to SQLite for persistence

Usage:
    from wallet_scanner.scanner import run_scanner
    asyncio.run(run_scanner())
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp
from aiohttp import web

from shared.config import Config, load_config
from shared.constants import WALLET_SCANNER_ENDPOINTS, SERVER_PORTS
from shared.logger import get_logger
from shared.models import SignalDirection, WalletSignal

from wallet_scanner.exchange_flow import ExchangeFlowMonitor
from wallet_scanner.whale_tracker import WhaleTracker
from wallet_scanner.funding_rate import FundingRateMonitor
from wallet_scanner.signal_aggregator import SignalAggregator, AggregatedSignal

logger = get_logger("wallet_scanner.scanner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8902
_POLL_INTERVAL_SEC = 3600  # 1 hour (main aggregation cycle)
_SQLITE_DB = "data/wallet_signals.db"
_MAX_SIGNALS_CACHED = 500


# ---------------------------------------------------------------------------
# SQLite Setup
# ---------------------------------------------------------------------------


def _init_db(db_path: str) -> sqlite3.Connection:
    """Initialise the SQLite database and return a connection.

    Creates the ``wallet_signals`` table if it doesn't exist.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wallet_signals (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            bias TEXT NOT NULL,
            overall_score REAL NOT NULL,
            confidence REAL NOT NULL,
            signal_count INTEGER NOT NULL,
            positive_count INTEGER NOT NULL,
            negative_count INTEGER NOT NULL,
            details TEXT,  -- JSON blob
            timestamp REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_wallet_signals_symbol
        ON wallet_signals(symbol)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_wallet_signals_timestamp
        ON wallet_signals(timestamp)
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Wallet Scanner
# ---------------------------------------------------------------------------


class WalletScanner:
    """Main wallet scanner: orchestrates sub-modules, aggregates signals,
    serves HTTP endpoints, and persists signal history to SQLite.

    Usage::

        scanner = WalletScanner()
        await scanner.start()
        # runs until Ctrl+C
        await scanner.stop()
    """

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
    ) -> None:
        self._host = host
        self._port = port
        self._cfg = load_config()
        self._log = logger

        # Database
        db_path = self._cfg.data.get(
            "wallet_scanner", {}
        ).get("database", {}).get("path", _SQLITE_DB)
        self._db_path = str(
            Path(self._cfg.data.get("project", {}).get("name", "aurora-trader")).parent
            / db_path
            if not os.path.isabs(db_path)
            else db_path
        )
        # Resolve relative to project root
        if not os.path.isabs(self._db_path):
            project_root = Path(__file__).resolve().parent.parent
            self._db_path = str(project_root / self._db_path)
        self._db: Optional[sqlite3.Connection] = None
        self._db_lock = asyncio.Lock()

        # Sub-modules
        self._exchange_flow = ExchangeFlowMonitor(config=self._cfg)
        self._whale_tracker = WhaleTracker(config=self._cfg)
        self._funding_rate = FundingRateMonitor(config=self._cfg)
        self._aggregator = SignalAggregator(config=self._cfg)

        # HTTP server
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None

        # State
        self._running = False
        self._tasks: Set[asyncio.Task] = set()

        # Cached aggregated results (for HTTP endpoints)
        self._cached_results: Dict[str, AggregatedSignal] = {}
        # Recent raw data (for /whale-movements, /exchange-flow)
        self._recent_whale_txs: List[Dict[str, Any]] = []
        self._recent_exchange_flows: List[Dict[str, Any]] = []
        # Signal history cache (ring buffer)
        self._signal_history: List[AggregatedSignal] = []
        self._max_history = _MAX_SIGNALS_CACHED

        # Last poll time
        self._last_poll: float = 0.0

        # Symbols to track
        self._symbols: List[str] = (
            self._cfg.data.get("trading_server", {})
            .get("symbols", ["BTCUSDT", "ETHUSDT"])
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise all components and start the event loop."""
        if self._running:
            self._log.warning("Scanner already running")
            return

        self._log.info(
            f"Wallet Scanner starting — host={self._host} port={self._port} symbols={self._symbols}"
        )

        try:
            # 1. Initialise database
            self._db = _init_db(self._db_path)
            self._log.info(f"Database initialised at {self._db_path}")

            # 2. Start sub-modules
            await self._exchange_flow.start()
            await self._whale_tracker.start()
            await self._funding_rate.start()
            self._log.info("All sub-modules started")

            # 3. Start HTTP server
            await self._start_http_server()
            self._log.info(
                f"HTTP server listening on {self._host}:{self._port}"
            )

            # 4. Start background tasks
            self._running = True
            self._tasks.add(
                asyncio.create_task(self._poll_aggregation_loop())
            )
            self._log.info("Background tasks started")

            # 5. Run initial aggregation
            await self._run_aggregation_cycle()

            self._log.info("Wallet Scanner started successfully")

        except Exception as exc:
            self._log.critical(
                f"Failed to start scanner: {exc}", exc_info=True
            )
            await self.stop()
            raise

    async def stop(self) -> None:
        """Gracefully shut down the scanner and all components."""
        self._log.info("Wallet Scanner shutting down...")
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

        # Stop sub-modules
        await self._exchange_flow.stop()
        await self._whale_tracker.stop()
        await self._funding_rate.stop()

        # Stop HTTP server
        if self._runner:
            try:
                await self._runner.cleanup()
            except Exception as exc:
                self._log.debug(f"HTTP cleanup error: {exc}")
            self._runner = None

        # Close database
        if self._db:
            try:
                self._db.close()
            except Exception as exc:
                self._log.debug(f"DB close error: {exc}")
            self._db = None

        self._log.info("Wallet Scanner stopped")

    # ------------------------------------------------------------------
    # HTTP Server
    # ------------------------------------------------------------------

    async def _start_http_server(self) -> None:
        """Set up aiohttp routes and start the HTTP server."""
        self._app = web.Application()

        # Register routes
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get(
            WALLET_SCANNER_ENDPOINTS.get(
                "whale_tracker", "/api/v1/whale/transactions"
            ),
            self._handle_whale_movements,
        )
        self._app.router.add_get(
            WALLET_SCANNER_ENDPOINTS.get(
                "exchange_flow", "/api/v1/exchange/flows"
            ),
            self._handle_exchange_flow,
        )
        self._app.router.add_get(
            WALLET_SCANNER_ENDPOINTS.get(
                "funding_rate", "/api/v1/funding/rates"
            ),
            self._handle_funding_rates,
        )
        # Also add short-friendly alias routes
        self._app.router.add_get("/signals", self._handle_signals)
        self._app.router.add_get(
            "/whale-movements", self._handle_whale_movements
        )
        self._app.router.add_get(
            "/exchange-flow", self._handle_exchange_flow
        )

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(
        self, request: web.Request
    ) -> web.Response:
        """GET /health — return service status."""
        return web.json_response(
            {
                "status": "ok",
                "service": "wallet_scanner",
                "version": "1.0.0",
                "uptime_seconds": time.time() - self._last_poll
                if self._last_poll
                else 0,
                "symbols_tracked": self._symbols,
                "submodules": {
                    "exchange_flow": "running"
                    if self._exchange_flow._running
                    else "stopped",
                    "whale_tracker": "running"
                    if self._whale_tracker._running
                    else "stopped",
                    "funding_rate": "running"
                    if self._funding_rate._running
                    else "stopped",
                },
                "last_poll_ts": self._last_poll,
            }
        )

    async def _handle_signals(
        self, request: web.Request
    ) -> web.Response:
        """GET /signals — return the latest aggregated signals for all
        tracked symbols.

        Query params:
            symbol (str, optional): Filter to a specific symbol.
        """
        symbol = request.query.get("symbol", "").upper()

        # Aggregate fresh signals on demand (cached if within TTL)
        results = await self._collect_aggregated_signals()

        if symbol:
            result = results.get(symbol)
            if result:
                return web.json_response(result.to_dict())
            return web.json_response(
                {"error": f"Symbol '{symbol}' not found"},
                status=404,
            )

        return web.json_response(
            {
                "count": len(results),
                "signals": {
                    sym: r.to_dict() for sym, r in results.items()
                },
            }
        )

    async def _handle_whale_movements(
        self, request: web.Request
    ) -> web.Response:
        """GET /whale-movements — return recent whale transactions
        and holder data.

        Query params:
            limit (int, optional): Max records to return (default 50).
        """
        limit = int(request.query.get("limit", "50"))
        raw_txs = self._whale_tracker.get_large_transactions(limit=limit)
        top_holders = {}
        for sym in self._symbols:
            base = sym.replace("USDT", "")
            holders = self._whale_tracker.get_top_holders(base)
            if holders:
                top_holders[base] = holders

        return web.json_response(
            {
                "symbols_tracked": self._symbols,
                "large_transactions": raw_txs[-limit:] if raw_txs else [],
                "top_holders": top_holders,
                "concentration": {
                    sym.replace("USDT", ""): (
                        self._whale_tracker.get_concentration(
                            sym.replace("USDT", "")
                        )
                    )
                    for sym in self._symbols
                },
            }
        )

    async def _handle_exchange_flow(
        self, request: web.Request
    ) -> web.Response:
        """GET /exchange-flow — return current exchange flow balances.

        Query params:
            symbol (str, optional): Filter to a specific symbol.
        """
        symbol = request.query.get("symbol", "").upper()

        balances = self._exchange_flow.get_all_balances()
        if symbol:
            bal = balances.get(symbol)
            if bal:
                return web.json_response(
                    {
                        "symbol": symbol,
                        "net_flow_24h": round(bal.net_flow_24h, 2),
                        "total_inflow_24h": round(
                            bal.total_inflow_24h, 2
                        ),
                        "total_outflow_24h": round(
                            bal.total_outflow_24h, 2
                        ),
                        "large_tx_count_24h": bal.large_tx_count_24h,
                        "is_bullish": bal.is_bullish,
                        "is_bearish": bal.is_bearish,
                        "last_updated": bal.last_updated,
                    }
                )
            return web.json_response(
                {"error": f"Symbol '{symbol}' not found"},
                status=404,
            )

        return web.json_response(
            {
                "balances": {
                    sym: {
                        "net_flow_24h": round(b.net_flow_24h, 2),
                        "total_inflow_24h": round(
                            b.total_inflow_24h, 2
                        ),
                        "total_outflow_24h": round(
                            b.total_outflow_24h, 2
                        ),
                        "large_tx_count_24h": b.large_tx_count_24h,
                        "is_bullish": b.is_bullish,
                        "is_bearish": b.is_bearish,
                        "last_updated": b.last_updated,
                    }
                    for sym, b in balances.items()
                }
            }
        )

    async def _handle_funding_rates(
        self, request: web.Request
    ) -> web.Response:
        """GET /api/v1/funding/rates — return current funding rate states.

        Query params:
            symbol (str, optional): Filter to a specific symbol.
        """
        symbol = request.query.get("symbol", "").upper()

        states = self._funding_rate.get_all_states()
        if symbol:
            state = states.get(symbol)
            if state:
                return web.json_response(
                    {
                        "symbol": symbol,
                        "current_rate": state.current_rate,
                        "avg_rate_24h": state.avg_rate_24h,
                        "current_oi": state.current_oi,
                        "oi_change_pct": round(
                            state.oi_change_pct, 2
                        ),
                        "is_extreme_positive": state.is_extreme_positive,
                        "is_extreme_negative": state.is_extreme_negative,
                        "is_oi_diverging": state.is_oi_diverging,
                        "last_updated": state.last_updated,
                    }
                )
            return web.json_response(
                {"error": f"Symbol '{symbol}' not found"},
                status=404,
            )

        return web.json_response(
            {
                "rates": {
                    sym: {
                        "current_rate": s.current_rate,
                        "avg_rate_24h": s.avg_rate_24h,
                        "current_oi": s.current_oi,
                        "oi_change_pct": round(s.oi_change_pct, 2),
                        "is_extreme_positive": s.is_extreme_positive,
                        "is_extreme_negative": s.is_extreme_negative,
                        "is_oi_diverging": s.is_oi_diverging,
                        "last_updated": s.last_updated,
                    }
                    for sym, s in states.items()
                }
            }
        )

    # ------------------------------------------------------------------
    # Background Aggregation Loop
    # ------------------------------------------------------------------

    async def _poll_aggregation_loop(self) -> None:
        """Periodically run the full aggregation cycle every hour."""
        while self._running:
            try:
                await asyncio.sleep(_POLL_INTERVAL_SEC)
                await self._run_aggregation_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(
                    f"Aggregation cycle error: {exc}", exc_info=True
                )

    async def _run_aggregation_cycle(self) -> None:
        """Run one full aggregation cycle: collect signals from all
        sub-modules, aggregate, cache, persist to SQLite."""
        self._log.info("Starting aggregation cycle...")
        self._last_poll = time.time()

        try:
            results = await self._collect_aggregated_signals()
            self._log.info(
                f"Aggregation complete — {len(results)} symbols evaluated"
            )
        except Exception as exc:
            self._log.error(
                f"Aggregation cycle failed: {exc}"
            )

    async def _collect_aggregated_signals(
        self,
    ) -> Dict[str, AggregatedSignal]:
        """Collect raw signals from all sub-modules and aggregate.

        Returns dict of symbol → AggregatedSignal.
        """
        # Collect raw signals from each sub-module
        exchange_signals = self._exchange_flow.get_signals()
        whale_signals = self._whale_tracker.get_signals()
        funding_signals = self._funding_rate.get_signals()

        self._log.debug(
            f"Raw signals: exchange={len(exchange_signals)} "
            f"whale={len(whale_signals)} funding={len(funding_signals)}"
        )

        # Aggregate
        results = self._aggregator.aggregate(
            exchange_flow_signals=exchange_signals,
            whale_signals=whale_signals,
            funding_signals=funding_signals,
        )

        # Cache and persist
        self._cached_results = results
        for symbol, result in results.items():
            self._signal_history.append(result)
            if len(self._signal_history) > self._max_history:
                self._signal_history.pop(0)

            # Persist to SQLite
            await self._persist_signal(result)

        # Also cache recent raw data for HTTP endpoints
        self._recent_whale_txs = (
            self._whale_tracker.get_large_transactions(limit=100)
        )
        self._recent_exchange_flows = [
            {
                "symbol": sym,
                "net_flow_24h": round(b.net_flow_24h, 2),
                "total_inflow_24h": round(b.total_inflow_24h, 2),
                "total_outflow_24h": round(b.total_outflow_24h, 2),
                "large_tx_count_24h": b.large_tx_count_24h,
                "is_bullish": b.is_bullish,
                "is_bearish": b.is_bearish,
            }
            for sym, b in self._exchange_flow.get_all_balances().items()
        ]

        return results

    # ------------------------------------------------------------------
    # SQLite Persistence
    # ------------------------------------------------------------------

    async def _persist_signal(
        self, signal: AggregatedSignal
    ) -> None:
        """Write an aggregated signal record to SQLite."""
        if not self._db:
            return

        try:
            async with self._db_lock:
                cursor = self._db.execute(
                    """
                    INSERT OR REPLACE INTO wallet_signals
                        (id, symbol, bias, overall_score, confidence,
                         signal_count, positive_count, negative_count,
                         details, timestamp, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex[:16],
                        signal.symbol,
                        signal.bias,
                        signal.overall_score,
                        signal.confidence,
                        signal.signal_count,
                        signal.positive_count,
                        signal.negative_count,
                        json.dumps(signal.details),
                        signal.timestamp,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                self._db.commit()
        except Exception as exc:
            self._log.error(
                "Failed to persist signal for %s: %s",
                signal.symbol,
                exc,
            )

    async def get_signal_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query signal history from SQLite.

        Args:
            symbol: Filter to a specific symbol (or None for all).
            limit: Max records to return.

        Returns:
            List of signal dicts ordered by timestamp descending.
        """
        if not self._db:
            return []

        try:
            async with self._db_lock:
                if symbol:
                    cursor = self._db.execute(
                        """
                        SELECT * FROM wallet_signals
                        WHERE symbol = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (symbol, limit),
                    )
                else:
                    cursor = self._db.execute(
                        """
                        SELECT * FROM wallet_signals
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (limit,),
                    )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as exc:
            self._log.error(
                "Failed to query signal history: %s", exc
            )
            return []


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


async def run_scanner(
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
) -> None:
    """Convenience function to create, start, and run the Wallet Scanner.

    Handles SIGINT/SIGTERM for graceful shutdown.

    Usage::

        asyncio.run(run_scanner())
    """
    scanner = WalletScanner(host=host, port=port)

    def _signal_handler() -> None:
        asyncio.create_task(scanner.stop())

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        await scanner.start()

        # Keep running until stopped
        while scanner._running:
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        pass
    finally:
        await scanner.stop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_scanner())
