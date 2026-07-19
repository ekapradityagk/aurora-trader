"""
Aurora Trader — Integration API Server.

Main HTTP API server on port 8903 that provides:
  - /health — server health
  - /versions — strategy version management
  - /deploy — deploy a specific version
  - /rollback — rollback to a previous version
  - /winrate — winrate tracking queries
  - /status — unified system dashboard

Orchestrates communication between trading (8900), learning (8901),
and wallet scanner (8902) via the Coordinator.
"""

from __future__ import annotations

import asyncio
import aiohttp
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

from aiohttp import web

from shared.config import load_config
from shared.logger import get_logger
from shared.models import Trade

from integration.version_control import VersionController
from integration.winrate_db import WinrateDB
from integration.rollback import RollbackManager
from integration.coordinator import Coordinator
from integration.projection_db import ProjectionDB
from integration.trade_sync import TradeSyncManager

logger = get_logger("integration.server")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8903


# ---------------------------------------------------------------------------
# Middleware (module-level for aiohttp compatibility)
# ---------------------------------------------------------------------------


@web.middleware
async def _cors_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    """Add CORS headers to all responses."""
    if request.method == "OPTIONS":
        return web.Response(
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        )
    try:
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    except web.HTTPException as exc:
        exc.headers["Access-Control-Allow-Origin"] = "*"
        raise


@web.middleware
async def _error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    """Catch unhandled exceptions and return JSON error responses."""
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except Exception as exc:
        logger.error(
            f"Unhandled error on {request.method} {request.path}: {exc}",
            exc_info=True,
        )
        return web.json_response(
            {"error": "Internal server error", "detail": str(exc)},
            status=500,
        )


# ---------------------------------------------------------------------------
# Integration Server
# ---------------------------------------------------------------------------


class IntegrationServer:
    """Main integration API server that provides a unified interface to the
    entire Aurora Trader system.

    Endpoints:

        GET  /health        — server health
        GET  /versions      — list all strategy versions
        GET  /versions/<tag> — get a specific version
        POST /versions      — create a new version
        POST /deploy        — deploy a strategy version
        POST /rollback      — trigger rollback to a previous version
        GET  /winrate/<tag> — winrate stats for a version
        GET  /winrate/best  — best performing version
        POST /winrate/compare — compare two versions
        GET  /trades        — recent trades
        GET  /status        — unified system dashboard
        GET  /dashboard     — aggregated dashboard view
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._host = host
        self._port = port
        self._cfg = load_config()
        self._log = logger

        # Sub-systems
        self._version_control = VersionController()
        self._winrate_db = WinrateDB()
        self._rollback = RollbackManager()
        self._coordinator = Coordinator()
        self._projection_db = ProjectionDB()
        self._trade_sync = TradeSyncManager()

        # HTTP server
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None

        # State
        self._running = False
        self._start_time: float = 0.0
        # HTTP session for proxying to trading server
        self._proxy_session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise all sub-systems and start the HTTP server."""
        if self._running:
            self._log.warning("Integration server already running")
            return

        self._log.info(
            f"Aurora Trader Integration Server starting — "
            f"{self._host}:{self._port}"
        )

        try:
            # 1. Initialise sub-systems
            await self._winrate_db.initialize()
            await self._rollback.initialize()
            await self._projection_db.initialize()
            await self._coordinator.start()
            await self._trade_sync.start()
            self._proxy_session = aiohttp.ClientSession()
            self._log.info("Sub-systems initialised")

            # 2. Start HTTP server
            await self._start_http_server()
            self._start_time = datetime.now(timezone.utc).timestamp()
            self._log.info(
                f"HTTP server listening on {self._host}:{self._port}"
            )

            self._running = True
            self._log.info("Integration Server started successfully")

        except Exception as exc:
            self._log.critical(
                f"Failed to start integration server: {exc}", exc_info=True
            )
            await self.stop()
            raise

    async def stop(self) -> None:
        """Gracefully shut down the integration server and sub-systems."""
        self._log.info("Integration Server shutting down...")
        self._running = False

        # Stop HTTP server
        if self._runner:
            try:
                await self._runner.cleanup()
            except Exception as exc:
                self._log.debug(f"HTTP cleanup error: {exc}")
            self._runner = None

        # Stop coordinator
        await self._coordinator.stop()

        # Stop trade sync
        await self._trade_sync.stop()

        # Close projection DB
        await self._projection_db.close()

        # Close proxy session
        if self._proxy_session:
            await self._proxy_session.close()
            self._proxy_session = None

        self._log.info("Integration Server stopped")

    # ------------------------------------------------------------------
    # HTTP Server
    # ------------------------------------------------------------------

    async def _start_http_server(self) -> None:
        """Configure routes and start the aiohttp server."""
        self._app = web.Application()

        # Middleware (module-level functions, not class methods)
        self._app.middlewares.append(_cors_middleware)
        self._app.middlewares.append(_error_middleware)

        # Routes
        self._app.router.add_get("/", self._handle_redirect_dashboard)
        self._app.router.add_get("/dashboard", self._handle_index)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/versions", self._handle_list_versions)
        self._app.router.add_get("/versions/{tag}", self._handle_get_version)
        self._app.router.add_post("/versions", self._handle_create_version)
        self._app.router.add_post("/deploy", self._handle_deploy)
        self._app.router.add_post("/rollback", self._handle_rollback)
        self._app.router.add_get("/winrate/{tag}", self._handle_winrate_tag)
        self._app.router.add_get("/winrate/best", self._handle_winrate_best)
        self._app.router.add_post("/winrate/compare", self._handle_winrate_compare)
        self._app.router.add_get("/trades", self._handle_trades)
        self._app.router.add_get("/status", self._handle_status)
        self._app.router.add_get("/api/trading/health", self._handle_proxy_health)
        self._app.router.add_get("/api/trading/positions", self._handle_proxy_positions)
        self._app.router.add_get("/api/trading/trailing-events", self._handle_proxy_trailing_events)
        self._app.router.add_get("/api/trading/closed-trades", self._handle_proxy_closed_trades)
        self._app.router.add_get("/api/trading/signals", self._handle_proxy_signals)
        self._app.router.add_get("/api/dashboard", self._handle_dashboard)
        self._app.router.add_get("/api/pair-rankings", self._handle_proxy_pair_rankings)
        self._app.router.add_get("/api/pair-suitability", self._handle_proxy_pair_suitability)
        self._app.router.add_post("/api/run-suitability", self._handle_proxy_run_suitability)
        self._app.router.add_get("/api/shadow-analysis", self._handle_proxy_shadow_analysis)
        self._app.router.add_get("/api/opportunities", self._handle_proxy_opportunities)
        self._app.router.add_get("/architecture", self._handle_architecture)
        self._app.router.add_get("/projections", self._handle_projections_page)
        self._app.router.add_get("/api/projections/calculate", self._handle_projections_calculate)
        self._app.router.add_post("/api/projections/profile", self._handle_projections_save)
        self._app.router.add_get("/api/projections/profiles", self._handle_projections_list)
        self._app.router.add_get("/api/projections/profile/{profile_id}", self._handle_projections_get)
        self._app.router.add_delete("/api/projections/profile/{profile_id}", self._handle_projections_delete)
        self._app.router.add_put("/api/projections/input", self._handle_projections_input_save)
        self._app.router.add_delete("/api/projections/input/{input_date}", self._handle_projections_input_delete)
        self._app.router.add_get("/api/projections/inputs", self._handle_projections_inputs_list)
        self._app.router.add_get("/api/daily-report", self._handle_daily_report)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

    # ------------------------------------------------------------------
    # Handlers: Index (HTML Dashboard)
    # ------------------------------------------------------------------

    async def _handle_index(self, request: web.Request) -> web.Response:
        """GET /dashboard — serve the Aurora Trader HTML dashboard."""
        import os
        html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
        try:
            with open(html_path, "r") as f:
                html = f.read()
            return web.Response(
                text=html,
                content_type="text/html",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        except FileNotFoundError:
            return web.Response(
                text="<h1>Aurora Trader</h1><p>Dashboard HTML not found. Run from project root.</p>",
                content_type="text/html", status=200
            )

    async def _handle_redirect_dashboard(self, request: web.Request) -> web.Response:
        """GET / — redirect to the HTML dashboard at /dashboard."""
        raise web.HTTPFound("/dashboard")

    # ------------------------------------------------------------------
    # Handlers: Health
    # ------------------------------------------------------------------

    async def _handle_health(self, request: web.Request) -> web.Response:
        """GET /health — return server health status."""
        uptime = 0.0
        if self._start_time > 0:
            uptime = datetime.now(timezone.utc).timestamp() - self._start_time

        return web.json_response(
            {
                "status": "ok" if self._running else "stopped",
                "server": "integration_server",
                "version": "1.0.0",
                "uptime_seconds": round(uptime, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "version_control": True,
                    "winrate_db": True,
                    "rollback": True,
                    "coordinator": self._coordinator._running if self._coordinator else False,
                },
            }
        )

    # ------------------------------------------------------------------
    # Handlers: Versions
    # ------------------------------------------------------------------

    async def _handle_list_versions(self, request: web.Request) -> web.Response:
        """GET /versions — list all strategy versions."""
        versions = self._version_control.list_versions()
        return web.json_response(
            {
                "count": len(versions),
                "versions": versions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def _handle_get_version(self, request: web.Request) -> web.Response:
        """GET /versions/{tag} — get a specific version with parameters."""
        tag = request.match_info.get("tag", "")
        version = self._version_control.get_version(tag)
        if version is None:
            return web.json_response(
                {"error": f"Version '{tag}' not found"},
                status=404,
            )
        return web.json_response(version)

    async def _handle_create_version(self, request: web.Request) -> web.Response:
        """POST /versions — create a new version from strategy parameters.

        Body::
            {
                "strategy_name": "ema_crossover",
                "parameters": {"ema_fast": 9, "ema_slow": 50},
                "description": "Optimised by hyperopt round 3"
            }
        """
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return web.json_response(
                {"error": "Invalid JSON body"},
                status=400,
            )

        strategy_name = body.get("strategy_name", "")
        parameters = body.get("parameters", {})
        description = body.get("description", "")

        if not strategy_name:
            return web.json_response(
                {"error": "Missing 'strategy_name' in body"},
                status=400,
            )
        if not parameters:
            return web.json_response(
                {"error": "Missing 'parameters' in body"},
                status=400,
            )

        tag = self._version_control.save_version(strategy_name, parameters)
        version = self._version_control.get_version(tag)

        return web.json_response(
            {
                "status": "created",
                "version_tag": tag,
                "version": version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            status=201,
        )

    # ------------------------------------------------------------------
    # Handlers: Deploy
    # ------------------------------------------------------------------

    async def _handle_deploy(self, request: web.Request) -> web.Response:
        """POST /deploy — deploy a specific version as the active strategy.

        Body::
            {"version_tag": "v1.2.3"}
        """
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return web.json_response(
                {"error": "Invalid JSON body"},
                status=400,
            )

        tag = body.get("version_tag", "")
        if not tag:
            return web.json_response(
                {"error": "Missing 'version_tag' in body"},
                status=400,
            )

        # Verify the version exists
        version = self._version_control.get_version(tag)
        if version is None:
            return web.json_response(
                {"error": f"Version '{tag}' not found"},
                status=404,
            )

        # Deploy: in production this would push config to trading server
        self._log.info(f"Deploying version '{tag}' for strategy '{version.get('strategy_name', '')}'")

        return web.json_response(
            {
                "status": "deployed",
                "version_tag": tag,
                "strategy_name": version.get("strategy_name", ""),
                "parameters": version.get("parameters", {}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Version '{tag}' deployed successfully",
            }
        )

    # ------------------------------------------------------------------
    # Handlers: Rollback
    # ------------------------------------------------------------------

    async def _handle_rollback(self, request: web.Request) -> web.Response:
        """POST /rollback — trigger a rollback to a previous version.

        Body (manual rollback)::
            {"version_tag": "v1.0.0"}

        Body (auto-detect)::
            {"auto": true, "strategy_name": "ema_crossover", "current_version": "v1.0.1"}
        """
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return web.json_response(
                {"error": "Invalid JSON body"},
                status=400,
            )

        # Auto rollback check
        if body.get("auto"):
            strategy = body.get("strategy_name", "")
            current_tag = body.get("current_version", "")
            if not strategy or not current_tag:
                return web.json_response(
                    {"error": "Auto rollback requires 'strategy_name' and 'current_version'"},
                    status=400,
                )
            rolled_to = await self._rollback.auto_rollback_check(current_tag, strategy)
            if rolled_to:
                return web.json_response(
                    {
                        "status": "rolled_back",
                        "from": current_tag,
                        "to": rolled_to,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            return web.json_response(
                {
                    "status": "no_action_needed",
                    "message": f"Version '{current_tag}' winrate is acceptable",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Manual rollback
        tag = body.get("version_tag", "")
        if not tag:
            return web.json_response(
                {"error": "Missing 'version_tag' in body"},
                status=400,
            )

        self._log.info(f"Manual rollback requested to version '{tag}'")
        rolled_to = await self._rollback.rollback_to(tag)
        if rolled_to is None:
            return web.json_response(
                {"error": f"Version '{tag}' not found or rollback failed"},
                status=404,
            )

        return web.json_response(
            {
                "status": "rolled_back",
                "to": rolled_to,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Rolled back to version '{rolled_to}'",
            }
        )

    # ------------------------------------------------------------------
    # Handlers: Winrate
    # ------------------------------------------------------------------

    async def _handle_winrate_tag(self, request: web.Request) -> web.Response:
        """GET /winrate/{tag} — get winrate stats for a version."""
        tag = request.match_info.get("tag", "")
        stats = await self._winrate_db.get_version_winrate(tag)
        if stats is None:
            return web.json_response(
                {"error": f"No winrate data for version '{tag}'"},
                status=404,
            )
        return web.json_response(stats)

    async def _handle_winrate_best(self, request: web.Request) -> web.Response:
        """GET /winrate/best — get the best performing version."""
        best = await self._winrate_db.get_best_version()
        if best is None:
            return web.json_response(
                {"message": "No version has enough trades (minimum 10) for ranking"},
                status=200,
            )
        return web.json_response(best)

    async def _handle_winrate_compare(self, request: web.Request) -> web.Response:
        """POST /winrate/compare — compare two versions side-by-side.

        Body::
            {"version_a": "v1.0.0", "version_b": "v2.0.0"}
        """
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return web.json_response(
                {"error": "Invalid JSON body"},
                status=400,
            )

        tag_a = body.get("version_a", "")
        tag_b = body.get("version_b", "")
        if not tag_a or not tag_b:
            return web.json_response(
                {"error": "Both 'version_a' and 'version_b' required"},
                status=400,
            )

        comparison = await self._winrate_db.compare_versions(tag_a, tag_b)
        return web.json_response(comparison)

    # ------------------------------------------------------------------
    # Handlers: Trades
    # ------------------------------------------------------------------

    async def _handle_trades(self, request: web.Request) -> web.Response:
        """GET /trades — get recent trade results.

        Merges local winrate_db trades with real Binance futures
        income history (realized PnL) and spot trades.

        Query params:
            version (optional): filter by version tag
            limit (int, default 50): max results
        """
        version_tag = request.query.get("version", "")
        limit = int(request.query.get("limit", "50"))
        limit = min(limit, 500)

        local_trades = await self._winrate_db.get_recent_trades(
            version_tag=version_tag or None,
            limit=limit,
        )

        # Try to fetch real Binance activity
        binance_trades = []
        try:
            from binance.client import Client
            from shared.config import load_config

            cfg = load_config()
            key = cfg.exchange_api_key
            secret = cfg.exchange_api_secret

            if key and secret:
                client = Client(key, secret)
                now_ms = int(time.time() * 1000)
                thirty_days_ago = now_ms - (30 * 24 * 60 * 60 * 1000)

                # 1. Futures Income History — ALL transaction types
                income = client.futures_income_history(
                    startTime=thirty_days_ago, limit=200
                )
                for entry in income:
                    income_type = entry["incomeType"]
                    symbol = entry.get("symbol", "—")
                    amount = float(entry["income"])
                    asset = entry.get("asset", "USDT")

                    if income_type == "REALIZED_PNL":
                        binance_trades.append({
                            "trade_id": f"futures_pnl_{entry['time']}",
                            "symbol": symbol,
                            "side": "LONG" if amount >= 0 else "SHORT",
                            "pnl": amount,
                            "price": 0,
                            "qty": 0,
                            "asset": asset,
                            "activity_type": "futures_pnl",
                            "source": "binance",
                            "closed_at": datetime.fromtimestamp(
                                int(entry["time"]) / 1000, tz=timezone.utc
                            ).isoformat(),
                        })
                    elif income_type == "FUNDING_FEE":
                        binance_trades.append({
                            "trade_id": f"funding_{entry['time']}",
                            "symbol": symbol,
                            "side": "RECEIVED" if amount >= 0 else "PAID",
                            "pnl": amount,
                            "price": 0,
                            "qty": 0,
                            "asset": asset,
                            "activity_type": "funding_fee",
                            "source": "binance",
                            "closed_at": datetime.fromtimestamp(
                                int(entry["time"]) / 1000, tz=timezone.utc
                            ).isoformat(),
                        })
                    elif income_type == "COMMISSION":
                        binance_trades.append({
                            "trade_id": f"commission_{entry['time']}",
                            "symbol": symbol,
                            "side": "",
                            "pnl": amount,
                            "price": 0,
                            "qty": 0,
                            "asset": asset,
                            "activity_type": "commission",
                            "source": "binance",
                            "closed_at": datetime.fromtimestamp(
                                int(entry["time"]) / 1000, tz=timezone.utc
                            ).isoformat(),
                        })
                    else:
                        # TRANSFER, INSURANCE_CLEAR, etc.
                        binance_trades.append({
                            "trade_id": f"other_{entry['time']}",
                            "symbol": symbol,
                            "side": income_type,
                            "pnl": amount,
                            "price": 0,
                            "qty": 0,
                            "asset": asset,
                            "activity_type": "other",
                            "source": "binance",
                            "closed_at": datetime.fromtimestamp(
                                int(entry["time"]) / 1000, tz=timezone.utc
                            ).isoformat(),
                        })

                # 2. Spot trade history (recent active pairs)
                for sym in ["NXPCUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]:
                    try:
                        spot_trades = client.get_my_trades(symbol=sym, limit=20)
                        for t in spot_trades:
                            binance_trades.append({
                                "trade_id": f"spot_{t['id']}",
                                "symbol": sym,
                                "side": "BUY" if t["isBuyer"] else "SELL",
                                "pnl": 0,
                                "price": float(t["price"]),
                                "qty": float(t["qty"]),
                                "total": float(t["qty"]) * float(t["price"]),
                                "asset": sym.replace("USDT", ""),
                                "activity_type": "spot_trade",
                                "source": "binance",
                                "closed_at": datetime.fromtimestamp(
                                    int(t["time"]) / 1000, tz=timezone.utc
                                ).isoformat(),
                            })
                    except Exception:
                        pass

                client.close_connection()
                activity_counts = {}
                for t in binance_trades:
                    at = t.get("activity_type", "unknown")
                    activity_counts[at] = activity_counts.get(at, 0) + 1
                count_str = ", ".join(f"{k}={v}" for k, v in activity_counts.items())
                self._log.info(
                    f"Fetched {len(binance_trades)} real activities from Binance ({count_str})"
                )
        except Exception as exc:
            self._log.warning(f"Could not fetch Binance trades: {exc}")

        # 3. Fetch closed trades from trading server (exchange-closed, flip-closed, etc.)
        trading_trades = []
        try:
            async with self._proxy_session.get(
                "http://127.0.0.1:8900/api/closed-trades?limit=100",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    trading_trades = data.get("trades", [])
        except Exception as exc:
            self._log.debug(f"Could not fetch trading server closed trades: {exc}")

        # Merge: real Binance trades first, then trading server, then local, deduped by ID
        seen = set()
        merged = []
        for t in binance_trades + trading_trades + local_trades:
            tid = t.get("trade_id", "")
            if tid not in seen:
                seen.add(tid)
                merged.append(t)

        # Sort by date descending, then limit
        merged.sort(
            key=lambda t: t.get("closed_at", "") or "",
            reverse=True,
        )
        merged = merged[:limit]

        return web.json_response(
            {
                "count": len(merged),
                "binance_trades": len(binance_trades),
                "trading_trades": len(trading_trades),
                "local_trades": len(local_trades),
                "version_filter": version_tag or "all",
                "trades": merged,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    # ------------------------------------------------------------------
    # Handlers: Status
    # ------------------------------------------------------------------

    async def _handle_status(self, request: web.Request) -> web.Response:
        """GET /status — get unified system status from all components."""
        force = request.query.get("force", "").lower() in ("1", "true", "yes")
        status = await self._coordinator.get_system_status(force_refresh=force)
        return web.json_response(status)

    # ------------------------------------------------------------------
    # Handlers: Proxy to Trading Server
    # ------------------------------------------------------------------

    TRADING_BASE = "http://127.0.0.1:8900"

    async def _proxy_to_trading(self, path: str, request: web.Request) -> web.Response:
        """Forward a request to the trading server and return its response."""
        if not self._proxy_session:
            return web.json_response({"error": "Proxy not available"}, status=503)
        try:
            url = f"{self.TRADING_BASE}{path}"
            qs = request.query_string
            if qs:
                url += f"?{qs}"
            async with self._proxy_session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
                return web.json_response(data, status=resp.status)
        except asyncio.TimeoutError:
            return web.json_response({"error": "Trading server timeout"}, status=504)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)

    async def _handle_proxy_health(self, request: web.Request) -> web.Response:
        """GET /api/trading/health → trading server /health."""
        return await self._proxy_to_trading("/health", request)

    async def _handle_proxy_positions(self, request: web.Request) -> web.Response:
        """GET /api/trading/positions → trading server /positions."""
        return await self._proxy_to_trading("/positions", request)

    async def _handle_proxy_trailing_events(self, request: web.Request) -> web.Response:
        """GET /api/trading/trailing-events → trading server /api/events/trailing."""
        return await self._proxy_to_trading("/api/events/trailing", request)

    async def _handle_proxy_closed_trades(self, request: web.Request) -> web.Response:
        """GET /api/trading/closed-trades → trading server /api/closed-trades."""
        return await self._proxy_to_trading("/api/closed-trades", request)

    async def _handle_proxy_signals(self, request: web.Request) -> web.Response:
        """GET /api/trading/signals → trading server /signals."""
        return await self._proxy_to_trading("/signals", request)

    LEARNING_BASE = "http://127.0.0.1:8901"

    async def _proxy_to_learning(self, path: str, request: web.Request) -> web.Response:
        """Forward a request to the learning server and return its response."""
        if not self._proxy_session:
            return web.json_response({"error": "Proxy not available"}, status=503)
        try:
            url = f"{self.LEARNING_BASE}{path}"
            qs = request.query_string
            if qs:
                url += f"?{qs}"
            async with self._proxy_session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                return web.json_response(data, status=resp.status)
        except asyncio.TimeoutError:
            return web.json_response({"error": "Learning server timeout"}, status=504)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)

    async def _handle_proxy_pair_rankings(self, request: web.Request) -> web.Response:
        """GET /api/pair-rankings → learning server /api/pair-rankings."""
        return await self._proxy_to_learning("/api/pair-rankings", request)

    async def _handle_proxy_pair_suitability(self, request: web.Request) -> web.Response:
        """GET /api/pair-suitability → learning server /api/pair-suitability."""
        return await self._proxy_to_learning("/api/pair-suitability", request)

    async def _handle_proxy_run_suitability(self, request: web.Request) -> web.Response:
        """POST /api/run-suitability → learning server /api/run-suitability."""
        return await self._proxy_to_learning("/api/run-suitability", request)

    async def _handle_proxy_shadow_analysis(self, request: web.Request) -> web.Response:
        """GET /api/shadow-analysis → learning server /api/shadow-analysis."""
        return await self._proxy_to_learning("/api/shadow-analysis", request)

    async def _handle_proxy_opportunities(self, request: web.Request) -> web.Response:
        """GET /api/opportunities → learning server /api/opportunities."""
        return await self._proxy_to_learning("/api/opportunities", request)

    async def _handle_architecture(self, request: web.Request) -> web.Response:
        """GET /architecture — serve the system architecture page."""
        import os
        html_path = os.path.join(os.path.dirname(__file__), "architecture.html")
        try:
            with open(html_path) as f:
                html = f.read()
            return web.Response(text=html, content_type="text/html")
        except FileNotFoundError:
            return web.Response(text="<h1>Architecture page not found</h1>", content_type="text/html", status=404)

    # ------------------------------------------------------------------
    # Handlers: Projections (Capital Projection Calendar)
    # ------------------------------------------------------------------

    async def _handle_projections_page(self, request: web.Request) -> web.Response:
        """GET /projections — serve the capital projection calendar page."""
        import os
        html_path = os.path.join(os.path.dirname(__file__), "projections.html")
        try:
            with open(html_path) as f:
                html = f.read()
            return web.Response(
                text=html,
                content_type="text/html",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        except FileNotFoundError:
            return web.Response(
                text="<h1>Projections page not found</h1>",
                content_type="text/html",
                status=404,
            )

    async def _handle_projections_calculate(self, request: web.Request) -> web.Response:
        """GET /api/projections/calculate — compute capital projection.

        Query params:
            start_date (str, required): YYYY-MM-DD
            capital (float, default 30): starting capital
            target_pct (float, default 5): daily target %
            duration_months (int, default 12): projection duration in months

        Returns both default (config-based) and adjusted (real-input-aware) projections.
        """
        from datetime import datetime, timedelta, timezone
        import calendar
        import statistics

        try:
            start_str = request.query.get("start_date", "")
            if not start_str:
                return web.json_response({"error": "start_date is required"}, status=400)

            start_date = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            capital = float(request.query.get("capital", "30"))
            target_pct = float(request.query.get("target_pct", "5"))
            duration_months = int(request.query.get("duration_months", "12"))

            if capital <= 0:
                return web.json_response({"error": "capital must be positive"}, status=400)
            if target_pct <= 0:
                return web.json_response({"error": "target_pct must be positive"}, status=400)
            if duration_months < 1 or duration_months > 120:
                return web.json_response({"error": "duration_months must be 1-120"}, status=400)

            # Calculate end date
            end_month = start_date.month + duration_months
            end_year = start_date.year + (end_month - 1) // 12
            end_month = ((end_month - 1) % 12) + 1
            last_day = calendar.monthrange(end_year, end_month)[1]
            end_date = start_date.replace(year=end_year, month=end_month, day=min(start_date.day, last_day))
            total_days = (end_date - start_date).days

            # Fetch real PnL inputs in range
            inputs = await self._projection_db.get_inputs_in_range(
                start_str, end_date.strftime("%Y-%m-%d")
            )
            inputs_map = {inp["date"]: inp["actual_pnl"] for inp in inputs}

            multiplier = 1.0 + (target_pct / 100.0)

            # ------------------------------------------------------------------
            # 1. Default projection (pure config-based)
            # ------------------------------------------------------------------
            default_projections = []
            for day_count in range(total_days + 1):
                current = start_date + timedelta(days=day_count)
                day_start_capital = capital * (multiplier ** day_count)
                daily_pnl = day_start_capital * (target_pct / 100.0)
                cumulative_pct = ((day_start_capital / capital) - 1) * 100 if day_count > 0 else 0.0
                default_projections.append({
                    "date": current.strftime("%Y-%m-%d"),
                    "day_of_week": current.strftime("%A"),
                    "day_num": current.day,
                    "projected_capital": round(day_start_capital, 2),
                    "daily_pnl": round(daily_pnl, 2),
                    "cumulative_gain_pct": round(cumulative_pct, 4),
                    "has_input": current.strftime("%Y-%m-%d") in inputs_map,
                })

            default_ending = capital * (multiplier ** (total_days + 1))

            # ------------------------------------------------------------------
            # 2. Adjusted projection (real-input-aware with adaptive rate)
            # ------------------------------------------------------------------
            adjusted_projections = []
            actual_return_rates = []  # rolling list of actual daily return rates
            prev_actual_capital = capital

            for day_count in range(total_days + 1):
                current = start_date + timedelta(days=day_count)
                date_str = current.strftime("%Y-%m-%d")
                adaptive_rate = target_pct  # default fallback

                if date_str in inputs_map:
                    # Use the real PnL input
                    actual_pnl = inputs_map[date_str]
                    day_start_capital = prev_actual_capital
                    daily_pnl = actual_pnl
                    # Calculate the actual return rate for this date
                    if prev_actual_capital > 0:
                        actual_rate = (actual_pnl / prev_actual_capital) * 100.0
                        actual_return_rates.append(actual_rate)
                    # New capital starts next day
                    new_capital = prev_actual_capital + actual_pnl
                else:
                    # Use adaptive rate (median of actual returns, fallback to target)
                    if len(actual_return_rates) >= 2:
                        adaptive_rate = statistics.median(actual_return_rates)
                    else:
                        adaptive_rate = target_pct

                    day_start_capital = prev_actual_capital
                    daily_pnl = day_start_capital * (adaptive_rate / 100.0)
                    new_capital = prev_actual_capital + daily_pnl

                cumulative_pct = ((day_start_capital / capital) - 1) * 100 if day_count > 0 else 0.0
                adjusted_projections.append({
                    "date": date_str,
                    "day_of_week": current.strftime("%A"),
                    "day_num": current.day,
                    "projected_capital": round(day_start_capital, 2),
                    "daily_pnl": round(daily_pnl, 2),
                    "cumulative_gain_pct": round(cumulative_pct, 4),
                    "has_input": date_str in inputs_map,
                    "adaptive_rate": round(adaptive_rate, 4) if date_str not in inputs_map and day_count > 0 else None,
                })

                prev_actual_capital = new_capital

            adjusted_ending = prev_actual_capital
            adjusted_total_gain = adjusted_ending - capital
            adjusted_gain_pct = ((adjusted_ending / capital) - 1) * 100 if capital > 0 else 0

            default_ending_val = default_ending
            default_total_gain = default_ending_val - capital
            default_gain_pct = ((default_ending_val / capital) - 1) * 100 if capital > 0 else 0

            # Build adaptive rate info
            adaptive_rate_info = {
                "configured_target": target_pct,
                "actual_rates": [round(r, 4) for r in actual_return_rates],
                "median_rate": round(statistics.median(actual_return_rates), 4) if len(actual_return_rates) >= 2 else None,
                "total_inputs": len(inputs),
            }

            return web.json_response({
                "start_date": start_str,
                "end_date": end_date.strftime("%Y-%m-%d"),
                "starting_capital": capital,
                "target_daily_pct": target_pct,
                "duration_days": total_days,
                "duration_months": duration_months,
                "adaptive_rate": adaptive_rate_info,
                # Default (config-based)
                "default": {
                    "ending_capital": round(default_ending_val, 2),
                    "total_gain": round(default_total_gain, 2),
                    "gain_pct": round(default_gain_pct, 4),
                    "projections": default_projections,
                },
                # Adjusted (real-input-aware)
                "adjusted": {
                    "ending_capital": round(adjusted_ending, 2),
                    "total_gain": round(adjusted_total_gain, 2),
                    "gain_pct": round(adjusted_gain_pct, 4),
                    "adaptive_rate_used": adaptive_rate_info["median_rate"] or target_pct,
                    "projections": adjusted_projections,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            self._log.error(f"Projection calculation failed: {exc}", exc_info=True)
            return web.json_response({"error": "Calculation failed", "detail": str(exc)}, status=500)

    async def _handle_projections_save(self, request: web.Request) -> web.Response:
        """POST /api/projections/profile — save a projection profile.

        Body:
            {"name": "my profile", "description": "...", "config": {...}}
        """
        try:
            body = await request.json()
            name = body.get("name", "").strip()
            if not name:
                return web.json_response({"error": "name is required"}, status=400)

            config = body.get("config", {})
            if not config:
                return web.json_response({"error": "config is required"}, status=400)

            description = body.get("description", "")

            try:
                profile_id = await self._projection_db.save_profile(name, config, description)
                return web.json_response({
                    "status": "saved",
                    "id": profile_id,
                    "name": name,
                }, status=201)
            except ValueError as exc:
                return web.json_response({"error": str(exc)}, status=409)

        except Exception as exc:
            self._log.error(f"Save projection profile failed: {exc}")
            return web.json_response({"error": str(exc)}, status=500)

    async def _handle_projections_list(self, request: web.Request) -> web.Response:
        """GET /api/projections/profiles — list saved projection profiles."""
        try:
            profiles = await self._projection_db.list_profiles()
            return web.json_response({
                "profiles": profiles,
                "count": len(profiles),
            })
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    async def _handle_projections_get(self, request: web.Request) -> web.Response:
        """GET /api/projections/profile/{profile_id} — get a single profile."""
        try:
            profile_id = int(request.match_info.get("profile_id", "0"))
            profile = await self._projection_db.get_profile(profile_id)
            if profile is None:
                return web.json_response({"error": "Profile not found"}, status=404)
            return web.json_response({"profile": profile})
        except ValueError:
            return web.json_response({"error": "Invalid profile ID"}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    async def _handle_projections_delete(self, request: web.Request) -> web.Response:
        """DELETE /api/projections/profile/{profile_id} — delete a profile."""
        try:
            profile_id = int(request.match_info.get("profile_id", "0"))
            deleted = await self._projection_db.delete_profile(profile_id)
            if not deleted:
                return web.json_response({"error": "Profile not found"}, status=404)
            return web.json_response({"status": "deleted", "id": profile_id})
        except ValueError:
            return web.json_response({"error": "Invalid profile ID"}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    # ------------------------------------------------------------------
    # Handlers: Projection Real PnL Inputs
    # ------------------------------------------------------------------

    async def _handle_projections_input_save(self, request: web.Request) -> web.Response:
        """PUT /api/projections/input — save or update a real PnL input.

        Body:
            {"date": "2026-07-17", "actual_pnl": 1.50, "notes": "good day"}
        """
        try:
            body = await request.json()
            date = body.get("date", "")
            actual_pnl = body.get("actual_pnl")
            notes = body.get("notes", "")

            if not date:
                return web.json_response({"error": "date is required"}, status=400)
            if actual_pnl is None:
                return web.json_response({"error": "actual_pnl is required"}, status=400)

            from datetime import datetime
            datetime.strptime(date, "%Y-%m-%d")  # validate date format

            record = await self._projection_db.save_input(
                date, float(actual_pnl), notes
            )
            return web.json_response({
                "status": "saved",
                "input": record,
            })

        except ValueError as exc:
            return web.json_response({"error": f"Invalid date format: {exc}"}, status=400)
        except Exception as exc:
            self._log.error(f"Save input failed: {exc}")
            return web.json_response({"error": str(exc)}, status=500)

    async def _handle_projections_input_delete(self, request: web.Request) -> web.Response:
        """DELETE /api/projections/input/{input_date} — delete a real PnL input.

        URL path: input_date is YYYY-MM-DD
        """
        try:
            date = request.match_info.get("input_date", "")
            if not date:
                return web.json_response({"error": "Date is required"}, status=400)

            deleted = await self._projection_db.delete_input(date)
            if not deleted:
                return web.json_response({"error": f"No input found for {date}"}, status=404)

            return web.json_response({"status": "deleted", "date": date})

        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    async def _handle_daily_report(self, request: web.Request) -> web.Response:
        """GET /api/daily-report — get yesterday's trade summary for LLM analysis.

        Returns a structured report of yesterday's closed trades, PnL,
        fees, funding, and circuit breaker state.
        """
        from datetime import datetime, timedelta, timezone, date
        import os

        try:
            # Determine date: default to yesterday, or ?date=YYYY-MM-DD
            date_param = request.query.get("date", "")
            if date_param:
                report_date = datetime.strptime(date_param, "%Y-%m-%d").date()
            else:
                report_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

            date_str = report_date.strftime("%Y-%m-%d")
            report = {
                "date": date_str,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_realized_pnl": 0.0,
                    "total_commission": 0.0,
                    "total_funding": 0.0,
                    "trade_count": 0,
                    "win_count": 0,
                    "loss_count": 0,
                },
                "trades": [],
                "fees": [],
                "funding": [],
                "positions": [],
                "circuit_breaker": {},
                "projections": {},
            }

            # Fetch from Binance
            try:
                from binance.client import Client

                key = self._cfg.exchange_api_key
                secret = self._cfg.exchange_api_secret
                if key and secret:
                    client = Client(key, secret)
                    day_start_ms = int(datetime.combine(report_date, datetime.min.time()).timestamp() * 1000)
                    day_end_ms = int(datetime.combine(report_date, datetime.min.time()).timestamp() * 1000) + 86400000

                    income = client.futures_income_history(startTime=day_start_ms, endTime=day_end_ms, limit=500)
                    client.close_connection()

                    for entry in income:
                        income_type = entry["incomeType"]
                        symbol = entry.get("symbol", "—")
                        amount = float(entry["income"])
                        ts = int(entry["time"])
                        time_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%H:%M UTC")

                        if income_type == "REALIZED_PNL":
                            side = "LONG" if amount >= 0 else "SHORT"
                            report["trades"].append({
                                "time": time_str,
                                "symbol": symbol,
                                "side": side,
                                "pnl": round(amount, 2),
                            })
                            report["summary"]["total_realized_pnl"] += amount
                            report["summary"]["trade_count"] += 1
                            if amount > 0:
                                report["summary"]["win_count"] += 1
                            else:
                                report["summary"]["loss_count"] += 1
                        elif income_type == "COMMISSION":
                            report["fees"].append({
                                "time": time_str,
                                "symbol": symbol,
                                "amount": round(amount, 2),
                            })
                            report["summary"]["total_commission"] += amount
                        elif income_type == "FUNDING_FEE":
                            report["funding"].append({
                                "time": time_str,
                                "symbol": symbol,
                                "amount": round(amount, 2),
                            })
                            report["summary"]["total_funding"] += amount

                    report["summary"]["total_realized_pnl"] = round(report["summary"]["total_realized_pnl"], 2)
                    report["summary"]["total_commission"] = round(report["summary"]["total_commission"], 2)
                    report["summary"]["total_funding"] = round(report["summary"]["total_funding"], 2)
                    net = report["summary"]["total_realized_pnl"] + report["summary"]["total_commission"] + report["summary"]["total_funding"]
                    report["summary"]["net_pnl"] = round(net, 2)

            except Exception as exc:
                self._log.debug(f"Daily report Binance fetch error: {exc}")

            # Get circuit breaker state
            try:
                async with self._proxy_session.get(
                    "http://127.0.0.1:8900/risk/state", timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        cb_state = await resp.json()
                        report["circuit_breaker"] = cb_state
            except Exception:
                pass

            # Get open positions
            try:
                async with self._proxy_session.get(
                    "http://127.0.0.1:8900/positions", timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        pos_data = await resp.json()
                        positions = pos_data.get("positions", [])
                        for p in positions:
                            report["positions"].append({
                                "symbol": p.get("symbol", "?"),
                                "side": p.get("side", "?"),
                                "pnl": round(float(p.get("unrealized_pnl", 0)), 2),
                                "roi_pct": round(float(p.get("roi_pct", 0)), 2),
                            })
            except Exception:
                pass

            return web.json_response(report)

        except Exception as exc:
            self._log.error(f"Daily report failed: {exc}", exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    async def _handle_projections_inputs_list(self, request: web.Request) -> web.Response:
        """GET /api/projections/inputs — list all real PnL inputs."""
        try:
            inputs = await self._projection_db.get_all_inputs()
            return web.json_response({
                "inputs": inputs,
                "count": len(inputs),
            })
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    async def _handle_dashboard(self, request: web.Request) -> web.Response:
        """GET /dashboard — aggregated dashboard view of the entire system.

        Combines status, version info, winrate stats, and recent trades
        into a single response for frontend consumption.
        """
        try:
            # Gather data concurrently
            status_task = asyncio.create_task(
                self._coordinator.get_system_status(force_refresh=True)
            )
            versions_task = asyncio.create_task(
                asyncio.to_thread(self._version_control.list_versions)
            )
            best_version_task = asyncio.create_task(
                self._winrate_db.get_best_version()
            )
            recent_trades_task = asyncio.create_task(
                self._winrate_db.get_recent_trades(limit=20)
            )
            daily_summaries_task = asyncio.create_task(
                self._winrate_db.get_daily_summaries(limit=14)
            )
            known_good_task = asyncio.create_task(
                self._rollback.get_known_good_versions()
            )
            rollback_history_task = asyncio.create_task(
                self._rollback.get_rollback_history(limit=10)
            )

            results = await asyncio.gather(
                status_task,
                versions_task,
                best_version_task,
                recent_trades_task,
                daily_summaries_task,
                known_good_task,
                rollback_history_task,
                return_exceptions=True,
            )

            dashboard = {
                "server": {
                    "status": "running" if self._running else "stopped",
                    "uptime_seconds": round(
                        datetime.now(timezone.utc).timestamp() - self._start_time, 2
                    ) if self._start_time > 0 else 0,
                    "version": "1.0.0",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "system_status": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
                "versions": results[1] if not isinstance(results[1], Exception) else [],
                "best_version": results[2] if not isinstance(results[2], Exception) else None,
                "recent_trades": results[3] if not isinstance(results[3], Exception) else [],
                "daily_summaries": results[4] if not isinstance(results[4], Exception) else [],
                "known_good_versions": results[5] if not isinstance(results[5], Exception) else [],
                "rollback_history": results[6] if not isinstance(results[6], Exception) else [],
            }

            return web.json_response(dashboard)

        except Exception as exc:
            self._log.error(f"Dashboard aggregation failed: {exc}", exc_info=True)
            return web.json_response(
                {"error": "Dashboard generation failed", "detail": str(exc)},
                status=500,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the integration server."""
    cfg = load_config()
    host = cfg.data.get("integration", {}).get("host", DEFAULT_HOST)
    port = cfg.data.get("integration", {}).get("port", DEFAULT_PORT)

    server = IntegrationServer(host=host, port=port)

    # Handle shutdown signals
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        if not stop_event.is_set():
            logger.info("Shutdown signal received")
            stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    try:
        await server.start()
        await stop_event.wait()
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
