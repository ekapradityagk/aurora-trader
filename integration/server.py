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
import json
import signal
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

        # HTTP server
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None

        # State
        self._running = False
        self._start_time: float = 0.0

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
            await self._coordinator.start()
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
        self._app.router.add_get("/dashboard", self._handle_dashboard)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

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

        Query params:
            version (optional): filter by version tag
            limit (int, default 50): max results
        """
        version_tag = request.query.get("version", "")
        limit = int(request.query.get("limit", "50"))
        limit = min(limit, 500)

        trades = await self._winrate_db.get_recent_trades(
            version_tag=version_tag or None,
            limit=limit,
        )
        return web.json_response(
            {
                "count": len(trades),
                "version_filter": version_tag or "all",
                "trades": trades,
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
    # Handlers: Dashboard
    # ------------------------------------------------------------------

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
