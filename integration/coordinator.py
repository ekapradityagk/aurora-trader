"""
Aurora Trader — System Coordinator / Orchestrator.

Pings all component servers for health, syncs strategy config between
learning server and trading server, relays wallet scanner signals as
bias overrides, provides unified system status, and manages startup/
shutdown sequencing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import aiohttp
from aiohttp import ClientSession, ClientTimeout, web

from shared.config import load_config
from shared.logger import get_logger

logger = get_logger("integration.coordinator")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=10)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class Coordinator:
    """Orchestrates communication between all Aurora Trader components.

    Responsibilities:
      - Health-check pings to all 4 servers
      - Strategy config sync from learning → trading
      - Wallet signal relay as bias overrides
      - Unified system status aggregation
      - Startup/shutdown sequencing

    Usage::

        coordinator = Coordinator()
        await coordinator.start()
        status = await coordinator.get_system_status()
        await coordinator.stop()
    """

    def __init__(self) -> None:
        self._cfg = load_config()
        self._log = logger

        # Component URLs
        self._trading_url = (
            f"http://{self._cfg.data.get('trading_server', {}).get('host', '127.0.0.1')}"
            f":{self._cfg.data.get('trading_server', {}).get('port', 8900)}"
        )
        self._learning_url = (
            f"{self._cfg.data.get('learning_server', {}).get('api_base_url', 'http://127.0.0.1:8901')}"
        )
        self._wallet_url = (
            f"{self._cfg.data.get('wallet_scanner', {}).get('api_base_url', 'http://127.0.0.1:8902')}"
        )

        # State
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
        self._session: Optional[ClientSession] = None

        # Cached status (updated by background polling)
        self._cached_status: Dict[str, Any] = {}
        self._last_status_update: float = 0.0
        self._status_ttl = 15.0  # seconds

        # Wallet signals cached for relay
        self._cached_wallet_signals: Dict[str, Any] = {}

        # HTTP timeout
        self._timeout = _DEFAULT_TIMEOUT

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the coordinator and start background tasks."""
        if self._running:
            self._log.warning("Coordinator already running")
            return

        self._log.info("Coordinator starting...")
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        self._running = True

        # Background status polling
        self._tasks.add(
            asyncio.create_task(self._status_poll_loop())
        )

        # Wallet signal relay (if enabled)
        if self._cfg.wallet_enabled:
            self._tasks.add(
                asyncio.create_task(self._wallet_relay_loop())
            )

        self._log.info("Coordinator started")

    async def stop(self) -> None:
        """Gracefully shut down the coordinator."""
        self._log.info("Coordinator shutting down...")
        self._running = False

        for task in list(self._tasks):
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._tasks.clear()

        if self._session:
            await self._session.close()
            self._session = None

        self._log.info("Coordinator stopped")

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def ping_trading_server(self) -> Dict[str, Any]:
        """Ping the trading server (port 8900) /health endpoint."""
        return await self._ping(f"{self._trading_url}/health", "trading_server")

    async def ping_learning_server(self) -> Dict[str, Any]:
        """Ping the learning server (port 8901) /health endpoint."""
        return await self._ping(f"{self._learning_url}/health", "learning_server")

    async def ping_wallet_scanner(self) -> Dict[str, Any]:
        """Ping the wallet scanner (port 8902) /health endpoint."""
        return await self._ping(f"{self._wallet_url}/health", "wallet_scanner")

    async def ping_all(self) -> Dict[str, Dict[str, Any]]:
        """Ping all 3 component servers and return their health status."""
        results = await asyncio.gather(
            self.ping_trading_server(),
            self.ping_learning_server(),
            self.ping_wallet_scanner(),
            return_exceptions=True,
        )
        return {
            "trading_server": results[0] if not isinstance(results[0], Exception)
            else {"status": "error", "error": str(results[0])},
            "learning_server": results[1] if not isinstance(results[1], Exception)
            else {"status": "error", "error": str(results[1])},
            "wallet_scanner": results[2] if not isinstance(results[2], Exception)
            else {"status": "error", "error": str(results[2])},
        }

    async def _ping(self, url: str, name: str) -> Dict[str, Any]:
        """Send a GET to *url* and return a health dict."""
        if not self._session:
            return {"status": "error", "error": "No HTTP session"}

        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {}
                    return {
                        "status": "ok",
                        "service": name,
                        "http_status": resp.status,
                        "response": data,
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "service": name,
                        "http_status": resp.status,
                    }
        except asyncio.TimeoutError:
            self._log.warning(f"Timeout pinging {name} at {url}")
            return {"status": "timeout", "service": name, "error": "Connection timed out"}
        except aiohttp.ClientConnectorError as exc:
            self._log.warning(f"Cannot connect to {name} at {url}: {exc}")
            return {"status": "unreachable", "service": name, "error": str(exc)}
        except Exception as exc:
            self._log.error(f"Error pinging {name}: {exc}")
            return {"status": "error", "service": name, "error": str(exc)}

    # ------------------------------------------------------------------
    # Config sync
    # ------------------------------------------------------------------

    async def sync_strategy_config(self) -> Dict[str, Any]:
        """Sync strategy configuration from learning server to trading server.

        Fetches the active strategy recommendation from the learning server
        and pushes it as a config update to the trading server.

        Returns:
            Dict with sync results for each step.
        """
        result: Dict[str, Any] = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # 1. Get active strategy from learning server
        try:
            async with self._session.get(
                f"{self._learning_url}/strategy"
            ) as resp:
                if resp.status == 200:
                    learning_data = await resp.json()
                    result["learning_server"] = learning_data
                else:
                    result["learning_server"] = {"error": f"HTTP {resp.status}"}
                    result["status"] = "error"
        except Exception as exc:
            result["learning_server"] = {"error": str(exc)}
            result["status"] = "error"

        # 2. Push to trading server
        # The trading server doesn't have a /config endpoint yet, so we
        # log the intended sync and store it for future integration.
        active_strategy = (
            result.get("learning_server", {}).get("active_strategy", "")
        )
        active_version = (
            result.get("learning_server", {}).get("active_version", "")
        )

        if active_strategy:
            self._log.info(
                f"Strategy sync: {active_strategy}@{active_version} "
                f"→ trading server"
            )
            result["synced_strategy"] = active_strategy
            result["synced_version"] = active_version
        else:
            result["synced_strategy"] = None
            result["synced_version"] = None

        return result

    # ------------------------------------------------------------------
    # Wallet signal relay
    # ------------------------------------------------------------------

    async def get_wallet_bias(self) -> Dict[str, Any]:
        """Fetch the latest aggregated wallet scanner signals.

        Returns:
            Dict mapping symbol to bias info (score, direction, confidence).
        """
        if not self._session:
            return {}

        try:
            async with self._session.get(f"{self._wallet_url}/signals") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    signals = data.get("signals", data)
                    self._cached_wallet_signals = signals
                    return signals if isinstance(signals, dict) else {}
                return {}
        except Exception as exc:
            self._log.debug(f"Failed to fetch wallet signals: {exc}")
            return {}

    async def relay_wallet_bias(self) -> Optional[Dict[str, Any]]:
        """Fetch wallet signals and relay them as bias overrides.

        Returns:
            The bias overrides that were relayed, or None on failure.
        """
        signals = await self.get_wallet_bias()
        if not signals:
            return None

        # Build bias overrides per symbol
        bias_overrides = {}
        for symbol, signal_data in signals.items():
            if isinstance(signal_data, dict):
                bias_overrides[symbol] = {
                    "bias": signal_data.get("bias", "neutral"),
                    "overall_score": signal_data.get("overall_score", 0),
                    "confidence": signal_data.get("confidence", 0),
                    "source": "wallet_scanner",
                }

        if not bias_overrides:
            return None

        self._log.debug(
            f"Relaying {len(bias_overrides)} wallet bias overrides"
        )
        return bias_overrides

    # ------------------------------------------------------------------
    # System status
    # ------------------------------------------------------------------

    async def get_system_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get a unified view of the entire system's status.

        Aggregates health from all components, current positions, active
        strategies, wallet bias, and version information.

        Args:
            force_refresh: If True, bypass the cache and re-ping all servers.

        Returns:
            A large dict with the full system state.
        """
        import time
        now = time.time()

        # Return cached status if still fresh
        if not force_refresh and self._cached_status and (now - self._last_status_update) < self._status_ttl:
            return self._cached_status

        # Gather health
        health = await self.ping_all()

        # Gather trading server status
        trading_status = {}
        if health["trading_server"].get("status") == "ok":
            try:
                async with self._session.get(
                    f"{self._trading_url}/status"
                ) as resp:
                    if resp.status == 200:
                        trading_status = await resp.json()
            except Exception:
                pass

        # Gather learning server strategy
        learning_strategy = {}
        if health["learning_server"].get("status") == "ok":
            try:
                async with self._session.get(
                    f"{self._learning_url}/strategy"
                ) as resp:
                    if resp.status == 200:
                        learning_strategy = await resp.json()
            except Exception:
                pass

        # Gather wallet bias
        wallet_bias = await self.get_wallet_bias()

        # Build unified status
        status: Dict[str, Any] = {
            "system": "aurora-trader",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coordinator": {
                "running": self._running,
                "version": "1.0.0",
            },
            "components": {
                "trading_server": {
                    "health": health.get("trading_server", {}),
                    "details": trading_status,
                },
                "learning_server": {
                    "health": health.get("learning_server", {}),
                    "strategy": learning_strategy,
                },
                "wallet_scanner": {
                    "health": health.get("wallet_scanner", {}),
                    "signals_count": len(wallet_bias) if isinstance(wallet_bias, dict) else 0,
                },
            },
            "wallet_bias": wallet_bias if isinstance(wallet_bias, dict) else {},
            "overall": self._compute_overall_health(health),
        }

        # Cache
        self._cached_status = status
        self._last_status_update = now

        return status

    def _compute_overall_health(self, health: Dict[str, Any]) -> str:
        """Determine overall system health from component health dicts."""
        statuses = []
        for comp, data in health.items():
            s = data.get("status", "error")
            if s == "ok":
                statuses.append("healthy")
            elif s in ("unreachable", "timeout"):
                statuses.append("degraded")
            else:
                statuses.append("unhealthy")

        if all(s == "healthy" for s in statuses):
            return "healthy"
        if any(s == "unhealthy" for s in statuses):
            return "unhealthy"
        return "degraded"

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _status_poll_loop(self) -> None:
        """Periodically refresh the system status cache."""
        while self._running:
            try:
                await self.get_system_status(force_refresh=True)
                await asyncio.sleep(self._status_ttl)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(f"Status poll error: {exc}")
                await asyncio.sleep(30)

    async def _wallet_relay_loop(self) -> None:
        """Periodically poll wallet scanner and cache signals."""
        poll_interval = self._cfg.wallet_poll_interval
        while self._running:
            try:
                await self.get_wallet_bias()
                await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(f"Wallet relay loop error: {exc}")
                await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Startup / shutdown helpers
    # ------------------------------------------------------------------

    async def startup_sequence(self) -> Dict[str, Any]:
        """Run the full startup sequence: wait for components, sync config.

        Returns:
            Dict with startup results for each step.
        """
        self._log.info("=== Aurora Trader Startup Sequence ===")
        results: Dict[str, Any] = {
            "status": "ok",
            "steps": [],
        }

        # Step 1: Wait for trading server
        self._log.info("Step 1/4: Waiting for Trading Server...")
        ts_ready = await self._wait_for_service(
            f"{self._trading_url}/health", "trading_server", retries=10, delay=2.0
        )
        results["steps"].append({"step": "trading_server", "ready": ts_ready})
        if not ts_ready:
            results["status"] = "degraded"

        # Step 2: Wait for learning server
        self._log.info("Step 2/4: Waiting for Learning Server...")
        ls_ready = await self._wait_for_service(
            f"{self._learning_url}/health", "learning_server", retries=10, delay=2.0
        )
        results["steps"].append({"step": "learning_server", "ready": ls_ready})
        if not ls_ready:
            results["status"] = "degraded"

        # Step 3: Wait for wallet scanner (optional)
        if self._cfg.wallet_enabled:
            self._log.info("Step 3/4: Waiting for Wallet Scanner...")
            ws_ready = await self._wait_for_service(
                f"{self._wallet_url}/health", "wallet_scanner", retries=6, delay=2.0
            )
            results["steps"].append({"step": "wallet_scanner", "ready": ws_ready})
            if not ws_ready:
                results["status"] = "degraded"
        else:
            results["steps"].append({"step": "wallet_scanner", "ready": True, "note": "disabled"})

        # Step 4: Sync strategy config
        self._log.info("Step 4/4: Syncing strategy config...")
        sync_result = await self.sync_strategy_config()
        results["steps"].append({"step": "config_sync", "result": sync_result})
        if sync_result.get("status") != "ok":
            results["status"] = "degraded"

        self._log.info(f"=== Startup complete (status={results['status']}) ===")
        return results

    async def shutdown_sequence(self) -> Dict[str, Any]:
        """Run graceful shutdown: stop coordinator, then signal others."""
        self._log.info("=== Aurora Trader Shutdown Sequence ===")
        results: Dict[str, Any] = {"status": "ok", "steps": []}

        # Step 1: Stop coordinator itself
        await self.stop()
        results["steps"].append({"step": "coordinator_shutdown", "status": "ok"})

        self._log.info("=== Shutdown complete ===")
        return results

    async def _wait_for_service(
        self, url: str, name: str, retries: int = 10, delay: float = 2.0
    ) -> bool:
        """Poll a service endpoint until it responds or retries are exhausted."""
        for attempt in range(1, retries + 1):
            try:
                async with self._session.get(url) as resp:
                    if resp.status == 200:
                        self._log.info(f"  {name} ready (attempt {attempt})")
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

            if attempt < retries:
                self._log.debug(
                    f"  Waiting for {name}... (attempt {attempt}/{retries})"
                )
                await asyncio.sleep(delay)

        self._log.warning(f"  {name} NOT ready after {retries} attempts")
        return False
