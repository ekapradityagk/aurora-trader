"""
Aurora Trader — Learning Server.

A standalone async HTTP server that:
    - Exposes /health, /optimize, /regime, /analysis endpoints
    - Runs weekly hyper-parameter optimisation (Optuna) in background
    - Manages strategy versions based on optimisation results
    - Saves optimised parameters to JSON for the trading server to pick up
    - Detects market regime and selects the best strategy
"""

from __future__ import annotations

import asyncio
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from aiohttp import web

from shared.config import load_config
from shared.logger import get_logger
from shared.models import MarketRegimeType

from learning_server.hyperopt import HyperoptOptimizer, OptimizationResult
from learning_server.regime import RegimeDetector, RegimeResult
from learning_server.analyzer import TradeAnalyzer, AnalysisReport
from learning_server.strategy_selector import StrategySelector, SelectionRecord
from learning_server.pair_ranker import PairRanker

logger = get_logger("learning_server.server")

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8901
HYPEROPT_INTERVAL_HOURS = 168  # 7 days (weekly)


# ---------------------------------------------------------------------------
# Learning Server
# ---------------------------------------------------------------------------


class LearningServer:
    """Main learning server: HTTP API + background scheduler.

    Usage::

        server = LearningServer()
        await server.start()
        # runs until Ctrl+C
        await server.stop()
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._host = host
        self._port = port

        # Config
        self._cfg = load_config()
        ls_cfg = self._cfg.data.get("learning_server", {})
        self._enabled = ls_cfg.get("enabled", True)

        # Sub-components
        self._optimizer = HyperoptOptimizer()
        self._regime_detector = RegimeDetector()
        self._analyzer = TradeAnalyzer()
        self._selector = StrategySelector()
        self._pair_ranker = PairRanker()

        # HTTP server state
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._running = False

        # Background tasks
        self._tasks: Set[asyncio.Task] = set()

        # Last optimisation result (cached for /health)
        self._last_optimization: Optional[OptimizationResult] = None
        self._last_analysis: Optional[AnalysisReport] = None
        self._last_regime: Optional[RegimeResult] = None

        # Scheduled hyperopt interval
        self._hyperopt_interval_hours = ls_cfg.get(
            "hyperopt_interval_hours", HYPEROPT_INTERVAL_HOURS
        )

        # Concurrency
        self._lock = asyncio.Lock()

        # Logging
        self._log = logger

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise all components and start the event loop."""
        if self._running:
            self._log.warning("Server already running")
            return

        self._log.info("Learning Server starting...")

        try:
            # 1. Start HTTP server
            await self._start_http_server()
            self._log.info(
                f"HTTP server listening on {self._host}:{self._port}"
            )

            # 2. Start background tasks
            self._running = True
            self._tasks.add(
                asyncio.create_task(self._hyperopt_scheduler_loop())
            )
            self._log.info(
                f"Hyperopt scheduler started "
                f"(interval={self._hyperopt_interval_hours}h)"
            )

            # 3. Run an initial optimisation if never done
            if self._optimizer.load_best_params() is None:
                self._log.info("No previous optimisation found — running initial...")
                self._tasks.add(
                    asyncio.create_task(self._run_initial_optimization())
                )

            self._log.info("Learning Server started successfully")

        except Exception as exc:
            self._log.critical(
                f"Failed to start server: {exc}", exc_info=True
            )
            await self.stop()
            raise

    async def stop(self) -> None:
        """Gracefully shut down the server and all components."""
        self._log.info("Learning Server shutting down...")
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

        self._log.info("Learning Server stopped")

    # ------------------------------------------------------------------
    # HTTP Server
    # ------------------------------------------------------------------

    async def _start_http_server(self) -> None:
        """Configure and start the aiohttp web server."""
        self._app = web.Application()

        # Middleware
        self._app.middlewares.append(_cors_middleware)
        self._app.middlewares.append(_error_middleware)

        # Routes
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_post("/optimize", self._handle_optimize)
        self._app.router.add_get("/regime", self._handle_regime)
        self._app.router.add_get("/analysis", self._handle_analysis)
        self._app.router.add_get("/strategy", self._handle_strategy)
        self._app.router.add_post("/strategy/select", self._handle_strategy_select)
        self._app.router.add_get("/selections", self._handle_selections)
        self._app.router.add_get("/api/pair-rankings", self._handle_pair_rankings)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

    # ------------------------------------------------------------------
    # Middleware (module-level for aiohttp compatibility)
    # ------------------------------------------------------------------


@web.middleware
async def _cors_middleware(
    request: web.Request, handler: Any
) -> web.StreamResponse:
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
async def _error_middleware(
    request: web.Request, handler: Any
) -> web.StreamResponse:
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
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: web.Request) -> web.Response:
        """GET /health — server status and component health."""
        health = {
            "status": "ok",
            "server": "learning_server",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "components": {
                "hyperopt": self._optimizer.load_best_params() is not None,
                "regime_detector": True,
                "analyzer": True,
                "strategy_selector": True,
            },
            "active_strategy": self._selector.get_active_strategy()[0],
            "active_version": self._selector.get_active_strategy()[1],
        }

        if self._last_optimization:
            health["last_optimization"] = {
                "timestamp": self._last_optimization.timestamp,
                "test_sharpe": self._last_optimization.test_sharpe,
                "mc_p_value": self._last_optimization.mc_p_value,
            }

        return web.json_response(health)

    async def _handle_optimize(self, request: web.Request) -> web.Response:
        """POST /optimize — trigger hyper-parameter optimisation.

        Optional JSON body:
            {"n_trials": 200, "timeout": 7200}
        """
        body: Dict[str, Any] = {}
        try:
            if request.can_read_body:
                body = await request.json()
        except (json.JSONDecodeError, ValueError):
            pass

        n_trials = body.get("n_trials")
        timeout = body.get("timeout")

        self._log.info("Manual optimisation triggered via API")

        try:
            result = await self._optimizer.run(
                n_trials=n_trials,
                timeout=timeout,
            )
            self._last_optimization = result

            # If optimisation succeeded, update strategy version
            if result.params:
                self._selector.update_params_version()

            return web.json_response({
                "status": "completed" if result.params else "skipped",
                "version_tag": result.version_tag,
                "params": result.params,
                "train_sharpe": result.train_sharpe,
                "test_sharpe": result.test_sharpe,
                "mc_p_value": result.mc_p_value,
                "n_trials": result.n_trials,
                "timestamp": result.timestamp,
            })
        except Exception as exc:
            self._log.error(f"Optimisation failed: {exc}")
            return web.json_response(
                {"error": "Optimisation failed", "detail": str(exc)},
                status=500,
            )

    async def _handle_regime(self, request: web.Request) -> web.Response:
        """GET /regime — get current market regime classification.

        Query params:
            symbol (optional, default: BTCUSDT)
            timeframe (optional, default: 1h)
        """
        symbol = request.query.get("symbol", "BTCUSDT")
        timeframe = request.query.get("timeframe", "1h")

        try:
            # Attempt to detect from cache; if no cached data exists,
            # we return a fallback indicating no data
            result = await self._regime_detector.detect(
                symbol=symbol,
                timeframe=timeframe,
            )
            self._last_regime = result

            # Also run strategy selection based on this regime
            selection = await self._selector.select(
                symbol=symbol,
                regime_result=result,
            )

            return web.json_response({
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": result.regime.value,
                "confidence": result.confidence,
                "scores": result.scores,
                "indicators": result.indicators,
                "recommended_strategy": selection.selected_strategy,
                "strategy_version": selection.version_tag,
            })
        except Exception as exc:
            # Return fallback if no data available
            active_strategy, active_version = self._selector.get_active_strategy()
            return web.json_response({
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": "unknown",
                "confidence": 0.0,
                "error": str(exc),
                "recommended_strategy": active_strategy,
                "strategy_version": active_version,
            })

    async def _handle_analysis(self, request: web.Request) -> web.Response:
        """GET /analysis — get trade history analysis.

        Query params:
            strategy (optional) — filter to a single strategy
        """
        strategy_filter = request.query.get("strategy")

        try:
            if strategy_filter:
                metrics = await self._analyzer.get_strategy_metrics(
                    strategy_filter
                )
                if metrics is None:
                    return web.json_response(
                        {"error": f"Strategy '{strategy_filter}' not found"},
                        status=404,
                    )
                return web.json_response({
                    "strategy": strategy_filter,
                    "metrics": {
                        "total_trades": metrics.total_trades,
                        "wins": metrics.wins,
                        "losses": metrics.losses,
                        "win_rate": metrics.win_rate,
                        "profit_factor": metrics.profit_factor,
                        "sharpe_ratio": metrics.sharpe_ratio,
                        "avg_risk_reward": metrics.avg_risk_reward,
                        "total_pnl": metrics.total_pnl,
                        "max_drawdown": metrics.max_drawdown,
                        "current_rolling_win_rate": metrics.current_rolling_win_rate,
                        "is_underperforming": metrics.is_underperforming,
                        "recommendations": metrics.recommendations,
                    },
                })
            else:
                report = await self._analyzer.analyze()
                self._last_analysis = report

                strategies = {}
                for name, m in report.strategies.items():
                    strategies[name] = {
                        "total_trades": m.total_trades,
                        "wins": m.wins,
                        "losses": m.losses,
                        "win_rate": m.win_rate,
                        "profit_factor": m.profit_factor,
                        "sharpe_ratio": m.sharpe_ratio,
                        "avg_risk_reward": m.avg_risk_reward,
                        "total_pnl": m.total_pnl,
                        "max_drawdown": m.max_drawdown,
                        "current_rolling_win_rate": m.current_rolling_win_rate,
                        "is_underperforming": m.is_underperforming,
                        "recommendations": m.recommendations,
                    }

                return web.json_response({
                    "timestamp": report.timestamp,
                    "total_trades": report.total_trades,
                    "strategy_count": report.strategy_count,
                    "global_win_rate": report.global_win_rate,
                    "global_profit_factor": report.global_profit_factor,
                    "global_sharpe": report.global_sharpe,
                    "underperforming_strategies": report.underperforming_strategies,
                    "strategies": strategies,
                })
        except Exception as exc:
            self._log.error(f"Analysis failed: {exc}")
            return web.json_response(
                {"error": "Analysis failed", "detail": str(exc)},
                status=500,
            )

    async def _handle_strategy(self, request: web.Request) -> web.Response:
        """GET /strategy — get current active strategy and version."""
        strategy, version = self._selector.get_active_strategy()
        return web.json_response({
            "active_strategy": strategy,
            "active_version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def _handle_strategy_select(
        self, request: web.Request
    ) -> web.Response:
        """POST /strategy/select — manually trigger strategy selection.

        Optional JSON body:
            {"symbol": "ETHUSDT"}
        """
        body: Dict[str, Any] = {}
        try:
            if request.can_read_body:
                body = await request.json()
        except (json.JSONDecodeError, ValueError):
            pass

        symbol = body.get("symbol", "BTCUSDT")

        try:
            selection = await self._selector.select(symbol=symbol)
            return web.json_response({
                "selected_strategy": selection.selected_strategy,
                "version_tag": selection.version_tag,
                "market_regime": selection.market_regime,
                "regime_confidence": selection.regime_confidence,
                "reason": selection.reason,
                "previous_strategy": selection.previous_strategy,
                "timestamp": selection.timestamp,
            })
        except Exception as exc:
            self._log.error(f"Strategy selection failed: {exc}")
            return web.json_response(
                {"error": "Strategy selection failed", "detail": str(exc)},
                status=500,
            )

    async def _handle_selections(self, request: web.Request) -> web.Response:
        """GET /selections — get recent strategy selection history."""
        limit = int(request.query.get("limit", 20))
        records = await self._selector.get_selection_history(limit=limit)
        return web.json_response({
            "count": len(records),
            "records": [
                {
                    "timestamp": r.timestamp,
                    "selected_strategy": r.selected_strategy,
                    "version_tag": r.version_tag,
                    "market_regime": r.market_regime,
                    "regime_confidence": r.regime_confidence,
                    "reason": r.reason,
                    "previous_strategy": r.previous_strategy,
                }
                for r in records
            ],
        })

    async def _handle_pair_rankings(self, request: web.Request) -> web.Response:
        """GET /api/pair-rankings — get ranked pair performance.

        Query params:
            window_days (int, default 7): look-back window
            min_trades (int, default 2): minimum trades to qualify
        """
        window_days = int(request.query.get("window_days", "7"))
        min_trades = int(request.query.get("min_trades", "2"))
        try:
            rankings = await self._pair_ranker.get_pair_rankings(
                window_days=window_days,
                min_trades=min_trades,
            )
            recommended = [r.symbol for r in rankings[:min(len(rankings), 6)]]
            retired = await self._pair_ranker.get_retired_pairs(
                window_days=window_days,
                min_trades=min_trades,
            )
            return web.json_response({
                "rankings": [
                    {
                        "symbol": r.symbol,
                        "total_trades": r.total_trades,
                        "wins": r.wins,
                        "losses": r.losses,
                        "total_pnl": r.total_pnl,
                        "win_rate": r.win_rate,
                        "avg_pnl": r.avg_pnl,
                        "profit_factor": r.profit_factor,
                        "sharpe": r.sharpe,
                        "score": r.score,
                        "trend": r.trend,
                    }
                    for r in rankings
                ],
                "recommended": recommended,
                "retired": retired,
                "window_days": window_days,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            self._log.error(f"Pair ranking failed: {exc}", exc_info=True)
            return web.json_response(
                {"error": "Pair ranking failed", "detail": str(exc)},
                status=500,
            )

    # ------------------------------------------------------------------
    # Background scheduler
    # ------------------------------------------------------------------

    async def _hyperopt_scheduler_loop(self) -> None:
        """Background loop that runs hyperopt on a weekly schedule."""
        while self._running:
            try:
                self._log.info(
                    f"Scheduled hyperopt starting "
                    f"(every {self._hyperopt_interval_hours}h)"
                )

                result = await self._optimizer.run()
                self._last_optimization = result

                if result.params:
                    self._log.info(
                        f"Hyperopt completed: Sharpe={result.test_sharpe:.4f}, "
                        f"p={result.mc_p_value:.4f}"
                    )
                    # Bump version
                    self._selector.update_params_version()

                    # Trigger strategy re-evaluation
                    try:
                        await self._selector.select()
                    except Exception as exc:
                        self._log.warning(
                            f"Strategy re-evaluation after hyperopt failed: {exc}"
                        )
                else:
                    self._log.info(
                        "Hyperopt skipped (no trades or no improvement)"
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(
                    f"Hyperopt scheduler error: {exc}", exc_info=True
                )

            # Wait for next interval
            for _ in range(self._hyperopt_interval_hours * 12):
                if not self._running:
                    break
                await asyncio.sleep(300)  # 5-minute granularity

    async def _run_initial_optimization(self) -> None:
        """Run a one-time optimisation on startup (if none exists)."""
        try:
            result = await self._optimizer.run()
            self._last_optimization = result
            if result.params:
                self._selector.update_params_version()
                self._log.info(
                    f"Initial optimisation complete: "
                    f"Sharpe={result.test_sharpe:.4f}"
                )
        except Exception as exc:
            self._log.warning(f"Initial optimisation failed: {exc}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the learning server.

    Handles SIGINT and SIGTERM for graceful shutdown.
    """
    server = LearningServer()

    async def start_and_wait() -> None:
        await server.start()
        # Keep running until stopped
        while server._running:
            await asyncio.sleep(1)

    def shutdown() -> None:
        nonlocal server
        if server._running:
            asyncio.create_task(server.stop())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Register signal handlers
    try:
        loop.add_signal_handler(signal.SIGINT, shutdown)
        loop.add_signal_handler(signal.SIGTERM, shutdown)
    except (NotImplementedError, RuntimeError):
        # Signal handlers not available on all platforms (e.g. Windows)
        pass

    try:
        loop.run_until_complete(start_and_wait())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.stop())
        loop.close()


if __name__ == "__main__":
    main()
