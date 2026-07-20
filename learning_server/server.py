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
import aiohttp

from shared.config import load_config
from shared.logger import get_logger
from shared.models import MarketRegimeType

from learning_server.hyperopt import HyperoptOptimizer, OptimizationResult
from learning_server.regime import RegimeDetector, RegimeResult
from learning_server.analyzer import TradeAnalyzer, AnalysisReport
from learning_server.strategy_selector import StrategySelector, SelectionRecord
from learning_server.pair_ranker import PairRanker
from learning_server.suitability_scorer import SuitabilityScorer, SuitabilityReport
from learning_server.shadow_analyzer import ShadowAnalyzer, ShadowReport
from learning_server.opportunity_spotter import OpportunitySpotter, ScanResult

logger = get_logger("learning_server.server")

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8901
HYPEROPT_INTERVAL_HOURS = 168  # 7 days (weekly)


# ---------------------------------------------------------------------------
# Middleware (module-level, aiohttp compatible)
# ---------------------------------------------------------------------------


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
        self._suitability_scorer = SuitabilityScorer()
        self._shadow_analyzer = ShadowAnalyzer()
        self._opportunity_spotter = OpportunitySpotter()

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
        self._last_suitability: Optional[Dict[str, Any]] = None

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
            self._tasks.add(
                asyncio.create_task(self._signal_executor_loop())
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
        self._app.router.add_get("/api/pair-suitability", self._handle_pair_suitability)
        self._app.router.add_post("/api/run-suitability", self._handle_run_suitability)
        self._app.router.add_get("/api/shadow-analysis", self._handle_shadow_analysis)
        self._app.router.add_get("/api/opportunities", self._handle_opportunities)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

    # ------------------------------------------------------------------
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
            # Fetch OHLCV data from Binance public API
            limit = 100  # enough for indicators
            import aiohttp as _
            binance_url = (
                f"https://api.binance.com/api/v3/klines"
                f"?symbol={symbol}&interval={timeframe}&limit={limit}"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(binance_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Binance API returned HTTP {resp.status}")
                    raw = await resp.json()

            # Convert to OHLCV dict format expected by regime detector
            ohlcv = []
            for k in raw:
                ohlcv.append({
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })

            result = await self._regime_detector.detect(
                symbol=symbol,
                timeframe=timeframe,
                ohlcv=ohlcv,
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

    async def _handle_pair_suitability(self, request: web.Request) -> web.Response:
        """GET /api/pair-suitability — get cached pair suitability scores.

        Returns the last scored report, or an empty result if never run.
        """
        if self._last_suitability:
            return web.json_response(self._last_suitability)
        return web.json_response({
            "pairs": [],
            "top_picks": [],
            "scan_timestamp": "",
            "total_scored": 0,
            "message": "No suitability scan has been run yet. POST /api/run-suitability to trigger.",
        })

    async def _handle_run_suitability(self, request: web.Request) -> web.Response:
        """POST /api/run-suitability — trigger a suitability scan now."""
        try:
            report = await self._suitability_scorer.score_universe()

            self._last_suitability = self._suitability_to_dict(report)

            # Auto-rotate if we have top picks
            if report.top_picks:
                await self._apply_rotation(report.top_picks)

            return web.json_response(self._last_suitability)
        except Exception as exc:
            self._log.error(f"Suitability scan failed: {exc}", exc_info=True)
            return web.json_response(
                {"error": "Suitability scan failed", "detail": str(exc)},
                status=500,
            )

    def _suitability_to_dict(self, report: SuitabilityReport) -> Dict[str, Any]:
        """Convert a SuitabilityReport to a JSON-friendly dict."""
        return {
            "total_scored": report.total_scored,
            "scan_timestamp": report.scan_timestamp,
            "top_picks": report.top_picks,
            "errors": report.errors,
            "pairs": [
                {
                    "symbol": p.symbol,
                    "composite_score": p.composite_score,
                    "movement_score": p.movement_score,
                    "trend_score": p.trend_score,
                    "volume_score": p.volume_score,
                    "funding_score": p.funding_score,
                    "smoothness_score": p.smoothness_score,
                    "atr_pct": p.atr_pct,
                    "adx": p.adx,
                    "volume_cv": p.volume_cv,
                    "funding_rate_annualised": p.funding_rate_annualised,
                    "avg_body_ratio": p.avg_body_ratio,
                    "avg_volume_usdt": p.avg_volume_usdt,
                    "close_price": p.close_price,
                    "regime_label": p.regime_label,
                }
                for p in report.pairs
            ],
        }

    async def _apply_rotation(self, top_picks: List[str]) -> None:
        """Update the trading server's active symbol list with top picks."""
        if not top_picks:
            return

        # Update config.yaml trading_server.symbols
        import yaml

        cfg_path = Path("config.yaml")
        if not cfg_path.is_file():
            self._log.warning("config.yaml not found — skipping rotation")
            return

        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

            # Update symbols
            cfg.setdefault("trading_server", {})["symbols"] = top_picks

            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

            self._log.info(
                f"Auto-rotation applied — active symbols: {', '.join(top_picks)}"
            )

            # Hot-reload the trading server without restart
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as session:
                    async with session.post(
                        "http://127.0.0.1:8900/api/reload-symbols"
                    ) as resp:
                        if resp.status == 200:
                            self._log.info("Trading server symbols reloaded ✅")
                        else:
                            self._log.warning(
                                f"Trading server reload returned HTTP {resp.status}"
                            )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                self._log.warning(f"Trading server reload failed: {exc}")
        except Exception as exc:
            self._log.error(f"Failed to write config.yaml for rotation: {exc}")

    async def _handle_opportunities(self, request: web.Request) -> web.Response:
        """GET /api/opportunities — scan for trading opportunities now."""
        try:
            symbols = self._cfg.data.get("trading_server", {}).get("symbols", [])
            result = await self._opportunity_spotter.scan(symbols=symbols)

            return web.json_response({
                "timestamp": result.timestamp,
                "total_scanned": result.total_scanned,
                "hot_list": result.hot_list,
                "watch_list": result.watch_list,
                "opportunities": [
                    {
                        "symbol": o.symbol,
                        "direction": o.direction,
                        "confidence": o.confidence,
                        "primary_timeframe": o.primary_timeframe,
                        "entry_notes": o.entry_notes,
                        "price": o.price,
                        "brewing": o.brewing,
                        "timeframes": {
                            tf: {
                                "direction": s.direction,
                                "confidence": s.confidence,
                                "rsi": s.rsi,
                                "bb_position": s.bb_position,
                                "ema_position": s.ema_position,
                                "adx": s.adx,
                                "reasons": s.reasons,
                            }
                            for tf, s in o.timeframes.items()
                        },
                    }
                    for o in result.opportunities
                ],
                "errors": result.errors[:5],
            })
        except Exception as exc:
            self._log.error(f"Opportunity scan failed: {exc}", exc_info=True)
            return web.json_response(
                {"error": "Opportunity scan failed", "detail": str(exc)},
                status=500,
            )

    async def _handle_shadow_analysis(self, request: web.Request) -> web.Response:
        """GET /api/shadow-analysis — run shadow analysis on recent trades.

        Query params:
            window_days (int, default 30): look-back window
            min_trades (int, default 3): minimum trades per strategy
        """
        window_days = int(request.query.get("window_days", "30"))
        min_trades = int(request.query.get("min_trades", "3"))

        try:
            report = await self._shadow_analyzer.analyze(
                window_days=window_days,
                min_trades=min_trades,
            )

            return web.json_response({
                "timestamp": report.timestamp,
                "total_trades_analyzed": report.total_trades_analyzed,
                "time_period_days": report.time_period_days,
                "strategies": {
                    name: {
                        "total_trades": p.total_trades,
                        "wins": p.wins,
                        "losses": p.losses,
                        "win_rate": p.win_rate,
                        "total_pnl": p.total_pnl,
                        "avg_pnl": p.avg_pnl,
                        "profit_factor": p.profit_factor,
                        "avg_rrr": p.avg_rrr,
                        "sharpe": p.sharpe,
                        "max_drawdown": p.max_drawdown,
                        "avg_holding_hours": p.avg_holding_hours,
                        "win_avg_holding_hours": p.win_avg_holding_hours,
                        "loss_avg_holding_hours": p.loss_avg_holding_hours,
                        "entry_reasons": p.entry_reasons,
                    }
                    for name, p in report.strategy_profiles.items()
                },
                "biases": {
                    "overall_health": report.bias_report.overall_health,
                    "disposition_effect": report.bias_report.disposition_effect,
                    "overtrading": report.bias_report.overtrading,
                    "chase_entries": report.bias_report.chase_entries,
                    "anchoring": report.bias_report.anchoring,
                },
                "entry_timing": report.entry_timing_analysis,
                "exit_timing": report.exit_timing_analysis,
                "top_performers": report.top_performers,
                "worst_performers": report.worst_performers,
                "recommendations": report.recommendations,
            })
        except Exception as exc:
            self._log.error(f"Shadow analysis failed: {exc}", exc_info=True)
            return web.json_response(
                {"error": "Shadow analysis failed", "detail": str(exc)},
                status=500,
            )

    # ------------------------------------------------------------------
    # Background scheduler
    # ------------------------------------------------------------------

    async def _signal_executor_loop(self) -> None:
        """Background loop: scan for high-confidence signals and auto-execute.

        Runs every 30 minutes. Scans active pairs, checks for signals
        with confidence >= 70% (not brewing), and forwards to the
        trading server for execution.
        """
        EXECUTOR_INTERVAL = 1800  # 30 minutes
        TRADING_EXECUTE_URL = "http://127.0.0.1:8900/api/execute"
        MIN_CONFIDENCE = 70

        await asyncio.sleep(60)  # Initial delay — let everything start up

        while self._running:
            try:
                symbols = self._cfg.data.get("trading_server", {}).get("symbols", [])
                if not symbols:
                    await asyncio.sleep(EXECUTOR_INTERVAL)
                    continue

                result = await self._opportunity_spotter.scan(symbols=symbols)

                # Persist opportunity scan results to shared DB
                try:
                    import sqlite3, json
                    from datetime import datetime, timezone
                    from pathlib import Path
                    scan_time = result.timestamp or datetime.now(timezone.utc).isoformat()
                    db_path = Path(__file__).resolve().parent.parent / "data" / "trading.db"
                    conn = sqlite3.connect(str(db_path))
                    conn.execute("PRAGMA journal_mode=WAL")
                    for opp in result.opportunities:
                        conn.execute(
                            "INSERT INTO opportunity_scans (scan_time, symbol, direction, confidence, primary_timeframe, entry_notes, price, brewing, raw_json) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (scan_time, opp.symbol, opp.direction, opp.confidence,
                             opp.primary_timeframe, opp.entry_notes, opp.price,
                             int(opp.brewing), json.dumps({
                                 "timeframes": {
                                     tf: {
                                         "direction": s.direction,
                                         "confidence": s.confidence,
                                         "rsi": s.rsi,
                                         "bb_position": s.bb_position,
                                         "ema_position": s.ema_position,
                                         "adx": s.adx,
                                         "reasons": s.reasons,
                                     }
                                     for tf, s in opp.timeframes.items()
                                 }
                             })),
                        )
                    conn.commit()
                    conn.close()
                    self._log.debug(f"Persisted {len(result.opportunities)} scan results")
                except Exception as exc:
                    self._log.debug(f"Failed to persist scan results: {exc}")

                # Find the best non-brewing signal
                best = None
                for opp in result.opportunities:
                    if (
                        not opp.brewing
                        and opp.confidence >= MIN_CONFIDENCE
                        and opp.direction in ("LONG", "SHORT")
                    ):
                        if best is None or opp.confidence > best.confidence:
                            best = opp

                if best is not None:
                    side = "BUY" if best.direction == "LONG" else "SELL"
                    payload = {
                        "symbol": best.symbol,
                        "side": side,
                        "direction": best.direction,
                        "confidence": best.confidence,
                        "reason": best.entry_notes,
                        "price": best.price,
                    }

                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as session:
                        try:
                            async with session.post(
                                TRADING_EXECUTE_URL, json=payload
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    self._log.info(
                                        f"🟢 AUTO-EXECUTED {best.direction} {best.symbol} "
                                        f"({best.confidence}%) → order {data.get('order_id', '?')}"
                                    )
                                else:
                                    err = await resp.text()
                                    self._log.warning(
                                        f"⚠️ Execute rejected for {best.symbol}: "
                                        f"HTTP {resp.status} — {err[:200]}"
                                    )
                        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                            self._log.warning(
                                f"⚠️ Execute failed for {best.symbol}: {exc}"
                            )
                else:
                    self._log.debug(
                        "Signal executor: no high-confidence signals found"
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error(
                    f"Signal executor error: {exc}", exc_info=True
                )

            # Wait for next cycle (check every 30s if stopped)
            for _ in range(EXECUTOR_INTERVAL // 30):
                if not self._running:
                    break
                await asyncio.sleep(30)

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
