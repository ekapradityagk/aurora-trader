"""
Aurora Trader — Learning Server.

The learning server is a standalone async process that runs weekly hyperopt
optimization, detects market regimes, and analyzes trade history to improve
strategy parameters.

Components:
    server          — aiohttp-based HTTP server exposing /health, /optimize,
                      /regime, /analysis endpoints
    hyperopt        — Bayesian hyper-parameter optimisation via Optuna (TPE)
    regime          — Market regime detector (TREND / RANGE / VOLATILE)
    analyzer        — Trade history analysis per strategy
    strategy_selector — Regime + performance → strategy selection logic
"""

from learning_server.hyperopt import HyperoptOptimizer
from learning_server.regime import RegimeDetector
from learning_server.analyzer import TradeAnalyzer
from learning_server.strategy_selector import StrategySelector

__all__ = [
    "HyperoptOptimizer",
    "RegimeDetector",
    "TradeAnalyzer",
    "StrategySelector",
]
