"""
Aurora Trader — Shared Module.

Central configuration, models, logging, and constants used across all
Aurora Trader sub-systems (trading_server, learning_server,
wallet_scanner, integration).
"""

from shared.config import Config, load_config
from shared.models import (
    Trade,
    Signal,
    MarketRegime,
    Position,
    WalletSignal,
    StrategyVersion,
)
from shared.logger import get_logger, AuroraLogger
from shared.constants import (
    TIMEFRAMES,
    INDICATOR_DEFAULTS,
    EXCHANGE_CONFIGS,
    WALLET_SCANNER_ENDPOINTS,
    RISK_LIMITS,
    STRATEGY_DEFAULTS,
)

__all__ = [
    # config
    "Config",
    "load_config",
    # models
    "Trade",
    "Signal",
    "MarketRegime",
    "Position",
    "WalletSignal",
    "StrategyVersion",
    # logger
    "get_logger",
    "AuroraLogger",
    # constants
    "TIMEFRAMES",
    "INDICATOR_DEFAULTS",
    "EXCHANGE_CONFIGS",
    "WALLET_SCANNER_ENDPOINTS",
    "RISK_LIMITS",
    "STRATEGY_DEFAULTS",
]
