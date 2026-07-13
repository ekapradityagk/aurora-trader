"""Aurora Trader — Trading Strategies.

Plug-in strategy base class and concrete implementations:
- Mean Reversion (Bollinger Bands + RSI)
- RSI Divergence + SMC
- Trend Following (Supertrend + EMA)
"""

from trading_server.strategies.base import BaseStrategy
from trading_server.strategies.mean_reversion import MeanReversionStrategy
from trading_server.strategies.rsi_divergence import RsiDivergenceStrategy
from trading_server.strategies.trend_follow import TrendFollowStrategy

__all__ = [
    "BaseStrategy",
    "MeanReversionStrategy",
    "RsiDivergenceStrategy",
    "TrendFollowStrategy",
]
