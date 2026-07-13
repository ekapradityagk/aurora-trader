"""Aurora Trader — Trading Server.

The core execution engine: connects to Binance via WebSocket and REST,
runs strategy analysis, manages positions and risk, and exposes HTTP
endpoints for monitoring and control.
"""

from trading_server.server import TradingServer

__all__ = ["TradingServer"]
