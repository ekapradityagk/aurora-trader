"""Aurora Trader — Risk Management.

Kelly-based position sizing, ATR-based stops, break-even and trailing
stop logic, and a circuit breaker that halts trading after a configurable
daily loss threshold.
"""

from trading_server.risk.manager import RiskManager
from trading_server.risk.circuit_breaker import CircuitBreaker

__all__ = ["RiskManager", "CircuitBreaker"]
