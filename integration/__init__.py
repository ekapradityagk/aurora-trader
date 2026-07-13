"""
Aurora Trader — Integration Layer.

The central orchestration and management layer that ties together:
  - Trading Server (port 8900)
  - Learning Server (port 8901)
  - Wallet Scanner (port 8902)

Provides version control for strategy parameters, winrate-based rollback,
system coordination, and a unified HTTP API server on port 8903.
"""

from integration.version_control import VersionController
from integration.winrate_db import WinrateDB
from integration.rollback import RollbackManager
from integration.coordinator import Coordinator
from integration.server import IntegrationServer

__all__ = [
    "VersionController",
    "WinrateDB",
    "RollbackManager",
    "Coordinator",
    "IntegrationServer",
]
