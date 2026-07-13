"""
Aurora Trader — Shared Data Models.

Pydantic models used across all sub-systems: trades, signals, market regime
classifications, positions, wallet on-chain signals, and strategy versioning
metadata.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class MarketRegimeType(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"
    PENDING = "pending"
    CANCELLED = "cancelled"


class TimeFrame(str, Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------


class Trade(BaseModel):
    """Represents a single executed trade (fill)."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    strategy_name: str
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    entry_price: Decimal
    exit_price: Optional[Decimal] = None
    quantity: Decimal
    quote_quantity: Decimal  # quote-currency notional
    leverage: int = 1
    pnl: Optional[Decimal] = None  # realised PnL after close
    pnl_pct: Optional[float] = None  # percentage return
    commission: Optional[Decimal] = None
    commission_asset: str = "USDT"
    entry_time: datetime
    exit_time: Optional[datetime] = None
    signal_id: Optional[str] = None  # reference to the originating signal
    timeframe: TimeFrame = TimeFrame.H1
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("quote_quantity", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        if isinstance(v, float):
            return Decimal(str(v))
        if isinstance(v, str):
            return Decimal(v)
        return v

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds()


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


class Signal(BaseModel):
    """A trading signal produced by a strategy."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    strategy_name: str
    symbol: str
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    price: Decimal
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timeframe: TimeFrame
    reason: str = ""
    indicators: Dict[str, Any] = Field(default_factory=dict)
    regime: Optional[MarketRegimeType] = None
    expiration: Optional[datetime] = None  # when this signal expires
    executed: bool = False
    trade_id: Optional[str] = None
    version: Optional[str] = None  # strategy version that produced this
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expiration is None:
            return False
        return datetime.utcnow() > self.expiration

    @property
    def is_valid(self) -> bool:
        return not self.is_expired and not self.executed and self.confidence >= 0.5


# ---------------------------------------------------------------------------
# Market Regime
# ---------------------------------------------------------------------------


class MarketRegime(BaseModel):
    """Market regime classification produced by the learning server."""

    symbol: str
    regime: MarketRegimeType
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timeframe: TimeFrame
    score: float = 0.0  # regime strength / intensity score
    indicators_used: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


class Position(BaseModel):
    """An open or closed position with full lifecycle details."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    strategy_name: str
    symbol: str
    side: OrderSide
    status: PositionStatus = PositionStatus.OPEN
    entry_price: Decimal = Field(ge=0)
    current_price: Optional[Decimal] = None
    liquidation_price: Optional[Decimal] = None
    quantity: Decimal = Field(ge=0)
    quote_quantity: Decimal = Field(ge=0)
    leverage: int = 1
    margin_type: str = "isolated"
    unrealized_pnl: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trailing_stop_pct: Optional[float] = None
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    signal_id: Optional[str] = None
    trades: List[Trade] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def pnl_pct(self) -> Optional[float]:
        if self.current_price is None or self.entry_price == 0:
            return None
        raw = (float(self.current_price) - float(self.entry_price)) / float(self.entry_price)
        if self.side == OrderSide.SELL:
            raw = -raw
        return raw * 100 * self.leverage

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.OPEN


# ---------------------------------------------------------------------------
# Wallet / On-Chain Signal
# ---------------------------------------------------------------------------


class WalletSignal(BaseModel):
    """An on-chain or exchange-flow signal from the wallet scanner."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    signal_type: str  # e.g. "whale_move", "exchange_inflow", "funding_spike"
    symbol: str
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    value: float  # numeric magnitude (e.g. USD amount, rate)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = ""  # e.g. "etherscan", "binance_flow", "coinglass"
    details: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy Version
# ---------------------------------------------------------------------------


class StrategyVersion(BaseModel):
    """Version metadata tracking a strategy iteration for winrate-based
    rollback decisions."""

    version: str  # semver, e.g. "1.2.3"
    strategy_name: str
    git_commit_hash: str
    git_tag: str
    author: str = ""
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = None
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    winrate: Optional[float] = None
    total_pnl: Optional[Decimal] = None
    active: bool = True
    parent_version: Optional[str] = None  # version this was forked from
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def winrate_pct(self) -> Optional[float]:
        if self.total_trades == 0:
            return None
        return (self.wins / self.total_trades) * 100.0

    def record_trade_result(self, won: bool, pnl: Decimal) -> None:
        """Record a trade result in this version's stats."""
        self.total_trades += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        if self.total_trades > 0:
            self.winrate = self.wins / self.total_trades
        if self.total_pnl is None:
            self.total_pnl = pnl
        else:
            self.total_pnl += pnl

    def should_rollback(self, min_winrate: float = 0.5, min_trades: int = 20) -> bool:
        """Return True if this version's winrate has dropped below the
        acceptable threshold over enough trades."""
        if self.total_trades < min_trades:
            return False
        current_wr = self.winrate if self.winrate is not None else 0.0
        return current_wr < min_winrate
