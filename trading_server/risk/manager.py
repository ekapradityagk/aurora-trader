"""Aurora Trader — Risk Manager.

Central risk management for position sizing, stop losses, and position
lifecycle (break-even, trailing stop).

Key features:
- Kelly-based position sizing (quarter Kelly)
- 1% risk per trade, 3% daily loss limit
- ATR-based stop loss calculation
- Break-even after +1R profit
- Trailing stop after +2R profit
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from shared.config import load_config
from shared.constants import RISK_LIMITS
from shared.logger import get_logger
from shared.models import Position, PositionStatus, OrderSide

logger = get_logger("trading_server.risk.manager")


class RiskManager:
    """Manages trade-level risk: position sizing, stop loss placement,
    break-even activation, and trailing stop updates.

    Usage::

        risk_mgr = RiskManager()
        size = await risk_mgr.calculate_position_size(
            account_balance=10000, entry_price=45000,
            stop_loss=44000, atr=800
        )
        new_sl = risk_mgr.update_stop_loss(position, current_price)
    """

    def __init__(self) -> None:
        cfg = load_config()
        risk_global = cfg.risk_global

        self._max_position_size = Decimal(
            str(risk_global.get("max_position_size_usd", 10000))
        )
        self._max_leverage = risk_global.get("max_leverage", 5)
        self._max_open_positions = risk_global.get("max_open_positions", 5)
        self._max_daily_trades = risk_global.get("max_daily_trades", 50)
        self._daily_loss_limit_pct = risk_global.get(
            "daily_loss_limit_pct", 5.0
        )

        # Risk per trade (% of account) — from config
        self._risk_per_trade = float(risk_global.get("risk_per_trade_pct", 2.0)) / 100.0

        # Kelly fraction (quarter Kelly)
        self._kelly_fraction = 0.25

        # Winrate estimates for Kelly (used until we have real stats)
        self._default_winrate = 0.50
        self._default_rr = 2.0

        self._log = logger
        self._log.info(
            f"RiskManager initialised: max_pos={self._max_position_size}, "
            f"max_leverage={self._max_leverage}, "
            f"risk_per_trade={self._risk_per_trade:.1%}"
        )

    # ------------------------------------------------------------------
    # Position Sizing (Quarter Kelly)
    # ------------------------------------------------------------------

    def calculate_kelly_fraction(
        self,
        winrate: float = 0.50,
        avg_win: float = 2.0,
        avg_loss: float = 1.0,
    ) -> float:
        """Calculate the Kelly fraction.

        Formula: K = W - (1 - W) / R
        where W = winrate, R = win/loss ratio.
        """
        if avg_loss == 0:
            return 0.0
        r = avg_win / avg_loss
        kelly = winrate - (1 - winrate) / r
        return max(0.0, min(kelly, 1.0))  # Clamp to [0, 1]

    async def calculate_position_size(
        self,
        account_balance: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        atr: Optional[Decimal] = None,
        winrate: Optional[float] = None,
        rr_ratio: Optional[float] = None,
    ) -> Decimal:
        """Calculate position size using quarter-Kelly sizing.

        Args:
            account_balance: Total account equity in quote currency.
            entry_price: Planned entry price.
            stop_loss: Planned stop-loss price.
            atr: ATR value (optional, used as fallback).
            winrate: Historical winrate (used for Kelly).
            rr_ratio: Win/loss ratio (used for Kelly).

        Returns:
            Position size in base currency (quantity).
        """
        # 1. Calculate the risk amount (1% of account)
        risk_amount = account_balance * Decimal(str(self._risk_per_trade))

        # 2. Calculate the per-unit risk in price terms
        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            self._log.warning("Entry and stop are equal — using ATR-based risk fallback")
            if atr and atr > 0:
                price_risk = atr * Decimal("2")
            else:
                return Decimal("0")

        # 3. Base position size from fixed-fractional risk
        base_size = risk_amount / price_risk

        # 4. Apply quarter-Kelly adjustment
        wr = winrate if winrate is not None else self._default_winrate
        rr = rr_ratio if rr_ratio is not None else self._default_rr
        kelly = self.calculate_kelly_fraction(wr, rr)
        # Use max(kelly * fraction, 0.5) so we never go below half the base
        kelly_factor = max(self._kelly_fraction * kelly, 0.5) if kelly > 0 else 0.5
        size = base_size * Decimal(str(kelly_factor))

        # 5. Cap by max position size
        max_by_balance = account_balance / entry_price * Decimal("0.5")  # max 50% in one position
        size = min(size, max_by_balance)

        max_by_config = self._max_position_size / entry_price
        size = min(size, max_by_config)

        # 6. Apply leverage
        size = size * Decimal(str(self._max_leverage))

        # Round to reasonable precision
        size = self._round_size(size, entry_price)

        self._log.trade(
            "POSITION_SIZE",
            entry_price=str(entry_price),
            stop_loss=str(stop_loss),
            risk_amount=str(risk_amount),
            kelly=round(kelly, 4),
            base_size=str(base_size),
            final_size=str(size),
        )
        return size

    # ------------------------------------------------------------------
    # Stop Loss Management
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_atr_stop(
        entry_price: Decimal,
        atr: Decimal,
        side: OrderSide,
        multiplier: Decimal = Decimal("2"),
    ) -> Decimal:
        """Calculate an ATR-based initial stop loss."""
        if side == OrderSide.BUY:
            return entry_price - (atr * multiplier)
        else:
            return entry_price + (atr * multiplier)

    def should_activate_break_even(
        self,
        position: Position,
        current_price: Decimal,
    ) -> bool:
        """Check if the position has reached +1R and should move to BE."""
        if not position.is_open or position.entry_price == 0:
            return False

        entry = position.entry_price
        risk_per_unit = abs(entry - (position.stop_loss or Decimal("0")))

        if risk_per_unit == 0:
            return False

        if position.side == OrderSide.BUY:
            profit = current_price - entry
        else:
            profit = entry - current_price

        return profit >= risk_per_unit

    def should_activate_trailing(
        self,
        position: Position,
        current_price: Decimal,
    ) -> bool:
        """Check if the position has reached +2R and should trail."""
        if not position.is_open or position.entry_price == 0:
            return False

        entry = position.entry_price
        risk_per_unit = abs(entry - (position.stop_loss or Decimal("0")))

        if risk_per_unit == 0:
            return False

        if position.side == OrderSide.BUY:
            profit = current_price - entry
        else:
            profit = entry - current_price

        return profit >= risk_per_unit * 2

    def update_trailing_stop(
        self,
        current_stop: Decimal,
        current_price: Decimal,
        side: OrderSide,
        trail_distance: Decimal,
    ) -> Decimal:
        """Update a trailing stop loss.

        For long positions, the stop only moves upward.
        For short positions, the stop only moves downward.
        """
        if side == OrderSide.BUY:
            new_stop = current_price - trail_distance
            return max(new_stop, current_stop)
        else:
            new_stop = current_price + trail_distance
            return min(new_stop, current_stop)

    def update_stop_loss(
        self,
        position: Position,
        current_price: Decimal,
    ) -> Tuple[Optional[Decimal], str]:
        """Evaluate and update the stop loss for a position.

        Returns:
            Tuple of (new_stop_loss, action_taken).
            ``action_taken`` is one of: ``"none"``, ``"break_even"``,
            ``"trailing"``, or ``"stopped_out"``.
        """
        if not position.is_open:
            return position.stop_loss, "none"

        atr = Decimal(str(position.metadata.get("atr", "0"))) if position.metadata else Decimal("0")
        if atr == 0:
            return position.stop_loss, "none"

        entry = position.entry_price
        current_sl = position.stop_loss or self.calculate_atr_stop(
            entry, atr, position.side
        )

        # Check break-even activation (+1R)
        if self.should_activate_break_even(position, current_price):
            if position.side == OrderSide.BUY:
                be_stop = entry
            else:
                be_stop = entry

            # Only move if it improves the stop
            if (position.side == OrderSide.BUY and be_stop > current_sl) or \
               (position.side == OrderSide.SELL and be_stop < current_sl):
                self._log.trade(
                    "BREAK_EVEN",
                    symbol=position.symbol,
                    entry=str(entry),
                    new_stop=str(be_stop),
                )
                # Check trailing activation (+2R)
                if self.should_activate_trailing(position, current_price):
                    trail_distance = atr * Decimal("2")
                    trail_stop = self.update_trailing_stop(
                        be_stop, current_price, position.side, trail_distance
                    )
                    self._log.trade(
                        "TRAILING_STOP",
                        symbol=position.symbol,
                        entry=str(entry),
                        price=str(current_price),
                        new_stop=str(trail_stop),
                    )
                    return trail_stop, "trailing"
                return be_stop, "break_even"

        # If trailing already active, continue updating
        if position.metadata and position.metadata.get("trailing_active"):
            trail_distance = atr * Decimal("2")
            trail_stop = self.update_trailing_stop(
                current_sl, current_price, position.side, trail_distance
            )
            return trail_stop, "trailing"

        return current_sl, "none"

    # ------------------------------------------------------------------
    # Risk Checks
    # ------------------------------------------------------------------

    def check_daily_loss_limit(
        self,
        daily_pnl: Decimal,
        starting_balance: Decimal,
    ) -> bool:
        """Return True if the daily loss limit has been breached."""
        if starting_balance == 0:
            return False
        loss_pct = abs(float(daily_pnl / starting_balance)) * 100
        return loss_pct >= self._daily_loss_limit_pct

    def can_open_new_position(
        self,
        open_positions: List[Position],
        daily_trade_count: int,
    ) -> Tuple[bool, str]:
        """Check if we are allowed to open a new position."""
        if len(open_positions) >= self._max_open_positions:
            return False, f"Max open positions ({self._max_open_positions}) reached"
        if daily_trade_count >= self._max_daily_trades:
            return False, f"Max daily trades ({self._max_daily_trades}) reached"
        return True, "ok"

    @staticmethod
    def _round_size(size: Decimal, price: Decimal) -> Decimal:
        """Round position size to a reasonable precision."""
        if size <= 0:
            return Decimal("0")

        # Determine precision based on price
        if price >= 1000:
            return size.quantize(Decimal("0.001"))
        elif price >= 100:
            return size.quantize(Decimal("0.0001"))
        elif price >= 1:
            return size.quantize(Decimal("0.00001"))
        else:
            return size.quantize(Decimal("0.000001"))
