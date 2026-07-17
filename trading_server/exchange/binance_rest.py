"""Aurora Trader — Binance REST Client.

Fetches kline history, places market/limit orders, retrieves account
info and balances. Handles rate limits with automatic retry logic
and exponential backoff.

Uses the ``python-binance`` async client under the hood.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from shared.config import load_config
from shared.constants import TIMEFRAME_BINANCE
from shared.logger import get_logger

logger = get_logger("trading_server.exchange.rest")

# ---------------------------------------------------------------------------
# Rate-limit-aware wrapper
# ---------------------------------------------------------------------------


class BinanceRestClient:
    """Async REST client for Binance exchange operations.

    Provides kline history fetching, order placement, and account queries
    with automatic rate-limit handling and retry logic.

    Usage::

        client = BinanceRestClient()
        klines = await client.get_klines("BTCUSDT", "1h", limit=100)
        order = await client.place_market_order("BTCUSDT", "BUY", 0.01)
        balance = await client.get_balance("USDT")
        await client.close()
    """

    def __init__(self) -> None:
        self._client: Any = None  # AsyncClient instance
        self._log = get_logger("trading_server.exchange.rest")
        self._lock = asyncio.Lock()
        self._rate_limit_remaining: int = 1200
        self._rate_limit_reset: float = 0.0

        cfg = load_config()
        self._api_key = cfg.exchange_api_key
        self._api_secret = cfg.exchange_api_secret
        self._testnet = cfg.exchange_testnet
        self._use_futures = cfg.exchange_use_futures
        self._default_leverage = cfg.exchange_default_leverage
        self._margin_type = cfg.exchange_margin_type
        self._max_retries = 3
        self._base_delay = 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _method_name(self, spot_name: str) -> str:
        """Map a generic method name to spot or futures variant."""
        if not self._use_futures:
            return spot_name
        # Map common methods to their futures counterparts
        mapping = {
            "get_account": "futures_account",
            "create_order": "futures_create_order",
            "cancel_order": "futures_cancel_order",
            "get_order": "futures_get_order",
            "get_open_orders": "futures_get_open_orders",
            "get_position_information": "futures_position_information",
            "get_exchange_info": "futures_exchange_info",
            "get_symbol_ticker": "futures_symbol_ticker",
        }
        return mapping.get(spot_name, spot_name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> Any:
        """Lazily initialise the Binance async client."""
        if self._client is not None:
            return self._client

        from binance import AsyncClient

        api_key = self._api_key or ""
        api_secret = self._api_secret or ""

        if self._testnet:
            self._client = await AsyncClient.create(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True,
            )
        else:
            self._client = await AsyncClient.create(
                api_key=api_key,
                api_secret=api_secret,
            )

        # Futures-specific setup
        if self._use_futures:
            try:
                # Set leverage
                await self._client.futures_change_leverage(
                    symbol="BTCUSDT",
                    leverage=self._default_leverage,
                )
                # Set margin type (isolated / cross)
                await self._client.futures_change_margin_type(
                    symbol="BTCUSDT",
                    marginType=self._margin_type.upper(),
                )
                self._log.info(
                    f"Futures configured: leverage={self._default_leverage}x, "
                    f"margin={self._margin_type}"
                )
            except Exception as exc:
                self._log.warning(f"Futures setup warning (may already be set): {exc}")

        self._log.info(
            f"Binance REST client initialised "
            f"{'testnet' if self._testnet else 'mainnet'}"
            f"{' futures' if self._use_futures else ' spot'}"
        )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client:
            try:
                await self._client.close_connection()
            except Exception as exc:
                self._log.debug(f"Error closing client: {exc}")
            self._client = None
            self._log.info("Binance REST client closed")

    # ------------------------------------------------------------------
    # Rate-limit handling
    # ------------------------------------------------------------------

    async def _throttle(self) -> None:
        """Apply rate-limit backpressure if we're close to the limit."""
        now = time.time()
        if (
            self._rate_limit_remaining < 10
            and self._rate_limit_reset > now
        ):
            sleep_for = self._rate_limit_reset - now + 0.1
            self._log.warning(
                f"Rate limit low ({self._rate_limit_remaining}), "
                f"pausing {sleep_for:.1f}s"
            )
            await asyncio.sleep(sleep_for)

    def _update_rate_limit(self, headers: Any) -> None:
        """Parse rate-limit headers from the response."""
        try:
            remaining = headers.get("X-MBX-Used-Weight", "")
            if remaining:
                self._rate_limit_remaining = max(
                    0, 1200 - int(remaining)
                )
            reset = headers.get("X-MBX-Order-Count-1s", None)
            if reset is not None:
                self._rate_limit_reset = time.time() + 1
        except (ValueError, AttributeError):
            pass

    async def _request_with_retry(
        self, method: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute an API request with retry logic."""
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                await self._throttle()
                client = await self._ensure_client()

                # Map our generic method names to the AsyncClient methods
                resolved = self._method_name(method)
                client_method = getattr(client, resolved, None)
                if client_method is None:
                    raise ValueError(f"Unknown method: {method} -> {resolved}")

                result = await client_method(*args, **kwargs)
                return result

            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()

                # Rate-limit specific retry
                if "too many requests" in err_str or "429" in err_str:
                    delay = self._base_delay * (2 ** attempt) + 1.0
                    self._log.warning(
                        f"Rate limited (attempt {attempt + 1}), "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue

                # Server errors
                if "5" in err_str[:10] and attempt < self._max_retries:
                    delay = self._base_delay * (2 ** attempt)
                    self._log.warning(
                        f"Server error (attempt {attempt + 1}): {exc}, "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue

                if attempt >= self._max_retries:
                    self._log.error(
                        f"Request failed after {self._max_retries} retries: {exc}"
                    )
                    raise

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Kline / Historical Data
    # ------------------------------------------------------------------

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_str: Optional[str] = None,
        end_str: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch historical kline (candlestick) data.

        Args:
            symbol: Trading pair, e.g. ``"BTCUSDT"``.
            interval: Binance interval string, e.g. ``"1h"``, ``"4h"``, ``"1d"``.
            limit: Number of candles (max 1000).
            start_str: Optional start time (ISO string or ms epoch).
            end_str: Optional end time (ISO string or ms epoch).

        Returns:
            List of kline dicts with keys: open_time, open, high, low, close,
            volume, close_time, quote_volume, count, taker_buy_volume,
            taker_buy_quote_volume, ignore.
        """
        binance_interval = TIMEFRAME_BINANCE.get(interval, interval)
        kwargs: Dict[str, Any] = {
            "symbol": symbol,
            "interval": binance_interval,
            "limit": min(limit, 1000),
        }
        if start_str:
            kwargs["start_str"] = start_str
        if end_str:
            kwargs["end_str"] = end_str

        raw = await self._request_with_retry(
            "get_klines",
            **kwargs,
        )

        # Transform raw list to dicts for readability
        keys = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        result: List[Dict[str, Any]] = []
        for row in raw:
            entry = dict(zip(keys, row))
            # Convert numeric fields from string to Decimal
            for num_key in (
                "open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_volume", "taker_buy_quote_volume",
            ):
                entry[num_key] = Decimal(entry[num_key])
            result.append(entry)

        self._log.debug(
            f"Fetched {len(result)} klines for {symbol} {interval}"
        )
        return result

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        quote_order_qty: Optional[Decimal] = None,
        new_client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Place a market order.

        Args:
            symbol: Trading pair.
            side: ``"BUY"`` or ``"SELL"``.
            quantity: Base asset quantity.
            quote_order_qty: Quote asset quantity (alternative to quantity).
            new_client_order_id: Optional client order ID.

        Returns:
            Order execution result dict.
        """
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
        }
        if quote_order_qty is not None:
            params["quoteOrderQty"] = str(quote_order_qty)
        else:
            params["quantity"] = str(quantity)
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id

        result = await self._request_with_retry(
            "create_order",
            **params,
        )
        self._log.trade(
            "MARKET_ORDER",
            symbol=symbol,
            side=side,
            quantity=str(quantity),
            price=result.get("price", "N/A"),
            status=result.get("status", "N/A"),
        )
        return result

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        time_in_force: str = "GTC",
        new_client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Place a limit order.

        Args:
            symbol: Trading pair.
            side: ``"BUY"`` or ``"SELL"``.
            quantity: Base asset quantity.
            price: Limit price.
            time_in_force: ``"GTC"``, ``"IOC"``, ``"FOK"``.
            new_client_order_id: Optional client order ID.

        Returns:
            Order execution result dict.
        """
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": time_in_force.upper(),
            "quantity": str(quantity),
            "price": str(price),
        }
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id

        result = await self._request_with_retry(
            "create_order",
            **params,
        )
        self._log.trade(
            "LIMIT_ORDER",
            symbol=symbol,
            side=side,
            quantity=str(quantity),
            price=str(price),
            status=result.get("status", "N/A"),
        )
        return result

    async def cancel_order(
        self, symbol: str, order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel an open order."""
        params: Dict[str, Any] = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id

        return await self._request_with_retry(
            "cancel_order",
            **params,
        )

    async def get_order_status(
        self, symbol: str, order_id: int
    ) -> Dict[str, Any]:
        """Check the status of an order."""
        return await self._request_with_retry(
            "get_order",
            symbol=symbol,
            orderId=order_id,
        )

    async def get_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all open orders, optionally filtered by symbol."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request_with_retry(
            "get_open_orders",
            **params,
        )

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_account_info(self) -> Dict[str, Any]:
        """Get full account information including all balances."""
        if self._use_futures:
            return await self._request_with_retry("get_account")
        return await self._request_with_retry("get_account")

    async def get_balance(self, asset: str = "USDT") -> Decimal:
        """Get the free balance for a specific asset.

        For futures, returns the wallet balance (available + used margin).
        For spot, returns the free balance.

        Returns:
            Decimal amount of balance.
        """
        info = await self.get_account_info()
        if self._use_futures:
            # Futures format: {"assets": [{"asset":"USDT","walletBalance":"43.98",...}]}
            for bal in info.get("assets", []):
                if bal.get("asset", "").upper() == asset.upper():
                    return Decimal(bal.get("walletBalance", "0"))
            return Decimal("0")
        else:
            # Spot format: {"balances": [{"asset":"USDT","free":"5.08","locked":"0.00"}]}
            for bal in info.get("balances", []):
                if bal["asset"] == asset.upper():
                    return Decimal(bal.get("free", "0"))
            return Decimal("0")

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open futures positions from Binance."""
        if not self._use_futures:
            return []
        try:
            positions = await self._request_with_retry("get_position_information")
            # Filter to positions with actual size (positionAmt != 0)
            return [p for p in positions if Decimal(p.get("positionAmt", "0")) != 0]
        except Exception as exc:
            self._log.warning(f"Failed to fetch open positions: {exc}")
            return []

    async def get_all_balances(self) -> Dict[str, Dict[str, Decimal]]:
        """Get all non-zero balances as a dict keyed by asset.

        Returns:
            ``{"BTC": {"free": Decimal, "locked": Decimal}, ...}``
        """
        info = await self.get_account_info()
        result: Dict[str, Dict[str, Decimal]] = {}
        if self._use_futures:
            for bal in info.get("assets", []):
                total = Decimal(bal.get("walletBalance", "0"))
                upnl = Decimal(bal.get("unrealizedProfit", "0"))
                if total > 0:
                    result[bal["asset"]] = {
                        "free": total + upnl,  # available = wallet + unrealized
                        "locked": Decimal("0"),
                        "wallet_balance": total,
                        "unrealized_pnl": upnl,
                    }
        else:
            for bal in info.get("balances", []):
                free = Decimal(bal.get("free", "0"))
                locked = Decimal(bal.get("locked", "0"))
                if free > 0 or locked > 0:
                    result[bal["asset"]] = {"free": free, "locked": locked}
        return result

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Get the latest price for a symbol."""
        ticker = await self._request_with_retry(
            "get_symbol_ticker",
            symbol=symbol,
        )
        return Decimal(ticker.get("price", "0"))

    async def get_exchange_info(
        self, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get exchange trading rules and symbol info."""
        if symbol:
            return await self._request_with_retry(
                "get_exchange_info",
                symbol=symbol,
            )
        return await self._request_with_retry("get_exchange_info")
