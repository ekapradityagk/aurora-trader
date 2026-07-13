"""
Aurora Trader — Shared Constants.

Timeframes, indicator defaults, exchange connection profiles, wallet scanner
endpoints, risk limits, and strategy default parameters used across the
entire project.
"""

from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Timeframes (seconds & human-readable keys)
# ---------------------------------------------------------------------------

TIMEFRAMES: Dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}

# Triple-timeframe strategy uses these by default
STRATEGY_TIMEFRAMES: Dict[str, str] = {
    "fast": "5m",
    "medium": "1h",
    "slow": "4h",
}

# Binance Kline interval strings (mapped to Binance API)
TIMEFRAME_BINANCE: Dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
}

# ---------------------------------------------------------------------------
# Indicator Defaults
# ---------------------------------------------------------------------------

INDICATOR_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "ema": {
        "fast_period": 9,
        "medium_period": 21,
        "slow_period": 50,
    },
    "rsi": {
        "period": 14,
        "oversold": 30,
        "overbought": 70,
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    },
    "bbands": {
        "period": 20,
        "std_dev": 2.0,
    },
    "stoch_rsi": {
        "period": 14,
        "k_period": 3,
        "d_period": 3,
    },
    "atr": {
        "period": 14,
        "multiplier": 1.5,
    },
    "volume_sma": {
        "period": 20,
    },
    "vwap": {
        "anchored": False,
    },
}

# ---------------------------------------------------------------------------
# Exchange Configs
# ---------------------------------------------------------------------------

EXCHANGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "binance": {
        "name": "binance",
        "rest_base_url": "https://api.binance.com",
        "wss_base_url": "wss://stream.binance.com:9443/ws",
        "wss_combined_url": "wss://stream.binance.com:9443/stream",
        "futures_rest_url": "https://fapi.binance.com",
        "futures_wss_url": "wss://fstream.binance.com/ws",
        "futures_wss_combined_url": "wss://fstream.binance.com/stream",
        "rate_limit_requests": 1200,
        "rate_limit_window_seconds": 60,
        "max_ws_subscriptions": 200,
        "default_leverage": 1,
        "margin_types": ["isolated", "cross"],
        "default_margin_type": "isolated",
    },
    "binance_testnet": {
        "name": "binance_testnet",
        "rest_base_url": "https://testnet.binance.vision",
        "wss_base_url": "wss://testnet.binance.vision/ws",
        "wss_combined_url": "wss://testnet.binance.vision/stream",
        "futures_rest_url": "https://testnet.binancefuture.com",
        "futures_wss_url": "wss://testnet.binancefuture.com/ws",
        "futures_wss_combined_url": "wss://testnet.binancefuture.com/stream",
        "rate_limit_requests": 1200,
        "rate_limit_window_seconds": 60,
        "max_ws_subscriptions": 200,
        "default_leverage": 1,
        "margin_types": ["isolated", "cross"],
        "default_margin_type": "isolated",
    },
}

# ---------------------------------------------------------------------------
# Wallet Scanner API Endpoints
# ---------------------------------------------------------------------------

WALLET_SCANNER_ENDPOINTS: Dict[str, str] = {
    "whale_tracker": "/api/v1/whale/transactions",
    "exchange_flow": "/api/v1/exchange/flows",
    "funding_rate": "/api/v1/funding/rates",
    "top_holders": "/api/v1/whale/holders",
    "smart_money": "/api/v1/whale/smart-money",
    "exchange_reserves": "/api/v1/exchange/reserves",
    "large_trades": "/api/v1/whale/large-trades",
    "stablecoin_flow": "/api/v1/exchange/stablecoin-flow",
}

# ---------------------------------------------------------------------------
# Risk Limits
# ---------------------------------------------------------------------------

RISK_LIMITS: Dict[str, Any] = {
    "max_position_size_usd": 10000.0,
    "max_leverage": 5,
    "min_leverage": 1,
    "max_daily_trades": 50,
    "max_open_positions": 5,
    "min_winrate_threshold": 0.50,  # auto-rollback if WR < 50%
    "winrate_evaluation_window": 20,  # trades to evaluate
    "max_drawdown_pct": 15.0,
    "daily_loss_limit_pct": 5.0,
    "max_slippage_pct": 0.1,
    "min_volume_usdt": 100000.0,
}

# ---------------------------------------------------------------------------
# Strategy Defaults
# ---------------------------------------------------------------------------

STRATEGY_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "cooldown_candles": 5,
    "max_signals_per_day": 10,
    "take_profit_pct": 1.5,
    "stop_loss_pct": 1.0,
    "trailing_stop_pct": 0.5,
    "min_confidence": 0.65,
    "timeframe_alignment": {
        "fast": "5m",
        "medium": "1h",
        "slow": "4h",
    },
}

# ---------------------------------------------------------------------------
# Server Ports
# ---------------------------------------------------------------------------

SERVER_PORTS: Dict[str, int] = {
    "trading": 8900,
    "learning": 8901,
    "wallet_scanner": 8902,
    "integration": 8903,
}
