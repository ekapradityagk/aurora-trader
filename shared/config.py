"""
Aurora Trader — Configuration Loader.

Loads project configuration from a YAML file with environment variable
overrides (especially for sensitive fields such as exchange API keys).

Design:
  - A single ``Config`` dataclass holds all settings.
  - ``load_config(path)`` reads a YAML file, applies defaults, then
    overlays environment variables.
  - Environment variables are mapped via a ``_ENV_OVERRIDES`` dict so that
    e.g. ``AURORA_BINANCE_API_KEY`` sets ``exchange.binance.api_key``.
  - Works standalone — no dependency on the other shared modules.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Default configuration (embedded so the project works even without a
# config.yaml present — though one should always be provided).
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "aurora-trader",
        "environment": "development",  # development | staging | production
        "log_level": "INFO",
        "log_dir": "logs",
        "json_logging": False,
    },
    "exchange": {
        "name": "binance",
        "testnet": True,
        "api_key": "",  # override via AURORA_BINANCE_API_KEY
        "api_secret": "",  # override via AURORA_BINANCE_API_SECRET
        "use_futures": False,
        "default_leverage": 1,
        "margin_type": "isolated",
        "rate_limit": {
            "requests_per_minute": 1200,
        },
    },
    "strategies": {
        "ema_crossover": {
            "enabled": True,
            "timeframes": {
                "fast": "5m",
                "medium": "1h",
                "slow": "4h",
            },
            "parameters": {
                "ema_fast": 9,
                "ema_medium": 21,
                "ema_slow": 50,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "min_volume_usdt": 100000.0,
            },
            "risk": {
                "max_position_size_usd": 1000.0,
                "stop_loss_pct": 1.0,
                "take_profit_pct": 1.5,
                "trailing_stop_pct": 0.5,
                "cooldown_candles": 5,
                "max_signals_per_day": 10,
                "min_confidence": 0.65,
            },
        },
        "rsi_divergence": {
            "enabled": False,
            "timeframes": {
                "fast": "5m",
                "medium": "1h",
                "slow": "4h",
            },
            "parameters": {
                "rsi_period": 14,
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "divergence_lookback": 30,
                "min_volume_usdt": 50000.0,
            },
            "risk": {
                "max_position_size_usd": 800.0,
                "stop_loss_pct": 1.2,
                "take_profit_pct": 2.0,
                "trailing_stop_pct": 0.6,
                "cooldown_candles": 8,
                "max_signals_per_day": 8,
                "min_confidence": 0.70,
            },
        },
        "macd_strategy": {
            "enabled": False,
            "timeframes": {
                "fast": "15m",
                "medium": "1h",
                "slow": "4h",
            },
            "parameters": {
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "histogram_threshold": 0.0,
                "min_volume_usdt": 100000.0,
            },
            "risk": {
                "max_position_size_usd": 900.0,
                "stop_loss_pct": 1.1,
                "take_profit_pct": 1.8,
                "trailing_stop_pct": 0.5,
                "cooldown_candles": 6,
                "max_signals_per_day": 8,
                "min_confidence": 0.60,
            },
        },
        "bbands_reversal": {
            "enabled": False,
            "timeframes": {
                "fast": "5m",
                "medium": "1h",
            },
            "parameters": {
                "bb_period": 20,
                "bb_std_dev": 2.0,
                "rsi_period": 14,
                "min_volume_usdt": 50000.0,
            },
            "risk": {
                "max_position_size_usd": 700.0,
                "stop_loss_pct": 0.8,
                "take_profit_pct": 1.2,
                "trailing_stop_pct": 0.4,
                "cooldown_candles": 5,
                "max_signals_per_day": 12,
                "min_confidence": 0.60,
            },
        },
        "vwap_trend": {
            "enabled": False,
            "timeframes": {
                "fast": "5m",
                "medium": "15m",
                "slow": "1h",
            },
            "parameters": {
                "vwap_period": 20,
                "deviation_multiplier": 1.5,
                "min_volume_usdt": 200000.0,
            },
            "risk": {
                "max_position_size_usd": 600.0,
                "stop_loss_pct": 0.9,
                "take_profit_pct": 1.4,
                "trailing_stop_pct": 0.5,
                "cooldown_candles": 4,
                "max_signals_per_day": 15,
                "min_confidence": 0.55,
            },
        },
    },
    "risk_management": {
        "global": {
            "max_position_size_usd": 10000.0,
            "max_leverage": 20,
            "max_daily_trades": 50,
            "max_open_positions": 6,
            "risk_per_trade_pct": 5.0,
            "max_drawdown_pct": 15.0,
            "daily_loss_limit_pct": 5.0,
            "max_slippage_pct": 0.1,
            "min_volume_usdt": 100000.0,
        },
        "auto_rollback": {
            "enabled": True,
            "min_winrate": 0.50,
            "evaluation_window": 20,
            "max_versions_to_keep": 10,
        },
    },
    "wallet_scanner": {
        "enabled": False,
        "api_base_url": "http://127.0.0.1:8902",
        "poll_interval_seconds": 60,
        "endpoints": {
            "whale_tracker": "/api/v1/whale/transactions",
            "exchange_flow": "/api/v1/exchange/flows",
            "funding_rate": "/api/v1/funding/rates",
            "top_holders": "/api/v1/whale/holders",
            "smart_money": "/api/v1/whale/smart-money",
            "exchange_reserves": "/api/v1/exchange/reserves",
            "large_trades": "/api/v1/whale/large-trades",
            "stablecoin_flow": "/api/v1/exchange/stablecoin-flow",
        },
        "api_key": "",  # optional; override via AURORA_WALLET_API_KEY
    },
    "learning_server": {
        "enabled": False,
        "api_base_url": "http://127.0.0.1:8901",
        "optuna": {
            "n_trials": 100,
            "timeout_seconds": 3600,
            "storage": "sqlite:///data/optuna_study.db",
            "direction": "maximize",
        },
        "regime_detection": {
            "enabled": True,
            "lookback_periods": 100,
            "min_regime_confidence": 0.6,
        },
    },
    "trading_server": {
        "host": "127.0.0.1",
        "port": 8900,
        "database": {
            "path": "data/trading.db",
        },
        "websocket": {
            "max_retries": 5,
            "retry_delay_seconds": 2.0,
            "ping_interval_seconds": 30,
        },
    },
    "integration": {
        "host": "127.0.0.1",
        "port": 8903,
        "git": {
            "auto_commit": True,
            "auto_tag": True,
            "tag_prefix": "v",
        },
        "database": {
            "path": "data/integration.db",
        },
    },
}

# ---------------------------------------------------------------------------
# Environment variable overrides
# Maps env var name -> dot-separated config key
# ---------------------------------------------------------------------------

_ENV_OVERRIDES: Dict[str, str] = {
    "AURORA_BINANCE_API_KEY": "exchange.api_key",
    "AURORA_BINANCE_API_SECRET": "exchange.api_secret",
    "AURORA_WALLET_API_KEY": "wallet_scanner.api_key",
    "AURORA_ENVIRONMENT": "project.environment",
    "AURORA_LOG_LEVEL": "project.log_level",
    "AURORA_TESTNET": "exchange.testnet",
    "AURORA_MAX_LEVERAGE": "risk_management.global.max_leverage",
    "AURORA_MAX_POSITION_SIZE": "risk_management.global.max_position_size_usd",
    "AURORA_MAX_DAILY_TRADES": "risk_management.global.max_daily_trades",
    "AURORA_MAX_OPEN_POSITIONS": "risk_management.global.max_open_positions",
    "AURORA_MAX_DRAWDOWN_PCT": "risk_management.global.max_drawdown_pct",
    "AURORA_DAILY_LOSS_LIMIT_PCT": "risk_management.global.daily_loss_limit_pct",
}

# ---------------------------------------------------------------------------
# Helper: dict dot-access
# ---------------------------------------------------------------------------


def _deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a value from a nested dict via dot-separated *path*."""
    keys = path.split(".")
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
            if d is None:
                return default
        else:
            return default
    return d


def _deep_set(d: Dict[str, Any], path: str, value: Any) -> None:
    """Set a value in a nested dict via dot-separated *path*."""
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class StrategyConfig:
    """Holds configuration for a single strategy."""

    name: str
    enabled: bool
    timeframes: Dict[str, str]
    parameters: Dict[str, Any]
    risk: Dict[str, Any]

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "StrategyConfig":
        return cls(
            name=name,
            enabled=data.get("enabled", True),
            timeframes=data.get("timeframes", {}),
            parameters=data.get("parameters", {}),
            risk=data.get("risk", {}),
        )


@dataclass
class Config:
    """Top-level configuration object for the entire Aurora Trader project.

    This is a flat-ish dataclass that exposes the most commonly accessed
    settings as properties, while also preserving the raw dictionary for
    any arbitrary lookups.
    """

    data: Dict[str, Any] = field(default_factory=lambda: _deep_copy(_DEFAULT_CONFIG))

    # Cached sub-configurations (populated on access)
    _strategies: Optional[Dict[str, StrategyConfig]] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    @property
    def project_name(self) -> str:
        return str(self.data.get("project", {}).get("name", "aurora-trader"))

    @property
    def environment(self) -> str:
        return str(self.data.get("project", {}).get("environment", "development"))

    @property
    def log_level(self) -> str:
        return str(self.data.get("project", {}).get("log_level", "INFO"))

    @property
    def log_dir(self) -> str:
        return str(self.data.get("project", {}).get("log_dir", "logs"))

    @property
    def json_logging(self) -> bool:
        return bool(self.data.get("project", {}).get("json_logging", False))

    # ------------------------------------------------------------------
    # Exchange
    # ------------------------------------------------------------------

    @property
    def exchange_name(self) -> str:
        return str(self.data.get("exchange", {}).get("name", "binance"))

    @property
    def exchange_testnet(self) -> bool:
        return bool(self.data.get("exchange", {}).get("testnet", True))

    @property
    def exchange_api_key(self) -> str:
        return str(self.data.get("exchange", {}).get("api_key", ""))

    @property
    def exchange_api_secret(self) -> str:
        return str(self.data.get("exchange", {}).get("api_secret", ""))

    @property
    def exchange_use_futures(self) -> bool:
        return bool(self.data.get("exchange", {}).get("use_futures", False))

    @property
    def exchange_default_leverage(self) -> int:
        return int(self.data.get("exchange", {}).get("default_leverage", 1))

    @property
    def exchange_margin_type(self) -> str:
        return str(self.data.get("exchange", {}).get("margin_type", "isolated"))

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    @property
    def strategies(self) -> Dict[str, StrategyConfig]:
        if self._strategies is None:
            self._strategies = {}
            raw = self.data.get("strategies", {})
            for name, cfg in raw.items():
                self._strategies[name] = StrategyConfig.from_dict(name, cfg)
        return self._strategies

    def is_strategy_enabled(self, name: str) -> bool:
        st = self.strategies.get(name)
        return st.enabled if st else False

    def strategy_params(self, name: str) -> Dict[str, Any]:
        st = self.strategies.get(name)
        return st.parameters if st else {}

    def strategy_risk(self, name: str) -> Dict[str, Any]:
        st = self.strategies.get(name)
        return st.risk if st else {}

    # ------------------------------------------------------------------
    # Risk management
    # ------------------------------------------------------------------

    @property
    def risk_global(self) -> Dict[str, Any]:
        return self.data.get("risk_management", {}).get("global", {})

    @property
    def auto_rollback(self) -> Dict[str, Any]:
        return self.data.get("risk_management", {}).get("auto_rollback", {})

    @property
    def max_position_size_usd(self) -> float:
        return float(self.risk_global.get("max_position_size_usd", 10000.0))

    @property
    def max_leverage(self) -> int:
        return int(self.risk_global.get("max_leverage", 5))

    @property
    def max_daily_trades(self) -> int:
        return int(self.risk_global.get("max_daily_trades", 50))

    @property
    def max_open_positions(self) -> int:
        return int(self.risk_global.get("max_open_positions", 5))

    @property
    def risk_per_trade_pct(self) -> float:
        return float(self.risk_global.get("risk_per_trade_pct", 2.0))

    @property
    def max_drawdown_pct(self) -> float:
        return float(self.risk_global.get("max_drawdown_pct", 15.0))

    @property
    def daily_loss_limit_pct(self) -> float:
        return float(self.risk_global.get("daily_loss_limit_pct", 5.0))

    @property
    def min_winrate(self) -> float:
        return float(self.auto_rollback.get("min_winrate", 0.5))

    @property
    def evaluation_window(self) -> int:
        return int(self.auto_rollback.get("evaluation_window", 20))

    # ------------------------------------------------------------------
    # Wallet scanner
    # ------------------------------------------------------------------

    @property
    def wallet_enabled(self) -> bool:
        return bool(self.data.get("wallet_scanner", {}).get("enabled", False))

    @property
    def wallet_base_url(self) -> str:
        return str(self.data.get("wallet_scanner", {}).get("api_base_url", "http://127.0.0.1:8902"))

    @property
    def wallet_poll_interval(self) -> int:
        return int(self.data.get("wallet_scanner", {}).get("poll_interval_seconds", 60))

    @property
    def wallet_endpoints(self) -> Dict[str, str]:
        return self.data.get("wallet_scanner", {}).get("endpoints", {})

    @property
    def wallet_api_key(self) -> str:
        return str(self.data.get("wallet_scanner", {}).get("api_key", ""))

    # ------------------------------------------------------------------
    # Learning server
    # ------------------------------------------------------------------

    @property
    def learning_enabled(self) -> bool:
        return bool(self.data.get("learning_server", {}).get("enabled", False))

    @property
    def learning_base_url(self) -> str:
        return str(self.data.get("learning_server", {}).get("api_base_url", "http://127.0.0.1:8901"))

    @property
    def optuna_config(self) -> Dict[str, Any]:
        return self.data.get("learning_server", {}).get("optuna", {})

    @property
    def regime_detection(self) -> Dict[str, Any]:
        return self.data.get("learning_server", {}).get("regime_detection", {})

    # ------------------------------------------------------------------
    # Trading server
    # ------------------------------------------------------------------

    @property
    def trading_host(self) -> str:
        return str(self.data.get("trading_server", {}).get("host", "127.0.0.1"))

    @property
    def trading_port(self) -> int:
        return int(self.data.get("trading_server", {}).get("port", 8900))

    @property
    def trading_db_path(self) -> str:
        return str(self.data.get("trading_server", {}).get("database", {}).get("path", "data/trading.db"))

    @property
    def ws_config(self) -> Dict[str, Any]:
        return self.data.get("trading_server", {}).get("websocket", {})

    # ------------------------------------------------------------------
    # Pair universe
    # ------------------------------------------------------------------

    @property
    def pair_universe(self) -> Dict[str, Any]:
        return self.data.get("pair_universe", {})

    @property
    def pair_universe_candidates(self) -> List[str]:
        return self.pair_universe.get("candidates", [])

    @property
    def pair_universe_active_count(self) -> int:
        return int(self.pair_universe.get("active_count", 6))

    @property
    def pair_universe_lookback_days(self) -> int:
        return int(self.pair_universe.get("lookback_days", 7))

    @property
    def pair_universe_min_volume(self) -> float:
        return float(self.pair_universe.get("min_volume_usdt", 50_000_000))

    # ------------------------------------------------------------------
    # Integration
    # Integration server
    # ------------------------------------------------------------------

    @property
    def integration_host(self) -> str:
        return str(self.data.get("integration", {}).get("host", "127.0.0.1"))

    @property
    def integration_port(self) -> int:
        return int(self.data.get("integration", {}).get("port", 8903))

    @property
    def git_auto_commit(self) -> bool:
        return bool(self.data.get("integration", {}).get("git", {}).get("auto_commit", True))

    @property
    def git_auto_tag(self) -> bool:
        return bool(self.data.get("integration", {}).get("git", {}).get("auto_tag", True))

    @property
    def git_tag_prefix(self) -> str:
        return str(self.data.get("integration", {}).get("git", {}).get("tag_prefix", "v"))

    @property
    def integration_db_path(self) -> str:
        return str(self.data.get("integration", {}).get("database", {}).get("path", "data/integration.db"))

    # ------------------------------------------------------------------
    # Dict access
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Access raw config by dot-separated key."""
        return _deep_get(self.data, key, default)

    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        return self.strategies.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return _deep_copy(self.data)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _deep_copy(d: Dict[str, Any]) -> Dict[str, Any]:
    """Simple deep-copy for dicts (sufficient for our config structures)."""
    import copy

    return copy.deepcopy(d)


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    """Read environment variables matching ``_ENV_OVERRIDES`` and apply
    them to the config dict.  Type coercion is attempted for bool and int
    values where the default is of that type.
    """
    for env_var, config_key in _ENV_OVERRIDES.items():
        raw_value = os.environ.get(env_var)
        if raw_value is None:
            continue

        # Try to coerce to the same type as the default for consistency
        current = _deep_get(cfg, config_key)
        if current is not None:
            try:
                if isinstance(current, bool):
                    coerced = raw_value.lower() in ("1", "true", "yes", "on")
                elif isinstance(current, int):
                    coerced = int(raw_value)
                elif isinstance(current, float):
                    coerced = float(raw_value)
                else:
                    coerced = raw_value
            except (ValueError, TypeError):
                coerced = raw_value
        else:
            coerced = raw_value

        _deep_set(cfg, config_key, coerced)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load and parse a YAML config file.  Returns an empty dict if yaml
    is not installed or the file does not exist."""
    if yaml is None:
        return {}
    if not path.is_file():
        return {}
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file, merge with defaults, and apply
    environment variable overrides.

    Args:
        path: Path to the YAML configuration file.  If ``None`` (default)
              the loader searches for ``config.yaml`` in the project root
              (parent of the ``shared/`` directory).  If the file doesn't
              exist the built-in defaults are used.

    Returns:
        A :class:`Config` instance.
    """
    cfg = _deep_copy(_DEFAULT_CONFIG)

    if path is None:
        # Walk up from this file's location to find the project root
        here = Path(__file__).resolve().parent
        candidate = here.parent / "config.yaml"
        if candidate.is_file():
            path = candidate

    if path is not None:
        p = Path(path)
        if p.is_file():
            yaml_cfg = _load_yaml(p)
            if yaml_cfg:
                cfg = _deep_merge(cfg, yaml_cfg)

    _apply_env_overrides(cfg)
    return Config(data=cfg)


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overlay* into *base* (modifies base in-place)."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
