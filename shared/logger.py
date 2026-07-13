"""
Aurora Trader — Async Structured Logger.

Provides a configurable, async-compatible logger with:
- JSON structured output (optional)
- File rotation (size-based and time-based)
- Multiple log levels
- A dedicated *trade* logging channel for trade audit trails
- Per-module level overrides
"""

from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union

# ---------------------------------------------------------------------------
# Logger registry — keep track of created loggers so we can reconfigure them
# ---------------------------------------------------------------------------

_loggers: Dict[str, "AuroraLogger"] = {}


def _default_log_dir() -> Path:
    """Return the default log directory relative to the project root."""
    # Walk up from shared/ to find the project root
    here = Path(__file__).resolve().parent
    return here.parent / "logs"


# ---------------------------------------------------------------------------
# Custom trade channel — a separate logger namespace
# ---------------------------------------------------------------------------

TRADE_LOG_NAME = "aurora.trade"
SYSTEM_LOG_NAME = "aurora.system"


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Output log records as newline-delimited JSON."""

    def format(self, record: logging.LogRecord) -> str:
        obj: Dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "name": record.name,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            obj["extra"] = record.extra
        if record.exc_info and record.exc_info[0] is not None:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str)


# ---------------------------------------------------------------------------
# Aurora Logger Wrapper
# ---------------------------------------------------------------------------


class AuroraLogger:
    """Wraps a standard Python logger with convenience methods and a
    dedicated trade-logging channel.

    Usage::

        log = get_logger("trading_server.strategy")
        log.info("Strategy started")
        log.trade("BUY", symbol="BTCUSDT", price="45000", qty="0.1")
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_dir: Optional[Union[str, Path]] = None,
        json_output: bool = False,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 10,
        use_timed_rotation: bool = False,
        when: str = "midnight",
    ) -> None:
        self._name = name
        self._log_dir = Path(log_dir) if log_dir else _default_log_dir()
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # --- Main system logger ---
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers.clear()
        self._logger.propagate = False

        # Console handler (always active)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        self._logger.addHandler(console_handler)

        # File handler (rotating)
        log_file = self._log_dir / f"{name.replace('.', '_')}.log"
        if use_timed_rotation:
            file_handler: logging.Handler = TimedRotatingFileHandler(
                log_file,
                when=when,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        file_handler.setLevel(level)
        if json_output:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        self._logger.addHandler(file_handler)

        # --- Trade-specific channel ---
        self._trade_logger = logging.getLogger(f"{TRADE_LOG_NAME}.{name}")
        self._trade_logger.setLevel(level)
        self._trade_logger.handlers.clear()
        self._trade_logger.propagate = False

        trade_log_file = self._log_dir / "trades.log"
        trade_handler = RotatingFileHandler(
            trade_log_file,
            maxBytes=max_bytes * 2,  # trade logs can be larger
            backupCount=backup_count,
            encoding="utf-8",
        )
        trade_handler.setLevel(level)
        if json_output:
            trade_handler.setFormatter(JsonFormatter())
        else:
            trade_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | TRADE | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        self._trade_logger.addHandler(trade_handler)

        # --- Store reference ---
        _loggers[name] = self

    # ------------------------------------------------------------------
    # Passthrough delegates
    # ------------------------------------------------------------------

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def trade_logger(self) -> logging.Logger:
        return self._trade_logger

    def debug(self, msg: str, **extra: Any) -> None:
        self._log(logging.DEBUG, msg, extra)

    def info(self, msg: str, **extra: Any) -> None:
        self._log(logging.INFO, msg, extra)

    def warning(self, msg: str, **extra: Any) -> None:
        self._log(logging.WARNING, msg, extra)

    def error(self, msg: str, **extra: Any) -> None:
        self._log(logging.ERROR, msg, extra)

    def critical(self, msg: str, **extra: Any) -> None:
        self._log(logging.CRITICAL, msg, extra)

    def exception(self, msg: str, **extra: Any) -> None:
        self._logger.exception(msg, extra={"extra": extra} if extra else None)

    def trade(self, action: str, **fields: Any) -> None:
        """Log a trade-specific event to the trade channel.

        Example::

            logger.trade("OPEN", symbol="BTCUSDT", side="buy",
                         price="45000.0", qty="0.1")
        """
        extra = self._build_extra(fields)
        trade_msg = f"[{action}] {' | '.join(f'{k}={v}' for k, v in fields.items())}"
        self._trade_logger.info(trade_msg, extra=extra)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log(self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        extra_dict = self._build_extra(extra)
        self._logger.log(level, msg, extra=extra_dict)

    @staticmethod
    def _build_extra(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"extra": extra or {}}

    def set_level(self, level: int) -> None:
        self._logger.setLevel(level)
        self._trade_logger.setLevel(level)

    def close(self) -> None:
        """Close and remove all handlers."""
        for h in self._logger.handlers[:]:
            h.close()
            self._logger.removeHandler(h)
        for h in self._trade_logger.handlers[:]:
            h.close()
            self._trade_logger.removeHandler(h)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    json_output: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 10,
    use_timed_rotation: bool = False,
    when: str = "midnight",
) -> AuroraLogger:
    """Get (or create) an :class:`AuroraLogger` for *name*.

    Calling ``get_logger`` multiple times with the same *name* returns the
    same instance, so you can call it at module level without worrying about
    duplicate handlers.
    """
    if name in _loggers:
        return _loggers[name]

    return AuroraLogger(
        name=name,
        level=level,
        log_dir=log_dir,
        json_output=json_output,
        max_bytes=max_bytes,
        backup_count=backup_count,
        use_timed_rotation=use_timed_rotation,
        when=when,
    )


# ---------------------------------------------------------------------------
# Convenience: set global log level
# ---------------------------------------------------------------------------


def set_global_level(level: int) -> None:
    """Update the log level of every registered :class:`AuroraLogger`."""
    for logger in _loggers.values():
        logger.set_level(level)
