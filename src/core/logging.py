"""Centralized logging configuration for CVGAI Group 4.

Provides a unified logging setup across application entry points and CLI tools.
Ensures standard formatting, environment-driven log levels, and library etiquette
with NullHandler.

Ref: Docs/implementation_plan.md, section 2.
"""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from typing import Any

#: Default log format with timestamps, severity, module name, and message.
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Valid log levels.
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def get_log_level(env_var: str = "LOG_LEVEL", default: str = "INFO") -> str:
    """Resolve and validate the effective log level from the environment.

    Args:
        env_var: Environment variable name to read.
        default: Fallback log level if unset or invalid.

    Returns:
        Upper-case valid log level name.
    """
    raw = os.getenv(env_var, default).strip().upper()
    return raw if raw in VALID_LOG_LEVELS else default.upper()


def setup_logging(
    level: str | None = None,
    stream: Any = None,
) -> None:
    """Configure application-wide console logging.

    Args:
        level: Explicit log level string (e.g., 'DEBUG', 'INFO'). When None,
            reads from the LOG_LEVEL environment variable (default: 'INFO').
        stream: Output stream for the console handler. Defaults to sys.stderr.
    """
    effective_level = level.strip().upper() if level and level.strip().upper() in VALID_LOG_LEVELS else get_log_level()

    target_stream = stream or sys.stderr

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": DEFAULT_LOG_FORMAT,
                "datefmt": DEFAULT_DATE_FORMAT,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": target_stream,
                "formatter": "standard",
            },
        },
        "loggers": {
            "src": {
                "handlers": ["console"],
                "level": effective_level,
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": "WARNING",
        },
    }
    logging.config.dictConfig(config)
