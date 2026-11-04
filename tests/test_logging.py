"""Unit tests for the centralized logging module."""

from __future__ import annotations

import importlib
import io
import logging

import pytest

import src
from src.core.logging import (
    VALID_LOG_LEVELS,
    get_log_level,
    setup_logging,
)


def test_package_has_null_handler() -> None:
    """The root 'src' package logger has a NullHandler attached upon import."""
    src_logger = logging.getLogger("src")
    src_logger.handlers.clear()
    importlib.reload(src)
    assert any(isinstance(h, logging.NullHandler) for h in src_logger.handlers)


def test_get_log_level_defaults_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_log_level returns INFO when LOG_LEVEL is not set."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert get_log_level() == "INFO"


def test_get_log_level_reads_valid_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_log_level reads and normalizes case from LOG_LEVEL."""
    for level in VALID_LOG_LEVELS:
        monkeypatch.setenv("LOG_LEVEL", level.lower())
        assert get_log_level() == level


def test_get_log_level_invalid_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid LOG_LEVEL values safely fall back to the default."""
    monkeypatch.setenv("LOG_LEVEL", "nonexistent_level")
    assert get_log_level(default="WARNING") == "WARNING"


def test_setup_logging_configures_stream() -> None:
    """setup_logging sends formatted output to the supplied stream."""
    stream = io.StringIO()
    setup_logging(level="DEBUG", stream=stream)

    test_logger = logging.getLogger("src.test_module")
    test_logger.debug("Mensagem de depuracao")

    output = stream.getvalue()
    assert "Mensagem de depuracao" in output
    assert "[DEBUG]" in output
    assert "src.test_module:" in output


def test_setup_logging_filters_by_level() -> None:
    """setup_logging suppresses messages below the configured severity level."""
    stream = io.StringIO()
    setup_logging(level="WARNING", stream=stream)

    test_logger = logging.getLogger("src.test_module_filter")
    test_logger.info("Informacao ignorada")
    test_logger.warning("Alerta emitido")

    output = stream.getvalue()
    assert "Informacao ignorada" not in output
    assert "Alerta emitido" in output
    assert "[WARNING]" in output
