"""Tests for structured logging setup."""

from __future__ import annotations

import io
import json
import logging
import sys

from app.utils.logging import configure_logging, get_logger


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("INFO")
    assert logging.getLogger().level == logging.INFO


def test_logger_emits_json_with_event_and_level(capsys) -> None:
    configure_logging("INFO", json_output=True)
    get_logger("test").info("imported_segments", count=3)
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert payload["event"] == "imported_segments"
    assert payload["count"] == 3
    assert payload["level"] == "info"
    assert "timestamp" in payload


def test_log_level_is_respected(capsys) -> None:
    configure_logging("WARNING", json_output=True)
    get_logger("test").info("should_not_appear")
    assert capsys.readouterr().out.strip() == ""


def test_logging_survives_the_configure_time_stdout_being_closed(capsys) -> None:
    """Loggers must not pin the stream that was installed when configure ran.

    Regression: the factory was built with ``file=sys.stdout``, so configuring under
    one pytest capture and logging under a later one wrote to a closed file and
    raised ``ValueError: I/O operation on closed file``.
    """
    stale = io.StringIO()
    original = sys.stdout
    try:
        sys.stdout = stale
        configure_logging("INFO", json_output=True)
    finally:
        sys.stdout = original
    stale.close()

    get_logger("test").warning("emitted_after_stdout_closed", key="clip")

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["event"] == "emitted_after_stdout_closed"
    assert payload["level"] == "warning"
