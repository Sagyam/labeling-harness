"""Tests for structured logging setup."""

from __future__ import annotations

import json
import logging

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
