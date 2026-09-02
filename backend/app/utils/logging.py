"""Structured logging.

One configuration call at process start; every module then uses :func:`get_logger`.
Timestamps are UTC ISO-8601 so log lines line up with the database.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure stdlib logging and structlog.

    Args:
        level: Standard logging level name.
        json_output: Emit JSON lines when true, human-readable console output when false.
    """
    global _configured
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        # Deliberately no ``file=``: PrintLogger only resolves ``sys.stdout`` at write
        # time when it was constructed without one. Passing ``file=sys.stdout`` here
        # would pin whatever stream is installed at configure time, and writes would
        # fail once that stream is replaced or closed.
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
