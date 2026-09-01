"""Shared entry-point setup for the CLI scripts.

Importing this puts ``backend/`` on ``sys.path`` so the scripts can import ``app``. Everything
else a script needs at start-up -- the ``.env`` file, settings, logging -- comes from
:func:`bootstrap`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.config import Settings, get_settings, load_dotenv
from app.utils.logging import configure_logging


def bootstrap(*, json_output: bool = False) -> Settings:
    """Load the environment and settings, configure logging, and hand back the settings.

    Args:
        json_output: Emit JSON log lines rather than console output. Scripts that print a
            machine-readable report on stdout pass their own ``--json`` flag through, so log
            lines and report stay in the same format.
    """
    load_dotenv()
    settings = get_settings()
    configure_logging(settings.app.log_level, json_output=json_output)
    return settings
