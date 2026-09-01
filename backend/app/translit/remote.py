"""Google Input Tools transliteration.

The endpoint is undocumented and may change without notice, so it lives behind the provider
interface, is called from the backend rather than the browser, has a short timeout, and degrades to
returning nothing rather than raising into the editor.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings, get_settings
from app.translit.base import TranslitProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Nepali transliteration input method code.
INPUT_METHOD = "ne-t-i0-und"


class GoogleInputToolsProvider(TranslitProvider):
    """Calls ``inputtools.google.com/request`` for Devanagari candidates."""

    name = "remote"

    def __init__(
        self, settings: Settings | None = None, client: httpx.Client | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.settings.translit.remote_timeout_seconds)
        return self._client

    @staticmethod
    def _parse(payload: Any) -> list[str]:
        """Extract candidates from ``["SUCCESS", [[token, [candidates], ...]]]``."""
        try:
            if not isinstance(payload, list) or payload[0] != "SUCCESS":
                return []
            candidates = payload[1][0][1]
            return [str(c) for c in candidates if isinstance(c, str)]
        except (IndexError, KeyError, TypeError):
            return []

    def suggest(self, latin_token: str) -> list[str]:
        """Return ranked candidates, or an empty list on any failure."""
        token = latin_token.strip()
        if not token:
            return []
        params = {
            "text": token,
            "itc": INPUT_METHOD,
            "num": str(self.settings.translit.max_candidates),
            "cp": "0",
            "cs": "1",
            "ie": "utf-8",
            "oe": "utf-8",
        }
        try:
            response = self._get_client().get(
                self.settings.translit.remote_endpoint,
                params=params,
                timeout=self.settings.translit.remote_timeout_seconds,
            )
            if response.status_code != 200:
                logger.info("translit_remote_bad_status", status=response.status_code)
                return []
            return self._parse(response.json())
        except Exception as exc:
            # A transliteration lookup must never surface as an error dialog mid-edit.
            logger.info("translit_remote_unavailable", error=str(exc)[:200])
            return []
