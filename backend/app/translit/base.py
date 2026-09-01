"""The transliteration provider interface.

The annotator types Latin and cannot type Devanagari directly, so this is not a convenience
feature -- it is the input method, and editing throughput depends on it entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TranslitProvider(ABC):
    """Turns a Latin token into ranked Devanagari candidates."""

    #: Short identifier recorded in ``translit_cache.provider``.
    name: str = "provider"

    @abstractmethod
    def suggest(self, latin_token: str) -> list[str]:
        """Return ranked Devanagari candidates, best first.

        Implementations return an empty list when they have nothing to offer. Raising is also
        acceptable -- the service treats a failure and an empty result the same way, by moving on
        to the next provider.
        """
