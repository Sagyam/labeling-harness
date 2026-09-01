"""Rule-based offline transliteration.

Quality on casual romanization is lower than the remote service -- podcast speakers write "kura",
not the ITRANS "kuraa" -- so the provider generates a small family of spellings and returns them
ranked. It has no network dependency, which is the point: the editor keeps working offline.
"""

from __future__ import annotations

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from app.translit.base import TranslitProvider

_VOWELS = "aeiou"
#: Schemes to try, best-behaved first.
_SCHEMES = (sanscript.ITRANS, sanscript.HK, sanscript.OPTITRANS)


def _variants(token: str) -> list[str]:
    """Spellings to feed the transliterator, most likely first."""
    forms = [token]
    if token and token[-1] not in _VOWELS:
        # Casual romanization drops the inherent final vowel: "kura" is कुरा, "kur" often कुर.
        forms.append(f"{token}a")
    if token.endswith("a") and not token.endswith("aa"):
        forms.insert(1, f"{token}a")
    return forms


class OfflineTranslitProvider(TranslitProvider):
    """Transliterates with :mod:`indic_transliteration`, no network required."""

    name = "offline"

    def suggest(self, latin_token: str) -> list[str]:
        """Return ranked Devanagari candidates derived from ITRANS-family schemes."""
        token = latin_token.strip().lower()
        if not token or not token.isascii():
            return []

        candidates: list[str] = []
        for form in _variants(token):
            for scheme in _SCHEMES:
                try:
                    rendered = transliterate(form, scheme, sanscript.DEVANAGARI)
                except Exception:  # pragma: no cover - defensive; the library is permissive
                    continue
                if rendered and rendered != form and rendered not in candidates:
                    candidates.append(rendered)
        return candidates
