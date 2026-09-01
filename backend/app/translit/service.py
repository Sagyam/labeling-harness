"""Transliteration lookup with a persistent cache and correction memory.

The cache is always consulted first. The same tokens recur constantly in one speaker community, so
after a few hundred segments most lookups never leave the database -- and the accumulated table is
itself worth keeping: it is a romanization lexicon for this community.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import TranslitCacheEntry
from app.translit.base import TranslitProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)


def build_providers(settings: Settings | None = None) -> list[TranslitProvider]:
    """Instantiate providers in the configured order."""
    settings = settings or get_settings()
    providers: list[TranslitProvider] = []
    for name in settings.translit.provider_order:
        if name == "remote":
            from app.translit.remote import GoogleInputToolsProvider

            providers.append(GoogleInputToolsProvider(settings))
        elif name == "offline":
            from app.translit.offline import OfflineTranslitProvider

            providers.append(OfflineTranslitProvider())
    return providers


class TransliterationService:
    """Cache-first transliteration over an ordered list of providers."""

    def __init__(
        self,
        session: Session,
        *,
        providers: Sequence[TranslitProvider] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.providers = (
            list(providers) if providers is not None else build_providers(self.settings)
        )

    @staticmethod
    def _key(latin_token: str) -> str:
        return latin_token.strip().lower()

    def suggest(self, latin_token: str) -> list[str]:
        """Return ranked Devanagari candidates for a Latin token.

        A cached token is answered from Postgres without touching any provider. Otherwise providers
        are tried in order; a provider that fails or returns nothing simply yields to the next, so a
        network outage degrades to offline transliteration rather than to an error.
        """
        key = self._key(latin_token)
        if not key:
            return []

        cached = self.session.get(TranslitCacheEntry, key)
        if cached is not None:
            cached.hit_count += 1
            self.session.flush()
            return list(cached.candidates_jsonb)[: self.settings.translit.max_candidates]

        for provider in self.providers:
            try:
                candidates = provider.suggest(key)
            except Exception as exc:
                logger.info(
                    "translit_provider_failed", provider=provider.name, error=str(exc)[:200]
                )
                continue
            if not candidates:
                continue
            capped = list(dict.fromkeys(candidates))[: self.settings.translit.max_candidates]
            self.session.add(
                TranslitCacheEntry(
                    latin_token=key, candidates_jsonb=capped, provider=provider.name, hit_count=0
                )
            )
            self.session.flush()
            return capped
        return []

    def record_choice(self, latin_token: str, devanagari: str) -> None:
        """Remember that this token was resolved to this form, and rank it first next time.

        This is the correction memory: within one project the same romanization almost always means
        the same word, so the annotator should not have to pick it twice.
        """
        key = self._key(latin_token)
        if not key or not devanagari:
            return
        entry = self.session.get(TranslitCacheEntry, key)
        if entry is None:
            self.session.add(
                TranslitCacheEntry(
                    latin_token=key,
                    candidates_jsonb=[devanagari],
                    provider="correction",
                    hit_count=0,
                )
            )
        else:
            remaining = [c for c in entry.candidates_jsonb if c != devanagari]
            entry.candidates_jsonb = [devanagari, *remaining][
                : self.settings.translit.max_candidates
            ]
        self.session.flush()
