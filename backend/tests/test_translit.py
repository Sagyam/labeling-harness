"""Tests for the Latin -> Devanagari input helper."""

from __future__ import annotations

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import load_settings
from app.models import TranslitCacheEntry
from app.translit.base import TranslitProvider
from app.translit.mock import StaticTranslitProvider
from app.translit.offline import OfflineTranslitProvider
from app.translit.remote import GoogleInputToolsProvider
from app.translit.service import TransliterationService

DEVANAGARI = range(0x0900, 0x0980)


def is_devanagari(text: str) -> bool:
    return any(ord(ch) in DEVANAGARI for ch in text)


# --- the interface and its implementations ----------------------------------------------


def test_mock_provider_satisfies_the_interface() -> None:
    provider = StaticTranslitProvider({"kura": ["कुरा", "कुर"]})
    assert isinstance(provider, TranslitProvider)
    assert provider.suggest("kura") == ["कुरा", "कुर"]
    assert provider.suggest("unknown") == []


def test_offline_provider_returns_devanagari() -> None:
    candidates = OfflineTranslitProvider().suggest("kura")
    assert candidates
    assert all(is_devanagari(c) for c in candidates)


def test_offline_provider_is_deterministic() -> None:
    provider = OfflineTranslitProvider()
    assert provider.suggest("garchhu") == provider.suggest("garchhu")


def test_offline_provider_handles_an_empty_token() -> None:
    assert OfflineTranslitProvider().suggest("") == []


def test_offline_provider_returns_unique_candidates() -> None:
    candidates = OfflineTranslitProvider().suggest("ma")
    assert len(candidates) == len(set(candidates))


def test_offline_provider_offers_a_final_vowel_variant() -> None:
    """Casual romanization drops the final vowel constantly: 'kura' is कुरा, not कुर."""
    assert "कुरा" in OfflineTranslitProvider().suggest("kura")


def test_remote_provider_parses_a_google_response() -> None:
    payload = ["SUCCESS", [["kura", ["कुरा", "कुर", "कुराः"], [], {}]]]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["itc"] == "ne-t-i0-und"
        assert request.url.params["text"] == "kura"
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GoogleInputToolsProvider(settings=load_settings(), client=client)
    assert provider.suggest("kura") == ["कुरा", "कुर", "कुराः"]


def test_remote_provider_returns_nothing_on_a_failed_response() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    provider = GoogleInputToolsProvider(settings=load_settings(), client=client)
    assert provider.suggest("kura") == []


def test_remote_provider_returns_nothing_on_a_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GoogleInputToolsProvider(settings=load_settings(), client=client)
    assert provider.suggest("kura") == []


def test_remote_provider_tolerates_an_unexpected_payload() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"unexpected": True}))
    )
    provider = GoogleInputToolsProvider(settings=load_settings(), client=client)
    assert provider.suggest("kura") == []


# --- the service: cache first, then providers in order -----------------------------------


class CountingProvider(TranslitProvider):
    """Records how many times it was consulted."""

    name = "counting"

    def __init__(self, candidates: list[str]) -> None:
        self.candidates = candidates
        self.calls = 0

    def suggest(self, latin_token: str) -> list[str]:
        self.calls += 1
        return list(self.candidates)


class ExplodingProvider(TranslitProvider):
    """Fails the way a flaky network dependency does."""

    name = "exploding"

    def __init__(self) -> None:
        self.calls = 0

    def suggest(self, latin_token: str) -> list[str]:
        self.calls += 1
        raise RuntimeError("network is down")


@pytest.mark.db
def test_first_lookup_consults_the_provider_and_caches(db_session: Session) -> None:
    provider = CountingProvider(["कुरा"])
    service = TransliterationService(db_session, providers=[provider], settings=load_settings())
    assert service.suggest("kura") == ["कुरा"]
    entry = db_session.get(TranslitCacheEntry, "kura")
    assert entry is not None
    assert entry.candidates_jsonb == ["कुरा"]
    assert entry.provider == "counting"


@pytest.mark.db
def test_cache_hit_never_touches_a_provider(db_session: Session) -> None:
    provider = CountingProvider(["कुरा"])
    service = TransliterationService(db_session, providers=[provider], settings=load_settings())
    service.suggest("kura")
    assert provider.calls == 1
    assert service.suggest("kura") == ["कुरा"]
    assert provider.calls == 1, "a cached token must not reach the network"


@pytest.mark.db
def test_cache_lookup_is_case_insensitive(db_session: Session) -> None:
    provider = CountingProvider(["कुरा"])
    service = TransliterationService(db_session, providers=[provider], settings=load_settings())
    service.suggest("Kura")
    service.suggest("KURA")
    assert provider.calls == 1
    assert db_session.get(TranslitCacheEntry, "kura") is not None


@pytest.mark.db
def test_cache_hits_are_counted(db_session: Session) -> None:
    service = TransliterationService(
        db_session, providers=[CountingProvider(["कुरा"])], settings=load_settings()
    )
    service.suggest("kura")
    service.suggest("kura")
    service.suggest("kura")
    assert db_session.get(TranslitCacheEntry, "kura").hit_count == 2


@pytest.mark.db
def test_a_failing_provider_falls_through_to_the_next(db_session: Session) -> None:
    remote = ExplodingProvider()
    offline = CountingProvider(["कुरा"])
    service = TransliterationService(
        db_session, providers=[remote, offline], settings=load_settings()
    )
    assert service.suggest("kura") == ["कुरा"]
    assert remote.calls == 1
    assert offline.calls == 1


@pytest.mark.db
def test_an_empty_provider_result_falls_through_to_the_next(db_session: Session) -> None:
    empty = CountingProvider([])
    offline = CountingProvider(["कुरा"])
    service = TransliterationService(
        db_session, providers=[empty, offline], settings=load_settings()
    )
    assert service.suggest("kura") == ["कुरा"]


@pytest.mark.db
def test_nothing_is_cached_when_every_provider_fails(db_session: Session) -> None:
    service = TransliterationService(
        db_session, providers=[ExplodingProvider()], settings=load_settings()
    )
    assert service.suggest("kura") == []
    assert db_session.get(TranslitCacheEntry, "kura") is None


@pytest.mark.db
def test_candidates_are_capped_at_the_configured_maximum(db_session: Session) -> None:
    many = [f"क{i}" for i in range(20)]
    service = TransliterationService(
        db_session, providers=[CountingProvider(many)], settings=load_settings()
    )
    assert len(service.suggest("kura")) == load_settings().translit.max_candidates


@pytest.mark.db
def test_correction_memory_ranks_a_previous_choice_first(db_session: Session) -> None:
    """If the annotator resolved this token before, that form is almost certainly right again."""
    service = TransliterationService(
        db_session, providers=[CountingProvider(["कुर", "कुरा"])], settings=load_settings()
    )
    assert service.suggest("kura")[0] == "कुर"
    service.record_choice("kura", "कुरा")
    assert service.suggest("kura")[0] == "कुरा"


@pytest.mark.db
def test_recording_an_unseen_choice_creates_an_entry(db_session: Session) -> None:
    service = TransliterationService(db_session, providers=[], settings=load_settings())
    service.record_choice("naya", "नयाँ")
    assert db_session.get(TranslitCacheEntry, "naya").candidates_jsonb == ["नयाँ"]


@pytest.mark.db
def test_an_empty_token_is_never_cached(db_session: Session) -> None:
    service = TransliterationService(
        db_session, providers=[CountingProvider(["क"])], settings=load_settings()
    )
    assert service.suggest("   ") == []
    assert db_session.scalar(sa.select(sa.func.count()).select_from(TranslitCacheEntry)) == 0
