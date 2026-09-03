"""Tests for the Google AI Studio client (Gemini 3.8 Flash on generateContent).

Google AI Studio is the third inference provider (D29), now calling an ordinary chat model with
the clip inlined rather than the Live API's transcription model (D31). These tests cover
response parsing, prompt steering, retry handling and dry-run safety.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import LlmDisabledError, LlmRequestFailed, LlmRouteNotConfigured
from app.llm.google import GoogleClient, parse_transcript

ROUTE = "asr_gemini_flash"


def _body(*texts: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": t} for t in texts]}}]}


def test_parse_transcript_joins_the_parts_of_a_candidate() -> None:
    assert parse_transcript(_body("आजको ", "meeting मा")) == "आजको meeting मा"


def test_parse_transcript_strips_surrounding_whitespace() -> None:
    assert parse_transcript(_body("  गुगल भन्छ  ")) == "गुगल भन्छ"


def test_parse_transcript_skips_a_candidate_carrying_no_text() -> None:
    """A blocked or tool-only candidate yields nothing; the next one is tried."""
    body = {
        "candidates": [
            {"finishReason": "SAFETY", "content": {"parts": []}},
            {"content": {"parts": [{"text": "real transcript"}]}},
        ]
    }
    assert parse_transcript(body) == "real transcript"


def test_parse_transcript_handles_empty_or_malformed() -> None:
    assert parse_transcript({}) == ""
    assert parse_transcript({"candidates": []}) == ""
    assert parse_transcript({"candidates": [{}]}) == ""
    assert parse_transcript({"candidates": [{"content": {"parts": [{"nope": 1}]}}]}) == ""


# --- Tests requiring database session ------------------------------------------------------


def routes(**kwargs: Any) -> LlmRoutes:
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 3,
        "retry_backoff_seconds": 0.0,
        "routes": {
            ROUTE: LlmRoute(
                provider="google",
                api="audio_chat",
                model="gemini-3.8-flash",
                system_id="gemini-3.8-flash",
                language="ne",
            )
        },
    }
    return LlmRoutes(**{**base, **kwargs})


def make_client(
    session: Session, handler, *, config: LlmRoutes | None = None, api_key: str = "test-key"
) -> GoogleClient:
    return GoogleClient(
        session,
        config=config or routes(),
        api_key=api_key,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.db
def test_a_disabled_google_client_refuses_to_transcribe(
    db_session: Session, tmp_path: Path
) -> None:
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"mock audio")
    client = make_client(db_session, lambda r: httpx.Response(200), config=routes(enabled=False))
    with pytest.raises(LlmDisabledError):
        client.transcribe(clip, route=ROUTE)


@pytest.mark.db
def test_an_unknown_google_route_is_refused(db_session: Session, tmp_path: Path) -> None:
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"mock audio")
    client = make_client(db_session, lambda r: httpx.Response(200))
    with pytest.raises(LlmRouteNotConfigured, match="nonexistent"):
        client.transcribe(clip, route="nonexistent")


@pytest.mark.db
def test_missing_google_api_key_raises(db_session: Session, tmp_path: Path) -> None:
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"mock audio")
    client = make_client(db_session, lambda r: httpx.Response(200), api_key="")
    with pytest.raises(LlmRequestFailed, match="GOOGLE_API_KEY is not set"):
        client.transcribe(clip, route=ROUTE)


@pytest.mark.db
def test_a_google_dry_run_never_calls_the_network(db_session: Session, tmp_path: Path) -> None:
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"mock audio")
    network_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal network_called
        network_called = True
        return httpx.Response(200, json={})

    client = make_client(db_session, handler, config=routes(dry_run=True))
    result = client.transcribe(clip, route=ROUTE)
    assert not network_called
    assert result.dry_run is True
    assert result.text


@pytest.mark.db
def test_a_successful_transcription_returns_verbatim_text(
    db_session: Session, tmp_path: Path
) -> None:
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"mock audio")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_body("आजको meeting मा data हेर्यौं"))

    client = make_client(db_session, handler)
    result = client.transcribe(clip, route=ROUTE)

    assert result.text == "आजको meeting मा data हेर्यौं"
    assert result.model == "gemini-3.8-flash"
    assert len(seen) == 1
    assert seen[0].headers.get("x-goog-api-key") == "test-key"


@pytest.mark.db
def test_the_request_targets_generate_content_for_the_configured_model(
    db_session: Session, tmp_path: Path
) -> None:
    """Not the Live API's /interactions endpoint, whose 100 RPD cap is why D31 exists."""
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"mock audio")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_body("ok"))

    make_client(db_session, handler).transcribe(clip, route=ROUTE)
    url = str(seen[0].url)
    assert url.endswith("/models/gemini-3.8-flash:generateContent")
    assert "interactions" not in url


@pytest.mark.db
def test_word_timestamps_never_come_from_the_model(db_session: Session, tmp_path: Path) -> None:
    """Word spans for this system are the forced aligner's job (D32), not Gemini's."""
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"mock audio")
    client = make_client(db_session, lambda r: httpx.Response(200, json=_body("गुगल भन्छ")))
    assert client.transcribe(clip, route=ROUTE).words is None


@pytest.mark.db
def test_the_transcript_prompt_reaches_the_model(db_session: Session, tmp_path: Path) -> None:
    """Regression guard: the prompt argument used to be dropped on the Google branch.

    The Live API's transcription model took no free-text prompt, so the corpus policy was
    never stated to Gemini at all. generateContent obeys one.
    """
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"mock audio")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_body("ok"))

    client = make_client(db_session, handler)
    client.transcribe(clip, route=ROUTE, prompt="Write Nepali in Devanagari.")

    parts = json.loads(seen[0].content)["contents"][0]["parts"]
    assert "Write Nepali in Devanagari." in parts[0]["text"]
    # The route's language code has no generateContent parameter, so it rides on the prompt.
    assert "ne" in parts[0]["text"]


@pytest.mark.db
def test_the_clip_is_inlined_as_base64_flac(db_session: Session, tmp_path: Path) -> None:
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"mock audio bytes")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_body("ok"))

    make_client(db_session, handler).transcribe(clip, route=ROUTE, prompt="p")

    parts = json.loads(seen[0].content)["contents"][0]["parts"]
    inline = parts[-1]["inline_data"]
    assert inline["mime_type"] == "audio/flac"
    assert base64.b64decode(inline["data"]) == b"mock audio bytes"


@pytest.mark.db
def test_a_retryable_status_is_retried_then_succeeds(db_session: Session, tmp_path: Path) -> None:
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"mock audio")
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={})
        return httpx.Response(200, json=_body("गुगल भन्छ"))

    result = make_client(db_session, handler).transcribe(clip, route=ROUTE)
    assert attempts == 2
    assert result.text == "गुगल भन्छ"
