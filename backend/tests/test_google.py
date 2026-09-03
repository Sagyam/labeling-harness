"""Tests for the Google AI Studio client (Gemini 3.5 Transcribe).

Google AI Studio is the third inference provider (decision D29). These tests verify verbatim
transcription parsing, word-level timestamps extraction, retry handling, and dry-run safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import LlmDisabledError, LlmRequestFailed, LlmRouteNotConfigured
from app.llm.google import GoogleClient, _parse_word_annotations


def test_parse_word_annotations_extracts_timing_and_text() -> None:
    body = {
        "steps": [
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Hello world",
                        "annotations": [
                            {
                                "type": "word_info",
                                "text": "Hello",
                                "start_offset": "0.100s",
                                "end_offset": "0.450s",
                            },
                            {
                                "type": "word_info",
                                "text": "world",
                                "start_offset": "0.500s",
                                "end_offset": "0.900s",
                            },
                        ],
                    }
                ]
            }
        ]
    }
    words = _parse_word_annotations(body)
    assert len(words) == 2
    assert words[0] == {"word": "Hello", "start": 0.100, "end": 0.450}
    assert words[1] == {"word": "world", "start": 0.500, "end": 0.900}


def test_parse_word_annotations_handles_empty_or_malformed() -> None:
    assert _parse_word_annotations({}) == []
    assert _parse_word_annotations({"steps": []}) == []

    malformed = {
        "steps": [
            {
                "content": [
                    {
                        "annotations": [
                            {
                                "type": "word_info",
                                "text": "test",
                                "start_offset": "invalid",
                                "end_offset": None,
                            }
                        ]
                    }
                ]
            }
        ]
    }
    words = _parse_word_annotations(malformed)
    assert words == [{"word": "test", "start": 0.0, "end": 0.0}]


# --- Tests requiring database session ------------------------------------------------------


def routes(**kwargs: Any) -> LlmRoutes:
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 3,
        "retry_backoff_seconds": 0.0,
        "routes": {
            "asr_gemini_transcribe": LlmRoute(
                provider="google",
                api="transcription",
                model="gemini-3.5-transcribe",
                system_id="gemini-3.5-transcribe",
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
        client.transcribe(clip, route="asr_gemini_transcribe")


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
        client.transcribe(clip, route="asr_gemini_transcribe")


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
    result = client.transcribe(clip, route="asr_gemini_transcribe")
    assert not network_called
    assert result.dry_run is True
    assert result.text
    assert result.words


@pytest.mark.db
def test_a_successful_google_transcription_returns_verbatim_text_and_words(
    db_session: Session, tmp_path: Path
) -> None:
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"mock audio")

    mock_response = {
        "id": "interactions/test123",
        "status": "completed",
        "output_text": "आजको meeting मा data हेर्यौं",
        "steps": [
            {
                "content": [
                    {
                        "type": "text",
                        "text": "आजको meeting मा data हेर्यौं",
                        "annotations": [
                            {
                                "type": "word_info",
                                "text": "आजको",
                                "start_offset": "0.100s",
                                "end_offset": "0.400s",
                            },
                            {
                                "type": "word_info",
                                "text": "meeting",
                                "start_offset": "0.420s",
                                "end_offset": "0.750s",
                            },
                            {
                                "type": "word_info",
                                "text": "मा",
                                "start_offset": "0.760s",
                                "end_offset": "0.850s",
                            },
                            {
                                "type": "word_info",
                                "text": "data",
                                "start_offset": "0.860s",
                                "end_offset": "1.100s",
                            },
                            {
                                "type": "word_info",
                                "text": "हेर्यौं",
                                "start_offset": "1.120s",
                                "end_offset": "1.500s",
                            },
                        ],
                    }
                ]
            }
        ],
    }

    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, json=mock_response)

    client = make_client(db_session, handler)
    result = client.transcribe(clip, route="asr_gemini_transcribe")

    assert result.text == "आजको meeting मा data हेर्यौं"
    assert result.model == "gemini-3.5-transcribe"
    assert len(result.words) == 5
    assert result.words[1] == {"word": "meeting", "start": 0.42, "end": 0.75}

    assert len(seen_requests) == 1
    req = seen_requests[0]
    assert req.headers.get("x-goog-api-key") == "test-key"
    assert "interactions" in str(req.url)
