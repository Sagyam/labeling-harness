"""Tests for the Google AI Studio / Gemini API client.

Gemini serves both speech recognition shapes the harness uses:
- A dedicated recogniser on ``POST /v1beta/interactions`` (gemini-3.5-transcribe), which returns
  word timings and speaker labels.
- A general chat model on ``POST /v1beta/models/{model}:generateContent`` (gemini-3.8-flash), which
  returns text.

These tests cover URL targets, payload construction, verbatim mode configuration, word and speaker
parsing, error handling, retries, and dry runs.
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
from app.llm.google import (
    GoogleClient,
    parse_generate_content,
    parse_interaction,
)

TRANSCRIBE_ROUTE = "asr_gemini_transcribe"
FLASH_ROUTE = "asr_gemini_flash"


def _generate_content_body(*texts: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": t} for t in texts]}}]}


def _interaction_body(text: str, words: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    annotations = []
    for w in words or []:
        annotations.append(
            {
                "type": "word_info",
                "text": w["text"],
                "start_offset": w.get("start_offset") or w.get("startOffset"),
                "end_offset": w.get("end_offset") or w.get("endOffset"),
                "speaker": w.get("speaker"),
            }
        )
    return {
        "status": "completed",
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": annotations,
                    }
                ],
            }
        ],
    }


# --- response parsing, no database ----------------------------------------------------------


def test_parse_generate_content_joins_the_parts_of_a_candidate() -> None:
    assert parse_generate_content(_generate_content_body("आजको ", "meeting मा")) == (
        "आजको meeting मा"
    )


def test_parse_generate_content_skips_a_candidate_carrying_no_text() -> None:
    """A blocked or tool-only candidate yields nothing; the next one is tried."""
    body = {
        "candidates": [
            {"finishReason": "SAFETY", "content": {"parts": []}},
            {"content": {"parts": [{"text": "real transcript"}]}},
        ]
    }
    assert parse_generate_content(body) == "real transcript"


def test_parse_generate_content_handles_empty_or_malformed() -> None:
    assert parse_generate_content({}) == ""
    assert parse_generate_content({"candidates": []}) == ""
    assert parse_generate_content({"candidates": [{}]}) == ""
    assert parse_generate_content({"candidates": [{"content": {"parts": [{"n": 1}]}}]}) == ""


def test_parse_interaction_reads_the_transcript_and_its_word_spans() -> None:
    body = _interaction_body(
        "आजको meeting",
        [
            {"text": "आजको", "start_offset": "0.20s", "end_offset": "0.61s", "speaker": "spk_1"},
            {"text": "meeting", "start_offset": "0.61s", "end_offset": "1.30s", "speaker": "spk_1"},
        ],
    )
    text, words = parse_interaction(body)
    assert text == "आजको meeting"
    assert words == [
        {"word": "आजको", "start": 0.2, "end": 0.61, "speaker": "spk_1"},
        {"word": "meeting", "start": 0.61, "end": 1.3, "speaker": "spk_1"},
    ]


def test_parse_interaction_keeps_the_speaker_label_of_every_word() -> None:
    """Diarization is why this route exists; a clip with a turn in it must say so."""
    body = _interaction_body(
        "हो yes",
        [
            {"text": "हो", "start_offset": "0s", "end_offset": "0.4s", "speaker": "spk_1"},
            {"text": "yes", "start_offset": "0.5s", "end_offset": "0.9s", "speaker": "spk_2"},
        ],
    )
    _, words = parse_interaction(body)
    assert [w["speaker"] for w in words or []] == ["spk_1", "spk_2"]


def test_parse_interaction_returns_none_rather_than_an_empty_word_list() -> None:
    """The scorer reads None as "no signal"; an empty list would read as "no words said"."""
    text, words = parse_interaction(_interaction_body("कुनै timestamp छैन"))
    assert text == "कुनै timestamp छैन"
    assert words is None


def test_parse_interaction_survives_an_unparseable_or_missing_offset() -> None:
    body = _interaction_body(
        "क",
        [
            {"text": "क", "start_offset": "not-a-duration", "end_offset": None},
        ],
    )
    _, words = parse_interaction(body)
    assert words == [{"word": "क", "start": None, "end": None, "speaker": None}]


def test_parse_interaction_handles_output_text_fallback() -> None:
    body = {"output_text": "fallback text", "steps": []}
    text, words = parse_interaction(body)
    assert text == "fallback text"
    assert words is None


def test_parse_interaction_handles_empty_or_malformed() -> None:
    assert parse_interaction({}) == ("", None)
    assert parse_interaction({"steps": []}) == ("", None)
    assert parse_interaction({"steps": [{"model_output": {}}]}) == ("", None)


# --- requests, with a database session ------------------------------------------------------


def routes(**kwargs: Any) -> LlmRoutes:
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 3,
        "retry_backoff_seconds": 0.0,
        "google_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "routes": {
            TRANSCRIBE_ROUTE: LlmRoute(
                provider="google",
                api="transcription",
                model="gemini-3.5-transcribe",
                system_id="gemini-3.5-transcribe",
                language_codes=["ne-NP", "en-US"],
            ),
            FLASH_ROUTE: LlmRoute(
                provider="google",
                api="audio_chat",
                model="gemini-3.8-flash",
                system_id="gemini-3.8-flash",
                language="ne",
                forced_align=True,
            ),
        },
    }
    return LlmRoutes(**{**base, **kwargs})


def make_client(
    session: Session,
    handler,
    *,
    config: LlmRoutes | None = None,
    api_key: str | None = "test-api-key",
) -> GoogleClient:
    return GoogleClient(
        session,
        config=config or routes(),
        api_key=api_key,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.flac"
    path.write_bytes(b"mock audio bytes")
    return path


@pytest.mark.db
def test_a_disabled_google_client_refuses_to_transcribe(db_session: Session, clip: Path) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200), config=routes(enabled=False))
    with pytest.raises(LlmDisabledError):
        client.transcribe(clip, route=TRANSCRIBE_ROUTE)


@pytest.mark.db
def test_an_unknown_google_route_is_refused(db_session: Session, clip: Path) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200))
    with pytest.raises(LlmRouteNotConfigured, match="nonexistent"):
        client.transcribe(clip, route="nonexistent")


@pytest.mark.db
def test_a_missing_api_key_is_refused_before_any_request(
    db_session: Session, clip: Path, monkeypatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = make_client(db_session, handler, api_key="")
    with pytest.raises(LlmRequestFailed, match="GEMINI_API_KEY"):
        client.transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert not called


@pytest.mark.db
def test_a_google_dry_run_never_calls_the_network(db_session: Session, clip: Path) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = make_client(db_session, handler, config=routes(dry_run=True))
    result = client.transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert not called
    assert result.dry_run is True
    assert result.text


@pytest.mark.db
def test_a_transcription_route_targets_interactions_endpoint(
    db_session: Session, clip: Path
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert str(seen[0].url) == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert seen[0].headers["x-goog-api-key"] == "test-api-key"
    assert seen[0].method == "POST"


@pytest.mark.db
def test_word_timestamps_and_diarization_are_in_verbatim_mode(
    db_session: Session, clip: Path
) -> None:
    """Timestamps and diarization must be placed inside mode: {type: verbatim, ...}."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)
    sent = json.loads(seen[0].content)
    mode = sent["generation_config"]["transcription_config"]["mode"]
    assert mode["type"] == "verbatim"
    assert mode["timestamp_granularities"] == ["word"]
    assert mode["diarization_mode"] == "speaker"


@pytest.mark.db
def test_both_language_codes_reach_a_code_switched_route(db_session: Session, clip: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)
    sent = json.loads(seen[0].content)
    config = sent["generation_config"]["transcription_config"]
    assert config["language_codes"] == ["ne-NP", "en-US"]


@pytest.mark.db
def test_a_route_without_language_codes_falls_back_to_its_single_hint(
    db_session: Session, clip: Path
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    table = routes()
    single = table.routes[TRANSCRIBE_ROUTE].model_copy(
        update={"language_codes": [], "language": "ne"}
    )
    make_client(
        db_session, handler, config=routes(routes={**table.routes, TRANSCRIBE_ROUTE: single})
    ).transcribe(clip, route=TRANSCRIBE_ROUTE)
    sent = json.loads(seen[0].content)
    assert sent["generation_config"]["transcription_config"]["language_codes"] == ["ne"]


@pytest.mark.db
def test_a_dedicated_transcription_route_omits_system_instruction(
    db_session: Session, clip: Path
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    make_client(db_session, handler).transcribe(
        clip, route=TRANSCRIBE_ROUTE, prompt="DO NOT TRANSLITERATE."
    )
    sent = json.loads(seen[0].content)
    assert "system_instruction" not in sent


@pytest.mark.db
def test_the_clip_is_inlined_as_base64_flac_on_interactions(
    db_session: Session, clip: Path
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)
    sent = json.loads(seen[0].content)
    audio = sent["input"][0]
    assert audio["type"] == "audio"
    assert audio["mime_type"] == "audio/flac"
    assert base64.b64decode(audio["data"]) == b"mock audio bytes"


@pytest.mark.db
def test_an_audio_chat_route_targets_generate_content(db_session: Session, clip: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_generate_content_body("chat transcript"))

    result = make_client(db_session, handler).transcribe(clip, route=FLASH_ROUTE)
    assert str(seen[0].url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    )
    assert seen[0].headers["x-goog-api-key"] == "test-api-key"
    assert result.text == "chat transcript"
    assert result.words is None


@pytest.mark.db
def test_an_audio_chat_route_inlines_audio_part(db_session: Session, clip: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_generate_content_body("ok"))

    make_client(db_session, handler).transcribe(clip, route=FLASH_ROUTE, prompt="Policy.")
    sent = json.loads(seen[0].content)
    parts = sent["contents"][0]["parts"]
    assert "Policy." in parts[0]["text"]
    assert parts[1]["inline_data"]["mime_type"] == "audio/flac"
    assert base64.b64decode(parts[1]["inline_data"]["data"]) == b"mock audio bytes"


@pytest.mark.db
def test_a_429_is_retried_and_succeeds(db_session: Session, clip: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=_interaction_body("recovered"))

    result = make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert calls == 2
    assert result.text == "recovered"


@pytest.mark.db
def test_exhausted_retries_fail_with_logged_row(db_session: Session, clip: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    with pytest.raises(LlmRequestFailed, match="Gemini transcription failed"):
        make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)
