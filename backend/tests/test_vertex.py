"""Tests for the Vertex AI client.

Vertex AI serves both Gemini shapes the harness uses (D35): a dedicated recogniser on
``interactions:create``, which returns word timings and speaker labels, and a general chat model
on ``generateContent``, which returns text. These tests cover the URL each shape targets, the
request each builds, the parsing of each response, and the dry-run and credential paths.
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
from app.llm.vertex import (
    VertexClient,
    parse_generate_content,
    parse_interaction,
)

TRANSCRIBE_ROUTE = "asr_gemini_transcribe"
FLASH_ROUTE = "asr_gemini_flash"


def _generate_content_body(*texts: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": t} for t in texts]}}]}


def _interaction_body(text: str, words: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    annotations = [{"wordInfo": w} for w in words or []]
    return {
        "status": "COMPLETED",
        "steps": [
            {"modelOutput": {"content": [{"text": {"text": text, "annotations": annotations}}]}}
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
            {"text": "आजको", "startOffset": "0.20s", "endOffset": "0.61s", "speaker": "spk_1"},
            {"text": "meeting", "startOffset": "0.61s", "endOffset": "1.30s", "speaker": "spk_1"},
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
            {"text": "हो", "startOffset": "0s", "endOffset": "0.4s", "speaker": "spk_1"},
            {"text": "yes", "startOffset": "0.5s", "endOffset": "0.9s", "speaker": "spk_2"},
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
            {"text": "क", "startOffset": "not-a-duration", "endOffset": None},
        ],
    )
    _, words = parse_interaction(body)
    assert words == [{"word": "क", "start": None, "end": None, "speaker": None}]


def test_parse_interaction_accepts_the_unnested_step_form() -> None:
    """The v1beta1 surface has moved once already; both step shapes are read."""
    body = {"steps": [{"content": [{"text": {"text": "flat"}}]}]}
    assert parse_interaction(body)[0] == "flat"


def test_parse_interaction_handles_empty_or_malformed() -> None:
    assert parse_interaction({}) == ("", None)
    assert parse_interaction({"steps": []}) == ("", None)
    assert parse_interaction({"steps": [{"modelOutput": {}}]}) == ("", None)


# --- requests, with a database session ------------------------------------------------------


def routes(**kwargs: Any) -> LlmRoutes:
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 3,
        "retry_backoff_seconds": 0.0,
        "vertex_project": "test-project",
        "vertex_location": "global",
        "routes": {
            TRANSCRIBE_ROUTE: LlmRoute(
                provider="vertex",
                api="transcription",
                model="gemini-3.5-transcribe",
                system_id="gemini-3.5-transcribe",
                language_codes=["ne-NP", "en-US"],
            ),
            FLASH_ROUTE: LlmRoute(
                provider="vertex",
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
    access_token: str | None = "test-token",
) -> VertexClient:
    return VertexClient(
        session,
        config=config or routes(),
        access_token=access_token,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.flac"
    path.write_bytes(b"mock audio bytes")
    return path


@pytest.mark.db
def test_a_disabled_vertex_client_refuses_to_transcribe(db_session: Session, clip: Path) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200), config=routes(enabled=False))
    with pytest.raises(LlmDisabledError):
        client.transcribe(clip, route=TRANSCRIBE_ROUTE)


@pytest.mark.db
def test_an_unknown_vertex_route_is_refused(db_session: Session, clip: Path) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200))
    with pytest.raises(LlmRouteNotConfigured, match="nonexistent"):
        client.transcribe(clip, route="nonexistent")


@pytest.mark.db
def test_a_missing_project_is_refused_before_any_request(
    db_session: Session, clip: Path, monkeypatch
) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    client = make_client(db_session, handler, config=routes(vertex_project=""))
    with pytest.raises(LlmRequestFailed, match="GOOGLE_CLOUD_PROJECT"):
        client.transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert not called


@pytest.mark.db
def test_a_vertex_dry_run_never_calls_the_network(db_session: Session, clip: Path) -> None:
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
def test_a_transcription_route_targets_interactions_create(db_session: Session, clip: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert str(seen[0].url) == (
        "https://aiplatform.googleapis.com/v1beta1/projects/test-project"
        "/locations/global/interactions:create"
    )
    assert seen[0].headers["authorization"] == "Bearer test-token"


@pytest.mark.db
def test_an_audio_chat_route_targets_generate_content(db_session: Session, clip: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_generate_content_body("ok"))

    make_client(db_session, handler).transcribe(clip, route=FLASH_ROUTE)
    assert str(seen[0].url).endswith(
        "/locations/global/publishers/google/models/gemini-3.8-flash:generateContent"
    )


@pytest.mark.db
def test_a_region_gets_its_own_host_and_global_does_not(db_session: Session, clip: Path) -> None:
    """`global` is a location, not a region: it has an unprefixed host of its own."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    make_client(db_session, handler, config=routes(vertex_location="us-central1")).transcribe(
        clip, route=TRANSCRIBE_ROUTE
    )
    assert str(seen[0].url).startswith("https://us-central1-aiplatform.googleapis.com/")
    assert "/locations/us-central1/" in str(seen[0].url)


@pytest.mark.db
def test_word_timestamps_and_diarization_are_always_requested(
    db_session: Session, clip: Path
) -> None:
    """They are the reason a route names this model rather than a chat model."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)
    sent = json.loads(seen[0].content)["interaction"]
    config = sent["modelInteraction"]["generationConfig"]["transcriptionConfig"]
    assert config["timestampGranularities"] == ["word"]
    assert config["diarizationMode"] == "speaker"


@pytest.mark.db
def test_both_language_codes_reach_a_code_switched_route(db_session: Session, clip: Path) -> None:
    """A code-switched clip is not one language with loanwords in it."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)
    sent = json.loads(seen[0].content)["interaction"]
    config = sent["modelInteraction"]["generationConfig"]["transcriptionConfig"]
    assert config["languageCodes"] == ["ne-NP", "en-US"]


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
    sent = json.loads(seen[0].content)["interaction"]
    assert sent["modelInteraction"]["generationConfig"]["transcriptionConfig"]["languageCodes"] == [
        "ne"
    ]


@pytest.mark.db
def test_the_transcript_policy_rides_on_the_system_instruction(
    db_session: Session, clip: Path
) -> None:
    """Verbatim mode has no field on this API, so the model is told in prose."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    make_client(db_session, handler).transcribe(
        clip, route=TRANSCRIBE_ROUTE, prompt="DO NOT TRANSLITERATE."
    )
    sent = json.loads(seen[0].content)["interaction"]
    assert sent["systemInstruction"] == "DO NOT TRANSLITERATE."


@pytest.mark.db
def test_custom_vocabulary_is_forwarded_when_given(db_session: Session, clip: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_interaction_body("ok"))

    make_client(db_session, handler).transcribe(
        clip, route=TRANSCRIBE_ROUTE, custom_vocabulary=["podcast", "data"]
    )
    sent = json.loads(seen[0].content)["interaction"]
    config = sent["modelInteraction"]["generationConfig"]["transcriptionConfig"]
    assert config["customVocabulary"] == ["podcast", "data"]


@pytest.mark.db
def test_the_clip_is_inlined_as_base64_flac_on_both_shapes(db_session: Session, clip: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "interactions" in str(request.url):
            return httpx.Response(200, json=_interaction_body("ok"))
        return httpx.Response(200, json=_generate_content_body("ok"))

    client = make_client(db_session, handler)
    client.transcribe(clip, route=TRANSCRIBE_ROUTE)
    client.transcribe(clip, route=FLASH_ROUTE)

    audio = json.loads(seen[0].content)["interaction"]["content"]["audio"]
    assert audio["mime_type"] == "audio/flac"
    assert base64.b64decode(audio["data"]) == b"mock audio bytes"

    inline = json.loads(seen[1].content)["contents"][0]["parts"][-1]["inline_data"]
    assert inline["mime_type"] == "audio/flac"
    assert base64.b64decode(inline["data"]) == b"mock audio bytes"


@pytest.mark.db
def test_a_transcription_route_returns_word_spans(db_session: Session, clip: Path) -> None:
    words = [{"text": "नमस्ते", "startOffset": "0s", "endOffset": "0.8s", "speaker": "spk_1"}]
    client = make_client(
        db_session, lambda r: httpx.Response(200, json=_interaction_body("नमस्ते", words))
    )
    result = client.transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert result.text == "नमस्ते"
    assert result.words == [{"word": "नमस्ते", "start": 0.0, "end": 0.8, "speaker": "spk_1"}]


@pytest.mark.db
def test_an_audio_chat_route_never_returns_word_spans(db_session: Session, clip: Path) -> None:
    """Spans for a chat model are the forced aligner's job (D32), not the model's."""
    client = make_client(
        db_session, lambda r: httpx.Response(200, json=_generate_content_body("गुगल भन्छ"))
    )
    assert client.transcribe(clip, route=FLASH_ROUTE).words is None


@pytest.mark.db
def test_a_retryable_status_is_retried_then_succeeds(db_session: Session, clip: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={})
        return httpx.Response(200, json=_interaction_body("गुगल भन्छ"))

    result = make_client(db_session, handler).transcribe(clip, route=TRANSCRIBE_ROUTE)
    assert attempts == 2
    assert result.text == "गुगल भन्छ"


@pytest.mark.db
def test_an_exhausted_retry_budget_is_logged_and_raised(db_session: Session, clip: Path) -> None:
    client = make_client(db_session, lambda r: httpx.Response(503, text="unavailable"))
    with pytest.raises(LlmRequestFailed, match="Vertex AI transcription failed"):
        client.transcribe(clip, route=TRANSCRIBE_ROUTE)
