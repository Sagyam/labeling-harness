"""Tests for the OpenRouter client.

All LLM inference from this codebase goes through OpenRouter, and no route is wired at MVP. These
tests run entirely against a mocked HTTP layer -- the suite never makes a paid call.
"""

from __future__ import annotations

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.openrouter import (
    LlmDisabledError,
    LlmRequestFailed,
    LlmRouteNotConfigured,
    OpenRouterClient,
)
from app.models import LlmRequest

pytestmark = pytest.mark.db


def routes(**kwargs) -> LlmRoutes:
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 3,
        "retry_backoff_seconds": 0.0,
        "routes": {"check": LlmRoute(model="anthropic/claude-sonnet-4", max_tokens=64)},
    }
    return LlmRoutes(**{**base, **kwargs})


def completion(text: str = "ok", prompt_tokens: int = 11, completion_tokens: int = 5) -> dict:
    return {
        "id": "gen-1",
        "model": "anthropic/claude-sonnet-4",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_cost": 0.00042,
        },
    }


def make_client(
    session: Session, handler, *, config: LlmRoutes | None = None, api_key: str = "test-key"
) -> OpenRouterClient:
    return OpenRouterClient(
        session,
        config=config or routes(),
        api_key=api_key,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


# --- the MVP posture: nothing is wired ---------------------------------------------------


def test_the_committed_configuration_has_asr_routes() -> None:
    from app.config import load_llm_routes

    config = load_llm_routes()
    assert config.enabled is True
    assert config.asr_route_names()


def test_a_disabled_client_refuses_to_call(db_session: Session) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200), config=routes(enabled=False))
    with pytest.raises(LlmDisabledError):
        client.complete("check", [{"role": "user", "content": "hi"}])


def test_an_unknown_route_is_refused(db_session: Session) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200))
    with pytest.raises(LlmRouteNotConfigured, match="nope"):
        client.complete("nope", [{"role": "user", "content": "hi"}])


# --- happy path --------------------------------------------------------------------------


def test_a_successful_call_returns_the_text(db_session: Session) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200, json=completion("नमस्ते")))
    result = client.complete("check", [{"role": "user", "content": "hi"}])
    assert result.text == "नमस्ते"
    assert result.model == "anthropic/claude-sonnet-4"
    assert result.prompt_tokens == 11


def test_the_request_targets_openrouter_with_the_route_model(db_session: Session) -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json=completion())

    make_client(db_session, handler).complete("check", [{"role": "user", "content": "hi"}])
    assert seen["url"].startswith("https://openrouter.ai/api/v1")
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "anthropic/claude-sonnet-4"
    assert seen["body"]["max_tokens"] == 64


def test_a_missing_api_key_is_refused(db_session: Session) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200), api_key="")
    with pytest.raises(LlmRequestFailed, match="OPENROUTER_API_KEY"):
        client.complete("check", [{"role": "user", "content": "hi"}])


# --- logging -----------------------------------------------------------------------------


def test_every_call_is_logged(db_session: Session) -> None:
    make_client(db_session, lambda r: httpx.Response(200, json=completion())).complete(
        "check", [{"role": "user", "content": "hi"}]
    )
    row = db_session.scalars(sa.select(LlmRequest)).one()
    assert row.route == "check"
    assert row.status == "succeeded"
    assert row.prompt_tokens == 11
    assert row.completion_tokens == 5
    assert float(row.estimated_cost_usd) == pytest.approx(0.00042)
    assert row.latency_ms is not None
    assert row.request_hash


def test_a_failure_is_logged_with_the_error(db_session: Session) -> None:
    client = make_client(
        db_session, lambda r: httpx.Response(400, text="bad request"), config=routes(max_retries=1)
    )
    with pytest.raises(LlmRequestFailed):
        client.complete("check", [{"role": "user", "content": "hi"}])
    row = db_session.scalars(sa.select(LlmRequest)).one()
    assert row.status == "failed"
    assert "400" in row.error_message


def test_the_logged_input_summary_is_truncated(db_session: Session) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200, json=completion()))
    client.complete("check", [{"role": "user", "content": "x" * 5000}])
    row = db_session.scalars(sa.select(LlmRequest)).one()
    assert len(row.input_summary) <= 1000


def test_identical_requests_hash_identically(db_session: Session) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200, json=completion()))
    messages = [{"role": "user", "content": "hi"}]
    client.complete("check", messages)
    client.complete("check", messages)
    hashes = {row.request_hash for row in db_session.scalars(sa.select(LlmRequest))}
    assert len(hashes) == 1


# --- dry run -----------------------------------------------------------------------------


def test_dry_run_makes_no_http_call(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("dry run must not reach the network")

    client = make_client(db_session, handler, config=routes(dry_run=True))
    result = client.complete("check", [{"role": "user", "content": "hi"}])
    assert result.dry_run is True
    assert result.text == ""


def test_dry_run_is_still_logged(db_session: Session) -> None:
    client = make_client(db_session, lambda r: httpx.Response(200), config=routes(dry_run=True))
    client.complete("check", [{"role": "user", "content": "hi"}])
    assert db_session.scalars(sa.select(LlmRequest)).one().status == "dry_run"


def test_per_call_dry_run_overrides_the_configuration(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("dry run must not reach the network")

    client = make_client(db_session, handler)
    assert client.complete("check", [{"role": "user", "content": "hi"}], dry_run=True).dry_run


# --- retries -----------------------------------------------------------------------------


def test_a_rate_limit_is_retried_then_succeeds(db_session: Session) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=completion())

    result = make_client(db_session, handler).complete("check", [{"role": "user", "content": "x"}])
    assert result.text == "ok"
    assert calls["n"] == 3
    assert db_session.scalars(sa.select(LlmRequest)).one().status == "succeeded"


def test_retries_are_bounded(db_session: Session) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    client = make_client(db_session, handler, config=routes(max_retries=2))
    with pytest.raises(LlmRequestFailed):
        client.complete("check", [{"role": "user", "content": "x"}])
    assert calls["n"] == 2


def test_a_client_error_is_not_retried(db_session: Session) -> None:
    """A 400 will fail identically every time; retrying it only spends money and time."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(LlmRequestFailed):
        make_client(db_session, handler).complete("check", [{"role": "user", "content": "x"}])
    assert calls["n"] == 1


def test_a_timeout_is_retried(db_session: Session) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("too slow", request=request)
        return httpx.Response(200, json=completion())

    assert make_client(db_session, handler).complete("check", [{"role": "user", "content": "x"}])
    assert calls["n"] == 2


# --- transcribe: the ASR path must never invent a transcript -----------------------------


def asr_routes(**kwargs) -> LlmRoutes:
    """Route table with a Cloud ASR route."""
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 3,
        "retry_backoff_seconds": 0.0,
        "routes": {"asr_whisper": LlmRoute(api="transcription", model="custom/asr-model")},
    }
    return LlmRoutes(**{**base, **kwargs})


def transcription(text: str = "आजको meeting मा data हेर्यौं", **extra) -> dict:
    return {"model": "custom/asr-model", "text": text, **extra}


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "seg.flac"
    path.write_bytes(b"not really audio, but the client only reads its size")
    return path


def test_a_missing_api_key_never_yields_a_fabricated_transcript(db_session: Session, clip) -> None:
    """The failure mode that matters: a deploy loses its key and the corpus fills with mock text."""
    client = make_client(db_session, lambda r: httpx.Response(200), config=asr_routes(), api_key="")
    with pytest.raises(LlmRequestFailed, match="OPENROUTER_API_KEY"):
        client.transcribe(clip, route="asr_whisper")

    logged = db_session.scalars(sa.select(LlmRequest)).all()
    assert [row.status for row in logged] == ["failed"]


def test_a_disabled_configuration_refuses_to_transcribe(db_session: Session, clip) -> None:
    client = make_client(
        db_session, lambda r: httpx.Response(200), config=asr_routes(enabled=False)
    )
    with pytest.raises(LlmDisabledError):
        client.transcribe(clip, route="asr_whisper")


def test_an_explicit_dry_run_is_marked_and_makes_no_http_call(db_session: Session, clip) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a dry run must not reach the network")

    result = make_client(db_session, handler, config=asr_routes(dry_run=True)).transcribe(
        clip, route="asr_whisper"
    )
    assert result.dry_run is True
    assert result.text
    # A mock carries no confidence signal, so it must not claim one.
    assert result.avg_logprob is None
    assert result.no_speech_prob is None
    assert db_session.scalars(sa.select(LlmRequest.status)).all() == ["dry_run"]


def test_leading_whitespace_is_stripped_from_a_transcript(db_session: Session, clip) -> None:
    """A recogniser may prefix its output with a space; verbatim it shifts every diff against it."""
    result = make_client(
        db_session,
        lambda r: httpx.Response(200, json=transcription(" आजको meeting ")),
        config=asr_routes(),
    ).transcribe(clip, route="asr_whisper")
    assert result.text == "आजको meeting"


def test_a_successful_transcription_returns_the_text_and_words(db_session: Session, clip) -> None:
    body = transcription(words=[{"word": "आजको", "start": 0.0, "end": 0.4}])
    result = make_client(
        db_session, lambda r: httpx.Response(200, json=body), config=asr_routes()
    ).transcribe(clip, route="asr_whisper")

    assert result.text == "आजको meeting मा data हेर्यौं"
    assert result.words == [{"word": "आजको", "start": 0.0, "end": 0.4}]
    assert result.dry_run is False


def test_absent_confidence_fields_stay_none_rather_than_being_invented(
    db_session: Session, clip
) -> None:
    """A default like -0.2 would silently drive the low_confidence term of the priority score."""
    result = make_client(
        db_session, lambda r: httpx.Response(200, json=transcription()), config=asr_routes()
    ).transcribe(clip, route="asr_whisper")

    assert result.avg_logprob is None
    assert result.no_speech_prob is None


def test_reported_confidence_fields_are_kept(db_session: Session, clip) -> None:
    body = transcription(avg_logprob=-0.8, no_speech_prob=0.42)
    result = make_client(
        db_session, lambda r: httpx.Response(200, json=body), config=asr_routes()
    ).transcribe(clip, route="asr_whisper")

    assert result.avg_logprob == -0.8
    assert result.no_speech_prob == 0.42


def test_the_transcription_request_carries_the_model_prompt_and_language(
    db_session: Session, clip
) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content
        return httpx.Response(200, json=transcription())

    make_client(db_session, handler, config=asr_routes()).transcribe(
        clip, route="asr_whisper", prompt="code-switched", language="ne"
    )
    assert seen["url"] == "https://openrouter.ai/api/v1/audio/transcriptions"
    body = bytes(seen["body"])
    assert b"custom/asr-model" in body
    assert b"code-switched" in body
    assert b"ne" in body


def test_every_transcription_is_logged(db_session: Session, clip) -> None:
    make_client(
        db_session, lambda r: httpx.Response(200, json=transcription()), config=asr_routes()
    ).transcribe(clip, route="asr_whisper")

    logged = db_session.scalars(sa.select(LlmRequest)).one()
    assert logged.route == "asr_whisper"
    assert logged.model == "custom/asr-model"
    assert logged.status == "succeeded"
    assert logged.latency_ms is not None


def test_a_rate_limited_transcription_is_retried_then_succeeds(db_session: Session, clip) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=transcription())

    result = make_client(db_session, handler, config=asr_routes()).transcribe(
        clip, route="asr_whisper"
    )
    assert calls["n"] == 2
    assert result.text


def test_a_client_error_is_not_retried_and_is_logged(db_session: Session, clip) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad audio")

    with pytest.raises(LlmRequestFailed):
        make_client(db_session, handler, config=asr_routes()).transcribe(clip, route="asr_whisper")

    assert calls["n"] == 1
    logged = db_session.scalars(sa.select(LlmRequest)).one()
    assert logged.status == "failed"
    assert "400" in logged.error_message


def test_transcription_retries_are_bounded(db_session: Session, clip) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    with pytest.raises(LlmRequestFailed):
        make_client(db_session, handler, config=asr_routes(max_retries=2)).transcribe(
            clip, route="asr_whisper"
        )
    assert calls["n"] == 2
