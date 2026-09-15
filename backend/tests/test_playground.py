"""The Models page playground (D85): a recording transcribed by a fine-tuned model on the CPU.

The sidecar that runs the model is never started here: every request goes to a mocked HTTP layer.
What is under test is the harness's half -- which weights a folder offers, how a recording is
prepared, and that every attempt is a routed ``llm_requests`` row (invariant 6).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes, load_llm_routes
from app.llm.base import LlmDisabledError, LlmRequestFailed
from app.llm.local_asr import PLAYGROUND_ROUTE, LocalAsrClient
from app.models import LlmRequest
from app.services import playground
from app.services.playground import PlaygroundError, cpu_weights, prepare_audio

SIDECAR_ANSWER = {"text": "आजको meeting मा data हेर्यौं", "retried": False, "compute_s": 0.8}


def routes(**kwargs) -> LlmRoutes:
    base = {
        "enabled": True,
        "dry_run": False,
        "max_retries": 2,
        "retry_backoff_seconds": 0.0,
        "local_base_url": "http://sidecar:8100",
        "routes": {
            PLAYGROUND_ROUTE: LlmRoute(provider="local", api="transcription", model="fine-tuned")
        },
    }
    return LlmRoutes(**{**base, **kwargs})


def wav(seconds: float, rate: int = 44_100) -> bytes:
    """A stereo tone at a browser's sample rate: a recording as it arrives, not as it is heard."""
    t = np.arange(int(seconds * rate)) / rate
    tone = 0.2 * np.sin(2 * np.pi * 220 * t)
    buf = io.BytesIO()
    sf.write(buf, np.stack([tone, tone], axis=1), rate, format="WAV")
    return buf.getvalue()


def write_weights(folder: Path, name: str, weights_file: str) -> None:
    (folder / name).mkdir(parents=True, exist_ok=True)
    (folder / name / "config.json").write_text("{}")
    (folder / name / weights_file).write_bytes(b"weights")


# --- which weights a model folder offers ---------------------------------------------------


def test_a_folder_without_weights_offers_none(tmp_path: Path) -> None:
    assert cpu_weights(tmp_path / "flex-ft", {}) is None


def test_the_card_names_the_export_the_notebook_accepted(tmp_path: Path) -> None:
    write_weights(tmp_path, "cpu", "model.int8.safetensors")
    write_weights(tmp_path, "best", "model.safetensors")
    assert cpu_weights(tmp_path, {"cpu": {"export": "best/"}}) == "best"
    assert cpu_weights(tmp_path, {"cpu": {"export": "cpu/"}}) == "cpu"


def test_without_a_card_block_int8_is_tried_first(tmp_path: Path) -> None:
    write_weights(tmp_path, "best", "model.safetensors")
    assert cpu_weights(tmp_path, {}) == "best"
    write_weights(tmp_path, "cpu", "model.int8.safetensors")
    assert cpu_weights(tmp_path, {}) == "cpu"


def test_the_card_named_export_falls_back_when_it_was_not_copied(tmp_path: Path) -> None:
    write_weights(tmp_path, "best", "model.safetensors")
    assert cpu_weights(tmp_path, {"cpu": {"export": "cpu/"}}) == "best"


def test_a_card_cannot_point_outside_the_model_folder(tmp_path: Path) -> None:
    write_weights(tmp_path, "elsewhere", "model.safetensors")
    folder = tmp_path / "flex-ft"
    folder.mkdir()
    assert cpu_weights(folder, {"cpu": {"export": "../elsewhere"}}) is None


def test_a_folder_with_only_a_config_is_not_weights(tmp_path: Path) -> None:
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "config.json").write_text("{}")
    assert cpu_weights(tmp_path, {}) is None


# --- preparing a recording -----------------------------------------------------------------


def test_a_recording_becomes_16_khz_mono_flac() -> None:
    flac, seconds = prepare_audio(wav(1.5), "take.wav")
    info = sf.info(io.BytesIO(flac))
    assert (info.samplerate, info.channels, info.format) == (16_000, 1, "FLAC")
    assert seconds == pytest.approx(1.5, abs=0.05)


def test_a_recording_longer_than_the_cap_is_refused() -> None:
    with pytest.raises(PlaygroundError, match="long-form"):
        prepare_audio(wav(playground.MAX_SECONDS + 1), "long.wav")


@pytest.mark.parametrize("data", [b"", b"this is not audio at all"])
def test_nothing_or_noise_is_refused(data: bytes) -> None:
    with pytest.raises(PlaygroundError):
        prepare_audio(data, "take.webm")


# --- the client: every attempt is a routed, logged request ---------------------------------


def _rows(session: Session) -> list[LlmRequest]:
    return list(session.scalars(sa.select(LlmRequest).order_by(LlmRequest.id)))


@pytest.mark.db
def test_a_transcription_names_the_model_and_its_weights(db_session: Session) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=SIDECAR_ANSWER)

    client = LocalAsrClient(
        db_session, config=routes(), client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    result = client.transcribe(b"flac", model="flex-ft", weights="cpu")

    assert result.text == SIDECAR_ANSWER["text"]
    assert str(seen[0].url) == "http://sidecar:8100/transcribe?model=flex-ft&weights=cpu"
    assert seen[0].content == b"flac"
    [row] = _rows(db_session)
    assert (row.route, row.model, row.status) == (PLAYGROUND_ROUTE, "flex-ft", "succeeded")
    assert row.output_json == SIDECAR_ANSWER
    assert row.estimated_cost_usd == 0


@pytest.mark.db
def test_a_dry_run_never_reaches_the_sidecar(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a dry run made a request")

    client = LocalAsrClient(
        db_session,
        config=routes(dry_run=True),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = client.transcribe(b"flac", model="flex-ft", weights="cpu")
    assert result.dry_run and result.text
    assert [r.status for r in _rows(db_session)] == ["dry_run"]


@pytest.mark.db
def test_a_sidecar_that_is_down_is_a_logged_failure(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = LocalAsrClient(
        db_session, config=routes(), client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(LlmRequestFailed, match="connection refused"):
        client.transcribe(b"flac", model="flex-ft", weights="cpu")
    [row] = _rows(db_session)
    assert row.status == "failed" and "connection refused" in row.error_message


@pytest.mark.db
def test_a_model_the_sidecar_cannot_load_is_not_retried(db_session: Session) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(422, json={"error": "no weights at /models/flex-ft/cpu"})

    client = LocalAsrClient(
        db_session, config=routes(), client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(LlmRequestFailed, match="422"):
        client.transcribe(b"flac", model="flex-ft", weights="cpu")
    assert len(calls) == 1


@pytest.mark.db
def test_inference_switched_off_transcribes_nothing(db_session: Session) -> None:
    client = LocalAsrClient(db_session, config=routes(enabled=False))
    with pytest.raises(LlmDisabledError):
        client.transcribe(b"flac", model="flex-ft", weights="cpu")


def test_the_committed_playground_route_is_local_and_never_an_ingest_system() -> None:
    config = load_llm_routes()
    route = config.routes[PLAYGROUND_ROUTE]
    assert (route.provider, route.api) == ("local", "transcription")
    assert PLAYGROUND_ROUTE not in config.asr_route_names()


def test_a_local_route_named_like_an_ingest_system_is_refused() -> None:
    with pytest.raises(ValueError, match="ingest system"):
        LlmRoutes(routes={"asr_flex": LlmRoute(provider="local", api="transcription", model="x")})


# --- the endpoint ----------------------------------------------------------------------------


@pytest.fixture
def imported_model(db_session: Session, settings) -> Path:
    """A model with a card and no runs: the playground needs no transcripts."""
    from app.services.model_import import import_model_dir

    folder = settings.models.root / "flex-ft"
    folder.mkdir(parents=True)
    card = {"name": "Flex FT", "created_at": "2026-09-15T12:00:00+00:00", "cpu": {"export": "cpu/"}}
    (folder / "model_card.json").write_text(json.dumps(card))
    import_model_dir(db_session, folder, settings=settings, actor="test")
    return folder


@pytest.fixture
def sidecar(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Point the endpoint's transcription at a mocked sidecar; returns the requests it saw."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=SIDECAR_ANSWER)

    real = playground.transcribe

    def mocked(*args, **kwargs):
        mock = httpx.Client(transport=httpx.MockTransport(handler))
        kwargs |= {"config": routes(), "client": mock}
        return real(*args, **kwargs)

    monkeypatch.setattr("app.api.asr_models.playground_transcribe", mocked)
    return seen


@pytest.mark.db
def test_a_model_says_whether_it_can_be_tried(client, imported_model: Path) -> None:
    assert client.get("/models/flex-ft").json()["playground"] is None
    write_weights(imported_model, "cpu", "model.int8.safetensors")
    assert client.get("/models").json()[0]["playground"] == "cpu"


@pytest.mark.db
def test_a_recording_is_transcribed_by_the_model(
    client, imported_model: Path, sidecar, db_session: Session
) -> None:
    write_weights(imported_model, "cpu", "model.int8.safetensors")
    response = client.post(
        "/models/flex-ft/transcribe", files={"audio": ("take.wav", wav(2.0), "audio/wav")}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == SIDECAR_ANSWER["text"]
    assert body["weights"] == "cpu"
    assert body["audio_s"] == pytest.approx(2.0, abs=0.05)
    sent = sf.info(io.BytesIO(sidecar[0].content))
    assert (sent.samplerate, sent.channels) == (16_000, 1)
    assert [r.model for r in _rows(db_session)] == ["flex-ft"]


@pytest.mark.db
def test_a_model_without_cpu_weights_cannot_be_tried(client, imported_model: Path) -> None:
    response = client.post(
        "/models/flex-ft/transcribe", files={"audio": ("take.wav", wav(1.0), "audio/wav")}
    )
    assert response.status_code == 409
    assert "cpu/" in response.json()["detail"]


@pytest.mark.db
def test_an_unreadable_recording_is_a_422(client, imported_model: Path, sidecar) -> None:
    write_weights(imported_model, "cpu", "model.int8.safetensors")
    response = client.post(
        "/models/flex-ft/transcribe", files={"audio": ("take.webm", b"garbage", "audio/webm")}
    )
    assert response.status_code == 422
    assert sidecar == []


@pytest.mark.db
def test_an_unknown_model_is_a_404(client) -> None:
    response = client.post(
        "/models/nope/transcribe", files={"audio": ("take.wav", wav(1.0), "audio/wav")}
    )
    assert response.status_code == 404
