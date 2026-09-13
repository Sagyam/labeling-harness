from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import DiarizationSettings, Settings
from app.services.diarization import (
    DiarizationServiceError,
    declared_speaker_count,
    diarize_audio,
)

ENDPOINT = "https://example.modal.run"


@pytest.fixture(autouse=True)
def _no_diarization_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scripts' tests load the real ``.env``; its token would replace the section built here."""
    for name in list(os.environ):
        if name.startswith("HARNESS_DIARIZATION__"):
            monkeypatch.delenv(name)


def _settings(**overrides) -> Settings:
    return Settings(diarization=DiarizationSettings(**{"enabled": True, **overrides}))


def _client(status: int = 200, payload: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock(status_code=status, text=text)
    resp.json.return_value = payload
    client = MagicMock()
    client.post.return_value = resp
    return client


def test_nothing_is_sent_when_diarization_is_disabled(tmp_path: Path) -> None:
    client = _client()
    settings = _settings(enabled=False, endpoint_url=ENDPOINT)
    assert diarize_audio(b"flac", settings=settings, client=client) is None
    client.post.assert_not_called()


def test_nothing_is_sent_without_an_endpoint() -> None:
    client = _client()
    assert diarize_audio(b"flac", settings=_settings(endpoint_url=""), client=client) is None
    client.post.assert_not_called()


def test_the_audio_speaker_count_and_proxy_token_are_sent(tmp_path: Path) -> None:
    audio = tmp_path / "episode.flac"
    audio.write_bytes(b"fake flac bytes")
    answer = {"turns": [[0.0, 5.0, "SPEAKER_00"], [5.0, 10.0, "SPEAKER_01"]]}
    client = _client(payload=answer)

    result = diarize_audio(
        audio,
        num_speakers=2,
        settings=_settings(endpoint_url=ENDPOINT, auth_token="wk-1.ws-2"),
        client=client,
    )

    assert result == answer
    (url,), kwargs = client.post.call_args
    assert url == ENDPOINT
    assert kwargs["data"] == {"num_speakers": "2"}
    assert kwargs["headers"] == {"Authorization": "Bearer wk-1.ws-2"}
    assert kwargs["files"]["file"][1] == b"fake flac bytes"


def test_an_unknown_speaker_count_is_left_to_the_diarizer() -> None:
    client = _client(payload={"turns": []})
    diarize_audio(
        b"flac", num_speakers=None, settings=_settings(endpoint_url=ENDPOINT), client=client
    )
    assert client.post.call_args.kwargs["data"] == {}
    assert client.post.call_args.kwargs["headers"] == {}


def test_a_service_error_is_raised_with_its_status() -> None:
    client = _client(status=401, text="missing credentials")
    with pytest.raises(DiarizationServiceError, match="answered 401"):
        diarize_audio(b"flac", settings=_settings(endpoint_url=ENDPOINT), client=client)


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"speakers": {"host": {"gender": "male"}, "guest": {}}}, 2),
        # A blank row on the form is in the count but not in the speaker block.
        ({"speaker_count": 3, "speakers": {"spk0": {"gender": "male"}}}, 3),
        ({"speaker_count": 0, "speakers": {"spk0": {}}}, 1),
        ({"speaker_count": True}, None),
        ({"speakers": [{"name": "a"}]}, 1),
        ({"speakers": {}}, None),
        ({"title": "no speakers"}, None),
        (None, None),
    ],
)
def test_the_declared_speaker_count_comes_from_the_metadata(metadata, expected) -> None:
    assert declared_speaker_count(metadata) == expected
