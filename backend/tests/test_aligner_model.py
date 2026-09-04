"""Tests for fetching the forced-alignment model.

Nothing here touches the network: the ~317 MB download is faked, and the tests that matter are the
ones proving it cannot leave a half-written file behind. A truncated ONNX graph that loads would
produce quietly wrong word spans on every clip, which is worse than having no aligner at all.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.services import aligner_model
from app.services.aligner_model import DISABLE_ENV, download_disabled, ensure_aligner_model
from app.services.forced_align import ForcedAligner

PAYLOAD = b"pretend onnx graph"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class _Response:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, _size: int = 0):
        yield from self._chunks

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _serving(chunks: list[bytes], seen: list[str] | None = None):
    def stream(method: str, url: str, **_kwargs: object) -> _Response:
        if seen is not None:
            seen.append(url)
        return _Response(chunks)

    return stream


def _refusing():
    def stream(*_args: object, **_kwargs: object):  # pragma: no cover - must not run
        raise AssertionError("the network was touched")

    return stream


def _pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point both artefacts at the same tiny fake payload."""
    monkeypatch.setattr(aligner_model, "MODEL_SHA256", DIGEST)
    monkeypatch.setattr(aligner_model, "VOCAB_SHA256", DIGEST)


# --- the switch -----------------------------------------------------------------------------


def test_the_download_is_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    assert download_disabled() is False


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_the_download_can_be_switched_off(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_ENV, value)
    assert download_disabled() is True


def test_a_disabled_download_never_touches_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DISABLE_ENV, "1")
    monkeypatch.setattr(aligner_model.httpx, "stream", _refusing())
    assert ensure_aligner_model(tmp_path / "m.onnx", tmp_path / "v.json") is False


# --- fetching -------------------------------------------------------------------------------


def test_files_already_present_are_not_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, vocab = tmp_path / "m.onnx", tmp_path / "v.json"
    model.write_bytes(PAYLOAD)
    vocab.write_bytes(PAYLOAD)
    monkeypatch.setattr(aligner_model.httpx, "stream", _refusing())
    assert ensure_aligner_model(model, vocab) is True


def test_a_good_download_lands_both_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(aligner_model.httpx, "stream", _serving([PAYLOAD], seen))
    model, vocab = tmp_path / "nested" / "m.onnx", tmp_path / "nested" / "v.json"

    assert ensure_aligner_model(model, vocab) is True
    assert model.read_bytes() == PAYLOAD
    assert vocab.read_bytes() == PAYLOAD
    # The container writes into a bind mount as root; 0600 would leave the host unable to read
    # or remove its own file.
    assert model.stat().st_mode & 0o077 == 0o044
    assert all(aligner_model.MODEL_REVISION in url for url in seen), (
        "a model must be pinned to a commit, or it can change under a finished benchmark"
    )


def test_only_the_missing_file_is_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(aligner_model.httpx, "stream", _serving([PAYLOAD], seen))
    model, vocab = tmp_path / "m.onnx", tmp_path / "v.json"
    model.write_bytes(PAYLOAD)

    assert ensure_aligner_model(model, vocab) is True
    assert len(seen) == 1
    assert seen[0].endswith(aligner_model.VOCAB_REMOTE)


def test_a_chunked_transfer_is_reassembled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch)
    monkeypatch.setattr(aligner_model.httpx, "stream", _serving([PAYLOAD[:5], PAYLOAD[5:]]))
    model, vocab = tmp_path / "m.onnx", tmp_path / "v.json"
    assert ensure_aligner_model(model, vocab) is True
    assert model.read_bytes() == PAYLOAD


# --- the part that must never go wrong ------------------------------------------------------


def test_a_corrupt_download_is_refused_and_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated graph that loads would put wrong spans on every clip in the corpus."""
    _pin(monkeypatch)
    monkeypatch.setattr(aligner_model.httpx, "stream", _serving([b"truncated"]))
    model, vocab = tmp_path / "m.onnx", tmp_path / "v.json"

    assert ensure_aligner_model(model, vocab) is False
    assert not model.exists()
    assert list(tmp_path.glob("*.part")) == [], "a failed transfer must not survive as a fragment"


def test_a_transport_failure_is_a_warning_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ingestion survives an absent aligner; it must survive an unreachable one too (D32)."""

    def exploding(*_args: object, **_kwargs: object):
        raise aligner_model.httpx.ConnectError("no route to host")

    monkeypatch.setattr(aligner_model.httpx, "stream", exploding)
    assert ensure_aligner_model(tmp_path / "m.onnx", tmp_path / "v.json") is False
    assert list(tmp_path.glob("*.part")) == []


# --- who gets a download --------------------------------------------------------------------


def test_an_explicit_path_is_never_downloaded_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A named path is a file the caller manages; fetching 317 MB over it is not helpful."""
    monkeypatch.setattr(aligner_model.httpx, "stream", _refusing())
    aligner = ForcedAligner(
        model_path=tmp_path / "absent.onnx", vocab_path=tmp_path / "absent.json"
    )
    assert aligner.available is False


def test_the_default_location_is_fetched_into(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructed with no arguments, a missing model fetches itself instead of warning."""
    called: list[tuple[Path, Path]] = []

    def fake(model_path: Path, vocab_path: Path, **_kwargs: object) -> bool:
        called.append((model_path, vocab_path))
        return False

    monkeypatch.setattr("app.services.forced_align.DEFAULT_MODEL_PATH", tmp_path / "m.onnx")
    monkeypatch.setattr("app.services.forced_align.DEFAULT_VOCAB_PATH", tmp_path / "v.json")
    monkeypatch.setattr("app.services.forced_align.ensure_aligner_model", fake)

    aligner = ForcedAligner()
    assert called == [(tmp_path / "m.onnx", tmp_path / "v.json")]
    assert aligner.available is False, "a failed fetch still degrades to no word spans (D32)"
