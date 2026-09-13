"""Overlapped-speech detection: the numpy port of pyannote's speaker count, and its model fetch."""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pytest

from app.services import overlap
from app.services.overlap import (
    FRAME_STEP,
    SAMPLE_RATE,
    STEP_SAMPLES,
    WINDOW_SAMPLES,
    OverlapDetector,
    aggregate_counts,
    chunk_starts,
    ensure_overlap_model,
    frames_to_spans,
    spans_within,
)

FRAMES = 589  # frames segmentation-3.0 emits for one 10 s window


# --- chunking ---------------------------------------------------------------------------------


def test_audio_shorter_than_a_window_is_one_padded_chunk() -> None:
    assert chunk_starts(SAMPLE_RATE * 3) == [0]


def test_exactly_one_window_is_one_chunk() -> None:
    assert chunk_starts(WINDOW_SAMPLES) == [0]


def test_windows_advance_by_the_step() -> None:
    assert chunk_starts(WINDOW_SAMPLES + 2 * STEP_SAMPLES) == [0, STEP_SAMPLES, 2 * STEP_SAMPLES]


def test_a_remainder_shorter_than_the_step_gets_its_own_padded_chunk() -> None:
    # pyannote adds a last chunk whenever the windows do not end exactly at the end of the audio.
    assert chunk_starts(WINDOW_SAMPLES + STEP_SAMPLES // 2) == [0, STEP_SAMPLES]


# --- aggregation ------------------------------------------------------------------------------


def test_overlapping_chunks_are_averaged_then_rounded() -> None:
    first = np.full(FRAMES, 2.0)
    second = np.full(FRAMES, 1.0)
    count = aggregate_counts(np.stack([first, second]), [0.0, 1.0])
    offset = round(1.0 / FRAME_STEP)
    assert count[0] == 2  # only the first chunk covers the start
    assert count[offset + 10] == 2  # mean 1.5 rounds to 2, as np.rint does in pyannote
    assert count[FRAMES + 10] == 1  # only the second chunk covers the end


def test_a_mean_of_one_half_rounds_down_like_numpy_rint() -> None:
    count = aggregate_counts(np.stack([np.ones(FRAMES), np.zeros(FRAMES)]), [0.0, 0.0])
    assert count.max() == 0


def test_aggregation_covers_the_last_chunk() -> None:
    count = aggregate_counts(np.stack([np.zeros(FRAMES), np.zeros(FRAMES)]), [0.0, 5.0])
    assert len(count) == round(5.0 / FRAME_STEP) + FRAMES


# --- spans ------------------------------------------------------------------------------------


def test_frames_become_merged_spans() -> None:
    mask = np.array([0, 0, 1, 1, 1, 0, 1], dtype=bool)
    spans = frames_to_spans(mask, frame_step=0.1)
    assert spans == [(0.2, 0.5), (0.6, 0.7)]


def test_no_overlap_is_an_empty_list() -> None:
    assert frames_to_spans(np.zeros(10, dtype=bool), frame_step=0.1) == []


def test_spans_are_cut_to_a_clip_and_made_clip_relative() -> None:
    spans = [(1.0, 2.0), (4.5, 6.0), (9.0, 9.5)]
    assert spans_within(spans, 5.0, 9.2) == [(0.0, 1.0), (4.0, 4.2)]


def test_a_clip_with_no_overlap_gets_an_empty_list() -> None:
    assert spans_within([(1.0, 2.0)], 5.0, 9.0) == []


# --- the detector, with a stand-in for the ONNX session ----------------------------------------


class _FakeSession:
    """Returns the same powerset class for every frame of every chunk."""

    def __init__(self, cls: int) -> None:
        self.cls = cls
        self.batches: list[tuple[int, ...]] = []

    def run(self, _outputs, feeds):
        batch = feeds["input_values"]
        self.batches.append(batch.shape)
        logits = np.full((batch.shape[0], FRAMES, 7), -10.0, dtype=np.float32)
        logits[..., self.cls] = 0.0
        return [logits]


def _detector(cls: int) -> OverlapDetector:
    detector = OverlapDetector(model_path=Path("/nonexistent/model.onnx"))
    detector._session = _FakeSession(cls)
    return detector


def test_two_speakers_everywhere_is_one_span_over_the_audio() -> None:
    audio = np.zeros(SAMPLE_RATE * 25, dtype=np.float32)
    spans = _detector(cls=4).detect(audio, SAMPLE_RATE)  # class 4 = speakers 1 and 2
    assert len(spans) == 1
    start, end = spans[0]
    assert start == 0.0
    # 589 frames at 16.875 ms cover 9.94 s of each 10 s window, so the grid stops ~60 ms short.
    assert end == pytest.approx(25.0, abs=0.1)


def test_one_speaker_everywhere_is_no_overlap() -> None:
    audio = np.zeros(SAMPLE_RATE * 25, dtype=np.float32)
    assert _detector(cls=1).detect(audio, SAMPLE_RATE) == []


def test_chunks_are_fed_a_batch_at_a_time() -> None:
    detector = _detector(cls=0)
    detector.detect(np.zeros(SAMPLE_RATE * 60, dtype=np.float32), SAMPLE_RATE, batch_size=8)
    shapes = detector._session.batches
    assert all(shape[0] <= 8 and shape[1:] == (1, WINDOW_SAMPLES) for shape in shapes)
    assert sum(shape[0] for shape in shapes) == len(chunk_starts(SAMPLE_RATE * 60))


def test_audio_at_another_rate_is_refused() -> None:
    with pytest.raises(ValueError, match="16"):
        _detector(cls=0).detect(np.zeros(8000, dtype=np.float32), 8000)


def test_an_explicit_missing_model_disables_detection_without_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("an explicit model path must never trigger a download")

    monkeypatch.setattr(httpx, "stream", refuse)
    detector = OverlapDetector(model_path=Path("/nonexistent/model.onnx"))
    assert detector.available is False
    assert detector.detect(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE) is None


# --- the model fetch --------------------------------------------------------------------------


def test_the_fetch_can_be_switched_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(overlap.DISABLE_ENV, "1")
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: pytest.fail("download attempted"))
    assert ensure_overlap_model(tmp_path / "m.onnx") is False


def test_a_present_model_is_not_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"graph")
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: pytest.fail("download attempted"))
    assert ensure_overlap_model(model) is True


def test_a_failed_fetch_degrades_to_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreachable(*_args, **_kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.delenv(overlap.DISABLE_ENV, raising=False)
    monkeypatch.setattr(httpx, "stream", unreachable)
    assert ensure_overlap_model(tmp_path / "m.onnx") is False
    assert not (tmp_path / "m.onnx").exists()


def test_the_pin_is_a_commit_and_a_digest() -> None:
    assert len(overlap.MODEL_REVISION) == 40
    assert len(overlap.MODEL_SHA256) == 64
