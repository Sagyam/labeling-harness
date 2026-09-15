"""Brouhaha over a whole recording: windowing and averaging onto one frame grid (D87)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.brouhaha import (
    FRAME_STEP,
    SAMPLE_RATE,
    STEP_SAMPLES,
    WINDOW_SAMPLES,
    Brouhaha,
    aggregate,
    window_starts,
)

FRAMES = 352  # frames Brouhaha emits for one 6 s window


def test_short_audio_is_one_padded_window() -> None:
    assert window_starts(SAMPLE_RATE * 2) == [0]


def test_windows_advance_by_the_step_and_reach_the_end() -> None:
    assert window_starts(WINDOW_SAMPLES + 2 * STEP_SAMPLES) == [0, STEP_SAMPLES, 2 * STEP_SAMPLES]
    starts = window_starts(WINDOW_SAMPLES + STEP_SAMPLES // 2)
    assert starts == [0, STEP_SAMPLES]  # the remainder gets its own padded window


def test_overlapping_windows_are_averaged_frame_by_frame() -> None:
    windows = np.stack([np.full((4, 3), 1.0), np.full((4, 3), 3.0)])
    grid = aggregate(windows, [0.0, 2 * FRAME_STEP], n_frames=6)
    assert grid[:, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


class _FakeSession:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def run(self, _outputs, feeds):
        batch = feeds["waveforms"]
        self.shapes.append(batch.shape)
        out = np.zeros((batch.shape[0], FRAMES, 3), dtype=np.float32)
        out[..., 1] = 20.0
        return [out]


def test_the_whole_recording_comes_back_on_one_grid() -> None:
    model = Brouhaha(model_path=Path("/nonexistent/brouhaha.onnx"))
    assert not model.available
    model._session = _FakeSession()
    frames = model.frames(np.zeros(SAMPLE_RATE * 20, dtype=np.float32), SAMPLE_RATE, batch_size=4)
    assert frames.shape == (int(np.ceil(20 / FRAME_STEP)), 3)
    read = ~np.isnan(frames[:, 1])
    assert np.allclose(frames[read, 1], 20.0)
    assert read[:-4].all() and not read[-1]  # ~60 ms past the last window's frames
    assert all(s[0] <= 4 and s[1:] == (1, WINDOW_SAMPLES) for s in model._session.shapes)


def test_audio_at_another_rate_is_refused() -> None:
    model = Brouhaha(model_path=Path("/nonexistent/brouhaha.onnx"))
    model._session = _FakeSession()
    with pytest.raises(ValueError, match="16000"):
        model.frames(np.zeros(8000, dtype=np.float32), 8000)
