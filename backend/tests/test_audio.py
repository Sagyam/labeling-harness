"""Tests for clip format validation and waveform peak precomputation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import load_settings
from app.services.audio import ClipFormatError, compute_peaks, probe, validate_clip
from app.services.silero_vad import SpeechTurn, speech_spans_within


def write_audio(
    path: Path,
    *,
    seconds: float = 1.0,
    samplerate: int = 16000,
    channels: int = 1,
    format: str = "FLAC",
    subtype: str | None = None,
) -> Path:
    frames = int(seconds * samplerate)
    t = np.linspace(0, seconds, frames, endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 220 * t)
    if channels > 1:
        data = np.stack([data] * channels, axis=1)
    sf.write(path, data, samplerate, format=format, subtype=subtype)
    return path


def test_probe_reports_format_and_duration(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "clip.flac", seconds=2.0)
    info = probe(path)
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert info.format == "FLAC"
    assert info.duration_seconds == pytest.approx(2.0, abs=1e-3)


def test_validate_accepts_16khz_mono_flac(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "clip.flac")
    assert validate_clip(path, load_settings()).format == "FLAC"


def test_validate_rejects_wav(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "clip.wav", format="WAV")
    with pytest.raises(ClipFormatError, match="FLAC"):
        validate_clip(path, load_settings())


def test_validate_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "clip.flac", samplerate=44100)
    with pytest.raises(ClipFormatError, match="16000"):
        validate_clip(path, load_settings())


def test_validate_rejects_stereo(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "clip.flac", channels=2)
    with pytest.raises(ClipFormatError, match="mono"):
        validate_clip(path, load_settings())


def test_validate_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ClipFormatError, match="not found"):
        validate_clip(tmp_path / "nope.flac", load_settings())


def test_validate_error_names_the_offending_file(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "bad.wav", format="WAV")
    with pytest.raises(ClipFormatError, match=r"bad\.wav"):
        validate_clip(path, load_settings())


def test_peaks_have_the_requested_bucket_count(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "clip.flac", seconds=3.0)
    peaks = compute_peaks(path, buckets=200)
    assert peaks["buckets"] == 200
    assert len(peaks["min"]) == 200
    assert len(peaks["max"]) == 200
    assert peaks["sample_rate"] == 16000
    assert peaks["duration_seconds"] == pytest.approx(3.0, abs=1e-3)


def test_peaks_bracket_the_signal(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "clip.flac")
    peaks = compute_peaks(path, buckets=50)
    assert all(-1.0 <= v <= 1.0 for v in peaks["min"])
    assert all(-1.0 <= v <= 1.0 for v in peaks["max"])
    assert all(lo <= hi for lo, hi in zip(peaks["min"], peaks["max"], strict=True))
    assert max(peaks["max"]) > 0.4
    assert min(peaks["min"]) < -0.4


def test_peaks_are_deterministic(tmp_path: Path) -> None:
    path = write_audio(tmp_path / "clip.flac", seconds=2.0)
    assert compute_peaks(path, buckets=64) == compute_peaks(path, buckets=64)


def test_peaks_handle_a_clip_shorter_than_the_bucket_count(tmp_path: Path) -> None:
    """A 10 ms clip has fewer frames than buckets; the arrays must still be well formed."""
    path = write_audio(tmp_path / "tiny.flac", seconds=0.01)
    peaks = compute_peaks(path, buckets=1000)
    assert len(peaks["min"]) == 1000
    assert len(peaks["max"]) == 1000


def test_peaks_of_silence_are_flat(tmp_path: Path) -> None:
    path = tmp_path / "silence.flac"
    sf.write(path, np.zeros(16000), 16000, format="FLAC")
    peaks = compute_peaks(path, buckets=10)
    assert peaks["min"] == [0.0] * 10
    assert peaks["max"] == [0.0] * 10


# --- clip-relative speech spans ------------------------------------------------------------


def test_speech_spans_are_reported_relative_to_the_clip() -> None:
    """Clip-relative by construction, like word spans (D26), so the two can be compared."""
    turns = [SpeechTurn(10.0, 12.0)]
    assert speech_spans_within(turns, 9.0, 14.0) == [(1.0, 3.0)]


def test_a_turn_is_clipped_to_the_segment_it_overlaps() -> None:
    turns = [SpeechTurn(0.0, 30.0)]
    assert speech_spans_within(turns, 5.0, 8.0) == [(0.0, 3.0)]


def test_turns_outside_the_segment_are_dropped() -> None:
    turns = [SpeechTurn(0.0, 1.0), SpeechTurn(50.0, 51.0)]
    assert speech_spans_within(turns, 10.0, 20.0) == []


def test_a_turn_touching_only_the_boundary_is_not_speech_in_this_clip() -> None:
    assert speech_spans_within([SpeechTurn(0.0, 5.0)], 5.0, 10.0) == []


def test_several_turns_inside_one_clip_are_all_reported_in_order() -> None:
    turns = [SpeechTurn(1.0, 2.0), SpeechTurn(3.0, 4.0)]
    assert speech_spans_within(turns, 0.0, 5.0) == [(1.0, 2.0), (3.0, 4.0)]


def test_no_turns_means_no_speech_spans() -> None:
    assert speech_spans_within([], 0.0, 5.0) == []
