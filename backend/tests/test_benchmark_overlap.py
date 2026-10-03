"""Crosstalk, SNR and room echo measured once on each public set (D111)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services.benchmark_overlap import (
    ACOUSTIC_COLUMNS,
    measure_acoustics,
    measure_rows,
    write_acoustics,
    write_overlap,
)
from app.services.error_mining import read_overlap, read_snr, unique_ids
from app.services.overlap import SAMPLE_RATE


class _Detector:
    """Two voices over the first second of every clip, or no measurement at all."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.seen: list[int] = []

    def detect(self, audio: np.ndarray, sample_rate: int):
        assert sample_rate == SAMPLE_RATE
        assert audio.dtype == np.float32 and np.abs(audio).max() <= 1.0
        self.seen.append(len(audio))
        return [(0.0, 1.0)]


def _row(clip_id: str, seconds: float) -> dict:
    audio = np.full(int(seconds * SAMPLE_RATE), 16384, dtype=np.int16)
    return {"segment_id": clip_id, "audio": audio}


def test_each_clip_gets_its_share_spans_and_duration() -> None:
    out = list(measure_rows(_Detector(), [_row("a", 4.0), _row("b", 0.5)]))
    assert out == [
        {"clip_id": "a", "duration": 4.0, "overlap_share": 0.25, "spans": [[0.0, 1.0]]},
        {"clip_id": "b", "duration": 0.5, "overlap_share": 1.0, "spans": [[0.0, 1.0]]},
    ]


def test_clips_already_measured_are_skipped_without_measuring() -> None:
    detector = _Detector()
    out = list(measure_rows(detector, [_row("a", 1.0), _row("b", 2.0)], done={"a"}))
    assert [r["clip_id"] for r in out] == ["b"]
    assert detector.seen == [2 * SAMPLE_RATE]


def test_repeated_ids_are_numbered_like_the_error_rows() -> None:
    rows = [_row("-None", 1.0), _row("x", 1.0), _row("-None", 1.0)]
    out = list(measure_rows(_Detector(), rows))
    assert [r["clip_id"] for r in out] == ["-None", "x", "-None#2"]
    assert list(unique_ids(["-None", "x", "-None"])) == ["-None", "x", "-None#2"]


def test_no_model_is_refused_rather_than_written_as_clean() -> None:
    with pytest.raises(RuntimeError, match="overlap model"):
        list(measure_rows(_Detector(available=False), [_row("a", 1.0)]))


def test_written_then_read_as_shares(tmp_path: Path) -> None:
    partial = tmp_path / "fleurs.jsonl"
    found = list(measure_rows(_Detector(), [_row("a", 4.0), _row("b", 2.0)]))
    partial.write_text("".join(json.dumps(r) + "\n" for r in found))
    path = tmp_path / "fleurs.parquet"
    write_overlap(partial, path)
    assert read_overlap(path) == {"a": 0.25, "b": 0.5}


# --- SNR, C50 and bandwidth: the other recording condition ---------------------------------------


class _Brouhaha:
    """Speech at 12 dB SNR and C50 30 dB over every frame of every clip."""

    available = True

    def frames(self, audio, sample_rate):
        from app.services.brouhaha import FRAME_STEP

        n = int(np.ceil(len(audio) / sample_rate / FRAME_STEP))
        out = np.zeros((n, 3))
        out[:, 0], out[:, 1], out[:, 2] = 0.9, 12.0, 30.0
        return out


def test_each_clip_gets_its_snr_c50_and_bandwidth() -> None:
    from app.services.acoustics import AcousticMeter

    out = list(measure_acoustics(AcousticMeter(_Brouhaha()), [_row("a", 2.0), _row("a", 1.0)]))
    assert [r["clip_id"] for r in out] == ["a", "a#2"]
    assert [r["duration"] for r in out] == [2.0, 1.0]
    assert all(r["snr_db"] == 12.0 and r["c50_db"] == 30.0 for r in out)
    assert set(out[0]) == set(ACOUSTIC_COLUMNS)


def test_a_clip_brouhaha_hears_no_speech_in_is_unmeasured_not_clean() -> None:
    from app.services.acoustics import AcousticMeter

    class Silent(_Brouhaha):
        def frames(self, audio, sample_rate):
            out = super().frames(audio, sample_rate)
            out[:, 0] = 0.0
            return out

    (found,) = measure_acoustics(AcousticMeter(Silent()), [_row("a", 1.0)])
    assert found["snr_db"] is None and found["c50_db"] is None


def test_acoustics_without_brouhaha_are_refused() -> None:
    from app.services.acoustics import AcousticMeter

    with pytest.raises(RuntimeError, match="Brouhaha"):
        list(measure_acoustics(AcousticMeter(), [_row("a", 1.0)]))


def test_acoustics_skip_clips_already_measured(tmp_path: Path) -> None:
    from app.services.acoustics import AcousticMeter

    meter = AcousticMeter(_Brouhaha())
    out = list(measure_acoustics(meter, [_row("a", 1.0), _row("b", 1.0)], done={"a"}))
    assert [r["clip_id"] for r in out] == ["b"]
    partial, path = tmp_path / "slr54.jsonl", tmp_path / "slr54.parquet"
    partial.write_text("".join(json.dumps(r) + "\n" for r in out))
    write_acoustics(partial, path)
    assert read_snr(path) == {"b": 12.0}
