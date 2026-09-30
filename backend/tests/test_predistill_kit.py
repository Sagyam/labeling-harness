"""PreDistill's per-file worker (notebooks/src/prekit.py, D101): one recording from the owner's zip
through ingest's own normalisation and VAD, into a whole 16 kHz FLAC and its clip rows. It runs in
Colab against the harness's own modules, so it is tested here against the same ones."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import soundfile as sf

from tests.ingest_support import make_test_audio

_SRC = Path(__file__).resolve().parents[2] / "notebooks" / "src"
sys.path.insert(0, str(_SRC))  # prekit imports the distill kit beside it
_spec = importlib.util.spec_from_file_location("prekit", _SRC / "prekit.py")
prekit = importlib.util.module_from_spec(_spec)
sys.modules["prekit"] = prekit
_spec.loader.exec_module(prekit)


def _run(tmp_path: Path, name: str = "Nepali_Podcast_07.wav", blocked=()):
    audio = make_test_audio(tmp_path / name, duration_seconds=45.0)
    # a model path that does not exist: the VAD's energy fallback, which hears the test tones
    return prekit.process_file(
        str(audio), str(tmp_path / "out"), list(blocked), model_path=str(tmp_path / "none.onnx")
    )


def test_a_recording_becomes_one_whole_flac_and_its_clip_rows(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result["status"] == "prepared" and result["clips"] >= 2
    assert (result["source_id"], result["channel"]) == ("Nepali_Podcast_07", "Nepali_Podcast")
    info = sf.info(str(tmp_path / "out" / "episodes" / "Nepali_Podcast_07.flac"))
    assert (info.samplerate, info.channels, info.format) == (16000, 1, "FLAC")
    saved = json.loads((tmp_path / "out" / "sources" / "Nepali_Podcast_07.json").read_text("utf-8"))
    assert len(saved["rows"]) == result["clips"]
    assert all(0 < r["end_time"] - r["start_time"] <= 20.0 for r in saved["rows"])
    assert all(r["end_time"] <= info.duration + 1e-6 for r in saved["rows"])


def test_a_blocked_channel_is_not_processed(tmp_path: Path) -> None:
    result = _run(tmp_path, "Chill_Pill_03.wav", blocked=["chill pill"])
    assert result["status"] == "blocked"
    assert not (tmp_path / "out" / "episodes").exists()


def test_a_file_that_cannot_be_read_fails_alone_and_leaves_nothing_behind(tmp_path: Path) -> None:
    broken = tmp_path / "Broken_01.mp3"
    broken.write_bytes(b"not audio at all")
    result = prekit.process_file(str(broken), str(tmp_path / "out"), [], model_path=None)
    assert result["status"] == "failed" and result["error"]
    assert not (tmp_path / "out" / "episodes" / "Broken_01.flac").exists()
    assert not (tmp_path / "out" / "sources" / "Broken_01.json").exists()


# --- overlapped speech on a recording already cut (D105) -----------------------------------------


def _cut(tmp_path: Path) -> tuple[Path, Path]:
    result = _run(tmp_path)
    out = tmp_path / "out"
    return (
        out / "episodes" / f"{result['source_id']}.flac",
        out / "sources" / f"{result['source_id']}.json",
    )


def test_overlap_is_written_into_each_clip_row_from_the_recording_spans(tmp_path: Path) -> None:
    flac, meta = _cut(tmp_path)
    before = json.loads(meta.read_text("utf-8"))
    first = before["rows"][0]
    span = (first["start_time"] + 0.5, first["start_time"] + 1.5)  # one second inside clip 0
    heard = []

    def detect(audio, sample_rate):
        heard.append((audio.dtype.name, audio.ndim, sample_rate))
        return [span]

    result = prekit.measure_overlap(str(flac), str(meta), str(tmp_path / "again"), detect=detect)
    assert heard == [("float32", 1, 16000)]
    assert result["status"] == "measured" and result["overlap_seconds"] == 1.0
    after = json.loads((tmp_path / "again" / "sources" / meta.name).read_text("utf-8"))
    assert after["overlap"] == "injected" and after["overlap_seconds"] == 1.0
    assert after["rows"][0]["overlap_spans"] == [[0.5, 1.5]]
    assert after["rows"][0]["overlap_share"] == round(1.0 / first["duration"], 4)
    assert all(r["overlap_spans"] == [] and r["overlap_share"] == 0.0 for r in after["rows"][1:])
    # everything the cut wrote is kept
    assert {k: v for k, v in after.items() if k not in ("rows", "overlap", "overlap_seconds")} == {
        k: v for k, v in before.items() if k != "rows"
    }
    assert [r["segment_id"] for r in after["rows"]] == [r["segment_id"] for r in before["rows"]]


def test_without_a_detector_a_recording_stays_unmeasured_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    flac, meta = _cut(tmp_path)
    result = prekit.measure_overlap(
        str(flac), str(meta), str(tmp_path / "again"), model_path=str(tmp_path / "none.onnx")
    )
    assert result["status"] == "unmeasured"
    assert not (tmp_path / "again").exists()


def test_a_recording_that_cannot_be_measured_fails_alone(tmp_path: Path) -> None:
    flac, meta = _cut(tmp_path)

    def detect(audio, sample_rate):
        raise RuntimeError("the graph rejected the input")

    result = prekit.measure_overlap(str(flac), str(meta), str(tmp_path / "again"), detect=detect)
    assert result["status"] == "failed" and "rejected" in result["error"]
    assert not (tmp_path / "again" / "sources" / meta.name).exists()
