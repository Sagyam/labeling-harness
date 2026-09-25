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
