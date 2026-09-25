"""Cutting one downloaded recording into the distillation corpus (D101), end to end: ffmpeg, the
VAD (its energy fallback, so no model is needed) and the clip files. No database."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from app.services.distill_corpus import SourceInput
from app.services.distill_prep import prepare_source
from app.services.silero_vad import SileroVAD
from tests.ingest_support import make_test_audio

VIDEO = "dQw4w9WgXcQ"


def _input(tmp_path: Path, *, seconds: float = 45.0) -> SourceInput:
    incoming = tmp_path / "incoming"
    incoming.mkdir(exist_ok=True)
    audio = make_test_audio(incoming / "Show ep 1.wav", duration_seconds=seconds)
    info = incoming / "Show ep 1.info.json"
    info.write_text(
        json.dumps({"id": VIDEO, "channel": "Some Podcast", "title": "Ep 1"}), encoding="utf-8"
    )
    return SourceInput(audio=audio, info=info)


def _prepare(tmp_path: Path, inp: SourceInput, **over):
    kwargs = {
        "root": tmp_path / "distill",
        "known_video_ids": set(),
        "blocked_channels": [],
        "vad": SileroVAD(model_path=tmp_path / "no-model.onnx"),
        "workers": 2,
    }
    return prepare_source(inp, **{**kwargs, **over})


def test_a_new_source_is_cut_into_16khz_mono_clips_with_a_manifest(tmp_path: Path) -> None:
    result = _prepare(tmp_path, _input(tmp_path))
    assert result.status == "prepared" and result.clips >= 2

    folder = tmp_path / "distill" / "sources" / f"yt-{VIDEO}"
    source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
    assert source["video_id"] == VIDEO and source["channel"] == "Some Podcast"
    assert source["clips"] == result.clips

    rows = [
        json.loads(line)
        for line in (folder / "clips.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == result.clips
    for row in rows:
        info = sf.info(str(tmp_path / "distill" / row["path"]))
        assert (info.samplerate, info.channels, info.format) == (16000, 1, "FLAC")
        assert 0 < row["duration"] <= 20.0
    # the whole normalised recording is not kept, and no half-finished folder is left behind
    assert not (folder / "normalized.flac").exists()
    assert not folder.with_name(folder.name + ".partial").exists()


def test_a_source_already_in_the_corpus_is_not_cut_again(tmp_path: Path) -> None:
    inp = _input(tmp_path)
    _prepare(tmp_path, inp)
    manifest = tmp_path / "distill" / "sources" / f"yt-{VIDEO}" / "clips.jsonl"
    before = manifest.stat().st_mtime_ns
    assert _prepare(tmp_path, inp).status == "done"
    assert manifest.stat().st_mtime_ns == before


def test_a_refused_source_is_logged_and_nothing_is_cut(tmp_path: Path) -> None:
    result = _prepare(tmp_path, _input(tmp_path), known_video_ids={VIDEO})
    assert result.status == "refused" and "already" in result.reason
    assert not (tmp_path / "distill" / "sources").exists()
    logged = json.loads((tmp_path / "distill" / "refused.jsonl").read_text(encoding="utf-8"))
    assert logged["video_id"] == VIDEO and "already" in logged["reason"]


def test_an_interrupted_source_is_redone_from_scratch(tmp_path: Path) -> None:
    partial = tmp_path / "distill" / "sources" / f"yt-{VIDEO}.partial"
    (partial / "clips").mkdir(parents=True)
    (partial / "clips" / "stale.flac").write_bytes(b"half a file")
    result = _prepare(tmp_path, _input(tmp_path))
    assert result.status == "prepared"
    assert not partial.exists()
    clips = sorted((tmp_path / "distill" / "sources" / f"yt-{VIDEO}" / "clips").iterdir())
    assert "stale.flac" not in [c.name for c in clips]
