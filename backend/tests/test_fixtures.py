"""Tests for the synthetic export fixture builder.

The fixture builder is what every importer and export test runs against, so its own guarantees
(valid manifest, real FLAC clips, deterministic output) are worth asserting directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.services.audio import probe
from app.services.fixtures import build_export_fixture
from app.services.manifest import read_manifest
from app.utils.hashing import sha256_file


def test_fixture_produces_a_readable_manifest(tmp_path: Path) -> None:
    root = build_export_fixture(tmp_path / "export_e", segments=4, systems=3)
    manifest = read_manifest(root)
    assert len(manifest.segments) == 4
    assert all(len(s["hypotheses"]) == 3 for s in manifest.segments)


def test_fixture_clips_are_16khz_mono_flac(tmp_path: Path) -> None:
    root = build_export_fixture(tmp_path / "export_e", segments=2)
    for clip in (root / "clips").glob("*.flac"):
        info = probe(clip)
        assert (info.sample_rate, info.channels, info.format) == (16000, 1, "FLAC")


def test_fixture_checksums_match_the_clips(tmp_path: Path) -> None:
    root = build_export_fixture(tmp_path / "export_e", segments=2)
    for segment in read_manifest(root).segments:
        assert sha256_file(root / segment["clip_path"]) == segment["clip_checksum"]


def test_fixture_is_deterministic(tmp_path: Path) -> None:
    a = build_export_fixture(tmp_path / "a", segments=3, systems=2, seed=7)
    b = build_export_fixture(tmp_path / "b", segments=3, systems=2, seed=7)
    assert (a / "segments.jsonl").read_text() == (b / "segments.jsonl").read_text()
    assert sha256_file(a / "clips" / "fixture-show_ep000_0000.flac") == sha256_file(
        b / "clips" / "fixture-show_ep000_0000.flac"
    )


def test_fixture_can_omit_peaks(tmp_path: Path) -> None:
    root = build_export_fixture(tmp_path / "export_e", segments=2, with_peaks=False)
    assert not (root / "peaks").exists()


def test_fixture_can_include_peaks(tmp_path: Path) -> None:
    root = build_export_fixture(tmp_path / "export_e", segments=2, with_peaks=True)
    payloads = list((root / "peaks").glob("*.json"))
    assert len(payloads) == 2
    assert json.loads(payloads[0].read_text())["buckets"] > 0


def test_fixture_can_omit_word_timestamps(tmp_path: Path) -> None:
    root = build_export_fixture(tmp_path / "export_e", segments=2, with_words=False)
    for segment in read_manifest(root).segments:
        assert all(not h.get("words") for h in segment["hypotheses"])


def test_fixture_can_include_word_timestamps(tmp_path: Path) -> None:
    root = build_export_fixture(tmp_path / "export_e", segments=2, with_words=True)
    words = read_manifest(root).segments[0]["hypotheses"][0]["words"]
    assert words
    assert {"word", "start", "end", "confidence"} <= set(words[0])


def test_fixture_writes_a_non_flac_clip_on_request(tmp_path: Path) -> None:
    """Import must reject WAV clips, so the fixture has to be able to produce one."""
    root = build_export_fixture(tmp_path / "export_e", segments=1, clip_format="WAV")
    assert probe(next((root / "clips").iterdir())).format == "WAV"


def test_fixture_writes_a_wrong_sample_rate_on_request(tmp_path: Path) -> None:
    root = build_export_fixture(tmp_path / "export_e", segments=1, sample_rate=44100)
    assert probe(next((root / "clips").iterdir())).sample_rate == 44100


def test_fixture_can_produce_a_segment_with_no_hypotheses(tmp_path: Path) -> None:
    """Segments with zero hypotheses must reach the error queue, so they must be constructible."""
    root = build_export_fixture(tmp_path / "export_e", segments=3, empty_hypothesis_segments=(1,))
    records = [
        json.loads(line)
        for line in (root / "segments.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert records[1]["hypotheses"] == []


def test_clip_bytes_depend_only_on_seed_and_segment_id(tmp_path: Path) -> None:
    """Changing an unrelated knob must not change the audio, or incremental fixtures would look
    to the importer like intentionally regenerated clips."""
    a = build_export_fixture(tmp_path / "a", segments=2, systems=1, seed=5)
    b = build_export_fixture(tmp_path / "b", segments=2, systems=3, seed=5, with_words=False)
    for index in range(2):
        name = f"fixture-show_ep000_{index:04d}.flac"
        assert sha256_file(a / "clips" / name) == sha256_file(b / "clips" / name)
