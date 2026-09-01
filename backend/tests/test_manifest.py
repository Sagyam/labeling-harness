"""Tests for manifest reading and JSON Schema validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.fixtures import build_export_fixture
from app.services.manifest import ManifestError, read_manifest, validate_episode, validate_segment


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    return build_export_fixture(
        tmp_path / "export_show-a_ep012",
        episode_id="show-a_ep012",
        segments=3,
        systems=2,
    )


def test_reads_a_valid_export(export_dir: Path) -> None:
    manifest = read_manifest(export_dir)
    assert manifest.episode["episode_id"] == "show-a_ep012"
    assert len(manifest.segments) == 3
    assert manifest.root == export_dir
    assert manifest.segments[0]["hypotheses"]


def test_clip_paths_resolve_to_real_files(export_dir: Path) -> None:
    manifest = read_manifest(export_dir)
    for segment in manifest.segments:
        assert (export_dir / segment["clip_path"]).is_file()


def test_missing_episode_json_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ManifestError, match=r"episode\.json"):
        read_manifest(tmp_path / "empty")


def test_missing_segments_jsonl_is_an_error(export_dir: Path) -> None:
    (export_dir / "segments.jsonl").unlink()
    with pytest.raises(ManifestError, match=r"segments\.jsonl"):
        read_manifest(export_dir)


def test_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="not a directory"):
        read_manifest(tmp_path / "nope")


def test_episode_without_an_id_is_rejected(export_dir: Path) -> None:
    (export_dir / "episode.json").write_text(json.dumps({"title": "x"}), encoding="utf-8")
    with pytest.raises(ManifestError, match="episode_id"):
        read_manifest(export_dir)


def test_segment_without_hypotheses_is_rejected(export_dir: Path) -> None:
    lines = (export_dir / "segments.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["hypotheses"] = []
    lines[0] = json.dumps(record)
    (export_dir / "segments.jsonl").write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ManifestError, match="hypotheses"):
        read_manifest(export_dir)


def test_segment_with_malformed_json_names_the_line(export_dir: Path) -> None:
    path = export_dir / "segments.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="line 5"):
        read_manifest(export_dir)


def test_blank_lines_in_segments_jsonl_are_skipped(export_dir: Path) -> None:
    path = export_dir / "segments.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n\n   \n", encoding="utf-8")
    assert len(read_manifest(export_dir).segments) == 3


def test_segment_referencing_another_episode_is_rejected(export_dir: Path) -> None:
    path = export_dir / "segments.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["episode_id"] = "some-other-episode"
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ManifestError, match="episode"):
        read_manifest(export_dir)


def test_duplicate_segment_ids_are_rejected(export_dir: Path) -> None:
    path = export_dir / "segments.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([*lines, lines[0]]), encoding="utf-8")
    with pytest.raises(ManifestError, match="duplicate"):
        read_manifest(export_dir)


def test_duplicate_system_ids_within_a_segment_are_rejected(export_dir: Path) -> None:
    path = export_dir / "segments.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["hypotheses"].append(dict(record["hypotheses"][0]))
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ManifestError, match="duplicate"):
        read_manifest(export_dir)


def test_end_before_start_is_rejected(export_dir: Path) -> None:
    path = export_dir / "segments.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["start_time"], record["end_time"] = 10.0, 5.0
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ManifestError, match="end_time"):
        read_manifest(export_dir)


def test_validate_segment_accepts_missing_optional_fields() -> None:
    validate_segment(
        {
            "segment_id": "s1",
            "episode_id": "e1",
            "start_time": 0.0,
            "end_time": 1.0,
            "clip_path": "clips/s1.flac",
            "hypotheses": [{"system_id": "sys", "text": "hello"}],
        },
        source="test",
    )


def test_validate_episode_accepts_unknown_extra_keys() -> None:
    """Upstream may add fields; the harness keeps them rather than failing."""
    validate_episode({"episode_id": "e1", "future_field": {"a": 1}}, source="test")
