"""Tests for dataset export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.models import AnnotationTask, Episode, Segment
from app.services.export import ExportError, export_dataset
from app.services.fixtures import build_export_fixture
from app.services.importer import import_manifest
from app.services.labeling import Decision, record_decision
from app.services.queue_builder import build_queue
from app.storage.local import LocalFilesystemStorage

pytestmark = pytest.mark.db


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(root=tmp_path / "objects")


def labeled_corpus(
    session: Session,
    tmp_path: Path,
    storage,
    settings: Settings,
    *,
    with_words: bool = True,
) -> dict[str, str]:
    """Import three episodes, force one into each split, then label every segment.

    Dispositions cycle so every export kind has something to select.
    """
    splits = {}
    for index, split in enumerate(("train", "val", "test")):
        episode_id = f"exp_ep{index:03d}"
        root = build_export_fixture(
            tmp_path / f"export_{episode_id}",
            episode_id=episode_id,
            segments=4,
            systems=2,
            with_words=with_words,
        )
        import_manifest(session, root, storage=storage, settings=settings)
        episode = session.scalars(sa.select(Episode).where(Episode.external_id == episode_id)).one()
        episode.split = split
        splits[episode_id] = split
    session.flush()
    build_queue(session, settings=settings, audit_sample_rate=0.0)

    dispositions = ["accepted_unchanged", "edited", "uncertain", "unusable_audio"]
    for position, task in enumerate(
        session.scalars(sa.select(AnnotationTask).order_by(AnnotationTask.id))
    ):
        disposition = dispositions[position % len(dispositions)]
        record_decision(
            session,
            task,
            Decision(
                disposition=disposition,
                final_text="मैले सच्याएको text" if disposition == "edited" else None,
            ),
            settings=settings,
        )
    session.flush()
    return splits


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --- training export ---------------------------------------------------------------------


def test_training_export_contains_only_train_and_val(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="training", output_root=tmp_path / "out")
    records = read_jsonl(result.data_path)
    assert records
    assert {r["split"] for r in records} <= {"train", "val"}


def test_a_test_segment_never_appears_in_the_training_export(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    """Asserted, not assumed: this is the failure that silently invalidates every benchmark."""
    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="training", output_root=tmp_path / "out")
    test_ids = {
        s.external_id
        for s in db_session.scalars(sa.select(Segment).join(Episode).where(Episode.split == "test"))
    }
    exported = {r["segment_id"] for r in read_jsonl(result.data_path)}
    assert exported & test_ids == set()


def test_training_export_excludes_unusable_and_uncertain(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="training", output_root=tmp_path / "out")
    assert {r["disposition"] for r in read_jsonl(result.data_path)} <= {
        "accepted_unchanged",
        "edited",
    }


def test_training_records_have_the_documented_shape(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="training", output_root=tmp_path / "out")
    record = read_jsonl(result.data_path)[0]
    assert set(record) >= {
        "segment_id",
        "episode_id",
        "audio_path",
        "start_time",
        "end_time",
        "text",
        "disposition",
        "seed_system_id",
        "label_version",
        "policy_version",
        "split",
        "code_switch_density",
    }
    assert record["text"]
    assert record["label_version"] == "v1"
    assert record["policy_version"] == "policy_v1"


def test_disposition_and_seed_system_are_retained(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    """These two fields are what make the dataset defensible to a reviewer."""
    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="training", output_root=tmp_path / "out")
    for record in read_jsonl(result.data_path):
        assert record["disposition"] in {"accepted_unchanged", "edited"}
        assert record["seed_system_id"]


def test_exported_text_matches_the_current_label(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="training", output_root=tmp_path / "out")
    from app.services.labeling import latest_label

    for record in read_jsonl(result.data_path):
        segment = db_session.scalars(
            sa.select(Segment).where(Segment.external_id == record["segment_id"])
        ).one()
        assert record["text"] == latest_label(db_session, segment.id).final_text


def test_only_the_latest_label_is_exported(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings)
    task = db_session.scalars(
        sa.select(AnnotationTask).join(Segment).join(Episode).where(Episode.split == "train")
    ).first()
    task.status = "pending"
    db_session.flush()
    record_decision(
        db_session, task, Decision(disposition="edited", final_text="the corrected version")
    )

    result = export_dataset(db_session, kind="training", output_root=tmp_path / "out")
    segment = db_session.get(Segment, task.segment_id)
    match = [r for r in read_jsonl(result.data_path) if r["segment_id"] == segment.external_id]
    assert [r["text"] for r in match] == ["the corrected version"]


# --- gold, analytics and error mining ----------------------------------------------------


def test_gold_export_contains_only_the_test_split(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="gold", output_root=tmp_path / "out")
    records = read_jsonl(result.data_path)
    assert records
    assert {r["split"] for r in records} == {"test"}
    assert all(r["seed_system_id"] for r in records)


def test_unusable_audio_never_appears_in_training_or_gold(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings)
    for kind in ("training", "gold"):
        result = export_dataset(db_session, kind=kind, output_root=tmp_path / f"out_{kind}")
        assert "unusable_audio" not in {r["disposition"] for r in read_jsonl(result.data_path)}


def test_analytics_export_includes_word_level_fields(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="analytics", output_root=tmp_path / "out")
    records = read_jsonl(result.data_path)
    assert records
    assert any(r["hypotheses"][0].get("words") for r in records)
    assert "scores" in records[0]
    # The speaker label is exported even where no transcriber reported one, so a consumer can
    # tell "nobody diarized this" from "the field was dropped".
    words = next(r["hypotheses"][0]["words"] for r in records if r["hypotheses"][0].get("words"))
    assert "speaker" in words[0]


def test_analytics_export_succeeds_without_word_timestamps(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings, with_words=False)
    result = export_dataset(db_session, kind="analytics", output_root=tmp_path / "out")
    records = read_jsonl(result.data_path)
    assert records
    assert all(r["hypotheses"][0]["words"] == [] for r in records)


def test_error_mining_export_holds_the_problem_segments(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="error_mining", output_root=tmp_path / "out")
    dispositions = {r["disposition"] for r in read_jsonl(result.data_path)}
    assert dispositions <= {"uncertain", "unusable_audio"}
    assert dispositions


def test_an_unknown_export_kind_is_refused(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    with pytest.raises(ExportError, match="unknown export kind"):
        export_dataset(db_session, kind="everything", output_root=tmp_path / "out")


# --- manifest and determinism ------------------------------------------------------------


def test_manifest_records_provenance(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="training", output_root=tmp_path / "out")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "training"
    assert manifest["label_version"] == "v1"
    assert manifest["policy_version"] == "policy_v1"
    assert manifest["filters"]["splits"] == ["train", "val"]
    assert manifest["row_counts_by_split"]
    assert manifest["files"][0]["sha256"]
    assert manifest["exported_at"]
    assert "git_commit" in manifest
    assert manifest["import_runs"]


def test_manifest_row_counts_match_the_files(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="training", output_root=tmp_path / "out")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_count"] == len(read_jsonl(result.data_path))
    assert sum(manifest["row_counts_by_split"].values()) == manifest["row_count"]


def test_manifest_checksum_matches_the_data_file(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    from app.utils.hashing import sha256_file

    labeled_corpus(db_session, tmp_path, storage, settings)
    result = export_dataset(db_session, kind="training", output_root=tmp_path / "out")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    entry = next(f for f in manifest["files"] if f["name"] == result.data_path.name)
    assert f"sha256:{entry['sha256']}" == sha256_file(result.data_path)


def test_two_consecutive_exports_are_byte_identical(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    from app.utils.hashing import sha256_file

    labeled_corpus(db_session, tmp_path, storage, settings)
    first = export_dataset(db_session, kind="training", output_root=tmp_path / "a")
    second = export_dataset(db_session, kind="training", output_root=tmp_path / "b")
    assert sha256_file(first.data_path) == sha256_file(second.data_path)


def test_export_of_an_empty_database_is_valid(
    db_session: Session, tmp_path: Path, settings: Settings
) -> None:
    result = export_dataset(db_session, kind="training", output_root=tmp_path / "out")
    assert result.row_count == 0
    assert json.loads(result.manifest_path.read_text())["row_count"] == 0
