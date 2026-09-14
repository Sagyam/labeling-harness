"""Importing a fine-tuned model's card and transcripts from its folder (D83)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import (
    AnnotationTask,
    AsrModel,
    AuditLog,
    Episode,
    ModelEvalClip,
    ModelEvalRun,
    SegmentLabel,
)
from app.models import Segment as SegmentRow
from app.services.importer import import_manifest
from app.services.labeling import Decision, record_decision
from app.services.model_import import (
    ModelImportError,
    import_model_dir,
    read_card,
    scan_models,
)
from app.services.queue_builder import build_queue
from app.storage.local import LocalFilesystemStorage
from tests.fixtures import build_export_fixture

CARD = {
    "name": "Flex FT 2026-09-12",
    "created_at": "2026-09-12T18:00:00+00:00",
    "description": "04c, standard decoder",
    "architecture": "Canary-style enc-dec",
    "decoder": "greedy+cap+retry",
}


def write_model(root: Path, slug: str, *, card=CARD, gold=None, val=None) -> Path:
    folder = root / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
    for name, rows in (("gold", gold), ("val", val)):
        if rows is not None:
            (folder / f"{name}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
            )
    return folder


# --- the card, no database --------------------------------------------------------------------


def test_the_card_needs_a_name(tmp_path: Path) -> None:
    folder = write_model(tmp_path, "m", card={"description": "no name"})
    with pytest.raises(ModelImportError, match="name"):
        read_card(folder)


def test_a_naive_date_is_read_as_utc(tmp_path: Path) -> None:
    folder = write_model(tmp_path, "m", card=CARD | {"created_at": "2026-09-12T18:00:00"})
    assert read_card(folder).trained_at.isoformat() == "2026-09-12T18:00:00+00:00"


def test_a_folder_without_a_card_is_refused(tmp_path: Path) -> None:
    (tmp_path / "m").mkdir()
    with pytest.raises(ModelImportError, match=r"model_card\.json"):
        read_card(tmp_path / "m")


# --- import, against the database -------------------------------------------------------------


@pytest.fixture
def corpus(db_session: Session, tmp_path: Path, settings: Settings) -> dict[str, str]:
    """One episode of four labelled clips: 0-2 gold, 3 train. Returns id -> final text."""
    root = build_export_fixture(tmp_path / "mx", episode_id="mx_ep", segments=4, systems=2)
    storage = LocalFilesystemStorage(root=tmp_path / "objects")
    import_manifest(db_session, root, storage=storage, settings=settings)
    episode = db_session.scalars(sa.select(Episode).where(Episode.external_id == "mx_ep")).one()
    episode.metadata_jsonb = (episode.metadata_jsonb or {}) | {"genre": "podcast"}
    segments = sorted(episode.segments, key=lambda s: s.external_id)
    for segment in segments[:3]:
        segment.pot = "gold"
    segments[0].overlap_spans_jsonb = [[0.0, 0.5]]
    db_session.flush()
    build_queue(db_session, settings=settings, audit_sample_rate=0.0)
    texts = {}
    for index, task in enumerate(db_session.scalars(sa.select(AnnotationTask))):
        text = f"एक दुई तीन चार {index}"
        record_decision(
            db_session, task, Decision(disposition="edited", final_text=text), settings=settings
        )
        texts[task.segment.external_id] = text
    db_session.flush()
    return texts


def _gold_ids(session: Session) -> list[str]:
    return sorted(
        session.scalars(sa.select(SegmentRow.external_id).where(SegmentRow.pot == "gold"))
    )


@pytest.mark.db
def test_a_gold_run_is_scored_against_the_current_labels(
    db_session: Session, corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    gold = _gold_ids(db_session)
    rows = [{"segment_id": gold[0], "text": "एक दुई तीन", "compute_s": 0.1}] + [
        {"segment_id": sid, "text": corpus[sid]} for sid in gold[1:]
    ]
    folder = write_model(tmp_path / "models", "flex-ft", gold=rows)

    report = import_model_dir(db_session, folder, settings=settings, actor="test")

    assert report.runs_created == 1
    model = db_session.scalars(sa.select(AsrModel).where(AsrModel.slug == "flex-ft")).one()
    assert model.name == CARD["name"] and model.trained_at is not None
    run = model.runs[0]
    assert (run.split, run.decoder, run.clip_count) == ("gold", "greedy+cap+retry", 3)
    assert run.metrics_jsonb["errors"] == 2  # one clip lost two words
    assert run.metrics_jsonb["by_overlap"]["0-5%"]["clips"] == 1  # 0.5 s of a longer clip
    assert run.metrics_jsonb["by_genre"]["podcast"]["clips"] == 3
    worst = db_session.scalars(
        sa.select(ModelEvalClip)
        .where(ModelEvalClip.run_id == run.id)
        .order_by(ModelEvalClip.errors.desc())
    ).first()
    assert worst.ref_text == corpus[gold[0]] and worst.deletions == 2 and worst.compute_s == 0.1
    assert (
        db_session.scalar(
            sa.select(sa.func.count())
            .select_from(AuditLog)
            .where(AuditLog.action == "model_eval_import")
        )
        == 1
    )


@pytest.mark.db
def test_the_same_file_twice_is_a_no_op(
    db_session: Session, corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    rows = [{"segment_id": sid, "text": corpus[sid]} for sid in _gold_ids(db_session)]
    folder = write_model(tmp_path / "models", "m", gold=rows)
    import_model_dir(db_session, folder, settings=settings, actor="test")
    again = import_model_dir(db_session, folder, settings=settings, actor="test")
    assert (again.runs_created, again.runs_unchanged) == (0, 1)
    assert db_session.scalar(sa.select(sa.func.count()).select_from(ModelEvalRun)) == 1


@pytest.mark.db
def test_an_edited_card_updates_the_model(
    db_session: Session, corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    folder = write_model(tmp_path / "models", "m")
    import_model_dir(db_session, folder, settings=settings, actor="test")
    write_model(tmp_path / "models", "m", card=CARD | {"description": "rewritten"})
    import_model_dir(db_session, folder, settings=settings, actor="test")
    model = db_session.scalars(sa.select(AsrModel).where(AsrModel.slug == "m")).one()
    assert model.description == "rewritten"


@pytest.mark.db
def test_an_unknown_clip_refuses_the_whole_file(
    db_session: Session, corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    rows = [{"segment_id": "not_a_clip_0001", "text": "x"}]
    folder = write_model(tmp_path / "models", "m", gold=rows)
    with pytest.raises(ModelImportError, match="not_a_clip_0001"):
        import_model_dir(db_session, folder, settings=settings, actor="test")


@pytest.mark.db
def test_an_empty_file_is_refused(
    db_session: Session, corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    folder = write_model(tmp_path / "models", "m", gold=[])
    with pytest.raises(ModelImportError, match="no clips"):
        import_model_dir(db_session, folder, settings=settings, actor="test")


@pytest.mark.db
def test_a_duplicate_clip_is_refused(
    db_session: Session, corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    sid = _gold_ids(db_session)[0]
    folder = write_model(tmp_path / "models", "m", gold=[{"segment_id": sid, "text": "a"}] * 2)
    with pytest.raises(ModelImportError, match="twice"):
        import_model_dir(db_session, folder, settings=settings, actor="test")


@pytest.mark.db
def test_clips_no_longer_in_the_split_are_skipped_and_counted(
    db_session: Session, corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    """Gold changes by hand (D71). A clip that has left gold since the notebook ran is not scored
    as gold, and the run says how many it left out."""
    train_id = next(sid for sid in corpus if sid not in _gold_ids(db_session))
    rows = [{"segment_id": sid, "text": corpus[sid]} for sid in [*_gold_ids(db_session), train_id]]
    folder = write_model(tmp_path / "models", "m", gold=rows)
    report = import_model_dir(db_session, folder, settings=settings, actor="test")
    run = db_session.scalars(sa.select(ModelEvalRun)).one()
    assert run.clip_count == 3
    assert run.metrics_jsonb["skipped"] == {"not_in_split": 1, "no_reference": 0}
    assert report.skipped == 1


@pytest.mark.db
def test_the_reference_is_a_snapshot(
    db_session: Session, corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    """Relabelling a clip after the import does not change what the run was scored against."""
    sid = _gold_ids(db_session)[0]
    folder = write_model(tmp_path / "models", "m", gold=[{"segment_id": sid, "text": "x"}])
    import_model_dir(db_session, folder, settings=settings, actor="test")
    old = db_session.scalars(
        sa.select(SegmentLabel).join(SegmentRow).where(SegmentRow.external_id == sid)
    ).one()
    db_session.add(
        SegmentLabel(
            segment_id=old.segment_id,
            label_version_id=old.label_version_id,
            final_text="नयाँ",
            disposition="edited",
            verification_tier="verified",
            annotator="test",
        )
    )
    db_session.flush()
    clip = db_session.scalars(sa.select(ModelEvalClip)).one()
    assert clip.ref_text == corpus[sid]


@pytest.mark.db
def test_scan_imports_every_model_folder(
    db_session: Session, corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    root = tmp_path / "models"
    write_model(root, "a")
    write_model(root, "b", card=CARD | {"name": "B"})
    (root / "not-a-model").mkdir()
    report = scan_models(db_session, root, settings=settings, actor="test")
    assert report.models == ["a", "b"]
    assert db_session.scalar(sa.select(sa.func.count()).select_from(AsrModel)) == 2
