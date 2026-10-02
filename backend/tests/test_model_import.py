"""Importing a fine-tuned model's card and transcripts from its folder (D83)."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import (
    AsrModel,
    AuditLog,
    DiarizationRun,
    ModelEvalClip,
    ModelEvalRun,
    SegmentLabel,
    SpeakerTurn,
)
from app.models import Segment as SegmentRow
from app.services.fold import fold_version
from app.services.model_import import (
    ModelImportError,
    import_model_dir,
    read_card,
    reclassify_runs,
    remove_model,
    rescore_runs,
    scan_models,
)
from tests.model_support import CARD, write_model

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


def _gold_ids(session: Session) -> list[str]:
    return sorted(
        session.scalars(sa.select(SegmentRow.external_id).where(SegmentRow.pot == "gold"))
    )


@pytest.mark.db
def test_a_gold_run_is_scored_against_the_current_labels(
    db_session: Session, model_corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    gold = _gold_ids(db_session)
    rows = [{"segment_id": gold[0], "text": "एक दुई तीन", "compute_s": 0.1}] + [
        {"segment_id": sid, "text": model_corpus[sid]} for sid in gold[1:]
    ]
    folder = write_model(tmp_path / "models", "flex-ft", gold=rows)

    report = import_model_dir(db_session, folder, settings=settings, actor="test")

    assert report.runs_created == 1
    model = db_session.scalars(sa.select(AsrModel).where(AsrModel.slug == "flex-ft")).one()
    assert model.name == CARD["name"] and model.trained_at is not None
    run = model.runs[0]
    assert (run.split, run.decoder, run.clip_count) == ("gold", "greedy+cap+retry", 3)
    assert run.metrics_jsonb["errors"] == 2  # one clip lost two words
    by_class = run.metrics_jsonb["by_class"]
    assert by_class["overlap"]["0-5%"]["clips"] == 1  # 0.5 s of a longer clip
    assert by_class["speakers"]["undiarized"]["clips"] == 3  # never diarized, not one speaker
    by_word = run.metrics_jsonb["by_word_class"]
    assert by_word["devanagari"]["words"] == 12  # every word but the trailing digit
    assert by_word["edge"]["wer"] > 0  # the lost words were the clip's last two
    assert run.metrics_jsonb["by_genre"]["podcast"]["clips"] == 3
    worst = db_session.scalars(
        sa.select(ModelEvalClip)
        .where(ModelEvalClip.run_id == run.id)
        .order_by(ModelEvalClip.errors.desc())
    ).first()
    assert (
        worst.ref_text == model_corpus[gold[0]] and worst.deletions == 2 and worst.compute_s == 0.1
    )
    assert 0 < worst.overlap_share < 0.05
    assert worst.classes_jsonb["overlap"] == "0-5%"
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
    db_session: Session, model_corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    rows = [{"segment_id": sid, "text": model_corpus[sid]} for sid in _gold_ids(db_session)]
    folder = write_model(tmp_path / "models", "m", gold=rows)
    import_model_dir(db_session, folder, settings=settings, actor="test")
    again = import_model_dir(db_session, folder, settings=settings, actor="test")
    assert (again.runs_created, again.runs_unchanged) == (0, 1)
    assert db_session.scalar(sa.select(sa.func.count()).select_from(ModelEvalRun)) == 1


@pytest.mark.db
def test_an_edited_card_updates_the_model(
    db_session: Session, model_corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    folder = write_model(tmp_path / "models", "m")
    import_model_dir(db_session, folder, settings=settings, actor="test")
    write_model(tmp_path / "models", "m", card=CARD | {"description": "rewritten"})
    import_model_dir(db_session, folder, settings=settings, actor="test")
    model = db_session.scalars(sa.select(AsrModel).where(AsrModel.slug == "m")).one()
    assert model.description == "rewritten"


@pytest.mark.db
def test_an_unknown_clip_refuses_the_whole_file(
    db_session: Session, model_corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    rows = [{"segment_id": "not_a_clip_0001", "text": "x"}]
    folder = write_model(tmp_path / "models", "m", gold=rows)
    with pytest.raises(ModelImportError, match="not_a_clip_0001"):
        import_model_dir(db_session, folder, settings=settings, actor="test")


@pytest.mark.db
def test_an_empty_file_is_refused(
    db_session: Session, model_corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    folder = write_model(tmp_path / "models", "m", gold=[])
    with pytest.raises(ModelImportError, match="no clips"):
        import_model_dir(db_session, folder, settings=settings, actor="test")


@pytest.mark.db
def test_a_duplicate_clip_is_refused(
    db_session: Session, model_corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    sid = _gold_ids(db_session)[0]
    folder = write_model(tmp_path / "models", "m", gold=[{"segment_id": sid, "text": "a"}] * 2)
    with pytest.raises(ModelImportError, match="twice"):
        import_model_dir(db_session, folder, settings=settings, actor="test")


@pytest.mark.db
def test_clips_no_longer_in_the_split_are_skipped_and_counted(
    db_session: Session, model_corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    """Gold changes by hand (D71). A clip that has left gold since the notebook ran is not scored
    as gold, and the run says how many it left out."""
    train_id = next(sid for sid in model_corpus if sid not in _gold_ids(db_session))
    rows = [
        {"segment_id": sid, "text": model_corpus[sid]} for sid in [*_gold_ids(db_session), train_id]
    ]
    folder = write_model(tmp_path / "models", "m", gold=rows)
    report = import_model_dir(db_session, folder, settings=settings, actor="test")
    run = db_session.scalars(sa.select(ModelEvalRun)).one()
    assert run.clip_count == 3
    assert run.metrics_jsonb["skipped"] == {"not_in_split": 1, "no_reference": 0}
    assert report.skipped == 1


@pytest.mark.db
def test_the_reference_is_a_snapshot(
    db_session: Session, model_corpus: dict[str, str], tmp_path: Path, settings: Settings
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
    assert clip.ref_text == model_corpus[sid]


@pytest.mark.db
def test_scan_imports_every_model_folder(
    db_session: Session, model_corpus: dict[str, str], tmp_path: Path, settings: Settings
) -> None:
    root = tmp_path / "models"
    write_model(root, "a")
    write_model(root, "b", card=CARD | {"name": "B"})
    (root / "not-a-model").mkdir()
    report = scan_models(db_session, root, settings=settings, actor="test")
    assert report.models == ["a", "b"]
    assert db_session.scalar(sa.select(sa.func.count()).select_from(AsrModel)) == 2


def test_reclassifying_a_run_rebuilds_its_classes_and_keeps_its_scores(
    db_session: Session, tmp_path: Path, settings: Settings, model_corpus: dict[str, str]
) -> None:
    gold = _gold_ids(db_session)
    rows = [{"segment_id": sid, "text": model_corpus[sid]} for sid in gold]
    import_model_dir(
        db_session, write_model(tmp_path / "models", "flex-ft", gold=rows), actor="test"
    )
    run = db_session.scalars(sa.select(ModelEvalRun)).one()
    before = dict(run.metrics_jsonb)
    assert list(before["by_class"]["speakers"]) == ["undiarized"]

    segment = db_session.scalars(
        sa.select(SegmentRow).where(SegmentRow.external_id == gold[0])
    ).one()
    db_session.add(
        DiarizationRun(
            episode_id=segment.episode_id,
            model="test",
            checksum="later",
            speakers_jsonb=["A"],
            turns=[SpeakerTurn(speaker="A", start_time=0.0, end_time=1000.0)],
        )
    )
    db_session.flush()

    report = reclassify_runs(db_session, actor="test")

    assert (report.runs, report.clips) == (1, 3)
    assert run.metrics_jsonb["by_class"]["speakers"]["1"]["clips"] == 3
    assert {k: v for k, v in run.metrics_jsonb.items() if k != "by_class"} == {
        k: v for k, v in before.items() if k != "by_class"
    }
    assert all(c.classes_jsonb["speakers"] == "1" for c in run.clips)
    assert run.metrics_jsonb["by_word_class"] == before["by_word_class"]


def test_reclassifying_a_run_picks_up_an_edited_genre(
    db_session: Session, tmp_path: Path, model_corpus: dict[str, str]
) -> None:
    gold = _gold_ids(db_session)
    rows = [{"segment_id": sid, "text": model_corpus[sid]} for sid in gold]
    import_model_dir(
        db_session, write_model(tmp_path / "models", "flex-ft", gold=rows), actor="test"
    )
    run = db_session.scalars(sa.select(ModelEvalRun)).one()
    assert list(run.metrics_jsonb["by_genre"]) == ["podcast"]

    for clip in run.clips:
        episode = clip.segment.episode
        episode.metadata_jsonb = (episode.metadata_jsonb or {}) | {"genre": "advert"}
    db_session.flush()
    reclassify_runs(db_session, actor="test")

    assert list(run.metrics_jsonb["by_genre"]) == ["advert"]


def _import_gold_run(db_session: Session, tmp_path: Path, model_corpus: dict[str, str]):
    gold = _gold_ids(db_session)
    rows = [{"segment_id": gold[0], "text": "एक दुई तीन"}] + [
        {"segment_id": sid, "text": model_corpus[sid]} for sid in gold[1:]
    ]
    import_model_dir(
        db_session, write_model(tmp_path / "models", "flex-ft", gold=rows), actor="test"
    )
    return db_session.scalars(sa.select(ModelEvalRun)).one()


@pytest.mark.db
def test_rescoring_scores_a_run_s_stored_texts_under_today_s_fold(
    db_session: Session, tmp_path: Path, model_corpus: dict[str, str]
) -> None:
    run = _import_gold_run(db_session, tmp_path, model_corpus)
    right = dict(run.metrics_jsonb)
    counts = {c.id: (c.errors, c.deletions, c.raw_errors) for c in run.clips}
    # As an older fold left it: other counts, another version.
    run.fold_version = "fold-v0+norm-v3"
    run.metrics_jsonb = right | {"wer": 99.0, "errors": 99}
    for clip in run.clips:
        clip.errors, clip.deletions, clip.raw_errors = 9, 9, 9
    db_session.flush()

    report = rescore_runs(db_session, actor="test")

    assert (report.runs, report.clips, report.unchanged) == (1, 3, 0)
    assert run.fold_version == fold_version()
    assert run.metrics_jsonb == right  # skipped counts and every breakdown included
    assert {c.id: (c.errors, c.deletions, c.raw_errors) for c in run.clips} == counts
    audit = db_session.scalars(
        sa.select(AuditLog).where(AuditLog.action == "model_eval_rescore")
    ).one()
    assert audit.old_values_jsonb == {"fold_version": "fold-v0+norm-v3", "wer": 99.0}
    assert audit.new_values_jsonb["wer"] == right["wer"]


@pytest.mark.db
def test_rescoring_keeps_the_reference_the_run_was_scored_against(
    db_session: Session, tmp_path: Path, model_corpus: dict[str, str]
) -> None:
    run = _import_gold_run(db_session, tmp_path, model_corpus)
    clip = run.clips[0]
    db_session.add(
        SegmentLabel(
            segment_id=clip.segment_id,
            label_version_id=db_session.get(SegmentLabel, clip.ref_label_id).label_version_id,
            final_text="नयाँ",
            disposition="edited",
            verification_tier="verified",
            annotator="test",
        )
    )
    run.fold_version = "fold-v0+norm-v3"
    db_session.flush()
    before = clip.ref_text

    rescore_runs(db_session, actor="test")

    assert clip.ref_text == before and clip.ref_text != "नयाँ"


@pytest.mark.db
def test_a_run_already_under_today_s_fold_is_left_alone(
    db_session: Session, tmp_path: Path, model_corpus: dict[str, str]
) -> None:
    _import_gold_run(db_session, tmp_path, model_corpus)

    report = rescore_runs(db_session, actor="test")

    assert (report.runs, report.clips, report.unchanged) == (0, 0, 1)
    assert not db_session.scalars(
        sa.select(AuditLog).where(AuditLog.action == "model_eval_rescore")
    ).all()


@pytest.mark.db
def test_removing_a_model_deletes_its_runs_and_clips_and_is_audited(
    db_session: Session, tmp_path: Path, model_corpus: dict[str, str]
) -> None:
    _import_gold_run(db_session, tmp_path, model_corpus)
    import_model_dir(db_session, write_model(tmp_path / "models", "kept"), actor="test")

    def count(table: type) -> int:
        return db_session.scalar(sa.select(sa.func.count()).select_from(table))

    runs, clips = count(ModelEvalRun), count(ModelEvalClip)
    removed = remove_model(db_session, "flex-ft", actor="test")

    assert (removed.runs, removed.clips) == (1, 3)
    assert db_session.scalars(sa.select(AsrModel.slug)).all() == ["kept"]
    assert (count(ModelEvalRun), count(ModelEvalClip)) == (runs - 1, clips - 3)
    audit = db_session.scalars(sa.select(AuditLog).where(AuditLog.action == "model_remove")).one()
    assert audit.entity_id == "flex-ft" and audit.old_values_jsonb["runs"] == 1


@pytest.mark.db
def test_removing_a_model_the_harness_does_not_have_is_refused(db_session: Session) -> None:
    with pytest.raises(ModelImportError, match="nope"):
        remove_model(db_session, "nope", actor="test")
