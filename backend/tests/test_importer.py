"""Tests for the manifest importer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.models import (
    AsrHypothesis,
    AsrSystem,
    Episode,
    HypothesisWord,
    ImportRun,
    Segment,
    SegmentScore,
)
from app.services.fixtures import build_export_fixture
from app.services.importer import ClipChangedError, ImportError_, import_manifest
from app.storage.local import LocalFilesystemStorage

pytestmark = pytest.mark.db


@pytest.fixture
def storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(root=tmp_path / "objects")


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    return build_export_fixture(
        tmp_path / "export_e", episode_id="imp_ep001", segments=4, systems=2
    )


def run_import(session: Session, root: Path, storage, settings: Settings, **kwargs):
    return import_manifest(session, root, storage=storage, settings=settings, **kwargs)


# --- the happy path ----------------------------------------------------------------------


def test_import_inserts_everything(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    report = run_import(db_session, export_dir, storage, settings)
    assert report.episode_created is True
    assert report.segments_inserted == 4
    assert report.hypotheses_inserted == 8
    assert report.words_inserted > 0
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Segment)) == 4
    assert db_session.scalar(sa.select(sa.func.count()).select_from(AsrHypothesis)) == 8
    assert db_session.scalar(sa.select(sa.func.count()).select_from(AsrSystem)) == 2


def test_import_records_an_import_run(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    report = run_import(db_session, export_dir, storage, settings)
    run = db_session.get(ImportRun, report.import_run_id)
    assert run.status == "succeeded"
    assert run.segments_inserted == 4
    assert run.hypotheses_inserted == 8
    assert run.pipeline_version == "fixture-v1"
    assert run.finished_at is not None


def test_segments_link_to_their_import_run(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    report = run_import(db_session, export_dir, storage, settings)
    segments = db_session.scalars(sa.select(Segment)).all()
    assert {s.import_run_id for s in segments} == {report.import_run_id}


def test_clips_are_uploaded_to_storage(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    run_import(db_session, export_dir, storage, settings)
    for segment in db_session.scalars(sa.select(Segment)):
        assert storage.exists(segment.clip_object_key)
        assert storage.size(segment.clip_object_key) > 0


def test_peaks_exist_for_every_segment(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    run_import(db_session, export_dir, storage, settings)
    for segment in db_session.scalars(sa.select(Segment)):
        assert segment.peaks_object_key
        payload = json.loads(storage.get_bytes(segment.peaks_object_key))
        assert payload["buckets"] == settings.importer.peaks_buckets
        assert len(payload["min"]) == settings.importer.peaks_buckets


def test_supplied_peaks_are_used_instead_of_recomputed(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    root = build_export_fixture(
        tmp_path / "export_p", episode_id="peaks_ep", segments=2, with_peaks=True
    )
    run_import(db_session, root, storage, settings)
    segment = db_session.scalars(sa.select(Segment)).first()
    payload = json.loads(storage.get_bytes(segment.peaks_object_key))
    assert payload["buckets"] == 200  # the fixture writes 200-bucket peaks


def test_scores_and_flags_are_stored_as_received(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    run_import(db_session, export_dir, storage, settings)
    scores = db_session.scalars(sa.select(SegmentScore)).all()
    assert len(scores) == 4
    assert all(s.word_disagreement_rate is not None for s in scores)


def test_rule_flags_are_computed_at_import(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    """A one-word 20-second segment is implausibly slow; the flag must exist without re-reading
    anything at queue-build time."""
    root = build_export_fixture(tmp_path / "export_f", episode_id="flag_ep", segments=4, systems=2)
    lines = (root / "segments.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["start_time"], record["end_time"] = 0.0, 20.0
    for hypothesis in record["hypotheses"]:
        hypothesis["text"] = "एउटा"
        hypothesis.pop("words", None)
    lines[0] = json.dumps(record, ensure_ascii=False)
    (root / "segments.jsonl").write_text("\n".join(lines), encoding="utf-8")

    run_import(db_session, root, storage, settings)
    segment = db_session.scalars(
        sa.select(Segment).where(Segment.external_id == record["segment_id"])
    ).one()
    assert "implausible_speaking_rate" in segment.scores.flags_jsonb


def test_imported_flags_are_preserved_alongside_computed_ones(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    root = build_export_fixture(tmp_path / "export_g", episode_id="keep_ep", segments=2, systems=2)
    lines = (root / "segments.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["flags"] = ["upstream_only_flag"]
    lines[0] = json.dumps(record, ensure_ascii=False)
    (root / "segments.jsonl").write_text("\n".join(lines), encoding="utf-8")

    run_import(db_session, root, storage, settings)
    segment = db_session.scalars(
        sa.select(Segment).where(Segment.external_id == record["segment_id"])
    ).one()
    assert "upstream_only_flag" in segment.scores.flags_jsonb


def test_missing_scores_are_stored_as_null(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    root = build_export_fixture(
        tmp_path / "export_n", episode_id="noscore_ep", segments=2, with_scores=False
    )
    run_import(db_session, root, storage, settings)
    for score in db_session.scalars(sa.select(SegmentScore)):
        assert score.word_disagreement_rate is None
        assert score.code_switch_density is None


def test_word_timestamps_are_imported_in_order(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    run_import(db_session, export_dir, storage, settings)
    hypothesis = db_session.scalars(sa.select(AsrHypothesis)).first()
    positions = [w.position for w in hypothesis.words]
    assert positions == sorted(positions)
    assert positions == list(range(len(positions)))


def test_a_speaker_label_survives_import_and_a_missing_one_stays_null(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    """Diarization is only useful if the label reaches the database (D36)."""
    root = build_export_fixture(tmp_path / "export_spk", episode_id="spk_ep", segments=1)
    manifest = root / "segments.jsonl"
    lines = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        words = record["hypotheses"][0]["words"]
        words[0]["speaker"] = "spk_1"
        if len(words) > 1:
            words[1]["speaker"] = "spk_2"
        lines.append(json.dumps(record, ensure_ascii=False))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_import(db_session, root, storage, settings)
    hypothesis = db_session.scalars(sa.select(AsrHypothesis)).first()
    labels = [w.speaker for w in hypothesis.words]
    assert labels[0] == "spk_1"
    assert labels[1] == "spk_2"
    # Every other transcriber reports none; absent must stay null, not become a default.
    assert all(label is None for label in labels[2:])


def test_import_succeeds_without_word_timestamps(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    root = build_export_fixture(
        tmp_path / "export_w", episode_id="nowords_ep", segments=2, with_words=False
    )
    report = run_import(db_session, root, storage, settings)
    assert report.words_inserted == 0
    assert db_session.scalar(sa.select(sa.func.count()).select_from(HypothesisWord)) == 0


def test_episode_metadata_is_stored(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    run_import(db_session, export_dir, storage, settings)
    episode = db_session.scalars(sa.select(Episode)).one()
    assert episode.title == "Synthetic fixture imp_ep001"
    assert episode.source_audio_checksum.startswith("sha256:")
    assert episode.duration_seconds > 0


# --- frozen splits -----------------------------------------------------------------------


def test_split_is_assigned_at_import(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    report = run_import(db_session, export_dir, storage, settings)
    episode = db_session.scalars(sa.select(Episode)).one()
    assert episode.split in {"train", "val", "test"}
    assert episode.split == report.split
    assert episode.split_seed == settings.importer.split_seed
    assert episode.split_assigned_at is not None


def test_split_is_stable_across_reimport(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    run_import(db_session, export_dir, storage, settings)
    episode = db_session.scalars(sa.select(Episode)).one()
    first_split, assigned_at = episode.split, episode.split_assigned_at

    run_import(db_session, export_dir, storage, settings)
    db_session.expire_all()
    episode = db_session.scalars(sa.select(Episode)).one()
    assert episode.split == first_split
    assert episode.split_assigned_at == assigned_at


def test_split_is_deterministic_for_a_given_seed(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    from app.services.splits import assign_split

    report = run_import(db_session, export_dir, storage, settings)
    assert report.split == assign_split(
        "imp_ep001", seed=settings.importer.split_seed, ratios=settings.importer.split_ratios
    )


# --- idempotency -------------------------------------------------------------------------


def test_reimport_of_an_unchanged_export_inserts_nothing(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    run_import(db_session, export_dir, storage, settings)
    report = run_import(db_session, export_dir, storage, settings)
    assert report.segments_inserted == 0
    assert report.hypotheses_inserted == 0
    assert report.segments_skipped == 4
    assert report.episode_created is False
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Segment)) == 4
    assert db_session.scalar(sa.select(sa.func.count()).select_from(AsrHypothesis)) == 8


def test_reimport_does_not_duplicate_words(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    run_import(db_session, export_dir, storage, settings)
    before = db_session.scalar(sa.select(sa.func.count()).select_from(HypothesisWord))
    run_import(db_session, export_dir, storage, settings)
    assert db_session.scalar(sa.select(sa.func.count()).select_from(HypothesisWord)) == before


def test_a_new_system_on_an_existing_segment_is_inserted(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    root = build_export_fixture(tmp_path / "e1", episode_id="grow_ep", segments=2, systems=1)
    run_import(db_session, root, storage, settings)
    root = build_export_fixture(tmp_path / "e1", episode_id="grow_ep", segments=2, systems=2)
    report = run_import(db_session, root, storage, settings)
    assert report.segments_inserted == 0
    assert report.hypotheses_inserted == 2
    assert db_session.scalar(sa.select(sa.func.count()).select_from(AsrHypothesis)) == 4


def test_new_segments_in_a_later_export_are_inserted(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    build_export_fixture(tmp_path / "e2", episode_id="more_ep", segments=2, systems=1)
    run_import(db_session, tmp_path / "e2", storage, settings)
    build_export_fixture(tmp_path / "e2", episode_id="more_ep", segments=3, systems=1)
    report = run_import(db_session, tmp_path / "e2", storage, settings)
    assert report.segments_inserted == 1
    assert report.segments_skipped == 2


# --- rejection paths ---------------------------------------------------------------------


def test_changed_clip_checksum_is_an_error(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    build_export_fixture(tmp_path / "e3", episode_id="chg_ep", segments=2, systems=1, seed=1)
    run_import(db_session, tmp_path / "e3", storage, settings)
    build_export_fixture(tmp_path / "e3", episode_id="chg_ep", segments=2, systems=1, seed=2)
    with pytest.raises(ClipChangedError, match="--allow-clip-change"):
        run_import(db_session, tmp_path / "e3", storage, settings)


def test_changed_clip_is_accepted_with_the_override(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    build_export_fixture(tmp_path / "e4", episode_id="ovr_ep", segments=2, systems=1, seed=1)
    run_import(db_session, tmp_path / "e4", storage, settings)
    first = [s.clip_checksum for s in db_session.scalars(sa.select(Segment).order_by(Segment.id))]

    build_export_fixture(tmp_path / "e4", episode_id="ovr_ep", segments=2, systems=1, seed=2)
    report = run_import(db_session, tmp_path / "e4", storage, settings, allow_clip_change=True)
    db_session.expire_all()
    second = [s.clip_checksum for s in db_session.scalars(sa.select(Segment).order_by(Segment.id))]
    assert report.clips_replaced == 2
    assert first != second


def test_non_flac_clip_is_rejected(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    root = build_export_fixture(tmp_path / "e5", episode_id="wav_ep", segments=2, clip_format="WAV")
    with pytest.raises(ImportError_, match="FLAC"):
        run_import(db_session, root, storage, settings)


def test_wrong_sample_rate_is_rejected(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    root = build_export_fixture(tmp_path / "e6", episode_id="sr_ep", segments=2, sample_rate=44100)
    with pytest.raises(ImportError_, match="16000"):
        run_import(db_session, root, storage, settings)


def test_stereo_clip_is_rejected(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    root = build_export_fixture(tmp_path / "e7", episode_id="st_ep", segments=1, channels=2)
    with pytest.raises(ImportError_, match="mono"):
        run_import(db_session, root, storage, settings)


def test_a_rejected_import_writes_nothing(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> None:
    root = build_export_fixture(tmp_path / "e8", episode_id="bad_ep", segments=3, clip_format="WAV")
    with pytest.raises(ImportError_):
        run_import(db_session, root, storage, settings)
    db_session.rollback()
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Episode)) == 0
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Segment)) == 0
    assert db_session.scalar(sa.select(sa.func.count()).select_from(ImportRun)) == 0


def test_a_malformed_manifest_writes_nothing(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    (export_dir / "segments.jsonl").write_text("{oops}\n", encoding="utf-8")
    with pytest.raises(ImportError_):
        run_import(db_session, export_dir, storage, settings)
    db_session.rollback()
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Episode)) == 0


def test_a_clip_whose_bytes_do_not_match_the_manifest_checksum_is_rejected(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    lines = (export_dir / "segments.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["clip_checksum"] = "sha256:" + "0" * 64
    lines[0] = json.dumps(record, ensure_ascii=False)
    (export_dir / "segments.jsonl").write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ImportError_, match="checksum"):
        run_import(db_session, export_dir, storage, settings)


def test_a_missing_clip_file_is_rejected(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    next((export_dir / "clips").iterdir()).unlink()
    with pytest.raises(ImportError_, match="not found"):
        run_import(db_session, export_dir, storage, settings)


# --- dry run -----------------------------------------------------------------------------


def test_dry_run_writes_nothing_to_the_database(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    report = run_import(db_session, export_dir, storage, settings, dry_run=True)
    assert report.dry_run is True
    assert report.segments_inserted == 4  # what *would* be inserted
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Episode)) == 0
    assert db_session.scalar(sa.select(sa.func.count()).select_from(Segment)) == 0
    assert db_session.scalar(sa.select(sa.func.count()).select_from(ImportRun)) == 0


def test_dry_run_writes_nothing_to_storage(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    run_import(db_session, export_dir, storage, settings, dry_run=True)
    assert not any((storage.root).rglob("*")) if storage.root.exists() else True


def test_dry_run_after_a_real_import_reports_a_no_op(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    run_import(db_session, export_dir, storage, settings)
    report = run_import(db_session, export_dir, storage, settings, dry_run=True)
    assert report.segments_inserted == 0
    assert report.segments_skipped == 4


def test_report_renders_a_readable_summary(
    db_session: Session, export_dir: Path, storage, settings: Settings
) -> None:
    report = run_import(db_session, export_dir, storage, settings, dry_run=True)
    text = report.render()
    assert "imp_ep001" in text
    assert "DRY RUN" in text
    assert "segments" in text
