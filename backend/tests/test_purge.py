"""Tests for permanently removing one ASR system from the corpus.

Deleting a hypothesis overrides D6's immutability guarantee by explicit decision, so the
dump-before-delete and the rescore are the parts that carry the weight here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import load_llm_routes
from app.models import AsrHypothesis, AsrSystem, AuditLog, Episode, Segment, SegmentScore
from app.models.content import HypothesisWord
from app.services.purge import PurgedSystemNotFound, purge_asr_system

pytestmark = pytest.mark.db

DOOMED = "gemini-3.5-transcribe"
KEPT = "elevenlabs-scribe-v2"


def _corpus(session: Session) -> None:
    """Two segments, each with a kept hypothesis and a doomed one carrying word spans."""
    episode = Episode(external_id="ep_purge", split="train")
    session.add(episode)
    session.flush()

    systems = {}
    for name in (KEPT, DOOMED):
        system = AsrSystem(system_id=name, model_id=name)
        session.add(system)
        systems[name] = system
    session.flush()

    for index, (kept_text, doomed_text) in enumerate(
        [("हामी meeting मा", "हामी meeting मा"), ("आजको data", "completely different words")]
    ):
        segment = Segment(
            episode_id=episode.id,
            external_id=f"seg_purge_{index}",
            start_time=float(index),
            end_time=float(index) + 2.0,
            duration_seconds=2.0,
            clip_object_key=f"clips/seg_purge_{index}.flac",
            clip_checksum="0" * 64,
        )
        session.add(segment)
        session.flush()
        session.add(
            SegmentScore(
                segment_id=segment.id,
                word_disagreement_rate=0.99,
                cer_between_hypotheses=0.99,
            )
        )
        for name, text in ((KEPT, kept_text), (DOOMED, doomed_text)):
            hypothesis = AsrHypothesis(
                segment_id=segment.id,
                asr_system_id=systems[name].id,
                text_raw=text,
            )
            session.add(hypothesis)
            session.flush()
            if name == DOOMED:
                session.add(
                    HypothesisWord(
                        hypothesis_id=hypothesis.id,
                        position=0,
                        word_raw=text.split()[0],
                        start_time=0.1,
                        end_time=0.5,
                    )
                )
    session.flush()


def test_purge_removes_the_named_system_and_leaves_the_others(
    db_session: Session, tmp_path: Path
) -> None:
    _corpus(db_session)
    report = purge_asr_system(db_session, DOOMED, dump_dir=tmp_path)

    assert report.hypotheses_deleted == 2
    assert report.words_deleted == 2
    assert sorted(report.segments_affected) == ["seg_purge_0", "seg_purge_1"]

    assert db_session.scalar(sa.select(AsrSystem).where(AsrSystem.system_id == DOOMED)) is None
    assert db_session.scalar(sa.select(AsrSystem).where(AsrSystem.system_id == KEPT)) is not None
    remaining = db_session.scalars(sa.select(AsrHypothesis)).all()
    assert len(remaining) == 2
    assert db_session.scalars(sa.select(HypothesisWord)).all() == []


def test_purge_dumps_every_row_before_deleting_it(db_session: Session, tmp_path: Path) -> None:
    """The purge is irreversible; the dump is what keeps it recoverable."""
    _corpus(db_session)
    report = purge_asr_system(db_session, DOOMED, dump_dir=tmp_path)

    assert report.dump_path is not None
    lines = [json.loads(line) for line in report.dump_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert {row["segment_external_id"] for row in lines} == {"seg_purge_0", "seg_purge_1"}
    assert all(row["system_id"] == DOOMED for row in lines)
    dumped_words = [w for row in lines for w in row["words"]]
    assert len(dumped_words) == 2
    assert dumped_words[0]["start"] == 0.1


def test_purge_rescores_the_segments_it_changed(db_session: Session, tmp_path: Path) -> None:
    """A score describing two hypotheses is wrong once one of them is gone."""
    _corpus(db_session)
    report = purge_asr_system(db_session, DOOMED, dump_dir=tmp_path)
    assert report.segments_rescored == 2

    scores = db_session.scalars(sa.select(SegmentScore)).all()
    # One hypothesis cannot disagree with anything, which the scorer reads as 0.0.
    assert {s.word_disagreement_rate for s in scores} == {0.0}
    assert {s.cer_between_hypotheses for s in scores} == {0.0}


def test_purge_can_leave_the_scores_alone(db_session: Session, tmp_path: Path) -> None:
    _corpus(db_session)
    report = purge_asr_system(db_session, DOOMED, dump_dir=tmp_path, rescore=False)
    assert report.segments_rescored == 0
    scores = db_session.scalars(sa.select(SegmentScore)).all()
    assert {s.word_disagreement_rate for s in scores} == {0.99}


def test_purge_writes_an_audit_entry(db_session: Session, tmp_path: Path) -> None:
    _corpus(db_session)
    purge_asr_system(db_session, DOOMED, dump_dir=tmp_path, actor="owner")

    entry = db_session.scalars(
        sa.select(AuditLog).where(AuditLog.entity_type == "asr_system")
    ).all()[-1]
    assert entry.entity_id == DOOMED
    assert entry.action == "purge"
    assert entry.actor == "owner"
    assert entry.old_values_jsonb["hypotheses_deleted"] == 2


def test_purging_an_unknown_system_is_refused(db_session: Session, tmp_path: Path) -> None:
    _corpus(db_session)
    with pytest.raises(PurgedSystemNotFound, match="never-existed"):
        purge_asr_system(db_session, "never-existed", dump_dir=tmp_path)


def test_purge_can_be_previewed_without_deleting_anything(
    db_session: Session, tmp_path: Path
) -> None:
    _corpus(db_session)
    report = purge_asr_system(db_session, DOOMED, dump_dir=tmp_path, dry_run=True)

    assert report.hypotheses_deleted == 2
    assert report.dump_path is None
    assert db_session.scalar(sa.select(AsrSystem).where(AsrSystem.system_id == DOOMED)) is not None
    assert len(db_session.scalars(sa.select(AsrHypothesis)).all()) == 4


def test_the_rescore_holds_out_the_same_systems_the_ingest_does(
    db_session: Session, tmp_path: Path
) -> None:
    """A system excluded at ingest and counted here would rewrite every score a purge touched.

    Nothing is held out in the committed table today (D41 lifted D40's hold-out), so the hold-out
    is injected here: the mechanism has to keep working for the next route that needs it.
    """
    episode = Episode(external_id="ep_holdout", split="train")
    db_session.add(episode)
    db_session.flush()

    third = "mai-transcribe-2"
    systems = {}
    for name in (KEPT, DOOMED, third):
        system = AsrSystem(system_id=name, model_id=name)
        db_session.add(system)
        systems[name] = system
    db_session.flush()

    segment = Segment(
        episode_id=episode.id,
        external_id="seg_holdout",
        start_time=0.0,
        end_time=2.0,
        duration_seconds=2.0,
        clip_object_key="clips/seg_holdout.flac",
        clip_checksum="0" * 64,
    )
    db_session.add(segment)
    db_session.flush()
    db_session.add(
        SegmentScore(
            segment_id=segment.id, word_disagreement_rate=0.99, cer_between_hypotheses=0.99
        )
    )
    for name, text in (
        (KEPT, "आजको meeting मा"),
        (DOOMED, "आजको मिटिङ मा"),  # same words, one script -- the artefact a hold-out exists for
        (third, "आजको meeting मा"),
    ):
        db_session.add(
            AsrHypothesis(segment_id=segment.id, asr_system_id=systems[name].id, text_raw=text)
        )
    db_session.flush()

    table = load_llm_routes()
    held = table.routes["asr_gemini_composite"].model_copy(
        update={"system_id": DOOMED, "exclude_from_disagreement": True}
    )
    config = table.model_copy(update={"routes": {**table.routes, "asr_gemini_composite": held}})
    purge_asr_system(db_session, system_id=third, dump_dir=tmp_path, rescore=True, config=config)

    score = db_session.get(SegmentScore, segment.id)
    assert score is not None
    # Only KEPT is left in the comparison, so nothing disagrees. Counting the held-out
    # hypothesis would score its single-script spelling as disagreement instead.
    assert score.word_disagreement_rate == 0.0
