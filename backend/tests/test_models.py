"""Schema tests: every core entity inserts and queries, and the database enforces its own rules."""

from __future__ import annotations

import datetime as dt

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    AnnotationEvent,
    AnnotationTask,
    AsrHypothesis,
    AsrSystem,
    AuditLog,
    Episode,
    HypothesisWord,
    ImportRun,
    LabelVersion,
    LlmRequest,
    Segment,
    SegmentLabel,
    SegmentScore,
    TranslitCacheEntry,
)

pytestmark = pytest.mark.db


def make_episode(session: Session, external_id: str = "show-a_ep012", **kwargs) -> Episode:
    episode = Episode(
        external_id=external_id,
        show_id="show-a",
        title="Example podcast",
        source_uri="https://example.test/ep012",
        published_at=dt.date(2026, 1, 1),
        source_audio_checksum="sha256:aaa",
        duration_seconds=4821.3,
        **kwargs,
    )
    session.add(episode)
    session.flush()
    return episode


def make_segment(session: Session, episode: Episode, index: int = 42, **kwargs) -> Segment:
    fields = {
        "episode_id": episode.id,
        "external_id": f"{episode.external_id}_{index:04d}",
        "speaker_id": "SPEAKER_01",
        "start_time": 123.4,
        "end_time": 135.2,
        "duration_seconds": 11.8,
        "clip_object_key": f"clips/{episode.external_id}/{index:04d}.flac",
        "clip_checksum": "sha256:bbb",
        "p_en": 0.31,
        "lid": "ne",
        **kwargs,
    }
    segment = Segment(**fields)
    session.add(segment)
    session.flush()
    return segment


def make_system(session: Session, system_id: str = "qwen-ne") -> AsrSystem:
    system = AsrSystem(system_id=system_id, model_id="sidskarki/Qwen3-ASR-Nepali")
    session.add(system)
    session.flush()
    return system


def make_hypothesis(
    session: Session, segment: Segment, system: AsrSystem, text: str = "So today म ..."
) -> AsrHypothesis:
    hypothesis = AsrHypothesis(
        segment_id=segment.id,
        asr_system_id=system.id,
        text_raw=text,
        text_normalized=text,
        avg_logprob=-0.34,
        no_speech_prob=0.01,
    )
    session.add(hypothesis)
    session.flush()
    return hypothesis


# --- basic round trips -------------------------------------------------------------------


def test_episode_round_trip(db_session: Session) -> None:
    episode = make_episode(db_session)
    fetched = db_session.get(Episode, episode.id)
    assert fetched is not None
    assert fetched.external_id == "show-a_ep012"
    assert fetched.split == "unassigned"
    assert fetched.created_at.tzinfo is not None


def test_segment_round_trip_and_relationship(db_session: Session) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    assert segment.pipeline_status == "imported"
    assert db_session.get(Segment, segment.id).episode.external_id == episode.external_id
    assert episode.segments[0].id == segment.id


def test_hypothesis_and_words_round_trip(db_session: Session) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    system = make_system(db_session)
    hypothesis = make_hypothesis(db_session, segment, system)
    db_session.add(
        HypothesisWord(
            hypothesis_id=hypothesis.id,
            position=0,
            word_raw="So",
            start_time=123.4,
            end_time=123.7,
            confidence=0.92,
            predicted_language="en",
            predicted_script="latin",
        )
    )
    db_session.flush()
    words = db_session.scalars(
        sa.select(HypothesisWord).where(HypothesisWord.hypothesis_id == hypothesis.id)
    ).all()
    assert [w.word_raw for w in words] == ["So"]
    assert hypothesis.system.system_id == "qwen-ne"


def test_segment_scores_round_trip(db_session: Session) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    db_session.add(
        SegmentScore(
            segment_id=segment.id,
            cer_between_hypotheses=0.18,
            word_disagreement_rate=0.22,
            script_conflict_rate=0.05,
            code_switch_density=0.42,
            flags_jsonb=["low_confidence"],
        )
    )
    db_session.flush()
    score = db_session.get(SegmentScore, segment.id)
    assert score.flags_jsonb == ["low_confidence"]
    assert score.word_disagreement_rate == pytest.approx(0.22)


def test_import_run_round_trip(db_session: Session) -> None:
    run = ImportRun(source_path="/data/export_show-a_ep012", pipeline_version="nb-v3")
    db_session.add(run)
    db_session.flush()
    assert run.status == "running"
    assert run.segments_inserted == 0


def test_annotation_task_round_trip(db_session: Session) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    system = make_system(db_session)
    hypothesis = make_hypothesis(db_session, segment, system)
    task = AnnotationTask(
        segment_id=segment.id,
        queue="review",
        priority_score=0.72,
        seed_hypothesis_id=hypothesis.id,
        reason_jsonb={"word_disagreement_rate": 0.22},
    )
    db_session.add(task)
    db_session.flush()
    assert task.status == "pending"
    assert task.seed_hypothesis.id == hypothesis.id


def test_label_round_trip_and_append_only_history(db_session: Session) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    version = LabelVersion(name="v1", policy_version="policy_v1")
    db_session.add(version)
    db_session.flush()
    for i, disposition in enumerate(["accepted_unchanged", "edited"]):
        db_session.add(
            SegmentLabel(
                segment_id=segment.id,
                label_version_id=version.id,
                final_text=f"text {i}",
                disposition=disposition,
                annotator="owner",
            )
        )
        db_session.flush()
    rows = db_session.scalars(
        sa.select(SegmentLabel)
        .where(SegmentLabel.segment_id == segment.id)
        .order_by(SegmentLabel.created_at.desc(), SegmentLabel.id.desc())
    ).all()
    assert len(rows) == 2
    assert rows[0].disposition == "edited"


def test_annotation_event_round_trip(db_session: Session) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    task = AnnotationTask(segment_id=segment.id, queue="review", priority_score=0.1)
    db_session.add(task)
    db_session.flush()
    now = dt.datetime.now(dt.UTC)
    db_session.add(
        AnnotationEvent(
            task_id=task.id,
            segment_id=segment.id,
            annotator="owner",
            opened_at=now,
            submitted_at=now + dt.timedelta(milliseconds=2400),
            duration_ms=2400,
            action="accept",
        )
    )
    db_session.flush()
    event = db_session.scalars(sa.select(AnnotationEvent)).one()
    assert event.duration_ms == 2400


def test_audit_log_round_trip(db_session: Session) -> None:
    db_session.add(
        AuditLog(
            entity_type="segment_labels",
            entity_id="1",
            action="insert",
            actor="owner",
            old_values_jsonb=None,
            new_values_jsonb={"final_text": "x"},
        )
    )
    db_session.flush()
    assert db_session.scalars(sa.select(AuditLog)).one().action == "insert"


def test_translit_cache_round_trip(db_session: Session) -> None:
    db_session.add(
        TranslitCacheEntry(latin_token="kura", candidates_jsonb=["कुरा", "कुरा"], provider="remote")
    )
    db_session.flush()
    entry = db_session.get(TranslitCacheEntry, "kura")
    assert entry.hit_count == 0
    assert entry.candidates_jsonb[0] == "कुरा"


def test_llm_request_round_trip(db_session: Session) -> None:
    db_session.add(
        LlmRequest(
            route="unused",
            model="anthropic/claude-sonnet-4",
            request_hash="deadbeef",
            status="dry_run",
        )
    )
    db_session.flush()
    assert db_session.scalars(sa.select(LlmRequest)).one().route == "unused"


# --- database-enforced rules -------------------------------------------------------------


def test_second_active_task_for_a_segment_is_rejected_by_the_database(
    db_session: Session,
) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    db_session.add(AnnotationTask(segment_id=segment.id, queue="review", priority_score=0.5))
    db_session.flush()
    db_session.add(
        AnnotationTask(
            segment_id=segment.id, queue="review", priority_score=0.5, status="in_progress"
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_a_finished_task_does_not_block_a_new_one(db_session: Session) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    db_session.add(
        AnnotationTask(segment_id=segment.id, queue="review", priority_score=0.5, status="done")
    )
    db_session.flush()
    db_session.add(AnnotationTask(segment_id=segment.id, queue="review", priority_score=0.5))
    db_session.flush()  # must not raise


def test_orphan_segment_is_rejected(db_session: Session) -> None:
    db_session.add(
        Segment(
            episode_id=999999,
            external_id="orphan_0001",
            start_time=0.0,
            end_time=1.0,
            duration_seconds=1.0,
            clip_object_key="clips/x.flac",
            clip_checksum="sha256:x",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_orphan_hypothesis_is_rejected(db_session: Session) -> None:
    db_session.add(AsrHypothesis(segment_id=999999, asr_system_id=999999, text_raw="x"))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_duplicate_hypothesis_for_same_segment_and_system_is_rejected(
    db_session: Session,
) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    system = make_system(db_session)
    make_hypothesis(db_session, segment, system)
    with pytest.raises(IntegrityError):
        make_hypothesis(db_session, segment, system, text="different text")


def test_duplicate_segment_external_id_is_rejected(db_session: Session) -> None:
    episode = make_episode(db_session)
    make_segment(db_session, episode)
    with pytest.raises(IntegrityError):
        make_segment(db_session, episode)


def test_invalid_split_value_is_rejected(db_session: Session) -> None:
    with pytest.raises(IntegrityError):
        make_episode(db_session, external_id="bad-split", split="holdout")


def test_invalid_disposition_is_rejected(db_session: Session) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    version = LabelVersion(name="v1", policy_version="policy_v1")
    db_session.add(version)
    db_session.flush()
    db_session.add(
        SegmentLabel(
            segment_id=segment.id,
            label_version_id=version.id,
            final_text="x",
            disposition="probably_fine",
            annotator="owner",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_invalid_pipeline_status_is_rejected(db_session: Session) -> None:
    episode = make_episode(db_session)
    with pytest.raises(IntegrityError):
        make_segment(db_session, episode, pipeline_status="in_flight")


def test_invalid_queue_is_rejected(db_session: Session) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    db_session.add(AnnotationTask(segment_id=segment.id, queue="triage", priority_score=0.1))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_segment_end_must_follow_start(db_session: Session) -> None:
    episode = make_episode(db_session)
    with pytest.raises(IntegrityError):
        make_segment(db_session, episode, start_time=10.0, end_time=5.0)


def test_deleting_a_hypothesis_cascades_to_its_words(db_session: Session) -> None:
    episode = make_episode(db_session)
    segment = make_segment(db_session, episode)
    system = make_system(db_session)
    hypothesis = make_hypothesis(db_session, segment, system)
    db_session.add(HypothesisWord(hypothesis_id=hypothesis.id, position=0, word_raw="So"))
    db_session.flush()
    db_session.execute(sa.delete(AsrHypothesis).where(AsrHypothesis.id == hypothesis.id))
    db_session.flush()
    assert db_session.scalars(sa.select(HypothesisWord)).all() == []


# --- indexes required for the hot paths --------------------------------------------------


@pytest.mark.parametrize(
    "index_name",
    [
        "uq_annotation_tasks_active_segment",
        "ix_annotation_tasks_queue_status_priority_score",
        "ix_segments_episode_id",
        "ix_segment_labels_segment_id_label_version_id",
        "ix_episodes_split",
    ],
)
def test_required_index_exists(db_session: Session, index_name: str) -> None:
    found = db_session.execute(
        sa.text("SELECT 1 FROM pg_indexes WHERE indexname = :n"), {"n": index_name}
    ).scalar()
    assert found == 1, f"missing index {index_name}"
