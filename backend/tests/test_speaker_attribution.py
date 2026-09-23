"""Tests for the speakers queue and per-speaker labels against the database (D98)."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import (
    AnnotationEvent,
    AnnotationTask,
    AuditLog,
    DiarizationRun,
    Episode,
    LabelWord,
    Segment,
    SegmentLabel,
)
from app.services.diarization_import import import_diarization
from app.services.labeling import Decision, LabelingError, latest_label, record_decision
from app.services.speaker_attribution import queue_for_speakers, rank_for_speakers
from app.services.stats import collect_stats

pytestmark = pytest.mark.db

MODEL = "pyannote/speaker-diarization-community-1"


def _segments(session: Session) -> list[Segment]:
    return list(session.scalars(sa.select(Segment).order_by(Segment.id)))


@pytest.fixture
def labelled(client, imported_episode: str, db_session: Session) -> list[Segment]:
    """Every clip verified as one stream, in gold, with two diarized voices crossing mid-clip."""
    for row in client.get("/queue", params={"limit": 50}).json():
        assert client.post(f"/tasks/{row['task_id']}/accept", json={}).status_code == 200
    segments = _segments(db_session)
    # The shared fixture writes episode-relative word spans; ingest stores them clip-relative
    # (D26), which is what the lanes are drawn on.
    for segment in segments:
        for hypothesis in segment.hypotheses:
            for w in hypothesis.words:
                if w.start_time is not None:
                    w.start_time = round(w.start_time - segment.start_time, 3)
                    w.end_time = round(w.end_time - segment.start_time, 3)
    turns = []
    for segment in segments:
        segment.pot = "gold"
        mid = (segment.start_time + segment.end_time) / 2
        turns.append([segment.start_time, mid + 0.3, "SPEAKER_00"])
        turns.append([mid - 0.3, segment.end_time, "SPEAKER_01"])
    # Crosstalk shares 0.0, 0.1, ... so the ranking has an order to find.
    for index, segment in enumerate(segments):
        segment.overlap_spans_jsonb = [[0.0, round(segment.duration_seconds * index / 10, 3)]]
    import_diarization(
        db_session,
        {imported_episode: {"turns": turns}},
        model=MODEL,
        source="test.json",
        actor="test",
    )
    db_session.flush()
    return segments


def _queue(db_session: Session, limit: int = 30) -> list[AnnotationTask]:
    queue_for_speakers(db_session, rank_for_speakers(db_session)[:limit], actor="test")
    return list(
        db_session.scalars(
            sa.select(AnnotationTask)
            .where(AnnotationTask.queue == "speakers")
            .order_by(AnnotationTask.priority_score.desc())
        )
    )


# --- which clips are queued -------------------------------------------------------------------


def test_the_most_crosstalk_comes_first(db_session: Session, labelled: list[Segment]) -> None:
    ranked = rank_for_speakers(db_session)
    assert [c.segment.id for c in ranked] == [s.id for s in reversed(labelled)]
    assert all(c.clip_speakers == 2 for c in ranked)


def test_only_the_asked_pot_is_ranked(db_session: Session, labelled: list[Segment]) -> None:
    labelled[-1].pot = "train"
    db_session.flush()
    assert labelled[-1].id not in {c.segment.id for c in rank_for_speakers(db_session)}


def test_a_clip_with_one_voice_is_not_ranked(db_session: Session, labelled: list[Segment]) -> None:
    run = db_session.scalars(sa.select(DiarizationRun)).one()
    first = labelled[0]
    for turn in run.turns:
        if turn.start_time < first.end_time and turn.end_time > first.start_time:
            turn.speaker = "SPEAKER_00"
    db_session.flush()
    assert first.id not in {c.segment.id for c in rank_for_speakers(db_session)}


def test_an_unlabelled_clip_is_not_ranked(db_session: Session, labelled: list[Segment]) -> None:
    label = latest_label(db_session, labelled[-1].id)
    label.disposition = "uncertain"
    db_session.flush()
    assert labelled[-1].id not in {c.segment.id for c in rank_for_speakers(db_session)}


def test_a_crosstalk_band_keeps_only_clips_inside_it(db_session: Session, labelled) -> None:
    # The fixture gives the clips crosstalk shares 0.0, 0.1, ..., 0.5.
    ranked = rank_for_speakers(db_session, min_overlap=0.1, max_overlap=0.3)
    shares = sorted(round(c.overlap_share, 2) for c in ranked)
    assert shares == [0.1, 0.2, 0.3]


def test_an_unmeasured_clip_is_outside_every_band(db_session: Session, labelled) -> None:
    labelled[2].overlap_spans_jsonb = None
    db_session.flush()
    ranked = rank_for_speakers(db_session, min_overlap=0.0, max_overlap=1.0)
    assert labelled[2].id not in {c.segment.id for c in ranked}


def test_a_one_voice_clip_can_be_asked_for(db_session: Session, labelled) -> None:
    """Crosstalk the diarizer heard as one voice is exactly what a diarizer check looks for."""
    run = db_session.scalars(sa.select(DiarizationRun)).one()
    first = labelled[1]
    for turn in run.turns:
        if turn.start_time < first.end_time and turn.end_time > first.start_time:
            turn.speaker = "SPEAKER_00"
    db_session.flush()
    ranked = rank_for_speakers(db_session, min_clip_speakers=1, min_overlap=1e-6)
    by_id = {c.segment.id: c for c in ranked}
    assert by_id[first.id].clip_speakers == 1
    assert labelled[0].id not in by_id  # no crosstalk at all


def test_queueing_leaves_the_clip_as_it_was(db_session: Session, labelled: list[Segment]) -> None:
    before = {s.id: (s.pot, s.pipeline_status, latest_label(db_session, s.id).id) for s in labelled}
    tasks = _queue(db_session, limit=3)
    assert len(tasks) == 3
    assert [t.segment_id for t in tasks] == [s.id for s in reversed(labelled)][:3]
    for segment in labelled:
        db_session.refresh(segment)
        after = (segment.pot, segment.pipeline_status, latest_label(db_session, segment.id).id)
        assert after == before[segment.id]
    audits = db_session.scalars(
        sa.select(AuditLog).where(AuditLog.action == "queue_speakers")
    ).all()
    assert len(audits) == 3


def test_a_queued_clip_is_not_queued_twice(db_session: Session, labelled: list[Segment]) -> None:
    _queue(db_session, limit=2)
    assert len(_queue(db_session, limit=2)) == 4


# --- serving lanes ----------------------------------------------------------------------------


def test_a_speakers_task_opens_with_lanes(client, db_session: Session, labelled) -> None:
    task = _queue(db_session, limit=1)[0]
    body = client.get(f"/tasks/{task.id}").json()
    lanes = body["lanes"]
    assert lanes["base"] == "label"
    assert sorted(s["label"] for s in lanes["speakers"]) == ["SPEAKER_00", "SPEAKER_01"]
    assert [s["number"] for s in lanes["speakers"]] == [1, 2]
    assert lanes["clip_speakers"] == [1, 2]
    label = latest_label(db_session, task.segment_id)
    assert " ".join(w["word"] for w in lanes["words"]) == label.final_text
    assert {w["speaker"] for w in lanes["words"]} <= {1, 2, None}


def test_a_review_task_has_no_lanes(client, imported_episode: str) -> None:
    task_id = client.get("/queue").json()[0]["task_id"]
    assert client.get(f"/tasks/{task_id}").json()["lanes"] is None


def test_the_speakers_queue_is_served_by_next(client, db_session: Session, labelled) -> None:
    tasks = _queue(db_session, limit=2)
    body = client.get("/tasks/next", params={"queue": "speakers"}).json()
    assert body["id"] == tasks[0].id and body["lanes"] is not None
    rows = client.get("/queue", params={"queue": "speakers"}).json()
    assert [r["task_id"] for r in rows] == [t.id for t in tasks]


# --- saving lanes -----------------------------------------------------------------------------


def _placed(lanes: dict) -> list[dict]:
    """The served words with every one on a lane and a span, as the editor would submit them."""
    words = []
    for w in lanes["words"]:
        if w["start"] is None:
            continue
        words.append({**w, "speaker": w["speaker"] or 1})
    return words


def _open(client, db_session: Session) -> tuple[AnnotationTask, dict]:
    task = _queue(db_session, limit=1)[0]
    return task, client.get(f"/tasks/{task.id}").json()["lanes"]


def test_saving_the_lanes_as_served_accepts_them(client, db_session: Session, labelled) -> None:
    task, lanes = _open(client, db_session)
    assert all(w["speaker"] is not None for w in lanes["words"])
    single_stream = latest_label(db_session, task.segment_id)
    labels_before = collect_stats(db_session)["labels"]["total"]

    response = client.post(
        f"/tasks/{task.id}/attribute",
        json={
            "diarization_run_id": lanes["diarization_run_id"],
            "words": lanes["words"],
            "duration_ms": 4200,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["disposition"] == "accepted_unchanged"

    label = db_session.get(SegmentLabel, response.json()["label_id"])
    assert label.label_version.name == "speakers-v1"
    assert label.diarization_run_id == lanes["diarization_run_id"]
    assert label.verification_tier == "verified"
    assert [w.word for w in label.words] == [w["word"] for w in lanes["words"]]
    assert {w.speaker for w in label.words} <= {"SPEAKER_00", "SPEAKER_01"}

    # The single-stream label is still the clip's label, and still counted once.
    assert latest_label(db_session, task.segment_id).id == single_stream.id
    assert collect_stats(db_session)["labels"]["total"] == labels_before
    segment = db_session.get(Segment, task.segment_id)
    assert segment.pipeline_status == "labeled"
    assert db_session.get(AnnotationTask, task.id).status == "done"
    assert (
        db_session.scalars(sa.select(AnnotationEvent).where(AnnotationEvent.task_id == task.id))
        .one()
        .duration_ms
        == 4200
    )
    assert db_session.scalars(
        sa.select(AuditLog).where(
            AuditLog.entity_type == "segment_labels", AuditLog.entity_id == str(label.id)
        )
    ).one().new_values_jsonb["words"] == len(lanes["words"])


def test_moving_and_adding_words_is_an_edit(client, db_session: Session, labelled) -> None:
    task, lanes = _open(client, db_session)
    words = _placed(lanes)
    moved = {**words[0], "speaker": 2 if words[0]["speaker"] == 1 else 1}
    added = {
        "word": "हजुर",
        "start": 0.5,
        "end": 0.8,
        "speaker": 2,
        "proposed_speaker": None,
        "source": "typed",
    }
    response = client.post(
        f"/tasks/{task.id}/attribute",
        json={
            "diarization_run_id": lanes["diarization_run_id"],
            "words": [moved, *words[1:], added],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["disposition"] == "edited"
    label = db_session.get(SegmentLabel, response.json()["label_id"])
    audit = db_session.scalars(
        sa.select(AuditLog).where(
            AuditLog.entity_type == "segment_labels", AuditLog.entity_id == str(label.id)
        )
    ).one()
    assert audit.new_values_jsonb["moved"] == 1
    assert audit.new_values_jsonb["added"] == 1
    assert "हजुर" in label.final_text.split()
    typed = db_session.scalars(
        sa.select(LabelWord).where(LabelWord.label_id == label.id, LabelWord.source == "typed")
    ).one()
    assert (typed.start_time, typed.end_time, typed.proposed_speaker) == (0.5, 0.8, None)


def test_a_reopened_clip_shows_what_was_saved(client, db_session: Session, labelled) -> None:
    task, lanes = _open(client, db_session)
    words = _placed(lanes)
    words[0] = {**words[0], "speaker": 2 if words[0]["speaker"] == 1 else 1}
    client.post(
        f"/tasks/{task.id}/attribute",
        json={"diarization_run_id": lanes["diarization_run_id"], "words": words},
    )
    requeued = AnnotationTask(
        segment_id=task.segment_id, queue="speakers", priority_score=0.0, status="pending"
    )
    db_session.add(requeued)
    db_session.flush()
    reopened = client.get(f"/tasks/{requeued.id}").json()["lanes"]
    assert reopened["base"] == "speakers"
    assert reopened["words"][0]["speaker"] == words[0]["speaker"]


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"speaker": 7}, "not one of the episode's"),
        ({"speaker": None}, "on no lane"),
        ({"start": None, "end": None}, "no span"),
        ({"end": 999.0}, "outside the clip"),
    ],
)
def test_a_bad_word_is_refused(
    client, db_session: Session, labelled, change: dict, fragment: str
) -> None:
    task, lanes = _open(client, db_session)
    words = _placed(lanes)
    words[0] = {**words[0], **change}
    response = client.post(
        f"/tasks/{task.id}/attribute",
        json={"diarization_run_id": lanes["diarization_run_id"], "words": words},
    )
    assert response.status_code == 409
    assert fragment in response.json()["detail"]
    assert db_session.get(AnnotationTask, task.id).status != "done"


def test_a_stale_diarization_run_is_refused(client, db_session: Session, labelled) -> None:
    task, lanes = _open(client, db_session)
    response = client.post(
        f"/tasks/{task.id}/attribute",
        json={"diarization_run_id": lanes["diarization_run_id"] + 1, "words": _placed(lanes)},
    )
    assert response.status_code == 409
    assert "re-diarized" in response.json()["detail"]


def test_a_review_task_takes_no_lanes(client, imported_episode: str, db_session: Session) -> None:
    task_id = client.get("/queue").json()[0]["task_id"]
    response = client.post(
        f"/tasks/{task_id}/attribute",
        json={
            "diarization_run_id": 1,
            "words": [{"word": "a", "start": 0.1, "end": 0.3, "speaker": 1}],
        },
    )
    assert response.status_code == 409
    assert "not in the speakers queue" in response.json()["detail"]


def test_a_speakers_task_cannot_be_accepted_as_text(client, db_session: Session, labelled) -> None:
    task = _queue(db_session, limit=1)[0]
    response = client.post(f"/tasks/{task.id}/accept", json={})
    assert response.status_code == 409
    assert "multitrack editor" in response.json()["detail"]


def test_a_speakers_task_cannot_be_screened(db_session: Session, labelled) -> None:
    task = _queue(db_session, limit=1)[0]
    labelled_clip = db_session.get(Segment, task.segment_id)
    labelled_clip.pot = "train"
    db_session.flush()
    with pytest.raises(LabelingError, match="has to be listened to"):
        record_decision(
            db_session, task, Decision(disposition="uncertain", verification_tier="screened")
        )


def test_flagging_a_speakers_task_keeps_the_clip_labelled(
    client, db_session: Session, labelled
) -> None:
    task = _queue(db_session, limit=1)[0]
    single_stream = latest_label(db_session, task.segment_id)
    response = client.post(
        f"/tasks/{task.id}/flag",
        json={"disposition": "uncertain", "notes": "diarizer merged two voices"},
    )
    assert response.status_code == 200
    label = db_session.get(SegmentLabel, response.json()["label_id"])
    assert label.label_version.name == "speakers-v1"
    assert label.words == []
    assert db_session.get(Segment, task.segment_id).pipeline_status == "labeled"
    assert latest_label(db_session, task.segment_id).id == single_stream.id


def test_the_speakers_queue_is_counted(client, db_session: Session, labelled) -> None:
    _queue(db_session, limit=4)
    assert collect_stats(db_session)["queues"]["speakers"] == 4


def test_the_single_stream_export_ignores_per_speaker_labels(
    client, db_session: Session, labelled
) -> None:
    episode = db_session.scalars(sa.select(Episode)).one()
    task, lanes = _open(client, db_session)
    client.post(
        f"/tasks/{task.id}/attribute",
        json={"diarization_run_id": lanes["diarization_run_id"], "words": _placed(lanes)},
    )
    version_names = {
        label.label_version.name
        for label in db_session.scalars(
            sa.select(SegmentLabel)
            .join(Segment, Segment.id == SegmentLabel.segment_id)
            .where(Segment.episode_id == episode.id)
        )
    }
    assert version_names == {"v1", "speakers-v1"}
    stats = collect_stats(db_session)
    assert stats["labels"]["total"] == len(labelled)


# --- reopening a saved clip -------------------------------------------------------------------


def test_a_saved_clip_can_be_reopened_from_its_saved_lanes(client, db_session, labelled) -> None:
    from app.services.speaker_attribution import reopen_for_speakers

    task, lanes = _open(client, db_session)
    words = _placed(lanes)
    words[0] = {**words[0], "speaker": 2 if words[0]["speaker"] == 1 else 1}
    saved = client.post(
        f"/tasks/{task.id}/attribute",
        json={"diarization_run_id": lanes["diarization_run_id"], "words": words},
    ).json()
    segment = db_session.get(Segment, task.segment_id)

    reopened = reopen_for_speakers(db_session, segment, actor="test")
    assert reopened.queue == "speakers" and reopened.status == "pending"
    assert reopened.reason_jsonb["speakers"]["reopened_label_id"] == saved["label_id"]
    event = db_session.scalars(
        sa.select(AnnotationEvent).where(AnnotationEvent.task_id == reopened.id)
    ).one()
    assert event.action == "reopen"
    assert db_session.scalars(
        sa.select(AuditLog).where(
            AuditLog.entity_type == "annotation_tasks", AuditLog.action == "reopen"
        )
    ).one()
    # The saved label is untouched and still current; the editor starts from it.
    body = client.get(f"/tasks/{reopened.id}").json()["lanes"]
    assert body["base"] == "speakers"
    assert body["words"][0]["speaker"] == words[0]["speaker"]


def test_a_clip_never_saved_cannot_be_reopened(db_session, labelled) -> None:
    from app.services.labeling import LabelingError
    from app.services.speaker_attribution import reopen_for_speakers

    with pytest.raises(LabelingError, match="no per-speaker label"):
        reopen_for_speakers(db_session, labelled[0], actor="test")


def test_a_clip_already_open_is_not_reopened_twice(client, db_session, labelled) -> None:
    from app.services.labeling import LabelingError
    from app.services.speaker_attribution import reopen_for_speakers

    task, lanes = _open(client, db_session)
    client.post(
        f"/tasks/{task.id}/attribute",
        json={"diarization_run_id": lanes["diarization_run_id"], "words": _placed(lanes)},
    )
    segment = db_session.get(Segment, task.segment_id)
    reopen_for_speakers(db_session, segment, actor="test")
    with pytest.raises(LabelingError, match="already has an open task"):
        reopen_for_speakers(db_session, segment, actor="test")
