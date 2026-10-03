"""Reopening a labelled clip for a second listen (D114).

A reopened clip goes back to triage with its current label untouched. The new decision is one
more append-only row, and accepting the seed can never quietly overwrite an edited label.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import AnnotationEvent, AnnotationTask, AuditLog, Segment, SegmentLabel
from app.services.labeling import LabelingError, latest_label, reopen_for_relabel

pytestmark = pytest.mark.db

EDITED = "So today म Python मा loops बारे कुरा गर्छु। अनि ho ho"


def _decided(client: TestClient, db_session: Session, *, edit: bool) -> tuple[Segment, dict]:
    """A clip with one current label, accepted from the seed or edited."""
    task = client.get("/tasks/next").json()
    if edit:
        body = client.post(f"/tasks/{task['id']}/label", json={"final_text": EDITED}).json()
    else:
        body = client.post(f"/tasks/{task['id']}/accept", json={}).json()
    db_session.expire_all()
    return db_session.get(Segment, task["segment"]["id"]), body


def _reopen(db_session: Session, segment: Segment, **kwargs) -> AnnotationTask:
    kwargs = {"actor": "test", "priority": 0.8, "details": "3.1 s: ho ho", **kwargs}
    task = reopen_for_relabel(db_session, segment, **kwargs)
    db_session.flush()
    return task


def test_a_labelled_clip_is_reopened_into_triage_with_its_label_untouched(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    segment, decided = _decided(client, db_session, edit=True)
    labels_before = db_session.scalar(
        sa.select(sa.func.count())
        .select_from(SegmentLabel)
        .where(SegmentLabel.segment_id == segment.id)
    )

    task = _reopen(db_session, segment, note="D100 heavy-crosstalk check")

    assert task.queue == "review" and task.status == "pending"
    assert task.priority_score == 0.8
    assert task.reason_jsonb["hazards"] == ["relabel"]
    assert task.reason_jsonb["hazard_details"]["relabel"] == "3.1 s: ho ho"
    assert task.reason_jsonb["relabel"] == {
        "label_id": decided["label_id"],
        "disposition": "edited",
        "note": "D100 heavy-crosstalk check",
    }
    # The label is neither touched nor superseded, and the clip stays labelled.
    assert latest_label(db_session, segment.id).id == decided["label_id"]
    assert (
        db_session.scalar(
            sa.select(sa.func.count())
            .select_from(SegmentLabel)
            .where(SegmentLabel.segment_id == segment.id)
        )
        == labels_before
    )
    assert segment.pipeline_status == "labeled"
    event = db_session.scalars(
        sa.select(AnnotationEvent).where(AnnotationEvent.task_id == task.id)
    ).one()
    assert event.action == "reopen"
    audit = db_session.scalars(
        sa.select(AuditLog).where(
            AuditLog.entity_type == "annotation_tasks",
            AuditLog.entity_id == str(task.id),
        )
    ).one()
    assert audit.action == "reopen"
    assert audit.old_values_jsonb == {"label_id": decided["label_id"], "disposition": "edited"}


def test_the_reopened_task_keeps_the_seed_the_label_was_made_from(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    segment, decided = _decided(client, db_session, edit=False)
    label = db_session.get(SegmentLabel, decided["label_id"])
    task = _reopen(db_session, segment)
    assert task.seed_hypothesis_id == label.seed_hypothesis_id


def test_triage_lists_the_reopened_clip_with_its_suspects(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    segment, _ = _decided(client, db_session, edit=False)
    task = _reopen(db_session, segment)
    row = next(r for r in client.get("/queue").json() if r["task_id"] == task.id)
    assert row["reason"]["hazards"] == ["relabel"]
    assert row["reason"]["hazard_details"]["relabel"] == "3.1 s: ho ho"


def test_the_editor_payload_carries_the_label_to_start_from(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    segment, _ = _decided(client, db_session, edit=True)
    task = _reopen(db_session, segment)
    body = client.get(f"/tasks/{task.id}").json()
    assert body["reason"]["relabel"]["disposition"] == "edited"
    assert body["segment"]["latest_label"]["final_text"] == EDITED


def test_a_clip_without_a_label_cannot_be_reopened(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task = client.get("/tasks/next").json()
    segment = db_session.get(Segment, task["segment"]["id"])
    with pytest.raises(LabelingError, match="no label"):
        reopen_for_relabel(db_session, segment, actor="test")


def test_a_clip_with_an_open_task_is_not_reopened_twice(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    segment, _ = _decided(client, db_session, edit=False)
    _reopen(db_session, segment)
    with pytest.raises(LabelingError, match="open task"):
        reopen_for_relabel(db_session, segment, actor="test")


def test_a_flagged_clip_is_not_reopened(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task = client.get("/tasks/next").json()
    client.post(f"/tasks/{task['id']}/flag", json={"disposition": "unusable_audio"})
    db_session.expire_all()
    segment = db_session.get(Segment, task["segment"]["id"])
    with pytest.raises(LabelingError, match="unusable_audio"):
        reopen_for_relabel(db_session, segment, actor="test")


# --- deciding a reopened clip ------------------------------------------------------------------


def test_accepting_the_seed_over_an_edited_label_is_refused(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    segment, decided = _decided(client, db_session, edit=True)
    task = _reopen(db_session, segment)

    response = client.post(f"/tasks/{task.id}/accept", json={})
    assert response.status_code == 409
    assert "edited" in response.json()["detail"]
    db_session.expire_all()
    assert latest_label(db_session, segment.id).id == decided["label_id"]


def test_bulk_accept_cannot_revert_an_edited_label_either(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    segment, decided = _decided(client, db_session, edit=True)
    task = _reopen(db_session, segment)
    response = client.post("/tasks/bulk-accept", json={"task_ids": [task.id]})
    assert response.status_code == 409
    db_session.expire_all()
    assert latest_label(db_session, segment.id).id == decided["label_id"]


def test_a_reopened_edited_clip_is_relabelled_by_a_new_row(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    segment, decided = _decided(client, db_session, edit=True)
    task = _reopen(db_session, segment)
    added = EDITED + " हजुर"
    body = client.post(f"/tasks/{task.id}/label", json={"final_text": added}).json()
    db_session.expire_all()
    current = latest_label(db_session, segment.id)
    assert current.id == body["label_id"] != decided["label_id"]
    assert current.final_text == added and current.disposition == "edited"
    assert db_session.get(SegmentLabel, decided["label_id"]).final_text == EDITED


def test_a_reopened_seed_label_can_be_accepted_again(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    segment, decided = _decided(client, db_session, edit=False)
    task = _reopen(db_session, segment)
    response = client.post(f"/tasks/{task.id}/accept", json={})
    assert response.status_code == 200
    db_session.expire_all()
    current = latest_label(db_session, segment.id)
    assert current.id != decided["label_id"]
    assert current.final_text == db_session.get(SegmentLabel, decided["label_id"]).final_text
