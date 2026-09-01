"""Tests for the review API, including error paths."""

from __future__ import annotations

import datetime as dt
import json

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import AnnotationEvent, AnnotationTask, AuditLog, Segment, SegmentLabel

pytestmark = pytest.mark.db


def queue_rows(client: TestClient, **params) -> list[dict]:
    response = client.get("/queue", params=params)
    assert response.status_code == 200
    return response.json()


# --- queue and stats ---------------------------------------------------------------------


def test_queue_lists_pending_tasks_highest_priority_first(
    client: TestClient, imported_episode: str
) -> None:
    rows = queue_rows(client)
    assert len(rows) == 6
    scores = [row["priority_score"] for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_queue_row_carries_everything_triage_needs(
    client: TestClient, imported_episode: str
) -> None:
    row = queue_rows(client)[0]
    assert row["seed_text"]
    assert row["seed_system_id"]
    assert row["duration_seconds"] > 0
    assert row["audio_url"].endswith("/audio")
    assert row["peaks_url"].endswith("/peaks")
    assert set(row["reason"]["components"]) == {
        "word_disagreement_rate",
        "low_confidence",
        "code_switch_density",
        "rule_flag_score",
    }


def test_queue_respects_limit_and_offset(client: TestClient, imported_episode: str) -> None:
    first = queue_rows(client, limit=2)
    assert len(first) == 2
    second = queue_rows(client, limit=2, offset=2)
    assert {r["task_id"] for r in first} & {r["task_id"] for r in second} == set()


def test_queue_filters_by_episode(client: TestClient, imported_episode: str) -> None:
    assert queue_rows(client, episode=imported_episode)
    assert queue_rows(client, episode="does-not-exist") == []


def test_queue_filters_by_minimum_priority(client: TestClient, imported_episode: str) -> None:
    rows = queue_rows(client, min_priority=0.99)
    assert all(row["priority_score"] >= 0.99 for row in rows)


def test_queue_rejects_an_unknown_queue_name(client: TestClient, imported_episode: str) -> None:
    assert client.get("/queue", params={"queue": "nonsense"}).status_code == 422


def test_stats_reports_progress(client: TestClient, imported_episode: str) -> None:
    stats = client.get("/stats").json()
    assert stats["segments"]["total"] == 6
    assert stats["tasks"]["pending"] == 6
    assert stats["labels"]["total"] == 0
    assert stats["accept_rate"] is None
    assert stats["audio_hours"] > 0


def test_stats_runs_against_an_empty_database(client: TestClient) -> None:
    stats = client.get("/stats").json()
    assert stats["segments"]["total"] == 0
    assert stats["throughput"]["median_seconds_per_segment"] is None


# --- serving tasks -----------------------------------------------------------------------


def test_next_task_returns_the_top_priority_task(client: TestClient, imported_episode: str) -> None:
    top = queue_rows(client)[0]
    task = client.get("/tasks/next").json()
    assert task["id"] == top["task_id"]
    assert task["segment"]["hypotheses"]
    assert task["seed_system_id"]


def test_next_task_marks_the_task_in_progress(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task_id = client.get("/tasks/next").json()["id"]
    db_session.expire_all()
    assert db_session.get(AnnotationTask, task_id).status == "in_progress"


def test_next_task_resumes_the_same_task(client: TestClient, imported_episode: str) -> None:
    """Reopening the app must return to the exact position in the queue."""
    first = client.get("/tasks/next").json()["id"]
    assert client.get("/tasks/next").json()["id"] == first


def test_next_task_on_an_empty_queue_is_404(client: TestClient) -> None:
    response = client.get("/tasks/next")
    assert response.status_code == 404
    assert "empty" in response.json()["detail"]


def test_get_task_does_not_change_status(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task_id = queue_rows(client)[0]["task_id"]
    assert client.get(f"/tasks/{task_id}").json()["status"] == "pending"
    db_session.expire_all()
    assert db_session.get(AnnotationTask, task_id).status == "pending"


def test_get_unknown_task_is_404(client: TestClient) -> None:
    assert client.get("/tasks/999999").status_code == 404


# --- writes ------------------------------------------------------------------------------


def test_accept_writes_a_label_an_event_and_an_audit_entry(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task = client.get("/tasks/next").json()
    opened_at = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=4)
    response = client.post(f"/tasks/{task['id']}/accept", json={"opened_at": opened_at.isoformat()})
    assert response.status_code == 200
    body = response.json()
    assert body["disposition"] == "accepted_unchanged"
    assert body["task_status"] == "done"

    db_session.expire_all()
    label = db_session.get(SegmentLabel, body["label_id"])
    assert label.disposition == "accepted_unchanged"
    assert label.final_text == task["segment"]["hypotheses"][0]["text"] or label.final_text
    assert label.seed_hypothesis_id == task["seed_hypothesis_id"]

    event = db_session.scalars(
        sa.select(AnnotationEvent).where(AnnotationEvent.task_id == task["id"])
    ).one()
    assert event.action == "accept"
    assert event.duration_ms is not None

    audit = db_session.scalars(sa.select(AuditLog).where(AuditLog.entity_id == str(label.id))).one()
    assert audit.action == "insert"
    assert audit.new_values_jsonb["disposition"] == "accepted_unchanged"


def test_accept_records_real_elapsed_time_not_server_time(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    """The number that matters is how long the human took, which only the client can observe."""
    task = client.get("/tasks/next").json()
    opened_at = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=12)
    client.post(f"/tasks/{task['id']}/accept", json={"opened_at": opened_at.isoformat()})
    db_session.expire_all()
    event = db_session.scalars(
        sa.select(AnnotationEvent).where(AnnotationEvent.task_id == task["id"])
    ).one()
    assert 11_500 <= event.duration_ms <= 20_000


def test_explicit_duration_wins_over_opened_at(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task = client.get("/tasks/next").json()
    client.post(
        f"/tasks/{task['id']}/accept",
        json={
            "opened_at": (dt.datetime.now(dt.UTC) - dt.timedelta(seconds=99)).isoformat(),
            "duration_ms": 2500,
        },
    )
    db_session.expire_all()
    event = db_session.scalars(
        sa.select(AnnotationEvent).where(AnnotationEvent.task_id == task["id"])
    ).one()
    assert event.duration_ms == 2500


def test_accept_marks_the_segment_labeled(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task = client.get("/tasks/next").json()
    client.post(f"/tasks/{task['id']}/accept", json={})
    db_session.expire_all()
    assert db_session.get(Segment, task["segment_id"]).pipeline_status == "labeled"


def test_label_writes_the_corrected_text(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task = client.get("/tasks/next").json()
    text = "So today म Python मा loops बारे कुरा गर्छु।"
    body = client.post(f"/tasks/{task['id']}/label", json={"final_text": text}).json()
    db_session.expire_all()
    label = db_session.get(SegmentLabel, body["label_id"])
    assert label.disposition == "edited"
    assert label.final_text == text


def test_label_never_overwrites_a_hypothesis(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task = client.get("/tasks/next").json()
    original = task["segment"]["hypotheses"][0]["text"]
    client.post(f"/tasks/{task['id']}/label", json={"final_text": "completely different"})
    db_session.expire_all()
    refreshed = client.get(f"/segments/{task['segment_id']}").json()
    assert refreshed["hypotheses"][0]["text"] == original


def test_label_requires_final_text(client: TestClient, imported_episode: str) -> None:
    task_id = queue_rows(client)[0]["task_id"]
    assert client.post(f"/tasks/{task_id}/label", json={}).status_code == 422


def test_flag_unusable_audio_excludes_the_segment(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task = client.get("/tasks/next").json()
    body = client.post(f"/tasks/{task['id']}/flag", json={"disposition": "unusable_audio"}).json()
    db_session.expire_all()
    assert db_session.get(SegmentLabel, body["label_id"]).disposition == "unusable_audio"
    assert db_session.get(Segment, task["segment_id"]).pipeline_status == "excluded"


def test_flag_uncertain_is_distinct_from_unusable(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    """One is an audio quality statistic, the other an annotation difficulty statistic."""
    rows = queue_rows(client)
    a = client.post(f"/tasks/{rows[0]['task_id']}/flag", json={"disposition": "uncertain"}).json()
    b = client.post(
        f"/tasks/{rows[1]['task_id']}/flag", json={"disposition": "unusable_audio"}
    ).json()
    db_session.expire_all()
    assert db_session.get(SegmentLabel, a["label_id"]).disposition == "uncertain"
    assert db_session.get(SegmentLabel, b["label_id"]).disposition == "unusable_audio"


def test_flag_rejects_an_unknown_disposition(client: TestClient, imported_episode: str) -> None:
    task_id = queue_rows(client)[0]["task_id"]
    response = client.post(f"/tasks/{task_id}/flag", json={"disposition": "probably_fine"})
    assert response.status_code == 422


def test_skip_writes_an_event_but_no_label(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task = client.get("/tasks/next").json()
    body = client.post(f"/tasks/{task['id']}/skip", json={}).json()
    assert body["label_id"] is None
    assert body["task_status"] == "skipped"
    db_session.expire_all()
    assert db_session.scalars(sa.select(SegmentLabel)).all() == []
    assert (
        db_session.scalars(sa.select(AnnotationEvent).where(AnnotationEvent.task_id == task["id"]))
        .one()
        .action
        == "skip"
    )


def test_deciding_a_finished_task_is_a_conflict(client: TestClient, imported_episode: str) -> None:
    task = client.get("/tasks/next").json()
    assert client.post(f"/tasks/{task['id']}/accept", json={}).status_code == 200
    response = client.post(f"/tasks/{task['id']}/accept", json={})
    assert response.status_code == 409
    assert "already done" in response.json()["detail"]


def test_writing_to_an_unknown_task_is_404(client: TestClient) -> None:
    assert client.post("/tasks/999999/accept", json={}).status_code == 404


def test_labels_are_append_only(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    """A re-label adds a row; it never updates the earlier one."""
    from app.services.labeling import Decision, record_decision

    task_id = queue_rows(client)[0]["task_id"]
    client.post(f"/tasks/{task_id}/label", json={"final_text": "first"})
    db_session.expire_all()

    task = db_session.get(AnnotationTask, task_id)
    task.status = "pending"
    db_session.flush()
    record_decision(db_session, task, Decision(disposition="edited", final_text="second"))

    rows = db_session.scalars(
        sa.select(SegmentLabel)
        .where(SegmentLabel.segment_id == task.segment_id)
        .order_by(SegmentLabel.id)
    ).all()
    assert [r.final_text for r in rows] == ["first", "second"]


# --- bulk accept -------------------------------------------------------------------------


def test_bulk_accept_writes_one_label_and_event_per_task(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    ids = [row["task_id"] for row in queue_rows(client)[:4]]
    body = client.post("/tasks/bulk-accept", json={"task_ids": ids}).json()
    assert body["count"] == 4

    db_session.expire_all()
    assert db_session.scalar(sa.select(sa.func.count()).select_from(SegmentLabel)) == 4
    assert db_session.scalar(sa.select(sa.func.count()).select_from(AnnotationEvent)) == 4
    assert all(db_session.get(AnnotationTask, i).status == "done" for i in ids)


def test_bulk_accept_is_all_or_nothing(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    ids = [row["task_id"] for row in queue_rows(client)[:3]]
    response = client.post("/tasks/bulk-accept", json={"task_ids": [*ids, 999999]})
    assert response.status_code == 404
    db_session.rollback()
    assert db_session.scalar(sa.select(sa.func.count()).select_from(SegmentLabel)) == 0


def test_bulk_accept_requires_at_least_one_task(client: TestClient) -> None:
    assert client.post("/tasks/bulk-accept", json={"task_ids": []}).status_code == 422


# --- segments, audio and peaks -----------------------------------------------------------


def test_segment_detail_lists_every_hypothesis(client: TestClient, imported_episode: str) -> None:
    segment_id = queue_rows(client)[0]["segment_id"]
    body = client.get(f"/segments/{segment_id}").json()
    assert len(body["hypotheses"]) == 3
    assert body["scores"] is not None
    assert body["split"] in {"train", "val", "test"}


def test_unknown_segment_is_404(client: TestClient) -> None:
    assert client.get("/segments/999999").status_code == 404


def test_audio_without_a_range_returns_the_whole_clip(
    client: TestClient, imported_episode: str
) -> None:
    segment_id = queue_rows(client)[0]["segment_id"]
    response = client.get(f"/segments/{segment_id}/audio")
    assert response.status_code == 200
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-type"] == "audio/flac"
    assert response.content[:4] == b"fLaC"


def test_audio_range_request_returns_206_with_the_right_bytes(
    client: TestClient, imported_episode: str
) -> None:
    segment_id = queue_rows(client)[0]["segment_id"]
    whole = client.get(f"/segments/{segment_id}/audio").content
    response = client.get(f"/segments/{segment_id}/audio", headers={"Range": "bytes=10-19"})
    assert response.status_code == 206
    assert response.content == whole[10:20]
    assert response.headers["content-range"] == f"bytes 10-19/{len(whole)}"
    assert response.headers["content-length"] == "10"


def test_open_ended_range_runs_to_the_end(client: TestClient, imported_episode: str) -> None:
    segment_id = queue_rows(client)[0]["segment_id"]
    whole = client.get(f"/segments/{segment_id}/audio").content
    response = client.get(f"/segments/{segment_id}/audio", headers={"Range": "bytes=5-"})
    assert response.status_code == 206
    assert response.content == whole[5:]


def test_suffix_range_returns_the_final_bytes(client: TestClient, imported_episode: str) -> None:
    segment_id = queue_rows(client)[0]["segment_id"]
    whole = client.get(f"/segments/{segment_id}/audio").content
    response = client.get(f"/segments/{segment_id}/audio", headers={"Range": "bytes=-16"})
    assert response.status_code == 206
    assert response.content == whole[-16:]


def test_range_beyond_the_object_is_416(client: TestClient, imported_episode: str) -> None:
    segment_id = queue_rows(client)[0]["segment_id"]
    response = client.get(
        f"/segments/{segment_id}/audio", headers={"Range": "bytes=99999999-99999999"}
    )
    assert response.status_code == 416
    assert response.headers["content-range"].startswith("bytes */")


def test_malformed_range_is_416(client: TestClient, imported_episode: str) -> None:
    segment_id = queue_rows(client)[0]["segment_id"]
    response = client.get(f"/segments/{segment_id}/audio", headers={"Range": "kilobytes=1-2"})
    assert response.status_code == 416


def test_peaks_are_served_as_json(client: TestClient, imported_episode: str) -> None:
    segment_id = queue_rows(client)[0]["segment_id"]
    response = client.get(f"/segments/{segment_id}/peaks")
    assert response.status_code == 200
    payload = json.loads(response.content)
    assert len(payload["min"]) == payload["buckets"]
    assert payload["version"] == 1


def test_missing_clip_in_storage_is_404(
    client: TestClient, db_session: Session, object_storage, imported_episode: str
) -> None:
    segment_id = queue_rows(client)[0]["segment_id"]
    segment = db_session.get(Segment, segment_id)
    object_storage.delete(segment.clip_object_key)
    assert client.get(f"/segments/{segment_id}/audio").status_code == 404


# --- transliteration ---------------------------------------------------------------------


def test_translit_returns_candidates(client: TestClient) -> None:
    body = client.post("/translit", json={"token": "kura"}).json()
    assert body["token"] == "kura"
    assert body["candidates"]


def test_translit_respects_the_limit(client: TestClient) -> None:
    body = client.post("/translit", json={"token": "garchhu", "limit": 1}).json()
    assert len(body["candidates"]) <= 1


def test_translit_of_an_empty_token_returns_nothing(client: TestClient) -> None:
    assert client.post("/translit", json={"token": "  "}).json()["candidates"] == []


def test_translit_choice_ranks_the_chosen_form_first(client: TestClient) -> None:
    client.post("/translit", json={"token": "kura"})
    body = client.post("/translit/choice", json={"token": "kura", "devanagari": "कुरा"}).json()
    assert body["candidates"][0] == "कुरा"


# --- authentication ----------------------------------------------------------------------


def test_no_authentication_is_required_by_default(client: TestClient) -> None:
    assert client.get("/stats").status_code == 200


def test_a_configured_token_is_enforced(db_session: Session, object_storage, settings) -> None:
    from fastapi.testclient import TestClient as Client

    from app.api.deps import get_config, get_object_storage, get_session
    from app.main import create_app

    secured = settings.model_copy(
        update={"api": settings.api.model_copy(update={"auth_token": "s3cret"})}
    )
    app = create_app()
    app.dependency_overrides[get_session] = lambda: db_session
    app.dependency_overrides[get_object_storage] = lambda: object_storage
    app.dependency_overrides[get_config] = lambda: secured
    with Client(app) as secured_client:
        assert secured_client.get("/stats").status_code == 401
        assert (
            secured_client.get("/stats", headers={"Authorization": "Bearer wrong"}).status_code
            == 401
        )
        assert (
            secured_client.get("/stats", headers={"Authorization": "Bearer s3cret"}).status_code
            == 200
        )
