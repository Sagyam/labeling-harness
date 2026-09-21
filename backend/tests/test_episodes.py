"""Tests for episode and segment listing and deletion endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import AnnotationTask, AuditLog, Episode, Segment

pytestmark = pytest.mark.db


def test_list_episodes_and_delete_cascade(
    client: TestClient, db_session: Session, tmp_path: Path
) -> None:
    # Create test episode
    ep = Episode(
        external_id="test_ep_del",
        title="Episode to Delete",
        show_id="show_1",
        duration_seconds=120.0,
    )
    db_session.add(ep)
    db_session.flush()

    # Create 2 segments
    seg1 = Segment(
        episode_id=ep.id,
        external_id="test_ep_del_001",
        start_time=0.0,
        end_time=10.0,
        duration_seconds=10.0,
        clip_object_key="clips/test_ep_del/test_ep_del_001.flac",
        clip_checksum="fakechk1",
    )
    seg2 = Segment(
        episode_id=ep.id,
        external_id="test_ep_del_002",
        start_time=10.0,
        end_time=20.0,
        duration_seconds=10.0,
        clip_object_key="clips/test_ep_del/test_ep_del_002.flac",
        clip_checksum="fakechk2",
    )
    db_session.add_all([seg1, seg2])
    db_session.flush()

    task1 = AnnotationTask(
        segment_id=seg1.id,
        queue="review",
        priority_score=0.8,
        status="pending",
    )
    db_session.add(task1)
    db_session.commit()

    task1_id = task1.id
    seg1_id = seg1.id
    seg2_id = seg2.id
    ep_id = ep.id

    # 1. GET /episodes
    resp = client.get("/episodes")
    assert resp.status_code == 200
    data = resp.json()
    assert any(e["external_id"] == "test_ep_del" for e in data)
    ep_entry = next(e for e in data if e["external_id"] == "test_ep_del")
    assert ep_entry["segment_count"] == 2
    assert ep_entry["pending_count"] == 1

    # 2. GET /episodes/{id}/segments
    resp_segs = client.get(f"/episodes/{ep_id}/segments")
    assert resp_segs.status_code == 200
    segs_data = resp_segs.json()
    assert segs_data[0]["external_id"] == "test_ep_del_001"
    assert segs_data[0]["task_id"] == task1_id
    assert segs_data[0]["task_status"] == "pending"
    assert segs_data[1]["task_id"] is None

    # 3. DELETE single segment /segments/{seg2_id}
    del_seg_resp = client.delete(f"/segments/{seg2_id}")
    assert del_seg_resp.status_code == 200
    assert del_seg_resp.json()["deleted"] is True
    # Verify seg2 is gone
    assert db_session.scalar(sa.select(Segment).where(Segment.id == seg2_id)) is None

    # 4. DELETE entire episode /episodes/{ep_id}
    del_ep_resp = client.delete(f"/episodes/{ep_id}")
    assert del_ep_resp.status_code == 200
    assert del_ep_resp.json()["deleted"] is True
    assert del_ep_resp.json()["deleted_segments"] == 1

    # Verify episode and remaining segment + task are gone
    assert db_session.scalar(sa.select(Episode).where(Episode.id == ep_id)) is None
    assert db_session.scalar(sa.select(Segment).where(Segment.id == seg1_id)) is None
    assert db_session.scalar(sa.select(AnnotationTask).where(AnnotationTask.id == task1_id)) is None


# --- counting when a segment carries more than one task ----------------------------------
#
# The partial unique index only forbids two *active* tasks per segment, so a skipped task
# alongside its replacement is a legal, ordinary state -- and it is what the counting and
# status queries used to get wrong.


@pytest.fixture
def segment_with_two_tasks(db_session: Session, imported_episode: str) -> int:
    """Mark one task skipped and queue a replacement, returning that segment's id."""
    task = db_session.scalars(sa.select(AnnotationTask)).first()
    task.status = "skipped"
    db_session.flush()
    db_session.add(
        AnnotationTask(
            segment_id=task.segment_id, queue="review", priority_score=0.9, status="pending"
        )
    )
    db_session.flush()
    return task.segment_id


def test_segment_count_is_not_inflated_by_a_second_task(
    client: TestClient, db_session: Session, imported_episode: str, segment_with_two_tasks: int
) -> None:
    real_segments = db_session.scalar(sa.select(sa.func.count()).select_from(Segment))
    episode = next(
        row for row in client.get("/episodes").json() if row["external_id"] == imported_episode
    )
    assert episode["segment_count"] == real_segments


def test_pending_count_counts_segments_not_tasks(
    client: TestClient, db_session: Session, imported_episode: str, segment_with_two_tasks: int
) -> None:
    episode = next(
        row for row in client.get("/episodes").json() if row["external_id"] == imported_episode
    )
    assert episode["pending_count"] <= episode["segment_count"]


def test_segment_task_status_reports_the_active_task(
    client: TestClient, imported_episode: str, segment_with_two_tasks: int
) -> None:
    rows = client.get(f"/episodes/{imported_episode}/segments").json()
    row = next(r for r in rows if r["id"] == segment_with_two_tasks)
    assert row["task_status"] == "pending"


def test_segment_task_status_is_none_once_no_task_is_active(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    task = db_session.scalars(sa.select(AnnotationTask)).first()
    task.status = "done"
    db_session.flush()

    rows = client.get(f"/episodes/{imported_episode}/segments").json()
    row = next(r for r in rows if r["id"] == task.segment_id)
    assert row["task_status"] is None


def test_a_segment_without_peaks_advertises_no_peaks_url(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    segment = db_session.scalars(sa.select(Segment)).first()
    segment.peaks_object_key = None
    db_session.flush()

    rows = client.get(f"/episodes/{imported_episode}/segments").json()
    row = next(r for r in rows if r["id"] == segment.id)
    assert row["peaks_url"] is None


def test_deleting_a_segment_writes_an_audit_entry(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    segment = db_session.scalars(sa.select(Segment)).first()
    external_id = segment.external_id

    assert client.delete(f"/segments/{segment.id}").json()["deleted"] is True

    entry = db_session.scalars(sa.select(AuditLog).where(AuditLog.entity_type == "segments")).one()
    assert entry.action == "delete"
    assert entry.old_values_jsonb["external_id"] == external_id


def test_deleting_an_episode_writes_an_audit_entry(
    client: TestClient, db_session: Session, imported_episode: str
) -> None:
    assert client.delete(f"/episodes/{imported_episode}").json()["deleted"] is True

    entry = db_session.scalars(sa.select(AuditLog).where(AuditLog.entity_type == "episodes")).one()
    assert entry.action == "delete"
    assert entry.old_values_jsonb["external_id"] == imported_episode
    assert entry.old_values_jsonb["segments"] > 0


# --- bulk delete -----------------------------------------------------------------------------


def test_bulk_delete_removes_every_segment_its_objects_and_audits_each(
    client: TestClient, db_session: Session, imported_episode: str, object_storage
) -> None:
    segments = db_session.scalars(sa.select(Segment).order_by(Segment.id)).all()[:3]
    ids = [s.id for s in segments]
    keys = [k for s in segments for k in (s.clip_object_key, s.peaks_object_key) if k]
    assert all(object_storage.exists(k) for k in keys)

    body = client.post("/segments/bulk-delete", json={"segment_ids": ids}).json()
    assert body["count"] == 3
    assert sorted(body["deleted"]) == sorted(ids)

    db_session.expire_all()
    remaining = db_session.scalars(sa.select(Segment.id).where(Segment.id.in_(ids))).all()
    assert remaining == []
    assert not any(object_storage.exists(k) for k in keys)
    entries = db_session.scalars(
        sa.select(AuditLog).where(AuditLog.entity_type == "segments", AuditLog.action == "delete")
    ).all()
    assert sorted(int(e.entity_id) for e in entries) == sorted(ids)
    assert all(e.old_values_jsonb["bulk"] is True for e in entries)


def test_bulk_delete_refuses_the_whole_batch_if_any_clip_is_gold(
    client: TestClient, db_session: Session, imported_episode: str, object_storage
) -> None:
    segments = db_session.scalars(sa.select(Segment).order_by(Segment.id)).all()[:3]
    segments[1].pot = "gold"
    db_session.flush()
    ids = [s.id for s in segments]

    response = client.post("/segments/bulk-delete", json={"segment_ids": ids})
    assert response.status_code == 409
    assert str(segments[1].id) in response.json()["detail"]

    db_session.expire_all()
    assert len(db_session.scalars(sa.select(Segment).where(Segment.id.in_(ids))).all()) == 3
    assert object_storage.exists(segments[0].clip_object_key)


def test_bulk_delete_is_all_or_nothing_on_an_unknown_id(
    client: TestClient, db_session: Session, imported_episode: str, object_storage
) -> None:
    segment = db_session.scalars(sa.select(Segment)).first()
    response = client.post("/segments/bulk-delete", json={"segment_ids": [segment.id, 999999]})
    assert response.status_code == 404

    db_session.expire_all()
    assert db_session.get(Segment, segment.id) is not None
    assert object_storage.exists(segment.clip_object_key)


def test_bulk_delete_requires_at_least_one_segment(client: TestClient) -> None:
    assert client.post("/segments/bulk-delete", json={"segment_ids": []}).status_code == 422
