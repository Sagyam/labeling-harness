"""Tests for episode and segment listing and deletion endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import AnnotationTask, Episode, Segment
from app.services.analysis import analyze_transcript
from app.services.flags import FlagHypothesis, compute_flags

pytestmark = pytest.mark.db


def test_hindi_intrusion_flag_detected() -> None:
    # Sentence with Hindi intrusion word "नहीं" and "था"
    hindi_text = "हामीले यो project मा काम गरेको था तर नहीं भयो"
    res = analyze_transcript(hindi_text, duration_seconds=4.0)
    assert "hindi_intrusion" in res.flags

    # Pure authentic Nepali sentence
    pure_nepali = "हामीले यो project मा काम गरेको थियो तर भएन"
    res_clean = analyze_transcript(pure_nepali, duration_seconds=4.0)
    assert "hindi_intrusion" not in res_clean.flags


def test_compute_flags_hindi_marker() -> None:
    flags = compute_flags(
        duration_seconds=5.0,
        hypotheses=[FlagHypothesis(text="यो कुरा सही नहीं हो")],
    )
    assert "hindi_intrusion" in flags


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
    assert len(segs_data) == 2
    assert segs_data[0]["external_id"] == "test_ep_del_001"

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
