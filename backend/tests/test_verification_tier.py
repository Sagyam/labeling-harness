"""Tests for the verification tier and the gold pot's purity rule (D63).

The tier is the corpus's claim about itself: a ``verified`` row was played and read, a ``screened``
row was waved through on the cross-ASR disagreement signal. These tests are about the two ways that
claim could quietly become false -- a screened row landing in the benchmark, and a screened row
exporting as if a human had confirmed it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.models import AnnotationTask, Episode, SegmentLabel
from app.services.export import GoldPurityError, export_dataset
from app.services.fixtures import build_export_fixture
from app.services.importer import import_manifest
from app.services.labeling import Decision, LabelingError, record_decision
from app.services.queue_builder import build_queue
from app.storage.local import LocalFilesystemStorage

pytestmark = pytest.mark.db


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(root=tmp_path / "objects")


@pytest.fixture
def queued(
    db_session: Session, tmp_path: Path, storage, settings: Settings
) -> list[AnnotationTask]:
    root = build_export_fixture(
        tmp_path / "export_tier", episode_id="tier_ep001", segments=6, systems=2, with_fusion=True
    )
    import_manifest(db_session, root, storage=storage, settings=settings)
    build_queue(db_session, settings=settings, audit_sample_rate=0.0)
    db_session.flush()
    return list(db_session.scalars(sa.select(AnnotationTask).order_by(AnnotationTask.id)))


def set_pot(db_session: Session, pot: str) -> None:
    """Put every clip of the one episode in ``pot`` -- gold is per clip since D71."""
    episode = db_session.scalars(sa.select(Episode)).one()
    for segment in episode.segments:
        segment.pot = pot
    db_session.flush()


# --- the default is the strong claim -----------------------------------------------------


def test_a_decision_defaults_to_verified(
    db_session: Session, queued: list[AnnotationTask], settings: Settings
) -> None:
    """A caller that says nothing must not be able to weaken what the corpus claims."""
    label = record_decision(
        db_session, queued[0], Decision(disposition="accepted_unchanged"), settings=settings
    )
    assert label.verification_tier == "verified"


def test_screening_is_recorded_as_screening(
    db_session: Session, queued: list[AnnotationTask], settings: Settings
) -> None:
    set_pot(db_session, "train")
    label = record_decision(
        db_session,
        queued[0],
        Decision(disposition="accepted_unchanged", verification_tier="screened"),
        settings=settings,
    )
    assert label.verification_tier == "screened"


def test_an_unknown_tier_is_refused(
    db_session: Session, queued: list[AnnotationTask], settings: Settings
) -> None:
    with pytest.raises(LabelingError, match="unknown verification tier"):
        record_decision(
            db_session,
            queued[0],
            Decision(disposition="accepted_unchanged", verification_tier="skimmed"),
            settings=settings,
        )


# --- the gold pot rule -------------------------------------------------------------------


def test_a_gold_segment_cannot_be_screened(
    db_session: Session, queued: list[AnnotationTask], settings: Settings
) -> None:
    """The whole value of the benchmark is that every row in it was actually checked."""
    set_pot(db_session, "gold")
    with pytest.raises(LabelingError, match="gold pot"):
        record_decision(
            db_session,
            queued[0],
            Decision(disposition="accepted_unchanged", verification_tier="screened"),
            settings=settings,
        )


def test_a_gold_segment_can_still_be_verified(
    db_session: Session, queued: list[AnnotationTask], settings: Settings
) -> None:
    set_pot(db_session, "gold")
    label = record_decision(
        db_session,
        queued[0],
        Decision(disposition="accepted_unchanged", verification_tier="verified"),
        settings=settings,
    )
    assert label.verification_tier == "verified"


def test_the_api_refuses_a_screened_gold_decision(
    client, imported_episode: str, db_session: Session
) -> None:
    episode = db_session.scalars(sa.select(Episode)).one()
    for segment in episode.segments:
        segment.pot = "gold"
    db_session.flush()

    task_id = client.get("/queue").json()[0]["task_id"]
    response = client.post(f"/tasks/{task_id}/accept", json={"verification_tier": "screened"})
    assert response.status_code == 409
    assert "gold pot" in response.json()["detail"]


def test_the_api_reports_the_tier_it_recorded(client, imported_episode: str) -> None:
    task_id = next(r for r in client.get("/queue").json() if not r["reason"]["hazards"])["task_id"]
    body = client.post(f"/tasks/{task_id}/accept", json={"verification_tier": "screened"}).json()
    assert body["verification_tier"] == "screened"


def test_an_unknown_tier_is_rejected_by_the_api(client, imported_episode: str) -> None:
    task_id = client.get("/queue").json()[0]["task_id"]
    response = client.post(f"/tasks/{task_id}/accept", json={"verification_tier": "skimmed"})
    assert response.status_code == 422


# --- export ------------------------------------------------------------------------------


def test_every_exported_row_carries_its_tier(
    db_session: Session, queued: list[AnnotationTask], settings: Settings, tmp_path: Path
) -> None:
    import json

    set_pot(db_session, "train")
    for index, task in enumerate(queued[:4]):
        record_decision(
            db_session,
            task,
            Decision(
                disposition="accepted_unchanged",
                verification_tier="screened" if index % 2 else "verified",
            ),
            settings=settings,
        )
    db_session.flush()

    result = export_dataset(
        db_session, kind="training", output_root=tmp_path / "out", settings=settings
    )
    rows = [json.loads(line) for line in result.data_path.read_text().splitlines()]
    assert rows
    assert all("verification_tier" in row for row in rows)
    assert {row["verification_tier"] for row in rows} == {"verified", "screened"}
    assert all(row["pot"] == "train" for row in rows)


def test_the_manifest_reports_the_tier_mix_per_split(
    db_session: Session, queued: list[AnnotationTask], settings: Settings, tmp_path: Path
) -> None:
    import json

    set_pot(db_session, "train")
    for index, task in enumerate(queued[:4]):
        record_decision(
            db_session,
            task,
            Decision(
                disposition="accepted_unchanged",
                verification_tier="screened" if index % 2 else "verified",
            ),
            settings=settings,
        )
    db_session.flush()

    result = export_dataset(
        db_session, kind="training", output_root=tmp_path / "out", settings=settings
    )
    manifest = json.loads(result.manifest_path.read_text())
    mix = manifest["verification_tiers_by_split"]["train"]
    assert mix == {"screened": 2, "verified": 2}


def test_the_gold_export_refuses_a_screened_row(
    db_session: Session, queued: list[AnnotationTask], settings: Settings, tmp_path: Path
) -> None:
    """`record_decision` blocks this, so reaching the export means it got in another way."""
    set_pot(db_session, "train")
    for task in queued[:3]:
        record_decision(
            db_session,
            task,
            Decision(disposition="accepted_unchanged", verification_tier="screened"),
            settings=settings,
        )
    db_session.flush()

    # The episode moves into gold after its clips were screened -- the exact drift the check is
    # there to catch.
    set_pot(db_session, "gold")

    with pytest.raises(GoldPurityError, match="not verified"):
        export_dataset(db_session, kind="gold", output_root=tmp_path / "out", settings=settings)


def test_the_gold_export_writes_when_every_row_is_verified(
    db_session: Session, queued: list[AnnotationTask], settings: Settings, tmp_path: Path
) -> None:
    set_pot(db_session, "gold")
    for task in queued[:3]:
        record_decision(
            db_session,
            task,
            Decision(disposition="accepted_unchanged", verification_tier="verified"),
            settings=settings,
        )
    db_session.flush()

    result = export_dataset(
        db_session, kind="gold", output_root=tmp_path / "out", settings=settings
    )
    assert result.row_count == 3


# --- append-only history keeps the tier --------------------------------------------------


def test_relabelling_records_a_new_tier_without_losing_the_old_one(
    db_session: Session, queued: list[AnnotationTask], settings: Settings
) -> None:
    set_pot(db_session, "train")
    task = queued[0]
    record_decision(
        db_session,
        task,
        Decision(disposition="accepted_unchanged", verification_tier="screened"),
        settings=settings,
    )
    db_session.flush()

    task.status = "pending"
    record_decision(
        db_session,
        task,
        Decision(disposition="edited", final_text="corrected", verification_tier="verified"),
        settings=settings,
    )
    db_session.flush()

    tiers = list(
        db_session.scalars(
            sa.select(SegmentLabel.verification_tier)
            .where(SegmentLabel.segment_id == task.segment_id)
            .order_by(SegmentLabel.id)
        )
    )
    assert tiers == ["screened", "verified"]


# --- a hazard gate is a promise that someone will listen (D74) ---------------------------------


def _gate(task: AnnotationTask, db_session: Session, *hazards: str) -> None:
    task.reason_jsonb = {**(task.reason_jsonb or {}), "hazards": list(hazards)}
    db_session.flush()


def test_a_gated_clip_cannot_be_screened(
    db_session: Session, queued: list[AnnotationTask], settings: Settings
) -> None:
    _gate(queued[0], db_session, "invention")
    with pytest.raises(LabelingError, match="invention"):
        record_decision(
            db_session,
            queued[0],
            Decision(disposition="accepted_unchanged", verification_tier="screened"),
            settings=settings,
        )


def test_a_gated_clip_can_still_be_verified(
    db_session: Session, queued: list[AnnotationTask], settings: Settings
) -> None:
    _gate(queued[0], db_session, "dropped")
    label = record_decision(
        db_session, queued[0], Decision(disposition="accepted_unchanged"), settings=settings
    )
    assert label.verification_tier == "verified"


def test_the_api_refuses_to_screen_a_gated_clip(client, imported_episode: str, db_session) -> None:
    task_id = client.get("/queue").json()[0]["task_id"]
    task = db_session.get(AnnotationTask, task_id)
    task.reason_jsonb = {**task.reason_jsonb, "hazards": ["seam_bleed"]}
    db_session.flush()
    response = client.post(f"/tasks/{task_id}/accept", json={"verification_tier": "screened"})
    assert response.status_code == 409
    assert "seam_bleed" in response.json()["detail"]
