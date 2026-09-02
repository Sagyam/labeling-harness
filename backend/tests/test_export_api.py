"""Tests for the dataset export API."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Episode
from app.services.fixtures import build_export_fixture
from app.services.importer import import_manifest
from app.services.labeling import Decision, record_decision
from app.services.queue_builder import build_queue
from app.storage.local import LocalFilesystemStorage

pytestmark = pytest.mark.db


@pytest.fixture
def export_corpus(
    db_session: Session, tmp_path: Path, settings: Settings
) -> tuple[str, LocalFilesystemStorage]:
    storage = LocalFilesystemStorage(root=tmp_path / "storage")
    root = build_export_fixture(
        tmp_path / "fixture", episode_id="exp_api_001", segments=4, systems=2, with_words=True
    )
    import_manifest(db_session, root, storage=storage, settings=settings)
    ep = db_session.scalars(sa.select(Episode).where(Episode.external_id == "exp_api_001")).one()
    ep.split = "train"
    db_session.flush()

    build_queue(db_session, settings=settings)

    # Label one segment so training export has data
    from app.models import AnnotationTask

    task = db_session.scalars(sa.select(AnnotationTask)).first()
    assert task is not None
    record_decision(
        db_session,
        task,
        Decision(disposition="accepted_unchanged", final_text="test", annotator="tester"),
        settings=settings,
    )
    db_session.commit()
    return "exp_api_001", storage


def test_export_api_rejects_unknown_kind(client: TestClient) -> None:
    resp = client.post("/export", json={"kind": "invalid_kind"})
    assert resp.status_code == 400
    assert "Unknown export kind" in resp.json()["detail"]


def test_export_api_generates_files_and_manifest(
    client: TestClient, export_corpus: tuple[str, LocalFilesystemStorage]
) -> None:
    resp = client.post("/export", json={"kind": "training"})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 1

    item = data["results"][0]
    assert item["kind"] == "training"
    assert item["row_count"] >= 1
    assert item["data_filename"] == "training.jsonl"
    assert item["manifest_filename"] == "manifest.json"
    assert "download_url" in item
    assert "manifest" in item
    assert item["manifest"]["kind"] == "training"


def test_export_download_endpoint_validation_and_streaming(
    client: TestClient, export_corpus: tuple[str, LocalFilesystemStorage]
) -> None:
    # 1. Invalid kind
    res404 = client.get("/export/download/bad_kind/bad_kind.jsonl")
    assert res404.status_code == 404

    # 2. Invalid filename (directory traversal or unexpected file)
    res400 = client.get("/export/download/training/../../secret.txt")
    assert res400.status_code in (400, 404)

    # 3. Generate export
    post_res = client.post("/export", json={"kind": "training"})
    assert post_res.status_code == 200

    # 4. Download JSONL
    dl_res = client.get("/export/download/training/training.jsonl")
    assert dl_res.status_code == 200
    assert "application/json" in dl_res.headers.get("content-type", "")
    assert len(dl_res.content) > 0

    # 5. Download manifest
    man_res = client.get("/export/download/training/manifest.json")
    assert man_res.status_code == 200
    assert "application/json" in man_res.headers.get("content-type", "")


def test_export_history_lists_existing_exports(
    client: TestClient, export_corpus: tuple[str, LocalFilesystemStorage]
) -> None:
    # Trigger export
    client.post("/export", json={"kind": "training"})

    # Check history
    resp = client.get("/export/history")
    assert resp.status_code == 200
    items = resp.json()
    assert any(item["kind"] == "training" for item in items)
    training_item = next(item for item in items if item["kind"] == "training")
    assert training_item["data_filename"] == "training.jsonl"
    assert training_item["file_bytes"] > 0
