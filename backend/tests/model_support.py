"""Helpers shared by the fine-tuned model test modules (D83)."""

from __future__ import annotations

import json
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import AnnotationTask, Episode
from app.services.importer import import_manifest
from app.services.labeling import Decision, record_decision
from app.services.queue_builder import build_queue
from app.storage.local import LocalFilesystemStorage
from tests.fixtures import build_export_fixture

CARD = {
    "name": "Flex FT 2026-09-12",
    "created_at": "2026-09-12T18:00:00+00:00",
    "description": "04c, standard decoder",
    "architecture": "Canary-style enc-dec",
    "decoder": "greedy+cap+retry",
}


def write_model(root: Path, slug: str, *, card=CARD, gold=None, val=None) -> Path:
    folder = root / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
    for name, rows in (("gold", gold), ("val", val)):
        if rows is not None:
            (folder / f"{name}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
            )
    return folder


def label_corpus(db_session: Session, tmp_path: Path, settings: Settings) -> dict[str, str]:
    """Import one podcast episode of four clips, put 0-2 in gold, label all four.

    Clip 0 has half a second of crosstalk. Returns external id -> final text.
    """
    root = build_export_fixture(tmp_path / "mx", episode_id="mx_ep", segments=4, systems=2)
    storage = LocalFilesystemStorage(root=tmp_path / "objects")
    import_manifest(db_session, root, storage=storage, settings=settings)
    episode = db_session.scalars(sa.select(Episode).where(Episode.external_id == "mx_ep")).one()
    episode.metadata_jsonb = (episode.metadata_jsonb or {}) | {"genre": "podcast"}
    segments = sorted(episode.segments, key=lambda s: s.external_id)
    for segment in segments[:3]:
        segment.pot = "gold"
    segments[0].overlap_spans_jsonb = [[0.0, 0.5]]
    db_session.flush()
    build_queue(db_session, settings=settings, audit_sample_rate=0.0)
    texts = {}
    for index, task in enumerate(db_session.scalars(sa.select(AnnotationTask))):
        text = f"एक दुई तीन चार {index}"
        record_decision(
            db_session, task, Decision(disposition="edited", final_text=text), settings=settings
        )
        texts[task.segment.external_id] = text
    db_session.flush()
    return texts
