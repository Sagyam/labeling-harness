"""Stage 4: fusion of the recognisers into one hypothesis per clip (D72)."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.services.ingest import (
    IngestJob,
    run_pipeline,
)
from tests.ingest_support import make_test_audio

pytestmark = pytest.mark.db


# --- stage 4: fusion (D72) --------------------------------------------------------------------


def _echo_first(messages):
    """A fuser that keeps recogniser A's reading for every target segment."""
    import json as _json
    import re as _re

    from app.llm.base import LlmResult

    block = messages[-1]["content"].split("### TRANSCRIBE THESE", 1)[1].split("###", 1)[0]
    items = [
        {"id": int(i), "t": a, "c": "k"}
        for i, a in _re.findall(r"^\[(\d+)\][^\n]*\n  A: ([^\n]*)", block, _re.M)
    ]
    return LlmResult(
        route="fuse_transcript",
        model="gemini-3.8-flash",
        text=_json.dumps(items, ensure_ascii=False),
        raw={"candidates": [{"finishReason": "STOP"}], "modelVersion": "gemini-3.8-flash-001"},
    )


def _run(db_session, object_storage, settings, tmp_path, episode_id: str) -> IngestJob:
    raw_audio = make_test_audio(tmp_path / f"{episode_id}.wav", duration_seconds=6.0)
    job = IngestJob(
        job_id=f"test-{episode_id}",
        episode_id=episode_id,
        show_id="podcast",
        title="Fusion stage",
        audio_path=raw_audio,
        work_dir=tmp_path / f"work_{episode_id}",
    )
    run_pipeline(job, lambda: db_session, object_storage, settings)
    return job


def test_a_dry_run_skips_fusion_and_says_so(
    db_session: Session, object_storage, settings, tmp_path: Path
) -> None:
    job = _run(db_session, object_storage, settings, tmp_path, "web_fuse_dry")
    assert job.status == "completed"
    assert any("Fusion skipped on a dry run" in entry.message for entry in job.logs)
    assert job.summary["fusion"] is None


def test_every_clip_is_imported_with_a_fused_hypothesis(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.models import AsrHypothesis, AsrSystem

    monkeypatch.setattr(
        "app.services.ingest.fusion._fusion_completer", lambda session, routes, route: _echo_first
    )
    job = _run(db_session, object_storage, settings, tmp_path, "web_fuse")

    assert job.status == "completed", job.error
    assert job.summary["fusion"]["fused"] == job.summary["segments"]
    assert any("Stage 4/6: Fusing" in entry.message for entry in job.logs)
    kinds = dict(db_session.execute(sa.select(AsrSystem.system_id, AsrSystem.kind)).all())
    fusion_ids = [name for name, kind in kinds.items() if kind == "fusion"]
    assert fusion_ids == ["fusion-gemini-3.8-flash-p1"]
    fused = db_session.scalars(
        sa.select(AsrHypothesis).join(AsrSystem).where(AsrSystem.kind == "fusion")
    ).all()
    assert len(fused) == job.summary["segments"]
    assert all(h.metadata_jsonb["fusion"]["code"] == "k" for h in fused)


def test_a_fuser_that_blows_up_never_fails_the_episode(
    db_session: Session, object_storage, settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(session, routes, route):
        raise RuntimeError("vertex is on fire")

    monkeypatch.setattr("app.services.ingest.fusion._fusion_completer", broken)
    job = _run(db_session, object_storage, settings, tmp_path, "web_fuse_broken")

    assert job.status == "completed"
    assert "vertex is on fire" in job.summary["fusion"]["error"]
