"""Importing speaker turns from a diarization run over the retained episode audio (D78)."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import load_settings
from app.models import AuditLog, DiarizationRun, Episode
from app.services.diarization_import import (
    DiarizationError,
    clip_speaker_turns,
    current_run,
    import_diarization,
    parse_diarization,
)
from app.services.fixtures import build_export_fixture
from app.services.importer import import_manifest
from app.storage.local import LocalFilesystemStorage

MODEL = "pyannote/speaker-diarization-community-1"


def payload(episode: str, turns, embeddings=None) -> dict:
    entry = {"turns": turns, "labels": sorted({t[2] for t in turns if len(t) == 3})}
    if embeddings is not None:
        entry["embeddings"] = embeddings
    return {episode: entry}


# --- parsing, no database ---------------------------------------------------------------------


def test_speakers_are_numbered_by_talk_time() -> None:
    parsed = parse_diarization(
        payload(
            "ep", [[0.0, 1.0, "SPEAKER_00"], [1.0, 9.0, "SPEAKER_01"], [9.0, 10.0, "SPEAKER_00"]]
        )
    )
    assert parsed["ep"].speakers == ["SPEAKER_01", "SPEAKER_00"]


def test_turns_are_sorted_and_the_checksum_ignores_input_order() -> None:
    a = parse_diarization(payload("ep", [[5.0, 6.0, "A"], [0.0, 1.0, "B"]]))["ep"]
    b = parse_diarization(payload("ep", [[0.0, 1.0, "B"], [5.0, 6.0, "A"]]))["ep"]
    assert [t[0] for t in a.turns] == [0.0, 5.0]
    assert a.checksum(MODEL) == b.checksum(MODEL)
    assert a.checksum(MODEL) != a.checksum("another/model")


def test_embeddings_are_keyed_by_label() -> None:
    parsed = parse_diarization(
        payload("ep", [[0.0, 1.0, "A"], [1.0, 2.0, "B"]], embeddings=[[0.1, 0.2], [0.3, 0.4]])
    )
    assert parsed["ep"].embeddings == {"A": [0.1, 0.2], "B": [0.3, 0.4]}


@pytest.mark.parametrize(
    "turns",
    [
        [[2.0, 1.0, "A"]],  # ends before it starts
        [[0.0, 1.0]],  # no speaker
        [[0.0, "x", "A"]],  # not a number
    ],
)
def test_a_malformed_turn_is_refused(turns) -> None:
    with pytest.raises(DiarizationError):
        parse_diarization(payload("ep", turns))


def test_an_episode_without_turns_is_refused() -> None:
    with pytest.raises(DiarizationError):
        parse_diarization({"ep": {"labels": []}})


def test_turns_are_cut_to_a_clip_and_numbered() -> None:
    turns = [(0.0, 4.0, "A"), (3.5, 8.0, "B"), (9.0, 12.0, "A")]
    clip = clip_speaker_turns(turns, ["B", "A"], start=3.0, end=10.0)
    assert clip == [
        {"speaker": 2, "start": 0.0, "end": 1.0},
        {"speaker": 1, "start": 0.5, "end": 5.0},
        {"speaker": 2, "start": 6.0, "end": 7.0},
    ]


# --- import, against the database -------------------------------------------------------------


@pytest.fixture
def episode(db_session: Session, tmp_path: Path) -> Episode:
    root = build_export_fixture(tmp_path / "dz", episode_id="dz_ep", segments=2, systems=2)
    storage = LocalFilesystemStorage(root=tmp_path / "objects")
    import_manifest(db_session, root, storage=storage, settings=load_settings())
    return db_session.scalars(sa.select(Episode).where(Episode.external_id == "dz_ep")).one()


TURNS = [[0.0, 3.0, "SPEAKER_00"], [2.5, 6.0, "SPEAKER_01"]]


@pytest.mark.db
def test_a_run_is_stored_for_a_known_episode(db_session: Session, episode: Episode) -> None:
    report = import_diarization(
        db_session, payload("dz_ep", TURNS), model=MODEL, source="d.json", actor="test"
    )
    assert report.runs_created == 1
    run = current_run(db_session, episode.id)
    assert run is not None and run.model == MODEL and run.source == "d.json"
    assert [(t.speaker, t.start_time, t.end_time) for t in run.turns] == [
        ("SPEAKER_00", 0.0, 3.0),
        ("SPEAKER_01", 2.5, 6.0),
    ]
    assert run.speakers_jsonb == ["SPEAKER_01", "SPEAKER_00"]  # 3.5 s of talk against 3.0 s


@pytest.mark.db
def test_unknown_episodes_are_reported_not_fatal(db_session: Session, episode: Episode) -> None:
    body = payload("dz_ep", TURNS) | payload("not_in_the_harness", TURNS)
    report = import_diarization(db_session, body, model=MODEL, source=None, actor="test")
    assert report.runs_created == 1
    assert report.unknown_episodes == ["not_in_the_harness"]


@pytest.mark.db
def test_importing_the_same_run_twice_is_a_no_op(db_session: Session, episode: Episode) -> None:
    import_diarization(db_session, payload("dz_ep", TURNS), model=MODEL, source=None, actor="t")
    report = import_diarization(
        db_session, payload("dz_ep", TURNS), model=MODEL, source=None, actor="t"
    )
    assert report.runs_created == 0 and report.runs_unchanged == 1
    count = db_session.scalar(
        sa.select(sa.func.count())
        .select_from(DiarizationRun)
        .where(DiarizationRun.episode_id == episode.id)
    )
    assert count == 1


@pytest.mark.db
def test_a_newer_run_supersedes_without_deleting(db_session: Session, episode: Episode) -> None:
    import_diarization(db_session, payload("dz_ep", TURNS), model=MODEL, source=None, actor="t")
    first = current_run(db_session, episode.id)
    newer = [[0.0, 2.0, "SPEAKER_00"], [2.0, 6.0, "SPEAKER_01"]]
    import_diarization(db_session, payload("dz_ep", newer), model=MODEL, source=None, actor="t")
    second = current_run(db_session, episode.id)
    assert second.id != first.id
    assert [t.end_time for t in second.turns] == [2.0, 6.0]
    assert db_session.get(DiarizationRun, first.id) is not None


@pytest.mark.db
def test_every_run_leaves_an_audit_row(db_session: Session, episode: Episode) -> None:
    import_diarization(db_session, payload("dz_ep", TURNS), model=MODEL, source=None, actor="me")
    row = db_session.scalars(
        sa.select(AuditLog).where(
            AuditLog.entity_type == "episode",
            AuditLog.entity_id == "dz_ep",
            AuditLog.action == "diarization_import",
        )
    ).one()
    assert row.actor == "me"
    assert row.new_values_jsonb["turns"] == 2
    assert row.new_values_jsonb["model"] == MODEL
