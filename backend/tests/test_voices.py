"""Linking diarized speakers across episodes into anonymous voices (D87)."""

from __future__ import annotations

import numpy as np
import pytest
import sqlalchemy as sa

from app.models import AuditLog, DiarizationRun, Episode
from app.services.voices import LINK_THRESHOLD, assign_voices, link_voices


def _unit(*values: float) -> list[float]:
    v = np.asarray(values, dtype=float)
    return list(v / np.linalg.norm(v))


A, B, C = _unit(1, 0, 0), _unit(0, 1, 0), _unit(0, 0, 1)
A_ISH = _unit(1, 0.2, 0)  # cosine 0.98 to A


def test_a_speaker_joins_the_voice_it_matches_and_a_stranger_starts_a_new_one() -> None:
    centroids: dict[str, np.ndarray] = {}
    first = assign_voices({"S0": A, "S1": B}, ["S0", "S1"], centroids)
    assert first == {"S0": "v001", "S1": "v002"}
    second = assign_voices({"S0": A_ISH, "S1": C}, ["S0", "S1"], centroids)
    assert second == {"S0": "v001", "S1": "v003"}


def test_two_speakers_of_one_episode_never_share_a_voice() -> None:
    """The diarizer already told them apart; linking must not undo that."""
    centroids: dict[str, np.ndarray] = {}
    assign_voices({"S0": A}, ["S0"], centroids)
    both_like_a = assign_voices({"S0": A_ISH, "S1": A}, ["S0", "S1"], centroids)
    assert sorted(both_like_a.values()) == ["v001", "v002"]
    assert both_like_a["S1"] == "v001"  # the closer match wins the voice


def test_a_match_below_the_threshold_is_a_new_voice() -> None:
    centroids: dict[str, np.ndarray] = {}
    assign_voices({"S0": A}, ["S0"], centroids)
    angle = np.arccos(LINK_THRESHOLD - 0.05)
    near_miss = _unit(np.cos(angle), np.sin(angle), 0)
    assert assign_voices({"S0": near_miss}, ["S0"], centroids) == {"S0": "v002"}


def test_a_speaker_without_an_embedding_gets_no_voice() -> None:
    centroids: dict[str, np.ndarray] = {}
    assert assign_voices({"S0": A}, ["S0", "S1"], centroids) == {"S0": "v001"}


# --- the database ------------------------------------------------------------------------------


def _run(session, episode: Episode, checksum: str, embeddings: dict[str, list[float]] | None):
    run = DiarizationRun(
        episode_id=episode.id,
        model="test",
        checksum=checksum,
        speakers_jsonb=sorted(embeddings or {"S0": []}),
        embeddings_jsonb=embeddings,
    )
    session.add(run)
    session.flush()
    return run


@pytest.fixture
def episodes(db_session):
    out = []
    for name in ("vx_a", "vx_b", "vx_c"):
        episode = Episode(external_id=name, split="train")
        db_session.add(episode)
        out.append(episode)
    db_session.flush()
    return out


@pytest.mark.db
def test_linking_names_voices_in_run_order_and_is_idempotent(db_session, episodes) -> None:
    a, b, c = episodes
    r1 = _run(db_session, a, "r1", {"S0": A, "S1": B})
    r2 = _run(db_session, b, "r2", {"S0": A_ISH})
    r3 = _run(db_session, c, "r3", None)

    report = link_voices(db_session, actor="test")

    assert r1.voices_jsonb == {"S0": "v001", "S1": "v002"}
    assert r2.voices_jsonb == {"S0": "v001"}
    assert r3.voices_jsonb == {}  # linked, nothing to link it by
    assert report.runs_linked == 3
    assert report.voices == 2

    again = link_voices(db_session, actor="test")
    assert again.runs_linked == 0

    later = _run(db_session, c, "r4", {"S0": C, "S1": _unit(0.1, 1, 0)})
    link_voices(db_session, actor="test")
    assert later.voices_jsonb == {"S1": "v002", "S0": "v003"}
    audit = db_session.scalars(sa.select(AuditLog).where(AuditLog.action == "voice_link"))
    assert len(list(audit)) == 4  # one per run linked


@pytest.mark.db
def test_relinking_starts_over(db_session, episodes) -> None:
    a, b, _ = episodes
    r1 = _run(db_session, a, "r1", {"S0": A})
    r2 = _run(db_session, b, "r2", {"S0": B})
    link_voices(db_session, actor="test")
    r1.voices_jsonb = {"S0": "v099"}
    db_session.flush()

    report = link_voices(db_session, actor="test", relink=True)

    assert report.runs_linked == 2
    assert r1.voices_jsonb == {"S0": "v001"}
    assert r2.voices_jsonb == {"S0": "v002"}
