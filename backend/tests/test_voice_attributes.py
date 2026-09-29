"""Tests for a voice's gender and age, assigned by the owner from listening (D104).

A voice is still an anonymous id (D56): what is stored is the two coarse, allowlisted fields a
declared speaker row can carry, never a name. The store is append-only, newest per voice current,
and every change has an audit row.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import AuditLog, DiarizationRun, Segment, VoiceAttribute
from app.services.diarization_import import import_diarization
from app.services.voice_attributes import (
    VoiceAttributeError,
    current_voice_attributes,
    set_voice_attributes,
)
from app.services.voices import link_voices

pytestmark = pytest.mark.db

DIM = 256


def axis(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


@pytest.fixture
def voices(client, imported_episode: str, db_session: Session) -> list[str]:
    """Two linked voices in the imported episode: SPEAKER_00 on the first three clips."""
    segments = list(db_session.scalars(sa.select(Segment).order_by(Segment.id)))
    turns = [
        [s.start_time, s.end_time, "SPEAKER_00" if i < 3 else "SPEAKER_01"]
        for i, s in enumerate(segments)
    ]
    import_diarization(
        db_session,
        {
            imported_episode: {
                "turns": turns,
                "labels": ["SPEAKER_00", "SPEAKER_01"],
                "embeddings": [axis(0), axis(1)],
            }
        },
        model="pyannote/speaker-diarization-community-1",
        source="t.json",
        actor="test",
    )
    link_voices(db_session, actor="test")
    db_session.flush()
    run = db_session.scalars(sa.select(DiarizationRun)).one()
    return [run.voices_jsonb["SPEAKER_00"], run.voices_jsonb["SPEAKER_01"]]


def test_setting_a_voices_gender_and_age_is_stored_and_audited(
    db_session: Session, voices: list[str]
) -> None:
    row = set_voice_attributes(
        db_session, voices[0], gender="female", age_bracket="40_59", annotator="owner"
    )
    assert row is not None
    assert current_voice_attributes(db_session) == {
        voices[0]: {"gender": "female", "age_bracket": "40_59"}
    }
    audit = db_session.scalars(
        sa.select(AuditLog).where(AuditLog.entity_type == "voice", AuditLog.entity_id == voices[0])
    ).one()
    assert audit.action == "voice_attributes"
    assert audit.old_values_jsonb == {"gender": None, "age_bracket": None}
    assert audit.new_values_jsonb == {"gender": "female", "age_bracket": "40_59"}


def test_a_change_appends_a_row_and_the_newest_is_current(
    db_session: Session, voices: list[str]
) -> None:
    set_voice_attributes(db_session, voices[0], gender="male", age_bracket=None, annotator="o")
    set_voice_attributes(db_session, voices[0], gender="female", age_bracket="20_39", annotator="o")
    assert db_session.scalar(sa.select(sa.func.count()).select_from(VoiceAttribute)) == 2
    assert current_voice_attributes(db_session)[voices[0]] == {
        "gender": "female",
        "age_bracket": "20_39",
    }


def test_an_unchanged_value_writes_nothing(db_session: Session, voices: list[str]) -> None:
    set_voice_attributes(db_session, voices[0], gender="male", age_bracket=None, annotator="o")
    again = set_voice_attributes(
        db_session, voices[0], gender="male", age_bracket=None, annotator="o"
    )
    assert again is None
    assert db_session.scalar(sa.select(sa.func.count()).select_from(VoiceAttribute)) == 1


def test_clearing_both_fields_leaves_the_voice_unassigned(
    db_session: Session, voices: list[str]
) -> None:
    set_voice_attributes(db_session, voices[0], gender="male", age_bracket="20_39", annotator="o")
    set_voice_attributes(db_session, voices[0], gender=None, age_bracket=None, annotator="o")
    assert current_voice_attributes(db_session) == {}


@pytest.mark.parametrize(
    ("gender", "age"),
    [("non_binary", None), (None, "30_49"), ("Male", None)],
)
def test_a_value_off_the_allowlist_is_refused(
    db_session: Session, voices: list[str], gender, age
) -> None:
    with pytest.raises(VoiceAttributeError):
        set_voice_attributes(db_session, voices[0], gender=gender, age_bracket=age, annotator="o")
    assert db_session.scalar(sa.select(sa.func.count()).select_from(VoiceAttribute)) == 0


def test_a_voice_nobody_is_linked_to_is_refused(db_session: Session, voices: list[str]) -> None:
    with pytest.raises(VoiceAttributeError):
        set_voice_attributes(db_session, "v999", gender="male", age_bracket=None, annotator="o")


# --- API ------------------------------------------------------------------------------------


def test_the_endpoint_sets_and_reports_a_voices_attributes(client, voices: list[str]) -> None:
    response = client.put(
        f"/voices/{voices[1]}/attributes", json={"gender": "female", "age_bracket": "60_79"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "voice": voices[1],
        "gender": "female",
        "age_bracket": "60_79",
        "changed": True,
    }
    again = client.put(
        f"/voices/{voices[1]}/attributes", json={"gender": "female", "age_bracket": "60_79"}
    )
    assert again.json()["changed"] is False


def test_the_endpoint_refuses_bad_values_and_unknown_voices(client, voices: list[str]) -> None:
    bad = client.put(f"/voices/{voices[0]}/attributes", json={"gender": "other"})
    assert bad.status_code == 422
    assert client.put("/voices/v999/attributes", json={"gender": "male"}).status_code == 404
    assert client.put("/voices/nonsense/attributes", json={"gender": "male"}).status_code == 404


def test_the_voice_list_carries_an_assigned_value_over_the_declared_rows(
    client, voices: list[str]
) -> None:
    client.put(f"/voices/{voices[0]}/attributes", json={"gender": "female", "age_bracket": None})
    body = client.get("/voices").json()
    listed = {v["voice"]: v for v in body["voices"]}
    assert listed[voices[0]]["gender"] == "female"
    assert listed[voices[0]]["manual"] == ["gender"]
    assert listed[voices[1]]["gender"] is None
    assert body["summary"]["voices"] == 2
    assert body["summary"]["gender_resolved"] == 1
