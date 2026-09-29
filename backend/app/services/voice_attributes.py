"""A voice's gender and age bracket, assigned by the owner from listening to it (D104).

Declared speaker rows reach a voice only where an episode's shape forces the match (D91), which
left most voices -- every reel, every multi-guest show -- with neither field. The voices page lets
the owner hear a voice and set the two fields directly. What is stored is exactly what a declared
row may carry (D56): two coarse, allowlisted values against an anonymous id, never a name.

Append-only, newest per voice current; every change writes an ``audit_logs`` row, and an unchanged
value writes nothing.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import AuditLog, VoiceAttribute
from app.models.enums import AGE_BRACKETS, GENDERS
from app.services.voice_clips import runs_of_voice

#: The fields a voice can be given, each with its allowed values.
VOICE_FIELDS: dict[str, tuple[str, ...]] = {"gender": GENDERS, "age_bracket": AGE_BRACKETS}


class VoiceAttributeError(ValueError):
    """A value off the allowlist, or a voice no diarized speaker is linked to."""


def current_voice_attributes(session: Session) -> dict[str, dict[str, str | None]]:
    """Each voice's newest assignment, leaving out voices whose newest row clears both fields."""
    latest: dict[str, VoiceAttribute] = {}
    for row in session.scalars(sa.select(VoiceAttribute).order_by(VoiceAttribute.id)):
        latest[row.voice] = row
    return {
        voice: {"gender": row.gender, "age_bracket": row.age_bracket}
        for voice, row in latest.items()
        if row.gender is not None or row.age_bracket is not None
    }


def set_voice_attributes(
    session: Session,
    voice: str,
    *,
    gender: str | None,
    age_bracket: str | None,
    annotator: str,
) -> VoiceAttribute | None:
    """Replace a voice's gender and age bracket; ``None`` for a field leaves it unassigned.

    Returns:
        The new row, or ``None`` when both values already stood and nothing was written.

    Raises:
        VoiceAttributeError: A value off :data:`VOICE_FIELDS`, or no current diarization run
            links a speaker to ``voice``.
    """
    new = {"gender": gender, "age_bracket": age_bracket}
    for key, value in new.items():
        if value is not None and value not in VOICE_FIELDS[key]:
            raise VoiceAttributeError(
                f"{key} {value!r} is not one of {', '.join(VOICE_FIELDS[key])}"
            )
    if not runs_of_voice(session, voice):
        raise VoiceAttributeError(f"no diarized speaker is linked to voice {voice!r}")
    old = current_voice_attributes(session).get(voice, {"gender": None, "age_bracket": None})
    if old == new:
        return None
    row = VoiceAttribute(voice=voice, gender=gender, age_bracket=age_bracket, annotator=annotator)
    session.add(row)
    session.add(
        AuditLog(
            entity_type="voice",
            entity_id=voice,
            action="voice_attributes",
            actor=annotator,
            old_values_jsonb=old,
            new_values_jsonb=new,
        )
    )
    session.flush()
    return row


__all__ = [
    "VOICE_FIELDS",
    "VoiceAttributeError",
    "current_voice_attributes",
    "set_voice_attributes",
]
