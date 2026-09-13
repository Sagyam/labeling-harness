"""Import speaker turns from a diarization run made outside the harness (D78).

The harness does not diarize (D58). A diarizer runs over the retained episode audio on a GPU --
pyannote's ``speaker-diarization-community-1`` today -- and writes one JSON file for many
episodes::

    {"<episode_id>": {"turns": [[start, end, "SPEAKER_00"], ...],
                      "labels": ["SPEAKER_00", ...],            # optional
                      "embeddings": [[...], ...]},              # optional, one per label
     ...}

Turns are episode-relative seconds and may overlap. Anything else in an entry is ignored. Parsing
is a pure function over the payload, so the file is fully validated before a row is written.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session, selectinload

from app.models import AuditLog, DiarizationRun, Episode, SpeakerTurn

Turn = tuple[float, float, str]


class DiarizationError(ValueError):
    """The diarization file is malformed; nothing was written."""


@dataclass(frozen=True)
class EpisodeDiarization:
    """One episode's turns, sorted, with its speakers ordered by talk time."""

    turns: list[Turn]
    speakers: list[str]
    embeddings: dict[str, list[float]] | None

    def checksum(self, model: str) -> str:
        """Identity of this run for this episode: the model and every turn, in order."""
        body = json.dumps([model, self.turns], separators=(",", ":"))
        return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


@dataclass
class ImportReport:
    runs_created: int = 0
    runs_unchanged: int = 0
    turns_inserted: int = 0
    unknown_episodes: list[str] = field(default_factory=list)


def parse_diarization(payload: dict[str, Any]) -> dict[str, EpisodeDiarization]:
    """Validate and normalise a diarization file.

    Raises:
        DiarizationError: An entry has no turns, or a turn is not ``[start, end, speaker]`` with
            ``end > start``.
    """
    if not isinstance(payload, dict):
        raise DiarizationError("expected an object keyed by episode id")
    parsed: dict[str, EpisodeDiarization] = {}
    for episode_id, entry in payload.items():
        raw = entry.get("turns") if isinstance(entry, dict) else None
        if not isinstance(raw, list):
            raise DiarizationError(f"{episode_id}: no 'turns' list")
        turns: list[Turn] = []
        for index, turn in enumerate(raw):
            try:
                start, end, speaker = turn
                start, end, speaker = float(start), float(end), str(speaker)
            except (TypeError, ValueError) as exc:
                raise DiarizationError(
                    f"{episode_id}: turn {index} is not [start, end, speaker]: {turn!r}"
                ) from exc
            if end <= start:
                raise DiarizationError(f"{episode_id}: turn {index} ends before it starts")
            turns.append((round(start, 3), round(end, 3), speaker))
        turns.sort()

        talk: dict[str, float] = defaultdict(float)
        for start, end, speaker in turns:
            talk[speaker] += end - start
        speakers = sorted(talk, key=lambda s: (-talk[s], s))

        embeddings = None
        vectors, labels = entry.get("embeddings"), entry.get("labels")
        if vectors and labels and len(vectors) == len(labels):
            embeddings = {
                str(label): [float(x) for x in vector]
                for label, vector in zip(labels, vectors, strict=True)
                if vector is not None
            }
        parsed[str(episode_id)] = EpisodeDiarization(turns, speakers, embeddings)
    return parsed


def import_diarization(
    session: Session,
    payload: dict[str, Any],
    *,
    model: str,
    source: str | None,
    actor: str,
) -> ImportReport:
    """Store one run per episode the harness knows, skipping runs already imported.

    Args:
        session: Open session; the caller commits.
        payload: The diarization file's parsed JSON.
        model: The diarizer that produced it, recorded on every run.
        source: Where the file came from, for provenance.
        actor: Recorded on each run's ``audit_logs`` row.
    """
    parsed = parse_diarization(payload)
    report = ImportReport()
    episodes = {
        e.external_id: e
        for e in session.scalars(sa.select(Episode).where(Episode.external_id.in_(list(parsed))))
    }
    for episode_id, diarization in sorted(parsed.items()):
        episode = episodes.get(episode_id)
        if episode is None:
            report.unknown_episodes.append(episode_id)
            continue
        checksum = diarization.checksum(model)
        exists = session.scalar(
            sa.select(DiarizationRun.id).where(
                DiarizationRun.episode_id == episode.id, DiarizationRun.checksum == checksum
            )
        )
        if exists is not None:
            report.runs_unchanged += 1
            continue

        run = DiarizationRun(
            episode_id=episode.id,
            model=model,
            source=source,
            checksum=checksum,
            speakers_jsonb=diarization.speakers,
            embeddings_jsonb=diarization.embeddings,
            turns=[
                SpeakerTurn(speaker=s, start_time=a, end_time=b) for a, b, s in diarization.turns
            ],
        )
        session.add(run)
        session.flush()
        session.add(
            AuditLog(
                entity_type="episode",
                entity_id=episode_id,
                action="diarization_import",
                actor=actor,
                new_values_jsonb={
                    "run_id": run.id,
                    "model": model,
                    "turns": len(diarization.turns),
                    "speakers": len(diarization.speakers),
                },
            )
        )
        report.runs_created += 1
        report.turns_inserted += len(diarization.turns)
    session.flush()
    return report


def current_run(session: Session, episode_id: int) -> DiarizationRun | None:
    """The newest diarization run of an episode, turns loaded, or ``None``."""
    return session.scalars(
        sa.select(DiarizationRun)
        .options(selectinload(DiarizationRun.turns))
        .where(DiarizationRun.episode_id == episode_id)
        .order_by(DiarizationRun.id.desc())
        .limit(1)
    ).first()


def segment_speaker_turns(
    session: Session, *, episode_id: int, start: float, end: float
) -> tuple[str | None, list[dict[str, Any]]]:
    """The newest run's model and its turns inside ``[start, end]``, clip-relative and numbered.

    Two small queries: the current run, then only the turns that touch the clip, which the
    ``(run_id, start_time)`` index serves.
    """
    run = session.execute(
        sa.select(DiarizationRun.id, DiarizationRun.model, DiarizationRun.speakers_jsonb)
        .where(DiarizationRun.episode_id == episode_id)
        .order_by(DiarizationRun.id.desc())
        .limit(1)
    ).first()
    if run is None:
        return None, []
    rows = session.execute(
        sa.select(SpeakerTurn.start_time, SpeakerTurn.end_time, SpeakerTurn.speaker)
        .where(
            SpeakerTurn.run_id == run.id,
            SpeakerTurn.start_time < end,
            SpeakerTurn.end_time > start,
        )
        .order_by(SpeakerTurn.start_time, SpeakerTurn.id)
    ).all()
    turns = [(float(a), float(b), str(s)) for a, b, s in rows]
    return run.model, clip_speaker_turns(turns, run.speakers_jsonb, start=start, end=end)


def clip_speaker_turns(
    turns: Sequence[Turn], speakers: Sequence[str], *, start: float, end: float
) -> list[dict[str, Any]]:
    """The turns inside one clip, clip-relative, with each speaker as its display number.

    Speaker numbers come from the run's talk-time order and start at 1, so they are the same in
    every clip of the episode. Turns keep their overlap; a word spoken over two of them is
    crosstalk.
    """
    number = {label: index + 1 for index, label in enumerate(speakers)}
    out: list[dict[str, Any]] = []
    for lo, hi, speaker in turns:
        a, b = max(lo, start), min(hi, end)
        if b > a:
            out.append(
                {
                    "speaker": number.get(speaker, len(number) + 1),
                    "start": round(a - start, 3),
                    "end": round(b - start, 3),
                }
            )
    return out
