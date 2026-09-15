"""Link diarized speakers across episodes into anonymous voices (D87).

A diarization run's labels (``SPEAKER_00``) mean nothing outside that run (D78). Its per-speaker
embeddings do: pyannote's ``speaker-diarization-community-1`` gives every speaker a 256-d voice
embedding, stored with the run. A speaker whose embedding is within :data:`LINK_THRESHOLD` cosine
of a known voice's centroid is that voice; anyone else is a new one.

On the 44 episodes of 2026-09-15 the cross-episode similarities split cleanly: the widest gap
runs from 0.53 to 0.66, and different speakers of one episode never passed 0.59. Linking at 0.5,
0.6 or 0.7 gives the same 32 voices, with the same recurring hosts as the EDA's independent
ECAPA voice prints (one tech-review host across 26 episodes, three podcast series hosts).

A voice is an anonymous id (``v001``) and nothing else: no name, no inferred gender or age
(D56, D58). Ids are handed out in run order, so linking new runs never renames an old voice;
``relink`` starts over and may.

Linking is incremental: each unlinked run is matched against the centroids of every voice linked
so far, which is exactly what ingest does for one new episode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import AuditLog, DiarizationRun

#: Cosine similarity at which a speaker is a known voice. The middle of the widest gap in the
#: corpus's cross-episode similarities (0.53 -> 0.66), and close to the 0.587 the EDA read off
#: its own ECAPA prints.
LINK_THRESHOLD = 0.6


@dataclass
class LinkReport:
    runs_linked: int = 0
    speakers_linked: int = 0
    voices: int = 0


def _unit(vector: Sequence[float]) -> np.ndarray | None:
    v = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(v)
    return v / norm if v.size and np.isfinite(norm) and norm > 0 else None


def _voice_id(number: int) -> str:
    return f"v{number:03d}"


def assign_voices(
    embeddings: Mapping[str, Sequence[float]],
    speakers: Sequence[str],
    centroids: dict[str, np.ndarray],
) -> dict[str, str]:
    """Give one run's speakers voices, and fold them into ``centroids``.

    The most similar (speaker, voice) pairs are matched first, each speaker and each voice at
    most once: the diarizer already told this run's speakers apart, so two of them never share a
    voice. Speakers left over start new voices, in talk-time order.

    Args:
        embeddings: The run's embeddings by label.
        speakers: The run's labels, most talk time first; a label without an embedding gets no
            voice.
        centroids: Voice id to the running sum of its members' unit embeddings. Updated in place.

    Returns:
        Label to voice id, for every speaker that has an embedding.
    """
    vectors = {s: v for s in speakers if (v := _unit(embeddings.get(s) or [])) is not None}
    pairs = sorted(
        (
            (float(vec @ (total / np.linalg.norm(total))), speaker, voice)
            for speaker, vec in vectors.items()
            for voice, total in centroids.items()
        ),
        key=lambda p: (-p[0], p[1], p[2]),
    )
    out: dict[str, str] = {}
    taken: set[str] = set()
    for similarity, speaker, voice in pairs:
        if similarity < LINK_THRESHOLD:
            break
        if speaker not in out and voice not in taken:
            out[speaker] = voice
            taken.add(voice)
    number = max((int(v[1:]) for v in centroids), default=0)
    for speaker in vectors:
        if speaker not in out:
            number += 1
            out[speaker] = _voice_id(number)
    for speaker, voice in out.items():
        centroids[voice] = centroids.get(voice, 0) + vectors[speaker]
    return out


def link_voices(session: Session, *, actor: str, relink: bool = False) -> LinkReport:
    """Link every diarization run not linked yet, oldest first.

    Args:
        session: Open session; the caller commits.
        actor: Recorded on each run's ``audit_logs`` row.
        relink: Forget every link and start over from the first run.
    """
    runs = list(session.scalars(sa.select(DiarizationRun).order_by(DiarizationRun.id)))
    if relink:
        for run in runs:
            run.voices_jsonb = None
    report = LinkReport()
    centroids: dict[str, np.ndarray] = {}
    for run in runs:
        if run.voices_jsonb is not None:
            for speaker, voice in run.voices_jsonb.items():
                vector = _unit((run.embeddings_jsonb or {}).get(speaker) or [])
                if vector is not None:
                    centroids[voice] = centroids.get(voice, 0) + vector
            continue
        voices = assign_voices(run.embeddings_jsonb or {}, run.speakers_jsonb, centroids)
        run.voices_jsonb = voices
        report.runs_linked += 1
        report.speakers_linked += len(voices)
        session.add(
            AuditLog(
                entity_type="diarization_run",
                entity_id=str(run.id),
                action="voice_link",
                actor=actor,
                new_values_jsonb={"voices": voices, "threshold": LINK_THRESHOLD},
            )
        )
    report.voices = len(centroids)
    session.flush()
    return report
