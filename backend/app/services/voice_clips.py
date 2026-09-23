"""A voice across the corpus: where it speaks, its solo clips, its print, its suggestions (D99).

A voice is the anonymous id D87 links diarized speakers into (``v125``), and nothing more: no
name is ever attached (D56). This module answers the question the multitrack editor raised --
*what does v125 sound like?* -- with clips the diarization says are that voice alone, lets the
owner confirm or reject them, and turns the confirmed ones into the voice's print.

A print is the normalised mean of the voice's confirmed clips' embeddings, across every episode.
Until a voice has one confirmed clip, the diarizer's own centroid for the speaker stands in: the
two live in one space, and the centroid picked the right speaker on 99% of clean 1 s windows.
"""

from __future__ import annotations

import io
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import (
    AnnotationTask,
    AsrHypothesis,
    AsrSystem,
    AuditLog,
    DiarizationRun,
    Episode,
    HypothesisWord,
    Segment,
    SpeakerTurn,
    VoiceConfirmation,
)
from app.services.diarization_import import clip_speaker_turns
from app.services.speaker_attribution import Lanes
from app.services.speaker_lanes import LaneWord, Turn
from app.services.voiceprint import Suggestion, VoiceEmbedder, suggest, unit, word_window
from app.storage.base import ObjectStorage

#: A stretch is offered as a voice alone when it is at least this long. Whole clips of one voice
#: left 55 of the corpus's 197 voices with nothing to hear -- a voice that only ever talks
#: between or over others never holds a clip -- while its longest stretch alone left 10.
MIN_ALONE_SECONDS = 1.5


class VoiceError(ValueError):
    """A voice request cannot be answered as asked."""


def current_runs(session: Session) -> dict[int, DiarizationRun]:
    """Every diarized episode's newest run, by episode id."""
    runs: dict[int, DiarizationRun] = {}
    for run in session.scalars(sa.select(DiarizationRun).order_by(DiarizationRun.id)):
        runs[run.episode_id] = run
    return runs


def runs_of_voice(session: Session, voice: str) -> list[tuple[DiarizationRun, str]]:
    """The current runs a voice speaks in, each with the speaker label it has there."""
    out = []
    for run in current_runs(session).values():
        for label, linked in (run.voices_jsonb or {}).items():
            if linked == voice:
                out.append((run, label))
    return out


def clip_turns_by_segment(
    session: Session, run: DiarizationRun, segments: Sequence[Segment]
) -> dict[int, list[Turn]]:
    """Each segment's clip-relative, numbered turns, from one read of the run's turns."""
    rows = session.execute(
        sa.select(SpeakerTurn.start_time, SpeakerTurn.end_time, SpeakerTurn.speaker)
        .where(SpeakerTurn.run_id == run.id)
        .order_by(SpeakerTurn.start_time, SpeakerTurn.id)
    ).all()
    if not rows:
        return {s.id: [] for s in segments}
    starts = np.array([float(r[0]) for r in rows])
    ends = np.array([float(r[1]) for r in rows])
    out: dict[int, list[Turn]] = {}
    for segment in segments:
        touching = np.nonzero((starts < segment.end_time) & (ends > segment.start_time))[0]
        turns = clip_speaker_turns(
            [(float(starts[i]), float(ends[i]), str(rows[i][2])) for i in touching],
            run.speakers_jsonb,
            start=segment.start_time,
            end=segment.end_time,
        )
        out[segment.id] = [Turn(t["speaker"], t["start"], t["end"]) for t in turns]
    return out


def alone_stretches(
    turns: Sequence[Turn], overlap_spans: Sequence[Sequence[float]], speaker: int
) -> list[tuple[float, float]]:
    """Where ``speaker`` is diarized and nobody else is, nor any detected crosstalk, in time order.

    The speaker's turns, less every other speaker's turns and every overlap span.
    """
    blocked = sorted(
        [(t.start, t.end) for t in turns if t.speaker != speaker]
        + [(float(a), float(b)) for a, b in overlap_spans]
    )
    out: list[tuple[float, float]] = []
    for start, end in sorted((t.start, t.end) for t in turns if t.speaker == speaker):
        cursor = start
        for a, b in blocked:
            if b <= cursor or a >= end:
                continue
            if a > cursor:
                out.append((cursor, a))
            cursor = max(cursor, b)
            if cursor >= end:
                break
        if cursor < end:
            out.append((cursor, end))
    # Touching turns of one speaker are one stretch.
    merged: list[tuple[float, float]] = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return [(round(a, 3), round(b, 3)) for a, b in merged]


def best_stretch(
    turns: Sequence[Turn], overlap_spans: Sequence[Sequence[float]], speaker: int
) -> tuple[float, float] | None:
    """The longest stretch a clip has of ``speaker`` alone, or None below MIN_ALONE_SECONDS."""
    stretches = alone_stretches(turns, overlap_spans, speaker)
    if not stretches:
        return None
    start, end = max(stretches, key=lambda s: (s[1] - s[0], -s[0]))
    return (start, end) if end - start >= MIN_ALONE_SECONDS - 1e-9 else None


def current_verdicts(
    session: Session, *, voice: str | None = None, segment_ids: Iterable[int] | None = None
) -> dict[tuple[str, int], VoiceConfirmation]:
    """The newest verdict per (voice, segment), ``cleared`` ones left out."""
    query = sa.select(VoiceConfirmation).order_by(VoiceConfirmation.id)
    if voice is not None:
        query = query.where(VoiceConfirmation.voice == voice)
    if segment_ids is not None:
        query = query.where(VoiceConfirmation.segment_id.in_(list(segment_ids)))
    latest: dict[tuple[str, int], VoiceConfirmation] = {}
    for row in session.scalars(query):
        latest[(row.voice, row.segment_id)] = row
    return {k: v for k, v in latest.items() if v.verdict != "cleared"}


# --- the voice page ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceEpisode:
    episode_id: int
    external_id: str
    title: str | None
    speaker_number: int
    talk_seconds: float
    #: Clips with a stretch of this voice alone long enough to offer.
    solo_clips: int


@dataclass(frozen=True)
class VoiceClip:
    """A clip with a stretch of one voice alone: the stretch is what is played and judged."""

    segment_id: int
    external_id: str
    episode_external_id: str
    pot: str
    duration_seconds: float
    text: str | None
    verdict: str | None
    start: float
    end: float
    #: The whole clip is this voice: no other speaker diarized in it, no crosstalk detected.
    whole: bool


@dataclass
class VoicePage:
    voice: str
    episodes: list[VoiceEpisode] = field(default_factory=list)
    clips: list[VoiceClip] = field(default_factory=list)
    total_clips: int = 0
    confirmed: int = 0
    rejected: int = 0
    reference: VoiceClip | None = None

    @property
    def talk_seconds(self) -> float:
        return round(sum(e.talk_seconds for e in self.episodes), 1)

    @property
    def print_source(self) -> str:
        return "confirmed" if self.confirmed else "diarizer"


def voice_page(
    session: Session, voice: str, *, episode: str | None = None, limit: int = 60
) -> VoicePage:
    """Where ``voice`` speaks and the clips it speaks alone in, longest first.

    Args:
        voice: The anonymous voice id.
        episode: Only this episode's clips (external id); every episode is still listed.
        limit: How many clips to return; ``total_clips`` counts them all.

    Raises:
        VoiceError: No current diarization run links any speaker to ``voice``.
    """
    runs = runs_of_voice(session, voice)
    if not runs:
        raise VoiceError(f"no diarized speaker is linked to voice {voice!r}")
    verdicts = current_verdicts(session, voice=voice)
    page = VoicePage(voice=voice)
    page.confirmed = sum(v.verdict == "confirmed" for v in verdicts.values())
    page.rejected = sum(v.verdict == "rejected" for v in verdicts.values())

    candidates: list[tuple[Segment, Episode, tuple[float, float]]] = []
    for run, label in runs:
        number = run.speakers_jsonb.index(label) + 1
        ep = session.get(Episode, run.episode_id)
        segments = list(
            session.scalars(
                sa.select(Segment).where(Segment.episode_id == run.episode_id).order_by(Segment.id)
            )
        )
        turns = clip_turns_by_segment(session, run, segments)
        offered = [
            (s, stretch)
            for s in segments
            if (stretch := best_stretch(turns[s.id], s.overlap_spans_jsonb or [], number))
        ]
        talk = session.scalar(
            sa.select(
                sa.func.coalesce(sa.func.sum(SpeakerTurn.end_time - SpeakerTurn.start_time), 0)
            ).where(SpeakerTurn.run_id == run.id, SpeakerTurn.speaker == label)
        )
        page.episodes.append(
            VoiceEpisode(
                episode_id=ep.id,
                external_id=ep.external_id,
                title=ep.title,
                speaker_number=number,
                talk_seconds=round(float(talk or 0.0), 1),
                solo_clips=len(offered),
            )
        )
        if episode is None or ep.external_id == episode:
            candidates.extend((s, ep, stretch) for s, stretch in offered)

    page.episodes.sort(key=lambda e: -e.talk_seconds)
    # Confirmed first, then stretches nobody has judged, rejected last; longest first within each.
    rank = {"confirmed": 0, None: 1, "rejected": 2}
    candidates.sort(
        key=lambda c: (
            rank[_verdict(verdicts, voice, c[0].id)],
            -(c[2][1] - c[2][0]),
            c[0].id,
        )
    )
    page.total_clips = len(candidates)
    chosen = candidates[:limit]
    texts = _stretch_texts(session, [(s.id, stretch) for s, _, stretch in chosen])
    page.clips = [
        VoiceClip(
            segment_id=s.id,
            external_id=s.external_id,
            episode_external_id=ep.external_id,
            pot=s.pot,
            duration_seconds=s.duration_seconds,
            text=texts.get(s.id),
            verdict=_verdict(verdicts, voice, s.id),
            start=start,
            end=end,
            whole=end - start >= s.duration_seconds - 0.25,
        )
        for s, ep, (start, end) in chosen
    ]
    page.reference = next(
        (c for c in page.clips if c.verdict == "confirmed"),
        next((c for c in page.clips if c.verdict is None), None),
    )
    return page


def _verdict(verdicts: dict[tuple[str, int], VoiceConfirmation], voice: str, segment_id: int):
    row = verdicts.get((voice, segment_id))
    return row.verdict if row else None


def _stretch_texts(
    session: Session, chosen: Sequence[tuple[int, tuple[float, float]]]
) -> dict[int, str]:
    """What is said in each stretch: the fused seed's words whose middle falls inside it.

    The seed rather than the verified label, because only the seed's words carry spans; a stretch
    is usually part of a clip, and the clip's whole text would not be what is played.
    """
    if not chosen:
        return {}
    stretches = dict(chosen)
    rows = session.execute(
        sa.select(
            AsrHypothesis.segment_id,
            AsrHypothesis.asr_system_id,
            HypothesisWord.word_raw,
            HypothesisWord.start_time,
            HypothesisWord.end_time,
        )
        .join(AsrSystem, AsrSystem.id == AsrHypothesis.asr_system_id)
        .join(HypothesisWord, HypothesisWord.hypothesis_id == AsrHypothesis.id)
        .where(AsrHypothesis.segment_id.in_(list(stretches)), AsrSystem.kind == "fusion")
        .order_by(AsrHypothesis.segment_id, AsrHypothesis.asr_system_id, HypothesisWord.position)
    ).all()
    newest = {}
    for segment_id, system_id, *_ in rows:
        newest[segment_id] = max(newest.get(segment_id, system_id), system_id)
    words: dict[int, list[str]] = defaultdict(list)
    for segment_id, system_id, word, start, end in rows:
        if system_id != newest[segment_id] or start is None or end is None:
            continue
        a, b = stretches[segment_id]
        if a <= (start + end) / 2 < b:
            words[segment_id].append(word)
    return {segment_id: " ".join(w) for segment_id, w in words.items() if w}


# --- verdicts ---------------------------------------------------------------------------------


def read_clip(storage: ObjectStorage, segment: Segment) -> np.ndarray:
    """A clip's samples, mono float32 at 16 kHz (invariant 7)."""
    samples, _ = sf.read(io.BytesIO(storage.get_bytes(segment.clip_object_key)), dtype="float32")
    return samples if samples.ndim == 1 else samples.mean(axis=1)


def record_verdict(
    session: Session,
    voice: str,
    segment_id: int,
    verdict: str,
    *,
    annotator: str,
    storage: ObjectStorage,
    embedder: VoiceEmbedder,
) -> VoiceConfirmation:
    """Store the owner's verdict on a clip's stretch of ``voice`` alone, with its audit row.

    The stretch is recomputed here, the same one the voice page played, rather than taken from
    the client. A confirmed stretch is embedded, when the model is available, and becomes part of
    the voice's print. The clip must be from an episode where the voice is diarized.

    Raises:
        VoiceError: Unknown segment, unknown verdict, or the voice does not speak in its episode.
    """
    if verdict not in ("confirmed", "rejected", "cleared"):
        raise VoiceError(f"unknown verdict {verdict!r}")
    segment = session.get(Segment, segment_id)
    if segment is None:
        raise VoiceError(f"no segment {segment_id}")
    match = [
        (run, label)
        for run, label in runs_of_voice(session, voice)
        if run.episode_id == segment.episode_id
    ]
    if not match:
        raise VoiceError(f"voice {voice} is not diarized in the episode of {segment.external_id}")
    run, label = match[0]
    number = run.speakers_jsonb.index(label) + 1
    turns = clip_turns_by_segment(session, run, [segment])[segment.id]
    stretch = best_stretch(turns, segment.overlap_spans_jsonb or [], number)
    if stretch is None:
        raise VoiceError(
            f"{segment.external_id} has no stretch of {voice} alone to judge"
            f" ({MIN_ALONE_SECONDS} s or longer)"
        )
    embedding = None
    if verdict == "confirmed" and embedder.available:
        samples = read_clip(storage, segment)
        chunk = samples[int(stretch[0] * 16_000) : int(stretch[1] * 16_000)]
        vector = embedder.embed([chunk])[0]
        embedding = None if vector is None else [round(float(x), 6) for x in vector]
    row = VoiceConfirmation(
        voice=voice,
        segment_id=segment.id,
        diarization_run_id=run.id,
        speaker=label,
        start_time=stretch[0],
        end_time=stretch[1],
        verdict=verdict,
        embedding_jsonb=embedding,
        annotator=annotator,
    )
    session.add(row)
    session.flush()
    session.add(
        AuditLog(
            entity_type="voice_confirmations",
            entity_id=str(row.id),
            action="insert",
            actor=annotator,
            new_values_jsonb={
                "voice": voice,
                "segment_id": segment.id,
                "verdict": verdict,
                "start_time": stretch[0],
                "end_time": stretch[1],
                "embedded": embedding is not None,
            },
        )
    )
    session.flush()
    return row


# --- prints and suggestions -------------------------------------------------------------------


@dataclass(frozen=True)
class VoicePrint:
    vector: np.ndarray
    #: ``confirmed`` (the owner's clips) or ``diarizer`` (the run's centroid).
    source: str
    clips: int = 0


def voice_prints(session: Session, run: DiarizationRun) -> dict[int, VoicePrint]:
    """A print for each of the run's speakers that has one, by display number."""
    voices = run.voices_jsonb or {}
    confirmed: dict[str, list[np.ndarray]] = defaultdict(list)
    wanted = set(voices.values())
    if wanted:
        for (voice, _), row in current_verdicts(session).items():
            if voice in wanted and row.verdict == "confirmed" and row.embedding_jsonb:
                vector = unit(row.embedding_jsonb)
                if vector is not None:
                    confirmed[voice].append(vector)
    prints: dict[int, VoicePrint] = {}
    for index, label in enumerate(run.speakers_jsonb):
        own = confirmed.get(voices.get(label, ""), [])
        if own:
            mean = unit(np.mean(own, axis=0))
            if mean is not None:
                prints[index + 1] = VoicePrint(mean, "confirmed", len(own))
                continue
        centroid = unit((run.embeddings_jsonb or {}).get(label) or [])
        if centroid is not None:
            prints[index + 1] = VoicePrint(centroid, "diarizer")
    return prints


def overlapped(word: LaneWord, turns: Sequence[Turn], spans: Sequence[Sequence[float]]) -> bool:
    """Whether a second voice is heard at the word's middle: two turns, or detected crosstalk."""
    mid = ((word.start or 0.0) + (word.end or 0.0)) / 2
    active = {t.speaker for t in turns if t.start <= mid < t.end}
    return len(active) > 1 or any(a <= mid < b for a, b in spans)


def lane_suggestions(
    words: Sequence[LaneWord],
    *,
    samples: np.ndarray,
    turns: Sequence[Turn],
    overlap_spans: Sequence[Sequence[float]],
    candidates: Sequence[int],
    prints: dict[int, VoicePrint],
    embedder: VoiceEmbedder,
) -> list[Suggestion | None]:
    """A voiceprint suggestion per word, aligned with ``words``; None where there is none.

    Only words no second voice is heard over are embedded, each over a 1 s window centred on it,
    and scored against the prints of the clip's speakers.
    """
    usable = [c for c in candidates if c in prints]
    out: list[Suggestion | None] = [None] * len(words)
    if not usable or not embedder.available:
        return out
    duration = len(samples) / 16_000
    todo = [
        i
        for i, w in enumerate(words)
        if w.start is not None and w.end is not None and not overlapped(w, turns, overlap_spans)
    ]
    chunks = []
    for i in todo:
        left, right = word_window(float(words[i].start), float(words[i].end), duration)  # type: ignore[arg-type]
        chunks.append(samples[int(left * 16_000) : int(right * 16_000)])
    for i, vector in zip(todo, embedder.embed(chunks), strict=True):
        if vector is None:
            continue
        sims = {c: float(vector @ prints[c].vector) for c in usable}
        out[i] = suggest(sims, current=words[i].speaker, overlapped=False)
    return out


def suggest_for_task(
    session: Session,
    task: AnnotationTask,
    lanes: Lanes,
    *,
    storage: ObjectStorage,
    embedder: VoiceEmbedder,
) -> tuple[list[Suggestion | None], dict[int, VoicePrint]]:
    """Voiceprint suggestions for a speakers-queue task's lanes, and the prints behind them.

    The clip's speakers are the candidates; a clip with no diarized turn falls back to every
    speaker of the episode.
    """
    run = session.get(DiarizationRun, lanes.diarization_run_id)
    if run is None:
        return [None] * len(lanes.words), {}
    prints = voice_prints(session, run)
    if not embedder.available:
        return [None] * len(lanes.words), prints
    segment = task.segment
    suggestions = lane_suggestions(
        lanes.words,
        samples=read_clip(storage, segment),
        turns=lanes.turns,
        overlap_spans=segment.overlap_spans_jsonb or [],
        candidates=lanes.clip_speakers or [s.number for s in lanes.speakers],
        prints=prints,
        embedder=embedder,
    )
    return suggestions, prints
