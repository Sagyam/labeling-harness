"""The multitrack editor against the database: serving lanes, saving them, queueing clips (D98).

:mod:`app.services.speaker_lanes` is the pure half. This module loads what it needs -- the clip's
verified text, its seed's word spans, its recognisers and the episode's current diarization run --
and writes a per-speaker label through :func:`app.services.labeling.record_decision`, so a save
is the same three rows in one transaction as every other decision.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import sqlalchemy as sa
from sqlalchemy.orm import Session, selectinload

from app.config import Settings, get_settings
from app.models import (
    AnnotationEvent,
    AnnotationTask,
    AsrHypothesis,
    AuditLog,
    DiarizationRun,
    LabelVersion,
    LabelWord,
    Segment,
    SegmentLabel,
    SpeakerTurn,
)
from app.models.enums import ACTIVE_TASK_STATUSES, APPROVED_DISPOSITIONS, SPEAKERS_QUEUE
from app.services.clip_classes import overlap_share
from app.services.consensus import ConsensusHypothesis, ConsensusWord
from app.services.diarization_import import clip_speaker_turns
from app.services.labeling import (
    Decision,
    LabelingError,
    SpeakerWord,
    latest_label,
    record_decision,
)
from app.services.queue_builder import select_seed_hypothesis
from app.services.speaker_lanes import (
    Candidate,
    LaneWord,
    TimedWord,
    Turn,
    check_attribution,
    flatten,
    in_time_order,
    is_unchanged,
    place_text_on_clock,
    propose_lanes,
    recogniser_candidates,
)
from app.services.stats import latest_labels_subquery


@dataclass(frozen=True)
class LaneSpeaker:
    """One of the episode's speakers: its display number, the run's raw label, and its voice."""

    number: int
    label: str
    voice: str | None


@dataclass(frozen=True)
class Lanes:
    """Everything the multitrack editor opens a clip with.

    Attributes:
        diarization_run_id: The run the speakers belong to. A save names it, and is refused if a
            newer run has been imported since.
        speakers: Every speaker of the episode, in display order. These are the only lanes there
            can be: speakers are tracked globally, so the editor cannot create one.
        clip_speakers: The display numbers diarized inside this clip, shown as lanes by default.
        words: The starting blocks.
        candidates: Recogniser words the verified text lacks, offered to be dragged onto a lane.
        base: What ``words`` came from: an earlier per-speaker label (``speakers``), the clip's
            verified single-stream label (``label``), or its seed when it has none (``seed``).
    """

    diarization_run_id: int
    diarization_model: str
    speakers: list[LaneSpeaker]
    clip_speakers: list[int]
    turns: list[Turn]
    words: list[LaneWord]
    candidates: list[Candidate]
    base: str


def current_run(session: Session, episode_id: int) -> DiarizationRun | None:
    """The episode's newest diarization run, or None when it was never diarized."""
    return session.scalars(
        sa.select(DiarizationRun)
        .where(DiarizationRun.episode_id == episode_id)
        .order_by(DiarizationRun.id.desc())
        .limit(1)
    ).first()


def _clip_turns(session: Session, run: DiarizationRun, segment: Segment) -> list[Turn]:
    rows = session.execute(
        sa.select(SpeakerTurn.start_time, SpeakerTurn.end_time, SpeakerTurn.speaker)
        .where(
            SpeakerTurn.run_id == run.id,
            SpeakerTurn.start_time < segment.end_time,
            SpeakerTurn.end_time > segment.start_time,
        )
        .order_by(SpeakerTurn.start_time, SpeakerTurn.id)
    ).all()
    turns = clip_speaker_turns(
        [(float(a), float(b), str(s)) for a, b, s in rows],
        run.speakers_jsonb,
        start=segment.start_time,
        end=segment.end_time,
    )
    return [Turn(t["speaker"], t["start"], t["end"]) for t in turns]


def _speakers(run: DiarizationRun) -> list[LaneSpeaker]:
    voices = run.voices_jsonb or {}
    return [
        LaneSpeaker(number=index + 1, label=label, voice=voices.get(label))
        for index, label in enumerate(run.speakers_jsonb)
    ]


def _clock(seed: AsrHypothesis | None, hypotheses: Sequence[AsrHypothesis]) -> list[TimedWord]:
    """The word spans the verified text is put on: the seed's, or a recogniser's without them.

    The fused seed's spans are measured by the aligner, which can fail on a clip; the recogniser
    with the most timed words stands in rather than leaving every block unplaced.
    """

    def timed(h: AsrHypothesis) -> int:
        return sum(1 for w in h.words if w.start_time is not None and w.end_time is not None)

    options = [seed] if seed is not None else []
    options += sorted((h for h in hypotheses if h.system.kind == "asr"), key=lambda h: -timed(h))
    for hypothesis in options:
        if timed(hypothesis):
            return [TimedWord(w.word_raw, w.start_time, w.end_time) for w in hypothesis.words]
    return []


def latest_speakers_label(
    session: Session, segment_id: int, settings: Settings | None = None
) -> SegmentLabel | None:
    """The clip's newest per-speaker label, words loaded, or None."""
    settings = settings or get_settings()
    version_id = session.scalar(
        sa.select(LabelVersion.id).where(
            LabelVersion.name == settings.labels.speakers_label_version
        )
    )
    if version_id is None:
        return None
    return session.scalars(
        sa.select(SegmentLabel)
        .options(selectinload(SegmentLabel.words))
        .where(SegmentLabel.segment_id == segment_id, SegmentLabel.label_version_id == version_id)
        .order_by(SegmentLabel.created_at.desc(), SegmentLabel.id.desc())
        .limit(1)
    ).first()


def build_lanes(
    session: Session, task: AnnotationTask, settings: Settings | None = None
) -> Lanes | None:
    """The lanes a speakers-queue task opens with, or None when the episode has no diarization.

    An earlier per-speaker label of the same run is reopened as it was saved. Otherwise the
    clip's verified text is put on its seed's clock and every word on the lane of the turn that
    covers most of it; without a verified text, the seed's own words are used.
    """
    settings = settings or get_settings()
    segment = task.segment
    run = current_run(session, segment.episode_id)
    if run is None:
        return None
    speakers = _speakers(run)
    number = {s.label: s.number for s in speakers}
    turns = _clip_turns(session, run, segment)

    hypotheses = list(segment.hypotheses)
    seed = task.seed_hypothesis or select_seed_hypothesis(hypotheses)
    recognisers = [
        ConsensusHypothesis(
            system_id=h.system.system_id,
            words=[
                ConsensusWord(w.position, w.word_raw, w.start_time, w.end_time) for w in h.words
            ],
        )
        for h in hypotheses
        if h.system.kind == "asr"
    ]

    previous = latest_speakers_label(session, segment.id, settings)
    if previous is not None and previous.diarization_run_id == run.id and previous.words:
        words = [
            LaneWord(
                word=w.word,
                start=w.start_time,
                end=w.end_time,
                speaker=number.get(w.speaker),
                proposed_speaker=number.get(w.proposed_speaker) if w.proposed_speaker else None,
                source=w.source,
                suggested_speaker=(
                    number.get(w.suggested_speaker) if w.suggested_speaker else None
                ),
            )
            for w in previous.words
        ]
        placed = [TimedWord(w.word, w.start, w.end) for w in words]
        base = "speakers"
    else:
        label = latest_label(session, segment.id)
        if label is not None and label.disposition in APPROVED_DISPOSITIONS and label.final_text:
            text, base = label.final_text, "label"
        else:
            text, base = (seed.text_raw if seed else ""), "seed"
        placed = place_text_on_clock(text, _clock(seed, hypotheses))
        words = propose_lanes(placed, turns)

    return Lanes(
        diarization_run_id=run.id,
        diarization_model=run.model,
        speakers=speakers,
        clip_speakers=sorted({t.speaker for t in turns}),
        turns=turns,
        words=words,
        candidates=recogniser_candidates(placed, recognisers),
        base=base,
    )


def record_attribution(
    session: Session,
    task: AnnotationTask,
    words: Sequence[LaneWord],
    *,
    diarization_run_id: int,
    annotator: str | None = None,
    notes: str | None = None,
    opened_at: dt.datetime | None = None,
    duration_ms: int | None = None,
    settings: Settings | None = None,
) -> SegmentLabel:
    """Store the lanes as a per-speaker label: label, words, event and audit in one transaction.

    Raises:
        LabelingError: The task is not in the speakers queue, the episode was re-diarized since
            the lanes were served, or a word cannot be stored (see
            :func:`app.services.speaker_lanes.check_attribution`).
    """
    settings = settings or get_settings()
    if task.queue != SPEAKERS_QUEUE:
        raise LabelingError(f"task {task.id} is not in the speakers queue")
    lanes = build_lanes(session, task, settings)
    if lanes is None:
        raise LabelingError(f"task {task.id}: the episode has no diarization to attribute to")
    if lanes.diarization_run_id != diarization_run_id:
        raise LabelingError(
            f"task {task.id}: the episode was re-diarized since this clip was opened; reopen it"
        )
    problem = check_attribution(
        words,
        clip_seconds=task.segment.duration_seconds,
        speakers=[s.number for s in lanes.speakers],
    )
    if problem:
        raise LabelingError("; ".join(problem.errors[:5]))

    label_of = {s.number: s.label for s in lanes.speakers}
    ordered = in_time_order(words)
    return record_decision(
        session,
        task,
        Decision(
            disposition="accepted_unchanged" if is_unchanged(lanes.words, words) else "edited",
            final_text=flatten(ordered),
            notes=notes,
            annotator=annotator,
            opened_at=opened_at,
            duration_ms=duration_ms,
            diarization_run_id=diarization_run_id,
            words=[
                SpeakerWord(
                    word=w.word.strip(),
                    start=round(float(w.start), 3),  # type: ignore[arg-type]
                    end=round(float(w.end), 3),  # type: ignore[arg-type]
                    speaker=label_of[w.speaker],  # type: ignore[index]
                    proposed_speaker=(
                        label_of.get(w.proposed_speaker) if w.proposed_speaker is not None else None
                    ),
                    source=w.source,
                    suggested_speaker=(
                        label_of.get(w.suggested_speaker)
                        if w.suggested_speaker is not None
                        else None
                    ),
                )
                for w in ordered
            ],
        ),
        settings=settings,
    )


# --- which clips get lanes --------------------------------------------------------------------


@dataclass
class SpeakersQueueReport:
    """What queueing clips for the multitrack editor did, or would have done."""

    queued: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False

    def render(self) -> str:
        lines = [
            f"{'DRY RUN -- ' if self.dry_run else ''}speakers queue: {len(self.queued)} queued,"
            f" {len(self.skipped)} skipped"
        ]
        lines += [f"  + {external_id}" for external_id in self.queued]
        lines += [f"  - {external_id}: {why}" for external_id, why in self.skipped.items()]
        return "\n".join(lines)


@dataclass(frozen=True)
class SpeakersCandidate:
    """A clip that could go to the multitrack editor, and why it ranks where it does."""

    segment: Segment
    overlap_share: float | None
    clip_speakers: int


def rank_for_speakers(
    session: Session,
    *,
    pot: str = "gold",
    settings: Settings | None = None,
    min_overlap: float | None = None,
    max_overlap: float | None = None,
    min_clip_speakers: int = 2,
) -> list[SpeakersCandidate]:
    """Clips worth attributing, most crosstalk first.

    A clip qualifies when it has a verified single-stream label to start from, its episode has a
    diarization run, and two or more speakers are diarized inside it: a one-speaker clip is one
    lane and no work. Clips already attributed or already in a queue are left out. The ranking is
    the overlap detector's crosstalk share (D77), unmeasured last.

    ``min_overlap`` and ``max_overlap`` keep only clips whose crosstalk share lies in that band,
    inclusive; a clip never measured is outside every band. ``min_clip_speakers`` of 1 also takes
    clips where crosstalk was detected but the diarizer heard one voice -- the case a check of the
    diarizer is after.
    """
    settings = settings or get_settings()
    current = latest_labels_subquery()
    verified = sa.select(current.c.segment_id).where(
        current.c.disposition.in_(APPROVED_DISPOSITIONS)
    )
    speakers_version = sa.select(LabelVersion.id).where(
        LabelVersion.name == settings.labels.speakers_label_version
    )
    segments = list(
        session.scalars(
            sa.select(Segment)
            .where(
                Segment.pot == pot,
                Segment.id.in_(verified),
                ~Segment.id.in_(
                    sa.select(SegmentLabel.segment_id).where(
                        SegmentLabel.label_version_id.in_(speakers_version)
                    )
                ),
                ~Segment.id.in_(
                    sa.select(AnnotationTask.segment_id).where(
                        AnnotationTask.status.in_(ACTIVE_TASK_STATUSES)
                    )
                ),
            )
            .order_by(Segment.id)
        )
    )

    by_episode: dict[int, list[Segment]] = defaultdict(list)
    for segment in segments:
        by_episode[segment.episode_id].append(segment)

    ranked: list[SpeakersCandidate] = []
    for episode_id, clips in by_episode.items():
        run = current_run(session, episode_id)
        if run is None:
            continue
        for segment in clips:
            speakers = {t.speaker for t in _clip_turns(session, run, segment)}
            if len(speakers) < min_clip_speakers:
                continue
            ranked.append(
                SpeakersCandidate(
                    segment=segment,
                    overlap_share=overlap_share(
                        segment.overlap_spans_jsonb, segment.duration_seconds
                    ),
                    clip_speakers=len(speakers),
                )
            )
    if min_overlap is not None or max_overlap is not None:
        low = -1.0 if min_overlap is None else min_overlap - 1e-9
        high = 2.0 if max_overlap is None else max_overlap + 1e-9
        ranked = [
            c for c in ranked if c.overlap_share is not None and low <= c.overlap_share <= high
        ]
    ranked.sort(key=lambda c: (c.overlap_share is None, -(c.overlap_share or 0.0), c.segment.id))
    return ranked


def queue_for_speakers(
    session: Session,
    candidates: Sequence[SpeakersCandidate],
    *,
    actor: str,
    dry_run: bool = False,
) -> SpeakersQueueReport:
    """Open a speakers-queue task for each clip, with an audit row per task.

    The clip's pot, split, pipeline status and single-stream label are untouched: a gold clip
    stays gold and stays labelled. Its priority is its crosstalk share, so triage and
    ``/tasks/next`` serve the heaviest overlap first.
    """
    report = SpeakersQueueReport(dry_run=dry_run)
    for candidate in candidates:
        segment = candidate.segment
        active = session.scalar(
            sa.select(AnnotationTask.id).where(
                AnnotationTask.segment_id == segment.id,
                AnnotationTask.status.in_(ACTIVE_TASK_STATUSES),
            )
        )
        if active is not None:
            report.skipped[segment.external_id] = f"already has active task {active}"
            continue
        report.queued.append(segment.external_id)
        if dry_run:
            continue
        seed = select_seed_hypothesis(list(segment.hypotheses))
        task = AnnotationTask(
            segment_id=segment.id,
            queue=SPEAKERS_QUEUE,
            priority_score=round(candidate.overlap_share or 0.0, 4),
            seed_hypothesis_id=seed.id if seed else None,
            reason_jsonb={
                "speakers": {
                    "overlap_share": candidate.overlap_share,
                    "clip_speakers": candidate.clip_speakers,
                }
            },
            status="pending",
        )
        session.add(task)
        session.flush()
        session.add(
            AuditLog(
                entity_type="annotation_tasks",
                entity_id=str(task.id),
                action="queue_speakers",
                actor=actor,
                old_values_jsonb=None,
                new_values_jsonb={
                    "segment_id": segment.id,
                    "queue": SPEAKERS_QUEUE,
                    "overlap_share": candidate.overlap_share,
                    "clip_speakers": candidate.clip_speakers,
                },
            )
        )
    session.flush()
    return report


def saved_speaker_words(
    session: Session, settings: Settings | None = None
) -> list[tuple[str | None, str | None, str | None, str]]:
    """Every word of every clip's current per-speaker label, as the suggestion report reads it:
    ``(speaker, proposed_speaker, suggested_speaker, source)`` (D99)."""
    settings = settings or get_settings()
    version_id = session.scalar(
        sa.select(LabelVersion.id).where(
            LabelVersion.name == settings.labels.speakers_label_version
        )
    )
    if version_id is None:
        return []
    ranked = (
        sa.select(
            SegmentLabel.id,
            sa.func.row_number()
            .over(
                partition_by=SegmentLabel.segment_id,
                order_by=(SegmentLabel.created_at.desc(), SegmentLabel.id.desc()),
            )
            .label("rank"),
        )
        .where(SegmentLabel.label_version_id == version_id)
        .subquery()
    )
    current = sa.select(ranked.c.id).where(ranked.c.rank == 1)
    return [
        (w.speaker, w.proposed_speaker, w.suggested_speaker, w.source)
        for w in session.scalars(sa.select(LabelWord).where(LabelWord.label_id.in_(current)))
    ]


def reopen_for_speakers(session: Session, segment: Segment, *, actor: str) -> AnnotationTask:
    """Put a clip whose per-speaker label was saved back in the speakers queue.

    Nothing is deleted: the saved label stays, current until the next save supersedes it
    (invariant 2), and the editor opens from it. The reopening writes an ``annotation_events``
    row with the ``reopen`` action and an audit row.

    Raises:
        LabelingError: The clip has no per-speaker label, or already has an open task.
    """
    label = latest_speakers_label(session, segment.id)
    if label is None:
        raise LabelingError(f"{segment.external_id} has no per-speaker label to reopen")
    active = session.scalar(
        sa.select(AnnotationTask.id).where(
            AnnotationTask.segment_id == segment.id,
            AnnotationTask.status.in_(ACTIVE_TASK_STATUSES),
        )
    )
    if active is not None:
        raise LabelingError(f"{segment.external_id} already has an open task ({active})")
    seed = select_seed_hypothesis(list(segment.hypotheses))
    task = AnnotationTask(
        segment_id=segment.id,
        queue=SPEAKERS_QUEUE,
        priority_score=round(
            overlap_share(segment.overlap_spans_jsonb, segment.duration_seconds) or 0.0, 4
        ),
        seed_hypothesis_id=seed.id if seed else None,
        reason_jsonb={"speakers": {"reopened_label_id": label.id}},
        status="pending",
    )
    session.add(task)
    session.flush()
    session.add(
        AnnotationEvent(
            task_id=task.id,
            segment_id=segment.id,
            annotator=actor,
            action="reopen",
        )
    )
    session.add(
        AuditLog(
            entity_type="annotation_tasks",
            entity_id=str(task.id),
            action="reopen",
            actor=actor,
            old_values_jsonb={"label_id": label.id, "disposition": label.disposition},
            new_values_jsonb={"segment_id": segment.id, "queue": SPEAKERS_QUEUE},
        )
    )
    session.flush()
    return task
