"""Time-aligned agreement between the ASR systems that transcribed one clip.

Every system heard the same audio, so their word spans are observations of a single timeline.
Putting them on that timeline and grouping by overlap gives the correspondence directly, where a
string comparison has to infer it -- and inferring it goes wrong on exactly the segments worth
looking at, because a repeated word can match the wrong occurrence and a re-ordering scores as
two errors.

The unit here is a **slot**: one stretch of time that one or more systems put a word in. A slot
where every system agrees on the token is settled; a slot where they do not is where the
annotator's attention is worth something. Word spans are clip-relative by construction (D26), so
no offset arithmetic is needed to compare two hypotheses of the same segment.

This is deliberately computed rather than stored. It is a pure function of ``hypothesis_words``,
which is immutable once imported, so the answer cannot go stale, and keeping it out of the schema
means the disagreement rule can be changed without a migration or a re-ingest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.services.alignment import normalize_token

#: Minimum intersection-over-union for two word spans to be treated as the same moment. Measured
#: over the pilot episode the choice is not delicate: anywhere in 0.2-0.6 ranks segments about
#: equally well, and the middle of that range is taken here.
DEFAULT_IOU_THRESHOLD = 0.45


@dataclass(frozen=True)
class ConsensusWord:
    """One word of one hypothesis. ``start``/``end`` are clip-relative seconds, or None."""

    position: int
    word: str
    start: float | None
    end: float | None


@dataclass(frozen=True)
class ConsensusHypothesis:
    """One system's words for a segment."""

    system_id: str
    words: Sequence[ConsensusWord]


@dataclass(frozen=True)
class SlotWord:
    """A word as it sits in a slot, tagged with the system that produced it."""

    system_id: str
    position: int
    word: str
    start: float
    end: float


@dataclass
class Slot:
    """One moment in the clip, and what each system heard there.

    At most one word per system: a system that split a word its neighbours kept whole
    contributes only its first piece here and the remainder opens a slot of its own, which is
    the honest reading -- the systems really did disagree about where a word boundary was.
    """

    start: float
    end: float
    by_system: dict[str, SlotWord] = field(default_factory=dict)

    @property
    def is_unanimous(self) -> bool:
        """Whether every system present wrote the same token, ignoring case and punctuation."""
        tokens = {normalize_token(w.word) for w in self.by_system.values()}
        return len(tokens) <= 1

    def overlap(self, start: float, end: float) -> float:
        """Intersection over union of this slot's extent with ``[start, end]``."""
        union = max(self.end, end) - min(self.start, start)
        if union <= 0:
            return 0.0
        return max(0.0, min(self.end, end) - max(self.start, start)) / union


@dataclass(frozen=True)
class Dispute:
    """A seed word that every other system that spoke at that moment contradicted."""

    seed_position: int
    seed_word: str
    start: float
    end: float
    alternatives: list[SlotWord]


def build_slots(
    hypotheses: Sequence[ConsensusHypothesis],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[Slot]:
    """Group every system's word spans into shared moments, in time order.

    Args:
        hypotheses: One entry per system. Words without both boundaries are skipped -- they
            cannot be placed on the timeline, and guessing a position for them would invent
            agreement or disagreement that was never measured.
        iou_threshold: How much two spans must overlap to count as the same moment.

    Returns:
        Slots ordered by start time.
    """
    events: list[SlotWord] = [
        SlotWord(
            system_id=hypothesis.system_id,
            position=word.position,
            word=word.word,
            start=float(word.start),
            end=float(word.end),
        )
        for hypothesis in hypotheses
        for word in hypothesis.words
        if word.start is not None and word.end is not None
    ]
    events.sort(key=lambda w: (w.start, w.end))

    slots: list[Slot] = []
    for event in events:
        # Events arrive in start order, so only slots that have not finished before this one
        # begins can still overlap it. Scanning back over those is enough.
        best: Slot | None = None
        best_overlap = 0.0
        for slot in reversed(slots):
            if slot.end <= event.start:
                break
            if event.system_id in slot.by_system:
                continue
            score = slot.overlap(event.start, event.end)
            if score >= iou_threshold and score > best_overlap:
                best, best_overlap = slot, score

        if best is None:
            slots.append(Slot(start=event.start, end=event.end, by_system={event.system_id: event}))
            continue
        best.by_system[event.system_id] = event
        best.start = min(best.start, event.start)
        best.end = max(best.end, event.end)

    slots.sort(key=lambda s: (s.start, s.end))
    return slots


def seed_disputes(slots: Sequence[Slot], *, seed_system_id: str) -> list[Dispute]:
    """Seed words that every other system present disagreed with.

    "Outvoted" rather than "any disagreement" is the rule on purpose. With three systems, one
    dissenter against one supporter is weak evidence and underlining it would mark most of the
    transcript; a word both other systems contradict is the case where the seed is probably
    wrong and there is a concrete alternative to offer.

    A slot the seed is absent from is not a dispute: there is no word of the seed's to swap, so
    the editor would have nothing to anchor a replacement to. That is a hole, and it belongs to
    the coverage measure rather than to this one.

    Args:
        slots: Output of :func:`build_slots`.
        seed_system_id: The system whose transcript the annotator is editing.

    Returns:
        One entry per disputed seed word, in time order.
    """
    disputes: list[Dispute] = []
    for slot in slots:
        seed = slot.by_system.get(seed_system_id)
        if seed is None:
            continue
        others = [w for sid, w in slot.by_system.items() if sid != seed_system_id]
        if not others:
            continue
        seed_token = normalize_token(seed.word)
        if any(normalize_token(w.word) == seed_token for w in others):
            continue
        disputes.append(
            Dispute(
                seed_position=seed.position,
                seed_word=seed.word,
                start=slot.start,
                end=slot.end,
                alternatives=sorted(others, key=lambda w: w.system_id),
            )
        )
    return disputes
