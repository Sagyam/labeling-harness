"""Per-speaker lanes for one clip: what the multitrack editor starts from and may save (D98).

The editor shows every speaker of the episode's diarization as a lane and every word as a block
on one of them, spanning the time it was said. The annotator moves words between lanes and adds
the words the transcript lacks; the pipeline is usually right about *what* was said and *when*,
and what is disputed is *who* said it.

Everything here is a pure function of the clip's verified text, its seed's word spans, its
recognisers' word spans and its diarized turns, all clip-relative seconds. Nothing is stored
until the annotator saves, so the proposal can change without a migration and cannot go stale.

Speakers are the run's **display numbers** (1 = most talk time in the episode), the same numbers
the karaoke colours use. The database layer maps them to the run's raw labels.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.models.enums import LABEL_WORD_SOURCES
from app.services.alignment import normalize_token
from app.services.consensus import ConsensusHypothesis, ConsensusWord, build_slots
from app.services.fold import align, same_word

#: The editor's grid. Finer than a listener can place a word boundary, coarser than nothing.
GRID_SECONDS = 0.01
#: The shortest block that can be saved; anything shorter cannot be clicked, let alone heard.
MIN_WORD_SECONDS = 0.02
#: How far past the clip's end a block may reach: the clip's own length is float-rounded.
END_TOLERANCE_SECONDS = 0.05
#: How many recognisers must agree on a word the verified text lacks before it is offered.
MIN_CANDIDATE_VOTES = 2
#: A recogniser word this close (midpoint to midpoint) to the same word in the text is that word.
SAME_WORD_SECONDS = 0.6
#: Two spans closer than this are the same span, for deciding whether anything was changed.
SAME_SPAN_SECONDS = 0.005

#: Where a word on a lane came from. Recorded per word, so words added and words moved per clip
#: can be counted without replaying the session.
WORD_SOURCES = LABEL_WORD_SOURCES

#: The pseudo-system the verified text takes part in the slot grouping as.
_LABEL_SYSTEM = "__label__"


@dataclass(frozen=True)
class TimedWord:
    """A token and its clip-relative span, which is absent when nothing measured one."""

    word: str
    start: float | None
    end: float | None


@dataclass(frozen=True)
class Turn:
    """One diarized speaker talking from ``start`` to ``end``, clip-relative."""

    speaker: int
    start: float
    end: float


@dataclass(frozen=True)
class LaneWord:
    """One block on the lanes.

    Attributes:
        word: The token as written.
        start: Clip-relative seconds, or None for a word that has to be placed by hand.
        end: As ``start``.
        speaker: The lane it sits on, or None while it waits to be placed.
        proposed_speaker: The lane the diarization put it on. Kept beside ``speaker`` so the
            words the annotator moved are a count, not a reconstruction.
        source: One of :data:`WORD_SOURCES`.
        suggested_speaker: The lane a voiceprint suggested when the lanes were served (D99), or
            None. Advice only: it never moves a word by itself.
    """

    word: str
    start: float | None
    end: float | None
    speaker: int | None
    proposed_speaker: int | None
    source: str = "label"
    suggested_speaker: int | None = None


@dataclass(frozen=True)
class Candidate:
    """A word the recognisers heard at a moment the verified text has nothing."""

    word: str
    start: float
    end: float
    systems: tuple[str, ...]


@dataclass
class AttributionProblem:
    """Why a submitted set of lanes cannot be stored."""

    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.errors)


def place_text_on_clock(text: str | None, seed_words: Sequence[TimedWord]) -> list[TimedWord]:
    """The verified text's words, each on the clock of the seed word it corresponds to.

    Close to the rules the karaoke line follows in the browser: a word the alignment matches or
    substitutes takes that seed word's span, and a word the seed does not have at all gets no
    span, because a guessed one would put it at a moment nobody said it. The alignment is the
    plain, unfolded one: folding merges a word with its neighbour at no cost (``team ले`` against
    ``team``), which is right for scoring and wrong for timing, since it would stretch one span
    over two words and push the next word onto the span of the one after it.
    """
    tokens = (text or "").split()
    if not tokens:
        return []
    if not seed_words:
        return [TimedWord(token, None, None) for token in tokens]

    placed: list[TimedWord] = []
    s = 0
    for op in align([w.word for w in seed_words], tokens, folded=False).ops:
        if op.kind == "ins":
            placed.append(TimedWord(op.hyp[0], None, None))
        elif op.hyp:
            placed.append(TimedWord(op.hyp[0], seed_words[s].start, seed_words[s].end))
        s += len(op.ref)
    return placed


def dominant_speaker(start: float, end: float, turns: Sequence[Turn]) -> int | None:
    """The speaker whose turns cover most of ``[start, end]``, or None when no turn touches it.

    The whole span rather than the midpoint the karaoke colours use: a lane is a claim about the
    word, not about one instant of it, and in crosstalk the midpoint is a coin toss. A tie goes to
    the lower display number, the voice with more talk time in the episode.
    """
    covered: dict[int, float] = {}
    for turn in turns:
        if end > start:
            share = min(end, turn.end) - max(start, turn.start)
            if share > 0:
                covered[turn.speaker] = covered.get(turn.speaker, 0.0) + share
        elif turn.start <= start < turn.end:
            covered[turn.speaker] = covered.get(turn.speaker, 0.0) + 1.0
    if not covered:
        return None
    return min(covered, key=lambda speaker: (-covered[speaker], speaker))


def propose_lanes(words: Sequence[TimedWord], turns: Sequence[Turn]) -> list[LaneWord]:
    """Every word on the lane of the diarized turn it falls in; untimed words on none."""
    out: list[LaneWord] = []
    for word in words:
        speaker = (
            dominant_speaker(word.start, word.end, turns)
            if word.start is not None and word.end is not None
            else None
        )
        out.append(LaneWord(word.word, word.start, word.end, speaker, speaker, "label"))
    return out


def recogniser_candidates(
    placed: Sequence[TimedWord], recognisers: Sequence[ConsensusHypothesis]
) -> list[Candidate]:
    """Words at least two recognisers heard where the verified text has none, one per moment.

    The verified text takes part in the slot grouping as one more system, so a slot it is absent
    from is a hole in it. Each hole offers the token most recognisers wrote there, on the span of
    the first of them that wrote it, when at least :data:`MIN_CANDIDATE_VOTES` of them wrote it:
    one recogniser alone is mostly a misheard fragment, and offering every one buried the real
    weak-voice words on the first clips tried. A word the text already has within
    :data:`SAME_WORD_SECONDS` is never offered either -- the recognisers' spans and the aligner's
    disagree by more than the slot overlap tolerates -- so nothing is scored twice.
    """
    label = ConsensusHypothesis(
        system_id=_LABEL_SYSTEM,
        words=[
            ConsensusWord(position=index, word=w.word, start=w.start, end=w.end)
            for index, w in enumerate(placed)
        ],
    )
    timed = [w for w in placed if w.start is not None and w.end is not None]
    candidates: list[Candidate] = []
    for slot in build_slots([label, *recognisers]):
        if _LABEL_SYSTEM in slot.by_system or not slot.by_system:
            continue
        heard = sorted(slot.by_system.values(), key=lambda w: w.system_id)
        votes = Counter(normalize_token(w.word) for w in heard)
        best = max(votes, key=lambda token: (votes[token], -_first(heard, token)))
        if not best or votes[best] < MIN_CANDIDATE_VOTES:
            continue
        chosen = [w for w in heard if normalize_token(w.word) == best]
        middle = (chosen[0].start + chosen[0].end) / 2
        if any(
            abs((w.start + w.end) / 2 - middle) <= SAME_WORD_SECONDS  # type: ignore[operator]
            and same_word(w.word, chosen[0].word)
            for w in timed
        ):
            continue
        candidates.append(
            Candidate(
                word=chosen[0].word,
                start=round(chosen[0].start, 3),
                end=round(chosen[0].end, 3),
                systems=tuple(w.system_id for w in chosen),
            )
        )
    return candidates


def _first(words: Sequence[Any], token: str) -> int:
    return next(i for i, w in enumerate(words) if normalize_token(w.word) == token)


def check_attribution(
    words: Sequence[LaneWord], *, clip_seconds: float, speakers: Sequence[int]
) -> AttributionProblem:
    """Everything that would make a submitted set of lanes unfit to store.

    Every word must have text, a span inside the clip at least :data:`MIN_WORD_SECONDS` long,
    and a lane that is one of the episode's speakers. Speakers are fixed (D98): a lane that is not
    in the run is a request to create a speaker, which this editor cannot do.
    """
    problem = AttributionProblem()
    if not words:
        problem.errors.append("no words: an empty clip is flagged, not attributed")
    allowed = set(speakers)
    for index, w in enumerate(words):
        where = f"word {index + 1} ({w.word!r})"
        if not w.word.strip() or any(c.isspace() for c in w.word.strip()):
            problem.errors.append(f"{where}: a word is one token, not blank and not two")
        if w.source not in WORD_SOURCES:
            problem.errors.append(f"{where}: unknown source {w.source!r}")
        if w.start is None or w.end is None:
            problem.errors.append(f"{where}: has no span; place it on the timeline")
        elif not (math.isfinite(w.start) and math.isfinite(w.end)):
            problem.errors.append(f"{where}: span is not a number")
        elif w.start < 0 or w.end > clip_seconds + END_TOLERANCE_SECONDS:
            problem.errors.append(f"{where}: span {w.start}-{w.end} is outside the clip")
        elif w.end - w.start < MIN_WORD_SECONDS - 1e-9:
            problem.errors.append(f"{where}: shorter than {int(MIN_WORD_SECONDS * 1000)} ms")
        if w.speaker is None:
            problem.errors.append(f"{where}: is on no lane")
        elif w.speaker not in allowed:
            problem.errors.append(f"{where}: speaker {w.speaker} is not one of the episode's")
        if w.suggested_speaker is not None and w.suggested_speaker not in allowed:
            problem.errors.append(f"{where}: suggested speaker {w.suggested_speaker} is unknown")
    return problem


def in_time_order(words: Sequence[LaneWord]) -> list[LaneWord]:
    """Words by start, then end, then lane: the order they are stored and flattened in."""
    return sorted(
        words,
        key=lambda w: (
            w.start if w.start is not None else math.inf,
            w.end if w.end is not None else math.inf,
            w.speaker if w.speaker is not None else 0,
        ),
    )


def flatten(words: Sequence[LaneWord]) -> str:
    """The single stream the lanes play as together: every word, in time order."""
    return " ".join(w.word for w in in_time_order(words))


def is_unchanged(proposal: Sequence[LaneWord], submitted: Sequence[LaneWord]) -> bool:
    """Whether the annotator saved the proposal as it was served: same words, lanes and spans."""
    if len(proposal) != len(submitted):
        return False
    for a, b in zip(in_time_order(proposal), in_time_order(submitted), strict=True):
        if a.word != b.word or a.speaker != b.speaker or a.source != b.source:
            return False
        if not (_close(a.start, b.start) and _close(a.end, b.end)):
            return False
    return True


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= SAME_SPAN_SECONDS
