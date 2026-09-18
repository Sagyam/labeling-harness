"""Every clip's classes: the conditions a WER is split by (roadmap item 1, D87).

A class is a bucket a clip falls into on one axis -- how much of it is crosstalk, how many people
talk in it, how band-limited its audio is. Each is computed from what the harness already stores
about the clip, never from its reference, so a class exists for every clip, train included, and
for audio nobody has labelled.

Classification is pure: :func:`classify` takes a :class:`ClipFacts` and returns ``{axis:
bucket}``. :func:`load_clip_facts` is the only function that reads the database. Buckets meaning
"never measured" are kept apart from every measured bucket: ``undiarized`` is not evidence of one
speaker, as ``unmeasured`` overlap is not evidence of a clean clip (D77).

CMI is an axis for description, not for errors: it did not split WER within an episode
(docs/findings.md), and it is kept for its sociolinguistic value. ``descriptive`` marks it.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from app.models import DiarizationRun, Episode, Segment, SegmentScore, SpeakerTurn

Turn = tuple[float, float, str]

#: Talk time, in the clip, that makes someone one of its speakers. A shorter backchannel ("हो",
#: "hm") is crosstalk if it overlaps, but not a second speaker; the crosstalk study counted a
#: two-speaker clip the same way (docs/findings.md).
MIN_TALK_SECONDS = 0.5


@dataclass(frozen=True)
class Axis:
    """One way of splitting clips.

    Attributes:
        name: The key in a clip's classes and in a run's ``by_class``.
        label: What the Models page calls it.
        buckets: Every bucket, in display order. Empty for an open set (voices).
        baseline: The bucket a within-episode rate ratio compares each other bucket against;
            ``None`` when the axis cannot vary within an episode, so no ratio means anything.
        unmeasured: The bucket for a clip the axis could not be measured on. It never gets a
            ratio: it is missing data, not a condition.
        descriptive: Kept to describe the corpus rather than to explain errors.
    """

    name: str
    label: str
    buckets: tuple[str, ...]
    baseline: str | None
    unmeasured: str | None = None
    descriptive: bool = False


AXES: tuple[Axis, ...] = (
    Axis("overlap", "Crosstalk", ("none", "0-5%", "5-15%", ">15%", "unmeasured"), "none",
         "unmeasured"),
    Axis("speakers", "Speakers in the clip", ("1", "2", "3+", "none", "undiarized"), "1",
         "undiarized"),
    Axis("turn_changes", "Turn changes", ("0", "1", "2+", "undiarized"), "0", "undiarized"),
    Axis("pause", "Pause share", ("<2%", "2-10%", ">10%", "unmeasured"), "<2%", "unmeasured"),
    Axis("duration", "Clip length", ("<5 s", "5-15 s", "15+ s"), "5-15 s"),
    Axis("bandwidth", "Audio bandwidth", ("<4.5 kHz", "4.5-6.5 kHz", "6.5+ kHz", "unmeasured"),
         "6.5+ kHz", "unmeasured"),
    # Brouhaha's speech-to-noise ratio and C50 (D87). Edges from the corpus of 2026-09-15:
    # SNR 194 / 752 / 1,355 / 1,656 / 3,114 clips; C50 292 / 398 / 498 / 5,883.
    Axis("snr", "Speech-to-noise",
         ("<15 dB", "15-25 dB", "25-35 dB", "35-45 dB", "45+ dB", "unmeasured"), "45+ dB",
         "unmeasured"),
    Axis("reverb", "Room (C50)", ("<40 dB", "40-50 dB", "50-55 dB", "55+ dB", "unmeasured"),
         "55+ dB", "unmeasured"),
    # The baseline is where most clips are, and where an episode's seen host is: "1 h+" is
    # almost only the tech-review host, who shares no episode with an unseen voice.
    Axis("voice_exposure", "Voice's hours in train",
         ("unseen", "<10 min", "10-60 min", "1 h+", "unlinked"), "10-60 min", "unlinked"),
    Axis("voice", "Voice", (), None, "unlinked"),
    Axis("gender", "Declared gender", ("male", "female", "mixed", "undeclared"), None,
         "undeclared"),
    Axis("age_bracket", "Declared age",
         ("under_20", "20_39", "40_59", "60_79", "80_plus", "mixed", "undeclared"), None,
         "undeclared"),
    Axis("cmi", "Code-mixing (CMI)", ("0", "<15", "15-30", "30+", "unmeasured"), "0",
         "unmeasured", descriptive=True),
)  # fmt: skip
AXIS_BY_NAME = {axis.name: axis for axis in AXES}


@dataclass(frozen=True)
class ClipFacts:
    """What classification reads about one clip. Times are clip-relative seconds.

    ``None`` always means "not measured": ``turns`` is ``None`` for an episode never diarized and
    ``[]`` for a clip the diarizer heard nobody in.
    """

    duration: float
    vad_spans: Sequence[Sequence[float]] | None
    overlap_spans: Sequence[Sequence[float]] | None
    turns: Sequence[Turn] | None
    cmi: float | None
    acoustics: Mapping[str, Any] | None
    voice: str | None
    voice_train_seconds: float | None
    gender: str
    age_bracket: str
    #: The episode's newest run's speaker label to linked voice, so a consumer can name each
    #: turn's voice. ``None`` until the run is linked; not read by :func:`classify`.
    voices: Mapping[str, str] | None = None


# --- one axis at a time -----------------------------------------------------------------------


def overlap_share(spans: Sequence[Sequence[float]] | None, duration: float) -> float | None:
    """The fraction of a clip spent in crosstalk; ``None`` when it was never measured."""
    if spans is None:
        return None
    if duration <= 0:
        return 0.0
    return min(1.0, sum(max(0.0, end - start) for start, end in spans) / duration)


def overlap_share_sql() -> sa.ColumnElement[float]:
    """:func:`overlap_share` as a SQL expression over ``segments``, for ordering and filtering.

    Must agree with the Python function clip for clip -- ``test_clip_classes.py`` checks that it
    does. Never measured stays null rather than sorting as a clean clip, and it has two spellings
    to catch: a SQL NULL, and the JSON ``null`` the ORM writes for a Python ``None``.
    """
    measured = sa.func.jsonb_typeof(Segment.overlap_spans_jsonb) == "array"
    # jsonb_array_elements raises on a scalar, and the planner may reach it whatever the outer
    # CASE says, so it is only ever handed an array.
    spans = sa.case((measured, Segment.overlap_spans_jsonb), else_=sa.cast("[]", JSONB))
    span = sa.func.jsonb_array_elements(spans).table_valued(sa.column("value", JSONB))
    seconds = (
        sa.select(
            sa.func.coalesce(
                sa.func.sum(
                    sa.func.greatest(0.0, span.c.value[1].as_float() - span.c.value[0].as_float())
                ),
                0.0,
            )
        )
        .select_from(span)
        .scalar_subquery()
    )
    return sa.case(
        (
            sa.and_(measured, Segment.duration_seconds > 0),
            sa.func.least(1.0, seconds / Segment.duration_seconds),
        ),
        (measured, sa.literal(0.0)),
        else_=sa.null(),
    )


def overlap_bucket(share: float | None) -> str:
    """The overlap bucket for a clip's overlap share (docs/findings.md)."""
    if share is None:
        return "unmeasured"
    if share <= 0:
        return "none"
    if share < 0.05:
        return "0-5%"
    if share <= 0.15:
        return "5-15%"
    return ">15%"


def _talk(turns: Iterable[Turn]) -> dict[str, float]:
    talk: dict[str, float] = defaultdict(float)
    for start, end, speaker in turns:
        talk[speaker] += max(0.0, end - start)
    return talk


def speakers_present(turns: Iterable[Turn]) -> list[str]:
    """The speakers with at least :data:`MIN_TALK_SECONDS` of talk, most talk first."""
    talk = _talk(turns)
    return sorted((s for s, t in talk.items() if t >= MIN_TALK_SECONDS), key=lambda s: -talk[s])


def dominant_speaker(turns: Iterable[Turn]) -> str | None:
    """The speaker with the most talk time in the clip, or ``None`` when nobody talks."""
    talk = _talk(turns)
    return max(sorted(talk), key=lambda s: talk[s]) if talk else None


def turn_changes(turns: Sequence[Turn]) -> int:
    """How often the floor passes between speakers who count, in order of turn start.

    Turns overlap, so this is the number of changes in the sequence of turn *starts*: a reply
    that begins under the other speaker's last words is one hand-over, as it would be heard.
    """
    present = set(speakers_present(turns))
    order = [speaker for _, _, speaker in sorted(turns) if speaker in present]
    return sum(1 for a, b in itertools.pairwise(order) if a != b)


def declared_value(metadata: Mapping[str, Any] | None, field: str) -> str:
    """A declared speaker field that holds for every clip of the episode, or why it does not.

    Diarization cannot say which voice is which declared speaker (D78), so a declared value
    reaches a clip only when every speaker of the episode shares it. ``mixed`` when they differ,
    ``undeclared`` when any speaker -- a form row left blank included -- has no value.
    """
    metadata = metadata or {}
    speakers = metadata.get("speakers")
    rows = list(speakers.values()) if isinstance(speakers, Mapping) else []
    count = metadata.get("speaker_count")
    if not rows or (isinstance(count, int) and count > len(rows)):
        return "undeclared"
    values = [row.get(field) if isinstance(row, Mapping) else None for row in rows]
    if any(not isinstance(v, str) or not v for v in values):
        return "undeclared"
    return values[0] if len(set(values)) == 1 else "mixed"


def _speakers_bucket(turns: Sequence[Turn] | None) -> str:
    if turns is None:
        return "undiarized"
    count = len(speakers_present(turns))
    return "none" if count == 0 else "3+" if count >= 3 else str(count)


def _turn_changes_bucket(turns: Sequence[Turn] | None) -> str:
    if turns is None:
        return "undiarized"
    changes = turn_changes(turns)
    return "2+" if changes >= 2 else str(changes)


def _pause_bucket(vad_spans: Sequence[Sequence[float]] | None, duration: float) -> str:
    if vad_spans is None or duration <= 0:
        return "unmeasured"
    speech = sum(max(0.0, end - start) for start, end in vad_spans)
    pause = max(0.0, 1.0 - speech / duration)
    return "<2%" if pause < 0.02 else "2-10%" if pause <= 0.10 else ">10%"


def _duration_bucket(duration: float) -> str:
    return "<5 s" if duration < 5 else "5-15 s" if duration < 15 else "15+ s"


def _bandwidth_bucket(acoustics: Mapping[str, Any] | None) -> str:
    hz = (acoustics or {}).get("bandwidth_hz")
    if not isinstance(hz, (int, float)):
        return "unmeasured"
    return "<4.5 kHz" if hz < 4500 else "4.5-6.5 kHz" if hz < 6500 else "6.5+ kHz"


def _snr_bucket(acoustics: Mapping[str, Any] | None) -> str:
    db = (acoustics or {}).get("snr_db")
    if not isinstance(db, (int, float)):
        return "unmeasured"
    for edge, bucket in ((15, "<15 dB"), (25, "15-25 dB"), (35, "25-35 dB"), (45, "35-45 dB")):
        if db < edge:
            return bucket
    return "45+ dB"


def _reverb_bucket(acoustics: Mapping[str, Any] | None) -> str:
    """C50 in dB: low is a reverberant room, high a dry one."""
    db = (acoustics or {}).get("c50_db")
    if not isinstance(db, (int, float)):
        return "unmeasured"
    return "<40 dB" if db < 40 else "40-50 dB" if db < 50 else "50-55 dB" if db < 55 else "55+ dB"


def _exposure_bucket(voice: str | None, seconds: float | None) -> str:
    if voice is None or seconds is None:
        return "unlinked"
    if seconds <= 0:
        return "unseen"
    return "<10 min" if seconds < 600 else "10-60 min" if seconds < 3600 else "1 h+"


def _cmi_bucket(cmi: float | None) -> str:
    if cmi is None:
        return "unmeasured"
    return "0" if cmi <= 0 else "<15" if cmi < 15 else "15-30" if cmi < 30 else "30+"


def classify(facts: ClipFacts) -> dict[str, str]:
    """Every axis's bucket for one clip."""
    return {
        "overlap": overlap_bucket(overlap_share(facts.overlap_spans, facts.duration)),
        "speakers": _speakers_bucket(facts.turns),
        "turn_changes": _turn_changes_bucket(facts.turns),
        "pause": _pause_bucket(facts.vad_spans, facts.duration),
        "duration": _duration_bucket(facts.duration),
        "bandwidth": _bandwidth_bucket(facts.acoustics),
        "snr": _snr_bucket(facts.acoustics),
        "reverb": _reverb_bucket(facts.acoustics),
        "voice_exposure": _exposure_bucket(facts.voice, facts.voice_train_seconds),
        "voice": facts.voice or "unlinked",
        "gender": facts.gender,
        "age_bracket": facts.age_bracket,
        "cmi": _cmi_bucket(facts.cmi),
    }


# --- reading the facts ------------------------------------------------------------------------


def clip_turns(turns: Iterable[Turn], start: float, end: float) -> list[Turn]:
    """Episode-relative turns cut to one clip and made clip-relative."""
    out: list[Turn] = []
    for lo, hi, speaker in turns:
        a, b = max(lo, start), min(hi, end)
        if b > a:
            out.append((a - start, b - start, speaker))
    return out


class _Diarization:
    """One episode's newest diarization run: its turns, indexed for cutting to clips, and its
    speakers' linked voices (``None`` until the run is linked)."""

    def __init__(self, turns: list[Turn], voices: Mapping[str, str] | None) -> None:
        turns = sorted(turns)
        self.turns = turns
        self.starts = np.array([t[0] for t in turns], dtype=np.float64)
        self.ends = np.array([t[1] for t in turns], dtype=np.float64)
        self.voices = voices

    def within(self, start: float, end: float) -> list[Turn]:
        hits = np.flatnonzero((self.starts < end) & (self.ends > start))
        return clip_turns((self.turns[i] for i in hits), start, end)

    def voice(self, clip: Sequence[Turn]) -> str | None:
        speaker = dominant_speaker(clip)
        return None if self.voices is None or speaker is None else self.voices.get(speaker)


def _diarizations(session: Session, episode_ids: Sequence[int] | None) -> dict[int, _Diarization]:
    """Each episode's newest diarization run, for these episodes or all of them."""
    query = sa.select(DiarizationRun.id, DiarizationRun.episode_id, DiarizationRun.voices_jsonb)
    if episode_ids is not None:
        query = query.where(DiarizationRun.episode_id.in_(episode_ids))
    newest: dict[int, tuple[int, Mapping[str, str] | None]] = {}
    for run_id, episode_id, voices in session.execute(query.order_by(DiarizationRun.id)):
        newest[episode_id] = (run_id, voices)  # the newest run wins
    turns: dict[int, list[Turn]] = defaultdict(list)
    for run_id, start, end, speaker in session.execute(
        sa.select(
            SpeakerTurn.run_id, SpeakerTurn.start_time, SpeakerTurn.end_time, SpeakerTurn.speaker
        ).where(SpeakerTurn.run_id.in_([run_id for run_id, _ in newest.values()]))
    ):
        turns[run_id].append((float(start), float(end), str(speaker)))
    return {
        episode_id: _Diarization(turns[run_id], voices)
        for episode_id, (run_id, voices) in newest.items()
    }


def voice_train_seconds(session: Session) -> dict[str, float]:
    """Each voice's talk time inside the clips a model trains on: the train pot of train-split
    episodes. Val episodes are held out of training (D71), so their clips do not count."""
    diarizations = _diarizations(session, None)
    seconds: dict[str, float] = defaultdict(float)
    for episode_id, start, end in session.execute(
        sa.select(Segment.episode_id, Segment.start_time, Segment.end_time)
        .join(Episode, Episode.id == Segment.episode_id)
        .where(Segment.pot == "train", Episode.split == "train")
    ):
        diarization = diarizations.get(episode_id)
        if diarization is None or diarization.voices is None:
            continue
        for speaker, talk in _talk(diarization.within(start, end)).items():
            if (voice := diarization.voices.get(speaker)) is not None:
                seconds[voice] += talk
    return dict(seconds)


def load_clip_facts(session: Session, segments: Sequence[Segment]) -> dict[int, ClipFacts]:
    """Everything :func:`classify` needs for these clips, keyed by segment id.

    A handful of queries whatever the number of clips: each episode's newest diarization run
    and its turns, every voice's talk time in train, the clips' CMI, and the episodes' declared
    speakers.
    """
    episode_ids = sorted({s.episode_id for s in segments})
    episodes = {
        e.id: e for e in session.scalars(sa.select(Episode).where(Episode.id.in_(episode_ids)))
    }
    diarizations = _diarizations(session, episode_ids)
    exposure = voice_train_seconds(session)
    cmi = dict(
        session.execute(
            sa.select(SegmentScore.segment_id, SegmentScore.code_switch_density).where(
                SegmentScore.segment_id.in_([s.id for s in segments])
            )
        )
        .tuples()
        .all()
    )

    facts: dict[int, ClipFacts] = {}
    for segment in segments:
        episode = episodes[segment.episode_id]
        diarization = diarizations.get(segment.episode_id)
        turns = None
        voice = None
        if diarization is not None:
            turns = diarization.within(segment.start_time, segment.end_time)
            voice = diarization.voice(turns)
        density = cmi.get(segment.id)
        facts[segment.id] = ClipFacts(
            duration=segment.duration_seconds,
            vad_spans=segment.vad_spans_jsonb,
            overlap_spans=segment.overlap_spans_jsonb,
            turns=turns,
            cmi=None if density is None else 100.0 * density,
            acoustics=segment.acoustics_jsonb,
            voice=voice,
            voice_train_seconds=None if voice is None else exposure.get(voice, 0.0),
            gender=declared_value(episode.metadata_jsonb, "gender"),
            age_bracket=declared_value(episode.metadata_jsonb, "age_bracket"),
            voices=None if diarization is None else diarization.voices,
        )
    return facts


def classify_segments(session: Session, segments: Sequence[Segment]) -> dict[int, dict[str, str]]:
    """:func:`classify` for every clip, keyed by segment id."""
    return {sid: classify(f) for sid, f in load_clip_facts(session, segments).items()}
