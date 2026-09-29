"""The categories the corpus page cuts clips by (D91, D104).

Every clip carries one bucket on each category, so hours cut by any of them sum to the pot. What
counts as thin, enough or plenty is configured per pot in ``dataset.coverage`` and applied in
:mod:`app.services.inventory.coverage`.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.llm.topic import TOPIC_LABELS
from app.services.clip_classes import AXIS_BY_NAME
from app.services.episode_meta import GENRE_LABELS
from app.services.speaker_meta import ALLOWED_VALUES


@dataclass(frozen=True)
class Category:
    """One way of subdividing the corpus.

    Attributes:
        key: The key in every clip's buckets and in the API payload.
        label: What the page calls it.
        group: ``people``, ``content``, ``speech`` or ``acoustics``; the page's row.
        unit: What a thin bucket can be short of -- ``voices`` (and hours) where the split is
            about people or content, ``hours`` alone where it is about audio conditions.
        buckets: Display order for a closed set; empty for an open set filled from the data.
        unknown: Buckets meaning not measured or not declared. Never a stratum and never a gap:
            they are paperwork, reported as such.
        why: One line on what the split is for.
        defect: Measured buckets nobody should be asked to record more of -- a clip the
            diarizer heard nobody in is a segmentation fault, not a condition.
        rated: Whether its buckets get a coverage status at all. A show is where audio comes
            from, not a stratum anyone fills, so its buckets are listed but never rated.
    """

    key: str
    label: str
    group: str
    unit: str
    buckets: tuple[str, ...]
    unknown: tuple[str, ...]
    why: str
    defect: tuple[str, ...] = ()
    rated: bool = True


GROUPS: tuple[tuple[str, str], ...] = (
    ("people", "People"),
    ("content", "Content"),
    ("speech", "Speech"),
    ("acoustics", "Acoustics"),
)

#: Twenty-year brackets, in age order rather than by talk time.
AGE_BRACKETS: tuple[str, ...] = ("under_20", "20_39", "40_59", "60_79", "80_plus")
GENDERS: tuple[str, ...] = tuple(sorted(ALLOWED_VALUES["gender"], reverse=True))

#: Words per second of VAD speech. Edges sit near the 10th, 30th, 70th and 93rd percentiles of the
#: corpus on 2026-09-18 (median 2.9 w/s), so the middle bucket is where most speech is and the
#: outer two are genuinely slow and genuinely fast talkers.
SPEAKING_RATE_EDGES: tuple[float, ...] = (2.0, 2.6, 3.2, 3.8)
SPEAKING_RATE_BUCKETS: tuple[str, ...] = (
    "<2.0 w/s",
    "2.0-2.6 w/s",
    "2.6-3.2 w/s",
    "3.2-3.8 w/s",
    "3.8+ w/s",
    "unmeasured",
)

#: A voice's reference words attributed inside a bucket before it counts toward that bucket's
#: speakers. The same floor the sociolinguistics notebook uses for a usable voice.
MIN_VOICE_WORDS = 300

#: A clip's words are credited to a voice only when it holds this share of the clip's diarized
#: talk; anything more shared is nobody's, so a two-person clip never inflates one person's n.
ATTRIBUTION_SHARE = 0.9

#: A host with this many host-guest episodes is a recurring host, the unit the accommodation
#: design needs; a host alone in a monologue accommodates nobody.
RECURRING_HOST_EPISODES = 5


def _axis(name: str) -> tuple[str, ...]:
    return AXIS_BY_NAME[name].buckets


CATEGORIES: tuple[Category, ...] = (
    Category(
        "gender", "Gender", "people", "voices",
        (*GENDERS, "unresolved", "undeclared"), ("unresolved", "undeclared"),
        "Declared per speaker and reached per voice, or set on the voice by ear (D104).",
    ),
    Category(
        "age_bracket", "Age", "people", "voices",
        (*AGE_BRACKETS, "unresolved", "undeclared"), ("unresolved", "undeclared"),
        "Twenty-year brackets, declared from watching or set on the voice by ear (D104).",
    ),
    Category(
        "role", "Role", "people", "voices",
        (), ("unresolved", "undeclared"),
        "Host or guest in this recording. Address forms and accommodation are read across it.",
    ),
    Category(
        "voice_exposure", "Voice seen in train", "people", "hours",
        _axis("voice_exposure"), ("unlinked",),
        "How much of this voice a model trains on. Unseen voices are what a benchmark should hold.",
    ),
    Category(
        "topic", "Topic", "content", "voices",
        (*TOPIC_LABELS, "untagged"), ("untagged",),
        "Closed taxonomy per episode. Topic drives English share, so it is a confound to hold.",
    ),
    Category(
        "genre", "Genre", "content", "voices",
        (*GENRE_LABELS, "untagged"), ("untagged",),
        "Closed list of recording formats: who talks to whom, scripted or not (D102).",
    ),
    Category(
        "show", "Show", "content", "voices",
        (), ("no show id",), "The series. A value carried by one show is not independent of it.",
        rated=False,
    ),
    Category(
        "cmi", "Code-mixing", "content", "voices",
        _axis("cmi"), ("unmeasured",),
        "Minority-script share of the clip's tokens. Descriptive: it does not split WER.",
    ),
    Category(
        "speaking_rate", "Speaking speed", "speech", "hours",
        SPEAKING_RATE_BUCKETS, ("unmeasured",),
        "Reference words per second of VAD speech. Fast talk is where a recogniser drops words.",
    ),
    Category(
        "duration", "Clip length", "speech", "hours",
        _axis("duration"), (), "How long the clip is. Short clips carry the edge effects.",
    ),
    Category(
        "crosstalk", "Crosstalk", "speech", "hours",
        _axis("overlap"), ("unmeasured",),
        "Share of the clip where two people talk at once (D77).",
    ),
    Category(
        "speakers", "Speakers in the clip", "speech", "hours",
        _axis("speakers"), ("undiarized",),
        "How many people the diarizer heard for half a second or more.", defect=("none",),
    ),
    Category(
        "noise", "Noise (SNR)", "acoustics", "hours",
        _axis("snr"), ("unmeasured",),
        "Brouhaha's speech-to-noise ratio; low is a noisy recording (D87).",
    ),
    Category(
        "reverb", "Room (C50)", "acoustics", "hours",
        _axis("reverb"), ("unmeasured",),
        "Brouhaha's clarity index; low is a reverberant room.",
    ),
    Category(
        "bandwidth", "Bandwidth", "acoustics", "hours",
        _axis("bandwidth"), ("unmeasured",),
        "Highest frequency with energy; a phone line or a bad codec sits under 4.5 kHz.",
    ),
)  # fmt: skip
CATEGORY_BY_KEY: dict[str, Category] = {c.key: c for c in CATEGORIES}
