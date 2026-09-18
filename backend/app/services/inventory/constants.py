"""The categories the corpus page cuts clips by, and the thresholds that turn a cut into advice.

Every number here is a judgement about what makes this corpus better for its two uses --
fine-tuning an ASR model and describing Nepali-English code-mixing -- made visible so it can be
argued with (D91).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.llm.topic import TOPIC_LABELS
from app.services.clip_classes import AXIS_BY_NAME
from app.services.speaker_meta import ALLOWED_VALUES

#: Which of the corpus's two purposes a category serves. A recommendation inherits the goal of
#: the category it came from, so the page can be read for one purpose at a time.
Goal = str
ASR: Goal = "asr"
PAPER: Goal = "paper"


@dataclass(frozen=True)
class Category:
    """One way of subdividing the corpus.

    Attributes:
        key: The key in every clip's buckets and in the API payload.
        label: What the page calls it.
        group: ``people``, ``content``, ``speech`` or ``acoustics``; the page's row.
        goals: Which purpose the split serves.
        unit: What a thin bucket is short of -- ``voices`` where the claim is about people,
            ``hours`` where it is about audio conditions.
        buckets: Display order for a closed set; empty for an open set filled from the data.
        unknown: Buckets meaning not measured or not declared. Never a stratum and never a gap:
            they are paperwork, reported as such.
        why: One line on what the split is for.
        defect: Measured buckets nobody should be asked to record more of -- a clip the
            diarizer heard nobody in is a segmentation fault, not a condition.
    """

    key: str
    label: str
    group: str
    goals: tuple[Goal, ...]
    unit: str
    buckets: tuple[str, ...]
    unknown: tuple[str, ...]
    why: str
    defect: tuple[str, ...] = ()


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


def _axis(name: str) -> tuple[str, ...]:
    return AXIS_BY_NAME[name].buckets


CATEGORIES: tuple[Category, ...] = (
    Category(
        "gender", "Gender", "people", (ASR, PAPER), "voices",
        (*GENDERS, "unresolved", "undeclared"), ("unresolved", "undeclared"),
        "Declared per speaker and reached per voice; the first variable a mixing paper splits on.",
    ),
    Category(
        "age_bracket", "Age", "people", (ASR, PAPER), "voices",
        (*AGE_BRACKETS, "unresolved", "undeclared"), ("unresolved", "undeclared"),
        "Twenty-year brackets typed from watching; the cells above 40 are the corpus's weakest.",
    ),
    Category(
        "role", "Role", "people", (PAPER,), "voices",
        (), ("unresolved", "undeclared"),
        "Host or guest in this recording. Address forms and accommodation are read across it.",
    ),
    Category(
        "voice_exposure", "Voice seen in train", "people", (ASR,), "hours",
        _axis("voice_exposure"), ("unlinked",),
        "How much of this voice a model trains on. Unseen voices are what a benchmark should hold.",
    ),
    Category(
        "voice", "Voice", "people", (ASR, PAPER), "voices",
        (), ("unlinked",),
        "Anonymous voices linked across episodes by diarizer embedding; the paper's unit of n.",
    ),
    Category(
        "topic", "Topic", "content", (ASR, PAPER), "voices",
        (*TOPIC_LABELS, "untagged"), ("untagged",),
        "Closed taxonomy per episode. Topic drives English share, so it is a confound to hold.",
    ),
    Category(
        "genre", "Genre", "content", (ASR, PAPER), "voices",
        (), ("untagged",),
        "Podcast, tech review, reel: the register of the recording, typed at ingest.",
    ),
    Category(
        "show", "Show", "content", (ASR, PAPER), "voices",
        (), ("no show id",), "The series. A value carried by one show is not independent of it.",
    ),
    Category(
        "cmi", "Code-mixing", "content", (PAPER,), "voices",
        _axis("cmi"), ("unmeasured",),
        "Minority-script share of the clip's tokens. Descriptive: it does not split WER.",
    ),
    Category(
        "speaking_rate", "Speaking speed", "speech", (ASR, PAPER), "hours",
        SPEAKING_RATE_BUCKETS, ("unmeasured",),
        "Reference words per second of VAD speech. Fast talk is where a recogniser drops words.",
    ),
    Category(
        "duration", "Clip length", "speech", (ASR,), "hours",
        _axis("duration"), (), "How long the clip is. Short clips carry the edge effects.",
    ),
    Category(
        "crosstalk", "Crosstalk", "speech", (ASR, PAPER), "hours",
        _axis("overlap"), ("unmeasured",),
        "Share of the clip where two people talk at once (D77).",
    ),
    Category(
        "speakers", "Speakers in the clip", "speech", (ASR,), "hours",
        _axis("speakers"), ("undiarized",),
        "How many people the diarizer heard for half a second or more.", defect=("none",),
    ),
    Category(
        "noise", "Noise (SNR)", "acoustics", (ASR,), "hours",
        _axis("snr"), ("unmeasured",),
        "Brouhaha's speech-to-noise ratio; low is a noisy recording (D87).",
    ),
    Category(
        "reverb", "Room (C50)", "acoustics", (ASR,), "hours",
        _axis("reverb"), ("unmeasured",),
        "Brouhaha's clarity index; low is a reverberant room.",
    ),
    Category(
        "bandwidth", "Bandwidth", "acoustics", (ASR,), "hours",
        _axis("bandwidth"), ("unmeasured",),
        "Highest frequency with energy; a phone line or a bad codec sits under 4.5 kHz.",
    ),
)  # fmt: skip
CATEGORY_BY_KEY: dict[str, Category] = {c.key: c for c in CATEGORIES}

#: What closes an unmeasured bucket, per category: the sentence the recommendation prints.
UNMEASURED_FIX: dict[str, str] = {
    "gender": "declare the speakers of the episodes that have none, and link voices to rows",
    "age_bracket": "declare the speakers of the episodes that have none, and link voices to rows",
    "role": "declare a role for every speaker row",
    "voice_exposure": "link voices (scripts/link_voices.py) or diarize the episodes without turns",
    "voice": "link voices (scripts/link_voices.py) or diarize the episodes without turns",
    "topic": "tag the untagged episodes with a topic from the taxonomy",
    "genre": "tag the untagged episodes with a genre",
    "show": "give every episode a show id",
    "cmi": "re-import the scores for clips without a code-switch density",
    "speaking_rate": "clips with no reference text and no seed cannot be timed",
    "crosstalk": "run scripts/backfill_overlap.py",
    "speakers": "diarize the episodes without turns",
    "noise": "run scripts/backfill_acoustics.py with the Brouhaha model present",
    "reverb": "run scripts/backfill_acoustics.py with the Brouhaha model present",
    "bandwidth": "run scripts/backfill_acoustics.py",
}

#: A bucket holding more than this share of a category's measured hours is reported as
#: dominance. Half is the point past which "the corpus" and "that bucket" mean the same thing.
DOMINANT_SHARE = 0.5

#: One voice holding more than this share of the corpus's hours is a model learning a person.
VOICE_DOMINANT_SHARE = 0.25

#: Below this many verified hours, a bucket a paper would compare on rests on screened labels,
#: whose script choice is the fused seed's convention rather than a checked fact (D63, D74).
VERIFIED_FLOOR_HOURS = 0.25

#: Above this share of a category's hours sitting in unknown buckets, the measurement itself is
#: the thing to fix before any gap in it can be trusted.
UNMEASURED_SHARE = 0.25

#: The accommodation design wants hosts with this many host-guest episodes (a host alone in a
#: monologue accommodates nobody), and this many of them; and guests who appear with more than
#: one host.
RECURRING_HOST_EPISODES = 5
HOSTS_WANTED = 3

#: Absent topics are listed, not enumerated: sixteen labels with a few carriers between them would
#: bury every recommendation that matters under categories nobody is short of.
MAX_TOPIC_RECOMMENDATIONS = 3

#: How much a gap in each category matters, before its kind and severity scale it. Speakers
#: first: the corpus is limited by how many different people are in it.
CATEGORY_WEIGHT: dict[str, float] = {
    "gender": 1.0,
    "age_bracket": 0.95,
    "voice": 1.0,
    "role": 0.7,
    "voice_exposure": 0.8,
    "cmi": 0.8,
    "show": 0.7,
    "genre": 0.6,
    "topic": 0.45,
    "speaking_rate": 0.6,
    "crosstalk": 0.7,
    "noise": 0.7,
    "reverb": 0.5,
    "bandwidth": 0.5,
    "duration": 0.4,
    "speakers": 0.4,
}

#: How much each kind of gap matters. Absence outranks thinness; a measurement gap ranks under
#: both because it is paperwork rather than recording.
KIND_WEIGHT: dict[str, float] = {
    "absent": 1.0,
    "thin": 0.9,
    "recurrence": 0.9,
    "dominant": 0.7,
    "no_gold": 0.6,
    "single_show": 0.5,
    "unverified": 0.5,
    "unmeasured": 0.4,
}
