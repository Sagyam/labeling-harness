"""Thresholds and vocabularies for the corpus inventory (D69).

Every number here is a judgement about what makes this corpus better, made visible so it can be
argued with; see the package docstring for the two measurement decisions behind them.
"""

from __future__ import annotations

from app.llm.topic import TOPIC_LABELS
from app.services.speaker_meta import ALLOWED_VALUES

#: Closed vocabularies, per dimension key. A dimension with one can say what is *absent* rather
#: than only what is present -- the difference between an inventory and a gap report. ``show_id``
#: is deliberately open: there is no list of every Nepali YouTube channel.
VOCABULARIES: dict[str, tuple[str, ...]] = {
    "topic": TOPIC_LABELS,
    "gender": tuple(sorted(ALLOWED_VALUES["gender"])),
    "age_bracket": ("under_20", "20_39", "40_59", "60_79", "80_plus"),
}

#: Speaker dimensions, in the order a gap in one hurts most. Ordering matches
#: ``dataset.coverage_keys`` for the same reason it is ordered there.
SPEAKER_KEYS: tuple[str, ...] = ("gender", "age_bracket")

#: Code-switching bands, as ``(name, lower, upper)`` over ``code_switch_density`` (the minority
#: script's share of tokens -- in practice the English share). The boundaries are where the
#: measured corpus actually sits: every show so far falls between 0.13 and 0.36, so the interesting
#: question is whether anything exists outside that, and a three-way cut answers it without
#: inventing precision.
REGISTER_BANDS: tuple[tuple[str, float, float], ...] = (
    ("low", 0.0, 0.10),
    ("mid", 0.10, 0.25),
    ("high", 0.25, 1.01),
)

#: What a band means in the world, so a recommendation can say what to search for rather than
#: printing a number at someone.
BAND_DESCRIPTION: dict[str, str] = {
    "low": "little English mixed in -- rural, older or non-media speakers, formal Nepali",
    "mid": "moderate mixing -- the register most of the corpus already sits in",
    "high": "heavily English-mixed -- tech, startup, finance or diaspora speech",
}

#: Show mean code-switch densities closer together than this leave the dependent variable of a
#: code-switching study with almost no variance to explain: every show says roughly the same thing
#: about how much English gets mixed in, so nothing in the corpus can distinguish a cause from a
#: constant. Measured across *shows*, not clips -- clip-level density spans the whole range inside
#: any single show, which says nothing about who was recorded.
NARROW_SHOW_SPREAD = 0.15

#: Bins for the code-switching histogram. Ten is enough to see a distribution's shape and few
#: enough that each bin holds something at corpus sizes of a few hours.
HISTOGRAM_BINS = 10

#: A show holding more than this share of the corpus's hours is reported as concentration. Half is
#: the point past which "the corpus" and "that show" start to mean the same thing.
DOMINANT_SHARE = 0.5

#: Episodes longer than this are the long-podcast material the corpus is moving away from: the
#: pivot is toward 5-20 minute videos chosen for the speaker they add rather than their duration.
LONG_EPISODE_MINUTES = 45.0

#: Above this share of hours sitting in long episodes, sourcing itself is the thing to change.
LONG_EPISODE_SHARE = 0.5

#: How much each kind of gap matters, before severity scales it. Speakers first: the corpus is
#: limited by how many different people are in it, not by how many hours it holds.
GAP_WEIGHT: dict[str, float] = {
    "gender": 1.0,
    "age_bracket": 0.95,
    "register": 0.9,
    "register_spread": 0.9,
    "show_concentration": 0.8,
    "gold_coverage": 0.7,
    "episode_length": 0.5,
    "topic": 0.4,
}

#: Absent topics are listed, not enumerated: sixteen labels with two carriers between them would
#: otherwise bury every recommendation that matters under a list of categories nobody is short of.
MAX_TOPIC_RECOMMENDATIONS = 3
