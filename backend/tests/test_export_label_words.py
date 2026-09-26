"""Word timings for the exported label text (2026-09-26): the pure half of export.py.

A training row carries the words of its `text` with clip-relative spans, so a crosstalk mix can
write both voices' words in time order (roadmap C). The spans come from the seed's aligned words
when the label is the seed's text; otherwise the clip is realigned. A span is never guessed."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.export import export_words, seed_spans
from app.services.normalize import Ruleset

NONE = Ruleset(version="t", tokens={})


def _seed(*words: tuple[str, float | None, float | None]):
    return SimpleNamespace(
        words=[SimpleNamespace(word_raw=w, start_time=s, end_time=e) for w, s, e in words]
    )


def test_seed_spans_are_used_when_the_label_is_the_seed_text():
    seed = _seed(("मलाई", 0.0, 0.4), ("best", 0.4, 0.7), ("लाग्यो।", 0.7, 1.2))
    assert seed_spans("मलाई best लाग्यो।", seed) == [
        ("मलाई", 0.0, 0.4),
        ("best", 0.4, 0.7),
        ("लाग्यो।", 0.7, 1.2),
    ]


def test_seed_spans_are_refused_for_an_edited_label_or_a_missing_span():
    seed = _seed(("मलाई", 0.0, 0.4), ("best", 0.4, 0.7))
    assert seed_spans("मलाई worst", seed) is None
    assert seed_spans("मलाई best", _seed(("मलाई", 0.0, 0.4), ("best", None, None))) is None
    assert seed_spans("मलाई best", None) is None
    assert seed_spans("", seed) is None


def test_export_words_follow_the_exported_text_danda_and_rules_included():
    rules = Ruleset(version="t", tokens={"गर्नुभयो": "गर्नु"})
    spans = [("मलाई", 0.0, 0.4), ("गर्नुभयो।", 0.4, 1.0), ("OK", 1.0, 1.3)]
    assert export_words(spans, "मलाई गर्नु। OK", rules) == [
        {"word": "मलाई", "start": 0.0, "end": 0.4},
        {"word": "गर्नु", "start": 0.4, "end": 1.0},
        {"word": "OK", "start": 1.0, "end": 1.3},
    ]


def test_a_rule_that_writes_two_scripts_shares_the_span_between_them():
    """norm-v3 rewrites कम्पनीले as companyले, which the text splits into two words; both keep
    the spoken word's span rather than an invented boundary between them."""
    rules = Ruleset(version="t", tokens={"कम्पनीले": "companyले"})
    assert export_words([("कम्पनीले", 1.0, 1.6)], "companyले", rules) == [
        {"word": "company", "start": 1.0, "end": 1.6},
        {"word": "ले", "start": 1.0, "end": 1.6},
    ]


def test_export_words_refuse_spans_that_do_not_spell_the_text():
    assert export_words([("मलाई", 0.0, 0.4)], "मलाई best", NONE) is None
