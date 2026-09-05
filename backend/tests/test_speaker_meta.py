"""Tests for the speaker-metadata allowlist (D56).

Every test here protects one property: **a speaker's identity never reaches the database**. The
allowlist is enforced at the importer, not at the form, because the form is only one of the two
ways episode metadata arrives -- an upstream ``episode.json`` is the other, and it is written by a
pipeline this repository does not own.
"""

from __future__ import annotations

from app.services.speaker_meta import ALLOWED_SPEAKER_FIELDS, strip_speaker_pii


def test_a_speaker_name_is_dropped() -> None:
    cleaned = strip_speaker_pii({"speakers": {"spk0": {"name": "Sushant", "role": "host"}}})
    assert cleaned == {"speakers": {"spk0": {"role": "host"}}}


def test_dialect_and_origin_are_dropped() -> None:
    cleaned = strip_speaker_pii(
        {"speakers": {"spk0": {"origin": "Kathmandu", "dialect": "eastern", "gender": "female"}}}
    )
    assert cleaned == {"speakers": {"spk0": {"gender": "female"}}}


def test_an_unknown_field_is_dropped_rather_than_kept() -> None:
    """An allowlist, not a blocklist: a field nobody has thought about yet is not PII-safe."""
    cleaned = strip_speaker_pii({"speakers": {"spk0": {"employer": "Some Co", "role": "guest"}}})
    assert cleaned == {"speakers": {"spk0": {"role": "guest"}}}


def test_every_allowed_field_survives() -> None:
    speaker = dict.fromkeys(ALLOWED_SPEAKER_FIELDS, "x")
    cleaned = strip_speaker_pii({"speakers": {"spk0": speaker}})
    assert cleaned["speakers"]["spk0"] == speaker


def test_a_speaker_with_nothing_left_is_removed_entirely() -> None:
    """A key whose whole value was a name is not worth an empty object."""
    cleaned = strip_speaker_pii({"speakers": {"spk0": {"name": "Kusang"}}})
    assert cleaned == {}


def test_metadata_outside_speakers_is_untouched() -> None:
    cleaned = strip_speaker_pii({"topic": "tech_gadgets", "genre": "podcast", "speakers": {}})
    assert cleaned == {"topic": "tech_gadgets", "genre": "podcast"}


def test_a_malformed_speakers_block_is_dropped_not_raised() -> None:
    """Ingest must not fail on a bad upstream field; it must refuse to store it."""
    assert strip_speaker_pii({"speakers": "Sushant and Kusang"}) == {}
    assert strip_speaker_pii({"speakers": ["Sushant"]}) == {}
    assert strip_speaker_pii({"speakers": {"spk0": "Sushant"}}) == {}


def test_none_and_empty_are_handled() -> None:
    assert strip_speaker_pii(None) == {}
    assert strip_speaker_pii({}) == {}


def test_the_input_is_not_mutated() -> None:
    original = {"speakers": {"spk0": {"name": "Sushant", "role": "host"}}}
    strip_speaker_pii(original)
    assert original == {"speakers": {"spk0": {"name": "Sushant", "role": "host"}}}
