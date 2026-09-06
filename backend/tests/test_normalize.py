"""Tests for orthographic normalization of label text."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.normalize import (
    DEFAULT_NORMALIZATION_PATH,
    NormalizationError,
    Ruleset,
    load_ruleset,
    normalize_text,
)


@pytest.fixture
def ruleset() -> Ruleset:
    return Ruleset(version="test-v1", tokens={"चैँ": "चाहिँ", "हरु": "हरू", "टिम": "team"})


class TestNormalizeText:
    def test_replaces_a_whole_token(self, ruleset: Ruleset) -> None:
        assert normalize_text("यो चैँ राम्रो", ruleset) == "यो चाहिँ राम्रो"

    def test_leaves_unlisted_tokens_alone(self, ruleset: Ruleset) -> None:
        assert normalize_text("यो राम्रो छ", ruleset) == "यो राम्रो छ"

    def test_does_not_touch_a_substring_of_a_longer_word(self, ruleset: Ruleset) -> None:
        # चैँको is a different word. A str.replace() normalizer corrupts it; this one must not.
        assert normalize_text("चैँको कुरा", ruleset) == "चैँको कुरा"

    def test_danda_is_not_glued_onto_the_preceding_word(self, ruleset: Ruleset) -> None:
        # U+0964 DEVANAGARI DANDA sits inside the [ऀ-ॿ] block, so a token class written as that
        # range swallows the sentence terminator and the rule never fires. Guard the boundary.
        assert normalize_text("यो चैँ। अर्को", ruleset) == "यो चाहिँ। अर्को"
        assert normalize_text("यो चैँ॥", ruleset) == "यो चाहिँ॥"

    def test_preserves_punctuation_and_whitespace_exactly(self, ruleset: Ruleset) -> None:
        assert normalize_text("  यो,  चैँ\t(हरु)\n", ruleset) == "  यो,  चाहिँ\t(हरू)\n"

    def test_leaves_latin_and_digits_alone(self, ruleset: Ruleset) -> None:
        assert normalize_text("So today म Python मा loops", ruleset) == (
            "So today म Python मा loops"
        )

    def test_romanizes_a_devanagari_loanword(self, ruleset: Ruleset) -> None:
        assert normalize_text("हाम्रो टिम राम्रो", ruleset) == "हाम्रो team राम्रो"

    def test_empty_and_none_are_passed_through(self, ruleset: Ruleset) -> None:
        assert normalize_text("", ruleset) == ""
        assert normalize_text(None, ruleset) is None

    def test_is_idempotent(self, ruleset: Ruleset) -> None:
        once = normalize_text("चैँ हरु टिम", ruleset)
        assert normalize_text(once, ruleset) == once

    def test_empty_ruleset_is_the_identity(self) -> None:
        empty = Ruleset(version="none", tokens={})
        assert normalize_text("चैँ हरु", empty) == "चैँ हरु"


class TestRulesetValidation:
    def test_rejects_a_chained_rule(self) -> None:
        # A -> B, B -> C makes the result depend on iteration order and breaks idempotence.
        with pytest.raises(NormalizationError, match="chain"):
            Ruleset(version="bad", tokens={"क": "ख", "ख": "ग"})

    def test_rejects_a_rule_that_maps_a_token_to_itself(self) -> None:
        with pytest.raises(NormalizationError, match="itself"):
            Ruleset(version="bad", tokens={"चैँ": "चैँ"})

    def test_rejects_a_multi_token_source(self) -> None:
        # The normalizer substitutes one token at a time; a spaced source could never match.
        with pytest.raises(NormalizationError, match="single token"):
            Ruleset(version="bad", tokens={"रेस्क्यु एफर्ट": "rescue effort"})

    def test_rejects_a_source_containing_a_danda(self) -> None:
        with pytest.raises(NormalizationError, match="single token"):
            Ruleset(version="bad", tokens={"चैँ।": "चाहिँ।"})


class TestShippedRuleset:
    def test_the_repository_ruleset_loads_and_is_valid(self) -> None:
        rules = load_ruleset()
        assert rules.version
        assert rules.tokens

    def test_it_fixes_the_word_the_corpus_is_split_on(self) -> None:
        rules = load_ruleset()
        assert normalize_text("यो चैँ हो", rules) == "यो चाहिँ हो"

    def test_it_does_not_rewrite_pronouns_that_need_a_human(self) -> None:
        # ऊ/उ and ई/इ are separate letters, not spelling variants. Mining them out of the edit
        # history is exactly the mistake this ruleset exists to avoid, so they stay unlisted.
        rules = load_ruleset()
        for risky in ("ऊ", "उ", "ई", "इ", "जे"):
            assert risky not in rules.tokens

    def test_a_missing_file_is_an_error_not_a_silent_identity(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_ruleset(tmp_path / "nope.yaml")

    def test_default_path_points_at_the_committed_file(self) -> None:
        assert DEFAULT_NORMALIZATION_PATH.exists()
