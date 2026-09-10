"""Tests for the script-folding comparison normalizer."""

from __future__ import annotations

import pytest

from app.services.fold import (
    FOLD_VERSION,
    align,
    fold_tokens,
    fold_version,
    same_word,
    similarity,
    skeleton,
    spelling_key,
    word_errors,
)
from app.services.normalize import Ruleset


@pytest.fixture
def ruleset() -> Ruleset:
    return Ruleset(version="test-v1", tokens={"चैँ": "चाहिँ"})


class TestSameWord:
    @pytest.mark.parametrize(
        ("dev", "latin"),
        [
            ("एक्टिभ", "active"),
            ("कफी", "coffee"),
            ("क्याप्टन", "Captain"),
            ("टिम", "team"),
            ("टाइप", "type"),
            ("फोन", "phone"),
            ("ट्राफिक", "traffic"),
            ("पुलिस", "police"),
            ("स्टेसन", "station"),
            ("बजेट", "budget"),  # soft g
            ("जनरल", "general"),
            ("गर्ल", "girl"),  # hard g
            ("कम्प्युटर", "computer"),
        ],
    )
    def test_english_spelled_in_devanagari_is_the_same_word(self, dev: str, latin: str) -> None:
        assert same_word(dev, latin)
        assert same_word(latin, dev)

    @pytest.mark.parametrize(("dev", "latin"), [("छ", "chha"), ("गर्नु", "garnu"), ("हो", "ho")])
    def test_nepali_a_recogniser_romanized_is_the_same_word(self, dev: str, latin: str) -> None:
        assert same_word(dev, latin)

    def test_a_latin_stem_with_a_devanagari_suffix_matches_the_all_devanagari_form(self) -> None:
        assert same_word("phoneमा", "फोनमा")
        assert same_word("typeको", "टाइपको")

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("हरु", "हरू"),  # vowel length
            ("गरीब", "गरिब"),
            ("भनौं", "भनौँ"),  # anusvara against chandrabindu
            ("२०", "20"),  # digits in either script
            ("Team", "team"),
        ],
    )
    def test_spelling_variants_in_one_script_are_the_same_word(self, left: str, right: str) -> None:
        assert same_word(left, right)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("गर्नु", "गर्ने"),  # infinitive and participle: different words, one skeleton
            ("थियो", "थिए"),
            ("the", "द"),  # too short for a sound match to be evidence
            ("of", "ऑफ"),
            ("ठीक", "shit"),
            ("राम्रो", "नराम्रो"),
            # Three-consonant skeletons one letter apart (ktn / ktr): at that length a single
            # consonant is a third of the word, and accepting it matched these two.
            ("एकदमै", "actually"),
            ("म", "me"),
        ],
    )
    def test_different_words_stay_different(self, left: str, right: str) -> None:
        assert not same_word(left, right)

    def test_same_script_words_are_never_matched_on_sound(self) -> None:
        # Two Devanagari words with one consonant skeleton are two Nepali words, and folding
        # them would erase grammar. The sound match is only for crossing scripts.
        assert skeleton("गर्नु") == skeleton("गर्ने")
        assert not same_word("गर्नु", "गर्ने")

    def test_the_orthography_table_applies_before_comparison(self, ruleset: Ruleset) -> None:
        assert fold_tokens("यो चैँ हो", ruleset) == fold_tokens("यो चाहिँ हो", ruleset)


class TestSpellingKey:
    def test_strips_nukta_and_final_virama(self) -> None:
        assert spelling_key("ज़") == spelling_key("ज")
        assert spelling_key("गर्") == spelling_key("गर")

    def test_latin_is_lowercased_without_apostrophes(self) -> None:
        assert spelling_key("Don't") == "dont"


class TestFoldTokens:
    def test_punctuation_and_danda_are_dropped(self, ruleset: Ruleset) -> None:
        assert fold_tokens("यो, राम्रो। OK!", ruleset) == ["यो", "राम्रो", "OK"]

    def test_empty_and_none(self, ruleset: Ruleset) -> None:
        assert fold_tokens("", ruleset) == []
        assert fold_tokens(None, ruleset) == []


class TestAlign:
    def test_counts_a_transliteration_as_a_match(self, ruleset: Ruleset) -> None:
        result = word_errors("हाम्रो team राम्रो छ", "हाम्रो टिम राम्रो छ", ruleset=ruleset)
        assert result.errors == 0
        assert result.ref_words == 4

    def test_a_real_substitution_still_counts(self, ruleset: Ruleset) -> None:
        result = word_errors("म घर जान्छु", "म वन जान्छु", ruleset=ruleset)
        assert result.substitutions == 1
        assert result.errors == 1

    def test_raw_mode_counts_the_transliteration(self, ruleset: Ruleset) -> None:
        result = word_errors("हाम्रो team", "हाम्रो टिम", ruleset=ruleset, folded=False)
        assert result.errors == 1

    def test_a_spacing_split_is_not_two_errors(self, ruleset: Ruleset) -> None:
        # गर्नुभयो and गर्नु भयो are the same honorific written with and without the space --
        # the single most common reason two correct Nepali transcripts disagree.
        result = word_errors("उहाँले गर्नुभयो", "उहाँले गर्नु भयो", ruleset=ruleset)
        assert result.errors == 0
        result = word_errors("उहाँले गर्नु भयो", "उहाँले गर्नुभयो", ruleset=ruleset)
        assert result.errors == 0

    def test_a_split_english_compound_matches_across_scripts(self, ruleset: Ruleset) -> None:
        result = word_errors("Facebook मा", "फेस बुक मा", ruleset=ruleset)
        assert result.errors == 0

    def test_insertions_and_deletions_are_reported_in_place(self) -> None:
        alignment = align(["क", "ख", "ग", "घ"], ["क", "ग", "घ", "ङ"])
        kinds = [op.kind for op in alignment.ops]
        assert kinds == ["match", "del", "match", "match", "ins"]

    def test_a_folded_match_is_labelled_as_one(self) -> None:
        (op,) = align(["team"], ["टिम"]).ops
        assert op.kind == "fold"

    def test_substitutions_carry_their_sound_similarity(self) -> None:
        alignment = align(["जमै"], ["जम्मै"])
        (op,) = alignment.ops
        assert op.kind == "sub"
        assert op.similarity >= 0.6

    def test_empty_reference(self, ruleset: Ruleset) -> None:
        result = word_errors("", "केही", ruleset=ruleset)
        assert result.insertions == 1
        assert result.rate == 1.0
        assert word_errors("", "", ruleset=ruleset).rate == 0.0


class TestSimilarity:
    def test_identical_and_disjoint(self) -> None:
        assert similarity("राम्रो", "राम्रो") == 1.0
        assert similarity("ठीक", "shit") < 0.6


def test_the_version_names_the_orthography_table_it_was_built_on(ruleset: Ruleset) -> None:
    # A reported WER is only reproducible with both halves: the fold rules and the D64 table.
    assert fold_version(ruleset) == f"{FOLD_VERSION}+test-v1"
