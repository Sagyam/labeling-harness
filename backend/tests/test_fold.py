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

    @pytest.mark.parametrize(
        ("word", "key"),
        [("कीरा", "किरा"), ("हीरा", "हिरा"), ("लाइन", "लाइन"), ("बल्ले", "बल्ले"), ("नाम", "नाम"),
         ("कम", "कम"), ("निको", "निको"), ("साथी", "साथि")],
    )  # fmt: skip
    def test_a_word_no_rule_is_about_keeps_its_spelling_key(self, word: str, key: str) -> None:
        assert spelling_key(word) == key

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
        assert spelling_key("O'Neil") == "oneil"
        assert spelling_key("Don't") == "donot"  # a contraction is expanded first


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
        alignment = align(["जमा"], ["जम्मा"])
        (op,) = alignment.ops
        assert op.kind == "sub"
        assert op.similarity >= 0.6

    def test_empty_reference(self, ruleset: Ruleset) -> None:
        result = word_errors("", "केही", ruleset=ruleset)
        assert result.insertions == 1
        assert result.rate == 1.0
        assert word_errors("", "", ruleset=ruleset).rate == 0.0


class TestFillers:
    def test_hesitations_are_dropped_from_both_texts(self, ruleset: Ruleset) -> None:
        assert fold_tokens("अँ यो uh राम्रो, Mm छ उम्", ruleset) == ["यो", "राम्रो", "छ"]
        result = word_errors("यो uh राम्रो छ", "अँ यो राम्रो um छ", ruleset=ruleset)
        assert result.errors == 0
        assert result.ref_words == 3

    def test_a_missed_filler_is_not_an_error(self, ruleset: Ruleset) -> None:
        assert word_errors("hmm ठिक छ", "ठिक छ", ruleset=ruleset).errors == 0

    def test_a_word_that_looks_like_a_filler_is_kept(self, ruleset: Ruleset) -> None:
        # हम् is a hum; हम without the virama is Hindi "we". उहुँ is a "no", not a hesitation.
        assert fold_tokens("हम् हम उहुँ", ruleset) == ["हम", "उहुँ"]


class TestNumbers:
    @pytest.mark.parametrize(
        ("digits", "words"),
        [
            ("15", "पन्ध्र"),
            ("१५", "fifteen"),
            ("२६", "छब्बिस"),  # the short-i spelling
            ("45", "पैँतालीस"),
            ("6", "छ"),
            ("0", "zero"),
            ("5000", "पाँचहजार"),  # two words joined by a merge
            ("2009", "twothousandnine"),
            ("2014", "twentyfourteen"),  # read as a year
            ("1979", "nineteenseventynine"),
            ("150000", "लाखपचासहजार"),  # "lakh" alone is one lakh
            ("5000को", "पाँचहजारको"),  # the suffix must agree
            ("1st", "first"),
            ("2nd", "दोस्रो"),
        ],
    )
    def test_digits_match_the_number_spelled_out(self, digits: str, words: str) -> None:
        assert same_word(digits, words)
        assert same_word(words, digits)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("one", "एक"),  # two languages, no digits: a different word was said
            ("पाँच", "five"),
            ("45", "fiftyfour"),
            ("fortyfive", "4005"),  # not a year: the second half must be 10-99
            ("5000को", "पाँचहजारमा"),
            ("1st", "one"),  # an ordinal is not a cardinal
            ("15", "16"),
        ],
    )
    def test_different_numbers_stay_different(self, left: str, right: str) -> None:
        assert not same_word(left, right)

    def test_digit_groups_are_one_number(self, ruleset: Ruleset) -> None:
        assert fold_tokens("5,000 र 1,00,000", ruleset) == ["5000", "र", "100000"]

    @pytest.mark.parametrize(
        ("reference", "hypothesis"),
        [
            ("पाँच हजार रुपैयाँ", "5,000 रुपैयाँ"),
            ("40,000 को phone", "चालिस हजारको phone"),  # two words against two words
            ("in two thousand nine", "in 2009"),  # three words against one
            ("twenty percent", "20%"),
        ],
    )
    def test_a_number_written_either_way_is_not_an_error(
        self, ruleset: Ruleset, reference: str, hypothesis: str
    ) -> None:
        assert word_errors(reference, hypothesis, ruleset=ruleset).errors == 0
        assert word_errors(hypothesis, reference, ruleset=ruleset).errors == 0


class TestContractions:
    @pytest.mark.parametrize(
        ("contracted", "expanded"),
        [
            ("you're right", "you are right"),
            ("I'm fine", "I am fine"),
            ("I've seen", "I have seen"),
            ("we'll go", "we will go"),
            ("don't know", "do not know"),
            ("can't say", "cannot say"),
            ("it's good", "it is good"),
            ("gonna go", "going to go"),
            ("wanna go", "want to go"),
        ],
    )
    def test_a_contraction_matches_its_expansion(
        self, ruleset: Ruleset, contracted: str, expanded: str
    ) -> None:
        assert word_errors(contracted, expanded, ruleset=ruleset).errors == 0
        assert word_errors(expanded, contracted, ruleset=ruleset).errors == 0

    def test_a_possessive_is_not_expanded(self) -> None:
        # Only pronouns take 's as "is": John's stays one word, and still matches Johns.
        assert spelling_key("John's") == spelling_key("Johns")


class TestColloquialVariants:
    """fold-v3: colloquial Nepali in one form, one test group per rule group in ``fold.py``."""

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            # 1. Contracted verb forms
            ("भएको", "भाको"),
            ("भएको", "भको"),
            ("गएको", "गाको"),
            ("गएको", "गको"),
            ("आएको", "आको"),
            ("पाएको", "पाको"),
            ("लगाएको", "लाको"),
            ("गर्नुभएको", "गर्नुभाको"),
            ("भएछ", "भाछ"),
            ("भए", "भा"),
            ("भयो", "भो"),
            ("थियो", "थ्यो"),
            ("थिएँ", "थेँ"),
            ("थिइनँ", "थिएन"),
            ("होइन", "हैन"),
            ("रहेछ", "रैछ"),
            ("हुँदैन", "हुन्न"),
            ("राखेछु", "राखीछु"),
            ("दिएछु", "देछु"),
            ("लिएपछि", "लेपछि"),
            ("खै", "खोइ"),
            # 2. The western -या participle
            ("गर्या", "गरेको"),
            ("भन्या", "भनेको"),
            ("राखिराखेको", "राखिराख्या"),  # progressive and -या together
            ("गरिराखेको", "गरिराख्या"),
            # 3. The progressive
            ("गरिरहेको", "गरिराको"),
            ("गरिराखेको", "गरिराको"),
            ("भइरहेछ", "भइराछ"),
            ("गरिरहेको", "गरिरा"),
            ("गइरहेको", "गइरा"),
            ("भइरहन्थ्यो", "भइराथ्यो"),
            ("बसिरहन्थेँ", "बसिराहुन्थेँ"),
            ("भइराख्थेन", "भइराथेन"),
            # 4. The benefactive
            ("गरिदियो", "गर्दियो"),
            ("भइदिएला", "भइदेला"),
            ("गरिदिए", "गर्दिया"),
            ("छोडिदेऊ", "छोड्देऊ"),
            ("छोडिदेऊ", "छोड्दे"),
            ("हालिदिएर", "हाल्देर"),
            ("छोडिदिएँ", "छोड्देँ"),
            ("भनिदिइहाल्छु", "भन्दिहाल्छु"),
            ("बेचिदिइराखेको", "बेच्दिराखेको"),
            # 5. First person plural
            ("भनौँ", "भनुम्"),
            ("भनौँ", "भनम्"),
            ("भनौँ", "भनूँ"),
            ("गरौँ", "गरूँ"),
            ("जाऊँ", "जाऔँ"),
            ("जाऊँ", "जाम्"),
            ("गर्यौँ", "गरेम्"),
            ("खायौँ", "खाएम्"),
            ("निस्क्यौँ", "निस्किम्"),
            ("दिन्छौँ", "दिन्छम्"),
            ("छौँ", "छम्"),
            ("थियौँ", "थिम्"),
            # 6. Contracted pronouns
            ("उसले", "उल्ले"),
            ("त्यसलाई", "तेल्लाई"),
            ("त्यसलाई", "त्यलाई"),
            ("जसले", "जोले"),
            ("जसले", "जल्ले"),
            ("यसले", "एले"),
            ("अरूले", "अर्ले"),
            ("कसले", "कस्ले"),
            # 7. लाउनु for लगाउनु
            ("लगाएर", "लाएर"),
            ("लगाउने", "लाउने"),
            ("लगाउँछ", "लाउँछ"),
            ("नलगाउने", "नलाउने"),
            ("लगाइदिनुभयो", "लाइदिनुभयो"),
            # 8. Emphatic -ै and doubled consonants
            ("मात्र", "मात्रै"),
            ("एकदम", "एकदमै"),
            ("अझ", "अझै"),
            ("आज", "आजै"),
            ("सुरुमा", "सुरुमै"),
            ("मा", "मै"),
            ("बिस्तारै", "बिस्तारो"),
            ("लिएर", "लिएरै"),
            ("बाहिर", "बाहिरै"),
            ("सकेसम्म", "सक्केसम्म"),
            ("भेटाएको", "भेट्टाएको"),
            ("खतरा", "खत्रा"),
            ("खतरा", "खत्त्रै"),
            ("बबाल", "बब्बाल"),
            ("कत्तिको", "कतिको"),
            ("मज्जाले", "मजाले"),
            ("आफैँले", "आफैले"),
            ("सबैभन्दा", "सबभन्दा"),
            # 9. Loose pairs, folded by the owner's choice
            ("पनि", "नि"),
            ("गर्नु", "गर्न"),
            ("बाँच्नुलाई", "बाँच्नलाई"),
            ("भन्दा", "भन्दाखेरि"),
            ("हेर्दा", "हेर्दाखेरि"),
            ("अलि", "अलिकति"),
            ("अलि", "अलिकता"),
            ("अलि", "अलिक"),
            ("अलि", "अलिअलि"),
            ("यहाँ", "या"),
            ("छैन", "छुइनँ"),
            ("खायो", "खाइयो"),
            ("देख्यो", "देखियो"),
            # 10. Spoken forms not yet seen in the corpus
            ("जान्दिनँ", "जान्न"),
            ("भन्ने", "भन्नि"),
            ("भनेपछि", "भनेसि"),
            ("गर्छु", "गर्चु"),
            ("गर्ने", "गर्न्या"),
            ("हुने", "हुन्या"),
            ("ल्याएको", "ल्याको"),
            ("बोलाएको", "बोलाको"),
            # 11. Table words keep their fold under a suffix
            ("पहिलेदेखि", "पहिलादेखि"),
            ("पहिलेदेखि", "पहिल्यैदेखि"),
            ("रुपैयाँको", "रुपियाँको"),
            ("पहिले", "पहिला"),
            ("रुपैयाँ", "रुपियाँ"),
            ("सम्पन्न", "संपन्न"),  # orthography: a nasal consonant against anusvara
            ("घण्टा", "घंटा"),
            ("गर्नुहोस्", "गर्नुस्"),
            ("हुनुपऱ्यो", "हुनपऱ्यो"),
            ("गर्नुको", "गर्नको"),
        ],
    )
    def test_colloquial_spellings_of_one_word_are_the_same_word(
        self, left: str, right: str
    ) -> None:
        assert same_word(left, right)
        assert same_word(right, left)

    @pytest.mark.parametrize(
        ("standard", "spoken"),
        [
            ("भएको थियो", "भाथ्यो"),
            ("आएको थियो", "आथ्यो"),
            ("गएको थिएँ", "गाथेँ"),
            ("लागेको थियो", "लागेथ्यो"),
            ("गरिरहेको छ", "गरिराछ"),
            ("गरिरहेको छु", "गरिराछु"),
            ("गरिरहेको थिएँ", "गरिराथेँ"),
            ("भएको छैन", "भाछैन"),
            ("गरे पनि", "गरेनि"),
            ("भए पनि", "भएनि"),
        ],
    )
    def test_one_spoken_word_matches_two_written_words(self, standard: str, spoken: str) -> None:
        assert word_errors(standard, spoken).errors == 0
        assert word_errors(spoken, standard).errors == 0

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("क्या", "केको"),  # Hindi "what" is not "of what"
            ("गर्दिन", "गरिदिन"),  # "I don't do" is not "to do for"
            ("भन्ने", "भने"),
            ("गर्नु", "गर्ने"),  # infinitive and participle
            ("छौँ", "छौ"),
            ("बल्ले", "बसले"),  # "at last" is not "by bus"
            ("लाइन", "लगाइन"),  # "line"
            ("अनि", "पनि"),
            ("गर्दिन", "गर्दिनु"),  # "I don't do" is not "to do for"
            ("हुँ", "हौँ"),  # "I am"
            ("कीरा", "कीराको"),
        ],
    )
    def test_different_words_stay_different(self, left: str, right: str) -> None:
        assert not same_word(left, right)


class TestTokens:
    def test_a_lone_quote_mark_is_not_a_word(self, ruleset: Ruleset) -> None:
        assert fold_tokens("उसले ' हो ' भन्यो", ruleset) == ["उसले", "हो", "भन्यो"]

    def test_a_three_way_spacing_split_is_not_an_error(self, ruleset: Ruleset) -> None:
        assert word_errors("X Y Z", "XYZ", ruleset=ruleset).errors == 0


class TestSimilarity:
    def test_identical_and_disjoint(self) -> None:
        assert similarity("राम्रो", "राम्रो") == 1.0
        assert similarity("ठीक", "shit") < 0.6


def test_the_version_names_the_orthography_table_it_was_built_on(ruleset: Ruleset) -> None:
    # A reported WER is only reproducible with both halves: the fold rules and the D64 table.
    assert fold_version(ruleset) == f"{FOLD_VERSION}+test-v1"
