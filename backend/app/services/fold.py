"""Script-folding comparison of transcripts: the one way this repository says two texts agree.

Two correct transcripts of the same Nepanglish clip routinely disagree on how a word is *written*
rather than on what was *said*. Scribe and MAI spell English phonetically in Devanagari (`एक्टिभ`
for active, `कफी` for coffee) whatever they are told; the annotator writes `चैँ` one day and
`चाहिँ` the next; `गर्नुभयो` is `गर्नु भयो` with the space left out. A plain word error rate charges
every one of these as a recognition error. Measured on the gold pot, about half of every system's
substitutions were respellings of this kind -- noise that would otherwise sit in every reported
WER and every disagreement signal.

This module removes that noise and nothing else. It is a *comparison*, never a rewrite: no label
or hypothesis text is changed by it. The D64 table in ``normalize.py`` is still what decides the
exported spelling; folding sits on top of it and is only ever consulted when two texts are
compared -- by the fusion hazard gates, by the queue score, and by any WER this corpus reports.

Before any word is compared, the text is split into words with the D64 table applied, and three
kinds of token leave it: punctuation (a lone ``'`` used as a quote mark included), digit-group
commas (``5,000`` is one number), and **fillers** -- the hesitations and hums in :data:`FILLERS`.
Fillers are dropped from both texts, as Whisper's normalizer drops them and NIST scoring makes
them optionally deletable: a filler carries no word, and which of `अँ`, ``uh`` and ``um`` a
transcriber wrote, or whether they wrote one at all, is not what a WER is for. ``%`` becomes the
word ``percent``.

Four rules, in the order they are tried on a pair of words:

1. **Spelling.** Both words are reduced to a spelling key -- NFC, Latin lowercased, and the
   Devanagari distinctions Nepali writers do not hold consistently folded away: vowel length
   (ि/ी, ु/ू), chandrabindu against anusvara, nukta, a word-final virama, a nasal consonant
   against anusvara (`सम्पन्न`/`संपन्न`), and the script of the digits. The key also expands English
   contractions (``you're`` is ``youare``, so it matches ``you are`` through a merge) and writes
   colloquial Nepali in one form (:data:`_COLLOQUIAL_WORDS`, :data:`_COLLOQUIAL`): `हैन`/`होइन`,
   `भाको`/`भएको`, `गरिराको`/`गरिरहेको`, `गर्दियो`/`गरिदियो`, `गर्या`/`गरेको`. Equal keys are the same
   word.
2. **Numbers.** Digits match the number spelled out in either language -- `15`, `पन्ध्र` and
   ``fifteen``; `5,000` and `पाँच हजार`; `2009` and ``two thousand nine`` -- provided any suffix
   agrees (``40,000 को`` and `चालिस हजारको`). One side must be written in digits: ``one`` and `एक`
   are two languages, and a speaker said one of them.
3. **Across scripts, by sound.** When one word is Latin and the other Devanagari (or a Latin stem
   carrying a Devanagari suffix, ``phoneमा``), both are reduced to a coarse consonant skeleton and
   compared. This is the rule that makes `एक्टिभ` and ``active`` one word.
4. **Never by sound within one script.** `गर्नु` and `गर्ने` share a skeleton and are different
   words -- an infinitive and a participle. Matching Devanagari against Devanagari on sound would
   erase exactly the grammar a WER is supposed to see, so rule 3 is only ever applied across
   scripts. The colloquial table is the only same-script equivalence, and every entry in it was
   checked against the corpus vocabulary for words it would wrongly join.

Alignment also accepts zero-cost merges, which is how a spacing difference (`गर्नुभयो` /
`गर्नु भयो`, ``Facebook`` / `फेस बुक`) stops costing two errors. Two words may merge against one by
any of the rules above; wider merges, up to three words a side, only when the joined words have
the same spelling key or are the same number (``two thousand nine`` / `2009`). Wider merges by
sound were tried and matched unrelated phrases.

The cost, accepted deliberately: a WER computed here cannot see a wrong script. A system that
writes ``Captain`` as `क्याप्टन` scores the same as one that follows the policy. The script policy
is enforced where text is produced -- by the fuser -- and not in the metric. ``fold_version``
names both halves of the rule set so a reported number can be reproduced.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from app.services.normalize import Ruleset, load_ruleset, normalize_text

#: Bumped whenever a rule below changes what counts as the same word.
FOLD_VERSION = "fold-v3"

#: Romanized forms (vowels kept) are accepted as one word at or below this normalized distance.
#: The EDA notebook swept 0.25-0.6; this is the conservative end that still catches inflection.
CROSS_SCRIPT_THRESHOLD = 0.34
#: A one-consonant skeleton collides with too much of the language to be evidence of anything.
MIN_SKELETON = 2
#: Below this romanized length, compare whole romanized strings rather than skeletons.
SHORT_TOKEN = 4
#: The most adjacent words, on either side, that alignment will join into one (``two thousand
#: nine`` / `2009`).
MAX_MERGE = 3

_DEV = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
#: Everything that is not part of a word, in either script. The danda and double danda
#: (U+0964, U+0965) are cut out of the Devanagari range for the same reason as in ``normalize.py``.
_NON_WORD = re.compile(r"[^A-Za-z0-9'ऀ-ॣ०-ॿ]+")
_JOINERS = str.maketrans("", "", "‌‍")

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_DEV_FOLD = str.maketrans(
    {
        "़": None,  # nukta
        "ँ": "ं",  # chandrabindu -> anusvara
        "ी": "ि",  # ी -> ि
        "ू": "ु",  # ू -> ु
        "ई": "इ",  # ई -> इ
        "ऊ": "उ",  # ऊ -> उ
    }
)
_VIRAMA = "्"

#: Devanagari to ASCII, longest match first. Deliberately crude -- it feeds a skeleton and a
#: similarity ratio, never a reader -- and it writes no inherent vowel, which a skeleton would
#: drop anyway. Ported from the old 01-EDA notebook (section 8d, now in git history).
_ROMAN = {
    "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gy", "श्र": "shr",
    "ख": "kh", "घ": "gh", "छ": "ch", "झ": "jh", "ठ": "th", "ढ": "dh", "थ": "th", "ध": "dh",
    "फ": "ph", "भ": "bh", "क": "k", "ग": "g", "ङ": "n", "च": "c", "ज": "j", "ञ": "n",
    "ट": "t", "ड": "d", "ण": "n", "त": "t", "द": "d", "न": "n", "प": "p", "ब": "b",
    "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s",
    "ह": "h", "ळ": "l", "ऱ": "r",
    "अ": "a", "आ": "a", "इ": "i", "उ": "u", "ऋ": "ri", "ए": "e",
    "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e",
    "ा": "a", "ि": "i", "ु": "u", "ृ": "ri", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o",
    "्": "", "ं": "n", "ः": "h", "ऽ": "",  # noqa: RUF001 -- visarga, not a colon
}  # fmt: skip
_ROMAN_RE = re.compile("|".join(re.escape(k) for k in sorted(_ROMAN, key=len, reverse=True)))

#: English spellings whose sound Nepali writes differently, rewritten before the skeleton is
#: taken: ``police`` is पुलिस, ``station`` is स्टेसन, ``budget`` is बजेट.
_LATIN_SOUNDS = (
    (re.compile(r"dg(?=[eiy])"), "j"),
    (re.compile(r"c(?=[eiy])"), "s"),
    (re.compile(r"tion"), "shan"),
)
#: English g before e/i/y is soft in ``general`` (जनरल) and hard in ``girl`` (गर्ल), and spelling
#: cannot tell which. A Latin word gets both skeletons and either may match.
_SOFT_G = re.compile(r"g(?=[eiy])")
#: How Nepali is usually typed in Latin script, folded toward the crude romanization above so a
#: recogniser that romanized Nepali (``chha`` for छ) is compared on sound too.
_ROMANIZED_NEPALI = (("chh", "ch"), ("aa", "a"), ("ee", "i"), ("oo", "u"))
_DIGRAPHS = (("ph", "p"), ("bh", "b"), ("kh", "k"), ("gh", "k"),
             ("th", "t"), ("dh", "t"), ("ch", "k"), ("sh", "s"))  # fmt: skip
#: Vowels and ``y`` leave the skeleton: ``type`` and टाइप, ``Captain`` and क्याप्टन.
_SKELETON_DROP = str.maketrans("", "", "aeiouy")
#: Nepali has no v/w contrast and renders English f, z and th with फ, ज and थ.
_CONSONANT_CLASS = str.maketrans(
    {
        "w": "b", "v": "b", "f": "p", "z": "j", "q": "k", "x": "k", "c": "k", "g": "k",
        "d": "t", "m": "n", "l": "r",
    }
)  # fmt: skip
_REPEATS = re.compile(r"(.)\1+")

#: Hesitations and hums, dropped before alignment. Matched after NFC, lowercasing and apostrophe
#: removal but *before* the spelling key, which would strip the virama that separates the hum
#: हम् from Hindi हम ("we"). उहुँ ("no") and अहँ ("no") are answers, not fillers, and stay.
FILLERS = frozenset(
    {"अँ", "अं", "अ", "उम्", "अम्", "हम्", "हुम्", "उहुम्", "अह",
     "uh", "uhh", "um", "umm", "uhm", "er", "erm", "hmm", "hm", "mm", "mmm", "mhm"}
)  # fmt: skip
_DIGIT_GROUP = re.compile(r"(?<=\d),(?=\d)")

#: English contractions, written as the words they stand for so that a merge matches the
#: expansion. ``'s`` is only ``is`` after these pronouns: ``John's`` is a possessive.
_CONTRACTED_WORDS = {
    "gonna": "goingto", "wanna": "wantto", "gotta": "gotto", "gimme": "giveme",
    "lemme": "letme", "kinda": "kindof", "sorta": "sortof", "outta": "outof",
    "dunno": "donotknow", "let's": "letus", "can't": "cannot", "won't": "willnot",
}  # fmt: skip
_CONTRACTION = re.compile(r"([a-z]+)'(m|re|ve|ll|s)|([a-z]+)n't")
_CONTRACTION_TAILS = {"m": "am", "re": "are", "ve": "have", "ll": "will", "s": "is"}
_IS_PRONOUNS = frozenset(
    {"it", "that", "there", "what", "he", "she", "here", "where", "who", "how"}
)

#: Colloquial Nepali (D84, D89). Spoken Nepali contracts what the written language spells out,
#: and a transcriber may write either. Each group below is one kind of spoken form: a table of
#: whole words (spoken -> written, before any rule runs) and the productive rewrites it needs.
#: The groups are applied in the order of :data:`_COLLOQUIAL_GROUPS`; the comment on a group says
#: when that order matters. fold-v3 (D89) folds every group the owner listed, loose pairs
#: included, and leaves tightening to listening.
_C = "[क-ह]"

#: 1. Contracted verb forms: भाको, भाछ, थ्यो, थेँ, रैछ, हुन्न.
_CONTRACTED_VERB_WORDS = {
    "हैन": "होइन", "भो": "भयो", "भा": "भए", "भको": "भएको", "नभको": "नभएको", "गको": "गएको",
    "लाको": "लगाएको", "थ्यो": "थियो", "थ्यौँ": "थियौँ", "थेँ": "थिएँ", "थिएन": "थिइनँ",
    "थेन": "थिइनँ", "हुन्न": "हुँदैन", "खोइ": "खै", "चै": "चाहिँ",
}  # fmt: skip
_CONTRACTED = (
    # Perfect participle: भएको -> भाको, गएका -> गाका, आएको -> आको, ल्याएको -> ल्याको.
    (rf"(?<={_C})एक(?=[ोा])", "ाक"),
    (r"(?<=[ाआ])एक(?=[ोा])", "क"),
    # Mirative on the same stems: भएछ -> भाछ, आएछ -> आछ.
    (rf"(?<={_C})एछ", "ाछ"),
    (r"(?<=[ाआ])एछ", "छ"),
    (r"(?<=[दल])िए(?=छ|पछि)", "े"),  # दिएछु -> देछु, लिएपछि -> लेपछि
    (rf"(?<={_C})िछु$", "ेछु"),  # राखीछु -> राखेछु
    (r"रैछ", "रहेछ"),
)

#: 2. The western -या participle: गर्या -> गरेको. Runs before the progressive, which only
#: recognises गरिराख्या once it reads गरिराखेको.
_WESTERN = (
    (r"(?<=्न)्या$", "े"),  # गर्न्या -> गर्ने
    # Two letters before it, so क्या ("what") is spared.
    (rf"(?<=.{_C})्या$", "ेको"),
)

#: 3. The progressive: गरिरहेको, गरिराखेको, गरिरा -> गरिराको.
_PROGRESSIVE = (
    (r"(?<=[िइ])र(?:हे|ाखे)", "रा"),
    # A bare -रा needs a stem before -इ-, so कीरा ("insect") and हीरा ("diamond") are spared.
    (r"(?<=..ि)रा$", "राको"),
    (r"(?<=.इ)रा$", "राको"),
    (r"(?<=[िइ])र(?:हं|ाहुं|ाख्)थ", "राथ"),  # भइरहन्थ्यो -> भइराथ्यो (after the nasal rule)
    (r"(?<=[िइ])राहुं", "रहं"),  # हिँडिराहुन्छु -> हिँडिरहन्छु
)

#: 4. The benefactive: गरिदियो -> गर्दियो, हालिदिएर -> हाल्देर.
_BENEFACTIVE = (
    # Not a final -दिन, where गर्दिन is "I don't do".
    (rf"(?<={_C})िदि(?!न$)", "्दि"),
    (rf"(?<={_C})िदे", "्दे"),
    (r"दिइ(?=र|ह)", "दि"),  # बेचिदिइराखेको -> बेच्दिराखेको
    (r"(?<=्द)िया$", "िए"),  # गर्दिया -> गर्दिए
    (r"्दिए(?=र$|ं$|$)", "्दे"),
    (r"दिए(?=क[ोा]|ला)", "दे"),
    (r"्देउ$", "्दे"),  # छोड्देऊ -> छोड्दे
)

#: 5. First person plural, in one spelling: भनूँ, भनौँ -> भनौं; जाऊँ, जाऔँ -> जाऔं. The -म् forms
#: (भनुम्, गरेम्, जाम्) are in :func:`_first_plural`, because the virama that marks them is gone
#: from the key.
_FIRST_PLURAL = (
    (rf"(?<=.{_C})ुं$", "ौं"),  # two letters: हुँ ("I am") is not हौँ
    (r"(?<=[ािीेो])(?:उं|औं)$", "ौं"),
    (rf"(?<={_C})ियौं$", "्यौं"),  # थियौँ -> थ्यौं
)

#: 6. Contracted pronouns. A table: -ल्ले is also बल्ले ("at last").
_PRONOUN_WORDS = {
    "उल्ले": "उसले", "तेल्लाई": "त्यसलाई", "त्यलाई": "त्यसलाई", "जोले": "जसले", "जल्ले": "जसले",
    "एले": "यसले", "यल्ले": "यसले", "त्यल्ले": "त्यसले", "अर्ले": "अरूले", "कस्ले": "कसले",
}  # fmt: skip

#: 7. लाउनु for लगाउनु. Runs before the participle rules, so लाएको meets लगाएको. लाइ- only
#: before a verb, so लाइन ("line") and लाइक ("like") are spared.
_LAUNU = (
    (r"^(न?)ला(?=उ|ए)", r"\1लगा"),
    (r"^(न?)लाइ(?=दि|स|हाल|रा|रह)", r"\1लगाइ"),
)

#: 8. The emphatic -ै, and doubled consonants. Both are audible and are in by the owner's choice.
_EMPHATIC_WORDS = {
    "मात्रै": "मात्र", "एकदमै": "एकदम", "ठ्याक्कै": "ठ्याक्क", "भर्खरै": "भर्खर", "अझै": "अझ",
    "आजै": "आज", "बाहिरै": "बाहिर", "मेरै": "मेरो", "मै": "मा", "बिस्तारो": "बिस्तारै",
    "पहिल्यै": "पहिले", "खत्रा": "खतरा", "खत्त्रै": "खतरा", "बब्बाल": "बबाल",
    "सक्केसम्म": "सकेसम्म",
}  # fmt: skip
_EMPHATIC = (
    (r"(?<=..)मै$", "मा"),  # सुरुमै -> सुरुमा
    (r"एरै$", "एर"),  # लिएरै -> लिएर
    # Emphatic gemination and doublets: कत्तिको -> कतिको, मज्जाले -> मजाले, आफैँले -> आफैले.
    (r"^(क|त्य|य|उ|ज)त्ति", r"\1ति"),
    (r"^मज्जा", "मजा"),
    (r"^आफैं", "आफै"),
    (r"^भेट्टा", "भेटा"),
    (r"^(फुल|बिल|पेल)्ल", r"\1"),
    (r"^सब(?=भंदा|ले$)", "सबै"),  # after the nasal rule has made भन्दा भंदा
)

#: 9. Loose pairs: forms that may also be a different word (नि is a particle, या is "or"). The
#: owner folds them and will tighten by ear.
_LOOSE_WORDS = {
    "नि": "पनि", "या": "यहाँ", "छुइनँ": "छैन", "अलिकति": "अलि", "अलिकता": "अलि", "अलिक": "अलि",
    "अलिअलि": "अलि", "अलिकत्ति": "अलि",
}  # fmt: skip
_LOOSE = (
    # Infinitive: गर्नु -> गर्न, हुनुपऱ्यो -> हुनपऱ्यो, बाँच्नुलाई -> बाँच्नलाई.
    # Not after the benefactive: गर्दिनु ("to do for") is not गर्दिन ("I don't do").
    (r"(?<=[्ािुउ])(?<!दि)नु(?=$|लाइ$|मा$)", "न"),
    (r"नु(?=प[रऱ]|को$)", "न"),
    (r"नुहोस$", "नुस"),  # गर्नुहोस् -> गर्नुस्
    (r"दाखेरि?$", "दा"),  # भन्दाखेरि -> भन्दा
    (r"(?<=ा)इयो$", "यो"),  # खाइयो -> खायो
    (rf"(?<={_C})ियो$", "्यो"),  # देखियो -> देख्यो; थियो and थ्यो meet here too
)

#: 10. Spoken forms the corpus has not yet held: जान्न, भन्नि, भनेसि, गर्चु.
_UNSEEN_WORDS = {"जान्न": "जान्दिनँ", "भन्नि": "भन्ने", "हुन्या": "हुने"}
_UNSEEN = (
    (r"(?<=े)सि$", "पछि"),  # भनेसि -> भनेपछि
    (r"(?<=्)च(?=ु$|ौं$)", "छ"),  # गर्चु -> गर्छु
)

#: One spoken word for two written ones, so that a merge matches them: भाथ्यो / भएको थियो,
#: गरिराछ / गरिरहेको छ, गरेनि / गरे पनि. Runs last, on the participles the groups above made.
_JOINED = (
    (r"ाको(?=छ)", "ा"),
    (r"कोथिएं$", "कोथें"),
    (r"कोथिए$", "कोथे"),
    (r"को(?=थ्यो$|थें$|थे$)", ""),
    (r"(?<=[ेए])नि$", "पनि"),
)

#: A nasal consonant before its own class is an anusvara: सम्पन्न -> संपन्न, घण्टा -> घंटा.
#: Orthography rather than dialect. First, because later rules read the anusvara, and again last,
#: for the nasal clusters the benefactive makes (भनिदिइ -> भन्दि).
_NASAL = ((r"ङ्(?=[कखगघ])|ञ्(?=[चछजझ])|ण्(?=[टठडढ])|न्(?=[तथदध])|म्(?=[पफबभ])", "ं"),)

_OTHER_WORDS = {
    "पहिला": "पहिले", "पहिलाको": "पहिलेको", "रुपियाँ": "रुपैयाँ", "बुवा": "बुबा", "बिहा": "बिहे",
    "जवाब": "जवाफ", "गलती": "गल्ती", "हजुरबा": "हजुरबुबा",
}  # fmt: skip

_COLLOQUIAL_WORDS = {
    **_CONTRACTED_VERB_WORDS, **_PRONOUN_WORDS, **_EMPHATIC_WORDS, **_LOOSE_WORDS, **_UNSEEN_WORDS,
    **_OTHER_WORDS,
}  # fmt: skip
_COLLOQUIAL_GROUPS = (
    _NASAL, _WESTERN, _LAUNU, _CONTRACTED, _PROGRESSIVE, _BENEFACTIVE, _FIRST_PLURAL, _EMPHATIC,
    _LOOSE, _UNSEEN, _JOINED, _NASAL,
)  # fmt: skip
_COLLOQUIAL = tuple(
    (re.compile(pattern), repl) for group in _COLLOQUIAL_GROUPS for pattern, repl in group
)
#: A table word keeps its fold under a case ending: रुपियाँको -> रुपैयाँको, पहिलादेखि -> पहिलेदेखि.
_TABLE_SUFFIXES = ("देखि", "सम्म", "लाइ", "बाट", "संग", "हरु", "को", "का", "कि", "मा", "ले")
#: -म् first person plural endings, read on the key once the virama is gone.
_FIRST_PLURAL_M = tuple(
    (re.compile(pattern), repl)
    for pattern, repl in (
        (r"ेम$", "्यौं"),  # गरेम् -> गर्यौं
        (r"एम$", "यौं"),  # खाएम् -> खायौं
        (r"^थिम$", "थ्यौं"),
        (r"(?<=[दल]ि)म$", "ौं"),  # छोड्दिम् -> छोड्दिऊँ
        (rf"(?<={_C})िम$", "्यौं"),  # निस्किम् -> निस्क्यौं
        (r"(?<=[ा])म$", "ौं"),  # जाम् -> जाऔं
        (rf"(?<={_C})ु?म$", "ौं"),  # भनुम्, भनम् -> भनौं
    )
)

#: Number words, by value. Nepali has a distinct word for every number below a hundred.
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "lakh": 10**5, "million": 10**6, "crore": 10**7, "billion": 10**9,
    **{word: value for value, word in enumerate((  # noqa: SIM905 -- a hundred words read best as prose
        "शून्य एक दुई तीन चार पाँच छ सात आठ नौ दस एघार बाह्र तेह्र चौध पन्ध्र सोह्र सत्र अठार "
        "उन्नाइस बीस एक्काइस बाइस तेइस चौबीस पच्चीस छब्बीस सत्ताइस अट्ठाइस उनन्तीस तीस एकतीस "
        "बत्तीस तेत्तीस चौँतीस पैँतीस छत्तीस सैँतीस अठतीस उनन्चालीस चालीस एकचालीस बयालीस "
        "त्रिचालीस चवालीस पैँतालीस छयालीस सतचालीस अठचालीस उनन्चास पचास एकाउन्न बाउन्न त्रिपन्न "
        "चउन्न पचपन्न छपन्न सन्ताउन्न अन्ठाउन्न उनन्साठी साठी एकसट्ठी बयसट्ठी त्रिसट्ठी चौसट्ठी "
        "पैँसट्ठी छयसट्ठी सतसट्ठी अठसट्ठी उनन्सत्तरी सत्तरी एकहत्तर बहत्तर त्रिहत्तर चौहत्तर "
        "पचहत्तर छयहत्तर सतहत्तर अठहत्तर उनासी असी एकासी बयासी त्रियासी चौरासी पचासी छयासी सतासी "
        "अठासी उनान्नब्बे नब्बे एकानब्बे बयानब्बे त्रियानब्बे चौरानब्बे पन्चानब्बे छयानब्बे "
        "सन्तानब्बे अन्ठानब्बे उनान्सय"
    ).split())},
    "सय": 100, "हजार": 1000, "लाख": 10**5, "करोड": 10**7, "अर्ब": 10**9,
}  # fmt: skip
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "पहिलो": 1, "दोस्रो": 2, "तेस्रो": 3, "चौथो": 4,
}  # fmt: skip
_MULTIPLIERS = frozenset({1000, 10**5, 10**6, 10**7, 10**9})


@dataclass(frozen=True)
class AlignOp:
    """One step of a word alignment.

    Attributes:
        kind: ``match`` (same spelling key), ``fold`` (same word in another script), ``merge``
            (two words on one side are one on the other), ``sub``, ``del`` or ``ins``.
        ref: The reference words this step consumed; empty for an insertion.
        hyp: The hypothesis words this step consumed; empty for a deletion.
        similarity: Romanized similarity of a substitution's two words, 0-1. Lets a caller tell
            a near-miss spelling from an unrelated word; 1.0 for every non-substitution.
    """

    kind: str
    ref: tuple[str, ...]
    hyp: tuple[str, ...]
    similarity: float = 1.0

    @property
    def is_error(self) -> bool:
        return self.kind in {"sub", "del", "ins"}


@dataclass
class Alignment:
    """A word alignment and the error counts it implies."""

    ops: list[AlignOp] = field(default_factory=list)

    @property
    def substitutions(self) -> int:
        return sum(1 for op in self.ops if op.kind == "sub")

    @property
    def deletions(self) -> int:
        return sum(1 for op in self.ops if op.kind == "del")

    @property
    def insertions(self) -> int:
        return sum(1 for op in self.ops if op.kind == "ins")

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def ref_words(self) -> int:
        return sum(len(op.ref) for op in self.ops)

    @property
    def hyp_words(self) -> int:
        return sum(len(op.hyp) for op in self.ops)

    @property
    def rate(self) -> float:
        """Errors per reference word. An empty reference is 0.0 if the hypothesis is empty too,
        and 1.0 otherwise -- insertions over nothing have no denominator to divide by."""
        if self.ref_words == 0:
            return 0.0 if self.hyp_words == 0 else 1.0
        return self.errors / self.ref_words


@lru_cache(maxsize=1)
def _default_ruleset() -> Ruleset:
    return load_ruleset()


def fold_version(ruleset: Ruleset | None = None) -> str:
    """The identifier a reported number needs to be reproduced: fold rules plus the D64 table."""
    return f"{FOLD_VERSION}+{(ruleset or _default_ruleset()).version}"


def fold_tokens(text: str | None, ruleset: Ruleset | None = None) -> list[str]:
    """Split text into comparable words: D64 table applied, punctuation, danda and fillers removed.

    Words keep their original spelling and case; everything else about comparison happens in
    :func:`spelling_key` and :func:`same_word`. A Latin stem carrying a Devanagari suffix
    (``phoneमा``) stays one word, as it was written. Digit groups are joined (``5,000`` is
    ``5000``) and ``%`` is the word ``percent``.
    """
    if not text:
        return []
    text = unicodedata.normalize("NFC", text).translate(_JOINERS)
    text = normalize_text(text, ruleset or _default_ruleset()) or ""
    text = _DIGIT_GROUP.sub("", text).replace("%", " percent ")
    return [
        token
        for token in _NON_WORD.sub(" ", text).split()
        if not is_filler(token) and spelling_key(token)
    ]


def is_filler(token: str) -> bool:
    """Whether the token is a hesitation or hum from :data:`FILLERS`."""
    return unicodedata.normalize("NFC", token).lower().replace("'", "") in FILLERS


@lru_cache(maxsize=65536)
def spelling_key(token: str) -> str:
    """The word with every spelling distinction this corpus does not hold consistently removed."""
    key = _orthographic_key(token)
    if _DEV.search(key):
        key = _table_word(key)
        if token.endswith("म्") and len(key) >= 2:
            key = _first_plural(key)
        for pattern, repl in _COLLOQUIAL:
            key = pattern.sub(repl, key)
        key = _table_word(key)
    return key


def _table_word(key: str) -> str:
    if key in _COLLOQUIAL_KEYS:
        return _COLLOQUIAL_KEYS[key]
    for suffix in _TABLE_SUFFIXES:
        stem = key.removesuffix(suffix)
        if stem != key and len(stem) > 2 and stem in _COLLOQUIAL_KEYS:
            return _COLLOQUIAL_KEYS[stem] + suffix
    return key


def _first_plural(key: str) -> str:
    for pattern, repl in _FIRST_PLURAL_M:
        folded = pattern.sub(repl, key)
        if folded != key:
            return folded
    return key


def _orthographic_key(token: str) -> str:
    """Folding that is about the script, not the dialect: everything but :data:`_COLLOQUIAL`."""
    text = unicodedata.normalize("NFC", token).translate(_JOINERS).translate(_DEV_DIGITS)
    text = unicodedata.normalize("NFD", text).translate(_DEV_FOLD)
    text = unicodedata.normalize("NFC", text).lower()
    if not _DEV.search(text):
        text = _expand_contraction(text)
    return text.replace("'", "").removesuffix(_VIRAMA)


def _expand_contraction(word: str) -> str:
    if word in _CONTRACTED_WORDS:
        return _CONTRACTED_WORDS[word]
    match = _CONTRACTION.fullmatch(word)
    if not match:
        return word
    stem, tail, negated = match.groups()
    if negated:
        return negated + "not"
    if tail == "s" and stem not in _IS_PRONOUNS:
        return word
    return stem + _CONTRACTION_TAILS[tail]


_COLLOQUIAL_KEYS = {
    _orthographic_key(word): _orthographic_key(form) for word, form in _COLLOQUIAL_WORDS.items()
}
_NUMBER_KEYS = {spelling_key(word): value for word, value in _NUMBER_WORDS.items()}
_ORDINAL_KEYS = {spelling_key(word): value for word, value in _ORDINAL_WORDS.items()}
_NUMBER_ATOM = re.compile(
    r"[0-9]+|" + "|".join(re.escape(w) for w in sorted(_NUMBER_KEYS, key=len, reverse=True))
)
_DIGIT_ORDINAL = re.compile(r"([0-9]+)(?:st|nd|rd|th)")


def _combine(values: list[int]) -> int | None:
    """The value of a run of number words in speaking order, or None if the run is not a number.

    ``five hundred forty two`` is 542 and ``एक लाख पचास हजार`` is 150000; ``five twenty`` and
    ``forty fifty`` are not numbers, so a units word may only follow a round ten or hundred.
    """
    total = current = 0
    for value in values:
        if value == 100:
            if current >= 100:
                return None
            current = (current or 1) * 100
        elif value in _MULTIPLIERS:
            total += (current or 1) * value
            current = 0
        else:
            round_ten = 20 <= current < 100 and current % 10 == 0 and value < 10
            if current and not (current % 100 == 0 or round_ten):
                return None
            current += value
    return total + current


@lru_cache(maxsize=65536)
def _number(key: str) -> tuple[frozenset[int], str, bool, bool] | None:
    """Read a spelling key as a number: (values, suffix, written in digits, ordinal) or None.

    Several values when the words allow two readings -- ``twenty fourteen`` is 34 as arithmetic
    and 2014 as a year, and only the year is kept because the arithmetic breaks :func:`_combine`.
    """
    if key in _ORDINAL_KEYS:
        return frozenset({_ORDINAL_KEYS[key]}), "", False, True
    if ordinal := _DIGIT_ORDINAL.fullmatch(key):
        return frozenset({int(ordinal.group(1))}), "", True, True
    atoms: list[int] = []
    digits = False
    position = 0
    while match := _NUMBER_ATOM.match(key, position):
        atom = match.group(0)
        if atom[0].isdigit():
            if atoms:  # digits lead a number ("5 हजार"), never follow a word
                break
            digits = True
            atoms.append(int(atom))
        else:
            atoms.append(_NUMBER_KEYS[atom])
        position = match.end()
    if not atoms:
        return None
    values = set()
    if (value := _combine(atoms)) is not None:
        values.add(value)
    if not digits and len(atoms) >= 2 and 10 <= atoms[0] < 100:
        rest = _combine(atoms[1:])
        if rest is not None and 10 <= rest < 100:
            values.add(atoms[0] * 100 + rest)  # a year: nineteen seventy nine
    return (frozenset(values), key[position:], digits, False) if values else None


def is_number(token: str) -> bool:
    """Whether a word reads as a number -- digits, a number word, an ordinal -- by rule 2."""
    return _number(spelling_key(token)) is not None


def _same_number(a: str, b: str) -> bool:
    """Rule 2: one number, digits on at least one side, and the same suffix."""
    left, right = _number(spelling_key(a)), _number(spelling_key(b))
    if not left or not right or not (left[2] or right[2]):
        return False
    same_kind = left[1] == right[1] and left[3] == right[3]  # suffix, and ordinal or cardinal
    return same_kind and bool(left[0] & right[0])


def _script(token: str) -> str:
    dev, lat = bool(_DEV.search(token)), bool(_LATIN.search(token))
    if dev and lat:
        return "mix"
    return "dev" if dev else "lat" if lat else "none"


@lru_cache(maxsize=65536)
def _ascii(token: str) -> str:
    """Romanized, lowercase ASCII letters only. Vowels kept."""
    text = _ROMAN_RE.sub(lambda m: _ROMAN[m.group(0)], spelling_key(token))
    if _LATIN.search(token):
        for typed, folded in _ROMANIZED_NEPALI:
            text = text.replace(typed, folded)
    return re.sub(r"[^a-z]", "", text)


def romanized(token: str) -> str:
    """The word's crude romanization, vowels kept -- a script-independent measure of its length."""
    return _ascii(token)


def _reduce(text: str) -> str:
    for digraph, single in _DIGRAPHS:
        text = text.replace(digraph, single)
    text = text.translate(_SKELETON_DROP).translate(_CONSONANT_CLASS)
    return _REPEATS.sub(r"\1", text)


@lru_cache(maxsize=65536)
def skeletons(token: str) -> frozenset[str]:
    """Every consonant skeleton the word could have: one, or two for a Latin word with a g."""
    text = _ascii(token)
    if not _LATIN.search(token):
        return frozenset({_reduce(text)})
    for pattern, sound in _LATIN_SOUNDS:
        text = pattern.sub(sound, text)
    return frozenset({_reduce(text), _reduce(_SOFT_G.sub("j", text))})


def skeleton(token: str) -> str:
    """Coarse consonant skeleton: romanize, fold digraphs, drop vowels, merge confusable classes.

    Only meaningful *across* scripts -- see the module docstring for why two Devanagari words with
    one skeleton are still two words. A Latin word's hard-g reading is the one returned.
    """
    text = _ascii(token)
    if _LATIN.search(token):
        for pattern, sound in _LATIN_SOUNDS:
            text = pattern.sub(sound, text)
    return _reduce(text)


def _skeletons_close(a: str, b: str) -> bool:
    """Equal, or one consonant apart once the skeleton is long enough for that to be noise.

    At three consonants one letter is a third of the word -- ``ktn``/``ktr`` is एकदमै against
    ``actually`` -- so short skeletons must be identical.
    """
    if min(len(a), len(b)) < MIN_SKELETON:
        return False
    longest = max(len(a), len(b))
    allowed = 0 if longest <= 3 else 1 if longest <= 7 else 2
    return _levenshtein(a, b) <= allowed


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _normalized_distance(a: str, b: str) -> float:
    longest = max(len(a), len(b))
    return _levenshtein(a, b) / longest if longest else 0.0


def similarity(a: str, b: str) -> float:
    """How alike two words sound, 0-1, over their romanized forms (vowels kept)."""
    if spelling_key(a) == spelling_key(b):
        return 1.0
    left, right = _ascii(a), _ascii(b)
    if not left or not right:
        return 0.0
    return max(0.0, 1.0 - _normalized_distance(left, right))


@lru_cache(maxsize=262144)
def same_word(a: str, b: str) -> bool:
    """Whether two words are one word written two ways. See the module docstring for the rules."""
    if spelling_key(a) == spelling_key(b) or _same_number(a, b):
        return True
    sa, sb = _script(a), _script(b)
    if sa == sb or "none" in (sa, sb):
        return False
    if any(_skeletons_close(ka, kb) for ka in skeletons(a) for kb in skeletons(b)):
        return True
    left, right = _ascii(a), _ascii(b)
    if min(len(left), len(right)) < MIN_SKELETON:
        return False
    if _normalized_distance(left, right) <= CROSS_SCRIPT_THRESHOLD:
        return True
    # "cha" / छ: too short for a ratio to mean anything, so allow one character.
    return max(len(left), len(right)) <= SHORT_TOKEN and _levenshtein(left, right) <= 1


def align(ref: list[str], hyp: list[str], *, folded: bool = True) -> Alignment:
    """Minimum-edit word alignment of two token lists.

    With ``folded`` the alignment uses :func:`same_word` and allows zero-cost merges of up to
    :data:`MAX_MERGE` adjacent words on either side (see :func:`_merges`); without it, only
    identical strings match and it is a plain Levenshtein WER alignment. Ties are broken toward
    matches, then the smallest merge, then substitutions, so the result is deterministic.
    """
    m, n = len(ref), len(hyp)

    def same(a: str, b: str) -> bool:
        return same_word(a, b) if folded else a == b

    inf = m + n + 1
    cost = [[inf] * (n + 1) for _ in range(m + 1)]
    #: Each cell's step as (reference words, hypothesis words) consumed.
    back: list[list[tuple[int, int]]] = [[(0, 0)] * (n + 1) for _ in range(m + 1)]
    cost[0][0] = 0
    spans = [(a, b) for a in range(1, MAX_MERGE + 1) for b in range(1, MAX_MERGE + 1) if a + b > 2]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            options: list[tuple[int, int, tuple[int, int]]] = []
            if i and j:
                matched = same(ref[i - 1], hyp[j - 1])
                options.append((cost[i - 1][j - 1] + (0 if matched else 1), 0, (1, 1)))
            if folded:
                for a, b in spans:
                    if a <= i and b <= j and _merges(ref[i - a : i], hyp[j - b : j]):
                        options.append((cost[i - a][j - b], a + b - 2, (a, b)))
            if i:
                options.append((cost[i - 1][j] + 1, 10, (1, 0)))
            if j:
                options.append((cost[i][j - 1] + 1, 11, (0, 1)))
            best = min(options)
            cost[i][j], back[i][j] = best[0], best[2]

    ops: list[AlignOp] = []
    i, j = m, n
    while i or j:
        a, b = back[i][j]
        r, h = tuple(ref[i - a : i]), tuple(hyp[j - b : j])
        if (a, b) == (1, 1):
            if spelling_key(r[0]) == spelling_key(h[0]) or (not folded and r[0] == h[0]):
                ops.append(AlignOp("match", r, h))
            elif same(r[0], h[0]):
                ops.append(AlignOp("fold", r, h))
            else:
                ops.append(AlignOp("sub", r, h, round(similarity(r[0], h[0]), 4)))
        elif b == 0:
            ops.append(AlignOp("del", r, ()))
        elif a == 0:
            ops.append(AlignOp("ins", (), h))
        else:
            ops.append(AlignOp("merge", r, h))
        i, j = i - a, j - b
    ops.reverse()
    return Alignment(ops=ops)


def _merges(ref: list[str], hyp: list[str]) -> bool:
    """Whether adjacent words on each side are one word, or one number, written differently.

    Two against one may match by any rule, sound included (``Facebook`` / `फेस बुक`). Anything
    wider must be the same spelling or the same number once joined: by sound, three words against
    two matched unrelated phrases (`थाहा नै रहेनछ` / ``Thane नै रहेछ``).
    """
    left, right = "".join(ref), "".join(hyp)
    if len(ref) + len(hyp) == 3:
        return same_word(left, right)
    return spelling_key(left) == spelling_key(right) or _same_number(left, right)


def word_errors(
    reference: str | None,
    hypothesis: str | None,
    *,
    ruleset: Ruleset | None = None,
    folded: bool = True,
) -> Alignment:
    """Align two transcripts and count their word errors.

    Args:
        reference: The text treated as correct.
        hypothesis: The text being scored.
        ruleset: The D64 table; defaults to ``config/normalization.yaml``. Applied only when
            ``folded``: the raw rate is a plain WER over punctuation-stripped words.
        folded: Script-folded comparison (the default), or the raw lexical rate.
    """
    if folded:
        ref, hyp = fold_tokens(reference, ruleset), fold_tokens(hypothesis, ruleset)
    else:
        ref, hyp = _raw_tokens(reference), _raw_tokens(hypothesis)
    return align(ref, hyp, folded=folded)


def _raw_tokens(text: str | None) -> list[str]:
    if not text:
        return []
    text = unicodedata.normalize("NFC", text)
    return [t if _DEV.search(t) else t.lower() for t in _NON_WORD.sub(" ", text).split()]
