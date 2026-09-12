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

Three rules, in the order they are tried on a pair of words:

1. **Spelling.** Both words are reduced to a spelling key -- the D64 table applied, NFC, Latin
   lowercased, and the Devanagari distinctions Nepali writers do not hold consistently folded
   away: vowel length (ि/ी, ु/ू), chandrabindu against anusvara, nukta, a word-final virama, and
   the script of the digits. Equal keys are the same word.
2. **Across scripts, by sound.** When one word is Latin and the other Devanagari (or a Latin stem
   carrying a Devanagari suffix, ``phoneमा``), both are reduced to a coarse consonant skeleton and
   compared. This is the rule that makes `एक्टिभ` and ``active`` one word.
3. **Never by sound within one script.** `गर्नु` and `गर्ने` share a skeleton and are different
   words -- an infinitive and a participle. Matching Devanagari against Devanagari on sound would
   erase exactly the grammar a WER is supposed to see, so rule 2 is only ever applied across
   scripts.

Alignment also accepts a zero-cost merge of two adjacent words on one side against one word on the
other, which is how a spacing difference (`गर्नुभयो` / `गर्नु भयो`, ``Facebook`` / `फेस बुक`) stops
costing two errors.

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
FOLD_VERSION = "fold-v1"

#: Romanized forms (vowels kept) are accepted as one word at or below this normalized distance.
#: The EDA notebook swept 0.25-0.6; this is the conservative end that still catches inflection.
CROSS_SCRIPT_THRESHOLD = 0.34
#: A one-consonant skeleton collides with too much of the language to be evidence of anything.
MIN_SKELETON = 2
#: Below this romanized length, compare whole romanized strings rather than skeletons.
SHORT_TOKEN = 4

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
    """Split text into comparable words: D64 table applied, punctuation and danda removed.

    Words keep their original spelling and case; everything else about comparison happens in
    :func:`spelling_key` and :func:`same_word`. A Latin stem carrying a Devanagari suffix
    (``phoneमा``) stays one word, as it was written.
    """
    if not text:
        return []
    text = unicodedata.normalize("NFC", text).translate(_JOINERS)
    text = normalize_text(text, ruleset or _default_ruleset()) or ""
    return _NON_WORD.sub(" ", text).split()


@lru_cache(maxsize=65536)
def spelling_key(token: str) -> str:
    """The word with every spelling distinction this corpus does not hold consistently removed."""
    text = unicodedata.normalize("NFC", token).translate(_JOINERS).translate(_DEV_DIGITS)
    text = unicodedata.normalize("NFD", text).translate(_DEV_FOLD)
    text = unicodedata.normalize("NFC", text).lower().replace("'", "")
    return text.removesuffix(_VIRAMA)


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
    if spelling_key(a) == spelling_key(b):
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

    With ``folded`` the alignment uses :func:`same_word` and allows a zero-cost two-to-one merge
    in either direction; without it, only identical strings match and it is a plain Levenshtein
    WER alignment. Ties are broken toward matches, then merges, then substitutions, so the
    result is deterministic.
    """
    m, n = len(ref), len(hyp)

    def same(a: str, b: str) -> bool:
        return same_word(a, b) if folded else a == b

    inf = m + n + 1
    cost = [[inf] * (n + 1) for _ in range(m + 1)]
    back: list[list[str]] = [[""] * (n + 1) for _ in range(m + 1)]
    cost[0][0] = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            options: list[tuple[int, int, str]] = []
            if i and j:
                matched = same(ref[i - 1], hyp[j - 1])
                options.append((cost[i - 1][j - 1] + (0 if matched else 1), 0, "diag"))
            if folded and i >= 2 and j >= 1 and same(ref[i - 2] + ref[i - 1], hyp[j - 1]):
                options.append((cost[i - 2][j - 1], 1, "merge_ref"))
            if folded and i >= 1 and j >= 2 and same(ref[i - 1], hyp[j - 2] + hyp[j - 1]):
                options.append((cost[i - 1][j - 2], 1, "merge_hyp"))
            if i:
                options.append((cost[i - 1][j] + 1, 2, "del"))
            if j:
                options.append((cost[i][j - 1] + 1, 3, "ins"))
            best = min(options)
            cost[i][j], back[i][j] = best[0], best[2]

    ops: list[AlignOp] = []
    i, j = m, n
    while i or j:
        step = back[i][j]
        if step == "diag":
            r, h = ref[i - 1], hyp[j - 1]
            if spelling_key(r) == spelling_key(h) or (not folded and r == h):
                ops.append(AlignOp("match", (r,), (h,)))
            elif same(r, h):
                ops.append(AlignOp("fold", (r,), (h,)))
            else:
                ops.append(AlignOp("sub", (r,), (h,), round(similarity(r, h), 4)))
            i, j = i - 1, j - 1
        elif step == "merge_ref":
            ops.append(AlignOp("merge", (ref[i - 2], ref[i - 1]), (hyp[j - 1],)))
            i, j = i - 2, j - 1
        elif step == "merge_hyp":
            ops.append(AlignOp("merge", (ref[i - 1],), (hyp[j - 2], hyp[j - 1])))
            i, j = i - 1, j - 2
        elif step == "del":
            ops.append(AlignOp("del", (ref[i - 1],), ()))
            i -= 1
        else:
            ops.append(AlignOp("ins", (), (hyp[j - 1],)))
            j -= 1
    ops.reverse()
    return Alignment(ops=ops)


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
