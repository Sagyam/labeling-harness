"""A small synthetic Nepanglish corpus used by the development seed and the test fixtures.

Real audio and real transcripts come from the upstream pipeline. These sentences exist only so
development and tests have code-switched text with the shape of the target policy: English in Latin
script, Nepali in Devanagari.
"""

from __future__ import annotations

import random

#: Sentences in the target transcript policy.
SENTENCES: tuple[str, ...] = (
    "So today म Python मा loops बारे कुरा गर्छु।",
    "यो episode मा हामी machine learning को basics हेर्छौं।",
    "मलाई लाग्छ यो approach सबैभन्दा राम्रो हो।",
    "First of all, तपाईंले environment setup गर्नुपर्छ।",
    "अब हामी dataset लाई train र test मा split गर्छौं।",
    "यो function ले array लाई sort गर्छ, पछि हामी index हेर्छौं।",
    "मेरो experience अनुसार debugging मै धेरै समय जान्छ।",
    "Actually त्यो bug production मा मात्र देखिन्थ्यो।",
    "हामीले database को schema पहिले नै design गरिसक्यौं।",
    "यो model को accuracy अलिकति मात्र बढ्यो।",
    "Thanks for listening, अर्को episode मा भेटौंला।",
    "उहाँले भन्नुभयो कि startup मा patience चाहिन्छ।",
    "यो त simple concept हो तर implementation गाह्रो छ।",
    "म आफैंले त्यो script लेखेको थिएँ, तर काम गरेन।",
    "Basically, हामीले सबै logic लाई refactor गर्नुपर्यो।",
    "यो feature लाई next sprint मा ship गर्ने plan छ।",
    "तपाईंहरूले comment मा प्रश्न सोध्न सक्नुहुन्छ।",
    "त्यो conference मा धेरै interesting talks थिए।",
    "यो problem को solution एकदमै elegant छ।",
    "हामी अब break लिन्छौं, अनि फर्केर आउँछौं।",
)

#: Realistic upstream system identities. The first is the strongest, as in the real pipeline.
SYSTEMS: tuple[tuple[str, str], ...] = (
    ("qwen-ne", "sidskarki/Qwen3-ASR-Nepali"),
    ("whisper-lv3", "openai/whisper-large-v3"),
    ("indicwhisper", "ai4bharat/indicwhisper"),
    ("seamless-m4t", "facebook/seamless-m4t-v2-large"),
    ("mms-ne", "facebook/mms-1b-all"),
)

#: Perturbations a weaker system might make: transliteration slips and dropped words.
_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("गर्छु", "गर्दछु"),
    ("गर्छौं", "गर्छो"),
    ("Python", "पाइथन"),
    ("loops", "लुप्स"),
    ("machine learning", "मेसिन लर्निङ"),
    ("database", "डाटाबेस"),
    ("model", "मोडल"),
    ("script", "स्क्रिप्ट"),
    ("भेटौंला", "भेटौला"),
    ("।", ""),
)


def sentence_for(rng: random.Random) -> str:
    """Pick a reference sentence."""
    return rng.choice(SENTENCES)


def perturb(text: str, rng: random.Random, *, strength: float) -> str:
    """Return a plausibly wrong variant of ``text``.

    Args:
        text: The reference sentence.
        rng: Seeded random source, so a run is reproducible.
        strength: 0 returns the text unchanged; 1 perturbs aggressively.
    """
    if strength <= 0:
        return text
    result = text
    for source, target in _SUBSTITUTIONS:
        if source in result and rng.random() < strength:
            result = result.replace(source, target)
    words = result.split()
    if words and rng.random() < strength * 0.35:
        del words[rng.randrange(len(words))]
    if words and rng.random() < strength * 0.2:
        index = rng.randrange(len(words))
        words.insert(index, words[index])  # a repeated word, as hallucinations do
    return " ".join(words)
