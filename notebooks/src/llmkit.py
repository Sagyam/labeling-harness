"""Pure helpers for the speech-LLM students, 07a and 07b (D122).

An LLM decoder answers in free text and its training targets sit after a prompt whose length
varies with the clip, so a speech-LLM student needs a few things the ASR students do not: the
answer cleaned the way the bake-off scored it, a cap on how long it may run, where each answer
and its labels go in a padded batch, a micro-batch sized without running out of memory, and
Omnilingual LLM-ASR's loss turned into the summed loss the training loop expects. Pure Python;
importable without torch, on Python 3.10+ (07b runs it in fairseq2's environment).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_QUOTES = {'"': '"', "'": "'", "`": "`", "\u201c": "\u201d", "\u2018": "\u2019", "\u00ab": "\u00bb"}


def clean(text: str) -> str:
    """The answer as scored: whitespace (newlines included) collapsed to single spaces, and one
    pair of quotes or backticks around the whole answer removed. Nothing else: a preamble such as
    "Here is the transcription:" stays, and is charged as insertions."""
    text = " ".join(text.split())
    if len(text) >= 2 and text[0] in _QUOTES and text[-1] == _QUOTES[text[0]]:
        inner = text[1:-1]
        if text[0] not in inner and _QUOTES[text[0]] not in inner:
            text = inner.strip()
    return text


def token_cap(
    tokens_per_s: float, seconds: float, slack: float = 1.5, extra: int = 8, ceiling: int = 1024
) -> int:
    """The most tokens an answer may run to: `slack` times the densest training label's rate (in
    the student's own tokens) over the clip, plus `extra`, at most `ceiling`. A runaway is cut
    there rather than left to fill the context."""
    return min(ceiling, math.ceil(slack * tokens_per_s * seconds) + extra)


def fit_items(peak_one: float, peak_two: float, limit: float, ceiling: int = 256) -> int:
    """The most clips a micro-batch holds within `limit` bytes, from the measured peaks of one and
    of two clips: each clip adds `peak_two - peak_one`, so n fit while
    `peak_one + (n - 1) * step <= limit`. 0 when one clip does not fit; `ceiling` when a second
    clip measured no growth. The probe then never runs a batch it expects to run out of memory,
    because a run of OOMs left memory that nothing in Python held (2026-10-07)."""
    if peak_one > limit:
        return 0
    step = peak_two - peak_one
    if step <= 0:
        return ceiling
    return min(ceiling, 1 + math.floor((limit - peak_one) / step))


def answer_spans(
    prefix_lens: Sequence[int], answer_lens: Sequence[int]
) -> tuple[int, list[tuple[int, int]]]:
    """Where each clip's answer goes in a right-padded batch: right after its own prompt, which is
    as long as its audio makes it. Returns the batch width and each answer's [start, end)."""
    if len(prefix_lens) != len(answer_lens):
        raise ValueError("one answer per prompt")
    if any(n <= 0 for n in answer_lens):
        raise ValueError("an answer carries at least its closing token")
    spans = [(p, p + a) for p, a in zip(prefix_lens, answer_lens, strict=True)]
    return max(end for _, end in spans), spans


def omni_loss_scale(target_lens: Sequence[int]) -> tuple[float, int]:
    """Omnilingual LLM-ASR's forward returns its summed cross-entropy divided by the batch's target
    tokens and multiplied by its clips (`Wav2Vec2LlamaModel.compute_loss`), each target counted
    with the end-of-sequence token it gains. Returns the factor that turns that back into the sum,
    and the token count the training loop divides by."""
    tokens = sum(n + 1 for n in target_lens)
    return tokens / len(target_lens), tokens
