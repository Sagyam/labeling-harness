"""The speech-LLM students' pure helpers (07a, 07b): notebooks/src/llmkit.py.

They run in Colab, not in the app, but where the labels sit decides what a student is trained on,
and the loss conversion decides how its steps are weighted, so both are tested here."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "notebooks" / "src"
_spec = importlib.util.spec_from_file_location("llmkit", _SRC / "llmkit.py")
llmkit = importlib.util.module_from_spec(_spec)
sys.modules["llmkit"] = llmkit
_spec.loader.exec_module(llmkit)


# --- cleaning ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  यो\nphone  राम्रो\tछ ", "यो phone राम्रो छ"),  # yo phone raamro chha
        ('"यो phone राम्रो छ"', "यो phone राम्रो छ"),
        ("“यो phone”", "यो phone"),
        ("`यो`", "यो"),
        ('"one" and "two"', '"one" and "two"'),  # quotes that do not wrap the whole answer stay
        ("", ""),
    ],
)
def test_clean_only_collapses_whitespace_and_strips_wrapping_quotes(raw: str, clean: str) -> None:
    assert llmkit.clean(raw) == clean


def test_clean_keeps_a_preamble_the_insertions_have_to_see() -> None:
    text = "Here is the transcription: यो phone राम्रो छ"
    assert llmkit.clean(text) == text


# --- generation length ---------------------------------------------------------------------------


def test_the_token_cap_scales_with_the_clip_and_has_a_ceiling() -> None:
    assert llmkit.token_cap(10.0, 4.0) == 68  # ceil(1.5 * 10 * 4) + 8
    assert llmkit.token_cap(10.0, 400.0, ceiling=600) == 600


# --- the micro-batch -----------------------------------------------------------------------------


def test_the_micro_batch_is_extrapolated_from_two_measured_peaks() -> None:
    """The probe never runs a batch it expects to run out of memory: a run of OOMs in the
    2026-10-07 bake-off left ~46 GiB allocated that nothing in Python held."""
    gib = 2**30
    # 30 GiB with one clip, 32 with two: 2 GiB a clip, so 30 + 2 (n - 1) <= 60 -> n = 16
    assert llmkit.fit_items(30 * gib, 32 * gib, 60 * gib) == 16
    assert llmkit.fit_items(30 * gib, 32 * gib, 61.9 * gib) == 16  # whole clips only


def test_the_extrapolated_micro_batch_has_a_floor_and_a_ceiling() -> None:
    gib = 2**30
    assert llmkit.fit_items(70 * gib, 72 * gib, 60 * gib) == 0  # not even one clip
    # a second clip that measured no growth
    assert llmkit.fit_items(30 * gib, 30 * gib, 60 * gib, ceiling=256) == 256
    assert llmkit.fit_items(30 * gib, 30.001 * gib, 60 * gib, ceiling=64) == 64


# --- where the answer goes -----------------------------------------------------------------------


def test_each_answer_follows_its_own_prompt() -> None:
    """Clips of different lengths bring prompts of different lengths (their audio tokens): each
    answer starts where its own prompt ends, not at the batch's longest prompt."""
    width, spans = llmkit.answer_spans(prefix_lens=[5, 8], answer_lens=[3, 2])
    assert spans == [(5, 8), (8, 10)]
    assert width == 10


def test_the_first_labelled_position_is_the_shortest_prompt() -> None:
    """The loss keeps logits only from the first labelled position of the batch on."""
    _, spans = llmkit.answer_spans(prefix_lens=[7, 4, 9], answer_lens=[1, 6, 2])
    assert min(start for start, _ in spans) == 4


def test_an_answer_must_not_be_empty() -> None:
    """Every answer carries at least its closing token, or the clip trains on nothing."""
    with pytest.raises(ValueError):
        llmkit.answer_spans(prefix_lens=[5], answer_lens=[0])
    with pytest.raises(ValueError):
        llmkit.answer_spans(prefix_lens=[5, 6], answer_lens=[1])


# --- Omnilingual LLM-ASR's loss ------------------------------------------------------------------


def test_omnilingual_loss_is_turned_into_a_sum_over_its_tokens() -> None:
    """The model returns the per-token mean times the number of clips; the training loop wants a
    sum and the number of units it sums over. Each target gains its end-of-sequence token."""
    scale, tokens = llmkit.omni_loss_scale(target_lens=[3, 5])
    assert tokens == (3 + 1) + (5 + 1)
    # mean 0.5 a token, times 2 clips, is what the model returns; times the scale it is the sum
    assert (0.5 * 2) * scale == pytest.approx(0.5 * tokens)
