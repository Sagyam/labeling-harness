"""Unit tests for the local CTC forced aligner.

The trellis, the romanization seam and the char-to-word grouping are all exercised against
synthetic emission matrices, so the suite never needs the ONNX model file.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np

from app.services.alignment import WordSpan
from app.services.forced_align import (
    ForcedAligner,
    align_emissions,
    align_text,
    build_target,
    ctc_forced_align,
    romanize,
)

# blank, then one id per letter used below. No word delimiter: MMS-style romanized heads
# do not have one, and align_emissions only inserts a delimiter when the vocabulary carries it.
VOCAB = {"<blank>": 0, "a": 1, "b": 2, "c": 3}
VOCAB_WITH_DELIMITER = {**VOCAB, "|": 4}


def _emissions(rows: list[dict[int, float]], vocab_size: int = 4) -> np.ndarray:
    """Build a [T, V] log-probability matrix from per-frame {token_id: prob} maps."""
    out = np.full((len(rows), vocab_size), 0.001, dtype=np.float32)
    for t, row in enumerate(rows):
        for idx, prob in row.items():
            out[t, idx] = prob
    out /= out.sum(axis=1, keepdims=True)
    return np.log(out)


# --- romanization seam ------------------------------------------------------------------


def test_romanize_maps_devanagari_into_the_model_alphabet() -> None:
    roman = romanize("नेपाली")
    assert roman
    assert roman.isascii()
    assert roman.islower()
    assert all(ch.isalpha() for ch in roman)


def test_romanize_passes_latin_through_lowercased() -> None:
    assert romanize("Meeting") == "meeting"
    assert romanize("5G") == "g"


def test_romanize_returns_empty_when_nothing_survives() -> None:
    """A token with no alignable letters cannot enter the CTC target."""
    assert romanize("123") == ""
    assert romanize("!!") == ""


# --- target construction ----------------------------------------------------------------


def test_build_target_concatenates_words_and_records_their_char_ranges() -> None:
    ids, ranges = build_target(["ab", "c"], VOCAB, delimiter=None)
    assert ids == [1, 2, 3]
    assert ranges == [(0, 2), (2, 3)]


def test_build_target_inserts_a_delimiter_outside_the_word_ranges() -> None:
    ids, ranges = build_target(["ab", "c"], VOCAB_WITH_DELIMITER, delimiter="|")
    assert ids == [1, 2, 4, 3]
    # The delimiter at index 2 belongs to neither word.
    assert ranges == [(0, 2), (3, 4)]


def test_build_target_drops_characters_the_vocab_does_not_have() -> None:
    ids, ranges = build_target(["azb"], VOCAB, delimiter=None)
    assert ids == [1, 2]
    assert ranges == [(0, 2)]


# --- the trellis ------------------------------------------------------------------------


def test_ctc_forced_align_recovers_a_known_path() -> None:
    log_probs = _emissions(
        [{1: 0.98}, {1: 0.98}, {1: 0.98}, {2: 0.98}, {2: 0.98}, {2: 0.98}],
    )
    path = ctc_forced_align(log_probs, [1, 2], blank_id=0)
    # Extended sequence is [blank, a, blank, b, blank]: state 1 is "a", state 3 is "b".
    assert path.tolist() == [1, 1, 1, 3, 3, 3]


def test_ctc_forced_align_emits_every_target_at_least_once() -> None:
    """Even when the audio favours one label throughout, the path must cover both targets."""
    log_probs = _emissions([{1: 0.9}, {1: 0.9}, {1: 0.9}, {1: 0.9}])
    path = ctc_forced_align(log_probs, [1, 2], blank_id=0)
    assert 1 in path.tolist()
    assert 3 in path.tolist()


def test_ctc_forced_align_is_monotonic() -> None:
    log_probs = _emissions(
        [{1: 0.9}, {0: 0.9}, {2: 0.9}, {2: 0.9}, {0: 0.9}, {3: 0.9}],
    )
    path = ctc_forced_align(log_probs, [1, 2, 3], blank_id=0)
    assert all(b >= a for a, b in itertools.pairwise(path))


def test_ctc_forced_align_returns_empty_for_an_impossible_target() -> None:
    """Fewer frames than the target needs cannot produce a valid path."""
    log_probs = _emissions([{1: 0.9}])
    assert ctc_forced_align(log_probs, [1, 2, 3], blank_id=0).size == 0


# --- emissions to word spans ------------------------------------------------------------


def test_align_emissions_converts_frames_to_clip_relative_seconds() -> None:
    log_probs = _emissions(
        [{1: 0.98}, {1: 0.98}, {1: 0.98}, {2: 0.98}, {2: 0.98}, {2: 0.98}],
    )
    spans = align_emissions(log_probs, ["a", "b"], VOCAB, duration_seconds=0.12)
    assert spans is not None
    assert [s.word for s in spans] == ["a", "b"]
    # Six frames over 0.12 s is a 20 ms stride.
    assert math.isclose(spans[0].start, 0.0, abs_tol=1e-6)
    assert math.isclose(spans[0].end, 0.06, abs_tol=1e-6)
    assert math.isclose(spans[1].start, 0.06, abs_tol=1e-6)
    assert math.isclose(spans[1].end, 0.12, abs_tol=1e-6)


def test_align_emissions_spans_are_ordered_and_inside_the_clip() -> None:
    log_probs = _emissions(
        [{1: 0.9}, {1: 0.9}, {0: 0.9}, {2: 0.9}, {2: 0.9}, {0: 0.9}, {3: 0.9}, {3: 0.9}],
    )
    spans = align_emissions(log_probs, ["a", "b", "c"], VOCAB, duration_seconds=0.16)
    assert spans is not None
    assert len(spans) == 3
    for span in spans:
        assert 0.0 <= span.start < span.end <= 0.16
    assert all(b.start >= a.end for a, b in itertools.pairwise(spans))


def test_align_emissions_interpolates_a_token_it_cannot_align() -> None:
    """A token with no alignable characters still needs a span, borrowed from its neighbours."""
    log_probs = _emissions(
        [{1: 0.98}, {1: 0.98}, {1: 0.98}, {2: 0.98}, {2: 0.98}, {2: 0.98}],
    )
    spans = align_emissions(log_probs, ["a", "123", "b"], VOCAB, duration_seconds=0.12)
    assert spans is not None
    assert [s.word for s in spans] == ["a", "123", "b"]
    assert spans[1].start >= spans[0].end
    assert spans[1].end <= spans[2].start
    # Interpolated spans are marked less trustworthy than measured ones.
    assert spans[1].confidence is not None
    assert spans[1].confidence < 1.0


def test_align_emissions_returns_none_when_no_token_can_be_aligned() -> None:
    log_probs = _emissions([{1: 0.9}, {1: 0.9}])
    assert align_emissions(log_probs, ["123", "!!"], VOCAB, duration_seconds=0.04) is None


def test_align_emissions_returns_none_for_an_empty_token_list() -> None:
    log_probs = _emissions([{1: 0.9}])
    assert align_emissions(log_probs, [], VOCAB, duration_seconds=0.02) is None


# --- degradation ------------------------------------------------------------------------


def test_a_missing_model_degrades_to_none_rather_than_raising(tmp_path: Path) -> None:
    """Ingestion must survive an absent aligner model, exactly as it survives an absent VAD."""
    aligner = ForcedAligner(
        model_path=tmp_path / "absent.onnx",
        vocab_path=tmp_path / "absent.json",
    )
    assert aligner.available is False
    assert aligner.align(tmp_path / "nonexistent.flac", ["a"]) is None


# --- text to importer word dicts --------------------------------------------------------


class _StubAligner:
    """Stands in for a loaded ForcedAligner; align_text only needs .align()."""

    def __init__(self, spans: list[WordSpan] | None) -> None:
        self.spans = spans
        self.seen_tokens: list[str] = []

    def align(self, audio_path, tokens, sample_rate: int = 16000):
        self.seen_tokens = tokens
        return self.spans


def test_align_text_returns_word_dicts_shaped_for_the_importer(tmp_path: Path) -> None:
    stub = _StubAligner(
        [
            WordSpan(word="हामी", start=0.1, end=0.5, confidence=0.9),
            WordSpan(word="meeting", start=0.6, end=1.1, confidence=0.8),
        ]
    )
    words = align_text(stub, tmp_path / "clip.flac", "हामी meeting")
    assert words == [
        {"word": "हामी", "start": 0.1, "end": 0.5, "confidence": 0.9},
        {"word": "meeting", "start": 0.6, "end": 1.1, "confidence": 0.8},
    ]


def test_align_text_tokenizes_mixed_script_text(tmp_path: Path) -> None:
    stub = _StubAligner([])
    align_text(stub, tmp_path / "clip.flac", "आजको meeting मा, data हेर्यौं!")
    assert stub.seen_tokens == ["आजको", "meeting", "मा", "data", "हेर्यौं"]


def test_align_text_returns_none_when_the_aligner_declines(tmp_path: Path) -> None:
    assert align_text(_StubAligner(None), tmp_path / "clip.flac", "हामी") is None
    assert align_text(_StubAligner([]), tmp_path / "clip.flac", "हामी") is None


def test_align_text_returns_none_when_there_is_nothing_to_align(tmp_path: Path) -> None:
    stub = _StubAligner([WordSpan(word="x", start=0.0, end=0.1)])
    assert align_text(stub, tmp_path / "clip.flac", "") is None
    assert align_text(stub, tmp_path / "clip.flac", "   ...   ") is None
