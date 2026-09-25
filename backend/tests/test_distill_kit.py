"""The distillation notebook's pure helpers (roadmap §B): notebooks/src/distill.py.

It runs in Colab, not in the app, but it decides what text shapes the students' tokenizer and
what their scores are paired against, so it is tested here."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "src" / "distill.py"
_spec = importlib.util.spec_from_file_location("distill", _PATH)
distill = importlib.util.module_from_spec(_spec)
sys.modules["distill"] = distill
_spec.loader.exec_module(distill)


def _row(sid: str, text: str = "x", **classes: str) -> dict:
    return {"segment_id": sid, "episode_id": f"ep-{sid}", "text": text, "classes": classes}


def _splits() -> dict[str, list[dict]]:
    return {
        "train": [_row("t1", "एक one"), _row("t2", "दुई two")],
        "val": [_row("v1", "val text")],
        "gold": [_row("g1", "gold text")],
    }


# --- tokenizer corpus ----------------------------------------------------------------------------


def test_tokenizer_texts_are_train_labels_only():
    assert distill.tokenizer_texts(_splits()) == ["एक one", "दुई two"]


def test_tokenizer_texts_refuse_a_held_out_clip_in_train():
    splits = _splits()
    splits["train"].append(_row("g1", "gold text"))
    with pytest.raises(ValueError, match="g1"):
        distill.tokenizer_texts(splits)


# --- reference transcripts -----------------------------------------------------------------------


def test_reference_texts_follow_row_order():
    rows = [_row("a"), _row("b")]
    assert distill.reference_texts(rows, {"b": "B", "a": "A"}) == ["A", "B"]


def test_reference_texts_name_the_missing_clips():
    with pytest.raises(ValueError, match="b"):
        distill.reference_texts([_row("a"), _row("b")], {"a": "A"})


# --- scores per clip class -----------------------------------------------------------------------
#
# by_class groups per-clip counts, aligned once by the caller, and hands each group to `summarize`:
# re-aligning every clip once per class key was the notebook's slowest step.


def _summarize(clips):
    return {"clips": len(clips), "items": list(clips)}


def test_by_class_summarizes_each_value_of_each_key():
    rows = [_row("a", overlap="none", snr="high"), _row("b", overlap=">15%", snr="high")]
    out = distill.by_class(rows, ["ca", "cb"], _summarize, keys=("overlap", "snr"))
    assert out["overlap"]["none"] == {"clips": 1, "items": ["ca"]}
    assert out["overlap"][">15%"] == {"clips": 1, "items": ["cb"]}
    assert out["snr"]["high"] == {"clips": 2, "items": ["ca", "cb"]}


def test_by_class_skips_clips_without_the_key():
    rows = [_row("a", overlap="none"), {"segment_id": "b", "classes": None}]
    out = distill.by_class(rows, ["ca", "cb"], _summarize, keys=("overlap",))
    assert out == {"overlap": {"none": {"clips": 1, "items": ["ca"]}}}


def test_by_class_defaults_to_every_key_present():
    rows = [_row("a", overlap="none"), _row("b", gender="female")]
    out = distill.by_class(rows, ["ca", "cb"], _summarize)
    assert set(out) == {"overlap", "gender"}


def test_by_class_needs_one_count_per_row():
    with pytest.raises(ValueError):
        distill.by_class([_row("a", overlap="none")], [], _summarize)


def test_by_class_never_realigns_a_clip():
    # each clip's counts reach `summarize` as given, once per key it carries, never recomputed
    rows = [_row(s, overlap="none", snr="high", cmi="low") for s in "abc"]
    seen = []
    distill.by_class(rows, ["ca", "cb", "cc"], lambda clips: seen.extend(clips))
    assert sorted(seen) == sorted(["ca", "cb", "cc"] * 3)


# --- step 3: filtering the teacher's labels ------------------------------------------------------


def _pseudo(sid, *, text="कुरा गर्नुभयो", tokens=10, duration=5.0, logprob=-0.1, looped=False):
    return {
        "segment_id": sid,
        "text": text,
        "n_tokens": tokens,
        "duration": duration,
        "mean_logprob": logprob,
        "looped": looped,
    }


def test_filter_drops_looped_empty_and_out_of_rate_clips_before_confidence():
    rows = [
        _pseudo("loop", looped=True),
        _pseudo("empty", text="  ", tokens=0),
        _pseudo("fast", tokens=100, duration=5.0),  # 20 tokens/s
        _pseudo("slow", tokens=1, duration=10.0),  # 0.1 tokens/s
        _pseudo("ok", tokens=10, duration=5.0),
    ]
    kept, report = distill.filter_pseudo(rows, tokens_per_s=(0.5, 13.0), drop_fraction=0.0)
    assert [r["segment_id"] for r in kept] == ["ok"]
    assert report["dropped"] == {"loop": 1, "empty": 1, "rate": 2, "confidence": 0}


def test_filter_drops_the_least_confident_fraction_of_what_is_left():
    rows = [_pseudo(f"c{i}", logprob=-i / 10) for i in range(10)]  # c9 is the least confident
    kept, report = distill.filter_pseudo(rows, tokens_per_s=(0.5, 13.0), drop_fraction=0.2)
    assert {r["segment_id"] for r in kept} == {f"c{i}" for i in range(8)}
    assert report["dropped"]["confidence"] == 2
    assert report["logprob_cut"] == -0.8


def test_filter_reports_what_it_kept_in_clips_and_hours():
    rows = [_pseudo(f"c{i}", duration=3600.0, tokens=7200) for i in range(3)]
    _, report = distill.filter_pseudo(rows, tokens_per_s=(0.5, 13.0), drop_fraction=0.0)
    assert (report["kept"], report["kept_hours"], report["total"]) == (3, 3.0, 3)


def test_filter_refuses_a_fraction_outside_zero_to_one():
    with pytest.raises(ValueError):
        distill.filter_pseudo([], tokens_per_s=(0.5, 13.0), drop_fraction=1.0)


def test_latin_share_counts_words_written_in_latin_script():
    assert distill.latin_share("म phone किन्छु, camera राम्रो") == 0.4
    assert distill.latin_share("सबै नेपाली") == 0.0
    assert distill.latin_share("") == 0.0


def test_mixing_bucket_matches_the_corpus_cmi_classes():
    assert [distill.mixing_bucket(x) for x in (0.0, 0.1, 0.2, 0.5)] == ["0", "<15", "15-30", "30+"]
