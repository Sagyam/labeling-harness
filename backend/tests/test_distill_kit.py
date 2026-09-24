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


def _count(refs, hyps):
    return {"clips": len(refs), "hyps": list(hyps)}


def test_by_class_scores_each_value_of_each_key():
    rows = [_row("a", overlap="none", snr="high"), _row("b", overlap=">15%", snr="high")]
    out = distill.by_class(rows, ["ra", "rb"], ["ha", "hb"], _count, keys=("overlap", "snr"))
    assert out["overlap"]["none"] == {"clips": 1, "hyps": ["ha"]}
    assert out["overlap"][">15%"] == {"clips": 1, "hyps": ["hb"]}
    assert out["snr"]["high"] == {"clips": 2, "hyps": ["ha", "hb"]}


def test_by_class_skips_clips_without_the_key():
    rows = [_row("a", overlap="none"), {"segment_id": "b", "classes": None}]
    out = distill.by_class(rows, ["ra", "rb"], ["ha", "hb"], _count, keys=("overlap",))
    assert out == {"overlap": {"none": {"clips": 1, "hyps": ["ha"]}}}


def test_by_class_defaults_to_every_key_present():
    rows = [_row("a", overlap="none"), _row("b", gender="female")]
    out = distill.by_class(rows, ["ra", "rb"], ["ha", "hb"], _count)
    assert set(out) == {"overlap", "gender"}


def test_by_class_needs_one_hypothesis_per_row():
    with pytest.raises(ValueError):
        distill.by_class([_row("a", overlap="none")], ["ra"], [], _count)
