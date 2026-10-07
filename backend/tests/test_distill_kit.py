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
    assert report["dropped"] == {"overlap": 0, "loop": 1, "empty": 1, "rate": 2, "confidence": 0}


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


def test_filter_drops_a_clip_overlapped_beyond_the_threshold_first():
    rows = [
        {**_pseudo("heavy", looped=True), "overlap_share": 0.4},  # overlap is judged before loops
        {**_pseudo("at", logprob=-0.2), "overlap_share": 0.15},  # at the threshold: kept
        {**_pseudo("clean"), "overlap_share": 0.0},
        _pseudo("unmeasured"),  # never measured is not the same as overlapped
    ]
    kept, report = distill.filter_pseudo(
        rows, tokens_per_s=(0.5, 13.0), drop_fraction=0.0, max_overlap_share=0.15
    )
    assert [r["segment_id"] for r in kept] == ["at", "clean", "unmeasured"]
    assert report["dropped"]["overlap"] == 1 and report["dropped"]["loop"] == 0
    assert (report["max_overlap_share"], report["overlap_unmeasured"]) == (0.15, 1)


def test_filter_without_a_threshold_drops_nothing_for_overlap():
    rows = [{**_pseudo("heavy"), "overlap_share": 0.9}]
    kept, report = distill.filter_pseudo(rows, tokens_per_s=(0.5, 13.0), drop_fraction=0.0)
    assert len(kept) == 1 and report["dropped"]["overlap"] == 0
    assert report["max_overlap_share"] is None


def test_filter_refuses_an_overlap_threshold_outside_zero_to_one():
    with pytest.raises(ValueError):
        distill.filter_pseudo(
            [], tokens_per_s=(0.5, 13.0), drop_fraction=0.0, max_overlap_share=1.5
        )


def test_latin_share_counts_words_written_in_latin_script():
    assert distill.latin_share("म phone किन्छु, camera राम्रो") == 0.4
    assert distill.latin_share("सबै नेपाली") == 0.0
    assert distill.latin_share("") == 0.0


def test_mixing_bucket_matches_the_corpus_cmi_classes():
    assert [distill.mixing_bucket(x) for x in (0.0, 0.1, 0.2, 0.5)] == ["0", "<15", "15-30", "30+"]


# --- PreDistill: sources from the owner's zip (D101) ---------------------------------------------


def test_a_source_is_named_by_its_file_and_its_channel_by_the_name_before_the_number():
    assert distill.source_from_filename("Nepali_Podcast_07.mp3") == (
        "Nepali_Podcast_07",
        "Nepali_Podcast",
    )
    assert distill.source_from_filename("sub/dir/talk-show_12.MP3") == ("talk-show_12", "talk-show")
    assert distill.source_from_filename("nonumber.mp3") == ("nonumber", "nonumber")


def test_a_source_id_keeps_only_characters_safe_in_a_path():
    # a run of unsafe characters becomes one "_"; Devanagari is kept
    assert distill.source_from_filename("My Show: ep 1_03.mp3") == (
        "My_Show_ep_1_03",
        "My_Show_ep_1",
    )
    assert distill.source_from_filename("नेपाली कुरा_02.mp3") == ("नेपाली_कुरा_02", "नेपाली_कुरा")


def test_a_channel_is_blocked_when_its_name_contains_a_blocked_name():
    assert distill.channel_blocked("The_Chill_Pill_Show", ["chill pill"])
    assert not distill.channel_blocked("Nepali_Podcast", ["chill pill", "prime television"])


def test_slice_rows_name_each_clip_under_its_recording():
    rows = distill.slice_rows("pod_01", "pod", [(0.5, 12.25), (13.0, 20.0)])
    assert rows == [
        {
            "segment_id": "pod_01_00000",
            "episode_id": "pod_01",
            "source_id": "pod_01",
            "channel": "pod",
            "start_time": 0.5,
            "end_time": 12.25,
            "duration": 11.75,
        },
        {
            "segment_id": "pod_01_00001",
            "episode_id": "pod_01",
            "source_id": "pod_01",
            "channel": "pod",
            "start_time": 13.0,
            "end_time": 20.0,
            "duration": 7.0,
        },
    ]


# --- overlapped speech on the unlabelled corpus (D105) -------------------------------------------


def test_the_overlap_helpers_are_the_harness_own():
    """distill.py repeats three pure functions of the harness, which Colab cannot import."""
    from app.services import clip_classes, overlap

    spans = [(1.0, 2.5), (9.0, 12.0), (30.0, 31.0)]
    for start, end in ((0.0, 5.0), (2.0, 10.0), (12.0, 20.0), (9.5, 9.75)):
        mine = distill.spans_within(spans, start, end)
        assert mine == overlap.spans_within(spans, start, end)
        assert distill.overlap_share(mine, end - start) == clip_classes.overlap_share(
            mine, end - start
        )
    assert distill.overlap_share(None, 5.0) is clip_classes.overlap_share(None, 5.0) is None
    assert distill.overlap_share([(0.0, 9.0)], 0.0) == clip_classes.overlap_share([(0.0, 9.0)], 0.0)
    for share in (None, 0.0, 0.01, 0.0499, 0.05, 0.1, 0.15, 0.1501, 0.6, 1.0):
        assert distill.overlap_bucket(share) == clip_classes.overlap_bucket(share)


def _clip(sid: str, start: float, end: float, channel: str = "Show", **extra) -> dict:
    return {"segment_id": sid, "channel": channel, "start_time": start, "end_time": end, **extra}


def test_with_overlap_gives_each_clip_its_spans_and_share():
    rows = [_clip("a", 0.0, 10.0), _clip("b", 10.0, 20.0), _clip("c", 20.0, 30.0)]
    out = distill.with_overlap(rows, [(8.0, 12.0), (25.0, 26.0)])
    assert [r["overlap_spans"] for r in out] == [[[8.0, 10.0]], [[0.0, 2.0]], [[5.0, 6.0]]]
    assert [r["overlap_share"] for r in out] == [0.2, 0.2, 0.1]
    assert [r["segment_id"] for r in out] == ["a", "b", "c"] and "overlap_share" not in rows[0]


def test_a_clean_clip_is_measured_clean_and_an_unmeasured_one_is_none():
    clean = distill.with_overlap([_clip("a", 0.0, 10.0)], [])[0]
    assert (clean["overlap_spans"], clean["overlap_share"]) == ([], 0.0)
    unmeasured = distill.with_overlap([_clip("a", 0.0, 10.0)], None)[0]
    assert (unmeasured["overlap_spans"], unmeasured["overlap_share"]) == (None, None)


def test_overlap_loss_says_what_each_threshold_costs_per_channel():
    hour = 3600.0
    rows = [
        _clip("a", 0.0, hour, "Round_Table", overlap_share=0.3),
        _clip("b", 0.0, hour, "Round_Table", overlap_share=0.1),
        _clip("c", 0.0, hour, "Solo", overlap_share=0.0),
        _clip("d", 0.0, hour, "Solo", overlap_share=None),
    ]
    loss = distill.overlap_loss(rows, (0.05, 0.15))
    assert list(loss) == ["Round_Table", "Solo", "all"]
    assert loss["Round_Table"]["dropped_hours"] == {0.05: 2.0, 0.15: 1.0}
    assert loss["Solo"] == {
        "hours": 2.0,
        "unmeasured_hours": 1.0,
        "dropped_hours": {0.05: 0.0, 0.15: 0.0},
    }
    assert (loss["all"]["hours"], loss["all"]["dropped_hours"][0.15]) == (4.0, 1.0)


def test_overlap_loss_drops_what_the_filter_drops():
    rows = [
        {**_pseudo(f"c{i}", duration=3600.0, tokens=7200), **_clip(f"c{i}", 0.0, 3600.0)}
        for i in range(4)
    ]
    for r, share in zip(rows, (0.0, 0.15, 0.16, 0.5), strict=True):
        r["overlap_share"] = share
    kept, _ = distill.filter_pseudo(
        rows, tokens_per_s=(0.5, 13.0), drop_fraction=0.0, max_overlap_share=0.15
    )
    assert distill.overlap_loss(rows, (0.15,))["all"]["dropped_hours"][0.15] == 4 - len(kept)


# --- stage 2's mixture (D106) --------------------------------------------------------------------


def test_the_human_labels_take_their_share_and_channels_split_the_rest_by_root_hours():
    groups = [None, None, "A", "B", "B"]
    hours = [1.0, 3.0, 4.0, 8.0, 8.0]  # human 4 h; A 4 h (root 2); B 16 h (root 4)
    w = distill.mixture_weights(groups, hours, human_share=0.4)
    assert sum(w) == pytest.approx(1.0)
    assert w[0] + w[1] == pytest.approx(0.4)
    assert w[2] == pytest.approx(0.6 * 2 / 6)
    assert w[3] + w[4] == pytest.approx(0.6 * 4 / 6)


def test_inside_a_group_a_clip_is_drawn_by_its_length():
    w = distill.mixture_weights([None, None, "A", "A"], [1.0, 3.0, 2.0, 6.0], human_share=0.5)
    assert w[1] == pytest.approx(3 * w[0]) and w[3] == pytest.approx(3 * w[2])


def test_one_kind_of_label_alone_takes_every_draw():
    assert sum(distill.mixture_weights([None, None], [1.0, 1.0], 0.5)) == pytest.approx(1.0)
    only_pseudo = distill.mixture_weights(["A", "B"], [1.0, 4.0], 0.5)
    assert only_pseudo == pytest.approx([1 / 3, 2 / 3])


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1])
def test_the_human_share_is_strictly_between_zero_and_one(share):
    with pytest.raises(ValueError):
        distill.mixture_weights([None, "A"], [1.0, 1.0], share)


def test_the_mixture_needs_one_length_per_clip_and_positive_lengths():
    with pytest.raises(ValueError):
        distill.mixture_weights([None, "A"], [1.0], 0.5)
    with pytest.raises(ValueError):
        distill.mixture_weights([None, "A"], [1.0, 0.0], 0.5)


def test_an_epoch_is_drawn_by_weight_and_is_repeatable_from_its_seed():
    weights = [0.0, 0.75, 0.25]
    draw = distill.draw_epoch(weights, 4000, seed=3)
    assert draw == distill.draw_epoch(weights, 4000, seed=3)
    assert draw != distill.draw_epoch(weights, 4000, seed=4)
    assert 0 not in draw and draw.count(1) / 4000 == pytest.approx(0.75, abs=0.03)


def test_smoke_reads_its_own_predecessor_when_there_is_one() -> None:
    present = {"teacher-smoke.json", "teacher.json"}
    chosen = distill.smoke_source(True, "teacher-smoke.json", "teacher.json", present.__contains__)
    assert chosen == "teacher-smoke.json"


def test_smoke_falls_back_to_the_real_predecessor() -> None:
    # 03e and 05 ran for real without smoke runs first, so no smoke output exists to read.
    present = {"teacher.json"}
    chosen = distill.smoke_source(True, "teacher-smoke.json", "teacher.json", present.__contains__)
    assert chosen == "teacher.json"


def test_a_real_run_never_reads_a_smoke_output() -> None:
    def exists(path: str) -> bool:
        raise AssertionError("a real run does not look for a smoke output")

    chosen = distill.smoke_source(False, "teacher-smoke.json", "teacher.json", exists)
    assert chosen == "teacher.json"


def test_a_smoke_run_writes_under_its_own_name() -> None:
    """09's smoke run reads the real Flex folder when there is no smoke teacher, so it must not
    write report.md there: it would replace the real report with smoke numbers."""
    assert distill.smoke_name(True, "report") == "report-smoke"
    assert distill.smoke_name(False, "report") == "report"
