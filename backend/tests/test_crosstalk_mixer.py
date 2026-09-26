"""Synthetic crosstalk for fine-tuning (roadmap item 3): notebooks/src/xtalk.py.

The mixer runs in the Colab notebook, not in the app, but it decides what the model is trained on,
so it is tested here. Its distributions are the measured ones in docs/findings.md, *Overlap
windows: how long they are and who is louder* (2026-09-15)."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "src" / "xtalk.py"
_spec = importlib.util.spec_from_file_location("xtalk", _PATH)
xtalk = importlib.util.module_from_spec(_spec)
sys.modules["xtalk"] = xtalk
_spec.loader.exec_module(xtalk)

SR = xtalk.SR


def _tone(seconds: float, amp: float, hz: float = 220.0) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def _row(sid, episode, start, end, turns, overlap=None, split="train", pot="train"):
    return {
        "segment_id": sid,
        "episode_id": episode,
        "start_time": start,
        "end_time": end,
        "split": split,
        "pot": pot,
        "overlap_spans": [] if overlap is None else overlap,
        "speaker_turns": turns,
    }


def _turn(start, end, voice, speaker="SPEAKER_00"):
    return {"start": start, "end": end, "voice": voice, "speaker": speaker}


# --- distributions --------------------------------------------------------------------------------


def test_window_durations_follow_the_measured_deciles():
    rng = np.random.default_rng(0)
    d = np.array([xtalk.sample_window_s(rng) for _ in range(20_000)])
    assert d.min() >= 0.05 and d.max() <= 8.94
    assert np.median(d) == pytest.approx(0.42, abs=0.03)
    assert np.quantile(d, 0.9) == pytest.approx(1.33, abs=0.08)
    assert (d < 1.0).mean() == pytest.approx(0.83, abs=0.03)


def test_level_gaps_are_near_equal_and_either_voice_can_be_louder():
    rng = np.random.default_rng(1)
    g = np.array([xtalk.sample_gap_db(rng) for _ in range(20_000)])
    assert np.abs(g).max() <= 3.0
    assert np.median(np.abs(g)) == pytest.approx(1.59, abs=0.15)
    assert (g > 0).mean() == pytest.approx(0.5, abs=0.02)


def test_share_targets_weight_the_error_buckets():
    rng = np.random.default_rng(2)
    s = np.array([xtalk.sample_share(rng) for _ in range(20_000)])
    assert s.min() >= 0.01 and s.max() <= 0.40
    assert (s > 0.15).mean() == pytest.approx(0.3, abs=0.02)
    assert ((s >= 0.05) & (s <= 0.15)).mean() == pytest.approx(0.3, abs=0.02)


def test_active_level_ignores_silence():
    tone = _tone(1.0, 3000)
    padded = np.concatenate([np.zeros(SR * 3, np.int16), tone, np.zeros(SR * 3, np.int16)])
    assert xtalk.active_level_db(padded) == pytest.approx(xtalk.active_level_db(tone), abs=0.2)
    assert xtalk.active_level_db(np.zeros(SR, np.int16)) == -np.inf


# --- donors ---------------------------------------------------------------------------------------


def test_solo_stretches_drop_other_voices_and_overlap_and_are_episode_absolute():
    row = _row(
        "a",
        "ep",
        100.0,
        110.0,
        [_turn(0.0, 6.0, "v1"), _turn(5.0, 10.0, "v2", "SPEAKER_01")],
        overlap=[[5.0, 6.0]],
    )
    got = xtalk.solo_stretches(row, min_s=0.3)
    assert got == [(100.0, 105.0, "v1"), (106.0, 110.0, "v2")]


def test_solo_stretches_key_unlinked_speakers_by_episode():
    row = _row("a", "ep", 0.0, 4.0, [_turn(0.0, 4.0, None, "SPEAKER_03")])
    assert xtalk.solo_stretches(row, min_s=0.3) == [(0.0, 4.0, "ep:SPEAKER_03")]


def test_pool_refuses_anything_but_train():
    with pytest.raises(ValueError):
        xtalk.DonorPool([_row("g", "ep", 0.0, 4.0, [_turn(0, 4, "v1")], split="test", pot="gold")])
    with pytest.raises(ValueError):
        xtalk.DonorPool([_row("v", "ep", 0.0, 4.0, [_turn(0, 4, "v1")], split="val")])


def test_pool_prefers_another_voice_from_the_same_episode():
    rows = [
        _row("a", "ep1", 0.0, 5.0, [_turn(0, 5, "v1")]),
        _row("b", "ep1", 10.0, 15.0, [_turn(0, 5, "v2")]),
        _row("c", "ep2", 0.0, 5.0, [_turn(0, 5, "v3")]),
    ]
    pool = xtalk.DonorPool(rows)
    rng = np.random.default_rng(3)
    for _ in range(50):
        d = pool.pick(rng, "ep1", {"v1"}, 1.0)
        assert (d.episode, d.voice, d.same_episode) == ("ep1", "v2", True)
        assert d.start >= 10.0 and d.start + 1.0 <= 15.0 + 1e-9


def test_pool_falls_back_to_another_episode_and_never_to_a_target_voice():
    rows = [
        _row("a", "ep1", 0.0, 5.0, [_turn(0, 5, "v1")]),
        _row("c", "ep2", 0.0, 5.0, [_turn(0, 5, "v3")]),
        _row("d", "ep3", 0.0, 5.0, [_turn(0, 5, "v1")]),  # the same person in another show
    ]
    pool = xtalk.DonorPool(rows)
    rng = np.random.default_rng(4)
    for _ in range(50):
        d = pool.pick(rng, "ep1", {"v1"}, 1.0)
        assert (d.episode, d.voice, d.same_episode) == ("ep2", "v3", False)


def test_pool_returns_none_when_no_stretch_is_long_enough():
    rows = [
        _row("a", "ep1", 0.0, 5.0, [_turn(0, 5, "v1")]),
        _row("b", "ep1", 9.0, 10.0, [_turn(0, 1, "v2")]),
    ]
    pool = xtalk.DonorPool(rows)
    assert pool.pick(np.random.default_rng(5), "ep1", {"v1"}, 2.0) is None


# --- placement and mixing -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_windows_fit_the_clip_do_not_touch_and_reach_the_share(seed):
    rng = np.random.default_rng(seed)
    clip_s, share = 12.0, 0.2
    wins = xtalk.plan_windows(rng, clip_s, share)
    assert wins
    ordered = sorted(wins)
    for (s1, d1), (s2, _) in itertools.pairwise(ordered):
        assert s1 + d1 < s2
    assert all(s >= 0 and s + d <= clip_s + 1e-9 for s, d in wins)
    total = sum(d for _, d in wins)
    assert total <= xtalk.MAX_SHARE * clip_s + 1e-9


def test_windows_reach_the_share_on_average():
    rng = np.random.default_rng(6)
    shares = [sum(d for _, d in xtalk.plan_windows(rng, 10.0, 0.2)) / 10.0 for _ in range(500)]
    assert np.mean(shares) >= 0.2


def test_windows_do_not_overshoot_a_light_share():
    """A median burst is 8% of a 5 s clip; a 2% target must not end up in the 5-15% bucket."""
    rng = np.random.default_rng(9)
    shares = [sum(d for _, d in xtalk.plan_windows(rng, 5.0, 0.02)) / 5.0 for _ in range(500)]
    assert max(shares) <= 0.02 * xtalk.OVERSHOOT + 1e-9


def test_mix_leaves_audio_outside_windows_untouched_and_sets_the_level_gap():
    target = _tone(4.0, 4000, 220.0)
    piece = _tone(0.5, 500, 700.0)
    out = xtalk.mix(target, [(SR, piece, -2.0)])
    assert out.dtype == np.int16 and len(out) == len(target)
    np.testing.assert_array_equal(out[:SR], target[:SR])
    np.testing.assert_array_equal(out[SR + len(piece) :], target[SR + len(piece) :])
    added = out[SR : SR + len(piece)].astype(np.float64) - target[SR : SR + len(piece)]
    inner = added[len(added) // 4 : -len(added) // 4]  # clear of the fades and the int16 rounding
    gap = xtalk.active_level_db(inner) - xtalk.active_level_db(target)
    assert gap == pytest.approx(-2.0, abs=0.3)


def test_mix_truncates_a_window_at_the_clip_end_and_never_clips():
    target = _tone(1.0, 30000)
    out = xtalk.mix(target, [(SR - 800, _tone(1.0, 30000, 300.0), 3.0)])
    assert len(out) == len(target)
    assert np.abs(out.astype(np.int32)).max() <= 32767


def test_mixer_only_touches_measured_clean_clips():
    episode = np.concatenate([_tone(10.0, 3000), _tone(10.0, 2000, 500.0)])
    rows = [
        _row("a", "ep", 0.0, 10.0, [_turn(0, 10, "v1")]),
        _row("b", "ep", 10.0, 20.0, [_turn(0, 10, "v2")]),
    ]
    mixer = xtalk.Mixer(
        xtalk.DonorPool(rows),
        lambda ep, s, e: episode[round(s * SR) : round(e * SR)],
        p=1.0,
    )
    rng = np.random.default_rng(7)
    clip = episode[: 10 * SR]

    out, info = mixer(rows[0], clip, rng)
    assert info is not None and info["windows"] and info["same_episode"] == len(info["windows"])
    assert not np.array_equal(out, clip)

    for skip in (
        {**rows[0], "overlap_spans": [[1.0, 2.0]]},  # already has real crosstalk
        {**rows[0], "overlap_spans": None},  # never measured
        {**rows[0], "speaker_turns": None},  # never diarized
    ):
        same, none = mixer(skip, clip, rng)
        assert none is None and same is clip

    off = xtalk.Mixer(mixer.pool, mixer.fetch, p=0.0)
    same, none = off(rows[0], clip, rng)
    assert none is None and same is clip


def test_mixer_is_reproducible_for_a_seed():
    episode = np.concatenate([_tone(10.0, 3000), _tone(10.0, 2000, 500.0)])
    rows = [
        _row("a", "ep", 0.0, 10.0, [_turn(0, 10, "v1")]),
        _row("b", "ep", 10.0, 20.0, [_turn(0, 10, "v2")]),
    ]
    mixer = xtalk.Mixer(
        xtalk.DonorPool(rows), lambda ep, s, e: episode[round(s * SR) : round(e * SR)], p=1.0
    )
    a, _ = mixer(rows[0], episode[: 10 * SR], np.random.default_rng(8))
    b, _ = mixer(rows[0], episode[: 10 * SR], np.random.default_rng(8))
    np.testing.assert_array_equal(a, b)


# --- configurable crosstalk (roadmap C, 2026-09-26) -----------------------------------------------


def _two_voice_episode():
    episode = np.concatenate([_tone(20.0, 3000), _tone(20.0, 2000, 500.0)])
    rows = [
        _row("a", "ep", 0.0, 20.0, [_turn(0, 20, "v1")]),
        _row("b", "ep", 20.0, 40.0, [_turn(0, 20, "v2")]),
    ]
    return episode, rows, lambda ep, s, e: episode[round(s * SR) : round(e * SR)]


def test_default_config_is_the_measured_d96_mixer():
    cfg = xtalk.CrosstalkConfig()
    assert (cfg.seconds, cfg.gap_db, cfg.share, cfg.donor) == (None, None, None, "stretch")
    assert (cfg.max_share, cfg.overshoot, cfg.clean_only) == (
        xtalk.MAX_SHARE,
        xtalk.OVERSHOOT,
        True,
    )


def test_config_rejects_nonsense():
    for bad in (
        {"seconds": (5.0, 2.0)},
        {"seconds": (0.0, 2.0)},
        {"gap_db": (3.0, -3.0)},
        {"share": (0.5, 0.2)},
        {"max_share": 1.5},
        {"donor": "tts"},
        {"p": 1.5},
    ):
        with pytest.raises(ValueError):
            xtalk.CrosstalkConfig(**bad)


@pytest.mark.parametrize("seed", range(10))
def test_long_windows_at_a_chosen_level_stay_in_their_ranges(seed):
    _, rows, fetch = _two_voice_episode()
    cfg = xtalk.CrosstalkConfig(
        p=1.0,
        seconds=(3.0, 6.0),
        gap_db=(-6.0, -3.0),
        share=(0.3, 0.5),
        max_share=0.8,
        overshoot=None,
    )
    mixer = xtalk.Mixer(xtalk.DonorPool(rows), fetch, config=cfg)
    _, info = mixer(rows[0], fetch("ep", 0.0, 20.0), np.random.default_rng(seed))
    assert info is not None and info["windows"]
    for w in info["windows"]:
        assert 3.0 <= w["seconds"] <= 6.0
        assert -6.0 <= w["gap_db"] <= -3.0
    assert info["share"] <= 0.8 + 1e-9


def test_p_argument_overrides_the_config():
    _, rows, fetch = _two_voice_episode()
    mixer = xtalk.Mixer(xtalk.DonorPool(rows), fetch, p=0.0, config=xtalk.CrosstalkConfig(p=1.0))
    clip = fetch("ep", 0.0, 20.0)
    same, none = mixer(rows[0], clip, np.random.default_rng(1))
    assert none is None and same is clip


def test_clip_pool_refuses_anything_but_train():
    with pytest.raises(ValueError):
        xtalk.ClipDonorPool(
            [_row("g", "ep", 0.0, 4.0, [_turn(0, 4, "v1")], split="test", pot="gold")]
        )


def test_clip_pool_picks_whole_clips_in_range_by_another_voice_same_episode_first():
    rows = [
        _row("a", "ep1", 0.0, 10.0, [_turn(0, 10, "v1")]),
        _row("b", "ep1", 10.0, 14.0, [_turn(0, 4, "v2")]),
        _row("c", "ep1", 20.0, 29.0, [_turn(0, 9, "v2")]),  # too long for the range
        _row("d", "ep1", 30.0, 34.0, [_turn(0, 4, "v1")]),  # the target's own voice
        _row("e", "ep2", 0.0, 4.0, [_turn(0, 4, "v3")]),
    ]
    pool = xtalk.ClipDonorPool(rows)
    rng = np.random.default_rng(2)
    for _ in range(30):
        d = pool.pick(rng, rows[0], (3.0, 5.0))
        assert (d.segment_id, d.start, d.end, d.same_episode) == ("b", 10.0, 14.0, True)
    assert pool.pick(rng, rows[0], (20.0, 30.0)) is None


def test_clip_pool_never_uses_the_same_episode_for_an_undiarized_target():
    rows = [
        _row("a", "ep1", 0.0, 10.0, None),
        _row("b", "ep1", 10.0, 14.0, None),
        _row("e", "ep2", 0.0, 4.0, None),
    ]
    pool = xtalk.ClipDonorPool(rows)
    rng = np.random.default_rng(3)
    for _ in range(30):
        assert pool.pick(rng, rows[0], (1.0, 5.0)).segment_id == "e"


def test_clip_donors_are_laid_whole_and_named_for_their_labels():
    episode = np.concatenate([_tone(12.0, 3000), _tone(4.0, 2000, 500.0)])
    rows = [
        _row("a", "ep", 0.0, 12.0, [_turn(0, 12, "v1")]),
        _row("b", "ep", 12.0, 16.0, [_turn(0, 4, "v2")]),
    ]
    fetch = lambda ep, s, e: episode[round(s * SR) : round(e * SR)]  # noqa: E731
    cfg = xtalk.CrosstalkConfig(
        p=1.0, donor="clip", seconds=(2.0, 5.0), share=(0.3, 0.4), max_share=0.9, overshoot=None
    )
    mixer = xtalk.Mixer(xtalk.ClipDonorPool(rows), fetch, config=cfg)
    out, info = mixer(rows[0], episode[: 12 * SR], np.random.default_rng(4))
    assert info is not None and len(info["windows"]) == 1
    w = info["windows"][0]
    assert w["donor_segment_id"] == "b" and w["seconds"] == pytest.approx(4.0)
    assert 0.0 <= w["offset"] <= 8.0 + 1e-9
    assert len(out) == 12 * SR


def test_clean_only_off_mixes_an_undiarized_clip_from_another_episode():
    episode = np.concatenate([_tone(10.0, 3000), _tone(10.0, 2000, 500.0)])
    rows = [
        _row("a", "ep1", 0.0, 10.0, None, overlap=None),
        _row("b", "ep2", 10.0, 20.0, [_turn(0, 10, "v2")]),
    ]
    rows[0]["overlap_spans"] = None
    fetch = lambda ep, s, e: episode[round(s * SR) : round(e * SR)]  # noqa: E731
    on = xtalk.Mixer(xtalk.DonorPool(rows[1:]), fetch, config=xtalk.CrosstalkConfig(p=1.0))
    assert on(rows[0], episode[: 10 * SR], np.random.default_rng(5))[1] is None
    off = xtalk.Mixer(
        xtalk.DonorPool(rows[1:]), fetch, config=xtalk.CrosstalkConfig(p=1.0, clean_only=False)
    )
    _, info = off(rows[0], episode[: 10 * SR], np.random.default_rng(5))
    assert info is not None and info["same_episode"] == 0


def test_donor_fx_processes_each_second_voice_before_the_level_is_set():
    _, rows, fetch = _two_voice_episode()
    seen = []

    def fx(audio, rng):
        seen.append(len(audio))
        return np.zeros_like(audio), {"muted": True}

    cfg = xtalk.CrosstalkConfig(p=1.0, seconds=(2.0, 3.0), share=(0.2, 0.3), overshoot=None)
    mixer = xtalk.Mixer(xtalk.DonorPool(rows), fetch, config=cfg, donor_fx=fx)
    clip = fetch("ep", 0.0, 20.0)
    out, info = mixer(rows[0], clip, np.random.default_rng(6))
    assert seen and all(w["fx"] == {"muted": True} for w in info["windows"])
    np.testing.assert_array_equal(out, clip)  # a silent donor adds nothing


# --- "everything said" labels (D100, 2026-09-26) --------------------------------------------------


def _worded(sid, episode, start, end, text, words, voice):
    row = _row(sid, episode, start, end, [_turn(0, end - start, voice)])
    row["text"] = text
    row["label_words"] = [{"word": w, "start": s, "end": e} for w, s, e in words]
    return row


def test_everything_needs_whole_clip_donors():
    with pytest.raises(ValueError):
        xtalk.CrosstalkConfig(label="everything")  # a cut stretch has no transcript
    with pytest.raises(ValueError):
        xtalk.CrosstalkConfig(label="both", donor="clip")
    assert xtalk.CrosstalkConfig(label="everything", donor="clip").label == "everything"


def test_text_token_pattern_is_the_exports():
    from app.services.normalize import WORD_TOKEN_RE

    assert xtalk.TEXT_TOKEN_RE.pattern == WORD_TOKEN_RE.pattern


def test_merge_interleaves_by_start_time_and_keeps_punctuation():
    target = {
        "text": "मलाई best लाग्यो। Turkish cuisine,",
        "label_words": [
            {"word": "मलाई", "start": 0.0, "end": 0.4},
            {"word": "best", "start": 0.5, "end": 0.8},
            {"word": "लाग्यो", "start": 0.9, "end": 1.3},
            {"word": "Turkish", "start": 2.0, "end": 2.4},
            {"word": "cuisine", "start": 2.5, "end": 3.0},
        ],
    }
    donor = {
        "text": "हो, OK!",
        "label_words": [
            {"word": "हो", "start": 0.0, "end": 0.3},
            {"word": "OK", "start": 0.4, "end": 0.7},
        ],
    }
    assert xtalk.merge_labels(target, [(0.45, donor)]) == (
        "मलाई हो, best OK! लाग्यो। Turkish cuisine,"
    )


def test_merge_refuses_words_that_do_not_spell_the_text():
    bad = {"text": "मलाई best", "label_words": [{"word": "मलाई", "start": 0.0, "end": 0.4}]}
    ok = {"text": "हो", "label_words": [{"word": "हो", "start": 0.0, "end": 0.3}]}
    assert xtalk.merge_labels(bad, [(0.0, ok)]) is None
    assert xtalk.merge_labels(ok, [(0.0, bad)]) is None
    assert xtalk.merge_labels({"text": "हो", "label_words": None}, [(0.0, ok)]) is None


def _everything_mixer(rows, episode, **cfg):
    fetch = lambda ep, s, e: episode[round(s * SR) : round(e * SR)]  # noqa: E731
    config = xtalk.CrosstalkConfig(
        p=1.0,
        donor="clip",
        label="everything",
        seconds=(2.0, 5.0),
        share=(0.2, 0.4),
        max_share=0.9,
        overshoot=None,
        **cfg,
    )
    return xtalk.Mixer(xtalk.ClipDonorPool(rows), fetch, config=config)


def test_everything_label_holds_both_voices_in_time_order():
    episode = np.concatenate([_tone(10.0, 3000), _tone(3.0, 2000, 500.0)])
    target = _worded(
        "a",
        "ep",
        0.0,
        10.0,
        "एक दुई तीन चार।",
        [("एक", 0.5, 1.0), ("दुई", 3.0, 3.5), ("तीन", 6.0, 6.5), ("चार", 9.0, 9.5)],
        "v1",
    )
    donor = _worded("b", "ep", 10.0, 13.0, "yes no", [("yes", 0.2, 0.6), ("no", 2.0, 2.4)], "v2")
    mixer = _everything_mixer([target, donor], episode)
    _, info = mixer(target, episode[: 10 * SR], np.random.default_rng(1))
    assert info is not None and info["windows"][0]["donor_segment_id"] == "b"
    offset = info["windows"][0]["offset"]
    timed = sorted(
        [(s, w) for w, s in (("एक", 0.5), ("दुई", 3.0), ("तीन", 6.0), ("चार।", 9.0))]
        + [(offset + 0.2, "yes"), (offset + 2.0, "no")],
        key=lambda t: t[0],
    )
    assert info["text"] == " ".join(w for _, w in timed)


def test_everything_skips_a_target_without_words_and_donors_without_words():
    episode = np.concatenate([_tone(10.0, 3000), _tone(3.0, 2000, 500.0)])
    target = _worded("a", "ep", 0.0, 10.0, "एक", [("एक", 0.5, 1.0)], "v1")
    wordless = {**_worded("b", "ep", 10.0, 13.0, "yes", [], "v2"), "label_words": None}
    mixer = _everything_mixer([target, wordless], episode)
    clip = episode[: 10 * SR]
    same, none = mixer(target, clip, np.random.default_rng(2))
    assert none is None and same is clip  # the only donor has no words

    donor = _worded("c", "ep", 10.0, 13.0, "yes", [("yes", 0.2, 0.6)], "v2")
    mixer = _everything_mixer([{**target, "label_words": None}, donor], episode)
    same, none = mixer({**target, "label_words": None}, clip, np.random.default_rng(3))
    assert none is None and same is clip


def test_target_label_mode_never_rewrites_the_text():
    episode = np.concatenate([_tone(10.0, 3000), _tone(3.0, 2000, 500.0)])
    target = _worded("a", "ep", 0.0, 10.0, "एक", [("एक", 0.5, 1.0)], "v1")
    donor = _worded("b", "ep", 10.0, 13.0, "yes", [("yes", 0.2, 0.6)], "v2")
    fetch = lambda ep, s, e: episode[round(s * SR) : round(e * SR)]  # noqa: E731
    cfg = xtalk.CrosstalkConfig(
        p=1.0, donor="clip", seconds=(2.0, 5.0), share=(0.2, 0.4), max_share=0.9, overshoot=None
    )
    _, info = xtalk.Mixer(xtalk.ClipDonorPool([target, donor]), fetch, config=cfg)(
        target, episode[: 10 * SR], np.random.default_rng(4)
    )
    assert info is not None and "text" not in info
