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
