"""The training augmenter (roadmap C, 2026-09-26): notebooks/src/augment.py.

It runs in the Colab notebooks, not in the app, but it decides what every student is trained on,
so it is tested here. Every stage is off unless its `p` says otherwise."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2] / "notebooks" / "src"
for _name in ("xtalk", "augment"):
    _spec = importlib.util.spec_from_file_location(_name, _SRC / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
import augment  # noqa: E402
import xtalk  # noqa: E402

SR = augment.SR


def _tone(seconds: float, amp: float, hz: float = 220.0) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def _peak_hz(audio: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64)))
    return float(np.argmax(spectrum) * SR / len(audio))


def _row(sid="a", split="train", pot="train", **extra):
    return {"segment_id": sid, "episode_id": "ep", "split": split, "pot": pot, **extra}


# --- speed ----------------------------------------------------------------------------------------


def test_speed_changes_length_and_pitch_together():
    clip = _tone(2.0, 8000, 440.0)
    fast = augment.change_speed(clip, 1.1)
    assert fast.dtype == np.int16
    assert len(fast) == round(len(clip) / 1.1)
    assert _peak_hz(fast) == pytest.approx(440.0 * 1.1, abs=2.0)
    slow = augment.change_speed(clip, 0.9)
    assert len(slow) == round(len(clip) / 0.9)
    assert _peak_hz(slow) == pytest.approx(440.0 * 0.9, abs=2.0)


def test_speed_of_one_is_identity():
    clip = _tone(1.0, 8000)
    np.testing.assert_array_equal(augment.change_speed(clip, 1.0), clip)


# --- gain -----------------------------------------------------------------------------------------


def test_gain_scales_and_never_clips():
    clip = _tone(1.0, 4000)
    louder = augment.apply_gain(clip, 6.0)
    assert np.abs(louder).max() == pytest.approx(np.abs(clip).max() * 10 ** (6 / 20), rel=0.01)
    loud = augment.apply_gain(_tone(1.0, 30000), 12.0)
    assert np.abs(loud.astype(np.int32)).max() <= 32767


# --- noise ----------------------------------------------------------------------------------------


def test_noise_is_added_at_the_requested_snr():
    clip = _tone(3.0, 6000)
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SR * 5) * 1000).astype(np.int16)
    out = augment.add_noise(clip, noise[: len(clip)], snr_db=10.0)
    added = out.astype(np.float64) - clip
    snr = xtalk.active_level_db(clip) - 10 * np.log10(np.mean(added**2))
    assert snr == pytest.approx(10.0, abs=0.3)


def test_noise_bank_loops_a_short_source_and_filters_by_category():
    bank = augment.NoiseBank(
        [("hum", "noise", _tone(0.5, 1000, 50.0)), ("song", "music", _tone(4.0, 1000, 300.0))]
    )
    rng = np.random.default_rng(1)
    for _ in range(10):
        name, category, audio = bank.excerpt(rng, SR * 2, categories=("noise",))
        assert (name, category, len(audio)) == ("hum", "noise", SR * 2)
    with pytest.raises(ValueError):
        bank.excerpt(rng, SR, categories=("babble",))


# --- reverb ---------------------------------------------------------------------------------------


def test_synthetic_rir_decays_60_db_per_rt60_and_keeps_the_direct_path():
    rng = np.random.default_rng(2)
    rt60 = 0.5
    h = augment.synthetic_rir(rng, rt60=rt60, drr_db=5.0)
    assert h[0] == 1.0
    t = np.arange(len(h)) / SR
    early = np.mean(h[(t > 0.02 * rt60) & (t < 0.08 * rt60)] ** 2)
    mid = np.mean(h[(t > 0.47 * rt60) & (t < 0.53 * rt60)] ** 2)
    assert 10 * np.log10(early / mid) == pytest.approx(60 * 0.45, abs=3.0)
    tail = np.sum(h[1:] ** 2)
    assert 10 * np.log10(1.0 / tail) == pytest.approx(5.0, abs=0.01)


def test_reverb_keeps_length_and_speech_level():
    clip = _tone(2.0, 6000)
    h = augment.synthetic_rir(np.random.default_rng(3), rt60=0.6, drr_db=0.0)
    out = augment.convolve(clip, h)
    assert len(out) == len(clip) and out.dtype == np.int16
    assert xtalk.active_level_db(out) == pytest.approx(xtalk.active_level_db(clip), abs=0.5)


# --- channel --------------------------------------------------------------------------------------


def test_microphone_response_cuts_outside_its_band():
    rng = np.random.default_rng(4)
    low, mid, high = _tone(2.0, 6000, 60.0), _tone(2.0, 6000, 1000.0), _tone(2.0, 6000, 7500.0)
    clip = (low.astype(np.int32) + mid + high).clip(-32768, 32767).astype(np.int16)
    cfg = augment.ChannelConfig(
        p=1.0, low_cut_hz=(200.0, 200.0), high_cut_hz=(4000.0, 4000.0), peaks=0, drive=None
    )
    out, info = augment.channel(clip, rng, cfg)
    spectrum = np.abs(np.fft.rfft(out.astype(np.float64)))
    at = lambda hz: spectrum[round(hz * len(out) / SR)]  # noqa: E731
    assert at(60.0) < at(1000.0) / 10 and at(7500.0) < at(1000.0) / 10
    assert info["low_cut_hz"] == 200.0 and info["high_cut_hz"] == 4000.0


# --- codec ----------------------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("name", ["mp3", "opus", "aac", "mulaw"])
def test_codec_round_trip_keeps_length_and_alignment(name):
    rng = np.random.default_rng(5)
    clip = (np.convolve(rng.standard_normal(SR * 2), np.ones(8) / 8, "same") * 6000).astype(
        np.int16
    )
    out = augment.codec_round_trip(clip, name, kbps=32)
    assert len(out) == len(clip) and out.dtype == np.int16
    corr = np.corrcoef(out.astype(np.float64), clip.astype(np.float64))[0, 1]
    assert corr > 0.7, f"{name}: the decoded audio is misaligned or ruined (r={corr:.2f})"


# --- spec -----------------------------------------------------------------------------------------


def test_spec_augment_masks_within_its_widths_and_keeps_shape():
    feats = np.ones((80, 500), dtype=np.float32)
    cfg = augment.SpecConfig(p=1.0, freq_masks=2, freq_width=10, time_masks=5, time_width=0.02)
    out, info = augment.spec_augment(feats, np.random.default_rng(6), cfg)
    assert out.shape == feats.shape and feats.min() == 1.0  # the input is not modified
    masked_rows = np.flatnonzero((out == 0).all(axis=1))
    masked_cols = np.flatnonzero((out == 0).all(axis=0))
    assert 0 < len(masked_rows) <= 2 * 10
    assert 0 < len(masked_cols) <= 5 * 10  # 2% of 500 frames
    assert info["freq"] and info["time"]


def test_spec_augment_off_returns_the_input():
    feats = np.ones((80, 100), dtype=np.float32)
    out, info = augment.spec_augment(feats, np.random.default_rng(7), augment.SpecConfig())
    assert out is feats and info is None


# --- the chain ------------------------------------------------------------------------------------


def test_everything_off_by_default_returns_the_clip_untouched():
    aug = augment.Augmenter(augment.AugmentConfig())
    clip = _tone(1.0, 4000)
    out, info = aug(_row(), clip, np.random.default_rng(8))
    assert out is clip and info == {"segment_id": "a", "stages": []}


def test_augmenter_refuses_held_out_audio():
    aug = augment.Augmenter(augment.AugmentConfig(gain=augment.GainConfig(p=1.0)))
    for row in (_row(split="val"), _row(split="test", pot="gold"), _row(pot="gold")):
        with pytest.raises(ValueError):
            aug(row, _tone(1.0, 4000), np.random.default_rng(9))


def test_augmenter_needs_the_resources_its_stages_use():
    with pytest.raises(ValueError):
        augment.Augmenter(augment.AugmentConfig(noise=augment.NoiseConfig(p=0.5)))
    with pytest.raises(ValueError):
        augment.Augmenter(augment.AugmentConfig(crosstalk=xtalk.CrosstalkConfig(p=0.5)))


def test_chain_runs_in_order_and_is_reproducible():
    bank = augment.NoiseBank(
        [
            (
                "hiss",
                "noise",
                (np.random.default_rng(0).standard_normal(SR * 3) * 500).astype(np.int16),
            )
        ]
    )
    cfg = augment.AugmentConfig(
        speed=augment.SpeedConfig(p=1.0, factors=(1.1,)),
        reverb=augment.ReverbConfig(p=1.0),
        channel=augment.ChannelConfig(p=1.0),
        noise=augment.NoiseConfig(p=1.0, snr_db=(10.0, 20.0)),
        gain=augment.GainConfig(p=1.0),
    )
    aug = augment.Augmenter(cfg, noise=bank)
    clip = _tone(2.0, 6000)
    a, info = aug(_row(), clip, np.random.default_rng(10))
    b, _ = aug(_row(), clip, np.random.default_rng(10))
    np.testing.assert_array_equal(a, b)
    assert [s["stage"] for s in info["stages"]] == ["speed", "reverb", "channel", "noise", "gain"]
    assert len(a) == round(len(clip) / 1.1)


def test_crosstalk_donors_get_their_own_room_and_microphone():
    episode = np.concatenate([_tone(10.0, 3000), _tone(10.0, 2000, 500.0)])
    rows = [
        {
            **_row("a"),
            "start_time": 0.0,
            "end_time": 10.0,
            "overlap_spans": [],
            "speaker_turns": [{"start": 0, "end": 10, "voice": "v1", "speaker": "S0"}],
        },
        {
            **_row("b"),
            "start_time": 10.0,
            "end_time": 20.0,
            "overlap_spans": [],
            "speaker_turns": [{"start": 0, "end": 10, "voice": "v2", "speaker": "S1"}],
        },
    ]
    cfg = augment.AugmentConfig(
        crosstalk=xtalk.CrosstalkConfig(
            p=1.0, seconds=(2.0, 3.0), share=(0.2, 0.3), overshoot=None
        ),
        donor_reverb=augment.ReverbConfig(p=1.0, rt60=(0.4, 0.4), drr_db=(-3.0, -3.0)),
        donor_channel=augment.ChannelConfig(p=1.0),
    )
    aug = augment.Augmenter(
        cfg,
        donors=xtalk.DonorPool(rows),
        fetch=lambda ep, s, e: episode[round(s * SR) : round(e * SR)],
    )
    _, info = aug(rows[0], episode[: 10 * SR], np.random.default_rng(11))
    (stage,) = info["stages"]
    assert stage["stage"] == "crosstalk"
    for w in stage["windows"]:
        assert [f["stage"] for f in w["fx"]["stages"]] == ["reverb", "channel"]
        assert w["fx"]["stages"][0]["rt60"] == 0.4


def test_config_from_dict_names_every_stage_and_refuses_typos():
    cfg = augment.AugmentConfig.from_dict(
        {
            "spec": {"p": 1.0},
            "speed": {"p": 0.3, "factors": [0.9, 1.1]},
            "crosstalk": {"p": 0.5, "seconds": [2, 8], "gap_db": [-6, 0], "donor": "clip"},
            "donor_reverb": {"p": 0.5},
        }
    )
    assert cfg.speed.factors == (0.9, 1.1)
    assert cfg.crosstalk.seconds == (2, 8) and cfg.crosstalk.donor == "clip"
    assert cfg.noise.p == 0.0
    with pytest.raises(ValueError):
        augment.AugmentConfig.from_dict({"sped": {"p": 1.0}})
    with pytest.raises(TypeError):
        augment.AugmentConfig.from_dict({"speed": {"probability": 1.0}})
