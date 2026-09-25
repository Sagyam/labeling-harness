"""The distillation corpus's pure rules (D101): which files make a source, what a source is,
which sources are refused, and the voiceprint screen against gold's voices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services import distill_corpus as dc

# --- pairing audio with its info JSON -----------------------------------------------------------


def test_pair_inputs_matches_audio_with_its_info_json():
    paths = [Path("in/Show ep 1.mp3"), Path("in/Show ep 1.info.json"), Path("in/Other.m4a")]
    pairs = dc.pair_inputs(paths)
    assert pairs == [
        dc.SourceInput(audio=Path("in/Other.m4a"), info=None),
        dc.SourceInput(audio=Path("in/Show ep 1.mp3"), info=Path("in/Show ep 1.info.json")),
    ]


def test_pair_inputs_ignores_everything_that_is_not_audio():
    paths = [Path("a.info.json"), Path("notes.txt"), Path("cover.jpg"), Path("b.opus")]
    assert [p.audio.name for p in dc.pair_inputs(paths)] == ["b.opus"]


# --- what a source is ---------------------------------------------------------------------------

INFO = {
    "id": "dQw4w9WgXcQ",
    "channel": "Some Podcast",
    "channel_id": "UC123",
    "playlist_title": "Season 2",
    "title": "Episode 7",
    "duration": 3600.5,
}


def test_read_source_takes_provenance_from_the_info_json():
    s = dc.read_source(Path("x.mp3"), INFO, audio_sha256="ab" * 32)
    assert s.source_id == "yt-dQw4w9WgXcQ"
    assert (s.video_id, s.channel, s.channel_id, s.playlist, s.title) == (
        "dQw4w9WgXcQ",
        "Some Podcast",
        "UC123",
        "Season 2",
        "Episode 7",
    )
    assert s.duration == 3600.5


def test_read_source_without_a_usable_video_id_is_named_by_its_audio():
    for info in (None, {**INFO, "id": "not an id!"}):
        s = dc.read_source(Path("x.mp3"), info, audio_sha256="0123456789ab" + "0" * 52)
        assert s.source_id == "file-0123456789ab"
        assert s.video_id is None


def test_read_source_falls_back_to_uploader_and_playlist():
    info = {"id": "dQw4w9WgXcQ", "uploader": "Uploader", "playlist": "Pl"}
    s = dc.read_source(Path("x.mp3"), info, audio_sha256="a" * 64)
    assert (s.channel, s.playlist) == ("Uploader", "Pl")


# --- refusals -----------------------------------------------------------------------------------


def _source(**over):
    return dc.read_source(Path("x.mp3"), {**INFO, **over}, audio_sha256="a" * 64)


def test_a_recording_already_in_the_harness_is_refused():
    reason = dc.refusal(_source(), known_video_ids={"dQw4w9WgXcQ"}, blocked_channels=[])
    assert reason is not None and "already" in reason


def test_a_blocked_channel_is_refused_by_name_or_id_whatever_the_case():
    for blocked in (["some podcast"], ["UC123"]):
        assert dc.refusal(_source(), known_video_ids=set(), blocked_channels=blocked) is not None


def test_a_blocked_name_blocks_every_channel_that_contains_it():
    # a missed gold channel costs more than a refused clean one
    s = _source(channel="The Chill Pill Podcast Nepal", channel_id="UC999")
    assert dc.refusal(s, known_video_ids=set(), blocked_channels=["chill pill"]) is not None


def test_an_unknown_recording_from_an_open_channel_is_accepted():
    assert (
        dc.refusal(_source(), known_video_ids={"other00000a"}, blocked_channels=["Chill Pill"])
        is None
    )


# --- screening windows --------------------------------------------------------------------------


def test_window_starts_tile_the_clip_without_running_past_it():
    assert dc.window_starts(5.3, 2.0) == [0.0, 2.0]
    assert dc.window_starts(2.0, 2.0) == [0.0]
    assert dc.window_starts(1.5, 2.0) == []


def _unit(*v):
    x = np.asarray(v, dtype=float)
    return x / np.linalg.norm(x)


GOLD = {"v001": _unit(1, 0, 0), "v002": _unit(0, 1, 0)}


def test_a_source_that_sounds_like_a_gold_voice_long_enough_is_quarantined():
    windows = [(f"c{i}", 0.0, _unit(1, 0.1, 0)) for i in range(6)] + [("d", 0.0, _unit(0, 0, 1))]
    r = dc.screen_source(windows, GOLD, threshold=0.8, window_seconds=2.0, min_seconds=10.0)
    assert r.verdict == "quarantine"
    assert (r.voice, r.seconds) == ("v001", 12.0)
    assert len(r.windows) == 5 and all(w[0].startswith("c") for w in r.windows)


def test_a_few_close_windows_are_not_enough():
    windows = [("c", 0.0, _unit(1, 0.1, 0)), ("c", 2.0, _unit(1, 0.1, 0))]
    r = dc.screen_source(windows, GOLD, threshold=0.8, window_seconds=2.0, min_seconds=10.0)
    assert r.verdict == "clear"
    assert (r.voice, r.seconds) == ("v001", 4.0)


def test_seconds_are_counted_per_gold_voice_not_pooled():
    windows = [("a", 0.0, _unit(1, 0, 0))] * 3 + [("b", 0.0, _unit(0, 1, 0))] * 3
    r = dc.screen_source(windows, GOLD, threshold=0.8, window_seconds=2.0, min_seconds=10.0)
    assert r.verdict == "clear" and r.seconds == 6.0


def test_windows_too_short_to_embed_are_skipped():
    windows = [("a", 0.0, None)] * 10
    r = dc.screen_source(windows, GOLD, threshold=0.8, window_seconds=2.0, min_seconds=2.0)
    assert r.verdict == "clear" and r.seconds == 0.0


def test_the_screen_needs_gold_voices_to_compare_with():
    with pytest.raises(ValueError):
        dc.screen_source(
            [("a", 0.0, _unit(1, 0, 0))], {}, threshold=0.8, window_seconds=2.0, min_seconds=2.0
        )


# --- the clip manifest --------------------------------------------------------------------------


def test_clip_rows_name_each_clip_under_its_source():
    seg = dc.ClipSpan(segment_id="yt-dQw4w9WgXcQ_00003", start=12.5, end=20.0, checksum="c" * 64)
    rows = dc.clip_rows(_source(), [seg])
    assert rows == [
        {
            "segment_id": "yt-dQw4w9WgXcQ_00003",
            "source_id": "yt-dQw4w9WgXcQ",
            "path": "sources/yt-dQw4w9WgXcQ/clips/yt-dQw4w9WgXcQ_00003.flac",
            "start": 12.5,
            "end": 20.0,
            "duration": 7.5,
            "checksum": "c" * 64,
            "channel": "Some Podcast",
        }
    ]
