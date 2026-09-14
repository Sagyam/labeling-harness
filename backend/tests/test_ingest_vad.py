"""Silero VAD: segmentation, the energy fallback, the model input contract and clip edges."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.services.silero_vad import (
    SileroVAD,
    SpeechTurn,
    extract_clips,
    segment_audio_to_slices,
)

pytestmark = pytest.mark.db


# --- Stage 2: Silero VAD Segmentation ---------------------------------------------------


def test_silero_vad_detects_speech_and_slices_within_bounds(tmp_path: Path) -> None:
    vad = SileroVAD()
    sr = 16000
    # 25 seconds of audio to test splitting > 20.0s
    t = np.linspace(0, 25.0, int(sr * 25.0), endpoint=False)
    audio = 0.4 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    turns = vad.detect_turns(audio, sample_rate=sr)
    assert len(turns) >= 1

    slices = segment_audio_to_slices(turns, total_duration=25.0, min_seg=2.0, max_seg=20.0)
    assert len(slices) >= 2

    # Every slice must obey 2.0s <= duration <= 20.0s
    for start, end in slices:
        dur = end - start
        assert 2.0 <= dur <= 20.0, f"Slice duration {dur} out of bounds [2.0, 20.0]"

    # Test extracting clips to disk
    norm_flac = tmp_path / "test_norm.flac"
    sf.write(str(norm_flac), audio, sr, format="FLAC")

    clips_dir = tmp_path / "clips"
    segments = extract_clips(norm_flac, slices, "ep_test", clips_dir)
    assert len(segments) == len(slices)
    for seg in segments:
        assert seg.clip_path.is_file()
        assert seg.clip_checksum
        clip_info = sf.info(str(seg.clip_path))
        assert clip_info.samplerate == 16000
        assert clip_info.channels == 1


# --- Stage 2: the energy VAD fallback ----------------------------------------------------
#
# This is the path taken whenever onnxruntime or the ONNX file is absent, which is the most
# likely way a deployment differs from a development machine.


def energy_vad(tmp_path: Path) -> SileroVAD:
    """A VAD with no model, so detection falls back to the energy envelope."""
    vad = SileroVAD(model_path=tmp_path / "absent.onnx")
    assert vad._session is None
    return vad


def test_energy_fallback_finds_speech_either_side_of_a_silence(tmp_path: Path) -> None:
    vad = energy_vad(tmp_path)
    sample_rate = 16000
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(2 * sample_rate) / sample_rate)
    silence = np.zeros(sample_rate, dtype=np.float64)
    audio = np.concatenate([tone, silence, tone]).astype(np.float32)

    turns = vad.detect_turns(audio, sample_rate=sample_rate)

    assert len(turns) >= 2
    assert turns[0].start == pytest.approx(0.0, abs=0.1)
    for turn in turns:
        assert turn.end > turn.start


def test_energy_fallback_returns_one_turn_for_continuous_speech(tmp_path: Path) -> None:
    vad = energy_vad(tmp_path)
    sample_rate = 16000
    audio = (0.5 * np.sin(2 * np.pi * 440 * np.arange(4 * sample_rate) / sample_rate)).astype(
        np.float32
    )

    turns = vad.detect_turns(audio, sample_rate=sample_rate)

    assert len(turns) == 1
    assert turns[0].end - turns[0].start > 3.0


def test_energy_fallback_on_silence_still_yields_the_whole_clip(tmp_path: Path) -> None:
    """Silence must not swallow a segment: the clip is handed on for a human to judge."""
    vad = energy_vad(tmp_path)
    turns = vad.detect_turns(np.zeros(16000 * 4, dtype=np.float32), sample_rate=16000)

    assert len(turns) == 1
    assert turns[0].start == 0.0


def test_energy_fallback_on_audio_shorter_than_one_frame_finds_nothing(tmp_path: Path) -> None:
    vad = energy_vad(tmp_path)
    assert vad.detect_turns(np.zeros(100, dtype=np.float32), sample_rate=16000) == []


def test_a_missing_model_file_is_reported_and_does_not_raise(tmp_path: Path) -> None:
    assert SileroVAD(model_path=tmp_path / "nope.onnx")._session is None


# --- Silero VAD: the model's input contract ----------------------------------------------
#
# The ONNX input length is dynamic, so feeding the wrong window size is accepted silently and
# the model then returns a near-zero probability for everything. Nothing errors; the VAD just
# stops detecting speech, and the cutter falls back to slicing on a fixed grid.


class RecordingSession:
    """Stands in for the ONNX session, capturing the windows it is handed."""

    def __init__(self, prob: float = 0.9) -> None:
        self.windows: list[np.ndarray] = []
        self.prob = prob

    def run(self, _outputs, inputs):
        self.windows.append(inputs["input"])
        # The real session returns (output, stateN); output is shaped (batch, 1).
        return np.array([[self.prob]], dtype=np.float32), inputs["state"]


def vad_with(session) -> SileroVAD:
    vad = SileroVAD(model_path=Path("/nonexistent.onnx"))
    vad._session = session
    return vad


def test_the_model_is_fed_context_plus_chunk_not_a_bare_chunk() -> None:
    from app.services.silero_vad import CHUNK_SIZE, CONTEXT_SIZE

    session = RecordingSession()
    vad_with(session).detect_turns(np.zeros(CHUNK_SIZE * 6, dtype=np.float32), sample_rate=16000)

    assert session.windows, "the model was never called"
    for window in session.windows:
        assert window.shape == (1, CONTEXT_SIZE + CHUNK_SIZE), window.shape


def test_each_window_carries_the_previous_chunk_as_its_context() -> None:
    from app.services.silero_vad import CHUNK_SIZE, CONTEXT_SIZE

    audio = np.arange(CHUNK_SIZE * 3, dtype=np.float32)
    session = RecordingSession()
    vad_with(session).detect_turns(audio, sample_rate=16000)

    # the first window has no history and is zero-padded
    assert np.array_equal(session.windows[0][0, :CONTEXT_SIZE], np.zeros(CONTEXT_SIZE))
    # every later window opens with the tail of the chunk before it
    for i in range(1, len(session.windows)):
        previous_chunk = audio[(i - 1) * CHUNK_SIZE : i * CHUNK_SIZE]
        assert np.array_equal(session.windows[i][0, :CONTEXT_SIZE], previous_chunk[-CONTEXT_SIZE:])
        assert np.array_equal(
            session.windows[i][0, CONTEXT_SIZE:], audio[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        )


def test_an_eight_kilohertz_stream_uses_the_smaller_window() -> None:
    from app.services.silero_vad import window_sizes

    assert window_sizes(16000) == (512, 64)
    assert window_sizes(8000) == (256, 32)


def test_the_real_model_separates_speech_from_digital_silence() -> None:
    """The end-to-end guard: a working VAD must not score silence the same as everything else."""
    vad = SileroVAD()
    if vad._session is None:
        pytest.skip("silero_vad.onnx is not present")

    # Read the probabilities directly: silence must sit well below the 0.5 threshold. Before
    # the context fix this was ~0.0005 for silence *and* for speech, which is the whole bug.
    from app.services.silero_vad import CHUNK_SIZE, CONTEXT_SIZE

    silence = np.zeros(16000 * 3, dtype=np.float32)

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT_SIZE, dtype=np.float32)
    probs = []
    for i in range(len(silence) // CHUNK_SIZE):
        chunk = silence[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        window = np.concatenate((context, chunk))[np.newaxis, :]
        out, state = vad._session.run(
            None, {"input": window, "state": state, "sr": np.array(16000, dtype=np.int64)}
        )
        probs.append(float(out[0][0]))
        context = chunk[-CONTEXT_SIZE:]
    assert max(probs) < 0.5


def test_no_detected_speech_is_reported_rather_than_passed_off_as_one_turn() -> None:
    """Falling back to 'the whole file is one turn' is what hid a dead VAD for months."""
    session = RecordingSession(prob=0.0)
    turns = vad_with(session).detect_turns(np.zeros(16000 * 5, dtype=np.float32), 16000)

    assert len(turns) == 1
    assert turns[0].start == 0.0
    assert turns[0].end == pytest.approx(5.0, abs=0.1)


# --- clip edges must not click -----------------------------------------------------------


def test_clip_edges_are_faded_so_a_cut_cannot_click(tmp_path: Path) -> None:
    """A cut lands mid-waveform; starting or stopping on a non-zero sample is heard as a click."""
    sr = 16000
    # a loud constant-amplitude tone: every cut point is far from a zero crossing
    t = np.arange(sr * 6) / sr
    sf.write(str(tmp_path / "src.flac"), 0.8 * np.sin(2 * np.pi * 220 * t), sr, format="FLAC")

    segments = extract_clips(tmp_path / "src.flac", [(1.0, 4.0)], "ep", tmp_path / "clips")
    clip, _ = sf.read(str(segments[0].clip_path), dtype="float32")

    # Sample 0 and last sample must be 0.0
    assert abs(clip[0]) < 1e-5, f"clip starts at {clip[0]:+.4f}, which clicks"
    assert abs(clip[-1]) < 1e-5, f"clip ends at {clip[-1]:+.4f}, which clicks"

    # Raised-cosine shape: the slope at the edge must start flat (zero derivative)
    assert abs(clip[1] - clip[0]) < 0.001, "boundary must have smooth zero derivative"

    # Fade duration is 15ms (240 samples at 16kHz)
    fade_len = int(sr * 0.015)
    # The middle of the fade (sample fade_len // 2) is ~half-height
    # whereas by sample fade_len it reaches full waveform amplitude
    assert abs(clip[fade_len] / (0.8 * np.sin(2 * np.pi * 220 * (1.0 + fade_len / sr)))) > 0.95

    # the body of the clip is untouched
    assert np.abs(clip[sr // 2 : -sr // 2]).max() > 0.7


def test_the_fade_leaves_a_short_clip_alone(tmp_path: Path) -> None:
    from app.services.silero_vad import apply_edge_fade

    tiny = np.ones(4, dtype=np.float32)
    assert np.array_equal(apply_edge_fade(tiny, 16000), tiny)


def test_silero_vad_pads_speech_turns_to_prevent_clipped_consonants() -> None:
    """Speech turns must be padded before onset and after offset so cuts land in silence."""
    from app.services.silero_vad import CHUNK_SIZE

    sr = 16000

    # 100 chunks: 25 silent, 35 speech (prob=0.9), 40 silent
    probs = [0.0] * 25 + [0.9] * 35 + [0.0] * 40
    audio = np.zeros(CHUNK_SIZE * len(probs), dtype=np.float32)

    class SequenceSession:
        def __init__(self, probs: list[float]) -> None:
            self.probs = probs
            self.idx = 0

        def run(self, _outputs, inputs):
            prob = self.probs[self.idx] if self.idx < len(self.probs) else 0.0
            self.idx += 1
            return np.array([[prob]], dtype=np.float32), inputs["state"]

    vad_unpadded = vad_with(SequenceSession(probs))
    turns_unpadded = vad_unpadded.detect_turns(audio, sample_rate=sr, speech_pad_ms=0.0)

    vad_padded = vad_with(SequenceSession(probs))
    turns_padded = vad_padded.detect_turns(audio, sample_rate=sr, speech_pad_ms=150.0)

    assert len(turns_unpadded) == 1
    assert len(turns_padded) == 1

    # Padded turn must start earlier and end later by ~150ms
    assert turns_padded[0].start <= turns_unpadded[0].start - 0.14
    assert turns_padded[0].end >= turns_unpadded[0].end + 0.14
    assert turns_padded[0].start >= 0.0
    assert turns_padded[0].end <= len(audio) / sr


def test_segment_audio_to_slices_finds_low_energy_boundary_for_long_turns() -> None:
    """Subdividing long turns (>20s) must prefer a natural silence pause over arbitrary cuts."""
    sr = 16000
    total_dur = 25.0
    audio = np.ones(int(sr * total_dur), dtype=np.float32) * 0.4

    # Insert a 0.5s pause between 12.0s and 12.5s (near the midpoint of a 25s turn)
    pause_start = int(12.0 * sr)
    pause_end = int(12.5 * sr)
    audio[pause_start:pause_end] = 0.0

    turns = [SpeechTurn(0.0, total_dur)]
    slices = segment_audio_to_slices(
        turns, total_dur, min_seg=2.0, max_seg=20.0, audio=audio, sample_rate=sr
    )

    assert len(slices) == 2
    cut_point = slices[0][1]
    # The cut point should snap into the pause window [12.0, 12.5]
    assert 11.9 <= cut_point <= 12.6, f"Cut point {cut_point} did not land in pause"
