"""Silero VAD speech activity detection and utterance cutting.

Performs CPU-based speech turn detection on 16 kHz mono audio using the official Silero VAD
ONNX model with fallback to energy-based segmentation if ONNX runtime is unavailable.
Enforces 2.0s - 20.0s utterance boundaries per the corpus specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.utils.hashing import sha256_file
from app.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"
#: Utterance bounds the cutter enforces. These are deliberately inside the flag thresholds in
#: ``queue.min_duration_seconds`` / ``queue.max_duration_seconds``: the cutter decides what a
#: segment *is*, the flags decide what looks wrong about one that arrived from anywhere.
MIN_SEG_SECONDS = 2.0
MAX_SEG_SECONDS = 20.0
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 32 ms at 16 kHz
#: Silero v5 conditions each window on the tail of the previous one: the tensor it expects is
#: ``CONTEXT_SIZE + CHUNK_SIZE`` samples, not a bare chunk. The ONNX input is declared with a
#: dynamic length, so feeding a bare chunk is accepted silently and the model then returns a
#: near-zero probability for everything -- speech, noise and digital silence alike.
CONTEXT_SIZE = 64


#: Ramp applied to each clip's first and last samples. Long enough to remove the step at a cut,
#: smooth enough to have zero derivative at boundaries, and short enough to be inaudible on speech.
FADE_MS = 15.0


def window_sizes(sample_rate: int) -> tuple[int, int]:
    """Chunk and context lengths in samples for a supported Silero sample rate.

    Silero v5 accepts 16 kHz and 8 kHz only, with a fixed window per rate.
    """
    if sample_rate == 8000:
        return CHUNK_SIZE // 2, CONTEXT_SIZE // 2
    return CHUNK_SIZE, CONTEXT_SIZE


@dataclass(frozen=True)
class SpeechTurn:
    """A detected continuous speech turn with start and end times in seconds."""

    start: float
    end: float


@dataclass(frozen=True)
class CutSegment:
    """A bounded segment (2.0s - 20.0s) cut from the normalized audio."""

    segment_id: str
    start_time: float
    end_time: float
    duration: float
    clip_path: Path
    clip_rel_path: str
    clip_checksum: str


class SileroVAD:
    """Fast CPU-based VAD using Silero VAD ONNX model."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._session: Any | None = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.is_file():
            logger.warning("silero_vad_model_missing", path=str(self.model_path))
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            logger.info("silero_vad_loaded", path=str(self.model_path))
        except Exception as exc:
            logger.warning("silero_vad_load_failed", error=str(exc))
            self._session = None

    def detect_turns(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = 0.5,
        min_speech_duration_ms: float = 250.0,
        min_silence_duration_ms: float = 300.0,
        speech_pad_ms: float = 150.0,
    ) -> list[SpeechTurn]:
        """Detect speech turns. Uses ONNX model if available, else energy VAD."""
        if self._session is None:
            return self._energy_detect_turns(audio, sample_rate, speech_pad_ms=speech_pad_ms)

        # Silero VAD requires float32 in [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        sr_tensor = np.array(sample_rate, dtype=np.int64)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        chunk_size, context_size = window_sizes(sample_rate)

        speech_probs: list[float] = []
        # The window handed to the model is the previous chunk's tail followed by this chunk.
        # The first window has no history, so it is padded with zeros.
        context = np.zeros(context_size, dtype=np.float32)
        num_chunks = len(audio) // chunk_size
        for i in range(num_chunks):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            window = np.concatenate((context, chunk))[np.newaxis, :]
            ort_inputs = {"input": window, "state": state, "sr": sr_tensor}
            out, state = self._session.run(None, ort_inputs)
            speech_probs.append(float(out[0][0]))
            context = chunk[-context_size:]

        # State machine to find speech turn start and end times
        total_duration = len(audio) / sample_rate
        chunk_sec = chunk_size / sample_rate
        pad_sec = speech_pad_ms / 1000.0
        min_speech_chunks = math.ceil((min_speech_duration_ms / 1000.0) / chunk_sec)
        min_silence_chunks = math.ceil((min_silence_duration_ms / 1000.0) / chunk_sec)

        turns: list[SpeechTurn] = []
        is_speech = False
        speech_start_idx = 0
        silence_count = 0

        for idx, prob in enumerate(speech_probs):
            if prob >= threshold:
                if not is_speech:
                    is_speech = True
                    speech_start_idx = idx
                silence_count = 0
            else:
                if is_speech:
                    silence_count += 1
                    if silence_count >= min_silence_chunks:
                        speech_end_idx = idx - silence_count
                        if speech_end_idx - speech_start_idx >= min_speech_chunks:
                            raw_s = speech_start_idx * chunk_sec - pad_sec
                            raw_e = speech_end_idx * chunk_sec + pad_sec
                            turns.append(
                                SpeechTurn(
                                    start=max(0.0, round(raw_s, 3)),
                                    end=min(round(total_duration, 3), round(raw_e, 3)),
                                )
                            )
                        is_speech = False
                        silence_count = 0

        if is_speech:
            speech_end_idx = len(speech_probs)
            if speech_end_idx - speech_start_idx >= min_speech_chunks:
                raw_s = speech_start_idx * chunk_sec - pad_sec
                raw_e = speech_end_idx * chunk_sec + pad_sec
                turns.append(
                    SpeechTurn(
                        start=max(0.0, round(raw_s, 3)),
                        end=min(round(total_duration, 3), round(raw_e, 3)),
                    )
                )

        if not turns:
            # Nothing crossed the threshold. Treating the file as one turn keeps the pipeline
            # running, but it means the cutter is about to slice on a fixed grid rather than on
            # silence -- so say so loudly. A whole episode with no detected speech is far more
            # likely to be a broken VAD than a silent recording.
            total_duration = len(audio) / sample_rate
            logger.warning(
                "silero_vad_no_speech_detected",
                duration_seconds=round(total_duration, 2),
                chunks=len(speech_probs),
                max_speech_prob=round(max(speech_probs), 4) if speech_probs else None,
            )
            if total_duration >= MIN_SEG_SECONDS:
                turns.append(SpeechTurn(start=0.0, end=round(total_duration, 3)))

        return turns

    def _energy_detect_turns(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        speech_pad_ms: float = 150.0,
    ) -> list[SpeechTurn]:
        """Simple energy envelope fallback when ONNX runtime is absent."""
        frame_len = int(sample_rate * 0.03)  # 30ms
        num_frames = len(audio) // frame_len
        if num_frames == 0:
            return []

        total_duration = len(audio) / sample_rate
        pad_sec = speech_pad_ms / 1000.0

        energies = [
            float(np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2)))
            for i in range(num_frames)
        ]
        threshold = max(0.01, float(np.mean(energies) * 0.4))

        turns: list[SpeechTurn] = []
        is_speech = False
        start_f = 0
        silence_f = 0

        for f, e in enumerate(energies):
            if e >= threshold:
                if not is_speech:
                    is_speech = True
                    start_f = f
                silence_f = 0
            else:
                if is_speech:
                    silence_f += 1
                    if silence_f >= 10:  # 300ms
                        end_f = f - silence_f
                        if (end_f - start_f) * 0.03 >= 1.0:
                            raw_s = start_f * 0.03 - pad_sec
                            raw_e = end_f * 0.03 + pad_sec
                            turns.append(
                                SpeechTurn(
                                    start=max(0.0, round(raw_s, 3)),
                                    end=min(round(total_duration, 3), round(raw_e, 3)),
                                )
                            )
                        is_speech = False
                        silence_f = 0

        if is_speech:
            end_f = num_frames
            if (end_f - start_f) * 0.03 >= 1.0:
                raw_s = start_f * 0.03 - pad_sec
                raw_e = end_f * 0.03 + pad_sec
                turns.append(
                    SpeechTurn(
                        start=max(0.0, round(raw_s, 3)),
                        end=min(round(total_duration, 3), round(raw_e, 3)),
                    )
                )

        if not turns and len(audio) / sample_rate >= MIN_SEG_SECONDS:
            turns.append(SpeechTurn(start=0.0, end=round(len(audio) / sample_rate, 3)))

        return turns


def _find_best_cut_point(
    audio: np.ndarray,
    sample_rate: int,
    target_time: float,
    window_sec: float = 0.75,
) -> float:
    """Find the point of minimum short-term RMS energy around target_time."""
    start_sec = max(0.0, target_time - window_sec)
    end_sec = min(len(audio) / sample_rate, target_time + window_sec)
    start_frame = int(start_sec * sample_rate)
    end_frame = int(end_sec * sample_rate)

    frame_len = int(sample_rate * 0.02)  # 20ms
    if end_frame - start_frame < frame_len:
        return target_time

    search_data = audio[start_frame:end_frame]
    hop = sample_rate // 100  # 10ms hop
    num_steps = (len(search_data) - frame_len) // hop
    if num_steps <= 0:
        return target_time

    min_energy = float("inf")
    best_offset = 0
    for s in range(num_steps):
        chunk = search_data[s * hop : s * hop + frame_len]
        energy = float(np.mean(chunk**2))
        if energy < min_energy:
            min_energy = energy
            best_offset = s * hop + frame_len // 2

    best_frame = start_frame + best_offset
    return round(best_frame / sample_rate, 3)


def segment_audio_to_slices(
    turns: list[SpeechTurn],
    total_duration: float,
    min_seg: float = MIN_SEG_SECONDS,
    max_seg: float = MAX_SEG_SECONDS,
    audio: np.ndarray | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> list[tuple[float, float]]:
    """Cut raw speech turns into bounded [min_seg, max_seg] intervals.

    Follows the reference partitioning algorithm:
    1. Merge consecutive turns separated by < 0.4s.
    2. Long turns (> max_seg) are subdivided into n steps where min_seg <= step <= max_seg,
       snapping cut points to natural low-energy pauses when audio is provided.
    3. Short turns (< min_seg) are merged with adjacent turns or padded to min_seg if isolated.
    """
    if not turns:
        if total_duration >= min_seg:
            turns = [SpeechTurn(0.0, total_duration)]
        else:
            return []

    # Merge nearby turns
    merged: list[SpeechTurn] = []
    for turn in turns:
        if not merged:
            merged.append(turn)
        else:
            prev = merged[-1]
            if turn.start - prev.end <= 0.4:
                merged[-1] = SpeechTurn(prev.start, max(prev.end, turn.end))
            else:
                merged.append(turn)

    slices: list[tuple[float, float]] = []
    for turn in merged:
        dur = turn.end - turn.start
        if dur <= 0:
            continue

        if dur > max_seg:
            n = max(1, math.ceil(dur / max_seg))
            step = dur / n
            current_start = turn.start
            for i in range(n):
                if i == n - 1:
                    target_end = turn.end
                else:
                    ideal_cut = turn.start + (i + 1) * step
                    if audio is not None:
                        earliest = max(current_start + min_seg, ideal_cut - 0.75)
                        latest = min(current_start + max_seg, ideal_cut + 0.75)
                        if earliest < latest:
                            win = min(0.75, (latest - earliest) / 2)
                            best = _find_best_cut_point(
                                audio, sample_rate, ideal_cut, window_sec=win
                            )
                            target_end = max(earliest, min(latest, best))
                        else:
                            target_end = ideal_cut
                    else:
                        target_end = ideal_cut

                a = current_start
                b = target_end
                if b - a >= min_seg or not slices:
                    slices.append((round(a, 3), round(b, 3)))
                    current_start = b
                elif slices:
                    # Append remainder to previous slice if below min_seg
                    prev_a, _ = slices[-1]
                    slices[-1] = (prev_a, round(b, 3))
                    current_start = b
        elif dur < min_seg:
            if slices and (turn.end - slices[-1][0]) <= max_seg:
                # Merge into previous slice
                prev_a, _ = slices[-1]
                slices[-1] = (prev_a, round(turn.end, 3))
            else:
                # Pad turn up to min_seg if space permits
                pad_end = min(total_duration, turn.start + min_seg)
                slices.append((round(turn.start, 3), round(pad_end, 3)))
        else:
            slices.append((round(turn.start, 3), round(turn.end, 3)))

    # Final pass: guarantee all bounds [min_seg, max_seg]
    cleaned: list[tuple[float, float]] = []
    for a, b in slices:
        b = min(total_duration, b)
        dur = b - a
        if dur < min_seg:
            b = min(total_duration, a + min_seg)
            dur = b - a
        if dur >= min_seg:
            cleaned.append((round(a, 3), round(b, 3)))

    return cleaned


def apply_edge_fade(
    clip: np.ndarray, sample_rate: int, milliseconds: float = FADE_MS
) -> np.ndarray:
    """Ramp a clip in and out using a raised-cosine curve so its edges cannot click.

    A cut lands wherever the boundary falls, which is rarely at a zero crossing. Starting or
    ending playback on a non-zero sample is a step change, and a step is heard as a click. A
    raised-cosine ramp removes it smoothly with zero first derivatives at both boundaries,
    eliminating high-frequency spectral splatter while remaining inaudible on speech.
    """
    fade = int(sample_rate * milliseconds / 1000.0)
    if fade <= 0 or clip.size < 2 * fade:
        return clip
    faded = clip.copy()
    n = np.arange(fade, dtype=clip.dtype)
    ramp = 0.5 * (1.0 - np.cos(np.pi * n / fade))
    faded[:fade] *= ramp
    faded[-fade:] *= ramp[::-1]
    return faded


def extract_clips(
    audio_path: Path | str,
    slices: list[tuple[float, float]],
    episode_id: str,
    output_dir: Path,
) -> list[CutSegment]:
    """Slice 16 kHz mono audio into segment FLAC files.

    Writes clips to ``output_dir / f'{segment_id}.flac'``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_data, sr = sf.read(str(audio_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    segments: list[CutSegment] = []
    for i, (start_sec, end_sec) in enumerate(slices):
        seg_id = f"{episode_id}_{i:05d}"
        clip_file = output_dir / f"{seg_id}.flac"

        start_frame = max(0, round(start_sec * sr))
        end_frame = min(len(audio_data), round(end_sec * sr))
        clip_data = apply_edge_fade(audio_data[start_frame:end_frame], sr)

        sf.write(str(clip_file), clip_data, sr, format="FLAC", subtype="PCM_16")
        checksum = sha256_file(clip_file)

        segments.append(
            CutSegment(
                segment_id=seg_id,
                start_time=start_sec,
                end_time=end_sec,
                duration=round(end_sec - start_sec, 3),
                clip_path=clip_file,
                clip_rel_path=f"clips/{seg_id}.flac",
                clip_checksum=checksum,
            )
        )

    return segments
