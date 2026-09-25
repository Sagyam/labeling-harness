"""PreDistill's per-file worker (roadmap §B step 1, D101).

One recording from the owner's zip goes through ingest's own stage 1 (``normalize_audio``: two-pass
loudness normalisation to 16 kHz mono FLAC) and stage 2 (Silero VAD, ``segment_audio_to_slices``:
2-20 s), imported from the harness's modules, which the notebook fetches at a pinned commit. What
it keeps is the whole normalised recording and its clip rows, the layout ``ftkit.AudioStore``
reads. One process per core: the VAD runs single-threaded here, so twelve workers do not fight
over twelve cores; its output does not depend on the thread count.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import distill
import soundfile as sf

_VAD: dict[str | None, Any] = {}


def _vad(model_path: str | None):
    """This process's VAD, loaded once: the harness's SileroVAD on one thread."""
    if model_path not in _VAD:
        from app.services.silero_vad import SileroVAD

        class OneThreadVAD(SileroVAD):
            def _load_model(self) -> None:
                if not self.model_path.is_file():
                    self._session = None  # detect_turns falls back to energy, as in ingest
                    return
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(
                    str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
                )

        _VAD[model_path] = OneThreadVAD(model_path)
    return _VAD[model_path]


def process_file(
    audio_path: str, out_dir: str, blocked: list[str], model_path: str | None = None
) -> dict[str, Any]:
    """Normalise and cut one recording into ``out_dir/episodes/<id>.flac`` and
    ``out_dir/sources/<id>.json`` (its metadata and clip rows).

    Returns ``status`` ``prepared``, ``blocked`` (a channel in ``distill.blocked_channels``) or
    ``failed`` (with ``error``; nothing is left behind), so one bad file never stops the run.

    Args:
        audio_path: The MP3 (or any audio ffmpeg reads), named ``<channel_name>_<NN>``.
        out_dir: Where the chunk's files go before they are uploaded.
        blocked: Channel names that may not enter the corpus.
        model_path: The Silero ONNX; None for the harness's default location.
    """
    started = time.perf_counter()
    source_id, channel = distill.source_from_filename(audio_path)
    base = {"source_id": source_id, "channel": channel, "file": Path(audio_path).name}
    if distill.channel_blocked(channel, blocked):
        return {**base, "status": "blocked"}
    from app.services.ingest.audio import normalize_audio
    from app.services.silero_vad import segment_audio_to_slices

    out = Path(out_dir)
    flac = out / "episodes" / f"{source_id}.flac"
    meta_path = out / "sources" / f"{source_id}.json"
    try:
        flac.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        duration = normalize_audio(Path(audio_path), flac)
        audio, sample_rate = sf.read(str(flac), dtype="float32")
        vad = _vad(model_path)
        turns = vad.detect_turns(audio, sample_rate=sample_rate)
        slices = segment_audio_to_slices(turns, duration, audio=audio, sample_rate=sample_rate)
        rows = distill.slice_rows(source_id, channel, slices)
        meta = {
            **base,
            "duration": round(duration, 3),
            "clips": len(rows),
            "speech_seconds": round(sum(r["duration"] for r in rows), 3),
            "vad": "silero" if vad.available else "energy",
        }
        meta_path.write_text(json.dumps({**meta, "rows": rows}, ensure_ascii=False), "utf-8")
        return {
            **meta,
            "status": "prepared",
            "seconds_taken": round(time.perf_counter() - started, 1),
        }
    except Exception as exc:  # one bad file is reported, never fatal
        flac.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
