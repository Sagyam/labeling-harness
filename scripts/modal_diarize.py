"""Speaker diarization on a Modal GPU, for ingest to call (D79).

pyannote's ``speaker-diarization-community-1`` on an L4, behind a web endpoint that takes a whole
episode's 16 kHz mono FLAC and answers with one entry of the file
``app.services.diarization_import`` reads: turns, exclusive turns, labels, embeddings -- the same
shape, pipeline and model version as the Colab runs D78 imported.

The endpoint requires a Modal proxy-auth token (``modal workspace proxy-tokens create``); ingest
sends it from ``HARNESS_DIARIZATION__AUTH_TOKEN``. The model is gated: the Modal secret
``huggingface-secret`` holds an ``HF_TOKEN`` whose account accepted its terms.

    notebooks/.venv/bin/modal deploy scripts/modal_diarize.py
    notebooks/.venv/bin/modal run scripts/modal_diarize.py --audio-path episode.flac
"""

from __future__ import annotations

import io
import math
import os
from typing import Any

import modal
from fastapi import File, Form, UploadFile

MODEL_ID = "pyannote/speaker-diarization-community-1"

app = modal.App("nepanglish-diarization")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("pyannote.audio==4.0.7", "soundfile==0.13.1", "fastapi[standard]")
)


@app.cls(
    image=image,
    gpu="L4",
    timeout=900,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    # Idle GPU time is billed, and ingest's next request comes minutes later, after the episode
    # has transcribed. So the container stops the moment it answers, and every call is a cold
    # start (~25 s) rather than paying for a warm GPU nobody is about to use (D79).
    scaledown_window=2,  # Modal's minimum
)
class Diarizer:
    @modal.enter()
    def load_model(self) -> None:
        import torch
        from pyannote.audio import Pipeline

        self.pipeline = Pipeline.from_pretrained(MODEL_ID, token=os.environ["HF_TOKEN"])
        self.pipeline.to(torch.device("cuda"))

    def _diarize(self, audio: bytes, num_speakers: int | None) -> dict[str, Any]:
        import soundfile as sf
        import torch

        wav, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(wav.mean(axis=1))[None]  # (1, samples)
        kwargs = {"num_speakers": int(num_speakers)} if num_speakers and num_speakers > 0 else {}
        out = self.pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)

        def tracks(ann: Any) -> list[list[Any]]:
            return [
                [round(s.start, 3), round(s.end, 3), label]
                for s, _, label in ann.itertracks(yield_label=True)
            ]

        def vector(row: Any) -> list[float] | None:
            # A speaker it was told to find but never heard has a NaN embedding. JSON has no
            # NaN, so say "no embedding" rather than let the encoder pick a spelling.
            values = [float(x) for x in row]
            if not all(math.isfinite(x) for x in values):
                return None
            return [round(x, 5) for x in values]

        emb = out.speaker_embeddings
        labels = out.speaker_diarization.labels()
        return {
            "num_speakers": len(labels),
            "seconds": round(waveform.shape[-1] / sr, 3),
            "turns": tracks(out.speaker_diarization),  # overlapping: >1 speaker at once is kept
            "exclusive": tracks(out.exclusive_speaker_diarization),  # one speaker at a time
            "labels": labels,
            "embeddings": None if emb is None else [vector(row) for row in emb],
        }

    @modal.method()
    def diarize_bytes(self, audio: bytes, num_speakers: int | None = None) -> dict[str, Any]:
        return self._diarize(audio, num_speakers)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def diarize(
        self, file: UploadFile = File(...), num_speakers: int | None = Form(None)
    ) -> dict[str, Any]:
        return self._diarize(file.file.read(), num_speakers)


@app.local_entrypoint()
def main(audio_path: str, num_speakers: int = 0) -> None:
    from pathlib import Path

    audio = Path(audio_path).read_bytes()
    res = Diarizer().diarize_bytes.remote(audio, num_speakers or None)
    print(f"{res['seconds']:.0f} s of audio, {res['num_speakers']} speakers {res['labels']}")
    print(f"{len(res['turns'])} turns; first five: {res['turns'][:5]}")
