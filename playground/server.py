"""The playground sidecar (D85): one fine-tuned Flex model on the CPU, over HTTP.

The harness backend never imports torch (D32, D79). It sends a recording here and logs the answer
(``app/llm/local_asr.py``). This process holds one model at a time and loads it from the folder
the Models page reads, ``data/models/asr/<slug>/``, mounted read-only at ``/models``:

* ``cpu/``: 04c's weight-only int8 export, loaded with the ``cpukit.py`` it carries;
* ``best/``: the bf16 fine-tune, used when 04c rejected int8.

Decoding is the standard decoder (docs/findings.md, decoder search):
* greedy, capped at 13 tokens per second of audio;
* a clip whose output loops (a 3-word run repeated 5+ times) is decoded again with a repetition
  penalty of 1.2 and no repeated 6-gram.

    GET  /health                                   {"loaded": {...} | null, "threads": n}
    POST /transcribe?model=<slug>&weights=cpu|best  body: 16 kHz mono audio (FLAC or WAV)

Requests are served one at a time: a second recording waits for the first. Errors that retrying
cannot fix (unknown model, unreadable audio) are 4xx, so the backend's retry policy leaves them.
"""

from __future__ import annotations

import gc
import io
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import soundfile as sf
import torch

MODELS = Path(os.environ.get("PLAYGROUND_MODELS", "/models"))
THREADS = int(os.environ.get("PLAYGROUND_THREADS", "8"))
PORT = int(os.environ.get("PLAYGROUND_PORT", "8100"))
LANG, MODE = "ne", "mixed"
SAMPLE_RATE = 16_000
MAX_BODY = 25 * 1024 * 1024
#: The densest training label is 12.6 tokens/s; more than this is the decoder running on.
MAX_TOKENS_PER_S = 13.0
RETRY = {"repetition_penalty": 1.2, "no_repeat_ngram_size": 6}
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
WEIGHTS = ("cpu", "best")

torch.set_num_threads(THREADS)
_lock = threading.Lock()
_loaded: dict | None = None


class BadRequest(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def is_loop(text: str) -> bool:
    """A 3-word sequence repeated 5 or more times: the decoder is stuck, not transcribing."""
    toks = text.split()
    top = Counter(zip(toks, toks[1:], toks[2:], strict=False)).most_common(1)
    return bool(top) and top[0][1] >= 5


def _load(slug: str, weights: str) -> dict:
    """The model in ``/models/<slug>/<weights>``, loading it (and dropping any other) if needed."""
    global _loaded
    key = f"{slug}/{weights}"
    if _loaded and _loaded["key"] == key:
        return _loaded
    path = MODELS / slug / weights
    if not (path / "config.json").is_file():
        raise BadRequest(404, f"no model at {path}")
    if _loaded:
        _loaded = None
        gc.collect()
    started = time.perf_counter()
    # The folder carries the model's own code (and, for int8, cpukit.py); import it from there.
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from indic_transcribe import MODES, IndicTranscribe

    if (path / "int8_modules.json").is_file():
        import cpukit

        asr, variant = cpukit.load_int8(path), "int8-weight-only"
    else:
        asr = IndicTranscribe.from_pretrained(str(path), device="cpu", dtype=torch.bfloat16)
        variant = "bf16"
    itn, romanized = MODES[MODE]
    _loaded = {
        "key": key,
        "asr": asr,
        "prompt": asr.tokenizer.encode_prompt(LANG, itn=itn, romanized=romanized),
        "dtype": next(asr.model.parameters()).dtype,
        "variant": variant,
        "load_s": round(time.perf_counter() - started, 2),
    }
    print(f"loaded {key} ({variant}) in {_loaded['load_s']} s", flush=True)
    return _loaded


def _decode(model: dict, wav: torch.Tensor, **gen) -> tuple[str, int]:
    asr, prompt = model["asr"], model["prompt"]
    feats, mask = asr._features(wav)
    cap = min(300, math.ceil(MAX_TOKENS_PER_S * wav.shape[0] / SAMPLE_RATE))
    with torch.inference_mode():
        out = asr.model.generate(
            input_features=feats.to(model["dtype"]),
            attention_mask=mask,
            decoder_input_ids=torch.tensor([prompt]),
            max_new_tokens=cap,
            do_sample=False,
            num_beams=1,
            **gen,
        )
    ids = out[0].tolist()
    text = asr.tokenizer.decode(asr.tokenizer.strip_prompt_and_trim(ids, prompt)).strip()
    return text, len(ids) - len(prompt)


def transcribe(slug: str, weights: str, body: bytes) -> dict:
    if not SLUG.fullmatch(slug) or weights not in WEIGHTS:
        raise BadRequest(400, "model must be a folder name and weights one of cpu, best")
    try:
        audio, rate = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
    except Exception as exc:
        raise BadRequest(400, f"unreadable audio: {exc}") from exc
    if rate != SAMPLE_RATE:
        raise BadRequest(400, f"audio must be {SAMPLE_RATE} Hz, got {rate}")
    wav = torch.from_numpy(audio.mean(axis=1))
    with _lock:
        loaded_already = bool(_loaded and _loaded["key"] == f"{slug}/{weights}")
        model = _load(slug, weights)
        started = time.perf_counter()
        text, tokens = _decode(model, wav)
        first = None
        if is_loop(text):
            first = text
            text, tokens = _decode(model, wav, **RETRY)
        compute_s = time.perf_counter() - started
    audio_s = wav.shape[0] / SAMPLE_RATE
    return {
        "text": text,
        "retried": first is not None,
        "first_text": first,
        "tokens": tokens,
        "audio_s": round(audio_s, 2),
        "compute_s": round(compute_s, 2),
        "rtf": round(compute_s / max(audio_s, 1e-6), 3),
        "load_s": None if loaded_already else model["load_s"],
        "variant": model["variant"],
        "threads": THREADS,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/health":
            return self._send(404, {"error": "not found"})
        loaded = _loaded
        info = {k: loaded[k] for k in ("key", "variant", "load_s")} if loaded else None
        self._send(200, {"status": "ok", "loaded": info, "threads": THREADS})

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path != "/transcribe":
            return self._send(404, {"error": "not found"})
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            return self._send(400, {"error": f"body must be 1 byte to {MAX_BODY} bytes"})
        body = self.rfile.read(length)
        try:
            self._send(200, transcribe(query.get("model", ""), query.get("weights", ""), body))
        except BadRequest as exc:
            self._send(exc.status, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            # A model that fails to load or decode fails the same way on a retry: 422, not 500.
            self._send(422, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} {format % args}", flush=True)


if __name__ == "__main__":
    print(f"playground on :{PORT}, models under {MODELS}, {THREADS} threads", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
