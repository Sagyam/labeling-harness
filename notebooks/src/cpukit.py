"""Weight-only int8 for Indic-Transcribe-Flex on a CPU (04c's CPU export).

Written out by a %%writefile cell in 04c and copied into `OUT/cpu/`, so the export loads anywhere
with plain PyTorch -- no torchao, no ONNX.

**Why weight-only.** Measured on the owner's Ryzen 7 7700X, 2026-09-15, base Flex on 10 gold clips:
fp32 RTF 0.41, bf16 RTF 0.18 at the same WER (21.8%). PyTorch's *dynamic* int8, which also
quantizes activations, broke the model: WER 97-113%, loops, and English written in Devanagari.
Decoding is memory-bound (each token re-reads the 24 decoder layers), so the speed lives in the
weight bytes. This keeps activations in bf16 and stores weights as int8 with one scale per output
channel: half the bytes of bf16 per token, none of dynamic int8's activation rounding.

Only the transformer layers are quantized. `pre_encode.out` must keep real parameters (the encoder
reads its dtype from them) and the tied `lm_head` is small.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch import nn

QUANT_MANIFEST = "int8_modules.json"
INT8_WEIGHTS = "model.int8.safetensors"


class Int8Linear(nn.Module):
    """y = x @ (q * s)^T + b with q int8 (out, in) and one scale s per output channel."""

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.out_features, self.in_features = qweight.shape
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, dtype: torch.dtype) -> Int8Linear:
        # Quantize from the bf16 values `best/` stores, so the fp32 model scored on the GPU and the
        # CPU export built from `best/` hold exactly the same int8 weights.
        w = linear.weight.detach().to(torch.bfloat16).float()
        scales = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        q = torch.round(w / scales[:, None]).clamp(-127, 127).to(torch.int8)
        bias = None if linear.bias is None else linear.bias.detach().to(dtype)
        return cls(q, scales.to(dtype), bias)

    @classmethod
    def empty(
        cls, in_features: int, out_features: int, bias: bool, dtype: torch.dtype
    ) -> Int8Linear:
        return cls(
            torch.zeros(out_features, in_features, dtype=torch.int8),
            torch.ones(out_features, dtype=dtype),
            torch.zeros(out_features, dtype=dtype) if bias else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        if x2.is_cuda or not hasattr(torch, "_weight_int8pack_mm"):
            # The same arithmetic, dequantized: on the GPU (accuracy only) or on an older torch.
            y = x2 @ (self.qweight.to(x2.dtype) * self.scales.to(x2.dtype)[:, None]).t()
        else:
            # PyTorch's CPU kernel for int8 weights with bf16/fp32 activations.
            y = torch._weight_int8pack_mm(x2.contiguous(), self.qweight, self.scales.to(x2.dtype))
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.reshape(*shape[:-1], self.out_features)


def _targets(model: nn.Module) -> list[str]:
    """Names of the nn.Linear modules inside the encoder and decoder layers."""
    inner = model.model
    names = []
    for part in ("encoder", "decoder"):
        for name, module in getattr(inner, part).layers.named_modules():
            if isinstance(module, nn.Linear):
                names.append(f"model.{part}.layers.{name}")
    return names


def _set(model: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent), leaf, module)


def quantize_(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> list[str]:
    """Swap every transformer-layer Linear for an Int8Linear, in place; returns their names."""
    names = _targets(model)
    for name in names:
        _set(model, name, Int8Linear.from_linear(model.get_submodule(name), dtype))
    return names


def save_int8(model: nn.Module, names: list[str], src: Path, dst: Path) -> None:
    """Write a loadable CPU folder: int8 weights, the list of quantized modules, and the model's
    code, tokenizer and feature files copied from ``src`` (a `best/` folder)."""
    import shutil

    from safetensors.torch import save_file

    dst.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    # lm_head is tied to the decoder embedding; keep one copy, as save_pretrained does.
    state.pop("lm_head.weight", None)
    save_file(state, str(dst / INT8_WEIGHTS))
    (dst / QUANT_MANIFEST).write_text(json.dumps(names))
    for f in src.iterdir():
        if f.name != "model.safetensors" and f.is_file():
            shutil.copy(f, dst / f.name)
    shutil.copy(Path(__file__), dst / "cpukit.py")


def load_int8(path: Path | str, threads: int | None = None):
    """An IndicTranscribe on the CPU from a folder written by :func:`save_int8`.

    Usage, from anywhere::

        import sys; sys.path.insert(0, "<folder>")
        import cpukit
        asr = cpukit.load_int8("<folder>")
        asr.transcribe("clip.wav", lang="ne", mode="mixed")
    """
    import sys

    from safetensors.torch import load_file

    path = Path(path)
    sys.path.insert(0, str(path))
    from configuration_indic_canary import IndicCanaryConfig
    from feature_extraction_indic_canary import IndicCanaryFeatureExtractor
    from indic_transcribe import IndicTranscribe
    from modeling_indic_canary import IndicCanaryForConditionalGeneration
    from tokenization_indic_canary import IndicCanaryTokenizer

    if threads:
        torch.set_num_threads(threads)
    config = IndicCanaryConfig.from_pretrained(path)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # build the skeleton at half the RAM of fp32
    try:
        model = IndicCanaryForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(previous)
    for name in json.loads((path / QUANT_MANIFEST).read_text()):
        linear = model.get_submodule(name)
        _set(
            model,
            name,
            Int8Linear.empty(
                linear.in_features, linear.out_features, linear.bias is not None, torch.bfloat16
            ),
        )
    missing, unexpected = model.load_state_dict(load_file(str(path / INT8_WEIGHTS)), strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(
            f"int8 export does not fit the model: missing {missing[:3]}, "
            f"unexpected {unexpected[:3]}"
        )
    model.tie_weights()
    model.eval()
    return IndicTranscribe(
        model,
        IndicCanaryFeatureExtractor.from_pretrained(path, device="cpu"),
        IndicCanaryTokenizer.from_pretrained(path),
        "cpu",
    )


def time_clips(
    asr, clips: list[tuple[str, torch.Tensor]], *, lang: str, mode: str, tokens_per_s: float = 13.0
) -> dict:
    """Greedy-decode each (id, 16 kHz float waveform) on the CPU: RTF, and wall time per decoded
    token (encoder included, so it slightly overstates the decoder's own cost).

    The first clip is decoded twice and only the second is timed (kernel selection, allocations).
    """
    from indic_transcribe import MODES

    itn, romanized = MODES[mode]
    prompt = asr.tokenizer.encode_prompt(lang, itn=itn, romanized=romanized)
    dtype = next(p for p in asr.model.parameters()).dtype
    texts, audio_s, total_s, tokens = {}, 0.0, 0.0, 0

    def one(wav):
        feats, mask = asr._features(wav)
        seconds = wav.shape[0] / asr.fe.sample_rate
        cap = min(300, math.ceil(tokens_per_s * seconds))
        with torch.inference_mode():
            t0 = time.perf_counter()
            out = asr.model.generate(
                input_features=feats.to(dtype),
                attention_mask=mask,
                decoder_input_ids=torch.tensor([prompt]),
                max_new_tokens=cap,
            )
            took = time.perf_counter() - t0
        text = asr.tokenizer.decode(asr.tokenizer.strip_prompt_and_trim(out[0].tolist(), prompt))
        return text, seconds, took, out.shape[1] - len(prompt)

    one(clips[0][1])
    for clip_id, wav in clips:
        text, seconds, took, n = one(wav)
        texts[clip_id] = text
        audio_s, total_s, tokens = audio_s + seconds, total_s + took, tokens + n
    return {
        "clips": len(clips),
        "audio_s": round(audio_s, 1),
        "rtf": round(total_s / audio_s, 3),
        "ms_per_token": round(1000 * total_s / max(1, tokens), 1),
        "tokens_per_audio_s": round(tokens / audio_s, 2),
        "texts": texts,
    }
