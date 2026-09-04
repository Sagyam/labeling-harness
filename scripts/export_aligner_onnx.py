#!/usr/bin/env python
"""Export the forced-alignment acoustic model to ONNX, once.

This is a one-time build tool, not part of the harness runtime. It needs ``torch`` and
``transformers``, which are deliberately NOT in backend/pyproject.toml: the aligner runs on
onnxruntime, which is already a dependency, and adding a multi-gigabyte training stack to a
FastAPI service to produce one file would be absurd. Run it in a throwaway virtualenv, exactly
as backend/app/services/models/silero_vad.onnx was a one-time fetch:

    python -m venv /tmp/aligner-export
    /tmp/aligner-export/bin/pip install torch transformers onnx onnxruntime
    /tmp/aligner-export/bin/python scripts/export_aligner_onnx.py

It writes two files next to silero_vad.onnx, both gitignored -- the model is far too large to
commit the way the 2.3 MB VAD was:

    backend/app/services/models/mms_fa.onnx        int8, ~300 MB (fp32 is ~1.2 GB)
    backend/app/services/models/mms_fa_vocab.json  the CTC label set, a few KB

The default model is a romanizing multilingual CTC head. Do not substitute a monolingual one:
this corpus mixes Devanagari and Latin inside single utterances, and a head trained on one
script cannot place the other (D32).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "MahmoudAshraf/mms-300m-1130-forced-aligner"
DEFAULT_OUT_DIR = REPO_ROOT / "backend" / "app" / "services" / "models"
#: Any waveform length works at runtime -- the time axis is exported dynamic. This is only the
#: shape torch traces with.
TRACE_SAMPLES = 16000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HuggingFace model id")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="keep the fp32 export (~1.2 GB) instead of quantizing to int8",
    )
    args = parser.parse_args(argv)

    try:
        import torch
        from transformers import AutoTokenizer, Wav2Vec2ForCTC
    except ImportError:
        print(
            "torch and transformers are required for the export and are not runtime "
            "dependencies. See this script's docstring for the throwaway-venv recipe.",
            file=sys.stderr,
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.out_dir / "mms_fa_vocab.json"
    model_path = args.out_dir / "mms_fa.onnx"

    print(f"loading {args.model_id}")
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    vocab = tokenizer.get_vocab()
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"wrote {vocab_path} ({len(vocab)} labels)")

    class Emissions(torch.nn.Module):
        """Logits only. The runtime does its own log-softmax and Viterbi decode in numpy."""

        def __init__(self, ctc: torch.nn.Module) -> None:
            super().__init__()
            self.ctc = ctc

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return self.ctc(waveform).logits

    fp32_path = model_path if args.no_quantize else model_path.with_suffix(".fp32.onnx")
    print(f"exporting to {fp32_path}")
    torch.onnx.export(
        Emissions(model),
        (torch.zeros(1, TRACE_SAMPLES),),
        str(fp32_path),
        input_names=["waveform"],
        output_names=["logits"],
        # Clips run 2-20 s, so the time axis must not be baked in at the traced length.
        dynamic_axes={"waveform": {1: "samples"}, "logits": {1: "frames"}},
        opset_version=args.opset,
    )

    if not args.no_quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(fp32_path),
            str(model_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )
        fp32_path.unlink(missing_ok=True)
        data_path = fp32_path.with_name(f"{fp32_path.name}.data")
        data_path.unlink(missing_ok=True)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"wrote {model_path} ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
