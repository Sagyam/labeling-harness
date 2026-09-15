#!/usr/bin/env python
"""Export pyannote's Brouhaha (speech, SNR and C50 per frame) to ONNX, once (D87).

A one-time build tool, not part of the harness runtime. It needs ``torch``, ``onnx`` and
``huggingface_hub``, none of which may enter backend/pyproject.toml; run it in a throwaway
virtualenv, as ``export_aligner_onnx.py`` is run:

    python -m venv /tmp/brouhaha-export
    /tmp/brouhaha-export/bin/pip install torch onnx onnxruntime huggingface_hub numpy
    /tmp/brouhaha-export/bin/python scripts/export_brouhaha_onnx.py

``pyannote/brouhaha`` is gated: the Hugging Face account behind ``HF_TOKEN`` (or the cached
login) must have accepted its terms. Its licence is OpenRAIL, so the export stays on this
machine, in the gitignored ``data/models/``, and is never published.

**No pyannote.** The checkpoint is a pyannote 2.x ``PyanNet``: SincNet (80 learnable band-pass
filters, stride 10) -> 3-layer bidirectional LSTM (256) -> two 128-unit linear layers -> three
heads. It is rebuilt here in plain torch, and the pickled checkpoint is read with every class
outside torch replaced by an inert stub, so none of the file's own code runs. The heads are
brouhaha-vad's: ``vad = sigmoid(x)``, ``snr = 80 - 95 sigmoid(x)`` dB and
``c50 = 60 - 70 sigmoid(x)`` dB.

The export takes ``(batch, 1, 96000)`` -- six-second windows, the length it was trained on -- and
gives ``(batch, frames, 3)`` of ``[vad, snr, c50]`` on a 16.875 ms frame grid.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import pickle
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "pyannote/brouhaha"
#: A commit, never a branch: the weights must not change under a measurement that already ran.
MODEL_REVISION = "c93c9b537732dd50c28c0366c73f560c3a7aeb02"
DEFAULT_OUT = REPO_ROOT / "data" / "models" / "brouhaha.onnx"
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 6 * SAMPLE_RATE

SNR_MIN, SNR_MAX = -15.0, 80.0
C50_MIN, C50_MAX = -10.0, 60.0


def _stub_unpickler():
    """A pickle module whose unpickler turns every non-torch class into a do-nothing stub."""

    class Stub:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __setstate__(self, state) -> None:
            pass

    allowed = {"torch", "collections", "numpy", "builtins", "_codecs", "copyreg"}

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in allowed:
                return super().find_class(module, name)
            return type(name, (Stub,), {"__module__": module})

    return types.SimpleNamespace(Unpickler=Unpickler, load=pickle.load, __name__="stub_pickle")


def build_model(state_dict):
    """Brouhaha as a plain ``torch.nn.Module`` with the checkpoint's weights."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    def sinc_filters(low_hz_, band_hz_, window_, n_, min_low_hz=50.0, min_band_hz=50.0):
        """asteroid-filterbanks' ParamSincFB: cosine then sine band-pass filters, 251 taps."""
        low = min_low_hz + torch.abs(low_hz_)
        high = torch.clamp(low + min_band_hz + torch.abs(band_hz_), min_low_hz, SAMPLE_RATE / 2)
        band = (high - low)[:, 0]
        ft_low, ft_high = low @ n_, high @ n_
        cos_left = ((torch.sin(ft_high) - torch.sin(ft_low)) / (n_ / 2)) * window_
        cos = torch.cat([cos_left, 2 * band.view(-1, 1), torch.flip(cos_left, dims=[1])], dim=1)
        sin_left = ((torch.cos(ft_low) - torch.cos(ft_high)) / (n_ / 2)) * window_
        sin = torch.cat(
            [sin_left, torch.zeros_like(band.view(-1, 1)), -torch.flip(sin_left, dims=[1])], dim=1
        )
        filters = torch.cat([cos, sin], dim=0) / (2 * torch.cat([band, band])[:, None])
        return filters.view(-1, 1, filters.shape[-1])

    class Brouhaha(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            prefix = "sincnet.conv1d.0.filterbank."
            fb = [state_dict[prefix + k] for k in ("low_hz_", "band_hz_", "window_", "n_")]
            self.register_buffer("sinc", sinc_filters(*fb))
            self.wav_norm = nn.InstanceNorm1d(1, affine=True)
            self.conv1 = nn.Conv1d(80, 60, 5)
            self.conv2 = nn.Conv1d(60, 60, 5)
            self.norms = nn.ModuleList(
                [nn.InstanceNorm1d(80, affine=True), nn.InstanceNorm1d(60, affine=True),
                 nn.InstanceNorm1d(60, affine=True)]
            )  # fmt: skip
            self.lstm = nn.LSTM(60, 256, num_layers=3, bidirectional=True, batch_first=True)
            self.linear = nn.ModuleList([nn.Linear(512, 128), nn.Linear(128, 128)])
            self.heads = nn.ModuleDict({k: nn.Linear(128, 1) for k in ("vad", "snr", "c50")})

        def forward(self, waveforms):
            x = self.wav_norm(waveforms)
            x = F.conv1d(x, self.sinc, stride=10)
            x = torch.abs(x)  # SincNet's first layer, as pyannote's does
            for conv, norm in zip((None, self.conv1, self.conv2), self.norms, strict=True):
                if conv is not None:
                    x = conv(x)
                x = F.leaky_relu(norm(F.max_pool1d(x, 3, stride=3)))
            x, _ = self.lstm(x.transpose(1, 2))
            for linear in self.linear:
                x = F.leaky_relu(linear(x))
            vad = torch.sigmoid(self.heads["vad"](x))
            snr = (SNR_MIN - SNR_MAX) * torch.sigmoid(self.heads["snr"](x)) + SNR_MAX
            c50 = (C50_MIN - C50_MAX) * torch.sigmoid(self.heads["c50"](x)) + C50_MAX
            return torch.cat([vad, snr, c50], dim=-1)

    model = Brouhaha()
    renamed = {
        "wav_norm.weight": "sincnet.wav_norm1d.weight",
        "wav_norm.bias": "sincnet.wav_norm1d.bias",
        "conv1.weight": "sincnet.conv1d.1.weight",
        "conv1.bias": "sincnet.conv1d.1.bias",
        "conv2.weight": "sincnet.conv1d.2.weight",
        "conv2.bias": "sincnet.conv1d.2.bias",
    }
    for i in range(3):
        renamed[f"norms.{i}.weight"] = f"sincnet.norm1d.{i}.weight"
        renamed[f"norms.{i}.bias"] = f"sincnet.norm1d.{i}.bias"
    for name in model.state_dict():
        if name.startswith(("lstm.", "linear.")):
            renamed[name] = name
        elif name.startswith("heads."):
            renamed[name] = name.replace("heads.", "classifier.linears.")
    missing = [ours for ours, theirs in renamed.items() if theirs not in state_dict]
    if missing:
        raise KeyError(f"checkpoint lacks {missing}")
    model.load_state_dict(
        {"sinc": model.sinc} | {ours: state_dict[theirs] for ours, theirs in renamed.items()}
    )
    return model.eval()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, help="a local pytorch_model.bin instead")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("torch is required for the export; see this script's docstring.", file=sys.stderr)
        return 1

    path = args.checkpoint
    if path is None:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(MODEL_REPO, "pytorch_model.bin", revision=MODEL_REVISION))
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=_stub_unpickler()
    )
    model = build_model(checkpoint["state_dict"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(2, 1, WINDOW_SAMPLES)
    # The TorchScript exporter: torch >= 2.5 defaults to dynamo, which older torch lacks.
    exporter = inspect.signature(torch.onnx.export).parameters
    legacy = {"dynamo": False} if "dynamo" in exporter else {}
    torch.onnx.export(
        model,
        (dummy,),
        str(args.out),
        input_names=["waveforms"],
        output_names=["frames"],
        dynamic_axes={"waveforms": {0: "batch"}, "frames": {0: "batch"}},
        opset_version=args.opset,
        **legacy,
    )
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
