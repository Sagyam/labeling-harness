"""What every notebook builder shares: the cell helpers and the cells repeated across notebooks.

A builder (`build_*.py`) is a list of cells written as strings; this module turns them into an
`.ipynb`. It runs where the builders run, not in Colab. `python notebooks/src/build_all.py`
rebuilds every notebook, and backend/tests/test_notebook_builds.py fails when a committed notebook
is not what its builder writes.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def notebook(cells):
    nb = {
        "cells": cells,
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": [], "machine_shape": "hm"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    }
    for i, c in enumerate(cells):
        c["id"] = f"c{i:02d}"
    return nb


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.strip("\n"),
    }


GPU_NOTE = """
**Keeping the A100 busy** (all in `ftkit.py`):
- **No disk I/O in the loop.** Audio sits in RAM as int16, and a clip is a slice of its episode.
- **Little padding.** Batches are built by duration and padded to whole seconds, so there are only
  ~20 distinct shapes and cudnn picks its kernels once per shape.
- **The CPU never stalls the GPU.** DataLoader workers collate and pin the next batches while the
  current one runs.
- **Measured batch size.** A probe finds the largest micro-batch that survives forward+backward on
  the longest clip, with the optimizer state already allocated. Training uses 90% of it, and
  gradient accumulation makes up the effective batch.
- **A100 arithmetic.** bf16 autocast, TF32 matmuls and fused AdamW.
- **Measured, not assumed.** The log reports throughput (× realtime), padding waste, GPU
  utilisation and memory every few steps. Single-digit utilisation or high padding waste means a
  setting needs changing.
"""


def kit(name: str) -> dict:
    """A cell that writes `notebooks/src/<name>.py` beside the notebook, so the kernel and the
    DataLoader workers import the code this commit holds."""
    return code(f"%%writefile /content/ft/{name}.py\n" + (HERE / f"{name}.py").read_text())


def kits(*names: str) -> list[dict]:
    return [kit(name) for name in names]


_SETUP = r"""
%pip install -q __PIP__
import json
import os
import sys
from pathlib import Path

# Before torch touches CUDA: lets the allocator grow segments instead of fragmenting when batch
# shapes vary (a fresh kernel only).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

IN_COLAB = "google.colab" in sys.modules
# Secrets only work from a cell run in the Colab UI. When cells are driven from outside (the Colab
# MCP), run this cell by hand once. The token is then kept in the hub's own token file on the VM
# (where `hf auth login` puts it), so a kernel restart needs no click; the file goes when the
# runtime is deleted.
if IN_COLAB:
    from google.colab import userdata

    TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists()
                                  else userdata.get("HF_TOKEN"))
    if not TOKEN_FILE.exists():
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(os.environ["HF_TOKEN"])
        TOKEN_FILE.chmod(0o600)
FT = Path("/content/ft") if IN_COLAB else Path.cwd() / ".cache-ft"
FT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(FT))
"""

_RUN_FOLDER = r"""
if SMOKE:
    RUN_PREFIX += "-smoke"  # a smoke run's outputs never land among the real runs
OUT_ROOT = FT / "out" / RUN_PREFIX
OUT_ROOT.mkdir(parents=True, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
print("runs:", RUN_PREFIX, "| outputs:", OUT_ROOT, "| SMOKE RUN" if SMOKE else "")
"""


def setup(pip: str, tail: str = _RUN_FOLDER) -> dict:
    """The Setup cell: `pip` packages, the HF token, and the working folder `FT`, which is on the
    import path for the kits. `tail` follows; by default it names the run folder from
    `RUN_PREFIX` and `SMOKE`, which the Config cell defines."""
    return code(_SETUP.replace("__PIP__", pip).rstrip("\n") + "\n" + tail.strip("\n"))


SMOKE_NOTE = """
**Smoke run first.** `SMOKE = True` in Config trains and scores on a few hundred clips for one
epoch and writes under `<RUN_PREFIX>-smoke`, so every cell runs end to end in minutes before a
real run is paid for. Nothing here has been verified on a GPU until its smoke run has passed.
"""


def write(notebooks: dict[str, list], out_dir: Path) -> None:
    """Write `{file name: cells}` into `out_dir`, in the form the committed notebooks have."""
    for name, cells in notebooks.items():
        path = Path(out_dir) / name
        path.write_text(dumps(cells), encoding="utf-8")
        print("wrote", path, len(cells), "cells")


def dumps(cells: list) -> str:
    return json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n"
