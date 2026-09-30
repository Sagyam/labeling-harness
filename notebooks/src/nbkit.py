"""What every notebook builder shares: the cell helpers and the notes repeated across notebooks.

A builder (`build_*.py`) is a list of cells written as strings; this module turns them into an
`.ipynb`. It runs where the builders run, not in Colab.
"""


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
