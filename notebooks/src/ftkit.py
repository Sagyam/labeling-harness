"""Shared fine-tuning kit for the Nepanglish ASR notebooks (Flex, 03a-03e; the students, 06).

Everything that is not model-specific: the dataset held in RAM, duration-bucketed batches, the
harness scorer, a GPU utilisation monitor, a batch-size probe and one training loop. Each notebook
writes this file out, so it imports exactly the same code as the main kernel. Keep it importable
on Python 3.10+ with numpy 1.x or 2.x.

How the GPU is kept busy:
  * audio is decoded once into RAM as int16; a clip is a slice, so no disk I/O in the loop;
  * batches are built by duration and padded to whole seconds, so there is little padding and
    only ~20 distinct shapes (cudnn.benchmark can then pick kernels once per shape);
  * DataLoader workers collate and pin the next batches while the GPU runs the current one;
  * a probe finds the largest micro-batch that survives forward+backward at the longest clip,
    with the optimizer state already allocated, and training uses a fixed fraction of it;
  * bf16 autocast, TF32 matmuls and fused AdamW; gradient accumulation reaches the effective
    batch; utilisation, throughput and padding waste are logged, not assumed.
"""

from __future__ import annotations

import glob
import json
import math
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

REPO = "Sagyam/nepanglish-asr"
SR = 16_000


# --- data --------------------------------------------------------------------------------------


def download_dataset(
    local: str | None = None, episodes: Sequence[str] = (), analytics: bool = False
) -> Path:
    """The HF dataset snapshot, or a local copy in the same layout.

    Labels, manifests and the harness's scorer always come; audio only for the recordings named
    in `episodes` (whole FLACs, 8 GiB for every split), and the analytics export, which carries
    the speaker turns, only when asked. Call it once without episodes to read the splits, then
    again with the episodes the notebook decodes or trains on: both calls return the same folder.
    """
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    patterns = ["training/training.jsonl", "training/manifest.json", "gold/*", "harness/*"]
    if analytics:
        patterns.append("analytics/*")
    patterns += [f"training/episodes/{glob.escape(e)}.flac" for e in sorted(set(episodes))]
    return Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=patterns))


def load_splits(data: Path) -> dict[str, list[dict]]:
    """train / val from the training export, gold from the gold export; asserts disjointness."""
    rows = [
        json.loads(line) for line in (data / "training" / "training.jsonl").open(encoding="utf-8")
    ]
    gold = [json.loads(line) for line in (data / "gold" / "gold.jsonl").open(encoding="utf-8")]
    splits = {
        "train": [r for r in rows if r["split"] == "train"],
        "val": [r for r in rows if r["split"] == "val"],
        "gold": gold,
    }
    trained = {r["segment_id"] for r in rows}
    assert not trained & {r["segment_id"] for r in gold}, "a gold clip is in the training export"
    return splits


def attach_speaker_turns(data: Path, rows: Sequence[dict]) -> int:
    """Copy each row's diarized turns, with their linked voices (D78, D87), from the analytics
    export: the training export does not carry them. Returns how many rows have turns."""
    want = {r["segment_id"]: r for r in rows}
    with (data / "analytics" / "analytics.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            a = json.loads(line)
            if a["segment_id"] in want:
                want[a["segment_id"]]["speaker_turns"] = a.get("speaker_turns")
    return sum(bool(r.get("speaker_turns")) for r in rows)


def duration(row: dict) -> float:
    return row["end_time"] - row["start_time"]


class AudioStore:
    """Every episode decoded once into RAM as int16; a clip is a slice of it (a view, no copy).

    ``folder`` is where the whole recordings are, under ``data``: the labelled export's
    ``training/episodes`` by default, or the distillation corpus's ``distill/episodes`` (D101)."""

    def __init__(self, data: Path, episode_ids: Sequence[str], folder: str = "training/episodes"):
        self.audio: dict[str, np.ndarray] = {}
        for ep in sorted(set(episode_ids)):
            audio, sr = sf.read(data / folder / f"{ep}.flac", dtype="int16")
            assert sr == SR and audio.ndim == 1, (ep, sr, audio.shape)
            self.audio[ep] = audio

    def clip(self, row: dict) -> np.ndarray:
        if "audio" in row:  # a public-benchmark clip carries its own audio (evalkit)
            return row["audio"]
        audio = self.audio[row["episode_id"]]
        return audio[round(row["start_time"] * SR) : round(row["end_time"] * SR)]

    def clip_f32(self, row: dict) -> torch.Tensor:
        return torch.from_numpy(self.clip(row).astype(np.float32) / 32768.0)

    @property
    def gib(self) -> float:
        return sum(a.nbytes for a in self.audio.values()) / 2**30


def bucket_batches(
    rows: Sequence[dict],
    *,
    budget_s: float,
    max_items: int,
    pad_to_s: float,
    shuffle: bool,
    seed: int = 0,
    stretch: float = 1.0,
) -> list[list[int]]:
    """Index batches whose padded size (items x longest clip, rounded up to `pad_to_s`) fits
    `budget_s`. Sorting by duration keeps padding low; shuffling reorders whole batches and
    jitters lengths slightly so the same clips do not always share a batch.

    `stretch` sizes every clip as if it were that many times longer, as a clip slowed by speed
    perturbation is, so the batch still fits after it. It is applied before the padding: a 2.9 s
    clip slowed to 0.9x pads to 4 s, not 3 s, which is why shrinking the budget by the speed
    factor is not enough (03c's speed run ran out of memory that way on 2026-10-03)."""
    rng = random.Random(seed)
    jitter = [rng.uniform(-0.3, 0.3) if shuffle else 0.0 for _ in rows]
    order = sorted(range(len(rows)), key=lambda i: duration(rows[i]) + jitter[i])
    batches: list[list[int]] = []
    cur: list[int] = []
    longest = 0.0
    for i in order:
        d = math.ceil(duration(rows[i]) * stretch / pad_to_s) * pad_to_s
        if cur and ((len(cur) + 1) * max(longest, d) > budget_s or len(cur) >= max_items):
            batches.append(cur)
            cur, longest = [], 0.0
        cur.append(i)
        longest = max(longest, d)
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


def pad_len(samples: int, pad_to_s: float) -> int:
    step = int(pad_to_s * SR)
    return math.ceil(samples / step) * step


class _Batches(torch.utils.data.Dataset):
    def __init__(self, rows, batches, collate):
        self.rows, self.batches, self.collate = rows, batches, collate

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.collate([self.rows[j] for j in self.batches[i]])


def loader(rows, batches, collate, workers: int) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        _Batches(rows, batches, collate),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=4 if workers else None,
        persistent_workers=False,
    )


# --- scoring -----------------------------------------------------------------------------------


def is_loop(text: str) -> bool:
    """A 3-word sequence repeated 5 or more times: the decoder is stuck, not transcribing."""
    toks = text.split()
    top = Counter(zip(toks, toks[1:], toks[2:], strict=False)).most_common(1)
    return bool(top) and top[0][1] >= 5


class RetryLoops:
    """Wraps a batch `transcribe`: a clip whose first decode loops is decoded again, alone, by
    `retry` (anti-repetition settings); every other clip keeps its first decode untouched.
    `log` keeps (segment_id, first, retried), so the effect is measurable within one run."""

    def __init__(self, transcribe: Callable[[list[dict]], list[str]], retry: Callable[[dict], str]):
        self.transcribe, self.retry = transcribe, retry
        self.log: list[tuple[str, str, str]] = []

    def __call__(self, rows: list[dict]) -> list[str]:
        texts = self.transcribe(rows)
        for i, text in enumerate(texts):
            if is_loop(text):
                texts[i] = self.retry(rows[i])
                self.log.append((rows[i]["segment_id"], text, texts[i]))
        return texts


def harness_scorer(data: Path, work: Path) -> Callable[[Sequence[str], Sequence[str]], dict]:
    """fold.py from the dataset's harness/, imported from a real copy: the HF cache stores files as
    symlinks into its blob store, and normalize.py finds its config via its resolved path.

    Two layouts are accepted: the repo's (backend/app/services/, config/), and the flat one the
    2026-09-21 export uploaded (fold.py, normalize.py, normalization.yaml side by side), which
    is laid back out here because normalize.py looks for ../../../config/. error_mining.py,
    error_store.py and attribution.py travel with fold.py when the copy has them
    (D110); evalkit writes no error files without them.

    Each `per_clip` count keeps the clip's folded `alignment`, so mining its errors aligns nothing
    again; `summarize` adds up everything else."""
    src, dst = data / "harness", work / "harness"
    if not dst.exists():
        if (src / "fold.py").exists():
            (dst / "backend" / "app" / "services").mkdir(parents=True)
            (dst / "config").mkdir()
            for name in (
                "fold.py",
                "normalize.py",
                "error_mining.py",
                "error_store.py",
                "attribution.py",
            ):
                if name in ("fold.py", "normalize.py") or (src / name).exists():
                    shutil.copyfile(src / name, dst / "backend" / "app" / "services" / name)
            shutil.copyfile(src / "normalization.yaml", dst / "config" / "normalization.yaml")
        else:
            shutil.copytree(src, dst)
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[name]
    sys.path.insert(0, str(dst / "backend"))
    from app.services.fold import fold_tokens, fold_version, word_errors
    from rapidfuzz.distance import Levenshtein

    dev = re.compile(r"[ऀ-ॿ]")

    def chars(text: str) -> str:
        return " ".join(t if dev.search(t) else t.lower() for t in fold_tokens(text))

    def per_clip(refs: Sequence[str], hyps: Sequence[str]) -> list[dict]:
        """Every count a score needs, per clip, from one alignment each: `summarize` adds any
        subset of them up, so scoring per class or pairing against another system aligns
        nothing again. errors = sub + del + ins (folded)."""
        out = []
        for ref, hyp in zip(refs, hyps, strict=True):
            folded = word_errors(ref, hyp)
            raw = word_errors(ref, hyp, folded=False)
            ref_chars, hyp_chars = chars(ref), chars(hyp)
            out.append(
                {
                    "words": len(fold_tokens(ref)),
                    "errors": folded.errors,
                    "sub": folded.substitutions,
                    "del": folded.deletions,
                    "ins": folded.insertions,
                    "raw_words": raw.ref_words,
                    "raw_errors": raw.errors,
                    "chars": len(ref_chars),
                    "char_errors": Levenshtein.distance(ref_chars, hyp_chars),
                    "loop": int(is_loop(hyp)),
                    "alignment": folded,
                }
            )
        return out

    def summarize(clips: Sequence[dict]) -> dict:
        """The score of a set of clips from their `per_clip` counts."""
        t = (
            {k: sum(c[k] for c in clips) for k in clips[0] if k != "alignment"}
            if clips
            else Counter()
        )
        words = max(t["words"], 1)
        folded = t["sub"] + t["del"] + t["ins"]
        assert folded == t["errors"], "S + D + I must add up to the folded errors"
        return {
            "wer": 100 * t["errors"] / words,
            "raw_wer": 100 * t["raw_errors"] / max(t["raw_words"], 1),
            "cer": 100 * t["char_errors"] / max(t["chars"], 1),
            # folded, per 100 reference words. In crosstalk a label keeps what was audible, which is
            # not always both voices, so a model that writes the other one can be charged insertions
            "sub": 100 * t["sub"] / words,
            "del": 100 * t["del"] / words,
            "ins": 100 * t["ins"] / words,
            "loops": t["loop"],
            "clips": len(clips),
        }

    def score(refs: Sequence[str], hyps: Sequence[str]) -> dict:
        return summarize(per_clip(refs, hyps))

    score.per_clip = per_clip
    score.summarize = summarize
    score.fold_version = fold_version()  # for the model card: which fold the numbers are in
    return score


def transcribe_rows(
    rows: Sequence[dict],
    transcribe: Callable[[list[dict]], list[str]],
    *,
    budget_s: float,
    max_items: int,
    pad_to_s: float,
) -> tuple[list[str], list[float]]:
    """Run `transcribe` over duration-bucketed batches; returns texts and per-clip compute seconds
    in the original row order."""
    texts: list[str] = [""] * len(rows)
    compute = [0.0] * len(rows)
    for batch in bucket_batches(
        rows, budget_s=budget_s, max_items=max_items, pad_to_s=pad_to_s, shuffle=False
    ):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = transcribe([rows[i] for i in batch])
        torch.cuda.synchronize()
        took = time.perf_counter() - t0
        audio_s = sum(duration(rows[i]) for i in batch)
        for i, text in zip(batch, out, strict=True):
            texts[i] = text.strip()
            compute[i] = took * duration(rows[i]) / audio_s
    return texts, compute


def write_hyps(
    path: Path, rows: Sequence[dict], texts: Sequence[str], compute: Sequence[float]
) -> None:
    """The hypothesis-cache format the bake-off comparisons used, so the result can join them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r, text, c in zip(rows, texts, compute, strict=True):
            fh.write(
                json.dumps(
                    {"segment_id": r["segment_id"], "text": text, "compute_s": c},
                    ensure_ascii=False,
                )
                + "\n"
            )


# --- GPU ---------------------------------------------------------------------------------------


def fast_cuda() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # safe: shapes are padded to whole seconds
    torch.set_float32_matmul_precision("high")


class GpuMonitor:
    """Samples nvidia-smi in a background thread; `window()` returns the mean utilisation and the
    peak memory since the previous call."""

    def __init__(self, every: float = 1.0):
        self.every, self._util, self._mem = every, [], []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                util, mem = (float(x) for x in out.stdout.strip().splitlines()[0].split(","))
                self._util.append(util)
                self._mem.append(mem)
            except Exception:
                pass
            self._stop.wait(self.every)

    def start(self) -> GpuMonitor:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def window(self) -> dict:
        util, mem = self._util, self._mem
        self._util, self._mem = [], []
        return {
            "gpu_util": float(np.mean(util)) if util else float("nan"),
            "gpu_mem_gib": max(mem) / 1024 if mem else float("nan"),
        }


def init_optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Allocate AdamW's moment buffers before probing, without moving any weight: with zero
    gradients and zero weight decay an AdamW step is a no-op, but it creates the state."""
    decay = [g["weight_decay"] for g in optimizer.param_groups]
    for g in optimizer.param_groups:
        g["weight_decay"] = 0.0
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for g, d in zip(optimizer.param_groups, decay, strict=True):
        g["weight_decay"] = d
    for state in optimizer.state.values():
        if "step" in state:
            state["step"].zero_()


@contextmanager
def batchnorm_kept(model: torch.nn.Module) -> Iterator[None]:
    """Restore every BatchNorm layer's running statistics on exit. A forward pass in train mode
    updates them with no optimizer step at all: on 2026-10-01 the probe's random noise had
    rewritten about half of Flex's, and 03a's "val WER before training" read 13.81 for a model
    that scores 9.24."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    saved = [{k: v.clone() for k, v in m.state_dict().items()} for m in bns]
    try:
        yield
    finally:
        for m, s in zip(bns, saved, strict=True):
            m.load_state_dict(s)


def probe_max_items(step: Callable[[int], None], lo: int, hi: int, model: torch.nn.Module) -> int:
    """The largest n in [lo, hi] for which `step(n)` (one forward+backward at the worst case)
    fits in memory. `step` must not touch the gradients. The model's weights never move, and its
    BatchNorm statistics are restored after, so it leaves the model as it found it.

    The gradient buffer stays allocated throughout, as it does in training from the second
    micro-batch of every accumulated step: freeing it between probes measured the activations
    without it and picked a batch that ran out of memory once training accumulated."""
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = torch.zeros_like(p)
    best = 0
    with batchnorm_kept(model):
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                step(mid)
                torch.cuda.synchronize()
                best, lo = mid, mid + 1
            except torch.cuda.OutOfMemoryError:
                hi = mid - 1
            for p in params:
                p.grad.zero_()
            torch.cuda.empty_cache()
    for p in params:
        p.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return best


# --- training ----------------------------------------------------------------------------------


@dataclass
class TrainConfig:
    name: str
    out: str
    epochs: int = 10
    lr: float = 1e-5
    schedule: str = "linear"  # linear warmup+decay, or "tristage" (fairseq2's default)
    warmup_frac: float = 0.1
    effective_s: float = 720.0  # audio seconds per optimizer step, reached by accumulation
    weight_decay: float = 0.0
    clip_norm: float = 1.0
    patience: int = 3  # evaluations without a val-WER improvement before stopping
    # WER points an evaluation must gain on the last counted one to reset `patience`. 0 counts
    # any gain, so a crawl of 0.2 points an epoch never stops. The best weights are kept either way.
    min_delta: float = 0.0
    seed: int = 0
    log_every: int = 10
    workers: int = 6


def lr_factor(step: int, total: int, cfg: TrainConfig) -> float:
    warm = max(1, int(cfg.warmup_frac * total))
    if step < warm:
        return (step + 1) / warm
    if cfg.schedule == "tristage":  # 10% warmup, 40% hold, 50% exponential decay to 5%
        hold_end = int(0.5 * total)
        if step < hold_end:
            return 1.0
        frac = (step - hold_end) / max(1, total - hold_end)
        return math.exp(math.log(0.05) * frac)
    return max(0.0, (total - step) / max(1, total - warm))


def group_steps(rows, batches, effective_s: float) -> list[list[list[int]]]:
    """Micro-batches grouped into optimizer steps of about `effective_s` seconds of real audio."""
    steps, cur, acc = [], [], 0.0
    for b in batches:
        cur.append(b)
        acc += sum(duration(rows[i]) for i in b)
        if acc >= effective_s:
            steps.append(cur)
            cur, acc = [], 0.0
    if cur:
        steps.append(cur)
    return steps


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Print `label...` before a step and `label: done in N s` after it, so a step that stalls
    shows which one it is instead of leaving a cell silent."""
    print(f"{label}...", flush=True)
    t0 = time.perf_counter()
    yield
    print(f"{label}: done in {time.perf_counter() - t0:.0f} s", flush=True)


def sid(m: dict) -> str:
    """A score's error split, per 100 reference words: S + D + I is its folded WER. Read all three
    on crosstalk: the labels drop the other voice, so hearing it costs insertions."""
    return f"S {m['sub']:.2f}  D {m['del']:.2f}  I {m['ins']:.2f}"


def patience_step(wer: float, counted: float, bad: int, min_delta: float) -> tuple[float, int]:
    """Early stopping after one evaluation: the val WER that last reset the count, and the
    evaluations since. An evaluation at 100 or more counts for nothing: a model that writes only
    blanks, or more garbage than the references hold, has not started transcribing. From random
    weights that can last several epochs while the loss falls (06f, 2026-10-05), and patience
    used to stop it there."""
    if wer >= 100.0:
        return counted, bad
    if wer < counted - min_delta:  # at 0: the old rule, any strict gain
        return wer, 0
    return counted, bad + 1


def retry_note(val: dict) -> str:
    """For an `evaluate` that wraps its decode in RetryLoops and reports the greedy score too."""
    if "retried" not in val:
        return ""
    return f"  (greedy WER {val['greedy_wer']:.2f}, {val['retried']} retried)"


def speed_check(
    model: torch.nn.Module,
    *,
    rows: Sequence[dict],
    batches: list[list[int]],
    collate: Callable[[list[dict]], Any],
    loss_fn: Callable[[torch.nn.Module, Any], tuple[torch.Tensor, int]],
    evaluate: Callable[[torch.nn.Module], dict],
    val_rows: Sequence[dict],
    gold_rows: Sequence[dict],
    epochs: int,
    monitor: GpuMonitor,
    workers: int = 6,
    n_micro: int = 10,
) -> dict:
    """Time the run before committing to it: forward+backward on `n_micro` real training
    micro-batches (no optimizer step, so no weight moves), then one full val pass, which is also
    the val WER before training. Projects the wall time of every epoch and the gold pass.

    The micro-batches run twice and only the second pass is timed, so cudnn's per-shape kernel
    search and worker start-up are excluded. Gradients stay allocated between micro-batches, as
    they do under accumulation, so the memory reading is training's. BatchNorm running
    statistics are restored after."""
    sample = batches[:n_micro]
    it = iter(loader(rows, sample + sample, collate, workers))

    def run() -> tuple[float, float, int]:
        audio = padded = 0.0
        clips = 0
        for _ in sample:
            b = next(it)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = loss_fn(model, b)
            loss.backward()
            model.zero_grad(set_to_none=False)
            audio, padded = audio + b["seconds"], padded + b["padded_seconds"]
            clips += len(b["lens"]) if "lens" in b else b["wav"].shape[0]
        torch.cuda.synchronize()
        return audio, padded, clips

    model.train()
    with batchnorm_kept(model):
        run()
        monitor.window()
        t0 = time.perf_counter()
        audio, padded, clips = run()
        train_dt = time.perf_counter() - t0
        gpu = monitor.window()
        model.zero_grad(set_to_none=True)

    model.eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        val = evaluate(model)
    torch.cuda.synchronize()
    val_dt = time.perf_counter() - t0
    model.train()

    secs = lambda rs: sum(duration(r) for r in rs)  # noqa: E731
    epoch_s = secs(rows) / (audio / train_dt)
    gold_s = val_dt * secs(gold_rows) / secs(val_rows)
    rec = {
        "train_x_realtime": audio / train_dt,
        "train_ms_per_clip": 1000 * train_dt / clips,
        "padding_waste": 1 - audio / padded,
        **gpu,
        "val_s": val_dt,
        "val_ms_per_clip": 1000 * val_dt / len(val_rows),
        **{f"val_before_{k}": v for k, v in val.items()},
        "epoch_min": epoch_s / 60,
        "projected_h": (epochs * (epoch_s + val_dt) + gold_s) / 3600,
    }
    print(
        f"train: {rec['train_x_realtime']:.0f}x realtime = "
        f"{rec['train_ms_per_clip']:.0f} ms per clip "
        f"(forward+backward, {clips} clips), pad waste {rec['padding_waste']:.0%}, "
        f"GPU {rec['gpu_util']:.0f}% util, {rec['gpu_mem_gib']:.1f} GiB\n"
        f"val:   {val_dt:.0f} s for {len(val_rows)} clips = "
        f"{rec['val_ms_per_clip']:.0f} ms per clip "
        f"(batched decode); WER before training {val['wer']:.2f} ({sid(val)}), loops {val['loops']}"
        f"{retry_note(val)}\n"
        f"projected: {rec['epoch_min']:.1f} min per epoch + {val_dt / 60:.1f} min val -> "
        f"at most {rec['projected_h']:.2f} h for {epochs} epochs and gold "
        "(early stopping can cut it)",
        flush=True,
    )
    return rec


class HubResume:
    """One run's resume point in a scratch model repo on the hub: `<path>/resume.pt`.

    `train` saves it after every epoch's evaluation and reads it when it starts, so a lost
    runtime costs the epoch in progress, not the run. It holds the weights as they are, the best
    weights when they differ, the counters and the history; not the optimizer's moments, which
    are twice the model's size and rebuild within a few steps. Every save replaces the last one,
    and the blobs it supersedes are deleted from the repo's storage, which is why this lives in a
    repo of its own: nothing else can be deleted by mistake. A failed save, load or clean-up is
    reported and never stops training."""

    NAME = "resume.pt"

    def __init__(self, api: Any, repo: str, path: str, local: Path):
        self.api, self.repo, self.path, self.local = api, repo, path, Path(local)

    @property
    def remote(self) -> str:
        return f"{self.path}/{self.NAME}"

    def load(self) -> dict | None:
        try:
            if not self.api.file_exists(self.repo, self.remote):
                return None
            from huggingface_hub import hf_hub_download

            with timed(f"resuming from {self.repo}/{self.remote}"):
                path = hf_hub_download(self.repo, self.remote, token=self.api.token)
                return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # a broken resume point means starting over, not failing
            print(
                f"no usable resume point ({type(exc).__name__}: {exc}); starting over", flush=True
            )
            return None

    def save(self, state: dict) -> None:
        try:
            self.local.mkdir(parents=True, exist_ok=True)
            torch.save(state, self.local / self.NAME)
            with timed(f"saving the resume point after epoch {state['epoch']}"):
                self.api.upload_file(
                    path_or_fileobj=str(self.local / self.NAME),
                    path_in_repo=self.remote,
                    repo_id=self.repo,
                    commit_message=f"{self.path}: after epoch {state['epoch']}",
                )
            self._purge(keep_newest=True)
        except Exception as exc:
            print(f"resume point not saved ({type(exc).__name__}: {exc})", flush=True)

    def clear(self) -> None:
        """Remove the resume point once the run's weights are safely uploaded elsewhere."""
        try:
            if self.api.file_exists(self.repo, self.remote):
                self.api.delete_file(
                    self.remote, self.repo, commit_message=f"{self.path}: run finished"
                )
            self._purge(keep_newest=False)
            shutil.rmtree(self.local, ignore_errors=True)
        except Exception as exc:
            print(f"resume point not cleared ({type(exc).__name__}: {exc})", flush=True)

    def _purge(self, keep_newest: bool) -> None:
        """Delete this run's superseded blobs for good: a replaced file stays in the repo's
        history, and its storage stays counted, until it is removed from LFS as well."""
        mine = sorted(
            (f for f in self.api.list_lfs_files(self.repo) if f.filename == self.remote),
            key=lambda f: f.pushed_at,
        )
        stale = mine[:-1] if keep_newest else mine
        if stale:
            self.api.permanently_delete_lfs_files(self.repo, stale)


def train(
    model: torch.nn.Module,
    *,
    cfg: TrainConfig,
    rows: Sequence[dict],
    make_batches: Callable[[int], list[list[int]]],
    collate: Callable[[list[dict]], Any],
    loss_fn: Callable[[torch.nn.Module, Any], tuple[torch.Tensor, int]],
    evaluate: Callable[[torch.nn.Module], dict],
    save_best: Callable[[torch.nn.Module], None],
    optimizer: torch.optim.Optimizer,
    monitor: GpuMonitor,
    resume: Any | None = None,
) -> dict:
    """Train with bf16 autocast and gradient accumulation; evaluate on val after every epoch;
    keep the best weights (by val WER) in CPU memory and hand them to `save_best`. A hand stop
    (KeyboardInterrupt) after the first evaluation ends training the same way.

    `loss_fn` returns a *summed* loss and the number of units it sums over (clips for CTC, tokens
    for cross-entropy); gradients are divided by the step's total units, so accumulated
    micro-batches of different sizes are weighted exactly. Count the units on the CPU (in
    `collate`): an `int()` of a GPU tensor stalls the host once per micro-batch.

    `resume` (a `HubResume`) makes the run restartable: its state is saved after every epoch and,
    when one is found at the start, training continues from the epoch after it, on the same
    batches and at the same point of the LR schedule. The optimizer's moments start again."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    plan = [group_steps(rows, make_batches(e), cfg.effective_s) for e in range(cfg.epochs)]
    total = sum(len(p) for p in plan)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    params = [p for p in model.parameters() if p.requires_grad]
    history, best, best_state, bad, step = [], float("inf"), None, 0, 0
    counted = float("inf")  # the val WER that last reset `bad`
    first_epoch, finished = 0, False
    state = resume.load() if resume is not None else None
    if state is not None:
        model.load_state_dict(state["model"])
        best_state = state["best_model"] or state["model"]
        history, best, counted = state["history"], state["best"], state["counted"]
        bad, step, first_epoch, finished = (
            state["bad"],
            state["step"],
            state["epoch"],
            state["done"],
        )
        print(
            f"{cfg.name}: resumed after epoch {first_epoch} (best val WER {best:.2f})", flush=True
        )
    print(
        f"{cfg.name}: {total} optimizer steps over {cfg.epochs} epochs "
        f"(~{cfg.effective_s / 60:.0f} min of audio each), peak lr {cfg.lr:g}, {cfg.schedule}"
    )
    stopped_by_hand = False
    try:
        for epoch in range(first_epoch, 0 if finished else cfg.epochs):
            model.train()
            flat = [b for s in plan[epoch] for b in s]
            sizes = [len(s) for s in plan[epoch]]
            it = iter(loader(rows, flat, collate, cfg.workers))
            t_log, audio_log, padded_log, loss_log, units_log = (
                time.perf_counter(),
                0.0,
                0.0,
                0.0,
                0,
            )
            monitor.window()
            for n_micro in sizes:
                for g, lr0 in zip(optimizer.param_groups, base_lrs, strict=True):
                    g["lr"] = lr0 * lr_factor(step, total, cfg)
                step_units = 0
                for _ in range(n_micro):
                    batch = next(it)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, units = loss_fn(model, batch)
                    loss.backward()
                    step_units += units
                    loss_log += loss.detach()  # stays on the GPU: no host sync per micro-batch
                    units_log += units
                    audio_log += batch["seconds"]
                    padded_log += batch["padded_seconds"]
                torch._foreach_div_(
                    [p.grad for p in params if p.grad is not None], max(step_units, 1)
                )
                gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.log_every == 0:
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t_log
                    gpu = monitor.window()
                    rec = {
                        "step": step,
                        "epoch": epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss": float(loss_log) / max(units_log, 1),
                        "grad_norm": float(gnorm),
                        "audio_x_realtime": audio_log / dt,
                        "padding_waste": 1 - audio_log / max(padded_log, 1e-9),
                        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                        **gpu,
                    }
                    history.append(rec)
                    print(
                        f"step {step:5d}/{total} ep {epoch + 1} lr {rec['lr']:.2e} "
                        f"loss {rec['loss']:.4f} | {rec['audio_x_realtime']:6.0f}x realtime, "
                        f"pad waste {rec['padding_waste']:.0%}, "
                        f"GPU {rec['gpu_util']:.0f}% util, {rec['gpu_mem_gib']:.1f} GiB",
                        flush=True,
                    )
                    t_log, audio_log, padded_log, loss_log, units_log = (
                        time.perf_counter(),
                        0.0,
                        0.0,
                        0.0,
                        0,
                    )
            model.eval()
            with torch.no_grad():
                val = evaluate(model)
            history.append(
                {"epoch": epoch + 1, "step": step, **{f"val_{k}": v for k, v in val.items()}}
            )
            improved = val["wer"] < best
            print(
                f"== epoch {epoch + 1}: val WER {val['wer']:.2f} ({sid(val)})  "
                f"CER {val['cer']:.2f}  raw WER {val['raw_wer']:.2f}  "
                f"loops {val['loops']}{retry_note(val)}" + ("  (best)" if improved else ""),
                flush=True,
            )
            (out / "history.json").write_text(json.dumps(history, indent=1))
            current = None
            if improved or resume is not None:  # a copy on the CPU: the run's only other one
                current = {
                    k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                }
            if improved:
                best_state, best = current, val["wer"]
            counted, bad = patience_step(val["wer"], counted, bad, cfg.min_delta)
            finished = bad >= cfg.patience
            if resume is not None:
                resume.save(
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "best": best,
                        "counted": counted,
                        "bad": bad,
                        "done": finished,
                        "history": history,
                        "model": current,
                        "best_model": None if improved else best_state,
                    }
                )
            del current
            if finished:
                print(
                    f"val WER gained less than {cfg.min_delta:g} points in {cfg.patience} "
                    "evaluations; stopping"
                )
                break
    except KeyboardInterrupt:
        # Colab's stop button: end training here and keep the best weights so far, so they are
        # saved and scored like a finished run's. Before any evaluation there are none to keep.
        if best_state is None:
            raise
        stopped_by_hand = True
        print(f"stopped by hand; keeping the best weights (val WER {best:.2f})", flush=True)
    monitor.window()
    with timed("saving the best weights"):
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        save_best(model)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    return {
        "best_val_wer": best,
        "steps": step,
        "history": history,
        "stopped_by_hand": stopped_by_hand,
    }
