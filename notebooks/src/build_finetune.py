"""Build notebooks/Finetune.ipynb: `python notebooks/src/build_finetune.py`. Shared code lives in ftkit.py, written out by
a %%writefile cell in each notebook."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
FTKIT = (HERE / "ftkit.py").read_text()
CPUKIT = (HERE / "cpukit.py").read_text()
XTALK = (HERE / "xtalk.py").read_text()
SWEEPKIT = (HERE / "sweep.py").read_text()
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent


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

SETUP_COMMON = r"""
%pip install -q rapidfuzz
import json
import os
import sys
from pathlib import Path

# Before torch touches CUDA: lets the allocator grow segments instead of fragmenting when batch
# shapes vary (a fresh kernel only; the Omnilingual subprocess inherits it).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

IN_COLAB = "google.colab" in sys.modules
# Secrets and the Drive consent popup only work from a cell run in the Colab UI. When cells are
# driven from outside (the Colab MCP), run this cell by hand once. The token is then kept in the
# hub's own token file on the VM (where `hf auth login` puts it), so a kernel restart needs no
# click; the file goes when the runtime is deleted.
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
    if USE_DRIVE and not Path("/content/drive/MyDrive").exists():
        from google.colab import drive

        drive.mount("/content/drive")
FT = Path("/content/ft") if IN_COLAB else Path.cwd() / ".cache-ft"
FT.mkdir(parents=True, exist_ok=True)
OUT_ROOT = (Path("/content/drive/MyDrive/nepanglish-asr") if IN_COLAB and USE_DRIVE else FT / "out") / RUN_PREFIX
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT = OUT_ROOT  # each run points this at its own folder
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
print("sweep:", RUN_PREFIX, "|", len(SWEEP), "run(s) | outputs:", OUT_ROOT)
"""

SWEEP_REPORT = r"""
import matplotlib.pyplot as plt

trained = [r for r in sweep_rows if not r.get("reference")]
reference = next((r for r in sweep_rows if r.get("reference")), None)
winner, why = sweep.choose_winner(trained)
base = min((r for r in trained if r["xtalk_p"] == 0 and r["seed"] == 0), key=lambda r: r["val_wer"], default=None)
gold_eps = [r["episode_id"] for r in splits["gold"]]
bucket_of = [(r.get("classes") or {}).get("overlap") for r in splits["gold"]]


def counts(row, idx):
    return [dict(zip(("errors", "words"), row["per_clip"]["gold"][splits["gold"][i]["segment_id"]])) for i in idx]


def compare(a, b):
    # WER(b) - WER(a) on gold, overall and per crosstalk bucket, paired and resampled by episode
    out = {}
    for name in ("all",) + BUCKETS:
        idx = [i for i, bk in enumerate(bucket_of) if name == "all" or bk == name]
        if idx:
            out[name] = sweep.paired_bootstrap(counts(a, idx), counts(b, idx), [gold_eps[i] for i in idx])
    return out


def line(m):
    return f"{m['wer']:6.2f} ({ftkit.sid(m)})"


print("folded WER, S + D + I per 100 ref words. Winner chosen on val only (D96):", why)
print(f"\n{'run':<22}{'epoch':>6}  {'val':<34}{'gold':<34}")
for r in sweep_rows:
    tag = " <- winner" if r is winner else ""
    label = "09-17 on this export" if r.get("reference") else f"p={r['xtalk_p']:.2f} seed {r['seed']}"
    print(f"{label:<22}{str(r.get('best_epoch') or '-'):>6}  {line(r['val']):<34}{line(r['gold']):<34}{tag}")
print("\ngold by crosstalk bucket:")
for bucket in BUCKETS:
    print(f"  {bucket}  ({winner['gold']['by_overlap'].get(bucket, {}).get('clips', 0)} clips)")
    for r in sweep_rows:
        m = r["gold"]["by_overlap"].get(bucket)
        if m:
            label = "09-17" if r.get("reference") else f"p={r['xtalk_p']:.2f} s{r['seed']}"
            print(f"    {label:<12} {line(m)}")

comparisons = {}
if base:
    print("\npaired against p=0 seed 0 on gold, WER(other) - WER(p=0) [95% CI, episodes resampled]:")
    for r in sweep_rows:
        if r is base:
            continue
        label = "09-17 (old data)" if r.get("reference") else f"p={r['xtalk_p']:.2f} seed {r['seed']}"
        comparisons[r["run_name"]] = compare(base, r)
        cells = "  ".join(f"{k} {d:+.2f} [{lo:+.2f},{hi:+.2f}]" for k, (d, lo, hi) in comparisons[r["run_name"]].items())
        print(f"  {label:<18} {cells}")
    if reference:
        print("  (09-17 minus p=0 is the effect of the data added since, with the sign flipped: positive = new data helps)")

fig, axes = plt.subplots(1, len(BUCKETS) + 1, figsize=(18, 3.4), sharex=True)
for ax, name in zip(axes, ("all",) + BUCKETS):
    pick = (lambda r: r["gold"]) if name == "all" else (lambda r, n=name: r["gold"]["by_overlap"].get(n))
    for seed, style in ((0, dict(marker="o", color="#2a78d6", label="seed 0")),
                        (1, dict(marker="o", mfc="none", color="#2a78d6", linestyle="none", label="seed 1"))):
        pts = sorted((r["xtalk_p"], pick(r)["wer"]) for r in trained if r["seed"] == seed and pick(r))
        if pts:
            ax.plot(*zip(*pts), **style)
    if reference and pick(reference):
        ax.axhline(pick(reference)["wer"], color="#8a8983", linestyle="--", linewidth=1.5, label="09-17")
    ax.set_title(f"gold {name}: folded WER %", loc="left")
    ax.set_xlabel("XTALK_P")
    ax.grid(color="#ecebe7")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=3)
fig.tight_layout()
plt.show()

summary = {"sweep": RUN_PREFIX, "rule": "val WER; augmentation must beat p=0 by more than the p=0 seed gap (D96)",
           "winner": winner["run_name"], "why": why,
           "rows": [{k: v for k, v in r.items() if k != "per_clip"} for r in sweep_rows],
           "paired_vs_p0_seed0_gold": comparisons}
(OUT_ROOT / "sweep.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
print("\nwrote", OUT_ROOT / "sweep.json")
"""

SPEED_NOTE = """
## Speed check (before committing to the run)

This times forward+backward on 10 real training micro-batches, then one full batched val pass,
and projects the whole run. The val pass doubles as the **val WER before training**, the baseline
that fine-tuning has to beat. No optimizer step runs, so no weight moves. If the projection is
too long, or GPU utilisation is low, stop here and fix it.
"""

flex = [
    md(
        """
# 04c — Fine-tune Indic-Transcribe-Flex (mixed-script mode)

A full fine-tune of `bodhan-ai/indic-transcribe-flex`: a 1.2 B Canary-style encoder-decoder with
a 32-layer conformer encoder and a 24-layer transformer decoder. It trains on the train
split with the `ne` + mixed-script prompt, uses val for model selection and gold for the final
score.

**A sweep (D96).** `SWEEP` lists `(XTALK_P, seed)` points. Each is trained from the same base
weights on the same export and batch budget, scored on val and gold, and uploaded as it finishes
(metrics, transcripts, harness folder; no weights), so a dropped session resumes where it stopped.
The winner is chosen **on val alone**, by a rule fixed before any result was read: augmentation has
to beat p = 0 by more than the gap between the two p = 0 seeds, otherwise p = 0 is kept. Gold never
takes part, so every gold number stays a held-out score. Only the winner's weights are exported
for the CPU and uploaded. `REFERENCE_RUN` (the 2026-09-17 model) is decoded on this export's val
and gold under the same decoder: against p = 0 it measures the data added since. One entry in
`SWEEP` is a plain single run.

**Why this model.** It led the zero-shot comparisons at 18.2% folded WER on gold (CI 14.0–21.2),
flat across code-mixing tiers. Its mixed mode already writes the corpus's convention: 33.6% Latin
tokens against the references' 36.0%.

**What had to be built, because the release is inference-only.**
- **Loss.** The model's `forward(labels=...)` raises `NotImplementedError`. The loss here is
  cross-entropy on teacher-forced logits. The input is the fixed 10-token prompt followed by the
  target, and loss is taken on the target tokens and the end-of-text only.
- **Target encoding.** The tokenizer has no text `encode()`. Targets are the multilingual
  SentencePiece ids offset by the 1,152 special tokens, which round-trips exactly under the
  model's own `decode`.
- **Regularisation.** The port has no dropout, so SpecAugment (2 frequency masks of ≤27 bins and
  10 time masks of ≤5%, Canary's defaults) is applied to the features on the GPU.
- **Labels mapped to what the tokenizer can write.** Without this, 76% of labels would contain an
  unknown token. `।` becomes `.`, Devanagari digits become Latin, and ZWJ/ZWNJ are removed.
  `fold.py` scores each pair as identical, so WER is unaffected.

**Licence.** A fine-tuned model is a derivative under the Indic Open Model License v1.0. You can
use it privately. Giving it to anyone passes the same licence on, and hosting it as a service for
others needs Bodhan AI's written sign-off. Keep the weights private.

**Choices.** Peak LR 1e-5. Linear decay, 10% warmup, up to 6 epochs.

**Synthetic crosstalk (roadmap item 3), on when `XTALK_P > 0`.** Each epoch, a fresh `XTALK_P`
share of the train clips measured clean gets short bursts of another voice mixed over it, in the
DataLoader workers (`xtalk.py`). The label is unchanged, so the model learns to write only the
clip's own speaker. Durations and level gaps are the measured ones (findings.md, *Overlap
windows*). Bursts have a median of ~0.4 s, and the two voices are within 3 dB with either one
louder. The interrupter comes from the same episode when it has another voice, so it cannot be told
apart by room or microphone. Clips with real crosstalk are left as they are. Val and gold are never
mixed, and never used as a source. Most of the overlapped share goes to the 5–15% and >15% buckets,
where the errors are. The report gives gold WER with S/D/I per crosstalk bucket for every point.

**Every score splits into S, D and I** (`ftkit.sid`): substitutions, deletions and insertions per
100 folded reference words, which add up to the folded WER. They are printed each epoch, stored in
`history.json`, `gold_metrics.json` (overall and per crosstalk bucket), `cpu_bench.json` and the
model card, and plotted under Results. Read them together: a change can move errors between kinds
without moving WER.

**Decoding: greedy, plus a loop retry.** A clip whose greedy output repeats a 3-word sequence 5+
times is decoded again, alone, with a repetition penalty, no repeated 6-token phrase and a
length cap from its duration. The trigger reads only the model's own output (the idea of
Whisper's compression-ratio fallback), so it is usable on any audio. The settings were fixed
before scoring and never tuned; the Gold cell records the greedy-only score from the same run.
Run 2026-09-12: gold 13.20% greedy -> 11.44% with the retry (7 loops -> 0); val unchanged at 7.60%.
"""
        + GPU_NOTE
    ),
    md("## Config"),
    code(r"""
RUN_PREFIX = "flex-xtalk-sweep-2026-09-22"  # OUT_REPO/<RUN_PREFIX>/<RUN_PREFIX>-p<share>-s<seed>/ per run
MODEL_ID = "bodhan-ai/indic-transcribe-flex"
LANG, MODE = "ne", "mixed"
USE_DRIVE = False          # outputs stay on the VM and go to OUT_REPO; True also keeps them on Drive
OUT_REPO = "Sagyam/nepanglish-asr-flex-ft"  # private HF model repo
EPOCHS, LR, WARMUP = 6, 1e-5, 0.1
# (XTALK_P, seed) per run. XTALK_P is the share of measured-clean train clips given synthetic
# crosstalk each epoch (0 = off). The second p=0 seed measures run-to-run noise, which the winner
# rule needs (D96). Fixed before the sweep: do not edit it after reading results.
SWEEP = [(0.0, 0), (0.1, 0), (0.2, 0), (0.3, 0), (0.5, 0), (0.0, 1)]
REFERENCE_RUN = "indic-transcribe-flex-ft-2026-09-17"  # in OUT_REPO; decoded here for the added-data comparison, None to skip
# Shown on the harness's Models page (D83).
MODEL_NAME = "Indic-Transcribe-Flex FT"
EFFECTIVE_S = 720.0       # ~12 min of audio per optimizer step
PAD_TO_S = 1.0
PROBE_FRACTION = 0.9
GRAD_CKPT = False         # checkpoint the conformer layers if the probe finds a small batch
EVAL_BUDGET_S, EVAL_ITEMS = 1200.0, 96
"""),
    md("## Setup"),
    code(SETUP_COMMON),
    code("%%writefile /content/ft/ftkit.py\n" + FTKIT),
    code("%%writefile /content/ft/xtalk.py\n" + XTALK),
    code("%%writefile /content/ft/sweep.py\n" + SWEEPKIT),
    code(r'''
sys.path.insert(0, str(FT))
import warnings

import torch
import transformers
from huggingface_hub import snapshot_download

import ftkit

# transformers' generate() warns on every call (max_new_tokens vs max_length); it is noise here and
# buries the training log. Errors still show.
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", module="transformers")
ftkit.fast_cuda()
DATA = ftkit.download_dataset()
splits = ftkit.load_splits(DATA)
store = ftkit.AudioStore(DATA, [r["episode_id"] for rows in splits.values() for r in rows])
score = ftkit.harness_scorer(DATA, FT)
print({k: len(v) for k, v in splits.items()}, f"audio in RAM: {store.gib:.1f} GiB")

FLEX_DIR = snapshot_download(MODEL_ID)
sys.path.insert(0, FLEX_DIR)
from indic_transcribe import MODES, IndicTranscribe  # noqa: E402



def load_model(path):
    """Fresh weights on the GPU, as `model` (which transcribe and the loss read)."""
    global asr, model, featurize
    asr = IndicTranscribe.from_pretrained(str(path), device="cuda", dtype=torch.float32)
    model, featurize = asr.model, asr.fe
    if GRAD_CKPT:
        for layer in model.model.encoder.layers:
            layer._forward = layer.forward
            layer.forward = lambda *a, _l=layer, **k: torch.utils.checkpoint.checkpoint(_l._forward, *a, use_reentrant=False, **k)
    return model


import torch.utils.checkpoint  # noqa: E402

load_model(FLEX_DIR)
tk = asr.tokenizer
itn, romanized = MODES[MODE]
PROMPT = tk.encode_prompt(LANG, itn=itn, romanized=romanized)
EOS, PAD, OFFSET = tk.eos_id, tk.pad_id, tk.spl_size
UNK_MULTI = tk.multi.unk_id()

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize(text: str) -> str:
    """Map the characters the tokenizer lacks onto ones fold.py scores as identical."""
    return (text.replace("।", ".").translate(DEV_DIGITS).replace("—", "-")
            .replace("‍", "").replace("‌", ""))


for name in ("train", "val"):
    kept = []
    for r in splits[name]:
        ids = tk.multi.encode(normalize(r["text"]), out_type=int)
        if UNK_MULTI not in ids:
            r["ids"] = [OFFSET + i for i in ids] + [EOS]
            kept.append(r)
    print(f"{name}: dropped {len(splits[name]) - len(kept)} clip(s) the tokenizer cannot write")
    splits[name] = kept
if any(p for p, _ in SWEEP):
    n = ftkit.attach_speaker_turns(DATA, splits["train"])
    print(f"speaker turns for {n} of {len(splits['train'])} train clips")
r = splits["train"][0]
assert tk.decode(r["ids"][:-1]) == normalize(r["text"]).strip(), "target encoding does not round-trip"
print("prompt", PROMPT, "| longest target", max(len(r["ids"]) for r in splits["train"]), "tokens")
'''),
    md("## Model, optimizer and the batch-size probe"),
    code(r'''
# The probe runs on the base weights loaded in Setup; every sweep run reloads fresh ones and uses
# the budget measured here, so all runs see the same batches.
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, fused=True)
ftkit.init_optimizer_state(model, optimizer)


def spec_augment(feats: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Canary's defaults: 2 frequency masks of <= 27 bins, 10 time masks of <= 5% of each clip."""
    b, f, t = feats.shape
    fr = torch.arange(f, device=feats.device)[None, :]
    tr = torch.arange(t, device=feats.device)[None, :]
    mask = torch.zeros(b, f, t, dtype=torch.bool, device=feats.device)
    for _ in range(2):
        w = torch.randint(0, 28, (b, 1), device=feats.device)
        s = (torch.rand(b, 1, device=feats.device) * (f - w)).long()
        mask |= ((fr >= s) & (fr < s + w))[:, :, None]
    for _ in range(10):
        w = (torch.rand(b, 1, device=feats.device) * 0.05 * lens[:, None]).long()
        s = (torch.rand(b, 1, device=feats.device) * (lens[:, None] - w).clamp(min=1)).long()
        mask |= ((tr >= s) & (tr < s + w))[:, None, :]
    return feats.masked_fill(mask, 0.0)


def features(wav: torch.Tensor, lens: torch.Tensor):
    feats, flens = featurize(wav, lens)  # fp32 even inside autocast: it disables autocast itself
    mask = (torch.arange(feats.size(2), device=feats.device)[None, :] < flens[:, None]).long()
    return feats, flens, mask


def forward_loss(feats, mask, inp, lab):
    logits = model(input_features=feats, attention_mask=mask, decoder_input_ids=inp, use_cache=False).logits
    return torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), lab.flatten(), ignore_index=-100,
                                             reduction="sum")


longest = max(splits["train"], key=ftkit.duration)
max_len = ftkit.pad_len(round(ftkit.duration(longest) * ftkit.SR), PAD_TO_S)
max_t = len(PROMPT) + max(len(r["ids"]) for r in splits["train"]) - 1


def probe_step(n):
    wav = torch.randn(n, max_len, device="cuda") * 0.1
    feats, _, mask = features(wav, torch.full((n,), max_len, device="cuda"))
    inp = torch.randint(OFFSET, OFFSET + 6000, (n, max_t), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = forward_loss(feats, mask, inp, inp)
    loss.backward()


max_items = ftkit.probe_max_items(probe_step, 1, 256, [p for p in model.parameters() if p.requires_grad])
BUDGET_S = max_items * max_len / ftkit.SR * PROBE_FRACTION
print(f"largest micro-batch at {max_len / ftkit.SR:.0f} s x {max_t} tokens: {max_items} clips -> "
      f"budget {BUDGET_S:.0f} s of padded audio per micro-batch")
'''),
    md("## Batches, loss and decoding"),
    code(r"""
import math

import numpy as np

import xtalk

N_PROMPT = len(PROMPT)
SEED = 0
MIXER = None  # set per run
POOL = xtalk.DonorPool(splits["train"]) if any(p for p, _ in SWEEP) else None


def fetch(ep, s, e):
    return store.audio[ep][round(s * ftkit.SR):round(e * ftkit.SR)]


def train_clip(row, rng):
    clip = store.clip(row)
    return MIXER(row, clip, rng) if MIXER else (clip, None)


def collate(rows):
    # torch reseeds each DataLoader worker per epoch, so every epoch draws fresh mixes, and the
    # run is still repeatable from its seed
    rng = np.random.default_rng(torch.randint(2**62, (1,)).item())
    clips = [torch.from_numpy(train_clip(r, rng)[0].copy()) for r in rows]
    wav = torch.zeros(len(rows), ftkit.pad_len(max(len(c) for c in clips), PAD_TO_S), dtype=torch.int16)
    for i, c in enumerate(clips):
        wav[i, :len(c)] = c
    width = N_PROMPT + max(len(r["ids"]) for r in rows) - 1
    inp = torch.full((len(rows), width), PAD, dtype=torch.long)
    lab = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        full = PROMPT + r["ids"]
        inp[i, :len(full) - 1] = torch.tensor(full[:-1])
        lab[i, N_PROMPT - 1:len(full) - 1] = torch.tensor(r["ids"])  # predict target + eos only
    return {"wav": wav, "lens": torch.tensor([len(c) for c in clips]), "inp": inp, "lab": lab,
            "tokens": sum(len(r["ids"]) for r in rows),
            "seconds": sum(ftkit.duration(r) for r in rows), "padded_seconds": wav.numel() / ftkit.SR}


def loss_fn(model, b):
    wav = b["wav"].cuda(non_blocking=True).float() / 32768.0
    feats, flens, mask = features(wav, b["lens"].cuda(non_blocking=True))
    feats = spec_augment(feats, flens)
    loss = forward_loss(feats, mask, b["inp"].cuda(non_blocking=True), b["lab"].cuda(non_blocking=True))
    return loss, b["tokens"]


def transcribe(rows, **gen):
    clips = [store.clip_f32(r) for r in rows]
    wav = torch.zeros(len(rows), ftkit.pad_len(max(len(c) for c in clips), PAD_TO_S))
    for i, c in enumerate(clips):
        wav[i, :len(c)] = c
    feats, _, mask = features(wav.cuda(), torch.tensor([len(c) for c in clips], device="cuda"))
    prompt = torch.tensor([PROMPT] * len(rows), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(input_features=feats, attention_mask=mask, decoder_input_ids=prompt,
                             do_sample=False, num_beams=1, eos_token_id=EOS, pad_token_id=PAD,
                             **{"max_new_tokens": 300, **gen})
    return [tk.decode(tk.strip_prompt_and_trim(o.tolist(), PROMPT)) for o in out]


# Loop retry, fixed before scoring anything with it: greedy decoding occasionally sticks on a filler
# ("अँ. अँ. अँ.") to the length limit. A global repetition ban would also hit real repeats the
# references keep ("कोही छैन। कोही छैन"), so only a clip whose greedy output loops is decoded
# again, alone, with a repetition penalty, no repeated 6-token phrase, and a length cap from its
# duration (the densest training label is 12.6 tokens/s).
MAX_TOKENS_PER_S = 13.0
RETRY = {"repetition_penalty": 1.2, "no_repeat_ngram_size": 6}


def retry_one(row):
    cap = min(300, math.ceil(MAX_TOKENS_PER_S * ftkit.duration(row)))
    return transcribe([row], max_new_tokens=cap, **RETRY)[0]


def evaluate(model, rows=None):
    rows = rows or splits["val"]
    texts, _ = ftkit.transcribe_rows(rows, ftkit.RetryLoops(transcribe, retry_one), budget_s=EVAL_BUDGET_S,
                                     max_items=EVAL_ITEMS, pad_to_s=PAD_TO_S)
    return score([r["text"] for r in rows], texts)


def save_best(model):
    import shutil

    dst = OUT / "best"
    model.save_pretrained(dst, state_dict={k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
    for f in Path(FLEX_DIR).iterdir():  # code, tokenizer and feature files, so IndicTranscribe loads it
        if f.suffix in {".py", ".model", ".md"} or f.name in {"tokenizer_config.json", "generation_config.json",
                                                             "feature_extractor.safetensors"}:
            shutil.copy(f, dst / f.name)


def make_batches(epoch):
    # seed 0 keeps the order every earlier run used; another seed also reorders the batches
    return ftkit.bucket_batches(splits["train"], budget_s=BUDGET_S, max_items=256, pad_to_s=PAD_TO_S,
                                shuffle=True, seed=epoch + 1000 * SEED)


monitor = ftkit.GpuMonitor().start()
"""),
    md("""
## Synthetic crosstalk: check it before training

What the mixer does to 2,000 train clips drawn at random, against what it was built to do.
- **Windows.** Median ~0.3 s: light targets redraw long bursts, so this is a little under the
  measured 0.42 s.
- **Level gaps.** Within ±3 dB.
- **Share of each mixed clip.** Spread over the <5%, 5–15% and >15% buckets.
- **Same episode.** How often the other voice came from the clip's own episode. Single-voice
  episodes have to borrow one from another show.

Six examples go to `OUT_ROOT/xtalk_examples/`, each saved clean and mixed, and three play below.
**Listen to them before starting the run.** If the other voice is inaudible, or clearly from
another room, stop and fix the mixer.
"""),
    code(r"""
import soundfile as sf
from IPython.display import Audio, display

if POOL:
    rng = np.random.default_rng(0)
    eligible = [r for r in splits["train"] if r["overlap_spans"] == [] and r.get("speaker_turns")]
    always = xtalk.Mixer(POOL, fetch, p=1.0)
    infos = [always(r, store.clip(r), rng)[1] for r in rng.choice(eligible, 2000, replace=False)]
    infos = [i for i in infos if i]
    wins = [w for i in infos for w in i["windows"]]
    shares = np.array([i["share"] for i in infos])
    stats = {
        "pool_stretches": len(MIXER.pool), "pool_voices": len(MIXER.pool.voices),
        "eligible_clips": len(eligible), "train_clips": len(splits["train"]),
        "mixed_of_2000": len(infos),
        "window_s_median": float(np.median([w["seconds"] for w in wins])),
        "gap_db_abs_median": float(np.median([abs(w["gap_db"]) for w in wins])),
        "same_episode": float(np.mean([w["same_episode"] for w in wins])),
        "share_buckets": {"<5%": float(np.mean(shares < 0.05)),
                          "5-15%": float(np.mean((shares >= 0.05) & (shares <= 0.15))),
                          ">15%": float(np.mean(shares > 0.15))},
    }
    (OUT_ROOT / "xtalk_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    ex = OUT_ROOT / "xtalk_examples"
    ex.mkdir(exist_ok=True)
    picks = [r for r in rng.choice(eligible, 60, replace=False) if 6 <= ftkit.duration(r) <= 15][:6]
    for k, r in enumerate(picks):
        clean = store.clip(r)
        mixed, info = always(r, clean, rng)
        sf.write(ex / f"{k}_clean.flac", clean, ftkit.SR)
        sf.write(ex / f"{k}_mixed.flac", mixed, ftkit.SR)
        (ex / f"{k}.json").write_text(json.dumps({**(info or {}), "text": r["text"]}, ensure_ascii=False, indent=1))
        if k < 3:
            print(f"\n{r['segment_id']}: {r['text']}")
            print("  windows:", [(round(w["offset"], 2), round(w["seconds"], 2), round(w["gap_db"], 1),
                                  "same ep" if w["same_episode"] else "other ep") for w in (info or {}).get("windows", [])])
            display(Audio(mixed, rate=ftkit.SR))
else:
    print("every XTALK_P in SWEEP is 0: no synthetic crosstalk")
"""),
    md("""
## One clip per call vs batched

The zero-shot comparisons decoded one clip per call (`asr(path)`), at about 1.4 s per clip
(RTF 0.12). Here the same 16 val clips are decoded both ways, under the same bf16 autocast, so the
only difference is batching. The cell reports the speed-up and whether the texts agree; the port's
batched `generate` had never been run before.
"""),
    code(r"""
import time

sample = splits["val"][:16]
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    transcribe(sample)  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    single = [asr.transcribe(store.clip_f32(r), lang=LANG, mode=MODE, max_new_tokens=300) for r in sample]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    batched = transcribe(sample)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
model.train()
same = sum(a.strip() == b.strip() for a, b in zip(single, batched))
print(f"one clip per call: {(t1 - t0) / len(sample):.2f} s per clip | batched: {1000 * (t2 - t1) / len(sample):.0f} ms "
      f"per clip ({(t1 - t0) / (t2 - t1):.0f}x) | identical text on {same}/{len(sample)} clips, folded WER of "
      f"batched against one-per-call {score(single, batched)['wer']:.2f}%")
"""),
    md(SPEED_NOTE),
    code(r"""
speed = ftkit.speed_check(model, rows=splits["train"], batches=make_batches(0), collate=collate,
                          loss_fn=loss_fn, evaluate=evaluate, val_rows=splits["val"], gold_rows=splits["gold"],
                          epochs=EPOCHS, monitor=monitor)
(OUT_ROOT / "speed_check.json").write_text(json.dumps(speed, indent=1))
print(f"sweep: {len(SWEEP)} run(s) -> about {len(SWEEP) * speed['projected_h']:.1f} h of training and gold, "
      "plus val decoding per run and the reference")
"""),
    md("""
## Sweep: train, score and upload every point

For each `(XTALK_P, seed)` in `SWEEP`: fresh base weights, train with early stopping on val, then
decode gold and val with the best weights under the same decoder, and score them overall and per
crosstalk bucket, with S/D/I. Each run's folder goes to `OUT_REPO/<RUN_PREFIX>/<run>/` as soon as it
finishes: metrics, `history.json`, transcripts, per-clip counts and the harness folder (D83), but
not `best/`. The weights stay on the VM until the winner is known. A run already uploaded is
skipped, so after a dropped session, run the notebook again with the same `RUN_PREFIX`. If the
winner then turns out to be a run whose weights were on the old VM, the export cell says so, and
that point has to be trained again.

The harness scores text against its *current* labels, so a clip relabeled since this export is
scored against the new label on the Models page.
"""),
    code(r"""
import datetime as dt
import gc

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

import sweep

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(OUT_REPO, repo_type="model", private=True, exist_ok=True)
export = json.loads((DATA / "training" / "manifest.json").read_text())
BUCKETS = ("none", "0-5%", "5-15%", ">15%")


def free_model():
    for name in ("optimizer", "model", "asr"):
        globals().pop(name, None)
    gc.collect()
    torch.cuda.empty_cache()


def decode(rows):
    d = ftkit.RetryLoops(transcribe, retry_one)
    texts, compute = ftkit.transcribe_rows(rows, d, budget_s=EVAL_BUDGET_S, max_items=EVAL_ITEMS, pad_to_s=PAD_TO_S)
    return texts, compute, d.log


def score_split(rows, texts, compute, log):
    refs = [r["text"] for r in rows]
    m = score(refs, texts)
    m["rtf"] = sum(compute) / sum(ftkit.duration(r) for r in rows)
    first = {sid: text for sid, text, _ in log}  # the same run's greedy output, before any retry
    m["greedy_only"] = score(refs, [first.get(r["segment_id"], t) for r, t in zip(rows, texts)])
    m["retried"] = [{"segment_id": s, "first": f, "retry": t} for s, f, t in log]
    m["by_overlap"] = {}
    for bucket in BUCKETS:
        idx = [i for i, r in enumerate(rows) if (r.get("classes") or {}).get("overlap") == bucket]
        if idx:
            m["by_overlap"][bucket] = score([refs[i] for i in idx], [texts[i] for i in idx])
    per = score.per_clip(refs, texts)
    return m, {r["segment_id"]: [c["errors"], c["words"]] for r, c in zip(rows, per)}


def score_and_write(out, run, *, p, seed, history, description, reference=False):
    # decode gold and val with the weights in `model`, write every result file, return the sweep row
    model.eval()
    results, per_clip = {}, {}
    for name in ("gold", "val"):
        texts, compute, log = decode(splits[name])
        results[name], per_clip[name] = score_split(splits[name], texts, compute, log)
        ftkit.write_hyps(out / "harness" / f"{name}.jsonl", splits[name], texts, compute)
        if name == "gold":  # the bake-off's cache format, to score beside the zero-shot systems
            ftkit.write_hyps(out / "hyps" / f"{run}.jsonl", splits[name], texts, compute)
        (out / f"{name}_metrics.json").write_text(json.dumps(results[name], indent=1, ensure_ascii=False))
    evals = [h for h in history if "val_wer" in h]
    gold, val = results["gold"], results["val"]
    card = {
        "name": MODEL_NAME if not reference else f"{MODEL_NAME} ({REFERENCE_RUN})",
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "description": description,
        "architecture": "Canary-style enc-dec: 32L conformer + 24L transformer decoder, 1.2B",
        "base_model": MODEL_ID,
        "decoder": "greedy+retry",
        "run_name": run,
        "sweep": RUN_PREFIX,
        "epochs": EPOCHS,
        "best_epoch": min(evals, key=lambda h: h["val_wer"])["epoch"] if evals else None,
        "lr": LR,
        "xtalk_p": p,
        "seed": seed,
        "val_wer": val["wer"],
        "gold_wer": gold["wer"],
        # folded, per 100 reference words; S + D + I = WER
        "val_sid": {k: val[k] for k in ("sub", "del", "ins")},
        "gold_sid": {k: gold[k] for k in ("sub", "del", "ins")},
        "train_export": {k: export.get(k) for k in ("exported_at", "git_commit", "row_count",
                                                    "normalization_version", "label_version")},
    }
    (out / "harness" / "model_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False))
    keep = ("wer", "raw_wer", "cer", "sub", "del", "ins", "loops", "clips", "by_overlap")
    row = {"run_name": run, "xtalk_p": p, "seed": seed, "reference": reference,
           "best_epoch": card["best_epoch"], "val_wer": val["wer"],
           "val": {k: val[k] for k in keep}, "gold": {k: gold[k] for k in keep}, "per_clip": per_clip}
    (out / "sweep_row.json").write_text(json.dumps(row, ensure_ascii=False))
    print(f"{run}: val {val['wer']:.2f} ({ftkit.sid(val)}) | gold {gold['wer']:.2f} ({ftkit.sid(gold)})")
    for bucket, m in gold["by_overlap"].items():
        print(f"  gold {bucket:>6}: {m['clips']:4d} clips  WER {m['wer']:6.2f}  {ftkit.sid(m)}")
    return row


def uploaded_row(run):
    remote = f"{RUN_PREFIX}/{run}/sweep_row.json"
    if not api.file_exists(OUT_REPO, remote):
        return None
    return json.loads(Path(hf_hub_download(OUT_REPO, remote, token=os.environ["HF_TOKEN"])).read_text())


def upload_run(out, run, message):
    api.upload_folder(repo_id=OUT_REPO, folder_path=str(out), path_in_repo=f"{RUN_PREFIX}/{run}",
                      ignore_patterns=["best/*"], commit_message=message)


free_model()  # the probe's weights and optimizer
sweep_rows = []
for p, seed in SWEEP:
    run = sweep.run_name(RUN_PREFIX, p, seed)
    if (row := uploaded_row(run)) is not None:
        print(f"{run}: already in {OUT_REPO}, skipped")
        sweep_rows.append(row)
        continue
    OUT = OUT_ROOT / run
    OUT.mkdir(parents=True, exist_ok=True)
    SEED = seed
    MIXER = xtalk.Mixer(POOL, fetch, p=p) if p else None
    load_model(FLEX_DIR).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, fused=True)
    ftkit.init_optimizer_state(model, optimizer)
    cfg = ftkit.TrainConfig(name=run, out=str(OUT), epochs=EPOCHS, lr=LR, warmup_frac=WARMUP,
                            effective_s=EFFECTIVE_S, patience=2, seed=seed)
    print(f"\n=== {run}: XTALK_P={p}, seed {seed}")
    result = ftkit.train(model, cfg=cfg, rows=splits["train"], make_batches=make_batches, collate=collate,
                         loss_fn=loss_fn, evaluate=evaluate, save_best=save_best, optimizer=optimizer,
                         monitor=monitor)
    description = f"04c sweep {RUN_PREFIX}: synthetic crosstalk p={p}, seed {seed}"
    sweep_rows.append(score_and_write(OUT, run, p=p, seed=seed, history=result["history"], description=description))
    upload_run(OUT, run, f"{run}: {description}")
    free_model()
MIXER, SEED = None, 0
"""),
    md("""
## Reference: the 2026-09-17 model on this export

The same decoder on this export's val and gold, so it pairs clip for clip with the sweep. Against
p = 0 seed 0 it is the effect of the data added since 09-17. The val set also grew with the new
episodes, which changes which epoch early stopping picks, so read the gold comparison.
"""),
    code(r"""
if REFERENCE_RUN:
    ref_run = f"{RUN_PREFIX}-reference"
    if (row := uploaded_row(ref_run)) is None:
        OUT = OUT_ROOT / ref_run
        OUT.mkdir(parents=True, exist_ok=True)
        ref_dir = snapshot_download(OUT_REPO, allow_patterns=[f"{REFERENCE_RUN}/best/*"], token=os.environ["HF_TOKEN"])
        load_model(Path(ref_dir) / REFERENCE_RUN / "best")
        row = score_and_write(OUT, ref_run, p=None, seed=None, history=[], reference=True,
                              description=f"{REFERENCE_RUN}, decoded on the {export.get('exported_at', '')[:10]} export")
        upload_run(OUT, ref_run, f"{ref_run}: {REFERENCE_RUN} on this export")
        free_model()
    sweep_rows.append(row)
"""),
    md("""
## Sweep report

Every point on val and gold, with S/D/I, then gold per crosstalk bucket, then each run paired
against p = 0 seed 0 by an episode bootstrap (`sweep.paired_bootstrap`). The seed 0/seed 1 gap at
p = 0 is the noise every other difference has to clear, and the winner rule uses it. The gold
columns are held-out scores: the winner was chosen on val.
"""),
    code(SWEEP_REPORT),
    md("""
## Winner: reload its weights

The CPU export below and the final upload work on this run only.
"""),
    code(r"""
run = winner["run_name"]
OUT, HARNESS = OUT_ROOT / run, OUT_ROOT / run / "harness"
if not (OUT / "best").exists():
    raise RuntimeError(f"{run} won, but its weights are not on this VM (it ran in an earlier session). "
                       f"Train that point again: SWEEP = [({winner['xtalk_p']}, {winner['seed']})] with a new "
                       "RUN_PREFIX, or delete its folder in OUT_REPO and rerun.")
load_model(OUT / "best").eval()


def read_texts(path):
    return [json.loads(line)["text"] for line in path.open(encoding="utf-8")]


texts, val_texts = read_texts(HARNESS / "gold.jsonl"), read_texts(HARNESS / "val.jsonl")
card = json.loads((HARNESS / "model_card.json").read_text())
card["sweep_winner"] = why
(HARNESS / "model_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False))
print("winner:", run, "|", why)
"""),
    md("""
## CPU export: quantize, benchmark, export

The target is a CPU at a desk, not this GPU: the harness's mic playground (D85).
**What was measured locally (Ryzen 7 7700X, 8 threads, 2026-09-15)**, base Flex on 10 gold clips:
- fp32 runs at RTF 0.41 and bf16 at RTF 0.18, with the same WER (21.8%). bf16 is realtime and
  free, and `best/` already stores bf16, so it is the CPU model if nothing else passes.
- PyTorch's **dynamic int8** (activations quantized too) broke the model: WER 97-113%, loops, and
  English written in Devanagari. It is not tried here.
- **Weight-only int8** (`cpukit.py`: int8 weights with one scale per output channel, bf16
  activations) decodes 1.5x faster than bf16 on a full-size Flex (16.5 vs 24.5 ms/token, random
  weights). Decoding is memory-bound, so halving the weight bytes is where CPU speed comes from.

**What this section measures** is the part that was not measured locally: what int8 costs in
accuracy on *this* model.
1. **Accuracy, on the GPU.** The trained model is quantized in place (this is its last use) and
   val and gold are decoded with the same decoder as above. The int8 arithmetic is dequantized on
   the GPU, so the numbers carry over to the CPU kernel.
2. **Rule, fixed before any run:** export int8 if its val WER is within **+0.3** of bf16 (the
   run-to-run noise) and it loops no more often. Otherwise the CPU model is `best/` in bf16.
3. **Speed, on this VM's CPU.** The same `CPU_TIMING_CLIPS` val clips are timed for each variant.
   Colab's CPU is not your machine (it may lack bf16 instructions entirely), so only the ratio
   between variants means anything. Measure the absolute speed where the model will run.
4. **Export.** An accepted int8 model goes to `OUT/cpu/` (~1.3 GB against 2.5 GB, with its code
   and `cpukit.py`; load it with `cpukit.load_int8`). `cpu_bench.json` holds every number, and the
   harness model card gets a `cpu` block.

ONNX is not needed: plain PyTorch bf16 is already realtime on the target CPU.
"""),
    code("%%writefile /content/ft/cpukit.py\n" + CPUKIT),
    code(r"""
import cpukit

CPU_TIMING_CLIPS = 8   # val clips timed on this VM's CPU; accuracy uses all of val and gold on the GPU
MAX_WER_COST = 0.3     # points of val WER int8 may cost against bf16 (the run-to-run noise)

val_refs, gold_refs = [r["text"] for r in splits["val"]], [r["text"] for r in splits["gold"]]
bf16_scores = {"val": score(val_refs, val_texts), "gold": score(gold_refs, texts)}
cpukit.quantize_(model, dtype=torch.float32)  # in place: the GPU copy is not needed after this
int8_texts = {}
for name, rows in (("val", splits["val"]), ("gold", splits["gold"])):
    int8_texts[name], _ = ftkit.transcribe_rows(rows, ftkit.RetryLoops(transcribe, retry_one),
                                                budget_s=EVAL_BUDGET_S, max_items=EVAL_ITEMS, pad_to_s=PAD_TO_S)
    ftkit.write_hyps(OUT / "int8" / f"{name}.jsonl", rows, int8_texts[name], [0.0] * len(rows))
int8_scores = {"val": score(val_refs, int8_texts["val"]), "gold": score(gold_refs, int8_texts["gold"])}
same = sum(a == b for a, b in zip(val_texts, int8_texts["val"]))
accept = (int8_scores["val"]["wer"] <= bf16_scores["val"]["wer"] + MAX_WER_COST
          and int8_scores["val"]["loops"] <= bf16_scores["val"]["loops"])
for name in ("val", "gold"):
    print(f"{name}: bf16 {bf16_scores[name]['wer']:.2f}% ({ftkit.sid(bf16_scores[name])}, "
          f"{bf16_scores[name]['loops']} loops) | int8 {int8_scores[name]['wer']:.2f}% "
          f"({ftkit.sid(int8_scores[name])}, {int8_scores[name]['loops']} loops)")
print(f"int8 text identical to bf16 on {same}/{len(val_texts)} val clips ->",
      "EXPORT int8" if accept else "int8 rejected; the CPU model is best/ in bf16")
"""),
    code(r"""
import subprocess

step = max(1, len(splits["val"]) // CPU_TIMING_CLIPS)
timing_rows = sorted(splits["val"], key=ftkit.duration)[::step][:CPU_TIMING_CLIPS]
clips = [(r["segment_id"], torch.as_tensor(store.clip_f32(r)).float().cpu()) for r in timing_rows]
threads = os.cpu_count()
torch.set_num_threads(threads)
cpu_name = subprocess.run("lscpu | sed -n 's/^Model name: *//p'", shell=True, capture_output=True,
                          text=True).stdout.strip()
cpu_flags = open("/proc/cpuinfo").read()
host = {"cpu": cpu_name, "threads": threads, "avx512_bf16": "avx512_bf16" in cpu_flags,
        "amx_bf16": "amx_bf16" in cpu_flags}
print(host)

timing = {}
cpu_asr = IndicTranscribe.from_pretrained(str(OUT / "best"), device="cpu", dtype=torch.bfloat16)
timing["bf16"] = cpukit.time_clips(cpu_asr, clips, lang=LANG, mode=MODE)
if accept:
    names = cpukit.quantize_(cpu_asr.model)
    cpukit.save_int8(cpu_asr.model, names, OUT / "best", OUT / "cpu")
    del cpu_asr
    cpu_asr = cpukit.load_int8(OUT / "cpu")  # time what was written, not what is in memory
    timing["int8"] = cpukit.time_clips(cpu_asr, clips, lang=LANG, mode=MODE)
del cpu_asr
for variant, t in timing.items():
    print(f"{variant}: RTF {t['rtf']:.3f}, {t['ms_per_token']:.1f} ms/token over {t['audio_s']} s of audio")

bench = {
    "variant": "int8-weight-only" if accept else "bf16",
    "export": "cpu/" if accept else "best/",
    "rule": f"int8 if val WER <= bf16 + {MAX_WER_COST} and no more loops",
    "scores": {"bf16": bf16_scores, "int8": int8_scores},
    "int8_identical_val_texts": same,
    "timing_host": host,
    "timing": {v: {k: x for k, x in t.items() if k != "texts"} for v, t in timing.items()},
}
(OUT / "cpu_bench.json").write_text(json.dumps(bench, indent=1))
card = json.loads((HARNESS / "model_card.json").read_text())
card["cpu"] = {
    "variant": bench["variant"],
    "export": bench["export"],
    "val_wer_bf16": bf16_scores["val"]["wer"],
    "val_wer_int8": int8_scores["val"]["wer"],
    "timing_host": host["cpu"],
    "rtf": {v: t["rtf"] for v, t in timing.items()},
}
(HARNESS / "model_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False))
print("wrote", OUT / "cpu_bench.json", "and the card's cpu block")
"""),
    md("""
## Upload the winner and the sweep summary

The winner's whole folder goes to `OUT_REPO/<RUN_PREFIX>/<winner>/`: `best/` (bf16), `cpu/` (when
int8 was accepted), `int8/`, `cpu_bench.json` and the card with its `cpu` block. The other runs are
already there without weights. `sweep.json`, the speed check and the crosstalk examples go to
`OUT_REPO/<RUN_PREFIX>/`. The VM can be deleted once this prints a commit.

**The token must be able to write.** The cell reads the `HF_TOKEN` secret again, because Setup
cached whatever token was there when it ran, so a secret swapped mid-session is picked up.

Then, in the harness checkout, every run (and the reference) can go on the Models page for error
mining; only the winner has `cpu/` for the mic playground:
```bash
hf download Sagyam/nepanglish-asr-flex-ft --include "<RUN_PREFIX>/*" --exclude "*/best/*" --local-dir exports/<RUN_PREFIX>
for d in exports/<RUN_PREFIX>/<RUN_PREFIX>/<RUN_PREFIX>-*/; do
  run=$(basename "$d"); mkdir -p data/models/asr/$run && cp "$d"harness/* data/models/asr/$run/
done
cp -r exports/<RUN_PREFIX>/<RUN_PREFIX>/<winner>/cpu data/models/asr/<winner>/   # mic playground (D85)
```
Press **Rescan** on the Models page. The playground sidecar starts with `docker compose up -d`.
"""),
    code(r"""
if IN_COLAB:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    (Path.home() / ".cache" / "huggingface" / "token").write_text(os.environ["HF_TOKEN"])
api = HfApi(token=os.environ["HF_TOKEN"])
commit = api.upload_folder(repo_id=OUT_REPO, folder_path=str(OUT), path_in_repo=f"{RUN_PREFIX}/{run}",
                           commit_message=f"{run}: sweep winner ({why})")
api.upload_folder(repo_id=OUT_REPO, folder_path=str(OUT_ROOT), path_in_repo=RUN_PREFIX,
                  allow_patterns=["sweep.json", "speed_check.json", "xtalk_stats.json", "xtalk_examples/*"],
                  commit_message=f"{RUN_PREFIX}: sweep summary")
size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
print(f"{size / 2**30:.2f} GiB -> https://huggingface.co/{OUT_REPO}/tree/main/{RUN_PREFIX}/{run} @ {commit.oid[:7]}")
"""),
]


for name, cells in [("Finetune.ipynb", flex)]:
    (OUT_DIR / name).write_text(
        json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote", OUT_DIR / name, len(cells), "cells")
