"""Build notebooks/04a, 04b and 04c: `python notebooks/src/build_finetune.py`. Shared code lives in ftkit.py, written out by
a %%writefile cell in each notebook so every notebook and the Omnilingual subprocess use it."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
FTKIT = (HERE / "ftkit.py").read_text()
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
OUT = (Path("/content/drive/MyDrive/nepanglish-asr") if IN_COLAB and USE_DRIVE else FT / "out") / RUN_NAME
OUT.mkdir(parents=True, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
print("run:", RUN_NAME, "| outputs:", OUT)
"""

RESULTS = r"""
import json

import matplotlib.pyplot as plt

hist = json.loads((OUT / "history.json").read_text())
steps = [h for h in hist if "loss" in h]
evals = [h for h in hist if "val_wer" in h]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
axes[0].plot([h["step"] for h in steps], [h["loss"] for h in steps], color="#2a78d6")
axes[0].set_title("train loss per unit", loc="left")
axes[1].plot([h["epoch"] for h in evals], [h["val_wer"] for h in evals], marker="o", color="#2a78d6")
axes[1].set_title("val folded WER %", loc="left")
axes[2].plot([h["step"] for h in steps], [h["gpu_util"] for h in steps], color="#2a78d6")
axes[2].set_ylim(0, 100)
axes[2].set_title("GPU utilisation %", loc="left")
for ax in axes:
    ax.grid(color="#ecebe7")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
print(json.dumps(json.loads((OUT / "gold_metrics.json").read_text()), indent=1))
"""

SPEED_NOTE = """
## Speed check (before committing to the run)

This times forward+backward on 10 real training micro-batches, then one full batched val pass,
and projects the whole run. The val pass doubles as the **val WER before training**, the baseline
that fine-tuning has to beat. No optimizer step runs, so no weight moves. If the projection is
too long, or GPU utilisation is low, stop here and fix it.
"""

BAKEOFF_LINK = """
The gold hypotheses go to `OUT/hyps/<RUN_NAME>.jsonl`, in the bake-off's cache format, and are also
copied to `/content/bakeoff/hyps/` when that folder exists. Re-running the bake-off's **Score** cell
then puts the fine-tuned model in the same table as the zero-shot ones.
"""


# =============================================================================================
# 04a Whisper-large-v3-turbo
# =============================================================================================

whisper = [
    md(
        """
# 04a — Fine-tune Whisper-large-v3-turbo

A full fine-tune of `openai/whisper-large-v3-turbo` (809 M parameters) on the 18.3 h train split.
Model selection uses val WER (403 clips); the final score is on gold (706 clips) with the bake-off's
scorer.

**Why it needs fine-tuning.** Zero-shot on gold it scored 123% folded WER with the `ne` prompt and
121% with `hi`. It fell into repetition loops on 271 of 706 clips and wrote English words in
Devanagari (17% Latin tokens against the references' 36%). Fine-tuning has to teach the corpus's
script convention (Nepali in Devanagari, English in Latin) and stop the loops.

**Choices, and where they come from:**
- **Peak LR 1e-5, linear decay, 10% warmup, 5 epochs** — the Nepali Whisper benchmark (arXiv 2608.12327).
- **Effective batch ≈ 64 clips.**
- **Language token `ne`.** Both prompts were equally broken zero-shot, `ne` is the true language,
  and fine-tuning redefines what it means.
- **SpecAugment on.** Whisper's own time masking, p = 0.05.
- **One label dropped.** One training label exceeds the 448-token decoder window.

**Decoding: greedy, plus the loop retry from 04c.** A clip whose greedy output repeats a 3-word
sequence 5+ times is decoded again, alone, with repetition penalty 1.2, no repeated 6-token phrase
and a length cap from its duration. The rule and settings are 04c's, unchanged, so the two models
are compared under the same decoder. The cap is the one Whisper-specific number: the densest train
label in Whisper tokens per second, computed from train. Whisper's own temperature fallback was
not used, because it samples and so makes a score irreproducible. Val (model selection) uses the
retry; the Gold cell records the greedy-only score from the same run.

**Run 2026-09-13 (A100 40 GB).** Val 55.24% before training (113.19% greedy; 150 of 403 clips
retried) -> 8.98% at epoch 5, the best and last epoch. Gold **14.62%** folded WER, CER 8.68%, 0
loops, so the retry never fired and greedy is identical. Flex (04c) scored 11.44% on gold.

**Run on an A100 runtime.** Before the first run, add the `HF_TOKEN` Colab secret. Nothing here
needs Python 3.12; it runs in the default kernel.
"""
        + GPU_NOTE
    ),
    md("## Config"),
    code(r"""
RUN_NAME = "whisper-turbo-ft"
MODEL_ID = "openai/whisper-large-v3-turbo"
LANG = "ne"
USE_DRIVE = True          # best weights + history survive a disconnect; set False to keep in /content
EPOCHS, LR, WARMUP = 5, 1e-5, 0.1
EFFECTIVE_S = 64 * 12.0   # ~64 clips of the 12 s median per optimizer step
PROBE_FRACTION = 0.9      # train at 90% of the largest micro-batch the probe found
GRAD_CKPT = False         # trade ~30% speed for ~3x batch if the probe finds < 16 clips
EVAL_BATCH = 64
"""),
    md("## Setup"),
    code(SETUP_COMMON),
    code("%%writefile /content/ft/ftkit.py\n" + FTKIT),
    code(r"""
sys.path.insert(0, str(FT))
import warnings

import numpy as np
import torch
import transformers
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizerFast

import ftkit

# transformers' generate() and Whisper warn on every call (max_new_tokens vs max_length, deprecated
# generation arguments); they are noise here and bury the training log. Errors still show.
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", module="transformers")
ftkit.fast_cuda()
DATA = ftkit.download_dataset()
splits = ftkit.load_splits(DATA)
store = ftkit.AudioStore(DATA, [r["episode_id"] for rows in splits.values() for r in rows])
score = ftkit.harness_scorer(DATA, FT)
print({k: len(v) for k, v in splits.items()}, f"audio in RAM: {store.gib:.1f} GiB")

tok = WhisperTokenizerFast.from_pretrained(MODEL_ID)
tok.set_prefix_tokens(language=LANG, task="transcribe", predict_timestamps=False)
fe = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
MAX_LABEL = 448 - 1  # the decoder window, less the start token the model prepends
for name in ("train", "val"):
    kept = []
    for r in splits[name]:
        r["ids"] = tok(r["text"]).input_ids[1:]  # drop <|startoftranscript|>; the model adds it back
        if len(r["ids"]) <= MAX_LABEL:
            kept.append(r)
    print(f"{name}: dropped {len(splits[name]) - len(kept)} label(s) longer than the decoder window")
    splits[name] = kept
"""),
    md("""
## Log-mel on the GPU

Whisper's feature extractor runs its STFT on the CPU, one clip at a time. This is the same
computation (the same filters, clamp, per-clip max − 8 and scaling) done batched on the GPU. The
assert checks it against the reference extractor before any training.
"""),
    code(r'''
N_SAMPLES = 30 * ftkit.SR
WINDOW = torch.hann_window(fe.n_fft, device="cuda")
MEL = torch.from_numpy(np.asarray(fe.mel_filters)).to("cuda", torch.float32)  # (n_freq, n_mels)


@torch.autocast("cuda", enabled=False)  # fp32 always: training calls this inside bf16 autocast
def log_mel(wav: torch.Tensor) -> torch.Tensor:
    """(B, 480000) float32 on the GPU -> (B, 128, 3000), identical to WhisperFeatureExtractor."""
    spec = torch.stft(wav, fe.n_fft, fe.hop_length, window=WINDOW, return_complex=True)
    power = spec[..., :-1].abs() ** 2
    logs = torch.clamp(MEL.T @ power, min=1e-10).log10()
    logs = torch.maximum(logs, logs.amax(dim=(1, 2), keepdim=True) - 8.0)
    return (logs + 4.0) / 4.0


check = splits["val"][:4]
ref = fe([store.clip_f32(r).numpy() for r in check], sampling_rate=ftkit.SR, return_tensors="pt").input_features
wav = torch.zeros(len(check), N_SAMPLES)
for i, r in enumerate(check):
    c = store.clip_f32(r)
    wav[i, :len(c)] = c
ours = log_mel(wav.cuda()).cpu()
print("max |ours - reference| =", float((ours - ref).abs().max()))
assert torch.allclose(ours, ref, atol=2e-3), "GPU log-mel does not match the reference extractor"
'''),
    md("## Model, optimizer and the batch-size probe"),
    code(r"""
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.float32,
                                                        attn_implementation="sdpa").cuda()
model.config.apply_spec_augment = True
model.config.mask_time_prob = 0.05
model.config.use_cache = False
model.generation_config.language, model.generation_config.task = LANG, "transcribe"
if GRAD_CKPT:
    model.gradient_checkpointing_enable()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, fused=True)
ftkit.init_optimizer_state(model, optimizer)

longest_label = max(len(r["ids"]) for r in splits["train"])


def probe_step(n: int) -> None:
    mel = log_mel(torch.randn(n, N_SAMPLES, device="cuda") * 0.1)
    labels = torch.randint(0, 50000, (n, longest_label), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_features=mel, labels=labels).loss.backward()


model.train()
max_items = ftkit.probe_max_items(probe_step, 1, 256, [p for p in model.parameters() if p.requires_grad])
MICRO = max(1, int(max_items * PROBE_FRACTION))
print(f"largest micro-batch at 30 s x {longest_label} tokens: {max_items} clips -> training with {MICRO}")
"""),
    md("## Batches, loss and decoding"),
    code(r"""
import math


def collate(rows):
    wav = torch.zeros(len(rows), N_SAMPLES, dtype=torch.int16)
    labels = torch.full((len(rows), max(len(r["ids"]) for r in rows)), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        c = torch.from_numpy(store.clip(r).copy())
        wav[i, :len(c)] = c
        labels[i, :len(r["ids"])] = torch.tensor(r["ids"])
    return {"wav": wav, "labels": labels, "tokens": sum(len(r["ids"]) for r in rows),
            "seconds": sum(ftkit.duration(r) for r in rows), "padded_seconds": 30.0 * len(rows)}


def loss_fn(model, b):
    mel = log_mel(b["wav"].cuda(non_blocking=True).float() / 32768.0)
    labels = b["labels"].cuda(non_blocking=True)
    return model(input_features=mel, labels=labels).loss * b["tokens"], b["tokens"]


def transcribe(rows, **gen):
    wav = torch.zeros(len(rows), N_SAMPLES, device="cuda")
    for i, r in enumerate(rows):
        c = store.clip_f32(r)
        wav[i, :len(c)] = c.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # use_cache explicitly: model.config.use_cache is off for training
        out = model.generate(input_features=log_mel(wav), language=LANG, task="transcribe",
                             do_sample=False, num_beams=1, use_cache=True, **{"max_new_tokens": 440, **gen})
    return tok.batch_decode(out, skip_special_tokens=True)


# Loop retry, the same rule and settings as 04c, fixed before scoring anything with it: a clip whose
# greedy output repeats a 3-word sequence 5+ times is decoded again, alone, with a repetition
# penalty, no repeated 6-token phrase, and a length cap from its duration. Only the cap is
# Whisper's own: its byte-level BPE spends ~2.3x Flex's tokens on Devanagari, so the cap is the
# densest train label in Whisper tokens (text + end-of-text), ~31 per second.
MAX_TOKENS_PER_S = math.ceil(max((len(r["ids"]) - 3) / ftkit.duration(r) for r in splits["train"]))
RETRY = {"repetition_penalty": 1.2, "no_repeat_ngram_size": 6}
print("retry length cap:", MAX_TOKENS_PER_S, "tokens per second")


def retry_one(row):
    cap = min(440, math.ceil(MAX_TOKENS_PER_S * ftkit.duration(row)))
    return transcribe([row], max_new_tokens=cap, **RETRY)[0]


def evaluate(model, rows=None):
    rows = rows or splits["val"]
    decode = ftkit.RetryLoops(transcribe, retry_one)
    texts, _ = ftkit.transcribe_rows(rows, decode, budget_s=30.0 * EVAL_BATCH, max_items=EVAL_BATCH,
                                     pad_to_s=30.0)
    refs = [r["text"] for r in rows]
    first = {sid: text for sid, text, _ in decode.log}  # greedy output of the retried clips
    greedy = score(refs, [first.get(r["segment_id"], t) for r, t in zip(rows, texts)])
    return {**score(refs, texts), "retried": len(decode.log), "greedy_wer": greedy["wer"],
            "greedy_loops": greedy["loops"]}


def save_best(model):
    model.save_pretrained(OUT / "best", state_dict={k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
    tok.save_pretrained(OUT / "best")
    fe.save_pretrained(OUT / "best")


def make_batches(epoch):
    return ftkit.bucket_batches(splits["train"], budget_s=30.0 * MICRO, max_items=MICRO, pad_to_s=30.0,
                                shuffle=True, seed=epoch)


monitor = ftkit.GpuMonitor().start()
"""),
    md(SPEED_NOTE),
    code(r"""
speed = ftkit.speed_check(model, rows=splits["train"], batches=make_batches(0), collate=collate,
                          loss_fn=loss_fn, evaluate=evaluate, val_rows=splits["val"], gold_rows=splits["gold"],
                          epochs=EPOCHS, monitor=monitor)
(OUT / "speed_check.json").write_text(json.dumps(speed, indent=1))
"""),
    md("## Train"),
    code(r"""
cfg = ftkit.TrainConfig(name=RUN_NAME, out=str(OUT), epochs=EPOCHS, lr=LR, warmup_frac=WARMUP,
                        effective_s=EFFECTIVE_S, patience=2)
result = ftkit.train(model, cfg=cfg, rows=splits["train"], make_batches=make_batches, collate=collate,
                     loss_fn=loss_fn, evaluate=evaluate, save_best=save_best, optimizer=optimizer,
                     monitor=monitor)
print("best val WER", result["best_val_wer"])
"""),
    md("## Gold" + BAKEOFF_LINK),
    code(r"""
import shutil

decode = ftkit.RetryLoops(transcribe, retry_one)
texts, compute = ftkit.transcribe_rows(splits["gold"], decode, budget_s=30.0 * EVAL_BATCH,
                                       max_items=EVAL_BATCH, pad_to_s=30.0)
refs = [r["text"] for r in splits["gold"]]
gold = score(refs, texts)
gold["rtf"] = sum(compute) / sum(ftkit.duration(r) for r in splits["gold"])
first = {sid: text for sid, text, _ in decode.log}  # the same run's greedy output, before any retry
gold["greedy_only"] = score(refs, [first.get(r["segment_id"], t) for r, t in zip(splits["gold"], texts)])
gold["retried"] = [{"segment_id": s, "first": f, "retry": t} for s, f, t in decode.log]
(OUT / "gold_metrics.json").write_text(json.dumps(gold, indent=1, ensure_ascii=False))
ftkit.write_hyps(OUT / "hyps" / f"{RUN_NAME}.jsonl", splits["gold"], texts, compute)
if Path("/content/bakeoff/hyps").exists():
    shutil.copy(OUT / "hyps" / f"{RUN_NAME}.jsonl", "/content/bakeoff/hyps/")
print("with loop retry:", {k: v for k, v in gold.items() if k not in ("greedy_only", "retried")})
print("greedy only:    ", gold["greedy_only"])
for s, f, t in decode.log[:20]:
    print(f"\n{s}\n  greedy: ...{f[-90:]}\n  retry:  ...{t[-90:]}")
"""),
    md("## Results"),
    code(RESULTS),
]


# =============================================================================================
# 04b Omnilingual CTC-1B v2
# =============================================================================================

OMNI_TRAIN = r'''
"""Fine-tune Omnilingual omniASR_CTC_1B_v2 with CTC. Runs in the Python 3.12 environment."""
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ftkit  # noqa: E402
from fairseq2.nn import BatchLayout  # noqa: E402
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # noqa: E402

C = json.loads(Path(sys.argv[1]).read_text())
OUT, FT = Path(C["out"]), Path(C["ft"])
ftkit.fast_cuda()
DATA = ftkit.download_dataset()
splits = ftkit.load_splits(DATA)
store = ftkit.AudioStore(DATA, [r["episode_id"] for rows in splits.values() for r in rows])
score = ftkit.harness_scorer(DATA, FT)
print({k: len(v) for k, v in splits.items()}, f"audio in RAM: {store.gib:.1f} GiB", flush=True)

# The same loader the bake-off used, in float32 master weights for training.
pipe = ASRInferencePipeline(model_card=C["model_card"], dtype=torch.float32)
model, tokenizer = pipe.model, pipe.tokenizer
encode, decode = tokenizer.create_encoder(), tokenizer.create_decoder(skip_special_tokens=True)
UNK, PAD = tokenizer.vocab_info.unk_idx, tokenizer.vocab_info.pad_idx
FRAMES_PER_S = 50  # the wav2vec2 frontend strides 320 samples


def normalize(text: str) -> str:
    # ZWJ/ZWNJ are the only characters of the corpus missing from the written_v2 vocabulary
    # besides "ॐ" (3 uses); they change rendering, not the word.
    return text.replace("‍", "").replace("‌", "")


for name in ("train", "val"):
    kept = []
    for r in splits[name]:
        ids = encode(normalize(r["text"]))
        # fairseq2's BatchLayout rejects a zero-length target, so a clip whose text is empty (one
        # 2 s train clip with no speech) cannot train.
        if (UNK is None or not bool((ids == UNK).any())) and 0 < len(ids) < ftkit.duration(r) * FRAMES_PER_S:
            r["ids"] = ids.tolist()
            kept.append(r)
    print(f"{name}: dropped {len(splits[name]) - len(kept)} clip(s) with an unknown character, an empty "
          "target or an impossible CTC alignment", flush=True)
    splits[name] = kept

# The official recipe always freezes the convolutional feature extractor (recipe.py).
for p in model.encoder_frontend.feature_extractor.parameters():
    p.requires_grad_(False)
model.train()
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=C["lr"],
                              betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0, fused=True)
ftkit.init_optimizer_state(model, optimizer)


def waveforms(rows, pad_to_s):
    """Per-clip layer norm, as the recipe's `normalize_audio` and the inference pipeline do."""
    clips = [torch.nn.functional.layer_norm(c, c.shape) for c in (store.clip_f32(r) for r in rows)]
    wav = torch.zeros(len(rows), ftkit.pad_len(max(len(c) for c in clips), pad_to_s))
    for i, c in enumerate(clips):
        wav[i, :len(c)] = c
    return wav, [len(c) for c in clips]


def collate(rows):
    wav, lens = waveforms(rows, C["pad_to_s"])
    targets = torch.full((len(rows), max(len(r["ids"]) for r in rows)), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        targets[i, :len(r["ids"])] = torch.tensor(r["ids"])
    return {"wav": wav, "lens": lens, "targets": targets, "tlens": [len(r["ids"]) for r in rows],
            "seconds": sum(ftkit.duration(r) for r in rows), "padded_seconds": wav.numel() / ftkit.SR}


def loss_fn(model, b):
    wav = b["wav"].cuda(non_blocking=True)
    targets = b["targets"].cuda(non_blocking=True)
    layout = BatchLayout(wav.shape, seq_lens=b["lens"], device=wav.device)
    tlayout = BatchLayout(targets.shape, seq_lens=b["tlens"], device=wav.device)
    # Summed CTC over the batch; the recipe normalises by the number of clips.
    return model(wav, layout, targets, tlayout), len(b["lens"])


def transcribe(rows):
    wav, lens = waveforms(rows, C["pad_to_s"])
    wav = wav.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, layout = model(wav, BatchLayout(wav.shape, seq_lens=lens, device=wav.device))
    pred = logits.argmax(-1)
    texts = []
    for i in range(pred.shape[0]):  # the inference pipeline's greedy CTC decode
        seq = pred[i, :layout.seq_lens[i]]
        keep = torch.ones_like(seq, dtype=torch.bool)
        keep[1:] = seq[1:] != seq[:-1]
        texts.append(decode(seq[keep]))
    return texts


def evaluate(model, rows=None):
    rows = rows or splits["val"]
    texts, _ = ftkit.transcribe_rows(rows, transcribe, budget_s=C["eval_budget_s"], max_items=128,
                                     pad_to_s=C["pad_to_s"])
    return score([r["text"] for r in rows], texts)


# Probe: the longest padded clip, with its longest plausible target.
longest = max(splits["train"], key=ftkit.duration)
max_len = ftkit.pad_len(round(ftkit.duration(longest) * ftkit.SR), C["pad_to_s"])
max_t = max(len(r["ids"]) for r in splits["train"])


def probe_step(n):
    wav = torch.randn(n, max_len, device="cuda")
    targets = torch.randint(10, tokenizer.vocab_info.size, (n, max_t), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(wav, BatchLayout(wav.shape, seq_lens=[max_len] * n, device=wav.device),
                     targets, BatchLayout(targets.shape, seq_lens=[max_t] * n, device=wav.device))
    loss.backward()


max_items = ftkit.probe_max_items(probe_step, 1, 256, [p for p in model.parameters() if p.requires_grad])
budget_s = max_items * max_len / ftkit.SR * C["probe_fraction"]
print(f"largest micro-batch at {max_len / ftkit.SR:.0f} s: {max_items} clips -> budget {budget_s:.0f} s "
      "of padded audio per micro-batch", flush=True)


def save_best(model):
    (OUT / "best").mkdir(parents=True, exist_ok=True)
    torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, OUT / "best" / "model.pt")
    (OUT / "best" / "README.txt").write_text(
        f"state_dict for {C['model_card']} (bf16). Load with fairseq2 load_model('{C['model_card']}') "
        "then model.load_state_dict(torch.load(...)), and decode with ASRInferencePipeline(model=..., tokenizer=...).\n")


def make_batches(epoch):
    return ftkit.bucket_batches(splits["train"], budget_s=budget_s, max_items=256, pad_to_s=C["pad_to_s"],
                                shuffle=True, seed=epoch)


# Time the run before committing to it; the val pass is also the val WER before training.
monitor = ftkit.GpuMonitor().start()
speed = ftkit.speed_check(model, rows=splits["train"], batches=make_batches(0), collate=collate,
                          loss_fn=loss_fn, evaluate=evaluate, val_rows=splits["val"], gold_rows=splits["gold"],
                          epochs=C["epochs"], monitor=monitor, workers=C["workers"])
(OUT / "speed_check.json").write_text(json.dumps(speed, indent=1))
if C.get("speed_check_only"):
    sys.exit(0)

cfg = ftkit.TrainConfig(name=C["run_name"], out=str(OUT), epochs=C["epochs"], lr=C["lr"],
                        schedule="tristage", warmup_frac=0.1, effective_s=C["effective_s"],
                        patience=C["patience"], workers=C["workers"])
result = ftkit.train(model, cfg=cfg, rows=splits["train"], make_batches=make_batches, collate=collate,
                     loss_fn=loss_fn, evaluate=evaluate, save_best=save_best, optimizer=optimizer,
                     monitor=monitor)

texts, compute = ftkit.transcribe_rows(splits["gold"], transcribe, budget_s=C["eval_budget_s"],
                                       max_items=128, pad_to_s=C["pad_to_s"])
gold = score([r["text"] for r in splits["gold"]], texts)
gold["rtf"] = sum(compute) / sum(ftkit.duration(r) for r in splits["gold"])
gold["best_val_wer"] = result["best_val_wer"]
(OUT / "gold_metrics.json").write_text(json.dumps(gold, indent=1))
ftkit.write_hyps(OUT / "hyps" / f"{C['run_name']}.jsonl", splits["gold"], texts, compute)
if Path("/content/bakeoff/hyps").exists():
    shutil.copy(OUT / "hyps" / f"{C['run_name']}.jsonl", "/content/bakeoff/hyps/")
print("gold", gold, flush=True)
'''

omni = [
    md(
        """
# 04b — Fine-tune Omnilingual CTC-1B v2

A full CTC fine-tune of Meta's `omniASR_CTC_1B_v2`. The model has about 1 B parameters, and the
convolutional feature extractor stays frozen, as in the official recipe. It trains on the 18.3 h
train split, uses val for model selection and gold for the final score.

**Why this model.** It was already trained for ASR on Nepali among 1,600+ languages. It has no
language token, which suits code-switching. Its character vocabulary covers all but 0.02% of the
corpus's characters, including capitals, digits and punctuation, so the pretrained CTC head is
kept.

**Why a script, not cells.** `omnilingual-asr` needs Python <= 3.12 and `fairseq2` 0.6, which pins
`torch==2.8.0`, numpy 1.x and `huggingface_hub` 0.x. None of that installs into the Colab kernel.
Training therefore runs in a `uv` environment as `train_omni.py`, and this notebook writes and
launches it.

**Choices, and where they come from.** All of these follow the official fairseq2 recipe
(`ctc-finetune-recommendation.yaml`, `recipe.py`, `criterion.py`):
- **Peak LR 1e-5.** Meta recommends it for small-language CTC fine-tunes.
- **Tri-stage schedule.** 10% warmup, 40% hold, 50% decay.
- **AdamW** with betas (0.9, 0.98).
- **Summed CTC normalised per clip.**
- **Per-clip waveform layer norm.**
- **The model's own masker as SpecAugment.**
- **Frozen feature extractor.**
"""
        + GPU_NOTE
    ),
    md("## Config"),
    code(r"""
RUN_NAME = "omni-ctc-1b-v2-ft"
USE_DRIVE = True
CONFIG = {
    "run_name": RUN_NAME,
    "model_card": "omniASR_CTC_1B_v2",
    "epochs": 20, "lr": 1e-5, "patience": 4,
    "effective_s": 720.0,     # ~12 min of audio per optimizer step
    "pad_to_s": 1.0,          # pad batches to whole seconds -> ~20 distinct shapes
    "probe_fraction": 0.9,
    "eval_budget_s": 1200.0,  # padded seconds per inference batch (no gradients, so larger)
    "workers": 6,
    "speed_check_only": False,  # True: stop after the timing and projection, before training
}
"""),
    md("## Setup"),
    code(SETUP_COMMON),
    code(r"""
OMNI_ENV = Path("/content/omni-env") if IN_COLAB else FT / "omni-env"
OMNI_PY = OMNI_ENV / "bin" / "python"
if not OMNI_PY.exists():
    !pip install -q uv
    !uv venv -q --python 3.12 {OMNI_ENV}
    !uv pip install -q --python {OMNI_PY} "omnilingual-asr==0.2.0" "torch==2.8.0" "torchaudio==2.8.0"
!uv pip install -q --python {OMNI_PY} soundfile rapidfuzz pyyaml
!{OMNI_PY} -c "import torch, omnilingual_asr; print('omni env: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"""),
    code("%%writefile /content/ft/ftkit.py\n" + FTKIT),
    code("%%writefile /content/ft/train_omni.py\n" + OMNI_TRAIN.strip("\n")),
    md(
        "## Train\n\nThe script streams its log here: the probe result, then a speed check (10 timed "
        "training micro-batches, one val pass that is also the val WER before training, and a projected "
        "run time; set `speed_check_only` to stop there), then every 10 steps throughput, padding waste "
        "and GPU utilisation, then val WER after each epoch and gold at the end."
    ),
    code(r"""
CONFIG.update(out=str(OUT), ft=str(FT))
(FT / "omni_config.json").write_text(json.dumps(CONFIG, indent=1))
!{OMNI_PY} {FT / "train_omni.py"} {FT / "omni_config.json"}
"""),
    md("## Results" + BAKEOFF_LINK),
    code(RESULTS),
]


# =============================================================================================
# 04c Indic-Transcribe-Flex
# =============================================================================================

flex = [
    md(
        """
# 04c — Fine-tune Indic-Transcribe-Flex (mixed-script mode)

A full fine-tune of `bodhan-ai/indic-transcribe-flex`: a 1.2 B Canary-style encoder-decoder with
a 32-layer conformer encoder and a 24-layer transformer decoder. It trains on the 18.3 h train
split with the `ne` + mixed-script prompt, uses val for model selection and gold for the final
score.

**Why this model.** It led the zero-shot bake-off at 18.2% folded WER on gold (CI 14.0–21.2),
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

**Choices.** Peak LR 1e-5, the same as the other two notebooks, so the comparison is about the
models. Linear decay, 10% warmup, up to 6 epochs.

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
RUN_NAME = "indic-transcribe-flex-ft"
MODEL_ID = "bodhan-ai/indic-transcribe-flex"
LANG, MODE = "ne", "mixed"
USE_DRIVE = True
EPOCHS, LR, WARMUP = 6, 1e-5, 0.1
EFFECTIVE_S = 720.0       # ~12 min of audio per optimizer step
PAD_TO_S = 1.0
PROBE_FRACTION = 0.9
GRAD_CKPT = False         # checkpoint the conformer layers if the probe finds a small batch
EVAL_BUDGET_S, EVAL_ITEMS = 1200.0, 96
"""),
    md("## Setup"),
    code(SETUP_COMMON),
    code("%%writefile /content/ft/ftkit.py\n" + FTKIT),
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

asr = IndicTranscribe.from_pretrained(FLEX_DIR, device="cuda", dtype=torch.float32)
model, featurize, tk = asr.model, asr.fe, asr.tokenizer
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
r = splits["train"][0]
assert tk.decode(r["ids"][:-1]) == normalize(r["text"]).strip(), "target encoding does not round-trip"
print("prompt", PROMPT, "| longest target", max(len(r["ids"]) for r in splits["train"]), "tokens")
'''),
    md("## Model, optimizer and the batch-size probe"),
    code(r'''
import torch.utils.checkpoint

if GRAD_CKPT:
    for layer in model.model.encoder.layers:
        layer._forward = layer.forward
        layer.forward = lambda *a, _l=layer, **k: torch.utils.checkpoint.checkpoint(_l._forward, *a, use_reentrant=False, **k)
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

N_PROMPT = len(PROMPT)


def collate(rows):
    clips = [torch.from_numpy(store.clip(r).copy()) for r in rows]
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
    return ftkit.bucket_batches(splits["train"], budget_s=BUDGET_S, max_items=256, pad_to_s=PAD_TO_S,
                                shuffle=True, seed=epoch)


monitor = ftkit.GpuMonitor().start()
"""),
    md("""
## One clip per call vs batched

The bake-off decoded one clip per call (`asr(path)`), at about 1.4 s per clip (RTF 0.12). Here the
same 16 val clips are decoded both ways, under the same bf16 autocast, so the only difference is
batching. The cell reports the speed-up and whether the texts agree; the port's batched
`generate` had never been run before.
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
(OUT / "speed_check.json").write_text(json.dumps(speed, indent=1))
"""),
    md("## Train"),
    code(r"""
cfg = ftkit.TrainConfig(name=RUN_NAME, out=str(OUT), epochs=EPOCHS, lr=LR, warmup_frac=WARMUP,
                        effective_s=EFFECTIVE_S, patience=2)
result = ftkit.train(model, cfg=cfg, rows=splits["train"], make_batches=make_batches, collate=collate,
                     loss_fn=loss_fn, evaluate=evaluate, save_best=save_best, optimizer=optimizer,
                     monitor=monitor)
print("best val WER", result["best_val_wer"])
"""),
    md("## Gold" + BAKEOFF_LINK),
    code(r"""
import shutil

decode = ftkit.RetryLoops(transcribe, retry_one)
texts, compute = ftkit.transcribe_rows(splits["gold"], decode, budget_s=EVAL_BUDGET_S,
                                       max_items=EVAL_ITEMS, pad_to_s=PAD_TO_S)
refs = [r["text"] for r in splits["gold"]]
gold = score(refs, texts)
gold["rtf"] = sum(compute) / sum(ftkit.duration(r) for r in splits["gold"])
first = {sid: text for sid, text, _ in decode.log}  # the same run's greedy output, before any retry
gold["greedy_only"] = score(refs, [first.get(r["segment_id"], t) for r, t in zip(splits["gold"], texts)])
gold["retried"] = [{"segment_id": s, "first": f, "retry": t} for s, f, t in decode.log]
(OUT / "gold_metrics.json").write_text(json.dumps(gold, indent=1, ensure_ascii=False))
ftkit.write_hyps(OUT / "hyps" / f"{RUN_NAME}.jsonl", splits["gold"], texts, compute)
if Path("/content/bakeoff/hyps").exists():
    shutil.copy(OUT / "hyps" / f"{RUN_NAME}.jsonl", "/content/bakeoff/hyps/")
print("with loop retry:", {k: v for k, v in gold.items() if k not in ("greedy_only", "retried")})
print("greedy only:    ", gold["greedy_only"])
for s, f, t in decode.log:
    print(f"\n{s}\n  greedy: ...{f[-90:]}\n  retry:  ...{t[-90:]}")
"""),
    md("## Results"),
    code(RESULTS),
]


for name, cells in [
    ("04a-finetune-whisper-turbo.ipynb", whisper),
    ("04b-finetune-omnilingual-ctc.ipynb", omni),
    ("04c-finetune-indic-transcribe.ipynb", flex),
]:
    (OUT_DIR / name).write_text(
        json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote", OUT_DIR / name, len(cells), "cells")
