"""Build notebooks/04c: `python notebooks/src/build_finetune.py`. Shared code lives in ftkit.py, written out by
a %%writefile cell in each notebook."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
FTKIT = (HERE / "ftkit.py").read_text()
CPUKIT = (HERE / "cpukit.py").read_text()
XTALK = (HERE / "xtalk.py").read_text()
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
fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
axes[0].plot([h["step"] for h in steps], [h["loss"] for h in steps], color="#2a78d6")
axes[0].set_title("train loss per unit", loc="left")
axes[1].plot([h["epoch"] for h in evals], [h["val_wer"] for h in evals], marker="o", color="#2a78d6")
axes[1].set_title("val folded WER %", loc="left")
epochs = [h["epoch"] for h in evals]
for key, label, colour in (("sub", "S", "#2a78d6"), ("del", "D", "#eb6834"), ("ins", "I", "#1baf7a")):
    ys = [h[f"val_{key}"] for h in evals]
    axes[2].plot(epochs, ys, marker="o", color=colour, label=label)
    if ys:
        axes[2].annotate(label, (epochs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                         va="center", color="#52514e")
axes[2].set_ylim(bottom=0)
axes[2].legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
axes[2].set_title("val S / D / I per 100 ref words", loc="left")
axes[3].plot([h["step"] for h in steps], [h["gpu_util"] for h in steps], color="#2a78d6")
axes[3].set_ylim(0, 100)
axes[3].set_title("GPU utilisation %", loc="left")
for ax in axes:
    ax.grid(color="#ecebe7")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
gold_m = json.loads((OUT / "gold_metrics.json").read_text())
print("gold, folded, per 100 ref words (S + D + I = WER):")
for name, m in [("all", gold_m)] + list(gold_m.get("by_overlap", {}).items()):
    print(f"  {name:>6}: {m['clips']:4d} clips  WER {m['wer']:6.2f}  {ftkit.sid(m)}")
print(json.dumps(gold_m, indent=1))
"""

SPEED_NOTE = """
## Speed check (before committing to the run)

This times forward+backward on 10 real training micro-batches, then one full batched val pass,
and projects the whole run. The val pass doubles as the **val WER before training**, the baseline
that fine-tuning has to beat. No optimizer step runs, so no weight moves. If the projection is
too long, or GPU utilisation is low, stop here and fix it.
"""

BAKEOFF_LINK = """
The gold hypotheses go to `OUT/hyps/<RUN_NAME>.jsonl`, in the bake-off's cache format. Copy them to
the bake-off checkout's `hyps/` folder to score them beside the zero-shot systems.
"""


flex = [
    md(
        """
# 04c — Fine-tune Indic-Transcribe-Flex (mixed-script mode)

A full fine-tune of `bodhan-ai/indic-transcribe-flex`: a 1.2 B Canary-style encoder-decoder with
a 32-layer conformer encoder and a 24-layer transformer decoder. It trains on the 18.3 h train
split with the `ne` + mixed-script prompt, uses val for model selection and gold for the final
score.

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
where the errors are. Judge the run on the gold crosstalk buckets, and on the Models page's
within-episode ratios, not on overall WER.

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
RUN_NAME = "indic-transcribe-flex-ft-YYYY-MM-DD"  # the date makes the folder in OUT_REPO and on the Models page unique
MODEL_ID = "bodhan-ai/indic-transcribe-flex"
LANG, MODE = "ne", "mixed"
USE_DRIVE = False          # outputs stay on the VM and go to OUT_REPO at the end; True also keeps them on Drive
OUT_REPO = "Sagyam/nepanglish-asr-flex-ft"  # private HF model repo; every run lands in <RUN_NAME>/
EPOCHS, LR, WARMUP = 6, 1e-5, 0.1
XTALK_P = 0.3              # share of measured-clean train clips given synthetic crosstalk each epoch; 0 = off
# Shown on the harness's Models page (D83). Say what is different about this run.
MODEL_NAME = "Indic-Transcribe-Flex FT"
MODEL_DESCRIPTION = (f"04c full fine-tune, standard settings + synthetic crosstalk (xtalk.py, p={XTALK_P})"
                     if XTALK_P else "04c full fine-tune, standard settings.")
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
if XTALK_P:
    n = ftkit.attach_speaker_turns(DATA, splits["train"])
    print(f"speaker turns for {n} of {len(splits['train'])} train clips")
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

import numpy as np

import xtalk

N_PROMPT = len(PROMPT)
MIXER = None
if XTALK_P:
    MIXER = xtalk.Mixer(xtalk.DonorPool(splits["train"]),
                        lambda ep, s, e: store.audio[ep][round(s * ftkit.SR):round(e * ftkit.SR)], p=XTALK_P)


def train_clip(row, rng):
    clip = store.clip(row)
    return MIXER(row, clip, rng) if MIXER else (clip, None)


def collate(rows):
    # torch reseeds each DataLoader worker per epoch, so every epoch draws fresh mixes, and the
    # run is still repeatable from cfg.seed
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
    return ftkit.bucket_batches(splits["train"], budget_s=BUDGET_S, max_items=256, pad_to_s=PAD_TO_S,
                                shuffle=True, seed=epoch)


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

Six examples go to `OUT/xtalk_examples/`, each saved clean and mixed, and three play below.
**Listen to them before starting the run.** If the other voice is inaudible, or clearly from
another room, stop and fix the mixer.
"""),
    code(r"""
import soundfile as sf
from IPython.display import Audio, display

if MIXER:
    rng = np.random.default_rng(0)
    eligible = [r for r in splits["train"] if r["overlap_spans"] == [] and r.get("speaker_turns")]
    always = xtalk.Mixer(MIXER.pool, MIXER.fetch, p=1.0)
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
    (OUT / "xtalk_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    ex = OUT / "xtalk_examples"
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
    print("XTALK_P = 0: no synthetic crosstalk in this run")
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
decode = ftkit.RetryLoops(transcribe, retry_one)
texts, compute = ftkit.transcribe_rows(splits["gold"], decode, budget_s=EVAL_BUDGET_S,
                                       max_items=EVAL_ITEMS, pad_to_s=PAD_TO_S)
refs = [r["text"] for r in splits["gold"]]
gold = score(refs, texts)
gold["rtf"] = sum(compute) / sum(ftkit.duration(r) for r in splits["gold"])
first = {sid: text for sid, text, _ in decode.log}  # the same run's greedy output, before any retry
gold["greedy_only"] = score(refs, [first.get(r["segment_id"], t) for r, t in zip(splits["gold"], texts)])
gold["retried"] = [{"segment_id": s, "first": f, "retry": t} for s, f, t in decode.log]
ftkit.write_hyps(OUT / "hyps" / f"{RUN_NAME}.jsonl", splits["gold"], texts, compute)
BUCKETS = ("none", "0-5%", "5-15%", ">15%")
gold["by_overlap"] = {}
for bucket in BUCKETS:
    idx = [i for i, r in enumerate(splits["gold"]) if (r.get("classes") or {}).get("overlap") == bucket]
    if idx:
        gold["by_overlap"][bucket] = score([refs[i] for i in idx], [texts[i] for i in idx])
(OUT / "gold_metrics.json").write_text(json.dumps(gold, indent=1, ensure_ascii=False))
print("with loop retry:", {k: v for k, v in gold.items() if k not in ("greedy_only", "retried", "by_overlap")})
print("greedy only:    ", gold["greedy_only"])
print("\nfolded, per 100 ref words (S + D + I = WER):")
print(f"  {'all':>6}: {gold['clips']:4d} clips  WER {gold['wer']:6.2f}  {ftkit.sid(gold)}")
for bucket, m in gold["by_overlap"].items():
    print(f"  {bucket:>6}: {m['clips']:4d} clips  WER {m['wer']:6.2f}  {ftkit.sid(m)}")
for s, f, t in decode.log:
    print(f"\n{s}\n  greedy: ...{f[-90:]}\n  retry:  ...{t[-90:]}")
"""),
    md("""
## Harness model folder

Everything the harness's **Models** page needs to show this run (D83): the model card, plus gold
and val transcripts from the best weights under the same decoder. No weights: the page scores
text, it never runs the model. After the upload at the end, the files of `harness/` go into the
harness checkout as `data/models/asr/<RUN_NAME>/` (the folder name becomes the model's id); press **Rescan** on the
Models page. The harness scores the text against its *current* labels, so a clip relabeled since
this export is scored against the new label.
"""),
    code(r"""
import datetime as dt

HARNESS = OUT / "harness"
val_decode = ftkit.RetryLoops(transcribe, retry_one)
val_texts, val_compute = ftkit.transcribe_rows(splits["val"], val_decode, budget_s=EVAL_BUDGET_S,
                                               max_items=EVAL_ITEMS, pad_to_s=PAD_TO_S)
ftkit.write_hyps(HARNESS / "gold.jsonl", splits["gold"], texts, compute)
ftkit.write_hyps(HARNESS / "val.jsonl", splits["val"], val_texts, val_compute)
evals = [h for h in result["history"] if "val_wer" in h]
val_scores = score([r["text"] for r in splits["val"]], val_texts)
export = json.loads((DATA / "training" / "manifest.json").read_text())
card = {
    "name": MODEL_NAME,
    "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    "description": MODEL_DESCRIPTION,
    "architecture": "Canary-style enc-dec: 32L conformer + 24L transformer decoder, 1.2B",
    "base_model": MODEL_ID,
    "decoder": "greedy+retry",
    "run_name": RUN_NAME,
    "epochs": EPOCHS,
    "best_epoch": min(evals, key=lambda h: h["val_wer"])["epoch"] if evals else None,
    "lr": LR,
    "xtalk_p": XTALK_P,
    "val_wer": result["best_val_wer"],
    "gold_wer": gold["wer"],
    # folded, per 100 reference words; S + D + I = WER
    "val_sid": {k: val_scores[k] for k in ("sub", "del", "ins")},
    "gold_sid": {k: gold[k] for k in ("sub", "del", "ins")},
    "train_export": {k: export.get(k) for k in ("exported_at", "git_commit", "row_count",
                                                "normalization_version", "label_version")},
}
(HARNESS / "model_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False))
print(f"val {val_scores['wer']:.2f}% ({ftkit.sid(val_scores)}) | wrote", HARNESS)
"""),
    md("## Results"),
    code(RESULTS),
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
## Upload to Hugging Face

Every file of this run goes to the private model repo `OUT_REPO`, under `<RUN_NAME>/`: `best/` (bf16),
`cpu/` (when int8 was accepted), `harness/`, `hyps/`, `int8/` and the metrics json. Nothing has to
be pulled from Drive, and the VM can be deleted once this prints a commit.

**The token must be able to write.** The cell reads the `HF_TOKEN` secret again, because Setup
cached whatever token was there when it ran, so a secret swapped mid-session is picked up.

Then, in the harness checkout:
```bash
hf download Sagyam/nepanglish-asr-flex-ft --include "<RUN_NAME>/*" --local-dir exports/<RUN_NAME>
mkdir -p data/models/asr/<RUN_NAME>
cp exports/<RUN_NAME>/<RUN_NAME>/harness/* data/models/asr/<RUN_NAME>/
cp -r exports/<RUN_NAME>/<RUN_NAME>/cpu data/models/asr/<RUN_NAME>/   # mic playground (D85)
```
Press **Rescan** on the Models page. The playground sidecar starts with `docker compose up -d`.
"""),
    code(r"""
from huggingface_hub import HfApi

if IN_COLAB:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    (Path.home() / ".cache" / "huggingface" / "token").write_text(os.environ["HF_TOKEN"])
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(OUT_REPO, repo_type="model", private=True, exist_ok=True)
commit = api.upload_folder(repo_id=OUT_REPO, folder_path=str(OUT), path_in_repo=RUN_NAME,
                           commit_message=f"{RUN_NAME}: {MODEL_DESCRIPTION}")
size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
print(f"{size / 2**30:.2f} GiB -> https://huggingface.co/{OUT_REPO}/tree/main/{RUN_NAME} @ {commit.oid[:7]}")
"""),
]


for name, cells in [("Finetune.ipynb", flex)]:
    (OUT_DIR / name).write_text(
        json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote", OUT_DIR / name, len(cells), "cells")
