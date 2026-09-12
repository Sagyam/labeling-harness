"""Build notebooks/03-asr-bakeoff.ipynb: `python notebooks/src/build_bakeoff.py`."""

import json
import sys
from pathlib import Path

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.strip("\n"),
        }
    )


md("""
# 03 — Zero-shot ASR bake-off on gold

Which pretrained recogniser starts closest to this corpus, before any fine-tuning? Three candidates
from the literature review, each scored untrained on the 706 gold clips with the harness's own
scorer (`fold.py`, the same one `EDA.ipynb` uses):

| model | why it is here |
|---|---|
| `openai/whisper-large-v3-turbo` | most evidence at 10–20 h of fine-tuning data; run with the `ne` and the `hi` prompt |
| `bodhan-ai/indic-transcribe-flex`, `mode="mixed"` | writes this corpus's convention (Devanagari + English in Latin) natively |
| Omnilingual `omniASR_CTC_1B_v2` | already ASR-trained on Nepali; CTC with no language token |

**Run on a Colab GPU runtime** (L4 or A100). Before the first run:
1. add a Colab secret `HF_TOKEN` that can read the private `Sagyam/nepanglish-asr`;
2. accept the terms on <https://huggingface.co/bodhan-ai/indic-transcribe-flex> (gated, auto-approved).

Every model cell appends its hypotheses to `/content/bakeoff/hyps/<model>.jsonl` and skips clips
already there, so a *Restart session* (which keeps `/content`) loses nothing. Re-run **Setup**
after any restart. Omnilingual runs in its own Python 3.12 environment (see its section).

**Read every number as distance from the reference, not accuracy.** The references are a
lightly edited LLM fusion of three commercial recognisers, and gold is not speaker-held-out
(which does not matter for zero-shot, but will for the fine-tunes that follow).
""")

md("## Setup (re-run after any restart)")

code(r'''
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import userdata

    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
WORK = Path("/content/bakeoff") if IN_COLAB else Path.cwd() / ".cache-bakeoff"
HYPS, CLIPS = WORK / "hyps", WORK / "clips"
HYPS.mkdir(parents=True, exist_ok=True)
CLIPS.mkdir(parents=True, exist_ok=True)
SR = 16_000

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

REPO = "Sagyam/nepanglish-asr"
# BAKEOFF_DATA points at a local copy in the same layout (e.g. exports/hf-nepanglish-asr).
if "BAKEOFF_DATA" in os.environ:
    DATA = Path(os.environ["BAKEOFF_DATA"])
    GOLD = [json.loads(line) for line in open(DATA / "gold" / "gold.jsonl", encoding="utf-8")]
else:
    gold_path = hf_hub_download(REPO, "gold/gold.jsonl", repo_type="dataset")
    GOLD = [json.loads(line) for line in open(gold_path, encoding="utf-8")]
    DATA = Path(snapshot_download(
        REPO, repo_type="dataset",
        allow_patterns=["gold/*", "harness/*", "analytics/analytics.jsonl",
                        *[f"training/episodes/{e}.flac" for e in sorted({r["episode_id"] for r in GOLD})]],
    ))
assert len(GOLD) == 706, len(GOLD)

# Cut each clip out of its episode exactly as the dataset card specifies: round(t * 16000).
def clip_path(row):
    return CLIPS / f"{row['segment_id']}.flac"

todo = [r for r in GOLD if not clip_path(r).exists()]
for ep in sorted({r["episode_id"] for r in todo}):
    audio, sr = sf.read(DATA / "training" / "episodes" / f"{ep}.flac", dtype="float32")
    assert sr == SR and audio.ndim == 1, (ep, sr, audio.shape)
    for r in (r for r in todo if r["episode_id"] == ep):
        sf.write(clip_path(r), audio[round(r["start_time"] * SR):round(r["end_time"] * SR)], SR)
print(f"{len(GOLD)} gold clips, {sum(r['end_time'] - r['start_time'] for r in GOLD) / 3600:.2f} h, "
      f"{len(todo)} cut this run")


def run_model(name, transcribe, batch_size):
    """Append `name`'s hypotheses for every gold clip not already cached; time each batch."""
    out = HYPS / f"{name}.jsonl"
    done = {json.loads(line)["segment_id"] for line in out.open(encoding="utf-8")} if out.exists() else set()
    rows = [r for r in GOLD if r["segment_id"] not in done]
    with out.open("a", encoding="utf-8") as fh:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            t0 = time.perf_counter()
            texts = transcribe(batch)
            took = time.perf_counter() - t0
            audio_s = sum(r["end_time"] - r["start_time"] for r in batch)
            for r, text in zip(batch, texts, strict=True):
                share = took * (r["end_time"] - r["start_time"]) / audio_s
                fh.write(json.dumps({"segment_id": r["segment_id"], "text": text.strip(),
                                     "compute_s": share}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"\r{name}: {len(done) + i + len(batch)}/{len(GOLD)}", end="")
    print(f"\r{name}: {len(GOLD)}/{len(GOLD)} cached")
''')

md("## Whisper-large-v3-turbo (`ne` and `hi` prompts)")

code(r"""
import torch
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
               dtype=torch.float16, device="cuda:0")


def whisper(lang):
    def transcribe(batch):
        audio = [{"raw": sf.read(clip_path(r), dtype="float32")[0], "sampling_rate": SR} for r in batch]
        out = asr(audio, batch_size=len(batch),
                  generate_kwargs={"language": lang, "task": "transcribe", "max_new_tokens": 440})
        return [o["text"] for o in out]
    return transcribe


for lang in ("ne", "hi"):
    run_model(f"whisper-turbo-{lang}", whisper(lang), batch_size=16)
del asr
torch.cuda.empty_cache()
""")

md('## Indic-Transcribe-Flex (mixed-script mode, `lang="ne"`)')

code(r"""
import inspect

import torch

flex_dir = snapshot_download("bodhan-ai/indic-transcribe-flex")
sys.path.insert(0, flex_dir)  # the model code ships inside the download
from indic_transcribe import IndicTranscribe  # noqa: E402

kwargs = {"device": "cuda"} if "device" in inspect.signature(IndicTranscribe.from_pretrained).parameters else {}
flex = IndicTranscribe.from_pretrained(flex_dir, **kwargs)
run_model("indic-transcribe-flex-mixed",
          lambda batch: [flex(str(clip_path(r)), lang="ne", mode="mixed") for r in batch], batch_size=8)
del flex
torch.cuda.empty_cache()
""")

md("""
## Omnilingual CTC-1B v2 (in its own Python 3.12 environment)

`omnilingual-asr` requires Python <= 3.12 and pins `fairseq2` 0.6, whose native library ships
only cp310-cp312 wheels built against `torch==2.8.0`, plus `numpy~=1.23` and
`huggingface_hub~=0.32`. None of that installs into a Colab image on Python 3.13 with numpy 2 and
hub 1.x, so it gets a separate `uv` environment and runs as a script that writes the same
hypothesis cache as every other model.
""")

code(r"""
OMNI_ENV = Path("/content/omni-env") if IN_COLAB else WORK / "omni-env"
OMNI_PY = OMNI_ENV / "bin" / "python"
if not OMNI_PY.exists():
    !pip install -q uv
    !uv venv -q --python 3.12 {OMNI_ENV}
    !uv pip install -q --python {OMNI_PY} "omnilingual-asr==0.2.0" "torch==2.8.0" "torchaudio==2.8.0"
!{OMNI_PY} -c "import torch, omnilingual_asr; print('omni env: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
""")

code(r'''
OMNI_SCRIPT = WORK / "omni_infer.py"
OMNI_SCRIPT.write_text("""
import json, sys, time
from pathlib import Path

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

gold_path, clips, out = map(Path, sys.argv[1:4])
gold = [json.loads(line) for line in gold_path.open(encoding="utf-8")]
done = {json.loads(line)["segment_id"] for line in out.open(encoding="utf-8")} if out.exists() else set()
rows = [r for r in gold if r["segment_id"] not in done]
pipe = ASRInferencePipeline(model_card="omniASR_CTC_1B_v2")  # CTC ignores `lang`, so none is passed
with out.open("a", encoding="utf-8") as fh:
    for i in range(0, len(rows), 16):
        batch = rows[i:i + 16]
        t0 = time.perf_counter()
        texts = pipe.transcribe([str(clips / f"{r['segment_id']}.flac") for r in batch], batch_size=len(batch))
        took = time.perf_counter() - t0
        audio_s = sum(r["end_time"] - r["start_time"] for r in batch)
        for r, text in zip(batch, texts, strict=True):
            share = took * (r["end_time"] - r["start_time"]) / audio_s
            fh.write(json.dumps({"segment_id": r["segment_id"], "text": text.strip(),
                                 "compute_s": share}, ensure_ascii=False) + "\\n")
        fh.flush()
        print(f"omni-ctc-1b-v2: {len(done) + i + len(batch)}/{len(gold)}", flush=True)
""")
!{OMNI_PY} {OMNI_SCRIPT} {DATA / "gold" / "gold.jsonl"} {CLIPS} {HYPS / "omni-ctc-1b-v2.jsonl"}
''')

md("""
## Score

Folded WER forgives Devanagari/Latin spellings of one word (`टिम` / `team`); raw WER does not. CER is
over the folded words with Latin lowercased. The three commercial recognisers are included for scale,
but they were **fused into the reference**, so their numbers are flattered by construction.
""")

code(r"""
%pip install -q rapidfuzz pandas matplotlib
import re
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from rapidfuzz.distance import Levenshtein

# The HF cache stores files as symlinks into its blob store, and normalize.py finds
# config/normalization.yaml relative to its own *resolved* path, so import a real copy.
HARNESS = WORK / "harness"
if not HARNESS.exists():
    shutil.copytree(DATA / "harness", HARNESS)
for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
    del sys.modules[name]
sys.path.insert(0, str(HARNESS / "backend"))
from app.services.fold import fold_tokens, fold_version, word_errors  # noqa: E402

print("fold", fold_version())

# Copied from EDA.ipynb so the tiers and the CER mean the same thing in both notebooks.
DEV = re.compile(r"[ऀ-ॿ]")
LAT = re.compile(r"[A-Za-z]")
TIERS = ["Low (<10)", "Medium (10-30)", "Heavy (>=30)", "English only"]
TIER_COLOR = dict(zip(TIERS, ["#9cc3ee", "#4d8fdc", "#1d5aa6", "#8f8d86"]))


def chars(text):
    return " ".join(t if DEV.search(t) else t.lower() for t in fold_tokens(text))


REFERENCE_SYSTEMS = {"elevenlabs-scribe-v2": "Scribe (in ref)", "gemini-3.8-flash": "Gemini (in ref)",
                     "mai-transcribe-2": "MAI (in ref)"}
gold_ids = {r["segment_id"] for r in GOLD}
hyps = {}  # model -> {segment_id: (text, compute_s)}
for path in sorted(HYPS.glob("*.jsonl")):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    hyps[path.stem] = {r["segment_id"]: (r["text"], r["compute_s"]) for r in rows}
with open(DATA / "analytics" / "analytics.jsonl", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        if r["segment_id"] in gold_ids:
            for h in r["hypotheses"]:
                if h["system_id"] in REFERENCE_SYSTEMS:
                    hyps.setdefault(REFERENCE_SYSTEMS[h["system_id"]], {})[r["segment_id"]] = (h["text"] or "", None)
MODELS = [m for m in hyps if "(in ref)" not in m] + list(REFERENCE_SYSTEMS.values())
incomplete = {m: len(gold_ids - hyps[m].keys()) for m in MODELS if gold_ids - hyps[m].keys()}
assert not incomplete, f"missing hypotheses: {incomplete}"

recs = []
for r in GOLD:
    ref = r["text"]
    toks = fold_tokens(ref)
    n_dev = sum(bool(DEV.search(t)) for t in toks)
    n_lat = sum(bool(LAT.search(t)) and not DEV.search(t) for t in toks)
    rec = {"segment_id": r["segment_id"], "episode": r["episode_id"], "words": len(toks),
           "n_dev": n_dev, "n_lat": n_lat, "ref_chars": len(chars(ref)),
           "seconds": r["end_time"] - r["start_time"]}
    for m in MODELS:
        hyp, compute_s = hyps[m][r["segment_id"]]
        rec[f"{m}|werr"] = word_errors(ref, hyp).errors
        raw = word_errors(ref, hyp, folded=False)
        rec[f"{m}|rerr"], rec[f"{m}|rwords"] = raw.errors, raw.ref_words
        rec[f"{m}|cerr"] = Levenshtein.distance(chars(ref), chars(hyp))
        rec[f"{m}|compute"] = compute_s
        htoks = hyp.split()
        rec[f"{m}|lat"] = sum(bool(LAT.search(t)) for t in htoks)
        rec[f"{m}|toks"] = len(htoks)
    recs.append(rec)
seg = pd.DataFrame(recs).set_index("segment_id")
n_mixed = seg.n_dev + seg.n_lat
seg["cmi"] = np.where(n_mixed > 0, 100 * (n_mixed - np.maximum(seg.n_dev, seg.n_lat)) / n_mixed.clip(lower=1), 0.0)
seg["cmi_tier"] = np.select(
    [(seg.n_dev == 0) & (seg.n_lat > 0), seg.cmi < 10, seg.cmi < 30], [TIERS[3], TIERS[0], TIERS[1]],
    default=TIERS[2],
)
scored = seg[seg.words > 0]
ref_latin = 100 * sum(len([t for t in r["text"].split() if LAT.search(t)]) for r in GOLD) / sum(
    len(r["text"].split()) for r in GOLD)


def wer_table(frame, cluster="episode", n_boot=2000, seed=0):
    out = {}
    for m in MODELS:
        g = frame.groupby(cluster).agg(e=(f"{m}|werr", "sum"), w=("words", "sum"))
        pick = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
        boot = 100 * g.e.to_numpy()[pick].sum(1) / g.w.to_numpy()[pick].sum(1)
        compute = frame[f"{m}|compute"]
        out[m] = {
            "WER %": 100 * frame[f"{m}|werr"].sum() / frame.words.sum(),
            "WER lo": np.percentile(boot, 2.5), "WER hi": np.percentile(boot, 97.5),
            "raw WER %": 100 * frame[f"{m}|rerr"].sum() / frame[f"{m}|rwords"].sum(),
            "CER %": 100 * frame[f"{m}|cerr"].sum() / frame.ref_chars.sum(),
            "Latin tok %": 100 * frame[f"{m}|lat"].sum() / max(frame[f"{m}|toks"].sum(), 1),
            "RTF": compute.sum() / frame.seconds.sum() if compute.notna().all() else np.nan,
        }
    return pd.DataFrame(out).T.round(3)


pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
print(f"{len(scored)} scored clips; the references are {ref_latin:.1f}% Latin tokens")
overall = wer_table(scored)
by_tier = pd.DataFrame({t: wer_table(scored[scored.cmi_tier == t])["WER %"] for t in TIERS
                        if (scored.cmi_tier == t).any()})
by_tier.columns = [f"{c} (n={(scored.cmi_tier == c).sum()})" for c in by_tier.columns]
display(overall)
display(by_tier.round(1))
overall.join(by_tier).to_csv(WORK / "bakeoff_results.csv")
""")

code(r"""
tiers = [t for t in TIERS if (scored.cmi_tier == t).any()]
fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(MODELS) + 1.2))
height = 0.8 / len(tiers)
for k, t in enumerate(tiers):
    vals = [wer_table(scored[scored.cmi_tier == t]).loc[m, "WER %"] for m in MODELS]
    ax.barh(np.arange(len(MODELS)) + k * height, vals, height=height - 0.02, color=TIER_COLOR[t], label=t)
ax.set_yticks(np.arange(len(MODELS)) + height * (len(tiers) - 1) / 2, MODELS)
ax.invert_yaxis()
ax.set_xlabel("folded WER % against the gold references (lower is closer)")
ax.set_title("Zero-shot distance from gold, by code-mixing tier", loc="left", fontweight="bold")
ax.grid(axis="x", color="#ecebe7")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(title="CMI tier", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), borderaxespad=2.2,
          ncol=len(tiers))
fig.tight_layout()
fig.savefig(WORK / "bakeoff_wer_by_tier.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
## Fine-tune pre-flight (measured locally on 2026-09-12, no GPU needed)

* **Whisper's 448-token decoder cap is not a constraint.** Train+val label lengths in the
  whisper-large-v3-turbo tokenizer: median 147, p95 320, p99 366, max 449. One label exceeds the
  ~443 tokens left after the prompt, and 7 exceed 400. Drop or split that one clip.
* **Omnilingual's CTC head can be kept.** `omniASR_tokenizer_written_v2` (10,288 characters) has all
  26 uppercase Latin letters, the 10 digits and `, ? . - ' ! : %`. Of the corpus's 159 characters,
  only ZWJ (209 uses), ZWNJ (1) and `ॐ` (3) are missing: 0.02% of non-space characters.

Whether each model actually *writes* case, punctuation and Latin-script English zero-shot is in the
table above: compare the `Latin tok %` column with the references' share.
""")

md("""
## Findings (run 2026-09-12, Colab A100 40 GB)

All numbers are folded WER on the 706 gold clips, as distance from the fused reference, with 95%
episode-bootstrap intervals.

- **Indic-Transcribe-Flex (mixed mode) is the clear starting point: 18.2% (14.0–21.2), CER 14.0%.**
  - It is flat across code-mixing tiers (17.5 / 19.9 / 17.1 / 16.9).
  - It already writes the corpus's script convention: 33.6% Latin tokens against the references'
    36.0%.
  - It loops on only 4 clips.
  - Unbatched, it ran at RTF 0.12.
- **Omnilingual CTC-1B v2: 41.3% (33.5–47.3), CER 43.7%.**
  - It transliterates English into Devanagari: only 12% of its tokens are Latin. fold.py forgives
    that at the word level, but the CER does not.
  - Its best tier is English-only (18.5%).
  - It is the fastest model by far (RTF < 0.01).
- **Whisper-large-v3-turbo is unusable zero-shot: 123% with `ne` and 121% with `hi`.**
  - It falls into repetition loops on 271 and 283 of the 706 clips.
  - Its median hypothesis length matches the reference, so the loops, not the recognition, drive
    the error.
- **The commercial recognisers score 12–13%, flattered by construction** because they were fused
  into the reference.
- **Conventions fine-tuning has to teach.**
  - Punctuation: the references use `।` (1,382 times). Flex and Omnilingual write almost no
    sentence punctuation.
  - Numbers: Flex's ITN writes 695 Latin digits, while the references mostly spell numbers in
    words (311 Latin and 71 Devanagari digits). fold.py cannot forgive a word against a digit.

**Decision.** Fine-tune all three: `04a`, `04b` and `04c`.
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
OUT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).parent.parent / "03-asr-bakeoff.ipynb"
)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", OUT, len(cells), "cells")
