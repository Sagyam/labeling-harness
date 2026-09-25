"""Build notebooks/Teacher.ipynb: `python notebooks/src/build_teacher.py`. Roadmap §B steps 2-3:
Flex, the teacher, transcribes the unlabelled distillation corpus (D101), and its labels are
filtered. Shared code lives in ftkit.py and distill.py, written out by %%writefile cells."""

import json
import sys
from pathlib import Path

from build_finetune import code, md, notebook

HERE = Path(__file__).parent
FTKIT = (HERE / "ftkit.py").read_text()
DISTILLKIT = (HERE / "distill.py").read_text()
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent

INTRO = """
# Distillation, steps 2-3 — the teacher labels the unlabelled corpus

Roadmap §B. Flex, the teacher, transcribes every clip of the unlabelled corpus that
`PreDistill.ipynb` cut into `DATASET_REPO/distill/` (D101), and its labels are filtered before any
student trains on them. Never the paid routes: Flex on this
GPU is the only model.

**Teacher.** Flex p00-s0, the 2026-09-22 run on the current export: the model the step-0
students were scored against (gold 11.56, val 7.19). Greedy decoding, with per-token log-probs
kept. There is no loop retry: step 3 drops every clip whose greedy output loops, so retrying would
spend GPU on labels that are thrown away.

**Step 3, cheapest rule first** (`distill.filter_pseudo`): drop a clip whose output loops; an
empty transcript; a rate outside the tokens per second train's labels span, in the teacher's own
tokens; then the least confident `DROP_FRACTION` of what is left, by mean log-prob. The report
splits what was kept and the teacher's confidence by the share of English words, because every
step-0 student fell furthest behind Flex on pure Nepali.

**Small blast radius.** Labels go to `DATASET_REPO/<LABELS>/shards/` every `SHARD_CLIPS` clips, and
a rerun skips every clip already in a shard, so a lost runtime costs at most one shard. `LIMIT`
decodes only the first clips, for a smoke run before the whole corpus.
"""

CONFIG = r"""
DATASET_REPO = "Sagyam/nepanglish-asr"     # PreDistill.ipynb wrote the corpus to its distill/
PREFIX = "distill"
FLEX_REPO = "Sagyam/nepanglish-asr-flex-ft"
TEACHER = "flex-xtalk-sweep-2026-09-22/flex-xtalk-sweep-2026-09-22-p00-s0"
LABELS = f"{PREFIX}/pseudo/flex-p00-s0"      # DATASET_REPO/<LABELS>/: shards, labels.jsonl, report.json
LANG, MODE = "ne", "mixed"
LIMIT = None               # decode only the first N clips (a smoke run); None for the whole corpus
SHARD_CLIPS = 2000         # clips per uploaded shard: a lost runtime loses at most one
DECODE_BUDGET_S, DECODE_ITEMS = 1200.0, 64
PAD_TO_S = 1.0
MAX_NEW_TOKENS = 300       # as the students' reference decoder
DROP_FRACTION = 0.15       # step 3: the least confident share dropped (roadmap: 10-20%)
"""

SETUP = r"""
%pip install -q rapidfuzz
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
IN_COLAB = "google.colab" in sys.modules
# Secrets only work from a cell run in the Colab UI: run this cell by hand once when cells are
# driven from outside (the Colab MCP); the token is then kept in the hub's token file on the VM.
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
OUT = FT / "out" / LABELS
(OUT / "shards").mkdir(parents=True, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"""

DATA = r"""
sys.path.insert(0, str(FT))
import warnings

import torch
import transformers
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

import distill
import ftkit

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", module="transformers")
ftkit.fast_cuda()
TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)

with ftkit.timed("downloading the corpus"):
    CORPUS = Path(snapshot_download(DATASET_REPO, repo_type="dataset", token=TOKEN,
                                    allow_patterns=[f"{PREFIX}/clips.jsonl", f"{PREFIX}/episodes/*"]))
clips = [json.loads(line) for line in (CORPUS / PREFIX / "clips.jsonl").read_text("utf-8").splitlines() if line]
if LIMIT:
    clips = clips[:LIMIT]
with ftkit.timed("reading the recordings into RAM"):
    store = ftkit.AudioStore(CORPUS, [c["episode_id"] for c in clips], folder=f"{PREFIX}/episodes")
print(f"audio in RAM: {store.gib:.1f} GiB")
with ftkit.timed("downloading the teacher"):
    TEACHER_DIR = Path(snapshot_download(FLEX_REPO, allow_patterns=[f"{TEACHER}/best/*"], token=TOKEN)) / TEACHER / "best"
sys.path.insert(0, str(TEACHER_DIR))
from indic_transcribe import MODES, IndicTranscribe  # noqa: E402

asr = IndicTranscribe.from_pretrained(str(TEACHER_DIR), device="cuda", dtype=torch.float32)
model, featurize, tk = asr.model, asr.fe, asr.tokenizer
model.eval()
itn, romanized = MODES[MODE]
PROMPT = tk.encode_prompt(LANG, itn=itn, romanized=romanized)
EOS, PAD = tk.eos_id, tk.pad_id
print(f"{len(clips)} clips, {sum(ftkit.duration(c) for c in clips) / 3600:.1f} h, from "
      f"{len({c['source_id'] for c in clips})} sources | teacher {TEACHER}")
"""

RATE_NOTE = """
## The rate train's labels span, in the teacher's tokens

Step 3's rate rule keeps a clip whose tokens per second fall inside the range the verified train
labels span when the teacher's tokenizer writes them, as Finetune.ipynb encodes its targets.
"""

RATE = r'''
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize(text: str) -> str:
    """Finetune.ipynb's target normalisation: the characters the tokenizer lacks, mapped onto ones
    fold.py scores as identical."""
    return (text.replace("।", ".").translate(DEV_DIGITS).replace("—", "-")
            .replace("‍", "").replace("‌", ""))


with ftkit.timed("reading train's labels"):
    splits = ftkit.load_splits(ftkit.download_dataset())
rates = sorted(len(tk.multi.encode(normalize(r["text"]), out_type=int)) / ftkit.duration(r)
               for r in splits["train"])
RATE_RANGE = (rates[0], rates[-1])
print(f"train: {len(rates)} clips, {RATE_RANGE[0]:.2f} to {RATE_RANGE[1]:.2f} teacher tokens/s "
      f"(median {rates[len(rates) // 2]:.2f})")
'''

DECODE_NOTE = """
## Step 2: decode, shard by shard

Greedy, fp32 weights under bf16 autocast (the decoding Finetune.ipynb scores with). Each clip
keeps its text, its token count and its mean log-prob over the tokens up to and including EOS,
and whether its output loops. A shard is uploaded as soon as it is decoded.
"""

DECODE = r'''
def transcribe_scored(rows):
    """Greedy decode of a batch, with each clip's mean log-prob and token count."""
    audio = [store.clip_f32(r) for r in rows]
    wav = torch.zeros(len(rows), ftkit.pad_len(max(len(a) for a in audio), PAD_TO_S))
    for i, a in enumerate(audio):
        wav[i, :len(a)] = a
    lens = torch.tensor([len(a) for a in audio], device="cuda")
    feats, flens = featurize(wav.cuda(), lens)  # fp32: it disables autocast itself
    mask = (torch.arange(feats.size(2), device="cuda")[None, :] < flens[:, None]).long()
    prompt = torch.tensor([PROMPT] * len(rows), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(input_features=feats, attention_mask=mask, decoder_input_ids=prompt,
                             do_sample=False, num_beams=1, eos_token_id=EOS, pad_token_id=PAD,
                             max_new_tokens=MAX_NEW_TOKENS, output_scores=True,
                             return_dict_in_generate=True)
    logprobs = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float()
    new = out.sequences[:, -logprobs.shape[1]:]
    eos = new == EOS
    counted = (eos.cumsum(1) - eos.long()) == 0  # every step up to and including the first EOS
    mean_lp = (logprobs.nan_to_num(0.0) * counted).sum(1) / counted.sum(1).clamp(min=1)
    n_tokens = counted.sum(1) - eos.any(1).long()
    texts = [tk.decode(tk.strip_prompt_and_trim(o.tolist(), PROMPT)) for o in out.sequences]
    return [{"text": t.strip(), "n_tokens": int(n), "mean_logprob": round(float(lp), 5),
             "looped": ftkit.is_loop(t)}
            for t, n, lp in zip(texts, n_tokens.tolist(), mean_lp.tolist(), strict=True)]


def shard_names():
    prefix = f"{LABELS}/shards/"
    return sorted(f for f in api.list_repo_files(DATASET_REPO, repo_type="dataset") if f.startswith(prefix))


done = set()
for name in shard_names():
    for line in Path(hf_hub_download(DATASET_REPO, name, repo_type="dataset", token=TOKEN)).read_text("utf-8").splitlines():
        done.add(json.loads(line)["segment_id"])
todo = [c for c in clips if c["segment_id"] not in done]
print(f"{len(done)} clips already labelled, {len(todo)} to decode "
      f"({sum(ftkit.duration(c) for c in todo) / 3600:.1f} h)")
first = len(shard_names())
for k in range(0, len(todo), SHARD_CLIPS):
    part, number = todo[k:k + SHARD_CLIPS], first + k // SHARD_CLIPS + 1
    rows, compute = [], 0.0
    with ftkit.timed(f"shard {number}: decoding {len(part)} clips"):
        for batch in ftkit.bucket_batches(part, budget_s=DECODE_BUDGET_S, max_items=DECODE_ITEMS,
                                          pad_to_s=PAD_TO_S, shuffle=False):
            chunk = [part[i] for i in batch]
            for c, r in zip(chunk, transcribe_scored(chunk), strict=True):
                rows.append({"segment_id": c["segment_id"], "source_id": c["source_id"],
                             "duration": round(ftkit.duration(c), 3), **r, "teacher": TEACHER})
    looped = sum(r["looped"] for r in rows)
    print(f"shard {number}: {looped} looped, mean log-prob "
          f"{sum(r['mean_logprob'] for r in rows) / len(rows):.3f}")
    local = OUT / "shards" / f"part-{number:05d}.jsonl"
    local.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    with ftkit.timed(f"shard {number}: uploading"):
        api.upload_file(path_or_fileobj=str(local), path_in_repo=f"{LABELS}/shards/{local.name}",
                        repo_id=DATASET_REPO, repo_type="dataset",
                        commit_message=f"{LABELS}: shard {number} ({len(rows)} clips)")
'''

FILTER_NOTE = """
## Step 3: filter, and what the teacher is sure of

Every shard is read back from the repo, so this cell works on whatever has been labelled so far.
The kept labels are what a student trains on; the report says what each rule dropped and how
confident the teacher was, split by the share of English words (the corpus's CMI buckets).
"""

FILTER = r"""
rows = []
for name in shard_names():
    path = Path(hf_hub_download(DATASET_REPO, name, repo_type="dataset", token=TOKEN))
    rows += [json.loads(line) for line in path.read_text("utf-8").splitlines()]
kept, report = distill.filter_pseudo(rows, tokens_per_s=RATE_RANGE, drop_fraction=DROP_FRACTION)
kept_ids = {r["segment_id"] for r in kept}
by_mix = {}
for r in rows:
    b = by_mix.setdefault(distill.mixing_bucket(distill.latin_share(r["text"])), {"clips": 0, "kept": 0, "lp": 0.0, "h": 0.0})
    b["clips"] += 1
    b["kept"] += r["segment_id"] in kept_ids
    b["lp"] += r["mean_logprob"]
    b["h"] += r["duration"] / 3600
report["by_english_share"] = {k: {"clips": v["clips"], "hours": round(v["h"], 2),
                                  "kept": round(v["kept"] / v["clips"], 3),
                                  "mean_logprob": round(v["lp"] / v["clips"], 4)}
                              for k, v in sorted(by_mix.items())}
report["teacher"], report["rate_range"] = TEACHER, list(RATE_RANGE)
print(json.dumps(report, indent=1))
(OUT / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), "utf-8")
(OUT / "report.json").write_text(json.dumps(report, indent=1), "utf-8")
with ftkit.timed("uploading the filtered labels and the report"):
    for f in ("labels.jsonl", "report.json"):
        api.upload_file(path_or_fileobj=str(OUT / f), path_in_repo=f"{LABELS}/{f}", repo_id=DATASET_REPO,
                        repo_type="dataset", commit_message=f"{LABELS}: step 3, {report['kept']} clips kept")
"""

cells = [
    md(INTRO),
    md("## Config"),
    code(CONFIG),
    md("## Setup"),
    code(SETUP),
    code("%%writefile /content/ft/ftkit.py\n" + FTKIT),
    code("%%writefile /content/ft/distill.py\n" + DISTILLKIT),
    code(DATA),
    md(RATE_NOTE),
    code(RATE),
    md(DECODE_NOTE),
    code(DECODE),
    md(FILTER_NOTE),
    code(FILTER),
]

if __name__ == "__main__":
    path = OUT_DIR / "Teacher.ipynb"
    path.write_text(
        json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote", path, len(cells), "cells")
