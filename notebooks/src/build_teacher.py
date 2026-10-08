"""Build notebooks/05_Teacher.ipynb: `python notebooks/src/build_teacher.py`. Step 5 of the
protocol (D105; roadmap §B steps 2-3): the frozen teacher transcribes the unlabelled distillation
corpus (D101), and its labels are filtered. Shared code lives in ftkit.py and distill.py, written
out by %%writefile cells."""

import sys
from pathlib import Path

import nbkit
from nbkit import code, md

HERE = Path(__file__).parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent

INTRO = """
# 05 — The teacher labels the unlabelled corpus

Step 5 of the protocol (D105; roadmap §B, steps 2-3). The teacher transcribes every clip of the
unlabelled corpus that `04_PreDistill.ipynb` cut into `DATASET_REPO/distill/` (D101), and its
labels are filtered before any student trains on them. Never the paid routes: the teacher on this
GPU is the only model.

**Teacher.** The model `03e_Flex_Ship.ipynb` froze: `FLEX_REPO/teacher.json` names it, and this
notebook reads that file rather than naming a model of its own. Greedy decoding, with per-token
log-probs kept. There is no loop retry: the filter drops every clip whose greedy output loops, so
retrying would spend GPU on labels that are thrown away. The labels go to a folder named after the
teacher, so another teacher's labels never mix with these.

**The filter, cheapest rule first** (`distill.filter_pseudo`): drop a clip overlapped for more
than `MAX_OVERLAP_SHARE` of its length (crosstalk is where the teacher is weakest; 04 measured it);
a clip whose output loops; an empty transcript; a rate outside the tokens per second train's
labels span, in the teacher's own tokens; then the least confident `DROP_FRACTION` of what is
left, by mean log-prob.

**Choose `MAX_OVERLAP_SHARE` from 04's table**, which prices each threshold in hours per channel.
It is `None` until then, which drops nothing for overlap. Every clip is decoded whatever the
threshold (about one GPU-hour per 100 h), so changing it later reruns only the filter cell.

**The report** splits what was kept by channel, by overlap bucket and by the share of English
words: every student fell furthest behind Flex on pure Nepali, and a threshold that empties the
round-table channels leaves a corpus of few voices.

**Small blast radius.** Labels go to `DATASET_REPO/<LABELS>/shards/` every `SHARD_CLIPS` clips, and
a rerun skips every clip already in a shard, so a lost runtime costs at most one shard.

**Smoke run first.** `SMOKE = True` reads `teacher-smoke.json` (or `teacher.json` when no smoke
03e has run), decodes the first 500 clips and
writes under `distill/pseudo-smoke/`. The log-prob masking and the memory that `output_scores`
takes have not run on a GPU until it has.
"""

CONFIG = r"""
DATASET_REPO = "Sagyam/nepanglish-asr"     # 04_PreDistill.ipynb wrote the corpus to its distill/
PREFIX = "distill"
FLEX_REPO = "Sagyam/nepanglish-asr-flex-ft"   # its teacher.json names the model (03e_Flex_Ship.ipynb)
LANG, MODE = "ne", "mixed"
SMOKE = False              # True: teacher-smoke.json if any, the first 500 clips, labels under pseudo-smoke/
LIMIT = None               # decode only the first N clips; None for the whole corpus
SHARD_CLIPS = 2000         # clips per uploaded shard: a lost runtime loses at most one
DECODE_BUDGET_S, DECODE_ITEMS = 1200.0, 64
PAD_TO_S = 1.0
MAX_NEW_TOKENS = 300       # as the decoder gold is scored with
MAX_OVERLAP_SHARE = None   # drop a clip overlapped for more than this share; None drops none. From 04's table
DROP_FRACTION = 0.15       # the least confident share dropped (roadmap: 10-20%)
"""

SETUP_TAIL = r"""
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"""

DATA = r"""
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

# A smoke run reads teacher-smoke.json if a smoke 03e wrote one, else the real teacher; its labels
# still go under pseudo-smoke/.
TEACHER_FILE = distill.smoke_source(SMOKE, "teacher-smoke.json", "teacher.json",
                                    lambda path: api.file_exists(FLEX_REPO, path))
teacher = json.loads(Path(hf_hub_download(FLEX_REPO, TEACHER_FILE, token=TOKEN)).read_text("utf-8"))
TEACHER = teacher["run"]
LABELS = f"{PREFIX}/{'pseudo-smoke' if SMOKE else 'pseudo'}/{TEACHER.replace('/', '--')}"
if SMOKE:
    LIMIT = LIMIT or 500
OUT = FT / "out" / LABELS
(OUT / "shards").mkdir(parents=True, exist_ok=True)
print("teacher:", TEACHER, f"(val {teacher['val_wer']:.2f}, gold {teacher['gold_wer']:.2f}) | labels:", LABELS)

with ftkit.timed("downloading the corpus"):
    CORPUS = Path(snapshot_download(DATASET_REPO, repo_type="dataset", token=TOKEN,
                                    allow_patterns=[f"{PREFIX}/clips.jsonl", f"{PREFIX}/episodes/*"]))
clips = [json.loads(line) for line in (CORPUS / PREFIX / "clips.jsonl").read_text("utf-8").splitlines() if line]
unmeasured = sum(c.get("overlap_share") is None for c in clips)
if unmeasured:
    print(f"{unmeasured} of {len(clips)} clips have no overlap measured: run 04_PreDistill.ipynb's overlap "
          "pass first, or the overlap rule keeps them all")
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
      f"{len({c['source_id'] for c in clips})} sources")
"""

RATE_NOTE = """
## The rate train's labels span, in the teacher's tokens

The filter's rate rule keeps a clip whose tokens per second fall inside the range the verified
train labels span when the teacher's tokenizer writes them, as `03a_Flex_Train.ipynb` encodes its
targets. The labels are the export the teacher was trained on.
"""

RATE = r'''
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize(text: str) -> str:
    """03a_Flex_Train.ipynb's target normalisation: the characters the tokenizer lacks, mapped onto
    ones fold.py scores as identical."""
    return (text.replace("।", ".").translate(DEV_DIGITS).replace("—", "-")
            .replace("‍", "").replace("‌", ""))


with ftkit.timed("reading train's labels"):
    labelled = ftkit.download_dataset()
    splits = ftkit.load_splits(labelled)
exported = json.loads((labelled / "training" / "manifest.json").read_text())["exported_at"]
if exported != teacher["dataset_export"]:
    print(f"the labelled export on the hub ({exported}) is not the one the teacher trained on "
          f"({teacher['dataset_export']}); the rate range below is this export's")
rates = sorted(len(tk.multi.encode(normalize(r["text"]), out_type=int)) / ftkit.duration(r)
               for r in splits["train"])
RATE_RANGE = (rates[0], rates[-1])
print(f"train: {len(rates)} clips, {RATE_RANGE[0]:.2f} to {RATE_RANGE[1]:.2f} teacher tokens/s "
      f"(median {rates[len(rates) // 2]:.2f})")
'''

DECODE_NOTE = """
## Decode, shard by shard

Greedy, fp32 weights under bf16 autocast (the decoding gold is scored with). Each clip keeps its
text, its token count and its mean log-prob over the tokens up to and including EOS, and whether
its output loops. A shard is uploaded as soon as it is decoded.
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
## Filter, and what is left to train on

Every shard is read back from the repo, so this cell works on whatever has been labelled so far,
and it is the only cell to run again after changing `MAX_OVERLAP_SHARE` or `DROP_FRACTION`. Each
label is joined with its clip's times, channel and overlap, so `labels.jsonl` is all a student
notebook needs beside the audio. The report says what each rule dropped, and what was kept per
channel, per overlap bucket and per share of English words.
"""

FILTER = r"""
by_id = {c["segment_id"]: c for c in clips}
rows = []
for name in shard_names():
    path = Path(hf_hub_download(DATASET_REPO, name, repo_type="dataset", token=TOKEN))
    for line in path.read_text("utf-8").splitlines():
        r = json.loads(line)
        if (c := by_id.get(r["segment_id"])) is not None:  # a clip outside LIMIT is left out
            rows.append({**r, **{k: c.get(k) for k in ("episode_id", "channel", "start_time", "end_time",
                                                       "overlap_share")}})
kept, report = distill.filter_pseudo(rows, tokens_per_s=RATE_RANGE, drop_fraction=DROP_FRACTION,
                                     max_overlap_share=MAX_OVERLAP_SHARE)
kept_ids = {r["segment_id"] for r in kept}


def tally(key):
    # clips, hours and kept hours per value of `key(row)`, and the teacher's mean confidence
    out = {}
    for r in rows:
        b = out.setdefault(key(r), {"clips": 0, "hours": 0.0, "kept_hours": 0.0, "lp": 0.0})
        b["clips"] += 1
        b["hours"] += r["duration"] / 3600
        b["kept_hours"] += r["duration"] / 3600 * (r["segment_id"] in kept_ids)
        b["lp"] += r["mean_logprob"]
    return {str(k): {"clips": v["clips"], "hours": round(v["hours"], 2), "kept_hours": round(v["kept_hours"], 2),
                     "mean_logprob": round(v["lp"] / v["clips"], 4)}
            for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}


report["by_channel"] = tally(lambda r: r["channel"])
report["by_overlap"] = tally(lambda r: distill.overlap_bucket(r["overlap_share"]))
report["by_english_share"] = tally(lambda r: distill.mixing_bucket(distill.latin_share(r["text"])))
report["teacher"], report["rate_range"] = TEACHER, list(RATE_RANGE)
print(json.dumps({k: v for k, v in report.items() if not k.startswith("by_")}, indent=1))
for name in ("by_channel", "by_overlap", "by_english_share"):
    print(f"\n{name[3:]:<40} {'clips':>7} {'hours':>7} {'kept h':>7} {'kept':>6} {'log-prob':>9}")
    for value, b in report[name].items():
        print(f"  {value:<38} {b['clips']:>7} {b['hours']:>7.1f} {b['kept_hours']:>7.1f} "
              f"{b['kept_hours'] / max(b['hours'], 1e-9):>6.0%} {b['mean_logprob']:>9.3f}")
(OUT / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), "utf-8")
(OUT / "report.json").write_text(json.dumps(report, indent=1), "utf-8")
with ftkit.timed("uploading the filtered labels and the report"):
    for f in ("labels.jsonl", "report.json"):
        api.upload_file(path_or_fileobj=str(OUT / f), path_in_repo=f"{LABELS}/{f}", repo_id=DATASET_REPO,
                        repo_type="dataset", commit_message=f"{LABELS}: filtered, {report['kept']} clips kept")
"""

COUNTS_NOTE = """
## The training corpus's word counts (D123)

A student trains on the human train labels (stage 1) and on these labels (stage 2). The evaluation
splits every error by how often its word occurs in the two together (`evalkit.rarity`: never, 1-9,
10-49, 50-99, 100+), so the counts are made here, the moment the labels are final, and go beside
them and to the dataset's `harness/word_counts.json`, which every notebook's download already
brings. A smoke run writes them beside its own labels only. Run this cell again after the filter.
"""

COUNTS = r"""
from huggingface_hub import CommitOperationAdd

import evalkit

DATA = ftkit.download_dataset()          # the labels and the scorer, no audio
score = ftkit.harness_scorer(DATA, FT)   # puts the dataset's fold.py and error mining on the path
export = json.loads((DATA / "training" / "manifest.json").read_text())
counts = evalkit.count_training_words(OUT / "word_counts.json", ftkit.load_splits(DATA)["train"], kept,
                                      export=export["exported_at"], labels=LABELS)
remote = [f"{LABELS}/word_counts.json"] + ([] if SMOKE else [evalkit.WORD_COUNTS_FILE])
with ftkit.timed(f"uploading the word counts to {', '.join(remote)}"):
    api.create_commit(repo_id=DATASET_REPO, repo_type="dataset",
                      operations=[CommitOperationAdd(r, str(counts)) for r in remote],
                      commit_message=f"{LABELS}: the training corpus's word counts")
"""

cells = [
    md(INTRO),
    md("## Config"),
    code(CONFIG),
    md("## Setup"),
    nbkit.setup("rapidfuzz duckdb", tail=SETUP_TAIL),
    *nbkit.kits("ftkit", "distill", "sweep", "evalkit"),
    code(DATA),
    md(RATE_NOTE),
    code(RATE),
    md(DECODE_NOTE),
    code(DECODE),
    md(FILTER_NOTE),
    code(FILTER),
    md(COUNTS_NOTE),
    code(COUNTS),
]

NOTEBOOKS = {"05_Teacher.ipynb": cells}

if __name__ == "__main__":
    nbkit.write(NOTEBOOKS, OUT_DIR)
