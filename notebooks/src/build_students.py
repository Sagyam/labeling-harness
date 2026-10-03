"""Build the student notebooks, 06a onwards: `python notebooks/src/build_students.py` (D105, D106).

One notebook per student, each the same three stages: fine-tune on the human labels, continue
those weights on the teacher's pseudo-labels mixed with the human labels, and do that again with
the augmentation that won on Flex. Every stage is scored through evalkit, on gold and val and
then on the public sets. What differs between notebooks is one cell: how the student is loaded,
what its loss is and how it decodes. Shared code lives in ftkit.py, evalkit.py, sweep.py,
distill.py, xtalk.py and augment.py, written out by %%writefile cells."""

import ast
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import nbkit
from build_flex import AUG_KIT as FLEX_AUG_KIT
from build_flex import SETS as PUBLIC_SETS
from nbkit import GPU_NOTE, SMOKE_NOTE, code, md

HERE = Path(__file__).parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent

# --- what every student notebook shares ----------------------------------------------------------

PROTOCOL = """
**Three stages, the same for every student** (D105, D106), each a cell below:
1. **human.** The student, from its pretrained weights, is fine-tuned on the verified train labels
   alone. This is the point with no pseudo-labels, and the baseline distillation has to beat.
2. **distill.** The weights of stage 1 continue on a mixture: the teacher's filtered
   pseudo-labels (`05_Teacher.ipynb`) and the human labels. The human labels take `HUMAN_SHARE` of
   every epoch's draws; the rest is split between the unlabelled channels by the square root of
   their hours, so one deep playlist does not carry the corpus. Val is still the human val split,
   and gold is never trained on.
3. **distill-aug.** Stage 2 again, from stage 1's weights, with the augmentation recipe that won
   Flex's ablation (`03c_Flex_Augment.ipynb`) switched on. When vanilla won there, or the ablation
   was not run, there is nothing to switch on and the stage is skipped.

**Why stage 2 continues stage 1** (D106, the owner's choice on 2026-09-30, reversing the
fresh-weights rule of 2026-09-26). Each student is trained once on the human labels and carried
forward, instead of being trained again from its pretrained weights on the mixture. It costs
less, and it means stage 2's gain is measured on a model that has already seen the human labels:
the difference between the two stages is the pseudo-labels plus the extra training, not the
pseudo-labels alone. The human labels stay in stage 2's mixture, so the model does not end on the
teacher's errors only.

**The teacher** is whatever `FLEX_REPO/teacher.json` names (`03e_Flex_Ship.ipynb`). Its labels are
read from the folder named after it, and every stage is paired against it clip by clip.

**One evaluation** (`evalkit`, the same code as Flex's). Gold and val, folded and raw, split into
S, D and I, overall and per clip class, and paired against the teacher and the earlier stages
with episodes resampled. Then the five public Nepali sets, where a student that never knew Nepali
has nothing to forget: for it they measure how far the training generalises.

**The success bar** (roadmap §B): a student whose interval against the teacher contains zero, or
lies below it, on gold and val, class by class. Crosstalk classes are read with care: clips
overlapped beyond `MAX_OVERLAP_SHARE` were left out of the pseudo-labels in step 5.

**Small blast radius.** A stage's best weights go to `OUT_REPO` as soon as training ends, before
scoring; a stage whose result is there is skipped; a stage whose weights are there but not its
scores is scored without training again. During training a resume point is saved to `RESUME_REPO`
after every epoch, so a lost runtime costs the epoch in progress. The cells are independent: after
a kernel restart, run everything above the stages and then the stage you were on.
"""

COMMON_CONFIG = r"""
RUN_PREFIX = "students-2026-09-30"   # OUT_REPO/<RUN_PREFIX>/<student>-<stage>/
DATASET_EXPORT = "2026-09-30"        # the export every model trains and is scored on; another is refused
DATASET_REPO = "Sagyam/nepanglish-asr"
OUT_REPO = "Sagyam/nepanglish-asr-students"   # private HF model repo, created if missing
RESUME_REPO = "Sagyam/nepanglish-asr-resume"  # scratch: one resume point per unfinished stage
FLEX_REPO = "Sagyam/nepanglish-asr-flex-ft"   # teacher.json, and the ablation's winning recipe
SMOKE = False             # True: a few hundred clips, one epoch, under <RUN_PREFIX>-smoke
PATIENCE, WARMUP = 3, 0.1
MIN_DELTA = 0.2           # val WER points PATIENCE epochs must gain between them, or training stops
HUMAN_SHARE = 0.5         # of each distillation epoch's draws that are human-labelled clips
EFFECTIVE_S = 720.0       # ~12 min of audio per optimizer step
PAD_TO_S = 1.0
PROBE_FRACTION = 0.9
EVAL_BUDGET_S, EVAL_ITEMS = 1200.0, 96
DEVICE = "cuda"
DATA_LOCAL = None         # a local export in the HF layout instead of the download
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"   # stage 3, if the recipe adds noise
NOISE_SECONDS = 60        # of each MUSAN file
"""

DATA = r"""
import gc
import glob
import traceback

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

import distill
import evalkit
import ftkit
import sweep

ftkit.fast_cuda()
TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)
for repo in (OUT_REPO, RESUME_REPO):
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)

DATA = ftkit.download_dataset(DATA_LOCAL)  # labels and the scorer; the audio follows
export = json.loads((DATA / "training" / "manifest.json").read_text())
evalkit.check_export(export, DATASET_EXPORT)
splits = ftkit.load_splits(DATA)
TOKENIZER_TEXTS = distill.tokenizer_texts(splits)  # train labels only, before any smoke subset
if SMOKE:
    splits = {"train": splits["train"][::50], "val": splits["val"][::30], "gold": splits["gold"][::16]}
episodes = [r["episode_id"] for rows in splits.values() for r in rows]
with ftkit.timed(f"downloading {len(set(episodes))} recordings"):
    DATA = ftkit.download_dataset(DATA_LOCAL, episodes=episodes, analytics=True)
with ftkit.timed("reading them into RAM"):
    store = ftkit.AudioStore(DATA, episodes)
score = ftkit.harness_scorer(DATA, FT)
print({k: len(v) for k, v in splits.items()}, f"| audio in RAM: {store.gib:.1f} GiB | export",
      export["exported_at"][:10], "|", score.fold_version)

# The teacher: the model 03e froze. Its per-clip counts are what every stage is paired against.
teacher = json.loads(Path(hf_hub_download(FLEX_REPO, "teacher-smoke.json" if SMOKE else "teacher.json",
                                          token=TOKEN)).read_text("utf-8"))
teacher_counts = json.loads(Path(hf_hub_download(FLEX_REPO, f"{teacher['run']}/per_clip.json",
                                                 token=TOKEN)).read_text("utf-8"))
LABELS = f"distill/{'pseudo-smoke' if SMOKE else 'pseudo'}/{teacher['run'].replace('/', '--')}"
print("teacher:", teacher["run"], f"(val {teacher['val_wer']:.2f}, gold {teacher['gold_wer']:.2f})",
      "| its labels:", LABELS)
if teacher["dataset_export"] != export["exported_at"]:
    print(f"the teacher was trained on the {teacher['dataset_export']} export, not this one: its pairing "
          "covers only the clips both have")


def load_pseudo():
    # The teacher's filtered labels as training rows, with their recordings added to `store`.
    global pseudo
    if "pseudo" in globals():
        return pseudo
    path = hf_hub_download(DATASET_REPO, f"{LABELS}/labels.jsonl", repo_type="dataset", token=TOKEN)
    labels = [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line]
    if SMOKE:
        labels = labels[:300]
    sources = sorted({r["episode_id"] for r in labels})
    clash = set(sources) & set(store.audio)
    assert not clash, f"an unlabelled recording shares its id with a labelled one: {sorted(clash)[:5]}"
    with ftkit.timed(f"downloading {len(sources)} unlabelled recordings"):
        root = Path(snapshot_download(DATASET_REPO, repo_type="dataset", token=TOKEN,
                                      allow_patterns=[f"distill/episodes/{glob.escape(s)}.flac" for s in sources]))
    with ftkit.timed("reading them into RAM"):
        store.audio.update(ftkit.AudioStore(root, sources, folder="distill/episodes").audio)
    pseudo = [{"segment_id": r["segment_id"], "episode_id": r["episode_id"], "start_time": r["start_time"],
               "end_time": r["end_time"], "text": r["text"], "channel": r["channel"], "pseudo": True}
              for r in labels]
    hours = sum(ftkit.duration(r) for r in pseudo) / 3600
    print(f"pseudo-labels: {len(pseudo)} clips, {hours:.1f} h, {len({r['channel'] for r in pseudo})} channels "
          f"| audio in RAM: {store.gib:.1f} GiB")
    return pseudo
"""

AUG_KIT = FLEX_AUG_KIT.replace("from IPython.display import Audio, display\n", "")
assert "IPython" not in AUG_KIT

STAGES_NOTE = """
## The stages

Everything a stage does, shared by every student: the batch probe, the speed check (it projects
the run and gives val WER before training; **read its line before walking away**), training with
early stopping on val and a resume point per epoch, the best weights uploaded, gold and val
scored and paired, then the public sets.
"""

STAGES = r'''
OUT = OUT_ROOT
AUG, STRETCH = None, 1.0  # stage 3's augmenter, and how much longer a slowed clip can be (batch sizing)
monitor = ftkit.GpuMonitor().start()
STAGE_NOTE = {
    "human": "fine-tuned on the verified train labels",
    "distill": "stage 1 continued on the teacher's pseudo-labels mixed with the human labels",
    "distill-aug": "stage 2 from stage 1's weights, with the augmentation that won Flex's ablation",
}


def line(m):
    return f"{m['wer']:6.2f} ({ftkit.sid(m)})"


def fetch_json(remote, repo=None):
    """A JSON file of OUT_REPO (or `repo`), or None when it is not there."""
    repo = repo or OUT_REPO
    if not api.file_exists(repo, remote):
        return None
    return json.loads(Path(hf_hub_download(repo, remote, token=TOKEN)).read_text("utf-8"))


def uploaded(run):
    """A run's result row: OUT_REPO has it once the stage is trained, scored and uploaded."""
    return fetch_json(f"{RUN_PREFIX}/{run}/result.json")


def run_counts(run):
    return fetch_json(f"{RUN_PREFIX}/{run}/per_clip.json")


def upload(out, run, message, weights):
    with ftkit.timed(f"uploading {run}: {message}"):
        api.upload_folder(repo_id=OUT_REPO, folder_path=str(out), path_in_repo=f"{RUN_PREFIX}/{run}",
                          ignore_patterns=None if weights else ["best/*"], commit_message=f"{run}: {message}")


def download_weights(run):
    """`<run>/best/` of OUT_REPO as a local folder `load_student` reads."""
    with ftkit.timed(f"downloading the weights of {run}"):
        root = snapshot_download(OUT_REPO, allow_patterns=[f"{RUN_PREFIX}/{run}/best/*"], token=TOKEN)
    return Path(root) / RUN_PREFIX / run / "best"


def free_model():
    for var in ("optimizer", "model"):
        globals().pop(var, None)
    gc.collect()
    torch.cuda.empty_cache()


def train_batch(rows):
    """The int16 clips of a training micro-batch and the rows whose targets they carry: the clips
    as they are, or through the stage's augmenter. When the augmenter rewrites a label (two voices,
    everything said) the row is the student's `retarget` of it; a label the student cannot write
    leaves the clip clean."""
    if AUG is None:
        return [store.clip(r) for r in rows], list(rows)
    # torch reseeds each DataLoader worker per epoch, so every epoch draws fresh augmentations
    rng = np.random.default_rng(torch.randint(2**62, (1,)).item())
    clips, targets = [], []
    for r in rows:
        clean = store.clip(r)
        audio, info = AUG(r, clean, rng)
        target = r if info.get("text") is None else retarget(r, info["text"])
        clips.append(clean if target is None else audio)
        targets.append(r if target is None else target)
    return clips, targets


def decode(rows):
    # `decoder()` is the student's: greedy for a transducer or CTC, greedy + loop retry otherwise
    model.eval()
    d = decoder()
    with torch.no_grad():
        texts, compute = ftkit.transcribe_rows(rows, d, budget_s=EVAL_BUDGET_S, max_items=EVAL_ITEMS,
                                               pad_to_s=PAD_TO_S)
    return texts, compute, getattr(d, "log", [])


def evaluate(model, rows=None):
    rows = rows or splits["val"]
    texts, _, _ = decode(rows)
    return score([r["text"] for r in rows], texts)


def card(stage):
    return {
        "name": f"{STUDENT} ({stage})",
        "description": f"{RUN_PREFIX}: {STUDENT}, {STAGE_NOTE[stage]}",
        "architecture": ARCHITECTURE,
        "base_model": BASE_MODEL,
        "decoder": DECODER_NAME,
        "lr": LR_CARD,
        "notebook": NOTEBOOK,
        "teacher": teacher["run"],
        "dataset_export": export["exported_at"],
        "train_export": {k: export.get(k) for k in ("exported_at", "git_commit", "row_count",
                                                    "normalization_version", "label_version")},
        "fold_version": score.fold_version,
        "kits": evalkit.kit_digests(FT),
    }


def winning_recipe():
    """The augmentation stages of Flex's ablation winner, as augment.AugmentConfig.from_dict reads
    them: {} when vanilla won, or when 03c was not run."""
    ablation = fetch_json(f"{teacher['run'].rsplit('/', 1)[0]}/ablation.json", FLEX_REPO)
    if not ablation:
        return {}
    winner = next(r for r in ablation["rows"] if r["run_name"] == ablation["winner"])
    return winner.get("stages") or {}


def train_stage(stage, run, recipe):
    """Train one stage and upload its best weights. Returns what `trained.json` records."""
    global AUG, STRETCH, optimizer
    load_student(None if stage == "human" else download_weights(f"{STUDENT}-human"))
    weights, pseudo_hours = None, 0.0
    if stage == "human":
        rows = prepare_rows(list(splits["train"]))
    else:
        rows = prepare_rows(list(splits["train"]) + load_pseudo())
        weights = distill.mixture_weights([r["channel"] if r.get("pseudo") else None for r in rows],
                                          [ftkit.duration(r) / 3600 for r in rows], HUMAN_SHARE)
        pseudo_hours = sum(ftkit.duration(r) for r in rows if r.get("pseudo")) / 3600
    print(f"{run}: {len(rows)} clips the student can train on, {pseudo_hours:.1f} h of them pseudo-labelled")
    optimizer = make_optimizer(model)
    budget, items = probe(model, optimizer, rows)
    if recipe:
        AUG, STRETCH = make_augmenter(recipe)

    def make_batches(epoch):
        if weights is None:
            return ftkit.bucket_batches(rows, budget_s=budget, max_items=items, pad_to_s=PAD_TO_S,
                                        shuffle=True, seed=epoch, stretch=STRETCH)
        # an epoch of the mixture: as many draws as it has clips, by weight, repeatable from the epoch
        drawn = distill.draw_epoch(weights, len(rows), seed=epoch)
        batches = ftkit.bucket_batches([rows[i] for i in drawn], budget_s=budget, max_items=items,
                                       pad_to_s=PAD_TO_S, shuffle=True, seed=epoch, stretch=STRETCH)
        return [[drawn[j] for j in b] for b in batches]

    epochs = 1 if SMOKE else EPOCHS["human" if stage == "human" else "distill"]
    resume = ftkit.HubResume(api, RESUME_REPO, f"{RUN_PREFIX}/{run}", FT / "resume" / run)
    if not api.file_exists(RESUME_REPO, resume.remote):
        # Val before training on every 12th val clip: with fresh heads it is gibberish, and only its
        # timing matters. The projection's val passes are for that sample, so the full-val time follows.
        sample = splits["val"][::12]
        speed = ftkit.speed_check(model, rows=rows, batches=make_batches(0), collate=collate,
                                  loss_fn=loss_fn, evaluate=lambda m: evaluate(m, sample), val_rows=sample,
                                  gold_rows=splits["gold"], epochs=epochs, monitor=monitor)
        full_val_min = speed["val_s"] * len(splits["val"]) / len(sample) / 60
        print(f"a full val pass takes about {full_val_min:.1f} min, {epochs} of them at most "
              f"{epochs * full_val_min / 60:.1f} h on top of training")
        (OUT / "speed_check.json").write_text(json.dumps(speed, indent=1))
    cfg = ftkit.TrainConfig(name=run, out=str(OUT), epochs=epochs, lr=PEAK_LR, schedule=SCHEDULE,
                            warmup_frac=WARMUP, effective_s=EFFECTIVE_S, patience=PATIENCE, min_delta=MIN_DELTA)
    result = ftkit.train(model, cfg=cfg, rows=rows, make_batches=make_batches, collate=collate,
                         loss_fn=loss_fn, evaluate=evaluate, save_best=lambda m: save_weights(m, OUT / "best"),
                         optimizer=optimizer, monitor=monitor, resume=resume)
    AUG, STRETCH = None, 1.0
    evals = [h for h in result["history"] if "val_wer" in h]
    trained = {"best_epoch": min(evals, key=lambda h: h["val_wer"])["epoch"] if evals else None,
               "epochs": epochs, "stopped_by_hand": result["stopped_by_hand"], "train_clips": len(rows),
               "pseudo_hours": round(pseudo_hours, 2), "human_share": HUMAN_SHARE if weights else None}
    (OUT / "trained.json").write_text(json.dumps(trained, indent=1))
    upload(OUT, run, "best weights, before scoring", weights=True)
    resume.clear()
    return trained


def run_stage(stage):
    """Train (unless already trained), score and upload one stage, then score it on the public
    sets. A stage that is wholly in OUT_REPO is skipped. A failure frees the GPU before it is
    raised, so another cell can run."""
    global OUT, AUG, STRETCH
    run = f"{STUDENT}-{stage}"
    if uploaded(run) is not None and done_sets(run) >= set(SETS):
        print(f"{run}: already in {OUT_REPO}, skipped")
        return
    recipe = {}
    if stage == "distill-aug":
        recipe = winning_recipe()
        if not recipe:
            print("Flex's ablation kept vanilla, or 03c was not run: no augmentation to switch on, no stage 3")
            return
    if stage != "human" and fetch_json(f"{RUN_PREFIX}/{STUDENT}-human/trained.json") is None:
        raise RuntimeError(f"{STUDENT}-human is not in {OUT_REPO}: run stage 1 first")
    OUT = OUT_ROOT / run
    OUT.mkdir(parents=True, exist_ok=True)
    failure, stopped = None, False
    try:
        print(f"\n=== {run}" + (f": {json.dumps(recipe)}" if recipe else ""))
        free_model()
        if uploaded(run) is None:
            trained = fetch_json(f"{RUN_PREFIX}/{run}/trained.json")
            if trained is None:
                trained = train_stage(stage, run, recipe)
            else:
                print(f"{run}: weights already in {OUT_REPO}; scoring them")
                load_student(download_weights(run))
            references = {"teacher": teacher_counts}
            for other in {"human": (), "distill": ("human",), "distill-aug": ("human", "distill")}[stage]:
                if (c := run_counts(f"{STUDENT}-{other}")) is not None:
                    references[f"{STUDENT}-{other}"] = c
            meta = {"student": STUDENT, "stage": stage, "recipe": stage, "seed": 0, "stages": recipe,
                    "teacher": teacher["run"], **trained}
            evalkit.evaluate_run(OUT, run, splits=splits, decode=decode, score=score, card=card(stage),
                                 meta=meta, references=references)
            upload(OUT, run, "scores on gold and val", weights=False)
        else:
            load_student(download_weights(run))
        score_sets(run)
    except BaseException as e:
        failure, stopped = traceback.format_exc(), isinstance(e, KeyboardInterrupt)
    finally:
        AUG, STRETCH = None, 1.0
        free_model()
    if stopped:
        raise KeyboardInterrupt(f"{run} stopped by hand outside training; the GPU is freed")
    if failure:
        print(failure)
        raise RuntimeError(f"{run} failed (traceback above); the GPU is freed for the next stage")
'''

RUN_NOTES = {
    "human": """
### Stage 1 — human labels

From the pretrained weights, on the verified train labels alone.
""",
    "distill": """
### Stage 2 — pseudo-labels and human labels

Stage 1's weights, continued on the mixture. It needs `05_Teacher.ipynb`'s filtered labels for the
frozen teacher.
""",
    "distill-aug": """
### Stage 3 — the same, with Flex's winning augmentation

From stage 1's weights again, so its difference to stage 2 is the augmentation alone. Skipped when
vanilla won Flex's ablation.
""",
}

REPORT_NOTE = """
## Report

Each stage against the teacher on the same clips: folded WER with S/D/I, then the paired
difference (stage minus reference) per clip class, then the public sets. Gold is the held-out
score: nothing here was chosen on it.
"""

REPORT = r"""
results = [r for r in (uploaded(f"{STUDENT}-{stage}") for stage in STAGE_NOTE) if r]
print(f"{'run':<26}{'epoch':>6}  {'val':<34}{'gold':<34}{'gold raw':>9}{'gold RTF':>10}")
for r in results:
    print(f"{r['run_name']:<26}{str(r.get('best_epoch') or '-'):>6}  {line(r['val']):<34}{line(r['gold']):<34}"
          f"{r['gold']['raw_wer']:9.2f}{r['gold']['rtf']:10.4f}")
for r in results:
    for split in ("val", "gold"):
        for name, vs in r[split]["vs"].items():
            cells = [f"{k} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]" for k, v in vs.items() if k != "clips"]
            print(f"\n{r['run_name']} {split} minus {name}, WER points [95% CI, episodes resampled], "
                  f"{vs['clips']} clips:\n    " + "\n    ".join(cells))
table, means = public_table([r["run_name"] for r in results], against=f"{STUDENT}-human")
summary = {"student": STUDENT, "teacher": teacher["run"], "stages": results, "public": table, "public_mean": means}
(OUT_ROOT / f"summary-{STUDENT}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
api.upload_file(path_or_fileobj=str(OUT_ROOT / f"summary-{STUDENT}.json"),
                path_in_repo=f"{RUN_PREFIX}/summary-{STUDENT}.json", repo_id=OUT_REPO,
                commit_message=f"{RUN_PREFIX}: {STUDENT} summary")
print("\nuploaded", f"https://huggingface.co/{OUT_REPO}/tree/main/{RUN_PREFIX}")
"""

# --- the transducers: IndicConformer and Parakeet (NeMo) -----------------------------------------

NEMO_CONFIG = r"""
PARAKEET_ID = "nvidia/parakeet-tdt-0.6b-v2"
INDIC_URL = ("https://objectstore.e2enetworks.net/indicconformer/models/"
             "indicconformer_stt_ne_hybrid_rnnt_large.nemo")
NEMO_VERSION = "3.0.0"    # the version smoke-tested on CPU
VOCAB_SIZE = 1024
# Per stage: the most epochs, which is also the length of the linear LR decay.
EPOCHS = {"human": __HUMAN_EPOCHS__, "distill": __DISTILL_EPOCHS__}
LR_ENCODER, LR_HEADS = __LR_ENCODER__, __LR_HEADS__
"""

CONFORMER_CONFIG = r"""
# The Conformer trained from scratch: IndicConformer's architecture (17 layers, d_model 512, 121M)
# at this size, with random weights. A first guess at a size 150 h can train.
CONFORMER = {"d_model": 256, "n_layers": 16, "n_heads": 4}
"""

CONFORMER_LOADER = r'''def load_conformer():
    """IndicConformer's architecture at a smaller size, with random weights: no pretraining at all,
    and the same tokenizer, heads and recipe as the two pretrained transducers. The checkpoint is
    downloaded for its config alone."""
    _, cfg = _indic_checkpoint()
    with open_dict(cfg):
        cfg.encoder.d_model = CONFORMER["d_model"]
        cfg.encoder.n_layers = CONFORMER["n_layers"]
        cfg.encoder.n_heads = CONFORMER["n_heads"]
        if "model_defaults" in cfg:  # what the constructor sizes the joint's encoder input from
            cfg.model_defaults.enc_hidden = CONFORMER["d_model"]
        cfg.joint.jointnet.encoder_hidden = CONFORMER["d_model"]
        cfg.aux_ctc.decoder.feat_in = CONFORMER["d_model"]
    model = EncDecHybridRNNTCTCBPEModel(cfg=cfg)
    print(f"Conformer from scratch: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, "
          "random weights")
    return model
'''

NEMO_SETUP_TAIL = r"""
import torch

# The transducer losses run as numba CUDA kernels. NeMo's own cu12 extra would also pin a
# different torch, so only the kernels' package is added, for this runtime's CUDA.
CUDA_MAJOR = torch.version.cuda.split(".")[0]
%pip install -q "numba-cuda[cu{CUDA_MAJOR}]"
"""

TOKENIZER_NOTE = """
## The shared tokenizer

SentencePiece BPE, 1,024 tokens, full character coverage, trained on train labels only
(`distill.tokenizer_texts` refuses a val or gold clip in train), and never on a pseudo-label: the
vocabulary is the verified corpus's. Unlike Flex's tokenizer it writes `।` and Devanagari digits,
so raw WER is not handicapped on the human labels. The teacher's labels hold neither, so a student
trained on both learns the two conventions at once; folded WER hides this and raw WER does not. A
clip with an unknown token is dropped from training; val and gold are scored whole.
"""

TOKENIZER = r"""
import statistics

import sentencepiece as spm

TOK_DIR = FT / f"tokenizer-{VOCAB_SIZE}"
TOK_DIR.mkdir(exist_ok=True)
corpus = FT / "tokenizer_train.txt"
corpus.write_text("\n".join(TOKENIZER_TEXTS), encoding="utf-8")
spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(TOK_DIR / "tokenizer"),
                               vocab_size=VOCAB_SIZE, model_type="bpe", character_coverage=1.0,
                               bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0, minloglevel=2)
with (TOK_DIR / "tokenizer.vocab").open(encoding="utf-8") as fh:
    (TOK_DIR / "vocab.txt").write_text("".join(line.split("\t")[0] + "\n" for line in fh), encoding="utf-8")
sp = spm.SentencePieceProcessor(model_file=str(TOK_DIR / "tokenizer.model"))
for name, rows in splits.items():
    ids = [sp.encode(r["text"]) for r in rows]
    tps = sorted(len(i) / ftkit.duration(r) for i, r in zip(ids, rows))
    print(f"{name:5s} tokens/s median {statistics.median(tps):.1f}, p99 {tps[int(0.99 * len(tps))]:.1f}, "
          f"max {tps[-1]:.1f} | clips with an unknown token: {sum(0 in i for i in ids)}")
"""

STUDENT_NOTE = """
## The student: loading, loss and decoding

Everything model-specific: what `train_stage` and the scoring call.
"""

NEMO_STUDENT = r'''
import io
import logging
import tarfile
import urllib.request

from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCBPEModel
from nemo.utils import logging as nemo_logging

nemo_logging.setLevel(logging.ERROR)


def _sum_losses(model):
    """Summed over clips, so ftkit.train can divide by the step's tokens (its contract)."""
    model.loss.reduction = "sum"
    if getattr(model, "ctc_loss_weight", 0) > 0:
        model.ctc_loss = CTCLoss(num_classes=model.ctc_decoder.num_classes_with_blank - 1,
                                 zero_infinity=True, reduction="sum")


def _greedy(model):
    """Greedy batch decoding on the transducer head, whatever the checkpoint shipped with."""
    cfg = model.cfg.decoding
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        # CUDA graphs off: a graph captured before the batch probe hit illegal memory accesses
        # after the probe's out-of-memory recoveries (A100, NeMo 3.0.0, 2026-09-24).
        if "greedy" not in cfg:
            cfg.greedy = {}
        cfg.greedy.use_cuda_graph_decoder = False
        # At most 3 tokens per encoder frame (NeMo's default is 10). The densest train label is
        # 12.8 tokens/s, about one per Parakeet frame, so this binds only on an untrained head,
        # whose thousands of tokens per clip took fold.py's alignment hours to score.
        cfg.greedy.max_symbols = 3
    if hasattr(model, "cur_decoder"):
        model.change_decoding_strategy(decoding_cfg=cfg, decoder_type="rnnt", verbose=False)
    else:
        model.change_decoding_strategy(decoding_cfg=cfg, verbose=False)


def _tdt_loss(model):
    """The loss the model's own constructor builds, from its `cfg.loss`.

    NeMo 3.0.0's BPE `change_vocabulary` rebuilds the loss as `RNNTLoss(num_classes=...)` with no
    loss name, which is plain RNN-T with its blank on the joint's *last* output. On a TDT joint
    that output is the longest duration, so the 2026-09-25 run learned "blank" as "skip 4 frames",
    and the TDT decoder, whose blank is before the durations, dropped ~40% of the words."""
    loss_name, loss_kwargs = model.extract_rnnt_loss_cfg(model.cfg.get("loss", None))
    num_classes = model.joint.num_classes_with_blank - 1
    if loss_name == "tdt":
        num_classes -= model.joint.num_extra_outputs
    model.loss = RNNTLoss(num_classes=num_classes, loss_name=loss_name, loss_kwargs=loss_kwargs)
    if model.joint.fuse_loss_wer:
        model.joint.set_loss(model.loss)
    assert model.loss._blank == model.tokenizer.vocab_size, "the loss's blank must be the decoder's"


def load_parakeet():
    """Parakeet-TDT v2 with our tokenizer: `change_vocabulary` keeps the preprocessor and the
    encoder, and builds a fresh prediction network and joint for the new vocabulary. Its loss is
    then rebuilt, because the one `change_vocabulary` leaves is not TDT (see `_tdt_loss`)."""
    model = ASRModel.from_pretrained(PARAKEET_ID, map_location="cpu")
    model.change_vocabulary(new_tokenizer_dir=str(TOK_DIR), new_tokenizer_type="bpe")
    _tdt_loss(model)
    return model


def _indic_checkpoint():
    """IndicConformer's `ne` checkpoint unpacked, and its config with our tokenizer and stock heads.

    The checkpoint only loads in AI4Bharat's NeMo fork: its tokenizer is a custom 22-language
    aggregate and its heads use a per-language softmax. Both are replaced anyway, so the model is
    built from the checkpoint's own config with those parts swapped for stock ones."""
    path = FT / "indicconformer_ne.nemo"
    if not path.exists():
        urllib.request.urlretrieve(INDIC_URL, path)
    unpacked = FT / "indicconformer_ne"
    if not unpacked.exists():
        with tarfile.open(path) as tf:
            tf.extractall(unpacked, filter="data")
    cfg = OmegaConf.load(unpacked / "model_config.yaml")
    with open_dict(cfg):
        cfg.tokenizer = {"dir": str(TOK_DIR), "type": "bpe"}
        for node in (cfg.decoder, cfg.joint, cfg.aux_ctc.decoder, cfg.decoding, cfg.aux_ctc.decoding):
            for key in ("multisoftmax", "multilingual", "language_keys"):
                node.pop(key, None)
        cfg.aux_ctc.decoder.num_classes = -1  # sized from the tokenizer by the constructor
        for split in ("train_ds", "validation_ds", "test_ds"):
            cfg.pop(split, None)
    return unpacked, cfg


def load_indicconformer():
    """IndicConformer's encoder in a stock NeMo hybrid RNN-T/CTC model with our tokenizer: only the
    preprocessor and encoder weights are copied over."""
    unpacked, cfg = _indic_checkpoint()
    model = EncDecHybridRNNTCTCBPEModel(cfg=cfg)
    state = torch.load(unpacked / "model_weights.ckpt", map_location="cpu", weights_only=True)
    keep = {k: v for k, v in state.items() if k.startswith(("preprocessor.", "encoder."))}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    assert not unexpected, unexpected[:5]
    lost = [k for k in missing if k.startswith(("preprocessor.", "encoder."))]
    assert not lost, f"encoder weights not loaded: {lost[:5]}"
    print(f"IndicConformer: {len(keep)} encoder/preprocessor tensors copied, {len(missing)} head tensors fresh")
    return model


__EXTRA_LOADERS__
LOADERS = {"parakeet": load_parakeet, "indicconformer": load_indicconformer__EXTRA_LOADER_NAMES__}
ARCHITECTURE = __ARCHITECTURE__
BASE_MODEL = __BASE_MODEL__
DECODER_NAME = "greedy"
LR_CARD = {"encoder": LR_ENCODER, "heads": LR_HEADS}
PEAK_LR, SCHEDULE = LR_HEADS, "linear"
ENCODER_PREFIXES = ("encoder.", "preprocessor.")


def load_student(weights=None):
    """The student on DEVICE as `model` (which the loss and decoder read): from its pretrained
    checkpoint, or with the weights of a stage's `best/` folder loaded over it."""
    global model
    model = LOADERS[STUDENT]()
    if weights is not None:
        with tarfile.open(Path(weights) / f"{STUDENT}.nemo") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("model_weights.ckpt"))
            state = torch.load(io.BytesIO(tf.extractfile(member).read()), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"{STUDENT}: {len(state)} tensors loaded from {weights}")
    _sum_losses(model)
    _greedy(model)
    model = model.to(DEVICE)
    return model


def prepare_rows(rows):
    """Token ids per row from the shared tokenizer; drops rows that encode an unknown token."""
    tok = model.tokenizer
    kept = []
    for r in rows:
        ids = tok.text_to_ids(r["text"])
        if ids and tok.tokenizer.unk_id() not in ids:
            kept.append({**r, "ids": ids})
    return kept


def retarget(row, text):
    """`row` with `text` as its target, or None when the tokenizer cannot write it."""
    ids = model.tokenizer.text_to_ids(text)
    if not ids or model.tokenizer.tokenizer.unk_id() in ids:
        return None
    return {**row, "text": text, "ids": ids}


def save_weights(model, folder):
    Path(folder).mkdir(parents=True, exist_ok=True)
    model.save_to(str(Path(folder) / f"{STUDENT}.nemo"))


def decoder():
    return transcribe


def make_optimizer(model):
    enc = [p for n, p in model.named_parameters() if n.startswith(ENCODER_PREFIXES) and p.requires_grad]
    heads = [p for n, p in model.named_parameters() if not n.startswith(ENCODER_PREFIXES) and p.requires_grad]
    return torch.optim.AdamW([{"params": enc, "lr": LR_ENCODER}, {"params": heads, "lr": LR_HEADS}],
                             weight_decay=0.0, fused=DEVICE == "cuda")


def probe(model, optimizer, rows):
    """The largest micro-batch at the longest clip carrying the longest target, with the optimizer
    state allocated: (seconds of padded audio per micro-batch, clips per micro-batch)."""
    longest = max(rows, key=ftkit.duration)
    max_len = ftkit.pad_len(round(ftkit.duration(longest) * ftkit.SR), PAD_TO_S)
    max_u = max(len(r["ids"]) for r in rows)

    def probe_step(n):
        wav = torch.randn(n, max_len, device=DEVICE) * 0.1
        ys = torch.randint(1, VOCAB_SIZE, (n, max_u), device=DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            loss = forward_loss(model, wav, torch.full((n,), max_len, device=DEVICE), ys,
                                torch.full((n,), max_u, device=DEVICE))
        loss.backward()

    model.train()
    ftkit.init_optimizer_state(model, optimizer)
    n = ftkit.probe_max_items(probe_step, 1, 256, model)
    budget = n * max_len / ftkit.SR * PROBE_FRACTION
    print(f"largest micro-batch at {max_len / ftkit.SR:.0f} s x {max_u} tokens: {n} clips -> "
          f"budget {budget:.0f} s of padded audio per micro-batch")
    return budget, 256


def collate(rows):
    audio, rows = train_batch(rows)
    clips = [torch.from_numpy(np.array(a, dtype=np.int16)) for a in audio]
    wav = torch.zeros(len(rows), ftkit.pad_len(max(len(c) for c in clips), PAD_TO_S), dtype=torch.int16)
    for i, c in enumerate(clips):
        wav[i, :len(c)] = c
    ys = torch.zeros(len(rows), max(len(r["ids"]) for r in rows), dtype=torch.long)
    for i, r in enumerate(rows):
        ys[i, :len(r["ids"])] = torch.tensor(r["ids"])
    return {"wav": wav, "lens": torch.tensor([len(c) for c in clips]), "ys": ys,
            "ylens": torch.tensor([len(r["ids"]) for r in rows]),
            "tokens": sum(len(r["ids"]) for r in rows),
            "seconds": sum(len(c) for c in clips) / ftkit.SR, "padded_seconds": wav.numel() / ftkit.SR}


def encode(model, wav, lens, augment):
    feats, flens = model.preprocessor(input_signal=wav, length=lens)
    if augment and model.spec_augmentation is not None:
        feats = model.spec_augmentation(input_spec=feats, length=flens)
    return model.encoder(audio_signal=feats, length=flens)


def forward_loss(model, wav, lens, ys, ylens):
    """NeMo's training_step without Lightning: the transducer loss through the fused joint (which
    bounds memory by computing it in sub-batches), plus the CTC head's on the hybrid."""
    enc, elens = encode(model, wav, lens, augment=model.training)
    dec, _, _ = model.decoder(targets=ys, target_length=ylens)
    # The fused joint sets the reduction to None around each sub-batch and restores it only if
    # nothing raises in between. A probe that runs out of memory there leaves it at None, and every
    # later loss would come back per clip, so it is set again on every call.
    model.loss.reduction = "sum"
    loss, _, _, _ = model.joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=elens,
                                transcripts=ys, transcript_lengths=ylens, compute_wer=False)
    if getattr(model, "ctc_loss_weight", 0) > 0:
        ctc = model.ctc_loss(log_probs=model.ctc_decoder(encoder_output=enc), targets=ys,
                             input_lengths=elens, target_lengths=ylens)
        loss = (1 - model.ctc_loss_weight) * loss + model.ctc_loss_weight * ctc
    return loss


def loss_fn(model, b):
    wav = b["wav"].to(DEVICE, non_blocking=True).float() / 32768.0
    loss = forward_loss(model, wav, b["lens"].to(DEVICE, non_blocking=True),
                        b["ys"].to(DEVICE, non_blocking=True), b["ylens"].to(DEVICE, non_blocking=True))
    return loss, b["tokens"]


def transcribe(rows):
    """Greedy batch decode. The encoder runs in bf16; the decoding loop gets fp32 encodings and
    runs outside autocast, where NeMo's CUDA-graph greedy decoder expects to be."""
    clips = [store.clip_f32(r) for r in rows]
    wav = torch.zeros(len(rows), ftkit.pad_len(max(len(c) for c in clips), PAD_TO_S))
    for i, c in enumerate(clips):
        wav[i, :len(c)] = c
    lens = torch.tensor([len(c) for c in clips], device=DEVICE)
    with torch.no_grad():
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            enc, elens = encode(model, wav.to(DEVICE), lens, augment=False)
        hyps = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc.float(), encoded_lengths=elens)
    return [h.text for h in hyps]
'''

# --- the students that write our text as it is: Whisper and Qwen (transformers) ------------------

HF_CONFIG = r"""
QWEN_ID = "Qwen/Qwen3-ASR-0.6B"
WHISPER_ID = "openai/whisper-large-v3-turbo"
QWEN_ASR_VERSION = "0.0.6"   # pins transformers 4.57.6; the version smoke-tested on CPU
# Per stage: the most epochs, which is also the length of the linear LR decay.
EPOCHS = {"human": 8, "distill": 6}
LR = __LR__
"""

HF_STUDENT = r'''
import math
import shutil
import warnings

import transformers
from qwen_asr import Qwen3ASRModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", module="transformers")

# The recogniser is told the language in its prompt. Nepali is not one of its 30, and
# "language None" means "no speech" to it, so the prompt names Nepali and the loss is on the
# transcript alone: the tag is a fixed prompt, not something the model has to learn to say.
QWEN_LANGUAGE = "Nepali"
ARCHITECTURE = __ARCHITECTURE__
BASE_MODEL = __BASE_MODEL__
DECODER_NAME = "greedy+retry"
LR_CARD = LR
PEAK_LR, SCHEDULE = LR, "linear"


def floats(clips):
    return [np.asarray(c).astype(np.float32) / 32768.0 for c in clips]


# --- Qwen3-ASR ---------------------------------------------------------------------------------


def load_qwen(source):
    global model, processor, PROMPT
    wrapper = Qwen3ASRModel.from_pretrained(str(source), dtype=torch.float32, device_map=None)
    model, processor = wrapper.model.to(DEVICE), wrapper.processor
    # The shipped generation config sets a sampling temperature alongside greedy decoding, which
    # transformers refuses to save; decoding here is always greedy.
    for key in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, key, None)
    msgs = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    PROMPT = (processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
              + f"language {QWEN_LANGUAGE}<asr_text>")
    return model


def _qwen_inputs(clips, texts, side):
    # the processor ignores tokenizer.padding_side; it has to be passed per call
    return processor(text=texts, audio=clips, return_tensors="pt", padding=True, padding_side=side)


def qwen_collate(rows):
    """Prompt + transcript + EOS, right-padded; labels only on the transcript and EOS."""
    audio, rows = train_batch(rows)
    clips = floats(audio)
    eos = processor.tokenizer.eos_token
    full = _qwen_inputs(clips, [PROMPT + r["text"] + eos for r in rows], "right")
    prefix = _qwen_inputs(clips, [PROMPT] * len(rows), "right")
    labels = full["input_ids"].clone()
    for i, n in enumerate(prefix["attention_mask"].sum(dim=1).tolist()):
        labels[i, :n] = -100
    labels[full["attention_mask"] == 0] = -100
    return {**full, "labels": labels, "tokens": int((labels[:, 1:] != -100).sum()),
            "lens": torch.tensor([len(c) for c in clips]), "seconds": sum(len(c) for c in clips) / ftkit.SR,
            "padded_seconds": len(rows) * max(len(c) for c in clips) / ftkit.SR}


QWEN_KEYS = ("input_ids", "attention_mask", "input_features", "feature_attention_mask")


def qwen_loss(model, b):
    """Summed token cross-entropy over the transcript, with logits only where there are labels.

    The thinker has no `logits_to_keep`: it runs its 152k-word head over every position, audio
    and prompt included, and HF's loss then upcasts all of it to fp32 (the 2026-09-25 OOM). Here
    the head is swapped for an identity during the forward, so it returns hidden states, and the
    real head runs on the labelled positions alone; the causal shift is HF's (position t predicts
    label t+1)."""
    kw = {k: b[k].to(DEVICE, non_blocking=True) for k in QWEN_KEYS}
    thinker = model.thinker
    head, thinker.lm_head = thinker.lm_head, torch.nn.Identity()
    try:
        hidden = thinker(**kw).logits
    finally:
        thinker.lm_head = head
    labels = b["labels"].to(DEVICE, non_blocking=True)[:, 1:]
    keep = labels != -100
    logits = head(hidden[:, :-1][keep]).float()
    return torch.nn.functional.cross_entropy(logits, labels[keep], reduction="sum"), b["tokens"]


def qwen_transcribe(rows, **gen):
    inputs = _qwen_inputs(floats(store.clip(r) for r in rows), [PROMPT] * len(rows), "left").to(DEVICE)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        out = model.generate(**inputs, do_sample=False, num_beams=1, **{"max_new_tokens": MAX_NEW_TOKENS, **gen})
    seqs = out.sequences if hasattr(out, "sequences") else out
    return processor.batch_decode(seqs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)


def qwen_tokens(text):
    return len(processor.tokenizer(text, add_special_tokens=False).input_ids)


def save_qwen(model, dst):
    model.save_pretrained(dst, state_dict={k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
    processor.save_pretrained(dst)
    base = Path(snapshot_download(QWEN_ID))  # the files qwen-asr needs to load a checkpoint
    for f in base.iterdir():
        if f.suffix in {".json", ".txt"} and not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)


# --- Whisper -----------------------------------------------------------------------------------


def load_whisper(source):
    global model, processor, PREFIX, EOT
    processor = WhisperProcessor.from_pretrained(WHISPER_ID)
    model = WhisperForConditionalGeneration.from_pretrained(str(source), torch_dtype=torch.float32).to(DEVICE)
    model.generation_config.forced_decoder_ids = None
    tok = processor.tokenizer
    PREFIX = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|ne|>", "<|transcribe|>", "<|notimestamps|>"])
    EOT = tok.eos_token_id
    return model


def whisper_collate(rows):
    """Log-mel features, every clip padded to 30 s (the encoder takes nothing else), and the prefix
    + transcript + end-of-text, with loss on the transcript and end-of-text only."""
    audio, rows = train_batch(rows)
    clips = floats(audio)
    feats = processor.feature_extractor(clips, sampling_rate=ftkit.SR, return_tensors="pt").input_features
    ids = [PREFIX + processor.tokenizer(r["text"], add_special_tokens=False).input_ids + [EOT] for r in rows]
    width = max(len(i) for i in ids) - 1
    inp = torch.full((len(rows), width), EOT, dtype=torch.long)
    lab = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, full in enumerate(ids):
        inp[i, :len(full) - 1] = torch.tensor(full[:-1])
        lab[i, len(PREFIX) - 1:len(full) - 1] = torch.tensor(full[len(PREFIX):])
    return {"input_features": feats, "decoder_input_ids": inp, "labels": lab, "tokens": int((lab != -100).sum()),
            "lens": torch.tensor([len(c) for c in clips]), "seconds": sum(len(c) for c in clips) / ftkit.SR,
            "padded_seconds": 30.0 * len(rows)}


def whisper_loss(model, b):
    logits = model(input_features=b["input_features"].to(DEVICE, non_blocking=True),
                   decoder_input_ids=b["decoder_input_ids"].to(DEVICE, non_blocking=True)).logits
    loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1),
                                             b["labels"].to(DEVICE, non_blocking=True).flatten(),
                                             ignore_index=-100, reduction="sum")
    return loss, b["tokens"]


def whisper_transcribe(rows, **gen):
    feats = processor.feature_extractor(floats(store.clip(r) for r in rows), sampling_rate=ftkit.SR,
                                        return_tensors="pt").input_features
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        out = model.generate(input_features=feats.to(DEVICE), language="ne", task="transcribe",
                             do_sample=False, num_beams=1, **{"max_new_tokens": MAX_NEW_TOKENS, **gen})
    return processor.batch_decode(out, skip_special_tokens=True)


def whisper_tokens(text):
    return len(processor.tokenizer(text, add_special_tokens=False).input_ids)


def save_whisper(model, dst):
    model.save_pretrained(dst, state_dict={k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
    processor.save_pretrained(dst)


# --- dispatch and the loop retry ---------------------------------------------------------------

KIT = {
    "qwen": (QWEN_ID, load_qwen, qwen_collate, qwen_loss, qwen_transcribe, qwen_tokens, save_qwen),
    "whisper": (WHISPER_ID, load_whisper, whisper_collate, whisper_loss, whisper_transcribe, whisper_tokens,
                save_whisper),
}
RETRY = {"repetition_penalty": 1.2, "no_repeat_ngram_size": 6}
# Output caps, measured on the labels (2026-09-22 export). Both tokenizers are byte-level and spend
# several tokens per Devanagari character: gold runs to 452 Qwen tokens and 518 Whisper tokens.
# Whisper's decoder stops at 448 positions, 4 of them the prefix, so 5 gold clips cannot be written
# whole by it in one pass; that is the architecture, and it stays in the score.
MAX_NEW_TOKENS = {"qwen": 600, "whisper": 444}[STUDENT]


def load_student(weights=None):
    """The student on DEVICE as `model`: from its pretrained checkpoint, or from a stage's `best/`
    folder. Points collate, loss_fn and transcribe at its kit, and measures the densest human train
    label in its own tokens, for the retry's length cap."""
    global collate, loss_fn, transcribe, count_tokens, save_to, MAX_TOKENS_PER_S
    base, load, collate, loss_fn, transcribe, count_tokens, save_to = KIT[STUDENT]
    load(weights if weights is not None else base)
    MAX_TOKENS_PER_S = max(count_tokens(r["text"]) / ftkit.duration(r) for r in splits["train"])
    print(f"{STUDENT}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, densest train "
          f"label {MAX_TOKENS_PER_S:.1f} tokens/s")
    return model


def prepare_rows(rows):
    """The rows whose transcript fits the decoder's length."""
    kept = [r for r in rows if r["text"].strip() and count_tokens(r["text"]) + 1 <= MAX_NEW_TOKENS]
    print(f"{STUDENT}: dropped {len(rows) - len(kept)} clip(s) empty or longer than {MAX_NEW_TOKENS} tokens")
    return kept


def retarget(row, text):
    """`row` with `text` as its target, or None when it does not fit the decoder's length."""
    return {**row, "text": text} if count_tokens(text) + 1 <= MAX_NEW_TOKENS else None


def save_weights(model, folder):
    Path(folder).mkdir(parents=True, exist_ok=True)
    save_to(model, Path(folder))


def retry_one(row):
    cap = min(MAX_NEW_TOKENS, math.ceil(MAX_TOKENS_PER_S * ftkit.duration(row)) + 2)
    return transcribe([row], max_new_tokens=cap, **RETRY)[0]


def decoder():
    return ftkit.RetryLoops(transcribe, retry_one)


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, fused=DEVICE == "cuda")


def probe(model, optimizer, rows):
    """The largest micro-batch at the longest clip carrying the longest transcript, with the
    optimizer state allocated: (seconds of audio per micro-batch, clips per micro-batch). Whisper
    pays for 30 s whatever a clip's length, so for it the budget is a clip count, not seconds. For
    Qwen the clip count is probed as well (see below)."""
    def probe_at(clip):
        def step(n):
            b = collate([clip] * n)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss, _ = loss_fn(model, b)
            loss.backward()

        return ftkit.probe_max_items(step, 1, 256, model)

    longest = max(rows, key=ftkit.duration)
    wordiest = max(rows, key=lambda r: len(r["text"]))
    model.train()
    ftkit.init_optimizer_state(model, optimizer)
    n = probe_at({**longest, "text": wordiest["text"]})
    if STUDENT == "whisper":
        budget, items = float("inf"), max(1, int(n * PROBE_FRACTION))
    else:
        budget = n * ftkit.duration(longest) * PROBE_FRACTION
        # A budget of seconds packs more sequence positions from short clips than from long ones:
        # each brings its own prompt and a transcript. So the clip count is probed too, at the
        # wordiest of the shortest tenth of the clips (Qwen ran out of memory on 2026-09-25 with
        # only the long probe).
        by_length = sorted(rows, key=ftkit.duration)
        short = max(by_length[: max(1, len(by_length) // 10)], key=lambda r: len(r["text"]))
        n_short = probe_at(short)
        items = max(1, int(n_short * PROBE_FRACTION))
        print(f"largest micro-batch at {ftkit.duration(short):.1f} s: {n_short} clips")
    print(f"largest micro-batch at {ftkit.duration(longest):.0f} s: {n} clips -> "
          f"{items} clips or {budget:.0f} s of audio per micro-batch")
    return budget, items
'''


# --- Omnilingual CTC: a script in its own Python environment (fairseq2) --------------------------

OMNI_CONFIG = r"""
OMNI_CARD = "omniASR_CTC_1B_v2"   # the card the 2026-09-14 bake-off fine-tuned; the 300M is a one-line change
OMNI_VERSION = "0.2.0"            # omnilingual-asr; it needs Python <= 3.12 and fairseq2 0.6
# Per stage: the most epochs, which is also the length of the LR schedule.
EPOCHS = {"human": 20, "distill": 10}
LR = 1e-5
EVAL_ITEMS = 128
"""

OMNI_ENV = r"""
# omnilingual-asr needs Python <= 3.12 and fairseq2 0.6, which pins torch 2.8.0, numpy 1.x and
# huggingface_hub 0.x. None of that installs into the Colab kernel, so the student runs in an
# environment of its own, as a script this notebook writes and launches.
OMNI_ENV = Path("/content/omni-env") if IN_COLAB else FT / "omni-env"
OMNI_PY = OMNI_ENV / "bin" / "python"
if not OMNI_PY.exists():
    !pip install -q uv
    !uv venv -q --python 3.12 {OMNI_ENV}
    !uv pip install -q --python {OMNI_PY} "omnilingual-asr=={OMNI_VERSION}" "torch==2.8.0" "torchaudio==2.8.0"
!uv pip install -q --python {OMNI_PY} soundfile rapidfuzz pyyaml scipy pyarrow requests
!{OMNI_PY} -c "import torch, omnilingual_asr; print('omni env: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"""

OMNI_STUDENT = r'''
from fairseq2.nn import BatchLayout
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

ARCHITECTURE = "Omnilingual ASR CTC: wav2vec 2.0 encoder + CTC head; its own character vocabulary"
BASE_MODEL = OMNI_CARD
DECODER_NAME = "greedy CTC"
LR_CARD = LR
PEAK_LR, SCHEDULE = LR, "tristage"  # fairseq2's recipe: 10% warmup, 40% hold, 50% decay
FRAMES_PER_S = 50  # the wav2vec2 frontend strides 320 samples


def normalize(text: str) -> str:
    # ZWJ/ZWNJ are the only characters of the corpus missing from the written_v2 vocabulary
    # besides "ॐ" (3 uses); they change rendering, not the word.
    return text.replace("\u200d", "").replace("\u200c", "")


def load_student(weights=None):
    """The student as `model`: the pretrained card in float32 master weights (the loader the
    bake-off used), or with the weights of a stage's `best/` folder loaded over it."""
    global model, encode_text, decode_ids, UNK, PAD, VOCAB
    pipe = ASRInferencePipeline(model_card=OMNI_CARD, dtype=torch.float32)
    model, tokenizer = pipe.model, pipe.tokenizer
    if weights is not None:
        state = torch.load(Path(weights) / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
        print(f"{STUDENT}: {len(state)} tensors loaded from {weights}")
    encode_text, decode_ids = tokenizer.create_encoder(), tokenizer.create_decoder(skip_special_tokens=True)
    UNK, PAD, VOCAB = tokenizer.vocab_info.unk_idx, tokenizer.vocab_info.pad_idx, tokenizer.vocab_info.size
    # The official recipe always freezes the convolutional feature extractor (recipe.py).
    for p in model.encoder_frontend.feature_extractor.parameters():
        p.requires_grad_(False)
    return model


def _target(text, seconds):
    """The CTC target of `text`, or None when the vocabulary cannot write it, it is empty
    (fairseq2's BatchLayout rejects a zero-length target) or it has more tokens than frames."""
    ids = encode_text(normalize(text))
    if UNK is not None and bool((ids == UNK).any()):
        return None
    return ids.tolist() if 0 < len(ids) < seconds * FRAMES_PER_S else None


def prepare_rows(rows):
    kept = []
    for r in rows:
        ids = _target(r["text"], ftkit.duration(r))
        if ids is not None:
            kept.append({**r, "ids": ids})
    print(f"{STUDENT}: dropped {len(rows) - len(kept)} clip(s) with an unknown character, an empty target "
          "or an impossible CTC alignment")
    return kept


def retarget(row, text):
    ids = _target(text, ftkit.duration(row))
    return None if ids is None else {**row, "text": text, "ids": ids}


def save_weights(model, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()},
               folder / "model.pt")
    (folder / "README.txt").write_text(
        f"state_dict for {OMNI_CARD} (bf16). Load the card with ASRInferencePipeline(model_card=...), then "
        "model.load_state_dict(torch.load(...)) with the floats cast to float32.\n")


def decoder():
    return transcribe


def make_optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, betas=(0.9, 0.98),
                             eps=1e-8, weight_decay=0.0, fused=True)


def waveforms(clips):
    """Per-clip layer norm, as the recipe's `normalize_audio` and the inference pipeline do."""
    clips = [torch.nn.functional.layer_norm(c, c.shape) for c in clips]
    wav = torch.zeros(len(clips), ftkit.pad_len(max(len(c) for c in clips), PAD_TO_S))
    for i, c in enumerate(clips):
        wav[i, :len(c)] = c
    return wav, [len(c) for c in clips]


def collate(rows):
    audio, rows = train_batch(rows)
    wav, lens = waveforms([torch.from_numpy(np.asarray(a).astype(np.float32) / 32768.0) for a in audio])
    targets = torch.full((len(rows), max(len(r["ids"]) for r in rows)), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        targets[i, :len(r["ids"])] = torch.tensor(r["ids"])
    return {"wav": wav, "lens": lens, "targets": targets, "tlens": [len(r["ids"]) for r in rows],
            "seconds": sum(lens) / ftkit.SR, "padded_seconds": wav.numel() / ftkit.SR}


def loss_fn(model, b):
    wav = b["wav"].cuda(non_blocking=True)
    targets = b["targets"].cuda(non_blocking=True)
    layout = BatchLayout(wav.shape, seq_lens=b["lens"], device=wav.device)
    tlayout = BatchLayout(targets.shape, seq_lens=b["tlens"], device=wav.device)
    # Summed CTC over the batch; the recipe normalises by the number of clips.
    return model(wav, layout, targets, tlayout), len(b["lens"])


def transcribe(rows):
    wav, lens = waveforms([store.clip_f32(r) for r in rows])
    wav = wav.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, layout = model(wav, BatchLayout(wav.shape, seq_lens=lens, device=wav.device))
    pred = logits.argmax(-1)
    texts = []
    for i in range(pred.shape[0]):  # the inference pipeline's greedy CTC decode
        seq = pred[i, :layout.seq_lens[i]]
        keep = torch.ones_like(seq, dtype=torch.bool)
        keep[1:] = seq[1:] != seq[:-1]
        texts.append(decode_ids(seq[keep]))
    return texts


def probe(model, optimizer, rows):
    """The largest micro-batch at the longest padded clip with the longest target, with the
    optimizer state allocated: (seconds of padded audio per micro-batch, clips per micro-batch)."""
    longest = max(rows, key=ftkit.duration)
    max_len = ftkit.pad_len(round(ftkit.duration(longest) * ftkit.SR), PAD_TO_S)
    max_t = max(len(r["ids"]) for r in rows)

    def probe_step(n):
        wav = torch.randn(n, max_len, device="cuda")
        targets = torch.randint(10, VOCAB, (n, max_t), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(wav, BatchLayout(wav.shape, seq_lens=[max_len] * n, device=wav.device),
                         targets, BatchLayout(targets.shape, seq_lens=[max_t] * n, device=wav.device))
        loss.backward()

    model.train()
    ftkit.init_optimizer_state(model, optimizer)
    n = ftkit.probe_max_items(probe_step, 1, 256, model)
    budget = n * max_len / ftkit.SR * PROBE_FRACTION
    print(f"largest micro-batch at {max_len / ftkit.SR:.0f} s: {n} clips -> budget {budget:.0f} s of padded "
          "audio per micro-batch")
    return budget, 256
'''


def config_names(*parts: str) -> list[str]:
    """Every name a Config cell assigns, in order: what the Omnilingual script is handed."""
    names: list[str] = []
    for node in ast.parse("\n".join(parts)).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names += [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
    return list(dict.fromkeys(names))


def omni_script(names: list[str]) -> str:
    """`student_omni.py`: the cells every other student notebook runs in its kernel, as one script
    for the environment fairseq2 needs. Run as `student_omni.py <config.json> <stage|report>`."""
    preamble = [
        '"""One stage of the Omnilingual student (D105), in the Python 3.12 environment fairseq2 needs.',
        "",
        "Written by the notebook from the same cell sources the other students run in their kernels:",
        "build_students.py assembles it, so the stages, the scoring and the uploads are one code.",
        '"""',
        "import json",
        "import os",
        "import sys",
        "from pathlib import Path",
        "",
        "CONFIG = json.loads(Path(sys.argv[1]).read_text())  # the notebook's Config cell",
        "ACTION = sys.argv[2]  # a stage, or `report`",
        *[f'{name} = CONFIG["{name}"]' for name in names],
        'FT = Path(CONFIG["FT"])',
        "sys.path.insert(0, str(FT))",
        'OUT_ROOT = FT / "out" / RUN_PREFIX',
        "OUT_ROOT.mkdir(parents=True, exist_ok=True)",
    ]
    main = (
        'if ACTION == "report":\n'
        + textwrap.indent(REPORT.strip("\n"), "    ")
        + "\nelse:\n    run_stage(ACTION)\nmonitor.stop()"
    )
    cells = [DATA, OMNI_STUDENT, AUG_KIT, PUBLIC_SETS, STAGES]
    return "\n\n\n".join(["\n".join(preamble), *[c.strip("\n") for c in cells], main]) + "\n"


OMNI_LAUNCH = r"""
# What the script is handed: this notebook's Config, as it stands after Setup.
CONFIG_NAMES = __CONFIG_NAMES__
(FT / "omni_config.json").write_text(json.dumps({**{name: globals()[name] for name in CONFIG_NAMES}, "FT": str(FT)}))


def stage(action):
    !{OMNI_PY} {FT / "student_omni.py"} {FT / "omni_config.json"} {action}
"""


@dataclass(frozen=True)
class Student:
    """One student's notebook: its name, where its cells come from, and what its intro says."""

    key: str
    file: str
    title: str
    family: str  # "nemo", "hf" or "omni"
    intro: str
    architecture: str
    base_model: str  # a Python expression, read in the notebook
    settings: dict


STUDENTS = (
    Student(
        key="whisper",
        file="06a_Student_Whisper.ipynb",
        title="Whisper-large-v3-turbo",
        family="hf",
        intro="""
**The student.** Whisper-large-v3-turbo (OpenAI, MIT): a 32-layer encoder and a 4-layer decoder,
0.8 B, with a byte-level BPE that already writes Devanagari and Latin, so nothing is swapped. It
knows Nepali (`<|ne|>`) but loops on it zero-shot (123% WER in the bake-off). On the verified
labels alone it was the closest student to Flex on 2026-09-25 (gold 19.77 against 11.56), and the
gap was largest on pure Nepali. It is DiCoW's backbone, so a Nepali-strong Whisper feeds straight
into the overlap models of the roadmap. Every clip is padded to 30 s, its encoder's only input
length, and its decoder stops at 448 positions: a few gold clips cannot be written whole in one
pass, which is the architecture and stays in the score.

**Choices, fixed before any run.** Peak LR 1e-5, linear decay after 10% warmup, up to 8 epochs on
the human labels and 6 on the mixture, stopping once val WER has gained less than 0.2 points over
3 epochs; a hand stop keeps the best weights so far. ~12 min of audio per optimizer step, summed
token cross-entropy divided by the step's tokens. Decoding is greedy with Flex's loop retry.
""",
        architecture="Whisper-large-v3-turbo: 32L encoder + 4L decoder, 0.8B; no vocabulary change",
        base_model="WHISPER_ID",
        settings={"__LR__": "1e-5"},
    ),
    Student(
        key="qwen",
        file="06b_Student_Qwen.ipynb",
        title="Qwen3-ASR-0.6B",
        family="hf",
        intro="""
**The student.** Qwen3-ASR-0.6B (Alibaba, Apache-2.0): an audio encoder feeding a Qwen3 decoder,
trained on 30 languages including Hindi, not Nepali. Its byte-level BPE writes our text as it is.
Its prompt names the language, and `language None` means "no speech" to it, so the prompt is fixed
at `language Nepali<asr_text>` and the loss is on the transcript alone. On the verified labels
alone it reached gold 22.66 on 2026-09-25, still improving at its last epoch, and memorised the
train voices (val 12.85). It needs its own runtime: `qwen-asr` pins transformers 4.57.6.

**Choices, fixed before any run.** Peak LR 2e-5 (its official recipe), linear decay after 10%
warmup, up to 8 epochs on the human labels and 6 on the mixture, stopping once val WER has gained
less than 0.2 points over 3 epochs; a hand stop keeps the best weights so far. ~12 min of audio
per optimizer step, summed token cross-entropy divided by the step's tokens. Decoding is greedy
with Flex's loop retry.
""",
        architecture="Qwen3-ASR: audio encoder + Qwen3 decoder, 0.6B; byte-level BPE, no vocabulary change",
        base_model="QWEN_ID",
        settings={"__LR__": "2e-5"},
    ),
    Student(
        key="indicconformer",
        file="06c_Student_IndicConformer.ipynb",
        title="IndicConformer",
        family="nemo",
        intro="""
**The student.** IndicConformer (AI4Bharat, MIT): a 121M hybrid RNN-T/CTC Conformer-L whose
encoder has heard Nepali. Its `ne` checkpoint only loads in AI4Bharat's NeMo fork (a 22-language
aggregate tokenizer with no Latin at all, per-language softmax heads). Both are replaced here, so
a stock NeMo model is built from the checkpoint's config and only the preprocessor and encoder
weights are copied. It shares its tokenizer, heads and recipe with Parakeet (06d): between the two
only the encoder differs, a Nepali one against an English one. On the verified labels alone it
reached gold 22.28 on 2026-09-25.

**Choices, fixed before any run.** Peak LR 1e-4 for the encoder and 3e-4 for the fresh heads,
linear decay after 10% warmup, up to 20 epochs on the human labels and 12 on the mixture, stopping
once val WER has gained less than 0.2 points over 3 epochs; a hand stop keeps the best weights so
far. ~12 min of audio per optimizer step, NeMo's own SpecAugment. The loss is NeMo's transducer
loss through the fused joint, plus the CTC head's at weight 0.3, summed over clips and divided by
the step's tokens. Greedy batch decoding; a transducer has no loop retry to add.
""",
        architecture="Conformer-L hybrid RNN-T/CTC, 121M; Indic encoder, fresh 1,024-token heads",
        base_model="INDIC_URL",
        settings={
            "__HUMAN_EPOCHS__": "20",
            "__DISTILL_EPOCHS__": "12",
            "__LR_ENCODER__": "1e-4",
            "__LR_HEADS__": "3e-4",
        },
    ),
    Student(
        key="parakeet",
        file="06d_Student_Parakeet.ipynb",
        title="Parakeet-TDT-0.6B-v2",
        family="nemo",
        intro="""
**The student.** Parakeet-TDT-0.6B-v2 (NVIDIA, CC-BY-4.0): a 618M FastConformer TDT trained on
120k h of English and no Nepali. `change_vocabulary` keeps its encoder and builds a fresh
prediction network and joint for our tokenizer. It is IndicConformer's controlled pair (06c): the
same tokenizer, heads and recipe, and only the encoder differs. On the verified labels alone its
English encoder cost it 13 points against the Nepali one (gold 35.58 on 2026-09-25); its curve
says whether pseudo-labels shrink that.

**Choices, fixed before any run.** Peak LR 1e-4 for the encoder and 3e-4 for the fresh heads,
linear decay after 10% warmup, up to 20 epochs on the human labels and 12 on the mixture, stopping
once val WER has gained less than 0.2 points over 3 epochs; a hand stop keeps the best weights so
far. ~12 min of audio per optimizer step, NeMo's own SpecAugment. The loss is the TDT loss the
model's own constructor builds (`change_vocabulary` leaves a plain RNN-T loss, which cost the
2026-09-25 run 40% of its words), summed over clips and divided by the step's tokens. Greedy batch
decoding.
""",
        architecture="FastConformer TDT, 0.6B; English encoder, fresh 1,024-token decoder and joint",
        base_model="PARAKEET_ID",
        settings={
            "__HUMAN_EPOCHS__": "20",
            "__DISTILL_EPOCHS__": "12",
            "__LR_ENCODER__": "1e-4",
            "__LR_HEADS__": "3e-4",
        },
    ),
    Student(
        key="omnilingual",
        file="06e_Student_Omnilingual.ipynb",
        title="Omnilingual CTC",
        family="omni",
        intro="""
**The student.** Meta's Omnilingual ASR, CTC variant (`omniASR_CTC_1B_v2`): a self-supervised
wav2vec 2.0 encoder with a CTC head, already trained for ASR on Nepali among 1,600+ languages. It
has no language token, which suits code-switching, and its character vocabulary covers all but
0.02% of the corpus's characters, so the pretrained head is kept. It is the one family the other
students leave out, and the one least able to use the context a code-switch needs: CTC predicts
each token independently. In the 2026-09-14 bake-off it reached about 16.6% on val before it was
stopped at epoch 6, under another protocol.

**It runs as a script.** `omnilingual-asr` needs Python <= 3.12 and `fairseq2` 0.6, which pins
`torch==2.8.0`, numpy 1.x and `huggingface_hub` 0.x, none of which installs into the Colab kernel.
The stages therefore run in a `uv` environment as `student_omni.py`, which this notebook writes
and launches. The script is the very cells the other student notebooks run in their kernels,
joined end to end, so the stages, the scoring and the uploads are one code; only the model cell
is its own.

**Not yet run in this form.** The model code (loading, loss, decoding, probe) is the 2026-09-14
fine-tuning notebook's, restored from git history. The stages around it, and the environment's
older `huggingface_hub`, have never run together. The smoke run is the first test.

**Choices, from the official fairseq2 recipe** (`ctc-finetune-recommendation.yaml`, `recipe.py`,
`criterion.py`): peak LR 1e-5 with the tri-stage schedule (10% warmup, 40% hold, 50% decay), AdamW
with betas (0.9, 0.98), summed CTC normalised per clip, per-clip waveform layer norm, a frozen
convolutional feature extractor. Up to 20 epochs on the human labels and 10 on the mixture,
stopping once val WER has gained less than 0.2 points over 3 epochs. Greedy CTC decoding.
""",
        architecture="",
        base_model="",
        settings={},
    ),
    Student(
        key="conformer",
        file="06f_Student_Conformer.ipynb",
        title="a Conformer from scratch",
        family="nemo",
        intro="""
**The student.** A hybrid RNN-T/CTC Conformer with **random weights**: no pretraining at all. It
takes IndicConformer's architecture at a smaller size (`CONFORMER` in Config) and shares the
tokenizer, heads and recipe of the two pretrained transducers (06c, 06d). It asks one question:
once pseudo-labels and human labels together reach about 150 h, does pretraining still matter?
Its stage 1, on the human labels alone, is the point with the least to learn from and is expected
to be poor; the curve is what it is for.

**Not yet run.** This loader has run nowhere: the resized config is built here for the first
time, and the smoke run is its first test. If NeMo's constructor rejects it, the error names the
field; the three sizes are set in `load_conformer`. IndicConformer's checkpoint is downloaded for
its config alone.

**Choices, first guesses fixed before any run.** Peak LR 1e-3 for the whole model (nothing is
pretrained, so there is no slower group), linear decay after 10% warmup, up to 40 epochs on the
human labels and 30 on the mixture, stopping once val WER has gained less than 0.2 points over 3
epochs. ~12 min of audio per optimizer step, NeMo's own SpecAugment. A model trained from scratch
usually wants a longer schedule than a fine-tune; if the curve is still falling when the epochs
run out, raise `EPOCHS`.
""",
        architecture="Conformer hybrid RNN-T/CTC from random weights; 1,024-token heads",
        base_model=repr("none: random weights, IndicConformer's config resized"),
        settings={
            "__HUMAN_EPOCHS__": "40",
            "__DISTILL_EPOCHS__": "30",
            "__LR_ENCODER__": "1e-3",
            "__LR_HEADS__": "1e-3",
        },
    ),
)


def fill(template: str, student: Student, **extra: str) -> str:
    """A family's cell with one student's settings put in."""
    scratch = student.key == "conformer"
    values = {
        "__ARCHITECTURE__": repr(student.architecture),
        "__BASE_MODEL__": student.base_model,
        "__EXTRA_LOADERS__": CONFORMER_LOADER if scratch else "",
        "__EXTRA_LOADER_NAMES__": ', "conformer": load_conformer' if scratch else "",
        **student.settings,
        **extra,
    }
    for key, value in values.items():
        template = template.replace(key, value)
    assert "__" not in template.replace("__init__", "").replace("__call__", ""), student.key
    return template


def stage_cells() -> list[dict]:
    out = []
    for stage, note in RUN_NOTES.items():
        out += [md(note), code(f'run_stage("{stage}")')]
    return out


def student_cells(student: Student) -> list[dict]:
    notebook = student.file.removesuffix(".ipynb")
    intro = f"# {notebook[:3]} — Student: {student.title}\n\nStep 6 of the protocol (D105)."
    head = f'NOTEBOOK = "{notebook}"\nSTUDENT = "{student.key}"'
    if student.family == "omni":
        return omni_cells(student, intro, head)
    if student.family == "nemo":
        config = [head, COMMON_CONFIG, fill(NEMO_CONFIG, student)]
        if student.key == "conformer":
            config.append(CONFORMER_CONFIG)
        setup = nbkit.setup(
            'rapidfuzz duckdb pyarrow "nemo_toolkit[asr]=={NEMO_VERSION}"',
            tail=NEMO_SETUP_TAIL + nbkit.RUN_FOLDER,
        )
        model_cells = [
            md(TOKENIZER_NOTE),
            code(TOKENIZER),
            md(STUDENT_NOTE),
            code(fill(NEMO_STUDENT, student)),
        ]
    else:
        config = [head, COMMON_CONFIG, fill(HF_CONFIG, student)]
        setup = nbkit.setup('rapidfuzz duckdb pyarrow "qwen-asr=={QWEN_ASR_VERSION}"')
        model_cells = [md(STUDENT_NOTE), code(fill(HF_STUDENT, student))]
    return [
        md(intro + "\n" + student.intro + PROTOCOL + SMOKE_NOTE + GPU_NOTE),
        md("## Config"),
        code("\n".join(part.strip("\n") for part in config)),
        md("## Setup"),
        setup,
        *nbkit.kits("ftkit", "evalkit", "sweep", "distill", "xtalk", "augment"),
        code(DATA),
        *model_cells,
        md("## Augmentation, for stage 3\n\nThe augmenter of `03c_Flex_Augment.ipynb`, unchanged."),
        code(AUG_KIT),
        md(STAGES_NOTE),
        code(PUBLIC_SETS),
        code(STAGES),
        *stage_cells(),
        md(REPORT_NOTE),
        code(REPORT),
    ]


def omni_cells(student: Student, intro: str, head: str) -> list[dict]:
    config = [head, COMMON_CONFIG, OMNI_CONFIG]
    names = config_names(*config)
    out = [
        md(intro + "\n" + student.intro + PROTOCOL + SMOKE_NOTE + GPU_NOTE),
        md("## Config"),
        code("\n".join(part.strip("\n") for part in config)),
        md("## Setup"),
        nbkit.setup("rapidfuzz"),
        code(OMNI_ENV),
        *nbkit.kits("ftkit", "evalkit", "sweep", "distill", "xtalk", "augment"),
        md(
            "## The script\n\nThe data, the student, the augmenter, the public sets and the stages, "
            "as the other student notebooks run them cell by cell."
        ),
        code("%%writefile /content/ft/student_omni.py\n" + omni_script(names)),
        code(OMNI_LAUNCH.replace("__CONFIG_NAMES__", repr(names))),
    ]
    for stage, note in RUN_NOTES.items():
        out += [md(note), code(f'stage("{stage}")')]
    return [*out, md(REPORT_NOTE), code('stage("report")')]


NOTEBOOKS = {student.file: student_cells(student) for student in STUDENTS}
#: The Omnilingual script as it is written into its notebook, for the builds test to read.
OMNI_SCRIPT = omni_script(config_names(COMMON_CONFIG, OMNI_CONFIG, 'NOTEBOOK = ""\nSTUDENT = ""'))

if __name__ == "__main__":
    nbkit.write(NOTEBOOKS, OUT_DIR)
