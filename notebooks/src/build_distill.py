"""Build notebooks/Distill.ipynb: `python notebooks/src/build_distill.py`. Roadmap §B, step 0: each
student fine-tuned on the verified labels alone, the baseline distillation has to beat. Shared code
lives in ftkit.py, sweep.py and distill.py, written out by %%writefile cells."""

import json
import sys
from pathlib import Path

from build_finetune import GPU_NOTE, code, md, notebook

HERE = Path(__file__).parent
FTKIT = (HERE / "ftkit.py").read_text()
SWEEPKIT = (HERE / "sweep.py").read_text()
DISTILLKIT = (HERE / "distill.py").read_text()
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent

INTRO = """
# Distillation, step 0 — fine-tune the students on the verified labels

Roadmap §B. Before Flex writes a single pseudo-label, each student is fine-tuned on the 30 h of
verified train labels alone and scored on val and gold, folded and raw, split into S/D/I, per clip
class. This is the baseline distillation has to beat, and it may end §B early: a student that
already matches Flex here needs no unlabelled audio.

**Students.**
- **IndicConformer** (AI4Bharat, MIT): a 121M hybrid RNN-T/CTC Conformer-L whose encoder has heard
  Nepali. Its `ne` checkpoint only loads in AI4Bharat's NeMo fork (a 22-language aggregate
  tokenizer, per-language softmax heads). Both are replaced here, so a stock NeMo model is built
  from the checkpoint's config and only the preprocessor and encoder weights are copied.
- **Parakeet-TDT-0.6B-v2** (NVIDIA, CC-BY-4.0): a 618M FastConformer TDT trained on English only.
  `change_vocabulary` keeps its encoder and builds a fresh prediction network and joint.

**One tokenizer for both.** Neither checkpoint can write our text: IndicConformer's vocabularies
have no Latin at all, and Parakeet's has no Devanagari. Both get the same SentencePiece BPE of
1,024 tokens (Parakeet's own size), trained on **train labels only**. Unlike Flex's tokenizer it
writes `।` and Devanagari digits, so raw WER is not handicapped. With the tokenizer held equal, the
comparison is an English encoder against a Nepali one.

**Reference.** Flex's 2026-09-22 run with no augmentation (`p00-s0`), trained on this export. Its
gold and val transcripts are read from its repo and rescored here with the same scorer, then
paired with each student clip by clip, with episodes resampled (`sweep.paired_bootstrap`).

**Choices, fixed before any run.** Peak LR 1e-4 for the encoder and 3e-4 for the fresh heads
(the overfit check on four clips converged at 3e-4), linear decay after 10% warmup, up to 20
epochs with early stopping on val WER (patience 3), ~12 min of audio per optimizer step, NeMo's
own SpecAugment. The loss is NeMo's transducer loss through the fused joint (plus the CTC head's
at weight 0.3 on the hybrid), summed over clips and divided by the step's tokens. Greedy batch
decoding; a transducer has no loop retry to add.

**Smoke-tested on CPU (2026-09-24, NeMo 3.0.0).** Both students load, train and decode on real
clips, and 60 steps on four clips took IndicConformer's loss per token from 25.6 to 0.6 and
Parakeet's from 16.9 to 3.2. What the CPU could not test: the numba CUDA transducer loss, bf16 on
the GPU, and the batch-size probe. The speed check before each run is where those would fail.

**Outputs** go to the private model repo `OUT_REPO`, under `<RUN_PREFIX>/<RUN_PREFIX>-<student>/`:
`best/<student>.nemo`, `history.json`, metrics with S/D/I per clip class and against Flex, and
`harness/` (transcripts and card) for the Models page (D83). A student already uploaded is
skipped, so after a dropped session, run the notebook again with the same `RUN_PREFIX`.
"""

CONFIG = r"""
RUN_PREFIX = "distill-step0-2026-09-24"   # OUT_REPO/<RUN_PREFIX>/<RUN_PREFIX>-<student>/
NOTEBOOK = "nemo"                          # names this notebook's summary file
STUDENTS = ["indicconformer", "parakeet"]  # in this order: the small one first
PARAKEET_ID = "nvidia/parakeet-tdt-0.6b-v2"
INDIC_URL = ("https://objectstore.e2enetworks.net/indicconformer/models/"
             "indicconformer_stt_ne_hybrid_rnnt_large.nemo")
NEMO_VERSION = "3.0.0"                     # the version smoke-tested on CPU
VOCAB_SIZE = 1024
OUT_REPO = "Sagyam/nepanglish-asr-students"   # private HF model repo, created if missing
FLEX_REPO = "Sagyam/nepanglish-asr-flex-ft"
FLEX_REFERENCE = "flex-xtalk-sweep-2026-09-22/flex-xtalk-sweep-2026-09-22-p00-s0"
EPOCHS, PATIENCE, WARMUP = 20, 3, 0.1
LR_ENCODER, LR_HEADS = 1e-4, 3e-4
EFFECTIVE_S = 720.0       # ~12 min of audio per optimizer step
PAD_TO_S = 1.0
PROBE_FRACTION = 0.9
EVAL_BUDGET_S, EVAL_ITEMS = 1200.0, 96
DEVICE = "cuda"
DATA_LOCAL = None         # a local export in the HF layout instead of the download
"""

SETUP = r"""
%pip install -q rapidfuzz "nemo_toolkit[asr]=={NEMO_VERSION}"
import json
import os
import sys
from pathlib import Path

import torch

# The transducer losses run as numba CUDA kernels. NeMo's own cu12 extra would also pin a
# different torch, so only the kernels' package is added, for this runtime's CUDA.
CUDA_MAJOR = torch.version.cuda.split(".")[0]
%pip install -q "numba-cuda[cu{CUDA_MAJOR}]"

IN_COLAB = "google.colab" in sys.modules
# Secrets only work from a cell run in the Colab UI. When cells are driven from outside (the Colab
# MCP), run this cell by hand once; the token is then kept in the hub's token file on the VM.
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
OUT_ROOT = FT / "out" / RUN_PREFIX
OUT_ROOT.mkdir(parents=True, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
print("students:", STUDENTS, "| outputs:", OUT_ROOT)
"""

DATA = r"""
sys.path.insert(0, str(FT))
import distill
import ftkit
import sweep

ftkit.fast_cuda()
DATA = ftkit.download_dataset(DATA_LOCAL)
splits = ftkit.load_splits(DATA)
store = ftkit.AudioStore(DATA, [r["episode_id"] for rows in splits.values() for r in rows])
score = ftkit.harness_scorer(DATA, FT)
export = json.loads((DATA / "training" / "manifest.json").read_text())
print({k: len(v) for k, v in splits.items()}, f"audio in RAM: {store.gib:.1f} GiB |",
      "export", export.get("exported_at"))
"""

TOKENIZER_NOTE = """
## The shared tokenizer

SentencePiece BPE, 1,024 tokens, full character coverage, trained on train labels only
(`distill.tokenizer_texts` refuses a val or gold clip in train). Measured on the 2026-09-22 export
before this notebook existed: median 6.1 tokens/s on train (max 12.6), and one unknown token each
in val and gold, from characters train never uses. A train clip with an unknown token is dropped
from training; val and gold are scored whole.
"""

TOKENIZER = r"""
import statistics

import sentencepiece as spm

TOK_DIR = FT / f"tokenizer-{VOCAB_SIZE}"
TOK_DIR.mkdir(exist_ok=True)
corpus = FT / "tokenizer_train.txt"
corpus.write_text("\n".join(distill.tokenizer_texts(splits)), encoding="utf-8")
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

REFERENCE_NOTE = """
## Flex, the reference

Flex's transcripts of this export's val and gold, rescored here. The numbers should match its card
(val 7.19, gold 11.56 on the 2026-09-22 export); a clip relabelled since moves them slightly.
"""

REFERENCE = r"""
from huggingface_hub import HfApi, hf_hub_download

BUCKETS = ("none", "0-5%", "5-15%", ">15%")
REPORT_KEYS = ("overlap", "snr", "speakers", "cmi", "duration")  # also paired against Flex


def read_hyps(path):
    return {j["segment_id"]: j["text"] for j in map(json.loads, Path(path).open(encoding="utf-8"))}


flex = {}
for name in ("val", "gold"):
    path = hf_hub_download(FLEX_REPO, f"{FLEX_REFERENCE}/harness/{name}.jsonl", token=os.environ["HF_TOKEN"])
    flex[name] = distill.reference_texts(splits[name], read_hyps(path))
flex_scores = {name: score([r["text"] for r in splits[name]], flex[name]) for name in flex}
for name, m in flex_scores.items():
    print(f"Flex {name}: WER {m['wer']:.2f} ({ftkit.sid(m)}), raw {m['raw_wer']:.2f}, CER {m['cer']:.2f}")
"""

STUDENTS_NOTE = """
## The students: loading, loss and decoding

Everything model-specific. The loss follows NeMo's own `training_step` without Lightning, so
ftkit's loop (batch probe, accumulation, bf16, GPU monitor, early stopping) trains both.
"""

STUDENTS = r'''
import logging
import tarfile
import urllib.request

from omegaconf import OmegaConf, open_dict

from nemo.collections.asr.losses.ctc import CTCLoss
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


def load_parakeet():
    """Parakeet-TDT v2 with our tokenizer: `change_vocabulary` keeps the preprocessor and the
    encoder, and builds a fresh prediction network and joint for the new vocabulary."""
    model = ASRModel.from_pretrained(PARAKEET_ID, map_location="cpu")
    model.change_vocabulary(new_tokenizer_dir=str(TOK_DIR), new_tokenizer_type="bpe")
    return model


def load_indicconformer():
    """IndicConformer's encoder in a stock NeMo hybrid RNN-T/CTC model with our tokenizer.

    The checkpoint only loads in AI4Bharat's NeMo fork: its tokenizer is a custom 22-language
    aggregate and its heads use a per-language softmax. Both are replaced anyway, so the model is
    built from the checkpoint's own config with those parts swapped for stock ones, and only the
    preprocessor and encoder weights are copied over."""
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
    model = EncDecHybridRNNTCTCBPEModel(cfg=cfg)
    state = torch.load(unpacked / "model_weights.ckpt", map_location="cpu", weights_only=True)
    keep = {k: v for k, v in state.items() if k.startswith(("preprocessor.", "encoder."))}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    assert not unexpected, unexpected[:5]
    lost = [k for k in missing if k.startswith(("preprocessor.", "encoder."))]
    assert not lost, f"encoder weights not loaded: {lost[:5]}"
    print(f"IndicConformer: {len(keep)} encoder/preprocessor tensors copied, {len(missing)} head tensors fresh")
    return model


LOADERS = {"parakeet": load_parakeet, "indicconformer": load_indicconformer}
ARCHITECTURE = {
    "parakeet": "FastConformer TDT, 0.6B; English encoder, fresh 1,024-token decoder and joint",
    "indicconformer": "Conformer-L hybrid RNN-T/CTC, 121M; Indic encoder, fresh 1,024-token heads",
}
BASE_MODEL = {"parakeet": PARAKEET_ID, "indicconformer": INDIC_URL}
CARD_NOTE = f"shared {VOCAB_SIZE}-token tokenizer"
DECODER_NAME = "greedy"


def lr_card(name):
    return {"encoder": LR_ENCODER, "heads": LR_HEADS}


def decoder():
    return transcribe


ENCODER_PREFIXES = ("encoder.", "preprocessor.")


def optimizer_for(model):
    enc = [p for n, p in model.named_parameters() if n.startswith(ENCODER_PREFIXES) and p.requires_grad]
    heads = [p for n, p in model.named_parameters() if not n.startswith(ENCODER_PREFIXES) and p.requires_grad]
    return torch.optim.AdamW([{"params": enc, "lr": LR_ENCODER}, {"params": heads, "lr": LR_HEADS}],
                             weight_decay=0.0, fused=DEVICE == "cuda")


def probe_budget(model, optimizer):
    longest = max(splits["train"], key=ftkit.duration)
    max_len = ftkit.pad_len(round(ftkit.duration(longest) * ftkit.SR), PAD_TO_S)
    max_u = max(len(r["ids"]) for r in splits["train"])

    def probe_step(n):
        wav = torch.randn(n, max_len, device=DEVICE) * 0.1
        ys = torch.randint(1, VOCAB_SIZE, (n, max_u), device=DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            loss = forward_loss(model, wav, torch.full((n,), max_len, device=DEVICE), ys,
                                torch.full((n,), max_u, device=DEVICE))
        loss.backward()

    model.train()
    ftkit.init_optimizer_state(model, optimizer)
    n = ftkit.probe_max_items(probe_step, 1, 256, [p for p in model.parameters() if p.requires_grad])
    budget = n * max_len / ftkit.SR * PROBE_FRACTION
    print(f"largest micro-batch at {max_len / ftkit.SR:.0f} s x {max_u} tokens: {n} clips -> "
          f"budget {budget:.0f} s of padded audio per micro-batch")
    return budget


def load_student(name):
    """A fresh student on DEVICE as `model` (which the loss and decoder read)."""
    global model
    model = LOADERS[name]()
    _sum_losses(model)
    _greedy(model)
    model = model.to(DEVICE)
    return model


def encode_targets(rows):
    """Token ids per row from the shared tokenizer; drops rows that encode an unknown token."""
    tok = model.tokenizer
    kept = []
    for r in rows:
        ids = tok.text_to_ids(r["text"])
        if tok.tokenizer.unk_id() not in ids:
            r["ids"] = ids
            kept.append(r)
    return kept


def collate(rows):
    clips = [torch.from_numpy(store.clip(r).copy()) for r in rows]
    wav = torch.zeros(len(rows), ftkit.pad_len(max(len(c) for c in clips), PAD_TO_S), dtype=torch.int16)
    for i, c in enumerate(clips):
        wav[i, :len(c)] = c
    ys = torch.zeros(len(rows), max(len(r["ids"]) for r in rows), dtype=torch.long)
    for i, r in enumerate(rows):
        ys[i, :len(r["ids"])] = torch.tensor(r["ids"])
    return {"wav": wav, "lens": torch.tensor([len(c) for c in clips]), "ys": ys,
            "ylens": torch.tensor([len(r["ids"]) for r in rows]),
            "tokens": sum(len(r["ids"]) for r in rows),
            "seconds": sum(ftkit.duration(r) for r in rows), "padded_seconds": wav.numel() / ftkit.SR}


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

HELPERS_NOTE = """
## Training and scoring helpers

The probe finds the largest micro-batch at the longest train clip with the longest target, with
the optimizer state already allocated, per student. Scoring decodes gold and val with the best
weights and writes: overall and per clip class scores with S/D/I, and the paired difference to
Flex (student minus Flex, in WER points, 95% CI over episodes) overall and per class value of
`REPORT_KEYS`.
"""

HELPERS = r"""
import datetime as dt
import gc

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(OUT_REPO, repo_type="model", private=True, exist_ok=True)
def decode(rows):
    # `decoder()` is the student cell's: greedy for a transducer, greedy + loop retry for the others
    d = decoder()
    texts, compute = ftkit.transcribe_rows(rows, d, budget_s=EVAL_BUDGET_S, max_items=EVAL_ITEMS,
                                           pad_to_s=PAD_TO_S)
    return texts, compute, getattr(d, "log", [])


def evaluate(model, rows=None):
    rows = rows or splits["val"]
    texts, _, _ = decode(rows)
    return score([r["text"] for r in rows], texts)


def paired(rows, refs, a, b, keys):
    # WER(b) - WER(a) in points [95% CI, episodes resampled], overall and per class value
    ca, cb, eps = score.per_clip(refs, a), score.per_clip(refs, b), [r["episode_id"] for r in rows]
    out = {"all": sweep.paired_bootstrap(ca, cb, eps)}
    for key in keys:
        for value in sorted({str((r.get("classes") or {}).get(key)) for r in rows} - {"None"}):
            idx = [i for i, r in enumerate(rows) if str((r.get("classes") or {}).get(key)) == value]
            out[f"{key}={value}"] = sweep.paired_bootstrap([ca[i] for i in idx], [cb[i] for i in idx],
                                                           [eps[i] for i in idx])
    return out


def score_and_write(out, run, name, history):
    model.eval()
    results = {}
    for split in ("gold", "val"):
        rows = splits[split]
        refs = [r["text"] for r in rows]
        texts, compute, log = decode(rows)
        m = score(refs, texts)
        m["rtf"] = sum(compute) / sum(ftkit.duration(r) for r in rows)
        m["retried"] = [{"segment_id": s, "first": f, "retry": t} for s, f, t in log]
        m["by_class"] = distill.by_class(rows, refs, texts, score)
        m["vs_flex"] = paired(rows, refs, flex[split], texts, REPORT_KEYS)
        results[split] = m
        ftkit.write_hyps(out / "harness" / f"{split}.jsonl", rows, texts, compute)
        (out / f"{split}_metrics.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    evals = [h for h in history if "val_wer" in h]
    gold, val = results["gold"], results["val"]
    card = {
        "name": f"{name} FT (distil step 0)",
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "description": f"{RUN_PREFIX}: {name} fine-tuned on the verified train labels only, "
                       f"{CARD_NOTE} (roadmap §B step 0)",
        "architecture": ARCHITECTURE[name],
        "base_model": BASE_MODEL[name],
        "decoder": DECODER_NAME,
        "run_name": run,
        "epochs": EPOCHS,
        "best_epoch": min(evals, key=lambda h: h["val_wer"])["epoch"] if evals else None,
        "lr": lr_card(name),
        "val_wer": val["wer"],
        "gold_wer": gold["wer"],
        "val_sid": {k: val[k] for k in ("sub", "del", "ins")},
        "gold_sid": {k: gold[k] for k in ("sub", "del", "ins")},
        "train_export": {k: export.get(k) for k in ("exported_at", "git_commit", "row_count",
                                                    "normalization_version", "label_version")},
    }
    (out / "harness" / "model_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False))
    keep = ("wer", "raw_wer", "cer", "sub", "del", "ins", "loops", "clips", "rtf", "vs_flex")
    row = {"run_name": run, "student": name, "best_epoch": card["best_epoch"],
           **{split: {k: results[split][k] for k in keep} for split in results},
           "gold_by_class": {k: gold["by_class"].get(k) for k in REPORT_KEYS}}
    (out / "result.json").write_text(json.dumps(row, indent=1, ensure_ascii=False))
    for split in ("val", "gold"):
        d, lo, hi = results[split]["vs_flex"]["all"]
        print(f"{run} {split}: WER {results[split]['wer']:.2f} ({ftkit.sid(results[split])}) | "
              f"minus Flex {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    return row


def uploaded_result(run):
    remote = f"{RUN_PREFIX}/{run}/result.json"
    if not api.file_exists(OUT_REPO, remote):
        return None
    return json.loads(Path(hf_hub_download(OUT_REPO, remote, token=os.environ["HF_TOKEN"])).read_text())


def free_model():
    for var in ("optimizer", "model"):
        globals().pop(var, None)
    gc.collect()
    torch.cuda.empty_cache()


monitor = ftkit.GpuMonitor().start()
"""

RUN_NOTE = """
## Train, score and upload each student

Per student: fresh weights, the batch probe, the speed check (it projects the run and gives val
WER before training), training with early stopping on val, then gold and val decoded with the best
weights and scored. The folder, weights included, goes to `OUT_REPO` as soon as it finishes. **Read
the speed check's line before walking away**: low GPU utilisation or a projection of many hours
means a setting needs changing.
"""

RUN = r"""
results = []
for name in STUDENTS:
    run = f"{RUN_PREFIX}-{name}"
    if (row := uploaded_result(run)) is not None:
        print(f"{run}: already in {OUT_REPO}, skipped")
        results.append(row)
        continue
    OUT = OUT_ROOT / run
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {run}")
    load_student(name)
    splits["train"] = encode_targets(splits["train"])
    print(f"train: {len(splits['train'])} clips the tokenizer can write")
    optimizer = optimizer_for(model)
    budget = probe_budget(model, optimizer)

    def make_batches(epoch, budget=budget):
        return ftkit.bucket_batches(splits["train"], budget_s=budget, max_items=256, pad_to_s=PAD_TO_S,
                                    shuffle=True, seed=epoch)

    # Val before training is scored on every 12th val clip: with fresh heads it is gibberish, and
    # only its timing matters. The projection's val passes are for that sample, so the full-val
    # time is printed after it.
    sample = splits["val"][::12]
    speed = ftkit.speed_check(model, rows=splits["train"], batches=make_batches(0), collate=collate,
                              loss_fn=loss_fn, evaluate=lambda m: evaluate(m, sample), val_rows=sample,
                              gold_rows=splits["gold"], epochs=EPOCHS, monitor=monitor)
    full_val_min = speed["val_s"] * len(splits["val"]) / len(sample) / 60
    print(f"a full val pass takes about {full_val_min:.1f} min, {EPOCHS} of them at most "
          f"{EPOCHS * full_val_min / 60:.1f} h on top of training")
    (OUT / "speed_check.json").write_text(json.dumps(speed, indent=1))

    def save_best(model, out=OUT, name=name):
        (out / "best").mkdir(exist_ok=True)
        model.save_to(str(out / "best" / f"{name}.nemo"))

    cfg = ftkit.TrainConfig(name=run, out=str(OUT), epochs=EPOCHS, lr=LR_HEADS, warmup_frac=WARMUP,
                            effective_s=EFFECTIVE_S, patience=PATIENCE)
    result = ftkit.train(model, cfg=cfg, rows=splits["train"], make_batches=make_batches, collate=collate,
                         loss_fn=loss_fn, evaluate=evaluate, save_best=save_best, optimizer=optimizer,
                         monitor=monitor)
    results.append(score_and_write(OUT, run, name, result["history"]))
    api.upload_folder(repo_id=OUT_REPO, folder_path=str(OUT), path_in_repo=f"{RUN_PREFIX}/{run}",
                      commit_message=f"{run}: step 0 fine-tune")
    free_model()
"""

REPORT_NOTE = """
## Report

Each student against Flex on the same clips, folded WER with S/D/I, then the paired difference
(student minus Flex) per clip class. **The success bar** (roadmap §B) is a student whose interval
contains zero, or lies below it, on gold and val, class by class. Gold is the held-out score:
nothing here was chosen on it.
"""

REPORT = r"""
def line(m):
    return f"{m['wer']:6.2f} ({ftkit.sid(m)})"


print(f"{'system':<16}{'epoch':>6}  {'val':<34}{'gold':<34}{'gold raw':>9}{'gold RTF':>10}")
print(f"{'Flex p00-s0':<16}{'-':>6}  {line(flex_scores['val']):<34}{line(flex_scores['gold']):<34}"
      f"{flex_scores['gold']['raw_wer']:9.2f}{'-':>10}")
for r in results:
    print(f"{r['student']:<16}{str(r['best_epoch']):>6}  {line(r['val']):<34}{line(r['gold']):<34}"
          f"{r['gold']['raw_wer']:9.2f}{r['gold']['rtf']:10.4f}")
for r in results:
    print(f"\n{r['student']} minus Flex, WER points [95% CI, episodes resampled]:")
    for split in ("val", "gold"):
        cells = [f"{k} {d:+.2f} [{lo:+.2f},{hi:+.2f}]" for k, (d, lo, hi) in r[split]["vs_flex"].items()]
        print(f"  {split}: " + "\n        ".join(cells))
summary = {"run_prefix": RUN_PREFIX, "flex_reference": FLEX_REFERENCE,
           "flex": {k: {x: v[x] for x in ("wer", "raw_wer", "cer", "sub", "del", "ins", "clips")}
                    for k, v in flex_scores.items()},
           "students": results}
(OUT_ROOT / f"summary-{NOTEBOOK}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
api.upload_file(path_or_fileobj=str(OUT_ROOT / f"summary-{NOTEBOOK}.json"),
                path_in_repo=f"{RUN_PREFIX}/summary-{NOTEBOOK}.json",
                repo_id=OUT_REPO, commit_message=f"{RUN_PREFIX}: summary")
print("\nuploaded", f"https://huggingface.co/{OUT_REPO}/tree/main/{RUN_PREFIX}")
"""

cells = [
    md(INTRO + GPU_NOTE),
    md("## Config"),
    code(CONFIG),
    md("## Setup"),
    code(SETUP),
    code("%%writefile /content/ft/ftkit.py\n" + FTKIT),
    code("%%writefile /content/ft/sweep.py\n" + SWEEPKIT),
    code("%%writefile /content/ft/distill.py\n" + DISTILLKIT),
    code(DATA),
    md(TOKENIZER_NOTE),
    code(TOKENIZER),
    md(REFERENCE_NOTE),
    code(REFERENCE),
    md(STUDENTS_NOTE),
    code(STUDENTS),
    md(HELPERS_NOTE),
    code(HELPERS),
    md(RUN_NOTE),
    code(RUN),
    md(REPORT_NOTE),
    code(REPORT),
]

if __name__ == "__main__":
    path = OUT_DIR / "Distill.ipynb"
    path.write_text(
        json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote", path, len(cells), "cells")
