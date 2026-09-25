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
epochs each, stopping once val WER has gained less than 0.2 points over 3 epochs, ~12 min of audio
per optimizer step, NeMo's own SpecAugment. The first run (2026-09-25, 20 epochs, patience 3)
lost its IndicConformer: it sat near 62% val WER from epoch 3 (mostly deletions) and crawled by
about 0.2 points an epoch, and when it was stopped by hand at about epoch 12 its weights went with
the kernel. A hand stop (Colab's stop button) now ends training like early stopping does: the best
weights so far are uploaded and scored. The loss is NeMo's transducer loss through the fused joint (plus the CTC head's
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
# Per student: the most epochs, which is also the length of the linear LR decay. Both students
# start from fresh heads, and Parakeet from an encoder that never heard Nepali.
EPOCHS = {"indicconformer": 20, "parakeet": 20}
PATIENCE, WARMUP = 3, 0.1
MIN_DELTA = 0.2           # val WER points 3 epochs must gain between them, or training stops
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


flex, flex_clips = {}, {}  # Flex's per-clip counts are aligned once, here, and paired with each student
for name in ("val", "gold"):
    path = hf_hub_download(FLEX_REPO, f"{FLEX_REFERENCE}/harness/{name}.jsonl", token=os.environ["HF_TOKEN"])
    flex[name] = distill.reference_texts(splits[name], read_hyps(path))
    flex_clips[name] = score.per_clip([r["text"] for r in splits[name]], flex[name])
flex_scores = {name: score.summarize(clips) for name, clips in flex_clips.items()}
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
import traceback

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


def paired(rows, ca, cb, keys):
    # WER(b) - WER(a) in points [95% CI, episodes resampled], overall and per class value, from
    # per-clip counts already aligned (score.per_clip)
    eps = [r["episode_id"] for r in rows]
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
        clips = score.per_clip(refs, texts)  # each clip aligned once; every score below adds these up
        m = score.summarize(clips)
        m["rtf"] = sum(compute) / sum(ftkit.duration(r) for r in rows)
        m["retried"] = [{"segment_id": s, "first": f, "retry": t} for s, f, t in log]
        m["by_class"] = distill.by_class(rows, clips, score.summarize)
        m["vs_flex"] = paired(rows, flex_clips[split], clips, REPORT_KEYS)
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
        "epochs": EPOCHS[name],
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


def upload(out, run, message, **kwargs):
    api.upload_folder(repo_id=OUT_REPO, folder_path=str(out), path_in_repo=f"{RUN_PREFIX}/{run}",
                      commit_message=f"{run}: {message}", **kwargs)


# Train, score and upload one student, unless OUT_REPO already has its result. A failure frees
# the GPU before it is raised: the traceback is printed as text, so no frame keeps the model or its
# optimizer state alive, and the next student's cell can run.
def train_student(name):
    run = f"{RUN_PREFIX}-{name}"
    if uploaded_result(run) is not None:
        print(f"{run}: already in {OUT_REPO}, skipped")
        return
    failure, stopped = None, False
    try:
        train_and_score(name, run)
    except BaseException as e:
        failure, stopped = traceback.format_exc(), isinstance(e, KeyboardInterrupt)
    finally:
        free_model()
    if stopped:
        raise KeyboardInterrupt(f"{run} stopped by hand outside training; the GPU is freed")
    if failure:
        print(failure)
        raise RuntimeError(f"{run} failed (traceback above); the GPU is freed for the next student")


monitor = ftkit.GpuMonitor().start()
"""

RUN_NOTE = """
## Train, score and upload each student

Each student has its own cell below, so a crash takes down one student, not the rest. Per student:
fresh weights, the batch probe, the speed check (it projects the run and gives val WER before
training), training with early stopping on val, then gold and val decoded with the best weights and
scored. **The best weights go to `OUT_REPO` as soon as training ends**, before scoring; the scores
follow when scoring finishes. A student whose result is already there is skipped, so after a
kernel restart, re-run the cells above and then every student cell. **Read the speed check's line
before walking away**: low GPU utilisation or a projection of many hours means a setting needs
changing.
"""

RUN = r"""
def train_and_score(name, run):
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
                              gold_rows=splits["gold"], epochs=EPOCHS[name], monitor=monitor)
    full_val_min = speed["val_s"] * len(splits["val"]) / len(sample) / 60
    print(f"a full val pass takes about {full_val_min:.1f} min, {EPOCHS[name]} of them at most "
          f"{EPOCHS[name] * full_val_min / 60:.1f} h on top of training")
    (OUT / "speed_check.json").write_text(json.dumps(speed, indent=1))

    def save_best(model, out=OUT, name=name):
        (out / "best").mkdir(exist_ok=True)
        model.save_to(str(out / "best" / f"{name}.nemo"))

    cfg = ftkit.TrainConfig(name=run, out=str(OUT), epochs=EPOCHS[name], lr=LR_HEADS, warmup_frac=WARMUP,
                            effective_s=EFFECTIVE_S, patience=PATIENCE, min_delta=MIN_DELTA)
    result = ftkit.train(model, cfg=cfg, rows=splits["train"], make_batches=make_batches, collate=collate,
                         loss_fn=loss_fn, evaluate=evaluate, save_best=save_best, optimizer=optimizer,
                         monitor=monitor)
    upload(OUT, run, "best weights (step 0, before scoring)")
    score_and_write(OUT, run, name, result["history"])
    upload(OUT, run, "step 0 scores", ignore_patterns=["best/*"])
"""

REPORT_NOTE = """
## Report

Each student against Flex on the same clips, folded WER with S/D/I, then the paired difference
(student minus Flex) per clip class. **The success bar** (roadmap §B) is a student whose interval
contains zero, or lies below it, on gold and val, class by class. Gold is the held-out score:
nothing here was chosen on it.
"""

REPORT = r"""
results = []
for name in STUDENTS:
    if (row := uploaded_result(f"{RUN_PREFIX}-{name}")) is None:
        print(f"{name}: no result in {OUT_REPO}, left out of the report")
    else:
        results.append(row)


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

HF_INTRO = """
# Distillation, step 0 — the students that write our text as it is

The companion of `Distill.ipynb` (roadmap §B, step 0), for the two students whose tokenizers
already cover Devanagari and Latin, so nothing is swapped: each is fine-tuned on the 30 h of
verified train labels and scored exactly as the transducers are, against the same Flex reference.
It needs its own runtime: `qwen-asr` pins transformers 4.57.6, and NeMo's kernel runs 5.x.

**Students.**
- **Qwen3-ASR-0.6B** (Alibaba, Apache-2.0): an audio encoder feeding a Qwen3 decoder, trained on 30
  languages including Hindi, not Nepali. Its prompt names the language, and `language None` means
  "no speech" to it, so the prompt is fixed at `language Nepali<asr_text>` and the loss is on the
  transcript alone. Chosen over the 1.7B because it is Parakeet's size and smaller than Flex; the
  1.7B is a one-line change if the 0.6B comes close.
- **Whisper-large-v3-turbo** (OpenAI, MIT): it knows Nepali (`<|ne|>`) but loops on it zero-shot
  (123% WER in the bake-off). 04a fine-tuned it to 14.62 on the 2026-09-12 gold; this re-runs it on
  the current export. Every clip is padded to 30 s, which is its encoder's only input length.

**Choices, fixed before any run.** Peak LR 2e-5 for Qwen (its official recipe) and 1e-5 for Whisper
(04a), linear decay after 10% warmup, up to 8 epochs (Whisper-turbo was still improving at 04a's 5),
stopping once val WER has gained less than 0.2 points over 3 epochs, a hand stop keeping the best
weights so far,
~12 min of audio per optimizer step, summed token cross-entropy divided by the step's tokens.
Decoding is greedy with Flex's loop retry: a clip whose output repeats a 3-word sequence 5+ times is
decoded again, alone, with a repetition penalty, no repeated 6-token phrase and a length cap from
its duration (the densest train label, in this model's tokens).

**Smoke-tested on CPU (2026-09-24, transformers 4.57.6).** Zero-shot, Qwen already writes rough
Nepanglish; 30 steps on four clips took its loss per token from 0.81 to 0.00, with the references
reproduced exactly, `।` and English included. The processor left-pads whatever the tokenizer says,
so the padding side is passed per call; with left padding the label mask had covered audio tokens.
"""

HF_CONFIG = r"""
RUN_PREFIX = "distill-step0-2026-09-24"   # the same folder as Distill.ipynb's students
NOTEBOOK = "hf"                            # names this notebook's summary file
STUDENTS = ["qwen", "whisper"]
QWEN_ID = "Qwen/Qwen3-ASR-0.6B"
WHISPER_ID = "openai/whisper-large-v3-turbo"
QWEN_ASR_VERSION = "0.0.6"                 # pins transformers 4.57.6; the version smoke-tested on CPU
OUT_REPO = "Sagyam/nepanglish-asr-students"
FLEX_REPO = "Sagyam/nepanglish-asr-flex-ft"
FLEX_REFERENCE = "flex-xtalk-sweep-2026-09-22/flex-xtalk-sweep-2026-09-22-p00-s0"
# Per student: the most epochs, which is also the length of the linear LR decay. Whisper-turbo
# was still improving at 04a's 5; both keep their pretrained heads.
EPOCHS = {"qwen": 8, "whisper": 8}
PATIENCE, WARMUP = 3, 0.1
MIN_DELTA = 0.2           # val WER points 3 epochs must gain between them, or training stops
LR = {"qwen": 2e-5, "whisper": 1e-5}
EFFECTIVE_S = 720.0       # ~12 min of audio per optimizer step
PAD_TO_S = 1.0
PROBE_FRACTION = 0.9
EVAL_BUDGET_S, EVAL_ITEMS = 1200.0, 96
DEVICE = "cuda"
DATA_LOCAL = None         # a local export in the HF layout instead of the download
"""

HF_STUDENTS_NOTE = """
## The students: loading, loss and decoding

Everything model-specific, one set of functions per student. `load_student` points `collate`,
`loss_fn` and `transcribe` at the right one, so ftkit's loop and the scoring cells are shared.
"""

HF_STUDENTS = r'''
import math
import shutil
import warnings

import numpy as np
import transformers
from huggingface_hub import snapshot_download
from qwen_asr import Qwen3ASRModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", module="transformers")

# The recogniser is told the language in its prompt. Nepali is not one of its 30, and
# "language None" means "no speech" to it, so the prompt names Nepali and the loss is on the
# transcript alone: the tag is a fixed prompt, not something the model has to learn to say.
QWEN_LANGUAGE = "Nepali"
ARCHITECTURE = {
    "qwen": "Qwen3-ASR: audio encoder + Qwen3 decoder, 0.6B; byte-level BPE, no vocabulary change",
    "whisper": "Whisper-large-v3-turbo: 32L encoder + 4L decoder, 0.8B; no vocabulary change",
}
BASE_MODEL = {"qwen": QWEN_ID, "whisper": WHISPER_ID}
CARD_NOTE = "its own tokenizer"
DECODER_NAME = "greedy+retry"


def lr_card(name):
    return LR[name]


def floats(rows):
    return [store.clip(r).astype(np.float32) / 32768.0 for r in rows]


# --- Qwen3-ASR ---------------------------------------------------------------------------------


def load_qwen():
    global model, processor, PROMPT
    wrapper = Qwen3ASRModel.from_pretrained(QWEN_ID, dtype=torch.float32, device_map=None)
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
    clips = floats(rows)
    eos = processor.tokenizer.eos_token
    full = _qwen_inputs(clips, [PROMPT + r["text"] + eos for r in rows], "right")
    prefix = _qwen_inputs(clips, [PROMPT] * len(rows), "right")
    labels = full["input_ids"].clone()
    for i, n in enumerate(prefix["attention_mask"].sum(dim=1).tolist()):
        labels[i, :n] = -100
    labels[full["attention_mask"] == 0] = -100
    return {**full, "labels": labels, "tokens": int((labels[:, 1:] != -100).sum()),
            "lens": torch.tensor([len(c) for c in clips]), "seconds": sum(ftkit.duration(r) for r in rows),
            "padded_seconds": len(rows) * max(len(c) for c in clips) / ftkit.SR}


QWEN_KEYS = ("input_ids", "attention_mask", "input_features", "feature_attention_mask")


def qwen_loss(model, b):
    """Summed token cross-entropy: HF's mean over the label tokens times their count."""
    kw = {k: b[k].to(DEVICE, non_blocking=True) for k in QWEN_KEYS}
    out = model.thinker(**kw, labels=b["labels"].to(DEVICE, non_blocking=True))
    return out.loss * b["tokens"], b["tokens"]


def qwen_transcribe(rows, **gen):
    inputs = _qwen_inputs(floats(rows), [PROMPT] * len(rows), "left").to(DEVICE)
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


def load_whisper():
    global model, processor, PREFIX, EOT
    processor = WhisperProcessor.from_pretrained(WHISPER_ID)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_ID, torch_dtype=torch.float32).to(DEVICE)
    model.generation_config.forced_decoder_ids = None
    tok = processor.tokenizer
    PREFIX = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|ne|>", "<|transcribe|>", "<|notimestamps|>"])
    EOT = tok.eos_token_id
    return model


def whisper_collate(rows):
    """Log-mel features, every clip padded to 30 s (the encoder takes nothing else), and the prefix
    + transcript + end-of-text, with loss on the transcript and end-of-text only."""
    clips = floats(rows)
    feats = processor.feature_extractor(clips, sampling_rate=ftkit.SR, return_tensors="pt").input_features
    ids = [PREFIX + processor.tokenizer(r["text"], add_special_tokens=False).input_ids + [EOT] for r in rows]
    width = max(len(i) for i in ids) - 1
    inp = torch.full((len(rows), width), EOT, dtype=torch.long)
    lab = torch.full((len(rows), width), -100, dtype=torch.long)
    for i, full in enumerate(ids):
        inp[i, :len(full) - 1] = torch.tensor(full[:-1])
        lab[i, len(PREFIX) - 1:len(full) - 1] = torch.tensor(full[len(PREFIX):])
    return {"input_features": feats, "decoder_input_ids": inp, "labels": lab, "tokens": int((lab != -100).sum()),
            "lens": torch.tensor([len(c) for c in clips]), "seconds": sum(ftkit.duration(r) for r in rows),
            "padded_seconds": 30.0 * len(rows)}


def whisper_loss(model, b):
    logits = model(input_features=b["input_features"].to(DEVICE, non_blocking=True),
                   decoder_input_ids=b["decoder_input_ids"].to(DEVICE, non_blocking=True)).logits
    loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1),
                                             b["labels"].to(DEVICE, non_blocking=True).flatten(),
                                             ignore_index=-100, reduction="sum")
    return loss, b["tokens"]


def whisper_transcribe(rows, **gen):
    feats = processor.feature_extractor(floats(rows), sampling_rate=ftkit.SR, return_tensors="pt").input_features
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
    "qwen": (load_qwen, qwen_collate, qwen_loss, qwen_transcribe, qwen_tokens, save_qwen),
    "whisper": (load_whisper, whisper_collate, whisper_loss, whisper_transcribe, whisper_tokens, save_whisper),
}
RETRY = {"repetition_penalty": 1.2, "no_repeat_ngram_size": 6}
# Output caps, measured on the labels (2026-09-22 export). Both tokenizers are byte-level and spend
# several tokens per Devanagari character: gold runs to 452 Qwen tokens and 518 Whisper tokens.
# Whisper's decoder stops at 448 positions, 4 of them the prefix, so 5 gold clips cannot be written
# whole by it in one pass; that is the architecture, and it stays in the score.
MAX_NEW = {"qwen": 600, "whisper": 444}


def load_student(name):
    """A fresh student on DEVICE; points collate, loss_fn, transcribe and save_to at its kit, and
    measures its densest train label in its own tokens for the retry's length cap."""
    global collate, loss_fn, transcribe, save_to, MAX_TOKENS_PER_S, MAX_NEW_TOKENS
    load, collate, loss_fn, transcribe, count, save_to = KIT[name]
    load()
    MAX_NEW_TOKENS = MAX_NEW[name]
    before = len(splits["train"])
    splits["train"] = [r for r in splits["train"] if count(r["text"]) + 1 <= MAX_NEW_TOKENS]
    print(f"{name}: dropped {before - len(splits['train'])} train clip(s) longer than {MAX_NEW_TOKENS} tokens")
    MAX_TOKENS_PER_S = max(count(r["text"]) / ftkit.duration(r) for r in splits["train"])
    print(f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, densest train "
          f"label {MAX_TOKENS_PER_S:.1f} tokens/s")
    return model


def retry_one(row):
    cap = min(MAX_NEW_TOKENS, math.ceil(MAX_TOKENS_PER_S * ftkit.duration(row)) + 2)
    return transcribe([row], max_new_tokens=cap, **RETRY)[0]


def decoder():
    return ftkit.RetryLoops(transcribe, retry_one)


def optimizer_for(model, name):
    return torch.optim.AdamW(model.parameters(), lr=LR[name], weight_decay=0.0, fused=DEVICE == "cuda")


def probe_budget(model, optimizer, name):
    """The largest micro-batch at the longest train clip carrying the longest transcript, with the
    optimizer state allocated. Whisper pays for 30 s whatever a clip's length, so for it the
    budget is a clip count, not seconds of audio."""
    longest = max(splits["train"], key=ftkit.duration)
    wordiest = max(splits["train"], key=lambda r: len(r["text"]))
    worst = {**longest, "text": wordiest["text"]}

    def probe_step(n):
        b = collate([worst] * n)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            loss, _ = loss_fn(model, b)
        loss.backward()

    model.train()
    ftkit.init_optimizer_state(model, optimizer)
    n = ftkit.probe_max_items(probe_step, 1, 256, [p for p in model.parameters() if p.requires_grad])
    if name == "whisper":
        budget, items = float("inf"), max(1, int(n * PROBE_FRACTION))
    else:
        budget, items = n * ftkit.duration(longest) * PROBE_FRACTION, 256
    print(f"largest micro-batch at {ftkit.duration(longest):.0f} s: {n} clips -> "
          f"{items} clips or {budget:.0f} s of audio per micro-batch")
    return budget, items
'''

HF_RUN = r"""
def train_and_score(name, run):
    OUT = OUT_ROOT / run
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {run}")
    load_student(name)
    optimizer = optimizer_for(model, name)
    budget, items = probe_budget(model, optimizer, name)

    def make_batches(epoch, budget=budget, items=items):
        return ftkit.bucket_batches(splits["train"], budget_s=budget, max_items=items, pad_to_s=PAD_TO_S,
                                    shuffle=True, seed=epoch)

    # Val before training on every 12th val clip, as in Distill.ipynb; the full-val time follows
    sample = splits["val"][::12]
    speed = ftkit.speed_check(model, rows=splits["train"], batches=make_batches(0), collate=collate,
                              loss_fn=loss_fn, evaluate=lambda m: evaluate(m, sample), val_rows=sample,
                              gold_rows=splits["gold"], epochs=EPOCHS[name], monitor=monitor)
    full_val_min = speed["val_s"] * len(splits["val"]) / len(sample) / 60
    print(f"a full val pass takes about {full_val_min:.1f} min, {EPOCHS[name]} of them at most "
          f"{EPOCHS[name] * full_val_min / 60:.1f} h on top of training")
    (OUT / "speed_check.json").write_text(json.dumps(speed, indent=1))

    def save_best(model, out=OUT):
        (out / "best").mkdir(exist_ok=True)
        save_to(model, out / "best")

    cfg = ftkit.TrainConfig(name=run, out=str(OUT), epochs=EPOCHS[name], lr=LR[name], warmup_frac=WARMUP,
                            effective_s=EFFECTIVE_S, patience=PATIENCE, min_delta=MIN_DELTA)
    result = ftkit.train(model, cfg=cfg, rows=splits["train"], make_batches=make_batches, collate=collate,
                         loss_fn=loss_fn, evaluate=evaluate, save_best=save_best, optimizer=optimizer,
                         monitor=monitor)
    upload(OUT, run, "best weights (step 0, before scoring)")
    score_and_write(OUT, run, name, result["history"])
    upload(OUT, run, "step 0 scores", ignore_patterns=["best/*"])
"""

_NUMBA = """# The transducer losses run as numba CUDA kernels. NeMo's own cu12 extra would also pin a
# different torch, so only the kernels' package is added, for this runtime's CUDA.
CUDA_MAJOR = torch.version.cuda.split(".")[0]
%pip install -q "numba-cuda[cu{CUDA_MAJOR}]"

"""
HF_SETUP = SETUP.replace(
    '%pip install -q rapidfuzz "nemo_toolkit[asr]=={NEMO_VERSION}"',
    '%pip install -q rapidfuzz "qwen-asr=={QWEN_ASR_VERSION}"',
).replace(_NUMBA, "")
assert "numba" not in HF_SETUP and "qwen-asr" in HF_SETUP


def student_cells(names, config):
    out = []
    for name in names:
        assert f'"{name}"' in config, name
        out += [md(f"### {name}"), code(f'train_student("{name}")')]
    return out


hf_cells = [
    md(HF_INTRO + GPU_NOTE),
    md("## Config"),
    code(HF_CONFIG),
    md("## Setup"),
    code(HF_SETUP),
    code("%%writefile /content/ft/ftkit.py\n" + FTKIT),
    code("%%writefile /content/ft/sweep.py\n" + SWEEPKIT),
    code("%%writefile /content/ft/distill.py\n" + DISTILLKIT),
    code(DATA),
    md(REFERENCE_NOTE),
    code(REFERENCE),
    md(HF_STUDENTS_NOTE),
    code(HF_STUDENTS),
    md(HELPERS_NOTE),
    code(HELPERS),
    md(RUN_NOTE),
    code(HF_RUN),
    *student_cells(["qwen", "whisper"], HF_CONFIG),
    md(REPORT_NOTE),
    code(REPORT),
]

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
    *student_cells(["indicconformer", "parakeet"], CONFIG),
    md(REPORT_NOTE),
    code(REPORT),
]

if __name__ == "__main__":
    for name, nb_cells in (("Distill.ipynb", cells), ("DistillHF.ipynb", hf_cells)):
        path = OUT_DIR / name
        path.write_text(
            json.dumps(notebook(nb_cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("wrote", path, len(nb_cells), "cells")
