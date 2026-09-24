# Custom model building: a distilled student, a conditioned fine-tune, a transplant

Design notes for building models on top of the Flex fine-tune, written and parked on
2026-09-23. The active plan is [roadmap.md](roadmap.md); its sections B and D name two of the
three ideas here as priorities. This file holds the working detail so the roadmap stays short.

**Revisit trigger (owner).** Reopen this file when overlap work resumes (on hold since D100). There
is no attributed real gold (roadmap A stopped), so an uplift is first shown on synthetic mixes and
then checked on real crosstalk as roadmap D says. Until then the bottleneck is evaluation, not models.

## Why nothing here trains from scratch

Already measured:

- The 6.5% clean-gold WER is inherited from Indic-Transcribe's pretraining; the corpus's own
  ~30 h steered it (findings.md, *The retrain on fold-v3*).
- The learning curve is flat: 25% to 100% of train moved val by nothing and gold by ~0.25 per
  doubling, mostly episodes entering training rather than hours helping (findings.md, *Learning
  curve*).
- Gold resolves ~1.5 WER points on clean clips and ~5.5 in the >15% crosstalk bucket
  (findings.md, *Can gold measure a crosstalk fix?*). A size sweep on 30-100 h would mostly
  measure noise.

A novel architecture trained from scratch on 30-100 h lands at tens of percent WER, not 6.5.
Every idea below therefore starts from pretrained weights. "Grow the model until gold stops
improving" survives in exactly one place: sweeping student sizes during distillation, where the
teacher supplies the labels and the 30 h of human annotation is no longer the cap.

## The attribution asymmetry: train is free, gold is not

Word-level per-speaker attribution is hand-paid exactly once, where measurement must be true:
gold and val (~250 overlapped gold clips, plus a crosstalk-holding val split; roadmap A). Train
never gets hand attribution, because it gets per-speaker supervision two cheaper ways:

- **Synthetic mixes: attribution by construction.** Mixing two whole verified train clips
  (roadmap C1) means both transcripts were verified before the mix existed; who said what and
  when is known exactly, including the activity mask. This lifts D95's blocker without a single
  labelling hour. The measured mixer already exists (`notebooks/src/xtalk.py`, D96).
- **Real overlapped clips: the D95 convention is approximately the right target for a
  conditioned model.** Keep-the-main-voice labels poisoned *evaluation* (a correct second-voice
  transcription reads as insertions -- the D96 trap), but a model handed a diarization mask
  saying "follow this voice" is being trained to do almost exactly what those labels do. Label
  noise hurts evaluation catastrophically and training mildly.

D96's negative result (*Synthetic crosstalk does not move real crosstalk*, findings.md) does not
prophesy failure for the ideas below: D96 mixed 0.3 s bursts into a single-stream model with no
conditioning. Roadmap C2 fixes the shape (FastMSS turn-taking, boosted sustained overlap; per
[Mind the Gap](https://arxiv.org/abs/2605.15442), synthetic pretrain + real fine-tune was best,
8.7 against 9.9 tcpWER), and conditioning fixes the target. If it still fails, that is the
sharper, publishable null.

## 1. Model surgery via distillation (roadmap B)

**Teacher.** The current Flex fine-tune with its standard decoder (greedy + cap + retry).
Flex is `bodhan-ai/indic-transcribe-flex`: a 1.2 B Canary-style encoder-decoder, 32-layer
conformer encoder, 24-layer transformer decoder, SentencePiece targets offset by 1,152 special
tokens. The release is inference-only; 04c (`notebooks/src/build_finetune.py`) already built the
training pieces -- teacher-forced cross-entropy, target encoding, SpecAugment -- and a full
fine-tune is ~40 min on an A100.

**Students, in roadmap order:**

- **Whisper-large-v3-turbo (0.9 B).** Already knows Devanagari; is DiCoW's backbone, so this
  student feeds §2. Tokenizer mismatch with Flex means sequence-level distillation.
- **Parakeet / FastConformer (TDT or CTC, 0.6 B class).** Streaming, and word timestamps are
  intrinsic to the alignment -- the need that Flex structurally cannot serve. The surgery: the
  English tokenizer has no Devanagari, so fit a new SentencePiece on the corpus's mixed text,
  resize the output vocabulary, reinitialise the prediction/joint networks (TDT) or CTC head,
  keep the encoder. Where the new tokenizer shares Latin tokens with the old, copy the old
  embeddings rather than random-init. The
  [Hindi recipe](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/discussions/45) documents
  this operation.
- **AI4Bharat IndicConformer.** NeMo hybrid CTC/RNN-T covering Nepali among 22 languages;
  starts closest to our Nepali. Check checkpoint and licence first.

**Method.** Pseudo-label unlabelled Nepanglish audio with the teacher, filtered -- decoder
agreement or a [label-free filter](https://arxiv.org/abs/2407.01257); retry-caught loops are
dropped -- plus the verified labels. KL on teacher token distributions where tokenizers match,
sequence-level otherwise. Reference point: Distil-Whisper used 22k WER-filtered hours
([arXiv 2311.00430](https://arxiv.org/abs/2311.00430)); the student here already knows speech
and is learning a domain, not hearing for the first time, so the need is smaller but unknown
(the flat learning curve measured a fine-tune, not a student).

**Details that repeat from 04c.** Labels must be mapped to what the student's tokenizer can
write (04c: `।` to `.`, Devanagari digits to Latin, ZWJ/ZWNJ stripped -- without it 76% of
labels held an unknown token; `fold.py` scores each pair as identical, so WER is blind to it).

**Open, before building.** Where the unlabelled audio comes from (lives outside the harness
database, teacher-decoded only -- needs a decision entry), and the success bar: student within a
stated margin of the teacher on gold and val, folded and raw, S/D/I, per clip class. Gold and
val audio are never pseudo-labelled for training (D76).

**Licence flag.** The Flex fine-tune is a derivative under the Indic Open Model License v1.0:
private use is fine, distribution passes the licence on, hosting as a service needs Bodhan AI's
written sign-off. A student trained on Flex's *outputs* is a grey zone the licence was not
obviously written for. Training and measuring privately is clearly fine; read the licence before
publishing student weights.

## 2. DiCoW fine-tune on local conditioning and data (roadmap D1)

**The method (BUT-FIT).** [DiCoW v3.3](https://huggingface.co/BUT-FIT/DiCoW_v3_3):
Whisper-large-v3-turbo conditioned on a per-frame diarization mask -- silence, target, other,
overlap -- injected at every encoder layer. Given audio plus a diarization, it writes the target
speaker's words through overlap. [SE-DiCoW](https://arxiv.org/html/2601.19194v1) adds enrolment
of the target's clearest stretch and roughly halves tcpWER. It stays multilingual after
English-only fine-tuning. Weights CC-BY-4.0, code Apache-2.0, training code public
([BUTSpeechFIT/TS-ASR-Whisper](https://github.com/BUTSpeechFIT/TS-ASR-Whisper)).

**The conditioning already exists.** `speaker_turns` from the Modal pyannote run (D78, D79),
joined by time, give per-frame activity; building DiCoW's four-class mask per clip per diarized
speaker is a reshaping, not a new measurement. Overlap spans are measured at ingest (D77). For
SE-DiCoW, enrolment audio is the diarized solo turns.

**The data** is the attribution asymmetry's two sources: C1/C2 synthetic mixes in bulk (masks
known exactly, both transcripts verified) plus real overlapped clips under their existing
main-voice labels with masks from the diarizer.

**Order.** Zero-shot first -- days, not weeks: download, mask pipeline, decode, score. That
number is the baseline every later result is paired against. Then fine-tune.

**The risk, and why §1 comes first.** Whisper-turbo is weak on our Nepali: 14.62 against Flex's
11.44 folded WER on the 2026-09-12 gold. Fine-tuning DiCoW from vanilla Whisper-turbo asks one
run to learn "follow the mask" and "Nepali" together. Fine-tuning from the distilled
Nepali-strong Whisper student (§1) isolates conditioning as the only new skill.

**Scoring.** cpWER, tcpWER, ORC-WER via [MeetEval](https://github.com/fgnt/meeteval) on the
attributed gold, per crosstalk bucket, clean bucket as the no-harm check, paired bootstrap by
episode, a cost term in the decision rule (roadmap E).

## 3. Method transplant: multitalker speaker kernels into Flex's encoder (roadmap D4)

**The method (NVIDIA).**
[multitalker-parakeet-streaming-0.6b-v1](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1):
speaker kernels are built from the diarizer's per-frame activity and injected into a
FastConformer encoder; one model instance per speaker, each writing its own voice through
overlap. It streams, and it was built to be fine-tuned from a strong single-speaker checkpoint.
The released weights are English-only (NVIDIA open model licence), which is why this is a
transplant of the method, not a use of the checkpoint.

**The transplant.** Flex's encoder is a 32-layer conformer -- the same family as Parakeet's, so
the graft has somewhere to attach:

1. **Conditioning module, designed here.** Per-frame diarization activity (already in
   `speaker_turns`) through a learned projection, injected into chosen conformer layers --
   residual add or FiLM-style scale/shift. This is the novel piece and the research risk.
2. **Keep Flex's decoder.** The autoregressive decoder attends the conditioned encoder output.
   Honest caveat: NVIDIA demonstrated the method on a streaming RNN-T/TDT; an encoder-decoder
   port is unproven. DiCoW shows mask conditioning works in an encoder-decoder (Whisper), so
   the bet is not wild.
3. **Training data is C1 mixes.** Target is donor A's verified transcript; the conditioning
   mask is donor A's activity, known exactly. Loss is 04c's teacher-forced cross-entropy.
4. **Inference.** pyannote, one instance per speaker, merge the streams by time.

**Effort and standing.** The largest lift of the three: new module, custom training, no public
training code. A null result still isolates conditioning as the variable, which is publishable.
Do not build it alongside §2: if the DiCoW fine-tune works, the transplant's marginal value
shrinks to streaming and owning the stack; if it fails, the transplant is the fallback with the
conditioning under local control.

## How they interlock

```text
§1 distillation ── Whisper student ──► §2 DiCoW fine-tune ──► tcpWER on attributed gold
     │                                      ▲
     └── Parakeet/conformer student ──► §3 transplant
     │
     └── alone worth having: word timestamps, streaming, an ownable small model
```

§1 is measured on single-stream WER, so it needs nothing from the stopped roadmap A. §2 zero-shot
needs nothing that does not exist; its fine-tune waits on C's data and is selected on synthetic
mixes (D100). §3 waits on the §2
verdict. Every run is scored per roadmap E: folded and raw, S/D/I, per crosstalk bucket, paired
bootstrap by episode, cost term included.
