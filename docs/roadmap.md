# ASR roadmap

What is next for the ASR work on the Nepanglish corpus, as planned on 2026-09-15. What has been
measured so far, and the numbers each item starts from, are in [findings.md](findings.md). Earlier
plans (decoder search, learning curve, benchmark sanity checks) are done or dropped; their outcomes
are in the findings.

The thread through all five: **crosstalk is the largest measured source of error.** Podcast WER
climbs from 9% on clean clips to 24% on clips more than 15% overlapped. Removing that effect would
take gold from ~11.5% to ~8.8% (fold-v1). Item 1 is the measuring stick for items 2–4, and item 5
makes the benchmark worth measuring against.

## 1. Classify every clip by acoustic condition

**In the standard pipeline since 2026-09-15 (D87).** Every clip has a bucket on every axis, every
model run is split by all of them, and the Models page shows each axis's verdict (*splits* /
*ruled out*). The first reading is in findings.md, clip classes. What it measures:

| class | source | status |
|---|---|---|
| crosstalk, speakers in clip, turn changes, pause share, clip length | stored spans and the newest diarization run | computed on the fly for every clip |
| bandwidth | `segments.acoustics_jsonb`, measured at ingest | backfilled for all 7,075 clips |
| voice, voice's hours in train | voices linked across episodes, `diarization_runs.voices_jsonb` | 32 voices over the 44 runs |
| declared gender, age | episode metadata, only where every speaker of the episode shares it | per episode, so no within-episode ratio |
| CMI | `segment_scores.code_switch_density` | kept as a descriptive attribute, crossed with the rest |
| word classes: script, number, code-switch, clip edge | the aligned reference words of a scored run | per run |
| **noise (SNR) and reverb (C50)** | pyannote Brouhaha, as ONNX beside the overlap detector | **not yet**: the model is gated, awaiting access |
| clipping | — | not measured: stored audio is resampled before anything reads it |

**What earns a class** is unchanged: computable without the reference, a within-episode ratio
per bucket, and a lever it points at. An axis whose interval holds 1 stays in, as a condition
ruled out on every later model.

**Still open.**
- Brouhaha SNR/C50, once the model is available (a new `ACOUSTICS_VERSION` and a backfill).
- Each ratio adjusts for the episode only. Crosstalk, speakers and turn changes move together,
  so a joint fit (the crosstalk study's Poisson) is the way to tell them apart if it matters.
- Listen to the band-limited podcast clips: they score *better* within their episodes.
- The voice links have not been checked by ear; they agree with the EDA's independent ECAPA
  voices on every recurring host.

## 2. Experiments on overlapped audio: target-voice extraction

Can extracting the target speaker's voice before recognition win back the crosstalk errors?
MossFormer2 is the first candidate for the separation.

- **Test set.** The overlapped clips, stratified by the crosstalk buckets in findings.md.
- **Target voice.** Which speaker is the target, and how the extractor is told (an enrolment clip
  from the diarizer's embeddings?), is part of the question.
- **Measure.** WER before and after extraction, per bucket, and on clean clips too: extraction
  must not hurt audio that had no crosstalk.
- **Reference caveat.** In overlap the fused reference is at its least reliable, so some
  "deletions" may be reference choices. Listen to a sample before trusting a delta.

## 3. Synthetic data augmentation for noise and crosstalk resilience

Train a model that holds up under noise and crosstalk by mixing them into the training audio at
controlled levels:
- **crosstalk:** other speakers' speech, from other episodes, at a chosen overlap share;
- **noise:** background noise at a chosen SNR.

Compare against the current model (findings.md) on the same buckets as item 2, with gold and val
unchanged, so the gain is attributable. Gold audio must never be a source of mixed-in speech or
noise (D76).

## 4. Explore newer architectures

Look for models that are realtime (streaming, unlike Flex, which decodes whole utterances) and
resilient to crosstalk. Judge each on the bucketed WER from item 1 and on CPU cost: the CPU
numbers for the current model are the bar (findings.md, CPU inference).

## 5. New held-out voices and microphones for the gold pot

Gold today shares speakers and episodes with train: 36 of 42 episodes have clips in both pots.

- **Collect** new gold with new voices and new microphone setups: people who have never been in
  the corpus, recorded on equipment it has never heard.
- **Then move the current gold into train** (706 clips, ~2.3 h of verified audio).
- **Rules that apply.** Gold stays chosen by hand, clip by clip, and every move goes through
  `set_segment_pot` with an audit row (invariant 4, D71). A screened clip cannot enter gold
  (invariant 5).

---

## Standing practice

- **Decoder.** Every model is decoded with the standard decoder: greedy + length cap + loop retry.
- **Reporting.** Report folded and raw WER side by side, with the fold version.
- **After every fine-tune:**
  - copy 04c's `<RUN_NAME>-playground.tar` into `data/models/asr/` and press **Rescan**;
  - the Models page shows its worst clips (D83) and its split by every clip class (D87);
  - the playground runs it on the CPU (D85).
- **After a backfill or a voice link:** `scripts/reclassify_runs.py`, so every run's classes
  catch up with what is now known about its clips.
- **After an experiment:**
  - write what was measured into findings.md;
  - discard the experiment code.
