# ASR roadmap

What is next for the ASR work on the Nepanglish corpus, as planned on 2026-09-15. What has been
measured so far, and the numbers each item starts from, are in [findings.md](findings.md). Earlier
plans (decoder search, learning curve, benchmark sanity checks) are done or dropped; their outcomes
are in the findings.

The thread through items 1–5: **crosstalk is the largest measured source of error.** Podcast WER
climbs from 9% on clean clips to 24% on clips more than 15% overlapped. Removing that effect would
take gold from ~11.5% to ~8.8% (fold-v1). Item 1 is the measuring stick for items 2–4, and item 5
makes the benchmark worth measuring against. On the 2026-09-16 split, crosstalk costs val 5.46
points and the speaker-held-out gold 0.17. Item 6, the spelling convention, is folded (fold-v3).

## ~~1. Classify every clip by acoustic condition~~

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
| noise (SNR) and reverb (C50) | pyannote Brouhaha, as ONNX beside the overlap detector | backfilled for 7,071 clips (4 have no speech Brouhaha hears) |
| clipping | — | not measured: stored audio is resampled before anything reads it |

**What earns a class** is unchanged: computable without the reference, a within-episode ratio
per bucket, and a lever it points at. An axis whose interval holds 1 stays in, as a condition
ruled out on every later model.

The joint fit is run (findings.md, *The joint fit*): overlapped time and rapid hand-overs each
carry their own weight, and a second voice with at most one hand-over is exonerated. It is what
splits items 2 and 3 — extraction pays only on the overlapped seconds.

**Still open.**
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

**Crosstalk: built 2026-09-21, swept next (D95, D96).** `notebooks/src/xtalk.py` mixes bursts of
another voice into the clean train clips, shaped by the measured overlap windows. The notebook
sweeps `XTALK_P` over 0, 0.1, 0.2, 0.3 and 0.5, with a second seed at 0, on the 2026-09-21 export,
and decodes the 2026-09-17 model beside them, so the effects of the augmentation and of the added
data are both measured on the same gold. The winner is chosen on val. Noise augmentation is not
built yet.

## 4. Explore newer architectures

Look for models that are realtime (streaming, unlike Flex, which decodes whole utterances) and
resilient to crosstalk. Judge each on the bucketed WER from item 1 and on CPU cost: the CPU
numbers for the current model are the bar (findings.md, CPU inference).

## ~~5. New held-out voices and microphones for the gold pot~~

**Done 2026-09-16.** Gold is now 505 clips from 82 shorts that share no recording with train or
val, with 93 linked voices and none shared with train. The old 706 gold clips moved into train.
First score: findings.md, *Speaker-held-out gold*. **Still missing: crosstalk.** 0.4% of gold audio
is overlapped, so a held-out crosstalk number needs podcasts with new voices.

The plan as written before it was done:

Gold today shares speakers and episodes with train: 36 of 42 episodes have clips in both pots.

- **Collect** new gold with new voices and new microphone setups: people who have never been in
  the corpus, recorded on equipment it has never heard.
- **Then move the current gold into train** (706 clips, ~2.3 h of verified audio).
- **Rules that apply.** Gold stays chosen by hand, clip by clip, and every move goes through
  `set_segment_pot` with an audit row (invariant 4, D71). A screened clip cannot enter gold
  (invariant 5).

## ~~6. Decide a spelling convention for spoken Nepali~~

**Folded 2026-09-17 (D89, fold-v3).** The owner chose to fold every colloquial form the error
mining found, in ten groups, and to tighten later if a listening check by native speakers finds the
metric too kind. Gain: gold 6.58 → 6.25, val 12.59 → 12.05 (findings.md, *fold-v3*).

**Still open.**
- The listening check, and any group it tightens.
- The references still mix both forms, so the model still cannot tell which one is wanted. Only a
  relabel to one form would change what it learns; the fold only changes what is counted.
- Most of the 2.02-point same-word ceiling is grammar, not spelling, and is untouched.

---

## Standing practice

- **Decoder.** Every model is decoded with the standard decoder: greedy + length cap + loop retry.
- **Reporting.** Report folded and raw WER side by side, with the fold version, and split every
  WER into substitutions, deletions and insertions (per 100 folded reference words).
- **After every fine-tune:**
  - copy 04c's `<RUN_NAME>-playground.tar` into `data/models/asr/` and press **Rescan**;
  - the Models page shows its worst clips (D83) and its split by every clip class (D87);
  - the playground runs it on the CPU (D85).
- **After a backfill or a voice link:** `scripts/reclassify_runs.py`, so every run's classes
  catch up with what is now known about its clips.
- **After an experiment:**
  - write what was measured into findings.md;
  - discard the experiment code.
