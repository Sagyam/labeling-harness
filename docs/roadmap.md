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

Tag each clip with the conditions that drive errors, so any WER can be split by them and
training data can be chosen or weighted by them. This may prove useful well beyond the first
experiment.

**What earns a class.** Content measures of the transcript do not split WER: speaking rate and CMI
had within-episode rate ratios of 1.01 and 1.06, i.e. no effect (findings.md, crosstalk). What did
split it was acoustic or speech-style: overlap, two speakers, fillers. Even those explain only
14–17% of clip-level deviance, and the rest sits at episode or speaker level (`ep_447`: 20.7% WER
at 1.8% overlap). A class becomes a WER bucket only if:
- it can be computed without the reference, so it applies to train clips and new audio too;
- it splits WER within an episode, by the crosstalk study's Poisson fit with episode fixed
  effects, under a rule fixed before the run (e.g. the rate-ratio CI excludes 1);
- it points at a lever: separation (item 2), augmentation (item 3) or new voices (item 5).

**Clip-level acoustic classes**, cheapest first:

| class | buckets | source | status |
|---|---|---|---|
| **crosstalk** | none / 0–5 / 5–15 / >15% overlap | ingest's overlap detector (D77), `segments.overlap_spans_jsonb` | in the database for all 7,075 clips; the Models page splits by it |
| **speakers in clip** | 1 / 2 / 3+ | distinct speakers in the clip's `speaker_turns` (D78/D79) | turns stored; derive on the fly |
| **turn changes in clip** | 0 / 1 / 2+ | same | derive on the fly. Turn changes are where the diarizers disagree (findings.md) |
| **pause share** | share of clip time outside VAD speech | `segments.vad_spans_jsonb` | derive on the fly. Greedy loops start on pauses and fillers (loop anatomy) |
| **duration** | <5 / 5–15 / 15+ s | `segments.duration_seconds` | derive on the fly |
| **noise and room** | SNR buckets; reverb (C50) | pyannote Brouhaha: per-frame SNR, C50 and speech, one pass. WADA-SNR is a crude numpy fallback | nothing yet |
| **channel** | full-band / narrow-band (a remote guest on a call); clipping share | spectral rolloff, energy above 4 kHz | nothing yet |

Noise and channel are the best guess at the episode-level residual: in a podcast, the remote
guest's line is usually what varies most. Measure them per speaker turn and give the clip the
time-weighted mean, since one clip can hold a studio host and a compressed guest.

**Speaker-level classes:**
- **Voice.** Link voices across episodes from the per-speaker embeddings stored with each
  diarization run (D78); nothing reads them yet. The key is an anonymous voice ID, never a name
  (D56). Per-voice WER shows whether `ep_447` is one voice or one recording setup.
- **Voice's hours in train:** 0 / < 10 min / > 1 h. This is the measure item 5 needs: the learning
  curve put an episode's own speakers at +0.62 [−0.16, +1.39], and this would settle it on every
  run.
- **Gender and age bracket**, as declared on the ingest form (D58), are already there to split by.

**Word-level classes.** A clip-level average dilutes where errors land, so also split WER and CER
by the class of each aligned reference word:
- script: Devanagari / Latin;
- number;
- filler;
- at a code-switch boundary;
- at a clip edge, where VAD may cut mid-word.

This is the right test of code-switching: whether switch points cost more, not whether mixed
clips do. CER beside WER per class also separates spelling near-misses from misrecognitions.

**CMI stays as a descriptive attribute.** It has sociolinguistic value (corpus statistics, the
dataset card, selecting heavily mixed clips) even though it is not an error bucket. It can still
be crossed with the acoustic classes, e.g. crosstalk WER in high- vs low-CMI clips.

**Where the classes live.**
- **Clip level.** The clip is what the model hears: a dialogue episode holds many single-speaker
  clips (D52 measured 51 of 68). Episodes are the grouping for confidence intervals, not a class.
- **Derived from stored spans** (speakers, turn changes, pause share, duration): computed on the
  fly, with no migration.
- **Needing a model** (noise, room, channel): a column written at ingest plus a backfill, as
  overlap was (D77, `overlap_backfill.py`), with the feature version stored beside the value.

**First step.**
1. Compute the derived classes, plus Brouhaha SNR/C50 and bandwidth, on the 1,109 gold and val
   clips.
2. Refit the fixed-effects model and keep the classes that pass the rule.
3. Write the result into findings.md.
4. Listen to `ep_447` against its channel numbers before anything goes into ingest.

**Caveats.**
- Single-host episodes were diarized with one speaker, so they have no crosstalk by construction,
  and a guest who was never declared cannot be found.
- The filler effect (1.28 per 10 points) was measured under fold-v1. fold-v2 drops fillers on
  both sides, so refit it before keeping fillers as a class.

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
  - the Models page shows its worst clips (D83);
  - the playground runs it on the CPU (D85).
- **After an experiment:**
  - write what was measured into findings.md;
  - discard the experiment code.
