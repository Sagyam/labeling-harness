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

| class | source | status |
|---|---|---|
| **crosstalk** | overlap spans from ingest's overlap detector (D77), `segments.overlap_spans_jsonb` | in the database for all 7,075 clips |
| **monologue / dialogue / three or more speakers** | distinct speakers in the clip's turns from the diarizer (D78/D79), `speaker_turns` | turns are in the database; the per-clip count is not derived yet |
| **noise** | an estimate of the background noise level or SNR | nothing yet: the method is open |

**Open questions:**
- Clip-level or episode-level? A dialogue episode holds many single-speaker clips (D52 measured 51
  of 68), and the clip is what the model hears.
- Where should the classes live? Columns derived at ingest (as overlap is), or computed on the
  fly from the turns?
- Single-host episodes were diarized with one speaker, so they have no crosstalk by
  construction.

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
