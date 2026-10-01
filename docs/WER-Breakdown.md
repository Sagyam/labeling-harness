# WER breakdown

A single WER hides why a model lost points. On 2026-10-01, 03b charged the fine-tune +1.14 on
FLEURS, and about half of that turned out to be number formats rather than hearing.
Every evaluation therefore reports the blocks below beside WER and its S/D/I split: gold, val and
each public set, for every run.

**The rule: a breakdown tags errors, it never forgives them.** Folded WER stays exactly what
`fold.py` says (D84, D89); nothing here changes a folding rule or the headline number. The blocks
say where the errors are, so that a model is not judged on errors that belong to a convention, a
format or the audio rather than to its hearing.

Everything is computed from the alignment `fold.align` already makes (`AlignOp`: `match`, `fold`,
`merge`, `sub`, `del`, `ins`), in `notebooks/src/evalkit.py`, so every notebook gets it by scoring
the way it already does. Each run's transcripts are kept in the model repo, so earlier runs
(03a, 03b) are re-scored offline, with no GPU.

## 1. Crosstalk buckets

Crosstalk is the model's weakest point: on gold, clips with any overlap hold 33% of the words and
61% of vanilla-s1's errors (WER 20.1 against 6.2 on clean clips).

- **Measured on every clip, whatever the dataset claims.** A set described as single-speaker read
  speech is checked, not assumed. The detector is the one ingest uses, `app/services/overlap.py`
  (pyannote `segmentation-3.0`, ported to numpy and onnxruntime, no torch; D77): about 25 s of CPU
  per hour of audio. Our own clips already carry its spans in the export; any `unmeasured` clip is
  measured too.
- **Computed once per public set**, since their audio never changes, and stored next to the sets'
  results in the model repo (`benchmarks/overlap/<set>.jsonl`, spans per clip id), so the notebooks
  need neither the model nor the CPU time again.
- **Buckets** are the corpus's own (`clip_classes.overlap_bucket`): none, 0-5%, 5-15%, >15% of the
  clip's duration.
- **Reported per bucket:** clips, share of words, share of errors, WER with S/D/I, and each run
  minus base with its interval, resampling the set's own unit.
- **Read with care:** music, echo or a reader's own breath can look like a second voice. Before a
  public set's buckets are believed, listen to its ten highest-overlap clips.

## 2. Dates, times and numbers

There is no way to normalise every format (`802.11a`, `2.4Ghz`, `06:30`, `2007`, `साढे छ बजे`), and
folding more of them would let real mistakes through: `छत्तीस` for `06:30` is misheard, not
differently written. Plain whole numbers already match their digits (fold.py, D84); everything else
is detected and set apart, never matched.

- **A word is numeric** if `fold.is_number` says so, it contains a digit, or it is a format word: a
  time, date or decimal pattern (`06:30`, `2.4`, `2026-10-01`, `1st`), a unit stuck to a number
  (`2.4Ghz`, `5km`, `%`), or a word from a short list (`बजे`, `साढे`, `सवा`, `पौने`, `गते`, `साल`,
  `दशमलव`, ``point``, the Nepali and English month names). The list will be incomplete at first;
  its misses are found by reading a sample of tagged and untagged errors.
- **An error is numeric** if any word on either side of it is numeric.
- **Reported:** numeric errors with S/D/I and their share of all errors; WER with them left out,
  beside the full WER (an extra column, never a replacement); and the clips whose reference
  contains a numeric word, with their count and WER.
- **Listed:** the most frequent numeric substitutions, reference against model, as examples.

## 3. Most common substitutions, deletions and insertions

- Per set and run: the 20 most frequent substitution pairs (reference word, model word), deleted
  words and inserted words, with counts and each one's share of its kind, written as the
  transcripts wrote them.
- Against base: the pairs and words that grew most, which is what the fine-tune introduced.
- Short words dominate these lists without dominating the errors (particles were 13% of
  vanilla-s1's gold errors), so each list is read beside its share, not on its own.

## Outputs

`<split>_metrics.json` and `benchmarks/<set>.json` gain a `breakdown` object with the three
blocks, and the notebooks' report cells print it. `evalkit` stays importable without torch, so the
breakdown is tested in `backend/tests/test_eval_kit.py`.

## Not in this round

- **S/D/I by language** (Nepali, English, numbers), by the script of the reference word.
- **Spelling convention:** counting the word pairs fold.py forgave (another script, vowel length,
  a split word), which shows each dataset's own convention from its data.
- **Phonetic similarity:** tagging each substitution as a near miss or a different word, with a
  threshold calibrated on pairs the owner judges by ear.
- **Reference audit:** flagging public-set clips where the models agree with each other and not
  with the reference, and listening to the top of that list. FLEURS, OpenSLR 54 and Common Voice
  references are the text the speaker was asked to read, not a transcript of what was said.
