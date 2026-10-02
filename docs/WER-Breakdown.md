# WER breakdown

A single WER hides why a model lost points. On 2026-10-01, 03b charged the fine-tune +1.14 on
FLEURS, and about half of that turned out to be number formats rather than hearing. Every
evaluation therefore reports a breakdown beside WER and its S/D/I split, and every error is kept
for analysis rather than read once and thrown away.

## Direction: one rulebook, with tags for what it cannot decide (D112)

The first direction (2026-10-01) was **folding rules get smaller and the breakdown gets finer**:
a static table always misses cases and sometimes forgives a real mistake, while the breakdown
decides nothing. The 2026-10-02 audit kept the second half and refined the first. The fold had
been wrong both ways, but the over-folding was bugs and rules too broad, and the under-folding a
short list that cannot join two different words. So fold-v4 is one rulebook (D112):

- **A rule folds only what cannot be two words** (tiers 1-3). Anything that may join two
  different words -- `केस` "case" and `केश` "hair" -- is a **tag** (tier 4): it describes a charged
  error and never forgives it. Nuance still belongs in the breakdown.
- **Every rule is held to evidence.** Each has examples and counterexamples a test checks, each
  forgiven row names the rule that forgave it, and the Rulebook page shows how often each fired
  per model and set, with samples.
- **A breakdown tags errors, it never forgives them.** Folded WER is whatever `fold.py` says.

## The rulebook (fold-v4)

`RULEBOOK` and `TAGS` in `backend/app/services/fold.py` are the rules; the Rulebook page
(`GET /fold/rulebook`) is how they are read, with their evidence. In short:

| tier | what | rules |
|---|---|---|
| 1 | orthography: one word written another way | case, contraction, apostrophe, english-variant, letter-names, digits, vowel-length, chandrabindu, doubled-sign, nukta, final-virama, visarga, nasal-cluster, number, spacing |
| 2 | across scripts, by sound | sound-skeleton, sound-ratio, sound-short; a case ending must agree, a Nepali function word matches only its romanization, a merge may not swallow a word |
| 3 | colloquial Nepali (D84, D89) | contracted-verb, western-participle, progressive, benefactive, first-plural, pronoun, launu, emphatic, loose, unseen, joined, other-words |
| 4 | tags, still errors | ba-va, sibilant, inner-virama, nasal-dropped, nasal-added, au-o, repetition, particle (D113) |

How much each tier forgives, vanilla-s1, pairs per 100 reference words (tags from mine-v4):

| set | tier 1 | tier 2 | tier 3 | tier-4 tags (charged) | largest tags |
|---|---|---|---|---|---|
| gold | 3.36 | 0.70 | 0.93 | 1.03 | particle 0.57, repetition 0.22 |
| val | 2.87 | 0.30 | 1.09 | 0.66 | particle 0.30, repetition 0.18 |
| FLEURS | 6.40 | 3.80 | 0.13 | 0.85 | ba-va 0.26, sibilant 0.16 |
| OpenSLR 54 | 5.33 | 2.54 | 0.05 | 1.31 | nasal-added 0.42, ba-va 0.27 |
| Common Voice | 8.96 | 1.63 | 1.34 | 2.33 | nasal-added 1.28, ba-va 0.29 |
| IndicVoices | 6.19 | 5.20 | 1.56 | 1.36 | particle 0.77, repetition 0.26 |
| nepali_cs | 10.60 | 0.51 | 1.14 | 0.71 | particle 0.32, repetition 0.28 |

Common Voice's `nasal-added` (1.28 of its 8.55) is mostly a reference that dropped a chandrabindu
the model wrote (`हामी नेपाली हौ`/`हौँ`): read against its tag, it is the reference's error.

## Error mining

Today the harness shows a model's mistakes clip by clip. That answers "what went wrong here" and
nothing across clips. Error mining stores **every aligned word pair of every evaluation**,
classified, so that any question about a model's mistakes is a query rather than a new script.
It is designed fresh, not fitted to the existing evaluation tables.

### What is recorded

One row per aligned pair from `fold.align`, for every run on every set (gold, val and each public
set), with what is needed to find and reproduce it: the run, the set, the clip, the pair's
position in the clip, the reference word(s) and the model word(s) as written, and the
`fold_version` and classifier version that produced the row.

**Forgiven pairs are recorded too**, not only errors. A pair `fold.py` matched across scripts, by
spelling or by merging words is a convention difference, and some of those forgivenesses are
wrong (below). They can only be audited if they are kept.

Rows are derived data: they are recomputed from the stored reference and model text whenever the
fold or classifier version changes, and a row always names the versions it came from.

**They are Parquet files, not tables** (owner, 2026-10-02). The rows are analytical: written once,
never updated, read only through group-bys. A run is about 225k pairs, ~22k of them errors
(vanilla-s1, measured 2026-10-02: public sets 126k pairs and 13.7k errors, gold ~39k words, val
~58k), so a file per run and set is a few MB, and DuckDB answers any of the queries below over it
in milliseconds. Losing a file costs a re-derivation from the transcripts (seconds of CPU), so
nothing about them needs Postgres's guarantees.

Identical matches are kept as well as the rest: they are the denominators. A bucket's WER, or how
often a word is right, is a sum over its rows only because every reference word has one.

The tail is long. The public sets' 13,732 errors are 9,567 distinct (reference, model) pairs, 8,137
of them seen once, and the top 500 cover 27%. A confusion table does not collapse the errors into
a list one can read; the classes do, and reading is done by sampling within a class.

### How each pair is classified

In this order, each answering one question:

1. **S, D or I** -- or forgiven, with how: `match` with a spelling difference, `fold` (another
   script, or a number written another way), or `merge` (split or joined words).
2. **Script** of each side: Devanagari, Latin or mixed (`queriesहरू`). A deletion has only a
   reference side and an insertion only a model side.
3. **Number or word.** No finer split into years, dates, times or ordinals: each finer class adds
   rules that need maintaining and new ways to be wrong. `fold.is_number` (tightened 2026-10-02)
   takes a token led by digits, or a number word followed by nothing but a counter (`वटा`,
   `जना`), the ordinal ending or a case ending; `छ`, `एक`, `छौँ` and `तिन` with a case ending are
   words, not numbers.
4. **Convention mismatch**: the same word, written differently. Transliteration (`मेसी`/`Messi`,
   `B`/`बी`), a spelling variant within Devanagari (`चिपकाउनुपर्छ`/`चिप्काउनुपर्छ`), or a split
   word. Deciding it can use a dictionary of word pairs, best learned from the data -- every pair
   `fold.py` forgave across scripts in any run is evidence that the two spellings are one word --
   rather than written by hand.
5. **Near miss or different word**, for a substitution that is not a convention mismatch: did the
   model hear something close (`cache`/`cage`, `घोल्नुपर्छ`/`खोल्नुपर्छ`) or something unrelated
   (`क्याट्रिना`/`Captain`)? Compared after romanization, so `छ`, ``chha`` and ``xa`` are one
   sound, by consonant skeleton as `fold.py` already does across scripts. The threshold is
   calibrated on pairs the owner judges by ear, not chosen.

### What it is read through

- **A confusion table**: how often word A was written as B, and B as A, per run and across runs,
  filterable by every class above. This is the "most common S/D/I" list, kept instead of printed.
- **The breakdown blocks** below are fixed queries over the same rows.
- **The clip view** stays, and links each pair to its clip and audio.

## The breakdown blocks

Reported for every run on every set, each beside its WER.

### 1. Crosstalk buckets

Crosstalk is the model's weakest point: on gold, clips with any overlap hold 33% of the words and
61% of vanilla-s1's errors (WER 20.1 against 6.2 on clean clips).

- **Measured on every clip, whatever the dataset claims.** A set described as single-speaker read
  speech is checked, not assumed. The detector is the one ingest uses, `app/services/overlap.py`
  (pyannote `segmentation-3.0`, ported to numpy and onnxruntime, no torch; D77): about 25 s of CPU
  per hour of audio. Our own clips carry its spans in the export; any `unmeasured` clip is
  measured too. A public set's audio never changes, so it is measured once and the spans kept.
- **Buckets** are the corpus's own (`clip_classes.overlap_bucket`): none, 0-5%, 5-15%, >15%.
- **Per bucket:** clips, share of words, share of errors, WER with S/D/I, and each run minus base
  with its interval, resampling the set's own unit.
- **Read with care:** music, echo or a reader's breath can look like a second voice. Listen to a
  public set's ten highest-overlap clips before believing its buckets.

### 2. Numbers

There is no way to normalise every format (`802.11a`, `2.4Ghz`, `06:30`, `2007`, `साढे छ बजे`), and
folding more of them would let real mistakes through: `छत्तीस` for `06:30` is misheard, not
differently written. Plain whole numbers already match their digits (D84); everything else is
classified, never matched.

- An error is a number error when either side is a number (class 3 above).
- **Reported:** number errors with S/D/I and their share of all errors; WER with them left out,
  beside the full WER (an extra column, never a replacement); the clips whose reference contains
  a number, with their count and WER.

### 3. Most common substitutions, deletions and insertions

The confusion table's top rows per set and run: substitution pairs both ways, deleted and
inserted words, each with its count and share of its kind, and the rows that grew most against
base. Short words dominate these lists without dominating the errors (particles were 13% of
vanilla-s1's gold errors), so a row is read beside its share.

### 4. What each recording condition costs (D111)

Every clip that counts toward a WER is tested on two fronts, crosstalk and speech-to-noise ratio:
every public-set clip the owner heard on the crosstalk detector's word was one or the other.
Crosstalk alone cannot tell them apart (OpenSLR 54 is read speech, and its "crosstalk" clips
measure 4.6-16.6 dB), so both are measured on every clip, and the card puts each clip in one
cell: its crosstalk bucket if it has any, else its SNR bucket (the corpus's own, D87), else the
baseline (no crosstalk, 45+ dB) or unmeasured.

- **Per condition:** clips, words, WER, the ratio to its baseline and the points of the set's
  WER it costs, `errors x (1 - 1/ratio)`, with intervals from resampling the set's unit.
- **Within episode.** Each clip is compared only with baseline clips of its own episode
  (Mantel-Haenszel, as the By class panel does): the condition's own cost. The clean-floor
  figure, against the set's pooled baseline, is one sentence in the card's explanation: it also
  charges the condition for the shows it comes in.
- **The rest**, everything no condition took, by kind of error: a number, a deletion, an
  insertion, a substitution across scripts, between English words, between Devanagari words at
  similarity 0.75 or more (mostly suffixes), and any other substitution. The rows sum to the WER.
- **Against another model**, every row gains this model minus that one, on the clips both
  scored, with a paired interval: which rows a fine-tune moved.

## The subcases as measured (2026-10-01)

vanilla-s1, per 100 reference words, on gold (ours), FLEURS (English written in Devanagari) and
nepali_cs (code-switched lectures):

| | gold | FLEURS | nepali_cs |
|---|---|---|---|
| forgiven: another script | 0.70 | 3.49 | 0.55 |
| forgiven: spelling | 1.68 | 1.81 | 2.20 |
| forgiven: split or joined words | 2.09 | 4.14 | 3.10 |
| forgiven: number | 0.32 | 0.43 | 3.05 |
| charged: number | 0.61 | 0.78 | 1.17 |
| charged: cross-script, similar (>= 0.5) | 0.10 | 0.23 | 0.23 |
| charged: cross-script, unlike | 0.58 | 0.58 | 0.68 |
| charged: same script, similarity >= 0.75 | 1.39 | 3.85 | 1.20 |

What these showed:

- **Each dataset's convention is visible in its forgiven pairs.** FLEURS's 3.49 cross-script
  forgivenesses are `स्पेनिश`/``Spanish``, `भर्चुअल`/``virtual``: it writes English in Devanagari.
  No convention has to be known in advance.
- **Convention still leaks into errors**: single letters and short words fail the cross-script
  rule (`B`/`बी`, `UN`/`युएन`, `द`/``the``, `अफ`/``of``, `किमी`/``kilometer``), and an internal
  virama is charged at similarity 1.00 (`चिपकाउनुपर्छ`/`चिप्काउनुपर्छ`).
- **Folding also forgives real mistakes.** `Sir`/`सहर` ("city") and ``best छ``/`भेट्छ` ("meets")
  match by consonant skeleton. A two-against-one merge across scripts swallows a whole extra
  word: `छ Option`/``Option``, ``number``/``number बी``, `र script`/``script``. Such merges are at
  most 0.50 per 100 words on gold, 0.32 on FLEURS and 0.86 on nepali_cs; some of them are
  legitimate (`मिनेटपछि`/``minute पछि``). Today's folded WER is therefore slightly flattering.
- **A similar Nepali word is usually grammar, not sound.** Most close same-script substitutions
  differ by a suffix: `भाषाहरूमा`/`भाषाहरू`, `रहेको`/`रहेका`, `उनीहरूले`/`उनीहरू`. They are real
  errors, and a near-miss tag based on similarity alone would mislabel them as hearing.

## What was built (2026-10-02)

Classes 1-3 and the three blocks. Classes 4 (convention) and 5 (near miss) are not classified:
they come from reading what 1-3 surface, as a new miner version. The rows carry what that
reading needs, each substitution's `similarity` and both sides romanized, so the page sorts and
filters by them without a threshold having been chosen. Error rows are Parquet files beside their
model, not tables (D110).

- **The classifier**, `backend/app/services/error_mining.py` (`MINER_VERSION = "mine-v4"`; v2 added each clip's SNR, v3 `fold_rule` and `variant`, v4 the `particle` tag):
  `pairs` turns a clip into one row per alignment step, reusing an alignment the scorer already
  made; `rows` adds the clip's columns; `write` and `read` are the only writer and reader. It
  imports only `fold.py`, the standard library and DuckDB. A clip id repeated in a set (nepali_cs
  has three clips named `-None`) is numbered `#2`, `#3` in file order.
- **The columns**: `run`, `set`, `clip_id`, `group` (the resampling unit), `pos`, `ref`, `hyp`,
  `kind`, `identical`, `forgiven` (`spelling`, `script`, `number`, `merge`; a fold is `number`
  when rule 2 matched it), `ref_script`, `hyp_script` (`dev`, `lat`, `mix`, `none`; null on the
  absent side), `number` (either side), `ref_number` (the reference side: block 2's clips),
  `similarity`, `fold_rule` (the rulebook rule that forgave the pair), `variant` (the tier-4
  tag of an error), `ref_roman`, `hyp_roman`, `overlap_share`, `overlap_bucket`, `snr_db`,
  `snr_bucket` (the corpus's SNR buckets, D87), `by`. Metadata:
  `run`, `set`, `fold_version`, `miner_version`, `created_at`.
- **The queries**, `backend/app/services/error_store.py`: `breakdown` (WER with S/D/I and
  interval, blocks 1 and 2, each against a base on the clips both scored with the set's group
  resampled), `confusion` (block 3), `occurrences` (with context, or a seeded sample), `clip_ops`
  and `report` (what a notebook stores). "WER without numbers" is the WER over every row that
  involves no number, its reference words left out with it.
- **Where the files live**: `data/models/asr/<slug>/errors/<set>.parquet`, all from one run. They
  arrive by upload (`POST /models/{slug}/errors`), with a folder copied from the hub (Rescan lists
  them), or from `scripts/mine_errors.py`, which derives them from a model's imported runs and the
  `benchmarks/<set>.jsonl` in its folder.
- **The public sets' recording conditions**: `scripts/measure_benchmark_overlap.py` streams each
  set through `app/services/overlap.py` into `data/benchmarks/overlap/<set>.parquet`, and through
  the acoustic meter (Brouhaha's SNR and C50, and bandwidth; D87) into
  `data/benchmarks/acoustics/<set>.parquet`, about 18 s of CPU per hour of audio. A public clip
  is measured on its own audio, a corpus clip from its whole episode. `--listen` keeps the
  highest-overlap clips as FLAC, `--upload` puts both files in the model repo under
  `benchmarks/`, where the notebooks read them (`evalkit.fetch_conditions`).
- **The page**: the Errors section of the Models page (import, set and base pickers, blocks 1 and
  2, the confusion table and its filters, occurrences with Sample 50 and its seed, a clip: gold
  and val with audio, a public set as its alignment). Driven with Playwright on vanilla-s1.
- **The notebooks**: `evalkit` writes `harness/errors/<set>.parquet` for gold, val and every
  public set, and a `breakdown` into each metrics file; `07_Report` prints blocks 1 and 2 per run
  and set. `scripts/upload_harness_copy.py` carries `error_mining.py` and `error_store.py` into
  the dataset's `harness/` beside `fold.py`; a notebook on a copy without them still scores and
  writes no error files.

## Measured (2026-10-02)

**Size.** vanilla-s1's rows over gold, val and the five public sets: 224,274 pairs, about 70 s of
CPU to derive, each file under 1 MB; a breakdown with its bootstrap reads in well under a second.

**The number tag's false positives**, read off every word it tagged in gold, val and the public
sets: besides the three named above, `छौँ` ("we are", 445 times) and `तिन` with a case ending
(`तिनको`, `तिनले`: "their", about 70 times). Bare `तिन` is three (`दुई तिन दिन`) and stays one.

**Crosstalk on the public sets**, measured on every clip. Not yet heard, so not yet believed:

| | none | 0-5% | 5-15% | >15% |
|---|---|---|---|---|
| FLEURS | 724 | 2 | 0 | 0 |
| Common Voice | 283 | 0 | 4 | 0 |
| OpenSLR 54 | 13,339 | 48 | 65 | 157 |
| IndicVoices | 2,634 | 63 | 42 | 90 |
| nepali_cs | 1,446 | 81 | 106 | 132 |

OpenSLR 54 is read speech, so its 157 clips over 15% are more likely noise than a second voice,
and all ten of nepali_cs's highest-overlap clips (77-99%) are from one video, which suggests music
or echo. Until they are heard, read the public sets' buckets as "the detector fired", not as
crosstalk.

**vanilla-s1 minus base Flex**, folded, with each set's own unit resampled:

| | WER | minus base | number errors, minus base | without numbers, minus base |
|---|---|---|---|---|
| gold | 10.76 | −2.64 [−3.28, −2.01] | −284 | −2.04 [−2.66, −1.45] |
| val | 6.93 | −2.31 [−3.10, −1.42] | −283 | −1.89 [−2.73, −1.00] |
| FLEURS | 12.24 | +1.14 [+0.47, +1.81] | +14 | +1.02 [+0.48, +1.56] |
| OpenSLR 54 | 7.96 | −0.14 [−0.42, +0.13] | −218 | +0.34 [+0.07, +0.60] |
| Common Voice | 8.73 | +0.00 [−1.16, +1.05] | −10 | +0.55 [−0.39, +1.23] |
| IndicVoices | 12.59 | −0.13 [−0.69, +0.36] | −501 | +1.26 [+0.93, +1.57] |
| nepali_cs | 12.00 | +0.95 [−0.43, +2.45] | +42 | +0.95 [−0.41, +2.67] |

- **Fine-tuning taught the model to write numbers the way these references do, and that gain
  hides a loss elsewhere.** On OpenSLR 54 and IndicVoices the headline difference is a tie, but
  only because number errors fell by 218 and 501; everything else got worse, by 0.34 and 1.26
  points with intervals clear of zero.
- **FLEURS's +1.14 is mostly not numbers.** Number rows account for about 0.1 point of it; the WER
  over everything else is still +1.02. Clips whose reference holds a number carry +1.54
  [+0.15, +3.27] on a third of the words, which is about half the gap, so "half of it is numbers"
  was the clips, not the numbers themselves. Its top new substitutions are spelling variants
  (`छनौट`/`छनोट`, `सामान्यतया`/`सामान्यतः`) and `802.11` read out as words.
- **On gold the gain grows with crosstalk**: −1.91 on clips with none, −2.95, −4.01, then −4.98
  [−9.00, −2.82] over 15%.

## Not in this round

- **Class 5** (near miss). Class 4 (convention mismatch) became the rulebook's tier-4 tags and
  its tier-1 folds in fold-v4 (D112).
- **Notes while reading**: marking a pair as a convention or a real error from the page needs
  storage of its own, and a decision on what such a mark may change.
- **S/D/I by language** (Nepali, English, numbers) as a reported block; the rows carry script, so
  it is a query once error mining exists.
- **Reference audit:** flag public-set clips where the models agree with each other and not with
  the reference, and listen to the top of that list. FLEURS, OpenSLR 54 and Common Voice
  references are the text the speaker was asked to read, not a transcript of what was said.
