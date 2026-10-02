# WER breakdown

A single WER hides why a model lost points. On 2026-10-01, 03b charged the fine-tune +1.14 on
FLEURS, and about half of that turned out to be number formats rather than hearing. Every
evaluation therefore reports a breakdown beside WER and its S/D/I split, and every error is kept
for analysis rather than read once and thrown away.

## Direction: smaller folding, finer breakdown

The owner's direction (2026-10-01): **folding rules get smaller and the breakdown gets finer.**

A folding rule decides, silently and for good, that two spellings are one word. A large static
table of such decisions always misses cases and sometimes forgives a real mistake. The breakdown
decides nothing: it only says what kind of error an error is, so it can be as fine as is useful
without changing the score. Nuance belongs there.

- **A breakdown tags errors, it never forgives them.** Folded WER is whatever `fold.py` says; no
  block here changes it.
- **Shrinking `fold.py` is a later, separate step** (a fold-v4, with a decision entry): every
  WER moves when it lands. The breakdown comes first, so that when folding shrinks, the WER it
  adds is already explained by kind.

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
   rules that need maintaining and new ways to be wrong. The detector needs care, though:
   `fold.is_number` reads a number word at the start of a token and accepts any remainder, so
   `तिनीहरूका` ("their"), `छो` and `ElevenLabs` are numbers to it today.
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

## Build

What has to change, in the order to build it. One commit per numbered step at least, each with its
tests first (AGENTS.md). Steps 1-3 are the harness's code that the notebooks also run; 4-6 the
harness's page; 7-9 the notebooks; 10 the documents.

### What this round builds and what it leaves

Classes 1-3 are built. Classes 4 (convention) and 5 (near miss) are **not** classified in this
round: they come from reading what 1-3 surface, and land later as a new classifier version. The
rows carry what that reading needs — the alignment's `similarity` and each side's romanization —
so the page can sort and filter by them without anyone having decided a threshold.

### 1. The classifier: `backend/app/services/error_mining.py` (new)

One module, pure, importing only `fold.py` and the standard library, so the dataset's `harness/`
copy can carry it beside `fold.py` and the notebooks run the same code the page reads.

- `MINER_VERSION = "mine-v1"`. Bump it whenever a column's meaning changes.
- `pairs(ref_text, hyp_text, *, alignment=None) -> list[dict]`: one row per `AlignOp` of
  `fold.word_errors(ref, hyp)`. Takes an existing `Alignment` so a caller that already aligned
  (the scorer) does not align again.
- `rows(run, set, clips) -> list[dict]`: `clips` is an iterable of `{clip_id, group, ref, hyp,
  overlap_share, by}`; adds the clip-level columns to each pair.
- `write(rows, path)` and `read(path)`: the only writer and reader of the format, through DuckDB,
  so the notebook and the harness cannot drift. The Parquet file's key-value metadata holds
  `run`, `set`, `fold_version`, `miner_version`, `created_at` (UTC).

Columns, one row per aligned pair:

| column | type | meaning |
|---|---|---|
| `run` | str | the run's name (`vanilla-s1`) |
| `set` | str | `gold`, `val` or a `evalkit.BENCHMARKS` name |
| `clip_id` | str | segment `external_id` on gold/val; the benchmark's clip id otherwise |
| `group` | str | the resampling unit: episode, speaker, sentence or video |
| `pos` | int | the pair's index in the clip's alignment, so a clip reassembles in order |
| `ref`, `hyp` | list[str] | the words as written; `ref` empty for an insertion, `hyp` for a deletion |
| `kind` | str | `fold.AlignOp.kind`: `match`, `fold`, `merge`, `sub`, `del`, `ins` |
| `identical` | bool | a `match` whose words are the same string |
| `forgiven` | str or null | for a non-identical non-error: `spelling`, `script`, `number`, `merge` |
| `ref_script`, `hyp_script` | str or null | `dev`, `lat`, `mix`, `none`; null on the absent side |
| `number` | bool | either side is a number (step 2's detector) |
| `similarity` | float | `AlignOp.similarity`: romanized similarity of a substitution, 1.0 otherwise |
| `ref_roman`, `hyp_roman` | str | `fold.romanized` of each side, words joined by a space |
| `overlap_share` | float or null | the clip's measured overlap; null when never measured |
| `overlap_bucket` | str | `none`, `0-5%`, `5-15%`, `>15%`, `unmeasured` (as `clip_classes`) |
| `by` | str or null | the set's own split column (`scenario`, `speech_type`), null elsewhere |

`forgiven`: a `fold` step is `number` when both sides are numbers, `script` otherwise; a `merge`
is `merge`; a non-identical `match` is `spelling`.

Tests (`backend/tests/test_error_mining.py`, not `db`):

- **The rows reproduce the score**: on a fixture of real gold and FLEURS pairs, `sub`/`del`/`ins`
  counts and `sum(len(ref))` equal `word_errors`' S/D/I and `ref_words`, clip by clip. This is the
  test everything else rests on.
- Every row of a clip, ordered by `pos`, gives back the clip's folded tokens on both sides.
- One case per `forgiven` value and per script value, including `queriesहरू` as `mix`.
- `write` then `read` round-trips rows and metadata; a file with another `miner_version` is
  refused by `read` with a clear error.

### 2. The number detector

`fold.is_number` reads a number word at the start of a token and accepts any remainder, so
`तिनीहरूका`, `छो` and `ElevenLabs` are numbers to it. It decides no fold (only tagging calls it:
`model_eval.word_class_counts`), so tightening it moves no WER and needs no fold version.

- Tests first: those three are not numbers; `45`, `2007`, `06:30`, `2.4`, `पैंतालीस`, `forty`,
  `तेस्रो`, `1st`, `दुईटा`, `45 लाखमा`'s `लाखमा` are. The builder adds the false positives a
  sample of tagged words shows; every addition is a test.
- Move `model_eval._NOT_NUMBERS` (`छ`, `एक`) into the detector, so the Models page's word classes
  and the miner agree.
- The Models page's number word class changes as a result; say so in the commit.

### 3. Public sets' crosstalk: `scripts/measure_benchmark_overlap.py` (new)

Each public set measured once with `app/services/overlap.py`, written to
`benchmarks/overlap/<set>.parquet` (`clip_id`, `overlap_share`, spans) and uploaded to the model
repo, where the notebooks and the harness both read it.

- The clip ids are `evalkit.benchmark_row`'s. Reuse the loaders rather than writing new ones; if
  they cannot be imported from `backend/`, the script runs in a Colab CPU runtime instead.
- **Streams**: one parquet shard or tar at a time, audio dropped after measuring; downloads go
  under `data/`, never `/tmp`, which is RAM on the owner's machine. Prints progress per shard and
  resumes from what is written.
- About 25 s of CPU per hour of audio. Then the owner listens to each set's ten highest-overlap
  clips before its buckets are believed (block 1).

### 4. Where the files live in the harness, and how they arrive

Beside the model they belong to: `data/models/asr/<slug>/errors/<set>.parquet`, the folder D83
already reads. No table and no migration: a new decision entry records this (step 10).

- **Upload**: `POST /models/{slug}/errors`, multipart, one or more `.parquet`. Each is read with
  `error_mining.read`, refused whole on a missing column, an unknown `set`, a `run` that differs
  between files, or a size over a cap (50 MB). The stored name comes from the validated `set`,
  never from the uploaded file name. Writes an `audit_logs` row.
- **Rescan** (`POST /models/rescan`, `scripts/import_models.py`) also lists what is in `errors/`,
  so a folder copied from the hub with `errors/` in it needs no upload.
- **Backfill**: `scripts/mine_errors.py <slug>` derives the files for a model already imported,
  from `model_eval_clips` (gold/val: the reference snapshot and the model text) and, when the
  folder has them, `benchmarks/<set>.jsonl` (public: `ref`, `hyp`). The same `error_mining.rows`,
  so it equals what a notebook would write. This is how 03a/03b's runs get files without a GPU.

### 5. Reading them: `backend/app/services/error_store.py` (new) and the API

DuckDB over the model's files; each request opens its own in-memory connection, reads only paths
built from validated names, and parametrises every filter.

Endpoints, all under the models router and its auth:

- `GET /models/{slug}/errors`: the files: set, run, rows, errors, WER and S/D/I from the rows,
  `fold_version` and whether it is the current one, `miner_version`.
- `GET /models/{slug}/errors/{set}/breakdown?base=<slug>`: WER with S/D/I; **block 1** per bucket
  (clips, share of words, share of errors, WER with S/D/I); **block 2** (number errors with S/D/I
  and share, WER without them, clips with a number in the reference and their WER); and with
  `base`, each block's difference against the base model's file for the same set. Intervals
  resample `group` with `model_eval`'s seeded bootstrap.
- `GET /models/{slug}/errors/{set}/confusion`: **block 3**. Filters: `kind`, `forgiven`,
  `ref_script`, `hyp_script`, `number`, `overlap_bucket`, `by`, `similarity_min`/`_max`. Groups
  by `(ref, hyp)`, with an option to fold A→B and B→A into one row; each row has its count and
  share of its kind and, with `base`, its count there and the change. Paged.
- `GET /models/{slug}/errors/{set}/pairs`: the occurrences behind a filter or a confusion row,
  each with ±5 words of context from its clip. `sample=N&seed=S` draws N at random instead of the
  top: this is how a class is read.
- `GET /models/{slug}/errors/{set}/clips/{clip_id}`: one clip's rows in `pos` order, in the shape
  `AlignedDiff` already renders (`kind`, `ref`, `hyp`, `similarity`).

Tests: `error_store` against fixture files written by `error_mining.write` (not `db`); each block's
numbers checked by hand on a small fixture; the API's upload refusals and the path rule (`db`, for
the model lookup).

**When the page and the file disagree.** A gold/val file was scored against the export's labels;
the page's run against the current ones. When the file's WER differs from the imported run's for
the same split, the panel says so, with both numbers. Neither is wrong; the labels moved.

### 6. The page: an Errors section on the Models page

Below the existing run summary, for the selected model, without changing the clip table.

- **Import**: a button taking `.parquet` files, and a list of what is loaded (set, run, versions;
  a stale `fold_version` warned as the clip panel already does).
- **Set and base pickers**: one set at a time; base is any other model with a file for that set.
- **Blocks 1 and 2** as tables beside the set's WER, each against base when one is picked. Clicking
  a bucket filters everything below it.
- **Confusion table** with the filters of step 5 as chips, sorted by count or by change against
  base; a row opens its occurrences.
- **Occurrences** with context, and a **Sample 50** button with a visible seed, for reading a class.
- **A clip**: on gold and val, the existing `ClipPanel` (audio, diff), found by `external_id`; on a
  public set, the alignment alone, since the harness does not hold that audio.

Every text on the panel comes from a model, a label or a public dataset. React escapes it; no
`dangerouslySetInnerHTML`.

`npm run build` passes. Driven once with Playwright on vanilla-s1's backfilled files: import,
filter, sample, open a gold clip and a FLEURS clip.

### 7. The notebooks write the files: `notebooks/src/evalkit.py`

- `ftkit.harness_scorer`'s `per_clip` keeps each clip's `Alignment` (it already makes it), so
  mining aligns nothing again.
- `evaluate_run` writes `errors/<split>.parquet` per split and `run_benchmarks` writes
  `errors/<set>.parquet` per set, through `error_mining`, as each finishes, like every other file.
  The public sets' `overlap_share` comes from `benchmarks/overlap/<set>.parquet` on the hub when it
  exists, else `unmeasured`.
- Each split's and set's metrics gain a `breakdown` object with blocks 1-3 (block 3: the top 20
  rows of each kind), computed by the same queries as the page (`error_store` reads a file;
  passing it the rows in memory is fine). The report cells print it beside WER.
- `harness/` gains `errors/` in what the Models page copies (`build_flex.py`'s download line).
- `error_mining.py` must reach the dataset's `harness/` copy, the way `fold.py` does; the builder
  finds that step (it is not in `export.py`) and the next upload carries it. The upload does not
  change `exported_at`, so `check_export` keeps passing.
- `notebooks/requirements.txt`: add `duckdb`.
- Tests in `backend/tests/test_eval_kit.py`: `evaluate_run` and `run_benchmarks` on the existing
  fakes write files whose rows reproduce the metrics they wrote.
- `python notebooks/src/build_all.py`, and commit the regenerated notebooks with the kit.

### 8. The report: `07_Report`

Blocks 1 and 2 per run and set, beside the WER table, from each metrics file's `breakdown`.
Nothing recomputed.

### 9. Dependency

`duckdb` in `backend/pyproject.toml`, then `uv lock`, both committed (D43). No `pyarrow`, no
`pandas`: DuckDB reads and writes Parquet itself.

### 10. Documents

- `docs/decisions.md`: a new entry. Error rows are Parquet files beside the model, read by DuckDB,
  not Postgres tables: derived, write-once, analytical, rebuilt from transcripts. Reversal cost: a
  migration and an importer that loads the files into a table; the classifier and the queries
  carry over.
- `docs/architecture.md`: the Models page section, the new endpoints, the `errors/` folder.
- This file: replace the build plan with what was built and what was measured.
- `AGENTS.md`, Gotchas: error rows are derived; never edit one, regenerate it; a file is only
  comparable with another of the same `fold_version` and `miner_version`.

## Not in this round

- **Classes 4 and 5** (convention mismatch, near miss): after reading what 1-3 surface.
- **Notes while reading**: marking a pair as a convention or a real error from the page needs
  storage of its own, and a decision on what such a mark may change.
- **S/D/I by language** (Nepali, English, numbers) as a reported block; the rows carry script, so
  it is a query once error mining exists.
- **Reference audit:** flag public-set clips where the models agree with each other and not with
  the reference, and listen to the top of that list. FLEURS, OpenSLR 54 and Common Voice
  references are the text the speaker was asked to read, not a transcript of what was said.
