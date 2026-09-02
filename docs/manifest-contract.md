# Manifest contract

The manifest is the boundary between an upstream GPU pipeline and the harness. In-app ingestion
(see [architecture](architecture.md#ingestion-and-cloud-asr)) produces the same rows directly, but a
manifest directory remains a first-class way in — `scripts/import_manifest.py` imports one.

The executable form of this contract is the pair of JSON Schemas in `backend/app/schemas/`
(`episode.schema.json`, `segment.schema.json`), validated before a single row is written. What
follows is the readable form; where the two disagree, the schemas win.

```text
export_<episode_id>/
  episode.json
  segments.jsonl
  clips/<segment_id>.flac    16 kHz mono FLAC
  peaks/<segment_id>.json    optional; generated at import when absent
```

## episode.json

```json
{
  "episode_id": "show-a_ep012",
  "show_id": "show-a",
  "title": "Example podcast",
  "source_uri": "https://...",
  "published_at": "2026-01-01",
  "source_audio_checksum": "sha256:...",
  "duration_seconds": 4821.3,
  "pipeline_version": "nb-v3",
  "pipeline_commit": "a1b2c3d"
}
```

## segments.jsonl — one object per line

```json
{
  "segment_id": "show-a_ep012_0042",
  "episode_id": "show-a_ep012",
  "speaker_id": "SPEAKER_01",
  "start_time": 123.4,
  "end_time": 135.2,
  "clip_path": "clips/show-a_ep012_0042.flac",
  "clip_checksum": "sha256:...",
  "p_en": 0.31,
  "lid": "ne",
  "hypotheses": [
    {
      "system_id": "qwen-ne",
      "model_id": "sidskarki/Qwen3-ASR-Nepali",
      "text": "So today म Python मा loops बारे कुरा गर्छु।",
      "avg_logprob": -0.34,
      "no_speech_prob": 0.01,
      "words": [
        {"word": "So", "start": 0.0, "end": 0.31, "confidence": 0.92,
         "predicted_language": "en", "predicted_script": "latin"}
      ]
    }
  ],
  "scores": {
    "cer_between_hypotheses": 0.18,
    "word_disagreement_rate": 0.22,
    "script_conflict_rate": 0.05,
    "code_switch_density": 0.42
  },
  "flags": ["repeated_ngram"]
}
```

Rules:

- `hypotheses` must contain at least one entry; `words` is optional and may be absent or empty.
- Word `start` and `end` are **seconds from the start of the clip**, not from the start of the
  episode. The segment's own `start_time` and `end_time` are episode-relative, so the two live on
  different timelines on purpose: a word list travels with the clip beside it, and
  `start_time + word.start` is the episode offset when one is wanted. The harness rebases nothing
  at import — a manifest that supplies episode-relative word times is stored exactly as written and
  is simply wrong. See [decision D26](decisions.md).
- `scores` may be partially absent; the harness recomputes none of them, it stores what it receives
  and treats a missing score as null.
- `flags` is top-level, not inside `scores`, and each entry should be one of the seven rule flags
  listed in [architecture.md](architecture.md#rule-flags-computed-at-import) — a name from outside
  that list is stored but scores nothing. Flags are the exception to the rule above: the importer
  recomputes them over the hypotheses and stores the **union** of what it received and what it
  computed, so an omitted `flags` key costs nothing.
- Clips are **16 kHz mono FLAC**. Reject WAV or MP3 clips at import with a clear error — the source
  is already lossy and re-encoding the exact audio you will train on is not acceptable. The original
  episode file is archived separately and is not needed by the harness.
- Import is idempotent, keyed on `(segment_id, system_id)` for hypotheses and `segment_id` for
  segments. Re-importing an unchanged export is a no-op. Re-importing with a changed clip checksum
  is an error unless `--allow-clip-change` is passed.

## Transcript policy

The target output format is English in Latin script, Nepali in Devanagari:

```text
So today म Python मा loops बारे कुरा गर्छु।
```

The policy is not enforced by the system. It is recorded as a `policy_version` string on every
label, so an export made a year from now is still interpretable. There is deliberately no policy
linter, guideline editor or in-app guideline panel.
