#!/usr/bin/env python
"""Screen the distillation corpus against gold's voices, then rebuild its manifest (D101).

python scripts/screen_distill_audio.py              # screen the sources not screened yet
python scripts/screen_distill_audio.py --rescreen   # screen every source again

A source with distill.screen_min_seconds of 2 s windows at or above distill.screen_threshold to
one gold voice is quarantined: it stays out of clips.jsonl, and its closest windows are listed to
be heard. A cleared source joins the manifest. Postgres is only read, for gold's voices.
"""

from __future__ import annotations

import argparse
import json

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import REPO_ROOT, bootstrap
from app.db.session import session_scope
from app.models import Segment
from app.services.distill_corpus import gold_voice_centroids
from app.services.distill_prep import build_manifest, screen_folder
from app.services.voice_clips import current_runs
from app.services.voiceprint import VoiceEmbedder
from sqlalchemy import select


def gold_voices() -> dict:
    """Every voice diarized in an episode that holds a gold clip, as a unit centroid."""
    with session_scope() as session:
        runs = current_runs(session)
        episodes = set(session.scalars(select(Segment.episode_id).where(Segment.pot == "gold")))
        return gold_voice_centroids(
            [(runs[e].voices_jsonb, runs[e].embeddings_jsonb) for e in episodes if e in runs]
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rescreen", action="store_true", help="screen sources already screened")
    args = parser.parse_args(argv)

    distill = bootstrap().distill
    if distill.screen_threshold is None:
        print("distill.screen_threshold is not set: it is measured before the screen runs (D101)")
        return 1
    root = (REPO_ROOT / distill.root).resolve()
    folders = sorted(
        p
        for p in (root / "sources").glob("*")
        if p.is_dir()
        and not p.name.endswith(".partial")
        and (args.rescreen or not (p / "screen.json").is_file())
    )
    gold = gold_voices()
    embedder = VoiceEmbedder()
    if not embedder.available:
        print("the voiceprint model did not load; nothing can be screened")
        return 1
    print(
        f"{len(folders)} source(s) to screen against {len(gold)} gold voices, "
        f"threshold {distill.screen_threshold}, {distill.screen_min_seconds:g} s to quarantine"
    )
    for n, folder in enumerate(folders, 1):
        r = screen_folder(
            folder,
            gold,
            embedder,
            threshold=distill.screen_threshold,
            window_seconds=distill.screen_window_seconds,
            min_seconds=distill.screen_min_seconds,
        )
        closest = f", closest {r.voice} for {r.seconds:g} s" if r.voice else ""
        print(f"[{n}/{len(folders)}] {folder.name}: {r.verdict}{closest}", flush=True)

    report = build_manifest(root)
    print(f"manifest: {report.clips} clips, {report.hours:.1f} h in {root / 'clips.jsonl'}")
    if report.unscreened:
        print(f"not screened, left out: {', '.join(report.unscreened)}")
    for source_id in report.quarantined:
        saved = json.loads((root / "sources" / source_id / "screen.json").read_text("utf-8"))
        where = "; ".join(
            f"{clip} at {start:g} s ({sim:.2f})" for clip, start, sim in saved["windows"]
        )
        print(
            f"QUARANTINED {source_id}: sounds like gold voice {saved['voice']} for "
            f"{saved['seconds']:g} s. Hear: {where}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
