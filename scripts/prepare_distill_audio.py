#!/usr/bin/env python
"""Cut the owner's downloaded audio into the distillation corpus (D101).

python scripts/prepare_distill_audio.py              # every new source in distill.incoming
python scripts/prepare_distill_audio.py --incoming ~/Downloads/podcasts

Each source is cut like ingest's stages 1-2 into distill.root/sources/<source_id>/, one progress
line per source; a finished source is skipped, so the script can be re-run after an interruption.
Postgres is only read, for the video ids already in the harness, and must be up: that check is
not skippable (D101).
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import REPO_ROOT, bootstrap
from app.db.session import session_scope
from app.models import Episode
from app.services.distill_corpus import pair_inputs
from app.services.distill_prep import prepare_source
from app.services.silero_vad import SileroVAD
from app.services.youtube import InvalidYouTubeUrl, parse_video_id
from sqlalchemy import select


def known_video_ids() -> set[str]:
    """Video ids of every episode in the harness: labelled audio, possibly gold or val."""
    with session_scope() as session:
        uris = session.scalars(
            select(Episode.source_uri).where(Episode.source_uri.is_not(None))
        ).all()
    ids = set()
    for uri in uris:
        try:
            ids.add(parse_video_id(uri))
        except InvalidYouTubeUrl:
            continue
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--incoming", type=Path, help="folder of audio + .info.json (default: distill.incoming)"
    )
    args = parser.parse_args(argv)

    settings = bootstrap()
    distill = settings.distill
    root = (REPO_ROOT / distill.root).resolve()
    incoming = (args.incoming or REPO_ROOT / distill.incoming).expanduser().resolve()
    inputs = pair_inputs(p for p in incoming.iterdir() if p.is_file()) if incoming.is_dir() else []
    if not inputs:
        print(f"no audio in {incoming}")
        return 1

    vad = SileroVAD()
    if not vad.available:
        print("the Silero VAD model did not load; the labelled corpus was cut with it: stopping")
        return 1
    known = known_video_ids()
    print(
        f"{len(inputs)} source(s) in {incoming} | {len(known)} video ids already in the harness | "
        f"blocked channels: {', '.join(distill.blocked_channels) or 'none'}"
    )

    counts = {"prepared": 0, "done": 0, "refused": 0}
    hours = 0.0
    for n, inp in enumerate(inputs, 1):
        r = prepare_source(
            inp,
            root=root,
            known_video_ids=known,
            blocked_channels=distill.blocked_channels,
            vad=vad,
            workers=settings.ingest.cpu_workers,
        )
        counts[r.status] += 1
        hours += r.speech_seconds / 3600
        what = {
            "prepared": f"{r.clips} clips, {r.speech_seconds / 60:.1f} min of speech, "
            f"{r.seconds_taken:.0f} s",
            "done": "already in the corpus",
            "refused": r.reason,
        }[r.status]
        channel = r.source.channel or "no channel"
        print(
            f"[{n}/{len(inputs)}] {r.source.source_id} ({channel}): {r.status}, {what}", flush=True
        )
    print(
        f"prepared {counts['prepared']}, already done {counts['done']}, "
        f"refused {counts['refused']} | "
        f"{hours:.1f} h of speech added | corpus: {root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
