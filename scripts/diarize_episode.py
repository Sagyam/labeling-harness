#!/usr/bin/env python
"""Diarize imported episodes on the Modal GPU and store their speaker turns (D79).

Ingest diarizes every new episode itself; this is for episodes imported before it did, or for
re-running one after the diarizer changes. A new run is appended; the newest is current (D78).

python scripts/diarize_episode.py show-a_ep012
python scripts/diarize_episode.py show-a_ep012 show-a_ep013 --num-speakers 2
"""

from __future__ import annotations

import argparse

import sqlalchemy as sa

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.models import Episode
from app.services.diarization import declared_speaker_count, diarize_audio
from app.services.diarization_import import import_diarization
from app.storage import build_storage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episodes", nargs="+", help="external ids of the episodes to diarize")
    parser.add_argument(
        "--num-speakers", type=int, help="fix the speaker count (default: the declared speakers)"
    )
    parser.add_argument("--actor", default="diarize_episode", help="recorded in audit_logs")
    args = parser.parse_args(argv)

    settings = bootstrap()
    conf = settings.diarization
    if not conf.enabled or not conf.endpoint_url:
        print("diarization is disabled or has no endpoint_url (config/settings.yaml)")
        return 1
    storage = build_storage(settings)

    failed = 0
    for external_id in args.episodes:
        with session_scope() as session:
            episode = session.scalar(sa.select(Episode).where(Episode.external_id == external_id))
            if episode is None or not episode.audio_object_key:
                print(f"{external_id}: {'not found' if episode is None else 'no retained audio'}")
                failed += 1
                continue
            num_speakers = args.num_speakers or declared_speaker_count(episode.metadata_jsonb)
            result = diarize_audio(
                storage.get_bytes(episode.audio_object_key),
                num_speakers=num_speakers,
                settings=settings,
            )
            report = import_diarization(
                session,
                {external_id: result},
                model=conf.model,
                source=conf.endpoint_url,
                actor=args.actor,
            )
        state = "stored" if report.runs_created else "unchanged, already stored"
        print(
            f"{external_id}: {len(result['labels'])} speakers "
            f"(asked {num_speakers or 'auto'}), {len(result['turns'])} turns, {state}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
