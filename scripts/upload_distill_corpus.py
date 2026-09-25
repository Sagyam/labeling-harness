# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub==0.36.2"]
# ///
"""Upload the distillation corpus to its private HF dataset, for Colab (D101).

uv run scripts/upload_distill_corpus.py                 # distill.root -> distill.hf_repo
uv run scripts/upload_distill_corpus.py --dry-run       # list what would be sent

Run it with `uv run`, not the backend's virtualenv: it declares its own pinned huggingface_hub, so
the backend never depends on it. Only clips.jsonl and the sources the screen cleared are sent,
named one by one, so incoming audio, quarantined sources and the refusal log cannot be. Nothing is
sent while any source is unscreened: run scripts/screen_distill_audio.py first. The upload
resumes where it stopped if interrupted. It needs a write token (HF_TOKEN or `hf auth login`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class NotReady(RuntimeError):
    """The corpus has a source the screen has not judged."""


def allowed_patterns(root: Path) -> list[str]:
    """The manifest and every cleared source, as upload patterns.

    Raises:
        NotReady: A source has no screen verdict yet.
    """
    cleared, unscreened = [], []
    for folder in sorted(p for p in (root / "sources").glob("*") if p.is_dir()):
        if folder.name.endswith(".partial"):
            continue
        screen = folder / "screen.json"
        if not screen.is_file():
            unscreened.append(folder.name)
        elif json.loads(screen.read_text(encoding="utf-8"))["verdict"] == "clear":
            cleared.append(folder.name)
    if unscreened:
        raise NotReady(f"not screened yet: {', '.join(unscreened)}")
    return ["clips.jsonl"] + [f"sources/{name}/*" for name in cleared]


def _setting(key: str) -> str:
    """One scalar of the distill section: ``HARNESS_DISTILL__<KEY>`` if set, as the backend's
    settings read it, else config/settings.yaml, without importing the backend."""
    if value := os.environ.get(f"HARNESS_DISTILL__{key.upper()}"):
        return value
    text = (REPO_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8")
    section = text.split("\ndistill:", 1)[1]
    match = re.search(rf"^  {key}:\s*(\S+)", section, flags=re.M)
    if not match:
        raise KeyError(f"distill.{key} is not in config/settings.yaml")
    return match.group(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="list what would be sent")
    args = parser.parse_args(argv)

    root = (REPO_ROOT / _setting("root")).resolve()
    repo = _setting("hf_repo")
    try:
        patterns = allowed_patterns(root)
    except NotReady as exc:
        print(f"{exc}; run scripts/screen_distill_audio.py first")
        return 1
    print(
        f"{len(patterns) - 1} cleared source(s) and the manifest, from {root} to {repo} (private)"
    )
    if not (root / "clips.jsonl").is_file():
        print(f"no manifest in {root}; run scripts/screen_distill_audio.py first")
        return 1
    if args.dry_run:
        print("\n".join(patterns))
        return 0
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo, folder_path=str(root), repo_type="dataset", allow_patterns=patterns
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
