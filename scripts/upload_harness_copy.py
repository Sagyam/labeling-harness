#!/usr/bin/env python
"""Upload the harness code the notebooks score with to the dataset's harness/ folder.

  uv run --no-sync --project backend --with huggingface_hub python scripts/upload_harness_copy.py

The notebooks never import the harness from GitHub: `ftkit.harness_scorer` lays out the flat
copy in the HF dataset's `harness/` -- fold.py with its normalizer and table, and error mining
(D110) -- so a score is always made by the code the dataset was exported with.
Run this after changing any of those files. It changes nothing else in the dataset, so
`exported_at` and every notebook's `check_export` stay as they were.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_REPO = "Sagyam/nepanglish-asr"


def harness_files(root: Path = REPO_ROOT) -> dict[str, Path]:
    """{path in the dataset: local file} for the flat `harness/` copy."""
    services = root / "backend" / "app" / "services"
    files = {
        name: services / name
        for name in (
            "fold.py",
            "normalize.py",
            "error_mining.py",
            "error_store.py",
            "attribution.py",
        )
    }
    files["normalization.yaml"] = root / "config" / "normalization.yaml"
    return {f"harness/{name}": path for name, path in files.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DATASET_REPO)
    parser.add_argument("--dry-run", action="store_true", help="list the files and stop")
    args = parser.parse_args(argv)
    files = harness_files()
    for remote, local in files.items():
        print(f"{local.relative_to(REPO_ROOT)} -> {args.repo}/{remote}")
    if args.dry_run:
        return 0
    from huggingface_hub import CommitOperationAdd, HfApi

    commit = HfApi().create_commit(
        repo_id=args.repo,
        repo_type="dataset",
        operations=[CommitOperationAdd(remote, str(local)) for remote, local in files.items()],
        commit_message="harness/: the scorer and error mining the notebooks use",
    )
    print(commit.commit_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
