#!/usr/bin/env python
"""How voiceprint suggestions fared against the saved speaker lanes (D99).

python scripts/voiceprint_report.py

Precision: of the words saved with a suggestion, how many the annotator left on the suggested
lane. Moves foreseen: of the words moved off the diarized lane, how many a suggestion named.
Suggestions apply themselves only if this says they should (D99).
"""

from __future__ import annotations

# Importing _bootstrap puts backend/ on sys.path, so `app` is importable below.
from _bootstrap import bootstrap
from app.db.session import session_scope
from app.services.speaker_attribution import saved_speaker_words
from app.services.voiceprint import suggestion_report


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value:.0%}"


def main(argv: list[str] | None = None) -> int:
    settings = bootstrap()
    with session_scope() as session:
        report = suggestion_report(saved_speaker_words(session, settings))
    print(f"words of verified text on saved lanes  {report.words}")
    print(f"moved off the diarized lane            {report.moved}")
    print(f"saved with a suggestion                {report.suggested}")
    print(f"  left on the suggested lane           {report.taken}  ({_pct(report.precision)})")
    print(
        f"moves a suggestion foresaw             {report.moved_suggested}  ({_pct(report.recall)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
