"""The ingest's fusion stage: every clip's recognisers in, one fused hypothesis per clip out (D72).

Runs between transcription and import, over the in-memory segment records the pipeline is about to
write as a manifest. For each clip the fuser answered, it appends one more hypothesis --
``kind: fusion`` -- after the recognisers', spans measured by the CTC aligner, and re-measures
code-mixing on the fused text. That text follows the script policy, so it is the first transcript
in the pipeline whose Latin tokens are English and whose Devanagari tokens are Nepali; Scribe's
`टिम` for ``team`` hid the switch from the old primary-route measurement.

A clip the fuser did not answer is left with its recognisers alone. The queue then seeds it from
a recogniser and the hazard gates send it to review, so a failed window costs speed, not labels.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import FusionSettings, LlmRoute, Settings
from app.llm.base import LlmResult
from app.llm.fusion import PROMPT_VERSION, FusionSegment, fuse
from app.services.analysis import analyze_transcript
from app.services.forced_align import ForcedAligner, align_text_with_fit

#: ``asr_systems.kind`` and the manifest key that sets it. A fusion hypothesis is derived from the
#: recognisers', so it is never a disagreement signal and never an input to another fusion.
FUSION_KIND = "fusion"


def fusion_system_id(route: LlmRoute) -> str:
    """The system a fused hypothesis is recorded under: the route's id plus the prompt version.

    Hypotheses are unique per (segment, system) and immutable, so a changed prompt has to be a
    new system -- which is also what keeps two prompts' output from ever being compared as one.
    """
    base = route.system_id or f"fusion-{route.model}"
    return f"{base}-{PROMPT_VERSION}"


@dataclass
class FusionStageReport:
    """What the stage did, for the ingest log and the completion summary."""

    system_id: str
    segments: int = 0
    fused: int = 0
    unfused: list[str] = field(default_factory=list)
    windows: int = 0
    requests: int = 0
    thought_tokens: int = 0
    cost_usd: float = 0.0
    codes: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["unfused"] = len(self.unfused)
        return out


def fuse_records(
    records: list[dict[str, Any]],
    *,
    complete: Callable[[list[dict[str, Any]]], LlmResult],
    route: LlmRoute,
    fusion: FusionSettings,
    settings: Settings,
    aligner: ForcedAligner | None,
    clip_path_for: Callable[[str], Path],
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = lambda _message: None,
    max_workers: int = 4,
) -> FusionStageReport:
    """Fuse one episode's segment records in place.

    Args:
        records: Manifest segment records, each with its recognisers' hypotheses. Mutated: a
            fused hypothesis is appended and ``scores`` re-measured on it.
        complete: One routed, logged chat request.
        route: The fusion route's configuration, for the system id and provenance.
        fusion: Window sizing.
        settings: Thresholds for the re-measured analysis.
        aligner: The CTC aligner, or ``None`` for no word spans.
        clip_path_for: Segment id to its clip on disk, for alignment.
        should_stop: Checked between requests (the ingest's stop button).
        log: The ingest log.
        max_workers: Alignment threads; the ONNX session is shared and thread-safe.
    """
    system_id = fusion_system_id(route)
    report = FusionStageReport(system_id=system_id, segments=len(records))
    ordered = sorted(records, key=lambda r: float(r["start_time"]))
    by_key = {str(r["segment_id"]): r for r in ordered}

    segments = [
        FusionSegment(
            key=str(r["segment_id"]),
            start=float(r["start_time"]),
            end=float(r["end_time"]),
            hypotheses=tuple(
                str(h.get("text") or "")
                for h in r["hypotheses"]
                if h.get("kind", "asr") != FUSION_KIND
            ),
        )
        for r in ordered
    ]
    outcome = fuse(
        segments,
        complete=complete,
        target_seconds=fusion.window_target_seconds,
        lookahead_seconds=fusion.lookahead_seconds,
        carryover_seconds=fusion.carryover_seconds,
        max_segments=fusion.max_window_segments,
        max_depth=fusion.max_depth,
        should_stop=should_stop,
        log=log,
    )
    report.unfused = list(outcome.unfused)
    report.windows = len({c.window_index for c in outcome.calls})
    report.requests = len(outcome.calls)
    report.thought_tokens = sum(c.thought_tokens or 0 for c in outcome.calls)
    report.cost_usd = outcome.cost_usd
    versions = {c.window_index: c.model_version for c in outcome.calls if c.model_version}

    def align(key: str, text: str) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
        if aligner is None or not text.strip():
            return None, None
        return align_text_with_fit(aligner, clip_path_for(key), text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        placed = {key: pool.submit(align, key, fused.text) for key, fused in outcome.fused.items()}
        for key, fused in outcome.fused.items():
            record = by_key[key]
            words, fit = placed[key].result()
            record["hypotheses"].append(
                {
                    "system_id": system_id,
                    "model_id": route.model,
                    "kind": FUSION_KIND,
                    "text": fused.text,
                    "words": words,
                    # How well the fused text explains the clip, against the aligner's own
                    # reading -- the acoustic leash the hazard score reads (D74). None when no
                    # aligner ran, which the score treats as unmeasured, not as a fit.
                    "acoustic": fit,
                    # Provenance, not a hypothesis field: the importer files unknown keys under
                    # `metadata_jsonb`. The model version is what makes a mid-build vendor
                    # rotation visible after the fact (new-plan §3.7).
                    "fusion": {
                        "code": fused.code,
                        "window": fused.window_index,
                        "depth": fused.window_depth,
                        "prompt_version": PROMPT_VERSION,
                        "model_version": versions.get(fused.window_index),
                        "thinking_budget": route.thinking_budget,
                    },
                }
            )
            analysis = analyze_transcript(
                fused.text,
                duration_seconds=float(record["end_time"]) - float(record["start_time"]),
                settings=settings,
            )
            scores = record.setdefault("scores", {})
            scores["cmi"] = analysis.cmi
            scores["code_switch_density"] = analysis.code_switch_density
            scores["switch_point_count"] = analysis.switch_point_count
            scores["discourse_marker_count"] = analysis.discourse_marker_count
            scores["flags"] = sorted(set(scores.get("flags") or []) | set(analysis.flags))
            report.fused += 1
            report.codes[fused.code] = report.codes.get(fused.code, 0) + 1
    return report
