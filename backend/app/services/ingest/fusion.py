"""Stage 4 glue: the fusion route, topic labelling, and the seed text of a record.

The reconciliation itself (the prompt, the provider call, the hypothesis shape) lives in
:mod:`app.services.fusion_stage`; this is only how the pipeline invokes it for one episode.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes, Settings
from app.llm.base import LlmResult
from app.llm.openrouter import OpenRouterClient
from app.llm.topic import classify_topic, sample_transcript
from app.llm.vertex import VertexClient
from app.services.forced_align import ForcedAligner
from app.services.fusion_stage import FUSION_KIND, fuse_records
from app.utils.logging import get_logger

from .checkpoint import Checkpoint, request_key
from .job import IngestJob, LockedSession

logger = get_logger(__name__)


def _best_text(record: dict[str, Any]) -> str:
    """The fused transcript when there is one, otherwise the first recogniser's."""
    for hypothesis in record["hypotheses"]:
        if hypothesis.get("kind") == FUSION_KIND:
            return str(hypothesis.get("text") or "")
    return str(record["hypotheses"][0].get("text") or "")


def _fusion_completer(
    session: Session, routes: LlmRoutes, route_name: str
) -> Callable[[list[dict[str, Any]]], LlmResult] | None:
    """A routed, logged completion on the fusion route, or ``None`` when fusion cannot run.

    ``None`` on a dry run -- the recognisers returned canned text and there is nothing to
    reconcile -- and when the route is not configured. Every real request goes through the
    provider client, so it writes its ``llm_requests`` row like any other inference (invariant 6).
    """
    route = routes.routes.get(route_name) if route_name else None
    if route is None or routes.dry_run:
        return None
    client: VertexClient | OpenRouterClient = (
        VertexClient(session, config=routes)
        if route.provider == "vertex"
        else OpenRouterClient(session, config=routes)
    )
    return lambda messages: client.complete(route_name, messages)


def _checkpointed(
    complete: Callable[[list[dict[str, Any]]], LlmResult],
    checkpoint: Checkpoint,
    session: Session,
    route: LlmRoute,
) -> Callable[[list[dict[str, Any]]], LlmResult]:
    """``complete``, answering from the checkpoint what an interrupted run already paid for.

    Windows run in order and each carries the previous one's answer forward, so a resumed run
    rebuilds exactly the requests the first one sent -- until the first it never finished, which
    is paid for as usual. Each fresh answer is committed before it is kept: its ``llm_requests``
    row used to wait for the end of the stage, and a power cut lost the record of every window
    already bought (invariant 6).
    """

    def complete_once(messages: list[dict[str, Any]]) -> LlmResult:
        key = request_key("fusion", route.model_dump(mode="json"), messages)
        replayed = checkpoint.load_completion(key)
        if replayed is not None:
            return replayed
        result = complete(messages)
        session.commit()
        if not result.dry_run:
            checkpoint.save_completion(key, result)
        return result

    return complete_once


def _run_fusion_stage(
    job: IngestJob,
    segment_records: list[dict[str, Any]],
    segments: list[Any],
    session_factory: Callable[[], Session],
    settings: Settings,
    *,
    routes: LlmRoutes,
    aligner: ForcedAligner | None,
    checkpoint: Checkpoint | None = None,
) -> dict[str, Any] | None:
    """Stage 4: fuse the episode's recognisers into one hypothesis per clip (D72).

    Mutates ``segment_records`` in place. Never raises: a stage that fails outright is logged, the
    clips keep their recognisers, and the queue seeds and routes them without a fused text.
    """
    route_name = settings.fusion.route
    route = routes.routes.get(route_name) if route_name else None
    if route is None:
        job.log(
            "Stage 4/6: Fusion skipped -- no fusion route configured; seeds fall back to one "
            "recogniser",
            "warn",
        )
        return None

    clip_paths = {seg.segment_id: Path(seg.clip_path) for seg in segments}
    try:
        with session_factory() as session:
            complete = _fusion_completer(LockedSession(session), routes, route_name)
            if complete is None:
                job.log("Stage 4/6: Fusion skipped on a dry run -- canned text has nothing to fuse")
                return None
            if checkpoint is not None:
                complete = _checkpointed(complete, checkpoint, session, route)
            job.log(
                f"Stage 4/6: Fusing {len(segment_records)} segments with {route.model} "
                f"(windows of ~{settings.fusion.window_target_words} words sent)..."
            )
            report = fuse_records(
                segment_records,
                complete=complete,
                route=route,
                fusion=settings.fusion,
                settings=settings,
                aligner=aligner if aligner is not None and aligner.available else None,
                clip_path_for=clip_paths.__getitem__,
                should_stop=lambda: job.scrammed,
                log=job.log,
                max_workers=settings.ingest.max_segment_concurrency,
            )
            session.commit()
            reused = checkpoint.reused.get("completion", 0) if checkpoint is not None else 0
    except Exception as exc:  # the recognisers' work is paid for; see the docstring
        logger.warning("fusion_stage_failed", error=str(exc))
        job.log(f"Fusion failed ({type(exc).__name__}: {exc}); seeds fall back to one recogniser",
                "warn")  # fmt: skip
        return {"error": f"{type(exc).__name__}: {exc}"}

    if reused:
        job.log(f"Reused {reused} fusion request(s) paid for before the interruption", "success")
    job.log(
        f"Fusion: {report.fused}/{report.segments} segments fused in {report.windows} window(s), "
        f"{report.requests} request(s), {report.thought_tokens:,} thought tokens, "
        f"${report.cost_usd:.3f}"
        + (f" -- {len(report.unfused)} left unfused" if report.unfused else "")
    )
    return report.as_dict()


def _classify_episode_topic(
    job: IngestJob,
    segment_records: list[dict[str, Any]],
    session_factory: Callable[[], Session],
    settings: Settings,
    *,
    routes: LlmRoutes,
) -> dict[str, Any]:
    """Ask a model what this episode is about, when nobody has said.

    Never raises and never fails the ingest. The episode's audio, transcripts and queue are all
    already produced by this point, and a metadata field is not worth throwing them away for -- a
    failed classification leaves the topic empty and an annotator can still type one.

    Returns:
        Metadata to merge into ``episode.json``: the ``topic`` when one was decided, plus
        provenance saying where it came from. Empty when the topic was already set by hand, no
        route is configured, or the attempt failed.
    """
    route = settings.ingest.topic_route
    if not route or (job.metadata or {}).get("topic"):
        return {}
    if route not in routes.routes:
        logger.warning("topic_route_not_configured", route=route)
        return {}

    excerpt = sample_transcript(
        [_best_text(record) for record in segment_records if record["hypotheses"]]
    )
    try:
        with session_factory() as session:
            topic, meta = classify_topic(
                session,
                title=job.title,
                transcript=excerpt,
                route=route,
                config=routes,
            )
            session.commit()
    except Exception as exc:  # see the docstring: a metadata field never fails an ingest
        logger.warning("topic_classification_failed", route=route, error=str(exc))
        job.log("Topic classification failed; leaving the episode's topic empty")
        return {}

    if topic:
        job.log(f"Topic classified as '{topic}'")
        return {"topic": topic, **meta}
    return meta
