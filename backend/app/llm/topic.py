"""Give an episode a topic label, from what was actually said in it (D57).

The ingest form has always had a free-text ``Topic / Domain`` box and it was nearly always left
empty, which is the worst possible outcome for a variable whose whole purpose is stratification:
an unevenly-filled column cannot be reported on. So the topic is inferred instead, and the box
becomes an override rather than a chore.

Two design choices carry the module:

* **It classifies the transcript, not the title.** A YouTube title is marketing, an uploaded file
  has no title worth the name, and both are short enough that a model would be pattern-matching a
  few words. The transcript is the episode. The title rides along as a hint only.
* **It picks from a closed taxonomy, or it fails.** A free-text answer would produce a hundred
  one-episode categories with no two spelled alike. A label outside :data:`TOPIC_LABELS` is
  treated as no answer at all, and the episode keeps an empty topic that a human can fill in --
  which is strictly better than a category that exists once.

One call per episode, at import, after ASR. It is routed and logged like every other inference
(invariant 5), so what topic labelling costs shows up in ``llm_requests`` beside the transcripts.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.config import LlmRoutes
from app.llm.base import LlmRouteNotConfigured
from app.llm.openrouter import OpenRouterClient
from app.llm.vertex import VertexClient
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: The closed vocabulary. Chosen to cut Nepali-English podcast material into groups big enough to
#: report on, because a taxonomy that splits the corpus into singletons answers nothing. Adding a
#: label is cheap; renaming one is not, since every episode already labelled keeps the old string.
TOPIC_LABELS: tuple[str, ...] = (
    "technology",
    "business_finance",
    "politics_governance",
    "news_current_affairs",
    "education_career",
    "health_wellness",
    "science_environment",
    "sports",
    "entertainment_media",
    "music_arts",
    "culture_religion",
    "travel_food",
    "lifestyle_relationships",
    "comedy",
    "personal_story",
    "other",
)

#: How much transcript the model is shown. Roughly 2k tokens of Devanagari: enough that the sample
#: spans the episode, small enough that one call per episode stays a rounding error next to ASR.
MAX_TRANSCRIPT_CHARS = 6000

INSTRUCTION = (
    "You are labelling the TOPIC of a Nepali-English code-switched podcast episode.\n\n"
    "You will be given the episode title and an excerpt sampled evenly across the whole "
    "episode, so the excerpt jumps between moments and will read disjointedly. That is expected: "
    "judge what the episode is ABOUT overall, not what any one passage says.\n\n"
    "Choose exactly ONE label from this list:\n"
    + "\n".join(f"- {label}" for label in TOPIC_LABELS)
    + "\n\nRULES:\n"
    "1. Answer with the label alone -- no prose, no punctuation, no explanation, no code fence.\n"
    "2. Use a label from the list, spelled exactly as written above. Never invent one.\n"
    "3. Ignore advertising, sponsor reads and channel promotion; they are not the topic.\n"
    "4. If the episode fits no label, or ranges too widely for one, answer `other`.\n"
)


def sample_transcript(texts: list[str], *, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Join a spread of segment transcripts, in order, within a character budget.

    Sampled evenly across the episode rather than taken off the front. Nepali podcasts open with
    greetings, sponsor reads and channel promotion, so a prefix of the transcript is systematically
    the least topical part of it -- a front-loaded excerpt would classify the advertising.

    Args:
        texts: Every segment's primary hypothesis, in episode order.
        max_chars: Budget for the joined result.

    Returns:
        The chosen segments joined by spaces, in episode order. Empty when nothing was said.
    """
    usable = [text.strip() for text in texts if text and text.strip()]
    if not usable:
        return ""

    total = sum(len(text) for text in usable) + len(usable) - 1
    if total <= max_chars:
        return " ".join(usable)

    # Estimate how many segments fit, then spread that many picks from the first to the last so
    # the excerpt reaches the end of the episode. The budget is still enforced as we go: segment
    # lengths vary by an order of magnitude, so the estimate is a starting point, not a promise.
    mean_cost = sum(len(text) for text in usable) / len(usable) + 1
    wanted = min(len(usable), max(1, int((max_chars + 1) / mean_cost)))
    stride = (len(usable) - 1) / (wanted - 1) if wanted > 1 else 0.0

    chosen: list[str] = []
    used = 0
    for i in range(wanted):
        text = usable[min(len(usable) - 1, round(i * stride))]
        cost = len(text) + (1 if chosen else 0)
        if used + cost > max_chars:
            break
        chosen.append(text)
        used += cost

    return " ".join(chosen)


def _client_for(
    session: Session,
    route: str,
    *,
    config: LlmRoutes,
    client: httpx.Client | None,
) -> OpenRouterClient | VertexClient:
    """The provider the route names, as a client with a ``complete`` of the same shape."""
    route_config = config.routes.get(route)
    if route_config is None:
        raise LlmRouteNotConfigured(f"no route named {route!r} in llm_routes.yaml")
    if route_config.provider == "vertex":
        return VertexClient(session, config=config, client=client)
    return OpenRouterClient(session, config=config, client=client)


def _extract_label(content: str) -> str | None:
    """Find a taxonomy label in a completion, tolerating a fence, quotes or a sentence.

    Longest match first, so ``music_arts`` is never read out of a reply that says
    ``entertainment_media`` -- no label is a substring of another today, but the taxonomy is
    expected to grow and this removes the ordering trap before it is set.
    """
    lowered = content.lower()
    for label in sorted(TOPIC_LABELS, key=len, reverse=True):
        if re.search(rf"(?<![a-z_]){re.escape(label)}(?![a-z_])", lowered):
            return label
    return None


def classify_topic(
    session: Session,
    *,
    title: str,
    transcript: str,
    route: str,
    config: LlmRoutes,
    client: httpx.Client | None = None,
    dry_run: bool | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Label one episode's topic from its transcript.

    Args:
        session: Session the ``llm_requests`` row is written to.
        title: The episode title, as a hint. May be empty.
        transcript: An excerpt of what was said, ideally from :func:`sample_transcript`.
        route: Name of the text route doing the classification.
        config: Routing table.
        client: Optional shared HTTPX client.
        dry_run: Override the configured dry-run mode.

    Returns:
        ``(topic, metadata)``. ``topic`` is a member of :data:`TOPIC_LABELS`, or ``None`` when
        there was nothing to classify, the run was dry, or the model answered off-taxonomy.
        ``metadata`` is provenance for the episode's ``metadata_jsonb`` and always says which of
        those happened.

    Raises:
        LlmRouteNotConfigured: The route does not exist.
        LlmRequestFailed: The provider could not be reached. Deliberately not caught here: whether
            a missing topic is worth failing an ingest over is the caller's call, not this
            module's.
    """
    if not transcript.strip():
        return None, {"topic_error": "no_transcript"}

    prompt = f"{INSTRUCTION}\n\nTITLE: {title.strip() or '(none)'}\n\nEXCERPT:\n{transcript}"
    llm = _client_for(session, route, config=config, client=client)
    result = llm.complete(
        route, [{"role": "user", "content": prompt}], dry_run=dry_run, temperature=0.0
    )
    if result.dry_run:
        return None, {"topic_source": "dry_run"}

    label = _extract_label(result.text)
    if label is None:
        # Not retried. A model that answered off-taxonomy once at temperature 0 will do it again,
        # and an empty topic an annotator can fill in is a better outcome than a billed retry loop.
        logger.warning("topic_off_taxonomy", route=route, answer=result.text[:120])
        return None, {"topic_error": "off_taxonomy", "topic_model": result.model}

    return label, {"topic_source": "llm", "topic_model": result.model}
