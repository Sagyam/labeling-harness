"""Pure parsers for Vertex AI ``generateContent`` responses.

Everything here is a function of the response body and nothing else -- no client, no session, no
network -- so the failure shapes Vertex actually produces (withheld candidates, thinking switches,
missing usage metadata, offset timings) can be tested without mocking a transport. The client
that sends the requests lives in :mod:`app.llm.vertex`.
"""

from __future__ import annotations

from typing import Any

#: ``finishReason`` values that mean the answer was withheld rather than finished.
#:
#: This is the field that actually carries a block on Gemini 3.x (D45). Measured over 229 ad hoc
#: calls: ``promptFeedback`` was never populated once, not even on a forced block -- the block
#: arrives as ``candidates[0].finishReason: "SAFETY"`` with a populated ``safetyRatings`` and a
#: ``content`` of ``{"role": "model"}`` carrying no parts. Reading only ``promptFeedback`` sees an
#: ordinary empty answer, which for ``audio_chat`` is a legitimate result (``ASR_PROMPT`` asks for
#: an empty string over silence) -- so a blocked clip became an empty hypothesis, logged as
#: succeeded.
#:
#: ``PROHIBITED_CONTENT``, ``BLOCKLIST`` and ``SPII`` come from non-configurable filters that no
#: ``safetySettings`` threshold can switch off, so they must be caught here regardless of D39's
#: ``OFF``. ``MAX_TOKENS`` and ``RECITATION`` are not safety, but they mean the same thing for a
#: transcript: what came back is not what the model was asked for.
WITHHELD_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "SPII",
        "RECITATION",
        "IMAGE_SAFETY",
        "MAX_TOKENS",
        "OTHER",
    }
)


def withheld_reason(body: dict[str, Any]) -> str | None:
    """Why this response carries no usable answer, or ``None`` if it finished normally.

    Checks both halves of Gemini's split: ``promptFeedback.blockReason`` is the *input* being
    refused, ``candidates[0].finishReason`` is the *output* being withheld. The harness watched
    only the first for a long time and Gemini only ever uses the second (D45).
    """
    block_reason = (body.get("promptFeedback") or {}).get("blockReason")
    if block_reason:
        return f"promptFeedback.blockReason={block_reason}"

    for candidate in body.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        finish = candidate.get("finishReason")
        if finish in WITHHELD_FINISH_REASONS:
            blocked = [
                r.get("category")
                for r in candidate.get("safetyRatings") or []
                if isinstance(r, dict) and r.get("blocked")
            ]
            detail = f" ({', '.join(c for c in blocked if c)})" if blocked else ""
            return f"finishReason={finish}{detail}"
    return None


def apply_thinking(
    generation_config: dict[str, Any],
    reasoning_enabled: bool | None,
    budget: int | None = None,
) -> None:
    """Write Gemini's thinking switch into a ``generationConfig``, when the route has an opinion.

    ``thinkingBudget: 0`` is how Vertex spells "do not think"; OpenRouter spells the same route
    field as ``reasoning: {enabled: false}``. Measured on a 20 s clip: ``audio_chat`` spends a
    mean 895 thought tokens against 88 tokens of transcript -- 91% of billed output -- and
    ``budget: 0`` takes that to 0 with the transcript unchanged (D45).

    Never call this for ``api: transcription``. The dedicated recogniser answers any
    ``thinkingConfig`` at all with ``400 Thinking is not enabled for this model``, and it reports
    zero thought tokens anyway, so there is nothing there to switch off.
    """
    if budget is not None:
        generation_config["thinkingConfig"] = {"thinkingBudget": budget}
        return
    if reasoning_enabled is None:
        return
    generation_config["thinkingConfig"] = {"thinkingBudget": -1 if reasoning_enabled else 0}


def billed_output_tokens(usage: dict[str, Any]) -> int | None:
    """Output tokens as Google bills them: the answer *and* the thoughts behind it.

    ``candidatesTokenCount`` alone is only the visible answer. On a thinking route the thoughts
    are most of the bill -- the fusion pilot spent 754k thought tokens against 75k of answer --
    and an estimate without them understated its cost by a factor of five.
    """
    answer = usage.get("candidatesTokenCount") or usage.get("candidates_token_count")
    thoughts = usage.get("thoughtsTokenCount") or usage.get("thoughts_token_count")
    if answer is None and thoughts is None:
        return None
    return int(answer or 0) + int(thoughts or 0)


def _parts(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Every part of every candidate, in order."""
    out: list[dict[str, Any]] = []
    for candidate in body.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if isinstance(part, dict):
                out.append(part)
    return out


def parse_generate_content(body: dict[str, Any]) -> str:
    """Join the text parts of the first ``generateContent`` candidate that has any."""
    for candidate in body.get("candidates") or []:
        content = candidate.get("content") or {}
        texts = [
            str(part["text"])
            for part in content.get("parts") or []
            if isinstance(part, dict) and part.get("text")
        ]
        if texts:
            return "".join(texts).strip()
    return ""


def _duration_seconds(value: Any) -> float | None:
    """Read a duration (``"1.75s"`` or numeric seconds) as float. Absent/unparseable means None."""
    if value is None:
        return None
    try:
        return float(str(value).rstrip("s"))
    except ValueError:
        return None


def parse_transcription(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
    """Read the transcript and word spans out of an ``audioTranscription`` response.

    Vertex returns one ``Part`` per speaker segment, each carrying its own ``speakerLabel`` and
    its own ``words``. The label is therefore per segment, not per word, so it is fanned down onto
    every word of its segment: ``hypothesis_words.speaker`` is a per-word column, and the
    comparison it exists for is *within* one clip (D36).

    Returns:
        ``(text, words)``. ``words`` is ``None`` when the response carried no word spans, never an
        empty list -- the scorer reads ``None`` as "no signal".
    """
    texts: list[str] = []
    words: list[dict[str, Any]] = []

    for part in _parts(body):
        transcription = part.get("audioTranscription") or part.get("audio_transcription")
        if not isinstance(transcription, dict):
            # A part with no transcription payload still carries the plain text of the segment.
            if part.get("text"):
                texts.append(str(part["text"]))
            continue

        # Prefer the transcription's own text: the sibling ``part["text"]`` repeats it verbatim,
        # and taking both would double every segment.
        segment_text = transcription.get("text") or part.get("text")
        if segment_text:
            texts.append(str(segment_text))

        speaker = transcription.get("speakerLabel") or transcription.get("speaker_label")
        for word in transcription.get("words") or []:
            if not isinstance(word, dict):
                continue
            token = word.get("word") or word.get("text")
            if not token:
                continue
            words.append(
                {
                    "word": str(token),
                    "start": _duration_seconds(word.get("startOffset") or word.get("start_offset")),
                    "end": _duration_seconds(word.get("endOffset") or word.get("end_offset")),
                    "speaker": speaker,
                }
            )

    return " ".join(t.strip() for t in texts if t.strip()).strip(), words or None
