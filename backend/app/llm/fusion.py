"""Transcript fusion: a reasoning model reconciles three ASR hypotheses into the seed (D72).

Each recogniser hears one clip and nothing else, so each gets wrong what only context could fix: a
name said clearly a minute earlier, an English term the speaker keeps returning to, the end of a
sentence that started in the previous clip. Fusion hands a reasoning model all three hypotheses
for a long run of consecutive clips -- plus a little of what comes next and its own output for what
came before -- and asks for one verbatim transcript per clip. On the gold pot's 896 labels the
pilot scored 0.192 WER against 0.262 for the best single recogniser, each measured only where it
had not seeded the label.

It never hears the audio. That is what the hazard gates in ``hazards.py`` and the forced aligner
are for: this module produces text, and the waveform adjudicates it.

The unit of work is a **window**: about thirty minutes of target clips, balanced so an episode is
cut into equal windows rather than full ones and a stub. Most sources are 5-20 minute videos and
are one window, with no seams at all. Around the targets a window carries

* **after** -- the raw hypotheses of the clips that follow, so a sentence or a name can be seen
  finishing before it is committed to; and
* **carry-over** -- the model's *own* fused text for the clips just before, so a long episode
  keeps the spellings and names it has already settled.

The response contract is one JSON object per target id, no more, no fewer. A broken contract --
a missing or invented id, unparseable output, an answer truncated by its own thinking -- is never
patched: the window is retried once, then split in half and each half tried on its own, down to a
bounded depth. What still fails is reported as unfused and falls back to a plain ASR seed; nothing
is guessed. This module is pure over a batch: the model is a callable, so the loop is testable
without a network and the same code serves ingestion and a re-fusion of an imported episode.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.llm.base import LlmRequestFailed, LlmResult

#: Bumped whenever the instruction or the rendering changes what the model is asked. It is part of
#: the fused system's id, so a changed prompt is a new system rather than an overwrite (D72).
PROMPT_VERSION = "p1"

#: Hypotheses are shown under these labels, in the configured route order. Anonymous on purpose:
#: the fuser is a Gemini model and one of the three transcripts is Gemini's (new-plan §3.5).
SYSTEM_LABELS = ("A", "B", "C", "D", "E")

#: What the model may say it did with a segment. Anything else is recorded as ``?``.
CODES = frozenset({"k", "s", "m", "c", "u"})

#: A quota refusal is waited out rather than counted as a failed attempt. Minutes, not the
#: client's half-second retry: a 429 on a thinking model is a rate limit, not a blip.
QUOTA_WAITS_SECONDS = (30.0, 120.0, 480.0)

SYSTEM_INSTRUCTION = """\
You reconcile automatic transcripts of Nepali-language audio in which speakers code-switch \
freely into English. Several independent speech recognisers each transcribed the same short \
segment of audio. You see all of them. You do not hear the audio.

Your output is a VERBATIM transcript. It is training data for a speech model, so it must record \
what was actually said, not what would have been better said.

Rules, in priority order:

1. VERBATIM. Never correct the speaker. Keep false starts, repetitions, self-interruptions, \
fillers (अँ, उम्, हैन, like, you know) and ungrammatical constructions exactly as spoken. If a \
speaker states a wrong name, date, number or fact, transcribe the wrong one. Do not regularise \
dialect toward standard Nepali. Fluency is not the goal; accuracy is.
2. SCRIPT. Nepali words in Devanagari, English words in Latin script. An English word a recogniser \
spelled phonetically in Devanagari (क्याप्टन, टिम, टाइप) is written in Latin (Captain, team, type). \
A Nepali word one recogniser romanised is written back in Devanagari.
3. RECONCILE. Where the recognisers disagree, choose the reading the audio most plausibly \
supports, using the surrounding conversation for context. Agreement between two is evidence but \
not proof. Where all are garbled, prefer the least invented reading over a fluent guess.
4. SEGMENTS ARE FIXED. Emit exactly one entry per target segment, with its id. Never merge, split, \
reorder or drop a segment, and never move words between segments: a segment's recognisers are \
evidence for that segment only, even when a sentence runs on into the next one. If a segment \
genuinely contains no speech, emit an empty string -- do not invent filler for it.
5. NO ADDITIONS. No translations, speaker names, timestamps, commentary or rewording for \
punctuation. Only the words spoken.

Each target segment also gets a one-letter code saying what you did:
  k = kept one recogniser's reading essentially as-is
  s = same words, script or spelling normalised per rule 2
  m = merged readings from more than one recogniser
  c = corrected from surrounding context (a word no recogniser got right)
  u = uncertain -- the recognisers conflict and context does not resolve it

Answer with JSON only: an array of objects {"id": <int>, "t": "<transcript>", "c": "<code>"}. \
No markdown fence, no prose before or after."""


@dataclass(frozen=True)
class FusionSegment:
    """One clip as the fuser sees it: where it sits and what each recogniser wrote."""

    key: str
    start: float
    end: float
    #: One text per ASR system, in configured route order.
    hypotheses: tuple[str, ...]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def words(self) -> int:
        """Total words sent for this segment across all its hypotheses."""
        return sum(len((h or "").split()) for h in self.hypotheses)


@dataclass
class Window:
    """One fusion request: segments to transcribe, and the raw segments that follow them."""

    index: int
    targets: list[FusionSegment]
    after: list[FusionSegment]
    #: Segment key to the integer id the model sees -- its position in the whole episode.
    ids: dict[str, int] = field(default_factory=dict)
    #: 0 for a planned window; each bisection adds one.
    depth: int = 0

    def split(self) -> tuple[Window, Window]:
        """Two halves of one window, each looking ahead at what follows it.

        The first half looks into the second (at least a few segments of it, so a sentence that
        crosses the cut is still visible); the second keeps the window's own lookahead.
        """
        middle = max(1, len(self.targets) // 2)
        left, right = self.targets[:middle], self.targets[middle:]
        reach = max(len(self.after), 5)
        return (
            Window(self.index, left, (right + self.after)[:reach], self.ids, self.depth + 1),
            Window(self.index, right, self.after, self.ids, self.depth + 1),
        )


@dataclass(frozen=True)
class ParsedWindow:
    texts: dict[int, str]
    codes: dict[int, str]
    problems: list[str]


@dataclass(frozen=True)
class FusedSegment:
    key: str
    text: str
    code: str
    window_index: int
    window_depth: int


@dataclass
class FusionCall:
    """One request to the model, and what came back. Kept so a run's cost and failures are
    visible per window, beside the ``llm_requests`` row the client wrote."""

    window_index: int
    depth: int
    targets: int
    finish_reason: str | None = None
    model_version: str | None = None
    prompt_tokens: int | None = None
    thought_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float = 0.0
    problems: list[str] = field(default_factory=list)


@dataclass
class FusionOutcome:
    fused: dict[str, FusedSegment] = field(default_factory=dict)
    unfused: list[str] = field(default_factory=list)
    calls: list[FusionCall] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.calls), 6)


def plan_windows(
    segments: Sequence[FusionSegment],
    *,
    target_words: int = 7000,
    lookahead_seconds: float = 120.0,
    max_segments: int = 250,
) -> list[Window]:
    """Cut an episode into balanced windows; every segment is a target exactly once.

    The number of windows is set by total words being sent (``ceil(total / target)``) and by
    ``max_segments``, then segments are dealt out so each window holds about the same number
    of words.
    """
    if not segments:
        return []
    ids = {s.key: position for position, s in enumerate(segments)}
    total_words = sum(s.words for s in segments)
    count = max(
        1,
        math.ceil(total_words / target_words) if target_words > 0 else 1,
        math.ceil(len(segments) / max_segments),
    )
    share = total_words / count

    groups: list[list[FusionSegment]] = [[]]
    elapsed = 0
    for segment in segments:
        room = len(groups) < count and groups[-1]
        if share > 0:
            boundary = share * len(groups)
            threshold_crossed = elapsed + segment.words / 2 > boundary
        else:
            seg_boundary = (len(segments) / count) * len(groups)
            threshold_crossed = sum(len(g) for g in groups) >= seg_boundary
        if room and (threshold_crossed or len(groups[-1]) >= max_segments):
            groups.append([])
        groups[-1].append(segment)
        elapsed += segment.words

    windows: list[Window] = []
    position = 0
    for index, group in enumerate(groups):
        position += len(group)
        after: list[FusionSegment] = []
        ahead = 0.0
        for segment in segments[position:]:
            if ahead >= lookahead_seconds:
                break
            after.append(segment)
            ahead += segment.duration
        windows.append(Window(index=index, targets=group, after=after, ids=ids))
    return windows


def _render_segment(segment: FusionSegment, head: str) -> str:
    lines = [f"{head} {segment.start:.1f}-{segment.end:.1f}s"]
    for label, text in zip(SYSTEM_LABELS, segment.hypotheses, strict=False):
        lines.append(f"  {label}: {(text or '').strip()}")
    return "\n".join(lines)


def render_window(window: Window, *, carryover: Sequence[tuple[FusionSegment, str]]) -> str:
    """The user message: settled text behind, segments to fuse, raw hypotheses ahead."""
    parts: list[str] = []
    if carryover:
        parts.append(
            "### ALREADY TRANSCRIBED (context only -- do not re-emit)\n"
            "The segments immediately before, as you already reconciled them."
        )
        for segment, text in carryover:
            parts.append(f"[-] {segment.start:.1f}-{segment.end:.1f}s\n  {text}")

    parts.append(
        "### TRANSCRIBE THESE\n"
        f"{len(window.targets)} segments. Emit exactly one JSON object for each id below, in order."
    )
    for segment in window.targets:
        parts.append(_render_segment(segment, f"[{window.ids[segment.key]}]"))

    if window.after:
        parts.append(
            "### WHAT COMES NEXT (context only -- do not emit)\n"
            "Raw hypotheses for the segments that follow, so you can see where a sentence, a "
            "name or an argument is going before you commit to it."
        )
        for segment in window.after:
            parts.append(_render_segment(segment, "[-]"))

    parts.append(
        f"Now emit the JSON array for the {len(window.targets)} ids in ### TRANSCRIBE THESE, "
        "and nothing else."
    )
    return "\n\n".join(parts)


def parse_response(text: str, expected_ids: Sequence[int]) -> ParsedWindow:
    """Read the model's array. Every way the contract can break is a problem, never a guess."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.rsplit("```", 1)[0]
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end <= start:
        return ParsedWindow({}, {}, ["no JSON array in the response"])
    try:
        items = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        return ParsedWindow({}, {}, [f"no JSON array in the response ({exc.msg})"])
    if not isinstance(items, list):
        return ParsedWindow({}, {}, ["no JSON array in the response"])

    texts: dict[int, str] = {}
    codes: dict[int, str] = {}
    problems: list[str] = []
    duplicates: set[int] = set()
    for item in items:
        if not isinstance(item, dict) or "id" not in item:
            problems.append(f"malformed item {str(item)[:60]!r}")
            continue
        try:
            sid = int(item["id"])
        except (TypeError, ValueError):
            problems.append(f"non-integer id {item['id']!r}")
            continue
        if sid in texts:
            duplicates.add(sid)
        texts[sid] = str(item.get("t") or "").strip()
        code = str(item.get("c") or "").strip().lower()
        codes[sid] = code if code in CODES else "?"

    expected = set(expected_ids)
    if duplicates:
        problems.append(f"duplicate ids {sorted(duplicates)[:10]}")
    if missing := expected - set(texts):
        problems.append(f"missing {len(missing)} ids {sorted(missing)[:10]}")
    if extra := set(texts) - expected:
        problems.append(f"unexpected {len(extra)} ids {sorted(extra)[:10]}")
    return ParsedWindow(texts, codes, problems)


def _call_record(window: Window, result: LlmResult | None) -> FusionCall:
    call = FusionCall(window_index=window.index, depth=window.depth, targets=len(window.targets))
    if result is None:
        return call
    raw: dict[str, Any] = result.raw or {}
    candidates = raw.get("candidates") or [{}]
    usage = raw.get("usageMetadata") or {}
    call.finish_reason = (candidates[0] or {}).get("finishReason")
    call.model_version = raw.get("modelVersion") or result.model
    call.prompt_tokens = usage.get("promptTokenCount", result.prompt_tokens)
    call.thought_tokens = usage.get("thoughtsTokenCount")
    call.output_tokens = usage.get("candidatesTokenCount")
    call.cost_usd = float(result.estimated_cost_usd or 0)
    return call


def fuse(
    segments: Sequence[FusionSegment],
    *,
    complete: Callable[[list[dict[str, Any]]], LlmResult],
    target_words: int = 7000,
    lookahead_seconds: float = 120.0,
    carryover_seconds: float = 300.0,
    max_segments: int = 250,
    max_depth: int = 2,
    should_stop: Callable[[], bool] = lambda: False,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = lambda _message: None,
) -> FusionOutcome:
    """Fuse an episode window by window.

    Args:
        segments: The episode's clips, in time order, each with every ASR system's text.
        complete: Sends one chat request and returns the result -- in production a routed,
            logged ``VertexClient.complete`` bound to the fusion route.
        target_words: Words being sent per window across recognisers.
        lookahead_seconds: Raw hypotheses shown after a window's targets.
        carryover_seconds: The model's own earlier output shown before a window's targets.
        max_segments: Hard cap on targets per window, whatever their duration.
        max_depth: How many times a failing window may be halved.
        should_stop: Checked before every request; ``True`` ends the run where it stands.
        sleep: Injected so tests do not wait out a quota.
        log: One line per request, for the ingest log.

    Returns:
        What was fused, what was not, and one record per request.
    """
    outcome = FusionOutcome()
    order = list(segments)
    position = {s.key: i for i, s in enumerate(order)}

    def carryover_for(window: Window) -> list[tuple[FusionSegment, str]]:
        first = position[window.targets[0].key]
        behind: list[tuple[FusionSegment, str]] = []
        elapsed = 0.0
        for segment in reversed(order[:first]):
            fused = outcome.fused.get(segment.key)
            if fused is None or elapsed >= carryover_seconds:
                break
            behind.append((segment, fused.text))
            elapsed += segment.duration
        return list(reversed(behind))

    def request(window: Window) -> tuple[ParsedWindow | None, FusionCall, bool]:
        """One attempt. Returns (parsed, record, truncated)."""
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": render_window(window, carryover=carryover_for(window))},
        ]
        waits = iter(QUOTA_WAITS_SECONDS)
        while True:
            try:
                result = complete(messages)
                break
            except LlmRequestFailed as exc:
                wait = next(waits, None) if "429" in str(exc) else None
                if wait is None:
                    call = _call_record(window, None)
                    call.problems.append(str(exc)[:300])
                    return None, call, False
                log(f"fusion window {window.index}: quota refusal, waiting {wait:.0f}s")
                sleep(wait)
        call = _call_record(window, result)
        truncated = (call.finish_reason or "STOP") not in ("STOP", "FINISH_REASON_UNSPECIFIED")
        parsed = parse_response(result.text, [window.ids[s.key] for s in window.targets])
        call.problems.extend(parsed.problems)
        if truncated:
            call.problems.append(f"finishReason={call.finish_reason}")
        return parsed, call, truncated

    def run(window: Window) -> None:
        attempts = 0
        while True:
            if should_stop():
                outcome.unfused.extend(s.key for s in window.targets)
                return
            parsed, call, truncated = request(window)
            attempts += 1
            outcome.calls.append(call)
            log(
                f"fusion window {window.index}"
                + (f".{window.depth}" if window.depth else "")
                + f": {len(window.targets)} segments, {call.thought_tokens or 0} thought / "
                f"{call.output_tokens or 0} out"
                + (f" -- {'; '.join(call.problems)}" if call.problems else "")
            )
            if parsed is not None and not call.problems:
                for segment in window.targets:
                    sid = window.ids[segment.key]
                    outcome.fused[segment.key] = FusedSegment(
                        key=segment.key,
                        text=parsed.texts[sid],
                        code=parsed.codes[sid],
                        window_index=window.index,
                        window_depth=window.depth,
                    )
                return
            # A truncated answer is too big for its budget; asking again at the same size buys
            # nothing. Anything else gets one more try before the window is halved.
            if not truncated and attempts < 2:
                continue
            if window.depth >= max_depth or len(window.targets) < 2:
                outcome.unfused.extend(s.key for s in window.targets)
                return
            left, right = window.split()
            run(left)
            run(right)
            return

    windows = plan_windows(
        order,
        target_words=target_words,
        lookahead_seconds=lookahead_seconds,
        max_segments=max_segments,
    )
    for window in windows:
        run(window)
    return outcome
