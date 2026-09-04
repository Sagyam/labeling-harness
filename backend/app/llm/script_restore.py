"""Restore the orthography of a transcript the recogniser wrote in one script.

Gemini 3.5 Transcribe is the only configured transcriber that reports word spans *and* speaker
labels, and it accepts no steering whatsoever -- prose is refused, a text part is ignored, and
custom vocabulary silently costs the speaker labels (D39). So it writes every word in Devanagari,
English included: ``active`` comes back as ``एक्टिभ``.

This module is the second half of the ``gemini-composite`` system (D41). It takes the token list
the recogniser produced and asks a reasoning model to put each token back into the script its own
language uses, then hands the spans through unchanged. The recogniser decides *what was said and
when*; this step decides only *how it is spelled*.

The whole design rests on one constraint: **one output token per input token, in order**. That is
what lets each restored word inherit its own acoustic span exactly, with no re-alignment and no
forced aligner. A rewrite that returns a different number of tokens is rejected rather than
patched up -- a silently misaligned transcript would corrupt every span in the segment, which is
far worse than a segment that fails loudly.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.config import LlmRoutes
from app.llm.base import LlmRequestFailed
from app.llm.openrouter import OpenRouterClient
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Devanagari, the block the recogniser writes everything in.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

#: How many times to ask again when the model returns the wrong number of tokens.
MAX_ALIGNMENT_ATTEMPTS = 3

INSTRUCTION = (
    "You are restoring the ORTHOGRAPHY of an already-correct Nepali-English code-switched "
    "transcript. A speech recogniser wrote every word in Devanagari, including the English words, "
    "which it spelled phonetically.\n\n"
    "For each numbered token, output the SAME WORD in the script its own language uses:\n"
    "- A genuinely Nepali word stays in Devanagari, unchanged, character for character.\n"
    "- An English word spelled phonetically in Devanagari becomes its ordinary English spelling "
    "in Latin. For example एक्टिभ -> active, रेन्ज -> range, टनेल -> tunnel.\n\n"
    "RULES:\n"
    "1. Output EXACTLY one entry per input token, in the same order. Never merge, split, drop or "
    "add a token, even when the natural English rendering would use a different number of words.\n"
    "2. Do NOT correct, translate, punctuate or improve anything. Change the script and nothing "
    "else. Keep any punctuation already attached to a token.\n"
    "3. If you are not confident a token is English, leave it in Devanagari unchanged.\n\n"
    "Return ONLY a JSON array of strings, with no commentary and no code fence."
)


def _is_devanagari(token: str) -> bool:
    """True when a token carries Devanagari and no Latin letters."""
    return bool(_DEVANAGARI.search(token)) and not re.search(r"[A-Za-z]", token)


def _extract_array(content: str) -> list[str]:
    """Pull the JSON array out of a completion, tolerating a code fence or stray prose."""
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array in the response")
    parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("response was not a JSON array of strings")
    return parsed


def count_same_script_edits(tokens: list[str], restored: list[str]) -> int:
    """Tokens the model rewrote *without* changing script -- i.e. it edited the words.

    Rule 2 forbids this: a Devanagari token that comes back as different Devanagari is the model
    correcting the recogniser rather than transliterating it, which silently replaces measured ASR
    output with a reasoning model's opinion of it. Counted and reported rather than raised, so one
    disputed token cannot fail an episode, but never ignored -- the count rides along in the
    hypothesis metadata so a suspect segment can be found again.
    """
    return sum(
        1
        for original, new in zip(tokens, restored, strict=True)
        if original != new and _is_devanagari(original) and _is_devanagari(new)
    )


def restore_script(
    session: Session,
    tokens: list[str],
    *,
    route: str,
    config: LlmRoutes,
    client: httpx.Client | None = None,
    dry_run: bool | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Rewrite each token into its own language's script, preserving the token count.

    Args:
        session: Session the ``llm_requests`` row is written to.
        tokens: The recogniser's words, in order.
        route: Name of the text route doing the rewrite, from ``config/llm_routes.yaml``.
        config: Routing table.
        client: Optional shared HTTPX client.
        dry_run: Override the configured dry-run mode.

    Returns:
        ``(restored, metadata)``. ``restored`` has exactly the length of ``tokens``. ``metadata``
        records what the rewrite did, for the hypothesis's ``metadata_jsonb``.

    Raises:
        LlmRequestFailed: The rewrite never returned a correctly aligned token list.
    """
    if not tokens:
        return [], {"script_restore_tokens": 0}

    payload = json.dumps(tokens, ensure_ascii=False)
    prompt = f"{INSTRUCTION}\n\nThere are exactly {len(tokens)} tokens.\n\nTOKENS:\n{payload}"
    llm = OpenRouterClient(session, config=config, client=client)

    last_error = "no attempt was made"
    for attempt in range(MAX_ALIGNMENT_ATTEMPTS):
        result = llm.complete(
            route, [{"role": "user", "content": prompt}], dry_run=dry_run, temperature=0.0
        )
        if result.dry_run:
            return list(tokens), {"script_restore_tokens": len(tokens), "script_restore": "dry_run"}
        try:
            restored = _extract_array(result.text)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = f"unparseable rewrite: {exc}"
            logger.warning("script_restore_unparseable", route=route, attempt=attempt + 1)
            continue

        if len(restored) != len(tokens):
            # Never patch this up by padding or truncating: the spans are positional, so a
            # length mismatch means every word after the first divergence would carry the wrong
            # timing. Ask again, then give up.
            last_error = f"token count {len(restored)} != {len(tokens)}"
            logger.warning(
                "script_restore_misaligned",
                route=route,
                attempt=attempt + 1,
                expected=len(tokens),
                got=len(restored),
            )
            continue

        edits = count_same_script_edits(tokens, restored)
        if edits:
            logger.warning("script_restore_same_script_edits", route=route, tokens=edits)
        changed = sum(1 for a, b in zip(tokens, restored, strict=True) if a != b)
        return restored, {
            "script_restore_tokens": len(tokens),
            "script_restore_changed": changed,
            "script_restore_same_script_edits": edits,
            "script_restore_model": result.model,
        }

    raise LlmRequestFailed(f"script restoration failed: {last_error}")
