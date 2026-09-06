"""Orthographic normalization of label text (D64).

A corpus assembled one clip at a time drifts. The annotator fixes a spelling on the clip in front
of them and does not fix it on the ninety clips they have already finished, and the result is a
training set that contains both spellings in identical contexts -- which teaches a model that the
choice is free when the whole point of fixing it was that it is not.

This module is the deterministic repair: a reviewed table of whole-token substitutions, applied on
the way *out* of the database rather than on the way in. ``segment_labels.final_text`` keeps
exactly what was typed, so the table can be revised, re-run over the whole corpus, or discarded
without a migration and without losing what a human actually wrote. The ruleset version travels
in the export manifest because a consumer evaluating against this corpus has to apply the same
table to its own references; otherwise pure spelling disagreement is counted as word error.

Substitution is whole-token. ``str.replace`` is the wrong tool -- it fires inside longer words,
turning ``चैँको`` into ``चाहिँको`` -- and so is a token class written as the full Devanagari block,
because ``।`` (U+0964 DANDA) lives inside that block and would be glued onto the preceding word,
silently stopping every rule at the end of a sentence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NORMALIZATION_PATH = REPO_ROOT / "config" / "normalization.yaml"

#: One word. Latin runs (with the apostrophe, so ``don't`` stays whole), or Devanagari runs with
#: the danda and double danda cut out of the range: U+0900-U+0963 is letters, signs and matras,
#: U+0964 and U+0965 are the sentence terminators, and U+0966 onward resumes with the digits.
WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[ऀ-ॣ०-ॿ]+")


class NormalizationError(Exception):
    """The ruleset is not one this module can apply deterministically."""


@dataclass(frozen=True)
class Ruleset:
    """A reviewed table of whole-token substitutions, and the version stamped onto exports.

    Attributes:
        version: Identifier written into the export manifest, e.g. ``norm-v1``.
        tokens: Source token to replacement token. Validated on construction.

    Raises:
        NormalizationError: A rule is not a single token, maps a token to itself, or chains into
            another rule.
    """

    version: str
    tokens: Mapping[str, str]

    def __post_init__(self) -> None:
        for source, target in self.tokens.items():
            if not source or not target:
                raise NormalizationError("a rule may not have an empty source or target")
            if source == target:
                raise NormalizationError(f"rule {source!r} maps a token to itself")
            if WORD_TOKEN_RE.fullmatch(source) is None:
                raise NormalizationError(
                    f"rule source {source!r} is not a single token; sources may not contain"
                    " spaces, punctuation or a danda"
                )
        # A -> B where B -> C would make the output depend on which rule ran first, and would
        # break idempotence. Reject the table rather than pick an order.
        chained = set(self.tokens.values()) & set(self.tokens)
        if chained:
            raise NormalizationError(
                f"rules chain: {sorted(chained)} appear as both a target and a source"
            )


def normalize_text(text: str | None, ruleset: Ruleset) -> str | None:
    """Apply ``ruleset`` to every whole token in ``text``.

    Whitespace, punctuation and casing outside the matched tokens are preserved byte for byte, so
    the result differs from the input only where a rule fired.

    Args:
        text: Label text, or ``None``.
        ruleset: The table to apply.

    Returns:
        The normalized text, or ``None`` if ``text`` was ``None``.
    """
    if not text:
        return text
    if not ruleset.tokens:
        return text
    return WORD_TOKEN_RE.sub(
        lambda m: ruleset.tokens.get(m.group(0), m.group(0)),
        text,
    )


def load_ruleset(path: Path | str | None = None) -> Ruleset:
    """Load the normalization table from ``config/normalization.yaml``.

    Args:
        path: Ruleset file. Defaults to the copy committed at the repository root.

    Raises:
        FileNotFoundError: The file does not exist. A missing table is an error rather than an
            empty ruleset, because silently exporting un-normalized text looks identical to
            exporting normalized text until someone trains on it.
        NormalizationError: The file is malformed or a rule is not applicable.
    """
    resolved = Path(path) if path else DEFAULT_NORMALIZATION_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"normalization ruleset not found: {resolved}")
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise NormalizationError(f"{resolved} must contain a YAML mapping at the top level")
    version = data.get("version")
    if not isinstance(version, str) or not version:
        raise NormalizationError(f"{resolved} must set a non-empty string `version`")
    tokens = data.get("tokens") or {}
    if not isinstance(tokens, dict):
        raise NormalizationError(f"{resolved} `tokens` must be a mapping")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in tokens.items()):
        raise NormalizationError(f"{resolved} `tokens` must map strings to strings")
    return Ruleset(version=version, tokens=dict(tokens))
