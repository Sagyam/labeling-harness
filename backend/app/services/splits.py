"""Frozen train/val/test split assignment.

The split for an episode is a pure function of ``(external_id, seed)``, computed once at import and
written to ``episodes.split``. It is never recomputed: if it were, adding episodes would move
existing ones across the train/test boundary, which means a model could be trained on audio it was
already benchmarked against, and two exports of "the same" dataset would not be the same dataset.

Splits are per *episode*, never per segment -- segments from one episode share speaker, recording
conditions and topic, so a segment-level split leaks.
"""

from __future__ import annotations

import hashlib

#: Order matters: buckets are laid out along [0, 1) in this order, so a given hash keeps its
#: split as long as the ratios do not change.
SPLIT_ORDER = ("train", "val", "test")

_MAX = float(1 << 64)


def split_hash_unit(external_id: str, *, seed: int) -> float:
    """Map ``(external_id, seed)`` to a stable float in [0, 1).

    BLAKE2b is used rather than :func:`hash` because the built-in is salted per process and would
    produce a different split on every run.
    """
    digest = hashlib.blake2b(f"{external_id}:{seed}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / _MAX


def assign_split(external_id: str, *, seed: int, ratios: dict[str, float]) -> str:
    """Return ``train``, ``val`` or ``test`` for an episode.

    Args:
        external_id: The episode's manifest id.
        seed: Corpus-wide split seed, stored alongside the assignment.
        ratios: Fractions per split; must name exactly train, val and test and sum to 1.

    Raises:
        ValueError: The ratios are missing a split or do not sum to 1.
    """
    if set(ratios) != set(SPLIT_ORDER):
        raise ValueError(f"ratios must name exactly {SPLIT_ORDER}, got {sorted(ratios)}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    position = split_hash_unit(external_id, seed=seed)
    cumulative = 0.0
    for name in SPLIT_ORDER:
        cumulative += ratios[name]
        if position < cumulative:
            return name
    return SPLIT_ORDER[-1]  # pragma: no cover - only reachable through float rounding
