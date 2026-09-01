"""Small statistical helpers shared by the progress counters and the status report."""

from __future__ import annotations

from collections.abc import Sequence


def median(values: Sequence[float]) -> float | None:
    """Median of ``values``, or ``None`` when there are none.

    An even-length sequence averages the two middle values, so the result is the usual median
    rather than the upper of the pair.
    """
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
