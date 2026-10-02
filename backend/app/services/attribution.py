"""How many points of a WER each recording condition costs (docs/WER-Breakdown.md, D111).

A condition's cost is its attributable errors: the errors on its clips beyond what the same clips
would have made without it, ``errors x (1 - 1/ratio)`` for the ratio of its error rate to a
baseline's. Two ratios are reported, and they answer different questions:

- **within**: the Mantel-Haenszel ratio pooled over the set's own unit (an episode, a speaker),
  so a clip is only compared with baseline clips of its own episode -- same voices, microphone,
  room and topic. This is the estimate of what the condition itself costs.
- **floor**: the crude ratio of the condition's pooled rate to the baseline's over the whole set.
  It also charges the condition for whatever comes with it: on gold, crosstalk lives in podcasts
  and talk shows, which are harder even where one voice speaks.

Pure and importing only the standard library: the dataset's ``harness/`` copy carries it beside
``error_store.py``. Keep it importable on Python 3.10+.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

#: Per group (episode, speaker), ``(errors, reference words)`` in one bucket.
Totals = Mapping[str, tuple[int, int]]


def mh_rate_ratio(
    exposed: Totals, baseline: Totals, groups: Iterable[str]
) -> tuple[float | None, int]:
    """Mantel-Haenszel error-rate ratio of ``exposed`` to ``baseline``, pooled over groups.

    Returns:
        The ratio -- ``None`` when no group holds words in both, or the baseline has no errors
        where it can be compared -- and how many groups could compare. A group repeated in
        ``groups`` (a bootstrap draw) counts each time.
    """
    numerator = denominator = 0.0
    informative = 0
    for group in groups:
        a, t1 = exposed.get(group, (0, 0))
        b, t0 = baseline.get(group, (0, 0))
        if not t1 or not t0:
            continue
        informative += 1
        numerator += a * t0 / (t0 + t1)
        denominator += b * t1 / (t0 + t1)
    return (numerator / denominator if denominator > 0 else None), informative
