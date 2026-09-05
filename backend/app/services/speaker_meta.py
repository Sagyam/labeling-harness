"""What the corpus is allowed to record about a speaker.

Episode metadata arrives from two places -- the ingest form in this repository, and an upstream
``episode.json`` written by a pipeline that is not in this repository -- and ``episode.schema.json``
deliberately keeps unknown properties so a new upstream field is never silently lost. That is the
right default for provenance and the wrong default for people: it means any key the upstream
pipeline invents lands in ``episodes.metadata_jsonb`` and then in every export.

So the speaker block is an **allowlist**, applied at the importer, which is the one choke point
both paths pass through. A field that is not on the list is dropped, whether it is a name, a
hometown, or something nobody has thought of yet (D56).

Two fields are gone on purpose:

* ``name`` -- identity, and being a public figure does not make it less so. A voice clip plus a
  name is the join key that turns a research corpus into a dossier.
* ``origin``/``dialect`` -- unreliable at the source. The owner cannot label Nepali dialect
  consistently, and a confidently wrong dialect tag is worse than no tag: it would be used as a
  stratification variable and would silently bias every result computed over it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: The only per-speaker facts the corpus stores. ``role`` is about the recording, not the person;
#: ``gender`` and ``age_bracket`` are coarse variables a code-switching study genuinely stratifies
#: on, and both are typed by a human because neither can be inferred from the audio (D58).
ALLOWED_SPEAKER_FIELDS = frozenset({"role", "gender", "age_bracket"})

#: Closed vocabularies, for the same reason the topic taxonomy is closed (D57): a stratification
#: variable spelled three different ways is three variables. A value outside these is dropped
#: exactly like an unknown field -- the form only ever sends these, so anything else came from an
#: upstream manifest that means something this corpus does not record.
ALLOWED_VALUES: dict[str, frozenset[str]] = {
    "gender": frozenset({"male", "female", "non_binary", "other"}),
    #: Twenty-year buckets. Deliberately coarse: the owner is guessing from having watched the
    #: episode, and a bucket someone can place a stranger in confidently is worth more than a
    #: finer one they cannot (D58).
    "age_bracket": frozenset({"under_20", "20_39", "40_59", "60_79", "80_plus"}),
}


def _keeps(field: str, value: Any) -> bool:
    """Whether one speaker field survives: on the allowlist, and a value the corpus knows."""
    if field not in ALLOWED_SPEAKER_FIELDS:
        return False
    vocabulary = ALLOWED_VALUES.get(field)
    return vocabulary is None or value in vocabulary


def strip_speaker_pii(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return ``metadata`` with its ``speakers`` block reduced to :data:`ALLOWED_SPEAKER_FIELDS`.

    Values are checked as well as field names: a field in :data:`ALLOWED_VALUES` keeps only the
    values listed there. ``role`` is free text, because it describes the recording rather than
    grouping the corpus.

    Never raises. A malformed ``speakers`` value -- a string, a list, a speaker that is not an
    object -- is dropped rather than rejected: a bad metadata field must not fail an ingest that
    otherwise produced good audio, but it must not be stored either.

    Args:
        metadata: Episode metadata, as it arrived. Not mutated.

    Returns:
        A new dict. ``speakers`` is omitted entirely when nothing survived the allowlist.
    """
    if not metadata:
        return {}

    cleaned = {key: value for key, value in metadata.items() if key != "speakers"}

    speakers = metadata.get("speakers")
    if isinstance(speakers, Mapping):
        kept = {
            str(speaker_id): allowed
            for speaker_id, fields in speakers.items()
            if isinstance(fields, Mapping)
            and (allowed := {k: v for k, v in fields.items() if _keeps(k, v)})
        }
        if kept:
            cleaned["speakers"] = kept

    return cleaned
