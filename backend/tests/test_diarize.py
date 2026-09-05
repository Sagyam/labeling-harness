"""Tests for episode-level speaker diarization (D58).

The clustering is tested against synthetic embeddings rather than audio, because what it has to
get right is a decision rule, not an acoustic one: how many speakers there are, and which turns
belong together. Well-separated clouds are the right fixture for that, and they are deliberately
*easier* than real embeddings: on real Nepali and Hindi speech the within-speaker and
between-speaker distance distributions overlap, and no threshold separates them cleanly. These
tests pin the decision rule; the threshold itself was set by sweeping it against real recordings,
and the numbers are recorded in `app/services/diarize.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.diarize import (
    DEFAULT_THRESHOLD,
    SpeakerTurn,
    assign_turn_speakers,
    cluster_embeddings,
    label_for,
)


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def cloud(centre: np.ndarray, n: int, spread: float, seed: int) -> list[np.ndarray]:
    """``n`` unit vectors scattered around ``centre``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        noisy = centre + rng.normal(0.0, spread, size=centre.shape).astype(np.float32)
        out.append(noisy / np.linalg.norm(noisy))
    return out


# --- clustering -------------------------------------------------------------------------------


def test_two_well_separated_speakers_are_found() -> None:
    a, b = unit(1, 0, 0), unit(0, 1, 0)
    embeddings = cloud(a, 8, 0.10, seed=1) + cloud(b, 8, 0.10, seed=2)
    labels = cluster_embeddings(embeddings, threshold=DEFAULT_THRESHOLD)

    assert len(set(labels)) == 2
    assert len(set(labels[:8])) == 1
    assert len(set(labels[8:])) == 1
    assert labels[0] != labels[8]


def test_one_speaker_stays_one_speaker() -> None:
    """The common case for this corpus: a monologue must not be split into imaginary speakers."""
    embeddings = cloud(unit(1, 0, 0), 12, 0.10, seed=3)
    assert len(set(cluster_embeddings(embeddings, threshold=DEFAULT_THRESHOLD))) == 1


def test_three_speakers_are_found() -> None:
    clouds = [
        cloud(unit(1, 0, 0), 6, 0.08, seed=4),
        cloud(unit(0, 1, 0), 6, 0.08, seed=5),
        cloud(unit(0, 0, 1), 6, 0.08, seed=6),
    ]
    labels = cluster_embeddings([e for c in clouds for e in c], threshold=DEFAULT_THRESHOLD)
    assert len(set(labels)) == 3


def test_a_speaker_cap_merges_rather_than_inventing_speakers() -> None:
    """A cap is a guard against a noisy episode fragmenting into a dozen phantom speakers."""
    centres = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    clouds = [cloud(unit(*e), 4, 0.05, seed=i) for i, e in enumerate(centres)]
    embeddings = [e for c in clouds for e in c]
    labels = cluster_embeddings(embeddings, threshold=DEFAULT_THRESHOLD, max_speakers=2)
    assert len(set(labels)) == 2


def test_labels_are_numbered_by_first_appearance() -> None:
    """spk0 is whoever spoke first, so a label means the same thing across a re-run."""
    a, b = unit(1, 0, 0), unit(0, 1, 0)
    embeddings = cloud(b, 3, 0.05, seed=7) + cloud(a, 3, 0.05, seed=8)
    labels = cluster_embeddings(embeddings, threshold=DEFAULT_THRESHOLD)
    assert labels[0] == 0
    assert labels[3] == 1


def test_an_empty_episode_clusters_to_nothing() -> None:
    assert cluster_embeddings([], threshold=DEFAULT_THRESHOLD) == []


def test_a_single_turn_is_speaker_zero() -> None:
    assert cluster_embeddings([unit(1, 0, 0)], threshold=DEFAULT_THRESHOLD) == [0]


def test_the_threshold_controls_how_readily_speakers_are_split() -> None:
    embeddings = cloud(unit(1, 0, 0), 6, 0.12, seed=9) + cloud(unit(0.9, 0.4, 0), 6, 0.12, seed=10)
    generous = len(set(cluster_embeddings(embeddings, threshold=0.9)))
    strict = len(set(cluster_embeddings(embeddings, threshold=0.2)))
    assert generous <= strict


# --- turn assignment --------------------------------------------------------------------------


def test_a_segment_takes_the_speaker_of_the_turn_it_overlaps_most() -> None:
    turns = [SpeakerTurn(0.0, 10.0, 0), SpeakerTurn(10.0, 20.0, 1)]
    assert assign_turn_speakers([(1.0, 4.0)], turns) == ["spk0"]
    assert assign_turn_speakers([(12.0, 18.0)], turns) == ["spk1"]
    # Straddling the boundary: 3 s against spk0, 7 s against spk1.
    assert assign_turn_speakers([(7.0, 17.0)], turns) == ["spk1"]


def test_a_segment_overlapping_nothing_falls_back_to_the_first_speaker() -> None:
    """Better a defensible default than a null that every consumer has to handle."""
    assert assign_turn_speakers([(50.0, 55.0)], [SpeakerTurn(0.0, 10.0, 0)]) == ["spk0"]


def test_no_turns_at_all_gives_every_segment_the_same_speaker() -> None:
    assert assign_turn_speakers([(0.0, 5.0), (6.0, 9.0)], []) == ["spk0", "spk0"]


@pytest.mark.parametrize(("index", "expected"), [(0, "spk0"), (1, "spk1"), (11, "spk11")])
def test_label_format(index: int, expected: str) -> None:
    assert label_for(index) == expected
