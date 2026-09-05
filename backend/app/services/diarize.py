"""Work out who is speaking, once per episode, before the audio is cut into clips (D58).

Until now every web-ingested segment was written with ``speaker_id = "spk0"`` regardless of who
said it. That is what D52 was describing when it said speaker identity is a full-episode problem
that belongs to a stage running *before* segmentation: a clip almost always holds one speaker, so
a clip-local label answers nothing, and ``spk:0`` in one clip has no relation to ``spk:0`` in the
next. This module is that stage.

It reuses the VAD turns the pipeline already computed -- so there is no second speech detection
pass -- embeds each one with a speaker-verification model, and clusters the embeddings across the
whole episode. Segments then inherit the speaker of the turn they overlap most.

**Why agglomerative clustering and not k-means.** The number of speakers is the thing being
discovered; a podcast episode is two people, or three, or one person reading. Average-linkage
agglomerative clustering with a cosine-distance stopping threshold decides the count from the
data, and the threshold is the one parameter.

**The threshold is measured, not guessed, and the measurement is not flattering.** On real Nepali
and Hindi speech, cosine distance between two turns of the *same* speaker averages 0.28 at 4 s
turns (0.44 at 2 s), and between *different* speakers 0.66 (0.75 at 2 s). Those two distributions
overlap at every turn length tried: at 4 s the within-speaker 90th percentile is 0.43 and the
between-speaker 10th percentile is 0.31. **No threshold separates them cleanly**, so this stage is
a good guess and must not be read as ground truth. Sweeping the threshold against 40 synthesised
two-speaker episodes put the optimum at 0.62-0.65, worth 83-90% correct partitions with only three
turns per speaker; a real episode gives the clustering fifty turns per speaker to work with, which
conditions it considerably better than that floor.

**It degrades rather than fails.** No model, too little audio, or an embedding that will not
compute leaves every segment on ``spk0`` -- exactly what the pipeline did before this existed.
A wrong speaker label is worse than a missing one, because it is a grouping variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.services.fbank import compute_fbank
from app.utils.logging import get_logger

if TYPE_CHECKING:
    from app.services.silero_vad import SpeechTurn as VadTurn

logger = get_logger(__name__)

#: Cosine *distance* above which two clusters are different speakers. Swept against 40 synthesised
#: two-speaker episodes built from real Nepali and Hindi recordings: 0.45 scored 70%, 0.55 scored
#: 83%, 0.65 scored 90% and 0.75 fell back to 78%. The optimum is a plateau rather than a point,
#: and 0.62 sits on it while staying below the value at which two real speakers start merging.
DEFAULT_THRESHOLD = 0.62

#: Never return more speakers than this. A guard against a noisy episode fragmenting into phantom
#: speakers, not a claim about podcasts: the cost of merging two real speakers is a mixed label,
#: the cost of inventing eight is a grouping variable that means nothing at all.
DEFAULT_MAX_SPEAKERS = 6

#: Turns shorter than this are not embedded. Measured: within-speaker distance rises from 0.28 at
#: 4 s to 0.44 at 2 s, so a short turn contributes more noise than evidence. The segmenter already
#: refuses to cut anything below 2 s, so this drops backchannels rather than real segments.
MIN_TURN_SECONDS = 2.0

#: Frames needed before the embedding graph is worth running, at 10 ms per frame.
MIN_FRAMES = 30


@dataclass(frozen=True)
class SpeakerTurn:
    """One speech turn with the speaker cluster it was assigned to."""

    start: float
    end: float
    speaker: int


def label_for(index: int) -> str:
    """The stored form of a speaker index."""
    return f"spk{index}"


def cluster_embeddings(
    embeddings: list[np.ndarray],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
) -> list[int]:
    """Group unit-norm embeddings into speakers with average-linkage agglomerative clustering.

    Args:
        embeddings: One unit-norm vector per turn, in episode order.
        threshold: Cosine distance at which merging stops. Higher merges more readily.
        max_speakers: Merging continues past ``threshold`` while more clusters than this remain.

    Returns:
        One cluster index per embedding, numbered by first appearance so ``0`` is always whoever
        spoke first and a re-run of the same episode produces the same labels.
    """
    if not embeddings:
        return []
    if len(embeddings) == 1:
        return [0]

    matrix = np.vstack(embeddings).astype(np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    distance = 1.0 - matrix @ matrix.T
    np.fill_diagonal(distance, np.inf)

    # Each cluster is the list of member indices; average linkage is the mean pairwise distance
    # between two clusters' members, which is cheap to recompute at this size (turns, not frames).
    clusters: list[list[int]] = [[i] for i in range(len(embeddings))]
    while len(clusters) > 1:
        best = (np.inf, -1, -1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = float(distance[np.ix_(clusters[i], clusters[j])].mean())
                if d < best[0]:
                    best = (d, i, j)

        gap, i, j = best
        if gap > threshold and len(clusters) <= max_speakers:
            break
        clusters[i].extend(clusters[j])
        clusters.pop(j)

    labels = [0] * len(embeddings)
    for order, members in enumerate(sorted(clusters, key=min)):
        for member in members:
            labels[member] = order
    return labels


def assign_turn_speakers(
    segments: list[tuple[float, float]], turns: list[SpeakerTurn]
) -> list[str]:
    """Give each segment the speaker of the turn it shares the most time with.

    Args:
        segments: ``(start, end)`` in episode-relative seconds, in any order.
        turns: Diarized turns. May be empty.

    Returns:
        One speaker label per segment, in the order given. Everything falls back to ``spk0``,
        which is what the pipeline used unconditionally before diarization existed.
    """
    if not turns:
        return [label_for(0)] * len(segments)

    labels = []
    for start, end in segments:
        overlaps: dict[int, float] = {}
        for turn in turns:
            shared = min(end, turn.end) - max(start, turn.start)
            if shared > 0:
                overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + shared
        best = max(overlaps, key=lambda spk: overlaps[spk]) if overlaps else 0
        labels.append(label_for(best))
    return labels


def _embed_turn(session: Any, samples: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """One unit-norm embedding for one turn's audio, or None when there is too little of it."""
    features = compute_fbank(samples, sample_rate=sample_rate)
    if features.shape[0] < MIN_FRAMES:
        return None
    embedding = session.run(None, {"feats": features[None].astype(np.float32)})[0][0]
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        return None
    return embedding / norm


def diarize_turns(
    audio_path: Path,
    turns: list[VadTurn],
    *,
    model_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
) -> list[SpeakerTurn]:
    """Label every VAD turn in one episode with a speaker.

    Args:
        audio_path: The episode's normalised 16 kHz mono audio.
        turns: Speech turns from the VAD that already ran.
        model_path: The speaker-embedding ONNX graph.
        threshold: Cosine distance at which clustering stops merging.
        max_speakers: Upper bound on speakers found.

    Returns:
        One :class:`SpeakerTurn` per input turn that was long enough to embed. Empty when the
        model is unavailable or nothing could be embedded, which the caller reads as "one
        speaker" rather than as an error.
    """
    import onnxruntime as ort
    import soundfile as sf

    if not model_path.is_file():
        logger.warning("diarize_no_model", path=str(model_path))
        return []

    long_enough = [t for t in turns if t.end - t.start >= MIN_TURN_SECONDS]
    if not long_enough:
        return []

    try:
        audio, sample_rate = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        logger.warning("diarize_setup_failed", error=str(exc))
        return []

    embeddings: list[np.ndarray] = []
    kept: list[VadTurn] = []
    for turn in long_enough:
        window = audio[int(turn.start * sample_rate) : int(turn.end * sample_rate)]
        try:
            embedding = _embed_turn(session, window, sample_rate)
        except Exception as exc:
            logger.warning("diarize_embed_failed", start=turn.start, error=str(exc))
            continue
        if embedding is not None:
            embeddings.append(embedding)
            kept.append(turn)

    if not embeddings:
        return []

    labels = cluster_embeddings(embeddings, threshold=threshold, max_speakers=max_speakers)
    logger.info("diarize_done", turns=len(kept), speakers=len(set(labels)))
    return [
        SpeakerTurn(start=turn.start, end=turn.end, speaker=label)
        for turn, label in zip(kept, labels, strict=True)
    ]
