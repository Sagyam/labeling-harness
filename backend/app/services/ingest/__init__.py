"""Backend ingestion service for podcast episodes.

Coordinates the 6-stage ingestion pipeline:
1. Audio normalization via FFmpeg with loudnorm (16 kHz mono FLAC)
2. Utterance segmentation via Silero VAD (2.0s - 20.0s boundaries)
3. Cloud ASR inference across every configured `asr*` route (logged to llm_requests)
4. Fusion: a reasoning model reconciles the recognisers into one transcript per clip (D72)
5. Orthography-aware token tagging, CMI, and rule flags -- on the fused text where there is one
6. Manifest generation and direct database import + queue building

A segment that cannot be transcribed by every configured system is discarded and the run carries
on (D46). Stage 3 is where all the money is: it dispatches every `asr*` route at every clip, so
one refused clip near the end of a two-hour episode used to throw away hours of paid inference
that had already succeeded. Only an episode with nothing left fails outright. What was dropped,
and which system dropped it, rides in the completion summary.
"""

from .audio import normalize_audio
from .job import DiscardedSegment, IngestJob, IngestLog, LockedSession
from .manager import IngestionManager, manager
from .pipeline import run_pipeline

__all__ = [
    "DiscardedSegment",
    "IngestJob",
    "IngestLog",
    "IngestionManager",
    "LockedSession",
    "manager",
    "normalize_audio",
    "run_pipeline",
]
