"""A fictional corpus that fills every page of the harness, for screenshots and demos.

:mod:`app.services.seed` writes bare rows, which is enough for the queue and the API but leaves
the editor without audio and the corpus, voices, costs and models pages empty. This seed goes the
long way round instead: it writes manifest directories with real 16 kHz mono FLAC, imports them
with the real importer, imports a diarization for every episode and links its voices, puts clips
in gold through :func:`~app.services.pots.set_segment_pot`, builds the queue, decides most of it
through :func:`~app.services.labeling.record_decision`, logs what the transcription would have
cost, and imports one fine-tuned model with its error files. Every row is one the real code paths
write, so every invariant holds.

Nothing in it is real. The shows, people and sentences are invented, and the audio is
syllable-shaped babble: it looks like speech on a waveform and is not intelligible. It must never
share a database with a real corpus, so :func:`require_demo_database` refuses any database whose
name does not end in ``_demo``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import AnnotationTask, Episode, LlmRequest, Segment
from app.services.audio import compute_peaks
from app.services.corpus import perturb
from app.services.diarization_import import import_diarization
from app.services.error_backfill import derive_error_files
from app.services.importer import import_manifest
from app.services.labeling import Decision, record_decision
from app.services.model_import import CARD_NAME, import_model_dir
from app.services.pots import set_segment_pot
from app.services.queue_builder import build_queue
from app.services.voice_attributes import set_voice_attributes
from app.services.voices import link_voices
from app.storage.base import ObjectStorage
from app.utils.hashing import sha256_bytes, sha256_file

ACTOR = "demo-seed"
SAMPLE_RATE = 16_000

#: The three recognisers and the fuser, under the system ids the real routes use.
RECOGNISERS: tuple[tuple[str, str, str], ...] = (
    ("elevenlabs-scribe-v2", "scribe_v2", "asr_scribe_v2"),
    ("mai-transcribe-2", "microsoft/mai-transcribe-2", "asr_mai_transcribe_2"),
    ("gemini-3.8-flash", "gemini-3.8-flash", "asr_gemini_flash"),
)
FUSION_SYSTEM = "fusion-gemini-3.8-flash-p1"
DEMO_MODEL_SLUG = "demo-flex-ft"


@dataclass(frozen=True)
class Person:
    """An invented speaker: a fixed voice embedding, a pitch, and what the owner declared."""

    key: str
    gender: str
    age_bracket: str
    pitch_hz: float


#: Hosts recur across their show's episodes, so the voice linker joins them; guests do not.
PEOPLE: tuple[Person, ...] = (
    Person("host-a", "female", "20_39", 210.0),
    Person("host-b", "male", "20_39", 120.0),
    Person("host-c", "male", "40_59", 105.0),
    Person("guest-1", "female", "40_59", 195.0),
    Person("guest-2", "male", "20_39", 135.0),
    Person("guest-3", "female", "20_39", 225.0),
    Person("guest-4", "male", "60_79", 98.0),
    Person("guest-5", "female", "under_20", 240.0),
    Person("guest-6", "male", "20_39", 128.0),
    Person("guest-7", "female", "40_59", 188.0),
)


@dataclass(frozen=True)
class Show:
    """An invented show, with its genre, topic, host and the sentences its episodes draw from."""

    show_id: str
    name: str
    genre: str
    topic: str
    host: str
    sentences: tuple[str, ...]


SHOWS: tuple[Show, ...] = (
    Show(
        "chiya-chat",
        "Chiya Chat",
        "podcast",
        "technology",
        "host-a",
        (
            "आज हामी open source को बारेमा कुरा गर्छौं।",
            "मेरो पहिलो laptop त second hand थियो।",
            "Code review मा सबैभन्दा धेरै कुरा सिकिन्छ।",
            "यो project को deadline अर्को हप्ता छ।",
            "Remote work मा discipline नै सबैभन्दा ठूलो कुरा हो।",
            "मैले Python बाट programming सुरु गरेँ।",
            "Startup मा funding भन्दा team महत्त्वपूर्ण हुन्छ।",
            "हाम्रो app को users अहिले दस हजार पुगे।",
            "Bug fix गर्न पूरै रात लाग्यो।",
            "Interview मा system design को प्रश्न आयो।",
            "Cloud को bill हेरेर म छक्क परेँ।",
            "AI ले हाम्रो काम सजिलो बनाउँछ कि गाह्रो?",
        ),
    ),
    Show(
        "pahad-pod",
        "Pahad Pod",
        "interview",
        "travel_food",
        "host-b",
        (
            "Mustang जाने बाटो अहिले धेरै राम्रो छ।",
            "त्यहाँको apple pie साँच्चै famous छ।",
            "Trekking मा जाँदा water bottle नबिर्सनुहोस्।",
            "हामीले homestay मा तीन रात बितायौं।",
            "Local खाना नै travel को सबैभन्दा राम्रो भाग हो।",
            "Bus ticket online नै book गर्न सकिन्छ।",
            "बिहान सबेरै sunrise हेर्न उठ्यौं।",
            "Budget travel गर्दा planning चाहिन्छ।",
            "त्यो village मा network नै थिएन।",
            "Photography को लागि golden hour उत्तम हुन्छ।",
        ),
    ),
    Show(
        "bazaar-brief",
        "Bazaar Brief",
        "commentary",
        "business_finance",
        "host-c",
        (
            "आज share market अलिकति तल झर्यो।",
            "Interest rate बढ्दा loan महँगो हुन्छ।",
            "Small business लाई digital payment ले सजिलो बनायो।",
            "यो quarter मा remittance बढेको देखिन्छ।",
            "Investment गर्नु अघि risk बुझ्नुपर्छ।",
            "Inflation ले सबैको budget मा असर पार्छ।",
            "नयाँ policy ले import मा tax बढायो।",
            "Mutual fund मा monthly SIP राम्रो option हो।",
            "Real estate को demand अहिले slow छ।",
            "Savings account को return धेरै कम छ।",
        ),
    ),
)


class DemoSeedError(RuntimeError):
    """The demo seed refused to run."""


@dataclass(frozen=True)
class DemoSummary:
    """What a demo seed run inserted. Zeroes mean the demo was already there."""

    episodes: int
    clips: int
    labelled: int


def require_demo_database(name: str) -> None:
    """Refuse any database but a demo one, so the seed can never write into a real corpus.

    Raises:
        DemoSeedError: ``name`` does not end in ``_demo``.
    """
    if not name.endswith("_demo"):
        raise DemoSeedError(
            f"refusing to seed {name!r}: the demo seed only writes to a database named *_demo"
        )


def _rng(*parts: object) -> random.Random:
    return random.Random(":".join(str(p) for p in parts))


def _embedding(person: Person, episode: str) -> list[float]:
    """A 256-d voice embedding: the person's fixed direction plus a little per-episode noise."""
    base = np.random.default_rng(int(hashlib.sha256(person.key.encode()).hexdigest()[:8], 16))
    noise = np.random.default_rng(int(hashlib.sha256(episode.encode()).hexdigest()[:8], 16))
    vector = base.normal(size=256) + 0.25 * noise.normal(size=256)
    return [round(float(x), 5) for x in vector / np.linalg.norm(vector)]


def speechlike(seconds: float, pitch_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Syllable-shaped babble: a voiced harmonic stack under bursts of about five per second.

    It draws like speech on a waveform -- syllables, short pauses, a wandering pitch -- and says
    nothing.
    """
    frames = int(seconds * SAMPLE_RATE)
    t = np.arange(frames) / SAMPLE_RATE
    contour = pitch_hz * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.3, 0.9) * t))
    phase = 2 * np.pi * np.cumsum(contour) / SAMPLE_RATE
    voiced = sum(np.sin(k * phase) / k for k in range(1, 10))
    envelope = np.zeros(frames)
    cursor = rng.uniform(0.05, 0.2)
    while cursor < seconds - 0.1:
        length = rng.uniform(0.1, 0.28)
        a, b = int(cursor * SAMPLE_RATE), min(frames, int((cursor + length) * SAMPLE_RATE))
        envelope[a:b] = np.hanning(b - a) * rng.uniform(0.5, 1.0)
        cursor += length + (rng.uniform(0.3, 0.6) if rng.random() < 0.12 else rng.uniform(0, 0.06))
    noise = rng.normal(scale=0.004, size=frames)
    return np.clip(0.28 * voiced * envelope + noise, -1, 1).astype(np.float32)


def _words(text: str, duration: float) -> list[dict[str, Any]]:
    """Clip-relative word spans spread over the clip (D26), weighted by word length."""
    tokens = text.split()
    weights = [len(token) + 2 for token in tokens]
    total, cursor, out = sum(weights), 0.15, []
    usable = max(duration - 0.3, 0.1)
    for token, weight in zip(tokens, weights, strict=True):
        span = usable * weight / total
        out.append(
            {
                "word": token,
                "start": round(cursor, 3),
                "end": round(cursor + span * 0.9, 3),
                "confidence": 0.9,
                "predicted_language": "en" if token.isascii() else "ne",
                "predicted_script": "latin" if token.isascii() else "devanagari",
            }
        )
        cursor += span
    return out


@dataclass
class _Clip:
    external_id: str
    reference: str
    speaker: str
    start: float
    end: float


def _write_episode(
    root: Path, show: Show, number: int, guests: list[Person], clips: int
) -> tuple[str, list[_Clip], dict[str, Any]]:
    """Write one episode's manifest directory; returns its id, its clips and its diarization."""
    episode_id = f"{show.show_id}_ep{number:03d}"
    rng = _rng("demo", episode_id)
    nprng = np.random.default_rng(int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16))
    host = next(p for p in PEOPLE if p.key == show.host)
    cast = [host, *guests]
    (root / "clips").mkdir(parents=True, exist_ok=True)

    records, out_clips, turns, audio = [], [], [], []
    cursor = 0.0
    speaker = 0
    for index in range(clips):
        if rng.random() < 0.55:
            speaker = (speaker + 1) % len(cast)
        person = cast[speaker]
        reference = rng.choice(show.sentences)
        duration = round(1.4 + 0.42 * len(reference.split()) + rng.uniform(0, 1.5), 2)
        data = speechlike(duration, person.pitch_hz, nprng)
        overlap: list[list[float]] = []
        if len(cast) > 1 and rng.random() < 0.2:  # a backchannel from the other side
            other = cast[(speaker + 1) % len(cast)]
            at, length = rng.uniform(0.3, duration - 1.0), rng.uniform(0.3, 0.8)
            a = int(at * SAMPLE_RATE)
            burst = speechlike(length, other.pitch_hz, nprng)
            data[a : a + len(burst)] += 0.6 * burst[: len(data) - a]
            overlap = [[round(at, 3), round(at + length, 3)]]
            turns.append(
                [
                    round(cursor + at, 3),
                    round(cursor + at + length, 3),
                    f"spk{(speaker + 1) % len(cast)}",
                ]
            )

        segment_id = f"{episode_id}_{index:04d}"
        clip_path = root / "clips" / f"{segment_id}.flac"
        sf.write(clip_path, np.clip(data, -1, 1), SAMPLE_RATE, format="FLAC")
        peaks_path = root / "peaks" / f"{segment_id}.json"
        peaks_path.parent.mkdir(parents=True, exist_ok=True)
        peaks_path.write_text(json.dumps(compute_peaks(clip_path, 200)), encoding="utf-8")
        gap = round(rng.uniform(0.2, 0.7), 2)
        audio += [np.clip(data, -1, 1), np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32)]
        turns.append([round(cursor, 3), round(cursor + duration, 3), f"spk{speaker}"])

        difficulty = rng.random()
        hypotheses = []
        for rank, (system_id, model_id, _route) in enumerate(RECOGNISERS):
            text = perturb(reference, rng, strength=difficulty * (0.4 + 0.3 * rank))
            hypotheses.append(
                {
                    "system_id": system_id,
                    "model_id": model_id,
                    "text": text,
                    "avg_logprob": round(-0.1 - difficulty * rng.uniform(0.2, 1.4), 4)
                    if rank == 0
                    else None,
                    "words": _words(text, duration),
                }
            )
        # Most seeds are right; a few fail the way a fuser does, so the hazard gates fire (D74):
        # words no recogniser heard, or a run all of them heard that the seed dropped.
        if difficulty > 0.93:
            fused = f"{reference} अनि त्यो कुरा पनि हो"
        elif difficulty > 0.87 and len(reference.split()) > 5:
            fused = " ".join(reference.split()[:-3])
        elif difficulty > 0.75:
            fused = perturb(reference, rng, strength=0.6)
        else:
            fused = reference
        hypotheses.append(
            {
                "system_id": FUSION_SYSTEM,
                "model_id": "gemini-3.8-flash",
                "kind": "fusion",
                "text": fused,
                "words": _words(fused, duration),
                "fusion": {"code": "m", "window": 0, "depth": 0},
            }
        )
        records.append(
            {
                "segment_id": segment_id,
                "episode_id": episode_id,
                "speaker_id": f"spk{speaker}",
                "start_time": round(cursor, 3),
                "end_time": round(cursor + duration, 3),
                "clip_path": f"clips/{segment_id}.flac",
                "clip_checksum": sha256_file(clip_path),
                "peaks_path": f"peaks/{segment_id}.json",
                "p_en": round(
                    sum(w.isascii() for w in reference.split()) / len(reference.split()), 3
                ),
                "lid": "mixed",
                "hypotheses": hypotheses,
                "vad_spans": [[0.15, round(duration - 0.15, 3)]],
                "overlap_spans": overlap,
                "acoustics": {
                    "version": "acoustics-v2",
                    "bandwidth_hz": rng.choice([7800.0, 7900.0, 5400.0]),
                    "snr_db": round(rng.uniform(8, 38), 1),
                    "c50_db": round(rng.uniform(5, 45), 1),
                },
            }
        )
        out_clips.append(_Clip(segment_id, reference, f"spk{speaker}", cursor, cursor + duration))
        cursor += duration + gap

    episode_audio = root / "audio" / "episode.flac"
    episode_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(episode_audio, np.concatenate(audio), SAMPLE_RATE, format="FLAC")
    titles = _TITLES[show.show_id]
    episode = {
        "episode_id": episode_id,
        "show_id": show.show_id,
        "title": f"{show.name} #{number}: {titles[number % len(titles)]}",
        "source_uri": f"https://example.invalid/{episode_id}",
        "published_at": (dt.date(2026, 6, 1) + dt.timedelta(days=17 * number)).isoformat(),
        "source_audio_checksum": sha256_bytes(f"demo:{episode_id}".encode()),
        "duration_seconds": round(cursor, 3),
        "pipeline_version": "demo-v1",
        "pipeline_commit": "0000000",
        "audio_path": "audio/episode.flac",
        "genre": show.genre,
        "topic": show.topic,
        "speaker_count": len(cast),
        "speakers": {
            f"spk{i}": {
                "role": "host" if i == 0 else "guest",
                "gender": p.gender,
                "age_bracket": p.age_bracket,
            }
            for i, p in enumerate(cast)
        },
    }
    (root / "episode.json").write_text(json.dumps(episode, ensure_ascii=False), encoding="utf-8")
    (root / "segments.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    diarization = {
        "turns": sorted(turns),
        "labels": [f"spk{i}" for i in range(len(cast))],
        "embeddings": [_embedding(p, episode_id) for p in cast],
    }
    return episode_id, out_clips, diarization


_TITLES: dict[str, tuple[str, ...]] = {
    "chiya-chat": (
        "Shipping on a Friday",
        "Your first open-source PR",
        "Burnout and side projects",
    ),
    "pahad-pod": ("Ten days in Mustang", "Homestays done right", "Trekking on a budget"),
    "bazaar-brief": ("What rate hikes mean for you", "Remittance season", "SIPs explained"),
}


def _log_costs(session: Session, episode_id: str, clips: list[_Clip]) -> None:
    """The ``llm_requests`` rows the ingest would have written: three recognisers per clip, one
    fusion and one topic call per episode. Dry-run-shaped numbers, not real prices."""
    rng = _rng("costs", episode_id)
    base = dt.datetime(2026, 9, 1, tzinfo=dt.UTC) + dt.timedelta(
        days=int(hashlib.sha256(episode_id.encode()).hexdigest()[:4], 16) % 50
    )
    rows = []
    for clip in clips:
        seconds = clip.end - clip.start
        for _system, model, route in RECOGNISERS:
            rows.append(
                (route, model, Decimal(str(round(seconds * rng.uniform(0.00008, 0.0002), 6))))
            )
    rows.append(("fuse_transcript", "gemini-3.8-flash", Decimal("0.0412")))
    rows.append(("classify_topic", "gemini-3.8-flash", Decimal("0.0009")))
    for index, (route, model, cost) in enumerate(rows):
        failed = rng.random() < 0.02
        session.add(
            LlmRequest(
                route=route,
                model=model,
                request_hash=hashlib.sha256(f"{episode_id}:{index}".encode()).hexdigest(),
                input_summary=f"demo {episode_id}",
                prompt_tokens=rng.randint(200, 900),
                completion_tokens=rng.randint(20, 120),
                estimated_cost_usd=Decimal(0) if failed else cost,
                latency_ms=rng.randint(700, 4200),
                status="failed" if failed else "succeeded",
                error_message="HTTP 503 (demo)" if failed else None,
                created_at=base + dt.timedelta(seconds=index * 7),
            )
        )


def _decide(session: Session, settings: Settings, references: dict[str, str]) -> int:
    """Decide about two thirds of the queue, highest priority first; the rest stays open."""
    tasks = list(
        session.scalars(
            sa.select(AnnotationTask)
            .where(AnnotationTask.status == "pending")
            .order_by(AnnotationTask.priority_score.desc(), AnnotationTask.id)
        )
    )
    rng = _rng("decisions")
    decided = 0
    base = dt.datetime(2026, 9, 20, 9, 0, tzinfo=dt.UTC)
    for index, task in enumerate(tasks):
        if index % 3 == 2:
            continue
        segment = task.segment
        reference = references[segment.external_id]
        seed = next((h.text_raw for h in segment.hypotheses if h.system.kind == "fusion"), "")
        submitted = base + dt.timedelta(minutes=3 * index)
        roll = rng.random()
        if seed != reference:
            decision = Decision(
                disposition="edited",
                final_text=reference,
                duration_ms=rng.randint(7000, 24000),
                submitted_at=submitted,
            )
        elif roll < 0.05:
            decision = Decision(
                disposition="unusable_audio",
                duration_ms=rng.randint(2000, 5000),
                submitted_at=submitted,
            )
        elif segment.pot == "train" and roll < 0.4:
            decision = Decision(
                disposition="accepted_unchanged",
                final_text=reference,
                verification_tier="screened",
                duration_ms=rng.randint(300, 900),
                submitted_at=submitted,
            )
        else:
            decision = Decision(
                disposition="accepted_unchanged",
                final_text=reference,
                duration_ms=rng.randint(1200, 4500),
                submitted_at=submitted,
            )
        decision = Decision(**{**decision.__dict__, "annotator": "demo"})
        record_decision(session, task, decision, settings=settings)
        decided += 1
    return decided


def _write_model(session: Session, settings: Settings, references: dict[str, str]) -> None:
    """One fine-tuned model whose gold and val transcripts are the labels, lightly wrong."""
    folder = Path(settings.models.root) / DEMO_MODEL_SLUG
    folder.mkdir(parents=True, exist_ok=True)
    card = {
        "name": "Demo fine-tune",
        "created_at": "2026-10-01T12:00:00+00:00",
        "description": "Invented transcripts for the demo; scored like any other run.",
        "architecture": "encoder-decoder",
        "decoder": "greedy",
    }
    (folder / CARD_NAME).write_text(json.dumps(card), encoding="utf-8")
    rng = _rng("model")
    rows: dict[str, list[str]] = {"gold": [], "val": []}
    for segment in session.scalars(sa.select(Segment).join(Episode).order_by(Segment.external_id)):
        split = "gold" if segment.pot == "gold" else segment.episode.split
        if split not in rows:
            continue
        text = perturb(references[segment.external_id], rng, strength=rng.random() * 0.8)
        rows[split].append(
            json.dumps({"segment_id": segment.external_id, "text": text}, ensure_ascii=False)
        )
    for split, lines in rows.items():
        if lines:
            (folder / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import_model_dir(session, folder, settings=settings, actor=ACTOR)
    session.flush()
    derive_error_files(session, folder)


def seed_demo(
    session: Session,
    *,
    storage: ObjectStorage,
    work_root: Path,
    settings: Settings | None = None,
    episodes: int = 8,
    clips_per_episode: int = 24,
) -> DemoSummary:
    """Write and import the demo corpus, unless it is already there.

    Args:
        session: Open session; the caller commits. The caller also checks the database is a demo
            one (:func:`require_demo_database`); tests run this against their own database.
        storage: Where clips and peaks go.
        work_root: Scratch directory for the manifest directories.
        settings: Configuration override; ``models.root`` receives the demo model's folder.
        episodes: Episodes to create, spread round-robin over the three shows.
        clips_per_episode: Clips per episode.
    """
    settings = settings or get_settings()
    if session.scalar(sa.select(sa.func.count()).select_from(Episode)):
        return DemoSummary(episodes=0, clips=0, labelled=0)

    references: dict[str, str] = {}
    payload: dict[str, Any] = {}
    guests = [p for p in PEOPLE if p.key.startswith("guest")]
    for number in range(episodes):
        show = SHOWS[number % len(SHOWS)]
        cast = [guests[(number * 3 + k) % len(guests)] for k in range(1 + number % 2)]
        root = work_root / f"episode_{number:02d}"
        episode_id, clips, diarization = _write_episode(
            root, show, number // len(SHOWS) + 1, cast, clips_per_episode
        )
        import_manifest(session, root, storage=storage, settings=settings)
        references |= {c.external_id: c.reference for c in clips}
        payload[episode_id] = diarization
        _log_costs(session, episode_id, clips)
    session.flush()

    import_diarization(session, payload, model="demo-diarizer", source="demo", actor=ACTOR)
    link_voices(session, actor=ACTOR)
    session.flush()
    _declare_voices(session)

    # Gold is chosen by hand, clip by clip (D71): every fourth clip of the first two shows.
    for segment in session.scalars(sa.select(Segment).join(Episode).order_by(Segment.external_id)):
        if segment.episode.show_id != SHOWS[2].show_id and int(segment.external_id[-4:]) % 4 == 0:
            set_segment_pot(session, segment, "gold", actor=ACTOR, reason="demo")
    session.flush()

    build_queue(session, settings=settings, audit_sample_rate=0.0)
    session.flush()
    labelled = _decide(session, settings, references)
    session.flush()
    _write_model(session, settings, references)
    return DemoSummary(episodes=episodes, clips=len(references), labelled=labelled)


def _declare_voices(session: Session) -> None:
    """Give each linked voice the gender and age its person was declared with, as by ear (D104)."""
    from app.models import DiarizationRun

    for run in session.scalars(sa.select(DiarizationRun).order_by(DiarizationRun.id)):
        episode = session.get(Episode, run.episode_id)
        speakers = (episode.metadata_jsonb or {}).get("speakers") or {}
        for label, voice in (run.voices_jsonb or {}).items():
            fields = speakers.get(label) or {}
            set_voice_attributes(
                session,
                voice,
                gender=fields.get("gender"),
                age_bracket=fields.get("age_bracket"),
                annotator=ACTOR,
            )
