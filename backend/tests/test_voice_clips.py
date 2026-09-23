"""Tests for the voice page, verdicts, prints and voiceprint suggestions (D99)."""

from __future__ import annotations

import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import (
    AnnotationTask,
    AuditLog,
    DiarizationRun,
    HypothesisWord,
    LabelWord,
    Segment,
    SegmentLabel,
)
from app.services.diarization_import import import_diarization
from app.services.speaker_attribution import queue_for_speakers, rank_for_speakers
from app.services.voice_clips import voice_prints
from app.services.voices import link_voices

pytestmark = pytest.mark.db

MODEL = "pyannote/speaker-diarization-community-1"
DIM = 256


def axis(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


class FakeEmbedder:
    """Every chunk sounds like ``vector``; counts what it was asked to embed."""

    available = True

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float64)
        self.calls: list[int] = []

    def embed(self, chunks):
        self.calls.append(len(chunks))
        return [self.vector.copy() for _ in chunks]


@pytest.fixture
def embedder(client) -> FakeEmbedder:
    """Sounds like SPEAKER_01's centroid, wired into the app in place of the offline model."""
    from app.api.deps import get_voice_embedder

    fake = FakeEmbedder(axis(1))
    client.app.dependency_overrides[get_voice_embedder] = lambda: fake
    return fake


def _segments(session: Session) -> list[Segment]:
    return list(session.scalars(sa.select(Segment).order_by(Segment.id)))


def _long(clips: list[Segment]) -> Segment:
    return max(clips, key=lambda s: s.duration_seconds)


def _clip_relative_words(segments: list[Segment]) -> None:
    # The shared fixture writes episode-relative word spans; ingest stores them clip-relative.
    for segment in segments:
        for hypothesis in segment.hypotheses:
            for w in hypothesis.words:
                if w.start_time is not None:
                    w.start_time = round(w.start_time - segment.start_time, 3)
                    w.end_time = round(w.end_time - segment.start_time, 3)


def _diarize(session: Session, episode: str, turns: list) -> None:
    import_diarization(
        session,
        {
            episode: {
                "turns": turns,
                "labels": ["SPEAKER_00", "SPEAKER_01"],
                "embeddings": [axis(0), axis(1)],
            }
        },
        model=MODEL,
        source="t.json",
        actor="test",
    )
    link_voices(session, actor="test")
    session.flush()


@pytest.fixture
def solo(client, imported_episode: str, db_session: Session) -> dict[str, list[Segment]]:
    """Clips 0-2 are SPEAKER_00 alone, 3-5 SPEAKER_01 alone; both voices linked."""
    segments = _segments(db_session)
    turns = [
        [s.start_time, s.end_time, "SPEAKER_00" if i < 3 else "SPEAKER_01"]
        for i, s in enumerate(segments)
    ]
    _diarize(db_session, imported_episode, turns)
    run = db_session.scalars(sa.select(DiarizationRun)).one()
    by_voice: dict[str, list[Segment]] = {}
    for i, s in enumerate(segments):
        voice = run.voices_jsonb["SPEAKER_00" if i < 3 else "SPEAKER_01"]
        by_voice.setdefault(voice, []).append(s)
    return by_voice


# --- the voice page ---------------------------------------------------------------------------


def test_a_voice_page_lists_its_episode_and_solo_clips(client, solo) -> None:
    voice, clips = next(iter(solo.items()))
    body = client.get(f"/voices/{voice}").json()
    assert body["voice"] == voice
    assert [e["external_id"] for e in body["episodes"]] == ["api_ep001"]
    long_enough = [s for s in clips if s.duration_seconds >= 1.5]
    assert body["total_clips"] == len(long_enough)
    assert {c["segment_id"] for c in body["clips"]} == {s.id for s in long_enough}
    durations = [c["duration_seconds"] for c in body["clips"]]
    assert durations == sorted(durations, reverse=True)
    assert all(c["whole"] and c["start"] == 0.0 for c in body["clips"])
    assert body["print_source"] == "diarizer"
    assert body["reference"]["segment_id"] == body["clips"][0]["segment_id"]
    assert body["clips"][0]["audio_url"].endswith("/audio")


def test_a_clip_with_another_voice_offers_only_the_stretch_alone(
    client, imported_episode, db_session
) -> None:
    segments = _segments(db_session)
    first = max(segments, key=lambda s: s.duration_seconds)
    turns = [[s.start_time, s.end_time, "SPEAKER_00"] for s in segments]
    turns.append([first.start_time, first.start_time + 0.5, "SPEAKER_01"])
    _diarize(db_session, imported_episode, turns)
    run = db_session.scalars(sa.select(DiarizationRun)).one()
    body = client.get(f"/voices/{run.voices_jsonb['SPEAKER_00']}").json()
    offered = next(c for c in body["clips"] if c["segment_id"] == first.id)
    assert offered["start"] == pytest.approx(0.5)
    assert offered["end"] == pytest.approx(first.duration_seconds)
    assert offered["whole"] is False


def test_detected_crosstalk_is_left_out_of_the_stretch(client, solo, db_session) -> None:
    voice, clips = next(iter(solo.items()))
    clip = _long(clips)
    clip.overlap_spans_jsonb = [[0.0, 1.0]]
    db_session.flush()
    body = client.get(f"/voices/{voice}").json()
    offered = next(c for c in body["clips"] if c["segment_id"] == clip.id)
    assert offered["start"] == pytest.approx(1.0)


def test_a_stretch_shows_only_the_words_said_in_it(client, solo, db_session) -> None:
    voice, clips = next(iter(solo.items()))
    clip = _long(clips)
    fused = next(h for h in clip.hypotheses if h.system.kind == "fusion")
    # The fixture's fused seed has no word spans; give it one word a second.
    words = [
        HypothesisWord(position=i, word_raw=f"w{i}", start_time=i + 0.1, end_time=i + 0.9)
        for i in range(int(clip.duration_seconds))
    ]
    fused.words.extend(words)
    cut = words[len(words) // 2].start_time
    clip.overlap_spans_jsonb = [[0.0, cut]]
    db_session.flush()
    body = client.get(f"/voices/{voice}").json()
    offered = next(c for c in body["clips"] if c["segment_id"] == clip.id)
    said = [w.word_raw for w in words if (w.start_time + w.end_time) / 2 >= offered["start"]]
    assert offered["text"] == " ".join(said)


def test_an_unknown_voice_is_404(client, solo) -> None:
    assert client.get("/voices/v999").status_code == 404
    assert client.get("/voices/nonsense").status_code == 404


# --- verdicts ---------------------------------------------------------------------------------


def test_confirming_a_clip_embeds_it_and_audits_it(client, solo, embedder, db_session) -> None:
    voice, clips = next(iter(solo.items()))
    clip = _long(clips)
    response = client.post(f"/voices/{voice}/clips/{clip.id}", json={"verdict": "confirmed"})
    assert response.status_code == 200, response.text
    assert response.json()["embedded"] is True
    assert response.json()["confirmed"] == 1
    assert embedder.calls == [1]
    body = client.get(f"/voices/{voice}").json()
    assert body["print_source"] == "confirmed"
    assert body["clips"][0]["segment_id"] == clip.id
    assert body["clips"][0]["verdict"] == "confirmed"
    audit = db_session.scalars(
        sa.select(AuditLog).where(AuditLog.entity_type == "voice_confirmations")
    ).one()
    assert audit.new_values_jsonb["verdict"] == "confirmed"
    assert audit.new_values_jsonb["end_time"] == pytest.approx(clip.duration_seconds)


def test_without_the_model_a_confirmation_is_kept_unembedded(client, solo) -> None:
    voice, clips = next(iter(solo.items()))
    response = client.post(f"/voices/{voice}/clips/{clips[0].id}", json={"verdict": "confirmed"})
    assert response.status_code == 200
    assert response.json()["embedded"] is False


def test_a_rejected_clip_sorts_last_and_a_cleared_one_is_unjudged(client, solo) -> None:
    voice = next(iter(solo))
    offered = [c["segment_id"] for c in client.get(f"/voices/{voice}").json()["clips"]]
    first = offered[0]
    client.post(f"/voices/{voice}/clips/{first}", json={"verdict": "rejected"})
    body = client.get(f"/voices/{voice}").json()
    assert body["clips"][-1]["segment_id"] == first
    assert body["clips"][-1]["verdict"] == "rejected"
    assert body["rejected"] == 1
    client.post(f"/voices/{voice}/clips/{first}", json={"verdict": "cleared"})
    body = client.get(f"/voices/{voice}").json()
    assert body["rejected"] == 0
    assert body["clips"][0]["verdict"] is None


def test_a_clip_from_an_episode_without_the_voice_is_refused(client, solo) -> None:
    clips_b = list(solo.values())[1]
    unrelated = client.post(f"/voices/v998/clips/{clips_b[0].id}", json={"verdict": "confirmed"})
    assert unrelated.status_code == 409


def test_an_unknown_verdict_is_refused(client, solo) -> None:
    voice, clips = next(iter(solo.items()))
    response = client.post(f"/voices/{voice}/clips/{clips[0].id}", json={"verdict": "maybe"})
    assert response.status_code == 422


# --- prints -----------------------------------------------------------------------------------


def test_a_confirmed_clip_replaces_the_diarizer_centroid(client, solo, db_session) -> None:
    from app.api.deps import get_voice_embedder

    voice, clips = next(iter(solo.items()))
    run = db_session.scalars(sa.select(DiarizationRun)).one()
    number = next(
        i + 1 for i, lab in enumerate(run.speakers_jsonb) if run.voices_jsonb[lab] == voice
    )
    before = voice_prints(db_session, run)[number]
    assert before.source == "diarizer"

    elsewhere = FakeEmbedder(axis(7))
    client.app.dependency_overrides[get_voice_embedder] = lambda: elsewhere
    client.post(f"/voices/{voice}/clips/{clips[0].id}", json={"verdict": "confirmed"})
    after = voice_prints(db_session, run)[number]
    assert after.source == "confirmed" and after.clips == 1
    assert after.vector[7] == pytest.approx(1.0)


# --- suggestions in the editor ----------------------------------------------------------------


@pytest.fixture
def crossing(client, imported_episode: str, db_session: Session) -> AnnotationTask:
    """Labelled gold clips where SPEAKER_00 hands over to SPEAKER_01 mid-clip; one task queued."""
    for row in client.get("/queue", params={"limit": 50}).json():
        client.post(f"/tasks/{row['task_id']}/accept", json={})
    segments = _segments(db_session)
    _clip_relative_words(segments)
    turns = []
    for segment in segments:
        segment.pot = "gold"
        mid = (segment.start_time + segment.end_time) / 2
        turns.append([segment.start_time, mid + 0.3, "SPEAKER_00"])
        turns.append([mid - 0.3, segment.end_time, "SPEAKER_01"])
    _diarize(db_session, imported_episode, turns)
    queue_for_speakers(db_session, rank_for_speakers(db_session)[:1], actor="test")
    return db_session.scalars(
        sa.select(AnnotationTask).where(AnnotationTask.queue == "speakers")
    ).one()


def test_clean_words_on_the_other_lane_get_a_suggestion(
    client, crossing, embedder, db_session
) -> None:
    lanes = client.get(f"/tasks/{crossing.id}").json()["lanes"]
    assert lanes["voiceprint"] is True
    run = db_session.scalars(sa.select(DiarizationRun)).one()
    s01 = run.speakers_jsonb.index("SPEAKER_01") + 1
    s00 = run.speakers_jsonb.index("SPEAKER_00") + 1
    segment = db_session.get(Segment, crossing.segment_id)
    mid = segment.duration_seconds / 2
    for w in lanes["words"]:
        if w["start"] is None:
            continue
        centre = (w["start"] + w["end"]) / 2
        in_crosstalk = mid - 0.3 <= centre < mid + 0.3
        if w["speaker"] == s00 and not in_crosstalk:
            assert w["suggested_speaker"] == s01, w
            assert w["suggestion_margin"] == pytest.approx(1.0)
        else:
            assert w["suggested_speaker"] is None, w
    assert {s["print_source"] for s in lanes["speakers"]} == {"diarizer"}


def test_without_the_model_the_lanes_come_without_suggestions(client, crossing) -> None:
    lanes = client.get(f"/tasks/{crossing.id}").json()["lanes"]
    assert lanes["voiceprint"] is False
    assert all(w["suggested_speaker"] is None for w in lanes["words"])
    assert lanes["words"]


def test_a_failing_voiceprint_still_serves_the_lanes(client, crossing) -> None:
    from app.api.deps import get_voice_embedder

    class Broken(FakeEmbedder):
        def embed(self, chunks):
            raise RuntimeError("onnx fell over")

    client.app.dependency_overrides[get_voice_embedder] = lambda: Broken(axis(0))
    response = client.get(f"/tasks/{crossing.id}")
    assert response.status_code == 200
    assert response.json()["lanes"]["words"]


def test_a_saved_word_keeps_the_suggestion_it_was_served_with(
    client, crossing, embedder, db_session
) -> None:
    lanes = client.get(f"/tasks/{crossing.id}").json()["lanes"]
    words = [w for w in lanes["words"] if w["start"] is not None and w["speaker"] is not None]
    suggested = next(w for w in words if w["suggested_speaker"] is not None)
    response = client.post(
        f"/tasks/{crossing.id}/attribute",
        json={"diarization_run_id": lanes["diarization_run_id"], "words": words},
    )
    assert response.status_code == 200, response.text
    label = db_session.get(SegmentLabel, response.json()["label_id"])
    stored = db_session.scalars(
        sa.select(LabelWord).where(
            LabelWord.label_id == label.id, LabelWord.suggested_speaker.is_not(None)
        )
    ).all()
    assert stored and all(w.suggested_speaker == "SPEAKER_01" for w in stored)
    assert suggested["word"] in {w.word for w in stored}


def test_the_report_reads_the_saved_suggestions(client, crossing, embedder, db_session) -> None:
    from app.services.speaker_attribution import saved_speaker_words
    from app.services.voiceprint import suggestion_report

    lanes = client.get(f"/tasks/{crossing.id}").json()["lanes"]
    words = [w for w in lanes["words"] if w["start"] is not None and w["speaker"] is not None]
    # Take every suggestion, as an annotator who agreed with the voiceprint would.
    taken = [{**w, "speaker": w["suggested_speaker"] or w["speaker"]} for w in words]
    client.post(
        f"/tasks/{crossing.id}/attribute",
        json={"diarization_run_id": lanes["diarization_run_id"], "words": taken},
    )
    report = suggestion_report(saved_speaker_words(db_session))
    suggested = sum(w["suggested_speaker"] is not None for w in words)
    assert report.suggested == suggested > 0
    assert report.precision == 1.0
    assert report.moved == report.moved_suggested == suggested
