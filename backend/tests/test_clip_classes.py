"""Every clip's classes: the conditions a WER is split by (roadmap item 1, D87)."""

from __future__ import annotations

import pytest

from app.services.clip_classes import (
    AXES,
    ClipFacts,
    classify,
    declared_value,
    dominant_speaker,
    overlap_bucket,
    overlap_share,
    speakers_present,
    turn_changes,
)


def _facts(**overrides) -> ClipFacts:
    base = {
        "duration": 10.0,
        "vad_spans": [[0.0, 10.0]],
        "overlap_spans": [],
        "turns": [(0.0, 10.0, "A")],
        "cmi": 0.0,
        "acoustics": None,
        "voice": None,
        "voice_train_seconds": None,
        "gender": "male",
        "age_bracket": "20_39",
    }
    return ClipFacts(**(base | overrides))


def test_every_axis_names_its_buckets_and_its_baseline_is_one_of_them() -> None:
    for axis in AXES:
        if axis.baseline is not None:
            assert axis.baseline in axis.buckets, axis.name
        if axis.unmeasured is not None and axis.buckets:  # voices are an open set
            assert axis.unmeasured in axis.buckets, axis.name
            assert axis.unmeasured != axis.baseline


def test_classify_gives_every_axis_a_bucket_it_declares() -> None:
    classes = classify(_facts())
    for axis in AXES:
        assert axis.name in classes
        if axis.buckets:
            assert classes[axis.name] in axis.buckets, axis.name


# --- crosstalk --------------------------------------------------------------------------------


def test_overlap_share_is_the_fraction_of_the_clip_in_crosstalk() -> None:
    assert overlap_share([[0.0, 1.0], [3.0, 4.0]], 10.0) == pytest.approx(0.2)
    assert overlap_share([], 10.0) == 0.0
    assert overlap_share(None, 10.0) is None  # never measured is not clean


@pytest.mark.parametrize(
    ("share", "bucket"),
    [
        (None, "unmeasured"),
        (0.0, "none"),
        (0.03, "0-5%"),
        (0.05, "5-15%"),
        (0.15, "5-15%"),
        (0.2, ">15%"),
    ],
)
def test_overlap_buckets_follow_the_crosstalk_findings(share: float | None, bucket: str) -> None:
    assert overlap_bucket(share) == bucket


# --- speakers and turns -----------------------------------------------------------------------


def test_a_speaker_counts_only_after_half_a_second_of_talk() -> None:
    turns = [(0.0, 6.0, "A"), (6.0, 6.3, "B"), (6.3, 10.0, "A")]
    assert speakers_present(turns) == ["A"]
    assert classify(_facts(turns=turns))["speakers"] == "1"


def test_speaker_buckets() -> None:
    two = [(0.0, 5.0, "A"), (5.0, 10.0, "B")]
    three = [*two, (8.0, 9.0, "C")]
    assert classify(_facts(turns=two))["speakers"] == "2"
    assert classify(_facts(turns=three))["speakers"] == "3+"
    assert classify(_facts(turns=[]))["speakers"] == "none"
    assert classify(_facts(turns=None))["speakers"] == "undiarized"


def test_turn_changes_count_hand_overs_between_speakers_who_count() -> None:
    assert turn_changes([(0.0, 5.0, "A"), (5.0, 10.0, "A")]) == 0
    assert turn_changes([(0.0, 5.0, "A"), (5.0, 10.0, "B")]) == 1
    assert turn_changes([(0.0, 4.0, "A"), (4.0, 7.0, "B"), (7.0, 10.0, "A")]) == 2
    # A 0.2 s backchannel is not a speaker, so it is not a hand-over either.
    assert turn_changes([(0.0, 5.0, "A"), (5.0, 5.2, "B"), (5.2, 10.0, "A")]) == 0


def test_turn_change_buckets() -> None:
    many = [(float(i), float(i + 1), "AB"[i % 2]) for i in range(10)]
    assert classify(_facts(turns=many))["turn_changes"] == "2+"
    assert classify(_facts(turns=None))["turn_changes"] == "undiarized"


def test_the_dominant_speaker_has_the_most_talk_time() -> None:
    assert dominant_speaker([(0.0, 3.0, "A"), (3.0, 10.0, "B")]) == "B"
    assert dominant_speaker([]) is None


# --- pauses and duration ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spans", "bucket"),
    [
        ([[0.0, 10.0]], "<2%"),
        ([[0.0, 4.5], [5.0, 10.0]], "2-10%"),  # 5% silence
        ([[0.0, 4.0], [6.0, 10.0]], ">10%"),  # 20% silence
        (None, "unmeasured"),
    ],
)
def test_pause_share_is_the_clip_outside_speech(spans, bucket: str) -> None:
    assert classify(_facts(vad_spans=spans))["pause"] == bucket


@pytest.mark.parametrize(("seconds", "bucket"), [(3.0, "<5 s"), (10.0, "5-15 s"), (18.0, "15+ s")])
def test_duration_buckets(seconds: float, bucket: str) -> None:
    assert classify(_facts(duration=seconds, vad_spans=[[0.0, seconds]]))["duration"] == bucket


# --- acoustics --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("acoustics", "bucket"),
    [
        ({"bandwidth_hz": 3900.0}, "<4.5 kHz"),
        ({"bandwidth_hz": 5300.0}, "4.5-6.5 kHz"),
        ({"bandwidth_hz": 7800.0}, "6.5+ kHz"),
        (None, "unmeasured"),
        ({}, "unmeasured"),
    ],
)
def test_bandwidth_buckets(acoustics, bucket: str) -> None:
    assert classify(_facts(acoustics=acoustics))["bandwidth"] == bucket


# --- voices -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("voice", "seconds", "bucket"),
    [
        (None, None, "unlinked"),
        ("v01", 0.0, "unseen"),
        ("v01", 300.0, "<10 min"),
        ("v01", 1800.0, "10-60 min"),
        ("v01", 7200.0, "1 h+"),
    ],
)
def test_voice_exposure_is_that_voices_hours_in_train(voice, seconds, bucket: str) -> None:
    classes = classify(_facts(voice=voice, voice_train_seconds=seconds))
    assert classes["voice_exposure"] == bucket
    assert classes["voice"] == (voice or "unlinked")


# --- declared speakers ------------------------------------------------------------------------


def test_a_declared_value_holds_for_a_clip_only_when_every_speaker_shares_it() -> None:
    one = {"speakers": {"spk0": {"gender": "female"}}}
    same = {"speakers": {"spk0": {"gender": "male"}, "spk1": {"gender": "male"}}}
    mixed = {"speakers": {"spk0": {"gender": "male"}, "spk1": {"gender": "female"}}}
    blank = {"speakers": {"spk0": {"gender": "male"}, "spk1": {"role": "guest"}}}
    short = {"speakers": {"spk0": {"gender": "male"}}, "speaker_count": 2}
    assert declared_value(one, "gender") == "female"
    assert declared_value(same, "gender") == "male"
    assert declared_value(mixed, "gender") == "mixed"
    assert declared_value(blank, "gender") == "undeclared"
    assert declared_value(short, "gender") == "undeclared"  # a row left blank is someone
    assert declared_value(None, "gender") == "undeclared"


# --- CMI, descriptive -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cmi", "bucket"),
    [(0.0, "0"), (7.7, "<15"), (20.0, "15-30"), (42.0, "30+"), (None, "unmeasured")],
)
def test_cmi_buckets(cmi, bucket: str) -> None:
    assert classify(_facts(cmi=cmi))["cmi"] == bucket


def test_cmi_is_marked_descriptive() -> None:
    assert {a.name for a in AXES if a.descriptive} == {"cmi"}


# --- reading the facts from the database ------------------------------------------------------


@pytest.mark.db
def test_facts_come_from_the_newest_diarization_run_and_the_stored_spans(
    db_session, imported_episode
) -> None:
    import sqlalchemy as sa

    from app.models import DiarizationRun, Episode, SpeakerTurn
    from app.services.clip_classes import classify_segments, load_clip_facts

    episode = db_session.scalars(
        sa.select(Episode).where(Episode.external_id == imported_episode)
    ).one()
    segments = sorted(episode.segments, key=lambda s: s.start_time)
    first = segments[0]

    undiarized = classify_segments(db_session, segments)
    assert {c["speakers"] for c in undiarized.values()} == {"undiarized"}

    def run(checksum: str, turns: list[tuple[float, float, str]]) -> None:
        db_session.add(
            DiarizationRun(
                episode_id=episode.id,
                model="test",
                checksum=checksum,
                speakers_jsonb=sorted({s for _, _, s in turns}),
                turns=[SpeakerTurn(speaker=s, start_time=a, end_time=b) for a, b, s in turns],
            )
        )
        db_session.flush()

    t0, t1 = first.start_time, first.end_time
    run("old", [(t0, t1, "X")])
    run("new", [(t0, (t0 + t1) / 2, "A"), ((t0 + t1) / 2, t1, "B")])

    facts = load_clip_facts(db_session, [first])[first.id]
    assert [s for _, _, s in facts.turns] == ["A", "B"]
    assert facts.turns[0][0] == pytest.approx(0.0)  # clip-relative
    classes = classify_segments(db_session, [first])[first.id]
    assert classes["speakers"] == "2"
    assert classes["turn_changes"] == "1"
    later = classify_segments(db_session, [segments[1]])[segments[1].id]
    assert later["speakers"] == "none"  # diarized, and nobody heard in this clip


@pytest.mark.db
def test_a_clip_gets_its_dominant_speakers_voice_and_that_voices_hours_in_train(
    db_session, imported_episode
) -> None:
    import sqlalchemy as sa

    from app.models import DiarizationRun, Episode, SpeakerTurn
    from app.services.clip_classes import classify_segments, load_clip_facts, voice_train_seconds

    episode = db_session.scalars(
        sa.select(Episode).where(Episode.external_id == imported_episode)
    ).one()
    episode.split = "train"
    gold, *train = sorted(episode.segments, key=lambda s: s.start_time)
    gold.pot = "gold"
    # A (voice v001) holds every train clip; B (v002) talks only in the gold clip.
    turns = [(gold.start_time, gold.end_time, "B")] + [
        (s.start_time, s.end_time, "A") for s in train
    ]
    db_session.add(
        DiarizationRun(
            episode_id=episode.id,
            model="test",
            checksum="voices",
            speakers_jsonb=["A", "B"],
            voices_jsonb={"A": "v001", "B": "v002"},
            turns=[SpeakerTurn(speaker=s, start_time=a, end_time=b) for a, b, s in turns],
        )
    )
    db_session.flush()

    exposure = voice_train_seconds(db_session)
    assert exposure == {"v001": pytest.approx(sum(s.duration_seconds for s in train))}

    facts = load_clip_facts(db_session, [gold])[gold.id]
    assert facts.voice == "v002" and facts.voice_train_seconds == 0.0
    classes = classify_segments(db_session, [gold, train[0]])
    assert classes[gold.id]["voice_exposure"] == "unseen"
    assert classes[train[0].id]["voice"] == "v001"

    episode.split = "val"  # a val episode is held out of training
    db_session.flush()
    assert voice_train_seconds(db_session) == {}
