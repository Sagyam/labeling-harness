"""Tests for the corpus page: clips cut by category, voices followed across episodes, and the
advice both produce (D91).

Almost everything here needs no database. The package is a pure pipeline after ``load_rows`` --
resolve, bucket, total, profile, advise -- so a corpus is described in a few lines of
:func:`clip` and :func:`episode` calls. The database tests check only that the loader reads real
rows into that pipeline and that reading never writes.

Two properties carry the design and are tested from several angles:

* **Every category's hours sum to the corpus.** A clip has one bucket per category, so no cut
  can hold more audio than exists -- the promise D69's episode attribution could not make.
* **A declared value reaches a clip only when the episode forces it.** A voice takes a row by
  one of three rules or not at all; nothing is guessed from audio.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import AuditLog
from app.services.inventory import (
    CATEGORIES,
    ClipRow,
    EpisodeRow,
    assemble,
    assign_buckets,
    attributed_words,
    build_categories,
    build_voices,
    clip_table,
    collect_inventory,
    recommend,
    resolve_corpus,
    resolve_episode,
    speaking_rate_bucket,
    summarize_voices,
)
from app.services.inventory.constants import MIN_VOICE_WORDS
from app.services.inventory.voices import host_guest_episodes

# --- factories -----------------------------------------------------------------------------

CLEAN_CLASSES: dict[str, str] = {
    "overlap": "none",
    "speakers": "1",
    "turn_changes": "0",
    "pause": "<2%",
    "duration": "5-15 s",
    "bandwidth": "6.5+ kHz",
    "snr": "45+ dB",
    "reverb": "55+ dB",
    "voice_exposure": "10-60 min",
    "voice": "unlinked",
    "gender": "undeclared",
    "age_bracket": "undeclared",
    "cmi": "15-30",
}


def episode(
    external_id: str,
    *,
    show_id: str | None = "show-a",
    genre: str | None = "podcast",
    topic: str | None = "technology",
    declared: tuple[Mapping[str, str], ...] = (),
    voices: tuple[str, ...] = (),
    split: str = "train",
    published_at: str | None = "2026-01-01",
) -> EpisodeRow:
    return EpisodeRow(
        external_id=external_id,
        title=external_id,
        show_id=show_id,
        genre=genre,
        topic=topic,
        published_at=published_at,
        split=split,
        declared=declared,
        diarized=bool(voices),
        voices=voices,
    )


_ids = iter(range(1, 1_000_000))


def clip(
    ep: str,
    *,
    seconds: float = 10.0,
    voice: str | None = None,
    talk: Mapping[str, float] | None = None,
    words: int | None = 30,
    tier: str | None = "verified",
    pot: str = "train",
    cmi: float | None = 20.0,
    speech: float | None = None,
    **classes: str,
) -> ClipRow:
    """One clip with workable defaults: ten clean seconds of one voice, verified, in train."""
    if talk is None:
        talk = {voice: seconds} if voice else {}
    return ClipRow(
        segment_id=next(_ids),
        episode=ep,
        duration=seconds,
        speech_seconds=seconds if speech is None else speech,
        words=words,
        text_source="label" if words is not None else None,
        tier=tier,
        pot=pot,
        voice=voice,
        voice_talk=talk,
        cmi=cmi,
        classes={**CLEAN_CLASSES, "voice": voice or "unlinked", **classes},
    )


def cut(episodes, clips):
    """Resolve and bucket, the way ``assemble`` does, and hand back what the tests read."""
    by_id = {e.external_id: e for e in episodes}
    per_episode, per_voice = resolve_corpus(episodes)
    rows = assign_buckets(clips, by_id, per_episode, per_voice)
    return rows, build_categories(rows, by_id), per_episode, per_voice


def buckets_of(rows, key: str) -> list[str]:
    return [r.buckets[key] for r in rows]


def advise(episodes, clips, *, hours: float = 1.0, voices: int = 5):
    rows, categories, per_episode, per_voice = cut(episodes, clips)
    by_id = {e.external_id: e for e in episodes}
    summary = summarize_voices(build_voices(rows, by_id, per_episode, per_voice))
    return recommend(categories, summary, min_stratum_hours=hours, min_stratum_voices=voices)


def kinds(recs, category: str | None = None) -> list[tuple[str, str | None]]:
    return [(r.kind, r.bucket) for r in recs if category is None or r.category == category]


# --- resolving rows to voices -----------------------------------------------------------------


def test_one_row_and_one_voice_is_the_row() -> None:
    out = resolve_episode(
        ({"role": "host", "gender": "female", "age_bracket": "20_39"},), ("v1",), ()
    )
    assert out["v1"].gender == "female"
    assert out["v1"].age_bracket == "20_39"
    assert out["v1"].role == "host"
    assert out["v1"].by == "only_pair"


def test_the_one_recurring_voice_is_the_one_declared_host() -> None:
    rows = (
        {"role": "host", "gender": "male", "age_bracket": "40_59"},
        {"role": "guest", "gender": "female", "age_bracket": "20_39"},
    )
    out = resolve_episode(rows, ("v9", "v1"), recurring={"v1"})
    assert out["v1"].role == "host" and out["v1"].gender == "male"
    assert out["v1"].by == "recurring_host"
    # The one row left is the one voice left: the guest resolves too.
    assert out["v9"].gender == "female" and out["v9"].by == "rows_agree"


def test_remaining_rows_that_agree_resolve_field_by_field() -> None:
    rows = (
        {"role": "guest", "gender": "male", "age_bracket": "40_59"},
        {"role": "guest", "gender": "female", "age_bracket": "40_59"},
    )
    out = resolve_episode(rows, ("v1", "v2"), ())
    assert {v.age_bracket for v in out.values()} == {"40_59"}
    assert {v.gender for v in out.values()} == {None}
    assert {v.role for v in out.values()} == {"guest"}


def test_more_voices_than_rows_is_refused_rather_than_guessed() -> None:
    """A voice nobody declared must not inherit the declared rows' values."""
    rows = ({"gender": "male"}, {"gender": "male"})
    assert resolve_episode(rows, ("v1", "v2", "v3"), ()) == {}


def test_two_hosts_or_two_recurring_voices_force_no_person() -> None:
    """Two co-hosts: both voices are hosts, but which is the man is nobody's guess."""
    rows = ({"role": "host", "gender": "male"}, {"role": "host", "gender": "female"})
    out = resolve_episode(rows, ("v1", "v2"), recurring={"v1"})
    assert {v.role for v in out.values()} == {"host"}
    assert {v.gender for v in out.values()} == {None}
    rows = ({"role": "host", "gender": "male"}, {"role": "guest", "gender": "female"})
    assert resolve_episode(rows, ("v1", "v2"), recurring={"v1", "v2"}) == {}


def test_nothing_resolves_without_rows_or_without_voices() -> None:
    assert resolve_episode((), ("v1",), ()) == {}
    assert resolve_episode(({"gender": "male"},), (), ()) == {}


def test_a_resolved_voice_carries_its_gender_to_every_episode_it_appears_in() -> None:
    """The host resolves in ep1 alone; ep2 cannot force anything, but the voice is known."""
    episodes = [
        episode("ep1", declared=({"role": "host", "gender": "female"},), voices=("v1",)),
        episode(
            "ep2",
            declared=(
                {"role": "host", "gender": "female"},
                {"role": "guest", "gender": "male"},
                {"role": "guest", "gender": "female"},
            ),
            voices=("v1", "v2", "v3"),
        ),
    ]
    per_episode, per_voice = resolve_corpus(episodes)
    assert per_voice["v1"].gender == "female" and per_voice["v1"].status == "resolved"
    assert per_voice["v2"].status == "unresolved"
    # In ep2 the recurring-host rule reaches v1 even though rows_agree cannot reach the guests.
    assert per_episode[("ep2", "v1")].by == "recurring_host"
    # The two guests agree on role and nothing else.
    assert per_episode[("ep2", "v2")].role == "guest"
    assert per_episode[("ep2", "v2")].gender is None


def test_episodes_that_disagree_about_a_voice_make_it_a_conflict_not_an_average() -> None:
    episodes = [
        episode("ep1", declared=({"gender": "female"},), voices=("v1",)),
        episode("ep2", declared=({"gender": "male"},), voices=("v1",)),
    ]
    _, per_voice = resolve_corpus(episodes)
    assert per_voice["v1"].status == "conflict"
    assert per_voice["v1"].gender is None


# --- bucketing clips --------------------------------------------------------------------------


def test_a_clip_takes_its_voices_resolved_gender_over_the_episodes_agreement() -> None:
    episodes = [
        episode("ep1", declared=({"role": "host", "gender": "female"},), voices=("v1",)),
        episode(
            "ep2",
            declared=({"role": "host", "gender": "female"}, {"role": "guest", "gender": "male"}),
            voices=("v1", "v2"),
        ),
    ]
    clips = [clip("ep2", voice="v1", gender="mixed"), clip("ep2", voice="v2", gender="mixed")]
    rows, *_ = cut(episodes, clips)
    assert buckets_of(rows, "gender") == ["female", "male"]
    assert buckets_of(rows, "role") == ["host", "guest"]


def test_an_unlinked_clip_falls_back_to_the_whole_episodes_declared_value() -> None:
    """D87's rule still holds where no voice was linked: the value reaches the clip only when
    every declared speaker shares it."""
    episodes = [episode("ep1", declared=({"gender": "male"}, {"gender": "male"}))]
    rows, *_ = cut(episodes, [clip("ep1", gender="male"), clip("ep1", gender="mixed")])
    assert buckets_of(rows, "gender") == ["male", "unresolved"]


def test_unresolved_and_undeclared_are_different_kinds_of_unknown() -> None:
    episodes = [
        episode(
            "declared", declared=({"gender": "male"}, {"gender": "female"}), voices=("v1", "v2")
        ),
        episode("silent"),
    ]
    rows, *_ = cut(episodes, [clip("declared", voice="v1", gender="mixed"), clip("silent")])
    assert buckets_of(rows, "gender") == ["unresolved", "undeclared"]
    assert buckets_of(rows, "age_bracket") == ["unresolved", "undeclared"]
    assert buckets_of(rows, "role") == ["unresolved", "undeclared"]


@pytest.mark.parametrize(
    ("rate", "bucket"),
    [
        (None, "unmeasured"),
        (1.9, "<2.0 w/s"),
        (2.0, "2.0-2.6 w/s"),
        (2.9, "2.6-3.2 w/s"),
        (3.5, "3.2-3.8 w/s"),
        (4.4, "3.8+ w/s"),
    ],
)
def test_speaking_rate_buckets(rate, bucket: str) -> None:
    assert speaking_rate_bucket(rate) == bucket


def test_speaking_rate_is_words_per_second_of_speech_not_of_clip() -> None:
    """Thirty words in ten seconds of clip, of which only six are speech: fast talk."""
    fast = clip("ep1", seconds=10.0, speech=6.0, words=30)
    assert fast.words_per_second == pytest.approx(5.0)
    silent = clip("ep1", words=None)
    assert silent.words_per_second is None
    rows, *_ = cut([episode("ep1")], [fast, silent])
    assert buckets_of(rows, "speaking_rate") == ["3.8+ w/s", "unmeasured"]


def test_content_buckets_come_from_the_episode_and_conditions_from_the_classes() -> None:
    episodes = [episode("ep1", topic=None, genre=None, show_id=None)]
    row = cut(episodes, [clip("ep1", overlap=">15%", snr="<15 dB", duration="<5 s")])[0][0]
    assert row.buckets["topic"] == "untagged"
    assert row.buckets["genre"] == "untagged"
    assert row.buckets["show"] == "no show id"
    assert row.buckets["crosstalk"] == ">15%"
    assert row.buckets["noise"] == "<15 dB"
    assert row.buckets["duration"] == "<5 s"


# --- totalling ----------------------------------------------------------------------------------


def test_every_category_sums_to_the_corpus_and_shares_sum_to_one() -> None:
    episodes = [
        episode("ep1", declared=({"role": "host", "gender": "male"},), voices=("v1",)),
        episode("ep2", show_id="show-b", topic=None),
    ]
    clips = [
        clip("ep1", seconds=90, voice="v1"),
        clip("ep1", seconds=30, voice="v1", snr="<15 dB"),
        clip("ep2", seconds=60, tier=None, pot="gold"),
    ]
    _, categories, *_ = cut(episodes, clips)
    corpus = 180 / 3600
    for report in categories.values():
        assert sum(b.hours for b in report.buckets) == pytest.approx(corpus, abs=1e-3), report.key
        measured = [b for b in report.buckets if b.bucket not in report.unknown and b.hours]
        if measured:
            assert sum(b.share for b in measured) == pytest.approx(1.0, abs=1e-3), report.key


def test_a_bucket_carries_its_tiers_and_pots_separately() -> None:
    clips = [
        clip("ep1", seconds=3600, tier="verified", pot="gold"),
        clip("ep1", seconds=1800, tier="screened", pot="train"),
        clip("ep1", seconds=900, tier=None, pot="val"),
    ]
    _, categories, *_ = cut([episode("ep1")], clips)
    entry = categories["topic"].stats("technology")
    assert entry is not None
    assert (entry.hours, entry.verified_hours, entry.screened_hours) == (1.75, 1.0, 0.5)
    assert (entry.gold_hours, entry.train_hours, entry.val_hours) == (1.0, 0.5, 0.25)
    assert entry.clips == 3 and entry.episodes == 1 and entry.shows == 1


def test_a_voice_counts_toward_a_bucket_only_with_enough_words_inside_it() -> None:
    episodes = [episode("ep1", voices=("v1", "v2"))]
    clips = [
        clip("ep1", voice="v1", words=MIN_VOICE_WORDS, snr="<15 dB"),
        clip("ep1", voice="v2", words=MIN_VOICE_WORDS - 1, snr="<15 dB"),
        clip("ep1", voice="v2", words=MIN_VOICE_WORDS, snr="45+ dB"),
    ]
    _, categories, *_ = cut(episodes, clips)
    noisy = categories["noise"].stats("<15 dB")
    assert noisy is not None
    assert noisy.voices == 2
    assert noisy.usable_voices == 1  # v2 is usable in the corpus, not in this bucket


def test_a_shared_clips_words_are_credited_to_nobody() -> None:
    """Two people in one clip: the words go to the dominant voice only when it holds 90%."""
    solo = clip("ep1", voice="v1", talk={"v1": 9.5, "v2": 0.5}, words=100)
    shared = clip("ep1", voice="v1", talk={"v1": 6.0, "v2": 4.0}, words=100)
    assert attributed_words(solo) == 100
    assert attributed_words(shared) == 0
    assert attributed_words(clip("ep1", words=100)) == 0  # no voice at all


def test_a_closed_category_lists_what_is_absent_and_keeps_its_empty_buckets() -> None:
    episodes = [
        episode("ep1", declared=({"gender": "male", "age_bracket": "20_39"},), voices=("v1",))
    ]
    _, categories, *_ = cut(episodes, [clip("ep1", voice="v1")])
    age = categories["age_bracket"]
    assert age.absent == ["under_20", "40_59", "60_79", "80_plus"]
    assert [b.bucket for b in age.buckets] == [
        "under_20", "20_39", "40_59", "60_79", "80_plus", "unresolved", "undeclared",
    ]  # fmt: skip
    assert age.unknown == ["unresolved", "undeclared"]
    assert categories["gender"].absent == ["female"]


def test_an_open_category_orders_by_hours_with_the_unknown_last() -> None:
    episodes = [
        episode("a", show_id="small"),
        episode("b", show_id="big"),
        episode("c", show_id=None),
    ]
    clips = [clip("a", seconds=10), clip("b", seconds=100), clip("c", seconds=1000)]
    _, categories, *_ = cut(episodes, clips)
    assert [b.bucket for b in categories["show"].buckets] == ["big", "small", "no show id"]
    assert categories["show"].absent == []
    assert categories["show"].top_bucket == "big"
    assert categories["show"].top_share == pytest.approx(100 / 110, abs=1e-3)


def test_an_off_taxonomy_topic_is_flagged_as_dirt_and_sits_before_untagged() -> None:
    episodes = [episode("a", topic="cooking"), episode("b", topic=None)]
    _, categories, *_ = cut(episodes, [clip("a"), clip("b")])
    order = [b.bucket for b in categories["topic"].buckets]
    assert order.index("cooking") < order.index("untagged")
    assert categories["topic"].stats("cooking").off_vocabulary is True
    assert categories["topic"].stats("technology").off_vocabulary is False


def test_the_clip_table_round_trips_to_the_category_totals() -> None:
    """The page re-cuts the table client-side; its sums must be the server's sums."""
    episodes = [episode("ep1", voices=("v1",)), episode("ep2", show_id="show-b")]
    clips = [
        clip("ep1", seconds=20, voice="v1", snr="<15 dB"),
        clip("ep1", seconds=40, voice="v1"),
        clip("ep2", seconds=60, pot="gold", tier="screened"),
    ]
    rows, categories, *_ = cut(episodes, clips)
    table = clip_table(rows, categories, {"ep1": 0, "ep2": 1}, {"v1": 0})
    assert len(table["rows"]) == 3
    columns = table["columns"]
    for key, report in categories.items():
        column = columns.index(key)
        sums = [0.0] * len(report.buckets)
        for row in table["rows"]:
            sums[row[column]] += row[columns.index("seconds")] / 3600
        assert sums == pytest.approx([b.hours for b in report.buckets], abs=1e-3), key
    assert table["buckets"]["noise"] == [b.bucket for b in categories["noise"].buckets]
    voice_column, pot_column, tier_column = (columns.index(c) for c in ("voice", "pot", "tier"))
    assert [r[voice_column] for r in table["rows"]] == [0, 0, -1]
    assert table["pots"][table["rows"][2][pot_column]] == "gold"
    assert table["tiers"][table["rows"][2][tier_column]] == "screened"


# --- voices ---------------------------------------------------------------------------------


def test_a_voice_is_followed_across_episodes_with_its_talk_share_and_company() -> None:
    episodes = [
        episode("ep1", declared=({"role": "host", "gender": "female"},), voices=("v1",),
                published_at="2026-02-01"),
        episode("ep2", show_id="show-b",
                declared=({"role": "host", "gender": "female"},
                          {"role": "guest", "gender": "male"}),
                voices=("v1", "v2"), published_at="2026-01-01"),
    ]  # fmt: skip
    clips = [
        clip("ep1", seconds=60, voice="v1", words=400, pot="val"),
        clip("ep2", seconds=30, voice="v1", talk={"v1": 25.0, "v2": 5.0}, words=100),
        clip("ep2", seconds=30, voice="v2", talk={"v2": 30.0}, words=200, tier="screened"),
    ]
    rows, _, per_episode, per_voice = cut(episodes, clips)
    by_id = {e.external_id: e for e in episodes}
    profiles = {p.voice: p for p in build_voices(rows, by_id, per_episode, per_voice)}
    host = profiles["v1"]
    assert host.talk_minutes == pytest.approx(85 / 60, abs=0.01)
    assert host.clips == 2 and host.episode_count == 2
    assert host.words == 400  # the ep2 clip is shared 25/30 < 90%, so its words go to nobody
    assert host.usable is True
    assert host.gender == "female" and host.identity == "resolved"
    assert host.roles == {"host": 2}
    assert host.shows == ["show-a", "show-b"]
    assert [e.external_id for e in host.episodes] == ["ep2", "ep1"]  # by date
    assert host.episodes[0].share == pytest.approx(25 / 60, abs=1e-3)
    assert host.episodes[0].with_voices == ["v2"]
    assert host.co_voices == {"v2": 1}
    assert host.hours == pytest.approx(
        {"gold": 0.0, "train": 30 / 3600, "val": 60 / 3600}, abs=1e-3
    )
    assert (host.first_seen, host.last_seen) == ("2026-01-01", "2026-02-01")
    guest = profiles["v2"]
    assert guest.gender == "male" and guest.roles == {"guest": 1}
    assert guest.usable is False and guest.screened_minutes == 0.5


def test_a_recurring_host_needs_guests_in_the_room() -> None:
    """A monologue host in twenty episodes accommodates nobody."""
    solo = [episode(f"m{i}", declared=({"role": "host"},), voices=("v1",)) for i in range(6)]
    paired = [
        episode(f"p{i}", declared=({"role": "host"}, {"role": "guest"}), voices=("v2", f"g{i}"))
        for i in range(5)
    ]
    episodes = solo + paired
    clips = [clip(e.external_id, voice=e.voices[0]) for e in episodes] + [
        clip(e.external_id, voice=e.voices[1]) for e in paired
    ]
    rows, _, per_episode, per_voice = cut(episodes, clips)
    by_id = {e.external_id: e for e in episodes}
    profiles = {p.voice: p for p in build_voices(rows, by_id, per_episode, per_voice)}
    assert host_guest_episodes(profiles["v1"]) == 0
    assert host_guest_episodes(profiles["v2"]) == 5
    summary = summarize_voices(list(profiles.values()))
    assert summary["recurring_hosts"] == 1
    assert summary["recurring_host_voices"] == ["v2"]
    assert summary["recurring"] == 2
    assert summary["single_episode"] == 5


# --- advice -----------------------------------------------------------------------------------


def _people_corpus(*, female_voices: int, shows: int = 2, verified: bool = True):
    """``female_voices`` female guests spread over ``shows`` shows, plus one male host."""
    episodes, clips = [], []
    for i in range(female_voices):
        show = f"show-{i % shows}"
        ep = f"ep{i}"
        episodes.append(
            episode(ep, show_id=show, voices=(f"g{i}", "h"),
                    declared=({"role": "guest", "gender": "female", "age_bracket": "20_39"},
                              {"role": "host", "gender": "male", "age_bracket": "20_39"}))
        )  # fmt: skip
        clips.append(clip(ep, seconds=1200, voice=f"g{i}", words=500,
                          tier="verified" if verified else "screened"))  # fmt: skip
        clips.append(clip(ep, seconds=600, voice="h", words=500))
    return episodes, clips


def test_an_absent_bucket_is_asked_for_first_and_a_thin_one_says_how_short_it_is() -> None:
    episodes, clips = _people_corpus(female_voices=2)
    recs = advise(episodes, clips)
    gender = [r for r in recs if r.category == "gender"]
    assert kinds(gender)[0][0] in ("thin", "dominant")
    thin = next(r for r in gender if r.kind == "thin" and r.bucket == "female")
    assert thin.voices_present == 2 and thin.voices_needed == 3
    assert thin.goal == "both"
    assert "2 voice(s)" in thin.reason
    age = [r for r in recs if r.category == "age_bracket"]
    absent = [r for r in age if r.kind == "absent"]
    assert {r.bucket for r in absent} == {"under_20", "40_59", "60_79", "80_plus"}
    assert absent[0].priority > thin.priority


def test_a_people_bucket_with_enough_voices_is_not_asked_for() -> None:
    episodes, clips = _people_corpus(female_voices=6)
    recs = advise(episodes, clips)
    assert ("thin", "female") not in kinds(recs, "gender")


def test_a_condition_bucket_is_thin_in_hours_and_needs_gold() -> None:
    clips = [clip("ep1", seconds=1800, snr="<15 dB"), clip("ep1", seconds=7200, pot="gold")]
    recs = advise([episode("ep1")], clips)
    noise = {(r.kind, r.bucket): r for r in recs if r.category == "noise"}
    assert noise[("thin", "<15 dB")].hours_needed == pytest.approx(0.5)
    assert noise[("thin", "<15 dB")].goal == "asr"
    # 45+ dB has gold; a thin bucket is asked for before it is asked for gold.
    assert ("no_gold", "45+ dB") not in noise
    clips = [clip("ep1", seconds=7200, snr="<15 dB"), clip("ep1", seconds=7200, pot="gold")]
    recs = advise([episode("ep1")], clips)
    assert ("no_gold", "<15 dB") in kinds(recs, "noise")


def test_gold_is_not_asked_for_on_people_or_content_buckets() -> None:
    """Gold holds voices train never sees (D76): a female bucket without gold is the design."""
    episodes, clips = _people_corpus(female_voices=6)
    recs = advise(episodes, clips)
    assert not any(r.kind == "no_gold" for r in recs if r.category in ("gender", "topic", "show"))


def test_a_paper_bucket_resting_on_screened_labels_is_asked_to_verify_a_sample() -> None:
    episodes, clips = _people_corpus(female_voices=6, verified=False)
    recs = advise(episodes, clips)
    unverified = next(r for r in recs if r.category == "gender" and r.kind == "unverified")
    assert unverified.bucket == "female" and unverified.goal == "paper"
    episodes, clips = _people_corpus(female_voices=6, verified=True)
    assert not any(r.kind == "unverified" for r in advise(episodes, clips) if r.bucket == "female")


def test_a_value_carried_by_one_show_is_reported_at_half_severity() -> None:
    episodes, clips = _people_corpus(female_voices=6, shows=1)
    recs = advise(episodes, clips)
    single = next(r for r in recs if r.category == "gender" and r.kind == "single_show")
    assert single.bucket == "female"
    episodes, clips = _people_corpus(female_voices=6, shows=2)
    assert ("single_show", "female") not in kinds(advise(episodes, clips), "gender")


def test_a_dominant_bucket_is_reported_even_though_nothing_is_absent() -> None:
    clips = [clip("ep1", seconds=9000, overlap="none"), clip("ep1", seconds=900, overlap=">15%")]
    recs = advise([episode("ep1")], clips)
    dominant = next(r for r in recs if r.category == "crosstalk" and r.kind == "dominant")
    assert dominant.bucket == "none" and "91%" in dominant.reason


def test_too_much_unmeasured_is_reported_as_paperwork_with_the_fix() -> None:
    clips = [clip("ep1", seconds=3600, snr="unmeasured"), clip("ep1", seconds=3600)]
    recs = advise([episode("ep1")], clips)
    unmeasured = next(r for r in recs if r.category == "noise" and r.kind == "unmeasured")
    assert "backfill_acoustics" in unmeasured.target and "50%" in unmeasured.reason
    clips = [clip("ep1", seconds=360, snr="unmeasured"), clip("ep1", seconds=3600)]
    assert not any(
        r.kind == "unmeasured" for r in advise([episode("ep1")], clips) if r.category == "noise"
    )


def test_shows_are_not_asked_for_more_speakers_but_a_dominant_show_is_reported() -> None:
    episodes = [episode("a", show_id="big"), episode("b", show_id="small")]
    clips = [clip("a", seconds=9000), clip("b", seconds=600)]
    recs = advise(episodes, clips)
    show = kinds(recs, "show")
    assert ("thin", "small") not in show
    assert ("dominant", "big") in show


def test_the_diarizers_empty_clip_is_a_defect_not_a_stratum() -> None:
    clips = [clip("ep1", seconds=5, speakers="none"), clip("ep1", seconds=7200)]
    recs = advise([episode("ep1")], clips)
    assert not any(r.bucket == "none" for r in recs if r.category == "speakers")


def test_voice_recurrence_advice_names_the_design_it_serves() -> None:
    episodes, clips = _people_corpus(female_voices=6)
    recs = [r for r in advise(episodes, clips) if r.category == "voice"]
    assert any(r.kind == "recurrence" and "host" in r.target and r.goal == "paper" for r in recs)
    assert any(r.kind == "recurrence" and "guest on more than one show" in r.target for r in recs)


def test_recommendations_come_back_highest_priority_first_with_numbers_in_every_reason() -> None:
    episodes, clips = _people_corpus(female_voices=2, shows=1)
    recs = advise(episodes, clips)
    assert recs
    assert [r.priority for r in recs] == sorted((r.priority for r in recs), reverse=True)
    assert all(any(ch.isdigit() for ch in r.reason) for r in recs)
    assert all(r.goal in ("asr", "paper", "both") for r in recs)


# --- assembling ----------------------------------------------------------------------------------


def test_assemble_produces_the_payload_the_page_reads() -> None:
    episodes = [episode("ep1", declared=({"role": "host", "gender": "female"},), voices=("v1",))]
    clips = [clip("ep1", seconds=60, voice="v1", words=400), clip("ep1", seconds=60, tier=None)]
    payload = assemble(
        episodes, clips, min_stratum_hours=1.0, min_stratum_voices=5, gold_target_hours=3.0
    ).as_dict()
    assert payload["totals"]["hours"] == pytest.approx(120 / 3600, abs=1e-3)
    assert payload["totals"]["unlabeled_hours"] == pytest.approx(60 / 3600, abs=1e-3)
    assert payload["totals"]["clips"] == 2 and payload["totals"]["episodes"] == 1
    assert [c["key"] for c in payload["categories"]] == [c.key for c in CATEGORIES]
    assert payload["voices"][0]["voice"] == "v1"
    assert payload["voice_summary"]["usable"] == 1
    assert {g["key"] for g in payload["groups"]} == {"people", "content", "speech", "acoustics"}
    assert len(payload["clips"]["rows"]) == 2
    assert {r["field"] for r in payload["records"]} >= {"genre", "topic", "speakers", "diarized"}
    assert payload["episodes"][0]["voices"] == ["v1"]


# --- database --------------------------------------------------------------------------------


@pytest.mark.db
def test_the_inventory_reads_a_real_corpus_into_clips_and_voices(
    db_session: Session, imported_episode: str, settings: Settings
) -> None:
    from app.models import DiarizationRun, Episode, Segment, SpeakerTurn

    ep = db_session.scalars(sa.select(Episode).where(Episode.external_id == imported_episode)).one()
    ep.metadata_jsonb = {
        **(ep.metadata_jsonb or {}),
        "genre": "podcast",
        "speakers": {"spk:0": {"role": "host", "gender": "female", "age_bracket": "40_59"}},
    }
    segments = sorted(ep.segments, key=lambda s: s.start_time)
    segments[0].pot = "gold"
    db_session.add(
        DiarizationRun(
            episode_id=ep.id,
            model="test",
            checksum="inventory",
            speakers_jsonb=["A"],
            voices_jsonb={"A": "v001"},
            turns=[
                SpeakerTurn(speaker="A", start_time=s.start_time, end_time=s.end_time)
                for s in segments
            ],
        )
    )
    db_session.flush()

    payload = collect_inventory(db_session, settings=settings).as_dict()
    seconds = db_session.scalar(sa.select(sa.func.sum(Segment.duration_seconds)))
    assert payload["totals"]["hours"] == pytest.approx(seconds / 3600, abs=1e-3)
    assert payload["totals"]["clips"] == len(segments)
    assert payload["totals"]["gold_hours"] == pytest.approx(
        segments[0].duration_seconds / 3600, abs=1e-3
    )
    assert len(payload["clips"]["rows"]) == len(segments)
    voice = payload["voices"][0]
    assert voice["voice"] == "v001" and voice["gender"] == "female"
    assert voice["resolved_by"] == {"only_pair": 1}
    gender = next(c for c in payload["categories"] if c["key"] == "gender")
    assert next(b for b in gender["buckets"] if b["bucket"] == "female")["clips"] == len(segments)
    # Words come from the fused seed for clips nobody has labelled.
    rate = next(c for c in payload["categories"] if c["key"] == "speaking_rate")
    assert sum(b["clips"] for b in rate["buckets"] if b["bucket"] != "unmeasured") == len(segments)


@pytest.mark.db
def test_the_inventory_reads_and_never_writes(
    db_session: Session, imported_episode: str, settings: Settings
) -> None:
    before = db_session.scalar(sa.select(sa.func.count()).select_from(AuditLog))
    collect_inventory(db_session, settings=settings)
    db_session.flush()
    assert db_session.scalar(sa.select(sa.func.count()).select_from(AuditLog)) == before


@pytest.mark.db
def test_the_inventory_endpoint_serves_the_payload(client, imported_episode: str) -> None:
    response = client.get("/stats/inventory")
    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["clips"] == 6
    assert "clips" in body and "voices" in body and "recommendations" in body
