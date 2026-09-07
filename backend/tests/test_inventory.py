"""Tests for the corpus inventory and the sourcing recommendations it produces (D69).

Most of these need no database. That is the point of the split: :func:`recommend_sources` reads
aggregates and returns advice, so what it says about a corpus can be tested by describing a corpus
in six lines rather than by building one. The database tests below check only that the aggregation
feeding it counts real rows correctly.

Two properties are worth more than the rest and are tested from several angles:

* **An episode is attributed once per value**, however many speakers carry that value, so no cell
  of the speaker matrix can hold more audio than the corpus does.
* **An unfilled metadata field is not a gap in the world.** An episode with no recorded gender
  must not be counted as evidence for or against any gender.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.config import Settings, load_settings
from app.services.inventory import (
    DOMINANT_SHARE,
    NARROW_SHOW_SPREAD,
    EpisodeFacts,
    build_dimension,
    build_length_profile,
    build_metadata_completeness,
    build_shows,
    build_speaker_matrix,
    collect_inventory,
    load_episode_facts,
    recommend_sources,
)
from tests.test_pots import make_episode


def fact(
    external_id: str,
    *,
    hours: float = 1.0,
    show_id: str | None = "show-a",
    topic: str | None = "technology",
    speakers: tuple[dict[str, str], ...] = (
        {"role": "host", "gender": "male", "age_bracket": "20_39"},
    ),
    segments: int = 10,
    labeled_hours: float | None = None,
    verified_hours: float = 0.0,
    mean_cmi: float | None = 0.25,
) -> EpisodeFacts:
    """One episode reduced to what the inventory counts, with workable defaults."""
    return EpisodeFacts(
        external_id=external_id,
        title=external_id,
        show_id=show_id,
        published_at="2026-01-01",
        pot="train",
        split="train",
        hours=hours,
        segments=segments,
        labeled_hours=hours if labeled_hours is None else labeled_hours,
        verified_hours=verified_hours,
        screened_hours=0.0,
        topic=topic,
        topic_source="llm",
        speakers=speakers,
        mean_cmi=mean_cmi,
        min_cmi=mean_cmi,
        max_cmi=mean_cmi,
    )


def recommend(facts, **overrides):
    """Run the recommender over ``facts`` with everything else neutral."""
    keys = ("show_id", "topic", "gender", "age_bracket", "role")
    kwargs = {
        "dimensions": {key: build_dimension(facts, key) for key in keys},
        "register": {"bands": [], "measured_hours": 0.0, "show_mean_spread": None},
        "length_profile": {"long_episode_share": 0.0, "long_episode_hours": 0.0},
        "gold_coverage_gaps": {},
        "corpus_hours": sum(f.hours for f in facts),
        "min_stratum_hours": 1.0,
    }
    kwargs.update(overrides)
    return recommend_sources(**kwargs)


def kinds(recommendations) -> list[str]:
    return [r.kind for r in recommendations]


# --- attribution -------------------------------------------------------------------------


def test_an_episode_counts_once_per_value_however_many_speakers_carry_it() -> None:
    """Three male speakers in one episode are one episode's worth of male audio, not three."""
    facts = [
        fact(
            "ep1",
            hours=2.0,
            speakers=(
                {"gender": "male", "age_bracket": "20_39"},
                {"gender": "male", "age_bracket": "20_39"},
                {"gender": "male", "age_bracket": "20_39"},
            ),
        )
    ]
    assert build_dimension(facts, "gender").values[0].hours == pytest.approx(2.0)
    assert build_speaker_matrix(facts)["male"]["20_39"] == pytest.approx(2.0)


def test_an_episode_with_two_genders_counts_toward_both() -> None:
    """No route diarizes, so per-speaker time does not exist; the episode counts on both sides."""
    facts = [
        fact(
            "ep1",
            hours=2.0,
            speakers=(
                {"gender": "male", "age_bracket": "20_39"},
                {"gender": "female", "age_bracket": "40_59"},
            ),
        )
    ]
    hours = {value.value: value.hours for value in build_dimension(facts, "gender").values}
    assert hours == {"male": pytest.approx(2.0), "female": pytest.approx(2.0)}


def test_no_matrix_cell_holds_more_audio_than_the_corpus() -> None:
    facts = [fact(f"ep{i}", hours=1.0) for i in range(4)]
    matrix = build_speaker_matrix(facts)
    corpus_hours = sum(f.hours for f in facts)
    assert max(h for ages in matrix.values() for h in ages.values()) <= corpus_hours


def test_the_matrix_keeps_its_empty_cells() -> None:
    """A blank square is the finding. Omitting it makes the frontend guess at the vocabulary."""
    matrix = build_speaker_matrix([fact("ep1")])
    assert matrix["female"]["60_79"] == 0.0
    assert set(matrix) == {"male", "female", "non_binary", "other"}


def test_an_episode_with_no_recorded_gender_is_unknown_not_evidence() -> None:
    facts = [fact("ep1", hours=1.0), fact("ep2", hours=3.0, speakers=())]
    dimension = build_dimension(facts, "gender")
    assert dimension.unknown_episodes == 1
    assert dimension.unknown_hours == pytest.approx(3.0)
    assert dimension.values[0].hours == pytest.approx(1.0)


# --- what the vocabulary says is missing --------------------------------------------------


def test_a_closed_vocabulary_reports_what_is_absent() -> None:
    dimension = build_dimension([fact("ep1")], "gender")
    assert dimension.absent == ["female", "non_binary", "other"]


def test_an_off_taxonomy_topic_is_dirt_not_a_gap() -> None:
    """A free-text topic typed into the form cannot be stratified against, so it is named (D57)."""
    dimension = build_dimension([fact("ep1", topic="phone_review")], "topic")
    assert dimension.off_vocabulary == ["phone_review"]
    assert "phone_review" not in dimension.absent


def test_an_open_dimension_has_nothing_absent() -> None:
    """There is no list of every Nepali YouTube channel, so show_id cannot report absences."""
    dimension = build_dimension([fact("ep1")], "show_id")
    assert dimension.absent == []
    assert dimension.off_vocabulary == []


def test_a_value_carried_by_one_show_is_counted_as_one_show() -> None:
    facts = [fact("ep1", show_id="a"), fact("ep2", show_id="a"), fact("ep3", show_id="b")]
    assert build_dimension(facts, "gender").values[0].shows == 2


# --- recommendations ---------------------------------------------------------------------


def test_an_absent_speaker_stratum_is_asked_for_first() -> None:
    top = recommend([fact("ep1")])[0]
    assert top.kind == "gender"
    assert top.priority == 1.0
    assert top.hours_present == 0.0
    assert "any show, any topic" in top.target


def test_a_thin_stratum_ranks_below_an_absent_one_and_says_how_short_it_is() -> None:
    facts = [
        fact("ep1", hours=5.0),
        fact("ep2", hours=0.25, speakers=({"gender": "female", "age_bracket": "20_39"},)),
    ]
    thin = [r for r in recommend(facts) if "more" in r.target and "female" in r.target]
    assert thin, "a stratum under the floor should be asked for"
    assert thin[0].hours_present == pytest.approx(0.25)
    assert thin[0].hours_needed == pytest.approx(0.75)
    assert thin[0].priority < 1.0


def test_a_stratum_over_the_floor_is_not_asked_for() -> None:
    facts = [
        fact("ep1", hours=2.0),
        fact("ep2", hours=2.0, speakers=({"gender": "female", "age_bracket": "40_59"},)),
    ]
    targets = [r.target for r in recommend(facts)]
    assert not any(t.startswith("more speakers female") for t in targets)


def test_a_dominant_value_is_reported_even_though_nothing_is_absent() -> None:
    """Absence is not the only way a stratum fails: five-sixths in one value supports no
    comparison across it, and a rule that only asks "does this value exist" cannot see that."""
    facts = [fact(f"m{i}", hours=1.0) for i in range(5)] + [
        fact("f1", hours=1.0, speakers=({"gender": "female", "age_bracket": "40_59"},))
    ]
    dimension = build_dimension(facts, "gender")
    assert dimension.top_share > DOMINANT_SHARE
    assert any("not male" in r.target for r in recommend(facts))


def test_a_balanced_dimension_produces_no_imbalance_recommendation() -> None:
    facts = [fact(f"m{i}", hours=1.0) for i in range(3)] + [
        fact(f"f{i}", hours=1.0, speakers=({"gender": "female", "age_bracket": "40_59"},))
        for i in range(3)
    ]
    assert not any("not male" in r.target for r in recommend(facts))


def test_a_missing_register_pole_is_asked_for_in_words_not_numbers() -> None:
    register = {
        "measured_hours": 10.0,
        "show_mean_spread": None,
        "bands": [
            {
                "name": "low",
                "lower": 0.0,
                "upper": 0.1,
                "description": "little English",
                "hours": 0.0,
                "segments": 0,
            },
            {
                "name": "mid",
                "lower": 0.1,
                "upper": 0.25,
                "description": "moderate",
                "hours": 6.0,
                "segments": 5,
            },
            {
                "name": "high",
                "lower": 0.25,
                "upper": 1.0,
                "description": "heavy",
                "hours": 4.0,
                "segments": 5,
            },
        ],
    }
    found = [r for r in recommend([fact("ep1")], register=register) if r.kind == "register"]
    assert len(found) == 1
    assert "little English" in found[0].target


def test_shows_bunched_together_are_reported_as_no_variance_to_explain() -> None:
    """Clip-level density spans the whole range inside one show. Show means are what vary by
    speaker, so a narrow spread across them is the finding a wide histogram would hide."""
    register = {
        "measured_hours": 10.0,
        "bands": [],
        "show_means": [{"show_id": "a"}, {"show_id": "b"}],
        "show_mean_min": 0.20,
        "show_mean_max": 0.20 + NARROW_SHOW_SPREAD / 2,
        "show_mean_spread": NARROW_SHOW_SPREAD / 2,
    }
    found = [r for r in recommend([fact("ep1")], register=register) if r.kind == "register_spread"]
    assert len(found) == 1
    assert found[0].priority > 0


def test_a_wide_spread_across_shows_is_not_a_gap() -> None:
    register = {
        "measured_hours": 10.0,
        "bands": [],
        "show_means": [{"show_id": "a"}, {"show_id": "b"}],
        "show_mean_min": 0.05,
        "show_mean_max": 0.60,
        "show_mean_spread": 0.55,
    }
    assert "register_spread" not in kinds(recommend([fact("ep1")], register=register))


def test_one_show_holding_most_of_the_corpus_is_reported_with_the_hours_that_would_fix_it() -> None:
    facts = [fact("big", hours=8.0, show_id="a"), fact("small", hours=2.0, show_id="b")]
    found = [r for r in recommend(facts) if r.kind == "show_concentration"]
    assert len(found) == 1
    # 8 h at 50% needs 16 h total, so 6 more hours from anything else.
    assert found[0].hours_needed == pytest.approx(6.0)


def test_a_spread_corpus_is_not_reported_as_concentrated() -> None:
    facts = [fact(f"ep{i}", hours=1.0, show_id=f"show-{i}") for i in range(5)]
    assert "show_concentration" not in kinds(recommend(facts))


def test_long_episodes_are_a_sourcing_habit_worth_naming() -> None:
    profile = {"long_episode_share": 0.8, "long_episode_hours": 8.0, "median_minutes": 90.0}
    found = [
        r for r in recommend([fact("ep1")], length_profile=profile) if r.kind == "episode_length"
    ]
    assert len(found) == 1
    assert "5-20 minute" in found[0].target


def test_gold_coverage_gaps_arrive_as_recommendations() -> None:
    gaps = {"show_id": ["b", "c"], "topic": ["comedy"]}
    found = [
        r for r in recommend([fact("ep1")], gold_coverage_gaps=gaps) if r.kind == "gold_coverage"
    ]
    assert len(found) == 2


def test_missing_topics_are_listed_not_enumerated() -> None:
    """Sixteen topic labels would otherwise bury every recommendation that matters."""
    found = [r for r in recommend([fact("ep1")]) if r.kind == "topic"]
    assert len(found) == 3


def test_recommendations_come_back_highest_priority_first() -> None:
    priorities = [r.priority for r in recommend([fact("ep1", hours=8.0, show_id="a")])]
    assert priorities == sorted(priorities, reverse=True)


def test_a_complete_corpus_asks_for_nothing_it_already_has() -> None:
    """Every vocabulary value present, no dominance, no gaps: silence on those dimensions."""
    facts = []
    for gender in ("male", "female", "non_binary", "other"):
        for age in ("under_20", "20_39", "40_59", "60_79", "80_plus"):
            facts.append(
                fact(
                    f"{gender}_{age}",
                    hours=1.0,
                    show_id=f"show-{gender}-{age}",
                    speakers=({"role": "host", "gender": gender, "age_bracket": age},),
                )
            )
    produced = kinds(recommend(facts))
    assert "gender" not in produced
    assert "age_bracket" not in produced
    assert "show_concentration" not in produced


# --- the supporting tables ----------------------------------------------------------------


def test_length_profile_separates_short_videos_from_long_podcasts() -> None:
    facts = [fact("short", hours=0.2), fact("long", hours=2.0)]
    profile = build_length_profile(facts)
    assert profile["long_episode_share"] == pytest.approx(2.0 / 2.2, abs=1e-3)
    assert {b["name"]: b["episodes"] for b in profile["buckets"]}["5_20m"] == 1


def test_metadata_completeness_names_the_episodes_to_go_and_fix() -> None:
    facts = [fact("good"), fact("bad", topic=None, speakers=())]
    rows = {row["field"]: row for row in build_metadata_completeness(facts)}
    assert rows["topic"]["missing_episodes"] == ["bad"]
    assert rows["gender"]["filled"] == 1
    assert rows["topic"]["fraction"] == pytest.approx(0.5)


def test_an_off_taxonomy_topic_counts_as_filled_but_not_as_in_taxonomy() -> None:
    rows = {
        row["field"]: row for row in build_metadata_completeness([fact("ep1", topic="cooking")])
    }
    assert rows["topic"]["filled"] == 1
    assert rows["topic_in_taxonomy"]["filled"] == 0


def test_a_show_rollup_carries_the_cmi_range_a_sourcing_decision_needs() -> None:
    facts = [
        fact("a1", show_id="a", hours=1.0, mean_cmi=0.1),
        fact("a2", show_id="a", hours=3.0, mean_cmi=0.5),
    ]
    row = build_shows(facts)[0]
    assert row["episodes"] == 2
    # Hours-weighted: three hours at 0.5 outweighs one at 0.1.
    assert row["mean_cmi"] == pytest.approx(0.4)


def test_shows_come_back_largest_first() -> None:
    facts = [fact("a", show_id="small", hours=1.0), fact("b", show_id="big", hours=5.0)]
    assert [row["show_id"] for row in build_shows(facts)] == ["big", "small"]


# --- against a real database --------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.mark.db
def test_facts_take_their_hours_from_segments_not_the_episode_column(db_session: Session) -> None:
    """Budgeting against the raw recording would over-count the silence VAD already removed."""
    episode = make_episode(db_session, "ep1", minutes=10, segment_seconds=60.0)
    episode.duration_seconds = 3600.0  # a much longer raw recording
    db_session.flush()

    facts = load_episode_facts(db_session)
    assert facts[0].hours == pytest.approx(10 / 60, abs=1e-6)
    assert facts[0].segments == 10


@pytest.mark.db
def test_the_inventory_counts_a_real_corpus(db_session: Session, settings: Settings) -> None:
    make_episode(db_session, "ep1", minutes=30, show_id="a", gender="male", age_bracket="20_39")
    make_episode(db_session, "ep2", minutes=30, show_id="b", gender="female", age_bracket="40_59")

    inventory = collect_inventory(db_session, settings=settings).as_dict()

    assert inventory["totals"]["episodes"] == 2
    assert inventory["totals"]["shows"] == 2
    assert inventory["totals"]["hours"] == pytest.approx(1.0, abs=0.01)
    assert inventory["totals"]["speaker_profiles"] == 2
    assert inventory["speaker_matrix"]["male"]["20_39"] == pytest.approx(0.5, abs=0.01)
    assert inventory["speaker_matrix"]["female"]["40_59"] == pytest.approx(0.5, abs=0.01)


@pytest.mark.db
def test_the_inventory_reads_and_never_writes(db_session: Session, settings: Settings) -> None:
    """Opening the analytics page must not move an episode between pots (D63).

    `pot_status` is a read and `collect_inventory` calls it, but a page that assigned as a side
    effect of being looked at would be a trap, and the failure would be silent.
    """
    make_episode(db_session, "ep1", minutes=30, show_id="a")
    make_episode(db_session, "ep2", minutes=30, show_id="b")

    collect_inventory(db_session, settings=settings)

    assert [f.pot for f in load_episode_facts(db_session)] == ["unassigned", "unassigned"]
