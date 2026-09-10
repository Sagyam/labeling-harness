"""Tests for the ingest's fusion stage: records in, a fused hypothesis per clip out (D72)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.config import FusionSettings, LlmRoute, load_settings
from app.llm.base import LlmResult
from app.llm.fusion import PROMPT_VERSION
from app.services.fusion_stage import FUSION_KIND, fuse_records, fusion_system_id

ROUTE = LlmRoute(
    provider="vertex",
    api="chat",
    model="gemini-3.8-flash",
    system_id="fusion-gemini-3.8-flash",
    reasoning_enabled=True,
    thinking_budget=24576,
)


def record(index: int, texts: tuple[str, str, str]) -> dict[str, Any]:
    start = index * 10.0
    return {
        "segment_id": f"ep_{index:05d}",
        "start_time": start,
        "end_time": start + 8.0,
        "hypotheses": [
            {"system_id": "elevenlabs-scribe-v2", "text": texts[0], "words": None},
            {"system_id": "mai-transcribe-2", "text": texts[1], "words": None},
            {"system_id": "gemini-3.8-flash", "text": texts[2], "words": None},
        ],
        "scores": {"cmi": 0.0, "code_switch_density": 0.0, "flags": ["low_confidence"]},
    }


def echo_b(messages: list[dict[str, Any]]) -> LlmResult:
    """A fuser that takes recogniser B's reading for every segment."""
    block = messages[-1]["content"].split("### TRANSCRIBE THESE", 1)[1].split("###", 1)[0]
    items = [
        {"id": int(i), "t": b, "c": "k"}
        for i, b in re.findall(r"^\[(\d+)\][^\n]*\n  A: [^\n]*\n  B: ([^\n]*)", block, re.M)
    ]
    return LlmResult(
        route="fuse_transcript",
        model="gemini-3.8-flash",
        text=json.dumps(items, ensure_ascii=False),
        raw={"candidates": [{"finishReason": "STOP"}], "modelVersion": "gemini-3.8-flash-001"},
    )


@pytest.fixture
def records() -> list[dict[str, Any]]:
    return [
        record(0, ("हाम्रो टिम राम्रो छ", "हाम्रो team राम्रो छ", "हाम्रो team राम्रो")),
        record(1, ("म घर जान्छु", "म घर जान्छु", "म घर जान्छु")),
    ]


def run(records: list[dict[str, Any]], **overrides: Any):
    return fuse_records(
        records,
        complete=overrides.pop("complete", echo_b),
        route=ROUTE,
        fusion=overrides.pop("fusion", FusionSettings()),
        settings=load_settings(),
        aligner=None,
        clip_path_for=lambda _segment_id: Path("/nonexistent.flac"),
        **overrides,
    )


def test_every_clip_gains_a_fused_hypothesis_after_the_recognisers(records) -> None:
    report = run(records)
    assert report.fused == 2
    assert report.unfused == []
    for rec in records:
        fused = rec["hypotheses"][-1]
        assert fused["kind"] == FUSION_KIND
        assert fused["system_id"] == fusion_system_id(ROUTE)
        assert [h["system_id"] for h in rec["hypotheses"][:3]] == [
            "elevenlabs-scribe-v2",
            "mai-transcribe-2",
            "gemini-3.8-flash",
        ]
    assert records[0]["hypotheses"][-1]["text"] == "हाम्रो team राम्रो छ"


def test_the_prompt_version_is_part_of_the_system_id() -> None:
    """A changed prompt is a new system, never an overwrite of the old one's hypotheses."""
    assert fusion_system_id(ROUTE) == f"fusion-gemini-3.8-flash-{PROMPT_VERSION}"


def test_the_fused_hypothesis_carries_its_provenance(records) -> None:
    run(records)
    fused = records[0]["hypotheses"][-1]
    assert fused["fusion"] == {
        "code": "k",
        "window": 0,
        "depth": 0,
        "prompt_version": PROMPT_VERSION,
        "model_version": "gemini-3.8-flash-001",
        "thinking_budget": 24576,
    }
    assert fused["model_id"] == "gemini-3.8-flash"


def test_code_mixing_is_measured_on_the_fused_text(records) -> None:
    """Scribe spells team in Devanagari, so its CMI misses the switch; the fused text does not."""
    run(records)
    assert records[0]["scores"]["code_switch_density"] > 0.0
    assert "low_confidence" in records[0]["scores"]["flags"]


def test_an_unfused_clip_keeps_only_its_recognisers(records) -> None:
    def refuses(messages: list[dict[str, Any]]) -> LlmResult:
        return LlmResult(route="fuse_transcript", model="m", text="no.", raw={})

    report = run(records, complete=refuses, fusion=FusionSettings(max_depth=0))
    assert report.fused == 0
    assert sorted(report.unfused) == ["ep_00000", "ep_00001"]
    assert all(len(rec["hypotheses"]) == 3 for rec in records)


def test_an_earlier_fusion_hypothesis_is_not_shown_to_the_fuser(records) -> None:
    """Re-fusing must read the recognisers only, never a previous fuser's answer."""
    records[0]["hypotheses"].append(
        {"system_id": "fusion-old", "kind": FUSION_KIND, "text": "OLD FUSED TEXT"}
    )
    seen: list[str] = []

    def recording(messages: list[dict[str, Any]]) -> LlmResult:
        seen.append(messages[-1]["content"])
        return echo_b(messages)

    run(records, complete=recording)
    assert "OLD FUSED TEXT" not in seen[0]
    assert "D: " not in seen[0]


def test_the_report_sums_the_run(records) -> None:
    report = run(records)
    assert report.windows == 1
    assert report.system_id == fusion_system_id(ROUTE)
    assert report.as_dict()["fused"] == 2
