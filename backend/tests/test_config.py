"""Tests for YAML + environment configuration loading."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.config import (
    DEFAULT_SETTINGS_PATH,
    Settings,
    load_dotenv,
    load_llm_routes,
    load_settings,
)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_loads_repository_default_settings() -> None:
    settings = load_settings()
    assert settings.app.name == "labeling-harness"
    assert settings.database.name == "harness"
    assert settings.storage.backend == "local"
    assert settings.importer.expected_sample_rate == 16000


def test_queue_weights_sum_to_one() -> None:
    settings = load_settings()
    total = (
        settings.queue.weights.unsupported_rate
        + settings.queue.weights.dropped_rate
        + settings.queue.weights.asr_disagreement
        + settings.queue.weights.acoustic_gap
        + settings.queue.weights.low_confidence
        + settings.queue.weights.rule_flag_score
    )
    assert total == pytest.approx(1.0)


def test_dataset_targets_are_durations_not_ratios() -> None:
    """D63: the corpus is sized in hours, because a ratio cannot say "five hours of benchmark"."""
    settings = load_settings()
    assert settings.dataset.gold_hours_target > 0
    assert settings.dataset.train_hours_target > 0
    assert 0.0 <= settings.dataset.val_fraction < 1.0
    assert settings.dataset.coverage_keys


def test_the_importer_no_longer_decides_splits() -> None:
    settings = load_settings()
    assert not hasattr(settings.importer, "split_ratios")
    assert not hasattr(settings.importer, "split_seed")


def test_fusion_settings_window_targets_words() -> None:
    settings = load_settings()
    assert settings.fusion.window_target_words == 3000
    assert not hasattr(settings.fusion, "window_target_seconds")


def test_env_var_overrides_yaml_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(
        tmp_path,
        """
        database:
          host: localhost
          port: 5432
          name: harness
          user: harness
        """,
    )
    monkeypatch.setenv("HARNESS_DATABASE__PORT", "6543")
    settings = load_settings(path)
    assert settings.database.port == 6543


def test_nested_env_var_overrides_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, "storage:\n  backend: minio\n")
    monkeypatch.setenv("HARNESS_STORAGE__MINIO__ACCESS_KEY", "from-env")
    settings = load_settings(path)
    assert settings.storage.minio.access_key == "from-env"


def test_no_secrets_are_committed_to_yaml() -> None:
    """The committed YAML leaves every secret empty; real values arrive from the environment."""
    raw = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    assert raw["database"]["password"] == ""
    assert raw["storage"]["minio"]["access_key"] == ""
    assert raw["storage"]["minio"]["secret_key"] == ""
    assert raw["api"]["auth_token"] == ""


def test_database_url_is_assembled_from_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("HARNESS_DATABASE__PASSWORD", "s3cret")
    settings = load_settings()
    assert settings.database.url == ("postgresql+psycopg://harness:s3cret@localhost:5432/harness")


def test_database_url_env_var_wins_over_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/other")
    settings = load_settings()
    assert settings.database.url == "postgresql+psycopg://u:p@db:5432/other"


def test_database_url_quotes_special_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("HARNESS_DATABASE__PASSWORD", "p@ss:word/1")
    settings = load_settings()
    assert "p%40ss%3Aword%2F1" in settings.database.url


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "app:\n  name: x\n  nonsense: 1\n")
    with pytest.raises(ValueError, match="nonsense"):
        load_settings(path)


def test_invalid_storage_backend_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HARNESS_STORAGE__BACKEND", raising=False)
    path = _write(tmp_path, "storage:\n  backend: s3\n")
    with pytest.raises(ValueError, match="backend"):
        load_settings(path)


def test_missing_settings_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.yaml")


def test_settings_are_immutable() -> None:
    settings = load_settings()
    with pytest.raises(ValidationError):
        settings.app.name = "changed"  # type: ignore[misc]


def test_local_root_resolves_relative_to_repo_root() -> None:
    settings = load_settings()
    assert settings.storage.local_root.is_absolute()


def test_llm_routes_configured_for_cloud_asr() -> None:
    routes = load_llm_routes()
    assert routes.enabled is True
    assert routes.base_url.startswith("https://openrouter.ai")
    assert routes.asr_route_names() == [
        "asr_scribe_v2",
        "asr_mai_transcribe_2",
        "asr_gemini_flash",
    ]


def test_the_committed_transcribers_name_their_provider_and_api() -> None:
    routes = load_llm_routes().routes
    scribe = routes["asr_scribe_v2"]
    assert (scribe.provider, scribe.api) == ("elevenlabs", "transcription")
    assert scribe.model == "scribe_v2"
    assert scribe.language == "ne", "Scribe takes no prompt; the language code is its steering"

    mai = routes["asr_mai_transcribe_2"]
    assert (mai.provider, mai.api) == ("openrouter", "transcription")
    assert mai.model == "microsoft/mai-transcribe-2"
    assert mai.system_id == "mai-transcribe-2"
    assert mai.language == "ne"

    assert "asr_gemini_composite" not in routes, (
        "the composite recogniser was removed in D51; it deleted the dependent variable"
    )
    # D73: raw Scribe feeds the fuser, which writes the script policy itself. `script_restore.py`
    # stays in the tree (D51) for a recogniser whose spans must survive a respelling.
    assert scribe.restore_script_route is None
    assert "script_restore" not in routes

    fusion = routes["fuse_transcript"]
    assert (fusion.provider, fusion.api) == ("vertex", "chat")
    assert fusion.reasoning_enabled is True
    # Thinking and answer share max_tokens; an unbounded budget is what truncates a window.
    assert fusion.thinking_budget is not None
    assert fusion.thinking_budget < fusion.max_tokens

    gemini = routes["asr_gemini_flash"]
    assert (gemini.provider, gemini.api) == ("vertex", "audio_chat"), (
        "a general model asked to transcribe, not a dedicated recogniser -- name it so"
    )
    assert gemini.model == "gemini-3.8-flash"
    assert gemini.system_id == "gemini-3.8-flash"
    assert gemini.language == "ne"


def test_no_transcriber_is_configured_on_an_openrouter_batch_variant() -> None:
    """OpenRouter's Batch API is text-only: a `:batch` slug can never carry a clip (D22).

    The synchronous endpoint rejects the slug outright, and the Batch API accepts the submission
    and then terminally fails validation, so the failure would surface an episode late rather
    than at configuration time.
    """
    for name, route in load_llm_routes().routes.items():
        if name.startswith("asr"):
            assert not route.model.endswith(":batch"), name


def test_settings_type_is_exported() -> None:
    assert isinstance(load_settings(), Settings)


def test_dotenv_populates_environment_without_overriding_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A repo-root .env supplies local secrets; a real environment variable still wins."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nexport HARNESS_DATABASE__PASSWORD="from-dotenv"\nDATABASE_URL=x\n\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("HARNESS_DATABASE__PASSWORD", raising=False)
    monkeypatch.setenv("DATABASE_URL", "already-set")
    load_dotenv(env_file)
    assert os.environ["HARNESS_DATABASE__PASSWORD"] == "from-dotenv"
    assert os.environ["DATABASE_URL"] == "already-set"


def test_dotenv_keys_outside_the_settings_model_are_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Compose-only variables such as POSTGRES_HOST_PORT must not break settings validation."""
    env_file = tmp_path / ".env"
    env_file.write_text("POSTGRES_HOST_PORT=5555\nTEST_DATABASE_URL=postgresql://x\n")
    load_dotenv(env_file)
    assert load_settings().database.port == 5432


def test_load_settings_does_not_read_dotenv_implicitly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only entry points load .env, so tests control the environment they validate against."""
    env_file = tmp_path / ".env"
    env_file.write_text("HARNESS_APP__ENVIRONMENT=from-dotenv\n")
    monkeypatch.delenv("HARNESS_APP__ENVIRONMENT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_settings().app.environment == "local"


def test_only_the_transcriber_without_timestamps_asks_for_forced_alignment() -> None:
    """A route that reports its own word spans must keep them (D32).

    Scribe and Gemini 3.5 Transcribe measure their own word timings, and overwriting them with
    the aligner's would destroy the independent references D33's boundary check compares.
    """
    routes = load_llm_routes().routes
    aligned = {name for name, route in routes.items() if route.forced_align}
    assert aligned == {"asr_gemini_flash"}


def test_no_route_is_held_out_of_the_disagreement_score() -> None:
    """The hold-out set is empty because its only member is gone (D51).

    D50 held the composite out rather than removing it, because it was one of only two sources of
    speaker labels. D51 removed the route outright: the diarization argument collapsed once
    clip-local labels were shown to be unmeasurable, and what remained was the least accurate of
    the four systems with a bias against the dependent variable. Nothing else has ever needed
    holding out, so an empty set here is the honest state -- not a forgotten flag.
    """
    routes = load_llm_routes().routes
    assert {name for name, route in routes.items() if route.exclude_from_disagreement} == set()


def test_no_route_asks_for_speaker_labels() -> None:
    """Segment-level diarization is off for every cloud ASR route (D52).

    D49 turned Scribe's on because a second source made a label checkable. D51 removed the other
    source, and the measurement behind it showed clip-local labels cannot answer the question they
    were collected for. What remains is storage and a column inviting a join that would be wrong,
    so nothing is asked for a speaker any more. Speaker identity is a full-episode problem.
    """
    routes = load_llm_routes().routes
    assert {name for name, route in routes.items() if route.diarize} == set()
