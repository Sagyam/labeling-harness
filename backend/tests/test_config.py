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
        settings.queue.weights.word_disagreement_rate
        + settings.queue.weights.low_confidence
        + settings.queue.weights.code_switch_density
        + settings.queue.weights.rule_flag_score
    )
    assert total == pytest.approx(1.0)


def test_split_ratios_sum_to_one() -> None:
    settings = load_settings()
    ratios = settings.importer.split_ratios
    assert ratios["train"] + ratios["val"] + ratios["test"] == pytest.approx(1.0)


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

    Scribe measures word timings and per-word logprobs itself, and overwriting them with the
    aligner's would destroy the independent reference D27's boundary check compares against.
    """
    routes = load_llm_routes().routes
    aligned = {name for name, route in routes.items() if route.forced_align}
    assert aligned == {"asr_gemini_flash"}
