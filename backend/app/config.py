"""Configuration loading.

Non-secret configuration lives in ``config/settings.yaml``. Every value can be overridden by an
environment variable named ``HARNESS_<SECTION>__<KEY>`` (a double underscore separates nesting
levels). Secrets are never read from YAML in practice -- the committed file leaves them empty and
the deployment supplies them through the environment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"
DEFAULT_LLM_ROUTES_PATH = REPO_ROOT / "config" / "llm_routes.yaml"

_STRICT = ConfigDict(extra="forbid", frozen=True)


class AppSettings(BaseModel):
    """Process-level identity and logging."""

    model_config = _STRICT

    name: str = "labeling-harness"
    environment: str = "local"
    log_level: str = "INFO"


class DatabaseSettings(BaseModel):
    """Postgres connection parameters."""

    model_config = _STRICT

    host: str = "localhost"
    port: int = 5432
    name: str = "harness"
    user: str = "harness"
    password: str = ""
    pool_size: int = 5

    @property
    def url(self) -> str:
        """SQLAlchemy URL. ``DATABASE_URL`` in the environment wins over the assembled parts."""
        override = os.environ.get("DATABASE_URL")
        if override:
            return override
        user = quote(self.user, safe="")
        password = quote(self.password, safe="")
        credentials = f"{user}:{password}" if password else user
        return f"postgresql+psycopg://{credentials}@{self.host}:{self.port}/{self.name}"


class MinioSettings(BaseModel):
    """MinIO / S3-compatible object storage parameters."""

    model_config = _STRICT

    endpoint: str = "localhost:9000"
    bucket: str = "harness"
    secure: bool = False
    access_key: str = ""
    secret_key: str = ""


class StorageSettings(BaseModel):
    """Object storage selection. ``local`` keeps the harness usable without MinIO running."""

    model_config = _STRICT

    backend: Literal["local", "minio"] = "local"
    local_root: Path = Path("./data/objects")
    minio: MinioSettings = Field(default_factory=MinioSettings)

    @field_validator("local_root")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()


class ApiSettings(BaseModel):
    """HTTP server settings. An empty ``auth_token`` disables authentication (local dev)."""

    model_config = _STRICT

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    auth_token: str = ""


class ImporterSettings(BaseModel):
    """Manifest import behaviour, including the frozen-split assignment."""

    model_config = _STRICT

    split_seed: int = 20260101
    split_ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    peaks_buckets: int = 1000
    expected_sample_rate: int = 16000
    expected_channels: int = 1
    expected_format: str = "FLAC"

    @model_validator(mode="after")
    def _check_ratios(self) -> ImporterSettings:
        if set(self.split_ratios) != {"train", "val", "test"}:
            raise ValueError("split_ratios must have exactly the keys train, val and test")
        if abs(sum(self.split_ratios.values()) - 1.0) > 1e-9:
            raise ValueError("split_ratios must sum to 1.0")
        return self


class QueueWeights(BaseModel):
    """Priority score weights. Must sum to 1.0 so the score stays in 0-1."""

    model_config = _STRICT

    word_disagreement_rate: float = 0.40
    low_confidence: float = 0.25
    code_switch_density: float = 0.20
    rule_flag_score: float = 0.15

    @model_validator(mode="after")
    def _check_sum(self) -> QueueWeights:
        total = (
            self.word_disagreement_rate
            + self.low_confidence
            + self.code_switch_density
            + self.rule_flag_score
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"queue weights must sum to 1.0, got {total}")
        return self


class QueueSettings(BaseModel):
    """Queue building: priority weights, rule-flag thresholds and audit sampling."""

    model_config = _STRICT

    weights: QueueWeights = Field(default_factory=QueueWeights)
    audit_sample_rate: float = 0.05
    audit_seed: int = 1234
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 30.0
    logprob_floor: float = -2.0
    no_speech_prob_threshold: float = 0.6
    max_speaking_rate_wps: float = 6.0
    min_speaking_rate_wps: float = 0.3
    repeated_ngram_size: int = 4
    repeated_ngram_threshold: int = 3


class TranslitSettings(BaseModel):
    """Latin to Devanagari input helper. The cache is always consulted before any provider."""

    model_config = _STRICT

    provider_order: list[Literal["remote", "offline"]] = Field(
        default_factory=lambda: ["remote", "offline"]
    )
    remote_endpoint: str = "https://inputtools.google.com/request"
    remote_timeout_seconds: float = 1.5
    max_candidates: int = 5


class LabelSettings(BaseModel):
    """Defaults stamped onto labels written by this deployment."""

    model_config = _STRICT

    policy_version: str = "policy_v1"
    default_label_version: str = "v1"
    default_annotator: str = "owner"


class ExportSettings(BaseModel):
    """Where export directories are written."""

    model_config = _STRICT

    output_root: Path = Path("./exports")

    @field_validator("output_root")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()


class Settings(BaseSettings):
    """Root settings object, assembled from YAML then overlaid with environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="HARNESS_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Environment variables override the YAML file, which is passed in as init values."""
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    importer: ImporterSettings = Field(default_factory=ImporterSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    translit: TranslitSettings = Field(default_factory=TranslitSettings)
    labels: LabelSettings = Field(default_factory=LabelSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)


class LlmRoute(BaseModel):
    """One named LLM route. No route is wired to a pipeline stage at MVP."""

    model_config = _STRICT

    model: str
    timeout_seconds: float | None = None
    max_tokens: int | None = None
    temperature: float | None = None


class LlmRoutes(BaseModel):
    """OpenRouter routing table. All LLM inference in this codebase goes through OpenRouter."""

    model_config = _STRICT

    enabled: bool = False
    base_url: str = "https://openrouter.ai/api/v1"
    default_timeout_seconds: float = 30.0
    default_max_tokens: int = 1024
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5
    dry_run: bool = True
    routes: dict[str, LlmRoute] = Field(default_factory=dict)


def load_dotenv(path: Path | None = None) -> None:
    """Populate ``os.environ`` from a repo-root ``.env``, without overriding real environment.

    Keys are taken verbatim, so both prefixed settings (``HARNESS_DATABASE__PASSWORD``) and bare
    ones the settings model does not own (``DATABASE_URL``, ``POSTGRES_HOST_PORT``) work. Values may
    be quoted; ``export`` prefixes and ``#`` comments are ignored.
    """
    path = path or (REPO_ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings from YAML, then let ``HARNESS_*`` environment variables override.

    Args:
        path: Settings file. Defaults to ``config/settings.yaml`` at the repository root.

    Entry points call :func:`load_dotenv` first; loading is not implicit here so that tests and
    callers keep full control over the environment they are validating against.

    Raises:
        FileNotFoundError: The settings file does not exist.
        ValueError: The file contains unknown keys or invalid values.
    """
    return Settings(**_read_yaml(Path(path) if path else DEFAULT_SETTINGS_PATH))


def load_llm_routes(path: Path | str | None = None) -> LlmRoutes:
    """Load the OpenRouter routing table from ``config/llm_routes.yaml``."""
    data = _read_yaml(Path(path) if path else DEFAULT_LLM_ROUTES_PATH)
    if "HARNESS_LLM__DRY_RUN" in os.environ:
        data["dry_run"] = os.environ["HARNESS_LLM__DRY_RUN"].lower() in ("true", "1", "yes")
    return LlmRoutes(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton, for FastAPI dependencies."""
    return load_settings()
