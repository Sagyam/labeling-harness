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
    """Manifest import behaviour.

    A new episode draws its train/val split here, from a hash of its id against
    ``dataset.val_fraction``. Gold is not decided at import at all: it is chosen per clip, by hand
    (D71).
    """

    model_config = _STRICT

    peaks_buckets: int = 1000
    expected_sample_rate: int = 16000
    expected_channels: int = 1
    expected_format: str = "FLAC"


class DatasetSettings(BaseModel):
    """Pot targets and the train/val line (D71).

    Gold is chosen per clip by hand, so nothing here decides what goes into it; the targets are
    progress bars for the dashboard. Durations, not ratios, because a duration is the thing anyone
    actually wants from a corpus ("three hours of benchmark audio").
    """

    model_config = _STRICT

    #: Hours of audio the gold pot is aiming at. Informational: drives the dashboard.
    gold_hours_target: float = Field(default=5.0, gt=0)
    #: Hours the train pot is aiming at. Informational, like the gold target.
    #: Twenty rather than the original fifty: the from-scratch scaling curve the larger figure came
    #: from does not apply here. Nepali already has ~160 h public, and adapting a pretrained model
    #: to a domain saturates roughly an order of magnitude earlier. The corpus is short of
    #: speakers rather than hours (D69), and no number of extra hours from the same twelve people
    #: fixes that.
    train_hours_target: float = Field(default=20.0, gt=0)
    #: Share of the train pot held back as val. Redrawable: train and val hold the same standard
    #: of data, so moving an episode between them costs nothing (unlike moving one out of gold).
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    #: Seed for the train/val draw, so it is reproducible; change it to redraw the line.
    pot_seed: int = 20260101
    #: Stratification variables the dashboard checks gold against. Ordered by how much a gap in
    #: one would hurt: an all-one-show benchmark is the worst outcome.
    coverage_keys: list[Literal["show_id", "gender", "age_bracket", "topic"]] = Field(
        default_factory=lambda: ["show_id", "gender", "age_bracket", "topic"]
    )
    #: Hours below which the inventory calls a stratum thin and asks for more of it (D69). A
    #: floor rather than a target share: there is no defensible ideal share for "speech from
    #: 60-79 year olds", but there is a point below which a stratum supports no claim at all.
    min_stratum_hours: float = Field(default=1.0, gt=0)


class QueueWeights(BaseModel):
    """Priority score weights. Must sum to 1.0 so the score stays in 0-1."""

    model_config = _STRICT

    #: Share of speech time where every other system contradicts the seed. Halved from 0.60 by
    #: D67: it ranks the tails well and cannot separate the middle, because it measures how *much*
    #: of a clip is wrong when the annotator's cost is driven by *whether* any of it is.
    seed_outvoted: float = 0.30
    low_confidence: float = 0.125
    rule_flag_score: float = 0.075
    #: ``seed_outvoted`` with the clock removed -- the share of seed tokens no other system has.
    #: On its own it scores AUC 0.740 against realized edits, against 0.747 for the whole
    #: superseded formula (D67).
    seed_orphan_rate: float = 0.30
    #: Latin-script tokens every other system agreed on and the seed lacks: the English-in-Latin
    #: policy (D64) as a ranking signal.
    roman_gap: float = 0.20

    @model_validator(mode="after")
    def _check_sum(self) -> QueueWeights:
        total = (
            self.seed_outvoted
            + self.low_confidence
            + self.rule_flag_score
            + self.seed_orphan_rate
            + self.roman_gap
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"queue weights must sum to 1.0, got {total}")
        return self


class LegacyQueueWeights(BaseModel):
    """The superseded terms, kept so the old score can be recorded beside the new one (D54).

    These do not rank anything and deliberately do not sum to 1: they exist to let the first full
    run adjudicate between the two formulas on more than the 22 labels behind the change.
    """

    model_config = _STRICT

    word_disagreement_rate: float = 0.40
    code_switch_density: float = 0.20


class QueueSettings(BaseModel):
    """Queue building: priority weights, rule-flag thresholds and audit sampling."""

    model_config = _STRICT

    weights: QueueWeights = Field(default_factory=QueueWeights)
    legacy_weights: LegacyQueueWeights = Field(default_factory=LegacyQueueWeights)
    audit_sample_rate: float = 0.05
    audit_seed: int = 1234
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 30.0
    #: Where confidence is treated as exhausted. -0.5 rather than -2.0: the only system reporting
    #: an ``avg_logprob`` spans about -0.68 to -0.004, so the old floor confined the term to the
    #: bottom third of 0-1 and its 0.25 weight never bought more than about 0.08 (D54).
    logprob_floor: float = -0.5
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


class YouTubeSettings(BaseModel):
    """Fetching an episode's audio straight from a YouTube URL, via yt-dlp.

    ``max_duration_seconds`` is a spend guard rather than a technical limit: every configured
    ``asr*`` route transcribes every clip, so the cost of an ingest is linear in the source
    duration and a mistyped link to an eight-hour livestream is an expensive mistake.
    """

    model_config = _STRICT

    #: yt-dlp format selector. Audio-only, because stage 1 re-encodes to 16 kHz mono FLAC anyway.
    format: str = "bestaudio/best"
    max_duration_seconds: float = 14400.0
    probe_timeout_seconds: float = 30.0
    download_timeout_seconds: float = 1800.0
    #: Netscape-format cookie jar, for videos YouTube will not serve anonymously. Path only --
    #: the file itself is a secret and belongs outside the repository.
    cookies_file: str = ""


class IngestSettings(BaseModel):
    """Web ingestion scratch space.

    Resolved against the repository root like every other configured path, so an upload does not
    land wherever the server process happened to be started from.
    """

    model_config = _STRICT

    work_root: Path = Path("./data/ingest_work")
    max_segment_concurrency: int = Field(default=4, ge=1, le=16)
    youtube: YouTubeSettings = Field(default_factory=YouTubeSettings)
    #: Text route that labels an episode's topic from its transcript (D57). One call per episode,
    #: only when the form left the topic blank. Empty disables it; a name that is not in
    #: `llm_routes.yaml` is logged and skipped, because a metadata field never fails an ingest.
    topic_route: str = "classify_topic"

    @field_validator("work_root")
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
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    translit: TranslitSettings = Field(default_factory=TranslitSettings)
    labels: LabelSettings = Field(default_factory=LabelSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)


class LlmRoute(BaseModel):
    """One named route: which vendor, which API shape on it, and which model.

    A route named ``asr*`` becomes one ASR system during ingestion. ``api`` selects the request
    shape, because a transcription endpoint and a chat endpoint that happens to accept audio are
    not interchangeable:

    * ``transcription`` -- a dedicated speech-to-text endpoint. The model is a speech
      recogniser, and it is the only shape that can return word timings of its own.
    * ``audio_chat`` -- chat completions with the clip attached as an ``input_audio`` part. The
      model is a general LLM being asked to transcribe, so it obeys a prompt but may also
      editorialise or hallucinate over silence.
    * ``chat`` -- ordinary text chat completions.
    """

    model_config = _STRICT

    provider: Literal["openrouter", "elevenlabs", "vertex"] = "openrouter"
    api: Literal["chat", "transcription", "audio_chat"] = "chat"
    model: str
    #: Name this system is recorded under in ``asr_systems`` and every export. Defaults to the
    #: route name with its ``asr_`` prefix stripped. Set it explicitly and leave it alone: it is
    #: how a hypothesis is attributed for the life of the corpus.
    system_id: str | None = None
    #: ISO language hint passed to the provider, where the provider takes one.
    language: str | None = None
    #: BCP-47 codes for a provider that accepts several at once. A code-switched clip is not
    #: one language with loanwords in it, so a transcriber that can be told about both is told
    #: about both. Where this is empty, ``language`` is used as the single hint.
    language_codes: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    #: Whether the model may think before answering. Provider-normalised: OpenRouter sends
    #: ``reasoning: {enabled: ...}``, Vertex sends ``generationConfig.thinkingConfig``.
    #:
    #: ``None`` sends nothing and takes the provider's default, which for Gemini 3.x is thinking
    #: ON -- and expensively so. On text it truncated 47% of the first episode's script
    #: restorations at ``max_tokens`` with 96% of the budget spent reasoning (D44); on
    #: ``audio_chat`` it spends a mean 895 thought tokens against 88 tokens of transcript, 91% of
    #: billed output, for no measured gain (D45). Set it False on any route whose task is
    #: mechanical.
    #:
    #: Never set on an ``api: transcription`` route: the dedicated recogniser answers any
    #: ``thinkingConfig`` with ``400 Thinking is not enabled for this model``.
    reasoning_enabled: bool | None = None
    #: Upper bound on thought tokens, for the one kind of route where thinking is the job rather
    #: than a default left on (transcript fusion, D72). Sent as Vertex's ``thinkingBudget``;
    #: without it ``reasoning_enabled: true`` means unbounded (-1). Thought tokens are billed as
    #: output and count against ``max_tokens``, so the budget is what keeps a long answer from
    #: being truncated by its own reasoning.
    thinking_budget: int | None = Field(default=None, gt=0)
    #: Fill this route's word spans with the local CTC forced aligner (D32). For a transcriber
    #: that returns no timestamps of its own; a route that reports them keeps what it reported.
    forced_align: bool = False
    #: Name of the text route that puts this route's transcript back into mixed script (D41).
    #: Set only on a recogniser that writes everything in one script and cannot be told not to.
    #: The rewrite preserves the token count exactly, so each restored word keeps the span the
    #: recogniser measured for it -- this is not forced alignment and must not be confused with
    #: ``forced_align``.
    restore_script_route: str | None = None
    #: Ask this route's provider for a speaker label per word. Only a transcriber that offers
    #: diarization honours it; the rest ignore it and their words keep a null speaker, which is
    #: the honest value for "not diarized". Labels are clip-local -- ``spk:0`` in one clip is
    #: unrelated to ``spk:0`` in the next -- so they support agreement between systems on one
    #: clip, not speaker identity across an episode (D49).
    diarize: bool = False
    #: Keep this route's hypothesis out of the cross-system disagreement scores (D39). For a
    #: transcriber whose disagreement is an artefact of its orthography rather than of what it
    #: heard: `word_disagreement_rate` is a raw token comparison and carries 0.40 of the priority
    #: score, so a system writing another script disagrees on every foreign word for free. The
    #: hypothesis is still stored, exported and shown -- only the comparison drops it.
    exclude_from_disagreement: bool = False

    @model_validator(mode="after")
    def _check_provider_api(self) -> LlmRoute:
        if self.thinking_budget is not None and self.reasoning_enabled is False:
            raise ValueError("thinking_budget is set on a route with reasoning_enabled: false")
        if self.provider == "elevenlabs" and self.api != "transcription":
            raise ValueError("the elevenlabs provider only offers api: transcription")
        if self.provider == "vertex" and self.api not in ("transcription", "audio_chat", "chat"):
            raise ValueError(
                "the vertex provider only offers api: transcription, audio_chat or chat"
            )
        return self


class LlmRoutes(BaseModel):
    """Routing table for every inference provider.

    OpenRouter is the default and carries all text inference. A provider is called directly when
    OpenRouter cannot reach it -- ElevenLabs Scribe (D21), Gemini on Vertex AI (D35, D39).
    What every route shares is that its calls are logged to ``llm_requests``, which is the only
    spend record there is.
    """

    model_config = _STRICT

    enabled: bool = False
    base_url: str = "https://openrouter.ai/api/v1"
    elevenlabs_base_url: str = "https://api.elevenlabs.io/v1"
    vertex_base_url: str = "https://aiplatform.googleapis.com/v1beta1"
    #: GCP project the Vertex calls bill and quota against. Blank here so the committed file names
    #: no project; ``GOOGLE_CLOUD_PROJECT`` fills it in.
    vertex_project: str = ""
    #: Region serving the models. ``global`` serves both; ``us-central1`` is the only other
    #: location that carries the recogniser.
    vertex_location: str = ""
    default_timeout_seconds: float = 30.0
    default_max_tokens: int = 1024
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5
    dry_run: bool = True
    routes: dict[str, LlmRoute] = Field(default_factory=dict)

    def asr_route_names(self) -> list[str]:
        """Every route that becomes an ASR system during ingestion, in configured order."""
        return [name for name in self.routes if name.startswith("asr")]


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
