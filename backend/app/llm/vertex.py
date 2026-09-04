"""Vertex AI client for Google's Gemini models.

Google serves its models through Vertex AI, so this is the client for all of them (D35). It
replaces the AI Studio client, whose free-tier quotas made an ingest a coin flip.

Two request shapes reach a model here, matching the two route shapes in ``config.py``:

* ``api: transcription`` posts to ``interactions:create`` with a ``transcriptionConfig``. The
  model is a dedicated speech recogniser -- ``gemini-3.5-transcribe`` -- and it returns word
  text, word timings and a speaker label per word.
* ``api: audio_chat`` posts to ``publishers/google/models/{model}:generateContent`` with the clip
  inlined. The model is a general LLM being asked to transcribe: it obeys the transcript policy,
  returns text and no timings, and may editorialise or hallucinate over silence.

Authentication is Application Default Credentials, not an API key: a service account JSON at
``GOOGLE_APPLICATION_CREDENTIALS``, a `gcloud auth application-default login` session, or the
metadata server on a GCP host. The project and region come from ``config/llm_routes.yaml``, or
from ``GOOGLE_CLOUD_PROJECT`` / ``GOOGLE_CLOUD_LOCATION`` when the file leaves them blank.
"""

from __future__ import annotations

import base64
import os
import threading
from pathlib import Path
from typing import Any, NoReturn

import httpx
from sqlalchemy.orm import Session

from app.config import LlmRoute, LlmRoutes
from app.llm.base import (
    AsrResult,
    LlmDisabledError,
    LlmRequestFailed,
    LlmRouteNotConfigured,
    ProviderClient,
    dry_run_transcript,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Environment variables consulted when the routing table leaves project or region blank.
PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"
LOCATION_ENV = "GOOGLE_CLOUD_LOCATION"

#: The scope every Vertex AI call needs.
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

#: Clips are FLAC by invariant 6.
AUDIO_MIME_TYPE = "audio/flac"

#: Only version carrying ``interactions:create``. ``generateContent`` is on it too, so one
#: version serves both shapes.
API_VERSION = "v1beta1"

_credentials_lock = threading.Lock()
_credentials: Any = None


def _default_credentials() -> Any:
    """Application Default Credentials, resolved once per process and refreshed in place.

    Segment workers all call this, so the lookup is locked: ADC discovery reads the filesystem
    and may hit the metadata server, and a token refresh is not worth doing four times over.
    """
    global _credentials
    with _credentials_lock:
        if _credentials is None:
            import google.auth

            _credentials, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
        return _credentials


def parse_generate_content(body: dict[str, Any]) -> str:
    """Join the text parts of the first ``generateContent`` candidate that has any.

    A model may split its answer over several parts, and may return a candidate carrying only
    metadata (a safety block, a tool call); such a candidate yields no text and the next one is
    tried.
    """
    for candidate in body.get("candidates") or []:
        content = candidate.get("content") or {}
        texts = [
            str(part["text"])
            for part in content.get("parts") or []
            if isinstance(part, dict) and part.get("text")
        ]
        if texts:
            return "".join(texts).strip()
    return ""


def _content_blocks(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Every model-output content block of an interaction, in order.

    A step is a union; only ``modelOutput`` carries the transcript. The un-nested ``content``
    form is accepted too, because the v1beta1 surface has already moved once.
    """
    blocks: list[dict[str, Any]] = []
    for step in body.get("steps") or []:
        if not isinstance(step, dict):
            continue
        output = step.get("modelOutput") or step.get("model_output") or step
        for block in output.get("content") or []:
            if isinstance(block, dict):
                blocks.append(block)
    return blocks


def _duration_seconds(value: Any) -> float | None:
    """Read a protobuf duration (``"1.75s"``) as seconds. Absent or unparseable means unknown."""
    if value is None:
        return None
    try:
        return float(str(value).rstrip("s"))
    except ValueError:
        return None


def parse_interaction(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
    """Read the transcript and its word annotations out of an interaction response.

    Returns:
        ``(text, words)``. ``words`` is ``None`` when the response carried no word annotations,
        never an empty list -- the scorer reads ``None`` as "no signal".
    """
    texts: list[str] = []
    words: list[dict[str, Any]] = []
    for block in _content_blocks(body):
        text_block = block.get("text")
        if not isinstance(text_block, dict):
            continue
        if text_block.get("text"):
            texts.append(str(text_block["text"]))
        for annotation in text_block.get("annotations") or []:
            if not isinstance(annotation, dict):
                continue
            info = annotation.get("wordInfo") or annotation.get("word_info")
            if not isinstance(info, dict) or not info.get("text"):
                continue
            words.append(
                {
                    "word": str(info["text"]),
                    "start": _duration_seconds(info.get("startOffset") or info.get("start_offset")),
                    "end": _duration_seconds(info.get("endOffset") or info.get("end_offset")),
                    "speaker": info.get("speaker"),
                }
            )
    return "".join(texts).strip(), words or None


class VertexClient(ProviderClient):
    """A logged, retrying client for Gemini models served from Vertex AI."""

    def __init__(
        self,
        session: Session,
        *,
        config: LlmRoutes | None = None,
        access_token: str | None = None,
        project: str | None = None,
        location: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(session, config=config, client=client)
        self._access_token = access_token
        self.project = project or self.config.vertex_project or os.environ.get(PROJECT_ENV, "")
        self.location = (
            location or self.config.vertex_location or os.environ.get(LOCATION_ENV) or "global"
        )

    def _bearer_token(self) -> str:
        """A live access token, refreshed when the cached one has expired."""
        if self._access_token is not None:
            return self._access_token
        import google.auth.transport.requests

        credentials = _default_credentials()
        with _credentials_lock:
            if not credentials.valid:
                credentials.refresh(google.auth.transport.requests.Request())
            return str(credentials.token)

    def _base_url(self) -> str:
        """The regional Vertex AI host and the project path under it.

        ``global`` is not a region prefix -- it has its own unprefixed host -- so it is spelt
        out rather than formatted in.
        """
        host = (
            "aiplatform.googleapis.com"
            if self.location == "global"
            else f"{self.location}-aiplatform.googleapis.com"
        )
        return f"https://{host}/{API_VERSION}/projects/{self.project}/locations/{self.location}"

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        route: str,
        prompt: str | None = None,
        language: str | None = None,
        custom_vocabulary: list[str] | None = None,
        dry_run: bool | None = None,
        timeout_seconds: float | None = None,
    ) -> AsrResult:
        """Transcribe one clip with whichever Gemini model and API shape the route names.

        Args:
            audio_path: Path to the audio file (16 kHz mono FLAC).
            route: Route name from ``config/llm_routes.yaml``.
            prompt: Transcript policy. Both shapes take one -- ``audio_chat`` as the first
                content part, ``transcription`` as the interaction's system instruction.
            language: BCP-47 language hint overriding the route's. The route's
                ``language_codes`` is what a code-switched clip actually needs, and this
                overrides only the single-code fallback.
            custom_vocabulary: Terms to bias recognition towards. ``transcription`` only.
            dry_run: Override the configured dry-run mode.
            timeout_seconds: Override default timeout.

        Returns:
            The transcription. ``words`` is populated only on an ``api: transcription`` route;
            ``audio_chat`` returns text, and its spans come from the forced aligner (D32).

        Raises:
            LlmRouteNotConfigured: If the route is missing from configuration.
            LlmDisabledError: If inference is globally disabled.
            LlmRequestFailed: If the project, credentials or retries run out.
        """
        route_config = self.config.routes.get(route)
        if route_config is None:
            raise LlmRouteNotConfigured(f"no route named {route!r} in llm_routes.yaml")

        audio_file = Path(audio_path)
        model_id = route_config.model
        request_hash = self._audio_hash(audio_file, model_id)
        summary = f"vertex_transcribe: {audio_file.name} model={model_id} api={route_config.api}"
        effective_dry_run = self.config.dry_run if dry_run is None else dry_run

        if effective_dry_run:
            self._log(
                route=route,
                model=model_id,
                request_hash=request_hash,
                input_summary=summary,
                status="dry_run",
            )
            logger.info("vertex_dry_run", route=route, model=model_id, file=str(audio_file))
            text, words = dry_run_transcript(request_hash)
            return AsrResult(route=route, model=model_id, text=text, words=words, dry_run=True)

        if not self.config.enabled:
            raise LlmDisabledError(
                "LLM inference is disabled (config/llm_routes.yaml: enabled: false)"
            )

        if not self.project:
            self._fail(
                route,
                model_id,
                request_hash,
                summary,
                "no Vertex AI project: set vertex_project in config/llm_routes.yaml "
                f"or {PROJECT_ENV} in the environment",
            )

        try:
            token = self._bearer_token()
        except Exception as exc:  # google.auth raises several unrelated types
            self._fail(
                route,
                model_id,
                request_hash,
                summary,
                f"Vertex AI credentials are unavailable: {type(exc).__name__}: {exc}",
            )

        timeout = (
            timeout_seconds or route_config.timeout_seconds or self.config.default_timeout_seconds
        )
        audio_b64 = base64.b64encode(audio_file.read_bytes()).decode("ascii")
        wants_words = route_config.api == "transcription"
        if wants_words:
            url = f"{self._base_url()}/interactions:create"
            payload = self._interaction_payload(
                route_config,
                audio_b64=audio_b64,
                instruction=prompt,
                language=language,
                custom_vocabulary=custom_vocabulary,
            )
        else:
            url = f"{self._base_url()}/publishers/google/models/{model_id}:generateContent"
            payload = self._generate_content_payload(
                route_config, audio_b64=audio_b64, instruction=prompt, language=language
            )

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        def send() -> httpx.Response:
            return self._get_client().post(url, json=payload, headers=headers, timeout=timeout)

        response, last_error, latency_ms = self._send_with_retries(send)
        if response is None:
            self._log(
                route=route,
                model=model_id,
                request_hash=request_hash,
                input_summary=summary,
                status="failed",
                latency_ms=latency_ms,
                error=last_error,
            )
            logger.info("vertex_request_failed", route=route, error=last_error)
            raise LlmRequestFailed(f"Vertex AI transcription failed: {last_error}")

        body = response.json()
        if wants_words:
            text, words = parse_interaction(body)
        else:
            text, words = parse_generate_content(body), None

        self._log(
            route=route,
            model=model_id,
            request_hash=request_hash,
            input_summary=summary,
            status="succeeded",
            output=body,
            latency_ms=latency_ms,
        )
        return AsrResult(
            route=route,
            model=model_id,
            text=text,
            words=words,
            latency_ms=latency_ms,
            raw=body,
        )

    def _fail(
        self, route: str, model: str, request_hash: str, summary: str, message: str
    ) -> NoReturn:
        """Log a request that never left the process, then raise. Never returns."""
        self._log(
            route=route,
            model=model,
            request_hash=request_hash,
            input_summary=summary,
            status="failed",
            error=message,
        )
        raise LlmRequestFailed(message)

    def _interaction_payload(
        self,
        route_config: LlmRoute,
        *,
        audio_b64: str,
        instruction: str | None,
        language: str | None,
        custom_vocabulary: list[str] | None,
    ) -> dict[str, Any]:
        """Build one ``interactions:create`` body for a dedicated transcription model.

        Word timestamps and speaker diarization are always requested. They are the reason a
        route names this model rather than a chat model, and a transcript without them would be
        the more expensive way to buy what ``audio_chat`` already returns.
        """
        transcription: dict[str, Any] = {
            "timestampGranularities": ["word"],
            "diarizationMode": "speaker",
        }
        codes = list(route_config.language_codes)
        if not codes:
            hint = language or route_config.language
            codes = [hint] if hint else []
        if codes:
            transcription["languageCodes"] = codes
        if custom_vocabulary:
            transcription["customVocabulary"] = list(custom_vocabulary)

        generation: dict[str, Any] = {"transcriptionConfig": transcription}
        if route_config.temperature is not None:
            generation["temperature"] = route_config.temperature

        interaction: dict[str, Any] = {
            "modelInteraction": {
                "model": route_config.model,
                "generationConfig": generation,
            },
            "content": {
                "audio": {
                    "data": audio_b64,
                    "mime_type": AUDIO_MIME_TYPE,
                    "sampleRate": 16000,
                    "channels": 1,
                }
            },
        }
        # Verbatim transcription has no field on this API: the model is told in prose instead,
        # which is also where the no-transliteration rule has to go.
        if instruction:
            interaction["systemInstruction"] = instruction
        return {"interaction": interaction}

    @staticmethod
    def _generate_content_payload(
        route_config: LlmRoute,
        *,
        audio_b64: str,
        instruction: str | None,
        language: str | None,
    ) -> dict[str, Any]:
        """Build one ``generateContent`` body for a general model asked to transcribe."""
        target_lang = language or route_config.language
        text = instruction or ""
        if target_lang:
            hint = f"Primary language code: {target_lang}."
            text = f"{text}\n\n{hint}".strip() if text else hint

        parts: list[dict[str, Any]] = []
        if text:
            parts.append({"text": text})
        parts.append({"inline_data": {"mime_type": AUDIO_MIME_TYPE, "data": audio_b64}})

        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": route_config.temperature or 0.0,
                "responseModalities": ["TEXT"],
            },
        }
