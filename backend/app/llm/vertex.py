"""Vertex AI client for the Gemini speech-to-text models.

Google serves the same Gemini models from two surfaces that are not interchangeable, and telling
them apart is the whole reason this module exists (D39):

* AI Studio (``generativelanguage.googleapis.com``) calls the recogniser ``gemini-3.5-transcribe``
  and drives it through ``POST /v1beta/interactions``. Its quota is per API key and cannot be
  raised, which is what defeated D29, D30 and D38.
* Vertex AI (``aiplatform.googleapis.com``) calls the same recogniser
  ``gemini-3.5-transcribe-preview`` -- the ``-preview`` suffix is not optional, and asking for the
  AI Studio id here is a 404 in every region -- and drives it through the ordinary
  ``:generateContent`` endpoint with an ``audioTranscriptionConfig``. There is no Interactions API
  on Vertex. Quota is project-scoped and can be raised.

Both request shapes here are ``:generateContent``; only the body differs:

* ``api: transcription`` sends ``generationConfig.audioTranscriptionConfig`` and gets back word
  spans and a speaker label per segment. The model is a dedicated recogniser and **rejects**
  ``systemInstruction`` outright (``400 The input system_instruction is not supported.``), so
  ``audioTranscriptionConfig`` is the only steering it accepts -- see ``transcription.py``.
* ``api: audio_chat`` inlines the clip and takes a ``systemInstruction``. The model is a general
  LLM being asked to transcribe: it obeys the transcript policy, returns no timings, and may
  editorialise or hallucinate over silence. Spans come from the local forced aligner (D32).

Authentication is a single API key restricted to ``aiplatform.googleapis.com``, sent as an
``x-goog-api-key`` header rather than a ``?key=`` query parameter so that the secret cannot reach
a log line or an httpx error message.
"""

from __future__ import annotations

import base64
import os
from decimal import Decimal
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
from app.llm.cost import calculate_vertex_cost, get_audio_duration
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Environment variables consulted for the Vertex API key.
PRIMARY_API_KEY_ENV = "VERTEX_API_KEY"
FALLBACK_API_KEY_ENV = "GOOGLE_API_KEY"

#: Environment fallbacks for the project and location, when the routing table leaves them blank.
PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"
LOCATION_ENV = "GOOGLE_CLOUD_LOCATION"

#: Serves both models, and the only location that serves the recogniser besides ``us-central1``.
DEFAULT_LOCATION = "global"

#: Clips are FLAC by invariant 6.
AUDIO_MIME_TYPE = "audio/flac"

#: Harm categories to turn off for speech recognition.
#:
#: A transcriber must not decline to write down what was said. Safety thresholds are set to OFF
#: so that audio is not blocked by prompt filters.
_HARM_CATEGORIES = (
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
)
SAFETY_SETTINGS = [{"category": c, "threshold": "OFF"} for c in _HARM_CATEGORIES]


def _parts(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Every part of every candidate, in order."""
    out: list[dict[str, Any]] = []
    for candidate in body.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if isinstance(part, dict):
                out.append(part)
    return out


def parse_generate_content(body: dict[str, Any]) -> str:
    """Join the text parts of the first ``generateContent`` candidate that has any."""
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


def _duration_seconds(value: Any) -> float | None:
    """Read a duration (``"1.75s"`` or numeric seconds) as float. Absent/unparseable means None."""
    if value is None:
        return None
    try:
        return float(str(value).rstrip("s"))
    except ValueError:
        return None


def parse_transcription(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
    """Read the transcript and word spans out of an ``audioTranscription`` response.

    Vertex returns one ``Part`` per speaker segment, each carrying its own ``speakerLabel`` and
    its own ``words``. The label is therefore per segment, not per word, so it is fanned down onto
    every word of its segment: ``hypothesis_words.speaker`` is a per-word column, and the
    comparison it exists for is *within* one clip (D36).

    Returns:
        ``(text, words)``. ``words`` is ``None`` when the response carried no word spans, never an
        empty list -- the scorer reads ``None`` as "no signal".
    """
    texts: list[str] = []
    words: list[dict[str, Any]] = []

    for part in _parts(body):
        transcription = part.get("audioTranscription") or part.get("audio_transcription")
        if not isinstance(transcription, dict):
            # A part with no transcription payload still carries the plain text of the segment.
            if part.get("text"):
                texts.append(str(part["text"]))
            continue

        # Prefer the transcription's own text: the sibling ``part["text"]`` repeats it verbatim,
        # and taking both would double every segment.
        segment_text = transcription.get("text") or part.get("text")
        if segment_text:
            texts.append(str(segment_text))

        speaker = transcription.get("speakerLabel") or transcription.get("speaker_label")
        for word in transcription.get("words") or []:
            if not isinstance(word, dict):
                continue
            token = word.get("word") or word.get("text")
            if not token:
                continue
            words.append(
                {
                    "word": str(token),
                    "start": _duration_seconds(word.get("startOffset") or word.get("start_offset")),
                    "end": _duration_seconds(word.get("endOffset") or word.get("end_offset")),
                    "speaker": speaker,
                }
            )

    return " ".join(t.strip() for t in texts if t.strip()).strip(), words or None


class VertexClient(ProviderClient):
    """A logged, retrying client for Gemini speech models on Vertex AI."""

    def __init__(
        self,
        session: Session,
        *,
        config: LlmRoutes | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(session, config=config, client=client)
        self.api_key = (
            api_key
            or os.environ.get(PRIMARY_API_KEY_ENV)
            or os.environ.get(FALLBACK_API_KEY_ENV)
            or ""
        )
        self.base_url = (self.config.vertex_base_url or "").rstrip("/")
        self.project = self.config.vertex_project or os.environ.get(PROJECT_ENV, "")
        self.location = (
            self.config.vertex_location or os.environ.get(LOCATION_ENV) or DEFAULT_LOCATION
        )

    def _model_url(self, model_id: str) -> str:
        """The project-scoped publisher endpoint, which bills and quotas against the project."""
        return (
            f"{self.base_url}/projects/{self.project}/locations/{self.location}"
            f"/publishers/google/models/{model_id}:generateContent"
        )

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
        """Transcribe one clip with whichever Gemini model and body shape the route names.

        Args:
            audio_path: Path to the audio file (16 kHz mono FLAC).
            route: Route name from ``config/llm_routes.yaml``.
            prompt: Transcript policy, for ``audio_chat`` only. The dedicated recogniser rejects
                ``systemInstruction``, so callers pass ``None`` for an ``api: transcription``
                route and steer it through ``custom_vocabulary`` instead.
            language: BCP-47 language hint overriding route config.
            custom_vocabulary: Ignored, and warned about. Vertex takes it beside diarization
                without complaint and then returns no speaker labels at all -- see
                ``_transcription_payload``.
            dry_run: Override configured dry-run mode.
            timeout_seconds: Override default timeout.

        Returns:
            The transcription. ``words`` is populated only on an ``api: transcription`` route;
            ``audio_chat`` returns text, and its spans come from the forced aligner (D32).
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
                cost=Decimal("0.0"),
            )
            logger.info("vertex_dry_run", route=route, model=model_id, file=str(audio_file))
            text, words = dry_run_transcript(request_hash)
            return AsrResult(
                route=route,
                model=model_id,
                text=text,
                words=words,
                dry_run=True,
                estimated_cost_usd=Decimal("0.0"),
            )

        if not self.config.enabled:
            raise LlmDisabledError(
                "LLM inference is disabled (config/llm_routes.yaml: enabled: false)"
            )

        if not self.api_key:
            self._fail(
                route,
                model_id,
                request_hash,
                summary,
                f"no Vertex API key: set {PRIMARY_API_KEY_ENV} in .env or environment",
            )

        if not self.project:
            self._fail(
                route,
                model_id,
                request_hash,
                summary,
                f"no GCP project: set vertex_project in llm_routes.yaml or {PROJECT_ENV}",
            )

        timeout = (
            timeout_seconds or route_config.timeout_seconds or self.config.default_timeout_seconds
        )
        audio_b64 = base64.b64encode(audio_file.read_bytes()).decode("ascii")
        wants_words = route_config.api == "transcription"

        if wants_words:
            payload = self._transcription_payload(
                route_config,
                audio_b64=audio_b64,
                language=language,
                custom_vocabulary=custom_vocabulary,
            )
        else:
            payload = self._generate_content_payload(
                route_config,
                audio_b64=audio_b64,
                instruction=prompt,
                language=language,
            )

        url = self._model_url(model_id)
        # The key rides in a header, never the URL: httpx puts the URL in its error strings and
        # `_send_with_retries` copies those into `llm_requests.error_message`.
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        def send() -> httpx.Response:
            return self._get_client().post(url, json=payload, headers=headers, timeout=timeout)

        # A 200 carrying no transcript is not a success, and Gemini returns one in two measured
        # cases (D41): a spurious `blockReason: SAFETY` on `audio_chat`, which clears on a retry
        # with the request unchanged, and a recogniser clip past the duration limit that multiple
        # language codes impose. Neither is visible to `_send_with_retries`, which only ever sees
        # a 200, so the emptiness is judged here and retried from here.
        #
        # An empty `audio_chat` answer with no block reason is left alone: ASR_PROMPT asks for an
        # empty string when there is no intelligible speech, so that one is the model obeying.
        response: httpx.Response | None = None
        body: dict[str, Any] = {}
        text, words = "", None
        latency_ms, last_error = 0, "no attempt was made"

        for attempt in range(max(1, self.config.max_retries)):
            response, last_error, latency_ms = self._send_with_retries(send)
            if response is None:
                break
            body = response.json()
            if wants_words:
                text, words = parse_transcription(body)
            else:
                text, words = parse_generate_content(body), None

            block_reason = (body.get("promptFeedback") or {}).get("blockReason")
            if text.strip() or (not wants_words and not block_reason):
                break

            last_error = f"HTTP 200 with an empty transcript (blockReason={block_reason!r})"
            logger.warning(
                "vertex_empty_transcript",
                route=route,
                model=model_id,
                attempt=attempt + 1,
                reason=block_reason,
            )
            response = None

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
            raise LlmRequestFailed(f"Vertex transcription failed: {last_error}")

        usage = body.get("usageMetadata") or body.get("usage_metadata") or {}
        prompt_tokens = usage.get("promptTokenCount") or usage.get("prompt_token_count")
        completion_tokens = usage.get("candidatesTokenCount") or usage.get("candidates_token_count")
        duration = get_audio_duration(audio_file)
        cost = calculate_vertex_cost(
            model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_seconds=duration,
        )

        self._log(
            route=route,
            model=model_id,
            request_hash=request_hash,
            input_summary=summary,
            status="succeeded",
            output=body,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
        )
        return AsrResult(
            route=route,
            model=model_id,
            text=text,
            words=words,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
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

    @staticmethod
    def _language_codes(route_config: LlmRoute, language: str | None) -> list[str]:
        """Every BCP-47 code to send, falling back to the single ``language`` hint."""
        codes = list(route_config.language_codes)
        if codes:
            return codes
        hint = language or route_config.language
        return [hint] if hint else []

    @classmethod
    def _transcription_payload(
        cls,
        route_config: LlmRoute,
        *,
        audio_b64: str,
        language: str | None,
        custom_vocabulary: list[str] | None,
    ) -> dict[str, Any]:
        """Build a ``generateContent`` body for the dedicated recogniser.

        No ``systemInstruction`` and no text part: Vertex rejects the first with a 400 and ignores
        the second, so ``audioTranscriptionConfig`` carries all the steering there is. Only the
        current fields are sent -- ``languageHints``, ``languageAuto``, ``adaptationPhrases``,
        ``timestampGranularities`` and ``diarizationMode`` are all deprecated.

        ``customVocabulary`` is dropped rather than sent. Vertex accepts it beside ``diarization``
        with a 200 -- where AI Studio at least answers 400 -- and then silently returns no
        ``speakerLabel`` at all. Measured on one clip, three runs each: with vocabulary every
        segment came back unlabelled, without it every segment was labelled ``spk:0``. Speaker
        labels are the reason this route exists, and trading them for term biasing that does not
        even fix the script would be the expensive way to buy nothing (D36, D39).
        """
        transcription_config: dict[str, Any] = {
            "diarization": True,
            "wordTimestamp": True,
        }

        codes = cls._language_codes(route_config, language)
        if codes:
            transcription_config["languageCodes"] = codes
        if custom_vocabulary:
            logger.warning(
                "vertex_custom_vocabulary_dropped",
                reason="Vertex silently suppresses speakerLabel when customVocabulary is sent",
                terms=len(custom_vocabulary),
            )

        generation_config: dict[str, Any] = {"audioTranscriptionConfig": transcription_config}
        if route_config.temperature is not None:
            generation_config["temperature"] = route_config.temperature

        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"inlineData": {"mimeType": AUDIO_MIME_TYPE, "data": audio_b64}}],
                }
            ],
            "generationConfig": generation_config,
        }

    @staticmethod
    def _generate_content_payload(
        route_config: LlmRoute,
        *,
        audio_b64: str,
        instruction: str | None,
        language: str | None,
    ) -> dict[str, Any]:
        """Build a ``generateContent`` body for a general model asked to transcribe."""
        target_lang = language or route_config.language
        text = instruction or ""
        if target_lang:
            hint = f"Primary language code: {target_lang}."
            text = f"{text}\n\n{hint}".strip() if text else hint

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"inlineData": {"mimeType": AUDIO_MIME_TYPE, "data": audio_b64}}],
                }
            ],
            "generationConfig": {
                "temperature": route_config.temperature or 0.0,
                "responseModalities": ["TEXT"],
            },
            "safetySettings": SAFETY_SETTINGS,
        }
        if text:
            payload["systemInstruction"] = {"parts": [{"text": text}]}
        return payload
