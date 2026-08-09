"""
app/services/speech_service.py

Speech transcription service for Atlas AI Financial Assistant.

Transcribes Telegram voice messages (OGG/Opus audio) to plain text using
the Gemini API's native audio understanding, so users can speak their
financial questions instead of typing them.

Designed to be reusable from `app/bot/telegram_bot.py`'s voice message
handler (see the `transcribe_audio` call assumed there) as well as
directly from `app/langgraph/graph.py` if voice-originated context is
ever needed there.

NOTE ON ASSUMPTIONS:
- No dedicated speech-to-text provider was present in "Already
  Implemented". Gemini's multimodal audio input is used here since Gemini
  is already the project's LLM provider, avoiding a second API
  integration. TODO: swap this for a dedicated STT provider (e.g. Google
  Cloud Speech-to-Text) if transcription quality/cost/latency demands it.
- `app.config.settings` is assumed to expose `gemini_api_key: str` and
  optionally a `gemini_transcription_model_name: str`. Marked with TODOs.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional, Union

from app.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# TODO: Confirm attribute name on `settings`; assuming `gemini_api_key` is
# defined in app/config.py (shared with langgraph/graph.py's Gemini usage).
GEMINI_API_KEY: Final[str] = (
    settings.gemini_api_key.get_secret_value()
    if settings.gemini_api_key
    else ""
)

# Reuses the project's already-configured `settings.gemini_model` (the
# same model already proven working for chat generation elsewhere in this
# project) as the transcription model default. Gemini's flash-tier models
# are natively multimodal (audio input included), so a second,
# independently-versioned model name here serves no purpose and is a
# liability: it can silently drift out of date without anyone noticing,
# exactly as the prior hardcoded "gemini-2.5-flash" default did here.
#
# Still overridable via a `gemini_transcription_model_name` setting if one
# is ever added to app/config.py.
GEMINI_TRANSCRIPTION_MODEL_NAME: Final[str] = getattr(
    settings, "gemini_transcription_model_name", settings.gemini_model
)

DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
MAX_RETRIES: Final[int] = 2
RETRY_BACKOFF_SECONDS: Final[float] = 1.0

# Telegram voice notes are OGG/Opus by default.
DEFAULT_AUDIO_MIME_TYPE: Final[str] = "audio/ogg"

# TODO: Confirm actual policy; assuming a conservative 20 MB cap
# consistent with the PDF limit used in telegram_bot.py.
MAX_AUDIO_SIZE_BYTES: Final[int] = 20 * 1024 * 1024

_TRANSCRIPTION_PROMPT: Final[str] = (
    "Transcribe the following audio message exactly as spoken, in the "
    "original language. Return ONLY the transcription text, with no "
    "preamble, no quotation marks, and no commentary. If the audio is "
    "silent, unintelligible, or contains no speech, return an empty "
    "string."
)

try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai import types as genai_types  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - allows module import without the SDK
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SpeechServiceError(Exception):
    """Base exception for all speech transcription failures."""


class SpeechServiceConfigError(SpeechServiceError):
    """Raised when the service is used without a configured API key/SDK."""


class AudioTooLargeError(SpeechServiceError):
    """Raised when the provided audio exceeds `MAX_AUDIO_SIZE_BYTES`."""


class TranscriptionRequestError(SpeechServiceError):
    """Raised when the transcription backend call fails after retries."""


class EmptyTranscriptionError(SpeechServiceError):
    """Raised when transcription succeeds but yields no discernible speech."""


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TranscriptionResult:
    """Normalized result of a transcription request."""

    text: str
    mime_type: str
    source_size_bytes: int


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SpeechService:
    """
    Async service for transcribing voice audio to plain text via the
    Gemini API.

    Intended to be instantiated once (e.g. alongside the other services
    in `app/main.py`'s lifespan) and shared across requests.
    """

    def __init__(
        self,
        *,
        api_key: str = GEMINI_API_KEY,
        model_name: str = GEMINI_TRANSCRIPTION_MODEL_NAME,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_audio_size_bytes: int = MAX_AUDIO_SIZE_BYTES,
        client: Optional[Any] = None,
    ) -> None:
        """
        Args:
            api_key: Gemini API key. TODO: sourced from
                `settings.gemini_api_key`; raises
                `SpeechServiceConfigError` at call time if empty or the
                SDK is unavailable.
            model_name: Gemini model used for audio transcription.
            timeout_seconds: Per-request timeout applied via
                `asyncio.wait_for`.
            max_audio_size_bytes: Reject audio payloads larger than this.
            client: Optional pre-configured `genai.Client` (useful for
                testing/dependency injection). If omitted, one is
                constructed lazily on first use.
        """
        self._api_key = api_key
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._max_audio_size_bytes = max_audio_size_bytes
        self._client = client

    # -- client lifecycle ------------------------------------------------

    def _get_client(self) -> Any:
        if genai is None:
            raise SpeechServiceConfigError(
                "google-genai SDK is not installed; cannot transcribe audio."
            )
        if not self._api_key:
            raise SpeechServiceConfigError(
                "GEMINI_API_KEY is not configured. Set it in app.config.settings."
            )
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    # -- input normalization ----------------------------------------------

    @staticmethod
    async def _read_audio_bytes(source: Union[bytes, str, Path]) -> bytes:
        """
        Normalize a transcription input (raw bytes or a file path) into
        raw audio bytes.

        Raises:
            SpeechServiceError: if a path is given and the file cannot be
                read.
        """
        if isinstance(source, bytes):
            return source
        if isinstance(source, bytearray):
            return bytes(source)

        path = Path(source)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise SpeechServiceError(
                f"Failed to read audio file at {path!s}: {exc}"
            ) from exc

    @staticmethod
    def _infer_mime_type(source: Union[bytes, str, Path], explicit: Optional[str]) -> str:
        """Infer an audio MIME type from a file extension, or fall back to OGG."""
        if explicit:
            return explicit

        if isinstance(source, (str, Path)):
            suffix = Path(source).suffix.lower()
            mime_by_suffix = {
                ".ogg": "audio/ogg",
                ".oga": "audio/ogg",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".m4a": "audio/mp4",
                ".flac": "audio/flac",
                ".aac": "audio/aac",
            }
            if suffix in mime_by_suffix:
                return mime_by_suffix[suffix]

        return DEFAULT_AUDIO_MIME_TYPE

    # -- core transcription ------------------------------------------------

    async def _call_gemini_transcription(self, audio_bytes: bytes, mime_type: str) -> str:
        """
        Perform the actual Gemini API call to transcribe audio, with
        timeout + retry handling.

        Raises:
            TranscriptionRequestError: on failure after exhausting retries.
        """
        client = self._get_client()

        if genai_types is None:
            raise SpeechServiceConfigError(
                "google-genai SDK is not installed; cannot transcribe audio."
            )

        audio_part = genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 2):  # initial attempt + retries
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=self._model_name,
                        contents=[_TRANSCRIPTION_PROMPT, audio_part],
                    ),
                    timeout=self._timeout_seconds,
                )
                if response is not None and response.text:
                    return response.text.strip()
                return ""
            except asyncio.TimeoutError as exc:
                last_exc = exc
                if attempt <= MAX_RETRIES:
                    logger.warning(
                        "Gemini transcription timed out (attempt %d); retrying.",
                        attempt,
                    )
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                logger.error("Gemini transcription timed out after retries.")
                raise TranscriptionRequestError(
                    "Transcription request timed out after retries."
                ) from exc
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt <= MAX_RETRIES:
                    logger.warning(
                        "Gemini transcription failed (attempt %d): %s; retrying.",
                        attempt,
                        exc,
                    )
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                logger.exception("Gemini transcription failed after retries.")
                raise TranscriptionRequestError(
                    f"Transcription request failed after {MAX_RETRIES} "
                    f"retries: {exc}"
                ) from exc

        # Unreachable, but keeps type checkers happy.
        raise TranscriptionRequestError(
            "Transcription request failed unexpectedly."
        ) from last_exc

    # -- public API ------------------------------------------------------

    async def transcribe(
        self,
        source: Union[bytes, str, Path],
        *,
        mime_type: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio source (raw bytes or a file path) to plain
        text.

        Args:
            source: Either raw audio bytes, or a path (str/Path) to an
                audio file on disk.
            mime_type: Explicit MIME type override. If omitted, inferred
                from the file extension when `source` is a path, or
                defaults to `audio/ogg` (Telegram voice notes) otherwise.

        Returns:
            A `TranscriptionResult` containing the transcribed text.

        Raises:
            SpeechServiceConfigError: if the service is misconfigured
                (missing API key/SDK).
            AudioTooLargeError: if the audio exceeds the configured size
                limit.
            SpeechServiceError: if reading a file path source fails.
            TranscriptionRequestError: if the backend call fails after
                retries.
            EmptyTranscriptionError: if transcription succeeds but no
                speech is detected.
        """
        audio_bytes = await self._read_audio_bytes(source)

        if len(audio_bytes) == 0:
            raise SpeechServiceError("Received empty audio payload; nothing to transcribe.")

        if len(audio_bytes) > self._max_audio_size_bytes:
            raise AudioTooLargeError(
                f"Audio payload ({len(audio_bytes)} bytes) exceeds the "
                f"{self._max_audio_size_bytes}-byte limit."
            )

        resolved_mime_type = self._infer_mime_type(source, mime_type)

        text = await self._call_gemini_transcription(audio_bytes, resolved_mime_type)

        if not text:
            logger.info("Transcription returned no discernible speech.")
            raise EmptyTranscriptionError("No speech detected in the provided audio.")

        logger.info(
            "Transcribed %d bytes of %s audio into %d characters of text.",
            len(audio_bytes),
            resolved_mime_type,
            len(text),
        )

        return TranscriptionResult(
            text=text,
            mime_type=resolved_mime_type,
            source_size_bytes=len(audio_bytes),
        )

    async def transcribe_audio(self, source: Union[bytes, str, Path]) -> str:
        """
        Convenience wrapper returning plain transcribed text only.

        This matches the interface assumed by
        `app.bot.telegram_bot.handle_voice_message`, which calls
        `document_service.transcribe_audio(...)` today.

        TODO: `telegram_bot.py` currently calls `transcribe_audio` on
        `DocumentService` rather than this `SpeechService`. Update
        `telegram_bot.py`'s voice handler to depend on `SpeechService`
        instead (constructed in `app/main.py`'s lifespan alongside the
        other services) so transcription lives in one place.

        Returns:
            The transcribed plain text. Returns an empty string (rather
            than raising) if no speech was detected, so callers can
            treat "nothing said" as a soft case without a try/except.

        Raises:
            SpeechServiceConfigError: if misconfigured.
            AudioTooLargeError: if the audio is too large.
            SpeechServiceError: if reading a file path source fails.
            TranscriptionRequestError: if the backend call fails after
                retries.
        """
        try:
            result = await self.transcribe(source)
        except EmptyTranscriptionError:
            return ""
        return result.text