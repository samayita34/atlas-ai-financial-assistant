"""
app/config.py

Centralized application configuration.

All environment-dependent values (API keys, database URL, scheduling defaults,
runtime flags) are declared here as a single Pydantic `Settings` object and
loaded exactly once at import time. No other module in the codebase should
call `os.environ` directly — everything goes through `get_settings()` so
there is one source of truth and one place to change if a variable is
renamed or a default changes.

Fails fast: if a required variable is missing or malformed, the app raises
on startup instead of failing later, mid-conversation, with a confusing
downstream error (e.g. a bad DATABASE_URL surfacing as an opaque
SQLAlchemy connection error three layers deep).
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment. Used to toggle logging verbosity, docs
    exposure, and other environment-specific behavior in main.py."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Restricts LOG_LEVEL to values Python's logging module actually
    understands, instead of accepting an arbitrary string."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """
    Application settings, populated from environment variables (and a local
    `.env` file in development). See `.env.example` for the full list of
    variables this expects.

    SecretStr is used for anything sensitive (API keys, tokens) so secrets
    are never accidentally leaked into logs, tracebacks, or repr() output —
    printing a Settings instance shows `SecretStr('**********')`, not the
    real value. Call `.get_secret_value()` explicitly when the raw value is
    actually needed (e.g. handing it to an SDK client).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Reject unknown environment variables typo'd into .env instead of
        # silently ignoring them (e.g. "GEMENI_API_KEY" would otherwise
        # fail silently and leave GEMINI_API_KEY unset).
        extra="forbid",
    )

    # ------------------------------------------------------------------
    # Core / runtime
    # ------------------------------------------------------------------
    app_name: str = Field(
        default="Atlas Financial Assistant",
        description="Human-readable app name, used in logs and startup banner.",
    )
    environment: Environment = Field(
        default=Environment.LOCAL,
        description="Deployment environment. Controls debug behavior in main.py.",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Root logging level for utils/logging.py.",
    )

    # ------------------------------------------------------------------
    # Telegram
    # ------------------------------------------------------------------
    telegram_bot_token: SecretStr = Field(
        ...,
        description="Bot token from @BotFather. Required — the app cannot start without it.",
    )

    @field_validator("telegram_bot_token")
    @classmethod
    def validate_telegram_token_format(cls, v: SecretStr) -> SecretStr:
        """Telegram bot tokens always follow `<bot_id>:<auth_string>`, e.g.
        `123456789:AAExampleTokenStringHere`. Catching an obviously
        malformed token here is cheaper than discovering it after aiogram
        fails to authenticate against the Telegram API on startup."""
        token = v.get_secret_value()
        if ":" not in token or not token.split(":")[0].isdigit():
            raise ValueError(
                "telegram_bot_token must look like '<numeric_id>:<auth_string>' "
                "(the token format issued by @BotFather)."
            )
        return v

    # ------------------------------------------------------------------
    # LLM / AI providers
    # ------------------------------------------------------------------
    gemini_api_key: SecretStr = Field(
        ...,
        description="Google AI Studio API key used for all Gemini calls in agent/.",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model id used for agent reasoning and generation.",
    )
    embedding_model: str = Field(
        default="text-embedding-004",
        description="Embedding model id used by rag/embeddings.py for document and query embeddings.",
    )
    transcription_api_key: SecretStr | None = Field(
        default=None,
        description="Optional API key for an external speech-to-text provider (e.g. OpenAI Whisper).",
    )

    # ------------------------------------------------------------------
    # Financial data providers
    # ------------------------------------------------------------------
    finnhub_api_key: SecretStr = Field(
        ...,
        description="Finnhub API key — the single source of truth for quotes, news, and earnings data.",
    )
    sec_edgar_user_agent: str = Field(
        default="AtlasFinancialAssistant contact@example.com",
        description=(
            "SEC EDGAR requires a descriptive User-Agent header identifying the "
            "caller on every request, or it will reject the request. Set this to "
            "a real app name + contact email before deploying."
        ),
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: PostgresDsn = Field(
        ...,
        description=(
            "Async Postgres connection string, e.g. "
            "'postgresql+asyncpg://user:password@host:5432/atlas'. "
            "Must use the asyncpg driver since db/database.py uses an async engine."
        ),
    )

    @field_validator("database_url")
    @classmethod
    def validate_async_driver(cls, v: PostgresDsn) -> PostgresDsn:
        """The app uses SQLAlchemy's async engine end to end. A DSN using the
        sync 'postgresql://' scheme (missing '+asyncpg') will construct a
        sync engine that silently breaks every 'await session...' call
        downstream — surface that mistake here instead."""
        if v.scheme != "postgresql+asyncpg":
            raise ValueError(
                f"database_url must use the 'postgresql+asyncpg' scheme for "
                f"async SQLAlchemy, got '{v.scheme}'. "
                f"Example: postgresql+asyncpg://user:pass@host:5432/dbname"
            )
        return v

    db_pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="SQLAlchemy async engine connection pool size.",
    )

    # ------------------------------------------------------------------
    # Scheduler / proactive briefings
    # ------------------------------------------------------------------
    default_brief_timezone: str = Field(
        default="UTC",
        description=(
            "Fallback IANA timezone (e.g. 'America/New_York') used for a user's "
            "daily brief cron job when no per-user timezone has been captured."
        ),
    )
    default_brief_time: str = Field(
        default="08:00",
        description="Fallback brief send time in 24h 'HH:MM' format, used until a user sets their own.",
    )

    @field_validator("default_brief_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Cheap sanity check so a malformed default never reaches
        APScheduler's cron trigger parsing at job-registration time."""
        try:
            hour_str, minute_str = v.split(":")
            hour, minute = int(hour_str), int(minute_str)
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"default_brief_time must be 'HH:MM' 24h format, got '{v}'"
            ) from exc
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"default_brief_time out of range: '{v}'")
        return v

    # ------------------------------------------------------------------
    # RAG / document ingestion
    # ------------------------------------------------------------------
    rag_chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Target token size per document chunk during ingestion.",
    )
    rag_chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Token overlap between consecutive chunks to preserve context across chunk boundaries.",
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks retrieved per document question in rag/retriever.py.",
    )

    @field_validator("rag_chunk_overlap")
    @classmethod
    def validate_overlap_smaller_than_chunk(cls, v: int, info) -> int:
        """An overlap >= chunk size would make chunking loop or duplicate
        content indefinitely — invalid by construction."""
        chunk_size = info.data.get("rag_chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError(
                f"rag_chunk_overlap ({v}) must be smaller than rag_chunk_size ({chunk_size})."
            )
        return v

    # ------------------------------------------------------------------
    # Conversation memory
    # ------------------------------------------------------------------
    conversation_history_window: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of most recent messages loaded as short-term memory for each agent turn.",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns the process-wide `Settings` singleton.

    `lru_cache` ensures the environment is parsed and validated exactly once
    per process, not once per import — every module should call
    `get_settings()` rather than instantiating `Settings()` directly, so
    there is a single validated instance shared across the app.

    Usage:
        from app.config import get_settings
        settings = get_settings()
    """
    return Settings()
