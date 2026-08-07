"""
app/database/models.py

SQLAlchemy 2.0 ORM models for the application.

This module defines the declarative `Base` class and all mapped models
for the finalized MVP schema:
    - User: application/bot users, including bot preferences and
      personalization settings.
    - ConversationMessage: chat history exchanged between a user and the bot.
    - Document: source documents uploaded/ingested for a user (e.g. for RAG).
    - DocumentChunk: chunked, embeddable segments of a Document, with a
      pgvector embedding column for similarity search.
    - DailyBriefLog: record of daily brief generations/deliveries per user.

Backing store: PostgreSQL with the `pgvector` extension (per the finalized
architecture). The extension must be enabled on the target database
(`CREATE EXTENSION IF NOT EXISTS vector;`) before `document_chunks` can be
created/migrated.

This file contains ONLY model/schema definitions. No CRUD, no session
logic, no business logic.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, time

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Time,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

# Dimensionality of the embedding vectors stored in `document_chunks`.
# Matches the output dimension of Gemini's `text-embedding-004` model,
# which is the embedding model used by the application's ingestion
# pipeline. Update if the embedding model choice changes.
EMBEDDING_DIM = 768


class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.

    All mapped classes in the application must inherit from this base so
    their metadata is collected together (e.g. for `Base.metadata.create_all`
    or Alembic autogeneration).
    """

    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class UserRole(str, enum.Enum):
    """Access role of a user within the application."""

    ADMIN = "admin"
    USER = "user"


class MessageRole(str, enum.Enum):
    """Role of the sender of a conversation message."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class BriefStatus(str, enum.Enum):
    """Delivery status of a generated daily brief."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------
class User(Base):
    """
    An application user (bot end-user).

    Acts as the root entity that conversation messages, documents, and
    daily brief logs belong to. Stores the user's bot-facing personalization
    and delivery preferences (followed companies/sectors, insight
    preferences, brief delivery time/timezone) in addition to identity and
    onboarding state.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )

    # Identity
    telegram_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, nullable=False, index=True
    )
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role"),
        nullable=False,
        default=UserRole.USER,
        server_default=text("'USER'"),
    )

    # Personalization / preferences
    followed_companies: Mapped[list | dict | None] = mapped_column(
        JSONB,
        nullable=True,
        doc="List of companies/tickers the user follows.",
    )
    followed_sectors: Mapped[list | dict | None] = mapped_column(
        JSONB,
        nullable=True,
        doc="List of sectors/industries the user follows.",
    )
    insight_preferences: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        doc="User-configurable preferences for insight/brief content and format.",
    )
    brief_time: Mapped[time | None] = mapped_column(
        Time,
        nullable=True,
        doc="Local time of day the user's daily brief should be delivered.",
    )
    timezone: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc="IANA timezone name (e.g. 'Asia/Kolkata') used to interpret brief_time.",
    )
    notes: Mapped[str | None] = mapped_column(
        Text, nullable=True, doc="Freeform internal/user notes."
    )
    onboarding_completed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    conversation_messages: Mapped[list["ConversationMessage"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    documents: Mapped[list["Document"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    daily_brief_logs: Mapped[list["DailyBriefLog"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"<User id={self.id} telegram_id={self.telegram_id}>"


# ---------------------------------------------------------------------------
# ConversationMessage
# ---------------------------------------------------------------------------
class ConversationMessage(Base):
    """
    A single message in a user's conversation history with the bot.

    Stores both user-authored and assistant-authored turns so the full
    dialogue can be reconstructed for context windows, auditing, or
    analytics. Indexed on (user_id, created_at) to efficiently fetch a
    user's recent message history in order.
    """

    __tablename__ = "conversation_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, name="message_role"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversation_messages")

    __table_args__ = (
        Index(
            "ix_conversation_messages_user_id_created_at",
            "user_id",
            "created_at",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return (
            f"<ConversationMessage id={self.id} user_id={self.user_id} "
            f"role={self.role}>"
        )


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
class Document(Base):
    """
    A source document belonging to a user (e.g. uploaded file or ingested
    text) that has been or will be chunked for retrieval.

    Tracks basic metadata about the document; the actual retrievable
    content lives in the related `DocumentChunk` rows.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str | None] = mapped_column(
        String(1000), nullable=True, doc="Origin of the document (URL, filename, etc.)"
    )
    is_processed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentChunk.chunk_index",
    )

    __table_args__ = (
        Index("ix_documents_user_id_created_at", "user_id", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"<Document id={self.id} user_id={self.user_id} title={self.title!r}>"


# ---------------------------------------------------------------------------
# DocumentChunk
# ---------------------------------------------------------------------------
class DocumentChunk(Base):
    """
    A chunked, embeddable segment of a `Document`.

    Each chunk holds a slice of the parent document's text along with its
    position (`chunk_index`) so ordering can be preserved, and an
    `embedding` column (pgvector `Vector`) holding the vectorized
    representation used for similarity search. A uniqueness constraint on
    (document_id, chunk_index) prevents duplicate/overlapping chunk
    numbering for the same document.
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="pgvector embedding of `content`, dimension EMBEDDING_DIM.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint(
            "document_id", "chunk_index", name="uq_document_chunks_document_id_chunk_index"
        ),
        Index("ix_document_chunks_document_id", "document_id"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return (
            f"<DocumentChunk id={self.id} document_id={self.document_id} "
            f"chunk_index={self.chunk_index}>"
        )


# ---------------------------------------------------------------------------
# DailyBriefLog
# ---------------------------------------------------------------------------
class DailyBriefLog(Base):
    """
    Log entry recording the generation/delivery of a daily brief for a user.

    Used for auditing what was sent (or attempted) to a user on a given
    day, and for troubleshooting delivery failures via `status`/`error_message`.
    A uniqueness constraint on (user_id, brief_date) ensures at most one
    log entry per user per day.
    """

    __tablename__ = "daily_brief_log"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    brief_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="Calendar date (UTC) the brief was generated for.",
    )
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[BriefStatus] = mapped_column(
        Enum(BriefStatus, name="brief_status"),
        nullable=False,
        default=BriefStatus.PENDING,
        server_default=text("'PENDING'"),
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="daily_brief_logs")

    __table_args__ = (
        UniqueConstraint(
            "user_id", "brief_date", name="uq_daily_brief_log_user_id_brief_date"
        ),
        Index("ix_daily_brief_log_user_id_brief_date", "user_id", "brief_date"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return (
            f"<DailyBriefLog id={self.id} user_id={self.user_id} "
            f"brief_date={self.brief_date} status={self.status}>"
        )