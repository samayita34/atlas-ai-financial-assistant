"""
app/services/memory_service.py

Memory service: owns all long-term user memory.

This service is the single place responsible for reading and writing
`User` and `ConversationMessage` state. It encapsulates:
    - User creation/lookup by Telegram ID.
    - Onboarding state updates.
    - Conversation history storage and retrieval.
    - Personalization field updates (followed companies/sectors, insight
      preferences, notes, onboarding_completed).

Design notes:
    - This module contains ONLY business logic. It knows nothing about
      FastAPI routes or Telegram handlers — callers (API routes, bot
      handlers, schedulers, etc.) are expected to obtain an `AsyncSession`
      (e.g. via `app.database.database.get_db`) and pass it in.
    - Per the project's session-handling convention, this service is
      responsible for committing its own units of work. The session
      dependency itself does not auto-commit.
    - Methods that mutate state flush (and commit) so callers get back
      fully persisted, ID-populated objects.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ConversationMessage, MessageRole, User


class MemoryService:
    """
    Service encapsulating all long-term user memory operations.

    Instances are lightweight wrappers around a single `AsyncSession` and
    are intended to be constructed per request/unit-of-work, e.g.:

        async def handler(db: AsyncSession = Depends(get_db)):
            memory = MemoryService(db)
            user = await memory.get_or_create_user(telegram_id=123)
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the service with an active async session.

        Args:
            session: An `AsyncSession` bound to the application's engine.
                The caller owns the session's lifecycle (creation and
                closing); this service only reads from and writes to it.
        """
        self._session = session

    # -------------------------------------------------------------------
    # User lookup / creation
    # -------------------------------------------------------------------
    async def get_user_by_telegram_id(self, telegram_id: int) -> User | None:
        """
        Retrieve a user's profile by their Telegram ID.

        Args:
            telegram_id: The user's external Telegram identifier.

        Returns:
            The matching `User`, or `None` if no such user exists.
        """
        stmt = select(User).where(User.telegram_id == telegram_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_or_create_user(
        self,
        telegram_id: int,
        username: str | None = None,
        full_name: str | None = None,
    ) -> User:
        """
        Fetch a user by Telegram ID, creating one if it does not exist.

        If the user already exists, their record is returned as-is (this
        method does not overwrite `username`/`full_name` on an existing
        user — use an explicit update for that).

        Args:
            telegram_id: The user's external Telegram identifier.
            username: Optional Telegram username, used only on creation.
            full_name: Optional display name, used only on creation.

        Returns:
            The existing or newly created `User`.
        """
        existing_user = await self.get_user_by_telegram_id(telegram_id)
        if existing_user is not None:
            return existing_user

        user = User(
            telegram_id=telegram_id,
            username=username,
            full_name=full_name,
        )
        self._session.add(user)
        try:
            await self._session.commit()
        except IntegrityError:
            # Concurrent creation for the same telegram_id: back off and
            # fetch the row the other transaction committed.
            await self._session.rollback()
            existing_user = await self.get_user_by_telegram_id(telegram_id)
            if existing_user is None:
                raise
            return existing_user

        await self._session.refresh(user)
        return user

    # -------------------------------------------------------------------
    # Onboarding
    # -------------------------------------------------------------------
    async def update_onboarding(
        self,
        user_id: Any,
        *,
        followed_companies: list[str] | None = None,
        followed_sectors: list[str] | None = None,
        insight_preferences: dict[str, Any] | None = None,
        brief_time: Any = None,
        timezone: str | None = None,
        onboarding_completed: bool | None = None,
    ) -> User:
        """
        Update onboarding-related fields for a user in a single call.

        Only fields explicitly passed (non-`None`) are updated, so this
        can be called incrementally as a user progresses through an
        onboarding flow without clobbering fields already set.

        Args:
            user_id: Primary key of the user to update.
            followed_companies: Companies/tickers the user selected.
            followed_sectors: Sectors/industries the user selected.
            insight_preferences: Content/format preferences for briefs.
            brief_time: Local time of day to deliver the daily brief.
            timezone: IANA timezone name for interpreting `brief_time`.
            onboarding_completed: Whether onboarding has been completed.

        Returns:
            The updated `User`.

        Raises:
            ValueError: If no user exists with the given `user_id`.
        """
        user = await self._get_user_by_id(user_id)

        if followed_companies is not None:
            user.followed_companies = followed_companies
        if followed_sectors is not None:
            user.followed_sectors = followed_sectors
        if insight_preferences is not None:
            user.insight_preferences = insight_preferences
        if brief_time is not None:
            user.brief_time = brief_time
        if timezone is not None:
            user.timezone = timezone
        if onboarding_completed is not None:
            user.onboarding_completed = onboarding_completed

        await self._session.commit()
        await self._session.refresh(user)
        return user

    # -------------------------------------------------------------------
    # Personalization
    # -------------------------------------------------------------------
    async def update_personalization(
        self,
        user_id: Any,
        *,
        followed_companies: list[str] | None = None,
        followed_sectors: list[str] | None = None,
        insight_preferences: dict[str, Any] | None = None,
        notes: str | None = None,
        onboarding_completed: bool | None = None,
    ) -> User:
        """
        Update a user's personalization fields.

        Only fields explicitly passed (non-`None`) are updated. Use this
        for post-onboarding preference changes (e.g. a user adding a
        followed company from a chat command); use `update_onboarding`
        for the initial onboarding flow if you also need to set
        `brief_time`/`timezone`.

        Args:
            user_id: Primary key of the user to update.
            followed_companies: Replacement list of followed companies/tickers.
            followed_sectors: Replacement list of followed sectors/industries.
            insight_preferences: Replacement insight/brief preferences.
            notes: Freeform internal/user notes.
            onboarding_completed: Whether onboarding has been completed.

        Returns:
            The updated `User`.

        Raises:
            ValueError: If no user exists with the given `user_id`.
        """
        user = await self._get_user_by_id(user_id)

        if followed_companies is not None:
            user.followed_companies = followed_companies
        if followed_sectors is not None:
            user.followed_sectors = followed_sectors
        if insight_preferences is not None:
            user.insight_preferences = insight_preferences
        if notes is not None:
            user.notes = notes
        if onboarding_completed is not None:
            user.onboarding_completed = onboarding_completed

        await self._session.commit()
        await self._session.refresh(user)
        return user

    # -------------------------------------------------------------------
    # Conversation history
    # -------------------------------------------------------------------
    async def add_message(
        self,
        user_id: Any,
        role: MessageRole,
        content: str,
    ) -> ConversationMessage:
        """
        Store a single conversation message for a user.

        Args:
            user_id: Primary key of the user the message belongs to.
            role: Who authored the message (`user`, `assistant`, or `system`).
            content: The message text.

        Returns:
            The persisted `ConversationMessage`.
        """
        message = ConversationMessage(
            user_id=user_id,
            role=role,
            content=content,
        )
        self._session.add(message)
        await self._session.commit()
        await self._session.refresh(message)
        return message

    async def get_recent_messages(
        self,
        user_id: Any,
        limit: int = 20,
    ) -> list[ConversationMessage]:
        """
        Retrieve a user's most recent conversation messages.

        Args:
            user_id: Primary key of the user whose history to fetch.
            limit: Maximum number of messages to return.

        Returns:
            A list of `ConversationMessage` ordered oldest-to-newest
            (i.e. ready to feed directly into a chat context window),
            containing at most `limit` messages.
        """
        stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.user_id == user_id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        messages = list(result.scalars().all())
        messages.reverse()
        return messages

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------
    async def _get_user_by_id(self, user_id: Any) -> User:
        """
        Fetch a user by primary key or raise if not found.

        Args:
            user_id: Primary key of the user to fetch.

        Returns:
            The matching `User`.

        Raises:
            ValueError: If no user exists with the given `user_id`.
        """
        user = await self._session.get(User, user_id)
        if user is None:
            raise ValueError(f"No user found with id={user_id!r}")
        return user