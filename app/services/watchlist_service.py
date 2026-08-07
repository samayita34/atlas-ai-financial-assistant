"""
app/services/watchlist_service.py

Watchlist management service for Atlas AI Financial Assistant.

Handles adding, removing, and listing tickers on a user's watchlist,
backed by PostgreSQL via async SQLAlchemy. Ticker validity is confirmed
against Finnhub (through `FinancialDataService`) before persisting, so
users can't add junk symbols to their watchlist.

Designed to be used both directly (e.g. from `app/langgraph/graph.py`'s
watchlist node) and as the eventual home for the watchlist persistence
logic currently assumed to live on `MemoryService` in that module and in
`app/scheduler.py`.

NOTE ON ASSUMPTIONS:
- `app.database.models` is assumed to define a `WatchlistItem` model
  (or equivalent) with at least: `id`, a user-identifying column, a
  `symbol` column, and a `created_at` timestamp, plus a unique
  constraint on (user, symbol) to prevent duplicates at the DB level.
  Since the exact model name/columns weren't provided, this module
  defines its own minimal expectations and marks them with TODOs so they
  can be reconciled against the real `models.py`.
- User identification is assumed to be the Telegram user id (`telegram_id:
  int`), consistent with `MemoryService`'s assumed interface used
  elsewhere in the project (`telegram_bot.py`, `langgraph/graph.py`,
  `scheduler.py`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.financial_data_service import (
    FinancialDataError,
    FinancialDataService,
    SymbolNotFoundError,
)

# TODO: `app.database.models` is assumed to define a `WatchlistItem` ORM
# model shaped roughly like:
#
#     class WatchlistItem(Base):
#         __tablename__ = "watchlist_items"
#         id: Mapped[int] = mapped_column(primary_key=True)
#         telegram_id: Mapped[int] = mapped_column(BigInteger, index=True)
#         symbol: Mapped[str] = mapped_column(String(16))
#         created_at: Mapped[datetime] = mapped_column(
#             DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
#         )
#         __table_args__ = (
#             UniqueConstraint("telegram_id", "symbol", name="uq_watchlist_user_symbol"),
#         )
#
# If the real model uses a different name/column layout (e.g. a FK to a
# `users.id` primary key instead of a raw `telegram_id`), update the
# import and the attribute names below accordingly.
from app.database.models import WatchlistItem  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

MAX_WATCHLIST_SIZE: Final[int] = 25  # TODO: confirm desired cap with product


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WatchlistError(Exception):
    """Base exception for all watchlist service failures."""


class InvalidTickerError(WatchlistError):
    """Raised when a ticker cannot be validated against live market data."""


class DuplicateTickerError(WatchlistError):
    """Raised when attempting to add a ticker already on the watchlist."""


class TickerNotOnWatchlistError(WatchlistError):
    """Raised when attempting to remove a ticker that isn't on the watchlist."""


class WatchlistFullError(WatchlistError):
    """Raised when a user's watchlist is already at `MAX_WATCHLIST_SIZE`."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WatchlistEntry:
    """Normalized, service-layer representation of a watchlist item."""

    symbol: str
    added_at: Optional[datetime] = None

    @classmethod
    def from_orm(cls, item: WatchlistItem) -> "WatchlistEntry":
        return cls(
            symbol=getattr(item, "symbol"),
            added_at=getattr(item, "created_at", None),
        )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class WatchlistService:
    """
    Async service for managing per-user stock watchlists.

    Instances are cheap and stateless aside from holding a reference to a
    `FinancialDataService` for ticker validation; a single instance can
    safely be shared across requests (e.g. constructed once in
    `app/main.py`'s lifespan alongside the other services).
    """

    def __init__(self, *, financial_data_service: FinancialDataService) -> None:
        """
        Args:
            financial_data_service: Shared FinancialDataService instance,
                used to resolve/validate tickers before persisting them.
        """
        self._financial_data_service = financial_data_service

    # -- validation -------------------------------------------------------

    async def _validate_and_resolve_symbol(self, raw_symbol: str) -> str:
        """
        Resolve free-text input (ticker or company name) to a canonical
        ticker symbol and confirm it corresponds to real market data.

        Raises:
            InvalidTickerError: if the symbol cannot be resolved or
                Finnhub has no data for it.
        """
        cleaned = raw_symbol.strip().upper()
        if not cleaned:
            raise InvalidTickerError("Ticker symbol cannot be empty.")

        try:
            resolved = await self._financial_data_service.resolve_symbol(cleaned)
            # Confirm the symbol actually has live data behind it (catches
            # stale/delisted symbols that still resolve via /search).
            await self._financial_data_service.get_quote(resolved)
        except SymbolNotFoundError as exc:
            raise InvalidTickerError(
                f"'{raw_symbol}' doesn't match a known ticker or company."
            ) from exc
        except FinancialDataError as exc:
            logger.warning(
                "Ticker validation failed for %r due to a data service error: %s",
                raw_symbol,
                exc,
            )
            raise InvalidTickerError(
                f"Couldn't verify '{raw_symbol}' right now. Please try again shortly."
            ) from exc

        return resolved

    # -- queries -----------------------------------------------------------

    async def list_watchlist(
        self, session: AsyncSession, *, telegram_id: int
    ) -> list[WatchlistEntry]:
        """
        Return all tickers on a user's watchlist, ordered by when they
        were added (oldest first).
        """
        stmt = (
            select(WatchlistItem)
            .where(WatchlistItem.telegram_id == telegram_id)
            .order_by(WatchlistItem.created_at.asc())
        )
        try:
            result = await session.execute(stmt)
            items = result.scalars().all()
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to list watchlist for telegram_id=%s", telegram_id
            )
            raise WatchlistError("Failed to load watchlist.") from exc

        return [WatchlistEntry.from_orm(item) for item in items]

    async def get_symbols(self, session: AsyncSession, *, telegram_id: int) -> list[str]:
        """Convenience accessor returning just the ticker symbols, in add order."""
        entries = await self.list_watchlist(session, telegram_id=telegram_id)
        return [entry.symbol for entry in entries]

    async def is_on_watchlist(
        self, session: AsyncSession, *, telegram_id: int, symbol: str
    ) -> bool:
        """Check whether a (already-resolved) ticker symbol is on the user's watchlist."""
        normalized = symbol.strip().upper()
        stmt = select(WatchlistItem.id).where(
            WatchlistItem.telegram_id == telegram_id,
            WatchlistItem.symbol == normalized,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None

    # -- mutations -----------------------------------------------------------

    async def add_ticker(
        self, session: AsyncSession, *, telegram_id: int, symbol: str
    ) -> WatchlistEntry:
        """
        Validate and add a ticker to a user's watchlist.

        Args:
            session: Active async DB session (caller-managed transaction
                scope, consistent with the rest of the service layer).
            telegram_id: Telegram user id owning the watchlist.
            symbol: Raw ticker or company name input from the user.

        Returns:
            The newly created `WatchlistEntry`.

        Raises:
            InvalidTickerError: if the symbol can't be validated.
            WatchlistFullError: if the user is already at the max size.
            DuplicateTickerError: if the ticker is already on the
                watchlist.
            WatchlistError: on unexpected persistence failures.
        """
        resolved_symbol = await self._validate_and_resolve_symbol(symbol)

        current_count_stmt = select(WatchlistItem.id).where(
            WatchlistItem.telegram_id == telegram_id
        )
        current_count = len((await session.execute(current_count_stmt)).scalars().all())
        if current_count >= MAX_WATCHLIST_SIZE:
            raise WatchlistFullError(
                f"Watchlist is full (max {MAX_WATCHLIST_SIZE} tickers). "
                "Remove one before adding another."
            )

        if await self.is_on_watchlist(session, telegram_id=telegram_id, symbol=resolved_symbol):
            raise DuplicateTickerError(f"{resolved_symbol} is already on your watchlist.")

        item = WatchlistItem(
            telegram_id=telegram_id,
            symbol=resolved_symbol,
            created_at=datetime.now(timezone.utc),
        )

        try:
            session.add(item)
            await session.flush()
        except IntegrityError as exc:
            # Handles the race where a duplicate was inserted concurrently
            # between the `is_on_watchlist` check and this flush, assuming
            # a unique (telegram_id, symbol) constraint exists per the
            # TODO on the WatchlistItem model above.
            await session.rollback()
            logger.info(
                "Duplicate watchlist insert race for telegram_id=%s symbol=%s",
                telegram_id,
                resolved_symbol,
            )
            raise DuplicateTickerError(
                f"{resolved_symbol} is already on your watchlist."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            await session.rollback()
            logger.exception(
                "Failed to add %s to watchlist for telegram_id=%s",
                resolved_symbol,
                telegram_id,
            )
            raise WatchlistError("Failed to add ticker to watchlist.") from exc

        await session.commit()
        logger.info("Added %s to watchlist for telegram_id=%s", resolved_symbol, telegram_id)
        return WatchlistEntry.from_orm(item)

    async def remove_ticker(
        self, session: AsyncSession, *, telegram_id: int, symbol: str
    ) -> None:
        """
        Remove a ticker from a user's watchlist.

        Args:
            session: Active async DB session.
            telegram_id: Telegram user id owning the watchlist.
            symbol: Ticker symbol to remove (case-insensitive; resolved
                via FinancialDataService only if needed to normalize
                casing/format — removal does not require the ticker to
                still be valid/tradeable).

        Raises:
            TickerNotOnWatchlistError: if the ticker isn't on the
                watchlist.
            WatchlistError: on unexpected persistence failures.
        """
        normalized = symbol.strip().upper()

        stmt = select(WatchlistItem).where(
            WatchlistItem.telegram_id == telegram_id,
            WatchlistItem.symbol == normalized,
        )
        try:
            result = await session.execute(stmt)
            item = result.scalar_one_or_none()

            if item is None:
                raise TickerNotOnWatchlistError(
                    f"{normalized} isn't on your watchlist."
                )

            await session.delete(item)
            await session.commit()
        except TickerNotOnWatchlistError:
            raise
        except Exception as exc:  # noqa: BLE001
            await session.rollback()
            logger.exception(
                "Failed to remove %s from watchlist for telegram_id=%s",
                normalized,
                telegram_id,
            )
            raise WatchlistError("Failed to remove ticker from watchlist.") from exc

        logger.info("Removed %s from watchlist for telegram_id=%s", normalized, telegram_id)

    async def clear_watchlist(self, session: AsyncSession, *, telegram_id: int) -> int:
        """
        Remove all tickers from a user's watchlist.

        Returns:
            The number of tickers removed.
        """
        entries = await self.list_watchlist(session, telegram_id=telegram_id)
        if not entries:
            return 0

        stmt = select(WatchlistItem).where(WatchlistItem.telegram_id == telegram_id)
        try:
            result = await session.execute(stmt)
            items = result.scalars().all()
            for item in items:
                await session.delete(item)
            await session.commit()
        except Exception as exc:  # noqa: BLE001
            await session.rollback()
            logger.exception("Failed to clear watchlist for telegram_id=%s", telegram_id)
            raise WatchlistError("Failed to clear watchlist.") from exc

        logger.info("Cleared %d watchlist item(s) for telegram_id=%s", len(items), telegram_id)
        return len(items)