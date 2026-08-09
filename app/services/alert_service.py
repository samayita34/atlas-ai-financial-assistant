"""
app/services/alert_service.py

Async service for managing user-created price alerts (e.g. "alert me
when AAPL goes above $320"). Mirrors the style/conventions of
WatchlistService: alerts are looked up by raw Telegram ID, use a plain
AsyncSession passed in per-call, and normalize persistence failures into
a small exception hierarchy.

Unlike WatchlistService, this service does not depend on
FinancialDataService -- ticker validation and live price checks are the
caller's responsibility (the LangGraph node validates on creation; the
scheduler job checks prices periodically), keeping this service a thin,
dependency-free persistence layer.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import AlertCondition, PriceAlert

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AlertError(Exception):
    """Base exception for all price alert service failures."""


class InvalidAlertConditionError(AlertError):
    """Raised when a condition string can't be parsed as above/below."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AlertEntry:
    """Normalized, service-layer representation of a price alert."""

    id: str
    telegram_id: int
    symbol: str
    condition: AlertCondition
    target_price: float
    is_active: bool
    created_at: Optional[datetime] = None
    triggered_at: Optional[datetime] = None

    @classmethod
    def from_orm(cls, item: PriceAlert) -> "AlertEntry":
        return cls(
            id=str(item.id),
            telegram_id=item.telegram_id,
            symbol=item.symbol,
            condition=item.condition,
            target_price=item.target_price,
            is_active=item.is_active,
            created_at=getattr(item, "created_at", None),
            triggered_at=getattr(item, "triggered_at", None),
        )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class AlertService:
    """
    Async service for creating, listing, and resolving per-user price
    alerts. Stateless -- a single instance can safely be shared, or a
    fresh one constructed per call (it holds no external resources).
    """

    async def create_alert(
        self,
        session: AsyncSession,
        *,
        telegram_id: int,
        symbol: str,
        condition: AlertCondition | str,
        target_price: float,
    ) -> AlertEntry:
        """
        Create and persist a new active price alert.

        Args:
            session: Active async DB session (caller-managed transaction
                scope, consistent with the rest of the service layer).
            telegram_id: Telegram user id owning the alert.
            symbol: Ticker symbol. Caller is responsible for
                resolving/validating it beforehand (same division of
                responsibility as WatchlistService.add_ticker).
            condition: "above"/"below", or an `AlertCondition` member.
            target_price: Price threshold that triggers the alert.

        Returns:
            The newly created `AlertEntry`.

        Raises:
            InvalidAlertConditionError: if `condition` isn't recognized.
            AlertError: on unexpected persistence failures.
        """
        if isinstance(condition, str):
            try:
                condition = AlertCondition(condition.strip().lower())
            except ValueError as exc:
                raise InvalidAlertConditionError(
                    f"Unrecognized alert condition {condition!r}; "
                    "expected 'above' or 'below'."
                ) from exc

        item = PriceAlert(
            telegram_id=telegram_id,
            symbol=symbol.strip().upper(),
            condition=condition,
            target_price=target_price,
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )

        try:
            session.add(item)
            await session.commit()
        except Exception as exc:  # noqa: BLE001
            await session.rollback()
            logger.exception(
                "Failed to create price alert for telegram_id=%s symbol=%s",
                telegram_id,
                symbol,
            )
            raise AlertError("Failed to create price alert.") from exc

        await session.refresh(item)
        logger.info(
            "Created price alert for telegram_id=%s: %s %s %.2f",
            telegram_id,
            item.symbol,
            condition.value,
            target_price,
        )
        return AlertEntry.from_orm(item)

    async def get_active_alerts(
        self,
        session: AsyncSession,
        *,
        telegram_id: Optional[int] = None,
    ) -> list[AlertEntry]:
        """
        Return all currently active alerts, optionally filtered to one
        user. Called with no filter by the scheduler job (checks every
        active alert across all users); the `telegram_id` filter is
        available for a future "show my alerts" feature.
        """
        stmt = select(PriceAlert).where(PriceAlert.is_active.is_(True))
        if telegram_id is not None:
            stmt = stmt.where(PriceAlert.telegram_id == telegram_id)

        try:
            result = await session.execute(stmt)
            items = result.scalars().all()
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to load active alerts (telegram_id=%s)", telegram_id
            )
            raise AlertError("Failed to load active alerts.") from exc

        return [AlertEntry.from_orm(item) for item in items]

    async def mark_triggered(self, session: AsyncSession, *, alert_id: str) -> None:
        """
        Deactivate an alert and stamp its trigger time, so it is never
        checked or notified again.

        Idempotent: if the alert is already inactive (e.g. seen twice
        across adjacent scheduler runs under unusual timing), this is a
        silent no-op rather than an error or a duplicate notification.

        Raises:
            AlertError: on unexpected persistence failures, or an
                invalid `alert_id`.
        """
        try:
            alert_uuid = uuid.UUID(alert_id)
        except ValueError as exc:
            raise AlertError(f"Invalid alert_id {alert_id!r}.") from exc

        item = await session.get(PriceAlert, alert_uuid)
        if item is None:
            logger.warning("mark_triggered called for missing alert_id=%s", alert_id)
            return
        if not item.is_active:
            return  # Already resolved -- never re-trigger or re-notify.

        item.is_active = False
        item.triggered_at = datetime.now(timezone.utc)

        try:
            await session.commit()
        except Exception as exc:  # noqa: BLE001
            await session.rollback()
            logger.exception("Failed to mark alert_id=%s as triggered", alert_id)
            raise AlertError("Failed to mark alert as triggered.") from exc

        logger.info("Marked alert_id=%s as triggered.", alert_id)
