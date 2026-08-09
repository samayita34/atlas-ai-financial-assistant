"""
app/scheduler.py

Scheduled jobs for Atlas AI Financial Assistant.

Runs background jobs (via APScheduler's AsyncIOScheduler) that:
- Send each user a personalized daily financial briefing covering the
  companies/tickers on their watchlist.
- Periodically check watchlist tickers for significant intraday moves and
  send proactive alerts.

Designed to be started/stopped from `app/main.py`'s lifespan handler
alongside the Telegram bot and database.

NOTE ON ASSUMPTIONS:
- Exact MemoryService methods for enumerating all users and their
  watchlists/preferences are not specified in the provided context.
  Assumed signatures are marked with `# TODO:` comments.
- Briefing/alert delivery uses a raw `aiogram.Bot.send_message` call
  rather than routing through the LangGraph agent, since briefings are
  system-initiated (not a reply to a user message) and should not incur
  LLM classification overhead for a templated summary. The summary text
  itself is still generated via Gemini through a small local helper.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Final, Optional

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.job import Job
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config import get_settings

settings = get_settings()
from app.database.database import AsyncSessionLocal
from app.database.models import AlertCondition
from app.services.alert_service import AlertError, AlertService
from app.services.financial_data_service import (
    FinancialDataError,
    FinancialDataService,
)
from app.services.memory_service import MemoryService
from app.services.watchlist_service import WatchlistError, WatchlistService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# `Settings` (app/config.py) exposes the send time as a single "HH:MM"
# string (`default_brief_time`) plus `default_brief_timezone` — there is
# no `daily_briefing_hour`/`daily_briefing_minute`/`scheduler_timezone`
# field. The previous getattr(..., default) calls always missed (Settings
# is a Pydantic model, not a dict) and silently fell back to the hardcoded
# defaults below, so the configured brief time/timezone was never actually
# honored. Parse the real field instead.
_brief_hour_str, _brief_minute_str = settings.default_brief_time.split(":")
DAILY_BRIEFING_HOUR: Final[int] = int(_brief_hour_str)
DAILY_BRIEFING_MINUTE: Final[int] = int(_brief_minute_str)
DAILY_BRIEFING_TIMEZONE: Final[str] = settings.default_brief_timezone

WATCHLIST_ALERT_INTERVAL_MINUTES: Final[int] = getattr(
    settings, "watchlist_alert_interval_minutes", 30
)
# Percent move within the interval that triggers a proactive alert.
WATCHLIST_ALERT_THRESHOLD_PERCENT: Final[float] = getattr(
    settings, "watchlist_alert_threshold_percent", 3.0
)

MAX_BRIEFING_TICKERS: Final[int] = 5  # cap per-user briefing length
BRIEFING_JOB_ID: Final[str] = "daily_market_briefing"
WATCHLIST_ALERT_JOB_ID: Final[str] = "watchlist_alert_check"

PRICE_ALERT_INTERVAL_MINUTES: Final[int] = getattr(
    settings, "price_alert_interval_minutes", 1
)
PRICE_ALERT_JOB_ID: Final[str] = "price_alert_check"

SEND_CONCURRENCY_LIMIT: Final[int] = 10  # cap parallel Telegram sends


# ---------------------------------------------------------------------------
# LLM summary helper (reused pattern from langgraph/graph.py)
# ---------------------------------------------------------------------------

# TODO: `app.langgraph.graph` does not currently export a standalone LLM
# helper. Duplicating a minimal Gemini call here to avoid a circular
# import between `scheduler.py` and `langgraph/graph.py`. Consider
# factoring a shared `app/services/llm_service.py` out of both in a
# future refactor.
try:
    from google import genai  # type: ignore[import-not-found]

    # `settings.gemini_api_key` is a SecretStr (see app/config.py) — the
    # genai SDK needs the plain string. Passing the SecretStr object
    # directly caused every briefing to silently fail this construction
    # (or the first call) and fall through to the deterministic template,
    # with no visible error.
    _genai_client: Optional["genai.Client"] = genai.Client(
        api_key=settings.gemini_api_key.get_secret_value()
    )
except Exception:  # pragma: no cover - allows import without the SDK
    genai = None  # type: ignore[assignment]
    _genai_client = None
    logger.exception("Failed to construct Gemini client for scheduled briefings.")

# `Settings` defines `gemini_model`, not `gemini_model_name` — the old
# getattr(..., default) always missed and silently used the hardcoded
# fallback below regardless of what was actually configured.
GEMINI_MODEL_NAME: Final[str] = settings.gemini_model


async def _summarize_briefing(user_first_name: str, overviews: list[dict[str, Any]]) -> str:
    """
    Generate a concise, friendly daily briefing message from a list of
    company overview dicts (as produced by
    `FinancialDataService.get_company_overview`).

    Falls back to a simple templated summary if the LLM call fails or is
    unavailable, so briefings are never silently dropped.
    """
    if _genai_client is not None:
        try:
            prompt = (
                f"Write a concise (under 120 words) daily financial briefing "
                f"for {user_first_name or 'the user'}, covering these "
                f"watchlist companies. Use plain language, mention notable "
                f"price moves and any major news headline per company. "
                f"Telegram Markdown is supported.\n\n"
                f"Data:\n{overviews}"
            )
            response = await _genai_client.aio.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception:  # noqa: BLE001
            logger.exception("Failed to generate LLM briefing summary; using fallback.")

    return _fallback_briefing_text(overviews)


def _fallback_briefing_text(overviews: list[dict[str, Any]]) -> str:
    """Deterministic, template-based briefing used if the LLM is unavailable."""
    if not overviews:
        return "📊 *Daily Briefing*\n\nNo watchlist data available today."

    lines = ["📊 *Daily Briefing*", ""]
    for item in overviews:
        symbol = item.get("symbol", "?")
        quote = item.get("quote") or {}
        price = quote.get("current_price")
        pct = quote.get("percent_change")
        price_str = f"${price:,.2f}" if isinstance(price, (int, float)) else "n/a"
        pct_str = f"{pct:+.2f}%" if isinstance(pct, (int, float)) else "n/a"
        lines.append(f"• *{symbol}*: {price_str} ({pct_str})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UserBriefingTarget:
    """Minimal user info needed to send a scheduled message."""

    telegram_id: int
    first_name: Optional[str]
    watchlist: list[str]


async def _get_all_briefing_targets(
    memory_service: MemoryService,
    watchlist_service: WatchlistService,
    *,
    only_telegram_id: Optional[int] = None,
) -> list[UserBriefingTarget]:
    """
    Enumerate all users who should receive scheduled messages, along with
    their watchlist.

    Watchlist data is read via `WatchlistService.get_symbols(...)`, which
    is backed by the `WatchlistItem` table (the feature already tested
    end-to-end through Telegram). It is intentionally NOT read from
    `MemoryService.get_watchlist()` / `User.followed_companies` — that
    field is only ever populated by the onboarding flow and is a separate,
    unrelated store from the actual watchlist.

    Args:
        only_telegram_id: If provided, only that user is included (used by
            the manual/debug trigger so a test run doesn't message every
            real user).
    """
    targets: list[UserBriefingTarget] = []
    async with AsyncSessionLocal() as session:
        users = await memory_service.list_all_users(session)

        for user in users:
            telegram_id = getattr(user, "telegram_id", None)
            if telegram_id is None:
                continue
            if only_telegram_id is not None and telegram_id != only_telegram_id:
                continue

            try:
                watchlist = await watchlist_service.get_symbols(
                    session, telegram_id=telegram_id
                )
            except WatchlistError:
                logger.exception(
                    "Failed to load watchlist for telegram_id=%s", telegram_id
                )
                watchlist = []

            first_name = getattr(user, "first_name", None) or getattr(user, "full_name", None)
            targets.append(
                UserBriefingTarget(
                    telegram_id=telegram_id,
                    first_name=first_name,
                    watchlist=list(watchlist or []),
                )
            )

    return targets


# ---------------------------------------------------------------------------
# Job implementations
# ---------------------------------------------------------------------------


async def _send_message_safely(
    bot: Bot, telegram_id: int, text: str, *, semaphore: asyncio.Semaphore
) -> None:
    """Send a Telegram message, logging (not raising) on failure."""
    async with semaphore:
        try:
            await bot.send_message(chat_id=telegram_id, text=text)
        except TelegramAPIError:
            logger.exception("Failed to deliver scheduled message to telegram_id=%s", telegram_id)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Unexpected error sending scheduled message to telegram_id=%s", telegram_id
            )


async def _build_briefing_for_user(
    target: UserBriefingTarget,
    financial_data_service: FinancialDataService,
) -> Optional[str]:
    """
    Build the briefing text for a single user, or None if they have no
    watchlist (in which case no briefing is sent).
    """
    if not target.watchlist:
        return None

    tickers = target.watchlist[:MAX_BRIEFING_TICKERS]
    overviews: list[dict[str, Any]] = []

    for ticker in tickers:
        try:
            overview = await financial_data_service.get_company_overview(ticker)
            overviews.append(overview)
        except FinancialDataError:
            logger.warning(
                "Skipping ticker %r in briefing for telegram_id=%s (data unavailable)",
                ticker,
                target.telegram_id,
            )
            continue

    if not overviews:
        return None

    return await _summarize_briefing(target.first_name or "", overviews)


async def run_daily_briefing_job(
    bot: Bot,
    memory_service: MemoryService,
    financial_data_service: FinancialDataService,
    watchlist_service: WatchlistService,
    *,
    only_telegram_id: Optional[int] = None,
) -> dict[str, Any]:
    """
    Send each user with a non-empty watchlist a personalized daily
    financial briefing.

    Registered as a cron job (see `create_scheduler`), typically run once
    per day at `DAILY_BRIEFING_HOUR:DAILY_BRIEFING_MINUTE`.

    Args:
        only_telegram_id: If provided, restricts this run to a single
            user. Used by the manual/debug trigger (see
            `trigger_daily_briefing_now`) so a test run never messages
            every real user.

    Returns:
        A small summary dict (`targets_found`, `briefings_sent`,
        `skipped_no_watchlist`, `failed`) — mainly useful for the manual
        trigger to report back what actually happened, since the
        scheduled path only logs this.
    """
    logger.info("Running daily briefing job (only_telegram_id=%s)...", only_telegram_id)

    summary: dict[str, Any] = {
        "targets_found": 0,
        "briefings_sent": 0,
        "skipped_no_watchlist": 0,
        "failed": 0,
    }

    try:
        targets = await _get_all_briefing_targets(
            memory_service, watchlist_service, only_telegram_id=only_telegram_id
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to enumerate briefing targets; aborting job run.")
        summary["failed"] += 1
        return summary

    summary["targets_found"] = len(targets)

    if not targets:
        logger.info("No users found for daily briefing job.")
        return summary

    semaphore = asyncio.Semaphore(SEND_CONCURRENCY_LIMIT)
    send_tasks: list[asyncio.Task[None]] = []

    for target in targets:
        try:
            briefing_text = await _build_briefing_for_user(target, financial_data_service)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to build briefing for telegram_id=%s", target.telegram_id
            )
            summary["failed"] += 1
            continue

        if briefing_text is None:
            summary["skipped_no_watchlist"] += 1
            continue

        send_tasks.append(
            asyncio.create_task(
                _send_message_safely(
                    bot, target.telegram_id, briefing_text, semaphore=semaphore
                )
            )
        )

    if send_tasks:
        await asyncio.gather(*send_tasks, return_exceptions=True)

    summary["briefings_sent"] = len(send_tasks)
    logger.info("Daily briefing job complete. Sent to %d user(s).", len(send_tasks))
    return summary


async def run_watchlist_alert_job(
    bot: Bot,
    memory_service: MemoryService,
    financial_data_service: FinancialDataService,
    watchlist_service: WatchlistService,
) -> None:
    """
    Check each user's watchlist tickers for significant intraday moves
    (>= `WATCHLIST_ALERT_THRESHOLD_PERCENT`) and send a proactive alert.

    Registered as an interval job (see `create_scheduler`), running every
    `WATCHLIST_ALERT_INTERVAL_MINUTES` minutes.

    TODO: This uses Finnhub's `dp` (percent change from previous close)
    as a proxy for "significant move," which is simple but coarse (it
    doesn't dedupe an alert already sent earlier in the day for the same
    move). A production version should persist "last alerted price/time"
    per (user, ticker) via MemoryService to avoid repeat alerts.
    """
    logger.info("Running watchlist alert check...")

    try:
        targets = await _get_all_briefing_targets(memory_service, watchlist_service)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to enumerate watchlist targets; aborting job run.")
        return

    # De-duplicate ticker lookups across users to minimize Finnhub calls.
    unique_tickers: set[str] = {t for target in targets for t in target.watchlist}
    if not unique_tickers:
        logger.info("No watchlist tickers found; skipping alert check.")
        return

    quotes: dict[str, Optional[float]] = {}
    for ticker in unique_tickers:
        try:
            quote = await financial_data_service.get_quote(ticker)
            quotes[ticker] = quote.percent_change
        except FinancialDataError:
            logger.warning("Skipping alert check for %r (quote unavailable)", ticker)
            quotes[ticker] = None

    semaphore = asyncio.Semaphore(SEND_CONCURRENCY_LIMIT)
    send_tasks: list[asyncio.Task[None]] = []

    for target in targets:
        triggered: list[tuple[str, float]] = []
        for ticker in target.watchlist:
            pct = quotes.get(ticker)
            if pct is not None and abs(pct) >= WATCHLIST_ALERT_THRESHOLD_PERCENT:
                triggered.append((ticker, pct))

        if not triggered:
            continue

        lines = ["⚡ *Watchlist Alert*", ""]
        for ticker, pct in triggered:
            direction = "📈" if pct >= 0 else "📉"
            lines.append(f"{direction} *{ticker}*: {pct:+.2f}% today")
        alert_text = "\n".join(lines)

        send_tasks.append(
            asyncio.create_task(
                _send_message_safely(
                    bot, target.telegram_id, alert_text, semaphore=semaphore
                )
            )
        )

    if send_tasks:
        await asyncio.gather(*send_tasks, return_exceptions=True)

    logger.info("Watchlist alert job complete. Alerts sent to %d user(s).", len(send_tasks))


async def run_price_alert_job(
    bot: Bot,
    financial_data_service: FinancialDataService,
) -> None:
    """
    Check all active price alerts, fetch current prices via
    FinancialDataService, and trigger (notify + deactivate) any alert
    whose condition is now met.

    Registered as an interval job (see `create_scheduler`), running every
    `PRICE_ALERT_INTERVAL_MINUTES` minutes.

    Each alert is deactivated in its own short-lived session BEFORE its
    Telegram notification is sent -- this ordering (not the reverse)
    guarantees an alert can never be re-triggered and re-notified twice
    for the same crossing, even if the Telegram send itself fails
    afterward (that failure is only logged, per `_send_message_safely`).
    """
    logger.info("Running price alert check...")

    alert_service = AlertService()

    async with AsyncSessionLocal() as session:
        try:
            active_alerts = await alert_service.get_active_alerts(session)
        except AlertError:
            logger.exception("Failed to load active price alerts; aborting job run.")
            return

    if not active_alerts:
        logger.info("No active price alerts; skipping.")
        return

    # De-duplicate ticker lookups across alerts to minimize Finnhub calls
    # (mirrors the same dedup pattern used in run_watchlist_alert_job).
    unique_symbols = {alert.symbol for alert in active_alerts}
    quotes: dict[str, Optional[float]] = {}
    for symbol in unique_symbols:
        try:
            quote = await financial_data_service.get_quote(symbol)
            quotes[symbol] = quote.current_price
        except FinancialDataError:
            logger.warning("Skipping price alert check for %r (quote unavailable)", symbol)
            quotes[symbol] = None

    semaphore = asyncio.Semaphore(SEND_CONCURRENCY_LIMIT)
    send_tasks: list[asyncio.Task[None]] = []
    triggered_count = 0

    for alert in active_alerts:
        current_price = quotes.get(alert.symbol)
        if current_price is None:
            continue

        condition_met = (
            alert.condition == AlertCondition.ABOVE and current_price > alert.target_price
        ) or (
            alert.condition == AlertCondition.BELOW and current_price < alert.target_price
        )
        if not condition_met:
            continue

        # Deactivate first, in its own session -- see docstring note on
        # ordering. If this fails, skip the notification entirely rather
        # than notify-then-fail-to-deactivate (which would repeat-notify
        # on every subsequent run).
        async with AsyncSessionLocal() as session:
            try:
                await alert_service.mark_triggered(session, alert_id=alert.id)
            except AlertError:
                logger.exception(
                    "Failed to mark alert_id=%s as triggered; skipping "
                    "notification to avoid a repeat trigger next run.",
                    alert.id,
                )
                continue

        triggered_count += 1
        direction_word = "risen above" if alert.condition == AlertCondition.ABOVE else "fallen below"
        arrow = "📈" if alert.condition == AlertCondition.ABOVE else "📉"
        alert_text = (
            f"🔔 *Price Alert Triggered*\n\n"
            f"{arrow} *{alert.symbol}* has {direction_word} "
            f"${alert.target_price:,.2f}\n"
            f"Current price: ${current_price:,.2f}"
        )
        send_tasks.append(
            asyncio.create_task(
                _send_message_safely(bot, alert.telegram_id, alert_text, semaphore=semaphore)
            )
        )

    if send_tasks:
        await asyncio.gather(*send_tasks, return_exceptions=True)

    logger.info("Price alert job complete. %d alert(s) triggered.", triggered_count)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def create_scheduler(
    *,
    bot: Bot,
    memory_service: MemoryService,
    financial_data_service: FinancialDataService,
    watchlist_service: WatchlistService,
) -> AsyncIOScheduler:
    """
    Construct (but do not start) an `AsyncIOScheduler` with Atlas's
    scheduled jobs registered.

    Args:
        bot: Shared aiogram Bot instance used to deliver messages.
        memory_service: Shared MemoryService instance.
        financial_data_service: Shared FinancialDataService instance.
        watchlist_service: Shared WatchlistService instance (source of
            truth for per-user watchlists — see `_get_all_briefing_targets`).

    Returns:
        A configured `AsyncIOScheduler`. Call `.start()` to begin running
        jobs (typically from `app/main.py`'s lifespan startup), and
        `.shutdown(wait=...)` on application shutdown.
    """
    scheduler = AsyncIOScheduler(
        executors={"default": AsyncIOExecutor()},
        timezone=DAILY_BRIEFING_TIMEZONE,
    )

    scheduler.add_job(
        run_daily_briefing_job,
        trigger=CronTrigger(
            hour=DAILY_BRIEFING_HOUR,
            minute=DAILY_BRIEFING_MINUTE,
            timezone=DAILY_BRIEFING_TIMEZONE,
        ),
        id=BRIEFING_JOB_ID,
        name="Daily market briefing",
        kwargs={
            "bot": bot,
            "memory_service": memory_service,
            "financial_data_service": financial_data_service,
            "watchlist_service": watchlist_service,
        },
        replace_existing=True,
        misfire_grace_time=60 * 30,  # tolerate up to 30 min scheduler downtime
        coalesce=True,
        max_instances=1,
    )

    scheduler.add_job(
        run_watchlist_alert_job,
        trigger=IntervalTrigger(minutes=WATCHLIST_ALERT_INTERVAL_MINUTES),
        id=WATCHLIST_ALERT_JOB_ID,
        name="Watchlist alert check",
        kwargs={
            "bot": bot,
            "memory_service": memory_service,
            "financial_data_service": financial_data_service,
            "watchlist_service": watchlist_service,
        },
        replace_existing=True,
        misfire_grace_time=60 * 5,
        coalesce=True,
        max_instances=1,
    )

    scheduler.add_job(
        run_price_alert_job,
        trigger=IntervalTrigger(minutes=PRICE_ALERT_INTERVAL_MINUTES),
        id=PRICE_ALERT_JOB_ID,
        name="Price alert check",
        kwargs={
            "bot": bot,
            "financial_data_service": financial_data_service,
        },
        replace_existing=True,
        misfire_grace_time=60 * 2,
        coalesce=True,
        max_instances=1,
    )

    return scheduler


def get_job(scheduler: AsyncIOScheduler, job_id: str) -> Optional[Job]:
    """Convenience accessor for inspecting a registered job (e.g. in tests/health checks)."""
    return scheduler.get_job(job_id)


# ---------------------------------------------------------------------------
# Manual trigger (for testing without waiting for the cron time)
# ---------------------------------------------------------------------------


async def trigger_daily_briefing_now(
    *,
    bot: Bot,
    memory_service: MemoryService,
    financial_data_service: FinancialDataService,
    watchlist_service: WatchlistService,
    telegram_id: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run the exact same daily-briefing logic as the scheduled cron job,
    on demand, without touching the registered `CronTrigger` or its
    schedule.

    This calls `run_daily_briefing_job` directly with the *same* bot and
    service instances the real scheduled job uses — it is not a
    simulation or a separate code path, so a successful manual run is
    genuine end-to-end proof the scheduled job will also work.

    Args:
        telegram_id: If provided, only that user's briefing is built and
            sent (recommended for testing, so a manual trigger doesn't
            message every real user). Omit to run exactly like the real
            scheduled job (all users).

    Returns:
        Summary dict from `run_daily_briefing_job` — see its docstring.
    """
    return await run_daily_briefing_job(
        bot=bot,
        memory_service=memory_service,
        financial_data_service=financial_data_service,
        watchlist_service=watchlist_service,
        only_telegram_id=telegram_id,
    )


# ---------------------------------------------------------------------------
# Standalone bootstrap (optional local/dev usage)
# ---------------------------------------------------------------------------


async def _run_forever(scheduler: AsyncIOScheduler) -> None:  # pragma: no cover
    """Keep the process alive while the scheduler runs in the background."""
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":  # pragma: no cover
    # TODO: Local dev convenience only. In production, `create_scheduler`
    # is expected to be wired into `app/main.py`'s lifespan alongside the
    # shared Bot/MemoryService/FinancialDataService instances rather than
    # constructing new ones here.
    import logging as _logging

    from app.bot.telegram_bot import create_bot

    _logging.basicConfig(level=_logging.INFO)

    async def _main() -> None:
        bot = create_bot()
        memory_service = MemoryService()  # type: ignore[call-arg]
        financial_data_service = FinancialDataService()
        watchlist_service = WatchlistService(financial_data_service=financial_data_service)

        scheduler = create_scheduler(
            bot=bot,
            memory_service=memory_service,
            financial_data_service=financial_data_service,
            watchlist_service=watchlist_service,
        )
        scheduler.start()
        logger.info("Scheduler started standalone. Press Ctrl+C to exit.")

        try:
            await _run_forever(scheduler)
        finally:
            scheduler.shutdown(wait=False)
            await bot.session.close()
            await financial_data_service.aclose()

    asyncio.run(_main())