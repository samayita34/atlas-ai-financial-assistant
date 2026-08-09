"""
app/main.py

FastAPI application entrypoint for Atlas AI Financial Assistant.

Responsibilities:
- Construct the FastAPI app with an async lifespan handler.
- Initialize the database (engine/tables) on startup.
- Initialize the Telegram bot + LangGraph-backed services and start
  polling as a background task (webhook mode can be swapped in later).
- Expose minimal health-check routing.
- Ensure clean, ordered shutdown of the bot, polling task, and DB engine.

NOTE ON ASSUMPTIONS:
The exact constructor signatures of MemoryService, FinancialDataService,
and DocumentService, as well as the precise shape of `database.py`'s
init/dispose helpers, are not fully specified in the provided context.
Where an interface is missing or ambiguous, the smallest reasonable
assumption is made and marked with a `# TODO:` comment.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from aiogram import Bot, Dispatcher
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.bot.telegram_bot import setup_bot
from app.config import Environment, get_settings

settings = get_settings()
from app.database.database import (
    AsyncSessionLocal,
    close_db,
    init_db,
)
from app.scheduler import create_scheduler, trigger_daily_briefing_now
from app.services.document_service import DocumentService
from app.services.financial_data_service import FinancialDataService
from app.services.memory_service import MemoryService
from app.services.watchlist_service import WatchlistService

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=getattr(settings.log_level, "value", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class AppState:
    """
    Simple container for objects that must persist across the app's
    lifespan and be reachable from lifespan startup/shutdown.
    """

    def __init__(self) -> None:
        self.memory_service: Optional[MemoryService] = None
        self.financial_data_service: Optional[FinancialDataService] = None
        self.document_service: Optional[DocumentService] = None
        self.watchlist_service: Optional[WatchlistService] = None
        self.bot: Optional[Bot] = None
        self.dispatcher: Optional[Dispatcher] = None
        self.polling_task: Optional[asyncio.Task[None]] = None
        self.scheduler: Optional[AsyncIOScheduler] = None


app_state = AppState()


async def _start_bot_polling(bot: Bot, dispatcher: Dispatcher) -> None:
    """
    Run aiogram long-polling as a background task.

    TODO: For production, consider switching to webhook mode behind
    FastAPI (e.g. a POST /telegram/webhook route calling
    `dispatcher.feed_webhook_update`) instead of polling, to avoid
    running two long-lived network loops in one process. Polling is used
    here as the smallest reasonable default consistent with
    `telegram_bot.run_polling`.
    """
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        await dispatcher.start_polling(bot, handle_signals=False)
    except asyncio.CancelledError:
        logger.info("Telegram polling task cancelled; shutting down cleanly.")
        raise
    except Exception:  # noqa: BLE001
        logger.exception("Telegram polling task crashed unexpectedly.")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan handler: manages startup and shutdown of the
    database, service layer, and Telegram bot.
    """
    logger.info("Starting Atlas AI Financial Assistant...")

    # ---- Database ----------------------------------------------------
    try:
        # TODO: `init_db` is assumed to be an async callable (e.g. it
        # runs `Base.metadata.create_all` via the async engine, and/or
        # verifies pgvector extension availability). Adjust if
        # `database.py` instead exposes a differently named helper.
        await init_db()
        logger.info("Database initialized successfully.")
    except Exception:
        logger.exception("Failed to initialize database. Aborting startup.")
        raise

    # ---- Service layer --------------------------------------------------
    # TODO: Exact constructor signatures for these services are assumed
    # to be parameterless / self-configuring from `app.config.settings`
    # (e.g. reading API keys, DB session factory, etc. internally). Adjust
    # if they require explicit dependencies (session factory, http
    # client, embedding model, etc.) to be passed in.
    session = AsyncSessionLocal()
    memory_service = MemoryService(session)
    financial_data_service = FinancialDataService()
    document_service = DocumentService(session)
    # NOTE: previously not constructed here at all — telegram_bot.py's
    # per-request fallback (`_watchlist_service or WatchlistService(...)`)
    # meant a fresh instance was created on every message. WatchlistService
    # is stateless aside from holding `financial_data_service`, so this is
    # a pure wiring fix (same pattern as the other services), not a change
    # to the Watchlist implementation itself. It's also required so the
    # scheduler can query the same watchlist data the bot uses.
    watchlist_service = WatchlistService(financial_data_service=financial_data_service)

    app_state.memory_service = memory_service
    app_state.financial_data_service = financial_data_service
    app_state.document_service = document_service
    app_state.watchlist_service = watchlist_service

    # ---- Telegram bot ----------------------------------------------------
    bot, dispatcher = setup_bot(
        memory_service=memory_service,
        financial_data_service=financial_data_service,
        document_service=document_service,
        watchlist_service=watchlist_service,
    )
    app_state.bot = bot
    app_state.dispatcher = dispatcher

    polling_task = asyncio.create_task(
        _start_bot_polling(bot, dispatcher), name="telegram-polling"
    )
    app_state.polling_task = polling_task
    logger.info("Telegram bot polling started.")

    # ---- Scheduler (daily briefing + watchlist alerts) -------------------
    # This was previously never constructed/started anywhere in the app —
    # `create_scheduler` existed in scheduler.py but nothing called it
    # outside that module's own standalone `__main__` block, so the daily
    # briefing cron job never ran in the deployed service.
    scheduler = create_scheduler(
        bot=bot,
        memory_service=memory_service,
        financial_data_service=financial_data_service,
        watchlist_service=watchlist_service,
    )
    scheduler.start()
    app_state.scheduler = scheduler
    logger.info(
        "Scheduler started. Daily briefing job registered: %s",
        scheduler.get_job("daily_market_briefing") is not None,
    )

    try:
        yield
    finally:
        logger.info("Shutting down Atlas AI Financial Assistant...")

        # ---- Stop scheduler (before the bot session closes, so no
        # in-flight job tries to send through a closed session) ---------
        if app_state.scheduler is not None:
            try:
                app_state.scheduler.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                logger.exception("Error while shutting down scheduler.")

        # ---- Stop Telegram polling ----------------------------------
        if app_state.dispatcher is not None:
            try:
                await app_state.dispatcher.stop_polling()
            except Exception:  # noqa: BLE001
                logger.exception("Error while stopping dispatcher polling.")

        if app_state.polling_task is not None:
            app_state.polling_task.cancel()
            try:
                await app_state.polling_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                logger.exception("Error while awaiting polling task shutdown.")

        # ---- Close bot session ---------------------------------------
        if app_state.bot is not None:
            try:
                await app_state.bot.session.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error while closing Telegram bot session.")

        try:
            await session.close()
            await close_db()
            logger.info("Database engine disposed.")
        except Exception:  # noqa: BLE001
            logger.exception("Error while disposing database engine.")

        logger.info("Atlas shutdown complete.")


app = FastAPI(
    title="Atlas AI Financial Assistant",
    description=(
        "Backend service for Atlas, a Telegram-based AI financial "
        "analyst assistant."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
async def health_check() -> JSONResponse:
    """
    Liveness/readiness probe.

    Reports whether the Telegram polling task is alive as a coarse
    signal of overall service health. Extend with a DB ping if a
    dedicated health-check query is added to `database.py`.
    """
    polling_alive = (
        app_state.polling_task is not None and not app_state.polling_task.done()
    )
    status = "ok" if polling_alive else "degraded"
    return JSONResponse(
        status_code=200 if polling_alive else 503,
        content={
            "status": status,
            "bot_polling": polling_alive,
        },
    )


@app.get("/", tags=["system"])
async def root() -> JSONResponse:
    """Basic root endpoint confirming the service is up."""
    return JSONResponse(content={"service": "atlas-ai-financial-assistant", "status": "running"})


@app.post("/debug/trigger-daily-briefing", tags=["debug"])
async def debug_trigger_daily_briefing(telegram_id: Optional[int] = None) -> JSONResponse:
    """
    Manually run the daily briefing job right now, using the exact same
    scheduler-registered `bot`/`memory_service`/`financial_data_service`/
    `watchlist_service` instances the real cron job uses — this is not a
    simulation, it IS the job, invoked on demand.

    Query param `telegram_id`: if provided, only that user's briefing is
    built and sent (recommended for testing). Omit to run against every
    user, exactly like the real 08:00 cron firing would.

    This does NOT modify the registered `CronTrigger` — the scheduled job
    still fires at its configured time regardless of how many times this
    endpoint is called.

    Disabled in production as a safety measure, since it can message real
    users on demand.
    """
    if settings.environment == Environment.PRODUCTION:
        return JSONResponse(
            status_code=403,
            content={"error": "Manual briefing trigger is disabled in production."},
        )

    if (
        app_state.bot is None
        or app_state.memory_service is None
        or app_state.financial_data_service is None
        or app_state.watchlist_service is None
    ):
        return JSONResponse(
            status_code=503,
            content={"error": "Services not fully initialized yet."},
        )

    summary = await trigger_daily_briefing_now(
        bot=app_state.bot,
        memory_service=app_state.memory_service,
        financial_data_service=app_state.financial_data_service,
        watchlist_service=app_state.watchlist_service,
        telegram_id=telegram_id,
    )
    return JSONResponse(content={"telegram_id_filter": telegram_id, **summary})


# TODO: If Atlas grows additional HTTP surface area (e.g. a REST API for
# a companion web dashboard, or a Telegram webhook route as an
# alternative to polling — see `_start_bot_polling` TODO above), register
# those routers here via `app.include_router(...)`.


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=getattr(settings, "host", "0.0.0.0"),
        port=getattr(settings, "port", 8000),
        reload=getattr(settings, "debug", False),
    )