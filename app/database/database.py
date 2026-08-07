"""
app/database/database.py

Database engine and session management for the application.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Environment, get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ---------------------------------------------------------------------------
# Database Engine
# ---------------------------------------------------------------------------

engine: AsyncEngine = create_async_engine(
    str(settings.database_url),
    echo=settings.environment == Environment.LOCAL,
    pool_pre_ping=True,
)

# ---------------------------------------------------------------------------
# Session Factory
# ---------------------------------------------------------------------------

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# ---------------------------------------------------------------------------
# FastAPI Dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a database session.

    Transactions are controlled explicitly by the service layer.
    """

    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Startup Helper
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """
    Verify database connectivity during application startup.
    """

    logger.info("Initializing database connection...")

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

        logger.info("Database connection established successfully.")

    except Exception:
        logger.exception("Failed to establish database connection.")
        raise


# ---------------------------------------------------------------------------
# Shutdown Helper
# ---------------------------------------------------------------------------

async def close_db() -> None:
    """
    Dispose the SQLAlchemy engine and close all pooled connections.
    """

    logger.info("Closing database connections...")

    await engine.dispose()

    logger.info("Database connections closed.")


