"""Async SQLAlchemy engine, session factory, and DB lifecycle helpers."""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

async_session_maker = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session for use as a FastAPI dependency."""
    async with async_session_maker() as session:
        yield session


async def init_db() -> None:
    """Verify the database is reachable.

    Schema management is handled by Alembic (`uv run alembic upgrade head`), not here -
    this used to call `Base.metadata.create_all`, which is no longer safe now that the
    schema has foreign keys and indexes migrations need to own. This just fails fast on
    startup if the DB is unreachable, rather than surfacing on the first request.
    """
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("Database connection OK (schema is Alembic-managed - run `alembic upgrade head`)")
