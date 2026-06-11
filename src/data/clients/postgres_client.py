import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config.settings import settings

_engine: AsyncEngine | None = None
logger = logging.getLogger(__name__)


async def get_or_create_engine() -> AsyncEngine:
    global _engine

    if _engine is None:
        logger.info("Creating async engine")
        _engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=settings.DB_POOL_TIMEOUT,
            pool_recycle=settings.DB_POOL_RECYCLE,
            pool_pre_ping=True,
            connect_args={
                "timeout": settings.DB_CONNECT_TIMEOUT,
                "command_timeout": settings.DB_COMMAND_TIMEOUT,
                "server_settings": {
                    "statement_timeout": str(settings.DB_STATEMENT_TIMEOUT_MS)
                },
            },
        )

    return _engine


async def get_session_factory() -> async_sessionmaker[AsyncSession]:
    engine = await get_or_create_engine()
    return async_sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    SessionLocal = await get_session_factory()
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    engine = await get_or_create_engine()
    async with engine.connect():
        logger.info("Database connection established.")


async def ping_db() -> bool:
    engine = await get_or_create_engine()
    try:
        async with engine.connect():
            return True
    except Exception:
        logger.exception("Database health check failed.")
        return False


async def close_db() -> None:
    if _engine is not None:
        await _engine.dispose()
