import logging

from redis.asyncio import Redis, from_url

from src.config.settings import settings

_client: Redis | None = None
logger = logging.getLogger(__name__)


async def get_or_create_client() -> Redis:
    global _client

    if _client is None:
        logger.info("Creating redis client")
        _client = from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
            health_check_interval=settings.REDIS_HEALTHCHECK_INTERVAL,
        )

    return _client


async def init_redis() -> None:
    client = await get_or_create_client()
    await client.ping()
    logger.info("Redis connection established.")


async def ping_redis() -> bool:
    try:
        client = await get_or_create_client()
        return bool(await client.ping())
    except Exception:
        logger.exception("Redis health check failed.")
        return False


async def close_redis() -> None:
    if _client is not None:
        await _client.aclose()
