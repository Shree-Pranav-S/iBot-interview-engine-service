"""Health endpoints for interview-engine-service."""

from fastapi import APIRouter

from src.data.clients.postgres_client import ping_db
from src.data.clients.redis_client import ping_redis

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    summary="Check interview-engine dependencies",
    description="Report PostgreSQL and Redis connectivity for health monitoring.",
)
async def health() -> dict[str, str]:
    """Return aggregate and dependency health states."""

    db_ok = await ping_db()
    redis_ok = await ping_redis()
    return {
        "status": "ok" if db_ok and redis_ok else "degraded",
        "database": "ok" if db_ok else "error",
        "redis": "ok" if redis_ok else "error",
    }
