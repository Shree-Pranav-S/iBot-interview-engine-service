"""Health endpoints for interview-engine-service."""

from fastapi import APIRouter

from src.clients.core_api_client import get_core_api_client
from src.data.clients.redis_client import ping_redis

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    summary="Check interview-engine dependencies",
    description="Report core-api and Redis connectivity for health monitoring.",
)
async def health() -> dict[str, str]:
    """Return aggregate and dependency health states."""

    core_api_ok = await get_core_api_client().health_ping()
    redis_ok = await ping_redis()
    return {
        "status": "ok" if core_api_ok and redis_ok else "degraded",
        "core_api": "ok" if core_api_ok else "error",
        "redis": "ok" if redis_ok else "error",
    }
