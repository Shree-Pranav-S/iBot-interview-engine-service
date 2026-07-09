"""Helper functions and state for the core API client."""

from __future__ import annotations

import httpx

from src.core.exceptions import (
    AppException,
    AuthenticationException,
    BadGatewayException,
    BadRequestException,
    ConflictException,
    ForbiddenException,
    InternalServerException,
    NotFoundException,
)

_INTERNAL_HEADERS = {"X-Internal-Service": "interview-engine"}
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_HTTP_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)
_STATUS_EXCEPTIONS: dict[int, type[AppException]] = {
    400: BadRequestException,
    401: AuthenticationException,
    403: ForbiddenException,
    404: NotFoundException,
    409: ConflictException,
    500: InternalServerException,
    502: BadGatewayException,
}
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=_TIMEOUT, limits=_HTTP_LIMITS)
    return _http_client


async def close_core_api_http_client() -> None:
    """Close the shared pooled HTTP client during process shutdown."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _exception_for_status(status_code: int, message: str) -> AppException:
    exc_type = _STATUS_EXCEPTIONS.get(status_code, AppException)
    return exc_type(message, status_code=status_code)
