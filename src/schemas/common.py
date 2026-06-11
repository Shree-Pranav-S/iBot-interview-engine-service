"""
Common schemas.

Generic response envelopes and health check responses.
"""

from typing import Generic, TypeVar

from pydantic import Field

from src.schemas.base import AppBaseModel

T = TypeVar("T")


# ── Generic response envelope ─────────────────────────────────────────────────


class APIResponse(AppBaseModel, Generic[T]):
    """Standard JSON envelope wrapping all successful API responses."""

    success: bool = True
    message: str = "OK"
    data: T | None = None


class PaginatedResponse(AppBaseModel, Generic[T]):
    """Paginated list response."""

    items: list[T]
    total: int
    page: int = Field(..., ge=1)
    page_size: int = Field(..., ge=1, le=100)
    total_pages: int


# ── Error response ────────────────────────────────────────────────────────────


class ErrorDetail(AppBaseModel):
    """Single validation or field-level error detail."""

    field: str | None = None
    message: str


class ErrorResponse(AppBaseModel):
    """Standard error envelope returned on 4xx / 5xx responses."""

    success: bool = False
    message: str
    errors: list[ErrorDetail] | None = None


# ── Health check ──────────────────────────────────────────────────────────────


class HealthResponse(AppBaseModel):
    """GET /health response."""

    status: str = "ok"
    service: str
    environment: str
    database: str = "ok"
    redis: str = "ok"
