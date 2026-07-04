"""Common API response envelopes."""

from typing import Generic, TypeVar

from src.schemas.base import AppBaseModel

T = TypeVar("T")


class APIResponse(AppBaseModel, Generic[T]):
    """Standard JSON envelope wrapping all successful API responses."""

    success: bool = True
    message: str = "OK"
    data: T | None = None


class ErrorDetail(AppBaseModel):
    """Single validation or field-level error detail."""

    field: str | None = None
    message: str


class ErrorResponse(AppBaseModel):
    """Standard error envelope returned on 4xx / 5xx responses."""

    success: bool = False
    message: str
    errors: list[ErrorDetail] | None = None
