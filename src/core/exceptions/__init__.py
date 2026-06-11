"""Custom exceptions package init."""

from src.core.exceptions.base import (
    AppException,
    AuthenticationException,
    BadRequestException,
    ConflictException,
    NotFoundException,
)

__all__ = [
    "AppException",
    "AuthenticationException",
    "BadRequestException",
    "ConflictException",
    "NotFoundException",
]
