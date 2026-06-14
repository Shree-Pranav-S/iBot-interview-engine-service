"""Application exception hierarchy."""

from http import HTTPStatus
from typing import Any


class AppException(Exception):
    """Base class for domain exceptions rendered by the error middleware."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    message: str = "Internal server error."

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int | None = None,
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        self.message = message or self.message
        self.status_code = status_code or self.status_code
        self.details = details
        super().__init__(self.message)


class BadRequestException(AppException):
    status_code = HTTPStatus.BAD_REQUEST
    message = "Bad request."


class AuthenticationException(AppException):
    status_code = HTTPStatus.UNAUTHORIZED
    message = "Authentication failed."


class ConflictException(AppException):
    status_code = HTTPStatus.CONFLICT
    message = "Resource conflict."


class NotFoundException(AppException):
    status_code = HTTPStatus.NOT_FOUND
    message = "Resource not found."


class ForbiddenException(AppException):
    status_code = HTTPStatus.FORBIDDEN
    message = "Forbidden."


class InternalServerException(AppException):
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    message = "Internal server error."


class BadGatewayException(AppException):
    status_code = HTTPStatus.BAD_GATEWAY
    message = "Bad gateway."
