"""Exception handlers for FastAPI responses in interview-engine-service."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette import status
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.exceptions import AppException
from src.schemas.common import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


def _json_error(
    *,
    status_code: int,
    message: str,
    errors: list[ErrorDetail] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(message=message, errors=errors)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Register application, validation, and database exception handlers."""

    @app.exception_handler(AppException)
    async def app_exception_handler(
        request: Request,
        exc: AppException,
    ) -> JSONResponse:
        logger.warning(
            "Application exception",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code,
                "error_code": getattr(exc, "error_code", None),
                "exception_type": type(exc).__name__,
            },
        )
        errors = (
            [ErrorDetail(**detail) for detail in exc.details]
            if exc.details is not None
            else None
        )
        return _json_error(
            status_code=exc.status_code,
            message=exc.message,
            errors=errors,
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        logger.warning(
            "HTTP exception occurred",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code,
            },
        )
        return _json_error(
            status_code=exc.status_code,
            message=exc.detail,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.warning(
            "Request validation failed",
            extra={"path": request.url.path, "method": request.method},
        )
        errors = [
            ErrorDetail(
                field=".".join(str(part) for part in error["loc"]),
                message=str(error["msg"]),
            )
            for error in exc.errors()
        ]
        return _json_error(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Request validation failed.",
            errors=errors,
        )

    @app.exception_handler(IntegrityError)
    async def integrity_exception_handler(
        request: Request,
        exc: IntegrityError,
    ) -> JSONResponse:
        """Render database integrity conflicts without exposing internals."""
        logger.exception(
            "Database integrity error",
            extra={"path": request.url.path, "method": request.method},
        )
        return _json_error(
            status_code=status.HTTP_409_CONFLICT,
            message="A data conflict occurred.",
        )

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exception_handler(
        request: Request,
        exc: SQLAlchemyError,
    ) -> JSONResponse:
        """Render SQLAlchemy failures as database errors."""
        logger.exception(
            "Database error",
            extra={"path": request.url.path, "method": request.method},
        )
        return _json_error(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="A database error occurred.",
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception(
            "Unhandled exception",
            extra={"path": request.url.path, "method": request.method},
        )
        return _json_error(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Internal server error.",
        )
