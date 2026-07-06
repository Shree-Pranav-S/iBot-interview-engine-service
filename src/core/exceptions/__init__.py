"""Custom exceptions package init."""

from src.core.exceptions.base import (
    AppException,
    AuthenticationException,
    BadGatewayException,
    BadRequestException,
    ConflictException,
    ForbiddenException,
    InternalServerException,
    NotFoundException,
)
from src.core.exceptions.evaluation import (
    EvaluationError,
    EvaluationNotReadyError,
    EvaluationSchemaError,
    PermanentEvaluationError,
    TransientEvaluationError,
)

__all__ = [
    "AppException",
    "AuthenticationException",
    "BadRequestException",
    "ConflictException",
    "ForbiddenException",
    "InternalServerException",
    "BadGatewayException",
    "NotFoundException",
    "EvaluationError",
    "EvaluationNotReadyError",
    "EvaluationSchemaError",
    "PermanentEvaluationError",
    "TransientEvaluationError",
]
