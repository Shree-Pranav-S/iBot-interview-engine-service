"""Typed failure categories for holistic evaluation orchestration."""


class EvaluationError(RuntimeError):
    """Base class for holistic evaluation failures."""


class PermanentEvaluationError(EvaluationError):
    """Invalid or missing source data that should not be retried blindly."""


class TransientEvaluationError(EvaluationError):
    """Temporary provider, readiness, or infrastructure failure."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class EvaluationNotReadyError(TransientEvaluationError):
    """The interview is still finalizing durable transcript state."""


class EvaluationSchemaError(PermanentEvaluationError):
    """DeepSeek failed the strict output contract after repair attempts."""
