"""Structured logging setup for interview-engine-service."""

import json
import logging
import sys
from datetime import UTC, datetime

from src.config.settings import settings


class JsonLogFormatter(logging.Formatter):
    """Small stdlib JSON formatter for application logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON object string."""
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Configure process-wide JSON logging for the application."""

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())

    root_logger.addHandler(handler)
    root_logger.setLevel(
        logging.INFO if settings.APP_ENV != "test" else logging.WARNING
    )
