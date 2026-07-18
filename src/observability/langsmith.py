"""Fail-fast, non-network LangSmith runtime configuration."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from src.config.settings import settings

logger = logging.getLogger(__name__)


def configure_langsmith() -> None:
    """Propagate validated settings for LangGraph and traced provider calls.

    Validation is intentionally local and does not contact LangSmith. A temporary
    observability outage must not add interview startup or turn latency, while a
    deployment that enables tracing without a credential should fail visibly.
    """

    if not settings.LANGSMITH_TRACING:
        logger.info("LangSmith tracing is disabled")
        return

    api_key = settings.LANGSMITH_API_KEY.strip()
    project = settings.LANGSMITH_PROJECT.strip()
    endpoint = settings.LANGSMITH_ENDPOINT.strip()
    parsed_endpoint = urlparse(endpoint)

    if not api_key:
        raise RuntimeError(
            "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is missing"
        )
    if not project:
        raise RuntimeError(
            "LANGSMITH_TRACING is enabled but LANGSMITH_PROJECT is missing"
        )
    if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
        raise RuntimeError("LANGSMITH_ENDPOINT must be a valid HTTP(S) URL")

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)
    os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    if settings.LANGSMITH_WORKSPACE_ID:
        os.environ.setdefault(
            "LANGSMITH_WORKSPACE_ID",
            settings.LANGSMITH_WORKSPACE_ID.strip(),
        )

    logger.info(
        "LangSmith tracing configured",
        extra={
            "langsmith_endpoint": endpoint,
            "langsmith_project": project,
            "langsmith_workspace_configured": bool(settings.LANGSMITH_WORKSPACE_ID),
            "environment": settings.APP_ENV,
        },
    )
