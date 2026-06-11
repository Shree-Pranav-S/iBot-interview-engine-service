"""ASGI entrypoint for interview-engine-service."""

from src.api.rest.app import app

__all__ = ["app"]
