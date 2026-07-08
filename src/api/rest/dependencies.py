"""Common FastAPI dependencies for the REST layer."""

from src.core.services.core_api_session_service import CoreApiSessionService
from src.core.services.livekit_token_service import LiveKitTokenService


def get_livekit_token_service() -> LiveKitTokenService:
    """Build the LiveKit token service for the request."""
    return LiveKitTokenService()


def get_core_api_session_service() -> CoreApiSessionService:
    """Build the core-api session lifecycle delegate."""
    return CoreApiSessionService()


__all__ = [
    "get_core_api_session_service",
    "get_livekit_token_service",
]
