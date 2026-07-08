"""LiveKit and candidate session token exceptions."""

from src.core.exceptions.base import InternalServerException


class LiveKitConfigurationException(InternalServerException):
    message = "LiveKit is not configured."
    error_code = "SESSION_LIVEKIT_CONFIGURATION_ERROR"


class SessionTokenExpiryInvalidException(InternalServerException):
    message = "Session token expiry is invalid."
    error_code = "SESSION_TOKEN_EXPIRY_INVALID"


class SessionTokenExpiryMissingException(InternalServerException):
    message = "Session token expiry is missing."
    error_code = "SESSION_TOKEN_EXPIRY_MISSING"
