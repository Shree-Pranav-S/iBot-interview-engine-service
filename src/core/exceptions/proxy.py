"""Core API proxy and downstream service exceptions."""

from src.core.exceptions.base import BadGatewayException


class CoreApiUnreachableException(BadGatewayException):
    message = "core-api is unreachable."
    error_code = "PROXY_CORE_API_UNREACHABLE"


class CoreApiRequestFailedException(BadGatewayException):
    message = "core-api request failed."
    error_code = "PROXY_CORE_API_REQUEST_FAILED"
