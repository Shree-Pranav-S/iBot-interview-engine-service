"""Groq and in-interview LLM provider exceptions."""

from src.core.exceptions.base import BadGatewayException, InternalServerException


class LLMConfigurationException(InternalServerException):
    message = "LLM provider is not configured."
    error_code = "LLM_CONFIGURATION_ERROR"


class LLMProviderException(BadGatewayException):
    message = "LLM provider request failed."
    error_code = "LLM_PROVIDER_ERROR"
