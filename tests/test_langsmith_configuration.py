"""LangSmith runtime configuration contracts."""

import os

import pytest

from src.config.settings import settings
from src.observability.langsmith import configure_langsmith


def test_enabled_tracing_requires_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "")

    with pytest.raises(RuntimeError, match="LANGSMITH_API_KEY is missing"):
        configure_langsmith()


def test_valid_settings_are_propagated_without_network_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LANGSMITH_PROJECT", "test-project")
    monkeypatch.setattr(
        settings,
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com",
    )
    monkeypatch.setattr(settings, "LANGSMITH_WORKSPACE_ID", None)
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)

    configure_langsmith()

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "test-key"
    assert os.environ["LANGSMITH_PROJECT"] == "test-project"
