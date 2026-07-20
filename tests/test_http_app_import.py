"""HTTP application import smoke test for deployment entrypoint integrity."""

from src.api.rest.app import app


def test_http_application_entrypoint_imports() -> None:
    assert app.title == "interview-engine-service"
