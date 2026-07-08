"""Smoke tests for interview-engine custom exception JSON responses."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware.error_handler import register_exception_handlers
from src.core.exceptions import BadGatewayException, NotFoundException


def test_app_exception_returns_standard_envelope() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/missing")
    def missing() -> None:
        raise NotFoundException("Interview session not found.")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/missing")

    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["message"] == "Interview session not found."


def test_bad_gateway_exception_status() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/upstream")
    def upstream() -> None:
        raise BadGatewayException("Core API unavailable.")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/upstream")

    assert response.status_code == 502
    assert response.json()["success"] is False
