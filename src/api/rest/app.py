"""FastAPI application factory for interview-engine-service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware.error_handler import register_exception_handlers
from src.api.middleware.logging import request_logging_middleware
from src.api.rest.routes.health import router as health_router
from src.api.rest.routes.livekit import router as livekit_router
from src.clients.core_api_client import close_core_api_http_client
from src.config.settings import settings
from src.control.agents.graphs import close_graph, init_graph
from src.data.clients.redis_client import close_redis, init_redis
from src.observability.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and close shared service resources."""

    await init_redis()
    await init_graph()

    try:
        yield
    finally:
        await close_graph()
        await close_redis()
        await close_core_api_http_client()


def create_app() -> FastAPI:
    """Build the configured FastAPI application."""
    configure_logging()
    app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
    register_exception_handlers(app)
    app.middleware("http")(request_logging_middleware)
    app.include_router(health_router)
    app.include_router(livekit_router)
    return app


app = create_app()
