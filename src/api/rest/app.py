"""FastAPI application factory for interview-engine-service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware.error_handler import register_exception_handlers
from src.api.rest.routes.demo_websocket import router as demo_ws_router
from src.api.rest.routes.health import router as health_router
from src.api.rest.routes.websocket import router as ws_router
from src.config.settings import settings
from src.data.clients.postgres_client import close_db, init_db
from src.data.clients.redis_client import close_redis, init_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await init_redis()
    try:
        yield
    finally:
        await close_redis()
        await close_db()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(ws_router)
    app.include_router(demo_ws_router)
    return app


app = create_app()
