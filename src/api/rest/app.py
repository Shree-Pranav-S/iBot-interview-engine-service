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

    from urllib.parse import quote_plus

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from src.control.agents.graphs.interview_graph import build_interview_graph

    db_uri = (
        f"postgresql://{quote_plus(settings.POSTGRES_USER)}"
        f":{quote_plus(settings.POSTGRES_PASSWORD)}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}"
        f"/{settings.POSTGRES_DB}"
    )

    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
        await checkpointer.setup()
        app.state.compiled_graph = build_interview_graph().compile(
            checkpointer=checkpointer
        )
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
