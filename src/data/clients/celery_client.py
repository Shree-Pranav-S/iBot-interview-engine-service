"""Celery client configuration for interview engine background workloads."""

import socket
from typing import Any

from celery import Celery
from celery.signals import worker_process_init
from kombu import Queue

from src.config.settings import settings
from src.observability.logging import configure_logging


def _redis_broker_transport_options() -> dict[str, Any]:
    """Keep broker connections alive across Cloud Run / Memorystore idle drops."""
    keepalive_options: dict[int, int] = {}
    if hasattr(socket, "TCP_KEEPIDLE"):
        keepalive_options[socket.TCP_KEEPIDLE] = 60
    if hasattr(socket, "TCP_KEEPINTVL"):
        keepalive_options[socket.TCP_KEEPINTVL] = 10
    if hasattr(socket, "TCP_KEEPCNT"):
        keepalive_options[socket.TCP_KEEPCNT] = 3

    return {
        "visibility_timeout": 7200,
        "socket_keepalive": True,
        "socket_keepalive_options": keepalive_options,
        "health_check_interval": settings.REDIS_HEALTHCHECK_INTERVAL,
        "retry_on_timeout": True,
        "socket_connect_timeout": settings.REDIS_SOCKET_CONNECT_TIMEOUT,
        "socket_timeout": max(settings.REDIS_SOCKET_TIMEOUT, 30),
    }


@worker_process_init.connect
def _configure_worker_logging(**_: object) -> None:
    """Apply JSON logging inside each Celery worker subprocess."""

    configure_logging()


celery_app = Celery(
    "interview_engine_service",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.handlers.celery_tasks.evaluation_tasks"],
)

celery_app.conf.update(
    accept_content=["json"],
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None,
    broker_transport_options=_redis_broker_transport_options(),
    enable_utc=True,
    result_serializer="json",
    task_acks_late=True,
    task_default_queue="interview.default",
    task_ignore_result=True,
    task_publish_retry=True,
    task_publish_retry_policy={
        "interval_max": 2.0,
        "interval_start": 0.2,
        "interval_step": 0.5,
        "max_retries": 3,
    },
    task_queues=(
        Queue("interview.default"),
        Queue("interview.evaluation"),
    ),
    task_routes={
        "core.process_final_evaluation": {"queue": "interview.evaluation"},
    },
    task_serializer="json",
    task_track_started=True,
    timezone="UTC",
    worker_prefetch_multiplier=1,
)
