"""Celery client configuration for interview engine background workloads."""

from celery import Celery
from kombu import Queue

from src.config.settings import settings

celery_app = Celery(
    "interview_engine_service",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["src.handlers.celery_tasks.evaluation_tasks"],
)

celery_app.conf.update(
    accept_content=["json"],
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
