"""Run Celery with a minimal HTTP health endpoint for Cloud Run."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.observability.logging import configure_logging

logger = logging.getLogger(__name__)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args) -> None:
        return


def _serve_health(port: int) -> None:
    HTTPServer(("0.0.0.0", port), _HealthHandler).serve_forever()


def _run_celery_with_restart(celery_cmd: list[str]) -> None:
    while True:
        logger.info("Starting Celery worker subprocess")
        process = subprocess.Popen(celery_cmd)
        exit_code = process.wait()
        if exit_code == 0:
            logger.info("Celery worker exited cleanly")
            return
        logger.error(
            "Celery worker exited with code %s; restarting in 5 seconds",
            exit_code,
        )
        time.sleep(5)


def main() -> None:
    celery_cmd = [
        "celery",
        "-A",
        "src.data.clients.celery_client.celery_app",
        "worker",
        "--loglevel=INFO",
        "--queues=interview.evaluation,interview.default",
        "--pool=prefork",
        "--concurrency=1",
    ]

    port = int(os.environ.get("PORT", "8080"))
    threading.Thread(target=_serve_health, args=(port,), daemon=True).start()
    _run_celery_with_restart(celery_cmd)


if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
