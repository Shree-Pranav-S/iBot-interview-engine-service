"""Run LiveKit agent with a minimal HTTP health endpoint for Cloud Run."""

from __future__ import annotations

import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.observability.langsmith import configure_langsmith
from src.observability.logging import configure_logging


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    configure_logging()
    configure_langsmith()
    agent_cmd = [
        sys.executable,
        "-m",
        "src.control.agents.livekit_agent",
        "start",
    ]
    subprocess.Popen(agent_cmd)

    port = int(os.environ.get("PORT", "8080"))
    HTTPServer(("0.0.0.0", port), _HealthHandler).serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
