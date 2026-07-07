# Stage 1: Builder
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv venv /app/.venv \
    && uv export \
        --frozen \
        --no-dev \
        --no-emit-project \
        --output-file /tmp/requirements.txt \
    && VIRTUAL_ENV=/app/.venv uv pip install \
        --no-cache \
        --requirements /tmp/requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv ./.venv

COPY src ./src
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8001

ENTRYPOINT ["sh", "-c", "uvicorn src.api.rest.app:app --host 0.0.0.0 --port ${PORT}"]
