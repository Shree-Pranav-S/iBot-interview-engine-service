FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=8001

WORKDIR /app

COPY pyproject.toml ./
COPY uv.lock ./

RUN uv export \
        --frozen \
        --no-dev \
        --no-emit-project \
        --output-file /tmp/requirements.txt \
    && uv pip install \
        --system \
        --requirements /tmp/requirements.txt

COPY src ./src
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8001

ENTRYPOINT ["sh", "-c", "uvicorn src.api.rest.app:app --host 0.0.0.0 --port ${PORT}"]
