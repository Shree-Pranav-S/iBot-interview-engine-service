# Interview Engine Service

Runs candidate entry and reconnection, the LiveKit voice worker, the LangGraph
interview workflow, and asynchronous holistic evaluation.

## Architecture

- `src/api`: FastAPI routes and middleware.
- `src/clients`: pooled HTTP access to the core API's internal interview routes.
- `src/core/services`: session, LiveKit, graph-bridge, and evaluation orchestration.
- `src/data/clients`: Redis and Celery client lifecycle.
- `src/control/agents`: LiveKit and LangGraph runtime.
- `src/handlers`: Celery entrypoints.
- `src/schemas`: request, response, graph, and LLM contracts.

The core API owns application persistence and schema migrations. The interview
engine reaches it through internal HTTP routes; PostgreSQL access here is limited
to LangGraph checkpoints.

## Development checks

```shell
uv sync --group dev
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest
```
