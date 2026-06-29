# Interview Engine Service

Runs candidate entry and reconnection, the LiveKit voice worker, the LangGraph
interview workflow, and asynchronous holistic evaluation.

## Architecture

- `src/api`: FastAPI routes and middleware.
- `src/core/services`: orchestration and business rules; no database sessions.
- `src/data/repositories`: session-injected repositories and the transactional
  `InterviewUnitOfWork`.
- `src/data/models/postgres`: SQLAlchemy mappings used by this service.
- `src/control/agents`: LiveKit and LangGraph runtime.
- `src/handlers`: Celery entrypoints.
- `src/schemas`: request, response, graph, and LLM contracts.
- `src/observability`: logging setup.

The core API owns shared PostgreSQL schema migrations. This service maps the
shared candidate and assessment tables, but does not duplicate their migrations.

## Development checks

```shell
uv sync --group dev
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run python -m unittest discover -s tests -v
```
