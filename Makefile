.PHONY: run lint typecheck test

run:
	uv run uvicorn src.api.rest.app:app --host 0.0.0.0 --port 8001 --reload

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest
