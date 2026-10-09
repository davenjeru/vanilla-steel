.PHONY: install check lint format test seed

install:
	uv sync

## Exactly what CI runs. Run before every push.
check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

seed:
	uv run python scripts/seed.py
