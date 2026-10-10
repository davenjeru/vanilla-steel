.PHONY: install check lint format test seed eval-extraction eval-extraction-live eval-matching record-fixtures

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

## Rebuild data/steel.db from the deterministic seed.
seed:
	uv run python scripts/seed.py

## Eval runs: each uploads one run to Confident AI. Commit first so the run's SHA is clean.
eval-extraction:
	uv run python -m evals.runs.extraction

## Calls Claude (costs money); measures the model and prompt, not just the code around them.
eval-extraction-live:
	uv run --env-file .env python -m evals.runs.extraction --live

eval-matching:
	uv run python -m evals.runs.matching

## Calls Claude for every extraction golden case and saves the answers to fixtures/llm/. Commit them.
record-fixtures:
	uv run --env-file .env python -m scripts.record_fixtures
