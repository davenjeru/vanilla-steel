.PHONY: install check lint format test seed eval-extraction eval-extraction-live eval-matching eval-agent eval-agent-live record-fixtures run-agent run-agent-live export-candidates

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

## Replays recorded agent runs; cases without a recording are listed, not scored.
## Reads .env because DeepEval's ToolCorrectnessMetric needs an OpenAI key to be built (it calls no model here).
eval-agent:
	uv run --env-file .env python -m evals.runs.agent

## Calls Claude for every agent case (costs money) and records the answers to fixtures/llm/. Commit them.
eval-agent-live:
	uv run --env-file .env python -m evals.runs.agent --live

## Calls Claude for every extraction golden case and saves the answers to fixtures/llm/. Commit them.
record-fixtures:
	uv run --env-file .env python -m scripts.record_fixtures

## Run one extraction golden case through the agent and print its trace: make run-agent CASE=en-001-st37-sheet-poznan
run-agent:
	uv run python -m scripts.run_agent --case $(CASE)

## Same, calling Claude (costs money) and recording every turn to fixtures/llm/. Commit them.
run-agent-live:
	uv run --env-file .env python -m scripts.run_agent --case $(CASE) --live

## Feedback loop (D13): escalated, failed or flagged enquiries become candidate cases in evals/candidates/ (git-ignored).
## make export-candidates TENANT=tnt_bauer [ENQUIRIES="enq_0042 enq_0057"]
export-candidates:
	uv run python -m scripts.export_candidates --tenant $(TENANT) $(foreach id,$(ENQUIRIES),--enquiry $(id))
