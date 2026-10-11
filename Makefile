.PHONY: install check lint format test seed eval-extraction eval-extraction-live eval-matching eval-duplicates eval-duplicates-live eval-agent eval-agent-live eval-agent-judge-recorded eval-quote-letter eval-quote-letter-live eval-quote-letter-judge-recorded record-fixtures prune-fixtures run-agent run-agent-live export-candidates

install:
	uv sync

## Lint and tests. CI runs this plus every eval on recorded answers (.github/workflows/ci.yml).
## Add --verified-only to any eval module to run only reviewed cases.
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

## Eval runs (evals/test_<eval>.py under `deepeval test run`, started by evals/run.py with the run's identifier).
## Each uploads one run to Confident AI when CONFIDENT_API_KEY is set. Commit first so the run's SHA is clean.
## Replay needs no keys. A failed case fails the run; judged metrics never decide a case (D35).
eval-extraction:
	uv run python -m evals.run extraction

## Calls Claude (costs money); measures the model and prompt, not just the code around them.
eval-extraction-live:
	uv run --env-file .env python -m evals.run extraction --live

## Duplicate check (D36): labelled email pairs, precision and recall. Replays recorded extraction.
eval-duplicates:
	uv run python -m evals.run duplicates

## Same, calling Claude for extraction and recording the answers (costs money). Commit fixtures/extraction/.
eval-duplicates-live:
	uv run --env-file .env python -m evals.run duplicates --live

## Replays recorded agent runs; cases without a recording are skipped by name, not scored.
eval-agent:
	uv run python -m evals.run agent

## Calls Claude for every agent case (costs money), records the answers to fixtures/agent/ (commit them),
## and adds the judged metrics (gpt-6-sol, report-only).
eval-agent-live:
	uv run --env-file .env python -m evals.run agent --live

## Judges the recorded agent runs again without re-running the agent (costs judge calls only).
eval-agent-judge-recorded:
	uv run --env-file .env python -m evals.run agent --judge-recorded

## Quote letters (Slice 5). Replay: recorded letters, deterministic metrics only, free.
eval-quote-letter:
	uv run python -m evals.run quote_letter

## New letters from Claude (recorded, commit them) plus the judged metrics (gpt-6-sol, report-only).
eval-quote-letter-live:
	uv run --env-file .env python -m evals.run quote_letter --live

## Judges the recorded letters again without regenerating them, e.g. after changing judge criteria.
eval-quote-letter-judge-recorded:
	uv run --env-file .env python -m evals.run quote_letter --judge-recorded

## Calls Claude for every extraction golden case and saves the answers to fixtures/extraction/. Commit them.
record-fixtures:
	uv run --env-file .env python -m scripts.record_fixtures

## Delete recordings no eval reads any more (old prompt versions). Run after a live re-record; commit the deletions.
prune-fixtures:
	uv run python -m scripts.prune_fixtures

## Run one extraction golden case through the agent and print its trace: make run-agent CASE=en-001-st37-sheet-poznan
run-agent:
	uv run python -m scripts.run_agent --case $(CASE)

## Same, calling Claude (costs money) and recording every turn to fixtures/agent/<case>/. Commit them.
run-agent-live:
	uv run --env-file .env python -m scripts.run_agent --case $(CASE) --live

## Feedback loop (D13): escalated, failed or flagged enquiries become candidate cases in evals/candidates/ (git-ignored).
## make export-candidates TENANT=tnt_bauer [ENQUIRIES="enq_0042 enq_0057"]
export-candidates:
	uv run python -m scripts.export_candidates --tenant $(TENANT) $(foreach id,$(ENQUIRIES),--enquiry $(id))

