.PHONY: install check lint format test seed eval-extraction eval-duplicates eval-agent eval-quote-letter push-prompts push-datasets pull-verified serve demo run-agent export-candidates

install:
	uv sync

## Lint and tests, no keys. CI runs exactly this (.github/workflows/ci.yml); the evals run on request.
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

## Eval runs: evals/test_<eval>.py under `deepeval test run`, started by evals/run.py with the run's identifier.
## Every eval calls Claude, its judged metrics call OpenAI, and with CONFIDENT_API_KEY set the run uploads
## (keys from .env). A failed deterministic check fails the run; judged metrics never decide a case (D35).
## Commit first so the run's SHA is clean. Add --verified-only through VERIFIED=1 to run reviewed cases only.
VERIFIED_FLAG = $(if $(VERIFIED),--verified-only,)

eval-extraction:
	uv run --env-file .env python -m evals.run extraction $(VERIFIED_FLAG)

## Duplicate check (D36): labelled email pairs, precision and recall printed over the run.
eval-duplicates:
	uv run --env-file .env python -m evals.run duplicates $(VERIFIED_FLAG)

## The agent on every case: decision, tool path, denylist, spec passed on, letters checked; fairness,
## cost and latency reported over the run.
eval-agent:
	uv run --env-file .env python -m evals.run agent $(VERIFIED_FLAG)

## Quote letters (Slice 5) on their own, for products and terms the agent cases never reach.
eval-quote-letter:
	uv run --env-file .env python -m evals.run quote_letter $(VERIFIED_FLAG)

## Prompt Studio and datasets on Confident AI (D42). Git stays the source of truth; nothing is pulled at run time.
## Push each product prompt once per version and record its commit in evals/prompts.json (commit the file).
push-prompts:
	uv run --env-file .env python -m scripts.push_prompts

## Push golden datasets, one alias per manifest version: verified cases finalized, the rest queued for review.
## All datasets, or some: make push-datasets DATASETS="agent extraction"
push-datasets:
	uv run --env-file .env python -m scripts.push_datasets $(DATASETS)

## Mark cases verified in git when their golden was finalized on Confident AI. Commit the changed case.json files.
pull-verified:
	uv run --env-file .env python -m scripts.pull_verified

## Hands-on demo. `make seed` first, then `make serve` in one terminal and `make demo` in another (PAUSE=1 to
## step through by hand). The demo sends golden emails as the three tenants: happy path, duplicates, injection,
## missing and ambiguous fields, tenant separation, idempotency, supplier anonymity.
serve:
	uv run --env-file .env python -m scripts.serve

demo:
	uv run python -m scripts.demo $(if $(PAUSE),--pause,)

## Run one golden case through the agent with Claude and print its trace: make run-agent CASE=agent-001-s355-plate-rfq
## Not an eval run, so its trace is not posted even with a Confident AI key in .env.
run-agent:
	CONFIDENT_TRACE_SAMPLE_RATE=0 uv run --env-file .env python -m scripts.run_agent --case $(CASE)

## Feedback loop (D13): escalated, failed or flagged enquiries become candidate cases in evals/candidates/ (git-ignored).
## make export-candidates TENANT=tnt_bauer [ENQUIRIES="enq_0042 enq_0057"]
export-candidates:
	uv run python -m scripts.export_candidates --tenant $(TENANT) $(foreach id,$(ENQUIRIES),--enquiry $(id))

