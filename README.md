# vanilla-steel

A small replica of a European steel procurement pipeline, and the test and evaluation harness
around it. Customer emails and spreadsheets come in; an extractor reads the material spec; the
matcher ranks supplier stock by delivered cost; an LLM agent checks for duplicates, requests a
quotation for one lot and drafts the buyer's quote, or hands the enquiry to a human trader.

The product is a means to an end. The point of the repository is to show how an agentic system
like this is tested: which checks are deterministic, which are judged by a model, what gates a
release, and how golden data, prompts and runs stay traceable.

## What the tests protect

| Risk | What must hold | Where it is checked |
|---|---|---|
| Extraction accuracy | Grade, dimensions, tonnage, coating, Incoterm, delivery place and date are correct and normalised (St37-2 is S235JR, kg becomes t); nothing is guessed; the customer's language is read, since the quote letter is written in it | `tests/unit/` (normalisation), `evals/test_extraction.py` (14 fields and the language per golden case) |
| Data separation | No tenant reads or affects another's data; a buyer never learns who the supplier is | `tests/api/` (cross-tenant reads, idempotency keys), `tests/agent/test_runner.py` (the run acts for one tenant), "Reveals no supplier to the buyer" and the letter checks in `evals/test_agent.py` |
| Duplication | A resend, forward or reworded repeat never becomes a second RFQ; a correction is handed over, not re-ordered | `tests/unit/test_duplicates.py`, `evals/test_duplicates.py` (18 labelled pairs, precision and recall), the duplicate cases in `evals/test_agent.py` |
| Business correctness | Matches respect hard constraints, ranking is by delivered cost, the agent escalates when unsure instead of guessing | `tests/unit/test_matching_golden_cases.py`, `evals/test_agent.py` (decision, exact tool path, denylist per case, spec passed on unchanged), run-wide fairness, cost and latency reports |

## Quick start

```
uv sync
make check                  # ruff, 780+ unit and API tests, matching golden cases; no keys, about 5 s
make eval-agent             # the 44 agent cases against Claude; keys from .env
make eval-agent VERIFIED=1  # only the cases a person has reviewed
```

`make check` needs no keys and is what CI runs. Every eval calls Claude, its judged metrics call
OpenAI, and with a Confident AI key the run and its traces are uploaded; the keys live in `.env`
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `CONFIDENT_API_KEY`). The Makefile itself is the command
reference: every target has a comment above it.

## Demo: the pipeline end to end

```
make seed          # a fresh database, so the first enquiry is not a duplicate of yesterday's
make serve         # terminal 1: the API on http://127.0.0.1:8000 with Claude behind it
make demo PAUSE=1  # terminal 2: ten scenarios, Enter between them
```

The demo sends golden emails as the three seeded tenants and prints what the API answered: the
happy path (one RFQ, a priced quote, a cover letter), a retry with the same `Idempotency-Key`
answered once, an exact resend and a correction caught as duplicates, a prompt injection in the
email and one in a spreadsheet handed over as suspicious before any search, a missing quantity and
an ambiguous grade handed to a trader, the same email from another tenant quoted on its own while
neither tenant can read the other's, and a buyer asking who the supplier is, answered with a quote
that names none. With `CONFIDENT_API_KEY` in `.env`, every enquiry's trace appears on Confident AI.

To send your own enquiry, the development keys are in `steel/storage/seed.py`:

```
curl -s http://127.0.0.1:8000/v1/enquiries -H "X-API-Key: dev-key-bauer-0c1f6a" \
  -H "Content-Type: application/json" \
  -d '{"sender": "anna@example.com", "subject": "Enquiry S355J2 plates",
       "body": "please quote 30 t of S355J2 plate, 15 x 1250 mm, DAP Poznań, Poland."}'
curl -s http://127.0.0.1:8000/v1/quotes/<quote_id> -H "X-API-Key: dev-key-bauer-0c1f6a"
```

## How an eval runs

Each eval is a pytest file under `evals/`, run by `deepeval test run` through `evals/run.py`,
which adds the run's identifier (`sha=... dataset=agent-v9 component=agent-v2`) so runs are
comparable on Confident AI. One golden case is one test: the product runs inside a trace it
records itself (`steel/tracing.py`: an agent span, an LLM span per model call with model,
tokens and agreed price, a tool span per call, a refusal as an errored span), then
`assert_test(golden=..., metrics=...)` scores that trace.

- **Deterministic metrics decide.** Right decision and lot, exact tool path, no forbidden tool
  (a denylist per case), the stock search uses exactly the extracted spec, no supplier named to
  the buyer, no loop; for letters, only the quoted figures, no supplier, only products we sell.
  A failed case fails the run, in every mode.
- **Judged metrics report.** Task completion and role violation on the agent; on letters, a
  decision tree for the language (which language is the prose in, and only then: does it read as
  a business letter in that language), letter quality (G-Eval), PII leakage, non-advice and
  toxicity. All judged by `gpt-6-sol`
  and `flaky` in DeepEval's sense: scored and shown, never deciding a case. No calibration is
  claimed for them. The safety ones are read together with the trace tag `risk:security`.
- **Run-wide reports print at the end.** Fairness across counterfactual groups (the same order in
  English, German and Polish must get the same outcome), cost per enquiry against the agreed
  limits, model latency (live only), and duplicate precision and recall.

Verdicts live in DeepEval's output and on Confident AI; there is no separate release gate.

## Datasets and prompts

Golden cases are folders under `evals/datasets/<dataset>/`: `email.txt`, `case.json` with the
labels and a plain-words `reason`, attachments as real files. Each dataset has a versioned
manifest. About 40% of cases per dataset are marked `verified: true` after a person's review;
`--verified-only` runs only those, which is what the paid live workflow does.

Git is the source of truth. Confident AI holds copies for review and comparison:

```
make push-prompts      # each prompt text once per version; evals/prompts.json records the commit
make push-datasets     # one alias per manifest version (agent-v10); verified finalized, the rest queued
make pull-verified     # a golden finalized on Confident AI marks its case verified in git
```

`evals/prompts.json` and `evals/datasets.json` record what was pushed; `make check` fails when a
prompt version, a manifest version or a new case was never pushed, and says which command to run.

Nothing is pulled at run time. Every LLM span names the prompt it ran (alias, git version and,
once pushed, the Prompt Studio commit), and the run's hyperparameters name the same.

### Changing a prompt

A prompt never changes under its version. The whole change is one pull request:

1. Add the new text as `<id>_<version>.md` beside the old one and bump `PROMPT_VERSION` in the
   module that runs it, with a note of what the version is for.
2. `make push-prompts` (needs `CONFIDENT_API_KEY`): the text goes to Prompt Studio once and
   `evals/prompts.json` records the commit. Commit that file with the prompt.
3. Open the pull request; `make check` fails until step 2 is in it. Add `ready-for-live-verified`
   to see the new version against the verified cases, linked to its Prompt Studio commit.

## Continuous integration

`.github/workflows/ci.yml` runs `make check` on each push: lint, the unit and API tests and the
matcher's golden cases, with no keys and no model calls. The evals call live models, so they run
on purpose: `.github/workflows/live-evals.yml` runs the verified cases on a pull request against
`main` once it carries the label `ready-for-live-verified`, and by hand (one eval or all);
`.github/workflows/pre-release.yml`, by hand only, runs
every case of every eval, unverified ones included, as the last look before a release. Both upload
to Confident AI (D43). There is no schedule.

## Where things are decided

- `docs/plan.md`: every design decision, numbered (D1 to D48), the interfaces per slice and the
  planned metrics with their status.
- `docs/findings.md`: every failure the harness found, its cause, and what was done (F-001 to F-013).
- `docs/test-strategy.md`: the layers, what gates, and what is deliberately not covered.
- `CLAUDE.md`: the code standards, which apply to tests as much as to the product.

## Known limitations

- `agent-024` (F-013) fails on every run so far: after a refused stock search the agent still
  writes the guessed Incoterm into its hand-over note. The agent eval is red until the prompt is
  fixed.
- DeepEval 4.2.8 drops a span's prompt fields inside a test run, would re-push a `Prompt`
  hyperparameter on every keyed run, and loses a pulled golden's dataset link on the trace-scoped
  path; the prompt link therefore lives in span metadata, the hyperparameters carry text, and the
  run is stamped with its dataset by hand (D42, D44).
- Every eval run spends money (about $0.30 for the 18 verified agent cases, $0.75 for all 44 with
  the judge), so there is no free check that the agent still behaves; `make check` covers the
  code around the model.
