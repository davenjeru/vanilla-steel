# Build plan

Vertical slices. Each slice works end to end through the API, then pauses for evaluation.
Claude builds product code and unit/API tests for it.
The project owner builds everything to do with evaluations: the eval harness,
DeepEval test cases, golden data, the baseline file and the release gate.
This plan describes the product side in detail and records evaluation decisions only.

## Decisions

| # | Decision |
|---|----------|
| D1 | Tools and trace recorder first, then a thin LLM tool loop (~80 lines). If we fall back to a fixed pipeline, disclose it in the README and keep only argument and tenant-scope trajectory evals. |
| D2 | `verified` is `true`, `false` or `"back-translated"`. Back-translated cases count toward extraction metrics only. Unverified cases never count. About 60% of cases are English. |
| D3 | *(Superseded by D39: no release gate.)* Per-metric gate verdict uses the 95% Wilson interval. Pass: lower >= threshold. Fail: upper < threshold. Otherwise uncertain. Extraction is scored per field. Regression = paired pass-to-fail flips on the same cases. Cross-tenant access, supplier-name leaks and injection compliance are zero-tolerance. |
| D4 | CI runs unit, API and deterministic evals on replay. *(The baseline gating below is superseded by D39.)* Model-graded metrics are gated against the committed live-run record in `evals/baseline.json` (SHA, model, prompt hash, dataset version, timestamp). If prompt or component code changed since, those metrics are "stale" and count as uncertain. Judged (LLM-judge) metrics never gate: see D35. |
| D5 | Prices and line items are deterministic. The LLM writes only the quote cover text; G-Eval and the leak check target it. |
| D6 | Duplicate detection is per tenant. A cross-tenant duplicate behaves exactly like a fresh enquiry (same status code, same shape, new row). |
| D7 | One `MaterialSpec` per enquiry. Stock and suppliers are platform-wide; enquiries, RFQs, quotes and traces are tenant-owned. Exact idempotency replay returns the identical original response. A near-duplicate is stored and escalated by the agent as `duplicate`, never turned into an RFQ (amended by D36). Prompt injection: escalate, no tool calls outside the allowed set, no supplier data in output. |
| D8 | Run history and run comparison live in Confident AI; every eval run is uploaded. This replaces the earlier `results/runs.jsonl` and `scripts/compare_runs.py`. Each run is labelled with `hyperparameters` (model, temperature, prompt version/hash) and `identifier` (one string packing git SHA, dataset version, component version). |
| D9 | *(Superseded by D39: no baseline file.)* The accepted baseline is marked `official=True` on Confident AI **and** written to a committed `evals/baseline.json` (run metadata plus per-case pass/fail) from the same run. The release gate reads only the committed file, so it works offline in CI and reviewers can see it. The README links to or screenshots the Confident AI runs. |
| D10 | CI needs `CONFIDENT_API_KEY` as a secret to upload runs, but no LLM keys. |
| D11 | Golden cases are stored one folder per case (`email.txt`, `case.json`, attachments as real files) under `evals/datasets/<dataset>/`, with the dataset version in `manifest.json`. `evals/runs/shared/golden_cases.py` validates on load and builds DeepEval goldens via `Golden(**case.as_golden_fields())`. Git is the source of truth; Confident AI is for viewing results. |
| D12 | Every case records its `source`: `hand-written`, `llm-drafted` or `production`. |
| D14 | Extraction uses Claude Haiku through the Anthropic Messages API, called with httpx (no SDK dependency), temperature 0 and a forced tool call for structured output. The model ID lives in `Settings.extraction_model`. G-Eval judge choice is paused until Slice 5. |
| D15 | The LLM only reads and classifies (`RawSpec`: values and units as written). Deterministic code normalises grades (`domain/grades.py`), units (`domain/units.py`), coatings and country codes, and infers rolling only where a standard defines it (DC01, DC04). Unnormalisable values become null with a warning, never a guess. |
| D16 | LLM answers are recorded to `fixtures/` (committed), one folder per prompt: `extraction/`, `quote_letter/`, `agent/` (matching calls no model, so has none), by `scripts/record_fixtures.py` and replayed by `ReplayLLMClient`, which never falls back to a live call. The fixture key hashes prompt ID, version, text, model, input and output schema, so any change forces a new recording. API enquiries and golden `email.txt` render identically, so both share recordings. Agent conversations are additionally scoped by golden case, in `fixtures/agent/<case>/` (finding F-007); extraction recordings stay shared. |
| D17 | Out of scope for Slice 1: relative dates ("in three weeks") and calendar weeks ("KW 48") are left null; legacy `.xls` and PDF attachments are skipped with a warning; tenancy uses a temporary `X-Tenant-ID` header (approved) that Slice 3 replaces with API keys (done, D36). |
| D18 | Attachments: CSV (standard library) and `.xlsx` (`openpyxl`, approved) become text tables, every sheet included, formulas shown as their cached values. An unreadable attachment is skipped with a warning and never fails the enquiry. |
| D19 | Matching rules (owner accepted defaults): exact normalised grade; form, and rolling and coating when stated, must match; prime only unless second choice is requested; thickness within max(5%, 0.05 mm), width and length within 1% (no cutting service); one lot must cover the tonnage. Ranking by delivered cost per tonne = price + 0.08 EUR/t/km x great-circle distance, ties broken by stock ID. Unknown delivery location ranks by price with a warning. Rules live in `domain/matching.py` and `domain/tolerances.py`. |
| D20 | Matches are internal: they name suppliers and cost prices, which would reveal supplier identity and the platform's margin. They are stored with the enquiry for the agent and RFQs; the buyer API shows only status `matched` and `offers_available`. Zero-tolerance API tests scan responses for supplier names, IDs, cities, stock IDs and cost fields. |
| D21 | Slice order changed to 2 -> 4 -> 5 -> 3 -> 6 to reach the agent and LLM-judge metrics sooner. Judge model for G-Eval and other judge metrics: OpenAI (a different model family from the Claude extractor, avoiding self-preference bias). A judge experiment on extraction compares judge verdicts with FieldMatch ground truth; learning only, never a gate. |
| D22 | Matching evals run against the deterministic seed stock (`build_stock(DEFAULT_SEED)`), not the database, so a result changes only when the matcher does; the runner logs `stock_seed` as a hyperparameter. **For the test strategy:** in production, each matching dataset version would be labelled against a dated stock snapshot (for example "stock as of 2026-10-01") recorded in its manifest; the replica's fixed seed plays the role of that snapshot. Database loading and pipeline matching are covered by the storage round-trip and API matching tests instead. |
| D23 | Everyday commands live in the `Makefile`: `make check`, `make seed`, `make eval-extraction`, `make eval-extraction-live`, `make eval-matching`, `make record-fixtures`. Module paths: `evals.runs.extraction`, `evals.runs.matching`, `scripts.record_fixtures`, `scripts.annotate_trace`. *Since D41: evals start with `python -m evals.run <eval>`; matching is a unit test and has no eval target.* |
| D24 | The agent is Claude Haiku (`Settings.agent_model`) in a tool loop (`steel/agent/runner.py`, prompt `agent_v1.md`) over five tools: extract_specs, check_duplicate, search_stock, create_rfq, escalate_to_human. draft_quote arrives in Slice 5. Every run must end with exactly one decision: an RFQ or an escalation. |
| D25 | Rules enforced by code, not trusted to the prompt: the tenant comes from `ToolContext` and no tool takes a tenant argument; search_stock needs an extracted spec; create_rfq accepts only lots the last search offered; the database allows one RFQ per enquiry; escalation reasons agent_step_limit and agent_gave_no_decision are reserved for code; the loop stops after 8 turns and a run without a decision is escalated. |
| D26 | check_duplicate uses an exact fingerprint (case, spacing and Re:/Fwd:/AW:/WG: prefixes ignored), scoped to the tenant. Near-duplicates are Slice 3. A duplicate is escalated, never turned into an RFQ. |
| D27 | Every agent run is stored (`agent_runs`, internal) as a trace of LLM turns and tool calls with arguments, results, tokens and latency; this feeds the agent metrics. It also keeps what each turn received (the opening message, then the tool results or refusals it asked for) and, for a model a tool calls, what that model was sent and answered, so every span on Confident AI shows its input and output; the eval places spans on the clock with their recorded durations and marks refused tool calls as errors (owner, 2026-10-10). API and unit tests drive the agent with a reference policy (`tests/factories/llm.py`), the path prompt v1 describes as code; only live or replayed runs exercise the real model. `steel/sandbox.py` runs one enquiry in a fresh database with predictable IDs so recorded agent runs replay exactly. |
| D28 | Agent golden cases (`evals/datasets/agent`, kind `agent`): an email plus `expected_agent` (decision, RFQ lot or acceptable escalation reasons, tool path, forbidden tools), a required `risk`, optional `earlier_enquiries` (same or other tenant), `counterfactual_group` (fairness: cases differing only in sender, language or tone must get the same outcome) and `known_limitation` (expected to fail until a later slice; never hidden). |
| D29 | Quote pricing (owner, Slice 5): a 4% margin on the delivered cost (supplier price plus freight), the price per tonne rounded up to the next whole euro, EUR net of VAT, valid for 3 working days (Monday to Friday, holidays ignored). No quote without a known freight cost. The buyer's line shows the lot actually supplied. Rules in `steel/domain/quote.py`; the buyer's price is `price_per_tonne_eur`, never the internal cost name `price_eur_per_t`. |
| D30 | The agent drafts the quote with a `draft_quote` tool right after `create_rfq` (owner's choice: DeepEval evaluates agents, so the agent owns the step). An RFQ alone does not end a run: a run that stops after the RFQ is escalated by code. **Alternative considered:** code drafts the quote automatically after a successful `create_rfq`. Trade-offs: the alternative has one fewer step for the model to get wrong, kept every Slice 4 recording valid, and makes "a quote for every RFQ" a code guarantee; the chosen design lets the evals measure whether the agent completes the sale (tool path, task completion), at the cost of re-recording every agent case and relying on the code safety net when the model stops early. |
| D31 | The quote's cover letter is the only model-written text a buyer reads. The writer sees the priced quote and the customer's language only (no supplier data, no customer email); code scans the letter for supplier identity before it is stored. The customer's language is extracted (extraction prompt v3); unknown means English. |
| D32 | Judge: `gpt-6-sol` (owner: OpenAI's equivalent of Claude Sonnet; same tier and price in DeepEval 4.2.8), pinned in `evals/runs/shared/judge.py`. Judged metrics are report-only (D35). Run modes: replay (deterministic only, CI), live (record and judge), judge-recorded (judge recorded outputs again). Letter: G-Eval quality, PromptAlignment, NonAdvice (financial, legal), PIILeakage; agent: ArgumentCorrectness, TaskCompletion. Not added: Bias and Toxicity (little information on letters built from structured lines), StepEfficiency (the exact tool-path check covers it deterministically), PlanAdherence and PlanQuality (they need a written plan the agent does not produce). |
| D33 | After the Slice 5 live run (owner): the token and cost limits are rebased to the quote path with the same rule, a clean path plus one spare turn: 21,000 input tokens, 1,500 output, $0.030 per enquiry. Obvious injections (visible in the subject or body by the product's own patterns, or a removed fake end-of-email marker) are escalated at once, before any other tool (agent prompt v3); an injection hidden in an attachment is escalated after the extraction warning. |
| D34 | After the judged Slice 5 runs (owner): G-Eval letter quality passes at 0.8 (at the 0.5 default a letter calling a coil a tube passed); a deterministic "Names only products we sell" check backs it up. Letter prompt v2 asks for the full description (dimensions thickness × width × length, quality), the trade word for the form, and no invented sender name; Prompt Alignment's instruction list follows the prompt. The argument judge reads each tool's contract generated from its schema; the letter judge explains in English. **Amended (owner, after the agent-v8 run):** Argument Correctness judges only tools that take arguments (`evals/metrics/agent/argument_correctness.py`). DeepEval's judge prompt answers "no" for every call without input parameters, so extract_specs and check_duplicate capped each path's score by their count (duplicate escalation at most 0.33, full sale 0.6) and caused 7 false failures; the contract text could not change that. Calls are chosen by the tool's schema, not by the call, so a search_stock call missing its spec is still judged. What the left-out calls returned (the extracted spec, attachment warnings, duplicate_of) is shown to the judge with the email: the first re-judged run without it failed every duplicate escalation for "no evidence", because the judge could not see what the agent saw. |
| D35 | Owner, after the Slice 5 judged runs: the release gate never needs calibration. Judged metrics (G-Eval, PromptAlignment, NonAdvice, PIILeakage, ArgumentCorrectness, TaskCompletion) are report-only: the gate shows them as advisory and they never change pass, fail or uncertain. Only deterministic metrics gate. Why: a judge may only block a release once it is shown to agree with a human, and hand-labelling verdicts is work the owner chose not to do now; the live run showed the risk (Argument Correctness failed three duplicate cases for omitting arguments the tools do not take). Cost: letter and argument quality cannot stop a release; the deterministic letter checks (only quoted figures, no supplier, only products we sell) and the exact tool-path checks carry the gate. To gate on a judge later: label ~10 verdicts per metric, commit them (D9: the gate reads only committed files), and gate only on metrics that agree. The calibration-sheet script that prepared this was removed; it is in git history (commit fc735f6). |
| D36 | Slice 3 (owner, 2026-10-10). **API keys:** an `X-API-Key` header identifies the tenant; `X-Tenant-ID` is gone. Keys are stored as SHA-256 hashes (long random keys, so a fast hash suffices); development keys for the three seed tenants are published in `steel/storage/seed.py`. An unknown key gets the same 401 `INVALID_API_KEY` as a missing one. **Duplicates:** an enquiry duplicates an earlier one of the same tenant when either the fingerprint matches exactly, now including attachment content (F-005), or the extracted order matches (grade, form, thickness, width, length, tonnage and delivery place, after normalisation) and the earlier enquiry arrived within 14 days. Compared on the extracted order, not the text, so forwards, replies quoting the original, rewordings and translations are caught; a new spreadsheet is a new order. Trade-off: the identical lot ordered twice within 14 days is escalated to a human (a delay), which the business prefers to a second RFQ. Text similarity was rejected: it misses rewordings and translations and cannot see attachments. **D7 amended:** every enquiry is stored and the agent escalates a duplicate (as D26), because the order-based check needs extraction; the buyer's response names the earlier enquiry in `duplicate_of`. **Idempotency** as the Slice 3 interface: `Idempotency-Key`, unique per tenant; same key and body replays the original response with no second agent run; same key with a different body is 409. **Amended (owner, 2026-10-10):** the 14-day window applies to exact copies too; an enquiry that failed on our side is not on file, so a resend after it is handled (it has no RFQ to double); a correction (F-012) is recognised by extraction (`is_correction`, extraction prompt v4) and points to the tenant's latest enquiry on file, escalated as a duplicate with `is_correction: true` for the trader (agent prompt v4). Rules in `steel/domain/duplicates.py`. |
| D37 | Owner, finding F-011: the generic flat-steel words (German Blech, Polish blacha) are decided by thickness, by the DIN convention: sheet under 3 mm, plate from 3 mm (`steel/domain/forms.py`). The extractor reports them as `sheet_or_plate` (extraction prompt v4) and never decides by thickness itself (D15); explicit words (Feinblech, Grobblech, blacha gruba, sheet, plate) are kept as written. Without a thickness the form stays empty with a warning. Seed stock still lists some sheet lots of 3 mm and more; they are left as they are, so stock IDs in golden cases stay stable. |
| D38 | Owner, 2026-10-11, before Slice 6. **Verified cases:** at least 40% of each dataset is marked `verified: true` (agent 18/44, duplicates 8/18, extraction 5/11, matching 2/5, quote letter 4/8), so the gate has cases to judge. Claude chose them to cover every risk, language and outcome, verified counterfactual groups whole (a group counts only when every member is verified) and included the known failure agent-024; the owner did not re-review the labels case by case. German and Polish cases are marked `true`, not `back-translated` (D2): no back-translation was done. **Readable reasons:** a case's `reason` is plain words for a reviewer in Confident AI; finding and decision codes go in the new `references` field (e.g. `["F-012"]`). Loading rejects a reason containing such codes or snake_case names (`stk_058`, `no_stock`). |
| D39 | Owner, 2026-10-11: **no release gate.** DeepEval's results (Confident AI and the console) are the verdict; `scripts/release_gate.py` and `evals/baseline.json` are not built, which supersedes D3, D9 and the baseline part of D4. `verified` is a filter (`--verified-only` on every eval run), not a gate. **CI** (`.github/workflows/ci.yml`, every push and pull request, free): `make check` plus every eval on recorded answers, no keys and no upload; a replay run exits 1 when any case fails, because replay is deterministic (owner, option a). The agent and quote-letter replays get a placeholder `OPENAI_API_KEY`: ToolCorrectnessMetric demands one when built but replay calls no model. **Live evals** (`.github/workflows/live-evals.yml`): started by hand, verified cases only, keys from GitHub secrets, `CONFIDENT_TRACE_FLUSH=1`; they report and never fail the job, since live output varies; new recordings are a 7-day artifact, never committed. Counterfactual-group and latency verdicts are printed, not DeepEval results, so they never fail CI. |
| D40 | Owner, 2026-10-11: **the product records its own trace.** The agent loop, every model call and every tool call are `@observe`d where they happen (`steel/tracing.py`; `deepeval` is a runtime dependency), so an eval run and a production run produce the same spans: real order and durations, a refused tool call as an errored span, the model calls tools make nested inside the tool's span, model and tokens with the agreed price per token (`steel/llm/pricing.py`) on every LLM span. The eval no longer rebuilds spans from the stored `AgentRun` (the `Timeline` and span re-emission in `evals/runs/agent.py` are gone); each case runs inside one eval-side root span so the earlier enquiries' extraction calls and the agent's run form one trace, named after the case and tagged (`dataset:`, `mode:`, `verified`/`unverified`, `language:`, `risk:`, `known-limitation`; `evals/runs/shared/trace_tags.py`). `CONFIDENT_TRACE_ENVIRONMENT` cannot tell live from replay: DeepEval files every eval trace under `testing`, so the mode is a tag. The stored `AgentRun` (D27) stays the product's record for the API and the feedback loop. Agreed target for the eval side (design review, 2026-10-11): metrics read the public `LLMTestCase` fields (`tools_called`, `actual_output`, `expected_output`) rather than an `AgentOutcome` blob; eval runs move to pytest with `assert_test(golden=...)`, which adds per-case metrics (`ToolPermissionMetric(denied_tools=...)`) to the shared list where `evals_iterator(metrics=...)` would replace them; the quote-letter eval folds into the agent eval and matching moves to pytest; prompts are pushed to Prompt Studio once per version and linked on spans, datasets pushed once per manifest version with verified cases finalized, both never pulled at run time (git stays the source of truth, D11); replay is removed in its own step once the live spans are verified against the recordings. |
| D41 | 2026-10-11, D40 step 3: **evals are pytest files under `deepeval test run`.** `evals/test_<eval>.py` runs one golden per test and calls `assert_test(golden=..., metrics=...)`, so each case gets its own metrics (the golden's forbidden tools as `ToolPermissionMetric(denied_tools=...)`) on top of the shared list, and the cover letter the agent writes gets the letter checks on its `write_cover_letter` span (`evals/harness/spans.py`), the same three checks as the stand-alone letter eval. `uv run python -m evals.run <eval> [--live \| --judge-recorded] [--verified-only]` adds the run identifier and the replay placeholder key; the mode flags are pytest options (`evals/conftest.py`), `--verified-only` deselects unreviewed cases. Matching is deterministic code and moved to `tests/unit/test_matching_golden_cases.py`. Run-wide reports (fairness, cost, latency, duplicate rates) print when the module's cases finish. **Change to D39:** a failed deterministic check fails the run in every mode, live included, because `assert_test` raises; judged metrics are `flaky` in DeepEval's sense, scored and shown but never deciding a case (D35). The letter dataset is kept: its 8 cases cover coils, coatings, DDP and an unknown language, which no agent case reaches (owner to confirm). |
| D42 | 2026-10-11, D40 step 4: **prompts and datasets on Confident AI, pushed from git, never pulled at run time** (D11 stands). `make push-prompts` (`scripts/push_prompts.py`) pushes each product prompt's text once per `PROMPT_VERSION` and records the Prompt Studio commit hash and a text digest in `evals/prompts.json`; a changed text under the same version is refused, a pushed version is skipped. `make push-datasets` pushes each golden dataset to the alias `<dataset>-v<manifest version>`: verified cases finalized, unverified queued for review, and only names not already on the alias (pushing appends). `make pull-verified` marks a case `verified: true` when its golden was finalized on Confident AI, never the other way, and leaves `back-translated` alone. At run time the product names the prompt on every LLM span (`steel.tracing.record_prompt`: alias and git version), the eval adds the registered commit hash (`evals/harness/prompts.py`), and the run's hyperparameters name alias and commit. **Two DeepEval 4.2.8 limits:** a span's prompt fields are lost inside a test run (serialised under `promptAlias`, read back under `prompt_alias`, between the reloads `deepeval test run` does), so the link is also written to span metadata, which survives; and a `Prompt` given as a hyperparameter is pushed again on every keyed run unless DeepEval's private prompt id is set, so hyperparameters carry plain text. Dataset versions on Confident AI need the Team plan, hence the version in the alias. Not verified against Confident AI yet: the scripts need keys, the owner runs them. |
| D43 | Owner, 2026-10-11, D40 step 5: **no replay.** The first live run under the pytest harness (`cmv3vouyt001dl60t4xal6qrc`, 18 verified cases) gave the same verdict as the recordings on every case, which was the condition for removing replay (D40, option A). Gone: `steel/llm/replay.py`, `fixtures/` (214 recordings), the record and prune scripts, the run modes, the placeholder key. Every eval calls Claude; judged metrics always run and never decide (D35). `ci.yml` runs `make check` only; evals run on request through `live-evals.yml` (`--verified-only`), no schedule yet. Supersedes D4 and D16, and the replay part of D39. Also found on that run: F-014. |
| D44 | Owner, 2026-10-11: **a run links to its dataset on Confident AI.** DeepEval stamps a test run with a dataset only through goldens that were pulled, so the first live run showed "Create dataset". A run now pulls the alias of its manifest version, matches each golden to its case by the `case` key, checks that input and expected output equal git's, and scores the pulled golden; a difference stops the run (`DatasetDiffersFromGit`): git stays the source of truth (D11), the copy must agree with it. A case not on the alias runs from git, unlinked, and is named in the output; without a key nothing is pulled. **Third DeepEval 4.2.8 limit:** on the trace-scoped path the golden's dataset link is handed to a pydantic model as private keywords and dropped, so the first dispatched run (sha 93020f6) still showed no dataset; the run is now stamped by hand from the pulled golden (`link_run_to_dataset`, public fields on DeepEval's `TestRun`). The Diagnosis diagram showing `write_cover_letter` as the LLM component is Confident AI's own inference (the only LLM span carrying metrics); the agent is the `agent` span. |
| D13 | Slice 4 adds a feedback-loop script: escalated or wrongly handled enquiries become candidate case folders (`verified: false`, `source: production`) with the system's extraction pre-filled for correction. Full production sampling (stratified, PII-scrubbed, tenant-consented) is described in the test strategy as a planned layer; a tenant's enquiry never enters a shared dataset unless anonymised and permitted. |

## Repo layout

```
src/steel/
  domain/          pure business rules (no I/O)
    grades.py        grade equivalences, e.g. St37-2 -> S235JR                (Slice 1)
    tolerances.py    dimension tolerances for matching                         (Slice 2)
  storage/         SQLite schema, connection, repositories, seed
  llm/             LLMClient protocol, Anthropic client, record/replay      (Slice 1)
  extraction/      extractor and attachment parsing                          (Slice 1)
  agent/           tools, trace recorder, tool loop                          (Slice 4)
  api/             FastAPI app factory, auth, routes, schemas
  quotes/          pricing and the cover-letter writer                        (Slice 5)
  tracing.py       the spans the product records as it runs                   (D40)
  settings.py      runtime configuration only (database path, service limits)
evals/
  test_<eval>.py   one pytest file per eval, run by `deepeval test run` via evals/run.py (D41)
  harness/         golden case loader, modes, run labels, trace tags, per-eval helpers, prompt registry
  metrics/         custom DeepEval metrics, one business rule each
  datasets/        golden cases, one folder per case, a versioned manifest per dataset (D11)
  prompts.json     which Prompt Studio commit each prompt version became (D42)
scripts/           seed, record and prune fixtures, run one case, push prompts and datasets, pull reviews,
                   export feedback-loop candidates, annotate a trace
tests/             unit/, api/, agent/, storage/, evals/ (the harness itself), scripts/, factories/
fixtures/          recorded LLM answers: extraction/, quote_letter/, agent/<case>/ (D16)
docs/              plan.md, findings.md, test-strategy.md
```

## Interfaces by slice

### Slice 0: Foundation
```python
# steel.domain.models
class Tenant(BaseModel): id: str; name: str; country: str
class Supplier(BaseModel): id: str; name: str; city: str; country: str; latitude: float; longitude: float
class StockItem(BaseModel):
    id: str; supplier_id: str; grade: str; form: ProductForm
    thickness_mm: float; width_mm: float; length_mm: float | None   # None for coils
    quantity_t: float; coating: str | None; quality: Quality; price_eur_per_t: Decimal
class MaterialSpec(BaseModel):
    grade: str | None; grade_raw: str | None; grade_ambiguous: bool = False
    form: ProductForm | None
    thickness_mm / width_mm / length_mm: float | None
    quantity_t: float | None; coating: str | None; quality: Quality | None
    incoterm: Incoterm | None
    delivery_city: str | None; delivery_country: str | None        # ISO 3166-1 alpha-2
    delivery_date: date | None
    missing_fields -> list[str]                                    # property
class EnquiryStatus(StrEnum): RECEIVED, EXTRACTED, MATCHED, QUOTED, ESCALATED, FAILED

# steel.storage
connect(path) -> sqlite3.Connection;  create_schema(conn);  seed_database(conn)
list_suppliers(conn) -> list[Supplier];  list_stock(conn) -> list[StockItem];  get_tenant(conn, tenant_id)

# API
create_app(settings) -> FastAPI;  GET /health
```

### Slice 1: Extraction
```python
class LLMClient(Protocol):
    def complete_structured(self, *, prompt_id, prompt_version, messages, schema: type[T]) -> LLMResult[T]
class ReplayLLMClient(LLMClient)   # raises MissingFixture; never falls back to a live call
class EnquiryDocument(BaseModel): subject: str; body: str; attachments: list[Attachment]
class RawSpec(BaseModel)           # LLM output: grade as written, value + unit pairs
normalise_grade(raw: str) -> GradeResult            # grade | None, ambiguous, candidates
to_tonnes(value: float, unit: str) -> float         # raises UnknownUnit
normalise_spec(raw: RawSpec) -> MaterialSpec        # pure
extract_spec(doc, llm) -> ExtractionResult          # raw, spec, warnings, usage
POST /v1/enquiries  EnquiryCreate -> 201 EnquiryOut{id, status, status_history, spec}
```

### Slice 2: Matching
```python
find_matches(spec, stock, suppliers, delivery: DeliveryLocation) -> MatchOutcome
class MatchOutcome: matches: list[Match]; rejections: list[Rejection]; missing_fields; warnings
class Match: rank; stock_item_id; supplier_id; price_eur_per_t; distance_km; freight_eur_per_t; delivered_eur_per_t
class Rejection: stock_item_id; reasons: list[RejectionReason]   # every broken constraint
locate_delivery(city, country) -> DeliveryLocation                 # city table, country centre, or unknown
EnquiryOut.offers_available: int                                   # matches stay internal (D20)
```

### Slice 3: Duplicates and tenancy
```python
X-API-Key -> tenant (dependency);  Idempotency-Key header
fingerprint(doc) -> str                              # strips Fwd:/Re:, quoted headers, signatures, whitespace
find_near_duplicate(conn, tenant_id, doc) -> DuplicateCheck
POST /v1/enquiries: 201 new | 200 key replay (identical body) | 200 near-duplicate (duplicate_of) | 409 key reuse with new body
GET /v1/enquiries/{id}: 404 for another tenant's enquiry
UNIQUE (tenant_id, idempotency_key)
```

### Slice 4: Agent
```python
ToolContext(tenant_id, enquiry_id, repos, trace)    # tenant injected, never an LLM argument
extract_specs(ctx) / check_duplicate(ctx) / search_stock(ctx, spec, limit) /
create_rfq(ctx, stock_item_ids) / draft_quote(ctx, rfq_id) / escalate_to_human(ctx, reason, detail)
run_agent(ctx, llm, max_steps=8) -> AgentRun
TraceEvent(run_id, seq, tenant_id, kind, name, args, result, error, latency_ms, tokens_in, tokens_out, cost_eur)
GET /v1/enquiries/{id}/trace -> AgentTrace
UNIQUE (enquiry_id) on rfqs
```

### Slice 5: Quote
```python
price_quote(rfq, stock) -> QuoteLines               # deterministic
write_cover_text(lines, llm) -> str                 # LLM, only free text
find_supplier_leaks(text, suppliers) -> list[Leak]  # deterministic
GET /v1/quotes/{id} -> BuyerQuote                   # no supplier fields in the schema
```

### Slice 6: Release
**Done.** `.github/workflows/ci.yml` runs `make check` and the replay evals on every push (D39); `live-evals.yml`
runs the verified cases live on request. `README.md` and `docs/test-strategy.md` describe the harness and its
decisions. The regression demo is F-013 (`agent-024`, red in CI until fixed). The release gate was dropped (D39).

## Planned eval metrics by slice

Agreed list of DeepEval built-in and custom metrics. Claude flags each group at
the start of its slice. Metric display names (`__name__`) are business-readable
statements where a pass is good news; class names may stay technical. Rule: deterministic wherever the answer is knowable;
judge-model metrics only where code cannot decide.

| Slice | Metric | Type | Checks | Risk |
|---|---|---|---|---|
| 1 | `FieldMatch` per field | Custom, deterministic | **Done.** Each of 14 MaterialSpec fields against its label | Extraction |
| 1 | `JsonCorrectnessMetric` | Built-in | Optional: output conforms to the schema. Low value, since the forced tool call and RawSpec validation already enforce it | Extraction |
| 2 | **No unsuitable offers** (hard constraints) | Unit test, zero tolerance | **Moved** (D41) to `tests/unit/test_matching_golden_cases.py`: none of the labelled "must not match" stock is offered: wrong grade, size outside tolerance, second choice for prime | Business correctness |
| 2 | **Best offer comes first** (top-1) | Unit test | **Moved** (D41): the expected best offer is ranked first | Business correctness |
| 2 | **Best offer in the top 3** (top-k, k=3) | Unit test | **Dropped** (D41): implied by best offer first | Business correctness |
| 2 | **Offers in the right order** (ranking) | Unit test | **Moved** (D41): expected offers appear in the expected order, by delivered cost | Business correctness |
| 2 | **Finds every suitable offer** (completeness) | Unit test | **Moved** (D41): every expected offer appears and nothing else; catches lost sales | Business correctness |
| 3 | Near-duplicate precision and recall | Custom, deterministic | **Done** (`evals/test_duplicates.py`, `make eval-duplicates`). "Duplicate decision is correct" per labelled pair; precision and recall printed per run. Pairs: forwarded, reply quoting, reworded, translated, kilograms, correction (known limitation), new spreadsheet, other tenant, other tonnage, thickness, city, outside the window | Duplication |
| 4 | `ToolPermissionMetric` | Built-in, deterministic | **Done** (D41): built per case from the golden's `forbidden_tools` (`evals/harness/agent.py::agent_metrics`), zero tolerance | Data separation, business correctness |
| 4 | `ToolCorrectnessMetric` | Built-in, deterministic | **Done**, exact match on the golden's `tool_path` (names and order); arguments have their own check | Business correctness |
| 4 | `AgentLoopDetectionMetric` | Built-in, deterministic | **Done.** No repeated or cyclic tool calls | Business correctness |
| 4 | Tenant scope | Unit test | **Moved** (D40): `tests/agent/test_runner.py::test_run_acts_only_for_the_enquirys_tenant`; the eval metric only re-read a stamp code set, the test checks what the run created | Data separation |
| 4 | Cost and latency | Custom, run-wide reports | **Done**, printed per run, never per case (D40): `evals/metrics/agent/cost.py` (tokens, dollars, unpriced models; same in replay) and `latency.py` (live only). Confident AI prices each LLM span itself | Release criteria |
| 4 | `ArgumentCorrectnessMetric` | Built-in, judge | **Dropped** (D40): judging only tools with arguments (D34) needed a subclass and a contract text per tool; the arguments that matter have deterministic checks | Business correctness |
| 4 | `StepEfficiencyMetric` | Built-in, judge | **Not built:** the exact tool-path check already flags every extra step, deterministically | Business correctness |
| 4 | `TaskCompletionMetric` | Built-in, judge | **Done**, live and judge-recorded modes, `flaky` (report-only, D35, D41); its value is on enquiries without labels | Business correctness |
| 5 | Supplier leak check | Custom, deterministic, zero tolerance | Supplier names, IDs (sup_, stk_) or addresses in cover text. `PatternMatchMetric` passes on a match, so it cannot be used for this. Since D41 the three letter checks also run on every letter the agent writes, on its `write_cover_letter` span | Data separation |
| 5 | `GEval` quote quality | Built-in, judge | **Done**, judge `gpt-6-sol` (D32), threshold 0.8 (D34), `flaky` (report-only): commercially sensible, professional, in the customer's language | Business correctness |
| 5 | `FaithfulnessMetric` / `HallucinationMetric` | Built-in, judge | **Not built:** "Uses only the quoted figures" checks every number and date in the letter against the quote deterministically | Business correctness |
| 5 | `PromptAlignmentMetric` | Built-in, judge | **Dropped** with `NonAdviceMetric` (D40): both scored 1.0 on every letter; figures, supplier names and product words have deterministic checks | Business correctness |
| 5 | `PIILeakageMetric` | Built-in, judge, secondary | **Done**, `flaky` (report-only): indirect leaks the deterministic check misses, e.g. "our partner mill in Linz" | Data separation |
| 5 | `DAGMetric` | Built-in, judge, option | **Not built:** one G-Eval at 0.8 plus the deterministic letter checks have been enough; revisit if G-Eval scores prove inconsistent | Business correctness |
| 6 | Judge calibration | Process | **Deferred (D35).** Judged metrics stay report-only, so the gate needs no calibration | All |

Not relevant here: RAG contextual metrics, conversational and turn metrics, image, audio, voice, MCP, bias, toxicity, summarisation, non-advice.

## Testability requirements

1. LLM injected through `create_app` and dependencies. Replay client fails loudly on a missing fixture.
2. Fixture key = hash(prompt_id, prompt_version, model, normalised input). Recording only via an explicit script.
3. `steel.domain` is pure.
4. `tenant_id` is a required argument in every tenant-owned repository function. The LLM never supplies it.
5. Idempotency and one-RFQ-per-enquiry enforced by UNIQUE constraints.
6. Status changes go through one function with a transition table and append to `status_history`.
7. Buyer-facing schemas contain no supplier fields; a leak scanner checks free text.
8. Matcher returns rejection reasons; normaliser flags ambiguity instead of guessing.
9. Traces record model, prompt version, tokens, cost, latency (fixture latency in replay).
10. Clock and ID generation injectable; seed data deterministic.
11. Error responses use stable codes.
12. Per-test SQLite database via the app factory.

## GitHub Actions budget

About 570 minutes. `make check` locally before every push. CI: one Ubuntu job, uv cache,
10-minute timeout, cancel superseded runs, skip docs-only changes. Live evals run locally
or via manual `workflow_dispatch` only.
