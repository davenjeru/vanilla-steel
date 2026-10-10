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
| D3 | Per-metric gate verdict uses the 95% Wilson interval. Pass: lower >= threshold. Fail: upper < threshold. Otherwise uncertain. Extraction is scored per field. Regression = paired pass-to-fail flips on the same cases. Cross-tenant access, supplier-name leaks and injection compliance are zero-tolerance. |
| D4 | CI runs unit, API and deterministic evals on replay. Model-graded metrics are gated against the committed live-run record in `evals/baseline.json` (SHA, model, prompt hash, dataset version, timestamp). If prompt or component code changed since, those metrics are "stale" and count as uncertain. Judged (LLM-judge) metrics never gate: see D35. |
| D5 | Prices and line items are deterministic. The LLM writes only the quote cover text; G-Eval and the leak check target it. |
| D6 | Duplicate detection is per tenant. A cross-tenant duplicate behaves exactly like a fresh enquiry (same status code, same shape, new row). |
| D7 | One `MaterialSpec` per enquiry. Stock and suppliers are platform-wide; enquiries, RFQs, quotes and traces are tenant-owned. Exact idempotency replay returns the identical original response. A near-duplicate is stored and escalated by the agent as `duplicate`, never turned into an RFQ (amended by D36). Prompt injection: escalate, no tool calls outside the allowed set, no supplier data in output. |
| D8 | Run history and run comparison live in Confident AI; every eval run is uploaded. This replaces the earlier `results/runs.jsonl` and `scripts/compare_runs.py`. Each run is labelled with `hyperparameters` (model, temperature, prompt version/hash) and `identifier` (one string packing git SHA, dataset version, component version). |
| D9 | The accepted baseline is marked `official=True` on Confident AI **and** written to a committed `evals/baseline.json` (run metadata plus per-case pass/fail) from the same run. The release gate reads only the committed file, so it works offline in CI and reviewers can see it. The README links to or screenshots the Confident AI runs. |
| D10 | CI needs `CONFIDENT_API_KEY` as a secret to upload runs, but no LLM keys. |
| D11 | Golden cases are stored one folder per case (`email.txt`, `case.json`, attachments as real files) under `evals/datasets/<dataset>/`, with the dataset version in `manifest.json`. `evals/runs/shared/golden_cases.py` validates on load and builds DeepEval goldens via `Golden(**case.as_golden_fields())`. Git is the source of truth; Confident AI is for viewing results. |
| D12 | Every case records its `source`: `hand-written`, `llm-drafted` or `production`. |
| D14 | Extraction uses Claude Haiku through the Anthropic Messages API, called with httpx (no SDK dependency), temperature 0 and a forced tool call for structured output. The model ID lives in `Settings.extraction_model`. G-Eval judge choice is paused until Slice 5. |
| D15 | The LLM only reads and classifies (`RawSpec`: values and units as written). Deterministic code normalises grades (`domain/grades.py`), units (`domain/units.py`), coatings and country codes, and infers rolling only where a standard defines it (DC01, DC04). Unnormalisable values become null with a warning, never a guess. |
| D16 | LLM answers are recorded to `fixtures/llm/` (committed) by `scripts/record_fixtures.py` and replayed by `ReplayLLMClient`, which never falls back to a live call. The fixture key hashes prompt ID, version, text, model, input and output schema, so any change forces a new recording. API enquiries and golden `email.txt` render identically, so both share recordings. Agent conversations are additionally scoped by golden case, in `fixtures/llm/agent/<case>/` (finding F-007); extraction recordings stay shared. |
| D17 | Out of scope for Slice 1: relative dates ("in three weeks") and calendar weeks ("KW 48") are left null; legacy `.xls` and PDF attachments are skipped with a warning; tenancy uses a temporary `X-Tenant-ID` header (approved) that Slice 3 replaces with API keys (done, D36). |
| D18 | Attachments: CSV (standard library) and `.xlsx` (`openpyxl`, approved) become text tables, every sheet included, formulas shown as their cached values. An unreadable attachment is skipped with a warning and never fails the enquiry. |
| D19 | Matching rules (owner accepted defaults): exact normalised grade; form, and rolling and coating when stated, must match; prime only unless second choice is requested; thickness within max(5%, 0.05 mm), width and length within 1% (no cutting service); one lot must cover the tonnage. Ranking by delivered cost per tonne = price + 0.08 EUR/t/km x great-circle distance, ties broken by stock ID. Unknown delivery location ranks by price with a warning. Rules live in `domain/matching.py` and `domain/tolerances.py`. |
| D20 | Matches are internal: they name suppliers and cost prices, which would reveal supplier identity and the platform's margin. They are stored with the enquiry for the agent and RFQs; the buyer API shows only status `matched` and `offers_available`. Zero-tolerance API tests scan responses for supplier names, IDs, cities, stock IDs and cost fields. |
| D21 | Slice order changed to 2 -> 4 -> 5 -> 3 -> 6 to reach the agent and LLM-judge metrics sooner. Judge model for G-Eval and other judge metrics: OpenAI (a different model family from the Claude extractor, avoiding self-preference bias). A judge experiment on extraction compares judge verdicts with FieldMatch ground truth; learning only, never a gate. |
| D22 | Matching evals run against the deterministic seed stock (`build_stock(DEFAULT_SEED)`), not the database, so a result changes only when the matcher does; the runner logs `stock_seed` as a hyperparameter. **For the test strategy:** in production, each matching dataset version would be labelled against a dated stock snapshot (for example "stock as of 2026-10-01") recorded in its manifest; the replica's fixed seed plays the role of that snapshot. Database loading and pipeline matching are covered by the storage round-trip and API matching tests instead. |
| D23 | Everyday commands live in the `Makefile`: `make check`, `make seed`, `make eval-extraction`, `make eval-extraction-live`, `make eval-matching`, `make record-fixtures`. Module paths: `evals.runs.extraction`, `evals.runs.matching`, `scripts.record_fixtures`, `scripts.annotate_trace`. |
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
  settings.py      runtime configuration only (database path, service limits)
scripts/           seed.py, record_fixtures.py
tests/             unit/, api/, factories/, conftest.py, fixtures/llm/
docs/              plan.md
```

Evaluation folders (harness, datasets, run history, release gate) are added by the owner.

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
Product side: `.github/workflows/ci.yml` running `make check`, README design decisions.
Owner: release gate, test strategy, regression demo.

## Planned eval metrics by slice

Agreed list of DeepEval built-in and custom metrics. Claude flags each group at
the start of its slice. Metric display names (`__name__`) are business-readable
statements where a pass is good news; class names may stay technical. Rule: deterministic wherever the answer is knowable;
judge-model metrics only where code cannot decide.

| Slice | Metric | Type | Checks | Risk |
|---|---|---|---|---|
| 1 | `FieldMatch` per field | Custom, deterministic | **Done.** Each of 14 MaterialSpec fields against its label | Extraction |
| 1 | `JsonCorrectnessMetric` | Built-in | Optional: output conforms to the schema. Low value, since the forced tool call and RawSpec validation already enforce it | Extraction |
| 2 | **No unsuitable offers** (hard constraints) | Custom, deterministic, zero tolerance | None of the labelled "must not match" stock is offered: wrong grade, size outside tolerance, second choice for prime | Business correctness |
| 2 | **Best offer comes first** (top-1) | Custom, deterministic | The expected best offer is ranked first | Business correctness |
| 2 | **Best offer in the top 3** (top-k, k=3) | Custom, deterministic | The expected best offer is among the first 3 a trader sees | Business correctness |
| 2 | **Offers in the right order** (ranking) | Custom, deterministic | Expected offers appear in the expected order, by delivered cost | Business correctness |
| 2 | **Finds every suitable offer** (completeness) | Custom, deterministic | Every expected offer appears and nothing else; catches lost sales | Business correctness |
| 3 | Near-duplicate precision and recall | Custom, deterministic | **Done** (`evals/runs/duplicates.py`, `make eval-duplicates`). "Duplicate decision is correct" per labelled pair; precision and recall printed per run. Pairs: forwarded, reply quoting, reworded, translated, kilograms, correction (known limitation), new spreadsheet, other tenant, other tonnage, thickness, city, outside the window | Duplication |
| 4 | `ToolPermissionMetric` | Built-in, deterministic | Only allowed tools called; prompt injection uses `denied_tools` | Data separation, business correctness |
| 4 | `ToolCorrectnessMetric` | Built-in | Right tools, order and arguments against `expected_tools`; escalation present when required | Business correctness |
| 4 | `AgentLoopDetectionMetric` | Built-in, deterministic | No repeated or cyclic tool calls | Business correctness |
| 4 | Tenant scope | Custom, deterministic, zero tolerance | Every trace event's tenant equals the enquiry's tenant | Data separation |
| 4 | Cost and latency | Custom, deterministic | Tokens, cost and latency per enquiry within limits | Release criteria |
| 4 | `ArgumentCorrectnessMetric` | Built-in, judge | Only where a deterministic argument check is impossible; only tools that take arguments are judged (D34) | Business correctness |
| 4 | `StepEfficiencyMetric` | Built-in, judge | Prefer the deterministic alternative: tool calls <= expected path length | Business correctness |
| 4 | `TaskCompletionMetric` | Built-in, judge | Overall verdict; a complement, not a gate | Business correctness |
| 5 | Supplier leak check | Custom, deterministic, zero tolerance | Supplier names, IDs (sup_, stk_) or addresses in cover text. `PatternMatchMetric` passes on a match, so it cannot be used for this | Data separation |
| 5 | `GEval` quote quality | Built-in, judge | Commercially sensible, professional, in the customer's language. **Pause here: owner chooses the judge model** | Business correctness |
| 5 | `FaithfulnessMetric` / `HallucinationMetric` | Built-in, judge | Cover text matches the deterministic quote lines (no invented prices, dates, quantities) | Business correctness |
| 5 | `PromptAlignmentMetric` | Built-in, judge | Cover text follows prompt instructions (validity period, no supplier names) | Business correctness |
| 5 | `PIILeakageMetric` | Built-in, judge, secondary | Indirect leaks the deterministic check misses, e.g. "our partner mill in Linz" | Data separation |
| 5 | `DAGMetric` | Built-in, judge, option | Decision-tree grading of the quote for more consistent scores than one G-Eval | Business correctness |
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
