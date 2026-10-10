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
| D4 | CI runs unit, API and deterministic evals on replay. Model-graded metrics are gated against the committed live-run record in `evals/baseline.json` (SHA, model, prompt hash, dataset version, timestamp). If prompt or component code changed since, those metrics are "stale" and count as uncertain. |
| D5 | Prices and line items are deterministic. The LLM writes only the quote cover text; G-Eval and the leak check target it. |
| D6 | Duplicate detection is per tenant. A cross-tenant duplicate behaves exactly like a fresh enquiry (same status code, same shape, new row). |
| D7 | One `MaterialSpec` per enquiry. Stock and suppliers are platform-wide; enquiries, RFQs, quotes and traces are tenant-owned. Exact idempotency replay returns the identical original response. A near-duplicate without a key returns the original enquiry with `duplicate_of` and stores nothing. Prompt injection: escalate, no tool calls outside the allowed set, no supplier data in output. |
| D8 | Run history and run comparison live in Confident AI; every eval run is uploaded. This replaces the earlier `results/runs.jsonl` and `scripts/compare_runs.py`. Each run is labelled with `hyperparameters` (model, temperature, prompt version/hash) and `identifier` (one string packing git SHA, dataset version, component version). |
| D9 | The accepted baseline is marked `official=True` on Confident AI **and** written to a committed `evals/baseline.json` (run metadata plus per-case pass/fail) from the same run. The release gate reads only the committed file, so it works offline in CI and reviewers can see it. The README links to or screenshots the Confident AI runs. |
| D10 | CI needs `CONFIDENT_API_KEY` as a secret to upload runs, but no LLM keys. |
| D11 | Golden cases are stored one folder per case (`email.txt`, `case.json`, attachments as real files) under `evals/datasets/<dataset>/`, with the dataset version in `manifest.json`. `evals/runs/shared/golden_cases.py` validates on load and builds DeepEval goldens via `Golden(**case.as_golden_fields())`. Git is the source of truth; Confident AI is for viewing results. |
| D12 | Every case records its `source`: `hand-written`, `llm-drafted` or `production`. |
| D14 | Extraction uses Claude Haiku through the Anthropic Messages API, called with httpx (no SDK dependency), temperature 0 and a forced tool call for structured output. The model ID lives in `Settings.extraction_model`. G-Eval judge choice is paused until Slice 5. |
| D15 | The LLM only reads and classifies (`RawSpec`: values and units as written). Deterministic code normalises grades (`domain/grades.py`), units (`domain/units.py`), coatings and country codes, and infers rolling only where a standard defines it (DC01, DC04). Unnormalisable values become null with a warning, never a guess. |
| D16 | LLM answers are recorded to `fixtures/llm/` (committed) by `scripts/record_fixtures.py` and replayed by `ReplayLLMClient`, which never falls back to a live call. The fixture key hashes prompt ID, version, text, model, input and output schema, so any change forces a new recording. API enquiries and golden `email.txt` render identically, so both share recordings. |
| D17 | Out of scope for Slice 1: relative dates ("in three weeks") and calendar weeks ("KW 48") are left null; legacy `.xls` and PDF attachments are skipped with a warning; tenancy uses a temporary `X-Tenant-ID` header (approved) that Slice 3 replaces with API keys. |
| D18 | Attachments: CSV (standard library) and `.xlsx` (`openpyxl`, approved) become text tables, every sheet included, formulas shown as their cached values. An unreadable attachment is skipped with a warning and never fails the enquiry. |
| D19 | Matching rules (owner accepted defaults): exact normalised grade; form, and rolling and coating when stated, must match; prime only unless second choice is requested; thickness within max(5%, 0.05 mm), width and length within 1% (no cutting service); one lot must cover the tonnage. Ranking by delivered cost per tonne = price + 0.08 EUR/t/km x great-circle distance, ties broken by stock ID. Unknown delivery location ranks by price with a warning. Rules live in `domain/matching.py` and `domain/tolerances.py`. |
| D20 | Matches are internal: they name suppliers and cost prices, which would reveal supplier identity and the platform's margin. They are stored with the enquiry for the agent and RFQs; the buyer API shows only status `matched` and `offers_available`. Zero-tolerance API tests scan responses for supplier names, IDs, cities, stock IDs and cost fields. |
| D21 | Slice order changed to 2 -> 4 -> 5 -> 3 -> 6 to reach the agent and LLM-judge metrics sooner. Judge model for G-Eval and other judge metrics: OpenAI (a different model family from the Claude extractor, avoiding self-preference bias). A judge experiment on extraction compares judge verdicts with FieldMatch ground truth; learning only, never a gate. |
| D22 | Matching evals run against the deterministic seed stock (`build_stock(DEFAULT_SEED)`), not the database, so a result changes only when the matcher does; the runner logs `stock_seed` as a hyperparameter. **For the test strategy:** in production, each matching dataset version would be labelled against a dated stock snapshot (for example "stock as of 2026-10-01") recorded in its manifest; the replica's fixed seed plays the role of that snapshot. Database loading and pipeline matching are covered by the storage round-trip and API matching tests instead. |
| D23 | Everyday commands live in the `Makefile`: `make check`, `make seed`, `make eval-extraction`, `make eval-extraction-live`, `make eval-matching`, `make record-fixtures`. Module paths: `evals.runs.extraction`, `evals.runs.matching`, `scripts.record_fixtures`, `scripts.annotate_trace`. |
| D24 | The agent is Claude Haiku (`Settings.agent_model`) in a tool loop (`steel/agent/runner.py`, prompt `agent_v1.md`) over five tools: extract_specs, check_duplicate, search_stock, create_rfq, escalate_to_human. draft_quote arrives in Slice 5. Every run must end with exactly one decision: an RFQ or an escalation. |
| D25 | Rules enforced by code, not trusted to the prompt: the tenant comes from `ToolContext` and no tool takes a tenant argument; search_stock needs an extracted spec; create_rfq accepts only lots the last search offered; the database allows one RFQ per enquiry; escalation reasons agent_step_limit and agent_gave_no_decision are reserved for code; the loop stops after 8 turns and a run without a decision is escalated. |
| D26 | check_duplicate uses an exact fingerprint (case, spacing and Re:/Fwd:/AW:/WG: prefixes ignored), scoped to the tenant. Near-duplicates are Slice 3. A duplicate is escalated, never turned into an RFQ. |
| D27 | Every agent run is stored (`agent_runs`, internal) as a trace of LLM turns and tool calls with arguments, results, tokens and latency; this feeds the agent metrics. API and unit tests drive the agent with a reference policy (`tests/factories/llm.py`), the path prompt v1 describes as code; only live or replayed runs exercise the real model. `steel/sandbox.py` runs one enquiry in a fresh database with predictable IDs so recorded agent runs replay exactly. |
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
| 3 | Near-duplicate precision and recall | Custom, deterministic | Over labelled email pairs (forwarded, reworded, resent) | Duplication |
| 4 | `ToolPermissionMetric` | Built-in, deterministic | Only allowed tools called; prompt injection uses `denied_tools` | Data separation, business correctness |
| 4 | `ToolCorrectnessMetric` | Built-in | Right tools, order and arguments against `expected_tools`; escalation present when required | Business correctness |
| 4 | `AgentLoopDetectionMetric` | Built-in, deterministic | No repeated or cyclic tool calls | Business correctness |
| 4 | Tenant scope | Custom, deterministic, zero tolerance | Every trace event's tenant equals the enquiry's tenant | Data separation |
| 4 | Cost and latency | Custom, deterministic | Tokens, cost and latency per enquiry within limits | Release criteria |
| 4 | `ArgumentCorrectnessMetric` | Built-in, judge | Only where a deterministic argument check is impossible | Business correctness |
| 4 | `StepEfficiencyMetric` | Built-in, judge | Prefer the deterministic alternative: tool calls <= expected path length | Business correctness |
| 4 | `TaskCompletionMetric` | Built-in, judge | Overall verdict; a complement, not a gate | Business correctness |
| 5 | Supplier leak check | Custom, deterministic, zero tolerance | Supplier names, IDs (sup_, stk_) or addresses in cover text. `PatternMatchMetric` passes on a match, so it cannot be used for this | Data separation |
| 5 | `GEval` quote quality | Built-in, judge | Commercially sensible, professional, in the customer's language. **Pause here: owner chooses the judge model** | Business correctness |
| 5 | `FaithfulnessMetric` / `HallucinationMetric` | Built-in, judge | Cover text matches the deterministic quote lines (no invented prices, dates, quantities) | Business correctness |
| 5 | `PromptAlignmentMetric` | Built-in, judge | Cover text follows prompt instructions (validity period, no supplier names) | Business correctness |
| 5 | `PIILeakageMetric` | Built-in, judge, secondary | Indirect leaks the deterministic check misses, e.g. "our partner mill in Linz" | Data separation |
| 5 | `DAGMetric` | Built-in, judge, option | Decision-tree grading of the quote for more consistent scores than one G-Eval | Business correctness |
| 6 | Judge calibration | Process | Owner judges ~10 cases; each gated judge metric must agree with the human | All |

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
