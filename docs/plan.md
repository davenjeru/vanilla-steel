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
| D11 | Golden cases are stored one folder per case (`email.txt`, `case.json`, attachments as real files) under `evals/datasets/<dataset>/`, with the dataset version in `manifest.json`. `evals/golden_cases.py` validates on load and builds DeepEval goldens via `Golden(**case.as_golden_fields())`. Git is the source of truth; Confident AI is for viewing results. |
| D12 | Every case records its `source`: `hand-written`, `llm-drafted` or `production`. |
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
find_matches(spec, stock, suppliers, delivery: GeoPoint, rules: MatchRules) -> MatchOutcome
class MatchOutcome: matches: list[Match]; rejections: list[Rejection]
class Match: stock_item_id; supplier_id; score; breakdown: ScoreBreakdown   # price, distance_km, sub-scores
class Rejection: stock_item_id; reasons: list[RejectionReason]
GET /v1/enquiries/{id}/matches -> MatchesOut (internal view)
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
