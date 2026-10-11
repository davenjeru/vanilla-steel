# Test strategy

How the steel enquiry pipeline is tested, what each layer is for, what decides a release, and
what is deliberately left out. Decisions referenced as D-numbers are in `plan.md`; failures as
F-numbers in `findings.md`.

## Principles

1. **Deterministic wherever the answer is knowable.** A grade equivalence, a tolerance, a tool
   path, a figure in a letter: code decides, with a reason a reviewer can read. A model judges
   only what code cannot (is this letter commercially sensible?), and never decides a case (D35).
2. **Test the business risk, not the component.** Every test and golden case traces to one of
   four risks: extraction accuracy, data separation, duplication, business correctness. A case's
   `reason` says in plain words what would go wrong for the business.
3. **The product records its own trace** (D40). Evals read what the product recorded, through
   DeepEval's public fields; nothing is rebuilt after the fact, so an eval run and a production
   run look the same on Confident AI.
4. **Git is the source of truth** (D11). Cases, prompts and the registry of what was pushed live
   in the repository; Confident AI holds copies for review and comparison and is never pulled at
   run time.

## Layers

| Layer | Runs | Answers | Decides? |
|---|---|---|---|
| Unit (`tests/unit/`) | `make check`, every push | Pure domain rules: grades, units, tolerances, forms, fingerprints, injection patterns, supplier-leak scan, the matcher against its labelled cases | Yes |
| API and integration (`tests/api/`, `tests/agent/`, `tests/storage/`) | `make check` | Contracts and schemas, API keys, cross-tenant reads, idempotency keys, status transitions, the agent loop's safety nets (step limit, refused tools, one RFQ per enquiry), that a run acts for one tenant only, the spans the product records | Yes |
| Evals (`evals/test_*.py`) | `make eval-*` with keys from `.env`; the verified cases via the manual workflow | Each golden case through the real pipeline against Claude, deterministic metrics and judged metrics | Deterministic metrics yes: a failed case fails the run; judged metrics report only |
| Run-wide reports | Printed at the end of an eval | Fairness across counterfactual groups, cost per enquiry, latency, duplicate precision and recall | No: printed, never per case |

Model-based grading is used for what code cannot read and never decides a case: whether the agent
completed its task as a whole (`TaskCompletionMetric`) and stayed in its role
(`RoleViolationMetric`, the judged side of the prompt-injection cases, read with the trace tag
`risk:security`); whether a cover letter reads as a sensible commercial
letter (G-Eval), leaks no personal data, gives no legal, tax or customs advice and is not toxic.
All are DeepEval `flaky` metrics: scored and shown, never failing a case. Their judge
(`gpt-6-sol`) accepts no temperature, so its verdicts vary; no calibration against human labels
is claimed.

## Datasets

Five datasets, one folder per case, each dataset with a versioned manifest and a changelog:

| Dataset | Cases | Verified | Covers |
|---|---|---|---|
| extraction | 11 | 5 | English, German and Polish emails; decimal commas; a spreadsheet-only order; missing and ambiguous fields; a conflict between subject and body |
| matching | 5 | 2 | Hard constraints, ranking by delivered cost, second choice only when accepted (run as unit tests, D41) |
| duplicates | 18 | 8 | Forwards, replies, rewording, translation, kilograms, a correction, a new spreadsheet, another tenant, other tonnage, outside the window |
| agent | 44 | 18 | The full path: RFQ and quote, escalations (no stock, missing quantity, ambiguous grade, duplicate, correction), prompt injections in body, subject and attachment, genuine emails that resemble injections, counterfactual groups in three languages |
| quote_letter | 8 | 4 | Letters in three languages, DDP, second choice, coated coil, a large coil to a supplier city |

`verified: true` means a person reviewed the labels; `back-translated` means a translation was
checked that way. The pull-request workflow (on the label `ready-for-live-verified`) runs verified
cases only, so the spend goes to labels a person stands behind; the pre-release workflow runs every case, and a failure on an unverified one
is a question for the reviewer before it is a verdict on the product. A review on Confident AI (finalizing the golden) flows back to git with
`make pull-verified`; a label edit is a commit, through the loader's validation.

Cases are added in three ways: hand-written for a known business rule, drafted by a model and
reviewed (`source: llm-drafted`), and exported from escalated or wrongly handled enquiries
(`make export-candidates`, D13), which arrive as `verified: false` with the system's own
extraction pre-filled for correction.

## Live only

Every eval calls the models (D43). Recording and replaying answers was the harness's first design
(D4, D16) and carried it until the product recorded its own trace and a live run under the final
harness had been checked against the recordings; it then went, because a replay measures the
code around a frozen answer, never the model, and kept two sources of truth. The price is that
nothing about the model's behaviour is checked for free: `make check` covers the code, the evals
cost a few dollars a run and are started on purpose. `--verified-only` deselects unreviewed cases.

## What decides a release

There is no release gate script and no stored baseline (D39). The verdict is DeepEval's: every
case must pass its deterministic metrics, and the run fails otherwise. Runs are compared on
Confident AI by their identifier (`sha=`, dataset version, component version) and
hyperparameters (models, prompt versions and commits, judge). Thresholds are never lowered to
make a run pass; a threshold that looks wrong is a finding.

## Traceability

- Each trace is named after its case and tagged (`dataset:`, `mode:`, `verified`, `language:`,
  `risk:`, `known-limitation`).
- Each LLM span names its model, tokens, agreed price per token, and prompt (alias, git version,
  Prompt Studio commit once pushed, D42).
- Each run names the commit it ran on, with `-dirty` when the tree had uncommitted changes.
- Each finding is annotated on its trace on Confident AI (`scripts/annotate_trace.py`) and logged
  in `findings.md` with the case that reproduces it.

## Deliberately not covered, and why

- **Judge calibration.** Judged metrics never decide, so agreement with human labels is not
  measured (D35). Revisit if a judged metric is ever asked to gate.
- **Production sampling.** The feedback loop exports chosen enquiries by hand (D13); stratified,
  PII-scrubbed sampling of tenant traffic needs consent and is described, not built.
- **Load and latency under traffic.** Latency is judged per enquiry on live runs (30 s ceiling,
  15 s at the 95th percentile) because the agent runs inside a synchronous request; throughput
  is out of scope for a replica.
- **Attachments on Confident AI.** A golden holds the attachment's file name only; the
  spreadsheet itself lives in git and its parsed text is visible on the extraction span.
