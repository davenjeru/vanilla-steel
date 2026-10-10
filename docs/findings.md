# Findings

Failures found by evals and tests, what caused them, and what was done.
Each finding is also annotated on its trace in Confident AI.

## F-001: Extractor inferred an Incoterm the customer never wrote

| | |
|---|---|
| Found | 2026-10-10, replay eval, dataset extraction-v1, `sha=8f051a7`, component extractor-v1 |
| Case | `de-001-blech-dezimalkomma-posen` |
| Metric | `FieldMatch[incoterm]`: expected `None`, got `'DDP'` |
| Layer | Model (prompt adherence). The recording shows Claude itself returned `DDP`; the normaliser passed it through. |
| Business risk | Business correctness. DAP and DDP put import duties and taxes on different parties, so a guessed Incoterm can make a quote wrong without anyone noticing. |

**Cause.** The email says "Lieferung frei Haus Posen". "Frei Haus" is a German
trade term, not an Incoterm. The prompt says to record an Incoterm only if one
is stated, but the model inferred DDP.

**Decision.** The label stays `null`: an unstated term must not be guessed.
Fix chosen (option A): a deterministic evidence check in
`steel/extraction/evidence.py`. An Incoterm is kept only if its code or
official name appears in the text the model read (email and attachment
tables); otherwise it becomes `null` with a warning. This no longer depends on
the model obeying the prompt. Not chosen for now (option B): a prompt v2 that
names "frei Haus" explicitly, which still trusts the model.

**Evidence.**
- Regression tests written first and seen failing:
  `test_incoterm_the_customer_never_wrote_is_dropped`,
  `test_dropped_incoterm_is_reported_as_a_warning` (`tests/unit/test_extractor.py`).
- Replay on the same recordings after the fix: 9 of 9 cases pass, component extractor-v2.
- The raw model answer (`DDP`) stays in `ExtractionResult.raw`, so the model's behaviour remains visible.

**Open.** One case cannot say how often the model does this. Cases with
"franco", "free delivered" or "frachtfrei" would measure it.
