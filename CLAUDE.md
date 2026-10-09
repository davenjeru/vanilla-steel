# Code standards

Readability is the top priority. A reviewer should understand any file in one read.

- Names describe business meaning: `match_stock_to_spec`, not `process`; `thickness_mm`, not `t`.
- Functions do one thing and stay under ~30 lines. Extract a named helper rather than nesting.
- Type hints on every function signature. Pydantic models for anything crossing a boundary.
- No clever one-liners, no metaprogramming, no premature abstraction. Prefer explicit over DRY when DRY hurts clarity.
- Comments explain *why* (business rules, steel domain logic), never *what*.
- Domain rules live in one place: grade equivalences in `domain/grades.py`, tolerances in `domain/tolerances.py`.
- Tests read as specifications: `test_rejects_cross_tenant_rfq_read`, arrange/act/assert, one behaviour per test.
- Keep dependencies minimal. Ask before adding a library.
- After each change, run `ruff check` and `ruff format`.