# Coding standards

The judgement calls a reviewer applies to every diff. The mechanical rules (formatting, line
length, imports, types) belong to ruff and mypy, configured in `pyproject.toml` and run by
`.githooks/pre-commit` and CI.

## Every module

- **Cite the plan.** A module's docstring names the PLAN_v4.md section or amendment behind it. A
  runnable script also gives its command: `Run from the project root:  uv run python -m <package>.<module>`.
- **Comments give the reason**: the decision, the gotcha or the amendment behind a line, in one or
  two lines. The harness's comment density is the bar.
- **Glossary vocabulary.** Names, docstrings and messages use GLOSSARY.md's terms (policy, trace,
  config ladder, reference, task, task mix), never a word it lists under _Avoid_.
- **Plain data.** Rows are dicts, records are frozen dataclasses, arrays are numpy. Pandas and
  notebooks stay out (Amendment A8, item 11).
- **Hand-wrapped statements** that read better than ruff's layout end in `# fmt: skip`.

## Checks

- **Checks are scripts**: `check_*.py`, run with `uv run python -m`, shaped like
  `harness/check_collect.py`: a `check(ok, what)` that prints PASS or FAIL, `--` section headers,
  a closing "all checks passed" or failure count, and exit 1 on any failure.
- **Expected values come from an independent source**: a hand-worked example, an amendment's
  table, a stored value. An assertion that recomputes the answer the way the code does proves
  nothing.
- **A guard proves it can fail.** A search must find a known positive, as `check_sealed` finds
  `harness/collect.py`, and a refusal check feeds the bad case.
- **Refusals are whole.** Invalid input raises a specific error (`TraceError`, `MatrixError`)
  before anything is computed, so nothing is evaluated partially (Amendment A8, item 4). Each
  refusal has its own check.

## Numbers in documents

A number quoted in PLAN_v4.md, the README or a figure caption comes from a committed output
(`figures/*.json`) or a check script's printout, and a check compares the document's copy with
it. A hand-copied number drifts: A8 item 1's table carried a rounding slip until `check_simulator`
compared it.

## Sealed data

AGENTS.md's sealed-store rule binds review as well: a diff that reads test answers, grades or gate
verdicts before Stage 8 is a hard violation. `harness/check_sealed.py` catches code that names the
store; reading it some other way is the reviewer's to catch.
