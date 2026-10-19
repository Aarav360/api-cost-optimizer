## Sealed test answers

`results/test_sealed.sqlite` holds the raw test-split answers and stays sealed until Stage 8 (PLAN_v4.md §5–6, Amendment A6).

- Never open, query, copy or print it, not even to check its schema or row count. Only `harness/collect.py` names its path; don't name it anywhere else.
- Before Stage 8, nothing reads, scores or inspects test answers, grades or gate verdicts, and no router or gate runs on test items. Fit, tune and select on train and dev only.
- If a task seems to need test data, stop and ask.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on Aarav360/api-cost-optimizer, managed through the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root, plus the plan's amendments. See `docs/agents/domain.md`.
