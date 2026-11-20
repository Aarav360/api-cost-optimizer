## Sealed test answers

`results/test_sealed.sqlite` holds the raw test-split answers. It stayed sealed until Stage 8 and the go has run (PLAN_v4.md §5–6, Amendments A6 and A19), so test is now read, but only through the unlock's handle.

- Never open, query, copy, hash or print the store, not even to check its schema or row count. Only `harness/collect.py` names its path; don't name it anywhere else.
- Test is read only through the handle that `analysis/unlock.py`'s unlock returns, by the final evaluation's modules (the collectors, the grading, the table builders and the live replay). Only the unlock reads the store's path (`analysis/check_read_path.py`), and the go's entry point only passes it in.
- The unlock runs on the real store only at the researcher's go, and a resume needs a new go. No check, rehearsal, script or agent passes it the real path; build and rehearse on made-up worlds and stores (`analysis/made_up_world.py`).
- Test stays read-only data and is never used to select, fit or tune anything: no threshold, cutoff, operating point or policy changes because of what it shows (A19 items 4 and 7). Fit, tune and select on train and dev only.
- If a task seems to need test data some other way, stop and ask.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on Aarav360/api-cost-optimizer, managed through the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root, plus the plan's amendments. See `docs/agents/domain.md`.
