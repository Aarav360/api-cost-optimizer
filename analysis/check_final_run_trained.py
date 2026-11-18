"""Check the rehearsal with the real embedding model, on made-up requests (PLAN_v4.md Amendment A11, items 17 and 19; A19, item
13; issue #83).

Run from the project root:  uv run python -m analysis.check_final_run_trained

Needs the pinned model snapshot under `.cache/` (not committed), so the hook runs it and CI doesn't (`HOOK_ONLY` in
`analysis/check_guardrails.py`). It is the rehearsal's F path (`analysis/check_trained_drift.py`, `analysis/check_final_routers.py`)
with no fake embedder: the world names none, so F's drift check re-embeds the made-up dev requests with the real model and the
calls step scores the rehearsal's eight made-up test requests through `routers.trained.score`, the committed files and the verified
snapshot. The made-up dev table is scored by the same model in float64, as A11 item 19's drift measurement did, so the check
sees a real, tiny difference. No results/, no Jev or Anthropic call, no benchmark item, and nothing near the real store.

- the run goes through the unlock, F's drift check and the calls on test, with `default_router()` loaded once (the real model);
- the drift check's largest difference is above 0 (float32 against float64 is a real difference) and within the drift limit, so
  the run goes on, and it is under 1e-6 here, so no dev quality and cost report is made (the made-up requests are short);
- each stored router score of the eight test requests is within the drift limit of an independent float64 route (a router built
  on the same model upcast), and each fold is the plan's hash, worked independently;
- the truncation and emptied counts are zero (the requests are short), and the default router's own counters agree;
- F's drift check on dev itself, the one place the real dev side runs: the 744 dev requests embedded again against the table the
  dev cache gives, the largest difference within the limit, and that table at the six frozen points giving table v1's committed
  quality and cost for F (so the cache is the table dev had).
"""

import dataclasses
import json
import sys

from analysis import pareto
from analysis import rehearsal_trained as rt
from analysis.backends import TrainedStandIn
from analysis.check_final_routers import TEST_REQUESTS, independent_fold
from analysis.check_final_run import rehearsed
from analysis.final_routers import SplitScores, read_scores
from analysis.final_run import STEPS, Outcome
from analysis.measure_trained_router import drift_limit
from analysis.operating_eval import frozen_points
from analysis.trained_drift import REPORT_ABOVE, dev_outcomes, drift_report, real_dev_side, rescore
from routers import trained
from routers.ladder import SCORED
from routers.trained import TrainedRouter

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def main() -> int:
    print("-- the rehearsal with the real embedding model --")
    limit = drift_limit()
    try:
        real = trained.default_router()
    except trained.SnapshotError as error:
        print(f"FAIL  the pinned snapshot isn't available: {error}")
        return 1
    float64 = real.backend.upcast()  # type: ignore[attr-defined]
    steps = tuple(s for s in STEPS if s.name in ("unlock", "drift_trained_router", "calls_rules_trained"))
    with rehearsed() as r:
        stand_in = TrainedStandIn(embedder=None, dev=lambda: rt.dev_side(table_backend=float64))
        r.world = dataclasses.replace(r.world, backends=dataclasses.replace(r.world.backends, trained=stand_in))
        outcome, _, _ = r.go(steps)
        check(
            outcome == Outcome(), f"the unlock, F's drift check and the calls on test run with the real model {outcome}"
        )
        drift = next(s for s in r.record["gos"][0]["steps"] if s["name"] == "drift_trained_router")["detail"]
        largest = drift["largest_difference"]
        check(0 < largest <= REPORT_ABOVE and "dev_outcomes" not in drift,
              f"float32 against float64 on the made-up dev requests: a real difference ({largest:.2e}), within 1e-6, so no dev report")  # fmt: skip
        check(largest <= limit, f"and within the drift limit {limit:.3g}")
        scores: SplitScores = read_scores(r.world.results / "test_router_scores.json", TEST_REQUESTS)
        by_hand = TrainedRouter(trained.read_settings(), trained.read_fold_weights(), backend=float64)
        worst = max(abs(scores.trained[i][c] - by_hand.score(q)[c]) for i, q in TEST_REQUESTS.items() for c in SCORED)
        check(
            worst <= limit,
            f"F's stored scores of the eight made-up test requests are within the limit of a float64 route ({worst:.2e})",
        )
        check(all(scores.folds[i] == independent_fold(q) for i, q in TEST_REQUESTS.items()),
              "each request's fold is the plan's hash, worked independently")  # fmt: skip
        check(scores.truncated == 0 and scores.emptied == 0 and (real.counts.truncated, real.counts.emptied) == (0, 0),
              "no request was truncated or emptied, and the default router's own counters agree")  # fmt: skip
    real_dev_side_checks(real, limit)
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


def real_dev_side_checks(real: TrainedRouter, limit: float) -> None:
    """F's drift check on the real dev side: dev's 744 requests embedded again with the model, against dev's table from the
    embedding cache. Dev's own requests, never a test item."""
    print("\n-- F's drift check on dev itself (744 requests, about 30 s) --")
    side = real_dev_side()
    router = TrainedRouter(trained.read_settings(), trained.read_fold_weights(), backend=real.backend)
    rescored = rescore(router, side)
    points = frozen_points("trained")
    report = drift_report(side, rescored, points, limit)
    flipped = sum(len(f.items) for f in report.flips)
    check(len(side.requests) == 744 and report.largest <= limit,
          f"744 dev requests embedded again: largest difference {report.largest:.2e}, within the limit {limit:.3g}; "
          f"{flipped} decisions flip at F's three thresholds")  # fmt: skip
    # The table dev had, evaluated at the frozen points, is the committed table v1's own quality and cost for F.
    committed = json.loads(pareto.FIGURE_DATA.read_text(encoding="utf-8"))["panels"]
    old = dev_outcomes(side.matrix(), side.table, points)
    same = all(
        old[mix][target]["quality"]
        == committed[mix]["routers"]["trained"]["operating_points"][target]["quality"]["value"]
        and old[mix][target]["cost_per_1k"]
        == committed[mix]["routers"]["trained"]["operating_points"][target]["cost_per_1k"]["value"]
        for mix in old
        for target in old[mix]
    )
    check(
        same,
        "dev's table from the cache, evaluated at the six frozen points, gives table v1's committed quality and cost",
    )


if __name__ == "__main__":
    sys.exit(main())
