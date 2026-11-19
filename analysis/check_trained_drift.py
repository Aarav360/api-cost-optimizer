"""Check F's drift check, under A11 item 17's limits (PLAN_v4.md Amendment A11, items 17 and 19; A19, items 2 and 13; issue #83).

Run from the project root:  uv run python -m analysis.check_trained_drift

On hand-worked tables and a made-up dev side, and in the rehearsal with a fake embedder (`analysis/rehearsal_trained.py`): no
results/, no model snapshot, no Jev or Anthropic call, and nothing near the real store or a test item.

- **The limits come from the plan's own text,** not from this check's code: A11 item 17 says "above 1e-6" and A11 item 19
  records the drift limit, which `drift_limit()` reads (4.32e-05, 100 times the 4.32e-07 gap);
- **the rule, worked by hand** on a table of eight dev items: identical router scores differ by nothing and flip nothing; an
  item whose router score sits exactly on a threshold flips at that threshold, and only that one, under a drift of 1e-7 (a
  tie counts, so any drop flips it); a drift of exactly 1e-6 is not reported on, and 5e-6 is, with dev's quality and cost at the
  frozen thresholds before and after (worked from the matrix's costs and successes) and the run going on; a drift past the
  limit refuses, and one exactly at it does not;
- **the rehearsal**, the whole runner on a made-up world with a fake embedder: unchanged vectors give no difference and the run
  goes on; vectors moved slightly (above 1e-6, under the limit) give the dev quality and cost report in the record and the run
  goes on to the calls; vectors moved past the limit refuse (`DriftRefusal`, named in the record, its message shown) before any
  call on test: nothing but dev's requests reached the embedder, the calls step never ran and no router score of test's was
  stored. The flips the record holds are those of an independent recomputation (scores compared with `>=` by hand), and the
  real model is never loaded (`default_router` is made to raise).
"""

import dataclasses
import sys
from collections.abc import Callable
from dataclasses import dataclass
from unittest import mock

from analysis import rehearsal_trained as rt
from analysis.check_final_run import rehearsed
from analysis.drift import DriftRefusal
from analysis.final_run import STEPS, Outcome
from analysis.measure_trained_router import drift_limit
from analysis.operating_eval import FrozenPoint, frozen_points
from analysis.plan import PLAN, amendment_item
from analysis.router_table import ItemRouterScores, RouterScoreTable
from analysis.trained_drift import REPORT_ABOVE, DevSide, drift_report
from routers import trained
from routers.ladder import SCORED
from routers.trained import TrainedRouter

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def close(a: float | None, b: float, tolerance: float = 1e-12) -> bool:
    return a is not None and abs(a - b) <= tolerance


def refuses(call: Callable[[], object]) -> bool:
    try:
        call()
    except DriftRefusal:
        return True
    return False


# ---- the limits are the plan's -----------------------------------------------------------------------------------


def limits() -> None:
    print("-- the limits are A11's own --")
    item17 = amendment_item(PLAN.read_text(encoding="utf-8"), "A11", 17)
    check(
        "Above 1e-6" in item17 and REPORT_ABOVE == 1e-6, "A11 item 17 says 'Above 1e-6', and the check's figure is 1e-6"
    )
    check(drift_limit() == 4.32e-05 and "`4.32e-05`, 100 times the overall gap" in
          amendment_item(PLAN.read_text(encoding="utf-8"), "A11", 19),
          "the drift limit read from A11 item 19 is 4.32e-05, 100 times the overall gap")  # fmt: skip


# ---- the rule, by hand -------------------------------------------------------------------------------------------

LIMIT = 4.32e-05
THRESHOLD = 0.9
# Eight dev items (`rehearsal_trained.DEV_ITEMS`: two per task). Every item scores haiku .95 (so haiku at a threshold of .9)
# and sonnet and opus_low .5, except ifeval:made-dev-1, whose haiku score is exactly .9 in the old table (a tie, which counts).
UNMOVED = {item: {"haiku": 0.95, "sonnet": 0.5, "opus_low": 0.5} for item in rt.DEV_ITEMS}
OLD = UNMOVED | {"ifeval:made-dev-1": {"haiku": THRESHOLD, "sonnet": 0.5, "opus_low": 0.5}}
POINTS = (
    FrozenPoint("trained", "realistic", 1.0, None, None, True),
    FrozenPoint("trained", "realistic", 0.99, THRESHOLD, None, False),
    FrozenPoint("trained", "uniform", 0.97, THRESHOLD, None, False),
    FrozenPoint("trained", "uniform", 0.99, 0.5, None, False),  # a second threshold, where nothing changes
)


def table(scores: dict[str, dict[str, float]]) -> RouterScoreTable:
    return RouterScoreTable.build([ItemRouterScores(i, s, 0.0) for i, s in scores.items()], list(scores))


def dev_with(old: dict[str, dict[str, float]]) -> DevSide:
    requests = rt.dev_requests()
    return DevSide(requests, dict.fromkeys(requests, 0), table(old), rt.dev_matrix)


def moved(by: float) -> RouterScoreTable:
    """The old table with ifeval:made-dev-1's haiku score lowered by `by`."""
    scores = {item: dict(s) for item, s in OLD.items()}
    scores["ifeval:made-dev-1"]["haiku"] -= by
    return table(scores)


HALF = {i: {"haiku": 0.5, "sonnet": 0.25, "opus_low": 0.25} for i in rt.DEV_ITEMS}


def lowered_by(amount: float) -> RouterScoreTable:
    scores = {i: dict(s) for i, s in HALF.items()}
    scores["ifeval:made-dev-1"]["haiku"] -= amount
    return table(scores)


def the_rule() -> None:
    print("\n-- the rule, worked by hand --")
    dev = dev_with(OLD)
    same = drift_report(dev, table(OLD), POINTS, LIMIT)
    check(same.largest == 0.0 and all(f.items == () for f in same.flips) and same.outcomes is None,
          "identical router scores: no difference, no flip, no dev report")  # fmt: skip
    check([f.threshold for f in same.flips] == [THRESHOLD, 0.5]
          and [f.points for f in same.flips] == [("realistic 99%", "uniform 97%"), ("uniform 99%",)],
          "a flip is reported for each distinct frozen threshold, highest first, with the points that sit on it")  # fmt: skip

    tiny = drift_report(dev, moved(1e-7), POINTS, LIMIT)
    check(close(tiny.largest, 1e-7, 1e-15) and [len(f.items) for f in tiny.flips] == [1, 0]
          and tiny.flips[0].items == ("ifeval:made-dev-1",),
          "a score exactly on the threshold flips under a drift of 1e-7: one decision at .9, none at .5")  # fmt: skip
    check(tiny.outcomes is None, "at 1e-7 (not above 1e-6) there is no dev quality and cost report")
    # 2**-20 (9.5e-7) is under 1e-6 and 2**-19 (1.9e-6) over it, and both are exact in binary, so the edge is worked exactly.
    under = drift_report(dev_with(HALF), lowered_by(2**-20), POINTS, LIMIT)
    over = drift_report(dev_with(HALF), lowered_by(2**-19), POINTS, LIMIT)
    check(under.largest == 2**-20 and under.outcomes is None and over.largest == 2**-19 and over.outcomes is not None,
          "a drift of 9.5e-7 makes no dev report, and one of 1.9e-6 does")  # fmt: skip

    slight = drift_report(dev, moved(5e-6), POINTS, LIMIT)
    check(close(slight.largest, 5e-6, 1e-15) and slight.over_report and slight.outcomes is not None,
          "5e-6, above 1e-6 and under the limit: the dev report is made, and the run goes on")  # fmt: skip
    assert slight.outcomes is not None
    # By hand (matrix: haiku $.001 and opus $.010 an item; every task is two items; haiku is right on one of each task's two,
    # opus on both of ifeval's, both of banking77's, one of mmlu_pro's and both of mbpp+'s). Old decisions at .9: all haiku, so
    # every task is at .5 and $1.00 per 1k. New: ifeval's second item goes to opus, so ifeval is at 1.0 and $5.50 per 1k.
    old, new = slight.outcomes["realistic"]["99%"]["old"], slight.outcomes["realistic"]["99%"]["new"]
    check(close(old["quality"], 0.5) and close(old["cost_per_1k"], 1.0),
          "dev at .9 with the old scores: every task at 50% and $1.00 per 1k, under any mix")  # fmt: skip
    check(close(new["quality"], 0.1 * 1.0 + 0.9 * 0.5) and close(new["cost_per_1k"], 0.1 * 5.5 + 0.9 * 1.0),
          "with the new scores, realistic (IFEval .10): quality .55, cost $1.45 per 1k")  # fmt: skip
    new_uniform = slight.outcomes["uniform"]["97%"]["new"]
    check(close(new_uniform["quality"], 0.25 * 1.0 + 0.75 * 0.5) and close(new_uniform["cost_per_1k"], 0.25 * 5.5 + 0.75),
          "uniform (.25 each): quality .625, cost $2.125 per 1k")  # fmt: skip
    fell = slight.outcomes["realistic"]["100%"]["new"]
    check(close(fell["quality"], 0.975) and close(fell["cost_per_1k"], 10.0) and fell == slight.outcomes["realistic"]["100%"]["old"],
          "a point that fell back is A in both: 97.5% at $10.00, so the drift can't move it")  # fmt: skip
    check([len(f.items) for f in slight.flips] == [1, 0], "the report's flips are the same one decision")

    check(refuses(lambda: drift_report(dev, moved(5e-5), POINTS, LIMIT)), "5e-5, past the drift limit: it refuses")
    check(not refuses(lambda: drift_report(dev, moved(4e-5), POINTS, LIMIT)), "4e-5, under the limit: it goes on")
    # A difference exactly at the limit isn't past it: 0.5 and 0.25 differ by exactly 0.25.
    far = lowered_by(0.25)
    check(not refuses(lambda: drift_report(dev_with(HALF), far, POINTS, 0.25))
          and refuses(lambda: drift_report(dev_with(HALF), far, POINTS, 0.2499)),
          "a difference exactly at the limit goes on, and one past it refuses")  # fmt: skip
    other = table({i: s for i, s in list(OLD.items())[:-1]})
    try:
        drift_report(dev, other, POINTS, LIMIT)
        refused_outright = False
    except ValueError:
        refused_outright = True
    check(refused_outright, "tables over other dev items are refused outright (a ValueError), never read as drift")


# ---- the rehearsal -----------------------------------------------------------------------------------------------

STEPS_UNDER_TEST = tuple(s for s in STEPS if s.name in ("unlock", "drift_trained_router", "calls_rules_trained"))


def independent_flips(shift: float) -> dict[float, list[str]]:
    """The dev items whose decision changes at each of F's frozen thresholds, found without the check's own functions:
    each item's router scores from the router directly, compared by hand with `>=` in the ladder's order."""
    settings, weights = trained.read_settings(), trained.read_fold_weights()
    old = TrainedRouter(settings, weights, backend=rt.SeededBackend(0.0))
    new = TrainedRouter(settings, weights, backend=rt.SeededBackend(shift))
    side = rt.dev_side()
    found: dict[float, list[str]] = {}
    for point in frozen_points("trained"):
        if point.threshold is None:
            continue
        flipped = []
        for item, request in side.requests.items():
            before, after = old.score_with(side.fold_of[item], request), new.score_with(side.fold_of[item], request)

            def decision(scores: dict[str, float], t: float = point.threshold) -> str:
                return next((c for c in SCORED if scores[c] >= t), "opus")

            if decision(before) != decision(after):
                flipped.append(item)
        found[point.threshold] = flipped
    return found


@dataclass
class Ran:
    outcome: Outcome
    record: dict
    log: str
    stored: bool  # whether test's router scores were stored
    embedded: list[str]  # the texts that reached the fake embedder, in order


def go(shift: float) -> Ran:
    """The unlock, F's drift check and the calls on a made-up world whose embedder has moved by `shift`."""
    with rehearsed() as r:
        stand_in = rt.stand_in(shift)
        r.world = dataclasses.replace(r.world, backends=dataclasses.replace(r.world.backends, trained=stand_in))
        with mock.patch.object(trained, "default_router", side_effect=AssertionError("the real model was loaded")):
            outcome, _, _ = r.go(STEPS_UNDER_TEST)
        return Ran(outcome, r.record, r.log, (r.world.results / "test_router_scores.json").exists(),
                   stand_in.embedder.texts())  # fmt: skip


def rehearsal() -> None:
    print("\n-- the rehearsal, with a fake embedder --")
    dev_texts = set(rt.DEV_PROMPTS.values())

    ran = go(0.0)
    drift = next(s for s in ran.record["gos"][0]["steps"] if s["name"] == "drift_trained_router")["detail"]
    check(ran.outcome == Outcome() and drift["largest_difference"] == 0.0 and "dev_outcomes" not in drift
          and all(f["count"] == 0 for f in drift["flips"]) and ran.stored,
          "unchanged vectors: no difference, no flip, no dev report, and the run goes on to the calls")  # fmt: skip

    shift = 2e-4
    ran = go(shift)
    drift = next(s for s in ran.record["gos"][0]["steps"] if s["name"] == "drift_trained_router")["detail"]
    check(ran.outcome == Outcome() and REPORT_ABOVE < drift["largest_difference"] < drift["drift_limit"] and ran.stored,
          f"vectors moved slightly ({drift['largest_difference']:.2e}, above 1e-6 and under the limit): the run goes on")  # fmt: skip
    check(set(drift["dev_outcomes"]) == {"realistic", "uniform"} and set(drift["dev_outcomes"]["realistic"]) == {"100%", "99%", "97%"}
          and all(set(row) == {"old", "new"} for by in drift["dev_outcomes"].values() for row in by.values()),
          "the record holds dev's quality and cost at each of F's six points, old scores and new")  # fmt: skip
    wanted = independent_flips(shift)
    held = {f["threshold"]: f["items"] for f in drift["flips"]}
    check(held == wanted, f"the flips at each threshold are an independent recomputation's {held}")
    check("going on" in ran.log and "drift_trained_router: done" in ran.log,
          "the log says the report was made and the run went on")  # fmt: skip

    ran = go(1e-2)
    stopped = ran.record["gos"][0]["stopped"]
    check(ran.outcome == Outcome("drift_trained_router", "DriftRefusal", None)
          and stopped["step"] == "drift_trained_router" and stopped["check"] == "drift_trained_router",
          f"vectors moved past the limit: the run stops at the drift check with a DriftRefusal {ran.outcome}")  # fmt: skip
    check("past the drift limit" in ran.log and "new amendment" in ran.log,
          "the refusal's message is shown, and names the way on (a new amendment)")  # fmt: skip
    check(set(ran.embedded) == dev_texts and len(ran.embedded) == len(dev_texts),
          f"before any call on test, only dev's requests reached the embedder, once each ({len(ran.embedded)} embeddings)")  # fmt: skip
    check(not ran.stored and "calls_rules_trained: starts" not in ran.log,
          "the calls step never ran, and no router score of test's was stored")  # fmt: skip


def main() -> int:
    limits()
    the_rule()
    rehearsal()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
