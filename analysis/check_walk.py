"""Check threshold selection: the walk and the operating-points file (PLAN_v4.md §8; Amendment A9, items 9, 10, 13).

Run from the project root:  uv run python -m analysis.check_walk

On synthetic matrices and router-score tables only, with every answer worked out by hand. No router runs on dev
here, and never on a test item.

- an item's breakpoints are its router scores strictly higher than every cheaper config's: a lower or equal one
  isn't one;
- the steps: every distinct breakpoint when there are 24 or fewer, otherwise the 0%, 5%, ..., 95% quantiles of the
  pooled breakpoints (lower empirical quantile, with duplicates removed afterwards); they come from the table alone,
  so shuffling the outcomes or the table's order leaves them unchanged;
- the walk starts at always-frontier, tries the steps from the highest threshold down, and stops at the first that
  doesn't clear even where a later step would; a step's margin is a hand-worked paired standard error, and the
  frontier's is 0;
- the operating point is the cheapest certified step, which beats the last where per-item cost isn't monotone on the
  ladder, with ties to the earlier step; where the first step fails, or no certified step costs less than always-
  frontier, it falls back to A, whose captured fraction is 0;
- the quality targets are met or missed by the walk exactly at the boundary, whatever rounding the sums carry;
- the operating-points file round-trips, writing one router's section leaves the others' unchanged, and a section's
  hash is stable and matches an independent sha256.
"""

import hashlib
import inspect
import json
import math
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from analysis.bootstrap import Bootstrap, resample
from analysis.check_simulator import synthetic
from analysis.matrix import Matrix
from analysis.operating_points import (
    SELECTION_METHOD,
    OperatingPointsError,
    build_section,
    read_section,
    section_hash,
    write_section,
)
from analysis.references import TARGETS
from analysis.router_table import ItemRouterScores, RouterScoreTable
from analysis.simulator import TraceError, always, simulate
from analysis.walk import Walk, WalkError, item_breakpoints, step_thresholds, walk

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(error: type[Exception], call: Callable[[], object]) -> bool:
    try:
        call()
    except error:
        return True
    return False


def close(a: float | None, b: float, tolerance: float = 1e-12) -> bool:
    return a is not None and math.isclose(a, b, rel_tol=0, abs_tol=tolerance)


def table_of(router_scores: dict[str, dict[str, float]], overhead: float = 0.0) -> RouterScoreTable:
    return RouterScoreTable.build(
        [ItemRouterScores(i, s, overhead) for i, s in router_scores.items()], list(router_scores)
    )


def haiku_only(haiku: dict[str, float]) -> dict[str, dict[str, float]]:
    """Items scored on haiku alone: sonnet and opus_low at 0, below it, so each item has one breakpoint."""
    return {i: {"haiku": s, "sonnet": 0.0, "opus_low": 0.0} for i, s in haiku.items()}


def thresholds_of(result: Walk) -> list[float | None]:
    return [step.threshold for step in result.steps]


MIX = {"t1": 1.0}
# Four resamples of four items, each its own source group, drawn by hand (rows sum to 4): under MIX each item weighs its
# count / 4. Columns 1 and 3 always match, so a policy that differs from A only on i1 and i3, in opposite directions,
# has a paired difference of 0 in every resample.
DRAWS = Bootstrap(
    ("i0", "i1", "i2", "i3"), ("t1",) * 4, np.array([[1, 1, 1, 1], [2, 1, 0, 1], [0, 2, 0, 2], [2, 0, 2, 0]])
)

# M1. Opus (A) is right on i0, i1, i2 and wrong on i3, so A is at 75% and the 100% target is 75% (99%: 74.25%,
# 97%: 72.75%). Haiku is right on i0, i2, i3 and wrong on i1. Haiku costs $1 and opus $10 per 1k on every item,
# before the 1/4 weight: all four on opus cost $10 per 1k.
M1 = synthetic({
    "i0": ("t1", {"haiku": (0.001, 1), "opus": (0.010, 1)}),
    "i1": ("t1", {"haiku": (0.001, 0), "opus": (0.010, 1)}),
    "i2": ("t1", {"haiku": (0.001, 1), "opus": (0.010, 1)}),
    "i3": ("t1", {"haiku": (0.001, 1), "opus": (0.010, 0)}),
})  # fmt: skip
# Haiku's router scores, the only breakpoints: i0 .9, i1 .7, i3 .5, i2 .1. So the steps are .9, .7, .5, .1.
M1_TABLE = table_of(haiku_only({"i0": 0.9, "i1": 0.7, "i2": 0.1, "i3": 0.5}))
# Same, but haiku's wrong answer i1 scores highest, so the first step below A already loses an item.
M1_FIRST_FAILS = table_of(haiku_only({"i0": 0.7, "i1": 0.9, "i2": 0.1, "i3": 0.5}))

# M2, where per-item cost isn't monotone on the ladder. Opus is right on every item, so A is at 100% for $10 per 1k.
# i1 costs $8 on haiku, $6 on sonnet, $4 on opus_low and $10 on opus. Every config is right on every item but i2's
# haiku. i1 scores haiku .1, sonnet .4, opus_low .7, so its breakpoints are .1, .4 and .7; i0 scores haiku .9 and i2
# and i3 .05. The steps are .9, .7, .4, .1, .05.
M2_I1: dict[str, tuple[float, float]] = {
    "haiku": (0.008, 1),
    "sonnet": (0.006, 1),
    "opus_low": (0.004, 1),
    "opus": (0.010, 1),
}
M2_TABLE = table_of({
    "i0": {"haiku": 0.9, "sonnet": 0.0, "opus_low": 0.0},
    "i1": {"haiku": 0.1, "sonnet": 0.4, "opus_low": 0.7},
    "i2": {"haiku": 0.05, "sonnet": 0.0, "opus_low": 0.0},
    "i3": {"haiku": 0.05, "sonnet": 0.0, "opus_low": 0.0},
})  # fmt: skip


def m2(i1: dict[str, tuple[float, float]]) -> Matrix:
    return synthetic({
        "i0": ("t1", {"haiku": (0.001, 1), "opus": (0.010, 1)}),
        "i1": ("t1", i1),
        "i2": ("t1", {"haiku": (0.001, 0), "opus": (0.010, 1)}),
        "i3": ("t1", {"haiku": (0.001, 1), "opus": (0.010, 1)}),
    })  # fmt: skip


# M3, a mix whose weights make the targets differ. Three t1 items at .99 / 3 = .33 each and one t2 item at .01. Opus is
# right on all four; haiku on the three t1 items only. j0 scores haiku .9 and the others .2, so the steps are .9 and .2,
# and both lose j0: quality .99 = 99% of A, below the 100% target by .01 and at the 99% target exactly. j0 is always
# drawn, once, so its weight is .01 in every resample and the paired difference is -.01 in all: a margin of 0.
M3 = synthetic({
    **{f"i{k}": ("t1", {"haiku": (0.001, 1), "opus": (0.010, 1)}) for k in range(3)},
    "j0": ("t2", {"haiku": (0.001, 0), "opus": (0.010, 1)}),
})  # fmt: skip
M3_MIX = {"t1": 0.99, "t2": 0.01}
M3_DRAWS = Bootstrap(("i0", "i1", "i2", "j0"), ("t1", "t1", "t1", "t2"),
                     np.array([[3, 0, 0, 1], [1, 1, 1, 1], [0, 2, 1, 1], [1, 0, 2, 1]]))  # fmt: skip
M3_TABLE = table_of(haiku_only({"i0": 0.2, "i1": 0.2, "i2": 0.2, "j0": 0.9}))

# M4, a step that is level with A up to rounding. Opus scores .1 on i1 and .2 on i3; haiku .3 and 0: one gains what the
# other loses, so haiku on both is exactly A's quality, and in every resample too (columns 1 and 3 match). Summed in
# floating point the two differ by a few 1e-18, so the step's margin is 5.7e-18 and its quality less the target is 0.
M4 = synthetic({
    "i0": ("t1", {"haiku": (0.001, 1), "opus": (0.010, 1)}),
    "i1": ("t1", {"haiku": (0.001, 0.3), "opus": (0.010, 0.1)}),
    "i2": ("t1", {"haiku": (0.001, 1), "opus": (0.010, 1)}),
    "i3": ("t1", {"haiku": (0.001, 0.0), "opus": (0.010, 0.2)}),
})  # fmt: skip
M4_TABLE = table_of(haiku_only({"i0": 0.05, "i1": 0.9, "i2": 0.05, "i3": 0.9}))


def main() -> int:
    print("-- breakpoints --")
    # haiku .5, sonnet .4, opus_low .7: haiku's is a breakpoint (nothing is cheaper); sonnet's is below haiku's and
    # can never win; opus_low's is higher than both.
    check(item_breakpoints({"haiku": 0.5, "sonnet": 0.4, "opus_low": 0.7}) == [0.5, 0.7],
          "a router score below a cheaper config's isn't a breakpoint")  # fmt: skip
    check(item_breakpoints({"haiku": 0.3, "sonnet": 0.3, "opus_low": 0.3}) == [0.3],
          "a router score equal to a cheaper config's isn't one either: only haiku's is")  # fmt: skip
    check(item_breakpoints({"haiku": 0.1, "sonnet": 0.4, "opus_low": 0.7}) == [0.1, 0.4, 0.7],
          "router scores that rise up the ladder are all breakpoints")  # fmt: skip
    check(item_breakpoints({"haiku": 0.9, "sonnet": 0.5, "opus_low": 0.95}) == [0.9, 0.95],
          "opus_low must beat the higher of the cheaper two, not only the nearest")  # fmt: skip
    check(item_breakpoints({"haiku": 0.5, "sonnet": 0.6, "opus_low": 0.55}) == [0.5, 0.6],
          "opus_low below sonnet is no breakpoint though above haiku")  # fmt: skip

    print("\n-- the steps --")
    # 24 distinct breakpoints, .01 to .24: each is a step, in ascending order.
    each = step_thresholds(table_of(haiku_only({f"i{k}": k / 100 for k in range(1, 25)})))
    check(each == tuple(k / 100 for k in range(1, 25)), f"24 distinct breakpoints are 24 steps: {each}")
    # 25 distinct, .01 to .25 (n = 25): the k/20 quantile is the ceil(25k/20)-th smallest, in 1-based positions
    # 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24 for k = 0..19.
    twenty = step_thresholds(table_of(haiku_only({f"i{k}": k / 100 for k in range(1, 26)})))
    positions = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24]
    check(twenty == tuple(p / 100 for p in positions), f"25 distinct breakpoints give the 20 quantiles: {twenty}")
    # .01 to .24 once each and .5 on 100 items: n = 124. Positions ceil(124k/20) are 1 (k = 0), 7, 13, 19 and then 25
    # or more, all of them .5: five steps once duplicates are removed. Counting each distinct breakpoint once would
    # not give these.
    heavy = haiku_only({**{f"i{k}": k / 100 for k in range(1, 25)}, **{f"h{k}": 0.5 for k in range(100)}})
    check(step_thresholds(table_of(heavy)) == (0.01, 0.07, 0.13, 0.19, 0.5),
          "pooled with their multiplicity, quantiles that land on one breakpoint make one step")  # fmt: skip
    check(len(twenty) <= 20 and twenty[0] == 0.01, "at most 20 steps, the lowest reaching the bottom of the range")

    # Steps read the table alone, before any outcome is looked up.
    check(list(inspect.signature(step_thresholds).parameters) == ["table"], "the steps take the table and nothing else")
    reversed_success = {key: type(cell)(cell.cost, 1 - cell.success) for key, cell in M1.cells.items()}
    flipped = Matrix(M1.items, reversed_success)
    shuffled_table = RouterScoreTable(tuple(reversed(M1_TABLE.items)))
    check(step_thresholds(M1_TABLE) == (0.1, 0.5, 0.7, 0.9) == step_thresholds(shuffled_table),
          "M1's steps are .1, .5, .7, .9, whatever order the table lists its items in")  # fmt: skip
    one = walk(M1, M1_TABLE, DRAWS, MIX, 1.0)
    other = walk(flipped, M1_TABLE, DRAWS, MIX, 1.0)
    tried = (None, 0.9, 0.7, 0.5, 0.1)
    check(thresholds_of(one) == list(tried[: len(one.steps)]) and thresholds_of(other) == list(tried[: len(other.steps)]),
          "shuffling outcomes (here, flipping every success) changes where the walk stops, not the order of its steps")  # fmt: skip

    print("\n-- the walk --")
    # M1 at 100%: A is 75% for $10 per 1k, margin 0. At .9 only i0 goes to haiku (right): 75%, $7.75 (haiku $1 + three
    # on opus $30, x 1/4 x 1000 = 7.75), identical to A on every item so its margin is 0 and it clears exactly at the
    # target. At .7 i1 goes to haiku too and is wrong: 50% for $5.50, its difference from A is -c1/4 = -.25, -.25, -.5, 0
    # over the resamples (c1 = 1, 1, 2, 0), mean -.25, deviations 0, 0, -.25, .25: sd = sqrt(.125 / 3) = sqrt(1/24).
    result = walk(M1, M1_TABLE, DRAWS, MIX, 1.0)
    check(
        thresholds_of(result) == [None, 0.9, 0.7],
        f"the walk starts at A, tries .9 and .7 and stops: {thresholds_of(result)}",
    )
    check([close(s.quality, q) for s, q in zip(result.steps, (0.75, 0.75, 0.5))] == [True] * 3,
          "A, .9 and .7 are at 75%, 75% and 50%")  # fmt: skip
    check([close(s.cost_per_1k, c) for s, c in zip(result.steps, (10.0, 7.75, 5.5))] == [True] * 3,
          "and cost $10, $7.75 and $5.50 per 1k")  # fmt: skip
    check(result.steps[0].margin == 0 and result.steps[1].margin == 0 and close(result.steps[2].margin, math.sqrt(1 / 24)),
          f"margins 0, 0 and the hand-worked paired standard error sqrt(1/24) = {math.sqrt(1 / 24):.6f}")  # fmt: skip
    check(close(result.target, 0.75) and [s.clears for s in result.steps] == [True, True, False],
          "the 100% target is 75%: A and .9 clear it (exactly), .7 is 25 points short")  # fmt: skip
    check([s.threshold for s in result.certified] == [None, 0.9], "certified: A and .9")
    # Steps .5 and .1 would clear: i0, i1, i3 (and i2) on haiku is right on i0, i2, i3, wrong on i1, 75% again, and
    # its difference from A is (c3 - c1) / 4 = 0 in every resample. The walk never looked.
    traces = M1_TABLE.traces(0.5)
    policy, frontier = simulate(M1, traces), simulate(M1, always(M1, "opus"))
    check(close(DRAWS.safety_margin(policy, frontier, MIX), 0) and close(sum(o.success for o in policy) / 4, 0.75),
          "step .5, which the walk stopped before, would have cleared at 75% with margin 0")  # fmt: skip
    check(0.5 not in thresholds_of(result) and 0.1 not in thresholds_of(result),
          "yet the walk stops at the first failing step, and neither later step was tried")  # fmt: skip
    check(result.operating.threshold == 0.9 and not result.fell_back and close(result.operating.cost_per_1k, 7.75),
          "the operating point is .9, the cheapest certified step ($7.75 against A's $10)")  # fmt: skip
    for share, target in ((0.99, 0.7425), (0.97, 0.7275)):
        at = walk(M1, M1_TABLE, DRAWS, MIX, share)
        check(close(at.target, target) and thresholds_of(at) == [None, 0.9, 0.7] and at.operating.threshold == 0.9,
              f"at {share:.0%} the target is {target:.2%}, and the walk again stops after .7")  # fmt: skip
    check(
        refused(WalkError, lambda: walk(M1, M1_TABLE, DRAWS, MIX, 0.5)),
        "a quality target that isn't 100, 99 or 97% is refused",
    )
    other_items = table_of(haiku_only({"i0": 0.9, "i1": 0.7, "i2": 0.1, "x": 0.5}))
    check(refused(TraceError, lambda: walk(M1, other_items, DRAWS, MIX, 1.0)),
          "a table for other items than the matrix's is refused whole, by the simulator")  # fmt: skip
    drawn = walk(M1, M1_TABLE, resample(M1, 50, 1), MIX, 1.0)
    check(thresholds_of(drawn)[0] is None and drawn.steps[0].margin == 0 and drawn.steps[0].clears,
          "the walk also runs on the bootstrap's own resamples, with A first and clearing")  # fmt: skip

    print("\n-- the operating point --")
    non_monotone = walk(m2(M2_I1), M2_TABLE, DRAWS, MIX, 1.0)
    check(step_thresholds(M2_TABLE) == (0.05, 0.1, 0.4, 0.7, 0.9), "M2's steps are .05, .1, .4, .7, .9")
    # .9: i0 on haiku, $1 + i1, i2, i3 on opus $30 = $7.75. .7: i1 on opus_low $4: $25 / 4 = $6.25. .4: i1 on sonnet
    # $6: $6.75. .1: i1 on haiku $8: $7.25. .05: i2 and i3 on haiku too: $2.75, but i2 is wrong there: 75%.
    check(thresholds_of(non_monotone) == [None, 0.9, 0.7, 0.4, 0.1, 0.05]
          and [close(s.cost_per_1k, c) for s, c in zip(non_monotone.steps, (10, 7.75, 6.25, 6.75, 7.25, 2.75))] == [True] * 6,
          "the walk's costs are $10, 7.75, 6.25, 6.75, 7.25 and 2.75, the last step failing")  # fmt: skip
    check([s.clears for s in non_monotone.steps] == [True] * 5 + [False] and len(non_monotone.certified) == 5,
          "five steps are certified: A and the four down to .1")  # fmt: skip
    check(non_monotone.operating.threshold == 0.7 and close(non_monotone.operating.cost_per_1k, 6.25)
          and non_monotone.certified[-1].threshold == 0.1 and close(non_monotone.certified[-1].cost_per_1k, 7.25),
          "the operating point is .7 at $6.25, cheaper than the last certified step, .1 at $7.25")  # fmt: skip
    tie = walk(m2({**M2_I1, "sonnet": (0.004, 1)}), M2_TABLE, DRAWS, MIX, 1.0)
    check(close(tie.certified[2].cost_per_1k, 6.25) and close(tie.certified[3].cost_per_1k, 6.25)
          and tie.operating.threshold == 0.7,
          "with .7 and .4 both at $6.25 the tie goes to the earlier step, .7")  # fmt: skip

    print("\n-- the fallback to A --")
    first_fails = walk(M1, M1_FIRST_FAILS, DRAWS, MIX, 1.0)
    check(thresholds_of(first_fails) == [None, 0.9] and not first_fails.steps[1].clears,
          "when the first step below A fails, the walk ends there")  # fmt: skip
    a_outcomes = simulate(M1, always(M1, "opus"))
    check(first_fails.fell_back and first_fails.operating.threshold is None and len(first_fails.certified) == 1,
          "and the operating point is A, labeled as a fallback, with nothing else certified")  # fmt: skip
    check(first_fails.operating.outcomes == tuple(a_outcomes) and first_fails.operating.margin == 0,
          "A's outcomes: no router call, no overhead, margin 0")  # fmt: skip
    savings = DRAWS.savings(M1, a_outcomes, MIX)
    check(all(savings.captured(list(first_fails.operating.outcomes), share).value == 0.0 for share in TARGETS),
          "the fallback's captured fraction is 0.00 at every target")  # fmt: skip
    # Every step below A is certified, but with an overhead of $.02 an item each costs more than A's $10.
    dear = walk(M1, table_of(haiku_only({"i0": 0.9, "i1": 0.7, "i2": 0.1, "i3": 0.5}), overhead=0.02), DRAWS, MIX, 1.0)
    check(len(dear.certified) == 2 and dear.fell_back and dear.steps[1].cost_per_1k > dear.steps[0].cost_per_1k,
          "a certified step that costs more than A (here, by its overhead) isn't chosen: the operating point falls back")  # fmt: skip

    print("\n-- the targets, to the last bit --")
    # M3: A's quality is .33 x 3 + .01, a sum that need not be exactly 1, and 99% of it is .99 x that. Step .9 loses j0:
    # quality .99, short of the 100% target by .01, and level with the 99% target up to rounding, with margin 0.
    for share, steps, operating in (
        (1.0, [None, 0.9], None),
        (0.99, [None, 0.9, 0.2], 0.2),
        (0.97, [None, 0.9, 0.2], 0.2),
    ):
        at = walk(M3, M3_TABLE, M3_DRAWS, M3_MIX, share)
        check(thresholds_of(at) == steps and at.operating.threshold == operating,
              f"M3 at {share:.0%}: steps {steps}, operating threshold {operating}")  # fmt: skip
    m3 = walk(M3, M3_TABLE, M3_DRAWS, M3_MIX, 0.99)
    check(close(m3.steps[1].margin, 0, 1e-12) and close(m3.steps[1].cost_per_1k, 9.91) and close(m3.steps[2].cost_per_1k, 1.0),
          "its margin is 0, and the steps cost $9.91 and $1.00 per 1k (.33 x 3 x $10 + .01 x $1, then .99 x $1 + .01 x $1)")  # fmt: skip

    # M4 at 100%: the step's quality less the target is exactly 0 and its margin a rounding error, 5.7e-18. A strict
    # comparison would reject it; a step level with the target must clear, as the frontier's own answers do.
    level = walk(M4, M4_TABLE, DRAWS, MIX, 1.0)
    check(thresholds_of(level) == [None, 0.9, 0.05] and all(step.clears for step in level.steps)
          and 0 < level.steps[1].margin < 1e-15,
          f"a step level with the target up to rounding clears, though its margin is {level.steps[1].margin:.1e}")  # fmt: skip

    print("\n-- the operating-points file --")
    walks = {"uniform": {share: walk(M1, M1_TABLE, DRAWS, MIX, share) for share in TARGETS}}
    fallbacks = {"uniform": {share: walk(M1, M1_FIRST_FAILS, DRAWS, MIX, share) for share in TARGETS}}
    section = build_section(walks)
    entry = section["uniform"]["100%"]
    check(set(section["uniform"]) == {"100%", "99%", "97%"}
          and set(entry) == {"threshold", "margin", "target", "selection_method", "steps", "fell_back"},
          "a section holds, per mix and target, the threshold, margin, target, selection method, steps and fallback")  # fmt: skip
    check(entry["threshold"] == 0.9 and entry["margin"] == 0 and entry["fell_back"] is False and len(entry["steps"]) == 3
          and entry["steps"][0]["threshold"] is None and close(entry["steps"][2]["margin"], math.sqrt(1 / 24))
          and entry["steps"][2]["clears"] is False,
          "its values are the walk's: .9, margin 0, three steps from A (threshold null) to the failing one")  # fmt: skip
    fall = build_section(fallbacks)["uniform"]["100%"]
    check(fall["threshold"] is None and fall["fell_back"] is True and fall["margin"] == 0,
          "a fallback has no threshold")  # fmt: skip
    method = entry["selection_method"]
    check(method == SELECTION_METHOD and all(part in method for part in ("24", "quantile", "paired standard error", "ties")),
          "the selection method names the 24-step rule, the quantile method, the margin and the tie rule")  # fmt: skip
    check(refused(OperatingPointsError, lambda: build_section({"uniform": {1.0: walks["uniform"][1.0]}})),
          "a section missing a quality target is refused")  # fmt: skip
    check(refused(OperatingPointsError, lambda: build_section({})), "a section with no mix is refused")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "operating_points.json"
        write_section("E", section, path)
        check(read_section("E", path) == json.loads(json.dumps(section)), "a section read back is the section written")
        text_e = path.read_text()
        write_section("F", build_section(fallbacks), path)
        check(read_section("E", path) == json.loads(json.dumps(section)) and set(json.loads(path.read_text())) == {"E", "F"},
              "writing F's section leaves E's unchanged")  # fmt: skip
        write_section("E", section, path)
        write_section("F", build_section(walks), path)
        write_section("F", build_section(fallbacks), path)
        check(read_section("E", path) == json.loads(json.dumps(section)) and read_section("F", path)["uniform"]["100%"]["fell_back"],
              "rewriting a section replaces that router's alone")  # fmt: skip
        only_e = Path(tmp) / "only_e.json"
        write_section("E", section, only_e)
        check(only_e.read_text() == text_e, "the file's text is the same every time it is written")
        parsed = json.loads(path.read_text())
        check(list(parsed) == sorted(parsed) and list(parsed["E"]["uniform"]) == sorted(parsed["E"]["uniform"]),
              "the file is written with sorted keys")  # fmt: skip
        check(refused(OperatingPointsError, lambda: read_section("Z", path)), "a router with no section is refused")
        # A file that isn't a JSON object is refused whole, on read and before a write could overwrite it.
        broken = Path(tmp) / "broken.json"
        broken.write_text('{"E": ')
        check(
            refused(OperatingPointsError, lambda: read_section("E", broken)), "a file that isn't valid JSON is refused"
        )
        check(refused(OperatingPointsError, lambda: write_section("E", section, broken)) and broken.read_text() == '{"E": ',
              "and a write into it is refused, leaving the file as it was")  # fmt: skip
        listed = Path(tmp) / "listed.json"
        listed.write_text('[{"E": {}}]')
        check(refused(OperatingPointsError, lambda: read_section("E", listed)), "a file holding a JSON list is refused")
        check(refused(OperatingPointsError, lambda: write_section("E", section, listed)) and listed.read_text() == '[{"E": {}}]',
              "and a write into it is refused, leaving the file as it was")  # fmt: skip

    digest = section_hash(section)
    check(digest == section_hash(build_section(walks)) and digest == section_hash(json.loads(json.dumps(section))),
          "a section hashes the same when built again and when read back")  # fmt: skip
    check(section_hash(json.loads(json.dumps(section, sort_keys=False))) == digest
          and section_hash(dict(reversed(section.items()))) == digest,
          "and whatever order its keys were inserted in")  # fmt: skip
    moved: dict[str, Any] = json.loads(json.dumps(section))
    moved["uniform"]["100%"]["threshold"] += 1e-9
    check(section_hash(moved) != digest, "a threshold moved in the ninth decimal changes the hash")
    # The canonical text is sorted keys, two-space indent, the format of the file itself: hashed here by hand.
    tiny = {"b": 1, "a": None}
    check(section_hash(tiny) == hashlib.sha256(b'{\n  "a": null,\n  "b": 1\n}').hexdigest(),
          "the hash is the sha256 of the section's sorted, two-space-indented JSON text")  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
