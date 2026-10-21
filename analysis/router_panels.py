"""A router's part of chart and table v1: its steps and operating points (PLAN_v4.md §12; Amendment A9, items 14, 15).

Everything here is JSON-ready data that `analysis/pareto.py` adds to table v0's panels under new keys and plots, so a
router's part never changes a number of table v0. It reads a router's walks (`analysis/walk.py`), never a router.

Per mix, a router has its steps, the same on both mixes' panels since they come from router scores: always-frontier
(A) first, then each step from the highest threshold to the lowest, which is the walk's order and may double back in
cost. And at each of its six operating points: the walk's verdict, quality and cost with intervals, margin, paired
difference from A, savings against A, captured fraction with the share of resamples that meet the target, the routing
mix and the comparison with the envelope at matched cost (item 15). Per task, the steps again, and at each operating
point the quality and cost with intervals and McNemar's test against A (item 11), null for a row that falls back to A.
"""

from dataclasses import asdict

from analysis.bootstrap import Bootstrap, Savings, TotalEstimate
from analysis.matrix import LADDER, Matrix
from analysis.mcnemar import mcnemar
from analysis.references import TARGETS
from analysis.router_table import RouterScoreTable
from analysis.simulator import Outcome, item_weights, simulate, task_totals, totals
from analysis.walk import Walk, step_thresholds
from routers.ladder import FRONTIER

A = "A"  # always-frontier, the walk's first step; its threshold is None


def target_key(share: float) -> str:
    return f"{share:.0%}"


def step_outcomes(dev: Matrix, table: RouterScoreTable, frontier: list[Outcome]) -> dict[float | None, list[Outcome]]:
    """Every step's outcomes on dev, in walk order: A, then the thresholds from the highest down.

    All of them, not only those a walk tried: the chart draws E's whole line. The walks themselves never simulate a
    step beyond the first that misses.
    """
    return {None: frontier} | {t: simulate(dev, table.traces(t)) for t in reversed(step_thresholds(table))}


def routing_mix(outcomes: list[Outcome], mix: dict[str, float]) -> dict[str, float]:
    """The share of requests each config answered under the mix: each item's weight on the config that answered."""
    weights = item_weights(outcomes, mix)
    shares = dict.fromkeys(LADDER, 0.0)
    for outcome in outcomes:
        shares[outcome.configs[-1]] += weights[outcome.item_id]
    return shares


def mix_router(
    draws: Bootstrap,
    savings: Savings,
    steps: dict[float | None, list[Outcome]],
    walks: dict[float, Walk],
    fixed: dict[str, list[Outcome]],
    mix: dict[str, float],
) -> dict:
    """One router's part of a mix's panel: its steps for the line, and its operating points for the crosses."""
    frontier = fixed[FRONTIER]
    frontier_total = totals(frontier, mix)
    line = [{"threshold": threshold, **asdict(totals(outcomes, mix))} for threshold, outcomes in steps.items()]
    points = {}
    for share in TARGETS:
        result = walks[share]
        outcomes = steps[result.operating.threshold]
        total = totals(outcomes, mix)
        points[target_key(share)] = {
            "threshold": result.operating.threshold,
            "fell_back": result.fell_back,
            "steps_certified": len(result.certified) - 1,  # A is certified without a step
            "steps_tried": len(result.steps) - 1,
            "target": result.target,
            "margin": result.operating.margin,
            **asdict(draws.totals(outcomes, mix)),
            "difference_from_A": asdict(draws.difference(outcomes, frontier, mix)),
            "savings_vs_A": {
                "per_1k": frontier_total.cost_per_1k - total.cost_per_1k,
                "share": 1 - total.cost_per_1k / frontier_total.cost_per_1k,
            },  # fmt: skip
            "captured": asdict(savings.captured(outcomes, share)),
            "routing": routing_mix(outcomes, mix),
            "matched_cost": asdict(draws.matched_cost(outcomes, fixed, mix)),
        }
    return {"steps": line, "operating_points": points}


def mcnemar_row(policy: list[Outcome], frontier: list[Outcome], task: str) -> dict[str, float] | None:
    """McNemar's b, c and p of a policy against A within one task, or None where the policy is A itself."""
    if policy is frontier:
        return None
    found = mcnemar([o for o in policy if o.task == task], [o for o in frontier if o.task == task], "policy", A)
    return {"b": found.b, "c": found.c, "p": found.p}


def task_router(
    draws: Bootstrap,
    steps: dict[float | None, list[Outcome]],
    walks: dict[str, dict[float, Walk]],
    task: str,
) -> dict:
    """One router's part of a task's panel: its steps for the line, and each of the six operating points' quality and
    cost with intervals and McNemar's test against A. The operating points are tabled, not drawn: verdicts are given
    for the mixes only (Amendment A7, item 2)."""
    line = [{"threshold": threshold, **asdict(task_totals(outcomes)[task])} for threshold, outcomes in steps.items()]
    frontier = steps[None]
    spread: dict[float | None, TotalEstimate] = {}
    points: dict[str, dict] = {}
    for mix, by_share in walks.items():
        points[mix] = {}
        for share, result in by_share.items():
            threshold = result.operating.threshold
            if threshold not in spread:  # only the operating points' steps get intervals
                spread[threshold] = draws.task_totals(steps[threshold])[task]
            # A row that falls back to A is A, so it has nothing to test against.
            tested = None if result.fell_back else mcnemar_row(steps[threshold], frontier, task)
            points[mix][target_key(share)] = {"threshold": threshold, "fell_back": result.fell_back,
                                              **asdict(spread[threshold]), "mcnemar": tested}  # fmt: skip
    return {"steps": line, "operating_points": points}


def config_mcnemar(fixed: dict[str, list[Outcome]], task: str) -> dict[str, dict[str, float] | None]:
    """McNemar's test of each fixed config against A within a task; A's own row is None."""
    return {config: mcnemar_row(outcomes, fixed[FRONTIER], task) for config, outcomes in fixed.items()}
