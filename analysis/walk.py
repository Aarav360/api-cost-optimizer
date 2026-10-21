"""Threshold selection: breakpoints, steps and the walk (PLAN_v4.md §8; Amendment A9, items 9 and 10).

Each threshold is its own policy, with its own safety margin (Amendment A8, item 9), and thresholds are tried in a
fixed sequence, so trying many doesn't raise the chance of missing a target on test. Everything here reads a
router-score table and a results matrix, never a router, and runs on any router's table.

An item's breakpoints are its router scores strictly higher than every cheaper config's: the thresholds at which its
config changes. The steps come from the table's breakpoints alone, before any outcome is looked up, which is what keeps
the guarantee. The walk, one per task mix and quality target, starts at always-frontier (A: no router call, no
overhead) and tries the steps from the highest threshold to the lowest. A step clears if its quality less the target
is at least its margin, one paired standard error of its quality difference from A. The walk stops at the first step
that doesn't clear, and every step before it is certified. The operating point is the cheapest certified step by
cost per 1k, ties to the earlier step, or A itself where nothing certified costs less.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from analysis.bootstrap import Bootstrap
from analysis.matrix import Matrix
from analysis.references import ROUNDING, TARGETS, quality_targets
from analysis.router_table import RouterScoreTable
from analysis.simulator import Outcome, always, simulate, totals
from routers.ladder import FRONTIER, SCORED

ALL_STEPS_UP_TO = 24  # distinct breakpoints up to this many are each a step (E's eight buckets x 3 configs)
QUANTILE_STEPS = 20  # otherwise the 0%, 5%, ..., 95% quantiles of the pooled breakpoints


class WalkError(ValueError):
    """A walk that can't be run: a quality target that isn't one of §11's. Nothing is walked."""


def item_breakpoints(router_scores: Mapping[str, float]) -> list[float]:
    """One item's breakpoints: its router scores strictly higher than every cheaper config's, cheapest config first.

    The cheapest config's router score is one, as nothing is cheaper. A router score that is not higher than a
    cheaper config's can never change the item's decision (the decision rule takes the cheaper config first), so an
    equal one isn't one either.
    """
    found: list[float] = []
    for config in SCORED:
        if all(router_scores[config] > router_scores[cheaper] for cheaper in SCORED[: SCORED.index(config)]):
            found.append(router_scores[config])
    return found


def step_thresholds(table: RouterScoreTable) -> tuple[float, ...]:
    """The walk's steps, in ascending order of threshold, from the table alone.

    Every distinct breakpoint if there are 24 or fewer. Otherwise the 0%, 5%, ..., 95% quantiles of the pooled
    breakpoints, each item's counted once per breakpoint so a quantile moves about that share of decisions, with
    duplicate steps removed. The quantile is the lower empirical one (numpy's `inverted_cdf`): the smallest pooled
    breakpoint with at least that share of them at or below it, so every step is a router score the router gave.
    """
    pooled = sorted(value for entry in table.items for value in item_breakpoints(entry.router_scores))
    distinct = sorted(set(pooled))
    if len(distinct) <= ALL_STEPS_UP_TO:
        return tuple(distinct)
    n = len(pooled)
    # The k/20 quantile is the ceil(n k / 20)-th smallest (1-based, and the first for k = 0), in whole numbers so no
    # float product can tip a ceiling.
    positions = [max(1, -(-n * k // QUANTILE_STEPS)) for k in range(QUANTILE_STEPS)]
    return tuple(sorted({pooled[p - 1] for p in positions}))


@dataclass(frozen=True)
class Step:
    """A policy tried by the walk: A (threshold None) or the router at a threshold, on dev."""

    threshold: float | None
    quality: float  # mix-weighted, 0 to 1
    cost_per_1k: float  # dollars at list price, overhead included
    margin: float  # one paired standard error of the quality difference from A; exactly 0 for A
    clears: bool  # quality less the target is at least the margin
    # The policy's outcomes, for the intervals and the captured fraction that table v1 reports.
    outcomes: tuple[Outcome, ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class Walk:
    """One walk, for one task mix and quality target."""

    share: float  # the quality target as a share of A's quality (§11)
    target: float  # that share of A's quality under the mix, on dev
    steps: tuple[Step, ...]  # A first, then each step tried, the last one the first that didn't clear (if any)

    @property
    def certified(self) -> tuple[Step, ...]:
        """A and every step before the first that didn't clear."""
        stop = next((i for i, step in enumerate(self.steps) if not step.clears), len(self.steps))
        return self.steps[:stop]

    @property
    def operating(self) -> Step:
        """The cheapest certified step by cost per 1k, ties to the earlier; A where nothing certified costs less.

        A is a candidate, so a router whose certified steps all cost more than A (an overhead can do that) falls
        back to it. Costs that differ only by the order floating-point sums were added in count as a tie.
        """
        best = self.steps[0]
        for step in self.certified[1:]:
            if step.cost_per_1k < best.cost_per_1k - ROUNDING:
                best = step
        return best

    @property
    def fell_back(self) -> bool:
        """Whether the operating point is A itself, "falls back to A": no router call, captured fraction 0.00."""
        return self.operating.threshold is None


def walk(matrix: Matrix, table: RouterScoreTable, draws: Bootstrap, mix: dict[str, float], share: float) -> Walk:
    """The walk down the table's steps under one mix, for the quality target that is `share` of A's quality.

    Refuses (WalkError) a share that isn't one of §11's targets, and (TraceError) a table for other items than the
    matrix's. Later steps than the first failure are never simulated, so they reveal no outcome.
    """
    if share not in TARGETS:
        raise WalkError(f"the quality target must be one of {TARGETS} times A's quality, not {share}")
    frontier = simulate(matrix, always(matrix, FRONTIER))
    frontier_total = totals(frontier, mix)
    target = quality_targets(frontier_total.quality)[share]
    # A's margin is exactly 0, and it clears: its quality is the frontier's, and no target is above that.
    steps = [Step(None, frontier_total.quality, frontier_total.cost_per_1k, 0.0, True, tuple(frontier))]
    for threshold in reversed(step_thresholds(table)):
        outcomes = simulate(matrix, table.traces(threshold))
        total, margin = totals(outcomes, mix), draws.safety_margin(outcomes, frontier, mix)
        # The tolerance is for sums added in another order: a step level with the target, such as one that
        # reproduces A's answers, must clear it with a margin of 0, not miss it by a rounding error.
        clears = total.quality - target >= margin - ROUNDING
        steps.append(Step(threshold, total.quality, total.cost_per_1k, margin, clears, tuple(outcomes)))
        if not clears:
            break
    return Walk(share, target, tuple(steps))
