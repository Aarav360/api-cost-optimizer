"""Reference curves (PLAN_v4.md §7; Amendment A8, items 6 and 7).

A reference curve is a set of corners, each a quality and a cost per 1,000 requests,
joined by straight lines in linear cost. It answers both ways: the cost of reaching a
quality, for the captured fraction, and the quality reachable at a cost, for comparisons
at matched cost. A query outside the curve's range gives n/a, returned as None.

Reference C, the random-mixture envelope, is such a curve through fixed-config points.
Reference D, the hindsight oracle, is the plan's LP solved greedily over each item's
hull; its curve runs from every item on its cheapest config up to its maximum quality.
"""

import heapq
import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from analysis.matrix import LADDER, Matrix
from analysis.simulator import Total, item_weights

START = "every item on its cheapest config"  # the oracle curve's first corner
TARGETS = (1.0, 0.99, 0.97)  # §11's quality targets, as shares of frontier quality
# How far apart two sums of the same scores or dollars can land, added in another order: far below any digit reported.
ROUNDING = 1e-9


def quality_targets(frontier_quality: float) -> dict[float, float]:
    """Each quality target under one mix, keyed by its share of the frontier's quality there (§11).

    A resample recomputes them from its own frontier quality, so they stay paired (Amendment A8, item 8).
    """
    return {share: share * frontier_quality for share in TARGETS}


class CurveError(ValueError):
    """Corners that don't make a reference curve. No curve is built."""


@dataclass(frozen=True)
class Curve:
    # Each corner's point, cheapest first, keyed by its name: the config at an envelope's corner, or the upgrade
    # that reaches an oracle's.
    corners: dict[str, Total]

    def __post_init__(self) -> None:
        # Interpolation assumes rising corners and doesn't check, so a bad curve would answer wrongly, not fail.
        if not self.corners:
            raise CurveError("a curve needs at least one corner")
        if unfinite := [name for name, point in self.corners.items() if not _finite(point)]:
            raise CurveError(f"corners need a finite quality and cost, and {unfinite} don't: {self.corners}")
        for (lower, low), (higher, high) in pairwise(self.corners.items()):
            if not (low.quality < high.quality and low.cost_per_1k < high.cost_per_1k):
                raise CurveError(f"quality and cost must both rise from corner to corner, but {lower} is at {low} "
                                 f"and {higher} at {high}")  # fmt: skip

    def cost_at(self, quality: float) -> float | None:
        """Dollars per 1,000 requests to reach `quality`, or None outside the curve's qualities."""
        points = self.corners.values()
        return _along([p.quality for p in points], [p.cost_per_1k for p in points], quality)

    def quality_at(self, cost_per_1k: float) -> float | None:
        """The quality reachable at `cost_per_1k`, or None outside the curve's costs."""
        points = self.corners.values()
        return _along([p.cost_per_1k for p in points], [p.quality for p in points], cost_per_1k)

    def meeting(self, quality: float) -> Total | None:
        """The curve's cheapest point whose quality is at least `quality`, or None above its last corner.

        A quality at or below the first corner's is met by that corner, as picking its config alone meets it.
        """
        first = next(iter(self.corners.values()))
        if quality <= first.quality:
            return first
        cost = self.cost_at(quality)
        return None if cost is None else Total(quality, cost)

    @property
    def peak(self) -> Total:
        """The last corner: the curve's highest quality."""
        return list(self.corners.values())[-1]


def _along(xs: list[float], ys: list[float], x: float) -> float | None:
    """y at x on the straight lines through the points (xs, ys), xs rising; None outside xs's range."""
    if not xs[0] <= x <= xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def _finite(point: Total) -> bool:
    return math.isfinite(point.quality) and math.isfinite(point.cost_per_1k)


def _rise(low: Total, high: Total) -> float:
    """Quality gained per dollar per 1k, from one point to a costlier one."""
    return (high.quality - low.quality) / (high.cost_per_1k - low.cost_per_1k)


def _climb(points: dict[str, Total]) -> list[str]:
    """The configs at the corners of the upper concave hull of `points` in linear cost, cheapest first.

    It starts at the cheapest point and climbs, each time to the point above with the steepest rise in quality per
    dollar, so points below the hull and points costlier with no higher quality drop out. It ends at the highest
    quality.
    """
    # Cheapest first, then the higher quality; exact ties go to the lower config (Amendment A8, item 5).
    # Every point above a corner then costs more than it, so a rise never divides by zero.
    order = sorted(points, key=lambda config: (points[config].cost_per_1k, -points[config].quality,
                                               LADDER.index(config)))  # fmt: skip
    configs = [order[0]]
    while above := [config for config in order if points[config].quality > points[configs[-1]].quality]:
        here = points[configs[-1]]
        # Of points equally steep, the farthest: the nearer lies on the line, not at a corner.
        configs.append(max(above, key=lambda config: (_rise(here, points[config]), points[config].cost_per_1k)))
    return configs


def envelope(points: dict[str, Total]) -> Curve:
    """Reference C, the random-mixture envelope of the fixed-config points under one mix.

    The upper concave hull of the points in linear cost, since mixing two configs at fixed
    odds traces the straight line between them. It ends at its highest-quality corner.
    """
    if not points:
        raise CurveError("the envelope needs at least one config point")
    # Before the climb, which can't order a config off the ladder and would quietly drop a point that isn't a number.
    if off_ladder := sorted(set(points) - set(LADDER)):
        raise CurveError(f"{off_ladder} are not configs on the ladder {LADDER}")
    if unfinite := [config for config, point in points.items() if not _finite(point)]:
        raise CurveError(f"config points need a finite quality and cost, and {unfinite} don't: {points}")
    return Curve({config: points[config] for config in _climb(points)})


class OracleError(ValueError):
    """A matrix or item weights the oracle refuses. Nothing is solved."""


def _validate(matrix: Matrix, weights: dict[str, float]) -> None:
    # An item with no stored answer has no config to start on.
    if bare := sorted(set(matrix.items) - {item_id for item_id, _ in matrix.cells}):
        raise OracleError(f"{len(bare)} items have no stored answer, e.g. {bare[0]}")
    # A weight left out would drop the item's cost and quality, as a skipped item would a policy's; a resample
    # gives an item it didn't draw a weight of 0 instead.
    if missing := sorted(set(matrix.items) - set(weights)):
        raise OracleError(f"{len(missing)} items have no weight, e.g. {missing[0]}")
    if unknown := sorted(set(weights) - set(matrix.items)):
        raise OracleError(f"{len(unknown)} weights are for items not in the matrix, e.g. {unknown[0]}")
    if bad := sorted(item_id for item_id, weight in weights.items() if not (math.isfinite(weight) and weight >= 0)):
        raise OracleError(f"weights must be finite and at least 0, and {len(bad)} aren't, e.g. {bad[0]}: "
                          f"{weights[bad[0]]}")  # fmt: skip
    # Quality is a weighted mean, so its weights sum to 1, as every mix's and every resample's do.
    if not math.isclose(total := sum(weights.values()), 1, rel_tol=1e-9):
        raise OracleError(f"weights must sum to 1, and these sum to {total}")


@dataclass(frozen=True)
class Upgrade:
    """One step up an item's hull, from one config to the next."""

    item_id: str
    before: str
    after: str
    gain: Total  # what the whole step adds to the oracle's quality and cost per 1,000 requests


@dataclass(frozen=True)
class Oracle:
    """Reference D, the hindsight oracle, solved for one set of item weights."""

    weights: dict[str, float]
    start: dict[str, str]  # each item's cheapest config, where it starts
    upgrades: tuple[Upgrade, ...]  # in the order taken, most quality per dollar first
    curve: Curve  # every item on its cheapest config, then the point after each upgrade

    def cost_at(self, quality: float) -> float | None:
        """Dollars per 1,000 requests to reach `quality`, or None above the oracle's maximum: n/a (infeasible)."""
        # Sending every item to its cheapest config, the first corner, already meets a target at or below its
        # quality (A8 item 6).
        point = self.curve.meeting(quality)
        return None if point is None else point.cost_per_1k

    def quality_at(self, cost_per_1k: float) -> float | None:
        """The quality reachable at `cost_per_1k`, or None outside the curve's costs."""
        return self.curve.quality_at(cost_per_1k)

    def routing_at(self, quality: float) -> dict[str, float] | None:
        """The share of requests on each config when the oracle meets `quality`, or None where it can't.

        Shares are weighted like quality and cost, so a per-task oracle's are its shares of items.
        """
        if self.cost_at(quality) is None:
            return None
        shares = dict.fromkeys(LADDER, 0.0)
        for item_id, config in self.start.items():
            shares[config] += self.weights[item_id]
        for upgrade, (low, high) in zip(self.upgrades, pairwise(self.curve.corners.values())):
            if quality <= low.quality:
                break
            # The last upgrade taken may be a part of one: a mix of the item's two configs, which makes this an LP.
            moved = self.weights[upgrade.item_id] * min(1.0, (quality - low.quality) / (high.quality - low.quality))
            shares[upgrade.before] -= moved
            shares[upgrade.after] += moved
        return shares


def oracle(matrix: Matrix, weights: dict[str, float]) -> Oracle:
    """Reference D, the hindsight oracle: §7's LP, solved greedily (Amendment A8, item 6).

    `weights` gives each item's weight; A5 item 6's are `item_weights(matrix.items.values(), mix)`. Each item's
    hull runs over the scores and costs of the configs it has answers for, so on train, where the frontier never
    runs (Amendment A1, item 3), it never reaches opus. Dominated configs drop out and the lower convex hull is
    kept.
    Every item starts at its cheapest hull point, and steps up the hulls are taken across all items in decreasing
    order of score gained per dollar.

    A weight scales an item's score and its cost alike, since cost per request is mix-weighted too (A5 item 6):
    the oracle minimizes the cost policies are scored on. So the weight sets what a step adds, not where it falls
    in the order.
    """
    _validate(matrix, weights)
    start, steps = {}, {}
    for item_id in matrix.items:
        # Each config's point for the item on its own: its score, and its cost per 1,000 requests.
        points = {config: Total(cell.success, 1000 * cell.cost) for config in LADDER
                  if (cell := matrix.cells.get((item_id, config)))}  # fmt: skip
        hull = _climb(points)
        start[item_id] = hull[0]
        # Stepping up an item that weighs nothing, one a resample didn't draw, would buy no quality.
        steps[item_id] = [(before, after, points[before], points[after]) for before, after in pairwise(hull)
                          if weights[item_id] > 0]  # fmt: skip

    # A heap holds each item's next step, so an item's steps are taken in order. Of steps equally steep, the one
    # to the lower config goes first (Amendment A8, item 5), then the one for the item first in the matrix.
    order = {item_id: i for i, item_id in enumerate(matrix.items)}
    queue: list[tuple[float, int, int, str, int]] = []

    def push(item_id: str, step: int) -> None:
        if step < len(steps[item_id]):
            _, after, low, high = steps[item_id][step]
            heapq.heappush(queue, (-_rise(low, high), LADDER.index(after), order[item_id], item_id, step))

    for item_id in matrix.items:
        push(item_id, 0)
    upgrades = []
    while queue:
        *_, item_id, step = heapq.heappop(queue)
        before, after, low, high = steps[item_id][step]
        weight = weights[item_id]
        gain = Total(weight * (high.quality - low.quality), weight * (high.cost_per_1k - low.cost_per_1k))
        upgrades.append(Upgrade(item_id, before, after, gain))
        push(item_id, step + 1)

    point = Total(sum(weights[i] * matrix.cells[(i, c)].success for i, c in start.items()),
                  sum(weights[i] * 1000 * matrix.cells[(i, c)].cost for i, c in start.items()))  # fmt: skip
    corners = {START: point}
    for upgrade in upgrades:
        point = Total(point.quality + upgrade.gain.quality, point.cost_per_1k + upgrade.gain.cost_per_1k)
        corners[f"{upgrade.item_id} to {upgrade.after}"] = point
    return Oracle(weights, start, tuple(upgrades), Curve(corners))


def task_oracles(matrix: Matrix) -> dict[str, Oracle]:
    """Each task's own oracle, over only its items, in the order tasks first appear (Amendment A8, item 6)."""
    oracles: dict[str, Oracle] = {}
    for task in dict.fromkeys(item.task for item in matrix.items.values()):
        own = matrix.task(task)
        oracles[task] = oracle(own, item_weights(own.items.values(), {task: 1.0}))
    return oracles


class BoundError(RuntimeError):
    """A policy met a quality target for less than the hindsight oracle's cost there, or met one above the oracle's
    maximum: a captured fraction above 1, which can't happen (Amendment A8, item 6). The policy's accounting or the
    oracle is wrong, so the run stops."""


def captured_fraction(reached: Total | None, frontier: Total, solved: Oracle, target: float) -> float | None:
    """The share of the oracle's savings at a quality target that a policy captures, or None: n/a (§7).

    `reached` is the policy's totals, or None for a reference that can't reach the target. The fraction is
    (frontier cost - policy cost) / oracle savings, where the oracle saves frontier cost - its cost at the target.
    It is n/a where the policy misses the target or the oracle saves nothing there.
    """
    if reached is None or reached.quality < target:
        return None
    # The oracle picks each item's config with hindsight, so no policy can meet a target it can't, or for less.
    # A target a rounding error above its maximum is the maximum, which it always reaches.
    peak = solved.curve.peak.quality
    if target > peak + ROUNDING or (oracle_cost := solved.cost_at(min(target, peak))) is None:
        raise BoundError(f"a policy at {reached} meets a target of {target}, above the oracle's maximum quality, "
                         f"{peak}")  # fmt: skip
    if reached.cost_per_1k < oracle_cost - ROUNDING:
        raise BoundError(f"a policy at {reached} meets a target of {target} for less than the oracle's "
                         f"${oracle_cost} per 1k")  # fmt: skip
    if (savings := frontier.cost_per_1k - oracle_cost) <= 0:
        return None
    # A policy on the oracle's own point can land a rounding error below its cost, which counts as its cost.
    return (frontier.cost_per_1k - max(reached.cost_per_1k, oracle_cost)) / savings
