"""Reference curves (PLAN_v4.md §7; Amendment A8, item 7).

A reference curve is a set of corners, each a quality and a cost per 1,000 requests,
joined by straight lines in linear cost. It answers both ways: the cost of reaching a
quality, for the captured fraction, and the quality reachable at a cost, for comparisons
at matched cost. A query outside the curve's range gives n/a, returned as None.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from analysis.matrix import LADDER
from analysis.simulator import Total


class CurveError(ValueError):
    """Corners that don't make a reference curve. No curve is built."""


@dataclass(frozen=True)
class Curve:
    corners: tuple[Total, ...]  # cheapest first; quality and cost both rise from corner to corner
    configs: tuple[str, ...]  # the config at each corner

    def __post_init__(self) -> None:
        # Interpolation assumes rising corners and doesn't check, so a bad curve would answer wrongly, not fail.
        if not self.corners:
            raise CurveError("a curve needs at least one corner")
        if len(self.configs) != len(self.corners):
            raise CurveError(f"the curve has {len(self.corners)} corners but names {len(self.configs)} configs "
                             f"{self.configs}")  # fmt: skip
        for (low, high), (lower, higher) in zip(pairwise(self.corners), pairwise(self.configs)):
            if not (low.quality < high.quality and low.cost_per_1k < high.cost_per_1k):
                raise CurveError(f"quality and cost must both rise from corner to corner, but {lower} is at {low} "
                                 f"and {higher} at {high}")  # fmt: skip

    def cost_at(self, quality: float) -> float | None:
        """Dollars per 1,000 requests to reach `quality`, or None outside the curve's qualities."""
        return _along([c.quality for c in self.corners], [c.cost_per_1k for c in self.corners], quality)

    def quality_at(self, cost_per_1k: float) -> float | None:
        """The quality reachable at `cost_per_1k`, or None outside the curve's costs."""
        return _along([c.cost_per_1k for c in self.corners], [c.quality for c in self.corners], cost_per_1k)


def _along(xs: list[float], ys: list[float], x: float) -> float | None:
    """y at x on the straight lines through the points (xs, ys), xs rising; None outside xs's range."""
    if not xs[0] <= x <= xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def _rise(low: Total, high: Total) -> float:
    """Quality gained per dollar per 1k, from one point to a costlier one."""
    return (high.quality - low.quality) / (high.cost_per_1k - low.cost_per_1k)


def envelope(points: dict[str, Total]) -> Curve:
    """Reference C, the random-mixture envelope of the fixed-config points under one mix.

    The upper concave hull of the points in linear cost, since mixing two configs at fixed
    odds traces the straight line between them. It starts at the cheapest point and climbs,
    each time to the point above with the steepest rise in quality per dollar, so points
    below the hull and points costlier with no higher quality drop out. It ends at its
    highest-quality corner.
    """
    if not points:
        raise CurveError("the envelope needs at least one config point")
    # Cheapest first, then the higher quality; exact ties go to the lower config (Amendment A8, item 5).
    # Every point above a corner then costs more than it, so a rise never divides by zero.
    order = sorted(points, key=lambda config: (points[config].cost_per_1k, -points[config].quality,
                                               LADDER.index(config)))  # fmt: skip
    configs = [order[0]]
    while above := [config for config in order if points[config].quality > points[configs[-1]].quality]:
        here = points[configs[-1]]
        # Of points equally steep, the farthest: the nearer lies on the line, not at a corner.
        configs.append(max(above, key=lambda config: (_rise(here, points[config]), points[config].cost_per_1k)))
    return Curve(tuple(points[config] for config in configs), tuple(configs))
