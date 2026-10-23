"""The same-target comparison of two operating points (PLAN_v4.md §8; Amendment A11, item 13).

Operating point against operating point, at one task mix and one quality target: the paired differences in cost per
1k and in quality, first less second, from the bootstrap's own resamples (so they pair with every other interval).
Where either side falls back to A, that side is A, with A's outcomes; where both do, the result reads "both fall
back to A" and holds no difference. It is descriptive, with no multiplicity correction: at 95%, about one interval in
twenty excludes 0 by chance. Reading one router's quality at the other's cost off its step curve is rejected: the
curve can double back (A9 item 14), so one cost can have several qualities.
"""

from dataclasses import dataclass

from analysis.bootstrap import Bootstrap, Estimate, percentile_estimate
from analysis.simulator import totals
from analysis.walk import Walk


class SameTargetError(ValueError):
    """Walks that can't be compared at one target: for different quality targets, or with no steps. Nothing is
    compared."""


@dataclass(frozen=True)
class SameTarget:
    first_fell_back: bool
    second_fell_back: bool
    cost: Estimate | None  # first less second, dollars per 1k at list price; None where both fall back to A
    quality: Estimate | None  # first less second, 0 to 1 under the mix; None where both fall back to A

    @property
    def both_fall_back(self) -> bool:
        return self.first_fell_back and self.second_fell_back

    @property
    def reading(self) -> str:
        """What a table row says in place of a difference, empty where there is one."""
        return "both fall back to A" if self.both_fall_back else ""


def compare_same_target(draws: Bootstrap, mix: dict[str, float], first: Walk, second: Walk) -> SameTarget:
    """The operating point of `first`'s walk less that of `second`'s, under `mix`: cost per 1k and quality, each with
    its percentile 95% interval over the resamples, each resample weighing both alike.

    Refuses (SameTargetError) walks for different quality targets or with no steps, (BootstrapError) outcomes that
    don't cover the resampled items exactly, once each. A side that falls back is read as A, whose outcomes its walk
    holds as its operating point.
    """
    if first.share != second.share:
        raise SameTargetError(f"the walks must be for the same target, but they are for {first.share} and {second.share} "
                              f"of A's quality")  # fmt: skip
    if not (first.steps and second.steps):
        raise SameTargetError("a walk with no steps has no operating point, not even A")
    if first.fell_back and second.fell_back:
        return SameTarget(True, True, None, None)
    first_quality, first_cost = draws.resampled(list(first.operating.outcomes), mix)
    second_quality, second_cost = draws.resampled(list(second.operating.outcomes), mix)
    first_total, second_total = (
        totals(list(first.operating.outcomes), mix),
        totals(list(second.operating.outcomes), mix),
    )
    return SameTarget(
        first.fell_back, second.fell_back,
        percentile_estimate(first_total.cost_per_1k - second_total.cost_per_1k, first_cost - second_cost),
        percentile_estimate(first_total.quality - second_total.quality, first_quality - second_quality),
    )  # fmt: skip
