"""The paired bootstrap (PLAN_v4.md §4; Amendment A8, items 8 and 9).

Source groups are resampled within each task, so a group's items are always drawn together, and the mix
weights are applied afterwards: 2,000 resamples from a fixed seed, with percentile 95% intervals. The same
resamples serve every policy and reference, so a policy's quality difference from the frontier is paired:
the luck of which items were drawn moves both alike. So is its captured fraction: each resample re-solves the
oracle on its own weights and takes its quality targets from its own frontier quality.
"""

import math
from dataclasses import dataclass

import numpy as np

from analysis.matrix import Matrix
from analysis.references import Curve, Oracle, captured_fraction, envelope, oracle
from analysis.simulator import Outcome, Total, check_mix, item_weights, totals

RESAMPLES, SEED = 2000, 2026


class BootstrapError(ValueError):
    """Outcomes that don't cover exactly the resampled items, once each. Nothing is estimated."""


@dataclass(frozen=True)
class Estimate:
    value: float  # on the items as drawn for dev
    low: float  # the percentile 95% interval over the resamples
    high: float
    standard_error: float  # the resamples' standard deviation


@dataclass(frozen=True)
class TotalEstimate:
    """`Total`'s quality and cost per 1,000 requests, each with its interval."""

    quality: Estimate
    cost_per_1k: Estimate


@dataclass(frozen=True)
class Captured:
    """A captured fraction with its paired interval, None being n/a (§7; Amendment A8, item 8)."""

    value: float | None  # on dev
    low: float | None  # the percentile 95% interval over every resample, n/a ranked below every number
    high: float | None
    met: float  # the share of resamples in which the policy meets the target


@dataclass(frozen=True)
class MatchedCost:
    """A policy's paired quality difference from the random-mixture envelope at the policy's own cost, with its
    interval, None being n/a (Amendment A9, item 12; A8 items 7 and 8)."""

    envelope: float | None  # the envelope's quality at the policy's cost, on dev
    difference: float | None  # the policy's quality less that, on dev
    low: float | None  # the percentile 95% interval over every resample, n/a ranked below every number
    high: float | None
    unread: float  # the share of resamples in which the policy costs less than the envelope's cheapest corner


@dataclass(frozen=True)
class Bootstrap:
    item_ids: tuple[str, ...]  # the order of `counts`' columns
    tasks: tuple[str, ...]  # each column's item's task
    counts: np.ndarray  # (resamples, items): how often each resample draws each item, with its source group

    def weights(self, mix: dict[str, float]) -> np.ndarray:
        """Each resample's weight on each item, (resamples, items): its task's mix weight, shared among the items
        drawn from the task by how often each was drawn (Amendment A5, item 6, applied after resampling)."""
        return self._weights_for(self.counts, mix)

    def item_weights(self, mix: dict[str, float], resample: int) -> dict[str, float]:
        """One resample's weight on each item, as the hindsight oracle takes them; an item not drawn weighs 0."""
        return dict(zip(self.item_ids, self._weights_for(self.counts[[resample]], mix)[0].tolist()))

    def resampled(self, outcomes: list[Outcome], mix: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """A policy's quality and cost per 1,000 requests under a mix, in each resample."""
        success, cost = self._columns(outcomes)
        weights = self.weights(mix)
        return weights @ success, 1000 * weights @ cost

    def resampled_totals(self, outcomes: list[Outcome], mix: dict[str, float]) -> list[Total]:
        """A policy's totals under a mix, in each resample."""
        quality, cost = self.resampled(outcomes, mix)
        return [Total(q, c) for q, c in zip(quality.tolist(), cost.tolist())]

    def totals(self, outcomes: list[Outcome], mix: dict[str, float]) -> TotalEstimate:
        """A policy's totals under a mix (`simulator.totals`), each with its interval over the resamples."""
        (quality, cost), total = self.resampled(outcomes, mix), totals(outcomes, mix)
        return TotalEstimate(_estimate(total.quality, quality), _estimate(total.cost_per_1k, cost))

    def task_totals(self, outcomes: list[Outcome]) -> dict[str, TotalEstimate]:
        """Each task's totals on its own (`simulator.task_totals`), with intervals over the same resamples, in the
        order tasks first appear.

        Tasks are resampled apart, so a task's share of every resample is a bootstrap of that task alone. A mix that
        weighs only the task gives it.
        """
        tasks = dict.fromkeys(outcome.task for outcome in outcomes)
        return {task: self.totals(outcomes, {other: float(other == task) for other in tasks}) for task in tasks}

    def difference(self, policy: list[Outcome], frontier: list[Outcome], mix: dict[str, float]) -> Estimate:
        """The policy's quality less the frontier's, under a mix, each resample weighing both alike."""
        (success, _), (frontier_success, _) = self._columns(policy), self._columns(frontier)
        value = totals(policy, mix).quality - totals(frontier, mix).quality
        return _estimate(value, self.weights(mix) @ (success - frontier_success))

    def safety_margin(self, policy: list[Outcome], frontier: list[Outcome], mix: dict[str, float]) -> float:
        """How far the policy's dev quality must clear a quality target under the mix: one standard error of its
        paired quality difference from the frontier (Amendment A8, item 9). Each candidate threshold is its own
        policy, with its own margin.

        Paired, because quality targets are shares of frontier quality on the same items: the luck of the draw moves
        the target and the policy together, so whether the target is met depends on their difference.
        """
        return self.difference(policy, frontier, mix).standard_error

    def matched_cost(
        self, policy: list[Outcome], fixed: dict[str, list[Outcome]], mix: dict[str, float]
    ) -> MatchedCost:
        """The envelope's quality at the policy's cost under a mix, and the policy's paired quality difference from
        it, with an interval (Amendment A9, items 12 and 15).

        The envelope is read as A8 item 7 settles: flat at its top corner's quality past its costliest corner, n/a
        below its cheapest. Each resample rebuilds the envelope from its own fixed-config points and reads it at its
        own policy cost, as the envelope's captured fraction does, so the difference is paired. A resample that
        reads n/a ranks below every number (A8 item 8), so the lower bound reads n/a when more than 2.5% do.
        """
        on_dev = envelope({config: totals(outcomes, mix) for config, outcomes in fixed.items()})
        point = totals(policy, mix)
        reading = on_dev.quality_at(point.cost_per_1k)
        differences = self.matched_cost_differences(policy, fixed, mix)
        low, high = _ranked_interval(differences)
        unread = sum(difference is None for difference in differences) / len(differences)
        return MatchedCost(reading, None if reading is None else point.quality - reading, low, high, unread)

    def matched_cost_differences(
        self, policy: list[Outcome], fixed: dict[str, list[Outcome]], mix: dict[str, float]
    ) -> list[float | None]:
        """In each resample, the policy's quality less the quality its envelope reads at the policy's cost there, or
        None (n/a) where the policy costs less than the envelope's cheapest corner."""
        resampled = {config: self.resampled_totals(outcomes, mix) for config, outcomes in fixed.items()}
        differences: list[float | None] = []
        for r, own in enumerate(self.resampled_totals(policy, mix)):
            reading = envelope({config: points[r] for config, points in resampled.items()}).quality_at(own.cost_per_1k)
            differences.append(None if reading is None else own.quality - reading)
        return differences

    def savings(self, matrix: Matrix, frontier: list[Outcome], mix: dict[str, float]) -> "Savings":
        """What captured fractions under a mix are measured against. The oracle is solved once on dev and once on
        each resample's weights, then shared by every policy and quality target (Amendment A8, item 8)."""
        resampled_oracles = tuple(oracle(matrix, self.item_weights(mix, r)) for r in range(len(self.counts)))
        return Savings(self, mix, totals(frontier, mix), oracle(matrix, item_weights(matrix.items.values(), mix)),
                       tuple(self.resampled_totals(frontier, mix)), resampled_oracles)  # fmt: skip

    def _columns(self, outcomes: list[Outcome]) -> tuple[np.ndarray, np.ndarray]:
        """The outcomes' successes and costs, in the order of `counts`' columns."""
        # Outcomes of another matrix's items would be weighed by the wrong draws, or not at all.
        by_item = {outcome.item_id: outcome for outcome in outcomes}
        if len(by_item) != len(outcomes) or by_item.keys() != set(self.item_ids):
            raise BootstrapError(f"{len(outcomes)} outcomes for {len(by_item)} items, but the bootstrap resamples "
                                 f"{len(self.item_ids)}, and they must cover them exactly, once each")  # fmt: skip
        ordered = [by_item[item_id] for item_id in self.item_ids]
        return np.array([o.success for o in ordered]), np.array([o.cost for o in ordered])

    def _weights_for(self, counts: np.ndarray, mix: dict[str, float]) -> np.ndarray:
        check_mix(self.tasks, mix)
        weights = np.zeros(counts.shape)
        for task, share in mix.items():
            in_task = np.array([item_task == task for item_task in self.tasks])
            drawn = counts[:, in_task]
            weights[:, in_task] = share * drawn / drawn.sum(axis=1, keepdims=True)
        return weights


@dataclass(frozen=True)
class Savings:
    """What captured fractions under one mix are measured against, on dev and in each resample: the frontier's
    totals, and the hindsight oracle solved on that resample's weights (Amendment A8, item 8)."""

    draws: Bootstrap
    mix: dict[str, float]
    frontier: Total  # on dev
    oracle: Oracle
    resampled_frontiers: tuple[Total, ...]
    resampled_oracles: tuple[Oracle, ...]

    def captured(self, policy: list[Outcome], share: float) -> Captured:
        """A policy's captured fraction at the quality target that is `share` of frontier quality, with its interval."""
        # A policy is a curve of one point: it meets a target at or below its quality, at its cost.
        return self._captured(share, Curve({"policy": totals(policy, self.mix)}),
                              [Curve({"policy": point}) for point in self.draws.resampled_totals(policy, self.mix)])  # fmt: skip

    def envelope_captured(self, fixed: dict[str, list[Outcome]], share: float) -> Captured:
        """The random-mixture envelope's captured fraction at a quality target, the bar any router must clear: its
        cheapest mix of the fixed configs that meets the target, each resample's envelope from the configs' points
        there."""
        resampled = {config: self.draws.resampled_totals(outcomes, self.mix) for config, outcomes in fixed.items()}
        curves = [envelope({config: points[r] for config, points in resampled.items()})
                  for r in range(len(self.resampled_frontiers))]  # fmt: skip
        return self._captured(share, envelope({config: totals(outcomes, self.mix) for config, outcomes in fixed.items()}),
                              curves)  # fmt: skip

    def _captured(self, share: float, on_dev: Curve, resampled: list[Curve]) -> Captured:
        # Each resample's target is the share of its own frontier quality, so the target and the policy stay paired.
        targets = [share * frontier.quality for frontier in self.resampled_frontiers]
        reached = [curve.meeting(target) for curve, target in zip(resampled, targets)]
        values = [captured_fraction(point, frontier, solved, target) for point, frontier, solved, target
                  in zip(reached, self.resampled_frontiers, self.resampled_oracles, targets)]  # fmt: skip
        target = share * self.frontier.quality
        value = captured_fraction(on_dev.meeting(target), self.frontier, self.oracle, target)
        met = sum(point is not None for point in reached) / len(reached)
        return Captured(value, *_ranked_interval(values), met)


def resample(matrix: Matrix, resamples: int = RESAMPLES, seed: int = SEED) -> Bootstrap:
    """Each resample draws, from each task, as many of its source groups as it has, with replacement."""
    rng = np.random.default_rng(seed)
    item_ids = tuple(matrix.items)
    column = {item_id: i for i, item_id in enumerate(item_ids)}
    counts = np.zeros((resamples, len(item_ids)), dtype=np.int64)
    # Tasks, groups and their items are taken in sorted order, so the draws don't depend on the order the store
    # returns its rows in.
    for task in sorted({item.task for item in matrix.items.values()}):
        groups: dict[str, list[int]] = {}
        for item_id in sorted(matrix.items):
            if (item := matrix.items[item_id]).task == task:
                groups.setdefault(item.source_group, []).append(column[item_id])
        members = [groups[group] for group in sorted(groups)]
        draws = rng.integers(len(members), size=(resamples, len(members)))
        drawn = np.zeros((resamples, len(members)), dtype=np.int64)
        np.add.at(drawn, (np.arange(resamples)[:, None], draws), 1)
        for group, columns in enumerate(members):
            counts[:, columns] = drawn[:, [group]]
    return Bootstrap(item_ids, tuple(matrix.items[item_id].task for item_id in item_ids), counts)


def _estimate(value: float, resampled: np.ndarray) -> Estimate:
    low, high = np.percentile(resampled, [2.5, 97.5])
    return Estimate(value, float(low), float(high), float(np.std(resampled, ddof=1)))


def _ranked_interval(values: list[float | None]) -> tuple[float | None, float | None]:
    """The percentile 95% interval of values where None, n/a, ranks below every number (Amendment A8, item 8).

    The lower bound reads n/a when more than 2.5% of the values are n/a, the upper when more than 97.5% are.
    Otherwise a bound interpolates between ranks as numpy's percentiles do, and one whose rank falls just short of
    the first number, beside an n/a, is that number.
    """
    numbers = sorted(value for value in values if value is not None)
    missing = len(values) - len(numbers)
    bounds: list[float | None] = []
    for share in (0.025, 0.975):
        rank = share * (len(values) - 1)
        below = math.floor(rank)
        if missing > share * len(values):
            bounds.append(None)
        elif below < missing:
            bounds.append(numbers[0])
        else:
            low, high = numbers[below - missing], numbers[min(below + 1, len(values) - 1) - missing]
            bounds.append(low + (rank - below) * (high - low))
    return bounds[0], bounds[1]
