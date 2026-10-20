"""The paired bootstrap (PLAN_v4.md §4; Amendment A8, items 8 and 9).

Source groups are resampled within each task, so a group's items are always drawn together, and the mix
weights are applied afterwards: 2,000 resamples from a fixed seed, with percentile 95% intervals. The same
resamples serve every policy and reference, so a policy's quality difference from the frontier is paired:
the luck of which items were drawn moves both alike.
"""

from dataclasses import dataclass

import numpy as np

from analysis.matrix import Matrix
from analysis.simulator import Outcome, check_mix, totals

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

    def totals(self, outcomes: list[Outcome], mix: dict[str, float]) -> TotalEstimate:
        """A policy's totals under a mix (`simulator.totals`), each with its interval over the resamples."""
        success, cost = self._columns(outcomes)
        total, weights = totals(outcomes, mix), self.weights(mix)
        return TotalEstimate(_estimate(total.quality, weights @ success),
                             _estimate(total.cost_per_1k, 1000 * weights @ cost))  # fmt: skip

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
