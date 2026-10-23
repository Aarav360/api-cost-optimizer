"""Within-task concordance and log loss, with paired bootstrap intervals (PLAN_v4.md §8; Amendment A11, items 7, 12).

Both read a config's router scores and its successes, one of each per item, and the bootstrap's resamples
(`analysis.bootstrap`), so a metric and the difference between two of them are paired: the luck of which source
groups were drawn moves both alike. Neither reads a mix; they count items or pairs, as McNemar's test does.

**Within-task concordance** is the share of (success, failure) pairs within one cell whose router scores are in the
right order, ties counting half. A cell is a task and, for cross-fitted router scores, a fold (a train fold on train);
for a single model it is the task alone. Pairs across cells are never counted: each fold's task rates come from the
other folds' labels, so pairs across folds are ordered against the labels, and a router that knows only the task
would score below 0.5 (item 12). A resample draws pairs by how often it draws the two items, and one that draws no
pair is n/a: it is left out of the interval, whose `undefined` says what share that was.

**Log loss** is the mean of -ln of the router score for the item's outcome: the router score itself where the config
succeeded, one less the router score where it failed. A router score of exactly 0 or 1 is clipped to the float epsilon, as scikit-learn's log loss does, so it costs a finite amount.

Both are reported pooled over items (or pairs) and per task, with A7 item 2's warnings left to the reader of the
tables; `difference` takes two readings over the same resamples, as the learning curve's 75% to 100% step does.
"""

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field

import numpy as np

from analysis.bootstrap import Bootstrap

_EPSILON = float(np.finfo(np.float64).eps)  # scikit-learn's log loss clips router scores to [eps, 1 - eps]


class MetricsError(ValueError):
    """Router scores, successes or cells a metric refuses: not exactly the resampled items, or out of range. Nothing
    is computed."""


@dataclass(frozen=True)
class PairedEstimate:
    value: float | None  # on dev; None where there is nothing to read (no pair)
    low: float | None  # the percentile 95% interval over the resamples that have a value; None if none has
    high: float | None
    undefined: float  # the share of resamples with no value


@dataclass(frozen=True)
class Reading:
    """A metric pooled over all items and per task, in the order tasks first appear."""

    pooled: PairedEstimate
    per_task: dict[str, PairedEstimate]
    size: dict[str | None, int]  # pairs (concordance) or items (log loss) on dev: None pooled, then each task
    # Each resample's value (NaN where undefined), for `difference`: None pooled, then each task.
    resampled: dict[str | None, np.ndarray] = field(repr=False, compare=False)
    draws: Bootstrap = field(repr=False, compare=False)


@dataclass(frozen=True)
class Difference:
    """The first reading less the second, over the same resamples."""

    pooled: PairedEstimate
    per_task: dict[str, PairedEstimate]


def _paired(value: float | None, resampled: np.ndarray) -> PairedEstimate:
    defined = resampled[~np.isnan(resampled)]
    undefined = 1 - len(defined) / len(resampled)
    if len(defined) == 0:
        return PairedEstimate(value, None, None, undefined)
    low, high = np.percentile(defined, [2.5, 97.5])
    return PairedEstimate(value, float(low), float(high), undefined)


def _reading(draws: Bootstrap, value: dict[str | None, float | None], size: dict[str | None, int],
             resampled: dict[str | None, np.ndarray]) -> Reading:  # fmt: skip
    estimates = {key: _paired(value[key], resampled[key]) for key in value}
    per_task = {key: estimate for key, estimate in estimates.items() if key is not None}
    return Reading(estimates[None], per_task, size, resampled, draws)


def _check_items(draws: Bootstrap, scores: Mapping[str, float], successes: Mapping[str, float]) -> None:
    for what, values in (("router scores", scores), ("successes", successes)):
        if set(values) != set(draws.item_ids):
            raise MetricsError(f"the {what} must be of every item the resamples hold, once each, but there are "
                               f"{len(values)} for {len(draws.item_ids)} items, and "
                               f"{len(set(values) ^ set(draws.item_ids))} are in only one set")  # fmt: skip
    if bad := [v for v in successes.values() if not (v == 0 or v == 1)]:
        raise MetricsError(f"every success must be 0 or 1, and {len(bad)} aren't, e.g. {bad[0]}")
    # NaN fails both comparisons, so a router score that isn't a number is refused with the others.
    if bad := [v for v in scores.values() if not (isinstance(v, int | float) and math.isfinite(v))]:
        raise MetricsError(f"every router score must be a number, and {len(bad)} aren't, e.g. {bad[0]}")


def _tasks_in_order(draws: Bootstrap) -> list[str]:
    return list(dict.fromkeys(draws.tasks))


def within_task_concordance(
    draws: Bootstrap, scores: Mapping[str, float], successes: Mapping[str, float], cells: Mapping[str, Hashable]
) -> Reading:
    """A config's within-task concordance: the share of its (success, failure) pairs within one cell whose router
    scores are in the right order, a tie counting half, pooled over the cells' pairs and per task.

    `scores` and `successes` are by item ID; `cells` gives each item's cell, task and fold for cross-fitted router
    scores or the task alone for a single model. Refuses (MetricsError) any of the three not covering the resampled
    items exactly, a success that isn't 1 or 0, a router score that isn't a number, and a cell that spans tasks.
    A task with no pair reads n/a (None) with no interval.
    """
    _check_items(draws, scores, successes)
    if set(cells) != set(draws.item_ids):
        raise MetricsError(f"the cells must be of every item the resamples hold, once each, but there are {len(cells)} "
                           f"for {len(draws.item_ids)} items")  # fmt: skip
    members: dict[Hashable, list[int]] = {}
    for column, item_id in enumerate(draws.item_ids):
        members.setdefault(cells[item_id], []).append(column)
    for cell, columns in members.items():
        if len(spanned := {draws.tasks[c] for c in columns}) != 1:
            raise MetricsError(f"cell {cell!r} must be within one task, but spans {sorted(spanned)}")
    keys: list[str | None] = [None, *_tasks_in_order(draws)]
    won = {key: np.zeros(len(draws.counts)) for key in keys}
    pairs = {key: np.zeros(len(draws.counts)) for key in keys}
    dev_won, dev_pairs = dict.fromkeys(keys, 0.0), dict.fromkeys(keys, 0)
    for columns in members.values():
        task = draws.tasks[columns[0]]
        right = [c for c in columns if successes[draws.item_ids[c]] == 1]
        wrong = [c for c in columns if successes[draws.item_ids[c]] == 0]
        if not (right and wrong):
            continue
        up = np.array([scores[draws.item_ids[c]] for c in right])[:, None]
        down = np.array([scores[draws.item_ids[c]] for c in wrong])[None, :]
        credit = (up > down) + 0.5 * (up == down)  # (successes, failures): 1 in the right order, half for a tie
        cell_won = ((draws.counts[:, right] @ credit) * draws.counts[:, wrong]).sum(axis=1)
        cell_pairs = draws.counts[:, right].sum(axis=1) * draws.counts[:, wrong].sum(axis=1)
        for key in (None, task):
            won[key] += cell_won
            pairs[key] += cell_pairs
            dev_won[key] += float(credit.sum())
            dev_pairs[key] += credit.size
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled = {key: np.where(pairs[key] > 0, won[key] / pairs[key], np.nan) for key in keys}
    value = {key: dev_won[key] / dev_pairs[key] if dev_pairs[key] else None for key in keys}
    return _reading(draws, value, dev_pairs, resampled)


def log_loss(draws: Bootstrap, scores: Mapping[str, float], successes: Mapping[str, float]) -> Reading:
    """A config's log loss: the mean over items of -ln of its router score for the item's outcome,
    pooled and per task, each resample weighing items by how often it draws them.

    Refuses (MetricsError) what `within_task_concordance` refuses of scores and successes, and a router score outside
    0 to 1. A task with no item reads n/a, which a resample can't produce: it always draws every task's groups.
    """
    _check_items(draws, scores, successes)
    if bad := [v for v in scores.values() if not 0 <= v <= 1]:
        raise MetricsError(f"every router score must be from 0 to 1, and {len(bad)} aren't, e.g. {bad[0]}")
    p = np.clip(np.array([scores[i] for i in draws.item_ids]), _EPSILON, 1 - _EPSILON)
    y = np.array([successes[i] for i in draws.item_ids])
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    keys: list[str | None] = [None, *_tasks_in_order(draws)]
    value: dict[str | None, float | None] = {}
    size: dict[str | None, int] = {}
    resampled: dict[str | None, np.ndarray] = {}
    for key in keys:
        in_key = np.ones(len(loss), dtype=bool) if key is None else np.array([task == key for task in draws.tasks])
        counts = draws.counts[:, in_key]
        value[key], size[key] = float(loss[in_key].mean()), int(in_key.sum())
        resampled[key] = (counts @ loss[in_key]) / counts.sum(axis=1)
    return _reading(draws, value, size, resampled)


def difference(first: Reading, second: Reading) -> Difference:
    """The first reading less the second, pooled and per task, taken in each resample so both share its luck.

    Refuses (MetricsError) readings from other resamples or of other tasks. A resample where either reading is n/a
    is n/a. A value is None where either is.
    """
    a, b = first.draws, second.draws
    if not (a is b or (a.item_ids == b.item_ids and a.tasks == b.tasks and np.array_equal(a.counts, b.counts))):
        raise MetricsError("the readings must be over the same resamples, but they are over different ones")
    if first.resampled.keys() != second.resampled.keys():
        raise MetricsError("the readings must be of the same tasks")
    estimates = {}
    for key in first.resampled:
        one, other = (first.pooled, second.pooled) if key is None else (first.per_task[key], second.per_task[key])
        value = None if one.value is None or other.value is None else one.value - other.value
        estimates[key] = _paired(value, first.resampled[key] - second.resampled[key])
    return Difference(estimates[None], {key: estimate for key, estimate in estimates.items() if key is not None})
