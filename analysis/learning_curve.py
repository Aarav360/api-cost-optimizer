"""The learning curve's numbers: would more labels help F? (PLAN_v4.md §8 3b; Amendment A11, items 12 and 15)

Run from the project root:  uv run python -m analysis.learning_curve

Within each fold, the fold model is refit on nested subsamples of 25, 50, 75 and 100% of its training rows (all of
train and the dev rows outside the fold), with the same settings and C, through the fit script's own code
(`analysis.fit_trained_router`, frozen). Each refit scores its held-out fold; pooled over the five folds those are the
size's out-of-fold router scores. Per config and size: out-of-fold log loss and within-task concordance (pairs within
a task and a fold), with paired intervals from the existing 2,000 resamples (`analysis.router_metrics`). F-train
stands beside the curve as a separate point, and the headline is the paired difference from 75% to 100%.

- **The subsamples** draw whole source groups, within each stratum of task and split (train or dev), so the mix of
  train and dev rows stays the same along the curve. A group's place is the sha256 of "seed:curve:task:split:group",
  as the fold builder orders groups, so there is no random generator to depend on. A stratum at size p keeps the
  prefix of its groups, in that order, whose item count is nearest p of the stratum's items (at least one group), so a
  smaller subsample is always inside the next. The same order serves every fold, which is why a fold's subsample is
  its own prefix of the same order.
- **No walk at any size** (item 12: each is a fresh threshold selection, whose savings move in steps that are mostly
  noise), and nothing here is written to the operating-points file. Only train and dev are read, never a test item.
- **The 100% point is F:** the same rows in the same order through the same `fit_head`, scored through the router's
  own `score_features`, so it reproduces F's out-of-fold router scores (`check_learning_curve` compares them).
"""

import hashlib
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from analysis import fit_trained_router as fit
from analysis import matrix as results_matrix
from analysis import trained_table
from analysis.benchmark import read_rows
from analysis.bootstrap import Bootstrap, resample
from analysis.folds import N_FOLDS, read_folds
from analysis.router_metrics import (
    Difference,
    PairedEstimate,
    Reading,
    difference,
    log_loss,
    within_task_concordance,
)
from routers import trained
from routers.ladder import SCORED
from routers.request import from_row
from routers.trained import Settings, TrainedRouter

SEED = 2026  # the bootstrap's and the fold builder's seed (A11 item 8)
SIZES = (0.25, 0.5, 0.75, 1.0)  # shares of a fold model's training rows (item 12)


class CurveError(ValueError):
    """Rows, sizes or router scores the learning curve refuses. Nothing is computed."""


class Unit(NamedTuple):
    """One training row as the subsample sees it: its task, its split (train or dev) and its source group."""

    task: str
    split: str
    group: str


def _place(seed: int, unit: Unit) -> str:
    return hashlib.sha256(f"{seed}:curve:{unit.task}:{unit.split}:{unit.group}".encode()).hexdigest()


def nested_subsamples(
    units: Sequence[Unit], sizes: Sequence[float] = SIZES, seed: int = SEED
) -> dict[float, np.ndarray]:
    """For each size, a boolean mask over `units` (the rows kept), each mask inside the next size's.

    Per (task, split) stratum, groups are put in hash order and the stratum keeps the prefix whose item count is
    nearest `size` of the stratum's items, never fewer than one group. Refuses (CurveError) no rows, sizes that
    don't increase to 100%, and a source group in more than one stratum.
    """
    if not units:
        raise CurveError("no rows to subsample")
    if list(sizes) != sorted(set(sizes)) or not all(0 < s <= 1 for s in sizes):
        raise CurveError(f"the sizes must increase, each above 0 and at most 100%: {tuple(sizes)}")
    if sizes[-1] != 1.0:
        raise CurveError(f"the last size must be 100% of the rows, not {sizes[-1]:.0%}")
    strata: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    home: dict[str, tuple[str, str]] = {}
    for i, unit in enumerate(units):
        stratum = (unit.task, unit.split)
        if home.setdefault(unit.group, stratum) != stratum:
            raise CurveError(f"source group {unit.group} is in two strata, {home[unit.group]} and {stratum}")
        strata[stratum][unit.group].append(i)
    masks = {size: np.zeros(len(units), dtype=bool) for size in sizes}
    for by_group in strata.values():
        ordered = sorted(by_group, key=lambda g: _place(seed, units[by_group[g][0]]))
        cumulative = np.cumsum([len(by_group[g]) for g in ordered])
        for size in sizes:
            groups = int(np.argmin(np.abs(cumulative - size * cumulative[-1]))) + 1
            for g in ordered[:groups]:
                masks[size][by_group[g]] = True
    return masks


# ---------------------------------------------------------------------------------------------------------------
# The refits (item 12): the fit script's own `fit_head`, on each fold's subsample


class _NoEmbedding:
    """A backend that can't embed: the refits score feature rows, never text."""

    def encode(self, text: str) -> list[int]:
        raise CurveError("the learning curve scores feature rows; it never embeds")

    def cls_vector(self, ids: list[int]) -> np.ndarray:
        raise CurveError("the learning curve scores feature rows; it never embeds")


@dataclass(frozen=True)
class Refits:
    """The out-of-fold router scores at each size, per config and in dev's row order, and the rows each size trained on."""

    scores: dict[float, dict[str, np.ndarray]]
    rows: dict[float, tuple[int, ...]]  # per size, each fold model's number of training rows


def refit_out_of_fold(
    settings: Settings,
    train: tuple[np.ndarray, Mapping[str, np.ndarray], Sequence[Unit]],
    dev: tuple[np.ndarray, Mapping[str, np.ndarray], Sequence[Unit]],
    dev_folds: np.ndarray,
    sizes: Sequence[float] = SIZES,
    seed: int = SEED,
) -> Refits:
    """Refit every fold model at every size and score its held-out dev fold.

    `train` and `dev` hold each split's feature rows (the router's `features`), its run-0 successes per config and
    each row's `Unit`. Fold model k's rows are all of train and the dev rows outside fold k, stacked in that order
    as `fit_fold_models` stacks them, then thinned to the size's subsample; each config's head is fitted at its own C
    by `fit_head` and scored through `TrainedRouter.score_features`, as F's table is.
    """
    train_x, train_y, train_units = train
    dev_x, dev_y, dev_units = dev
    scores = {size: {config: np.zeros(len(dev_x)) for config in SCORED} for size in sizes}
    rows: dict[float, list[int]] = {size: [] for size in sizes}
    for k in range(N_FOLDS):
        held = dev_folds == k
        x = np.vstack([train_x, dev_x[~held]])
        y = {config: np.concatenate([train_y[config], dev_y[config][~held]]) for config in SCORED}
        units = [*train_units, *(u for u, h in zip(dev_units, held, strict=True) if not h)]
        masks = nested_subsamples(units, sizes, seed)
        for size in sizes:
            keep = masks[size]
            heads = {config: fit.fit_head(x[keep], y[config][keep], settings.c[config]) for config in SCORED}
            router = TrainedRouter(settings, (heads,) * trained.FOLDS, backend=_NoEmbedding())
            for i in np.flatnonzero(held):
                # A copy, as the router's own `features` gives a fresh array: a dot product's last bit can depend on
                # the array's alignment, and a row of a 769-wide matrix isn't aligned like one (15 of 744 differed).
                for config, value in router.score_features(0, np.array(dev_x[i])).items():
                    scores[size][config][i] = value
            rows[size].append(int(keep.sum()))
    return Refits(scores, {size: tuple(n) for size, n in rows.items()})


# ---------------------------------------------------------------------------------------------------------------
# The numbers: log loss and within-task concordance per config and size, F-train's point, the headline


@dataclass(frozen=True)
class Point:
    """One config at one size (or F-train): the two readings, each pooled and per task with paired intervals."""

    labels: float  # training rows, averaged over the fold models (F-train: its train rows)
    log_loss: Reading
    concordance: Reading


@dataclass(frozen=True)
class Step:
    """The paired difference from one size to the next, over the same resamples: `larger` less `smaller`."""

    log_loss: Difference  # negative means more labels lowered the loss
    concordance: Difference  # positive means more labels ordered more pairs correctly


@dataclass(frozen=True)
class LearningCurve:
    sizes: tuple[float, ...]
    points: dict[str, dict[float, Point]]  # config, then size
    f_train: dict[str, Point]  # a separate point, on dev's items, never joined to the curve
    headline: dict[str, Step]  # per config, 100% less 75% (the last size less the one before)


def _point(draws: Bootstrap, labels: float, scores: Mapping[str, float], success: Mapping[str, float],
           cells: Mapping[str, Hashable]) -> Point:  # fmt: skip
    return Point(labels, log_loss(draws, scores, success), within_task_concordance(draws, scores, success, cells))


def measure(
    draws: Bootstrap,
    refits: Refits,
    item_ids: Sequence[str],
    success: Mapping[str, Mapping[str, float]],
    cells: Mapping[str, Hashable],
    f_train: tuple[int, Mapping[str, Mapping[str, float]], Mapping[str, Hashable]],
) -> LearningCurve:
    """The curve's readings on dev's items, all over `draws`.

    `refits.scores` are in `item_ids` order; `success` is each config's run-0 success by item; `cells` is each item's
    (task, fold) for the curve; `f_train` is its number of training rows, its router scores by config and item, and
    its cells (the task alone: one model).
    """
    sizes = tuple(refits.rows)
    points = {
        config: {
            size: _point(draws, float(np.mean(refits.rows[size])), dict(zip(item_ids, refits.scores[size][config],
                                                                         strict=True)), success[config], cells)
            for size in sizes
        }
        for config in SCORED
    }  # fmt: skip
    rows, scores, f_train_cells = f_train
    alone = {config: _point(draws, float(rows), scores[config], success[config], f_train_cells) for config in SCORED}
    headline = {
        config: Step(
            difference(points[config][sizes[-1]].log_loss, points[config][sizes[-2]].log_loss),
            difference(points[config][sizes[-1]].concordance, points[config][sizes[-2]].concordance),
        )
        for config in SCORED
    }
    return LearningCurve(sizes, points, alone, headline)


# ---------------------------------------------------------------------------------------------------------------
# On the benchmark's train and dev rows


def _split_inputs(
    settings: Settings, matrix: results_matrix.Matrix, split: str
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], list[Unit]]:
    """A split's item IDs in the reader's order, feature rows from the embedding cache, run-0 successes, units."""
    rows = read_rows(split=split)
    ids = [row["item_id"] for row in rows]
    x = fit.feature_matrix(settings, fit.read_cache(split).embeddings, [from_row(row) for row in rows])
    cells = matrix.split(split).cells
    y = {config: np.array([cells[(i, config)].success for i in ids]) for config in SCORED}
    return ids, x, y, [Unit(row["source"], split, row["source_group"]) for row in rows]


def build(sizes: Sequence[float] = SIZES) -> tuple[LearningCurve, Refits]:
    """Refit every fold model at every size on train and dev, and read the curve on dev (no walk, no file written).

    Needs the embedding caches and the dev results matrix. Returns the curve and the refits behind it.
    """
    settings = trained.read_settings()
    matrix = results_matrix.load()
    _, train_x, train_y, train_units = _split_inputs(settings, matrix, "train")
    dev_ids, dev_x, dev_y, dev_units = _split_inputs(settings, matrix, "dev")
    dev_fold_of = read_folds("dev")
    dev_folds = np.array([dev_fold_of[i] for i in dev_ids])
    refits = refit_out_of_fold(settings, (train_x, train_y, train_units), (dev_x, dev_y, dev_units), dev_folds, sizes)
    dev = matrix.split("dev")
    success = {config: {i: dev.cells[(i, config)].success for i in dev_ids} for config in SCORED}
    task = {i: dev.items[i].task for i in dev_ids}
    f_train = trained_table.f_train_table()
    f_train_scores = {config: {e.item_id: e.router_scores[config] for e in f_train.items} for config in SCORED}
    curve = measure(
        resample(dev), refits, dev_ids, success, {i: (task[i], dev_fold_of[i]) for i in dev_ids},
        (len(train_units), f_train_scores, task),
    )  # fmt: skip
    return curve, refits


def learning_curve() -> LearningCurve:
    """The learning curve's numbers on dev, for the chart script."""
    return build()[0]


def _reading_data(reading: Reading | Difference) -> dict:
    def estimate(e: PairedEstimate) -> dict:
        return {"value": e.value, "low": e.low, "high": e.high, "undefined": e.undefined}

    out = {"pooled": estimate(reading.pooled), "per_task": {t: estimate(e) for t, e in reading.per_task.items()}}
    if isinstance(reading, Reading):
        out["size"] = {"pooled" if k is None else k: n for k, n in reading.size.items()}  # pairs or items
    return out


def as_data(curve: LearningCurve) -> dict:
    """The curve as plain JSON-able data: `points[config][size]`, `f_train[config]` and `headline[config]`."""

    def point(p: Point) -> dict:
        return {"labels": p.labels, "log_loss": _reading_data(p.log_loss), "concordance": _reading_data(p.concordance)}

    return {
        "sizes": list(curve.sizes),
        "points": {c: {f"{s:g}": point(p) for s, p in by_size.items()} for c, by_size in curve.points.items()},
        "f_train": {c: point(p) for c, p in curve.f_train.items()},
        "headline": {
            c: {"log_loss": _reading_data(s.log_loss), "concordance": _reading_data(s.concordance)}
            for c, s in curve.headline.items()
        },
    }


def _fmt(e: PairedEstimate, places: int = 4, signed: bool = False) -> str:
    if e.value is None or e.low is None or e.high is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{e.value:{sign}.{places}f} ({e.low:{sign}.{places}f} to {e.high:{sign}.{places}f})"


def main() -> None:
    curve, _ = build()
    print("Learning curve on dev: out-of-fold log loss and within-task concordance (pairs within a task and fold),")
    print("pooled, with paired 95% intervals. No walk at any size.\n")
    for config in SCORED:
        print(config)
        for size, p in curve.points[config].items():
            print(f"  {size:4.0%} ({p.labels:6.1f} labels)  log loss {_fmt(p.log_loss.pooled)}   "
                  f"concordance {_fmt(p.concordance.pooled)}")  # fmt: skip
        p = curve.f_train[config]
        print(f"  F-train ({p.labels:.0f} labels)  log loss {_fmt(p.log_loss.pooled)}   "
              f"concordance {_fmt(p.concordance.pooled)}")  # fmt: skip
        step = curve.headline[config]
        print(f"  {curve.sizes[-1]:.0%} less {curve.sizes[-2]:.0%}:  log loss {_fmt(step.log_loss.pooled, 4, True)}   "
              f"concordance {_fmt(step.concordance.pooled, 4, True)}")  # fmt: skip
        for task in step.log_loss.per_task:
            print(f"      {task:12} log loss {_fmt(step.log_loss.per_task[task], 4, True)}   "
                  f"concordance {_fmt(step.concordance.per_task[task], 4, True)}")  # fmt: skip


if __name__ == "__main__":
    main()
