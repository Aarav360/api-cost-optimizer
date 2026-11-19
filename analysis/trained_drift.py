"""F's drift check: dev's requests embedded and scored again, before any call on test (PLAN_v4.md Amendment A11, items 17
and 19; Amendment A19, items 2 and 10; issue #77, #83).

A11 item 17 specifies it: dev's requests are embedded and scored again, through the frozen router, and compared with
dev's router-score table. The check

- **always reports** the largest difference in any router score and, at each of F's frozen operating thresholds, the dev
  decisions that flip. A threshold is some dev item's router score (the walk takes thresholds from the table's breakpoints),
  so that item sits on it and `>=` decides it: a flip can come with any drift at all, so a flip count is reported, never asserted zero;
- **above 1e-6** (`REPORT_ABOVE`, A11 item 17's own figure; it is in no other code) also reports dev quality and cost at the
  frozen thresholds with the new router scores beside the old, and **goes on**: no walk is run again and no threshold moves;
- **above the drift limit** (A11 item 19: 100 times the overall gap measured at freeze, read by `drift_limit()`) **refuses**
  (`DriftRefusal`): a gap that large means something broke, not rounding, and the way on is a new amendment.

Dev's router-score table is the one dev had: the router scored from the dev embedding cache that the fit wrote, through the
same frozen heads (`real_dev_side`), since the table `analysis/trained_table.py` builds is rebuilt live each time and would
compare the machine with itself. The re-scoring goes through `score_with` with the dev fold file's folds, as dev's table
does, on a router instance of its own, so the counts of test's route (`default_router()`) stay test's alone. Test is not
touched here; the step runs before the calls on test.

Both sides are a `DevSide`, so a rehearsal gives its own (`analysis/backends.py`'s `TrainedStandIn`): the same function runs
on a made-up dev side and a fake embedder whose vectors move slightly, then a lot.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from analysis.backends import stand_in_of
from analysis.drift import DriftRefusal
from analysis.drift_rule import Flip, decision_flips
from analysis.final_run import Run, StepResult
from analysis.matrix import Matrix
from analysis.operating_eval import FrozenPoint, evaluate, frozen_points, router_traces
from analysis.router_table import LOCAL_OVERHEAD, ItemRouterScores, RouterScoreTable
from analysis.simulator import always, load_mixes, simulate, totals
from routers.ladder import FRONTIER, SCORED
from routers.request import Request

CHECK = "drift_trained_router"
REPORT_ABOVE = 1e-6  # A11 item 17: above this the check also reports dev's quality and cost at the frozen thresholds
SECTION = "trained"  # F's section of the operating-points file


@dataclass(frozen=True)
class DevSide:
    """Dev as the check reads it: its requests (in the table's order), each item's fold, the table dev had, and dev's
    results matrix (read only if the check has to report quality and cost)."""

    requests: Mapping[str, Request]
    fold_of: Mapping[str, int]
    table: RouterScoreTable
    matrix: Callable[[], Matrix]


@dataclass(frozen=True)
class DriftReport:
    largest: float  # the largest absolute difference in any router score, over every dev item and config
    limit: float
    flips: tuple[Flip, ...]  # one per distinct frozen threshold, highest first
    # Set only above `REPORT_ABOVE`: {mix: {target: {"old": {quality, cost_per_1k}, "new": {...}}}} on dev
    outcomes: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]] | None = field(default=None)

    @property
    def over_report(self) -> bool:
        return self.largest > REPORT_ABOVE

    def record(self) -> dict[str, Any]:
        """What the record keeps (A19 item 10): F's largest difference and the decisions that flip at each threshold."""
        out: dict[str, Any] = {
            "largest_difference": self.largest,
            "drift_limit": self.limit,
            "report_above": REPORT_ABOVE,
            "flips": [f.record() for f in self.flips],
        }
        if self.outcomes is not None:
            out["dev_outcomes"] = {mix: {t: {k: dict(v) for k, v in row.items()} for t, row in by.items()}
                                   for mix, by in self.outcomes.items()}  # fmt: skip
        return out


def largest_difference(dev_table: RouterScoreTable, rescored: RouterScoreTable) -> float:
    """The largest absolute difference between two tables' router scores. Tables of other items are refused."""
    if [e.item_id for e in dev_table.items] != [e.item_id for e in rescored.items]:
        raise ValueError("the two tables are not over the same dev items, in the same order")
    return max(
        abs(new.router_scores[config] - old.router_scores[config])
        for old, new in zip(dev_table.items, rescored.items, strict=True)
        for config in SCORED
    )


def flips_at(
    dev_table: RouterScoreTable, rescored: RouterScoreTable, points: tuple[FrozenPoint, ...]
) -> tuple[Flip, ...]:
    """At each distinct frozen threshold, the dev items whose decision (`choose`, `>=`) differs between the two tables."""
    labels: dict[float, list[str]] = {}
    for point in points:
        if point.threshold is not None:
            labels.setdefault(point.threshold, []).append(f"{point.mix} {point.target}")
    old = {e.item_id: e.router_scores for e in dev_table.items}
    new = {e.item_id: e.router_scores for e in rescored.items}
    return decision_flips(old, new, labels)


def dev_outcomes(
    matrix: Matrix, table: RouterScoreTable, points: tuple[FrozenPoint, ...]
) -> dict[str, dict[str, dict]]:
    """Dev's quality and cost per 1,000 requests at each frozen point, with this table's router scores: the frozen
    thresholds evaluated, never walked (`analysis/operating_eval.py`). A point that fell back is A's."""
    mixes = load_mixes()
    frontier = simulate(matrix, always(matrix, FRONTIER))
    found = evaluate(matrix, frontier, points, router_traces(table.traces))
    rows: dict[str, dict[str, dict]] = {}
    for point, outcomes in found.items():
        total = totals(outcomes, mixes[point.mix])
        rows.setdefault(point.mix, {})[point.target] = {"quality": total.quality, "cost_per_1k": total.cost_per_1k}
    return rows


def drift_report(
    dev: DevSide, rescored: RouterScoreTable, points: tuple[FrozenPoint, ...], limit: float
) -> DriftReport:
    """The check's rule on dev's table and the re-scored one. Raises `DriftRefusal` above `limit`."""
    largest = largest_difference(dev.table, rescored)
    if largest > limit:
        raise DriftRefusal(CHECK, f"F's dev router scores moved by up to {largest:.3g} on re-embedding, past the drift "
                                  f"limit {limit:.3g} (A11 item 19); something broke, and the way on is a new amendment")  # fmt: skip
    flips = flips_at(dev.table, rescored, points)
    outcomes = None
    if largest > REPORT_ABOVE:
        matrix = dev.matrix()
        old, new = dev_outcomes(matrix, dev.table, points), dev_outcomes(matrix, rescored, points)
        outcomes = {mix: {t: {"old": old[mix][t], "new": new[mix][t]} for t in by} for mix, by in new.items()}
    return DriftReport(largest, limit, flips, outcomes)


def rescore(router: Any, dev: DevSide) -> RouterScoreTable:
    """Dev's router scores again through the router's own `score_with` and dev's fold of each item, as dev's table was built."""
    items = [
        ItemRouterScores(item_id, router.score_with(dev.fold_of[item_id], request), LOCAL_OVERHEAD)
        for item_id, request in dev.requests.items()
    ]
    return RouterScoreTable.build(items, list(dev.requests))


# The real dev side, read from dev's stored records


def real_dev_side() -> DevSide:
    """Dev's requests and folds from the reader, the table dev had (the cached embeddings through the frozen heads, never
    embedding), and the dev results matrix to be read only when needed."""
    from analysis import fit_trained_router as fit
    from analysis import matrix as results_matrix
    from analysis.benchmark import read_rows
    from analysis.folds import read_folds
    from analysis.no_backend import NoBackend
    from routers import trained
    from routers.request import from_row
    from routers.trained import TrainedRouter

    rows = read_rows(split="dev")
    cache = fit.read_cache("dev")
    fold_of = read_folds("dev")
    cached = TrainedRouter(trained.read_settings(), trained.read_fold_weights(), backend=NoBackend())
    requests = {row["item_id"]: from_row(row) for row in rows}
    items = [
        ItemRouterScores(
            row["item_id"],
            cached.score_features(fold_of[row["item_id"]], cached.features(embedding, requests[row["item_id"]])),
            LOCAL_OVERHEAD,
        )
        for row, embedding in zip(rows, cache.embeddings, strict=True)
    ]
    dev_table = RouterScoreTable.build(items, list(requests))
    return DevSide(requests, fold_of, dev_table, lambda: results_matrix.load().split("dev"))


def drift_step(run: Run) -> StepResult:
    """A19 item 2.3's F drift check: dev's requests embedded again with the pinned model (or the world's stand-in), compared with
    dev's table, under A11 item 17's limits. It says dev's own numbers only, and stops the run on a refusal."""
    from analysis.measure_trained_router import drift_limit
    from routers import trained
    from routers.trained import TrainedRouter

    stand_in = stand_in_of(run.world.backends, "trained")
    dev = stand_in.dev() if stand_in is not None and stand_in.dev is not None else real_dev_side()
    backend = stand_in.embedder if stand_in is not None and stand_in.embedder is not None else None
    # A router of its own: a stand-in's backend, or the real model the route to test will use too, but counts apart.
    router = TrainedRouter(
        trained.read_settings(), trained.read_fold_weights(),
        backend=backend if backend is not None else trained.default_router().backend,
    )  # fmt: skip
    rescored = rescore(router, dev)
    report = drift_report(dev, rescored, frozen_points(SECTION), drift_limit())
    flipped = {f.value: len(f.items) for f in report.flips}
    run.say(f"{CHECK}: {len(dev.requests)} dev requests embedded again; largest difference {report.largest:.3g} "
            f"(limit {report.limit:.3g}); dev decisions that flip at F's thresholds {flipped}")  # fmt: skip
    if report.outcomes is not None:
        run.say(
            f"{CHECK}: above {REPORT_ABOVE:g}, so dev quality and cost at the frozen thresholds are in the record; going on"
        )
    return StepResult(detail=report.record())
