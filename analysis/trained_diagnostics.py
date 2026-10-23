"""Stage 5's diagnostics on dev: F-train and the task-rates diagnostic walked, within-task concordance, log loss and
the same-target comparisons (PLAN_v4.md §8 3b; Amendment A11, items 12, 13 and 15).

Run from the project root:  uv run python -m analysis.trained_diagnostics

Declared before any F number existed and run after F's walk. None selects a setting or threshold, gets test operating
points or enters the operating-points file: nothing here writes to it, and the walks of F-train and of the task-rates
diagnostic live in memory only. Stage 5 · 9's chart script calls these functions for table v2 and A11 item 21;
`main` prints what they return.

- **The task-rates diagnostic** (`task_rates_table`): a dev item's router score for a config is its task's success rate
  over exactly the labels the item's fold model saw, train plus the other four dev folds, so each fold has its own
  rates. It is walked with A9 item 9's rule like any router's (`analysis.walk`), quantile steps included.
- **Within-task concordance** (`concordance_readings`): cells are task and fold for F and the task-rates diagnostic,
  the task alone for F-train, which is one model. The task-rates diagnostic's is exactly 0.5 by construction.
- **Log loss** (`log_loss_readings`): the task-rates diagnostic's out-of-fold log loss per config is the learning curve's
  dashed reference line (Stage 5 · 9).
- **Same-target comparisons** (`same_target_cells`): F against E, F-train against E and F against F-train at every mix
  and target, with `analysis.same_target`'s fallback rules.
- **Per-task warnings** (`task_warning`): A7 item 2's, on IFEval and MBPP+.
"""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

from analysis import matrix as results_matrix
from analysis import rules_table, trained_table
from analysis.bootstrap import Bootstrap, Estimate, resample
from analysis.folds import read_folds
from analysis.matrix import Matrix
from analysis.router_metrics import Difference, PairedEstimate, Reading, difference, log_loss, within_task_concordance
from analysis.router_table import ItemRouterScores, RouterScoreTable
from analysis.rules_walk import MIXES, dev_walks
from analysis.same_target import SameTarget, compare_same_target
from analysis.simulator import load_mixes
from analysis.walk import ALL_STEPS_UP_TO, Walk, item_breakpoints, step_thresholds
from routers.ladder import SCORED

OVERHEAD = 0.0  # a table of rates is not a router run: an explicit $0, as F's and E's
E, F, F_TRAIN, TASK_RATES = "E", "F", "F-train", "task rates"  # the routers' names in these readings
DIAGNOSTICS = (F, F_TRAIN, TASK_RATES)  # the routers with concordance and log loss readings, in print order
# A11 item 13's comparisons, first less second.
COMPARISONS = ((F, E), (F_TRAIN, E), (F, F_TRAIN))
# Fewer items than the "few hundred" §5 asks for before a per-task claim (A7 item 2); the chart script's own list.
WARNED = ("ifeval", "mbppplus")


class DiagnosticsError(ValueError):
    """Inputs a diagnostic refuses: matrices of the wrong split, or dev items without a fold. Nothing is computed."""


def task_warning(task: str, dev_items: int) -> str | None:
    """A7 item 2's warning for a small task's per-task results, or None for the tasks that carry none."""
    if task not in WARNED:
        return None
    return f"Only {dev_items} dev items, below the plan's minimum of a few hundred: indicative"


def task_rates_table(train: Matrix, dev: Matrix, fold_of: Mapping[str, int]) -> RouterScoreTable:
    """The task-rates diagnostic's dev table: each item's router score per config is its task's success rate over
    train and the dev items outside the item's fold, the labels the item's fold model saw (item 12).

    Run 0 labels, as the fit uses. Refuses (DiagnosticsError) a train matrix with a non-train item, a dev matrix with
    a non-dev item, or a dev item without a fold. A task with no label in a fold's training rows can't happen here
    (train holds every task) and would be refused by the divisor.
    """
    if bad := [i for i, item in train.items.items() if item.split != "train"]:
        raise DiagnosticsError(f"the train matrix must hold train items only, but {bad[0]} is not one")
    if bad := [i for i, item in dev.items.items() if item.split != "dev"]:
        raise DiagnosticsError(f"the dev matrix must hold dev items only, but {bad[0]} is not a dev item")
    if missing := sorted(set(dev.items) - set(fold_of)):
        raise DiagnosticsError(f"{len(missing)} dev items have no fold, e.g. {missing[0]}")
    # [successes, items] per task and config over train, and per fold, task and config over that dev fold; a fold's
    # rates add every dev fold but its own to train's.
    totals: dict[tuple[str, str], list[float]] = {}
    by_fold: dict[tuple[int, str, str], list[float]] = {}
    for item_id, item in train.items.items():
        for config in SCORED:
            cell = totals.setdefault((item.task, config), [0.0, 0])
            cell[0] += train.cells[(item_id, config)].success
            cell[1] += 1
    for item_id, item in dev.items.items():
        for config in SCORED:
            cell = by_fold.setdefault((fold_of[item_id], item.task, config), [0.0, 0])
            cell[0] += dev.cells[(item_id, config)].success
            cell[1] += 1
    folds = sorted({fold_of[i] for i in dev.items})

    def rate(fold: int, task: str, config: str) -> float:
        total, count = totals[(task, config)]
        for other in folds:
            if other != fold and (other, task, config) in by_fold:
                total, count = total + by_fold[(other, task, config)][0], count + by_fold[(other, task, config)][1]
        return total / count

    entries = [
        ItemRouterScores(item_id, {c: rate(fold_of[item_id], item.task, c) for c in SCORED}, OVERHEAD)
        for item_id, item in dev.items.items()
    ]
    return RouterScoreTable.build(entries, list(dev.items))


def cells_by_task_and_fold(dev: Matrix, fold_of: Mapping[str, int]) -> dict[str, tuple[str, int]]:
    """Concordance cells for out-of-fold router scores: an item's task and its dev fold (item 12)."""
    return {item_id: (item.task, fold_of[item_id]) for item_id, item in dev.items.items()}


def cells_by_task(dev: Matrix) -> dict[str, str]:
    """Concordance cells for a single model (F-train): the task alone."""
    return {item_id: item.task for item_id, item in dev.items.items()}


def successes_of(dev: Matrix, config: str) -> dict[str, float]:
    """A config's run-0 success per dev item."""
    return {item_id: dev.cells[(item_id, config)].success for item_id in dev.items}


def scores_of(table: RouterScoreTable, config: str) -> dict[str, float]:
    return {entry.item_id: entry.router_scores[config] for entry in table.items}


def concordance_readings(
    dev: Matrix, draws: Bootstrap, table: RouterScoreTable, cells: Mapping[str, Hashable]
) -> dict[str, Reading]:
    """Within-task concordance per config, pooled and per task, over `cells` (`cells_by_task_and_fold` or
    `cells_by_task`)."""
    return {c: within_task_concordance(draws, scores_of(table, c), successes_of(dev, c), cells) for c in SCORED}


def log_loss_readings(dev: Matrix, draws: Bootstrap, table: RouterScoreTable) -> dict[str, Reading]:
    """Log loss per config, pooled and per task. Of the task-rates table it is the diagnostic's out-of-fold log loss."""
    return {c: log_loss(draws, scores_of(table, c), successes_of(dev, c)) for c in SCORED}


def gaps(first: Mapping[str, Reading], second: Mapping[str, Reading]) -> dict[str, Difference]:
    """The first router's reading less the second's, per config, over the same resamples."""
    return {c: difference(first[c], second[c]) for c in SCORED}


def same_target_cells(
    draws: Bootstrap, walks: Mapping[str, Mapping[str, Mapping[float, Walk]]]
) -> dict[tuple[str, str], dict[str, dict[float, SameTarget]]]:
    """Each of item 13's comparisons at every mix and target: `walks` is by router name (E, F, F-train), mix, share."""
    mixes = load_mixes()
    return {
        (a, b): {
            mix: {
                share: compare_same_target(draws, mixes[mix], walks[a][mix][share], walks[b][mix][share])
                for share in walks[a][mix]
            }  # fmt: skip
            for mix in MIXES
        }
        for a, b in COMPARISONS
    }


def breakpoints(table: RouterScoreTable) -> tuple[int, int]:
    """A table's number of distinct breakpoints and of steps, for the note that the quantile steps apply."""
    return len({v for e in table.items for v in item_breakpoints(e.router_scores)}), len(step_thresholds(table))


@dataclass(frozen=True)
class Diagnostics:
    """Every diagnostic number, computed once, in memory. Nothing here is written to any file."""

    dev: Matrix
    draws: Bootstrap
    fold_of: Mapping[str, int]
    tables: dict[str, RouterScoreTable]  # E, F, F-train, task rates
    walks: dict[str, dict[str, dict[float, Walk]]]  # by router, mix, share
    concordance: dict[str, dict[str, Reading]]  # DIAGNOSTICS, then config
    log_loss: dict[str, dict[str, Reading]]
    concordance_gaps: dict[tuple[str, str], dict[str, Difference]]  # F less task rates, F less F-train
    log_loss_gaps: dict[tuple[str, str], dict[str, Difference]]
    same_target: dict[tuple[str, str], dict[str, dict[float, SameTarget]]]


GAPS = ((F, TASK_RATES), (F, F_TRAIN))  # F against what it knows beyond the task, and against what dev labels buy


def compute() -> Diagnostics:
    """All of Stage 5 · 7's numbers, from the dev matrix, the dev caches and the committed files."""
    full = results_matrix.load()
    dev, train = full.split("dev"), full.split("train")
    fold_of = read_folds("dev")
    draws = resample(dev)
    tables = {
        E: rules_table.dev_table(),
        F: trained_table.dev_table(),
        F_TRAIN: trained_table.f_train_table(),
        TASK_RATES: task_rates_table(train, dev, fold_of),
    }
    walks = {name: dev_walks(dev, table, draws) for name, table in tables.items()}
    cells: dict[str, Mapping[str, Hashable]] = {F: cells_by_task_and_fold(dev, fold_of), F_TRAIN: cells_by_task(dev),
             TASK_RATES: cells_by_task_and_fold(dev, fold_of)}  # fmt: skip
    concordance = {name: concordance_readings(dev, draws, tables[name], cells[name]) for name in DIAGNOSTICS}
    losses = {name: log_loss_readings(dev, draws, tables[name]) for name in DIAGNOSTICS}
    return Diagnostics(
        dev, draws, fold_of, tables, walks, concordance, losses,
        {pair: gaps(concordance[pair[0]], concordance[pair[1]]) for pair in GAPS},
        {pair: gaps(losses[pair[0]], losses[pair[1]]) for pair in GAPS},
        same_target_cells(draws, walks),
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# JSON-ready data, for table v2 and A11 item 21


def _estimate_data(estimate: PairedEstimate | Estimate | None) -> dict[str, float | None] | None:
    if estimate is None:
        return None
    return {"value": estimate.value, "low": estimate.low, "high": estimate.high,
            "undefined": getattr(estimate, "undefined", 0.0)}  # fmt: skip


def _reading_data(reading: Reading | Difference, items: Mapping[str, int]) -> dict[str, object]:
    size = getattr(reading, "size", None)
    return {
        "pooled": _estimate_data(reading.pooled),
        "per_task": {t: {**(_estimate_data(e) or {}), "warning": task_warning(t, items[t])}
                     for t, e in reading.per_task.items()},
        "size": None if size is None else {"pooled" if k is None else k: n for k, n in size.items()},
    }  # fmt: skip


def to_data(result: Diagnostics) -> dict[str, object]:
    """The diagnostics' numbers as plain JSON-ready data, keyed by router, config, mix and target.

    Per-task entries carry A7 item 2's warning (null where there is none). A same-target cell holds the cost and
    quality differences (first less second) as value, low, high, or null where both fall back to A, and `reading`.
    """
    items = {t: sum(1 for i in result.dev.items.values() if i.task == t) for t in dict.fromkeys(
        i.task for i in result.dev.items.values())}  # fmt: skip
    return {
        "within_task_concordance": {n: {c: _reading_data(r, items) for c, r in by.items()}
                                    for n, by in result.concordance.items()},
        "log_loss": {n: {c: _reading_data(r, items) for c, r in by.items()} for n, by in result.log_loss.items()},
        "concordance_differences": {f"{a} less {b}": {c: _reading_data(d, items) for c, d in by.items()}
                                    for (a, b), by in result.concordance_gaps.items()},
        "log_loss_differences": {f"{a} less {b}": {c: _reading_data(d, items) for c, d in by.items()}
                                 for (a, b), by in result.log_loss_gaps.items()},
        "same_target": {
            f"{a} less {b}": {
                mix: {f"{share:.0%}": {"reading": cell.reading, "first_fell_back": cell.first_fell_back,
                                       "second_fell_back": cell.second_fell_back,
                                       "cost_per_1k": _estimate_data(cell.cost), "quality": _estimate_data(cell.quality)}
                      for share, cell in by_share.items()}
                for mix, by_share in by_mix.items()}
            for (a, b), by_mix in result.same_target.items()},
    }  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# Printing


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _interval(estimate) -> str:  # type: ignore[no-untyped-def]
    if estimate.low is None:
        return f"{_pct(estimate.value)} (no interval)"
    undefined = f", {estimate.undefined:.0%} of resamples n/a" if estimate.undefined else ""
    return f"{_pct(estimate.value)} [{estimate.low:.3f}, {estimate.high:.3f}]{undefined}"


def print_readings(result: Diagnostics) -> None:
    items = {task: sum(1 for i in result.dev.items.values() if i.task == task) for task in dict.fromkeys(
        i.task for i in result.dev.items.values())}  # fmt: skip
    for title, readings in (("Within-task concordance (95% paired intervals)", result.concordance),
                            ("Log loss, out of fold where cross-fitted (95% paired intervals)", result.log_loss)):  # fmt: skip
        print(f"\n{title}")
        for name in DIAGNOSTICS:
            for config in SCORED:
                reading = readings[name][config]
                size = reading.size[None]
                print(f"  {name:10} {config:9} pooled {_interval(reading.pooled)} ({size} {'pairs' if 'oncordance' in title else 'items'})")  # fmt: skip
                for task, estimate in reading.per_task.items():
                    warn = task_warning(task, items[task])
                    print(f"  {'':10} {'':9}   {task:9} {_interval(estimate)}" + (f"   [{warn}]" if warn else ""))
    for label, differences in (("concordance", result.concordance_gaps), ("log loss", result.log_loss_gaps)):
        for (a, b), by_config in differences.items():
            print(f"\nPaired difference in {label}, {a} less {b}")
            for config, diff in by_config.items():
                print(f"  {config:9} pooled {_interval(diff.pooled)}")


def print_walks(result: Diagnostics) -> None:
    print("\nWalks on dev (diagnostics: F-train and the task rates are never written to the operating-points file)")
    for name in (F_TRAIN, TASK_RATES):
        distinct, steps = breakpoints(result.tables[name])
        note = "quantile steps" if distinct > ALL_STEPS_UP_TO else "every breakpoint a step"
        print(f"  {name}: {distinct} distinct breakpoints, {steps} steps ({note})")
        for mix, by_share in result.walks[name].items():
            for share, walk_ in by_share.items():
                op = walk_.operating
                where = "falls back to A" if walk_.fell_back else f"threshold {op.threshold:.4f}"
                print(f"    {mix:9} {share:4.0%}: {where}; {len(walk_.certified) - 1} of {len(walk_.steps) - 1} steps "
                      f"tried certified; quality {100 * op.quality:.1f}%, ${op.cost_per_1k:.2f} per 1k")  # fmt: skip


def print_same_target(result: Diagnostics) -> None:
    print("\nSame-target comparisons, first less second (cost in dollars per 1k, quality in points; 95% intervals)")
    for (a, b), by_mix in result.same_target.items():
        print(f"  {a} against {b}")
        for mix, by_share in by_mix.items():
            for share, cell in by_share.items():
                if cell.both_fall_back:
                    print(f"    {mix:9} {share:4.0%}: both fall back to A")
                    continue
                assert cell.cost is not None and cell.quality is not None
                sides = ("first" if cell.first_fell_back else "") + ("second" if cell.second_fell_back else "")
                print(f"    {mix:9} {share:4.0%}: cost {cell.cost.value:+.2f} [{cell.cost.low:+.2f}, {cell.cost.high:+.2f}]; "
                      f"quality {100 * cell.quality.value:+.1f} [{100 * cell.quality.low:+.1f}, "
                      f"{100 * cell.quality.high:+.1f}]" + (f"   ({sides} side falls back to A)" if sides else ""))  # fmt: skip


def main() -> None:
    result = compute()
    print_readings(result)
    print_walks(result)
    print_same_target(result)


if __name__ == "__main__":
    main()
