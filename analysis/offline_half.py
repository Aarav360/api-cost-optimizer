"""The offline half of the final evaluation: tables from a split's stored inputs (PLAN_v4.md Amendment A19, items 2, 8
and 13; issue #77, #79).

One function builds the tables for a split from that split's stored inputs (`Inputs`): dev's through the existing readers
(`dev_inputs`), test's through the handle and test's results matrix (`test_inputs`). On dev it must reproduce the
committed dev numbers to the last stored digit (`analysis/check_final_run_dev.py`, the dev reproduction); on test it
writes `tables_test.json` in the world's output folder, a file of its own, and chart v4's dev files stay as they are.

What is built is a list of parts, each a function of the inputs, with the committed dev numbers it must reproduce:

- `references`: the references A to D (A: always `opus`, B: always `haiku`, C: the random-mixture envelope, D: the
  hindsight oracle) under both task mixes and per task, as `analysis/pareto.py` builds them for dev, with IFEval's
  and MBPP+'s per-task entries marked indicative (A7 item 2; the words name the split, `indicative_for`);
- `stops`: how many answers each config's graded rows hold of each status, which carries the refusal and truncation counts
  that A19 item 5 has reported with the tables and not before.

How to extend: a policy or a table is one `Part` line in `PARTS` (its function in a module of its own, the committed dev
numbers it must reproduce as a second function, or None where dev has none), and what it reads beyond the matrix is one
`Extra` line in `EXTRAS`: a name and how to load it from dev's stored records and from test's through the handle. A part
reads `inputs.extras[name]`. Nothing else here changes, and the dev reproduction and the tables step run every line.

Only the tables step calls `tables_step`, after grading. Reading test's rows happens there and nowhere earlier.
"""

import json
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analysis import pareto
from analysis.bootstrap import RESAMPLES, SEED, Bootstrap, resample
from analysis.final_grading import matrix_path
from analysis.final_run import Run, StepResult
from analysis.matrix import LADDER, Matrix, MatrixError, list_price, load, read_store
from analysis.simulator import always, load_mixes, simulate
from analysis.unlock import Handle
from harness.client import load_models
from harness.collect import MAIN

TABLES_FILE = "tables_{split}.json"
MIXES = pareto.PANELS  # realistic, then uniform: the figures' order


class OfflineError(ValueError):
    """The stored inputs of a split don't make its tables. Nothing is built."""


@dataclass(frozen=True)
class Inputs:
    """One split's stored inputs: its matrix (run 0), the resamples drawn from it (one set serves every policy, mix and
    task, so every difference is paired), each config's count of graded rows by status, and what the parts that need
    more load (`EXTRAS`)."""

    split: str
    matrix: Matrix
    draws: Bootstrap
    statuses: Mapping[str, Mapping[str, int]]
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Extra:
    name: str
    dev: Callable[[], Any]  # from dev's stored records, through the existing readers
    test: Callable[[Run], Any]  # from test's, through the handle (`run.unlocked()`)


@dataclass(frozen=True)
class Part:
    key: str
    build: Callable[[Inputs], dict]
    committed: Callable[[], dict] | None  # the committed dev numbers `build` must give on dev, or None if dev has none


# What parts read beyond the matrix: one line each, in a ticket's own place. None yet: A to D need only the matrix.
EXTRAS: tuple[Extra, ...] = (
    # #83: E's and F's router-score tables
    #
    # #84: G's and G+'s router-score tables
    #
    # #85, #86: the gate-score table
)


def indicative_for(split: str, warning: str | None) -> str | None:
    """A7 item 2's warning, worded for `split`: `analysis/task_warnings.py` says "dev items", true of dev."""
    return None if warning is None else warning.replace(" dev items", f" {split} items")


def references_part(inputs: Inputs) -> dict:
    """A to D under each mix and per task: the numbers `analysis/pareto.py`'s `mix_panel` and `task_panels` build for dev."""
    matrix, draws, mixes = inputs.matrix, inputs.draws, load_mixes()
    outcomes = {config: simulate(matrix, always(matrix, config)) for config in LADDER}
    panels = {}
    for mix in MIXES:
        savings = draws.savings(matrix, outcomes["opus"], mixes[mix])
        panels[mix] = pareto.mix_panel(matrix, draws, outcomes, mixes[mix], savings)
    tasks = pareto.task_panels(matrix, draws, outcomes)
    for panel in tasks.values():
        panel["warning"] = indicative_for(inputs.split, panel["warning"])
    return {"mixes": panels, "tasks": tasks}


def committed_references() -> dict:
    """Dev's A to D as committed: chart v0's keys of each panel in `figures/pareto.json` and `figures/pareto_tasks.json`."""
    mix_keys = ("configs", "envelope", "oracle", "oracle_max_quality", "failed_by_every_config", "targets")
    task_keys = ("items", "warning", "configs", "envelope", "oracle", "oracle_max_quality")
    headline = json.loads(pareto.FIGURE_DATA.read_text(encoding="utf-8"))["panels"]
    per_task = json.loads(pareto.TASK_FIGURE_DATA.read_text(encoding="utf-8"))["panels"]
    return {
        "mixes": {mix: {key: headline[mix][key] for key in mix_keys} for mix in MIXES},
        "tasks": {task: {key: per_task[task][key] for key in task_keys} for task in pareto.TASKS},
    }


def stops_part(inputs: Inputs) -> dict:
    """Each config's graded rows by status (correct, wrong, no_answer, or the stop reason): refusals and truncations are
    counted here, with the tables."""
    return {config: dict(sorted(inputs.statuses.get(config, {}).items())) for config in LADDER}


# A part per line, in the order the tables file holds them. A policy's or a table's part goes in its own place.
PARTS: tuple[Part, ...] = (
    Part("references", references_part, committed_references),
    Part("stops", stops_part, None),
    # #83: E and F at their operating points
    #
    # #84: G and G+
    #
    # #85: the free checks and the Jev gate: haiku/free and haiku/Jev
    #
    # #86: the Sonnet gate: haiku/Sonnet
    #
    # #87: the twelve router-started cascades
    #
    # #88: target met, the oracle, the envelope and the headline
    #
    # #89: the final table, the 51 comparisons and the replication summary
)


def build_tables(inputs: Inputs, parts: tuple[Part, ...] = PARTS) -> dict:
    """The tables for one split: its facts, and each part's numbers by its key."""
    return {
        "built_by": "analysis/offline_half.py",
        "split": inputs.split,
        "run": 0,
        "n": len(inputs.matrix.items),
        "price_snapshot": load_models()["price_snapshot"],
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "interval": "percentile 95%"},
        "parts": {part.key: part.build(inputs) for part in parts},
    }


# Dev's inputs, through the existing readers


def read_statuses(path: Path, split: str) -> dict[str, dict[str, int]]:
    """Run 0's rows of `split` counted by config and status."""
    query = "SELECT config, status, COUNT(*) FROM rows WHERE run = 0 AND split = ? GROUP BY config, status"
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        counts: dict[str, dict[str, int]] = {}
        for config, status, count in conn.execute(query, (split,)):
            counts.setdefault(config, {})[status] = count
        return counts


def dev_inputs() -> Inputs:
    """Dev's matrix and statuses from the dev results matrix in results/ (a rebuild takes about a minute)."""
    matrix = load().split("dev")
    return Inputs("dev", matrix, resample(matrix), read_statuses(MAIN.matrix, "dev"),
                  {extra.name: extra.dev() for extra in EXTRAS})  # fmt: skip


# Test's inputs, through the handle


def test_matrix(rows: list[dict], groups: Mapping[str, tuple[str, str]]) -> Matrix:
    """Test's matrix from its graded rows, checked as `analysis/matrix.py` checks dev's: the benchmark's source and group
    for each item (from the handle's graded items, since the benchmark reader refuses test), the price snapshot, and
    each row's cost recomputed from its tokens at list price, which must be twice the stored batch cost. Only run 0 goes in."""
    models = load_models()
    built = []
    for row in rows:
        if row["split"] != "test":
            raise OfflineError(f"{row['item_id']} is a {row['split']} row in test's matrix")
        if row["item_id"] not in groups or groups[row["item_id"]][0] != row["source"]:
            raise OfflineError(f"{row['item_id']} is not a test item of the benchmark")
        if row["price_snapshot"] != models["price_snapshot"]:
            raise OfflineError(f"{row['item_id']} is priced at {row['price_snapshot']}")
        cost = list_price(row, models["prices"])
        if cost != row["cost_usd"] / models["batch_discount"]:
            raise OfflineError(f"{row['item_id']} {row['config']}: list price is not twice the stored batch cost")
        if row["run"] == 0:
            built.append({"item_id": row["item_id"], "split": "test", "task": row["source"],
                          "source_group": groups[row["item_id"]][1], "config": row["config"], "cost": cost,
                          "success": row["success"]})  # fmt: skip
    try:
        return Matrix.from_rows(built)
    except MatrixError as error:
        raise OfflineError(str(error)) from None


def test_inputs(run: Run) -> Inputs:
    """Test's matrix and statuses from test's results matrix, with each item's source group from the handle."""
    handle: Handle = run.unlocked()
    graded = handle.graded_items()
    groups = {g.item["item_id"]: (g.item["source"], g.item["source_group"]) for g in graded}
    path = matrix_path(run.world.results)
    if not path.is_file():
        raise OfflineError("test's results matrix does not exist: grading has not run")
    rows = read_store(path)
    stored = {g.custom_id for g in graded}
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        graded_ids = {row[0] for row in conn.execute("SELECT custom_id FROM rows")}
    if graded_ids != stored:
        raise OfflineError(f"{len(stored - graded_ids)} stored answers are ungraded, and {len(graded_ids - stored)} "
                           f"graded rows have no stored answer")  # fmt: skip
    matrix = test_matrix(rows, groups)
    return Inputs("test", matrix, resample(matrix), read_statuses(path, "test"),
                  {extra.name: extra.test(run) for extra in EXTRAS})  # fmt: skip


def tables_step(run: Run) -> StepResult:
    """Build test's tables and write them to `tables_test.json` in the output folder. From here on, outcomes may be shown."""
    inputs = test_inputs(run)
    tables = build_tables(inputs)
    path = run.world.output / TABLES_FILE.format(split="test")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
    run.say(f"tables: wrote {path.name}, {len(inputs.matrix.items)} test items")
    for mix in MIXES:
        pareto.print_mix(mix, tables["parts"]["references"]["mixes"][mix])
    return StepResult(detail={"file": path.name, "items": len(inputs.matrix.items)})
