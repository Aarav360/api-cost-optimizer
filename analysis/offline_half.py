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

- `flip_rates`, `target_met` and `headline` (#88), after the policies: dev's flip rates, which carry over to test's references
  (`analysis/flip_rates.py`); target met at a margin of zero for every policy's points, read from what the policies' parts
  evaluated (`analysis/target_met.py`); and the headline cascade chosen by A17 item 24's rule on dev's table
  (`analysis/final_headline.py`). Each part is built in order and kept in `inputs.cache["built"]`, so these three read the others.

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

from analysis import final_gates, final_headline, final_jev, final_routers, flip_rates, pareto, target_met
from analysis.bootstrap import RESAMPLES, SEED, Bootstrap, resample
from analysis.final_grading import matrix_path
from analysis.final_run import Run, StepResult
from analysis.matrix import LADDER, Matrix, MatrixError, list_price, load, read_store
from analysis.operating_eval import forget_evaluated, shared_fixed, shared_savings
from analysis.router_names import G_PLUS, NAMED_ROUTERS
from analysis.simulator import load_mixes
from analysis.unlock import Handle
from harness.client import load_models
from harness.collect import MAIN

TABLES_FILE = "tables_{split}.json"
G_PLUS_SECTION = NAMED_ROUTERS[G_PLUS]  # G+'s section of the operating-points file, spelled in one place
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
    # What parts share, filled as they build (`operating_eval.shared_fixed` and `shared_savings`): the fixed configs'
    # outcomes and each mix's resampled oracles, solved once. Not part of the inputs' identity.
    cache: dict = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class Extra:
    name: str
    dev: Callable[[], Any]  # from dev's stored records, through the existing readers
    test: Callable[[Run], Any]  # from test's, through the handle (`run.unlocked()`)
    # What the record keeps of the loaded value on test (and the tables step says), or None to keep nothing: a fact of the
    # run (fold shares, a truncation count), never a grade. Shown from the tables step on, as item 5 has it.
    record: Callable[[Any], Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class Part:
    key: str
    build: Callable[[Inputs], dict]
    committed: Callable[[], dict] | None  # the committed dev numbers `build` must give on dev, or None if dev has none


# What parts read beyond the matrix: one line each, in a ticket's own place. None yet: A to D need only the matrix.
EXTRAS: tuple[Extra, ...] = (
    # #83: E's and F's router-score tables, and F's facts (fold shares, truncations: kept in the record)
    Extra("rules_table", final_routers.rules_dev, final_routers.rules_test),
    Extra("trained_table", final_routers.trained_dev, final_routers.trained_test),
    Extra("trained_facts", lambda: None, final_routers.stored_scores, final_routers.trained_facts_record),
    #
    # #84: G's and G+'s router-score tables, and their test calls' facts (refusals, truncations: kept in the record)
    Extra("jev_table", final_jev.dev_table("jev"), final_jev.test_table("jev")),
    Extra("jev_bolstered_table", final_jev.dev_table(G_PLUS_SECTION), final_jev.test_table(G_PLUS_SECTION)),
    Extra("jev_facts", lambda: None, final_jev.facts, final_jev.facts_record),
    #
    # #85, #86: the gate-score table (the Jev gate's and the Sonnet gate's columns, `final_gates.test_data` and `dev_data`)
    Extra(final_gates.GATE_TABLE, final_gates.dev_data, final_gates.test_data, final_gates.gate_data_record),
)


def indicative_for(split: str, warning: str | None) -> str | None:
    """A7 item 2's warning, worded for `split`: `analysis/task_warnings.py` says "dev items", true of dev."""
    return None if warning is None else warning.replace(" dev items", f" {split} items")


def references_part(inputs: Inputs) -> dict:
    """A to D under each mix and per task: the numbers `analysis/pareto.py`'s `mix_panel` and `task_panels` build for dev."""
    matrix, draws, mixes = inputs.matrix, inputs.draws, load_mixes()
    outcomes = shared_fixed(inputs.cache, matrix)
    panels = {}
    for mix in MIXES:
        savings = shared_savings(inputs.cache, matrix, draws, mix)
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


STOPS_LABEL = "refusals and truncations by config"
NOT_STOPS = ("correct", "wrong", "no_answer")  # a graded row's other statuses are the stop reasons


def stops_report(stops: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    """Each config's stops other than a finished answer (refusals, cap stops, any other stop reason), as the tables report them."""
    return {config: {status: n for status, n in by_status.items() if status not in NOT_STOPS}
            for config, by_status in stops.items()}  # fmt: skip


# A part per line, in the order the tables file holds them. A policy's or a table's part goes in its own place.
PARTS: tuple[Part, ...] = (
    Part("references", references_part, committed_references),
    Part("stops", stops_part, None),
    # #83: E and F at their operating points
    Part("rules", final_routers.router_part("rules", "rules_table"), final_routers.committed_part("rules")),
    Part("trained", final_routers.router_part("trained", "trained_table"), final_routers.committed_part("trained")),
    #
    # #84: G and G+ at their operating points
    Part("jev", final_routers.router_part("jev", "jev_table"), final_routers.committed_part("jev")),
    Part(
        G_PLUS_SECTION,
        final_routers.router_part(G_PLUS_SECTION, "jev_bolstered_table"),
        final_routers.committed_part(G_PLUS_SECTION),
    ),  # fmt: skip
    #
    # #85: the free checks and the Jev gate: haiku/free and haiku/Jev (dev's per-task rows have no committed numbers)
    Part(
        final_gates.FREE_SECTION,
        final_gates.cascade_part(final_gates.FREE_SECTION),
        final_gates.committed_part(final_gates.FREE_SECTION),
    ),  # fmt: skip
    Part(
        final_gates.tasks_key(final_gates.FREE_SECTION),
        final_gates.cascade_part(final_gates.FREE_SECTION, per_task=True),
        None,
    ),  # fmt: skip
    Part(
        final_gates.JEV_SECTION,
        final_gates.cascade_part(final_gates.JEV_SECTION),
        final_gates.committed_part(final_gates.JEV_SECTION),
    ),  # fmt: skip
    Part(
        final_gates.tasks_key(final_gates.JEV_SECTION),
        final_gates.cascade_part(final_gates.JEV_SECTION, per_task=True),
        None,
    ),  # fmt: skip
    #
    # #86: the Sonnet gate: haiku/Sonnet (dev's per-task rows have no committed numbers)
    Part(
        final_gates.SONNET_SECTION,
        final_gates.cascade_part(final_gates.SONNET_SECTION),
        final_gates.committed_part(final_gates.SONNET_SECTION),
    ),  # fmt: skip
    Part(
        final_gates.tasks_key(final_gates.SONNET_SECTION),
        final_gates.cascade_part(final_gates.SONNET_SECTION, per_task=True),
        None,
    ),  # fmt: skip
    #
    # #87: the twelve router-started cascades
    #
    # #88: dev's flip rates (they carry over to test's references), then target met over every policy built above, then the
    # headline from it: keep these after the policies' parts, and the headline after target met
    Part("flip_rates", flip_rates.part, flip_rates.committed),
    Part("target_met", target_met.part, None),
    Part("headline", final_headline.part, None),
    #
    # #89: the final table, the 51 comparisons and the replication summary
)


def build_tables(inputs: Inputs, parts: tuple[Part, ...] = PARTS) -> dict:
    """The tables for one split: its facts, and each part's numbers by its key."""
    # The parts built so far are kept in `inputs.cache["built"]` for the parts that read the others (target met, the headline),
    # and the policies' outcomes that target met reads start afresh.
    built: dict[str, dict] = {}
    inputs.cache["built"] = built
    forget_evaluated(inputs.cache)
    for part in parts:
        built[part.key] = part.build(inputs)
    return {
        "built_by": "analysis/offline_half.py",
        "split": inputs.split,
        "run": 0,
        "n": len(inputs.matrix.items),
        "price_snapshot": load_models()["price_snapshot"],
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "interval": "percentile 95%"},
        "parts": built,
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
    kept = {extra.name: extra.record(inputs.extras[extra.name]) for extra in EXTRAS if extra.record is not None}
    for name, facts in kept.items():
        run.say(f"tables: {name}: {json.dumps(facts)}")
    stops = stops_report(tables["parts"]["stops"])
    run.say(
        f"tables: {STOPS_LABEL}: {json.dumps(stops)}"
    )  # a refusal is a rejection and a cap stop a failure: outcomes (item 5)
    for mix in MIXES:
        pareto.print_mix(mix, tables["parts"]["references"]["mixes"][mix])
    counts = tables["parts"]["target_met"]["counts"]["all"]
    run.say(f"tables: target met at a margin of zero: {json.dumps(counts)}")
    headline = tables["parts"]["headline"]
    run.say(f"tables: headline, {headline['label']} at realistic 99%: {headline['headline']['reading']}")
    # The stops are in the log and the tables file; the record's own fields for them are item 10's (#92).
    return StepResult(detail={"file": path.name, "items": len(inputs.matrix.items), "extras": kept, "target_met": counts,
                              "headline": headline["headline"]["reading"]})  # fmt: skip
