"""E and F on test: their router scores, their router-score tables, and their frozen operating points (PLAN_v4.md Amendment
A9, item 7; A11, items 9 and 17; A19, items 2 and 8; issue #77, #83).

**The calls step** (`calls_step`, A19 item 2.4) scores every test request through the handle's requests: E by
`routers.rules.score(request)`, F by `routers.trained.score(request)`, the hash path, which picks each request's fold model
(a rehearsal's world names a fake embedder, and the same `TrainedRouter.score` runs on it). The scores, each request's fold
(`routers.trained.fold`, the same hash) and F's truncation and emptied counts are stored in `test_router_scores.json` under
the world's results folder, so a resume embeds nothing twice and the tables step reads what the calls step stored. F-train
and the diagnostics have no code here and none that this module's steps import: they read dev's cache, dev's matrix and
dev's reader (`analysis/check_final_routers.py` shows each refuses test).

**The tables step reads them back** through two `Extra`s (`analysis/offline_half.py`): E's and F's router-score tables over
the handle's request IDs (`RouterScoreTable.build`, the table's own refusals), and F's facts, which the record keeps and the
tables step says from then on: each fold model's share of test and the truncation count (item 5 reports a count that may carry
an outcome only with the tables, so the calls step shows neither). Two `Part`s then evaluate each router at its six frozen
operating points (`analysis/operating_eval.py`) under both mixes and per task. On dev, the same parts reproduce the numbers
table v1 committed for E and F.

Nothing here takes a plain `"test"`: `score_test` refuses any handle the unlock didn't issue.
"""

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from analysis import pareto
from analysis.final_run import ItemFailure, Run, StepResult
from analysis.operating_eval import (
    evaluate,
    frozen_points,
    mix_rows,
    router_traces,
    shared_fixed,
    shared_savings,
    task_rows,
)
from analysis.router_table import LOCAL_OVERHEAD, ItemRouterScores, RouterScoreTable
from analysis.simulator import load_mixes
from analysis.unlock import Handle, require_issued
from analysis.world import World
from routers import rules
from routers.request import Request

if TYPE_CHECKING:
    from analysis.offline_half import Inputs

SCORES_FILE = "test_router_scores.json"  # in the world's results folder: test's record of E's and F's router scores
RULES, TRAINED = "rules", "trained"  # E's and F's sections of the operating-points file
FOLDS = 5


class RoutersError(ValueError):
    """Stored router scores that are not the handle's requests', or a file that isn't what the step wrote. Nothing is read."""


@dataclass(frozen=True)
class SplitScores:
    """E's and F's router scores for each request, each request's fold model, and F's counts."""

    rules: dict[str, dict[str, float]]
    trained: dict[str, dict[str, float]]
    folds: dict[str, int]
    truncated: int  # requests longer than F's window: their head and tail were embedded
    emptied: int  # requests whose template lines left nothing

    def fold_counts(self) -> dict[int, int]:
        counts = dict.fromkeys(range(FOLDS), 0)
        for fold in self.folds.values():
            counts[fold] += 1
        return counts

    def record(self) -> dict[str, Any]:
        """What the record keeps and the tables step says: each fold model's count and share of test, and the counts."""
        counts, total = self.fold_counts(), len(self.folds)
        return {
            "items": total,
            "fold_counts": {str(k): n for k, n in counts.items()},
            "fold_shares": {str(k): n / total for k, n in counts.items()},
            "truncated": self.truncated,
            "emptied": self.emptied,
        }


def scores_path(results: Path) -> Path:
    return results / SCORES_FILE


# Scoring


def trained_scorer(world: World) -> tuple[Callable[[Request], dict[str, float]], Callable[[], tuple[int, int]]]:
    """F's `score(request)` and a reading of its (truncated, emptied) counts. The real route is `routers.trained.score`, the
    committed files and the verified snapshot; a world that names a stand-in embedder runs `TrainedRouter.score`, the same
    method, on it."""
    from routers import trained
    from routers.trained import TrainedRouter

    stand_in = world.backends.trained if world.backends is not None else None
    if stand_in is not None and stand_in.embedder is not None:
        router = TrainedRouter(trained.read_settings(), trained.read_fold_weights(), backend=stand_in.embedder)
        return router.score, lambda: (router.counts.truncated, router.counts.emptied)
    counts = trained.default_router().counts
    return trained.score, lambda: (counts.truncated, counts.emptied)


def score_test(handle: Handle, world: World) -> SplitScores:
    """Score every test request through the handle with E and F. A plain `"test"`, or a handle the unlock didn't issue, is
    refused (HandleError); a request that fails names its item (`ItemFailure`)."""
    from routers import trained

    requests = require_issued(handle).requests()
    score, counts = trained_scorer(world)
    before = counts()
    scored: dict[str, dict[str, dict[str, float]]] = {RULES: {}, TRAINED: {}}
    folds: dict[str, int] = {}
    for item_id, request in requests.items():
        try:
            scored[RULES][item_id] = rules.score(request)
            scored[TRAINED][item_id] = score(request)
            folds[item_id] = trained.fold(request)
        except Exception as error:
            raise ItemFailure(item_id) from error
    after = counts()
    return SplitScores(scored[RULES], scored[TRAINED], folds, after[0] - before[0], after[1] - before[1])


# Storing and reading back


def write_scores(path: Path, scores: SplitScores) -> None:
    data = {"rules": scores.rules, "trained": scores.trained, "folds": scores.folds, "truncated": scores.truncated,
            "emptied": scores.emptied}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_scores(path: Path, item_ids: Mapping[str, Any]) -> SplitScores:
    """The stored scores, refused (RoutersError) unless they cover exactly `item_ids`, each request once."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        scores = SplitScores(data["rules"], data["trained"], data["folds"], data["truncated"], data["emptied"])
    except (OSError, KeyError, ValueError) as error:
        raise RoutersError(
            f"{path.name} is missing or isn't what the calls step wrote ({type(error).__name__})"
        ) from None
    wanted = set(item_ids)
    for name, held in (("rules", scores.rules), ("trained", scores.trained), ("folds", scores.folds)):
        if set(held) != wanted:
            raise RoutersError(f"{path.name}'s {name} are not the handle's {len(wanted)} requests: "
                               f"{len(set(held) - wanted)} others, {len(wanted - set(held))} missing")  # fmt: skip
    if not all(isinstance(fold, int) and fold in range(FOLDS) for fold in scores.folds.values()):
        raise RoutersError(f"{path.name} holds a fold outside 0 to {FOLDS - 1}")
    return scores


def table_of(held: Mapping[str, Mapping[str, float]], item_ids: list[str]) -> RouterScoreTable:
    """A router-score table over test's items in the handle's order: E's and F's overhead is an explicit $0 (A9 item 6, A11 item 11)."""
    items = [ItemRouterScores(i, dict(held[i]), LOCAL_OVERHEAD) for i in item_ids]
    return RouterScoreTable.build(items, item_ids)


def calls_step(run: Run) -> StepResult:
    """A19 item 2.4 for E and F: score every test request once, store the scores, and say how many. A resume finds them stored."""
    handle = run.unlocked()
    requests = handle.requests()
    path = scores_path(run.world.results)
    if path.is_file():
        read_scores(path, requests)
        run.say(f"calls_rules_trained: {len(requests)} requests already scored and stored")
        return StepResult(
            requests={"sent": 0, "stored": len(requests)}, detail={"items": len(requests), "stored": True}
        )
    scores = score_test(handle, run.world)
    write_scores(path, scores)
    run.say(f"calls_rules_trained: E and F scored {len(requests)} test requests, through the handle")
    return StepResult(requests={"sent": len(requests), "stored": 0}, detail={"items": len(requests), "stored": False})


# The offline half's loaders


def stored_scores(run: Run) -> SplitScores:
    """The calls step's stored scores, over the handle's request IDs."""
    requests = run.unlocked().requests()
    return read_scores(scores_path(run.world.results), requests)


def rules_test(run: Run) -> RouterScoreTable:
    return table_of(stored_scores(run).rules, list(run.unlocked().requests()))


def trained_test(run: Run) -> RouterScoreTable:
    return table_of(stored_scores(run).trained, list(run.unlocked().requests()))


def rules_dev() -> RouterScoreTable:
    from analysis import rules_table

    return rules_table.dev_table()


def trained_dev() -> RouterScoreTable:
    from analysis import trained_table

    return trained_table.dev_table()


def trained_facts_record(scores: SplitScores | None) -> Mapping[str, Any]:
    return {} if scores is None else scores.record()


# The parts


def router_part(section: str, table_name: str) -> Callable[["Inputs"], dict]:
    """A router's part: its frozen points evaluated on the split (`analysis/operating_eval.py`), under both mixes and per
    task. `table_name` is the `Extra` holding the router's table for this split."""

    def build(inputs: "Inputs") -> dict:
        table: RouterScoreTable = inputs.extras[table_name]
        fixed = shared_fixed(inputs.cache, inputs.matrix)
        points = frozen_points(section)
        outcomes = evaluate(inputs.matrix, fixed["opus"], points, router_traces(table.traces))
        rows = mix_rows(inputs.draws, lambda mix: shared_savings(inputs.cache, inputs.matrix, inputs.draws, mix),
                        fixed, outcomes, load_mixes())  # fmt: skip
        return {"mixes": rows, "tasks": task_rows(inputs.draws, fixed, outcomes, list(pareto.TASKS))}

    return build


MIX_KEYS = ("threshold", "fell_back", "target", "quality", "cost_per_1k", "difference_from_A", "savings_vs_A", "captured",
            "routing", "matched_cost")  # fmt: skip


def committed_part(section: str) -> Callable[[], dict]:
    """The numbers table v1 committed for the router on dev (`figures/pareto.json` and `figures/pareto_tasks.json`): the keys a
    frozen row has. A walk's margin and steps tried are the walk's, and are not reproduced."""

    def committed() -> dict:
        headline = json.loads(pareto.FIGURE_DATA.read_text(encoding="utf-8"))["panels"]
        per_task = json.loads(pareto.TASK_FIGURE_DATA.read_text(encoding="utf-8"))["panels"]
        mixes = {mix: {target: {key: point[key] for key in MIX_KEYS} for target, point in
                       headline[mix]["routers"][section]["operating_points"].items()} for mix in pareto.PANELS}  # fmt: skip
        tasks = {task: per_task[task]["routers"][section]["operating_points"] for task in pareto.TASKS}
        return {"mixes": mixes, "tasks": tasks}

    return committed
