"""Check the shared router foundation (PLAN_v4.md §8; Amendment A9, items 1, 3, 7 and 8).

Run from the project root:  uv run python -m analysis.check_routers

On synthetic data only, with answers worked out by hand. Never a real test item.

- the request keeps a row's prompt and output schema and nothing else;
- the decision rule: a router score equal to the threshold counts, nothing reaching it goes to the
  frontier, and a dearer config scored below a cheaper one is never chosen;
- the ladder the results-matrix module re-exports is the router package's own object and ends at
  the frontier;
- the benchmark reader yields train and dev rows from a synthetic file with test rows too, one split's
  rows on request, a request for a test item or for the test split is refused whole, and so is a file with a
  row of a split the benchmark doesn't have;
- the router-score table is refused whole for each bad case, and at a threshold becomes traces by
  the decision rule that carry its overhead, which the simulator accepts;
- no router-package module imports from analysis or harness, and the guard finds the import it
  looks for in a module that has one.
"""

import ast
import json
import sys
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from analysis import matrix as matrix_module
from analysis.benchmark import BenchmarkError, SealedItemError, read_rows
from analysis.matrix import Matrix
from analysis.router_table import ItemRouterScores, RouterScoreTable, TableError
from analysis.simulator import simulate
from harness.client import PROJECT_ROOT
from harness.collect import FRONTIER
from routers import ladder
from routers.decision import choose
from routers.request import Request, from_row

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(error: type[Exception], call: Callable[[], object]) -> bool:
    try:
        call()
    except error:
        return True
    return False


def imported_packages(path: Path) -> set[str]:
    """The top-level packages a module imports (a relative import stays inside its package)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


# A row as the benchmark stores it, with every field a router must not see.
ROW = {"item_id": "x:1", "split": "dev", "source": "banking77", "source_group": "x:g", "task_type": "classification",
       "evaluation_method": "exact_label", "expected_answer": "card_arrival", "meta": {"k": 1},
       "prompt": "Classify this.", "output_schema": {"type": "object"}}  # fmt: skip
SCHEMA_LESS = {k: v for k, v in ROW.items() if k != "output_schema"}
REQUEST = Request("Classify this.", {"type": "object"})

# Four dev items. Hand-worked configs at threshold 0.9: a haiku (0.9 equals it), b sonnet (haiku 0.5 is below),
# c the frontier (nothing reaches it), d haiku. At 0.96: a opus_low (0.99), and the frontier for b, c and d
# (d's haiku 0.95 and opus_low 0.9 are below it, and so is its sonnet 0.3).
ROUTER_SCORES = {
    "a": {"haiku": 0.9, "sonnet": 0.95, "opus_low": 0.99},
    "b": {"haiku": 0.5, "sonnet": 0.92, "opus_low": 0.6},
    "c": {"haiku": 0.1, "sonnet": 0.2, "opus_low": 0.3},
    "d": {"haiku": 0.95, "sonnet": 0.3, "opus_low": 0.9},
}
OVERHEAD = {"a": 0.0, "b": 0.00001, "c": 0.00002, "d": 0.00003}
ITEMS = list(ROUTER_SCORES)


def entries(**changes: dict[str, float]) -> list[ItemRouterScores]:
    return [ItemRouterScores(i, changes.get(i, ROUTER_SCORES[i]), OVERHEAD[i]) for i in ITEMS]


def build(items: list[ItemRouterScores]) -> RouterScoreTable:
    return RouterScoreTable.build(items, ITEMS)


def with_haiku_router_score(router_score: Any) -> RouterScoreTable:
    return build(entries(a={**ROUTER_SCORES["a"], "haiku": router_score}))


def with_overhead(overhead: float) -> RouterScoreTable:
    return build([ItemRouterScores("a", ROUTER_SCORES["a"], overhead), *entries()[1:]])


def synthetic_file(directory: Path) -> Path:
    splits = ("train", "dev", "test")
    rows = [{**ROW, "item_id": f"x:{split}-{i}", "split": split} for split in splits for i in range(2)]
    path = directory / "synthetic.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def main() -> int:
    print("-- the request --")
    request = from_row(ROW)
    check(request == REQUEST, "a row with a schema gives its prompt and schema")
    check(from_row(SCHEMA_LESS) == Request("Classify this.", None), "a row without a schema gives no schema")
    check(set(vars(request)) == {"prompt", "output_schema"},
          "the request holds only a prompt and an output schema: no task, split, source group, meta or answer")  # fmt: skip
    check(refused(AttributeError, lambda: setattr(request, "prompt", "other")), "the request is frozen")

    print("\n-- the decision rule --")
    check(choose(ROUTER_SCORES["a"], 0.9) == "haiku", "a router score equal to the threshold counts")
    check(choose(ROUTER_SCORES["b"], 0.9) == "sonnet", "the cheapest config that reaches the threshold is chosen")
    check(choose(ROUTER_SCORES["c"], 0.9) == FRONTIER, "nothing reaches the threshold: the frontier")
    check(choose({"haiku": 0.6, "sonnet": 0.4, "opus_low": 0.7}, 0.65) == "opus_low",
          "a dearer config scored below a cheaper one is passed over")  # fmt: skip
    check(
        choose({"haiku": 0.6, "sonnet": 0.9, "opus_low": 0.9}, 0.5) == "haiku", "a lower threshold keeps the cheapest"
    )
    check(choose(ROUTER_SCORES["c"], 0.0) == "haiku", "a threshold of 0 takes haiku")

    print("\n-- the ladder --")
    check(matrix_module.LADDER is ladder.LADDER, "the matrix module's LADDER is the router package's own object")
    check(ladder.LADDER[-1] == FRONTIER and ladder.FRONTIER == FRONTIER, "the ladder ends at the collection frontier")
    check(ladder.LADDER == ("haiku", "sonnet", "opus_low", "opus"), "the ladder is Amendment A8 item 5's order")

    print("\n-- the benchmark reader --")
    with tempfile.TemporaryDirectory() as tmp:
        path = synthetic_file(Path(tmp))
        kept = ["x:dev-0", "x:dev-1", "x:train-0", "x:train-1"]
        check(sorted(r["item_id"] for r in read_rows(path)) == kept,
              "from a file with train, dev and test rows it yields only train and dev")  # fmt: skip
        check([r["item_id"] for r in read_rows(path, ["x:dev-1"])] == ["x:dev-1"], "a named item comes back")
        check(refused(SealedItemError, lambda: read_rows(path, ["x:dev-0", "x:test-1"])),
              "asking for a test item is refused whole, even beside a dev item")  # fmt: skip
        check(refused(BenchmarkError, lambda: read_rows(path, ["x:nowhere"])), "an item not in the file is refused")
        check([r["item_id"] for r in read_rows(path, split="dev")] == ["x:dev-0", "x:dev-1"],
              "asking for one split yields just its rows")  # fmt: skip
        check(refused(SealedItemError, lambda: read_rows(path, split="test")), "asking for the test split is refused")
        check(refused(BenchmarkError, lambda: read_rows(path, split="holdout")), "asking for another split is refused")
        check(refused(BenchmarkError, lambda: read_rows(path, ["x:train-0"], split="dev")),
              "a named item outside the split asked for is refused")  # fmt: skip
        # A synthetic file only, never a real test item: a row in a split the benchmark doesn't have.
        stray = Path(tmp) / "stray.jsonl"
        stray.write_text(path.read_text() + json.dumps({**ROW, "item_id": "x:stray-0", "split": "holdout"}) + "\n")
        check(refused(BenchmarkError, lambda: read_rows(stray)),
              "a file with a row of a split other than train, dev or test is refused whole")  # fmt: skip
        check(not refused(BenchmarkError, lambda: read_rows(path)), "known positive: the same file without it is read")

    print("\n-- the router-score table: refusals --")
    table = build(entries())
    check(refused(TableError, lambda: build(entries()[:3])), "a missing item is refused")
    check(refused(TableError, lambda: build([*entries(), entries()[0]])), "a repeated item is refused")
    check(refused(TableError, lambda: build([*entries(), ItemRouterScores("z", ROUTER_SCORES["a"], 0.0)])),
          "an item outside the split is refused")  # fmt: skip
    check(refused(TableError, lambda: build(entries(a={**ROUTER_SCORES["a"], "opus": 1.0}))),
          "a router score for the frontier is refused")  # fmt: skip
    check(refused(TableError, lambda: build(entries(a={"haiku": 0.9, "sonnet": 0.95}))),
          "a config below the frontier left out is refused")  # fmt: skip
    for bad in (1.5, -0.1, float("nan"), float("inf"), "0.5", True):
        check(refused(TableError, partial(with_haiku_router_score, bad)),
              f"a router score of {bad!r} is refused")  # fmt: skip
    for bad in (-0.001, float("nan"), float("inf")):
        check(refused(TableError, partial(with_overhead, bad)),
              f"an overhead of {bad!r} is refused")  # fmt: skip
    check(not refused(TableError, lambda: build(entries(a={"haiku": 0, "sonnet": 1, "opus_low": 0.5}))),
          "router scores of exactly 0 and 1 are accepted")  # fmt: skip

    print("\n-- the router-score table: traces at a threshold --")
    costs = {config: 0.001 * (n + 1) for n, config in enumerate(ladder.LADDER)}
    matrix = Matrix.from_rows([
        {"item_id": i, "split": "dev", "task": "t", "source_group": i, "config": c, "cost": cost, "success": 1.0}
        for i in ITEMS for c, cost in costs.items()
    ])  # fmt: skip
    for threshold, expected in ((0.9, ["haiku", "sonnet", "opus", "haiku"]),
                                (0.96, ["opus_low", "opus", "opus", "opus"])):  # fmt: skip
        traces = table.traces(threshold)
        check([t.configs for t in traces] == [(c,) for c in expected], f"at {threshold} the configs are {expected}")
        check([t.overhead for t in traces] == list(OVERHEAD.values()), f"at {threshold} each trace has its overhead")
        check([o.configs for o in simulate(matrix, traces)] == [(c,) for c in expected],
              f"at {threshold} the simulator accepts the traces")  # fmt: skip

    print("\n-- no router-package module imports analysis or harness --")
    guarded = sorted((PROJECT_ROOT / "routers").glob("*.py"))
    check(len(guarded) >= 4, f"the guard sees the router package's modules {[p.name for p in guarded]}")
    for module in guarded:
        check(not imported_packages(module) & {"analysis", "harness"}, f"routers/{module.name} imports neither")
    check("harness" in imported_packages(PROJECT_ROOT / "analysis" / "matrix.py"),
          "known positive: analysis/matrix.py does import harness, and the guard sees it")  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
