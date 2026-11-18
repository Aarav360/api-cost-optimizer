"""Check a policy evaluated at its frozen operating points, with no walk (PLAN_v4.md Amendment A19, items 6 and 8; issue #83).

Run from the project root:  uv run python -m analysis.check_operating_eval

On a hand-worked synthetic matrix, resamples and router-score table, and the committed operating-points file. No results/,
no router runs on a real item, and nothing reads test.

- the file's six points per section are read as it holds them, and the routing counts are A19 item 6's (E 4, F 3 route);
  a section that lacks a point, holds a non-bool flag or gives a fallen-back point a dial is refused;
- a router's points are evaluated at their thresholds, a point whose policy ties the threshold's score counts (`>=`), and
  quality, cost, difference from A, savings, routing mix and McNemar's counts are the hand-worked ones;
- **the file's `fell_back` decides, not the split's costs.** A point the file says does not fall back, at a threshold above
  every router score (so it sends everything to A and costs what A costs), is labelled not fallen back and is tested
  against A; a hand-built walk's "falls back" would call it A. A point the file says fell back is A itself, whatever the
  table would do at any threshold, with no McNemar row;
- the row's target is the share of A's quality on this split, and a walk's margin and steps are not in it;
- the same numbers come from `router_panels.point_numbers`, so a row here and a row of table v1 are one computation.
"""

import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from analysis.bootstrap import Bootstrap
from analysis.matrix import Matrix
from analysis.operating_eval import (
    FrozenPoint,
    FrozenPointError,
    evaluate,
    fixed_outcomes,
    frozen_points,
    mix_rows,
    router_traces,
    task_rows,
)
from analysis.operating_points import OperatingPointsError, read_section
from analysis.router_table import ItemRouterScores, RouterScoreTable
from analysis.walk import Walk, walk

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


def close(a: float | None, b: float, tolerance: float = 1e-12) -> bool:
    return a is not None and abs(a - b) <= tolerance


# Four items of one task, each its own source group. Per-request dollars: haiku .001, sonnet .002, opus_low .004, opus
# .010 (so $10 per 1k on A). A (opus) is right on i0, i1, i2 and wrong on i3: 75%. haiku is right on i0, i2, i3 and wrong
# on i1; sonnet and opus_low are right everywhere.
COSTS = {"haiku": 0.001, "sonnet": 0.002, "opus_low": 0.004, "opus": 0.010}
RIGHT = {
    "i0": {"haiku": 1, "sonnet": 1, "opus_low": 1, "opus": 1},
    "i1": {"haiku": 0, "sonnet": 1, "opus_low": 1, "opus": 1},
    "i2": {"haiku": 1, "sonnet": 1, "opus_low": 1, "opus": 1},
    "i3": {"haiku": 1, "sonnet": 1, "opus_low": 1, "opus": 0},
}
MATRIX = Matrix.from_rows([
    {"item_id": i, "split": "test", "task": "t1", "source_group": i, "config": c, "cost": COSTS[c], "success": s}
    for i, by_config in RIGHT.items() for c, s in by_config.items()
])  # fmt: skip
# Router scores (haiku, sonnet, opus_low). At threshold .8: i0 haiku (.9); i1 sonnet (haiku .7 misses, sonnet .8 ties the
# threshold and counts); i2 opus (nothing reaches .8); i3 sonnet (haiku .5 misses, sonnet .85). At .6: i0 haiku; i1 haiku
# (.7); i2 opus; i3 sonnet.
SCORES = {
    "i0": {"haiku": 0.9, "sonnet": 0.9, "opus_low": 0.9},
    "i1": {"haiku": 0.7, "sonnet": 0.8, "opus_low": 0.9},
    "i2": {"haiku": 0.1, "sonnet": 0.2, "opus_low": 0.3},
    "i3": {"haiku": 0.5, "sonnet": 0.85, "opus_low": 0.9},
}
TABLE = RouterScoreTable.build([ItemRouterScores(i, s, 0.0) for i, s in SCORES.items()], list(SCORES))
DRAWS = Bootstrap(("i0", "i1", "i2", "i3"), ("t1",) * 4,
                  np.array([[1, 1, 1, 1], [2, 1, 0, 1], [0, 2, 0, 2], [2, 0, 2, 0]]))  # fmt: skip
MIXES = {"realistic": {"t1": 1.0}, "uniform": {"t1": 1.0}}
MCNEMAR = {"b": 1, "c": 0, "p": 1.0}  # at .8 the policy gets i3 right where A does not, and loses nothing


def section(points: dict[str, Any]) -> dict:
    """A section with the same points under both mixes: {target: point}."""
    return {"realistic": points, "uniform": points}


FELL = {"threshold": None, "fell_back": True}
HELD = section({
    "100%": FELL,
    "99%": {"threshold": 0.8, "fell_back": False},
    "97%": {"threshold": 0.6, "fell_back": False},
})  # fmt: skip


def file_with(sections: dict, folder: Path) -> Path:
    path = folder / "operating_points.json"
    path.write_text(json.dumps(sections), encoding="utf-8")
    return path


def evaluated(held: dict, matrix: Matrix = MATRIX, table: RouterScoreTable = TABLE, draws: Bootstrap = DRAWS):
    with tempfile.TemporaryDirectory() as folder:
        points = frozen_points("toy", file_with({"toy": held}, Path(folder)))
    fixed = fixed_outcomes(matrix)
    outcomes = evaluate(matrix, fixed["opus"], points, router_traces(table.traces))
    savings = {mix: draws.savings(matrix, fixed["opus"], MIXES[mix]) for mix in MIXES}
    rows = mix_rows(draws, savings.__getitem__, fixed, outcomes, MIXES)
    return points, outcomes, rows, task_rows(draws, fixed, outcomes, ["t1"])


def reading() -> None:
    print("-- the file's points, as it holds them --")
    for name, routes in (("rules", 4), ("trained", 3)):
        points = frozen_points(name)
        check(
            len(points) == 6
            and [(p.mix, p.share) for p in points]
            == [(m, s) for m in ("realistic", "uniform") for s in (1.0, 0.99, 0.97)],  # fmt: skip
            f"{name}: six points, both mixes, targets from 100% down",
        )
        check(sum(not p.fell_back for p in points) == routes and all((p.threshold is None) == p.fell_back for p in points),
              f"{name}: {routes} of the six route (A19 item 6), and a point that fell back has no threshold")  # fmt: skip
    held = read_section("rules")
    check({p.target: p.threshold for p in frozen_points("rules") if p.mix == "realistic"}
          == {"100%": None, "99%": held["realistic"]["99%"]["threshold"], "97%": held["realistic"]["97%"]["threshold"]}
          and frozen_points("rules")[1].threshold == 0.9171597633136095,
          "E's realistic 99% threshold is the file's 0.9171597633136095, unrounded")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        where = Path(folder)
        check(refused(OperatingPointsError, lambda: frozen_points("toy", file_with({"toy": {"realistic": HELD["realistic"]}}, where))),
              "a section with no uniform mix is refused")  # fmt: skip
        bad = section({"100%": {"threshold": None, "fell_back": "yes"}, "99%": FELL, "97%": FELL})
        check(refused(FrozenPointError, lambda: frozen_points("toy", file_with({"toy": bad}, where))),
              "a flag that is not a bool is refused")  # fmt: skip
        dialled = section({"100%": {"threshold": 0.9, "fell_back": True}, "99%": FELL, "97%": FELL})
        check(refused(FrozenPointError, lambda: frozen_points("toy", file_with({"toy": dialled}, where))),
              "a point that fell back but holds a dial is refused")  # fmt: skip
        check(refused(OperatingPointsError, lambda: frozen_points("nowhere", file_with({"toy": HELD}, where))),
              "a section the file lacks is refused")  # fmt: skip


def hand_worked() -> None:
    print("\n-- the points evaluated by hand --")
    points, outcomes, rows, tasks = evaluated(HELD)
    by_key = {(p.mix, p.target): p for p in points}
    at_99 = outcomes[by_key[("realistic", "99%")]]
    check({o.item_id: o.configs for o in at_99} == {"i0": ("haiku",), "i1": ("sonnet",), "i2": ("opus",), "i3": ("sonnet",)},
          "at .8 the decisions are haiku, sonnet (its .8 ties the threshold and counts), opus, sonnet")  # fmt: skip
    row = rows["realistic"]["99%"]
    check(row["threshold"] == 0.8 and row["fell_back"] is False and "cutoff" not in row,
          "the 99% row carries the file's threshold and flag, and no cutoff for a router")  # fmt: skip
    check(close(row["quality"]["value"], 1.0) and close(row["cost_per_1k"]["value"], 3.75),
          "its quality is 4 of 4 and its cost (.001 + .002 + .010 + .002) / 4 = $3.75 per 1k")  # fmt: skip
    check(close(row["difference_from_A"]["value"], 0.25) and close(row["savings_vs_A"]["per_1k"], 6.25)
          and close(row["savings_vs_A"]["share"], 0.625), "against A (75%, $10): +25 points, $6.25 and 62.5% saved")  # fmt: skip
    check(row["routing"] == {"haiku": 0.25, "sonnet": 0.5, "opus_low": 0.0, "opus": 0.25},
          "the routing mix is a quarter haiku, half sonnet, a quarter opus")  # fmt: skip
    check(close(row["target"], 0.99 * 0.75), "the target is 99% of A's 75% on this split")
    check("margin" not in row and "steps_tried" not in row, "a walk's margin and steps are not in a frozen row")
    low = rows["realistic"]["97%"]
    check(close(low["quality"]["value"], 0.75) and close(low["cost_per_1k"]["value"], 3.5)
          and close(low["target"], 0.97 * 0.75), "at .6 i1 goes to haiku and is wrong: 75% at $3.50, target 97% of 75%")  # fmt: skip
    check(rows["uniform"] == rows["realistic"], "both mixes are one task here, so they agree")
    t99 = tasks["t1"]["realistic"]["99%"]
    check(t99["threshold"] == 0.8 and t99["fell_back"] is False and t99["mcnemar"] == MCNEMAR
          and close(t99["quality"]["value"], 1.0) and close(t99["cost_per_1k"]["value"], 3.75),
          "per task: the policy's quality and cost, and McNemar's b=1 (i3), c=0, p=1 against A")  # fmt: skip


def flag_decides() -> None:
    print("\n-- the file's flag decides, not the split's costs --")
    # A threshold above every score sends all four to A: the policy costs exactly A, and a walk would call it A.
    held = section({"100%": {"threshold": 0.99, "fell_back": False}, "99%": FELL, "97%": FELL})
    points, outcomes, rows, tasks = evaluated(held)
    row = rows["realistic"]["100%"]
    check(close(row["cost_per_1k"]["value"], 10.0) and close(row["quality"]["value"], 0.75)
          and row["fell_back"] is False and row["threshold"] == 0.99,
          "a point the file says routes, whose policy costs what A costs, is labelled as the file has it: not fallen back")  # fmt: skip
    check(tasks["t1"]["realistic"]["100%"]["mcnemar"] == {"b": 0, "c": 0, "p": 1.0},
          "and it is tested against A (a tie, b = c = 0), where a point that fell back has no test")  # fmt: skip
    # Where haiku costs what A costs, a walk that tries the router's one step finds it no cheaper than A and falls back; the
    # file, frozen on dev, says whether the point routes, and the label follows it.
    tie = Matrix.from_rows([
        {"item_id": i, "split": "test", "task": "t1", "source_group": i, "config": c, "cost": 0.010, "success": 1}
        for i in ("j0", "j1") for c in COSTS
    ])  # fmt: skip
    tie_scores = dict.fromkeys(SCORES["i0"], 0.9)
    tie_table = RouterScoreTable.build([ItemRouterScores(i, tie_scores, 0.0) for i in ("j0", "j1")], ["j0", "j1"])
    tie_draws = Bootstrap(("j0", "j1"), ("t1", "t1"), np.array([[1, 1], [2, 0], [0, 2]]))
    walked: Walk = walk(tie, tie_table, tie_draws, MIXES["realistic"], 1.0)
    check(
        walked.fell_back, "known positive: a walk over a policy that costs what A costs reads it as falling back to A"
    )
    routes = section({"100%": {"threshold": 0.9, "fell_back": False}, "99%": FELL, "97%": FELL})
    _, _, tied_rows, tied_tasks = evaluated(routes, tie, tie_table, tie_draws)
    check(tied_rows["realistic"]["100%"]["fell_back"] is False
          and tied_tasks["t1"]["realistic"]["100%"]["mcnemar"] is not None,
          "the frozen point at that threshold keeps the file's flag, not fallen back, and is tested against A")  # fmt: skip
    fell = section({"100%": FELL, "99%": FELL, "97%": FELL})
    _, outcomes, rows, tasks = evaluated(fell)
    a = fixed_outcomes(MATRIX)["opus"]
    row = rows["realistic"]["99%"]
    check(row["fell_back"] is True and row["threshold"] is None and close(row["cost_per_1k"]["value"], 10.0)
          and close(row["quality"]["value"], 0.75) and row["difference_from_A"]["value"] == 0
          and row["savings_vs_A"] == {"per_1k": 0.0, "share": 0.0} and row["captured"]["value"] == 0,
          "a point the file says fell back is A, though the table would route cheaply at any threshold")  # fmt: skip
    check(tasks["t1"]["realistic"]["99%"]["mcnemar"] is None and all(found == a for found in outcomes.values()),
          "it has no McNemar row, and its outcomes are A's")  # fmt: skip
    check(refused(FrozenPointError, lambda: evaluate(MATRIX, a, (FrozenPoint("x", "realistic", 0.99, None, None, False),),
                                                     router_traces(TABLE.traces))),
          "a router point that does not fall back and has no threshold is refused")  # fmt: skip


def main() -> int:
    reading()
    hand_worked()
    flag_decides()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
