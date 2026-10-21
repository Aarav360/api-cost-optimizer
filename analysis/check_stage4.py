"""Check E on dev and Stage 4's outputs (PLAN_v4.md §8, §12; Amendment A9, items 3, 13-15 and 18).

Run from the project root:  uv run python -m analysis.check_stage4

Needs the dev results matrix in results/ (not committed), so the hook runs it where the matrix is present, like
check_simulator. It reads train and dev only, through the reader, and never a test item.

- E's dev router-score table is `score(request)` over every dev request from the reader, with $0 overhead, and it
  passes the table's refusal; the refusal itself is shown on a table that leaves out an item and on one that scores
  the frontier;
- the dev task-inference diagnostic (item 5) is a recount of the inferred type against each dev item's task, in the
  words item 13 asks for ("on this benchmark's templates"), and it can fail: with the tasks shifted it changes;
- the walk on dev, for both mixes and all three targets, reproduces E's section of the operating-points file to the
  last bit (and so its hash); in that file every certified step clears its target by its margin, the walk stops at
  the first step that doesn't, and the operating point is the cheapest certified step by the file's own costs.
"""

import sys
from collections.abc import Callable
from typing import Any

from analysis import matrix as results_matrix
from analysis.benchmark import read_rows
from analysis.dev_diagnostic import dev_diagnostic, dev_inputs
from analysis.fit_rules_router import diagnostic
from analysis.operating_points import read_section, section_hash
from analysis.references import ROUNDING, TARGETS
from analysis.router_table import ItemScores, RouterScoreTable, TableError
from analysis.rules_table import dev_table
from analysis.rules_walk import ROUTER, rebuilt_section
from routers import rules
from routers.ladder import FRONTIER, SCORED
from routers.request import Request, from_row

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object]) -> bool:
    try:
        call()
    except TableError:
        return True
    return False


def main() -> int:
    print("-- E's dev router-score table (Amendment A9, item 3) --")
    rows = [row for row in read_rows() if row["split"] == "dev"]
    dev_ids = list(results_matrix.load().split("dev").items)
    table = dev_table()
    check(len(table.items) == len(rows) == 744, f"the table has an entry for each of the {len(rows)} dev requests")
    check([entry.item_id for entry in table.items] == [row["item_id"] for row in rows],
          "its entries are in the reader's dev order")  # fmt: skip
    check(set(entry.item_id for entry in table.items) == set(dev_ids),
          "they are the dev results matrix's items")  # fmt: skip
    check(all(entry.scores == rules.score(from_row(row)) for entry, row in zip(table.items, rows)),
          "every entry's router scores are score(request) for that dev request, from the reader")  # fmt: skip
    check(all(set(entry.scores) == set(SCORED) for entry in table.items),
          f"every entry scores {', '.join(SCORED)} and never {FRONTIER}")  # fmt: skip
    check(all(entry.overhead == 0.0 for entry in table.items), "every overhead is exactly $0: E runs locally")
    check(RouterScoreTable.build(list(table.items), dev_ids).items == table.items,
          "the table passes the table's refusal against the matrix's dev items")  # fmt: skip
    check(refused(lambda: RouterScoreTable.build(list(table.items)[1:], dev_ids)),
          "the refusal fires on a table that leaves out a dev item")  # fmt: skip
    first = table.items[0]
    scored_frontier = ItemScores(first.item_id, {**first.scores, FRONTIER: 0.5}, 0.0)
    check(refused(lambda: RouterScoreTable.build([scored_frontier, *table.items[1:]], dev_ids)),
          "the refusal fires on a table that scores the frontier")  # fmt: skip

    print("\n-- the dev task-inference diagnostic (Amendment A9, items 5 and 13) --")
    # Each task's own type, written here from the plan's item 5 and not taken from the script under check.
    own_type = {"banking77": rules.CLASSIFICATION, "mmlu_pro": rules.MULTIPLE_CHOICE, "mbppplus": rules.CODE,
                "ifeval": rules.OPEN_ENDED}  # fmt: skip
    requests, tasks = dev_inputs()
    full = sum(rules.infer_type(request) == own_type[tasks[i]] for i, request in requests.items())
    bare = sum(rules.infer_type(Request(request.prompt)) == own_type[tasks[i]] for i, request in requests.items())
    said = dev_diagnostic()
    print(f"      {said}")
    check(len(requests) == len(rows) and set(requests) == set(dev_ids), "it covers exactly the dev items")
    check(said.startswith("On this benchmark's templates the inferred type is the item's task for ")
          and f"for {full} of {len(requests)} dev items" in said and f"and for {bare} of {len(requests)}" in said
          and said.endswith("with the output schema withheld"),
          "it states the counts a recount gives, with the schema and with it withheld, on this benchmark's templates")  # fmt: skip
    ordered = sorted(tasks)
    shuffled = dict(zip(ordered, [tasks[i] for i in ordered[1:] + ordered[:1]]))
    check(f"for {full} of" not in diagnostic(requests, shuffled, split="dev"),
          "the diagnostic can fail: with the tasks shifted by one item the count changes")  # fmt: skip

    print("\n-- the walk on dev and the operating-points file (Amendment A9, items 9, 10 and 13) --")
    section = read_section(ROUTER)
    check(list(section) == ["realistic", "uniform"] and all(set(by_target) == {"100%", "99%", "97%"}
                                                           for by_target in section.values()),
          "E's section holds both mixes and all three targets: six operating points")  # fmt: skip
    rebuilt = rebuilt_section()
    check(rebuilt == section and section_hash(rebuilt) == section_hash(section),
          "the walk on dev reproduces E's section of the operating-points file, so its hash too")  # fmt: skip
    for mix, by_target in section.items():
        for key, point in by_target.items():
            steps: list[dict[str, Any]] = point["steps"]
            target = point["target"]
            where = f"{mix} mix, {key} of A"
            stop = next((i for i, step in enumerate(steps) if not step["clears"]), len(steps))
            certified = steps[:stop]
            check(steps[0]["threshold"] is None and steps[0]["margin"] == 0 and float(key.rstrip("%")) / 100 in TARGETS,
                  f"{where}: the walk starts at A, threshold null, margin 0")  # fmt: skip
            check(all(step["quality"] - target >= step["margin"] - ROUNDING for step in certified),
                  f"{where}: all {len(certified)} certified steps (A included) clear the target by their margin")  # fmt: skip
            # A step that clears is certified only if every step before it did; the walk ends at the first miss.
            missed = steps[stop:]
            check(not missed or (len(missed) == 1 and missed[0]["quality"] - target < missed[0]["margin"] - ROUNDING),
                  f"{where}: the walk ends at the first step that misses its target, or tried them all")  # fmt: skip
            thresholds = [step["threshold"] for step in steps[1:]]
            check(thresholds == sorted(thresholds, reverse=True),
                  f"{where}: the steps run from the highest threshold down")  # fmt: skip
            cheapest = certified[0]
            for step in certified[1:]:
                # Costs within rounding of each other are a tie, which goes to the earlier step.
                if step["cost_per_1k"] < cheapest["cost_per_1k"] - ROUNDING:
                    cheapest = step
            check(point["threshold"] == cheapest["threshold"] and point["margin"] == cheapest["margin"]
                  and point["fell_back"] == (point["threshold"] is None),
                  f"{where}: the operating point is the cheapest certified step: "
                  + ("A, it falls back" if point["fell_back"] else f"threshold {point['threshold']:.3f}"))  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
