"""Check E on dev and Stage 4's outputs (PLAN_v4.md §8, §12; Amendment A9, items 3, 13-15 and 18).

Run from the project root:  uv run python -m analysis.check_stage4

Needs the dev results matrix in results/ (not committed), so the hook runs it where the matrix is present, like
check_simulator. It reads train and dev only, through the reader, and never a test item.

- E's dev router-score table is `score(request)` over every dev request from the reader, with $0 overhead, and it
  passes the table's refusal; the refusal itself is shown on a table that leaves out an item and on one that scores
  the frontier.
"""

import sys
from collections.abc import Callable

from analysis import matrix as results_matrix
from analysis.benchmark import read_rows
from analysis.router_table import ItemScores, RouterScoreTable, TableError
from analysis.rules_table import dev_table
from routers import rules
from routers.ladder import FRONTIER, SCORED
from routers.request import from_row

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

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
