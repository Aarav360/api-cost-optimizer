"""E's dev router-score table (PLAN_v4.md §8; Amendment A9, items 3, 6 and 13).

Run from the project root:  uv run python -m analysis.rules_table

Calls the rules router's own `score(request)` on every dev request from the reader and keeps the router scores in a
router-score table, so the table and the router can't drift apart. E runs locally, so every overhead is an explicit $0
(item 6). Building the table passes it through the table's refusal. This is the first of the three steps item 13 puts
after the freeze, in this order: this table, the dev task-inference diagnostic, the walk. It reads no outcome.
"""

from analysis.benchmark import read_rows
from analysis.router_table import ItemScores, RouterScoreTable
from routers import rules
from routers.ladder import SCORED
from routers.request import from_row

OVERHEAD = 0.0  # E runs locally: no API call, no dollars (item 6)


def dev_table() -> RouterScoreTable:
    """The table over every dev item, in the reader's order, or the table's refusal (TableError)."""
    rows = [row for row in read_rows() if row["split"] == "dev"]
    items = [ItemScores(row["item_id"], rules.score(from_row(row)), OVERHEAD) for row in rows]
    return RouterScoreTable.build(items, [row["item_id"] for row in rows])


def main() -> None:
    table = dev_table()
    distinct = {tuple(entry.scores[config] for config in SCORED) for entry in table.items}
    print(f"E's dev router-score table: {len(table.items)} dev requests, each scored by score(request), overhead $0, "
          f"{len(distinct)} distinct triples of router scores; it passes the table's refusal")  # fmt: skip


if __name__ == "__main__":
    main()
