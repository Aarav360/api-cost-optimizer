"""The dev task-inference diagnostic (PLAN_v4.md §8 3a; Amendment A9, items 5 and 13).

Run from the project root:  uv run python -m analysis.dev_diagnostic

How often the rules router's inferred type is the item's task, over the dev requests, with the output schema and with
it withheld. It uses the dev items' task labels, which the router never sees, so item 13 puts it after the dev
router-score table and before the walk. It is a statement about this benchmark's templates, not evidence that the cues
transfer to other traffic. The train diagnostic is item 17's, from `analysis/fit_rules_router.py`, whose function and
task-to-type map this reuses.
"""

from analysis.benchmark import read_rows
from analysis.fit_rules_router import diagnostic
from routers.request import Request, from_row


def dev_inputs() -> tuple[dict[str, Request], dict[str, str]]:
    """The dev requests and, for the diagnostic only, each item's task."""
    rows = [row for row in read_rows() if row["split"] == "dev"]
    return {row["item_id"]: from_row(row) for row in rows}, {row["item_id"]: row["source"] for row in rows}


def dev_diagnostic() -> str:
    requests, tasks = dev_inputs()
    return diagnostic(requests, tasks, split="dev")


def main() -> None:
    print(dev_diagnostic())


if __name__ == "__main__":
    main()
