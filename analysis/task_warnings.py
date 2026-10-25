"""A7 item 2's warning on a small task's per-task results (PLAN_v4.md §5; Amendment A7, item 2).

Fewer items than the "few hundred" §5 asks for before a per-task claim: IFEval and MBPP+ here. The chart script and the
diagnostics both attach it, so the text and the list of tasks live once.
"""

WARNED = ("ifeval", "mbppplus")  # in A5's order


def warning(items: int) -> str:
    """A7 item 2's warning on a small task's per-task results."""
    return f"Only {items} dev items, below the plan's minimum of a few hundred: indicative"


def task_warning(task: str, items: int) -> str | None:
    """The warning for `task`'s `items` dev items, or None for the tasks that carry none."""
    return warning(items) if task in WARNED else None
