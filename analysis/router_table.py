"""The router-score table (PLAN_v4.md §8; Amendment A9, item 3).

A table holds a router's router scores for `haiku`, `sonnet` and `opus_low`, and its overhead in dollars at list
price, for every item of a split. The walk and the curves read only the table, never a router: `traces` builds the
router's traces at any threshold from it, one config per item by the decision rule, each carrying the table's
overhead.

`RouterScoreTable.build` refuses the whole table (TableError), before anything is computed, if it misses or repeats
an item of the split, scores the frontier, leaves out a config below it, has a router score that isn't a number from 0
to 1, or has an overhead that is negative or not finite (as the simulator refuses traces, Amendment A8, item 4).
"""

import math
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass

from analysis.simulator import Trace
from routers.decision import choose
from routers.ladder import SCORED


class TableError(ValueError):
    """A router-score table that fails a check. Nothing is built."""


@dataclass(frozen=True)
class ItemScores:
    item_id: str
    scores: dict[str, float]  # router score for each config below the frontier
    overhead: float  # the router's cost for this item, dollars at list price


def is_number(value: object) -> bool:
    # A bool is an int in Python, and a string or NaN would spoil every comparison with a threshold.
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class RouterScoreTable:
    items: tuple[ItemScores, ...]  # in the split's order

    @classmethod
    def build(cls, items: list[ItemScores], split_item_ids: Collection[str]) -> "RouterScoreTable":
        """The table for the items of one split, or a refusal (TableError)."""
        counts = Counter(entry.item_id for entry in items)
        if repeated := sorted(item_id for item_id, n in counts.items() if n > 1):
            raise TableError(f"{len(repeated)} items are in the table more than once, e.g. {repeated[0]}")
        if unknown := sorted(set(counts) - set(split_item_ids)):
            raise TableError(f"{len(unknown)} items are not in the split, e.g. {unknown[0]}")
        if missing := sorted(set(split_item_ids) - set(counts)):
            raise TableError(f"{len(missing)} items of the split are not in the table, e.g. {missing[0]}")
        for entry in items:
            # The frontier is the fallback and needs no score (Amendment A1, item 3), so a table that scored it
            # would be read as a router that had; any other key is just as unexpected.
            if extra := sorted(set(entry.scores) - set(SCORED)):
                raise TableError(f"{entry.item_id} has a router score for {extra}, which are not configs below "
                                 f"the frontier")  # fmt: skip
            if left_out := [config for config in SCORED if config not in entry.scores]:
                raise TableError(f"{entry.item_id} has no router score for {left_out}")
            for config, score in entry.scores.items():
                if not (is_number(score) and 0 <= score <= 1):
                    raise TableError(f"{entry.item_id}'s router score for {config} is {score!r}, which must be a "
                                     f"number from 0 to 1")  # fmt: skip
            if not (is_number(entry.overhead) and entry.overhead >= 0):
                raise TableError(f"{entry.item_id}'s overhead is {entry.overhead!r}, which must be a finite number "
                                 f"of dollars, at least 0")  # fmt: skip
        return cls(tuple(items))

    def traces(self, threshold: float) -> list[Trace]:
        """One trace per item at a threshold: its config by the decision rule, and the table's overhead."""
        return [Trace(entry.item_id, (choose(entry.scores, threshold),), entry.overhead) for entry in self.items]
