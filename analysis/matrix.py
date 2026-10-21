"""The results matrix, held in memory (PLAN_v4.md §6; Amendment A8, items 2 and 5).

A matrix is every item's list-price cost and success for each config, one cell per
(item, config). It is built from plain rows, so every analysis function works the
same on the dev matrix and on small synthetic ones.

`load` reads the dev results matrix, the only store the analysis code reads, at the
path the collection runner exports. It refuses the whole store if any row is outside
train and dev, or if any row's cost recomputed from its tokens at the 2026-10-12 list
prices isn't exactly twice its stored batch cost (Amendment A8, item 2). Only run 0
goes into the matrix: runs 1 and 2 feed only the noise measure (Amendment A6, item 1).
"""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from analysis.benchmark import SPLITS, read_rows
from harness.client import cost_usd, load_models
from harness.collect import FRONTIER, MAIN
from routers.ladder import LADDER

# The config ladder lives in routers/ladder.py, which can't import the harness; this module re-exports it (so every
# existing import keeps working) and holds the assertion that it ends at the collection runner's frontier.
assert LADDER[-1] == FRONTIER

# A task (GLOSSARY.md) is the store's `source` column. Its `task_type` column holds a coarser label
# (coding, classification, ...) that the analysis doesn't use.
STORE_COLUMNS = ("item_id", "split", "source", "config", "model", "run", "input_tokens", "output_tokens", "cost_usd",
                 "success", "price_snapshot")  # fmt: skip


class MatrixError(ValueError):
    """The results store fails a loader check. Nothing is loaded."""


@dataclass(frozen=True)
class Item:
    item_id: str
    split: str
    task: str  # the benchmark source: ifeval, banking77, mmlu_pro or mbppplus
    source_group: str


@dataclass(frozen=True)
class Cell:
    cost: float  # dollars at list price
    # The item's score. Every task here scores 1 or 0, so it is the success rule's verdict (Amendment A5, item 5),
    # but §4 allows a score from 0 to 1, which the oracle's checks use.
    success: float


@dataclass(frozen=True)
class Matrix:
    items: dict[str, Item]
    cells: dict[tuple[str, str], Cell]  # keyed by (item_id, config)

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "Matrix":
        """Rows hold item_id, split, task, source_group, config, cost and success."""
        items, cells = {}, {}
        for row in rows:
            key = (row["item_id"], row["config"])
            if key in cells:
                raise MatrixError(f"two rows for {key}")
            # The oracle reads an item's configs off the ladder, so it would skip a cell for any other.
            if row["config"] not in LADDER:
                raise MatrixError(f"{key} is for a config off the ladder {LADDER}")
            items[row["item_id"]] = Item(row["item_id"], row["split"], row["task"], row["source_group"])
            cells[key] = Cell(row["cost"], row["success"])
        return cls(items, cells)

    def split(self, name: str) -> "Matrix":
        return self._only({item_id for item_id, item in self.items.items() if item.split == name})

    def task(self, name: str) -> "Matrix":
        return self._only({item_id for item_id, item in self.items.items() if item.task == name})

    def _only(self, kept: set[str]) -> "Matrix":
        items = {item_id: item for item_id, item in self.items.items() if item_id in kept}
        return Matrix(items, {key: cell for key, cell in self.cells.items() if key[0] in kept})


def read_store(path: Path = MAIN.matrix) -> list[dict]:
    """Every row of the dev results matrix, every run, with only the columns the analysis needs."""
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        return [dict(zip(STORE_COLUMNS, row)) for row in conn.execute(f"SELECT {', '.join(STORE_COLUMNS)} FROM rows")]


def list_price(row: dict, prices: dict) -> float:
    """A stored row's cost recomputed from its tokens at list price, by the harness's own formula."""
    return cost_usd({"request": {"model": row["model"]}, "response": {"usage": row}}, prices)


def from_store(rows: list[dict]) -> Matrix:
    """Check every stored row, then build the matrix from the run-0 rows."""
    models = load_models()
    # The benchmark gives each item's source group, through the reader, which yields train and dev items only.
    benchmark = {item["item_id"]: item for item in read_rows()}

    matrix_rows = []
    for row in rows:
        if row["split"] not in SPLITS:
            raise MatrixError(f"{row['item_id']} is a {row['split']} row; only train and dev rows are loaded")
        item = benchmark.get(row["item_id"])
        if item is None or (item["split"], item["source"]) != (row["split"], row["source"]):
            raise MatrixError(f"{row['item_id']} ({row['split']}) is not a {row['split']} item of the benchmark")
        if row["price_snapshot"] != models["price_snapshot"]:
            raise MatrixError(f"{row['item_id']} is priced at {row['price_snapshot']}, not {models['price_snapshot']}")
        cost = list_price(row, models["prices"])
        # Halving and doubling are exact in floating point, so this holds to the last bit.
        if cost != row["cost_usd"] / models["batch_discount"]:
            raise MatrixError(f"{row['item_id']} {row['config']} run {row['run']}: list price ${cost} is not "
                              f"twice the stored batch cost ${row['cost_usd']}")  # fmt: skip
        if row["run"] == 0:
            matrix_rows.append({"item_id": row["item_id"], "split": row["split"], "task": row["source"],
                                "source_group": item["source_group"], "config": row["config"], "cost": cost,
                                "success": row["success"]})  # fmt: skip
    return Matrix.from_rows(matrix_rows)


def load(path: Path = MAIN.matrix) -> Matrix:
    return from_store(read_store(path))
