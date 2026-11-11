"""The gates' answer reader (PLAN_v4.md §9; Amendment A17, item 2).

A gate sees a request and an answer. The answer is its final text (`answer`, text blocks only) and the API's stop
reason; thinking, token counts, latency and the config's name are left out. The results matrix stores each answer next
to its `success`, `quality`, `parsed` and `detail`, which hold the grader's verdict, so the leakage rule (§9) is kept in
the query: `QUERY` names the five columns of `ANSWER_COLUMNS` and no other, and a check reads the SQL the reader runs.

The split is chosen by the benchmark reader, not by a `split` column: it yields the items of train or dev and refuses
the test split (SealedItemError), and the rows kept are those of its items. Only run 0 is read (A17 item 7).
Requests come through the benchmark reader as before (`requests_of`).
"""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from analysis.benchmark import read_rows
from harness.collect import MAIN
from routers.request import Request, from_row

ANSWER_COLUMNS = ("item_id", "config", "run", "answer", "stop_reason")
# Run 0 only: runs 1 and 2 feed the flip rate and nothing else. `run` is one of the five columns, so the filter names
# no other.
QUERY = f"SELECT {', '.join(ANSWER_COLUMNS)} FROM rows WHERE run = 0"


@dataclass(frozen=True)
class Answer:
    """What a gate may see of a stored answer: whose it is, the text and the API's stop reason."""

    item_id: str
    config: str
    run: int
    answer: str
    stop_reason: str | None


def read_answers(split: str, path: Path = MAIN.matrix, benchmark_file: Path | None = None) -> list[Answer]:
    """Run 0's answers for the items of `split`, in the store's order. Refuses the test split (SealedItemError, from
    the benchmark reader) before the store is opened. `benchmark_file` is a benchmark for a check to feed in."""
    rows = read_rows(split=split) if benchmark_file is None else read_rows(benchmark_file, split=split)
    items = {row["item_id"] for row in rows}
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        stored = conn.execute(QUERY).fetchall()
    return [Answer(*row) for row in stored if row[0] in items]


def requests_of(split: str, benchmark_file: Path | None = None) -> dict[str, Request]:
    """The requests of `split`'s items, from the benchmark reader: prompt and output schema, nothing else."""
    rows = read_rows(split=split) if benchmark_file is None else read_rows(benchmark_file, split=split)
    return {row["item_id"]: from_row(row) for row in rows}
