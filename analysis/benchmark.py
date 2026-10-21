"""The benchmark reader: the only path from the benchmark to analysis code and routers (PLAN_v4.md §5–6; Amendment
A6; Amendment A9, item 7).

It yields train and dev rows only. A request for a test item is refused whole (SealedItemError), before any row is
returned, so a router can't be handed a test request by accident. The benchmark file is a parameter, defaulting to
the frozen benchmark, so a check can feed it a synthetic file. Stage 8's unlock is a separate function built in Stage
8, not a flag here. `harness/check_sealed.py` allows no other analysis module to name the file.
"""

import json
from collections.abc import Collection
from pathlib import Path

from harness.build_benchmark import BENCHMARK_FILE

SPLITS = ("train", "dev")  # test stays sealed until Stage 8 (Amendment A6)


class BenchmarkError(ValueError):
    """A request the reader refuses. Nothing is returned."""


class SealedItemError(BenchmarkError):
    """A request for a test item, which stays sealed until Stage 8."""


def read_rows(
    path: Path = BENCHMARK_FILE, item_ids: Collection[str] | None = None, split: str | None = None
) -> list[dict]:
    """The train and dev rows of the benchmark in file order, or just the named items' rows, or just one split's.

    Refuses the whole request (SealedItemError) if any named item is a test item or the split asked for is test,
    and (BenchmarkError) if any named item is not in the file, the split is not train or dev, or the file holds a
    row of a split other than train, dev or test.
    """
    if split == "test":
        raise SealedItemError("the test split stays sealed until Stage 8")
    if split is not None and split not in SPLITS:
        raise BenchmarkError(f"split {split!r} is not train or dev")
    allowed, sealed = {}, set()
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            if row["split"] in SPLITS:
                if split in (None, row["split"]):
                    allowed[row["item_id"]] = row
            elif row["split"] == "test":
                sealed.add(row["item_id"])
            else:
                raise BenchmarkError(f"{row['item_id']} is in split {row['split']!r}, not train, dev or test")
    if item_ids is None:
        return list(allowed.values())
    if asked := sorted(set(item_ids) & sealed):
        raise SealedItemError(f"{len(asked)} of the items asked for are test items, e.g. {asked[0]}; they stay "
                              f"sealed until Stage 8")  # fmt: skip
    if absent := sorted(set(item_ids) - set(allowed)):
        raise BenchmarkError(f"{len(absent)} of the items asked for are not among its {split or 'train and dev'} "
                             f"rows, e.g. {absent[0]}")  # fmt: skip
    return [allowed[item_id] for item_id in dict.fromkeys(item_ids)]
