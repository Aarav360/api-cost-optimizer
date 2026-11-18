"""The unlock and the handle: the one way to read test (PLAN_v4.md Amendment A19, items 3 and 4; issue #77, #78).

`unlock(world)` runs the unlock's checks in order and returns a `Handle`, or raises one `UnlockRefusal` naming the
check that failed, before the sealed store is opened. The handle is the only way to get test's requests and answers,
and every module that runs on test takes one. Its three reads, one per consumer:

- `requests()`: each test item's prompt and output schema, for the routers and the live replay;
- `answers()`: run 0's item ID, config, run, answer text and stop reason, for the gates (A17 item 2's `Answer`);
- `graded_items()`: each stored result with its item's expected answer and the whole response, for grading only.

A handle the unlock didn't issue is refused by every read (`HandleError`): `require_issued` is what a module that
only passes a handle on (a collector, a table builder) calls first.

This is the only module that reads the sealed store's path: `World.sealed_store`, which the go's entry point fills from
the collection module's own paths. `analysis/check_read_path.py` fails any other module that reads it, with this one
as its known positive, so the handle can't be bypassed unnoticed. The benchmark reader (`analysis/benchmark.py`)
and the results-matrix loader go on refusing a plain `"test"`; this module reads test's rows from the world's
benchmark file itself, as a separate function and not a flag (A9 item 7).

Adding to the unlock (#80, #81): a check is a function `(World) -> CheckResult` that raises `UnlockRefusal(name,
why)`. `CHECKS` runs in order, so the hook runner and the git facts (#80) go first, then the verify list (#81)
before the sealed store's own hash, which is the one check that reads the store's bytes. The unlock never opens
the store as a database, so a refusal leaves it unopened.

Its check is `analysis/check_unlock.py`.
"""

import hashlib
import json
import sqlite3
import weakref
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any

from analysis.gate_answers import ANSWER_COLUMNS, Answer
from analysis.world import World
from harness.client import PROJECT_ROOT
from harness.collect import MAIN
from routers.request import Request, from_row

MANIFEST = MAIN.manifest.relative_to(PROJECT_ROOT)  # where the collection manifest sits under a world's root
# Run 0 only, and the five columns of `ANSWER_COLUMNS` and no other: no expected answer, grade or token count can
# reach the gates (the leakage rule, A17 item 2). A check reads this SQL.
ANSWERS_QUERY = f"SELECT {', '.join(ANSWER_COLUMNS)} FROM answers WHERE run = 0 ORDER BY item_id, config"
# What grading needs of a stored result, beside its item: the request's model and effort, the request hash and the batch
# are what the collection run's row building reads from a job and its record.
GRADED_COLUMNS = (
    "custom_id", "item_id", "config", "model", "effort", "run", "batch_id", "request_hash", "created_at",
    "response_json",
)  # fmt: skip
GRADED_QUERY = f"SELECT {', '.join(GRADED_COLUMNS)} FROM answers ORDER BY item_id, config, run"


class UnlockRefusal(Exception):
    """The unlock refused: one check failed, and nothing was opened. `check` names it."""

    def __init__(self, check: str, why: str):
        super().__init__(f"the unlock refused at check {check!r}: {why}")
        self.check = check


class HandleError(Exception):
    """A read refused: the handle wasn't issued by the unlock, or the store and the benchmark disagree."""


@dataclass(frozen=True)
class CheckResult:
    """One passed check, as the record keeps it."""

    name: str
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GradedItem:
    """One stored result with what grading needs: the item's whole row (its expected answer among it) and the whole
    API response. For grading only: no router, gate or table before the tables step is given one."""

    custom_id: str
    item: dict
    config: str
    run: int
    model: str
    effort: str
    request_hash: str
    batch_id: str
    created_at: str
    response: dict


# The checks


def sealed_store_path(world: World) -> CheckResult:
    """A world names a sealed store, and it is a file. There is no default path to fall back on."""
    if world.sealed_store is None:
        raise UnlockRefusal("sealed_store_path", "the world names no sealed store")
    if not world.sealed_store.is_file():
        raise UnlockRefusal("sealed_store_exists", "the sealed store's path is not a file")
    return CheckResult("sealed_store_path")


def recorded_sha256(world: World) -> str:
    """The sha256 the collection manifest records for this store, or the refusal that it records none."""
    manifest_path = world.root / MANIFEST
    if not manifest_path.is_file():
        raise UnlockRefusal("manifest_record", "the world's root holds no collection manifest")
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))["test_sealed"]
        file_name, digest = record["file"], record["sha256"]
    except (ValueError, KeyError, TypeError) as error:
        raise UnlockRefusal(
            "manifest_record", f"the collection manifest holds no sealed-store record: {error!r}"
        ) from None
    assert world.sealed_store is not None  # `sealed_store_path` ran first
    if file_name != world.sealed_store.name:
        raise UnlockRefusal("manifest_record", "the collection manifest doesn't record this store")
    return str(digest)


def manifest_record(world: World) -> CheckResult:
    """The collection manifest records the store the world names, by file name."""
    return CheckResult("manifest_record", {"sha256": recorded_sha256(world)})


def sealed_store_sha256(world: World) -> CheckResult:
    """The store's bytes hash to the manifest's record: the one check that reads them, before the store is opened as
    a database. Nothing else in the project hashes the real store (AGENTS.md)."""
    recorded = recorded_sha256(world)
    assert world.sealed_store is not None
    digest = hashlib.sha256()
    with world.sealed_store.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != recorded:
        raise UnlockRefusal("sealed_store_sha256", "the sealed store's sha256 differs from the manifest's record")
    return CheckResult("sealed_store_sha256", {"sha256": recorded})


CHECKS: tuple[Callable[[World], CheckResult], ...] = (sealed_store_path, manifest_record, sealed_store_sha256)


# The handle

_ISSUED: "weakref.WeakSet[Handle]" = weakref.WeakSet()  # every handle the unlock returned, and no other


def require_issued(handle: object) -> "Handle":
    """The handle, if the unlock issued it. A copy, a subclass or a handle built by hand was not issued."""
    if type(handle) is not Handle or handle not in _ISSUED:
        raise HandleError("this handle was not issued by the unlock")
    return handle


class Handle:
    """What the unlock returns. It carries the world and the unlock's results (the record's checks), and offers the
    three reads. Build one only through `unlock`: any other is refused by every read."""

    __slots__ = ("world", "results", "__weakref__")

    def __init__(self, world: World, results: tuple[CheckResult, ...]):
        self.world = world
        self.results = results

    def _test_rows(self) -> list[dict]:
        """The benchmark's test rows in file order: the final evaluation's items."""
        with self.world.benchmark.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
        return [row for row in rows if row["split"] == "test"]

    def _stored(self, query: str) -> list[tuple]:
        assert self.world.sealed_store is not None  # the unlock refused a world without one
        with closing(sqlite3.connect(self.world.sealed_store.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            return conn.execute(query).fetchall()

    def requests(self) -> dict[str, Request]:
        """Each test item's request (prompt and output schema), by item ID: for the routers and the live replay."""
        require_issued(self)
        return {row["item_id"]: from_row(row) for row in self._test_rows()}

    def answers(self) -> list[Answer]:
        """Run 0's answers, one per item and config, in item and config order: for the gates. Five fields and no
        other: the item's expected answer and every grade stay out."""
        require_issued(self)
        stored = self._stored(ANSWERS_QUERY)
        self._refuse_strangers({row[0] for row in stored})
        return [Answer(*row) for row in stored]

    def graded_items(self) -> list[GradedItem]:
        """Every stored result with its item (the expected answer among its fields) and the whole response: for
        grading only."""
        require_issued(self)
        items = {row["item_id"]: row for row in self._test_rows()}
        stored = self._stored(GRADED_QUERY)
        self._refuse_strangers({row[1] for row in stored}, items)
        graded = []
        for custom_id, item_id, config, model, effort, run, batch_id, request_hash, created_at, response in stored:
            graded.append(GradedItem(custom_id, items[item_id], config, run, model, effort, request_hash, batch_id,
                                     created_at, json.loads(response)))  # fmt: skip
        return graded

    def _refuse_strangers(self, stored_ids: set[str], items: dict | None = None) -> None:
        """A stored item that is no test item of the benchmark refuses the read whole. The message names the item."""
        known = set(items) if items is not None else {row["item_id"] for row in self._test_rows()}
        if strangers := sorted(stored_ids - known):
            raise HandleError(f"{len(strangers)} stored items are no test item of the benchmark, e.g. {strangers[0]}")


def unlock(world: World) -> Handle:
    """Run every check in order and return the handle, or raise the first check's `UnlockRefusal`."""
    results = tuple(check(world) for check in CHECKS)
    handle = Handle(world, results)
    _ISSUED.add(handle)
    return handle
