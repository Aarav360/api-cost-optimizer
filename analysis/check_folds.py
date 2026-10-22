"""Check the fold builder and its two committed files (PLAN_v4.md §8; Amendment A11, item 8).

Run from the project root:  uv run python -m analysis.check_folds

Synthetic benchmark files with answers worked out by hand, then the committed benchmark and the committed fold files.
No results store is opened, and no test item is read.

- on synthetic files: every item of the split gets one fold in 0-4 and no other item appears; no source group is split
  across folds; each task's fold sizes are within one group of each other (a task of singleton groups: within one
  item); a second build from the seed, and a build from the rows in another order, equal the first; another seed
  gives another assignment; the guards below can fail (a split group and an unbalanced task, fed to them);
- refusals, each whole: a task with fewer groups than folds, a source group spanning two tasks, the test split, a
  split that isn't train or dev, a file with a malformed fold file's shape;
- on the committed benchmark: the committed dev and train fold files equal a fresh build, byte for byte; each holds
  its own split's item IDs and nothing else; the seed is the bootstrap's;
- the fold builder's imports never reach the results store: the closure of its imports, stopping at the benchmark
  reader (the sanctioned door, guarded by harness/check_sealed.py), names none of the store's modules, with a module
  that does as the known positive, directly and through another module.
"""

import ast
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from analysis import bootstrap, folds
from analysis.benchmark import BenchmarkError, read_rows
from analysis.folds import N_FOLDS, SEED, FoldError, build_folds, folds_text, read_folds
from harness.client import PROJECT_ROOT

# The modules that open the results stores (the matrix reader, the collection runner that writes both stores, the
# response store) and the library that opens them.
STORE_MODULES = {"analysis.matrix", "harness.collect", "harness.store", "sqlite3"}
READER = "analysis.benchmark"  # the fold builder's one door to the benchmark
CODE_PACKAGES = ("analysis", "harness", "routers")

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object], error: type[Exception] | tuple[type[Exception], ...] = FoldError) -> bool:
    try:
        call()
    except error:
        return True
    return False


def row(item_id: str, task: str, group: str, split: str = "dev") -> dict:
    return {"item_id": item_id, "task_type": task, "source_group": group, "split": split}


def write_rows(directory: Path, rows: list[dict], name: str = "synthetic.jsonl") -> Path:
    path = directory / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def synthetic_rows() -> list[dict]:
    """Dev: task a has 23 groups (4 of two items), task b 12 groups (3 of three), task c 5 singletons, task d 10
    singletons. Train and test rows ride along and must not leak into dev's folds."""
    rows = []
    for g in range(23):
        for k in range(2 if g < 4 else 1):
            rows.append(row(f"a:{g}.{k}", "a", f"a:{g}"))
    for g in range(12):
        for k in range(3 if g < 3 else 1):
            rows.append(row(f"b:{g}.{k}", "b", f"b:{g}"))
    rows += [row(f"c:{g}", "c", f"c:{g}") for g in range(5)]
    rows += [row(f"d:{g}", "d", f"d:{g}") for g in range(10)]
    rows += [row(f"t:{g}", "a", f"t:{g}", "train") for g in range(9)]
    rows += [row(f"x:{g}", "a", f"x:{g}", "test") for g in range(9)]
    return rows


def problems_with(fold_of: dict[str, int], rows: list[dict]) -> list[str]:
    """What is wrong with an assignment of these rows (all one split): the independent statement of A11 item 8's
    rules, which the builder is held to. Empty when the assignment is as stated."""
    found = []
    if set(fold_of) != {r["item_id"] for r in rows}:
        found.append("items assigned are not the rows' items")
    if not set(fold_of.values()) <= set(range(N_FOLDS)):
        found.append("a fold outside 0-4")
    group_folds: dict[str, set[int]] = defaultdict(set)
    group_sizes: dict[str, int] = defaultdict(int)
    for r in rows:
        group_folds[r["source_group"]].add(fold_of[r["item_id"]])
        group_sizes[r["source_group"]] += 1
    found += [f"group {g} is split across folds" for g, f in group_folds.items() if len(f) > 1]
    for task in sorted({r["task_type"] for r in rows}):
        in_task = [r for r in rows if r["task_type"] == task]
        sizes = [sum(1 for r in in_task if fold_of[r["item_id"]] == k) for k in range(N_FOLDS)]
        biggest = max(group_sizes[r["source_group"]] for r in in_task)
        if max(sizes) - min(sizes) > biggest:
            found.append(f"task {task}: fold sizes {sizes} differ by more than one group ({biggest})")
    return found


def task_sizes(fold_of: dict[str, int], rows: list[dict], task: str) -> list[int]:
    return [sum(1 for r in rows if r["task_type"] == task and fold_of[r["item_id"]] == k) for k in range(N_FOLDS)]


def imports_of(root: Path, module: str) -> set[str]:
    """Every module name a file imports, each with its parent packages, `from a import b` also as `a.b`."""
    names: set[str] = set()
    for node in ast.walk(ast.parse((root.joinpath(*module.split(".")).with_suffix(".py")).read_text())):
        if isinstance(node, ast.Import):
            found = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        else:
            continue
        for name in found:
            parts = name.split(".")
            names |= {".".join(parts[:end]) for end in range(1, len(parts) + 1)}
    return names


def reaches(root: Path, entry: str) -> set[str]:
    """Every module the entry module's imports lead to, in the project's own packages followed file by file and
    stopping at the benchmark reader, plus the third-party and standard library names met on the way."""
    seen: set[str] = set()
    todo = [entry]
    while todo:
        module = todo.pop()
        for name in imports_of(root, module):
            if name in seen:
                continue
            seen.add(name)
            if (
                name != READER
                and name.split(".")[0] in CODE_PACKAGES
                and root.joinpath(*name.split(".")).with_suffix(".py").exists()
            ):
                todo.append(name)
    return seen


def main() -> int:
    print("-- synthetic benchmark file --")
    rows = synthetic_rows()
    dev_rows = [r for r in rows if r["split"] == "dev"]
    with tempfile.TemporaryDirectory() as tmp:
        path = write_rows(Path(tmp), rows)
        built = build_folds("dev", path)
        check(
            problems_with(built, dev_rows) == [],
            f"dev's build is as A11 item 8 states: {problems_with(built, dev_rows)}",
        )
        check(len(built) == len(dev_rows), f"every dev item has a fold and no other item does ({len(built)})")
        # Hand-worked: 5 singleton groups put one item in every fold; 10 put two.
        check(
            task_sizes(built, dev_rows, "c") == [1] * 5,
            f"5 singleton groups: one item per fold {task_sizes(built, dev_rows, 'c')}",
        )
        check(
            task_sizes(built, dev_rows, "d") == [2] * 5,
            f"10 singleton groups: two items per fold {task_sizes(built, dev_rows, 'd')}",
        )
        check(max(task_sizes(built, dev_rows, "a")) - min(task_sizes(built, dev_rows, "a")) <= 1, f"task a (27 items, 4 pairs): sizes within one item here {task_sizes(built, dev_rows, 'a')}")  # fmt: skip
        check(build_folds("dev", path) == built, "a second build from the seed equals the first")
        check(list(build_folds("dev", path)) == list(built), "and lists the items in the same order")
        reordered = write_rows(Path(tmp), rows[::-1], "reordered.jsonl")
        check(build_folds("dev", reordered) == built, "a build from the rows in another order equals the first")
        check(build_folds("train", path) != build_folds("dev", path), "train's build is its own (different items)")
        train_rows = [r for r in rows if r["split"] == "train"]
        check(set(build_folds("train", path)) == {r["item_id"] for r in train_rows}, "train's build holds train's items only")  # fmt: skip
        original_seed = folds.SEED
        folds.SEED = 7
        try:
            other = build_folds("dev", path)
        finally:
            folds.SEED = original_seed
        check(
            other != built and problems_with(other, dev_rows) == [],
            "another seed gives another assignment, still as stated",
        )
        check(folds_text("dev", built) == folds_text("dev", dict(sorted(built.items(), reverse=True))), "the file text doesn't depend on the order built")  # fmt: skip

        print("\n-- the guards can fail --")
        pair = next(r for r in dev_rows if r["source_group"] == "a:0" and r["item_id"] != "a:0.0")
        split_group = dict(built)
        split_group[pair["item_id"]] = (built[pair["item_id"]] + 1) % N_FOLDS
        check(
            any("split across folds" in p for p in problems_with(split_group, dev_rows)),
            "a group split across folds is found",
        )
        lopsided = {r["item_id"]: 0 for r in dev_rows if r["task_type"] == "d"}
        check(any("task d" in p for p in problems_with(lopsided, [r for r in dev_rows if r["task_type"] == "d"])), "an unbalanced task is found")  # fmt: skip
        check(any("not the rows" in p for p in problems_with({**built, "stray": 0}, dev_rows)), "a stray item is found")

        print("\n-- refusals --")
        few = [row(f"e:{g}", "e", f"e:{g}") for g in range(4)]
        check(
            refused(lambda: build_folds("dev", write_rows(Path(tmp), few, "few.jsonl"))),
            "a task with 4 groups can't fill 5 folds",
        )
        spanning = [*dev_rows, row("s:1", "b", "a:0")]
        check(refused(lambda: build_folds("dev", write_rows(Path(tmp), spanning, "spanning.jsonl"))), "a source group in two tasks is refused")  # fmt: skip
        check(refused(lambda: build_folds("test", path), BenchmarkError), "the test split is refused by the reader")
        check(refused(lambda: build_folds("pilot", path), (BenchmarkError, FoldError)), "a split other than train or dev is refused")  # fmt: skip
        good = json.loads(folds_text("dev", built))
        for what, bad in {
            "a fold outside 0-4": {**good, "fold_of": {**good["fold_of"], dev_rows[0]["item_id"]: 5}},
            "another fold count": {**good, "n_folds": 10},
            "the wrong split": {**good, "split": "train"},
            "no fold_of": {k: v for k, v in good.items() if k != "fold_of"},
        }.items():
            badfile = Path(tmp) / "bad.json"
            badfile.write_text(json.dumps(bad))
            check(refused(lambda: read_folds("dev", badfile)), f"a fold file with {what} is refused")
        goodfile = Path(tmp) / "good.json"
        goodfile.write_text(folds_text("dev", built))
        check(read_folds("dev", goodfile) == built, "a fold file reads back as written")

    print("\n-- the committed benchmark and fold files --")
    check(SEED == bootstrap.SEED == 2026, f"the seed is 2026, the bootstrap's ({SEED})")
    for split in ("dev", "train"):
        committed = folds.FOLD_FILES[split]
        split_rows = read_rows(split=split)
        fresh = build_folds(split)
        check(
            committed.read_text() == folds_text(split, fresh), f"{committed.name} equals a fresh build, byte for byte"
        )
        on_file = read_folds(split)
        check(set(on_file) == {r["item_id"] for r in split_rows}, f"{committed.name} holds {split}'s {len(split_rows)} item IDs and no other")  # fmt: skip
        check(problems_with(on_file, split_rows) == [], f"{committed.name}: no group split across folds, each task's sizes within one group")  # fmt: skip
        for task in sorted({r["task_type"] for r in split_rows}):
            print(f"      {split} {task}: {task_sizes(on_file, split_rows, task)}")
        print(f"      {committed.name} sha256 {hashlib.sha256(committed.read_bytes()).hexdigest()}")

    print("\n-- the fold builder never reaches the results store --")
    reached = reaches(PROJECT_ROOT, "analysis.folds")
    check(READER in reached, f"the closure sees the benchmark reader ({READER}), so it isn't empty")
    check(not reached & STORE_MODULES, f"none of {sorted(STORE_MODULES)} is reached: {sorted(reached & STORE_MODULES)}")
    check(
        bool(reaches(PROJECT_ROOT, "analysis.matrix") & STORE_MODULES),
        "known positive: analysis.matrix reaches the store",
    )
    check(bool(reaches(PROJECT_ROOT, "analysis.walk") & STORE_MODULES), "known positive: analysis.walk reaches it through other modules")  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "analysis"
        package.mkdir()
        (package / "leaky.py").write_text("import sqlite3\n")
        (package / "via.py").write_text("from analysis import leaky\n")
        (package / "clean.py").write_text("import json\nfrom analysis.benchmark import read_rows\n")
        check("sqlite3" in reaches(Path(tmp), "analysis.leaky"), "known positive: a module importing sqlite3")
        check("sqlite3" in reaches(Path(tmp), "analysis.via"), "known positive: a module reaching it through another")
        check(not reaches(Path(tmp), "analysis.clean") & STORE_MODULES, "a clean module is not flagged")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
