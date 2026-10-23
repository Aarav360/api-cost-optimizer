"""The fold builder: dev's five folds and train's five cross-validation folds (PLAN_v4.md §8; Amendment A11, item 8).

Cross-fitting scores each dev item with a fold model that never saw its fold, and train's folds choose each config's C
(item 7). Both are built here, before any outcome is read, from item IDs, tasks and source groups read through the
benchmark reader and nothing else: this module never imports the results store, and `analysis/check_folds.py` works
that out from its imports.

- **Grouped by source group:** a group's items share a fold, so a fold model never trains on a near-duplicate of an
  item it scores.
- **Stratified by task:** each task's groups are dealt out on their own, always to the fold holding the fewest of that
  task's items, so a task's fold sizes differ by at most its largest group. Ties go to the fold with fewer items
  overall, then to the lower fold, which keeps the folds equal in total too.
- **Seeded with 2026:** the order the groups are dealt in is their sha256 of "seed:group", largest groups first. A hash
  order, not a shuffle, so the files don't depend on a Python version's random generator or on the benchmark file's
  row order.

Each split's assignment is a committed JSON file holding that split's item IDs only. Its sha256 goes into the freeze
(Amendment A11 item 19) and the file is not edited afterwards.

Rebuild both files from the project root:  uv run python -m analysis.folds
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from analysis.benchmark import read_rows

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # not harness.client's: that module imports the response store

SEED = 2026  # the bootstrap's seed (A11 item 8); analysis/check_folds.py compares the two
N_FOLDS = 5
FOLD_FILES = {
    "dev": PROJECT_ROOT / "data" / "dev_folds.json",
    "train": PROJECT_ROOT / "data" / "train_folds.json",
}


class FoldError(ValueError):
    """Rows or a fold file the builder refuses. Nothing is returned."""


def build_folds(split: str, path: Path | None = None) -> dict[str, int]:
    """Each item of the split mapped to its fold, 0 to 4, items in ID order.

    Reads the rows through the benchmark reader (`path` is a benchmark file for a check to feed it; the frozen
    benchmark by default), and uses only each row's item ID, task and source group. Refuses (FoldError) a split that
    isn't train or dev (the reader refuses it, and the test split by name), a source group in two tasks, or a task with fewer groups than folds, since a fold model would
    then learn from none of that task.
    """
    rows = read_rows(split=split) if path is None else read_rows(path, split=split)
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))  # task, then group, then items
    group_task: dict[str, str] = {}
    for row in rows:
        task, group = row["task_type"], row["source_group"]
        if group_task.setdefault(group, task) != task:
            raise FoldError(f"source group {group} is in tasks {group_task[group]} and {task}")
        groups[task][group].append(row["item_id"])
    if short := sorted(task for task, by_group in groups.items() if len(by_group) < N_FOLDS):
        raise FoldError(f"task {short[0]} has {len(groups[short[0]])} source groups, fewer than the {N_FOLDS} folds")
    fold_of: dict[str, int] = {}
    totals = [0] * N_FOLDS
    for task in sorted(groups):
        counts = [0] * N_FOLDS
        by_group = groups[task]
        for group in sorted(
            by_group, key=lambda g: (-len(by_group[g]), hashlib.sha256(f"{SEED}:{g}".encode()).hexdigest())
        ):
            fold = min(range(N_FOLDS), key=lambda k: (counts[k], totals[k], k))
            for item_id in by_group[group]:
                fold_of[item_id] = fold
            counts[fold] += len(by_group[group])
            totals[fold] += len(by_group[group])
    return dict(sorted(fold_of.items()))


def folds_text(split: str, fold_of: dict[str, int]) -> str:
    """The fold file's exact text: sorted keys, one item per line, so a diff and a hash are stable."""
    record = {"fold_of": dict(sorted(fold_of.items())), "n_folds": N_FOLDS, "seed": SEED, "split": split}
    return json.dumps(record, indent=1, sort_keys=True) + "\n"


def read_folds(split: str, path: Path | None = None) -> dict[str, int]:
    """An item ID's fold for every item of the split, from its fold file: `read_folds("dev")[item_id]`.

    Refuses (FoldError) a file of another split or fold count, or a fold outside 0 to 4.
    """
    if split not in FOLD_FILES:
        raise FoldError(f"split {split!r} has no folds: dev and train only")
    record = json.loads((FOLD_FILES[split] if path is None else path).read_text())
    if record.get("split") != split:
        raise FoldError(f"the fold file is for split {record.get('split')!r}, not {split!r}")
    if record.get("n_folds") != N_FOLDS:
        raise FoldError(f"the fold file has {record.get('n_folds')!r} folds, not {N_FOLDS}")
    fold_of = record.get("fold_of")
    if not isinstance(fold_of, dict) or not all(fold in range(N_FOLDS) for fold in fold_of.values()):
        raise FoldError(f"the fold file's fold_of is missing or holds a fold outside 0-{N_FOLDS - 1}")
    return fold_of


def main() -> int:
    for split, path in FOLD_FILES.items():
        fold_of = build_folds(split)
        text = folds_text(split, fold_of)
        path.write_text(text)
        digest = hashlib.sha256(text.encode()).hexdigest()
        print(f"{path.relative_to(PROJECT_ROOT)}: {len(fold_of)} items, sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
