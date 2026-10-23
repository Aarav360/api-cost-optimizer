"""Check the trained router's fit commit (PLAN_v4.md §8 3b; Amendment A11, items 15 and 20).

Run from the project root:  uv run python -m analysis.check_trained_fit

Item 20 against the committed fold-weights file, then the refit. Never a test item. The refit reads dev outcomes and
this script prints nothing about them: only whether the refit weights match, and how far apart they are.

- the file (also in CI): five fold models of three heads, each as wide as the standardizer; its sha256 is item 20's,
  exactly; item 20's row counts are train's rows plus the dev rows outside each fold;
- the git history (where the freeze commit is present; CI's shallow clone lacks it): item 20's freeze commit is an
  ancestor of the commit that added the weights, it held every file item 19 hashes as item 19 records them, and it
  held no fold weights, so the order is provable from the log;
- the refit (the hook only, since it needs the caches, the model snapshot's settings and results/): the five fold
  models refit from the frozen inputs match the file to 1e-9, with known positives (a head against another fold
  model's, and the file with one byte changed).
"""

import hashlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from analysis import fit_trained_router as fit
from analysis import matrix as results_matrix
from analysis.benchmark import read_rows
from analysis.folds import N_FOLDS, read_folds
from analysis.plan import PLAN, amendment_item
from harness.client import PROJECT_ROOT
from routers import trained
from routers.ladder import SCORED
from routers.request import from_row

HASHED = re.compile(r"`([\w./-]+)`: `([0-9a-f]{64})`")  # a path and its sha256, as items 19 and 20 write them
FREEZE = re.compile(r"\*\*The freeze commit\*\*: `([0-9a-f]{40})`")
ROWS = re.compile(r"fold model (\d), (\d+)")
TOLERANCE = 1e-9  # the refit tolerates the order sums are added in (item 15)

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)


def largest_gap(a: Sequence[Mapping[str, trained.Head]], b: Sequence[Mapping[str, trained.Head]]) -> float:
    """The largest absolute difference between two sets of fold models, over every coefficient and intercept."""
    return max(
        max(float(np.abs(x[c].coef - y[c].coef).max()), abs(x[c].intercept - y[c].intercept))
        for x, y in zip(a, b, strict=True)
        for c in SCORED
    )


def refit(settings: trained.Settings) -> tuple[list[dict[str, trained.Head]], list[int]]:
    """The fold models as `python -m analysis.fit_trained_router folds` fits them, from the same inputs."""
    matrix = results_matrix.load()
    inputs = {}
    for split in ("train", "dev"):
        rows = read_rows(split=split)
        ids = [row["item_id"] for row in rows]
        cache = fit.read_cache(split)
        x = fit.feature_matrix(settings, cache.embeddings, [from_row(row) for row in rows])
        cells = matrix.split(split).cells
        inputs[split] = (ids, x, {config: np.array([cells[(i, config)].success for i in ids]) for config in SCORED})
    dev_fold_of = read_folds("dev")
    dev_folds = np.array([dev_fold_of[i] for i in inputs["dev"][0]])
    return fit.fit_fold_models(
        inputs["train"][1], inputs["train"][2], inputs["dev"][1], inputs["dev"][2], dev_folds, settings.c
    )


def main() -> int:
    record = amendment_item(PLAN.read_text(), "A11", 20)
    recorded = dict(HASHED.findall(record))
    path = trained.FOLD_WEIGHTS_FILE
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    settings = trained.read_settings()

    print("-- the weights file against item 20 --")
    check(
        hashlib.sha256(path.read_bytes()).hexdigest() == recorded.get(relative),
        f"{relative} hashes exactly as item 20 records it",
    )
    weights = trained.read_fold_weights(path)
    width = len(settings.mean) + 1
    check(
        len(weights) == N_FOLDS
        and all(set(heads) == set(SCORED) and all(len(h.coef) == width for h in heads.values()) for heads in weights),
        f"the file holds {N_FOLDS} fold models of {len(SCORED)} heads, each {width} wide",
    )
    check(
        path.read_text() == fit.fold_weights_text(list(weights)),
        "the file is in the fit script's own format, so reading it and writing it again changes nothing",
    )
    dev_fold_of = read_folds("dev")
    train_rows = len(read_rows(split="train"))
    expected = [train_rows + sum(1 for fold in dev_fold_of.values() if fold != k) for k in range(N_FOLDS)]
    check(
        [(int(k), int(n)) for k, n in ROWS.findall(record)] == list(enumerate(expected)),
        f"item 20's training rows are train's {train_rows} plus the dev rows outside each fold: {expected}",
    )
    check(
        sum(expected) - N_FOLDS * train_rows == 4 * len(dev_fold_of),
        "every dev item is in the training rows of exactly four of the five fold models",
    )
    check("prints nothing about them" in record, "item 20 says the fit prints nothing about dev outcomes")

    print("\n-- the order in the git history (where the freeze commit is present) --")
    freeze = FREEZE.search(record)
    check(freeze is not None, "item 20 names the freeze commit by its full hash")
    if freeze and git("cat-file", "-e", f"{freeze[1]}^{{commit}}").returncode == 0:
        added = git("log", "--diff-filter=A", "--format=%H", "--", relative).stdout.split()
        check(len(added) == 1, "one commit added the fold weights")
        if added:
            check(
                git("merge-base", "--is-ancestor", freeze[1], added[0]).returncode == 0,
                "item 20's freeze commit is an ancestor of the commit that added the fold weights",
            )
        check(
            git("cat-file", "-e", f"{freeze[1]}:{relative}").returncode != 0,
            "the freeze commit held no fold weights",
        )
        item19 = dict(HASHED.findall(amendment_item(PLAN.read_text(), "A11", 19)))
        frozen = [p for p in item19 if p.startswith(("routers/", "analysis/", "data/"))]
        held = {p: hashlib.sha256(git_bytes(freeze[1], p)).hexdigest() for p in frozen}
        check(
            bool(frozen) and held == {p: item19[p] for p in frozen},
            f"the freeze commit held each of item 19's {len(frozen)} frozen files as item 19 records it",
        )
        check(
            all(hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() == item19[p] for p in frozen),
            "and each is still as the freeze commit left it",
        )
        check(
            git("cat-file", "-e", f"{'0' * 40}^{{commit}}").returncode != 0,
            "known positive: a hash that names no commit is found missing",
        )
        check(
            git("merge-base", "--is-ancestor", added[0] if added else "HEAD", freeze[1]).returncode != 0,
            "known positive: the fit commit is not an ancestor of the freeze commit",
        )
    else:
        print("(the freeze commit is not in this clone: the history section is skipped)")

    cache = fit.cache_paths("dev")[0]
    matrix = PROJECT_ROOT / "results" / "dev_matrix.sqlite"
    if cache.exists() and fit.cache_paths("train")[0].exists() and matrix.exists():
        print("\n-- the refit (the hook only) --")
        refitted, rows = refit(settings)
        gap = largest_gap(refitted, weights)
        check(rows == expected, "the refit fold models learn from the rows item 20 records")
        check(gap < TOLERANCE, f"the refit fold models match the file to {TOLERANCE:g} (largest gap {gap:.1e})")
        rotated = largest_gap(refitted, [*weights[1:], weights[0]])
        check(
            rotated > TOLERANCE,
            "known positive: another fold model's heads are far from the refit (the refit can fail)",
        )
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "weights.json"
            copy.write_bytes(path.read_bytes() + b" ")
            check(
                hashlib.sha256(copy.read_bytes()).hexdigest() != recorded.get(relative),
                "known positive: one byte added to a copy of the file changes its hash",
            )
    else:
        print("\n(the caches or the dev matrix are missing: the refit is skipped)")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


def git_bytes(commit: str, relative: str) -> bytes:
    done = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=PROJECT_ROOT, capture_output=True, check=True)
    return done.stdout


if __name__ == "__main__":
    sys.exit(main())
