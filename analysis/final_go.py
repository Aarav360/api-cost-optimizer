"""The go's entry point: the final evaluation on the real world (PLAN_v4.md Amendment A19, items 2 to 4; issue #79).

Run from the project root, by the researcher and only at the go:  uv run python -m analysis.final_go --go

It builds the real world, whose sealed-store path comes from the collection module's own paths (A19 item 4: no default
path elsewhere, so a rehearsal can't open the real store by accident), the benchmark file from the benchmark reader's own
default, and the project's own folders, then hands it to the runner (`analysis/final_run.py`), the same one the rehearsal
runs on a made-up world. A resume is this command again: it runs the unlock again and goes on from what is stored.

**Nothing runs this before the go.** No check imports or runs it (`analysis/check_read_path.py` holds that), the hook doesn't,
and it does nothing at import. The `--go` flag is a second thing to type, so a stray `python -m` does not open anything.

It writes under `results/final/` (the record, the log, the tables), which git ignores, so a resume finds a clean tree;
test's results matrix is `results/test_matrix.sqlite`, beside dev's. The results PR copies what it commits.
"""

import inspect
import sys
from collections.abc import Sequence

from analysis.benchmark import read_rows
from analysis.final_run import run
from analysis.unlock_facts import read_git_facts, run_hook
from analysis.world import World
from harness.client import PROJECT_ROOT
from harness.collect import MAIN, RESULTS

OUTPUT = RESULTS / "final"


def real_world() -> World:
    """The real world: the collection module's sealed-store path, the benchmark reader's default file, the project's folders."""
    benchmark = inspect.signature(read_rows).parameters["path"].default
    # The real hook runner runs the tree's own `.githooks/pre-commit` by its path under the root; the real git facts fetch
    # origin/main and read the tree and the two commits through `analysis/checking.py`.
    return World(root=PROJECT_ROOT, sealed_store=MAIN.sealed, benchmark=benchmark, results=RESULTS, output=OUTPUT,
                 hook_runner=run_hook, git_facts=read_git_facts)  # fmt: skip


def main(argv: Sequence[str] = ()) -> int:
    if list(argv) != ["--go"]:
        print("This runs the final evaluation on the real sealed test answers, once, at the researcher's go. "
              "Run it with --go.")  # fmt: skip
        return 2
    outcome = run(real_world())
    return 0 if outcome.stopped_at is None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
