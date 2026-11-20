"""The world: everything the final evaluation touches, in one value (PLAN_v4.md Amendment A19, items 3 and 4; issue #77).

The go and the rehearsal share one entry point, and they differ only in the world they build: the go's entry point
builds the real one (its sealed-store path taken from the collection module's own paths), a rehearsal builds a made-up
one (`analysis/made_up_world.py`). The unlock (`analysis/unlock.py`) takes a world and returns a handle or refuses.

The fields below the output folder are for later tickets to fill in: #80 gave the hook runner, the git facts and the
replay's size their types; #81 gave the verify list its type (the unlock's list of recorded hashes and pins, which a
made-up world leaves empty) and the backends (`analysis/backends.py`) came with the steps that call them. A field added later takes a default,
so a world built before it still builds.

This module only names things. It opens nothing, and the sealed store's path is a field the unlock alone reads
(`analysis/check_read_path.py` fails any other module that does).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from analysis.backends import Backends
from analysis.unlock_facts import REPLAY_PER_TASK, GitFacts, HookResult
from analysis.verify_list import Entry


@dataclass(frozen=True)
class World:
    root: Path  # where the frozen files are read from: the project root, or a made-up one holding the same layout
    sealed_store: Path | None  # the sealed store's path, with no default; None is refused (check `sealed_store_path`)
    benchmark: Path  # the benchmark file, whose test rows are the final evaluation's items
    results: Path  # the folder test's records go to
    output: Path  # the folder for the record, the tables and the figures
    backends: Backends | None = (
        None  # stand-ins for a rehearsal (`analysis/backends.py`); None is the go's world: the real ones
    )
    hook_runner: Callable[[Path], HookResult] | None = None  # takes the root: the full hook, or a rehearsal's stand-in
    git_facts: Callable[[Path], GitFacts] | None = None  # takes the root: clean tree, `HEAD`, `origin/main`
    verify: tuple[Entry, ...] = ()  # what the unlock verifies, each with its recorded value (`analysis/verify_list.py`)
    replay_per_task: int = REPLAY_PER_TASK  # the live replay's items per task (A17 item 26.5)
