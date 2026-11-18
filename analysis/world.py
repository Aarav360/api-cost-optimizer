"""The world: everything the final evaluation touches, in one value (PLAN_v4.md Amendment A19, items 3 and 4; issue #77).

The go and the rehearsal share one entry point, and they differ only in the world they build: the go's entry point
builds the real one (its sealed-store path taken from the collection module's own paths), a rehearsal builds a made-up
one (`analysis/made_up_world.py`). The unlock (`analysis/unlock.py`) takes a world and returns a handle or refuses.

The fields below the output folder are placeholders: later tickets fill them in (#80 the hook runner and the git
facts, #81 the verify list, the backends with the steps that call them). A field added later takes a default, so a
world built before it still builds.

This module only names things. It opens nothing, and the sealed store's path is a field the unlock alone reads
(`analysis/check_read_path.py` fails any other module that does).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class World:
    root: Path  # where the frozen files are read from: the project root, or a made-up one holding the same layout
    sealed_store: Path | None  # the sealed store's path, with no default; None is refused (check `sealed_store_path`)
    benchmark: Path  # the benchmark file, whose test rows are the final evaluation's items
    results: Path  # the folder test's records go to
    output: Path  # the folder for the record, the tables and the figures
    backends: Any = None  # placeholder: a Jev endpoint, an Anthropic client, the sandbox, the embedding backend
    hook_runner: Any = None  # placeholder (#80): runs the full pre-commit hook, or a rehearsal's stand-in
    git_facts: Any = None  # placeholder (#80): whether the tree is clean, `HEAD` and `origin/main`'s tip
    verify: tuple[Any, ...] = ()  # placeholder (#81): what the unlock verifies, each with its expected hash or pin
