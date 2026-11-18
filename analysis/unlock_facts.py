"""What the unlock learns from outside the project's files: the hook's result, the git facts, the environment and the
live replay's draw (PLAN_v4.md Amendment A19, items 3 and 10; A17 item 26.5; issue #77, #80).

The unlock (`analysis/unlock.py`) takes each of these from the world, so the go and a rehearsal differ only in the world:

- **`run_hook(root)`** is the real world's hook runner: the full `.githooks/pre-commit`, started by its path under the
  world's root (`bash <root>/.githooks/pre-commit`, with `root` as its working folder). It doesn't rely on
  `core.hooksPath`, which is an absolute path in a clone and can name another checkout's copy. A rehearsal's world names a
  stand-in instead (`analysis/made_up_world.py`), since the hook runs the rehearsal and the real one would recurse.
- **`read_git_facts(root)`** is the real world's git facts: it fetches `origin/main`, then reads whether the tree is clean
  and what `HEAD` and `origin/main` point at. Every git call goes through `analysis/checking.py`, the one module that may
  start git (`analysis/check_guardrails.py`).
- **`environment()`** is the macOS build, the chip, PyTorch's math library, Python and the library versions A11 item 19
  names, as `analysis/measure_trained_router.py` recorded them for F's freeze. Recorded, never pinned here: the unlock's
  verify list (#81) holds the versions.
- **`draw_replay(rows, per_task)`** is the live replay's draw: `per_task` test items for each task, ordered by sha256 of
  "2026:replay:<task>:<item ID>" as the project's other seeded draws are (`analysis.build_jev_settings.draw_order`), so the
  draw depends on the items' IDs alone: not on the benchmark's row order, and the same on a resume.
"""

import contextlib
import functools
import io
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.build_jev_settings import draw_order
from analysis.checking import git

HOOK = Path(".githooks") / "pre-commit"  # under the world's root
REMOTE_BRANCH = "origin/main"
FETCH = ("fetch", "origin", "+refs/heads/main:refs/remotes/origin/main")  # the refspec names the ref it updates
SEED = 2026  # the bootstrap's seed, and A17 item 26.5's draw
REPLAY_PER_TASK = 25  # A17 item 26.5: 100 items, 25 per task
TAIL_LINES = 6  # how much of a failing hook's output a refusal quotes


@dataclass(frozen=True)
class HookResult:
    """One run of the full hook: whether it passed, its exit code, its time and its last line (a pass names its total)."""

    passed: bool
    exit_code: int
    seconds: float
    last_line: str
    tail: tuple[str, ...] = ()  # the last lines of its output, for a refusal's message


@dataclass(frozen=True)
class GitFacts:
    """What the unlock asks of git: a clean tree, `HEAD`, and `origin/main`'s tip after a fetch."""

    clean: bool
    head: str
    origin_main: str


class GitFactsError(Exception):
    """A git call the facts need failed (no `origin`, no network, not a repository). The message says which."""


def run_hook(root: Path) -> HookResult:
    """Run the full pre-commit hook of the tree at `root`, by its path there, and report how it ended."""
    started = time.monotonic()
    done = subprocess.run(
        ["bash", str(root / HOOK)], cwd=root, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL
    )
    lines = [line for line in (done.stdout + done.stderr).splitlines() if line.strip()]
    return HookResult(done.returncode == 0, done.returncode, time.monotonic() - started,
                      lines[-1] if lines else "", tuple(lines[-TAIL_LINES:]))  # fmt: skip


def read_git_facts(root: Path) -> GitFacts:
    """Fetch `origin/main`, then read the tree's state and the two commits. Raises `GitFactsError` if a call fails."""
    steps = {
        "fetch": FETCH,
        "status": ("status", "--porcelain"),
        "HEAD": ("rev-parse", "--verify", "HEAD"),
        REMOTE_BRANCH: ("rev-parse", "--verify", REMOTE_BRANCH),
    }
    out = {}
    for name, args in steps.items():
        done = git(*args, repo=root)
        if done.returncode != 0:
            raise GitFactsError(f"git {name} failed: {done.stderr.strip()[:200]}")
        out[name] = done.stdout.strip()
    return GitFacts(out["status"] == "", out["HEAD"], out[REMOTE_BRANCH])


@functools.cache
def environment() -> Mapping[str, Any]:
    """The machine as A11 item 19 records it, for the unlock's entry: macOS build, chip, math library, Python, versions."""
    # Importing PyTorch on macOS writes a notice to the error stream. The run's log tees that stream, and a notice isn't
    # something the run did, so it goes nowhere.
    with contextlib.redirect_stderr(io.StringIO()):
        from analysis import measure_trained_router as measure

        found = {**measure.environment(), "libraries": measure.library_versions()}
    return found


def draw_replay(rows: list[dict], per_task: int = REPLAY_PER_TASK, seed: int = SEED) -> list[str]:
    """The live replay's items: for each task (sorted), the first `per_task` of its test items in the seeded order.
    Raises `ValueError` for no test rows or a task with fewer than `per_task`."""
    by_task: dict[str, list[str]] = {}
    for row in rows:
        by_task.setdefault(row["source"], []).append(row["item_id"])
    if not by_task:
        raise ValueError("the benchmark has no test items to draw from")
    drawn: list[str] = []
    for task in sorted(by_task):
        if len(by_task[task]) < per_task:
            raise ValueError(f"{task} has {len(by_task[task])} test items, fewer than the {per_task} to draw")
        drawn += draw_order(by_task[task], f"replay:{task}", seed)[:per_task]
    return drawn
