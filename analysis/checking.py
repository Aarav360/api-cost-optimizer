"""The two helpers the Stage 6 check scripts share (CODING_STANDARDS.md, Checks; Amendment A16, item 1).

Not a check: a module the `check_jev_*.py` scripts and `check_stage6.py` import, so `refused` has one argument order
and `git` one failure behaviour in all of them.

- `refused(error, call)`: whether `call` raises `error`. Any other exception passes through, so a refusal check that
  feeds the wrong input fails loudly instead of counting as a refusal.
- `git(*args, repo=...)`: runs git and returns the finished process whatever its exit code, never raising, so a failed
  git call is the caller's `FAIL` to print (a missing commit, a clone without history), not a crash.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

from harness.client import PROJECT_ROOT


def refused(error: type[BaseException], call: Callable[[], object]) -> bool:
    """Whether `call` raises `error` (a subclass counts)."""
    try:
        call()
    except error:
        return True
    return False


def git(*args: str, repo: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    """`git <args>` in `repo`, text output, with the exit code left for the caller to read."""
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
