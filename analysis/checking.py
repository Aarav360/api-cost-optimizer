"""The two helpers the Stage 6 check scripts share (CODING_STANDARDS.md, Checks; Amendment A16, item 1).

Not a check: a module the `check_jev_*.py` scripts and `check_stage6.py` import, so `refused` has one argument order
and `git` one failure behaviour in all of them.

- `refused(error, call)`: whether `call` raises `error`. Any other exception passes through, so a refusal check that
  feeds the wrong input fails loudly instead of counting as a refusal.
- `git(*args, repo=...)`: runs git and returns the finished process whatever its exit code, never raising, so a failed
  git call is the caller's `FAIL` to print (a missing commit, a clone without history), not a crash. Aimed at any repo
  but the project's own it runs with git's repository-location variables removed: a commit hook run by `git commit`
  in a linked worktree has GIT_INDEX_FILE (an absolute path into the worktree's git dir) and others exported, and a
  throwaway repository under a temporary folder would otherwise read and write the worktree's real index. Aimed at the
  project root it inherits them, so in a hook it sees the index being committed.
"""

import os
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


# The variables with which git names the repository, its index and its objects. Each is a hook's to export, none a
# throwaway repository's to inherit.
REPOSITORY_VARIABLES = (
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_WORK_TREE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
)


def git(*args: str, repo: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    """`git <args>` in `repo`, text output, with the exit code left for the caller to read. Any repo but the project's
    own runs without `REPOSITORY_VARIABLES`; the project root inherits the environment."""
    env = None
    if Path(repo).resolve() != PROJECT_ROOT.resolve():
        env = {k: v for k, v in os.environ.items() if k not in REPOSITORY_VARIABLES}
    return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=False)
