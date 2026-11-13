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
- `git_bytes(*args, repo=...)`: the same, with the output as bytes, for a file's content to hash (`git show
  <commit>:<path>`): text output would turn a CRLF into LF. The Stage 7 review moved `check_gates_freeze`'s raw
  `git show` here and gave `check_stage7` its git calls through `git` and this (Amendment A18, item 1); some older
  checks (`check_jev_freeze`, `check_jev_dev`, `check_trained_fit`) still run git themselves.
- `full_history(repo=...)`: whether the clone holds history, not a shallow copy such as CI's checkout, where the
  oldest commit seems to add every file and a "first commit that added" search would name it.
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


def _environment(repo: Path) -> dict[str, str] | None:
    """The environment git runs with in `repo`: without `REPOSITORY_VARIABLES` for any repo but the project's own."""
    if Path(repo).resolve() == PROJECT_ROOT.resolve():
        return None
    return {k: v for k, v in os.environ.items() if k not in REPOSITORY_VARIABLES}


def git(*args: str, repo: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    """`git <args>` in `repo`, text output, with the exit code left for the caller to read. Any repo but the project's
    own runs without `REPOSITORY_VARIABLES`; the project root inherits the environment."""
    return subprocess.run(["git", *args], cwd=repo, env=_environment(repo), capture_output=True, text=True, check=False)


def git_bytes(*args: str, repo: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[bytes]:
    """`git <args>` as `git` runs it, with the output as bytes: a committed file's content exactly, to hash."""
    return subprocess.run(["git", *args], cwd=repo, env=_environment(repo), capture_output=True, check=False)


def full_history(repo: Path = PROJECT_ROOT) -> bool:
    """Whether `repo` is a git clone with its history: not a shallow one (CI's checkout fetches one commit)."""
    shallow = git("rev-parse", "--is-shallow-repository", repo=repo)
    return shallow.returncode == 0 and shallow.stdout.strip() == "false"
