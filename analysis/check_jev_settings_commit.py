"""Check the settings commit's promises (PLAN_v4.md §8 3c; Amendment A15 item 13; issue #38).

Run from the project root:  uv run python -m analysis.check_jev_settings_commit

Calls no Jev. The settings commit holds the settings file, the call-building code, the settings builder and the smoke
test's record, and A15 item 19 records their sha256s, so after it none of them may change:

- the budgets: both `budget_bytes` are whole numbers, set from the smoke test (known negatives: null, zero, a float);
- the rebuild: the committed file equals what the builder gives from its seed and train's run-0 outcomes, apart from
  nothing (where results/ is present, since the outcomes live there);
- unchanged since the commit: a file of the commit's list that differs from it, in the working tree, the index or a
  later commit, is named. The guard runs first on a throwaway git repository (a known positive for each way a file
  can change), then on this repository, where the commit is the one that first added the settings file; before that
  commit exists the section says so and passes;
- the record: its budgets are the settings file's, and no made-up request is a train or dev request (the benchmark
  reader refuses test, so a test request can't be compared here and none is read).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from analysis import build_jev_settings as builder
from analysis.benchmark import read_rows
from harness.collect import MAIN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS = "routers/jev_settings.json"
SMOKE_REQUESTS = "analysis/jev_smoke_requests.json"
SMOKE_RECORD = "analysis/jev_smoke_record.json"
# The settings commit's files (issue #38): the settings, the call-building code (routers/jev.py's import closure less
# the client, which freezes later), the settings builder and the smoke test's record with the made-up requests and
# the script that sent them.
COMMIT_FILES = (
    SETTINGS,
    "routers/jev_call.py",
    "routers/ladder.py",
    "routers/request.py",
    "analysis/build_jev_settings.py",
    "analysis/smoke_jev.py",
    SMOKE_REQUESTS,
    SMOKE_RECORD,
)

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return done.stdout


def budgets_set(settings: dict) -> bool:
    """Both routers' `budget_bytes` are whole numbers of at least 1: the smoke test filled them in (A15 item 8)."""
    budgets = settings.get("budget_bytes")
    return (
        isinstance(budgets, dict)
        and set(budgets) == {"jev", "jev_bolstered"}
        and all(type(b) is int and b >= 1 for b in budgets.values())
    )


def settings_commit(repo: Path, path: str) -> str | None:
    """The commit that first added `path`, or None while no commit has."""
    added = git(repo, "log", "--diff-filter=A", "--format=%H", "--", path).split()
    return added[-1] if added else None


def changed_since(repo: Path, commit: str, files: tuple[str, ...]) -> list[str]:
    """The files that are not as `commit` has them: edited in the working tree, the index or a later commit, removed,
    or never in the commit. The working tree is compared, so an edit about to be committed is named too."""
    named = []
    for file in files:
        held = subprocess.run(["git", "cat-file", "-e", f"{commit}:{file}"], cwd=repo, capture_output=True, check=False)
        same = subprocess.run(["git", "diff", "--quiet", commit, "--", file], cwd=repo, check=False)
        if held.returncode != 0 or same.returncode != 0:
            named.append(file)
    return named


def main() -> int:
    print("-- the budgets --")
    good = {"budget_bytes": {"jev": 24000, "jev_bolstered": 9000}}
    check(budgets_set(good), "two whole-number budgets are set")
    for what, value in {"null": None, "zero": 0, "a float": 9000.0, "a bool": True, "a string": "9000"}.items():
        check(
            not budgets_set({"budget_bytes": {"jev": 24000, "jev_bolstered": value}}), f"a budget of {what} is not set"
        )
    check(not budgets_set({"budget_bytes": {"jev": 24000}}), "a missing router's budget is not set")

    print("\n-- unchanged since the settings commit --")
    with tempfile.TemporaryDirectory() as folder:
        repo = Path(folder)
        git(repo, "init", "-q")
        git(repo, "config", "user.email", "check@example.com")
        git(repo, "config", "user.name", "check")
        (repo / "other.txt").write_text("before\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "before")
        check(settings_commit(repo, "settings.json") is None, "no commit yet adds a missing file")
        (repo / "settings.json").write_text("{}\n")
        (repo / "code.py").write_text("x = 1\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "settings")
        first = git(repo, "rev-parse", "HEAD").strip()
        check(settings_commit(repo, "settings.json") == first, "the commit that added the file is found")
        files = ("settings.json", "code.py")
        check(changed_since(repo, first, files) == [], "nothing is named while the files are as committed")
        (repo / "other.txt").write_text("after\n")
        git(repo, "commit", "-aq", "-m", "other file")
        check(changed_since(repo, first, files) == [], "a later commit of another file names nothing")
        (repo / "code.py").write_text("x = 2\n")
        check(changed_since(repo, first, files) == ["code.py"], "an edit in the working tree is named")
        git(repo, "add", "code.py")
        check(changed_since(repo, first, files) == ["code.py"], "a staged edit is named")
        git(repo, "commit", "-q", "-m", "edit")
        check(changed_since(repo, first, files) == ["code.py"], "a later commit of an edit is named")
        check(settings_commit(repo, "settings.json") == first, "an edit does not move the settings commit")
        git(repo, "rm", "-q", "settings.json")
        git(repo, "commit", "-q", "-m", "remove")
        check(changed_since(repo, first, files) == ["settings.json", "code.py"], "a removed file is named")
        check(changed_since(repo, first, ("never.txt",)) == ["never.txt"], "a file the commit never held is named")

    print("\n-- this repository --")
    path = PROJECT_ROOT / SETTINGS
    if not path.exists():
        print("      (no routers/jev_settings.json yet: Stage 6 · 8 commits it)")
    else:
        settings = json.loads(path.read_text(encoding="utf-8"))
        check(budgets_set(settings), f"both budgets are set: {settings['budget_bytes']}")
        if MAIN.matrix.exists():
            built = builder.build_from_files().settings
            built["budget_bytes"] = settings["budget_bytes"]
            check(builder.settings_text(built) == builder.settings_text(settings),
                  "the committed file equals the builder's from its seed and train's outcomes")  # fmt: skip
        else:
            print("      (results/ is absent: the rebuild is skipped)")
        record = json.loads((PROJECT_ROOT / SMOKE_RECORD).read_text(encoding="utf-8"))
        check(record["budget_bytes"] == settings["budget_bytes"], "the smoke record's budgets are the settings file's")
        made_up = {r["prompt"] for r in json.loads((PROJECT_ROOT / SMOKE_REQUESTS).read_text(encoding="utf-8"))}
        real = {r["prompt"] for r in read_rows()} | {p["prompt"] for p in settings["g_plus"]["past_requests"]}
        check(len(made_up) >= 4 and not made_up & real, f"{len(made_up)} made-up requests, none a train or dev request")
        commit = settings_commit(PROJECT_ROOT, SETTINGS)
        if commit is None:
            print("      (the settings file is not committed yet: this is the settings commit)")
        else:
            check(changed_since(PROJECT_ROOT, commit, COMMIT_FILES) == [],
                  f"the settings commit's files are unchanged since {commit[:10]}")  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
