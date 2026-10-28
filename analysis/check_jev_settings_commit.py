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

import contextlib
import io
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

from analysis import build_jev_settings as builder
from analysis.benchmark import read_rows
from analysis.checking import git
from analysis.router_names import G_PLUS, NAMED_ROUTERS, G
from harness.collect import MAIN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS = "routers/jev_settings.json"
ROUTERS = (NAMED_ROUTERS[G], NAMED_ROUTERS[G_PLUS])  # the settings file's budget keys: the sections' names
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

# Trees the section "this repository" must fail on and never crash on: the settings file without its budgets, or its
# G+ part, and a record without budgets.
SETTINGS_OK = {"budget_bytes": {"jev": 1, "jev_bolstered": 1}, "g_plus": {"past_requests": []}}  # the file's own keys
BROKEN_TREES = {
    "a settings file with no budget_bytes": {SETTINGS: json.dumps({"g_plus": {"past_requests": []}})},
    "a settings file with no g_plus part": {SETTINGS: json.dumps({"budget_bytes": SETTINGS_OK["budget_bytes"]})},
    "a settings file that is not JSON": {SETTINGS: "not json"},
    "a settings file that is a list": {SETTINGS: "[]"},
    "a smoke record with no budget_bytes": {SETTINGS: json.dumps(SETTINGS_OK), SMOKE_RECORD: "{}"},
    "a missing smoke record": {SETTINGS: json.dumps(SETTINGS_OK)},
}
problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def failures_of(section: Callable[[], None]) -> int:
    """How many FAILs a section prints, without counting them as this script's own: the way a check proves that a
    section prints FAIL for a bad input, instead of raising."""
    before = len(problems)
    with contextlib.redirect_stdout(io.StringIO()):
        section()
    failed = len(problems) - before
    del problems[before:]
    return failed


def read_json(path: Path) -> object:
    """A JSON file's content, or None if it is missing or isn't JSON (the caller's check says which file)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def budgets_set(settings: dict) -> bool:
    """Both routers' `budget_bytes` are whole numbers of at least 1: the smoke test filled them in (A15 item 8)."""
    budgets = settings.get("budget_bytes")
    return (
        isinstance(budgets, dict)
        and set(budgets) == set(ROUTERS)
        and all(type(b) is int and b >= 1 for b in budgets.values())
    )


def settings_commit(repo: Path, path: str) -> str | None:
    """The commit that first added `path`, or None while no commit has."""
    added = git("log", "--diff-filter=A", "--format=%H", "--", path, repo=repo).stdout.split()
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


def repository(root: Path, *, rebuild: bool) -> None:
    """The promises about the settings file and the smoke record under `root`. A missing file, key or git history is a
    FAIL, never a crash. `rebuild` runs the builder against train's real outcomes, which only the real tree needs."""
    path = root / SETTINGS
    if not path.exists():
        print("      (no routers/jev_settings.json yet: Stage 6 · 8 commits it)")
        return
    settings = read_json(path)
    data = settings if isinstance(settings, dict) else {}
    budgets = data.get("budget_bytes")
    check(budgets_set(data), f"both budgets are set: {budgets}")
    if rebuild and MAIN.matrix.exists():
        built = builder.build_from_files().settings
        if budgets is None:
            check(False, "the committed file can't be compared with the builder's: it has no budget_bytes")
        else:
            built["budget_bytes"] = budgets
            check(builder.settings_text(built) == builder.settings_text(data),
                  "the committed file equals the builder's from its seed and train's outcomes")  # fmt: skip
    elif rebuild:
        print("      (results/ is absent: the rebuild is skipped)")
    record = read_json(root / SMOKE_RECORD)
    recorded = record.get("budget_bytes") if isinstance(record, dict) else None
    check(recorded is not None and recorded == budgets, "the smoke record's budgets are the settings file's")
    requests = read_json(root / SMOKE_REQUESTS)
    made_up = {r.get("prompt") for r in requests if isinstance(r, dict)} if isinstance(requests, list) else set()
    g_plus = data.get("g_plus")
    past = g_plus.get("past_requests") if isinstance(g_plus, dict) else None
    check(isinstance(past, list), "the settings file lists G+'s past requests")
    kept = {p.get("prompt") for p in past if isinstance(p, dict)} if isinstance(past, list) else set()
    real = {r["prompt"] for r in read_rows()} | kept
    check(len(made_up) >= 4 and not made_up & real, f"{len(made_up)} made-up requests, none a train or dev request")
    if git("rev-parse", "--git-dir", repo=root).returncode != 0:
        check(False, "git can read this tree's history")
        return
    commit = settings_commit(root, SETTINGS)
    if commit is None:
        print("      (the settings file is not committed yet: this is the settings commit)")
    else:
        check(changed_since(root, commit, COMMIT_FILES) == [],
              f"the settings commit's files are unchanged since {commit[:10]}")  # fmt: skip


def main() -> int:
    print("-- the budgets --")
    good = {"budget_bytes": {ROUTERS[0]: 24000, ROUTERS[1]: 9000}}
    check(budgets_set(good), "two whole-number budgets are set")
    for what, value in {"null": None, "zero": 0, "a float": 9000.0, "a bool": True, "a string": "9000"}.items():
        check(
            not budgets_set({"budget_bytes": {ROUTERS[0]: 24000, ROUTERS[1]: value}}), f"a budget of {what} is not set"
        )
    check(not budgets_set({"budget_bytes": {ROUTERS[0]: 24000}}), "a missing router's budget is not set")

    print("\n-- unchanged since the settings commit --")
    with tempfile.TemporaryDirectory() as folder:
        repo = Path(folder)
        broke = 0  # git commands that failed while the throwaway repository was set up and changed
        broke += git("init", "-q", repo=repo).returncode != 0
        broke += git("config", "user.email", "check@example.com", repo=repo).returncode != 0
        broke += git("config", "user.name", "check", repo=repo).returncode != 0
        (repo / "other.txt").write_text("before\n")
        broke += git("add", "-A", repo=repo).returncode != 0
        broke += git("commit", "-q", "-m", "before", repo=repo).returncode != 0
        check(settings_commit(repo, "settings.json") is None, "no commit yet adds a missing file")
        (repo / "settings.json").write_text("{}\n")
        (repo / "code.py").write_text("x = 1\n")
        broke += git("add", "-A", repo=repo).returncode != 0
        broke += git("commit", "-q", "-m", "settings", repo=repo).returncode != 0
        first = git("rev-parse", "HEAD", repo=repo).stdout.strip()
        check(settings_commit(repo, "settings.json") == first, "the commit that added the file is found")
        files = ("settings.json", "code.py")
        check(changed_since(repo, first, files) == [], "nothing is named while the files are as committed")
        (repo / "other.txt").write_text("after\n")
        broke += git("commit", "-aq", "-m", "other file", repo=repo).returncode != 0
        check(changed_since(repo, first, files) == [], "a later commit of another file names nothing")
        (repo / "code.py").write_text("x = 2\n")
        check(changed_since(repo, first, files) == ["code.py"], "an edit in the working tree is named")
        broke += git("add", "code.py", repo=repo).returncode != 0
        check(changed_since(repo, first, files) == ["code.py"], "a staged edit is named")
        broke += git("commit", "-q", "-m", "edit", repo=repo).returncode != 0
        check(changed_since(repo, first, files) == ["code.py"], "a later commit of an edit is named")
        check(settings_commit(repo, "settings.json") == first, "an edit does not move the settings commit")
        broke += git("rm", "-q", "settings.json", repo=repo).returncode != 0
        broke += git("commit", "-q", "-m", "remove", repo=repo).returncode != 0
        check(changed_since(repo, first, files) == ["settings.json", "code.py"], "a removed file is named")
        check(changed_since(repo, first, ("never.txt",)) == ["never.txt"], "a file the commit never held is named")
        check(broke == 0, "every git command that set up and changed the throwaway repository succeeded")

    print("\n-- a broken tree prints FAIL and does not raise --")
    for what, tree in BROKEN_TREES.items():
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative, text in tree.items():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_text(text, encoding="utf-8")
            failed = failures_of(partial(repository, root, rebuild=False))
            check(failed >= 1, f"known positive: {what} is a FAIL, not a crash ({failed} failed)")
    with tempfile.TemporaryDirectory() as folder:
        check(
            failures_of(lambda: repository(Path(folder), rebuild=False)) == 0,
            "a tree with no settings file is the settings commit's to come: nothing fails",
        )
        check(
            not git("log", repo=Path(folder)).returncode == 0,
            "git run where there is no repository reports its failure, it does not raise",
        )

    print("\n-- this repository --")
    repository(PROJECT_ROOT, rebuild=True)

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
