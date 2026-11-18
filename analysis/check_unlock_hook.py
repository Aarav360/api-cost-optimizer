"""Check the unlock's hook, git facts, record fields and live-replay draw (PLAN_v4.md Amendment A19, items 3 and 10; A17
item 26.5; issue #80).

Run from the project root:  uv run python -m analysis.check_unlock_hook

On made-up worlds only, with stand-in hooks and git facts; the real hook runner and the real git facts run on throwaway
folders and repositories under a temporary folder (every git call through `analysis/checking.py`). No results/, no Docker,
no Jev or Anthropic call, nothing near the real sealed store, and the real hook is not run (it would run this check).

- **The hook runs first.** A failing stand-in hook refuses at `hook` with the manifest gone, the store tampered, the tree
  dirty and `HEAD` off the tip all at once, and the git facts never asked: nothing after it ran. With the hook passing,
  the same world refuses at the next check, so the first refusal is the hook's and not an accident of the world. The
  store's hash is shown unread on a failing hook (`hashlib.sha256` and `sqlite3.connect` are made to fail, and shown to
  bite on a passing world).
- **The git facts**, each its own refusal and each opening nothing: a dirty tree, a `HEAD` that isn't `origin/main`'s tip,
  a failed fetch, a world naming no hook runner and one naming no git facts; a hook that dirties the tree is caught
  because the facts are read after it.
- **The real runners**, on throwaway folders: the hook runs by its path under the root it is given (a different
  `core.hooksPath` is ignored), a failing hook reports its exit code and last lines; the git facts fetch (origin moves,
  and the fetch shows it), see a dirty tree (untracked, modified, staged) and a local commit not on origin, ignore an
  ignored file, and fail for a repository with no origin. The go's world names both runners (read from its source).
- **The rehearsal's refusals**: the whole runner on the rehearsal's made-up world stops at the unlock for each of the
  three cases, naming the check, with the sandbox given nothing, no matrix, no tables and nothing under the results folder.
- **The record** of a go that passes holds the commit, the time, the hook's result and the environment (the macOS build,
  the chip and the library versions, against `importlib.metadata`), and a resume's unlock entry sits beside the first's.
- **The draw**: 25 test items per task, seeded with 2026, on a made-up benchmark of 30 per task, against a draw worked out
  by hand (the items sorted by the sha256 of "2026:replay:<task>:<item ID>", taken with `shasum` and not by this project's
  code); two draws agree, the benchmark's row order doesn't matter, another seed differs, train and dev items are never
  drawn, a task with fewer than 25 refuses at `replay_draw`, and the handle keeps the draw. The real benchmark's manifest
  counts at least 25 test items in each of its four tasks, so the real draw is 100.
"""

import ast
import contextlib
import dataclasses
import importlib.metadata
import io
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from unittest import mock

from analysis import check_final_run as rehearsal_check
from analysis import final_grading
from analysis import unlock as unlock_module
from analysis.checking import git, refused
from analysis.final_run import Outcome
from analysis.made_up_world import COMMIT, Content, StandInGit, StandInHook, made_up_world, row
from analysis.unlock import CHECKS, Handle, HandleError, UnlockRefusal, require_issued, unlock
from analysis.unlock_facts import GitFacts, GitFactsError, draw_replay, environment, read_git_facts, run_hook
from analysis.world import World
from harness.client import PROJECT_ROOT

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refusal_of(world: World) -> str | None:
    """The check an unlock of `world` refuses at, or None if it unlocks."""
    try:
        unlock(world)
    except UnlockRefusal as refusal:
        return refusal.check
    return None


def with_runners(world: World, hook: object = None, facts: object = None) -> World:
    return dataclasses.replace(world, hook_runner=hook or world.hook_runner, git_facts=facts or world.git_facts)  # type: ignore[arg-type]


def tamper(made) -> None:
    """Everything that would refuse later: a changed store, and then no manifest at all."""
    os.chmod(made.paths.sealed, 0o644)
    made.paths.sealed.write_bytes(made.paths.sealed.read_bytes() + b"x")
    made.paths.manifest.unlink()


# ---- the hook first ----------------------------------------------------------------------------------------------


def hook_first() -> None:
    print("-- the unlock runs the hook first, and a failing hook refuses before anything else is read --")
    check([c.__name__ for c in CHECKS[:2]] == ["hook", "git_facts"],
          "the unlock's checks begin with the hook, then the git facts")  # fmt: skip
    with made_up_world() as made:
        hook, facts = StandInHook(), StandInGit()
        handle = unlock(with_runners(made.world, hook, facts))
        check(hook.calls == [made.world.root] and facts.calls == [made.world.root],
              "a passing unlock runs the hook once and reads the git facts once, both given the world's root")  # fmt: skip
        entry = handle.results[0]
        check(entry.name == "hook" and entry.detail["passed"] is True and entry.detail["exit_code"] == 0,
              "the hook's result is the unlock's first result")  # fmt: skip
        check(handle.results[1].name == "git_facts" and handle.results[1].detail["head"] == COMMIT,
              "the git facts' result, with the commit, is its second")  # fmt: skip

    with made_up_world() as made:
        tamper(made)
        everything_wrong = with_runners(made.world, StandInHook(passed=False), StandInGit(clean=False, head="other"))
        check(refusal_of(with_runners(everything_wrong, StandInHook(), StandInGit(clean=False, head="other")))
              == "git_clean", "the same world with a passing hook refuses at the next check, the git facts")  # fmt: skip
        hook, facts = StandInHook(passed=False), StandInGit(clean=False, head="other")
        sha = mock.patch.object(unlock_module.hashlib, "sha256", side_effect=AssertionError("the store was hashed"))
        db = mock.patch.object(unlock_module.sqlite3, "connect", side_effect=AssertionError("the store was opened"))
        with sha, db:
            refused_at = refusal_of(with_runners(made.world, hook, facts))
        check(refused_at == "hook" and facts.calls == [],
              "a failing hook refuses at `hook`, though the manifest, the store, the tree and HEAD are all wrong, and the "
              "git facts are never asked")  # fmt: skip
        try:
            unlock(with_runners(made.world, StandInHook(passed=False)))
        except UnlockRefusal as refusal:
            check("stand-in hook failed" in str(refusal) and "exit code 1" in str(refusal),
                  "the refusal quotes the hook's exit code and its last lines")  # fmt: skip

    with made_up_world() as made:
        sha = mock.patch.object(unlock_module.hashlib, "sha256", side_effect=AssertionError("the store was hashed"))
        with sha:
            check(refused(AssertionError, lambda: unlock(made.world)),
                  "the guard bites: a passing hook goes on to the store's hash, which the patch makes fail")  # fmt: skip
            check(refusal_of(with_runners(made.world, StandInHook(passed=False))) == "hook",
                  "a failing hook stops before the hash is read")  # fmt: skip

    with made_up_world() as made:
        check(refusal_of(dataclasses.replace(made.world, hook_runner=None)) == "hook",
              "a world that names no hook runner refuses at `hook`, not passes")  # fmt: skip
        hook = StandInHook()
        check(refusal_of(with_runners(dataclasses.replace(made.world, git_facts=None), hook)) == "git_facts"
              and len(hook.calls) == 1, "a world that names no git facts refuses at `git_facts`, after the hook ran")  # fmt: skip


def git_facts_refusals() -> None:
    print("\n-- the git facts: a dirty tree, a HEAD that isn't origin/main's tip, a failed fetch --")
    cases = {
        "a dirty tree": (StandInGit(clean=False), "git_clean"),
        "a HEAD that isn't origin/main's tip": (StandInGit(head="a" * 40, origin_main="b" * 40),
                                                "git_head_is_origin_main"),
    }  # fmt: skip
    for what, (facts, expected) in cases.items():
        with made_up_world() as made:
            tamper(made)  # later checks would refuse too, so only the order's first refusal can name the git check
            db = mock.patch.object(unlock_module.sqlite3, "connect", side_effect=AssertionError("the store was opened"))
            with db:
                refused_at = refusal_of(with_runners(made.world, StandInHook(), facts))
            check(refused_at == expected, f"{what} refuses at {expected}, opening nothing")

    def failing_fetch(root: Path) -> GitFacts:
        raise GitFactsError("git fetch failed: no origin")

    with made_up_world() as made:
        check(refusal_of(with_runners(made.world, facts=failing_fetch)) == "git_facts",
              "a fetch that fails refuses at git_facts")  # fmt: skip

    with made_up_world() as made:
        state = {"dirty": False}

        def hook(root: Path):
            state["dirty"] = True  # a hook that leaves a file behind
            return StandInHook()(root)

        def tree(root: Path) -> GitFacts:
            return GitFacts(not state["dirty"], COMMIT, COMMIT)

        check(refusal_of(with_runners(made.world, hook, tree)) == "git_clean",
              "a hook that dirties the tree is caught, since the facts are read after it")  # fmt: skip


# ---- the real runners ----------------------------------------------------------------------------------------------


def commit(repo: Path, message: str = "a commit") -> None:
    done = git("-c", "user.name=check", "-c", "user.email=check@example.com", "-c", "commit.gpgsign=false",
               "-c", "core.hooksPath=/dev/null", "commit", "-q", "-am", message, repo=repo)  # fmt: skip
    assert done.returncode == 0, done.stderr


def run_git(repo: Path, *args: str) -> str:
    done = git(*args, repo=repo)
    assert done.returncode == 0, f"git {args}: {done.stderr}"
    return done.stdout.strip()


def real_hook() -> None:
    print("\n-- the real hook runner runs the hook by its path under the root --")
    with tempfile.TemporaryDirectory() as folder:
        root, other = Path(folder) / "root", Path(folder) / "other"
        for place, body in ((root, 'pwd > ran_in.txt\necho "FAIL  one"\necho "the hook passed in 0 min 2 s"\n'),
                            (other, "exit 9\n")):  # fmt: skip
            (place / ".githooks").mkdir(parents=True)
            (place / ".githooks" / "pre-commit").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
        result = run_hook(root)
        check(result.passed and result.exit_code == 0 and result.last_line == "the hook passed in 0 min 2 s",
              f"a hook that exits 0 passes, with its last line {result.last_line!r}")  # fmt: skip
        check((root / "ran_in.txt").read_text().strip() == str(root.resolve()),
              "it ran in the root it was given, from that root's own .githooks/pre-commit")  # fmt: skip
        failing = run_hook(other)
        check(not failing.passed and failing.exit_code == 9, "a hook that exits 9 fails, with its exit code")
        (other / ".githooks" / "pre-commit").write_text("echo one\necho two\necho FAIL  three\nexit 1\n")
        run_git(root, "init", "-q")
        run_git(root, "config", "core.hooksPath", str(other / ".githooks"))  # an absolute path to another copy
        again = run_hook(root)
        check(again.passed and again.last_line == "the hook passed in 0 min 2 s",
              "a core.hooksPath that names another copy of the hook is ignored: the root's own runs")  # fmt: skip
        failing = run_hook(other)
        check(
            failing.tail == ("one", "two", "FAIL three"), f"a failing hook's tail holds its last lines {failing.tail}"
        )

    hook_file = PROJECT_ROOT / ".githooks" / "pre-commit"
    text = hook_file.read_text(encoding="utf-8")
    check(text.startswith("#!/usr/bin/env bash") and "run_check analysis.check_unlock_hook" in text,
          "the project's own hook is where the runner looks, and it runs this check")  # fmt: skip


def real_git_facts() -> None:
    print("\n-- the real git facts fetch, then read the tree and the two commits --")
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        run_git(base, "init", "-q", "--bare", "-b", "main", "origin.git")
        run_git(base, "clone", "-q", "origin.git", "work")
        work, second = base / "work", base / "second"
        run_git(work, "checkout", "-q", "-B", "main")
        (work / "a.txt").write_text("one\n")
        (work / ".gitignore").write_text("ignored/\n")
        run_git(work, "add", "a.txt", ".gitignore")
        commit(work)
        run_git(work, "push", "-q", "origin", "main")
        tip = run_git(
            base / "origin.git", "rev-parse", "main"
        )  # the bare repository's own answer: independent of the clone

        facts = read_git_facts(work)
        check(facts == GitFacts(True, tip, tip), "a clean clone at origin/main's tip: clean, HEAD and the tip equal")
        (work / "ignored").mkdir()
        (work / "ignored" / "f.txt").write_text("x")
        check(
            read_git_facts(work).clean,
            "an ignored file leaves the tree clean (results/final is ignored so a resume is clean)",
        )
        (work / "b.txt").write_text("new\n")
        check(not read_git_facts(work).clean, "an untracked file makes the tree dirty")
        (work / "b.txt").unlink()
        (work / "a.txt").write_text("changed\n")
        check(not read_git_facts(work).clean, "a modified tracked file makes it dirty")
        run_git(work, "add", "a.txt")
        check(not read_git_facts(work).clean, "a staged change makes it dirty")
        run_git(work, "reset", "-q", "--hard")

        run_git(base, "clone", "-q", "origin.git", "second")
        (second / "a.txt").write_text("two\n")
        commit(second)
        run_git(second, "push", "-q", "origin", "main")
        moved = run_git(base / "origin.git", "rev-parse", "main")
        stale = run_git(work, "rev-parse", "origin/main")
        facts = read_git_facts(work)
        check(stale == tip and moved != tip and facts.origin_main == moved and facts.head == tip,
              "origin moved and the clone hadn't fetched: the facts fetch first, so origin/main is the new tip, not HEAD")  # fmt: skip

        run_git(work, "pull", "-q", "--ff-only", "origin", "main")
        (work / "a.txt").write_text("three\n")
        commit(work)
        facts = read_git_facts(work)
        check(
            facts.head != facts.origin_main and facts.clean, "a local commit that isn't on origin: HEAD is not the tip"
        )

        # The hook's own writes: a hook that leaves a file, then the facts.
        run_git(work, "reset", "-q", "--hard", "origin/main")
        (work / ".githooks").mkdir()
        (work / ".githooks" / "pre-commit").write_text("echo done > left_behind.txt\n")
        run_git(work, "add", ".githooks")
        commit(work)
        run_git(work, "push", "-q", "origin", "main")
        check(read_git_facts(work).clean and run_hook(work).passed and not read_git_facts(work).clean,
              "a hook that leaves a file behind turns a clean tree dirty, which the facts read after it show")  # fmt: skip

        bare = base / "no_origin"
        run_git(base, "init", "-q", "-b", "main", "no_origin")
        check(refused(GitFactsError, lambda: read_git_facts(bare)), "a repository with no origin raises GitFactsError")


def go_world_wiring() -> None:
    print("\n-- the go's world names the real runners --")
    tree = ast.parse((PROJECT_ROOT / "analysis" / ("final_go" + ".py")).read_text(encoding="utf-8"))
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "World":
            names = {k.arg: getattr(k.value, "id", None) for k in node.keywords}
    check(names.get("hook_runner") == run_hook.__name__ and names.get("git_facts") == read_git_facts.__name__,
          f"the real world's hook runner and git facts are the real ones {names.get('hook_runner')} {names.get('git_facts')}")  # fmt: skip
    plain = World(Path("."), None, Path("."), Path("."), Path("."))
    check(plain.hook_runner is None and plain.git_facts is None and refusal_of(plain) == "hook",
          "a world built without them refuses at the hook, so a forgotten field can't skip it")  # fmt: skip


# ---- the rehearsal's refusals and the record -----------------------------------------------------------------------


def rehearsal_refusals() -> None:
    print("\n-- the rehearsal refuses at the unlock for each case, giving every fake nothing --")
    cases = {
        "a failing stand-in hook": (StandInHook(passed=False), StandInGit(), "hook"),
        "a dirty tree": (StandInHook(), StandInGit(clean=False), "git_clean"),
        "a HEAD that isn't origin/main's tip": (StandInHook(), StandInGit(head="a" * 40), "git_head_is_origin_main"),
    }  # fmt: skip
    for what, (hook, facts, expected) in cases.items():
        with rehearsal_check.rehearsed() as r:
            r.world = with_runners(r.world, hook, facts)
            db = mock.patch.object(unlock_module.sqlite3, "connect", side_effect=AssertionError("the store was opened"))
            with db:
                outcome, out, err = r.go()
            go = r.record["gos"][0]
            check(outcome == Outcome("unlock", "UnlockRefusal", None) and go["stopped"]["check"] == expected
                  and go["unlock"] is None and [s["name"] for s in go["steps"]] == ["unlock"],
                  f"{what}: the run stops at the unlock, naming {expected}, with no unlock entry and no later step")  # fmt: skip
            nothing = (r.sandbox_calls == 0 and not final_grading.matrix_path(r.world.results).exists()
                       and not (r.world.output / "tables_test.json").exists() and not r.world.results.exists())  # fmt: skip
            check(nothing, f"{what}: the sandbox got nothing, and no matrix, tables or results folder was made")
            check(len(hook.calls) == 1 and len(facts.calls) == (0 if expected == "hook" else 1),
                  f"{what}: the hook ran {len(hook.calls)} time, the git facts {len(facts.calls)}")  # fmt: skip
            check(expected in out and err == "", f"{what}: the refusal names its check on the screen")  # fmt: skip


def expected_environment() -> None:
    env = dict(environment())
    libraries = ("torch", "transformers", "tokenizers", "numpy", "scikit-learn")  # A11 item 19's, by hand
    check(set(env) == {"os", "chip", "math library", "python", "libraries"},
          f"the environment names the OS, chip, math library, Python and libraries {sorted(env)}")  # fmt: skip
    check(env["libraries"] == {name: importlib.metadata.version(name) for name in libraries},
          "its library versions are the installed ones, A11 item 19's five")  # fmt: skip
    check(env["python"] == platform.python_version() and bool(env["chip"]) and bool(env["os"]),
          "its Python is the running one, and its OS and chip are named")  # fmt: skip
    if platform.system() == "Darwin":
        check(env["os"].startswith(f"macOS {platform.mac_ver()[0]} (build "),
              f"on macOS it names the build as 'macOS <version> (build ...)': {env['os']}")  # fmt: skip


def record_fields() -> None:
    print("\n-- the record holds each go's commit, time, hook result and environment --")
    with rehearsal_check.rehearsed() as r:
        hook = StandInHook()
        r.world = with_runners(r.world, hook)
        with contextlib.redirect_stdout(io.StringIO()):
            first = r.go(rehearsal_check.STEPS[:1])[0]
            second = r.go(rehearsal_check.STEPS[:1])[0]
        gos = r.record["gos"]
        check(first == second == Outcome() and len(gos) == 2, "two gos ran the unlock, a resume's beside the first")
        entry = gos[0]["unlock"]
        check(entry["commit"] == COMMIT, "the entry holds the commit the go ran from")
        check(entry["time"] == gos[0]["steps"][0]["started"] or entry["time"] > gos[0]["started"],
              "it holds the time the unlock passed")  # fmt: skip
        check(entry["hook"] == {"passed": True, "exit_code": 0, "seconds": 0.0, "last_line": "the stand-in hook passed"},
              f"it holds the hook's result {entry['hook']}")  # fmt: skip
        check(entry["environment"] == dict(environment()), "it holds the environment")
        names = [c["name"] for c in entry["checks"]]
        check(names[:3] == ["hook", "git_facts", "replay_draw"], f"its checks begin with the hook, the git facts and the draw {names}")  # fmt: skip
        draws = [g["unlock"]["checks"][2]["detail"]["item_ids"] for g in gos]
        check(
            draws[0] == draws[1] and len(draws[0]) == 8,
            "a resume's unlock entry holds the same replay items as the first's",
        )
        check(len(hook.calls) == 2, "each go ran the hook")
    expected_environment()


# ---- the draw ------------------------------------------------------------------------------------------------------

TASKS = ("banking77", "ifeval", "mbppplus", "mmlu_pro")
# By hand: each task's 30 items "<task>:00" to "<task>:29" sorted by the sha256 of "2026:replay:<task>:<item ID>" (taken with
# `shasum -a 256` and `sort`, outside this project's code), the first 25.
HAND_DRAW = {
    "banking77": [9, 4, 15, 22, 1, 26, 29, 0, 6, 2, 20, 27, 24, 18, 8, 12, 14, 17, 19, 11, 25, 16, 28, 3, 10],
    "ifeval": [8, 17, 27, 13, 9, 21, 28, 16, 20, 22, 18, 11, 23, 0, 1, 7, 2, 24, 29, 12, 14, 10, 26, 19, 25],
    "mbppplus": [26, 6, 0, 21, 24, 10, 13, 20, 23, 1, 28, 4, 11, 16, 17, 9, 18, 15, 14, 27, 7, 29, 8, 19, 3],
    "mmlu_pro": [7, 27, 19, 16, 4, 18, 15, 22, 11, 28, 3, 6, 21, 0, 5, 29, 23, 14, 13, 1, 12, 10, 26, 17, 25],
}  # fmt: skip
EXPECTED_DRAW = [f"{task}:{n:02d}" for task in TASKS for n in HAND_DRAW[task]]


def benchmark_rows(per_task: int = 30) -> list[dict]:
    test = [row(f"{t}:{n:02d}", "test", "p", "e") for t in TASKS for n in range(per_task)]
    others = [row(f"{t}:{s}-{n:02d}", s, "p", "e") for t in TASKS for s in ("train", "dev") for n in range(40)]
    return others + test


def the_draw() -> None:
    print("\n-- the live replay's draw: 25 per task, seeded with 2026 --")
    rows = benchmark_rows()
    test = [r for r in rows if r["split"] == "test"]
    drawn = draw_replay(test)
    check(len(EXPECTED_DRAW) == 100 and len(set(EXPECTED_DRAW)) == 100, "the hand-worked draw is 100 distinct items")
    check(drawn == EXPECTED_DRAW, "the draw equals the hand-worked one, item for item and in order")
    check(draw_replay(test) == drawn, "two draws on the same items agree")
    check(draw_replay(list(reversed(test))) == drawn, "the benchmark's row order doesn't change the draw")
    check(draw_replay(test, seed=2027) != drawn, "another seed gives another draw, so the seed is in use")
    check(
        all(item.split(":")[1].isdigit() for item in drawn),
        "no train or dev item is drawn: the draw is given test rows",
    )
    few = [r for r in test if r["source"] != "ifeval" or int(r["item_id"][-2:]) < 24]  # ifeval keeps 24
    check(refused(ValueError, lambda: draw_replay(few)) and refused(ValueError, lambda: draw_replay([])),
          "a task with 24 test items, or no test items, can't give 25")  # fmt: skip

    with made_up_world(Content(rows, [])) as made:
        check(made.world.replay_per_task == 25, "the made-up world with 30 test items per task draws the real 25")
        handle = unlock(made.world)
        check(list(handle.replay_items()) == EXPECTED_DRAW,
              "the unlock draws the hand-worked 100 and the handle keeps them")  # fmt: skip
        detail = handle.results[2].detail
        check(handle.results[2].name == "replay_draw" and detail["item_ids"] == EXPECTED_DRAW and detail["per_task"] == 25,
              "the record's entry for the draw holds the same 100, and the 25")  # fmt: skip
        check(list(unlock(made.world).replay_items()) == EXPECTED_DRAW, "a resume's unlock draws the same 100")
        check(refused(HandleError, lambda: Handle(made.world, handle.results).replay_items())
              and refused(HandleError, lambda: require_issued(Handle(made.world, handle.results))),
              "a handle built by hand is refused its draw")  # fmt: skip
        shuffled = list(reversed(rows))
        made.world.benchmark.write_text("".join(json.dumps(r) + "\n" for r in shuffled), encoding="utf-8")
        check(
            list(unlock(made.world).replay_items()) == EXPECTED_DRAW,
            "a benchmark file in another row order draws the same 100",
        )

    with made_up_world(Content(benchmark_rows(24), [])) as made:
        short = dataclasses.replace(made.world, replay_per_task=25)
        check(refusal_of(short) == "replay_draw", "a world with 24 test items in a task refuses at replay_draw")

    manifest = json.loads((PROJECT_ROOT / "data" / "benchmark_manifest.json").read_text(encoding="utf-8"))
    counts = {task: info["splits"]["test"] for task, info in manifest["tasks"].items()}
    check(sorted(counts) == sorted(TASKS) and all(n >= 25 for n in counts.values()),
          f"the real benchmark has four tasks with at least 25 test items each {counts}, so the real draw is 100")  # fmt: skip


def main() -> int:
    hook_first()
    git_facts_refusals()
    real_hook()
    real_git_facts()
    go_world_wiring()
    rehearsal_refusals()
    record_fields()
    the_draw()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
