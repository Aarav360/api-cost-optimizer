"""Check the guardrails themselves: every check script runs somewhere, and git starts in one module (CODING_STANDARDS.md,
Checks; Amendment A17, item 27; Amendment A18, items 1 and 4).

Run from the project root:  uv run python -m analysis.check_guardrails

- **Every check script runs somewhere.** Each `check_*.py` is a `run_check` line in `.githooks/pre-commit`, unless
  `MANUAL` gives the reason it runs by hand, as the README's "Running it" shows. CI's job runs every hook check except
  those `HOOK_ONLY` gives a reason for, and nothing the hook doesn't. A17 item 27 put `harness/check_sandbox.py` in the
  hook, and nothing noticed it was missing until the Stage 7 review. Each list must yield a name with a digit in it
  (`analysis.check_chart_v4`), so a reading that drops such names can't pass by missing them.
- **Git starts only in `analysis/checking.py`.** A check's throwaway repository inherited a commit hook's repository
  variables and wrote into a worktree's index and the shared `.git/config` (A18 item 4). `git` and `git_bytes` there
  drop those variables for any repository but the project's own, so every other module runs git through them. The
  search looks for an argument list that starts with "git", and must find that module's own two calls, so it can't
  pass by matching nothing.
"""

import re
import sys
from collections.abc import Mapping

from analysis.checking import git
from harness.client import PROJECT_ROOT

HOOK = PROJECT_ROOT / ".githooks" / "pre-commit"
CI = PROJECT_ROOT / ".github" / "workflows" / "check.yml"
README = PROJECT_ROOT / "README.md"
HELPER = "analysis/checking.py"

# The hook's checks CI's job can't run (it has no results/, model snapshot or Docker).
HOOK_ONLY = {
    "analysis.check_gates_design": "recomputes A17's facts from train's and dev's stored rows in results/",
    "analysis.check_gates_sandbox": "runs the code check in the real sandbox, which needs Docker",
    "analysis.check_simulator": "checks the analysis package on dev, from the dev matrix in results/",
    "analysis.check_stage4": "checks E on dev, from the dev matrix in results/",
    "analysis.check_trained_dev": "embeds the dev requests with the model snapshot, and walks the dev matrix",
    "analysis.check_trained_diagnostics": "needs the dev matrix and the dev embedding cache",
    "harness.check_sandbox": "runs every MBPP+ reference in the pinned sandbox image, which needs Docker",
}
# The checks run by hand, each at the moment its docstring names.
MANUAL = {
    "harness.check_benchmark": "before the collection run and before Stage 8 scores test",
    "harness.check_collect": "after a change to the collection runner; takes minutes",
    "harness.check_evaluators": "after a change to an evaluator, a template or uv.lock; needs Docker, takes minutes",
}

HOOK_LINE = re.compile(r"^\s*run_check\s+([\w.]+)", re.M)  # \w takes digits too
CI_LINE = re.compile(r"^\s*- run: uv run python -m ([\w.]+)\s*$", re.M)
STARTS_GIT = re.compile(r"""[\[(]\s*["']git["']""")  # a list or tuple whose first item is "git", as subprocess takes it

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def check_scripts() -> set[str]:
    """Every check script in the project, as the module `python -m` runs."""
    return {f"{p.parent.name}.{p.stem}" for p in PROJECT_ROOT.glob("*/check_*.py") if not p.parent.name.startswith(".")}


def unwired(scripts: set[str], hook: set[str], manual: set[str]) -> list[str]:
    """The check scripts that neither the hook nor a person runs."""
    return sorted(scripts - hook - manual)


def git_starters(texts: Mapping[str, str]) -> set[str]:
    """The files whose code starts git itself."""
    return {path for path, text in texts.items() if STARTS_GIT.search(text)}


def wiring() -> None:
    print("-- every check script runs somewhere --")
    hook = set(HOOK_LINE.findall(HOOK.read_text(encoding="utf-8")))
    ci = set(CI_LINE.findall(CI.read_text(encoding="utf-8")))
    readme = README.read_text(encoding="utf-8")
    scripts = check_scripts()
    check({"analysis.check_chart_v4", "harness.check_sandbox"} <= hook and "analysis.check_chart_v4" in ci,
          "the hook's run_check lines and CI's steps are read, names with a digit among them")  # fmt: skip
    missing = unwired(scripts, hook, set(MANUAL))
    check(not missing, f"every check script is in the hook or named in MANUAL {missing}")
    made_up = {"analysis.check_hooked", "harness.check_by_hand", "analysis.check_not_wired"}
    check(unwired(made_up, {"analysis.check_hooked"}, {"harness.check_by_hand"}) == ["analysis.check_not_wired"],
          "a made-up check script in neither is caught, and one in either isn't")  # fmt: skip
    both = sorted(set(MANUAL) & (hook | ci))
    check(not both, f"no manual check also runs in the hook or CI {both}")
    undocumented = sorted(m for m in MANUAL if f"uv run python -m {m}" not in readme)
    check(not undocumented, f"the README's 'Running it' gives every manual check's command {undocumented}")
    check(ci <= hook, f"CI runs nothing the hook doesn't {sorted(ci - hook)}")
    unlisted, run_after_all = sorted(hook - ci - set(HOOK_ONLY)), sorted(set(HOOK_ONLY) & ci)
    check(not unlisted and not run_after_all,
          f"CI skips exactly the HOOK_ONLY checks: unlisted {unlisted}, run by CI after all {run_after_all}")  # fmt: skip
    named = hook | ci | set(HOOK_ONLY) | set(MANUAL)
    check(named <= scripts, f"every check the hook, CI and these lists name is a script {sorted(named - scripts)}")


def git_calls() -> None:
    print("\n-- git starts only in analysis/checking.py --")
    listed = git("ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py")
    paths = [p for p in listed.stdout.split() if (PROJECT_ROOT / p).is_file()]
    starters = git_starters({p: (PROJECT_ROOT / p).read_text(encoding="utf-8") for p in paths})
    check(listed.returncode == 0 and HELPER in starters, f"the search finds {HELPER}'s own git calls")
    check(starters <= {HELPER}, f"no other module starts git itself {sorted(starters - {HELPER})}")
    made_up = "subprocess.run([" + '"git", "status"], cwd=elsewhere)'  # split so this file doesn't match itself
    check(git_starters({"raw.py": made_up, "helper.py": 'git("status", repo=elsewhere)'}) == {"raw.py"},
          "a made-up raw git call is caught, and a call through the helper isn't")  # fmt: skip


def main() -> int:
    wiring()
    git_calls()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
