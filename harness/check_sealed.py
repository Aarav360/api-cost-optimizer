"""Check that no code but harness/collect.py names the sealed test store (AGENTS.md; Amendment A6, item 4).

Run from the project root:  uv run python -m harness.check_sealed

Searches every file git sees, tracked or not (ignored ones such as results/ aside),
for the sealed store's file name. The name comes from the collection runner's own
paths, so this check never spells it out either. Besides harness/collect.py, only
the record of its hash (data/collection_manifest.json) and the Markdown docs that
describe it may name it. The search has to find harness/collect.py, so it can't
pass by matching nothing.

A second section guards the benchmark file the same way (Amendment A9, item 7): the reader in
analysis/benchmark.py is the only path from it to a router, so besides the four harness modules that
name it today (the build script, its check, the collection runner and the collection review), only that
reader, the benchmark manifest (whose hash keys hold the file name) and the Markdown docs may name its
constant or its file name. That search must find the build script, so it can't pass by matching nothing.
"""

import inspect
import subprocess
import sys

from analysis.benchmark import read_rows
from harness import build_benchmark
from harness.client import PROJECT_ROOT
from harness.collect import MAIN

ALLOWED = {"harness/collect.py", "data/collection_manifest.json"}  # and every Markdown file
BENCHMARK_ALLOWED = {
    "harness/build_benchmark.py", "harness/check_benchmark.py", "harness/collect.py", "harness/review_collection.py",
    "analysis/benchmark.py", "data/benchmark_manifest.json",
}  # fmt: skip  # and every Markdown file
# The constant's name and the file's, both taken from the module so this check never spells either out:
# a path written out by hand is caught as surely as the constant.
BENCHMARK_PATH = inspect.signature(read_rows).parameters["path"].default  # the reader's default: the frozen benchmark
BENCHMARK_NAMES = [next(k for k, v in vars(build_benchmark).items() if v is BENCHMARK_PATH), BENCHMARK_PATH.name]


def naming_files() -> list[str]:
    """Every file outside ALLOWED and the Markdown docs that names the sealed store."""
    search = subprocess.run(["git", "grep", "-l", "--untracked", "-F", MAIN.sealed.name], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=False)  # fmt: skip
    found = search.stdout.split("\n")
    if "harness/collect.py" not in found:
        raise RuntimeError(f"the search didn't find harness/collect.py, so it can't be trusted: {search.stderr}")
    return sorted(name for name in found if name and name not in ALLOWED and not name.endswith(".md"))


def benchmark_naming_files() -> list[str]:
    """Every file outside BENCHMARK_ALLOWED and the Markdown docs that names the benchmark file or its constant."""
    patterns = [arg for name in BENCHMARK_NAMES for arg in ("-e", name)]
    search = subprocess.run(["git", "grep", "-l", "--untracked", "-F", *patterns], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=False)  # fmt: skip
    found = search.stdout.split("\n")
    if "harness/build_benchmark.py" not in found:
        raise RuntimeError(
            f"the search didn't find harness/build_benchmark.py, so it can't be trusted: {search.stderr}"
        )
    return sorted(name for name in found if name and name not in BENCHMARK_ALLOWED and not name.endswith(".md"))


def main() -> int:
    problems = []

    def check(ok: bool, what: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            problems.append(what)

    print("-- the sealed store's name --")
    named = naming_files()
    check(not named, f"only harness/collect.py names the sealed store in code {named}")
    print("\n-- the benchmark file's name --")
    named = benchmark_naming_files()
    check(not named, f"only the allowlisted modules name the benchmark file or its constant in code {named}")
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
