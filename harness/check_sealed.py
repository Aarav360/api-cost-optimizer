"""Check that no code but harness/collect.py names the sealed test store (AGENTS.md; Amendment A6, item 4).

Run from the project root:  uv run python -m harness.check_sealed

Searches every file git sees, tracked or not (ignored ones such as results/ aside),
for the sealed store's file name. The name comes from the collection runner's own
paths, so this check never spells it out either. Besides harness/collect.py, only
the record of its hash (data/collection_manifest.json) and the Markdown docs that
describe it may name it. The search has to find harness/collect.py, so it can't
pass by matching nothing.
"""

import subprocess
import sys

from harness.client import PROJECT_ROOT
from harness.collect import MAIN

ALLOWED = {"harness/collect.py", "data/collection_manifest.json"}  # and every Markdown file


def naming_files() -> list[str]:
    """Every file outside ALLOWED and the Markdown docs that names the sealed store."""
    search = subprocess.run(["git", "grep", "-l", "--untracked", "-F", MAIN.sealed.name], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=False)  # fmt: skip
    found = search.stdout.split("\n")
    if "harness/collect.py" not in found:
        raise RuntimeError(f"the search didn't find harness/collect.py, so it can't be trusted: {search.stderr}")
    return sorted(name for name in found if name and name not in ALLOWED and not name.endswith(".md"))


def main() -> int:
    named = naming_files()
    print(f"{'PASS' if not named else 'FAIL'}  only harness/collect.py names the sealed store in code {named}")
    print("\nall checks passed" if not named else "\n1 check failed")
    return 1 if named else 0


if __name__ == "__main__":
    sys.exit(main())
