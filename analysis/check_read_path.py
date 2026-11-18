"""Check that no module but the unlock reads the sealed store's path (PLAN_v4.md Amendment A19, item 4; issue #78).

Run from the project root:  uv run python -m analysis.check_read_path

`harness/check_sealed.py` keeps the store's file name out of every module but the collection runner, and stays as it
is. This is its second half: the unlock (`analysis/unlock.py`) is the only module that may read the path, so the handle
can't be bypassed unnoticed. Two ways to read it, found in each Python file git sees (tracked or not, ignored aside) by
parsing it, not by matching text, so a comment or a docstring never counts:

- the world's field: any `.sealed_store` attribute (what the unlock opens and hashes);
- the collection module's field: a `.sealed` attribute, or `open_sealed` / `sealed_ids`, in a module that also names
  the collection module's real paths (`MAIN`, `SMOKE`, `RESULTS`). A throwaway `paths_under(folder).sealed`, as
  `harness/check_collect.py` and the made-up world use, names none of those, and `MAIN.sealed.name` (the file's name,
  as `harness/check_sealed.py` takes it) isn't a read of the path. A module that builds the world keyword-wise
  (`World(sealed_store=path)`) reads nothing.

Allowed: the unlock; the collection module (the store's writer); and the go's entry point (#79), which only passes the
path. The search has to find the unlock and the collection module, so it can't pass by matching nothing, and it has to
flag each planted module below, so it can't pass by missing a way of reading.

The field and function names come from the modules themselves (`Paths`, `World`), so this check never spells the
path's field out of nothing. Not covered: a module that builds a `Paths` over the real results folder with
`paths_under`; the store's file name in such a module is `harness/check_sealed.py`'s to catch.
"""

import ast
import dataclasses
import sys
from collections.abc import Mapping

from analysis import unlock
from analysis.checking import git
from analysis.world import World
from harness import collect
from harness.client import PROJECT_ROOT

ALLOWED = {
    "analysis/unlock.py": "the unlock opens and hashes the store",
    "harness/collect.py": "the collection module writes the store",
    "analysis/final_go.py": "the go's entry point (#79) passes the real path into the world",
}
# From the modules' own fields, so a rename shows here as a lookup failure and not as a search that finds nothing.
COLLECTION_FIELD = next(f.name for f in dataclasses.fields(collect.Paths) if "sealed" in f.name)
WORLD_FIELD = next(f.name for f in dataclasses.fields(World) if "sealed" in f.name)
COLLECTION_FUNCTIONS = {collect.open_sealed.__name__, collect.sealed_ids.__name__}
REAL_PATHS = {"MAIN", "SMOKE", "RESULTS"}  # the collection module's constants that name real files

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def readers_in(source: str) -> list[str]:
    """How `source` reads the sealed store's path, one reason each (empty if it doesn't)."""
    tree = ast.parse(source)
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom | ast.Import):
            names.update(alias.name.split(".")[-1] for alias in node.names)
    names_real_paths = bool(names & REAL_PATHS)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute | ast.Name | ast.ImportFrom):
            continue
        if isinstance(node, ast.Attribute) and node.attr == WORLD_FIELD:
            found.append(f"line {node.lineno}: reads the world's {WORLD_FIELD}")
        elif isinstance(node, ast.Attribute) and node.attr == COLLECTION_FIELD and names_real_paths:
            above = parent.get(node)
            name_only = isinstance(above, ast.Attribute) and above.attr == "name" and above.value is node
            if not name_only:
                found.append(f"line {node.lineno}: reads the collection module's {COLLECTION_FIELD} path")
        elif names_real_paths and (
            (isinstance(node, ast.Attribute) and node.attr in COLLECTION_FUNCTIONS)
            or (isinstance(node, ast.Name) and node.id in COLLECTION_FUNCTIONS)
            or (isinstance(node, ast.ImportFrom) and any(a.name in COLLECTION_FUNCTIONS for a in node.names))
        ):
            found.append(f"line {node.lineno}: opens the store through the collection module")
    return found


def readers(texts: Mapping[str, str]) -> set[str]:
    """The files whose code reads the store's path."""
    return {path for path, text in texts.items() if readers_in(text)}


def main() -> int:
    print("-- the search finds a read of the path in each way there is --")
    planted = {
        "plain.py": f"from harness.collect import MAIN\ndata = MAIN.{COLLECTION_FIELD}.read_bytes()\n",
        "aliased.py": f"import harness.collect as c\nstore = c.MAIN\nraw = store.{COLLECTION_FIELD}.open('rb')\n",
        "world.py": f"def peek(world):\n    return world.{WORLD_FIELD}.read_bytes()\n",
        "opens.py": f"from harness.collect import MAIN, {collect.open_sealed.__name__}\nconn = open_sealed(MAIN)\n",
        "smoke.py": f"from harness.collect import SMOKE\nSMOKE.{COLLECTION_FIELD}.stat()\n",
    }
    for name, text in planted.items():
        check(readers({name: text}) == {name}, f"a planted module reading the path is found: {name}")
    clean = {
        "name_only.py": f"from harness.collect import MAIN\nname = MAIN.{COLLECTION_FIELD}.name\n",
        "throwaway.py": f"from harness.collect import paths_under\nsize = paths_under(folder).{COLLECTION_FIELD}.stat()\n",
        "builds.py": f"from analysis.world import World\nworld = World(root, bench, out, out, {WORLD_FIELD}=path)\n",
        "words.py": f'"""MAIN.{COLLECTION_FIELD} and world.{WORLD_FIELD} in a docstring."""\n# MAIN.{COLLECTION_FIELD}\n',
    }
    for name, text in clean.items():
        check(readers({name: text}) == set(), f"a module that reads no path isn't flagged: {name}")

    print("\n-- the project's own modules --")
    listed = git("ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py")
    paths = [p for p in listed.stdout.split() if (PROJECT_ROOT / p).is_file()]
    found = readers({p: (PROJECT_ROOT / p).read_text(encoding="utf-8") for p in paths})
    check(listed.returncode == 0 and unlock.__name__.replace(".", "/") + ".py" in found,
          "the search finds the unlock, its known positive")  # fmt: skip
    check("harness/collect.py" in found, "the search finds the collection module, the writer, too")
    others = sorted(found - set(ALLOWED))
    check(not others, f"no other module reads the sealed store's path {others}")
    missing = sorted(name for name in ALLOWED if not (PROJECT_ROOT / name).is_file() and name != "analysis/final_go.py")
    check(not missing, f"every allowed module exists, the go's entry point (not built yet) aside {missing}")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
