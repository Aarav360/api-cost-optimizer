"""The rules router's freeze: its import closure and the hashes (PLAN_v4.md §8; Amendment A9, item 13).

The closure is worked out from the imports, starting at the rules router: every `routers/` module it imports,
directly or through another, with each package's `__init__.py`. `analysis/check_rules_router.py` fails if the list
recorded in Amendment A9 item 17 differs from it, so a module imported later can't sit outside the freeze. Hashes are
sha256 of the file's bytes, as in A7 item 3.
"""

import ast
import hashlib
from pathlib import Path

from harness.client import PROJECT_ROOT

ENTRY = "routers/rules.py"
RATES = "routers/rules_rates.json"


def module_files(root: Path, dotted: str) -> list[str]:
    """The files a dotted module name stands for under root: each package on the way, then the module itself."""
    found = []
    parts = dotted.split(".")
    for end in range(1, len(parts) + 1):
        path = root.joinpath(*parts[:end])
        if (path / "__init__.py").exists():
            found.append((path / "__init__.py").relative_to(root).as_posix())
        elif path.with_suffix(".py").exists():
            found.append(path.with_suffix(".py").relative_to(root).as_posix())
    return found


def imports_of(root: Path, relative: str) -> list[str]:
    """The files under root's `routers` package that a file imports (a package's own import of it included)."""
    package = relative.split("/")[:-1]
    found: list[str] = []
    for node in ast.walk(ast.parse((root / relative).read_text())):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = (
                ".".join(package[: len(package) - node.level + 1] + ([node.module] if node.module else []))
                if node.level
                else node.module
            )
            if base:
                # `from routers import ladder` imports the module `routers.ladder`, if there is one.
                names = [base, *(f"{base}.{alias.name}" for alias in node.names)]
        for name in names:
            if name.split(".")[0] == "routers":
                found += module_files(root, name)
    return found


def closure(entry: str = ENTRY, root: Path = PROJECT_ROOT) -> list[str]:
    """The entry module and every `routers/` file it imports, sorted, `__init__.py` included."""
    seen: set[str] = set()
    todo = [entry]
    while todo:
        relative = todo.pop()
        if relative not in seen:
            seen.add(relative)
            todo += imports_of(root, relative)
    return sorted(seen)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Each module in the closure and the rates file, by path from the project root."""
    return {relative: sha256(root / relative) for relative in [*closure(root=root), RATES]}
