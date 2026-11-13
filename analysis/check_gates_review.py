"""Check Amendment A18's records against their sources (PLAN_v4.md; Amendment A18; the Stage 7 branch review).

Run from the project root:  uv run python -m analysis.check_gates_review

Calls no Jev and no Anthropic API, reads no stored record and no test item, so CI runs it. A18 says which files the two
freezes hash, what the review recorded in them, what the history shows about chart v3 and about changes made outside
A17's items, and what it changed in the figures; each is compared with its source, as CODING_STANDARDS.md's "Numbers in
documents" asks:

- the amendment: it is in the plan, with a version-history row and the header's Amended row, eight items and no ninth;
- item 1's lists of hashed files are the ones the freeze checks hold (known positive: a list that lost a file is found),
  the fixes it names are in the code it names, the checks it says still hash or run git themselves do, and its numbers
  are the committed records' (dev's calls and verdicts from
  `analysis/gates_dev_record.json`, the 299 tokens from the smoke record);
- item 2's recorded findings still hold in the hashed files they name: each symbol, string and import is there (known
  positive: the same searches fail on what isn't);
- item 3, where the clone holds the history: the prefactor's merge changed the four files it names and nothing under
  `figures/`, removed only the two lines it names from `analysis/chart_routers.py`, only its docstring's first line from
  `analysis/router_names.py`, and in `analysis/operating_points.py` widened the allowlist and removed no function (known
  positive: the commit that first drew the router-started figure did change `figures/`);
- item 4: the two commits it names changed the files it says, and A17 now holds the three corrected numbers and not the
  old ones (the history where the clone holds it);
- items 5 to 8: item 29 holds its operating points' table, the headline code draws every fall back, the Sonnet gate's
  colour and its contrast are the ones item 7 quotes, and the switch item 8 keeps refuses any other gate.

Where the clone has no history (CI's checkout is shallow) the sections that need it print a SKIP line.
"""

import ast
import json
import re
import sys
from collections.abc import Mapping
from typing import Any

from analysis.chart_routers import CASCADE_STYLES
from analysis.check_gates_settings_commit import COMMIT_FILES, SETTINGS, SMOKE_RECORD
from analysis.check_jev_names import contrast
from analysis.check_jev_settings_commit import settings_commit
from analysis.checking import full_history, git
from analysis.gate_dev_run import STORED as DEV_STORED
from analysis.gate_table import GateRow, gate_problem
from analysis.gate_train_report import CUTOFFS_FILE
from analysis.gate_train_report import STORED as TRAIN_STORED
from analysis.pareto import SURFACE
from analysis.plan import PLAN, amendment_item
from analysis.router_names import SONNET_GATE
from harness.client import PROJECT_ROOT

ITEMS = 8
LISTED = re.compile(r"`((?:analysis|routers|sandbox|results)/[\w./-]+)`")
COMMIT = re.compile(r"`([0-9a-f]{40})`")
problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def read(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def bullet(item: str, marker: str) -> str:
    """The bullet of `item` that opens with `marker`, up to the next bullet at its depth; empty if there is none."""
    start = item.find(marker)
    return "" if start == -1 else item[start:].split("\n    - ", 1)[0]


def listed(text: str) -> set[str]:
    """The paths a bullet lists, after its parenthesis: the list of files, not the scripts the parenthesis names."""
    return set(LISTED.findall(text.split("):", 1)[-1]))


def defined(source: str) -> set[str]:
    """The classes, functions and top-level names a module defines."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
    return names


def imported(source: str) -> set[str]:
    """The modules a module imports, by their dotted names."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def keys_holding(data: Any, word: str) -> list[str]:
    """Every key of a JSON value, at any depth, that holds `word`."""
    if isinstance(data, Mapping):
        return [k for k in data if word in k] + [k for v in data.values() for k in keys_holding(v, word)]
    if isinstance(data, list):
        return [k for v in data for k in keys_holding(v, word)]
    return []


def amendment(plan: str) -> None:
    print("-- Amendment A18 is in the plan --")
    check("\n## Amendment A18 — Stage 7 review: what the freezes hash, and records of what the code does (2026-11-11)\n" in plan
          and "\n| v4-A18 | 2026-11-11 | " in plan,
          "A18 has its heading and a version-history row")  # fmt: skip
    check("A18, the Stage 7 review's records" in plan.split("\n## ", 1)[0], "the header's Amended row names A18")
    check(all(amendment_item(plan, "A18", n).startswith(f"{n}. **") for n in range(1, ITEMS + 1)),
          f"A18 has items 1 to {ITEMS}")  # fmt: skip
    try:
        amendment_item(plan, "A18", ITEMS + 1)
        check(False, f"there is no item {ITEMS + 1}")
    except ValueError:
        check(
            True,
            f"known negative: there is no item {ITEMS + 1}, and asking for one is refused, so the count means something",
        )
    check(plan.index("## Amendment A18") > plan.index("## Amendment A17") and "Eight records" in amendment_text(plan),
          "A18 follows A17 and says it holds eight records")  # fmt: skip


def amendment_text(plan: str) -> str:
    return re.split(r"\n## |\n---\n", plan[plan.index("## Amendment A18") :], maxsplit=1)[0]


def hashed_files(item: str) -> None:
    print("\n-- item 1: the files the freezes hash, and what the review fixed outside them --")
    settings_bullet = bullet(item, "    - **The settings commit's files**")
    check(listed(settings_bullet) == set(COMMIT_FILES) and len(COMMIT_FILES) == 18,
          f"the settings commit's {len(COMMIT_FILES)} files are those of `COMMIT_FILES`")  # fmt: skip
    shorter = settings_bullet.replace("`routers/ladder.py`, ", "", 1)
    check(shorter != settings_bullet and listed(shorter) != set(COMMIT_FILES),
          "known positive: a list that lost `routers/ladder.py` is not the settings commit's")  # fmt: skip
    freeze_bullet = bullet(item, "    - **The freeze commit's files**")
    check(listed(freeze_bullet) == {CUTOFFS_FILE, *TRAIN_STORED} and "those eighteen" in freeze_bullet,
          "the freeze commit's are those eighteen, the cutoffs file and train's four stored record files (item 28's)")  # fmt: skip
    after = bullet(item, "    - **Hashed after the freeze:**")
    check(set(LISTED.findall(after)) >= {f"results/gates/{name}" for name in DEV_STORED}
          | {"analysis/gates_dev_record.json", "routers/operating_points.json"},
          "after the freeze, the dev record's three stored files and the operating-points file are named")  # fmt: skip
    check(listed("no such bullet") == set() and bullet(item, "    - **No such bullet**") == "",
          "a missing bullet lists nothing")  # fmt: skip

    named = {
        "analysis/router_names.py": "parse_cascade",
        "analysis/gate_table.py": "gate_problem",
        "analysis/checking.py": "git_bytes",
        "analysis/freeze.py": "sha256",
        "analysis/gate_spend.py": "billed_tokens",
        "analysis/cascade_comparisons.py": "ROUTER_LETTERS",
    }
    check(all(name in defined(read(path)) and f"`{name}`" in item for path, name in named.items())
          and "full_history" in defined(read("analysis/checking.py")),
          "each fix item 1 names is in the module it names")  # fmt: skip
    copies = [
        p.name for p in sorted((PROJECT_ROOT / "analysis").glob("*.py")) if "file_sha256" in defined(p.read_text())
    ]
    check(copies == [] and "NoSuchName" not in defined(read("analysis/freeze.py")),
          f"no module defines its own file hash any more {copies}, and the search fails for a name nothing defines")  # fmt: skip
    hashing = ["analysis/check_gates_freeze.py", "analysis/check_stage7.py", "analysis/check_jev_freeze.py"]
    check(all("hashlib" in imported(read(path)) and f"`{path}`" in item for path in hashing)
          and "sha256_of" in defined(read(hashing[0])) and "`sha256_of`" in item,
          "the checks item 1 says still hash with hashlib directly do, `sha256_of` among them")  # fmt: skip
    raw = [
        "analysis/check_jev_freeze.py",
        "analysis/check_jev_dev.py",
        "analysis/check_trained_fit.py",
        "harness/check_sealed.py",
    ]
    starts_git = r'subprocess\.run\(\s*\[\s*"git"'
    check(bool(re.search(starts_git, read("analysis/checking.py"))) and not any(re.search(starts_git, read(path)) for path in raw)
          and all(f"`{path}`" in item for path in raw) and "`analysis/check_guardrails.py`" in item
          and "subprocess" not in imported(read("analysis/check_stage7.py")),
          "the checks item 1 says moved onto the helper run no git themselves (the search finds the helper's own), "
          "nor does `analysis/check_stage7.py`, and item 1 names the check that keeps it so")  # fmt: skip
    record = json.loads(read("analysis/gates_dev_record.json"))
    calls = sum(c["jev_calls"] for c in record["counts"].values())
    verdicts = sum(c["sonnet_calls"] for c in record["counts"].values())
    check(f"dev's {calls:,} Jev calls and {verdicts:,} Sonnet verdicts" in item,
          f"item 1 quotes dev's {calls:,} Jev calls and {verdicts:,} Sonnet verdicts, as the dev run's record counts them")  # fmt: skip
    measured = json.loads(read(SMOKE_RECORD))["measured"]["jev_fixed_overhead_tokens"]
    check(f"the {measured} tokens the smoke test measured" in item and f"the {measured + 1} tokens" not in item,
          f"item 1 quotes the {measured} tokens the smoke test measured, and not a number one off")  # fmt: skip
    hook = read(".githooks/pre-commit")
    check("run_check harness.check_sandbox" in hook
          and "harness.check_sandbox" not in read(".github/workflows/check.yml"),
          "the hook runs harness/check_sandbox.py and CI doesn't (it needs Docker)")  # fmt: skip


def recorded_findings(item: str) -> None:
    print("\n-- item 2: what the review recorded in hashed files is still so --")
    check(all(path in COMMIT_FILES for path in ("routers/gates.py", SETTINGS, "routers/sandbox.py", "routers/sonnet_gate.py",
                                                "routers/jev.py", "analysis/gate_cutoff.py")),
          "every file item 2 records a finding in is one the settings commit froze")  # fmt: skip
    gates = read("routers/gates.py")
    check("zip(FREE_FIELDS, (" in gates and "strict=True" in gates and "FREE_FIELDS" in defined(gates),
          "`free_checks` zips `FREE_FIELDS` with a tuple, strict")  # fmt: skip
    settings = json.loads(read(SETTINGS))
    check("cascades" in settings and "CascadeShape" in defined(gates) and "settings.cascades" in read("analysis/check_gates.py"),
          "the settings file and `CascadeShape` restate the cascades, and check_gates holds them equal")  # fmt: skip
    check("_was_right" in defined(read("analysis/gate_cutoff.py")) and "_success_of" in defined(read("analysis/gate_metrics.py")),
          "`_was_right` and `_success_of` are both there")  # fmt: skip
    head = (ast.get_docstring(ast.parse(read("routers/sandbox.py"))) or "").split("\n\n", 1)[0]
    check(head != "" and not re.search(r"PLAN_v4|Amendment|A1\d|§", head),
          "the sandbox's docstring head names no section or amendment")  # fmt: skip
    check(re.search(r"PLAN_v4|Amendment|A1\d|§", "Run code (PLAN_v4.md §9; Amendment A17, item 4).") is not None,
          "known positive: a head that cites the plan is found")  # fmt: skip
    check(keys_holding(settings, "refus") == [] and keys_holding({"a": [{"refusal_rules": 1}]}, "refus") == ["refusal_rules"],
          "the settings file holds no refusal rule, and the search finds one in a made-up file")  # fmt: skip
    check('refused("unparseable")' in read("routers/sonnet_gate.py") and "parse_answers" in defined(read("routers/jev.py")),
          "the refusal rules are in `routers/sonnet_gate.py` and the frozen `parse_answers`")  # fmt: skip
    cutoff = imported(read("analysis/gate_cutoff.py"))
    check({"analysis.gate_table", "analysis.matrix"} <= cutoff
          and not {"analysis/gate_table.py", "analysis/matrix.py"} & set(COMMIT_FILES),
          "the cutoff code imports `analysis/gate_table.py` and `analysis/matrix.py`, which no freeze hashes")  # fmt: skip
    check("analysis/gate_cutoff.py" in item and "item 28" in item, "item 2 names the cutoff code and item 28")


def prefactor(item: str) -> None:
    print("\n-- item 3: chart v3 and the prefactor, from history --")
    found = COMMIT.findall(item)
    if not full_history():
        print("SKIP  the prefactor's merge: this clone has no history (CI's checkout is shallow)")
        return
    merge = found[0] if found else ""
    subject = git("show", "-s", "--format=%s%n%P", merge).stdout.splitlines()
    check(len(found) == 1 and len(subject) == 2 and subject[0].endswith("(#60)") and len(subject[1].split()) == 2,
          f"item 3 names the prefactor's merge, {merge[:10]}: a merge whose subject ends (#60)")  # fmt: skip
    if len(subject) != 2:
        return
    changed = git("diff", "--name-only", f"{merge}^1", merge).stdout.split()
    four = [
        "analysis/chart_routers.py",
        "analysis/check_jev_names.py",
        "analysis/operating_points.py",
        "analysis/router_names.py",
    ]
    check(changed == four and all(f"`{path}`" in item for path in four),
          "it changed the four files item 3 names, and no other")  # fmt: skip
    check(git("diff", "--quiet", f"{merge}^1", merge, "--", "figures/").returncode == 0,
          "and left every file under figures/ byte for byte as its first parent held it")  # fmt: skip
    drawn_later = settings_commit(PROJECT_ROOT, "figures/pareto_routed.json")
    check(drawn_later is not None and git("diff", "--quiet", f"{drawn_later}^1", drawn_later, "--", "figures/").returncode == 1,
          "known positive: the commit that first drew the router-started figure did change figures/")  # fmt: skip
    diff = git("diff", f"{merge}^1", merge, "--", "analysis/chart_routers.py").stdout.splitlines()
    removed = [line[1:].strip() for line in diff if line.startswith("-") and not line.startswith("---")]
    check(removed == ["from analysis.router_names import G_PLUS, LABELS, NAMED_ROUTERS, E, F, G",
                      "linestyle: str  # matplotlib line style of the steps' line"],
          "in analysis/chart_routers.py it removed only the import it widened and the line-style field's old type")  # fmt: skip
    names = git("diff", f"{merge}^1", merge, "--", "analysis/router_names.py").stdout.splitlines()
    gone = [line[1:] for line in names if line.startswith("-") and not line.startswith("---")]
    check(len(gone) == 1 and gone[0].startswith('"""The routers\' names, in one place') and "only added the cascades' names" in item,
          "in analysis/router_names.py it removed only its docstring's first line, which it widened: it only added")  # fmt: skip
    points = git("diff", f"{merge}^1", merge, "--", "analysis/operating_points.py").stdout.splitlines()
    dropped = [line[1:].strip() for line in points if line.startswith("-") and not line.startswith("---")]
    check("allowed = set(NAMED_ROUTERS.values())" in dropped and not any(line.startswith("def ") for line in dropped)
          and "widened the allowlist of `check_sections`" in item,
          "in analysis/operating_points.py it widened `check_sections`' allowlist and removed no function")  # fmt: skip


def outside_changes(item: str, plan: str) -> None:
    print("\n-- item 4: changes outside A17's items --")
    a17 = {n: amendment_item(plan, "A17", n) for n in (3, 4, 20)}
    check("in 5,949 rows" in a17[3] and "6,549" not in a17[3] and "for 140 of them" in a17[4]
          and "for 135 of them" not in a17[4] and "about 3.4× as long as MMLU-Pro's, in mean input tokens" in a17[20]
          and "about 4× as long" not in a17[20],
          "A17 holds the three corrected numbers, and not the old ones")  # fmt: skip
    check(all(text in item for text in ("6,549 to 5,949", "135 to 140", '"about 4×" to "about 3.4×, in mean input tokens"')),
          "item 4 names each correction, old number to new")  # fmt: skip
    if not full_history():
        print("SKIP  the two commits item 4 names: this clone has no history (CI's checkout is shallow)")
        return
    fix = git("show", "--name-only", "--format=", "429cfa8").stdout.split()
    check("`429cfa8`" in item and sorted(fix) == ["analysis/check_jev_settings_commit.py", "analysis/checking.py"],
          "429cfa8 changed `analysis/checking.py` and `analysis/check_jev_settings_commit.py`, no other")  # fmt: skip
    before = git("show", "6a933a5^:PLAN_v4.md").stdout
    after = git("show", "6a933a5:PLAN_v4.md").stdout
    check("`6a933a5`" in item and "in 6,549 rows" in before and "in 5,949 rows" in after and "for 135 of them" in before
          and "for 140 of them" in after,
          "6a933a5 made the corrections: the plan before it held the old numbers, and after it the new ones")  # fmt: skip


def later_items(items: Mapping[int, str], plan: str) -> None:
    print("\n-- items 5 to 8 --")
    record = amendment_item(plan, "A17", 29)
    check("    - **The operating points.**" in record and "      | haiku/Jev: cutoff at 100%, 99%, 97% of A's quality |" in record,
          "item 5: A17 item 29 holds the operating points' table")  # fmt: skip
    pareto = read("analysis/pareto.py")
    check("kept_points" not in defined(pareto) and "fall_back_note" not in defined(pareto)
          and "point_labels(part[\"operating_points\"])" in pareto,
          "item 6: the headline code draws every operating point, falls back to A included, and has no note instead")  # fmt: skip
    headline = json.loads(read("figures/pareto.json"))
    fell = sum(p["fell_back"] for panel in headline["panels"].values()
               for part in [*panel["routers"].values(), *(c for c in panel["cascades"].values() if "point" not in c)]
               for p in part["operating_points"].values())  # fmt: skip
    check(fell == 10 and "so ten operating points had none" in items[6],
          f"item 6: the headline's routers and pure cascades with crosses fall back to A at {fell} operating points, "
          "the ten item 6 counts")  # fmt: skip
    colour = CASCADE_STYLES[SONNET_GATE].color
    quoted = re.findall(r"`(#[0-9a-f]{6})`, (\d\.\d\d):1", items[7])
    old = re.findall(r"`(#[0-9a-f]{6})` stood (\d\.\d\d):1", items[7])
    check(quoted == [(colour, f"{contrast(colour, SURFACE):.2f}")]
          and old == [("#cca4f8", f"{contrast('#cca4f8', SURFACE):.2f}")],
          f"item 7 quotes the Sonnet gate's colour {colour} and both contrasts as they compute")  # fmt: skip
    check(re.findall(r"`(#[0-9a-f]{6})`, (\d\.\d\d):1", "is now `#bfa0ee`, 2.17:1") != quoted,
          "known positive: a contrast one hundredth off is not the record's")  # fmt: skip
    fields = set(GateRow.__dataclass_fields__)
    check({"jev", "sonnet"} <= fields and gate_problem("oracle", 0.5) is not None and gate_problem("oracle", None) is not None,
          "item 8: the row keeps its two typed fields, and the one guard refuses any other gate")  # fmt: skip


def main() -> int:
    plan = PLAN.read_text(encoding="utf-8")
    amendment(plan)
    items = {n: amendment_item(plan, "A18", n) for n in range(1, ITEMS + 1)}
    hashed_files(items[1])
    recorded_findings(items[2])
    prefactor(items[3])
    outside_changes(items[4], plan)
    later_items(items, plan)
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
