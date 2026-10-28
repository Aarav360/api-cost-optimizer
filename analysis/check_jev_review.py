"""Check Amendment A16's records against their sources (PLAN_v4.md; Amendment A16; the Stage 6 branch review).

Run from the project root:  uv run python -m analysis.check_jev_review

Calls no Jev and reads no test item. A16 says which files the two freezes hash, what the review recorded in them and
what the code does that A15 does not say, and quotes numbers. Each of those is compared with its source, as
CODING_STANDARDS.md's "Numbers in documents" asks:

- the amendment: it is in the plan, with a version-history row, nine items and no tenth;
- item 1's two lists of hashed files are the ones the two freeze checks work out (known positive: a list missing a file
  is found);
- items 2, 5, 6 and 7, the recorded findings, still hold in the files they name: each symbol, string and import is
  there (known positive: the same search fails for a name that isn't);
- item 4: the settings commit's own change adds the four files it says, and the other four were last committed by the
  commits it names (where those commits are present);
- item 8's numbers: the budgets are the settings file's, the overhead, past-request and schema figures the smoke
  record's, and the longest train and dev prompts the benchmark reader's (the prompts alone, no outcome), with the
  margins worked from them (known positive: a number with one digit changed is not in the record);
- item 7's count of refusals (where `results/jev` holds dev's responses) and item 9's two sentences.

Item 3's numbers, the machine, network and dates, are compared with table v3 by `analysis/check_stage6.py`.
"""

import ast
import json
import re
import sys

from analysis.check_jev_freeze import frozen_paths
from analysis.check_jev_settings_commit import COMMIT_FILES, SETTINGS, SMOKE_RECORD, settings_commit
from analysis.checking import git, refused
from analysis.freeze import closure
from analysis.plan import PLAN, amendment_item
from analysis.router_names import G_PLUS, NAMED_ROUTERS, G
from harness.client import PROJECT_ROOT
from routers import jev_call

problems: list[str] = []
LISTED = re.compile(r"`((?:analysis|routers)/[\w./-]+\.(?:py|json))`")
HOOK = PROJECT_ROOT / ".githooks" / "pre-commit"


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def listed_after(item: str, marker: str) -> set[str]:
    """The paths in the bullet that opens with `marker`, after its parenthesis: the list of files, not the scripts the
    parenthesis names. Empty if there is no such bullet."""
    start = item.find(marker)
    if start == -1:
        return set()
    bullet = item[start:].split("\n    - ", 1)[0]
    return set(LISTED.findall(bullet.split("):", 1)[-1]))


def defined(source: str) -> set[str]:
    """The classes, functions and string-valued names a module defines at its top level, and the methods of its classes
    as `Class.method`."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            names |= {node.name, *(f"{node.name}.{m.name}" for m in node.body if isinstance(m, ast.FunctionDef))}
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    return names


def read(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def amendment() -> None:
    print("-- Amendment A16 is in the plan --")
    plan = PLAN.read_text(encoding="utf-8")
    check("## Amendment A16 — " in plan and "\n| v4-A16 | " in plan, "A16 has its heading and a version-history row")
    check("A16, the Stage 6 review's records" in plan.split("\n## ", 1)[0], "the header's Amended row names A16")
    check(all(amendment_item(plan, "A16", n).startswith(f"{n}. **") for n in range(1, 10)), "A16 has items 1 to 9")
    check(refused(ValueError, lambda: amendment_item(plan, "A16", 10)),
          "known negative: there is no item 10, and asking for one is refused, so the count means something")  # fmt: skip


def hashed_files(item: str) -> None:
    print("\n-- item 1: the files the two freezes hash --")
    settings_list = listed_after(item, "    - **The settings commit's files**")
    freeze_list = listed_after(item, "    - **The freeze commit's files**")
    check(settings_list == set(COMMIT_FILES),
          f"the settings commit's {len(COMMIT_FILES)} files are those of `COMMIT_FILES`")  # fmt: skip
    check(freeze_list == set(frozen_paths()),
          f"the freeze commit's {len(frozen_paths())} files are those `frozen_paths` works out from `routers/jev.py`'s "
          "imports, the builder, the fit script and the two JSON files")  # fmt: skip
    missing_one = item.replace(
        "`routers/ladder.py`, `routers/request.py`, `analysis/build", "`routers/request.py`, `analysis/build", 1
    )
    check(missing_one != item and listed_after(missing_one, "    - **The settings commit's files**") != set(COMMIT_FILES),
          "known positive: a list that lost `routers/ladder.py` is not the settings commit's")  # fmt: skip
    check(
        listed_after("no such bullet", "    - **The settings commit's files**") == set(),
        "a missing bullet lists nothing",
    )
    check("results/jev/" in item and "train's two stored responses" in item,
          "item 1 names train's stored responses, which item 19 hashes outside the repository")  # fmt: skip


def recorded_findings(plan_items: dict[int, str]) -> None:
    print("\n-- items 2, 5, 6 and 7: what the review recorded is still so --")
    fit = defined(read("analysis/fit_jev_router.py"))
    trained = defined(read("analysis/fit_trained_router.py"))
    duplicated = {"concordance", "fit_model", "mean_fold_loss", "choose_c", "ChosenC"}
    check(duplicated <= fit and duplicated <= trained and all(n in plan_items[2] for n in duplicated),
          "item 2: the fit script repeats F's `concordance`, `fit_model`, `mean_fold_loss`, `choose_c` and `ChosenC`")  # fmt: skip
    check({"G", "G_PLUS", "G_PLUS_RECORDS", "G_RECORDS"} <= fit and 'G = "G"' in read("analysis/fit_jev_router.py")
          and f'G_PLUS = "{G_PLUS}"' in read("analysis/fit_jev_router.py"),
          "item 2: the fit script spells G, G+ and the records' router names out")  # fmt: skip
    check(f'"{NAMED_ROUTERS[G_PLUS]}"' in read("routers/jev_call.py") and f'"{NAMED_ROUTERS[G_PLUS]}"' in read("analysis/smoke_jev.py")
          and f'"{NAMED_ROUTERS[G_PLUS]}"' in read("analysis/build_jev_settings.py"),
          "item 2: the call building, the smoke test and the builder spell the settings file's budget keys out")  # fmt: skip
    router = defined(read("routers/jev.py"))
    shared = {"GRouter.ask", "GPlusRouter.ask", "GScores", "Counts", "Asked", "Exchange"}
    check(
        shared <= router, "item 2: `routers/jev.py` holds both `ask` bodies, `GScores` and the collector's bookkeeping"
    )
    check(not {"NoSuchRouter.ask"} <= router, "known positive: the search fails for a name the file does not define")
    check(all(name in plan_items[2] for name in ("GRouter", "GPlusRouter", "GScores", "Counts", "Asked", "Exchange")),
          "item 2 names each of them")  # fmt: skip

    check("routers/jev.py" not in COMMIT_FILES and "routers/jev_call.py" in COMMIT_FILES,
          "item 5: the settings commit froze `routers/jev_call.py`, not `routers/jev.py`")  # fmt: skip
    call_closure = {f for f in closure("routers/jev_call.py") if f != "routers/__init__.py"}
    check(call_closure == {"routers/jev_call.py", "routers/ladder.py", "routers/request.py"}
          and call_closure <= set(COMMIT_FILES),
          "item 5: the call building's import closure, `__init__.py` apart, is the settings commit's three router files")  # fmt: skip
    check("routers/jev_call.py" in closure("routers/jev.py") and "routers/jev.py" not in closure("routers/jev_call.py"),
          "item 5: `routers/jev.py` imports `routers/jev_call.py` and not the other way round")  # fmt: skip

    client = read("routers/jev.py")
    for text, what in (
        ("retry-after-ms", "`retry-after-ms` is honoured"),
        ("plain http is for loopback only", "plain http is refused except to loopback"),
        ('replace(self._key, "<key>")', "the key is struck out of a quoted reply"),
        ('FailedAttempt(200, "unreadable response")', "an unreadable 200 is a failed attempt"),
    ):
        check(text in client, f"item 6: {what}")
    check("The refusal rule (A15 item 9)" in client and "answers` is not an object" in client,
          "item 7: the refusal rule is in `routers/jev.py`'s docstring")  # fmt: skip
    check("retry-after-ms" in read("analysis/check_jev_client.py") and "<key>" in read("analysis/check_jev_client.py"),
          "item 6: `analysis/check_jev_client.py` holds the client behaviours")  # fmt: skip


def settings_commit_files(item: str) -> None:
    print("\n-- item 4: how the settings commit holds its files --")
    # A shallow clone (CI's checkout) grafts its one commit as the file's first, so the history is read only in a full one.
    shallow = git("rev-parse", "--is-shallow-repository").stdout.strip() != "false"
    commit = settings_commit(PROJECT_ROOT, SETTINGS)
    if shallow or commit is None or git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        print("      (the settings commit is not in this clone: this section is skipped)")
        return
    added = {line.split("\t")[1] for line in git("show", "--name-status", "--format=", commit).stdout.splitlines()
             if line.startswith("A\t")}  # fmt: skip
    new = {"analysis/smoke_jev.py", "analysis/jev_smoke_requests.json", "analysis/jev_smoke_record.json", SETTINGS}
    check(added == new and len(new) == 4 and "adds the other four files" in item,
          "the settings commit's own change adds the settings file, the smoke test, its requests and its record, the "
          "other four of its eight files")  # fmt: skip
    older = {"routers/jev_call.py": "41e264a", "routers/ladder.py": "e69067d", "routers/request.py": "e69067d",
             "analysis/build_jev_settings.py": "42fa2ab"}  # fmt: skip
    for file, quoted in older.items():
        last = git("log", "-1", "--abbrev=7", "--format=%h", commit, "--", file).stdout.strip()
        check(last == quoted and quoted in item and last != commit[:7],
              f"{file} was last committed by {quoted}, before the settings commit")  # fmt: skip


def budget_numbers(item: str) -> None:
    print("\n-- item 8: the byte budget's numbers --")
    from analysis.benchmark import read_rows

    settings = json.loads(read(SETTINGS))
    smoke = json.loads(read(SMOKE_RECORD))
    budgets = settings["budget_bytes"]
    g, plus = budgets[NAMED_ROUTERS[G]], budgets[NAMED_ROUTERS[G_PLUS]]
    check(budgets == smoke["budget_bytes"], "the settings file's budgets are the smoke record's")
    parts = smoke["fixed_parts_tokens"]
    state = smoke["measurements"]["g_plus_past_requests"]["body_added_bytes"]["state"]
    longest = {
        split: max(sum(jev_call.serialized_size(c) for c in row["prompt"]) for row in read_rows(split=split))
        for split in ("train", "dev")
    }
    quoted = {
        "G's budget": g, "G+'s budget": plus, "the overhead in tokens": smoke["fixed_overhead_tokens"],
        "the past requests' bytes": state, "the past requests' tokens": parts["g_plus_past_requests"],
        "the longest schema's bytes": smoke["longest_schema_bytes"], "the longest train prompt": longest["train"],
        "the longest dev prompt": longest["dev"], "G+'s margin": plus - longest["train"],
        "G's margin": g - longest["train"],
    }  # fmt: skip
    for what, value in quoted.items():
        check(f"{value:,}" in item, f"item 8 quotes {what}: {value:,}")
    wrong = {**quoted, "G+'s budget": plus + 1}
    check(f"{wrong['G+\'s budget']:,}" not in item, f"known positive: {wrong['G+\'s budget']:,} for G+'s budget is not in the record")  # fmt: skip
    check(round(state / parts["g_plus_past_requests"], 1) == 3.2 and "3.2 bytes a token" in item,
          "item 8 says the past requests run at about 3.2 bytes a token: their bytes over their tokens")  # fmt: skip
    check(max(longest.values()) < min(g, plus) and "no train or dev prompt was cut" in item,
          "no train or dev prompt is over either budget, so none was cut")  # fmt: skip
    check("Item 8's \"roughly 25,000 of room\" is stale" in item and g > 25000,
          "item 8's stale 25,000 is under G's budget now")  # fmt: skip
    plan = PLAN.read_text(encoding="utf-8")
    check(
        "roughly 25,000 of room" in amendment_item(plan, "A15", 8),
        "A15 item 8 still says it, so the record is of something real",
    )


def refusals(item: str) -> None:
    print("\n-- item 7: refusals --")
    from analysis import collect_jev

    plan = PLAN.read_text(encoding="utf-8")
    check(item.count("A15 item 19 records 0 refused on train") == 1 and "refused 0" in amendment_item(plan, "A15", 19),
          "item 7 quotes item 19's 0 refused on train")  # fmt: skip
    stored = [PROJECT_ROOT / "results" / "jev" / f"{r}_dev.jsonl" for r in collect_jev.ROUTERS]
    if not all(path.exists() for path in stored):
        print("      (results/jev holds no dev responses here: the dev count is skipped)")
        return
    records = {r: collect_jev.read_records(r, "dev") for r in collect_jev.ROUTERS}
    loaded = collect_jev.load_settings()
    check(all(len(rs) == 744 for rs in records.values()) and "dev's 744 requests" in item
          and all(not any(x["refused"] for x in rs) and collect_jev.refused_items(rs, loaded) == [] for rs in records.values()),
          "none of dev's 744 requests was refused by either router: the flag and the parser agree")  # fmt: skip


def latency_wording(item: str) -> None:
    print("\n-- item 9: how latency is checked --")
    plan = PLAN.read_text(encoding="utf-8")
    hook = " ".join(HOOK.read_text(encoding="utf-8").replace("#", " ").split())  # a comment wraps across lines
    check("isn't re-checked" in amendment_item(plan, "A15", 19) and "isn't re-checked" in item,
          "A15 item 19 says latency 'isn't re-checked' and item 9 explains it")  # fmt: skip
    check("recomputed from the stored records" in hook and "recomputed from the stored records" in item,
          "the hook's comment and item 9 both say the figures are recomputed from the stored records")  # fmt: skip
    check("dev latency, recomputed from the raw stored lines" in read("analysis/check_jev_comparisons.py")
          and "report.latency(" in read("analysis/check_jev_freeze.py"),
          "the checks do recompute them: dev's from the raw lines, train's through the report's measures")  # fmt: skip


def main() -> int:
    amendment()
    plan = PLAN.read_text(encoding="utf-8")
    items = {n: amendment_item(plan, "A16", n) for n in range(1, 10)}
    hashed_files(items[1])
    recorded_findings(items)
    settings_commit_files(items[4])
    budget_numbers(items[8])
    refusals(items[7])
    latency_wording(items[9])
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
