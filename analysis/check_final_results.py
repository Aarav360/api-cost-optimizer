"""Check the committed results of the final evaluation: its record, test's tables and test's figures (PLAN_v4.md Amendment A19,
items 8, 10 and 13; issue #77, #95).

Run from the project root:  uv run python -m analysis.check_final_results

The go ran once, and its outputs were copied from `results/final/` into the repository: the record to
`analysis/final_record.json` (as `analysis/gates_dev_record.json` is dev's), test's tables to `figures/tables_test.json` and the
six test figure files to `figures/`. This check holds those committed files, so CI runs it, where `results/` is absent. It
reads no test item, opens no store and calls no model: the files are the go's outputs, and the expected values come from other
sources than the code that wrote them (the benchmark manifest's counts, dev's committed table v4, the plan's text, and the
tables' own rows read apart from the parts that summarise them).

- **The record.** Every field of item 10 is there (`final_record_fields.field_problems`), the go ran every step and stopped at
  none, and the hook the unlock ran passed. The record's sha256 of `tables_test.json` is that of the committed tables, byte for
  byte. The skeleton of the write-up was committed before the unlock's commit, and the unlock's commit is an ancestor of this one
  (where the clone holds the history). **Where `results/` is present (the hook),** the record is also the one the go wrote, and
  `final_record_fields.verify` holds it whole: every test record file on disk against its sha256 in the record, and the tables
  against the stored files, the steps' counts and the spend.
- **The tables.** Test's split, run, item count (the manifest's test total), resamples and seed (dev's), and all 19 policies.
  The headline is the rule's cascade, and its point is recomputed from the rows' raw quality and cost (savings, the margin, the
  share of resamples meeting the target, apart from the headline part). The counts of met, missed and falling back are recounted
  from the 114 rows' readings, and the 72 that route are A19 item 6's. Each of the 51 comparisons' 306 cells is held to the two
  policies' rows: which side fell back to A, and the difference in cost and in quality equal to the rows' own. The replication
  counts are recounted from dev's table v4 and test's cells.
- **The figures.** The six files are there; each PNG is a PNG; each JSON says test, run 0 and the tables' item count; each cross
  in them is the tables' own row, point by point, under both mixes, and the references are the tables'. The explorer opens on test
  and holds both splits. A row changed in a figure, a byte changed in the tables, a field removed from the record and a PNG
  that is not one are each found (the guards below).

The live replay's numbers the write-up quotes are held in `analysis/check_writeup.py`; here the record's replay is held to itself
(each quality is a count of correct answers over its n, and the tasks' mean is the whole).
"""

import copy
import hashlib
import json
import sys
from typing import Any

from analysis import check_explorer, final_headline, writeup_results
from analysis.checking import full_history, git
from analysis.final_design import MIXES, expected_misses
from analysis.final_record_fields import field_problems, last_go, verify
from analysis.references import ROUNDING, TARGETS
from analysis.router_names import CASCADES, NAMED_ROUTERS, parse_cascade
from harness.client import PROJECT_ROOT

problems: list[str] = []
RECORD = writeup_results.RECORD
TABLES = writeup_results.TABLES
DEV = writeup_results.DEV
FIGURES = PROJECT_ROOT / "figures"
MANIFEST = PROJECT_ROOT / "data" / "benchmark_manifest.json"
RESULTS_FINAL = PROJECT_ROOT / "results" / "final"
FIGURE_FILES = {  # each test figure's PNG and JSON
    "headline": ("pareto_test.png", "pareto_test.json"),
    "tasks": ("pareto_tasks_test.png", "pareto_tasks_test.json"),
    "routed": ("pareto_routed_test.png", "pareto_routed_test.json"),
}
PNG = b"\x89PNG\r\n\x1a\n"
TARGET_KEYS = tuple(f"{t:.0%}" for t in TARGETS)
CELLS = tuple((mix, target) for mix in MIXES for target in TARGET_KEYS)
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # the standard test vector for b"abc"


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def same_bootstrap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two tables' bootstrap settings agree on resamples, seed and interval (dev's also says what it resamples)."""
    return all(a[k] == b[k] for k in ("resamples", "seed", "interval"))


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(a - b) <= tolerance


# ---- the record -------------------------------------------------------------------------------------------------------


def tables_hold_record(record: dict[str, Any], tables_bytes: bytes) -> bool:
    """Whether the record's hash and size of `tables_test.json` are those of `tables_bytes`."""
    entry = last_go(record)["steps"][-1]["detail"]["files"]["final/tables_test.json"]
    return bool(entry["sha256"] == sha256(tables_bytes) and entry["bytes"] == len(tables_bytes))


def ancestor(older: str, newer: str) -> bool:
    return git("merge-base", "--is-ancestor", older, newer).returncode == 0


def record_checks(record: dict[str, Any], tables_bytes: bytes) -> None:
    print("-- the committed record --")
    check(sha256(b"abc") == ABC_SHA256, "the hash function gives the standard SHA-256 of b'abc'")
    lacking = field_problems(record)
    check(not lacking, f"the record holds every field of A19 item 10 {lacking}")
    gos = record["gos"]
    go = last_go(record)
    check(len(gos) >= 1 and go["stopped"] is None and all(s["status"] == "done" for s in go["steps"]),
          f"the last of the {len(gos)} gos ran every step and stopped at none")  # fmt: skip
    check(go["unlock"]["hook"]["passed"] is True and go["unlock"]["hook"]["exit_code"] == 0,
          "the hook the unlock ran passed, as the record holds it")  # fmt: skip
    check(
        tables_hold_record(record, tables_bytes),
        "the record's sha256 and size of tables_test.json are the committed file's",
    )
    check(
        not tables_hold_record(record, tables_bytes + b" "),
        "known positive: the tables with one byte added are no longer the record's",
    )
    broken = copy.deepcopy(record)
    del broken["gos"][-1]["unlock"]["hook"]
    check("go 1: unlock.hook" in field_problems(broken),
          "known positive: a record without its unlock's hook result is refused, naming the field")  # fmt: skip
    commit = go["unlock"]["commit"]
    if full_history():
        from analysis.check_writeup import skeleton_commit

        skeleton = skeleton_commit()
        check(skeleton is not None and ancestor(skeleton, commit) and skeleton != commit,
              "the write-up's skeleton was committed before the unlock's commit")  # fmt: skip
        check(ancestor(commit, "HEAD"), f"the unlock's commit {commit[:9]} is an ancestor of this commit")
        check(skeleton is not None and not ancestor(commit, skeleton),
              "known positive: the unlock's commit is no ancestor of the skeleton's, so the order is not reversed")  # fmt: skip
    else:
        print("SKIP  this clone has no history: the order of the skeleton, the unlock and this commit is not checked")
    replay = writeup_results.replay_of(record)
    unlock_items = next(c for c in go["unlock"]["checks"] if c["name"] == "replay_draw")["detail"]["item_ids"]
    check(replay["items"] == len(unlock_items) == sum(replay["per_task"].values()) == 100
          and set(replay["per_task"].values()) == {25},
          "the replay's 100 items are the unlock's draw, 25 for each task")  # fmt: skip
    consistent = True
    for entry in replay["results"].values():
        live, offline = entry["live"], entry["offline"]
        for quality, n in ((live["quality"]["value"], live["quality"]["n"]),):
            consistent &= close(quality * n, round(quality * n))
        consistent &= close(sum(t["value"] for t in live["quality_by_task"].values()) / 4, live["quality"]["value"])
        consistent &= close(sum(q for q, _ in offline["quality_by_task"].values()) / 4, offline["quality"])
    check(
        consistent,
        "each replay quality is a count of correct answers over its n, and the four tasks' mean is the whole",
    )


# ---- the tables -------------------------------------------------------------------------------------------------------


def readings(tables: dict[str, Any]) -> dict[str, int]:
    """The readings of the 114 policy rows, counted from the table of every policy (`per_policy`)."""
    count = {"met": 0, "not met": 0, "falls back to A": 0}
    for body in tables["parts"]["final_table"]["per_policy"].values():
        for mix, target in CELLS:
            count[body["mixes"][mix][target]["reading"]] += 1
    return count


def section_of_label(tables: dict[str, Any]) -> dict[str, str]:
    """Each policy's name in a comparison ('G', 'haiku/Jev') and its section, from the table of every policy's labels."""
    labels = {}
    for section, body in tables["parts"]["final_table"]["per_policy"].items():
        label = body["label"]
        labels[label.split(" ")[0] if section in NAMED_ROUTERS.values() else label] = section
    return labels


def cell_problems(tables: dict[str, Any]) -> tuple[int, list[str]]:
    """Each comparison cell held to the rows of its two policies: which side fell back, and the difference in cost and in quality."""
    labels = section_of_label(tables)
    rows = tables["parts"]["final_table"]["per_policy"]
    wrong, seen = [], 0
    for group in tables["parts"]["comparisons"]["groups"].values():
        for pair, by_mix in group["pairs"].items():
            first, second = (labels[name] for name in pair.split(" less "))
            for mix, target in CELLS:
                cell, seen = by_mix[mix][target], seen + 1
                a, b = rows[first]["mixes"][mix][target], rows[second]["mixes"][mix][target]
                fell = (a["reading"] == "falls back to A", b["reading"] == "falls back to A")
                if fell != (cell["first_fell_back"], cell["second_fell_back"]):
                    wrong.append(f"{pair} {mix} {target}: fell back {fell}")
                    continue
                if all(fell):
                    if cell["cost_per_1k"] is not None or cell["reading"] != "both fall back to A":
                        wrong.append(f"{pair} {mix} {target}: both fell back and the cell holds a difference")
                    continue
                for measure in ("cost_per_1k", "quality"):
                    point = a[measure]["value"] - b[measure]["value"]
                    held = cell[measure]
                    if not close(held["value"], point, 1e-8) or held["low"] > held["high"]:
                        wrong.append(
                            f"{pair} {mix} {target}: {measure} difference {held['value']} is not the rows' {point}"
                        )
    return seen, wrong


def replication_recount(dev: dict[str, Any], tables: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    """Dev's intervals that excluded 0, whether test's has the same sign, and whether it also excludes 0, counted from table
    v4's comparisons and test's, apart from the part that lists them."""
    count = {"cost_per_1k": [0, 0, 0], "quality": [0, 0, 0]}
    test = {
        pair: by_mix for g in tables["parts"]["comparisons"]["groups"].values() for pair, by_mix in g["pairs"].items()
    }
    for g in dev["comparisons"]["groups"].values():
        for pair, by_mix in g["pairs"].items():
            for mix, target in CELLS:
                cell = by_mix[mix][target]
                for measure in count:
                    held = cell[measure]
                    if held is None or not (held["low"] > 0 or held["high"] < 0):
                        continue
                    sign = 1 if held["low"] > 0 else -1
                    there = test[pair][mix][target][measure]
                    count[measure][0] += 1
                    if there is not None and there["value"] * sign > 0:
                        count[measure][1] += 1
                        wholly = there["low"] > 0 if sign > 0 else there["high"] < 0
                        count[measure][2] += wholly
    return {k: (v[0], v[1], v[2]) for k, v in count.items()}


def table_checks(tables: dict[str, Any], dev: dict[str, Any]) -> None:
    print("\n-- test's committed tables --")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]
    n = sum(info["splits"]["test"] for info in manifest.values())
    parts = tables["parts"]
    check(tables["split"] == "test" and tables["run"] == 0 and tables["n"] == n,
          f"the tables are test's, run 0, over the manifest's {n} test items")  # fmt: skip
    check(same_bootstrap(tables["bootstrap"], dev["bootstrap"]) and tables["price_snapshot"] == dev["price_snapshot"],
          "the resamples, seed and price snapshot are dev's")  # fmt: skip
    policies = [*NAMED_ROUTERS.values(), *CASCADES]
    check(len(policies) == 19 and all(p in parts and p in parts["final_table"]["per_policy"] for p in policies),
          "all 19 policies are in the tables and in the table of every policy")  # fmt: skip
    check(all(sum(parts["stops"][c].values()) == n for c in parts["stops"]) and len(parts["stops"]) == 4,
          f"each of the four configs holds {n} graded answers")  # fmt: skip

    # the headline, recomputed from the rows
    best = final_headline.best_cascade()
    head = parts["headline"]
    point = head["headline"]
    rows = parts["final_table"]["by_mix"][point["mix"]][point["target"]]
    a_row = next(r for r in rows if r["policy"] == "A")
    best_row = next(r for r in rows if r["policy"] == "I")
    cost, a_cost = best_row["cost_per_1k"]["value"], a_row["cost_per_1k"]["value"]
    target = float(point["target"].rstrip("%")) / 100 * a_row["quality"]["value"]
    margin = best_row["quality"]["value"] - target
    met = parts["target_met"]["points"][best][point["mix"]][point["target"]]
    check(head["cascade"] == best and parts["final_table"]["best_cascade"]["section"] == best,
          f"the headline's cascade is the rule's on dev's committed table, {best}")  # fmt: skip
    check(
        (point["mix"], point["target"]) == ("realistic", "99%"),
        "the headline is on the realistic mix at the 99% target",
    )
    check(close(point["savings_vs_A"]["share"], 1 - cost / a_cost) and close(point["cost_per_1k"]["value"], cost)
          and close(point["A"]["cost_per_1k"], a_cost),
          "the headline's savings and costs are the rows': savings are one less the cost over A's")  # fmt: skip
    check(close(point["target_quality"], target) and close(point["difference"]["value"], margin)
          and close(point["quality"]["value"], best_row["quality"]["value"]),
          "the headline's margin is the point's quality less the target's share of A's quality, from the rows")  # fmt: skip
    check(point["reading"] == ("met" if margin >= -ROUNDING else "not met") == best_row["reading"]
          and close(point["met_share"], met["met_share"]) and 0 <= point["met_share"] <= 1,
          "the headline's reading follows its margin at zero, and its share of resamples is the target-met part's")  # fmt: skip

    # the counts of met and missed, recounted from the rows
    counted = readings(tables)
    ex = expected_misses()
    every = parts["target_met"]["counts"]["all"]
    check(counted["met"] == every["met"] and counted["not met"] == every["missed"]
          and counted["falls back to A"] == every["falls_back"] and sum(counted.values()) == every["points"] == 114,
          f"met, missed and falling back, recounted from the 114 rows: {counted}")  # fmt: skip
    check(counted["met"] + counted["not met"] == every["routes"] == ex.total,
          f"the points that route are {ex.total}, A19 item 6's count")  # fmt: skip

    # the comparisons
    groups = parts["comparisons"]["groups"]
    pairs = [pair for g in groups.values() for pair in g["pairs"]]
    check(len(pairs) == len(set(pairs)) == parts["comparisons"]["count"] == 51
          and sorted(pairs) == sorted(parts["declared"]["order"]),
          "the 51 comparisons are 51 distinct pairs, and the declared order holds each once")  # fmt: skip
    declared = [p for entry in parts["declared"]["declared"] for p in entry["pairs"]]
    check(
        declared == parts["declared"]["order"] and len(declared) == 51,
        "the seven questions' pairs, in order, are the declared order",
    )
    seen, wrong = cell_problems(tables)
    check(seen == 51 * 6 and not wrong, f"each of the {seen} cells is held to its two policies' rows {wrong[:3]}")
    spoiled = copy.deepcopy(tables)
    first = next(iter(spoiled["parts"]["comparisons"]["groups"]["gate_adds"]["pairs"].values()))["realistic"]["99%"]
    first["cost_per_1k"]["value"] += 0.01
    check(cell_problems(spoiled)[1] != [], "known positive: a cell with its difference moved by a cent is found")
    flipped = copy.deepcopy(tables)
    flipped["parts"]["comparisons"]["groups"]["gate_adds"]["pairs"]["E/free less E"]["realistic"]["99%"][
        "first_fell_back"
    ] = True
    check(
        cell_problems(flipped)[1] != [],
        "known positive: a cell that says a side fell back that its row does not is found",
    )

    # the replication summary
    again = replication_recount(dev, tables)
    held = parts["replication"]["counts"]
    check(all(again[m] == (held[m]["dev_excluded"], held[m]["same_sign"], held[m]["same_sign_excluding_zero"]) for m in again),
          f"the replication counts are recounted from dev's table v4 and test's cells: {again}")  # fmt: skip
    check(len(parts["replication"]["listed"]) == sum(c["dev_excluded"] for c in held.values()),
          "the replication list holds one entry for each dev interval excluding 0")  # fmt: skip


# ---- the figures ------------------------------------------------------------------------------------------------------


def row_problems(figs: dict[str, dict[str, Any]], tables: dict[str, Any]) -> list[str]:
    """Each cross of the headline and router-started figures against the tables' row of that policy, mix and target."""
    wrong = []
    for mix in MIXES:
        for section in [*NAMED_ROUTERS.values(), *CASCADES]:
            if section in CASCADES:
                start = parse_cascade(section)[0]
                where = (figs["headline"]["panels"][mix]["cascades"][section] if start == "haiku"
                         else figs["routed"]["panels"][mix]["routers"][start]["lines"][section])  # fmt: skip
            else:
                where = figs["headline"]["panels"][mix]["routers"][section]
            if where["operating_points"] != tables["parts"][section]["mixes"][mix]:
                wrong.append(f"{section} {mix}")
    return wrong


def figure_checks(tables: dict[str, Any]) -> None:
    print("\n-- test's committed figures --")
    check(all((FIGURES / name).is_file() for pair in FIGURE_FILES.values() for name in pair),
          "the headline, per-task and router-started figures are committed, each a PNG and a JSON, named for test")  # fmt: skip
    check(all((FIGURES / png).read_bytes()[:8] == PNG and (FIGURES / png).stat().st_size > 20_000 for png, _ in FIGURE_FILES.values()),
          "each PNG is a PNG of some size")  # fmt: skip
    check((FIGURES / "pareto_test.json").read_bytes()[:8] != PNG,
          "known positive: a JSON is not taken for a PNG")  # fmt: skip
    figs = {kind: json.loads((FIGURES / js).read_text(encoding="utf-8")) for kind, (_, js) in FIGURE_FILES.items()}
    check(all(f["split"] == "test" and f["run"] == 0 and f["n"] == tables["n"] and f["operating_points"] == "frozen"
              and same_bootstrap(f["bootstrap"], tables["bootstrap"]) and f["price_snapshot"] == tables["price_snapshot"] for f in figs.values()),
          "each JSON says test, run 0, the tables' item count, frozen points and the tables' resamples")  # fmt: skip
    check(all("Test split" in f["caption"] for f in figs.values()), "each caption opens on the test split")
    wrong = row_problems(figs, tables)
    check(not wrong, f"each cross is the tables' own row, for each of the 19 policies under both mixes {wrong[:4]}")
    spoiled = copy.deepcopy(figs)
    spoiled["headline"]["panels"]["realistic"]["routers"]["rules"]["operating_points"]["99%"]["quality"]["value"] += (
        1e-6
    )
    check(
        row_problems(spoiled, tables) == ["rules realistic"],
        "known positive: a cross moved by a millionth is found, named",
    )
    references = tables["parts"]["references"]
    check(all(figs["headline"]["panels"][mix][key] == references["mixes"][mix][key] for mix in MIXES
              for key in ("configs", "envelope", "oracle", "oracle_max_quality", "failed_by_every_config")),
          "the headline figure's references are the tables': configs, envelope and oracle, under both mixes")  # fmt: skip
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]
    check(all(figs["tasks"]["panels"][t]["items"] == manifest[t]["splits"]["test"]
              and figs["tasks"]["panels"][t]["configs"] == references["tasks"][t]["configs"] for t in manifest),
          "the per-task figure's items are the manifest's test counts, and its configs the tables'")  # fmt: skip
    html = (FIGURES / "pareto_explorer.html").read_text(encoding="utf-8")
    data = check_explorer.embedded(html) or {}
    check(data.get("open") == "test" and list(data.get("splits", {})) == ["test", "dev"],
          "the committed explorer opens on test and holds test and dev")  # fmt: skip


# ---- where results/ is present ----------------------------------------------------------------------------------------


def stored_checks(record: dict[str, Any], tables: dict[str, Any]) -> None:
    print("\n-- against the go's own outputs and stored files --")
    if not (RESULTS_FINAL / "final_record.json").is_file():
        print("SKIP  no results/ here (CI): the record is held whole in the hook")
        return
    results = PROJECT_ROOT / "results"
    for name, committed in (("final_record.json", RECORD), ("tables_test.json", TABLES),
                            *((js, FIGURES / js) for _, js in FIGURE_FILES.values()),
                            *((png, FIGURES / png) for png, _ in FIGURE_FILES.values())):  # fmt: skip
        check(
            (RESULTS_FINAL / name).read_bytes() == committed.read_bytes(),
            f"the committed {name} is the go's output, byte for byte",
        )
    found = verify(record, tables, results, RESULTS_FINAL)
    check(found == [], f"the record holds whole against the tables and every stored test record file {found[:3]}")
    spoiled = copy.deepcopy(tables)
    spoiled["n"] += 1
    check(verify(record, spoiled, results, RESULTS_FINAL) != [],
          "known positive: tables that count one item more are refused against the record")  # fmt: skip


def main() -> int:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    tables_bytes = TABLES.read_bytes()
    tables = json.loads(tables_bytes)
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    record_checks(record, tables_bytes)
    table_checks(tables, dev)
    figure_checks(tables)
    stored_checks(record, tables)
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
