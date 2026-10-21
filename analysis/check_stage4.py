"""Check E on dev and Stage 4's outputs (PLAN_v4.md §8, §12; Amendment A9, items 3, 13-15 and 18).

Run from the project root:  uv run python -m analysis.check_stage4

Needs the dev results matrix in results/ (not committed), so the hook runs it where the matrix is present, like
check_simulator. It reads train and dev only, through the reader, and never a test item.

- E's dev router-score table is `score(request)` over every dev request from the reader, with $0 overhead, and it
  passes the table's refusal; the refusal itself is shown on a table that leaves out an item and on one that scores
  the frontier;
- the dev task-inference diagnostic (item 5) is a recount of the inferred type against each dev item's task, in the
  words item 13 asks for ("on this benchmark's templates"), and it can fail: with the tasks shifted it changes;
- the walk on dev, for both mixes and all three targets, reproduces E's section of the operating-points file to the
  last bit (and so its hash); in that file every certified step clears its target by its margin, the walk stops at
  the first step that doesn't, and the operating point is the cheapest certified step by the file's own costs;
- table v0's keys are kept, and chart and table v1 are current: E's steps, operating points and McNemar rows in both
  figure JSONs equal a rebuild, and what they hold is consistent: E's line starts at A and runs the table's steps from
  the highest threshold down, each cross sits on the step the file picked, savings, the routing mix, the envelope's
  reading at E's cost and every McNemar count (recounted from the matrix's cells) are right, and every captured
  fraction is at most 1 and 0 for a fallback;
- Amendment A9 item 18's record: every number in it is the one the JSONs give at the precision it prints, and the
  hash is E's section's, with guards that a one-digit slip and an uncompared number both fail.
"""

import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import astuple
from fractions import Fraction
from itertools import pairwise, zip_longest
from math import comb
from typing import Any

from analysis import matrix as results_matrix
from analysis.benchmark import read_rows
from analysis.bootstrap import resample
from analysis.dev_diagnostic import dev_diagnostic, dev_inputs
from analysis.fit_rules_router import diagnostic
from analysis.matrix import Matrix
from analysis.operating_points import OPERATING_POINTS_FILE, read_section, section_hash
from analysis.pareto import FIGURE_DATA, PANELS, TASK_FIGURE_DATA
from analysis.plan import PLAN, amendment_item
from analysis.references import ROUNDING, TARGETS
from analysis.router_panels import config_mcnemar, mix_router, step_outcomes, task_router
from analysis.router_table import ItemScores, RouterScoreTable, TableError
from analysis.rules_table import dev_table
from analysis.rules_walk import ROUTER, dev_walks, rebuilt_section
from analysis.simulator import always, load_mixes, simulate, totals
from analysis.walk import step_thresholds
from routers import rules
from routers.decision import choose
from routers.ladder import FRONTIER, LADDER, SCORED
from routers.request import Request, from_row

DEV_TASKS = ("ifeval", "banking77", "mmlu_pro", "mbppplus")  # A5's order, as the per-task figure has them

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object]) -> bool:
    try:
        call()
    except TableError:
        return True
    return False


# Table v0's keys, as chart v0 (Stage 3, Amendment A8 item 10) wrote them. Table v1 only adds to them (A9 item 15).
V0_TOP = ["figure", "built_by", "split", "run", "n", "price_snapshot", "bootstrap", "units", "panels"]
V0_MIX_PANEL = ["configs", "envelope", "oracle", "oracle_max_quality", "failed_by_every_config", "targets"]
V0_TASK_PANEL = ["items", "warning", "configs", "envelope", "oracle", "oracle_max_quality"]
V0_CONFIG_ROW = ["quality", "cost_per_1k", "savings_vs_A", "difference_from_A"]
V0_TASK_CONFIG_ROW = ["quality", "cost_per_1k"]
V0_TARGET = ["quality", "oracle", "envelope"]
V0_HEADLINE_UNITS = ["quality", "cost_per_1k", "configs", "envelope", "oracle", "oracle_max_quality", "savings_vs_A",
                     "difference_from_A", "failed_by_every_config", "targets"]  # fmt: skip
V0_TASK_UNITS = ["quality", "cost_per_1k", "configs", "envelope", "oracle", "oracle_max_quality", "items", "warning"]
V1_ADDED_UNITS = ["routers", "steps", "operating_points", "threshold", "fell_back", "steps_certified", "steps_tried",
                  "target", "margin", "captured", "routing", "matched_cost", "mcnemar", "mcnemar_vs_A", "label",
                  "operating_points_file", "operating_points_sha256", "task_inference_dev"]  # fmt: skip


def mcnemar_p(b: int, c: int) -> float:
    """The exact two-sided p, from the binomial with exact fractions, as A9 item 11 defines it."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(Fraction(1), Fraction(2 * sum(comb(n, k) for k in range(min(b, c) + 1)), 2**n)))


def read_envelope(corners: dict[str, dict]) -> Callable[[float], float | None]:
    """The envelope's quality at a cost, read from the figure JSON's corners as A8 item 7 says: along the straight
    lines in linear cost, flat at the top corner's quality past its cost, n/a (None) below the first corner."""
    points = [(corner["cost_per_1k"], corner["quality"]) for corner in corners.values()]

    def at(cost: float) -> float | None:
        if cost < points[0][0]:
            return None
        if cost >= points[-1][0]:
            return points[-1][1]
        for (c0, q0), (c1, q1) in pairwise(points):
            if c0 <= cost <= c1:
                return q0 + (cost - c0) * (q1 - q0) / (c1 - c0)
        raise AssertionError("unreachable")

    return at


def v0_keys_kept(headline: dict, per_task: dict) -> None:
    check(set(headline) == {*V0_TOP, "routers"} and list(per_task) == V0_TOP,
          "both figure JSONs keep every top-level key of table v0; the headline's adds only `routers`, E's name, "
          "diagnostic and hash")  # fmt: skip
    check(all(list(p)[: len(V0_MIX_PANEL)] == V0_MIX_PANEL and set(p) == {*V0_MIX_PANEL, "routers"}
              for p in headline["panels"].values()),
          "each headline panel keeps v0's keys and adds only `routers`")  # fmt: skip
    check(all(list(p)[: len(V0_TASK_PANEL)] == V0_TASK_PANEL and set(p) == {*V0_TASK_PANEL, "mcnemar_vs_A", "routers"}
              for p in per_task["panels"].values()),
          "each per-task panel keeps v0's keys and adds only `mcnemar_vs_A` and `routers`")  # fmt: skip
    check(all(list(row) == V0_CONFIG_ROW for p in headline["panels"].values() for row in p["configs"].values())
          and all(list(row) == V0_TASK_CONFIG_ROW for p in per_task["panels"].values() for row in p["configs"].values())
          and all(list(t) == V0_TARGET for p in headline["panels"].values() for t in p["targets"].values()),
          "the config rows and the targets' rows are exactly v0's: new keys sit beside them, not in them")  # fmt: skip
    check(set(V0_HEADLINE_UNITS) <= set(headline["units"]) and set(V0_TASK_UNITS) <= set(per_task["units"]),
          "both figure JSONs keep every units entry of table v0")  # fmt: skip
    for data, v0_units in ((headline, V0_HEADLINE_UNITS), (per_task, V0_TASK_UNITS)):
        added = set(data["units"]) - set(v0_units)
        check(added == set(V1_ADDED_UNITS) - ({"mcnemar", "mcnemar_vs_A"} if data is headline else set()),
              f"{data['figure']}: table v1's new keys each have a units entry, and it adds nothing else: "
              f"{len(added)} entries")  # fmt: skip


def chart_checks(dev: Matrix, table: RouterScoreTable, section: dict[str, Any]) -> tuple[dict, dict]:
    """Chart and table v1 against the package and the file: they are current, and what they hold is consistent.
    Returns the two committed JSONs."""
    headline, per_task = (json.loads(path.read_text()) for path in (FIGURE_DATA, TASK_FIGURE_DATA))
    print("\n-- table v0's keys are kept (Amendment A9, item 15) --")
    v0_keys_kept(headline, per_task)

    print("\n-- chart and table v1 are current: rebuilt from the package and the walk --")
    mixes = load_mixes()
    fixed = {config: simulate(dev, always(dev, config)) for config in LADDER}
    draws = resample(dev)
    walks = dev_walks(dev, table, draws)
    steps = step_outcomes(dev, table, fixed[FRONTIER])
    for mix in PANELS:
        savings = draws.savings(dev, fixed[FRONTIER], mixes[mix])
        rebuilt = mix_router(draws, savings, steps, walks[mix], fixed, mixes[mix])
        check(json.loads(json.dumps(rebuilt)) == headline["panels"][mix]["routers"][ROUTER],
              f"{mix} mix: E's steps and operating points in figures/pareto.json are what the package gives now")  # fmt: skip
    for task in DEV_TASKS:
        rebuilt = {
            "mcnemar_vs_A": config_mcnemar(fixed, task),
            "routers": {ROUTER: task_router(draws, steps, walks, task)},
        }
        committed = {key: per_task["panels"][task][key] for key in rebuilt}
        check(json.loads(json.dumps(rebuilt)) == committed,
              f"{task}: E's steps, operating points and McNemar rows in figures/pareto_tasks.json are current")  # fmt: skip

    print("\n-- E's steps and operating points, read from the JSON (Amendment A9, items 9, 10, 14 and 15) --")
    thresholds = [None, *reversed(step_thresholds(table))]
    a_totals = {mix: totals(fixed[FRONTIER], mixes[mix]) for mix in PANELS}
    for mix in PANELS:
        router = headline["panels"][mix]["routers"][ROUTER]
        line, points = router["steps"], router["operating_points"]
        check([step["threshold"] for step in line] == thresholds,
              f"{mix} mix: E's line runs A, then the table's steps from the highest threshold down: {len(line)} points")  # fmt: skip
        check((line[0]["quality"], line[0]["cost_per_1k"]) == astuple(a_totals[mix]),
              f"{mix} mix: the line starts at A, as E has no overhead")  # fmt: skip
        check(set(points) == set(section[mix]), f"{mix} mix: an operating point for each of the file's targets")
        for key, point in points.items():
            filed = section[mix][key]
            where = f"{mix} mix, {key} of A"
            check((point["threshold"], point["fell_back"], point["target"], point["margin"])
                  == (filed["threshold"], filed["fell_back"], filed["target"], filed["margin"]),
                  f"{where}: threshold, fallback, target and margin are the operating-points file's")  # fmt: skip
            on_line = [s for s in line if s["threshold"] == point["threshold"]]
            check(len(on_line) == 1 and (point["quality"]["value"], point["cost_per_1k"]["value"])
                  == (on_line[0]["quality"], on_line[0]["cost_per_1k"]),
                  f"{where}: the cross sits on the step the file picked")  # fmt: skip
            stop = next((i for i, s in enumerate(filed["steps"]) if not s["clears"]), len(filed["steps"]))
            check(point["steps_certified"] == stop - 1 and point["steps_tried"] == len(filed["steps"]) - 1,
                  f"{where}: {point['steps_certified']} of {point['steps_tried']} steps tried certified, A not counted")  # fmt: skip
            saved = point["savings_vs_A"]
            check(abs(saved["per_1k"] - (a_totals[mix].cost_per_1k - point["cost_per_1k"]["value"])) < ROUNDING
                  and abs(saved["share"] - saved["per_1k"] / a_totals[mix].cost_per_1k) < ROUNDING,
                  f"{where}: savings against A are A's cost less E's: ${saved['per_1k']:.2f} per 1k ({saved['share']:.1%})")  # fmt: skip
            check(abs(sum(point["routing"].values()) - 1) < ROUNDING
                  and (abs(point["routing"][FRONTIER] - 1) < ROUNDING if point["fell_back"] else True),
                  f"{where}: the routing mix sums to 1" + (", all on opus: it falls back to A" if point["fell_back"] else ""))  # fmt: skip
            captured = point["captured"]
            bounds = [captured[b] for b in ("value", "low", "high") if captured[b] is not None]
            check(all(0 <= b <= 1 for b in bounds) and 0 <= captured["met"] <= 1
                  and (captured["value"] == 0 if point["fell_back"] else captured["value"] is not None and captured["value"] > 0),
                  f"{where}: the captured fraction, interval included, is at most 1"
                  + (" and 0 for a fallback" if point["fell_back"] else f" ({captured['value']:.2f})"))  # fmt: skip
            # The envelope's quality at E's cost, read from the figure's own corners by hand.
            reading = read_envelope(headline["panels"][mix]["envelope"])(point["cost_per_1k"]["value"])
            matched = point["matched_cost"]
            check(reading is not None and abs(matched["envelope"] - reading) < ROUNDING
                  and abs(matched["difference"] - (point["quality"]["value"] - reading)) < ROUNDING,
                  f"{where}: the envelope's quality at E's cost is read along its corners, and E's difference from it "
                  f"is E less that ({shown_pts(matched['difference'])} points)")  # fmt: skip

    print("\n-- per task: McNemar's test, counted from the matrix's own cells (Amendment A9, items 11 and 15) --")
    scores = {entry.item_id: entry.scores for entry in table.items}

    def right(item_id: str, config: str) -> bool:
        return dev.cells[(item_id, config)].success == 1

    for task in DEV_TASKS:
        ids = [i for i, item in dev.items.items() if item.task == task]
        panel = per_task["panels"][task]

        def counted(chosen: dict[str, str]) -> tuple[int, int]:
            b = sum(right(i, chosen[i]) and not right(i, FRONTIER) for i in ids)
            c = sum(right(i, FRONTIER) and not right(i, chosen[i]) for i in ids)
            return b, c

        rows = {config: counted(dict.fromkeys(ids, config)) for config in LADDER if config != FRONTIER}
        check(panel["mcnemar_vs_A"][FRONTIER] is None
              and all(panel["mcnemar_vs_A"][c] == {"b": b, "c": cc, "p": mcnemar_p(b, cc)} for c, (b, cc) in rows.items()),
              f"{task}: the fixed configs' b, c and p against A are the counts {rows}, A's own row null")  # fmt: skip
        shown_rows = []
        for mix, by_target in panel["routers"][ROUTER]["operating_points"].items():
            for key, point in by_target.items():
                threshold = point["threshold"]
                if point["fell_back"]:
                    check(threshold is None and point["mcnemar"] is None,
                          f"{task}, {mix} {key}: a fallback has no McNemar row, shown as —")  # fmt: skip
                    continue
                b, c = counted({i: choose(scores[i], threshold) for i in ids})
                shown_rows.append((b, c))
                check(point["mcnemar"] == {"b": b, "c": c, "p": mcnemar_p(b, c)},
                      f"{task}, {mix} {key}: E at {threshold:.4f} against A: b = {b}, c = {c}, p = {mcnemar_p(b, c):.3f}")  # fmt: skip
        check(len(shown_rows) == sum(not p["fell_back"] for by in panel["routers"][ROUTER]["operating_points"].values()
                                     for p in by.values()), f"{task}: every operating point that doesn't fall back has a row")  # fmt: skip
    return headline, per_task


def shown_pts(value: float) -> str:
    return f"{100 * value:+.1f}"


def record_lines(headline: dict, section: dict[str, Any]) -> list[str]:
    """Amendment A9 item 18's lines that hold a number, as chart v1's JSON and the operating-points file give them,
    at the precision the record prints. None of the record's other lines holds one."""
    router = headline["routers"][ROUTER]
    mixes = [headline["panels"][mix]["routers"][ROUTER]["operating_points"] for mix in ("realistic", "uniform")]

    def row(label: str, cell: Callable[[dict], str]) -> str:
        return f"| {label} | {' | '.join(', '.join(cell(p) for p in points.values()) for points in mixes)} |"

    def threshold(point: dict) -> str:
        return "falls back to A" if point["fell_back"] else f"{point['threshold']:.4f}"

    targets = ", ".join(mixes[0])
    bootstrap = headline["bootstrap"]
    return [
        f"18. **Record.** E's operating points on {headline['split']} (run {headline['run']}, n = {headline['n']}, "
        f"list prices of {headline['price_snapshot']}), from `figures/pareto.json`, and the hash of its section of "
        f"`routers/operating_points.json`. Intervals are percentile 95% from {bootstrap['resamples']:,} paired "
        f"resamples, and `analysis/check_stage4.py` compares every number here with those two files.",
        row(f"E: threshold at {targets} of A's quality", threshold),
        row("E: quality there", lambda p: f"{100 * p['quality']['value']:.1f}%"),
        row("E: $ per 1k there", lambda p: f"${p['cost_per_1k']['value']:.2f}"),
        row("E: savings against A there", lambda p: f"{p['savings_vs_A']['share']:.1%}"),
        row("E: captured fraction there", lambda p: f"{p['captured']['value']:.2f}"),
        row("E: share of resamples meeting the target there", lambda p: f"{p['captured']['met']:.1%}"),
        '- Where a target reads "falls back to A", the walk certified nothing cheaper than A (item 10): the operating '
        "point is A itself, with no router call and no overhead, and its captured fraction is 0.00.",
        f"- {router['task_inference_dev']}. It says nothing about traffic outside these templates.",
        f"- E's section of `routers/operating_points.json`, written with sorted keys, has the 64-digit sha256 "
        f"`{section_hash(section)}`.",
    ]


def record_mismatches(record: list[str], headline: dict, section: dict[str, Any]) -> list[tuple[str, str]]:
    """Each line of the record holding a number that differs from the line the JSON gives, paired with it; a line
    missing on either side is paired with ''. A digit after a letter is part of a name, as in v1 or A9."""
    held = [line.strip() for line in record if re.search(r"(?<![A-Za-z0-9_.])\d", line)]
    return [(says, gives) for says, gives in zip_longest(held, record_lines(headline, section), fillvalue="")
            if says != gives]  # fmt: skip


def record_checks(headline: dict, section: dict[str, Any]) -> None:
    print("\n-- E's record (Amendment A9, item 18) --")
    check(headline["routers"][ROUTER]["operating_points_sha256"] == section_hash(section)
          == hashlib.sha256(json.dumps(section, sort_keys=True, indent=2).encode()).hexdigest(),
          "the figure JSON's hash of E's section is the sha256 of the section read from the file, sorted keys")  # fmt: skip
    check(json.loads(OPERATING_POINTS_FILE.read_text())[ROUTER] == section,
          "E's section is the one in the operating-points file, which holds it with sorted keys")  # fmt: skip
    record = amendment_item(PLAN.read_text(), "A9", 18).splitlines()
    mismatches = record_mismatches(record, headline, section)
    check(not mismatches,
          "every number in Amendment A9 item 18's record is the one figures/pareto.json and the operating-points "
          "file give, at the precision the record prints, and the hash is the section's"
          + "".join(f"\n      record: {says!r}\n      JSON:   {gives!r}" for says, gives in mismatches))  # fmt: skip
    # A guard proves it can fail: a slip in one digit, and a number the check doesn't know.
    quality = next(i for i, line in enumerate(record) if line.strip().startswith("| E: quality there"))
    slip = re.sub(r"(\d+\.)(\d)%", lambda m: f"{m[1]}{(int(m[2]) + 1) % 10}%", record[quality], count=1)
    check(len(record_mismatches([*record[:quality], slip, *record[quality + 1 :]], headline, section)) == 1,
          f"the record check catches a slip in one number: {slip.strip()!r}")  # fmt: skip
    check(len(record_mismatches([*record, "    - One number more: 1.0 points."], headline, section)) == 1,
          "the record check catches a number it doesn't compare")  # fmt: skip
    check("On this benchmark's templates" in "".join(record), "the record words the diagnostic as holding on this benchmark's templates")  # fmt: skip


def main() -> int:
    print("-- E's dev router-score table (Amendment A9, item 3) --")
    rows = read_rows(split="dev")
    dev = results_matrix.load().split("dev")
    dev_ids = list(dev.items)
    table = dev_table()
    check(len(table.items) == len(rows) == 744, f"the table has an entry for each of the {len(rows)} dev requests")
    check([entry.item_id for entry in table.items] == [row["item_id"] for row in rows],
          "its entries are in the reader's dev order")  # fmt: skip
    check(set(entry.item_id for entry in table.items) == set(dev_ids),
          "they are the dev results matrix's items")  # fmt: skip
    check(all(entry.scores == rules.score(from_row(row)) for entry, row in zip(table.items, rows)),
          "every entry's router scores are score(request) for that dev request, from the reader")  # fmt: skip
    check(all(set(entry.scores) == set(SCORED) for entry in table.items),
          f"every entry scores {', '.join(SCORED)} and never {FRONTIER}")  # fmt: skip
    check(all(entry.overhead == 0.0 for entry in table.items), "every overhead is exactly $0: E runs locally")
    check(RouterScoreTable.build(list(table.items), dev_ids).items == table.items,
          "the table passes the table's refusal against the matrix's dev items")  # fmt: skip
    check(refused(lambda: RouterScoreTable.build(list(table.items)[1:], dev_ids)),
          "the refusal fires on a table that leaves out a dev item")  # fmt: skip
    first = table.items[0]
    scored_frontier = ItemScores(first.item_id, {**first.scores, FRONTIER: 0.5}, 0.0)
    check(refused(lambda: RouterScoreTable.build([scored_frontier, *table.items[1:]], dev_ids)),
          "the refusal fires on a table that scores the frontier")  # fmt: skip

    print("\n-- the dev task-inference diagnostic (Amendment A9, items 5 and 13) --")
    # Each task's own type, written here from the plan's item 5 and not taken from the script under check.
    own_type = {"banking77": rules.CLASSIFICATION, "mmlu_pro": rules.MULTIPLE_CHOICE, "mbppplus": rules.CODE,
                "ifeval": rules.OPEN_ENDED}  # fmt: skip
    requests, tasks = dev_inputs()
    full = sum(rules.infer_type(request) == own_type[tasks[i]] for i, request in requests.items())
    bare = sum(rules.infer_type(Request(request.prompt)) == own_type[tasks[i]] for i, request in requests.items())
    said = dev_diagnostic()
    print(f"      {said}")
    check(len(requests) == len(rows) and set(requests) == set(dev_ids), "it covers exactly the dev items")
    check(said.startswith("On this benchmark's templates the inferred type is the item's task for ")
          and f"for {full} of {len(requests)} dev items" in said and f"and for {bare} of {len(requests)}" in said
          and said.endswith("with the output schema withheld"),
          "it states the counts a recount gives, with the schema and with it withheld, on this benchmark's templates")  # fmt: skip
    ordered = sorted(tasks)
    shuffled = dict(zip(ordered, [tasks[i] for i in ordered[1:] + ordered[:1]]))
    check(f"for {full} of" not in diagnostic(requests, shuffled, split="dev"),
          "the diagnostic can fail: with the tasks shifted by one item the count changes")  # fmt: skip

    print("\n-- the walk on dev and the operating-points file (Amendment A9, items 9, 10 and 13) --")
    section = read_section(ROUTER)
    check(list(section) == ["realistic", "uniform"] and all(set(by_target) == {"100%", "99%", "97%"}
                                                           for by_target in section.values()),
          "E's section holds both mixes and all three targets: six operating points")  # fmt: skip
    rebuilt = rebuilt_section()
    check(rebuilt == section and section_hash(rebuilt) == section_hash(section),
          "the walk on dev reproduces E's section of the operating-points file, so its hash too")  # fmt: skip
    for mix, by_target in section.items():
        for key, point in by_target.items():
            steps: list[dict[str, Any]] = point["steps"]
            target = point["target"]
            where = f"{mix} mix, {key} of A"
            stop = next((i for i, step in enumerate(steps) if not step["clears"]), len(steps))
            certified = steps[:stop]
            check(steps[0]["threshold"] is None and steps[0]["margin"] == 0 and float(key.rstrip("%")) / 100 in TARGETS,
                  f"{where}: the walk starts at A, threshold null, margin 0")  # fmt: skip
            check(all(step["quality"] - target >= step["margin"] - ROUNDING for step in certified),
                  f"{where}: all {len(certified)} certified steps (A included) clear the target by their margin")  # fmt: skip
            # A step that clears is certified only if every step before it did; the walk ends at the first miss.
            missed = steps[stop:]
            check(not missed or (len(missed) == 1 and missed[0]["quality"] - target < missed[0]["margin"] - ROUNDING),
                  f"{where}: the walk ends at the first step that misses its target, or tried them all")  # fmt: skip
            thresholds = [step["threshold"] for step in steps[1:]]
            check(thresholds == sorted(thresholds, reverse=True),
                  f"{where}: the steps run from the highest threshold down")  # fmt: skip
            cheapest = certified[0]
            for step in certified[1:]:
                # Costs within rounding of each other are a tie, which goes to the earlier step.
                if step["cost_per_1k"] < cheapest["cost_per_1k"] - ROUNDING:
                    cheapest = step
            check(point["threshold"] == cheapest["threshold"] and point["margin"] == cheapest["margin"]
                  and point["fell_back"] == (point["threshold"] is None),
                  f"{where}: the operating point is the cheapest certified step: "
                  + ("A, it falls back" if point["fell_back"] else f"threshold {point['threshold']:.3f}"))  # fmt: skip

    headline, _ = chart_checks(dev, table, section)
    record_checks(headline, section)

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
