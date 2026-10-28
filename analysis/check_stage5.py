"""Check Stage 5's chart and table v2, the learning-curve figure and F's record (PLAN_v4.md §12; Amendment A11, items 13,
16 and 21).

Run from the project root:  uv run python -m analysis.check_stage5
Print the record's numeric lines as the JSONs give them:  uv run python -m analysis.check_stage5 --print

Reads the committed figure JSONs, `routers/operating_points.json` and PLAN_v4.md, so CI runs it; the sections that
rebuild from the dev matrix run only where `results/` holds it, as `check_simulator` does. Never a test item.

- table v2 keeps table v1: both figure JSONs hold E's part and F's (then, since Stage 6, G's and G+'s), as the same keys,
  and nothing of a diagnostic; the
  operating-points file holds E's section and F's, and no other, with the figure JSON's hash of F's;
- F's rows, read from the JSON by hand as `check_stage4` reads E's: each cross sits on the step the file picked, the
  certified steps, the savings, the routing mix, the envelope's quality at F's cost along its corners, every captured
  fraction, and every per-task McNemar count recounted from the matrix's cells (dev only);
- table v2's file: the same-target comparisons are the differences of the operating points in the figure JSON, and
  "both fall back to A" where both fell back; the diagnostics are labeled "diagnostic, dev only", F-train's and the
  task rates' operating points are the walks' and neither is in the operating-points file; the learning curve's
  headline is the 100% point less the 75% point, and `figures/learning_curve.json` holds the same numbers;
- the learning-curve figure drawn from the committed numbers: F's line joins its four points in F's colour, F-train is
  one separate marker in another, the dashed line is the task-rates diagnostic's log loss and 0.5 marks the concordance
  panels;
- Amendment A11 item 21's record: every number in it is the one the committed files give, at the precision it prints,
  through the plan's amendment-item reader, with guards that a one-digit slip and an uncompared number both fail, and
  the comparisons are called descriptive with no multiplicity correction;
- on dev (hook only): F's parts and table v2 equal a rebuild, and the rebuild writes nothing to the operating-points
  file.
"""

import hashlib
import json
import re
import sys
import tempfile
from collections.abc import Callable
from fractions import Fraction
from itertools import pairwise, zip_longest
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from analysis.chart_routers import F_STYLE
from analysis.learning_figure import F_TRAIN_COLOR, LEARNING_FIGURE_DATA, draw_learning_curve
from analysis.operating_points import (
    OPERATING_POINTS_FILE,
    OperatingPointsError,
    check_sections,
    read_section,
    section_hash,
)
from analysis.pareto import FIGURE_DATA, PANELS, TASK_FIGURE_DATA
from analysis.plan import PLAN, amendment_item
from analysis.references import ROUNDING
from analysis.router_names import F_TRAIN, G_PLUS, LABELS, NAMED_ROUTERS, TASK_RATES, E, F, G
from analysis.rules_walk import MIXES
from analysis.table_v2 import DESCRIPTIVE, DIAGNOSTIC_LABEL, TABLE_V2
from harness.client import PROJECT_ROOT
from routers.ladder import FRONTIER, LADDER, SCORED

E_SECTION, F_SECTION = (
    NAMED_ROUTERS[E],
    NAMED_ROUTERS[F],
)  # the routers' names in the operating-points file and the JSONs
# Stage 6 added G's and G+'s sections to the same file (A15 item 14); E's and F's are checked below as they were.
JEV_SECTIONS = (NAMED_ROUTERS[G], NAMED_ROUTERS[G_PLUS])
TARGETS = ("100%", "99%", "97%")
DEV_TASKS = ("ifeval", "banking77", "mmlu_pro", "mbppplus")
PAIRS = ((F, E), (F_TRAIN, E), (F, F_TRAIN))  # item 13's comparisons, first less second
DIAGNOSTICS = (F_TRAIN, TASK_RATES)  # the diagnostics that are walked like routers; never in the file
problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object]) -> bool:
    try:
        call()
    except OperatingPointsError:
        return True
    return False


# ---------------------------------------------------------------------------------------------------------------
# Small readers, written here from the plan's rules (A8 item 7, A9 item 11) and not taken from the code under check


def envelope_at(corners: dict[str, dict], cost: float) -> float | None:
    """The envelope's quality at a cost along its corners: straight lines in linear cost, flat past the top corner,
    n/a (None) below the first."""
    points = [(corner["cost_per_1k"], corner["quality"]) for corner in corners.values()]
    if cost < points[0][0]:
        return None
    if cost >= points[-1][0]:
        return points[-1][1]
    for (c0, q0), (c1, q1) in pairwise(points):
        if c0 <= cost <= c1:
            return q0 + (cost - c0) * (q1 - q0) / (c1 - c0)
    raise AssertionError("unreachable")


def mcnemar_p(b: int, c: int) -> float:
    """The exact two-sided p, from the binomial with exact fractions."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(Fraction(1), Fraction(2 * sum(comb(n, k) for k in range(min(b, c) + 1)), 2**n)))


# ---------------------------------------------------------------------------------------------------------------
# Item 21's record from the committed files


def points_text(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.1f}"


def interval(estimate: dict, scale: float, places: int, signed: bool) -> str:
    sign = "+" if signed else ""
    value, low, high = (f"{scale * estimate[k]:{sign}.{places}f}" for k in ("value", "low", "high"))
    return f"{value} ({low} to {high})"


def threshold_text(point: dict) -> str:
    return "falls back to A" if point["fell_back"] else f"{point['threshold']:.4f}"


def comparison_text(cell: dict, which: str) -> str:
    """One same-target cell's cost in dollars per 1k or quality in points, or what stands in for the difference."""
    if cell["reading"]:
        return cell["reading"]
    if which == "cost":
        return interval(cell["cost_per_1k"], 1, 2, True)
    return interval(cell["quality"], 100, 1, True)


def record_lines(headline: dict, table: dict, section: dict[str, Any]) -> list[str]:
    """Amendment A11 item 21's lines that hold a number, as the figure JSONs and the operating-points file give them,
    at the precision the record prints."""
    mixes = [headline["panels"][mix]["routers"][F_SECTION]["operating_points"] for mix in MIXES]
    bootstrap = headline["bootstrap"]
    diag = table["diagnostics"]

    def row(label: str, cell: Callable[[dict], str]) -> str:
        return f"| {label} | {' | '.join(', '.join(cell(p) for p in points.values()) for points in mixes)} |"

    def matched(point: dict) -> str:
        gap = point["matched_cost"]
        return f"{points_text(gap['difference'])} ({points_text(gap['low'])} to {points_text(gap['high'])})"

    targets = ", ".join(mixes[0])
    lines = [
        f"21. **Record.** F's operating points on {headline['split']} (run {headline['run']}, n = {headline['n']}, "
        f"list prices of {headline['price_snapshot']}), from `figures/pareto.json`, and the hash of its section of "
        f"`routers/operating_points.json`, as A9 item 18 records E's; the diagnostics' and comparisons' numbers are "
        f"from `figures/table_v2.json`. Intervals are percentile 95% from {bootstrap['resamples']:,} paired "
        f"resamples, and `analysis/check_stage5.py` compares every number here with those files.",
        row(f"F: threshold at {targets} of A's quality", threshold_text),
        row("F: quality there", lambda p: f"{100 * p['quality']['value']:.1f}%"),
        row("F: $ per 1k there", lambda p: f"${p['cost_per_1k']['value']:.2f}"),
        row("F: savings against A there", lambda p: f"{p['savings_vs_A']['share']:.1%}"),
        row("F: captured fraction there", lambda p: f"{p['captured']['value']:.2f}"),
        row("F: share of resamples meeting the target there", lambda p: f"{p['captured']['met']:.1%}"),
        row("F: paired quality difference from the envelope at F's cost there, in points", matched),
        '- Where a target reads "falls back to A", the walk certified nothing cheaper than A (item 10): the operating '
        "point is A itself, with no router call and no overhead, and its captured fraction is 0.00.",
        f"- F's section of `routers/operating_points.json`, written with sorted keys, has the 64-digit sha256 "
        f"`{section_hash(section)}`.",
        f"- The comparisons below are {DESCRIPTIVE}.",
    ]
    for pair in PAIRS:
        by_mix = table["comparisons"]["same_target"][f"{pair[0]} less {pair[1]}"]
        for which, unit in (("cost", "$ per 1k"), ("quality", "quality in points")):
            lines.append(f"| {pair[0]} less {pair[1]} at the same target: {unit} | " + " | ".join(
                ", ".join(comparison_text(cell, which) for cell in by_mix[mix].values()) for mix in MIXES) + " |")  # fmt: skip
    lines.append(f"- Within-task concordance, pooled over pairs within a task and fold ({DIAGNOSTIC_LABEL}), with its "
                 f"95% paired interval:")  # fmt: skip
    for name, label in ((F, F), (F_TRAIN, F_TRAIN), (TASK_RATES, "The task-rates diagnostic")):
        lines.append(f"| {label}, concordance | " + " | ".join(
            interval(diag["within_task_concordance"][name][c]["pooled"], 1, 3, False) for c in SCORED) + " |")  # fmt: skip
    for gap, label in (("F less task rates", "F less the task-rates diagnostic"), ("F less F-train", "F less F-train")):
        lines.append(f"| {label}, concordance | " + " | ".join(
            interval(diag["concordance_differences"][gap][c]["pooled"], 1, 3, True) for c in SCORED) + " |")  # fmt: skip
    steps = diag["learning_curve"]["headline"]
    lines.append(f"- The learning curve's paired difference from 75% to 100% of the fold models' training rows "
                 f"({DIAGNOSTIC_LABEL}), log loss then concordance:")  # fmt: skip
    for config in SCORED:
        lines.append(f"| {config}, 100% less 75% | {interval(steps[config]['log_loss']['pooled'], 1, 4, True)} | "
                     f"{interval(steps[config]['concordance']['pooled'], 1, 4, True)} |")  # fmt: skip
    return lines


def numeric(lines: list[str]) -> list[str]:
    """The lines holding a number, stripped. A digit after a letter is part of a name, as in v1 or A9."""
    return [line.strip() for line in lines if re.search(r"(?<![A-Za-z0-9_.])\d", line)]


def record_mismatches(record: list[str], headline: dict, table: dict, section: dict[str, Any]) -> list[tuple[str, str]]:
    """Each numeric line of the record that differs from the line the JSON gives, paired with it; a line missing on
    either side is paired with ''."""
    gives = numeric(record_lines(headline, table, section))
    return [(says, given) for says, given in zip_longest(numeric(record), gives, fillvalue="") if says != given]


def record_checks(headline: dict, table: dict, section: dict[str, Any]) -> None:
    print("\n-- F's record, Amendment A11 item 21 --")
    record = amendment_item(PLAN.read_text(), "A11", 21).splitlines()
    mismatches = record_mismatches(record, headline, table, section)
    check(not mismatches,
          "every number in Amendment A11 item 21's record is the one the committed files give, at the precision the "
          "record prints, and the hash is F's section's"
          + "".join(f"\n      record: {says!r}\n      JSON:   {gives!r}" for says, gives in mismatches))  # fmt: skip
    text = "".join(record)
    check("To be filled in" not in text and len(numeric(record)) == len(numeric(record_lines(headline, table, section))),
          "the record is filled in: every line the files give is in it")  # fmt: skip
    check(DESCRIPTIVE in text and "no multiplicity correction" in text,
          "the record calls the comparisons descriptive, with no multiplicity correction")  # fmt: skip
    check(DIAGNOSTIC_LABEL in text, f'and labels the diagnostics "{DIAGNOSTIC_LABEL}"')
    # A guard proves it can fail, on the lines the files give: a slip in one digit, and a number the check doesn't know.
    good = record_lines(headline, table, section)
    check(
        not record_mismatches(good, headline, table, section),
        "known positive: the lines the files give match themselves",
    )
    quality = next(i for i, line in enumerate(good) if line.startswith("| F: quality there"))
    slip = re.sub(r"(\d+\.)(\d)%", lambda m: f"{m[1]}{(int(m[2]) + 1) % 10}%", good[quality], count=1)
    check(slip != good[quality] and len(record_mismatches([*good[:quality], slip, *good[quality + 1 :]], headline, table, section)) == 1,
          f"the record check catches a slip in one number: {slip!r}")  # fmt: skip
    concordance = next(i for i, line in enumerate(good) if line.startswith("| F, concordance"))
    slip = re.sub(r"0\.(\d)", lambda m: f"0.{(int(m[1]) + 1) % 10}", good[concordance], count=1)
    check(slip != good[concordance] and len(record_mismatches([*good[:concordance], slip, *good[concordance + 1 :]], headline, table, section)) == 1,
          f"and in a diagnostic's: {slip!r}")  # fmt: skip
    check(len(record_mismatches([*good, "    - One number more: 1.0 points."], headline, table, section)) == 1,
          "the record check catches a number it doesn't compare")  # fmt: skip
    check(len(record_mismatches(good[:-1], headline, table, section)) == 1, "and a line missing from the record")


# ---------------------------------------------------------------------------------------------------------------
# The committed files


def v1_kept(headline: dict, per_task: dict) -> None:
    print("\n-- table v2 keeps table v1, and the figures show policies only (Amendment A11, items 13 and 16) --")
    # Chart v3 (Stage 6 · 12) adds G's and G+'s entries after E's and F's; E's and F's, in this order, are what v2 pinned.
    named = [E_SECTION, F_SECTION, *JEV_SECTIONS]
    check(list(headline["routers"]) == named and all(list(p["routers"]) == named for p in headline["panels"].values()),
          "figures/pareto.json holds E's entry and F's and, since Stage 6, G's and G+'s, in that order, under their "
          "names in the operating-points file")  # fmt: skip
    check(all(list(p["routers"]) == named for p in per_task["panels"].values()),
          "figures/pareto_tasks.json holds E's part and F's, then G's and G+'s, in each task")  # fmt: skip
    check(headline["routers"][F_SECTION]["label"] == LABELS[F], f"F's label is the chart's: {LABELS[F]}")
    for panel in headline["panels"].values():
        check(list(panel["routers"][F_SECTION]) == list(panel["routers"][E_SECTION]) == ["steps", "operating_points"],
              "F's part of a panel has E's keys: its steps and its operating points")  # fmt: skip
    for task, panel in per_task["panels"].items():
        check(list(panel["routers"][F_SECTION]) == list(panel["routers"][E_SECTION]), f"{task}: F's part has E's keys")
    text = FIGURE_DATA.read_text() + TASK_FIGURE_DATA.read_text()
    check(F_TRAIN not in text and TASK_RATES not in text and "diagnostic, dev only" not in text,
          "neither figure JSON holds a diagnostic: F-train and the task-rates diagnostic are not policies")  # fmt: skip
    check(not any("band" in key for key in headline), "no band")
    check(set(json.loads(OPERATING_POINTS_FILE.read_text())) == {E_SECTION, F_SECTION, *JEV_SECTIONS},
          "the operating-points file holds E's section and F's and, since Stage 6, G's and G+'s, and no other")  # fmt: skip
    check_sections(required=(E_SECTION, F_SECTION, *JEV_SECTIONS))
    check(refused(lambda: check_sections(required=(F_TRAIN,))), f"known positive: the allowlist refuses {F_TRAIN}")


def f_rows(headline: dict, section: dict[str, Any]) -> None:
    print("\n-- F's steps and operating points, read from the JSON (Amendment A9, items 9, 10, 14 and 15) --")
    check(headline["routers"][F_SECTION]["operating_points_sha256"] == section_hash(section)
          == hashlib.sha256(json.dumps(section, sort_keys=True, indent=2).encode()).hexdigest(),
          "the figure JSON's hash of F's section is the sha256 of the section read from the file, sorted keys")  # fmt: skip
    for mix in PANELS:
        panel = headline["panels"][mix]
        part = panel["routers"][F_SECTION]
        line, points = part["steps"], part["operating_points"]
        a = panel["configs"][FRONTIER]
        check(line[0]["threshold"] is None and line[0]["quality"] == a["quality"]["value"]
              and line[0]["cost_per_1k"] == a["cost_per_1k"]["value"],
              f"{mix} mix: F's line starts at A, as F has no overhead")  # fmt: skip
        thresholds = [s["threshold"] for s in line[1:]]
        check(
            thresholds == sorted(thresholds, reverse=True), f"{mix} mix: F's steps run from the highest threshold down"
        )
        check(set(points) == set(section[mix]) == set(TARGETS), f"{mix} mix: an operating point for each target")
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
            check(abs(saved["per_1k"] - (a["cost_per_1k"]["value"] - point["cost_per_1k"]["value"])) < ROUNDING
                  and abs(saved["share"] - saved["per_1k"] / a["cost_per_1k"]["value"]) < ROUNDING,
                  f"{where}: savings against A are A's cost less F's: ${saved['per_1k']:.2f} per 1k ({saved['share']:.1%})")  # fmt: skip
            check(abs(sum(point["routing"].values()) - 1) < ROUNDING
                  and (abs(point["routing"][FRONTIER] - 1) < ROUNDING if point["fell_back"] else True),
                  f"{where}: the routing mix sums to 1" + (", all on opus: it falls back to A" if point["fell_back"] else ""))  # fmt: skip
            captured = point["captured"]
            bounds = [captured[b] for b in ("value", "low", "high") if captured[b] is not None]
            check(all(0 <= b <= 1 for b in bounds) and 0 <= captured["met"] <= 1
                  and (captured["value"] == 0 if point["fell_back"] else captured["value"] is not None and captured["value"] > 0),
                  f"{where}: the captured fraction, interval included, is at most 1"
                  + (" and 0 for a fallback" if point["fell_back"] else f" ({captured['value']:.2f})"))  # fmt: skip
            reading = envelope_at(panel["envelope"], point["cost_per_1k"]["value"])
            gap = point["matched_cost"]
            check(reading is not None and abs(gap["envelope"] - reading) < ROUNDING
                  and abs(gap["difference"] - (point["quality"]["value"] - reading)) < ROUNDING
                  and gap["low"] is not None and gap["low"] <= gap["difference"] + ROUNDING,
                  f"{where}: the envelope's quality at F's cost is read along its corners, and F's difference from it "
                  f"is F less that ({points_text(gap['difference'])} points)")  # fmt: skip


def includes_zero(headline: dict) -> None:
    """The README says F's paired difference from the envelope at matched cost includes 0 at every operating point."""
    gaps = [(mix, key, point["matched_cost"]) for mix in PANELS
            for key, point in headline["panels"][mix]["routers"][F_SECTION]["operating_points"].items()]  # fmt: skip
    check(len(gaps) == 6 and all(g["low"] <= 0 <= g["high"] for _, _, g in gaps),
          "F's paired difference from the envelope at matched cost has an interval that includes 0 at each of its six "
          "operating points, as the README says")  # fmt: skip


def table_v2_file(headline: dict, table: dict, curve: dict) -> None:
    print("\n-- table v2's file: the comparisons and the diagnostics (Amendment A11, items 12, 13 and 16) --")
    check(table["split"] == headline["split"] and table["n"] == headline["n"]
          and table["price_snapshot"] == headline["price_snapshot"] and table["bootstrap"] == headline["bootstrap"],
          "it is of the same split, items, prices and resamples as figures/pareto.json")  # fmt: skip
    check(table["comparisons"]["description"] == DESCRIPTIVE and DESCRIPTIVE in table["units"]["comparisons"],
          "the comparisons are described as descriptive, with no multiplicity correction")  # fmt: skip
    check(table["diagnostics"]["label"] == DIAGNOSTIC_LABEL and DIAGNOSTIC_LABEL in table["units"]["diagnostics"],
          f"the diagnostics are labeled \"{DIAGNOSTIC_LABEL}\"")  # fmt: skip
    check(list(table["comparisons"]["same_target"]) == [f"{a} less {b}" for a, b in PAIRS],
          "the comparisons are F against E, F-train against E and F against F-train, first less second")  # fmt: skip
    points = {mix: {E: headline["panels"][mix]["routers"][E_SECTION]["operating_points"],
                    F: headline["panels"][mix]["routers"][F_SECTION]["operating_points"],
                    **{name: table["diagnostics"]["operating_points"][name][mix] for name in DIAGNOSTICS}}
              for mix in MIXES}  # fmt: skip
    for pair in PAIRS:
        by_mix = table["comparisons"]["same_target"][f"{pair[0]} less {pair[1]}"]
        for mix in MIXES:
            for key in TARGETS:
                cell, first, second = by_mix[mix][key], points[mix][pair[0]][key], points[mix][pair[1]][key]
                where = f"{pair[0]} less {pair[1]}, {mix} {key}"
                both = first["fell_back"] and second["fell_back"]
                if both:
                    check(cell["reading"] == "both fall back to A" and cell["cost_per_1k"] is None
                          and cell["quality"] is None, f"{where}: both fall back to A, no difference")  # fmt: skip
                    continue
                cost = first["cost_per_1k"]["value"] - second["cost_per_1k"]["value"]
                quality = first["quality"]["value"] - second["quality"]["value"]
                est_cost, est_quality = cell["cost_per_1k"], cell["quality"]
                check(not cell["reading"] and abs(est_cost["value"] - cost) < ROUNDING
                      and abs(est_quality["value"] - quality) < ROUNDING
                      and est_cost["low"] <= est_cost["value"] <= est_cost["high"]
                      and est_quality["low"] <= est_quality["value"] <= est_quality["high"],
                      f"{where}: cost {cost:+.2f}, quality {100 * quality:+.1f} points: the operating points' "
                      f"difference, inside its interval")  # fmt: skip
                check((cell["first_fell_back"], cell["second_fell_back"]) == (first["fell_back"], second["fell_back"]),
                      f"{where}: a side that fell back is marked")  # fmt: skip
    for name in DIAGNOSTICS:
        rows = table["diagnostics"]["operating_points"][name]
        check(set(rows) == set(MIXES) and all(set(by) == set(TARGETS) for by in rows.values()),
              f"{name}'s operating points: both mixes and all three targets, none in the operating-points file")  # fmt: skip
    check(
        set(json.loads(OPERATING_POINTS_FILE.read_text())) == {E_SECTION, F_SECTION, *JEV_SECTIONS},
        "and the file still holds the routers an amendment names only (E, F, G and G+), no diagnostic",
    )
    concordance = table["diagnostics"]["within_task_concordance"]
    check(all(concordance[TASK_RATES][c]["pooled"][k] == 0.5 for c in SCORED for k in ("value", "low", "high")),
          "the task-rates diagnostic's within-task concordance is exactly 0.5, with no width, as it must be")  # fmt: skip
    check(all(0.4 < concordance[n][c]["pooled"]["value"] < 0.7 for n in (F, F_TRAIN) for c in SCORED)
          and all(concordance[n][c]["pooled"]["low"] < concordance[n][c]["pooled"]["value"]
                  < concordance[n][c]["pooled"]["high"] for n in (F, F_TRAIN) for c in SCORED),
          "F's and F-train's concordance sit near chance, inside their intervals")  # fmt: skip
    check(all(set(concordance[n][c]["per_task"]) == set(DEV_TASKS) for n in concordance for c in SCORED)
          and all((concordance[n][c]["per_task"][t]["warning"] is not None) == (t in ("ifeval", "mbppplus"))
                  for n in concordance for c in SCORED for t in DEV_TASKS),
          "concordance is given per task, with A7 item 2's warning on IFEval and MBPP+ only")  # fmt: skip
    fold = table["diagnostics"]["fold_quantiles"]
    check(list(fold) == ["0", "1", "2", "3", "4"] and sum(f["items"] for f in fold.values()) == table["n"]
          and all(list(f["by_config"]) == list(SCORED) for f in fold.values())
          and all(list(v.values()) == sorted(v.values()) for f in fold.values() for v in f["by_config"].values()),
          "the per-fold quantiles: five fold models covering every dev item, each config's quantiles in order")  # fmt: skip

    learning = table["diagnostics"]["learning_curve"]
    check(curve["curve"] == learning, "figures/learning_curve.json holds table v2's learning curve, number for number")
    check(learning["sizes"] == [0.25, 0.5, 0.75, 1.0], "the curve's sizes are 25, 50, 75 and 100%")
    for config in SCORED:
        at = learning["points"][config]
        labels = [at[s]["labels"] for s in ("0.25", "0.5", "0.75", "1")]
        check(labels == sorted(labels) and learning["f_train"][config]["labels"] == 991,
              f"{config}: more labels along the curve ({labels[0]:.0f} to {labels[-1]:.0f}), F-train's point at 991")  # fmt: skip
        for metric in ("log_loss", "concordance"):
            step = learning["headline"][config][metric]["pooled"]
            gap = at["1"][metric]["pooled"]["value"] - at["0.75"][metric]["pooled"]["value"]
            check(abs(step["value"] - gap) < ROUNDING and step["low"] <= step["value"] <= step["high"],
                  f"{config}: the {metric.replace('_', ' ')} headline is the 100% point less the 75% point "
                  f"({step['value']:+.4f})")  # fmt: skip
        reference = curve["reference"]["task_rates_log_loss"][config]["value"]
        reading = table["diagnostics"]["log_loss"][TASK_RATES][config]["pooled"]["value"]
        check(
            reference == reading, f"{config}: the dashed line is the task-rates diagnostic's log loss ({reference:.4f})"
        )


def learning_figure(curve: dict) -> None:
    print("\n-- the learning-curve figure, drawn from the committed numbers (Amendment A11, item 16) --")
    with tempfile.TemporaryDirectory() as folder:
        fig: Figure = draw_learning_curve(curve, Path(folder) / "learning_curve.png")
    axes = fig.axes
    check(len(axes) == 6, "two rows (log loss, concordance) of three panels, one per config")
    # An error bar's data line carries the series' marker; its caps carry "_".
    joined = [line for ax in axes for line in ax.get_lines() if line.get_marker() == "o"]
    check(len(joined) == 6 and all(np.asarray(line.get_xdata()).size == 4 and str(line.get_color()).lower() == F_STYLE.color
                                   for line in joined),
          "F's line joins its four points on every panel, in F's colour")  # fmt: skip
    separate = [line for ax in axes for line in ax.get_lines() if line.get_marker() == "s"]
    check(len(separate) == 6 and all(np.asarray(line.get_xdata()).size == 1 and str(line.get_color()).lower() == F_TRAIN_COLOR
                                     and line.get_linestyle() == "None" for line in separate),
          "F-train is one separate marker on every panel, in its own colour, joined to nothing")  # fmt: skip
    for column, config in enumerate(SCORED):
        dashed = [line for line in axes[column].get_lines() if line.get_linestyle() == "--"]
        reference = curve["reference"]["task_rates_log_loss"][config]["value"]
        check(len(dashed) == 1 and np.asarray(dashed[0].get_ydata()).tolist() == [reference, reference],
              f"{config}: one dashed line at the task-rates diagnostic's log loss ({reference:.4f})")  # fmt: skip
        dotted = [line for line in axes[3 + column].get_lines() if line.get_linestyle() == ":"]
        check(
            len(dotted) == 1 and np.asarray(dotted[0].get_ydata()).tolist() == [0.5, 0.5],
            f"{config}: 0.5 marks the concordance panel",
        )
        check(not [line for line in axes[column].get_lines() if line.get_linestyle() == ":"],
              f"{config}: and not the log-loss panel")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# On dev, where results/ holds the matrix


def on_dev(headline: dict, per_task: dict, table: dict, curve: dict) -> None:
    from analysis import matrix as results_matrix
    from analysis import table_v2, trained_table, trained_walk
    from analysis.bootstrap import resample
    from analysis.router_panels import (
        config_mcnemar,
        router_part_of_mix_panel,
        router_part_of_task_panel,
        step_outcomes,
    )
    from analysis.rules_walk import dev_walks
    from analysis.simulator import always, load_mixes, simulate
    from routers.decision import choose

    print("\n-- on dev: F's parts rebuild, and McNemar's counts are the matrix's (Amendment A9, item 11) --")
    before = OPERATING_POINTS_FILE.read_bytes()
    dev = results_matrix.load().split("dev")
    mixes = load_mixes()
    fixed = {config: simulate(dev, always(dev, config)) for config in LADDER}
    draws = resample(dev)
    f_table = trained_table.dev_table()
    walks = dev_walks(dev, f_table, draws)
    steps = step_outcomes(dev, f_table, fixed[FRONTIER])
    check(
        trained_walk.rebuilt_section() == read_section(F_SECTION), "the walk on dev reproduces F's section of the file"
    )
    for mix in PANELS:
        savings = draws.savings(dev, fixed[FRONTIER], mixes[mix])
        rebuilt = router_part_of_mix_panel(draws, savings, steps, walks[mix], fixed, mixes[mix])
        check(json.loads(json.dumps(rebuilt)) == headline["panels"][mix]["routers"][F_SECTION],
              f"{mix} mix: F's steps and operating points in figures/pareto.json are what the package gives now")  # fmt: skip
    scores = {entry.item_id: entry.router_scores for entry in f_table.items}

    def right(item_id: str, config: str) -> bool:
        return dev.cells[(item_id, config)].success == 1

    for task in DEV_TASKS:
        rebuilt_task = {"mcnemar_vs_A": config_mcnemar(fixed, task),
                        "routers": {F: router_part_of_task_panel(draws, steps, walks, task)}}  # fmt: skip
        committed = {"mcnemar_vs_A": per_task["panels"][task]["mcnemar_vs_A"],
                     "routers": {F: per_task["panels"][task]["routers"][F_SECTION]}}  # fmt: skip
        check(json.loads(json.dumps(rebuilt_task)) == committed,
              f"{task}: F's steps, operating points and McNemar rows in figures/pareto_tasks.json are current")  # fmt: skip
        ids = [i for i, item in dev.items.items() if item.task == task]
        shown = 0
        for mix, by_target in per_task["panels"][task]["routers"][F_SECTION]["operating_points"].items():
            for key, point in by_target.items():
                if point["fell_back"]:
                    check(point["threshold"] is None and point["mcnemar"] is None,
                          f"{task}, {mix} {key}: a fallback has no McNemar row, shown as —")  # fmt: skip
                    continue
                chosen = {i: choose(scores[i], point["threshold"]) for i in ids}
                b = sum(right(i, chosen[i]) and not right(i, FRONTIER) for i in ids)
                c = sum(right(i, FRONTIER) and not right(i, chosen[i]) for i in ids)
                shown += 1
                check(point["mcnemar"] == {"b": b, "c": c, "p": mcnemar_p(b, c)},
                      f"{task}, {mix} {key}: F at {point['threshold']:.4f} against A: b = {b}, c = {c}, "
                      f"p = {mcnemar_p(b, c):.3f}")  # fmt: skip
        check(shown > 0, f"{task}: F has operating points that don't fall back, each with a row")

    print("\n-- on dev: table v2 and the learning curve rebuild, and nothing enters the operating-points file --")
    rebuilt_table, rebuilt_figure = table_v2.build()
    check(json.loads(json.dumps(rebuilt_table)) == table, "figures/table_v2.json is what the package gives now")
    check(json.loads(json.dumps(rebuilt_figure)) == curve, "figures/learning_curve.json is what the package gives now")
    check(OPERATING_POINTS_FILE.read_bytes() == before, "building them leaves the operating-points file byte for byte")


def main() -> int:
    if "--print" in sys.argv:
        headline = json.loads(FIGURE_DATA.read_text())
        table = json.loads(TABLE_V2.read_text())
        print("\n".join(record_lines(headline, table, read_section(F_SECTION))))
        return 0
    headline, per_task = (json.loads(path.read_text()) for path in (FIGURE_DATA, TASK_FIGURE_DATA))
    table, curve = (json.loads(path.read_text()) for path in (TABLE_V2, LEARNING_FIGURE_DATA))
    section = read_section(F_SECTION)
    v1_kept(headline, per_task)
    f_rows(headline, section)
    includes_zero(headline)
    table_v2_file(headline, table, curve)
    learning_figure(curve)
    record_checks(headline, table, section)
    if (PROJECT_ROOT / "results" / "dev_matrix.sqlite").exists():
        on_dev(headline, per_task, table, curve)
    else:
        print("\n(no results/dev_matrix.sqlite here: the sections that rebuild from the dev matrix are skipped)")
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
