"""Check Stage 6's chart and table v3 and G's and G+'s record (PLAN_v4.md §12; Amendment A15, items 14, 15 and 20).

Run from the project root:  uv run python -m analysis.check_stage6
Print the record's numeric lines as the JSONs give them:  uv run python -m analysis.check_stage6 --print

Reads the committed figure JSONs, `routers/operating_points.json`, `routers/jev_settings.json` and PLAN_v4.md, so CI
runs it. The section that compares with the stored dev responses and rebuilds from the dev matrix runs only where
`results/` holds them, as `check_stage5` does, and the git comparison with Stage 5's files only where that commit is
in the clone. Never a test item, and no Jev call.

- chart v3 keeps v1 and v2: tables v1 and v2's files are byte for byte what Stage 5 committed, E's and F's parts of
  both figure JSONs are as they were, and the figure JSONs hold G's and G+'s parts after them, as the same keys and
  nothing of a diagnostic or a band;
- G's and G+'s steps and operating points, read from the JSON by hand as `check_stage5` reads F's: each cross sits on
  the step the file picked, the thresholds, the fallbacks, the hash of each section;
- the headline and per-task figures drawn from the committed numbers show all four routers, each in its own colour,
  with a legend entry each, a label offset for every operating-point label and none stale, and the caption line that
  says G's and G+'s costs include Jev's overhead per request and E's and F's have none;
- table v3's file: its operating points are the figure JSON's, the same-target comparisons are the differences of the
  operating points and "both fall back to A" where both did, G against F-train is not among them, McNemar's p is the
  exact binomial of its b and c, the matched-cost envelope is read along the corners, the diagnostics are labeled
  "diagnostic, dev only", and every table has its units;
- Amendment A15 item 20's record: every number in it is the one the committed files give, at the precision it
  prints, through the plan's amendment-item reader, with guards that a one-digit slip and an uncompared number both
  fail, and the comparisons are called descriptive with no multiplicity correction;
- the README's numbers about G and G+: each sentence that quotes one is the one the committed files give, with a guard
  that a one-digit slip is not in the README;
- on dev (hook only): G's and G+'s parts and table v3 equal a rebuild, the first dev call's server `Date` is the
  stored records', E's and F's tables charge no overhead and G's and G+'s charge it on every request.
"""

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from analysis.chart_routers import ROUTERS
from analysis.check_chart_routers import colours
from analysis.check_stage5 import envelope_at, mcnemar_p, numeric, threshold_text
from analysis.operating_points import OPERATING_POINTS_FILE, build_section, read_section, section_hash
from analysis.pareto import (
    FIGURE_DATA,
    OVERHEAD_LINE,
    PANELS,
    TASK_FIGURE_DATA,
    TASKS,
    draw_headline,
    draw_tasks,
    headline_caption,
    point_labels,
    task_caption,
)
from analysis.plan import PLAN, amendment_item
from analysis.references import ROUNDING
from analysis.router_names import F_TRAIN, G_PLUS, LABELS, NAMED_ROUTERS, E, F, G
from analysis.rules_walk import MIXES
from analysis.table_v2 import DESCRIPTIVE, DIAGNOSTIC_LABEL
from analysis.table_v3 import TABLE_V3
from harness.client import PROJECT_ROOT
from routers.ladder import FRONTIER, SCORED

# The commit Stage 6 · 11 left on the branch: tables v1 and v2 and E's and F's parts are as it has them.
STAGE5_FILES_AT = "9217e7536071f69e52c146c70319009cad5c5e9c"
SETTINGS_FILE = PROJECT_ROOT / "routers" / "jev_settings.json"
E_SECTION, F_SECTION = NAMED_ROUTERS[E], NAMED_ROUTERS[F]
JEV_SECTIONS = {G: NAMED_ROUTERS[G], G_PLUS: NAMED_ROUTERS[G_PLUS]}
TARGETS = ("100%", "99%", "97%")
DEV_TASKS = tuple(TASKS)
# A15 item 14's comparisons, first less second, written here from the amendment: G against F-train is rejected.
PAIRS = ((G, E), (G, F), (G_PLUS, E), (G_PLUS, F_TRAIN), (G_PLUS, F), (G_PLUS, G))
NOTES = {"realistic": "a note"}  # the headline figure sizes its titles by its notes
UNITS_NEEDED = ("operating_points", "comparisons", "same_target", "matched_cost", "mcnemar", "diagnostics",
                "within_task_concordance", "concordance_differences", "router_score_quantiles", "latency",
                "dev_calls")  # fmt: skip
problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)


# ---------------------------------------------------------------------------------------------------------------
# Item 20's record from the committed files


def interval(estimate: Mapping[str, float], scale: float, places: int) -> str:
    return f"{scale * estimate['value']:.{places}f}"


def record_lines(
    headline: dict, sections: Mapping[str, dict[str, Any]], dates: Mapping[str, str], price: float
) -> list[str]:
    """Amendment A15 item 20's lines that hold a number, as the figure JSONs, the operating-points file and the settings
    give them, at the precision the record prints."""
    bootstrap = headline["bootstrap"]
    targets = ", ".join(TARGETS)
    lines = [
        f"20. **Record.** G's and G+'s operating points on {headline['split']} (run {headline['run']}, n = "
        f"{headline['n']}, list prices of {headline['price_snapshot']}, Jev's overhead at ${price:g} per million input "
        f"tokens), from `figures/pareto.json`, and the hash of each router's section of "
        f"`routers/operating_points.json`, as A9 item 18 and A11 item 21 record E's and F's; the comparisons' and "
        f"diagnostics' numbers are from `figures/table_v3.json`. Intervals are percentile 95% from "
        f"{bootstrap['resamples']:,} paired resamples, and `analysis/check_stage6.py` compares every number here with "
        f"those files.",
        "    | | Realistic | Uniform |",
        "    |---|---|---|",
    ]
    for letter, name in JEV_SECTIONS.items():
        mixes = [headline["panels"][mix]["routers"][name]["operating_points"] for mix in MIXES]

        def row(label: str, cell: Callable[[dict], str], mixes: list[dict] = mixes, letter: str = letter) -> str:
            return f"    | {letter}: {label} | {' | '.join(', '.join(cell(p) for p in points.values()) for points in mixes)} |"

        lines += [
            row(f"threshold at {targets} of A's quality", threshold_text),
            row("quality there", lambda p: f"{100 * p['quality']['value']:.1f}%"),
            row("$ per 1k there", lambda p: f"${p['cost_per_1k']['value']:.2f}"),
            row("savings against A there", lambda p: f"{p['savings_vs_A']['share']:.1%}"),
            row("captured fraction there", lambda p: f"{p['captured']['value']:.2f}"),
            row("share of resamples meeting the target there", lambda p: f"{p['captured']['met']:.1%}"),
        ]
    lines += [
        '    - Where a target reads "falls back to A", the walk certified nothing cheaper than A (A9 item 10): the '
        "operating point is A itself, with no router call and no overhead, and its captured fraction is 0.00. "
        "Everywhere else the cost includes Jev's overhead on every request.",
        f"    - G's section of `routers/operating_points.json`, written with sorted keys, has the 64-digit sha256 "
        f"`{section_hash(sections[G])}`.",
        f"    - G+'s section of `routers/operating_points.json`, written with sorted keys, has the 64-digit sha256 "
        f"`{section_hash(sections[G_PLUS])}`.",
        f"    - The first dev call's server `Date` is `{dates['first_call_server_date']}`, after the freeze commit "
        f"(item 19). It is in GMT.",
        f"    - The comparisons in `figures/table_v3.json` are {DESCRIPTIVE}, and its diagnostics are labeled "
        f'"{DIAGNOSTIC_LABEL}".',
    ]
    return lines


def record_mismatches(record: Sequence[str], good: Sequence[str]) -> list[tuple[str, str]]:
    """Each numeric line of the record that differs from the line the files give, paired with it; a line missing on
    either side is paired with ''."""
    return [(says, given) for says, given in zip_longest(numeric(list(record)), numeric(list(good)), fillvalue="")
            if says != given]  # fmt: skip


def record_checks(
    headline: dict, sections: Mapping[str, dict[str, Any]], dates: Mapping[str, str], price: float
) -> None:
    print("\n-- G's and G+'s record, Amendment A15 item 20 --")
    record = amendment_item(PLAN.read_text(), "A15", 20).splitlines()
    good = record_lines(headline, sections, dates, price)
    mismatches = record_mismatches(record, good)
    check(not mismatches,
          "every number in Amendment A15 item 20's record is the one the committed files give, at the precision the "
          "record prints, and the hashes are G's and G+'s sections'"
          + "".join(f"\n      record: {says!r}\n      JSON:   {gives!r}" for says, gives in mismatches))  # fmt: skip
    text = "".join(record)
    check("To be filled in" not in text and len(numeric(record)) == len(numeric(good)),
          "the record is filled in: every line the files give is in it")  # fmt: skip
    check(DESCRIPTIVE in text and "no multiplicity correction" in text,
          "the record calls the comparisons descriptive, with no multiplicity correction")  # fmt: skip
    check(DIAGNOSTIC_LABEL in text, f'and labels the diagnostics "{DIAGNOSTIC_LABEL}"')
    # A guard proves it can fail, on the lines the files give: a slip in one digit, and a number the check doesn't know.
    check(not record_mismatches(good, good), "known positive: the lines the files give match themselves")
    quality = next(i for i, line in enumerate(good) if line.startswith("    | G: quality there"))
    slip = re.sub(r"(\d+\.)(\d)%", lambda m: f"{m[1]}{(int(m[2]) + 1) % 10}%", good[quality], count=1)
    check(slip != good[quality] and len(record_mismatches([*good[:quality], slip, *good[quality + 1 :]], good)) == 1,
          f"the record check catches a slip in one number: {slip!r}")  # fmt: skip
    plus = next(i for i, line in enumerate(good) if line.startswith("    | G+: threshold"))
    slip = re.sub(r"0\.(\d)", lambda m: f"0.{(int(m[1]) + 1) % 10}", good[plus], count=1)
    check(slip != good[plus] and len(record_mismatches([*good[:plus], slip, *good[plus + 1 :]], good)) == 1,
          f"and in G+'s: {slip!r}")  # fmt: skip
    date = next(i for i, line in enumerate(good) if "server `Date`" in line)
    slip = good[date].replace(":54 GMT", ":55 GMT").replace(":50 GMT", ":51 GMT")
    check(slip != good[date] and len(record_mismatches([*good[:date], slip, *good[date + 1 :]], good)) == 1,
          "and in the first dev call's Date")  # fmt: skip
    check(len(record_mismatches([*good, "    - One number more: 1.0 points."], good)) == 1,
          "the record check catches a number it doesn't compare")  # fmt: skip
    check(len(record_mismatches(good[:-1], good)) == 1, "and a line missing from the record")


# ---------------------------------------------------------------------------------------------------------------
# Chart v3 and the committed files


def stage5_kept(headline: dict, per_task: dict) -> None:
    print("\n-- chart v3 keeps tables v1 and v2, and the figures show policies only (A15 items 14 and 15) --")
    named = [E_SECTION, F_SECTION, *JEV_SECTIONS.values()]
    check(
        [r.name for r in ROUTERS] == named,
        "the router list is E, F, G and G+, under their names in the operating-points file",
    )
    check(list(headline["routers"]) == named and all(list(p["routers"]) == named for p in headline["panels"].values()),
          "figures/pareto.json holds E's entry, F's, G's and G+'s, in that order")  # fmt: skip
    check(all(list(p["routers"]) == named for p in per_task["panels"].values()),
          "figures/pareto_tasks.json holds the four routers' parts in each task")  # fmt: skip
    for name in JEV_SECTIONS.values():
        check(headline["routers"][name]["label"] == LABELS[G if name == NAMED_ROUTERS[G] else G_PLUS],
              f"{name}'s label is the chart's")  # fmt: skip
        for panel in headline["panels"].values():
            check(list(panel["routers"][name]) == list(panel["routers"][E_SECTION]) == ["steps", "operating_points"],
                  f"{name}'s part of a panel has E's keys: its steps and its operating points")  # fmt: skip
        check(all(list(p["routers"][name]) == list(p["routers"][E_SECTION]) for p in per_task["panels"].values()),
              f"{name}'s part of each task has E's keys")  # fmt: skip
    text = FIGURE_DATA.read_text() + TASK_FIGURE_DATA.read_text()
    check(F_TRAIN not in text and "task rates" not in text and DIAGNOSTIC_LABEL not in text,
          "neither figure JSON holds a diagnostic: F-train and the task-rates diagnostic are not policies")  # fmt: skip
    check(not any("band" in key for key in headline), "no band")

    wanted = git("cat-file", "-e", f"{STAGE5_FILES_AT}^{{commit}}").returncode == 0
    if not wanted:
        print(f"(commit {STAGE5_FILES_AT[:7]} is not in this clone: the comparison with Stage 5's files is skipped)")
        return
    for file in ("figures/table_v2.json", "figures/learning_curve.json", "figures/learning_curve.png"):
        check(git("diff", "--quiet", STAGE5_FILES_AT, "--", file).returncode == 0,
              f"{file} is byte for byte what Stage 5 committed")  # fmt: skip
    for file, now in (("figures/pareto.json", headline), ("figures/pareto_tasks.json", per_task)):
        before = json.loads(git("show", f"{STAGE5_FILES_AT}:{file}").stdout)
        check(only_added(before, now, (E_SECTION, F_SECTION)), f"{file}: E's and F's parts, every number and key, are as "
              f"Stage 5 committed them; G's and G+'s are added beside them")  # fmt: skip
    bad = json.loads(json.dumps(headline))
    bad["panels"]["realistic"]["routers"][E_SECTION]["steps"][1]["quality"] += 0.001
    before = json.loads(git("show", f"{STAGE5_FILES_AT}:figures/pareto.json").stdout)
    check(not only_added(before, bad, (E_SECTION, F_SECTION)),
          "known positive: a changed number in E's part is found")  # fmt: skip


def only_added(before: dict, now: dict, kept: Sequence[str]) -> bool:
    """Whether `now` is `before` with other routers' entries added: the routers named in `kept` are equal wherever the
    figure JSON holds routers, and everything else is equal outright."""

    def strip(data: dict) -> dict:
        data = json.loads(json.dumps(data))
        if "routers" in data:
            data["routers"] = {k: v for k, v in data["routers"].items() if k in kept}
        for panel in data["panels"].values():
            panel["routers"] = {k: v for k, v in panel["routers"].items() if k in kept}
        return data

    return strip(now) == before


def jev_rows(headline: dict, name: str, section: dict[str, Any]) -> None:
    print(
        f"\n-- {name}'s steps and operating points, read from the JSON (A9 items 9, 10, 14 and 15; A15 items 9 and 10) --"
    )
    check(headline["routers"][name]["operating_points_sha256"] == section_hash(section)
          == hashlib.sha256(json.dumps(section, sort_keys=True, indent=2).encode()).hexdigest(),
          f"the figure JSON's hash of {name}'s section is the sha256 of the section read from the file, sorted keys")  # fmt: skip
    for mix in PANELS:
        panel = headline["panels"][mix]
        part = panel["routers"][name]
        line, points = part["steps"], part["operating_points"]
        a = panel["configs"][FRONTIER]
        check(line[0]["threshold"] is None and line[0]["quality"] == a["quality"]["value"]
              and line[0]["cost_per_1k"] == a["cost_per_1k"]["value"],
              f"{mix} mix: {name}'s line starts at A, which makes no router call")  # fmt: skip
        thresholds = [s["threshold"] for s in line[1:]]
        check(
            thresholds == sorted(thresholds, reverse=True), f"{mix} mix: the steps run from the highest threshold down"
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
            check(point["fell_back"] == (point["threshold"] is None)
                  and (not point["fell_back"] or point["captured"]["value"] == 0),
                  f"{where}: a fallback is A itself, with captured fraction 0")  # fmt: skip
            rows = on_line[0]
            check(abs(point["savings_vs_A"]["share"] - (1 - rows["cost_per_1k"] / a["cost_per_1k"]["value"])) < ROUNDING,
                  f"{where}: savings against A are A's cost less the step's, over A's")  # fmt: skip
            read = envelope_at(panel["envelope"], rows["cost_per_1k"])
            check(point["matched_cost"]["envelope"] == read
                  or (read is not None and abs(point["matched_cost"]["envelope"] - read) < ROUNDING),
                  f"{where}: the envelope's quality at {name}'s cost, along its corners")  # fmt: skip


def label_offsets(headline: dict) -> None:
    print("\n-- label offsets are placed for every operating-point label, after the walks (A15 item 15) --")
    for router in ROUTERS[2:]:
        for mix in PANELS:
            labels = {
                label for _, label in point_labels(headline["panels"][mix]["routers"][router.name]["operating_points"])
            }
            placed = {label for m, label in router.style.point_offsets if m == mix}
            check(
                labels == placed,
                f"{router.short}, {mix} mix: an offset for each label drawn ({sorted(labels)}), none stale",
            )


def drawn(headline: dict, per_task: dict) -> None:
    print("\n-- chart v3 drawn from the committed numbers (A15 item 15) --")
    colours_of = {r.name: r.style.color for r in ROUTERS}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        caption = headline_caption("run 0, n = 744.", "Crosses.", ROUTERS)
        fig = draw_headline(headline["panels"], NOTES, caption, folder / "a.png", ROUTERS)
        legend = [t.get_text() for t in fig.legends[0].get_texts()]
        steps = [sum(label.startswith(f"{r.label}, its steps") for label in legend) for r in ROUTERS]
        points = [sum(label.startswith(f"{r.short}'s operating points") for label in legend) for r in ROUTERS]
        check(steps == [1] * 4 and points == [1] * 4,
              "the headline legend has each of the four routers' steps and operating points once")  # fmt: skip
        check(set(colours_of.values()) <= colours(fig), "the headline figure draws all four routers' colours")
        check(has_overhead_line(fig), "the headline caption says G's and G+'s costs include Jev's overhead per request, "
              "and E's and F's have none")  # fmt: skip
        plain = draw_headline(headline["panels"], NOTES, "caption", folder / "b.png", ROUTERS)
        check(not has_overhead_line(plain), "known positive: a caption without the line is found")
        tasks = draw_tasks(
            per_task["panels"], task_caption("run 0.", "counts", "Crosses.", ROUTERS), folder / "c.png", ROUTERS
        )
        legend = [t.get_text() for t in tasks.legends[0].get_texts()]
        check(set(colours_of.values()) <= colours(tasks) and not any("operating points" in label for label in legend),
              "the per-task figure draws all four routers' steps and, as ever, no operating points")  # fmt: skip
        check(all(sum(label.startswith(f"{r.label}, its steps") for label in legend) == 1 for r in ROUTERS)
              and has_overhead_line(tasks),
              "with each router once in its legend and the overhead line in its caption")  # fmt: skip


def has_overhead_line(fig: Figure) -> bool:
    """Whether a figure's text holds the overhead sentence, whatever the caption's line breaks."""
    return any(OVERHEAD_LINE in " ".join(t.get_text().split()) for t in fig.texts)


# ---------------------------------------------------------------------------------------------------------------
# Table v3's file


def table_v3_file(headline: dict, per_task: dict, table: dict, table_v2: dict) -> None:
    print("\n-- table v3's file: the operating points, comparisons and diagnostics (A15 items 14 and 15) --")
    check(table["split"] == headline["split"] and table["n"] == headline["n"] and table["run"] == headline["run"]
          and table["price_snapshot"] == headline["price_snapshot"] and table["bootstrap"] == headline["bootstrap"],
          "it is of the same split, items, prices and resamples as figures/pareto.json")  # fmt: skip
    check(set(UNITS_NEEDED) <= set(table["units"]), "every table in it has a units entry")
    check(table["comparisons"]["description"] == DESCRIPTIVE and DESCRIPTIVE in table["units"]["comparisons"],
          "the comparisons are described as descriptive, with no multiplicity correction")  # fmt: skip
    check(table["diagnostics"]["label"] == DIAGNOSTIC_LABEL and DIAGNOSTIC_LABEL in table["units"]["diagnostics"],
          f"the diagnostics are labeled \"{DIAGNOSTIC_LABEL}\"")  # fmt: skip
    check(list(table["operating_points"]) == [G, G_PLUS], "operating points for G and G+")
    for letter, name in JEV_SECTIONS.items():
        check(all(table["operating_points"][letter][mix] == headline["panels"][mix]["routers"][name]["operating_points"]
                  for mix in MIXES),
              f"{letter}'s operating points are the figure JSON's, number for number")  # fmt: skip
    pairs = [f"{a} less {b}" for a, b in PAIRS]
    check(list(table["comparisons"]["same_target"]) == pairs and f"{G} less {F_TRAIN}" not in pairs,
          "the comparisons are the six A15 item 14 names, first less second, and not G against F-train")  # fmt: skip
    points = {mix: {E: headline["panels"][mix]["routers"][E_SECTION]["operating_points"],
                    F: headline["panels"][mix]["routers"][F_SECTION]["operating_points"],
                    G: headline["panels"][mix]["routers"][NAMED_ROUTERS[G]]["operating_points"],
                    G_PLUS: headline["panels"][mix]["routers"][NAMED_ROUTERS[G_PLUS]]["operating_points"],
                    F_TRAIN: table_v2["diagnostics"]["operating_points"][F_TRAIN][mix]}
              for mix in MIXES}  # fmt: skip
    for pair in PAIRS:
        by_mix = table["comparisons"]["same_target"][f"{pair[0]} less {pair[1]}"]
        for mix in MIXES:
            for key in TARGETS:
                cell, first, second = by_mix[mix][key], points[mix][pair[0]][key], points[mix][pair[1]][key]
                where = f"{pair[0]} less {pair[1]}, {mix} {key}"
                if first["fell_back"] and second["fell_back"]:
                    check(cell["reading"] == "both fall back to A" and cell["cost_per_1k"] is None
                          and cell["quality"] is None, f"{where}: both fall back to A, no difference")  # fmt: skip
                    continue
                cost = first["cost_per_1k"]["value"] - second["cost_per_1k"]["value"]
                quality = first["quality"]["value"] - second["quality"]["value"]
                est_cost, est_quality = cell["cost_per_1k"], cell["quality"]
                check(not cell["reading"] and abs(est_cost["value"] - cost) < ROUNDING
                      and abs(est_quality["value"] - quality) < ROUNDING
                      and est_cost["low"] <= est_cost["value"] <= est_cost["high"]
                      and est_quality["low"] <= est_quality["value"] <= est_quality["high"]
                      and (cell["first_fell_back"], cell["second_fell_back"]) == (first["fell_back"], second["fell_back"]),
                      f"{where}: cost {cost:+.2f}, quality {100 * quality:+.1f} points: the operating points' "
                      f"difference, inside its interval, a side that fell back marked")  # fmt: skip

    for letter, name in JEV_SECTIONS.items():
        for mix in MIXES:
            panel = headline["panels"][mix]
            for key in TARGETS:
                point = panel["routers"][name]["operating_points"][key]
                matched = table["comparisons"]["matched_cost"][letter][mix][key]
                read = envelope_at(panel["envelope"], point["cost_per_1k"]["value"])
                check(matched == point["matched_cost"] and matched["low"] <= matched["difference"] <= matched["high"]
                      and (read is None) == (matched["envelope"] is None),
                      f"{letter}, {mix} {key}: the matched-cost row is the figure JSON's, its difference inside its interval")  # fmt: skip
        for task in DEV_TASKS:
            for mix in MIXES:
                for key in TARGETS:
                    row = table["comparisons"]["mcnemar"][letter][task][mix][key]
                    point = per_task["panels"][task]["routers"][name]["operating_points"][mix][key]
                    fell = headline["panels"][mix]["routers"][name]["operating_points"][key]["fell_back"]
                    check(row == point["mcnemar"] and (row is None) == fell
                          and (row is None or abs(row["p"] - mcnemar_p(int(row["b"]), int(row["c"]))) < 1e-12),
                          f"{letter}, {task}, {mix} {key}: McNemar's row is the per-task figure's, null where it falls back to A, "
                          f"p the exact binomial of b and c")  # fmt: skip

    concordance = table["diagnostics"]["within_task_concordance"]
    check(list(concordance) == [G, G_PLUS] and all(list(concordance[n]) == list(SCORED) for n in concordance),
          "within-task concordance for G and G+, per config")  # fmt: skip
    check(all(concordance[n][c]["pooled"]["low"] <= concordance[n][c]["pooled"]["value"]
              <= concordance[n][c]["pooled"]["high"] and 0.5 < concordance[n][c]["pooled"]["value"] < 1
              for n in concordance for c in SCORED), "each reads above chance, inside its interval")  # fmt: skip
    check(all(set(concordance[n][c]["per_task"]) == set(DEV_TASKS) for n in concordance for c in SCORED)
          and all((concordance[n][c]["per_task"][t]["warning"] is not None) == (t in ("ifeval", "mbppplus"))
                  for n in concordance for c in SCORED for t in DEV_TASKS),
          "concordance is given per task, with A7 item 2's warning on IFEval and MBPP+ only")  # fmt: skip
    gap = table["diagnostics"]["concordance_differences"][f"{G_PLUS} less {G}"]
    check(all(abs(gap[c]["pooled"]["value"] - (concordance[G_PLUS][c]["pooled"]["value"] - concordance[G][c]["pooled"]["value"]))
              < ROUNDING and gap[c]["pooled"]["low"] <= gap[c]["pooled"]["value"] <= gap[c]["pooled"]["high"] for c in SCORED),
          "G+ less G is the difference of their pooled concordance, inside its interval")  # fmt: skip
    quantiles = table["diagnostics"]["router_score_quantiles"]
    check(all(list(quantiles[n][c]) == ["0", "0.05", "0.25", "0.5", "0.75", "0.95", "1"]
              and list(quantiles[n][c].values()) == sorted(quantiles[n][c].values())
              and 0 <= quantiles[n][c]["0"] and quantiles[n][c]["1"] <= 1 for n in (G, G_PLUS) for c in SCORED),
          "each router's router-score quantiles, per config, in order within [0, 1]")  # fmt: skip
    latency = table["latency"]
    check(list(latency) == [G, G_PLUS]
          and all(latency[n]["calls"] + 1 + len(latency[n]["reconnects"]) == table["n"]
                  and latency[n]["median"] <= latency[n]["p90"] and set(latency[n]["per_task"]) == set(DEV_TASKS)
                  and sum(t["calls"] for t in latency[n]["per_task"].values()) == latency[n]["calls"]
                  for n in latency),
          "dev latency: every dev call counted once, as the first call, a reconnect or a timed call, the median under the p90")  # fmt: skip
    dates = table["dev_calls"]
    check(set(dates) == {"first_call_server_date", G, G_PLUS}
          and dates["first_call_server_date"] in (dates[G], dates[G_PLUS])
          and dates["first_call_server_date"].endswith(" GMT"),
          "the first dev call's server Date is the earlier of the two routers' first calls, in GMT")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# The README's numbers


README = PROJECT_ROOT / "README.md"


def readme_facts(headline: dict, table: dict, table_v2: dict) -> list[str]:
    """The README's sentences that quote a number about G or G+, as the committed files give it."""
    realistic = headline["panels"]["realistic"]
    a = realistic["configs"][FRONTIER]["cost_per_1k"]["value"]
    cost = {n: realistic["routers"][name]["operating_points"]["97%"]["cost_per_1k"]["value"]
            for n, name in JEV_SECTIONS.items()}  # fmt: skip
    haiku = {n: table["diagnostics"]["within_task_concordance"][n]["haiku"]["pooled"]["value"] for n in (G, G_PLUS)}
    f_haiku = table_v2["diagnostics"]["within_task_concordance"][F]["haiku"]["pooled"]["value"]
    excluding: dict[str, list[str]] = {}
    for n in (G, G_PLUS):
        excluding[n] = []
        for mix in MIXES:
            for key, row in table["comparisons"]["matched_cost"][n][mix].items():
                if row["low"] > 0 or row["high"] < 0:
                    excluding[n].append(f"{mix} mix, {key} target: {100 * row['difference']:+.1f} points, interval "
                                        f"{100 * row['low']:+.1f} to {100 * row['high']:+.1f}")  # fmt: skip
    where = f" ({'; '.join(excluding[G_PLUS])})" if excluding[G_PLUS] else ""
    return [
        f"At the 97% target on the realistic mix G costs ${cost[G]:.2f} per 1,000 requests and G+ ${cost[G_PLUS]:.2f}, "
        f"against A's ${a:.2f}, with Jev's overhead in both.",
        f"Within-task concordance on `haiku` is {haiku[G]:.3f} for G and {haiku[G_PLUS]:.3f} for G+, against F's "
        f"{f_haiku:.3f} (diagnostics, dev only).",
        f"At matched cost G's paired difference from the random-mixture envelope excludes 0 at {len(excluding[G])} of "
        f"its six operating points and G+'s at {len(excluding[G_PLUS])}{where}, a descriptive reading with no "
        f"multiplicity correction.",
    ]


def readme_checks(headline: dict, table: dict, table_v2: dict) -> None:
    print("\n-- the README's numbers about G and G+ --")
    text = " ".join(README.read_text().split())
    facts = readme_facts(headline, table, table_v2)
    for fact in facts:
        check(fact in text, f"the README says: {fact}")
    slip = re.sub(r"(\d)(?=[ ,.)]|$)", lambda m: str((int(m[1]) + 1) % 10), facts[0], count=1)
    check(
        slip != facts[0] and slip not in text,
        "known positive: a one-digit slip in the first sentence is not in the README",
    )


# ---------------------------------------------------------------------------------------------------------------
# On dev, where the stored records and the dev matrix are


def on_dev(headline: dict, per_task: dict, table: dict) -> None:
    from analysis import collect_jev, jev_comparisons
    from analysis import table_v3 as built
    from analysis.router_panels import router_part_of_mix_panel, router_part_of_task_panel, step_outcomes
    from analysis.rules_table import dev_table as rules_dev_table
    from analysis.simulator import always, load_mixes, simulate
    from analysis.trained_table import dev_table as trained_dev_table
    from routers.ladder import LADDER

    print("\n-- on dev: a rebuild, the stored records and the overhead (hook only) --")
    result = jev_comparisons.compute()
    check(json.loads(json.dumps(built.build(result))) == table, "table v3 equals a rebuild from the dev matrix and the stored records")  # fmt: skip
    dev, draws, mixes = result.dev, result.draws, load_mixes()
    fixed = {config: simulate(dev, always(dev, config)) for config in LADDER}
    steps = {name: step_outcomes(dev, result.tables[name], fixed[FRONTIER]) for name in (G, G_PLUS)}
    savings = {mix: draws.savings(dev, fixed[FRONTIER], mixes[mix]) for mix in PANELS}
    for letter, name in JEV_SECTIONS.items():
        walks = {mix: dict(by_share) for mix, by_share in result.walks[letter].items()}
        for mix in PANELS:
            rebuilt = router_part_of_mix_panel(draws, savings[mix], steps[letter], walks[mix], fixed, mixes[mix])
            check(json.loads(json.dumps(rebuilt)) == headline["panels"][mix]["routers"][name],
                  f"{mix} mix: {letter}'s steps and operating points in figures/pareto.json are what the package gives now")  # fmt: skip
        for task in DEV_TASKS:
            rebuilt = router_part_of_task_panel(draws, steps[letter], walks, task)
            check(json.loads(json.dumps(rebuilt)) == per_task["panels"][task]["routers"][name],
                  f"{task}: {letter}'s steps, operating points and McNemar rows in figures/pareto_tasks.json are current")  # fmt: skip
    check(set(json.loads(OPERATING_POINTS_FILE.read_text())) == {E_SECTION, F_SECTION, *JEV_SECTIONS.values()}
          and all(json.loads(json.dumps(build_section(result.walks[n]))) == read_section(NAMED_ROUTERS[n])
                  for n in (G, G_PLUS)),
          "the walks reproduce G's and G+'s sections of the operating-points file")  # fmt: skip

    dates = {}
    for letter, router in zip((G, G_PLUS), collect_jev.ROUTERS, strict=True):
        dates[letter] = str(collect_jev.read_records(router, "dev")[0]["server_date"])
    check(table["dev_calls"][G] == dates[G] and table["dev_calls"][G_PLUS] == dates[G_PLUS],
          "each router's first dev call's server Date in table v3 is the stored record's")  # fmt: skip

    overhead = {E: {e.overhead for e in rules_dev_table().items}, F: {e.overhead for e in trained_dev_table().items}}
    check(overhead[E] == {0.0} and overhead[F] == {0.0},
          "E's and F's dev tables charge no overhead on any request, as the caption says")  # fmt: skip
    check(all(min(e.overhead for e in result.tables[n].items) > 0 for n in (G, G_PLUS)),
          "G's and G+'s dev tables charge Jev's overhead on every request, as the caption says")  # fmt: skip


def main() -> int:
    headline, per_task = (json.loads(path.read_text()) for path in (FIGURE_DATA, TASK_FIGURE_DATA))
    table, table_v2 = json.loads(TABLE_V3.read_text()), json.loads((TABLE_V3.parent / "table_v2.json").read_text())
    sections = {letter: read_section(name) for letter, name in JEV_SECTIONS.items()}
    price = json.loads(SETTINGS_FILE.read_text())["jev"]["price_per_million_input_tokens"]
    if "--print-readme" in sys.argv[1:]:
        print("\n".join(readme_facts(headline, table, table_v2)))
        return 0
    if "--print" in sys.argv[1:]:
        print("\n".join(line for line in record_lines(headline, sections, table["dev_calls"], price)))
        return 0
    stage5_kept(headline, per_task)
    for letter, name in JEV_SECTIONS.items():
        jev_rows(headline, name, sections[letter])
    label_offsets(headline)
    drawn(headline, per_task)
    table_v3_file(headline, per_task, table, table_v2)
    record_checks(headline, sections, table["dev_calls"], price)
    readme_checks(headline, table, table_v2)
    if (PROJECT_ROOT / "results" / "dev_matrix.sqlite").exists() and (PROJECT_ROOT / "results" / "jev").exists():
        on_dev(headline, per_task, table)
    else:
        print("\n(results/ is not here: the rebuild from the dev matrix and the stored records is skipped)")
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
