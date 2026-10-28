"""Check the interactive Pareto explorer against chart v3's JSONs (PLAN_v4.md §12; CODING_STANDARDS.md, Numbers in
documents).

Run from the project root:  uv run python -m analysis.check_explorer

Reads the committed figure JSONs, the task mixes, the template and figures/pareto_explorer.html, so CI runs it. No
results matrix, never a test item.

- the committed page is the build from the committed JSONs, byte for byte (known positive: a JSON with one config's
  quality nudged by 1e-6 builds a different page), so a rebuilt chart with a stale page fails;
- the numbers the page carries are the JSONs', read straight from both files and not through the build: every view,
  every config's quality and cost as [value, low, high], every router's steps, and at each mix's targets each operating
  point's threshold, fallback, quality, cost, savings against A and matched-cost difference;
- the page's own words agree with the JSONs: the split and run in its footer, the kind of interval, the targets its
  caption names, the routers' labels;
- the template: one data placeholder, Plotly pinned to an exact version on jsDelivr, and its router list is chart
  v3's, by name and short name, in order.
"""

import copy
import json
import re
import sys

from analysis.chart_routers import ROUTERS
from analysis.explorer import EXPLORER, PLACEHOLDER, TEMPLATE, page
from analysis.pareto import FIGURE_DATA, PANELS, TASK_FIGURE_DATA, TASKS
from analysis.simulator import load_mixes

problems: list[str] = []
SPREAD = ("value", "low", "high")  # the order the page reads an estimate's three numbers in


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def embedded(html: str) -> dict | None:
    """The data the page carries: the JSON after `const DATA = `, to the end of that line."""
    found = re.search(r"^const DATA = (.*);$", html, re.MULTILINE)
    return json.loads(found[1].replace("<\\/", "</")) if found else None


def spread(estimate: dict) -> list[float]:
    return [estimate[key] for key in SPREAD]


def check_view(name: str, view: dict, source: dict, mix: bool) -> None:
    """One view's numbers against its panel in the figure JSON."""
    configs = {row["key"]: row for row in view["configs"]}
    check(list(configs) == list(source["configs"])
          and all(configs[c]["q"] == spread(row["quality"]) and configs[c]["c"] == spread(row["cost_per_1k"])
                  for c, row in source["configs"].items()),
          f"{name}: each config's quality and cost, with intervals")  # fmt: skip
    check(all(low <= value <= high for row in view["configs"] for value, low, high in (row["q"], row["c"])),
          f"{name}: each config's value sits inside its interval, so the page reads [value, low, high]")  # fmt: skip
    for router in ROUTERS:
        part, drawn = source["routers"][router.name], view["routers"][router.name]
        check(drawn["steps"] == [[s["cost_per_1k"], s["quality"], s["threshold"]] for s in part["steps"]],
              f"{name}: {router.short}'s {len(part['steps'])} steps, in walk order")  # fmt: skip
        if not mix:
            continue
        ops = {op["target"]: op for op in drawn["ops"]}
        for target, point in part["operating_points"].items():
            op, matched = ops.get(target, {}), point.get("matched_cost") or {}
            check(op.get("threshold") == point["threshold"] and op.get("fell_back") == point["fell_back"]
                  and op.get("q") == spread(point["quality"]) and op.get("c") == spread(point["cost_per_1k"])
                  and op.get("save_share") == (point.get("savings_vs_A") or {}).get("share")
                  and op.get("matched") == ([matched[k] for k in ("difference", "low", "high")] if matched else None),
                  f"{name}: {router.short} at {target}: threshold, fallback, quality, cost, savings, matched cost")  # fmt: skip


def main() -> int:
    head, tasks = (json.loads(path.read_text(encoding="utf-8")) for path in (FIGURE_DATA, TASK_FIGURE_DATA))
    mixes = load_mixes()
    committed = EXPLORER.read_text(encoding="utf-8") if EXPLORER.exists() else ""

    print("-- the committed page is the build from the committed JSONs --")
    check(page(head, tasks, mixes) == committed,
          "figures/pareto_explorer.html is the build, byte for byte (rerun analysis.explorer after analysis.pareto)")  # fmt: skip
    nudged = copy.deepcopy(head)
    nudged["panels"]["realistic"]["configs"]["haiku"]["quality"]["value"] += 1e-6
    check(page(nudged, tasks, mixes) != committed, "known positive: a JSON with one quality nudged by 1e-6 builds "
          "a different page")  # fmt: skip

    print("\n-- the page's numbers are the JSONs' --")
    data = embedded(committed)
    check(data is not None, "the page carries its data on one `const DATA = ...;` line")
    if data is None:
        print(f"\n{len(problems)} checks failed")
        return 1
    check(list(data["views"]) == [*PANELS, *TASKS], f"the page's views are the mixes and the tasks, in the figures' "
          f"order: {', '.join(data['views'])}")  # fmt: skip
    for name in PANELS:
        check_view(name, data["views"][name], head["panels"][name], mix=True)
    for name in TASKS:
        check_view(name, data["views"][name], tasks["panels"][name], mix=False)
        check(data["views"][name]["items"] == tasks["panels"][name]["items"], f"{name}: its dev item count")
    check(data["meta"] == {"n": head["n"], "prices": head["price_snapshot"], "resamples": head["bootstrap"]["resamples"]},
          f"the run's facts: {head['n']} dev requests, prices of {head['price_snapshot']}, "
          f"{head['bootstrap']['resamples']:,} resamples")  # fmt: skip

    print("\n-- the page's words agree with the JSONs --")
    template = TEMPLATE.read_text(encoding="utf-8")
    check(all(figure["split"] == "dev" and figure["run"] == 0 for figure in (head, tasks)) and "Dev split, run 0."
          in template, "the footer says dev, run 0, as both JSONs do")  # fmt: skip
    check(all(figure["bootstrap"]["interval"] == "percentile 95%" for figure in (head, tasks))
          and "Intervals are percentile 95%" in template
          and head["bootstrap"]["resampled"].startswith("source groups within each task")
          and "resamples of source groups within each task" in template,
          "the footer's interval kind and what is resampled are the JSONs'")  # fmt: skip
    targets = [list(head["panels"][mix]["targets"]) for mix in PANELS]
    named = f"the {', '.join(targets[0][:-1])} and {targets[0][-1]} targets"
    check(all(t == targets[0] for t in targets) and named in template, f"the caption names {named}, as both mixes "
          "hold them")  # fmt: skip
    check(data["labels"] == {router.name: router.label for router in ROUTERS},
          "the routers' labels are chart v3's (analysis/chart_routers.py)")  # fmt: skip

    print("\n-- the template --")
    check(template.count(PLACEHOLDER) == 1, f"the template has one data placeholder, {PLACEHOLDER}")
    scripts = re.findall(r'<script src="([^"]+)"', template)
    check(len(scripts) == 1 and re.fullmatch(r"https://cdn\.jsdelivr\.net/npm/plotly\.js-dist-min@\d+\.\d+\.\d+/"
          r"plotly\.min\.js", scripts[0]) is not None,
          f"it loads one script, Plotly at an exact version on jsDelivr: {scripts}")  # fmt: skip
    listed = re.findall(r'\{ key: "(\w+)", short: "([^"]+)"', template)
    check(listed == [(router.name, router.short) for router in ROUTERS],
          f"its router list is chart v3's, in order: {', '.join(short for _, short in listed)}")  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
