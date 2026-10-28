"""The interactive Pareto explorer: chart v3's numbers in one HTML page, to toggle, zoom and hover (PLAN_v4.md §12).

Run from the project root:  uv run python -m analysis.explorer

Reads figures/pareto.json and figures/pareto_tasks.json, keeps what the page draws, and writes
figures/pareto_explorer.html: `analysis/explorer_template.html` with those numbers embedded, so the page opens from the
file with no server. It loads Plotly from jsDelivr, so it needs a network connection to draw.

The page is a view of chart v3, not a figure of record: the PNGs and their JSONs are. Every number it shows is one the
JSONs hold, and the realistic mix's note is the headline figure's (`realistic_note`). Nothing reads the results matrix,
so it builds anywhere. Rerun it after `analysis.pareto`; `analysis/check_explorer.py` rebuilds the page and compares it
with the committed file, so a rebuilt chart with a stale page fails.

The routers keep chart v3's markers and line styles. E's colour is chart v3's purple and G's is near-black; F's and G+'s
are an orange and a green, so all four stay apart under simulated colour blindness on one panel, where the references
are grey.
"""

import json
import sys
from pathlib import Path

from analysis.chart_routers import ROUTERS
from analysis.pareto import FIGURE_DATA, FIGURES, LABELS, PANELS, TASK_FIGURE_DATA, TASKS, realistic_note
from analysis.simulator import load_mixes
from harness.client import PROJECT_ROOT

TEMPLATE = Path(__file__).with_name("explorer_template.html")
EXPLORER = FIGURES / "pareto_explorer.html"
PLACEHOLDER = "/*DATA*/null"  # the template's `const DATA = ...;`, valid JavaScript before the build fills it


def interval(estimate: dict) -> list[float]:
    """An estimate as the page reads it: [value, low, high]."""
    return [estimate["value"], estimate["low"], estimate["high"]]


def operating_point(target: str, point: dict) -> dict:
    """One operating point at one target: its threshold, fallback, estimates, savings and matched-cost difference."""
    savings, matched = point.get("savings_vs_A") or {}, point.get("matched_cost") or {}
    return {
        "target": target,
        "threshold": point["threshold"],
        "fell_back": point["fell_back"],
        "certified": point.get("steps_certified"),
        "tried": point.get("steps_tried"),
        "q": interval(point["quality"]),
        "c": interval(point["cost_per_1k"]),
        "save_share": savings.get("share"),
        "save_per_1k": savings.get("per_1k"),
        # A fallback to A has no matched-cost row.
        "matched": [matched["difference"], matched["low"], matched["high"]] if "difference" in matched else None,
    }


def panel(source: dict, mix: bool) -> dict:
    """One panel of a figure JSON as the page draws it; a mix's panel adds its targets and operating points."""
    envelope = source["envelope"].values() if isinstance(source["envelope"], dict) else source["envelope"]
    out: dict = {
        "configs": [
            {"key": config, "label": LABELS[config], "q": interval(row["quality"]), "c": interval(row["cost_per_1k"])}
            for config, row in source["configs"].items()
        ],
        "envelope": sorted([corner["cost_per_1k"], corner["quality"]] for corner in envelope),
        "oracle": [[corner["cost_per_1k"], corner["quality"]] for corner in source["oracle"]],
        "routers": {},
    }
    if mix:
        out["targets"] = [{"name": name, "q": target["quality"]} for name, target in source["targets"].items()]
    for router in ROUTERS:
        part = source["routers"][router.name]
        steps = [[s["cost_per_1k"], s["quality"], s["threshold"]] for s in part["steps"]]
        entry: dict = {"steps": steps}
        out["routers"][router.name] = entry
        if mix:
            entry["ops"] = [operating_point(target, part["operating_points"][target]) for target in source["targets"]]
    return out


def page_data(head: dict, tasks: dict, mixes: dict[str, dict[str, float]]) -> dict:
    """What the page embeds: the run's facts, the routers' labels, and one view per mix and per task."""
    views = {
        mix: {"title": f"{mix.capitalize()} mix", "kind": "mix", **panel(head["panels"][mix], mix=True), "note": None}
        for mix in PANELS
    }
    views["realistic"]["note"] = realistic_note(mixes["realistic"], tasks["panels"]["mbppplus"]["items"])
    for task, title in TASKS.items():
        source = tasks["panels"][task]
        views[task] = {"title": title, "kind": "task", "items": source["items"], "note": source["warning"],
                       **panel(source, mix=False)}  # fmt: skip
    return {
        "meta": {"n": head["n"], "prices": head["price_snapshot"], "resamples": head["bootstrap"]["resamples"]},
        "labels": {router.name: head["routers"][router.name]["label"] for router in ROUTERS},
        "views": views,
    }


def page(head: dict, tasks: dict, mixes: dict[str, dict[str, float]]) -> str:
    """The page: the template with the data in place of its placeholder."""
    # "</" escaped, so no string in the data can close the page's <script> element.
    blob = json.dumps(page_data(head, tasks, mixes), separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, blob, 1)


def main() -> int:
    head, tasks = (json.loads(path.read_text(encoding="utf-8")) for path in (FIGURE_DATA, TASK_FIGURE_DATA))
    html = page(head, tasks, load_mixes())
    EXPLORER.write_text(html, encoding="utf-8")
    print(f"wrote {EXPLORER.relative_to(PROJECT_ROOT)}: {len(html):,} characters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
