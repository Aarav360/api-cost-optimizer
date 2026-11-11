"""The interactive Pareto explorer: chart v4's numbers in one HTML page, to toggle, zoom and hover (PLAN_v4.md §12;
Amendment A17, item 23).

Run from the project root:  uv run python -m analysis.explorer

Reads figures/pareto.json, figures/pareto_tasks.json and figures/pareto_routed.json, keeps what the page draws, and writes
figures/pareto_explorer.html: `analysis/explorer_template.html` with those numbers embedded, so the page opens from the
file with no server. It loads Plotly from jsDelivr, so it needs a network connection to draw.

The page is a view of chart v4, not a figure of record: the PNGs and their JSONs are. Every number it shows is one the
JSONs hold, and the realistic mix's note is the headline figure's (`realistic_note`). Each of the four routers and each of
the fifteen cascades is a series to switch on and off: the routers and the three pure cascades are on at first, and the
twelve router-started cascades (whose lines are in the router-started figure's JSON) are off. A task view holds the routers
and the pure cascades only, which is what the per-task figure draws. Nothing reads the results matrix,
so it builds anywhere. Rerun it after `analysis.pareto`; `analysis/check_explorer.py` rebuilds the page and compares it
with the committed file, so a rebuilt chart with a stale page fails.

The routers keep chart v3's markers and line styles. E's colour is chart v3's purple and G's is near-black; F's and G+'s
are an orange and a green, so all four stay apart under simulated colour blindness on one panel, where the references
are grey. A cascade takes its gate's colour and line from `CASCADE_STYLES`, as on the charts: a pure cascade its gate's
marker too, a router-started one its router's markers, so the gate reads from the colour and the line and the start from
the marker.
"""

import json
import sys
from pathlib import Path

from analysis.chart_routers import CASCADE_STYLES, ROUTERS, LineStyle
from analysis.pareto import FIGURE_DATA, FIGURES, LABELS, PANELS, TASK_FIGURE_DATA, TASKS, realistic_note
from analysis.routed_figure import ROUTED_FIGURE_DATA
from analysis.router_names import (
    CASCADE_STARTS,
    CASCADES,
    G_PLUS,
    GATE_LABELS,
    HAIKU,
    JEV_GATE,
    LETTER_OF,
    NAMED_ROUTERS,
    G,
    cascade_section,
)
from analysis.simulator import load_mixes
from harness.client import PROJECT_ROOT

TEMPLATE = Path(__file__).with_name("explorer_template.html")
EXPLORER = FIGURES / "pareto_explorer.html"
PLACEHOLDER = "/*DATA*/null"  # the template's `const DATA = ...;`, valid JavaScript before the build fills it


# Plotly's names for the chart's matplotlib markers.
MARKERS = {"s": "square", "D": "diamond", "^": "triangle-up", "v": "triangle-down", "o": "circle", "P": "cross", "X": "x",
           "p": "pentagon", "h": "hexagon", "H": "hexagon2", "<": "triangle-left", ">": "triangle-right", "*": "star",
           "8": "octagon"}  # fmt: skip


def dash_of(style: LineStyle) -> str:
    """A matplotlib line style as a Plotly dash: a name as it is, a pattern in points as pixel lengths."""
    if isinstance(style, str):
        return {"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}[style]
    return ",".join(f"{2 * n:g}px" for n in style[1])


def cascade_series() -> list[dict]:
    """The fifteen cascades as the page lists them, in item 8's order: label, start, gate, and the style each draws in.

    A cascade has its gate's colour and line (`CASCADE_STYLES`). A pure cascade has its gate's markers; a router-started
    one has its router's, which says where it started."""
    routers = {LETTER_OF[router.name]: router for router in ROUTERS}
    found = []
    for start, letter in CASCADE_STARTS.items():
        for gate in GATE_LABELS:
            style = CASCADE_STYLES[gate]
            marked = style if start == HAIKU else routers[letter].style
            found.append({
                "key": cascade_section(start, gate), "label": CASCADES[cascade_section(start, gate)],
                "start": letter, "gate": GATE_LABELS[gate], "pure": start == HAIKU, "color": style.color,
                "dash": dash_of(style.linestyle), "step": MARKERS[marked.marker], "point": MARKERS[marked.point_marker],
            })  # fmt: skip
    return found


def presets(cascades: list[dict]) -> list[dict]:
    """The page's presets: a set of series each switches on, the rest off."""
    routers = [router.name for router in ROUTERS]
    jevs = [c for c in cascades if c["gate"] == GATE_LABELS[JEV_GATE]]
    pure_jev = next(c for c in jevs if c["pure"])
    sets = [
        ("Routers", routers),
        ("Pure cascades", [c["key"] for c in cascades if c["pure"]]),
        (f"{pure_jev['label']} against the Jev routers", [pure_jev["key"], NAMED_ROUTERS[G], NAMED_ROUTERS[G_PLUS]]),
    ]
    for letter in (G, G_PLUS):
        sets.append(
            (
                f"{letter} and its cascades",
                [NAMED_ROUTERS[letter], *(c["key"] for c in cascades if c["start"] == letter)],
            )
        )
    sets += [("The Jev gate on every start", [c["key"] for c in jevs]), ("All off", [])]
    return [{"label": label, "on": on} for label, on in sets]


def interval(estimate: dict) -> list[float]:
    """An estimate as the page reads it: [value, low, high]."""
    return [estimate["value"], estimate["low"], estimate["high"]]


def operating_point(target: str, point: dict) -> dict:
    """One operating point at one target: its threshold, fallback, estimates, savings and matched-cost difference."""
    savings, matched = point.get("savings_vs_A") or {}, point.get("matched_cost") or {}
    return {
        "target": target,
        "threshold": point["threshold"],
        "cutoff": point.get("cutoff"),
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


def cascade_entry(part: dict, targets: list[str] | None) -> dict:
    """One cascade's series on a panel: its steps as [cost, quality, threshold, cutoff], its operating points on a mix's
    panel and, for the free checks alone, its one point with its intervals."""
    entry: dict = {"steps": [[s["cost_per_1k"], s["quality"], s["threshold"], s["cutoff"]] for s in part["steps"]]}
    if targets is not None:
        entry["ops"] = [operating_point(target, part["operating_points"][target]) for target in targets]
    if "point" in part:
        entry["point"] = {"q": interval(part["point"]["quality"]), "c": interval(part["point"]["cost_per_1k"])}
    return entry


def panel(source: dict, mix: bool, routed: dict | None = None) -> dict:
    """One panel of a figure JSON as the page draws it; a mix's panel adds its targets and operating points, and the
    router-started cascades' lines from the router-started figure's panel `routed`."""
    envelope = source["envelope"].values() if isinstance(source["envelope"], dict) else source["envelope"]
    targets = list(source["targets"]) if mix else None
    out: dict = {
        "configs": [
            {"key": config, "label": LABELS[config], "q": interval(row["quality"]), "c": interval(row["cost_per_1k"])}
            for config, row in source["configs"].items()
        ],
        "envelope": sorted([corner["cost_per_1k"], corner["quality"]] for corner in envelope),
        "oracle": [[corner["cost_per_1k"], corner["quality"]] for corner in source["oracle"]],
        "routers": {},
        "cascades": {},
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
    for key, part in source["cascades"].items():
        out["cascades"][key] = cascade_entry(part, targets)
    if routed is not None:
        for router in routed["routers"].values():
            for key, line in router["lines"].items():
                if key in CASCADES:
                    out["cascades"][key] = cascade_entry(line, targets)
    return out


def page_data(head: dict, tasks: dict, routed: dict, mixes: dict[str, dict[str, float]]) -> dict:
    """What the page embeds: the run's facts, the routers' labels, the fifteen cascades and the presets, and one view per
    mix and per task."""
    views = {
        mix: {"title": f"{mix.capitalize()} mix", "kind": "mix",
              **panel(head["panels"][mix], mix=True, routed=routed["panels"][mix]), "note": None}
        for mix in PANELS
    }  # fmt: skip
    views["realistic"]["note"] = realistic_note(mixes["realistic"], tasks["panels"]["mbppplus"]["items"])
    for task, title in TASKS.items():
        source = tasks["panels"][task]
        views[task] = {"title": title, "kind": "task", "items": source["items"], "note": source["warning"],
                       **panel(source, mix=False)}  # fmt: skip
    cascades = cascade_series()
    return {
        "meta": {"n": head["n"], "prices": head["price_snapshot"], "resamples": head["bootstrap"]["resamples"]},
        "labels": {router.name: head["routers"][router.name]["label"] for router in ROUTERS},
        "cascades": cascades,
        "presets": presets(cascades),
        "views": views,
    }


def page(head: dict, tasks: dict, routed: dict, mixes: dict[str, dict[str, float]]) -> str:
    """The page: the template with the data in place of its placeholder."""
    # "</" escaped, so no string in the data can close the page's <script> element.
    blob = json.dumps(page_data(head, tasks, routed, mixes), separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, blob, 1)


def main() -> int:
    head, tasks, routed = (json.loads(path.read_text(encoding="utf-8"))
                           for path in (FIGURE_DATA, TASK_FIGURE_DATA, ROUTED_FIGURE_DATA))  # fmt: skip
    html = page(head, tasks, routed, load_mixes())
    EXPLORER.write_text(html, encoding="utf-8")
    print(f"wrote {EXPLORER.relative_to(PROJECT_ROOT)}: {len(html):,} characters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
