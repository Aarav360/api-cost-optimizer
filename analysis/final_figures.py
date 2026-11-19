"""A split's headline, per-task and router-started figures at the frozen operating points (PLAN_v4.md Amendment A19, items 8 and
13; A17 item 23; issue #77, #90).

Chart v4 (`analysis/pareto.py`, `analysis/routed_figure.py`) draws dev. Its lines are walks laid out whole, which re-derive
their steps from the split's own tables, so run on test they would choose test's steps. Test chooses nothing (A19 item 8), so
this module builds the same figures from what `analysis/offline_half.py` already evaluated at the operating-points file's
frozen points, and draws them with chart v4's own functions (`pareto.draw_headline`, `pareto.draw_tasks`,
`routed_figure.draw_routed`):

- **a cross is a frozen operating point**: each row of a policy's part of the tables, labeled with its target. A point the file
  says fell back is at A and labeled so;
- **a line is A and then the distinct frozen dials that route**, strictest first (a router's thresholds highest first, a pure
  cascade's cutoffs highest first), each simulated on the split once by `offline_half` and weighed here under each mix or task.
  Nothing is walked and no step is derived from the split's own scores. The lines are therefore short and say so in the captions;
  dev's lines show every step the walk tried, which is the record of how the points were chosen;
- **the free checks alone** are one point with intervals, as on dev: the cascade has no dial, so nothing is chosen by evaluating it;
- the files are `pareto_<split>.json`, `pareto_tasks_<split>.json`, `pareto_routed_<split>.json` and their PNGs, in the output
  folder, so chart v4's dev files are never touched.

`build(inputs, tables, lines=None)` takes the offline half's inputs after `build_tables` and the tables it built; `lines` replaces the
frozen lines with others (the dev reproduction, `analysis/check_final_figures_dev.py`, gives it dev's walked lines and compares the
result with the committed chart). Every policy's part is read from `tables["parts"]` by its key, so a policy added there is drawn
with no edit here; a part the figures need and the tables lack refuses (`FigureError`) before anything is drawn.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from analysis import cascade_charts, pareto, routed_figure
from analysis.cascade_charts import Line
from analysis.chart_routers import ROUTERS
from analysis.final_gates import GATE_TABLE
from analysis.gate_walk import pure_traces
from analysis.operating_eval import shared_fixed
from analysis.operating_points import OPERATING_POINTS_FILE, read_section, section_hash
from analysis.router_names import CASCADES, FREE_GATE, GATE_LABELS, HAIKU, cascade_section
from analysis.router_panels import config_mcnemar
from analysis.simulator import Outcome, load_mixes, simulate, task_totals, totals
from harness.client import PROJECT_ROOT
from routers.ladder import FRONTIER

if TYPE_CHECKING:
    from analysis.offline_half import Inputs

FREE_SECTION = cascade_section(HAIKU, FREE_GATE)  # the cascade with no dial, drawn as one point
NO_DIAL = float("-inf")  # sorts a missing dial last


class FigureError(ValueError):
    """The tables lack a part the figures need, or a line holds no step. Nothing is drawn."""


def names_of(split: str) -> dict[str, tuple[str, str]]:
    """Each figure's PNG and JSON file names for `split`: beside chart v4's, with the split in the name."""
    return {
        "headline": (f"pareto_{split}.png", f"pareto_{split}.json"),
        "tasks": (f"pareto_tasks_{split}.png", f"pareto_tasks_{split}.json"),
        "routed": (f"pareto_routed_{split}.png", f"pareto_routed_{split}.json"),
    }


@dataclass(frozen=True)
class Figures:
    """What `build` makes: the three figures' numbers (JSON-ready, as the files hold them) and what their drawing takes."""

    split: str
    headline: dict
    tasks: dict
    routed: dict
    notes: dict[str, str]
    captions: dict[str, str]


# ---- what the figures read ------------------------------------------------------------------------------------------


def policy_sections() -> list[str]:
    """The four routers and the fifteen cascades, in the figures' order."""
    return [*(router.name for router in ROUTERS), *CASCADES]


def required_parts(sections: list[str]) -> list[str]:
    """The part keys the figures read: each policy's part, and for a cascade its per-task part too (the part `<key>_tasks`)."""
    return [*sections, *(f"{key}_tasks" for key in CASCADES), "references"]


def check_parts(parts: Mapping[str, Any], sections: list[str]) -> None:
    if missing := [key for key in required_parts(sections) if key not in parts]:
        raise FigureError(f"the tables hold no part for {missing}; the figures draw every router and cascade")


def tasks_part(parts: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    """A policy's per-task rows: the router's own part holds them, a cascade's are a part of their own."""
    return parts[section if section not in CASCADES else f"{section}_tasks"]["tasks"]


def rows_like_table_v4(parts: Mapping[str, Any]) -> dict[str, Any]:
    """The cascades' rows in table v4's shape (`cascade_charts.points_of` reads them): by section, start, gate and the mixes' rows."""
    from analysis.router_names import parse_cascade

    return {"cascades": {key: {"start": parse_cascade(key)[0], "gate": parse_cascade(key)[1],
                               "operating_points": parts[key]["mixes"]} for key in CASCADES}}  # fmt: skip


# ---- the lines ------------------------------------------------------------------------------------------------------


def dial_order(dial: tuple[float | None, float | None]) -> tuple[float, float]:
    return (NO_DIAL if dial[0] is None else dial[0], NO_DIAL if dial[1] is None else dial[1])


def frozen_lines(inputs: "Inputs", sections: list[str]) -> dict[str, Line]:
    """Each policy's line on the split: A, then each distinct dial of its section's frozen points that routes, strictest first,
    with the outcomes `offline_half` evaluated there (`Fixed.evaluated`). The free checks' one step has no dial and is
    simulated from the gate-score table; nothing else is simulated here."""
    fixed = shared_fixed(inputs.cache, inputs.matrix)
    frontier = tuple(fixed[FRONTIER])
    lines: dict[str, Line] = {}
    for section in sections:
        dials: dict[tuple[float | None, float | None], list[Outcome]] = {}
        for point, outcomes in fixed.evaluated.items():
            if point.section == section and not point.fell_back:
                dials.setdefault((point.threshold, point.cutoff), outcomes)
        if section == FREE_SECTION:
            table = inputs.extras[GATE_TABLE].table
            dials[(None, None)] = simulate(inputs.matrix, pure_traces(table, FREE_GATE, None))
        lines[section] = Line(
            (
                (None, None, frontier),
                *((t, c, tuple(dials[(t, c)])) for t, c in sorted(dials, key=dial_order, reverse=True)),
            )
        )
    return lines


# ---- the panels -----------------------------------------------------------------------------------------------------


def router_mix_part(line: Line, mix: Mapping[str, float], rows: Mapping[str, Any]) -> dict[str, Any]:
    return {"steps": [{"threshold": t, **asdict(totals(list(o), dict(mix)))} for t, _, o in line.steps],
            "operating_points": dict(rows)}  # fmt: skip


def router_task_part(line: Line, task: str, rows: Mapping[str, Any]) -> dict[str, Any]:
    return {"steps": [{"threshold": t, **asdict(task_totals(list(o))[task])} for t, _, o in line.steps],
            "operating_points": dict(rows)}  # fmt: skip


def build(inputs: "Inputs", tables: Mapping[str, Any], lines: Mapping[str, Line] | None = None) -> Figures:
    """The split's three figures' numbers from its tables (`offline_half.build_tables(inputs)`'s result) and the outcomes the
    build kept. `lines` replaces the frozen lines (a dev reproduction passes dev's walked ones)."""
    split, parts, mixes = inputs.split, tables["parts"], load_mixes()
    sections = policy_sections()
    check_parts(parts, sections)
    fixed = shared_fixed(inputs.cache, inputs.matrix)
    lines = lines if lines is not None else frozen_lines(inputs, sections)
    v4 = rows_like_table_v4(parts)
    refs = parts["references"]
    panels: dict[str, dict] = {}
    for mix in pareto.PANELS:
        routers = {
            router.name: router_mix_part(lines[router.name], mixes[mix], parts[router.name]["mixes"][mix])
            for router in ROUTERS
        }
        panels[mix] = {**refs["mixes"][mix], "routers": routers,
                       "cascades": cascade_charts.headline_parts(lines, inputs.draws, mix, mixes[mix], v4)}  # fmt: skip
    per_task: dict[str, dict] = {}
    for task in pareto.TASKS:
        routers = {router.name: router_task_part(lines[router.name], task, tasks_part(parts, router.name)[task])
                   for router in ROUTERS}  # fmt: skip
        per_task[task] = {**refs["tasks"][task], "mcnemar_vs_A": config_mcnemar(fixed, task), "routers": routers,
                          "cascades": cascade_charts.task_parts(lines, task)}  # fmt: skip
    routed = {mix: cascade_charts.routed_panel(lines, panels[mix], mix, mixes[mix], v4) for mix in pareto.PANELS}

    basis = pareto.basis_line(len(inputs.matrix.items), tables["price_snapshot"])
    counts = ", ".join(f"{pareto.TASKS[task]} {panel['items']}" for task, panel in per_task.items())
    meta = {"built_by": "analysis/final_figures.py", "split": split, "run": 0, "n": len(inputs.matrix.items),
            "price_snapshot": tables["price_snapshot"],
            "bootstrap": {**tables["bootstrap"], "resampled": RESAMPLED}, "operating_points": FROZEN_BASIS}  # fmt: skip
    router_meta = {router.name: {"label": router.label, **file_facts(router.name)} for router in ROUTERS}
    cascade_meta = cascade_charts.cascade_meta(v4)
    names = names_of(split)
    units = units_of(split)
    note = pareto.realistic_note(mixes["realistic"], per_task["mbppplus"]["items"]).replace(
        " dev items", f" {split} items"
    )
    captions = {
        "headline": headline_caption(split, basis),
        "tasks": task_caption(split, basis, counts),
        "routed": routed_caption(split, basis),
    }
    # The text printed under each figure (and over the realistic panel) is in its JSON too, so the words are held with the numbers.
    headline = {"figure": names["headline"][0], **meta, "caption": captions["headline"], "note": note,
                "units": units["headline"], "routers": router_meta, "cascades": cascade_meta,
                "cascade_units": units["cascades"], "panels": panels}  # fmt: skip
    tasks = {"figure": names["tasks"][0], **meta, "caption": captions["tasks"], "units": units["tasks"],
             "cascades": cascade_meta, "cascade_units": units["cascades"], "panels": per_task}  # fmt: skip
    routed_data = {"figure": names["routed"][0], **meta, "caption": captions["routed"], "units": units["cascades"],
                   "routers": router_meta, "cascades": cascade_meta, "panels": routed}  # fmt: skip
    as_files = [json.loads(json.dumps(d)) for d in (headline, tasks, routed_data)]  # as the JSON files will hold them
    return Figures(split, as_files[0], as_files[1], as_files[2], {"realistic": note}, captions)


def file_facts(section: str) -> dict[str, str]:
    """A policy's entry's pointer to the operating-points file and the hash of its section, which ties the figure to the frozen
    points it marks."""
    return {"operating_points_file": str(OPERATING_POINTS_FILE.relative_to(PROJECT_ROOT)),
            "operating_points_sha256": section_hash(read_section(section))}  # fmt: skip


# ---- words ----------------------------------------------------------------------------------------------------------

FROZEN_BASIS = "frozen"
RESAMPLED = "source groups within each task, the mix weights applied afterwards"
FROZEN_LINE = (
    "Each cross is a frozen operating point: the threshold or cutoff the walk on dev chose before test was touched "
    "(routers/operating_points.json), evaluated here and not walked, labeled with its target. A step line joins A to the "
    "distinct frozen dials that route, strictest first, each evaluated on the split once; it is not a walk and shows no step "
    "the walk did not choose, so it is shorter than dev's, which shows every step the walk tried."
)


def headline_caption(split: str, basis: str) -> str:
    free, jev, sonnet = (CASCADES[cascade_section(HAIKU, gate)] for gate in GATE_LABELS)
    names = [f"{router.short}'s" for router in ROUTERS]
    whose = f"{', '.join(names[:-1])} and {names[-1]}"
    return (f"{split.capitalize()} split, {basis}. {pareto.CROSSES_LINE} Differences between policies are judged by the paired "
            f"intervals in the tables, not by whether crosses overlap. {FROZEN_LINE} {whose} crosses mark their operating "
            f"points; their step lines are in the router-started figure. {free} is one point with 95% intervals (it has no dial); "
            f"{jev}'s and {sonnet}'s steps are joined by straight lines in linear cost from A (a frozen point that fell back to A "
            f"has its cross at A). {pareto.OVERHEAD_LINE} {pareto.GATE_COSTS_LINE}")  # fmt: skip


def task_caption(split: str, basis: str, counts: str) -> str:
    return (f"{split.capitalize()} split, {basis} ({counts}). Each task on its own, with its own envelope and oracle. "
            f"{pareto.CROSSES_LINE} No quality targets: verdicts are given for the two mixes only (Amendment A7). Each policy's "
            f"line joins A to its frozen dials that route, strictest first, evaluated on the task; there are no operating "
            f"points here. {pareto.OVERHEAD_LINE} {pareto.GATE_COSTS_LINE}")  # fmt: skip


def routed_caption(split: str, basis: str) -> str:
    return (f"{split.capitalize()} split, {basis}. One panel per router and mix. Each panel draws the router alone and its three "
            f"cascades (the free checks, the Jev gate, the Sonnet gate) on the router's frozen thresholds, joined by straight "
            f"lines in linear cost from A, with a cross at each operating point (one per quality target: 100%, 99% and 97% of "
            f"A's quality, the dotted lines). {pareto.CROSSES_LINE} {FROZEN_LINE} A cascade that falls back to A has its cross at "
            f"A. Where two lines coincide the later shows: the free checks' line is drawn first, wider and fainter, and the "
            f"router's own on top. A router-started cascade's steps are its router's frozen thresholds at the gate's fixed "
            f"cutoff (from train), and its costs include the router's overhead where it has one (G and G+: Jev's) and every "
            f"gate call made. Differences between policies are judged by the paired intervals in the tables, not by whether "
            f"crosses overlap.")  # fmt: skip


def units_of(split: str) -> dict[str, dict[str, str]]:
    """What each key of the figure JSONs means on `split`: chart v4's units with its walk-only keys left out and its words about
    dev made about `split`, and the frozen points' own."""
    estimate = (f"its value on {split}, the point plotted, with its percentile 95% interval (low, high), the cross's arm, and its "
                f"standard error, the standard deviation over the resamples")  # fmt: skip
    base = {
        "quality": "share of items answered correctly, 0 to 1, mix-weighted in the headline figure",
        "cost_per_1k": "dollars per 1,000 requests, at list price, mix-weighted in the headline figure",
        "configs": f"each config's quality and cost_per_1k: {estimate}",
        "envelope": "the envelope's corners, cheapest first, joined by straight lines in linear cost",
        "oracle": f"the oracle's corners: {pareto.START}, then the point after each upgrade, in the order taken, joined by "
                  f"straight lines in linear cost",
        "oracle_max_quality": "the quality at the oracle's last corner",
    }  # fmt: skip
    policies = {
        "routers": "the routers on the chart, keyed by their name in the operating-points file; E is `rules`, the rules router",
        "steps": f"A (threshold null) first, then each distinct frozen threshold that routes, from the highest down, each with its "
        f"quality and cost_per_1k on {split} (mix-weighted in the headline figure), joined by straight lines in linear cost: "
        f"the frozen points' dials evaluated on {split}, never a walk",
        "operating_points": f"the policy's frozen operating points, keyed by quality target (the headline figure's panel, one mix) or "
        f"by mix and then quality target (the per-task figure): the dials the operating-points file holds, "
        f"evaluated on {split}, whose quality and cost_per_1k are each " + estimate,
        "threshold": "the frozen point's threshold, null for a fallback to A",
        "fell_back": "the operating-points file's flag: the point is A itself (no router call, no overhead, captured fraction 0)",
        "target": f"the quality target on {split}: its share of A's quality under the mix",
        "captured": "the point's captured fraction of the oracle's savings at its target (§7), with its percentile 95% interval "
        "over every resample (null being n/a), and met, the share of resamples in which it meets the target",
        "routing": "the share of requests answered by each config, weighted like quality and cost",
        "matched_cost": "the envelope's quality at the point's cost and the point's paired quality difference from it, with its "
        "percentile 95% interval (low, high; null being n/a), and unread (Amendment A9, items 12 and 15)",
        "label": "the router's name on the chart",
        "operating_points_file": "the file holding the router's section: the frozen points and the walk that chose them",
        "operating_points_sha256": "the sha256 of the router's section of the operating-points file, written with sorted keys",
        "mcnemar": "McNemar's test of the row against A within the task: b, the items only the row gets right; c, the items only "
        "A gets right; p, exact and two-sided, descriptive only. Null (shown as —) for a point that falls back to A",
        "mcnemar_vs_A": "the same for each fixed config against A; A's own row is null",
    }
    cascades = {
        "cascades": "the fifteen cascades, keyed by section key: label, start (the pure cascades start at haiku, the others at a "
        "router), gate, the operating-points file that holds the section and the sha256 of the section",
        "panel_cascades": f"a panel's `cascades`: per cascade its label, gate and steps from A, then each distinct frozen dial that "
        f"routes, strictest first, with its quality and cost_per_1k on {split}, joined by straight lines in linear cost",
        "point": f"the free-checks-only cascade's one step as a point with a cross: its quality and cost_per_1k on {split}, each its "
        f"value with its percentile 95% interval (low, high) and standard error; the cascade has no dial",
        "operating_points": "the frozen points' rows, keyed by quality target: threshold (the router's), cutoff (the gate's), "
        "fell_back, quality and cost_per_1k with intervals, difference from A, savings against A, captured "
        "fraction, routing mix, escalation rate and the matched-cost comparison",
        "routed_panels": "the router-started figure's panels, one per mix: A, the envelope's corners, the targets, and per router four "
        "lines on the router's frozen thresholds: the router alone and its three cascades, each with steps and "
        "operating points",
    }
    headline = (
        base
        | policies
        | {
            "savings_vs_A": "A's cost per 1k less the config's, the oracle's or the point's (per_1k), and that as a share of A's (share)",
            "difference_from_A": "quality less A's, paired: the same resamples weigh both; its standard_error is the safety margin",
            "failed_by_every_config": f"the {split} items no config gets right",
            "targets": "each quality target, keyed by its share of A's (opus's) quality under the mix: its quality, the oracle's cost there "
            "(null: infeasible), savings against A and routing mix, and the envelope's cost there and captured fraction",
        }
    )
    tasks = base | policies | {"items": f"the task's {split} items",
                               "warning": "Amendment A7 item 2's warning on the task's per-task results, or null"}  # fmt: skip
    return {"headline": headline, "tasks": tasks, "cascades": cascades}


# ---- drawing and writing --------------------------------------------------------------------------------------------


def draw(figures: Figures, folder: Path) -> list[Path]:
    """Draw the three figures into `folder` with chart v4's own functions, from the numbers as the JSON files hold them."""
    names = names_of(figures.split)
    paths = {kind: folder / png for kind, (png, _) in names.items()}
    pareto.draw_headline(figures.headline["panels"], figures.notes, figures.captions["headline"], paths["headline"])
    pareto.draw_tasks(figures.tasks["panels"], figures.captions["tasks"], paths["tasks"])
    routed_figure.draw_routed(figures.routed["panels"], figures.captions["routed"], paths["routed"])
    return list(paths.values())


def write(figures: Figures, folder: Path) -> list[Path]:
    """Write the three JSON files and draw the three PNGs into `folder`; returns every path."""
    folder.mkdir(parents=True, exist_ok=True)
    names = names_of(figures.split)
    written = []
    for kind, data in (("headline", figures.headline), ("tasks", figures.tasks), ("routed", figures.routed)):
        path = folder / names[kind][1]
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return [*written, *draw(figures, folder)]


def steps_of(figures: Figures, mix: str, section: str) -> list[dict]:
    """A policy's line on a mix's panel as the figure JSONs hold it: a router's or a pure cascade's in the headline panel, a
    router-started cascade's in its router's router-started panel."""
    from analysis.router_names import parse_cascade

    panel = figures.headline["panels"][mix]
    if section in panel["routers"]:
        return list(panel["routers"][section]["steps"])
    if section in panel["cascades"]:
        return list(panel["cascades"][section]["steps"])
    start = parse_cascade(section)[0]
    return list(figures.routed["panels"][mix]["routers"][start]["lines"][section]["steps"])
