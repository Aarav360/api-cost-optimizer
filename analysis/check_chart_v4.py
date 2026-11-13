"""Check chart v4: the cascades' step lines, the router-started figure, the calibration figure (PLAN_v4.md §12; Amendment
A17, items 21 and 23; issue #72).

Run from the project root:  uv run python -m analysis.check_chart_v4

Reads the committed figure JSONs, `figures/table_v4.json` and `routers/operating_points.json`, so CI runs it. The last
section rebuilds the figures' numbers from the dev matrix and the stored gate records and runs only where `results/` holds
them. Never a test item, no gate and no API call.

- the cascade lines on a hand-worked table: A first, each cutoff in walk order, the numbers worked by hand; a
  router-started line on the router's own steps at the gate's fixed cutoff; the free checks alone as one point with a
  cross;
- the committed JSONs: every cascade in the headline, per-task and router-started figures, steps starting at A, each cross
  on a step of its line, operating points equal to table v4's rows and to the operating-points file's sections;
- the figures drawn from the committed numbers: every series once in the legend, in its gate's colour, the router-started
  figure's four lines per panel, the headline without the routers' step lines;
- the headline drawn as `analysis.pareto` draws it, caption and note included: no router's or cascade's label overlaps
  another label or an error bar (known positive: a label moved onto its own cross is found);
- the calibration figure: its bins, its hollow train markers and the caption that says why a cell is empty;
- the Sonnet gate's outline (Amendment A18, item 7): drawn from the committed numbers, every line, marker, cross and label
  in the lavender carries it, in the legends too, and nothing in another colour carries any (known positive: a lavender
  without its outline and a Jev gate with one are both found); in the committed PNGs most lavender pixels sit beside a
  dark one and the Jev gate's pale green's mostly don't (known positive: a lavender line drawn without the outline); in
  the committed explorer every Sonnet-gate cascade carries the outline and no other series does, and, where Node is
  installed, the page's own script gives each lavender line an outline trace beneath it and each lavender marker an
  outline edge, on every view, and gives no other series either;
- on dev (hook only): a rebuild of every figure JSON equals the committed one.
"""

import copy
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.lines import Line2D
from matplotlib.patheffects import Stroke
from matplotlib.text import Annotation, Text

from analysis import cascade_charts as cch
from analysis.bootstrap import resample
from analysis.calibration_figure import (
    CALIBRATION_DATA,
    caption_of,
    draw_calibration,
    empty_cells,
)
from analysis.chart_routers import CASCADE_STYLES, ROUTERS
from analysis.check_chart_routers import APART, NOTES, gap
from analysis.check_explorer import embedded
from analysis.check_table_v4 import TINY, tiny_table
from analysis.explorer import EXPLORER
from analysis.gate_table import PAID_GATES
from analysis.operating_points import read_section, section_hash
from analysis.pareto import (
    CROSSES_LINE,
    FIGURE,
    FIGURE_DATA,
    PANELS,
    SERIES_1,
    SERIES_2,
    SURFACE,
    TASK_FIGURE,
    TASK_FIGURE_DATA,
    TASKS,
    basis_line,
    cascade_point_label,
    cascade_points_label,
    cascade_steps_label,
    draw_headline,
    draw_tasks,
    headline_caption,
    realistic_note,
    task_caption,
)
from analysis.references import ROUNDING
from analysis.routed_figure import ROUTED_FIGURE, ROUTED_FIGURE_DATA, draw_routed, routed_caption
from analysis.router_names import (
    CASCADES,
    FREE_GATE,
    GATE_LABELS,
    HAIKU,
    JEV_GATE,
    LABELS,
    LETTER_OF,
    NAMED_ROUTERS,
    SONNET_GATE,
    cascade_section,
)
from analysis.simulator import always, load_mixes, simulate
from analysis.table_v4 import TABLE_V4
from harness.client import PROJECT_ROOT
from routers.ladder import FRONTIER, LADDER

problems: list[str] = []
MIX = {"t1": 0.5, "t2": 0.5}


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def near(a: float | None, b: float, tolerance: float = 1e-9) -> bool:
    return a is not None and abs(a - b) < tolerance


def line_checks() -> None:
    print("-- the cascade lines on a hand-worked table --")
    table = tiny_table()
    frontier = simulate(TINY, always(TINY, "opus"))
    line = cch.pure_line(TINY, table, JEV_GATE, frontier)
    rows = cch.mix_steps(line, MIX)
    # Under the mix t1 0.5, t2 0.5 each t1 item weighs 1/6 and y1 weighs 1/2. A (opus) is right on x1, x2, y1 and wrong on
    # x3: quality 5/6, cost 1000 x (3 x 1/6 x .004 + 1/2 x .004) = $4.00. haiku/Jev's Jev scores for haiku's passing
    # answers are x1 .9, x3 .8, y1 .4, so its cutoffs are .9, .8, .4. x2 fails a free check at once and pays haiku and opus
    # ($.005) but no gate; every other haiku answer pays haiku $.001 and the gate's $.0001.
    # At .9 only x1 stays (right): x3 and y1 pay opus too ($.0051 each), x3's opus answer being wrong. Quality 5/6, cost
    # 1000 x (1/6 x (.0011 + .005 + .0051) + 1/2 x .0051) = $4.41667.
    # At .8 x3 stays too (right, $.0011): quality 1, cost 1000 x (1/6 x (.0011 + .005 + .0011) + 1/2 x .0051) = $3.75.
    # At .4 y1 stays: quality 1, cost 1000 x (1/6 x (.0011 + .005 + .0011) + 1/2 x .0011) = $1.75.
    wanted = [(None, 5 / 6, 4.0), (0.9, 5 / 6, 4.416667), (0.8, 1.0, 3.75), (0.4, 1.0, 1.75)]
    check(
        [r["cutoff"] for r in rows] == [w[0] for w in wanted] and all(r["threshold"] is None for r in rows),
        "A first, then the cutoffs strictest first, a pure cascade having a cutoff and no threshold",
    )
    check(
        all(
            near(r["quality"], q) and near(r["cost_per_1k"], c, 1e-4) for r, (_, q, c) in zip(rows, wanted, strict=True)
        ),
        "each step's quality and cost per 1,000 are the hand-worked ones",
    )
    check(
        near(rows[0]["quality"], 5 / 6) and near(rows[0]["cost_per_1k"], 4.0),
        "the first step is A: always opus, which pays no gate",
    )
    draws = resample(TINY)
    free = cch.pure_line(TINY, table, FREE_GATE, frontier)
    check(len(free.steps) == 2, "haiku/free has A and one step")
    part = cch.pure_part_of_mix_panel("k", "label", FREE_GATE, free, draws, MIX, {})
    # haiku alone with the free checks: x2 is rejected and climbs to opus ($.005, right); x1, x3, y1 stay on haiku, all
    # right with no gate cost. Quality 1, cost 1000 x (1/6 x (.001 + .005 + .001) + 1/2 x .001) = $1.66667.
    check(
        near(part["point"]["quality"]["value"], 1.0)
        and near(part["point"]["cost_per_1k"]["value"], 1.666667, 1e-4)
        and part["point"]["quality"]["low"] <= part["point"]["quality"]["value"] <= part["point"]["quality"]["high"],
        "the free checks alone are one point with a cross: its quality and cost with their 95% intervals",
    )
    check(
        "point" not in cch.pure_part_of_mix_panel("k", "label", JEV_GATE, line, draws, MIX, {}),
        "a pure cascade with a gate is a curve and has no single point",
    )
    task = cch.task_steps(line, "t2")
    # Within t2 alone (y1): A right at $4.00; at .9 y1 is rejected and pays $.0051 (right); at .4 it stays at $.0011.
    check(
        near(task[0]["cost_per_1k"], 4.0)
        and near(task[1]["cost_per_1k"], 5.1)
        and near(task[3]["cost_per_1k"], 1.1)
        and all(near(s["quality"], 1.0) for s in task),
        "within a task the line is that task's own totals",
    )


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def on_line(point: dict, steps: list[dict]) -> bool:
    """Whether an operating point sits on a step of its line: same dial values, same quality and cost."""
    return any(
        s["threshold"] == point["threshold"] and s["cutoff"] == point["cutoff"]
        and abs(s["quality"] - point["quality"]["value"]) < ROUNDING
        and abs(s["cost_per_1k"] - point["cost_per_1k"]["value"]) < ROUNDING
        for s in steps
    )  # fmt: skip


def prefix_of_section(steps: list[dict], section: dict, mix: str) -> bool:
    """Whether the walk's steps (A, then each step tried) in the operating-points file are the first steps of the line:
    the line is the walk laid out whole, so the two agree as far as the walk went."""
    ok = True
    for point in section[mix].values():
        tried = point["steps"]
        ok &= len(tried) <= len(steps) and all(
            abs(a["quality"] - b["quality"]) < ROUNDING and abs(a["cost_per_1k"] - b["cost_per_1k"]) < ROUNDING
            and a.get("cutoff") == b["cutoff"] and a.get("threshold", b["threshold"]) == b["threshold"]
            for a, b in zip(tried, steps, strict=False)
        )  # fmt: skip
    return ok


def line_shape_checks(name: str, steps: list[dict], a: dict, gate: str, start_is_pure: bool) -> None:
    """A line's steps start at A and run in walk order: strictest first."""
    check(steps[0]["threshold"] is None and steps[0]["cutoff"] is None
          and steps[0]["quality"] == a["quality"]["value"] and steps[0]["cost_per_1k"] == a["cost_per_1k"]["value"],
          f"{name}: the line starts at A, which calls no gate")  # fmt: skip
    dial = "cutoff" if start_is_pure else "threshold"
    values = [s[dial] for s in steps[1:]]
    if not (start_is_pure and gate == FREE_GATE):
        check(values == sorted(values, reverse=True), f"{name}: the steps run from the strictest {dial} down")
    else:
        check(len(steps) == 2, f"{name}: the free checks alone have A and one step")


def committed_checks(headline: dict, per_task: dict, routed: dict, table: dict) -> None:
    print("\n-- the committed JSONs hold every cascade (A17 items 8 and 23) --")
    cutoffs = load(PROJECT_ROOT / "routers" / "gate_cutoffs.json")
    keys = list(CASCADES)
    for figure in (headline, per_task, routed):
        check(list(figure["cascades"]) == keys,
              f"{figure['figure']}: the fifteen cascades, in item 8's order, as the operating-points file names them")  # fmt: skip
        check(all(figure["cascades"][k]["label"] == CASCADES[k] and figure["cascades"][k]["operating_points_sha256"]
                  == section_hash(read_section(k)) for k in keys),
              f"{figure['figure']}: each label is the chart's and each section hash is the committed file's")  # fmt: skip
    check(set(headline["cascade_units"]) >= {"cascades", "panel_cascades", "point", "operating_points", "routed_panels"}
          and set(routed["units"]) == set(headline["cascade_units"]),
          "the cascades' keys each have a units entry, beside the older `units`")  # fmt: skip
    check(not any("band" in key for figure in (headline, per_task, routed) for key in figure),
          "no band, and no diagnostic in any figure JSON")  # fmt: skip

    pure = {cascade_section(HAIKU, gate): gate for gate in GATE_LABELS}
    for mix in PANELS:
        panel = headline["panels"][mix]
        a = panel["configs"][FRONTIER]
        check(list(panel["cascades"]) == list(pure), f"{mix} mix: the headline panel holds the three pure cascades")
        for key, gate in pure.items():
            part = panel["cascades"][key]
            rows = table["cascades"][key]["operating_points"][mix]
            name = f"{mix} mix, {CASCADES[key]}"
            line_shape_checks(name, part["steps"], a, gate, True)
            check(part["operating_points"] == rows and part["gate"] == gate,
                  f"{name}: the operating points are table v4's rows, number for number")  # fmt: skip
            check(all(on_line(p, part["steps"]) if not p["fell_back"] else (p["cutoff"] is None and p["threshold"] is None)
                      for p in part["operating_points"].values()),
                  f"{name}: each cross sits on a step of its line, or at A where the walk fell back")  # fmt: skip
            check(prefix_of_section(part["steps"], read_section(key), mix),
                  f"{name}: the walk in the operating-points file is the first steps of the line")  # fmt: skip
            if gate == FREE_GATE:
                low, high = part["point"]["quality"], part["point"]["cost_per_1k"]
                step = part["steps"][1]
                check("point" in part and low["value"] == step["quality"] and high["value"] == step["cost_per_1k"]
                      and low["low"] <= low["value"] <= low["high"] and high["low"] <= high["value"] <= high["high"],
                      f"{name}: the free checks alone are one point with a cross, on the line's one step")  # fmt: skip
            else:
                check("point" not in part, f"{name}: a curve, not a point")
    for task in TASKS:
        panel = per_task["panels"][task]
        a = panel["configs"][FRONTIER]
        check(list(panel["cascades"]) == list(pure), f"{task}: the per-task panel holds the three pure cascades")
        for key, gate in pure.items():
            part = panel["cascades"][key]
            line_shape_checks(f"{task}, {CASCADES[key]}", part["steps"], a, gate, True)
            check("operating_points" not in part and "point" not in part,
                  f"{task}, {CASCADES[key]}: a curve with no operating points (A7 item 2)")  # fmt: skip
            check([s["cutoff"] for s in part["steps"]] == [s["cutoff"] for s in headline["panels"]["realistic"]["cascades"][key]["steps"]],
                  f"{task}, {CASCADES[key]}: on the same steps as the headline's")  # fmt: skip

    print("\n-- the router-started figure: four lines to a panel, on the router's steps --")
    check(list(routed["panels"]) == list(PANELS), "a column of panels for each mix")
    for mix, panel in routed["panels"].items():
        head = headline["panels"][mix]
        check(panel["A"] == head["configs"][FRONTIER] and panel["envelope"] == head["envelope"]
              and panel["targets"] == head["targets"], f"{mix} mix: A, the envelope and the targets are the headline's")  # fmt: skip
        check(
            list(panel["routers"]) == list(NAMED_ROUTERS.values()), f"{mix} mix: a panel for each of the four routers"
        )
        for name, router in panel["routers"].items():
            letter = LETTER_OF[name]
            lines = router["lines"]
            wanted = [name, *(cascade_section(name, gate) for gate in GATE_LABELS)]
            check(list(lines) == wanted and router["label"] == LABELS[letter],
                  f"{mix} mix, {letter}: four lines, the router alone then its free, Jev and Sonnet cascades")  # fmt: skip
            alone = lines[name]
            check(alone["steps"] == head["routers"][name]["steps"]
                  and alone["operating_points"] == head["routers"][name]["operating_points"],
                  f"{mix} mix, {letter}: the router alone is its part of the headline panel")  # fmt: skip
            thresholds = [s["threshold"] for s in alone["steps"]]
            for key in wanted[1:]:
                line, gate = lines[key], lines[key]["gate"]
                name_of = f"{mix} mix, {CASCADES[key]}"
                check([s["threshold"] for s in line["steps"]] == thresholds,
                      f"{name_of}: on exactly the router's {len(thresholds) - 1} steps")  # fmt: skip
                fixed = None if gate == FREE_GATE else cutoffs[gate]
                check(all(s["cutoff"] == fixed for s in line["steps"][1:]) and line["steps"][0]["cutoff"] is None,
                      f"{name_of}: every step at the gate's fixed cutoff ({fixed}), A at none")  # fmt: skip
                line_shape_checks(name_of, line["steps"], panel["A"], gate, False)
                check(line["operating_points"] == table["cascades"][key]["operating_points"][mix],
                      f"{name_of}: the operating points are table v4's rows")  # fmt: skip
                check(all(on_line(p, line["steps"]) if not p["fell_back"] else p["threshold"] is None
                          for p in line["operating_points"].values()),
                      f"{name_of}: each cross sits on a step of its line, or at A where the walk fell back")  # fmt: skip
                check(prefix_of_section(line["steps"], read_section(key), mix),
                      f"{name_of}: the walk in the operating-points file is the first steps of the line")  # fmt: skip


def legend_of(fig: Figure) -> list[str]:
    return [t.get_text() for t in fig.legends[0].get_texts()]


def drawn_colours(axes: list[Any]) -> set[str]:
    """Every colour drawn on the axes, as lowercase hex: lines, their marker edges and the bars of a cross."""
    found: set[str] = set()
    for ax in axes:
        found |= {to_hex(line.get_color()) for line in ax.get_lines()}
        found |= {to_hex(line.get_markeredgecolor()) for line in ax.get_lines()}
        found |= {to_hex(color) for c in ax.collections for color in c.get_edgecolor()}
    return found


def fall_back_crosses(panels: dict) -> dict[str, list[tuple[str, str, tuple[float, float]]]]:
    """Per headline panel, the fall-back crosses chart v3 drew and A17 item 23 keeps: for each router and each pure
    cascade with crosses (not `haiku/free`, which is one point), its colour, the label of the targets at which it falls
    back to A ("100%, 99%: falls back to A") and A's point, where they sit."""
    wanted: dict[str, list[tuple[str, str, tuple[float, float]]]] = {}
    for mix, panel in panels.items():
        parts = [(r.style.color, panel["routers"][r.name]["operating_points"]) for r in ROUTERS]
        parts += [(CASCADE_STYLES[p["gate"]].color, p["operating_points"]) for p in panel["cascades"].values()
                  if "point" not in p]  # fmt: skip
        wanted[mix] = []
        for colour, points in parts:
            fell = [(t, p) for t, p in points.items() if p["fell_back"]]
            if fell:
                at = (round(fell[0][1]["cost_per_1k"]["value"], 9), round(fell[0][1]["quality"]["value"], 9))
                wanted[mix].append((colour, f"{', '.join(t for t, _ in fell)}: falls back to A", at))
    return wanted


def drawn_checks(headline: dict, per_task: dict, routed: dict, calibration: dict) -> dict[str, Figure]:
    """The checks on the figures drawn from the committed numbers; returns the figures, by name."""
    print("\n-- drawn from the committed numbers --")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        caption = headline_caption("run 0, n = 744.", "Crosses.", ROUTERS)
        fig = draw_headline(headline["panels"], NOTES, caption, folder / "a.png", ROUTERS)
        drawn_figures = {"the headline figure": fig}
        legend = legend_of(fig)
        parts = headline["panels"]["realistic"]["cascades"]
        free, jev, sonnet = list(parts)
        check(legend.count(cascade_point_label(parts[free])) == 1 and cascade_steps_label(parts[free]) not in legend
              and legend.count(cascade_steps_label(parts[jev])) == 1 and legend.count(cascade_points_label(parts[jev])) == 1
              and legend.count(cascade_steps_label(parts[sonnet])) == 1 and legend.count(cascade_points_label(parts[sonnet])) == 1,
              "the headline legend has the free checks as one point and the other two pure cascades as steps and crosses, once each")  # fmt: skip
        check(not any(", its steps from A" in label and label.startswith(("E:", "F:", "G:", "G+:")) for label in legend),
              "and no router's step line: those moved to the router-started figure")  # fmt: skip
        wanted, missing = fall_back_crosses(headline["panels"]), []
        for ax, mix in zip(fig.axes, headline["panels"], strict=False):
            drawn = {(to_hex(t.get_color()), t.get_text(), tuple(round(v, 9) for v in t.xy)) for t in ax.texts
                     if hasattr(t, "xy")}  # fmt: skip
            missing += [
                f"{mix}: {text!r} in {colour}" for colour, text, xy in wanted[mix] if (colour, text, xy) not in drawn
            ]
        check(sum(len(v) for v in wanted.values()) > 0 and not missing,
              "every router's and pure cascade's fall back to A is drawn at A and labeled with its targets, as chart v3 "
              "drew them" + "".join(f"\n      missing: {m}" for m in missing))  # fmt: skip
        check({CASCADE_STYLES[g].color for g in GATE_LABELS} <= drawn_colours(fig.axes[:1])
              and {r.style.color for r in ROUTERS} <= drawn_colours(fig.axes[:1]),
              "the headline draws each gate's colour and each router's")  # fmt: skip
        check("pareto_routed.png" in caption and "one point" in caption,
              "the caption points to the router-started figure and says what the free checks' point is")  # fmt: skip
        tasks = draw_tasks(
            per_task["panels"], task_caption("run 0", "counts", "Crosses.", ROUTERS), folder / "b.png", ROUTERS
        )
        drawn_figures["the per-task figure"] = tasks
        legend = legend_of(tasks)
        check(all(legend.count(cascade_steps_label(part)) == 1 for part in per_task["panels"]["ifeval"]["cascades"].values())
              and not any("operating points" in label for label in legend),
              "the per-task legend has each pure cascade's steps once, and no operating points")  # fmt: skip

        fig = draw_routed(routed["panels"], routed_caption("run 0, n = 744.", "Crosses."), folder / "c.png")
        drawn_figures["the router-started figure"] = fig
        check(len(fig.axes) == 8, "the router-started figure has a panel per router and mix: eight")
        for ax, (name, mix) in zip(fig.axes, [(n, m) for n in NAMED_ROUTERS.values() for m in PANELS], strict=True):
            router_color = next(r.style.color for r in ROUTERS if r.name == name)
            have = drawn_colours([ax])
            check({router_color, *(CASCADE_STYLES[g].color for g in GATE_LABELS)} <= have,
                  f"{LETTER_OF[name]}, {mix} mix: the router's colour and the three gates' are drawn")  # fmt: skip
        legend = legend_of(fig)
        check(
            sum(label.startswith("Router/") for label in legend) == 3,
            "the legend names the three gates' lines once each",
        )
        gates = [CASCADE_STYLES[g].color for g in GATE_LABELS]
        for router in ROUTERS:
            # The new series, the three gates' colours, against every other colour on a router's panel and each other
            # (item 23). The older pairs on the panel, A's blue against E's purple among them, are Stages 3 to 6's.
            others = [SERIES_1, SERIES_2, router.style.color, *gates]
            worst = min(gap(g, o) for g in gates for o in others if o != g)
            check(worst >= APART, f"{LETTER_OF[router.name]}'s panel: the gates' colours are {worst:.1f} from every other "
                  f"colour on it, at least {APART} in every view")  # fmt: skip
        check(gap("#2a78d6", "#2b79d7") < APART, "known positive: two near-identical colours are found too close")

        fig = draw_calibration(calibration, folder / "d.png")
        drawn_figures["the calibration figure"] = fig
        check(len(fig.axes) == 8, "the calibration figure has a panel per gate and config: eight")
        texts = " ".join(t.get_text() for t in fig.texts)
        check(all(reason in " ".join(" ".join(t.get_text().split()) for t in fig.texts) for _, _, reason in empty_cells(calibration)),
              "the calibration caption gives the reason for every empty cell")  # fmt: skip
        check("hollow" in texts or any("hollow" in label.lower() for label in legend_of(fig)),
              "and the legend says train is hollow")  # fmt: skip
    return drawn_figures


# ---- the Sonnet gate's outline (Amendment A18, item 7) ------------------------------------------------------------

LAVENDER = CASCADE_STYLES[SONNET_GATE].color
# Each figure's lavender artists, by kind, that must be there; the calibration figure draws no cascade's style.
OUTLINED_KINDS = {
    "the headline figure": {"line", "marker", "cross", "label"},
    "the per-task figure": {"line", "marker"},
    "the router-started figure": {"line", "marker", "cross"},
    "the calibration figure": set(),
}
COMMITTED_PNGS = {"the headline figure": FIGURE, "the per-task figure": TASK_FIGURE,
                  "the router-started figure": ROUTED_FIGURE}  # fmt: skip
# A pixel is dark under a relative luminance of 0.2: below the lavender's (0.42) and any blend of it with the page. A dark
# one within `REACH` pixels rings a lavender pixel; at least `RINGED` of the lavender's pixels are ringed, and at most
# `UNRINGED` of the Jev gate's pale green's, which has no outline.
DARK, REACH, RINGED, UNRINGED = 0.2, 2, 0.6, 0.3


def own_colour(artist: Artist) -> str:
    """The colour an artist is drawn in: a line's or a label's, a hollow marker's edge, or a cross's arms or a leader."""
    if isinstance(artist, Line2D) and str(artist.get_markerfacecolor()) == "none":
        return to_hex(artist.get_markeredgecolor())
    if isinstance(artist, Line2D | Text):
        return to_hex(artist.get_color())
    edges = np.atleast_2d(artist.get_edgecolor())  # type: ignore[attr-defined]
    return to_hex(edges[0]) if len(edges) else "none"


def kind_of(artist: Artist) -> str:
    if isinstance(artist, Text):
        return "label"
    if isinstance(artist, Line2D):
        return "marker" if artist.get_linestyle() in ("None", "", " ") else "line"
    return "cross"


def rings(artist: Artist) -> set[str]:
    """The colours an artist draws round itself other than its own and the page's halo: a stroke under its line, its
    arms or its letters (and its label's leader), and its markers' edge."""
    strokes = [getattr(e, "_gc", {}) for e in artist.get_path_effects() or [] if isinstance(e, Stroke)]
    found = {to_hex(gc["foreground"]) for gc in strokes if gc.get("foreground") is not None}
    if isinstance(artist, Line2D) and artist.get_marker() not in ("None", "", " ", None):
        found.add(to_hex(artist.get_markeredgecolor()))
    if isinstance(artist, Annotation) and artist.arrow_patch is not None:
        found |= rings(artist.arrow_patch)
    return found - {own_colour(artist), SURFACE}


def outline_findings(fig: Figure) -> tuple[set[str], list[str]]:
    """The kinds of lavender artist a figure draws, its legends included, and every artist that breaks the rule: a
    lavender one without exactly the Sonnet gate's outline, or one in another colour with any outline."""
    outline = CASCADE_STYLES[SONNET_GATE].outline
    artists: list[Artist] = [a for ax in fig.axes for a in (*ax.get_lines(), *ax.collections, *ax.texts)]
    artists += [h for legend in fig.legends for h in legend.legend_handles if isinstance(h, Line2D)]
    kinds, found = set(), []
    for artist in artists:
        colour, ring = own_colour(artist), rings(artist)
        if colour == LAVENDER:
            kinds.add(kind_of(artist))
            if ring != {outline}:
                found.append(f"a lavender {kind_of(artist)} with {sorted(ring) or 'no outline'}")
        elif ring:
            found.append(f"a {kind_of(artist)} in {colour} with an outline, {sorted(ring)}")
    return kinds, [f"{n} x {what}" for what, n in Counter(found).items()]


def ringed(path: Path, colour: str) -> tuple[int, float]:
    """How many of a PNG's pixels are `colour`, and the share of them that a dark pixel sits within `REACH` of."""
    image = imread(path)[..., :3]
    linear = np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)
    dark = linear @ np.array([0.2126, 0.7152, 0.0722]) < DARK
    near = np.zeros_like(dark)
    for dy in range(-REACH, REACH + 1):
        for dx in range(-REACH, REACH + 1):
            near |= np.roll(np.roll(dark, dy, axis=0), dx, axis=1)
    mine = np.all(np.abs(image - np.array(to_rgb(colour))) < 8 / 255, axis=-1)
    return int(mine.sum()), float((mine & near).sum() / max(mine.sum(), 1))


def outline_checks(figures: dict[str, Figure], per_task: dict) -> None:
    print("\n-- the Sonnet gate's lavender carries a thin dark outline (A18 item 7) --")
    outline = CASCADE_STYLES[SONNET_GATE].outline
    for name, fig in figures.items():
        kinds, found = outline_findings(fig)
        wanted = OUTLINED_KINDS[name]
        what = (f"every lavender {', '.join(sorted(wanted))} carries the outline ({outline}), the legend's too, and "
                "nothing in another colour carries one" if wanted else "it draws no lavender and no outline")  # fmt: skip
        check(kinds == wanted and not found, f"{name}: {what}" + "".join(f"\n      found: {f}" for f in found))
    sonnet, jev = CASCADE_STYLES[SONNET_GATE], CASCADE_STYLES[JEV_GATE]
    try:
        CASCADE_STYLES[SONNET_GATE] = replace(sonnet, outline=None)
        CASCADE_STYLES[JEV_GATE] = replace(jev, outline=outline)
        with tempfile.TemporaryDirectory() as tmp:
            _, found = outline_findings(draw_tasks(per_task["panels"], "caption", Path(tmp) / "e.png", ROUTERS))
    finally:
        CASCADE_STYLES[SONNET_GATE], CASCADE_STYLES[JEV_GATE] = sonnet, jev
    check(any(f.endswith("x a lavender line with no outline") for f in found)
          and any(f"in {jev.color} with an outline" in f for f in found),
          "known positive: drawn with the outline moved to the Jev gate, the bare lavender and the outlined green are found")  # fmt: skip

    for name, path in COMMITTED_PNGS.items():
        count, share = ringed(path, LAVENDER)
        green_count, green_share = ringed(path, jev.color)
        check(count > 0 and share >= RINGED and green_count > 0 and green_share <= UNRINGED,
              f"{path.relative_to(PROJECT_ROOT)}: {share:.0%} of the lavender's {count:,} pixels sit within {REACH} of a "
              f"dark one, at least {RINGED:.0%}, where the Jev gate's green's {green_share:.0%} do, at most {UNRINGED:.0%}")  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        bare = Path(tmp) / "bare.png"
        fig = Figure(figsize=(3, 2), dpi=170, facecolor=SURFACE)
        fig.subplots().plot([0, 1], [0, 1], color=LAVENDER, linewidth=1.2)
        fig.savefig(bare, facecolor=SURFACE)
        count, share = ringed(bare, LAVENDER)
    check(count > 0 and share < UNRINGED,
          f"known positive: a lavender line drawn without the outline has {share:.0%} of its pixels ringed")  # fmt: skip


# The explorer's own script, run in Node with a stand-in for the page: every series on, each view's traces printed.
TRACES_JS = r"""
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[1], "utf8");
const script = html.match(/<script>\n([\s\S]*?)<\/script>/)[1];
const any = new Proxy(function () {}, {
  get: (_, key) => (key === Symbol.toPrimitive ? () => "" : any), apply: () => any, set: () => true,
});
const page = {
  document: any, location: { hash: "" }, window: { addEventListener() {} }, localStorage: { getItem: () => null, setItem() {} },
  matchMedia: () => ({ addEventListener() {} }), MutationObserver: class { observe() {} },
  getComputedStyle: () => ({ getPropertyValue: (name) => name }), Plotly: { react() {} },
};
vm.createContext(page);
vm.runInContext(script, page);
process.stdout.write(vm.runInContext(
  "for (const r of SERIES) state.series[r.key] = true;" +
  "JSON.stringify(Object.fromEntries(Object.keys(DATA.views).map((k) => [k, traces(DATA.views[k])])))", page));
"""


def explorer_outline_findings(views: dict[str, list[dict]], outlined: set[str], outline: str) -> list[str]:
    """Per view, each trace that breaks the rule: a lavender line with no wider outline trace just beneath it, a
    lavender marker without the outline edge, or another series with the outline."""
    found = []
    for view, traces in views.items():
        for i, trace in enumerate(traces):
            name, line, marker = trace.get("name", ""), trace.get("line") or {}, trace.get("marker") or {}
            series = name.removesuffix(" points")
            edge = (marker.get("line") or {}).get("color")
            if series not in outlined:
                if outline in (line.get("color"), edge):
                    found.append(f"{view}: {name} has the outline")
                continue
            if trace.get("mode") == "lines" and line.get("color") == LAVENDER:
                under = traces[i - 1] if i else {}
                if not (under.get("name") == name and (under.get("line") or {}).get("color") == outline
                        and under.get("x") == trace["x"] and under["line"]["width"] > line["width"]):  # fmt: skip
                    found.append(f"{view}: {name}'s line has no outline beneath it")
            if "markers" in trace.get("mode", "") and marker.get("color") == LAVENDER and edge != outline:
                found.append(f"{view}: {name}'s markers have no outline edge")
    return found


def explorer_outline_checks() -> None:
    print("\n-- the explorer draws the same outline --")
    outline = CASCADE_STYLES[SONNET_GATE].outline
    data = embedded(EXPLORER.read_text(encoding="utf-8")) or {"cascades": []}
    sonnet = [c for c in data["cascades"] if c["gate"] == GATE_LABELS[SONNET_GATE]]
    check(len(sonnet) == 5 and all(c.get("outline") == outline for c in sonnet)
          and all(c.get("outline", "missing") is None for c in data["cascades"] if c not in sonnet),
          f"the committed page gives the five Sonnet-gate cascades the outline ({outline}) and the other ten none")  # fmt: skip
    node = shutil.which("node")
    if node is None:
        print("SKIP  the page's traces: Node is not installed")
        return
    run = subprocess.run([node, "-e", TRACES_JS, str(EXPLORER)], capture_output=True, text=True, timeout=60)
    views = json.loads(run.stdout) if run.returncode == 0 else {}
    outlined = {c["label"] for c in sonnet}
    found = explorer_outline_findings(views, outlined, outline or "")
    drawn = {t["name"] for traces in views.values() for t in traces if (t.get("line") or {}).get("color") == outline}
    check(run.returncode == 0 and drawn == outlined and not found,
          f"its script draws an outline under each Sonnet-gate line ({len(drawn)} of {len(outlined)}) and round each of "
          "their markers, on every view, and on no other series" + "".join(f"\n      {f}" for f in found[:12])
          + (f"\n      {run.stderr.strip()[:300]}" if run.returncode else ""))  # fmt: skip
    bare = {v: [t for t in traces if (t.get("line") or {}).get("color") != outline] for v, traces in views.items()}
    check(any("has no outline beneath it" in f for f in explorer_outline_findings(bare, outlined, outline or "")),
          "known positive: the same traces with the outline traces taken out are found")  # fmt: skip


def label_clashes(fig: Figure) -> list[str]:
    """Each router's or cascade's label (a text in its policy's colour) that overlaps another label or an error bar,
    the one of its own cross included, by the text's box; a leader may cross a bar."""
    renderer = FigureCanvasAgg(fig).get_renderer()
    colours = {r.style.color for r in ROUTERS} | {s.color for s in CASCADE_STYLES.values()}
    found = []
    for ax, mix in zip(fig.axes, PANELS, strict=False):
        # Every label but the panel's note, the one text that runs to more than one line.
        texts = [t for t in ax.texts if isinstance(t, Annotation) and t.get_text() and "\n" not in t.get_text()]
        boxes = [(t, Text.get_window_extent(t, renderer)) for t in texts]
        bars = [ax.transData.transform(segment) for c in ax.collections if isinstance(c, LineCollection)
                for segment in c.get_segments()]  # fmt: skip
        for t, box in boxes:
            if to_hex(t.get_color()) not in colours:
                continue
            name = f"{mix}: {t.get_text()!r} in {to_hex(t.get_color())}"
            found += [f"{name} on {o.get_text()!r}" for o, other in boxes if o is not t and box.overlaps(other)]
            if any(min(x0, x1) <= box.x1 and max(x0, x1) >= box.x0 and min(y0, y1) <= box.y1 and max(y0, y1) >= box.y0
                   for (x0, y0), (x1, y1) in bars):  # fmt: skip
                found.append(f"{name} on an error bar")
    return found


def label_checks(headline: dict, per_task: dict) -> None:
    print("\n-- the headline's labels clear each other and the error bars, as analysis.pareto draws it --")
    note = {"realistic": realistic_note(load_mixes()["realistic"], per_task["panels"]["mbppplus"]["items"])}
    caption = headline_caption(basis_line(headline["n"], headline["price_snapshot"]), CROSSES_LINE, ROUTERS)
    with tempfile.TemporaryDirectory() as tmp:
        clashes = label_clashes(draw_headline(headline["panels"], note, caption, Path(tmp) / "a.png", ROUTERS))
        check(not clashes, "no router's or cascade's label overlaps another label or an error bar"
              + "".join(f"\n      {c}" for c in clashes))  # fmt: skip
        g = next(r for r in ROUTERS if r.short == "G")
        kept = g.style.point_offsets.get(("realistic", "97%"))
        try:
            g.style.point_offsets[("realistic", "97%")] = (0, 0, "left")
            moved = label_clashes(draw_headline(headline["panels"], note, caption, Path(tmp) / "b.png", ROUTERS))
        finally:
            if kept is None:
                del g.style.point_offsets[("realistic", "97%")]
            else:
                g.style.point_offsets[("realistic", "97%")] = kept
        check(any(c.startswith("realistic: '97%' in") and c.endswith("on an error bar") for c in moved),
              "known positive: G's realistic 97% label moved onto its own cross is found on an error bar")  # fmt: skip


def calibration_checks(data: dict, dev_record: dict) -> None:
    print("\n-- the calibration figure (A17 item 21) --")
    gates = data["gates"]
    check(list(gates) == list(PAID_GATES) and all(list(g) == list(LADDER) for g in gates.values()),
          "the Jev gate and the Sonnet gate, each on the four configs")  # fmt: skip
    empties = empty_cells(data)
    check(empties == [(SONNET_GATE, FRONTIER, "the Sonnet gate judges no opus answer (A17 item 7)")],
          "the one empty cell is the Sonnet gate on opus, which no policy gates (A17 item 7)")  # fmt: skip
    check(all(gates[g][FRONTIER]["train"] is None for g in gates) and all(
        gates[g][c]["train"] for g in gates for c in LADDER if c != FRONTIER),
        "train has hollow markers for every config but opus, which train did not run (A1 item 3)")  # fmt: skip
    problems_found = []
    for gate, by_config in gates.items():
        for config, cell in by_config.items():
            for split in ("dev", "train"):
                bins = cell[split]
                if bins is None:
                    continue
                means = [b["mean_score"] for b in bins]
                ok = means == sorted(means) and 4 <= len(bins) <= 10
                for b in bins:
                    ok &= (
                        b["first"] - 1e-9 <= b["mean_score"] <= b["last"] + 1e-9
                        and 0 <= b["success_rate"] <= 1
                        and b["size"] > 0
                    )  # a mean of equal scores may differ from them by float dust
                    if b["low"] is not None:
                        ok &= b["low"] <= b["success_rate"] <= b["high"]
                ok &= all(a["last"] < b["first"] for a, b in zip(bins, bins[1:], strict=False))  # ties never split
                if not ok:
                    problems_found.append(f"{gate} {config} {split}")
    check(not problems_found, "every cell's bins run up in score, hold their mean, never split a tie and keep each interval "
          f"around its rate{problems_found if problems_found else ''}")  # fmt: skip
    counts = dev_record["counts"]
    sizes = {(g, c): sum(b["size"] for b in gates[g][c]["dev"]) for g in gates for c in LADDER if gates[g][c]["dev"]}
    check(all(sizes[(JEV_GATE, c)] == counts[c]["jev_calls"] - counts[c]["jev_refused"] for c in LADDER)
          and all(sizes[(SONNET_GATE, c)] == counts[c]["sonnet_calls"] - counts[c]["sonnet_refused"] for c in LADDER[:3]),
          "dev's bins hold every answer the gate scored: the record's calls less its refused ones, a refused call having "
          "no score")  # fmt: skip
    caption = caption_of(data)
    check("empty" in caption.lower() and "opus" in caption and "A17 item 7" in caption and "hollow" in caption.lower(),
          "the caption says why the cell is empty")  # fmt: skip
    other = copy.deepcopy(data)
    other["gates"][JEV_GATE]["haiku"]["dev"] = None
    other["gates"][JEV_GATE]["haiku"]["empty"] = {"dev": "a made-up reason"}
    check("made-up reason" in caption_of(other) and "made-up reason" not in caption,
          "known positive: an empty cell's reason is the one the caption gives")  # fmt: skip


def rebuild_checks(headline: dict, per_task: dict, routed: dict, calibration: dict, table: dict) -> None:
    from analysis import calibration_figure, table_v4
    from analysis.simulator import load_mixes

    print("\n-- on dev: a rebuild of every figure's numbers (hook only) --")
    inputs = table_v4.real_inputs()
    dev, draws, mixes = inputs.dev, inputs.draws, load_mixes()
    lines = cch.lines_of(dev, inputs.table, inputs.routers, inputs.cutoffs)
    for mix in PANELS:
        rebuilt = cch.headline_parts(lines, draws, mix, mixes[mix], table)
        check(json.loads(json.dumps(rebuilt)) == headline["panels"][mix]["cascades"],
              f"{mix} mix: the headline's cascades equal a rebuild from the dev matrix and the stored records")  # fmt: skip
        again = cch.routed_panel(lines, headline["panels"][mix], mix, mixes[mix], table)
        check(json.loads(json.dumps(again)) == routed["panels"][mix],
              f"{mix} mix: the router-started panel equals a rebuild")  # fmt: skip
    for task in TASKS:
        check(json.loads(json.dumps(cch.task_parts(lines, task))) == per_task["panels"][task]["cascades"],
              f"{task}: the per-task cascades equal a rebuild")  # fmt: skip
    rebuilt_data = calibration_figure.calibration_data(calibration_figure.real_splits(), calibration["price_snapshot"])
    check(json.loads(json.dumps(rebuilt_data)) == calibration, "the calibration figure's numbers equal a rebuild")


def main() -> int:
    line_checks()
    headline, per_task, routed = (load(p) for p in (FIGURE_DATA, TASK_FIGURE_DATA, ROUTED_FIGURE_DATA))
    table, calibration = load(TABLE_V4), load(CALIBRATION_DATA)
    committed_checks(headline, per_task, routed, table)
    figures = drawn_checks(headline, per_task, routed, calibration)
    outline_checks(figures, per_task)
    explorer_outline_checks()
    label_checks(headline, per_task)
    calibration_checks(calibration, load(PROJECT_ROOT / "analysis" / "gates_dev_record.json"))
    if (PROJECT_ROOT / "results" / "dev_matrix.sqlite").exists() and (PROJECT_ROOT / "results" / "gates").exists():
        rebuild_checks(headline, per_task, routed, calibration, table)
    else:
        print("\n(results/ is not here: the rebuild from the dev matrix and the stored records is skipped)")
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
