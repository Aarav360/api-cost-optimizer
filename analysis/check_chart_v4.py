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
- the calibration figure: its bins, its hollow train markers and the caption that says why a cell is empty;
- on dev (hook only): a rebuild of every figure JSON equals the committed one.
"""

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from matplotlib.colors import to_hex
from matplotlib.figure import Figure

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
from analysis.check_table_v4 import TINY, tiny_table
from analysis.gate_table import PAID_GATES
from analysis.operating_points import read_section, section_hash
from analysis.pareto import (
    FIGURE_DATA,
    PANELS,
    SERIES_1,
    SERIES_2,
    TASK_FIGURE_DATA,
    TASKS,
    cascade_point_label,
    cascade_points_label,
    cascade_steps_label,
    draw_headline,
    draw_tasks,
    headline_caption,
    task_caption,
)
from analysis.references import ROUNDING
from analysis.routed_figure import ROUTED_FIGURE_DATA, draw_routed, routed_caption
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
from analysis.simulator import always, simulate
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


def drawn_checks(headline: dict, per_task: dict, routed: dict, calibration: dict) -> None:
    print("\n-- drawn from the committed numbers --")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        caption = headline_caption("run 0, n = 744.", "Crosses.", ROUTERS)
        fig = draw_headline(headline["panels"], NOTES, caption, folder / "a.png", ROUTERS)
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
        legend = legend_of(tasks)
        check(all(legend.count(cascade_steps_label(part)) == 1 for part in per_task["panels"]["ifeval"]["cascades"].values())
              and not any("operating points" in label for label in legend),
              "the per-task legend has each pure cascade's steps once, and no operating points")  # fmt: skip

        fig = draw_routed(routed["panels"], routed_caption("run 0, n = 744.", "Crosses."), folder / "c.png")
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
        check(len(fig.axes) == 8, "the calibration figure has a panel per gate and config: eight")
        texts = " ".join(t.get_text() for t in fig.texts)
        check(all(reason in " ".join(" ".join(t.get_text().split()) for t in fig.texts) for _, _, reason in empty_cells(calibration)),
              "the calibration caption gives the reason for every empty cell")  # fmt: skip
        check("hollow" in texts or any("hollow" in label.lower() for label in legend_of(fig)),
              "and the legend says train is hollow")  # fmt: skip


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
    drawn_checks(headline, per_task, routed, calibration)
    calibration_checks(calibration, load(PROJECT_ROOT / "analysis" / "gates_dev_record.json"))
    if (PROJECT_ROOT / "results" / "dev_matrix.sqlite").exists() and (PROJECT_ROOT / "results" / "gates").exists():
        rebuild_checks(headline, per_task, routed, calibration, table)
    else:
        print("\n(results/ is not here: the rebuild from the dev matrix and the stored records is skipped)")
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
