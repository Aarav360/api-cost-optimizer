"""Check that chart and table take a list of routers (PLAN_v4.md §12; Amendment A11, item 16).

Run from the project root:  uv run python -m analysis.check_chart_routers

Reads only the committed figure JSONs, no results matrix, so CI runs it. It draws the committed numbers to a temporary
folder and never rewrites a figure. That a rebuild with E alone equals the committed JSONs is `check_stage4`'s, which
needs `results/`.

- the router list holds E, F, G and G+ in that order, under their names in the operating-points file, and each router
  has a style of its own: a colour, a marker and a line that no other router's shares, and F's colour at least 15 apart
  (OKLab x100, in normal vision and under simulated protan, deutan and tritan vision) from every other colour the
  figures draw: the configs' blue, the envelope's orange, the oracle's green and E's purple (known positive: F's
  earlier red was 5 from the envelope's orange);
- a headline figure and a per-task figure drawn from E's committed numbers draw E's layer in E's style, and nothing in
  F's;
- the same two figures drawn with a second router, copied from E's numbers under another name, draw both routers'
  layers, each in its own style, each with its own legend entries, and the legend keeps E's entries first;
- a router the figure's panels hold no part for is refused before anything is drawn.
"""

import json
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from matplotlib.figure import Figure

from analysis.chart_routers import F_STYLE, G_PLUS_STYLE, G_STYLE, ROUTERS, RULES, TRAINED
from analysis.pareto import (
    FIGURE_DATA,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    TASK_FIGURE_DATA,
    ChartError,
    draw_headline,
    draw_tasks,
)
from analysis.router_names import G_PLUS, LABELS, NAMED_ROUTERS, E, F, G

# Machado, Oliveira and Fernandes (2009) simulations of protan, deutan and tritan vision at severity 1, on linear RGB.
SIMULATIONS = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}
APART = 15  # the data-viz reference's floor for two colours that can sit side by side, in OKLab x100
NOTES = {"realistic": "a note"}  # the headline figure sizes its titles by its notes
problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def oklab(rgb: list[float]) -> list[float]:
    """OKLab (Ottosson, 2020) of linear RGB."""
    r, g, b = rgb
    cone = [(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3),
            (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3),
            (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)]  # fmt: skip
    return [0.2104542553 * cone[0] + 0.7936177850 * cone[1] - 0.0040720468 * cone[2],
            1.9779984951 * cone[0] - 2.4285922050 * cone[1] + 0.4505937099 * cone[2],
            0.0259040371 * cone[0] + 0.7827717662 * cone[1] - 0.8086757660 * cone[2]]  # fmt: skip


def linear_rgb(colour: str) -> list[float]:
    values = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    return [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]


def gap(first: str, second: str) -> float:
    """The smallest OKLab distance x100 between two hex colours, as seen by normal vision and by each simulated colour
    blindness: how far apart they read to the readers who see them least apart."""
    one, other = linear_rgb(first), linear_rgb(second)
    views = [(one, other)]
    for matrix in SIMULATIONS.values():
        views.append(tuple([min(1.0, max(0.0, sum(row[k] * rgb[k] for k in range(3)))) for row in matrix]  # type: ignore[arg-type]
                           for rgb in (one, other)))  # fmt: skip
    return min(100 * math.dist(oklab(a), oklab(b)) for a, b in views)


def colours(fig: Figure) -> set[str]:
    """Every colour drawn on the first panel, lines and markers alike, lowercase."""
    ax = fig.axes[0]
    found = {str(line.get_color()).lower() for line in ax.get_lines()}
    found |= {str(color).lower() for collection in ax.collections for color in collection.get_edgecolor()}
    return found


def refused(call: object) -> bool:
    try:
        call()  # type: ignore[operator]
    except ChartError:
        return True
    return False


def main() -> int:
    headline = json.loads(FIGURE_DATA.read_text())
    per_task = json.loads(TASK_FIGURE_DATA.read_text())

    print("-- the router list --")
    check([r.name for r in ROUTERS] == ["rules", "trained", "jev", "jev_bolstered"]
          == [NAMED_ROUTERS[E], NAMED_ROUTERS[F], NAMED_ROUTERS[G], NAMED_ROUTERS[G_PLUS]],
          "the list holds E, F, G and G+ in that order, under their names in the operating-points file")  # fmt: skip
    for entry, letter, style in ((ROUTERS[2], G, G_STYLE), (ROUTERS[3], G_PLUS, G_PLUS_STYLE)):
        check(entry.label == headline["routers"][entry.name]["label"] == LABELS[letter] and entry.style == style,
              f"{letter}'s label is the committed one and its style is {letter}'s")  # fmt: skip
    check(TRAINED.label == headline["routers"][TRAINED.name]["label"] == LABELS[F] and TRAINED.style == F_STYLE,
          "F's label is the committed one and its style is F's")  # fmt: skip
    check(RULES.label == headline["routers"][RULES.name]["label"] == LABELS[E],
          "E's label is the committed one")  # fmt: skip
    twin = replace(RULES, name="twin", label="F: twin", short="F", style=F_STYLE)
    styles = [r.style for r in ROUTERS]
    for field in ("color", "marker", "linestyle"):
        check(
            len({getattr(s, field) for s in styles}) == len(styles), f"the four routers' styles differ in their {field}"
        )
    check(all(len(s.color) == 7 and s.color.startswith("#") for s in styles), "each style's colour is a hex colour")
    check(RULES.style.color == "#8e5bd0", "E's style is the one chart v1 drew it in")
    check(abs(gap("#000000", "#ffffff") - 100) < 1e-3 and gap("#2a78d6", "#2a78d6") == 0,
          "the colour gap reads 100 from black to white (OKLab lightness 0 to 1) and 0 from a colour to itself")  # fmt: skip
    drawn = {"the configs' blue": SERIES_1, "the envelope's orange": SERIES_2, "the oracle's green": SERIES_3,
             "E's purple": RULES.style.color}  # fmt: skip
    for what, colour in drawn.items():
        check(gap(F_STYLE.color, colour) >= APART, f"F's colour is {gap(F_STYLE.color, colour):.1f} from {what}, "
              f"at least {APART} in every view")  # fmt: skip
    check(gap("#e34948", SERIES_2) < APART,
          "known positive: F's earlier red (#e34948) was too close to the envelope's orange, and the check says so")  # fmt: skip

    print("\n-- drawn with E alone --")
    e_steps = "E: rules router, its steps from A (straight lines in linear cost)"
    e_points = "E's operating points, with 95% intervals"
    f_steps = "F: twin, its steps from A (straight lines in linear cost)"
    f_points = "F's operating points, with 95% intervals"
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        panels = headline["panels"]
        fig = draw_headline(panels, NOTES, "caption", folder / "a.png", (RULES,))
        legend = [t.get_text() for t in fig.legends[0].get_texts()]
        check(e_steps in legend and e_points in legend, "the headline legend has E's steps and operating points")
        check(f_steps not in legend and f_points not in legend, "and nothing of F's")
        check(RULES.style.color in colours(fig) and F_STYLE.color not in colours(fig),
              "the headline figure draws E's colour and not F's")  # fmt: skip
        tasks = draw_tasks(per_task["panels"], "caption", folder / "b.png", (RULES,))
        legend = [t.get_text() for t in tasks.legends[0].get_texts()]
        check(RULES.style.color in colours(tasks) and F_STYLE.color not in colours(tasks),
              "the per-task figure draws E's colour and not F's")  # fmt: skip
        check(e_steps in legend and e_points not in legend,
              "the per-task legend has E's steps and, as ever, no operating points (Amendment A7, item 2)")  # fmt: skip

        print("\n-- drawn with a second router --")
        two = (RULES, twin)
        with_twin = {
            mix: {**panel, "routers": {**panel["routers"], "twin": panel["routers"]["rules"]}}
            for mix, panel in panels.items()
        }  # fmt: skip
        fig = draw_headline(with_twin, NOTES, "caption", folder / "c.png", two)
        legend = [t.get_text() for t in fig.legends[0].get_texts()]
        check(all(legend.count(label) == 1 for label in (e_steps, e_points, f_steps, f_points)),
              "the legend has each router's steps and operating points once")  # fmt: skip
        check(max(legend.index(e_steps), legend.index(e_points)) < min(legend.index(f_steps), legend.index(f_points)),
              "E's entries come before F's")  # fmt: skip
        check({RULES.style.color, F_STYLE.color} <= colours(fig), "the headline figure draws both colours")
        tasks_with_twin = {
            task: {**panel, "routers": {**panel["routers"], "twin": panel["routers"]["rules"]}}
            for task, panel in per_task["panels"].items()
        }  # fmt: skip
        tasks = draw_tasks(tasks_with_twin, "caption", folder / "d.png", two)
        check({RULES.style.color, F_STYLE.color} <= colours(tasks), "the per-task figure draws both colours")

        print("\n-- a router the panels have no part for --")
        check(refused(lambda: draw_headline(panels, NOTES, "caption", folder / "e.png", two)),
              "the headline figure refuses a router its panels hold nothing for")  # fmt: skip
        check(not (folder / "e.png").exists(), "and draws nothing")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
