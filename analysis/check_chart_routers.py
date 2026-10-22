"""Check that chart and table take a list of routers (PLAN_v4.md §12; Amendment A11, item 16).

Run from the project root:  uv run python -m analysis.check_chart_routers

Reads only the committed figure JSONs, no results matrix, so CI runs it. It draws the committed numbers to a temporary
folder and never rewrites a figure. That a rebuild with E alone equals the committed JSONs is `check_stage4`'s, which
needs `results/`.

- the router list holds E first, under its name in the operating-points file, and each router has a style of its own:
  a colour, a marker and a line that no other router's shares, and F's is ready beside E's;
- a headline figure and a per-task figure drawn from E's committed numbers draw E's layer in E's style, and nothing in
  F's;
- the same two figures drawn with a second router, copied from E's numbers under another name, draw both routers'
  layers, each in its own style, each with its own legend entries, and the legend keeps E's entries first;
- a router the figure's panels hold no part for is refused before anything is drawn.
"""

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from matplotlib.figure import Figure

from analysis.chart_routers import F_STYLE, ROUTERS, RULES
from analysis.pareto import FIGURE_DATA, TASK_FIGURE_DATA, ChartError, draw_headline, draw_tasks

NOTES = {"realistic": "a note"}  # the headline figure sizes its titles by its notes
problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


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
    check([r.name for r in ROUTERS] == ["rules"], "the list holds E alone for now, under its name in the file")
    check(RULES.label == headline["routers"]["rules"]["label"] == "E: rules router",
          "E's label is the committed one")  # fmt: skip
    twin = replace(RULES, name="twin", label="F: twin", short="F", style=F_STYLE)
    styles = [RULES.style, F_STYLE]
    for field in ("color", "marker", "linestyle"):
        check(len({getattr(s, field) for s in styles}) == 2, f"E's and F's styles differ in their {field}")
    check(all(len(s.color) == 7 and s.color.startswith("#") for s in styles), "each style's colour is a hex colour")
    check(RULES.style.color == "#8e5bd0", "E's style is the one chart v1 drew it in")

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
