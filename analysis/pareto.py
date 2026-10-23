"""Pareto chart v2 on dev (PLAN_v4.md §12; Amendment A8, item 10; Amendment A9, items 14 and 15; Amendment A11, item 16).

Run from the project root:  uv run python -m analysis.pareto

Evaluates the four fixed configs as policies through the simulator, on dev (run 0),
and totals them under both task mixes and per task, with reference C, the
random-mixture envelope of their points (Amendment A8, item 7), and reference D, the
hindsight oracle (item 6): under each mix's item weights, and each task's own. The
paired bootstrap (items 8 and 9) gives every total its interval, each config its
quality difference from A and its safety margin, and the envelope's captured fraction
at each quality target (§11) its interval.

Prints table v0 (A8 item 10): per mix, each config's quality and cost with intervals,
its savings against A and its paired quality difference from A; at each quality
target, the oracle's cost, savings and routing mix and the envelope's cost and
captured fraction; the oracle's maximum quality and the items every config fails; and
per task, each config's quality and cost with intervals and the task's own oracle's
maximum quality, with no target rows (A7 item 2).

Draws two figures, each with cost per 1,000 requests at list price on a log x-axis
and quality on y, each config a point with its intervals as a cross:
- the headline figure, figures/pareto.png: a realistic and a uniform panel, with the
  envelope, the oracle and dotted lines at the quality targets;
- the per-task figure, figures/pareto_tasks.png: a 2×2 of the tasks, each with its own
  envelope and oracle, and no target lines, as verdicts are given for the mixes only.
Each is written with a JSON of every number it plots, table v0 included:
figures/pareto.json and figures/pareto_tasks.json. All four are committed and
overwritten as later stages add to the chart.

Chart and table v1 add the rules router, E, to both (Amendment A9, items 14 and 15), keeping every number and key of v0
and adding new keys:
- the headline figure draws E's steps as markers joined by straight lines in linear cost, in walk order from A, doubling
  back where they do and never hulled, the same on both panels, and a cross with 95% intervals at each of E's operating
  points, points that coincide sharing one with a combined label; there is no band;
- the per-task figure draws E's steps in each task as a line with markers and no operating points;
- table v1 adds, per mix at each operating point, the walk's verdict, quality and cost with intervals, the margin, the
  paired difference from A, savings against A, the captured fraction with the share of resamples meeting the target,
  the routing mix and the comparison with the envelope at matched cost; and per task each operating point's quality and
  cost with intervals and McNemar's test against A, with "—" for a row that falls back to A, the fixed configs'
  rows included. E's walks come from the operating-points file's rules (`analysis/rules_walk.py`), and the run stops
  if the file is not what the walk gives now.

The script takes a list of routers (`analysis/chart_routers.py`, Amendment A11, item 16): each entry has its name in the
operating-points file, its label, its style, its dev table and its walks, and its steps, operating points, McNemar rows
and matched-cost comparison are drawn and computed per entry. Chart and table v2 put the trained router, F, in the list
beside E: F's steps and crosses in the headline figure, F's steps per task, and F's rows under the key `trained`, as E's
are under `rules`. E's part of every JSON and every v1 key and value are as chart v1 committed them, and the headline
figure shows policies only (no diagnostic, no band). The rest of table v2, the same-target comparisons and the
diagnostics, and the learning-curve figure come from `analysis/table_v2.py`; `analysis/check_chart_routers.py` checks the
list and draws two routers from the committed numbers, and `analysis/check_stage5.py` checks chart and table v2 and
Amendment A11's item 21.
"""

import json
import sys
import textwrap
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator, PercentFormatter

from analysis.bootstrap import RESAMPLES, SEED, Bootstrap, Estimate, Savings, TotalEstimate, resample
from analysis.chart_routers import ROUTERS, RouterEntry
from analysis.matrix import LADDER, Matrix, load
from analysis.operating_points import OPERATING_POINTS_FILE, build_section, read_section, section_hash
from analysis.references import START, Curve, Oracle, envelope, oracle, quality_targets, task_oracles
from analysis.router_panels import config_mcnemar, router_part_of_mix_panel, router_part_of_task_panel, step_outcomes
from analysis.simulator import Outcome, Total, always, item_weights, load_mixes, simulate, task_totals, totals
from analysis.task_warnings import task_warning
from harness.client import PROJECT_ROOT, load_models
from harness.run_pilot import print_table

FIGURES = PROJECT_ROOT / "figures"
FIGURE, FIGURE_DATA = FIGURES / "pareto.png", FIGURES / "pareto.json"  # the headline figure and its plotted numbers
TASK_FIGURE, TASK_FIGURE_DATA = FIGURES / "pareto_tasks.png", FIGURES / "pareto_tasks.json"  # the per-task figure
PANELS = ("realistic", "uniform")
TASKS = {"ifeval": "IFEval", "banking77": "Banking77", "mmlu_pro": "MMLU-Pro", "mbppplus": "MBPP+"}  # in A5's order
LABELS = {"haiku": "B: haiku", "sonnet": "sonnet", "opus_low": "opus_low", "opus": "A: opus"}
# Label offsets in points, so opus_low and opus, close together on both axes, don't collide, haiku's label sits
# below the envelope, which leaves haiku's point rising to the right, and opus's sits above the 100% target line,
# which runs through its point.
OFFSETS = {"haiku": (8, -9, "left"), "sonnet": (8, 0, "left"), "opus_low": (-8, 6, "right"), "opus": (8, 9, "left")}
# Per task, sonnet's cost arm is wider, so its label drops below it; on MBPP+, opus_low's quality arm runs through
# where that would put it.
TASK_OFFSETS = {task: OFFSETS | {"sonnet": (8, -9, "left")} for task in TASKS}
TASK_OFFSETS["mbppplus"] |= {"sonnet": (-8, -9, "right")}

# The data-viz reference palette's light surface and inks; the config points are series 1, the envelope series 2
# and the oracle series 3. Each router has its own colour, in `analysis/chart_routers.py`.
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"
ENVELOPE_LABEL = "C: random-mixture envelope"
ORACLE_LABEL = "D: hindsight oracle (optimistic upper bound)"  # run 0 only, its flip rates over 5% (A6 item 2)
CONFIGS_LABEL = "Fixed configs, with 95% intervals"
TARGETS_LABEL = "Quality targets, % of A's quality"
CAPTION_WIDTH, NOTE_WIDTH = 150, 80  # characters per line at 8.5 and 8 points, across a 10-inch figure and a panel
LINE = 1 / 6  # inches per line of 8.5-point text, for the space under the panels


class ChartError(ValueError):
    """A figure asked for with a router its panels hold no part for. Nothing is drawn."""


def steps_label(router: RouterEntry) -> str:
    return f"{router.label}, its steps from A (straight lines in linear cost)"


def points_label(router: RouterEntry) -> str:
    return f"{router.short}'s operating points, with 95% intervals"


def require_parts(panels: dict[str, dict], routers: tuple[RouterEntry, ...]) -> None:
    """Refuses (ChartError), before anything is drawn, a panel that holds no part for one of the routers."""
    for name, panel in panels.items():
        if missing := [router.name for router in routers if router.name not in panel.get("routers", {})]:
            raise ChartError(f"panel {name!r} holds no part for the routers {missing}")


def realistic_note(mix: dict[str, float], items: int) -> str:
    """A7 item 2's note on the realistic mix, beside A5 item 6's caveat."""
    return (f"{mix['mbppplus']:.0%} MBPP+, the task with the fewest dev items ({items}), so its intervals are wider "
            f"than the uniform mix's. MBPP+ is easier than typical coding traffic, so this mix likely overstates "
            f"coding savings.")  # fmt: skip


def savings_against(frontier: Total, cost_per_1k: float | None) -> dict[str, float] | None:
    """Savings against A at a cost per 1,000 requests: dollars per 1k and the share of A's cost (§10), or None."""
    if cost_per_1k is None:
        return None
    saved = frontier.cost_per_1k - cost_per_1k
    return {"per_1k": saved, "share": saved / frontier.cost_per_1k}


def failed_by_every_config(matrix: Matrix) -> int:
    """The items no config gets right, which every reference leaves on its cheapest config at a score of 0."""
    return sum(max(cell.success for config in LADDER if (cell := matrix.cells.get((item_id, config)))) == 0
               for item_id in matrix.items)  # fmt: skip


def references(curve: Curve, solved: Oracle) -> dict:
    """A panel's references, C and D, as their corners, with D's maximum quality."""
    return {"envelope": {config: asdict(point) for config, point in curve.corners.items()},
            # A list, not keyed by the upgrade's item: the figures hold aggregates of dev only (A8 item 10).
            "oracle": [asdict(point) for point in solved.curve.corners.values()],
            "oracle_max_quality": solved.curve.peak.quality}  # fmt: skip


def mix_panel(dev: Matrix, draws: Bootstrap, outcomes: dict[str, list[Outcome]], mix: dict[str, float],
              savings: Savings | None = None) -> dict:  # fmt: skip
    """One mix's panel of the headline figure, with its part of table v0: every number plotted or tabled.

    `savings` is the mix's resampled oracles, solved once if the caller has them (E's part of table v1 needs them
    too); otherwise they are solved here."""
    frontier = totals(outcomes["opus"], mix)
    points = {config: totals(outcomes[config], mix) for config in LADDER}
    curve = envelope(points)
    solved = oracle(dev, item_weights(dev.items.values(), mix))
    # The oracle is re-solved on every resample's weights, for every target (A8 item 8).
    savings = savings or draws.savings(dev, outcomes["opus"], mix)
    configs = {config: {**asdict(draws.totals(outcomes[config], mix)),
                        "savings_vs_A": savings_against(frontier, points[config].cost_per_1k),
                        "difference_from_A": asdict(draws.difference(outcomes[config], outcomes["opus"], mix))}
               for config in LADDER}  # fmt: skip
    targets = {}
    for share, quality in quality_targets(frontier.quality).items():
        cost, reached = solved.cost_at(quality), curve.meeting(quality)
        targets[f"{share:.0%}"] = {
            "quality": quality,
            "oracle": {"cost_per_1k": cost, "savings_vs_A": savings_against(frontier, cost),
                       "routing": solved.routing_at(quality)},
            "envelope": {"cost_per_1k": None if reached is None else reached.cost_per_1k,
                         "captured": asdict(savings.envelope_captured(outcomes, share))},
        }  # fmt: skip
    return {"configs": configs, **references(curve, solved), "failed_by_every_config": failed_by_every_config(dev),
            "targets": targets}  # fmt: skip


def task_panels(dev: Matrix, draws: Bootstrap, outcomes: dict[str, list[Outcome]]) -> dict[str, dict]:
    """Each task's panel of the per-task figure, with its part of table v0, in A5's order of tasks."""
    spreads = {config: draws.task_totals(outcomes[config]) for config in LADDER}
    by_task = {config: task_totals(outcomes[config]) for config in LADDER}
    oracles = task_oracles(dev)
    panels = {}
    for task in TASKS:
        items = sum(item.task == task for item in dev.items.values())
        curve = envelope({config: by_task[config][task] for config in LADDER})
        panels[task] = {"items": items, "warning": task_warning(task, items),
                        "configs": {config: asdict(spreads[config][task]) for config in LADDER},
                        **references(curve, oracles[task])}  # fmt: skip
    return panels


def shown(estimate: dict, scale: float, form: str) -> str:
    """An estimate and its interval, scaled and formatted alike: 83.3 (78.9 to 87.6)."""
    value, low, high = (form.format(scale * estimate[bound]) for bound in ("value", "low", "high"))
    return f"{value} ({low} to {high})"


def shown_fraction(value: float | None) -> str:
    """A captured fraction to 2 decimals, or n/a."""
    return "n/a" if value is None else f"{value:.2f}"


def shown_savings(savings: dict[str, float] | None) -> str:
    return "n/a" if savings is None else f"${savings['per_1k']:.2f} ({savings['share']:.1%})"


def print_mix(mix: str, panel: dict) -> None:
    print(f"\n{mix.capitalize()} mix")
    rows = [[LABELS[c], f"{shown(row['quality'], 100, '{:.1f}')}%", f"${shown(row['cost_per_1k'], 1, '{:.2f}')}",
             shown_savings(row["savings_vs_A"]), f"{shown(row['difference_from_A'], 100, '{:+.1f}')} pts",
             f"{100 * row['difference_from_A']['standard_error']:.2f} pts"]
            for c, row in panel["configs"].items()]  # fmt: skip
    print_table(["config", "quality", "$ per 1k", "savings against A", "difference from A", "safety margin"], rows)
    corners = ", ".join(f"{config} ({100 * point['quality']:.1f}%, ${point['cost_per_1k']:.2f})"
                        for config, point in panel["envelope"].items())  # fmt: skip
    print(f"{ENVELOPE_LABEL}: {corners}")
    low, high = panel["oracle"][0], panel["oracle"][-1]
    print(f"{ORACLE_LABEL}: from {100 * low['quality']:.1f}% at ${low['cost_per_1k']:.2f} per 1k, {START}, up to its "
          f"maximum quality, {100 * panel['oracle_max_quality']:.1f}% at ${high['cost_per_1k']:.2f}. "
          f"{panel['failed_by_every_config']} items fail on every config.")  # fmt: skip
    rows = []
    for share, target in panel["targets"].items():
        cost, routing = target["oracle"]["cost_per_1k"], target["oracle"]["routing"]
        rows.append([f"{share} of A", f"{100 * target['quality']:.1f}%",
                     "n/a (infeasible)" if cost is None else f"${cost:.2f}", shown_savings(target["oracle"]["savings_vs_A"]),
                     *(f"{100 * routing[c]:.1f}%" if routing else "n/a" for c in LADDER)])  # fmt: skip
    print("D at each quality target: its cost, its savings against A and its routing mix")
    print_table(["quality target", "quality", "D $ per 1k", "D savings against A", *LADDER], rows)
    rows = []
    for share, target in panel["targets"].items():
        cost, captured = target["envelope"]["cost_per_1k"], target["envelope"]["captured"]
        rows.append([f"{share} of A", "n/a" if cost is None else f"${cost:.2f}",
                     f"{shown_fraction(captured['value'])} ({shown_fraction(captured['low'])} to "
                     f"{shown_fraction(captured['high'])})", f"{captured['met']:.1%}"])  # fmt: skip
    print("C at each quality target, the bar a router must clear: its cost, and its captured fraction of D's savings "
          "with its paired 95% interval and the share of resamples in which it meets the target")  # fmt: skip
    print_table(["quality target", "C $ per 1k", "C captured", "resamples meeting"], rows)


def print_task(task: str, panel: dict) -> None:
    print(f"\n{TASKS[task]}" + (f". {panel['warning']}" if panel["warning"] else f", {panel['items']} dev items"))
    rows = [[LABELS[c], f"{shown(row['quality'], 100, '{:.1f}')}%", f"${shown(row['cost_per_1k'], 1, '{:.2f}')}"]
            for c, row in panel["configs"].items()]  # fmt: skip
    print_table(["config", "quality", "$ per 1k"], rows)
    print(f"Its own hindsight oracle's maximum quality: {100 * panel['oracle_max_quality']:.1f}%")


def sampled(curve: Curve) -> tuple[list[float], np.ndarray]:
    """Points along a reference curve for the log cost axis, where its straight segments in linear cost bend.

    Each segment gets 100 evenly spaced samples on the log axis, so a short one is drawn as finely as a long one.
    """
    corners = list(curve.corners.values())
    costs = [corners[0].cost_per_1k]
    for low, high in pairwise(corners):
        costs += [float(cost) for cost in np.geomspace(low.cost_per_1k, high.cost_per_1k, 100)[1:]]
    # geomspace keeps both ends exact, so no sample falls outside the curve; one that did, n/a, would be a gap (nan).
    return costs, np.array([curve.quality_at(cost) for cost in costs], dtype=float)


def as_curve(corners: dict[str, dict] | list[dict]) -> Curve:
    """A panel's reference curve back from its JSON-ready corners."""
    named = corners if isinstance(corners, dict) else {str(i): corner for i, corner in enumerate(corners)}
    return Curve({name: Total(**corner) for name, corner in named.items()})


def as_spread(row: dict) -> TotalEstimate:
    return TotalEstimate(Estimate(**row["quality"]), Estimate(**row["cost_per_1k"]))


def step_line(steps: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """A router's steps joined by straight lines in linear cost, sampled along each segment so the log axis bends them.

    In walk order, so the line doubles back where a step costs less than the one before it, and is never hulled.
    """
    costs, qualities = [steps[0]["cost_per_1k"]], [steps[0]["quality"]]
    for low, high in pairwise(steps):
        along = np.linspace(0, 1, 100)[1:]
        costs += (low["cost_per_1k"] + along * (high["cost_per_1k"] - low["cost_per_1k"])).tolist()
        qualities += (low["quality"] + along * (high["quality"] - low["quality"])).tolist()
    return np.array(costs), np.array(qualities)


def point_labels(points: dict[str, dict]) -> list[tuple[dict, str]]:
    """A router's operating points grouped where they coincide, each with its combined label: "99%, 97%", or
    "100%: falls back to A"."""
    groups: dict[tuple[float, float], list[str]] = {}
    for target, point in points.items():
        groups.setdefault((point["quality"]["value"], point["cost_per_1k"]["value"]), []).append(target)
    labelled = []
    for targets in groups.values():
        point = points[targets[0]]
        labelled.append((point, ", ".join(targets) + (": falls back to A" if point["fell_back"] else "")))
    return labelled


def draw_steps(ax: Axes, part: dict, router: RouterEntry) -> None:
    """A router's steps as a line with markers, in its style, on every panel. No band: a router's uncertainty is the
    crosses and the table's paired intervals (item 14)."""
    style = router.style
    costs, qualities = step_line(part["steps"])
    ax.plot(costs, qualities, color=style.color, linestyle=style.linestyle, linewidth=1.2, zorder=2,
            label=steps_label(router))  # fmt: skip
    ax.plot([s["cost_per_1k"] for s in part["steps"]], [s["quality"] for s in part["steps"]], style.marker,
            markersize=4, color=style.color, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)  # fmt: skip


def draw_operating_points(ax: Axes, part: dict, router: RouterEntry, mix: str) -> None:
    """A cross at each of a router's operating points on a headline panel, `mix` naming it; the per-task panels show
    them in the table only (Amendment A7, item 2)."""
    style = router.style
    shown_points = point_labels(part["operating_points"])
    for point, _ in shown_points:
        spread = as_spread(point)
        quality, cost = spread.quality, spread.cost_per_1k
        ax.vlines(cost.value, quality.low, quality.high, color=style.color, linewidth=1.2, zorder=4)
        ax.hlines(quality.value, cost.low, cost.high, color=style.color, linewidth=1.2, zorder=4)
    ax.plot([as_spread(p).cost_per_1k.value for p, _ in shown_points],
            [as_spread(p).quality.value for p, _ in shown_points], style.point_marker, markersize=6.5,
            color=style.color, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5,
            label=points_label(router))  # fmt: skip
    for point, label in shown_points:
        dx, dy, ha = style.point_offsets.get((mix, label), (8, -10, "left"))
        # A thin leader to a label that sits clear of the crowd.
        leader = {"arrowstyle": "-", "color": style.color, "linewidth": 0.6, "shrinkA": 2, "shrinkB": 4}
        ax.annotate(label, (point["cost_per_1k"]["value"], point["quality"]["value"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=8.5, color=style.color, zorder=6,
                    arrowprops=leader if max(abs(dx), abs(dy)) > 20 else None,
                    path_effects=[withStroke(linewidth=3, foreground=SURFACE)])  # fmt: skip


def draw_references(ax: Axes, panel: dict, offsets: dict) -> None:
    """A panel's references and config points with their crosses. The routers' layers go on next, before
    `frame_panel` fixes the axes to everything drawn."""
    ax.set_facecolor(SURFACE)
    ax.plot(*sampled(as_curve(panel["envelope"])), color=SERIES_2, linewidth=1.8, zorder=2, label=ENVELOPE_LABEL)
    ax.plot(*sampled(as_curve(panel["oracle"])), color=SERIES_3, linewidth=1.8, zorder=2, label=ORACLE_LABEL)
    spreads = {config: as_spread(row) for config, row in panel["configs"].items()}
    # Each arm spans its interval, which comes from the resamples and needn't be centred on the point.
    for spread in spreads.values():
        quality, cost = spread.quality, spread.cost_per_1k
        ax.vlines(cost.value, quality.low, quality.high, color=SERIES_1, linewidth=1, alpha=0.6, zorder=2)
        ax.hlines(quality.value, cost.low, cost.high, color=SERIES_1, linewidth=1, alpha=0.6, zorder=2)
    ax.plot([s.cost_per_1k.value for s in spreads.values()], [s.quality.value for s in spreads.values()], "o",
            markersize=7, color=SERIES_1, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label=CONFIGS_LABEL)  # fmt: skip
    for config, spread in spreads.items():
        dx, dy, ha = offsets[config]
        # A halo in the surface color keeps a label legible where an arm or a target line runs under it.
        ax.annotate(LABELS[config], (spread.cost_per_1k.value, spread.quality.value), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=9, color=INK, zorder=4,
                    path_effects=[withStroke(linewidth=3, foreground=SURFACE)])  # fmt: skip


def frame_panel(ax: Axes, title: str, note: str | None, note_lines: int) -> None:
    """The shared axes style, the title and the note.

    `note_lines` is the most lines any panel's note in the figure takes, so panels side by side keep their titles
    level.
    """
    ax.set_xscale("log")
    # Dollars, not powers of ten: the ticks a reader prices requests in.
    ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    # Whole-point steps: the crosses widen the range enough for 2.5-point ticks, which whole percents misprint.
    ax.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.margins(y=0.25)
    # The margin above a task's crosses mustn't run past 100%, a quality no policy can exceed.
    ax.set_ylim(top=min(ax.get_ylim()[1], 1))
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8 + 11 * note_lines)
    if note:
        ax.annotate(textwrap.fill(note, NOTE_WIDTH), (0, 1), xycoords="axes fraction", xytext=(0, 5),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8, color=INK_SECONDARY,
                    linespacing=1.3)  # fmt: skip
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def finish(fig: Figure, caption: str, path: Path, routers: tuple[RouterEntry, ...]) -> Figure:
    """One legend under every panel, which plot the same series, so the first panel's handles serve; inside a panel
    it would cover a curve. The routers' entries follow the references', in the routers' order. Under it the caption,
    and the figure is saved and returned."""
    handles, labels = fig.axes[0].get_legend_handles_labels()
    wanted = (ENVELOPE_LABEL, ORACLE_LABEL, CONFIGS_LABEL,
              *(label for router in routers for label in (steps_label(router), points_label(router))),
              TARGETS_LABEL)  # fmt: skip
    order = [labels.index(label) for label in wanted if label in labels]
    caption = textwrap.fill(caption, CAPTION_WIDTH)
    # The panels end above the legend's rows (two entries to a row) and the caption's lines.
    legend_rows = -(-len(order) // 2)
    bottom = (LINE * (legend_rows + 1 + caption.count("\n") + 1) + 0.15) / fig.get_figheight()
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
               bbox_to_anchor=(0.5, bottom), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)  # fmt: skip
    fig.text(0.01, 0.01, caption, fontsize=8.5, color=INK_SECONDARY, va="bottom")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    return fig


def draw_headline(panels: dict[str, dict], notes: dict[str, str], caption: str, path: Path,
                  routers: tuple[RouterEntry, ...] = ROUTERS) -> Figure:  # fmt: skip
    require_parts(panels, routers)
    fig = Figure(figsize=(10, 6), dpi=200, facecolor=SURFACE)
    note_lines = max(len(textwrap.wrap(note, NOTE_WIDTH)) for note in notes.values())
    for ax, (mix, panel) in zip(fig.subplots(1, 2), panels.items()):
        # Each target's share sits in the right margin, clear of the curves, which all cross the targets further left.
        for i, (share, target) in enumerate(panel["targets"].items()):
            ax.axhline(target["quality"], color=INK_SECONDARY, linestyle=":", linewidth=1, zorder=1,
                       label=TARGETS_LABEL if i == 0 else None)  # fmt: skip
            ax.annotate(share, (1, target["quality"]), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=8, color=INK_SECONDARY)  # fmt: skip
        draw_references(ax, panel, OFFSETS)
        for router in routers:
            draw_steps(ax, panel["routers"][router.name], router)
            draw_operating_points(ax, panel["routers"][router.name], router, mix)
        frame_panel(ax, f"{mix.capitalize()} mix", notes.get(mix), note_lines)
        ax.set_xlim(0.2, 30)
        ax.set_xlabel("Cost per 1,000 requests, list price (log scale)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("Mix-weighted quality", fontsize=9, color=INK_SECONDARY)
    return finish(fig, caption, path, routers)


def draw_tasks(panels: dict[str, dict], caption: str, path: Path,
               routers: tuple[RouterEntry, ...] = ROUTERS) -> Figure:  # fmt: skip
    require_parts(panels, routers)
    fig = Figure(figsize=(10, 9), dpi=200, facecolor=SURFACE)
    # One cost axis for all four, so tasks' costs compare across panels; each task keeps its own quality range.
    axes = fig.subplots(2, 2, sharex=True)
    note_lines = max(len(textwrap.wrap(panel["warning"], NOTE_WIDTH)) for panel in panels.values() if panel["warning"])
    for ax, (task, panel) in zip(axes.flat, panels.items()):
        draw_references(ax, panel, TASK_OFFSETS[task])
        for router in routers:
            draw_steps(ax, panel["routers"][router.name], router)
        frame_panel(ax, TASKS[task], panel["warning"], note_lines)
    for ax in axes[1]:
        ax.set_xlabel("Cost per 1,000 requests, list price (log scale)", fontsize=9, color=INK_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("Quality", fontsize=9, color=INK_SECONDARY)
    costs = [row["cost_per_1k"][bound] for panel in panels.values() for row in panel["configs"].values()
             for bound in ("low", "high")]  # fmt: skip
    costs += [step["cost_per_1k"] for panel in panels.values() for router in routers
              for step in panel["routers"][router.name]["steps"]]  # fmt: skip
    axes[0, 0].set_xlim(min(costs) / 1.6, max(costs) * 1.6)
    return finish(fig, caption, path, routers)


def shown_threshold(point: dict) -> str:
    return "falls back to A" if point["fell_back"] else f"{point['threshold']:.4f}"


def shown_gap(matched: dict) -> str:
    """A router's paired quality difference from the envelope at its cost, with its interval, in points; n/a where unread."""

    def points(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:+.1f}"

    if matched["difference"] is None:
        return "n/a"
    return f"{points(matched['difference'])} ({points(matched['low'])} to {points(matched['high'])}) pts"


def print_router_mix(mix: str, part: dict, router: RouterEntry) -> None:
    """A router's part of table v1 for one mix (Amendment A9, item 15), at each of its operating points."""
    points, short = part["operating_points"], router.short
    print(f"\n{mix.capitalize()} mix, {short} at its operating points")
    rows = [[share, shown_threshold(p), f"{p['steps_certified']} of {p['steps_tried']}",
             f"{shown(p['quality'], 100, '{:.1f}')}%", f"${shown(p['cost_per_1k'], 1, '{:.2f}')}",
             f"{100 * p['margin']:.2f} pts", f"{shown(p['difference_from_A'], 100, '{:+.1f}')} pts",
             shown_savings(p["savings_vs_A"])] for share, p in points.items()]  # fmt: skip
    print_table(["quality target", "threshold", "steps certified", "quality", "$ per 1k", "safety margin",
                 "difference from A", "savings against A"], rows)  # fmt: skip
    rows = []
    for share, p in points.items():
        captured, matched = p["captured"], p["matched_cost"]
        reading = matched["envelope"]
        rows.append([share, f"{shown_fraction(captured['value'])} ({shown_fraction(captured['low'])} to "
                     f"{shown_fraction(captured['high'])})", f"{captured['met']:.1%}",
                     "n/a" if reading is None else f"{100 * reading:.1f}%", shown_gap(matched),
                     *(f"{100 * p['routing'][c]:.1f}%" for c in LADDER)])  # fmt: skip
    print(
        f"Captured fraction of D's savings with its paired 95% interval and the share of resamples meeting the "
        f"target; the envelope's quality at {short}'s cost and {short}'s paired difference from it (matched cost); "
        f"the routing mix"
    )
    print_table(["quality target", "captured", "resamples meeting", f"C at {short}'s cost", f"{short} less C", *LADDER],
                rows)  # fmt: skip


def print_router_task(task: str, panel: dict, routers: tuple[RouterEntry, ...]) -> None:
    """The fixed configs' and the routers' McNemar tests against A in one task, with their operating points there."""
    print(f"\n{TASKS[task]}: McNemar's test against A (b: items only the row gets right, c: only A; exact, two-sided, "
          f"descriptive only)")  # fmt: skip
    dash = ("—",) * 3
    rows = []
    for config, test in panel["mcnemar_vs_A"].items():
        row = panel["configs"][config]
        rows.append([LABELS[config], f"{shown(row['quality'], 100, '{:.1f}')}%",
                     f"${shown(row['cost_per_1k'], 1, '{:.2f}')}",
                     *(dash if test is None else (test["b"], test["c"], f"{test['p']:.3f}"))])  # fmt: skip
    for router in routers:
        for mix, by_share in panel["routers"][router.name]["operating_points"].items():
            for share, p in by_share.items():
                test = p["mcnemar"]
                rows.append([f"{router.short}, {mix} {share}: {shown_threshold(p)}",
                             f"{shown(p['quality'], 100, '{:.1f}')}%", f"${shown(p['cost_per_1k'], 1, '{:.2f}')}",
                             *(dash if test is None else (test["b"], test["c"], f"{test['p']:.3f}"))])  # fmt: skip
    print_table(["row", "quality", "$ per 1k", "b", "c", "p"], rows)


def main() -> int:
    dev = load().split("dev")
    snapshot = load_models()["price_snapshot"]
    mixes = load_mixes()
    outcomes = {config: simulate(dev, always(dev, config)) for config in LADDER}
    # One set of resamples serves every config, mix and task, so each difference from A is paired (A8 item 8).
    draws = resample(dev)
    # Each router's walks, which must be the ones the operating-points file holds: the chart draws the file's
    # operating points.
    tables = {router.name: router.dev_table() for router in ROUTERS}
    walks = {router.name: router.walks(dev, tables[router.name], draws) for router in ROUTERS}
    for router in ROUTERS:
        if build_section(walks[router.name]) != read_section(router.name):
            print(f"{OPERATING_POINTS_FILE.relative_to(PROJECT_ROOT)}'s section for {router.name} is not what the walk "
                  f"on dev gives; run the router's walk first, and see why they differ")  # fmt: skip
            return 1
    steps = {router.name: step_outcomes(dev, tables[router.name], outcomes["opus"]) for router in ROUTERS}
    savings = {mix: draws.savings(dev, outcomes["opus"], mixes[mix]) for mix in PANELS}
    panels = {mix: mix_panel(dev, draws, outcomes, mixes[mix], savings[mix]) for mix in PANELS}
    per_task = task_panels(dev, draws, outcomes)
    basis = f"run 0, n = {len(dev.items)}, list prices {snapshot}"

    print(f"Table v0: fixed configs and references on dev ({basis}), with percentile 95% intervals from "
          f"{RESAMPLES:,} paired bootstrap resamples (seed {SEED}). The safety margin is one standard error of the "
          f"quality difference from A.")  # fmt: skip
    for mix, panel in panels.items():
        print_mix(mix, panel)
    print("\nPer task, each on its own, with its own oracle. No quality targets: verdicts are given for the mixes only "
          "(Amendment A7, item 2).")  # fmt: skip
    for task, panel in per_task.items():
        print_task(task, panel)

    # Table v1 adds keys to table v0's panels and changes none of them (Amendment A9, item 15).
    for mix in PANELS:
        parts = {router.name: router_part_of_mix_panel(draws, savings[mix], steps[router.name],
                                                       walks[router.name][mix], outcomes, mixes[mix])
                 for router in ROUTERS}  # fmt: skip
        panels[mix] = {**panels[mix], "routers": parts}
    for task in TASKS:
        parts = {router.name: router_part_of_task_panel(draws, steps[router.name], walks[router.name], task)
                 for router in ROUTERS}  # fmt: skip
        per_task[task] = {**per_task[task], "mcnemar_vs_A": config_mcnemar(outcomes, task), "routers": parts}
    sections = {router.name: read_section(router.name) for router in ROUTERS}
    facts = {
        router.name: router.facts() for router in ROUTERS
    }  # one pass over the dev requests, for the printout and the JSON
    where = OPERATING_POINTS_FILE.relative_to(PROJECT_ROOT)
    for router in ROUTERS:
        said = "".join(f" {fact}." for fact in facts[router.name].values())
        print(f"\nTable v1: {router.label}, on dev.{said} Its operating points are the cheapest certified steps of "
              f"the walk (Amendment A9, items 9 and 10), written to {where}.")  # fmt: skip
        for mix in PANELS:
            print_router_mix(mix, panels[mix]["routers"][router.name], router)
    print("\nPer task, the fixed configs' and the routers' operating points' McNemar tests (Amendment A9, item 11). —: "
          "a row that falls back to A, or A itself.")  # fmt: skip
    for task, panel in per_task.items():
        print_router_task(task, panel, ROUTERS)

    bootstrap = {"resamples": RESAMPLES, "seed": SEED, "interval": "percentile 95%",
                 "resampled": "source groups within each task, the mix weights applied afterwards"}  # fmt: skip
    meta = {"built_by": "analysis/pareto.py", "split": "dev", "run": 0, "n": len(dev.items),
            "price_snapshot": snapshot, "bootstrap": bootstrap}  # fmt: skip
    estimate = ("its value on dev, the point plotted, with its percentile 95% interval (low, high), the cross's arm, "
                "and its standard error, the standard deviation over the resamples")  # fmt: skip
    units = {
        "quality": "share of items answered correctly, 0 to 1, mix-weighted in the headline figure",
        "cost_per_1k": "dollars per 1,000 requests, at list price, mix-weighted in the headline figure",
        "configs": f"each config's quality and cost_per_1k: {estimate}",
        "envelope": "the envelope's corners, cheapest first, joined by straight lines in linear cost",
        "oracle": f"the oracle's corners: {START}, then the point after each upgrade, in the order taken, joined "
                  f"by straight lines in linear cost",
        "oracle_max_quality": "the quality at the oracle's last corner",
    }  # fmt: skip
    headline_units = units | {
        "savings_vs_A": "A's cost per 1k less the config's or the oracle's (per_1k), and that as a share of A's (share)",
        "difference_from_A": "each config's quality less A's, paired: the same resamples weigh both; its "
                             "standard_error is the config's safety margin (Amendment A8, item 9)",
        "failed_by_every_config": "the dev items no config gets right",
        "targets": "each quality target, keyed by its share of A's (opus's) quality under the mix: its quality, the "
                   "oracle's cost there (null: infeasible), savings against A and routing mix (each config's share "
                   "of requests), and the envelope's cost there and captured fraction of the oracle's savings (§7), "
                   "with its percentile 95% interval over every resample, null being n/a, ranked below every "
                   "number, and met, the share of resamples in which the envelope meets the target (A8 item 8)",
    }  # fmt: skip
    task_units = units | {"items": "the task's dev items",
                          "warning": "Amendment A7 item 2's warning on the task's per-task results, or null"}  # fmt: skip
    # Table v1's keys, added to v0's (Amendment A9, items 14 and 15).
    router_units = {
        "routers": "the routers on the chart, keyed by their name in the operating-points file; E is `rules`, the "
        "rules router",
        "steps": "E's steps in walk order: A (threshold null) first, then each step from the highest threshold to the "
        "lowest, each with its quality and cost_per_1k on dev (mix-weighted in the headline figure), joined "
        "by straight lines in linear cost and never hulled, so the line can double back; the same steps on "
        "both mixes' panels since they come from router scores",
        "operating_points": "E's operating points, keyed by quality target (the headline figure's panel, one mix) or "
        "by mix and then quality target (the per-task figure), each the cheapest certified step "
        "of the walk (Amendment A9, items 9 and 10), whose quality and cost_per_1k are each "
        + estimate
        + ", and whose difference_from_A and savings_vs_A are the config rows' (against A)",
        "threshold": "the operating point's threshold, null for a fallback to A",
        "fell_back": "whether the walk certified nothing cheaper than A, so the operating point is A itself: no "
        "router call, no overhead, captured fraction 0",
        "steps_certified": "the steps the walk certified, A not counted",
        "steps_tried": "the steps the walk tried, the last of which may be the first that missed",
        "target": "the quality target on dev: its share of A's quality under the mix",
        "margin": "the safety margin: one paired standard error of the point's quality difference from A "
        "(Amendment A8, item 9)",
        "captured": "the operating point's captured fraction of the oracle's savings at its target (§7), with its "
        "percentile 95% interval over every resample (null being n/a, ranked below every number), and "
        "met, the share of resamples in which it meets the target (Amendment A8, item 8)",
        "routing": "the share of requests answered by each config, weighted like quality and cost",
        "matched_cost": "the envelope's quality at the operating point's cost (envelope; flat at its top quality "
        "past its costliest corner, Amendment A8 item 7; null below its cheapest), the operating "
        "point's paired quality difference from it (difference) with its percentile 95% interval "
        "(low, high; each resample rebuilds the envelope from its own config points, null being "
        "n/a), and unread, the share of resamples in which the point costs less than the "
        "envelope's cheapest corner (Amendment A9, items 12 and 15)",
        "label": "the router's name on the chart",
        "operating_points_file": "the file holding the router's section: every threshold, margin and step of its walks",
        "operating_points_sha256": "the sha256 of E's section of the operating-points file, written with sorted "
        "keys (Amendment A9, item 18)",
        "task_inference_dev": "how often E's inferred type is the item's task over the dev requests, on this "
        "benchmark's templates only (Amendment A9, item 5)",
    }
    mcnemar_units = {
        "mcnemar": "McNemar's test of the row against A within the task (Amendment A9, item 11): b, the items only "
        "the row gets right; c, the items only A gets right; p, exact and two-sided, descriptive only. "
        "Null (shown as —) for an operating point that falls back to A",
        "mcnemar_vs_A": "the same for each fixed config against A; A's own row is null",
    }
    headline_units |= router_units
    task_units |= router_units | mcnemar_units
    section_name = str(OPERATING_POINTS_FILE.relative_to(PROJECT_ROOT))
    router_meta = {router.name: {"label": router.label, **facts[router.name], "operating_points_file": section_name,
                                 "operating_points_sha256": section_hash(sections[router.name])}
                   for router in ROUTERS}  # fmt: skip
    FIGURES.mkdir(exist_ok=True)
    for path, data in ((FIGURE_DATA, {"figure": FIGURE.name, **meta, "units": headline_units, "routers": router_meta,
                                      "panels": panels}),
                       (TASK_FIGURE_DATA, {"figure": TASK_FIGURE.name, **meta, "units": task_units,
                                           "panels": per_task})):  # fmt: skip
        path.write_text(json.dumps(data, indent=2) + "\n")

    crosses = f"Crosses are 95% intervals from {RESAMPLES:,} resamples of source groups within each task."
    whose = " and ".join(f"{router.short}'s" for router in ROUTERS)
    stepped = f"{whose} steps are joined by straight lines in linear cost, in walk order from A"
    draw_headline(panels, {"realistic": realistic_note(mixes["realistic"], per_task["mbppplus"]["items"])},
                  f"Dev split, {basis}. {crosses} Differences between policies are judged by the paired intervals "
                  f"in tables v0 and v1 ({FIGURE_DATA.relative_to(PROJECT_ROOT)}), not by whether crosses overlap. "
                  f"{stepped}; their crosses mark their operating points, each labeled with its target.", FIGURE)  # fmt: skip
    counts = ", ".join(f"{TASKS[task]} {panel['items']}" for task, panel in per_task.items())
    draw_tasks(per_task, f"Dev split, {basis} ({counts}). Each task on its own, with its own envelope and oracle. "
                         f"{crosses} No quality targets: verdicts are given for the two mixes only (Amendment A7). "
                         f"{stepped}, with no operating points.", TASK_FIGURE)  # fmt: skip
    written = (FIGURE, FIGURE_DATA, TASK_FIGURE, TASK_FIGURE_DATA)
    print(f"\nWrote {', '.join(str(path.relative_to(PROJECT_ROOT)) for path in written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
