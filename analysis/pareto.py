"""Pareto chart v0 on dev (PLAN_v4.md §12; Amendment A8, item 10).

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

from analysis.bootstrap import RESAMPLES, SEED, Bootstrap, Estimate, TotalEstimate, resample
from analysis.matrix import LADDER, Matrix, load
from analysis.references import START, Curve, Oracle, envelope, oracle, quality_targets, task_oracles
from analysis.simulator import Outcome, Total, always, item_weights, load_mixes, simulate, task_totals, totals
from harness.client import PROJECT_ROOT, load_models
from harness.run_pilot import print_table

FIGURES = PROJECT_ROOT / "figures"
FIGURE, FIGURE_DATA = FIGURES / "pareto.png", FIGURES / "pareto.json"  # the headline figure and its plotted numbers
TASK_FIGURE, TASK_FIGURE_DATA = FIGURES / "pareto_tasks.png", FIGURES / "pareto_tasks.json"  # the per-task figure
PANELS = ("realistic", "uniform")
TASKS = {"ifeval": "IFEval", "banking77": "Banking77", "mmlu_pro": "MMLU-Pro", "mbppplus": "MBPP+"}  # in A5's order
# Fewer items than the "few hundred" §5 asks for before a per-task claim, so their results carry a warning (A7 item 2).
WARNED = ("ifeval", "mbppplus")
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
# and the oracle series 3.
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"
ENVELOPE_LABEL = "C: random-mixture envelope"
ORACLE_LABEL = "D: hindsight oracle (optimistic upper bound)"  # run 0 only, its flip rates over 5% (A6 item 2)
CONFIGS_LABEL = "Fixed configs, with 95% intervals"
TARGETS_LABEL = "Quality targets, % of A's quality"
CAPTION_WIDTH, NOTE_WIDTH = 150, 80  # characters per line at 8.5 and 8 points, across a 10-inch figure and a panel
LINE = 1 / 6  # inches per line of 8.5-point text, for the space under the panels


def warning(items: int) -> str:
    """A7 item 2's warning on a small task's per-task results."""
    return f"Only {items} dev items, below the plan's minimum of a few hundred: indicative"


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


def mix_panel(dev: Matrix, draws: Bootstrap, outcomes: dict[str, list[Outcome]], mix: dict[str, float]) -> dict:
    """One mix's panel of the headline figure, with its part of table v0: every number plotted or tabled."""
    frontier = totals(outcomes["opus"], mix)
    points = {config: totals(outcomes[config], mix) for config in LADDER}
    curve = envelope(points)
    solved = oracle(dev, item_weights(dev.items.values(), mix))
    # The oracle is re-solved on every resample's weights, for every target (A8 item 8).
    savings = draws.savings(dev, outcomes["opus"], mix)
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
        panels[task] = {"items": items, "warning": warning(items) if task in WARNED else None,
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


def draw_panel(ax: Axes, panel: dict, title: str, note: str | None, note_lines: int, offsets: dict) -> None:
    """A panel's references and config points with their crosses, on the shared axes style.

    `note_lines` is the most lines any panel's note in the figure takes, so panels side by side keep their titles
    level.
    """
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


def finish(fig: Figure, caption: str, path: Path) -> None:
    """One legend under every panel, which plot the same series, so the first panel's handles serve; inside a panel
    it would cover a curve. Under it the caption, and the figure is saved."""
    handles, labels = fig.axes[0].get_legend_handles_labels()
    order = [labels.index(label) for label in (ENVELOPE_LABEL, ORACLE_LABEL, CONFIGS_LABEL, TARGETS_LABEL)
             if label in labels]  # fmt: skip
    caption = textwrap.fill(caption, CAPTION_WIDTH)
    # The panels end above the legend's two rows and the caption's lines.
    bottom = (LINE * (3 + caption.count("\n") + 1) + 0.15) / fig.get_figheight()
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
               bbox_to_anchor=(0.5, bottom), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)  # fmt: skip
    fig.text(0.01, 0.01, caption, fontsize=8.5, color=INK_SECONDARY, va="bottom")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, facecolor=SURFACE)


def draw_headline(panels: dict[str, dict], notes: dict[str, str], caption: str, path: Path) -> None:
    fig = Figure(figsize=(10, 6), dpi=200, facecolor=SURFACE)
    note_lines = max(len(textwrap.wrap(note, NOTE_WIDTH)) for note in notes.values())
    for ax, (mix, panel) in zip(fig.subplots(1, 2), panels.items()):
        # Each target's share sits in the right margin, clear of the curves, which all cross the targets further left.
        for i, (share, target) in enumerate(panel["targets"].items()):
            ax.axhline(target["quality"], color=INK_SECONDARY, linestyle=":", linewidth=1, zorder=1,
                       label=TARGETS_LABEL if i == 0 else None)  # fmt: skip
            ax.annotate(share, (1, target["quality"]), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=8, color=INK_SECONDARY)  # fmt: skip
        draw_panel(ax, panel, f"{mix.capitalize()} mix", notes.get(mix), note_lines, OFFSETS)
        ax.set_xlim(0.2, 30)
        ax.set_xlabel("Cost per 1,000 requests, list price (log scale)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("Mix-weighted quality", fontsize=9, color=INK_SECONDARY)
    finish(fig, caption, path)


def draw_tasks(panels: dict[str, dict], caption: str, path: Path) -> None:
    fig = Figure(figsize=(10, 9), dpi=200, facecolor=SURFACE)
    # One cost axis for all four, so tasks' costs compare across panels; each task keeps its own quality range.
    axes = fig.subplots(2, 2, sharex=True)
    note_lines = max(len(textwrap.wrap(panel["warning"], NOTE_WIDTH)) for panel in panels.values() if panel["warning"])
    for ax, (task, panel) in zip(axes.flat, panels.items()):
        draw_panel(ax, panel, TASKS[task], panel["warning"], note_lines, TASK_OFFSETS[task])
    for ax in axes[1]:
        ax.set_xlabel("Cost per 1,000 requests, list price (log scale)", fontsize=9, color=INK_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("Quality", fontsize=9, color=INK_SECONDARY)
    costs = [row["cost_per_1k"][bound] for panel in panels.values() for row in panel["configs"].values()
             for bound in ("low", "high")]  # fmt: skip
    axes[0, 0].set_xlim(min(costs) / 1.6, max(costs) * 1.6)
    finish(fig, caption, path)


def main() -> int:
    dev = load().split("dev")
    snapshot = load_models()["price_snapshot"]
    mixes = load_mixes()
    outcomes = {config: simulate(dev, always(dev, config)) for config in LADDER}
    # One set of resamples serves every config, mix and task, so each difference from A is paired (A8 item 8).
    draws = resample(dev)
    panels = {mix: mix_panel(dev, draws, outcomes, mixes[mix]) for mix in PANELS}
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
    FIGURES.mkdir(exist_ok=True)
    for path, data in ((FIGURE_DATA, {"figure": FIGURE.name, **meta, "units": headline_units, "panels": panels}),
                       (TASK_FIGURE_DATA, {"figure": TASK_FIGURE.name, **meta, "units": task_units, "panels": per_task})):  # fmt: skip
        path.write_text(json.dumps(data, indent=2) + "\n")

    crosses = f"Crosses are 95% intervals from {RESAMPLES:,} resamples of source groups within each task."
    draw_headline(panels, {"realistic": realistic_note(mixes["realistic"], per_task["mbppplus"]["items"])},
                  f"Dev split, {basis}. {crosses} Differences between policies are judged by the paired intervals "
                  f"in table v0 ({FIGURE_DATA.relative_to(PROJECT_ROOT)}), not by whether crosses overlap.", FIGURE)  # fmt: skip
    counts = ", ".join(f"{TASKS[task]} {panel['items']}" for task, panel in per_task.items())
    draw_tasks(per_task, f"Dev split, {basis} ({counts}). Each task on its own, with its own envelope and oracle. "
                         f"{crosses} No quality targets: verdicts are given for the two mixes only (Amendment A7).",
               TASK_FIGURE)  # fmt: skip
    written = (FIGURE, FIGURE_DATA, TASK_FIGURE, TASK_FIGURE_DATA)
    print(f"\nWrote {', '.join(str(path.relative_to(PROJECT_ROOT)) for path in written)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
