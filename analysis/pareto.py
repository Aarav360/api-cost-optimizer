"""Pareto chart v0 on dev (PLAN_v4.md §12; Amendment A8, item 10).

Run from the project root:  uv run python -m analysis.pareto

Evaluates the four fixed configs as policies through the simulator, on dev (run 0),
and totals them under both task mixes. Prints the totals and draws the headline
figure: a realistic and a uniform panel, cost per 1,000 requests at list price on a
log x-axis and mix-weighted quality on y. Writes figures/pareto.png and, with every
plotted number, figures/pareto.json. Both are committed and overwritten as later
stages add to the chart.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, PercentFormatter

from analysis.matrix import LADDER, Matrix, load
from analysis.simulator import Total, always, load_mixes, simulate, totals
from harness.client import PROJECT_ROOT, load_models
from harness.run_pilot import print_table

FIGURES = PROJECT_ROOT / "figures"
FIGURE, FIGURE_DATA = FIGURES / "pareto.png", FIGURES / "pareto.json"  # the headline figure and its plotted numbers
PANELS = ("realistic", "uniform")
LABELS = {"haiku": "B: haiku", "sonnet": "sonnet", "opus_low": "opus_low", "opus": "A: opus"}
# Label offsets in points, so opus_low and opus, close together on both axes, don't collide.
OFFSETS = {"haiku": (8, 0, "left"), "sonnet": (8, 0, "left"), "opus_low": (-8, 6, "right"), "opus": (8, -9, "left")}

# The data-viz reference palette's light surface and inks; the config points are series 1.
SURFACE, INK, INK_SECONDARY, GRID, SERIES_1 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def config_points(dev: Matrix, mixes: dict) -> dict[str, dict[str, Total]]:
    """Each fixed config's totals under each mix: {mix: {config: Total}}."""
    outcomes = {config: simulate(dev, always(dev, config)) for config in LADDER}
    return {mix: {config: totals(outcomes[config], mixes[mix]) for config in LADDER} for mix in PANELS}


def draw(points: dict[str, dict[str, Total]], caption: str, path: Path) -> None:
    fig = Figure(figsize=(10, 4.4), dpi=200, facecolor=SURFACE)
    for ax, mix in zip(fig.subplots(1, 2), PANELS):
        ax.set_facecolor(SURFACE)
        for config, total in points[mix].items():
            ax.plot(total.cost_per_1k, total.quality, "o", markersize=7, color=SERIES_1, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3)  # fmt: skip
            dx, dy, ha = OFFSETS[config]
            ax.annotate(LABELS[config], (total.cost_per_1k, total.quality), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=9, color=INK)  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlim(0.2, 25)
        ax.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.margins(y=0.25)
        ax.set_title(f"{mix.capitalize()} mix", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Cost per 1,000 requests, list price (log scale)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("Mix-weighted quality", fontsize=9, color=INK_SECONDARY)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    fig.text(0.01, 0.01, caption, fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor=SURFACE)


def main() -> int:
    dev = load().split("dev")
    snapshot = load_models()["price_snapshot"]
    points = config_points(dev, load_mixes())
    basis = f"run 0, n = {len(dev.items)}, list prices {snapshot}"

    print(f"Fixed configs on dev ({basis}): quality, $ per 1,000 requests")
    rows = [[LABELS[c]] + [f"{100 * points[m][c].quality:.1f}%  ${points[m][c].cost_per_1k:.2f}" for m in PANELS]
            for c in LADDER]  # fmt: skip
    print_table(["config"] + list(PANELS), rows)

    units = {
        "quality": "mix-weighted share of items answered correctly, 0 to 1",
        "cost_per_1k": "mix-weighted dollars per 1,000 requests, at list price",
    }
    panels = {mix: {"configs": {c: asdict(t) for c, t in points[mix].items()}} for mix in PANELS}
    data = {"figure": FIGURE.name, "built_by": "analysis/pareto.py", "split": "dev", "run": 0, "n": len(dev.items),
            "price_snapshot": snapshot, "units": units, "panels": panels}  # fmt: skip
    FIGURES.mkdir(exist_ok=True)
    FIGURE_DATA.write_text(json.dumps(data, indent=2) + "\n")
    draw(points, f"Dev split, {basis}.", FIGURE)
    print(f"\nWrote {FIGURE.relative_to(PROJECT_ROOT)} and {FIGURE_DATA.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
