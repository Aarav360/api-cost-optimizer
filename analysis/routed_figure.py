"""The router-started figure of chart v4 (PLAN_v4.md §12; Amendment A17, items 8 and 23).

One panel per router and mix, eight in all. Each panel draws four lines on the same steps: the router alone and its three
cascades (the free checks, the Jev gate and the Sonnet gate), each a line with markers and a cross at each of its operating
points, with A and the random-mixture envelope for reference. The router keeps its own style (`analysis/chart_routers.py`)
and each cascade takes its gate's (`CASCADE_STYLES`), the same as the pure cascade with that gate has on the headline
figure. Drawn from `figures/pareto_routed.json`'s numbers alone, so a check can draw the committed ones without `results/`.

`analysis/pareto.py` writes the figure and its JSON, since the router-started panels are built from the headline panels'
numbers (A, the envelope, the targets and each router's own line).
"""

import textwrap
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke

from analysis.chart_routers import CASCADE_STYLES, ROUTERS, RouterStyle
from analysis.pareto import (
    CAPTION_WIDTH,
    ENVELOPE_LABEL,
    GRID,
    INK,
    INK_SECONDARY,
    LINE,
    SERIES_1,
    SERIES_2,
    SURFACE,
    TARGETS_LABEL,
    as_curve,
    as_spread,
    draw_cross,
    frame_panel,
    line_effects,
    marker_edge,
    sampled,
    step_line,
)
from analysis.router_names import FREE_GATE, GATE_LABELS, JEV_GATE, LETTER_OF, NAMED_ROUTERS, SONNET_GATE
from harness.client import PROJECT_ROOT

ROUTED_FIGURE = PROJECT_ROOT / "figures" / "pareto_routed.png"
ROUTED_FIGURE_DATA = PROJECT_ROOT / "figures" / "pareto_routed.json"
A_LABEL = "A: opus, with 95% intervals"
CROSS_LABEL = "Cross: an operating point, with 95% intervals"
PANEL_ORDER = tuple(NAMED_ROUTERS.values())


def style_of(name: str, line: dict) -> RouterStyle:
    """A line's style: the router's own for the router alone, the gate's for a cascade."""
    if line["gate"] is None:
        return next(router.style for router in ROUTERS if router.name == name)
    return CASCADE_STYLES[line["gate"]]


def draw_line(ax: Axes, line: dict, style: RouterStyle) -> None:
    """One line's steps joined in walk order, as chart v3 joins a router's, with its markers. The free checks' line is
    drawn wider, fainter and beneath, since it can lie on its router's; where two lines coincide the one drawn later shows."""
    costs, qualities = step_line(line["steps"])
    free = line["gate"] == FREE_GATE
    ax.plot(
        costs,
        qualities,
        color=style.color,
        linestyle=style.linestyle,
        linewidth=2.2 if free else 1.4,
        alpha=0.5 if free else 1.0,
        zorder=2,
        path_effects=line_effects(style, 2.2 if free else 1.4),
    )
    ax.plot([s["cost_per_1k"] for s in line["steps"]], [s["quality"] for s in line["steps"]], style.marker,
            markersize=3.5, color=style.color, zorder=3, **marker_edge(style, 0.6))  # fmt: skip


def draw_crosses(ax: Axes, line: dict, style: RouterStyle) -> None:
    """A cross with 95% intervals at each of a line's three operating points on this panel's mix; points that coincide
    (a fall back to A) are drawn once."""
    seen: set[tuple[float, float]] = set()
    for point in line["operating_points"].values():
        spread = as_spread(point)
        quality, cost = spread.quality, spread.cost_per_1k
        if (cost.value, quality.value) in seen:
            continue
        seen.add((cost.value, quality.value))
        draw_cross(ax, spread, style, 0.8, alpha=0.6)
        ax.plot([cost.value], [quality.value], style.point_marker, markersize=5.5, color=style.color, zorder=5,
                **marker_edge(style, 0.9))  # fmt: skip


def draw_panel(ax: Axes, name: str, mix: str, panel: dict) -> None:
    ax.set_facecolor(SURFACE)
    for share, target in panel["targets"].items():
        ax.axhline(target["quality"], color=INK_SECONDARY, linestyle=":", linewidth=0.9, zorder=1)
        ax.annotate(share, (1, target["quality"]), xycoords=("axes fraction", "data"), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=7.5, color=INK_SECONDARY)  # fmt: skip
    ax.plot(*sampled(as_curve(panel["envelope"])), color=SERIES_2, linewidth=1.6, zorder=2)
    a = as_spread(panel["A"])
    ax.vlines(a.cost_per_1k.value, a.quality.low, a.quality.high, color=SERIES_1, linewidth=1, alpha=0.6, zorder=2)
    ax.hlines(a.quality.value, a.cost_per_1k.low, a.cost_per_1k.high, color=SERIES_1, linewidth=1, alpha=0.6, zorder=2)
    ax.plot([a.cost_per_1k.value], [a.quality.value], "o", markersize=6.5, color=SERIES_1, markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3)  # fmt: skip
    ax.annotate("A", (a.cost_per_1k.value, a.quality.value), xytext=(7, 7), textcoords="offset points", fontsize=8.5,
                color=INK, path_effects=[withStroke(linewidth=3, foreground=SURFACE)])  # fmt: skip
    lines = panel["routers"][name]["lines"]
    # The router alone comes first in `lines` and is drawn last, on top.
    for line in reversed(lines.values()):
        draw_line(ax, line, style_of(name, line))
    for line in reversed(lines.values()):
        draw_crosses(ax, line, style_of(name, line))
    frame_panel(ax, f"{panel['routers'][name]['label']}, {mix} mix", None, 0)
    ax.set_xlim(0.25, 16)
    ax.tick_params(labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.7)


def legend_entries() -> tuple[list[Line2D], list[str]]:
    """The one legend under every panel: the references, each router's own line, each gate's cascade line and a cross."""
    handles: list[Line2D] = [
        Line2D([], [], color=SERIES_1, marker="o", markersize=6, linestyle="none"),
        Line2D([], [], color=SERIES_2, linewidth=1.6),
        Line2D([], [], color=INK_SECONDARY, linestyle=":", linewidth=1),
    ]
    labels = [A_LABEL, ENVELOPE_LABEL, TARGETS_LABEL]
    for router in ROUTERS:
        handles.append(Line2D([], [], color=router.style.color, linestyle=router.style.linestyle, linewidth=1.2,  # type: ignore[arg-type]
                              marker=router.style.marker, markersize=4))  # fmt: skip
        labels.append(f"{LETTER_OF[router.name]} alone: its own steps (the router's own style)")
    for gate, text in ((FREE_GATE, "the free checks"), (JEV_GATE, "the Jev gate"), (SONNET_GATE, "the Sonnet gate")):
        style = CASCADE_STYLES[gate]
        # An outlined style's legend line carries its outline, as its lines do; the others' are as they were.
        outlined = {} if style.outline is None else {"path_effects": line_effects(style, 1.2), **marker_edge(style, 0)}
        handles.append(Line2D([], [], color=style.color, linestyle=style.linestyle, linewidth=1.2,  # type: ignore[arg-type]
                              marker=style.marker, markersize=4, **outlined))  # fmt: skip
        labels.append(f"Router/{GATE_LABELS[gate]}: the router's pick, then {text}, on the router's steps")
    handles.append(Line2D([], [], color=INK, marker="+", markersize=7, linestyle="none"))
    labels.append(CROSS_LABEL)
    return handles, labels


def draw_routed(panels: dict[str, dict], caption: str, path: Path) -> Figure:
    """The router-started figure from `figures/pareto_routed.json`'s `panels`, saved to `path`."""
    fig = Figure(figsize=(12, 17), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(len(PANEL_ORDER), len(panels), sharex=True)
    for row, name in enumerate(PANEL_ORDER):
        for column, (mix, panel) in enumerate(panels.items()):
            draw_panel(axes[row, column], name, mix, panel)
    for ax in axes[-1]:
        ax.set_xlabel("Cost per 1,000 requests, list price (log scale)", fontsize=9, color=INK_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mix-weighted quality", fontsize=9, color=INK_SECONDARY)
    handles, labels = legend_entries()
    caption = textwrap.fill(caption, CAPTION_WIDTH + 20)
    legend_rows = -(-len(labels) // 2)
    bottom = (LINE * (legend_rows + 1 + caption.count("\n") + 1) + 0.2) / fig.get_figheight()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, bottom), ncol=2, frameon=False, fontsize=8.5,
               labelcolor=INK)  # fmt: skip
    fig.text(0.01, 0.01, caption, fontsize=8.5, color=INK_SECONDARY, va="bottom")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    return fig


def routed_caption(basis: str, crosses: str) -> str:
    return (f"Dev split, {basis}. One panel per router and mix. Each panel draws the router alone and its three cascades "
            f"(the free checks, the Jev gate, the Sonnet gate) on the router's own steps, joined by straight lines in "
            f"linear cost in walk order from A, with a cross at each operating point (one per quality target: 100%, 99% "
            f"and 97% of A's quality, the dotted lines). {crosses} A cascade that falls back to A has its cross at A. Where two "
            f"lines coincide the later shows: the free checks' line is drawn first, wider and fainter, and the router's "
            f"own on top. A "
            f"router-started cascade's steps are its router's thresholds at the gate's fixed cutoff (from train), and its "
            f"costs include the router's overhead where it has one (G and G+: Jev's) and every gate call made. "
            f"Differences between policies are judged by the paired intervals in table v4 "
            f"(figures/table_v4.json), not by whether crosses overlap.")  # fmt: skip
