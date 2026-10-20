"""Pareto chart v0 on dev (PLAN_v4.md §12; Amendment A8, item 10).

Run from the project root:  uv run python -m analysis.pareto

Evaluates the four fixed configs as policies through the simulator, on dev (run 0),
and totals them under both task mixes, with reference C, the random-mixture envelope
of their points (Amendment A8, item 7), and reference D, the hindsight oracle, under
each mix's item weights (item 6). The paired bootstrap (items 8 and 9) gives each
config's totals their intervals, and its quality difference from A and its safety
margin. Prints the totals with their intervals, differences and margins, the
envelope's corners and the oracle's range, and its cost and routing mix at each
quality target (§11). Draws the headline figure: a realistic and a uniform panel, cost
per 1,000 requests at list price on a log x-axis and mix-weighted quality on y, each
config a point with its intervals as a cross, and dotted lines at the targets.
Writes figures/pareto.png and, with every plotted number, figures/pareto.json. Both
are committed and overwritten as later stages add to the chart.
"""

import json
import sys
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator, PercentFormatter

from analysis.bootstrap import RESAMPLES, SEED, Estimate, TotalEstimate, resample
from analysis.matrix import LADDER, Matrix, load
from analysis.references import START, Curve, Oracle, envelope, oracle, quality_targets
from analysis.simulator import Total, always, item_weights, load_mixes, simulate, totals
from harness.client import PROJECT_ROOT, load_models
from harness.run_pilot import print_table

FIGURES = PROJECT_ROOT / "figures"
FIGURE, FIGURE_DATA = FIGURES / "pareto.png", FIGURES / "pareto.json"  # the headline figure and its plotted numbers
PANELS = ("realistic", "uniform")
LABELS = {"haiku": "B: haiku", "sonnet": "sonnet", "opus_low": "opus_low", "opus": "A: opus"}
# Label offsets in points, so opus_low and opus, close together on both axes, don't collide, and haiku's label sits
# below the envelope, which leaves haiku's point rising to the right.
OFFSETS = {"haiku": (8, -9, "left"), "sonnet": (8, 0, "left"), "opus_low": (-8, 6, "right"), "opus": (8, -9, "left")}

# The data-viz reference palette's light surface and inks; the config points are series 1, the envelope series 2
# and the oracle series 3.
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES_1, SERIES_2, SERIES_3 = "#2a78d6", "#eb6834", "#1baf7a"
ENVELOPE_LABEL = "C: random-mixture envelope"
ORACLE_LABEL = "D: hindsight oracle (optimistic upper bound)"  # run 0 only, its flip rates over 5% (A6 item 2)
TARGETS_LABEL = "Quality targets, % of A's quality"


def config_points(dev: Matrix, mixes: dict) -> dict[str, dict[str, Total]]:
    """Each fixed config's totals under each mix: {mix: {config: Total}}."""
    outcomes = {config: simulate(dev, always(dev, config)) for config in LADDER}
    return {mix: {config: totals(outcomes[config], mixes[mix]) for config in LADDER} for mix in PANELS}


def shown(estimate: Estimate, scale: float, form: str) -> str:
    """An estimate and its interval, scaled and formatted alike: 83.3 (78.9 to 87.6)."""
    value, low, high = (form.format(scale * n) for n in (estimate.value, estimate.low, estimate.high))
    return f"{value} ({low} to {high})"


def sampled(curve: Curve) -> tuple[list[float], list[float | None]]:
    """Points along a reference curve for the log cost axis, where its straight segments in linear cost bend.

    Each segment gets 100 evenly spaced samples on the log axis, so a short one is drawn as finely as a long one.
    """
    corners = list(curve.corners.values())
    costs = [corners[0].cost_per_1k]
    for low, high in pairwise(corners):
        costs += [float(cost) for cost in np.geomspace(low.cost_per_1k, high.cost_per_1k, 100)[1:]]
    return costs, [curve.quality_at(cost) for cost in costs]


def draw(points: dict[str, dict[str, Total]], spreads: dict[str, dict[str, TotalEstimate]],
         envelopes: dict[str, Curve], oracles: dict[str, Oracle], targets: dict[str, dict[float, float]], caption: str,
         path: Path) -> None:  # fmt: skip
    fig = Figure(figsize=(10, 5.2), dpi=200, facecolor=SURFACE)
    for ax, mix in zip(fig.subplots(1, 2), PANELS):
        ax.set_facecolor(SURFACE)
        # Each target's share sits in the right margin, clear of the curves, which all cross the targets further left.
        for i, (share, quality) in enumerate(targets[mix].items()):
            ax.axhline(quality, color=INK_SECONDARY, linestyle=":", linewidth=1, zorder=1,
                       label=TARGETS_LABEL if i == 0 else None)  # fmt: skip
            ax.annotate(f"{share:.0%}", (1, quality), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=8, color=INK_SECONDARY)  # fmt: skip
        ax.plot(*sampled(envelopes[mix]), color=SERIES_2, linewidth=1.8, zorder=2, label=ENVELOPE_LABEL)
        ax.plot(*sampled(oracles[mix].curve), color=SERIES_3, linewidth=1.8, zorder=2, label=ORACLE_LABEL)
        # Each arm spans its interval, which comes from the resamples and needn't be centred on the point.
        for config, total in points[mix].items():
            spread = spreads[mix][config]
            ax.vlines(total.cost_per_1k, spread.quality.low, spread.quality.high, color=SERIES_1, linewidth=1,
                      alpha=0.6, zorder=2)  # fmt: skip
            ax.hlines(total.quality, spread.cost_per_1k.low, spread.cost_per_1k.high, color=SERIES_1, linewidth=1,
                      alpha=0.6, zorder=2)  # fmt: skip
        ax.plot([total.cost_per_1k for total in points[mix].values()], [total.quality for total in points[mix].values()],
                "o", markersize=7, color=SERIES_1, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                label="Fixed configs, with 95% intervals")  # fmt: skip
        for config, total in points[mix].items():
            dx, dy, ha = OFFSETS[config]
            ax.annotate(LABELS[config], (total.cost_per_1k, total.quality), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=9, color=INK)  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlim(0.2, 25)
        ax.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
        # Whole-point steps: the crosses widen the range enough for 2.5-point ticks, which whole percents misprint.
        ax.yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 5, 10]))
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
    # One legend under both panels, which plot the same series; inside a panel it would cover a curve. The target
    # lines are drawn first, under the curves, but listed last.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="lower center", bbox_to_anchor=(0.5, 0.04),
               ncol=2, frameon=False, fontsize=8.5, labelcolor=INK)  # fmt: skip
    fig.text(0.01, 0.01, caption, fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, facecolor=SURFACE)


def main() -> int:
    dev = load().split("dev")
    snapshot = load_models()["price_snapshot"]
    mixes = load_mixes()
    points = config_points(dev, mixes)
    envelopes = {mix: envelope(points[mix]) for mix in PANELS}
    oracles = {mix: oracle(dev, item_weights(dev.items.values(), mixes[mix])) for mix in PANELS}
    targets = {mix: quality_targets(points[mix]["opus"].quality) for mix in PANELS}
    # One set of resamples serves every config and mix, so each difference from A is paired (A8 item 8).
    draws = resample(dev)
    outcomes = {config: simulate(dev, always(dev, config)) for config in LADDER}
    spreads = {mix: {c: draws.totals(outcomes[c], mixes[mix]) for c in LADDER} for mix in PANELS}
    differences = {mix: {c: draws.difference(outcomes[c], outcomes["opus"], mixes[mix]) for c in LADDER}
                   for mix in PANELS}  # fmt: skip
    margins = {mix: {c: draws.safety_margin(outcomes[c], outcomes["opus"], mixes[mix]) for c in LADDER}
               for mix in PANELS}  # fmt: skip
    basis = f"run 0, n = {len(dev.items)}, list prices {snapshot}"

    print(f"Fixed configs on dev ({basis}), with percentile 95% intervals from {RESAMPLES:,} paired bootstrap "
          f"resamples (seed {SEED}). The safety margin is one standard error of the quality difference from A.")  # fmt: skip
    for mix in PANELS:
        rows = [[LABELS[c], f"{shown(spreads[mix][c].quality, 100, '{:.1f}')}%",
                 f"${shown(spreads[mix][c].cost_per_1k, 1, '{:.2f}')}",
                 f"{shown(differences[mix][c], 100, '{:+.1f}')} pts", f"{100 * margins[mix][c]:.2f} pts"]
                for c in LADDER]  # fmt: skip
        print(f"\n{mix.capitalize()} mix")
        print_table(["config", "quality", "$ per 1k", "difference from A", "safety margin"], rows)
    for mix, curve in envelopes.items():
        corners = ", ".join(f"{config} ({100 * total.quality:.1f}%, ${total.cost_per_1k:.2f})"
                            for config, total in curve.corners.items())  # fmt: skip
        print(f"{ENVELOPE_LABEL}, {mix} mix: {corners}")
    for mix, solved in oracles.items():
        low, high = solved.curve.corners[START], list(solved.curve.corners.values())[-1]
        print(f"\n{ORACLE_LABEL}, {mix} mix: from {100 * low.quality:.1f}% at ${low.cost_per_1k:.2f} per 1k, every "
              f"item on its cheapest config, up to {100 * high.quality:.1f}% at ${high.cost_per_1k:.2f}")  # fmt: skip
        rows = []
        for share, quality in targets[mix].items():
            cost, routing = solved.cost_at(quality), solved.routing_at(quality)
            rows.append([f"{share:.0%} of A", f"{100 * quality:.1f}%", "n/a (infeasible)" if cost is None else f"${cost:.2f}",
                         *(f"{100 * routing[c]:.1f}%" if routing else "n/a" for c in LADDER)])  # fmt: skip
        print_table(["quality target", "quality", "oracle $ per 1k", *LADDER], rows)

    units = {
        "quality": "mix-weighted share of items answered correctly, 0 to 1",
        "cost_per_1k": "mix-weighted dollars per 1,000 requests, at list price",
        "envelope": "the envelope's corners, cheapest first, joined by straight lines in linear cost",
        "oracle": f"the oracle's corners: {START}, then the point after each upgrade, in the order taken, joined "
                  f"by straight lines in linear cost",
        "targets": "quality targets, as shares of A's (opus's) quality under the mix",
        "intervals": "each config's quality and cost per 1k with its percentile 95% interval (low, high) and standard "
                     "error, the standard deviation over the resamples",
        "difference_from_A": "each config's quality less A's, paired: the same resamples weigh both; its "
                             "standard_error is the config's safety margin (Amendment A8, item 9)",
    }  # fmt: skip
    panels = {mix: {"configs": {c: asdict(t) for c, t in points[mix].items()},
                    "envelope": {c: asdict(t) for c, t in envelopes[mix].corners.items()},
                    # A list, not keyed by the upgrade's item: the figures hold aggregates of dev only (A8 item 10).
                    "oracle": [asdict(t) for t in oracles[mix].curve.corners.values()],
                    "targets": {f"{share:.0%}": quality for share, quality in targets[mix].items()},
                    "intervals": {c: asdict(spread) for c, spread in spreads[mix].items()},
                    "difference_from_A": {c: asdict(difference) for c, difference in differences[mix].items()}}
              for mix in PANELS}  # fmt: skip
    bootstrap = {"resamples": RESAMPLES, "seed": SEED, "interval": "percentile 95%",
                 "resampled": "source groups within each task, the mix weights applied afterwards"}  # fmt: skip
    data = {"figure": FIGURE.name, "built_by": "analysis/pareto.py", "split": "dev", "run": 0, "n": len(dev.items),
            "price_snapshot": snapshot, "bootstrap": bootstrap, "units": units, "panels": panels}  # fmt: skip
    FIGURES.mkdir(exist_ok=True)
    FIGURE_DATA.write_text(json.dumps(data, indent=2) + "\n")
    caption = (f"Dev split, {basis}. Crosses are 95% intervals from {RESAMPLES:,} resamples of source groups within "
               f"each task.")  # fmt: skip
    draw(points, spreads, envelopes, oracles, targets, caption, FIGURE)
    print(f"\nWrote {FIGURE.relative_to(PROJECT_ROOT)} and {FIGURE_DATA.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
