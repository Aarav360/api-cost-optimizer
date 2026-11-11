"""The calibration figure (PLAN_v4.md §9, §12; Amendment A17, items 21 and 23).

Run from the project root:  uv run python -m analysis.calibration_figure

One panel per paid gate and config: the Jev gate on all four configs, the Sonnet gate on three (it never judges `opus`,
item 7). Each draws gate score (a bin's mean) against success rate, in about ten equal-count bins whose equal scores stay
together (`analysis.gate_metrics.calibration_bins`), with the percentile 95% interval of the success rate over paired
resamples as a bar. Dev is the filled markers; train is the hollow ones (train has no `opus`). The Sonnet gate's whole-number
scores tie heavily, so it has fewer, uneven bins. A cell with no answers to plot is drawn empty and the caption says why.

Reads train's stored gate records and the train matrix (as `analysis.gate_train_report` does), dev's stored records and the
dev matrix. Never a test item, and no gate is called. Writes figures/calibration.png and figures/calibration.json, which holds
every number the figure plots. `draw_calibration` draws from the JSON's numbers alone, so a check can draw the committed ones.
"""

import json
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from analysis.bootstrap import RESAMPLES, SEED, Bootstrap
from analysis.gate_metrics import BINS, GateMetricsError, gate_calibration
from analysis.gate_table import PAID_GATES, GateScoreTable
from analysis.matrix import Matrix
from analysis.pareto import FIGURES, GRID, INK, INK_SECONDARY, SERIES_1, SURFACE
from analysis.router_names import GATE_LABELS
from harness.client import PROJECT_ROOT, load_models
from routers.ladder import FRONTIER, LADDER

CALIBRATION_FIGURE, CALIBRATION_DATA = FIGURES / "calibration.png", FIGURES / "calibration.json"
SPLITS = ("dev", "train")
CAPTION_WIDTH, LINE = 150, 1 / 6
DEV_LABEL = "Dev: bin mean gate score against success rate, with 95% intervals"
TRAIN_LABEL = "Train (hollow): the same on train, which has no opus answers"
DIAGONAL_LABEL = "Perfect calibration: success rate equal to gate score"

UNITS = {
    "gates": "per paid gate (keyed by its name) and config: for each split (dev, train) the bins, lowest score first, or null "
    "where the gate judged none of the config's answers on that split, with `empty` giving the reason for each split that has none",
    "bin": "size (answers in the bin), first and last (its lowest and highest gate score), mean_score (the mean gate score, "
    "the x), success_rate (the share of the bin's answers that were right, the y), low and high (the percentile 95% "
    "interval of the success rate over the paired resamples that draw the bin; null if none does) and undefined (the "
    "share of resamples that draw none of the bin's answers)",
    "binning": f"about {BINS} equal-count bins in score order; equal scores are never split across bins (A17 item 21), so "
    "heavily tied scores (the Sonnet gate's whole numbers) give fewer and uneven bins",
    "answers": "each paid gate is measured on the answers that pass the free checks; a refused call has no gate score and is "
    "left out of the bins",
}


def empty_reason(gate: str, config: str, split: str) -> str:
    """Why a (gate, config, split) cell has nothing to plot, or "" if it should have something."""
    if config == FRONTIER and split == "train":
        return "train has no opus answers (A1 item 3)"
    if config == FRONTIER and gate == PAID_GATES[1]:
        return "the Sonnet gate judges no opus answer (A17 item 7)"
    return ""


def cell_bins(
    draws: Bootstrap, table: GateScoreTable, matrix: Matrix, gate: str, config: str, split: str
) -> list[dict[str, Any]] | None:
    """One cell's bins as JSON-ready dicts, or None where the gate judged none of the config's answers (and the reason is
    one `empty_reason` knows: anything else is a bug and raises)."""
    try:
        return [asdict(b) for b in gate_calibration(draws, table, matrix, gate, config)]
    except GateMetricsError:
        if not empty_reason(gate, config, split):
            raise
        return None


def calibration_data(splits: dict[str, tuple[Bootstrap, GateScoreTable, Matrix]], snapshot: str = "") -> dict[str, Any]:
    """The figure's numbers: for each paid gate and config, dev's bins and train's (`splits` maps each split's name to its
    master resamples, gate-score table and matrix)."""
    gates: dict[str, Any] = {}
    for gate in PAID_GATES:
        gates[gate] = {}
        for config in LADDER:
            cell: dict[str, Any] = {}
            reasons = {}
            for split in SPLITS:
                draws, table, matrix = splits[split]
                cell[split] = cell_bins(draws, table, matrix, gate, config, split)
                if cell[split] is None:
                    reasons[split] = empty_reason(gate, config, split)
            cell["empty"] = reasons
            gates[gate][config] = cell
    return {
        "figure": CALIBRATION_FIGURE.name,
        "built_by": "analysis/calibration_figure.py",
        "price_snapshot": snapshot,
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "interval": "percentile 95%"},
        "units": UNITS,
        "gates": gates,
    }


def empty_cells(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(gate, config, reason) for every cell with no dev bins, in the figure's order."""
    return [
        (gate, config, cell["empty"]["dev"])
        for gate, by_config in data["gates"].items()
        for config, cell in by_config.items()
        if cell["dev"] is None
    ]


def caption_of(data: dict[str, Any]) -> str:
    """The figure's caption, with the reason for each empty cell and for each config train has no markers for."""
    empties = empty_cells(data)
    no_train = sorted({config for by_config in data["gates"].values() for config, cell in by_config.items()
                       if cell["train"] is None})  # fmt: skip
    gate_names = {gate: GATE_LABELS[gate] for gate in data["gates"]}
    why = "; ".join(
        f"the {gate_names[gate]} gate on {config} is empty because {reason}" for gate, config, reason in empties
    )
    text = (
        f"Dev split (filled markers) with train (hollow), run 0. Each paid gate is measured on the answers that pass the "
        f"free checks, in about {BINS} equal-count bins of its gate score; equal scores stay in one bin, so the Sonnet gate's "
        f"whole-number scores give fewer, uneven bins. Bars are 95% intervals from {data['bootstrap']['resamples']:,} paired "
        f"resamples of source groups within each task; a bin that no resample draws has none."
    )
    if empties:
        text += f" Empty cell: {why}."
    if no_train:
        text += f" Train has no markers for {', '.join(no_train)}: train has no opus answers (A1 item 3)."
    return text


def draw_calibration(data: dict[str, Any], path: Path, caption: str | None = None) -> Figure:
    """The calibration figure from `calibration_data`'s numbers (or the committed JSON), saved to `path`."""
    gates = data["gates"]
    fig = Figure(figsize=(12, 6.8), dpi=170, facecolor=SURFACE)
    axes = fig.subplots(len(gates), len(LADDER), sharex=True, sharey=True)
    for row, (gate, by_config) in enumerate(gates.items()):
        for column, (config, cell) in enumerate(by_config.items()):
            ax = axes[row, column]
            ax.set_facecolor(SURFACE)
            ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linestyle=":", linewidth=1, zorder=1)
            for split, filled in (("train", False), ("dev", True)):
                bins = cell[split]
                if not bins:
                    continue
                xs, ys = [b["mean_score"] for b in bins], [b["success_rate"] for b in bins]
                bars = [(b["mean_score"], b["low"], b["high"]) for b in bins if b["low"] is not None]
                ax.vlines([x for x, _, _ in bars], [lo for _, lo, _ in bars], [hi for _, _, hi in bars],
                          color=SERIES_1, linewidth=1.0, alpha=0.9 if filled else 0.5, zorder=2)  # fmt: skip
                ax.plot(xs, ys, "o", markersize=5.5, color=SERIES_1, markerfacecolor=SERIES_1 if filled else SURFACE,
                        markeredgecolor=SERIES_1, markeredgewidth=1.3, zorder=4 if filled else 3)  # fmt: skip
            if cell["dev"] is None:
                ax.text(0.5, 0.5, "Empty:\n" + textwrap.fill(cell["empty"]["dev"], 30), ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color=INK_SECONDARY,
                        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 3})  # fmt: skip
            ax.set_xlim(0, 1.03)
            ax.set_ylim(0, 1.03)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.set_title(f"{GATE_LABELS[gate]} gate, {config}", loc="left", fontsize=10, color=INK)
            if column == 0:
                ax.set_ylabel("Success rate in the bin", fontsize=9, color=INK_SECONDARY)
            if row == len(gates) - 1:
                ax.set_xlabel("Gate score (bin mean)", fontsize=9, color=INK_SECONDARY)
    handles = [
        Line2D([], [], color=SERIES_1, marker="o", markersize=6, linestyle="none"),
        Line2D([], [], color=SERIES_1, marker="o", markersize=6, markerfacecolor=SURFACE, linestyle="none"),
        Line2D([], [], color=INK_SECONDARY, linestyle=":", linewidth=1),
    ]
    text = textwrap.fill(caption or caption_of(data), CAPTION_WIDTH)
    bottom = (LINE * (1 + 1 + text.count("\n") + 1) + 0.2) / fig.get_figheight()
    fig.legend(handles, [DEV_LABEL, TRAIN_LABEL, DIAGONAL_LABEL], loc="upper center", bbox_to_anchor=(0.5, bottom),
               ncol=3, frameon=False, fontsize=8, labelcolor=INK)  # fmt: skip
    fig.text(0.01, 0.01, text, fontsize=8.5, color=INK_SECONDARY, va="bottom")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    return fig


def real_splits() -> dict[str, tuple[Bootstrap, GateScoreTable, Matrix]]:
    """Dev's and train's resamples, gate-score tables and matrices, from the stored records (about 20 s)."""
    from analysis import gate_dev_run, gate_train_report
    from analysis.bootstrap import resample
    from analysis.matrix import load

    train = gate_train_report.load()
    dev = load().split("dev")
    return {
        "dev": (resample(dev), gate_dev_run.dev_table(), dev),
        "train": (resample(train.matrix), train.table, train.matrix),
    }


def main() -> int:
    data = calibration_data(real_splits(), load_models()["price_snapshot"])
    FIGURES.mkdir(exist_ok=True)
    CALIBRATION_DATA.write_text(json.dumps(data, indent=2) + "\n")
    draw_calibration(data, CALIBRATION_FIGURE)
    print(f"Wrote {CALIBRATION_FIGURE.relative_to(PROJECT_ROOT)} and {CALIBRATION_DATA.relative_to(PROJECT_ROOT)}")
    for gate, config, why in empty_cells(data):
        print(f"empty: {GATE_LABELS[gate]} gate on {config}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
