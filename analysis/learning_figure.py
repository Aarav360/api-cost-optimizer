"""The learning-curve figure (PLAN_v4.md §12; Amendment A11, items 12 and 16).

Draws from `figures/learning_curve.json`'s numbers alone, so a check can draw the committed ones without `results/`.
Per config, log loss (top row) and within-task concordance (bottom row) against the number of labels, with 95% paired
intervals. F's points are joined by a line, in F's colour; F-train is a separate marker in another colour and shape,
never joined, since it is a different design (one model, refit on train alone). The reference lines say how far each
point gets beyond knowing the task: the task-rates diagnostic's out-of-fold log loss (dashed) on the log-loss panels,
and 0.5 on the concordance panels, which the task-rates diagnostic scores exactly. No walk at any size: the figure
shows no policy and no quality target. `analysis/table_v2.py` writes the figure and its numbers.
"""

import textwrap
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from analysis.chart_routers import F_STYLE
from analysis.pareto import GRID, INK, INK_SECONDARY, SERIES_1, SURFACE
from analysis.router_metrics import Reading, estimate_data
from harness.client import PROJECT_ROOT
from routers.ladder import SCORED

LEARNING_FIGURE = PROJECT_ROOT / "figures" / "learning_curve.png"
LEARNING_FIGURE_DATA = PROJECT_ROOT / "figures" / "learning_curve.json"
F_TRAIN_COLOR = SERIES_1  # F-train takes the configs' blue, in a square; F's colour is `F_STYLE.color`
TASK_RATES_LABEL = "Task-rates diagnostic's out-of-fold log loss (knows only the task)"
CHANCE_LABEL = "0.5: ordering by chance, or by the task alone"
F_LABEL = "F's fold models at 25, 50, 75 and 100% of their training rows, 95% intervals"
F_TRAIN_LABEL = "F-train (refit on train alone), a separate point, 95% interval"
CAPTION_WIDTH, LINE = 150, 1 / 6


def curve_data(learning: dict, task_rates_log_loss: dict[str, Reading]) -> dict:
    """The figure's numbers: the learning curve's (`analysis/learning_curve.py`'s `as_data`) and its two references."""
    return {
        "units": {
            "curve": "per config and share of each fold model's training rows: labels (their mean count), log loss "
                     "and within-task concordance, each pooled and per task with value, low and high (percentile "
                     "95%, paired over the same 2,000 resamples) and `undefined`; `f_train` is a separate point, "
                     "`headline` the paired difference 100% less 75%",
            "reference": "task_rates_log_loss: the task-rates diagnostic's out-of-fold log loss per config (the "
                         "dashed line); concordance: 0.5, which the task-rates diagnostic scores exactly",
        },
        "curve": learning,
        "reference": {"task_rates_log_loss": {c: estimate_data(task_rates_log_loss[c].pooled) for c in SCORED}, "concordance": 0.5},
    }  # fmt: skip


def draw_learning_curve(data: dict, path: Path, caption: str | None = None) -> Figure:
    """The learning-curve figure from `curve_data`'s numbers (or the committed JSON), saved to `path`."""
    curve, reference = data["curve"], data["reference"]
    sizes = [f"{s:g}" for s in curve["sizes"]]
    fig = Figure(figsize=(10, 6.5), dpi=200, facecolor=SURFACE)
    axes = fig.subplots(2, len(SCORED), sharex=True, sharey="row")
    for column, config in enumerate(SCORED):
        points = [curve["points"][config][s] for s in sizes]
        labels = [p["labels"] for p in points]
        f_train = curve["f_train"][config]
        for row, metric in enumerate(("log_loss", "concordance")):
            ax = axes[row, column]
            ax.set_facecolor(SURFACE)
            values = [p[metric]["pooled"] for p in points]
            yerr = [[v["value"] - v["low"] for v in values], [v["high"] - v["value"] for v in values]]
            ax.errorbar(labels, [v["value"] for v in values], yerr=yerr, color=F_STYLE.color, marker="o", markersize=5,
                        markeredgecolor=SURFACE, linewidth=1.2, capsize=3, zorder=3, label=F_LABEL)  # fmt: skip
            one = f_train[metric]["pooled"]
            ax.errorbar([f_train["labels"]], [one["value"]], yerr=[[one["value"] - one["low"]], [one["high"] - one["value"]]],
                        color=F_TRAIN_COLOR, marker="s", markersize=6, markeredgecolor=SURFACE, linestyle="none",
                        capsize=3, zorder=4, label=F_TRAIN_LABEL)  # fmt: skip
            if metric == "log_loss":
                ax.axhline(reference["task_rates_log_loss"][config]["value"], color=INK_SECONDARY, linestyle="--",
                           linewidth=1, zorder=1, label=TASK_RATES_LABEL)  # fmt: skip
            else:
                ax.axhline(reference["concordance"], color=INK_SECONDARY, linestyle=":", linewidth=1, zorder=1,
                           label=CHANCE_LABEL)  # fmt: skip
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            if row == 0:
                ax.set_title(config, loc="left", fontsize=11, color=INK)
            if column == 0:
                ax.set_ylabel("Log loss (lower is better)" if row == 0
                              else "Concordance (higher is better)", fontsize=8, color=INK_SECONDARY)  # fmt: skip
            if row == 1:
                ax.set_xlabel("Labels the fold models trained on", fontsize=9, color=INK_SECONDARY)
    wanted = {
        F_LABEL: Line2D([], [], color=F_STYLE.color, marker="o", markersize=5, linewidth=1.2),
        F_TRAIN_LABEL: Line2D([], [], color=F_TRAIN_COLOR, marker="s", markersize=6, linestyle="none"),
        TASK_RATES_LABEL: Line2D([], [], color=INK_SECONDARY, linestyle="--", linewidth=1),
        CHANCE_LABEL: Line2D([], [], color=INK_SECONDARY, linestyle=":", linewidth=1),
    }
    order = [F_LABEL, F_TRAIN_LABEL, TASK_RATES_LABEL, CHANCE_LABEL]
    text = caption or (
        "Dev split, run 0. Each fold model is refit on nested subsamples of its training rows (whole source groups, "
        "stratified by task and split), scoring its held-out fold; concordance counts pairs within a task and fold. "
        "Intervals are 95% from 2,000 paired resamples of source groups within each task. A diagnostic, dev only: "
        "no walk at any size, and nothing here selects a threshold."
    )
    text = textwrap.fill(text, CAPTION_WIDTH)
    bottom = (LINE * (4 + 1 + text.count("\n") + 1) + 0.15) / fig.get_figheight()
    fig.legend([wanted[n] for n in order], order, loc="upper center", bbox_to_anchor=(0.5, bottom), ncol=1,
               frameon=False, fontsize=8.5, labelcolor=INK)  # fmt: skip
    fig.text(0.01, 0.01, text, fontsize=8.5, color=INK_SECONDARY, va="bottom")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    return fig
