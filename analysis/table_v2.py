"""Table v2 and the learning-curve figure: Stage 5's additions on dev (PLAN_v4.md §12; Amendment A11, items 13, 16, 21).

Run from the project root:  uv run python -m analysis.table_v2

Table v2 is table v1 (`analysis/pareto.py`, which now carries F beside E: F's steps, operating points, McNemar rows
and matched-cost comparison under the key `trained`, exactly as E's) plus what this script writes to
`figures/table_v2.json`, with every v1 key and value as committed (A9 item 18's check, `check_stage4`, keeps passing):

- **same-target comparisons** (item 13): F against E, F-train against E and F against F-train, at each mix and quality
  target, paired differences in cost per 1k and in quality. They are descriptive, with no multiplicity correction: at
  95%, about one interval in twenty excludes 0 by chance;
- **diagnostics, labeled "diagnostic, dev only"** (item 12): F-train's and the task-rates diagnostic's operating
  points, within-task concordance and log loss (pooled and per task) with their paired differences, each fold model's
  router-score quantiles, and the learning curve's numbers. None is a policy and none enters the operating-points
  file: their walks live in memory and only their rows are written here.

It also draws the **learning-curve figure** (`figures/learning_curve.png`, with the plotted numbers in
`figures/learning_curve.json`): log loss and within-task concordance against the number of labels for each config,
with intervals, F-train a separate marker and the task-rates diagnostic's out-of-fold log loss and 0.5 as reference
lines (`analysis/learning_figure.py`). Only train and dev are read, never a test item.
"""

import json
import sys

from analysis import learning_curve as curves
from analysis import trained_diagnostics as diagnostics
from analysis import trained_table
from analysis.bootstrap import RESAMPLES, SEED
from analysis.learning_figure import LEARNING_FIGURE, LEARNING_FIGURE_DATA, curve_data, draw_learning_curve
from analysis.operating_points import OPERATING_POINTS_FILE
from analysis.router_panels import router_part_of_mix_panel, step_outcomes
from analysis.rules_walk import MIXES
from analysis.simulator import always, load_mixes, simulate
from harness.client import PROJECT_ROOT, load_models
from routers.ladder import FRONTIER, LADDER, SCORED

FIGURES = PROJECT_ROOT / "figures"
TABLE_V2 = FIGURES / "table_v2.json"
DIAGNOSTIC_LABEL = "diagnostic, dev only"
DESCRIPTIVE = ("descriptive, with no multiplicity correction: at 95%, about one interval in twenty excludes 0 by "
               "chance")  # fmt: skip


def diagnostic_operating_points(result: diagnostics.Diagnostics) -> dict[str, dict]:
    """F-train's and the task-rates diagnostic's operating points, per mix and quality target: the rows of table v1
    for a router, without the steps, since a diagnostic is no policy and gets no line on any figure."""
    dev, draws, mixes = result.dev, result.draws, load_mixes()
    fixed = {config: simulate(dev, always(dev, config)) for config in LADDER}
    savings = {mix: draws.savings(dev, fixed[FRONTIER], mixes[mix]) for mix in MIXES}
    rows = {}
    for name in (diagnostics.F_TRAIN, diagnostics.TASK_RATES):
        steps = step_outcomes(dev, result.tables[name], fixed[FRONTIER])
        rows[name] = {mix: router_part_of_mix_panel(draws, savings[mix], steps, result.walks[name][mix], fixed,
                                                    mixes[mix])["operating_points"] for mix in MIXES}  # fmt: skip
    return rows


def fold_quantile_data(result: diagnostics.Diagnostics) -> dict[str, dict]:
    """Each fold model's router scores on its held-out items, by quantile, from F's out-of-fold table (item 9)."""
    found = trained_table.fold_quantiles(result.tables[diagnostics.F], result.fold_of)
    return {str(k): {"items": reading["items"],
                     "by_config": {c: {f"{q:g}": v for q, v in by_q.items()} for c, by_q in reading["by_config"].items()}}
            for k, reading in found.items()}  # fmt: skip


UNITS = {
    "pareto_json": "figures/pareto.json and figures/pareto_tasks.json hold table v1 for E (key `rules`) and, with the "
    "same keys and meanings, for F (key `trained`); their `units` describe E's entries in E's words",
    "comparisons": f"same-target comparisons, {DESCRIPTIVE}",
    "same_target": "keyed by `first less second`, then mix, then quality target: the first router's operating point "
    "less the second's, paired over the same resamples. cost_per_1k: dollars per 1,000 requests; "
    "quality: 0 to 1 under the mix. Each is value, low, high (percentile 95%), `undefined` 0. Where "
    'either side falls back to A, that side is A; where both do, `reading` says "both fall back to '
    'A" and both estimates are null',
    "diagnostics": f'"{DIAGNOSTIC_LABEL}": none is a policy, selects a setting or threshold, enters the '
    f"operating-points file or runs on test",
    "operating_points": "F-train's and the task-rates diagnostic's operating points per mix and quality target, as "
    "table v1 gives a router's (walked like any router, in memory only)",
    "within_task_concordance": "per router and config: pooled over pairs within a task and fold (F, task rates) or a "
    "task (F-train), ties half, with its paired 95% interval (value, low, high; `undefined`, "
    "the share of resamples with no pair, left out), and per task with A7 item 2's "
    "warning; size is the number of pairs. The task-rates diagnostic is 0.5 by "
    "construction",
    "log_loss": "per router and config: the mean log loss of the router scores against run-0 success, out of fold "
    "where cross-fitted, with the same readings as concordance; size is the number of items",
    "concordance_differences": "first less second, per config, paired over the same resamples",
    "log_loss_differences": "first less second, per config, paired over the same resamples",
    "fold_quantiles": "each fold model's router scores on its held-out dev items, by quantile (item 9)",
    "learning_curve": "figures/learning_curve.json's numbers: per config and share of each fold model's training "
    "rows, `labels` (their mean count), log loss and within-task concordance with paired "
    "intervals; `f_train`, a separate point; `headline`, 100% less 75% (larger less smaller)",
}


def build() -> tuple[dict, dict]:
    """Table v2's data and the learning-curve figure's, from the dev matrix and the committed files. Nothing is
    written to the operating-points file."""
    result = diagnostics.compute()
    curve = curves.learning_curve()
    snapshot = load_models()["price_snapshot"]
    data = diagnostics.to_data(result)
    learning = curves.as_data(curve)
    meta = {"built_by": "analysis/table_v2.py", "split": "dev", "run": 0, "n": len(result.dev.items),
            "price_snapshot": snapshot,
            "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "interval": "percentile 95%",
                          "resampled": "source groups within each task, the mix weights applied afterwards"}}  # fmt: skip
    table = {
        "figure": TABLE_V2.name, **meta, "units": UNITS,
        "comparisons": {"description": DESCRIPTIVE, "same_target": data["same_target"]},
        "diagnostics": {
            "label": DIAGNOSTIC_LABEL,
            "operating_points": diagnostic_operating_points(result),
            **{key: data[key] for key in ("within_task_concordance", "log_loss", "concordance_differences",
                                          "log_loss_differences")},
            "fold_quantiles": fold_quantile_data(result),
            "learning_curve": learning,
        },
    }  # fmt: skip
    figure = {"figure": LEARNING_FIGURE.name, **meta, **curve_data(learning, result.log_loss[diagnostics.TASK_RATES])}
    return table, figure


def print_summary(table: dict) -> None:
    print(f"\nTable v2 additions ({DIAGNOSTIC_LABEL}, except the comparisons), written to "
          f"{TABLE_V2.relative_to(PROJECT_ROOT)}. The comparisons are {DESCRIPTIVE}.")  # fmt: skip
    for pair, by_mix in table["comparisons"]["same_target"].items():
        print(f"\n{pair}")
        for mix, by_share in by_mix.items():
            for share, cell in by_share.items():
                if cell["reading"]:
                    print(f"  {mix:9} {share:4}: {cell['reading']}")
                    continue
                cost, quality = cell["cost_per_1k"], cell["quality"]
                print(f"  {mix:9} {share:4}: cost {cost['value']:+.2f} ({cost['low']:+.2f} to {cost['high']:+.2f}); "
                      f"quality {100 * quality['value']:+.1f} ({100 * quality['low']:+.1f} to "
                      f"{100 * quality['high']:+.1f}) pts")  # fmt: skip
    for name, by_mix in table["diagnostics"]["operating_points"].items():
        print(f"\nOperating points of {name} ({DIAGNOSTIC_LABEL})")
        for mix, by_share in by_mix.items():
            for share, p in by_share.items():
                where = "falls back to A" if p["fell_back"] else f"threshold {p['threshold']:.4f}"
                print(f"  {mix:9} {share:4}: {where}; quality {100 * p['quality']['value']:.1f}%, "
                      f"${p['cost_per_1k']['value']:.2f} per 1k")  # fmt: skip
    print(f"\nWithin-task concordance, pooled ({DIAGNOSTIC_LABEL}; 95% paired intervals)")
    for name, by_config in table["diagnostics"]["within_task_concordance"].items():
        cells = []
        for config in SCORED:
            e = by_config[config]["pooled"]
            cells.append(f"{config} {e['value']:.3f} ({e['low']:.3f} to {e['high']:.3f})")
        print(f"  {name:10} " + "; ".join(cells))
    print(f"\nLearning curve, 100% less 75% of the training rows ({DIAGNOSTIC_LABEL}; paired 95% intervals)")
    for config, step in table["diagnostics"]["learning_curve"]["headline"].items():
        loss, conc = step["log_loss"]["pooled"], step["concordance"]["pooled"]
        print(f"  {config:9} log loss {loss['value']:+.4f} ({loss['low']:+.4f} to {loss['high']:+.4f}); concordance "
              f"{conc['value']:+.4f} ({conc['low']:+.4f} to {conc['high']:+.4f})")  # fmt: skip


def main() -> int:
    table, figure = build()
    FIGURES.mkdir(exist_ok=True)
    for path, data in ((TABLE_V2, table), (LEARNING_FIGURE_DATA, figure)):
        path.write_text(json.dumps(data, indent=2) + "\n")
    draw_learning_curve(figure, LEARNING_FIGURE)
    print_summary(table)
    print(f"\nF's section stays in {OPERATING_POINTS_FILE.relative_to(PROJECT_ROOT)}; nothing else is written there.")
    print(f"Wrote {TABLE_V2.relative_to(PROJECT_ROOT)}, {LEARNING_FIGURE.relative_to(PROJECT_ROOT)}, "
          f"{LEARNING_FIGURE_DATA.relative_to(PROJECT_ROOT)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
