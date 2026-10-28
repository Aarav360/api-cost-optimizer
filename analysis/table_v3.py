"""Table v3: Stage 6's operating points, comparisons and diagnostics on dev (PLAN_v4.md §12; Amendment A15, items 14, 15
and 20; issue #42).

Run from the project root:  uv run python -m analysis.table_v3

Table v3 is table v2 (`analysis/table_v2.py`) plus what this script writes to `figures/table_v3.json`, a new file, so
table v1's and v2's keys stay as committed (`check_stage4`'s and `check_stage5`'s checks keep passing). It holds, for G
and for G+, from `analysis/jev_comparisons.py` and the walks (`analysis/jev_walk.py`):

- the **operating points** per mix and quality target, the rows table v1 gives a router (the same ones
  `figures/pareto.json` holds under `jev` and `jev_bolstered`, walked from the same sections);
- the **comparisons** (item 14), descriptive with no multiplicity correction: same-target (G against E and F; G+
  against E, F-train, F and G, first less second), the matched-cost difference from the envelope, and per-task
  McNemar rows against A;
- the **diagnostics, labeled "diagnostic, dev only"** (as table v2 labels F's): within-task concordance per config and
  G+ less G, and router-score quantiles, which read no outcome. None is a policy, and none selects a setting;
- the **dev latency** (item 10) with its retries and failed attempts as counts beside it, and the **dev calls**: each
  router's first call's server `Date`, which item 20 records and `analysis/check_stage6.py` compares.

Only train and dev are read, never a test item, and no Jev call is made: the stored dev records and the dev matrix.
"""

import email.utils
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from analysis import collect_jev
from analysis import jev_comparisons as comparisons
from analysis import jev_train_report as report
from analysis.bootstrap import RESAMPLES, SEED
from analysis.router_panels import router_part_of_mix_panel, step_outcomes
from analysis.rules_walk import MIXES
from analysis.simulator import always, load_mixes, simulate
from analysis.table_v2 import DESCRIPTIVE, DIAGNOSTIC_LABEL, FIGURES
from harness.client import PROJECT_ROOT, load_models
from routers.ladder import FRONTIER, LADDER

TABLE_V3 = FIGURES / "table_v3.json"
JEV = comparisons.JEV  # G and G+, the routers this table adds, in print order
DATE_FORMAT = "%a, %d %b %Y %H:%M:%S GMT"  # the HTTP `Date` header as A15 item 19 quotes it

UNITS = {
    "table_v1_and_v2": "figures/pareto.json, figures/pareto_tasks.json and figures/table_v2.json hold tables v1 and v2, "
    "unchanged; the figure JSONs also hold G's steps and operating points under `jev` and G+'s under `jev_bolstered`, "
    "as E's are under `rules`, the plotted numbers of chart v3",
    "operating_points": "G's and G+'s operating points per mix and quality target, keyed by the router's letter: the "
    "rows table v1 gives a router (the same ones as figures/pareto.json's), each the cheapest certified step of "
    "the walk (Amendment A9, items 9 and 10). Costs include Jev's overhead on every routed request (A15 item 10); "
    "a point that falls back to A is A itself, with no router call and no overhead",
    "comparisons": f"{DESCRIPTIVE}",
    "same_target": "keyed by `first less second`, then mix, then quality target: the first router's operating point "
    "less the second's, paired over the same resamples. cost_per_1k: dollars per 1,000 requests; quality: 0 to 1 "
    "under the mix. Each is value, low, high (percentile 95%), `undefined` 0. Where either side falls back to A, "
    'that side is A; where both do, `reading` says "both fall back to A" and both estimates are null. G against '
    "F-train is not made: G uses no labels (A15 item 14)",
    "matched_cost": "per router, mix and target: the envelope's quality at the operating point's cost (envelope), the "
    "point's paired quality difference from it (difference) with its percentile 95% interval (low, high), and "
    "unread, the share of resamples in which the point costs less than the envelope's cheapest corner "
    "(Amendment A9, items 12 and 15)",
    "mcnemar": "per router, task, mix and target: McNemar's test of the operating point against A within the task "
    "(b: items only the point gets right; c: only A; p: exact, two-sided); null where the point falls back to A",
    "diagnostics": f'"{DIAGNOSTIC_LABEL}": none is a policy, selects a setting or threshold, enters the '
    f"operating-points file or runs on test",
    "within_task_concordance": "per router and config: pooled over pairs within a task (G and G+ are each one model, so "
    "cells are the task alone), ties half, with its paired 95% interval (value, low, high; `undefined`, the share of "
    "resamples with no pair, left out), and per task with A7 item 2's warning; size is the number of pairs",
    "concordance_differences": "`G+ less G`, per config, paired over the same resamples: what our labels add",
    "router_score_quantiles": "each router's dev router scores per config at the quantiles 0, 5, 25, 50, 75, 95 and "
    "100%; they read no outcome",
    "latency": "dev, in seconds, per router: the successful attempt's median and p90 (numpy's linear percentile) over "
    "calls, per task, with the first call and every reconnect apart and the retries and failed attempts as counts "
    "(A15 item 10). It depends on the machine and the network and is quoted as measured",
    "dev_calls": "the server `Date` header of each router's first dev call and the earlier of the two "
    "(first_call_server_date), as A15 item 20 records it",
}


def first_dates() -> dict[str, str]:
    """Each router's first stored dev call's server `Date`, as the header gave it, by letter, and `first_call_server_date`
    as the earlier of the two."""
    raw = {}
    for letter, router in zip(JEV, collect_jev.ROUTERS, strict=True):
        records = collect_jev.read_records(router, "dev")
        report.first_date(records)  # refuses a first call whose Date isn't a date
        raw[letter] = str(records[0]["server_date"])
    earliest = min(raw.values(), key=lambda text: email.utils.parsedate_to_datetime(text))
    parsed: datetime = email.utils.parsedate_to_datetime(earliest)
    return {"first_call_server_date": parsed.strftime(DATE_FORMAT), **raw}


def operating_points(result: comparisons.JevComparisons) -> dict[str, dict]:
    """G's and G+'s operating points per mix and target: table v1's rows for a router, from the walks `jev_walk` gives
    (which `analysis/pareto.py` also draws from, so the two files agree)."""
    dev, draws, mixes = result.dev, result.draws, load_mixes()
    fixed = {config: simulate(dev, always(dev, config)) for config in LADDER}
    savings = {mix: draws.savings(dev, fixed[FRONTIER], mixes[mix]) for mix in MIXES}
    rows = {}
    for name in JEV:
        steps = step_outcomes(dev, result.tables[name], fixed[FRONTIER])
        walks = {mix: dict(by_share) for mix, by_share in result.walks[name].items()}
        rows[name] = {
            mix: router_part_of_mix_panel(draws, savings[mix], steps, walks[mix], fixed, mixes[mix])["operating_points"]
            for mix in MIXES
        }
    return rows


def build(result: comparisons.JevComparisons | None = None) -> dict[str, Any]:
    """Table v3's data from the dev matrix, the stored dev records and the committed files. Nothing is written to the
    operating-points file."""
    result = result or comparisons.compute()
    data = comparisons.to_data(result)
    meta = {
        "built_by": "analysis/table_v3.py", "split": "dev", "run": 0, "n": len(result.dev.items),
        "price_snapshot": load_models()["price_snapshot"],
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "interval": "percentile 95%",
                      "resampled": "source groups within each task, the mix weights applied afterwards"},
    }  # fmt: skip
    return {
        "figure": TABLE_V3.name, **meta, "units": UNITS,
        "operating_points": operating_points(result),
        "comparisons": {"description": DESCRIPTIVE,
                        **{key: data[key] for key in ("same_target", "matched_cost", "mcnemar")}},
        "diagnostics": {"label": DIAGNOSTIC_LABEL,
                        **{key: data[key] for key in ("within_task_concordance", "concordance_differences",
                                                      "router_score_quantiles")}},
        "latency": data["latency"],
        "dev_calls": first_dates(),
    }  # fmt: skip


def shown(estimate: Mapping[str, float], scale: float, places: int) -> str:
    return (f"{scale * estimate['value']:+.{places}f} ({scale * estimate['low']:+.{places}f} to "
            f"{scale * estimate['high']:+.{places}f})")  # fmt: skip


def print_summary(table: Mapping[str, Any]) -> None:
    print(f"\nTable v3, written to {TABLE_V3.relative_to(PROJECT_ROOT)}. The comparisons are {DESCRIPTIVE}.")
    for name, by_mix in table["operating_points"].items():
        print(f"\nOperating points of {name}")
        for mix, by_share in by_mix.items():
            for share, p in by_share.items():
                where = "falls back to A" if p["fell_back"] else f"threshold {p['threshold']:.4f}"
                print(f"  {mix:9} {share:4}: {where}; quality {100 * p['quality']['value']:.1f}%, "
                      f"${p['cost_per_1k']['value']:.2f} per 1k, savings against A "
                      f"{p['savings_vs_A']['share']:.1%}, captured {p['captured']['value']:.2f}, "
                      f"{p['captured']['met']:.1%} of resamples meet the target")  # fmt: skip
    for pair, by_mix in table["comparisons"]["same_target"].items():
        print(f"\n{pair}")
        for mix, by_share in by_mix.items():
            for share, cell in by_share.items():
                if cell["reading"]:
                    print(f"  {mix:9} {share:4}: {cell['reading']}")
                    continue
                print(f"  {mix:9} {share:4}: cost {shown(cell['cost_per_1k'], 1, 2)}; "
                      f"quality {shown(cell['quality'], 100, 1)} pts")  # fmt: skip
    print(f"\nWithin-task concordance, pooled ({DIAGNOSTIC_LABEL}; 95% paired intervals)")
    for name, by_config in table["diagnostics"]["within_task_concordance"].items():
        cells = "; ".join(f"{c} {by_config[c]['pooled']['value']:.3f}" for c in by_config)
        print(f"  {name:3} {cells}")
    print(f"\nFirst dev call, server Date: {table['dev_calls']['first_call_server_date']}")


def main(argv: Sequence[str] = ()) -> int:
    table = build()
    FIGURES.mkdir(exist_ok=True)
    TABLE_V3.write_text(json.dumps(table, indent=2) + "\n")
    print_summary(table)
    print(f"\nWrote {TABLE_V3.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
