"""Table v4: Stage 7's operating points, comparisons, gate metrics and estimated latency on dev (PLAN_v4.md §9, §10, §12;
Amendment A17, items 16, 21, 22, 23 and 24; issue #71).

Run from the project root:  uv run python -m analysis.table_v4

Table v4 is table v3 (`analysis/table_v3.py`) plus what this script writes to `figures/table_v4.json`, a new file, so
tables v1 to v3 keep their keys as committed (the Stage 4 to 6 record checks keep passing). It holds:

- **the fifteen cascades' operating points** per mix and quality target, rows in table v1's style (A9 item 15) with the
  cutoff (a pure cascade) or the router's threshold and the fixed cutoff (a router-started one) and the escalation rate
  beside the routing mix;
- **the 51 comparisons** of item 22 in its five groups, descriptive with no multiplicity correction, the matched-cost
  difference from the envelope per cascade and per-task McNemar rows against A;
- **the gate metrics** of item 21: the free checks' row and each paid gate's false accept and false escalate rates beside
  raw counts, per config and task, at the fixed cutoff and at each pure cascade's operating-point cutoff, and within-task
  concordance per config;
- **the estimated p50 latency** of item 16 for every policy: the references, E to G+ and the fifteen cascades;
- **the best cascade**, chosen on dev by item 24's declared rule, and A7 item 2's warnings for IFEval and MBPP+.

Everything is read from stored train and dev gate records (through `analysis.gate_dev_run`), the dev matrix and committed
files. Only train and dev are read, never a test item, and no gate or API is called.
"""

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from analysis import cascade_comparisons as cc
from analysis import trained_diagnostics as diagnostics
from analysis.bootstrap import RESAMPLES, SEED, Bootstrap
from analysis.cascade_metrics import fixed_latency, gate_metrics_data, walk_latency
from analysis.gate_dev import cascade_walks
from analysis.gate_table import GateScoreTable
from analysis.jev_comparisons import mcnemar_rows, tasks_of
from analysis.latency_estimate import GateMedians, Seconds
from analysis.matrix import Matrix
from analysis.operating_points import target_key
from analysis.references import TARGETS
from analysis.router_names import (
    CASCADE_STARTS,
    CASCADES,
    G_PLUS,
    HAIKU,
    JEV_GATE,
    NAMED_ROUTERS,
    SONNET_GATE,
    G,
    cascade_section,
    parse_cascade,
)
from analysis.router_table import RouterScoreTable
from analysis.rules_walk import MIXES, dev_walks
from analysis.simulator import always, load_mixes, simulate
from analysis.table_v2 import DESCRIPTIVE, FIGURES
from analysis.task_warnings import task_warning
from harness.client import PROJECT_ROOT, load_models
from routers.ladder import FRONTIER, LADDER

TABLE_V4 = FIGURES / "table_v4.json"
HEADLINE = (
    (cascade_section(HAIKU, JEV_GATE), G),
    (cascade_section(HAIKU, JEV_GATE), G_PLUS),
)  # item 22: "haiku/Jev against G and against G+ matter most"

QUESTIONS = {
    "gate_adds": "What the gate adds: each router-started cascade against its router alone (12)",
    "router_adds": "What the router adds: each router-started cascade against the pure cascade with the same gate (12)",
    "paid_gate_over_free": "What a paid gate adds over the free checks: haiku/Jev and haiku/Sonnet against haiku/free, "
    "and each router's /Jev and /Sonnet against its /free (10)",
    "jev_against_sonnet": "The Jev gate against the Sonnet gate: haiku/Jev against haiku/Sonnet, and each router's /Jev "
    "against its /Sonnet (5), on the three configs both gates judge",
    "cascades_against_routers": "Cascades against routers: each pure cascade against each router alone (12); the plan's "
    "own questions of whether cascades lose to plain routing once gate costs are counted (section 18) and whether Jev "
    "fits better as a gate than as a router (section 9)",
}

RULE = (
    "Item 24: the cascade with the highest mean dev savings against A over its six operating points (two mixes, three "
    "quality targets); savings are the share of A's cost saved, a fall back to A counting 0. Ties go to a pure cascade, "
    "then to the cheaper gate (the free checks, then Jev, then Sonnet). A tie left after that is recorded, not chosen."
)

UNITS = {
    "table_v1_to_v3": "figures/pareto.json, figures/pareto_tasks.json, figures/table_v2.json and figures/table_v3.json "
    "hold tables v1 to v3, unchanged",
    "comparisons": f"{DESCRIPTIVE}. Fifty-one comparisons in each of six mix-and-target cells make about 300 intervals "
    "each for cost and quality, so the write-up leads with the declared questions (the five groups, and haiku/Jev "
    "against G and G+ first), not with whichever interval excludes 0",
    "cascades": "per cascade (keyed by its section key; label, start and gate beside), the operating points per mix and "
    "quality target: table v1's row (A9 item 15) with `threshold` (the router's, null for a pure cascade), `cutoff` (a "
    "pure cascade's walked cutoff or a router-started cascade's fixed one; null for the free checks alone and for A), "
    "`fell_back` (the walk certified nothing cheaper than A, so the point is A itself with no gate call and "
    "captured fraction 0), `escalation_rate` (share of requests under the mix whose first answer was rejected and "
    "sent to opus; routing is the share each config answered, so an item a router sent straight to opus is in "
    "routing and not in the escalation rate), quality and cost_per_1k (value, low, high, standard_error; dollars per "
    "1,000 requests at list price, gate calls included)",
    "same_target": "per group, keyed `first less second`, then mix, then quality target: the first policy's operating "
    "point less the second's, paired over the same resamples. cost_per_1k: dollars per 1,000 requests; quality: 0 to 1 "
    "under the mix. Each is value, low, high (percentile 95%), `undefined` 0. Where either side falls back to A, that "
    'side is A; where both do, `reading` says "both fall back to A" and both estimates are null',
    "matched_cost": "per cascade, mix and target: the envelope's quality at the operating point's cost (envelope), the "
    "point's paired quality difference from it (difference) with its percentile 95% interval, and unread, the share of "
    "resamples in which the point costs less than the envelope's cheapest corner (A9 items 12 and 15); the rows' own",
    "mcnemar": "per cascade, task, mix and target: McNemar's test of the operating point against A within the task (b: "
    "items only the point gets right; c: only A; p: exact, two-sided); null where the point falls back to A. "
    "IFEval and MBPP+ carry A7 item 2's warning (see task_warnings)",
    "gate_metrics": "item 21, measured on what the policy sees: a paid gate on the answers that pass the free checks, "
    "the free checks on all answers. Per config and task (`pooled` is the config over its tasks): false_accept_rate "
    "(wrong answers accepted over wrong answers) and false_escalate_rate (right answers rejected over right answers), "
    "null over nothing, and the raw counts out of all the config's answers (wrong_accepted, right_rejected; answers, "
    "gated, right, wrong). A refused call is a rejection. `operating_points` (pure cascades only) give the rates at "
    "each point's walked cutoff, null where the walk fell back to A; a router-started cascade's points all sit at "
    "`fixed_cutoff`. `within_task_concordance` is per config, pairs within a task only, ties half, with a paired 95% "
    "interval (`undefined`: the share of resamples with no pair). Opus has Jev entries and no Sonnet entry",
    "latency": "item 16, estimated p50 latency in seconds under each mix: the mix-weighted median of per-item sums along "
    "the trace of the pilot medians (A7 item 1), the router's median and each gate call's median. It is an estimate: "
    "summing medians ignores that escalated items run longer, and Stage 8's live replay is the real check. "
    "`inputs` holds the medians used",
    "best_cascade": f"item 24. {RULE}",
    "task_warnings": "A7 item 2: IFEval and MBPP+ have fewer dev items than the plan's minimum of a few hundred for a "
    "per-task claim, so their per-task entries (McNemar rows, gate rates by task, concordance) are indicative",
}


@dataclass(frozen=True)
class Inputs:
    """Everything table v4 reads: dev's matrix and resamples, dev's gate-score table, the four routers' dev router-score
    tables by section name, the fixed cutoffs, the gates' median times, the routers' median times by letter, and the
    dev items whose request has an `assert` line."""

    dev: Matrix
    draws: Bootstrap
    table: GateScoreTable
    routers: Mapping[str, RouterScoreTable]
    cutoffs: Mapping[str, float]
    medians: GateMedians
    router_seconds: Mapping[str, Seconds]
    code_items: frozenset[str]


def walks_of(inputs: Inputs) -> dict[str, cc.Walks]:
    """Every policy's walks on dev: the four routers by letter, the fifteen cascades by section key."""
    routers = {
        letter: dev_walks(inputs.dev, inputs.routers[section], inputs.draws)
        for letter, section in NAMED_ROUTERS.items()
    }
    return {**routers, **cascade_walks(inputs.dev, inputs.table, inputs.routers, dict(inputs.cutoffs), inputs.draws)}


def point_order() -> list[tuple[str, float]]:
    """A cascade's six operating points, in the order item 24's mean and the committed record read them."""
    return [(mix, share) for mix in MIXES for share in TARGETS]


def rows_of(inputs: Inputs, walks: Mapping[str, cc.Walks]) -> dict[str, dict[str, dict[str, dict]]]:
    """Each cascade's rows: key, mix, target."""
    dev, draws, mixes = inputs.dev, inputs.draws, load_mixes()
    fixed = {config: simulate(dev, always(dev, config)) for config in LADDER}
    savings = {mix: draws.savings(dev, fixed[FRONTIER], mixes[mix]) for mix in MIXES}
    return {
        key: {
            mix: {
                target_key(share): cc.policy_row(draws, savings[mix], walks[key][mix][share], fixed, mixes[mix])
                for share in TARGETS
            }
            for mix in MIXES
        }
        for key in CASCADES
    }


def best_cascade_data(rows: Mapping[str, Mapping[str, Mapping[str, dict]]]) -> dict[str, Any]:
    """Item 24's pick from the rows' savings against A, with the dollars-per-1k ranking beside it so the write-up can say
    whether the choice of measure (a share of A's cost, or dollars) matters."""
    order = [(mix, target_key(share)) for mix, share in point_order()]
    share_of = {key: [rows[key][mix][t]["savings_vs_A"]["share"] for mix, t in order] for key in CASCADES}
    dollars_of = {key: [rows[key][mix][t]["savings_vs_A"]["per_1k"] for mix, t in order] for key in CASCADES}
    found, by_dollars = cc.best_cascade(share_of), cc.best_cascade(dollars_of)
    return {
        "rule": RULE,
        "best": found.best,
        "label": None if found.best is None else CASCADES[found.best],
        "open_case": found.open_case,
        "tied_at_top": list(found.tied),
        "mean_savings_share": dict(found.means),
        "points": [f"{mix} {t}" for mix, t in order],
        "savings_share": share_of,
        "by_dollars": {
            "best": by_dollars.best,
            "open_case": by_dollars.open_case,
            "mean_savings_per_1k": dict(by_dollars.means),
        },  # fmt: skip
        "same_best_by_dollars": by_dollars.best == found.best,
        "falls_back_at": {
            key: [f"{mix} {t}" for mix, t in order if rows[key][mix][t]["fell_back"]]
            for key in CASCADES
            if any(rows[key][mix][t]["fell_back"] for mix, t in order)
        },  # fmt: skip
    }


def comparisons_data(inputs: Inputs, walks: Mapping[str, cc.Walks]) -> dict[str, Any]:
    found = cc.compare_pairs(inputs.draws, load_mixes(), walks)
    return {
        "description": DESCRIPTIVE,
        "count": sum(len(pairs) for pairs in found.values()),
        "headline": [f"{cc.label(a)} less {cc.label(b)}" for a, b in HEADLINE],
        "groups": {
            group: {
                "question": QUESTIONS[group],
                "pairs": diagnostics.same_target_data(
                    {(cc.label(a), cc.label(b)): cells for (a, b), cells in by_pair.items()}
                ),
            }
            for group, by_pair in found.items()
        },
    }


def latency_data(inputs: Inputs, walks: Mapping[str, cc.Walks]) -> dict[str, Any]:
    """Item 16's estimate for every policy at its operating point under each mix and target."""
    dev, table, mixes = inputs.dev, inputs.table, load_mixes()
    medians = inputs.medians

    def at(key: str, **kwargs: Any) -> dict[str, dict[str, float]]:
        return {
            mix: {target_key(share): walk_latency(dev, table, walks[key][mix][share], mixes[mix], **kwargs)
                  for share in TARGETS}
            for mix in MIXES}  # fmt: skip

    configs = {config: {mix: fixed_latency(dev, config, mixes[mix]) for mix in MIXES} for config in LADDER}
    return {
        "inputs": {
            "pilot_medians": "A7 item 1 (analysis.latency_estimate.PILOT_MEDIANS)",
            "code_check_s": medians.code_check,
            "jev_gate_s": medians.jev,
            "sonnet_gate_s": medians.sonnet,
            "routers_s": dict(inputs.router_seconds),
            "dev_items_with_a_code_check": len(inputs.code_items),
        },
        "references": {
            "A": configs[FRONTIER], "B": configs[LADDER[0]],
            "C": None, "D": None,
            "note": "A is always-frontier and B always-cheapest; C (a mixture of fixed configs) and D (the hindsight "
                    "oracle) are curves, not one policy's trace, and have no latency",
        },  # fmt: skip
        "configs": configs,
        "routers": {letter: at(letter, router=inputs.router_seconds[letter]) for letter in NAMED_ROUTERS},
        "cascades": {
            key: at(
                key,
                router=inputs.router_seconds[letter] if (letter := _letter_of(key)) else 0.0,
                gate=parse_cascade(key)[1], medians=medians, code_items=inputs.code_items,
            )
            for key in CASCADES
        },
    }  # fmt: skip


def _letter_of(key: str) -> str | None:
    """The router letter a router-started cascade starts at, None for a pure one."""
    start, _ = parse_cascade(key)
    return None if start == HAIKU else CASCADE_STARTS[start]


def mcnemar_data(inputs: Inputs, walks: Mapping[str, cc.Walks]) -> dict[str, Any]:
    mixes = load_mixes()
    return {
        key: {
            task: {mix: {target_key(share): row for share, row in by.items()} for mix, by in by_mix.items()}
            for task, by_mix in mcnemar_rows(inputs.dev, mixes, walks[key]).items()
        }
        for key in CASCADES
    }


def build(inputs: Inputs) -> dict[str, Any]:
    """Table v4's data from the dev inputs. Nothing is written to the operating-points file."""
    dev, draws = inputs.dev, inputs.draws
    walks = walks_of(inputs)
    rows = rows_of(inputs, walks)
    pure = {gate: walks[cascade_section(HAIKU, gate)] for gate in (JEV_GATE, SONNET_GATE)}
    items = {task: sum(1 for i in dev.items.values() if i.task == task) for task in tasks_of(dev)}
    return {
        "figure": TABLE_V4.name,
        "built_by": "analysis/table_v4.py",
        "split": "dev",
        "run": 0,
        "n": len(dev.items),
        "price_snapshot": load_models()["price_snapshot"],
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "interval": "percentile 95%",
                      "resampled": "source groups within each task, the mix weights applied afterwards"},  # fmt: skip
        "units": UNITS,
        "cascades": {
            key: {"label": CASCADES[key], "start": parse_cascade(key)[0],
                  "gate": parse_cascade(key)[1], "pure": _letter_of(key) is None, "operating_points": rows[key]}  # fmt: skip
            for key in CASCADES
        },
        "comparisons": {
            **comparisons_data(inputs, walks),
            "matched_cost": {key: {mix: {t: row["matched_cost"] for t, row in by.items()} for mix, by in by_mix.items()}
                             for key, by_mix in rows.items()},
            "mcnemar": mcnemar_data(inputs, walks),
        },
        "gate_metrics": gate_metrics_data(inputs.table, dev, draws, dict(inputs.cutoffs), pure),
        "latency": latency_data(inputs, walks),
        "best_cascade": best_cascade_data(rows),
        "task_warnings": {task: {"items": n, "warning": task_warning(task, n)} for task, n in items.items()},
    }  # fmt: skip


# ---- the real inputs ----------------------------------------------------------------------------------------------


def real_inputs() -> Inputs:
    """Dev's inputs from the stored records, the dev matrix and the committed files (about 40 s for the tables)."""
    from analysis import collect_sonnet_gate, gate_dev_run, gate_train, latency_estimate
    from analysis import matrix as results_matrix
    from analysis.benchmark import read_rows
    from analysis.bootstrap import resample
    from analysis.collect_gates import read_free_records
    from analysis.collect_jev_gate import read_records as read_jev_records
    from analysis.gate_answers import requests_of
    from routers.gates import assert_line

    dev = results_matrix.load().split("dev")
    task_of = {str(r["item_id"]): str(r["source"]) for split in ("dev", "train") for r in read_rows(split=split)}
    sonnet = gate_train.sonnet_latency(collect_sonnet_gate.read_latency_records("train"), task_of)
    medians = GateMedians(
        code_check=latency_estimate.code_check_median(read_free_records("dev")),
        jev=latency_estimate.jev_call_medians(read_jev_records("dev"), task_of),
        sonnet={task: median for task, (median, _, _) in sonnet.by_task.items()},
    )
    return Inputs(
        dev, resample(dev), gate_dev_run.dev_table(), gate_dev_run.routers_dev_tables(), gate_dev_run.read_cutoffs(),
        medians, latency_estimate.router_medians(),
        frozenset(item for item, request in requests_of("dev").items() if assert_line(request.prompt)),
    )  # fmt: skip


# ---- printing -----------------------------------------------------------------------------------------------------


def shown(estimate: Mapping[str, float], scale: float, places: int) -> str:
    return (f"{scale * estimate['value']:+.{places}f} ({scale * estimate['low']:+.{places}f} to "
            f"{scale * estimate['high']:+.{places}f})")  # fmt: skip


def print_summary(table: Mapping[str, Any]) -> None:
    print(f"\nTable v4, written to {TABLE_V4.relative_to(PROJECT_ROOT)}. The comparisons are {DESCRIPTIVE}.")
    best = table["best_cascade"]
    print(f"\nBest cascade (item 24): {best['label']}" + (" (rule left a tie open)" if best["open_case"] else ""))
    for key, mean in sorted(best["mean_savings_share"].items(), key=lambda kv: -kv[1]):
        print(f"  {CASCADES[key]:14} mean savings {mean:6.1%}")
    for group, body in table["comparisons"]["groups"].items():
        print(f"\n{body['question']}")
        for pair, by_mix in body["pairs"].items():
            for mix, by_share in by_mix.items():
                cells = []
                for share, cell in by_share.items():
                    cells.append(
                        f"{share} both fall back"
                        if cell["reading"]
                        else f"{share} cost {shown(cell['cost_per_1k'], 1, 2)} quality {shown(cell['quality'], 100, 1)}"
                    )
                print(f"  {pair:24} {mix:9} " + "; ".join(cells))


def main(argv: Sequence[str] = ()) -> int:
    table = build(real_inputs())
    FIGURES.mkdir(exist_ok=True)
    TABLE_V4.write_text(json.dumps(table, indent=2) + "\n")
    print_summary(table)
    print(f"\nWrote {TABLE_V4.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
