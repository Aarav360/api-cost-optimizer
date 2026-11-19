"""The final results table on a split, and the per-policy tables (PLAN_v4.md §10; Amendment A17 item 24; A19 items 6 and 8; issue #77, #89).

§10's table has one row per system and these columns: quality with its 95% interval, cost per 1k, savings against A, the share
of the oracle's savings captured at matched quality, and the estimated p50 latency. The rows are A (frontier only), B (cheapest
only), D (the oracle at the target), E (rules), F (trained), G (Jev), G+ (bolstered Jev) and I (the best cascade). The G+ row is
here because the plan's table predates A15 (A17 item 24).

**One table for each mix and quality target,** since every policy has an operating point for each (the realistic mix's 99% is
the headline's cell): `by_mix[mix][target]`. **Per task** (`by_task[task]`), the same rows without D, which has no per-task
target, and without a verdict or captured fraction, which are given for the mixes only (A7 item 2). IFEval and MBPP+ carry A7
item 2's warning, worded for the split, wherever they appear.

**The best cascade** is the one A17 item 24's rule chooses on dev's table (`analysis/final_headline.best_cascade`): chosen on
dev, reported on test. A policy that does better on test is a finding, never this row.

**The per-policy tables** (`per_policy`) hold every policy of the plan, the four routers and all fifteen cascades, in the same row
form under both mixes and all three targets and per task, so no cascade is shown only through the best one's row.

**Target met** is each policy's reading at a margin of zero (A19 item 6, `analysis/target_met.py`): met, not met, or falls back to
A, which meets every target trivially and is never read as met.

**The latency** is A17 item 16's estimate: the mix-weighted median of each item's sum along its trace of the pilot's median for
the config, the router's median and each gate call's median, with the medians dev's table v4 committed (the live replay is the
real check). A point that falls back is A: no router and no gate. Run on dev, the estimates equal table v4's.

Nothing here reads test beyond the parts built before it: the rows are read from them, and the latency from the outcomes noted by
`analysis/operating_eval.py`.
"""

from collections.abc import Collection, Mapping
from typing import Any

from analysis import final_gates, final_headline, target_met
from analysis.gate_table import GateScoreTable
from analysis.latency_estimate import GateMedians, cascade_gate_calls, trace_seconds, weighted_median
from analysis.operating_eval import FrozenPoint, shared_fixed
from analysis.operating_points import target_key
from analysis.references import TARGETS
from analysis.replay_plan import Medians, code_items_of, read_medians
from analysis.router_names import CASCADE_STARTS, CASCADES, G_PLUS, HAIKU, NAMED_ROUTERS, E, F, G, parse_cascade
from analysis.simulator import Outcome, item_weights, load_mixes
from routers.ladder import FRONTIER

MIXES = ("realistic", "uniform")
TARGET_KEYS = tuple(target_key(share) for share in TARGETS)
TASKS = ("ifeval", "banking77", "mmlu_pro", "mbppplus")
CODE_ITEMS = "code_items"  # the `Extra` holding the split's items whose request has an `assert` line
CHEAPEST = "haiku"
COLUMNS = ("quality", "cost_per_1k", "savings_vs_A", "captured", "latency_p50_s")
READING = {target_met.MET: "met", target_met.MISSED: "not met", target_met.FALLS_BACK: target_met.FALLS_BACK}

# The final table's rows, §10's order: the system's letter and words, and the policy's section of the operating-points file.
SYSTEMS: tuple[tuple[str, str, str | None], ...] = (
    ("A", "Frontier only", None),
    ("B", "Cheapest only", None),
    ("D", "Oracle at target", None),
    *((letter, label, NAMED_ROUTERS[letter]) for letter, label in
      ((E, "Rules"), (F, "Trained"), (G, "Jev"), (G_PLUS, "Bolstered Jev"))),
)  # fmt: skip
BEST = ("I", "Best cascade")


class FinalTableError(ValueError):
    """Parts the final table can't be read from: a policy that was not built, or a point nobody evaluated."""


# ---- the estimated latency -----------------------------------------------------------------------------------------


def trace_latencies(
    outcomes: list[Outcome], *, fell_back: bool, router: Any, gate: str | None, cutoff: float | None,
    table: GateScoreTable, medians: GateMedians, code_items: Collection[str],
) -> dict[str, float]:  # fmt: skip
    """Each item's estimated seconds along the trace its outcome called (A17 item 16). A point that fell back is A: `opus`
    alone, with no router and no gate. Otherwise `router` is paid on every item (0 for a pure cascade) and a cascade (`gate`
    the free checks or a paid gate, at the point's `cutoff`) adds its gate calls (`cascade_gate_calls`)."""
    seconds = {}
    for o in outcomes:
        if fell_back:
            seconds[o.item_id] = trace_seconds(o.task, (FRONTIER,))
            continue
        calls = (
            ()
            if gate is None
            else cascade_gate_calls(table, o.item_id, o.configs[0], gate, cutoff, medians, code_items)
        )
        seconds[o.item_id] = trace_seconds(o.task, o.configs, router, calls)
    return seconds


def median_seconds(seconds: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """The weighted median of the items `weights` names, which may be a task's alone."""
    return weighted_median({item: seconds[item] for item in weights}, weights)


# ---- the rows ------------------------------------------------------------------------------------------------------


def _interval(estimate: Mapping[str, Any]) -> dict[str, float | None]:
    return {"value": estimate["value"], "low": estimate["low"], "high": estimate["high"]}


def _row(system: str, label: str, section: str | None, **fields: Any) -> dict[str, Any]:
    return {"system": system, "label": label, "section": section, "fell_back": None, "reading": None, "threshold": None,
            "cutoff": None, "quality": None, "cost_per_1k": None, "savings_vs_A": None, "captured": None,
            "latency_p50_s": None, **fields}  # fmt: skip


def policy_row(system: str, label: str, section: str, row: Mapping[str, Any], reading: str | None, latency: float,
               frontier_cost: float | None = None) -> dict[str, Any]:  # fmt: skip
    """A policy's row from its part's row (a mix's row of table v1, or a task's, which has no savings or captured fraction:
    `frontier_cost` is A's cost there and the savings are worked from it)."""
    savings = row["savings_vs_A"]["share"] if frontier_cost is None else 1 - row["cost_per_1k"]["value"] / frontier_cost
    return _row(
        system, label, section, fell_back=row["fell_back"], reading=reading, threshold=row["threshold"],
        cutoff=row.get("cutoff"), quality=_interval(row["quality"]), cost_per_1k=_interval(row["cost_per_1k"]),
        savings_vs_A=savings, captured=row["captured"]["value"] if "captured" in row else None, latency_p50_s=latency,
    )  # fmt: skip


def reference_rows(panel: Mapping[str, Any], target: str, latencies: Mapping[str, float]) -> list[dict[str, Any]]:
    """A, B and D under one mix and target, from the references part's panel. A saves and captures nothing; B's captured
    fraction is "—" (None); D is the oracle at the target's quality, which captures all of its own savings and has no latency."""
    configs, oracle = panel["configs"], panel["targets"][target]
    return [
        _row("A", "Frontier only", None, quality=_interval(configs[FRONTIER]["quality"]),
             cost_per_1k=_interval(configs[FRONTIER]["cost_per_1k"]), savings_vs_A=0.0, captured=0.0,
             latency_p50_s=latencies[FRONTIER], fell_back=None),
        _row("B", "Cheapest only", None, quality=_interval(configs[CHEAPEST]["quality"]),
             cost_per_1k=_interval(configs[CHEAPEST]["cost_per_1k"]),
             savings_vs_A=configs[CHEAPEST]["savings_vs_A"]["share"], latency_p50_s=latencies[CHEAPEST]),
        _row("D", "Oracle at target", None, quality={"value": oracle["quality"], "low": None, "high": None},
             cost_per_1k={"value": oracle["oracle"]["cost_per_1k"], "low": None, "high": None},
             savings_vs_A=oracle["oracle"]["savings_vs_A"]["share"], captured=1.0),
    ]  # fmt: skip


def sections_of_policies() -> tuple[str, ...]:
    """The four routers' sections, then the fifteen cascades' keys, as the plan's per-policy tables list them."""
    return (*NAMED_ROUTERS.values(), *CASCADES)


def label_of(section: str) -> str:
    return CASCADES.get(section) or next(f"{letter} {words}" for letter, words, found in SYSTEMS if found == section)


def router_of(section: str) -> str | None:
    """The letter of the router a policy starts at: a router alone, or the start of a router-started cascade; None for a
    pure cascade."""
    if section in NAMED_ROUTERS.values():
        return CASCADE_STARTS[section]
    start = parse_cascade(section)[0]
    return None if start == HAIKU else CASCADE_STARTS[start]


def gate_of(section: str) -> str | None:
    return parse_cascade(section)[1] if section in CASCADES else None


def build(inputs: Any, best: str | None = None) -> dict[str, Any]:
    """The final table and the per-policy tables from the parts built so far and the points they evaluated. Refuses
    (FinalTableError) a policy with no part or a point nobody evaluated."""
    built: Mapping[str, Any] = inputs.cache.get("built", {})
    sections = sections_of_policies()
    if missing := [s for s in (*sections, "references") if s not in built]:
        raise FinalTableError(f"the final table stands after the parts of {', '.join(missing)}")
    fixed = shared_fixed(inputs.cache, inputs.matrix)
    at: dict[tuple[str, str, str], tuple[FrozenPoint, list[Outcome]]] = {
        (p.section, p.mix, p.target): (p, outcomes) for p, outcomes in fixed.evaluated.items()
    }
    mixes = load_mixes()
    met = target_met.rows_of(inputs.draws, fixed, fixed.evaluated, mixes)
    chosen = final_headline.best_cascade() if best is None else best
    if chosen not in CASCADES:
        raise FinalTableError(f"the best cascade {chosen!r} is not one of the fifteen")
    medians: Medians = read_medians()
    table: GateScoreTable = inputs.extras[final_gates.GATE_TABLE].table
    code_items: frozenset[str] = inputs.extras[CODE_ITEMS]
    items = list(inputs.matrix.items.values())
    tasks = [t for t in TASKS if any(i.task == t for i in items)]
    weights = {mix: item_weights(items, mixes[mix]) for mix in MIXES}
    task_weights = {t: {i.item_id: 1.0 for i in items if i.task == t} for t in tasks}

    def seconds_of(section: str, mix: str, target: str) -> dict[str, float]:
        try:
            point, outcomes = at[section, mix, target]
        except KeyError:
            raise FinalTableError(f"{section} has no evaluated point at {mix} {target}") from None
        letter = router_of(section)
        return trace_latencies(
            outcomes, fell_back=point.fell_back, router=0.0 if letter is None else medians.routers[letter],
            gate=gate_of(section), cutoff=point.cutoff, table=table, medians=medians.gates, code_items=code_items)  # fmt: skip

    fixed_seconds = {c: trace_latencies(fixed[c], fell_back=False, router=0.0, gate=None, cutoff=None, table=table,
                                        medians=medians.gates, code_items=code_items) for c in (FRONTIER, CHEAPEST)}  # fmt: skip
    seconds = {(s, m, t): seconds_of(s, m, t) for s in sections for m in MIXES for t in TARGET_KEYS}

    def reading(section: str, mix: str, target: str) -> str:
        return READING[met[section][mix][target]["reading"]]

    def mix_row(system: str, label: str, section: str, mix: str, target: str) -> dict[str, Any]:
        return policy_row(system, label, section, built[section]["mixes"][mix][target], reading(section, mix, target),
                          median_seconds(seconds[section, mix, target], weights[mix]))  # fmt: skip

    def task_part(section: str, task: str, mix: str, target: str) -> Mapping[str, Any]:
        key = section if section in NAMED_ROUTERS.values() else f"{section}_tasks"
        return built[key]["tasks"][task][mix][target]  # type: ignore[no-any-return]

    def task_row(system: str, label: str, section: str, task: str, mix: str, target: str) -> dict[str, Any]:
        a_cost = built["references"]["tasks"][task]["configs"][FRONTIER]["cost_per_1k"]["value"]
        return policy_row(system, label, section, task_part(section, task, mix, target), None,
                          median_seconds(seconds[section, mix, target], task_weights[task]), a_cost)  # fmt: skip

    label = CASCADES[chosen]
    final: dict[str, Any] = {}
    for mix in MIXES:
        latencies = {c: median_seconds(fixed_seconds[c], weights[mix]) for c in fixed_seconds}
        final[mix] = {}
        for target in TARGET_KEYS:
            rows = reference_rows(built["references"]["mixes"][mix], target, latencies)
            rows += [mix_row(system, words, section, mix, target) for system, words, section in SYSTEMS if section]
            rows.append(mix_row(BEST[0], f"{BEST[1]}: {label}", chosen, mix, target))
            final[mix][target] = rows
    per_task: dict[str, Any] = {}
    for task in tasks:
        task_panel = built["references"]["tasks"][task]
        a_cost = task_panel["configs"][FRONTIER]["cost_per_1k"]["value"]
        per_task[task] = {"items": task_panel["items"], "warning": task_panel["warning"], "mixes": {}}
        for mix in MIXES:
            per_task[task]["mixes"][mix] = {}
            for target in TARGET_KEYS:
                latencies = {c: median_seconds(fixed_seconds[c], task_weights[task]) for c in fixed_seconds}
                rows = [
                    _row("A", "Frontier only", None, quality=_interval(task_panel["configs"][FRONTIER]["quality"]),
                         cost_per_1k=_interval(task_panel["configs"][FRONTIER]["cost_per_1k"]), savings_vs_A=0.0,
                         latency_p50_s=latencies[FRONTIER]),
                    _row("B", "Cheapest only", None, quality=_interval(task_panel["configs"][CHEAPEST]["quality"]),
                         cost_per_1k=_interval(task_panel["configs"][CHEAPEST]["cost_per_1k"]),
                         savings_vs_A=1 - task_panel["configs"][CHEAPEST]["cost_per_1k"]["value"] / a_cost,
                         latency_p50_s=latencies[CHEAPEST]),
                ]  # fmt: skip
                rows += [
                    task_row(system, words, section, task, mix, target) for system, words, section in SYSTEMS if section
                ]
                rows.append(task_row(BEST[0], f"{BEST[1]}: {label}", chosen, task, mix, target))
                per_task[task]["mixes"][mix][target] = rows
    per_policy = {
        section: {
            "label": label_of(section),
            "mixes": {m: {t: mix_row("", label_of(section), section, m, t) for t in TARGET_KEYS} for m in MIXES},
            "tasks": {task: {m: {t: task_row("", label_of(section), section, task, m, t) for t in TARGET_KEYS}
                             for m in MIXES} for task in tasks},
        }
        for section in sections
    }  # fmt: skip
    return {
        "columns": list(COLUMNS),
        "best_cascade": {
            "section": chosen,
            "label": label,
            "chosen_on": "dev, by A17 item 24's rule",
            "also_in_per_policy": True,
        },  # fmt: skip
        "note": "Per task there is no oracle row, verdict or captured fraction: verdicts are given for the mixes only (A7 "
        "item 2), and IFEval's and MBPP+'s rows are indicative (see each task's warning). Latency is A17 item 16's "
        "estimate from dev's committed medians; the live replay is its check.",
        "by_mix": final,  # not "mixes": a part with that key is read as a policy's rows (`analysis/target_met.py`)
        "by_task": per_task,
        "per_policy": per_policy,
    }


def part(inputs: Any) -> dict[str, Any]:
    """The offline half's final table part: after every policy's part and the comparisons, so it reads what they built."""
    return build(inputs)


# ---- the code check's items, the one thing the latency needs beyond the matrix --------------------------------------


def code_items_dev() -> frozenset[str]:
    from analysis.gate_answers import requests_of

    requests = requests_of("dev")
    return frozenset(code_items_of(requests, list(requests)))


def code_items_test(run: Any) -> frozenset[str]:
    requests = run.unlocked().requests()
    return frozenset(code_items_of(requests, list(requests)))
