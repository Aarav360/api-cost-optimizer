"""The estimated p50 latency of a policy (PLAN_v4.md §10; Amendment A17, item 16; A8 item 4)."""

import json
import math
import statistics
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from analysis import jev_train_report
from analysis.gate_table import GateScoreTable, gate_problem
from analysis.matrix import Matrix
from analysis.router_names import G_PLUS, JEV_GATE, E, F, G
from analysis.simulator import Trace, check_mix, item_weights, validate
from harness.client import PROJECT_ROOT
from routers.cascade import paid_gate_called
from routers.ladder import FRONTIER, LADDER

# A median in seconds: one for every task, or one per task.
Seconds = float | Mapping[str, float]

# A7 item 1's pilot latency in seconds, median over 20 items per task and config, on the standard API with 8 calls in
# flight, in the formats the collection run uses (`pilot_medians` shows the filter). A check reads A7 item 1's table
# from the plan and holds these equal to it.
PILOT_MEDIANS: dict[str, dict[str, float]] = {
    "ifeval": {"haiku": 6.9, "sonnet": 10.0, "opus_low": 11.1, "opus": 14.0},
    "banking77": {"haiku": 2.0, "sonnet": 1.5, "opus_low": 1.7, "opus": 2.2},
    "mmlu_pro": {"haiku": 4.0, "sonnet": 3.0, "opus_low": 3.4, "opus": 4.5},
    "mbppplus": {"haiku": 1.6, "sonnet": 2.1, "opus_low": 2.3, "opus": 3.4},
}

# The pilot's formats (A7 item 1): the table is the collection run's, so Banking77 and MMLU-Pro (which ran in both
# formats) count their structured-output rows and IFEval and MBPP+ their free-text ones. A median over both formats,
# or over the other pilot tasks (CRUXEval, MATH), is not the table's.
PILOT_FORMATS = {"banking77": "structured", "mmlu_pro": "structured", "ifeval": "free", "mbppplus": "free"}

# E's median `score(request)` over the 744 dev requests, 4.1 microseconds (A9 item 17), and F's per-task medians in
# seconds (A11 item 11: 18.2, 25.3, 28.8 and 54.6 ms). Both are recorded in the plan as measured and aren't re-run; a
# check reads the plan's numbers. G's and G+'s come from their stored dev calls (`router_medians`).
E_MEDIAN = 4.1e-6
F_MEDIANS: dict[str, float] = {"banking77": 0.0182, "ifeval": 0.0253, "mbppplus": 0.0288, "mmlu_pro": 0.0546}
TABLE_V3 = PROJECT_ROOT / "figures" / "table_v3.json"


class LatencyError(ValueError):
    """Medians, weights or traces the estimate refuses. Nothing is estimated."""


def weighted_median(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """The weighted median of `values` by item ID: the value at which the sorted items' weight first reaches half the
    total. Where it reaches exactly half, the midpoint with the next value, as numpy's median takes it, so equal weights
    give the ordinary median. Refuses (LatencyError) no values, weights for other items or a weight that isn't above 0,
    and a value that isn't a finite number."""
    if not values:
        raise LatencyError("there are no values to take a median of")
    if set(values) != set(weights):
        raise LatencyError("the weights must be of exactly the items the values are of")
    if bad := [w for w in weights.values() if not (math.isfinite(w) and w > 0)]:
        raise LatencyError(f"every weight must be a number above 0, and {len(bad)} aren't, e.g. {bad[0]}")
    if bad := [v for v in values.values() if not (isinstance(v, int | float) and math.isfinite(v))]:
        raise LatencyError(f"every value must be a finite number, and {len(bad)} aren't, e.g. {bad[0]}")
    order = sorted(values, key=lambda item_id: values[item_id])
    half = math.fsum(weights.values()) / 2
    reached = 0.0
    for position, item_id in enumerate(order):
        reached += weights[item_id]
        if reached >= half - 1e-12:
            exactly = abs(reached - half) <= 1e-12 and position + 1 < len(order)
            return (values[item_id] + values[order[position + 1]]) / 2 if exactly else values[item_id]
    raise AssertionError("unreachable: the whole weight is at least half")


def _seconds(value: Seconds, task: str, what: str) -> float:
    """`value` for `task`: itself when it is one number, the task's entry when it is a mapping."""
    if isinstance(value, Mapping):
        if task not in value:
            raise LatencyError(f"{what} has no median for the task {task!r} ({sorted(value)})")
        value = value[task]
    if not (isinstance(value, int | float) and math.isfinite(value) and value >= 0):
        raise LatencyError(f"{what} is {value!r} for {task!r}, which must be a finite number of seconds, at least 0")
    return float(value)


def trace_seconds(
    task: str,
    configs: Sequence[str],
    router: Seconds = 0.0,
    gate_calls: Sequence[Seconds] = (),
    pilot: Mapping[str, Mapping[str, float]] = PILOT_MEDIANS,
) -> float:
    """One item's estimated latency: the sum along its trace of the pilot median for its task and each config called,
    the router's median and each gate call's median (A17 item 16). Refuses (LatencyError) a task or config the pilot
    holds no median for, and a median that isn't a finite number of seconds."""
    if task not in pilot:
        raise LatencyError(f"the pilot holds no medians for the task {task!r} ({sorted(pilot)})")
    total = _seconds(router, task, "the router's median")
    for config in configs:
        if config not in pilot[task]:
            raise LatencyError(f"the pilot holds no median for {config!r} on {task!r} ({sorted(pilot[task])})")
        total += _seconds(pilot[task][config], task, f"the pilot median of {config}")
    for number, call in enumerate(gate_calls):
        total += _seconds(call, task, f"gate call {number}'s median")
    return total


def estimate_latency(
    matrix: Matrix,
    traces: Sequence[Trace],
    mix: Mapping[str, float],
    router: Seconds = 0.0,
    gate_calls: Mapping[str, Sequence[Seconds]] | None = None,
    pilot: Mapping[str, Mapping[str, float]] = PILOT_MEDIANS,
) -> float:
    """A policy's estimated p50 latency in seconds under a task mix: the mix-weighted median, over the matrix's
    items, of each item's sum along its trace (`trace_seconds`). It works for any policy: a fixed config's traces
    (A to D), a router's (E to G+, with `router` its median, one number or one per task, paid on every item) and a
    cascade's (`gate_calls` gives each item's gate calls, see `cascade_gate_calls`).

    An estimate: summing medians ignores that escalated items run longer, so Stage 8's live replay is the real check.
    Refuses (TraceError) the traces the simulator refuses, (MixError) a mix that doesn't weigh the matrix's tasks, and
    (LatencyError) what `trace_seconds` refuses and gate calls for an item with no trace.
    """
    validate(matrix, list(traces))
    check_mix([item.task for item in matrix.items.values()], dict(mix))
    gate_calls = gate_calls or {}
    if stray := sorted(set(gate_calls) - set(matrix.items)):
        raise LatencyError(f"gate calls are given for items not in the matrix, e.g. {stray[0]}")
    seconds = {
        trace.item_id: trace_seconds(
            matrix.items[trace.item_id].task, trace.configs, router, gate_calls.get(trace.item_id, ()), pilot
        )
        for trace in traces
    }
    return weighted_median(seconds, item_weights(list(matrix.items.values()), dict(mix)))


def pilot_medians(rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    """A7 item 1's table from the pilot's rows (`source`, `format`, `config`, `latency_s`): the median of `latency_s` per
    task and config over the rows in the task's `PILOT_FORMATS` format. Other tasks' rows and rows with no latency
    (batch rows have none) are left out. Refuses (LatencyError) a task and config with no row left."""
    times: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        task = str(row["source"])
        if PILOT_FORMATS.get(task) == row["format"] and row["latency_s"] is not None:
            times.setdefault((task, str(row["config"])), []).append(float(str(row["latency_s"])))
    medians: dict[str, dict[str, float]] = {}
    for task in PILOT_FORMATS:
        for config in LADDER:
            if (task, config) not in times:
                raise LatencyError(f"the pilot rows hold no call for {config!r} on {task!r}")
            medians.setdefault(task, {})[config] = statistics.median(times[(task, config)])
    return medians


def router_medians(table_v3: Path = TABLE_V3) -> dict[str, Seconds]:
    """Each router's median in seconds, by letter: E's and F's as recorded in the plan, and G's and G+'s per task from
    their stored dev calls, as table v3 holds them (the first call and every reconnect left out, A15 item 10)."""
    latency = json.loads(table_v3.read_text())["latency"]
    return {
        E: E_MEDIAN,
        F: dict(F_MEDIANS),
        **{r: {task: float(v["median"]) for task, v in latency[r]["per_task"].items()} for r in (G, G_PLUS)},
    }


def call_median(seconds: Sequence[float], skip_first: bool = True) -> float:
    """The median of stored calls' times, the first call left out (A17 item 16, as for the Sonnet latency sample, since
    it opens the connection). Refuses (LatencyError) no call left."""
    kept = list(seconds[1:] if skip_first else seconds)
    if not kept:
        raise LatencyError("no call is left once the first call is left out")
    return float(statistics.median(kept))


def jev_call_medians(records: Sequence[Mapping[str, object]], task_of: Mapping[str, str]) -> dict[str, float]:
    """The Jev gate's median per task from its stored calls, in the order stored: the successful attempt's time, with
    the first call and every call that opened a new connection left out (A15 item 10), as the router calls' are. The
    noise pass (`pass` 2) is not part of the collection and is left out."""
    first_pass = [r for r in records if r.get("pass", 1) == 1]
    try:
        report = jev_train_report.latency(first_pass, task_of)
    except jev_train_report.ReportError as error:
        raise LatencyError(str(error)) from None
    return {task: median for task, (median, _, _) in report.by_task.items()}


def code_check_median(free_records: Iterable[Mapping[str, object]]) -> float:
    """The code check's median in seconds over the stored free-check records of the answers it applied to
    (`code_duration_s`, timed locally). Refuses (LatencyError) none."""
    times = [
        float(str(r["code_duration_s"])) for r in free_records if r["code_applied"] and r["code_duration_s"] is not None
    ]
    if not times:
        raise LatencyError("no stored answer has a timed code check")
    return float(statistics.median(times))


@dataclass(frozen=True)
class GateMedians:
    """The median seconds of one call of each gate: the code check (timed locally), the Jev gate and the Sonnet gate."""

    code_check: Seconds
    jev: Seconds
    sonnet: Seconds


def cascade_gate_calls(
    table: GateScoreTable,
    item_id: str,
    start: str,
    gate: str,
    cutoff: float | None,
    medians: GateMedians,
    code_items: Collection[str],
) -> tuple[Seconds, ...]:
    """The gate calls one item's trace makes in a cascade that starts at `start` and ends in `gate` at `cutoff`, as
    their medians, in the order made: the code check where it applies to the item (`code_items`: those whose request has
    an `assert` line), whatever the other free checks said, as every check runs on an answer; then the paid gate, when
    `routers.cascade.paid_gate_called` says the trace pays one. An item started at `opus` gets none. Refuses
    (LatencyError) a gate that is neither the free checks nor a paid gate, and a cutoff that doesn't go with it.
    """
    if problem := gate_problem(gate, cutoff):
        raise LatencyError(problem)
    if start == FRONTIER:
        return ()
    calls: list[Seconds] = [medians.code_check] if item_id in code_items else []
    if paid_gate_called(start, table.row(item_id, start).free_passed, cutoff):
        calls.append(medians.jev if gate == JEV_GATE else medians.sonnet)
    return tuple(calls)
