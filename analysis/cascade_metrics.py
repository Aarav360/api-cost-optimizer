"""Stage 7's gate metrics and estimated latency as table v4's data (PLAN_v4.md §9, §10; Amendment A17, items 16 and 21;
issue #71).

`analysis.gate_metrics` and `analysis.latency_estimate` compute the numbers; this module puts them where table v4 holds
them, in plain JSON-ready data:

- **Gate metrics** (`gate_metrics_data`): the free checks' rates (a row of their own, over all answers) and, per paid gate,
  its rates at the fixed cutoff (A17 item 12) and, for a pure cascade, at each of its operating points' cutoffs (a router-
  started cascade's operating points all sit at the fixed cutoff), plus within-task concordance per config, the summary
  that needs no cutoff. Rates are per config and task (`pooled` is the config over all tasks), false accept and false
  escalate beside the raw counts out of all the config's answers.
- **Estimated p50 latency** (`walk_latency`): a policy's operating point read from the walk's outcomes, A to G+ and the
  fifteen cascades alike.
"""

from collections.abc import Collection, Mapping
from dataclasses import asdict

from analysis.bootstrap import Bootstrap
from analysis.gate_metrics import GateRates, gate_concordance, gate_rates
from analysis.gate_table import PAID_GATES, GateScoreTable
from analysis.latency_estimate import GateMedians, Seconds, cascade_gate_calls, estimate_latency
from analysis.matrix import Matrix
from analysis.operating_points import target_key
from analysis.router_names import FREE_GATE
from analysis.simulator import Trace
from analysis.trained_diagnostics import reading_data
from analysis.walk import Walk
from routers.ladder import FRONTIER

Walks = Mapping[str, Mapping[float, Walk]]  # by mix and then target share


def rates_data(rates: Mapping[tuple[str, str | None], GateRates]) -> dict[str, dict[str, dict]]:
    """Rates as JSON-ready data: config, then task (`pooled` for the config over all its tasks), then the fields of
    `GateRates`. A rate over nothing is null."""
    data: dict[str, dict[str, dict]] = {}
    for (config, task), found in rates.items():
        data.setdefault(config, {})["pooled" if task is None else task] = asdict(found)
    return data


def _task_sizes(matrix: Matrix) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for item in matrix.items.values():
        sizes[item.task] = sizes.get(item.task, 0) + 1
    return sizes


def gate_metrics_data(
    table: GateScoreTable,
    dev: Matrix,
    draws: Bootstrap,
    cutoffs: Mapping[str, float],
    pure_walks: Mapping[str, Walks],
) -> dict[str, dict]:
    """The gate metrics of A17 item 21 on dev, for the free checks and each paid gate in `cutoffs`.

    `pure_walks` maps a paid gate to the walks of its pure cascade (`haiku/Jev`, `haiku/Sonnet`), whose operating points
    each have a walked cutoff; a gate without walks gets no `operating_points`. A point that fell back to A has no cutoff
    and no rates. Refuses (GateMetricsError) what `gate_rates` refuses."""
    data: dict[str, dict] = {FREE_GATE: {"rates": rates_data(gate_rates(table, dev, FREE_GATE, None))}}
    sizes = _task_sizes(dev)
    for gate, cutoff in cutoffs.items():
        if gate not in PAID_GATES:
            raise ValueError(f"{gate!r} is not a paid gate ({list(PAID_GATES)})")
        configs = list(dict.fromkeys(row.config for row in table.rows if row.paid(gate) is not None))
        cache: dict[float, dict] = {}

        def at(cut: float, gate: str = gate) -> dict:
            if cut not in cache:
                cache[cut] = rates_data(gate_rates(table, dev, gate, cut))
            return cache[cut]

        points: dict[str, dict] = {}
        for mix, by_share in pure_walks.get(gate, {}).items():
            points[mix] = {}
            for share, walk in by_share.items():
                point = walk.operating
                found = None if walk.fell_back else at(point.cutoff)  # type: ignore[arg-type]
                points[mix][target_key(share)] = {
                    "cutoff": None if walk.fell_back else point.cutoff,
                    "fell_back": walk.fell_back,
                    "rates": found,
                }
        data[gate] = {
            "fixed_cutoff": cutoff,
            "at_fixed_cutoff": at(cutoff),
            "within_task_concordance": {
                config: reading_data(gate_concordance(draws, table, dev, gate, config), sizes) for config in configs
            },
            "operating_points": points,
        }
    return data


# ---- estimated latency -------------------------------------------------------------------------------------------


def fixed_latency(dev: Matrix, config: str, mix: Mapping[str, float]) -> float:
    """A reference's estimated p50 latency: every item calls `config` and nothing else."""
    return estimate_latency(dev, [Trace(item_id, (config,)) for item_id in dev.items], mix)


def walk_latency(
    dev: Matrix,
    table: GateScoreTable,
    walk: Walk,
    mix: Mapping[str, float],
    *,
    router: Seconds = 0.0,
    gate: str | None = None,
    medians: GateMedians | None = None,
    code_items: Collection[str] = (),
) -> float:
    """The estimated p50 latency, in seconds under `mix`, of a walk's operating point (A17 item 16).

    The trace of each item is what the operating point's outcomes called. A walk that fell back is A: no router, no gate.
    Otherwise `router` is paid on every item, and a cascade (`gate` the free checks or a paid gate, with the cutoff the
    operating point holds) adds each item's gate calls (`cascade_gate_calls`: the code check where the request has an
    `assert` line, then the paid gate where the trace paid one). A plain router passes `gate=None`."""
    outcomes = list(walk.operating.outcomes)
    traces = [Trace(o.item_id, o.configs) for o in outcomes]
    if walk.fell_back:
        return estimate_latency(dev, [Trace(o.item_id, (FRONTIER,)) for o in outcomes], mix)
    calls = None
    if gate is not None:
        if medians is None:
            raise ValueError("a cascade's latency needs the gates' median times")
        cutoff = walk.operating.cutoff
        calls = {
            o.item_id: cascade_gate_calls(table, o.item_id, o.configs[0], gate, cutoff, medians, code_items)
            for o in outcomes
        }
    return estimate_latency(dev, traces, mix, router, calls)
