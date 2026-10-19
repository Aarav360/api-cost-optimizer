"""The offline simulator (PLAN_v4.md §7; Amendment A8, item 4).

A policy hands the simulator one trace per item: the configs it called, in order, and
its overhead in dollars at list price. The answer returned is the last config's. The
simulator looks each call up in the results matrix and returns one outcome per item,
which `totals` combines under a task mix.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from analysis.matrix import LADDER, Matrix
from harness.build_benchmark import TASK_MIX_FILE


@dataclass(frozen=True)
class Trace:
    item_id: str
    configs: tuple[str, ...]  # in the order called; the last one's answer is returned
    overhead: float = 0.0  # router and gate calls, dollars at list price


@dataclass(frozen=True)
class Outcome:
    item_id: str
    task: str
    source_group: str
    cost: float  # every config called plus the overhead, dollars at list price
    success: int  # the last config's
    configs: tuple[str, ...]


@dataclass(frozen=True)
class Total:
    quality: float  # mix-weighted share of items answered correctly, 0 to 1
    cost_per_1k: float  # mix-weighted dollars per 1,000 requests, at list price


def always(matrix: Matrix, config: str) -> list[Trace]:
    """The fixed-config policy: every item calls `config`, with no overhead."""
    return [Trace(item_id, (config,)) for item_id in matrix.items]


class TraceError(ValueError):
    """A set of traces the simulator refuses. Nothing in it is evaluated."""


def validate(matrix: Matrix, traces: list[Trace]) -> None:
    counts = Counter(trace.item_id for trace in traces)
    # A policy that skipped hard items would look cheaper, and one that covered an item
    # twice would count it twice, so the trace set must cover the matrix's items exactly.
    if repeated := sorted(item_id for item_id, n in counts.items() if n > 1):
        raise TraceError(f"{len(repeated)} items have more than one trace, e.g. {repeated[0]}")
    if unknown := sorted(set(counts) - set(matrix.items)):
        raise TraceError(f"{len(unknown)} traces are for items not in the matrix, e.g. {unknown[0]}")
    if missing := sorted(set(matrix.items) - set(counts)):
        raise TraceError(f"{len(missing)} items have no trace, e.g. {missing[0]}")
    for trace in traces:
        if not trace.configs:
            raise TraceError(f"{trace.item_id}'s trace calls no config")
        if len(set(trace.configs)) != len(trace.configs):
            raise TraceError(f"{trace.item_id}'s trace calls a config twice: {trace.configs}")
        for config in trace.configs:
            if config not in LADDER:
                raise TraceError(f"{trace.item_id} calls {config!r}, which is not a config on the ladder")
            if (trace.item_id, config) not in matrix.cells:
                raise TraceError(f"{trace.item_id} calls {config}, which has no stored answer for it")


def simulate(matrix: Matrix, traces: list[Trace]) -> list[Outcome]:
    """One outcome per item. Refuses the whole set (TraceError) if any trace is invalid."""
    validate(matrix, traces)
    outcomes = []
    for trace in traces:
        item = matrix.items[trace.item_id]
        cells = [matrix.cells[(trace.item_id, config)] for config in trace.configs]
        cost = sum(cell.cost for cell in cells) + trace.overhead
        outcomes.append(Outcome(item.item_id, item.task, item.source_group, cost, cells[-1].success, trace.configs))
    return outcomes


def load_mixes(path: Path = TASK_MIX_FILE) -> dict[str, dict[str, float]]:
    """The uniform and realistic task mixes (Amendment A5, item 6)."""
    return yaml.safe_load(path.read_text())


def item_weights(outcomes: list[Outcome], mix: dict[str, float]) -> dict[str, float]:
    """Each item weighs its task's mix weight divided by its task's item count (Amendment A5, item 6)."""
    counts = Counter(outcome.task for outcome in outcomes)
    if set(counts) != set(mix):
        raise ValueError(f"the mix weighs tasks {sorted(mix)}, but the outcomes are for {sorted(counts)}")
    return {outcome.item_id: mix[outcome.task] / counts[outcome.task] for outcome in outcomes}


def totals(outcomes: list[Outcome], mix: dict[str, float]) -> Total:
    """Quality is the weighted mean of task means; cost the weighted mean cost per request, per 1,000."""
    weights = item_weights(outcomes, mix)
    quality = sum(weights[outcome.item_id] * outcome.success for outcome in outcomes)
    cost = sum(weights[outcome.item_id] * outcome.cost for outcome in outcomes)
    return Total(quality, cost * 1000)
