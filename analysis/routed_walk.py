"""The walk of a router-started cascade on dev (PLAN_v4.md §9; Amendment A17, items 11 and 12).

A router-started cascade walks its router's own steps (`analysis.walk.step_thresholds` of the router's existing
router-score table, A9 item 9), with its gate's cutoff held fixed (`analysis.gate_cutoff.fixed_cutoff`, from train).
At each step the router picks an item's start config at the threshold, the cascade decision judges that config's
answer (the free checks, then the paid gate at the fixed cutoff), and a rejected answer goes to `opus`. An item the
router sends to `opus` reads no answer and pays no gate. Only the trace builder is new: the loop is
`analysis.walk.walk_candidates`.

So a router alone and the same router with a gate walk the same steps, and `aligned_steps` lines them up. Because the
gate catches some mistakes, the cascade's walk can certify further down than the router's.

`routed_sections` builds the twelve sections (E, F, G and G+, each with the free checks, the Jev gate and the Sonnet
gate); writing them into the committed operating-points file is Stage 7's last step (#70). Until then a check or a
script writes to a temporary file with `analysis.operating_points.write_section`.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any

from analysis.bootstrap import Bootstrap
from analysis.gate_table import GateScoreTable, cascade_trace
from analysis.gate_walk import MIXES
from analysis.matrix import Matrix
from analysis.operating_points import build_cascade_section
from analysis.references import TARGETS
from analysis.router_names import FREE_GATE, GATE_LABELS, NAMED_ROUTERS, cascade_section
from analysis.router_table import RouterScoreTable
from analysis.simulator import Trace, load_mixes
from analysis.walk import Candidate, Step, Walk, step_thresholds, walk_candidates
from routers.decision import choose


def routed_traces(
    router: RouterScoreTable, table: GateScoreTable, gate: str, cutoff: float | None, threshold: float
) -> list[Trace]:
    """One trace per item of the router's table: the router picks the start config at `threshold`, and
    `analysis.gate_table.cascade_trace` judges its answer (the free checks, then `gate` at `cutoff`) and charges the
    router's overhead plus the gate call it made. Refuses (GateTableError) what `cascade_trace` refuses."""
    return [
        cascade_trace(table, entry.item_id, choose(entry.router_scores, threshold), gate, cutoff, entry.overhead)
        for entry in router.items
    ]


def routed_candidates(
    router: RouterScoreTable, table: GateScoreTable, gate: str, cutoff: float | None
) -> list[Candidate]:
    """The walk's candidates, strictest first: the router's own steps, each with the gate's fixed `cutoff` (None for
    the free checks alone)."""
    return [
        Candidate(threshold, cutoff, partial(routed_traces, router, table, gate, cutoff, threshold))
        for threshold in reversed(step_thresholds(router))
    ]


def routed_walks(
    dev: Matrix, router: RouterScoreTable, table: GateScoreTable, gate: str, cutoff: float | None, draws: Bootstrap
) -> dict[str, dict[float, Walk]]:
    """One walk per task mix and quality target on dev for the router-started cascade ending in `gate`, keyed by mix
    name and then target share. The router's table and the gate-score table are dev's."""
    mixes = load_mixes()
    return {
        mix: {
            share: walk_candidates(dev, routed_candidates(router, table, gate, cutoff), draws, mixes[mix], share)
            for share in TARGETS
        }
        for mix in MIXES
    }


def routed_section(walks: Mapping[str, Mapping[float, Walk]], gate: str) -> dict[str, Any]:
    """The cascade's section: its router's threshold in each operating point and step, and the fixed cutoff unless the
    cascade has the free checks alone."""
    return build_cascade_section(walks, threshold=True, cutoff=gate != FREE_GATE)


def routed_sections(
    dev: Matrix,
    routers: Mapping[str, RouterScoreTable],
    table: GateScoreTable,
    cutoffs: Mapping[str, float],
    draws: Bootstrap,
) -> dict[str, dict[str, Any]]:
    """The twelve router-started cascades' sections, by their keys (`router_names.cascade_section`), in item 8's
    order. `routers` maps each named router's section name (`NAMED_ROUTERS`) to its dev router-score table, and
    `cutoffs` maps each paid gate to its fixed cutoff; the free checks need none. Refuses (ValueError) a router or a
    gate that is missing or that no amendment names."""
    if set(routers) != set(NAMED_ROUTERS.values()):
        raise ValueError(f"the routers must be exactly {sorted(NAMED_ROUTERS.values())}, not {sorted(routers)}")
    paid = [gate for gate in GATE_LABELS if gate != FREE_GATE]
    if set(cutoffs) != set(paid):
        raise ValueError(f"the fixed cutoffs must be exactly those of {paid}, not {sorted(cutoffs)}")
    sections = {}
    for name in NAMED_ROUTERS.values():
        for gate in GATE_LABELS:
            cutoff = None if gate == FREE_GATE else cutoffs[gate]
            walks = routed_walks(dev, routers[name], table, gate, cutoff, draws)
            sections[cascade_section(name, gate)] = routed_section(walks, gate)
    return sections


def aligned_steps(alone: Walk, cascaded: Walk) -> list[tuple[Step, Step]]:
    """A router's walk and the walk of its cascade, step by step, A first: the pairs both walks reached (a walk stops
    at its first step that doesn't clear, so one can be longer). Refuses (ValueError) walks whose steps are at
    different thresholds, which would be no comparison."""
    pairs = list(zip(alone.steps, cascaded.steps, strict=False))
    for place, (a, c) in enumerate(pairs):
        if a.threshold != c.threshold:
            raise ValueError(
                f"step {place} is at threshold {a.threshold} for the router and {c.threshold} for the cascade"
            )
    return pairs
