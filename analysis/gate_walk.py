"""The walk of a pure cascade on dev (PLAN_v4.md §8, §9; Amendment A17, items 10 and 14).

A pure cascade starts at `haiku` and walks its gate's cutoffs: its steps are the gate scores of `haiku`'s dev answers
that pass the free checks, under A9 item 9's rule (`analysis.walk.quantile_steps`), taken from the gate-score table
before any outcome is looked up. The walk starts at A and tries cutoffs from the strictest (the highest score) to the
most lenient; an answer is accepted when its gate score is at least the cutoff. The free-checks-only pure cascade is
a policy of its own with a single step: no cutoff, so the free checks alone judge. A rejected answer goes to `opus`.

`pure_walks` gives the walks by mix and target, as `rules_walk.dev_walks` does for a router, and `pure_section` builds
the section for the operating-points file. Writing it into the committed file is Stage 7's last step (#70): until then
a check or a script writes to a temporary file with `write_section`.

For a router-started cascade the steps are the router's own (`analysis.walk.step_thresholds`) and each step's traces come
from `analysis.gate_table.cascade_trace` with the router's pick as `start` and the fixed cutoff: wrap them in
`analysis.walk.Candidate`s and call `walk_candidates`, as `pure_walks` does here.
"""

from functools import partial
from typing import Any

from analysis.bootstrap import Bootstrap
from analysis.gate_table import GateScoreTable, pure_traces
from analysis.matrix import Matrix
from analysis.operating_points import build_cascade_section
from analysis.references import TARGETS
from analysis.router_names import FREE_GATE, HAIKU
from analysis.simulator import load_mixes
from analysis.walk import Candidate, Walk, quantile_steps, walk_candidates

MIXES = ("realistic", "uniform")  # the walk's task mixes, in the figure's panel order


def cutoff_steps(table: GateScoreTable, gate: str, start: str = HAIKU) -> tuple[float, ...]:
    """A pure cascade's steps, ascending: `quantile_steps` of the gate scores of `start`'s answers that pass the free
    checks. A refused call has no score and is no step. Empty for the free-checks-only cascade and for a gate that
    scored nothing, which leaves A the only operating point."""
    if gate == FREE_GATE:
        return ()
    scores = []
    for row in table.rows:
        entry = row.paid(gate) if row.config == start and row.free_passed else None
        if entry is not None and entry.score is not None:
            scores.append(entry.score)
    return quantile_steps(scores)


def pure_candidates(table: GateScoreTable, gate: str, start: str = HAIKU) -> list[Candidate]:
    """The walk's candidates, strictest first. The free-checks-only cascade's one candidate has neither a threshold
    nor a cutoff; a gate's have a cutoff each."""
    if gate == FREE_GATE:
        return [Candidate(None, None, partial(pure_traces, table, gate, None, start))]
    return [
        Candidate(None, cutoff, partial(pure_traces, table, gate, cutoff, start))
        for cutoff in reversed(cutoff_steps(table, gate, start))
    ]


def pure_walks(dev: Matrix, table: GateScoreTable, gate: str, draws: Bootstrap) -> dict[str, dict[float, Walk]]:
    """One walk per task mix and quality target on dev for the pure cascade ending in `gate`, keyed by mix name and
    then target share. The table is dev's, and the matrix must hold exactly its `haiku` items."""
    mixes = load_mixes()
    return {
        mix: {share: walk_candidates(dev, pure_candidates(table, gate), draws, mixes[mix], share) for share in TARGETS}
        for mix in MIXES
    }


def pure_section(walks: dict[str, dict[float, Walk]], gate: str) -> dict[str, Any]:
    """The pure cascade's section: a cutoff in each operating point and step, or none for the free-checks-only one."""
    return build_cascade_section(walks, threshold=False, cutoff=gate != FREE_GATE)
