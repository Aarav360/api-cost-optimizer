"""The twelve router-started cascades on test (PLAN_v4.md Amendment A17, items 8, 10 and 26.4; A19, items 6, 8 and 13; issue #77,
#87).

E, F, G and G+ each end in the free checks, the Jev gate or the Sonnet gate: twelve cascades, whose sections of
`routers/operating_points.json` Stage 7 froze. Each is evaluated on a split at its six frozen points with
`analysis/operating_eval.py`, and nothing is walked: the file decides each point's threshold (the router's), its cutoff (the
gate's fixed one) and whether it fell back, and a point that fell back is A itself.

**The traces** are Stage 7's own: `analysis.routed_walk.routed_traces` builds one per item from the router's score table (E and F
as `analysis/final_routers.py` loaded them, G and G+ as `analysis/final_jev.py` did) and the gate-score table
(`analysis/final_gates.py`: the free checks, the Jev column and the Sonnet column), and through `analysis.gate_table.cascade_trace`
asks the shared cascade decision (`routers/cascade.py`) what to call. The router picks the start at the point's threshold; an item
it sends to `opus` gets no gate; and the overhead is the router's plus each paid gate call actually made (a refused call is a
rejection charged its tokens). The same builders ran on dev, so on dev the parts here give the committed numbers of table v4.

**The parts** (`routed_part`) are two for each cascade: its rows under both mixes, which table v4 committed on dev, and its
per-task rows, which dev committed none of (so, as for `haiku/free`, they are a part of their own). Each reads two extras: the
router's table and the gate-score table. They are built with `mix_rows` and `task_rows`, so target met
(`analysis/target_met.py`) covers them with no line of its own.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from analysis.final_gates import GATE_TABLE
from analysis.gate_table import GateScoreTable
from analysis.operating_eval import FrozenPoint, at_threshold, cascade_rows, frozen_points
from analysis.operating_points import OPERATING_POINTS_FILE
from analysis.replay_plan import ROUTER_TABLES
from analysis.routed_walk import routed_traces
from analysis.router_names import GATE_LABELS, HAIKU, NAMED_ROUTERS, cascade_section, parse_cascade
from analysis.router_table import RouterScoreTable
from analysis.simulator import Trace

if TYPE_CHECKING:
    from analysis.offline_half import Inputs

__all__ = ["ROUTED", "ROUTER_TABLES", "FinalRoutedError", "routed_part", "routed_traces_of"]

# The twelve sections, in item 8's order: E, F, G and G+, each with the free checks, the Jev gate and the Sonnet gate.
ROUTED = tuple(cascade_section(start, gate) for start in NAMED_ROUTERS.values() for gate in GATE_LABELS)


class FinalRoutedError(ValueError):
    """A section that is no router-started cascade, or inputs that lack a table the cascade reads. Nothing is built."""


def routed_traces_of(
    router: RouterScoreTable, table: GateScoreTable, gate: str
) -> Callable[[FrozenPoint], list[Trace]]:
    """`traces_of` for `operating_eval.evaluate`: the cascade's traces at a point's threshold and cutoff. Refuses a point that
    routes and holds no threshold (FrozenPointError), and what `cascade_trace` refuses (GateTableError): a cutoff on the free
    checks, a paid gate with none, a gate no amendment names, and a gate call the table has no entry for."""

    return at_threshold(lambda point, threshold: routed_traces(router, table, gate, point.cutoff, threshold))


def routed_part(
    section: str, *, per_task: bool = False, path: Path = OPERATING_POINTS_FILE
) -> Callable[["Inputs"], dict]:
    """A router-started cascade's part: its six frozen points evaluated on the split, under both mixes (table v4's rows, which
    dev committed) or with `per_task` per task. `path` is the operating-points file (a rehearsal's is made up)."""
    try:
        start, gate = parse_cascade(section)
    except ValueError as error:
        raise FinalRoutedError(str(error)) from None
    if start == HAIKU or start not in ROUTER_TABLES:
        raise FinalRoutedError(f"{section} is not a router-started cascade ({list(ROUTED)})")

    def build(inputs: "Inputs") -> dict:
        name = ROUTER_TABLES[start]
        if name not in inputs.extras or GATE_TABLE not in inputs.extras:
            raise FinalRoutedError(f"{section} reads the extras {name!r} and {GATE_TABLE!r}; the inputs hold "
                                   f"{sorted(inputs.extras)}")  # fmt: skip
        router: RouterScoreTable = inputs.extras[name]
        gates: GateScoreTable = inputs.extras[GATE_TABLE].table
        return cascade_rows(
            inputs, frozen_points(section, path), routed_traces_of(router, gates, gate), per_task=per_task
        )

    return build
