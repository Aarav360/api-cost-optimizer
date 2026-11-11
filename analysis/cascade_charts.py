"""The cascades' parts of chart v4: step lines, operating points and the router-started panels (PLAN_v4.md §12;
Amendment A17, items 8 and 23).

Everything here is JSON-ready data that `analysis/pareto.py` adds to the headline and per-task panels under the key
`cascades`, and that `analysis/routed_figure.py` plots in the router-started figure. It adds keys and changes none of
table v1's, so a router's part of chart v3 stays as it was.

A cascade's **line** is its whole walk laid out: A first, then every step in walk order, strictest first, each with the
outcomes the simulator gives it on dev. The walks themselves stop at the first step that misses (they never simulate
past it), but a chart draws the whole line, as chart v1 drew E's whole line: the candidates are the walk's own
(`gate_walk.pure_candidates` for a pure cascade, `routed_walk.routed_candidates` for a router-started one), each
simulated here. A pure cascade's steps are its gate's cutoffs and a router-started cascade's are its router's thresholds
at the gate's fixed cutoff, so a router and its cascades are drawn on the same steps.

The operating points are not computed here: they are table v4's rows (`figures/table_v4.json`), copied in so the figure
JSON holds every number the figure plots. `analysis/check_chart_v4.py` checks that each cross sits on a step of its line.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from analysis.bootstrap import Bootstrap
from analysis.gate_table import GateScoreTable
from analysis.gate_walk import pure_candidates
from analysis.matrix import Matrix
from analysis.routed_walk import routed_candidates
from analysis.router_names import (
    CASCADE_STARTS,
    CASCADES,
    FREE_GATE,
    GATE_LABELS,
    HAIKU,
    LABELS,
    NAMED_ROUTERS,
    cascade_section,
)
from analysis.router_table import RouterScoreTable
from analysis.simulator import Outcome, always, simulate, task_totals, totals
from analysis.walk import Candidate
from routers.ladder import FRONTIER

Step = dict[str, float | None]


@dataclass(frozen=True)
class Line:
    """One cascade's walk laid out in full: A (no threshold, no cutoff), then each candidate's dial and outcomes."""

    steps: tuple[tuple[float | None, float | None, tuple[Outcome, ...]], ...]


def frontier_outcomes(dev: Matrix) -> list[Outcome]:
    return simulate(dev, always(dev, FRONTIER))


def line_of(dev: Matrix, candidates: Sequence[Candidate], frontier: Sequence[Outcome]) -> Line:
    """A, then every candidate in order, simulated on `dev`."""
    steps: list[tuple[float | None, float | None, tuple[Outcome, ...]]] = [(None, None, tuple(frontier))]
    steps += [(c.threshold, c.cutoff, tuple(simulate(dev, c.traces()))) for c in candidates]
    return Line(tuple(steps))


def pure_line(dev: Matrix, table: GateScoreTable, gate: str, frontier: Sequence[Outcome]) -> Line:
    """The pure cascade ending in `gate`: A, then its gate's cutoffs, strictest first (one step for the free checks)."""
    return line_of(dev, pure_candidates(table, gate), frontier)


def routed_line(
    dev: Matrix,
    router: RouterScoreTable,
    table: GateScoreTable,
    gate: str,
    cutoff: float | None,
    frontier: Sequence[Outcome],
) -> Line:
    """A router-started cascade: A, then its router's thresholds, highest first, each at the gate's fixed cutoff."""
    return line_of(dev, routed_candidates(router, table, gate, cutoff), frontier)


def mix_steps(line: Line, mix: Mapping[str, float]) -> list[Step]:
    """The line's steps under a task mix: the dials and the mix-weighted quality and cost per 1,000 requests."""
    rows = []
    for threshold, cutoff, outcomes in line.steps:
        total = totals(list(outcomes), dict(mix))
        rows.append(
            {"threshold": threshold, "cutoff": cutoff, "quality": total.quality, "cost_per_1k": total.cost_per_1k}
        )
    return rows


def task_steps(line: Line, task: str) -> list[Step]:
    """The line's steps within one task, on their own."""
    rows = []
    for threshold, cutoff, outcomes in line.steps:
        total = task_totals(list(outcomes))[task]
        rows.append(
            {"threshold": threshold, "cutoff": cutoff, "quality": total.quality, "cost_per_1k": total.cost_per_1k}
        )
    return rows


def describe(key: str, label: str, gate: str) -> dict[str, Any]:
    return {"label": label, "gate": gate, "free_only": gate == FREE_GATE}


def pure_part_of_mix_panel(
    key: str,
    label: str,
    gate: str,
    line: Line,
    draws: Bootstrap,
    mix: Mapping[str, float],
    points: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """A pure cascade's part of a headline panel: its steps for the curve and its operating points for the crosses.

    The free checks alone have one step and are drawn as that one point with its cross (A17 item 23): its `point` holds
    the step's quality and cost with their 95% intervals. Its operating points are table v4's rows, which fall back to A
    wherever the one step misses."""
    part: dict[str, Any] = {**describe(key, label, gate), "steps": mix_steps(line, mix)}
    if gate == FREE_GATE:
        outcomes = list(line.steps[1][2])
        part["point"] = asdict(draws.totals(outcomes, dict(mix)))
    part["operating_points"] = {target: dict(row) for target, row in points.items()}
    return part


def pure_part_of_task_panel(key: str, label: str, gate: str, line: Line, task: str) -> dict[str, Any]:
    """A pure cascade's part of a per-task panel: its curve, with no operating points (A7 item 2)."""
    return {**describe(key, label, gate), "steps": task_steps(line, task)}


def routed_line_part(
    key: str,
    label: str,
    gate: str,
    line: Line,
    mix: Mapping[str, float],
    points: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """One router-started cascade's line on a router-started panel: its steps and operating points."""
    return {
        **describe(key, label, gate),
        "steps": mix_steps(line, mix),
        "operating_points": {target: dict(row) for target, row in points.items()},
    }


def routed_keys() -> dict[str, list[tuple[str, str]]]:
    """Each named router's three cascades, as (section key, gate), in item 8's order: {router section: [...]}."""
    return {name: [(cascade_section(name, gate), gate) for gate in GATE_LABELS] for name in NAMED_ROUTERS.values()}


# ---- all fifteen lines, and the figures' parts ------------------------------------------------------------------


def pure_keys() -> dict[str, str]:
    """The three pure cascades' section keys and their gates, in item 8's order."""
    return {cascade_section(HAIKU, gate): gate for gate in GATE_LABELS}


def lines_of(
    dev: Matrix,
    table: GateScoreTable,
    routers: Mapping[str, RouterScoreTable],
    cutoffs: Mapping[str, float],
) -> dict[str, Line]:
    """The fifteen cascades' lines on dev, by section key in item 8's order: the pure three, then each router's free,
    Jev and Sonnet cascades. `routers` maps each named router's section name to its dev router-score table, and
    `cutoffs` each paid gate to its fixed cutoff."""
    frontier = frontier_outcomes(dev)
    lines: dict[str, Line] = {}
    for start in CASCADE_STARTS:
        for gate in GATE_LABELS:
            key = cascade_section(start, gate)
            if start == HAIKU:
                lines[key] = pure_line(dev, table, gate, frontier)
            else:
                cutoff = None if gate == FREE_GATE else cutoffs[gate]
                lines[key] = routed_line(dev, routers[start], table, gate, cutoff, frontier)
    return lines


def points_of(table_v4: Mapping[str, Any], key: str, mix: str) -> Mapping[str, Mapping[str, Any]]:
    """A cascade's operating points under a mix, table v4's rows, keyed by target."""
    return table_v4["cascades"][key]["operating_points"][mix]


def headline_parts(
    lines: Mapping[str, Line], draws: Bootstrap, mix_name: str, mix: Mapping[str, float], table_v4: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """The pure cascades' parts of one headline panel, by section key."""
    return {
        key: pure_part_of_mix_panel(
            key, CASCADES[key], gate, lines[key], draws, mix, points_of(table_v4, key, mix_name)
        )
        for key, gate in pure_keys().items()
    }


def task_parts(lines: Mapping[str, Line], task: str) -> dict[str, dict[str, Any]]:
    """The pure cascades' parts of one per-task panel, by section key."""
    return {
        key: pure_part_of_task_panel(key, CASCADES[key], gate, lines[key], task) for key, gate in pure_keys().items()
    }


def routed_panel(
    lines: Mapping[str, Line],
    headline_panel: Mapping[str, Any],
    mix_name: str,
    mix: Mapping[str, float],
    table_v4: Mapping[str, Any],
) -> dict[str, Any]:
    """One mix's router-started panel: A, the envelope and the targets from the headline panel, and for each router four
    lines on the same steps: the router alone (its part of the headline panel, copied) and its three cascades."""
    routers = {}
    for letter, name in NAMED_ROUTERS.items():
        alone = headline_panel["routers"][name]
        lines_out: dict[str, Any] = {
            name: {
                "label": letter,
                "gate": None,
                "steps": alone["steps"],
                "operating_points": alone["operating_points"],
            }
        }
        for key, gate in routed_keys()[name]:
            lines_out[key] = routed_line_part(
                key, CASCADES[key], gate, lines[key], mix, points_of(table_v4, key, mix_name)
            )
        routers[name] = {"label": LABELS[letter], "short": letter, "lines": lines_out}
    return {
        "A": headline_panel["configs"][FRONTIER],
        "envelope": headline_panel["envelope"],
        "targets": headline_panel["targets"],
        "routers": routers,
    }


# ---- what the figure JSONs say about the cascades ---------------------------------------------------------------

CASCADE_UNITS = {
    "cascades": "the fifteen cascades (A17 item 8), keyed by section key: label, start (the pure cascades start at "
    "haiku, the others at a router), gate (the free checks, the Jev gate or the Sonnet gate), the operating-points "
    "file that holds the section and the sha256 of the section, written with sorted keys",
    "panel_cascades": "a panel's `cascades` (the headline and per-task figures hold the three pure cascades): per "
    "cascade its label, gate and steps in walk order, A first (threshold and cutoff null), then each step from the "
    "strictest cutoff down, each with its quality and cost_per_1k on dev (mix-weighted in the headline figure), joined "
    "by straight lines in linear cost and never hulled; the walks stop at the first step that misses, the lines do not",
    "point": "the free-checks-only cascade's one step as a point with a cross: its quality and cost_per_1k, each its "
    "value on dev with its percentile 95% interval (low, high) and standard error; the cascade's operating points "
    "fall back to A wherever the step misses",
    "operating_points": "table v4's rows (figures/table_v4.json), keyed by quality target: threshold (the router's), "
    "cutoff (the gate's), fell_back, quality and cost_per_1k with intervals, margin, difference from A, savings against "
    "A, captured fraction, routing mix, escalation rate and the matched-cost comparison",
    "routed_panels": "the router-started figure's panels, one per mix: A (the opus config's row), the envelope's corners, "
    "the targets, and per router four lines on the router's steps: the router alone (its part of the headline panel) "
    "and its three cascades, each with steps and operating points",
}


def cascade_meta(table_v4: Mapping[str, Any]) -> dict[str, Any]:
    """The cascades' entries for the figure JSONs: label, start, gate and the section's hash in the committed file."""
    from analysis.operating_points import OPERATING_POINTS_FILE, read_section, section_hash
    from harness.client import PROJECT_ROOT

    where = str(OPERATING_POINTS_FILE.relative_to(PROJECT_ROOT))
    return {
        key: {
            "label": CASCADES[key],
            "start": table_v4["cascades"][key]["start"],
            "gate": table_v4["cascades"][key]["gate"],
            "operating_points_file": where,
            "operating_points_sha256": section_hash(read_section(key)),
        }
        for key in CASCADES
    }
