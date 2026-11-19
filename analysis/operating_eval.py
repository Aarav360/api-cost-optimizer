"""A policy evaluated at its frozen operating points, with no walk (PLAN_v4.md Amendment A19, items 6 and 8; issue #83).

Dev's tables come out of a walk, which picks each operating point from the outcomes. Test's come from the points the
operating-points file already holds, and test chooses nothing. This module is the piece that evaluates a policy at those
points on any split, and a ticket for a later policy (G, G+, a cascade) reuses it with its own traces:

    points = frozen_points("rules")                       # the file's six points: mix, target, dials, fell_back
    outcomes = evaluate(matrix, fixed["opus"], points, traces_of)  # {point: the policy's outcomes there}
    rows = mix_rows(draws, savings_of, fixed, outcomes)   # {mix: {"99%": row}}, table v1's numbers
    by_task = task_rows(draws, fixed, outcomes)           # {task: {mix: {"99%": row}}}

`traces_of(point)` is the only policy-specific part: a router's is `lambda p: table.traces(p.threshold)`, a router-started
cascade's builds its traces from the point's threshold and cutoff.

**The file's `fell_back` decides, never the split's own costs.** A walk's `Walk.fell_back` re-derives the flag from the
steps it tried, so a frozen point costing no less than A on test would be called "falls back" by a hand-built walk, and a
point the file says fell back could look like a cheaper policy. Here a point the file says fell back is A itself (the very
same outcomes, so McNemar's test is skipped as in the dev tables), and any other point is the policy at its dials. The row
carries the file's `fell_back`, the label "falls back to A" is drawn from it, and a row's `target` is the share of A's quality
under the mix on this split. Nothing in a row is a walk's: no margin, no steps tried (a margin of zero is #88's "target met").

Run on dev, the rows are the numbers table v1 committed for E and F (`analysis/check_final_run_dev.py`).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from analysis.bootstrap import Bootstrap, Savings, TotalEstimate
from analysis.matrix import Matrix
from analysis.operating_points import OPERATING_POINTS_FILE, OperatingPointsError, read_section, target_key
from analysis.references import TARGETS, quality_targets
from analysis.router_panels import point_numbers, task_point
from analysis.simulator import Outcome, Trace, always, load_mixes, simulate, totals
from routers.ladder import FRONTIER, LADDER

MIXES = ("realistic", "uniform")  # the file's two mixes, in the figures' panel order


class FrozenPointError(ValueError):
    """A frozen point that can't be evaluated: the file's section is malformed, or the policy has no dial it needs."""


@dataclass(frozen=True)
class FrozenPoint:
    """One operating point as the file holds it: a section's mix and target, its dials, and whether it fell back."""

    section: str
    mix: str
    share: float  # the quality target as a share of A's quality: 1.0, 0.99 or 0.97
    threshold: float | None  # a router's dial (None: falls back, or a pure cascade)
    cutoff: float | None  # a gate's dial (None: no paid gate, or falls back)
    fell_back: bool  # the file's flag, frozen with the point

    @property
    def target(self) -> str:
        return target_key(self.share)


def frozen_points(section: str, path: Path = OPERATING_POINTS_FILE) -> tuple[FrozenPoint, ...]:
    """The section's points in the figures' order (mix, then target from 100% down), read as the file holds them. A
    point that fell back holds no dial, and a section that lacks a mix or a target is refused."""
    held = read_section(section, path)
    points = []
    for mix in MIXES:
        for share in TARGETS:
            try:
                entry = held[mix][target_key(share)]
                fell_back = entry["fell_back"]
            except KeyError as error:
                raise OperatingPointsError(f"section {section!r} has no point for {mix} {target_key(share)} "
                                           f"({error})") from None  # fmt: skip
            threshold, cutoff = entry.get("threshold"), entry.get("cutoff")
            if not isinstance(fell_back, bool):
                raise FrozenPointError(f"{section} {mix} {target_key(share)}: fell_back is {fell_back!r}, not a bool")
            if fell_back and (threshold is not None or cutoff is not None):
                raise FrozenPointError(f"{section} {mix} {target_key(share)} fell back to A but holds a dial")
            points.append(FrozenPoint(section, mix, share, threshold, cutoff, fell_back))
    return tuple(points)


def fixed_outcomes(matrix: Matrix) -> dict[str, list[Outcome]]:
    """Each config always, as a policy: the references' own outcomes (A is `opus`)."""
    return {config: simulate(matrix, always(matrix, config)) for config in LADDER}


# The offline half's `Inputs.cache` holds what its parts share, so the references and every policy's part build the fixed
# configs' outcomes and each mix's resampled oracles (the slow part: one oracle per resample) once.
def shared_fixed(cache: dict, matrix: Matrix) -> dict[str, list[Outcome]]:
    if "fixed" not in cache:
        cache["fixed"] = fixed_outcomes(matrix)
    fixed: dict[str, list[Outcome]] = cache["fixed"]
    return fixed


def shared_savings(cache: dict, matrix: Matrix, draws: Bootstrap, mix: str) -> Savings:
    if ("savings", mix) not in cache:
        cache["savings", mix] = draws.savings(matrix, shared_fixed(cache, matrix)[FRONTIER], load_mixes()[mix])
    savings: Savings = cache["savings", mix]
    return savings


def router_traces(table_traces: Callable[[float], list[Trace]]) -> Callable[[FrozenPoint], list[Trace]]:
    """`traces_of` for a router: its table's traces at the point's threshold, refusing a point with none."""

    def traces_of(point: FrozenPoint) -> list[Trace]:
        if point.threshold is None:
            raise FrozenPointError(
                f"{point.section} {point.mix} {point.target} does not fall back but has no threshold"
            )
        return table_traces(point.threshold)

    return traces_of


def evaluate(
    matrix: Matrix,
    frontier: list[Outcome],
    points: tuple[FrozenPoint, ...],
    traces_of: Callable[[FrozenPoint], list[Trace]],
) -> dict[FrozenPoint, list[Outcome]]:
    """The policy's outcomes at each frozen point. A point that fell back gets `frontier` itself (A's outcomes, the
    very list `fixed_outcomes` gave, so McNemar's test finds it is A); any other is simulated from `traces_of(point)`,
    once for each distinct pair of dials (the two mixes and three targets share some)."""
    simulated: dict[tuple[float | None, float | None], list[Outcome]] = {}
    outcomes: dict[FrozenPoint, list[Outcome]] = {}
    for point in points:
        if point.fell_back:
            outcomes[point] = frontier
            continue
        dials = (point.threshold, point.cutoff)
        if dials not in simulated:
            simulated[dials] = simulate(matrix, traces_of(point))
        outcomes[point] = simulated[dials]
    return outcomes


def mix_rows(
    draws: Bootstrap,
    savings_of: Callable[[str], Savings],
    fixed: dict[str, list[Outcome]],
    outcomes: Mapping[FrozenPoint, list[Outcome]],
    mixes: Mapping[str, dict[str, float]],
) -> dict[str, dict[str, dict]]:
    """Each point's row under its mix: the file's dials and flag, the target on this split (`share` of A's quality
    there), and table v1's numbers. `savings_of(mix)` gives the mix's resampled oracles, solved once and shared."""
    rows: dict[str, dict[str, dict]] = {}
    for point, found in outcomes.items():
        mix = mixes[point.mix]
        row = {
            "threshold": point.threshold,
            "fell_back": point.fell_back,
            "target": quality_targets(totals(fixed[FRONTIER], mix).quality)[point.share],
        }
        if point.cutoff is not None:
            row["cutoff"] = point.cutoff
        rows.setdefault(point.mix, {})[point.target] = {
            **row,
            **point_numbers(draws, savings_of(point.mix), found, fixed, mix, point.share),
        }
    return rows


def task_rows(
    draws: Bootstrap, fixed: dict[str, list[Outcome]], outcomes: Mapping[FrozenPoint, list[Outcome]], tasks: list[str]
) -> dict[str, dict[str, dict[str, dict]]]:
    """Each point's row in each task: the policy's quality and cost there with intervals and McNemar's test against A
    (None where the point fell back). Keyed by task, then mix, then target."""
    frontier = fixed[FRONTIER]
    rows: dict[str, dict[str, dict[str, dict]]] = {task: {} for task in tasks}
    spreads: dict[tuple[bool, float | None, float | None, str], TotalEstimate] = {}
    for task in tasks:
        for point, found in outcomes.items():
            key = (point.fell_back, point.threshold, point.cutoff, task)
            if key not in spreads:
                spreads[key] = draws.task_totals(found)[task]
            rows[task].setdefault(point.mix, {})[point.target] = task_point(
                point.threshold, point.fell_back, spreads[key], found, frontier, task
            )
    return rows
