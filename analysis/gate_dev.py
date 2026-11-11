"""Stage 7's dev work after the gate calls: the fifteen walks, their sections and the committed file (PLAN_v4.md §9;
Amendment A17, items 8 to 11, 20.5 and 25; issue #70).

Dev's gate-score table, the four routers' dev router-score tables and the two fixed cutoffs go in; fifteen walks, one
per cascade, come out, and from them the fifteen sections of `routers/operating_points.json`. Nothing here calls a gate:
the table is built from the stored records (`analysis.gate_dev_run`), and the walks read the table and the matrix alone.

The walks are the ones Stage 4 built, with a new trace builder (`analysis.gate_walk` for the pure cascades,
`analysis.routed_walk` for the router-started ones). `write_sections` is the one place the cascade sections reach the
committed file, and it leaves E's, F's, G's and G+'s sections as they were (A17 item 25: the allowlist adds the fifteen).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from analysis.bootstrap import Bootstrap
from analysis.gate_table import PAID_GATES, GateScoreTable
from analysis.gate_walk import MIXES, pure_section, pure_walks
from analysis.matrix import Matrix
from analysis.operating_points import (
    OPERATING_POINTS_FILE,
    OperatingPointsError,
    _read_file,
    check_sections,
    write_section,
)
from analysis.references import ROUNDING, TARGETS
from analysis.routed_walk import routed_section, routed_walks
from analysis.router_names import (
    CASCADE_STARTS,
    CASCADES,
    FREE_GATE,
    GATE_LABELS,
    HAIKU,
    NAMED_ROUTERS,
    cascade_section,
    parse_cascade,
)
from analysis.router_table import RouterScoreTable
from analysis.simulator import always, load_mixes, simulate
from analysis.walk import Walk
from routers.ladder import FRONTIER

Walks = dict[str, dict[float, Walk]]  # one cascade's walks: mix name, then quality target share


def cascade_walks(
    dev: Matrix,
    table: GateScoreTable,
    routers: Mapping[str, RouterScoreTable],
    cutoffs: Mapping[str, float],
    draws: Bootstrap,
) -> dict[str, Walks]:
    """The fifteen cascades' walks on dev, by their section keys in item 8's order: the pure cascades (`haiku/free`,
    `haiku/Jev`, `haiku/Sonnet`), then E, F, G and G+ each with the free checks, the Jev gate and the Sonnet gate.

    `routers` maps each named router's section name to its dev router-score table; `cutoffs` maps each paid gate to its
    fixed cutoff (A17 item 12). Refuses (ValueError) a router or a gate that is missing or that no amendment names."""
    if set(routers) != set(NAMED_ROUTERS.values()):
        raise ValueError(f"the routers must be exactly {sorted(NAMED_ROUTERS.values())}, not {sorted(routers)}")
    if set(cutoffs) != set(PAID_GATES):
        raise ValueError(f"the fixed cutoffs must be exactly those of {list(PAID_GATES)}, not {sorted(cutoffs)}")
    walks: dict[str, Walks] = {}
    for start in CASCADE_STARTS:
        for gate in GATE_LABELS:
            key = cascade_section(start, gate)
            if start == HAIKU:
                walks[key] = pure_walks(dev, table, gate, draws)
            else:
                cutoff = None if gate == FREE_GATE else cutoffs[gate]
                walks[key] = routed_walks(dev, routers[start], table, gate, cutoff, draws)
    return walks


def cascade_sections(walks: Mapping[str, Walks]) -> dict[str, dict[str, Any]]:
    """The fifteen sections from the fifteen walks, under the same keys. Refuses (ValueError) walks that are not
    exactly the cascades' keys."""
    if list(walks) != list(CASCADES):
        raise ValueError(f"the walks must be the fifteen cascades {list(CASCADES)}, not {list(walks)}")
    sections = {}
    for start in CASCADE_STARTS:
        for gate in GATE_LABELS:
            key = cascade_section(start, gate)
            sections[key] = pure_section(walks[key], gate) if start == HAIKU else routed_section(walks[key], gate)
    return sections


# ---- the rules a section must keep ----------------------------------------------------------------------------


def dial_of(key: str) -> str:
    """The dial a cascade's steps are ordered by: the router's threshold for a router-started cascade, the gate's
    cutoff for a pure one (its free-checks-only form has none, its two steps being A and the one policy)."""
    start, _ = parse_cascade(key)
    return "cutoff" if start == HAIKU else "threshold"


def section_problems(section: Mapping[str, Any], key: str) -> list[str]:
    """What is wrong with a cascade's section by the rules of A9 item 9 (as A17 items 10 and 11 keep them): the walk
    starts at A, every certified step (A included) clears its target by its margin, the walk ends at the first step
    that misses, the steps run from the strictest down, and the operating point is the cheapest certified step, ties to
    the earlier. `key` is the cascade's section key."""
    dial = dial_of(key)
    free_pure = key == cascade_section(HAIKU, FREE_GATE)
    found = []
    for mix, by_target in section.items():
        for target_key, point in by_target.items():
            steps, target, where = point["steps"], point["target"], f"{key} {mix} {target_key}"
            stop = next((i for i, step in enumerate(steps) if not step["clears"]), len(steps))
            certified, missed = steps[:stop], steps[stop:]
            if steps[0].get(dial) is not None or steps[0]["margin"] != 0:
                found.append(f"{where}: the walk does not start at A")
            if any(step["quality"] - target < step["margin"] - ROUNDING for step in certified):
                found.append(f"{where}: a certified step misses its target by its margin")
            if missed and (len(missed) > 1 or missed[0]["quality"] - target >= missed[0]["margin"] - ROUNDING):
                found.append(f"{where}: the walk goes on past the first step that misses")
            if not free_pure:
                dials = [step[dial] for step in steps[1:]]
                if dials != sorted(dials, reverse=True):
                    found.append(f"{where}: the steps do not run from the strictest {dial} down")
            if not certified:  # A always clears its own target by a margin of 0, so this is a broken section
                found.append(f"{where}: the walk certifies no step, not even A")
                continue
            cheapest = certified[0]
            for step in certified[1:]:
                if step["cost_per_1k"] < cheapest["cost_per_1k"] - ROUNDING:
                    cheapest = step
            point_dial = point.get(dial)
            if point_dial != cheapest.get(dial) or point["fell_back"] != (cheapest is steps[0]):
                found.append(f"{where}: the operating point is not the cheapest certified step")
    return found


def captured_problems(values: Sequence[float | None]) -> list[str]:
    """Captured fractions that are not between 0 and 1; n/a (None) is allowed."""
    return [f"{v}" for v in values if v is not None and not 0 <= v <= 1 + ROUNDING]


def captured_fractions(
    walks: Mapping[str, Walks], dev: Matrix, draws: Bootstrap
) -> dict[str, dict[str, dict[float, float | None]]]:
    """Each cascade's captured fraction at each operating point, by key, mix and target share: how much of the
    oracle's savings the operating point's outcomes capture (A8 item 9), without an interval."""
    frontier = simulate(dev, always(dev, FRONTIER))
    mixes = load_mixes()
    result: dict[str, dict[str, dict[float, float | None]]] = {}
    for key, by_mix in walks.items():
        result[key] = {}
        for mix, by_share in by_mix.items():
            savings = draws.savings(dev, frontier, mixes[mix])
            result[key][mix] = {
                share: savings.captured(list(w.operating.outcomes), share).value
                for share in TARGETS
                for w in [by_share[share]]
            }
    return result


# ---- the committed file ---------------------------------------------------------------------------------------


def committed_problems(path: Path = OPERATING_POINTS_FILE) -> list[str]:
    """What is wrong with the operating-points file once the cascades are in: it holds exactly E's, F's, G's and G+'s
    sections and the fifteen cascades', and each cascade's section has both mixes and all three targets."""
    try:
        sections = _read_file(path)
    except (OperatingPointsError, OSError) as error:
        return [str(error)]
    wanted = {*NAMED_ROUTERS.values(), *CASCADES}
    found = []
    if missing := sorted(wanted - set(sections)):
        found.append(f"sections missing: {missing}")
    if extra := sorted(set(sections) - wanted):
        found.append(f"sections that no amendment names: {extra}")
    for key in CASCADES:
        section = sections.get(key)
        if section is not None and (
            list(section) != list(MIXES) or any(set(v) != {"100%", "99%", "97%"} for v in section.values())
        ):
            found.append(f"{key}: not both mixes with all three targets")
    return found


def write_sections(sections: Mapping[str, dict[str, Any]], path: Path = OPERATING_POINTS_FILE) -> None:
    """Writes the fifteen cascade sections into the file, each through `write_section`, so the routers' sections stay as
    they were. Refuses (OperatingPointsError) anything but exactly the fifteen cascades' keys, before writing."""
    if list(sections) != list(CASCADES):
        raise OperatingPointsError(f"the fifteen cascades' sections go in together, not {list(sections)}")
    for key, section in sections.items():
        write_section(key, section, path)
    check_sections(path, required=tuple(sections))
