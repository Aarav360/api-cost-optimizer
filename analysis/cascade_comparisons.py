"""Stage 7's comparisons on dev and the best cascade (PLAN_v4.md §9, §10; Amendment A17, items 22 and 24; issue #71).

Item 22 declares 51 same-target comparisons in five groups, descriptive with no multiplicity correction. Each pair is
two walks' operating points at one mix and target, first less second, paired over the same resamples
(`analysis.same_target`). This module names the pairs, in item 22's order and with the first side as the question puts
it, and applies item 24's rule for the best cascade; `analysis.table_v4` computes and writes the numbers.

A walk is named by a router's letter (E, F, G, G+) or a cascade's section key (`analysis.router_names.CASCADES`).
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from analysis.bootstrap import Bootstrap, Savings
from analysis.jev_comparisons import same_target_cells
from analysis.references import ROUNDING
from analysis.router_names import (
    CASCADES,
    FREE_GATE,
    GATE_LABELS,
    HAIKU,
    JEV_GATE,
    NAMED_ROUTERS,
    SONNET_GATE,
    cascade_section,
    parse_cascade,
)
from analysis.router_panels import routing_mix
from analysis.same_target import SameTarget
from analysis.simulator import Outcome, item_weights, totals
from analysis.walk import Walk
from routers.ladder import FRONTIER

# The routers' letters, E, F, G and G+ in item 8's order: the walks' names for the routers alone (`chart_routers.ROUTERS`
# is the chart's list of router entries, another thing).
ROUTER_LETTERS = tuple(NAMED_ROUTERS)
POINTS = 6  # operating points per cascade: two mixes, three quality targets
# Item 24's "cheaper gate", cheapest first: the free checks cost nothing, and a Jev gate call costs less than a Sonnet
# one (A17 item 15 prices both).
GATE_COST_ORDER = (FREE_GATE, JEV_GATE, SONNET_GATE)


class ComparisonError(ValueError):
    """Savings or pairs the comparisons refuse whole. Nothing is chosen."""


def label(name: str) -> str:
    """A walk's label: a router keeps its letter, a cascade its label (`E/Jev`, `haiku/free`)."""
    return CASCADES.get(name, name)


def comparison_pairs() -> dict[str, list[tuple[str, str]]]:
    """Item 22's 51 pairs by group, each (first, second), the comparison being first less second.

    - `gate_adds`: each router-started cascade against its router alone (12);
    - `router_adds`: each router-started cascade against the pure cascade with the same gate (12);
    - `paid_gate_over_free`: `haiku/Jev` and `haiku/Sonnet` against `haiku/free`, and each router's `/Jev` and `/Sonnet`
      against its `/free` (10);
    - `jev_against_sonnet`: `haiku/Jev` against `haiku/Sonnet` and each router's `/Jev` against its `/Sonnet` (5);
    - `cascades_against_routers`: each pure cascade against each router alone (12).
    """
    starts = [HAIKU, *NAMED_ROUTERS.values()]
    gate_adds = [
        (cascade_section(section, gate), letter) for letter, section in NAMED_ROUTERS.items() for gate in GATE_LABELS
    ]
    router_adds = [
        (cascade_section(section, gate), cascade_section(HAIKU, gate))
        for section in NAMED_ROUTERS.values()
        for gate in GATE_LABELS
    ]
    paid_over_free = [
        (cascade_section(start, gate), cascade_section(start, FREE_GATE))
        for start in starts
        for gate in (JEV_GATE, SONNET_GATE)
    ]
    jev_sonnet = [(cascade_section(start, JEV_GATE), cascade_section(start, SONNET_GATE)) for start in starts]
    against_routers = [(cascade_section(HAIKU, gate), letter) for gate in GATE_LABELS for letter in ROUTER_LETTERS]
    return {
        "gate_adds": gate_adds, "router_adds": router_adds, "paid_gate_over_free": paid_over_free,
        "jev_against_sonnet": jev_sonnet, "cascades_against_routers": against_routers,
    }  # fmt: skip


Walks = Mapping[str, Mapping[float, Walk]]  # one policy's walks, by mix and then target share


def compare_pairs(
    draws: Bootstrap, mixes: Mapping[str, dict[str, float]], walks: Mapping[str, Walks]
) -> dict[str, dict[tuple[str, str], dict[str, dict[float, SameTarget]]]]:
    """Item 22's 51 comparisons: group, then the pair `(first, second)` by the walks' names, then mix, then target. Each
    is the first walk's operating point less the second's, paired over `draws`, with `analysis.same_target`'s fallback
    rules (a side that falls back is A; both reads "both fall back to A"). The pair is kept as a pair, so a caller that
    labels it (`label`) never has to split a label back. Refuses (ComparisonError) a router or cascade with no walks."""
    groups = comparison_pairs()
    needed = {name for pairs in groups.values() for pair in pairs for name in pair}
    if missing := sorted(needed - set(walks)):
        raise ComparisonError(f"no walks for {', '.join(label(m) for m in missing)}")
    cells = same_target_cells(draws, mixes, walks, tuple(pair for pairs in groups.values() for pair in pairs))
    return {group: {pair: cells[pair] for pair in pairs} for group, pairs in groups.items()}


# ---- a cascade's row, in table v1's style (A9 item 15) ---------------------------------------------------------


def escalation_rate(outcomes: Sequence[Outcome], mix: dict[str, float]) -> float:
    """The share of requests under the mix whose first answer was rejected and sent on: the weight of the outcomes that
    called more than one config. An item a router sent straight to `opus` is no escalation (it was never rejected)."""
    weights = item_weights(list(outcomes), mix)
    return sum(weights[o.item_id] for o in outcomes if len(o.configs) > 1)


def policy_row(
    draws: Bootstrap, savings: Savings, walk: Walk, fixed: Mapping[str, list[Outcome]], mix: dict[str, float]
) -> dict:
    """One walk's operating point as table v1's row (`analysis.router_panels.router_part_of_mix_panel`): where the walk
    put it (the router's threshold, the cutoff, or both for a router-started cascade, null for A), quality and cost with
    intervals, the margin, the paired difference from A, savings against A, the captured fraction with the share of
    resamples meeting the target, the routing mix and the matched-cost comparison with the envelope; and, beside the
    routing mix, the escalation rate. A walk that fell back is A: no threshold or cutoff, no escalation."""
    point = walk.operating
    outcomes = list(point.outcomes)
    frontier = list(fixed[FRONTIER])
    frontier_total, total = totals(frontier, mix), totals(outcomes, mix)
    return {
        "threshold": point.threshold,
        "cutoff": point.cutoff,
        "fell_back": walk.fell_back,
        "steps_certified": len(walk.certified) - 1,  # A is certified without a step
        "steps_tried": len(walk.steps) - 1,
        "target": walk.target,
        "margin": point.margin,
        **asdict(draws.totals(outcomes, mix)),
        "difference_from_A": asdict(draws.difference(outcomes, frontier, mix)),
        "savings_vs_A": {
            "per_1k": frontier_total.cost_per_1k - total.cost_per_1k,
            "share": 1 - total.cost_per_1k / frontier_total.cost_per_1k,
        },
        "captured": asdict(savings.captured(outcomes, walk.share)),
        "routing": routing_mix(outcomes, mix),
        "escalation_rate": escalation_rate(outcomes, mix),
        "matched_cost": asdict(draws.matched_cost(outcomes, dict(fixed), mix)),
    }  # fmt: skip


# ---- item 24: the best cascade ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class BestCascade:
    """The outcome of item 24's rule. `best` is None where the rule leaves a tie open (`open_case`)."""

    best: str | None  # a cascade's section key
    means: Mapping[str, float]  # each cascade's mean savings over its six operating points, in item 8's order
    tied: tuple[str, ...]  # the cascades at the highest mean (within rounding), in item 8's order
    open_case: bool  # the rule's tie-breaks left more than one cascade


def _gate_of(key: str) -> str:
    return parse_cascade(key)[1]


def _is_pure(key: str) -> bool:
    return parse_cascade(key)[0] == HAIKU


def best_cascade(savings: Mapping[str, Sequence[float]]) -> BestCascade:
    """Item 24's rule: the cascade with the highest mean dev savings against A over its six operating points, a fall
    back to A counting 0. Ties (means within rounding) go to a pure cascade, then to the cheaper gate (the free checks,
    then Jev, then Sonnet). Where that leaves more than one (two router-started cascades with the same gate), the rule is
    silent: `best` is None and `open_case` says so, rather than any judgement choosing one.

    `savings` maps each of the fifteen cascades' keys to its six savings. Refuses (ComparisonError) anything else."""
    if set(savings) != set(CASCADES):
        raise ComparisonError(
            f"savings must be for exactly the fifteen cascades, not {sorted(set(savings) ^ set(CASCADES))}"
        )
    if bad := [key for key, values in savings.items() if len(values) != POINTS]:
        raise ComparisonError(f"each cascade has {POINTS} operating points, but {bad[0]} has {len(savings[bad[0]])}")
    means = {key: sum(savings[key]) / POINTS for key in CASCADES}
    top = max(means.values())
    tied = tuple(key for key in CASCADES if means[key] >= top - ROUNDING)
    pool = [key for key in tied if _is_pure(key)] or list(tied)
    cheapest = min(GATE_COST_ORDER.index(_gate_of(key)) for key in pool)
    pool = [key for key in pool if GATE_COST_ORDER.index(_gate_of(key)) == cheapest]
    return BestCascade(pool[0] if len(pool) == 1 else None, means, tied, len(pool) > 1)
