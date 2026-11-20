"""The 51 same-target comparisons on a split and the replication summary (PLAN_v4.md Amendment A19, item 8; A17 item 22; issue #77, #89).

Item 22 declares 51 comparisons in five groups: each router-started cascade against its router alone (12), against the pure
cascade with the same gate (12), each paid gate against the free checks (10), the Jev gate against the Sonnet gate (5) and each
pure cascade against each router alone (12). They are descriptive, with no multiplicity correction: at 95%, about one interval in
twenty excludes 0 by chance. Each is the first policy's operating point less the second's, at one mix and one quality target,
paired over the same resamples (`analysis/same_target.py`), in each of the six mix-and-target cells.

**Nothing is walked.** Dev's comparisons (`analysis/table_v4.py`) compare the walks' operating points. Test has the frozen points
of `routers/operating_points.json`, which `analysis/operating_eval.py` evaluates and notes on the offline half's shared
`Fixed`, one entry per policy and point. The part here reads every policy noted there, so it covers whatever policies the
parts before it built, and refuses a pair with a side that nobody evaluated. A point the file says fell back is A, as in dev's
table; where both sides fell back the cell reads "both fall back to A" and holds no difference. Run on dev, the part reproduces
table v4's comparisons (`committed`).

**Two orders, both kept.** The groups, and the pairs within each, are in item 22's order, which is table v4's. The write-up
leads with the declared questions (A19 item 8, `WRITEUP.md`): `haiku/Jev` against G and against G+ first, the other ten
cascades against the routers next, then the gate, the router, the paid gate and the Jev gate against the Sonnet gate. In
item 22 those two pairs stand 46th and 47th. The `declared` part holds the seven questions in the write-up's order, each with its pairs
in item 22's order, and `order` is the 51 pairs flat in that order: nothing is dropped and nothing is counted twice.

**The replication summary** (`replication`) takes each dev comparison whose interval excluded 0, in each mix-and-target cell and
for cost and quality apart, and says whether test's interval has the same sign and also excludes 0. A cell where both sides fell
back has no interval on either split and is never listed. Dev's side is table v4 as committed.
"""

import json
from collections.abc import Mapping
from typing import Any

from analysis import cascade_comparisons as cc
from analysis.operating_eval import MIXES, FrozenPoint
from analysis.references import TARGETS
from analysis.router_names import NAMED_ROUTERS
from analysis.same_target import SameTarget, compare_outcomes
from analysis.simulator import Outcome, load_mixes
from analysis.table_v2 import DESCRIPTIVE
from analysis.table_v4 import HEADLINE, QUESTIONS, TABLE_V4
from analysis.trained_diagnostics import same_target_data

# The write-up's declared questions, in its order (`WRITEUP.md`, "The declared questions"): the heading, the group of item 22
# the pairs come from, and the pairs taken (None: all of the group's). The first two pull `haiku/Jev` against G and G+ out of
# the cascades-against-routers group, which then holds the other ten.
QUESTION_HEADINGS: tuple[tuple[str, str], ...] = (
    ("haiku/Jev against G", "cascades_against_routers"),
    ("haiku/Jev against G+", "cascades_against_routers"),
    ("The other cascades against the routers", "cascades_against_routers"),
    ("What the gate adds to a router", "gate_adds"),
    ("What the router adds to a cascade", "router_adds"),
    ("What a paid gate adds over the free checks", "paid_gate_over_free"),
    ("The Jev gate against the Sonnet gate", "jev_against_sonnet"),
)
LEAD = tuple(f"{cc.label(a)} less {cc.label(b)}" for a, b in HEADLINE)  # `haiku/Jev less G`, `haiku/Jev less G+`

MEASURES = ("cost_per_1k", "quality")


class ComparisonsError(ValueError):
    """Comparisons or a replication summary the inputs can't make: a side nobody evaluated, or dev's cell missing on test."""


def section_of(name: str) -> str:
    """The operating-points section of a comparison's side: a router's letter names its section, a cascade's key is one."""
    return NAMED_ROUTERS.get(name, name)


def pair_name(pair: tuple[str, str]) -> str:
    return f"{cc.label(pair[0])} less {cc.label(pair[1])}"


def cells_of(
    draws: Any, mixes: Mapping[str, dict[str, float]], evaluated: Mapping[FrozenPoint, list[Outcome]]
) -> dict[str, dict[tuple[str, str], dict[str, dict[float, SameTarget]]]]:
    """Item 22's 51 comparisons: group, then the pair by the policies' names, then mix, then target share. Each side's
    outcomes are the policy's at its frozen point (A itself where the file says it fell back). Refuses (ComparisonsError) a
    policy with no point evaluated, or one that lacks a mix and target."""
    held = {(p.section, p.mix, p.share): (outcomes, p.fell_back) for p, outcomes in evaluated.items()}
    groups = cc.comparison_pairs()
    needed = {section_of(name) for pairs in groups.values() for pair in pairs for name in pair}
    if missing := sorted(needed - {section for section, _, _ in held}):
        raise ComparisonsError(f"no operating points were evaluated for {', '.join(missing)}: the comparisons stand after "
                               f"every policy's part")  # fmt: skip
    cells: dict[str, dict[tuple[str, str], dict[str, dict[float, SameTarget]]]] = {}
    for group, pairs in groups.items():
        cells[group] = {}
        for first, second in pairs:
            by_mix: dict[str, dict[float, SameTarget]] = {}
            for mix in MIXES:
                by_mix[mix] = {}
                for share in TARGETS:
                    try:
                        (a, a_fell), (b, b_fell) = (
                            held[section_of(first), mix, share],
                            held[section_of(second), mix, share],
                        )
                    except KeyError as error:
                        raise ComparisonsError(
                            f"{pair_name((first, second))} has no point at {error.args[0][1:]}"
                        ) from None
                    by_mix[mix][share] = compare_outcomes(draws, mixes[mix], a, a_fell, b, b_fell)
            cells[group][first, second] = by_mix
    return cells


def declared(groups: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The seven declared questions in the write-up's order, each with its pairs' names in item 22's order. `groups` is the
    comparisons' `groups` (by group, pair name)."""
    lead = list(LEAD)
    taken: list[list[str]] = []
    for position, (_, group) in enumerate(QUESTION_HEADINGS):
        names = list(groups[group]["pairs"])
        if position < 2:
            taken.append([lead[position]])
        elif group == "cascades_against_routers":
            taken.append([name for name in names if name not in lead])
        else:
            taken.append(names)
    return [
        {"question": heading, "group": group, "pairs": names}
        for (heading, group), names in zip(QUESTION_HEADINGS, taken)
    ]


def comparisons_data(cells: Mapping[str, Mapping[tuple[str, str], Any]]) -> dict[str, Any]:
    """The comparisons as table v4 holds them (`description`, `count`, `headline`, `groups`: the question and the pairs'
    cells by mix and target), so that on dev they equal table v4's own."""
    groups = {
        group: {
            "question": QUESTIONS[group],
            "pairs": same_target_data({(cc.label(a), cc.label(b)): by_mix for (a, b), by_mix in by_pair.items()}),
        }
        for group, by_pair in cells.items()
    }
    return {
        "description": DESCRIPTIVE,
        "count": sum(len(pairs) for pairs in cells.values()),
        "headline": [f"{cc.label(a)} less {cc.label(b)}" for a, b in HEADLINE],
        "groups": groups,
    }


def part(inputs: Any) -> dict[str, Any]:
    """The offline half's comparisons part: every policy the parts before it evaluated, compared in item 22's pairs."""
    from analysis.operating_eval import shared_fixed

    fixed = shared_fixed(inputs.cache, inputs.matrix)
    return comparisons_data(cells_of(inputs.draws, load_mixes(), fixed.evaluated))


def declared_part(inputs: Any) -> dict[str, Any]:
    """The offline half's declared-order part, from the comparisons part built before it: the seven declared questions of the
    write-up in its order (`declared`), and the 51 pairs flat in that order (`order`). Dev never held it."""
    built = inputs.cache.get("built", {})
    if "comparisons" not in built:
        raise ComparisonsError("the declared order stands after the comparisons part")
    questions = declared(built["comparisons"]["groups"])
    return {"declared": questions, "order": [name for question in questions for name in question["pairs"]]}


COMMITTED_KEYS = ("description", "count", "headline", "groups")


def committed() -> dict[str, Any]:
    """Table v4's comparisons as committed: what the part gives on dev."""
    held = json.loads(TABLE_V4.read_text(encoding="utf-8"))["comparisons"]
    return {key: held[key] for key in COMMITTED_KEYS}


# ---- the replication summary -----------------------------------------------------------------------------------


def excludes_zero(estimate: Mapping[str, float] | None) -> int:
    """+1 where a 95% interval sits wholly above 0, -1 wholly below, 0 where it holds 0 (and for a cell that holds none)."""
    if estimate is None:
        return 0
    return 1 if estimate["low"] > 0 else -1 if estimate["high"] < 0 else 0


def sign(value: float) -> int:
    return (value > 0) - (value < 0)


def replication(dev: Mapping[str, Any], test: Mapping[str, Any]) -> dict[str, Any]:
    """For each dev comparison whose interval excluded 0 (A19 item 8): whether test's has the same sign and also excludes 0.

    `dev` and `test` are the comparisons' `groups`. A comparison counts once for each mix-and-target cell where its dev
    interval excluded 0, and cost and quality are counted apart. **Same sign** is test's point value on the side of 0 dev's
    interval was on; **same sign, excluding 0** is test's own interval wholly on that side. A cell that test lacks, or holds
    no interval in, is neither. Only the dev intervals that excluded 0 are listed. Refuses (ComparisonsError) a pair or cell
    dev has and test doesn't."""
    counts = {measure: {"dev_excluded": 0, "same_sign": 0, "same_sign_excluding_zero": 0} for measure in MEASURES}
    listed: list[dict[str, Any]] = []
    for group, body in dev.items():
        for pair, by_mix in body["pairs"].items():
            for mix, by_target in by_mix.items():
                for target, cell in by_target.items():
                    try:
                        found = test[group]["pairs"][pair][mix][target]
                    except KeyError:
                        raise ComparisonsError(
                            f"test holds no cell for {pair} {mix} {target}, which dev holds"
                        ) from None
                    for measure in MEASURES:
                        side = excludes_zero(cell[measure])
                        if not side:
                            continue
                        there = found[measure]
                        same = there is not None and sign(there["value"]) == side
                        both = same and excludes_zero(there) == side
                        counts[measure]["dev_excluded"] += 1
                        counts[measure]["same_sign"] += same
                        counts[measure]["same_sign_excluding_zero"] += both
                        listed.append({"group": group, "pair": pair, "mix": mix, "target": target, "measure": measure,
                                       "dev": cell[measure], "test": there, "same_sign": same,
                                       "same_sign_excluding_zero": both})  # fmt: skip
    return {
        "definition": "for each dev comparison whose interval excluded 0, in each mix-and-target cell, whether test's has the "
        "same sign and also excludes 0; cost and quality are counted apart (A19 item 8)",
        "counts": counts,
        "listed": listed,
    }


def replication_part(inputs: Any) -> dict[str, Any]:
    """The offline half's replication part: dev's table v4 as committed against the comparisons part built before it. On dev
    itself every listed comparison replicates, trivially."""
    dev = json.loads(TABLE_V4.read_text(encoding="utf-8"))["comparisons"]["groups"]
    built = inputs.cache.get("built", {})
    if "comparisons" not in built:
        raise ComparisonsError("the replication summary stands after the comparisons part")
    return replication(dev, built["comparisons"]["groups"])
