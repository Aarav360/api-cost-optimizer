"""Target met on test, for every policy's operating points (PLAN_v4.md Amendment A19, item 6; GLOSSARY.md "target met"; issue #88).

On dev a walk calls an operating point met when the policy's quality clears the target by one paired standard error. On test
nothing is walked, and **target met** is the quantity the walk used with the margin taken to zero: the policy's quality less
the target's share of A's quality, on the same items, and in every resample (so the interval is paired). A point is **met**
when that difference is at least zero (less the rounding the walk allows, `analysis/references.ROUNDING`), and **missed**
otherwise. It is printed with the interval over the 2,000 resamples and with the share of resamples that meet the target, as
dev's tables print the share (`captured.met`, the same comparison, so the two agree).

A point that fell back to A is labelled **falls back to A**: it is A, it meets every target trivially, and counting it would pad
the count. It has no numbers and isn't counted.

**Every policy's rows, with no line per policy.** A policy's part builds its rows with `analysis/operating_eval.mix_rows` (or
`task_rows`), and those take the policy's outcomes at each frozen point. They note them on the offline half's shared `Fixed`
(`operating_eval.shared_fixed`), keyed by the point (which names its section), so `part` here reads the outcomes of every policy
that was built before it, whatever its dials or traces. The part stands after the policies in `analysis/offline_half.PARTS`, and
refuses a built part that holds policy rows nothing evaluated through `operating_eval`, so a policy that skips it is found, not
left out.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from analysis.bootstrap import Bootstrap, BootstrapError, percentile_estimate
from analysis.operating_eval import FrozenPoint, shared_fixed
from analysis.references import ROUNDING
from analysis.simulator import Outcome, load_mixes, totals
from routers.ladder import FRONTIER

MET, MISSED, FALLS_BACK = "met", "missed", "falls back to A"
MARGIN = 0  # a walk's margin on dev is one paired standard error; on test it is nothing


class TargetMetError(ValueError):
    """Outcomes that can't be compared with A's, or policy rows that were not evaluated at their frozen points."""


def point_row(
    draws: Bootstrap,
    frontier: list[Outcome],
    policy: list[Outcome],
    mix: Mapping[str, float],
    share: float,
    fell_back: bool,
) -> dict[str, Any]:
    """One operating point's target met under `mix`: its reading, the quality less `share` of A's quality (on the items,
    with the percentile interval and standard error over the resamples), and the share of resamples that meet the target.
    A point that fell back is `FALLS_BACK` and holds no numbers."""
    if fell_back:
        return {"reading": FALLS_BACK, "difference": None, "met_share": None}
    mix = dict(mix)
    try:
        quality, _ = draws.resampled(policy, mix)
        frontier_quality, _ = draws.resampled(frontier, mix)
    except BootstrapError as error:
        raise TargetMetError(str(error)) from None
    target = share * frontier_quality  # each resample's target is its own share of A's, so the two stay paired
    value = totals(policy, mix).quality - share * totals(frontier, mix).quality
    return {
        "reading": MET if value >= MARGIN - ROUNDING else MISSED,
        "difference": _as_dict(percentile_estimate(value, quality - target)),
        "met_share": float(np.mean(quality >= target)),
    }


def _as_dict(estimate: Any) -> dict[str, float]:
    return {"value": estimate.value, "low": estimate.low, "high": estimate.high,
            "standard_error": estimate.standard_error}  # fmt: skip


def rows_of(
    draws: Bootstrap, fixed: Mapping[str, list[Outcome]], evaluated: Mapping[FrozenPoint, list[Outcome]],
    mixes: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:  # fmt: skip
    """Target met at every evaluated point: section, then mix, then target."""
    rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for point, found in evaluated.items():
        rows.setdefault(point.section, {}).setdefault(point.mix, {})[point.target] = point_row(
            draws, fixed[FRONTIER], found, mixes[point.mix], point.share, point.fell_back
        )
    return rows


def count_of(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How many points a policy has, routes, meets, misses and falls back at. A point that falls back isn't counted as met."""
    readings = [row["reading"] for row in rows]
    return {
        "points": len(rows),
        "routes": sum(r != FALLS_BACK for r in readings),
        MET: readings.count(MET),
        MISSED: readings.count(MISSED),
        "falls_back": readings.count(FALLS_BACK),
    }


def counts_of(rows: Mapping[str, Mapping[str, Mapping[str, dict[str, Any]]]]) -> dict[str, dict[str, int]]:
    """Each section's counts, and all of them together under "all"."""
    by_section = {s: [r for by_mix in mixes.values() for r in by_mix.values()] for s, mixes in rows.items()}
    return {**{s: count_of(found) for s, found in by_section.items()},
            "all": count_of([r for found in by_section.values() for r in found])}  # fmt: skip


def holds_policy_rows(value: object) -> bool:
    """Whether a built part's value holds rows of a policy at its operating points (a `fell_back` flag in a mix's row)."""
    mixes = value.get("mixes") if isinstance(value, dict) else None
    return isinstance(mixes, dict) and any(
        isinstance(row, dict) and "fell_back" in row
        for by_target in mixes.values() if isinstance(by_target, dict) for row in by_target.values()
    )  # fmt: skip


def part(inputs: Any) -> dict[str, Any]:
    """The offline half's target-met part: every point the parts built before it evaluated, by section. Refuses a built part
    that holds policy rows for a section nothing evaluated, which this part would leave out."""
    fixed = shared_fixed(inputs.cache, inputs.matrix)
    sections = {point.section for point in fixed.evaluated}
    built = inputs.cache.get("built", {})
    unseen = [key for key, value in built.items() if holds_policy_rows(value) and key not in sections]
    if unseen:
        raise TargetMetError(f"{unseen} hold policy rows that were not evaluated with operating_eval, so target met would "
                             f"leave them out")  # fmt: skip
    if not sections:
        raise TargetMetError("no policy was evaluated before target met: it stands after the policies' parts")
    rows = rows_of(inputs.draws, fixed, fixed.evaluated, load_mixes())
    return {"margin": MARGIN, "points": rows, "counts": counts_of(rows)}
