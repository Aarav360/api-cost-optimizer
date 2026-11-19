"""The headline on test: the best cascade by A17 item 24's rule on dev's table, against A, on the realistic mix at the 99%
target (PLAN_v4.md Amendment A19, item 6; issue #88).

The rule is declared before test is seen and builds no new choice: **`best_cascade(table)`** takes dev's table v4 (the committed
`figures/table_v4.json`, or the re-selected dev table if a refusal path re-selects the cascades, which the caller passes) and
gives the key `analysis/table_v4.best_cascade_data` gives, which is `analysis/cascade_comparisons.best_cascade` over the six
operating points' savings. Nothing on test goes in: that cascade's cell on test is read out whatever the other policies did, and
a policy that does better on test is a finding for the write-up, not a headline. The live replay (#91) runs the same cascade, so
it calls `best_cascade` too and never picks its own.

The cell is the cascade's realistic 99% point with its quality and cost against A's, and its target met at a margin of zero
(`analysis/target_met.py`). It reads **met**, **not met** (with the quality and cost the cell holds) or **falls back to A**. The
uniform mix at 99% and the 97% target at both mixes are shown beside it (`beside`), never in its place. Dev's own savings and
share of resamples meeting the target at that point stand beside it, from dev's table.
"""

import json
from collections.abc import Mapping
from typing import Any

from analysis import target_met
from analysis.final_design import HEADLINE_MIX, HEADLINE_TARGET, TABLE_V4
from analysis.router_names import CASCADES

BESIDE = (("realistic", "97%"), ("uniform", "99%"), ("uniform", "97%"))  # beside the headline, never in its place
READING = {target_met.MET: "met", target_met.MISSED: "not met", target_met.FALLS_BACK: target_met.FALLS_BACK}


class HeadlineError(ValueError):
    """The rule leaves no cascade, or what the headline reads wasn't built."""


def dev_table() -> dict[str, Any]:
    """Dev's table v4 as committed."""
    table: dict[str, Any] = json.loads(TABLE_V4.read_text(encoding="utf-8"))
    return table


def best_cascade(table: Mapping[str, Any] | None = None) -> str:
    """The best cascade by A17 item 24's rule on dev's table `table` (default: the committed table v4): its section key.
    Refuses where the rule leaves a tie open, since no choice is made by hand."""
    from analysis.table_v4 import best_cascade_data

    held = dev_table() if table is None else table
    chosen = best_cascade_data({key: cascade["operating_points"] for key, cascade in held["cascades"].items()})
    best = chosen["best"]
    if best is None:
        raise HeadlineError(
            f"item 24's rule leaves a tie open between {chosen['tied_at_top']}: no cascade is chosen by hand"
        )
    return str(best)


def cell(built: Mapping[str, Any], key: str, mix: str, target: str) -> dict[str, Any]:
    """One point of the cascade `key` on test: its quality and cost beside A's, and its target met."""
    try:
        row = built[key]["mixes"][mix][target]
        met = built["target_met"]["points"][key][mix][target]
        a = built["references"]["mixes"][mix]["configs"]["opus"]
    except KeyError as error:
        raise HeadlineError(f"the headline reads {key} {mix} {target} from the parts built before it, and {error} is not "
                            f"there") from None  # fmt: skip
    return {
        "mix": mix,
        "target": target,
        "reading": READING[met["reading"]],
        "quality": row["quality"],
        "cost_per_1k": row["cost_per_1k"],
        "A": {"quality": a["quality"]["value"], "cost_per_1k": a["cost_per_1k"]["value"]},
        "target_quality": row["target"],
        "savings_vs_A": row["savings_vs_A"],
        "difference": met["difference"],
        "met_share": met["met_share"],
    }


def build(built: Mapping[str, Any], table: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The headline from the parts built so far (the policy's part, the references and target met), with dev's table choosing."""
    held = dev_table() if table is None else table
    key = best_cascade(held)
    if key not in built:
        raise HeadlineError(f"the headline's cascade {key} was not built before the headline")
    dev = held["cascades"][key]["operating_points"][HEADLINE_MIX][HEADLINE_TARGET]
    return {
        "rule": "A17 item 24 on dev's table: the best cascade, against A, on the realistic mix at the 99% target",
        "cascade": key,
        "label": CASCADES[key],
        "headline": cell(built, key, HEADLINE_MIX, HEADLINE_TARGET),
        "beside": {f"{mix} {target}": cell(built, key, mix, target) for mix, target in BESIDE},
        "dev": {"savings_share": dev["savings_vs_A"]["share"], "met_share": dev["captured"]["met"]},
    }


def part(inputs: Any) -> dict[str, Any]:
    """The offline half's headline part: after the policies and target met, and so after what it reads."""
    return build(inputs.cache.get("built", {}))
