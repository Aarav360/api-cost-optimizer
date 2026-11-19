"""What the live replay runs, and what it is set beside (PLAN_v4.md Amendment A17, items 24 and 26.5; A19, items 2.7 and 8;
issue #77, #91).

Pure planning, with no call and no store: the policies, the offline estimate for the same items, and the latency figures.
`analysis/final_replay.py` runs the policies and reports through these.

**The policies** (`make_plan`): A, then the best cascade at its realistic-mix 99% and 97% operating points, skipping a point
that falls back to A (it is A, which the replay already runs). The best cascade is item 24's rule applied to the committed
dev savings (`best_cascade_section`): the headline's own choice (`final_headline.best_cascade`), so no second rule is built here.
A cascade that starts at a router is supported for E and F, whose scores are local. G's and G+'s are Jev calls whose live
path belongs to #84, so a plan that needs one is refused (`ReplayPlanError`) before anything is called.

**The offline estimate** (`offline_estimate`) is for the same items: the policy's trace on each of them at the frozen point,
from test's results matrix and gate-score table, with equal weights (the 100 items are 25 for each task, which is the
uniform mix). The latency estimate is A17 item 16's: the sum along the trace of the pilot's median for the config, the
router's median and each gate call's median, the medians dev's table v4 committed.

**The latency figures** (`latency_stats`) are the median and p90 of the requests' times, with the first call, a reconnect and
a request a crash cut apart: each is a flag on the request's record, and a request with one is left out of the median and the
p90 and listed beside them (A17 items 16 and 26.5).
"""

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analysis import final_headline
from analysis.gate_table import GateScoreTable, pure_traces
from analysis.latency_estimate import GateMedians, cascade_gate_calls, trace_seconds
from analysis.matrix import Matrix
from analysis.operating_eval import frozen_points
from analysis.operating_points import OPERATING_POINTS_FILE
from analysis.routed_walk import routed_traces
from analysis.router_names import CASCADE_STARTS, CASCADES, HAIKU, LETTER_OF, NAMED_ROUTERS, parse_cascade
from analysis.simulator import Outcome, Trace, always, simulate
from analysis.table_v4 import TABLE_V4
from routers.gates import assert_line
from routers.ladder import FRONTIER

POINTS = (("realistic", "99%"), ("realistic", "97%"))  # A17 item 26.5: the realistic-mix 99% and 97% operating points
A_KEY = "A"
# The `Extra` holding each router's table (E and F: `analysis/final_routers.py`; G and G+: `analysis/final_jev.py`):
# rules_table ... jev_bolstered_table. The twelve router-started cascades read them (`analysis/final_routed.py`).
ROUTER_TABLES = {section: f"{section}_table" for section in NAMED_ROUTERS.values()}
# A router's score tables as the offline half's extras name them (`analysis/offline_half.py`), for the routers whose live
# path this replay has (E and F).
ROUTER_EXTRAS = {section: ROUTER_TABLES[section] for section in (NAMED_ROUTERS["E"], NAMED_ROUTERS["F"])}


class ReplayPlanError(ValueError):
    """A plan the replay can't run: the rule leaves a tie open, or the cascade starts at a router with no live path here."""


@dataclass(frozen=True)
class Policy:
    key: str  # a stored record's name for it: "A", or "<section>/<mix>/<target>"
    label: str  # the words the report uses: "A", "haiku/Jev realistic 99%"
    section: str | None  # the cascade's section in the operating-points file; None for A
    start: str | None  # the cascade's start part: "haiku" or a router's section name; None for A
    gate: str | None  # "free", "jev" or "sonnet"; None for A
    threshold: float | None  # the router's dial; None for A and a pure cascade
    cutoff: float | None  # the gate's dial; None for A and the free checks alone


@dataclass(frozen=True)
class Plan:
    section: str
    policies: tuple[Policy, ...]  # A first, then the points that route
    skipped: tuple[str, ...]  # the points that fall back to A, by label


def best_cascade_section(table: Path = TABLE_V4) -> str:
    """The best cascade, as the headline chooses it: `final_headline.best_cascade` (item 24's rule on dev's table `table`), so
    the replay and the headline can't choose differently. That refuses (here as ReplayPlanError) a tie the rule leaves open;
    this refuses besides a table whose own pick differs from the rule's."""
    held = json.loads(table.read_text(encoding="utf-8"))
    try:
        found = final_headline.best_cascade(held)
    except final_headline.HeadlineError as error:
        raise ReplayPlanError(str(error)) from None
    if held["best_cascade"]["best"] != found:
        raise ReplayPlanError(
            f"table v4 names {held['best_cascade']['best']} the best cascade, but the rule gives {found}"
        )
    return found


def make_plan(section: str, path: Path = OPERATING_POINTS_FILE) -> Plan:
    """A, then `section`'s realistic-mix 99% and 97% points, a point that falls back to A skipped (it is A)."""
    start, gate = parse_cascade(section)
    if start != HAIKU and start not in ROUTER_EXTRAS:
        raise ReplayPlanError(
            f"{CASCADES[section]} starts at {CASCADE_STARTS[start]}, whose live calls the replay doesn't have yet "
            f"(only the pure cascades and E and F are built); nothing was called"
        )
    policies = [Policy(A_KEY, "A", None, None, None, None, None)]
    skipped = []
    held = {(p.mix, p.target): p for p in frozen_points(section, path)}
    for mix, target in POINTS:
        point = held[mix, target]
        name = f"{CASCADES[section]} {mix} {target}"
        if point.fell_back:
            skipped.append(name)
            continue
        policies.append(Policy(f"{section}/{mix}/{target}", name, section, start, gate, point.threshold, point.cutoff))
    return Plan(section, tuple(policies), tuple(skipped))


# ---- the offline estimate for the same items ---------------------------------------------------------------------


@dataclass(frozen=True)
class Medians:
    """Dev's committed median seconds (table v4's `latency.inputs`): the gates' calls and the routers'."""

    gates: GateMedians
    routers: Mapping[str, Any]  # by letter, a number or one per task


def read_medians(table: Path = TABLE_V4) -> Medians:
    held = json.loads(table.read_text(encoding="utf-8"))["latency"]["inputs"]
    return Medians(GateMedians(held["code_check_s"], held["jev_gate_s"], held["sonnet_gate_s"]), held["routers_s"])


@dataclass(frozen=True)
class Estimate:
    """A policy's offline numbers over the replay's items: what the live ones are set beside."""

    n: int
    cost_per_request: float  # dollars at list price, from the stored answers and the gate-score table
    escalation_rate: float  # the share of items whose first answer was rejected and sent on
    quality: float
    quality_by_task: Mapping[str, tuple[float, int]]  # task -> (share correct, items)
    latency_median_s: float
    latency_p90_s: float


def traces_of(policy: Policy, matrix: Matrix, table: GateScoreTable, routers: Mapping[str, Any]) -> list[Trace]:
    """The policy's trace on every item of the matrix at its frozen dials: A always `opus`, a pure cascade from `haiku`, a
    router-started one from the router's pick at the threshold."""
    if policy.section is None:
        return always(matrix, FRONTIER)
    assert policy.gate is not None
    if policy.start == HAIKU:
        return pure_traces(table, policy.gate, policy.cutoff)
    assert policy.start is not None and policy.threshold is not None
    return routed_traces(routers[policy.start], table, policy.gate, policy.cutoff, policy.threshold)


def offline_estimate(
    policy: Policy,
    matrix: Matrix,
    table: GateScoreTable,
    routers: Mapping[str, Any],
    items: Sequence[str],
    code_items: Sequence[str],
    medians: Medians,
) -> Estimate:
    """The policy's estimate over `items` (every item of the matrix is traced, then these are taken): mean cost and quality,
    the escalation rate, and the median and p90 of each item's estimated seconds."""
    wanted = set(items)
    traces = {t.item_id: t for t in traces_of(policy, matrix, table, routers)}
    outcomes = [o for o in simulate(matrix, list(traces.values())) if o.item_id in wanted]
    if len(outcomes) != len(wanted):
        raise ReplayPlanError(f"the matrix holds {len(outcomes)} of the {len(wanted)} replay items")
    seconds = [item_seconds(policy, traces[o.item_id], o, table, code_items, medians) for o in outcomes]
    return Estimate(
        len(outcomes),
        statistics.fmean(o.cost for o in outcomes),
        sum(len(o.configs) > 1 for o in outcomes) / len(outcomes),
        statistics.fmean(o.success for o in outcomes),
        by_task(outcomes),
        float(np.median(seconds)),
        float(np.percentile(seconds, 90)),
    )


def by_task(outcomes: Sequence[Outcome]) -> dict[str, tuple[float, int]]:
    tasks = sorted({o.task for o in outcomes})
    return {t: (statistics.fmean(o.success for o in outcomes if o.task == t), sum(o.task == t for o in outcomes))
            for t in tasks}  # fmt: skip


def item_seconds(
    policy: Policy, trace: Trace, outcome: Outcome, table: GateScoreTable, code_items: Sequence[str], medians: Medians
) -> float:
    """One item's estimated seconds: the pilot's median for each config called, the router's median, each gate call's."""
    if policy.section is None:
        return trace_seconds(outcome.task, trace.configs)
    assert policy.gate is not None
    router: Any = 0.0
    if policy.start != HAIKU:
        assert policy.start is not None
        router = medians.routers[LETTER_OF[policy.start]]
    calls = cascade_gate_calls(table, trace.item_id, trace.configs[0], policy.gate, policy.cutoff, medians.gates,
                               set(code_items))  # fmt: skip
    return trace_seconds(outcome.task, trace.configs, router, calls)


def code_items_of(requests: Mapping[str, Any], items: Sequence[str]) -> list[str]:
    """The items whose request has an `assert` line, which is where the code check applies."""
    return [item for item in items if assert_line(requests[item].prompt)]


# ---- the live latency figures ------------------------------------------------------------------------------------


def latency_stats(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A policy's live requests' median and p90, in seconds, over the requests that are none of: the go's first call (the one
    that opened the connections), a request in which a Jev call reconnected, or one a crash cut. Those are listed apart, in
    the order they came, since each carries a cost the others don't. numpy's percentile, with linear interpolation, as the
    gates' latency is."""
    kept = [float(r["latency_s"]) for r in requests if not (r["first_call"] or r["reconnect"] or r["interrupted"])]
    apart = {
        key: [float(r["latency_s"]) for r in requests if r[flag]]
        for key, flag in (
            ("first_call_s", "first_call"),
            ("reconnects_s", "reconnect"),
            ("interrupted_s", "interrupted"),
        )
    }
    return {
        "requests": len(requests),
        "kept": len(kept),
        "median_s": float(np.median(kept)) if kept else None,
        "p90_s": float(np.percentile(kept, 90)) if kept else None,
        **apart,
    }
