"""Check Stage 7's comparisons and table v4 (PLAN_v4.md §9, §10, §12; Amendment A17, items 21 to 24; issue #71).

Run from the project root:  uv run python -m analysis.check_table_v4

Synthetic part (CI too): no results/ folder, no Docker, no Jev or Anthropic call, no benchmark answer, no test item.
- the 51 same-target comparisons: item 22's five groups, their sizes, their order and who is first;
- the best cascade (item 24): the highest mean savings against A over six operating points, a fall back counting 0,
  ties to a pure cascade and then to the cheaper gate, and an open case recorded instead of settled by judgement.
"""

import json
import sys
from fractions import Fraction

from analysis import cascade_comparisons as cc
from analysis import cascade_metrics as cm
from analysis import table_v4
from analysis.bootstrap import resample
from analysis.check_gates_dev import CUTOFFS, TASKS, synthetic_dev
from analysis.check_simulator import CASCADED, HAND, HAND_MIX, synthetic
from analysis.gate_table import GateRow, GateScoreTable, PaidScore
from analysis.latency_estimate import PILOT_MEDIANS, GateMedians
from analysis.router_names import CASCADES, FREE_GATE, G_PLUS, HAIKU, JEV_GATE, NAMED_ROUTERS, E, F, G, cascade_section
from analysis.simulator import Outcome, always, simulate
from analysis.walk import Step, Walk

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def pair_checks() -> None:
    print("-- the 51 comparisons --")
    groups = cc.comparison_pairs()
    check({g: len(p) for g, p in groups.items()} == {"gate_adds": 12, "router_adds": 12, "paid_gate_over_free": 10,
                                                     "jev_against_sonnet": 5, "cascades_against_routers": 12},
          "item 22's five groups hold 12, 12, 10, 5 and 12 pairs")  # fmt: skip
    pairs = [pair for group in groups.values() for pair in group]
    check(len(pairs) == 51 and len(set(pairs)) == 51, "51 distinct pairs")
    first = groups["gate_adds"]
    check(first[0] == (cascade_section(NAMED_ROUTERS[E], "free"), E)
          and first[-1] == (cascade_section(NAMED_ROUTERS[G_PLUS], "sonnet"), G_PLUS),
          "each router-started cascade against its router alone, first the cascade, E/free first, G+/Sonnet last")  # fmt: skip
    check(all(a in CASCADES and b in (E, F, G, G_PLUS) for a, b in first), "the second of a gate_adds pair is a router")
    check(groups["router_adds"][0] == (cascade_section(NAMED_ROUTERS[E], "free"), cascade_section("haiku", "free"))
          and groups["router_adds"][4] == (cascade_section(NAMED_ROUTERS[F], "jev"), cascade_section("haiku", "jev")),
          "each router-started cascade against the pure cascade with the same gate")  # fmt: skip
    check(groups["paid_gate_over_free"][:2] == [(cascade_section("haiku", "jev"), cascade_section("haiku", "free")),
                                                (cascade_section("haiku", "sonnet"), cascade_section("haiku", "free"))]
          and groups["paid_gate_over_free"][2] == (cascade_section(NAMED_ROUTERS[E], "jev"), cascade_section(NAMED_ROUTERS[E], "free")),
          "haiku/Jev and haiku/Sonnet against haiku/free, then each router's gates against its /free")  # fmt: skip
    check(all(a.endswith("_jev") and b.endswith("_sonnet") and a[:-3] == b[:-6] for a, b in groups["jev_against_sonnet"]),
          "the Jev gate against the Sonnet gate on the same start, five starts")  # fmt: skip
    check(groups["cascades_against_routers"][0] == (cascade_section("haiku", "free"), E)
          and groups["cascades_against_routers"][-1] == (cascade_section("haiku", "sonnet"), G_PLUS)
          and {a for a, _ in groups["cascades_against_routers"]} == {cascade_section("haiku", g) for g in ("free", "jev", "sonnet")},
          "each pure cascade against each router alone")  # fmt: skip
    check(cc.label(E) == "E" and cc.label(cascade_section(NAMED_ROUTERS[G_PLUS], "sonnet")) == f"{G_PLUS}/Sonnet",
          "a router keeps its letter and a cascade its label")  # fmt: skip


def best_checks() -> None:
    print("\n-- the best cascade --")
    keys = list(CASCADES)
    flat = dict.fromkeys(keys, [0.0] * 6)
    pure_jev, pure_free = cascade_section("haiku", "jev"), cascade_section("haiku", "free")
    f_jev, e_jev = cascade_section(NAMED_ROUTERS[F], "jev"), cascade_section(NAMED_ROUTERS[E], "jev")
    # haiku/Jev: (.5 + 0 + 0 + .5 + 0 + 0) / 6 = 1/6; F/Jev: six points of .2 -> .2: the higher mean wins, even
    # though haiku/Jev has the single best point.
    savings = {**flat, pure_jev: [0.5, 0, 0, 0.5, 0, 0], f_jev: [0.2] * 6}
    found = cc.best_cascade(savings)
    check(found.best == f_jev and not found.open_case and found.tied == (f_jev,),
          "the highest mean over the six points wins, and a fall back (0) counts as 0")  # fmt: skip
    check(abs(found.means[pure_jev] - 1 / 6) < 1e-12 and found.means[cascade_section("haiku", "free")] == 0,
          "the means are over six points")  # fmt: skip
    savings = {**flat, f_jev: [0.2] * 6, pure_jev: [0.1, 0.3] * 3}
    found = cc.best_cascade(savings)
    check(found.best == pure_jev and found.tied == (pure_jev, f_jev) and not found.open_case,
          "a tie goes to a pure cascade")  # fmt: skip
    savings = {**flat, pure_jev: [0.2] * 6, pure_free: [0.2] * 6, cascade_section("haiku", "sonnet"): [0.2] * 6}
    check(
        cc.best_cascade(savings).best == pure_free, "then to the cheaper gate: the free checks, then Jev, then Sonnet"
    )
    savings = {**flat, f_jev: [0.2] * 6, cascade_section(NAMED_ROUTERS[F], "sonnet"): [0.2] * 6,
               cascade_section(NAMED_ROUTERS[E], "sonnet"): [0.2] * 6}  # fmt: skip
    check(cc.best_cascade(savings).best == f_jev, "two router-started cascades tied: the cheaper gate")
    savings = {**flat, f_jev: [0.2] * 6, e_jev: [0.2] * 6}
    found = cc.best_cascade(savings)
    check(found.best is None and found.open_case and set(found.tied) == {f_jev, e_jev},
          "same pure-ness and same gate left tied: the rule leaves the case open, so no cascade is named")  # fmt: skip
    savings = {**flat, f_jev: [0.2 + 1e-12] * 6, e_jev: [0.2] * 6}
    check(cc.best_cascade(savings).open_case, "means that differ by float rounding only are a tie")
    check(cc.best_cascade(flat).best == pure_free and not cc.best_cascade(flat).open_case,
          "all fifteen at 0 is a tie the rule settles as it settles any: a pure cascade, the cheapest gate")  # fmt: skip
    for bad in ({**flat, f_jev: [0.2] * 5}, {k: flat[k] for k in keys[:14]}, {**flat, "cascade_x_y": [0.0] * 6}):
        try:
            cc.best_cascade(bad)
            refused = False
        except cc.ComparisonError:
            refused = True
        check(refused, "savings for other than fifteen cascades x six points are refused")


def walk_of(outcomes, fell_back: bool = False, cutoff: float | None = 0.5, threshold: float | None = None) -> Walk:
    """A two-step walk on HAND at the 100% target: A, then `outcomes` as the step tried (certified unless `fell_back`
    makes it miss)."""
    frontier = simulate(HAND, always(HAND, "opus"))
    a = Step(None, 0.75, 6.25 / 1000 * 1000, 0.0, True, tuple(frontier))
    step = Step(threshold, 0.75, 0.0, 0.0, not fell_back, tuple(outcomes), cutoff)
    return Walk(1.0, 0.75, (a, step))


def row_checks() -> None:
    print("\n-- escalation rate and a row --")
    outcomes = [
        Outcome("a", "t1", "a", 0.001, 1, ("haiku",)),
        Outcome("b", "t1", "b", 0.008, 1, ("haiku", "opus")),
        Outcome("c", "t2", "c", 0.010, 0, ("opus",)),
    ]
    # Weights under t1 0.75, t2 0.25: a and b weigh .375 each, c .25. Only b was rejected and sent on to opus; c went
    # straight to opus, which is no escalation.
    check(
        abs(cc.escalation_rate(outcomes, HAND_MIX) - 0.375) < 1e-12,
        "the escalation rate is the weight of rejected answers",
    )
    check(cc.escalation_rate(simulate(HAND, always(HAND, "opus")), HAND_MIX) == 0.0, "A escalates nothing")
    draws = resample(HAND)
    fixed = {c: simulate(HAND, always(HAND, c)) for c in ("haiku", "opus")}
    savings = draws.savings(HAND, fixed["opus"], HAND_MIX)
    cascaded = simulate(HAND, CASCADED)
    row = cc.policy_row(draws, savings, walk_of(cascaded), fixed, HAND_MIX)
    # CASCADED costs $4.225 per 1k at 75% (check_simulator's hand work); A costs $6.25, so it saves $2.025, 32.4%, and
    # captures 81/115 of the oracle's savings at 100%. Item b alone climbed to opus: .375.
    check(
        row["fell_back"] is False and row["cutoff"] == 0.5 and row["threshold"] is None,
        "the row names the cutoff, no threshold",
    )
    check(abs(row["cost_per_1k"]["value"] - 4.225) < 1e-9 and abs(row["quality"]["value"] - 0.75) < 1e-12,
          "the row's cost and quality are the operating point's")  # fmt: skip
    check(abs(row["savings_vs_A"]["per_1k"] - 2.025) < 1e-9 and abs(row["savings_vs_A"]["share"] - 2.025 / 6.25) < 1e-9,
          "savings against A, in dollars and as a share")  # fmt: skip
    check(abs(row["captured"]["value"] - 81 / 115) < 1e-9, "the captured fraction")
    check(abs(row["escalation_rate"] - 0.375) < 1e-12 and abs(row["routing"]["opus"] - 0.375) < 1e-12,
          "the escalation rate and the routing mix")  # fmt: skip
    check({"steps_certified", "steps_tried", "target", "margin", "difference_from_A", "matched_cost", "captured"} <= set(row),
          "table v1's keys")  # fmt: skip
    fallen = cc.policy_row(draws, savings, walk_of(cascaded, fell_back=True), fixed, HAND_MIX)
    check(fallen["fell_back"] is True and fallen["cutoff"] is None and fallen["escalation_rate"] == 0.0
          and fallen["savings_vs_A"]["share"] == 0.0 and fallen["captured"]["value"] == 0.0 and fallen["steps_certified"] == 0,
          "a fall back is A: no cutoff, no escalation, saves 0 and captures 0")  # fmt: skip
    routed = cc.policy_row(draws, savings, walk_of(cascaded, cutoff=0.7, threshold=0.4), fixed, HAND_MIX)
    check(
        routed["threshold"] == 0.4 and routed["cutoff"] == 0.7,
        "a router-started row names its threshold and the fixed cutoff",
    )


def compare_checks() -> None:
    print("\n-- the comparisons --")
    draws = resample(HAND)
    mixes = {"m": HAND_MIX}
    cascaded = walk_of(simulate(HAND, CASCADED))
    fell = walk_of(simulate(HAND, CASCADED), fell_back=True)
    walks = {name: {"m": {1.0: cascaded}} for name in (*cc.ROUTERS, *CASCADES)}
    walks[E] = {"m": {1.0: fell}}
    walks[cascade_section("haiku", "free")] = {"m": {1.0: fell}}
    walks[cascade_section(NAMED_ROUTERS[E], "free")] = {"m": {1.0: fell}}
    found = cc.compare_pairs(draws, mixes, walks)
    check({g: len(p) for g, p in found.items()} == {g: len(p) for g, p in cc.comparison_pairs().items()},
          "every group is compared pair by pair")  # fmt: skip
    cell = found["gate_adds"]["E/free less E"]["m"][1.0]
    check(cell.both_fall_back, "the cascade and the router both falling back read 'both fall back to A'")
    cell = found["cascades_against_routers"]["haiku/Jev less E"]["m"][1.0]
    check(cell.cost is not None and abs(cell.cost.value - (4.225 - 6.25)) < 1e-9 and cell.second_fell_back and not cell.first_fell_back,
          "haiku/Jev's $4.225 less E's fall back to A at $6.25")  # fmt: skip
    check(list(found["jev_against_sonnet"]) == ["haiku/Jev less haiku/Sonnet", "E/Jev less E/Sonnet", "F/Jev less F/Sonnet",
                                                "G/Jev less G/Sonnet", "G+/Jev less G+/Sonnet"],
          "pairs are keyed 'first less second' by label")  # fmt: skip
    short = dict(walks)
    del short[F]
    try:
        cc.compare_pairs(draws, mixes, short)
        refused = False
    except cc.ComparisonError:
        refused = True
    check(refused, "a walk missing for a named router is refused")


# Four items, two tasks, haiku and opus. haiku is right on x1, x3 and y1 and wrong on x2 (which a free check rejects);
# opus is right on x1, x2 and y1 and wrong on x3. Jev scores: haiku x1 .9, x3 .8, y1 .4; opus x1 .8, x2 .7, x3 .1, y1 .2.
TINY = synthetic({
    "x1": ("t1", {"haiku": (0.001, 1), "opus": (0.004, 1)}),
    "x2": ("t1", {"haiku": (0.001, 0), "opus": (0.004, 1)}),
    "x3": ("t1", {"haiku": (0.001, 1), "opus": (0.004, 0)}),
    "y1": ("t2", {"haiku": (0.001, 1), "opus": (0.004, 1)}),
})  # fmt: skip
JEV_SCORES = {("x1", "haiku"): 0.9, ("x3", "haiku"): 0.8, ("y1", "haiku"): 0.4,
              ("x1", "opus"): 0.8, ("x2", "opus"): 0.7, ("x3", "opus"): 0.1, ("y1", "opus"): 0.2}  # fmt: skip


def tiny_table() -> GateScoreTable:
    rows = []
    for item in TINY.items:
        for config in ("haiku", "opus"):
            failed = (item, config) == ("x2", "haiku")
            score = JEV_SCORES.get((item, config))
            rows.append(GateRow(item, config, not failed, "stop" if failed else None,
                                jev=None if score is None else PaidScore(score, 0.0001, False)))  # fmt: skip
    return GateScoreTable.build(rows, [(r.item_id, r.config) for r in rows])


def metric_checks() -> None:
    print("\n-- gate metrics as data --")
    table = tiny_table()
    free = cm.rates_data(cm.gate_rates(table, TINY, FREE_GATE, None))
    # The free checks see every answer: haiku has 4, three right and one wrong (x2), and x2 is the one they reject, so no
    # wrong answer is accepted and no right one rejected. opus passes all four.
    check(free["haiku"]["pooled"] == {"answers": 4, "gated": 4, "right": 3, "wrong": 1, "wrong_accepted": 0,
                                      "right_rejected": 0, "false_accept_rate": 0.0, "false_escalate_rate": 0.0},
          "the free checks' row: all four haiku answers gated, the one wrong answer rejected")  # fmt: skip
    check(free["opus"]["pooled"]["wrong"] == 1 and free["opus"]["pooled"]["wrong_accepted"] == 1
          and free["opus"]["pooled"]["false_accept_rate"] == 1.0,
          "opus's one wrong answer passes the free checks")  # fmt: skip
    jev = cm.rates_data(cm.gate_rates(table, TINY, JEV_GATE, 0.5))
    # At .5 haiku's gated answers are x1 .9 (right, accepted), x3 .8 (right, accepted), y1 .4 (right, rejected): 3 gated
    # of 4 answers, 3 right, none wrong, one right rejected -> false escalate 1/3. By task: t1 has 3 answers, 2 gated;
    # t2 has y1 alone, rejected.
    check(jev["haiku"]["pooled"] == {"answers": 4, "gated": 3, "right": 3, "wrong": 0, "wrong_accepted": 0,
                                     "right_rejected": 1, "false_accept_rate": None, "false_escalate_rate": 1 / 3},
          "Jev at .5 on haiku: one of three right answers rejected, no wrong answer to accept")  # fmt: skip
    check(jev["haiku"]["t1"]["gated"] == 2 and jev["haiku"]["t2"]["right_rejected"] == 1
          and jev["haiku"]["t2"]["false_escalate_rate"] == 1.0,
          "and by task")  # fmt: skip
    check(jev["opus"]["pooled"]["wrong_accepted"] == 0 and jev["opus"]["pooled"]["right_rejected"] == 1,
          "opus: x3 (wrong, .1) rejected, y1 (right, .2) rejected, x1 and x2 accepted -> one right rejected is y1")  # fmt: skip
    draws = resample(TINY)
    data = cm.gate_metrics_data(table, TINY, draws, {JEV_GATE: 0.5}, {})
    check(set(data) == {"free", JEV_GATE}, "one entry for the free checks and one per paid gate with a walk")
    check(data[JEV_GATE]["fixed_cutoff"] == 0.5 and data[JEV_GATE]["at_fixed_cutoff"] == jev,
          "a paid gate's rates at its fixed cutoff")  # fmt: skip
    # opus, task t1: right x1 .8 and x2 .7, wrong x3 .1: both pairs concordant -> 1.0 over 2 pairs. haiku has no wrong
    # answer, so no pair in t1 or t2.
    conc = data[JEV_GATE]["within_task_concordance"]
    check(conc["opus"]["pooled"]["value"] == 1.0 and conc["opus"]["size"]["t1"] == 2 and conc["haiku"]["pooled"]["value"] is None,
          "within-task concordance per config, none where there is no pair")  # fmt: skip
    cheapest = Step(None, 0.75, 1.0, 0.0, True, tuple(simulate(TINY, always(TINY, "haiku"))), 0.8)
    a = Step(None, 0.75, 4.0, 0.0, True, tuple(simulate(TINY, always(TINY, "opus"))))
    walks = {"m": {1.0: Walk(1.0, 0.75, (a, cheapest)), 0.99: Walk(0.99, 0.74, (a,))}}
    data = cm.gate_metrics_data(table, TINY, draws, {JEV_GATE: 0.5}, {JEV_GATE: walks})
    points = data[JEV_GATE]["operating_points"]["m"]
    check(points["100%"]["cutoff"] == 0.8 and points["100%"]["fell_back"] is False
          and points["100%"]["rates"]["haiku"]["pooled"]["right_rejected"] == 1,
          "a pure cascade's operating point: the gate measured at the cutoff it walked to (.8 accepts x1 .9 and x3 .8, rejects y1 .4)")  # fmt: skip
    check(points["99%"] == {"cutoff": None, "fell_back": True, "rates": None},
          "a point that fell back to A has no cutoff and no rates")  # fmt: skip


def latency_checks() -> None:
    print("\n-- estimated latency --")
    dev = synthetic({
        "x1": ("banking77", {"haiku": (0.001, 1), "opus": (0.004, 1)}),
        "x2": ("banking77", {"haiku": (0.001, 0), "opus": (0.004, 1)}),
        "m1": ("mbppplus", {"haiku": (0.001, 1), "opus": (0.004, 1)}),
    })  # fmt: skip
    mix = {"banking77": 0.5, "mbppplus": 0.5}  # x1 and x2 weigh .25 each, m1 .5
    rows = [GateRow("x1", "haiku", True, None, jev=PaidScore(0.9, 0.0, False)),
            GateRow("x2", "haiku", True, None, jev=PaidScore(0.2, 0.0, False)),
            GateRow("m1", "haiku", True, None, jev=PaidScore(0.8, 0.0, False)),
            GateRow("x1", "opus", True, None), GateRow("x2", "opus", True, None), GateRow("m1", "opus", True, None)]  # fmt: skip
    table = GateScoreTable.build(rows, [(r.item_id, r.config) for r in rows])
    medians = GateMedians(code_check=0.2, jev=0.06, sonnet=1.5)
    haiku_then_opus = [Outcome("x1", "banking77", "x1", 0.0, 1, ("haiku",)),
                       Outcome("x2", "banking77", "x2", 0.0, 1, ("haiku", "opus")),
                       Outcome("m1", "mbppplus", "m1", 0.0, 1, ("haiku",))]  # fmt: skip
    a_step = Step(None, 1.0, 1.0, 0.0, True, tuple(simulate(dev, always(dev, "opus"))))
    cascade = Walk(1.0, 1.0, (a_step, Step(None, 1.0, 0.5, 0.0, True, tuple(haiku_then_opus), 0.5)))
    # The pilot's medians: banking77 haiku 2.0, opus 2.2; mbppplus haiku 1.6. A Jev call 0.06, the code check 0.2 on the
    # one request with an assert line. x1: 2.0 + 0.06 = 2.06; x2: 2.0 + 0.06 + 2.2 = 4.26; m1: 0.2 + 0.06 + 1.6 = 1.86.
    # m1 weighs .5, exactly half, so the median is the midpoint with the next value, x1: (1.86 + 2.06) / 2 = 1.96.
    got = cm.walk_latency(dev, table, cascade, mix, gate=JEV_GATE, medians=medians, code_items={"m1"})
    check(abs(got - 1.96) < 1e-9, "a cascade's items sum the start, the gates called and the escalation: median 1.96 s")
    free = Walk(1.0, 1.0, (a_step, Step(None, 1.0, 0.5, 0.0, True, tuple(haiku_then_opus), None)))
    # With the free checks alone only m1 pays a check (0.2): x1 2.0, x2 2.0 + 2.2, m1 1.8 -> sorted 1.8 (.5): the
    # midpoint with 2.0 is 1.9.
    got = cm.walk_latency(dev, table, free, mix, gate=FREE_GATE, medians=medians, code_items={"m1"})
    check(abs(got - 1.9) < 1e-9, "the free-checks cascade pays only the code check, 1.9 s")
    # A router alone pays its time on every item: x1 2.0 + .01, x2 2.0 + 2.2 + .01, m1 1.6 + .01 -> 1.61 and 2.01 midpoint 1.81.
    got = cm.walk_latency(dev, table, cascade, mix, router=0.01)
    check(abs(got - 1.81) < 1e-9, "a router's time is paid on every item")
    fell = Walk(1.0, 1.0, (a_step,))
    # A: x1 2.2, x2 2.2, m1 3.4 -> exactly half at 2.2 + 2.2: midpoint (2.2 + 3.4) / 2 = 2.8, and no router is charged.
    check(abs(cm.walk_latency(dev, table, fell, mix, router=5.0, gate=JEV_GATE, medians=medians) - 2.8) < 1e-9
          and abs(cm.fixed_latency(dev, "opus", mix) - 2.8) < 1e-9, "a fall back is A: 2.8 s, no router, no gate")  # fmt: skip
    # haiku alone: x1 2.0, x2 2.0, m1 1.6 -> m1 (.5) is half: the midpoint with 2.0 is 1.8.
    check(
        abs(cm.fixed_latency(dev, "haiku", mix) - 1.8) < 1e-9, "a fixed config's latency is its pilot medians' (1.8 s)"
    )
    per_task = GateMedians(0.2, {"banking77": 0.05, "mbppplus": 0.1}, 1.5)
    got = cm.walk_latency(dev, table, cascade, mix, gate=JEV_GATE, medians=per_task, code_items={"m1"})
    # Per-task Jev medians: x1 2.05, x2 4.25, m1 0.2 + 0.1 + 1.6 = 1.9 -> sorted 1.9 (.5), 2.05: midpoint 1.975.
    check(abs(got - 1.975) < 1e-9, "a gate's median can be one per task")
    check(PILOT_MEDIANS["banking77"]["haiku"] == 2.0, "the pilot's median the work above uses")
    try:
        cm.walk_latency(dev, table, cascade, mix, gate=JEV_GATE)
        refused = False
    except ValueError:
        refused = True
    check(refused, "a cascade without the gates' medians is refused")


def build_checks() -> None:
    print("\n-- table v4 on a synthetic dev --")
    dev, table, routers = synthetic_dev()
    seconds = {task: 0.01 for task in TASKS}
    inputs = table_v4.Inputs(
        dev, resample(dev), table, routers, CUTOFFS, GateMedians(0.2, dict(seconds), {t: 1.5 for t in TASKS}),
        {E: 4.1e-6, F: dict(seconds), G: dict(seconds), G_PLUS: dict(seconds)},
        frozenset(i for i, item in dev.items.items() if item.task == "mbppplus"),
    )  # fmt: skip
    built = table_v4.build(inputs)
    text = json.dumps(built, allow_nan=False)  # a NaN or infinity anywhere would raise
    check(list(built["cascades"]) == list(CASCADES) and all(
        list(c["operating_points"]) == ["realistic", "uniform"]
        and all(list(by) == ["100%", "99%", "97%"] for by in c["operating_points"].values())
        for c in built["cascades"].values()),
        "fifteen cascades in item 8's order, each with both mixes and three targets")  # fmt: skip
    check(built["comparisons"]["count"] == 51 and {g: len(b["pairs"]) for g, b in built["comparisons"]["groups"].items()}
          == {"gate_adds": 12, "router_adds": 12, "paid_gate_over_free": 10, "jev_against_sonnet": 5,
              "cascades_against_routers": 12},
          "51 comparisons in item 22's five groups")  # fmt: skip
    check(built["comparisons"]["headline"] == ["haiku/Jev less G", "haiku/Jev less G+"]
          and "haiku/Jev less G+" in built["comparisons"]["groups"]["cascades_against_routers"]["pairs"],
          "the declared headline pairs are among the comparisons")  # fmt: skip
    check(set(built["comparisons"]["matched_cost"]) == set(CASCADES) and set(built["comparisons"]["mcnemar"]) == set(CASCADES)
          and set(built["comparisons"]["mcnemar"][cascade_section(HAIKU, JEV_GATE)]) == set(TASKS),
          "a matched-cost row and per-task McNemar rows for every cascade")  # fmt: skip
    check(set(built["gate_metrics"]) == {"free", JEV_GATE, "sonnet"}
          and set(built["gate_metrics"]["sonnet"]["within_task_concordance"]) == {"haiku", "sonnet", "opus_low", "opus"}
          and set(built["gate_metrics"][JEV_GATE]["within_task_concordance"]) == {"haiku", "sonnet", "opus_low", "opus"},
          "gate metrics for the free checks and both paid gates, a config per column the table holds")  # fmt: skip
    lat = built["latency"]
    check(set(lat["routers"]) == {E, F, G, G_PLUS} and set(lat["cascades"]) == set(CASCADES)
          and set(lat["references"]) == {"A", "B", "C", "D", "note"} and lat["references"]["C"] is None
          and set(lat["configs"]) == {"haiku", "sonnet", "opus_low", "opus"},
          "an estimated latency for the references, the routers and the fifteen cascades")  # fmt: skip
    best = built["best_cascade"]
    check(best["best"] in (*CASCADES, None) and (best["best"] is None) == best["open_case"]
          and set(best["savings_share"]) == set(CASCADES) and all(len(v) == 6 for v in best["savings_share"].values()),
          "the best cascade is named from six savings per cascade, or the open case recorded")  # fmt: skip
    rows = built["cascades"][cascade_section(HAIKU, FREE_GATE)]["operating_points"]["realistic"]
    check(all(not r["fell_back"] or (r["cutoff"] is None and r["threshold"] is None) for r in rows.values()),
          "a fall back has no cutoff and no threshold")  # fmt: skip
    check(built["task_warnings"]["ifeval"]["warning"] and built["task_warnings"]["mbppplus"]["warning"]
          and built["task_warnings"]["banking77"]["warning"] is None and built["task_warnings"]["mmlu_pro"]["warning"] is None,
          "A7 item 2's warnings on IFEval and MBPP+ only")  # fmt: skip
    check("realistic" in text and built["split"] == "dev" and "descriptive" in built["comparisons"]["description"],
          "labeled descriptive, dev")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# The committed table against the committed sections (CI too) and against the stored records (hook only)

STAGE7_70_MERGE = "7df1d11"  # the integration tip table v4 was built on: tables v2 and v3 are as it has them
MIXES = ("realistic", "uniform")
TARGET_KEYS = ("100%", "99%", "97%")
ROUTER_SECTIONS = dict(NAMED_ROUTERS)  # letter -> section name


def close(a: float | None, b: float | None, tol: float = 1e-9) -> bool:
    return (a is None and b is None) or (
        a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    )


def op_step(point: dict) -> dict:
    """The operating point's step as the committed section has it, found by the walk's rule from its steps alone: the
    cheapest step before the first that doesn't clear (A, the first step, counts), ties to the earlier."""
    steps = point["steps"]
    stop = next((i for i, step in enumerate(steps) if not step["clears"]), len(steps))
    best = steps[0]
    for step in steps[1:stop]:
        if step["cost_per_1k"] < best["cost_per_1k"] - 1e-9:
            best = step
    return best


def section_numbers(name: str, mix: str, target: str) -> tuple[dict, dict, dict]:
    """(the section's point, its operating step, A's step) for one policy, mix and target."""
    from analysis.operating_points import read_section

    point = read_section(name)[mix][target]
    return point, op_step(point), point["steps"][0]


def record_checks(table: dict) -> None:
    from analysis.check_stage5 import mcnemar_p

    print("\n-- the committed table against the committed sections --")
    check(list(table["cascades"]) == list(CASCADES) and table["split"] == "dev" and table["n"] == 744,
          "table v4 holds the fifteen cascades in item 8's order, on dev's 744 items")  # fmt: skip
    problems_found = []
    for key, body in table["cascades"].items():
        for mix in MIXES:
            for target in TARGET_KEYS:
                row = body["operating_points"][mix][target]
                point, step, a_step = section_numbers(key, mix, target)
                fell = step is a_step
                if (
                    row["fell_back"] != point["fell_back"]
                    or row["fell_back"] != fell
                    or row["threshold"] != point.get("threshold")
                    or row["cutoff"] != point.get("cutoff")
                    or not close(row["target"], point["target"])
                    or not close(row["margin"], point["margin"])
                    or not close(row["quality"]["value"], step["quality"])
                    or not close(row["cost_per_1k"]["value"], step["cost_per_1k"])
                    or not close(row["savings_vs_A"]["share"], 1 - step["cost_per_1k"] / a_step["cost_per_1k"])
                    or not close(row["savings_vs_A"]["per_1k"], a_step["cost_per_1k"] - step["cost_per_1k"])
                ):
                    problems_found.append(f"{key} {mix} {target}")
                if fell and (row["escalation_rate"] != 0.0 or row["captured"]["value"] != 0.0):
                    problems_found.append(f"{key} {mix} {target}: a fall back escalates or captures")
    check(
        problems_found == [],
        f"all 90 rows: where the walk stopped, quality, cost and savings are the section's ({problems_found[:2]})",
    )
    check(sum(r["fell_back"] for b in table["cascades"].values() for m in b["operating_points"].values() for r in m.values()) == 36,
          "36 of the 90 operating points fall back to A, as #70 recorded")  # fmt: skip

    best = table["best_cascade"]
    shares = {key: [1 - op_step(point)["cost_per_1k"] / point["steps"][0]["cost_per_1k"]
                    for mix in MIXES for target in TARGET_KEYS
                    for point in [section_numbers(key, mix, target)[0]]] for key in CASCADES}  # fmt: skip
    found = cc.best_cascade(shares)
    check(best["best"] == found.best and best["open_case"] == found.open_case and list(best["tied_at_top"]) == list(found.tied)
          and all(close(best["mean_savings_share"][k], v) for k, v in found.means.items()),
          f"the best cascade is item 24's rule on the sections' savings alone: {best['label']}")  # fmt: skip
    top = max(found.means.values())
    runners = sorted((v for v in found.means.values() if v < top), reverse=True)
    check(best["best"] is None or (found.means[best["best"]] >= top - 1e-9 and (not runners or runners[0] < top)),
          "no other cascade has a higher mean")  # fmt: skip

    groups = table["comparisons"]["groups"]
    cost_quality: dict[str, dict] = {}
    for name in (*ROUTER_SECTIONS, *CASCADES):
        section = ROUTER_SECTIONS.get(name, name)
        cost_quality[name] = {}
        for mix in MIXES:
            for t in TARGET_KEYS:
                _, step, a_step = section_numbers(section, mix, t)
                cost_quality[name][(mix, t)] = (step["cost_per_1k"], step["quality"], step is a_step)
    wrong = []
    for group, pairs in cc.comparison_pairs().items():
        body = groups[group]["pairs"]
        if list(body) != [f"{cc.label(a)} less {cc.label(b)}" for a, b in pairs]:
            wrong.append(f"{group}: pairs")
            continue
        for a, b in pairs:
            for mix in MIXES:
                for t in TARGET_KEYS:
                    cell = body[f"{cc.label(a)} less {cc.label(b)}"][mix][t]
                    (ca, qa, fa), (cb, qb, fb) = cost_quality[a][(mix, t)], cost_quality[b][(mix, t)]
                    both = fa and fb
                    if both:
                        ok = (
                            cell["reading"] == "both fall back to A"
                            and cell["cost_per_1k"] is None
                            and cell["quality"] is None
                        )
                    else:
                        ok = (cell["reading"] == "" and close(cell["cost_per_1k"]["value"], ca - cb)
                              and close(cell["quality"]["value"], qa - qb)
                              and cell["cost_per_1k"]["low"] <= cell["cost_per_1k"]["high"]
                              and cell["quality"]["low"] <= cell["quality"]["high"])  # fmt: skip
                    if not ok:
                        wrong.append(f"{cc.label(a)} less {cc.label(b)} {mix} {t}")
    check(
        wrong == [],
        f"all 51 comparisons x 6 cells: a point estimate is the sections' difference, both falling back reads so ({wrong[:2]})",
    )
    check(table["comparisons"]["count"] == 51 and "no multiplicity correction" in table["comparisons"]["description"],
          "51 comparisons, labeled descriptive with no multiplicity correction")  # fmt: skip
    bad = []
    for key, by_task in table["comparisons"]["mcnemar"].items():
        for task, by_mix in by_task.items():
            for mix, by_target in by_mix.items():
                for t, row in by_target.items():
                    fell = table["cascades"][key]["operating_points"][mix][t]["fell_back"]
                    if (row is None) != fell or (
                        row is not None and not close(row["p"], mcnemar_p(int(row["b"]), int(row["c"])))
                    ):
                        bad.append(f"{key} {task} {mix} {t}")
    check(bad == [] and set(next(iter(table["comparisons"]["mcnemar"].values()))) == {"banking77", "ifeval", "mbppplus", "mmlu_pro"},
          "per-task McNemar rows: null exactly where the point falls back, p the exact binomial of b and c")  # fmt: skip
    check(table["task_warnings"]["ifeval"]["warning"] and table["task_warnings"]["mbppplus"]["warning"]
          and table["task_warnings"]["banking77"]["warning"] is None,
          "A7 item 2's warnings are on IFEval and MBPP+")  # fmt: skip
    check(all(table["comparisons"]["matched_cost"][k][m][t] == table["cascades"][k]["operating_points"][m][t]["matched_cost"]
              for k in CASCADES for m in MIXES for t in TARGET_KEYS),
          "each cascade's matched-cost difference from the envelope is its row's")  # fmt: skip
    for unit in table_v4.UNITS:
        if unit not in table["units"]:
            check(False, f"units name {unit}")
    check(table["units"] == table_v4.UNITS, "every table has its units")
    from analysis.checking import git

    held_ok = True
    for earlier in ("table_v2.json", "table_v3.json"):
        held = git("show", f"{STAGE7_70_MERGE}:figures/{earlier}")
        if held.returncode == 0:
            held_ok &= (table_v4.FIGURES / earlier).read_text() == held.stdout
    check(held_ok, "tables v2 and v3 are as they were: table v4 has its own file")


def direct_rates(rows, matrix, gate: str, cutoff: float | None) -> dict:
    """Counts by plain loops over the table's rows, written apart from `gate_rates`: config, task or pooled."""
    out: dict = {}
    for row in rows:
        task = matrix.items[row.item_id].task
        right = matrix.cells[(row.item_id, row.config)].success == 1
        if (
            gate != FREE_GATE
            and row.paid(gate) is None
            and not any(r.paid(gate) is not None for r in rows if r.config == row.config)
        ):
            continue
        for key in ("pooled", task):
            d = out.setdefault(row.config, {}).setdefault(
                key, {"answers": 0, "gated": 0, "right": 0, "wrong": 0, "wrong_accepted": 0, "right_rejected": 0}
            )
            d["answers"] += 1
            if gate == FREE_GATE:
                accepted = row.free_passed
            elif not row.free_passed:
                continue
            else:
                score = row.paid(gate).score  # type: ignore[union-attr]
                accepted = score is not None and score >= cutoff  # type: ignore[operator]
            d["gated"] += 1
            d["right" if right else "wrong"] += 1
            d["wrong_accepted"] += (not right) and accepted
            d["right_rejected"] += right and not accepted
    return out


def failures_of(section) -> int:  # type: ignore[no-untyped-def]
    """How many FAILs a section prints, without counting them as this script's own: the way a check proves that a
    section prints FAIL for a bad input."""
    import contextlib
    import io

    before = len(problems)
    with contextlib.redirect_stdout(io.StringIO()):
        section()
    failed = len(problems) - before
    del problems[before:]
    return failed


def same(a, b, tol: float = 1e-9) -> bool:  # type: ignore[no-untyped-def]
    """Two JSON values equal, floats to `tol`."""
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        return close(a, b, tol)
    return bool(a == b)


def weighted_median(values: dict, weights: dict[str, Fraction]) -> float:
    """The weighted median in exact fractions: the value where the sorted items' weight first reaches half, the midpoint
    with the next one where it is exactly half. Written apart from `analysis.latency_estimate.weighted_median`."""
    order = sorted(values, key=lambda k: values[k])
    half = sum((weights[k] for k in order), Fraction(0)) / 2
    reached = Fraction(0)
    for position, k in enumerate(order):
        reached += weights[k]
        if reached > half or (reached == half and position + 1 == len(order)):
            return float(values[k])
        if reached == half:
            return float((Fraction(values[k]) + Fraction(values[order[position + 1]])) / 2)
    raise AssertionError("unreachable")


def stored_checks(table: dict) -> None:
    """Table v4 against the stored dev records (hook only)."""
    import json as _json
    import statistics

    from analysis import collect_sonnet_gate, gate_dev_run
    from analysis.benchmark import read_rows
    from analysis.collect_gates import read_free_records
    from analysis.gate_answers import requests_of
    from analysis.latency_estimate import PILOT_MEDIANS
    from analysis.matrix import load as load_matrix
    from analysis.simulator import load_mixes
    from harness.client import PROJECT_ROOT
    from routers.gates import assert_line

    dev = load_matrix().split("dev")
    gates = gate_dev_run.dev_table()
    cutoffs = gate_dev_run.read_cutoffs()
    mixes = load_mixes()
    task_of = {str(r["item_id"]): str(r["source"]) for split in ("dev", "train") for r in read_rows(split=split)}

    print("\n-- the gate metrics, counted again from the gate-score table --")
    sheet = table["gate_metrics"]
    cuts = [(JEV_GATE, cutoffs[JEV_GATE]), ("sonnet", cutoffs["sonnet"])]
    wrong = []
    free_found = direct_rates(gates.rows, dev, FREE_GATE, None)
    wrong += [f"free {c} {t}" for c, by in free_found.items() for t, d in by.items()
              if {k: sheet["free"]["rates"][c][t][k] for k in d} != d]  # fmt: skip
    for gate, cut in cuts:
        found = direct_rates(gates.rows, dev, gate, cut)
        got = sheet[gate]["at_fixed_cutoff"]
        wrong += [
            f"{gate} {c} {t}" for c, by in found.items() for t, d in by.items() if {k: got[c][t][k] for k in d} != d
        ]
        wrong += [f"{gate} configs" for _ in [0] if set(found) != set(got)]
        for mix, by_target in sheet[gate]["operating_points"].items():
            for target, point in by_target.items():
                if point["rates"] is None:
                    continue
                again = direct_rates(gates.rows, dev, gate, point["cutoff"])
                wrong += [f"{gate} {mix} {target} {c} {t}" for c, by in again.items() for t, d in by.items()
                          if {k: point["rates"][c][t][k] for k in d} != d]  # fmt: skip
    check(
        wrong == [],
        f"false accept / false escalate counts at the fixed and operating cutoffs, every config and task ({wrong[:2]})",
    )
    ratios = [(d["wrong_accepted"] / d["wrong"] if d["wrong"] else None, d["right_rejected"] / d["right"] if d["right"] else None)
              for by in sheet[JEV_GATE]["at_fixed_cutoff"].values() for d in by.values()]  # fmt: skip
    check(all(close(d["false_accept_rate"], r[0]) and close(d["false_escalate_rate"], r[1])
              for by, rs in [(sheet[JEV_GATE]["at_fixed_cutoff"], ratios)] for d, r in zip((x for b in by.values() for x in b.values()), rs, strict=True)),
          "and the rates are those counts' ratios")  # fmt: skip
    check(set(sheet["sonnet"]["at_fixed_cutoff"]) == {"haiku", "sonnet", "opus_low"}
          and set(sheet[JEV_GATE]["at_fixed_cutoff"]) == {"haiku", "sonnet", "opus_low", "opus"},
          "the Jev gate has four configs' rows and the Sonnet gate three (no opus entry)")  # fmt: skip

    print("\n-- the latency inputs, from the stored records --")
    free = read_free_records("dev")
    code_median = statistics.median(float(r["code_duration_s"]) for r in free if r["code_applied"])
    jev_lines = [
        _json.loads(x)
        for x in (PROJECT_ROOT / "results" / "gates" / "jev_gate_dev.jsonl").read_text().splitlines()
        if x
    ]
    jev_lines = [r for r in jev_lines if r["pass"] == 1]
    kept: dict[str, list[float]] = {}
    for number, r in enumerate(jev_lines):
        if number > 0 and not r["new_connection"]:
            kept.setdefault(task_of[r["item_id"]], []).append(float(r["seconds"]))
    jev_median = {t: statistics.median(v) for t, v in kept.items()}
    sonnet_kept: dict[str, list[float]] = {}
    for r in collect_sonnet_gate.read_latency_records("train"):
        if not r["first_call"]:
            sonnet_kept.setdefault(task_of[r["item_id"]], []).append(float(r["seconds"]))
    sonnet_median = {t: statistics.median(v) for t, v in sonnet_kept.items()}
    inputs = table["latency"]["inputs"]
    check(close(inputs["code_check_s"], code_median) and same(inputs["jev_gate_s"], dict(sorted(jev_median.items())))
          and same(inputs["sonnet_gate_s"], dict(sorted(sonnet_median.items()))),
          "the code check's median (dev), the Jev gate's per task (dev, first call and reconnects left out) and the "
          "Sonnet gate's per task (train sample, first call left out) are the stored records'")  # fmt: skip
    code_items = {item for item, r in requests_of("dev").items() if assert_line(r.prompt)}
    check(inputs["dev_items_with_a_code_check"] == len(code_items) == sum(1 for i in dev.items.values() if i.task == "mbppplus"),
          "the code check applies to the dev MBPP+ items, all of them")  # fmt: skip

    print("\n-- the pure cascades' latency and escalation, from the table alone --")
    haiku = {row.item_id: row for row in gates.rows if row.config == "haiku"}
    medians = {JEV_GATE: jev_median, "sonnet": sonnet_median}
    wrong = []
    for key, gate in (
        (cascade_section(HAIKU, FREE_GATE), FREE_GATE),
        (cascade_section(HAIKU, JEV_GATE), JEV_GATE),
        (cascade_section(HAIKU, "sonnet"), "sonnet"),
    ):
        for mix in MIXES:
            counts: dict[str, int] = {}
            for item in dev.items.values():
                counts[item.task] = counts.get(item.task, 0) + 1
            weights = {i: Fraction(str(mixes[mix][item.task])) / counts[item.task] for i, item in dev.items.items()}
            for target in TARGET_KEYS:
                row = table["cascades"][key]["operating_points"][mix][target]
                a_seconds = weighted_median({i: PILOT_MEDIANS[it.task]["opus"] for i, it in dev.items.items()}, weights)
                seconds, escalated = {}, 0.0
                for i, item in dev.items.items():
                    r = haiku[i]
                    cut = row["cutoff"]
                    if row["fell_back"]:
                        seconds[i] = PILOT_MEDIANS[item.task]["opus"]
                        continue
                    total = PILOT_MEDIANS[item.task]["haiku"] + (code_median if i in code_items else 0.0)
                    accepted = r.free_passed
                    if r.free_passed and gate != FREE_GATE:
                        total += medians[gate][item.task]
                        score = r.paid(gate).score  # type: ignore[union-attr]
                        accepted = score is not None and score >= cut
                    if not accepted:
                        total += PILOT_MEDIANS[item.task]["opus"]
                        escalated += weights[i]
                    seconds[i] = total
                want = weighted_median(seconds, weights) if not row["fell_back"] else a_seconds
                if not close(table["latency"]["cascades"][key][mix][target], want) or not close(
                    row["escalation_rate"], escalated
                ):
                    wrong.append(f"{key} {mix} {target}")
    check(
        wrong == [],
        f"haiku/free, haiku/Jev and haiku/Sonnet: all 18 estimated latencies and escalation rates recomputed ({wrong[:2]})",
    )
    for mix in MIXES:
        counts = {}
        for item in dev.items.values():
            counts[item.task] = counts.get(item.task, 0) + 1
        weights = {i: Fraction(str(mixes[mix][item.task])) / counts[item.task] for i, item in dev.items.items()}
        for config in ("haiku", "sonnet", "opus_low", "opus"):
            got = weighted_median({i: PILOT_MEDIANS[it.task][config] for i, it in dev.items.items()}, weights)
            if not close(table["latency"]["configs"][config][mix], got):
                check(False, f"fixed config {config} {mix} latency")
    check(close(table["latency"]["references"]["A"]["realistic"], table["latency"]["configs"]["opus"]["realistic"]),
          "the references A and B are always-opus and always-haiku")  # fmt: skip

    print("\n-- table v4 rebuilt --")
    rebuilt = _json.loads(_json.dumps(table_v4.build(table_v4.real_inputs())))
    check(
        same(rebuilt, table),
        "the committed figures/table_v4.json equals a rebuild from the stored records, the dev matrix and the committed files",
    )
    mutated = _json.loads(_json.dumps(table))
    mutated["cascades"][cascade_section(HAIKU, JEV_GATE)]["operating_points"]["realistic"]["97%"]["cost_per_1k"][
        "value"
    ] += 0.01
    check(not same(mutated, rebuilt), "known positive: a cost off by a cent is not the rebuild")


def main() -> int:
    pair_checks()
    best_checks()
    row_checks()
    compare_checks()
    metric_checks()
    latency_checks()
    build_checks()
    from analysis.operating_points import OPERATING_POINTS_FILE
    from harness.client import PROJECT_ROOT

    if table_v4.TABLE_V4.exists() and OPERATING_POINTS_FILE.exists():
        table = json.loads(table_v4.TABLE_V4.read_text())
        record_checks(table)
        slipped = json.loads(json.dumps(table))
        slipped["cascades"][cascade_section(HAIKU, JEV_GATE)]["operating_points"]["realistic"]["97%"]["cost_per_1k"][
            "value"
        ] += 0.01
        check(
            failures_of(lambda: record_checks(slipped)) > 0,
            "known positive: a cost off by a cent is found against the sections",
        )
        slipped = json.loads(json.dumps(table))
        pair = next(iter(slipped["comparisons"]["groups"]["gate_adds"]["pairs"].values()))
        cell = next(c for by in pair.values() for c in by.values() if c["cost_per_1k"])
        cell["cost_per_1k"]["value"] += 0.01
        check(failures_of(lambda: record_checks(slipped)) > 0, "known positive: a comparison off by a cent is found")
        results = PROJECT_ROOT / "results"
        if (results / "dev_matrix.sqlite").exists() and all(
            (results / "gates" / f"{n}_dev.jsonl").exists() for n in ("free_checks", "jev_gate", "sonnet_gate")
        ):
            stored_checks(table)
        else:
            print("\n(the dev matrix or the stored dev records are not here: the stored-record sections are skipped)")
    else:
        print("\n(figures/table_v4.json is not here yet)")
    print(f"\n{len(problems)} of the checks failed" if problems else "\nall checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
