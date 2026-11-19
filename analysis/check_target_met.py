"""Check target met, the references' flip rates and the headline on test (PLAN_v4.md Amendment A19, items 6 and 8; A17 item 24;
A8 item 4; issue #77, #88).

Run from the project root:  uv run python -m analysis.check_target_met

On hand-worked synthetic tables: a four-item matrix with two tasks, resamples written out by hand, and a dev table of fifteen
cascades made up for the rule. No results/, no router or gate on a real item, nothing reads test.

- **target met** is the policy's quality less the target's share of A's quality, on the same items and in every resample, at a
  margin of zero. The point value, the percentile interval and the standard error over the resamples, and the share of resamples
  that meet the target, equal the hand-worked ones (the expected values are written out below from each resample's weights, and
  the percentile by the formula, not from the module);
- a point that falls back to A is labelled **falls back to A**, has no numbers and isn't counted, whether it is on the policy's
  line or A's own; a point that misses is **missed**; the counts add up;
- the counts hold for every policy's rows without a line per policy: the rows that `analysis/operating_eval.py` builds are the
  rows target met reads, so a policy added by one line is covered, and a part that holds policy rows no one evaluated that way
  is refused;
- the headline takes the best cascade by A17 item 24's rule on dev's table and nothing else: a made-up test table where another
  cascade does better leaves it where it was, and a re-selected dev table moves it. A headline that misses says **not met**,
  with its quality and cost; one that meets says so; one that falls back says that;
- dev's flip rates, which carry over to test (test has no repeat runs), are the plan's numbers (A6), and the oracle is called an
  optimistic upper bound;
- the tables step reports refusal and truncation counts with the tables, and a step that tries to say them earlier is caught by
  the same search as the grades' markers.
"""

import json
import re
import sys
import tempfile
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from analysis import final_headline, flip_rates, target_met
from analysis.bootstrap import Bootstrap
from analysis.cascade_comparisons import GATE_COST_ORDER
from analysis.check_final_run import rehearsed
from analysis.final_design import TABLE_V4
from analysis.final_record import RECORD_FILE
from analysis.final_run import STEPS, Run, Step, StepResult
from analysis.matrix import Matrix
from analysis.offline_half import STOPS_LABEL
from analysis.operating_eval import (
    evaluate,
    fixed_outcomes,
    forget_evaluated,
    frozen_points,
    mix_rows,
    router_traces,
    shared_fixed,
    task_rows,
)
from analysis.rehearsal_world import GRADE_MARKER
from analysis.router_names import CASCADES, FREE_GATE, HAIKU, JEV_GATE, SONNET_GATE, cascade_section
from analysis.router_table import ItemRouterScores, RouterScoreTable
from analysis.simulator import Outcome
from harness.client import PROJECT_ROOT

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(error: type[Exception], call: Callable[[], object]) -> bool:
    try:
        call()
    except error:
        return True
    return False


def close(a: float | None, b: float, tolerance: float = 1e-12) -> bool:
    return a is not None and abs(a - b) <= tolerance


# Four items, each its own source group: a1 and a2 in task a, b1 and b2 in task b, weighed half and half. Four resamples,
# written out by hand (how often each item is drawn). A task's weight is shared among its drawn items, so a resample's
# weights are, in the order a1 a2 b1 b2:
#   r0 draws 1 1 1 1: .25 .25 .25 .25      r1 draws 2 0 1 1: .5 0 .25 .25
#   r2 draws 0 2 2 0: 0 .5 .5 0            r3 draws 1 1 0 2: .25 .25 0 .5
ITEMS = ("a1", "a2", "b1", "b2")
TASKS = ("a", "a", "b", "b")
DRAWS = Bootstrap(ITEMS, TASKS, np.array([[1, 1, 1, 1], [2, 0, 1, 1], [0, 2, 2, 0], [1, 1, 0, 2]]))
MIX = {"a": 0.5, "b": 0.5}
WEIGHTS = [(.25, .25, .25, .25), (.5, 0, .25, .25), (0, .5, .5, 0), (.25, .25, 0, .5)]  # fmt: skip


def outcomes_of(success: tuple[int, int, int, int]) -> list[Outcome]:
    return [Outcome(i, t, i, 0.001, s, ("opus",)) for i, t, s in zip(ITEMS, TASKS, success)]


# A is right on a1, a2, b2 and wrong on b1. P is right on a1, b1, b2 and wrong on a2. Q is right on a2 and b2 only.
A, P, Q = outcomes_of((1, 1, 0, 1)), outcomes_of((1, 0, 1, 1)), outcomes_of((0, 1, 0, 1))
# Each policy's quality in each resample (the dot product of the weights and its successes), by hand:
#   A: r0 .75  r1 .5 + .25 = .75  r2 .5  r3 .25 + .25 + .5 = 1.0     (on the items as drawn: .75)
#   P: r0 .75  r1 .5 + .25 + .25 = 1.0  r2 .5  r3 .25 + .5 = .75    (on the items as drawn: .75)
#   Q: r0 .5   r1 .25 (b2)  r2 .5 (a2)  r3 .25 + .5 = .75          (on the items as drawn: .5)
QUALITY = {"A": [0.75, 0.75, 0.5, 1.0], "P": [0.75, 1.0, 0.5, 0.75], "Q": [0.5, 0.25, 0.5, 0.75]}
ON_ITEMS = {"A": 0.75, "P": 0.75, "Q": 0.5}


def percentile(values: list[float], share: float) -> float:
    """numpy's linear percentile, by its definition: the value at rank share x (n - 1) between the sorted values."""
    ordered = sorted(Fraction(v).limit_denominator(10**9) for v in values)
    rank = Fraction(share) * (len(ordered) - 1)
    below = int(rank)
    above = min(below + 1, len(ordered) - 1)
    return float(ordered[below] + (rank - below) * (ordered[above] - ordered[below]))


def hand(policy: str, share: float) -> dict:
    """Target met worked by hand: each resample's quality less `share` of A's, the value on the items, the interval, the
    standard error and the share of resamples at or above zero."""
    diffs = [q - share * a for q, a in zip(QUALITY[policy], QUALITY["A"])]
    mean = sum(diffs) / len(diffs)
    spread = (sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
    return {
        "value": ON_ITEMS[policy] - share * ON_ITEMS["A"],
        "low": percentile(diffs, 0.025),
        "high": percentile(diffs, 0.975),
        "standard_error": spread,
        "met_share": sum(d >= 0 for d in diffs) / len(diffs),
    }


def worked() -> None:
    print("-- target met by hand: paired, at a margin of zero --")
    check(
        all(close(float(sum(w * s for w, s in zip(WEIGHTS[r], v))), QUALITY[name][r])
            for name, v in (("A", (1, 1, 0, 1)), ("P", (1, 0, 1, 1)), ("Q", (0, 1, 0, 1))) for r in range(4)),
        "the hand-written resample qualities are the weights' dot products with the successes",
    )  # fmt: skip
    for policy, share in (("P", 1.0), ("P", 0.99), ("Q", 1.0), ("Q", 0.97)):
        row = target_met.point_row(DRAWS, A, {"P": P, "Q": Q}[policy], MIX, share, False)
        wanted = hand(policy, share)
        difference = row["difference"]
        check(
            close(difference["value"], wanted["value"]) and close(difference["low"], wanted["low"])
            and close(difference["high"], wanted["high"]) and close(difference["standard_error"], wanted["standard_error"])
            and close(row["met_share"], wanted["met_share"]),
            f"{policy} at {share:.0%}: value {wanted['value']:+.4f}, interval {wanted['low']:+.5f} to {wanted['high']:+.5f}, "
            f"met in {wanted['met_share']:.0%} of the resamples",
        )  # fmt: skip
    p100 = target_met.point_row(DRAWS, A, P, MIX, 1.0, False)
    check(p100["reading"] == "met" and close(p100["difference"]["value"], 0.0),
          "P at 100% sits exactly on the target: at a margin of zero that meets it (a zero difference is met)")  # fmt: skip
    check(p100["met_share"] == 0.75 and close(hand("P", 1.0)["met_share"], 0.75),
          "and meets it in three of the four resamples (r0 and r2 on it, r1 above, r3 below)")  # fmt: skip
    p99 = target_met.point_row(DRAWS, A, P, MIX, 0.99, False)
    check(p99["reading"] == "met" and close(p99["difference"]["value"], 0.75 - 0.99 * 0.75),
          "P at 99% clears it by 1% of A's 75%")  # fmt: skip
    q = target_met.point_row(DRAWS, A, Q, MIX, 1.0, False)
    check(q["reading"] == "missed" and close(q["difference"]["value"], -0.25) and q["met_share"] == 0.25,
          "Q at 100% is 25 points under A: missed, and met only in r2")  # fmt: skip
    near = target_met.point_row(DRAWS, A, P, MIX, 1.0 + 5e-10, False)
    check(near["reading"] == "met", "a miss by under the rounding the walk allows (1e-9) is not a miss, as on dev")
    far = target_met.point_row(DRAWS, A, P, MIX, 1.0 + 1e-8, False)
    check(far["reading"] == "missed", "one past that rounding is a miss")
    check(refused(target_met.TargetMetError, lambda: target_met.point_row(DRAWS, A, P[:3], MIX, 1.0, False)),
          "outcomes that don't cover the resampled items are refused")  # fmt: skip


def falling_back() -> None:
    print("\n-- a point that falls back to A --")
    row = target_met.point_row(DRAWS, A, A, MIX, 0.99, True)
    check(row == {"reading": "falls back to A", "difference": None, "met_share": None},
          "a point the file says fell back is labelled falls back to A, with no numbers")  # fmt: skip
    check(target_met.FALLS_BACK == "falls back to A" and row["reading"] not in ("met", "missed"),
          "and is never met, nor missed")  # fmt: skip
    # A policy that falls back is A: were it read as a point, A against itself is met in every resample, trivially.
    as_point = target_met.point_row(DRAWS, A, A, MIX, 0.99, False)
    check(as_point["reading"] == "met" and as_point["met_share"] == 1.0,
          "known positive: A read as a routed point meets every target in every resample, which is why it would pad the count")  # fmt: skip
    rows = [
        target_met.point_row(DRAWS, A, P, MIX, 1.0, False),
        target_met.point_row(DRAWS, A, Q, MIX, 1.0, False),
        row,
        target_met.point_row(DRAWS, A, A, MIX, 1.0, True),
        target_met.point_row(DRAWS, A, P, MIX, 0.99, False),
    ]
    check(target_met.count_of(rows) == {"points": 5, "routes": 3, "met": 2, "missed": 1, "falls_back": 2},
          "five points, three of which route: two met and one missed, the two that fell back are in no count of met")  # fmt: skip


# A policy "toy" routes to haiku (right where P is) at its 99% and 97% points and falls back at 100%, and "toy2" routes to
# sonnet (right where Q is) at 100% and falls back elsewhere, both under both mixes (here one mix, over two tasks).
COSTS = {"haiku": 0.001, "sonnet": 0.002, "opus_low": 0.004, "opus": 0.010}
SUCCESS = {
    "haiku": (1, 0, 1, 1),
    "sonnet": (0, 1, 0, 1),
    "opus_low": (1, 1, 0, 1),
    "opus": (1, 1, 0, 1),
}
MATRIX = Matrix.from_rows([
    {"item_id": i, "split": "test", "task": t, "source_group": i, "config": c, "cost": COSTS[c], "success": SUCCESS[c][k]}
    for k, (i, t) in enumerate(zip(ITEMS, TASKS)) for c in COSTS
])  # fmt: skip
MIXES = {"realistic": MIX, "uniform": MIX}
TOY_SCORES = {
    "toy": {"haiku": 0.9, "sonnet": 0.9, "opus_low": 0.9},
    "toy2": {"haiku": 0.1, "sonnet": 0.9, "opus_low": 0.9},
}
FELL = {"threshold": None, "fell_back": True}


def held(points: dict[str, dict]) -> dict:
    return {"realistic": points, "uniform": points}


SECTIONS = {
    "toy": held(
        {"100%": FELL, "99%": {"threshold": 0.9, "fell_back": False}, "97%": {"threshold": 0.9, "fell_back": False}}
    ),
    "toy2": held({"100%": {"threshold": 0.5, "fell_back": False}, "99%": FELL, "97%": FELL}),
}


def evaluated_into(cache: dict, names: tuple[str, ...] = ("toy", "toy2")) -> dict[str, dict]:
    """Evaluate the toy policies as a policy's part does (`operating_eval`'s evaluate, mix_rows and task_rows over the shared
    fixed outcomes) and return the rows each built."""
    built = {}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "operating_points.json"
        path.write_text(json.dumps(SECTIONS), encoding="utf-8")
        fixed = shared_fixed(cache, MATRIX)
        for name in names:
            table = RouterScoreTable.build(
                [ItemRouterScores(i, dict(TOY_SCORES[name]), 0.0) for i in ITEMS], list(ITEMS)
            )
            outcomes = evaluate(MATRIX, fixed["opus"], frozen_points(name, path), router_traces(table.traces))
            savings = {mix: DRAWS.savings(MATRIX, fixed["opus"], MIXES[mix]) for mix in MIXES}
            built[name] = {"mixes": mix_rows(DRAWS, savings.__getitem__, fixed, outcomes, MIXES),
                           "tasks": task_rows(DRAWS, fixed, outcomes, ["a", "b"])}  # fmt: skip
    return built


def every_policy() -> None:
    print("\n-- every policy's rows, with no line per policy --")
    cache: dict = {}
    built = evaluated_into(cache)
    cache["built"] = {"references": {"mixes": {"realistic": {"configs": {}}}}, **built}
    inputs = SimpleNamespace(cache=cache, matrix=MATRIX, draws=DRAWS)
    check(built["toy"]["mixes"]["realistic"]["99%"]["quality"]["value"] == 0.75
          and built["toy2"]["mixes"]["realistic"]["100%"]["quality"]["value"] == 0.5,
          "the toy policies are evaluated as P (75%) and Q (50%)")  # fmt: skip
    with mock.patch.object(target_met, "load_mixes", return_value=MIXES):
        result = target_met.part(inputs)
    rows = result["points"]
    check(list(rows) == ["toy", "toy2"] and result["margin"] == 0,
          "target met covers both policies, in the order they were built, at a margin of zero")  # fmt: skip
    toy = rows["toy"]["realistic"]
    check(toy["100%"]["reading"] == "falls back to A" and toy["99%"]["reading"] == "met" and toy["97%"]["reading"] == "met"
          and close(toy["99%"]["difference"]["value"], hand("P", 0.99)["value"])
          and close(toy["97%"]["difference"]["value"], hand("P", 0.97)["value"])
          and close(toy["99%"]["difference"]["low"], hand("P", 0.99)["low"]),
          "toy: falls back at 100%, meets 99% and 97% by the hand-worked differences")  # fmt: skip
    toy2 = rows["toy2"]["uniform"]
    check(toy2["100%"]["reading"] == "missed" and close(toy2["100%"]["difference"]["value"], -0.25)
          and toy2["99%"]["reading"] == "falls back to A" and toy2["97%"]["reading"] == "falls back to A",
          "toy2: misses at 100% (25 points under A), falls back at the others")  # fmt: skip
    check(result["counts"]["toy"] == {"points": 6, "routes": 4, "met": 4, "missed": 0, "falls_back": 2}
          and result["counts"]["toy2"] == {"points": 6, "routes": 2, "met": 0, "missed": 2, "falls_back": 4}
          and result["counts"]["all"] == {"points": 12, "routes": 6, "met": 4, "missed": 2, "falls_back": 6},
          "the counts: 6 of 12 points route, 4 met, 2 missed, and the 6 that fall back are in none of them")  # fmt: skip
    check(toy["99%"]["met_share"] == target_met.point_row(DRAWS, A, P, MIX, 0.99, False)["met_share"]
          and toy["99%"]["met_share"] == built["toy"]["mixes"]["realistic"]["99%"]["captured"]["met"],
          "a point's share of resamples meeting the target is the dev tables' own captured.met, the same comparison")  # fmt: skip
    # a policy added by one line is covered: a third section evaluated the same way appears without an edit here
    cache2: dict = {}
    third = evaluated_into(cache2, ("toy",))
    cache2["built"] = third
    with mock.patch.object(target_met, "load_mixes", return_value=MIXES):
        only = target_met.part(SimpleNamespace(cache=cache2, matrix=MATRIX, draws=DRAWS))
    check(list(only["points"]) == ["toy"], "a build with one policy has target met for that policy alone")
    # refusals
    cache["built"]["extra"] = {"mixes": {"realistic": {"99%": {"fell_back": False, "quality": {}}}}}
    check(refused(target_met.TargetMetError, lambda: target_met.part(inputs)),
          "known positive: a part holding policy rows nothing evaluated through operating_eval is refused, not left out")  # fmt: skip
    del cache["built"]["extra"]
    forget_evaluated(cache)
    check(refused(target_met.TargetMetError, lambda: target_met.part(inputs)),
          "after the evaluated points are forgotten, target met has no policy to read and is refused")  # fmt: skip
    plain = fixed_outcomes(MATRIX)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "operating_points.json"
        path.write_text(json.dumps(SECTIONS), encoding="utf-8")
        table = RouterScoreTable.build([ItemRouterScores(i, dict(TOY_SCORES["toy"]), 0.0) for i in ITEMS], list(ITEMS))
        outcomes = evaluate(MATRIX, plain["opus"], frozen_points("toy", path), router_traces(table.traces))
        plain_rows = mix_rows(
            DRAWS, lambda mix: DRAWS.savings(MATRIX, plain["opus"], MIXES[mix]), plain, outcomes, MIXES
        )
    check(
        set(plain_rows) == {"realistic", "uniform"},
        "rows over a plain dict of fixed outcomes (as the earlier checks build them) still build",
    )


# ---- the headline ------------------------------------------------------------------------------------------------

HAIKU_JEV, HAIKU_FREE, HAIKU_SONNET = (cascade_section(HAIKU, gate) for gate in (JEV_GATE, FREE_GATE, SONNET_GATE))
ORDER = [(mix, target) for mix in ("realistic", "uniform") for target in ("100%", "99%", "97%")]


def dev_table(shares: dict[str, list[float]]) -> dict:
    """A made-up dev table (the keys table v4 holds that item 24's rule reads): each cascade's six savings shares, the
    rest of the fifteen at zero (every point a fall back to A)."""
    keys = [k for k in final_headline.CASCADES]
    table: dict = {"cascades": {}}
    for key in keys:
        found = shares.get(key, [0.0] * 6)
        table["cascades"][key] = {"operating_points": {mix: {} for mix in ("realistic", "uniform")}}
        for (mix, target), share in zip(ORDER, found):
            table["cascades"][key]["operating_points"][mix][target] = {
                "fell_back": share == 0.0,
                "savings_vs_A": {"share": share, "per_1k": share * 10.0},
                "captured": {"met": 0.9},
            }
    return table


def cell(quality: float, cost: float, target: float, fell_back: bool = False) -> dict:
    """A policy row of the offline half's part, as `operating_eval.mix_rows` builds it, with the numbers the headline reads."""
    estimate = lambda v: {"value": v, "low": v - 0.01, "high": v + 0.01, "standard_error": 0.005}  # noqa: E731
    return {"threshold": None, "fell_back": fell_back, "target": target, "quality": estimate(quality),
            "cost_per_1k": estimate(cost), "savings_vs_A": {"per_1k": 10.0 - cost, "share": (10.0 - cost) / 10.0}}  # fmt: skip


def built_for(rows: dict[str, tuple[float, float, str]]) -> dict:
    """The parts built so far: A (quality .80, cost $10 under both mixes), and each given cascade with its realistic 99% point
    (quality, cost, target-met reading) and the other five as the same (met)."""
    references = {"mixes": {mix: {"configs": {"opus": {"quality": {"value": 0.80}, "cost_per_1k": {"value": 10.0}}}}
                            for mix in ("realistic", "uniform")}}  # fmt: skip
    built: dict = {"references": references}
    met: dict = {}
    for key, (quality, cost, reading) in rows.items():
        built[key] = {"mixes": {mix: {t: cell(quality, cost, 0.792, reading == "falls back to A") for t in ("100%", "99%", "97%")}
                                for mix in ("realistic", "uniform")}}  # fmt: skip
        difference = None if reading == "falls back to A" else {"value": quality - 0.792, "low": -0.02, "high": 0.03,
                                                                 "standard_error": 0.01}  # fmt: skip
        met[key] = {mix: {t: {"reading": reading, "difference": difference, "met_share": None if difference is None else 0.7}
                          for t in ("100%", "99%", "97%")} for mix in ("realistic", "uniform")}  # fmt: skip
    built["target_met"] = {"margin": 0, "points": met, "counts": {}}
    return built


def headline() -> None:
    print("\n-- the headline's rule: dev's table picks the cascade, and nothing on test moves it --")
    top = dev_table({HAIKU_JEV: [0.30, 0.38, 0.40, 0.0, 0.20, 0.30], HAIKU_FREE: [0.0] * 6, HAIKU_SONNET: [0.1] * 6})
    check(final_headline.best_cascade(top) == HAIKU_JEV,
          "item 24's rule on a made-up dev table: the cascade with the highest mean savings (a fall back counting 0) is haiku/Jev")  # fmt: skip
    check(final_headline.best_cascade(dev_table({HAIKU_JEV: [0.2] * 6, HAIKU_FREE: [0.2] * 6, HAIKU_SONNET: [0.2] * 6}))
          == HAIKU_FREE and GATE_COST_ORDER[0] == "free",
          "a tie among pure cascades goes to the cheaper gate: the free checks, as item 24 says")  # fmt: skip
    router_tie = dev_table(
        {cascade_section("rules", JEV_GATE): [0.5] * 6, cascade_section("trained", JEV_GATE): [0.5] * 6}
    )
    check(refused(final_headline.HeadlineError, lambda: final_headline.best_cascade(router_tie)),
          "a tie the rule leaves open (two router-started cascades with one gate) refuses: no choice is made by hand")  # fmt: skip
    re_selected = dev_table({HAIKU_FREE: [0.45] * 6, HAIKU_JEV: [0.30, 0.38, 0.40, 0.0, 0.20, 0.30]})
    check(final_headline.best_cascade(re_selected) == HAIKU_FREE,
          "a re-selected dev table moves the headline, by the rule")  # fmt: skip

    # test's table, with haiku/free doing better than haiku/Jev at the headline's own point
    built = built_for({HAIKU_JEV: (0.79, 6.2, "missed"), HAIKU_FREE: (0.85, 4.0, "met")})
    shown = final_headline.build(built, top)
    cascade = shown["headline"]
    check(shown["cascade"] == HAIKU_JEV and shown["label"] == CASCADES[HAIKU_JEV] and cascade["mix"] == "realistic"
          and cascade["target"] == "99%",
          "the headline stays on haiku/Jev at realistic 99%, though haiku/free does better on test")  # fmt: skip
    check(final_headline.build(built, re_selected)["cascade"] == HAIKU_FREE and
          final_headline.build(built, re_selected)["headline"]["quality"]["value"] == 0.85,
          "and a re-selected dev table moves it, to haiku/free's cell")  # fmt: skip
    check(cascade["reading"] == "not met" and cascade["quality"]["value"] == 0.79 and cascade["cost_per_1k"]["value"] == 6.2
          and cascade["A"] == {"quality": 0.80, "cost_per_1k": 10.0},
          "a headline that misses says not met, with its quality (79.0%) and cost ($6.20 per 1k), set against A's")  # fmt: skip
    met = final_headline.build(built_for({HAIKU_JEV: (0.85, 6.2, "met")}), top)["headline"]
    check(met["reading"] == "met" and met["difference"]["value"] > 0, "one that meets its target says met")
    fell = final_headline.build(built_for({HAIKU_JEV: (0.80, 10.0, "falls back to A")}), top)["headline"]
    check(fell["reading"] == "falls back to A" and fell["quality"]["value"] == 0.80 and fell["difference"] is None,
          "one that fell back says so, with A's own quality and cost, and is not called met")  # fmt: skip
    check(list(shown["beside"]) == ["realistic 97%", "uniform 99%", "uniform 97%"],
          "the 97% target and the uniform mix are shown beside it, never in its place")  # fmt: skip
    check(shown["dev"] == {"savings_share": 0.38, "met_share": 0.9},
          "dev's own savings and share of resamples meeting the target stand beside it, from dev's table")  # fmt: skip
    check(refused(final_headline.HeadlineError, lambda: final_headline.build(built_for({HAIKU_FREE: (0.85, 4.0, "met")}), top)),
          "a headline whose cascade was not built is refused")  # fmt: skip
    without = {k: v for k, v in built.items() if k != "target_met"}
    check(refused(final_headline.HeadlineError, lambda: final_headline.build(without, top)),
          "a headline built before target met is refused")  # fmt: skip


# ---- dev's flip rates -------------------------------------------------------------------------------------------


def stored(item: str, config: str, source: str, successes: tuple[int, ...]) -> list[dict]:
    return [
        {"item_id": item, "config": config, "source": source, "run": run, "success": s}
        for run, s in enumerate(successes)
    ]


def flips() -> None:
    print("\n-- dev's flip rates, which carry over to test --")
    # By hand: of the noise subset's pairs, x on haiku flips (1, 0, 1), y on haiku doesn't (1, 1, 1), x on opus flips (0, 1, 1)
    # in task t1, and z on haiku in task t2 doesn't flip (0, 0, 0). w isn't in the noise subset, v has only two runs.
    rows = (stored("x", "haiku", "t1", (1, 0, 1)) + stored("y", "haiku", "t1", (1, 1, 1)) + stored("x", "opus", "t1", (0, 1, 1))
            + stored("z", "haiku", "t2", (0, 0, 0)) + stored("w", "haiku", "t1", (1, 0, 1)) + stored("v", "haiku", "t1", (1, 0)))  # fmt: skip
    found = flip_rates.with_rates(flip_rates.counts(rows, {"x", "y", "z", "v"}, ["haiku", "opus"]))
    check(found["haiku"]["flips"] == 1 and found["haiku"]["pairs"] == 3 and close(found["haiku"]["rate"], 1 / 3)
          and found["haiku"]["tasks"] == {"t1": {"flips": 1, "pairs": 2, "rate": 0.5}, "t2": {"flips": 0, "pairs": 1, "rate": 0.0}},
          "by hand: haiku flips on 1 of its 3 complete pairs (one in task t1 of 2, none in t2), and an item outside the noise "
          "subset or with two runs is not a pair")  # fmt: skip
    check(found["opus"]["flips"] == 1 and found["opus"]["pairs"] == 1 and found["opus"]["rate"] == 1.0,
          "and opus flips on its one pair")  # fmt: skip
    check(refused(flip_rates.FlipRatesError, lambda: flip_rates.with_rates(flip_rates.counts(rows, {"y"}, ["haiku", "opus"]))),
          "a config with no pair of three runs is refused, not given a rate")  # fmt: skip
    over = flip_rates.words({"haiku": {"rate": 0.0667}, "sonnet": {"rate": 0.04}, "opus_low": {"rate": 0.05}})
    under = flip_rates.words({"sonnet": {"rate": 0.04}, "opus": {"rate": 0.0499}})
    check(over["over_limit"] == ["haiku", "opus_low"] and over["oracle_described_as"] == "optimistic upper bound"
          and under["over_limit"] == [] and under["oracle_described_as"] is None,
          "A6 item 2: at 5% or more (opus_low's exactly 5% counts) the oracle is an optimistic upper bound; under it, not")  # fmt: skip
    carried = flip_rates.carried()
    plan = (PROJECT_ROOT / "PLAN_v4.md").read_text(encoding="utf-8")
    line = next(line for line in plan.splitlines() if "flip rates are `haiku`" in line)
    written = re.findall(r"`(\w+)` (\d+)/(\d+) \((\d+\.\d)%\)", line)

    def agrees(held: dict) -> bool:
        return (len(written) == 4 and {c: (int(f), int(n)) for c, f, n, _ in written}
                == {c: (h["flips"], h["pairs"]) for c, h in held.items()}
                and all(pct == f"{100 * held[c]['rate']:.1f}" for c, _, _, pct in written))  # fmt: skip

    check(agrees(carried),
          "the committed dev file holds A6's numbers as the plan writes them: haiku 5/75, sonnet 3/75, opus_low 4/75, opus 3/75")  # fmt: skip
    check(all(sum(t["flips"] for t in h["tasks"].values()) == h["flips"] and sum(t["pairs"] for t in h["tasks"].values()) == h["pairs"]
              and all(close(t["rate"], t["flips"] / t["pairs"]) for t in h["tasks"].values()) for h in carried.values()),
          "each config's per-task counts add up to its total")  # fmt: skip
    part = flip_rates.part(None)
    check(part["over_limit"] == ["haiku", "opus_low"] and part["oracle_described_as"] == "optimistic upper bound"
          and part["oracle"] == "D: hindsight oracle (optimistic upper bound)" and part["limit"] == 0.05,
          "the part carries them and describes the test oracle as an optimistic upper bound, as A6 does the dev one")  # fmt: skip
    bumped = json.loads(json.dumps(carried))
    bumped["haiku"]["flips"] += 1
    check(not agrees(bumped), "known positive: one flip more in the file disagrees with the plan's numbers")


# ---- the tables step on the rehearsal's made-up world ---------------------------------------------------------------------


def leaky_stops(run: Run) -> StepResult:
    """A step made to show a refusal count before the tables: the known positive of the search."""
    run.say(f"{STOPS_LABEL}: {json.dumps(STOPS)}")
    return StepResult()


SECTIONS_BUILT = ["rules", "trained", HAIKU_FREE, HAIKU_JEV]
STOPS = {
    "haiku": {GRADE_MARKER: 1},
    "sonnet": {"refusal": 1},
    "opus_low": {},
    "opus": {},
}  # the made-up store's stops, by hand


def shown(text: str) -> bool:
    return STOPS_LABEL in text or '"refusal": 1' in text


def rehearsal() -> None:
    print("\n-- the tables step on the rehearsal's made-up world --")
    with rehearsed() as r:
        outcome, _, err = r.go()
        parts = r.tables["parts"]
        met = parts["target_met"]
        check(outcome.stopped_at is None, "the whole made-up run goes through")
        check(list(met["points"]) == SECTIONS_BUILT,
              "target met covers E, F, haiku/free and haiku/Jev, which no line of this ticket names: whatever the policies' parts built")  # fmt: skip
        agree, label_ok, counted = True, True, {"routes": 0, "met": 0, "missed": 0, "falls_back": 0}
        for section in SECTIONS_BUILT:
            for mix in ("realistic", "uniform"):
                for target in ("100%", "99%", "97%"):
                    row, found = parts[section]["mixes"][mix][target], met["points"][section][mix][target]
                    if row["fell_back"]:
                        label_ok &= found == {"reading": "falls back to A", "difference": None, "met_share": None}
                        counted["falls_back"] += 1
                        continue
                    # independent of this module: the policy's own row holds its quality, its target and the dev tables' own
                    # share of resamples meeting it, from another code path (the captured fraction's)
                    value = row["quality"]["value"] - row["target"]
                    agree &= close(found["difference"]["value"], value) and found["met_share"] == row["captured"]["met"]
                    agree &= found["reading"] == ("met" if value >= -1e-9 else "missed")
                    agree &= found["difference"]["low"] <= found["difference"]["high"]
                    counted["routes"] += 1
                    counted[found["reading"]] += 1
        check(agree, "each routed point's difference is its row's quality less its target, its share of resamples meeting it is "
                     "the row's captured.met, and its reading follows the sign")  # fmt: skip
        check(
            label_ok and counted["falls_back"] > 0,
            f"each of the {counted['falls_back']} points that fall back reads falls back to A, with no numbers",
        )
        check(met["counts"]["all"] == {"points": 24, **counted} and counted["routes"] + counted["falls_back"] == 24
              and counted["met"] + counted["missed"] == counted["routes"],
              f"the counts are the rows': {counted}; the points that fall back are in no count of met")  # fmt: skip
        head = parts["headline"]
        best = json.loads(TABLE_V4.read_text(encoding="utf-8"))["best_cascade"]["best"]
        check(head["cascade"] == best == HAIKU_JEV and head["headline"]["mix"] == "realistic"
              and head["headline"]["target"] == "99%"
              and head["headline"]["reading"] == {"met": "met", "missed": "not met"}[
                  met["points"][best]["realistic"]["99%"]["reading"]],
              "the headline is the cascade table v4 itself names as best (haiku/Jev), at realistic 99%, reading as its target met does")  # fmt: skip
        check(head["headline"]["quality"] == parts[best]["mixes"]["realistic"]["99%"]["quality"]
              and head["headline"]["cost_per_1k"] == parts[best]["mixes"]["realistic"]["99%"]["cost_per_1k"],
              "and it carries that point's quality and cost")  # fmt: skip
        before, after = r.log.split("tables: starts")
        record = (r.world.output / RECORD_FILE).read_text(encoding="utf-8")
        check(not shown(before) and not shown(record) and not shown(err) and "target met at a margin" not in before
              and "tables: headline" not in before,
              "refusal and truncation counts, target met and the headline appear in no output and no record before the tables step")  # fmt: skip
        stops_line = next(line for line in after.splitlines() if STOPS_LABEL in line)
        check(json.loads(stops_line.split(f"{STOPS_LABEL}: ", 1)[1]) == STOPS,
              "and the tables step reports each config's refusals and truncations: the planted refusal on sonnet and the grade's stop on haiku")  # fmt: skip
        check("tables: target met at a margin of zero" in after and "tables: headline, haiku/Jev at realistic 99%:" in after,
              "with the tables it says the target-met counts and the headline's reading")  # fmt: skip
    with rehearsed() as leaking:
        leaking.go((STEPS[0], Step("leak", "analysis.check_target_met:leaky_stops")))
        check(shown(leaking.log), "known positive: a step made to say a planted refusal count before the tables is found by the same search")  # fmt: skip


def main() -> int:
    worked()
    falling_back()
    every_policy()
    headline()
    flips()
    rehearsal()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
