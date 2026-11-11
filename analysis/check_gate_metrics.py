"""Check the gate metrics, the calibration bins and the latency estimate (PLAN_v4.md §9; Amendment A17, items 16
and 21).

Run from the project root:  uv run python -m analysis.check_gate_metrics

On synthetic data only, with every expected value worked out by hand: no results/ folder, no Docker, no Jev or
Anthropic call, and no benchmark answer. The paid scores are made-up numbers plugged into the gate-score table.

- the rates and raw counts per gate, config and task at a cutoff: a paid gate on the answers that pass the free checks
  and the free checks on all of them, a refused call rejected, the bad calls refused;
- within-task concordance per paid gate and config, with intervals from the master resamples sliced to the gated
  answers, and the train signal check against 0.5;
- calibration bins: equal-count, tied scores never split, hand-drawn resamples;
- the latency estimate: pilot medians summed with the router's and the gates' along each trace, the mix-weighted
  median, for a fixed config, a router and a cascade.
"""

import json
import math
import re
import statistics
import sys
from collections.abc import Callable

import numpy as np

from analysis.bootstrap import Bootstrap
from analysis.check_simulator import synthetic
from analysis.checking import refused
from analysis.gate_metrics import (
    REFUSED_SCORE,
    GateMetricsError,
    GateRates,
    bin_items,
    calibration_bins,
    gate_calibration,
    gate_concordance,
    gate_rates,
    gate_scores,
    signal_check,
    subset_draws,
)
from analysis.gate_table import GateRow, GateScoreTable, PaidScore, cascade_trace
from analysis.latency_estimate import (
    E_MEDIAN,
    F_MEDIANS,
    PILOT_MEDIANS,
    GateMedians,
    LatencyError,
    call_median,
    cascade_gate_calls,
    code_check_median,
    estimate_latency,
    jev_call_medians,
    pilot_medians,
    router_medians,
    trace_seconds,
    weighted_median,
)
from analysis.plan import PLAN, amendment_item
from analysis.router_names import FREE_GATE, G_PLUS, JEV_GATE, SONNET_GATE, E, F, G
from analysis.simulator import MixError, Trace, TraceError
from harness.client import PROJECT_ROOT
from routers.gates import CODE

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def close(got: float | None, expected: float | None, tolerance: float = 1e-9) -> bool:
    if got is None or expected is None:
        return got is expected
    return abs(got - expected) <= tolerance


# ---- the hand-worked gate: ten haiku answers and ten opus answers --------------------------------------------------
#
# Task t1 holds a1..a6, task t2 b1..b4. Haiku's success and the Jev gate's score (n: no entry, r: refused call):
#
#   a1  right  .9      a2  right  .6      a3  wrong  .7      a4  wrong  .2      a5  right  .3
#   a6  wrong, failed the stop check      b1  right  .8      b2  wrong  .8      b3  right  refused
#   b4  right, failed the code check
#
# Opus passes every free check and is right everywhere but a4. Opus is never gated (no paid entry on its rows).

ITEMS = {
    "a1": ("t1", 1, 0.9), "a2": ("t1", 1, 0.6), "a3": ("t1", 0, 0.7), "a4": ("t1", 0, 0.2), "a5": ("t1", 1, 0.3),
    "a6": ("t1", 0, None), "b1": ("t2", 1, 0.8), "b2": ("t2", 0, 0.8), "b3": ("t2", 1, None), "b4": ("t2", 1, None),
}  # fmt: skip
FAILED = {"a6": "stop", "b4": "code"}
MATRIX = synthetic({
    item: (task, {"haiku": (0.001, right), "opus": (0.01, 0 if item == "a4" else 1)})
    for item, (task, right, _) in ITEMS.items()
})  # fmt: skip
ANSWERS = [(item, config) for item in ITEMS for config in ("haiku", "opus")]


def paid(score: float | None) -> PaidScore:
    return PaidScore(score, 0.0001, score is None)


def rows_of(scores: dict[str, float | None], gate: str = JEV_GATE) -> list[GateRow]:
    """Haiku's rows with `scores` as `gate`'s entries (an item left out has none), then opus's rows, never gated."""
    rows = []
    for item in ITEMS:
        failed = FAILED.get(item)
        entry = paid(scores[item]) if item in scores else None
        rows.append(GateRow(item, "haiku", failed is None, failed, entry if gate == JEV_GATE else None,
                            entry if gate == SONNET_GATE else None))  # fmt: skip
        rows.append(GateRow(item, "opus", True, None))
    return rows


JEV_SCORES = {item: score for item, (_, _, score) in ITEMS.items() if item not in FAILED}
TABLE = GateScoreTable.build(rows_of(JEV_SCORES), ANSWERS)


def rates_of(**counts: int) -> dict[str, int]:
    return counts


def main() -> int:
    print("-- the rates and raw counts at a cutoff --")
    jev = gate_rates(TABLE, MATRIX, JEV_GATE, 0.5)
    t1 = jev[("haiku", "t1")]
    # t1's gated answers a1..a5: wrong a3 (accepted at .7) and a4 (rejected); right a1, a2 (accepted) and a5 (.3, rejected).
    check(t1 == GateRates(answers=6, gated=5, right=3, wrong=2, wrong_accepted=1, right_rejected=1,
                          false_accept_rate=0.5, false_escalate_rate=1 / 3),
          f"the Jev gate on haiku's t1 answers at .5: {t1}")  # fmt: skip
    t2 = jev[("haiku", "t2")]
    # t2's gated answers b1..b3: wrong b2 (.8, accepted); right b1 (accepted) and b3 (refused, so rejected).
    check(t2 == GateRates(answers=4, gated=3, right=2, wrong=1, wrong_accepted=1, right_rejected=1,
                          false_accept_rate=1.0, false_escalate_rate=0.5),
          f"the Jev gate on haiku's t2 answers at .5, a refused call rejected: {t2}")  # fmt: skip
    pooled = jev[("haiku", None)]
    check(pooled == GateRates(answers=10, gated=8, right=5, wrong=3, wrong_accepted=2, right_rejected=2,
                              false_accept_rate=2 / 3, false_escalate_rate=0.4),
          f"the Jev gate on all of haiku's answers at .5: {pooled}")  # fmt: skip
    check(
        close(jev[("haiku", None)].false_accept_rate, 2 / 3) and close(jev[("haiku", "t2")].false_accept_rate, 1.0),
        "the false accept rate is wrong answers accepted over wrong answers",
    )
    check(("opus", None) not in jev and ("opus", "t1") not in jev, "a config the gate never judged has no row")
    cutoff_at_score = gate_rates(TABLE, MATRIX, JEV_GATE, 0.7)[("haiku", "t1")]
    # Accepting at a score at or above the cutoff: .7 itself is accepted, so a3 stays a false accept and a2 (.6) is
    # a false escalate along with a5.
    check(cutoff_at_score.wrong_accepted == 1 and cutoff_at_score.right_rejected == 2,
          f"a gate score equal to the cutoff is accepted: {cutoff_at_score}")  # fmt: skip

    print("\n-- the free checks' row, on all answers --")
    free = gate_rates(TABLE, MATRIX, FREE_GATE, None)
    expected_free = {
        # haiku: a6 (wrong) and b4 (right) fail a free check; a3, a4 and b2 are wrong answers that pass them.
        ("haiku", None): GateRates(10, 10, 6, 4, 3, 1, 3 / 4, 1 / 6),
        ("haiku", "t1"): GateRates(6, 6, 3, 3, 2, 0, 2 / 3, 0.0),
        ("haiku", "t2"): GateRates(4, 4, 3, 1, 1, 1, 1.0, 1 / 3),
        # opus passes every free check and is wrong only on a4; t2 has no wrong answer, so no false accept rate.
        ("opus", None): GateRates(10, 10, 9, 1, 1, 0, 1.0, 0.0),
        ("opus", "t1"): GateRates(6, 6, 5, 1, 1, 0, 1.0, 0.0),
        ("opus", "t2"): GateRates(4, 4, 4, 0, 0, 0, None, 0.0),
    }
    check(free == expected_free, f"the free checks' rows over all answers, one per config and task ({len(free)} rows)")
    check(
        list(free) == list(expected_free), "a config's pooled row comes first, then its tasks, configs in table order"
    )
    check(gate_rates(TABLE, MATRIX, SONNET_GATE, 0.5) == {}, "a paid gate with no entries at all has no rows")

    print("\n-- rates refusals, each fed its bad case --")
    refusals: dict[str, Callable[[], object]] = {
        "a cutoff given to the free checks": lambda: gate_rates(TABLE, MATRIX, FREE_GATE, 0.5),
        "no cutoff for a paid gate": lambda: gate_rates(TABLE, MATRIX, JEV_GATE, None),
        "a cutoff above 1": lambda: gate_rates(TABLE, MATRIX, JEV_GATE, 1.5),
        "a cutoff of NaN": lambda: gate_rates(TABLE, MATRIX, JEV_GATE, math.nan),
        "an unknown gate": lambda: gate_rates(TABLE, MATRIX, "oracle", 0.5),
        "an answer the matrix has no success for": lambda: gate_rates(
            TABLE, synthetic({item: (task, {"haiku": (0.001, right)}) for item, (task, right, _) in ITEMS.items()}),
            FREE_GATE, None),
        "an answer whose item the matrix lacks": lambda: gate_rates(
            TABLE, synthetic({"a1": ("t1", {"haiku": (0.001, 1), "opus": (0.01, 1)})}), JEV_GATE, 0.5),
        "a success that is neither 1 nor 0": lambda: gate_rates(
            TABLE, synthetic({item: (task, {"haiku": (0.001, 0.5), "opus": (0.01, 1)})
                              for item, (task, _, _) in ITEMS.items()}), JEV_GATE, 0.5),
        "a gated answer with no entry beside gated answers that have one": lambda: gate_rates(
            GateScoreTable.build(rows_of({k: v for k, v in JEV_SCORES.items() if k != "b1"}), ANSWERS),
            MATRIX, JEV_GATE, 0.5),
    }  # fmt: skip
    for what, call in refusals.items():
        check(refused(GateMetricsError, call), f"refused: {what}")

    print("\n-- within-task concordance, with intervals from the master resamples sliced to the gated answers --")
    ids = tuple(ITEMS)
    # Four hand-drawn resamples of the ten items, columns in the order a1..a6, b1..b4: all once; a1 twice and a2 not
    # at all; a3 and a4 (t1's wrong answers) not at all; b2 (t2's only wrong answer) not at all.
    draws = Bootstrap(ids, tuple(task for task, _, _ in ITEMS.values()), np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    ]))  # fmt: skip
    scores = gate_scores(TABLE, JEV_GATE, "haiku")
    check(scores == {"a1": 0.9, "a2": 0.6, "a3": 0.7, "a4": 0.2, "a5": 0.3, "b1": 0.8, "b2": 0.8, "b3": REFUSED_SCORE},
          f"a paid gate's scores are the gated answers', a refused call below every gate score: {scores}")  # fmt: skip
    check(gate_scores(TABLE, JEV_GATE, "haiku", refused=None) == {i: s for i, s in scores.items() if i != "b3"},
          "with no refused value, a refused call has no score")  # fmt: skip
    check(gate_scores(TABLE, JEV_GATE, "opus") == {}, "a config the gate judged no answer of has no scores")
    sub = subset_draws(draws, ["a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3"])
    check(sub.item_ids == ("a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3") and sub.tasks == ("t1",) * 5 + ("t2",) * 3
          and np.array_equal(sub.counts, draws.counts[:, [0, 1, 2, 3, 4, 6, 7, 8]]),
          "the gated answers' resamples are the master's columns for those items, the same draws")  # fmt: skip
    reading = gate_concordance(draws, TABLE, MATRIX, JEV_GATE, "haiku")
    # t1: pairs (right, wrong) a1 > a3, a1 > a4, a2 < a3, a2 > a4, a5 < a3, a5 > a4 -> 4 of 6; t2: b1 ties b2 (half) and
    # the refused b3 is below b2 -> 0.5 of 2.
    check(close(reading.pooled.value, 4.5 / 8) and close(reading.per_task["t1"].value, 4 / 6)
          and close(reading.per_task["t2"].value, 0.25) and reading.size == {None: 8, "t1": 6, "t2": 2},
          f"the Jev gate's concordance on haiku, pooled and per task, ties half, pairs counted: {reading.pooled.value}")  # fmt: skip
    # Resample values, pooled: 9/16; a1 twice and a2 gone -> t1 5 of 6 pairs, so (5 + .5) / 8; no t1 wrong answer -> t2
    # alone, .25; no t2 wrong answer -> t1 alone, 2/3. Per task, t1: 2/3, 5/6, n/a, 2/3; t2: .25, .25, .25, n/a.
    pooled_low, pooled_high = np.percentile([9 / 16, 11 / 16, 1 / 4, 2 / 3], [2.5, 97.5])
    check(close(reading.pooled.low, pooled_low) and close(reading.pooled.high, pooled_high)
          and reading.pooled.undefined == 0, "the pooled interval is the percentile interval of the four resamples")  # fmt: skip
    t1_high = 2 / 3 + 0.95 * (
        5 / 6 - 2 / 3
    )  # three defined values 2/3, 2/3, 5/6: the 97.5th percentile is at rank 1.95
    check(close(reading.per_task["t1"].low, 2 / 3) and close(reading.per_task["t1"].high, t1_high)
          and reading.per_task["t1"].undefined == 0.25 and reading.per_task["t2"].undefined == 0.25
          and close(reading.per_task["t2"].low, 0.25) and close(reading.per_task["t2"].high, 0.25),
          "a resample with no pair in a task is left out of that task's interval and counted undefined")  # fmt: skip

    print("\n-- the train signal check: concordance per config against 0.5 --")
    sonnet_scores = {"a1": 0.8, "a2": 0.8, "a5": 0.8, "a3": 0.2, "a4": 0.2, "b1": 0.5, "b2": 0.5, "b3": 0.5}
    both = GateScoreTable.build(
        [GateRow(r.item_id, r.config, r.free_passed, r.failed_first, r.jev,
                 paid(sonnet_scores[r.item_id]) if r.config == "haiku" and r.item_id in sonnet_scores else None)
         for r in rows_of(JEV_SCORES)],
        ANSWERS)  # fmt: skip
    # The code check applies to t2's answers only (as MBPP+'s prompts alone carry an assert line): it passes b1, b2 and
    # b3, and fails b4, whose answer was right.
    code_records = [{"item_id": i, "config": "haiku", "code_applied": i.startswith("b"), "code_passed": i != "b4"}
                    for i in ITEMS]  # fmt: skip
    found = signal_check(draws, MATRIX, both, code_records, ["haiku", "opus"])
    check([(c.gate, c.config) for c in found] == [(JEV_GATE, "haiku"), (SONNET_GATE, "haiku"), (CODE, "haiku")],
          f"one reading per gate and config the gate judged, opus (never gated) left out: {[(c.gate, c.config) for c in found]}")  # fmt: skip
    jev_read, sonnet_read, code_read = found
    check(close(jev_read.reading.pooled.value, 4.5 / 8) and jev_read.above_half and not jev_read.interval_above_half,
          "the Jev gate: 9/16 is above 0.5 but its interval, from .27, is not")  # fmt: skip
    # Sonnet: t1 separates perfectly (6 of 6 pairs), t2 ties everywhere (2 pairs at half): (6 + 1) / 8.
    check(close(sonnet_read.reading.pooled.value, 7 / 8) and close(sonnet_read.reading.per_task["t1"].value, 1.0)
          and close(sonnet_read.reading.per_task["t2"].value, 0.5) and sonnet_read.above_half,
          "the Sonnet gate: whole-number-style ties count half")  # fmt: skip
    # The code check scores 1 for a pass and 0 for a fail on the answers it applies to: b1 and b3 tie b2, b4 is below it.
    check(close(code_read.reading.pooled.value, 1 / 3) and not code_read.above_half
          and code_read.reading.size == {None: 3, "t2": 3}
          and close(code_read.reading.pooled.low, 1 / 3) and close(code_read.reading.pooled.high, 1 / 3)
          and code_read.reading.pooled.undefined == 0.25,
          "the code check on the answers it applies to: 1/3, below 0.5, a resample with no wrong answer undefined")  # fmt: skip

    print("\n-- calibration bins: equal count, tied scores never split --")
    # Twelve items g0..g11 in score order, in two tasks (odd ones in t2): the bin each lands in is worked out by hand.
    cal_scores = [0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.5, 0.6, 0.7, 0.7, 0.9, 0.9]
    cal_success = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1]
    names = [f"g{i}" for i in range(12)]
    by_name = dict(zip(names, cal_scores))
    # Four bins of 3 would put the four .3s in two bins, so the .3s stay together: 6, then .5 to .7 (4), then .9s (2).
    check(bin_items(by_name, 4) == [names[0:6], names[6:10], names[10:12]],
          "bins=4 over 12 items: ties kept whole, so bins of 6, 4 and 2 and none split")  # fmt: skip
    check(bin_items(by_name, 10) == [names[0:2], names[2:6], names[6:8], names[8:10], names[10:12]],
          "bins=10 over 12 items: five bins of 2, 4, 2, 2 and 2, fewer than ten")  # fmt: skip
    seven = {f"s{i}": i / 10 for i in range(7)}
    check([len(b) for b in bin_items(seven, 3)] == [3, 2, 2] and [len(b) for b in bin_items(seven, 7)] == [1] * 7
          and [len(b) for b in bin_items({f"d{i}": i / 20 for i in range(20)}, 10)] == [2] * 10,
          "with no ties the bins are as equal as the count allows: 7 into 3 is 3, 2, 2; 20 into 10 is ten pairs")  # fmt: skip
    check(bin_items({"x": 0.5, "y": 0.5, "z": 0.5}, 10) == [["x", "y", "z"]], "all scores tied: one bin")
    shuffled = dict(reversed(list(by_name.items())))
    check(bin_items(shuffled, 4) == bin_items(by_name, 4), "the bins don't depend on the order the scores come in")

    cal_matrix = synthetic({n: ("t2" if i % 2 else "t1", {"haiku": (0.001, y)})
                            for i, (n, y) in enumerate(zip(names, cal_success))})  # fmt: skip
    columns = list(reversed(range(12)))  # the resamples list the items g11 .. g0, not in score order
    in_score_order = np.array([
        [1] * 12,
        [1, 1, 3, 0, 1, 1, 0, 2, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 0],
        [1] * 12,
    ])  # fmt: skip
    cal_draws = Bootstrap(tuple(names[i] for i in columns), tuple(cal_matrix.items[names[i]].task for i in columns),
                          in_score_order[:, columns])  # fmt: skip
    bins = calibration_bins(cal_draws, cal_matrix, by_name, "haiku", 4)
    check([b.size for b in bins] == [6, 4, 2] and [b.first for b in bins] == [0.1, 0.5, 0.9]
          and [b.last for b in bins] == [0.3, 0.7, 0.9], f"the bins' sizes and score ranges: {[(b.size, b.first, b.last) for b in bins]}")  # fmt: skip
    check(all(close(b.mean_score, m) for b, m in zip(bins, [1.5 / 6, 2.5 / 4, 0.9]))
          and all(close(b.success_rate, r) for b, r in zip(bins, [2 / 6, 3 / 4, 1.0])),
          "each bin's mean gate score and success rate: .25, 1/3; .625, .75; .9, 1")  # fmt: skip
    # Per resample (bins 1, 2, 3): R0 1/3, .75, 1; R1 4/7 (g2 three times, g3 not at all), .5, n/a (neither drawn);
    # R2 n/a (none drawn), .75, 1 (g10 twice); R3 as R0.
    expected = [
        np.percentile([1 / 3, 4 / 7, 1 / 3], [2.5, 97.5]),
        np.percentile([0.75, 0.5, 0.75, 0.75], [2.5, 97.5]),
        np.percentile([1.0, 1.0, 1.0], [2.5, 97.5]),
    ]
    check(all(close(b.low, e[0]) and close(b.high, e[1]) for b, e in zip(bins, expected))
          and [b.undefined for b in bins] == [0.25, 0.0, 0.25],
          "each bin's interval is the percentile interval of its defined resamples, the rest counted undefined")  # fmt: skip

    # The Jev gate on haiku (ten answers, eight gated, b3 refused): seven scored answers a1..a5, b1, b2. Ten bins over
    # seven scores (.9 .6 .7 .2 .3 .8 .8, the .8s tied) close after .2, .3, .6, .7, then the pair of .8s, then .9.
    jev_bins = gate_calibration(draws, TABLE, MATRIX, JEV_GATE, "haiku")
    check([(b.size, b.first) for b in jev_bins] == [(1, 0.2), (1, 0.3), (1, 0.6), (1, 0.7), (2, 0.8), (1, 0.9)]
          and [b.success_rate for b in jev_bins] == [0.0, 1.0, 1.0, 0.0, 0.5, 1.0],
          "a paid gate's calibration leaves a refused call out and keeps the tied .8s (one right, one wrong) together")  # fmt: skip
    cal_refusals: dict[str, Callable[[], object]] = {
        "no scores": lambda: bin_items({}, 10),
        "fewer than one bin": lambda: bin_items(by_name, 0),
        "a score of NaN": lambda: bin_items({"x": math.nan}, 10),
        "a score that is not a number": lambda: bin_items({"x": "0.5"}, 10),  # type: ignore[dict-item]
        "an item the resamples lack": lambda: calibration_bins(cal_draws, cal_matrix, {"zz": 0.5}, "haiku"),
        "an answer the matrix lacks": lambda: calibration_bins(cal_draws, cal_matrix, by_name, "opus"),
    }  # fmt: skip
    for what, call in cal_refusals.items():
        check(refused(GateMetricsError, call), f"refused: {what}")

    print("\n-- the latency estimate: the mix-weighted median --")
    check(weighted_median({"p": 1.0, "q": 2.0, "r": 3.0}, {"p": 0.1, "q": 0.2, "r": 0.7}) == 3.0,
          "a weighted median is the value where the weight reaches half")  # fmt: skip
    check(close(weighted_median({"p": 1.0, "q": 2.0, "r": 3.0}, {"p": 0.2, "q": 0.3, "r": 0.5}), 2.5),
          "weight reaching exactly half at a value takes the midpoint with the next, as numpy's median does")  # fmt: skip
    four = {"p": 4.0, "q": 1.0, "r": 3.0, "s": 2.0}
    check(close(weighted_median(four, dict.fromkeys(four, 0.25)), statistics.median(four.values())),
          "equal weights give the ordinary median, whatever order the values come in")  # fmt: skip
    check(weighted_median({"p": 5.0}, {"p": 1.0}) == 5.0, "one value is its own median")
    median_refusals: dict[str, Callable[[], object]] = {
        "no values": lambda: weighted_median({}, {}),
        "weights for other items": lambda: weighted_median({"p": 1.0}, {"q": 1.0}),
        "a weight of 0": lambda: weighted_median({"p": 1.0, "q": 2.0}, {"p": 1.0, "q": 0.0}),
        "a value of NaN": lambda: weighted_median({"p": math.nan}, {"p": 1.0}),
    }  # fmt: skip
    for what, call in median_refusals.items():
        check(refused(LatencyError, call), f"refused: {what}")

    print("\n-- the pilot medians are A7 item 1's table --")
    a7 = amendment_item(PLAN.read_text(), "A7", 1)
    plan_rows = {m.group(1): m.groups()[1:] for m in re.finditer(
        r"^ *\| (IFEval|Banking77|MMLU-Pro|MBPP\+) \| ([\d.]+) \([\d.]+\) \| ([\d.]+) \([\d.]+\) \| ([\d.]+) \([\d.]+\) \| "
        r"([\d.]+) \([\d.]+\) \|$", a7, re.MULTILINE)}  # fmt: skip
    task_names = {"IFEval": "ifeval", "Banking77": "banking77", "MMLU-Pro": "mmlu_pro", "MBPP+": "mbppplus"}
    configs = ("haiku", "sonnet", "opus_low", "opus")
    from_plan = {task_names[n]: dict(zip(configs, map(float, v))) for n, v in plan_rows.items()}
    check(len(from_plan) == 4 and PILOT_MEDIANS == from_plan,
          f"PILOT_MEDIANS holds the medians A7 item 1's latency table gives, four tasks by four configs: {from_plan}")  # fmt: skip

    print("\n-- the sum along a trace, and the estimate for a fixed config, a router and a cascade --")
    check(close(trace_seconds("banking77", ("haiku", "opus")), 2.0 + 2.2) and close(trace_seconds("mbppplus", ("opus",)), 3.4),
          "a trace sums the pilot median of the task for each config called")  # fmt: skip
    check(close(trace_seconds("ifeval", ("sonnet",), router=0.5, gate_calls=(0.25, 0.125)), 10.0 + 0.5 + 0.25 + 0.125),
          "the router's median and each gate call's median are added")  # fmt: skip
    check(close(trace_seconds("mbppplus", ("haiku", "opus"), router={"mbppplus": 0.5, "ifeval": 9.0},
                              gate_calls=({"mbppplus": 0.25},)), 1.6 + 3.4 + 0.5 + 0.25),
          "a median can be given per task")  # fmt: skip
    LAT_ITEMS = {
        "x1": "banking77",
        "x2": "banking77",
        "x3": "banking77",
        "y1": "mbppplus",
        "y2": "mbppplus",
        "y3": "mbppplus",
    }
    lat = synthetic({i: (t, {"haiku": (0.001, 1), "opus": (0.01, 1)}) for i, t in LAT_ITEMS.items()})
    uniform, heavy = {"banking77": 0.5, "mbppplus": 0.5}, {"banking77": 0.2, "mbppplus": 0.8}
    all_opus = [Trace(i, ("opus",)) for i in LAT_ITEMS]
    # A: banking77 takes 2.2 s and MBPP+ 3.4 s. Under the uniform mix each task holds half the weight, exactly half is
    # reached at 2.2, so the estimate is the midpoint with 3.4. Under the heavy mix MBPP+ outweighs it: 3.4.
    check(close(estimate_latency(lat, all_opus, uniform), 2.8) and close(estimate_latency(lat, all_opus, heavy), 3.4),
          "A (always opus): 2.8 s under the uniform mix, 3.4 s under the mix that weighs MBPP+ four to one")  # fmt: skip
    check(close(estimate_latency(lat, [Trace(i, ("haiku",)) for i in LAT_ITEMS], uniform), (1.6 + 2.0) / 2),
          "B (always haiku): the midpoint of 1.6 and 2.0")  # fmt: skip
    # A router (F) sends x1, x3 and y3 to haiku, x2, y1 and y2 to opus, and costs its per-task median on every item.
    f_medians = {"banking77": 0.0182, "mbppplus": 0.0288}
    routed = [Trace("x1", ("haiku",)), Trace("x2", ("opus",)), Trace("x3", ("haiku",)),
              Trace("y1", ("opus",)), Trace("y2", ("opus",)), Trace("y3", ("haiku",))]  # fmt: skip
    # Sums: x1, x3 2.0182; x2 2.2182; y1, y2 3.4288; y3 1.6288. Heavy weights: y .8/3 each, x .2/3 each. Sorted: y3
    # .2667, x1 and x3 .1333 (.4), x2 .0667 (.4667), then y1 and y2 reach .5333 and 1.
    check(close(estimate_latency(lat, routed, heavy, router=f_medians), 3.4288),
          "a router's per-task median is added to every item's sum, escalated or not")  # fmt: skip
    check(close(estimate_latency(lat, routed, uniform, router=f_medians), (2.0182 + 2.2182) / 2),
          "the same router under the uniform mix: half the weight is reached at the third of six sums")  # fmt: skip

    print("\n-- the pilot medians from pilot rows: the formats the collection run uses, and no other --")
    pilot_rows: list[dict[str, object]] = []
    for task, per_config in PILOT_MEDIANS.items():
        structured = task in (
            "banking77",
            "mmlu_pro",
        )  # A7 item 1: these two ran in both formats; the table is structured
        for config, value in per_config.items():
            pilot_rows += [{"source": task, "format": "structured" if structured else "free", "config": config,
                            "latency_s": v} for v in (value - 1, value, value + 5)]  # fmt: skip
            if structured:  # the free-text rows of the same task, an hour slower: a median over both would move
                pilot_rows += [{"source": task, "format": "free", "config": config, "latency_s": 100.0}] * 3
    pilot_rows += [{"source": "cruxeval", "format": "free", "config": "haiku", "latency_s": 7.0},
                   {"source": "banking77", "format": "structured", "config": "haiku", "latency_s": None}]  # fmt: skip
    check(pilot_medians(pilot_rows) == PILOT_MEDIANS,
          "pilot rows give the plan's medians: structured for Banking77 and MMLU-Pro, free for IFEval and MBPP+, "
          "other tasks and rows with no latency (batch rows) left out")  # fmt: skip
    check(
        refused(LatencyError, lambda: pilot_medians([r for r in pilot_rows if r["config"] != "opus"])),
        "refused: pilot rows with no call for a config of a task",
    )

    print("\n-- the routers' medians and the gates' --")
    plan_text = PLAN.read_text()
    e_us = re.search(r"median ([\d.]+) us, p90", plan_text)
    f_tasks = re.search(r"Per task \(median, p90 in ms\): banking77 ([\d.]+), [\d.]+; ifeval ([\d.]+), [\d.]+; "
                        r"mbppplus ([\d.]+), [\d.]+; mmlu_pro ([\d.]+), ", plan_text)  # fmt: skip
    check(e_us is not None and close(E_MEDIAN, float(e_us.group(1)) * 1e-6, 1e-15),
          f"E's median is the plan's, 4.1 microseconds: {E_MEDIAN}")  # fmt: skip
    check(f_tasks is not None and all(close(F_MEDIANS[t], float(v) / 1000, 1e-12) for t, v in zip(
              ("banking77", "ifeval", "mbppplus", "mmlu_pro"), f_tasks.groups())),
          f"F's per-task medians are the plan's, in milliseconds there: {F_MEDIANS}")  # fmt: skip
    with_v3 = router_medians()
    v3 = json.loads((PROJECT_ROOT / "figures" / "table_v3.json").read_text())["latency"]
    check(set(with_v3) == {E, F, G, G_PLUS} and with_v3[E] == E_MEDIAN and with_v3[F] == F_MEDIANS
          and all(with_v3[r] == {t: v["median"] for t, v in v3[r]["per_task"].items()} for r in (G, G_PLUS)),
          "router_medians holds E's, F's and G's and G+'s per-task medians from table v3's stored dev calls")  # fmt: skip
    check(call_median([9.0, 1.0, 3.0, 2.0]) == 2.0 and call_median([9.0, 1.0, 3.0, 2.0], skip_first=False) == 2.5,
          "a stored-call median leaves the first call out (the Sonnet latency sample's rule)")  # fmt: skip
    check(refused(LatencyError, lambda: call_median([1.0])), "refused: a median of the first call alone")
    jev_calls = [
        {"item_id": "x1", "seconds": 9.0, "new_connection": True, "pass": 1},
        {"item_id": "x2", "seconds": 0.4, "new_connection": False, "pass": 1},
        {"item_id": "x3", "seconds": 0.6, "new_connection": False, "pass": 1},
        {"item_id": "y1", "seconds": 0.8, "new_connection": False, "pass": 1},
        {"item_id": "y2", "seconds": 5.0, "new_connection": True, "pass": 1},
        {"item_id": "y3", "seconds": 1.0, "new_connection": False, "pass": 1},
        {"item_id": "x2", "seconds": 99.0, "new_connection": False, "pass": 2},
    ]  # fmt: skip
    check(jev_call_medians(jev_calls, LAT_ITEMS) == {"banking77": 0.5, "mbppplus": 0.9},
          "the Jev gate's median per task leaves out the first call, a reconnect and the noise pass")  # fmt: skip
    code_records_timed = [
        {"item_id": "y1", "config": "haiku", "code_applied": True, "code_duration_s": 0.2},
        {"item_id": "y2", "config": "haiku", "code_applied": True, "code_duration_s": 0.4},
        {"item_id": "y3", "config": "haiku", "code_applied": True, "code_duration_s": 0.3},
        {"item_id": "x1", "config": "haiku", "code_applied": False, "code_duration_s": None},
    ]  # fmt: skip
    check(
        close(code_check_median(code_records_timed), 0.3), "the code check's median is over the answers it applied to"
    )
    check(refused(LatencyError, lambda: code_check_median(code_records_timed[3:])),
          "refused: a code check median with no answer it applied to")  # fmt: skip

    print("\n-- cascades: the free checks' and gates' calls on top of the configs --")
    items6 = list(LAT_ITEMS)
    cascade_rows = [
        GateRow("x1", "haiku", True, None, paid(0.9)), GateRow("x2", "haiku", True, None, paid(0.1)),
        GateRow("x3", "haiku", False, "stop"), GateRow("y1", "haiku", True, None, paid(0.9)),
        GateRow("y2", "haiku", True, None, paid(0.9)), GateRow("y3", "haiku", False, "code"),
    ]  # fmt: skip
    cascade_table = GateScoreTable.build(cascade_rows, [(i, "haiku") for i in items6])
    medians = GateMedians(code_check=0.3, jev=0.5, sonnet=9.0)
    code_items = {"y1", "y2", "y3"}

    def calls(item: str, start: str, gate: str, cutoff: float | None) -> tuple:
        return cascade_gate_calls(cascade_table, item, start, gate, cutoff, medians, code_items)

    check(calls("x1", "haiku", JEV_GATE, 0.5) == (0.5,) and calls("y1", "haiku", JEV_GATE, 0.5) == (0.3, 0.5),
          "a Jev cascade calls the Jev gate, after the code check where the code check applies")  # fmt: skip
    check(calls("x3", "haiku", JEV_GATE, 0.5) == () and calls("y3", "haiku", JEV_GATE, 0.5) == (0.3,),
          "an answer a free check rejected pays no paid gate call, and the code check still ran where it applies")  # fmt: skip
    check(calls("y1", "haiku", SONNET_GATE, 0.5) == (0.3, 9.0) and calls("x1", "haiku", FREE_GATE, None) == ()
          and calls("y1", "haiku", FREE_GATE, None) == (0.3,),
          "the Sonnet gate's median, and the free-checks cascade's code check alone")  # fmt: skip
    check(calls("y1", "opus", JEV_GATE, 0.5) == (), "an item started at opus gets no gate and no check")
    for what, call in {
        "a cutoff for the free checks": lambda: calls("x1", "haiku", FREE_GATE, 0.5),
        "no cutoff for a paid gate": lambda: calls("x1", "haiku", JEV_GATE, None),
        "an unknown gate": lambda: calls("x1", "haiku", "oracle", 0.5),
    }.items():  # fmt: skip
        check(refused(LatencyError, call), f"refused: {what}")
    cascade_policy = [cascade_trace(cascade_table, i, "haiku", JEV_GATE, 0.5) for i in items6]
    jev_calls_of = {i: calls(i, "haiku", JEV_GATE, 0.5) for i in items6}
    # Sums: x1 2.0 + .5; x2 (rejected, so to opus) 2.0 + 2.2 + .5; x3 2.0 + 2.2; y1, y2 1.6 + .3 + .5; y3 1.6 + 3.4 + .3.
    check(close(estimate_latency(lat, cascade_policy, heavy, gate_calls=jev_calls_of), 2.4)
          and close(estimate_latency(lat, cascade_policy, uniform, gate_calls=jev_calls_of), (2.5 + 4.2) / 2),
          "haiku/Jev: 2.4 s under the heavy MBPP+ mix, the midpoint of 2.5 and 4.2 under the uniform mix")  # fmt: skip
    free_policy = [cascade_trace(cascade_table, i, "haiku", FREE_GATE, None) for i in items6]
    free_calls = {i: calls(i, "haiku", FREE_GATE, None) for i in items6}
    check(close(estimate_latency(lat, free_policy, heavy, gate_calls=free_calls), 1.9)
          and close(estimate_latency(lat, free_policy, uniform, gate_calls=free_calls), 2.0),
          "haiku/free: 1.9 s under the heavy mix, 2.0 s under the uniform mix")  # fmt: skip
    # A router-started cascade (G+/Jev): the router sends x2 straight to opus and costs its per-task median everywhere.
    starts = {"x2": "opus"}
    g_plus = {"banking77": 0.1, "mbppplus": 0.12}
    started = [cascade_trace(cascade_table, i, starts.get(i, "haiku"), JEV_GATE, 0.5) for i in items6]
    started_calls = {i: calls(i, starts.get(i, "haiku"), JEV_GATE, 0.5) for i in items6}
    check(close(estimate_latency(lat, started, heavy, router=g_plus, gate_calls=started_calls), 2.52),
          "a router-started cascade: the router's median on every item, no gate for the item sent to opus")  # fmt: skip

    print("\n-- latency refusals, each fed its bad case --")
    cascade_refusals: dict[str, tuple[type[Exception], Callable[[], object]]] = {
        "a mix that weighs other tasks": (MixError, lambda: estimate_latency(lat, all_opus, {"banking77": 1.0})),
        "traces that miss an item": (TraceError, lambda: estimate_latency(lat, all_opus[1:], uniform)),
        "a router median missing a task": (LatencyError, lambda: estimate_latency(lat, all_opus, uniform, router={"banking77": 0.1})),
        "a negative router median": (LatencyError, lambda: estimate_latency(lat, all_opus, uniform, router=-1.0)),
        "a gate median of NaN": (LatencyError, lambda: estimate_latency(lat, all_opus, uniform, gate_calls={"x1": (math.nan,)})),
        "gate calls for an item not in the matrix": (LatencyError, lambda: estimate_latency(lat, all_opus, uniform, gate_calls={"zz": (1.0,)})),
        "a task the pilot has no medians for": (LatencyError, lambda: trace_seconds("cruxeval", ("haiku",))),
        "a config the pilot has no median for": (LatencyError, lambda: trace_seconds("banking77", ("gpt",))),
    }  # fmt: skip
    for what, (error, call) in cascade_refusals.items():
        check(refused(error, call), f"refused: {what}")
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
