"""Check Stage 5's paired metrics and the operating-points allowlist (PLAN_v4.md §8; Amendment A11, items 12 and 13).

Run from the project root:  uv run python -m analysis.check_metrics

On synthetic data, where the right answer is worked out by hand; no dev number is computed:

- within-task concordance: ties counting half, pairs counted only within a cell, pooled and per task; the task-rates
  diagnostic's router scores give exactly 0.5 in every task-and-fold cell, and, as the known positive, the same rates
  pooled across folds come out below 0.5; hand-drawn resamples (one with no pairs at all) and a paired difference;
- log loss: hand-worked values, pooled, per task and in hand-drawn resamples;
- the same-target comparison: hand-worked differences, one side falling back, and both falling back;
- the operating-points allowlist: E only passes, E and F passes, a section for F-train or a diagnostic is refused.
"""

import json
import math
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from analysis.bootstrap import Bootstrap, BootstrapError
from analysis.check_comparisons import MATRIX, MIX
from analysis.operating_points import (
    OPERATING_POINTS_FILE,
    OperatingPointsError,
    check_sections,
    write_section,
)
from analysis.router_metrics import MetricsError, difference, log_loss, within_task_concordance
from analysis.router_names import NAMED_ROUTERS, E, F
from analysis.same_target import SameTargetError, compare_same_target
from analysis.simulator import Trace, simulate
from analysis.walk import Step, Walk


def draws_of(tasks: Sequence[str], counts: Sequence[Sequence[int]]) -> Bootstrap:
    """Hand-drawn resamples of items i0, i1, ...: each row says how often a resample draws each item."""
    return Bootstrap(tuple(f"i{i}" for i in range(len(tasks))), tuple(tasks), np.array(counts))


def by_item(values: Sequence[float]) -> dict[str, float]:
    return {f"i{i}": value for i, value in enumerate(values)}


def main() -> int:
    problems = []

    def check(ok: bool, what: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            problems.append(what)

    def refuses(error: type[Exception], evaluate: Callable[[], object], what: str, says: str = "") -> None:
        """Passes if `evaluate` raises `error`, with `says` in its message when given."""
        try:
            evaluate()
        except error as refusal:
            check(says in str(refusal), f"{what} ({refusal})")
        else:
            check(False, what)

    def passes(evaluate: Callable[[], object], what: str) -> None:
        """Passes if `evaluate` raises nothing."""
        try:
            evaluate()
        except Exception as refusal:
            check(False, f"{what} ({refusal})")
        else:
            check(True, what)

    def close(got: float | None, expected: float, tolerance: float = 1e-9) -> bool:
        return got is not None and math.isclose(got, expected, abs_tol=tolerance)

    print("-- within-task concordance, ties counting half --")
    # Task t1, one cell: successes have router scores .9, .6, .5 and failures .7, .5. The pairs: .9 beats both failures (2), .6
    # beats .5 and loses to .7 (1), .5 loses to .7 and ties .5 (half): 3.5 of 6. Task t2 has two cells. In the first
    # the success's router score .2 is under the failure's .8 (0 of 1); in the second a success and a failure tie at .5 (half
    # of 1). Task t2 reads .5 / 2 = .25, and all three cells 4 / 8 = .5.
    tasks = ["t1"] * 5 + ["t2"] * 4
    scores = by_item([0.9, 0.6, 0.5, 0.7, 0.5, 0.2, 0.8, 0.5, 0.5])
    labels = by_item([1, 1, 1, 0, 0, 1, 0, 1, 0])
    cells = {f"i{i}": cell for i, cell in enumerate(["a"] * 5 + ["b", "b", "c", "c"])}
    dev = draws_of(tasks, [[1] * 9])
    got = within_task_concordance(dev, scores, labels, cells)
    check(close(got.per_task["t1"].value, 3.5 / 6) and got.size["t1"] == 6,
          f"task t1: 3.5 of 6 pairs in the right order: {got.per_task['t1'].value}, {got.size['t1']} pairs")  # fmt: skip
    check(close(got.per_task["t2"].value, 0.25) and got.size["t2"] == 2,
          f"task t2: pairs only within its two cells, 0.5 of 2: {got.per_task['t2'].value}")  # fmt: skip
    check(close(got.pooled.value, 0.5) and got.size[None] == 8,
          f"pooled over the cells' 8 pairs: 4 / 8 = 0.5: {got.pooled.value}, {got.size[None]} pairs")  # fmt: skip
    check(list(got.per_task) == ["t1", "t2"], "tasks are reported in the order they first appear")
    # With task t2 in one cell the pairs across the folds count too: .2 and .5 against .8 and .5 is .5 of 4.
    merged = within_task_concordance(dev, scores, labels, {**cells, "i7": "b", "i8": "b"})
    check(close(merged.per_task["t2"].value, 0.125) and merged.size["t2"] == 4,
          f"one cell for t2 counts the pairs across its two cells too: 0.5 of 4 = {merged.per_task['t2'].value}")  # fmt: skip
    # Router scores in the right order throughout, and in the wrong order throughout.
    right = within_task_concordance(dev, by_item([.9, .8, .7, .1, .2, .9, .1, .9, .1]), labels,
                                    cells)  # fmt: skip
    wrong = within_task_concordance(dev, by_item([0.1, 0.2, 0.3, 0.9, 0.8, 0.1, 0.9, 0.1, 0.9]), labels, cells)
    check(right.pooled.value == 1.0 and wrong.pooled.value == 0.0, "all in the right order is 1, all in the wrong 0")

    print("\n-- the task-rates diagnostic and the leave-fold-out effect --")
    # One task in two folds of four. Fold 0 has 3 successes and fold 1 has 1, and each fold's rate comes from the
    # other fold's labels: fold 0's items have router score 1/4, fold 1's 3/4. Within a fold every router score is equal.
    tasks = ["t1"] * 8
    labels = by_item([1, 1, 1, 0, 1, 0, 0, 0])
    rates = by_item([0.25] * 4 + [0.75] * 4)
    folds = {f"i{i}": f"t1|{i // 4}" for i in range(8)}
    dev = draws_of(tasks, [[1] * 8, [2, 0, 1, 1, 0, 1, 1, 1]])
    got = within_task_concordance(dev, rates, labels, folds)
    check(got.pooled.value == 0.5 and got.per_task["t1"].value == 0.5 and got.size[None] == 6,
          f"each fold's router scores are equal, so every pair ties: exactly 0.5 over {got.size[None]} pairs "
          f"({got.pooled.value})")  # fmt: skip
    check(got.pooled.low == 0.5 and got.pooled.high == 0.5, "and exactly 0.5 in every resample too")
    # Across the folds: 4 successes, 4 failures, 16 pairs. Fold 0's successes (1/4, three of them) lose to fold 1's
    # failures (3/4, three) in 9 pairs; the one fold-1 success beats the one fold-0 failure in 1; the other 6 tie.
    # (1 + 6 / 2) / 16 = 0.25: below 0.5 for router scores that know only the task.
    pooled = within_task_concordance(dev, rates, labels, {f"i{i}": "t1" for i in range(8)})
    check(close(pooled.pooled.value, 0.25) and (pooled.pooled.value or 1) < 0.5 and pooled.size[None] == 16,
          f"the same rates pooled across folds come out below 0.5, the known positive: {pooled.pooled.value}")  # fmt: skip

    print("\n-- concordance in hand-drawn resamples --")
    # Items i0 to i3 of one task and one cell: successes i0 (.9) and i2 (.3), failures i1 (.2) and i3 (.6). On dev
    # i0 beats both, i2 beats i1 and loses to i3: 3 of 4. Each resample's pairs are weighted by how often it draws
    # the two items:
    #   [1,1,1,1] 3/4 = .75      [2,0,0,1] i0 twice against i3: 2/2 = 1     [0,1,1,1] i2 against i1, i3: 1/2
    #   [1,1,0,0] i0 against i1: 1    [0,0,3,2] i2 three times against i3: 0/6 = 0
    #   [0,1,2,0] i2 twice against i1: 1    [1,0,1,0] no failure drawn: no pair, n/a
    counts = [[1, 1, 1, 1], [2, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 0], [0, 0, 3, 2], [0, 1, 2, 0], [1, 0, 1, 0]]
    one_cell = draws_of(["t1"] * 4, counts)
    labels = by_item([1, 0, 1, 0])
    scores = by_item([0.9, 0.2, 0.3, 0.6])
    got = within_task_concordance(one_cell, scores, labels, {f"i{i}": "t1|0" for i in range(4)})
    # The six resamples with pairs, ranked: 0, .5, .75, 1, 1, 1. Ranks .025 x 5 = .125 and .975 x 5 = 4.875.
    check(close(got.pooled.value, .75) and close(got.pooled.low, .125 * .5) and close(got.pooled.high, 1.0),
          f"the interval comes from the six resamples that have pairs: {got.pooled}")  # fmt: skip
    check(math.isclose(got.pooled.undefined, 1 / 7), f"and one resample in seven has no pair: {got.pooled.undefined}")
    check(got.pooled == got.per_task["t1"], "one task: the task's reading is the pooled one")
    resampled = got.resampled[None]
    expected = [0.75, 1, 0.5, 1, 0, 1, math.nan]
    check(np.allclose(resampled, expected, equal_nan=True, atol=1e-12), f"resample by resample: {resampled.tolist()}")

    print("\n-- the paired difference of two concordances --")
    flat = within_task_concordance(one_cell, by_item([0.5] * 4), labels, {f"i{i}": "t1|0" for i in range(4)})
    check(np.allclose(flat.resampled[None], [.5, .5, .5, .5, .5, .5, math.nan], equal_nan=True),
          "router scores that are all equal read 0.5 in every resample that has pairs")  # fmt: skip
    # Resample by resample, router scores less flat: .25, .5, 0, .5, -.5, .5 and n/a. Ranked -.5, 0, .25, .5, .5, .5.
    paired = difference(got, flat)
    check(close(paired.pooled.value, .25) and close(paired.pooled.low, -.5 + .125 * .5) and close(paired.pooled.high, .5)
          and math.isclose(paired.pooled.undefined, 1 / 7),
          f"the difference is taken in each resample, so the same resamples serve both: {paired.pooled}")  # fmt: skip
    zero = difference(got, got)
    check(zero.pooled.value == 0 and zero.pooled.low == 0 and zero.pooled.high == 0,
          "a concordance less itself is exactly 0 in every resample")  # fmt: skip
    check(list(paired.per_task) == ["t1"], "the difference is per task too")

    print("\n-- concordance refusals --")
    cell = {f"i{i}": "t1|0" for i in range(4)}
    refuses(MetricsError, lambda: within_task_concordance(one_cell, {"i0": .9}, labels, cell),
            "router scores that miss items are refused", "every item")  # fmt: skip
    refuses(MetricsError, lambda: within_task_concordance(one_cell, scores, {**labels, "i9": 1.0}, cell),
            "successes of an item the resamples don't hold are refused", "every item")  # fmt: skip
    refuses(MetricsError, lambda: within_task_concordance(one_cell, scores, labels, {"i0": "x"}),
            "cells that miss items are refused", "every item")  # fmt: skip
    refuses(MetricsError, lambda: within_task_concordance(one_cell, scores, by_item([1, 0, 1, .5]), cell),
            "a success that isn't 1 or 0 is refused", "0 or 1")  # fmt: skip
    refuses(MetricsError, lambda: within_task_concordance(one_cell, by_item([.9, .2, math.nan, .6]), labels, cell),
            "a router score that isn't a number is refused", "number")  # fmt: skip
    two = draws_of(["t1", "t1", "t2", "t2"], [[1, 1, 1, 1]])
    refuses(MetricsError, lambda: within_task_concordance(two, scores, labels, cell),
            "a cell that spans two tasks is refused", "one task")  # fmt: skip
    refuses(
        MetricsError,
        lambda: difference(
            got, within_task_concordance(two, scores, labels, {"i0": "a", "i1": "a", "i2": "b", "i3": "b"})
        ),
        "readings from other resamples are refused",
        "same resamples",
    )

    print("\n-- log loss --")
    # -ln .9 = .105360516, -ln .8 = .223143551, -ln .3 = 1.203972804, -ln .4 = .916290732 (labels 1, 0, 1, 0).
    loss = {"i0": 0.105360516, "i1": 0.223143551, "i2": 1.203972804, "i3": 0.916290732}
    got = log_loss(one_cell, scores, labels)
    check(close(got.pooled.value, 2.448767603 / 4, 1e-8) and got.size[None] == 4,
          f"mean of -ln .9, -ln .8, -ln .3 and -ln .4 is .6121919: {got.pooled.value}")  # fmt: skip
    hand = [
        0.612191901,
        1.127011764 / 3,
        2.343407087 / 3,
        0.164252034,
        5.444499876 / 5,
        2.631089159 / 3,
        1.309333320 / 2,
    ]
    check(
        np.allclose(got.resampled[None], hand, atol=1e-8), f"resample by resample: {np.round(got.resampled[None], 6)}"
    )
    low, high = np.percentile(hand, [2.5, 97.5])
    check(close(got.pooled.low, low, 1e-8) and close(got.pooled.high, high, 1e-8) and got.pooled.undefined == 0,
          f"with a percentile interval, every resample having items: {got.pooled}")  # fmt: skip
    two_tasks = draws_of(["t1", "t1", "t2", "t2"], [[1, 1, 1, 1]])
    per_task = log_loss(two_tasks, scores, labels)
    check(close(per_task.per_task["t1"].value, (loss["i0"] + loss["i1"]) / 2, 1e-8)
          and close(per_task.per_task["t2"].value, (loss["i2"] + loss["i3"]) / 2, 1e-8),
          "per task: each task's mean loss")  # fmt: skip
    check(close(log_loss(one_cell, by_item([.5] * 4), labels).pooled.value, math.log(2), 1e-12),
          "a router score of 0.5 for everything costs ln 2")  # fmt: skip
    check(
        close(
            log_loss(one_cell, by_item([0.0, 1.0, 0.0, 1.0]), labels).pooled.value,
            -math.log(2.220446049250313e-16),
            1e-9,
        ),  # fmt: skip
        "a router score of exactly 0 or 1 on the wrong side costs -ln of the float epsilon, clipped as scikit-learn does",
    )
    check(close(log_loss(one_cell, by_item([1.0, 0.0, 1.0, 0.0]), labels).pooled.value, 0, 1e-15),
          "and exactly right costs next to nothing, not infinity")  # fmt: skip
    check(difference(got, got).pooled.value == 0, "a log loss less itself is 0")
    refuses(MetricsError, lambda: log_loss(one_cell, by_item([.9, .2, 1.2, .6]), labels),
            "a router score above 1 is refused", "0 to 1")  # fmt: skip
    refuses(MetricsError, lambda: log_loss(one_cell, by_item([.9, .2, -.1, .6]), labels),
            "a router score below 0 is refused", "0 to 1")  # fmt: skip
    refuses(MetricsError, lambda: log_loss(one_cell, scores, by_item([1, 0, 1, 2])),
            "a success of 2 is refused", "0 or 1")  # fmt: skip

    print("\n-- the same-target comparison --")
    # On the four-item matrix (check_comparisons) the policy sends i0 to i2 to haiku and i3 to opus: right on i0 and i3,
    # quality .5 at $3.25 per 1k. All-sonnet is right on i0 and i1: .5 at $2. All-opus-low: .75 at $4. A is always-opus:
    # 1 at $10.
    mixed = simulate(MATRIX, [Trace(f"i{i}", ("opus",) if i == 3 else ("haiku",)) for i in range(4)])
    sonnet = simulate(MATRIX, [Trace(f"i{i}", ("sonnet",)) for i in range(4)])
    opus_low = simulate(MATRIX, [Trace(f"i{i}", ("opus_low",)) for i in range(4)])
    opus = simulate(MATRIX, [Trace(f"i{i}", ("opus",)) for i in range(4)])

    def operating(outcomes: list, share: float = 0.99, fell_back: bool = False) -> Walk:
        """A walk whose operating point is the outcomes' policy (or A, when `fell_back`)."""
        cost = sum(o.cost for o in outcomes) * 250
        step = Step(None if fell_back else 0.5, sum(o.success for o in outcomes) / 4, cost, 0.0, True, tuple(outcomes))
        dearer = Step(None, 1.0, 10.0, 0.0, True, tuple(opus))
        return Walk(share, 1.0, (dearer,) if fell_back else (dearer, step))

    # Resamples: [1,1,1,1], [0,0,0,4], [4,0,0,0] and [2,0,0,2], with item weights the counts over their sum. The
    # policy less all-sonnet, resample by resample: quality 0, 1, 0, .5; cost per 1k 1.25, 8, -1, 3.5.
    #   [1,1,1,1]: policy .5 at 3.25, sonnet .5 at 2.       [0,0,0,4]: policy 1 at 10, sonnet 0 at 2.
    #   [4,0,0,0]: policy 1 at 1, sonnet 1 at 2.             [2,0,0,2]: policy 1 at 5.5, sonnet .5 at 2.
    drawn = draws_of(["t1"] * 4, [[1, 1, 1, 1], [0, 0, 0, 4], [4, 0, 0, 0], [2, 0, 0, 2]])
    same = compare_same_target(drawn, MIX, operating(mixed), operating(sonnet))
    check(not same.both_fall_back and same.reading == "" and same.cost is not None and same.quality is not None,
          "two operating points that don't fall back are compared")  # fmt: skip
    assert same.cost is not None and same.quality is not None
    check(
        close(same.cost.value, 1.25) and close(same.quality.value, 0),
        f"the first less the second on dev: $1.25 more per 1k, the same quality: {same.cost.value}, {same.quality.value}",
    )
    check(close(same.cost.low, -1 + .075 * 2.25) and close(same.cost.high, 3.5 + .925 * 4.5)
          and close(same.quality.low, 0) and close(same.quality.high, .5 + .925 * .5),
          f"intervals from the same four resamples: cost {same.cost.low:.5f} to {same.cost.high:.5f}, "
          f"quality {same.quality.low:.5f} to {same.quality.high:.5f}")  # fmt: skip
    same = compare_same_target(drawn, MIX, operating(mixed), operating(opus_low))
    assert same.cost is not None and same.quality is not None
    check(
        close(same.cost.value, -0.75) and close(same.quality.value, -0.25), "another pair: -$0.75 per 1k, -.25 quality"
    )
    flipped = compare_same_target(drawn, MIX, operating(opus_low), operating(mixed))
    assert flipped.cost is not None and flipped.quality is not None
    check(close(flipped.cost.value, 0.75) and close(flipped.quality.value, 0.25)
          and close(flipped.cost.low, -same.cost.high) and close(flipped.cost.high, -same.cost.low),
          "swapping the two flips every sign and bound")  # fmt: skip

    print("\n-- one side, or both, falling back to A --")
    # The policy less A: quality .5 less 1, cost $3.25 less $10 per 1k. A less A is nothing to compare.
    same = compare_same_target(drawn, MIX, operating(mixed), operating(opus, fell_back=True))
    assert same.cost is not None and same.quality is not None
    check(not same.both_fall_back and (same.first_fell_back, same.second_fell_back) == (False, True)
          and close(same.cost.value, -6.75) and close(same.quality.value, -0.5),
          f"a side that falls back is A: -$6.75 per 1k and -.5 quality: {same.cost.value}, {same.quality.value}")  # fmt: skip
    same = compare_same_target(drawn, MIX, operating(opus, fell_back=True), operating(mixed))
    assert same.cost is not None and same.quality is not None
    check(
        (same.first_fell_back, same.second_fell_back) == (True, False)
        and close(same.cost.value, 6.75)
        and close(same.quality.value, 0.5),
        "and in the other order, A less the policy",
    )
    same = compare_same_target(drawn, MIX, operating(opus, fell_back=True), operating(opus, fell_back=True))
    check(same.both_fall_back and same.cost is None and same.quality is None and same.reading == "both fall back to A",
          f"both falling back reads {same.reading!r}, with no difference")  # fmt: skip
    refuses(SameTargetError, lambda: compare_same_target(drawn, MIX, operating(mixed, 0.99), operating(sonnet, 0.97)),
            "operating points for different targets are refused", "same target")  # fmt: skip
    refuses(BootstrapError, lambda: compare_same_target(drawn, MIX, operating(mixed), operating(sonnet[1:])),
            "outcomes that miss an item are refused")  # fmt: skip
    refuses(
        SameTargetError,
        lambda: compare_same_target(drawn, MIX, Walk(0.99, 1.0, ()), operating(sonnet)),
        "a walk with no steps is refused",
        "no steps",
    )

    print("\n-- the operating-points allowlist --")
    rules, trained = NAMED_ROUTERS[E], NAMED_ROUTERS[F]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "operating_points.json"
        write_section(rules, {}, path)
        passes(lambda: check_sections(path), f"a file with E's section ({rules!r}) only passes")
        write_section(trained, {}, path)
        passes(lambda: check_sections(path), f"a file with E's and F's sections ({rules!r}, {trained!r}) passes")
        for name in ("F-train", "f_train", "task_rates", "task-rates diagnostic", "concordance"):
            other = Path(folder) / "other.json"
            other.write_text(path.read_text())
            write_section(name, {}, other)
            refuses(OperatingPointsError, lambda: check_sections(other), f"a section for {name!r} is refused", name)
        only_f = Path(folder) / "only_f.json"
        write_section(trained, {}, only_f)
        passes(lambda: check_sections(only_f), "F's section alone passes the allowlist, which doesn't ask for E's")
        refuses(
            OperatingPointsError,
            lambda: check_sections(only_f, required=(rules, trained)),
            "a required section that is missing is refused (how Stage 5 · 6 adds 'F's section is present')",
            rules,
        )
        passes(
            lambda: check_sections(path, required=(rules, trained)), "and a file holding every required section passes"
        )
        refuses(OperatingPointsError, lambda: check_sections(path, required=("task_rates",)),
                "a required section must itself be a named router", "task_rates")  # fmt: skip
        missing = Path(folder) / "missing.json"
        refuses(OperatingPointsError, lambda: check_sections(missing), "a file that isn't there is refused", "no file")
        listed = Path(folder) / "listed.json"
        listed.write_text(json.dumps([rules]))
        refuses(OperatingPointsError, lambda: check_sections(listed), "a file holding a list is refused", "object")
    passes(lambda: check_sections(OPERATING_POINTS_FILE, required=(rules, trained)),
           "F's section is present in the committed file, beside E's")  # fmt: skip
    passes(lambda: check_sections(OPERATING_POINTS_FILE),
           f"the committed file passes: sections {sorted(json.loads(OPERATING_POINTS_FILE.read_text()))}")  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
