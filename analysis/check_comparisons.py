"""Check Stage 4's two comparisons for table v1 (PLAN_v4.md §4 and §8; Amendment A9, items 11, 12 and 15).

Run from the project root:  uv run python -m analysis.check_comparisons

On synthetic data, where the right answer is worked out by hand:

- McNemar's test: p from the discordant counts, hand-computed (0 and 5 give 0.0625; 1 and 9 give 22/1024), the same
  for b and c swapped, 1 for no discordant pairs, never above 1; which policy each count favours; and the refusal,
  whole, of outcomes that aren't the same items, that span more than one task, or whose success isn't 0 or 1;
- the matched-cost difference on a hand-worked matrix: the envelope's quality at a policy's cost, flat at its top
  corner's quality past the costliest corner and n/a below the cheapest; a fixed config that is a corner of the
  envelope differs from it by exactly 0 in every resample; and four hand-drawn resamples, one reading n/a, which rank
  below every number in the interval.
"""

import math
import sys
from collections.abc import Sequence

import numpy as np

from analysis.bootstrap import Bootstrap, BootstrapError, resample
from analysis.matrix import Matrix
from analysis.mcnemar import McNemarError, mcnemar
from analysis.simulator import Outcome, Trace, always, simulate


def outcomes(task: str, successes: Sequence[float], prefix: str = "i") -> list[Outcome]:
    """One outcome per success, items i0, i1, ..., each its own source group, at a cost of $0.001."""
    return [Outcome(f"{prefix}{i}", task, f"{prefix}{i}", 0.001, success, ("haiku",))
            for i, success in enumerate(successes)]  # fmt: skip


def discordant(b: int, c: int) -> tuple[list[Outcome], list[Outcome]]:
    """Two policies' outcomes on b + c + 2 items: the first alone is right on b, the second alone on c, and both are
    right on one and wrong on one."""
    first = [1] * b + [0] * c + [1, 0]
    second = [0] * b + [1] * c + [1, 0]
    return outcomes("t1", first), outcomes("t1", second)


# Four items of one task, each its own source group, so under the mix {t1: 1} each weighs 0.25. Each config's cost is
# the same on every item, and each config is right on the first so many: haiku 1 of 4 at $1 per 1k, sonnet 2 at $2,
# opus_low 3 at $4, opus 4 at $10.
COSTS = {"haiku": 0.001, "sonnet": 0.002, "opus_low": 0.004, "opus": 0.010}
RIGHT = {"haiku": 1, "sonnet": 2, "opus_low": 3, "opus": 4}
MATRIX = Matrix.from_rows([
    {"item_id": f"i{i}", "split": "dev", "task": "t1", "source_group": f"i{i}", "config": config,
     "cost": COSTS[config], "success": float(i < RIGHT[config])}
    for i in range(4) for config in COSTS
])  # fmt: skip
MIX = {"t1": 1.0}
# On dev the envelope's corners are all four configs: quality per dollar falls from 0.25 to 0.125 to 0.0417.
FIXED = {config: simulate(MATRIX, always(MATRIX, config)) for config in COSTS}


def main() -> int:
    problems = []

    def check(ok: bool, what: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            problems.append(what)

    def refuses(error: type[Exception], evaluate, what: str, says: str = "") -> None:
        """Passes if `evaluate` raises `error`, with `says` in its message when given."""
        try:
            evaluate()
        except error as refusal:
            check(says in str(refusal), f"{what} ({refusal})")
        else:
            check(False, what)

    print("-- McNemar's test --")
    # 0 and 5: 2 x C(5,0) / 2^5 = 1/16. 1 and 9: 2 x (C(10,0) + C(10,1)) / 2^10 = 22/1024. 3 and 3: 2 x 42 / 64 > 1.
    for b, c, p in ((0, 5, 0.0625), (1, 9, 22 / 1024), (5, 0, 0.0625), (9, 1, 22 / 1024), (3, 3, 1.0), (0, 1, 1.0)):
        first, second = discordant(b, c)
        test = mcnemar(first, second, "x", "y")
        check((test.b, test.c) == (b, c) and math.isclose(test.p, p, rel_tol=1e-12),
              f"{b} items only the first policy gets right, {c} only the second: p = {p:.6f}, got {test.p:.6f}")  # fmt: skip
    first, second = discordant(0, 0)
    check(mcnemar(first, second, "x", "y").p == 1, "no discordant pairs: p = 1")
    first, second = discordant(1, 9)
    test = mcnemar(first, second, "x", "y")
    check((test.b_favours, test.c_favours) == ("x", "y"),
          f"b counts the items only x gets right and c those only y does, each naming its policy: "
          f"{test.b_favours}, {test.c_favours}")  # fmt: skip
    check(mcnemar(first, second, "x", "y").p == mcnemar(second, first, "y", "x").p,
          "swapping the two policies leaves p unchanged")  # fmt: skip
    # 2 / 2^1000 is about 1.9e-301, the smallest this reaches before a float 2^1000 would overflow a naive version.
    first, second = discordant(0, 1000)
    check(mcnemar(first, second, "x", "y").p == 2 / 2**1000,
          "1,000 discordant pairs all one way: p = 2 / 2^1000, from exact integers")  # fmt: skip
    first, second = outcomes("t1", [1, 0, 1]), outcomes("t1", [0, 1, 0])[::-1]
    swapped = mcnemar(first, second, "x", "y")
    check((swapped.b, swapped.c) == (2, 1), "outcomes in another order are matched by item")

    good = outcomes("t1", [1, 0, 1])
    refuses(McNemarError, lambda: mcnemar(good, good[:2], "x", "y"), "outcomes that miss an item are refused",
            "same items")  # fmt: skip
    refuses(McNemarError, lambda: mcnemar(good, outcomes("t1", [1, 0, 1], "j"), "x", "y"),
            "outcomes of other items are refused", "same items")  # fmt: skip
    refuses(McNemarError, lambda: mcnemar(good + good[:1], good + good[:1], "x", "y"),
            "outcomes that cover an item twice are refused", "same items")  # fmt: skip
    two_tasks = outcomes("t1", [1, 0]) + outcomes("t2", [1, 0], "j")
    refuses(McNemarError, lambda: mcnemar(two_tasks, two_tasks, "x", "y"), "outcomes spanning two tasks are refused",
            "one task")  # fmt: skip
    refuses(McNemarError, lambda: mcnemar(good, outcomes("t2", [1, 0, 1]), "x", "y"),
            "the same items in different tasks are refused", "one task")  # fmt: skip
    for wrong in (0.5, 2.0, -1.0, math.nan):
        refuses(McNemarError, lambda: mcnemar(outcomes("t1", [1, 0, wrong]), good, "x", "y"),
                f"a success of {wrong} is refused", "0 or 1")  # fmt: skip
        refuses(McNemarError, lambda: mcnemar(good, outcomes("t1", [wrong, 0, 1]), "x", "y"),
                f"a success of {wrong} in the second policy is refused", "0 or 1")  # fmt: skip
    refuses(McNemarError, lambda: mcnemar([], [], "x", "y"), "no outcomes are refused", "no outcomes")

    print("\n-- the matched-cost difference on a hand-worked matrix --")
    draws = resample(MATRIX)
    # Quality and cost per 1k: haiku (0.25, $1), sonnet (0.5, $2), opus_low (0.75, $4), opus (1, $10).
    for config, quality in (("haiku", 0.25), ("sonnet", 0.5), ("opus_low", 0.75), ("opus", 1.0)):
        got = draws.matched_cost(FIXED[config], FIXED, MIX)
        check(got.envelope is not None and math.isclose(got.envelope, quality) and got.difference == 0,
              f"always-{config} is a corner of the envelope on dev: it reads {quality:.2f} at its own cost, "
              f"difference exactly 0 ({got.envelope}, {got.difference})")  # fmt: skip
    every = draws.matched_cost_differences(FIXED["haiku"], FIXED, MIX)
    check(len(every) == len(draws.counts) and all(difference == 0 for difference in every),
          "the cheapest config is a corner of the envelope in every resample: its difference is exactly 0 in all of them")  # fmt: skip
    below = [difference for config in COSTS for difference in draws.matched_cost_differences(FIXED[config], FIXED, MIX)]
    check(all(difference is not None and difference <= 1e-12 for difference in below),
          "no fixed config is ever above the envelope built from them, in any resample")  # fmt: skip

    # Always-haiku with $0.015 of overhead per request costs $16 per 1k at quality 25%, past opus's $10.
    pricey = simulate(MATRIX, [Trace(f"i{i}", ("haiku",), 0.015) for i in range(4)])
    got = draws.matched_cost(pricey, FIXED, MIX)
    check(got.envelope == 1.0 and got.difference is not None and math.isclose(got.difference, -0.75),
          f"a policy at $16 per 1k, past the costliest corner, is compared with that corner's 100%: "
          f"{got.envelope}, {got.difference}")  # fmt: skip
    check(got.high is not None and got.high <= 0 and got.unread == 0,
          "and no resample reads n/a, nor is any above the top corner's quality")  # fmt: skip

    # Without the cheapest corner, haiku at $1 per 1k costs less than the envelope's first corner, sonnet's $2.
    dearer = {config: FIXED[config] for config in ("sonnet", "opus_low", "opus")}
    got = draws.matched_cost(FIXED["haiku"], dearer, MIX)
    check((got.envelope, got.difference, got.low, got.high, got.unread) == (None, None, None, None, 1.0),
          f"a policy costing less than the cheapest corner reads n/a on dev and in every resample: {got}")  # fmt: skip
    at_corner = draws.matched_cost(FIXED["sonnet"], dearer, MIX)
    check(at_corner.envelope == 0.5 and at_corner.difference == 0,
          "one costing exactly the cheapest corner reads that corner's quality")  # fmt: skip

    print("\n-- four hand-drawn resamples, one of them n/a --")
    # The envelope of sonnet and opus alone; the policy is right on i0 and i3, haiku on i0 to i2 and opus on i3, so it
    # is $1 per 1k on those three items and $10 on i3. Each resample's weights are its counts over their sum.
    #   [1,1,1,1]: policy 0.5 at $3.25; sonnet 0.5, opus 1; the line reads 0.5 + 1.25 x 0.0625 = 0.578125 there.
    #   [0,0,0,4]: policy 1 at $10, sonnet 0, opus 1: reads 1 at its own cost, difference 0.
    #   [4,0,0,0]: policy 1 at $1, below sonnet's $2: n/a.
    #   [2,0,0,2]: policy 1 at $5.50; sonnet 0.5, opus 1: reads 0.5 + 3.5 x 0.0625 = 0.71875.
    counts = np.array([[1, 1, 1, 1], [0, 0, 0, 4], [4, 0, 0, 0], [2, 0, 0, 2]])
    drawn = Bootstrap(tuple(MATRIX.items), ("t1",) * 4, counts)
    policy = simulate(MATRIX, [Trace(f"i{i}", ("opus",) if i == 3 else ("haiku",)) for i in range(4)])
    # Haiku is right on i0 only and opus on all four, so the policy is right on i0 and i3 as written above.
    check([o.success for o in policy] == [1, 0, 0, 1], "the policy is right on i0 and i3, from the matrix itself")
    sonnet_opus = {config: FIXED[config] for config in ("sonnet", "opus")}
    expected = [-0.078125, 0.0, None, 0.28125]
    got_each = drawn.matched_cost_differences(policy, sonnet_opus, MIX)
    check(len(got_each) == 4
          and all(a is None if e is None else a is not None and math.isclose(a, e, abs_tol=1e-12)
                  for a, e in zip(got_each, expected, strict=True)),
          f"the four resamples read {expected}, the third n/a: {got_each}")  # fmt: skip
    # Ranked n/a, -0.078125, 0, 0.28125: the lower bound is n/a since 1 of 4 is more than 2.5%; the upper sits at rank
    # 0.975 x 3 = 2.925 of the four, between 0 and 0.28125: 0.925 x 0.28125.
    result = drawn.matched_cost(policy, sonnet_opus, MIX)
    check(result.envelope is not None and math.isclose(result.envelope, 0.578125)
          and result.difference is not None and math.isclose(result.difference, -0.078125)
          and result.low is None and result.high is not None and math.isclose(result.high, 0.925 * 0.28125)
          and result.unread == 0.25,
          f"on dev the envelope reads 0.578125 and the difference is -0.078125; with n/a ranked below every number the "
          f"interval is n/a to {0.925 * 0.28125:.6f}, and a quarter of resamples read n/a: {result}")  # fmt: skip

    for what, bad in (("miss an item", policy[1:]), ("cover an item twice", policy + policy[:1])):
        refuses(BootstrapError, lambda: drawn.matched_cost(bad, sonnet_opus, MIX),
                f"policy outcomes that {what} are refused")  # fmt: skip
        refuses(BootstrapError, lambda: drawn.matched_cost(policy, {**sonnet_opus, "opus": bad}, MIX),
                f"a fixed config's outcomes that {what} are refused")  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
