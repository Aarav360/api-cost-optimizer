"""Check the Stage 3 analysis package (PLAN_v4.md §7, Amendment A8).

Run from the project root:  uv run python -m analysis.check_simulator

On a synthetic matrix, where the right answer is worked out by hand:

- fixed configs' totals, and a routed policy's with overhead, under a task mix and per
  task;
- a cascade's accounting: a trace that climbs from haiku to opus pays for both calls
  and its overhead, and is right exactly when opus is;
- the simulator refuses a trace set that misses an item, covers one twice or has a
  trace for an item not in the matrix, and a trace that names an unknown config, calls
  a config with no stored answer, calls no config or calls one twice, or calls opus on
  a train item (with its own refusal, even where no opus answer is stored), and
  evaluates none of the set;
- a mix that doesn't weigh exactly the outcomes' tasks is refused;
- a reference curve gives the cost at a quality and the quality at a cost along the
  straight lines between its corners, both ends included, and n/a outside its range; it
  refuses no corners, a corner that isn't a finite number, and corners whose quality or
  cost doesn't rise; the envelope refuses no points, a point that isn't a finite number
  (even one that would drop out of the hull), and a point off the config ladder;
- the random-mixture envelope keeps four corners where each config buys less quality
  per dollar than the one below, drops a config below the hull and one costlier with no
  higher quality, gives an exact tie's corner to the lower config, returns the starting
  quality on a round trip, and gives n/a past both ends;
- the hindsight oracle's corners, its cost at a quality, its quality at a cost and its
  routing mix, with the last upgrade fractional, where an item every config fails stays
  on its cheapest config; with scores between 0 and 1, items with three and four hull
  points and a config below the hull never used; under a mix it trades quality between
  tasks, which each task's own oracle can't; every item starts on its own cheapest
  config, even where that isn't haiku; exact cost ties and equally steep upgrades go to
  the lower config; a target at or below always-cheapest's quality gets its cost, one
  above the maximum is n/a, and so is a cost outside the curve's; round trips from a
  quality and from a cost; an item that weighs 0 is never upgraded; it refuses weights
  that miss an item, name one not in the matrix, are negative, aren't a number or don't
  sum to 1, and an item with no stored answer; a matrix refuses a cell for a config off
  the ladder;
- the quality targets at a frontier quality of 90%: 100%, 99% and 97% of it (§11);
- on 300 random matrices, the oracle against a brute force over whole-item choices: the
  LP costs no more at any target, and less by at most the upgrade in progress; they
  agree at every corner, at the maximum quality and below the first corner.

On the dev results matrix:

- every row's cost recomputed from its tokens at list price is exactly twice its
  stored batch cost, and the loader refuses a row where it isn't, a row outside train
  and dev, a row for an item the benchmark doesn't hold in train or dev, a row priced
  at another snapshot, and two run-0 rows for one item and config;
- only train and dev items are loaded, and only run 0;
- the four fixed configs' totals match Amendment A8 item 1's table under both mixes;
- the config ladder is the order of mean cost per request in every task and both
  mixes, and its per-item inversions and exact cost ties are as Amendment A8 item 5
  gives them;
- under both mixes the envelope is one segment from haiku to opus_low, with sonnet and
  opus below it, as Amendment A8 item 7 says;
- the hindsight oracle as Amendment A8 item 6 gives it: 46 items fail on every config
  and on 28 opus fails and a cheaper config succeeds; under both mixes every item starts
  on haiku, at always-haiku's point, each hull has at most two points, the maximum is
  opus's quality plus the 28 items' weight, and all three quality targets are feasible;
  each task's own oracle starts at always-haiku's point and peaks at the share of its
  items some config gets right;
- a toy cascade (haiku, climbing to opus on MMLU-Pro and MBPP+) prints its per-task
  totals, each haiku's or opus's quality with the calls and gate it paid for, and its
  per-mix totals, each the mix's weighting of the per-task ones;
- no analysis module names the sealed store (check_sealed, over the whole repo).
"""

import math
import sys
from collections import Counter
from itertools import combinations, pairwise, product

import numpy as np

from analysis.matrix import LADDER, Item, Matrix, MatrixError, from_store, list_price, read_store
from analysis.references import (
    Curve,
    CurveError,
    OracleError,
    envelope,
    oracle,
    quality_targets,
    task_oracles,
)
from analysis.simulator import (
    MixError,
    Total,
    Trace,
    TraceError,
    always,
    item_weights,
    load_mixes,
    simulate,
    task_totals,
    totals,
)
from harness import check_sealed
from harness.client import load_models

DEV_TASKS = {"ifeval": 157, "banking77": 240, "mmlu_pro": 240, "mbppplus": 107}  # Amendment A5, item 3
# Amendment A8, item 1 (dev, run 0, list prices): quality % and $ per 1k under each mix.
A8_TABLE = {
    "haiku": {"uniform": (84.5, 0.43), "realistic": (83.3, 0.29)},
    "sonnet": {"uniform": (86.1, 5.81), "realistic": (84.5, 3.80)},
    "opus_low": {"uniform": (89.7, 9.61), "realistic": (87.8, 6.56)},
    "opus": {"uniform": (89.5, 12.01), "realistic": (86.7, 8.04)},
}
# Amendment A8, item 5 (dev): the % of items on which a config costs more than the next one up the ladder, and
# the exact cost ties between two configs on one item.
A8_ABOVE_NEXT = {("sonnet", "opus_low"): "4.8", ("opus_low", "opus"): "12.8"}
A8_TIES, A8_TIES_BANKING77 = {("opus_low", "opus"): 234}, 209
# Amendment A8, item 6 (dev): the items every config fails, and those opus fails and a cheaper config gets right.
A8_ORACLE_ITEMS = (46, 28)


def synthetic(items: dict[str, tuple[str, dict[str, tuple[float, float]]]], split: str = "dev") -> Matrix:
    """{item_id: (task, {config: (list-price cost, score)})}: items of one split, each its own source group."""
    return Matrix.from_rows([
        {"item_id": item_id, "split": split, "task": task, "source_group": item_id, "config": config,
         "cost": cost, "success": success}
        for item_id, (task, cells) in items.items() for config, (cost, success) in cells.items()
    ])  # fmt: skip


# Three items, two tasks. Under the mix t1 0.75, t2 0.25, each t1 item weighs 0.375 and c weighs 0.25.
HAND = synthetic({
    "a": ("t1", {"haiku": (0.001, 1), "opus": (0.004, 1)}),
    "b": ("t1", {"haiku": (0.002, 0), "opus": (0.006, 1)}),
    "c": ("t2", {"haiku": (0.003, 0), "opus": (0.010, 0)}),
})  # fmt: skip
HAND_MIX = {"t1": 0.75, "t2": 0.25}
# Train items, where the frontier never runs (Amendment A1, item 3). "v" holds an opus answer anyway,
# so only the frontier-on-train refusal can catch a trace that calls opus on it.
TRAIN = synthetic({
    "u": ("t1", {"haiku": (0.001, 1), "opus_low": (0.003, 1)}),
    "v": ("t1", {"haiku": (0.002, 0), "opus_low": (0.004, 1), "opus": (0.005, 1)}),
}, split="train")  # fmt: skip


# Scores between 0 and 1, as §4 allows, so an item's hull can have more than two points. In $ per 1k: p's four
# configs climb 0.4, 0.1 and then 0.025 points of score per dollar, all on its hull. q's sonnet ($3, 0.4) lies
# below the line from haiku ($1, 0.2) to opus_low ($4, 0.8), so q's hull is haiku, opus_low, opus.
SCORED = synthetic({
    "p": ("t1", {"haiku": (0.001, 0.2), "sonnet": (0.002, 0.6), "opus_low": (0.004, 0.8), "opus": (0.008, 0.9)}),
    "q": ("t2", {"haiku": (0.001, 0.2), "sonnet": (0.003, 0.4), "opus_low": (0.004, 0.8), "opus": (0.008, 1.0)}),
})  # fmt: skip
SCORED_WEIGHTS = {"p": 0.25, "q": 0.75}  # given directly, not by a mix
# Its upgrades, steepest first: p to sonnet, q to opus_low, p to opus_low, q to opus, p to opus. Each adds its
# weight times its score and cost: from (20%, $1) to (30%, $1.25), (75%, $3.50), (80%, $4), (95%, $7), (97.5%, $8).
SCORED_CORNERS = [(0.2, 1), (0.3, 1.25), (0.75, 3.5), (0.8, 4), (0.95, 7), (0.975, 8)]


# What dev hides: x's cheapest config is sonnet, not haiku. y's opus_low and opus tie exactly and are both right;
# z's haiku and sonnet tie exactly and are both wrong, as is every config. v and u climb equally steeply, 1 point
# per $2 per 1k, v to opus and u to sonnet. Each of the five items weighs 0.2.
TIES = synthetic({
    "x": ("t1", {"sonnet": (0.001, 0), "haiku": (0.002, 1)}),
    "y": ("t1", {"haiku": (0.001, 0), "opus_low": (0.004, 1), "opus": (0.004, 1)}),
    "z": ("t1", {"haiku": (0.001, 0), "sonnet": (0.001, 0), "opus": (0.003, 0)}),
    "v": ("t1", {"haiku": (0.001, 0), "opus": (0.003, 1)}),
    "u": ("t1", {"haiku": (0.001, 0), "sonnet": (0.003, 1)}),
})  # fmt: skip


BRUTE_FORCE_RUNS = 300

CLIMBS, TOY_GATE = {"mmlu_pro", "mbppplus"}, 0.0001  # the toy cascade's tasks that climb to opus, its gate's cost


def toy_cascade(matrix: Matrix) -> list[Trace]:
    """haiku answers every item; a toy gate checks each answer and rejects every one on MMLU-Pro and MBPP+."""
    return [Trace(item_id, ("haiku", "opus") if item.task in CLIMBS else ("haiku",), TOY_GATE)
            for item_id, item in matrix.items.items()]  # fmt: skip


def random_matrix(rng: np.random.Generator) -> tuple[Matrix, dict[str, float]]:
    """2 to 5 items, each with 1 to 4 configs at $1 to $6 per 1k, so exact cost ties are common, and either 0/1
    scores or scores in quarters. Random weights summing to 1, one of them sometimes 0."""
    scores = [0, 1] if rng.random() < 0.5 else [0, 0.25, 0.5, 0.75, 1]
    items = {f"i{i}": ("t1", {str(config): (int(rng.integers(1, 7)) / 1000, float(rng.choice(scores)))
                              for config in rng.choice(LADDER, size=rng.integers(1, 5), replace=False)})
             for i in range(rng.integers(2, 6))}  # fmt: skip
    weights = rng.dirichlet(np.ones(len(items)))
    if rng.random() < 0.2:
        weights[0] = 0
        weights /= weights.sum()
    return synthetic(items), {item_id: float(weight) for item_id, weight in zip(items, weights)}


def whole_items(matrix: Matrix, weights: dict[str, float]) -> list[tuple[float, float]]:
    """The quality and cost per 1k of every way to send each item, whole, to one config: the brute force."""
    options = [[(weights[item_id] * cell.success, 1000 * weights[item_id] * cell.cost)
                for config in LADDER if (cell := matrix.cells.get((item_id, config)))]
               for item_id in matrix.items]  # fmt: skip
    return [(sum(quality for quality, _ in choice), sum(cost for _, cost in choice)) for choice in product(*options)]


def cheapest_whole(choices: list[tuple[float, float]], quality: float) -> float | None:
    """The brute force's cost at a target: its cheapest choice that reaches `quality`, or None if none does."""
    return min((cost for reached, cost in choices if reached >= quality - 1e-12), default=None)


def against_brute_force(rng: np.random.Generator) -> list[str]:
    """How the oracle on a random matrix departs from the brute force, if it does, at every corner, between each
    two, below the first and above the last."""
    matrix, weights = random_matrix(rng)
    solved, choices = oracle(matrix, weights), whole_items(matrix, weights)
    corners = list(solved.curve.corners.values())
    misses = []
    for corner in corners:  # where the greedy lands on whole items, it is a whole-item choice
        if not near(cheapest_whole(choices, corner.quality), corner.cost_per_1k):
            misses.append(f"at the corner {corner} the brute force costs {cheapest_whole(choices, corner.quality)}")
    for low, high in pairwise(corners):
        for quality in (low.quality + share * (high.quality - low.quality) for share in (0.1, 0.5, 0.9)):
            lp, whole = solved.cost_at(quality), cheapest_whole(choices, quality)
            # Rounding the fractional upgrade up to its whole step is a whole-item choice, so the gap is at most it.
            if not (lp is not None and whole is not None
                    and lp - 1e-9 <= whole <= lp + high.cost_per_1k - low.cost_per_1k + 1e-9):  # fmt: skip
                misses.append(f"at {quality} between {low} and {high}, the LP costs {lp} and the brute force {whole}")
    top = max(quality for quality, _ in choices)
    if not (math.isclose(corners[-1].quality, top, abs_tol=1e-12) and solved.cost_at(top + 1e-9) is None):
        misses.append(f"the oracle's maximum is {corners[-1].quality}, the brute force's {top}")
    if not near(solved.cost_at(corners[0].quality - 0.01), min(cost for _, cost in choices)):
        misses.append(f"below its first corner the oracle costs {corners[0].cost_per_1k}, not the cheapest choice's")
    return misses


def same_total(total: Total, quality: float, cost_per_1k: float) -> bool:
    return math.isclose(total.quality, quality, abs_tol=1e-12) and math.isclose(total.cost_per_1k, cost_per_1k,
                                                                                abs_tol=1e-12)  # fmt: skip


def same_points(curve: Curve, points: list[tuple[float, float]]) -> bool:
    """A curve's corners are these (quality, cost per 1k) points, in order."""
    corners = list(curve.corners.values())
    return len(corners) == len(points) and all(same_total(total, *point) for total, point in zip(corners, points))


def same_shares(shares: dict[str, float] | None, expected: dict[str, float]) -> bool:
    """An oracle's routing mix is these shares, in ladder order, every config named."""
    return shares is not None and list(shares) == list(LADDER) and all(
        math.isclose(shares[config], expected.get(config, 0), abs_tol=1e-12) for config in LADDER)  # fmt: skip


def near(answer: float | None, expected: float) -> bool:
    """A reference's answer is a number close to `expected`, not n/a (None)."""
    return answer is not None and math.isclose(answer, expected, rel_tol=1e-9)


def rounded(total: Total) -> tuple[str, str]:
    """As A8's table gives it: quality % to 1 decimal, cost per 1k to 2."""
    return f"{100 * total.quality:.1f}", f"{total.cost_per_1k:.2f}"


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

    print("-- totals on a hand-worked matrix --")
    # always-haiku: t1 is 1 of 2 right at $0.0015 a request, t2 0 of 1 at $0.003.
    check(same_total(totals(simulate(HAND, always(HAND, "haiku")), HAND_MIX), 0.375, 1.875),
          "always-haiku: quality 0.75 x 0.5 = 37.5%, cost 0.75 x $1.50 + 0.25 x $3 = $1.875 per 1k")  # fmt: skip
    check(same_total(totals(simulate(HAND, always(HAND, "opus")), HAND_MIX), 0.75, 6.25),
          "always-opus: quality 0.75 x 1 = 75%, cost 0.75 x $5 + 0.25 x $10 = $6.25 per 1k")  # fmt: skip
    routed = [Trace("a", ("haiku",), 0.0001), Trace("b", ("opus",), 0.0001), Trace("c", ("haiku",), 0.0001)]
    check(same_total(totals(simulate(HAND, routed), HAND_MIX), 0.75, 3.475),
          "a routed policy pays $0.10 per 1k overhead: quality 75%, cost 0.75 x $3.60 + 0.25 x $3.10 = $3.475 per 1k")  # fmt: skip
    refuses(MixError, lambda: totals(simulate(HAND, routed), {"t1": 1.0}),
            "a mix that doesn't weigh exactly the outcomes' tasks is refused")  # fmt: skip
    by_task = task_totals(simulate(HAND, always(HAND, "haiku")))
    check(list(by_task) == ["t1", "t2"] and same_total(by_task["t1"], 0.5, 1.5) and same_total(by_task["t2"], 0, 3),
          "always-haiku per task: t1 50% at $1.50 per 1k, t2 0% at $3")  # fmt: skip
    by_task = task_totals(simulate(HAND, routed))
    check(list(by_task) == ["t1", "t2"] and same_total(by_task["t1"], 1, 3.6) and same_total(by_task["t2"], 0, 3.1),
          "the routed policy per task, overhead included: t1 100% at $3.60 per 1k, t2 0% at $3.10")  # fmt: skip

    print("\n-- a hand-worked cascade --")
    # haiku answers first; a $0.0002 gate call accepts its answer on a and climbs to opus on b and c.
    cascade = [Trace("a", ("haiku",), 0.0002), Trace("b", ("haiku", "opus"), 0.0002),
               Trace("c", ("haiku", "opus"), 0.0002)]  # fmt: skip
    b = {outcome.item_id: outcome for outcome in simulate(HAND, cascade)}["b"]
    check((b.item_id, b.task, b.source_group, b.configs, b.success) == ("b", "t1", "b", ("haiku", "opus"), 1)
          and math.isclose(b.cost, 0.0082, rel_tol=1e-12),
          "b climbs from haiku to opus: it pays $0.002 + $0.006 + $0.0002 = $0.0082 and is right, as opus is "
          "and haiku isn't")  # fmt: skip
    check(same_total(totals(simulate(HAND, cascade), HAND_MIX), 0.75, 6.825),
          "under the mix: quality 75%, cost 0.375 x $1.20 + 0.375 x $8.20 + 0.25 x $13.20 = $6.825 per 1k")  # fmt: skip
    by_task = task_totals(simulate(HAND, cascade))
    check(same_total(by_task["t1"], 1, 4.7) and same_total(by_task["t2"], 0, 13.2),
          "per task: t1 100% at ($1.20 + $8.20) / 2 = $4.70 per 1k, t2 0% at $13.20")  # fmt: skip

    print("\n-- refused trace sets --")
    haiku = always(HAND, "haiku")
    refuses(TraceError, lambda: simulate(HAND, haiku[:2]), "a trace set that misses an item is refused")
    refuses(TraceError, lambda: simulate(HAND, [*haiku, Trace("a", ("opus",))]),
            "a trace set that covers an item twice is refused")  # fmt: skip
    refuses(TraceError, lambda: simulate(HAND, [*haiku, Trace("z", ("haiku",))]),
            "a trace for an item not in the matrix is refused")  # fmt: skip
    refuses(TraceError, lambda: simulate(HAND, [*haiku[:2], Trace("c", ("gpt",))]),
            "a trace naming an unknown config is refused")  # fmt: skip
    refuses(TraceError, lambda: simulate(HAND, [*haiku[:2], Trace("c", ("sonnet",))]),
            "a trace that calls a config with no stored answer for its item is refused")  # fmt: skip
    refuses(TraceError, lambda: simulate(HAND, [*haiku[:2], Trace("c", ())]), "a trace that calls no config is refused")
    refuses(TraceError, lambda: simulate(HAND, [*haiku[:2], Trace("c", ("haiku", "haiku"))]),
            "a trace that calls a config twice on one item is refused")  # fmt: skip
    check(len(simulate(TRAIN, [Trace("u", ("haiku", "opus_low")), Trace("v", ("haiku", "opus_low"))])) == 2,
          "a cascade on train items that stops below the frontier is evaluated")  # fmt: skip
    refuses(TraceError, lambda: simulate(TRAIN, [Trace("u", ("haiku",)), Trace("v", ("haiku", "opus"))]),
            "a trace that calls opus on a train item is refused, even where the matrix holds an opus answer")  # fmt: skip
    refuses(TraceError, lambda: simulate(TRAIN, [Trace("u", ("haiku", "opus")), Trace("v", ("haiku",))]),
            "opus on a train item with no opus answer gets the frontier refusal, not the no-stored-answer one",
            says="Amendment A1")  # fmt: skip

    print("\n-- a reference curve, both ways --")
    # Corners $1 80%, $3 86%, $7 90% per 1k: the first segment climbs 3 points a dollar, the second 1.
    curve = Curve({"haiku": Total(0.80, 1), "sonnet": Total(0.86, 3), "opus_low": Total(0.90, 7)})
    check(near(curve.cost_at(0.83), 2) and near(curve.cost_at(0.88), 5),
          "cost at a quality, along the straight line between corners: 83% costs $2 per 1k, 88% costs $5")  # fmt: skip
    check(near(curve.quality_at(2), 0.83) and near(curve.quality_at(5), 0.88),
          "quality at a cost: $2 per 1k reaches 83%, $5 reaches 88%")  # fmt: skip
    check((curve.cost_at(0.80), curve.cost_at(0.90), curve.quality_at(1), curve.quality_at(7)) == (1, 7, 0.80, 0.90),
          "both ends are in range: the first and last corners answer with each other's coordinate")  # fmt: skip
    check((curve.cost_at(0.79), curve.cost_at(0.91), curve.quality_at(0.99), curve.quality_at(7.01)) == (None,) * 4,
          "a query outside the curve's range gives n/a (None), below it and above it, both ways")  # fmt: skip
    refuses(CurveError, lambda: Curve({"haiku": Total(0.80, 1), "sonnet": Total(0.86, 3), "opus": Total(0.85, 7)}),
            "a curve whose quality falls from one corner to the next is refused")  # fmt: skip
    refuses(CurveError, lambda: Curve({"haiku": Total(0.80, 1), "sonnet": Total(0.86, 1)}),
            "a curve whose cost doesn't rise from one corner to the next is refused")  # fmt: skip
    refuses(CurveError, lambda: Curve({}), "a curve with no corners is refused")
    refuses(CurveError, lambda: Curve({"haiku": Total(0.80, math.nan)}),
            "a curve with a corner whose cost isn't a number is refused, even its only corner")  # fmt: skip
    refuses(CurveError, lambda: envelope({}), "the envelope of no config points is refused")
    refuses(CurveError, lambda: envelope({"haiku": Total(0.80, 1), "opus": Total(math.nan, 5)}),
            "the envelope of a point whose quality isn't a number is refused, though it would drop out of the hull")  # fmt: skip
    refuses(CurveError, lambda: envelope({"haiku": Total(0.80, 1), "gpt": Total(0.90, 5)}),
            "the envelope of a point that isn't a config on the ladder is refused")  # fmt: skip

    print("\n-- the random-mixture envelope on hand-worked points --")
    # Each config up the ladder buys less quality per dollar: 6, 2, then 0.5 points per $1 per 1k.
    bending = {"haiku": Total(0.80, 1), "sonnet": Total(0.86, 2), "opus_low": Total(0.90, 4), "opus": Total(0.92, 8)}
    check(tuple(envelope(bending).corners) == LADDER,
          "four configs whose quality per dollar keeps falling are all corners: a four-corner envelope")  # fmt: skip
    # Mixing haiku and opus_low half and half costs $3 per 1k for 85%, more than sonnet's 82% at $3.
    # opus costs more than opus_low for the same 90%.
    dropped = {"haiku": Total(0.80, 1), "sonnet": Total(0.82, 3), "opus_low": Total(0.90, 5), "opus": Total(0.90, 9)}
    check(tuple(envelope(dropped).corners) == ("haiku", "opus_low"),
          "a config below the hull (sonnet) and one costlier with no higher quality (opus) are both dropped")  # fmt: skip
    tied = {"sonnet": Total(0.80, 1), "haiku": Total(0.80, 1), "opus": Total(0.90, 5)}
    check(tuple(envelope(tied).corners) == ("haiku", "opus"),
          "two configs at exactly the same point: the corner is the lower one on the ladder (A8 item 5)")  # fmt: skip
    four_corners = envelope(bending)
    qualities = [float(quality) for quality in np.linspace(0.80, 0.92, 25)]  # every corner and points between
    check(all((paid := four_corners.cost_at(quality)) is not None and near(four_corners.quality_at(paid), quality)
              for quality in qualities),
          "a round trip, the cost at a quality and then the quality at that cost, returns the starting quality, "
          "from 80% to 92% in steps of half a point")  # fmt: skip
    past_ends = (four_corners.cost_at(0.7999), four_corners.cost_at(0.9201), four_corners.quality_at(0.999),
                 four_corners.quality_at(8.001))  # fmt: skip
    check(past_ends == (None,) * 4, "the envelope gives n/a below haiku's point and above opus's, both ways")

    print("\n-- the hindsight oracle on a hand-worked matrix --")
    # Under HAND_MIX, a and b weigh 0.375 each and c 0.25. a is right on haiku, so opus can't improve it; c is
    # wrong on every config. Only b has an upgrade: haiku to opus, +0.375 quality for 0.375 x $4 = $1.50 per 1k.
    hand = oracle(HAND, item_weights(HAND.items.values(), HAND_MIX))
    check(same_points(hand.curve, [(0.375, 1.875), (0.75, 3.375)]),
          "it starts at always-haiku's 37.5% at $1.875 per 1k and climbs to 75% at $3.375 by moving b to opus")  # fmt: skip
    check(near(hand.cost_at(0.5625), 2.625) and near(hand.quality_at(2.625), 0.5625),
          "halfway up, the last upgrade is fractional: 56.25% costs $1.875 + $1.50 / 2 = $2.625 per 1k")  # fmt: skip
    check(same_shares(hand.routing_at(0.5625), {"haiku": 0.8125, "opus": 0.1875}),
          "its routing mix there: half of b's 37.5% on opus, the rest on haiku")  # fmt: skip
    check(same_shares(hand.routing_at(0.75), {"haiku": 0.625, "opus": 0.375}),
          "at its maximum, c, which every config fails, stays on haiku, its cheapest config, with score 0")  # fmt: skip
    refuses(MatrixError, lambda: synthetic({"a": ("t1", {"haiku": (0.001, 0), "gpt": (0.002, 1)})}),
            "a matrix with a cell for a config off the ladder is refused, so the oracle can't skip it")  # fmt: skip

    print("\n-- the hindsight oracle with scores between 0 and 1 --")
    scored = oracle(SCORED, SCORED_WEIGHTS)
    check(same_points(scored.curve, SCORED_CORNERS),
          "p's four hull points give three upgrades and q's three give two, q's sonnet below its hull never used: "
          "corners at (20%, $1), (30%, $1.25), (75%, $3.50), (80%, $4), (95%, $7), (97.5%, $8) per 1k")  # fmt: skip
    check(near(scored.cost_at(0.525), 2.375)
          and same_shares(scored.routing_at(0.525), {"haiku": 0.375, "sonnet": 0.25, "opus_low": 0.375}),
          "52.5% is halfway through q's step to opus_low: $1.25 + $2.25 / 2 = $2.375 per 1k, p on sonnet")  # fmt: skip
    check(near(scored.quality_at(5.5), 0.875)
          and same_shares(scored.routing_at(0.875), {"opus_low": 0.625, "opus": 0.375}),
          "$5.50 per 1k is halfway through q's step to opus: 80% + 15% / 2 = 87.5%, p on opus_low")  # fmt: skip
    # Each task's own oracle weighs only its item. To reach 75%, p's climbs from sonnet ($2, 0.6) to 3/4 of the
    # way to opus_low ($4, 0.8): $3.50 per 1k. q's climbs 0.55 / 0.6 of the way from haiku ($1) to opus_low ($4):
    # $3.75 per 1k.
    own = task_oracles(SCORED)
    check(list(own) == ["t1", "t2"] and near(own["t1"].cost_at(0.75), 3.5) and near(own["t2"].cost_at(0.75), 3.75),
          "a per-task oracle for each task: t1 reaches 75% at $3.50 per 1k, t2 at $3.75")  # fmt: skip
    mixed = oracle(SCORED, item_weights(SCORED.items.values(), {"t1": 0.25, "t2": 0.75}))
    check(same_points(mixed.curve, SCORED_CORNERS) and near(mixed.cost_at(0.75), 3.5),
          "under the mix t1 0.25, t2 0.75 it trades quality between tasks: 75% with t1 at 60% and t2 at 80% costs "
          "$3.50 per 1k, not the $3.6875 of each task reaching 75% on its own")  # fmt: skip

    print("\n-- the hindsight oracle's ties and cheapest configs --")
    on_ties = oracle(TIES, item_weights(TIES.items.values(), {"t1": 1.0}))
    check(same_shares(on_ties.routing_at(0), {"haiku": 0.8, "sonnet": 0.2}),
          "every item starts on its own cheapest config: x on sonnet, the rest on haiku")  # fmt: skip
    # Steepest first: x to haiku (1 point per $1), u and v (1 per $2), y (1 per $3); each adds 20% and 0.2 x its $.
    check(same_points(on_ties.curve, [(0, 1), (0.2, 1.2), (0.4, 1.6), (0.6, 2), (0.8, 2.6)]),
          "its corners: (0%, $1), (20%, $1.20), (40%, $1.60), (60%, $2), (80%, $2.60) per 1k")  # fmt: skip
    check(same_shares(on_ties.routing_at(0.8), {"haiku": 0.4, "sonnet": 0.2, "opus_low": 0.2, "opus": 0.2}),
          "exact cost ties go to the lower config: y, right on both, to opus_low; z, wrong on every config, stays "
          "on haiku with score 0")  # fmt: skip
    check(same_shares(on_ties.routing_at(0.3), {"haiku": 0.9, "sonnet": 0.1}),
          "of steps equally steep, the one to the lower config goes first: at 30%, u is halfway to sonnet and v, "
          "though first in the matrix, still on haiku")  # fmt: skip

    print("\n-- the hindsight oracle out of range, and both ways --")
    check(near(hand.cost_at(0.2), 1.875) and near(hand.cost_at(0.375), 1.875)
          and same_shares(hand.routing_at(0.2), {"haiku": 1}),
          "a target at or below always-cheapest's 37.5% costs its $1.875 per 1k, every item on its cheapest config")  # fmt: skip
    check(near(hand.cost_at(0.75), 3.375) and hand.cost_at(0.7501) is None and hand.routing_at(0.7501) is None,
          "its maximum, 75%, costs $3.375 per 1k; a target above it is n/a (infeasible), cost and routing mix")  # fmt: skip
    check(hand.quality_at(1.8) is None and hand.quality_at(3.4) is None,
          "a cost below always-cheapest's or above the maximum's gives n/a")  # fmt: skip
    qualities = [float(quality) for quality in np.linspace(0.2, 0.975, 63)]  # every corner and points between
    costs = [float(cost) for cost in np.linspace(1, 8, 29)]
    check(all((paid := scored.cost_at(quality)) is not None and near(scored.quality_at(paid), quality)
              for quality in qualities),
          "a round trip from a quality, the cost at it and then the quality at that cost, returns the starting "
          "quality, from 20% to 97.5% in steps of 1.25 points")  # fmt: skip
    check(all((reached := scored.quality_at(cost)) is not None and near(scored.cost_at(reached), cost)
              for cost in costs),
          "a round trip from a cost, the quality at it and then the cost of that quality, returns the starting "
          "cost, from $1 to $8 per 1k in steps of $0.25")  # fmt: skip

    targets = quality_targets(0.9)
    check(list(targets) == [1.0, 0.99, 0.97] and all(map(near, targets.values(), (0.9, 0.891, 0.873))),
          "the quality targets at a frontier quality of 90%: 100%, 99% and 97% of it, 90%, 89.1% and 87.3% (§11)")  # fmt: skip

    print("\n-- the hindsight oracle's weights --")
    # A resample's weights leave out no item, but give 0 to an item it didn't draw.
    undrawn = oracle(HAND, {"a": 0.5, "b": 0, "c": 0.5})
    check(same_points(undrawn.curve, [(0.5, 2)]) and undrawn.cost_at(0.51) is None,
          "an item that weighs 0 is never upgraded, since that buys no quality: b stays on haiku, and the oracle "
          "is one point, a and c on haiku, 50% at $2 per 1k")  # fmt: skip
    refuses(OracleError, lambda: oracle(HAND, {"a": 0.5, "b": 0.5}), "weights that miss an item are refused")
    refuses(OracleError, lambda: oracle(HAND, {"a": 0.25, "b": 0.25, "c": 0.25, "z": 0.25}),
            "weights for an item not in the matrix are refused")  # fmt: skip
    refuses(OracleError, lambda: oracle(HAND, {"a": -0.25, "b": 0.75, "c": 0.5}),
            "a negative weight is refused, though the weights sum to 1")  # fmt: skip
    refuses(OracleError, lambda: oracle(HAND, {"a": math.nan, "b": 0.5, "c": 0.5}),
            "a weight that isn't a number is refused")  # fmt: skip
    refuses(OracleError, lambda: oracle(HAND, {"a": 0.5, "b": 0.5, "c": 0.5}),
            "weights that don't sum to 1 are refused, since quality is a weighted mean")  # fmt: skip
    unanswered = Matrix({**HAND.items, "w": Item("w", "dev", "t1", "w")}, HAND.cells)
    refuses(OracleError, lambda: oracle(unanswered, {"a": 0.25, "b": 0.25, "c": 0.25, "w": 0.25}),
            "an item with no stored answer is refused, since it has no config to start on",
            says="no stored answer")  # fmt: skip

    print("\n-- the hindsight oracle against a brute force --")
    rng = np.random.default_rng(6)
    misses = [miss for _ in range(BRUTE_FORCE_RUNS) for miss in against_brute_force(rng)]
    check(not misses,
          f"on {BRUTE_FORCE_RUNS} random matrices, the LP costs no more than the cheapest whole-item choice at any "
          f"target, and less by at most the upgrade in progress; they agree at every corner, at the maximum quality "
          f"and below the first corner {misses[:3]}")  # fmt: skip

    print("\n-- the dev results matrix --")
    rows = read_store()
    prices = load_models()["prices"]
    check(all(list_price(row, prices) == 2 * row["cost_usd"] for row in rows),
          f"every row's list-price cost, from its tokens, is exactly twice its stored batch cost ({len(rows):,} rows)")  # fmt: skip
    matrix = from_store(rows)
    check({item.split for item in matrix.items.values()} == {"train", "dev"}, "only train and dev items are loaded")
    dev = matrix.split("dev")
    tasks = Counter(item.task for item in dev.items.values())
    check(tasks == DEV_TASKS, f"dev items per task, as in Amendment A5 item 3 {dict(tasks)}")
    run0 = sum(row["run"] == 0 for row in rows)
    check(len(matrix.cells) == run0 == 744 * 4 + 991 * 3 and len(rows) - run0 == 600,
          "run 0 only: one cell per item and config (train without the frontier), the 600 noise-run rows left out")  # fmt: skip
    real = next(row for row in rows if row["split"] == "dev" and row["run"] == 0)
    refuses(MatrixError, lambda: from_store([{**real, "split": "test"}]), "a row outside train and dev is refused")
    refuses(MatrixError, lambda: from_store([{**real, "item_id": "ifeval:0"}]),
            "a row for an item outside the benchmark's train and dev is refused")  # fmt: skip
    refuses(MatrixError, lambda: from_store([{**real, "cost_usd": real["cost_usd"] * (1 + 1e-9)}]),
            "a row whose list-price cost isn't exactly twice its stored cost is refused")  # fmt: skip
    refuses(MatrixError, lambda: from_store([{**real, "price_snapshot": "2026-01-01"}]),
            "a row priced at another snapshot is refused")  # fmt: skip
    refuses(MatrixError, lambda: from_store([real, real]), "two run-0 rows for one item and config are refused")

    print("\n-- fixed configs on dev --")
    mixes = load_mixes()
    fixed = {config: simulate(dev, always(dev, config)) for config in LADDER}
    for config, expected in A8_TABLE.items():
        got = {mix: rounded(totals(fixed[config], mixes[mix])) for mix in expected}
        want = {mix: (f"{quality:.1f}", f"{cost:.2f}") for mix, (quality, cost) in expected.items()}
        shown = ", ".join(f"{mix} {quality}% ${cost}" for mix, (quality, cost) in got.items())
        check(got == want, f"{config}: {shown} per 1k, as in Amendment A8 item 1")

    print("\n-- the config ladder on dev --")
    fixed_by_task = {config: task_totals(outcomes) for config, outcomes in fixed.items()}
    ladder_costs = {task: [fixed_by_task[config][task].cost_per_1k for config in LADDER] for task in DEV_TASKS}
    ladder_costs |= {f"{mix} mix": [totals(fixed[config], mixes[mix]).cost_per_1k for config in LADDER]
                     for mix in ("uniform", "realistic")}  # fmt: skip
    for where, costs in ladder_costs.items():
        check(all(lower < higher for lower, higher in pairwise(costs)),
              f"{where}: {' < '.join(f'${cost:.2f}' for cost in costs)} per 1k, the ladder's order (A8 item 5)")  # fmt: skip
    item_cost = {key: cell.cost for key, cell in dev.cells.items()}
    for (lower, higher), share in A8_ABOVE_NEXT.items():
        above = sum(item_cost[(item_id, lower)] > item_cost[(item_id, higher)] for item_id in dev.items)
        got_share = f"{100 * above / len(dev.items):.1f}"
        check(got_share == share,
              f"{lower} costs more than {higher} on {above} items ({got_share}%), as in Amendment A8 item 5")  # fmt: skip
    ties = [(item_id, pair) for item_id in dev.items for pair in combinations(LADDER, 2)
            if item_cost[(item_id, pair[0])] == item_cost[(item_id, pair[1])]]  # fmt: skip
    pairs, tasks = Counter(pair for _, pair in ties), Counter(dev.items[item_id].task for item_id, _ in ties)
    check(pairs == A8_TIES and tasks["banking77"] == A8_TIES_BANKING77,
          f"exact cost ties between two configs on one item {dict(pairs)}, {tasks['banking77']} of them on "
          f"Banking77, as in Amendment A8 item 5")  # fmt: skip

    print("\n-- the random-mixture envelope on dev --")
    for mix in ("uniform", "realistic"):
        points = {config: totals(fixed[config], mixes[mix]) for config in LADDER}
        sonnet, opus_low, opus = points["sonnet"], points["opus_low"], points["opus"]
        on_dev = envelope(points)
        corners = ", ".join("{} {}% ${}".format(config, *rounded(total)) for config, total in on_dev.corners.items())
        check(tuple(on_dev.corners) == ("haiku", "opus_low"),
              f"{mix} mix: one segment, {corners} per 1k, as in Amendment A8 item 7")  # fmt: skip
        reached = on_dev.quality_at(sonnet.cost_per_1k)
        check(reached is not None and reached > sonnet.quality,
              f"{mix} mix: sonnet falls below it: at sonnet's ${sonnet.cost_per_1k:.2f} per 1k the envelope reaches "
              f"{100 * (reached or 0):.1f}%, sonnet {100 * sonnet.quality:.1f}%")  # fmt: skip
        check(opus.cost_per_1k > opus_low.cost_per_1k and opus.quality < opus_low.quality,
              f"{mix} mix: opus falls below it, costing more than opus_low for less quality")  # fmt: skip

    print("\n-- the hindsight oracle on dev --")
    right = {item_id: {config for config in LADDER if dev.cells[(item_id, config)].success} for item_id in dev.items}
    fail_everywhere = [item_id for item_id, configs in right.items() if not configs]
    beat_opus = [item_id for item_id, configs in right.items() if configs and "opus" not in configs]
    check((len(fail_everywhere), len(beat_opus)) == A8_ORACLE_ITEMS,
          f"{len(fail_everywhere)} items fail on every config, and on {len(beat_opus)} opus fails and a cheaper "
          f"config succeeds, as in Amendment A8 item 6")  # fmt: skip
    climbers = sum(bool(configs) and "haiku" not in configs for configs in right.values())
    for mix in ("realistic", "uniform"):
        weights = item_weights(dev.items.values(), mixes[mix])
        dev_oracle = oracle(dev, weights)
        steps = list(dev_oracle.curve.corners.values())
        b_point, a_point = totals(fixed["haiku"], mixes[mix]), totals(fixed["opus"], mixes[mix])
        check(same_shares(dev_oracle.routing_at(0), {"haiku": 1})
              and same_total(steps[0], b_point.quality, b_point.cost_per_1k),
              "{} mix: every item's cheapest config is haiku, so the oracle starts at B, always-haiku ({}%, ${} per "
              "1k)".format(mix, *rounded(steps[0])))  # fmt: skip
        check(len(steps) == 1 + climbers,
              f"{mix} mix: each item's hull has at most two points, haiku and the cheapest config that succeeds, so "
              f"there is one upgrade for each of the {climbers} items haiku fails and another config gets right")  # fmt: skip
        peak = steps[-1]
        check(math.isclose(peak.quality, a_point.quality + sum(weights[item_id] for item_id in beat_opus))
              and math.isclose(1 - peak.quality, sum(weights[item_id] for item_id in fail_everywhere)),
              "{} mix: its maximum, {}% at ${} per 1k, is opus's quality plus the weight of the items opus fails and "
              "a cheaper config gets right, and short of 100% by the weight of those every config fails"
              .format(mix, *rounded(peak)))  # fmt: skip
        reach = {share: dev_oracle.cost_at(target) for share, target in quality_targets(a_point.quality).items()}
        shown = ", ".join(f"{share:.0%} ${cost:.2f}" for share, cost in reach.items() if cost is not None)
        check(None not in reach.values(), f"{mix} mix: all three quality targets are feasible: {shown} per 1k")
    for task, task_oracle in task_oracles(dev).items():
        steps = list(task_oracle.curve.corners.values())
        b_point = fixed_by_task["haiku"][task]
        answerable = sum(bool(right[item_id]) for item_id in dev.items if dev.items[item_id].task == task)
        check(same_total(steps[0], b_point.quality, b_point.cost_per_1k)
              and math.isclose(steps[-1].quality, answerable / DEV_TASKS[task]),
              f"{task}'s own oracle starts at always-haiku's {100 * b_point.quality:.1f}% and peaks at "
              f"{100 * steps[-1].quality:.1f}%, the share of its items some config gets right")  # fmt: skip

    print("\n-- a toy cascade on dev --")
    outcomes = simulate(dev, toy_cascade(dev))
    by_task = task_totals(outcomes)
    gate = 1000 * TOY_GATE
    for task, total in by_task.items():
        always_haiku, always_opus = fixed_by_task["haiku"][task], fixed_by_task["opus"][task]
        quality, cost = rounded(total)
        if task in CLIMBS:
            check(same_total(total, always_opus.quality, always_haiku.cost_per_1k + always_opus.cost_per_1k + gate),
                  f"{task}: {quality}% ${cost} per 1k, opus's quality at haiku's cost plus opus's plus the gate's "
                  f"${gate:.2f}")  # fmt: skip
        else:
            check(same_total(total, always_haiku.quality, always_haiku.cost_per_1k + gate),
                  f"{task}: {quality}% ${cost} per 1k, haiku's quality at its cost plus the gate's ${gate:.2f}")  # fmt: skip
    for mix in ("uniform", "realistic"):
        total = totals(outcomes, mixes[mix])
        quality, cost = rounded(total)
        weighed = [sum(weight * getattr(by_task[task], measure) for task, weight in mixes[mix].items())
                   for measure in ("quality", "cost_per_1k")]  # fmt: skip
        check(same_total(total, *weighed),
              f"{mix} mix: {quality}% ${cost} per 1k, the mix's weighting of the per-task totals (A5 item 6)")  # fmt: skip

    print("\n-- the sealed store --")
    named = check_sealed.naming_files()
    check(not named, f"no code but harness/collect.py names the sealed store, analysis modules included {named}")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
