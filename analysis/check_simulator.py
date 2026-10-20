"""Check the Stage 3 analysis package (PLAN_v4.md §7, Amendment A8).

Run from the project root:  uv run python -m analysis.check_simulator

On a synthetic matrix, where the right answer is worked out by hand:

- fixed configs' totals, and a routed policy's with overhead, under a task mix and per
  task;
- a cascade's accounting: a trace that climbs from haiku to opus pays for both calls
  and its overhead, and is right exactly when opus is;
- the simulator refuses a trace set that misses an item, covers one twice or has a
  trace for an item not in the matrix, and a trace that names an unknown config, calls
  a config with no stored answer, calls no config or calls one twice, calls configs
  down the config ladder, calls opus on a train item (with its own refusal, even where
  no opus answer is stored), or has an overhead that is negative, not a number or
  infinite, and evaluates none of the set;
- a mix that doesn't weigh exactly the outcomes' tasks is refused;
- a reference curve gives the cost at a quality and the quality at a cost along the
  straight lines between its corners, both ends included; a quality outside its range
  and a cost below it give n/a, and a cost past its last corner its top quality; it
  refuses no corners, a corner that isn't a finite number, and corners whose quality or
  cost doesn't rise; the envelope refuses no points, a point that isn't a finite number
  (even one that would drop out of the hull), and a point off the config ladder;
- the random-mixture envelope keeps four corners where each config buys less quality
  per dollar than the one below, drops a config below the hull and one costlier with no
  higher quality, gives an exact tie's corner to the lower config, returns the starting
  quality on a round trip, gives n/a past both ends of its qualities and below its
  cheapest point, and past its costliest point reaches that point's quality;
- the hindsight oracle's corners, its cost at a quality, its quality at a cost and its
  routing mix, with the last upgrade fractional, where an item every config fails stays
  on its cheapest config; with scores between 0 and 1, items with three and four hull
  points and a config below the hull never used; under a mix it trades quality between
  tasks, which each task's own oracle can't; every item starts on its own cheapest
  config, even where that isn't haiku; exact cost ties and equally steep upgrades go to
  the lower config; a target at or below always-cheapest's quality gets its cost, one
  above the maximum is n/a, and so is a cost below always-cheapest's, while a cost
  past the maximum's reaches the maximum; round trips from a
  quality and from a cost; an item that weighs 0 is never upgraded; it refuses weights
  that miss an item, name one not in the matrix, are negative, aren't a number or don't
  sum to 1, and an item with no stored answer; a matrix refuses a cell for a config off
  the ladder;
- the quality targets at a frontier quality of 90%: 100%, 99% and 97% of it (§11);
- on 300 random matrices, the oracle against a brute force over whole-item choices: the
  LP costs no more at any target, and less by at most the upgrade in progress; they
  agree at every corner, at the maximum quality and below the first corner;
- the captured fraction (§7) of a cascade with a gate, at 100% and 99%; 0 for the
  frontier at 100% and 1 for the oracle's own choice, even a rounding error cheaper; n/a
  where the policy misses the target and where the oracle saves nothing; a policy that
  meets a target for less than the oracle, or above its maximum, stops the run; the
  envelope's cheapest point that meets a target, and its captured fraction;
- the paired bootstrap's resamples keep each source group's items together, draw as many
  groups from each task as it has, with replacement, and are identical from the same
  seed whatever order the matrix holds its items in; the mix weights are applied after
  resampling, so each resample's weights sum to 1, and the oracle takes one resample's;
  a mix that doesn't weigh exactly the items' tasks is refused;
- its intervals and standard errors on four hand-drawn resamples, under a mix and per
  task, and on 400 coin flips, one or two to a source group, as the textbook standard
  error of a mean gives them;
- the paired difference from the frontier on the hand-drawn resamples; a policy less
  itself is exactly 0 with no width; on the coin flips, two policies' paired standard
  error is the one their disagreements give, well under each one's own; it refuses
  outcomes that miss an item, cover one twice or cover one not resampled;
- the captured fraction's paired intervals on the hand-drawn resamples, each with its
  own frontier, target and re-solved oracle, for a cascade, always-haiku (n/a ranked
  below its one number), the frontier (0 throughout) and the envelope rebuilt in each;
  a policy that misses its target in about 5% of resamples gets an n/a lower bound
  and the share it meets; on 40 random matrices, random multi-config policies with
  overhead and the envelope never capture more than 1 in any resample;
- the safety margin is the paired difference's standard error, and the frontier's is 0.

On the dev results matrix, where each number checked against Amendment A8 is read from
PLAN_v4.md itself, never copied into this script:

- every row's cost recomputed from its tokens at list price is exactly twice its
  stored batch cost, and the loader refuses a row where it isn't, a row outside train
  and dev, a row for an item the benchmark doesn't hold in train or dev, a row priced
  at another snapshot, and two run-0 rows for one item and config;
- only train and dev items are loaded, and only run 0;
- the four fixed configs' totals under both mixes are the rows of Amendment A8 item 1's
  table, and the check catches the rounding slip that table once carried;
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
- the paired bootstrap draws 2,000 resamples, identical from the same seed, and never
  splits MMLU-Pro's source groups of two; on the realistic mix, each config's own
  standard error, its paired standard error against opus and the bars an unpaired
  margin would set are as Amendment A8 item 9 gives them;
- under both mixes the frontier captures 0 at 100% in every resample, and the envelope's
  captured fraction at each target lies between 0 and 1 and meets it in every resample;
- a toy cascade (haiku, climbing to opus on MMLU-Pro and MBPP+) prints its per-task
  totals, each haiku's or opus's quality with the calls and gate it paid for, and its
  per-mix totals, each the mix's weighting of the per-task ones;
- chart v0's committed figures (analysis.pareto) are of dev, run 0, n = 744 at the
  snapshot's prices; their config points, under both mixes and per task, are the
  package's fixed-config totals to the last bit; both JSONs, rebuilt from the package
  with the same resamples, match the committed ones in every number; all three quality targets are feasible
  under both mixes; at each the oracle costs no more than the envelope; every captured
  fraction, interval bounds included, is at most 1; the items every config fails are
  Amendment A8 item 6's; and only IFEval's and MBPP+'s panels carry A7's warning, with
  their dev counts;
- every number in Amendment A8 item 12's record is the one figures/pareto.json gives, at
  the precision the record prints, and the check catches a slip in one number and a
  number it doesn't compare;
- no analysis module names the sealed store (check_sealed, over the whole repo).
"""

import json
import math
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import astuple
from itertools import combinations, pairwise, product, zip_longest
from operator import itemgetter

import numpy as np

from analysis.bootstrap import Bootstrap, BootstrapError, Captured, Estimate, resample
from analysis.matrix import LADDER, Item, Matrix, MatrixError, from_store, list_price, read_store
from analysis.pareto import FIGURE_DATA, PANELS, TASK_FIGURE_DATA, mix_panel, task_panels
from analysis.references import (
    BoundError,
    Curve,
    CurveError,
    OracleError,
    captured_fraction,
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
from harness.client import PROJECT_ROOT, load_models

DEV_TASKS = {"ifeval": 157, "banking77": 240, "mmlu_pro": 240, "mbppplus": 107}  # Amendment A5, item 3
# Amendment A7, item 2: the tasks whose per-task results carry a warning, which names the task's dev count.
A7_WARNED = {"ifeval", "mbppplus"}
# Amendment A8's numbers on dev are read from the plan itself, never copied here, so an edit there can't drift.
PLAN = PROJECT_ROOT / "PLAN_v4.md"


def synthetic(items: dict[str, tuple[str, dict[str, tuple[float, float]]]], split: str = "dev",
              groups: dict[str, str] | None = None) -> Matrix:  # fmt: skip
    """{item_id: (task, {config: (list-price cost, score)})}: items of one split, each its own source group unless
    `groups` gives it one."""
    return Matrix.from_rows([
        {"item_id": item_id, "split": split, "task": task, "source_group": (groups or {}).get(item_id, item_id),
         "config": config, "cost": cost, "success": success}
        for item_id, (task, cells) in items.items() for config, (cost, success) in cells.items()
    ])  # fmt: skip


# Three items, two tasks. Under the mix t1 0.75, t2 0.25, each t1 item weighs 0.375 and c weighs 0.25.
HAND = synthetic({
    "a": ("t1", {"haiku": (0.001, 1), "opus": (0.004, 1)}),
    "b": ("t1", {"haiku": (0.002, 0), "opus": (0.006, 1)}),
    "c": ("t2", {"haiku": (0.003, 0), "opus": (0.010, 0)}),
})  # fmt: skip
HAND_MIX = {"t1": 0.75, "t2": 0.25}
# Policies on HAND. ROUTED sends b to opus and a and c to haiku: the oracle's own choice at 75%, $3.375 per 1k.
# CASCADED climbs from haiku to opus on b and pays a gate of $0.0001 on every item: 0.375 x $1 + 0.375 x ($2 + $6) +
# 0.25 x $3 + $0.10 = $4.225 per 1k, also at 75%.
ROUTED = [Trace("a", ("haiku",)), Trace("b", ("opus",)), Trace("c", ("haiku",))]
CASCADED = [Trace("a", ("haiku",), 0.0001), Trace("b", ("haiku", "opus"), 0.0001), Trace("c", ("haiku",), 0.0001)]
# The frontier, always-opus, is at 75% and $6.25 per 1k; the oracle reaches 75% for $3.375 and 74.25% (99% of 75%)
# for $1.875 + 0.98 x $1.50 = $3.345. So at 100% the cascade captures ($6.25 - $4.225) / ($6.25 - $3.375) = 81/115
# of the oracle's savings, and at 99% $2.025 / $2.905. The envelope, haiku to opus, reaches 74.25% for $1.875 + 0.98 x
# $4.375 = $6.1625, capturing $0.0875 / $2.905.
HAND_CAPTURED = {"cascade at 100%": 81 / 115, "cascade at 99%": 2.025 / 2.905, "envelope at 99%": 0.0875 / 2.905}
# One item whose opus answer is both right and cheaper than haiku's: the oracle picks opus too, so it saves nothing.
NO_SAVINGS = synthetic({"s": ("t1", {"haiku": (0.002, 0), "opus": (0.001, 1)})})
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


# Source groups of three, two and one items in t1, and two groups of one in t2.
GROUPED = synthetic({item_id: (task, {"haiku": (0.001, 1)}) for item_id, task in
                     [("g1", "t1"), ("g2", "t1"), ("g3", "t1"), ("h1", "t1"), ("h2", "t1"), ("k", "t1"), ("m", "t2"),
                      ("n", "t2")]}, groups={"g1": "g", "g2": "g", "g3": "g", "h1": "h", "h2": "h"})  # fmt: skip
GROUPED_GROUPS = {"t1": [["g1", "g2", "g3"], ["h1", "h2"], ["k"]], "t2": [["m"], ["n"]]}
# Four resamples of HAND, drawn by hand: each draws two of t1's items (a, b) and t2's one (c). Under HAND_MIX, t1's
# 0.75 is shared among its draws, so their weights are a .75, b 0; then .375 each; b .75; .375 each. c weighs .25.
HAND_DRAWS = Bootstrap(("a", "b", "c"), ("t1", "t1", "t2"), np.array([[2, 0, 1], [1, 1, 1], [0, 2, 1], [1, 1, 1]]))
HAND_DRAW_WEIGHTS = [[0.75, 0, 0.25], [0.375, 0.375, 0.25], [0, 0.75, 0.25], [0.375, 0.375, 0.25]]
# Always-haiku over those resamples: quality 75%, 37.5%, 0, 37.5%, and $1.50, $1.875, $2.25, $1.875 per 1k. A
# percentile interpolates between the sorted values (numpy's default), so of four values the 2.5th sits 0.075 of
# the way from the first to the second and the 97.5th 0.925 of the way from the third to the fourth: quality 0 +
# 0.075 x 0.375 to 0.375 + 0.925 x 0.375. Both standard deviations are sqrt(2 x 0.375^2 / 3).
HAND_HAIKU = {"quality": (0.375, 0.028125, 0.721875), "cost_per_1k": (1.875, 1.528125, 2.221875)}
HAND_SPREAD = math.sqrt(2 * 0.375**2 / 3)
# Always-haiku's quality less always-opus's, resample by resample: 0, -37.5%, -75%, -37.5%; on the items as they
# are, 37.5% - 75%. Its interval and standard error are found the same way.
HAND_DIFFERENCE = (-0.375, -0.721875, -0.028125, HAND_SPREAD)
# Always-haiku within each task over the same resamples. t1's draws weigh a and b 1 and 0, then .5 each, 0 and 1, .5
# each: quality 100%, 50%, 0, 50% and $1, $1.50, $2, $1.50 per 1k, each with a standard deviation of sqrt(0.5^2 x 2 /
# 3). t2 draws c alone every time: 0 at $3. Percentiles as above.
HAND_TASK_SPREAD = math.sqrt(0.5**2 * 2 / 3)
HAND_TASKS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "t1": {"quality": (0.5, 0.0375, 0.9625, HAND_TASK_SPREAD), "cost_per_1k": (1.5, 1.0375, 1.9625, HAND_TASK_SPREAD)},
    "t2": {"quality": (0, 0, 0, 0), "cost_per_1k": (3, 3, 3, 0)},
}
# Each hand-drawn resample has its own frontier, target and oracle. Always-opus: 75% in each, at $5.50, $6.25, $7,
# $6.25 per 1k. At the 100% target, 75%, the oracle costs $1.50 (a and c on haiku already meet it, b undrawn), $3.375,
# $5.25 (only b, moved to opus), $3.375. The cascade costs $1.60, $4.225, $6.85, $4.225, so it captures 3.9 / 4,
# 81/115, 0.15 / 1.75, 81/115; percentiles as above. Always-haiku meets 75% only in the first, at $1.50: 4 / 4.
HAND_CASCADE_VALUES = sorted([3.9 / 4, 81 / 115, 0.15 / 1.75, 81 / 115])
# At 99% (74.25%) the envelope of haiku and opus is haiku alone in the first (it matches opus for less), so it meets
# the target at $1.50, as the oracle does: 1. In the others it captures $0.0875 / $2.905 (twice) and, from haiku at
# (0, $2.25) to opus at (75%, $7) against the oracle's $5.22, $0.0475 / $1.78.
HAND_ENVELOPE_VALUES = sorted([1, 0.0875 / 2.905, 0.0475 / 1.78, 0.0875 / 2.905])
# 400 items of one task, each its own source group, haiku right on the first 200 and opus on 20 to 219: each policy's
# quality has the standard error of a mean of 400 coin flips, sqrt(0.5 x 0.5 / 400) = 2.5 points. Their difference
# is +1 on 20 items, -1 on 20 and 0 on 360, so its standard error is sqrt((40 / 400) / 400) = 1.58 points. In COIN_PAIRS
# each pair of items (0 and 1, 2 and 3, ...) is one source group, so each policy's is that of 200 flips, 3.54 points.
COIN = {f"i{i:03}": ("t1", {"haiku": (0.001, float(i < 200)), "opus": (0.002, float(20 <= i < 220))})
        for i in range(400)}  # fmt: skip
COIN_SE, COIN_PAIRED_SE, COIN_PAIRS_SE = math.sqrt(0.25 / 400), math.sqrt(0.1 / 400), math.sqrt(0.25 / 200)
# 400 items, each its own source group: opus right on the first 200; haiku, at half the cost, loses 15 of those and
# gains 25 more. Haiku's paired lead over opus is 2.5 points with a standard error of sqrt((40 / 400 - 0.025^2) / 400)
# = 1.58, so it falls below the 100% target in about 5.6% of resamples (a normal tail at -1.59 standard errors). With
# a gate of $0.0002 a request it costs $1.20 per 1k; the oracle meets 50% with every item on haiku, for $1, and opus
# costs $2, so on dev it captures 0.8 / 1.
NEAR_MISS = {f"i{i:03}": ("t1", {"haiku": (0.001, float(15 <= i < 225)), "opus": (0.002, float(i < 200))})
             for i in range(400)}  # fmt: skip
NEAR_MISS_MET = (0.92, 0.97)  # the share of resamples meeting the target, around the normal tail's 94.4%
RANDOM_POLICY_RUNS, RANDOM_POLICY_RESAMPLES = 40, 50

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


def random_policies(rng: np.random.Generator) -> tuple[Matrix, dict[str, float], list[list[Trace]]]:
    """A matrix of 4 to 12 items in two tasks, every item on every config at $1 to $6 per 1k with 0/1 or quarter
    scores, some items sharing a source group; a random mix; and five random policies. Each item's trace calls 1 to
    4 random configs up the ladder, so most are cascades, with overhead of up to $0.002 a request."""
    scores = [0, 1] if rng.random() < 0.5 else [0, 0.25, 0.5, 0.75, 1]
    count = int(rng.integers(4, 13))
    items = {f"i{i:02}": ("t1" if i % 2 else "t2", {config: (int(rng.integers(1, 7)) / 1000, float(rng.choice(scores)))
                                                   for config in LADDER}) for i in range(count)}  # fmt: skip
    groups = {item_id: f"g{int(rng.integers(count))}" for item_id in items}
    matrix = synthetic(items, groups={item_id: f"{items[item_id][0]}-{group}" for item_id, group in groups.items()})
    share = float(rng.uniform(0.1, 0.9))
    policies = [[Trace(item_id, tuple(sorted((str(c) for c in rng.choice(LADDER, size=int(rng.integers(1, 5)), replace=False)),
                                            key=LADDER.index)),
                       float(rng.choice([0, 0.0001, 0.002]))) for item_id in matrix.items] for _ in range(5)]  # fmt: skip
    return matrix, {"t1": share, "t2": 1 - share}, policies


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


def same_estimate(estimate: Estimate, value: float, low: float, high: float, standard_error: float) -> bool:
    """An estimate is this value, interval and standard error, to rounding."""
    return all(math.isclose(got, want, rel_tol=0, abs_tol=1e-12) for got, want in
               zip((estimate.value, estimate.low, estimate.high, estimate.standard_error),
                   (value, low, high, standard_error)))  # fmt: skip


def same_captured(captured: Captured, value: float | None, low: float | None, high: float | None,
                  met: float) -> bool:  # fmt: skip
    """A captured fraction is this value and interval, n/a (None) where None is expected, and met in this share."""
    return math.isclose(captured.met, met, abs_tol=1e-12) and all(
        got is want if want is None else got is not None and math.isclose(got, want, rel_tol=0, abs_tol=1e-12)
        for got, want in zip((captured.value, captured.low, captured.high), (value, low, high)))  # fmt: skip


def of_four(values: list[float]) -> tuple[float, float]:
    """The percentile 95% interval of four sorted values, as worked by hand above HAND_HAIKU: 0.075 of the way from
    the first to the second, and 0.925 of the way from the third to the fourth."""
    return values[0] + 0.075 * (values[1] - values[0]), values[2] + 0.925 * (values[3] - values[2])


def plotted_point(config: dict) -> tuple[float, float]:
    """A figure JSON's point for a config: its quality and cost per 1k on dev, the values its intervals surround."""
    return config["quality"]["value"], config["cost_per_1k"]["value"]


def signed(points: float) -> str:
    """Points to 1 decimal, signed, with the plan's minus sign."""
    return f"{points:+.1f}".replace("-", "\u2212")


def record_lines(headline: dict) -> list[str]:
    """Amendment A8 item 12's lines that hold a number, as chart v0's JSON gives them, at the precision the record
    prints. None of the record's other lines holds one."""
    panels = [headline["panels"][mix] for mix in ("realistic", "uniform")]

    def row(label: str, cell: Callable[[dict], str]) -> str:
        return f"| {label} | {' | '.join(cell(panel) for panel in panels)} |"

    def point(config: str) -> Callable[[dict], str]:
        def cell(panel: dict) -> str:
            quality, cost = panel["configs"][config]["quality"], panel["configs"][config]["cost_per_1k"]
            return f"{100 * quality['value']:.1f}% ({100 * quality['low']:.1f} to {100 * quality['high']:.1f}), ${cost['value']:.2f}"  # fmt: skip

        return cell

    def at_targets(shown: Callable[[dict], str]) -> Callable[[dict], str]:
        return lambda panel: ", ".join(shown(target) for target in panel["targets"].values())

    def gap(panel: dict) -> str:
        return "{} points ({} to {})".format(*(signed(100 * panel["configs"]["opus_low"]["difference_from_A"][bound])
                                               for bound in ("value", "low", "high")))  # fmt: skip

    gaps = [panel["configs"]["opus_low"]["difference_from_A"] for panel in panels]
    holds = "contains 0 under both mixes" if all(d["low"] <= 0 <= d["high"] for d in gaps) else "doesn't contain 0"
    targets = ", ".join(panels[0]["targets"])
    return [
        f"12. **Record.** Chart v0's headline numbers on {headline['split']} (run {headline['run']}, n = "
        f"{headline['n']}, list prices of {headline['price_snapshot']}), from `figures/pareto.json`. Intervals are "
        f"percentile 95% from {headline['bootstrap']['resamples']:,} paired resamples, and "
        f"`analysis/check_simulator.py` compares every number here with that file.",
        row("A (`opus`): quality, $ per 1k", point("opus")),
        row("B (`haiku`): quality, $ per 1k", point("haiku")),
        row("`opus_low` \u2212 `opus`: paired quality difference", gap),
        row("D: maximum quality", lambda panel: f"{100 * panel['oracle_max_quality']:.1f}%"),
        row(f"D: $ per 1k at {targets} of A's quality", at_targets(lambda t: f"${t['oracle']['cost_per_1k']:.2f}")),
        row("D: savings against A there", at_targets(lambda t: f"{t['oracle']['savings_vs_A']['share']:.1%}")),
        row("C: captured fraction there", at_targets(lambda t: f"{t['envelope']['captured']['value']:.2f}")),
        f"- The paired interval of the `opus_low` \u2212 `opus` gap {holds}, so item 1's reading stands: the gap is "
        f"likely noise, and the frontier stays `opus`.",
    ]  # fmt: skip


def a8_item(plan: str, number: int) -> str:
    """Amendment A8 item `number`'s text, from its number to the next item's, or to the rule that ends the
    amendment."""
    amendment = plan[plan.index("## Amendment A8") :]
    amendment = amendment[: amendment.index("\n---\n")]
    start = amendment.index(f"\n{number}. **") + 1
    end = amendment.find(f"\n{number + 1}. **", start)
    return amendment[start : None if end == -1 else end]


def record_mismatches(record: list[str], headline: dict) -> list[tuple[str, str]]:
    """Each line of the record holding a number that differs from the line chart v0's JSON gives, paired with it; a
    line missing on either side is paired with ''. A digit after a letter is part of a name, as in v0 or A8."""
    held = [line.strip() for line in record if re.search(r"(?<![A-Za-z0-9_.])\d", line)]
    return [(says, gives) for says, gives in zip_longest(held, record_lines(headline), fillvalue="") if says != gives]


def rounded(total: Total) -> tuple[str, str]:
    """As A8's table gives it: quality % to 1 decimal, cost per 1k to 2."""
    return f"{100 * total.quality:.1f}", f"{total.cost_per_1k:.2f}"


def table_row(config: str, uniform: Total, realistic: Total) -> str:
    """A config's row of Amendment A8 item 1's table, as the plan prints it."""
    return "| `{}` | {}% | {} | {}% | {} |".format(config, *rounded(uniform), *rounded(realistic))


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
    refuses(TraceError, lambda: simulate(HAND, [*haiku[:2], Trace("c", ("opus", "haiku"))]),
            "a trace that calls configs down the config ladder is refused", says="ladder")  # fmt: skip
    # A negative overhead would make a policy look cheaper, and one that isn't a number would spoil every total.
    for what, overhead in (("negative", -0.0001), ("not a number", math.nan), ("infinite", math.inf)):
        refuses(TraceError, lambda: simulate(HAND, [*haiku[:2], Trace("c", ("haiku",), overhead)]),
                f"a trace whose overhead is {what} is refused", says="overhead")  # fmt: skip
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
    check((curve.cost_at(0.79), curve.cost_at(0.91), curve.quality_at(0.99)) == (None,) * 3,
          "a quality outside the curve's range gives n/a (None), below it and above it, and so does a cost below its "
          "first corner")  # fmt: skip
    check((curve.quality_at(7.01), curve.quality_at(100)) == (0.90, 0.90),
          "a cost past the last corner reaches the curve's top quality, 90%: a bigger budget needn't be spent")  # fmt: skip
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
    past_ends = (four_corners.cost_at(0.7999), four_corners.cost_at(0.9201), four_corners.quality_at(0.999))
    check(past_ends == (None,) * 3 and four_corners.quality_at(8.001) == 0.92,
          "the envelope gives n/a below haiku's point both ways and above opus's quality; past opus's cost it reaches "
          "opus's 92%, by picking opus alone")  # fmt: skip

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
    check(hand.quality_at(1.8) is None and hand.quality_at(3.4) == 0.75,
          "a cost below always-cheapest's $1.875 per 1k gives n/a; above its maximum's $3.375 the oracle reaches its "
          "maximum, 75%, without spending the rest")  # fmt: skip
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

    print("\n-- the captured fraction --")
    a_hand = totals(simulate(HAND, always(HAND, "opus")), HAND_MIX)
    as_oracle, cascade_total = (totals(simulate(HAND, traces), HAND_MIX) for traces in (ROUTED, CASCADED))
    check(same_total(a_hand, 0.75, 6.25) and same_total(cascade_total, 0.75, 4.225),
          "the frontier is at 75% and $6.25 per 1k, and the cascade, with its gate, at 75% and $4.225")  # fmt: skip
    check(captured_fraction(a_hand, a_hand, hand, 0.75) == 0,
          "the frontier's own captured fraction at the 100% target is 0: it saves nothing")  # fmt: skip
    check(near(captured_fraction(cascade_total, a_hand, hand, 0.75), HAND_CAPTURED["cascade at 100%"])
          and near(captured_fraction(cascade_total, a_hand, hand, 0.7425), HAND_CAPTURED["cascade at 99%"]),
          "the cascade captures ($6.25 - $4.225) / ($6.25 - $3.375) of the oracle's savings at 100%, and "
          "$2.025 / $2.905 at 99%, where the oracle costs $3.345 (§7)")  # fmt: skip
    check(near(captured_fraction(as_oracle, a_hand, hand, 0.75), 1),
          "a policy that makes the oracle's own choice captures all of its savings: 1")  # fmt: skip
    check(captured_fraction(Total(0.75, 3.375 - 1e-12), a_hand, hand, 0.75) == 1,
          "a policy a rounding error cheaper than the oracle counts as at its cost, so the fraction stays at most 1")  # fmt: skip
    b_hand = totals(simulate(HAND, always(HAND, "haiku")), HAND_MIX)
    check(captured_fraction(b_hand, a_hand, hand, 0.75) is None and captured_fraction(None, a_hand, hand, 0.75) is None,
          "n/a where the policy misses the target: always-haiku's 37.5% misses 75%, and so does a reference that can't "
          "reach it")  # fmt: skip
    a_lone = totals(simulate(NO_SAVINGS, always(NO_SAVINGS, "opus")), {"t1": 1.0})
    lone = oracle(NO_SAVINGS, {"s": 1.0})
    check(a_lone.quality == 1 and captured_fraction(a_lone, a_lone, lone, 1) is None,
          "n/a where the oracle saves nothing: when the frontier's answer is also the cheapest right one, the oracle "
          "pays the frontier's cost, and the frontier, though it meets the target, captures n/a, not 0 / 0")  # fmt: skip
    refuses(BoundError, lambda: captured_fraction(Total(0.75, 3.0), a_hand, hand, 0.75),
            "a policy that meets a target for less than the oracle's cost stops the run: its fraction would exceed 1")  # fmt: skip
    refuses(BoundError, lambda: captured_fraction(Total(0.8, 7), a_hand, hand, 0.8),
            "a policy that meets a target above the oracle's maximum stops the run")  # fmt: skip
    hand_envelope = envelope({"haiku": b_hand, "opus": a_hand})
    check(hand_envelope.meeting(0.3) == b_hand and hand_envelope.meeting(0.8) is None
          and same_total(hand_envelope.meeting(0.5) or Total(0, 0), 0.5, 1.875 + 4.375 / 3),
          "the envelope's cheapest point that meets a target: haiku's corner below its quality, the mix of haiku and "
          "opus at 50% ($1.875 + $4.375 / 3), and none above opus's 75%")  # fmt: skip
    check(near(captured_fraction(hand_envelope.meeting(0.7425), a_hand, hand, 0.7425), HAND_CAPTURED["envelope at 99%"])
          and captured_fraction(hand_envelope.meeting(0.75), a_hand, hand, 0.75) == 0,
          "the envelope captures $0.0875 / $2.905 of the oracle's savings at 99%, and none at 100%, which only "
          "always-opus reaches")  # fmt: skip

    print("\n-- the paired bootstrap's resamples --")
    grouped_draws = resample(GROUPED, 500, seed=1)
    column = {item_id: i for i, item_id in enumerate(grouped_draws.item_ids)}
    counts = {item_id: grouped_draws.counts[:, i] for item_id, i in column.items()}
    check(grouped_draws.counts.shape == (500, 8) and all(np.array_equal(counts[group[0]], counts[item_id])
                                                 for groups in GROUPED_GROUPS.values() for group in groups
                                                 for item_id in group),
          "500 resamples, and in every one a source group's items are drawn together, as often as each other")  # fmt: skip
    check(all(np.array_equal(sum(counts[group[0]] for group in groups), np.full(500, len(groups)))
              for groups in GROUPED_GROUPS.values()),
          "each resample draws as many source groups from each task as it has: 3 from t1, 2 from t2")  # fmt: skip
    check(all(len({int(n) for n in counts[group[0]]}) > 1 for groups in GROUPED_GROUPS.values() for group in groups),
          "groups are drawn with replacement: each group's count varies across resamples")  # fmt: skip
    check(np.array_equal(resample(GROUPED, 500, seed=1).counts, grouped_draws.counts)
          and not np.array_equal(resample(GROUPED, 500, seed=2).counts, grouped_draws.counts),
          "the same seed gives identical resamples, and another seed different ones")  # fmt: skip
    shuffled = Matrix(dict(reversed(GROUPED.items.items())), GROUPED.cells)
    again = resample(shuffled, 500, seed=1)
    check(all(np.array_equal(again.counts[:, again.item_ids.index(item_id)], counts[item_id]) for item_id in column),
          "the resamples don't depend on the order the matrix holds its items in")  # fmt: skip

    print("\n-- the paired bootstrap's weights --")
    check(np.allclose(HAND_DRAWS.weights(HAND_MIX), HAND_DRAW_WEIGHTS, rtol=0, atol=1e-12),
          "the mix weights are applied after resampling: a task's weight is shared among the items drawn from it, by "
          "how often each was drawn")  # fmt: skip
    first = HAND_DRAWS.item_weights(HAND_MIX, 0)
    check(first == {"a": 0.75, "b": 0, "c": 0.25} and same_points(oracle(HAND, first).curve, [(0.75, 1.5)]),
          "a resample's per-item weights, for the oracle to re-solve on: in the first, b isn't drawn and weighs 0, so "
          "the oracle never upgrades it, and a and c gain nothing from opus, leaving one point, 75% at $1.50 per 1k")  # fmt: skip
    resampled = grouped_draws.weights({"t1": 0.3, "t2": 0.7})
    in_t1 = np.array([task == "t1" for task in grouped_draws.tasks])
    check(np.allclose(resampled.sum(axis=1), 1, rtol=0, atol=1e-12)
          and np.allclose(resampled[:, in_t1].sum(axis=1), 0.3, rtol=0, atol=1e-12),
          "every resample's weights sum to 1, and t1's items to t1's mix weight, 0.3")  # fmt: skip
    refuses(MixError, lambda: HAND_DRAWS.weights({"t1": 1.0}),
            "a mix that doesn't weigh exactly the items' tasks is refused")  # fmt: skip

    print("\n-- the paired bootstrap's intervals --")
    hand_haiku, hand_opus = (simulate(HAND, always(HAND, config)) for config in ("haiku", "opus"))
    spread = HAND_DRAWS.totals(hand_haiku, HAND_MIX)
    check(all(same_estimate(getattr(spread, measure), *expected, HAND_SPREAD) for measure, expected in HAND_HAIKU.items()),
          "always-haiku over the four hand-drawn resamples: quality 37.5% (2.8% to 72.2%), cost $1.875 per 1k ($1.53 "
          "to $2.22), each with a standard error of sqrt(2 x 0.375^2 / 3)")  # fmt: skip
    steady = HAND_DRAWS.totals(hand_opus, HAND_MIX).quality
    check(same_estimate(steady, 0.75, 0.75, 0.75, 0),
          "always-opus is 75% in every hand-drawn resample, so its interval has no width and its standard error is 0")  # fmt: skip
    spread_by_task = HAND_DRAWS.task_totals(hand_haiku)
    check(list(spread_by_task) == list(HAND_TASKS)
          and all(same_estimate(getattr(spread_by_task[task], measure), *expected)
                  for task, measures in HAND_TASKS.items() for measure, expected in measures.items()),
          "always-haiku per task over the same resamples: t1 50% (3.75% to 96.25%) at $1.50 per 1k ($1.04 to $1.96), "
          "t2 0 at $3 in every one")  # fmt: skip
    for name, groups, own_se in (("one item per source group", None, COIN_SE),
                                 ("two items per source group", {f"i{i:03}": f"pair{i // 2}" for i in range(400)},
                                  COIN_PAIRS_SE)):  # fmt: skip
        coins = synthetic(COIN, groups=groups)
        flips = resample(coins).totals(simulate(coins, always(coins, "haiku")), {"t1": 1.0}).quality
        check(math.isclose(flips.standard_error, own_se, rel_tol=0.1)
              and abs(flips.low - (0.5 - 1.96 * own_se)) < 0.005 and abs(flips.high - (0.5 + 1.96 * own_se)) < 0.005,
              f"400 coin flips, {name}: 2,000 resamples give a standard error of {100 * flips.standard_error:.2f} "
              f"points (theory {100 * own_se:.2f}) and an interval of {100 * flips.low:.1f}% to "
              f"{100 * flips.high:.1f}% (theory 50% "
              f"± {196 * own_se:.1f})")  # fmt: skip

    print("\n-- the paired bootstrap's differences from the frontier --")
    check(same_estimate(HAND_DRAWS.difference(hand_haiku, hand_opus, HAND_MIX), *HAND_DIFFERENCE),
          "always-haiku less always-opus over the four hand-drawn resamples: -37.5 points (-72.2 to -2.8)")  # fmt: skip
    coins = synthetic(COIN)
    coin_draws = resample(coins)
    coin_haiku, coin_opus = (simulate(coins, always(coins, config)) for config in ("haiku", "opus"))
    itself = coin_draws.difference(coin_haiku, coin_haiku, {"t1": 1.0})
    check((itself.value, itself.low, itself.high, itself.standard_error) == (0, 0, 0, 0),
          "a policy less itself is exactly 0 in every resample: its interval has no width")  # fmt: skip
    paired = coin_draws.difference(coin_haiku, coin_opus, {"t1": 1.0})
    check(paired.value == 0 and math.isclose(paired.standard_error, COIN_PAIRED_SE, rel_tol=0.1),
          f"400 coin flips, two policies that disagree on 40 items: the paired difference's standard error is "
          f"{100 * paired.standard_error:.2f} points (theory {100 * COIN_PAIRED_SE:.2f}), well under each policy's "
          f"own {100 * COIN_SE:.2f}, since the luck of the draw moves both alike")  # fmt: skip
    for what, outcomes in (("miss an item", hand_haiku[1:]), ("cover an item twice", hand_haiku + hand_haiku[:1]),
                           ("cover an item not resampled", hand_haiku + simulate(TRAIN, always(TRAIN, "haiku"))[:1])):  # fmt: skip
        refuses(BootstrapError, lambda: HAND_DRAWS.totals(outcomes, HAND_MIX),
                f"a policy's outcomes that {what} are refused")  # fmt: skip
        refuses(BootstrapError, lambda: HAND_DRAWS.task_totals(outcomes),
                f"a policy's outcomes that {what} are refused per task too")  # fmt: skip
        refuses(BootstrapError, lambda: HAND_DRAWS.difference(hand_haiku, outcomes, HAND_MIX),
                f"frontier outcomes that {what} are refused")  # fmt: skip

    print("\n-- the captured fraction's paired intervals --")
    hand_savings = HAND_DRAWS.savings(HAND, hand_opus, HAND_MIX)
    hand_cascade = hand_savings.captured(simulate(HAND, CASCADED), 1.0)
    check(same_captured(hand_cascade, 81 / 115, *of_four(HAND_CASCADE_VALUES), 1),
          "the cascade over the four hand-drawn resamples, each with its own frontier, target and re-solved oracle: "
          "81/115 on dev, the interval of 0.975, 81/115, 0.086 and 81/115, and it meets the target in all four")  # fmt: skip
    check(same_captured(hand_savings.captured(hand_haiku, 1.0), None, None, 1, 0.25),
          "always-haiku meets 75% only in the first resample, where b isn't drawn, capturing 1: n/a on dev, and with "
          "n/a ranked below every number, 75% n/a makes the lower bound n/a but not the upper, and it meets the target "
          "in 25% of resamples")  # fmt: skip
    # 2,000 resamples of HAND drawn by hand, always-haiku missing 75% in some (b drawn) and capturing 1 in the rest.
    for missed, low in ((49, 1.0), (50, 1.0), (51, None)):
        edge = Bootstrap(("a", "b", "c"), ("t1", "t1", "t2"),
                         np.array([[1, 1, 1]] * missed + [[2, 0, 1]] * (2000 - missed)))  # fmt: skip
        check(same_captured(edge.savings(HAND, hand_opus, HAND_MIX).captured(hand_haiku, 1.0), None, low, 1,
                            1 - missed / 2000),
              f"n/a in {missed} of 2,000 resamples: the lower bound reads n/a only when more than 2.5% are "
              f"({'n/a' if low is None else low})")  # fmt: skip
    check(same_captured(hand_savings.captured(hand_opus, 1.0), 0, 0, 0, 1),
          "the frontier captures exactly 0 at the 100% target in every resample, where it meets its own quality")  # fmt: skip
    hand_fixed = {config: simulate(HAND, always(HAND, config)) for config in ("haiku", "opus")}
    check(same_captured(hand_savings.envelope_captured(hand_fixed, 0.99), 0.0875 / 2.905, *of_four(HAND_ENVELOPE_VALUES), 1),
          "the envelope at 99%, built afresh in each resample from the configs' points there: $0.0875 / $2.905 on "
          "dev, the interval of 1, 0.030, 0.027 and 0.030")  # fmt: skip

    near_miss = synthetic(NEAR_MISS)
    gated = simulate(near_miss, [Trace(item_id, ("haiku",), 0.0002) for item_id in near_miss.items])
    near_opus = simulate(near_miss, always(near_miss, "opus"))
    missing_some = resample(near_miss).savings(near_miss, near_opus, {"t1": 1.0}).captured(gated, 1.0)
    check(near(missing_some.value, 0.8) and missing_some.low is None and missing_some.high is not None
          and NEAR_MISS_MET[0] <= missing_some.met <= NEAR_MISS_MET[1],
          f"a gated policy 1.6 standard errors above the 100% target on dev meets it in {missing_some.met:.1%} of resamples, "
          f"reported beside its interval; the misses, more than 2.5%, rank below every number and aren't dropped, so "
          f"the lower bound reads n/a: {missing_some.value:.3f} (n/a to {missing_some.high:.3f})")  # fmt: skip
    rng = np.random.default_rng(8)
    fractions, breaches = [], []
    for _ in range(RANDOM_POLICY_RUNS):
        matrix_r, mix_r, policies = random_policies(rng)
        draws_r = resample(matrix_r, RANDOM_POLICY_RESAMPLES, seed=int(rng.integers(1000)))
        fixed_r = {config: simulate(matrix_r, always(matrix_r, config)) for config in LADDER}
        try:
            savings_r = draws_r.savings(matrix_r, fixed_r["opus"], mix_r)
            for target_share in (1.0, 0.99, 0.97, float(rng.uniform(0.5, 1))):
                fractions += [savings_r.captured(simulate(matrix_r, traces), target_share) for traces in policies]
                fractions.append(savings_r.envelope_captured(fixed_r, target_share))
        except BoundError as breach:
            breaches.append(str(breach))
    numbers = [n for f in fractions for n in (f.value, f.low, f.high) if n is not None]
    check(not breaches and max(numbers) <= 1 and sum(f.value is not None for f in fractions) > 100,
          f"{RANDOM_POLICY_RUNS} random matrices, {len(fractions)} random policies and envelopes at four targets, each in "
          f"{RANDOM_POLICY_RESAMPLES} resamples: no captured fraction exceeds 1 wherever the target is met, the "
          f"largest {max(numbers):.3f}, with {sum(f.value is not None for f in fractions)} met on the matrix itself "
          f"{breaches[:1]}")  # fmt: skip

    print("\n-- the safety margin --")
    check(math.isclose(HAND_DRAWS.safety_margin(hand_haiku, hand_opus, HAND_MIX), HAND_SPREAD, rel_tol=0, abs_tol=1e-12)
          and HAND_DRAWS.safety_margin(hand_opus, hand_opus, HAND_MIX) == 0,
          "one standard error of the policy's paired quality difference from the frontier: always-haiku's over the "
          "hand-drawn resamples is sqrt(2 x 0.375^2 / 3), and the frontier's own is 0 (Amendment A8, item 9)")  # fmt: skip

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
    plan = PLAN.read_text()
    mixes = load_mixes()
    fixed = {config: simulate(dev, always(dev, config)) for config in LADDER}
    table = a8_item(plan, 1)
    a8_rows = {config: table_row(config, totals(fixed[config], mixes["uniform"]), totals(fixed[config], mixes["realistic"]))
               for config in LADDER}  # fmt: skip
    for config, row in a8_rows.items():
        check(row in table, f"{config}: {row}, a row of Amendment A8 item 1's table in PLAN_v4.md")
    # A guard proves it can fail: the rounding slip item 1's table once carried, a cost's last digit one too high.
    slip = re.sub(r"(\d) \|$", lambda m: f"{(int(m[1]) + 1) % 10} |", a8_rows["opus_low"])
    check(a8_rows["opus_low"] not in table.replace(a8_rows["opus_low"], slip),
          f"the table check catches a slip in one number: {slip}")  # fmt: skip

    print("\n-- the config ladder on dev --")
    fixed_by_task = {config: task_totals(outcomes) for config, outcomes in fixed.items()}
    ladder_costs = {task: [fixed_by_task[config][task].cost_per_1k for config in LADDER] for task in DEV_TASKS}
    ladder_costs |= {f"{mix} mix": [totals(fixed[config], mixes[mix]).cost_per_1k for config in LADDER]
                     for mix in ("uniform", "realistic")}  # fmt: skip
    for where, costs in ladder_costs.items():
        check(all(lower < higher for lower, higher in pairwise(costs)),
              f"{where}: {' < '.join(f'${cost:.2f}' for cost in costs)} per 1k, the ladder's order (A8 item 5)")  # fmt: skip
    ladder = a8_item(plan, 5)
    item_cost = {key: cell.cost for key, cell in dev.cells.items()}
    above = {(lower, higher): f"{100 * sum(item_cost[(i, lower)] > item_cost[(i, higher)] for i in dev.items) / len(dev.items):.1f}"
             for lower, higher in (("sonnet", "opus_low"), ("opus_low", "opus"))}  # fmt: skip
    said = (f"`sonnet` costs more than `opus_low` on {above[('sonnet', 'opus_low')]}% of dev items, `opus_low` more "
            f"than `opus` on {above[('opus_low', 'opus')]}%")  # fmt: skip
    check(said in ladder, f"{said}, as Amendment A8 item 5 says")
    ties = [(item_id, pair) for item_id in dev.items for pair in combinations(LADDER, 2)
            if item_cost[(item_id, pair[0])] == item_cost[(item_id, pair[1])]]  # fmt: skip
    pairs, tasks = Counter(pair for _, pair in ties), Counter(dev.items[item_id].task for item_id, _ in ties)
    between = "all between `{}` and `{}`".format(*next(iter(pairs))) if len(pairs) == 1 else f"between {dict(pairs)}"
    said = f"on dev there are {len(ties)}, {between} ({tasks['banking77']} on Banking77"
    check(said in ladder, f"exact cost ties between two configs on one item: {said}), as Amendment A8 item 5 says")

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
    said = (f"{len(fail_everywhere)} dev items fail on every config and stay on `haiku` with score 0. On "
            f"{len(beat_opus)}, `opus` fails and a cheaper config succeeds.")  # fmt: skip
    check(said in a8_item(plan, 6), f"{said} As Amendment A8 item 6 says")
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

    print("\n-- the paired bootstrap on dev --")
    draws = resample(dev)
    sizes = Counter(item.source_group for item in dev.items.values())
    grouped = [[item_id for item_id, item in dev.items.items() if item.source_group == group]
               for group, size in sizes.items() if size > 1]  # fmt: skip
    place = {item_id: i for i, item_id in enumerate(draws.item_ids)}
    check(draws.counts.shape == (2000, 744) and np.array_equal(resample(dev).counts, draws.counts),
          "2,000 resamples of the 744 dev items, identical when drawn again from the same seed")  # fmt: skip
    check(len(grouped) == 5 and all(np.array_equal(draws.counts[:, place[a]], draws.counts[:, place[b]]) for a, b in grouped),
          "no resample splits any of MMLU-Pro's 5 source groups of two items")  # fmt: skip
    realistic = mixes["realistic"]
    dev_spread = {config: draws.totals(fixed[config], realistic) for config in LADDER}
    dev_se = {config: 100 * dev_spread[config].quality.standard_error for config in LADDER}
    margin = sum(dev_se.values()) / len(dev_se)
    margin_9 = a8_item(plan, 9)
    said = f"a fixed config's own standard error is about {margin:.1f} points"
    check(said in margin_9,
          f"realistic mix: {said} (on average; {', '.join(f'{config} {se:.2f}' for config, se in dev_se.items())}), "
          f"as Amendment A8 item 9 says")  # fmt: skip
    margins = {config: 100 * draws.safety_margin(fixed[config], fixed["opus"], realistic) for config in LADDER}
    paired_se = [margins[config] for config in LADDER[:-1]]
    said = f"its paired standard error against `opus` is {min(paired_se):.1f}\u2013{max(paired_se):.1f} points"
    check(said in margin_9 and margins["opus"] == 0,
          f"realistic mix: {said}, its safety margin ({', '.join(f'{c} {margins[c]:.2f}' for c in LADDER[:-1])}), as "
          f"Amendment A8 item 9 says, and opus's own is 0")  # fmt: skip
    # Item 9 adds the margin as it states it, to 1 decimal, to each quality target.
    a_quality = 100 * dev_spread["opus"].quality.value
    bars = [f"{target + round(margin, 1):.1f}%" for target in quality_targets(a_quality).values()]
    said = f"A {margin:.1f}-point margin would ask for dev quality of {bars[0]}, {bars[1]} and {bars[2]}"
    check(said in margin_9, f"realistic mix: an unpaired margin: {said}, as Amendment A8 item 9 says")

    print("\n-- the captured fraction on dev --")
    for mix in ("realistic", "uniform"):
        dev_savings = draws.savings(dev, fixed["opus"], mixes[mix])
        a_itself = dev_savings.captured(fixed["opus"], 1.0)
        check(same_captured(a_itself, 0, 0, 0, 1),
              f"{mix} mix: the frontier's own captured fraction at the 100% target is 0, in every resample")  # fmt: skip
        for target_share in (1.0, 0.99, 0.97):
            bar = dev_savings.envelope_captured(fixed, target_share)
            # 0 where opus is the envelope's cheapest point at the target, 1 where haiku alone meets it, as the oracle's
            # every-item-on-haiku point does: both happen in some resamples.
            check(bar.value is not None and bar.low is not None and bar.high is not None
                  and 0 <= bar.low <= bar.value <= bar.high <= 1 and bar.met == 1,
                  f"{mix} mix, {target_share:.0%} target: the envelope, the bar a router must clear, captures "
                  f"{bar.value:.3f} ({bar.low:.3f} to {bar.high:.3f}) of the oracle's savings, and meets the target in "
                  f"every resample")  # fmt: skip

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

    print("\n-- chart v0's figures --")
    # The committed JSONs, which every number quoted from chart v0 comes from, so a stale figure fails here too.
    headline, per_task = (json.loads(path.read_text()) for path in (FIGURE_DATA, TASK_FIGURE_DATA))
    snapshot = load_models()["price_snapshot"]
    check(all((data["split"], data["run"], data["n"], data["price_snapshot"]) == ("dev", 0, 744, snapshot)
              for data in (headline, per_task)),
          f"both figures are of dev, run 0, n = 744, at the list prices of {snapshot}")  # fmt: skip
    for mix in ("realistic", "uniform"):
        panel = headline["panels"][mix]
        plotted = {config: plotted_point(panel["configs"][config]) for config in panel["configs"]}
        check(plotted == {config: astuple(totals(fixed[config], mixes[mix])) for config in LADDER},
              f"{mix} mix: the figure's config points are the package's fixed-config totals, to the last bit")  # fmt: skip
        oracle_costs = {share: target["oracle"]["cost_per_1k"] for share, target in panel["targets"].items()}
        check(list(oracle_costs) == ["100%", "99%", "97%"] and None not in oracle_costs.values(),
              f"{mix} mix: all three quality targets are feasible on dev "
              f"({', '.join(f'{share} ${cost:.2f}' for share, cost in oracle_costs.items() if cost is not None)} per 1k)")  # fmt: skip
        envelope_costs = {share: target["envelope"]["cost_per_1k"] for share, target in panel["targets"].items()}
        check(all(oracle_costs[share] <= cost for share, cost in envelope_costs.items()),
              f"{mix} mix: at every target the oracle costs no more than the envelope "
              f"({', '.join(f'{share} ${cost:.2f}' for share, cost in envelope_costs.items())} per 1k)")  # fmt: skip
        bounds = [bound for target in panel["targets"].values()
                  for bound in itemgetter("value", "low", "high")(target["envelope"]["captured"])]  # fmt: skip
        check(all(bound <= 1 for bound in bounds if bound is not None),
              f"{mix} mix: every captured fraction, interval bounds included, is at most 1")  # fmt: skip
        check(panel["failed_by_every_config"] == len(fail_everywhere),
              f"{mix} mix: {panel['failed_by_every_config']} items fail on every config, as on the dev matrix")  # fmt: skip
    for task, panel in per_task["panels"].items():
        plotted = {config: plotted_point(panel["configs"][config]) for config in panel["configs"]}
        check(plotted == {config: astuple(fixed_by_task[config][task]) for config in LADDER},
              f"{task}: the per-task figure's config points are the package's per-task totals, to the last bit")  # fmt: skip
    # Rebuilt with the same resamples, both JSONs must match to the last bit: a figure left stale after the code or
    # the matrix changed fails here, whichever of its plotted or tabled numbers moved.
    for name, rebuilt, committed in (
        (FIGURE_DATA.name, {mix: mix_panel(dev, draws, fixed, mixes[mix]) for mix in PANELS}, headline["panels"]),
        (TASK_FIGURE_DATA.name, task_panels(dev, draws, fixed), per_task["panels"]),
    ):  # fmt: skip
        check(json.loads(json.dumps(rebuilt)) == committed,
              f"the committed figures/{name} is current: rebuilt from the package, every number it holds matches")  # fmt: skip
    warned = {task: panel["warning"] for task, panel in per_task["panels"].items() if panel["warning"]}
    check(list(per_task["panels"]) == list(DEV_TASKS) and set(warned) == A7_WARNED
          and all(f"{DEV_TASKS[task]} dev items" in warning for task, warning in warned.items()),
          f"the per-task figure has a panel per task, and only IFEval's and MBPP+'s carry Amendment A7's warning, "
          f"with their dev counts {warned}")  # fmt: skip

    print("\n-- chart v0's record (Amendment A8, item 12) --")
    record = a8_item(plan, 12).splitlines()
    mismatches = record_mismatches(record, headline)
    check(not mismatches,
          "every number in Amendment A8 item 12's record is the one figures/pareto.json gives, at the precision the "
          "record prints" + ("".join(f"\n      record: {says!r}\n      JSON:   {gives!r}" for says, gives in mismatches)))  # fmt: skip
    # A guard proves it can fail: a rounding slip in one number, as item 1's table once carried, and a number the
    # check doesn't know.
    a_row = next(i for i, line in enumerate(record) if line.strip().startswith("| A ("))
    slip = re.sub(r"\$(\d+\.\d)(\d)", lambda m: f"${m[1]}{(int(m[2]) + 1) % 10}", record[a_row], count=1)
    check(len(record_mismatches(record[:a_row] + [slip] + record[a_row + 1 :], headline)) == 1,
          f"the record check catches a slip in one number: {slip.strip()!r}")  # fmt: skip
    check(len(record_mismatches([*record, "    - One number more: 1.0 points."], headline)) == 1,
          "the record check catches a number it doesn't compare")  # fmt: skip

    print("\n-- the sealed store --")
    named = check_sealed.naming_files()
    check(not named, f"no code but harness/collect.py names the sealed store, analysis modules included {named}")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
