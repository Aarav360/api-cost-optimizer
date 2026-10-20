"""Check the Stage 3 analysis package (PLAN_v4.md §7, Amendment A8).

Run from the project root:  uv run python -m analysis.check_simulator

On a synthetic matrix, where the right answer is worked out by hand:

- fixed configs' totals, and a routed policy's with overhead, under a task mix and per
  task;
- a cascade's accounting: a trace that climbs from haiku to opus pays for both calls
  and its overhead, and is right exactly when opus is;
- the simulator refuses a trace set that misses an item or covers one twice, and a
  trace that names an unknown config, calls a config with no stored answer, calls no
  config or calls one twice, or calls opus on a train item (with its own refusal, even
  where no opus answer is stored), and evaluates none of the set;
- a mix that doesn't weigh exactly the outcomes' tasks is refused.

On the dev results matrix:

- every row's cost recomputed from its tokens at list price is exactly twice its
  stored batch cost, and the loader refuses a row where it isn't, a row outside train
  and dev, and a row for an item the benchmark doesn't hold in train or dev;
- only train and dev items are loaded, and only run 0;
- the four fixed configs' totals match Amendment A8 item 1's table under both mixes;
- a toy cascade (haiku, climbing to opus on MMLU-Pro and MBPP+) prints its per-task
  totals, each haiku's or opus's quality with the calls and gate it paid for, and its
  per-mix totals, each the mix's weighting of the per-task ones;
- no analysis module names the sealed store (check_sealed, over the whole repo).
"""

import math
import sys
from collections import Counter

from analysis.matrix import Matrix, MatrixError, from_store, list_price, read_store
from analysis.simulator import Total, Trace, TraceError, always, load_mixes, simulate, task_totals, totals
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


def synthetic(items: dict[str, tuple[str, dict[str, tuple[float, int]]]], split: str = "dev") -> Matrix:
    """{item_id: (task, {config: (list-price cost, success)})}: items of one split, each its own source group."""
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


CLIMBS, TOY_GATE = {"mmlu_pro", "mbppplus"}, 0.0001  # the toy cascade's tasks that climb to opus, its gate's cost


def toy_cascade(matrix: Matrix) -> list[Trace]:
    """haiku answers every item; a toy gate checks each answer and rejects every one on MMLU-Pro and MBPP+."""
    return [Trace(item_id, ("haiku", "opus") if item.task in CLIMBS else ("haiku",), TOY_GATE)
            for item_id, item in matrix.items.items()]  # fmt: skip


def same_total(total: Total, quality: float, cost_per_1k: float) -> bool:
    return math.isclose(total.quality, quality, abs_tol=1e-12) and math.isclose(total.cost_per_1k, cost_per_1k,
                                                                                abs_tol=1e-12)  # fmt: skip


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
    refuses(ValueError, lambda: totals(simulate(HAND, routed), {"t1": 1.0}),
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

    print("\n-- fixed configs on dev --")
    mixes = load_mixes()
    for config, expected in A8_TABLE.items():
        outcomes = simulate(dev, always(dev, config))
        got = {mix: rounded(totals(outcomes, mixes[mix])) for mix in expected}
        want = {mix: (f"{quality:.1f}", f"{cost:.2f}") for mix, (quality, cost) in expected.items()}
        shown = ", ".join(f"{mix} {quality}% ${cost}" for mix, (quality, cost) in got.items())
        check(got == want, f"{config}: {shown} per 1k, as in Amendment A8 item 1")

    print("\n-- a toy cascade on dev --")
    outcomes = simulate(dev, toy_cascade(dev))
    by_task = task_totals(outcomes)
    fixed = {config: task_totals(simulate(dev, always(dev, config))) for config in ("haiku", "opus")}
    gate = 1000 * TOY_GATE
    for task, total in by_task.items():
        always_haiku, always_opus = fixed["haiku"][task], fixed["opus"][task]
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
