"""Check the Stage 3 analysis package (PLAN_v4.md §7, Amendment A8).

Run from the project root:  uv run python -m analysis.check_simulator

On a synthetic matrix, where the right answer is worked out by hand:

- fixed configs' totals, and a routed policy's with overhead, under a task mix;
- the simulator refuses a trace set that misses an item or covers one twice, and a
  trace that names an unknown config, calls a config with no stored answer, calls no
  config or calls one twice, and evaluates none of the set;
- a mix that doesn't weigh exactly the outcomes' tasks is refused.

On the dev results matrix:

- every row's cost recomputed from its tokens at list price is exactly twice its
  stored batch cost, and the loader refuses a row where it isn't, a row outside train
  and dev, and a row for an item the benchmark doesn't hold in train or dev;
- only train and dev items are loaded, and only run 0;
- the four fixed configs' totals match Amendment A8 item 1's table under both mixes;
- no analysis module names the sealed store (check_sealed, over the whole repo).
"""

import math
import sys
from collections import Counter

from analysis.matrix import Matrix, MatrixError, from_store, list_price, read_store
from analysis.simulator import Total, Trace, TraceError, always, load_mixes, simulate, totals
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


def synthetic(items: dict[str, tuple[str, dict[str, tuple[float, int]]]]) -> Matrix:
    """{item_id: (task, {config: (list-price cost, success)})}: dev items, each its own source group."""
    return Matrix.from_rows([
        {"item_id": item_id, "split": "dev", "task": task, "source_group": item_id, "config": config,
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

    def refuses(error: type[Exception], evaluate, what: str) -> None:
        try:
            evaluate()
        except error as refusal:
            check(True, f"{what} ({refusal})")
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

    print("\n-- the sealed store --")
    named = check_sealed.naming_files()
    check(not named, f"no code but harness/collect.py names the sealed store, analysis modules included {named}")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
