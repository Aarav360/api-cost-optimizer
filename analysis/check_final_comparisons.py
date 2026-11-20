"""Check the 51 comparisons and the replication summary on a split (PLAN_v4.md Amendment A19, item 8; A17 item 22; issue #77, #89).

Run from the project root:  uv run python -m analysis.check_final_comparisons

On a hand-worked four-item matrix with resamples written out by hand, and made-up replication cells. No results/, no router or gate
on a real item, nothing reads test.

- **The comparisons:** every one of the 51 pairs, in each of the six mix-and-target cells, equals the hand-worked difference in
  cost per 1k and in quality, first less second: the value on the items, and the percentile interval over four resamples whose
  weights are written out from A5 item 6's rule (a task's weight shared among its drawn items), the percentile by its definition.
  A point the file says fell back is A's own outcomes, and where both sides fell back the cell reads "both fall back to A" with no
  numbers. A policy nobody evaluated is refused, and so is a pair with no point at a mix and target.
- **The order:** item 22's five groups hold 12, 12, 10, 5 and 12 pairs, and `haiku/Jev` against G and G+ stand 46th and 47th of
  the 51 in them, as the committed table v4 holds them (so a check of "item 22's order" can't also be one of "first"). The write-up's
  declared order (A19 item 8) puts those two first: `declared` holds the seven questions of `WRITEUP.md` in its order, with the
  headings and the pair counts the write-up states (1, 1, 10, 12, 12, 10, 5), each question's pairs in item 22's order, and `order`
  is the 51 pairs once each, the two first.
- **The replication summary:** on made-up dev and test cells worked by hand, including a sign flip, a test interval that includes 0
  with the right sign, a test cell with no interval, a dev interval that includes 0 (not listed) and a dev cell with none (not
  listed); cost and quality are counted apart, and only dev intervals that excluded 0 are listed. A test that lacks a dev cell is
  refused.
"""

import re
import sys
from collections.abc import Callable
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from analysis import final_comparisons as fc
from analysis.bootstrap import Bootstrap
from analysis.operating_eval import FrozenPoint
from analysis.references import TARGETS
from analysis.router_names import CASCADES, HAIKU, JEV_GATE, NAMED_ROUTERS, cascade_section
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


ITEMS = ("a1", "a2", "b1", "b2")
TASKS = ("a", "a", "b", "b")
COUNTS = [
    [1, 1, 1, 1],
    [2, 0, 1, 1],
    [0, 2, 2, 0],
    [1, 1, 0, 2],
]  # how often each item is drawn in each of four resamples
DRAWS = Bootstrap(ITEMS, TASKS, np.array(COUNTS))
MIXES = {"realistic": {"a": 0.75, "b": 0.25}, "uniform": {"a": 0.5, "b": 0.5}}
# What each of three made-up policies gets right on a1 a2 b1 b2, and what each call costs, in dollars.
SUCCESS = {"A": (1, 1, 0, 1), "P": (1, 0, 1, 1), "Q": (0, 1, 0, 1)}
COST = {"A": (0.004, 0.002, 0.006, 0.004), "P": (0.001, 0.003, 0.002, 0.001), "Q": (0.002, 0.001, 0.001, 0.003)}


def outcomes_of(name: str) -> list[Outcome]:
    return [Outcome(i, t, i, c, s, ("opus",)) for i, t, c, s in zip(ITEMS, TASKS, COST[name], SUCCESS[name])]


def percentile(values: list[float], share: float) -> float:
    """numpy's linear percentile by its definition: the value at rank share x (n - 1) between the sorted values."""
    ordered = sorted(Fraction(v).limit_denominator(10**9) for v in values)
    rank = Fraction(share) * (len(ordered) - 1)
    below = int(rank)
    above = min(below + 1, len(ordered) - 1)
    return float(ordered[below] + (rank - below) * (ordered[above] - ordered[below]))


def weights(mix: dict[str, float], draws: list[int] | None) -> list[float]:
    """A5 item 6: each task's weight shared among its drawn items (each task here has two items, and every resample draws two of
    each). `draws` None is the items as they are, one each."""
    return [mix[t] * (1 if draws is None else draws[i]) / 2 for i, t in enumerate(TASKS)]


def total(name: str, mix: dict[str, float], draws: list[int] | None, what: str) -> float:
    values = SUCCESS[name] if what == "quality" else tuple(1000 * c for c in COST[name])
    return sum(w * v for w, v in zip(weights(mix, draws), values))


def hand(first: str, second: str, mix: dict[str, float], what: str) -> dict[str, float]:
    diffs = [total(first, mix, c, what) - total(second, mix, c, what) for c in COUNTS]
    return {"value": total(first, mix, None, what) - total(second, mix, None, what),
            "low": percentile(diffs, 0.025), "high": percentile(diffs, 0.975)}  # fmt: skip


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9


# Every policy's six points: which made-up outcomes it has and whether the file says it fell back (then A's own).
SECTIONS = [*NAMED_ROUTERS.values(), *CASCADES]
ASSIGNED: dict[tuple[str, str, float], tuple[str, bool]] = {}
for s, section in enumerate(SECTIONS):
    for m, mix in enumerate(MIXES):
        for t, share in enumerate(TARGETS):
            fell = (s + m + t) % 4 == 0
            ASSIGNED[section, mix, share] = ("A" if fell else "PQ"[(s + m + t) % 2], fell)


def evaluated(without: str | None = None) -> dict[FrozenPoint, list[Outcome]]:
    return {
        FrozenPoint(section, mix, share, None if fell else 0.5, None, fell): outcomes_of(name)
        for (section, mix, share), (name, fell) in ASSIGNED.items()
        if section != without
    }


def comparisons() -> None:
    print("-- the 51 comparisons, each cell against the hand-worked difference --")
    cells = fc.cells_of(DRAWS, MIXES, evaluated())
    check(
        [len(pairs) for pairs in cells.values()] == [12, 12, 10, 5, 12],
        "item 22's five groups hold 12, 12, 10, 5 and 12 pairs",
    )
    wrong, both, one = [], 0, 0
    for group, by_pair in cells.items():
        for (first, second), by_mix in by_pair.items():
            for mix, by_share in by_mix.items():
                for share, cell in by_share.items():
                    (a, a_fell), (b, b_fell) = (
                        ASSIGNED[fc.section_of(first), mix, share],
                        ASSIGNED[fc.section_of(second), mix, share],
                    )
                    if a_fell and b_fell:
                        both += 1
                        if not (cell.reading == "both fall back to A" and cell.cost is None and cell.quality is None):
                            wrong.append((group, first, second, mix, share, "both fall back"))
                        continue
                    one += a_fell != b_fell
                    for what, got in (("cost", cell.cost), ("quality", cell.quality)):
                        want = hand(a, b, MIXES[mix], "quality" if what == "quality" else "cost")
                        if got is None or not all(close(getattr(got, k), want[k]) for k in want) or cell.reading:
                            wrong.append((group, first, second, mix, share, what))
    check(not wrong, f"all {sum(len(p) for p in cells.values())} x 6 cells equal the hand-worked cost and quality differences "
                     f"(value, low, high) {wrong[:2]}")  # fmt: skip
    check(
        both > 0 and one > 0,
        f"the made-up points include {both} cells where both sides fell back and {one} where one side did",
    )
    check(refused(fc.ComparisonsError, lambda: fc.cells_of(DRAWS, MIXES, evaluated(without=cascade_section(HAIKU, JEV_GATE)))),
          "a policy with no evaluated point is refused, not left out")  # fmt: skip
    partial = {
        p: o for p, o in evaluated().items() if not (p.section == "rules" and p.mix == "uniform" and p.share == 0.97)
    }
    check(
        refused(fc.ComparisonsError, lambda: fc.cells_of(DRAWS, MIXES, partial)),
        "a pair with no point at a mix and target is refused",
    )


def committed_order() -> list[str]:
    """Table v4's pairs in the order it holds them, which is item 22's: by group, then pair."""
    import json

    held = json.loads((PROJECT_ROOT / "figures" / "table_v4.json").read_text(encoding="utf-8"))["comparisons"]["groups"]
    return [pair for body in held.values() for pair in body["pairs"]]


def writeup_questions() -> list[tuple[str, int]]:
    """The declared questions' headings and stated counts, read from WRITEUP.md."""
    text = (PROJECT_ROOT / "WRITEUP.md").read_text(encoding="utf-8")
    body = text.split("### The declared questions", 1)[1].split("\n### ", 1)[0]
    found = re.findall(r"^#### (.+)\n\nComparisons: (\d+)\.", body, flags=re.MULTILINE)
    return [(heading, int(count)) for heading, count in found]


def order() -> None:
    print("\n-- item 22's order, and the write-up's declared order --")
    item_22 = committed_order()
    check(len(item_22) == 51 and len(set(item_22)) == 51, "table v4 holds 51 distinct pairs")
    check(item_22[45] == "haiku/Jev less G" and item_22[46] == "haiku/Jev less G+",
          "in item 22's own order haiku/Jev against G and G+ are the 46th and 47th of the 51")  # fmt: skip
    cells = fc.cells_of(DRAWS, MIXES, evaluated())
    groups = {group: {"pairs": {fc.pair_name(pair): None for pair in by_pair}} for group, by_pair in cells.items()}
    check([name for body in groups.values() for name in body["pairs"]] == item_22,
          "the pairs the code names are table v4's, in item 22's order")  # fmt: skip
    asked = fc.declared(groups)
    flat = [name for question in asked for name in question["pairs"]]
    check(
        [len(q["pairs"]) for q in asked] == [1, 1, 10, 12, 12, 10, 5],
        "the seven declared questions hold 1, 1, 10, 12, 12, 10 and 5 pairs",
    )
    check(flat[:2] == ["haiku/Jev less G", "haiku/Jev less G+"], "haiku/Jev against G and against G+ come first")
    check(
        sorted(flat) == sorted(item_22) and len(flat) == 51, "the declared order holds every one of the 51 pairs once"
    )
    check(all(q["pairs"] == [n for n in item_22 if n in set(q["pairs"])] for q in asked),
          "each question's pairs keep item 22's order")  # fmt: skip
    written = writeup_questions()
    check([(q["question"], len(q["pairs"])) for q in asked] == written and len(written) == 7,
          f"the questions' words and counts are the write-up's headings and 'Comparisons: n' lines {written}")  # fmt: skip
    data = fc.comparisons_data(cells)
    made = fc.declared_part(SimpleNamespace(cache={"built": {"comparisons": data}}))
    check(
        made["order"] == flat and made["declared"] == asked, "the declared part holds the questions and the flat order"
    )
    check(data["count"] == 51 and list(data["groups"]) == list(groups)
          and data["headline"] == ["haiku/Jev less G", "haiku/Jev less G+"],
          "the comparisons' data holds the groups in item 22's order, the headline and the count")  # fmt: skip


def cell(value: float, low: float, high: float) -> dict[str, float]:
    return {"value": value, "low": low, "high": high, "undefined": 0.0}


def both_cells(cost: dict | None, quality: dict | None) -> dict:
    return {"reading": "" if cost else "both fall back to A", "first_fell_back": False, "second_fell_back": False,
            "cost_per_1k": cost, "quality": quality}  # fmt: skip


def groups_of(cells: dict[tuple[str, str, str], dict]) -> dict:
    """Made-up groups: one group, pairs X less Y (`x`) and Z less Y (`z`), each with cells by mix and target."""
    out: dict = {"g": {"pairs": {}}}
    for (pair, mix, target), found in cells.items():
        out["g"]["pairs"].setdefault(pair, {}).setdefault(mix, {})[target] = found
    return out


def replication() -> None:
    print("\n-- the replication summary, worked by hand --")
    keys = [("X less Y", "realistic", "99%"), ("X less Y", "realistic", "97%"), ("X less Y", "uniform", "99%"),
            ("Z less Y", "realistic", "99%"), ("Z less Y", "uniform", "97%"), ("Z less Y", "uniform", "99%")]  # fmt: skip
    dev = groups_of(
        {
            keys[0]: both_cells(
                cell(2.0, 1.0, 3.0), cell(0.01, -0.02, 0.03)
            ),  # cost excludes 0 (above); quality includes 0
            keys[1]: both_cells(cell(-2.0, -3.0, -1.0), cell(-0.05, -0.08, -0.01)),  # cost below, quality below
            keys[2]: both_cells(cell(1.0, -0.5, 2.0), cell(0.04, 0.01, 0.06)),  # cost includes 0; quality above
            keys[3]: both_cells(cell(5.0, 2.0, 8.0), cell(-0.03, -0.05, -0.01)),  # cost above; quality below
            keys[4]: both_cells(None, None),  # both fall back: no interval
            keys[5]: both_cells(cell(1.5, 0.5, 2.5), cell(0.0, -0.01, 0.01)),  # cost above; quality includes 0
        }
    )
    test = groups_of(
        {
            keys[0]: both_cells(cell(1.8, 0.9, 2.7), cell(0.02, -0.01, 0.05)),  # cost: same sign, excludes 0
            keys[1]: both_cells(cell(1.0, -1.0, 3.0), cell(-0.02, -0.04, 0.01)),  # cost: flipped sign (point above 0);
            # quality: same sign (point below 0), interval includes 0
            keys[2]: both_cells(cell(-1.0, -2.0, 0.5), cell(0.03, 0.005, 0.05)),  # quality: same sign, excludes 0
            keys[3]: both_cells(
                cell(-3.0, -6.0, -1.0), cell(-0.04, -0.07, -0.002)
            ),  # cost: sign flipped and excluding 0
            # quality: same sign, excludes 0
            keys[4]: both_cells(None, None),
            keys[5]: both_cells(None, None),  # test has no interval where dev had one
        }
    )
    found = fc.replication(dev, test)
    # Dev intervals that excluded 0: cost in [0], [1], [3], [5] (4); quality in [1], [2], [3] (3).
    # Cost, same sign: [0] yes, [1] no (test +1.0, dev below), [3] no (test -3.0, dev above), [5] no cell: 1.
    #   same sign and excluding 0: [0] only: 1.
    # Quality, same sign: [1] yes (-0.02), [2] yes (0.03), [3] yes (-0.04): 3. Excluding 0 on test: [2], [3]: 2.
    want = {"cost_per_1k": {"dev_excluded": 4, "same_sign": 1, "same_sign_excluding_zero": 1},
            "quality": {"dev_excluded": 3, "same_sign": 3, "same_sign_excluding_zero": 2}}  # fmt: skip
    check(found["counts"] == want, f"the counts are the hand-worked ones {found['counts']}")
    check(len(found["listed"]) == 7 and all(row["dev"]["low"] > 0 or row["dev"]["high"] < 0 for row in found["listed"]),
          "only dev comparisons whose interval excluded 0 are listed (one row per measure and cell)")  # fmt: skip
    flipped = [r for r in found["listed"] if r["measure"] == "cost_per_1k" and r["target"] == "97%"]
    check(len(flipped) == 1 and not flipped[0]["same_sign"] and not flipped[0]["same_sign_excluding_zero"],
          "a sign flip is neither the same sign nor excluding 0 the same way")  # fmt: skip
    included = [r for r in found["listed"] if r["measure"] == "quality" and r["target"] == "97%"]
    check(len(included) == 1 and included[0]["same_sign"] and not included[0]["same_sign_excluding_zero"],
          "a test interval that includes 0 with the right sign has the same sign but does not replicate")  # fmt: skip
    gone = [r for r in found["listed"] if r["pair"] == "Z less Y" and r["target"] == "99%" and r["mix"] == "uniform"]
    check(len(gone) == 1 and gone[0]["test"] is None and not gone[0]["same_sign"],
          "a test cell with no interval where dev had one is listed and replicates nothing")  # fmt: skip
    missing = groups_of({k: v for k, v in [(k, test["g"]["pairs"][k[0]][k[1]][k[2]]) for k in keys[:5]]})
    check(refused(fc.ComparisonsError, lambda: fc.replication(dev, missing)), "a dev cell that test lacks is refused")
    check(fc.sign(0.0) == 0 and fc.excludes_zero(cell(0.0, 0.0, 1.0)) == 0 and fc.excludes_zero(cell(1.0, 1e-9, 2.0)) == 1,
          "an interval touching 0 does not exclude it")  # fmt: skip


def main() -> int:
    comparisons()
    order()
    replication()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
