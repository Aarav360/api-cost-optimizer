"""Check G's and G+'s comparisons and diagnostics on dev (PLAN_v4.md §8 3c and 3c+; Amendment A15, items 10, 14 and 16;
Amendment A11, items 12 and 13; Amendment A7, item 2; issue #41).

Run from the project root:  uv run python -m analysis.check_jev_comparisons

No Jev call, no test item. The synthetic sections run anywhere (CI included); the stored-record sections run where
`results/` holds the dev matrix and `results/jev/` the stored dev responses (the hook, not CI).

- on a synthetic dev matrix whose every number is worked out by hand: the six same-target comparisons (G against E and
  F; G+ against E, F-train, F and G) with the fallback rules, including "both fall back to A"; the matched-cost
  difference from the envelope, a corner of it reading exactly 0 and a policy between two corners reading the linear
  interpolation; per-task McNemar rows against A, null where a point falls back; within-task concordance for a single
  model (cross-task pairs never counted) and the difference G+ less G; A7 item 2's warnings; the refusals;
- where `results/` holds the stored records: the walks reproducing the committed sections of E, F, G and G+; each
  same-target cell equal to the committed operating points' difference; concordance against a plain double loop over
  item pairs; McNemar's counts from the dev matrix by the decision rule alone; the matched-cost reading against an
  envelope built by hand from the fixed configs; dev latency (median, p90, counts, the first call and reconnects)
  recomputed from the raw stored lines; the data JSON-ready and reproducible; and nothing written to the operating-points
  file, whose allowlist still passes.
"""

import ast
import dataclasses
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from analysis import jev_comparisons as comparisons
from analysis.bootstrap import Estimate, resample
from analysis.matrix import Matrix
from analysis.router_names import F_TRAIN, G_PLUS, E, F, G
from analysis.router_table import ItemRouterScores, RouterScoreTable
from analysis.simulator import Outcome, always, simulate, totals
from analysis.walk import Step, Walk
from harness.client import PROJECT_ROOT
from routers.ladder import SCORED

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object], error: type[Exception] = comparisons.JevComparisonsError) -> bool:
    try:
        call()
    except error:
        return True
    return False


def near(got: float | None, want: float, tolerance: float = 1e-9) -> bool:
    return got is not None and abs(got - want) <= tolerance


# Two tasks of four items, each item its own source group. Each config costs the same on every item: haiku $1 per 1k,
# sonnet $2, opus_low $4, opus $10. Successes, by item a0-a3 (task t1) and b0-b3 (task t2):
SUCCESS = {
    "haiku": ([1, 1, 0, 0], [1, 0, 0, 0]),
    "sonnet": ([1, 1, 1, 0], [1, 1, 0, 0]),
    "opus_low": ([1, 1, 1, 1], [1, 1, 0, 0]),
    "opus": ([1, 1, 1, 1], [1, 1, 1, 1]),
}
COSTS = {"haiku": 0.001, "sonnet": 0.002, "opus_low": 0.004, "opus": 0.010}
# Quality under the uniform mix {t1: .5, t2: .5} and the realistic {t1: .75, t2: .25}, from the successes above:
# haiku .375 / .4375, sonnet .625 / .6875, opus_low .75 / .875, opus 1 / 1.
MIXES = {"uniform": {"t1": 0.5, "t2": 0.5}, "realistic": {"t1": 0.75, "t2": 0.25}}
QUALITY = {
    "A": {"uniform": 1.0, "realistic": 1.0},
    "X": {"uniform": 0.75, "realistic": 0.875},  # always opus_low: $4 per 1k, a corner of the envelope
    "S": {"uniform": 0.625, "realistic": 0.6875},  # always sonnet: $2 per 1k
    "Z": {"uniform": 0.75, "realistic": 0.875},  # opus_low's answers at $6 per 1k (overhead on top of $4)
}
COST = {"A": 10.0, "X": 4.0, "S": 2.0, "Z": 6.0}
SHARES = (1.0, 0.99, 0.97)
ITEMS = [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)]


def synthetic_dev() -> Matrix:
    rows = []
    for config, (first, second) in SUCCESS.items():
        for item, success in zip(ITEMS, first + second, strict=True):
            rows.append({"item_id": item, "split": "dev", "task": "t1" if item[0] == "a" else "t2",
                         "source_group": item, "config": config, "cost": COSTS[config], "success": float(success)})  # fmt: skip
    return Matrix.from_rows(rows)


def hand_walk(dev: Matrix, mix: dict[str, float], share: float, outcomes: list[Outcome] | None) -> Walk:
    """A walk whose operating point is `outcomes` (certified, margin 0), or A itself where there are none."""
    frontier = simulate(dev, always(dev, "opus"))
    a_total = totals(frontier, mix)
    steps = [Step(None, a_total.quality, a_total.cost_per_1k, 0.0, True, tuple(frontier))]
    if outcomes is not None:
        total = totals(outcomes, mix)
        steps.append(Step(0.5, total.quality, total.cost_per_1k, 0.0, True, tuple(outcomes)))
    return Walk(share, 0.0, tuple(steps))


def table_of(scores: dict[str, tuple[list[float], list[float]]]) -> RouterScoreTable:
    """A table whose router score for each config is given for a0-a3 then b0-b3."""
    entries = [
        ItemRouterScores(item, {c: (scores[c][0] + scores[c][1])[number] for c in SCORED}, 0.0)
        for number, item in enumerate(ITEMS)
    ]
    return RouterScoreTable.build(entries, ITEMS)


def synthetic_checks() -> None:
    dev = synthetic_dev()
    draws = resample(dev)
    always_x = simulate(dev, always(dev, "opus_low"))
    always_s = simulate(dev, always(dev, "sonnet"))
    pricey = [dataclasses.replace(o, cost=0.006) for o in always_x]  # opus_low's answers, $0.002 more per request
    policies: dict[str, dict[float, list[Outcome] | None]] = {
        E: dict.fromkeys(SHARES),
        F: {1.0: None, 0.99: always_x, 0.97: always_x},
        F_TRAIN: dict.fromkeys(SHARES, always_s),
        G: {1.0: None, 0.99: always_x, 0.97: always_x},
        G_PLUS: dict.fromkeys(SHARES, pricey),
    }
    walks = {
        name: {mix: {share: hand_walk(dev, MIXES[mix], share, o) for share, o in by_share.items()} for mix in MIXES}
        for name, by_share in policies.items()
    }
    # G scores haiku [.9 .8 .4 .2 | .5 .6 .4 .3]; G+ the first task reversed and the second all tied; the other configs
    # are tied everywhere, so they score 0.5 where a task holds a success and a failure.
    flat = ([0.5] * 4, [0.5] * 4)
    tables = {
        G: table_of({"haiku": ([0.9, 0.8, 0.4, 0.2], [0.5, 0.6, 0.4, 0.3]), "sonnet": flat, "opus_low": flat}),
        G_PLUS: table_of({"haiku": ([0.2, 0.4, 0.8, 0.9], [0.5] * 4), "sonnet": flat, "opus_low": flat}),
    }
    found = comparisons.compare(dev, draws, MIXES, walks, tables)

    print("-- the same-target comparisons, worked out by hand --")
    check(set(found.same_target) == set(comparisons.COMPARISONS) and len(comparisons.COMPARISONS) == 6
          and comparisons.COMPARISONS == ((G, E), (G, F), (G_PLUS, E), (G_PLUS, F_TRAIN), (G_PLUS, F), (G_PLUS, G)),
          "A15 item 14's six: G against E and F; G+ against E, F-train, F and G (never G against F-train)")  # fmt: skip
    # First less second, by hand from COST and QUALITY: (pair) -> share -> (first policy, second policy) or None.
    plan: dict[tuple[str, str], dict[float, tuple[str, str] | None]] = {
        (G, E): {1.0: None, 0.99: ("X", "A"), 0.97: ("X", "A")},
        (G, F): {1.0: None, 0.99: ("X", "X"), 0.97: ("X", "X")},
        (G_PLUS, E): dict.fromkeys(SHARES, ("Z", "A")),
        (G_PLUS, F_TRAIN): dict.fromkeys(SHARES, ("Z", "S")),
        (G_PLUS, F): {1.0: ("Z", "A"), 0.99: ("Z", "X"), 0.97: ("Z", "X")},
        (G_PLUS, G): {1.0: ("Z", "A"), 0.99: ("Z", "X"), 0.97: ("Z", "X")},
    }
    for pair, by_share in plan.items():
        right = True
        for mix in MIXES:
            for share, sides in by_share.items():
                cell = found.same_target[pair][mix][share]
                if sides is None:
                    right &= cell.both_fall_back and cell.cost is None and cell.quality is None
                    right &= cell.reading == "both fall back to A"
                    continue
                first, second = sides
                assert cell.cost is not None and cell.quality is not None
                right &= not cell.both_fall_back and cell.reading == ""
                right &= near(cell.cost.value, COST[first] - COST[second])
                right &= near(cell.quality.value, QUALITY[first][mix] - QUALITY[second][mix])
                right &= cell.cost.low <= cell.cost.value <= cell.cost.high
        check(right, f"{pair[0]} less {pair[1]}: each cost and quality is the hand-worked difference at every mix and "
                     f"target, and 'both fall back to A' only where both do")  # fmt: skip
    cell = found.same_target[(G, E)]["uniform"][0.99]
    check(
        near(cell.cost.low, -6.0) and near(cell.cost.high, -6.0) and comparisons.excludes_zero(cell.cost),  # type: ignore[union-attr]
        "G less E, uniform, 99%: $4 against A's $10 in every resample, so the interval is -6 to -6 and excludes 0",
    )
    cell = found.same_target[(G_PLUS, G)]["uniform"][0.99]
    check(
        near(cell.quality.value, 0.0) and not comparisons.excludes_zero(cell.quality),  # type: ignore[union-attr]
        "G+ less G, 99%: the same answers, so a quality difference of 0, which no interval excludes",
    )
    check(cell.first_fell_back is False and found.same_target[(G_PLUS, F)]["uniform"][1.0].second_fell_back,
          "a side that fell back is marked, as the existing fallback rule reads it")  # fmt: skip
    check(comparisons.excludes_zero(Estimate(0.2, 0.1, 0.3, 0.0))
          and not comparisons.excludes_zero(Estimate(0.1, 0.0, 0.3, 0.0))
          and comparisons.excludes_zero(Estimate(-0.2, -0.3, -0.1, 0.0))
          and not comparisons.excludes_zero(None),
          "known positive for the reading: an interval excludes 0 only if it sits wholly above or below it")  # fmt: skip

    print("\n-- the matched-cost difference from the envelope --")
    mc = found.matched_cost
    check(set(mc) == {G, G_PLUS}, "G and G+ get matched-cost rows, as E and F do in table v1")
    check(all(near(mc[G][m][s].envelope, QUALITY["X"][m]) and mc[G][m][s].difference == 0 and near(mc[G][m][s].high, 0, 1e-12)
              for m in MIXES for s in (0.99, 0.97)),
          "G at 99% and 97% is always-opus_low, a corner of the envelope: it reads its own quality, difference exactly 0 on dev, and no resample puts it above the envelope")  # fmt: skip
    check(all(near(mc[G][m][1.0].envelope, 1.0) and mc[G][m][1.0].difference == 0 for m in MIXES),
          "G at 100% falls back to A, the envelope's top corner: difference exactly 0")  # fmt: skip
    # $6 lies between opus_low's $4 and opus's $10: the envelope reads opus_low + (opus - opus_low) x 2/6.
    for mix, reading in (("uniform", 0.75 + 0.25 * 2 / 6), ("realistic", 0.875 + 0.125 * 2 / 6)):
        row = mc[G_PLUS][mix][0.99]
        check(near(row.envelope, reading) and near(row.difference, QUALITY["Z"][mix] - reading)
              and row.high is not None and row.low is not None and row.low <= row.high,
              f"G+, {mix}: at $6 per 1k the envelope reads {reading:.4f}, so the difference is "
              f"{QUALITY['Z'][mix] - reading:+.4f}")  # fmt: skip

    print("\n-- per-task McNemar rows against A --")
    m = found.mcnemar
    check(set(m) == {G, G_PLUS} and set(m[G]) == {"t1", "t2"}, "G and G+ get a row per task")
    check(all(m[G][t]["uniform"][1.0] is None and m[G][t]["realistic"][1.0] is None for t in ("t1", "t2")),
          "a point that falls back to A has nothing to test against: null")  # fmt: skip
    # Opus_low against A: task t1 right on every item (no discordant pair, p = 1); t2 right on 2 of 4, so A alone is
    # right on 2 and the exact two-sided p is 2 x (1/4) = 0.5.
    check(m[G]["t1"]["uniform"][0.99] == {"b": 0, "c": 0, "p": 1.0}
          and m[G]["t2"]["uniform"][0.99] == {"b": 0, "c": 2, "p": 0.5}
          and m[G_PLUS]["t2"]["realistic"][0.97] == {"b": 0, "c": 2, "p": 0.5},
          "t1: no discordant pair, p = 1; t2: A alone right on 2 items, p = 2 x 1/4 = 0.5, for G and G+")  # fmt: skip

    print("\n-- within-task concordance for G and G+ --")
    g, plus = found.concordance[G]["haiku"], found.concordance[G_PLUS]["haiku"]
    # G, haiku: t1 holds 2 successes and 2 failures, 4 pairs, all in order = 1; t2 holds 1 success (.5) against failures
    # (.6 .4 .3), 3 pairs, 2 in order = 2/3; pooled 6 of 7. G+: t1 reversed, 0 of 4; t2 tied, 1.5 of 3; pooled 1.5 of 7.
    check(near(g.pooled.value, 6 / 7) and near(g.per_task["t1"].value, 1.0) and near(g.per_task["t2"].value, 2 / 3)
          and g.size == {None: 7, "t1": 4, "t2": 3},
          "G, haiku: 4 of 4 pairs in task t1, 2 of 3 in t2, 6 of 7 pooled; cross-task pairs are never counted")  # fmt: skip
    check(near(plus.pooled.value, 1.5 / 7) and near(plus.per_task["t1"].value, 0.0)
          and near(plus.per_task["t2"].value, 0.5),
          "G+, haiku: 0 of 4 in t1, a tie (half credit) on 3 of 3 in t2, 1.5 of 7 pooled")  # fmt: skip
    gap = found.concordance_gap["haiku"]
    check(near(gap.pooled.value, 1.5 / 7 - 6 / 7) and near(gap.per_task["t1"].value, -1.0)
          and near(gap.per_task["t2"].value, 0.5 - 2 / 3),
          "G+ less G, haiku: the readings' difference, pooled and per task, over the same resamples")  # fmt: skip
    check(near(found.concordance[G]["sonnet"].pooled.value, 0.5) and near(found.concordance_gap["sonnet"].pooled.value, 0)
          and found.concordance[G]["opus_low"].per_task["t1"].value is None
          and near(found.concordance[G]["opus_low"].pooled.value, 0.5),
          "tied scores read 0.5, and a task with no failure (opus_low in t1) has no pair: n/a, not 0")  # fmt: skip

    print("\n-- the data the chart script reads --")
    data = comparisons.to_data(found)
    check(json.loads(json.dumps(data)) == data, "to_data is plain JSON-ready data (it round-trips through json)")
    warned = data["within_task_concordance"][G]["haiku"]["per_task"]  # type: ignore[index]
    check(warned["t1"]["warning"] is None and warned["t2"]["warning"] is None,
          "tasks other than IFEval and MBPP+ carry no A7 item 2 warning")  # fmt: skip
    cell = data["same_target"]["G less E"]["uniform"]["100%"]  # type: ignore[index]
    check(cell["reading"] == "both fall back to A" and cell["cost_per_1k"] is None and cell["quality"] is None,
          "a cell where both fall back to A reads so in the data, with no difference")  # fmt: skip
    check(data["matched_cost"][G_PLUS]["uniform"]["99%"]["envelope"] == mc[G_PLUS]["uniform"][0.99].envelope  # type: ignore[index]
          and data["mcnemar"][G]["t2"]["uniform"]["99%"] == {"b": 0, "c": 2, "p": 0.5}  # type: ignore[index]
          and data["mcnemar"][G]["t2"]["uniform"]["100%"] is None,  # type: ignore[index]
          "matched-cost and McNemar rows are in the data under their target")  # fmt: skip
    check(data["descriptive"] == comparisons.DESCRIPTIVE and "multiplicity" in comparisons.DESCRIPTIVE,  # type: ignore[index]
          "the data calls itself descriptive, with no multiplicity correction")  # fmt: skip

    print("\n-- the refusals --")
    check(refused(lambda: comparisons.compare(dev, draws, MIXES, {k: v for k, v in walks.items() if k != F}, tables)),
          "a walk missing for a router the comparisons name is refused whole")  # fmt: skip
    check(refused(lambda: comparisons.compare(dev, draws, MIXES, walks, {G: tables[G]})),
          "a router-score table missing for G+ is refused whole")  # fmt: skip
    check(refused(lambda: comparisons.compare(dev, draws, {"uniform": MIXES["uniform"]}, walks, tables)),
          "walks for a mix the comparisons were not given are refused whole")  # fmt: skip
    check(refused(lambda: comparisons.compare(dev, draws, MIXES, {**walks, E: {"uniform": walks[E]["uniform"]}}, tables)),
          "a router whose walks lack a mix is refused whole")  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# Where results/ holds the stored records


def calls_to(path: Path, name: str) -> bool:
    """Whether a module's syntax tree names `name` (a call, an import or an attribute), as the AST sees it."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
        if isinstance(node, ast.alias) and name in (node.name, node.asname):
            return True
    return False


def brute_concordance(table: RouterScoreTable, dev: Matrix, config: str) -> float:
    """Pooled within-task concordance by a plain double loop over item pairs, ties half: the independent count."""
    won = pairs = 0.0
    scores = {e.item_id: e.router_scores[config] for e in table.items}
    for i, one in dev.items.items():
        for j, other in dev.items.items():
            if one.task != other.task or dev.cells[(i, config)].success != 1 or dev.cells[(j, config)].success != 0:
                continue
            won += 1.0 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0.0
            pairs += 1
    return won / pairs


def val(x: float | None) -> float:
    assert x is not None
    return x


def real_checks() -> None:
    from analysis import collect_jev, jev_walk
    from analysis import jev_train_report as report
    from analysis.benchmark import read_rows
    from analysis.operating_points import OPERATING_POINTS_FILE, build_section, check_sections, read_section
    from analysis.references import envelope
    from analysis.router_names import NAMED_ROUTERS
    from analysis.simulator import Total, load_mixes
    from routers.decision import choose

    before = hashlib.sha256(OPERATING_POINTS_FILE.read_bytes()).hexdigest()
    result = comparisons.compute()
    dev, draws, mixes = result.dev, result.draws, load_mixes()
    section = {name: read_section(NAMED_ROUTERS[name]) for name in (E, F, G, G_PLUS)}

    print("\n-- the walks --")
    for name in (E, F, G, G_PLUS):
        check(build_section(result.walks[name]) == section[name],
              f"{name}: the walks here are its committed section of the operating-points file")  # fmt: skip
    check(result.walks[G_PLUS] == jev_walk.walks_of(G_PLUS, draws),
          "G+'s walks are the ones jev_walk gives, from the same resamples")  # fmt: skip

    print("\n-- the same-target comparisons against the committed operating points --")

    def point(name: str, mix: str, share: float) -> tuple[float, float]:
        """An operating point's cost per 1k and quality: from the committed section for a named router, from the walk
        for F-train, which has no section."""
        if name == F_TRAIN:
            step = result.walks[name][mix][share].operating
            return step.cost_per_1k, step.quality
        found = section[name][mix][f"{share:.0%}"]
        steps = {s["threshold"]: s for s in found["steps"]}
        return steps[found["threshold"]]["cost_per_1k"], steps[found["threshold"]]["quality"]

    cells = right = 0
    for (first, second), by_mix_cells in result.same_target.items():
        for mix, by_share_cells in by_mix_cells.items():
            for share, st in by_share_cells.items():
                cells += 1
                (cost_a, quality_a), (cost_b, quality_b) = point(first, mix, share), point(second, mix, share)
                if st.both_fall_back:
                    right += cost_a == cost_b and st.cost is None
                else:
                    assert st.cost is not None and st.quality is not None
                    right += (
                        near(st.cost.value, cost_a - cost_b)
                        and near(st.quality.value, quality_a - quality_b)
                        and st.cost.low <= st.cost.value <= st.cost.high
                    )
    check(cells == 36 and right == cells,
          f"all {cells} cells (6 comparisons x 2 mixes x 3 targets) are the committed points' differences, A's where a "
          f"side falls back")  # fmt: skip

    print("\n-- the matched-cost differences against an envelope built by hand --")
    ok = True
    for name in (G, G_PLUS):
        for mix, by_share in result.matched_cost[name].items():
            weights = {i: mixes[mix][it.task] / sum(1 for x in dev.items.values() if x.task == it.task)
                       for i, it in dev.items.items()}  # fmt: skip
            points = {c: Total(sum(weights[i] * dev.cells[(i, c)].success for i in dev.items),
                               1000 * sum(weights[i] * dev.cells[(i, c)].cost for i in dev.items))
                      for c in ("haiku", "sonnet", "opus_low", "opus")}  # fmt: skip
            curve = envelope(points)
            for share, row in by_share.items():
                cost, quality = point(name, mix, share)
                reading = curve.quality_at(cost)
                ok &= (reading is None and row.envelope is None) or (
                    reading is not None and near(row.envelope, reading) and near(row.difference, quality - reading)
                )
    check(ok, "G's and G+'s matched-cost readings are the committed point's quality less an envelope worked out from "
              "the dev matrix's fixed configs")  # fmt: skip

    print("\n-- McNemar's counts from the dev matrix by the decision rule alone --")
    ok, tested = True, 0
    for name in (G, G_PLUS):
        by_item = {e.item_id: e.router_scores for e in result.tables[name].items}
        for task, mcnemar_by_mix in result.mcnemar[name].items():
            for mix, mcnemar_by_share in mcnemar_by_mix.items():
                for share, row_m in mcnemar_by_share.items():
                    op = section[name][mix][f"{share:.0%}"]
                    if op["fell_back"]:
                        ok &= row_m is None
                        continue
                    only_router = only_a = 0
                    for item_id, item in dev.items.items():
                        if item.task != task:
                            continue
                        got = dev.cells[(item_id, choose(by_item[item_id], op["threshold"]))].success
                        a_right = dev.cells[(item_id, "opus")].success
                        only_router += got == 1 and a_right == 0
                        only_a += got == 0 and a_right == 1
                    n = only_router + only_a
                    p = (
                        min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_router, only_a) + 1)) / 2**n)
                        if n
                        else 1.0
                    )
                    ok &= row_m is not None and (row_m["b"], row_m["c"]) == (only_router, only_a)
                    ok &= row_m is not None and near(row_m["p"], p, 1e-12)
                    tested += 1
    check(ok and tested > 0, f"{tested} per-task rows: b, c and the exact two-sided p are counted again from the "
                             f"matrix, router score and threshold")  # fmt: skip

    print("\n-- within-task concordance --")
    ok = True
    for name in (G, G_PLUS):
        for config in SCORED:
            ok &= near(
                result.concordance[name][config].pooled.value,
                brute_concordance(result.tables[name], dev, config),
                1e-12,
            )
    check(ok, "G's and G+'s pooled concordance, per config, equals a plain double loop over item pairs within a task")
    check(all(near(result.concordance_gap[c].pooled.value,
                   val(result.concordance[G_PLUS][c].pooled.value) - val(result.concordance[G][c].pooled.value), 1e-12)
              for c in SCORED),
          "G+ less G is the readings' difference")  # fmt: skip

    print("\n-- dev latency, recomputed from the raw stored lines --")
    task_of = {r["item_id"]: r["source"] for r in read_rows(split="dev")}
    ok = True
    for router in collect_jev.ROUTERS:
        lines = [json.loads(x) for x in collect_jev.records_path(router, "dev").read_text().splitlines()]
        kept = [r for n, r in enumerate(lines) if n > 0 and not r["new_connection"]]
        seconds = [r["seconds"] for r in kept]
        lat = result.latency[router]
        ok &= near(lat.median, statistics.median(seconds), 1e-12) and near(
            lat.p90, float(np.percentile(seconds, 90)), 1e-12
        )
        ok &= lat.count == len(kept) and near(lat.first, lines[0]["seconds"], 1e-12)
        ok &= lat.reconnects == [r["seconds"] for n, r in enumerate(lines) if n > 0 and r["new_connection"]]
        for task, (median, p90, count) in lat.by_task.items():
            times = [r["seconds"] for r in kept if task_of[r["item_id"]] == task]
            ok &= near(median, statistics.median(times), 1e-12) and count == len(times)
        counts = result.counts[router]
        ok &= counts.retries == sum(r["attempts"] - 1 for r in lines) and counts.failed_attempts == sum(
            len(r["failed_attempts"]) for r in lines
        )
        ok &= counts.calls == 744 and counts.pass_two == 0
        ok &= near(report.latency(collect_jev.read_records(router, "dev"), task_of).median, lat.median, 0)
    check(ok, "each router's dev median, p90, per-task medians, counts, first call and reconnects match the raw lines")

    print("\n-- the data, and nothing written --")
    data = comparisons.to_data(result)
    check(json.loads(json.dumps(data)) == data, "to_data is plain JSON-ready data")
    check(
        data == comparisons.to_data(comparisons.compute()),
        "computed again from the stored records, the data is identical",
    )
    check(hashlib.sha256(OPERATING_POINTS_FILE.read_bytes()).hexdigest() == before,
          "the operating-points file is byte for byte as it was before the comparisons ran")  # fmt: skip
    try:
        check_sections(required=tuple(NAMED_ROUTERS.values()))
        allowed = True
    except ValueError:
        allowed = False
    check(allowed, "the allowlist still passes: sections for E, F, G and G+, and none for F-train")
    root = PROJECT_ROOT / "analysis"
    check(calls_to(root / "jev_walk.py", "write_section"), "known positive: the walk module names the section writer")
    check(not calls_to(root / "jev_comparisons.py", "write_section")
          and not calls_to(root / "jev_comparisons.py", "OPERATING_POINTS_FILE"),
          "the comparisons module names neither the section writer nor the file")  # fmt: skip


def main() -> int:
    synthetic_checks()
    have = (
        (PROJECT_ROOT / "results" / "dev_matrix.sqlite").exists()
        and all((PROJECT_ROOT / "results" / "jev" / f"{r}_dev.jsonl").exists() for r in ("jev", "jev_bolstered"))
    )  # fmt: skip
    if have:
        real_checks()
    else:
        print("\n(the dev matrix or the stored dev responses are not here: the stored-record sections are skipped)")
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
