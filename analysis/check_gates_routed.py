"""Check the router-started cascades and the fixed cutoffs (PLAN_v4.md §9; Amendment A17, items 11, 12 and 15).

Run from the project root:  uv run python -m analysis.check_gates_routed

On synthetic tables only, with every expected value worked out by hand: no results/ folder, no Docker, no Jev or
Anthropic call, no benchmark answer, and no test item.

- the fixed-cutoff rule (Youden's J over the distinct gate scores, ties to the higher cutoff, pooled over `haiku`'s,
  `sonnet`'s and `opus_low`'s train answers that pass the free checks);
- a router-started cascade's traces, walk and section.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

from analysis.bootstrap import resample
from analysis.check_simulator import synthetic
from analysis.check_walk import DRAWS, MIX, table_of
from analysis.checking import refused
from analysis.gate_cutoff import CutoffError, fixed_cutoff, youden_cutoff
from analysis.gate_table import GateRow, GateScoreTable, GateTableError, PaidScore
from analysis.gate_walk import MIXES
from analysis.operating_points import check_sections, read_section, write_section
from analysis.routed_walk import aligned_steps, routed_candidates, routed_sections, routed_traces
from analysis.router_names import CASCADES, FREE_GATE, JEV_GATE, NAMED_ROUTERS, SONNET_GATE, cascade_section
from analysis.router_table import RouterScoreTable
from analysis.simulator import Trace, simulate
from analysis.walk import step_thresholds, walk, walk_candidates

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def main() -> int:
    print("-- the fixed-cutoff rule: Youden's J --")
    # Wrong answers score .2, .4 and .6 and right ones .5, .8, .9 and .9 (3 wrong, 4 right). Rejecting below a cutoff:
    #   .2: 0 - 0 = 0;  .4: 1/3 - 0;  .5: 2/3 - 0;  .6: 2/3 - 1/4 = 5/12;  .8: 3/3 - 1/4 = 3/4;  .9: 3/3 - 2/4 = 1/2.
    one = [(0.2, False), (0.4, False), (0.6, False), (0.5, True), (0.8, True), (0.9, True), (0.9, True)]
    check(youden_cutoff(one) == 0.8, "the cutoff with the highest J, 3/4, is .8")
    check(youden_cutoff(one[::-1]) == 0.8, "the order of the answers does not matter")
    # Wrong .2, .6 and right .4, .8: J is 0 at .2, 1/2 at .4, 1/2 - 1/2 = 0 at .6, 1 - 1/2 = 1/2 at .8: a tie.
    tie = [(0.2, False), (0.6, False), (0.4, True), (0.8, True)]
    check(youden_cutoff(tie) == 0.8, "a tie goes to the higher cutoff, .8 not .4")
    # A refused call is always rejected. Adding one wrong refusal to the tie data: wrong .2, .6, refused; right .4, .8:
    #   .2: 1/3;  .4: 2/3;  .6: 2/3 - 1/2 = 1/6;  .8: 1 - 1/2 = 1/2. The best is .4.
    check(youden_cutoff([*tie, (None, False)]) == 0.4, "a refused call is a rejection that counts in both rates")
    check(youden_cutoff([(0.3, False), (0.7, True)]) == 0.7, "two answers: the cutoff is the right one's score")
    check(youden_cutoff([(0.5, False), (0.5, True)]) == 0.5, "one distinct score: that score, whatever J")
    check(refused(CutoffError, lambda: youden_cutoff([(0.5, True), (0.7, True)])), "no wrong answer: refused")
    check(refused(CutoffError, lambda: youden_cutoff([(0.5, False), (0.7, False)])), "no right answer: refused")
    check(refused(CutoffError, lambda: youden_cutoff([(None, False), (None, True)])), "nothing scored: refused")
    print("\n-- the cutoff from a train gate-score table --")
    train = synthetic({
        "a": ("t1", {"haiku": (0.001, 1), "sonnet": (0.002, 1), "opus_low": (0.004, 1)}),
        "b": ("t1", {"haiku": (0.001, 0), "sonnet": (0.002, 0), "opus_low": (0.004, 1)}),
        "c": ("t1", {"haiku": (0.001, 0), "sonnet": (0.002, 1), "opus_low": (0.004, 0)}),
        "d": ("t1", {"haiku": (0.001, 1), "sonnet": (0.002, 0), "opus_low": (0.004, 0)}),
        "e": ("t1", {"haiku": (0.001, 0), "sonnet": (0.002, 0), "opus_low": (0.004, 0)}),
    }, split="train")  # fmt: skip

    def row(item: str, config: str, score: float | None, failed: str | None = None) -> GateRow:
        if failed is not None:
            return GateRow(item, config, False, failed)
        return GateRow(item, config, True, None, jev=PaidScore(score, 0.0, score is None))

    base = [
        row("a", "haiku", .9), row("a", "sonnet", .8), row("a", "opus_low", .7),
        row("b", "haiku", .2), row("b", "sonnet", .6), row("b", "opus_low", .5),
        row("c", "haiku", .4), row("c", "sonnet", .35), row("c", "opus_low", .3),
        # d fails a free check on all three answers (haiku's is right, the others wrong): none is pooled.
        row("d", "haiku", None, "code"), row("d", "sonnet", None, "stop"), row("d", "opus_low", None, "schema"),
    ]  # fmt: skip
    table = GateScoreTable.build(base, [(r.item_id, r.config) for r in base])
    # Wrong: .2 .6 .4 .3 (4). Right: .9 .8 .7 .5 .35 (5). J at each cutoff, rejecting below it:
    #   .2: 0;  .3: 1/4;  .35: 2/4;  .4: 2/4 - 1/5 = 3/10;  .5: 3/4 - 1/5 = 11/20;  .6: 3/4 - 2/5 = 7/20;
    #   .7: 4/4 - 2/5 = 3/5;  .8: 1 - 3/5 = 2/5;  .9: 1 - 4/5 = 1/5.  The best is .7.
    check(fixed_cutoff(table, train, JEV_GATE) == 0.7, "Jev's cutoff from the table is the best J, .7 (3/5)")
    # Two refused wrong answers (e haiku, e sonnet) are rejected at every cutoff, so wrong is 6 and J is
    #   .2: 2/6;  .3: 3/6;  .35: 4/6;  .4: 4/6 - 1/5 = 13/30;  .5: 5/6 - 1/5 = 19/30;  .6: 5/6 - 2/5 = 13/30;
    #   .7: 1 - 2/5 = 18/30;  .8: 1 - 3/5 = 12/30;  .9: 1 - 4/5 = 6/30.  The best is .35 (20/30).
    refusals = [row("e", "haiku", None), row("e", "sonnet", None), row("e", "opus_low", None, "code")]
    both = base + refusals
    with_refusals = GateScoreTable.build(both, [(r.item_id, r.config) for r in both])
    check(fixed_cutoff(with_refusals, train, JEV_GATE) == 0.35,
          "two refused wrong answers count as rejected, which moves the cutoff to .35")  # fmt: skip
    check(refused(CutoffError, lambda: fixed_cutoff(table, train, FREE_GATE)), "the free checks have no cutoff")
    check(refused(CutoffError, lambda: fixed_cutoff(table, train, SONNET_GATE)),
          "a paid gate with no entry on a passing answer has none")  # fmt: skip
    short = synthetic({"a": ("t1", {"haiku": (0.001, 1)})}, split="train")
    check(
        refused(CutoffError, lambda: fixed_cutoff(table, short, JEV_GATE)), "an answer with no train outcome is refused"
    )
    odd = synthetic(
        {i: ("t1", {c: (0.001, 0.5) for c in ("haiku", "sonnet", "opus_low")}) for i in "abcde"}, split="train"
    )
    check(
        refused(CutoffError, lambda: fixed_cutoff(table, odd, JEV_GATE)), "a success that is neither 0 nor 1 is refused"
    )

    print("\n-- a router-started cascade's traces --")
    # Four dev items. Costs: haiku $0.001, sonnet $0.002, opus_low $0.004, opus $0.010. Right (1) or wrong (0):
    #            haiku sonnet opus_low opus
    #   i0          1     1      1       1
    #   i1          0     1      1       1
    #   i2          0     1      1       1
    #   i3          1     1      1       0      so opus (A) is at 75%, and each gate call costs $0.0002.
    ladder = {"haiku": 0.001, "sonnet": 0.002, "opus_low": 0.004, "opus": 0.010}
    wins = {"i0": (1, 1, 1, 1), "i1": (0, 1, 1, 1), "i2": (0, 1, 1, 1), "i3": (1, 1, 1, 0)}
    dev = synthetic({i: ("t1", {c: (ladder[c], w) for c, w in zip(ladder, ws)}) for i, ws in wins.items()})
    # The router's scores for haiku, sonnet and opus_low, and $0.00001 of overhead per item. Breakpoints: i0 .9; i1 .3,
    # .8; i2 .1, .6; i3 .2. So the steps are .1 .2 .3 .6 .8 .9, tried from .9 down.
    router = table_of({"i0": {"haiku": .9, "sonnet": .9, "opus_low": .9}, "i1": {"haiku": .3, "sonnet": .8, "opus_low": .8},
                       "i2": {"haiku": .1, "sonnet": .1, "opus_low": .6}, "i3": {"haiku": .2, "sonnet": .2, "opus_low": .2}},
                      overhead=0.00001)  # fmt: skip
    check(
        step_thresholds(router) == (0.1, 0.2, 0.3, 0.6, 0.8, 0.9), "the router's steps are its own: .1 .2 .3 .6 .8 .9"
    )
    # Jev's scores (each call $0.0002) at the fixed cutoff .5; the free checks fail i2 sonnet (stop) and i3 haiku (code).
    jev = {
        ("i0", "haiku"): 0.8,
        ("i1", "haiku"): 0.4,
        ("i1", "sonnet"): 0.7,
        ("i1", "opus_low"): 0.6,
        ("i2", "haiku"): 0.9,
        ("i2", "opus_low"): 0.3,
        ("i3", "sonnet"): 0.6,
        ("i3", "opus_low"): 0.5,
        ("i0", "sonnet"): 0.9,
        ("i0", "opus_low"): 0.9,
    }
    failed = {("i2", "sonnet"): "stop", ("i3", "haiku"): "code"}

    def gate_table(overhead: float = 0.0002) -> GateScoreTable:
        rows = [GateRow(i, c, (i, c) not in failed, failed.get((i, c)),
                        jev=PaidScore(jev[(i, c)], overhead, False) if (i, c) in jev else None)
                for i in wins for c in ladder]  # fmt: skip
        return GateScoreTable.build(rows, [(i, c) for i in wins for c in ladder])

    gates = gate_table()

    def traces_at(t: float, gate: str = JEV_GATE, cutoff: float | None = 0.5) -> list[Trace]:
        return routed_traces(router, gates, gate, cutoff, t)

    # At .9 the router picks haiku for i0 (.9) and opus for the rest, which get no gate and pay the router alone.
    check(traces_at(.9) == [Trace("i0", ("haiku",), 0.00001 + 0.0002), Trace("i1", ("opus",), 0.00001),
                            Trace("i2", ("opus",), 0.00001), Trace("i3", ("opus",), 0.00001)],
          "at .9: i0 stays on haiku (.8 at the cutoff .5) and pays router and gate; the rest go to opus and pay the router only")  # fmt: skip
    # At .6: i0 haiku, i1 sonnet (haiku .3 is under .6), i2 opus_low (.6 equals the threshold), i3 opus. i2's opus_low
    # scores .3, under the cutoff: it is rejected and goes to opus.
    check(traces_at(.6) == [Trace("i0", ("haiku",), 0.00021), Trace("i1", ("sonnet",), 0.00021),
                            Trace("i2", ("opus_low", "opus"), 0.00021), Trace("i3", ("opus",), 0.00001)],
          "at .6: a rejected opus_low answer goes to opus after paying the gate; an accepted sonnet answer stays")  # fmt: skip
    # At .2: i1's haiku (.3) scores .4, under .5: rejected. i3's haiku (.2) failed the code check: rejected with no gate.
    check(traces_at(.2) == [Trace("i0", ("haiku",), 0.00021), Trace("i1", ("haiku", "opus"), 0.00021),
                            Trace("i2", ("opus_low", "opus"), 0.00021), Trace("i3", ("haiku", "opus"), 0.00001)],
          "at .2: a failed free check escalates with no gate call, a low gate score escalates after one")  # fmt: skip
    # At .4... an answer scoring exactly the cutoff is accepted: i3's opus_low (.5) when the router picks it.
    check(routed_traces(table_of({"i3": {"haiku": 0.0, "sonnet": 0.0, "opus_low": .7}}, overhead=0.0), gates, JEV_GATE, .5, .7)
          == [Trace("i3", ("opus_low",), 0.0002)], "a gate score equal to the cutoff is accepted")  # fmt: skip
    check(traces_at(.9, FREE_GATE, None)[0] == Trace("i0", ("haiku",), 0.00001)
          and traces_at(.2, FREE_GATE, None)[3] == Trace("i3", ("haiku", "opus"), 0.00001),
          "free checks only: no gate call is paid, a failed check still escalates")  # fmt: skip
    check(refused(GateTableError, lambda: traces_at(.9, JEV_GATE, None)) and refused(GateTableError, lambda: traces_at(.9, FREE_GATE, 0.5)),
          "a paid gate needs its cutoff and the free checks alone take none")  # fmt: skip
    check([round(o.cost, 6) for o in simulate(dev, traces_at(.6))] == [0.00121, 0.00221, 0.01421, 0.01001]
          and [o.success for o in simulate(dev, traces_at(.6))] == [1, 1, 1, 0],
          "the simulator costs them: opus_low $0.004 + opus $0.010 + gate and router for i2, and returns the last answer")  # fmt: skip

    print("\n-- a router-started cascade's walk --")
    # Per item per 1k (mean x 1000): quality .75 from .9 down to .2 (the cascade's i1 haiku is caught by the gate);
    # .9: (.00121 + 3 x .01001) / 4 = $7.81;  .8: i1 on sonnet: (.00121 + .00221 + 2 x .01001) / 4 = $5.86;
    # .6: $6.91 (i2's opus_low is rejected);  .3: i1 on haiku, rejected: $9.16;  .2: $9.41.
    # .1: i2 on haiku, accepted at .9 and wrong: quality .5, which misses all three targets.
    w = walk_candidates(dev, routed_candidates(router, gates, JEV_GATE, 0.5), DRAWS, MIX, 1.0)
    check([s.threshold for s in w.steps] == [None, .9, .8, .6, .3, .2, .1] and [s.cutoff for s in w.steps] == [None] + [0.5] * 6
          and [s.clears for s in w.steps] == [True] * 6 + [False],
          "it walks the router's steps from .9 down, each at the fixed cutoff .5, and stops at .1, which loses i2")  # fmt: skip
    check([round(s.cost_per_1k, 4) for s in w.steps[:6]] == [10.0, 7.81, 5.86, 6.91, 9.16, 9.41]
          and all(math.isclose(s.quality, 0.75) for s in w.steps[:6]) and math.isclose(w.steps[6].quality, 0.5),
          "its costs and qualities are the hand-worked ones")  # fmt: skip
    check(w.operating.threshold == .8 and w.operating.cutoff == 0.5 and not w.fell_back,
          "the operating point is the cheapest certified step, .8, not the last one")  # fmt: skip
    check(all(s.margin == 0 for s in w.steps[:6]) and w.steps[6].margin > 0,
          "a step that differs from A nowhere has margin 0; the one that loses i2 has a margin")  # fmt: skip
    dear = gate_table(0.02)
    w = walk_candidates(dev, routed_candidates(router, dear, JEV_GATE, 0.5), DRAWS, MIX, 1.0)
    check(w.steps[1].clears and w.steps[1].cost_per_1k > w.steps[0].cost_per_1k and w.fell_back and w.operating.threshold is None,
          "a gate that costs more than it saves leaves every certified step dearer than A: falls back to A")  # fmt: skip
    check(len(routed_candidates(router, gates, JEV_GATE, 0.5)) == 6 and
          [c.threshold for c in routed_candidates(router, gates, FREE_GATE, None)] == [.9, .8, .6, .3, .2, .1]
          and all(c.cutoff is None for c in routed_candidates(router, gates, FREE_GATE, None)),
          "the free-checks-only cascade has the router's steps and no cutoff")  # fmt: skip

    print("\n-- a router alone and its /free cascade --")
    alone = walk(dev, router, DRAWS, MIX, 1.0)
    free = walk_candidates(dev, routed_candidates(router, gates, FREE_GATE, None), DRAWS, MIX, 1.0)
    pairs = aligned_steps(alone, free)
    check([(a.threshold, c.threshold) for a, c in pairs] == [(None, None), (.9, .9), (.8, .8), (.6, .6), (.3, .3)],
          "the router's steps line up one for one with the cascade's, A first")  # fmt: skip
    # Alone, i1 on haiku (.3) is wrong: quality .5, the walk stops at .3. Free checks pass it, so /free fails there too.
    check(len(alone.steps) == 5 and not alone.steps[4].clears and not free.steps[4].clears
          and all(math.isclose(a.quality, c.quality) for a, c in pairs[:4]), "with no gate they agree until the free checks matter")  # fmt: skip
    gated = walk_candidates(dev, routed_candidates(router, gates, JEV_GATE, 0.5), DRAWS, MIX, 1.0)
    pairs = aligned_steps(alone, gated)
    check(len(pairs) == 5 and math.isclose(pairs[4][0].quality, 0.5) and math.isclose(pairs[4][1].quality, 0.75) and pairs[4][1].clears,
          "with the Jev gate the cascade clears .3, where the router alone loses i1, and walks further")  # fmt: skip
    other = table_of({"i0": {"haiku": .9, "sonnet": .9, "opus_low": .9}, "i1": {"haiku": .5, "sonnet": .5, "opus_low": .5},
                      "i2": {"haiku": .5, "sonnet": .5, "opus_low": .5}, "i3": {"haiku": .5, "sonnet": .5, "opus_low": .5}})  # fmt: skip
    check(refused(ValueError, lambda: aligned_steps(alone, walk(dev, other, DRAWS, MIX, 1.0))),
          "walks over different thresholds are not a comparison: refused")  # fmt: skip

    print("\n-- a gate at cutoff 0 changes nothing --")
    # Every answer passes the free checks and scores .4, at no cost: a cutoff of 0 accepts all, so the cascade is the
    # router alone, step for step, in quality and cost.
    flat = GateScoreTable.build(
        [GateRow(i, c, True, None, jev=PaidScore(.4, 0.0, False)) for i in wins for c in ladder],
        [(i, c) for i in wins for c in ladder],
    )  # fmt: skip
    same = walk_candidates(dev, routed_candidates(router, flat, JEV_GATE, 0.0), DRAWS, MIX, 1.0)
    check(len(same.steps) == len(alone.steps) and all(a.threshold == c.threshold and math.isclose(a.quality, c.quality)
                                                      and math.isclose(a.cost_per_1k, c.cost_per_1k)
                                                      for a, c in zip(alone.steps, same.steps)),
          "a free, always-accepting gate at cutoff 0 walks exactly the router alone")  # fmt: skip

    print("\n-- the twelve sections --")
    tasks = ("ifeval", "banking77", "mmlu_pro", "mbppplus")
    items = [f"{t}{n}" for t in tasks for n in range(4)]
    big_dev = synthetic({i: (i.rstrip("0123"), {c: (ladder[c], int((k + j) % 3 != 0)) for j, c in enumerate(ladder)})
                         for k, i in enumerate(items)})  # fmt: skip
    keys = [(i, c) for i in items for c in ladder]
    big_rows = [GateRow(i, c, (k % 7) != 0, None if (k % 7) else "stop",
                        jev=PaidScore(0.3 + 0.01 * k, 0.0001, False) if k % 7 else None,
                        sonnet=PaidScore(0.9 - 0.01 * k, 0.0002, False) if k % 7 else None)
                for k, (i, c) in enumerate(keys)]  # fmt: skip
    big_table = GateScoreTable.build(big_rows, keys)

    def router_of(seed: int) -> RouterScoreTable:
        return table_of({i: {c: ((k * 7 + j * 3 + seed) % 20) / 20 for j, c in enumerate(("haiku", "sonnet", "opus_low"))}
                         for k, i in enumerate(items)}, overhead=0.00001 * seed)  # fmt: skip

    routers = {name: router_of(seed) for seed, name in enumerate(NAMED_ROUTERS.values(), start=1)}
    sections = routed_sections(big_dev, routers, big_table, {JEV_GATE: 0.45, SONNET_GATE: 0.7}, resample(big_dev))
    wanted = [
        cascade_section(name, gate) for name in NAMED_ROUTERS.values() for gate in (FREE_GATE, JEV_GATE, SONNET_GATE)
    ]
    check(list(sections) == wanted and len(sections) == 12 and set(sections) <= set(CASCADES),
          "twelve sections, E, F, G and G+ each with free, Jev and Sonnet, under the cascades' keys")  # fmt: skip
    free_point = sections[cascade_section(NAMED_ROUTERS["E"], FREE_GATE)]["realistic"]["100%"]
    check(set(free_point) == {"threshold", "margin", "target", "selection_method", "steps", "fell_back"}
          and all("cutoff" not in s for s in free_point["steps"]),
          "a /free section has a threshold and no cutoff anywhere")  # fmt: skip
    for gate, fixed in ((JEV_GATE, 0.45), (SONNET_GATE, 0.7)):
        by_mix = sections[cascade_section(NAMED_ROUTERS["G"], gate)]
        point = by_mix["uniform"]["99%"]
        check(set(point) == {"threshold", "cutoff", "margin", "target", "selection_method", "steps", "fell_back"}
              and point["steps"][0]["threshold"] is None and point["steps"][0]["cutoff"] is None
              and all(s["cutoff"] == fixed for s in point["steps"][1:]) and set(by_mix) == set(MIXES)
              and all(set(by_mix[m]) == {"100%", "99%", "97%"} for m in MIXES),
              f"a {gate} section has a threshold and the fixed cutoff {fixed} in every step but A's, for each mix and target")  # fmt: skip
    steps_of = {name: [s["threshold"] for s in sections[cascade_section(name, JEV_GATE)]["realistic"]["100%"]["steps"][1:]]
                for name in NAMED_ROUTERS.values()}  # fmt: skip
    prefix = sorted(step_thresholds(routers[NAMED_ROUTERS["F"]]), reverse=True)
    check(steps_of[NAMED_ROUTERS["F"]] == prefix[:len(steps_of[NAMED_ROUTERS["F"]])] and len(steps_of[NAMED_ROUTERS["F"]]) >= 1,
          "a section's steps are the router's own thresholds, strictest first")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "operating_points.json"
        for key, section in sections.items():
            write_section(key, section, out)
        check_sections(out, required=tuple(sections))
        check(True, "the allowlist accepts the twelve sections")
        check(all(read_section(k, out) == json.loads(json.dumps(v)) for k, v in sections.items()),
              "and each reads back as written")  # fmt: skip
    check(refused(ValueError, lambda: routed_sections(big_dev, dict(list(routers.items())[:3]), big_table,
                                                     {JEV_GATE: 0.45, SONNET_GATE: 0.7}, resample(big_dev))),
          "a missing router is refused")  # fmt: skip
    check(refused(ValueError, lambda: routed_sections(big_dev, routers, big_table, {JEV_GATE: 0.45}, resample(big_dev))),
          "a missing fixed cutoff is refused")  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
