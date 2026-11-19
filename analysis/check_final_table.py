"""Check the final results table, the per-policy tables, the comparisons' wiring and the replication summary on the rehearsal (PLAN_v4.md
§10; Amendment A19, items 6 and 8; A17 items 16, 22 and 24; issue #77, #89).

Run from the project root:  uv run python -m analysis.check_final_table

Needs no results/: the estimated latency is worked by hand on a made-up gate-score table, and the rest runs on the rehearsal's
made-up world (the offline half on 8 made-up test items, the real frozen operating points, a stand-in sandbox). No router or gate
runs on a real item, nothing reads test, and no call is made.

- **The estimated latency** (A17 item 16) of a trace is the sum of the pilot's median for each config called, the router's median
  and each gate call's median, where the code check applies only to a request with an `assert` line and the paid gate only to an
  answer that passed the free checks from a start below `opus`; a point that fell back is A with no router and no gate. Worked by
  hand for a router alone, a free-checks cascade and a Jev cascade, with a median per task.
- **The final table** holds §10's eight rows (A, B, D, E, F, G, G+ and the best cascade) for each mix and each target; A and B are
  the references' own rows, D is the oracle at the target with all of its savings captured, and the best cascade is the one A17
  item 24's rule chooses on dev's table, whatever test shows. A's and B's latency are hand-worked from the pilot's medians. Every
  policy row equals the policy's own part (a different code path), a point that falls back reads **falls back to A** with A's
  latency, and a policy's reading is the target-met part's.
- **The per-policy tables** list the four routers and all fifteen cascades, each under both mixes and all three targets and per
  task; IFEval and MBPP+ carry A7 item 2's warning worded for test, and no per-task row has a verdict.
- **The comparisons' wiring:** the part holds 51 comparisons in item 22's groups and the declared order; each cell's value is the
  first policy's own row less the second's (cost and quality), and a cell reads "both fall back to A" exactly where both policies'
  rows say they fell back. The replication summary on the rehearsal counts, from the made-up test's cells and table v4's committed
  dev cells read apart from the module, the same numbers.
- **Refusals:** a final table with a policy's part missing, and a replication summary before the comparisons, are refused.
"""

import json
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from analysis import final_comparisons, final_table
from analysis.check_final_run import rehearsed
from analysis.gate_table import GateRow, GateScoreTable, PaidScore
from analysis.latency_estimate import PILOT_MEDIANS, GateMedians
from analysis.router_names import CASCADES, G_PLUS, HAIKU, JEV_GATE, NAMED_ROUTERS, cascade_section
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


def close(a: float | None, b: float, tolerance: float = 1e-9) -> bool:
    return a is not None and abs(a - b) <= tolerance


# ---- the estimated latency, by hand --------------------------------------------------------------------------------

TASK_OF = {"i1": "ifeval", "i2": "mbppplus", "i3": "banking77", "i4": "mmlu_pro"}
MEDIANS = GateMedians(
    code_check=0.2,
    jev={"ifeval": 0.3, "banking77": 0.4, "mmlu_pro": 0.5, "mbppplus": 0.1},
    sonnet={"ifeval": 2.0, "banking77": 2.0, "mmlu_pro": 2.0, "mbppplus": 2.0},
)
ROUTER = {"ifeval": 0.5, "banking77": 0.7, "mmlu_pro": 0.9, "mbppplus": 1.1}  # one median per task, as F's and G's are


def made_up_table() -> GateScoreTable:
    """Four answers that start at `haiku` and `opus` each; the banking77 request's haiku answer fails a free check."""
    rows = []
    for item in TASK_OF:
        for config in ("haiku", "opus"):
            failed = item == "i3" and config == "haiku"
            rows.append(GateRow(item, config, not failed, "stop" if failed else None,
                                jev=None if failed else PaidScore(0.9, 0.0, False)))  # fmt: skip
    return GateScoreTable.build(rows, [(r.item_id, r.config) for r in rows])


def trace(item: str, *configs: str) -> Outcome:
    return Outcome(item, TASK_OF[item], item, 0.001, 1, tuple(configs))


def latency() -> None:
    print("-- the estimated latency of a trace, by hand --")
    table, code = made_up_table(), frozenset({"i2"})
    # i1 escalates (haiku then opus), i2 stays at haiku, i3's haiku answer fails a free check and goes to opus, i4 starts at opus.
    outcomes = [trace("i1", "haiku", "opus"), trace("i2", "haiku"), trace("i3", "haiku", "opus"), trace("i4", "opus")]

    def seconds(**kwargs: Any) -> dict[str, float]:
        base: dict[str, Any] = {"fell_back": False, "router": ROUTER, "gate": None, "cutoff": None, "table": table,
                                "medians": MEDIANS, "code_items": code}  # fmt: skip
        return final_table.trace_latencies(outcomes, **{**base, **kwargs})

    # A router alone: the router's median for the task, then the pilot's median for each config called.
    #   i1 .5 + 6.9 + 14.0   i2 1.1 + 1.6   i3 .7 + 2.0 + 2.2   i4 .9 + 4.5
    alone = seconds()
    check(all(close(alone[i], s) for i, s in {"i1": 21.4, "i2": 2.7, "i3": 4.9, "i4": 5.4}.items()),
          f"a router alone: its median for the task, then each config's pilot median {alone}")  # fmt: skip
    # The free checks alone: the code check runs on i2's request (it has an assert line); no paid gate has a cutoff.
    #   i1 21.4   i2 2.7 + .2   i3 4.9   i4 5.4
    free = seconds(gate="free")
    check(all(close(free[i], s) for i, s in {"i1": 21.4, "i2": 2.9, "i3": 4.9, "i4": 5.4}.items()),
          f"the free checks alone: only a request with an assert line pays the code check {free}")  # fmt: skip
    # The Jev gate at a cutoff: i1 pays the gate (+.3 for ifeval), i2 pays the code check and the gate (+.2 +.1), i3's free check
    # failed so no paid gate (no assert line either), i4 started at opus and is not gated.
    gated = seconds(gate="jev", cutoff=0.8)
    check(all(close(gated[i], s) for i, s in {"i1": 21.7, "i2": 3.0, "i3": 4.9, "i4": 5.4}.items()),
          f"the Jev gate: paid on a start below opus whose answer passed the free checks, never after a failed check or at opus {gated}")  # fmt: skip
    # A point that fell back is A: opus alone, no router, no gate: ifeval 14.0, mbppplus 3.4, banking77 2.2, mmlu_pro 4.5
    a = seconds(gate="jev", cutoff=0.8, fell_back=True)
    check(all(close(a[i], s) for i, s in {"i1": 14.0, "i2": 3.4, "i3": 2.2, "i4": 4.5}.items()),
          f"a point that fell back is A's own latency: no router, no gate {a}")  # fmt: skip
    check(
        all(PILOT_MEDIANS[TASK_OF[i]]["opus"] == s for i, s in a.items()), "and those are the pilot's medians for opus"
    )
    weights = {"i1": 0.1, "i2": 0.65, "i3": 0.2, "i4": 0.05}
    # Weighted by a mix: sorted 2.7 (.65) already holds more than half the weight, so the median is 2.7.
    check(
        close(final_table.median_seconds(alone, weights), 2.7),
        "the median is the weighted one: the mix's weight, not the items' count",
    )
    only = {"i1": 1.0}
    check(close(final_table.median_seconds(alone, only), 21.4), "a task's own median takes that task's items alone")
    check(refused(Exception, lambda: final_table.trace_latencies(outcomes, fell_back=False, router=ROUTER, gate="jev", cutoff=None,
                                                                   table=table, medians=MEDIANS, code_items=code)),
          "a paid gate with no cutoff is refused by the shared guard")  # fmt: skip


# ---- the rehearsal -----------------------------------------------------------------------------------------------------

REALISTIC_A = {
    "realistic": 3.4,
    "uniform": 3.95,
}  # A's median seconds on the rehearsal's 2 items per task, worked by hand below
REALISTIC_B = {"realistic": 1.6, "uniform": 3.0}
# Realistic weights: mbppplus .65 (3.4 s for A, 1.6 for B), banking77 .20 (2.2; 2.0), ifeval .10 (14.0; 6.9), mmlu_pro .05 (4.5; 4.0).
#   A sorted: banking77 2.2 (.20 cumulative), mbppplus 3.4 (.85 >= .5): 3.4.   B sorted: mbppplus 1.6 (.65 >= .5): 1.6.
# Uniform weights .25 per task: A: 2.2 (.25), 3.4 (.5 exactly half: the midpoint with the next, 4.5): 3.95.
#   B: 1.6 (.25), 2.0 (.5, exactly half: the midpoint with the next, 4.0): 3.0.
TASK_A = {"ifeval": 14.0, "banking77": 2.2, "mmlu_pro": 4.5, "mbppplus": 3.4}
TASK_B = {"ifeval": 6.9, "banking77": 2.0, "mmlu_pro": 4.0, "mbppplus": 1.6}


def committed_dev_cells() -> dict[str, dict]:
    return {
        g: b["pairs"]
        for g, b in json.loads((PROJECT_ROOT / "figures" / "table_v4.json").read_text(encoding="utf-8"))["comparisons"][
            "groups"
        ].items()
    }


def rehearsal() -> None:
    print("\n-- the final table on the rehearsal's made-up world --")
    with rehearsed() as r:
        outcome, _, _ = r.go()
        parts = r.tables["parts"]
        check(outcome.stopped_at is None, "the whole made-up run goes through")
        table, comp = parts["final_table"], parts["comparisons"]
        check(all(k in parts for k in ("comparisons", "declared", "replication", "final_table")),
              "the tables hold the comparisons, the replication summary and the final table, in the tables file of test's own")  # fmt: skip
        by_mix = table["by_mix"]
        check(list(by_mix) == ["realistic", "uniform"] and all(list(by_mix[m]) == ["100%", "99%", "97%"] for m in by_mix),
              "one table for each mix and each of the three targets")  # fmt: skip
        ok = True
        for mix in by_mix:
            for target, rows in by_mix[mix].items():
                ok &= [row["policy"] for row in rows] == ["A", "B", "D", "E", "F", "G", G_PLUS, "I"]
        check(
            ok,
            "each holds §10's eight rows in its order: A, B, D, E, F, G, G+ and the best cascade (the G+ row and the best-cascade row)",
        )
        ok, hand_ok = True, True
        for mix in by_mix:
            for target, rows in by_mix[mix].items():
                row = {x["policy"]: x for x in rows}
                a, b, d = row["A"], row["B"], row["D"]
                hand_ok &= close(a["latency_p50_s"], REALISTIC_A[mix]) and close(b["latency_p50_s"], REALISTIC_B[mix])
                ok &= a["savings_vs_A"] == 0 and a["captured"] == 0 and b["captured"] is None and d["captured"] == 1.0
                ok &= d["latency_p50_s"] is None and d["quality"]["low"] is None
                configs = parts["references"]["mixes"][mix]["configs"]
                ok &= (
                    a["quality"]["value"] == configs["opus"]["quality"]["value"]
                    and b["cost_per_1k"]["value"] == configs["haiku"]["cost_per_1k"]["value"]
                )
                oracle = parts["references"]["mixes"][mix]["targets"][target]
                ok &= (
                    d["cost_per_1k"]["value"] == oracle["oracle"]["cost_per_1k"]
                    and d["quality"]["value"] == oracle["quality"]
                )
        check(ok, "A saves and captures nothing, B has no captured fraction, D is the oracle at the target with all its savings captured, "
                  "and A and B are the references' own numbers")  # fmt: skip
        check(hand_ok, "A's and B's estimated latency are the hand-worked weighted medians of the pilot's medians (3.4 and 1.6 s "
                       "realistic; 3.95 and 3.0 s uniform, where the weight reaches exactly half)")  # fmt: skip
        met = parts["target_met"]["points"]
        ok = True
        for mix in by_mix:
            for target, rows in by_mix[mix].items():
                for policy, section in [
                    ("E", "rules"),
                    ("F", "trained"),
                    ("G", "jev"),
                    (G_PLUS, NAMED_ROUTERS[G_PLUS]),
                    ("I", table["best_cascade"]["section"]),
                ]:
                    got = next(x for x in rows if x["policy"] == policy)
                    own = parts[section]["mixes"][mix][target]
                    reading = met[section][mix][target]["reading"]
                    ok &= got["section"] == section and got["fell_back"] == own["fell_back"]
                    ok &= [got["quality"][k] for k in ("value", "low", "high")] == [
                        own["quality"][k] for k in ("value", "low", "high")
                    ]
                    ok &= (
                        got["cost_per_1k"]["value"] == own["cost_per_1k"]["value"]
                        and got["savings_vs_A"] == own["savings_vs_A"]["share"]
                    )
                    ok &= got["captured"] == own["captured"]["value"]
                    ok &= (
                        got["reading"]
                        == {"met": "met", "missed": "not met", "falls back to A": "falls back to A"}[reading]
                    )
                    if own["fell_back"]:  # A itself: A's own latency, and never met
                        ok &= got["reading"] == "falls back to A" and close(got["latency_p50_s"], REALISTIC_A[mix])
        check(ok, "each policy row (E, F, G, G+ and the best cascade) equals the policy's own part and its target-met reading, and a "
                  "point that fell back reads falls back to A with A's latency")  # fmt: skip
        check(table["best_cascade"]["section"] == cascade_section(HAIKU, JEV_GATE)
              and table["best_cascade"]["label"] == CASCADES[cascade_section(HAIKU, JEV_GATE)],
              "the best cascade is haiku/Jev, the rule's choice on dev's table (A17 item 29), not whatever test shows")  # fmt: skip

        print("\n-- the per-policy tables, per task and the indicative marks --")
        policies = table["per_policy"]
        check(list(policies) == [*NAMED_ROUTERS.values(), *CASCADES] and len(policies) == 19,
              "the per-policy tables hold the four routers and all fifteen cascades")  # fmt: skip
        ok = True
        for section, body in policies.items():
            for mix in by_mix:
                for target in by_mix[mix]:
                    own = parts[section]["mixes"][mix][target]
                    row = body["mixes"][mix][target]
                    ok &= (
                        row["fell_back"] == own["fell_back"]
                        and row["cost_per_1k"]["value"] == own["cost_per_1k"]["value"]
                    )
                    ok &= (
                        row["reading"] in ("met", "not met", "falls back to A")
                        and (row["reading"] == "falls back to A") == own["fell_back"]
                    )
        check(
            ok,
            "each policy's rows under both mixes and all targets are its part's, with a reading that says falls back to A exactly where it did",
        )
        tasks = table["by_task"]
        check(
            list(tasks) == ["ifeval", "banking77", "mmlu_pro", "mbppplus"], "the per-task tables cover the four tasks"
        )
        warned = {task: body["warning"] for task, body in tasks.items()}
        check(warned["ifeval"] == "Only 2 test items, below the plan's minimum of a few hundred: indicative"
              and warned["mbppplus"] == "Only 2 test items, below the plan's minimum of a few hundred: indicative"
              and warned["banking77"] is None and warned["mmlu_pro"] is None,
              "IFEval and MBPP+ alone are marked indicative, in words that say test items")  # fmt: skip
        ok = True
        for task, body in tasks.items():
            for mix in by_mix:
                for rows in body["mixes"][mix].values():
                    letters = [x["policy"] for x in rows]
                    ok &= letters == ["A", "B", "E", "F", "G", G_PLUS, "I"] and all(x["reading"] is None for x in rows)
                    got = {x["policy"]: x for x in rows}
                    ok &= close(got["A"]["latency_p50_s"], TASK_A[task]) and close(
                        got["B"]["latency_p50_s"], TASK_B[task]
                    )
        check(ok, "per task: A, B, E, F, G, G+ and the best cascade, with no oracle row and no verdict, and A's and B's latency "
                  "the pilot's medians for the task")  # fmt: skip
        ok = all(
            row["fell_back"] == parts[s if s in NAMED_ROUTERS.values() else f"{s}_tasks"]["tasks"][t][m][g]["fell_back"]
            for s, body in policies.items()
            for t, by in body["tasks"].items()
            for m, bt in by.items()
            for g, row in bt.items()
        )
        check(ok, "each policy's per-task rows are its per-task part's")

        print("\n-- the comparisons' wiring and the replication summary --")
        check(comp["count"] == 51 and [len(b["pairs"]) for b in comp["groups"].values()] == [12, 12, 10, 5, 12]
              and parts["declared"]["order"][:2] == ["haiku/Jev less G", "haiku/Jev less G+"] and len(set(parts["declared"]["order"])) == 51,
              "51 comparisons in item 22's groups, and the declared order leads with haiku/Jev against G and G+")  # fmt: skip
        label_of = {
            **{letter: section for letter, section in NAMED_ROUTERS.items()},
            **{label: key for key, label in CASCADES.items()},
        }
        ok, both, cells = True, 0, 0
        for body in comp["groups"].values():
            for pair, by_mix in body["pairs"].items():
                first, second = (label_of[x] for x in pair.split(" less "))
                section_a, section_b = final_comparisons.section_of(first), final_comparisons.section_of(second)
                for mix, by_target in by_mix.items():
                    for target, cell in by_target.items():
                        a, b = parts[section_a]["mixes"][mix][target], parts[section_b]["mixes"][mix][target]
                        cells += 1
                        ok &= cell["first_fell_back"] == a["fell_back"] and cell["second_fell_back"] == b["fell_back"]
                        if a["fell_back"] and b["fell_back"]:
                            both += 1
                            ok &= (
                                cell["reading"] == "both fall back to A"
                                and cell["cost_per_1k"] is None
                                and cell["quality"] is None
                            )
                            continue
                        ok &= close(cell["cost_per_1k"]["value"], a["cost_per_1k"]["value"] - b["cost_per_1k"]["value"])
                        ok &= close(cell["quality"]["value"], a["quality"]["value"] - b["quality"]["value"])
                        ok &= (
                            cell["cost_per_1k"]["low"]
                            <= cell["cost_per_1k"]["value"] + 1e-9
                            <= cell["cost_per_1k"]["high"] + 2e-9
                        )
        check(ok and cells == 306, f"each of the 306 cells is the first policy's own row less the second's in cost and quality, and "
                                   f"reads both fall back to A exactly where both rows fell back ({both} such cells)")  # fmt: skip
        dev = committed_dev_cells()
        rep = parts["replication"]
        want = {"cost_per_1k": [0, 0, 0], "quality": [0, 0, 0]}
        for group, pairs in dev.items():
            for pair, by_mix in pairs.items():
                for mix, by_target in by_mix.items():
                    for target, cell in by_target.items():
                        for measure in want:
                            found = cell[measure]
                            side = 0 if found is None else 1 if found["low"] > 0 else -1 if found["high"] < 0 else 0
                            if not side:
                                continue
                            there = comp["groups"][group]["pairs"][pair][mix][target][measure]
                            same = there is not None and (there["value"] > 0) == (side > 0) and there["value"] != 0
                            both_ = same and ((there["low"] > 0) if side > 0 else (there["high"] < 0))
                            want[measure][0] += 1
                            want[measure][1] += same
                            want[measure][2] += both_
        got = {m: [c["dev_excluded"], c["same_sign"], c["same_sign_excluding_zero"]] for m, c in rep["counts"].items()}
        check(got == want and len(rep["listed"]) == want["cost_per_1k"][0] + want["quality"][0],
              f"the replication summary on the rehearsal equals a count made apart from the module over table v4's dev cells and the "
              f"made-up test cells {got}")  # fmt: skip

        print("\n-- refusals --")
        inputs = SimpleNamespace(cache={"built": {}}, matrix=None, extras={}, draws=None)
        check(
            refused(final_table.FinalTableError, lambda: final_table.build(inputs)),
            "a final table with no policy built is refused",
        )
        check(refused(final_comparisons.ComparisonsError, lambda: final_comparisons.replication_part(inputs)),
              "a replication summary before the comparisons is refused")  # fmt: skip


def main() -> int:
    latency()
    rehearsal()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
