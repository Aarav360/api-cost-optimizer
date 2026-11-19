"""Check the dev reproduction: the offline half on dev's stored inputs (PLAN_v4.md Amendment A19, item 13; issue #79).

Run from the project root:  uv run python -m analysis.check_final_run_dev

Needs the dev results matrix in results/ (not committed), so the hook runs it and CI doesn't (`HOOK_ONLY` in
`analysis/check_guardrails.py`). It reads dev only, and no gate, router or API is called.

The offline half (`analysis/offline_half.py`) builds a split's tables from that split's stored inputs. Here it builds
dev's from the dev matrix through the existing reader, and each part that dev has committed numbers for must reproduce
them to the last stored digit, compared as the committed JSON holds them: no tolerance. A mismatch is a finding to
explain, not a number to loosen.

- the references part (A to D) under both mixes and per task equals chart v0's keys of `figures/pareto.json` and
  `figures/pareto_tasks.json`, panel by panel, key by key;
- E's and F's parts (`analysis/final_routers.py`, issue #83) evaluate each router at the six frozen points of its section of
  `routers/operating_points.json`, with no walk, and reproduce table v1's committed numbers for it (`figures/pareto.json` and
  `figures/pareto_tasks.json`: quality and cost with intervals, difference from A, savings, captured fraction, routing mix,
  matched cost, McNemar's test) to the last stored digit; a walk's margin and steps tried are the walk's and aren't reproduced;
- the guard can fail: the same comparison on a copy with one stored digit changed, one key dropped and one key added is
  a mismatch each time (so the comparison finds a difference where there is one);
- every part of the tables names its committed counterpart or has none, and the parts that have none are checked against
  stored values counted outside this code (the stops part against the dev matrix's own counts of each config's non-correct
  rows by status: haiku wrong 111; sonnet wrong 102 and max_tokens 1; opus_low wrong 74 and refusal 1; opus wrong 73 and
  refusal 1; 2,976 rows in all);
- the twelve router-started cascades' parts (`analysis/final_routed.py`, issue #87: E, F, G and G+, each with the free checks,
  the Jev gate and the Sonnet gate) are evaluated at their frozen points, with each router's own dev table and the gate-score
  table, and reproduce table v4's committed rows to the last stored digit; each holds the file's threshold, cutoff and
  `fell_back`, and the number of points that route is the file's (A19 item 6: 46 of the twelve's 72, E 4, 4 and 0, F 3, 4
  and 0, G 6, 6 and 6, G+ 5, 5 and 3 by the free checks, Jev and Sonnet);
- dev's IFEval and MBPP+ entries carry the plan's dev words, unchanged by the test wording.
"""

import copy
import hashlib
import json
import sys
from typing import Any

from analysis import offline_half
from analysis.final_design import expected_misses, headline
from analysis.final_gates import tasks_key
from analysis.final_table import TARGET_KEYS
from analysis.operating_points import OPERATING_POINTS_FILE
from analysis.router_names import (
    CASCADES,
    FREE_GATE,
    G_PLUS,
    HAIKU,
    JEV_GATE,
    NAMED_ROUTERS,
    SONNET_GATE,
    cascade_section,
)
from analysis.task_warnings import task_warning
from harness.client import PROJECT_ROOT

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def differences(built: Any, committed: Any, path: str = "") -> list[str]:
    """Where two JSON values differ, key by key: a missing or extra key, or a number or string that isn't equal exactly."""
    if isinstance(built, dict) and isinstance(committed, dict):
        found = [f"{path}/{key}: not in the build" for key in committed if key not in built]
        found += [f"{path}/{key}: not committed" for key in built if key not in committed]
        for key in committed:
            if key in built:
                found += differences(built[key], committed[key], f"{path}/{key}")
        return found
    if isinstance(built, list) and isinstance(committed, list):
        if len(built) != len(committed):
            return [f"{path}: {len(built)} entries, {len(committed)} committed"]
        return [d for i, (b, c) in enumerate(zip(built, committed)) for d in differences(b, c, f"{path}[{i}]")]
    return [] if built == committed and type(built) is type(committed) else [f"{path}: {built!r} is not {committed!r}"]


def committed_of(by_key: dict[str, offline_half.Part], key: str) -> dict:
    """A part's committed dev numbers (only for a part that names them)."""
    committed = by_key[key].committed
    if committed is None:
        raise ValueError(f"part {key!r} has no committed counterpart")
    return committed()


TABLE_V4_SHA256 = "07fbf22813988b626ca8e5898f9dac4291c1741392e76b044a05caf8f3b8c8cf"  # figures/table_v4.json as committed (A19 item 8)
MIXES = ("realistic", "uniform")


def committed_json(name: str) -> Any:
    return json.loads((PROJECT_ROOT / "figures" / name).read_text(encoding="utf-8"))


def same_numbers(row: dict, committed: dict, latency: float | None) -> bool:
    """A final-table row against the committed row of the same policy and point: quality, cost, savings and captured fraction to
    the last stored digit, and the estimated latency table v4 committed. On dev every routed point is met."""
    fell_back = committed["fell_back"]
    return (
        row["fell_back"] == fell_back
        and row["reading"] == ("falls back to A" if fell_back else "met")
        and [row["quality"][k] for k in ("value", "low", "high")] == [committed["quality"][k] for k in ("value", "low", "high")]
        and [row["cost_per_1k"][k] for k in ("value", "low", "high")] == [committed["cost_per_1k"][k] for k in ("value", "low", "high")]
        and row["savings_vs_A"] == committed["savings_vs_A"]["share"]
        and row["captured"] == committed["captured"]["value"]
        and row["latency_p50_s"] == latency
    )  # fmt: skip


def comparisons_and_final_table(tables: dict[str, Any]) -> None:
    print("\n-- table v4's comparisons and rows, the replication summary and the final table (issue #89) --")
    held = committed_json("table_v4.json")
    check(hashlib.sha256((PROJECT_ROOT / "figures" / "table_v4.json").read_bytes()).hexdigest() == TABLE_V4_SHA256,
          "figures/table_v4.json is the committed file, byte for byte: test's tables are files of their own")  # fmt: skip
    parts = tables["parts"]
    check(all(key in parts and tasks_key(key) in parts for key in CASCADES),
          "all fifteen cascades have a part and a per-task part, so the comparisons cover every pair")  # fmt: skip
    group_names = {group: list(body["pairs"]) for group, body in parts["comparisons"]["groups"].items()}
    check(parts["comparisons"]["count"] == 51 and [len(p) for p in group_names.values()] == [12, 12, 10, 5, 12]
          and group_names == {g: list(b["pairs"]) for g, b in held["comparisons"]["groups"].items()},
          "the 51 comparisons are table v4's, in its groups and order (the part loop above compared every cell to the last stored digit)")  # fmt: skip
    check(parts["declared"]["order"][:2] == held["comparisons"]["headline"]
          and sorted(parts["declared"]["order"]) == sorted(p for ps in group_names.values() for p in ps),
          "the declared order leads with haiku/Jev against G and G+ and holds every pair once")  # fmt: skip
    # Table v4's per-task McNemar rows and matched-cost rows, which the cascades' parts reproduce (their per-task parts had no
    # committed numbers of their own)
    mcnemar_ok = all(
        parts[tasks_key(key)]["tasks"][task][mix][target]["mcnemar"] == held["comparisons"]["mcnemar"][key][task][mix][target]
        for key in CASCADES for task in held["comparisons"]["mcnemar"][key] for mix in MIXES for target in TARGET_KEYS
    )  # fmt: skip
    check(
        mcnemar_ok,
        "each cascade's per-task McNemar rows equal table v4's comparisons.mcnemar, to the last stored digit",
    )
    check(all(parts[key]["mixes"][mix][target]["matched_cost"] == held["comparisons"]["matched_cost"][key][mix][target]
              for key in CASCADES for mix in MIXES for target in TARGET_KEYS),
          "each cascade's matched-cost rows equal table v4's comparisons.matched_cost")  # fmt: skip

    # The replication summary on dev itself: every comparison whose dev interval excluded 0, counted straight from table v4's JSON
    scanned = {"cost_per_1k": 0, "quality": 0}
    for body in held["comparisons"]["groups"].values():
        for by_mix in body["pairs"].values():
            for by_target in by_mix.values():
                for found in by_target.values():
                    for measure in scanned:
                        cell = found[measure]
                        scanned[measure] += cell is not None and (cell["low"] > 0 or cell["high"] < 0)
    counts = parts["replication"]["counts"]
    check({m: counts[m]["dev_excluded"] for m in scanned} == scanned
          and all(counts[m]["same_sign"] == counts[m]["same_sign_excluding_zero"] == scanned[m] for m in scanned)
          and len(parts["replication"]["listed"]) == sum(scanned.values()),
          f"dev against itself: the {scanned['cost_per_1k']} cost and {scanned['quality']} quality intervals that excluded 0 "
          f"are listed, and each replicates (counted from table v4's JSON)")  # fmt: skip

    # The final table's rows against the committed rows of each policy, to the last stored digit, and table v4's latency
    final = parts["final_table"]
    pareto, per_task = committed_json("pareto.json")["panels"], committed_json("pareto_tasks.json")["panels"]
    latency = held["latency"]
    letter_of = {section: letter for letter, section in NAMED_ROUTERS.items()}
    ok, checked = True, 0
    for mix in MIXES:
        panel = pareto[mix]
        for target in TARGET_KEYS:
            by_system = {row["system"]: row for row in final["by_mix"][mix][target]}
            ok &= list(by_system) == ["A", "B", "D", "E", "F", "G", G_PLUS, "I"]
            for system, config in (("A", "opus"), ("B", "haiku")):
                row, committed = by_system[system], panel["configs"][config]
                ok &= (row["quality"] == {k: committed["quality"][k] for k in ("value", "low", "high")}
                       and row["cost_per_1k"]["value"] == committed["cost_per_1k"]["value"]
                       and row["savings_vs_A"] == committed["savings_vs_A"]["share"]
                       and row["latency_p50_s"] == latency["configs"][config][mix])  # fmt: skip
            oracle = panel["targets"][target]
            row = by_system["D"]
            ok &= (row["quality"]["value"] == oracle["quality"] and row["cost_per_1k"]["value"] == oracle["oracle"]["cost_per_1k"]
                   and row["savings_vs_A"] == oracle["oracle"]["savings_vs_A"]["share"] and row["captured"] == 1.0
                   and row["latency_p50_s"] is None)  # fmt: skip
            best = by_system["I"]
            ok &= best["section"] == held["best_cascade"]["best"] and best["label"].endswith(
                held["best_cascade"]["label"]
            )
            for system, section in [("E", "rules"), ("F", "trained"), ("G", "jev"), (G_PLUS, NAMED_ROUTERS[G_PLUS])]:
                committed = panel["routers"][section]["operating_points"][target]
                ok &= same_numbers(by_system[system], committed, latency["routers"][system][mix][target])
                checked += 1
            committed = held["cascades"][best["section"]]["operating_points"][mix][target]
            ok &= same_numbers(best, committed, latency["cascades"][best["section"]][mix][target])
            checked += 1
            for section, body in final["per_policy"].items():
                if section in CASCADES:
                    committed = held["cascades"][section]["operating_points"][mix][target]
                    seconds = latency["cascades"][section][mix][target]
                else:
                    committed = panel["routers"][section]["operating_points"][target]
                    seconds = latency["routers"][letter_of[section]][mix][target]
                ok &= same_numbers(body["mixes"][mix][target], committed, seconds)
                checked += 1
    check(ok, f"the final table's A, B, D, E, F, G, G+ and best-cascade rows under both mixes and all three targets, and the "
              f"per-policy tables' 19 policies, equal the committed rows and table v4's latency to the last stored digit ({checked} rows)")  # fmt: skip
    check(list(final["per_policy"]) == [*NAMED_ROUTERS.values(), *CASCADES]
          and final["best_cascade"]["section"] == held["best_cascade"]["best"],
          "the per-policy tables list the four routers and all fifteen cascades, and the best cascade is dev's rule's")  # fmt: skip
    ok = True
    for task, body in final["by_task"].items():
        ok &= body["warning"] == (task_warning(task, body["items"]) if task in ("ifeval", "mbppplus") else None)
        for mix in MIXES:
            for target in TARGET_KEYS:
                a_cost = per_task[task]["configs"]["opus"]["cost_per_1k"]["value"]
                by_system = {row["system"]: row for row in body["mixes"][mix][target]}
                ok &= list(by_system) == ["A", "B", "E", "F", "G", G_PLUS, "I"]
                for system, section in [
                    ("E", "rules"),
                    ("F", "trained"),
                    ("G", "jev"),
                    (G_PLUS, NAMED_ROUTERS[G_PLUS]),
                ]:
                    committed = per_task[task]["routers"][section]["operating_points"][mix][target]
                    row = by_system[system]
                    ok &= (row["quality"]["value"] == committed["quality"]["value"]
                           and row["cost_per_1k"]["value"] == committed["cost_per_1k"]["value"]
                           and row["savings_vs_A"] == 1 - committed["cost_per_1k"]["value"] / a_cost
                           and row["reading"] is None)  # fmt: skip
    check(ok, "per task, the rows equal pareto_tasks.json's committed quality and cost for E, F, G and G+, A7 item 2's warning "
              "stands on IFEval and MBPP+ alone, and no row carries a verdict")  # fmt: skip


def main() -> int:
    print("-- the guard can fail --")
    base: dict[str, Any] = {"a": {"x": 0.1, "y": [1, 2]}, "b": "words"}
    check(differences(copy.deepcopy(base), base) == [], "the same JSON has no difference")
    digit = copy.deepcopy(base)
    digit["a"]["x"] = 0.1 + 1e-16 * 2  # one digit in the last place
    check(len(differences(digit, base)) == 1, "a number changed in its last digit is a difference")
    dropped = copy.deepcopy(base)
    del dropped["a"]["y"]
    check(len(differences(dropped, base)) == 1, "a key dropped from the build is a difference")
    added = copy.deepcopy(base)
    added["c"] = 1
    check(len(differences(added, base)) == 1, "a key added to the build is a difference")
    check(len(differences({"n": 1.0}, {"n": 1})) == 1, "an integer built as a float is a difference")

    print("\n-- the offline half on dev's stored inputs --")
    inputs = offline_half.dev_inputs()
    check(inputs.split == "dev" and len(inputs.matrix.items) == 744, "dev's inputs hold its 744 items")
    tables = json.loads(json.dumps(offline_half.build_tables(inputs)))  # as the JSON file holds numbers
    check(tables["split"] == "dev" and tables["n"] == 744 and tables["price_snapshot"] == "2026-10-12",
          "the tables name dev, its 744 items and the price snapshot")  # fmt: skip
    for part in offline_half.PARTS:
        if part.committed is None:
            check(part.key in tables["parts"], f"part {part.key!r}: built, with no committed counterpart on dev")
            continue
        found = differences(tables["parts"][part.key], part.committed())
        check(not found, f"part {part.key!r} reproduces dev's committed numbers to the last stored digit {found[:3]}")

    print("\n-- E, F, G and G+ at their frozen operating points (issues #83 and #84) --")
    by_key = {part.key: part for part in offline_half.PARTS}
    check(all(by_key[k].committed is not None for k in ("rules", "trained", "jev", NAMED_ROUTERS[G_PLUS])),
          "E's, F's, G's and G+'s parts each name their committed dev numbers, so the loop above checked them")  # fmt: skip
    for key, routes in (("rules", 4), ("trained", 3), ("jev", 6), (NAMED_ROUTERS[G_PLUS], 5)):
        rows = tables["parts"][key]["mixes"]
        points = [row for by_target in rows.values() for row in by_target.values()]
        check(len(points) == 6 and sum(not row["fell_back"] for row in points) == routes
              and all(row["fell_back"] == (row["threshold"] is None) for row in points)
              and all(row["difference_from_A"]["value"] == 0 for row in points if row["fell_back"]),
              f"{key}: six points, {routes} of which route (A19 item 6), each fall back being A itself")  # fmt: skip
        tasks = tables["parts"][key]["tasks"]
        per_task = [row for task in tasks.values() for by_target in task.values() for row in by_target.values()]
        check(set(tasks) == {"ifeval", "banking77", "mmlu_pro", "mbppplus"} and len(per_task) == 24
              and all(row["mcnemar"] is None for row in per_task if row["fell_back"]),
              f"{key}: the per-task rows cover the four tasks, and a point that fell back has no McNemar row")  # fmt: skip

    print("\n-- the twelve router-started cascades at their frozen operating points (issue #87) --")
    held = json.loads(OPERATING_POINTS_FILE.read_text(encoding="utf-8"))
    # how many of each cascade's six points route in the file (46 in all: A19 item 6's 54 of the cascades, 8 of them haiku's),
    # by gate (free, Jev, Sonnet)
    routing = {"rules": (4, 4, 0), "trained": (3, 4, 0), "jev": (6, 6, 6), NAMED_ROUTERS[G_PLUS]: (5, 5, 3)}
    for router, by_gate in routing.items():
        for gate, n_routes in zip((FREE_GATE, JEV_GATE, SONNET_GATE), by_gate):
            key = cascade_section(router, gate)
            built = tables["parts"][key]["mixes"]
            points = [(mix, target) for mix in built for target in built[mix]]
            check(by_key[key].committed is not None and len(points) == 6
                  and sum(not built[m][t]["fell_back"] for m, t in points) == n_routes
                  and all(built[m][t]["fell_back"] == held[key][m][t]["fell_back"]
                          and built[m][t]["threshold"] == held[key][m][t].get("threshold")
                          and built[m][t]["cutoff"] == held[key][m][t].get("cutoff") for m, t in points)
                  and all(built[m][t]["escalation_rate"] == 0 for m, t in points if built[m][t]["fell_back"]),
                  f"{CASCADES[key]}: six points, {n_routes} of which route, each holding the file's own threshold and cutoff")  # fmt: skip
            per_task = tables["parts"][f"{key}_tasks"]["tasks"]
            check(set(per_task) == {"ifeval", "banking77", "mmlu_pro", "mbppplus"}
                  and all(row["mcnemar"] is None for task in per_task.values() for by_target in task.values()
                          for row in by_target.values() if row["fell_back"]),
                  f"{CASCADES[key]}: the per-task rows cover the four tasks (dev committed none to compare)")  # fmt: skip

    print("\n-- G's and G+'s D6 from Stage 6's stored train passes, as the drift step measures it (A15 item 19) --")
    from analysis import collect_jev, final_jev
    from routers import jev

    loaded = collect_jev.load_settings()

    def no_client() -> jev.JevClient:
        raise AssertionError("this section reads stored records and calls nothing")

    setup = final_jev.Setup(loaded, jev.read_weights(loaded.call), no_client, collect_jev.RESULTS_DIR)
    sample = collect_jev.noise_items(loaded)
    # A15 item 19's record, to four decimals: D6 by config, haiku, sonnet, opus_low
    recorded = {"jev": (0.0119, 0.0120, 0.0112), NAMED_ROUTERS[G_PLUS]: (0.0118, 0.0075, 0.0046)}
    for router, d6 in recorded.items():
        noise = final_jev.train_passes(router, setup, sample)[2]
        check(len(sample) == 100 and [round(noise[c].d6, 4) for c in ("haiku", "sonnet", "opus_low")] == list(d6),
              f"{router}: D6 over the 100 noise items is A15 item 19's recorded {d6}")  # fmt: skip

    print("\n-- the oracle, the envelope and the captured fractions on dev (issue #88) --")
    references = committed_of(by_key, "references")
    panel = references["mixes"]["realistic"]
    check(len(panel["oracle"]) > 1 and len(panel["envelope"]) >= 2
          and all("captured" in panel["targets"][t]["envelope"] for t in ("100%", "99%", "97%")),
          "the references' committed counterpart holds the oracle's curve, the envelope's corners and the envelope's captured "
          "fractions, so the loop above compared them to the last stored digit")  # fmt: skip
    policies = [
        key
        for key, part in by_key.items()
        if part.committed is not None and "mixes" in committed_of(by_key, key) and key != "references"
    ]
    check(bool(policies) and all("captured" in row for key in policies for by_target in committed_of(by_key, key)["mixes"].values()
                           for row in by_target.values()),
          f"each policy's committed rows hold the captured fraction ({', '.join(policies)}), compared in the same loop")  # fmt: skip
    check(all(tables["parts"]["flip_rates"]["configs"][c]["pairs"] == 75 for c in ("haiku", "sonnet", "opus_low", "opus")),
          "dev's flip rates are recounted from dev's runs 0 to 2 of the 75-item noise subset and equal the carried-over file's")  # fmt: skip

    print("\n-- target met at a margin of zero, against the dev tables' own numbers (issue #88) --")
    met = tables["parts"]["target_met"]
    check(list(met["points"]) == [k for k in tables["parts"] if "mixes" in tables["parts"][k] and k != "references"],
          "target met covers every policy's part, as built")  # fmt: skip
    ok, routed = True, 0
    for section in met["points"]:
        for mix, by_target in committed_of(by_key, section)["mixes"].items():
            for target, row in by_target.items():
                found = met["points"][section][mix][target]
                if row["fell_back"]:
                    ok &= found["reading"] == "falls back to A" and found["difference"] is None
                    continue
                routed += 1
                # dev's own tables: the walk's step cleared the target by its margin, so quality less target is not negative,
                # and the share of resamples meeting it is the committed captured.met
                ok &= (
                    found["reading"] == "met"
                    and found["difference"]["value"] == row["quality"]["value"] - row["target"]
                )
                ok &= found["met_share"] == row["captured"]["met"] and row["quality"]["value"] - row["target"] >= 0
    check(ok, "dev: each routed point is met at a margin of zero (its walk cleared the target by a margin), its difference is "
              "the committed quality less the committed target, and its share of resamples is the committed captured.met; "
              "each fall back reads falls back to A")  # fmt: skip
    wanted = sum(
        not p["fell_back"]
        for key in met["points"]
        for p in (q for mixed in committed_of(by_key, key)["mixes"].values() for q in mixed.values())
    )
    check(met["counts"]["all"]["routes"] == routed == wanted and met["counts"]["all"]["missed"] == 0
          and met["counts"]["all"]["met"] == routed,
          f"the counts: {routed} routed points, all met on dev (the rows' own fell_back flags), none missed")  # fmt: skip
    head = tables["parts"]["headline"]
    plan = headline()
    check(head["cascade"] == plan.best == cascade_section(HAIKU, JEV_GATE) and head["label"] == CASCADES[plan.best]
          and head["headline"]["reading"] == "met" and head["dev"] == {"savings_share": plan.saves, "met_share": plan.met}
          and head["headline"]["met_share"] == plan.met and head["headline"]["savings_vs_A"]["share"] == plan.saves,
          f"the headline on dev's own matrix is haiku/Jev at realistic 99%: met, saving {plan.saves:.1%} and meeting its target "
          f"in {plan.met:.2%} of resamples, as A19 item 6 quotes (read from the committed table by analysis/final_design.py)")  # fmt: skip

    # A19 item 6's expected misses, which no policy's own part could check until every section was built: the sum of 1 less the share
    # of resamples meeting the target over the 65 covered routed points (G's and G+'s, and every cascade's) is the plan's dev count
    misses = expected_misses()
    covered = [
        (section, mix, target)
        for section, by_mix in met["points"].items()
        if section in ("jev", NAMED_ROUTERS[G_PLUS]) or section in CASCADES
        for mix, by_target in by_mix.items()
        for target, found in by_target.items()
        if found["met_share"] is not None
    ]
    check(len(met["points"]) == 19 and len(covered) == misses.covered
          and abs(sum(1 - met["points"][s][m][t]["met_share"] for s, m, t in covered) - misses.dev_misses) < 1e-9,
          f"target met covers all 19 sections, and over its {len(covered)} covered routed points the shares of resamples missing "
          f"the target sum to A19 item 6's {misses.dev_misses:.2f} expected misses on dev")  # fmt: skip

    comparisons_and_final_table(tables)

    print("\n-- the parts without committed numbers, against stored counts --")
    stops = tables["parts"]["stops"]
    expected = {
        "haiku": {"wrong": 111},
        "sonnet": {"wrong": 102, "max_tokens": 1},
        "opus_low": {"wrong": 74, "refusal": 1},
        "opus": {"wrong": 73, "refusal": 1},
    }
    non_correct = {config: {s: n for s, n in counts.items() if s != "correct"} for config, counts in stops.items()}
    check(non_correct == expected, f"the stops part's non-correct counts are the dev matrix's own {non_correct}")
    check(sum(sum(counts.values()) for counts in stops.values()) == 2976 and all(
        sum(counts.values()) == 744 for counts in stops.values()), "each config holds 744 rows, 2,976 in all")  # fmt: skip

    print("\n-- the words --")
    dev_tasks = tables["parts"]["references"]["tasks"]
    check(dev_tasks["ifeval"]["warning"] == task_warning("ifeval", 157) and "dev items" in dev_tasks["ifeval"]["warning"],
          "dev's IFEval entry keeps the plan's dev words")  # fmt: skip
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
