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
- dev's IFEval and MBPP+ entries carry the plan's dev words, unchanged by the test wording.
"""

import copy
import json
import sys
from typing import Any

from analysis import offline_half
from analysis.final_design import headline
from analysis.router_names import CASCADES, HAIKU, JEV_GATE, cascade_section
from analysis.task_warnings import task_warning

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

    print("\n-- E and F at their frozen operating points (issue #83) --")
    by_key = {part.key: part for part in offline_half.PARTS}
    check(all(by_key[k].committed is not None for k in ("rules", "trained")),
          "E's and F's parts each name their committed dev numbers, so the loop above checked them")  # fmt: skip
    for key, routes in (("rules", 4), ("trained", 3)):
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
