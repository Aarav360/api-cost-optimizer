"""Check F and F-train on dev (PLAN_v4.md §8 3b; Amendment A11, items 9, 12, 13 and 15).

Run from the project root:  uv run python -m analysis.check_trained_dev

Needs the dev embedding cache, the fold weights, the model snapshot (F's table embeds the dev requests live) and, for
the walk, the dev results matrix, none of which CI has, so the hook runs it where they are present, like check_stage4.
It reads train and dev only, through the reader, and never a test item.

- F's dev router-score table gives each dev item its router scores from the fold model its fold-file fold names, which
  never saw the item; they equal hand-worked ones (the weights file's own floats, a plain dot product and sigmoid), they
  are not the router scores of another fold model (known positive), every overhead is an explicit $0, and the table passes
  the table's refusal;
- F-train's dev table is one model's, scored in the same way; it differs from F's;
- F's table is built through `score_with` (the syntax tree shows the call), with each item's fold from the fold file,
  and equals what the dev embedding cache's vectors give through the same fold models, within the drift limit
  (item 19), with a known positive;
- the per-fold quantiles: each fold model's router-score quantiles per config on its held-out items, from the table's
  own numbers, in order and inside 0 to 1, and over the fold file's held-out items.
"""

import ast
import copy
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from analysis import fit_trained_router as fit
from analysis import matrix as results_matrix
from analysis import trained_table, trained_walk
from analysis.benchmark import read_rows
from analysis.bootstrap import resample
from analysis.folds import N_FOLDS, read_folds
from analysis.measure_trained_router import drift_limit
from analysis.no_backend import NoBackend
from analysis.operating_points import (
    OPERATING_POINTS_FILE,
    OperatingPointsError,
    check_sections,
    read_section,
    section_hash,
)
from analysis.plan import PLAN, amendment_item
from analysis.references import ROUNDING, TARGETS
from analysis.router_names import NAMED_ROUTERS, E, F
from analysis.router_panels import router_part_of_mix_panel, step_outcomes
from analysis.router_table import RouterScoreTable, TableError
from analysis.rules_walk import MIXES, dev_walks
from analysis.simulator import always, load_mixes, simulate
from analysis.walk import ALL_STEPS_UP_TO, QUANTILE_STEPS, item_breakpoints, step_thresholds
from harness.client import PROJECT_ROOT
from routers import trained
from routers.ladder import FRONTIER, LADDER, SCORED
from routers.request import from_row
from routers.trained import TrainedRouter

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


TABLE_MODULE = PROJECT_ROOT / "analysis" / "trained_table.py"


def calls_in(path: Path, function: str, name: str) -> bool:
    """Whether the module's `function` calls something named `name` (a method or a function), as its syntax tree has it."""
    tree = ast.parse(path.read_text())
    (node,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function]
    return any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == name
               for call in ast.walk(node))  # fmt: skip


def refused(call: Callable[[], object]) -> bool:
    try:
        call()
    except TableError:
        return True
    return False


def by_hand(heads: dict, vector: np.ndarray, schema: bool, settings: dict) -> dict[str, float]:
    """A router score worked out from the JSON files' floats alone: length-normalize, standardize, append the flag,
    one dot product and a sigmoid per config."""
    unit = vector / np.sqrt(np.sum(vector**2))
    x = np.append((unit - np.array(settings["standardizer"]["mean"])) / np.array(settings["standardizer"]["scale"]),
                  1.0 if schema else 0.0)  # fmt: skip
    return {c: float(1 / (1 + np.exp(-(np.dot(heads[c]["coef"], x) + heads[c]["intercept"])))) for c in SCORED}


def table_checks() -> None:
    print("-- F's dev router-score table (Amendment A11, items 9 and 15) --")
    rows = read_rows(split="dev")
    ids = [row["item_id"] for row in rows]
    cache = fit.read_cache("dev")
    fold_of = read_folds("dev")
    table = trained_table.dev_table()
    check(len(table.items) == len(rows) == 744, f"the table has an entry for each of the {len(rows)} dev requests")
    check([entry.item_id for entry in table.items] == ids, "its entries are in the reader's dev order")
    check(all(set(e.router_scores) == set(SCORED) for e in table.items),
          f"every entry scores {', '.join(SCORED)} and never {FRONTIER}")  # fmt: skip
    check(all(e.overhead == 0.0 and isinstance(e.overhead, float) for e in table.items),
          "every overhead is an explicit $0: F runs locally")  # fmt: skip
    check(RouterScoreTable.build(list(table.items), ids).items == table.items,
          "the table passes the table's refusal against the reader's dev items")  # fmt: skip
    check(refused(lambda: RouterScoreTable.build(list(table.items)[1:], ids)),
          "the refusal fires on a table that leaves out a dev item")  # fmt: skip

    settings = json.loads(trained.SETTINGS_FILE.read_text())
    weights = json.loads(trained.FOLD_WEIGHTS_FILE.read_text())["folds"]
    worst, wrong = 0.0, 0.0
    for i in range(0, len(rows), 7):  # every 7th dev item: 107 of them, covering every fold
        row, entry = rows[i], table.items[i]
        hand = by_hand(weights[fold_of[ids[i]]], cache.embeddings[i], row.get("output_schema") is not None, settings)
        worst = max(worst, max(abs(hand[c] - entry.router_scores[c]) for c in SCORED))
        other = by_hand(weights[(fold_of[ids[i]] + 1) % N_FOLDS], cache.embeddings[i], row.get("output_schema") is not None,
                        settings)  # fmt: skip
        wrong = max(wrong, max(abs(other[c] - entry.router_scores[c]) for c in SCORED))
    check(
        worst < 1e-9, f"107 items' router scores equal ones worked out from the files' floats (largest gap {worst:.1e})"
    )
    check(wrong > 1e-3, "known positive: the next fold model's router scores for the same items are not the table's")
    check(set(fold_of.values()) == set(range(N_FOLDS)) and len(fold_of) == len(rows),
          "every dev item has a fold from the fold file, all five in use")  # fmt: skip

    print("\n-- F-train's dev router-score table (Amendment A11, item 12) --")
    f_train = trained_table.f_train_table()
    check([e.item_id for e in f_train.items] == ids and all(e.overhead == 0.0 for e in f_train.items),
          "it is in the reader's dev order with an explicit $0 overhead")  # fmt: skip
    check(RouterScoreTable.build(list(f_train.items), ids).items == f_train.items, "it passes the table's refusal too")
    one = json.loads(fit.F_TRAIN_WEIGHTS.read_text())
    gap = max(
        max(abs(by_hand(one, cache.embeddings[i], rows[i].get("output_schema") is not None, settings)[c]
                - f_train.items[i].router_scores[c]) for c in SCORED)
        for i in range(0, len(rows), 7)
    )  # fmt: skip
    check(
        gap < 1e-9, f"its router scores are one model's, F-train's weights file, for every item (largest gap {gap:.1e})"
    )
    check(any(a.router_scores != b.router_scores for a, b in zip(table.items, f_train.items)),
          "known positive: it is not F's table")  # fmt: skip

    print("\n-- the per-fold quantiles (Amendment A11, items 9 and 12) --")
    quantiles = trained_table.fold_quantiles(table, fold_of)
    check(sorted(quantiles) == list(range(N_FOLDS)), "one reading per fold model")
    held = {k: [e for e in table.items if fold_of[e.item_id] == k] for k in range(N_FOLDS)}
    check(all(q["items"] == len(held[k]) for k, q in quantiles.items()) and sum(len(v) for v in held.values()) == 744,
          f"each reading is over its fold's held-out items: {[len(held[k]) for k in range(N_FOLDS)]}")  # fmt: skip
    ok = True
    for k, reading in quantiles.items():
        for config in SCORED:
            values = reading["by_config"][config]
            scores = sorted(e.router_scores[config] for e in held[k])
            ok = (
                ok
                and list(values) == list(trained_table.QUANTILES)
                and values[0] == scores[0]
                and values[1.0] == scores[-1]
            )
            ok = ok and all(0 <= v <= 1 for v in values.values()) and list(values.values()) == sorted(values.values())
            median = float(np.median(scores))
            ok = ok and abs(values[0.5] - median) < 1e-12
    check(ok, "per fold and config the quantiles run from the held-out minimum to its maximum, in order, inside 0 to 1, "
              "with the median the held-out median")  # fmt: skip

    print("\n-- the table is built through score_with (Amendment A11, item 10) --")
    check(calls_in(TABLE_MODULE, "dev_table", "score_with") and not calls_in(TABLE_MODULE, "dev_table", "score_features"),
          "F's table calls the router's own score_with on each dev request, and not the heads' arithmetic itself")  # fmt: skip
    check(calls_in(TABLE_MODULE, "f_train_table", "score_features") and not calls_in(TABLE_MODULE, "f_train_table", "score_with"),
          "known positive: the search finds F-train's table, a diagnostic, scoring from the cache instead")  # fmt: skip
    # The independent route: the same router, fed the dev embedding cache's vectors, never embedding.
    cached = TrainedRouter(trained.read_settings(), trained.read_fold_weights(), backend=NoBackend())
    limit = drift_limit()
    gaps, moved = [], []
    for i, (entry, row) in enumerate(zip(table.items, rows, strict=True)):
        fold, request = fold_of[entry.item_id], from_row(row)
        features = cached.features(cache.embeddings[i], request)
        gaps.append(max(abs(cached.score_features(fold, features)[c] - entry.router_scores[c]) for c in SCORED))
        other = cached.score_features((fold + 1) % N_FOLDS, features)
        moved.append(max(abs(other[c] - entry.router_scores[c]) for c in SCORED))
    check(len(gaps) == 744 and max(gaps) <= limit,
          f"all 744 dev requests embedded live and scored by score_with equal the cache's vectors scored by the same "
          f"fold model, within the drift limit {limit:.3g} (largest gap {max(gaps):.2e}, "
          f"{sum(g == 0 for g in gaps)} identical)")  # fmt: skip
    check(max(moved) > limit, "known positive: the next fold model's router scores are further than the drift limit")


def certified_steps_clear(section: dict[str, Any]) -> bool:
    """Every certified step (A included) clears its point's target by its own margin, in a section's six walks."""
    for by_target in section.values():
        for point in by_target.values():
            steps = point["steps"]
            stop = next((i for i, s in enumerate(steps) if not s["clears"]), len(steps))
            if not all(s["quality"] - point["target"] >= s["margin"] - ROUNDING for s in steps[:stop]):
                return False
    return True


def walk_checks() -> None:
    print("\n-- the walk on dev and F's section of the operating-points file (Amendment A11, items 9, 13 and 15) --")
    name = trained_walk.ROUTER
    check(name == NAMED_ROUTERS[F] == "trained", f"F's section is {name!r}, the name the allowlist gives F")
    section = read_section(name)
    check(list(section) == list(MIXES) and all(set(b) == {"100%", "99%", "97%"} for b in section.values()),
          "F's section holds both mixes and all three targets: six operating points")  # fmt: skip
    rebuilt = trained_walk.rebuilt_section()
    check(rebuilt == section and section_hash(rebuilt) == section_hash(section),
          "the walk on dev reproduces F's section of the operating-points file, so its hash too")  # fmt: skip
    nudged = copy.deepcopy(section)
    nudged["realistic"]["99%"]["margin"] += 1e-12
    check(section_hash(nudged) != section_hash(section) and nudged != rebuilt,
          "known positive: a margin changed in the twelfth decimal changes the section's hash and fails the comparison")  # fmt: skip
    try:
        check_sections(OPERATING_POINTS_FILE, required=(NAMED_ROUTERS[E], name))
        allowed = True
    except OperatingPointsError:
        allowed = False
    check(allowed, "the allowlist passes with F's section present, and nothing else but E's")
    check(NAMED_ROUTERS[E] in json.loads(OPERATING_POINTS_FILE.read_text()),
          "E's section is still in the file")  # fmt: skip
    item18 = amendment_item(PLAN.read_text(), "A9", 18)
    recorded = re.search(r"E's section of `routers/operating_points.json`, written with sorted keys, has the 64-digit "
                         r"sha256 `([0-9a-f]{64})`", item18)  # fmt: skip
    check(recorded is not None and section_hash(read_section(NAMED_ROUTERS[E])) == recorded[1],
          "E's section is unchanged: its sha256 is still A9 item 18's")  # fmt: skip

    table = trained_table.dev_table()
    pooled = sorted(b for e in table.items for b in item_breakpoints(e.router_scores))
    steps = step_thresholds(table)
    check(len(set(pooled)) > ALL_STEPS_UP_TO and len(steps) <= QUANTILE_STEPS,
          f"A9 item 9's quantile steps apply: {len(set(pooled))} distinct breakpoints, more than {ALL_STEPS_UP_TO}, "
          f"give {len(steps)} steps")  # fmt: skip
    for mix, by_target in section.items():
        for key, point in by_target.items():
            filed = point["steps"]
            where = f"{mix} mix, {key} of A"
            stop = next((i for i, s in enumerate(filed) if not s["clears"]), len(filed))
            certified, missed = filed[:stop], filed[stop:]
            check(filed[0]["threshold"] is None and filed[0]["margin"] == 0 and float(key.rstrip("%")) / 100 in TARGETS,
                  f"{where}: the walk starts at A, threshold null, margin 0")  # fmt: skip
            check([s["threshold"] for s in filed[1:]] == list(reversed(steps))[: len(filed) - 1],
                  f"{where}: its {len(filed) - 1} steps tried are the table's quantile steps, highest threshold first")  # fmt: skip
            check(all(s["quality"] - point["target"] >= s["margin"] - ROUNDING for s in certified),
                  f"{where}: all {len(certified)} certified steps (A included) clear the target by their margin")  # fmt: skip
            check(not missed or (len(missed) == 1 and missed[0]["quality"] - point["target"] < missed[0]["margin"] - ROUNDING),
                  f"{where}: the walk ends at the first step that misses its target, or tried them all")  # fmt: skip
            cheapest = certified[0]
            for step in certified[1:]:
                if step["cost_per_1k"] < cheapest["cost_per_1k"] - ROUNDING:
                    cheapest = step
            check(point["threshold"] == cheapest["threshold"] and point["margin"] == cheapest["margin"]
                  and point["fell_back"] == (point["threshold"] is None),
                  f"{where}: the operating point is the cheapest certified step: "
                  + ("A, it falls back" if point["fell_back"] else f"threshold {point['threshold']:.4f}"))  # fmt: skip
    check(certified_steps_clear(section), "every certified step in F's section clears its target by its margin")
    broken = copy.deepcopy(section)
    broken["uniform"]["97%"]["steps"][1]["quality"] = broken["uniform"]["97%"]["target"] - 0.01
    check(not certified_steps_clear(broken),
          "known positive: a certified step with quality below the target is found")  # fmt: skip

    print("\n-- the captured fraction (Amendment A9 item 15, for F) --")
    dev = results_matrix.load().split("dev")
    mixes = load_mixes()
    fixed = {config: simulate(dev, always(dev, config)) for config in LADDER}
    draws = resample(dev)
    walks = dev_walks(dev, table, draws)
    outcomes = step_outcomes(dev, table, fixed[FRONTIER])
    for mix in MIXES:
        savings = draws.savings(dev, fixed[FRONTIER], mixes[mix])
        part = router_part_of_mix_panel(draws, savings, outcomes, walks[mix], fixed, mixes[mix])
        for key, point in part["operating_points"].items():
            captured = point["captured"]
            bounds = [captured[b] for b in ("value", "low", "high") if captured[b] is not None]
            fell_back = section[mix][key]["fell_back"]
            check(all(0 <= b <= 1 for b in bounds) and 0 <= captured["met"] <= 1
                  and (captured["value"] == 0 if fell_back else captured["value"] is not None and captured["value"] > 0),
                  f"{mix} mix, {key} of A: the captured fraction, interval included, is at most 1"
                  + (" and 0 for a fallback" if fell_back else f" ({captured['value']:.2f})"))  # fmt: skip
            check(point["threshold"] == section[mix][key]["threshold"],
                  f"{mix} mix, {key} of A: the point the chart would draw is the file's")  # fmt: skip


def main() -> int:
    needed = [fit.cache_paths("dev")[0], trained.FOLD_WEIGHTS_FILE, trained.SNAPSHOT / "model.safetensors",
              PROJECT_ROOT / "results" / "dev_matrix.sqlite"]  # fmt: skip
    if not all(p.exists() for p in needed):
        print(
            "(the dev cache, the fold weights, the model snapshot or the dev matrix is missing: nothing to check here)"
        )
        return 0
    table_checks()
    if trained.FOLD_WEIGHTS_FILE.exists() and trained_walk.ROUTER in json.loads(OPERATING_POINTS_FILE.read_text()):
        walk_checks()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
