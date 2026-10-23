"""Check F and F-train on dev (PLAN_v4.md §8 3b; Amendment A11, items 9, 12, 13 and 15).

Run from the project root:  uv run python -m analysis.check_trained_dev

Needs the dev embedding cache, the fold weights and (for the walk) the dev results matrix, none of which CI has, so
the hook runs it where they are present, like check_stage4. It reads train and dev only, through the reader, and
never a test item.

- F's dev router-score table gives each dev item its router scores from the fold model its fold-file fold names, which
  never saw the item; they equal hand-worked ones (the weights file's own floats, a plain dot product and sigmoid), they
  are not the scores of another fold model (known positive), every overhead is an explicit $0, and the table passes
  the table's refusal;
- F-train's dev table is one model's, scored in the same way; it differs from F's;
- with the model snapshot (the hook): the table equals `score_with` over the reader's dev requests, each embedded
  again, within the drift limit (item 19), with a known positive;
- the per-fold quantiles: each fold model's router-score quantiles per config on its held-out items, from the table's
  own numbers, in order and inside 0 to 1, and over the fold file's held-out items.
"""

import json
import sys
from collections.abc import Callable

import numpy as np

from analysis import fit_trained_router as fit
from analysis import trained_table
from analysis.benchmark import read_rows
from analysis.folds import N_FOLDS, read_folds
from analysis.measure_trained_router import drift_limit
from analysis.router_table import RouterScoreTable, TableError
from harness.client import PROJECT_ROOT
from routers import trained
from routers.ladder import FRONTIER, SCORED
from routers.request import from_row
from routers.trained import TrainedRouter

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


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

    snapshot = trained.SNAPSHOT / "model.safetensors"
    if snapshot.exists():
        print("\n-- the table against score_with over the reader's dev requests (the hook only) --")
        settings_obj = trained.read_settings()
        router = TrainedRouter(settings_obj, trained.read_fold_weights())
        limit = drift_limit()
        gaps = []
        for entry, row in zip(table.items, rows):
            again = router.score_with(fold_of[entry.item_id], from_row(row))
            gaps.append(max(abs(again[c] - entry.router_scores[c]) for c in SCORED))
        check(len(gaps) == 744 and max(gaps) <= limit,
              f"all 744 dev requests embedded again and scored by score_with: within the drift limit {limit:.3g} "
              f"(largest gap {max(gaps):.2e}, {sum(g == 0 for g in gaps)} identical)")  # fmt: skip
        shifted = [
            max(
                abs(router.score_with((fold_of[e.item_id] + 1) % N_FOLDS, from_row(r))[c] - e.router_scores[c])
                for c in SCORED
            )
            for e, r in list(zip(table.items, rows))[:20]
        ]
        check(
            max(shifted) > limit, "known positive: score_with from another fold model is further than the drift limit"
        )
    else:
        print("\n(the model snapshot is missing: the score_with comparison is skipped)")


def main() -> int:
    needed = [fit.cache_paths("dev")[0], trained.FOLD_WEIGHTS_FILE, PROJECT_ROOT / "results" / "dev_matrix.sqlite"]
    if not all(p.exists() for p in needed):
        print("(the dev cache, the fold weights or the dev matrix is missing: nothing to check here)")
        return 0
    table_checks()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
