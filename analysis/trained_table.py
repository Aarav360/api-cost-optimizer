"""F's and F-train's dev router-score tables, and the fold models' quantiles (PLAN_v4.md §8 3b; Amendment A11,
items 9, 11, 12 and 15).

Run from the project root:  uv run python -m analysis.trained_table

The first dev step after the fit commit: dev's tables for F and F-train, then the per-fold quantiles; the walks come
after, in `analysis/trained_walk.py`. It reads no outcome.

- F's table calls the deployable router's own `score_with` on every dev request, with the fold the fold file says is
  held out on it (`read_folds("dev")`), so table and router can't drift apart (item 10), as E's table calls E's
  `score`. It embeds the 744 dev requests live with the snapshot, about 30 s; the hook checks it against the router
  scores that the dev embedding cache gives.
- F-train's table is the one model refit on train alone (item 12), in all five slots, scoring dev from the embedding
  cache: it is a diagnostic, never a policy, and `score_with` would embed the 744 requests a second time.
- Every overhead is an explicit $0: the model runs locally (item 11).
- `fold_quantiles` reports, per fold model and config, the router scores' quantiles on its held-out items (item 9).
"""

import functools
from collections.abc import Mapping, Sequence

import numpy as np

from analysis import fit_trained_router as fit
from analysis.benchmark import read_rows
from analysis.folds import N_FOLDS, read_folds
from analysis.no_backend import NoBackend
from analysis.router_table import LOCAL_OVERHEAD, ItemRouterScores, RouterScoreTable
from routers import trained
from routers.ladder import SCORED
from routers.request import from_row
from routers.trained import TrainedRouter

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)  # the reported quantiles of a fold model's router scores


@functools.cache
def dev_table() -> RouterScoreTable:
    """F's table over every dev item, in the reader's order: each item's out-of-fold router scores from the router's
    own `score_with`, with the fold the fold file gives it, or the table's refusal.

    Embeds the dev requests with the verified snapshot (about 30 s), so a process builds it once.
    """
    router = TrainedRouter(trained.read_settings(), trained.read_fold_weights())
    fold_of = read_folds("dev")
    rows = read_rows(split="dev")
    items = [
        ItemRouterScores(row["item_id"], router.score_with(fold_of[row["item_id"]], from_row(row)), LOCAL_OVERHEAD)
        for row in rows
    ]
    return RouterScoreTable.build(items, [row["item_id"] for row in rows])


def f_train_table() -> RouterScoreTable:
    """F-train's table over every dev item: one model, fitted on train alone, scoring from the dev embedding cache (a
    diagnostic, never a policy). Fold 0 stands for the model in every slot, which all hold the same one."""
    router = TrainedRouter(trained.read_settings(), fit.read_f_train_weights(), backend=NoBackend())
    rows = read_rows(split="dev")
    embeddings = fit.read_cache("dev").embeddings
    items = [
        ItemRouterScores(
            row["item_id"], router.score_features(0, router.features(embedding, from_row(row))), LOCAL_OVERHEAD
        )
        for row, embedding in zip(rows, embeddings, strict=True)
    ]
    return RouterScoreTable.build(items, [row["item_id"] for row in rows])


def fold_quantiles(table: RouterScoreTable, fold_of: Mapping[str, int]) -> dict[int, dict]:
    """Per fold model: its number of held-out items and, per config, `QUANTILES` of its router scores on them.

    Numpy's default (linear) quantiles. The minimum and maximum are the fold's smallest and largest router scores.
    """
    out: dict[int, dict] = {}
    for k in range(N_FOLDS):
        held: Sequence[ItemRouterScores] = [e for e in table.items if fold_of[e.item_id] == k]
        out[k] = {
            "items": len(held),
            "by_config": {
                config: dict(zip(QUANTILES, np.quantile([e.router_scores[config] for e in held], QUANTILES).tolist()))
                for config in SCORED
            },
        }
    return out


def main() -> None:
    f, f_train = dev_table(), f_train_table()
    for name, table in (("F's (out-of-fold)", f), ("F-train's", f_train)):
        distinct = len({tuple(e.router_scores[c] for c in SCORED) for e in table.items})
        print(f"{name} dev router-score table: {len(table.items)} dev requests, overhead $0, {distinct} distinct "
              f"triples of router scores; it passes the table's refusal")  # fmt: skip
    print("\nEach fold model's router scores on its held-out dev items, by quantile "
          f"({', '.join(f'{q:.0%}' for q in QUANTILES)}):")  # fmt: skip
    for k, reading in fold_quantiles(f, read_folds("dev")).items():
        for config in SCORED:
            values = reading["by_config"][config].values()
            print(f"  fold model {k} ({reading['items']} items) {config:9}: " + "  ".join(f"{v:.4f}" for v in values))


if __name__ == "__main__":
    main()
