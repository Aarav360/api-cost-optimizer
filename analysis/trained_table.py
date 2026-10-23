"""F's and F-train's dev router-score tables, and the fold models' quantiles (PLAN_v4.md §8 3b; Amendment A11,
items 9, 11, 12 and 15).

Run from the project root:  uv run python -m analysis.trained_table

The first dev step after the fit commit: dev's tables for F and F-train, then the per-fold quantiles; the walks come
after, in `analysis/trained_walk.py`. It reads no outcome.

- F's table gives each dev item the router scores of its fold model, the one the fold file says is held out on it
  (`read_folds("dev")`), through the deployable router's own `features` and `score_features`. The vectors are the
  dev embedding cache's, which `score_with` computes again within the drift limit (item 19): the hook checks the table
  against `score_with` over every dev request, so table and router can't drift apart.
- F-train's table is the one model refit on train alone (item 12), in all five slots, scoring dev directly.
- Every overhead is an explicit $0: the model runs locally (item 11).
- `fold_quantiles` reports, per fold model and config, the router scores' quantiles on its held-out items (item 9).
"""

from collections.abc import Mapping, Sequence

import numpy as np

from analysis import fit_trained_router as fit
from analysis.benchmark import read_rows
from analysis.folds import N_FOLDS, read_folds
from analysis.router_table import ItemRouterScores, RouterScoreTable
from routers import trained
from routers.ladder import SCORED
from routers.request import from_row
from routers.trained import FoldWeights, TrainedRouter

OVERHEAD = 0.0  # F runs locally: no API call, no dollars (item 11)
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)  # the reported quantiles of a fold model's router scores


class _CachedOnly:
    """A backend that can't embed: the table reads cached vectors, so a stray call is a bug, not a silent load."""

    def encode(self, text: str) -> list[int]:
        raise RuntimeError("the dev tables read the embedding cache; they never embed")

    def cls_vector(self, ids: list[int]) -> np.ndarray:
        raise RuntimeError("the dev tables read the embedding cache; they never embed")


def _table(weights: FoldWeights, fold_of: Mapping[str, int] | None) -> RouterScoreTable:
    """A table from the router built on `weights`; `fold_of` None means fold 0's model for every item."""
    router = TrainedRouter(trained.read_settings(), weights, backend=_CachedOnly())
    rows = read_rows(split="dev")
    cache = fit.read_cache("dev")
    items = []
    for row, embedding in zip(rows, cache.embeddings, strict=True):
        fold = 0 if fold_of is None else fold_of[row["item_id"]]
        features = router.features(embedding, from_row(row))
        items.append(ItemRouterScores(row["item_id"], router.score_features(fold, features), OVERHEAD))
    return RouterScoreTable.build(items, [row["item_id"] for row in rows])


def dev_table() -> RouterScoreTable:
    """F's table over every dev item, in the reader's order: out-of-fold router scores, or the table's refusal."""
    return _table(trained.read_fold_weights(), read_folds("dev"))


def f_train_table() -> RouterScoreTable:
    """F-train's table over every dev item: one model, fitted on train alone (a diagnostic, never a policy)."""
    return _table(fit.read_f_train_weights(), None)


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
