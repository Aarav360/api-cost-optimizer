"""F's stand-ins for the rehearsal: a fake embedder and a made-up dev side (PLAN_v4.md Amendment A11, item 17; A19, items 2 and
13; issue #83).

`analysis/fake_backend.py`'s `FakeBackend` maps chosen texts to chosen vectors, which is the right size for a hand-worked
score and the wrong one for a rehearsal that scores every request of a made-up world. `SeededBackend` is a `FakeBackend` whose
vector for any text is derived from the text (a seeded Gaussian, 768 wide, as the model's), plus `shift` times a second vector
derived from the text too: the same machine at `shift = 0`, and a machine whose arithmetic has moved by `shift` otherwise. One
token per character (the fake's `encode`), so a long enough made-up request is truncated by F's real window.

`dev_side` is the drift check's made-up dev (`analysis/trained_drift.py`): eight requests that are none of the benchmark's,
five folds dealt round, the table "dev had" scored through `SeededBackend(0)`, and a results matrix of the same eight items
with hand-chosen costs and successes (two per task, since the mixes need every task), so quality and cost can be worked by hand.

Nothing here reads the benchmark, `results/` or a real item.
"""

import hashlib
from collections.abc import Callable

import numpy as np

from analysis.backends import TrainedStandIn
from analysis.fake_backend import FakeBackend
from analysis.matrix import Matrix
from analysis.router_table import LOCAL_OVERHEAD, ItemRouterScores, RouterScoreTable
from analysis.trained_drift import DevSide
from routers import trained
from routers.request import Request
from routers.trained import TrainedRouter

WIDTH = 768  # the model's hidden size, and the standardizer's width
TASKS = ("ifeval", "banking77", "mmlu_pro", "mbppplus")
CONFIGS = ("haiku", "sonnet", "opus_low", "opus")
DEV_ITEMS = [f"{task}:made-dev-{n}" for task in TASKS for n in (0, 1)]
DEV_PROMPTS = {item: f"Made-up dev request number {k} about {item.split(':')[0]}." for k, item in enumerate(DEV_ITEMS)}
# Dollars per request by config, the same for every item, and each item's successes on (haiku, sonnet, opus_low, opus).
COST = {"haiku": 0.001, "sonnet": 0.004, "opus_low": 0.008, "opus": 0.010}
SUCCESS = [
    (1, 1, 1, 1),
    (0, 1, 1, 1),
    (1, 1, 1, 1),
    (0, 0, 1, 1),
    (1, 1, 1, 0),
    (0, 1, 1, 1),
    (1, 1, 1, 1),
    (0, 0, 1, 1),
]


def seeded(text: str, salt: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(f"{salt}:{text}".encode()).digest()[:8], "big")
    return np.random.default_rng(seed).standard_normal(WIDTH)


class SeededBackend(FakeBackend):
    """A fake embedder whose vector for any text is a function of the text, moved by `shift`. Implements
    `routers.trained.EmbeddingBackend`; `embedded` lists the token ids that reached it, as the fake's does."""

    def __init__(self, shift: float = 0.0) -> None:
        super().__init__({})
        self.shift = shift

    def cls_vector(self, ids: list[int]) -> np.ndarray:
        self.embedded.append(list(ids))
        text = self.decode(ids)
        return (seeded(text, "base") + self.shift * seeded(text, "moved")).astype(self.dtype)

    def texts(self) -> list[str]:
        """What reached the embedder, as text, in order."""
        return [self.decode(ids) for ids in self.embedded]


def dev_requests() -> dict[str, Request]:
    return {item: Request(prompt, None) for item, prompt in DEV_PROMPTS.items()}


def dev_matrix() -> Matrix:
    rows = [
        {"item_id": item, "split": "dev", "task": item.split(":")[0], "source_group": item, "config": config,
         "cost": COST[config], "success": SUCCESS[k][c]}
        for k, item in enumerate(DEV_ITEMS) for c, config in enumerate(CONFIGS)
    ]  # fmt: skip
    return Matrix.from_rows(rows)


def dev_side(table_backend: object | None = None) -> DevSide:
    """The made-up dev side. Its table is scored through `table_backend` (the machine dev was built on): a `SeededBackend(0)` by default."""
    backend = table_backend if table_backend is not None else SeededBackend(0.0)
    router = TrainedRouter(trained.read_settings(), trained.read_fold_weights(), backend=backend)  # type: ignore[arg-type]
    requests = dev_requests()
    fold_of = {item: k % 5 for k, item in enumerate(requests)}
    items = [ItemRouterScores(item, router.score_with(fold_of[item], request), LOCAL_OVERHEAD)
             for item, request in requests.items()]  # fmt: skip
    return DevSide(requests, fold_of, RouterScoreTable.build(items, list(requests)), dev_matrix)


def stand_in(shift: float = 0.0, table_backend: object | None = None) -> TrainedStandIn:
    """F's stand-ins for a rehearsal: an embedder moved by `shift` from the one dev's table was scored with."""
    side: Callable[[], DevSide] = lambda: dev_side(table_backend)  # noqa: E731
    return TrainedStandIn(embedder=SeededBackend(shift), dev=side)
