"""An embedding backend that can't embed (PLAN_v4.md §8 3b; Amendment A11, item 10).

A `TrainedRouter` built to read feature rows or cached vectors, as F-train's table and the learning curve's refits do,
needs a backend only to satisfy its constructor. This one refuses to embed, so a stray `score` call is a loud error and
not a silent load of the model.
"""

import numpy as np


class NoEmbeddingError(RuntimeError):
    """Something asked a router built without a model to embed text."""


class NoBackend:
    """Implements `routers.trained.EmbeddingBackend` for a router that reads features, never text."""

    def encode(self, text: str) -> list[int]:
        raise NoEmbeddingError("this router reads features it was handed; it never embeds")

    def cls_vector(self, ids: list[int]) -> np.ndarray:
        raise NoEmbeddingError("this router reads features it was handed; it never embeds")
