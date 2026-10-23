"""A fake embedding backend for the trained router's checks (PLAN_v4.md §8 3b; Amendment A11, items 4 and 10).

It lives in `analysis/`, beside the checks, so it is never in the router's import closure and never reaches the freeze
or Stage 8 (`analysis/check_trained_router.py` proves that). It stands in for the tokenizer and model so a check can
work out a router score by hand: each text maps to a chosen vector, and the backend remembers what reached it.
"""

from collections.abc import Mapping, Sequence

import numpy as np

CLS, SEP = 1, 2  # the special tokens at each end, as the real tokenizer puts them
SHIFT = 10  # a character's token is its code point plus this, so no character is a special token


class FakeBackend:
    """Implements `routers.trained.EmbeddingBackend`.

    `vectors` maps the exact text the router should embed to the CLS vector it gets back, in `dtype`. A text
    that isn't in `vectors` gets `default` if one is given and is a KeyError if not, so a check can't pass by
    embedding the wrong text unnoticed. `embedded` lists the token ids that reached `cls_vector`, in order.
    """

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]],
        dtype: type[np.floating] = np.float64,
        default: Sequence[float] | None = None,
    ) -> None:
        self.vectors = dict(vectors)
        self.dtype = dtype
        self.default = default
        self.embedded: list[list[int]] = []

    def encode(self, text: str) -> list[int]:
        return [CLS, *(ord(char) + SHIFT for char in text), SEP]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i - SHIFT) for i in ids[1:-1])

    def cls_vector(self, ids: list[int]) -> np.ndarray:
        self.embedded.append(list(ids))
        text = self.decode(ids)
        if text in self.vectors:
            return np.array(self.vectors[text], dtype=self.dtype)
        if self.default is not None:
            return np.array(self.default, dtype=self.dtype)
        raise KeyError(f"the fake backend has no vector for {text!r}")
