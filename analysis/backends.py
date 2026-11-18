"""The backends a world names: what the final evaluation calls out to (PLAN_v4.md Amendment A19, items 2 and 13).

`World.backends` holds one of these, or None for the real ones. The go's entry point builds a world with none; a
rehearsal in CI builds one whose fields are stand-ins (a local runner for the sandbox, later the fake Jev server, the fake
Anthropic client and the fake embedder), so the same runner runs on both with only the world differing.

A field is None while it is the real one: a ticket that adds a backend adds a field here, with that default, and
reads it in its own step (`run.world.backends`).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from routers.sandbox import ExecResult

SandboxRunner = Callable[..., ExecResult]  # `run_python(source, timeout_s, memory)`, as `routers/sandbox.py` has it


@dataclass(frozen=True)
class TrainedStandIn:
    """What stands in for F's real parts (#83): the embedding backend, and dev as the drift check reads it."""

    # An `EmbeddingBackend` (routers/trained.py) in place of the pinned model, for the drift check's re-embedding of dev and
    # for scoring test; None is the real model, loaded from the verified snapshot.
    embedder: Any = None
    # A function giving the drift check's dev side (`analysis/trained_drift.py`'s `DevSide`): dev's requests, folds, router-score
    # table as dev had it and results matrix. None is the real one, read from dev's stored records.
    dev: Callable[[], Any] | None = None


@dataclass(frozen=True)
class Backends:
    sandbox: SandboxRunner | None = None  # the grader's code runner: a stand-in in CI, the real sandbox otherwise
    trained: TrainedStandIn | None = None  # F's embedder and dev side: stand-ins in a rehearsal, None for the real ones
