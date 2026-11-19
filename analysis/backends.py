"""The backends a world names: what the final evaluation calls out to (PLAN_v4.md Amendment A19, items 2 and 13).

`World.backends` holds one of these, or None for the real ones. The go's entry point builds a world with none; a
rehearsal in CI builds one whose fields are stand-ins (a local runner for the sandbox, later the fake Jev server, the fake
Anthropic client and the fake embedder), so the same runner runs on both with only the world differing.

A field is None while it is the real one: a ticket that adds a backend adds a field here, with that default, and
reads it in its own step (`run.world.backends`).
"""

from collections.abc import Callable, Sequence
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
    # The Anthropic client the liveness calls (and the live replay) use. Unlike the sandbox, None here is not "the real
    # one": a world that names backends and no client is refused by the liveness step, so a rehearsal that forgot its fake
    # can't reach the real API. The go's world names no backends at all (`World.backends` None), and gets the real client.
    anthropic: Any = None
    trained: TrainedStandIn | None = None  # F's embedder and dev side: stand-ins in a rehearsal, None for the real ones
    # The Jev endpoint for the Jev gate's drift resends and its calls on test (`analysis/final_gates.py`): a factory of a
    # context-managed `routers.jev.JevClient`, `with backends.jev_gate() as client:`. (G's and G+'s are `jev`, #84's.) As with the Anthropic client, None in a world
    # that names backends is refused, so a rehearsal can't reach the real endpoint; the go's world (no backends) builds the
    # real client from the pinned settings and the key in the environment.
    jev_gate: Callable[[], Any] | None = None
    # Train's answers for the Jev gate's drift resend: given the noise sample's (item, config) pairs, the `GateInput`s
    # (request, answer, stop reason) the resend sends again. None is the real reader (`collect_gates.gate_inputs("train")`);
    # a rehearsal gives made-up ones, since it holds no train matrix.
    gate_train: Callable[[Sequence[tuple[str, str]]], Sequence[Any]] | None = None
    # The sandbox image's digest a free-check record carries when `sandbox` is a stand-in (the real one is read from Docker).
    image_digest: str | None = None
