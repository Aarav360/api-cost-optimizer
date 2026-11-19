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
class JevStandIn:
    """What a rehearsal gives G's and G+'s steps in place of the real Jev (`analysis/rehearsal_jev.py` builds one): the
    settings on a fake server's address (with a noise sample of made-up train items), G+'s weights, and a client for the
    fake with its waits injected. The go's world names none and gets the real settings, weights and key."""

    loaded: Any  # `analysis.collect_jev.Loaded`
    weights: Any  # `routers.jev.GPlusWeights`
    client: Callable[[], Any]  # a `routers.jev.JevClient` on the fake
    # The results folder holding Stage 6's train records (`jev/<router>_train.jsonl`), which the drift check reads its first
    # two passes from. None is the world's own results folder, where the go's real ones are; a rehearsal keeps its made-up
    # ones in a folder of their own, so a refused unlock still leaves the world's results folder uncreated.
    train_results: Any = None
    # The Jev gate's client (`analysis/final_gates.py`, #85): a factory of a context-managed `routers.jev.JevClient`, `with
    # backends.jev.gate() as client:`, for the gate's drift resends and its calls on test. One field for both kinds of Jev
    # call, the Stage 8 review's O4 (#84 and #85 each added a stand-in of their own). None in a world that names a Jev
    # stand-in is refused by the Jev gate's steps, as the Jev routers' is, so a rehearsal can't reach the real endpoint.
    gate: Callable[[], Any] | None = None


@dataclass(frozen=True)
class Backends:
    sandbox: SandboxRunner | None = None  # the grader's code runner: a stand-in in CI, the real sandbox otherwise
    # The Anthropic client the liveness calls (and the live replay) use. Unlike the sandbox, None here is not "the real
    # one": a world that names backends and no client is refused by the liveness step, so a rehearsal that forgot its fake
    # can't reach the real API. The go's world names no backends at all (`World.backends` None), and gets the real client.
    anthropic: Any = None
    trained: TrainedStandIn | None = None  # F's embedder and dev side: stand-ins in a rehearsal, None for the real ones
    # Train's answers for the Jev gate's drift resend: given the noise sample's (item, config) pairs, the `GateInput`s
    # (request, answer, stop reason) the resend sends again. None is the real reader (`collect_gates.gate_inputs("train")`);
    # a rehearsal gives made-up ones, since it holds no train matrix.
    gate_train: Callable[[Sequence[tuple[str, str]]], Sequence[Any]] | None = None
    # The sandbox image's digest a free-check record carries when `sandbox` is a stand-in (the real one is read from Docker).
    image_digest: str | None = None
    # The Jev endpoint every Jev call uses: G's and G+'s drift resends and test calls, and (its `gate`) the Jev gate's. As with
    # the Anthropic client, None is not "the real one": a world that names backends and no Jev stand-in is refused by the Jev
    # steps, so a rehearsal can't reach the real service; the go's world (no backends) builds the real client from the pinned
    # settings and the key in the environment.
    jev: JevStandIn | None = None


def stand_in_of(backends: Backends | None, field: str) -> Any:
    """The stand-in a world names for `field`, or None for the real one: the go's world names no backends at all, and a
    rehearsal leaves None a backend it doesn't fake (the sandbox, the embedder, train's answers: the real one then stands)."""
    return None if backends is None else getattr(backends, field)


def required(backends: Backends | None, field: str, what: str, refusal: type[Exception]) -> Any:
    """The stand-in a world names for `field`, None for the go's world (no backends: the real one), and `refusal` for a world
    that names backends but no `field`, so a rehearsal that forgot its fake can't reach the real service (the Anthropic client,
    the Jev endpoint). `what` names it in the message."""
    stand_in = stand_in_of(backends, field)
    if backends is not None and stand_in is None:
        raise refusal(no_stand_in(what))
    return stand_in


def no_stand_in(what: str) -> str:
    return f"the world names backends but no {what}: only the go's world, which names none, gets the real one"
