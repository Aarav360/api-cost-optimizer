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
class Backends:
    sandbox: SandboxRunner | None = None  # the grader's code runner: a stand-in in CI, the real sandbox otherwise
    # The Anthropic client the liveness calls (and the live replay) use. Unlike the sandbox, None here is not "the real
    # one": a world that names backends and no client is refused by the liveness step, so a rehearsal that forgot its fake
    # can't reach the real API. The go's world names no backends at all (`World.backends` None), and gets the real client.
    anthropic: Any = None
