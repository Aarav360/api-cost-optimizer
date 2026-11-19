"""Check the rehearsal in the real sandbox (PLAN_v4.md Amendment A19, item 13; issue #79).

Run from the project root:  uv run python -m analysis.check_final_run_sandbox

Needs Docker and the pinned sandbox image, so the hook runs it and CI doesn't (`HOOK_ONLY` in
`analysis/check_guardrails.py`). It is `analysis/check_final_run.py`'s made-up world, run through the same runner with no
stand-in sandbox in the world, so the grader runs MBPP+'s made-up answers in the real one. The same hand-worked numbers
must come out, and the tables must equal those of the run with the stand-in: the stand-in is a local interpreter, and
this shows the real sandbox agrees with it on these answers. No results/, no Jev or Anthropic call, no real store.

- the runner on the made-up world, whose backends name no sandbox, runs every step and grades all 32 answers;
- the grader ran MBPP+'s eight answers in the real sandbox (`routers.sandbox.run_python` is called eight times, and the
  stand-in none), and the statuses of those eight are the hand-worked ones: code that adds (or multiplies) as the tests
  ask is correct, and code that subtracts (or adds, for the multiplying item) is wrong;
- every row of the results matrix has its hand-worked status, success and cost, as in the CI rehearsal;
- the tables equal the CI rehearsal's tables (a run with the stand-in sandbox), number for number.
"""

import contextlib
import dataclasses
import io
import json
import sys
from unittest import mock

from analysis import check_final_run as ci
from analysis import final_grading, rehearsal_gates
from analysis.backends import Backends
from analysis.final_run import Outcome
from analysis.made_up_world import made_up_world
from analysis.rehearsal_world import content
from routers import sandbox

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def main() -> int:
    print("-- the made-up world, graded in the real sandbox --")
    try:
        sandbox._ensure_image()
    except sandbox.SandboxError as error:
        print(f"FAIL  the real sandbox isn't available: {error}")
        return 1
    with (
        made_up_world(content()) as made,
        made_up_world(content()) as stand_in_made,
        rehearsal_gates.stand_in(made.world.results) as gates,
        rehearsal_gates.stand_in(stand_in_made.world.results) as stand_in_gates,
    ):
        # The same runner, a world that names no sandbox: the grader's own import of the real one stands.
        real = ci.Rehearsal(made, gates)
        real.world = dataclasses.replace(
            real.world,
            backends=Backends(
                anthropic=real.anthropic, trained=real.embedder, jev_gate=gates.jev, gate_train=gates.gate_train
            ),
        )
        stand_in = ci.Rehearsal(stand_in_made, stand_in_gates)
        calls = mock.Mock(wraps=sandbox.run_python)
        with mock.patch("harness.evaluators.run_python", calls):  # counts the real sandbox's runs, and calls through
            outcome, _, _ = real.go()
        check(outcome == Outcome(), f"the go runs every step with no stand-in sandbox {outcome}")
        check(calls.call_count == 8, f"MBPP+'s eight answers ran in the real sandbox ({calls.call_count} runs)")
        check(real.sandbox_calls == 0, "and none ran in the stand-in")
        rows = real.rows()
        mbpp = {(row["item_id"], row["config"]): row["status"] for row in rows if row["source"] == "mbppplus"}
        expected = {
            (item, config): ci.STATUS.get((item, config), "correct")
            for item in ("mbppplus:made-test-0", "mbppplus:made-test-1")
            for config in ci.CONFIGS
        }
        check(mbpp == expected and list(expected.values()).count("wrong") == 2,
              f"MBPP+'s statuses from the real sandbox are the hand-worked ones {mbpp}")  # fmt: skip
        right = {(i, c): ci.SUCCESS[p][k] for p, i in enumerate(ci.ITEMS) for k, c in enumerate(ci.CONFIGS)}
        check(len(rows) == 32 and all(row["status"] == ci.STATUS.get((row["item_id"], row["config"]), "correct")
                                      and row["success"] == right[(row["item_id"], row["config"])] for row in rows),
              "all 32 rows' statuses and successes are the hand-worked ones")  # fmt: skip
        with contextlib.redirect_stdout(io.StringIO()):
            stand_in.go()
        check(real.tables == stand_in.tables and json.dumps(real.tables) == json.dumps(stand_in.tables),
              "the tables equal the CI rehearsal's, run with the stand-in sandbox")  # fmt: skip
        check(final_grading.matrix_path(real.world.results).is_file() and ci.leaks(real.log.split("tables: starts")[0]) == [],
              "no marker is in what was logged before the tables step")  # fmt: skip
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
