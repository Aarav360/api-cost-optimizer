"""Check the free checks on test in the real sandbox (PLAN_v4.md Amendment A19, item 13; A17 items 4 and 26; issue #77, #85).

Run from the project root:  uv run python -m analysis.check_final_gates_sandbox

Needs Docker and the pinned sandbox image, so the hook runs it and CI doesn't (`HOOK_ONLY` in `analysis/check_guardrails.py`).
It is `analysis/check_final_gates.py`'s made-up world with an `assert` line in each MBPP+ prompt, run through the free-checks
step twice: once with the stand-in sandbox and once with the world naming no sandbox, so the code check runs MBPP+'s made-up
answers in the real one. No results/, no Jev or Anthropic call, no real store.

- the real sandbox ran the code check on MBPP+'s eight answers (`routers.sandbox.run_python` is called eight times);
- each answer's verdict is the hand-worked one: `add` written as asked passes, `mul` written with `+` fails its assert (exit
  code 1, no timeout), and the stop-reason rejections are the same as in the stand-in run;
- the real sandbox and the stand-in give the same verdicts, answer by answer (passed, first failure, whether the code check
  applied, exit code and timeout), so what the CI check shows with the stand-in holds for the real one;
- each record carries the image's digest as Docker reports it, and it is the one the gate settings record.
"""

import dataclasses
import sys
from unittest import mock

from analysis import check_final_gates as g
from analysis import check_final_run as ci
from analysis import final_gates
from analysis import rehearsal_world as rehearsal
from analysis.final_run import Outcome
from routers import sandbox
from routers.gates import read_gate_settings

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


VERDICT = ("passed", "failed_first", "code_applied", "code_passed", "code_returncode", "code_timed_out")


def verdicts(records: list[dict]) -> dict[tuple[str, str], tuple]:
    return {(r["item_id"], r["config"]): tuple(r[k] for k in VERDICT) for r in records}


def main() -> int:
    print("-- the free checks' code check on made-up MBPP+ answers, in the real sandbox --")
    try:
        sandbox._ensure_image()
    except sandbox.SandboxError as error:
        print(f"FAIL  the real sandbox isn't available: {error}")
        return 1
    items, content = g.assert_content()
    with mock.patch.object(rehearsal, "ITEMS", items), ci.rehearsed(content) as stand_in, ci.rehearsed(content) as real:
        real.world = dataclasses.replace(real.world, backends=dataclasses.replace(real.world.backends, sandbox=None))
        steps = g.steps_to("calls_free_checks")
        outcome, _, _ = stand_in.go(steps)
        check(
            outcome == Outcome() and stand_in.sandbox_calls == 8, f"the stand-in run checks the code 8 times {outcome}"
        )
        calls = mock.Mock(wraps=sandbox.run_python)
        with mock.patch("routers.gates.run_python", calls):
            outcome, _, _ = real.go(steps)
        check(outcome == Outcome(), f"the run with no stand-in sandbox runs the free checks {outcome}")
        check(calls.call_count == 8 and real.sandbox_calls == 0,
              f"the real sandbox ran the code check on MBPP+'s eight answers ({calls.call_count} runs)")  # fmt: skip
        records = final_gates.read_free_test(real.world.results)
        failed = {(r["item_id"], r["config"]): r["failed_first"] for r in records if not r["passed"]}
        check(failed == {**g.STOP_FAILS, **g.CODE_FAILS},
              f"the verdicts are the hand-worked ones: the stop reasons, and `mul` written with + failing its assert {failed}")  # fmt: skip
        check(all(r["code_returncode"] == 1 and r["code_timed_out"] is False for r in records
                  if (r["item_id"], r["config"]) in g.CODE_FAILS),
              "a failing answer exits with an assertion error, not a time-out")  # fmt: skip
        same = verdicts(records) == verdicts(final_gates.read_free_test(stand_in.world.results))
        check(same, "the real sandbox and the stand-in give the same verdicts, answer by answer")
        digest = read_gate_settings().code_check.image_digest
        check({r["code_image_digest"] for r in records} == {digest} and digest is not None,
              "each record carries the image's digest as Docker reports it, the one the gate settings record")  # fmt: skip
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
