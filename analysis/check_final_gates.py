"""Check the free checks and the Jev gate on test, and the Jev gate's drift check (PLAN_v4.md Amendment A19, items 2, 7, 9, 10
and 13; A17 items 13, 14 and 26; A15 item 17; issue #77, #85).

Run from the project root:  uv run python -m analysis.check_final_gates

On the rehearsal's made-up world (`analysis/rehearsal_world.py`: eight test items, four configs each) with a stand-in sandbox, a
made-up Stage 7 noise pass and a fake Jev server (`analysis/rehearsal_gates.py`). No results/, no Docker, no Jev or Anthropic
call, and nothing near the real store or a real test item. The hook runs `analysis/check_final_gates_sandbox.py` (the code check
in the real sandbox) and `analysis/check_final_gates_real.py` (the drift check's inputs against Stage 7's stored records) too.
Every expected number is worked by hand below.

- **What the gates see.** A reading of `analysis/final_gates.py` finds no graded item, expected answer or response JSON in it and
  the handle used for two reads only; the steps run with a spy on each gate, and what reached them is the request, the answer's
  text and the stop reason of each of the 32 stored answers, as the sealed store holds them. A marker in an expected answer, one
  in every thinking block and the grade marker (a stored stop reason, so a legitimate input to the free checks only) reach no
  Jev body and no stored record; the answer's own marker reaches the bodies, so the search finds what is there to find.
- **The drift check.** Stage 7's made-up passes give D6 = 0.02 and a shift of 0; the resend (`quiet`) gives D8 = 0.03 and a shift of
  0, so it reports and goes on, with the decisions that flip at each frozen cutoff worked by hand ([1, 2, 1, 2, 1, 0], highest
  cutoff first). Its calls' bodies hash to what the frozen code builds and to what Stage 7 stored, and they are in
  `jev_gate_drift.jsonl` and no other file. Each refusal stops the run before any test call, one case each: a model ID that
  isn't the pin, D8 = 0.08 above 3 x D6 = 0.06, and a uniform move (D8 = 0.03, shift +0.03) that the D8 rule lets through and
  the shift rule stops. A body that differs from Stage 7's stops it before any call. A rerun resends nothing.
- **The free checks** on all 32 answers, stand-in sandbox and its digest, the hand-worked verdicts and `failed_first` (a stop
  that isn't `end_turn` twice; the code check on MBPP+ prompts with an `assert` line, which the stand-in sandbox runs).
- **The Jev gate** on the 30 (or 28) answers that passed, in the handle's order, one call each and none for a rejected answer,
  each body the frozen call's; the records' fields, tokens and overhead; the ledger's recorded spend by hand; a call that fails
  stops the run naming the answer, and the next go sends the unstored ones only (the fake's log), leaving stored lines as they
  were; a projection over the cap refuses before the first call, at the cap's boundary.
- **Test's gate-score table** from the stored records, and refused whole on each of A17 item 13's conditions, one at a time.
- **The pure cascades** `haiku/free` and `haiku/Jev` at their frozen points, quality, cost and escalation worked by hand from
  the answers, the Jev scores, the task mixes and the price sheet; `haiku/free` is A everywhere.
- **Nothing shows before the tables:** no count of free-check rejections, Jev refusals or cuts is in the log before the tables
  step; the tables step's record holds them.
"""

import ast
import dataclasses
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from unittest import mock

from analysis import check_final_run as ci
from analysis import final_gates, final_ledger, rehearsal_gates
from analysis import rehearsal_world as rehearsal
from analysis.collect_gates import GateCollectError, GateInput
from analysis.final_ledger import CALLS
from analysis.final_run import STEPS, Outcome, Step
from analysis.gate_table import GateScoreTable, GateTableError, PaidScore
from analysis.made_up_world import Content
from analysis.router_names import SONNET_GATE
from routers import jev, jev_gate
from routers.request import Request

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def raises(error: type[Exception] | tuple[type[Exception], ...], call: Callable[[], object]) -> bool:
    try:
        call()
    except error:
        return True
    return False


def close(a: float | None, b: float, tolerance: float = 1e-9) -> bool:
    return a is not None and math.isclose(a, b, abs_tol=tolerance)


# ---- the world's answers, worked by hand --------------------------------------------------------------------------

CONFIGS = ci.CONFIGS
IDS = ci.ITEMS  # the eight test items in the world's order: banking 0, 1; MMLU-Pro 0, 1; IFEval 0, 1; MBPP+ 0, 1
STOP_FAILS = {(IDS[5], "haiku"): "stop", (IDS[3], "sonnet"): "stop"}  # a stop reason that isn't end_turn
CODE_FAILS = {(IDS[7], "haiku"): "code", (IDS[7], "sonnet"): "code"}  # with the asserts: `mul` written as `+`
SETTINGS = jev_gate.read_settings()
PRICE = jev.read_client_settings().price_per_million_input_tokens  # $0.042 per million input tokens
OVERHEAD = rehearsal_gates.TOKENS * PRICE / 1e6  # one made-up call: 1,000 tokens, $0.000042


def made_up_answers(items: list[dict] | None = None) -> dict[tuple[str, str], tuple[Request, str, str]]:
    """Every stored answer by (item, config): its request, its text and its stop reason, from the rehearsal world's own tables."""
    found = {}
    for position, made in enumerate(items or rehearsal.ITEMS):
        for index, config in enumerate(CONFIGS):
            answer = rehearsal.answer_text(position, bool(rehearsal.CORRECT[position][index]))
            stop = rehearsal.STOPS.get((position, config), "end_turn")
            found[(made["item_id"], config)] = (Request(made["prompt"], made.get("output_schema")), answer, stop)
    return found


def hand_body(request: Request, answer: str) -> bytes:
    """A call body written out as the plan words it (A17 item 5, A15 item 8): compact JSON, the model, then the state with the
    request before the answer, then the three questions in the settings' order."""
    questions = {s.id: {"type": "noul", "instructions": s.text} for s in SETTINGS.statements}
    text = json.dumps({"model": "jev-1.14.0",
                       "state": {"request": {"prompt": request.prompt, "output_schema": request.output_schema},
                                 "answer": answer},
                       "questions": questions}, separators=(",", ":"), ensure_ascii=False)  # fmt: skip
    return text.encode("utf-8")


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def without_liveness(steps: tuple[Step, ...]) -> tuple[Step, ...]:
    """The steps with the liveness step left out, whose projection would refuse first: a step's own cap check is under test."""
    return tuple(s for s in steps if s.name != "liveness")


# G's and G+'s steps (#84) are held by `analysis/check_final_jev.py`; here they would add their calls and their projection to the
# exact-dollar arithmetic of the gates' own caps, so the runs that stop at a gate step leave them out and `main` takes their
# projector out of the ledger's.
ROUTER_STEPS = ("drift_jev_router", "calls_jev_router")


def steps_to(name: str) -> tuple[Step, ...]:
    names = [s.name for s in STEPS]
    return tuple(s for s in STEPS[: names.index(name) + 1] if s.name not in ROUTER_STEPS)


def leak_search(text: str) -> list[str]:
    """Which of the world's markers `text` holds."""
    return [marker for marker in ci.MARKERS if marker in text]


def contents(path: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(path.rglob("*.jsonl")))


# ---- what the gates see -------------------------------------------------------------------------------------------

FORBIDDEN = {
    "graded_items",
    "GradedItem",
    "GRADED_QUERY",
    "GRADED_COLUMNS",
    "expected_answer",
    "response_json",
    "sealed_store",
}


def forbidden_names(source: str) -> set[str]:
    """The names a module's code uses that reach an outcome: every Name or attribute in the tree, not a word in a docstring."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit(".", 1)[-1])
    return names & FORBIDDEN


def handle_reads(source: str) -> set[str]:
    return {n.attr for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "handle"}  # fmt: skip


class Spy:
    """Wraps a gate's entry point, keeps what each call was given, and calls through."""

    def __init__(self, real: Callable[..., Any]):
        self.real = real
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.real(*args, **kwargs)


def what_the_gates_see() -> None:
    print("-- what the gates see: the request, the answer's text and its stop reason, and nothing else --")
    source = Path(final_gates.__file__).read_text(encoding="utf-8")
    check(
        forbidden_names(source) == set(),
        "the module that runs the gates names no graded item, expected answer or response JSON",
    )
    check(
        handle_reads(source) == {"requests", "answers"},
        f"and uses the handle for its two reads alone {handle_reads(source)}",
    )
    check(forbidden_names("handle.graded_items()[0].item.expected_answer") == {"graded_items", "expected_answer"}
          and handle_reads("handle.graded_items()") == {"graded_items"},
          "known positive: the same searches find a graded item and an expected answer where they are named")  # fmt: skip
    check(
        [f.name for f in dataclasses.fields(GateInput)] == ["item_id", "config", "request", "answer", "stop_reason"],
        "a gate's input type holds the item, the config, the request, the answer's text and the stop reason",
    )

    with ci.rehearsed() as r:
        free, call = Spy(final_gates.free_checks), Spy(jev_gate.gate_call)
        with mock.patch.object(final_gates, "free_checks", free), mock.patch.object(jev_gate, "gate_call", call):
            outcome, out, _ = r.go(steps_to("calls_jev_gate"))
        check(outcome == Outcome(), f"the rehearsal runs up to the Jev gate {outcome}")
        known = made_up_answers()
        gave = [(a[0].prompt, a[1], a[2]) for a, _ in free.calls]
        want = [(request.prompt, answer, stop) for (request, answer, stop) in known.values()]
        check(len(free.calls) == 32 and sorted(gave, key=repr) == sorted(want, key=repr)
              and all(isinstance(a[0], Request) and isinstance(a[1], str) for a, _ in free.calls),
              "the free checks were given 32 (request, answer text, stop reason) triples, the stored answers' own")  # fmt: skip
        test_calls = [c for c in call.calls if c[0][0].prompt in {p.prompt for p, _, _ in known.values()}]
        check(
            all(
                len(a) == 3 and isinstance(a[0], Request) and isinstance(a[1], str) and set(k) <= {"budget_bytes"}
                for a, k in test_calls
            ),
            "and the Jev gate's calls on test were built from a request and an answer's text alone (and the byte budget)",
        )
        everything = "\n".join(repr(a) + repr(k) for a, k in free.calls + call.calls)
        bodies = "\n".join(json.dumps(b) for b in r.gates.fake.bodies)
        stored = contents(r.world.results / "gates")
        for name, text in (
            ("the gates' arguments", everything),
            ("the bodies Jev received", bodies),
            ("the stored records", stored),
        ):
            check(rehearsal.EXPECTED_MARKER not in text and rehearsal.THINKING_MARKER not in text,
                  f"an expected answer's marker and a thinking block's reach none of {name}")  # fmt: skip
        check(rehearsal.ANSWER_MARKER in everything and rehearsal.ANSWER_MARKER in bodies,
              "known positive: an answer's own marker does reach the gates' arguments and the bodies Jev received")  # fmt: skip
        check(rehearsal.GRADE_MARKER in everything and rehearsal.GRADE_MARKER not in bodies + stored,
              "the stop reason that carries the grade marker reaches the free checks, which never store it, and no Jev body")  # fmt: skip
        check(leak_search(bodies + stored) == [rehearsal.ANSWER_MARKER] or rehearsal.ANSWER_MARKER not in stored,
              "the stored records hold at most the answer's own marker (the free checks keep a stderr tail, never the answer)")  # fmt: skip


# ---- the drift check ----------------------------------------------------------------------------------------------


def drift_detail(r: ci.Rehearsal) -> dict:
    return next(s for s in r.record["gos"][-1]["steps"] if s["name"] == "drift_jev_gate")["detail"]


def drift_quiet() -> None:
    print("\n-- the drift check under its limits reports and goes on --")
    with ci.rehearsed() as r:
        train_file = r.world.results / "gates" / "jev_gate_train.jsonl"
        before = train_file.read_bytes()
        outcome, out, _ = r.go(steps_to("drift_jev_gate"))
        check(outcome == Outcome(), f"the run goes on past the drift check {outcome}")
        detail = drift_detail(r)
        m = detail["measures"]["haiku"]
        check(close(m["d6"], 0.02) and close(m["d8"], 0.03) and close(m["shift"], 0.0),
              f"D6 0.02, D8 0.03 and shift 0 are the hand-worked ones {m}")  # fmt: skip
        flips = {f["threshold"]: f["count"] for f in detail["flips"]}
        hand = {0.9540063295697989: 1, 0.8149190161607497: 2, 0.8097612537302109: 1, 0.7307604185714591: 2,
                0.6333807685183769: 1, 0.5414067403925006: 0}  # fmt: skip
        check(flips == hand and [f["threshold"] for f in detail["flips"]] == sorted(hand, reverse=True),
              f"the decisions that flip at each frozen cutoff, highest first, are worked by hand {list(flips.values())}")  # fmt: skip
        by_hand = {0.9540063295697989: ["banking77:made-train-0 haiku"],
                   0.8149190161607497: ["banking77:made-train-1 haiku", "banking77:made-train-2 haiku"],
                   0.7307604185714591: ["banking77:made-train-3 haiku", "banking77:made-train-4 haiku"]}  # fmt: skip
        check(all([f["items"] for f in detail["flips"] if f["threshold"] == t] == [items] for t, items in by_hand.items()),
              "and name the answers: item 0 at 0.954, items 1 and 2 at 0.8149, items 3 and 4 at 0.7308")  # fmt: skip
        labelled = {p for f in detail["flips"] for p in f["points"]}
        check({"haiku/Jev realistic 99%", "G+/Jev uniform 99%", "E/Jev realistic 97%"} <= labelled and len(labelled) == 25,
              "each cutoff names the operating points that sit on it: haiku/Jev's six and the router-started cascades' shared one")  # fmt: skip
        check(detail["model_ids"] == ["jev-1.14.0"] and detail["pin"] == "jev-1.14.0" and detail["answers"] == 8,
              "the resends came back as the pin, 8 of them")  # fmt: skip
        check(
            "D6 0.0200, D8 0.0300" in out and "drift_jev_gate: done" in out,
            "the log says D6 and D8, which are train's numbers",
        )

        records = final_gates.read_jev("drift", r.world.results)
        sample = [(f"banking77:made-train-{k}", "haiku") for k in rehearsal_gates.SAMPLE]
        check([(x["item_id"], x["config"]) for x in records] == sample
              and {(x["split"], x["gate"], x["pass"]) for x in records} == {("drift", "jev", 1)},
              "the resends are in jev_gate_drift.jsonl: one record each, labelled drift, in the sample's order")  # fmt: skip
        hashes = []
        for k in rehearsal_gates.SAMPLE:
            item = rehearsal_gates.train_input(k)
            hashes.append(sha(hand_body(item.request, item.answer)))
        stage_7 = [
            x["body_sha256"]
            for x in final_gates.collect_jev_gate.read_records("train", r.world.results)
            if x["pass"] == 2
        ]
        received = [sha(x.body) for x in r.gates.fake.received]
        check(received == hashes == [x["body_sha256"] for x in records] == stage_7,
              "each resend's body hashes to the hand-written one, the frozen code's, and the one Stage 7 stored")  # fmt: skip
        check(train_file.read_bytes() == before and sorted(p.name for p in (r.world.results / "gates").iterdir())
              == ["jev_gate_drift.jsonl", "jev_gate_train.jsonl"],
              "Stage 7's file is as it was, and the drift's own is the only new file: no test file yet")  # fmt: skip
        spent = final_ledger.recorded_spend(r.world.results, r.world.output)
        check(
            close(spent.calls, 8 * OVERHEAD, 1e-12),
            f"the ledger counts the resends: 8 calls of 1,000 tokens, ${spent.calls:.6f}",
        )
        outcome, _, _ = r.go(steps_to("drift_jev_gate"))
        check(outcome == Outcome() and len(r.gates.fake.received) == 8,
              "a second go resends nothing: the fake's log holds the first go's 8 calls only")  # fmt: skip
        check(final_gates.read_jev("drift", r.world.results) == records and drift_detail(r)["measures"] == detail["measures"],
              "and reports the same numbers from what is stored")  # fmt: skip


def refused_at(
    scenario: str, what: str, expect: list[str], not_expect: list[str] | None = None, **changes: Any
) -> None:
    with ci.rehearsed() as r:
        r.gates.fake.resend(scenario)
        for name, value in changes.items():
            setattr(r.gates.fake, name, value)
        outcome, out, _ = r.go(steps_to("drift_jev_gate"))  # G's and G+'s own D8 lines are in the log of a full run
        calls = [json.loads(x.body)["state"]["request"]["prompt"] for x in r.gates.fake.received]
        shown = r.record["gos"][0]["stopped"]
        check(outcome == Outcome("drift_jev_gate", "DriftRefusal", None),
              f"{what}: the run stops at the drift check with a DriftRefusal {outcome}")  # fmt: skip
        check(all(text in out for text in expect) and not any(text in out for text in (not_expect or [])),
              f"{what}: the log says why, with train's numbers only {[t for t in expect if t not in out]}")  # fmt: skip
        check(len(calls) == 8 and all(p.startswith("Which kind of card problem") for p in calls),
              f"{what}: Jev was sent the 8 resends and no test request")  # fmt: skip
        check(not (r.world.results / "gates" / "free_checks_test.jsonl").exists()
              and not (r.world.results / "gates" / "jev_gate_test.jsonl").exists()
              and not (r.world.results / "test_matrix.sqlite").exists() and shown["step"] == "drift_jev_gate",
              f"{what}: no test record and no grade exists, and the record says where it stopped")  # fmt: skip


def drift_refusals() -> None:
    print("\n-- each refusal stops the run before any test call --")
    refused_at("quiet", "a model ID that isn't the pin", ["jev-1.15.0", "not the pin jev-1.14.0"], model="jev-1.15.0")
    refused_at("loud", "D8 above 3 x D6 (0.08 against 0.06)", ["haiku: D8 0.08", "more than 3 x D6 0.02"],
               ["mean signed difference"])  # fmt: skip
    refused_at("uniform", "a uniform move (D8 0.03, shift +0.03)", ["the shift (mean signed difference) +0.03", "larger than D6 0.02"],
               ["D8"])  # fmt: skip
    print("\n-- a body that differs from Stage 7's, a missing sample and a cap --")
    with ci.rehearsed() as r:
        path = r.world.results / "gates" / "jev_gate_train.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        for record in records:
            if record["item_id"].endswith("-3") and record["pass"] == 1:
                record["body_sha256"] = "0" * 64
        path.write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")
        outcome, out, _ = r.go(steps_to("drift_jev_gate"))
        check(outcome == Outcome("drift_jev_gate", "DriftRefusal", None) and r.gates.fake.received == []
              and "builds a different body than Stage 7 stored for 1 of the 8" in out,
              f"a body that isn't the one Stage 7 stored stops the run before any call {outcome}")  # fmt: skip
    with ci.rehearsed() as r:
        path = r.world.results / "gates" / "jev_gate_train.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_text("".join(json.dumps(x) + "\n" for x in records if x["pass"] == 1), encoding="utf-8")
        outcome, _, _ = r.go(steps_to("drift_jev_gate"))
        check(outcome == Outcome("drift_jev_gate", "FinalGatesError", None) and r.gates.fake.received == [],
              "a Stage 7 file with no noise pass has nothing to resend: refused, nothing called")  # fmt: skip
    with ci.rehearsed() as r:
        with mock.patch.dict(final_ledger.CAPS, {CALLS: 0.0001}):
            outcome, out, _ = r.go()
        check(outcome == Outcome("liveness", "LedgerRefusal", None) and r.gates.fake.received == [] and r.anthropic.calls == [],
              f"a projection over the cap refuses at the liveness step: no Jev call and no liveness call {outcome}")  # fmt: skip
        check("calls pool" in out and "over Stage 8's cap" in out and "nothing was sent" in out,
              "the log names the pool and the cap")  # fmt: skip
    with ci.rehearsed() as r:
        # The drift step's own cap check: the cap exactly at the resends' projection passes, a little below it refuses.
        first = final_gates.pending_calls(
            [rehearsal_gates.train_input(k) for k in rehearsal_gates.SAMPLE], set(), SETTINGS
        )
        need = final_gates.projected_dollars(first)
        with mock.patch.dict(final_ledger.CAPS, {CALLS: need + 1e-12}):
            outcome, _, _ = r.go(without_liveness(steps_to("drift_jev_gate")))
        check(
            outcome == Outcome(),
            f"the drift step passes with the cap exactly at the resends' projection ${need:.6f} {outcome}",
        )
    with ci.rehearsed() as r:
        need = final_gates.projected_dollars(
            final_gates.pending_calls([rehearsal_gates.train_input(k) for k in rehearsal_gates.SAMPLE], set(), SETTINGS)
        )
        with mock.patch.dict(final_ledger.CAPS, {CALLS: need - 1e-6}):
            outcome, out, _ = r.go(without_liveness(steps_to("drift_jev_gate")))
        check(outcome == Outcome("drift_jev_gate", "LedgerRefusal", None) and r.gates.fake.received == [],
              f"and refuses a hair below it, before the first call {outcome}")  # fmt: skip


# ---- the free checks ----------------------------------------------------------------------------------------------


def assert_content() -> tuple[list[dict], Content]:
    """The rehearsal world with an `assert` line in each MBPP+ prompt, so that the code check applies and runs."""
    asserts = {6: "assert add(1, 2) == 3", 7: "assert mul(2, 3) == 6"}
    items = [{**made, "prompt": made["prompt"] + f"\n{asserts[p]}"} if p in asserts else made
             for p, made in enumerate(rehearsal.ITEMS)]  # fmt: skip
    with mock.patch.object(rehearsal, "ITEMS", items):
        sealed = rehearsal.sealed_rows()
    benchmark = [
        {**made, "split": "train", "item_id": made["item_id"].replace("test", "train")} for made in items[:1]
    ] + items
    return items, Content(benchmark, sealed)


def free_checks() -> None:
    print("\n-- the free checks on every test answer, in the world's sandbox --")
    with ci.rehearsed() as r:
        outcome, out, _ = r.go(steps_to("calls_free_checks"))
        records = final_gates.read_free_test(r.world.results)
        expected_keys = sorted(made_up_answers())
        check(outcome == Outcome() and [(x["item_id"], x["config"]) for x in records] == expected_keys,
              "the step stores one record for each of the 32 answers, in the handle's order (item, then config)")  # fmt: skip
        failed = {(x["item_id"], x["config"]): x["failed_first"] for x in records if not x["passed"]}
        check(failed == STOP_FAILS, f"the hand-worked rejections: a stop reason that isn't end_turn, twice {failed}")
        check(all(x["code_image_digest"] == final_gates.STAND_IN_DIGEST and x["split"] == "test" and x["gate"] == "free"
                  and x["run"] == 0 and x["pass"] == 1 for x in records),
              "each record is test's, run 0, pass 1, with the stand-in sandbox's digest")  # fmt: skip
        check(r.sandbox_calls == 0, "no code check ran: the made-up MBPP+ prompts have no assert line")
        check("32 answers checked, 0 already stored" in out and "rejected" not in out and "failed" not in out.split("tables")[0],
              "the log says how many answers were checked, and not how many a check rejected")  # fmt: skip
        outcome, out, _ = r.go(steps_to("calls_free_checks"))
        check(outcome == Outcome() and "0 answers checked, 32 already stored" in out and r.sandbox_calls == 0
              and len(final_gates.read_free_test(r.world.results)) == 32,
              "a second go checks nothing and stores nothing more")  # fmt: skip

    items, content = assert_content()
    with mock.patch.object(rehearsal, "ITEMS", items), ci.rehearsed(content) as r:
        outcome, _, _ = r.go(steps_to("calls_free_checks"))
        by_key = {(x["item_id"], x["config"]): x for x in final_gates.read_free_test(r.world.results)}
        failed = {k: x["failed_first"] for k, x in by_key.items() if not x["passed"]}
        check(outcome == Outcome() and failed == {**STOP_FAILS, **CODE_FAILS},
              f"with an assert line the code check runs the answer's code: add passes, mul written as + fails {failed}")  # fmt: skip
        check(r.sandbox_calls == 8 and all(by_key[(i, c)]["code_applied"] for i in IDS[6:] for c in CONFIGS)
              and all(by_key[(i, c)]["code_returncode"] == 0 for i in IDS[6:] for c in CONFIGS if (i, c) not in CODE_FAILS)
              and all(by_key[k]["code_returncode"] == 1 and by_key[k]["code_timed_out"] is False for k in CODE_FAILS),
              f"the stand-in sandbox ran MBPP+'s 8 answers ({r.sandbox_calls}): exit 0 for 6, an assertion error for 2")  # fmt: skip
        check(all(not by_key[k]["code_applied"] for k in by_key if k[0] not in IDS[6:]),
              "and the code check applied to no other item")  # fmt: skip


# ---- the Jev gate's calls on test ---------------------------------------------------------------------------------


def passing(fails: Iterable[tuple[str, str]] = tuple(STOP_FAILS)) -> list[tuple[str, str]]:
    bad = set(fails)
    return [k for k in sorted(made_up_answers()) if k not in bad]


def test_calls() -> None:
    print("\n-- the Jev gate on every answer that passed the free checks --")
    answers = made_up_answers()
    with ci.rehearsed() as r:
        outcome, out, _ = r.go(steps_to("calls_jev_gate"))
        fake = r.gates.fake
        records = final_gates.read_jev("test", r.world.results)
        keys = passing()
        check(outcome == Outcome() and [(x["item_id"], x["config"]) for x in records] == keys and len(keys) == 30,
              "30 calls, one for each answer that passed, in the handle's order, none for the 2 rejected")  # fmt: skip
        test_bodies = fake.received[8:]  # the drift check's 8 resends came first
        hand = [hand_body(answers[k][0], answers[k][1]) for k in keys]
        check([x.body for x in test_bodies] == hand and [x["body_sha256"] for x in records] == [sha(b) for b in hand],
              "each body is the hand-written JSON of the frozen call, and each stored hash is its hash")  # fmt: skip
        rebuilt = [jev_gate.gate_call(answers[k][0], answers[k][1], SETTINGS).sha256 for k in keys]
        check(
            rebuilt == [x["body_sha256"] for x in records],
            "and the frozen code builds the same hashes from the answers",
        )
        check(all(x["split"] == "test" and x["gate"] == "jev" and x["run"] == 0 and x["pass"] == 1 and x["model"] == "jev-1.14.0"
                  and x["usage"] == {"input_tokens": 1000, "output_tokens": 38} and x["refused"] is False
                  and x["answer_cut_bytes"] == 0 and x["prompt_cut_bytes"] == 0 for x in records)
              and all(tuple(x) == final_gates.STORED_FIELDS for x in records),
              "each record is keyed as test's, holds the whole response, the usage and no cut, in A15 item 9's fields")  # fmt: skip
        check(
            fake.unexpected == [] and len(fake.received) == 8 + 30,
            "the fake saw 8 drift calls and 30 test calls, none unexpected",
        )
        spent = final_ledger.recorded_spend(r.world.results, r.world.output)
        check(
            close(spent.calls, 38 * OVERHEAD, 1e-12),
            f"the ledger's recorded spend is 38 calls of 1,000 tokens, ${spent.calls:.6f}",
        )
        step = next(s for s in r.record["gos"][0]["steps"] if s["name"] == "calls_jev_gate")
        check(step["requests"] == {"sent": 30, "stored": 0}, "the record holds the request counts")
        before_tables = "\n".join(line for line in out.split("tables: starts")[0].splitlines() if "Jev gate" in line)
        check("Jev gate: 30 calls sent, 0 already stored" in before_tables
              and not any(word in before_tables for word in ("refus", "rejected", "cut", "score", "passed")),
              "the log says how many calls were sent and not what came back: no score, refusal or cut")  # fmt: skip
        oldest = (r.world.results / "gates" / "jev_gate_test.jsonl").read_bytes()
        outcome, out, _ = r.go(steps_to("calls_jev_gate"))
        check(outcome == Outcome() and len(fake.received) == 38 and (r.world.results / "gates" / "jev_gate_test.jsonl").read_bytes() == oldest
              and "0 calls sent, 30 already stored" in out,
              "a second go sends nothing, and the file is as it was")  # fmt: skip


def resume() -> None:
    print("\n-- a failed call stops the run naming the answer, and a resume sends the unstored calls only --")
    answers = made_up_answers()
    keys = passing()
    with ci.rehearsed() as r:
        r.gates.fake.fail_at = {8 + 5}  # the fifth test call (after the drift check's 8)
        outcome, out, _ = r.go()
        record = r.record["gos"][0]
        named = f"{keys[4][0]} {keys[4][1]}"
        check(outcome == Outcome("calls_jev_gate", "Rejected", named),
              f"a 400 on the fifth test call stops the run on calls_jev_gate and names that answer and the exception {outcome}")  # fmt: skip
        check("made up" not in out and rehearsal.ANSWER_MARKER not in out and "400" not in record["stopped"].get("message", ""),
              "nothing of the reply or the answer is shown")  # fmt: skip
        stored = final_gates.read_jev("test", r.world.results)
        check(
            [(x["item_id"], x["config"]) for x in stored] == keys[:4],
            "the four calls before it are stored, the failed one is not",
        )
        first_lines = (r.world.results / "gates" / "jev_gate_test.jsonl").read_bytes()
        check(len(r.gates.fake.received) == 8 + 5 and not (r.world.results / "test_matrix.sqlite").exists(),
              "5 test calls were made, and nothing was graded")  # fmt: skip
        r.gates.fake.fail_at = set()
        outcome, out, _ = r.go(
            steps_to("tables")
        )  # through the tables: the live replay (#91) sends Jev calls of its own
        check(outcome == Outcome(), f"the next go runs on through the tables {outcome}")
        new = r.gates.fake.received[8 + 5 :]
        check([x.body for x in new] == [hand_body(*answers[k][:2]) for k in keys[4:]] and len(new) == 26,
              "it sent the 26 unstored calls only, in order, the failed one again: nothing stored was sent twice")  # fmt: skip
        check((r.world.results / "gates" / "jev_gate_test.jsonl").read_bytes().startswith(first_lines)
              and [(x["item_id"], x["config"]) for x in final_gates.read_jev("test", r.world.results)] == keys,
              "the stored lines are as they were, and the file now holds all 30")  # fmt: skip
        check(len(r.gates.fake.received) == 8 + 30 + 1 and len(final_gates.read_jev("drift", r.world.results)) == 8,
              "the drift check's resends were not sent again in the second go (8 + 30 + the failed call sent twice)")  # fmt: skip
        check(len(record["steps"]) < len(r.record["gos"][1]["steps"]) and len(r.record["gos"]) == 2,
              "the record keeps both gos")  # fmt: skip

    print("\n-- a projection over the cap refuses before the first test call --")
    with ci.rehearsed() as r:
        outcome, _, _ = r.go(steps_to("calls_free_checks"))
        pending = final_gates.pending_calls(
            [GateInput(k[0], k[1], *answers[k][:1], answers[k][1], answers[k][2]) for k in keys], set(), SETTINGS
        )
        need = final_gates.projected_dollars(pending)
        check(close(need, sum(len(hand_body(*answers[k][:2])) + 512 for k in keys) * PRICE / 1e6, 1e-12),
              f"the projection is each body's bytes plus 512 tokens at Jev's price, by hand: ${need:.6f}")  # fmt: skip
        calls_before = len(r.gates.fake.received)
        with mock.patch.dict(final_ledger.CAPS, {CALLS: need + 8 * OVERHEAD - 1e-6}):
            outcome, out, _ = r.go(without_liveness(steps_to("calls_jev_gate")))
        check(outcome == Outcome("calls_jev_gate", "LedgerRefusal", None) and len(r.gates.fake.received) == calls_before,
              f"a cap a hair below the recorded spend plus the projection refuses, and sends nothing {outcome}")  # fmt: skip
        check(not (r.world.results / "gates" / "jev_gate_test.jsonl").exists()
              or not final_gates.read_jev("test", r.world.results),
              "no test call is stored")  # fmt: skip
        spent = final_ledger.recorded_spend(r.world.results, r.world.output).calls
        with mock.patch.dict(final_ledger.CAPS, {CALLS: need + spent + 1e-12}):
            outcome, _, _ = r.go(steps_to("calls_jev_gate"))
        check(outcome == Outcome() and len(final_gates.read_jev("test", r.world.results)) == 30,
              "at the cap exactly, the calls go")  # fmt: skip

    print("\n-- a world that names backends and no Jev client is refused --")
    with ci.rehearsed() as r:
        r.world = dataclasses.replace(r.world, backends=dataclasses.replace(r.world.backends, jev_gate=None))
        outcome, _, _ = r.go(steps_to("drift_jev_gate"))
        check(
            outcome == Outcome("drift_jev_gate", "StepRefusal", None),
            f"no real client is built for a rehearsal {outcome}",
        )
        with mock.patch.object(jev, "JevClient", side_effect=AssertionError("the real client was built")) as real:
            r.world = dataclasses.replace(r.world, backends=None)
            try:
                final_gates.jev_client_of(mock.Mock(world=r.world))
            except AssertionError:
                pass
            check(
                real.call_count == 1,
                "the go's world, which names no backends, builds the real client from the settings",
            )


# ---- test's gate-score table --------------------------------------------------------------------------------------


def table_checks() -> None:
    print("\n-- test's gate-score table, from the stored records --")
    answers = sorted(made_up_answers())
    client = jev.read_client_settings()
    with ci.rehearsed() as r:
        r.go(steps_to("calls_jev_gate"))
        free = final_gates.read_free_test(r.world.results)
        calls = final_gates.read_jev("test", r.world.results)

        def build(
            free_records: list[dict] = free, jev_records: list[dict] = calls, keys: list = answers, **kw: Any
        ) -> GateScoreTable:
            return final_gates.build_test_table(free_records, jev_records, keys, SETTINGS, client, **kw)

        table = build()
        score = {(IDS[p], "haiku"): s for p, s in enumerate(rehearsal_gates.TEST_SCORES)}
        found = {k: e for k in answers if (e := table.row(*k).jev) is not None}
        check(len(table.rows) == 32 and sum(row.free_passed for row in table.rows) == 30 and len(found) == 30,
              "the table has a row for each of the 32 answers, 30 of them with a Jev entry")  # fmt: skip
        check(all(close(found[k].score, rehearsal_gates.TEST_SCORES[IDS.index(k[0])]) and found[k].refused is False
                  and close(found[k].overhead, OVERHEAD, 1e-12) for k in found),
              "each Jev entry is the item's scripted gate score, not refused, at 1,000 tokens' overhead of $0.000042")  # fmt: skip
        check(all(table.row(*k).jev is None and table.row(*k).failed_first == "stop" for k in STOP_FAILS),
              "a rejected answer has no Jev entry and says which check failed first")  # fmt: skip
        check(all(close(found[k].score, score[(IDS[i], "haiku")]) for i, k in enumerate(((IDS[i], "haiku") for i in range(8)))
                  if k in found), "haiku's scores are the hand-set ones")  # fmt: skip

        print("\n-- refused whole on each of A17 item 13's conditions --")
        first = (calls[0]["item_id"], calls[0]["config"])
        no_row = [x for x in free if (x["item_id"], x["config"]) != (IDS[5], "haiku")]
        check(raises(GateTableError, lambda: build(no_row)), "an answer missing from the free-check records")
        check(raises(GateTableError, lambda: build(keys=answers[1:])), "a record for an answer the split doesn't have")
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder)
            shutil.copytree(r.world.results / "gates", copy / "gates")
            path = copy / "gates" / "free_checks_test.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            path.write_text("".join(lines) + lines[0], encoding="utf-8")
            check(
                raises(GateCollectError, lambda: final_gates.read_free_test(copy)),
                "an answer repeated in the free-check file",
            )
            path.write_text("".join(lines), encoding="utf-8")
            jev_path = copy / "gates" / "jev_gate_test.jsonl"
            jev_lines = jev_path.read_text(encoding="utf-8").splitlines(keepends=True)
            jev_path.write_text("".join(jev_lines) + jev_lines[0], encoding="utf-8")
            check(
                raises(GateCollectError, lambda: final_gates.read_jev("test", copy)),
                "an answer repeated in the Jev gate's file",
            )
            moved = json.loads(jev_lines[0])
            moved["split"] = "dev"
            jev_path.write_text(json.dumps(moved) + "\n", encoding="utf-8")
            check(
                raises(GateCollectError, lambda: final_gates.read_jev("test", copy)),
                "a Jev record labelled as another split's",
            )
            jev_path.write_bytes(b"".join(x.encode() for x in jev_lines[:3]) + b'{"item_id": "torn')
            check(
                len(final_gates.read_jev("test", copy)) == 3,
                "known positive: a torn last line is a write cut short and is dropped",
            )
        stray = {**calls[0], "item_id": IDS[5], "config": "haiku"}
        check(raises(GateCollectError, lambda: build(jev_records=[*calls, stray])),
              "a paid score on an answer that failed a free check (a Jev call stored for it)")  # fmt: skip
        check(raises(GateCollectError, lambda: build(jev_records=calls[1:])), "a passing answer with no Jev call")
        check(raises(GateTableError, lambda: build(extra_paid={SONNET_GATE: {(IDS[5], "haiku"): PaidScore(0.5, 0.0, False)}})),
              "a second gate's score on an answer that failed a free check")  # fmt: skip

        def with_second(score: Any, overhead: float) -> Callable[[], object]:
            return lambda: build(extra_paid={SONNET_GATE: {first: PaidScore(score, overhead, False)}})

        for bad in (1.2, -0.1, float("nan"), True):
            check(
                raises(GateTableError, with_second(bad, 0.0)), f"a gate score that isn't a number from 0 to 1 ({bad!r})"
            )
        for bad in (-1e-9, float("nan"), float("inf")):
            check(
                raises(GateTableError, with_second(0.5, bad)), f"an overhead that is negative or not finite ({bad!r})"
            )
        negative = [{**calls[0], "usage": {"input_tokens": -1, "output_tokens": 38}}, *calls[1:]]
        check(
            raises(GateCollectError, lambda: build(jev_records=negative)), "a stored call with a negative token count"
        )
        flagged = [{**calls[0], "refused": True}, *calls[1:]]
        check(raises(GateCollectError, lambda: build(jev_records=flagged)),
              "a stored refusal flag the response doesn't bear out")  # fmt: skip
        check(raises(GateTableError, lambda: build(extra_paid={SONNET_GATE: {first: PaidScore(None, 0.0, False)}})),
              "a missing score on a call not marked refused")  # fmt: skip
        refused_call = {**calls[0], "response": {"model": "jev-1.14.0", "answers": {}, "usage": {"input_tokens": 1000, "output_tokens": 1}},
                        "refused": True}  # fmt: skip
        entry = build(jev_records=[refused_call, *calls[1:]]).row(*first).jev
        check(entry is not None and entry.score is None and entry.refused and close(entry.overhead, OVERHEAD, 1e-12),
              "known positive: a refused call is a rejection with no score, charged its tokens, and the table is built")  # fmt: skip


# ---- the pure cascades at their frozen points ---------------------------------------------------------------------

TASK_ITEMS = {
    "banking77": (0, 1),
    "mmlu_pro": (2, 3),
    "ifeval": (4, 5),
    "mbppplus": (6, 7),
}  # item positions of each task
# The file's dials: haiku/Jev's cutoff at each mix and target (the plan's committed A17 item 29 numbers, 4 decimals as written).
HAND_CUTOFFS = {
    "realistic": {"100%": 0.9540, "99%": 0.8149, "97%": 0.6334},
    "uniform": {"100%": 0.9540, "99%": 0.7308, "97%": 0.5414},
}


def hand_cascade(cutoff: float, mix: dict[str, float]) -> tuple[float, float, float]:
    """haiku/Jev at `cutoff` by hand: quality, dollars per 1,000 requests and the share escalated, under the mix. An answer is
    kept if it passed the free checks and the gate scored it at least `cutoff`; otherwise `opus` answers too. A call costs the
    start's price, the gate's 1,000 tokens if the free checks passed, and `opus`'s price if the answer is not kept."""
    quality = cost = escalated = 0.0
    for task, positions in TASK_ITEMS.items():
        column = 1 if task == "mbppplus" else 0
        for position in positions:
            passed = (IDS[position], "haiku") not in STOP_FAILS
            kept = passed and rehearsal_gates.TEST_SCORES[position] >= cutoff
            micro = (
                ci.MICRO["haiku"][column]
                + (OVERHEAD * 1e6 if passed else 0)
                + (0 if kept else ci.MICRO["opus"][column])
            )
            right = ci.SUCCESS[position][0] if kept else ci.SUCCESS[position][3]
            weight = mix[task] / len(positions)
            quality += weight * right
            cost += weight * micro / 1000
            escalated += 0 if kept else weight
    return quality, cost, escalated


def cascades() -> None:
    print("\n-- haiku/free and haiku/Jev at their frozen points, worked by hand --")
    with ci.rehearsed() as r:
        outcome, _, _ = r.go()
        check(outcome == Outcome(), f"the whole rehearsal runs, the tables included {outcome}")
        parts = r.tables["parts"]
        free, jev_rows = parts[final_gates.FREE_SECTION]["mixes"], parts[final_gates.JEV_SECTION]["mixes"]
        for mix, weights in ci.MIXES.items():
            a_quality, a_cost = ci.HAND_QUALITY[mix]["opus"], ci.HAND_COST[mix]["opus"]
            for target in ("100%", "99%", "97%"):
                row = free[mix][target]
                check(row["fell_back"] is True and row["threshold"] is None and row["cutoff"] is None
                      and close(row["quality"]["value"], a_quality) and close(row["cost_per_1k"]["value"], a_cost)
                      and row["escalation_rate"] == 0 and row["savings_vs_A"]["share"] == 0,
                      f"haiku/free {mix} {target}: the file says it fell back, so it is A: {a_quality} at ${a_cost}")  # fmt: skip
                row = jev_rows[mix][target]
                quality, cost, escalated = hand_cascade(HAND_CUTOFFS[mix][target], weights)
                check(row["fell_back"] is False and close(row["cutoff"], HAND_CUTOFFS[mix][target], 5e-5)
                      and row["threshold"] is None
                      and close(row["quality"]["value"], quality) and close(row["cost_per_1k"]["value"], cost)
                      and close(row["escalation_rate"], escalated) and close(row["savings_vs_A"]["per_1k"], a_cost - cost),
                      f"haiku/Jev {mix} {target}: quality {quality:.4f}, ${cost:.4f} per 1k, escalated {escalated:.3f} by hand")  # fmt: skip
        check(set(parts[final_gates.tasks_key(final_gates.JEV_SECTION)]["tasks"]) == set(TASK_ITEMS)
              == set(parts[final_gates.tasks_key(final_gates.FREE_SECTION)]["tasks"]),
              "each cascade also has a part of per-task rows")  # fmt: skip
        cut = {HAND_CUTOFFS[m][t] for m in HAND_CUTOFFS for t in HAND_CUTOFFS[m]}
        check(len(cut) == 5, "haiku/Jev's six points sit on five distinct cutoffs, the 100% points' shared")
        same = jev_rows["realistic"]["100%"], jev_rows["uniform"]["100%"]
        check(same[0]["cutoff"] == same[1]["cutoff"], "the two mixes' 100% points share a cutoff")
        trace_free = r.record["gos"][0]["steps"]
        check(next(s for s in trace_free if s["name"] == "tables")["status"] == "done", "the tables step is done")


# ---- nothing shown before the tables ------------------------------------------------------------------------------


def counts_with_the_tables() -> None:
    print("\n-- the counts that carry outcomes come with the tables --")
    with ci.rehearsed() as r:
        outcome, out, err = r.go()
        before = out.split("tables: starts")[0] + err
        check(outcome == Outcome(), f"the whole rehearsal runs {outcome}")
        check(not any(word in before for word in ("rejected", "refused", "refusal", "truncat", "cut ", "failed a")),
              "nothing before the tables step names a rejection, a refusal or a cut")  # fmt: skip
        extras = next(s for s in r.record["gos"][0]["steps"] if s["name"] == "tables")["detail"]["extras"][
            final_gates.GATE_TABLE
        ]
        haiku, sonnet = extras["haiku"], extras["sonnet"]
        check(haiku["answers"] == 8 and haiku["rejected"] == 1 and haiku["failed_first"] == {"stop": 1, "schema": 0, "code": 0}
              and haiku["jev_calls"] == 7 and haiku["jev_refused"] == 0 and haiku["jev_cut"] == 0
              and sonnet["rejected"] == 1 and sonnet["jev_calls"] == 7 and extras["opus"]["jev_calls"] == 8,
              f"the tables step's record holds the free-check rejections and the Jev gate's calls, refusals and cuts per config {haiku}")  # fmt: skip
        check(
            all(not key.startswith("sonnet") for c in extras.values() for key in c), "and no Sonnet gate field: #86's"
        )


def main() -> int:
    projectors = tuple(p for p in final_ledger.PROJECTORS if p.name != "jev_router")
    with mock.patch.object(final_ledger, "PROJECTORS", projectors):
        what_the_gates_see()
        drift_quiet()
        drift_refusals()
        free_checks()
        test_calls()
        resume()
        table_checks()
        cascades()
        counts_with_the_tables()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
