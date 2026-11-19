"""Check the Sonnet gate on test and its drift check (PLAN_v4.md Amendment A19, items 2, 5, 7, 9, 10 and 13; A17 items 13, 14, 17
and 26; A15 item 17; A6 item 6; issue #77, #86).

Run from the project root:  uv run python -m analysis.check_final_sonnet

On the rehearsal's made-up world (`analysis/rehearsal_world.py`: eight test items, four configs each) with a stand-in sandbox, a
made-up Stage 7 sample for the Sonnet gate and a scripted fake Anthropic client (`analysis/rehearsal_sonnet.py`, behind
`analysis/fake_anthropic.py`). No results/, no Docker, no Jev or Anthropic call, and nothing near the real store or a real test
item. The hook runs `analysis/check_final_sonnet_real.py` (the drift check's inputs against Stage 7's stored records) too. Every
expected number is worked by hand below.

- **What the gate sees.** A reading of `analysis/final_sonnet.py` finds no graded item, expected answer or response JSON in it;
  the steps run with a spy on the gate's request builder, and what reached it is the request and the answer's text of each
  gated answer that passed. A marker in an expected answer and one in every thinking block reach no request, no stored record
  and nothing the fake saw; the answer's own marker does (the search's known positive).
- **The drift check.** Stage 7's made-up sample gives D6 = 0.02 and a shift of 0 (the standard API 2 points off the batch's scores,
  above and below); the resend (`quiet`) gives D8 = 0.03 and a shift of 0, so it reports and goes on, with the decisions that flip
  at each frozen Sonnet cutoff worked by hand ([2, 2, 1] at 0.77, 0.5 and 0.3). Each resend is the frozen request: the fake's log
  holds the hand-written JSON and the sha256 Stage 7 stored, and the records are in `sonnet_gate_drift.jsonl` and no other file. Each
  refusal stops the run before any call on test, one case each: a model ID that isn't the pin, D8 = 0.08 above 3 x D6 = 0.06, and
  a uniform move (D8 = 0.03, shift +0.03) that the D8 rule lets through and the shift rule stops; the floor makes D6 0.01 where the
  standard API repeated the batch exactly. A request that differs from Stage 7's, a missing sample and the cap's boundary stop it
  before any call. A rerun resends nothing.
- **The test batch**, for the 22 answers of `haiku`, `sonnet` and `opus_low` that passed the free checks (none of `opus`'s): one
  batch, its requests the hand-written JSON of the frozen call in the handle's order, each request ID the answer's own, each
  verdict the item's scripted score under its own ID though the results arrive shuffled; the ledger's recorded spend by hand;
  the batch's ID and creation time in the step's record and in the batch log of the output folder, and Stage 7's batch log
  untouched. A refusal, a cap stop and an unparseable text are rejections charged their billed tokens, and the counts come with the
  tables. A rerun sends nothing.
- **The batch safeguards, one check each (A6 item 6, A17 item 14).** Batch IDs are saved on creation: a stop between two batches
  leaves the first one's ID in the log, and the next go waits for it and sends only the rest. Only errored or expired requests are
  resent, in a batch of exactly those; a canceled one stops the run. Submission is refused over the $25 cap, at the boundary and with
  nothing sent. Results are keyed by request ID: they arrive shuffled, and a result for an ID the batch never held stops the run.
- **Test's gate-score table** with the Sonnet column, refused whole on each of A17 item 13's conditions for it, one at a time.
- **The pure cascade** `haiku/Sonnet` at its frozen points, quality, cost and escalation worked by hand: the two points that route
  (97% under each mix, at cutoffs 0.5 and 0.3) and the four that fall back to A.
- **Nothing shows before the tables:** no refusal, cap stop or rejection is in the log before the tables step; the tables step's
  record holds them.
"""

import ast
import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest import mock

from analysis import check_final_gates as gates
from analysis import check_final_run as ci
from analysis import collect_sonnet_gate, final_gates, final_ledger, final_sonnet, rehearsal_gates, rehearsal_sonnet
from analysis import rehearsal_world as rehearsal
from analysis.collect_gates import GateCollectError
from analysis.collect_sonnet_gate import Stage8Files
from analysis.final_ledger import CALLS
from analysis.final_run import STEPS, Outcome, Step
from analysis.gate_table import GateScoreTable, GateTableError, PaidScore
from analysis.router_names import SONNET_GATE
from analysis.unlock import Handle, HandleError, unlock
from harness.client import PROJECT_ROOT
from routers import jev, sonnet_gate
from routers.request import Request

problems: list[str] = []
check, raises, close = gates.check, gates.raises, gates.close
# `gates.check` keeps its own list of problems, so a failure here is counted from it too.
RESULTS_FILES = ("sonnet_gate_test.jsonl", "sonnet_gate_drift.jsonl")


def failures() -> int:
    return len(gates.problems) + len(problems)


# ---- the world's numbers, worked by hand ----------------------------------------------------------------------------

SETTINGS = sonnet_gate.load_settings()
IDS, CONFIGS, STOP_FAILS = gates.IDS, gates.CONFIGS, gates.STOP_FAILS
POSITION = {item: p for p, item in enumerate(IDS)}
GATED = ("haiku", "sonnet", "opus_low")
# A verdict is billed 1,000 in and 50 out (a cap stop 16,000 out) at list $2 and $10 per million, and half of that in a batch.
LIST_CHECK = (1000 * 2 + 50 * 10) / 1e6  # $0.0025
LIST_CAP_STOP = (1000 * 2 + 16000 * 10) / 1e6  # $0.162
DRIFT_ANSWERS, TEST_VERDICTS = 6, 22
# The dev record's own numbers (A19 item 9: the Sonnet gate cost $6.98 on dev, $13.952102 at list price over 744 + 743 + 742
# verdicts), and the projection built from them: twice dev's list price per verdict, and half of that for a batch request.
DEV_LIST_DOLLARS, DEV_VERDICTS = 13.952102, 744 + 743 + 742
RATE = 2 * DEV_LIST_DOLLARS / DEV_VERDICTS  # a check's projected cost at list price
STAGE_7_FILES = ["jev_gate_train.jsonl", "sonnet_gate_latency.jsonl", "sonnet_gate_train.jsonl"]

SCORE = {IDS[p]: s for p, s in enumerate(rehearsal_sonnet.TEST_SCORES)}  # the item's whole-number score
KEYS = [k for k in gates.passing() if k[1] in GATED]  # the 22 requests, in the handle's order (item, then config)
CUSTOM = {k: f"{k[0].replace(':', '-')}_{k[1]}_r0_sg" for k in KEYS}  # the request ID, as A6 item 5 words it
FIRST_PLY = {"quiet": [2, 2, 1]}  # flips at 0.77, 0.5 and 0.3, worked below


def hand_params(request: Request, answer: str) -> dict:
    """A request body written out as A17 item 6 words it: the message, the schema block if the request has a schema, low effort,
    the whole-number schema, no temperature, no thinking field, no system prompt."""
    block = ""
    if request.output_schema is not None:
        block = f"<output_schema>\n{json.dumps(request.output_schema, ensure_ascii=False)}\n</output_schema>\n"
    content = (
        "You are checking another model's answer to a request. Judge whether the answer is fully correct: it answers what was "
        "asked, every fact, calculation, label or piece of code in it is right, and it follows every instruction in the "
        "request, including any rules about format, length or wording.\n\n<request>\n" + request.prompt + "\n</request>\n"
        + block + "<answer>\n" + answer + "\n</answer>\n\nGive the probability, from 0 to 100, that the answer is fully "
        "correct."
    )  # fmt: skip
    schema = {"type": "object", "properties": {"probability_correct": {"type": "integer", "enum": list(range(101))}},
              "required": ["probability_correct"], "additionalProperties": False}  # fmt: skip
    return {"model": "claude-sonnet-5-5", "max_tokens": 16000,
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            "messages": [{"role": "user", "content": content}]}  # fmt: skip


def sha(params: dict) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def hand_projection() -> float:
    """The Sonnet gate's projection at the liveness step of the rehearsal, by hand: the 6 drift resends at twice dev's list price
    per verdict, and the 24 gated test answers (no free check has run, so every one is assumed to pass) at half of that."""
    return DRIFT_ANSWERS * RATE + 24 * RATE * 0.5


def hand_spend(drift: int = DRIFT_ANSWERS, test_list: float = TEST_VERDICTS * LIST_CHECK) -> float:
    """Stage 8's recorded spend in the calls pool after the Jev gate's 8 + 30 calls and the Sonnet gate's: drift calls at list price
    and the test batch at half of its list cost."""
    return 38 * gates.OVERHEAD + drift * LIST_CHECK + test_list * 0.5


def to(name: str) -> tuple[Step, ...]:
    """Every step up to `name`, the Jev gate's included and the Jev routers' (#84's, held by `analysis/check_final_jev.py`) left out:
    their calls and spend would be counted in these steps' hand-worked numbers."""
    names = [s.name for s in STEPS]
    return tuple(s for s in STEPS[: names.index(name) + 1] if s.name not in gates.ROUTER_STEPS)


def calls_spend(r: ci.Rehearsal) -> float:
    """What the ledger counts for the gates' records alone: a full go also holds the Jev routers' calls, which have a check of
    their own, so the records of `results/jev/` are left out of the copy the ledger reads."""
    with tempfile.TemporaryDirectory() as folder:
        shutil.copytree(r.world.results / "gates", Path(folder) / "gates")
        return final_ledger.recorded_spend(Path(folder), r.world.output).calls


def detail(r: ci.Rehearsal, step: str, go: int = -1) -> dict:
    return next(s for s in r.record["gos"][go]["steps"] if s["name"] == step)["detail"]


def stored_test(r: ci.Rehearsal) -> list[dict]:
    return final_gates.read_sonnet_test(r.world.results)


def log_file(r: ci.Rehearsal) -> Path:
    return r.world.output / final_sonnet.BATCH_LOG


def log_entries(r: ci.Rehearsal) -> list[dict]:
    return collect_sonnet_gate.load_log(log_file(r))["batches"]


# ---- the dev record's numbers ---------------------------------------------------------------------------------------


def dev_numbers() -> None:
    print("-- the projection's inputs are the dev record's --")
    record = json.loads((PROJECT_ROOT / "analysis" / "gates_dev_record.json").read_text(encoding="utf-8"))
    counts = [record["counts"][c]["sonnet_calls"] for c in GATED]
    check(counts == [744, 743, 742] and record["spend"]["batch_list"] == DEV_LIST_DOLLARS,
          "the committed dev record holds 744 + 743 + 742 verdicts and $13.952102 at list price (the batch cost $6.976051)")  # fmt: skip
    check(close(final_sonnet.dev_dollars_per_check(), DEV_LIST_DOLLARS / DEV_VERDICTS, 1e-15)
          and close(final_sonnet.dollars_per_check(batch=False), RATE, 1e-15)
          and close(final_sonnet.dollars_per_check(batch=True), RATE / 2, 1e-15),
          f"a check is projected at ${RATE:.6f} (twice dev's per verdict) at list price, and half of that in a batch")  # fmt: skip
    check(
        24 * RATE / 2 + 80 * RATE < 25,
        "dev's own scale (about 2,230 verdicts and 80 resends, doubled) is under the $25 cap",
    )


# ---- what the gate sees ---------------------------------------------------------------------------------------------


class Spy:
    def __init__(self, real: Callable[..., Any]):
        self.real = real
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.real(*args, **kwargs)


def what_the_gate_sees() -> None:
    print("\n-- what the gate sees: the request and the answer's text, and nothing else --")
    source = Path(final_sonnet.__file__).read_text(encoding="utf-8")
    check(gates.forbidden_names(source) == set(),
          "the module that runs the gate names no graded item, expected answer or response JSON")  # fmt: skip
    check(gates.handle_reads(source) == set(),
          "and reaches the handle only through `final_gates.test_inputs`, its requests and answers read")  # fmt: skip
    check(gates.forbidden_names("handle.graded_items()[0].item.expected_answer") == {"graded_items", "expected_answer"},
          "known positive: the same search finds a graded item and an expected answer where they are named")  # fmt: skip
    tree = ast.parse(source)
    check(not any(isinstance(n, ast.Attribute) and n.attr in ("stop_reason", "expected_answer") for n in ast.walk(tree)),
          "the answer's stop reason is not read for the gate (it sees the answer's text, not the grade or the stop)")  # fmt: skip

    known = gates.made_up_answers()
    with ci.rehearsed() as r:
        spy = Spy(sonnet_gate.build_request)
        with mock.patch.object(sonnet_gate, "build_request", spy):
            outcome, out, _ = r.go(to("calls_sonnet_gate"))
        check(outcome == Outcome(), f"the rehearsal runs up to the Sonnet gate {outcome}")
        train_prompts = {
            rehearsal_gates.train_prompt(k): rehearsal_gates.train_answer(k) for k in rehearsal_sonnet.SAMPLE
        }
        asked = {(a[1].prompt, a[2]) for a, _ in spy.calls}
        wanted = {(request.prompt, answer) for request, answer, _ in known.values()} | set(train_prompts.items())
        check(all(len(a) == 3 and isinstance(a[0], sonnet_gate.SonnetSettings) and isinstance(a[1], Request)
                  and isinstance(a[2], str) and k == {} for a, k in spy.calls) and asked <= wanted,
              "the gate's request was built from the settings, a request and an answer's text alone, for the stored answers")  # fmt: skip
        sent = json.dumps(r.anthropic.calls + [x for batch in r.anthropic.created.values() for x in batch])
        stored = "\n".join(
            p.read_text(encoding="utf-8") for p in sorted((r.world.results / "gates").glob("sonnet_gate_*"))
        )
        for name, text in (("the requests the fake received", sent), ("the stored records", stored)):
            check(rehearsal.EXPECTED_MARKER not in text and rehearsal.THINKING_MARKER not in text,
                  f"an expected answer's marker and a thinking block's reach neither {name}")  # fmt: skip
        check(rehearsal.ANSWER_MARKER in sent and rehearsal.ANSWER_MARKER not in stored,
              "known positive: an answer's own marker does reach the requests (the records keep the reply, not the request)")  # fmt: skip
        check(rehearsal.GRADE_MARKER not in sent + stored + out,
              "the stop reason that carries the grade marker reaches no request, record or log line")  # fmt: skip
        everything = (r.world.output / "final_record.json").read_text(encoding="utf-8") + out
        check(rehearsal.EXPECTED_MARKER not in everything and rehearsal.THINKING_MARKER not in everything,
              "and no marker reaches the record or the log")  # fmt: skip


# ---- the drift check ------------------------------------------------------------------------------------------------

TRAIN = {k: rehearsal_gates.train_input(k) for k in rehearsal_sonnet.SAMPLE}


def drift_quiet() -> None:
    print("\n-- the drift check under its limits reports and goes on --")
    with ci.rehearsed() as r:
        latency_file = r.world.results / "gates" / "sonnet_gate_latency.jsonl"
        train_file = r.world.results / "gates" / "sonnet_gate_train.jsonl"
        before = latency_file.read_bytes() + train_file.read_bytes()
        outcome, out, _ = r.go(to("drift_sonnet_gate"))
        check(outcome == Outcome(), f"the run goes on past the drift check {outcome}")
        d = detail(r, "drift_sonnet_gate")
        m = d["measures"]["haiku"]
        check(close(m["d6"], 0.02) and close(m["d8"], 0.03) and close(m["shift"], 0.0),
              f"D6 0.02, D8 0.03 and shift 0 are the hand-worked ones {m}")  # fmt: skip
        hand = {0.77: ["banking77:made-train-1 haiku", "banking77:made-train-2 haiku"],
                0.5: ["banking77:made-train-4 haiku", "banking77:made-train-5 haiku"],
                0.3: ["banking77:made-train-3 haiku"]}  # fmt: skip
        check([f["cutoff"] for f in d["flips"]] == [0.77, 0.5, 0.3] and [f["items"] for f in d["flips"]] == list(hand.values())
              and [f["count"] for f in d["flips"]] == FIRST_PLY["quiet"],
              f"the decisions that flip at each frozen cutoff, highest first, are worked by hand {FIRST_PLY['quiet']}")  # fmt: skip
        labelled = {f["cutoff"]: f["points"] for f in d["flips"]}
        check("haiku/Sonnet realistic 97%" in labelled[0.5] and "haiku/Sonnet uniform 97%" in labelled[0.3]
              and len(labelled[0.77]) == 9 and all("Sonnet" in p for ps in labelled.values() for p in ps),
              "each cutoff names the points on it: haiku/Sonnet's two that route, and the nine of the router-started cascades")  # fmt: skip
        check(d["model_ids"] == ["claude-sonnet-5-5"] and d["pin"] == "claude-sonnet-5-5" and d["answers"] == 6
              and d["unscored"] == {"haiku": 0},
              "the resends came back as the pin, 6 of them, none unscored")  # fmt: skip
        check("D6 0.0200, D8 0.0300" in out and "drift_sonnet_gate: done" in out,
              "the log says D6 and D8, which are train's numbers")  # fmt: skip

        records = final_sonnet.read_drift(r.world.results)
        sample = [(TRAIN[k].item_id, "haiku") for k in rehearsal_sonnet.SAMPLE]
        check([(x["item_id"], x["config"]) for x in records] == sample
              and {(x["split"], x["gate"], x["pass"], x["api"], x["batch_id"]) for x in records}
              == {("drift", "sonnet", 2, "standard", None)} and [x["first_call"] for x in records] == [True] + [False] * 5,
              "the resends are in sonnet_gate_drift.jsonl: one record each, labelled drift, standard API, in the sample's order")  # fmt: skip
        hand_requests = [hand_params(TRAIN[k].request, TRAIN[k].answer) for k in rehearsal_sonnet.SAMPLE]
        stage_7 = [json.loads(line)["request_sha256"] for line in latency_file.read_text(encoding="utf-8").splitlines()]
        sent = r.anthropic.calls[2:]  # the liveness step's two calls came first
        check(sent == hand_requests and [sha(p) for p in sent] == [x["request_sha256"] for x in records] == stage_7,
              "each resend is the hand-written request, and its hash is the one Stage 7 stored for the same answer")  # fmt: skip
        check(r.anthropic.created == {} and [(x["usage"]["input_tokens"], x["usage"]["output_tokens"]) for x in records]
              == [(1000, 50)] * 6,
              "no batch was made, and each record holds its billed tokens")  # fmt: skip
        check(latency_file.read_bytes() + train_file.read_bytes() == before
              and sorted(p.name for p in (r.world.results / "gates").iterdir())
              == sorted([*STAGE_7_FILES, "jev_gate_drift.jsonl", "sonnet_gate_drift.jsonl"]),
              "Stage 7's files are as they were, and the drifts' own are the only new files: no test file yet")  # fmt: skip
        spent_calls = calls_spend(r)
        check(close(spent_calls, 8 * gates.OVERHEAD + 6 * LIST_CHECK, 1e-12),
              f"the ledger counts the resends at list price: 6 verdicts of $0.0025, and the Jev drift's 8 ${spent_calls:.6f}")  # fmt: skip
        outcome, _, _ = r.go(to("drift_sonnet_gate"))
        check(outcome == Outcome() and len(r.anthropic.calls) == 2 + 6 + 2,
              "a second go resends nothing: the fake's log holds the first go's 6 resends and only the liveness calls again")  # fmt: skip
        check(final_sonnet.read_drift(r.world.results) == records
              and detail(r, "drift_sonnet_gate")["measures"] == d["measures"],
              "and reports the same numbers from what is stored")  # fmt: skip


def refused_at(
    scenario: str, what: str, expect: list[str], not_expect: list[str] | None = None, **changes: Any
) -> None:
    with ci.rehearsed() as r:
        r.sonnet.resend(scenario)
        for name, value in changes.items():
            setattr(r.sonnet, name, value)
        outcome, out, _ = r.go()
        asked = [rehearsal_sonnet.prompt_of(p) for p in r.anthropic.calls[2:]]
        shown = r.record["gos"][0]["stopped"]
        check(outcome == Outcome("drift_sonnet_gate", "DriftRefusal", None),
              f"{what}: the run stops at the drift check with a DriftRefusal {outcome}")  # fmt: skip
        check(all(text in out for text in expect) and not any(text in out for text in (not_expect or [])),
              f"{what}: the log says why, with train's numbers only {[t for t in expect if t not in out]}")  # fmt: skip
        check(len(asked) == 6 and all(p.startswith("Which kind of card problem") for p in asked) and r.anthropic.created == {},
              f"{what}: the model was sent the 6 resends and no test request, and no batch was made")  # fmt: skip
        results = r.world.results / "gates"
        check(not (results / "free_checks_test.jsonl").exists() and not (results / "sonnet_gate_test.jsonl").exists()
              and not log_file(r).exists() and not (r.world.results / "test_matrix.sqlite").exists()
              and shown["step"] == "drift_sonnet_gate",
              f"{what}: no test record, no batch log and no grade exists, and the record says where it stopped")  # fmt: skip


def with_stage7(
    r: ci.Rehearsal, standard: tuple[int, ...] | None = None, tamper: Callable[[dict], None] | None = None
) -> None:
    """Rewrite the made-up Stage 7 files of the Sonnet gate: with another standard-API sample, and a change to one record."""
    for name in ("sonnet_gate_train.jsonl", "sonnet_gate_latency.jsonl"):
        (r.world.results / "gates" / name).unlink()
    rehearsal_sonnet.write_train(r.world.results, standard=standard or rehearsal_sonnet.STANDARD)
    if tamper is not None:
        path = r.world.results / "gates" / "sonnet_gate_latency.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        tamper(records[3])
        path.write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")


def drift_refusals() -> None:
    print("\n-- each refusal stops the run before any test call --")
    refused_at("quiet", "a model ID that isn't the pin", ["claude-sonnet-5-6", "not the pin claude-sonnet-5-5"],
               model="claude-sonnet-5-6")  # fmt: skip
    refused_at("loud", "D8 above 3 x D6 (0.08 against 0.06)", ["haiku: D8 0.08", "more than 3 x D6 0.02"],
               ["mean signed difference"])  # fmt: skip
    refused_at("uniform", "a uniform move (D8 0.03, shift +0.03)",
               ["the shift (mean signed difference) +0.03", "larger than D6 0.02"], ["more than 3 x"])  # fmt: skip

    print("\n-- the floor: D6 is at least 0.01, so a standard API that repeated the batch exactly still allows 0.03 --")
    flat = rehearsal_sonnet.BATCH[:6]
    with ci.rehearsed() as r:
        with_stage7(r, standard=flat)
        r.sonnet.move((2, -2, 2, -2, 2, -2))
        outcome, _, _ = r.go(to("drift_sonnet_gate"))
        m = detail(r, "drift_sonnet_gate")["measures"]["haiku"]
        check(outcome == Outcome() and close(m["d6"], 0.01) and close(m["d8"], 0.02) and close(m["shift"], 0.0),
              f"D6 is floored at 0.01 (it measures 0), and D8 0.02 is under 3 x 0.01 {m}")  # fmt: skip
    with ci.rehearsed() as r:
        with_stage7(r, standard=flat)
        r.sonnet.move((4, -4, 4, -4, 4, -4))
        outcome, out, _ = r.go(to("drift_sonnet_gate"))
        check(outcome == Outcome("drift_sonnet_gate", "DriftRefusal", None) and "haiku: D8 0.04" in out
              and "more than 3 x D6 0.01" in out,
              f"and a resend 0.04 off is refused against the floored 0.03 {outcome}")  # fmt: skip

    print("\n-- a request that differs from Stage 7's, a missing sample and a cap --")
    with ci.rehearsed() as r:
        with_stage7(r, tamper=lambda record: record.update(request_sha256="0" * 64))
        outcome, out, _ = r.go(to("drift_sonnet_gate"))
        check(outcome == Outcome("drift_sonnet_gate", "DriftRefusal", None) and len(r.anthropic.calls) == 2
              and "different from the one Stage 7 stored for 1 of the 6 resent answers" in out,
              f"a request that isn't the one Stage 7 stored stops the run before any call {outcome}")  # fmt: skip
    with ci.rehearsed() as r:
        (r.world.results / "gates" / "sonnet_gate_latency.jsonl").unlink()
        outcome, _, _ = r.go(to("drift_sonnet_gate"))
        check(outcome == Outcome("drift_sonnet_gate", "FinalSonnetError", None) and len(r.anthropic.calls) == 2,
              "a Stage 7 with no standard-API sample has nothing to resend: refused, nothing called")  # fmt: skip
    with ci.rehearsed() as r:
        with mock.patch.dict(final_ledger.CAPS, {CALLS: 0.0001}):
            outcome, out, _ = r.go()
        check(outcome == Outcome("liveness", "LedgerRefusal", None) and r.anthropic.calls == [],
              f"a projection over the cap refuses at the liveness step: no liveness call and no resend {outcome}")  # fmt: skip
    need = 6 * RATE
    jev_recorded = 8 * gates.OVERHEAD
    for label, cap, expect in (("exactly at", jev_recorded + need + 1e-12, Outcome()),
                               ("a hair below", jev_recorded + need - 1e-6, Outcome("drift_sonnet_gate", "LedgerRefusal", None))):  # fmt: skip
        with ci.rehearsed() as r, mock.patch.dict(final_ledger.CAPS, {CALLS: cap}):
            outcome, _, _ = r.go(gates.without_liveness(to("drift_sonnet_gate")))
            resends = [p for p in r.anthropic.calls if rehearsal_sonnet.prompt_of(p).startswith("Which kind")]
            check(outcome == expect and len(resends) == (6 if expect == Outcome() else 0),
                  f"the drift step's own cap check at {label} the recorded spend plus the resends' ${need:.6f}: {outcome}")  # fmt: skip

    print("\n-- a resend billed far above its projection still stops at the cap, before the next resend --")
    with (
        ci.rehearsed() as r,
        mock.patch.object(rehearsal_sonnet, "USAGE", (1_000_000, 50)),
        mock.patch.dict(final_ledger.CAPS, {CALLS: jev_recorded + need + 1e-12}),
    ):
        outcome, out, _ = r.go(gates.without_liveness(to("drift_sonnet_gate")))
        resends = [p for p in r.anthropic.calls if rehearsal_sonnet.prompt_of(p).startswith("Which kind")]
        check(outcome.stopped_at == "drift_sonnet_gate" and outcome.error == "LedgerRefusal" and len(resends) == 1,
              f"each resend is billed 1 million input tokens, the projection fits, and the run stops after the first {outcome}")  # fmt: skip
        check("over Stage 8's cap" in out, "and the refusal names the cap")

    print("\n-- the batch collector's Stage 8 files can be named only with a handle the unlock issued --")
    with ci.rehearsed() as r:
        records, log = r.world.results / "gates" / "sonnet_gate_test.jsonl", r.world.output / "log.json"
        for how, handle in (
            ("no handle", None),
            ("a handle built by hand", Handle(r.world, ())),
            ("a string", "handle"),
        ):
            check(raises(HandleError, lambda h=handle: Stage8Files(h, records, log)),  # type: ignore[arg-type,misc]
                  f"Stage8Files refuses {how}, before any file is read or any batch made")  # fmt: skip
        # With the files unnamed, submit takes Stage 7's own paths (the dev split here) and no handle: the old calls are as they were.
        plain = collect_sonnet_gate.submit(
            [], split="dev", client=None, cost_per_check=0.0, max_usd=1.0, results_dir=r.world.results
        )
        check(plain.sent == 0, "a call that names no Stage 8 file asks for no handle, as Stage 7's collection does")
        stage8 = Stage8Files(unlock(r.world), records, log)
        summary = collect_sonnet_gate.submit(
            [], split="test", client=None, cost_per_check=0.0, max_usd=1.0, results_dir=r.world.results, stage8=stage8
        )
        check(summary.sent == 0, "known positive: the issued handle's Stage8Files is accepted for the same call")


# ---- the batch on test ----------------------------------------------------------------------------------------------


def calls_default() -> None:
    print("\n-- the Sonnet gate on every gated answer that passed the free checks, in one batch --")
    known = gates.made_up_answers()
    without_routers = tuple(p for p in final_ledger.PROJECTORS if p.name != "jev_router")  # the Jev routers' are #84's
    with ci.rehearsed() as r, mock.patch.object(final_ledger, "PROJECTORS", without_routers):
        outcome, out, _ = r.go(to("calls_sonnet_gate"))
        fake = r.anthropic
        check(outcome == Outcome() and len(KEYS) == TEST_VERDICTS and not any(k[1] == "opus" for k in KEYS),
              "the step runs: 22 gated answers passed the free checks (haiku 7, sonnet 7, opus_low 8) and none is opus's")  # fmt: skip
        check(list(fake.created) == ["msgbatch_fake0"], "one batch was made")
        batch = fake.created["msgbatch_fake0"]
        hand = [{"custom_id": CUSTOM[k], "params": hand_params(known[k][0], known[k][1])} for k in KEYS]
        check(batch == hand, "its 22 requests are the hand-written frozen requests, with the answers' own request IDs, in the handle's order")  # fmt: skip
        check(all(c["custom_id"] == f"{k[0].replace(':', '-')}_{k[1]}_r0_sg" and len(c["custom_id"]) <= 64
                  and sonnet_gate.custom_id(SETTINGS, *k) == c["custom_id"] for c, k in zip(batch, KEYS, strict=True)),
              "each request ID is the item with its colon replaced, the config, run 0 and the suffix, within the API's limits")  # fmt: skip
        records = stored_test(r)
        check(sorted((x["item_id"], x["config"]) for x in records) == sorted(KEYS) and len(records) == 22,
              "the verdicts stored are those 22, one record each")  # fmt: skip
        by_key = {(x["item_id"], x["config"]): x for x in records}
        arrival = [item.custom_id for item in fake.messages.batches.results("msgbatch_fake0")]
        check(arrival != [c["custom_id"] for c in batch] and sorted(arrival) == sorted(CUSTOM.values()),
              "known positive: the results came back in another order than the requests, all of them")  # fmt: skip
        check(all(x["custom_id"] == CUSTOM[k] and x["batch_id"] == "msgbatch_fake0" and x["request_sha256"] == sha(hand_params(*known[k][:2]))
                  and x["split"] == "test" and x["gate"] == "sonnet" and x["pass"] == 1 and x["run"] == 0
                  and x["model"] == "claude-sonnet-5-5" and x["usage"] == {"input_tokens": 1000, "output_tokens": 50}
                  and x["refused"] is False and x["stop_reason"] == "end_turn"
                  and json.loads(x["result"]["message"]["content"][0]["text"]) == {"probability_correct": SCORE[k[0]]}
                  for k, x in by_key.items()) and all(tuple(x) == collect_sonnet_gate.RECORD_FIELDS for x in records),
              "each verdict is stored under its own request ID with the item's scripted score, though the results arrived shuffled")  # fmt: skip
        spent_calls = calls_spend(r)
        check(close(spent_calls, hand_spend(), 1e-12),
              f"the ledger's recorded spend is the Jev gate's 38 calls, 6 resends at list price and 22 verdicts at half: ${spent_calls:.6f}")  # fmt: skip
        step = detail(r, "calls_sonnet_gate")
        entry = log_entries(r)[0]
        created = datetime.fromisoformat(step["batches"][0]["created_at"])
        check(len(step["batches"]) == 1 and step["batches"][0]["id"] == "msgbatch_fake0" == entry["id"]
              and step["batches"][0]["created_at"] == entry["created_at"] and created.tzinfo is not None
              and step["batches"][0]["request_count"] == 22 and step["batch_resent"] == 0
              and step["batches"][0]["counts"] == {"succeeded": 22} and step["model_ids"] == ["claude-sonnet-5-5"],
              "the step's record holds the test batch's ID and creation time, its request count and its results' kinds (A19 item 10)")  # fmt: skip
        check(entry["split"] == "test" and entry["ingested_at"] is not None and entry["custom_ids"] == [c["custom_id"] for c in batch]
              and log_file(r) == r.world.output / "sonnet_batches_test.json"
              and not (r.world.results / "gates" / "sonnet_batches.json").exists(),
              "the batch log is Stage 8's own, in the output folder, and Stage 7's batch log was not touched")  # fmt: skip
        step_info = next(s for s in r.record["gos"][0]["steps"] if s["name"] == "calls_sonnet_gate")
        check(step_info["requests"] == {"sent": 22, "stored": 0}, "the record holds the request counts")
        before = out.split("tables: starts")[0]
        line = next(x for x in before.splitlines() if "Sonnet gate" in x and "requests sent" in x)
        check("22 requests sent in 1 batches, 0 already stored" in line and "score" not in line
              and not any(word in before for word in ("refus", "rejected", "cap stop", "unparseable")),
              "the log says how many requests were sent and not what came back: no score, refusal or cap stop")  # fmt: skip
        oldest = (r.world.results / "gates" / "sonnet_gate_test.jsonl").read_bytes()
        outcome, out, _ = r.go(to("calls_sonnet_gate"))
        check(outcome == Outcome() and list(fake.created) == ["msgbatch_fake0"]
              and (r.world.results / "gates" / "sonnet_gate_test.jsonl").read_bytes() == oldest
              and "0 requests sent in 0 batches, 22 already stored" in out and len(log_entries(r)) == 1,
              "a second go sends nothing, makes no batch, and the file is as it was")  # fmt: skip
        check(detail(r, "calls_sonnet_gate")["batches"][0]["id"] == "msgbatch_fake0",
              "and its record still names the test batch, read back from the log")  # fmt: skip
        projected = r.record["gos"][1]["steps"][1]["detail"]["spend"][CALLS]["projected"]
        check(projected == 0.0, f"the second go's whole-run projection sees nothing left to send: ${projected}")


def safeguards() -> None:
    print("\n-- the batch safeguards, one check each --")
    expected_ids = sorted(CUSTOM.values())
    print("   batch IDs saved on creation")
    with ci.rehearsed() as r, mock.patch.dict(final_sonnet.BATCH_LIMITS, {"max_requests": 10}):
        r.anthropic.create_error_after = 1  # the second `create` raises
        outcome, out, _ = r.go()
        entries = log_entries(r)
        check(outcome == Outcome("calls_sonnet_gate", "RuntimeError", None) and len(r.anthropic.created) == 1,
              f"a stop between two batches ends the run at the step; one batch was made {outcome}")  # fmt: skip
        check(len(entries) == 1 and entries[0]["id"] == "msgbatch_fake0" and entries[0]["request_count"] == 10
              and entries[0]["ingested_at"] is None and entries[0]["created_at"]
              and entries[0]["custom_ids"] == [CUSTOM[k] for k in KEYS[:10]] and stored_test(r) == [],
              "the first batch's ID, creation time and 10 request IDs are in the log, unfinished, with nothing stored")  # fmt: skip
        r.anthropic.create_error_after = None
        outcome, out, _ = r.go()
        sent = [c["custom_id"] for batch in r.anthropic.created.values() for c in batch]
        check(outcome == Outcome() and list(r.anthropic.created) == ["msgbatch_fake0", "msgbatch_fake1", "msgbatch_fake2"]
              and [len(b) for b in r.anthropic.created.values()] == [10, 10, 2]
              and sorted(sent) == expected_ids and len(set(sent)) == 22,
              "the next go finishes the first batch, then sends only the other 12 in batches of 10 and 2: no request twice")  # fmt: skip
        check(sorted(x["custom_id"] for x in stored_test(r)) == expected_ids
              and [b["id"] for b in detail(r, "calls_sonnet_gate")["batches"]] == ["msgbatch_fake0", "msgbatch_fake1", "msgbatch_fake2"],
              "all 22 verdicts are stored once, and the record names the three batches")  # fmt: skip

    print("   only errored or expired requests are resent")
    errored, expired = CUSTOM[(IDS[1], "sonnet")], CUSTOM[(IDS[2], "opus_low")]
    with ci.rehearsed() as r:
        r.anthropic.batch_failures = {errored: "errored", expired: "expired"}
        outcome, out, _ = r.go()
        batches = r.anthropic.created
        check(outcome == Outcome() and list(batches) == ["msgbatch_fake0", "msgbatch_fake1"]
              and len(batches["msgbatch_fake0"]) == 22
              and sorted(c["custom_id"] for c in batches["msgbatch_fake1"]) == sorted([errored, expired]),
              "the first batch's two failed results (errored, expired) are sent again in a batch of exactly those two")  # fmt: skip
        step = detail(r, "calls_sonnet_gate")
        check(sorted(x["custom_id"] for x in stored_test(r)) == expected_ids and len(stored_test(r)) == 22
              and step["batch_resent"] == 2 and step["batches"][0]["counts"] == {"succeeded": 20, "errored": 1, "expired": 1}
              and step["batches"][1]["counts"] == {"succeeded": 2}
              and next(s for s in r.record["gos"][0]["steps"] if s["name"] == "calls_sonnet_gate")["requests"]
              == {"sent": 24, "stored": 0},
              "22 verdicts are stored, each once; the record counts 2 resent and 24 sent")  # fmt: skip
    canceled = CUSTOM[(IDS[0], "haiku")]
    with ci.rehearsed() as r:
        r.anthropic.batch_failures = {canceled: "canceled"}
        outcome, _, _ = r.go()
        check(outcome == Outcome("calls_sonnet_gate", "SonnetCollectError", None) and len(r.anthropic.created) == 1
              and canceled not in {x["custom_id"] for x in stored_test(r)},
              "a canceled result stops the run, and nothing is sent again")  # fmt: skip

    print("   submission is refused over the cap")
    need = 22 * RATE / 2
    recorded = 38 * gates.OVERHEAD + 6 * LIST_CHECK
    with ci.rehearsed() as r, mock.patch.dict(final_ledger.CAPS, {CALLS: recorded + need - 1e-6}):
        outcome, out, _ = r.go(gates.without_liveness(to("calls_sonnet_gate")))
        check(outcome == Outcome("calls_sonnet_gate", "LedgerRefusal", None) and r.anthropic.created == {},
              f"a cap a hair below the recorded spend plus the batch's projection refuses, and no batch is made {outcome}")  # fmt: skip
        check(not log_file(r).exists() and stored_test(r) == [] and "calls pool" in out
              and "over Stage 8's cap" in out and "nothing was sent" in out,
              "no batch log and no verdict exist, and the log names the pool and the cap")  # fmt: skip
    with ci.rehearsed() as r, mock.patch.dict(final_ledger.CAPS, {CALLS: recorded + need + 1e-12}):
        outcome, _, _ = r.go(gates.without_liveness(to("calls_sonnet_gate")))
        check(outcome == Outcome() and len(stored_test(r)) == 22, "at the cap exactly, the batch goes")

    print("   results are keyed by request ID")
    for label, extra in (
        ("an ID the batch never held", {"custom_id": "not_sent_r0_sg"}),
        ("an ID returned twice", None),
    ):

        def tamper(batch_id: str, outcomes: list[dict], extra: Any = extra) -> list[dict]:
            return [*outcomes, {**outcomes[0], **(extra or {})}]

        with ci.rehearsed() as r:
            r.anthropic.tamper = tamper
            outcome, _, _ = r.go()
            check(outcome == Outcome("calls_sonnet_gate", "SonnetCollectError", None)
                  and "not_sent_r0_sg" not in {x["custom_id"] for x in stored_test(r)},
                  f"a result for {label} stops the run, and nothing is stored under it {outcome}")  # fmt: skip


# ---- rejections -----------------------------------------------------------------------------------------------------


def rejections() -> None:
    print("\n-- a refusal, a cap stop and an unparseable text are rejections, charged their billed tokens --")
    modes = {0: "REFUSE", 1: "CAP", 2: "GARBAGE"}
    with ci.rehearsed() as r:
        for p, mode in modes.items():
            r.sonnet.modes[rehearsal.ITEMS[p]["prompt"]] = mode
        outcome, out, _ = r.go()
        check(outcome == Outcome(), f"the whole rehearsal runs, the tables included {outcome}")
        by_key = {(x["item_id"], x["config"]): x for x in stored_test(r)}
        hand = {
            "REFUSE": ("refusal", 50, 0.0025),
            "CAP": ("max_tokens", 16000, 0.162),
            "GARBAGE": ("end_turn", 50, 0.0025),
        }
        for p, mode in modes.items():
            for config in GATED:
                if (IDS[p], config) in by_key:
                    x = by_key[(IDS[p], config)]
                    stop, output, _ = hand[mode]
                    check(x["refused"] is True and x["stop_reason"] == stop and x["usage"] == {"input_tokens": 1000, "output_tokens": output},
                          f"{IDS[p]} {config}: stored as refused, stop {stop!r}, billed 1,000 in and {output} out")  # fmt: skip
        free = final_gates.read_free_test(r.world.results)
        table = final_gates.build_test_table(free, final_gates.read_jev("test", r.world.results), sorted(gates.made_up_answers()),
                                             gates.SETTINGS, jev.read_client_settings(),
                                             extra_paid={SONNET_GATE: final_gates.sonnet_column(free, stored_test(r), SETTINGS)})  # fmt: skip
        entries = {(IDS[p], c): table.row(IDS[p], c).sonnet for p in modes for c in GATED if (IDS[p], c) in by_key}
        check(len(entries) == 9 and all(e is not None and e.score is None and e.refused for e in entries.values())
              and all(close(e.overhead, hand[modes[POSITION[k[0]]]][2], 1e-12) for k, e in entries.items() if e is not None),
              "in the table each is a rejection: no score, refused, its overhead the list price of its billed tokens")  # fmt: skip
        spent_calls = calls_spend(r)
        test_list = 19 * LIST_CHECK + 3 * LIST_CAP_STOP
        check(close(spent_calls, hand_spend(test_list=test_list), 1e-12),
              f"the ledger charges the refusals and the cap stops their billed tokens: ${spent_calls:.6f}")  # fmt: skip
        before = out.split("tables: starts")[0]
        check(not any(word in before for word in ("refus", "rejected", "cap stop", "unparseable", "max_tokens")),
              "nothing before the tables step names a refusal, a cap stop or a rejection")  # fmt: skip
        extras = detail(r, "tables")["extras"][final_gates.GATE_TABLE]
        check(extras["haiku"]["sonnet_calls"] == 7 and extras["haiku"]["sonnet_refused"] == 3 and extras["haiku"]["sonnet_cap_stops"] == 1
              and extras["sonnet"]["sonnet_calls"] == 7 and extras["sonnet"]["sonnet_refused"] == 3
              and extras["opus_low"]["sonnet_calls"] == 8 and extras["opus_low"]["sonnet_refused"] == 3
              and extras["opus_low"]["sonnet_cap_stops"] == 1 and extras["opus"]["sonnet_calls"] == 0,
              f"the tables step's record holds the verdicts, refusals and cap stops per config {extras['haiku']}")  # fmt: skip
        parts = r.tables["parts"][final_gates.SONNET_SECTION]["mixes"]
        scores = {p: (None if p in modes else rehearsal_sonnet.TEST_SCORES[p] / 100) for p in range(8)}
        overhead = {p: (LIST_CAP_STOP if p == 1 else LIST_CHECK) * 1e6 for p in range(8)}
        for mix, target, cutoff in (("realistic", "97%", 0.5), ("uniform", "97%", 0.3)):
            quality, cost, escalated = hand_cascade(cutoff, ci.MIXES[mix], scores, overhead)
            row = parts[mix][target]
            check(close(row["quality"]["value"], quality) and close(row["cost_per_1k"]["value"], cost)
                  and close(row["escalation_rate"], escalated),
                  f"haiku/Sonnet {mix} {target} with the three rejections: quality {quality:.4f}, ${cost:.4f} per 1k, escalated {escalated:.3f}")  # fmt: skip


def wrong_model() -> None:
    print("\n-- a stored verdict from a model that isn't the pin stops the run before grading --")
    with ci.rehearsed() as r:
        outcome, _, _ = r.go(to("drift_sonnet_gate"))
        r.sonnet.model = "claude-sonnet-5-6"
        outcome, out, _ = r.go(to("calls_sonnet_gate"))
        check(outcome == Outcome("calls_sonnet_gate", "ModelRefusal", None) and "claude-sonnet-5-6" in out
              and "not the pin claude-sonnet-5-5" in out and not (r.world.results / "test_matrix.sqlite").exists(),
              f"the run stops at the step naming the model, and nothing was graded {outcome}")  # fmt: skip
        check(len(stored_test(r)) == 22, "the 22 verdicts were stored and billed before it")


# ---- test's gate-score table ----------------------------------------------------------------------------------------


def table_checks() -> None:
    print("\n-- test's gate-score table with the Sonnet column, refused whole on each of A17 item 13's conditions --")
    answers = sorted(gates.made_up_answers())
    client = jev.read_client_settings()
    with ci.rehearsed() as r:
        r.go(to("calls_sonnet_gate"))
        free = final_gates.read_free_test(r.world.results)
        calls = final_gates.read_jev("test", r.world.results)
        verdicts = stored_test(r)

        def build(free_records: list[dict] = free, records: list[dict] = verdicts, **kw: Any) -> GateScoreTable:
            column = final_gates.sonnet_column(free_records, records, SETTINGS)
            return final_gates.build_test_table(
                free_records, calls, answers, gates.SETTINGS, client, extra_paid={SONNET_GATE: column}, **kw
            )

        table = build()
        found = {k: e for k in answers if (e := table.row(*k).sonnet) is not None}
        check(len(table.rows) == 32 and sorted(found) == sorted(KEYS) and len(found) == 22,
              "the table has a row for each of the 32 answers, 22 of them with a Sonnet entry: haiku's, sonnet's and opus_low's that passed")  # fmt: skip
        check(all(close(e.score, SCORE[k[0]] / 100) and e.refused is False and close(e.overhead, LIST_CHECK, 1e-12)
                  for k, e in found.items()),
              "each Sonnet entry is the item's scripted score over 100, not refused, at list price: $0.0025 (no batch discount)")  # fmt: skip
        check(all(table.row(*k).sonnet is None for k in answers if k[1] == "opus" or k in gates.STOP_FAILS),
              "an opus answer and an answer that failed a free check have no Sonnet entry")  # fmt: skip

        first = (verdicts[0]["item_id"], verdicts[0]["config"])
        check(raises(GateCollectError, lambda: build(records=verdicts[1:])), "a passing gated answer with no verdict")
        opus_record = {
            **verdicts[0],
            "config": "opus",
            "custom_id": CUSTOM[first].replace("_" + first[1] + "_", "_opus_"),
        }
        check(raises(GateCollectError, lambda: build(records=[*verdicts, opus_record])),
              "a verdict on an opus answer, which the gate never judges")  # fmt: skip
        failed = (IDS[5], "haiku")
        stray = {
            **verdicts[0],
            "item_id": failed[0],
            "config": failed[1],
            "custom_id": CUSTOM[(IDS[0], "haiku")].replace("banking77-made-test-0", "ifeval-made-test-1"),
        }
        check(raises(GateCollectError, lambda: build(records=[*verdicts, stray])),
              "a verdict on an answer that failed a free check")  # fmt: skip
        no_row = [x for x in free if (x["item_id"], x["config"]) != (IDS[5], "haiku")]
        check(raises(GateTableError, lambda: final_gates.build_test_table(no_row, calls, answers, gates.SETTINGS, client)),
              "an answer missing from the free-check records")  # fmt: skip
        check(raises(collect_sonnet_gate.SonnetCollectError, lambda: build(records=[{**verdicts[0], "refused": True}, *verdicts[1:]])),
              "a stored refusal flag the result doesn't bear out")  # fmt: skip
        check(raises(collect_sonnet_gate.SonnetCollectError,
                     lambda: build(records=[{**verdicts[0], "usage": {"input_tokens": 1, "output_tokens": 50}}, *verdicts[1:]])),
              "stored billed tokens the result doesn't bear out")  # fmt: skip
        check(raises(collect_sonnet_gate.SonnetCollectError, lambda: build(records=[{**verdicts[0], "result": {"type": "errored"}}, *verdicts[1:]])),
              "a stored result that is not a success")  # fmt: skip
        check(raises(collect_sonnet_gate.SonnetCollectError, lambda: build(records=[{**verdicts[0], "custom_id": "other_r0_sg"}, *verdicts[1:]])),
              "a verdict stored under a request ID that is not its own")  # fmt: skip
        check(raises((KeyError, ValueError), lambda: sonnet_gate.parse_response({"stop_reason": "end_turn", "content": []})),
              "a response with no billed tokens is no response")  # fmt: skip
        for bad in (1.2, -0.1, float("nan"), True):
            check(raises(GateTableError, bad_score(free, calls, answers, client, bad, 0.0)),
                  f"a gate score that isn't a number from 0 to 1 ({bad!r})")  # fmt: skip
        for bad in (-1e-9, float("nan"), float("inf")):
            check(raises(GateTableError, bad_score(free, calls, answers, client, 0.5, bad)),
                  f"an overhead that is negative or not finite ({bad!r})")  # fmt: skip
        check(raises(GateTableError, lambda: gates_table(free, calls, answers, client, None, 0.0)),
              "a missing score on a call not marked refused")  # fmt: skip
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder)
            (copy / "gates").mkdir()
            path = copy / "gates" / "sonnet_gate_test.jsonl"
            lines = (
                (r.world.results / "gates" / "sonnet_gate_test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines(keepends=True)
            )
            path.write_text("".join(lines) + lines[0], encoding="utf-8")
            check(
                raises(GateCollectError, lambda: final_gates.read_sonnet_test(copy)), "a verdict repeated in the file"
            )
            moved = {**json.loads(lines[0]), "split": "dev"}
            path.write_text(json.dumps(moved) + "\n", encoding="utf-8")
            check(
                raises(GateCollectError, lambda: final_gates.read_sonnet_test(copy)),
                "a record labelled as another split's",
            )
            path.write_text("".join(lines[:3]) + '{"item_id": "torn', encoding="utf-8")
            check(len(final_gates.read_sonnet_test(copy)) == 3,
                  "known positive: a torn last line is a write cut short and is dropped")  # fmt: skip
        refused = {
            **verdicts[0],
            "result": {"type": "succeeded", "message": rehearsal_sonnet.reply(None, "REFUSE", "claude-sonnet-5-5")},
            "refused": True,
            "stop_reason": "refusal",
            "usage": {"input_tokens": 1000, "output_tokens": 50},
        }
        entry = build(records=[refused, *verdicts[1:]]).row(first[0], first[1]).sonnet
        check(entry is not None and entry.score is None and entry.refused and close(entry.overhead, LIST_CHECK, 1e-12),
              "known positive: a refused verdict is a rejection with no score, charged its tokens, and the table is built")  # fmt: skip


def bad_score(
    free: list[dict], calls: list[dict], answers: list, client: Any, score: Any, overhead: Any
) -> Callable[[], object]:
    return lambda: gates_table(free, calls, answers, client, score, overhead)


def gates_table(
    free: list[dict], calls: list[dict], answers: list, client: Any, score: Any, overhead: float
) -> GateScoreTable:
    """The table with one Sonnet entry replaced by a bad score or overhead (and the other gates' columns as they are)."""
    verdicts = {k: PaidScore(0.5, 0.0, False) for k in KEYS}
    verdicts[KEYS[0]] = PaidScore(score, overhead, False)
    return final_gates.build_test_table(
        free, calls, answers, gates.SETTINGS, client, extra_paid={SONNET_GATE: verdicts}
    )


# ---- the pure cascade at its frozen points --------------------------------------------------------------------------

TASK_ITEMS = gates.TASK_ITEMS
# The file's dials: haiku/Sonnet routes at two points only, the 97% target under each mix (the plan's A17 item 29: cutoffs 0.5
# and 0.3); the other four fall back to A.
HAND_CUTOFFS = {"realistic": {"97%": 0.5}, "uniform": {"97%": 0.3}}
FELL_BACK = {"realistic": ("100%", "99%"), "uniform": ("100%", "99%")}


def hand_cascade(
    cutoff: float, mix: dict[str, float], scores: Mapping[int, float | None], overhead: Mapping[int, float]
) -> tuple[float, float, float]:
    """haiku/Sonnet at `cutoff` by hand: quality, dollars per 1,000 requests and the share escalated, under the mix. An answer is
    kept if it passed the free checks and the gate scored it at least `cutoff` (a refused call, no score, is below every cutoff);
    otherwise `opus` answers too. A call costs the start's price, the gate's list-price tokens if the free checks passed
    (`overhead`, in millionths of a dollar), and `opus`'s price if the answer is not kept."""
    quality = cost = escalated = 0.0
    for task, positions in TASK_ITEMS.items():
        column = 1 if task == "mbppplus" else 0
        for position in positions:
            passed = (IDS[position], "haiku") not in STOP_FAILS
            score = scores[position]
            kept = passed and score is not None and score >= cutoff
            micro = (
                ci.MICRO["haiku"][column]
                + (overhead[position] if passed else 0)
                + (0 if kept else ci.MICRO["opus"][column])
            )
            right = ci.SUCCESS[position][0] if kept else ci.SUCCESS[position][3]
            weight = mix[task] / len(positions)
            quality += weight * right
            cost += weight * micro / 1000
            escalated += 0 if kept else weight
    return quality, cost, escalated


def cascades() -> None:
    print("\n-- haiku/Sonnet at its frozen points, worked by hand --")
    with ci.rehearsed() as r:
        outcome, _, _ = r.go()
        check(outcome == Outcome(), f"the whole rehearsal runs, the tables included {outcome}")
        rows = r.tables["parts"][final_gates.SONNET_SECTION]["mixes"]
        scores = {p: s / 100 for p, s in enumerate(rehearsal_sonnet.TEST_SCORES)}
        overhead = {p: LIST_CHECK * 1e6 for p in range(8)}
        for mix, weights in ci.MIXES.items():
            a_quality, a_cost = ci.HAND_QUALITY[mix]["opus"], ci.HAND_COST[mix]["opus"]
            for target in FELL_BACK[mix]:
                row = rows[mix][target]
                check(row["fell_back"] is True and row["threshold"] is None and row["cutoff"] is None
                      and close(row["quality"]["value"], a_quality) and close(row["cost_per_1k"]["value"], a_cost)
                      and row["escalation_rate"] == 0 and row["savings_vs_A"]["share"] == 0,
                      f"haiku/Sonnet {mix} {target}: the file says it fell back, so it is A: {a_quality} at ${a_cost}")  # fmt: skip
            for target, cutoff in HAND_CUTOFFS[mix].items():
                row = rows[mix][target]
                quality, cost, escalated = hand_cascade(cutoff, weights, scores, overhead)
                check(row["fell_back"] is False and close(row["cutoff"], cutoff) and row["threshold"] is None
                      and close(row["quality"]["value"], quality) and close(row["cost_per_1k"]["value"], cost)
                      and close(row["escalation_rate"], escalated) and close(row["savings_vs_A"]["per_1k"], a_cost - cost),
                      f"haiku/Sonnet {mix} {target} at cutoff {cutoff}: quality {quality:.4f}, ${cost:.4f} per 1k, escalated {escalated:.3f} by hand")  # fmt: skip
        check(set(r.tables["parts"][final_gates.tasks_key(final_gates.SONNET_SECTION)]["tasks"]) == set(TASK_ITEMS),
              "the cascade also has a part of per-task rows")  # fmt: skip
        by_hand = json.loads((PROJECT_ROOT / "routers" / "operating_points.json").read_text(encoding="utf-8"))[
            final_gates.SONNET_SECTION
        ]
        routed = {(m, t) for m, by in by_hand.items() for t, p in by.items() if not p["fell_back"]}
        check(routed == {("realistic", "97%"), ("uniform", "97%")} and by_hand["realistic"]["97%"]["cutoff"] == 0.5
              and by_hand["uniform"]["97%"]["cutoff"] == 0.3,
              "the operating points file, read here directly, routes haiku/Sonnet at those two points and no others")  # fmt: skip


def projection() -> None:
    print("\n-- the whole run's projection holds the Sonnet gate's share --")
    from analysis.check_final_ledger import jev_gate_projection

    with (
        ci.rehearsed() as r,
        mock.patch.object(
            final_ledger, "PROJECTORS", tuple(p for p in final_ledger.PROJECTORS if p.name != "jev_router")
        ),
    ):  # the Jev routers' projection is #84's
        r.go(to("liveness"))
        spend = detail(r, "liveness")["spend"]
        check(close(spend[CALLS]["projected"], jev_gate_projection() + hand_projection(), 1e-12),
              f"at the liveness step the calls pool's projection is the Jev gate's plus the Sonnet gate's by hand: "
              f"${spend[CALLS]['projected']:.6f}")  # fmt: skip
    check(any(p.name == "sonnet_gate" and p.entry == "analysis.final_sonnet:project" and p.pool == CALLS
              for p in final_ledger.PROJECTORS),
          "the ledger's sonnet_gate line is this module's projector, in the calls pool")  # fmt: skip
    check([s.entry for s in STEPS if s.name in ("drift_sonnet_gate", "calls_sonnet_gate")]
          == ["analysis.final_sonnet:drift_step", "analysis.final_sonnet:calls_step"],
          "and the steps' lines are this module's two")  # fmt: skip


def main() -> int:
    dev_numbers()
    what_the_gate_sees()
    drift_quiet()
    drift_refusals()
    calls_default()
    safeguards()
    rejections()
    wrong_model()
    table_checks()
    cascades()
    projection()
    total = failures()
    print("\nall checks passed" if not total else f"\n{total} checks failed")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
