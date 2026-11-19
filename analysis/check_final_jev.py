"""Check G's and G+'s drift check and test calls (PLAN_v4.md Amendment A15, item 17; A19, items 2, 5, 9, 10 and 13; issue #77, #84).

Run from the project root:  uv run python -m analysis.check_final_jev

On a made-up world, a fake Jev server on a loopback port and the made-up Stage 6 passes it stored (`analysis/rehearsal_jev.py`):
no results/ folder, no Docker, no Jev or Anthropic call, no key, no test item and nothing near the real store. Expected values
are worked by hand below, from binary fractions so every float is exact.

- the drift resends reach the fake as the frozen code's bodies (bytes and hash), into the drift files, with Stage 7's train
  files left as they were;
- the rule's three refusals, each with its own scenario, stop the run before any test call: a model other than `jev-1.14.0`,
  D8 above 3 x D6, and a uniform move that leaves D8 inside 3 x D6 and must still refuse;
- inside the limits, D6, D8, the shifts and the flips at each frozen threshold are in the record and the log, and the run goes on;
- a projection over the $25 cap refuses before the first call, and under it the same run goes on; and a call billed far above its
  projection still stops the run at the cap before the next call (the ledger is asked before each call);
- a Stage 8 split (`test`, `drift`) is written by the collector only with a handle the unlock issued: a plain `"test"`, a handle
  built by hand and the old `splits=` argument are each refused, and an issued handle is accepted (the known positive);
- a fake that fails mid-run stops the run naming the item and the exception, and the next go sends only the requests with no
  stored response;
- the first test call's server `Date` is in the record, and not the drift resends';
- the calls' record says only counts: no refusal or truncation count before the tables.
"""

import functools
import json
import sys
from datetime import datetime, timezone
from typing import Any
from unittest import mock

from analysis import check_final_run as rehearsal
from analysis import collect_jev, final_jev, final_ledger
from analysis import rehearsal_jev as jev_world
from analysis import rehearsal_world as made_up
from analysis.check_final_ledger import write_lines
from analysis.checking import refused
from analysis.drift_rule import JEV_PIN
from analysis.fake_jev import Reply
from analysis.final_record import Record
from analysis.final_run import STEPS, Outcome, Run, Step
from analysis.rehearsal_jev import G_PLUS_ROUTER, G_ROUTER, g_body, plus_body
from analysis.router_names import G_PLUS
from analysis.unlock import Handle, HandleError, unlock
from routers import jev_call

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return abs(a - b) <= tolerance


NAMED = {step.name: step for step in STEPS}
UNLOCK, DRIFT, CALLS = NAMED["unlock"], NAMED["drift_jev_router"], NAMED["calls_jev_router"]

# ---- the made-up Stage 6 and the resends, by hand -----------------------------------------------------------------------
# Four train items are the noise sample. G's router scores are the Noul answers unchanged, so each is what was scripted.
# Pass 1:  haiku .875 .875 .75 .75     sonnet .25 .25 .25 .25     opus_low .125 each
# Pass 2:  haiku .9375 .8125 .8125 .6875 (moves +.0625 -.0625 +.0625 -.0625: D6 .0625, shift 0)
#          sonnet .375 .25 .25 .25 (one move of +.125: D6 .03125, shift +.03125)       opus_low unchanged: D6 0, shift 0
FIRST = {"haiku": [0.875, 0.875, 0.75, 0.75], "sonnet": [0.25] * 4, "opus_low": [0.125] * 4}
SECOND = {"haiku": [0.9375, 0.8125, 0.8125, 0.6875], "sonnet": [0.375, 0.25, 0.25, 0.25], "opus_low": [0.125] * 4}
# The resend inside the limits: haiku moves +.0625 +.0625 -.0625 -.0625 (D8 .0625, shift 0), sonnet -.03125 +.03125 0 0
# (D8 .015625, shift 0), opus_low unchanged (D8 0, shift 0): inside 3 x D6 (.1875, .09375, 0) and each shift inside D6.
RESEND = {"haiku": [0.9375, 0.9375, 0.6875, 0.6875], "sonnet": [0.21875, 0.28125, 0.25, 0.25], "opus_low": [0.125] * 4}
D6 = {"haiku": 0.0625, "sonnet": 0.03125, "opus_low": 0.0}
GOOD_MEASURES = {
    "haiku": {"d6": 0.0625, "d8": 0.0625, "shift": 0.0},
    "sonnet": {"d6": 0.03125, "d8": 0.015625, "shift": 0.0},
    "opus_low": {"d6": 0.0, "d8": 0.0, "shift": 0.0},
}
# F's table of frozen thresholds is G's (`routers/operating_points.json`, calls.md): realistic 0.93, 0.88 and 0.76, uniform
# 0.93, 0.85 and 0.76. A decision is the first config whose score reaches the threshold, else opus, so:
#   threshold .93: items 0 and 1 go from opus (haiku .875) to haiku (.9375); items 2 and 3 are opus both times: 2 flips
#   threshold .88: the same two items (.9375 reaches it, .875 did not): 2 flips
#   threshold .85: haiku .875 reaches it before and .9375 after: none flip; threshold .76: .75 misses it and .6875 too: none
FLIPS = {0.93: 2, 0.88: 2, 0.85: 0, 0.76: 0}
FLIPPED = ["banking77:made-train-0", "banking77:made-train-1"]
LABELS = {
    0.93: ["realistic 100%", "uniform 100%"],
    0.88: ["realistic 99%"],
    0.85: ["uniform 99%"],
    0.76: ["realistic 97%", "uniform 97%"],
}


def g_bodies(scores: dict[str, list[float]], *, model: str = JEV_PIN) -> list[dict]:
    return [g_body(scores["haiku"][k], scores["sonnet"][k], scores["opus_low"][k], model=model) for k in range(4)]


def passes() -> dict[str, Any]:
    plus = [plus_body() for _ in range(4)]
    return {
        "first": {G_ROUTER: g_bodies(FIRST), G_PLUS_ROUTER: plus},
        "second": {G_ROUTER: g_bodies(SECOND), G_PLUS_ROUTER: plus},
    }


def script(drift_g: list[dict], *later: Reply) -> list[Reply]:
    """The fake's replies in call order: G's four resends, G+'s four unchanged, then what follows (test's calls)."""
    return [*(Reply(200, body) for body in drift_g), *(Reply(200, plus_body()) for _ in range(4)), *later]


def world_with_four():
    return rehearsal.rehearsed(made_up.content(train_items=4), **passes())


def go(r: rehearsal.Rehearsal, steps: tuple[Step, ...]) -> tuple[Outcome, str]:
    outcome, out, err = r.go(steps)
    return outcome, out + err


def files(r: rehearsal.Rehearsal) -> set[str]:
    return {p.name for p in (r.world.results / "jev").glob("*.jsonl")}


def bodies_of(r: rehearsal.Rehearsal, router: str, items: list[collect_jev.Item]) -> list[bytes]:
    """What the frozen call building gives for each item: the bytes the fake should have received."""
    loaded = r.jev.loaded
    build = jev_call.g_call if router == G_ROUTER else jev_call.g_plus_call
    return [build(item.request, loaded.call, budget_bytes=jev_world.BUDGETS[router]).body for item in items]


# ---- the drift resends ---------------------------------------------------------------------------------------------------


def resends() -> None:
    print("-- the drift resends: the frozen bodies, stored apart from Stage 7's records --")
    with world_with_four() as r:
        train_dir = r.jev.stand_in.train_results / "jev"
        train_before = {p.name: p.read_bytes() for p in train_dir.glob("*.jsonl")}
        noise = r.jev.items(r.world)
        r.jev.fake.script = script(g_bodies(RESEND))
        outcome, out = go(r, (UNLOCK, DRIFT))
        check(outcome == Outcome(), f"the unlock and the drift step run clean {outcome}")
        want = bodies_of(r, G_ROUTER, noise) + bodies_of(r, G_PLUS_ROUTER, noise)
        got = [received.body for received in r.jev.fake.received]
        check(got == want and len(got) == 8,
              "the eight resends reached the fake as the frozen code's bodies, G's four then G+'s four, byte for byte")  # fmt: skip
        stored = collect_jev.read_records(G_ROUTER, "drift", r.world.results, collect_jev.STAGE_8_SPLITS)
        plus_stored = collect_jev.read_records(G_PLUS_ROUTER, "drift", r.world.results, collect_jev.STAGE_8_SPLITS)
        check([s["body_sha256"] for s in stored + plus_stored] == [jev_call.body_sha256(b) for b in want],
              "and each stored record's body hash is the hash of what the frozen code builds")  # fmt: skip
        check(files(r) == {"jev_drift.jsonl", "jev_bolstered_drift.jsonl"} and set(train_before) == {"jev_train.jsonl", "jev_bolstered_train.jsonl"}
              and {s["split"] for s in stored} == {"drift"} and {s["pass"] for s in stored} == {1},
              "they are stored in the drift files, labelled drift, under names of their own, and no test file appeared")  # fmt: skip
        check(all((train_dir / name).read_bytes() == data for name, data in train_before.items()),
              "Stage 6's train records are byte for byte what they were")  # fmt: skip
        check(len(collect_jev.read_records(G_ROUTER, "train", r.jev.stand_in.train_results)) == 8,
              "and still read as two passes of four items through the collection module's own reader")  # fmt: skip
        check("test" not in collect_jev.SPLITS and "drift" not in collect_jev.SPLITS,
              "a plain test or drift split is still refused by every Stage 7 caller: neither is in SPLITS")  # fmt: skip
        spend = final_ledger.recorded_spend(r.world.results, r.world.output)
        check(close(spend.calls, 8 * 300 * 0.042 / 1e6) and spend.files == 2,
              f"both drift files count in the $25 pool: 8 calls of 300 tokens at $0.042 per million, ${spend.calls:.7f}")  # fmt: skip
        check(not r.world.results.joinpath("jev", "jev_test.jsonl").exists(), "no test call was made by the drift step")
    return None


# ---- the three refusals --------------------------------------------------------------------------------------------------


def refusals() -> None:
    print("\n-- each refusal stops the run before any test call --")
    wrong_id = g_bodies(RESEND, model="jev-1.15.0")
    big = {**RESEND, "haiku": [0.625, 0.625, 1.0, 1.0]}  # moves of -.25 -.25 +.25 +.25: D8 .25, shift 0
    uniform = {**RESEND, "haiku": [h + 0.125 for h in FIRST["haiku"]]}  # every haiku score +0.125: D8 .125, shift +.125
    for name, bodies, words, absent in (
        ("a returned model ID other than jev-1.14.0", wrong_id, ["jev-1.15.0", "not the pin jev-1.14.0"], ["D8"]),
        ("D8 above 3 x D6: haiku moves .25 against a D6 of .0625 (.25 > .1875)", g_bodies(big), ["haiku", "D8 0.25"], ["shift"]),
        ("the shift rule's known positive, a uniform +.125 on haiku: D8 .125 is inside 3 x D6 (.1875) and only the shift, "
         "+.125 against a D6 of .0625, refuses", g_bodies(uniform), ["haiku", "shift (mean signed difference) +0.125"], ["D8"]),
    ):  # fmt: skip
        with world_with_four() as r:
            noise = r.jev.items(r.world)
            r.jev.fake.script = script(bodies)
            outcome, out = go(r, (UNLOCK, DRIFT, CALLS))
            check(
                outcome == Outcome("drift_jev_router", "DriftRefusal", None),
                f"{name}: the run stops at the drift step {outcome}",
            )
            check(all(w in out for w in words) and not any(a in out.split("refused:")[-1] for a in absent),
                  f"it says why in the log, naming {words}, and not {absent}")  # fmt: skip
            sent = [received.body for received in r.jev.fake.received]
            check(sent == bodies_of(r, G_ROUTER, noise),
                  "the fake received G's four resends and nothing else: no G+ resend and no test call")  # fmt: skip
            check("jev_test.jsonl" not in files(r) and "jev_bolstered_test.jsonl" not in files(r),
                  "no test record file exists")  # fmt: skip
            check(not any(s["name"] == "calls_jev_router" for s in r.record["gos"][0]["steps"]),
                  "the calls step never started")  # fmt: skip
            # A new go finds the resends stored, sends none of them again, and refuses again from the stored records.
            r.jev.fake.received.clear()
            outcome, _ = go(r, (UNLOCK, DRIFT, CALLS))
            check(outcome.error == "DriftRefusal" and r.jev.fake.received == [],
                  "a resume asks for no resend again and refuses again, from the stored records")  # fmt: skip


# ---- inside the limits ---------------------------------------------------------------------------------------------------


def goes_on() -> None:
    print("\n-- inside the limits: D6, D8, the shifts and the flips are reported, and the run goes on --")
    with world_with_four() as r:
        r.jev.fake.script = script(g_bodies(RESEND))
        outcome, out = go(r, (UNLOCK, DRIFT, CALLS))
        check(outcome == Outcome(), f"the run goes through the drift step and on to the calls {outcome}")
        drift = next(s for s in r.record["gos"][0]["steps"] if s["name"] == "drift_jev_router")
        g = drift["detail"]["routers"]["G"]
        check(
            g["measures"] == GOOD_MEASURES, f"G's D6, D8 and shift per config are the hand-worked ones {g['measures']}"
        )
        check(g["pin"] == JEV_PIN and g["model_ids"] == [JEV_PIN] and g["d8_factor"] == 3.0,
              "the record holds the pin, the model ID returned and the factor")  # fmt: skip
        check({f["threshold"]: f["count"] for f in g["flips"]} == FLIPS
              and {f["threshold"]: f["points"] for f in g["flips"]} == LABELS
              and all(f["items"] == (FLIPPED if f["count"] else []) for f in g["flips"]),
              "the decisions that flip at each frozen threshold are counted, with the points that sit on it")  # fmt: skip
        plus = drift["detail"]["routers"][G_PLUS]
        check(all(m == {"d6": 0.0, "d8": 0.0, "shift": 0.0} for m in plus["measures"].values()) and plus["resent"] == 4,
              "G+'s four resends are the unchanged answers, and its measures are all zero (zero goes on)")  # fmt: skip
        check(drift["requests"] == {"sent": 8, "stored": 0}, "the step sent eight resends and none were stored")
        check("drift_jev_router: G on jev-1.14.0: haiku D6 0.0625 D8 0.0625 shift +0.0000" in out
              and "decisions that flip by threshold {0.93: 2, 0.88: 2, 0.85: 0, 0.76: 0}" in out,
              "the log says D6, D8, the shift and the flips (train numbers only)")  # fmt: skip
        calls = next(s for s in r.record["gos"][0]["steps"] if s["name"] == "calls_jev_router")
        check(calls["status"] == "done" and calls["requests"] == {"sent": 16, "stored": 0},
              "and the calls step sent 16 requests: eight test items through each router")  # fmt: skip
        for router in (G_ROUTER, G_PLUS_ROUTER):
            test = collect_jev.read_records(router, "test", r.world.results, collect_jev.STAGE_8_SPLITS)
            check([t["item_id"] for t in test] == made_up.TEST_IDS and {t["split"] for t in test} == {"test"},
                  f"{router}'s test records are the eight test items in the handle's order, in the test file")  # fmt: skip
        test_bodies = [received.body for received in r.jev.fake.received[8:]]
        requests = rehearsal_requests(r)
        want = [
            jev_call.g_call(requests[i], r.jev.loaded.call, budget_bytes=jev_world.BUDGETS[G_ROUTER]).body
            for i in made_up.TEST_IDS
        ]
        want += [jev_call.g_plus_call(requests[i], r.jev.loaded.call, budget_bytes=jev_world.BUDGETS[G_PLUS_ROUTER]).body
                 for i in made_up.TEST_IDS]  # fmt: skip
        check(test_bodies == want, "the test calls reached the fake as the frozen code's bodies, each request once")
        check(len(r.jev.fake.received) == 24, "24 calls in all: eight resends and sixteen test calls")


def rehearsal_requests(r: rehearsal.Rehearsal) -> dict:
    """The made-up test requests by item ID, as the handle gives them (`from_row` of the made-up rows)."""
    from routers.request import from_row

    return {row["item_id"]: from_row(row) for row in made_up.ITEMS}


# ---- the cap -------------------------------------------------------------------------------------------------------------


def cap() -> None:
    print("\n-- a projection over the cap refuses before the first call --")
    for recorded, refuses in ((2_500_001, True), (2_490_000, False)):
        with world_with_four() as r:
            # A Stage 8 Sonnet drift record at list price: output tokens at $10 per million: $25.00001 or $24.90.
            write_lines(r.world.results / "gates", "sonnet_gate_drift.jsonl", [(0, recorded)])
            r.jev.fake.script = script(g_bodies(RESEND))
            outcome, out = go(r, (UNLOCK, DRIFT, CALLS))
            if refuses:
                check(outcome == Outcome("drift_jev_router", "LedgerRefusal", None) and r.jev.fake.received == [],
                      f"$25.00001 already recorded: refused before the first call, the fake's log empty {outcome}")  # fmt: skip
                check(
                    "over Stage 8's cap" in out and "nothing was sent" in out, "and the refusal is shown, with its cap"
                )
            else:
                check(outcome == Outcome() and len(r.jev.fake.received) == 24,
                      "$24.90 recorded and about a cent projected: the same run goes on")  # fmt: skip
                detail = next(s for s in r.record["gos"][0]["steps"] if s["name"] == "calls_jev_router")["detail"][
                    "spend"
                ]
                sent = [received.body for received in r.jev.fake.received]
                # The calls step's projection, by hand: each of the 16 test calls' body bytes plus the fixed 512 tokens, at
                # $0.042 per million (the drift resends were already stored, so none of them is projected).
                expected = sum(len(body) + 512 for body in sent[8:]) * 0.042 / 1e6
                check(close(detail[final_ledger.CALLS]["projected"], expected, 1e-9) and 0 < expected < 0.1,
                      f"the calls step's projection is the hand-worked ${expected:.6f}: body bytes plus 512 tokens each")  # fmt: skip
                check(close(detail[final_ledger.CALLS]["recorded"], 24.9 + 8 * 300 * 0.042 / 1e6, 1e-9),
                      "and the recorded spend it adds to is the drift record's $24.90 and the eight resends")  # fmt: skip


def metered_cap() -> None:
    print("\n-- a call billed far above its projection still stops at the cap, before the next call --")
    with world_with_four() as r:
        r.jev.fake.script = script(g_bodies(RESEND))
        outcome, _ = go(r, (UNLOCK, DRIFT))
        check(outcome == Outcome(), "the drift step goes first, as in the runs above")
        r.jev.fake.input_tokens = (
            5_000_000  # each test call is billed 5 million input tokens: $0.21 at $0.042 per million
        )
        recorded = final_ledger.recorded_spend(r.world.results, r.world.output).calls
        sent_before = len(r.jev.fake.received)
        with mock.patch.dict(final_ledger.CAPS, {final_ledger.CALLS: recorded + 0.05}):
            outcome, out = go(r, (UNLOCK, DRIFT, CALLS))
        stored = collect_jev.read_records(G_ROUTER, "test", r.world.results, collect_jev.STAGE_8_SPLITS)
        check(outcome.stopped_at == "calls_jev_router" and outcome.error == "LedgerRefusal",
              f"the whole projection fits under $0.05, and the run still stops at the calls step {outcome}")  # fmt: skip
        check(len(stored) == 1 and len(r.jev.fake.received) == sent_before + 1,
              "after one call: it was stored, and the next was refused before it was sent")  # fmt: skip
        check("over Stage 8's cap" in out, "and the refusal names the cap")


def handle_required() -> None:
    print("\n-- a Stage 8 split is written only with a handle the unlock issued --")
    with world_with_four() as r:
        setup = final_jev.setup_of(r.world)
        base: dict[str, Any] = {
            "router": G_ROUTER,
            "loaded": setup.loaded,
            "client": None,
            "results_dir": r.world.results,
        }
        none: list[collect_jev.Item] = []  # no item to send, so nothing is called, whatever the answer
        for split in ("test", "drift"):
            check(refused(collect_jev.CollectError, functools.partial(collect_jev.collect, none, split=split, **base)),
                  f"a plain {split!r} split with no handle is refused")  # fmt: skip
        check(refused(TypeError, lambda: collect_jev.collect(none, split="test", splits=collect_jev.STAGE_8_SPLITS, **base)),  # type: ignore[call-arg]
              "the old `splits=` argument is gone: a caller can't name its own labels")  # fmt: skip
        issued = unlock(r.world)
        forged = Handle(r.world, issued.results)
        check(refused(HandleError, lambda: collect_jev.collect(none, split="test", handle=forged, **base)),
              "a handle built by hand is refused, whatever split it names")  # fmt: skip
        check(refused(HandleError, lambda: collect_jev.collect(none, split="train", handle=forged, **base)),
              "even for a Stage 7 split: a handle that is given must be an issued one")  # fmt: skip
        summary = collect_jev.collect(none, split="test", handle=issued, **base)
        check(summary.new == 0 and refused(collect_jev.CollectError, lambda: collect_jev.collect(none, split="test", **base)),
              "known positive: the issued handle is accepted for the same split the plain call was refused for")  # fmt: skip
        check(refused(collect_jev.CollectError, lambda: collect_jev.collect(none, split="nowhere", handle=issued, **base)),
              "and an issued handle still refuses a label that is nobody's")  # fmt: skip


# ---- a failure mid-run, and the resume -----------------------------------------------------------------------------------


def failure_and_resume() -> None:
    print(
        "\n-- a fake that fails mid-run: the item and the exception named, and a resume sends only what is not stored --"
    )
    with world_with_four() as r:
        ok = [Reply(200, g_body(0.5, 0.5, 0.5)) for _ in range(3)]
        r.jev.fake.script = script(g_bodies(RESEND), *ok, Reply(400, {"detail": "no"}))
        outcome, out = go(r, (UNLOCK, DRIFT, CALLS))
        failed = "mmlu_pro:made-test-1"  # the fourth test item in the benchmark's order
        check(outcome == Outcome("calls_jev_router", "Rejected", failed),
              f"a 400 on the fourth test call stops the run, naming the item and the exception's class {outcome}")  # fmt: skip
        check(failed in out and "Rejected" in out and "no" not in out.split("Rejected")[-1][:20],
              "the log names the item and the class")  # fmt: skip
        stored = collect_jev.read_records(G_ROUTER, "test", r.world.results, collect_jev.STAGE_8_SPLITS)
        check(
            [s["item_id"] for s in stored] == made_up.TEST_IDS[:3],
            "the three calls before it are stored, the failed one is not",
        )
        check("jev_bolstered_test.jsonl" not in files(r), "G+ had made no test call")
        before = len(r.jev.fake.received)
        r.jev.fake.script = []
        outcome, _ = go(r, (UNLOCK, DRIFT, CALLS))
        check(outcome == Outcome(), f"the next go runs clean {outcome}")
        resent = [received.body for received in r.jev.fake.received[before:]]
        requests = rehearsal_requests(r)
        loaded = r.jev.loaded
        g_rest = [jev_call.g_call(requests[i], loaded.call, budget_bytes=jev_world.BUDGETS[G_ROUTER]).body
                  for i in made_up.TEST_IDS[3:]]  # fmt: skip
        plus_all = [jev_call.g_plus_call(requests[i], loaded.call, budget_bytes=jev_world.BUDGETS[G_PLUS_ROUTER]).body
                    for i in made_up.TEST_IDS]  # fmt: skip
        check(resent == g_rest + plus_all and len(resent) == 13,
              "the fake's log shows 13 calls: G's five unstored requests and G+'s eight, and none of the three stored")  # fmt: skip
        calls = next(s for s in r.record["gos"][1]["steps"] if s["name"] == "calls_jev_router")
        drift = next(s for s in r.record["gos"][1]["steps"] if s["name"] == "drift_jev_router")
        check(calls["requests"] == {"sent": 13, "stored": 3} and drift["requests"] == {"sent": 0, "stored": 8},
              "and the record says so: 13 sent and 3 stored, and none of the eight resends sent again")  # fmt: skip
        full = collect_jev.read_records(G_ROUTER, "test", r.world.results, collect_jev.STAGE_8_SPLITS)
        check([s["item_id"] for s in full] == made_up.TEST_IDS and full[:3] == stored,
              "the stored responses are untouched and complete the eight")  # fmt: skip


# ---- the Date ------------------------------------------------------------------------------------------------------------

D_DRIFT, D_TEST = "Wed, 18 Nov 2026 08:30:00 GMT", "Thu, 19 Nov 2026 10:00:00 GMT"


def first_date() -> None:
    print("\n-- the first test Jev call's Date is in the record --")
    with world_with_four() as r:
        r.jev.fake.date = D_DRIFT
        r.jev.fake.script = script(g_bodies(RESEND))
        outcome, _ = go(r, (UNLOCK, DRIFT))
        check(
            outcome == Outcome(), "a first go runs the drift check, with the fake's Date a day before the test calls'"
        )
        r.jev.fake.date = D_TEST
        outcome, out = go(r, (UNLOCK, DRIFT, CALLS))
        calls = next(s for s in r.record["gos"][1]["steps"] if s["name"] == "calls_jev_router")
        check(outcome == Outcome() and calls["detail"]["first_test_jev_date"] == D_TEST,
              f"the calls step's record holds the first test call's Date header, not the resends' {calls['detail']['first_test_jev_date']}")  # fmt: skip
        parsed = datetime(2026, 11, 19, 10, 0, 0, tzinfo=timezone.utc).isoformat()
        check(calls["detail"]["first_test_jev_date_parsed"] == parsed and calls["detail"]["first_test_jev_router"] == "G",
              "with its parsed time and the router whose call it was")  # fmt: skip
        check(D_DRIFT not in json.dumps(calls["detail"]), "and the drift's own Date is nowhere in it")
        records = collect_jev.read_records(G_ROUTER, "drift", r.world.results, collect_jev.STAGE_8_SPLITS)
        check({x["server_date"] for x in records} == {D_DRIFT}, "the drift records keep their own")


# ---- what the calls say --------------------------------------------------------------------------------------------------


def says_counts_only() -> None:
    print("\n-- the calls step says counts, not refusals or truncations (A19 item 5) --")
    with world_with_four() as r:
        refused = {
            "answers": {"success_haiku": {"type": "noul", "noul": 0.5}},
            "model": JEV_PIN,
            "usage": {"input_tokens": 9, "output_tokens": 1},
        }
        r.jev.fake.script = script(g_bodies(RESEND), Reply(200, refused))
        outcome, out = go(r, (UNLOCK, DRIFT, CALLS))
        calls = next(s for s in r.record["gos"][0]["steps"] if s["name"] == "calls_jev_router")
        check(outcome == Outcome(), "a response that leaves a question unanswered is stored, not a stop")
        text = out + json.dumps(calls)
        check("refus" not in text.lower() and "cut" not in text.lower() and "truncat" not in text.lower(),
              "neither the log nor the step's record says a refusal or a truncation: those go with the tables")  # fmt: skip
        test = collect_jev.read_records(G_ROUTER, "test", r.world.results, collect_jev.STAGE_8_SPLITS)
        check(test[0]["refused"] is True and [t["refused"] for t in test[1:]] == [False] * 7,
              "the stored record does hold the flag, from the response, for the first test item alone")  # fmt: skip


# ---- the tables' side ----------------------------------------------------------------------------------------------------


def unlocked_run(r: rehearsal.Rehearsal) -> Run:
    """A runner's `Run` with the handle the unlock issued, for reading what the tables step reads."""
    run = Run(r.world, Record(r.world.output / "scratch_record.json"), rehearsal.clock())
    run.handle = unlock(r.world)
    return run


def tables_side() -> None:
    print("\n-- the test tables from the stored responses, and the counts reported with them --")
    # G's budget of 40 bytes cuts every made-up prompt longer than that: the two Banking77, two MMLU-Pro and two IFEval prompts
    # (the template prompts are far longer, IFEval's is 47 bytes) and not MBPP+'s two (28 bytes): six cuts. G+'s budget is large.
    budgets = {G_ROUTER: 40, G_PLUS_ROUTER: 20_000}
    longer = [row["item_id"] for row in made_up.ITEMS if len(row["prompt"].encode()) > 40]
    check(len(longer) == 6 and not any(i.startswith("mbppplus") for i in longer),
          "by hand, six of the eight made-up prompts are longer than 40 bytes, MBPP+'s two are not")  # fmt: skip
    with rehearsal.rehearsed(made_up.content(train_items=4), budgets=budgets, **passes()) as r:
        refused = {"answers": {"success_haiku": {"type": "noul", "noul": 0.5}}, "model": JEV_PIN,
                   "usage": {"input_tokens": 9, "output_tokens": 1}}  # fmt: skip
        r.jev.fake.script = script(g_bodies(RESEND), Reply(200, refused))
        outcome, _ = go(r, (UNLOCK, DRIFT, CALLS))
        check(outcome == Outcome(), f"the go runs with G's budget at 40 bytes {outcome}")
        run = unlocked_run(r)
        found = final_jev.facts(run).record()
        check(found["G"] == {"calls": 8, "refused": 1, "cut": 6, "hash_mismatches": 0, "models": [JEV_PIN]},
              f"G's facts: eight calls, the one refusal, six cuts, every stored body hash the frozen code's {found['G']}")  # fmt: skip
        check(found[G_PLUS] == {"calls": 8, "refused": 0, "cut": 0, "hash_mismatches": 0, "models": [JEV_PIN]},
              "G+'s facts: eight calls, none refused or cut")  # fmt: skip
        table = final_jev.test_table(G_ROUTER)(run)
        check(
            [e.item_id for e in table.items] == made_up.TEST_IDS,
            "G's table is over the eight test items in the handle's order",
        )
        want = {"haiku": 0.91, "sonnet": 0.62, "opus_low": 0.33}
        check(table.items[0].router_scores == {"haiku": 0.0, "sonnet": 0.0, "opus_low": 0.0}
              and all(e.router_scores == want for e in table.items[1:]),
              "its router scores are the stored answers unchanged, 0 for each config on the refused response")  # fmt: skip
        check(all(close(e.overhead, 300 * 0.042 / 1e6) for e in table.items[1:]) and close(table.items[0].overhead, 9 * 0.042 / 1e6),
              "and each item's overhead is its call's input tokens at $0.042 per million (300; the refused response's 9)")  # fmt: skip
        plus = final_jev.test_table(G_PLUS_ROUTER)(run)
        schema_items = [e for e in plus.items if e.item_id.split(":")[0] in ("banking77", "mmlu_pro")]
        got = [(e.router_scores["haiku"], e.router_scores["sonnet"], e.router_scores["opus_low"]) for e in schema_items]
        # check_jev_plus works these by hand for the same answers and weights, with the schema flag set.
        check(len(got) == 4 and all(all(abs(a - b) < 1e-6 for a, b in zip(g, (0.885197, 0.509474, 0.257565), strict=True)) for g in got),
              "G+'s scores for the four items with a schema are check_jev_plus's hand-worked 0.885197, 0.509474, 0.257565")  # fmt: skip
        # A body hash that is not the frozen code's is counted: tamper with one stored hash.
        path = r.world.results / "jev" / "jev_test.jsonl"
        lines = path.read_text().splitlines()
        first = json.loads(lines[0])
        first["body_sha256"] = "0" * 64
        path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
        check(final_jev.facts(run).record()["G"]["hash_mismatches"] == 1,
              "known positive: a stored body hash that is not the frozen code's is counted")  # fmt: skip


def main() -> int:
    # Only the Jev routers' projector (and the liveness calls', which these runs leave out): the gates' projections are
    # `analysis/check_final_gates.py`'s, and here would add to the hand-worked dollars.
    ours = tuple(p for p in final_ledger.PROJECTORS if p.name in ("jev_router", "liveness"))
    with mock.patch.object(final_ledger, "PROJECTORS", ours):
        resends()
        refusals()
        goes_on()
        cap()
        metered_cap()
        handle_required()
        failure_and_resume()
        first_date()
        says_counts_only()
        tables_side()
    if problems:
        print(f"\n{len(problems)} checks failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
