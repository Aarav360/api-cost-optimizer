"""Check Stage 8's ledger, its up-front projection and the liveness calls, and that Stage 7's guards leave Stage 8's files
alone (PLAN_v4.md Amendment A19, items 2.2, 9 and 13; A17 item 18; issue #77, #82).

Run from the project root:  uv run python -m analysis.check_final_ledger

On made-up folders and fakes only: no results/ folder, no Docker, no Jev or Anthropic call, no key, no test item, and nothing
near the real store. Expected dollars are worked by hand below from the price sheet ($0.042 per million Jev input tokens,
Sonnet at $2 in and $10 out per million, batches at half, haiku $0.10 in and $0.50 out, opus $4 in and $20 out).

- the naming rule: a record file is Stage 8's when its stem ends in `test` or `drift`, and nothing else is;
- the ledger counts a made-up Stage 8 test or drift record, for Jev's routers, the Jev gate and the Sonnet gate, with
  hand-worked prices, and ignores Stage 7's train, dev and smoke records in the same folders; a cut-short last line is
  dropped, and an unreadable line, a record with no billed tokens and a Stage 8 file it has no price for refuse it whole;
- Stage 7's spend counts only Stage 7's files, so it is what it was before Stage 8's files existed (known positive: the
  old rule, every `jev_gate_*` and `sonnet_gate_*` file, gives another total);
- the two caps are $25 and $10, a projection over either refuses before the first call and one under both goes on, the
  recorded spend counts in the projection, and Stage 7's cap is still $25;
- the liveness calls reach a fake Anthropic client exactly as hand-written JSON, equal what the live replay's request
  builder builds, are never answered from a cache, and count under the $10 cap;
- a fake that returns another model ID stops the run at the liveness step, and its log shows no other call;
- the hook's "no test record" guards, run on a made-up results folder holding Stage 8's files: they fail before the
  unlock (known positive: the old guard fails on the same folder after it) and pass once the record shows one, and the
  numbers `analysis/check_final_design.py` quotes leave Stage 8's files out.
"""

import contextlib
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

from analysis import check_final_run as rehearsal
from analysis import collect_jev, final_ledger, final_liveness, gate_spend, live_calls, record_files
from analysis.backends import Backends
from analysis.check_final_design import jev_routers_dollars
from analysis.check_read_path import GO
from analysis.checking import refused
from analysis.fake_anthropic import FakeAnthropic, message_dict
from analysis.final_ledger import CALLS, LIVE, LedgerRefusal, Projector
from analysis.final_record import RECORD_FILE
from analysis.final_run import STEPS, Outcome
from harness.client import PROJECT_ROOT, load_models
from routers import jev, sonnet_gate

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(a - b) <= tolerance


def write_lines(
    folder: Path, name: str, usages: list[tuple[int, int]], *, torn: str = "", model: str | None = None
) -> None:
    """A record file whose records hold only what the spend reads: billed tokens (and a request's model, if `model`)."""
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = [{"usage": {"input_tokens": i, "output_tokens": o}} for i, o in usages]
    if model is not None:
        records = [{**r, "request": {"model": model}} for r in records]
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + torn, encoding="utf-8")


# ---- the naming rule -----------------------------------------------------------------------------------------------


def naming() -> None:
    print("-- a record file is Stage 8's when its stem ends in test or drift --")
    stage_8 = ["free_checks_test.jsonl", "jev_gate_test.jsonl", "jev_gate_drift.jsonl", "sonnet_gate_test.jsonl",
               "sonnet_gate_drift.jsonl", "jev_test.jsonl", "jev_bolstered_test.jsonl", "jev_drift.jsonl",
               "jev_bolstered_drift.jsonl"]  # fmt: skip
    stage_7 = ["free_checks_train.jsonl", "free_checks_dev.jsonl", "free_checks_smoke.jsonl", "jev_gate_train.jsonl",
               "jev_gate_dev.jsonl", "sonnet_gate_smoke.jsonl", "sonnet_gate_latency.jsonl",
               "sonnet_gate_latency_smoke.jsonl", "jev_dev.jsonl", "jev_bolstered_train.jsonl"]  # fmt: skip
    check(all(record_files.is_stage_8(name) for name in stage_8), "all nine Stage 8 file names are Stage 8's")
    check(not any(record_files.is_stage_8(name) for name in stage_7), "none of Stage 7's ten file names is")
    check(not record_files.is_stage_8("sonnet_batches.json") and not record_files.is_stage_8("contest.jsonl")
          and not record_files.is_stage_8("jev_gate_test.json"),
          "a log, a stem that only ends in the letters, and a file that isn't JSON Lines are not")  # fmt: skip
    listed = {name for names in record_files.STAGE_8_FILES.values() for name in names}
    check(listed == set(stage_8), "the ledger prices exactly those nine names")
    check(record_files.RECORD_NAME == RECORD_FILE, "the guard looks for the record file the runner writes")
    go_text = (PROJECT_ROOT / GO).read_text(encoding="utf-8")
    check(f'RESULTS / "{record_files.RECORD_FOLDER}"' in go_text,
          "and in the folder the go's entry point writes it to (read as text: nothing here imports the entry point)")  # fmt: skip


# ---- the ledger ----------------------------------------------------------------------------------------------------


def stage_8_records(results: Path) -> None:
    """Six priced Stage 8 files, worked by hand (Jev $0.042 per million input tokens; Sonnet $2 in and $10 out):
    jev_test 1,000,000 in = $0.042; jev_bolstered_drift 500,000 in = $0.021; jev_gate_test 2,000,000 in = $0.084;
    jev_gate_drift 1,000,000 in = $0.042; sonnet_gate_test (batch) 1,000,000 in and 100,000 out is $2 + $1 = $3 at list
    price, $1.50 at half; sonnet_gate_drift (standard API) 500,000 in and 200,000 out is $1 + $2 = $3. Total $4.689."""
    write_lines(results / "jev", "jev_test.jsonl", [(600_000, 61), (400_000, 61)])
    write_lines(results / "jev", "jev_bolstered_drift.jsonl", [(500_000, 61)])
    write_lines(results / "gates", "jev_gate_test.jsonl", [(2_000_000, 38)])
    write_lines(results / "gates", "jev_gate_drift.jsonl", [(1_000_000, 38)])
    write_lines(results / "gates", "sonnet_gate_test.jsonl", [(1_000_000, 100_000)])
    write_lines(results / "gates", "sonnet_gate_drift.jsonl", [(500_000, 200_000)])
    (results / "gates" / "free_checks_test.jsonl").write_text('{"usage": {"input_tokens": 9e9}}\n')  # free: not read


def stage_7_records(results: Path) -> None:
    """Stage 7's, in the same folders: Jev's routers $0.21 (5,000,000 in), the Jev gate $0.147 (3,500,000 in), the Sonnet
    batch $1.50 (1,000,000 in, 100,000 out at half price) and a latency file $3.00 (500,000 in, 200,000 out): $4.857."""
    write_lines(results / "jev", "jev_train.jsonl", [(5_000_000, 61)])
    write_lines(results / "gates", "jev_gate_train.jsonl", [(1_000_000, 40), (500_000, 40)])
    write_lines(results / "gates", "jev_gate_dev.jsonl", [(2_000_000, 40)])
    write_lines(results / "gates", "sonnet_gate_dev.jsonl", [(1_000_000, 100_000)])
    write_lines(results / "gates", "sonnet_gate_latency.jsonl", [(500_000, 200_000)])


def ledger() -> None:
    print("\n-- the ledger counts Stage 8's records and no others --")
    check(jev.read_client_settings().price_per_million_input_tokens == 0.042
          and (sonnet_gate.load_settings().input_per_million, sonnet_gate.load_settings().output_per_million) == (2.0, 10.0)
          and load_models()["batch_discount"] == 0.5,
          "the prices the hand-worked totals use are the settings' ($0.042; $2 and $10; half for a batch)")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        results, output = Path(folder) / "results", Path(folder) / "final"
        stage_7_records(results)
        none_yet = final_ledger.recorded_spend(results, output)
        check(none_yet.calls == 0.0 and none_yet.live == 0.0 and none_yet.files == 0,
              "a folder of Stage 7 records only is no Stage 8 spend, for the gates and Jev alike")  # fmt: skip
        stage_8_records(results)
        spend = final_ledger.recorded_spend(results, output)
        check(close(spend.calls, 4.689) and spend.live == 0.0,
              f"six Stage 8 files, Stage 7's five beside them: ${spend.calls:.4f}, the hand-worked $4.689")  # fmt: skip
        check(spend.files == 6, "six files were read: not the free checks' (no cost), not Stage 7's")
        stage_7 = gate_spend.recorded_spend(results)
        check(close(stage_7.jev, 0.147) and close(stage_7.sonnet_batch, 1.5) and close(stage_7.sonnet_standard, 3.0)
              and close(stage_7.total, 4.647) and stage_7.files == 4,
              f"Stage 7's spend with Stage 8's files beside it is its own, ${stage_7.total:.3f}: the gates' $4.647 "
              "(Jev's routers aren't in it)")  # fmt: skip
        with mock.patch.object(gate_spend, "stage_7_files", lambda folder: sorted(folder.glob("*.jsonl"))):
            old = gate_spend.recorded_spend(results)
        check(not close(old.total, stage_7.total) and close(old.total, 4.647 + 0.084 + 0.042 + 1.5 + 1.5),
              f"known positive: counting every gate file, as before, gives ${old.total:.3f}: Stage 8's four paid files in it, "
              "the drift resend even priced as a batch")  # fmt: skip
        write_lines(results / "gates", "sonnet_gate_test.jsonl", [(1_000_000, 100_000)], torn='{"usage": {"input_tok')
        check(close(final_ledger.recorded_spend(results, output).calls, 4.689), "a cut-short last line is dropped")
        write_lines(results / "gates", "sonnet_gate_test.jsonl", [(1_000_000, 100_000)], torn="not json\n")
        check(refused(gate_spend.GateSpendError, lambda: final_ledger.recorded_spend(results, output)),
              "a complete line that is not JSON refuses the count")  # fmt: skip
        (results / "gates" / "sonnet_gate_test.jsonl").write_text('{"usage": {"input_tokens": 5}}\n')
        check(refused(gate_spend.GateSpendError, lambda: final_ledger.recorded_spend(results, output)),
              "a record with no billed output tokens refuses it")  # fmt: skip
        write_lines(results / "gates", "sonnet_gate_test.jsonl", [(1_000_000, 100_000)])
        write_lines(results / "gates", "sonnet_gate_latency_drift.jsonl", [(1, 1)])
        check(refused(gate_spend.GateSpendError, lambda: final_ledger.recorded_spend(results, output)),
              "a Stage 8 file the ledger has no price for refuses the count: a spend it can't know is under no cap")  # fmt: skip
        (results / "gates" / "sonnet_gate_latency_drift.jsonl").unlink()
        write_lines(output, "liveness.jsonl", [(1_000, 2_000)], model="claude-haiku-5-5")
        write_lines(output, "liveness.jsonl", [(1_000, 2_000), (500, 300)], model="claude-haiku-5-5")
        with (output / "liveness.jsonl").open("a") as file:
            file.write(json.dumps({"usage": {"input_tokens": 500, "output_tokens": 300},
                                   "request": {"model": "claude-opus-5-5"}}) + "\n")  # fmt: skip
        live = final_ledger.recorded_spend(results, output)
        # haiku: 1,000 in at $0.10 and 2,000 out at $0.50 per million is $0.0001 + $0.001; its second record (500 in, 300
        # out) is $0.00005 + $0.00015; opus 500 in at $4 and 300 out at $20 is $0.002 + $0.006.
        check(close(live.live, 0.0011 + 0.0002 + 0.008) and close(live.calls, 4.689),
              f"the liveness file counts in the $10 pool, ${live.live:.5f} (hand-worked $0.0093), and not in the $25 one")  # fmt: skip
        write_lines(output, "liveness.jsonl", [(1, 1)], model="claude-nobody-1")
        check(refused(gate_spend.GateSpendError, lambda: final_ledger.recorded_spend(results, output)),
              "a live call to a model the price sheet doesn't list refuses the count")  # fmt: skip


def caps() -> None:
    print("\n-- two caps, counted afresh, and a projection over either refuses --")
    check(final_ledger.CAPS == {CALLS: 25.0, LIVE: 10.0} and gate_spend.STAGE_CAP_USD == 25.0,
          "Stage 8's caps are $25 for every call but the replay's and $10 for the liveness calls and the replay; Stage 7's is $25")  # fmt: skip
    none = final_ledger.Spend(0.0, 0.0, 0)
    fine = final_ledger.refusal(final_ledger.Spend(20.0, 4.0, 3), {CALLS: 5.0, LIVE: 6.0})
    check(fine is None, "exactly $25 of calls and exactly $10 of live spend are allowed")
    over_calls = final_ledger.refusal(final_ledger.Spend(20.0, 0.0, 3), {CALLS: 5.0001})
    over_live = final_ledger.refusal(none, {LIVE: 10.0001})
    check(over_calls is not None and "calls pool" in over_calls and "$25.00" in over_calls and over_live is not None
          and "live pool" in over_live and "$10.00" in over_live,
          "a projection a hair over either cap refuses, naming the pool and its cap")  # fmt: skip
    check(final_ledger.refusal(none, {CALLS: 24.99, LIVE: 9.99}) is None
          and final_ledger.refusal(final_ledger.Spend(0.0, 9.5, 1), {CALLS: 1.0, LIVE: 0.6}) is not None,
          "a projection under both goes on, and what is already recorded counts in the live pool too")  # fmt: skip
    check(over_calls is not None and "nothing was sent" in over_calls, "the refusal says nothing was sent")


# ---- the liveness calls --------------------------------------------------------------------------------------------

HAIKU_JSON = {"model": "claude-haiku-5-5", "max_tokens": 64000, "output_config": {"effort": "high"},
              "messages": [{"role": "user", "content": "Reply with the single word: ready."}]}  # fmt: skip
OPUS_JSON = {"model": "claude-opus-5-5", "max_tokens": 16000, "output_config": {"effort": "medium"},
             "messages": [{"role": "user", "content": "Reply with the single word: ready."}]}  # fmt: skip


def over_calls_projector(run) -> float:
    return 24.0


def go_with(r: rehearsal.Rehearsal, steps=STEPS[:2]) -> tuple[Outcome, str]:
    outcome, out, _ = r.go(steps)
    return outcome, out


def liveness() -> None:
    print("\n-- the liveness calls: hand-written requests, the pin, the cap --")
    with rehearsal.rehearsed() as r:
        outcome, _ = go_with(r)
        check(outcome == Outcome() and r.anthropic.calls == [HAIKU_JSON, OPUS_JSON],
              "the fake client received the two requests, as hand-written JSON: haiku at high, opus at medium")  # fmt: skip
        check(final_liveness.requests() == {"haiku": HAIKU_JSON, "opus": OPUS_JSON}
              and live_calls.live_request("haiku", final_liveness.PROMPT) == HAIKU_JSON
              and live_calls.live_request("opus", final_liveness.PROMPT) == OPUS_JSON,
              "and they equal what the live replay's request builder builds")  # fmt: skip
        check(
            len(final_liveness.PROMPT.encode()) == 34,
            "the made-up prompt is 34 bytes (the projection's hand-work uses it)",
        )
        go = r.record["gos"][0]
        liveness_step = go["steps"][1]
        check(liveness_step["name"] == "liveness" and liveness_step["status"] == "done"
              and liveness_step["requests"] == {"sent": 2, "stored": 0}
              and liveness_step["detail"]["models"] == {"haiku": "claude-haiku-5-5", "opus": "claude-opus-5-5"},
              "the record holds the step's two requests and the model IDs returned")  # fmt: skip
        # Upper bound by hand: haiku 34 bytes at $0.10 and the whole 64,000 at $0.50 per million is 3.4 + 32,000 micro-dollars;
        # opus 34 bytes at $4 and 16,000 at $20 is 136 + 320,000: $0.3521394 in all.
        spend = liveness_step["detail"]["spend"]
        check(close(spend[LIVE]["projected"], 0.3521394) and spend[LIVE]["recorded"] == 0.0 and spend[LIVE]["cap"] == 10.0
              and spend[CALLS]["cap"] == 25.0 and spend[CALLS]["recorded"] == 0.0 and 0 < spend[CALLS]["projected"] < 2.0
              and jev_gate_projection() + sonnet_gate_projection() < spend[CALLS]["projected"]
              < jev_gate_projection() + sonnet_gate_projection() + 1.0,
              f"it holds the projection against each cap, before any call: ${spend[LIVE]['projected']:.7f} live, the worst case")  # fmt: skip
        # The fake billed 20 in and 5 out per call: haiku 2 + 2.5 and opus 80 + 100 micro-dollars.
        recorded = final_ledger.recorded_spend(r.world.results, r.world.output)
        check(close(recorded.live, 184.5e-6) and recorded.calls == 0.0,
              f"the two calls count under the $10 cap: ${recorded.live:.7f}, the hand-worked $0.0001845")  # fmt: skip
        stored = [json.loads(line) for line in (r.world.output / "liveness.jsonl").read_text().splitlines()]
        check([s["config"] for s in stored] == ["haiku", "opus"] and all(s["request"] in (HAIKU_JSON, OPUS_JSON) for s in stored),
              "they are stored with their requests and responses, one record each")  # fmt: skip
        outcome, _ = go_with(r)
        check(outcome == Outcome() and r.anthropic.calls == [HAIKU_JSON, OPUS_JSON] * 2,
              "a second go asks again, and the fake received both again: no call is answered from a cache")  # fmt: skip
        final = r.record["gos"][1]["steps"][1]["detail"]["spend"]
        check(close(final[LIVE]["recorded"], 184.5e-6), "and its projection sees the first go's spend as recorded")
    with rehearsal.rehearsed() as r:
        outcome, _, _ = r.go((STEPS[0], STEPS[1], STEPS[-1]))
        closing = r.record["gos"][0]["steps"][-1]["detail"]["spend"]
        check(outcome == Outcome() and close(closing[LIVE]["recorded"], 184.5e-6) and closing[LIVE]["cap"] == 10.0
              and closing[CALLS] == {"cap": 25.0, "recorded": 0.0, "projected": 0.0},
              "the record's closing step holds the spend against each cap")  # fmt: skip


def sonnet_gate_projection() -> float:
    """The Sonnet gate's projection on the rehearsal's world, by hand (#86): 6 drift resends and 24 gated test answers, at twice
    dev's list price per verdict, the test batch's at half of it."""
    from analysis import check_final_sonnet as sonnet

    return sonnet.hand_projection()


def jev_gate_projection() -> float:
    """The Jev gate's projection on the rehearsal's world, by hand (#85): the 8 drift resends at the 1,000 tokens Stage 7 was
    billed for each plus 512, and each of the 32 test answers' call body in bytes plus 512, at Jev's $0.042 per million."""
    from analysis import check_final_gates as gates

    bodies = [len(gates.hand_body(request, answer)) for request, answer, _ in gates.made_up_answers().values()]
    return (8 * (1000 + 512) + sum(bodies) + 32 * 512) * gates.PRICE / 1e6


def wrong_model(model_for: dict[str, str]):
    def respond(params):
        return message_dict("ready", usage=(20, 5), model=model_for.get(params["model"], params["model"]))

    return respond


def pins() -> None:
    print("\n-- a model ID other than the pin stops the run at the liveness step --")
    for retired, name in (
        ({"claude-haiku-5-5": "claude-haiku-5-4"}, "haiku"),
        ({"claude-opus-5-5": "claude-opus-6-0"}, "opus"),
    ):
        with rehearsal.rehearsed() as r:
            r.anthropic = FakeAnthropic(respond=wrong_model(retired))
            r.world = dataclasses.replace(
                r.world, backends=Backends(sandbox=r.sandbox, anthropic=r.anthropic, jev=r.jev.stand_in)
            )
            outcome, out, _ = r.go()
            go = r.record["gos"][0]
            check(outcome == Outcome("liveness", "LivenessRefusal", None),
                  f"a fake answering {name} as another model stops the run at the liveness step {outcome}")  # fmt: skip
            check([c["model"] for c in r.anthropic.calls] == ["claude-haiku-5-5", "claude-opus-5-5"],
                  "the fake's log shows the two liveness calls and no other")  # fmt: skip
            check([s["name"] for s in go["steps"]] == ["unlock", "liveness"] and go["stopped"]["step"] == "liveness",
                  "no later step started, and the record says where it stopped")  # fmt: skip
            check(list(retired.values())[0] in out and "no test request was sent" in out,
                  "the log names the model that answered: a retired model's ID shows a mismatch")  # fmt: skip
            check(not (r.world.results / "test_matrix.sqlite").exists(), "nothing was graded")
    with rehearsal.rehearsed() as r:
        r.world = dataclasses.replace(r.world, backends=Backends(sandbox=r.sandbox, jev=r.jev.stand_in))
        with mock.patch.object(final_liveness, "make_client", side_effect=AssertionError("the real client was built")):
            outcome, _, _ = r.go()
        check(outcome == Outcome("liveness", "LivenessRefusal", None) and r.anthropic.calls == [],
              "a rehearsal world that names no Anthropic client is refused, and no real client is built")  # fmt: skip
    sentinel = object()
    with rehearsal.rehearsed() as r:
        r.world = dataclasses.replace(r.world, backends=None)
        run = mock.Mock(world=r.world)
        with mock.patch.object(final_liveness, "make_client", return_value=sentinel):
            check(
                final_liveness.client_of(run) is sentinel,
                "the go's world, which names no backends, gets the real client",
            )


def projection() -> None:
    print("\n-- the projection refuses before the first call --")
    extra = Projector("extra", CALLS, "analysis.check_final_ledger:over_calls_projector")
    for label, projector, recorded_calls, refuses in (
        ("$24 projected for calls and $0.35 for the liveness calls: both under", extra, 0.0, False),
        ("$24 projected with $1.01 already recorded: $25.01 of calls", extra, 1.01, True),
        ("$24 projected with $0.99 already recorded: $24.99 of calls", extra, 0.99, False),
        ("$10 projected for the live pool beside the liveness calls' $0.35",
         Projector("live", LIVE, "analysis.check_final_ledger:over_calls_projector"), 0.0, True),
    ):  # fmt: skip
        with rehearsal.rehearsed() as r:
            if recorded_calls:
                # A Stage 8 Sonnet drift record that is $recorded_calls at list price: output tokens at $10 per million.
                write_lines(r.world.results / "gates", "sonnet_gate_drift.jsonl", [(0, round(recorded_calls * 1e5))])
            # The Sonnet gate's own projection (#86, a few cents on this world) is left out, so the boundary is the extra's alone.
            others = tuple(p for p in final_ledger.PROJECTORS if p.name != "sonnet_gate")
            with mock.patch.object(final_ledger, "PROJECTORS", (*others, projector)):
                outcome, out, _ = r.go(STEPS[:2])
            if refuses:
                check(outcome == Outcome("liveness", "LedgerRefusal", None) and r.anthropic.calls == [],
                      f"{label}: refused before the first call, the fake's log empty")  # fmt: skip
                check(
                    "nothing was sent" in out and "over Stage 8's cap" in out, "and the refusal is shown, with its cap"
                )
            else:
                check(outcome == Outcome() and len(r.anthropic.calls) == 2, f"{label}: the run goes on")
    check(refused(LedgerRefusal, lambda: (_ for _ in ()).throw(LedgerRefusal("x"))), "the refusal is a LedgerRefusal")
    with rehearsal.rehearsed() as r:
        write_lines(r.world.results / "gates", "sonnet_gate_drift.jsonl", [(0, 1)], torn="not json\n")
        outcome, out, _ = r.go(STEPS[:2])
        check(outcome == Outcome("liveness", "LedgerRefusal", None) and r.anthropic.calls == [] and "can't be counted" in out,
              "a spend that can't be counted refuses before the first call too")  # fmt: skip


# ---- Stage 7's guards ----------------------------------------------------------------------------------------------


def jev_record(router: str, split: str, item: str, tokens: int) -> dict:
    values: dict = {field: None for field in collect_jev.STORED_FIELDS}
    values.update(router=router, split=split, item_id=item, usage={"input_tokens": tokens, "output_tokens": 61},
                  refused=False, **{"pass": 1})  # fmt: skip
    return values


def guards() -> None:
    print("\n-- Stage 7's guards on a made-up results folder holding Stage 8's files --")
    with tempfile.TemporaryDirectory() as folder:
        results = Path(folder) / "results"
        stage_7_records(results)
        write_lines(results / "gates", "free_checks_smoke.jsonl", [(1, 1)])
        check(record_files.unopened(results) == [] and not record_files.unlocked(results),
              "before the go the folder holds no Stage 8 file, and the guard passes")  # fmt: skip
        stage_8_records(results)
        names = record_files.unopened(results)
        check(sorted(names) == sorted(["gates/free_checks_test.jsonl", "gates/jev_gate_drift.jsonl", "gates/jev_gate_test.jsonl",
                                       "gates/sonnet_gate_drift.jsonl", "gates/sonnet_gate_test.jsonl", "jev/jev_bolstered_drift.jsonl",
                                       "jev/jev_test.jsonl"]),
              f"known positive: Stage 8 files with no unlock in the record are all found {sorted(names)}")  # fmt: skip
        old_guard = not list((results / "gates").glob("*_test.jsonl"))
        check(
            not old_guard,
            "known positive: the old guard, no `*_test.jsonl` at all, fails on this folder once it holds them",
        )
        record = results / record_files.RECORD_FOLDER / record_files.RECORD_NAME
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"record": "final evaluation", "gos": [{"go": 1, "unlock": None}]}))
        check(record_files.unopened(results) != [] and not record_files.unlocked(results),
              "a go that has started but not passed its unlock (the hook's own moment) still shows them: the guard bites")  # fmt: skip
        record.write_text(
            json.dumps({"record": "final evaluation", "gos": [{"go": 1, "unlock": {"time": "t", "checks": []}}]})
        )
        check(record_files.unopened(results) == [] and record_files.unlocked(results),
              "once a go's unlock is in the record, a resume's hook passes with Stage 8's files there")  # fmt: skip
        allowed = {"train:1", "dev:1"}
        for path in record_files.files_in(results / "gates"):
            path.write_text(json.dumps({"item_id": "train:1"}) + "\n")
        (results / "gates" / "jev_gate_test.jsonl").write_text(json.dumps({"item_id": "test:1"}) + "\n")
        (results / "gates" / "jev_gate_drift.jsonl").write_text(json.dumps({"item_id": "train:7"}) + "\n")

        def items(path: Path) -> set[str]:
            return {json.loads(line)["item_id"] for line in path.read_text().splitlines() if line}

        stage_7 = record_files.stage_7_files(results / "gates")
        check(all(items(p) <= allowed for p in stage_7) and "jev_gate_test.jsonl" not in {p.name for p in stage_7},
              "Stage 7's item check reads only Stage 7's files, and passes")  # fmt: skip
        check(not all(items(p) <= allowed for p in sorted((results / "gates").glob("*.jsonl"))),
              "known positive: the old check over every file fails on the same folder (a test item, a drift item)")  # fmt: skip
        check(all(r.suffix == ".jsonl" for r in stage_7) and len(stage_7) == len(record_files.files_in(results / "gates")) - 5,
              "five of the folder's files are Stage 8's, the rest Stage 7's")  # fmt: skip

        # The numbers check_final_design quotes: Jev's routers, Stage 7's splits only, even where the collector's splits
        # gain test and drift. Hand-worked: 5,000,000 + 1,000,000 input tokens in train and dev at $0.042 per million.
        jev_folder = results / "jev"
        for path in jev_folder.glob("*.jsonl"):
            path.unlink()
        rows = {"jev_train.jsonl": jev_record("jev", "train", "train:1", 5_000_000),
                "jev_dev.jsonl": jev_record("jev", "dev", "dev:1", 1_000_000),
                "jev_test.jsonl": jev_record("jev", "test", "test:1", 2_000_000),
                "jev_drift.jsonl": jev_record("jev", "drift", "train:1", 3_000_000)}  # fmt: skip
        for name, row in rows.items():
            (jev_folder / name).write_text(json.dumps(row) + "\n")
        with mock.patch.object(collect_jev, "SPLITS", ("train", "dev", "smoke", "test", "drift")):
            quoted = jev_routers_dollars(results)
            loaded = collect_jev.load_settings()
            everything = sum(
                collect_jev.overhead_dollars(r, loaded)
                for s in ("train", "dev", "test", "drift")
                for r in collect_jev.read_records("jev", s, results)
            )
        check(close(quoted, 0.252) and close(everything, 0.252 + 0.084 + 0.126),
              f"the quoted Jev spend is Stage 7's $0.252 though Stage 8's files are there; counting them all gives ${everything:.3f}")  # fmt: skip


def main() -> int:
    naming()
    ledger()
    caps()
    liveness()
    pins()
    projection()
    guards()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        sys.exit(main())
