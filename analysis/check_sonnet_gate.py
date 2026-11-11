"""Check the Sonnet gate end to end against a fake Anthropic client (PLAN_v4.md §9; Amendment A17, items 6, 7, 12, 14,
15, 16, 18 and 19; A6 item 6's safeguards).

Run from the project root:  uv run python -m analysis.check_sonnet_gate

On synthetic data only, with every expected value worked out by hand: no results/ folder, no Docker, no Anthropic call
and no key. The fake client (`analysis.fake_anthropic`) covers Messages and Message Batches and records exactly what
reached it, so the request, the batches and the safeguards are asserted at that seam.

- the request equals hand-written JSON, with and without an output schema, and is the one the harness's collection run
  would build for the same model, effort and format;
- the gate score is the integer over 100, and a refusal, a cap stop or an unparseable result is a refusal with its
  billed tokens kept; the overhead is the billed tokens at 2026-10-12's list prices;
- request IDs fit the API's limits, for every train and dev item and for synthetic ones;
- the batch collector's safeguards, each fed its case: only gated configs that passed the free checks are sent, a
  batch ID is saved as soon as the batch is created, only errored or expired requests are resent, results are matched
  by request ID, the cap refuses before anything is sent, and a rerun never resubmits a stored or in-flight request;
- the latency caller: standard API, one call at a time, a monotonic clock, the first call flagged;
- the stored records, the gate-score table's Sonnet column and the `haiku/Sonnet` walk and section;
- the deployable module imports nothing from the harness or the analysis, and the key is stored nowhere.
"""

import ast
import contextlib
import inspect
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from analysis import collect_sonnet_gate as collect
from analysis.benchmark import read_rows
from analysis.bootstrap import resample
from analysis.check_simulator import synthetic
from analysis.checking import refused
from analysis.collect_gates import GateInput
from analysis.fake_anthropic import FakeAnthropic, message_dict
from analysis.gate_table import PaidScore, build_table, pure_traces
from analysis.gate_walk import MIXES, pure_section, pure_walks
from analysis.operating_points import check_sections, read_section, write_section
from analysis.plan import amendment_item
from analysis.references import TARGETS
from analysis.router_names import HAIKU, SONNET_GATE, cascade_section
from harness.client import PROJECT_ROOT, build_request, load_models
from routers import sonnet_gate
from routers.request import Request

problems: list[str] = []

CUSTOM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # the API's rule, written out again here
GATED = ("haiku", "sonnet", "opus_low")
KEYS = [(i, c) for i in ("i0", "i1") for c in ("haiku", "sonnet", "opus_low", "opus")]


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


# ---- hand-written expectations ---------------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["a", "b"]}},
    "required": ["answer"],
    "additionalProperties": False,
}
INSTRUCTION = (
    "You are checking another model's answer to a request. Judge whether the answer is fully correct: it answers "
    "what was asked, every fact, calculation, label or piece of code in it is right, and it follows every "
    "instruction in the request, including any rules about format, length or wording."
)
CLOSING = "Give the probability, from 0 to 100, that the answer is fully correct."
ENUM_SCHEMA = {
    "type": "object",
    "properties": {"probability_correct": {"type": "integer", "enum": list(range(101))}},
    "required": ["probability_correct"],
    "additionalProperties": False,
}


def hand_request(content: str) -> dict:
    """A17 item 6's request, written out: nothing but the model, the cap, low effort with the enum schema as the
    output format, and one user message."""
    return {
        "model": "claude-sonnet-5-5",
        "max_tokens": 16000,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": ENUM_SCHEMA}},
        "messages": [{"role": "user", "content": content}],
    }


def imports_of(source: str) -> set[str]:
    """The top-level packages a module imports."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def reads_environment(source: str) -> bool:
    """Whether a module reads `environ`, calls `getenv` or names an `api_key`."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
            return True
        if isinstance(node, ast.keyword) and node.arg == "api_key":
            return True
        if isinstance(node, ast.Name) and node.id in ("environ", "getenv", "api_key"):
            return True
    return False


def free_record(item: str, config: str, failed: str | None) -> dict:
    return {"item_id": item, "config": config, "passed": failed is None, "failed_first": failed}


def message(text: str, stop: str = "end_turn", usage: tuple[int, int] = (100, 5)) -> dict:
    return {
        "id": "msg_x", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "stop_reason": stop,
        "stop_sequence": None, "content": [{"type": "text", "text": text}] if text else [],
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    }  # fmt: skip


def main() -> int:
    settings = sonnet_gate.load_settings()

    print("-- the request --")
    with_schema = (
        f"{INSTRUCTION}\n\n<request>\nClassify this.\n</request>\n<output_schema>\n{json.dumps(SCHEMA)}\n"
        f'</output_schema>\n<answer>\n{{"answer": "a"}}\n</answer>\n\n{CLOSING}'
    )
    without_schema = f"{INSTRUCTION}\n\n<request>\nWrite f.\n</request>\n<answer>\nANSWER\n</answer>\n\n{CLOSING}"
    got = sonnet_gate.build_request(settings, Request("Classify this.", SCHEMA), '{"answer": "a"}')
    check(got == hand_request(with_schema) and json.dumps(got) == json.dumps(hand_request(with_schema)),
          "a request with a schema equals the hand-written JSON, key order included")  # fmt: skip
    got = sonnet_gate.build_request(settings, Request("Write f.", None), "ANSWER")
    check(got == hand_request(without_schema) and "output_schema" not in got["messages"][0]["content"],
          "a request without one drops the output-schema block, and only that")  # fmt: skip
    check(not {"temperature", "thinking", "system", "cache_control", "top_p", "top_k"} & set(got)
          and "cache_control" not in json.dumps(got), "no temperature, thinking, system prompt or caching")  # fmt: skip
    tricky = sonnet_gate.build_request(settings, Request("p {answer} {prompt}", None), "a {schema_block} {prompt}")
    check(tricky["messages"][0]["content"] == hand_request(
              f"{INSTRUCTION}\n\n<request>\np {{answer}} {{prompt}}\n</request>\n<answer>\na {{schema_block}} "
              f"{{prompt}}\n</answer>\n\n{CLOSING}")["messages"][0]["content"],
          "a prompt or answer that spells a placeholder is not substituted into")  # fmt: skip
    check(got == build_request({"model": "claude-sonnet-5-5", "effort": "low"}, got["messages"][0]["content"], 16000,
                                output_schema=ENUM_SCHEMA),
          "it is the request the collection run's builder makes for this model, effort, cap and format")  # fmt: skip
    check(sonnet_gate.request_sha256(got) == sonnet_gate.request_sha256(json.loads(json.dumps(got)))
          and sonnet_gate.request_sha256(got) != sonnet_gate.request_sha256(hand_request("other")),
          "its hash follows its content")  # fmt: skip

    print("\n-- the gate score --")
    cases = [
        ("an integer 70", message(json.dumps({"probability_correct": 70})), 0.7, None),
        ("0 and 100 are both scores", message('{"probability_correct": 0}'), 0.0, None),
        ("100", message('{"probability_correct": 100}'), 1.0, None),
        ("a refusal stop", message("", "refusal", (120, 3)), None, "refusal"),
        ("a refusal stop, whatever text it carries", message('{"probability_correct": 90}', "refusal"), None, "refusal"),
        ("an output-cap stop", message("", "max_tokens", (120, 16000)), None, "max_tokens"),
        ("a cap stop with a valid-looking number", message('{"probability_correct": 90}', "max_tokens"), None,
         "max_tokens"),
        ("any other stop", message('{"probability_correct": 90}', "pause_turn"), None, "stop_reason: pause_turn"),
        ("text that is not JSON", message("I think it is fine"), None, "unparseable"),
        ("no text at all", message(""), None, "unparseable"),
        ("a list", message("[70]"), None, "unparseable"),
        ("another key", message('{"confidence": 70}'), None, "unparseable"),
        ("an extra key", message('{"probability_correct": 70, "why": "x"}'), None, "unparseable"),
        ("a float", message('{"probability_correct": 70.0}'), None, "unparseable"),
        ("a string", message('{"probability_correct": "70"}'), None, "unparseable"),
        ("a boolean", message('{"probability_correct": true}'), None, "unparseable"),
        ("101", message('{"probability_correct": 101}'), None, "unparseable"),
        ("-1", message('{"probability_correct": -1}'), None, "unparseable"),
    ]  # fmt: skip
    for what, response, score, reason in cases:
        verdict = sonnet_gate.parse_response(response)
        check(verdict.score == score and verdict.reason == reason and verdict.refused == (score is None)
              and (verdict.input_tokens, verdict.output_tokens) == tuple(response["usage"].values()),
              f"{what}: gate score {score}, refused {score is None}, billed tokens kept")  # fmt: skip
    two_blocks = message("")
    two_blocks["content"] = [{"type": "thinking", "thinking": "hmm", "signature": "s"},
                             {"type": "text", "text": '{"probability_correct": '}, {"type": "text", "text": "35}"}]  # fmt: skip
    check(sonnet_gate.parse_response(two_blocks).score == 0.35, "thinking blocks are ignored and text blocks joined")
    check(refused(ValueError, lambda: sonnet_gate.parse_response(message("{}", usage=(-1, 5))))
          and refused(KeyError, lambda: sonnet_gate.parse_response({"stop_reason": "end_turn"})),
          "a response with no billed tokens is no response, not a free call")  # fmt: skip

    print("\n-- the overhead --")
    ok = sonnet_gate.parse_response(message('{"probability_correct": 70}', usage=(1000, 500)))
    capped = sonnet_gate.parse_response(message("", "max_tokens", (100, 16000)))
    # 1,000 in at $2 per million is $0.002 and 500 out at $10 per million is $0.005. The capped call is charged 16,000
    # output tokens: $0.0002 + $0.16.
    check(abs(sonnet_gate.overhead_dollars(settings, ok) - 0.007) < 1e-12
          and abs(sonnet_gate.overhead_dollars(settings, capped) - 0.1602) < 1e-12,
          "billed tokens at $2 and $10 per million, a capped call charged what it was billed")  # fmt: skip
    prices = load_models()
    check(settings.price_snapshot == prices["price_snapshot"]
          and (settings.input_per_million, settings.output_per_million)
          == (prices["prices"]["claude-sonnet-5-5"]["input"], prices["prices"]["claude-sonnet-5-5"]["output"])
          and (settings.model, settings.effort, settings.max_tokens)
          == (prices["configs"]["sonnet"]["model"], "low", prices["configs"]["sonnet"]["max_tokens"]),
          "the settings' prices, snapshot, model and cap are configs/models.yaml's (effort is low, not sonnet's high)")  # fmt: skip

    print("\n-- the settings --")
    raw = json.loads((PROJECT_ROOT / "routers" / "gate_settings.json").read_text())["sonnet"]
    check(raw["output_schema"] == ENUM_SCHEMA and raw["gated_configs"] == ["haiku", "sonnet", "opus_low"]
          and raw["request_id_suffix"] == "sg", "the file holds A17 item 6's schema, the gated configs and the suffix")  # fmt: skip
    item6 = amendment_item((PROJECT_ROOT / "PLAN_v4.md").read_text(), "A17", 6)
    start = item6.index("```text") + 8
    plan_text = "\n".join(line[6:] for line in item6[start : item6.index("      ```", start)].split("\n"))
    ours = settings.prompt.replace("{schema_block}", settings.schema_block).replace("{output_schema}", "X")
    theirs = plan_text.rstrip("\n").replace("{the request's output schema as JSON}", "X")
    check(ours == theirs, "the prompt is the plan's item 6, word for word (the schema block in place)")
    for what, mutate in (
        ("a missing section", lambda r: r.pop("model")),
        ("effort ultra", lambda r: r.update(effort="ultra")),
        ("a cap of 0", lambda r: r.update(max_tokens=0)),
        ("an enum that stops at 99", lambda r: r["output_schema"]["properties"]["probability_correct"].update(
            enum=list(range(100)))),
        ("a minimum and maximum instead of the enum", lambda r: r["output_schema"]["properties"].update(
            probability_correct={"type": "integer", "minimum": 0, "maximum": 100})),
        ("a prompt without {answer}", lambda r: r.update(prompt=r["prompt"].replace("{answer}", ""))),
        ("a schema block without its schema", lambda r: r.update(schema_block="<output_schema>\n</output_schema>\n")),
        ("a suffix with a colon", lambda r: r.update(request_id_suffix="s:g")),
        ("no gated config", lambda r: r.update(gated_configs=[])),
        ("a negative price", lambda r: r["list_prices"].update(input_per_million=-1)),
    ):  # fmt: skip
        changed = json.loads(json.dumps(raw))
        mutate(changed)
        check(refused(sonnet_gate.SonnetSettingsError, lambda: sonnet_gate.read_sonnet_settings(changed)),
              f"a sonnet section with {what} is refused")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        none = Path(folder) / "none.json"
        check(
            refused(sonnet_gate.SonnetSettingsError, lambda: sonnet_gate.load_settings(none)),
            "and a file that isn't there",
        )

    print("\n-- request IDs --")
    check(sonnet_gate.custom_id(settings, "mmlu_pro:4346", "opus_low") == "mmlu_pro-4346_opus_low_r0_sg",
          "the item ID with ':' replaced by '-', then config, run and sg (A17 item 14's example)")  # fmt: skip
    check(sonnet_gate.custom_id(settings, "banking77:test-0008", "haiku") == "banking77-test-0008_haiku_r0_sg",
          "a Banking77 train ID that spells 'test' keeps it")  # fmt: skip
    check(refused(ValueError, lambda: sonnet_gate.custom_id(settings, "x" * 60, "opus_low"))
          and refused(ValueError, lambda: sonnet_gate.custom_id(settings, "", "haiku")),
          "one over 64 characters, or with no item, is refused")  # fmt: skip
    check(sonnet_gate.custom_id(settings, "é:1", "haiku") == "--1_haiku_r0_sg",
          "any other character the API rejects becomes '-' too, as the collection run's IDs do")  # fmt: skip
    synthetic_ids = [f"{task}:{n}" for task in ("ifeval", "banking77", "mmlu_pro", "mbppplus") for n in (0, 12345)]
    check(all(CUSTOM_ID.match(sonnet_gate.custom_id(settings, i, c)) for i in synthetic_ids for c in GATED),
          "synthetic IDs fit the API's limits")  # fmt: skip
    rows = read_rows(split="train") + read_rows(split="dev")
    ids = [sonnet_gate.custom_id(settings, r["item_id"], c) for r in rows for c in GATED]
    longest = max(map(len, ids))
    check(len(ids) == len(set(ids)) == 3 * len(rows) and all(CUSTOM_ID.match(i) for i in ids),
          f"all {len(rows)} train and dev items give {len(ids)} distinct IDs inside the limits (longest {longest} of 64)")  # fmt: skip

    print("\n-- what is sent --")
    free = [free_record(i, c, None) for i, c in KEYS]
    free[2]["passed"], free[2]["failed_first"] = False, "code"
    inputs = [GateInput(i, c, Request(f"prompt {i}", None), f"answer {i} {c} SCORE=70", "end_turn") for i, c in KEYS]
    jobs = collect.build_jobs(inputs, free, settings)
    expected = [sonnet_gate.custom_id(settings, i, c) for i, c in KEYS if c in GATED and (i, c) != KEYS[2]]
    check([j.custom_id for j in jobs] == expected and not any(j.config == "opus" for j in jobs),
          "only haiku's, sonnet's and opus_low's answers that passed the free checks, never opus's")  # fmt: skip
    check(jobs[0].params == sonnet_gate.build_request(settings, inputs[0].request, inputs[0].answer)
          and jobs[0].sha256 == sonnet_gate.request_sha256(jobs[0].params), "each job carries its request and hash")  # fmt: skip
    check(refused(collect.SonnetCollectError, lambda: collect.build_jobs(inputs, free[1:], settings))
          and [j.custom_id for j in collect.build_jobs(inputs[:1], free, settings)] == expected[:1],
          "a gated answer with no free-check record is refused: the free-check file is the filter")  # fmt: skip

    print("\n-- the batch collector --")
    # Seven answers that passed, one per item, with the score the fake gate will give each; two more that must never
    # be sent: opus's, and one that failed a free check.
    marks = {f"b{n}": mark for n, mark in enumerate(("SCORE=10", "SCORE=20", "SCORE=30", "SCORE=40", "REFUSE", "CAP",
                                                     "GARBAGE"))}  # fmt: skip
    gated = [
        GateInput(i, "haiku", Request(f"prompt {i}", None), f"answer {i} {m}", "end_turn") for i, m in marks.items()
    ]
    extra = [GateInput("b0", "opus", Request("prompt b0", None), "opus answer SCORE=99", "end_turn"),
             GateInput("bx", "sonnet", Request("prompt bx", SCHEMA), "failed answer SCORE=99", "end_turn")]  # fmt: skip
    free_all = [free_record(i.item_id, i.config, "stop" if i.item_id == "bx" else None) for i in gated + extra]
    jobs = collect.build_jobs(gated + extra, free_all, settings)
    ids = [j.custom_id for j in jobs]
    check(len(jobs) == 7 and all(j.config == "haiku" for j in jobs), "seven requests are built from the nine answers")
    with tempfile.TemporaryDirectory() as folder:
        results = Path(folder)
        fake = FakeAnthropic(polls_before_end=1)

        def sent() -> list[str]:
            return [r["custom_id"] for batch in fake.created.values() for r in batch]

        print("  cap")
        over = refused(collect.CapExceeded, lambda: collect.submit(
            jobs, split="dev", client=fake, cost_per_check=0.5, max_usd=3.4, results_dir=results))  # fmt: skip
        check(over and not fake.created and not collect.batch_log_path(results).exists(),
              "7 requests at $0.50 is $3.50: over a cap of $3.40, nothing is sent and no log is written")  # fmt: skip
        check(refused(collect.SonnetCollectError, lambda: collect.submit(
            jobs, split="dev", client=fake, cost_per_check=-1, max_usd=3.4, results_dir=results)),
              "a missing or negative cost per check is refused too")  # fmt: skip

        print("  a batch ID is saved as soon as the batch is created")
        fake.create_error_after = 1  # the second `create` fails, as a dropped connection would
        crashed = refused(RuntimeError, lambda: collect.submit(
            jobs, split="dev", client=fake, cost_per_check=0.5, max_usd=3.5, results_dir=results, max_requests=3))  # fmt: skip
        log = json.loads(collect.batch_log_path(results).read_text())["batches"]
        check(crashed and len(log) == 1 and log[0]["id"] == "msgbatch_fake0" and log[0]["request_count"] == 3
              and log[0]["custom_ids"] == ids[:3] and log[0]["split"] == "dev" and log[0]["ingested_at"] is None
              and log[0]["created_at"], "the first batch's ID, creation time and count are on disk when the second fails")  # fmt: skip

        print("  a rerun never resubmits an in-flight request")
        fake.create_error_after = None
        again = collect.submit(jobs, split="dev", client=fake, cost_per_check=0.5, max_usd=2.0, results_dir=results,
                               max_requests=3)  # fmt: skip
        check(again.sent == 4 and again.in_flight == 3 and again.stored == 0 and len(again.batches) == 2
              and sorted(sent()) == sorted(ids) and len(sent()) == 7,
              "the 3 in flight are left alone, the other 4 go in two batches, and every request is sent once")  # fmt: skip
        check(len(json.loads(collect.batch_log_path(results).read_text())["batches"]) == 3,
              "all three batches are in the log")  # fmt: skip
        idle = collect.submit(jobs, split="dev", client=fake, cost_per_check=0.5, max_usd=0.0, results_dir=results)
        check(idle.sent == 0 and idle.in_flight == 7 and len(fake.created) == 3,
              "a third run sends nothing, and a zero cap is no problem for nothing to send")  # fmt: skip

        print("  results are matched by request ID; only errored and expired requests come back")
        errored, expired = ids[1], ids[5]
        fake.batch_failures.update({errored: "errored", expired: "expired"})
        sleeps: list[float] = []
        collect.wait(jobs, split="dev", client=fake, results_dir=results, poll_s=7.0, sleep=sleeps.append,
                     now=lambda: "NOW")  # fmt: skip
        stored = collect.read_sonnet_records("dev", results)
        check(sleeps == [7.0] and fake.messages.batches.retrieves >= 4, "it polls until each batch has ended")
        check(sorted(r["custom_id"] for r in stored) == sorted(set(ids) - {errored, expired}),
              "the five successes are stored; the errored and the expired one are not")  # fmt: skip
        want = {"b0": 0.1, "b1": 0.2, "b2": 0.3, "b3": 0.4}
        by_item = {r["item_id"]: r for r in stored}
        check(all(by_item[i]["custom_id"] == sonnet_gate.custom_id(settings, i, "haiku") for i in by_item)
              and all(sonnet_gate.parse_response(by_item[i]["result"]["message"]).score == s
                      for i, s in want.items() if i in by_item),
              "each verdict is the one its own answer earned, though the results arrived shuffled")  # fmt: skip
        check(by_item["b4"]["refused"] and by_item["b4"]["stop_reason"] == "refusal"
              and by_item["b4"]["usage"]["input_tokens"] > 0
              and by_item["b6"]["refused"] and by_item["b6"]["stop_reason"] == "end_turn",
              "a refusal and an unparseable result are stored as verdicts, refused, with their billed tokens")  # fmt: skip
        check(all(r["batch_id"] in fake.created and r["request_sha256"] == jobs[ids.index(r["custom_id"])].sha256
                  and r["collected_at"] == "NOW" and r["model"] == "claude-sonnet-5-5" for r in stored),
              "every record names its batch, its request's hash, the model that answered and when")  # fmt: skip
        entries = json.loads(collect.batch_log_path(results).read_text())["batches"]
        check(all(e["ingested_at"] == "NOW" for e in entries)
              and sum(e["counts"].get("errored", 0) for e in entries) == 1
              and sum(e["counts"].get("expired", 0) for e in entries) == 1,
              "the log records each batch's outcomes")  # fmt: skip
        resend = collect.submit(jobs, split="dev", client=fake, cost_per_check=0.5, max_usd=1.0, results_dir=results)
        last = list(fake.created.values())[-1]
        check(resend.sent == 2 and resend.stored == 5 and [r["custom_id"] for r in last] == [errored, expired],
              "a rerun sends exactly the errored and the expired request, in one new batch")  # fmt: skip
        collect.wait(jobs, split="dev", client=fake, results_dir=results, sleep=sleeps.append)
        final = collect.read_sonnet_records("dev", results)
        check(len(final) == 7 and len({r["custom_id"] for r in final}) == 7,
              "and then all seven verdicts are stored, once each")  # fmt: skip
        before = len(fake.created)
        done = collect.submit(jobs, split="dev", client=fake, cost_per_check=0.5, max_usd=0.0, results_dir=results)
        check(done.sent == 0 and done.stored == 7 and len(fake.created) == before,
              "a rerun after that sends nothing: no stored request is resubmitted")  # fmt: skip
        counts = {i: sent().count(i) for i in set(sent())}
        check(set(counts) == set(ids) and counts[errored] == 2 and counts[expired] == 2
              and sum(counts.values()) == 9 and all(n == 1 for i, n in counts.items() if i not in (errored, expired)),
              "nine requests went out in all: the seven once, and only the errored and the expired one twice")  # fmt: skip
        check(all(r["params"] == jobs[ids.index(r["custom_id"])].params for batch in fake.created.values() for r in batch),
              "each batch request carries exactly the gate's request as its params")  # fmt: skip

    print("\n-- results that don't match --")
    for what, tamper in (
        ("a result for an ID the batch never held",
         lambda b, o: [*o, {"custom_id": "nobody_haiku_r0_sg", "result": {"type": "expired"}}]),
        ("the same ID twice", lambda b, o: [*o, o[0]]),
        ("a canceled request", lambda b, o: [{**o[0], "result": {"type": "canceled"}}, *o[1:]]),
    ):  # fmt: skip
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder)
            fake = FakeAnthropic(tamper=tamper)
            collect.submit(jobs, split="dev", client=fake, cost_per_check=0.5, max_usd=3.5, results_dir=results)
            check(refused(collect.SonnetCollectError, lambda: collect.wait(jobs, split="dev", client=fake,
                                                                           results_dir=results, sleep=lambda s: None)),
                  f"{what} stops the run")  # fmt: skip
            again = collect.submit(jobs, split="dev", client=fake, cost_per_check=0.5, max_usd=0.0, results_dir=results)
            check(again.sent == 0 and again.in_flight == 7, f"and after {what} the batch is still in flight, not resent")  # fmt: skip

    print("\n-- the latency caller --")
    probe = [GateInput(f"l{n}", "haiku", Request(f"prompt l{n}", None), f"answer l{n} SCORE={n}0", "end_turn")
             for n in (1, 2, 3)]  # fmt: skip
    probe_jobs = collect.build_jobs(probe, [free_record(i.item_id, "haiku", None) for i in probe], settings)
    with tempfile.TemporaryDirectory() as folder:
        results = Path(folder)
        path = collect.latency_path("train", results)
        seen_before: list[int] = []

        def respond(params):  # type: ignore[no-untyped-def]
            # What is on disk when each call arrives: one call at a time means the last call is already stored.
            seen_before.append(len(path.read_text().splitlines()) if path.exists() else 0)
            return message_dict('{"probability_correct": 77}', usage=(300, 40))

        fake = FakeAnthropic(respond=respond)
        times = iter([100.0, 101.5, 200.0, 200.25, 300.0, 300.5])
        made = collect.run_latency(probe_jobs, split="train", client=fake, settings=settings, cost_per_check=0.01,
                                   max_usd=1.0, results_dir=results, clock=lambda: next(times), now=lambda: "NOW")  # fmt: skip
        records = collect.read_latency_records("train", results)
        check(made == 3 and fake.calls == [j.params for j in probe_jobs] and not fake.created,
              "three calls through the standard API, none through a batch, each the gate's own request")  # fmt: skip
        check(seen_before == [0, 1, 2], "one at a time: each call is stored before the next is made")
        check([r["seconds"] for r in records] == [1.5, 0.25, 0.5]
              and [r["first_call"] for r in records] == [True, False, False],
              "each call's time is the clock's difference, and only the first call is flagged")  # fmt: skip
        check(all(r["pass"] == 2 and r["api"] == "standard" and r["batch_id"] is None
                  and r["usage"] == {"input_tokens": 300, "output_tokens": 40}
                  and r["result"]["type"] == "succeeded" and not r["refused"] for r in records),
              "records are pass 2 of the standard API, with the whole result and billed tokens")  # fmt: skip
        default_clock = inspect.signature(collect.run_latency).parameters["clock"].default
        check(default_clock is time.perf_counter, "the default clock is the monotonic time.perf_counter")
        check(collect.run_latency(probe_jobs, split="train", client=fake, settings=settings, cost_per_check=1,
                                  max_usd=0.0, results_dir=results) == 0 and len(fake.calls) == 3,
              "a rerun makes no call that is stored")  # fmt: skip
        check(refused(collect.CapExceeded, lambda: collect.run_latency(
            probe_jobs, split="smoke", client=fake, settings=settings, cost_per_check=1, max_usd=2.0,
            results_dir=results)), "three calls at $1 are over a cap of $2, and none is made")  # fmt: skip
        check(refused(collect.SonnetCollectError, lambda: collect.latency_path("dev", results)),
              "the latency sample is train answers (or made-up ones): dev has no latency file")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        results = Path(folder)
        fake = FakeAnthropic(message_errors={2: RuntimeError("fake: 400")})
        stopped = refused(RuntimeError, lambda: collect.run_latency(
            probe_jobs, split="train", client=fake, settings=settings, cost_per_check=0.01, max_usd=1.0,
            results_dir=results, clock=iter(range(100)).__next__))  # fmt: skip
        check(stopped and [r["item_id"] for r in collect.read_latency_records("train", results)] == ["l1"],
              "an API error stops the run, after storing the first call and as no verdict for the second")  # fmt: skip
        collect.run_latency(probe_jobs, split="train", client=fake, settings=settings, cost_per_check=0.01, max_usd=1.0,
                            results_dir=results, clock=iter(range(100)).__next__)  # fmt: skip
        check([r["item_id"] for r in collect.read_latency_records("train", results)] == ["l1", "l2", "l3"],
              "and the rerun carries on from the second")  # fmt: skip

    print("\n-- the stored records, the gate-score table's Sonnet column and the walk --")
    tasks = ("ifeval", "banking77", "mmlu_pro", "mbppplus")
    # Per task, four items: haiku is wrong on n = 0 and 2 and right on 1 and 3; opus is right on all. The gate gives
    # the wrong answers 20 and 30 and the right ones 90 and 80.
    marker = {0: "SCORE=20", 1: "SCORE=90", 2: "SCORE=30", 3: "SCORE=80"}
    dev = synthetic({f"{t}:{n}": (t, {"haiku": (0.001, n % 2), "opus": (0.010, 1)}) for t in tasks for n in range(4)})
    answers = [(i, c) for i in dev.items for c in ("haiku", "opus")]
    walk_inputs = [GateInput(i, c, Request(f"prompt {i}", None), f"{c} answer of {i} " + marker[int(i[-1])], "end_turn")
                   for i, c in answers]  # fmt: skip
    free_records = [free_record(i, c, None) for i, c in answers]
    with tempfile.TemporaryDirectory() as folder:
        results = Path(folder)
        fake = FakeAnthropic()
        walk_jobs = collect.build_jobs(walk_inputs, free_records, settings)
        collect.submit(walk_jobs, split="dev", client=fake, cost_per_check=0.01, max_usd=1.0, results_dir=results)
        collect.wait(walk_jobs, split="dev", client=fake, results_dir=results, sleep=lambda s: None)
        scores = collect.paid_scores("dev", settings, results)
        table = build_table(free_records, answers, {SONNET_GATE: scores})
        entry = table.row("banking77:1", "haiku").sonnet
        record = next(r for r in collect.read_sonnet_records("dev", results) if r["item_id"] == "banking77:1")
        usage = record["usage"]
        check(len(scores) == 16 and table.row("banking77:1", "opus").sonnet is None
              and entry is not None and entry.score == 0.9 and not entry.refused
              and abs(entry.overhead - (usage["input_tokens"] * 2 + usage["output_tokens"] * 10) / 1e6) < 1e-15,
              "the table's Sonnet column holds each gated answer's score ÷ 100 and its overhead at list price; opus's is empty")  # fmt: skip
        check(table.row("ifeval:0", "haiku").sonnet == PaidScore(0.2, scores[("ifeval:0", "haiku")].overhead, False),
              "an entry is score, overhead and refusal flag")  # fmt: skip
        walks = pure_walks(dev, table, SONNET_GATE, resample(dev))
        w = walks["realistic"][1.0]
        check(list(walks) == list(MIXES) and all(set(by) == set(TARGETS) for by in walks.values()),
              "haiku/Sonnet is walked on the real mixes and targets, as haiku/Jev is")  # fmt: skip
        check([s.cutoff for s in w.steps] == [None, 0.9, 0.8, 0.3] and w.operating.cutoff == 0.8 and not w.fell_back,
              "its steps are the distinct scores, strictest first; the walk stops where a wrong answer is accepted (.3), so .8 is its operating point")  # fmt: skip
        two = [t for t in pure_traces(table, SONNET_GATE, 0.8) if t.item_id.startswith("ifeval")]
        check(sorted(t.configs for t in two) == [("haiku",), ("haiku",), ("haiku", "opus"), ("haiku", "opus")]
              and all(t.overhead > 0 for t in two),
              "at .8, the two right answers stay and the two wrong ones go to opus, each paying the gate")  # fmt: skip
        section = pure_section(walks, SONNET_GATE)
        key = cascade_section(HAIKU, SONNET_GATE)
        target = results / "operating_points.json"
        write_section(key, section, target)
        check_sections(target, required=(key,))
        check(read_section(key, target)["realistic"]["100%"]["cutoff"] == 0.8,
              "its section is written under haiku/Sonnet's own key and passes the allowlist")  # fmt: skip
        # A refused call is a rejection in the table, charged what it was billed.
        refused_inputs = [*walk_inputs[:1], GateInput("r:1", "haiku", Request("p", None), "REFUSE", "end_turn")]
        refused_free = [free_record("ifeval:0", "haiku", None), free_record("r:1", "haiku", None)]
        refused_jobs = collect.build_jobs(refused_inputs, refused_free, settings)
        collect.submit(refused_jobs, split="train", client=fake, cost_per_check=0.01, max_usd=1.0, results_dir=results)
        collect.wait(refused_jobs, split="train", client=fake, results_dir=results, sleep=lambda s: None)
        train_scores = collect.paid_scores("train", settings, results)
        refused_entry = train_scores[("r:1", "haiku")]
        small = build_table(refused_free, [("ifeval:0", "haiku"), ("r:1", "haiku")], {SONNET_GATE: train_scores})
        trace = pure_traces(small, SONNET_GATE, 0.0)[1]
        check(refused_entry.score is None and refused_entry.refused and refused_entry.overhead > 0
              and trace.configs == ("haiku", "opus") and trace.overhead == refused_entry.overhead,
              "a refused call has no score, is charged its billed tokens, and escalates even at cutoff 0")  # fmt: skip
        # The stored record is checked against its own result when it is read back.
        stored_path = collect.gate_path("sonnet_gate", "train", results)
        lines = [json.loads(line) for line in stored_path.read_text().splitlines()]
        flipped = next(i for i, r in enumerate(lines) if r["refused"])
        lines[flipped]["refused"] = False
        stored_path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
        check(refused(collect.SonnetCollectError, lambda: collect.paid_scores("train", settings, results)),
              "a record whose refusal flag disagrees with its result is refused when the table is built")  # fmt: skip
        lines[flipped]["refused"] = True
        lines[flipped]["usage"]["output_tokens"] += 1
        stored_path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
        check(refused(collect.SonnetCollectError, lambda: collect.paid_scores("train", settings, results)),
              "and so is one whose usage differs")  # fmt: skip

    print("\n-- imports, and the key --")
    stdlib = sys.stdlib_module_names
    found = imports_of((PROJECT_ROOT / "routers" / "sonnet_gate.py").read_text())
    check(found <= stdlib | {"anthropic", "routers"} and "anthropic" in found,
          f"the deployable gate imports the standard library, the Anthropic SDK and the package: {sorted(found - stdlib)}")  # fmt: skip
    check(imports_of("import harness.client\nfrom analysis import x\nimport numpy") == {"harness", "analysis", "numpy"},
          "known positive: the import reader sees the harness, the analysis and a third party")  # fmt: skip
    key_value = "sk-ant-api03-THIS-IS-A-FAKE-KEY-FOR-THE-CHECK"
    reads_env = [
        name
        for name in ("routers/sonnet_gate.py", "analysis/collect_sonnet_gate.py")
        if reads_environment((PROJECT_ROOT / name).read_text())
    ]
    check(not reads_env, f"neither the gate nor its collector reads the environment or an api_key: {reads_env}")
    check(reads_environment("import os\nkey = os.environ['ANTHROPIC_API_KEY']") and reads_environment("k = os.getenv('X')")
          and reads_environment("Anthropic(api_key=k)"), "known positive: the reader sees environ, getenv and api_key")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        os.environ["ANTHROPIC_API_KEY"] = key_value
        try:
            results = Path(folder)
            fake = FakeAnthropic()
            collect.submit(walk_jobs, split="dev", client=fake, cost_per_check=0.01, max_usd=1.0, results_dir=results)
            collect.wait(walk_jobs, split="dev", client=fake, results_dir=results, sleep=lambda s: None)
            collect.run_latency(walk_jobs[:2], split="train", client=fake, settings=settings, cost_per_check=0.01,
                                max_usd=1.0, results_dir=results)  # fmt: skip
        finally:
            del os.environ["ANTHROPIC_API_KEY"]
        files = [f for f in Path(folder).rglob("*") if f.is_file()]
        check(len(files) == 3 and not any(key_value in f.read_text() for f in files)
              and not any(key_value in json.dumps(c) for c in fake.calls),
              f"with a key in the environment, no stored file ({len(files)}) and no request carries it")  # fmt: skip

    print("\n-- the command line's splits --")
    args = ["--cost-per-check", "0.001", "--max-usd", "1"]
    with contextlib.redirect_stderr(io.StringIO()):
        check(refused(SystemExit, lambda: collect.main(["batch", "--split", "test", *args])),
              "known positive: the command line refuses a test split before anything is read")  # fmt: skip
    check(all(split in collect.SPLITS for split in ("train", "dev", "smoke")) and "test" not in collect.SPLITS,
          "and takes the gates' files' splits, train, dev and smoke (`collect_gates.SPLITS`)")  # fmt: skip

    if problems:
        print(f"\n{len(problems)} checks failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
