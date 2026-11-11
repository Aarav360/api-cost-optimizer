"""Check the gates' first slice: the answer reader, the free checks, their storage, the gate-score table, the cascade
decision and a pure cascade's traces, walk and section (PLAN_v4.md §9; Amendment A17, items 2 to 4, 8 to 10, 13 to 15
and 19).

Run from the project root:  uv run python -m analysis.check_gates

On synthetic data only, with every expected value worked out by hand: no results/ folder, no Docker (the code check
runs against a recording stand-in for the sandbox; `analysis.check_gates_sandbox` runs the real one in the hook), no
Jev or Anthropic call, and no benchmark answer.

- the answer reader returns run 0's item, config, run, answer and stop reason for train or dev, refuses test, and
  leaves success, quality, parsed answer and grading detail out, in the SQL it runs as well as in the result;
- the stop check, the schema check, the `assert` line and its modules, and the code check's program, limits and
  verdicts, retried on a sandbox failure and never a verdict for one;
- storage: one free-check record per answer in the fixed format, a rerun skipping stored answers, made-up answers
  stored apart under the smoke split;
- the gate-score table, refused whole on each of A17 item 13's conditions, each fed its bad case;
- the cascade decision, a pure cascade's traces through the simulator and its walk, steps and section.
"""

import json
import math
import re
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from analysis import collect_gates, gate_answers
from analysis.benchmark import SealedItemError
from analysis.bootstrap import resample
from analysis.check_jev_settings import write_benchmark
from analysis.check_routers import imported_packages
from analysis.check_simulator import synthetic
from analysis.check_walk import DRAWS, M1, MIX
from analysis.checking import refused
from analysis.gate_table import (
    GateRow,
    GateScoreTable,
    GateTableError,
    PaidScore,
    build_table,
    cascade_trace,
    pure_traces,
)
from analysis.gate_walk import MIXES, cutoff_steps, pure_candidates, pure_section, pure_walks
from analysis.operating_points import (
    OperatingPointsError,
    build_cascade_section,
    build_section,
    check_sections,
    read_section,
    write_section,
)
from analysis.references import TARGETS
from analysis.router_names import CASCADES, FREE_GATE, HAIKU, JEV_GATE, NAMED_ROUTERS, SONNET_GATE, cascade_section
from analysis.simulator import Trace, TraceError, simulate
from analysis.walk import walk_candidates
from harness.client import PROJECT_ROOT
from harness.collect import MATRIX_COLUMNS
from harness.evaluators import extract_code as graders_extract_code
from routers import cascade, gates
from routers.request import Request
from routers.sandbox import IMAGE, ExecResult, SandboxError

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


# ---- the made-up benchmark and store ---------------------------------------------------------------------------

SENTINELS = {"success": 1, "quality": 0.5, "parsed": "PARSED-SENTINEL", "detail": "DETAIL-SENTINEL",
             "status": "STATUS-SENTINEL", "request_hash": "HASH-SENTINEL"}  # fmt: skip
ASSERT = "assert f(2)==4"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["a", "b"]}, "explanation": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def bench_row(item_id: str, split: str, task: str, prompt: str, schema: dict | None = None) -> dict:
    row = {"item_id": item_id, "split": split, "source": task, "task_type": task, "source_group": item_id,
           "prompt": prompt, "evaluation_method": "exact", "expected_answer": "EXPECTED-SENTINEL", "meta": {}}  # fmt: skip
    if schema is not None:
        row["output_schema"] = schema
    return row


CODE_PROMPT = f"Write f.\n\n{ASSERT}\n\nTask: double it."
BENCH = [
    bench_row("m:1", "train", "mbppplus", CODE_PROMPT),
    bench_row("b:1", "train", "banking77", "Classify.", SCHEMA),
    bench_row("m:d1", "dev", "mbppplus", CODE_PROMPT),
    bench_row("m:t1", "test", "mbppplus", CODE_PROMPT),
]
GOOD = "```python\ndef f(x):\n    return x * 2\n```"


def store_row(item_id: str, config: str, run: int, answer: str, stop: str | None) -> tuple:
    values = {c: f"{c}-filler" for c in MATRIX_COLUMNS} | SENTINELS
    values |= {"item_id": item_id, "config": config, "run": run, "answer": answer, "stop_reason": stop}
    return tuple(values[c] for c in MATRIX_COLUMNS)


def write_store(path: Path) -> None:
    rows = [
        store_row("m:1", "haiku", 0, GOOD, "end_turn"),
        store_row("m:1", "sonnet", 0, "no code here", "max_tokens"),
        store_row("m:1", "haiku", 1, "RUN-ONE-ANSWER", "end_turn"),
        store_row("b:1", "haiku", 0, '{"answer": "a"}', "end_turn"),
        store_row("m:d1", "haiku", 0, GOOD, "end_turn"),
        store_row("m:d1", "opus", 0, GOOD, "end_turn"),
        store_row("m:t1", "haiku", 0, "TEST-ITEM-ANSWER", "end_turn"),
    ]
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f"CREATE TABLE rows ({', '.join(MATRIX_COLUMNS)})")
        conn.executemany(f"INSERT INTO rows VALUES ({', '.join('?' * len(MATRIX_COLUMNS))})", rows)
        conn.commit()


@contextmanager
def traced_sql() -> Iterator[list[str]]:
    """The SQL the answer reader runs, captured through SQLite's own trace hook."""
    statements: list[str] = []
    real = sqlite3.connect

    def connect(*args, **kwargs):  # type: ignore[no-untyped-def]
        conn = real(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    gate_answers.sqlite3.connect = connect  # type: ignore[assignment]
    try:
        yield statements
    finally:
        gate_answers.sqlite3.connect = real  # type: ignore[assignment]


def columns_named(sql: str) -> set[str]:
    return set(re.findall(r"\w+", sql)) & set(MATRIX_COLUMNS)


# ---- a stand-in for the sandbox --------------------------------------------------------------------------------


class FakeSandbox:
    """Records what reached it and replies from a script (an ExecResult, or a SandboxError to raise)."""

    def __init__(self, *replies: ExecResult | Exception) -> None:
        self.replies = list(replies)
        self.received: list[tuple[str, float, str]] = []

    def __call__(self, source: str, timeout_s: float, memory: str) -> ExecResult:
        self.received.append((source, timeout_s, memory))
        reply = self.replies.pop(0) if self.replies else ExecResult(0, False, "", "", 0.0)
        if isinstance(reply, Exception):
            raise reply
        return reply


OK = ExecResult(0, False, "", "", 0.123)
SETTINGS = gates.read_gate_settings().code_check
CODE_REQUEST = Request(CODE_PROMPT)
NOW = lambda: "2026-11-10T12:00:00+00:00"  # noqa: E731
DIGEST = "sha256:" + "ab" * 32


def code(answer: str, sandbox: FakeSandbox | None = None, sleeps: list[float] | None = None) -> gates.CodeVerdict:
    sandbox = sandbox or FakeSandbox()
    sleeps = [] if sleeps is None else sleeps
    return gates.code_check(CODE_REQUEST, answer, SETTINGS, sandbox, sleeps.append)


# ---- the gate-score table's made-up rows -----------------------------------------------------------------------

ITEMS = ("i0", "i1", "i2", "i3")
KEYS = [(i, c) for i in ITEMS for c in ("haiku", "opus")]


def rows_of(free_failed: dict[str, str] | None = None, jev: dict[str, PaidScore] | None = None) -> list[GateRow]:
    """haiku's and opus's answers of i0 to i3; `free_failed` maps an item's haiku answer to the check it failed
    first, `jev` an item's haiku answer to its Jev entry. Opus passes everything and is never gated."""
    free_failed, jev = free_failed or {}, jev or {}
    rows = []
    for item in ITEMS:
        failed = free_failed.get(item)
        rows.append(GateRow(item, "haiku", failed is None, failed, jev.get(item)))
        rows.append(GateRow(item, "opus", True, None))
    return rows


def free_record(item: str, config: str, failed: str | None) -> dict:
    return {"item_id": item, "config": config, "passed": failed is None, "failed_first": failed}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        bench = write_benchmark(directory, BENCH)
        store = directory / "store.sqlite"
        write_store(store)

        print("-- the gates' answer reader --")
        check(gate_answers.ANSWER_COLUMNS == ("item_id", "config", "run", "answer", "stop_reason"),
              "it names the item, config, run, answer and stop reason")  # fmt: skip
        with traced_sql() as statements:
            train = gate_answers.read_answers("train", store, bench)
        got = [(a.item_id, a.config, a.run, a.answer, a.stop_reason) for a in train]
        check(got == [("m:1", "haiku", 0, GOOD, "end_turn"), ("m:1", "sonnet", 0, "no code here", "max_tokens"),
                      ("b:1", "haiku", 0, '{"answer": "a"}', "end_turn")],
              "train's run-0 answers come back with their item, config, run, answer and stop reason")  # fmt: skip
        dev_answers = gate_answers.read_answers("dev", store, bench)
        check([(a.item_id, a.config) for a in dev_answers] == [("m:d1", "haiku"), ("m:d1", "opus")],
              "dev's come back, and train's and test's do not")  # fmt: skip
        everything = repr(train) + repr(dev_answers)
        check(not any(s in everything for s in ("PARSED-SENTINEL", "DETAIL-SENTINEL", "STATUS-SENTINEL", "-filler"))
              and all(set(vars(a)) == set(gate_answers.ANSWER_COLUMNS) for a in train + dev_answers),
              "success, quality, parsed answer, grading detail and every other column never reach the result")  # fmt: skip
        check("RUN-ONE-ANSWER" not in everything and "TEST-ITEM-ANSWER" not in everything,
              "run 1's answer and a test item's are not returned")  # fmt: skip
        check(refused(SealedItemError, lambda: gate_answers.read_answers("test", directory / "no.sqlite", bench)),
              "the test split is refused, before the store is opened")  # fmt: skip
        sql = " ".join(statements)
        check(columns_named(sql) == set(gate_answers.ANSWER_COLUMNS) and "SELECT" in sql,
              "the SQL it ran names those five columns of the store and no other")  # fmt: skip
        check(columns_named("SELECT item_id, config, success FROM rows WHERE run = 0")
              == set(gate_answers.ANSWER_COLUMNS[:2]) | {"success", "run"} != set(gate_answers.ANSWER_COLUMNS),
              "known positive: a query that also names success is seen")  # fmt: skip
        check(columns_named(gate_answers.QUERY) <= set(gate_answers.ANSWER_COLUMNS), "its query text names no other")
        requests = gate_answers.requests_of("train", bench)
        check(requests == {"m:1": Request(CODE_PROMPT, None), "b:1": Request("Classify.", SCHEMA)},
              "requests come through the benchmark reader: prompt and output schema only")  # fmt: skip

        print("\n-- the stop check --")
        check(gates.stop_check("end_turn"), "an end_turn passes")
        check(not any(gates.stop_check(s) for s in ("max_tokens", "refusal", "pause_turn", "tool_use", "", None)),
              "any other stop rejects")  # fmt: skip

        print("\n-- the schema check --")
        req = Request("Classify.", SCHEMA)
        cases = [
            ('{"answer": "a"}', True, "valid, without the optional explanation"),
            ('{"answer": "b", "explanation": "x"}', True, "valid, with it"),
            ("not json", False, "not JSON"),
            ('{"answer": "c"}', False, "an answer outside the enum"),
            ('{"explanation": "x"}', False, "the required answer missing"),
            ('{"answer": "a", "extra": 1}', False, "a key the schema doesn't allow"),
            ('{"answer": 1}', False, "a wrong type"),
            ('"a"', False, "JSON that isn't an object"),
            ("", False, "an empty answer"),
        ]
        for answer, expected, what in cases:
            check(gates.schema_check(req, answer) == (expected, True), f"{what}: passed={expected}, applied")
        check(gates.schema_check(Request("Write f."), "plain, not JSON text") == (True, False),
              "with no output schema it passes and doesn't apply")  # fmt: skip
        check(gates.matches_schema(True, {"type": "integer"}) is False and gates.matches_schema(1, {"type": "integer"}),
              "a bool is not an integer")  # fmt: skip
        check(refused(gates.UnsupportedSchema, lambda: gates.matches_schema("a", {"type": "string", "pattern": "a"})),
              "a schema keyword it doesn't read is refused, not ignored")  # fmt: skip

        print("\n-- the assert line and its modules --")
        check(gates.assert_line(CODE_PROMPT) == ASSERT, "the line starting assert is found")
        check(gates.assert_line("Write a poem.\nAssertions are fun.\n  assert x") is None
              and gates.assert_line("Classify this.") is None,
              "a prompt with no line starting assert has none (not prose, not an indented line)")  # fmt: skip
        modules = {
            "assert f(2)==4": [],
            "assert math.isclose(f(1.0), 2.0)": ["math"],
            "assert sys.getsizeof(f(1)) == 56": ["sys"],
            "assert os.path.join(a) == math.floor(1) + math.ceil(2)": ["os", "math"],
            "assert f('math') == 1": [],
            "assert f(x, ": [],
        }
        for module_line, named in modules.items():
            check(gates.assert_modules(module_line) == named, f"{module_line!r} names {named}")
        check(gates.code_program("def f(x): return x", "assert math.isclose(f(1), 1)")
              == "import math\ndef f(x): return x\n\nassert math.isclose(f(1), 1)",
              "the program is the imports, the code, a blank line and the assert")  # fmt: skip
        for text in (
            "```python\nA\n```",
            "```py\nA\n```\n```python3\nB\n```",
            "```\nU\n```",
            "x ```Python\nP\n```",
            "```python\nA\n```\ntext\n```\nU\n```",
            "no block",
            "```python\n```",
            "```python\r\nA\r\n```",
        ):
            check(
                gates.extract_code(text) == graders_extract_code(text),
                f"the code extraction equals the grader's on {text!r}",
            )

        print("\n-- the code check --")
        box = FakeSandbox(OK)
        verdict = code(GOOD, box)
        check(verdict == gates.CodeVerdict(True, True, 0, False, 0.12, None), "a clean exit passes, with its details")
        check(box.received == [("def f(x):\n    return x * 2\n\n\n" + ASSERT, 20.0, "512m")],
              "the sandbox got the code and the assert at 20 s and 512 MB, no import since the line names none")  # fmt: skip
        check(SETTINGS.timeout_s == 20 and SETTINGS.memory == "512m" and SETTINGS.image == IMAGE,
              "the settings file holds 20 s, 512 MB and the sandbox image")  # fmt: skip
        boxed = FakeSandbox()
        gates.code_check(Request("Write f.\n\nassert math.isclose(f(1), 2)"), GOOD, SETTINGS, boxed)
        check(boxed.received[0][0].startswith("import math\ndef f"), "a module the assert names is imported first")
        failing = ExecResult(1, False, "", "Traceback (most recent call last):\n  File x\nAssertionError", 0.5)
        verdict = code(GOOD, FakeSandbox(failing))
        check(not verdict.passed and verdict.applied and verdict.returncode == 1 and verdict.stderr_tail == "AssertionError",
              "a failed assert rejects, with the last line of stderr")  # fmt: skip
        verdict = code(GOOD, FakeSandbox(ExecResult(None, True, "", "", 20.01)))
        check(not verdict.passed and verdict.timed_out is True and verdict.returncode is None, "a timeout rejects")
        check(not code(GOOD, FakeSandbox(ExecResult(0, True, "", "", 20.0))).passed,
              "a clean exit that also timed out rejects")  # fmt: skip
        for answer, what in (("no block at all", "no code block"), ("```python\n  \n\n```", "an empty block")):
            box = FakeSandbox()
            verdict = code(answer, box)
            check(verdict == gates.CodeVerdict(False, True) and box.received == [], f"{what} rejects, with no run")
        box = FakeSandbox()
        check(gates.code_check(Request("Classify."), "anything", SETTINGS, box) == gates.CodeVerdict(True, False)
              and box.received == [], "a request with no assert line passes and doesn't apply, with no run")  # fmt: skip
        sleeps: list[float] = []
        box = FakeSandbox(SandboxError("docker down"), SandboxError("docker down"), OK)
        check(code(GOOD, box, sleeps).passed and len(box.received) == 3 and sleeps == [5.0, 5.0],
              "a sandbox failure is retried, with a wait between tries, and the retry's verdict counts")  # fmt: skip
        box = FakeSandbox(*[SandboxError("docker down")] * 5)
        check(refused(SandboxError, lambda: code(GOOD, box, [])) and len(box.received) == 3,
              "a sandbox that keeps failing is raised after three tries, never a verdict")  # fmt: skip

        print("\n-- all three free checks, as a record's fields --")
        stored = gates.FREE_FIELDS

        def free(request: Request, answer: str, stop: str | None, reply: ExecResult = OK) -> dict:
            return gates.free_checks(request, answer, stop, SETTINGS, digest=DIGEST, now=NOW, run=FakeSandbox(reply),
                                     sleep=lambda s: None)  # fmt: skip

        record = free(CODE_REQUEST, GOOD, "end_turn")
        check(tuple(record) == stored, "the fields are FREE_FIELDS, in order")
        check(record["passed"] is True and record["failed_first"] is None and record["stop_passed"]
              and record["schema_applied"] is False and record["code_applied"] and record["code_image_digest"] == DIGEST
              and record["code_checked_at"] == NOW() and record["code_duration_s"] == 0.12,
              "a good answer passes all three, with the digest, the time and the duration")  # fmt: skip
        record = free(CODE_REQUEST, GOOD, "max_tokens")
        check(record["failed_first"] == "stop" and not record["passed"] and record["code_passed"],
              "a stop other than end_turn fails the stop check first, and the other verdicts stay on record")  # fmt: skip
        record = free(Request("Classify.", SCHEMA), "not json", "end_turn")
        check(record["failed_first"] == "schema" and record["schema_applied"] and record["code_applied"] is False
              and record["code_image_digest"] == DIGEST, "a schema failure is named, and the digest is there anyway")  # fmt: skip
        record = free(CODE_REQUEST, GOOD, "end_turn", failing)
        check(record["failed_first"] == "code" and record["code_returncode"] == 1, "a code failure is named")
        record = free(CODE_REQUEST, "no block", "refusal", failing)
        check(
            record["failed_first"] == "stop" and not record["code_passed"],
            "when several fail the first in order is named",
        )

        print("\n-- storage --")
        inputs = [
            collect_gates.GateInput("s1", "haiku", CODE_REQUEST, GOOD, "end_turn"),
            collect_gates.GateInput("s2", "haiku", CODE_REQUEST, "no block", "end_turn"),
            collect_gates.GateInput("s3", "haiku", Request("Classify.", SCHEMA), '{"answer": "a"}', "end_turn"),
            collect_gates.GateInput("s4", "opus", CODE_REQUEST, GOOD, "max_tokens"),
        ]
        results = directory / "results"
        sandbox = FakeSandbox()
        summary = collect_gates.collect_free(inputs, split="smoke", settings=SETTINGS, digest=DIGEST, results_dir=results,
                                             run=sandbox, now=NOW)  # fmt: skip
        path = results / "gates" / "free_checks_smoke.jsonl"
        lines = path.read_text().splitlines()
        check(summary == collect_gates.Summary(4, 0) and len(lines) == 4, "four made-up answers give four records")
        first = json.loads(lines[0])
        check(tuple(first) == ("item_id", "config", "run", "split", "gate", "pass", *stored)
              and tuple(first) == collect_gates.FREE_RECORD_FIELDS,
              "a record is its keys, then the free checks' fields, in #59's order")  # fmt: skip
        check(first["item_id"] == "s1" and first["run"] == 0 and first["split"] == "smoke" and first["gate"] == "free"
              and first["pass"] == 1, "the keys are item, config, run 0, split, gate and pass 1")  # fmt: skip
        check([json.loads(x)["failed_first"] for x in lines] == [None, "code", None, "stop"],
              "each record says which check failed first")  # fmt: skip
        check(len(sandbox.received) == 2, "the sandbox ran for the two code answers that had a block, s1 and s4")
        check(
            not (results / "gates" / "free_checks_train.jsonl").exists(), "made-up answers are stored under smoke only"
        )
        before = path.read_bytes()
        sandbox = FakeSandbox()
        summary = collect_gates.collect_free(inputs, split="smoke", settings=SETTINGS, digest=DIGEST, results_dir=results,
                                             run=sandbox, now=NOW)  # fmt: skip
        check(summary == collect_gates.Summary(0, 4) and path.read_bytes() == before and sandbox.received == [],
              "a rerun skips stored answers: no new record, no sandbox run")  # fmt: skip
        with path.open("a") as file:
            file.write('{"item_id": "torn')
        summary = collect_gates.collect_free(inputs + [collect_gates.GateInput("s5", "haiku", CODE_REQUEST, GOOD, "end_turn")],
                                             split="smoke", settings=SETTINGS, digest=DIGEST, results_dir=results,
                                             run=FakeSandbox(), now=NOW)  # fmt: skip
        check(summary == collect_gates.Summary(1, 4) and len(collect_gates.read_free_records("smoke", results)) == 5,
              "a torn last line is dropped and the run resumes with the next answer")  # fmt: skip
        down = FakeSandbox(*[SandboxError("down")] * 9)
        more = inputs[:1] + [collect_gates.GateInput("s6", "haiku", CODE_REQUEST, GOOD, "end_turn")]
        check(refused(SandboxError, lambda: collect_gates.collect_free(more, split="smoke", settings=SETTINGS, digest=DIGEST,
                                                                       results_dir=results, run=down, now=NOW,
                                                                       sleep=lambda s: None)),
              "a sandbox that stays down stops the run")  # fmt: skip
        check(
            len(collect_gates.read_free_records("smoke", results)) == 5,
            "and writes no verdict for the answer it failed on",
        )
        check(refused(collect_gates.GateCollectError, lambda: collect_gates.collect_free(inputs + inputs[:1], split="smoke",
                                                                                         settings=SETTINGS, digest=DIGEST,
                                                                                         results_dir=results, now=NOW)),
              "the same answer twice in the list is refused")  # fmt: skip
        check(refused(collect_gates.GateCollectError, lambda: collect_gates.gate_path("free_checks", "test", results)),
              "the test split has no file here")  # fmt: skip
        bad_cases = {
            "another split": lambda r: r | {"split": "train"},
            "another gate": lambda r: r | {"gate": "jev"},
            "an extra field": lambda r: r | {"extra": 1},
            "a repeat": None,
        }
        for what, mutate in bad_cases.items():
            broken = results / "broken"
            records = [json.loads(x) for x in lines[:2]]
            if mutate is None:
                records.append(records[0])
            else:
                records[1] = mutate(records[1])
            (broken / "gates").mkdir(parents=True, exist_ok=True)
            (broken / "gates" / "free_checks_smoke.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            check(refused(collect_gates.GateCollectError, lambda: collect_gates.read_free_records("smoke", broken)),
                  f"a stored file with {what} is refused whole")  # fmt: skip
        train_inputs = collect_gates.gate_inputs("train", bench, store)
        check([(i.item_id, i.config) for i in train_inputs] == [("m:1", "haiku"), ("m:1", "sonnet"), ("b:1", "haiku")],
              "train's inputs come through the reader and the benchmark reader")  # fmt: skip
        check(refused(SealedItemError, lambda: collect_gates.gate_inputs("test", bench, store)),
              "and the test split is refused")  # fmt: skip
        collect_gates.collect_free(train_inputs, split="train", settings=SETTINGS, digest=DIGEST, results_dir=results,
                                   run=FakeSandbox(), now=NOW)  # fmt: skip
        train_records = collect_gates.read_free_records("train", results)
        check([(r["item_id"], r["config"], r["failed_first"]) for r in train_records]
              == [("m:1", "haiku", None), ("m:1", "sonnet", "stop"), ("b:1", "haiku", None)],
              "train's records are stored under train, one per answer")  # fmt: skip

    print("\n-- the gate-score table --")
    table = build_table([free_record(i, c, None) for i, c in KEYS], KEYS)
    check(len(table.rows) == 8 and all(r.jev is None and r.sonnet is None for r in table.rows),
          "built from the records, every paid column is empty")  # fmt: skip
    check(table.row("i2", "opus") == GateRow("i2", "opus", True, None) and table.items_of("haiku") == list(ITEMS),
          "rows are found by item and config")  # fmt: skip
    filled = build_table([free_record(i, c, None) for i, c in KEYS], KEYS,
                         {JEV_GATE: {("i0", "haiku"): PaidScore(0.9, 0.0001, False), ("i1", "haiku"): PaidScore(None, 0.0001, True)}})  # fmt: skip
    check(filled.row("i0", "haiku").jev == PaidScore(0.9, 0.0001, False) and filled.row("i1", "haiku").jev == PaidScore(None, 0.0001, True)
          and filled.row("i2", "haiku").jev is None and filled.row("i0", "haiku").sonnet is None
          and filled.row("i0", "haiku").paid(JEV_GATE) == PaidScore(0.9, 0.0001, False),
          "a paid gate's entries fill its column and leave the other empty")  # fmt: skip
    good_rows = rows_of({"i1": "code"}, {"i0": PaidScore(0.5, 0.0, False)})
    check(len(GateScoreTable.build(good_rows, KEYS).rows) == 8, "a table with a failed answer and a score builds")
    bad: dict[str, Callable[[], object]] = {
        "an answer missing": lambda: GateScoreTable.build(rows_of()[:-1], KEYS),
        "an answer twice": lambda: GateScoreTable.build(rows_of() + rows_of()[:1], KEYS),
        "a row that is no answer of the split": lambda: GateScoreTable.build(
            rows_of() + [GateRow("x", "haiku", True, None)], KEYS
        ),
        "a gate score above 1": lambda: GateScoreTable.build(rows_of(jev={"i0": PaidScore(1.2, 0.0, False)}), KEYS),
        "a gate score below 0": lambda: GateScoreTable.build(rows_of(jev={"i0": PaidScore(-0.1, 0.0, False)}), KEYS),
        "a gate score of NaN": lambda: GateScoreTable.build(rows_of(jev={"i0": PaidScore(math.nan, 0.0, False)}), KEYS),
        "a gate score that is a bool": lambda: GateScoreTable.build(
            rows_of(jev={"i0": PaidScore(True, 0.0, False)}), KEYS
        ),
        "a gate score that is a string": lambda: GateScoreTable.build(
            rows_of(jev={"i0": PaidScore("0.5", 0.0, False)}),  # type: ignore[arg-type]
            KEYS,
        ),
        "a Jev score on an answer that failed a free check": lambda: GateScoreTable.build(
            rows_of({"i0": "stop"}, {"i0": PaidScore(0.5, 0.0, False)}), KEYS
        ),
        "a refused entry on an answer that failed a free check": lambda: GateScoreTable.build(
            rows_of({"i0": "code"}, {"i0": PaidScore(None, 0.0, True)}), KEYS
        ),
        "a negative overhead": lambda: GateScoreTable.build(rows_of(jev={"i0": PaidScore(0.5, -1e-9, False)}), KEYS),
        "an infinite overhead": lambda: GateScoreTable.build(
            rows_of(jev={"i0": PaidScore(0.5, math.inf, False)}), KEYS
        ),
        "an overhead of NaN": lambda: GateScoreTable.build(rows_of(jev={"i0": PaidScore(0.5, math.nan, False)}), KEYS),
        "a refused call that has a score": lambda: GateScoreTable.build(
            rows_of(jev={"i0": PaidScore(0.5, 0.0, True)}), KEYS
        ),
        "a call with neither score nor refusal": lambda: GateScoreTable.build(
            rows_of(jev={"i0": PaidScore(None, 0.0, False)}), KEYS
        ),
        "a failed answer with no failed check named": lambda: GateScoreTable.build(
            [GateRow("i0", "haiku", False, None)] + rows_of()[1:], KEYS
        ),
        "a failed check that is no check": lambda: GateScoreTable.build(
            [GateRow("i0", "haiku", False, "vibes")] + rows_of()[1:], KEYS
        ),
        "a paid entry for an answer with no free-check row": lambda: build_table(
            [free_record(i, c, None) for i, c in KEYS], KEYS, {JEV_GATE: {("zz", "haiku"): PaidScore(0.5, 0.0, False)}}
        ),
        "a paid gate that isn't one": lambda: build_table(
            [free_record(i, c, None) for i, c in KEYS], KEYS, {"free": {}}
        ),
    }
    for what, call in bad.items():
        check(refused(GateTableError, call), f"refused whole: {what}")
    check(not refused(GateTableError, lambda: GateScoreTable.build(rows_of(jev={"i0": PaidScore(0.0, 0.0, False)}), KEYS))
          and not refused(GateTableError, lambda: GateScoreTable.build(rows_of(jev={"i0": PaidScore(1.0, 0.0, False)}), KEYS)),
          "the scores 0 and 1 are in range")  # fmt: skip

    print("\n-- the cascade decision --")
    cases3 = [
        # (start, free passed, gate score, cutoff, configs, what)
        ("opus", True, 0.99, 0.5, ("opus",), "a start at opus is the start alone"),
        ("opus", False, None, 0.5, ("opus",), "even when the free checks failed"),
        ("opus", True, None, None, ("opus",), "and with no paid gate"),
        ("haiku", True, None, None, ("haiku",), "free checks pass, no paid gate: the start alone"),
        ("haiku", True, 0.8, 0.8, ("haiku",), "free checks pass and the gate score equals the cutoff: accepted"),
        ("haiku", True, 0.81, 0.8, ("haiku",), "above the cutoff: accepted"),
        ("haiku", True, 0.79, 0.8, ("haiku", "opus"), "just below the cutoff: the start, then opus"),
        ("haiku", True, None, 0.8, ("haiku", "opus"), "a refused call has no gate score and is a rejection"),
        ("haiku", False, 0.99, 0.8, ("haiku", "opus"), "a failed free check escalates whatever the gate score"),
        ("haiku", False, None, None, ("haiku", "opus"), "a failed free check escalates with no paid gate"),
        ("opus_low", True, 0.1, 0.8, ("opus_low", "opus"), "a router's pick of opus_low escalates to opus"),
        ("sonnet", True, 0.0, 0.0, ("sonnet",), "a cutoff of 0 accepts every score"),
    ]
    for start, passed, score, cutoff, called, what in cases3:
        check(cascade.configs_called(start, passed, score, cutoff) == called, what)
    check(
        refused(ValueError, lambda: cascade.configs_called("gpt", True, None, None)),
        "a start off the ladder is refused",
    )
    for start, passed, cutoff, expected in (("haiku", True, 0.5, True), ("haiku", False, 0.5, False),
                                            ("haiku", True, None, False), ("opus", True, 0.5, False)):  # fmt: skip
        check(
            cascade.paid_gate_called(start, passed, cutoff) is expected,
            f"a paid gate is called: {expected} for start {start}, free passed {passed}, cutoff {cutoff}",
        )
    for module in ("gates", "cascade"):
        found = imported_packages(PROJECT_ROOT / "routers" / f"{module}.py")
        check(not found & {"analysis", "harness", "numpy", "anthropic"}, f"routers/{module}.py imports {sorted(found)}")

    print("\n-- a pure cascade's traces --")
    haiku_free = rows_of({"i1": "code", "i3": "stop"})
    table = GateScoreTable.build(haiku_free, KEYS)
    got_traces = pure_traces(table, FREE_GATE, None)
    check(got_traces == [Trace("i0", ("haiku",)), Trace("i1", ("haiku", "opus")), Trace("i2", ("haiku",)),
                         Trace("i3", ("haiku", "opus"))],
          "free checks only: a failed answer goes to opus, a passed one stays, no overhead")  # fmt: skip
    entries = {"i0": PaidScore(0.9, 0.0002, False), "i2": PaidScore(0.6, 0.0001, False)}
    paid_table = GateScoreTable.build(rows_of({"i1": "code", "i3": "stop"}, entries), KEYS)
    got_traces = pure_traces(paid_table, JEV_GATE, 0.7)
    check(got_traces == [Trace("i0", ("haiku",), 0.0002), Trace("i1", ("haiku", "opus")),
                         Trace("i2", ("haiku", "opus"), 0.0001), Trace("i3", ("haiku", "opus"))],
          "Jev at 0.7: i0 accepted and paid, i2 rejected and paid, failed answers escalate and pay no gate")  # fmt: skip
    check(
        pure_traces(paid_table, JEV_GATE, 0.6)[2] == Trace("i2", ("haiku",), 0.0001),
        "a score at the cutoff is accepted",
    )
    check(cascade_trace(paid_table, "i0", "opus", JEV_GATE, 0.7, 0.00005) == Trace("i0", ("opus",), 0.00005),
          "a router-started trace sent to opus pays its router and no gate, and reads no answer")  # fmt: skip
    check(cascade_trace(paid_table, "i0", "haiku", JEV_GATE, 0.7, 0.00005) == Trace("i0", ("haiku",), 0.00005 + 0.0002),
          "a router-started trace pays the router's overhead plus the gate call")  # fmt: skip
    refused_table = GateScoreTable.build(rows_of(jev={"i0": PaidScore(None, 0.0003, True)}), KEYS)
    check(cascade_trace(refused_table, "i0", "haiku", JEV_GATE, 0.0) == Trace("i0", ("haiku", "opus"), 0.0003),
          "a refused gate call is a rejection even at a cutoff of 0, and is charged")  # fmt: skip
    check(refused(GateTableError, lambda: pure_traces(table, JEV_GATE, 0.7)),
          "a paid gate with no entry on a passing answer is refused")  # fmt: skip
    check(refused(GateTableError, lambda: pure_traces(table, JEV_GATE, None))
          and refused(GateTableError, lambda: pure_traces(table, FREE_GATE, 0.5)),
          "a paid gate needs a cutoff and the free checks alone take none")  # fmt: skip
    check(
        refused(GateTableError, lambda: pure_traces(table, "log-probabilities", 0.5)),
        "a gate no amendment names is refused",
    )

    dev = synthetic({i: ("t1", {"haiku": (0.001, h), "opus": (0.010, o)}) for i, h, o in
                     (("i0", 1, 1), ("i1", 0, 1), ("i2", 1, 1), ("i3", 1, 0))})  # fmt: skip
    outcomes = simulate(dev, pure_traces(paid_table, JEV_GATE, 0.7))
    check([round(o.cost, 6) for o in outcomes] == [0.0012, 0.011, 0.0111, 0.011]
          and [o.success for o in outcomes] == [1, 1, 1, 0],
          "the simulator costs the traces: haiku $0.001, opus $0.010, plus the gate calls, and returns the last answer")  # fmt: skip
    train_matrix = synthetic({i: ("t1", {"haiku": (0.001, 1), "opus": (0.010, 1)}) for i in ITEMS}, split="train")
    check(refused(TraceError, lambda: simulate(train_matrix, pure_traces(paid_table, JEV_GATE, 0.7))),
          "an escalation on train items is refused by the simulator, as opus never runs on train")  # fmt: skip
    check(refused(TraceError, lambda: simulate(dev, pure_traces(paid_table, JEV_GATE, 0.7)[:3])),
          "traces that miss an item are refused by the simulator")  # fmt: skip

    print("\n-- a pure cascade's steps --")
    thirty = [(f"k{k}", "haiku") for k in range(1, 31)]
    big = GateScoreTable.build(
        [GateRow(i, c, True, None, PaidScore(int(i[1:]) / 100, 0.0, False)) for i, c in thirty]
        + [GateRow("x0", "haiku", False, "code"), GateRow("x1", "sonnet", True, None, PaidScore(0.999, 0.0, False)),
           GateRow("x2", "haiku", True, None, PaidScore(None, 0.0, True))],
        [*thirty, ("x0", "haiku"), ("x1", "sonnet"), ("x2", "haiku")],
    )  # fmt: skip
    want = [k / 100 for k in (1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29)]
    check(cutoff_steps(big, JEV_GATE) == tuple(want),
          "30 distinct scores give the 0%, 5%, ..., 95% lower quantiles, the ceil(1.5k)-th smallest, 20 steps")  # fmt: skip
    check(max(cutoff_steps(big, JEV_GATE)) < 0.999 and 0.999 not in cutoff_steps(big, JEV_GATE),
          "a failed answer, a refused call and another config's answer are no step")  # fmt: skip
    few = GateScoreTable.build(rows_of(jev={i: PaidScore(s, 0.0, False) for i, s in
                                           (("i0", 0.9), ("i1", 0.7), ("i2", 0.6), ("i3", 0.5))}), KEYS)  # fmt: skip
    check(cutoff_steps(few, JEV_GATE) == (0.5, 0.6, 0.7, 0.9), "up to 24 distinct scores are each a step")
    check(cutoff_steps(few, FREE_GATE) == () and cutoff_steps(table, JEV_GATE) == (),
          "the free checks alone have no cutoff, and a gate that scored nothing has no step")  # fmt: skip

    print("\n-- a pure cascade's walk --")
    walk_keys = [(i, c) for i in ITEMS for c in ("haiku", "opus")]

    # M1: opus (A) is right on i0, i1, i2 and wrong on i3: 75% at $10 per 1k. Haiku is right on i0, i2, i3, wrong on i1,
    # at $0.001; opus is $0.010. DRAWS makes i1's and i3's weights equal in every resample.
    def walk_of(gate: str, table_: GateScoreTable, share: float):  # type: ignore[no-untyped-def]
        return walk_candidates(M1, pure_candidates(table_, gate), DRAWS, MIX, share)

    all_pass = GateScoreTable.build(rows_of(), walk_keys)
    w = walk_of(FREE_GATE, all_pass, 1.0)
    check(len(w.steps) == 2 and w.steps[0].threshold is None and w.steps[1].cutoff is None,
          "haiku/free's walk is A, then its single step")  # fmt: skip
    check(math.isclose(w.steps[1].quality, 0.75) and math.isclose(w.steps[1].cost_per_1k, 1.0)
          and w.steps[1].margin == 0 and w.steps[1].clears and w.target == 0.75,
          "all four stay on haiku: 75% (haiku's i1 loss cancels its i3 win) for $1 per 1k, margin 0, clears the 100% target")  # fmt: skip
    check(w.operating is w.steps[1] and not w.fell_back, "its operating point is that step, not A")
    two_fail = GateScoreTable.build(rows_of({"i1": "code", "i3": "code"}), walk_keys)
    w = walk_of(FREE_GATE, two_fail, 1.0)
    check(math.isclose(w.steps[1].quality, 0.75) and math.isclose(w.steps[1].cost_per_1k, 6.0) and w.steps[1].clears,
          "i1 and i3 failing the code check go to opus: 75% for ($2 + 2 x $22) / 4 = $6 per 1k, as A's answers on them")  # fmt: skip
    # Jev scores i0 .9, i1 .7, i2 .6, i3 .5, each call $0.0001. Steps .5, .6, .7, .9, tried .9 first.
    scored = {i: PaidScore(s, 0.0001, False) for i, s in (("i0", 0.9), ("i1", 0.7), ("i2", 0.6), ("i3", 0.5))}
    jev_table = GateScoreTable.build(rows_of(jev=scored), walk_keys)
    w = walk_of(JEV_GATE, jev_table, 1.0)
    check([s.cutoff for s in w.steps] == [None, 0.9, 0.7] and [s.clears for s in w.steps] == [True, True, False],
          "haiku/Jev tries cutoffs strictest first and stops at the first that fails: .9 clears, .7 loses i1")  # fmt: skip
    check(math.isclose(w.steps[1].quality, 0.75) and math.isclose(w.steps[1].cost_per_1k, 8.6),
          "at .9 only i0 stays: ($0.0011 + 3 x $0.0111) / 4 = $8.60 per 1k at 75%")  # fmt: skip
    check(
        w.operating.cutoff == 0.9 and not w.fell_back, "the operating point is the cheapest certified step, cutoff .9"
    )
    dear = {i: PaidScore(s, 0.02, False) for i, s in (("i0", 0.9), ("i1", 0.7), ("i2", 0.6), ("i3", 0.5))}
    w = walk_of(JEV_GATE, GateScoreTable.build(rows_of(jev=dear), walk_keys), 1.0)
    check(w.steps[1].clears and w.steps[1].cost_per_1k > w.steps[0].cost_per_1k and w.fell_back
          and w.operating.cutoff is None, "a gate dearer than escalating falls back to A")  # fmt: skip

    print("\n-- a cascade's section --")
    walks = {"realistic": {s: walk_of(JEV_GATE, jev_table, s) for s in TARGETS},
             "uniform": {s: walk_of(JEV_GATE, jev_table, s) for s in TARGETS}}  # fmt: skip
    section = pure_section(walks, JEV_GATE)
    point = section["realistic"]["100%"]
    check(set(point) == {"cutoff", "margin", "target", "selection_method", "steps", "fell_back"}
          and point["cutoff"] == 0.9 and point["fell_back"] is False and point["target"] == 0.75,
          "a pure cascade's operating point has a cutoff and no threshold")  # fmt: skip
    check([set(s) for s in point["steps"]] == [{"cutoff", "quality", "cost_per_1k", "margin", "clears"}] * 3
          and point["steps"][0]["cutoff"] is None and point["steps"][1]["cutoff"] == 0.9,
          "its steps carry a cutoff each, A's null")  # fmt: skip
    free_walks = {m: {s: walk_of(FREE_GATE, all_pass, s) for s in TARGETS} for m in MIXES}
    free_section = pure_section(free_walks, FREE_GATE)
    free_point = free_section["uniform"]["97%"]
    check(set(free_point) == {"margin", "target", "selection_method", "steps", "fell_back"}
          and all(set(s) == {"quality", "cost_per_1k", "margin", "clears"} for s in free_point["steps"])
          and len(free_point["steps"]) == 2, "haiku/free's section has a single step and neither threshold nor cutoff")  # fmt: skip
    started = build_cascade_section(walks, threshold=True, cutoff=True)["uniform"]["99%"]
    check({"threshold", "cutoff"} <= set(started) and set(started["steps"][1]) >= {"threshold", "cutoff"},
          "a router-started cascade's section has both a threshold and a cutoff")  # fmt: skip
    check(set(build_cascade_section(walks, threshold=True, cutoff=False)["uniform"]["99%"]) ==
          {"threshold", "margin", "target", "selection_method", "steps", "fell_back"},
          "and a router-started one with no paid gate has a threshold alone")  # fmt: skip
    check(refused(OperatingPointsError, lambda: pure_section({}, JEV_GATE))
          and refused(OperatingPointsError, lambda: pure_section({"realistic": {1.0: walks["realistic"][1.0]}}, JEV_GATE)),
          "an empty section, or a mix without all three targets, is refused")  # fmt: skip
    router_walk = walk_candidates(M1, [], DRAWS, MIX, 1.0)
    check(set(build_section({"realistic": {1.0: router_walk, 0.99: router_walk, 0.97: router_walk}})["realistic"]["100%"])
          == {"threshold", "margin", "target", "selection_method", "steps", "fell_back"}
          and set(build_section({"m": {s: router_walk for s in TARGETS}})["m"]["97%"]["steps"][0])
          == {"threshold", "quality", "cost_per_1k", "margin", "clears"},
          "a router's section keeps its fields: threshold alone, no cutoff")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "operating_points.json"
        key_free, key_jev = cascade_section(HAIKU, FREE_GATE), cascade_section(HAIKU, JEV_GATE)
        write_section(key_free, free_section, target)
        write_section(key_jev, section, target)
        check_sections(target, required=(key_free, key_jev))
        check(True, "both pure cascades' sections pass the allowlist")
        check(read_section(key_jev, target) == json.loads(json.dumps(section)), "and read back as written")
        write_section(NAMED_ROUTERS["E"], {}, target)
        check_sections(target, required=(NAMED_ROUTERS["E"], key_free))
        check(True, "beside a router's section, which the writer leaves as it was")

    print("\n-- pure walks on the real mixes --")
    tasks = ("ifeval", "banking77", "mmlu_pro", "mbppplus")
    big_dev = synthetic(
        {f"{t}{n}": (t, {"haiku": (0.001, n % 2), "opus": (0.010, 1)}) for t in tasks for n in range(4)}
    )
    big_keys = [(i, c) for i in big_dev.items for c in ("haiku", "opus")]
    big_rows = [GateRow(i, c, True, None, PaidScore(0.5 + 0.01 * k, 0.0, False) if c == "haiku" else None)
                for k, (i, c) in enumerate(big_keys)]  # fmt: skip
    pure = pure_walks(big_dev, GateScoreTable.build(big_rows, big_keys), JEV_GATE, resample(big_dev))
    check(list(pure) == list(MIXES) and all(set(by_share) == set(TARGETS) for by_share in pure.values()),
          "a walk per mix and target, on the real task mixes")  # fmt: skip
    check(all(w.steps[0].threshold is None and w.steps[0].cutoff is None for by in pure.values() for w in by.values()),
          "each starts at A")  # fmt: skip

    print("\n-- the settings file and the cascades' names --")
    settings = gates.read_gate_settings()
    check([c.key for c in settings.cascades] == list(CASCADES) and len(settings.cascades) == 15
          and all(CASCADES[c.key] == c.label for c in settings.cascades),
          "the file holds the fifteen cascades' keys and labels, in item 8's order")  # fmt: skip
    check(all(c.key == cascade_section(c.start, c.gate) for c in settings.cascades)
          and {c.gate for c in settings.cascades} == {FREE_GATE, JEV_GATE, SONNET_GATE}
          and [c.start for c in settings.cascades[:3]] == [HAIKU] * 3,
          "each shape's key is its start and gate's section key")  # fmt: skip
    pinned = settings.code_check
    check((pinned.image_digest is None) == (pinned.dockerfile_sha256 is None) and {"jev", "sonnet"} <= set(settings.raw),
          "the image digest and Dockerfile hash are both empty until the settings commit and both set after; the paid gates' sections are theirs")  # fmt: skip
    with tempfile.TemporaryDirectory() as folder:
        broken = Path(folder) / "s.json"
        raw = json.loads((PROJECT_ROOT / "routers" / "gate_settings.json").read_text())
        for what, mutate in (("a missing code_check", lambda r: r.pop("code_check")),
                             ("no attempts", lambda r: r["code_check"].update(attempts=0)),
                             ("a time limit of 0", lambda r: r["code_check"].update(timeout_s=0)),
                             ("a cascade without a gate", lambda r: r["cascades"][0].pop("gate"))):  # fmt: skip
            changed = json.loads(json.dumps(raw))
            mutate(changed)
            broken.write_text(json.dumps(changed))
            check(
                refused(gates.SettingsError, lambda: gates.read_gate_settings(broken)),
                f"a settings file with {what} is refused",
            )
        check(
            refused(gates.SettingsError, lambda: gates.read_gate_settings(Path(folder) / "none.json")),
            "and one that isn't there",
        )

    if problems:
        print(f"\n{len(problems)} checks failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
