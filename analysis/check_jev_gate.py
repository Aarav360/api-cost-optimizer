"""Check the Jev gate end to end, on synthetic data and the fake Jev server (PLAN_v4.md §9; Amendment A17, items 5, 7,
13, 14 and 15; A15 items 8 and 9; A16 items 7 and 8).

Run from the project root:  uv run python -m analysis.check_jev_gate

No results/ folder, no Docker, no Jev or Anthropic call, no key, no benchmark answer: the calls go to `analysis.fake_jev`
on a loopback port, with their waits injected, into a temporary folder. Every expected value is worked out by hand.

- the call body equals hand-written JSON, with a null schema and non-ASCII text, in the code's key order;
- the byte budget, from Jev's limits and a measured overhead, at a hand-worked boundary, and the cut of the answer
  first, then the prompt, with the bytes cut counted;
- the gate score: clipped, the mistake statement inverted, averaged on the log-odds scale, turned back;
- refusals, a 429 then a 200, a 400 that stops the run and names the answer;
- the collection: answers from the free-check file, none for a failed answer, a rerun skipping stored answers, the
  noise pass, the key kept out of the files;
- the table's Jev column and the walk of haiku/Jev, and the deployable gate's imports.
"""

import contextlib
import dataclasses
import hashlib
import io
import itertools
import json
import math
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from analysis import collect_gates, collect_jev_gate
from analysis.check_gates import BENCH, DIGEST, GOOD, NOW, OK, SCHEMA, FakeSandbox, write_store
from analysis.check_jev_client import RETRY
from analysis.check_jev_settings import write_benchmark
from analysis.check_routers import imported_packages
from analysis.check_walk import DRAWS, M1, MIX
from analysis.checking import refused
from analysis.collect_gates import GateCollectError, GateInput
from analysis.fake_jev import FakeJev, Reply, noul_answer, response_body
from analysis.gate_table import GateScoreTable, PaidScore, build_table
from analysis.gate_walk import pure_candidates, pure_section
from analysis.operating_points import check_sections, read_section, write_section
from analysis.references import TARGETS
from analysis.router_names import HAIKU, JEV_GATE, cascade_section
from analysis.walk import walk_candidates
from harness.client import PROJECT_ROOT
from routers import jev, jev_call, jev_gate
from routers.request import Request

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def close(a: float | None, b: float, tol: float = 1e-4) -> bool:
    return a is not None and math.isclose(a, b, abs_tol=tol)


# The settings file's byte budget is null until the smoke test and the smoke's number after the settings commit; these
# checks work with the budget left empty, so that a call without one is refused whichever state the file is in.
FILE_SETTINGS = jev_gate.read_settings()
SETTINGS = dataclasses.replace(FILE_SETTINGS, budget_bytes=None)
CORRECT = "This answer correctly and completely addresses the request."
FOLLOWS = "This answer is fully correct and follows every instruction in the request."
MISTAKE = "This answer contains at least one mistake."
QUESTIONS = (
    '{"jev_gate_correct":{"type":"noul","instructions":"' + CORRECT + '"},'
    '"jev_gate_follows":{"type":"noul","instructions":"' + FOLLOWS + '"},'
    '"jev_gate_mistake":{"type":"noul","instructions":"' + MISTAKE + '"}}'
)


KEY = "test-key-NOT-A-REAL-KEY-7f3a"
STAMP = "2026-11-10T12:00:00-08:00"


def client_settings(url: str, price: float = 0.042) -> jev.ClientSettings:
    return jev.parse_client_settings(
        {"jev": {"endpoint": url, "price_per_million_input_tokens": price}, "retry": RETRY}
    )


def new_client(settings: jev.ClientSettings, key: str = KEY) -> tuple[jev.JevClient, list[float]]:
    """A client with its waits recorded, not slept, and a clock ticking 0.25 per reading."""
    waits: list[float] = []
    ticks = itertools.count()
    client = jev.JevClient(settings, {jev.KEY_VARIABLE: key}, sleep=waits.append, clock=lambda: next(ticks) * 0.25,
                           jitter=lambda: 0.0)  # fmt: skip
    return client, waits


def reply(c: float, f: float, m: float, tokens: int = 1000) -> Reply:
    answers = {
        "jev_gate_correct": noul_answer(c),
        "jev_gate_follows": noul_answer(f),
        "jev_gate_mistake": noul_answer(m),
    }
    return Reply(200, response_body(answers, input_tokens=tokens))


def same(score: float, tokens: int = 1000) -> Reply:
    """A response whose gate score is `score`: correct and follows say it, mistake says the opposite."""
    return reply(score, score, round(1 - score, 2), tokens)


def run_collect(
    fake: FakeJev, directory: Path, inputs: Sequence[GateInput], split: str, **options
) -> collect_jev_gate.Summary:  # type: ignore[no-untyped-def]
    client, _ = new_client(client_settings(fake.url))
    with client:
        return collect_jev_gate.collect(inputs, split=split, settings=SETTINGS, client=client,
                                        results_dir=directory, now=lambda: STAMP, **options)  # fmt: skip


def free_check(inputs: Sequence[GateInput], split: str, directory: Path) -> None:
    settings = collect_gates.read_gate_settings().code_check
    collect_gates.collect_free(inputs, split=split, settings=settings, digest=DIGEST, results_dir=directory,
                               run=FakeSandbox(*[OK] * 20), sleep=lambda s: None, now=NOW)  # fmt: skip


def answer_of(
    item: str, config: str, text: str = "x", stop: str | None = "end_turn", prompt: str = "Classify."
) -> GateInput:
    return GateInput(item, config, Request(prompt, None), text, stop)


def collection_checks(directory: Path) -> None:
    print("\n-- the collection --")
    bench = write_benchmark(directory, BENCH)
    store = directory / "store.sqlite"
    write_store(store)
    inputs = collect_gates.gate_inputs("train", bench, store)
    free_check(inputs, "train", directory)
    with FakeJev(script=[same(0.9), same(0.4)]) as fake:
        summary = run_collect(fake, directory, inputs, "train", budget_bytes=5000)
        received = [json.loads(r.body) for r in fake.received]
    check(summary == collect_jev_gate.Summary(new=2, skipped=0, not_called=1, refused=0, cut=0),
          "two answers called; the one that failed the stop check gets no call")  # fmt: skip
    check([(r["state"]["request"]["prompt"], r["state"]["answer"]) for r in received]
          == [(BENCH[0]["prompt"], GOOD), ("Classify.", '{"answer": "a"}')]
          and received[1]["state"]["request"]["output_schema"] == SCHEMA,
          "the answers come from the gates' reader and the requests from the benchmark reader, schema included")  # fmt: skip
    wire = " ".join(json.dumps(r) for r in received)
    check(not any(x in wire for x in ("RUN-ONE-ANSWER", "TEST-ITEM-ANSWER", "EXPECTED-SENTINEL", "no code here")),
          "no run 1 answer, test answer, expected answer or failed answer reaches Jev")  # fmt: skip
    path = collect_jev_gate.jev_path("train", directory)
    records = collect_jev_gate.read_records("train", directory)
    check(
        path == directory / "gates" / "jev_gate_train.jsonl" and len(records) == 2,
        "two records in jev_gate_train.jsonl",
    )
    check(all(tuple(r) == collect_jev_gate.STORED_FIELDS for r in records)
          and tuple(records[0])[:6] == ("item_id", "config", "run", "split", "gate", "pass")
          and [(r["item_id"], r["config"], r["run"], r["split"], r["gate"], r["pass"]) for r in records]
          == [("m:1", "haiku", 0, "train", "jev", 1), ("b:1", "haiku", 0, "train", "jev", 1)],
          "each record is keyed by item, config, run 0, split, gate and pass, then A15 item 9's fields")  # fmt: skip
    check(records[0]["body_sha256"] == hashlib.sha256(fake.received[0].body).hexdigest()
          and records[0]["usage"] == {"input_tokens": 1000, "output_tokens": 38} and records[0]["attempts"] == 1
          and records[0]["answer_cut_bytes"] == 0 and records[0]["refused"] is False
          and records[0]["response"]["answers"]["jev_gate_correct"] == {"type": "noul", "noul": 0.9},
          "a record holds the whole response, usage, sha256, attempts, refusal and cut counts")  # fmt: skip
    check(KEY not in "".join(p.read_text() for p in directory.rglob("*.jsonl")), "the key is in no stored file")
    check(fake.received[0].headers.get("authorization", "").endswith(KEY) or KEY in str(fake.received[0].headers),
          "known positive: the key did reach the server in a header")  # fmt: skip
    with FakeJev() as fake2:
        again = run_collect(fake2, directory, inputs, "train", budget_bytes=5000)
        check(again.new == 0 and again.skipped == 2 and fake2.received == [], "a rerun skips stored answers: no call")
    check(refused(GateCollectError, lambda: run_collect(FakeJev(), directory, inputs, "train")),
          "no budget, given or in the settings, is refused before any call")  # fmt: skip
    check(refused(GateCollectError, lambda: run_collect(FakeJev(), directory, inputs, "test", budget_bytes=5000)),
          "the test split is refused")  # fmt: skip
    check(refused(GateCollectError, lambda: run_collect(FakeJev(), directory, [*inputs, inputs[0]], "train", budget_bytes=5000)),
          "an answer listed twice is refused")  # fmt: skip
    check(refused(GateCollectError, lambda: run_collect(FakeJev(), directory, inputs, "dev", budget_bytes=5000)),
          "answers with no free-check record for the split are refused")  # fmt: skip

    print("\n-- the noise pass --")
    pair = [i for i in inputs if i.item_id == "m:1" and i.config == "haiku"]
    with FakeJev(script=[same(0.8)]) as fake:
        noise = run_collect(fake, directory, pair, "train", pass_number=2, budget_bytes=5000)
    check(noise.new == 1 and [r["pass"] for r in collect_jev_gate.read_records("train", directory)] == [1, 1, 2],
          "a given list is sent again as pass 2, beside pass 1")  # fmt: skip
    column = collect_jev_gate.paid_scores(
        "train", jev_gate.read_settings(), client_settings("http://127.0.0.1:1/x"), directory
    )
    check(close(column[("m:1", "haiku")].score, 0.9) and close(column[("b:1", "haiku")].score, 0.4) and len(column) == 2,
          "the noise pass never enters the table: m:1's score is pass 1's 0.9, not pass 2's 0.8")  # fmt: skip
    with FakeJev() as fake:
        check(run_collect(fake, directory, pair, "train", pass_number=2, budget_bytes=5000).skipped == 1
              and fake.received == [], "a rerun of the noise pass skips what is stored")  # fmt: skip
    with tempfile.TemporaryDirectory() as other:
        fresh = Path(other)
        free_check(inputs, "train", fresh)
        check(refused(GateCollectError, lambda: run_collect(FakeJev(), fresh, pair, "train", pass_number=2, budget_bytes=5000)),
              "pass 2 needs pass 1 stored")  # fmt: skip
        free_check(inputs, "dev", fresh)
        check(refused(GateCollectError, lambda: run_collect(FakeJev(), fresh, pair, "dev", pass_number=2, budget_bytes=5000)),
              "the noise pass is for train")  # fmt: skip

    print("\n-- refusals, retries and run stops --")
    with tempfile.TemporaryDirectory() as other:
        d = Path(other)
        batch = [answer_of(f"a{n}", "haiku", prompt="Q?") for n in range(1, 5)] + [answer_of("a5", "sonnet")]
        batch[1] = answer_of("a2", "haiku", stop="max_tokens")
        free_check(batch, "train", d)
        missing = Reply(200, response_body({"jev_gate_correct": noul_answer(0.5)}, input_tokens=2000))
        script = [same(0.9), Reply(429, headers={"Retry-After": "2"}), same(0.7), same(0.3), missing]
        with FakeJev(script=script) as fake:
            client, waits = new_client(client_settings(fake.url))
            with client:
                summary = collect_jev_gate.collect(batch, split="train", settings=jev_gate.read_settings(), client=client,
                                                   results_dir=d, budget_bytes=5000, now=lambda: STAMP)  # fmt: skip
            check(fake.unexpected == [], "the fake had an answer for every question asked")
        check(summary == collect_jev_gate.Summary(new=4, skipped=0, not_called=1, refused=1, cut=0),
              "four calls, one answer with no call, one refusal")  # fmt: skip
        stored = collect_jev_gate.read_records("train", d)
        check([r["item_id"] for r in stored] == ["a1", "a3", "a4", "a5"] and stored[1]["attempts"] == 2
              and stored[1]["failed_attempts"] == [{"status": 429}] and waits == [2.0],
              "a 429 followed by a 200 is retried through the frozen client: 2 attempts, waited the 2 s it asked")  # fmt: skip
        check(stored[3]["refused"] is True and stored[3]["config"] == "sonnet" and stored[2]["refused"] is False,
              "a response missing a statement is a refusal, by A16 item 7's rule, and is recorded")  # fmt: skip
        column = collect_jev_gate.paid_scores(
            "train", jev_gate.read_settings(), client_settings("http://127.0.0.1:1/x"), d
        )
        check(set(column) == {("a1", "haiku"), ("a3", "haiku"), ("a4", "haiku"), ("a5", "sonnet")}
              and column[("a5", "sonnet")] == PaidScore(None, 2000 * 0.042 / 1e6, True),
              "the refusal is a table entry with no score, charged its input tokens, flagged")  # fmt: skip
        check(close(column[("a1", "haiku")].score, 0.9) and close(column[("a3", "haiku")].score, 0.7)
              and column[("a1", "haiku")].overhead == 1000 * 0.042 / 1e6,
              "scores come from the stored responses, the overhead from input tokens at $0.042 per million")  # fmt: skip
        stored_path = collect_jev_gate.jev_path("train", d)
        lines = stored_path.read_text().splitlines()
        stored_path.write_text(lines[0] + "\n")
        check(refused(GateCollectError, lambda: collect_jev_gate.paid_scores("train", jev_gate.read_settings(),
                                                                              client_settings("http://127.0.0.1:1/x"), d)),
              "a table is refused whole when an answer that passed the free checks has no call")  # fmt: skip

    with tempfile.TemporaryDirectory() as other:
        d = Path(other)
        one = [answer_of("s1", "haiku", prompt="Q?"), answer_of("s2", "haiku", prompt="Q2?")]
        free_check(one, "dev", d)
        with FakeJev(script=[same(0.9), Reply(400, {"detail": "bad"})]) as fake:
            try:
                run_collect(fake, d, one, "dev", budget_bytes=5000)
                stop = ""
            except jev.RunStop as error:
                stop = str(error)
        check("s2" in stop and "haiku" in stop and "s1" not in stop, "a 400 stops the run, naming the answer")
        check([r["item_id"] for r in collect_jev_gate.read_records("dev", d)] == ["s1"],
              "what was stored before the stop is valid")  # fmt: skip
        with FakeJev(script=[same(0.2)]) as fake:
            again = run_collect(fake, d, one, "dev", budget_bytes=5000)
        check(again.new == 1 and again.skipped == 1, "and a rerun resumes after it")
        long_one = [answer_of("c1", "haiku", text="A" * 100, prompt="P" * 10)]
        free_check(long_one, "smoke", d)
        with FakeJev(script=[same(0.5)]) as fake:
            summary = run_collect(fake, d, long_one, "smoke", budget_bytes=50)
            sent = json.loads(fake.received[0].body)["state"]
        cut_record = collect_jev_gate.read_records("smoke", d)[0]
        check(summary.cut == 1 and sent["answer"] == "A" * 40 and cut_record["answer_cut_bytes"] == 60
              and cut_record["prompt_cut_bytes"] == 0,
              "a 100-byte answer in a 50-byte budget with a 10-byte prompt keeps 20 + 20 bytes; 60 are recorded as cut")  # fmt: skip

    print("\n-- the command line --")
    with tempfile.TemporaryDirectory() as other:
        d = Path(other)
        made_up = [answer_of("k1", "haiku", prompt="Q?"), answer_of("k2", "haiku", prompt="Q2?")]
        free_check(made_up, "smoke", d)
        smoke_file = d / "made_up.json"
        smoke_file.write_text(json.dumps([{"item_id": i.item_id, "config": i.config, "prompt": i.request.prompt,
                                           "output_schema": None, "answer": i.answer, "stop_reason": i.stop_reason}
                                          for i in made_up]))  # fmt: skip
        with FakeJev(script=[same(0.6), Reply(400, {"detail": "bad"})]) as fake:
            client_file = d / "client.json"
            client_file.write_text(json.dumps({"jev": {"endpoint": fake.url, "price_per_million_input_tokens": 0.042},
                                               "retry": RETRY}))  # fmt: skip
            argv = ["collect", "--split", "smoke", "--smoke-file", str(smoke_file), "--budget-bytes", "5000",
                    "--results-dir", str(d), "--client-settings", str(client_file)]  # fmt: skip
            with contextlib.redirect_stderr(io.StringIO()) as message:
                code = collect_jev_gate.main(argv, {jev.KEY_VARIABLE: KEY})
            check(code == 1 and "k2" in message.getvalue() and KEY not in message.getvalue(),
                  "a run stop exits 1, naming the answer and not the key")  # fmt: skip
        check(len(collect_jev_gate.read_records("smoke", d)) == 1, "and leaves the first record stored")
        with contextlib.redirect_stderr(io.StringIO()):
            check(collect_jev_gate.main(argv, {}) == 1, "a missing key stops the run, exit 1")


def walk_checks() -> None:
    print("\n-- the table's Jev column and haiku/Jev's walk --")
    items = ("i0", "i1", "i2", "i3")
    inputs = [answer_of(i, c, prompt=f"Q {i}") for i in items for c in ("haiku", "opus")]
    haiku_scores = {"i0": 0.9, "i1": 0.7, "i2": 0.6, "i3": 0.5}
    script = [same(haiku_scores[i] if c == "haiku" else 0.99) for i in items for c in ("haiku", "opus")]
    with tempfile.TemporaryDirectory() as tmp, FakeJev(script=script) as fake:
        d = Path(tmp)
        free_check(inputs, "dev", d)
        run_collect(fake, d, inputs, "dev", budget_bytes=5000)
        # $0.1 per million and 1,000 input tokens make each call cost $0.0001, as in check_gates' hand-worked walk.
        column = collect_jev_gate.paid_scores("dev", jev_gate.read_settings(), client_settings(fake.url, price=0.1), d)
        table = build_table(collect_gates.read_free_records("dev", d), [(i.item_id, i.config) for i in inputs],
                            {JEV_GATE: column})  # fmt: skip
        check(isinstance(table, GateScoreTable) and all(table.row(i, "haiku").jev is not None for i in items)
              and close(table.row("i1", "haiku").jev.score, 0.7)  # type: ignore[union-attr]
              and math.isclose(table.row("i1", "opus").jev.overhead, 0.0001),  # type: ignore[union-attr]
              "the table's Jev column carries the score and the overhead of every answer, opus's included")  # fmt: skip
        walk = walk_candidates(M1, pure_candidates(table, JEV_GATE), DRAWS, MIX, 1.0)
        # A score is a float computed from the log-odds (0.9 comes back as 0.8999999999999999), so cutoffs are compared
        # to 12 places; the walk itself compares a score with a cutoff taken from the same table, exactly.
        check([None if s.cutoff is None else round(s.cutoff, 12) for s in walk.steps] == [None, 0.9, 0.7]
              and [s.clears for s in walk.steps] == [True, True, False],
              "the walk runs from A, strictest cutoff first: 0.9 clears, 0.7 loses i1")  # fmt: skip
        check(math.isclose(walk.steps[1].quality, 0.75) and math.isclose(walk.steps[1].cost_per_1k, 8.6)
              and close(walk.operating.cutoff, 0.9, 1e-12) and not walk.fell_back,
              "at 0.9 only i0 stays: ($0.0011 + 3 x $0.0111) / 4 = $8.60 per 1k; the cheapest certified step is 0.9")  # fmt: skip
        walks = {"realistic": {t: walk_candidates(M1, pure_candidates(table, JEV_GATE), DRAWS, MIX, t) for t in TARGETS},
                 "uniform": {t: walk_candidates(M1, pure_candidates(table, JEV_GATE), DRAWS, MIX, t) for t in TARGETS}}  # fmt: skip
        section = pure_section(walks, JEV_GATE)
        target = d / "operating_points_temp.json"
        key = cascade_section(HAIKU, JEV_GATE)
        write_section(key, section, target)
        check_sections(target, required=(key,))
        check(
            close(read_section(key, target)["realistic"]["100%"]["cutoff"], 0.9, 1e-12),
            "haiku/Jev's section is written to a temp file",
        )
    # the same walk with the fallback: a gate dearer than escalating (price $200 per million = $0.2 a call)
    with tempfile.TemporaryDirectory() as tmp, FakeJev(script=list(script)) as fake:
        d = Path(tmp)
        free_check(inputs, "dev", d)
        run_collect(fake, d, inputs, "dev", budget_bytes=5000)
        dear = collect_jev_gate.paid_scores("dev", jev_gate.read_settings(), client_settings(fake.url, price=200.0), d)
        table = build_table(collect_gates.read_free_records("dev", d), [(i.item_id, i.config) for i in inputs],
                            {JEV_GATE: dear})  # fmt: skip
        walk = walk_candidates(M1, pure_candidates(table, JEV_GATE), DRAWS, MIX, 1.0)
        check(walk.fell_back and walk.operating.cutoff is None, "a gate dearer than escalating falls back to A")


def imports_checks() -> None:
    print("\n-- the deployable gate's imports --")
    found = imported_packages(PROJECT_ROOT / "routers" / "jev_gate.py")
    check(found <= {"json", "math", "collections", "dataclasses", "pathlib", "typing", "routers"},
          f"routers/jev_gate.py imports {sorted(found)}: the standard library and the deployable package only")  # fmt: skip
    check(not found & {"analysis", "harness", "anthropic", "sklearn", "torch"}, "and nothing from analysis or harness")


def main() -> int:
    print("-- the settings --")
    check(SETTINGS.model == "jev-1.14.0", "the model is jev-1.14.0")
    check([s.id for s in SETTINGS.statements] == ["jev_gate_correct", "jev_gate_follows", "jev_gate_mistake"]
          and [s.inverted for s in SETTINGS.statements] == [False, False, True],
          "three statements under their IDs, the last read as 1 - p")  # fmt: skip
    check([s.text for s in SETTINGS.statements] == [CORRECT, FOLLOWS, MISTAKE], "with A17 item 5's texts")
    check(SETTINGS.model == jev_call.read_settings().model, "and it is the model the frozen Jev settings name")
    check(SETTINGS.clip == (0.005, 0.995) and (FILE_SETTINGS.budget_bytes is None or FILE_SETTINGS.budget_bytes >= 1),
          "the clip is [0.005, 0.995] and the file's byte budget is empty until the smoke test, a whole number after")  # fmt: skip

    raw = json.loads((PROJECT_ROOT / "routers" / "gate_settings.json").read_text())["jev"]
    for what, mutate in (
        ("two statements", lambda r: r["statements"].pop()),
        ("a repeated statement ID", lambda r: r["statements"][1].update(id="jev_gate_correct")),
        ("a clip of 0 to 1", lambda r: r.update(clip=[0, 1])),
        ("a budget of 0", lambda r: r.update(budget_bytes=0)),
        ("no model", lambda r: r.pop("model")),
        ("an inverted flag that is not a bool", lambda r: r["statements"][2].update(inverted=1)),
    ):
        changed = json.loads(json.dumps(raw))
        mutate(changed)
        check(
            refused(jev_gate.GateCallError, lambda: jev_gate.parse_settings(changed)),
            f"settings with {what} are refused",
        )

    print("\n-- the gate score --")
    cases = [
        ((0.5, 0.5, 0.5), 0.5, "all 0.5 stay 0.5"),
        ((0.9, 0.9, 0.9), 0.6753, "0.9, 0.9 and a mistake of 0.9 (read 0.1): odds 9, 9, 1/9, geometric mean 9^(1/3)"),
        ((0.8, 0.8, 0.2), 0.8, "a mistake of 0.2 reads 0.8, so all three agree"),
        ((1.00, 1.00, 0.00), 0.995, "1.00, 1.00 and a mistake of 0.00 clip to 0.995 each"),
        ((0.00, 0.00, 1.00), 0.005, "0.00, 0.00 and a mistake of 1.00 clip to 0.005 each"),
        ((1.00, 0.00, 0.50), 0.5, "0.995 and 0.005 have odds 199 and 1/199, which cancel against a neutral third"),
    ]
    for (c, f, m), want, what in cases:
        answers = {"jev_gate_correct": c, "jev_gate_follows": f, "jev_gate_mistake": m}
        check(close(jev_gate.gate_score(answers, SETTINGS), want), what)
    check(jev_gate.gate_score({"jev_gate_correct": 0.9, "jev_gate_follows": 0.9, "jev_gate_mistake": 0.9}, SETTINGS)
          != jev_gate.gate_score({"jev_gate_correct": 0.9, "jev_gate_follows": 0.9, "jev_gate_mistake": 0.1}, SETTINGS),
          "the mistake statement is inverted, not averaged as it is")  # fmt: skip

    print("\n-- the call body --")
    request = Request("Écris f, 日本", None)
    answer = 'say "hi"\n'
    call = jev_gate.gate_call(request, answer, SETTINGS, budget_bytes=1000)
    expected = (
        '{"model":"jev-1.14.0","state":{"request":{"prompt":"Écris f, 日本","output_schema":null},'
        '"answer":"say \\"hi\\"\\n"},"questions":' + QUESTIONS + "}"
    ).encode("utf-8")
    check(
        call.body == expected, "the body equals hand-written JSON: null schema, non-ASCII kept, request before answer"
    )
    check(call.answer_cut_bytes == 0 and call.prompt_cut_bytes == 0, "nothing cut within the budget")
    schema = {"type": "object", "properties": {"zeta": {"type": "string"}, "alpha": {"type": "integer"}}}
    with_schema = jev_gate.gate_call(Request("P", schema), "A", SETTINGS, budget_bytes=1000)
    check(b'"request":{"prompt":"P","output_schema":{"type":"object","properties":{"zeta":{"type":"string"},'
          b'"alpha":{"type":"integer"}}}},"answer":"A"}' in with_schema.body,
          "a schema goes in whole, its keys in the benchmark's order, not sorted")  # fmt: skip
    check(call.sha256 == hashlib.sha256(expected).hexdigest(), "the body's sha256 is stored with it")

    print("\n-- the byte budget --")
    # The state with an empty prompt and answer and a null schema is 58 bytes; the questions are 113, 128 and 96 bytes
    # as one-question objects (337 in all). With a measured 291 tokens of overhead: the 32K limit leaves
    # 32000 - (291 + 58) - 128 = 31523, the 64K limit 64000 - 349 - 337 = 63314, and the tighter is 31523.
    check(jev_gate.gate_budget_bytes(SETTINGS, overhead_tokens=291, schema=None) == 31523,
          "the budget is the tighter limit less the overhead, the empty state and the questions: 31523")  # fmt: skip
    # A schema of 58 bytes makes the empty state 112, so 32000 - (291 + 112) - 128 = 31469.
    sch = {"type": "object", "properties": {"label": {"type": "string"}}}
    check(jev_gate.gate_budget_bytes(SETTINGS, overhead_tokens=291, schema=sch) == 31469,
          "a 58-byte schema is reserved: 31469")  # fmt: skip
    check(refused(jev_gate.GateCallError, lambda: jev_gate.gate_budget_bytes(SETTINGS, overhead_tokens=40000, schema=None)),
          "an overhead that leaves no room is refused")  # fmt: skip
    check(refused(jev_gate.GateCallError, lambda: jev_gate.gate_call(request, answer, SETTINGS)),
          "a call with no budget, given or in the settings, is refused")  # fmt: skip

    print("\n-- the cut --")
    p10, a20 = "p" * 10, "a" * 20
    fit = jev_gate.fit_to_budget(p10, a20, 30)
    check((fit.prompt, fit.answer, fit.answer_cut_bytes, fit.prompt_cut_bytes) == (p10, a20, 0, 0),
          "prompt and answer of 10 + 20 bytes in a budget of 30 are not cut")  # fmt: skip
    fit = jev_gate.fit_to_budget(p10, a20, 29)
    check(fit.prompt == p10 and fit.answer == "a" * 18 and fit.answer_cut_bytes == 2 and fit.prompt_cut_bytes == 0,
          "one byte over: the answer alone is cut to its room of 19, 9 from each end, 2 bytes dropped")  # fmt: skip
    head_tail = "".join(chr(ord("a") + n) for n in range(20))
    fit = jev_gate.fit_to_budget(p10, head_tail, 20)
    check(fit.answer == "abcde" + "pqrst" and fit.answer_cut_bytes == 10,
          "the answer keeps its head and its tail (5 and 5 of 10 bytes of room)")  # fmt: skip
    fit = jev_gate.fit_to_budget("q" * 30, "z" * 5, 30)
    check(fit.prompt == "q" * 30 and fit.answer == "" and fit.answer_cut_bytes == 5 and fit.prompt_cut_bytes == 0,
          "a prompt that fills the budget leaves the answer no room, and the prompt is not cut")  # fmt: skip
    fit = jev_gate.fit_to_budget("q" * 40, "z" * 20, 30)
    check(fit.answer == "" and fit.answer_cut_bytes == 20 and fit.prompt == "q" * 30 and fit.prompt_cut_bytes == 10,
          "a prompt over the budget loses the whole answer first, then is cut itself: 10 bytes")  # fmt: skip
    fit = jev_gate.fit_to_budget('"é"', "xyz", 8)
    check(fit.prompt == '"é"' and fit.answer == "xz" and fit.answer_cut_bytes == 1,
          "a quote counts 2 bytes and é counts 2, so a prompt of 6 leaves room for 2 of the answer's 3 bytes")  # fmt: skip

    with tempfile.TemporaryDirectory() as tmp:
        collection_checks(Path(tmp))
    walk_checks()
    imports_checks()
    return 0


if __name__ == "__main__":
    code = main()
    if problems:
        print(f"\n{len(problems)} checks failed")
        sys.exit(1)
    print("\nall checks passed")
    sys.exit(code)
