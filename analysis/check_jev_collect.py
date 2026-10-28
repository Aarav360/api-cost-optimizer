"""Check G+ end to end and the Jev collection script (PLAN_v4.md §8 3c and 3c+; Amendment A15, items 9, 10, 12 and
13; issue #36).

Run from the project root:  uv run python -m analysis.check_jev_collect

Never a test item and never a Jev call: every request is made-up text or a row of a synthetic benchmark file, every
call goes to the fake Jev server (analysis/fake_jev.py) on a loopback port, the results folder is a temporary one, and
the key is a dummy. Retry waits are injected, so no check sleeps.

- G+ end to end: `score(request)` through the fake server gives hand-worked router scores, and a refusal gives 0;
- the stored record: every field of #30's record, in its order, with hand-checked values;
- resuming: a rerun never calls a stored request again, a torn last line is repaired, and a damaged file is refused;
- stopping: the run stops at a 4xx and at the retry cap, names the request, and what was stored stays valid;
- the noise pass: the sample's second send as pass 2, never over pass 1, and refused without a pass 1;
- the smoke split: made-up requests stored apart, and none reach the train or dev records;
- the sources: test refused, the reader's other refusals, and each malformed made-up request refused whole;
- stored records parse into the router scores a live call gave, for G and for G+, and refusals are counted from them;
- the key: a dummy key reaches the server and is in no stored byte;
- usage: overhead from a stored record, and the byte budget the stored usage fixes;
- the command line: an explicit `--budget-bytes` while the settings file's are null, made-up requests, the noise pass,
  dev, the stop's exit code, and each refused combination.
"""

import contextlib
import io
import itertools
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from analysis import collect_jev as cj
from analysis.benchmark import BenchmarkError, SealedItemError
from analysis.check_jev_client import KEY, synthetic_data
from analysis.check_jev_plus import synthetic_answers, synthetic_weights
from analysis.fake_jev import FakeJev, Reply, choice_answer, noul_answer, response_body
from routers import jev, jev_call
from routers.request import Request

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(error: type[Exception], call: Callable[[], object]) -> bool:
    try:
        call()
    except error:
        return True
    return False


G_ANSWERS = {
    "success_haiku": noul_answer(0.91),
    "success_sonnet": noul_answer(0.62),
    "success_opus_low": noul_answer(0.33),
}


def plus_answers() -> dict[str, dict]:
    """The fake's answer to every G+ question, from check_jev_plus's hand-worked answers."""
    answers = synthetic_answers()
    options = answers.pop("request_type")
    made = {name: noul_answer(value) for name, value in answers.items()}
    return made | {"request_type": choice_answer(options)}


ALL_ANSWERS = G_ANSWERS | plus_answers()
STORED_FIELDS = [
    "router", "split", "item_id", "pass", "body_sha256", "response", "model", "usage", "request_id", "server_date",
    "seconds", "attempts", "failed_attempts", "new_connection", "refused", "collected_at",
]  # fmt: skip  # issue #30's stored record, in its order, written out here and not taken from the script
STAMP = "2026-10-28T09:30:00+01:00"
SECRET = "sentinel-expected-answer-91c2"  # an expected answer: no body may ever carry it
ITEMS = [
    cj.Item("train-1", Request("First made-up request.", {"type": "object", "z": 1, "a": 2})),
    cj.Item("train-2", Request("Zweite Anfrage: Süß 日本")),
    cj.Item("train-3", Request("Third made-up request.")),
]


def settings_data(url: str, budgets: tuple[int | None, int | None] = (1000, 2000)) -> dict:
    """A settings file in #30's format on the fake server's URL, with a noise sample over two of ITEMS."""
    data = synthetic_data(url)
    data["budget_bytes"] = {"jev": budgets[0], "jev_bolstered": budgets[1]}
    data["noise_sample"] = {"seed": 2026, "item_ids": ["train-1", "train-3"]}
    return data


def settings_file(directory: Path, url: str, budgets: tuple[int | None, int | None] = (1000, 2000)) -> Path:
    path = directory / "settings.json"
    path.write_text(json.dumps(settings_data(url, budgets)), encoding="utf-8")
    return path


def client_on(loaded: cj.Loaded, key: str = KEY) -> tuple[jev.JevClient, list[float]]:
    """A client with its waits recorded, not slept, and a clock ticking 0.25 per reading (an attempt takes 0.25 s)."""
    waits: list[float] = []
    ticks = itertools.count()
    client = jev.JevClient(
        loaded.client, {jev.KEY_VARIABLE: key}, sleep=waits.append, clock=lambda: next(ticks) * 0.25, jitter=lambda: 0.0
    )
    return client, waits


def run(
    fake: FakeJev,
    directory: Path,
    router: str,
    items: list[cj.Item],
    *,
    split: str = "train",
    pass_number: int = 1,
    budget_bytes: int | None = None,
    budgets: tuple[int | None, int | None] = (1000, 2000),
) -> cj.Summary:
    """One collection into `directory` as the results folder, on a fresh client, as a rerun would be."""
    loaded = cj.load_settings(settings_file(directory, fake.url, budgets))
    client, _ = client_on(loaded)
    with client:
        return cj.collect(
            items, router=router, split=split, loaded=loaded, client=client, results_dir=directory,
            budget_bytes=budget_bytes, pass_number=pass_number, now=lambda: STAMP,
        )  # fmt: skip


def stored(directory: Path, router: str, split: str = "train") -> list[dict]:
    return cj.read_records(router, split, directory)


def file_of(directory: Path, router: str, split: str = "train") -> Path:
    return directory / "jev" / f"{router}_{split}.jsonl"


def stops(error: type[jev.RunStop], call: Callable[[], object]) -> str | None:
    try:
        call()
    except error as stop:
        return str(stop)
    return None


def section_plus() -> None:
    print("-- score(request) for G+, against the fake server --")
    request = Request("A made-up request.", {"type": "object"})
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        loaded = cj.load_settings(settings_file(Path(folder), fake.url))
        client, _ = client_on(loaded)
        weights = jev.parse_weights(synthetic_weights(), loaded.call)
        router = jev.GPlusRouter(loaded.call, weights, client)
        asked = router.ask(request, "item-1")
        got = [asked.scores["haiku"], asked.scores["sonnet"], asked.scores["opus_low"]]
        # Hand-worked in analysis/check_jev_plus.py from the same answers and weights.
        want = [0.885197, 0.509474, 0.257565]
        check(
            list(asked.scores) == ["haiku", "sonnet", "opus_low"]
            and all(abs(g - w) < 1e-6 for g, w in zip(got, want, strict=True)),
            "G+'s router scores through the fake server are hand-worked: 0.885197, 0.509474, 0.257565",
        )
        check(router.score(request) == asked.scores, "score(request) gives the same router scores")
        check(
            fake.received[0].body == jev_call.g_plus_call(request, loaded.call).body and fake.unexpected == [],
            "the body that reached Jev is G+'s built body, and the fake was asked nothing it didn't expect",
        )
        check(router.counts == jev.Counts(calls=2, refused=0, cut=0), "calls were counted")
    short = {name: answer for name, answer in ALL_ANSWERS.items() if name != "difficulty_3"}
    with tempfile.TemporaryDirectory() as folder, FakeJev(short, [Reply(200, response_body(short))]) as fake:
        loaded = cj.load_settings(settings_file(Path(folder), fake.url))
        client, _ = client_on(loaded)
        router = jev.GPlusRouter(loaded.call, jev.parse_weights(synthetic_weights(), loaded.call), client)
        asked = router.ask(request)
        check(
            asked.scores == {"haiku": 0.0, "sonnet": 0.0, "opus_low": 0.0}
            and asked.result.refused
            and router.counts.refused == 1,
            "a 200 that leaves a G+ question unanswered is a refusal: 0 for every config, flagged and counted",
        )


def section_storage() -> None:
    print("\n-- the stored record --")
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        directory = Path(folder)
        loaded = cj.load_settings(settings_file(directory, fake.url))
        summary = run(fake, directory, "jev", ITEMS)
        records = stored(directory, "jev")
        check(file_of(directory, "jev").is_file(), "records go to <results>/jev/<router>_<split>.jsonl")
        check(
            summary == cj.Summary(new=3, skipped=0, refused=0, cut=0) and len(records) == 3,
            "three requests, three records, none skipped, refused or cut",
        )
        check(
            all(list(r) == STORED_FIELDS for r in records),
            "every record holds every field of #30's stored record, in its order, and no other",
        )
        first = records[0]
        built = jev_call.g_call(ITEMS[0].request, loaded.call)
        check(
            first["router"] == "jev"
            and first["split"] == "train"
            and first["item_id"] == "train-1"
            and first["pass"] == 1,
            "router, split, item ID and pass 1",
        )
        check(
            first["body_sha256"] == built.sha256 and fake.received[0].body == built.body,
            "the stored hash is the sha256 of the body that reached Jev",
        )
        check(
            first["response"] == response_body(G_ANSWERS)
            and first["model"] == "jev-test-0"
            and first["usage"] == {"input_tokens": 300, "output_tokens": 38}
            and first["server_date"] == "Wed, 28 Oct 2026 00:01:57 GMT"
            and first["request_id"] == "req_" + "0" * 31 + "1",
            "the whole response, model ID, usage, request ID and server Date are stored",
        )
        check(
            first["seconds"] == 0.25
            and first["attempts"] == 1
            and first["failed_attempts"] == []
            and first["new_connection"] is True
            and records[1]["new_connection"] is False
            and first["refused"] is False
            and first["collected_at"] == STAMP,
            "seconds, attempts, failures, the new-connection flag (the first call only), refusal and local time",
        )
        check(fake.connections == 1 and fake.unexpected == [], "the whole run used one kept-open connection")
        text = file_of(directory, "jev").read_text(encoding="utf-8")
        check(
            text.count("\n") == 3 and all(isinstance(json.loads(line), dict) for line in text.splitlines()),
            "one JSON object per line, each ending in a newline",
        )
        check(
            "Süß 日本".encode() in fake.received[1].body,
            "a request's non-ASCII text reached Jev as UTF-8, kept as it is (A15 item 8)",
        )
    print("\n-- G+ records --")
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        directory = Path(folder)
        loaded = cj.load_settings(settings_file(directory, fake.url))
        summary = run(fake, directory, "jev_bolstered", ITEMS)
        records = stored(directory, "jev_bolstered")
        built = jev_call.g_plus_call(ITEMS[1].request, loaded.call)
        check(
            summary.new == 3
            and all(list(r) == STORED_FIELDS and r["router"] == "jev_bolstered" for r in records)
            and records[1]["body_sha256"] == built.sha256
            and not file_of(directory, "jev").exists(),
            "G+ is stored in its own file with G+'s body hash and every field, with no weights needed to collect",
        )
        check(
            all(SECRET not in body.body.decode() for body in fake.received), "no body carries anything but the request"
        )


def section_resume() -> None:
    print("\n-- resuming --")
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        directory = Path(folder)
        run(fake, directory, "jev", ITEMS)
        before = file_of(directory, "jev").read_bytes()
        calls = len(fake.received)
        again = run(fake, directory, "jev", ITEMS)
        check(
            again == cj.Summary(new=0, skipped=3, refused=0, cut=0)
            and len(fake.received) == calls
            and file_of(directory, "jev").read_bytes() == before,
            "a rerun skips every stored request, calls none of them again and changes nothing",
        )
        more = ITEMS + [cj.Item("train-4", Request("Fourth made-up request."))]
        grown = run(fake, directory, "jev", more)
        check(
            grown.new == 1
            and grown.skipped == 3
            and len(fake.received) == calls + 1
            and file_of(directory, "jev").read_bytes().startswith(before),
            "a rerun with one more request calls only that one and appends it after the stored lines",
        )
        # A write cut short leaves no newline. Reading drops it, and the next append trims it first.
        path = file_of(directory, "jev")
        intact = path.read_bytes()
        with path.open("ab") as file:
            file.write(b'{"router": "jev", "split": "tr')
        check(len(stored(directory, "jev")) == 4, "a torn last line is dropped when reading")
        extra = more + [cj.Item("train-5", Request("Fifth made-up request."))]
        run(fake, directory, "jev", extra)
        text = path.read_bytes()
        check(
            text.startswith(intact) and b'"tr\n' not in text and len(stored(directory, "jev")) == 5,
            "the next append trims the torn line first, so the file stays valid",
        )
    print("\n-- a damaged file is refused whole --")
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        directory = Path(folder)
        run(fake, directory, "jev", ITEMS[:1])
        path = file_of(directory, "jev")
        good = path.read_text(encoding="utf-8")
        path.write_text(good + good, encoding="utf-8")
        check(
            refused(cj.CollectError, lambda: stored(directory, "jev")),
            "a repeated item and pass is refused: a stored response is never overwritten",
        )
        record = json.loads(good)
        path.write_text(json.dumps({k: v for k, v in record.items() if k != "usage"}) + "\n", encoding="utf-8")
        check(refused(cj.CollectError, lambda: stored(directory, "jev")), "a record missing a field is refused")
        path.write_text(json.dumps(record | {"router": "jev_bolstered"}) + "\n", encoding="utf-8")
        check(refused(cj.CollectError, lambda: stored(directory, "jev")), "a record of another router is refused")
        path.write_text(json.dumps(record | {"refused": "no"}) + "\n", encoding="utf-8")
        check(refused(cj.CollectError, lambda: stored(directory, "jev")), "a refusal flag that isn't a bool is refused")
        path.write_text(json.dumps(record | {"pass": 3}) + "\n", encoding="utf-8")
        check(refused(cj.CollectError, lambda: stored(directory, "jev")), "a pass other than 1 or 2 is refused")
        path.write_text(good, encoding="utf-8")
        check(len(stored(directory, "jev")) == 1, "known positive: the undamaged file is read")
        path.write_text("not json\n", encoding="utf-8")
        check(refused(cj.CollectError, lambda: stored(directory, "jev")), "a line that isn't JSON is refused")
        calls = len(fake.received)
        check(
            refused(cj.CollectError, lambda: run(fake, directory, "jev", ITEMS)) and len(fake.received) == calls,
            "a collection into a damaged file is refused before any call",
        )


def section_stop() -> None:
    print("\n-- stopping --")
    good = Reply(200, response_body(G_ANSWERS))
    for status in (400, 401, 403, 422):
        with (
            tempfile.TemporaryDirectory() as folder,
            FakeJev(ALL_ANSWERS, [good, Reply(status, {"detail": "no"})]) as fake,
        ):
            directory = Path(folder)
            message = stops(jev.Rejected, lambda: run(fake, directory, "jev", ITEMS))
            check(
                message is not None and "jev train train-2" in message and str(status) in message,
                f"a {status} stops the run and names the request",
            )
            check(
                [r["item_id"] for r in stored(directory, "jev")] == ["train-1"] and len(fake.received) == 2,
                f"after a {status} the record stored before it is valid and nothing after it was called",
            )
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS, [good, *[Reply(500)] * 8]) as fake:
        directory = Path(folder)
        message = stops(jev.RetryCapReached, lambda: run(fake, directory, "jev", ITEMS))
        check(
            message is not None and "train-2" in message and "8 attempts" in message,
            "eight failed attempts stop the run at the retry cap and name the request",
        )
        before = file_of(directory, "jev").read_bytes()
        done = run(fake, directory, "jev", ITEMS)
        check(
            done.new == 2 and done.skipped == 1 and file_of(directory, "jev").read_bytes().startswith(before),
            "the rerun after a stop resumes: the stored request is skipped, the stopped one and the rest are sent",
        )
        check(
            stored(directory, "jev")[1]["item_id"] == "train-2" and stored(directory, "jev")[1]["attempts"] == 1,
            "the resumed request is stored once, as a first attempt",
        )
    with (
        tempfile.TemporaryDirectory() as folder,
        FakeJev(ALL_ANSWERS, [Reply(429, None, {"Retry-After": "2"}), good]) as fake,
    ):
        directory = Path(folder)
        loaded = cj.load_settings(settings_file(directory, fake.url))
        client, waits = client_on(loaded)
        with client:
            cj.collect(ITEMS[:1], router="jev", split="train", loaded=loaded, client=client, results_dir=directory)
        record = stored(directory, "jev")[0]
        check(
            record["attempts"] == 2 and record["failed_attempts"] == [{"status": 429}] and waits == [2.0],
            "a retried call stores its attempt count and its failed attempts, and Retry-After was honoured",
        )


def section_noise() -> None:
    print("\n-- the noise pass --")
    sample = [ITEMS[0], ITEMS[2]]
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        directory = Path(folder)
        calls = len(fake.received)
        message = refused(cj.CollectError, lambda: run(fake, directory, "jev", sample, pass_number=2))
        check(message and len(fake.received) == calls, "pass 2 is refused before any call if pass 1 isn't stored")
        run(fake, directory, "jev", ITEMS)
        pass_one = file_of(directory, "jev").read_bytes()
        summary = run(fake, directory, "jev", sample, pass_number=2)
        records = stored(directory, "jev")
        check(
            summary.new == 2
            and [(r["item_id"], r["pass"]) for r in records] == [("train-1", 1), ("train-2", 1), ("train-3", 1),
                                                                 ("train-1", 2), ("train-3", 2)],
            "the sample is sent again as pass 2, beside its pass 1",
        )  # fmt: skip
        check(
            file_of(directory, "jev").read_bytes().startswith(pass_one),
            "pass 2 never overwrites pass 1: the stored pass 1 lines are byte for byte as they were",
        )
        check(
            records[3]["body_sha256"] == records[0]["body_sha256"] and records[3]["collected_at"] == STAMP,
            "a pass 2 record is the same call as its pass 1 (the same body hash)",
        )
        calls = len(fake.received)
        again = run(fake, directory, "jev", sample, pass_number=2)
        check(
            again.new == 0 and again.skipped == 2 and len(fake.received) == calls,
            "a rerun of the noise pass sends nothing",
        )
        check(
            refused(cj.CollectError, lambda: run(fake, directory, "jev", sample, split="dev", pass_number=2)),
            "the noise pass is for train items only: dev is refused",
        )
        check(refused(cj.CollectError, lambda: run(fake, directory, "jev", sample, pass_number=3)), "no pass 3")


def section_smoke() -> None:
    print("\n-- the smoke split --")
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        directory = Path(folder)
        made_up = directory / "made_up.json"
        made_up.write_text(
            json.dumps(
                [{"prompt": "Made-up one."}, {"prompt": "Made-up two.", "item_id": "past-7", "output_schema": {"a": 1}}]
            ),
            encoding="utf-8",
        )
        items = cj.requests_for("smoke", made_up)
        check(
            [i.item_id for i in items] == ["smoke-001", "past-7"]
            and items[0].request == Request("Made-up one.")
            and items[1].request == Request("Made-up two.", {"a": 1}),
            "made-up requests default their item IDs and schema, and may name their own",
        )
        run(fake, directory, "jev", items, split="smoke", budget_bytes=5000, budgets=(None, None))
        run(fake, directory, "jev_bolstered", items, split="smoke", budget_bytes=5000, budgets=(None, None))
        run(fake, directory, "jev", ITEMS[:1])
        check(
            file_of(directory, "jev", "smoke").is_file() and file_of(directory, "jev_bolstered", "smoke").is_file(),
            "smoke records are stored in their own files, per router",
        )
        check(
            [r["item_id"] for r in stored(directory, "jev", "train")] == ["train-1"]
            and cj.read_records("jev", "dev", directory) == []
            and all(r["split"] == "smoke" for r in stored(directory, "jev", "smoke")),
            "the train and dev records hold no smoke record: a table builder reading them never sees one",
        )
        check(
            refused(cj.CollectError, lambda: run(fake, directory, "jev", items, split="smoke", pass_number=2)),
            "smoke has no noise pass",
        )
    print("\n-- the budget a rerun needs --")
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        directory = Path(folder)
        calls = len(fake.received)
        check(
            refused(cj.CollectError, lambda: run(fake, directory, "jev", ITEMS, budgets=(None, None)))
            and len(fake.received) == calls,
            "with the settings file's budget null and none given, the run is refused before any call",
        )
        summary = run(fake, directory, "jev", ITEMS, budgets=(None, None), budget_bytes=7)
        check(
            summary.new == 3 and summary.cut == 3,
            "an explicit budget stands in for the null one, and every prompt it cut is counted",
        )
        body = json.loads(fake.received[0].body)["state"]["request"]["prompt"]
        # By hand: half of 7 is 3, so the first 3 bytes ("Fir") and the last 3 ("st.") of the 22-byte prompt are kept.
        check(body == "First.", "a budget of 7 bytes keeps 3 bytes at the head and 3 at the tail: 'Fir' and 'st.'")


def write_bench(directory: Path) -> Path:
    """A synthetic benchmark file: two train rows (train-1 with a schema), a dev row and a test row, each carrying an
    expected answer that must never reach a body."""
    rows = [
        {"item_id": "train-1", "split": "train", "prompt": ITEMS[0].request.prompt, "output_schema": ITEMS[0].request.output_schema,
         "expected_answer": SECRET, "source": "t", "source_group": "g", "task_type": "t", "evaluation_method": "x", "meta": {}},
        {"item_id": "dev-1", "split": "dev", "prompt": "A dev prompt.", "expected_answer": SECRET, "source": "t",
         "source_group": "g2", "task_type": "t", "evaluation_method": "x", "meta": {}},
        {"item_id": "test-1", "split": "test", "prompt": "A test prompt.", "expected_answer": SECRET, "source": "t",
         "source_group": "g3", "task_type": "t", "evaluation_method": "x", "meta": {}},
        {"item_id": "train-3", "split": "train", "prompt": ITEMS[2].request.prompt, "expected_answer": SECRET, "source": "t",
         "source_group": "g4", "task_type": "t", "evaluation_method": "x", "meta": {}},
    ]  # fmt: skip
    bench = directory / "bench.jsonl"
    bench.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return bench


def section_sources() -> None:
    print("\n-- the sources --")
    with tempfile.TemporaryDirectory() as folder:
        sources(Path(folder))


def sources(directory: Path) -> None:
    bench = write_bench(directory)
    train = cj.benchmark_items("train", bench)
    check(
        [i.item_id for i in train] == ["train-1", "train-3"]
        and train[0].request == ITEMS[0].request
        and train[1].request == ITEMS[2].request,
        "train's requests come through the benchmark reader as requests: the prompt and schema, nothing else",
    )
    check([i.item_id for i in cj.benchmark_items("dev", bench)] == ["dev-1"], "dev's requests come the same way")
    check(
        any(json.loads(line)["split"] == "test" for line in bench.read_text().splitlines()),
        "the synthetic benchmark does hold a test row, so the refusals below could fail",
    )
    check(refused(SealedItemError, lambda: cj.benchmark_items("test", bench)), "the test split is refused")
    check(
        refused(SealedItemError, lambda: cj.requests_for("test", None, bench)),
        "the test split is refused through requests_for",
    )
    check(refused(BenchmarkError, lambda: cj.benchmark_items("smoke", bench)), "the reader gives no smoke split")
    check(
        refused(cj.CollectError, lambda: cj.requests_for("smoke", None, bench)),
        "smoke without made-up requests is refused",
    )
    made_up = directory / "made_up.json"
    made_up.write_text(json.dumps([{"prompt": "x"}]), encoding="utf-8")
    check(
        refused(cj.CollectError, lambda: cj.requests_for("train", made_up, bench)),
        "made-up requests under train are refused",
    )
    bad = {
        "not a list": {"prompt": "x"},
        "an empty list": [],
        "an unknown key": [{"prompt": "x", "expected_answer": "y"}],
        "an empty prompt": [{"prompt": ""}],
        "no prompt": [{"item_id": "a"}],
        "a prompt that isn't text": [{"prompt": 5}],
        "an empty item ID": [{"prompt": "x", "item_id": ""}],
        "a schema that isn't an object": [{"prompt": "x", "output_schema": [1]}],
        "an entry that isn't an object": ["x"],
    }
    for what, data in bad.items():
        made_up.write_text(json.dumps(data), encoding="utf-8")
        check(
            refused(cj.CollectError, lambda: cj.read_smoke_requests(made_up)),
            f"made-up requests with {what} are refused whole",
        )
    with FakeJev(ALL_ANSWERS) as fake:
        dup = [cj.Item("same", Request("a")), cj.Item("same", Request("b"))]
        check(
            refused(cj.CollectError, lambda: run(fake, directory, "jev", dup)) and fake.received == [],
            "a repeated item ID in the requests is refused before any call",
        )
        loaded = cj.load_settings(settings_file(directory, fake.url))
        noise = cj.noise_items(loaded, bench)
        check(
            [i.item_id for i in noise] == ["train-1", "train-3"],
            "the noise sample's items come through the reader, in the settings' order",
        )
        data = settings_data(fake.url)
        data["noise_sample"]["item_ids"] = ["train-1", "test-1"]
        path = directory / "noise_test.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        check(
            refused(SealedItemError, lambda: cj.noise_items(cj.load_settings(path), bench)),
            "a noise sample naming a test item is refused whole",
        )
        data["noise_sample"]["item_ids"] = ["train-1", "dev-1"]
        path.write_text(json.dumps(data), encoding="utf-8")
        check(
            refused(cj.CollectError, lambda: cj.noise_items(cj.load_settings(path), bench)),
            "a noise sample naming a dev item is refused",
        )
        data["noise_sample"] = {"seed": 2026}
        path.write_text(json.dumps(data), encoding="utf-8")
        check(
            refused(cj.CollectError, lambda: cj.noise_items(cj.load_settings(path), bench)),
            "a settings file with no noise sample is refused",
        )


def section_parse() -> None:
    print("\n-- stored records parse like live calls --")
    request = ITEMS[0].request
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        directory = Path(folder)
        loaded = cj.load_settings(settings_file(directory, fake.url))
        client, _ = client_on(loaded)
        weights = jev.parse_weights(synthetic_weights(), loaded.call)
        with client:
            g_live = jev.GRouter(loaded.call, client, budget_bytes=1000).score(request)
            plus_live = jev.GPlusRouter(loaded.call, weights, client, budget_bytes=2000).score(request)
        run(fake, directory, "jev", ITEMS[:1])
        run(fake, directory, "jev_bolstered", ITEMS[:1])
        g_record = stored(directory, "jev")[0]
        plus_record = stored(directory, "jev_bolstered")[0]
        g_scores = cj.record_scores(g_record, request, loaded)
        plus_scores = cj.record_scores(plus_record, request, loaded, weights)
        check(
            g_scores.scores == g_live == {"haiku": 0.91, "sonnet": 0.62, "opus_low": 0.33} and not g_scores.refused,
            "G: a stored record gives the router scores the live call gave, hand-worked 0.91, 0.62, 0.33",
        )
        want = {"haiku": 0.885197, "sonnet": 0.509474, "opus_low": 0.257565}
        check(
            plus_scores.scores == plus_live
            and all(abs(plus_scores.scores[c] - want[c]) < 1e-6 for c in want)
            and not plus_scores.refused,
            "G+: a stored record gives the router scores the live call gave, identical, and hand-worked",
        )
        check(
            refused(cj.CollectError, lambda: cj.record_scores(plus_record, request, loaded)),
            "G+'s router scores from a record need the weights: without them the call is refused",
        )
        check(
            refused(cj.CollectError, lambda: cj.record_scores(g_record | {"router": "other"}, request, loaded)),
            "a record of an unknown router is refused",
        )
    print("\n-- refusals counted from the stored records --")
    short_g = {name: ALL_ANSWERS[name] for name in ("success_haiku", "success_sonnet")}
    wrong = {n: a for n, a in ALL_ANSWERS.items() if n != "success_opus_low"} | {
        "success_opus_low": {"type": "noul", "noul": 1.5}
    }
    script = [
        Reply(200, response_body(ALL_ANSWERS)),
        Reply(200, response_body(short_g)),
        Reply(200, response_body(wrong)),
    ]
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS, script) as fake:
        directory = Path(folder)
        loaded = cj.load_settings(settings_file(directory, fake.url))
        summary = run(fake, directory, "jev", ITEMS)
        records = stored(directory, "jev")
        check(
            cj.refused_items(records, loaded) == ["train-2", "train-3"]
            and [r["refused"] for r in records] == [False, True, True]
            and summary.refused == 2,
            "refusals are counted from the stored responses: a missing answer and an out-of-range one, 2 of 3",
        )
        zero = cj.record_scores(records[1], ITEMS[1].request, loaded)
        check(
            zero.refused and zero.scores == {"haiku": 0.0, "sonnet": 0.0, "opus_low": 0.0},
            "a stored refusal gives 0 for every config",
        )
    zeros = {name: noul_answer(0.0) for name in G_ANSWERS}
    with tempfile.TemporaryDirectory() as folder, FakeJev(zeros) as fake:
        directory = Path(folder)
        loaded = cj.load_settings(settings_file(directory, fake.url))
        run(fake, directory, "jev", ITEMS[:1])
        check(
            cj.refused_items(stored(directory, "jev"), loaded) == [],
            "Jev answering 0.00 to everything is not a refusal",
        )


def section_key() -> None:
    print("\n-- the key --")
    dummy = "dummy-key-never-store-5d41"
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        directory = Path(folder)
        loaded = cj.load_settings(settings_file(directory, fake.url))
        client, _ = client_on(loaded, dummy)
        with client:
            cj.collect(ITEMS, router="jev", split="train", loaded=loaded, client=client, results_dir=directory)
            cj.collect(
                ITEMS, router="jev_bolstered", split="train", loaded=loaded, client=client, results_dir=directory
            )
        check(
            all(r.headers["authorization"] == f"Bearer {dummy}" for r in fake.received),
            "the dummy key reached the server on every call, so the search below can find it",
        )
        files = [p for p in directory.rglob("*") if p.is_file()]
        check(
            any(p.name.endswith(".jsonl") for p in files) and not any(dummy.encode() in p.read_bytes() for p in files),
            "the dummy key is in no stored byte: not in the records, the settings or anywhere under the results folder",
        )
        check(
            refused(jev.MissingKeyError, lambda: jev.JevClient(loaded.client, {})),
            "a run with no key in the environment stops",
        )


def section_usage() -> None:
    print("\n-- usage and the byte budget --")
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        directory = Path(folder)
        loaded = cj.load_settings(settings_file(directory, fake.url))
        run(fake, directory, "jev", ITEMS[:1], split="smoke")
        fake.input_tokens = 350
        run(fake, directory, "jev", ITEMS[1:2], split="smoke")
        run(fake, directory, "jev_bolstered", ITEMS[1:2], split="smoke")
        records = stored(directory, "jev", "smoke")
        check(
            [cj.input_tokens(r) for r in records] == [300, 350]
            and abs(cj.overhead_dollars(records[0], loaded) - 0.0000126) < 1e-12,
            "a record's input tokens are its stored usage, and 300 tokens at $0.042 per million is $0.0000126",
        )
        check(
            cj.measured_overhead_tokens(records) == 350, "the measured overhead is the largest stored input token count"
        )
        # The budget falls one for one with the overhead. check_jev_call pins the budget at 270 tokens, so the budget
        # at 350 is 80 below the same function's at 270.
        schema = {"type": "object"}
        check(
            cj.measured_budget_bytes("jev", records, loaded, schema)
            == jev_call.g_budget_bytes(loaded.call, overhead_tokens=270, schema=schema) - 80,
            "G's budget from stored usage is the settings' fixed parts less the measured overhead",
        )
        plus = stored(directory, "jev_bolstered", "smoke")
        check(
            cj.measured_budget_bytes("jev_bolstered", plus, loaded, None)
            == jev_call.g_plus_budget_bytes(loaded.call, overhead_tokens=270, schema=None) - 80,
            "G+'s budget from stored usage likewise, the past requests and questions as its fixed parts",
        )
        check(refused(cj.CollectError, lambda: cj.measured_overhead_tokens([])), "no records, no measurement")
        check(
            refused(cj.CollectError, lambda: cj.input_tokens(records[0] | {"usage": None})),
            "a record without usage is refused",
        )
        check(
            refused(cj.CollectError, lambda: cj.input_tokens(records[0] | {"usage": {"input_tokens": True}})),
            "a bool is not a token count",
        )


def command(args: list[str], environ: dict[str, str] | None) -> tuple[int, str, str]:
    """Run the command line, with the exit code and what it printed."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cj.main(args, environ)
        except SystemExit as exit_:
            code = int(str(exit_.code))
    return code, out.getvalue(), err.getvalue()


def section_command() -> None:
    print("\n-- the command line --")
    env = {jev.KEY_VARIABLE: KEY}
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS) as fake:
        directory = Path(folder)
        results = directory / "results"
        settings = settings_file(directory, fake.url, (None, None))
        made_up = directory / "made_up.json"
        made_up.write_text(json.dumps([{"prompt": "Made-up one."}, {"prompt": "Made-up two."}]), encoding="utf-8")
        base = ["--settings", str(settings), "--results-dir", str(results)]
        smoke = ["collect", "--router", "jev", "--split", "smoke", "--requests", str(made_up), *base]
        code, _, err = command(smoke, env)
        check(
            code == 2 and "budget" in err and fake.received == [],
            "null budgets and none given: refused, exit 2, no call",
        )
        code, out, _ = command([*smoke, "--budget-bytes", "5000"], env)
        check(
            code == 0
            and "jev smoke pass 1: 2 new, 0 already stored" in out
            and len(stored(results, "jev", "smoke")) == 2,
            "--budget-bytes stands in for the null budget, and made-up requests are stored under smoke",
        )
        code, out, _ = command([*smoke, "--budget-bytes", "5000"], env)
        check(
            code == 0 and "0 new, 2 already stored" in out and len(fake.received) == 2,
            "the same command again sends nothing",
        )
        code, _, err = command(
            [
                "collect",
                "--router",
                "both",
                "--split",
                "smoke",
                "--requests",
                str(made_up),
                "--budget-bytes",
                "5",
                *base,
            ],
            env,
        )
        check(code == 2 and "one router" in err, "--budget-bytes with both routers is refused")
        code, _, err = command(["collect", "--router", "jev", "--split", "smoke", *base, "--budget-bytes", "5000"], env)
        check(code == 2 and "made-up requests" in err, "smoke without --requests is refused")
        code, _, _ = command(["collect", "--router", "jev", "--split", "test", *base, "--budget-bytes", "5000"], env)
        check(code == 2, "--split test is not a choice")
        code, _, err = command([*smoke, "--budget-bytes", "5000"], {})
        check(
            code == 1 and "stopped" in err and jev.KEY_VARIABLE in err,
            "no key in the environment: the run stops, exit 1",
        )
        code, out, _ = command(["report", "--router", "jev", "--split", "smoke", *base], env)
        check(
            code == 0 and "2 records" in out and "refused: 0" in out and "input tokens: 600" in out,
            "report counts what is stored",
        )
        code, out, _ = command(["budget", "--router", "jev", "--split", "smoke", *base], env)
        check(code == 0 and "overhead 300 tokens" in out, "budget measures the overhead from the stored smoke usage")
        # Train, dev and the noise pass, on a synthetic benchmark whose train items are train-1 and train-3, the
        # settings' noise sample, and with budgets filled in (the smoke part above ran with them null).
        (directory / "filled").mkdir()
        filled = settings_file(directory / "filled", fake.url)
        bench = write_bench(directory)
        full = ["--settings", str(filled), "--results-dir", str(results), "--benchmark-file", str(bench)]
        code, out, err = command(["noise", "--router", "both", *full], env)
        check(code == 2 and "no pass 1" in err, "the noise pass before the train collection is refused, exit 2")
        code, out, _ = command(["collect", "--router", "both", "--split", "train", *full], env)
        check(
            code == 0 and "jev train pass 1: 2 new" in out and "jev_bolstered train pass 1: 2 new" in out,
            "collect --router both sends train through G, then G+",
        )
        code, out, _ = command(["noise", "--router", "both", *full], env)
        check(
            code == 0 and "jev train pass 2: 2 new" in out and "jev_bolstered train pass 2: 2 new" in out
            and [r["pass"] for r in stored(results, "jev")] == [1, 1, 2, 2],
            "the noise pass sends the sample as pass 2, for both routers",
        )  # fmt: skip
        code, out, _ = command(["noise", "--router", "both", *full], env)
        check(code == 0 and "0 new, 2 already stored" in out, "the noise pass again sends nothing")
        code, out, _ = command(["collect", "--router", "jev", "--split", "dev", *full], env)
        check(
            code == 0 and [r["item_id"] for r in stored(results, "jev", "dev")] == ["dev-1"],
            "dev is collected through the reader",
        )
        sent = " ".join(r.body.decode() for r in fake.received)
        check(
            SECRET not in sent and "A test prompt." not in sent,
            "no expected answer and no test request ever reached the fake server",
        )
        check(
            len(stored(results, "jev", "train")) == 4 and len(stored(results, "jev", "smoke")) == 2,
            "each split stays in its own file",
        )
    with tempfile.TemporaryDirectory() as folder, FakeJev(ALL_ANSWERS, [Reply(400, {"detail": "bad"})]) as fake:
        directory = Path(folder)
        settings = settings_file(directory, fake.url)
        made_up = directory / "made_up.json"
        made_up.write_text(json.dumps([{"prompt": "Made-up one."}]), encoding="utf-8")
        code, out, err = command(
            ["collect", "--router", "jev", "--split", "smoke", "--requests", str(made_up), "--settings", str(settings),
             "--results-dir", str(directory / "r")], env,
        )  # fmt: skip
        check(
            code == 1 and "jev smoke smoke-001" in err and "valid" in err,
            "a 400 stops the command with exit 1 and names the request",
        )


def main() -> int:
    section_plus()
    section_storage()
    section_resume()
    section_stop()
    section_noise()
    section_smoke()
    section_sources()
    section_parse()
    section_key()
    section_usage()
    section_command()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
