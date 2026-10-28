"""Check the Jev router's client and G's scoring (PLAN_v4.md §8 3c; Amendment A15, items 3, 4, 9 and 12).

Run from the project root:  uv run python -m analysis.check_jev_client

Never a test item and never a Jev call: every request is made-up text, and every call goes to the fake Jev server
(analysis/fake_jev.py) on a loopback port, with retry waits injected so no check sleeps.

- the parser on hand-made responses: raw answers become G's router scores, 0 and 1 included, and each way a 200 can
  fail to answer what was asked is a refusal, found from the response and never from a score;
- `score(request)` for G through the fake server: hand-worked router scores, the body that arrived, the key sent;
- retries: 429 then 200 with `Retry-After` honoured, repeated 500s stopping at the cap and naming the request,
  backoff doubling to its cap, a timeout retried, an unreadable 200 retried, a wait over the cap stopping the run,
  and 400, 401, 403 and 422 stopping at once;
- a connection the server closed, reopened and the call resent without using an attempt, with the contrast that a
  failure on a fresh connection does use one;
- what a call returns (the record's fields) and that the key appears in none of it;
- the client settings: each malformed part refused, with its own check;
- the imports: the router's closure uses the standard library and numpy only, and the fake server stays outside it.
"""

import ast
import itertools
import json
import subprocess
import sys
from collections.abc import Callable

from analysis.fake_jev import FakeJev, Reply, choice_answer, noul_answer, response_body
from analysis.freeze import closure
from harness.client import PROJECT_ROOT
from routers import jev, jev_call
from routers.request import Request

problems: list[str] = []

KEY = "test-key-7f3a9c"  # made up; the real key is never read by a check


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


def imported_packages(tree: ast.AST) -> set[str]:
    """The top-level packages a module imports (a relative import stays inside its package)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


RETRY = {"attempts": 8, "backoff_start_s": 1, "backoff_cap_s": 60, "retry_after_cap_s": 600, "timeout_s": 30}


def synthetic_data(url: str = "http://127.0.0.1:1/v1/systemone", retry: dict | None = None) -> dict:
    """A settings file in #30's format, as far as the client and G read it, with made-up statements."""
    return {
        "jev": {
            "endpoint": url,
            "model": "jev-test-0",
            "price_per_million_input_tokens": 0.042,
            "price_checked": "2026-10-25",
        },
        "display_names": {"haiku": "Small", "sonnet": "Middle", "opus_low": "Large"},
        "g": {
            "success_statements": [
                {"id": "success_haiku", "config": "haiku", "text": "Small text."},
                {"id": "success_sonnet", "config": "sonnet", "text": "Middle text."},
                {"id": "success_opus_low", "config": "opus_low", "text": "Large text."},
            ]
        },
        "budget_bytes": {"jev": 1000, "jev_bolstered": None},
        "retry": retry or RETRY,
    }


G_IDS = ("success_haiku", "success_sonnet", "success_opus_low")


def g_response(haiku: float, sonnet: float, opus_low: float) -> dict:
    values = dict(zip(G_IDS, (haiku, sonnet, opus_low), strict=True))
    return response_body({name: noul_answer(value) for name, value in values.items()})


def g_questions() -> dict[str, dict]:
    settings = jev_call.parse_settings(synthetic_data())
    return jev_call.g_questions(settings)


def client_for(
    fake: FakeJev, retry: dict | None = None, jitter: float = 0.0
) -> tuple[jev.JevClient, jev_call.Settings, list[float]]:
    """A client on the fake server with its waits recorded, not slept, and a clock that ticks 0.25 per reading, so
    an attempt's time is 0.25 whatever the machine does. Jitter is fixed (0 keeps each wait whole)."""
    waits: list[float] = []
    ticks = itertools.count()
    data = synthetic_data(fake.url, retry)
    client = jev.JevClient(
        jev.parse_client_settings(data),
        {jev.KEY_VARIABLE: KEY},
        sleep=waits.append,
        clock=lambda: next(ticks) * 0.25,
        jitter=lambda: jitter,
    )
    return client, jev_call.parse_settings(data), waits


def good_reply() -> Reply:
    return Reply(200, g_response(0.9, 0.8, 0.7))


def send(client: jev.JevClient, settings: jev_call.Settings, label: str = "item-7") -> jev.CallResult:
    call = jev_call.g_call(Request("A made-up request."), settings)
    return client.call(call.body, call.sha256, label)


def stops(error: type[jev.RunStop], client: jev.JevClient, settings: jev_call.Settings, label: str) -> str | None:
    """The message of the stop a call ends in, or None if it didn't stop with `error`."""
    try:
        send(client, settings, label)
    except error as stop:
        return str(stop)
    return None


G_ANSWERS = {
    "success_haiku": noul_answer(0.91),
    "success_sonnet": noul_answer(0.62),
    "success_opus_low": noul_answer(0.33),
}


def section_score() -> None:
    print("\n-- score(request) for G, against the fake server --")
    with FakeJev(G_ANSWERS) as fake:
        client, settings, waits = client_for(fake)
        router = jev.GRouter(settings, client)
        request = Request("A made-up request.", {"type": "object"})
        asked = router.ask(request, "item-1")
        check(
            asked.scores == {"haiku": 0.91, "sonnet": 0.62, "opus_low": 0.33},
            "G's router scores are the three answers, hand-worked: 0.91, 0.62, 0.33",
        )
        check(router.score(request) == asked.scores, "score(request) gives the same router scores")
        received = fake.received[0]
        check(received.body == jev_call.g_call(request, settings).body, "the body that reached Jev is the built body")
        check(
            received.headers["authorization"] == f"Bearer {KEY}"
            and received.headers["content-type"] == "application/json",
            "the call carries the key as a bearer token, so the key assertions below can fail",
        )
        record = asked.result.to_record()
        check(
            record["model"] == "jev-test-0"
            and record["usage"] == {"input_tokens": 300, "output_tokens": 38}
            and record["request_id"] == "req_" + "0" * 31 + "1"
            and record["server_date"] == "Wed, 28 Oct 2026 00:01:57 GMT"
            and record["response"] == response_body(G_ANSWERS),
            "a call returns the whole response, model ID, usage, request ID and the server's Date",
        )
        check(
            record["seconds"] == 0.25
            and record["attempts"] == 1
            and record["failed_attempts"] == []
            and record["new_connection"] is True
            and record["refused"] is False
            and record["body_sha256"] == asked.call.sha256,
            "a call returns the attempt's seconds, one attempt, no failures, a new connection (its first), not refused",
        )
        check(
            json.loads(json.dumps(record)) == record
            and KEY not in json.dumps(record)
            and KEY not in repr(asked.result),
            "the record is plain JSON, and the key appears nowhere in a result",
        )
        second = router.ask(request)
        check(
            second.result.new_connection is False and fake.connections == 1 and not waits,
            "the connection is kept open: the second call reuses it, and nothing waited",
        )
        check(
            abs(jev.overhead_dollars(300, client.settings) - 0.0000126) < 1e-12,
            "overhead is input tokens at the settings price: 300 tokens at $0.042 per million is $0.0000126",
        )
        check(fake.unexpected == [] and router.counts == jev.Counts(calls=3, refused=0, cut=0), "calls were counted")

    print("\n-- a refusal --")
    zeros = {name: noul_answer(0.0) for name in G_ANSWERS}
    with FakeJev(zeros) as fake:
        client, settings, _ = client_for(fake)
        router = jev.GRouter(settings, client)
        asked = router.ask(Request("Zero."))
        check(
            asked.scores == {"haiku": 0.0, "sonnet": 0.0, "opus_low": 0.0} and not asked.result.refused,
            "Jev answering 0.00 to everything gives zeros and no refusal flag",
        )
    short = {name: G_ANSWERS[name] for name in ("success_haiku", "success_sonnet")}
    with FakeJev(short, [Reply(200, response_body(short))]) as fake:
        client, settings, _ = client_for(fake)
        router = jev.GRouter(settings, client)
        asked = router.ask(Request("Refused."))
        check(
            asked.scores == {"haiku": 0.0, "sonnet": 0.0, "opus_low": 0.0}
            and asked.result.refused
            and asked.result.to_record()["refused"] is True
            and router.counts.refused == 1,
            "a 200 that leaves a question unanswered is a refusal: 0 for every config, flagged and counted",
        )
        check(asked.result.attempts == 1, "a refusal is an answer, not retried")


def section_retries() -> None:
    print("\n-- retries --")
    ok = good_reply
    with FakeJev(G_ANSWERS, [Reply(429, {}, {"Retry-After": "3"})]) as fake:
        client, settings, waits = client_for(fake)
        result = send(client, settings)
        check(
            waits == [3.0] and result.attempts == 2 and result.failed_attempts == (jev.FailedAttempt(429, None),),
            "429 then 200 is retried, waiting the 3 s Retry-After asks (over the 1 s backoff), and the failure is kept",
        )
        check(
            result.new_connection is False and result.seconds == 0.25 and len(fake.received) == 2,
            "the retry reuses the connection, and the time is the successful attempt's alone",
        )
    with FakeJev(
        G_ANSWERS, [Reply(429, {}, {"retry-after-ms": "2500"}), Reply(503, {}, {"Retry-After": "0.2"})]
    ) as fake:
        client, settings, waits = client_for(fake)
        send(client, settings)
        check(waits == [2.5, 2.0], "retry-after-ms is read, and a Retry-After under the backoff never shortens it")
    with FakeJev(G_ANSWERS, [Reply(429, {}, {"Retry-After": "soon"}), Reply(429)]) as fake:
        client, settings, waits = client_for(fake)
        send(client, settings)
        check(waits == [1.0, 2.0], "a Retry-After that isn't a number is ignored: the backoff doubles")
    with FakeJev(G_ANSWERS, [Reply(408), Reply(502), Reply(529), ok()]) as fake:
        client, settings, waits = client_for(fake)
        result = send(client, settings)
        check(
            [a.status for a in result.failed_attempts] == [408, 502, 529] and result.attempts == 4,
            "408, 502 and 529 are retried",
        )
    with FakeJev(G_ANSWERS, [Reply(500)] * 8) as fake:
        client, settings, waits = client_for(fake)
        message = stops(jev.RetryCapReached, client, settings, "item-9")
        check(
            message is not None and "item-9" in message and len(fake.received) == 8,
            "repeated 500s stop at the 8th attempt, with an error that names the request",
        )
        check(
            waits == [1, 2, 4, 8, 16, 32, 60],
            "the backoff doubles from 1 s to its 60 s cap, with no wait after the last",
        )
    with FakeJev(G_ANSWERS, [Reply(500)] * 8) as fake:
        client, settings, waits = client_for(fake, jitter=1.0)
        stops(jev.RetryCapReached, client, settings, "item-9")
        check(waits == [0.75, 1.5, 3, 6, 12, 24, 45], "jitter shortens each wait by up to a quarter (here the most)")
    with FakeJev(G_ANSWERS, [Reply(500)] * 3) as fake:
        client, settings, waits = client_for(fake, {**RETRY, "attempts": 3})
        check(
            stops(jev.RetryCapReached, client, settings, "x") is not None and len(fake.received) == 3,
            "the cap is the settings'",
        )
    with FakeJev(G_ANSWERS, [Reply(429, {}, {"Retry-After": "600"}), ok()]) as fake:
        client, settings, waits = client_for(fake)
        send(client, settings)
        check(waits == [600.0], "a Retry-After of exactly the 600 s cap is honoured")
    with FakeJev(G_ANSWERS, [Reply(429, {}, {"Retry-After": "601"}), ok()]) as fake:
        client, settings, waits = client_for(fake)
        message = stops(jev.RetryAfterTooLong, client, settings, "item-3")
        check(
            message is not None and "item-3" in message and not waits and len(fake.received) == 1,
            "a Retry-After of 601 s stops the run, naming the request, without waiting",
        )
    for status in (400, 401, 403, 422, 404):
        with FakeJev(G_ANSWERS, [Reply(status, {"detail": "no"}), ok()]) as fake:
            client, settings, waits = client_for(fake)
            message = stops(jev.Rejected, client, settings, "item-5")
            check(
                message is not None
                and str(status) in message
                and "item-5" in message
                and not waits
                and len(fake.received) == 1,
                f"{status} stops at once, naming the request, with no retry and no wait",
            )
    with FakeJev(G_ANSWERS, [Reply(401, f"bad key: Bearer {KEY}")]) as fake:
        client, settings, _ = client_for(fake)
        message = stops(jev.Rejected, client, settings, "item-5")
        check(
            message is not None
            and KEY not in message
            and "<key>" in message
            and KEY in fake.received[0].headers["authorization"],
            "a server echoing the key back doesn't put it in the stop's message",
        )
    with FakeJev(G_ANSWERS, [Reply(200, delay_s=0.6), ok()]) as fake:
        client, settings, waits = client_for(fake, {**RETRY, "timeout_s": 0.1})
        result = send(client, settings)
        check(
            result.failed_attempts == (jev.FailedAttempt(None, "timeout"),) and result.attempts == 2 and waits == [1.0],
            "a timeout is retried after the backoff",
        )
        check(result.new_connection is True, "the retry after a timeout runs on a new connection, so it is flagged")
    with FakeJev(G_ANSWERS, [Reply(200, "<html>bad gateway</html>"), Reply(200, [1, 2]), ok()]) as fake:
        client, settings, waits = client_for(fake)
        result = send(client, settings)
        check(
            result.attempts == 3 and [a.error for a in result.failed_attempts] == ["unreadable response"] * 2,
            "a 200 that isn't a JSON object is a failed attempt, retried",
        )
    check(
        refused(jev.MissingKeyError, lambda: jev.JevClient(jev.parse_client_settings(synthetic_data()), {})),
        "a client without the key in the environment is refused",
    )


def section_reconnect() -> None:
    print("\n-- a closed connection --")
    hang_up = Reply(200, g_response(0.9, 0.8, 0.7), hang_up_after=True)
    with FakeJev(G_ANSWERS, [hang_up]) as fake:
        client, settings, waits = client_for(fake)
        first = send(client, settings)
        second = send(client, settings)
        check(
            second.attempts == 1 and second.failed_attempts == () and not waits and fake.connections == 2,
            "a connection the server closed is reopened and resent at once: no attempt used, no wait",
        )
        check(
            first.new_connection and second.new_connection and [r.connection for r in fake.received] == [1, 2],
            "the call that reopened is flagged as a new connection, and reached the server on the second one",
        )
        third = send(client, settings)
        check(third.new_connection is False and fake.connections == 2, "the call after it reuses the new connection")
    with FakeJev(G_ANSWERS, [good_reply(), Reply(drop=True)]) as fake:
        client, settings, waits = client_for(fake)
        send(client, settings)
        second = send(client, settings)
        check(
            second.attempts == 1 and not waits and second.new_connection and fake.connections == 2,
            "a reused connection closed on receiving the call is reopened and resent without using an attempt",
        )
    with FakeJev(G_ANSWERS, [Reply(drop=True)]) as fake:
        client, settings, waits = client_for(fake)
        result = send(client, settings)
        check(
            result.attempts == 2
            and waits == [1.0]
            and result.failed_attempts == (jev.FailedAttempt(None, "RemoteDisconnected"),),
            "the contrast: a failure on a connection just opened is an ordinary failed attempt, with a wait",
        )
    with FakeJev(G_ANSWERS, [Reply(200, g_response(0.9, 0.8, 0.7), {"Connection": "close"})]) as fake:
        client, settings, waits = client_for(fake)
        send(client, settings)
        second = send(client, settings)
        check(
            second.new_connection and second.attempts == 1 and fake.connections == 2 and not waits,
            "after the server says `Connection: close` the next call opens a connection and is flagged, with no failure",
        )


def section_settings() -> None:
    print("\n-- the client settings --")
    parsed = jev.parse_client_settings(synthetic_data("https://api.typesafe.ai/v1/systemone"))
    check(
        parsed.endpoint == "https://api.typesafe.ai/v1/systemone"
        and parsed.price_per_million_input_tokens == 0.042
        and parsed.retry == jev.Retry(8, 1, 60, 600, 30),
        "the settings hold the endpoint, the price and A15 item 9's retry policy: 8, 1, 60, 600 and 30",
    )

    def mutated(change: Callable[[dict], object]) -> Callable[[], object]:
        data = synthetic_data()
        change(data)
        return lambda: jev.parse_client_settings(data)

    refusals: dict[str, Callable[[dict], object]] = {
        "no jev part": lambda d: d.pop("jev"),
        "no endpoint": lambda d: d["jev"].pop("endpoint"),
        "an endpoint with no scheme": lambda d: d["jev"].update(endpoint="api.typesafe.ai/v1/systemone"),
        "an ftp endpoint": lambda d: d["jev"].update(endpoint="ftp://api.typesafe.ai/x"),
        "plain http to a remote host (the key would travel in the clear)": lambda d: d["jev"].update(
            endpoint="http://api.typesafe.ai/v1/systemone"
        ),
        "a negative price": lambda d: d["jev"].update(price_per_million_input_tokens=-1),
        "a price that is a string": lambda d: d["jev"].update(price_per_million_input_tokens="0.042"),
        "no retry part": lambda d: d.pop("retry"),
        "a retry setting missing": lambda d: d["retry"].pop("timeout_s"),
        "an extra retry setting": lambda d: d["retry"].update(jitter=1),
        "zero attempts": lambda d: d["retry"].update(attempts=0),
        "attempts that aren't whole": lambda d: d["retry"].update(attempts=2.5),
        "attempts that are a bool": lambda d: d["retry"].update(attempts=True),
        "a negative timeout": lambda d: d["retry"].update(timeout_s=-1),
        "a timeout that is NaN": lambda d: d["retry"].update(timeout_s=float("nan")),
        "a backoff cap below its start": lambda d: d["retry"].update(backoff_cap_s=0.5),
    }
    for what, change in refusals.items():
        check(refused(jev.ClientSettingsError, mutated(change)), f"settings with {what} are refused")


def section_imports() -> None:
    print("\n-- imports --")
    files = closure("routers/jev.py")
    imported = set().union(*(imported_packages(ast.parse((PROJECT_ROOT / name).read_text())) for name in files))
    check(
        {"http", "json", "routers"} <= imported and "routers/jev_call.py" in files,
        "the guard finds the imports it looks for: http, json and routers, and the closure holds jev_call",
    )
    check(
        imported_packages(ast.parse("import requests\nimport sklearn.linear_model\nfrom analysis import fake_jev"))
        == {"requests", "sklearn", "analysis"},
        "the guard finds an import of requests, scikit-learn and analysis in a module that has them",
    )
    allowed = set(sys.stdlib_module_names) | {"numpy", "routers"}
    check(
        not imported - allowed,
        f"the closure imports the standard library and numpy only ({sorted(imported - allowed)})",
    )
    check(
        not any("fake_jev" in name for name in files) and "analysis" not in imported,
        "the fake server is outside the router's import closure",
    )
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, routers.jev; print(sorted(m for m in sys.modules if m.split('.')[0] in "
         "{'analysis', 'harness', 'sklearn', 'torch', 'transformers', 'requests'}))"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    check(probe.returncode == 0 and probe.stdout.strip() == "[]",
          f"importing the router loads no analysis, harness, scikit-learn, torch or requests module ({probe.stdout.strip()})")  # fmt: skip


def main() -> int:
    questions = g_questions()

    print("-- the parser --")
    parsed = jev.g_scores(g_response(0.91, 0.5, 0.0), questions)
    check(
        parsed.scores == {"haiku": 0.91, "sonnet": 0.5, "opus_low": 0.0} and not parsed.refused,
        "answers by question ID become G's router scores, unchanged, in the config's order",
    )
    check(
        jev.g_scores(g_response(1.0, 0.0, 1.0), questions).scores == {"haiku": 1.0, "sonnet": 0.0, "opus_low": 1.0},
        "0 and 1 are valid router scores, and an answer of 0.00 is not a refusal",
    )
    all_zero = jev.g_scores(g_response(0.0, 0.0, 0.0), questions)
    check(not all_zero.refused, "Jev answering 0.00 to every statement is no refusal: it is read from the response")

    # Each way a 200 can fail to answer what was asked. The known positive is the well-formed response above.
    good = g_response(0.4, 0.5, 0.6)

    def without(name: str) -> dict:
        answers = dict(good["answers"])
        del answers[name]
        return {**good, "answers": answers}

    def with_answer(name: str, answer: object) -> dict:
        return {**good, "answers": {**good["answers"], name: answer}}

    cases: dict[str, object] = {
        "a missing question ID": without("success_sonnet"),
        "an empty answers object": {**good, "answers": {}},
        "no answers key": {"model": "jev-test-0", "usage": {}},
        "answers that are a list": {**good, "answers": [1, 2, 3]},
        "a body that is not an object": [1, 2],
        "an answer of the wrong type": with_answer("success_haiku", {"type": "choice", "probabilities": {"a": 1.0}}),
        "an answer with no value": with_answer("success_haiku", {"type": "noul"}),
        "an answer that is not an object": with_answer("success_haiku", 0.5),
        "a value that is a string": with_answer("success_haiku", noul_answer("0.5")),  # type: ignore[arg-type]
        "a value that is a bool": with_answer("success_haiku", noul_answer(True)),
        "a value above 1": with_answer("success_haiku", noul_answer(1.01)),
        "a value below 0": with_answer("success_haiku", noul_answer(-0.01)),
        "a value that is NaN": with_answer("success_haiku", noul_answer(float("nan"))),
    }
    for what, response in cases.items():
        result = jev.g_scores(response, questions)
        check(
            result.refused and result.scores == {"haiku": 0.0, "sonnet": 0.0, "opus_low": 0.0},
            f"{what} is a refusal: 0 for every config, flagged",
        )
    extra = with_answer("unasked", noul_answer(0.3))
    check(jev.g_scores(extra, questions).scores["haiku"] == 0.4, "an answer to a question not asked is ignored")

    # A Choice answer is read by option, from `probabilities`, not from its winning `choice` (live call 2).
    kind = {"type": "choice", "instructions": "k", "criteria": {"a": None, "b": None}}
    both: dict[str, dict] = {"k": kind, "n": {"type": "noul", "instructions": "n"}}
    live = response_body({"k": choice_answer({"a": 0.99, "b": 0.01}), "n": noul_answer(0.87)})
    check(
        jev.parse_answers(live, both) == jev.Parsed({"k": {"a": 0.99, "b": 0.01}, "n": 0.87}, False),
        "a Noul answer is a float and a Choice answer a dict of its options' probabilities",
    )
    wrong_options = response_body({"k": choice_answer({"a": 0.5, "z": 0.5}), "n": noul_answer(0.1)})
    check(jev.parse_answers(wrong_options, both).refused, "a Choice answer with other options than asked is a refusal")
    check(
        jev.parse_answers(response_body({"k": {"type": "choice", "choice": "a"}, "n": noul_answer(0.1)}), both).refused,
        "a Choice answer without its probabilities is a refusal",
    )

    section_score()
    section_retries()
    section_reconnect()
    section_settings()
    section_imports()

    print(f"\n{len(problems)} checks failed" if problems else "\nall checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
