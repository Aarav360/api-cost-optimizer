"""The Jev router's client and scoring (PLAN_v4.md §8 3c; Amendment A15, items 4, 9 and 12).

`routers/jev_call.py` builds a call's body; this module sends it, reads the answer and turns it into router scores.
It is frozen at the freeze commit (A15 item 13). Standard library and numpy only (A15 item 12).

The refusal rule (A15 item 9). Jev's API documents no refusal signal, and item 9 sends any 4xx that isn't retried to
a run stop, so a refusal Jev can give for good is a 200 that doesn't answer what was asked. A response is a refusal
when `answers` is not an object, or lacks any asked question ID, or gives an answer of the wrong form for its
question: a Noul question needs `{"type": "noul", "noul": x}` and a Choice question needs
`{"type": "choice", "probabilities": {...}}` with exactly its options, every number finite and in [0, 1] (a bool is
not a number). It is found from the response alone, never from a router score, since Jev can itself answer 0.00.
A refused request gets router score 0 for every config and the call's `refused` flag is set. A 200 that isn't JSON
at all is not a refusal but a failed attempt, retried like a 5xx (a gateway fault, not an answer).

The client. One kept-open `http.client` connection, one call at a time, with the settings file's timeout.
- Attempts (the settings file's `retry`): 408, 429, 5xx, timeouts, dropped connections that were fresh and unreadable
  200s are retried, up to `attempts` in all, waiting `backoff_start_s` doubling to `backoff_cap_s`, less up to 25%
  jitter. A `retry-after-ms` or `Retry-After` header (seconds) is honoured, never waiting less than it says; one over
  `retry_after_cap_s` stops the run, as does the cap. Any other status stops the run at once.
- A kept-open connection the server closed (it fails on first use) is reopened and the call resent at once, with no
  wait and no attempt used. That call is flagged `new_connection`, as is a call's first, since both include a
  connect in the time. A failure on a connection opened for that very send is an ordinary failed attempt.
- A stop is a `RunStop` that names the request by the `label` the caller gave. The key is read from the environment
  (`TYPESAFE_API_KEY`) and never appears in a result, a record or a message.
- Latency is the successful attempt's time alone, on a monotonic clock (A15 item 10): retries and waits are counts.

Network, waiting and chance are injected (`sleep`, `clock`, `jitter`), so a check runs with none of them.
"""

import http.client
import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit

from routers.jev_call import Call, Settings, g_call, g_questions, read_settings
from routers.ladder import SCORED
from routers.request import Request

SETTINGS_FILE = Path(__file__).with_name("jev_settings.json")
KEY_VARIABLE = "TYPESAFE_API_KEY"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}  # the only hosts the key may travel to over plain http (the fake server)
JITTER = 0.25  # a backoff wait is shortened by up to this share, the SDK's own policy
RETRIED = {408, 429}  # and every 5xx


class ClientSettingsError(ValueError):
    """A settings file's `jev` or `retry` part this module refuses whole."""


class RunStop(RuntimeError):
    """A call that stops the run (A15 item 9). The message names the request."""


class Rejected(RunStop):
    """A status that isn't retried: 400, 401, 403, 422 and every other status but 200."""


class RetryCapReached(RunStop):
    """The attempts ran out."""


class RetryAfterTooLong(RunStop):
    """The server asked for a wait longer than the settings file allows."""


class MissingKeyError(RunStop):
    """The key isn't in the environment."""


@dataclass(frozen=True)
class Retry:
    attempts: int
    backoff_start_s: float
    backoff_cap_s: float
    retry_after_cap_s: float
    timeout_s: float


@dataclass(frozen=True)
class ClientSettings:
    endpoint: str
    price_per_million_input_tokens: float
    retry: Retry


def is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def parse_client_settings(data: dict) -> ClientSettings:
    """The client's part of a settings file (`jev` and `retry`), refusing it whole if any part is malformed."""
    try:
        endpoint = data["jev"]["endpoint"]
        price = data["jev"]["price_per_million_input_tokens"]
        raw = data["retry"]
    except (KeyError, TypeError) as error:
        raise ClientSettingsError(f"settings lack or misshape {error!r}") from error
    parts = urlsplit(endpoint) if isinstance(endpoint, str) else None
    if parts is None or parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ClientSettingsError(f"the endpoint is not an http(s) URL with a host: {endpoint!r}")
    if parts.scheme == "http" and parts.hostname not in LOOPBACK:
        raise ClientSettingsError("plain http is for loopback only: the key must not travel in the clear")
    if not is_number(price) or price < 0:
        raise ClientSettingsError("the price per million input tokens is a finite number of at least 0")
    names = ("attempts", "backoff_start_s", "backoff_cap_s", "retry_after_cap_s", "timeout_s")
    if not isinstance(raw, dict) or set(raw) != set(names):
        raise ClientSettingsError(f"retry must hold exactly {names}")
    if not all(is_number(raw[name]) and raw[name] > 0 for name in names):
        raise ClientSettingsError("every retry setting is a finite number above 0")
    if type(raw["attempts"]) is not int:
        raise ClientSettingsError("attempts is a whole number")
    if raw["backoff_cap_s"] < raw["backoff_start_s"]:
        raise ClientSettingsError("the backoff cap is below its start")
    return ClientSettings(endpoint, float(price), Retry(**raw))


def read_client_settings(path: Path = SETTINGS_FILE) -> ClientSettings:
    return parse_client_settings(json.loads(path.read_text(encoding="utf-8")))


def overhead_dollars(input_tokens: int, settings: ClientSettings) -> float:
    """A call's overhead in dollars: its input tokens at Jev's price; output tokens are free (A15 item 10)."""
    return input_tokens * settings.price_per_million_input_tokens / 1_000_000


# ---- the parser -----------------------------------------------------------------------------------------------


Answer = float | dict[str, float]  # a Noul answer, or a Choice's probability for each option


@dataclass(frozen=True)
class Parsed:
    """A response read against the questions asked: answers by question ID, or none and `refused`."""

    answers: dict[str, Answer]
    refused: bool


@dataclass(frozen=True)
class GScores:
    scores: dict[str, float]  # config -> router score, in the ladder's order; 0 for every config if refused
    refused: bool


def is_probability(value: object) -> bool:
    return is_number(value) and 0 <= value <= 1  # type: ignore[operator]


def read_answer(answer: object, question: Mapping[str, object]) -> Answer | None:
    """One answer, or None if it isn't the form its question needs (the refusal rule above)."""
    if not isinstance(answer, dict) or answer.get("type") != question.get("type"):
        return None
    if question.get("type") == "noul":
        value = answer.get("noul")
        return float(value) if is_probability(value) else None  # type: ignore[arg-type]
    if question.get("type") == "choice":
        options = question.get("criteria")
        probabilities = answer.get("probabilities")
        if not isinstance(options, dict) or not isinstance(probabilities, dict) or set(probabilities) != set(options):
            return None
        if not all(is_probability(p) for p in probabilities.values()):
            return None
        return {option: float(p) for option, p in probabilities.items()}
    return None


def parse_answers(response: object, questions: Mapping[str, Mapping[str, object]]) -> Parsed:
    """A response's answers by question ID (a Noul answer a float, a Choice answer a dict by option), or a refusal."""
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, dict):
        return Parsed({}, True)
    read: dict[str, Answer] = {}
    for name, question in questions.items():
        value = read_answer(answers.get(name), question)
        if value is None:
            return Parsed({}, True)
        read[name] = value
    return Parsed(read, False)


def g_scores(response: object, questions: Mapping[str, Mapping[str, object]]) -> GScores:
    """G's router scores: each success statement's answer, unchanged (A15 item 4). `questions` are G's, from
    `jev_call.g_questions`: their IDs are `success_<config>`. A refusal is 0 for every config (A15 item 9)."""
    parsed = parse_answers(response, questions)
    if parsed.refused:
        return GScores({config: 0.0 for config in SCORED}, True)
    return GScores({config: float(parsed.answers[f"success_{config}"]) for config in SCORED}, False)  # type: ignore[arg-type]


# ---- the client -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FailedAttempt:
    """An attempt that didn't answer: its status, or for a transport fault the error's name."""

    status: int | None
    error: str | None

    def to_record(self) -> dict:
        return {"status": self.status} if self.error is None else {"error": self.error, "status": self.status}


@dataclass(frozen=True)
class CallResult:
    """What a call returns: everything the stored record needs from it (issue #30's stored record).

    The key is in none of it: only the response's own fields and the client's counts."""

    body_sha256: str
    response: dict  # the whole JSON response
    model: str | None  # the versioned model ID that answered
    usage: dict | None  # input and output tokens
    request_id: str | None  # the `x-typesafe-request-id` header
    server_date: str | None  # the `Date` header
    seconds: float  # the successful attempt's, on a monotonic clock
    attempts: int  # including the successful one; a resend on a closed connection isn't one
    failed_attempts: tuple[FailedAttempt, ...]
    new_connection: bool  # the successful attempt opened its connection
    refused: bool

    def to_record(self) -> dict:
        """The stored record's fields from a call; the collector adds router, split, item, pass and collected_at."""
        return {
            "body_sha256": self.body_sha256,
            "response": self.response,
            "model": self.model,
            "usage": self.usage,
            "request_id": self.request_id,
            "server_date": self.server_date,
            "seconds": self.seconds,
            "attempts": self.attempts,
            "failed_attempts": [attempt.to_record() for attempt in self.failed_attempts],
            "new_connection": self.new_connection,
            "refused": self.refused,
        }


@dataclass(frozen=True)
class Exchange:
    status: int
    headers: dict[str, str]  # names lower-cased
    data: bytes
    seconds: float
    opened: bool  # the connection was opened for this send


# A reused connection the server closed fails with one of these on its first use.
CLOSED = (ConnectionError, http.client.RemoteDisconnected, http.client.ImproperConnectionState)


def retry_after(headers: Mapping[str, str]) -> float | None:
    """The wait a response asks for in seconds, from `retry-after-ms` or `Retry-After` (seconds); None if neither
    is a non-negative number. An HTTP-date isn't read: the wait then falls back to the backoff."""
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        try:
            seconds = float(headers[name]) / scale
        except (KeyError, ValueError):
            continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


def question_specs(body: bytes) -> dict[str, dict]:
    """The questions a body asks, by ID: what a response is read against."""
    return json.loads(body)["questions"]


def text_field(response: dict, name: str, kind: type) -> Any:
    value = response.get(name)
    return value if isinstance(value, kind) else None


class JevClient:
    """One kept-open connection, one call at a time. Use as a context manager, or call `close()`."""

    def __init__(
        self,
        settings: ClientSettings,
        environ: Mapping[str, str] | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        key = (os.environ if environ is None else environ).get(KEY_VARIABLE)
        if not key:
            raise MissingKeyError(f"{KEY_VARIABLE} is not set in the environment")
        self.settings = settings
        self._key = key
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter
        self._parts = urlsplit(settings.endpoint)
        self._connection: http.client.HTTPConnection | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, error: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _connect(self) -> http.client.HTTPConnection:
        host, port = self._parts.hostname or "", self._parts.port
        timeout = self.settings.retry.timeout_s
        if self._parts.scheme == "https":
            return http.client.HTTPSConnection(host, port, timeout=timeout)
        return http.client.HTTPConnection(host, port, timeout=timeout)

    def _send(self, body: bytes, connection: http.client.HTTPConnection) -> tuple[int, dict[str, str], bytes]:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        connection.request("POST", self._parts.path or "/", body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        return response.status, {name.lower(): value for name, value in response.getheaders()}, data

    def _exchange(self, body: bytes) -> Exchange:
        """One send and its reply. A reused connection that fails as closed is reopened and the body resent here, at
        once; any other failure, and a failure on a connection opened for this send, is raised to count as an attempt."""
        for first_pass in (True, False):
            if self._connection is None:
                self._connection = self._connect()
            # `sock` is None before the first send and after a `Connection: close`: http.client then connects by
            # itself, and that is a new connection to the call's latency.
            reused = self._connection.sock is not None
            start = self._clock()
            try:
                status, headers, data = self._send(body, self._connection)
            except CLOSED:
                self.close()
                if reused and first_pass:
                    continue
                raise
            except BaseException:
                self.close()  # a timeout or a fault leaves the connection in an unknown state
                raise
            seconds = self._clock() - start
            if headers.get("connection", "").lower() == "close":
                self.close()
            return Exchange(status, headers, data, seconds, opened=not reused)
        raise AssertionError("unreachable: the second pass returns or raises")

    def _backoff(self, failures: int) -> float:
        retry = self.settings.retry
        wait = min(retry.backoff_cap_s, retry.backoff_start_s * 2 ** (failures - 1))
        return wait * (1 - JITTER * self._jitter())

    def call(self, body: bytes, body_sha256: str, label: str) -> CallResult:
        """Send a body, retrying as the settings say; `label` names the request in any stop."""
        retry = self.settings.retry
        questions = question_specs(body)
        failed: list[FailedAttempt] = []
        while True:
            asked = None
            try:
                exchange = self._exchange(body)
            except TimeoutError:
                failed.append(FailedAttempt(None, "timeout"))
            except (OSError, http.client.HTTPException) as error:
                failed.append(FailedAttempt(None, type(error).__name__))
            else:
                status = exchange.status
                if status == 200:
                    try:
                        response = json.loads(exchange.data)
                    except ValueError:
                        response = None
                    if isinstance(response, dict):
                        return CallResult(
                            body_sha256,
                            response,
                            text_field(response, "model", str),
                            text_field(response, "usage", dict),
                            exchange.headers.get("x-typesafe-request-id"),
                            exchange.headers.get("date"),
                            exchange.seconds,
                            len(failed) + 1,
                            tuple(failed),
                            exchange.opened,
                            parse_answers(response, questions).refused,
                        )
                    failed.append(FailedAttempt(200, "unreadable response"))
                elif status in RETRIED or 500 <= status < 600:
                    failed.append(FailedAttempt(status, None))
                    asked = retry_after(exchange.headers)
                else:
                    self.close()
                    raise Rejected(f"{label}: status {status}: {self._excerpt(exchange.data)}")
            if len(failed) >= retry.attempts:
                self.close()
                raise RetryCapReached(f"{label}: {retry.attempts} attempts failed: {self._summary(failed)}")
            if asked is not None and asked > retry.retry_after_cap_s:
                self.close()
                raise RetryAfterTooLong(
                    f"{label}: asked to wait {asked:g} s, over the cap of {retry.retry_after_cap_s:g} s"
                )
            self._sleep(max(self._backoff(len(failed)), asked or 0.0))

    def _excerpt(self, data: bytes) -> str:
        """A reply's start for a stop's message, with the key struck out in case a server echoes it."""
        return data[:200].decode("utf-8", "replace").replace(self._key, "<key>")

    @staticmethod
    def _summary(failed: list[FailedAttempt]) -> str:
        return ", ".join(a.error or str(a.status) for a in failed)


# ---- G ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Asked:
    """One request through G: the call built, what came back, and the router scores."""

    call: Call
    result: CallResult
    scores: dict[str, float]


@dataclass
class Counts:
    """Counted events across a router's calls, reported by the collector (A15 item 9)."""

    calls: int = 0
    refused: int = 0
    cut: int = 0  # calls whose prompt the byte budget cut


class GRouter:
    """The Jev router G: `score(request)` builds the body, calls Jev and reads the three router scores."""

    def __init__(self, settings: Settings, client: JevClient, *, budget_bytes: int | None = None) -> None:
        self.settings = settings
        self.client = client
        self.budget_bytes = budget_bytes  # None: the settings file's; the smoke test gives its own (A15 item 13)
        self.counts = Counts()

    def ask(self, request: Request, label: str | None = None) -> Asked:
        """Score a request and keep the call's record. `label` names the request in a stop (an item ID); without
        one it is the body's hash and the prompt's start."""
        call = g_call(request, self.settings, budget_bytes=self.budget_bytes)
        name = label or f"{call.sha256[:12]} ({request.prompt[:40]!r})"
        result = self.client.call(call.body, call.sha256, name)
        scores = g_scores(result.response, g_questions(self.settings))
        self.counts.calls += 1
        self.counts.refused += scores.refused
        self.counts.cut += call.cut_bytes > 0
        return Asked(call, result, scores.scores)

    def score(self, request: Request) -> dict[str, float]:
        return self.ask(request).scores


@cache
def default_router() -> GRouter:
    return GRouter(read_settings(), JevClient(read_client_settings()))


def score(request: Request) -> dict[str, float]:
    """G's router scores for a request (the `Router` protocol's `score`, as `routers.rules` and `routers.trained`)."""
    return default_router().score(request)
