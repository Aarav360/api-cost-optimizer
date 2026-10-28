"""A fake Jev server for the checks (PLAN_v4.md §8 3c; Amendment A15, items 3 and 9; issue #30's checks).

It lives in analysis/, never in the router's import closure, like `analysis/fake_backend.py`. It is a real HTTP/1.1
server on a loopback port, so the router's own `http.client` code runs end to end: one kept-open connection, status
lines, headers and timeouts are the real ones. No check calls the real Jev (AGENTS.md; issue #30).

What it does:
- Answers come from `answers`, a map from question ID to a Noul value (a float) or a Choice's option probabilities
  (a dict). A question it has no answer for gets a 422 and is recorded in `unexpected`, so a check can't pass by
  asking something it didn't expect: `assert fake.unexpected == []`.
- `script` is a list of `Reply`s served first, one per call received, before it falls back to `answers`. A reply can
  carry any status, headers (`Retry-After`), a body, a delay (to exceed the client's timeout), a hang-up after it (an
  idle disconnect: the next call on that connection fails) or a drop (the server closes without replying).
- Everything that reached it is in `received`: the body bytes, the headers (so a check can see the key arrive) and
  the number of the connection it came on.

The response builders (`noul_answer`, `choice_answer`, `response_body`) give hand-made responses in the shape
recorded from live calls on made-up text (Jev's real answer shapes: typesafe_api notes, 2026-10-28).

Run as a library only:  from analysis.fake_jev import FakeJev
"""

import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Self

PATH = "/v1/systemone"
MODEL = "jev-test-0"


def noul_answer(value: float) -> dict:
    return {"type": "noul", "noul": value}


def choice_answer(probabilities: Mapping[str, float]) -> dict:
    best = max(probabilities, key=lambda option: probabilities[option])
    return {"type": "choice", "choice": best, "confidence": probabilities[best], "probabilities": dict(probabilities)}


def response_body(answers: Mapping[str, dict], *, model: str = MODEL, input_tokens: int = 300) -> dict:
    """A 200 body: the model, the answers by question ID and the usage, in the key order of live calls."""
    return {"model": model, "answers": dict(answers), "usage": {"input_tokens": input_tokens, "output_tokens": 38}}


@dataclass
class Reply:
    status: int = 200
    body: object = None  # a dict or list is sent as JSON, str or bytes as they are; None means an empty JSON object
    headers: dict[str, str] = field(default_factory=dict)
    delay_s: float = 0.0  # waited before replying, to outlast a client's timeout
    hang_up_after: bool = False  # reply, then close the connection without saying so: an idle disconnect
    drop: bool = False  # close the connection on receiving the call, replying nothing


@dataclass(frozen=True)
class Received:
    body: bytes
    headers: dict[str, str]  # names lower-cased
    connection: int  # which connection it came on, from 1


class FakeJev:
    """Use as a context manager: `with FakeJev(answers={...}) as fake:`, then `fake.url` is the endpoint."""

    def __init__(self, answers: Mapping[str, dict] | None = None, script: list[Reply] | None = None) -> None:
        self.answers = dict(answers or {})  # question ID -> the answer object to give (build with the helpers)
        self.script = list(script or [])
        self.received: list[Received] = []
        self.unexpected: list[str] = []
        self.connections = 0
        self.date = "Wed, 28 Oct 2026 00:01:57 GMT"
        self.request_ids = 0
        self.input_tokens = 300
        self._server = _Server(("127.0.0.1", 0), _handler_for(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}{PATH}"

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self, error: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._server.shutdown()
        self._server.server_close()

    def reply_to(self, body: bytes) -> Reply:
        """The next scripted reply, else the answers to the questions the body asks, else a loud 422."""
        if self.script:
            return self.script.pop(0)
        try:
            questions = json.loads(body)["questions"]
            answers = {name: self.answers[name] for name in questions}
        except (ValueError, KeyError, TypeError) as error:
            self.unexpected.append(f"{error!r}: {body[:200]!r}")
            return Reply(422, {"detail": [{"type": "missing", "msg": "the fake has no answer for this body"}]})
        return Reply(200, response_body(answers, input_tokens=self.input_tokens))


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        # A client that timed out closes its socket, so the server's late write fails. That is the scenario under
        # test, not an error worth a traceback in a check's output.
        pass


def _handler_for(fake: FakeJev) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, as Jev's

        def setup(self) -> None:
            super().setup()
            fake.connections += 1
            self.number = fake.connections

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            headers = {name.lower(): value for name, value in self.headers.items()}
            fake.received.append(Received(body, headers, self.number))
            reply = fake.reply_to(body)
            if reply.drop:
                self.close_connection = True
                return
            time.sleep(reply.delay_s)
            if isinstance(reply.body, bytes):
                data = reply.body
            elif isinstance(reply.body, str):
                data = reply.body.encode()
            else:
                data = json.dumps({} if reply.body is None else reply.body).encode()
            fake.request_ids += 1
            self.send_response_only(reply.status)
            self.send_header("Date", fake.date or formatdate(usegmt=True))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("x-typesafe-request-id", f"req_{fake.request_ids:032x}")
            for name, value in reply.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)
            if reply.hang_up_after:
                self.close_connection = True

        def log_message(self, format: str, *args: object) -> None:
            pass  # a check's output is its own

    return Handler
