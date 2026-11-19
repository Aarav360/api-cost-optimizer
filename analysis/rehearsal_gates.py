"""The rehearsal's Jev gate stand-ins: a made-up Stage 7 noise pass and a fake Jev server (PLAN_v4.md Amendment A19, items 2, 7
and 13; A15 item 17; A17 item 26; issue #77, #85).

Not a check: a module `analysis/check_final_run.py`'s rehearsal and `analysis/check_final_gates.py` import. It gives a made-up
results folder what the Jev gate's drift check reads from Stage 7, and a Jev server that answers as told:

- **Stage 7's side**: `jev_gate_train.jsonl` in the folder's `gates/`, ten made-up train answers of `haiku` with a first pass
  (`FIRST`), eight of them with a second (`SECOND`: the noise sample), each record built as the collector builds one (the
  frozen gate call's body and its sha256, the response the fake would give, `STORED_FIELDS` in order). Hand-worked: D6 is
  0.02 for `haiku` (every second pass is 0.02 from the first, four above and four below) and the shift 0.
- **The fake** (`ScriptedJev`): a real HTTP server (`analysis/fake_jev.py`) answering each call from `scores`, the gate score it
  should come out as, by the request's prompt (all three statements say it, the inverted one as the opposite). It answers as
  the pin unless `model` says otherwise, can refuse a numbered call with a 400 (`fail_at`), and logs a call it has no score
  for in `unexpected`. The resend `RESENDS` names are scenarios: what the drift check's resend scores are.
- **The test side**: `TEST_SCORES`, the gate score of each of the rehearsal world's eight test items, whatever config answered.

The client is the frozen `routers/jev.py` one, pointed at the fake's loopback port with its waits skipped. Nothing here touches
`results/`, the real store or the real endpoint.
"""

import dataclasses
import json
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from analysis import fake_jev
from analysis import rehearsal_world as rehearsal
from analysis.collect_gates import GateInput
from analysis.collect_jev import append_record
from analysis.collect_jev_gate import STORED_FIELDS
from analysis.fake_jev import FakeJev, Reply, noul_answer, response_body
from routers import jev, jev_gate
from routers.request import Request

PIN = "jev-1.14.0"
TOKENS = 1000  # input tokens a made-up call is billed
CONFIG = "haiku"
# Stage 7's noise sample: items 0 to 7, each with a first pass and a second; items 8 and 9 have a first pass only, so the
# drift check must take the sample from the passes, not from every record.
FIRST = (0.95, 0.83, 0.81, 0.74, 0.72, 0.64, 0.60, 0.52, 0.90, 0.15)
SECOND = (0.97, 0.81, 0.83, 0.72, 0.74, 0.62, 0.62, 0.50)  # 0.02 from the first, above for even items, below for odd
SAMPLE = tuple(range(len(SECOND)))
D6, D8_LIMIT = 0.02, 0.06  # by hand: every second pass is 0.02 from the first; 3 x 0.02
# What the drift check's resend scores are, by scenario: the first pass moved by a fixed amount, item by item.
RESENDS = {
    "quiet": tuple(0.03 if k % 2 == 0 else -0.03 for k in SAMPLE),  # D8 0.03, shift 0: within both limits
    "loud": tuple(-0.08 if k % 2 == 0 else 0.08 for k in SAMPLE),  # D8 0.08 > 3 x 0.02, shift 0
    "uniform": tuple(0.03 for _ in SAMPLE),  # D8 0.03 (well inside 3 x D6), shift +0.03 > 0.02: the shift rule alone
}
# The rehearsal world's eight test items' gate scores (banking 0, 1; MMLU-Pro 0, 1; IFEval 0, 1; MBPP+ 0, 1).
TEST_SCORES = (0.99, 0.60, 0.90, 0.30, 0.82, 0.75, 0.97, 0.50)


def train_prompt(k: int) -> str:
    return f"Which kind of card problem is number {k}?"


def train_answer(k: int) -> str:
    return f"It is a card problem of kind {k}."


def train_input(k: int) -> GateInput:
    return GateInput(f"banking77:made-train-{k}", CONFIG, Request(train_prompt(k), None), train_answer(k), "end_turn")


def gate_train(pairs: Sequence[tuple[str, str]]) -> list[GateInput]:
    """The `Backends.gate_train` stand-in: the made-up train answers named by the pairs."""
    by_key = {(i.item_id, i.config): i for i in (train_input(k) for k in range(len(FIRST)))}
    return [by_key[pair] for pair in pairs]


def answers_for(score: float, settings: jev_gate.JevGateSettings) -> dict[str, dict]:
    """The three statements' answers that give `score`: the inverted one says the opposite, as Jev's rounding would."""
    return {s.id: noul_answer(round(1 - score, 2) if s.inverted else score) for s in settings.statements}


def stored_record(k: int, number: int, score: float, settings: jev_gate.JevGateSettings) -> dict:
    """One stored Jev gate record of train item `k`, pass `number`, as the collector stores it."""
    item = train_input(k)
    call = jev_gate.gate_call(item.request, item.answer, settings, budget_bytes=settings.budget_bytes)
    record = {
        "item_id": item.item_id, "config": CONFIG, "run": 0, "split": "train", "gate": "jev", "pass": number,
        "body_sha256": call.sha256, "response": response_body(answers_for(score, settings), model=PIN, input_tokens=TOKENS),
        "model": PIN, "usage": {"input_tokens": TOKENS, "output_tokens": 38}, "request_id": f"req_{k:032x}",
        "server_date": "Wed, 28 Oct 2026 00:01:57 GMT", "seconds": 0.5, "attempts": 1, "failed_attempts": [],
        "new_connection": k == 0, "refused": False, "answer_cut_bytes": call.answer_cut_bytes,
        "prompt_cut_bytes": call.prompt_cut_bytes, "collected_at": "2026-11-10T12:00:00-08:00",
    }  # fmt: skip
    return {key: record[key] for key in STORED_FIELDS}


def write_train(results: Path, settings: jev_gate.JevGateSettings | None = None) -> Path:
    """Stage 7's made-up `jev_gate_train.jsonl` under `results/gates/`: pass 1 of ten answers, then pass 2 of eight."""
    settings = settings or jev_gate.read_settings()
    path = results / "gates" / "jev_gate_train.jsonl"
    for k, score in enumerate(FIRST):
        append_record(path, stored_record(k, 1, score, settings))
    for k, score in enumerate(SECOND):
        append_record(path, stored_record(k, 2, score, settings))
    return path


class ScriptedJev(FakeJev):
    """A fake Jev that gives each prompt its scripted gate score, as `model`, and refuses chosen calls with a 400."""

    def __init__(self, scores: Mapping[str, float], settings: jev_gate.JevGateSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or jev_gate.read_settings()
        self.scores = dict(scores)  # prompt -> gate score
        self.model = PIN
        self.tokens = TOKENS
        self.fail_at: set[int] = set()  # call numbers (from 1) answered with a 400
        self.calls = 0
        self.bodies: list[dict] = []
        # The replay's check (#91): a wait before each answer (a real one, on the server), the numbers of the calls the
        # server hangs up after (so the client's next call reconnects), and each call's (start, end) on `time.perf_counter`.
        self.delay_s = 0.0
        self.hang_up_after: set[int] = set()
        self.intervals: list[tuple[float, float]] = []

    def reply_to(self, body: bytes) -> Reply:
        self.calls += 1
        decoded = json.loads(body)
        self.bodies.append(decoded)
        prompt = decoded["state"]["request"]["prompt"]
        started = time.perf_counter()
        self.intervals.append((started, started + self.delay_s))  # the handler waits `delay_s` before it replies
        if self.calls in self.fail_at:
            return Reply(400, {"detail": "made up"})
        if prompt not in self.scores:
            self.unexpected.append(f"no score for {prompt!r}")
            return Reply(422, {"detail": "the fake has no score for this prompt"})
        return Reply(200, response_body(answers_for(self.scores[prompt], self.settings), model=self.model,
                                        input_tokens=self.tokens), delay_s=self.delay_s,
                     hang_up_after=self.calls in self.hang_up_after)  # fmt: skip

    def __enter__(self) -> "ScriptedJev":
        """Start a server on a fresh port, as often as asked: a step's session runs it only while it calls, since grading
        needs no other thread running. What was received and the call count carry over from one session to the next."""
        self._server = fake_jev._Server(("127.0.0.1", 0), fake_jev._handler_for(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, error, value, traceback) -> None:  # type: ignore[no-untyped-def]
        super().__exit__(error, value, traceback)
        self._thread.join()
        deadline = time.monotonic() + 5.0
        while threading.active_count() > 1 and time.monotonic() < deadline:
            time.sleep(0.005)  # a connection's handler thread ends once the client has closed it

    def resend(self, scenario: str) -> None:
        """Set the train prompts' scores to the scenario's resend: the first pass moved by the scenario's amounts."""
        for k, move in zip(SAMPLE, RESENDS[scenario], strict=True):
            self.scores[train_prompt(k)] = round(FIRST[k] + move, 4)


def test_scores() -> dict[str, float]:
    return {made["prompt"]: score for made, score in zip(rehearsal.ITEMS, TEST_SCORES, strict=True)}


def session_factory(fake: ScriptedJev) -> Callable[[], AbstractContextManager[jev.JevClient]]:
    """A factory of sessions, as `Backends.jev` takes: each starts the fake, gives the frozen client pointed at its port with
    its waits skipped and a made-up key, and stops the fake when the step is done with it."""

    @contextmanager
    def session() -> Iterator[jev.JevClient]:
        with fake:
            settings = dataclasses.replace(jev.read_client_settings(), endpoint=fake.url)
            with jev.JevClient(settings, {jev.KEY_VARIABLE: "rehearsal-not-a-key"}, sleep=lambda s: None,
                               jitter=lambda: 0.0) as client:  # fmt: skip
                yield client

    return session


@dataclass(frozen=True)
class GateStandIn:
    """What a rehearsal's backends take: the fake server (running only while a step calls it) and the two stand-ins."""

    fake: ScriptedJev
    jev: Callable[[], AbstractContextManager[jev.JevClient]]
    gate_train: Callable[[Sequence[tuple[str, str]]], list[GateInput]] = field(default=gate_train)


@contextmanager
def stand_in(results: Path, scenario: str = "quiet") -> Iterator[GateStandIn]:
    """Stage 7's made-up passes written under `results/gates/`, and a fake that answers the `scenario`'s resends and the test
    items' scores when a step starts it."""
    write_train(results)
    fake = ScriptedJev({**test_scores(), **{train_prompt(k): FIRST[k] for k in range(len(FIRST))}})
    fake.resend(scenario)
    yield GateStandIn(fake, session_factory(fake))


@contextmanager
def lone(scenario: str = "quiet") -> Iterator[tuple[Path, GateStandIn]]:
    """A folder with the stand-ins and nothing else, for a check that doesn't need a world."""
    with tempfile.TemporaryDirectory() as folder, stand_in(Path(folder), scenario) as gates:
        yield Path(folder), gates
