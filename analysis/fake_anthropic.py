"""A fake Anthropic client for the checks: Messages and Message Batches (PLAN_v4.md §9; Amendment A17, items 6 and 14;
A6 item 6).

It lives in analysis/, outside the routers' import closure, beside `analysis/fake_jev.py`. It stands in for an
`anthropic.Anthropic` client: `client.messages.create(**params)` and `client.messages.batches.create`, `.retrieve` and
`.results`, answering with the SDK's own types (`Message`, `MessageBatchIndividualResponse`), so the code under test
reads what it would read from the API. Checks built on `harness/check_collect.FakeBatches` could not import it: that
file needs Docker and the results folder as a whole.

What it does:
- Every request that reached it is kept, whole, in `calls` (Messages) and `created` (Batches), so a check asserts what
  was sent, not what the code meant to send. A batch request the real API would refuse (a request ID outside
  `^[A-Za-z0-9_-]{1,64}$`, two with the same ID, or one without `params`) raises here too.
- What the gate "thinks" comes from `respond`, which reads the request's content and returns a Message as a dict. The
  default reads a marker in the answer: `SCORE=70` gives 70; `REFUSE`, `CAP` and `GARBAGE` give a refusal stop, an
  output-cap stop and text that is no JSON; no marker gives 50. Tokens billed are the content's length over four, in,
  and 5, out, or 16000 out for `CAP`.
- `batch_failures` maps a request ID to `"errored"`, `"expired"` or `"canceled"`: that request fails the first time
  its batch's results are read and succeeds in the batch that resends it. `create_error_after` makes the Nth
  `batches.create` raise (to test what was saved before it). `polls_before_end` keeps a batch "in_progress" for that
  many `retrieve`s. `tamper` may rewrite a batch's results once built (an unrequested ID, a repeated ID).
- Results arrive shuffled, and the same on a second read.
- `message_errors` maps a call number of `messages.create` (from 1) to the exception raised on it.

Run as a library only:  from analysis.fake_anthropic import FakeAnthropic
"""

import json
import random
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from anthropic.types import Message
from anthropic.types.messages import MessageBatchIndividualResponse

MODEL = "claude-sonnet-5-5"
CUSTOM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # the API's rule for a batch request's custom_id


def message_dict(text: str, stop: str = "end_turn", usage: tuple[int, int] = (100, 5), model: str = MODEL) -> dict:
    """A Messages response, in the shape the SDK validates."""
    return {
        "id": "msg_fake", "type": "message", "role": "assistant", "model": model, "stop_reason": stop,
        "stop_sequence": None, "content": [{"type": "text", "text": text}] if text else [],
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    }  # fmt: skip


def default_respond(params: Mapping[str, Any]) -> dict:
    """The default "gate": a marker in the answer decides (see the module's docstring)."""
    content = params["messages"][0]["content"]
    answer = content.split("<answer>\n", 1)[1].rsplit("\n</answer>", 1)[0] if "<answer>\n" in content else ""
    billed = (len(content) // 4, 5)
    if "REFUSE" in answer:
        return message_dict("", "refusal", billed)
    if "CAP" in answer:
        return message_dict("", "max_tokens", (billed[0], 16000))
    if "GARBAGE" in answer:
        return message_dict("probably fine", "end_turn", billed)
    marker = re.search(r"SCORE=(\d+)", answer)
    return message_dict(json.dumps({"probability_correct": int(marker.group(1)) if marker else 50}), "end_turn", billed)


class FakeBatches:
    """`client.messages.batches`."""

    def __init__(self, owner: "FakeAnthropic"):
        self.owner = owner
        self.created: dict[str, list[dict]] = {}
        self.retrieves = 0
        self.results_reads = 0
        self._outcomes: dict[str, list] = {}

    def create(self, requests):
        if (n := self.owner.create_error_after) is not None and len(self.created) >= n:
            raise RuntimeError("fake: the connection dropped while creating a batch")
        ids = [r["custom_id"] for r in requests]
        if bad := [i for i in ids if not CUSTOM_ID.match(i)]:
            raise ValueError(f"fake API: custom_id {bad[0]!r} is not ^[A-Za-z0-9_-]{{1,64}}$")
        if len(set(ids)) != len(ids):
            raise ValueError("fake API: custom_ids must be unique within a batch")
        if any("params" not in r for r in requests):
            raise ValueError("fake API: every request needs params")
        batch_id = f"msgbatch_fake{len(self.created)}"
        self.created[batch_id] = [json.loads(json.dumps(r)) for r in requests]
        return SimpleNamespace(id=batch_id, created_at=datetime.now(timezone.utc))

    def retrieve(self, batch_id):
        self.retrieves += 1
        ended = self.retrieves > self.owner.polls_before_end
        return SimpleNamespace(
            id=batch_id, processing_status="ended" if ended else "in_progress",
            request_counts=SimpleNamespace(processing=0 if ended else 1, succeeded=0, errored=0),
        )  # fmt: skip

    def results(self, batch_id):
        self.results_reads += 1
        if batch_id not in self._outcomes:
            outcomes = []
            for request in self.created[batch_id]:
                custom_id = request["custom_id"]
                kind = self.owner.batch_failures.pop(custom_id, "succeeded")
                if kind == "succeeded":
                    result: dict = {"type": "succeeded", "message": self.owner.respond(request["params"])}
                elif kind == "errored":
                    result = {"type": "errored",
                              "error": {"type": "error", "error": {"type": "api_error", "message": "fake"}}}  # fmt: skip
                else:
                    result = {"type": kind}
                outcomes.append({"custom_id": custom_id, "result": result})
            random.Random(len(self._outcomes)).shuffle(outcomes)  # results arrive in any order
            if self.owner.tamper is not None:
                outcomes = self.owner.tamper(batch_id, outcomes)
            self._outcomes[batch_id] = [MessageBatchIndividualResponse.model_validate(o) for o in outcomes]
        return iter(self._outcomes[batch_id])


class FakeMessages:
    """`client.messages`: `create`, and `batches`."""

    def __init__(self, owner: "FakeAnthropic"):
        self.owner = owner
        self.batches = FakeBatches(owner)

    def create(self, **params):
        self.owner.calls.append(json.loads(json.dumps(params)))
        error = self.owner.message_errors.get(len(self.owner.calls))
        if error is not None:
            raise error
        return Message.model_validate(self.owner.respond(params))


class FakeAnthropic:
    def __init__(
        self,
        *,
        respond: Callable[[Mapping[str, Any]], dict] = default_respond,
        batch_failures: Mapping[str, str] | None = None,
        create_error_after: int | None = None,
        polls_before_end: int = 0,
        tamper: Callable[[str, list[dict]], list[dict]] | None = None,
        message_errors: Mapping[int, Exception] | None = None,
    ):
        self.respond = respond
        self.batch_failures = dict(batch_failures or {})
        self.create_error_after = create_error_after
        self.polls_before_end = polls_before_end
        self.tamper = tamper
        self.message_errors = dict(message_errors or {})
        self.calls: list[dict] = []  # every messages.create(**params) received
        self.messages = FakeMessages(self)

    @property
    def created(self) -> dict[str, list[dict]]:
        return self.messages.batches.created
