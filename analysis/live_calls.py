"""The live calls' request builder and sender: what the liveness calls and the live replay both send (PLAN_v4.md
Amendment A19, items 2.2 and 8; A17 item 26.5; issue #77, #82, #91).

One builder makes every request of the live calls, so the liveness calls reach the API exactly as the replay's do: the
collection harness's own `build_request` (`harness/client.py`), with the config's pinned model, its explicit effort and its
own `max_tokens` from `configs/models.yaml` (A1 item 1, A2). One sender makes the call: `harness.client.call`, which
streams (the SDK refuses a non-streaming request with a cap as high as haiku's 64,000 tokens) and times the call from
request to final message, with a store that never holds anything, so a live call can't be answered from a cache and
check nothing.

The replay (#91) builds its requests with `live_request` and sends them with `send`.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harness.client import build_request, call, load_models
from harness.store import ResponseStore


class Uncached(ResponseStore):
    """A response store that remembers nothing: every `get` misses and every `put` is dropped."""

    def get(self, request: dict, run: int = 0) -> dict | None:
        return None

    def put(self, request: dict, run: int, record: dict) -> Path:
        return self.root


UNCACHED = Uncached(".")


def live_request(config: str, prompt: str, *, output_schema: dict | None = None, models: Mapping | None = None) -> dict:
    """The request a live call sends for `config` (a name in `configs/models.yaml`): its pinned model and effort."""
    models = models or load_models()
    settings = models["configs"][config]
    return build_request(settings, prompt, settings["max_tokens"], output_schema=output_schema)


def send(client: Any, request: dict) -> dict:
    """One standard-API call, never from a cache. The record: `run`, `request`, `response` (the final message as a dict),
    `request_id`, `latency_s` and `created_at` (`harness.client.call`'s)."""
    record, _ = call(client, UNCACHED, request)
    return record


def worst_case_dollars(request: dict, models: Mapping | None = None) -> float:
    """An upper bound on a request's cost: its prompt's bytes as input tokens (one token per byte, which no tokenizer
    beats) and the whole of `max_tokens` as output, at the requested model's list prices."""
    models = models or load_models()
    price = models["prices"][request["model"]]
    prompt_bytes = sum(len(message["content"].encode("utf-8")) for message in request["messages"])
    return (prompt_bytes * float(price["input"]) + request["max_tokens"] * float(price["output"])) / 1e6
