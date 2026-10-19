"""Standard-API calls for the harness, cached through the response store.

Bulk collection (dev matrix, sealed test answers) goes through the Message
Batches API instead (Amendment A1, item 5); this module is for the pilot and
small live runs, where per-call latency matters.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv

from harness.store import ResponseStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reads ANTHROPIC_API_KEY from the project's .env without printing it.
load_dotenv(PROJECT_ROOT / ".env")


def load_models(path: Path = PROJECT_ROOT / "configs" / "models.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def make_client(max_retries: int = 5) -> anthropic.Anthropic:
    # The SDK retries 408/409/429/5xx and connection errors with backoff.
    return anthropic.Anthropic(max_retries=max_retries)


def build_request(
    config: dict, prompt: str, max_tokens: int, system: str | None = None, output_schema: dict | None = None
) -> dict:
    # No `temperature` (Claude 5.5 models reject it) and no `thinking` field
    # (omitting it runs adaptive thinking on all three models). Effort is the
    # only knob, and it is always set explicitly. `output_schema` turns on
    # structured outputs: the answer text is then JSON matching the schema.
    request = {
        "model": config["model"],
        "max_tokens": max_tokens,
        "output_config": {"effort": config["effort"]},
        "messages": [{"role": "user", "content": prompt}],
    }
    if output_schema is not None:
        request["output_config"]["format"] = {"type": "json_schema", "schema": output_schema}
    if system is not None:
        request["system"] = system
    return request


def call(client: anthropic.Anthropic, store: ResponseStore, request: dict, run: int = 0) -> tuple[dict, bool]:
    """Return (record, from_cache). Only successful API responses are stored.

    A `refusal` or `max_tokens` stop is still a successful response: it is
    stored, and scoring counts it as a failure (Amendment A1, item 1). API
    errors raise and are never cached.
    """
    cached = store.get(request, run)
    if cached is not None:
        return cached, True

    # Streamed, because the SDK refuses non-streaming requests whose max_tokens
    # could run past 10 minutes (about 21K tokens). The stored response is the
    # same final message either way, and latency is still request to last token.
    start = time.perf_counter()
    with client.messages.stream(**request) as stream:
        response = stream.get_final_message()
    latency_s = time.perf_counter() - start

    record = {
        "run": run,
        "request": request,
        "response": response.to_dict(mode="json"),
        "request_id": stream.request_id,
        "latency_s": latency_s,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    store.put(request, run, record)
    return record, False


def answer_text(response: dict) -> str:
    # Thinking blocks come back with empty text by default; only text blocks
    # are the model's answer.
    return "".join(block["text"] for block in response["content"] if block["type"] == "text")


def cost_usd(record: dict, prices: dict, batch_discount: float = 1.0) -> float:
    # Priced by the requested (pinned) model, so a mismatch between the
    # requested and reported model shows up as a logged discrepancy rather
    # than a KeyError here.
    usage = record["response"]["usage"]
    price = prices[record["request"]["model"]]
    dollars = (usage["input_tokens"] * price["input"] + usage["output_tokens"] * price["output"]) / 1_000_000
    return dollars * batch_discount
