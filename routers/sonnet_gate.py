"""The Sonnet gate: its request, its gate score and its price (PLAN_v4.md §9; Amendment A17, items 6, 12 and 15).

The gate asks `claude-sonnet-5-5`, at low effort, for the probability from 0 to 100 that another model's answer is fully
correct. One call per answer. This module builds the request, reads the response into a gate score and prices the
call. It is deployable: it imports the standard library, the Anthropic SDK and the deployable package, and never the
harness or the analysis. The SDK client is passed in, so a check can hand it a fake; the key is the SDK's own business
(it reads `ANTHROPIC_API_KEY` itself) and is not read, held or stored here.

The request has no `temperature` (Claude 5.5 models reject it), no `thinking` field (omitted, so adaptive: the model
rejects a request that disables it, and effort is the only control), no refusal fallbacks, no `cache_control` and no
system prompt, as in the collection run (A1 item 1). Its shape is `harness.client.build_request`'s, written again here
because a deployable module can't import the harness; a check proves the two equal.

The gate score is `probability_correct` over 100. A refusal stop, a cap stop, any other stop but `end_turn` and a
result that is not exactly `{"probability_correct": <integer 0..100>}` all give no gate score and a refused call, which
a cascade treats as a rejection (`routers.cascade`). The billed tokens are read either way, and charged.

Everything a person might tune is in `routers/gate_settings.json`, under `sonnet`.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from routers.request import Request

SETTINGS_FILE = Path(__file__).with_name("gate_settings.json")
CUSTOM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # what the Batches API accepts as a request ID
END_TURN = "end_turn"
SCORE_KEY = "probability_correct"
LEVELS = 100  # the integer runs from 0 to this
_PLACEHOLDER = re.compile(r"\{(prompt|answer|output_schema|schema_block)\}")


class SonnetSettingsError(ValueError):
    """A Sonnet-gate settings section that fails a check. Nothing is built from it."""


@dataclass(frozen=True)
class SonnetSettings:
    model: str
    effort: str
    max_tokens: int
    prompt: str  # A17 item 6's message, with {prompt}, {schema_block} and {answer} to fill
    schema_block: str  # dropped when the request has no output schema; {output_schema} inside it
    output_schema: dict[str, Any]  # the output format: one integer from the enum 0..100
    request_id_suffix: str
    gated_configs: tuple[str, ...]  # whose answers are gated (A17 item 7)
    price_snapshot: str
    input_per_million: float
    output_per_million: float


def read_sonnet_settings(section: Mapping[str, Any]) -> SonnetSettings:
    """The `sonnet` section of the gate settings, or a refusal (SonnetSettingsError)."""
    try:
        settings = SonnetSettings(
            str(section["model"]), str(section["effort"]), int(section["max_tokens"]), str(section["prompt"]),
            str(section["schema_block"]), dict(section["output_schema"]), str(section["request_id_suffix"]),
            tuple(str(c) for c in section["gated_configs"]), str(section["list_prices"]["price_snapshot"]),
            float(section["list_prices"]["input_per_million"]), float(section["list_prices"]["output_per_million"]),
        )  # fmt: skip
    except (KeyError, TypeError, ValueError) as error:
        raise SonnetSettingsError(f"the sonnet section is not a Sonnet-gate settings section: {error!r}") from error
    if settings.effort not in ("low", "medium", "high", "xhigh", "max") or settings.max_tokens < 1:
        raise SonnetSettingsError(f"effort {settings.effort!r} and a cap of {settings.max_tokens} are not a request")
    wanted = {
        "type": "object",
        "properties": {SCORE_KEY: {"type": "integer", "enum": list(range(LEVELS + 1))}},
        "required": [SCORE_KEY],
        "additionalProperties": False,
    }
    if settings.output_schema != wanted:
        raise SonnetSettingsError(f"the output schema must be one integer {SCORE_KEY} with the enum 0 to {LEVELS}")
    for text, names in (
        (settings.prompt, {"prompt", "answer", "schema_block"}),
        (settings.schema_block, {"output_schema"}),
    ):
        if set(_PLACEHOLDER.findall(text)) != names:
            raise SonnetSettingsError(f"a message part must hold exactly the placeholders {sorted(names)}")
    if not CUSTOM_ID.match(f"x_{settings.request_id_suffix}") or not settings.gated_configs:
        raise SonnetSettingsError("the request-ID suffix must fit the API's limits and some config must be gated")
    if settings.input_per_million < 0 or settings.output_per_million < 0:
        raise SonnetSettingsError("a price per million tokens can't be negative")
    return settings


def load_settings(path: Path = SETTINGS_FILE) -> SonnetSettings:
    try:
        section = json.loads(path.read_text(encoding="utf-8"))["sonnet"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SonnetSettingsError(f"{path} has no readable sonnet section: {error!r}") from error
    return read_sonnet_settings(section)


# ---- the request ----------------------------------------------------------------------------------------------


def _fill(template: str, values: Mapping[str, str]) -> str:
    # One pass, so a prompt or an answer that spells `{answer}` is never substituted into.
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def build_request(settings: SonnetSettings, request: Request, answer: str) -> dict[str, Any]:
    """The Messages request (and, as `params`, the Batches request) that asks whether `answer` is fully correct."""
    block = ""
    if request.output_schema is not None:
        block = _fill(settings.schema_block, {"output_schema": json.dumps(request.output_schema, ensure_ascii=False)})
    content = _fill(settings.prompt, {"prompt": request.prompt, "schema_block": block, "answer": answer})
    return {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "output_config": {
            "effort": settings.effort,
            "format": {"type": "json_schema", "schema": settings.output_schema},
        },
        "messages": [{"role": "user", "content": content}],
    }


def request_sha256(params: Mapping[str, Any]) -> str:
    """The request's hash: its sorted, compact JSON, so the same request has one hash."""
    text = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def custom_id(settings: SonnetSettings, item_id: str, config: str, run: int = 0) -> str:
    """The batch request ID: the item ID with every character the API rejects (":") replaced by "-", then the config,
    the run and the suffix, e.g. `mmlu_pro-4346_opus_low_r0_sg` (A6 item 5; A17 item 14). Refuses (ValueError) one
    that doesn't fit the API's limits of 64 characters from letters, digits, `_` and `-`."""
    if not item_id or not config:
        raise ValueError("a request ID needs an item and a config")
    result = f"{re.sub(r'[^A-Za-z0-9_-]', '-', item_id)}_{config}_r{run}_{settings.request_id_suffix}"
    if not CUSTOM_ID.match(result):
        raise ValueError(f"{result!r} is not a request ID the API accepts")
    return result


# ---- the response ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """What a response says: the gate score (None for a refused call), why it was refused, the billed tokens and the
    model that answered. Thinking is inside `output_tokens`."""

    score: float | None
    refused: bool
    reason: str | None  # "refusal", "max_tokens", "stop_reason: ..." or "unparseable"; None for a gate score
    input_tokens: int
    output_tokens: int
    model: str | None


def _tokens(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"billed tokens must be a whole number, not {value!r}")
    return value


def parse_response(response: Mapping[str, Any]) -> Verdict:
    """The response's gate score and billed tokens. A response with no billed tokens is no response (ValueError):
    every call is charged what it was billed."""
    usage = response["usage"]
    input_tokens, output_tokens = _tokens(usage["input_tokens"]), _tokens(usage["output_tokens"])
    model = response.get("model")
    model = model if isinstance(model, str) else None

    def refused(reason: str) -> Verdict:
        return Verdict(None, True, reason, input_tokens, output_tokens, model)

    stop = response.get("stop_reason")
    if stop in ("refusal", "max_tokens"):
        return refused(str(stop))
    if stop != END_TURN:
        return refused(f"stop_reason: {stop}")
    text = "".join(
        b["text"] for b in response.get("content") or [] if isinstance(b, Mapping) and b.get("type") == "text"
    )
    try:
        parsed = json.loads(text)
    except ValueError:
        return refused("unparseable")
    if not isinstance(parsed, dict) or set(parsed) != {SCORE_KEY}:
        return refused("unparseable")
    number = parsed[SCORE_KEY]
    if isinstance(number, bool) or not isinstance(number, int) or not 0 <= number <= LEVELS:
        return refused("unparseable")
    return Verdict(number / LEVELS, False, None, input_tokens, output_tokens, model)


def overhead_dollars(settings: SonnetSettings, verdict: Verdict) -> float:
    """What the call cost at 2026-10-12's list prices: billed input tokens and billed output tokens (thinking is
    output), with no batch discount (A17 item 15). A refused call is charged what it was billed."""
    return (
        verdict.input_tokens * settings.input_per_million + verdict.output_tokens * settings.output_per_million
    ) / 1e6


# ---- one live call --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    verdict: Verdict
    params: dict[str, Any]  # the request sent
    response: dict[str, Any]  # the whole response, as the SDK's `to_dict(mode="json")`
    request_id: str | None


def make_client(max_retries: int = 5) -> anthropic.Anthropic:
    """The SDK client; the SDK finds the key in the environment itself."""
    return anthropic.Anthropic(max_retries=max_retries)


def check_answer(client: anthropic.Anthropic, settings: SonnetSettings, request: Request, answer: str) -> CheckResult:
    """Ask the standard API whether `answer` is fully correct: one call, with no retry of its own beyond the SDK's.
    An API error (a 4xx, the SDK's retries spent) is raised, never a verdict."""
    params = build_request(settings, request, answer)
    message = client.messages.create(**params)
    response = message.to_dict(mode="json")
    return CheckResult(parse_response(response), params, response, getattr(message, "_request_id", None))
