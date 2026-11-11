"""The Jev gate: call building and the gate score (PLAN_v4.md §9; Amendment A17, item 5; A15 items 7 and 8; A16 items 7
and 8).

One call per answer to `jev-1.14.0` (A13). The Jev state holds the request and the answer, `request` before `answer`,
a missing schema as null, serialized as A15 item 8 does (compact separators, non-ASCII kept, UTF-8, keys in the code's
order, never sorted). Three Noul statements are asked, under the IDs the settings hold; the one marked `inverted`
(`jev_gate_mistake`) is read as 1 - p.

The gate score. Each answer is clipped to [0.005, 0.995] (Jev rounds to two decimals and can answer exactly 0.00 or
1.00, whose log-odds are infinite), the inverted one is turned, the three are averaged on the log-odds scale and the mean
goes back to a probability. Clipping is symmetric, so clipping before or after the inversion gives the same score.

The byte budget follows A16 item 8's method, as `jev_call.g_budget_bytes` does for G: the tighter of Jev's two limits
less a measured fixed overhead, the state's serialized size with an empty prompt and an empty answer (the longest output
schema included, as a reserve) and the questions. It is a budget for the prompt and the answer together, in serialized
bytes at one byte per token. A call over it cuts the answer first, then the prompt, each keeping its head and its tail,
and the bytes cut from each are counted in the call.

The frozen client and parser (`routers/jev.py`) send the call and read its response; this module neither edits them
nor sends anything. A refusal is A16 item 7's rule, `jev.parse_answers`. Settings: the `jev` section of
`routers/gate_settings.json`. Standard library and the deployable package only.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routers import jev, jev_call
from routers.request import Request

SETTINGS_FILE = Path(__file__).with_name("gate_settings.json")


class GateCallError(ValueError):
    """A settings section, budget or response this module refuses whole."""


@dataclass(frozen=True)
class GateStatement:
    id: str
    text: str
    inverted: bool  # read as 1 - p


@dataclass(frozen=True)
class JevGateSettings:
    model: str
    statements: tuple[GateStatement, ...]
    clip: tuple[float, float]
    budget_bytes: int | None  # null until the smoke test (A17 item 20) fills it in


@dataclass(frozen=True)
class GateCall:
    """A call body, ready to send, with its sha256 and the serialized bytes the cut dropped from each field."""

    body: bytes
    sha256: str
    answer_cut_bytes: int
    prompt_cut_bytes: int


@dataclass(frozen=True)
class Fitted:
    prompt: str
    answer: str
    answer_cut_bytes: int
    prompt_cut_bytes: int


@dataclass(frozen=True)
class GateResponse:
    """A response read against the three statements: the gate score, or none and `refused`."""

    score: float | None
    refused: bool


def parse_settings(data: Mapping[str, Any]) -> JevGateSettings:
    """The `jev` section of the gate settings file, refusing it whole if any part is missing or malformed."""
    try:
        model = data["model"]
        statements = tuple(GateStatement(s["id"], s["text"], s["inverted"]) for s in data["statements"])
        clip = tuple(data["clip"])
        budget = data["budget_bytes"]
    except (KeyError, TypeError) as error:
        raise GateCallError(f"the Jev gate's settings lack or misshape {error!r}") from error
    if not jev_call.is_text(model):
        raise GateCallError("the model is not a non-empty string")
    if len(statements) != 3 or len({s.id for s in statements}) != 3:
        raise GateCallError("the Jev gate asks three statements under distinct IDs")
    if not all(jev_call.is_text(s.id) and jev_call.is_text(s.text) and type(s.inverted) is bool for s in statements):
        raise GateCallError("a statement needs an ID, a text and an inverted flag that is a bool")
    if len(clip) != 2 or not all(type(c) in (int, float) for c in clip) or not 0 < clip[0] < clip[1] < 1:
        raise GateCallError("clip must be two numbers with 0 < low < high < 1")
    if budget is not None and not (type(budget) is int and budget >= 1):
        raise GateCallError("the budget is null or a whole number of bytes of at least 1")
    return JevGateSettings(model, statements, (float(clip[0]), float(clip[1])), budget)


def read_settings(path: Path = SETTINGS_FILE) -> JevGateSettings:
    try:
        return parse_settings(json.loads(path.read_text(encoding="utf-8"))["jev"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, GateCallError):
            raise
        raise GateCallError(f"{path} has no readable jev section: {error!r}") from error


def gate_questions(settings: JevGateSettings) -> dict[str, dict]:
    """The three statements as Noul questions, under their IDs, in the settings' order."""
    return {s.id: {"type": "noul", "instructions": s.text} for s in settings.statements}


def gate_state(prompt: str, schema: dict | None, answer: str) -> dict:
    """The Jev state (A17 item 5): the request, then the answer."""
    return jev_call.g_state(prompt, schema) | {"answer": answer}


def fit_to_budget(prompt: str, answer: str, budget_bytes: int) -> Fitted:
    """The prompt and answer within a budget of serialized bytes together: the answer is cut first, to the room the
    prompt leaves, and when the prompt alone is over the budget the answer goes entirely and the prompt is cut."""
    prompt_bytes = sum(jev_call.serialized_size(char) for char in prompt)
    answer, answer_cut = jev_call.cut_prompt(answer, max(0, budget_bytes - prompt_bytes))
    prompt, prompt_cut = jev_call.cut_prompt(prompt, budget_bytes)
    return Fitted(prompt, answer, answer_cut, prompt_cut)


def gate_budget_bytes(settings: JevGateSettings, *, overhead_tokens: int, schema: dict | None) -> int:
    """The budget for the settings file (A16 item 8's method): `schema` is the longest output schema the gate will see,
    since one number serves every request. A question counts as its own one-question object, a little over its share."""
    questions = [len(jev_call.dumps({key: question})) for key, question in gate_questions(settings).items()]
    state_bytes = len(jev_call.dumps(gate_state("", schema, "")))
    try:
        return jev_call.compute_budget_bytes(overhead_tokens=overhead_tokens, state_bytes=state_bytes,
                                             question_bytes=questions)  # fmt: skip
    except jev_call.CallError as error:
        raise GateCallError(str(error)) from error


def gate_call(request: Request, answer: str, settings: JevGateSettings, *, budget_bytes: int | None = None) -> GateCall:
    """The call body for an answer to a request. The budget is the settings' unless given; with neither, or one below 1,
    the call is refused, since an unbudgeted call could exceed Jev's limits. The schema stays whole."""
    budget = settings.budget_bytes if budget_bytes is None else budget_bytes
    if budget is None or budget < 1:
        raise GateCallError(f"no usable byte budget for the Jev gate ({budget!r})")
    fitted = fit_to_budget(request.prompt, answer, budget)
    state = gate_state(fitted.prompt, request.output_schema, fitted.answer)
    body = jev_call.dumps({"model": settings.model, "state": state, "questions": gate_questions(settings)})
    return GateCall(body, jev_call.body_sha256(body), fitted.answer_cut_bytes, fitted.prompt_cut_bytes)


def log_odds(p: float) -> float:
    return math.log(p / (1 - p))


def gate_score(answers: Mapping[str, float], settings: JevGateSettings) -> float:
    """The gate score of the three answers by statement ID: clip, invert, mean log-odds, back to a probability."""
    low, high = settings.clip
    total = 0.0
    for statement in settings.statements:
        p = min(max(answers[statement.id], low), high)
        total += log_odds(1 - p if statement.inverted else p)
    return 1 / (1 + math.exp(-total / len(settings.statements)))


def read_response(response: object, settings: JevGateSettings) -> GateResponse:
    """A response's gate score, or a refusal by A16 item 7's rule (a 200 that doesn't answer all three statements in the
    right form). A refusal has no score and becomes a rejection."""
    parsed = jev.parse_answers(response, gate_questions(settings))
    if parsed.refused:
        return GateResponse(None, True)
    return GateResponse(gate_score({k: float(v) for k, v in parsed.answers.items()}, settings), False)  # type: ignore[arg-type]
