"""Call building for the Jev routers (PLAN_v4.md §8 3c; Amendment A15, items 4, 8 and 12).

A request becomes G's call body, as bytes. Pure functions only: nothing here sends a call or reads an answer, which
is routers/jev.py's job. This module is frozen at the settings commit (A15 item 13), apart from the client and parser
so that a client bug found later never touches a frozen file. G+'s body joins it in Stage 6 · 3 (A15 items 5, 6
and 8): 16 past requests with their outcomes, then the request, and 18 questions.

It reads the settings file (`routers/jev_settings.json`, the format fixed in issue #30) for what it builds from:
`jev.model`, `display_names`, `g.success_statements`, `g_plus` and `budget_bytes`. Other keys are the client's and
are ignored here (`g_plus.rates` and `noise_sample` among them: the rates are already inside the statements' text).
`g_plus.clip` is read because the client's scoring needs it and a settings file is refused whole. Standard library
only.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from routers.ladder import SCORED
from routers.request import Request

SETTINGS_FILE = Path(__file__).with_name("jev_settings.json")
# Jev's limits in tokens (docs.typesafe.ai: 64k per request, 32k per question; A15 item 3 saw a state of about 33K
# refused). Read as 32,000 and 64,000, the tighter reading, so the budget is safe whichever the service means.
STATE_PLUS_ONE_QUESTION_LIMIT = 32_000
STATE_PLUS_ALL_QUESTIONS_LIMIT = 64_000


class CallError(ValueError):
    """A settings file or a budget this module refuses whole, before any body is built."""


@dataclass(frozen=True)
class Statement:
    """One success statement, sent to Jev as a Noul question under `id` (A15 item 4)."""

    id: str
    config: str
    text: str


@dataclass(frozen=True)
class PlusStatement:
    """One of G+'s nine success statements: a phrasing (`a`, `b` or `c`; `c` is the negated one) of a config's."""

    id: str
    config: str
    phrasing: str
    text: str


@dataclass(frozen=True)
class DifficultyStatement:
    """One of G+'s eight difficulty statements, sent as a Noul question under `difficulty_<n>` (A15 item 6)."""

    id: str
    text: str


@dataclass(frozen=True)
class RequestType:
    """G+'s request-type question: a Choice question whose options are sent with null criteria (A15 item 6)."""

    id: str
    text: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class PastRequest:
    """A past train request shown whole, with each config's run-0 outcome (True: correct), in state order."""

    item_id: str
    prompt: str
    schema: dict | None
    outcomes: dict[str, bool]


@dataclass(frozen=True)
class GPlusSettings:
    success_statements: tuple[PlusStatement, ...]
    difficulty_statements: tuple[DifficultyStatement, ...]
    request_type: RequestType
    past_requests: tuple[PastRequest, ...]
    clip: tuple[float, float]  # every answer used as a feature is clipped to this before log-odds (A15 item 7)


@dataclass(frozen=True)
class Settings:
    model: str
    display_names: dict[str, str]
    g_statements: tuple[Statement, ...]
    budget_bytes: dict[str, int | None]  # router name -> budget; null until the smoke test fills it
    g_plus: GPlusSettings


@dataclass(frozen=True)
class Call:
    """A call body, ready to send, with its sha256 for the stored record."""

    body: bytes
    sha256: str
    cut_bytes: int  # serialized prompt bytes the cut dropped; 0 if the prompt fit


def dumps(data: object) -> bytes:
    """A body's serialization (A15 item 8): compact separators, non-ASCII kept, UTF-8, keys in insertion order."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def body_sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def is_text(value: object) -> bool:
    return isinstance(value, str) and value != ""


PHRASINGS = ("a", "b", "c")  # `c` is the negated statement (A15 item 6)
DIFFICULTY_COUNT = 8
PAST_REQUEST_COUNT = 16  # A15 item 5: within each of 4 tasks, 2 that haiku failed and 2 it passed
REQUEST_TYPE_ID = "request_type"
REQUEST_TYPE_OPTIONS = 4


def parse_g_plus(data: dict) -> GPlusSettings:
    """G+'s part of a settings file (A15 items 5 to 7), refusing it whole if any part is missing or malformed."""
    try:
        statements = tuple(
            PlusStatement(s["id"], s["config"], s["phrasing"], s["text"]) for s in data["success_statements"]
        )
        difficulty = tuple(DifficultyStatement(s["id"], s["text"]) for s in data["difficulty_statements"])
        kind = RequestType(
            data["request_type"]["id"], data["request_type"]["text"], tuple(data["request_type"]["options"])
        )
        past = tuple(
            PastRequest(p["item_id"], p["prompt"], p["output_schema"], dict(p["outcomes"]))
            for p in data["past_requests"]
        )
        clip = tuple(data["clip"])
    except (KeyError, TypeError, ValueError) as error:
        raise CallError(f"g_plus settings lack or misshape {error!r}") from error
    # Nine statements: each scored config's three phrasings, in the ladder's order, under IDs success_<config>_<phrasing>.
    expected = [(config, phrasing) for config in SCORED for phrasing in PHRASINGS]
    if [(s.config, s.phrasing) for s in statements] != expected or any(
        s.id != f"success_{s.config}_{s.phrasing}" for s in statements
    ):
        raise CallError("G+'s success statements must be three phrasings (a, b, c) per config, in the ladder's order")
    if not all(is_text(s.text) for s in statements):
        raise CallError("a G+ success statement is empty or not a string")
    if [d.id for d in difficulty] != [f"difficulty_{n}" for n in range(1, DIFFICULTY_COUNT + 1)]:
        raise CallError(f"G+'s difficulty statements must be difficulty_1 to difficulty_{DIFFICULTY_COUNT}, in order")
    if not all(is_text(d.text) for d in difficulty):
        raise CallError("a difficulty statement is empty or not a string")
    if kind.id != REQUEST_TYPE_ID or not is_text(kind.text):
        raise CallError(f"the request-type question needs the ID {REQUEST_TYPE_ID} and a non-empty text")
    if len(kind.options) != REQUEST_TYPE_OPTIONS or len(set(kind.options)) != REQUEST_TYPE_OPTIONS:
        raise CallError(f"the request-type question needs {REQUEST_TYPE_OPTIONS} distinct options")
    if not all(is_text(option) for option in kind.options):
        raise CallError("a request-type option is empty or not a string")
    if len(past) != PAST_REQUEST_COUNT:
        raise CallError(f"G+ shows {PAST_REQUEST_COUNT} past requests, not {len(past)}")
    for p in past:
        if not is_text(p.item_id) or not is_text(p.prompt) or not (p.schema is None or isinstance(p.schema, dict)):
            raise CallError(
                f"past request {p.item_id!r} needs an item ID, a prompt and a schema that is an object or null"
            )
        if set(p.outcomes) != set(SCORED) or not all(type(v) is bool for v in p.outcomes.values()):
            raise CallError(f"past request {p.item_id!r} needs a true or false outcome for exactly {SCORED}")
    if len(clip) != 2 or not all(type(c) in (int, float) for c in clip) or not 0 < clip[0] < clip[1] < 1:
        raise CallError("clip must be two numbers with 0 < low < high < 1")
    return GPlusSettings(statements, difficulty, kind, past, (float(clip[0]), float(clip[1])))


def parse_settings(data: dict) -> Settings:
    """The call-building part of a settings file, refusing it whole if any part is missing or malformed."""
    try:
        model = data["jev"]["model"]
        names = data["display_names"]
        statements = tuple(Statement(s["id"], s["config"], s["text"]) for s in data["g"]["success_statements"])
        budgets = data["budget_bytes"]
        plus_data = data["g_plus"]
    except (KeyError, TypeError) as error:
        raise CallError(f"settings lack or misshape {error!r}") from error
    if not is_text(model):
        raise CallError("the model is not a non-empty string")
    if not isinstance(names, dict) or set(names) != set(SCORED) or not all(is_text(n) for n in names.values()):
        raise CallError(f"display names must be non-empty strings for exactly {SCORED}")
    # G's three statements: one per scored config, in the ladder's order, each under the ID success_<config> (A15 item 4).
    if tuple(s.config for s in statements) != SCORED or any(s.id != f"success_{s.config}" for s in statements):
        raise CallError(f"G's success statements must be {SCORED} in order, each with ID success_<config>")
    if not all(is_text(s.text) for s in statements):
        raise CallError("a success statement is empty or not a string")
    if not isinstance(budgets, dict) or set(budgets) != {"jev", "jev_bolstered"}:
        raise CallError("budget_bytes must hold exactly jev and jev_bolstered")
    if not all(b is None or (type(b) is int and b >= 1) for b in budgets.values()):
        raise CallError("a budget is null or a whole number of bytes of at least 1")
    return Settings(model, names, statements, budgets, parse_g_plus(plus_data))


def read_settings(path: Path = SETTINGS_FILE) -> Settings:
    return parse_settings(json.loads(path.read_text(encoding="utf-8")))


def serialized_size(char: str) -> int:
    """A character's bytes inside a serialized JSON string: UTF-8, or the escape's length for a quote, a backslash or
    a control character."""
    return len(json.dumps(char, ensure_ascii=False).encode("utf-8")) - 2  # less the two quotes


def cut_prompt(prompt: str, budget: int) -> tuple[str, int]:
    """The prompt within a byte budget, and the serialized bytes the cut dropped (A15 item 8).

    The budget counts the prompt's serialized bytes at one byte per token. A prompt over it keeps its head and its
    tail, half the budget each, whole characters only, so the kept parts can come to a little under the budget."""
    sizes = [serialized_size(char) for char in prompt]
    total = sum(sizes)
    if total <= budget:
        return prompt, 0
    half = budget // 2
    head = used = 0
    while head < len(prompt) and used + sizes[head] <= half:
        used += sizes[head]
        head += 1
    tail = len(prompt)
    used = 0
    while tail > head and used + sizes[tail - 1] <= half:
        tail -= 1
        used += sizes[tail]
    return prompt[:head] + prompt[tail:], total - sum(sizes[:head]) - sum(sizes[tail:])


def compute_budget_bytes(*, overhead_tokens: int, state_bytes: int, question_bytes: Sequence[int]) -> int:
    """A router's prompt budget in bytes, from the tighter of Jev's two limits (A15 item 8).

    Each limit less the measured fixed overhead, `state_bytes` (the state's serialized size with an empty prompt,
    so its past requests and schema included) and the questions: the longest one for the 32K limit, all of them for
    the 64K limit. Bytes stand for tokens one for one. Parts that leave no room are refused."""
    if not question_bytes:
        raise CallError("a call has at least one question")
    fixed = overhead_tokens + state_bytes
    one = STATE_PLUS_ONE_QUESTION_LIMIT - fixed - max(question_bytes)
    every = STATE_PLUS_ALL_QUESTIONS_LIMIT - fixed - sum(question_bytes)
    budget = min(one, every)
    if budget < 1:
        raise CallError(f"the parts leave no room for a prompt (budget {budget})")
    return budget


def g_questions(settings: Settings) -> dict[str, dict]:
    """G's three success statements as Noul questions, under their settings IDs, in the settings' order."""
    return {s.id: {"type": "noul", "instructions": s.text} for s in settings.g_statements}


def g_state(prompt: str, schema: dict | None) -> dict:
    return {"request": {"prompt": prompt, "output_schema": schema}}


def g_budget_bytes(settings: Settings, *, overhead_tokens: int, schema: dict | None) -> int:
    """G's budget for the settings file: `schema` is the longest output schema G will see, since one number serves
    every request. A question counts as its own one-question `questions` object, a little over its share of the body."""
    question_bytes = [len(dumps({key: question})) for key, question in g_questions(settings).items()]
    state_bytes = len(dumps(g_state("", schema)))
    return compute_budget_bytes(overhead_tokens=overhead_tokens, state_bytes=state_bytes, question_bytes=question_bytes)


def g_call(request: Request, settings: Settings, *, budget_bytes: int | None = None) -> Call:
    """G's call body for a request: one Noul question per success statement, the model from the settings.

    The budget is the settings' for G unless given (the smoke test, before Stage 6 · 8 fills it in); with neither,
    or with one below 1, the call is refused, since an unbudgeted call could exceed Jev's limits. Only the prompt is
    cut: its schema stays whole."""
    budget = settings.budget_bytes.get("jev") if budget_bytes is None else budget_bytes
    if budget is None or budget < 1:
        raise CallError(f"no usable byte budget for G ({budget!r})")
    prompt, cut = cut_prompt(request.prompt, budget)
    state = g_state(prompt, request.output_schema)
    body = dumps({"model": settings.model, "state": state, "questions": g_questions(settings)})
    return Call(body, body_sha256(body), cut)


def g_plus_questions(settings: Settings) -> dict[str, dict]:
    """G+'s 18 questions in the body's order (A15 item 6): the nine success statements and the eight difficulty
    statements as Noul questions, then the request-type question as a Choice question with each option's criterion
    null (Jev refuses a Choice question with no `criteria`, and reads a null description by the option's name)."""
    plus = settings.g_plus
    questions: dict[str, dict] = {s.id: {"type": "noul", "instructions": s.text} for s in plus.success_statements}
    questions |= {d.id: {"type": "noul", "instructions": d.text} for d in plus.difficulty_statements}
    kind = plus.request_type
    questions[kind.id] = {"type": "choice", "instructions": kind.text, "criteria": dict.fromkeys(kind.options)}
    return questions


def g_plus_state(prompt: str, schema: dict | None, settings: Settings) -> dict:
    """G+'s Jev state (A15 items 5 and 8): the past requests with their outcomes, then the request. An outcome is keyed
    by its config's display name and reads "correct" or "incorrect"; key order is the code's, never sorted."""
    names = settings.display_names
    past = [
        {
            "past_request": {"prompt": p.prompt, "output_schema": p.schema},
            "outcomes": {names[config]: "correct" if p.outcomes[config] else "incorrect" for config in SCORED},
        }
        for p in settings.g_plus.past_requests
    ]
    return {"past_requests_with_outcomes": past, "request": {"prompt": prompt, "output_schema": schema}}


def g_plus_budget_bytes(settings: Settings, *, overhead_tokens: int, schema: dict | None) -> int:
    """G+'s budget for the settings file, as G's: `schema` is the longest output schema G+ will see. The state counts
    with an empty prompt, so its 16 past requests are inside it and never cut."""
    question_bytes = [len(dumps({key: question})) for key, question in g_plus_questions(settings).items()]
    state_bytes = len(dumps(g_plus_state("", schema, settings)))
    return compute_budget_bytes(overhead_tokens=overhead_tokens, state_bytes=state_bytes, question_bytes=question_bytes)


def g_plus_call(request: Request, settings: Settings, *, budget_bytes: int | None = None) -> Call:
    """G+'s call body for a request: the past requests with outcomes, the request, then 18 questions.

    Budget and cut as `g_call`, from the settings' `jev_bolstered` entry unless given. Only the request's prompt is
    cut: its schema, the past requests and the questions stay whole."""
    budget = settings.budget_bytes.get("jev_bolstered") if budget_bytes is None else budget_bytes
    if budget is None or budget < 1:
        raise CallError(f"no usable byte budget for G+ ({budget!r})")
    prompt, cut = cut_prompt(request.prompt, budget)
    state = g_plus_state(prompt, request.output_schema, settings)
    body = dumps({"model": settings.model, "state": state, "questions": g_plus_questions(settings)})
    return Call(body, body_sha256(body), cut)
