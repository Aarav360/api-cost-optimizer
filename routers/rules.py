"""The rules router, E (PLAN_v4.md §8 3a; Amendment A9, items 4-6 and 13).

Hand-written rules put each request in a bucket: an inferred type, then one difficulty cue for that type. A bucket's
router score for `haiku`, `sonnet` and `opus_low` is that config's run-0 success rate on the bucket's train items,
fitted by `fit` and kept in the rates file beside this module, so `score` never reads train again.

The rules look for cues real traffic would also show, never a template line or a word that belongs to one dataset
(item 5). `analysis/check_rules_router.py` searches this file's source for the benchmark's own labels. Everything
this module imports from `routers/` is frozen with it (item 13), so its imports stay few: the ladder and the request.
"""

import json
import re
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from routers.ladder import SCORED
from routers.request import Request

# Fixed here, before anything is fitted (item 4): a bucket of fewer train items folds into its type's bucket.
MIN_BUCKET_SIZE = 30
RATES_FILE = Path(__file__).with_name("rules_rates.json")

MULTIPLE_CHOICE, CLASSIFICATION, CODE, OPEN_ENDED = "multiple_choice", "classification", "code", "open_ended"
TYPES = (MULTIPLE_CHOICE, CLASSIFICATION, CODE, OPEN_ENDED)
# Each type's two buckets, the side of its cut point first and the other side second.
BUCKETS = {
    MULTIPLE_CHOICE: ("multiple_choice/plain", "multiple_choice/calc"),
    CLASSIFICATION: ("classification/short", "classification/long"),
    CODE: ("code/short", "code/long"),
    OPEN_ENDED: ("open_ended/few", "open_ended/many"),
}
FLAG_CUT = 0.5  # multiple choice's cue is a yes/no flag (0 or 1), so its cut point is fixed, not fitted

# Patterns are compiled at import, so `score`'s latency is the matching alone (item 6).
OPTION_LINE = re.compile(r"^[ \t]*\(?([A-Z])[.):][ \t]+\S", re.MULTILINE)
CLASSIFY_VERB = re.compile(r"\b(?:classif(?:y|ication)|categori[sz]e)\b", re.IGNORECASE)
# A line of one to four plain words (a label, perhaps bulleted); three in a row make a list of labels.
LABEL_LINE = re.compile(r"^[ \t]*(?:[-*•][ \t]+)?[\w'/&?-]+(?: [\w'/&?-]+){0,3}[ \t]*$")
INLINE_LIST = re.compile(r"(?:\bone of\b|:)[ \t]*([^.\n]+)", re.IGNORECASE)
# A request to write something named in the same breath: "write a (recursive Python) function", at most two words
# between the article and the noun, so "write a review of a chess program" isn't one.
CODE_REQUEST = re.compile(
    r"\b(?:write|implement|create|complete|define|develop)\b(?:\s+(?:me|us))?\s+(?:an?\s+|the\s+|some\s+)?"
    r"(?:[\w+#-]+\s+){0,2}(?:function|program|script|method|class)\b"
    r"|\b(?:return|provide|give|put|wrap|include)\b[^.\n]{0,60}(?:code block|```)",
    re.IGNORECASE,
)
LATEX = re.compile(
    r"\\(?:frac|sqrt|sum|int|cdot|times|left|right|alpha|beta|theta|pi|infty|begin|text|mathrm|lim|partial|vec|hat|"
    r"bar|Delta|omega|mu|lambda|sigma|rho|phi|epsilon)\b|\\\(|\\\[|\$[^$\n]*[\\^_=][^$\n]*\$"
)
# Digits beside an operator ("3 + 4", "x = 5"), or beside a unit ("5 kg", "20 m/s"). The unit must end the word, so
# "5 more" isn't "5 m". Horizontal space only: a number at a line's end isn't beside the next line's first word.
DIGITS_WITH_OPERATOR = re.compile(r"\d[ \t]*[+*/^=×÷<>][ \t]*[\d(]|\d[ \t]+[-−–][ \t]+\d"
                                  r"|[A-Za-z][ \t]*=[ \t]*-?\d")  # fmt: skip
DIGITS_WITH_UNIT = re.compile(
    r"\d[ \t]?(?:°[CF]?|ft|lbs?|mph|atm|rpm|mol|Hz|Pa|eV|kg|km|cm|mm|mg|mL|ms|min|kJ|kW|kV|mA|nm|[gmsNJWVAKLC])"
    r"(?![A-Za-z])"
)
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
CONSTRAINT_MARKER = re.compile(
    r"\b(?:must|at least|at most|do not|don't|should not|exactly|no more than|never|make sure|"
    r"end your response with)\b",
    re.IGNORECASE,
)


class FitError(ValueError):
    """Requests and train successes that `fit` refuses. Nothing is fitted."""


def answer_choices(schema: dict | None) -> list[str] | None:
    """The first string-enum property of an output schema: the set of values the answer must be one of."""
    for prop in (schema or {}).get("properties", {}).values():
        choices = prop.get("enum")
        if isinstance(choices, list) and choices and all(isinstance(c, str) for c in choices):
            return choices
    return None


def has_option_list(prompt: str) -> bool:
    """Lettered options: lines starting `A.`, `B)`, `(C)` and so on, with an A line followed by a B line."""
    letters = [m[1] for m in OPTION_LINE.finditer(prompt)]
    return ("A", "B") in zip(letters, letters[1:])  # a roman-numeral list ahead of the options doesn't hide them


def has_label_list(prompt: str) -> bool:
    """Three or more consecutive short lines, or an inline list of three or more after `:` or `one of`."""
    run = 0
    for line in prompt.split("\n"):
        run = run + 1 if LABEL_LINE.match(line) else 0
        if run >= 3:
            return True
    return any(len(re.split(r",|\bor\b|\band\b", m[1])) >= 3 for m in INLINE_LIST.finditer(prompt))


def infer_type(request: Request) -> str:
    """The request's type, by the first check that holds (item 5); anything else is open-ended."""
    choices = answer_choices(request.output_schema)
    if choices is not None:
        return MULTIPLE_CHOICE if all(len(c) == 1 and c.isalpha() for c in choices) else CLASSIFICATION
    prompt = request.prompt
    if has_option_list(prompt):  # before code: some multiple-choice questions contain code
        return MULTIPLE_CHOICE
    if CLASSIFY_VERB.search(prompt) and has_label_list(prompt):
        return CLASSIFICATION
    if CODE_REQUEST.search(prompt):
        return CODE
    return OPEN_ENDED


def needs_calculation(prompt: str) -> bool:
    """Whether the question's stem (the text before the options) has LaTeX, digits with an operator or digits with a unit."""
    option = OPTION_LINE.search(prompt)
    stem = prompt[: option.start()] if option else prompt
    return bool(LATEX.search(stem) or DIGITS_WITH_OPERATOR.search(stem) or DIGITS_WITH_UNIT.search(stem))


def constraint_sentences(prompt: str) -> int:
    """The number of sentences, or lines, holding a general constraint marker."""
    return sum(bool(CONSTRAINT_MARKER.search(sentence)) for sentence in SENTENCE_BREAK.split(prompt))


def cue_value(kind: str, prompt: str) -> float:
    """The type's difficulty cue as a number: a yes/no flag, the prompt's length, or the constraint count."""
    if kind == MULTIPLE_CHOICE:
        return float(needs_calculation(prompt))
    return float(constraint_sentences(prompt) if kind == OPEN_ENDED else len(prompt))


def bucket_key(request: Request, cut_points: Mapping[str, float]) -> str:
    """The request's bucket: its type's lower bucket at or below the cut point, the other above it."""
    kind = infer_type(request)
    cut = FLAG_CUT if kind == MULTIPLE_CHOICE else cut_points[kind]
    return BUCKETS[kind][cue_value(kind, request.prompt) > cut]


def success_rates(successes: Mapping[str, Mapping[str, float]], item_ids: list[str]) -> dict[str, float]:
    return {config: sum(successes[i][config] for i in item_ids) / len(item_ids) for config in SCORED}


def fit(requests: Mapping[str, Request], successes: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """The rates file's contents, from train requests and their run-0 successes by item ID, both plain data.

    Pure: no file is read or written. The cut points are the types' train medians, which involve no outcomes. A
    bucket under MIN_BUCKET_SIZE items takes its type's rates; a type with no items takes the rates of all items.
    Refused whole (FitError) if the two don't name the same items or a success is missing or outside 0 to 1.
    """
    if not requests:
        raise FitError("no requests to fit")
    if set(requests) != set(successes):
        raise FitError(f"{len(set(requests) ^ set(successes))} items are in only one of requests and successes")
    for item_id, cells in successes.items():
        for config in SCORED:
            if config not in cells or not 0 <= cells[config] <= 1:
                raise FitError(f"{item_id} has no success from 0 to 1 for {config}: {cells.get(config)!r}")
    item_ids = sorted(requests)
    kinds = {i: infer_type(requests[i]) for i in item_ids}
    values = {i: cue_value(kinds[i], requests[i].prompt) for i in item_ids}
    cut_points = {}
    for kind in (CLASSIFICATION, CODE, OPEN_ENDED):
        cues = [values[i] for i in item_ids if kinds[i] == kind]
        cut_points[kind] = statistics.median(cues) if cues else 0.0
    overall = success_rates(successes, item_ids)
    types = {}
    for kind in TYPES:
        ids = [i for i in item_ids if kinds[i] == kind]
        types[kind] = {"count": len(ids), "rates": success_rates(successes, ids) if ids else overall}
    buckets = {}
    for kind in TYPES:
        cut = FLAG_CUT if kind == MULTIPLE_CHOICE else cut_points[kind]
        for side, key in enumerate(BUCKETS[kind]):
            ids = [i for i in item_ids if kinds[i] == kind and (values[i] > cut) == bool(side)]
            folded = len(ids) < MIN_BUCKET_SIZE
            buckets[key] = {"count": len(ids), "folded": folded,
                            "rates": types[kind]["rates"] if folded else success_rates(successes, ids)}  # fmt: skip
    return {"min_bucket_size": MIN_BUCKET_SIZE, "cut_points": cut_points, "types": types, "buckets": buckets}


def load_rates(path: Path = RATES_FILE) -> dict[str, Any]:
    return json.loads(path.read_text())


@dataclass(frozen=True)
class RulesRouter:
    rates: dict[str, Any]  # a rates file's contents

    def score(self, request: Request) -> dict[str, float]:
        """The request's bucket's train success rates for `haiku`, `sonnet` and `opus_low`. Overhead is an
        explicit $0 (item 6): this runs locally and the router-score table records it."""
        return dict(self.rates["buckets"][bucket_key(request, self.rates["cut_points"])]["rates"])


@cache
def default_router() -> RulesRouter:
    return RulesRouter(load_rates())


def score(request: Request) -> dict[str, float]:
    """Router scores from the committed rates file. Stage 8 and S4 call this."""
    return default_router().score(request)
