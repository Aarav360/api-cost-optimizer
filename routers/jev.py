"""The Jev routers' client and scoring (PLAN_v4.md §8 3c and 3c+; Amendment A15, items 7, 9 and 12).

This module is frozen at the freeze commit (A15 item 13), apart from call building (routers/jev_call.py), which is
frozen at the settings commit. Standard library and numpy only.

**G+'s scoring (Stage 6 · 3).** Jev's 18 answers become 16 features, then three router scores through numpy heads
read from the weights file (`routers/jev_weights.json`, the format fixed in issue #30):

1. **Answers in:** by question ID, as the parser gives them: a Noul answer is a float, the request-type question's
   Choice answer a dict from each option to its float. A missing or mistyped or out-of-range answer is refused
   (`GPlusError`). A refused response (the refusal flag the parser sets) never reaches the features: its router
   score is 0 for every config, so it goes to the frontier at any threshold above 0 (A15 item 9).
2. **Features (A15 item 7), in the weights file's `features` order:**
   - every answer used as a feature is clipped to the settings' `g_plus.clip` ([0.005, 0.995]), then turned into
     log-odds, so a 1.00 lands at 5.3 and a 0.99 at 4.6;
   - each config's success feature is the mean of its three phrasings' log-odds, phrasing `c` read as 1 - p (the
     log-odds of the clipped answer, negated);
   - the four request-type answers are used as Jev returns them (not clipped, not log-odds);
   - the schema flag is 1 if the request has an output schema, else 0 (A11 item 5).
3. **Heads:** the features are standardized with the weights file's means and scales, then each config's router
   score is sigmoid(coef . z + intercept). G's router scores, by contrast, are Jev's answers unchanged.

The feature names and order come from the settings (`feature_names`); a weights file whose names or order differ is
refused whole.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from routers.jev_call import Settings
from routers.ladder import SCORED
from routers.request import Request

WEIGHTS_FILE = Path(__file__).with_name("jev_weights.json")
SCHEMA_FLAG = "schema_flag"
FEATURE_COUNT = 16

# What the parser gives for one answer: a Noul answer is a float, a Choice answer a dict from option to float.
Answers = Mapping[str, float | Mapping[str, float]]


class GPlusError(ValueError):
    """A weights file or a set of answers G+'s scoring refuses whole, before any router score is given."""


@dataclass(frozen=True)
class Head:
    """One config's logistic regression: its C (kept for the record), coefficients and intercept."""

    c: float
    coef: np.ndarray
    intercept: float


@dataclass(frozen=True)
class GPlusWeights:
    features: tuple[str, ...]  # the weights file's order; the standardizer and every head follow it
    mean: np.ndarray
    scale: np.ndarray
    heads: dict[str, Head]  # config -> head, for exactly the configs below the frontier


def feature_names(settings: Settings) -> tuple[str, ...]:
    """The 16 feature names in order (A15 item 7): the three success features, the eight difficulty statements, the
    four request-type options in the settings' order, then the schema flag."""
    plus = settings.g_plus
    return (
        *(f"success_{config}" for config in SCORED),
        *(d.id for d in plus.difficulty_statements),
        *plus.request_type.options,
        SCHEMA_FLAG,
    )


def is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def finite_vector(value: object, what: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != FEATURE_COUNT or not all(is_number(v) for v in value):
        raise GPlusError(f"{what} must be a list of {FEATURE_COUNT} finite numbers")
    return np.array(value, dtype=float)


def check_feature_order(weights: GPlusWeights, settings: Settings) -> None:
    """Refuse a weights file whose feature names or order differ from the settings' (A15 item 7)."""
    expected = feature_names(settings)
    if weights.features != expected:
        raise GPlusError(f"the weights file's features {weights.features} are not the settings' {expected}")


def parse_weights(data: dict, settings: Settings) -> GPlusWeights:
    """G+'s weights file (the format fixed in issue #30), refused whole if any part is missing or malformed."""
    try:
        features = tuple(data["features"])
        mean = finite_vector(data["standardizer"]["mean"], "the standardizer's mean")
        scale = finite_vector(data["standardizer"]["scale"], "the standardizer's scale")
        raw_heads = data["heads"]
        heads = {
            config: Head(raw["c"], finite_vector(raw["coef"], f"{config}'s coefficients"), raw["intercept"])
            for config, raw in raw_heads.items()
        }
    except (KeyError, TypeError, AttributeError) as error:
        raise GPlusError(f"the weights file lacks or misshapes {error!r}") from error
    if not (scale > 0).all():
        raise GPlusError("a standardizer scale is not positive")
    if set(heads) != set(SCORED):
        raise GPlusError(f"the weights file needs a head for exactly {SCORED}, not {sorted(heads)}")
    for config, head in heads.items():
        if not is_number(head.c) or head.c <= 0 or not is_number(head.intercept):
            raise GPlusError(f"{config}'s head needs a positive C and a finite intercept")
    weights = GPlusWeights(features, mean, scale, {config: heads[config] for config in SCORED})
    check_feature_order(weights, settings)
    return weights


def read_weights(settings: Settings, path: Path = WEIGHTS_FILE) -> GPlusWeights:
    return parse_weights(json.loads(path.read_text(encoding="utf-8")), settings)


def noul_answer(answers: Answers, key: str) -> float:
    value = answers.get(key)
    if not is_number(value) or not 0 <= value <= 1:  # type: ignore[operator]
        raise GPlusError(f"the answer to {key!r} is not a number in [0, 1]: {value!r}")
    return float(value)  # type: ignore[arg-type]


def choice_answer(answers: Answers, key: str, options: tuple[str, ...]) -> dict[str, float]:
    value = answers.get(key)
    if not isinstance(value, Mapping) or set(value) != set(options):
        raise GPlusError(f"the answer to {key!r} is not a Choice answer over exactly {options}: {value!r}")
    return {option: noul_answer(value, option) for option in options}


def log_odds(p: float, clip: tuple[float, float]) -> float:
    """The log-odds of an answer clipped to `clip` (A15 item 7)."""
    p = min(max(p, clip[0]), clip[1])
    return math.log(p / (1 - p))


def g_plus_features(answers: Answers, request: Request, settings: Settings, weights: GPlusWeights) -> np.ndarray:
    """The 16 features in the weights file's order, from Jev's answers by question ID and the request's schema."""
    check_feature_order(weights, settings)
    plus = settings.g_plus
    named: dict[str, float] = {}
    for config in SCORED:
        # Phrasing `c` ("will make at least one mistake") is read as 1 - p: the clipped answer's log-odds, negated.
        parts = [
            (-1 if s.phrasing == "c" else 1) * log_odds(noul_answer(answers, s.id), plus.clip)
            for s in plus.success_statements
            if s.config == config
        ]
        named[f"success_{config}"] = sum(parts) / len(parts)
    for d in plus.difficulty_statements:
        named[d.id] = log_odds(noul_answer(answers, d.id), plus.clip)
    named |= choice_answer(answers, plus.request_type.id, plus.request_type.options)  # as Jev returns them
    named[SCHEMA_FLAG] = 0.0 if request.output_schema is None else 1.0
    return np.array([named[name] for name in weights.features], dtype=float)


def sigmoid(t: float) -> float:
    """1 / (1 + e^-t), without overflow at either end."""
    if t >= 0:
        return 1.0 / (1.0 + math.exp(-t))
    e = math.exp(t)
    return e / (1.0 + e)


def g_plus_heads(features: np.ndarray, weights: GPlusWeights) -> dict[str, float]:
    """Each config's router score: the features standardized with the weights file, then sigmoid(coef . z + intercept)."""
    z = (features - weights.mean) / weights.scale
    return {config: sigmoid(float(head.coef @ z + head.intercept)) for config, head in weights.heads.items()}


def g_plus_scores(
    answers: Answers, refused: bool, request: Request, settings: Settings, weights: GPlusWeights
) -> dict[str, float]:
    """G+'s router score per config below the frontier. A refused response gives 0 for every config (A15 item 9)."""
    if refused:
        return dict.fromkeys(SCORED, 0.0)
    return g_plus_heads(g_plus_features(answers, request, settings, weights), weights)
