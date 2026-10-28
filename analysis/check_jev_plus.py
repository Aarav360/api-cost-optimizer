"""Check G+'s scoring: Jev's 18 answers to 16 features to three router scores (PLAN_v4.md §8 3c+; Amendment A15,
items 7 and 9; issue #33).

Run from the project root:  uv run python -m analysis.check_jev_plus

Never a test item and never a Jev call: the answers, settings and weights are synthetic and in #30's formats. The
call-building half of #33 is checked in analysis/check_jev_call.py.

- the features, hand-worked: a 1.00 clips to 0.995 and becomes 5.3 (0.00 to -5.3), the success feature is the mean of
  the three phrasings with `c` read as 1 - p, the request-type answers pass through as Jev returns them, the schema
  flag is 0 or 1, and the features follow the weights file's order;
- the heads, hand-worked: standardized with the weights file's means and scales, then sigmoid(coef . z + intercept);
- a refusal gives 0 for every config, whatever the answers hold;
- refusals, each fed its bad case: a feature-order or name mismatch, a malformed weights file, a missing or mistyped
  or out-of-range answer;
- the imports: the module's closure uses the standard library and numpy only.
"""

import ast
import copy
import sys
from collections.abc import Callable

import numpy as np

from analysis.check_jev_call import imported_packages, synthetic_data
from analysis.freeze import closure
from harness.client import PROJECT_ROOT
from routers import jev, jev_call
from routers.request import Request

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object]) -> bool:
    try:
        call()
    except jev.GPlusError:
        return True
    return False


def close(got: object, want: object, tolerance: float = 1e-6) -> bool:
    return bool(np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float), rtol=0, atol=tolerance))


FEATURES = [
    "success_haiku", "success_sonnet", "success_opus_low",
    *[f"difficulty_{n}" for n in range(1, 9)],
    "opt one", "opt two", "opt three", "opt four",
    "schema_flag",
]  # fmt: skip


def synthetic_weights() -> dict:
    """A weights file in #30's format: sparse coefficients, so each score is worked by hand below."""
    mean, scale = [0.0] * 16, [1.0] * 16
    mean[0], scale[0] = 1.0, 2.0  # success_haiku
    mean[15], scale[15] = 0.5, 0.5  # schema_flag
    haiku, sonnet, opus_low = [0.0] * 16, [0.0] * 16, [0.0] * 16
    haiku[0], haiku[6], haiku[11], haiku[15] = 1.0, 0.1, 2.0, 0.5  # success_haiku, difficulty_4, "opt one", flag
    sonnet[1], sonnet[4] = 2.0, -1.0  # success_sonnet, difficulty_2
    opus_low[2] = 0.2  # success_opus_low
    return {
        "features": FEATURES,
        "standardizer": {"mean": mean, "scale": scale},
        "heads": {
            "haiku": {"c": 0.1, "coef": haiku, "intercept": -1.0},
            "sonnet": {"c": 1.0, "coef": sonnet, "intercept": 0.5},
            "opus_low": {"c": 0.01, "coef": opus_low, "intercept": 0.0},
        },
    }


def synthetic_answers() -> dict:
    """What a parser gives: a Noul answer is a float, the Choice answer a dict from each option to its answer."""
    noul = {
        "success_haiku_a": 1.00, "success_haiku_b": 0.50, "success_haiku_c": 0.20,
        "success_sonnet_a": 0.80, "success_sonnet_b": 0.80, "success_sonnet_c": 0.80,
        "success_opus_low_a": 0.00, "success_opus_low_b": 0.00, "success_opus_low_c": 1.00,
        "difficulty_1": 0.50, "difficulty_2": 0.80, "difficulty_3": 0.20, "difficulty_4": 1.00,
        "difficulty_5": 0.00, "difficulty_6": 0.50, "difficulty_7": 0.50, "difficulty_8": 0.50,
    }  # fmt: skip
    return {**noul, "request_type": {"opt one": 0.7, "opt two": 0.1, "opt three": 0.2, "opt four": 0.0}}


def main() -> int:
    settings = jev_call.parse_settings(synthetic_data())
    weights = jev.parse_weights(synthetic_weights(), settings)
    request = Request("Anything", {"type": "object"})
    answers = synthetic_answers()

    print("-- the features --")
    # By hand, with ln 199 = 5.293305 (the log-odds of 0.995) and ln 4 = 1.386294 (of 0.80):
    #   haiku:    a = 1.00 clips to 0.995 -> 5.293305; b = 0.50 -> 0; c = 0.20 read as 1 - p = 0.80 -> 1.386294;
    #             the mean is 6.679599 / 3 = 2.226533
    #   sonnet:   a, b = 0.80 -> 1.386294 each; c = 0.80 read as 0.20 -> -1.386294; the mean is 1.386294 / 3 = 0.462098
    #   opus_low: a, b = 0.00 clip to 0.005 -> -5.293305; c = 1.00 read as 1 - p = 0.00 -> -5.293305; the mean is -5.293305
    expected = [
        2.226533, 0.462098, -5.293305,
        0.0, 1.386294, -1.386294, 5.293305, -5.293305, 0.0, 0.0, 0.0,
        0.7, 0.1, 0.2, 0.0,
        1.0,
    ]  # fmt: skip
    features = jev.g_plus_features(answers, request, settings, weights)
    check(
        features.shape == (16,) and close(features, expected), "16 features, hand-worked, in the weights file's order"
    )
    check(close(features[3 + 3], 5.293305) and close(features[3 + 4], -5.293305),
          "a 1.00 clips to 0.995 and becomes 5.3, a 0.00 clips to 0.005 and becomes -5.3")  # fmt: skip
    check(close(features[11:15], [0.7, 0.1, 0.2, 0.0]),
          "the four request-type answers are used as Jev returns them: not clipped, not log-odds, a 0.0 stays 0")  # fmt: skip
    near = dict(answers, difficulty_1=0.99, difficulty_2=0.01, difficulty_3=0.995, difficulty_4=0.005)
    check(close(jev.g_plus_features(near, request, settings, weights)[3:7], [4.59512, -4.59512, 5.293305, -5.293305]),
          "0.99 becomes 4.6 and 0.01 -4.6, one rounding step from the clipped 5.3, and 0.995 and 0.005 are not moved")  # fmt: skip
    check(jev.g_plus_features(answers, Request("No schema"), settings, weights)[15] == 0.0,
          "the schema flag is 0 for a request with no output schema, 1 for one with a schema")  # fmt: skip
    check(jev.g_plus_features(answers, Request("A schema", {}), settings, weights)[15] == 1.0,
          "an empty schema object still counts as a schema: the flag is 1")  # fmt: skip
    # A weights file may list the features in another order only if it says so; here it lists the settings' order, and
    # the names the function builds from the settings are the ones the weights file must carry.
    check(jev.feature_names(settings) == tuple(FEATURES),
          "the feature names, from the settings, are the three success features, eight difficulty IDs, the four options "
          "and the schema flag")  # fmt: skip

    print("\n-- the heads --")
    # By hand, standardizing with the weights file (mean 1, scale 2 for success_haiku; mean 0.5, scale 0.5 for the flag;
    # mean 0, scale 1 elsewhere), z = 0.613267 (success_haiku), 5.293305 (difficulty_4), 0.7 (opt one), 1.0 (flag):
    #   haiku:    t = 0.613267 + 0.1 * 5.293305 + 2.0 * 0.7 + 0.5 * 1 - 1 = 2.042597,  sigmoid = 0.885197
    #   sonnet:   t = 2.0 * 0.462098 - 1.386294 + 0.5 = 0.037902,                      sigmoid = 0.509474
    #   opus_low: t = 0.2 * -5.293305 = -1.058661,                                      sigmoid = 0.257565
    scores = jev.g_plus_scores(answers, False, request, settings, weights)
    check(list(scores) == ["haiku", "sonnet", "opus_low"]
          and close([scores["haiku"], scores["sonnet"], scores["opus_low"]], [0.885197, 0.509474, 0.257565]),
          "three router scores, hand-worked, one per config below the frontier, in the ladder's order")  # fmt: skip
    check(all(type(v) is float for v in scores.values()), "the router scores are plain floats")
    # Features at the standardizer's means standardize to 0, so each score is sigmoid(intercept): sigmoid(-1) = 0.268941,
    # sigmoid(0.5) = 0.622459 and sigmoid(0) = 0.5.
    at_mean = jev.g_plus_heads(np.array(synthetic_weights()["standardizer"]["mean"]), weights)
    check(close([at_mean["haiku"], at_mean["sonnet"], at_mean["opus_low"]], [0.268941, 0.622459, 0.5]),
          "features at the standardizer's means give sigmoid(intercept) for each head")  # fmt: skip
    # Every coefficient sum is positive (haiku 3.6, sonnet 1.0, opus_low 0.2), so +1e6 and -1e6 push every head to 1 and 0.
    check(jev.g_plus_heads(np.full(16, 1e6), weights) == {"haiku": 1.0, "sonnet": 1.0, "opus_low": 1.0}
          and jev.g_plus_heads(np.full(16, -1e6), weights) == {"haiku": 0.0, "sonnet": 0.0, "opus_low": 0.0},
          "extreme feature vectors saturate the sigmoid to 1 and 0 without overflow")  # fmt: skip

    print("\n-- refusals --")
    refused_scores = jev.g_plus_scores({}, True, request, settings, weights)
    check(refused_scores == {"haiku": 0.0, "sonnet": 0.0, "opus_low": 0.0},
          "a refused response gives router score 0 for every config, whatever answers come with it")  # fmt: skip
    check(jev.g_plus_scores(answers, True, request, settings, weights)
          == {"haiku": 0.0, "sonnet": 0.0, "opus_low": 0.0},
          "a refused response gives 0 even when its answers are well-formed")  # fmt: skip

    print("\n-- the weights file --")

    def mutated(change: Callable[[dict], object]) -> Callable[[], object]:
        data = copy.deepcopy(synthetic_weights())
        change(data)
        return lambda: jev.parse_weights(data, settings)

    swap = lambda d: d["features"].insert(0, d["features"].pop(1))  # noqa: E731
    weight_refusals = {
        "two features swapped (a feature-order mismatch)": swap,
        "a renamed feature": lambda d: d["features"].__setitem__(5, "difficulty_9"),
        "a request-type option named differently": lambda d: d["features"].__setitem__(12, "opt 2"),
        "the schema flag moved first": lambda d: d["features"].insert(0, d["features"].pop()),
        "fifteen features": lambda d: d["features"].pop(),
        "a mean of 15": lambda d: d["standardizer"]["mean"].pop(),
        "a scale of 17": lambda d: d["standardizer"]["scale"].append(1.0),
        "a scale of 0": lambda d: d["standardizer"]["scale"].__setitem__(3, 0.0),
        "a negative scale": lambda d: d["standardizer"]["scale"].__setitem__(3, -1.0),
        "a NaN mean": lambda d: d["standardizer"]["mean"].__setitem__(3, float("nan")),
        "no sonnet head": lambda d: d["heads"].pop("sonnet"),
        "a head for opus": lambda d: d["heads"].update(opus=d["heads"]["haiku"]),
        "a coefficient list of 15": lambda d: d["heads"]["haiku"]["coef"].pop(),
        "an infinite coefficient": lambda d: d["heads"]["haiku"]["coef"].__setitem__(2, float("inf")),
        "a string intercept": lambda d: d["heads"]["haiku"].update(intercept="0"),
        "a C of 0": lambda d: d["heads"]["sonnet"].update(c=0.0),
        "no standardizer": lambda d: d.pop("standardizer"),
        "no features": lambda d: d.pop("features"),
    }
    for what, change in weight_refusals.items():
        check(refused(mutated(change)), f"a weights file with {what} is refused")
    check(not refused(mutated(lambda d: None)), "the guard lets a well-formed weights file through")

    # The function that orders the features also refuses a weights object whose order differs from the settings'.
    reordered = jev.GPlusWeights(
        features=tuple(reversed(FEATURES)), mean=weights.mean, scale=weights.scale, heads=weights.heads
    )
    check(refused(lambda: jev.g_plus_features(answers, request, settings, reordered)),
          "features are refused against a weights object whose order differs from the settings'")  # fmt: skip

    print("\n-- the answers --")

    def answered(change: Callable[[dict], object]) -> Callable[[], object]:
        data = copy.deepcopy(answers)
        change(data)
        return lambda: jev.g_plus_scores(data, False, request, settings, weights)

    answer_refusals = {
        "a missing success answer": lambda a: a.pop("success_sonnet_b"),
        "a missing difficulty answer": lambda a: a.pop("difficulty_8"),
        "a missing request-type answer": lambda a: a.pop("request_type"),
        "a Noul answer that is a dict": lambda a: a.update(difficulty_1={"x": 0.5}),
        "a Noul answer above 1": lambda a: a.update(difficulty_1=1.01),
        "a Noul answer below 0": lambda a: a.update(difficulty_1=-0.01),
        "a NaN Noul answer": lambda a: a.update(difficulty_1=float("nan")),
        "a boolean Noul answer": lambda a: a.update(difficulty_1=True),
        "a Choice answer that is a float": lambda a: a.update(request_type=0.5),
        "a Choice answer missing an option": lambda a: a["request_type"].pop("opt four"),
        "a Choice answer with an extra option": lambda a: a["request_type"].update(other=0.0),
        "a Choice probability above 1": lambda a: a["request_type"].update({"opt one": 1.5}),
    }
    for what, change in answer_refusals.items():
        check(refused(answered(change)), f"{what} is refused, not scored")
    check(not refused(answered(lambda a: None)), "the guard lets well-formed answers through")

    print("\n-- the weights file on disk --")
    check(jev.WEIGHTS_FILE.name == "jev_weights.json" and jev.WEIGHTS_FILE.parent == (PROJECT_ROOT / "routers"),
          "the weights file is routers/jev_weights.json, beside the router")  # fmt: skip

    print("\n-- imports --")
    sources = {relative: ast.parse((PROJECT_ROOT / relative).read_text()) for relative in closure("routers/jev.py")}
    imported = set().union(*(imported_packages(tree) for tree in sources.values()))
    check("numpy" in imported and "json" in imported and "routers" in imported,
          "the guard finds the imports it looks for: numpy, json and routers")  # fmt: skip
    check(imported_packages(ast.parse("import sklearn\nfrom analysis import freeze")) == {"sklearn", "analysis"},
          "the guard finds an import of scikit-learn and analysis in a module that has them")  # fmt: skip
    allowed = set(sys.stdlib_module_names) | {"numpy", "routers"}
    check(not imported - allowed,
          f"the closure imports the standard library and numpy only ({sorted(imported - allowed)})")  # fmt: skip
    check(
        "routers/jev_call.py" in sources, "the closure includes call building, so its imports are held to the same rule"
    )

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
