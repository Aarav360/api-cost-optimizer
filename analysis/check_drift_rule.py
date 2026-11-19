"""Check the shared drift rule of A15 item 17.3 (PLAN_v4.md Amendment A15 item 17; A17 item 26.2; A19 items 2 and 13; issue #77, #84).

Run from the project root:  uv run python -m analysis.check_drift_rule

Pure arithmetic on made-up numbers: no results/ folder, no Docker, no Jev or Anthropic call, no test item. Every expected
number is worked by hand below. The rule refuses (`DriftRefusal`) if a model ID isn't the pin, if D8 > 3 x D6 or if a key's
mean signed difference exceeds D6 in size, and otherwise reports D6, D8, the shifts and the flips.

- a model ID other than the pin refuses, whatever else is in range, and so does a missing one;
- D8 > 3 x D6 refuses, exactly 3 x D6 goes on;
- the shift rule's known positive: a uniform move of every router score that leaves D8 inside 3 x D6 still refuses;
- one key out of range refuses though the others are in range;
- within the limits, the report holds D6, D8 and the shift of every key, the model IDs and the flips, as the record keeps them;
- D8 and the shift come from `fit_jev_router.noise_difference` on hand-made scores;
- the decisions that flip at a threshold are counted by the router's own `choose` (`>=`), highest threshold first;
- the refusal is a `StepRefusal`, so the runner shows it.
"""

import sys

from analysis import drift_rule
from analysis.checking import refused
from analysis.drift import DriftRefusal
from analysis.drift_rule import JEV_PIN, Flip, Measure, apply_rule, decision_flips, measures_of
from analysis.final_run import StepRefusal
from analysis.fit_jev_router import Noise, noise_difference

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def why(**given) -> str | None:
    """The refusal's message for `apply_rule(**given)`, None if the rule goes on."""
    try:
        apply_rule("drift_test", **given)
    except DriftRefusal as error:
        return str(error)
    return None


GOOD = {"haiku": Measure(0.25, 0.5, 0.125), "sonnet": Measure(0.25, 0.75, -0.25)}  # binary fractions: exact in a float
PINNED = [JEV_PIN, JEV_PIN, JEV_PIN]

# The pin ------------------------------------------------------------------------------------------------------------
check(JEV_PIN == "jev-1.14.0", "the pin is A13's jev-1.14.0")
check(why(pin=JEV_PIN, model_ids=PINNED, measures=GOOD) is None, "a resend on the pin, within both limits, goes on")
message = why(pin=JEV_PIN, model_ids=[JEV_PIN, "jev-1.15.0"], measures=GOOD)
check(message is not None and "jev-1.15.0" in message and "drift_test" in message,
      "known positive: one resend from another model ID refuses, and its message names the check and the model")  # fmt: skip
check(why(pin=JEV_PIN, model_ids=[JEV_PIN, None], measures=GOOD) is not None, "a resend with no model ID refuses")
check(why(pin=JEV_PIN, model_ids=[], measures=GOOD) is not None, "no resend at all refuses: nothing was checked")
check(why(pin="claude-sonnet-5-5", model_ids=["claude-sonnet-5-5"], measures=GOOD) is None,
      "the pin is the caller's: the Sonnet gate's check passes its own")  # fmt: skip
check(why(pin="claude-sonnet-5-5", model_ids=[JEV_PIN], measures=GOOD) is not None, "a Jev ID is no Sonnet pin")

# D8 against 3 x D6 -------------------------------------------------------------------------------------------------
check(why(pin=JEV_PIN, model_ids=PINNED, measures={"haiku": Measure(0.25, 0.75, 0.0)}) is None,
      "D8 exactly 3 x D6 goes on")  # fmt: skip
message = why(pin=JEV_PIN, model_ids=PINNED, measures={"haiku": Measure(0.25, 0.7501, 0.0)})
check(message is not None and "haiku" in message and "D8" in message, "D8 above 3 x D6 refuses, naming the config")
check(why(pin=JEV_PIN, model_ids=PINNED, measures={"haiku": Measure(0.0, 0.0, 0.0)}) is None,
      "a D6 of zero and a D8 of zero go on (nothing moved)")  # fmt: skip
check(why(pin=JEV_PIN, model_ids=PINNED, measures={"haiku": Measure(0.0, 0.001, 0.0)}) is not None,
      "a D6 of zero and any D8 refuses")  # fmt: skip

# The shift ----------------------------------------------------------------------------------------------------------
check(why(pin=JEV_PIN, model_ids=PINNED, measures={"haiku": Measure(0.25, 0.25, 0.25)}) is None,
      "a shift exactly the size of D6 goes on")  # fmt: skip
check(why(pin=JEV_PIN, model_ids=PINNED, measures={"haiku": Measure(0.25, 0.25, -0.2501)}) is not None,
      "a shift beyond D6 in size refuses, whichever way it moves")  # fmt: skip

# The known positive: every score moves by the same 0.0125, with D6 at 0.0120 (G's haiku, to four decimals in A15 item 19).
first = {f"item-{n}": {"haiku": 0.25 + n / 100} for n in range(5)}
moved = {item: {"haiku": scores["haiku"] + 0.0125} for item, scores in first.items()}
noise = noise_difference(first, moved, ["haiku"])
check(abs(noise["haiku"].d6 - 0.0125) < 1e-12 and abs(noise["haiku"].shift - 0.0125) < 1e-12,
      "the uniform move: D8 and the shift are both the 0.0125 every score moved by")  # fmt: skip
uniform = measures_of({"haiku": Noise(0.0120, 0.0001)}, noise)
check(
    uniform["haiku"].d8 < 3 * uniform["haiku"].d6,
    "known positive: the uniform move leaves D8 (0.0125) well inside 3 x D6 (0.036)",
)
message = why(pin=JEV_PIN, model_ids=PINNED, measures=uniform)
check(message is not None and "shift" in message and "D8" not in message,
      "and the shift rule alone refuses it, naming the shift and not D8")  # fmt: skip
check(drift_rule.apply_rule.__doc__ is not None and "strictly" in drift_rule.apply_rule.__doc__,
      "the rule's words say 'strictly greater', as the equality cases above hold")  # fmt: skip

# One key out of range -----------------------------------------------------------------------------------------------
three = {**GOOD, "opus_low": Measure(0.25, 0.25, 0.5)}
message = why(pin=JEV_PIN, model_ids=PINNED, measures=three)
check(message is not None and "opus_low" in message and "haiku" not in message and "sonnet" not in message,
      "one key past a limit refuses and names that key alone")  # fmt: skip
both = why(pin=JEV_PIN, model_ids=["jev-x"], measures={"haiku": Measure(0.25, 1.0, 0.0)})
check(both is not None and "jev-x" in both and "D8" in both, "every broken rule is in the one message")

# The report ---------------------------------------------------------------------------------------------------------
report = apply_rule("drift_test", pin=JEV_PIN, model_ids=PINNED, measures=GOOD)
check(report.record()["measures"] == {"haiku": {"d6": 0.25, "d8": 0.5, "shift": 0.125},
                                      "sonnet": {"d6": 0.25, "d8": 0.75, "shift": -0.25}},
      "the report holds D6, D8 and the shift of every key")  # fmt: skip
check(report.record()["model_ids"] == [JEV_PIN] and report.record()["pin"] == JEV_PIN and report.record()["d8_factor"] == 3.0,
      "it holds the distinct model IDs, the pin and the factor")  # fmt: skip
check(report.record()["flips"] == [], "with no flips given, none are recorded")

# D8 and the shift from the frozen arithmetic, on a hand-made pair of passes ------------------------------------------------
pass_one = {"a": {"haiku": 0.5, "sonnet": 0.5}, "b": {"haiku": 0.75, "sonnet": 0.25}}
pass_two = {"a": {"haiku": 0.625, "sonnet": 0.5}, "b": {"haiku": 0.625, "sonnet": 0.5}}
d8 = noise_difference(pass_one, pass_two, ["haiku", "sonnet"])
made = measures_of(noise_difference(pass_one, pass_one, ["haiku", "sonnet"]), d8)
check(
    made["haiku"] == Measure(0.0, 0.125, 0.0),
    "haiku: moves of +0.125 and -0.125 make D8 0.125 and a shift of zero (D6 from equal passes: 0)",
)
check(made["sonnet"] == Measure(0.0, 0.125, 0.125), "sonnet: moves of 0 and +0.25 make D8 0.125 and a shift of +0.125")

# Flips ----------------------------------------------------------------------------------------------------------------
# `choose` takes the first of haiku, sonnet, opus_low whose score reaches the threshold, else opus (ties count).
before = {"p": {"haiku": 0.5, "sonnet": 0.5, "opus_low": 0.5},  # at 0.5: haiku; at 0.75: opus
          "q": {"haiku": 0.5, "sonnet": 0.75, "opus_low": 0.75},  # at 0.5: haiku; at 0.75: sonnet
          "r": {"haiku": 0.25, "sonnet": 0.25, "opus_low": 0.25}}  # fmt: skip
after = {"p": {"haiku": 0.5, "sonnet": 0.5, "opus_low": 0.5},  # unchanged
         "q": {"haiku": 0.5, "sonnet": 0.5, "opus_low": 0.75},  # at 0.5: haiku; at 0.75: opus_low
         "r": {"haiku": 0.5, "sonnet": 0.25, "opus_low": 0.25}}  # at 0.5: haiku now (was opus); at 0.75: opus  # fmt: skip
flips = decision_flips(before, after, {0.5: ["realistic 99%", "uniform 99%"], 0.75: ["realistic 97%"]})
check(flips == (Flip(0.75, ("realistic 97%",), ("q",)), Flip(0.5, ("realistic 99%", "uniform 99%"), ("r",))),
      "flips: highest threshold first, each with its labels and the items whose chosen config changed")  # fmt: skip
check(
    decision_flips(before, before, {0.5: ["x"]}) == (Flip(0.5, ("x",), ()),),
    "no drift, no flips, and the threshold is still listed",
)
check(refused(ValueError, lambda: decision_flips(before, {"p": after["p"]}, {0.5: ["x"]})),
      "scores for other items than the first pass's are refused")  # fmt: skip
with_flips = apply_rule("drift_test", pin=JEV_PIN, model_ids=PINNED, measures=GOOD, flips=flips)
check(with_flips.record()["flips"] == [{"threshold": 0.75, "points": ["realistic 97%"], "count": 1, "items": ["q"]},
                                       {"threshold": 0.5, "points": ["realistic 99%", "uniform 99%"], "count": 1, "items": ["r"]}],
      "the record holds each flip's threshold, labels, count and items")  # fmt: skip

# The runner shows it ------------------------------------------------------------------------------------------------------
check(issubclass(DriftRefusal, StepRefusal), "a drift refusal is a StepRefusal, so the runner shows its message")

if problems:
    print(f"\n{len(problems)} problem(s)")
    sys.exit(1)
print("\nall checks passed")
