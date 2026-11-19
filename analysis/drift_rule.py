"""The drift rule the Jev and Sonnet checks share (PLAN_v4.md Amendment A15, item 17.3; A17, item 26.2; A19, items 2 and 13;
issue #77, #84).

Before any call on test, a sample of train requests is sent again through the frozen code, and what comes back is compared
with what Stage 6 or 7 stored. Per config (or per key, for a gate) there are three numbers:

- **D6**, the noise the frozen system already had: the mean absolute difference between its two passes on train;
- **D8**, the mean absolute difference of the resend from the first pass;
- **the shift**, the mean signed difference of the resend from the first pass.

The rule **refuses** (`DriftRefusal`) if

1. a returned model ID isn't the pin,
2. D8 > 3 x D6 for any key (a fixed factor and not a test: a permutation test would refuse harmless changes, the question
   is whether the change voids the certification), or
3. a key's shift is larger than D6 in size (the mean absolute difference barely notices a uniform move, yet that is what
   hurts a threshold policy most: it pushes items across every threshold in one direction),

and otherwise returns a report of D6, D8, the shifts and the flips, and the run goes on. A refusal leads to A15 item 11's
path, a new amendment; nothing here chooses a substitute.

This module only applies the rule. The caller measures (`fit_jev_router.noise_difference` for the Jev routers, whose
arithmetic is frozen; `analysis/gate_train.py`'s for the gates), floors D6 where its scores repeat exactly (the Sonnet gate's
0.01) before building a `Measure`, and passes each decision that flips at a frozen threshold or cutoff (`decision_flips` for a
router; a gate builds its own `Flip`s from its cutoff). A message holds model IDs and train numbers, never a test outcome.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from analysis.drift import DriftRefusal
from routers.decision import choose

JEV_PIN = "jev-1.14.0"  # A13's pin, which A15 item 17 checks every resend against
D8_FACTOR = 3.0  # A15 item 17.3: refuse above 3 x D6


class Noisy(Protocol):
    """What `fit_jev_router.noise_difference` and `gate_train.difference` return for one key."""

    @property
    def d6(self) -> float: ...  # the mean absolute difference between two passes

    @property
    def shift(self) -> float: ...  # the mean signed difference, second less first


@dataclass(frozen=True)
class Measure:
    d6: float  # the frozen noise on train
    d8: float  # the resend's mean absolute difference from the first pass
    shift: float  # the resend's mean signed difference from the first pass


@dataclass(frozen=True)
class Flip:
    """The decisions that change at one frozen dial (a router's threshold, a gate's cutoff), and the points that sit on it."""

    threshold: float
    points: tuple[str, ...]  # "realistic 99%", ...
    items: tuple[str, ...]  # the items whose decision differs between the first pass and the resend


@dataclass(frozen=True)
class DriftReport:
    check: str
    pin: str
    model_ids: tuple[
        str, ...
    ]  # the distinct IDs the resends came back with: the pin alone, or the report was a refusal
    measures: Mapping[str, Measure]
    flips: tuple[Flip, ...] = ()

    def record(self) -> dict:
        """What a step keeps in its record (A19 item 10): D6, D8, the shifts and the decisions that flip at each dial."""
        return {
            "pin": self.pin,
            "model_ids": list(self.model_ids),
            "d8_factor": D8_FACTOR,
            "measures": {key: {"d6": m.d6, "d8": m.d8, "shift": m.shift} for key, m in self.measures.items()},
            "flips": [
                {"threshold": f.threshold, "points": list(f.points), "count": len(f.items), "items": list(f.items)}
                for f in self.flips
            ],  # fmt: skip
        }


def measures_of(d6: Mapping[str, Noisy], d8: Mapping[str, Noisy]) -> dict[str, Measure]:
    """Per key, D6 and the shift from the frozen noise pair and D8 from the resend's: `d6[key].d6` is the frozen noise,
    `d8[key].d6` is the mean absolute difference between the first pass and the resend (the same arithmetic, other
    passes), and `d8[key].shift` is the resend's shift. The two must hold the same keys."""
    if set(d6) != set(d8):
        raise ValueError(f"D6 is measured for {sorted(d6)} but D8 for {sorted(d8)}")
    return {key: Measure(d6[key].d6, d8[key].d6, d8[key].shift) for key in d6}


def decision_flips(
    first: Mapping[str, Mapping[str, float]],
    second: Mapping[str, Mapping[str, float]],
    labels: Mapping[float, Sequence[str]],
) -> tuple[Flip, ...]:
    """For a router: at each frozen threshold (`labels` maps it to the points that sit on it), the items whose chosen config
    (`routers.decision.choose`, `>=`) differs between two passes' scores. Highest threshold first. Scores for other items than
    the first pass's are refused."""
    if set(first) != set(second):
        raise ValueError("the two passes must score the same items")
    return tuple(
        Flip(
            threshold,
            tuple(labels[threshold]),
            tuple(item for item in first if choose(first[item], threshold) != choose(second[item], threshold)),
        )  # fmt: skip
        for threshold in sorted(labels, reverse=True)
    )


def apply_rule(
    check: str, *, pin: str, model_ids: Iterable[object], measures: Mapping[str, Measure], flips: Sequence[Flip] = ()
) -> DriftReport:
    """Raise `DriftRefusal(check, why)` if a model ID isn't `pin`, any key's D8 is above 3 x its D6, or any key's shift is
    larger than its D6 in size (both strictly greater: the equality passes). Every broken rule is in the one message.
    Otherwise return the report. Nothing resent at all refuses: nothing was checked."""
    returned = list(model_ids)
    broken = []
    if not returned:
        broken.append("no resend came back, so no model ID was checked")
    if wrong := sorted({str(m) for m in returned if m != pin}):
        broken.append(f"a resend came back as {', '.join(wrong)} and not the pin {pin}")
    for key, m in measures.items():
        if m.d8 > D8_FACTOR * m.d6:
            broken.append(f"{key}: D8 {m.d8:.4g} is more than {D8_FACTOR:g} x D6 {m.d6:.4g}")
        if abs(m.shift) > m.d6:
            broken.append(
                f"{key}: the shift (mean signed difference) {m.shift:+.4g} is larger than D6 {m.d6:.4g} in size"
            )
    if broken:
        raise DriftRefusal(
            check, "; ".join(broken) + "; the way on is a new amendment (A15 item 11), and nothing was called on test"
        )
    return DriftReport(check, pin, tuple(sorted({str(m) for m in returned})), dict(measures), tuple(flips))
