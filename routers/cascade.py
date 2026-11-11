"""The cascade decision (PLAN_v4.md §9; Amendment A17, items 8, 9 and 19): one function, shared by the simulated
traces and Stage 8's live cascade, so the offline and live cascades can't drift apart.

A cascade tries a start config and lets the free checks, then a paid gate if it has one, judge its answer. A rejected
answer goes straight to `opus`, whose answer is always accepted: one escalation, one cutoff, at most two model calls.
An item whose start is `opus` gets no gate. The free checks come first: a failure escalates at once and no paid gate
is called. A paid gate's refusal arrives as a gate score of None, which is a rejection.
"""

from routers.ladder import FRONTIER, LADDER


def paid_gate_called(start: str, free_passed: bool, cutoff: float | None) -> bool:
    """Whether a trace pays a paid gate call: the cascade has one (a cutoff), the start isn't `opus` and the free
    checks passed. A free-checks-only cascade has no cutoff."""
    return cutoff is not None and start != FRONTIER and free_passed


def configs_called(start: str, free_passed: bool, gate_score: float | None, cutoff: float | None) -> tuple[str, ...]:
    """The configs called, in order: the start alone when it is `opus`, or when the free checks pass and either the
    cascade has no paid gate (`cutoff` is None) or the gate score is at least the cutoff; otherwise the start, then
    `opus`. A gate score of None, a refused call, is below every cutoff. Refuses (ValueError) a start off the ladder."""
    if start not in LADDER:
        raise ValueError(f"{start!r} is not a config on the ladder {LADDER}")
    if start == FRONTIER:
        return (FRONTIER,)
    if not free_passed:
        return (start, FRONTIER)
    if cutoff is None or (gate_score is not None and gate_score >= cutoff):
        return (start,)
    return (start, FRONTIER)
