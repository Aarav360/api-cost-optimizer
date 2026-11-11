"""The fixed cutoffs of the paid gates (PLAN_v4.md §9; Amendment A17, item 12).

A router-started cascade walks its router's steps with its gate's cutoff held fixed. The cutoff is Youden's J on
train: over the distinct gate scores, accepting at a score at or above the cutoff, the one that maximizes the share of
wrong answers rejected minus the share of right answers rejected. Ties go to the higher (stricter) cutoff. It is pooled
over `haiku`'s, `sonnet`'s and `opus_low`'s train answers that pass the free checks, so each paid gate has one cutoff,
shared by all four routers. It reads the gate-score table and train outcomes, which are no `opus` outcome (`opus` never
runs on train), and nothing from dev.

A refused call has no gate score and is always a rejection (the cascade decision's), so it counts as rejected at every
cutoff, among the wrong answers or the right ones, and is no candidate cutoff itself. Shares are exact fractions, so a
tie is a tie and no float rounding can break it.
"""

from collections.abc import Iterable
from fractions import Fraction

from analysis.gate_table import PAID_GATES, GateScoreTable
from analysis.matrix import Matrix
from routers.ladder import SCORED


class CutoffError(ValueError):
    """A fixed cutoff that can't be computed: no paid gate, answers that miss a gate score or an outcome, or answers
    that are all right, all wrong or all unscored."""


def youden_cutoff(answers: Iterable[tuple[float | None, bool]]) -> float:
    """The cutoff that maximizes Youden's J over `(gate score, answer was right)` pairs, ties to the higher cutoff.
    A gate score of None is a refused call. Refuses (CutoffError) answers with no right one, no wrong one, or no score."""
    answers = list(answers)
    wrong = [score for score, right in answers if not right]
    right = [score for score, right in answers if right]
    if not wrong or not right:
        raise CutoffError(
            f"the cutoff needs both right and wrong answers, not {len(right)} right and {len(wrong)} wrong"
        )
    candidates = sorted({score for score, _ in answers if score is not None})
    if not candidates:
        raise CutoffError("every answer was refused, so there is no gate score to cut at")

    def rejected(scores: list[float | None], cutoff: float) -> int:
        return sum(score is None or score < cutoff for score in scores)

    best_cutoff, best_j = candidates[0], None
    for cutoff in candidates:
        j = Fraction(rejected(wrong, cutoff), len(wrong)) - Fraction(rejected(right, cutoff), len(right))
        if best_j is None or j >= best_j:  # candidates ascend, so >= lets a tie move to the higher cutoff
            best_cutoff, best_j = cutoff, j
    return best_cutoff


def fixed_cutoff(table: GateScoreTable, train: Matrix, gate: str) -> float:
    """The fixed cutoff of the paid gate `gate` from a train gate-score table and the train matrix: Youden's J over
    the answers of `haiku`, `sonnet` and `opus_low` that passed the free checks, right where the matrix says success.
    Refuses (CutoffError) a gate that isn't paid, a passing answer with no entry for the gate or no cell in the matrix,
    and what `youden_cutoff` refuses."""
    if gate not in PAID_GATES:
        raise CutoffError(f"{gate!r} is not a paid gate ({list(PAID_GATES)}); the free checks have no cutoff")
    answers: list[tuple[float | None, bool]] = []
    for row in table.rows:
        if row.config not in SCORED or not row.free_passed:
            continue
        entry = row.paid(gate)
        if entry is None:
            raise CutoffError(f"{row.item_id} {row.config} passed the free checks and has no {gate} entry")
        right = _was_right(train, row.item_id, row.config)
        answers.append((entry.score, right))
    return youden_cutoff(answers)


def _was_right(train: Matrix, item_id: str, config: str) -> bool:
    """Whether the matrix says the answer was right. Refuses (CutoffError) a missing cell or a success that isn't 0 or 1."""
    try:
        success = train.cells[(item_id, config)].success
    except KeyError:
        raise CutoffError(f"the matrix has no outcome for {item_id} {config}") from None
    if success not in (0, 1):
        raise CutoffError(f"{item_id} {config} has a success of {success!r}, which must be 0 or 1")
    return success == 1
