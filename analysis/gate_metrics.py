"""Gate metrics: rates and counts at a cutoff, within-task concordance, the train signal check and calibration bins
(PLAN_v4.md §9; Amendment A17, item 21).

Every function reads a gate-score table (`analysis.gate_table`) and the results matrix's outcomes; none reads a gate. A
paid gate is measured on the answers that pass the free checks, which is what the policy sees: an answer a free check
rejects never reaches it. The free checks get a row of their own, over all answers.

**The rates** (`gate_rates`), per gate, config and task, at a cutoff: the false accept rate is wrong answers accepted
over wrong answers, the false escalate rate right answers rejected over right answers. Beside them go the raw counts,
which convert straight into quality points lost and dollars spent: wrong answers accepted and right answers rejected,
out of all the config's answers (`answers`), whatever the free checks said. A paid gate accepts a gate score at or above
the cutoff, as `routers.cascade` does; a refused call has no gate score and is a rejection. The free checks have no
cutoff: they accept what passes them.

**Within-task concordance** (`gate_concordance`) is `router_metrics.within_task_concordance` on a gate's scores, cells
being the task alone. The gated answers are a subset of the items, so the intervals come from the master resamples
sliced to those items' columns (`subset_draws`), which keeps them paired with every other number; a fresh resample of
the subset would not be. A refused call counts as below every gate score (`REFUSED_SCORE`), since a cascade rejects it
at every cutoff. `train_signal_check` reads the Jev gate, the Sonnet gate and the code check per config against 0.5.

**Calibration bins** (`calibration_bins`): about ten equal-count bins over one gate's scores for one config and split,
tied scores never split across bins, so a bin can hold more than its share and there can be fewer bins. Each bin gives
its mean gate score and its success rate, with a percentile 95% interval over the master resamples (A8 item 8).
"""

import math
from collections import Counter
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from analysis.bootstrap import Bootstrap, percentile_interval
from analysis.gate_table import PAID_GATES, GateScoreTable, gate_problem
from analysis.matrix import Matrix
from analysis.router_metrics import Reading, within_task_concordance
from analysis.router_names import FREE_GATE
from routers.gates import CODE

# A refused call has no gate score and a cascade rejects it at every cutoff, so for a rank it sits below every gate score
# (all from 0 to 1), tied with other refused calls.
REFUSED_SCORE = -1.0


class GateMetricsError(ValueError):
    """A table, matrix, gate or cutoff the metrics refuse. Nothing is computed."""


@dataclass(frozen=True)
class GateRates:
    """One gate's rates and counts on one config's answers, of one task or pooled over tasks."""

    answers: int  # all the config's answers, whatever the free checks said
    gated: int  # the answers the gate judged: those passing the free checks (all of them, for the free checks)
    right: int  # of the gated answers, those that succeeded
    wrong: int
    wrong_accepted: int
    right_rejected: int
    false_accept_rate: float | None  # wrong answers accepted over wrong answers; None with no wrong answer
    false_escalate_rate: float | None  # right answers rejected over right answers; None with no right answer


def _success_of(matrix: Matrix, item_id: str, config: str) -> float:
    try:
        success = matrix.cells[(item_id, config)].success
    except KeyError:
        raise GateMetricsError(f"the matrix has no answer for {item_id} {config}") from None
    if success not in (0, 1):
        raise GateMetricsError(f"{item_id} {config} has a success of {success!r}, which must be 1 or 0")
    return success


def _rate(count: int, of: int) -> float | None:
    return count / of if of else None


def gate_rates(
    table: GateScoreTable, matrix: Matrix, gate: str, cutoff: float | None
) -> dict[tuple[str, str | None], GateRates]:
    """The rates and counts of `gate` at `cutoff`, by (config, task) and, under task None, pooled over a config's tasks.

    `gate` is the free checks (`FREE_GATE`, with no cutoff) or a paid gate (with one). Configs come in the table's
    order, each pooled row first and then its tasks in the order they first appear; a config the paid gate judged no
    answer of has no row. Refuses (GateMetricsError) a cutoff that doesn't go with the gate or isn't a number from 0
    to 1, an answer the matrix has no success for, and a gated answer with no entry for the gate.
    """
    if problem := gate_problem(gate, cutoff):
        raise GateMetricsError(problem)
    if cutoff is not None and not (isinstance(cutoff, int | float) and 0 <= cutoff <= 1):
        raise GateMetricsError(f"the cutoff must be a number from 0 to 1, not {cutoff!r}")
    tallies: dict[tuple[str, str | None], Counter[str]] = {}
    judged: set[str] = set()  # configs with at least one entry of the paid gate
    for row in table.rows:
        if gate in PAID_GATES and row.paid(gate) is not None:
            judged.add(row.config)
    for row in table.rows:
        if gate in PAID_GATES and row.config not in judged:
            continue
        task = matrix.items[row.item_id].task if row.item_id in matrix.items else None
        if task is None:
            raise GateMetricsError(f"the matrix has no item {row.item_id}")
        right = _success_of(matrix, row.item_id, row.config) == 1
        keys = [(row.config, None), (row.config, task)]
        for key in keys:
            tallies.setdefault(key, Counter())["answers"] += 1
        if not row.free_passed and gate == FREE_GATE:
            accepted = False
        elif gate == FREE_GATE:
            accepted = True
        elif not row.free_passed:
            continue  # a free check rejected it: the paid gate never saw it
        else:
            entry = row.paid(gate)
            if entry is None:
                raise GateMetricsError(f"{row.item_id} {row.config} passed the free checks and has no {gate} entry")
            assert cutoff is not None
            accepted = entry.score is not None and entry.score >= cutoff
        for key in keys:
            tally = tallies[key]
            tally["gated"] += 1
            tally["right" if right else "wrong"] += 1
            if right and not accepted:
                tally["right_rejected"] += 1
            if not right and accepted:
                tally["wrong_accepted"] += 1
    # A config's pooled row first, then its tasks; configs and tasks in the order the table gives them.
    configs = list(dict.fromkeys(config for config, _ in tallies))
    ordered = sorted(tallies, key=lambda key: (configs.index(key[0]), key[1] is not None))
    return {
        key: GateRates(
            tallies[key]["answers"], tallies[key]["gated"], tallies[key]["right"], tallies[key]["wrong"],
            tallies[key]["wrong_accepted"], tallies[key]["right_rejected"],
            _rate(tallies[key]["wrong_accepted"], tallies[key]["wrong"]),
            _rate(tallies[key]["right_rejected"], tallies[key]["right"]),
        )
        for key in ordered
    }  # fmt: skip


def gate_scores(
    table: GateScoreTable, gate: str, config: str, refused: float | None = REFUSED_SCORE
) -> dict[str, float | None]:
    """A paid gate's gate score for each of `config`'s answers it judged, by item ID, in the table's order.

    An answer a free check rejected was never judged, and an answer with no entry hasn't been collected. A refused
    call reads `refused`: below every gate score by default, or absent from the result when `refused` is None.
    """
    scores: dict[str, float | None] = {}
    for row in table.rows:
        if row.config != config or (entry := row.paid(gate)) is None:
            continue
        if entry.score is not None:
            scores[row.item_id] = entry.score
        elif refused is not None:
            scores[row.item_id] = refused
    return scores


def subset_draws(draws: Bootstrap, item_ids: Collection[str]) -> Bootstrap:
    """The master resamples cut to some items: their columns, in the master's order, the same draws.

    A gate judges a subset of the items, and a fresh resample of the subset would not be paired with any other number
    the project reports (the luck of which source groups were drawn); the master's columns are. Refuses (GateMetricsError)
    an item the master doesn't hold.
    """
    column = {item_id: i for i, item_id in enumerate(draws.item_ids)}
    if unknown := sorted(set(item_ids) - set(column)):
        raise GateMetricsError(f"{len(unknown)} items are not in the resamples, e.g. {unknown[0]}")
    wanted = set(item_ids)  # once: a set built inside the comprehension would be rebuilt for every column
    keep = [i for item_id, i in column.items() if item_id in wanted]
    return Bootstrap(tuple(draws.item_ids[i] for i in keep), tuple(draws.tasks[i] for i in keep), draws.counts[:, keep])


def score_concordance(draws: Bootstrap, matrix: Matrix, scores: Mapping[str, float], config: str) -> Reading:
    """Within-task concordance of some scores of `config`'s answers (the task alone is the cell), with intervals from
    the master `draws` cut to the scored items. Refuses (GateMetricsError) no scores."""
    if not scores:
        raise GateMetricsError(f"there are no scores of {config}'s answers to read")
    sub = subset_draws(draws, scores)
    successes = {item_id: _success_of(matrix, item_id, config) for item_id in sub.item_ids}
    cells: dict[str, Hashable] = dict(zip(sub.item_ids, sub.tasks))
    return within_task_concordance(sub, {item_id: scores[item_id] for item_id in sub.item_ids}, successes, cells)


def gate_concordance(draws: Bootstrap, table: GateScoreTable, matrix: Matrix, gate: str, config: str) -> Reading:
    """A paid gate's within-task concordance on `config`'s answers that passed the free checks: the summary that
    needs no cutoff (A11 item 12). A refused call is below every gate score."""
    scores = gate_scores(table, gate, config)
    return score_concordance(draws, matrix, {item_id: s for item_id, s in scores.items() if s is not None}, config)


@dataclass(frozen=True)
class SignalCheck:
    """One signal's within-task concordance on one config, against 0.5 (the value of a signal that knows nothing)."""

    gate: str  # `JEV_GATE`, `SONNET_GATE` or the code check, `routers.gates.CODE`
    config: str
    reading: Reading
    above_half: bool  # the pooled concordance is above 0.5
    interval_above_half: bool  # and so is the lower end of its 95% interval


def code_check_scores(free_records: Iterable[Mapping[str, object]], config: str) -> dict[str, float]:
    """The code check as a score for each of `config`'s answers it applied to: 1 for a pass, 0 for a fail."""
    return {
        str(r["item_id"]): 1.0 if r["code_passed"] else 0.0
        for r in free_records
        if r["config"] == config and r["code_applied"]
    }


def signal_check(
    draws: Bootstrap,
    matrix: Matrix,
    table: GateScoreTable,
    free_records: Sequence[Mapping[str, object]],
    configs: Sequence[str],
) -> list[SignalCheck]:
    """The train signal check (A17 item 21, as A15 item 19): within-task concordance, per config, of the Jev gate, the
    Sonnet gate and the code check, against 0.5. Gate by gate, then configs in the order given.

    A paid gate is read on the answers that passed the free checks, a refused call below every gate score; the code
    check on the answers it applies to, whatever the other free checks said. A signal that judged none of a config's
    answers has no reading. `draws` are the master resamples of the split (train, here).
    """
    found = []
    for gate in (*PAID_GATES, CODE):
        for config in configs:
            if gate == CODE:
                scores = code_check_scores(free_records, config)
            else:
                scores = {i: s for i, s in gate_scores(table, gate, config).items() if s is not None}
            if not scores:
                continue
            reading = score_concordance(draws, matrix, scores, config)
            value, low = reading.pooled.value, reading.pooled.low
            found.append(SignalCheck(gate, config, reading, value is not None and value > 0.5,
                                     low is not None and low > 0.5))  # fmt: skip
    return found


BINS = 10  # A17 item 21: about ten equal-count bins


@dataclass(frozen=True)
class CalibrationBin:
    """One bin of one gate's scores for one config and split."""

    size: int  # answers in the bin
    first: float  # its lowest and highest gate score
    last: float
    mean_score: float
    success_rate: float  # on the split, as it stands
    low: (
        float | None
    )  # the percentile 95% interval of the success rate over the resamples that draw the bin; None if none
    high: float | None
    undefined: float  # the share of resamples that draw none of the bin's answers


def bin_items(scores: Mapping[str, float], bins: int = BINS) -> list[list[str]]:
    """Item IDs grouped into about `bins` equal-count bins by score, lowest first, equal scores never split.

    Items are taken in score order, a whole group of equal scores at a time; after a group, if the items so far reach
    the next of the `bins` equal shares of the total, the bin is closed, and the shares it passed are used up. So a
    tie that straddles a boundary stays in one bin, which can run over its share and leave fewer bins than asked for.
    Refuses (GateMetricsError) no scores, fewer than one bin, and a score that is not a finite number.
    """
    if not scores:
        raise GateMetricsError("there are no scores to bin")
    if bins < 1:
        raise GateMetricsError(f"there must be at least one bin, not {bins}")
    if bad := [v for v in scores.values() if not (isinstance(v, int | float) and math.isfinite(v))]:
        raise GateMetricsError(f"every score must be a finite number, and {len(bad)} aren't, e.g. {bad[0]}")
    groups: dict[float, list[str]] = {}
    for item_id in sorted(scores):
        groups.setdefault(scores[item_id], []).append(item_id)
    total, closed, seen = len(scores), 0, 0
    result: list[list[str]] = [[]]
    for score in sorted(groups):
        result[-1].extend(groups[score])
        seen += len(groups[score])
        if seen * bins >= (closed + 1) * total:  # whole numbers, so no float decides a boundary
            closed = seen * bins // total
            if seen < total:
                result.append([])
    return result


def calibration_bins(
    draws: Bootstrap, matrix: Matrix, scores: Mapping[str, float], config: str, bins: int = BINS
) -> list[CalibrationBin]:
    """Calibration of some scores of `config`'s answers: the bins of `bin_items`, each with its mean gate score and its
    success rate, and the success rate's percentile 95% interval over the master `draws` (A8 item 8's resamples).

    A resample's success rate in a bin weighs each answer by how often the resample draws it; one that draws none of the
    bin is left out of the interval and counted in `undefined`. Pooled over tasks. Refuses (GateMetricsError) what
    `bin_items` refuses, and an answer the matrix or the resamples don't hold.
    """
    members = bin_items(scores, bins)
    sub = subset_draws(draws, scores)
    column = {item_id: i for i, item_id in enumerate(sub.item_ids)}
    success = np.array([_success_of(matrix, item_id, config) for item_id in sub.item_ids])
    result = []
    for items in members:
        cols = [column[item_id] for item_id in items]
        counts = sub.counts[:, cols]
        drawn = counts.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            resampled = np.where(drawn > 0, (counts @ success[cols]) / drawn, np.nan)
        defined = resampled[~np.isnan(resampled)]
        low, high = percentile_interval(defined) if len(defined) else (None, None)
        values = [scores[item_id] for item_id in items]
        result.append(
            CalibrationBin(
                len(items), min(values), max(values), sum(values) / len(values),
                float(success[cols].mean()), low, high, 1 - len(defined) / len(resampled),
            )
        )  # fmt: skip
    return result


def gate_calibration(
    draws: Bootstrap, table: GateScoreTable, matrix: Matrix, gate: str, config: str, bins: int = BINS
) -> list[CalibrationBin]:
    """A paid gate's calibration on `config`'s answers that passed the free checks and got a gate score. A refused call
    has none, so it is left out of the bins (`gate_rates` counts it as a rejection)."""
    scores = gate_scores(table, gate, config, refused=None)
    return calibration_bins(draws, matrix, {item_id: s for item_id, s in scores.items() if s is not None}, config, bins)
