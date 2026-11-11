"""The gate-score table and the cascade traces built from it (PLAN_v4.md §9; Amendment A17, items 13 and 15).

A table holds, for every (item, config) answer of a split, whether the free checks passed (and which failed first)
and, for each paid gate, either nothing or its gate score, its overhead in dollars at list price and its refusal flag.
Walks, traces and metrics read only the table, never a gate: it is built from the stored records, through each gate's
own parser. It has a column for each paid gate from the first slice on, empty until that gate's records exist.

`GateScoreTable.build` refuses the whole table (GateTableError), before anything is computed, if it misses or repeats
an answer of the split, has a gate score that isn't a number from 0 to 1, has a paid gate's score on an answer that
failed a free check, or has an overhead that is negative or not finite (as the router-score table refuses, A9 item 3).

`cascade_trace` is the one trace builder: it asks `routers.cascade` what to call and charges the paid gate call it
made. A pure cascade's traces are `pure_traces`; a router-started cascade builds each trace with `cascade_trace`, its
start config and the router's overhead.
"""

from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

from analysis.router_names import FREE_GATE, HAIKU, JEV_GATE, SONNET_GATE
from analysis.router_table import is_number
from analysis.simulator import Trace
from routers.cascade import configs_called, paid_gate_called
from routers.gates import CODE, SCHEMA, STOP
from routers.ladder import FRONTIER

PAID_GATES = (JEV_GATE, SONNET_GATE)  # the table's paid columns, in this order
FAILED_FIRST = (STOP, SCHEMA, CODE)


class GateTableError(ValueError):
    """A gate-score table that fails a check, or a trace asked of a table that can't give it. Nothing is built."""


@dataclass(frozen=True)
class PaidScore:
    """One paid gate's entry for one answer: the gate score, None for a refused call (a rejection), the call's
    overhead in dollars at list price, and whether the call was refused. A refused call has no gate score."""

    score: float | None
    overhead: float
    refused: bool


@dataclass(frozen=True)
class GateRow:
    item_id: str
    config: str
    free_passed: bool
    failed_first: str | None  # the first free check that failed, None if all passed
    jev: PaidScore | None = None  # None: the gate wasn't called on this answer (or hasn't been collected)
    sonnet: PaidScore | None = None

    def paid(self, gate: str) -> PaidScore | None:
        """The entry of the paid gate `gate` (`JEV_GATE` or `SONNET_GATE`)."""
        if gate not in PAID_GATES:
            raise GateTableError(f"{gate!r} is not a paid gate ({list(PAID_GATES)})")
        return self.jev if gate == JEV_GATE else self.sonnet


@dataclass(frozen=True)
class GateScoreTable:
    rows: tuple[GateRow, ...]  # in the order given
    _index: dict[tuple[str, str], GateRow] = field(init=False, repr=False, compare=False)

    @classmethod
    def build(cls, rows: Iterable[GateRow], answers: Collection[tuple[str, str]]) -> "GateScoreTable":
        """The table for the answers (`(item_id, config)` pairs) of one split, or a refusal (GateTableError)."""
        rows = list(rows)
        counts = Counter((row.item_id, row.config) for row in rows)
        if repeated := sorted(key for key, n in counts.items() if n > 1):
            raise GateTableError(f"{len(repeated)} answers are in the table more than once, e.g. {repeated[0]}")
        if unknown := sorted(set(counts) - set(answers)):
            raise GateTableError(f"{len(unknown)} rows are not answers of the split, e.g. {unknown[0]}")
        if missing := sorted(set(answers) - set(counts)):
            raise GateTableError(f"{len(missing)} answers of the split are not in the table, e.g. {missing[0]}")
        for row in rows:
            name = f"{row.item_id} {row.config}"
            if row.free_passed != (row.failed_first is None) or (
                row.failed_first is not None and row.failed_first not in FAILED_FIRST
            ):
                raise GateTableError(f"{name} says the free checks passed={row.free_passed} and failed first "
                                     f"{row.failed_first!r}")  # fmt: skip
            for gate in PAID_GATES:
                entry = row.paid(gate)
                if entry is None:
                    continue
                if not row.free_passed:
                    raise GateTableError(f"{name} failed the free check {row.failed_first!r} and has a {gate} entry; "
                                         f"a paid gate judges only answers that pass them")  # fmt: skip
                if entry.score is not None and not (is_number(entry.score) and 0 <= entry.score <= 1):
                    raise GateTableError(f"{name}'s {gate} gate score is {entry.score!r}, which must be a number "
                                         f"from 0 to 1")  # fmt: skip
                if (entry.score is None) != entry.refused:
                    raise GateTableError(f"{name}'s {gate} entry has a refusal flag of {entry.refused} and a gate "
                                         f"score of {entry.score!r}; only a refused call has none")  # fmt: skip
                if not (is_number(entry.overhead) and entry.overhead >= 0):
                    raise GateTableError(f"{name}'s {gate} overhead is {entry.overhead!r}, which must be a finite "
                                         f"number of dollars, at least 0")  # fmt: skip
        return cls(tuple(rows))

    def __post_init__(self) -> None:
        # A frozen dataclass can't assign, so the lookup is set the way the dataclass itself sets fields.
        object.__setattr__(self, "_index", {(row.item_id, row.config): row for row in self.rows})

    def row(self, item_id: str, config: str) -> GateRow:
        try:
            return self._index[(item_id, config)]
        except KeyError:
            raise GateTableError(f"the table has no answer for {item_id} {config}") from None

    def items_of(self, config: str) -> list[str]:
        """The items that have an answer from `config`, in the table's order."""
        return [row.item_id for row in self.rows if row.config == config]


def free_rows(records: Iterable[Mapping[str, object]]) -> list[GateRow]:
    """The table's rows from stored free-check records, with the paid columns empty. The records' `passed` and
    `failed_first` are read as stored."""
    return [GateRow(str(r["item_id"]), str(r["config"]), bool(r["passed"]),
                    None if r["failed_first"] is None else str(r["failed_first"]))
            for r in records]  # fmt: skip


def build_table(
    free_records: Iterable[Mapping[str, object]],
    answers: Collection[tuple[str, str]],
    paid: Mapping[str, Mapping[tuple[str, str], PaidScore]] | None = None,
) -> GateScoreTable:
    """The table of a split from its stored records. `paid` maps a paid gate to its entries, each gate's own parser
    having turned its records into `PaidScore`s keyed by `(item_id, config)`; a gate left out has an empty column.
    Refuses (GateTableError) what `GateScoreTable.build` refuses, and an entry for an answer with no free-check row."""
    paid = paid or {}
    if unknown_gate := sorted(set(paid) - set(PAID_GATES)):
        raise GateTableError(f"{unknown_gate} are not paid gates ({list(PAID_GATES)})")
    rows = free_rows(free_records)
    keys = {(row.item_id, row.config) for row in rows}
    for gate, entries in paid.items():
        if stray := sorted(set(entries) - keys):
            raise GateTableError(f"the {gate} gate has {len(stray)} entries for answers with no free-check row, "
                                 f"e.g. {stray[0]}")  # fmt: skip
    return GateScoreTable.build(
        [GateRow(row.item_id, row.config, row.free_passed, row.failed_first,
                 paid.get(JEV_GATE, {}).get((row.item_id, row.config)),
                 paid.get(SONNET_GATE, {}).get((row.item_id, row.config)))
         for row in rows],
        answers,
    )  # fmt: skip


def cascade_trace(
    table: GateScoreTable,
    item_id: str,
    start: str,
    gate: str,
    cutoff: float | None,
    router_overhead: float = 0.0,
) -> Trace:
    """One item's trace in a cascade that starts at `start` and ends in `gate` at `cutoff`.

    The configs are `routers.cascade.configs_called`'s. The overhead is the router's (`router_overhead`, 0 for a pure
    cascade) plus the paid gate's own overhead when `paid_gate_called` says it was called: nothing for the free
    checks, nothing for an answer a free check rejected, nothing for a start at `opus`. Refuses (GateTableError) a
    paid gate with no cutoff or a free-checks-only cascade with one, and a gate call the table has no entry for.
    """
    if (gate == FREE_GATE) != (cutoff is None):
        raise GateTableError(f"a cutoff goes with a paid gate and only with one, not {cutoff!r} with {gate!r}")
    if start == FRONTIER:  # no answer is looked up: an item sent straight to the frontier gets no gate
        return Trace(item_id, configs_called(start, False, None, cutoff), router_overhead)
    row = table.row(item_id, start)
    entry = None
    if paid_gate_called(start, row.free_passed, cutoff):
        entry = row.paid(gate)
        if entry is None:
            raise GateTableError(f"{item_id} {start} passed the free checks and has no {gate} entry")
    score = None if entry is None else entry.score
    configs = configs_called(start, row.free_passed, score, cutoff)
    return Trace(item_id, configs, router_overhead + (0.0 if entry is None else entry.overhead))


def pure_traces(table: GateScoreTable, gate: str, cutoff: float | None, start: str = HAIKU) -> list[Trace]:
    """A pure cascade's traces: one per item that has an answer from `start`, in the table's order."""
    return [cascade_trace(table, item_id, start, gate, cutoff) for item_id in table.items_of(start)]
