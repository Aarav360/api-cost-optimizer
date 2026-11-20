"""Stage 8's spending ledger: its own two caps, counted afresh, and the whole run's projection before its first call
(PLAN_v4.md Amendment A19, item 9; A17 item 18; issue #77, #82).

A17 item 18's caps hold, but Stage 8 counts them from nothing. `recorded_spend` in `analysis/gate_spend.py` sums Stage 7's
records and never reads `results/jev/`, so reused as it is it would hand Stage 8 what is left of Stage 7's $25. This ledger
reads only Stage 8's record files (`analysis/record_files.py`: `*_test.jsonl` and `*_drift.jsonl` in `results/gates/` and
`results/jev/`), from the stored records, never from a counter that could drift, with the gate spend module's billed-token
reader and prices and the Jev routers' price. Stage 7's cap and its records are untouched.

- **$25 (`CALLS`)** covers every Stage 8 call but the live replay's: the drift resends, G's and G+'s Jev calls on test and
  both gates.
- **$12 (`LIVE`)** covers the liveness calls and the live replay (A19 item 9 raised A17 item 18's $10). Their records are in the final evaluation's own folder
  (`world.output`), `LIVE_FILES`, since they are not router or gate records.

**The projection.** Each call ticket adds one line to `PROJECTORS`: a function `(run) -> dollars` still to be spent in its
pool (a resume projects only requests with no stored response). The liveness step, the first to call, adds them all up
(`check_projection`) and refuses (`LedgerRefusal`) when a pool's recorded spend plus its projection is over its cap, before
anything is sent. A step that sends in batches also asks `refusal` before each submission. A spend that can't be known can't
be under a cap, so a record without whole billed token counts, or a Stage 8 file the ledger has no price for, refuses.

**Before each call** (the Stage 8 review's spec 3: "collectors consult it before they submit"): a step that calls one by one
(G's and G+'s Jev calls, the Jev gate, the Sonnet gate's drift resends) holds a `Meter`: the pool's recorded spend read once from
the stored records when the step starts, plus the dollars each call of the step was billed, which it counts as the call returns.
`Meter.before(projected)` refuses (`LedgerRefusal`) a call whose projected cost would take the pool over its cap, so a run that
costs more than its projection still stops at the cap, and `MeteredJev` puts that around a Jev client. The meter counts what the
step itself sent; it doesn't re-read the files per call, which at about 20,000 stored records would cost more than the calls.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from analysis import record_files
from analysis.final_run import StepRefusal, resolve
from analysis.gate_spend import GateSpendError, billed_tokens, jev_dollars, jev_projected_cost, list_dollars
from analysis.record_files import BATCH, FREE, JEV, STAGE_8_FILES, files_in, is_stage_8
from analysis.world import World
from harness.client import load_models
from routers import jev, sonnet_gate

if TYPE_CHECKING:
    from analysis.final_run import Run

CALLS, LIVE = "calls", "live"
CAPS = {CALLS: 25.0, LIVE: 12.0}  # A17 item 18's $25, and A19 item 9's $12 for the live pool; both counted afresh
TOLERANCE_USD = 1e-9  # a float sum of many small prices should not tip a total that is exactly on a cap
# The final evaluation's own files, in `world.output`, priced as Anthropic calls at list price by the model each record
# requested: the liveness calls' (`analysis/final_liveness.py`) and the live replay's calls (`analysis/final_replay.py`),
# in which a record with `"service": "jev"` is a Jev gate call, priced at Jev's input-token price.
LIVE_FILES = ("liveness.jsonl", "replay.jsonl")


class LedgerRefusal(StepRefusal):
    """A projection over a cap, or a spend that can't be counted. Nothing was sent."""


@dataclass(frozen=True)
class Spend:
    calls: float  # in `CALLS`' pool
    live: float  # in `LIVE`'s
    files: int  # how many record files were read

    def of(self, pool: str) -> float:
        return {CALLS: self.calls, LIVE: self.live}[pool]


@dataclass(frozen=True)
class Projection:
    name: str
    pool: str
    dollars: float  # still to be spent: an upper bound, so a projection never lets through what the cap would stop


@dataclass(frozen=True)
class Projector:
    name: str
    pool: str
    entry: str | None  # "module:function" of `(run) -> float`, or None while the ticket that owns it isn't built


# One line per call ticket, each between comment lines so two tickets' lines merge without a conflict. A projector whose
# entry is None projects nothing yet; the ticket named above it fills it in.
PROJECTORS = (
    # #82: the liveness calls
    Projector("liveness", LIVE, "analysis.final_liveness:project"),
    # #84: the drift resends and test's calls for G and G+
    Projector("jev_router", CALLS, "analysis.final_jev:project"),
    # #85: the Jev gate's drift resends and test calls
    Projector("jev_gate", CALLS, "analysis.final_gates:project"),
    # #86: the Sonnet gate's drift resends and test batch
    Projector("sonnet_gate", CALLS, "analysis.final_sonnet:project"),
    # #91: the live replay
    Projector("replay", LIVE, "analysis.final_replay:project"),
)


class Meter:
    """One step's running spend in one pool: what the stored records held when it started, and what each call it has made was
    billed since. A running count, so not a record."""

    def __init__(self, recorded: float, pool: str = CALLS):
        self.pool, self.recorded, self.spent = pool, recorded, 0.0

    @classmethod
    def of(cls, run: "Run", pool: str = CALLS) -> "Meter":
        """A meter started from the pool's recorded spend (a spend that can't be counted refuses, as `check_projection` does)."""
        try:
            return cls(recorded_spend(run.world.results, run.world.output).of(pool), pool)
        except GateSpendError as error:
            raise LedgerRefusal(f"the recorded spend can't be counted: {error}; nothing was sent") from error

    def before(self, projected: float) -> None:
        """Refuse (LedgerRefusal) a call whose projected cost, with what is recorded and what the step has been billed, would be
        over the pool's cap. Call it before the call is sent."""
        total = self.recorded + self.spent
        words = refusal(Spend(total if self.pool == CALLS else 0.0, total if self.pool == LIVE else 0.0, 0),
                        {self.pool: projected})  # fmt: skip
        if words:
            raise LedgerRefusal(words)

    def paid(self, dollars: float) -> None:
        self.spent += dollars


class MeteredJev:
    """A Jev client that asks its meter before each call and counts the input tokens the call was billed, at Jev's price.
    Everything else about the client (its retries, its stop) is the inner client's."""

    def __init__(self, client: Any, meter: Meter, price_per_million: float):
        self.client, self.meter, self.price = client, meter, price_per_million

    def call(self, body: bytes, sha256: str, label: str) -> Any:
        self.meter.before(jev_projected_cost([len(body)], self.price))
        result = self.client.call(body, sha256, label)
        self.meter.paid(jev_dollars(billed_tokens([result.to_record()], label)[0], self.price))
        return result


def _records(path: Path) -> list[object]:
    """A Stage 8 record file's complete records (`record_files.read_jsonl`); a line that isn't JSON is a spend that can't be
    counted (GateSpendError)."""
    try:
        return record_files.read_jsonl(path)
    except ValueError as error:
        raise GateSpendError(str(error)) from error


def recorded_spend(
    results_dir: Path,
    output_dir: Path,
    *,
    jev_price_per_million: float | None = None,
    sonnet: sonnet_gate.SonnetSettings | None = None,
    models: Mapping | None = None,
) -> Spend:
    """What Stage 8's stored records say was spent, in dollars; the prices default to the project's settings.

    Refuses (GateSpendError) a record without whole, non-negative billed token counts, and a Stage 8 file in the gate
    and Jev folders that has no price here, since a file the ledger can't price is a spend it can't know."""
    if jev_price_per_million is None:
        jev_price_per_million = jev.read_client_settings().price_per_million_input_tokens
    sonnet = sonnet or sonnet_gate.load_settings()
    models = models or load_models()
    calls = live = 0.0
    files = 0
    for folder, priced in STAGE_8_FILES.items():
        for path in files_in(results_dir / folder):
            if not is_stage_8(path.name):
                continue
            if path.name not in priced:
                raise GateSpendError(f"{folder}/{path.name} is a Stage 8 record file the ledger has no price for")
            if priced[path.name] == FREE:
                continue
            files += 1
            billed_in, billed_out = billed_tokens(_records(path), path.name)
            if priced[path.name] == JEV:
                calls += jev_dollars(billed_in, jev_price_per_million)
            else:
                list_price = list_dollars(billed_in, billed_out, sonnet.input_per_million, sonnet.output_per_million)
                calls += list_price * (float(models["batch_discount"]) if priced[path.name] == BATCH else 1.0)
    live = live_spend(output_dir, models, jev_price_per_million)
    files += sum((output_dir / name).is_file() for name in LIVE_FILES)
    return Spend(calls, live, files)


def live_spend(output_dir: Path, models: Mapping | None = None, jev_price_per_million: float | None = None) -> float:
    """What the final evaluation's own record files hold, in dollars: the live pool's recorded spend. It reads only those
    files, so the replay can ask before each call without reading the gate and Jev folders."""
    models = models or load_models()
    if jev_price_per_million is None:
        jev_price_per_million = jev.read_client_settings().price_per_million_input_tokens
    total = 0.0
    for name in LIVE_FILES:
        path = output_dir / name
        if path.is_file():
            total += sum(call_dollars(record, name, models, jev_price_per_million) for record in _records(path))
    return total


def call_dollars(record: object, name: str, models: Mapping, jev_price_per_million: float) -> float:
    """One live call's cost: a Jev gate call (`"service": "jev"`) by its billed input tokens at Jev's price, any other by
    the model its request named, at list price."""
    if isinstance(record, Mapping) and record.get("service") == "jev":
        billed_in, _ = billed_tokens([record], name)
        return jev_dollars(billed_in, jev_price_per_million)
    return _anthropic_dollars(record, name, models)


def _anthropic_dollars(record: object, name: str, models: Mapping) -> float:
    """An Anthropic call's cost at list price, by the model its request named: billed tokens times that model's prices."""
    billed_in, billed_out = billed_tokens([record], name)
    request = record.get("request") if isinstance(record, Mapping) else None
    model = request.get("model") if isinstance(request, Mapping) else None
    price = models["prices"].get(model)
    if price is None:
        raise GateSpendError(f"{name} has a call to {model!r}, which the price sheet doesn't list")
    return list_dollars(billed_in, billed_out, float(price["input"]), float(price["output"]))


def refusal(recorded: Spend, projected: Mapping[str, float]) -> str | None:
    """None while each pool's recorded spend plus its projection is within its cap; otherwise the refusal's words.
    `projected` maps a pool to the dollars still to be spent in it."""
    words = []
    for pool, cap in CAPS.items():
        total = recorded.of(pool) + projected.get(pool, 0.0)
        if total > cap + TOLERANCE_USD:
            words.append(
                f"the {pool} pool has ${recorded.of(pool):.4f} recorded and ${projected.get(pool, 0.0):.4f} projected: "
                f"${total:.2f} is over Stage 8's cap of ${cap:.2f}"
            )
    return "; ".join(words) + "; nothing was sent" if words else None


def cap_words(run: "Run") -> Callable[[float], str | None]:
    """The refusal's words for a submission that would send `dollars` more in the calls pool: Stage 8's recorded spend plus
    those dollars against the $25 cap, or None. Read from the stored records at each ask."""

    def words(dollars: float) -> str | None:
        return refusal(recorded_spend(run.world.results, run.world.output), {CALLS: dollars})

    return words


def check_cap(run: "Run", dollars: float) -> None:
    """Refuse (LedgerRefusal) before a step's first call if Stage 8's recorded spend plus `dollars` is over the $25 cap."""
    if words := cap_words(run)(dollars):
        raise LedgerRefusal(words)


def totals(projections: Sequence[Projection]) -> dict[str, float]:
    return {pool: sum(p.dollars for p in projections if p.pool == pool) for pool in CAPS}


def report(recorded: Spend, projections: Sequence[Projection] = ()) -> dict[str, dict[str, float]]:
    """The spend against each cap, as the record keeps it: the cap, what is recorded and what is projected."""
    projected = totals(projections)
    return {
        pool: {"cap": cap, "recorded": recorded.of(pool), "projected": projected[pool]} for pool, cap in CAPS.items()
    }


def spend_report(world: World) -> dict[str, dict[str, float]]:
    """The record's closing field: each cap and what the stored records say was spent against it."""
    return report(recorded_spend(world.results, world.output))


def projections(run: "Run") -> list[Projection]:
    """What every built projector says is still to be spent. Run after the unlock, since a projector may read the handle."""
    result = []
    for projector in PROJECTORS:
        if projector.entry is None:
            continue
        result.append(Projection(projector.name, projector.pool, resolve(projector.entry)(run)))
    return result


def check_projection(run: "Run") -> dict[str, dict[str, float]]:
    """The whole run's spend, projected before its first call: refuses (LedgerRefusal) when a cap would be passed, and
    otherwise returns the record's account of each cap, recorded and projected."""
    try:
        recorded = recorded_spend(run.world.results, run.world.output)
    except GateSpendError as error:
        raise LedgerRefusal(f"the recorded spend can't be counted: {error}; nothing was sent") from error
    projected = projections(run)
    words = refusal(recorded, totals(projected))
    if words:
        raise LedgerRefusal(words)
    return report(recorded, projected)
