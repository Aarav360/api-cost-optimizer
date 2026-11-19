"""Stage 8's spending ledger: its own two caps, counted afresh, and the whole run's projection before its first call
(PLAN_v4.md Amendment A19, item 9; A17 item 18; issue #77, #82).

A17 item 18's caps hold, but Stage 8 counts them from nothing. `recorded_spend` in `analysis/gate_spend.py` sums Stage 7's
records and never reads `results/jev/`, so reused as it is it would hand Stage 8 what is left of Stage 7's $25. This ledger
reads only Stage 8's record files (`analysis/record_files.py`: `*_test.jsonl` and `*_drift.jsonl` in `results/gates/` and
`results/jev/`), from the stored records, never from a counter that could drift, with the gate spend module's billed-token
reader and prices and the Jev routers' price. Stage 7's cap and its records are untouched.

- **$25 (`CALLS`)** covers every Stage 8 call but the live replay's: the drift resends, G's and G+'s Jev calls on test and
  both gates.
- **$10 (`LIVE`)** covers the liveness calls and the live replay. Their records are in the final evaluation's own folder
  (`world.output`), `LIVE_FILES`, since they are not router or gate records.

**The projection.** Each call ticket adds one line to `PROJECTORS`: a function `(run) -> dollars` still to be spent in its
pool (a resume projects only requests with no stored response). The liveness step, the first to call, adds them all up
(`check_projection`) and refuses (`LedgerRefusal`) when a pool's recorded spend plus its projection is over its cap, before
anything is sent. A step that sends in batches also asks `refusal` before each submission. A spend that can't be known can't
be under a cap, so a record without whole billed token counts, or a Stage 8 file the ledger has no price for, refuses.
"""

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from analysis.final_run import StepRefusal
from analysis.gate_spend import GateSpendError, billed_tokens, jev_dollars, list_dollars
from analysis.record_files import BATCH, FREE, JEV, STAGE_8_FILES, files_in, is_stage_8
from analysis.world import World
from harness.client import load_models
from routers import jev, sonnet_gate

if TYPE_CHECKING:
    from analysis.final_run import Run

CALLS, LIVE = "calls", "live"
CAPS = {CALLS: 25.0, LIVE: 10.0}  # A17 item 18, counted afresh by A19 item 9
TOLERANCE_USD = 1e-9  # a float sum of many small prices should not tip a total that is exactly on a cap
# The final evaluation's own files, in `world.output`, priced as Anthropic calls at list price by the model each record
# requested: the liveness calls' (`analysis/final_liveness.py`). #91 adds the live replay's file here.
LIVE_FILES = ("liveness.jsonl",)


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
    Projector("jev_router", CALLS, None),
    # #85: the Jev gate's drift resends and test calls
    Projector("jev_gate", CALLS, "analysis.final_gates:project"),
    # #86: the Sonnet gate's drift resends and test batch
    Projector("sonnet_gate", CALLS, None),
    # #91: the live replay
    Projector("replay", LIVE, None),
)


def _records(path: Path) -> list[object]:
    """Each complete record of a stored file; a torn last line (no newline) is a write that did not finish."""
    lines = path.read_bytes().decode("utf-8").split("\n")[:-1]
    result = []
    for number, line in enumerate(lines, start=1):
        try:
            result.append(json.loads(line))
        except ValueError as error:
            raise GateSpendError(f"{path.name} line {number} is not JSON") from error
    return result


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
    for name in LIVE_FILES:
        path = output_dir / name
        if path.is_file():
            files += 1
            live += sum(_anthropic_dollars(record, name, models) for record in _records(path))
    return Spend(calls, live, files)


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
        module, _, function = projector.entry.partition(":")
        result.append(
            Projection(projector.name, projector.pool, getattr(importlib.import_module(module), function)(run))
        )
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
