"""Stage 7's cap on gate calls: $25 in total, across both paid gates (PLAN_v4.md §9; Amendment A17, item 18; A6 item 6).

Both paid collectors (`analysis.collect_jev_gate`, `analysis.collect_sonnet_gate`) refuse a submission when the spend
already recorded in `results/gates/` plus the projected cost of what is about to go out is over `STAGE_CAP_USD`. The
Sonnet collector also keeps its per-submission cap (A6 item 6); this one is cumulative, so a misconfigured effort level
can't run away over several runs, and the splits and the gates count together.

Recorded spend is read back from the stored records, never from a counter that could drift:
- the Jev gate's `jev_gate_*.jsonl`: each record's billed input tokens at Jev's price (A15 item 10);
- the Sonnet gate's batch verdicts, `sonnet_gate_{split}.jsonl`: billed tokens at list price less the batch discount,
  since that is what a batch is billed (the cost model ignores the discount, the invoice does not);
- the Sonnet gate's standard-API calls, `sonnet_gate_latency*.jsonl`: billed tokens at list price.
Every split counts, smoke included. A cut-short last line (no newline) is a write that did not finish and is dropped,
as the stores' own reader does; any other line that is not a record with whole, non-negative billed token counts refuses
the count (GateSpendError), since a spend that can't be known can't be under a cap.

A Jev call's cost is projected before it is made, from its body: the bytes at one token per byte, which no tokenizer
beats, plus a fixed overhead, at Jev's price. It is an upper bound, so a projection never lets through a call the cap
would have stopped.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from analysis.collect_jev import RESULTS_DIR
from harness.client import load_models
from routers import jev, sonnet_gate

STAGE_CAP_USD = 25.0
JEV_OVERHEAD_TOKENS = 512  # above the 291 the smoke test measured: a projection errs high
TOLERANCE_USD = 1e-9  # float sums of many small prices should not tip a total that is exactly on the cap


class GateSpendError(ValueError):
    """Stored records whose spend can't be counted. Nothing is sent on the strength of a count that failed."""


@dataclass(frozen=True)
class Spend:
    jev: float
    sonnet_batch: float  # at the batch discount
    sonnet_standard: float
    files: int  # how many gate files were read

    @property
    def total(self) -> float:
        return self.jev + self.sonnet_batch + self.sonnet_standard


def _billed(record: object, name: str, number: int) -> tuple[int, int]:
    usage = record.get("usage") if isinstance(record, dict) else None
    counts = (usage.get("input_tokens"), usage.get("output_tokens")) if isinstance(usage, dict) else (None, None)
    if not all(type(c) is int and c >= 0 for c in counts):
        raise GateSpendError(f"{name} line {number} has no whole, non-negative billed token counts")
    return counts  # type: ignore[return-value]


def _tokens(path: Path) -> list[tuple[int, int]]:
    """Each complete record's billed (input, output) tokens."""
    lines = path.read_bytes().decode("utf-8").split("\n")[:-1]
    result = []
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except ValueError as error:
            raise GateSpendError(f"{path.name} line {number} is not JSON") from error
        result.append(_billed(record, path.name, number))
    return result


def recorded_spend(
    results_dir: Path = RESULTS_DIR,
    *,
    jev_price_per_million: float | None = None,
    sonnet: sonnet_gate.SonnetSettings | None = None,
    batch_discount: float | None = None,
) -> Spend:
    """What the stored gate records say was spent, in dollars. The prices default to the project's settings."""
    if jev_price_per_million is None:
        jev_price_per_million = jev.read_client_settings().price_per_million_input_tokens
    sonnet = sonnet or sonnet_gate.load_settings()
    if batch_discount is None:
        batch_discount = float(load_models()["batch_discount"])
    folder = results_dir / "gates"
    jev_total = batch_total = standard_total = 0.0
    files = 0
    for path in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
        if path.name.startswith("jev_gate_"):
            files += 1
            jev_total += sum(i for i, _ in _tokens(path)) * jev_price_per_million / 1e6
        elif path.name.startswith("sonnet_gate_"):
            files += 1
            list_price = sum((i * sonnet.input_per_million + o * sonnet.output_per_million) / 1e6
                             for i, o in _tokens(path))  # fmt: skip
            if path.name.startswith("sonnet_gate_latency"):
                standard_total += list_price
            else:
                batch_total += list_price * batch_discount
    return Spend(jev_total, batch_total, standard_total, files)


def over_cap(recorded: float, projected: float, cap: float = STAGE_CAP_USD) -> str | None:
    """None while the recorded spend plus the projection is within the cap; otherwise the refusal's words."""
    if recorded + projected <= cap + TOLERANCE_USD:
        return None
    return (f"${recorded:.4f} is recorded in results/gates/ and this submission is projected at ${projected:.4f}: "
            f"${recorded + projected:.2f} is over Stage 7's cap of ${cap:.2f}; nothing was sent")  # fmt: skip


def jev_projected_cost(body_lengths: Sequence[int], price_per_million: float) -> float:
    """An upper bound on the Jev calls' cost: each body's bytes (one token per byte) plus the fixed overhead."""
    return sum(length + JEV_OVERHEAD_TOKENS for length in body_lengths) * price_per_million / 1e6
