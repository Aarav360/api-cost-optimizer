"""What Stage 7's train work measures from the stored records (PLAN_v4.md §9; Amendment A17, items 12, 16, 17, 18, 20.3
and 28; issue #69): D6 and the shift of both gates, counts per config, tokens and cost, latency, the fixed cutoffs'
Youden's J, and the hashes of the stored train records.

Everything here reads stored train records (and train's outcomes, for the cutoffs and the signal check) and calls no gate.
`analysis.gate_train_report` prints it, and writes A17 item 28 from it; `analysis.check_gates_freeze` recomputes it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from analysis import gate_spend
from routers import jev_gate, sonnet_gate

SONNET_FLOOR = 0.01  # A17 item 17: whole-number scores can repeat exactly, so the Sonnet gate's D6 is at least this


class GateTrainError(ValueError):
    """Stored records or measures this module refuses whole."""


@dataclass(frozen=True)
class Difference:
    """The second call of the same answers less the first, on the gate score as a probability."""

    pairs: int  # answers with a score in both calls
    unscored: int  # answers left out because either call was refused (it has no score)
    mean_abs: float
    d6: float  # the mean absolute difference, at least the floor
    shift: float  # the mean signed difference


def difference(first: Mapping[str, float | None], second: Mapping[str, float | None], floor: float = 0.0) -> Difference:
    """D6 and the shift between two calls of the same answers (item 17). A refused call (score None) in either leaves
    its answer out and is counted. Refuses (GateTrainError) calls over different answers, or none with a score in both."""
    if set(first) != set(second):
        raise GateTrainError("the two calls are not over the same answers")
    both = [(a, b) for item in first if (a := first[item]) is not None and (b := second[item]) is not None]
    if not both:
        raise GateTrainError("no answer has a score in both calls")
    mean_abs = sum(abs(b - a) for a, b in both) / len(both)
    return Difference(len(both), len(first) - len(both), mean_abs, max(mean_abs, floor),
                      sum(b - a for a, b in both) / len(both))  # fmt: skip


def jev_noise(records: Sequence[Mapping[str, Any]], settings: jev_gate.JevGateSettings) -> dict[str, Difference]:
    """The Jev gate's noise (item 17), per config: pass 2 less pass 1 over the answers the noise pass re-sent. Refuses
    (GateTrainError) a pass-2 record with no pass 1."""
    first: dict[tuple[str, str], float | None] = {}
    second: dict[tuple[str, str], float | None] = {}
    for r in records:
        key = (str(r["item_id"]), str(r["config"]))
        score = jev_gate.read_response(r["response"], settings).score
        {1: first, 2: second}[int(str(r["pass"]))][key] = score
    if missing := sorted(set(second) - set(first)):
        raise GateTrainError(f"{len(missing)} noise-pass answers have no pass 1, e.g. {missing[0]}")
    configs = sorted({config for _, config in second})
    return {
        config: difference({i: first[(i, c)] for i, c in second if c == config},
                           {i: second[(i, c)] for i, c in second if c == config})
        for config in configs
    }  # fmt: skip


def sonnet_drift(batch: Sequence[Mapping[str, Any]], standard: Sequence[Mapping[str, Any]]) -> dict[str, Difference]:
    """The Sonnet gate's drift (item 17), per config: the standard-API calls' score ÷ 100 less the same answers' train
    batch score, with D6 floored at 0.01. Refuses (GateTrainError) a standard call with no batch verdict."""

    def score(record: Mapping[str, Any]) -> float | None:
        result = record["result"]
        if not isinstance(result, dict) or result.get("type") != "succeeded":
            raise GateTrainError(f"{record['item_id']} {record['config']} stores a result that is not a success")
        return sonnet_gate.parse_response(result["message"]).score

    first = {(str(r["item_id"]), str(r["config"])): score(r) for r in batch}
    second = {(str(r["item_id"]), str(r["config"])): score(r) for r in standard}
    if missing := sorted(set(second) - set(first)):
        raise GateTrainError(f"{len(missing)} latency calls have no batch verdict, e.g. {missing[0]}")
    return {
        config: difference({i: first[(i, c)] for i, c in second if c == config},
                           {i: second[(i, c)] for i, c in second if c == config}, SONNET_FLOOR)
        for config in sorted({c for _, c in second})
    }  # fmt: skip


# ---- counts per config (item 28) ------------------------------------------------------------------------------

CHECKS = ("stop", "schema", "code")  # the free checks, in the order they are named


@dataclass(frozen=True)
class ConfigCounts:
    answers: int  # answers checked
    rejected: int  # of them, those a free check rejected
    failed_first: dict[str, int]  # rejections counted once each, under the check that failed first
    failing: dict[str, int]  # answers failing each check (an answer can fail more than one)
    jev_calls: int  # pass 1 only: the noise pass is a second send of some of these
    jev_refused: int
    jev_cut: int  # calls whose answer or prompt the byte budget cut
    jev_retries: int  # attempts beyond each call's first
    jev_failed_attempts: int
    sonnet_calls: int  # batch verdicts
    sonnet_refused: int  # a refusal, a cap stop or an unparseable result
    sonnet_cap_stops: int  # of them, those that hit the output cap (max_tokens)


def count_configs(
    free: Sequence[Mapping[str, Any]],
    jev: Sequence[Mapping[str, Any]],
    sonnet: Sequence[Mapping[str, Any]],
    configs: Sequence[str],
) -> dict[str, ConfigCounts]:
    """Item 28's counts per config from the stored free-check, Jev gate (either pass) and Sonnet batch records. Refuses
    (GateTrainError) a config with no free-check record."""
    result = {}
    for config in configs:
        mine = [r for r in free if r["config"] == config]
        if not mine:
            raise GateTrainError(f"no free-check record for {config!r}")
        failing = {
            "stop": sum(not r["stop_passed"] for r in mine),
            "schema": sum(bool(r["schema_applied"]) and not r["schema_passed"] for r in mine),
            "code": sum(bool(r["code_applied"]) and not r["code_passed"] for r in mine),
        }
        first = [r["failed_first"] for r in mine if r["failed_first"] is not None]
        calls = [r for r in jev if r["config"] == config and r["pass"] == 1]
        verdicts = [r for r in sonnet if r["config"] == config]
        result[config] = ConfigCounts(
            len(mine), len(first), {c: first.count(c) for c in CHECKS}, failing,
            len(calls), sum(bool(r["refused"]) for r in calls),
            sum(int(str(r["answer_cut_bytes"])) > 0 or int(str(r["prompt_cut_bytes"])) > 0 for r in calls),
            sum(int(str(r["attempts"])) - 1 for r in calls),
            sum(len(r["failed_attempts"]) for r in calls),
            len(verdicts), sum(bool(r["refused"]) for r in verdicts),
            sum(r["stop_reason"] == "max_tokens" for r in verdicts),
        )  # fmt: skip
    return result


def batch_retries(log: Mapping[str, Any], split: str) -> int:
    """The requests of a split's batches that were sent again: those whose result was errored or expired (A6 item 6)."""
    return sum(
        int(b["counts"].get("errored", 0)) + int(b["counts"].get("expired", 0))
        for b in log["batches"] if b["split"] == split
    )  # fmt: skip


# ---- the fixed cutoffs' J (item 12) ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CutoffReport:
    cutoff: float
    j: float  # the share of wrong answers rejected less the share of right ones rejected
    wrong: int
    right: int
    wrong_rejected: int
    right_rejected: int


def cutoff_report(answers: Sequence[tuple[float | None, bool]], cutoff: float) -> CutoffReport:
    """J and the two shares at a cutoff, over `(gate score, answer was right)` pairs; a refused call (None) is rejected.
    Refuses (GateTrainError) answers that are all right or all wrong."""
    wrong = [score for score, right in answers if not right]
    right = [score for score, right in answers if right]
    if not wrong or not right:
        raise GateTrainError(f"J needs right and wrong answers, not {len(right)} right and {len(wrong)} wrong")

    def rejected(scores: list[float | None]) -> int:
        return sum(score is None or score < cutoff for score in scores)

    wrong_rejected, right_rejected = rejected(wrong), rejected(right)
    return CutoffReport(cutoff, wrong_rejected / len(wrong) - right_rejected / len(right), len(wrong), len(right),
                        wrong_rejected, right_rejected)  # fmt: skip


# ---- latency of the Sonnet gate's standard-API calls (item 16) ------------------------------------------------


@dataclass(frozen=True)
class SampleLatency:
    median: float  # seconds
    p90: float
    count: int  # calls these two come from
    first: float  # the first call's time, left out of both
    by_task: dict[str, tuple[float, float, int]]  # task -> median, p90, calls


def _median_p90(seconds: Sequence[float]) -> tuple[float, float]:
    return float(np.median(seconds)), float(np.percentile(seconds, 90))


def sonnet_latency(records: Sequence[Mapping[str, Any]], task_of: Mapping[str, str]) -> SampleLatency:
    """The latency sample's median and p90, the call flagged `first_call` left out (item 16); p90 is numpy's percentile
    with linear interpolation. Refuses (GateTrainError) a sample without exactly one flagged first call, or with
    nothing left after it."""
    flagged = [r for r in records if r["first_call"]]
    if len(flagged) != 1:
        raise GateTrainError(f"the sample must flag exactly one first call, and flags {len(flagged)}")
    kept = [(task_of[str(r["item_id"])], float(str(r["seconds"]))) for r in records if not r["first_call"]]
    if not kept:
        raise GateTrainError("no call is left once the first call is left out")
    median, p90 = _median_p90([s for _, s in kept])
    by_task = {}
    for task in sorted({t for t, _ in kept}):
        times = [s for t, s in kept if t == task]
        by_task[task] = (*_median_p90(times), len(times))
    return SampleLatency(median, p90, len(kept), float(str(flagged[0]["seconds"])), by_task)


# ---- tokens and cost (item 28) --------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainSpend:
    jev_input_tokens: int
    jev: float  # dollars, at Jev's price
    batch_input_tokens: int
    batch_output_tokens: int
    batch_list: float  # the batch verdicts at list price (what a trace is charged)
    batch: float  # and as billed, at the batch discount
    latency_input_tokens: int
    latency_output_tokens: int
    latency: float  # the standard-API calls, billed at list price

    @property
    def total(self) -> float:
        return self.jev + self.batch + self.latency


def train_spend(
    *,
    jev: Sequence[Mapping[str, Any]],
    batch: Sequence[Mapping[str, Any]],
    latency: Sequence[Mapping[str, Any]],
    jev_price: float,
    input_price: float,
    output_price: float,
    batch_discount: float,
) -> TrainSpend:
    """The train calls' tokens and cost from the stored records' usage: Jev (both passes) at its price per million input
    tokens, the batch verdicts at list price and as billed (less the discount), the latency calls at list price. The
    tokens are read and priced by the $25 cap's own code (`analysis.gate_spend`), so the report and the cap agree;
    refuses (GateSpendError) what the cap refuses."""
    jev_in, _ = gate_spend.billed_tokens(jev, "a Jev gate record")
    batch_in, batch_out = gate_spend.billed_tokens(batch, "a Sonnet batch verdict")
    latency_in, latency_out = gate_spend.billed_tokens(latency, "a Sonnet latency call")
    batch_list = gate_spend.list_dollars(batch_in, batch_out, input_price, output_price)
    return TrainSpend(
        jev_in, gate_spend.jev_dollars(jev_in, jev_price), batch_in, batch_out, batch_list, batch_list * batch_discount,
        latency_in, latency_out, gate_spend.list_dollars(latency_in, latency_out, input_price, output_price),
    )  # fmt: skip
