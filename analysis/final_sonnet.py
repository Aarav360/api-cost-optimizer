"""The Sonnet gate on test, with its drift check (PLAN_v4.md Amendment A19, items 2, 5, 7, 9, 10 and 13; A17 items 13, 14, 17 and 26;
A15 item 17; issue #77, #86). The sibling of `analysis/final_gates.py`, whose pieces it reuses.

Two steps of the final evaluation, in A19 item 2's order:

- **`drift_step`** (`drift_sonnet_gate`, before any call on test): the Sonnet gate's 80 answers, the ones Stage 7 sent through the
  standard API for its latency sample, are sent once more through the standard API, into `results/gates/sonnet_gate_drift.jsonl`
  and nowhere else. **D8 is measured against the same answers' train batch scores**, as D6 is (the standard API's first
  sample, `sonnet_gate_latency.jsonl`, less the batch score), so D6 and D8 both compare the standard API with the Batches API and a
  fixed gap between the two shows in both and can't trip the shift rule alone (A17 item 26.2). Both go through
  `analysis/gate_train.sonnet_drift`, so D6 is floored at 0.01 (whole-number scores can repeat exactly) and so is D8, which
  changes no verdict: a floored D8 is at most 0.01 when it matters and 3 x D6 is at least 0.03. `analysis/drift_rule.py` (shared
  with the Jev gate and the Jev routers) applies A15 item 17's rule: a model ID that isn't the pin, D8 above 3 x D6, or a shift
  larger than D6 in size refuses (`DriftRefusal`) with every call on test still unmade; otherwise it reports D6, D8, the shifts
  and the decisions that flip at each frozen Sonnet cutoff and goes on. Each resend must hash to the request Stage 7 stored for
  that answer, so the frozen code is what is sent. A refusal has no substitute: the same model ID can't be replaced, so the way
  on is a new amendment (A17 item 26.3).
- **`calls_step`** (`calls_sonnet_gate`): the gate on `haiku`'s, `sonnet`'s and `opus_low`'s answers that passed the free
  checks, through the Batches API, into `sonnet_gate_test.jsonl`, under the ledger. The safeguards are A6 item 6's and A17 item
  14's, and the code is the collector's own (`analysis/collect_sonnet_gate.py`, given Stage 8's files): a batch's ID is written
  the moment it is created; a rerun sends only the requests with no stored verdict that are in no unfinished batch, so the only
  ones resent are those that came back errored or expired; every submission is refused over the $25 cap before anything is
  sent; results are matched to requests by ID and never by order. A refusal, a cap stop or an unparseable result is a
  rejection charged its billed tokens, stored as the gate's own parser reads it. **The batch's ID and creation time are in the
  step's record** (A19 item 10's order evidence), for every batch the run made, read back from the batch log in the output folder
  (Stage 7's `sonnet_batches.json` is Stage 7's: no Stage 8 entry goes in it). A stored verdict from a model other than the pin
  refuses (`ModelRefusal`) before grading.

**The projection** (`project`, the ledger's `sonnet_gate` line) is dev's cost per verdict, from the committed dev record
(`analysis/gates_dev_record.json`: the batch's list-price dollars over its verdicts), times 2 (`PROJECTION_FACTOR`), for the
drift resends at list price and the test batch at the batch discount. It is not a worst case (every call at its whole
`max_tokens` would be many times the cap), it is dev's measured cost with a margin; each submission is refused before it is sent
by what is recorded plus what that submission projects.

**What the gate sees** is `GateInput`: the request, the answer's text and its stop reason, from the handle's two reads.
`analysis/check_final_sonnet.py` holds that by reading this file and by running the steps on a made-up world.
**Nothing is shown before the tables**: the steps say counts of requests sent and stored, never a refusal or a cap stop; those come
with the tables step (`final_gates.gate_data_record`).
"""

import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from analysis import collect_sonnet_gate, drift_rule, gate_train
from analysis.collect_gates import GateInput, read_gate_records
from analysis.collect_jev import append_record, trim_torn_line
from analysis.collect_sonnet_gate import (
    LATENCY_FIELDS,
    MAX_BATCH_BYTES,
    MAX_BATCH_REQUESTS,
    STANDARD,
    STANDARD_PASS,
    CapExceeded,
    Job,
    Stage8Files,
    build_record,
)
from analysis.drift import DriftRefusal
from analysis.final_gates import (
    DRIFT,
    SONNET_NAME,
    TEST,
    FinalGatesError,
    cutoff_labels,
    drift_inputs,
    gate_file,
    gate_flips,
    label,
    named,
    read_free_test,
    read_sonnet_test,
    stamp,
    stored_free,
    test_inputs,
)
from analysis.final_ledger import LedgerRefusal, Meter, cap_words, check_cap
from analysis.final_run import ItemFailure, Run, StepRefusal, StepResult
from analysis.gate_spend import list_dollars
from analysis.live_calls import client_of
from analysis.router_names import SONNET_GATE
from harness.client import PROJECT_ROOT, load_models
from routers import sonnet_gate

DRIFT_CHECK = "drift_sonnet_gate"  # the drift check's name: the step's, and the refusal's
BATCH_LOG = (
    "sonnet_batches_test.json"  # in the output folder: Stage 8's batch log, apart from Stage 7's `sonnet_batches.json`
)
DEV_RECORD = PROJECT_ROOT / "analysis" / "gates_dev_record.json"
PROJECTION_FACTOR = 2.0  # the projection is dev's measured cost per verdict times this
BATCH_LIMITS: dict[str, Any] = {
    "max_requests": MAX_BATCH_REQUESTS,
    "max_bytes": MAX_BATCH_BYTES,
}  # the Batches API's, as the collector has them
MAX_ROUNDS = 5  # batches a go sends for one set of verdicts: the first, then the errored and expired ones sent again
POLL_S = 30.0  # between looks at an unfinished batch
clock = time.perf_counter  # names a check can replace, as `collect_sonnet_gate.run_latency` has its `clock`
sleep = time.sleep


class FinalSonnetError(FinalGatesError):
    """A stored file, a world or an answer these steps refuse. Nothing is sent."""


class ModelRefusal(StepRefusal):
    """A stored test verdict that came back from a model other than the pin. Its message names model IDs only."""


# ---- files ---------------------------------------------------------------------------------------------------------


def drift_file(results: Path) -> Path:
    return gate_file(SONNET_NAME, DRIFT, results)


def test_file(results: Path) -> Path:
    return gate_file(SONNET_NAME, TEST, results)


def batch_log(run: Run) -> Path:
    return run.world.output / BATCH_LOG


def read_drift(results: Path) -> list[dict]:
    """The stored resends (standard API, pass 2) in the order stored: the one reader of them."""
    return read_gate_records(drift_file(results), LATENCY_FIELDS, DRIFT, SONNET_GATE, passes=(STANDARD_PASS,))


# ---- the projection and the cap --------------------------------------------------------------------------------------


def dev_dollars_per_check() -> float:
    """Dev's list-price dollars per Sonnet verdict: the dev batch's list cost over its verdicts, from the committed record."""
    record = json.loads(DEV_RECORD.read_text(encoding="utf-8"))
    verdicts = sum(int(c["sonnet_calls"]) for c in record["counts"].values())
    if verdicts < 1:
        raise FinalSonnetError("the dev record holds no Sonnet verdicts to project from")
    return float(record["spend"]["batch_list"]) / verdicts


def dollars_per_check(*, batch: bool) -> float:
    """What a check is projected to cost: dev's per verdict at list price, times the margin, and at the batch discount for a
    batch request."""
    rate = dev_dollars_per_check() * PROJECTION_FACTOR
    return rate * float(load_models()["batch_discount"]) if batch else rate


def project(run: Run) -> float:
    """The Sonnet gate's projected spend for the ledger's whole-run projection (the calls pool): the drift resends with no
    stored record at the standard API's price and a request for each gated test answer with no stored verdict at the batch's,
    leaving out those a stored free check rejected. Before the free checks have run it assumes every gated answer passes, so it
    never lets through what the cap would stop. Without Stage 7's latency sample there is nothing to resend (the drift step
    itself refuses)."""
    results = run.world.results
    settings = sonnet_gate.load_settings()
    done = {r["custom_id"] for r in read_drift(results)}
    resends = sum(r["custom_id"] not in done for r in collect_sonnet_gate.read_latency_records("train", results))
    rejected = {(r["item_id"], r["config"]) for r in read_free_test(results) if not r["passed"]}
    stored = {(r["item_id"], r["config"]) for r in read_sonnet_test(results)}
    wanted = [
        i for i in test_inputs(run.unlocked())
        if i.config in settings.gated_configs and (i.item_id, i.config) not in rejected | stored
    ]  # fmt: skip
    return resends * dollars_per_check(batch=False) + len(wanted) * dollars_per_check(batch=True)


# ---- the drift check -------------------------------------------------------------------------------------------------


def drift_jobs(inputs: Sequence[GateInput], settings: sonnet_gate.SonnetSettings) -> list[Job]:
    """The frozen gate request for each resent answer, built before any is sent."""
    jobs = []
    for item in inputs:
        params = sonnet_gate.build_request(settings, item.request, item.answer)
        jobs.append(Job(sonnet_gate.custom_id(settings, item.item_id, item.config), item.item_id, item.config,
                        item.request, item.answer, params, sonnet_gate.request_sha256(params)))  # fmt: skip
    return jobs


def scores_of(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], float | None]:
    """Each record's gate score through the gate's own parser (None for a refused call)."""
    scores = {}
    for r in records:
        result = r["result"]
        if not isinstance(result, dict) or result.get("type") != "succeeded":
            raise FinalSonnetError(f"{r['item_id']} {r['config']} stores a result that is not a success")
        scores[(str(r["item_id"]), str(r["config"]))] = sonnet_gate.parse_response(result["message"]).score
    return scores


def model_ids(records: Sequence[Mapping[str, Any]]) -> list[object]:
    """Every model ID a set of calls came back with: the stored field and the response's own (`drift_rule.returned_models`)."""
    return drift_rule.returned_models(
        records, lambda r: r["result"]["message"] if isinstance(r["result"], dict) else None
    )


def drift_report(
    batch: Sequence[Mapping[str, Any]],
    latency: Sequence[Mapping[str, Any]],
    resent: Sequence[Mapping[str, Any]],
    settings: sonnet_gate.SonnetSettings,
    labels: Mapping[float, Sequence[str]],
) -> drift_rule.DriftReport:
    """The rule on Stage 7's train batch verdicts (`batch`), its standard-API calls of the sample (`latency`) and the resend
    (`resent`): D6 from the first two and D8 and the shift from the batch scores and the resend, per config, through
    `gate_train.sonnet_drift` (standard API less batch, as the plan words it), and the flips at each cutoff. Raises
    `DriftRefusal` on the rule."""
    sample = {(str(r["item_id"]), str(r["config"])) for r in latency}
    mine = [r for r in batch if (str(r["item_id"]), str(r["config"])) in sample]
    first = scores_of(mine)
    resend = scores_of(resent)
    if set(resend) != sample or set(first) != sample:
        raise FinalSonnetError("the resend and Stage 7's batch verdicts are not both over the latency sample")
    measures = drift_rule.measures_of(gate_train.sonnet_drift(mine, latency), gate_train.sonnet_drift(mine, resent))
    return drift_rule.apply_rule(DRIFT_CHECK, pin=sonnet_gate.load_settings().model, model_ids=model_ids(resent),
                                 measures=measures, flips=gate_flips(first, resend, labels))  # fmt: skip


def send_drift(run: Run, client: Any, settings: sonnet_gate.SonnetSettings, todo: Sequence[Job]) -> None:
    """Send each resend through the standard API, one at a time, and store its record before the next (pass 2, split drift,
    the latency record's fields). The first call of a go is flagged, as Stage 7's latency calls are."""
    path = drift_file(run.world.results)
    meter = Meter.of(run)
    for number, job in enumerate(todo, start=1):
        item = GateInput(job.item_id, job.config, job.request, job.answer, "end_turn")
        try:
            meter.before(dollars_per_check(batch=False))  # the pool's cap, with what this step has been billed so far
            start = clock()
            outcome = sonnet_gate.check_answer(client, settings, job.request, job.answer)
            seconds = clock() - start
            usage = outcome.response["usage"]
            meter.paid(list_dollars(usage["input_tokens"], usage["output_tokens"], settings.input_per_million,
                                    settings.output_per_million))  # fmt: skip
            result = {"type": "succeeded", "message": outcome.response}
            record = build_record(job, DRIFT, STANDARD_PASS, result, outcome.verdict, None, lambda: stamp(run))
            append_record(path, record | {"api": STANDARD, "seconds": seconds, "first_call": number == 1,
                                          "request_id": outcome.request_id})  # fmt: skip
        except Exception as error:
            raise ItemFailure(named(item)) from error


def drift_step(run: Run) -> StepResult:
    """A15 item 17 for the Sonnet gate (A17 item 26.2): the 80 answers of Stage 7's latency sample resent once through the
    standard API into `sonnet_gate_drift.jsonl`, D8 measured against their train batch scores. Stops the run on the shared rule
    with nothing called on test. Says train's numbers only."""
    results = run.world.results
    settings = sonnet_gate.load_settings()
    batch = collect_sonnet_gate.read_sonnet_records("train", results)
    latency = collect_sonnet_gate.read_latency_records("train", results)
    if not latency:
        raise FinalSonnetError("Stage 7 stored no standard-API calls for the Sonnet gate: there is nothing to resend")
    sample = [(str(r["item_id"]), str(r["config"])) for r in latency]
    jobs = drift_jobs(drift_inputs(run, sample), settings)
    first = {(str(r["item_id"]), str(r["config"])): r["request_sha256"] for r in latency}
    if differing := [label(j.item_id, j.config) for j in jobs
                     if j.sha256 != first[(j.item_id, j.config)]]:  # fmt: skip
        raise DriftRefusal(DRIFT_CHECK, f"the frozen gate request is different from the one Stage 7 stored for "
                                        f"{len(differing)} of the {len(jobs)} resent answers, e.g. {differing[0]}; "
                                        f"nothing was called on test")  # fmt: skip
    trim_torn_line(drift_file(results))
    stored = {r["custom_id"] for r in read_drift(results)}
    todo = [j for j in jobs if j.custom_id not in stored]
    check_cap(run, len(todo) * dollars_per_check(batch=False))
    if todo:
        send_drift(run, client_of(run, StepRefusal), settings, todo)
    resent = read_drift(results)
    report = drift_report(batch, latency, resent, settings, cutoff_labels(SONNET_GATE))
    run.say(f"{DRIFT_CHECK}: {len(jobs)} answers resent ({len(todo)} calls sent, {len(stored)} already stored); {report.said()}; "
            f"decisions that flip at the frozen cutoffs {[len(f.items) for f in report.flips]}")  # fmt: skip
    unscored = {config: d.unscored for config, d in gate_train.sonnet_drift(batch, resent).items()}
    return StepResult(requests={"sent": len(todo), "stored": len(stored)},
                      detail={**report.record(), "answers": len(jobs), "unscored": unscored})  # fmt: skip


# ---- the batch on test -----------------------------------------------------------------------------------------------


def batches_of(run: Run) -> list[dict[str, Any]]:
    """Every test batch the batch log holds, in order: its ID and creation time (A19 item 10's order evidence), the count of
    requests it held and, once ended, how many of its results came back as each kind."""
    log = collect_sonnet_gate.load_log(batch_log(run))["batches"]
    return [
        {"id": b["id"], "created_at": b["created_at"], "request_count": b["request_count"], "counts": b.get("counts", {})}
        for b in log if b["split"] == TEST
    ]  # fmt: skip


def check_models(verdicts: Sequence[Mapping[str, Any]], pin: str) -> None:
    """Refuse (ModelRefusal) a stored verdict that a model other than the pin gave."""
    if wrong := sorted({str(m) for m in model_ids(verdicts) if m != pin}):
        raise ModelRefusal(f"a test verdict came back from {', '.join(wrong)} and not the pin {pin}")


def calls_step(run: Run) -> StepResult:
    """The Sonnet gate on every `haiku`, `sonnet` and `opus_low` answer that passed the free checks, through the Batches API
    with the collector's safeguards. Needs every answer's free-check record first. Submits what has no stored verdict and is in
    no unfinished batch, waits for each batch to end, and sends again only what came back errored or expired; every
    submission is refused over the $25 cap first. A resume finishes an unfinished batch before it sends anything."""
    inputs = test_inputs(run.unlocked())
    results = run.world.results
    settings = sonnet_gate.load_settings()
    free = read_free_test(results)
    stored_free(inputs, results)  # refuses a free-check record for an answer that isn't one of the handle's
    jobs = collect_sonnet_gate.build_jobs(inputs, free, settings)  # refuses a gated answer with no free-check record
    client = client_of(run, StepRefusal)
    stage8 = Stage8Files(run.unlocked(), test_file(results), batch_log(run), cap_words(run))
    already = len(read_sonnet_test(results))
    sent = rounds = 0
    new_batches: list[str] = []
    while True:
        collect_sonnet_gate.wait(jobs, split=TEST, client=client, results_dir=results, poll_s=POLL_S, sleep=sleep,
                                 now=lambda: stamp(run), stage8=stage8)  # fmt: skip
        pending = collect_sonnet_gate.pending_jobs(jobs, TEST, results, stage8=stage8)[0]
        if not pending:
            break
        if rounds == MAX_ROUNDS:
            raise StepRefusal(f"{len(pending)} requests are still without a verdict after {rounds} batches")
        try:
            submitted = collect_sonnet_gate.submit(jobs, split=TEST, client=client, results_dir=results,
                                                   cost_per_check=dollars_per_check(batch=True), max_usd=math.inf,
                                                   now=lambda: stamp(run), stage8=stage8, **BATCH_LIMITS)  # fmt: skip
        except CapExceeded as error:
            raise LedgerRefusal(str(error)) from error
        sent += submitted.sent
        new_batches += submitted.batches
        rounds += 1
    verdicts = read_sonnet_test(results)
    check_models(verdicts, settings.model)
    run.say(f"Sonnet gate: {sent} requests sent in {len(new_batches)} batches, {already} already stored; "
            f"${sent * dollars_per_check(batch=True):.4f} was projected for them")  # fmt: skip
    batches = batches_of(run)
    return StepResult(requests={"sent": sent, "stored": already},
                      detail={"batches": batches, "batch_resent": sum(
                          int(b["counts"].get("errored", 0)) + int(b["counts"].get("expired", 0)) for b in batches),
                              "model_ids": sorted({str(m) for m in model_ids(verdicts)})})  # fmt: skip
