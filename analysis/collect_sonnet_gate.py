"""Collect the Sonnet gate's verdicts (PLAN_v4.md §9; Amendment A17, items 7, 14, 16 and 18; A6 item 6's safeguards).

Run from the project root:
  uv run python -m analysis.collect_sonnet_gate batch --split train|dev --cost-per-check 0.002 --max-usd 25
  uv run python -m analysis.collect_sonnet_gate batch --split smoke --smoke-file made_up.json ...
  uv run python -m analysis.collect_sonnet_gate latency --sample-file sample.json --cost-per-check 0.005 --max-usd 5

The key is the SDK's own business (it reads `ANTHROPIC_API_KEY` from the environment, and `harness.client` from the
project's `.env`); nothing here reads, holds or writes it.

What is sent. The gate judges `haiku`'s, `sonnet`'s and `opus_low`'s answers that passed the free checks, and nothing
else (item 7): `build_jobs` reads the free-check records and sends no answer they didn't pass, and none of `opus`'s.
A job's request ID is `routers.sonnet_gate.custom_id`, which refuses one outside the API's limits.

The batch safeguards (A6 item 6, A1 item 5), each with its check:
- A batch's ID is written to `results/gates/sonnet_batches.json` the moment the batch is created, before the next batch
  or anything else, so no stop from then on can lead to paying for it twice.
- A rerun submits only the requests with no stored verdict that are in no unfinished batch. So a request that was
  stored, or is in flight, is never sent twice, and the only ones sent again are those whose result was errored or
  expired. Any other kind of result that is not a success (canceled, which nothing here does) stops the run.
- Results are matched to jobs by request ID, never by order; a result for an ID the batch never held stops the run.
- Submission is refused, before anything is sent, if the projected cost (requests to send times a cost per check given
  on the command line, measured by the smoke test) is over the cap.

Storage is JSON Lines, uncommitted: `sonnet_gate_{split}.jsonl` holds one record per verdict, with the keys every gate
record has (`GATE_KEYS`) and `SONNET_FIELDS`: the request ID, the batch ID, the request's sha256, the whole batch
result, the model ID, usage, stop reason, refusal flag and collection time. A refused call is a verdict and is stored.
`sonnet_gate_latency.jsonl` (`sonnet_gate_latency_smoke.jsonl` for made-up answers) holds the standard-API calls, the
same fields as pass 2 with the call's api, its time on a monotonic clock, whether it was the run's first call (left out
of the latency) and the request ID. The gate-score table's Sonnet column is `paid_scores`, read through
`routers.sonnet_gate.parse_response`, the gate's own parser.
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.collect_gates import (
    GATE_KEYS,
    GateCollectError,
    GateInput,
    gate_inputs,
    gate_path,
    read_free_records,
    read_gate_records,
    read_smoke_inputs,
)
from analysis.collect_jev import RESULTS_DIR, append_record, now_local, trim_torn_line
from analysis.gate_table import PaidScore
from analysis.router_names import SONNET_GATE
from routers import sonnet_gate
from routers.request import Request
from routers.sonnet_gate import SonnetSettings

MAX_BATCH_REQUESTS = 100_000  # the Batches API's limits: requests, and bytes (under its 256 MB)
MAX_BATCH_BYTES = 200_000_000
RESUBMIT = ("errored", "expired")  # the only results whose request is sent again
SONNET_FIELDS = (
    "custom_id", "batch_id", "request_sha256", "result", "model", "usage", "stop_reason", "refused", "collected_at",
)  # fmt: skip
RECORD_FIELDS = (*GATE_KEYS, *SONNET_FIELDS)
LATENCY_FIELDS = (*RECORD_FIELDS, "api", "seconds", "first_call", "request_id")
BATCH_PASS, STANDARD_PASS = 1, 2  # the batch verdict is pass 1; the standard-API latency call is pass 2
STANDARD = "standard"


class SonnetCollectError(GateCollectError):
    """A collection or a stored file that is refused or stopped. What was stored before it stays valid."""


class CapExceeded(SonnetCollectError):
    """The projected cost is over the cap. Nothing was sent."""


@dataclass(frozen=True)
class Job:
    custom_id: str
    item_id: str
    config: str
    request: Request
    answer: str
    params: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class Submitted:
    sent: int  # requests put into new batches
    batches: tuple[str, ...]  # the new batches' IDs, in order
    in_flight: int  # requests left alone because their batch is unfinished
    stored: int  # requests left alone because their verdict is stored


# ---- the jobs -------------------------------------------------------------------------------------------------


def build_jobs(
    inputs: Sequence[GateInput], free_records: Sequence[Mapping[str, Any]], settings: SonnetSettings
) -> list[Job]:
    """The Sonnet gate's requests for `inputs`: the gated configs' answers that passed the free checks, in order.
    Refuses (SonnetCollectError) a gated answer with no free-check record, since the free-check file is the filter,
    and a repeated answer or request ID; and (ValueError, from the gate) a request ID outside the API's limits."""
    passed = {(r["item_id"], r["config"]): bool(r["passed"]) for r in free_records}
    jobs: list[Job] = []
    for item in inputs:
        if item.config not in settings.gated_configs:
            continue
        if (item.item_id, item.config) not in passed:
            raise SonnetCollectError(
                f"{item.item_id} {item.config} has no free-check record; run the free checks first"
            )
        if not passed[(item.item_id, item.config)]:
            continue
        params = sonnet_gate.build_request(settings, item.request, item.answer)
        jobs.append(Job(sonnet_gate.custom_id(settings, item.item_id, item.config), item.item_id, item.config,
                        item.request, item.answer, params, sonnet_gate.request_sha256(params)))  # fmt: skip
    ids = [job.custom_id for job in jobs]
    if len(set(ids)) != len(ids):
        raise SonnetCollectError("two answers have the same request ID")
    return jobs


# ---- storage --------------------------------------------------------------------------------------------------


def batch_log_path(results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / "gates" / "sonnet_batches.json"


def latency_path(split: str, results_dir: Path = RESULTS_DIR) -> Path:
    if split not in ("train", "smoke"):
        raise SonnetCollectError(
            f"the latency sample is train answers, or made-up ones for a smoke test, not {split!r}"
        )
    return (
        results_dir / "gates" / ("sonnet_gate_latency.jsonl" if split == "train" else "sonnet_gate_latency_smoke.jsonl")
    )


def load_log(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {"batches": []}
    try:
        log = json.loads(path.read_text(encoding="utf-8"))
        batches = log["batches"]
        assert isinstance(batches, list) and all(
            {"id", "split", "custom_ids", "ingested_at"} <= set(b) for b in batches
        )
    except (ValueError, KeyError, TypeError, AssertionError) as error:
        raise SonnetCollectError(f"{path.name} is not a batch log: {error!r}") from error
    return {"batches": batches}


def save_log(path: Path, log: Mapping[str, Any]) -> None:
    """Written whole and swapped in, so a stop never leaves half a log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(log, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_sonnet_records(split: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    """A split's stored batch verdicts (pass 1), or a refusal (GateCollectError) if the file is not a clean one."""
    return read_gate_records(gate_path("sonnet_gate", split, results_dir), RECORD_FIELDS, split, SONNET_GATE,
                             passes=(BATCH_PASS,))  # fmt: skip


def read_latency_records(split: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    return read_gate_records(latency_path(split, results_dir), LATENCY_FIELDS, split, SONNET_GATE,
                             passes=(STANDARD_PASS,))  # fmt: skip


def _record(job: Job, split: str, number: int, result: dict, verdict: sonnet_gate.Verdict, batch_id: str | None,
            now: Callable[[], str]) -> dict:  # fmt: skip
    keys = {"item_id": job.item_id, "config": job.config, "run": 0, "split": split, "gate": SONNET_GATE, "pass": number}
    return keys | {
        "custom_id": job.custom_id, "batch_id": batch_id, "request_sha256": job.sha256, "result": result,
        "model": verdict.model, "usage": {"input_tokens": verdict.input_tokens, "output_tokens": verdict.output_tokens},
        "stop_reason": result["message"].get("stop_reason"), "refused": verdict.refused, "collected_at": now(),
    }  # fmt: skip


# ---- batches --------------------------------------------------------------------------------------------------


def _chunks(jobs: Sequence[Job], max_requests: int, max_bytes: int) -> list[list[Job]]:
    chunks: list[list[Job]] = []
    size = 0
    for job in jobs:
        weight = len(json.dumps({"custom_id": job.custom_id, "params": job.params}).encode("utf-8"))
        if weight > max_bytes:
            raise SonnetCollectError(f"{job.custom_id} alone is {weight:,} bytes, over a batch's {max_bytes:,}")
        if not chunks or len(chunks[-1]) >= max_requests or size + weight > max_bytes:
            chunks.append([])
            size = 0
        chunks[-1].append(job)
        size += weight
    return chunks


def pending_jobs(jobs: Sequence[Job], split: str, results_dir: Path = RESULTS_DIR) -> tuple[list[Job], int, int]:
    """The jobs to send, how many are left alone for being in an unfinished batch, and how many for being stored."""
    in_flight = {c for b in load_log(batch_log_path(results_dir))["batches"] if b["ingested_at"] is None
                 for c in b["custom_ids"]}  # fmt: skip
    stored = {r["custom_id"] for r in read_sonnet_records(split, results_dir)}
    pending = [j for j in jobs if j.custom_id not in in_flight and j.custom_id not in stored]
    return pending, sum(j.custom_id in in_flight for j in jobs), sum(j.custom_id in stored for j in jobs)


def submit(
    jobs: Sequence[Job],
    *,
    split: str,
    client: Any,
    cost_per_check: float,
    max_usd: float,
    results_dir: Path = RESULTS_DIR,
    max_requests: int = MAX_BATCH_REQUESTS,
    max_bytes: int = MAX_BATCH_BYTES,
    now: Callable[[], str] = now_local,
) -> Submitted:
    """Send the jobs with no stored verdict and no unfinished batch, in as few batches as the API's limits allow.

    Refuses (CapExceeded) before sending anything if `cost_per_check` times the requests to send is over `max_usd`.
    Each batch's ID is written to the batch log as soon as `create` returns."""
    if not (cost_per_check >= 0 and max_usd >= 0):
        raise SonnetCollectError("the cost per check and the cap must be given, and can't be negative")
    trim_torn_line(gate_path("sonnet_gate", split, results_dir))
    pending, in_flight, stored = pending_jobs(jobs, split, results_dir)
    projected = cost_per_check * len(pending)
    if projected > max_usd:
        raise CapExceeded(f"{len(pending)} requests at ${cost_per_check:.4f} is a projected ${projected:.2f}, over the "
                          f"cap of ${max_usd:.2f}; nothing was sent")  # fmt: skip
    path = batch_log_path(results_dir)
    created: list[str] = []
    for chunk in _chunks(pending, max_requests, max_bytes):
        batch = client.messages.batches.create(requests=[{"custom_id": j.custom_id, "params": j.params} for j in chunk])
        log = load_log(path)
        log["batches"].append({
            "id": batch.id, "split": split, "created_at": batch.created_at.isoformat(), "request_count": len(chunk),
            "custom_ids": [j.custom_id for j in chunk], "ingested_at": None,
        })  # fmt: skip
        save_log(path, log)
        created.append(batch.id)
    return Submitted(len(pending), tuple(created), in_flight, stored)


def ingest(
    entry: dict[str, Any],
    jobs: Sequence[Job],
    *,
    client: Any,
    results_dir: Path = RESULTS_DIR,
    now: Callable[[], str] = now_local,
) -> dict[str, int]:
    """Store an ended batch's succeeded results, matched to jobs by request ID. Returns the count of each kind of
    result. A result for an ID the batch didn't hold, or a result that is neither a success nor errored or expired,
    stops the run (SonnetCollectError); verdicts stored before it stay."""
    split = entry["split"]
    by_id = {j.custom_id: j for j in jobs}
    held = set(entry["custom_ids"])
    path = gate_path("sonnet_gate", split, results_dir)
    trim_torn_line(path)
    stored = {r["custom_id"] for r in read_sonnet_records(split, results_dir)}
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for item in client.messages.batches.results(entry["id"]):
        cid = item.custom_id
        if cid not in held or cid not in by_id or cid in seen:
            raise SonnetCollectError(f"batch {entry['id']} returned a result for {cid!r}, which it was not sent or "
                                     f"already returned")  # fmt: skip
        seen.add(cid)
        kind = item.result.type
        counts[kind] = counts.get(kind, 0) + 1
        if kind in RESUBMIT:
            continue
        if kind != "succeeded":
            raise SonnetCollectError(f"batch {entry['id']}: {cid!r} came back {kind!r}; nothing here cancels a request")
        if cid in stored:
            continue
        result = item.result.to_dict(mode="json")
        try:
            verdict = sonnet_gate.parse_response(result["message"])
        except (KeyError, TypeError, ValueError) as error:
            raise SonnetCollectError(f"batch {entry['id']}: {cid!r} has no billed tokens ({error!r})") from error
        append_record(path, _record(by_id[cid], split, BATCH_PASS, result, verdict, entry["id"], now))
    entry.update(ingested_at=now(), counts=counts, missing=len(held - seen))
    return counts


def wait(
    jobs: Sequence[Job],
    *,
    split: str,
    client: Any,
    results_dir: Path = RESULTS_DIR,
    poll_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], str] = now_local,
) -> None:
    """Poll the split's unfinished batches until each has ended, storing each as it ends."""
    path = batch_log_path(results_dir)
    while True:
        log = load_log(path)
        open_entries = [b for b in log["batches"] if b["split"] == split and b["ingested_at"] is None]
        if not open_entries:
            return
        for entry in open_entries:
            if client.messages.batches.retrieve(entry["id"]).processing_status == "ended":
                ingest(entry, jobs, client=client, results_dir=results_dir, now=now)
                save_log(path, log)
        if any(b["ingested_at"] is None for b in log["batches"] if b["split"] == split):
            sleep(poll_s)


# ---- the standard-API latency calls ---------------------------------------------------------------------------


def run_latency(
    jobs: Sequence[Job],
    *,
    split: str,
    client: Any,
    settings: SonnetSettings,
    cost_per_check: float,
    max_usd: float,
    results_dir: Path = RESULTS_DIR,
    clock: Callable[[], float] = time.perf_counter,
    now: Callable[[], str] = now_local,
) -> int:
    """Send each job through the standard API, one call at a time, timing it on `clock` (a monotonic clock), and store
    each call as it is made. The first call of the run is flagged `first_call`; the latency leaves it out. Calls
    already stored are skipped. Refuses (CapExceeded) over the cap, like a batch. An API error stops the run and is no
    verdict. Returns the number of calls made."""
    path = latency_path(split, results_dir)
    trim_torn_line(path)
    stored = {r["custom_id"] for r in read_latency_records(split, results_dir)}
    todo = [j for j in jobs if j.custom_id not in stored]
    if cost_per_check * len(todo) > max_usd:
        raise CapExceeded(f"{len(todo)} calls at ${cost_per_check:.4f} is over the cap of ${max_usd:.2f}; none made")
    for number, job in enumerate(todo, start=1):
        start = clock()
        outcome = sonnet_gate.check_answer(client, settings, job.request, job.answer)
        seconds = clock() - start
        result = {"type": "succeeded", "message": outcome.response}
        record = _record(job, split, STANDARD_PASS, result, outcome.verdict, None, now)
        append_record(path, record | {"api": STANDARD, "seconds": seconds, "first_call": number == 1,
                                      "request_id": outcome.request_id})  # fmt: skip
    return len(todo)


# ---- the gate-score table's Sonnet column ---------------------------------------------------------------------


def paid_scores(
    split: str, settings: SonnetSettings, results_dir: Path = RESULTS_DIR
) -> dict[tuple[str, str], PaidScore]:
    """The Sonnet column of the gate-score table, from the stored batch verdicts: each record's result read again
    through the gate's own parser, its overhead at list price, its refusal flag. Refuses (SonnetCollectError) a record
    whose stored flag, usage or request ID disagrees with what the parser reads, or whose result is no success."""
    scores: dict[tuple[str, str], PaidScore] = {}
    for record in read_sonnet_records(split, results_dir):
        name = f"{record['item_id']} {record['config']}"
        result = record["result"]
        if not isinstance(result, dict) or result.get("type") != "succeeded":
            raise SonnetCollectError(f"{name} stores a result that is not a success")
        verdict = sonnet_gate.parse_response(result["message"])
        if record["custom_id"] != sonnet_gate.custom_id(settings, record["item_id"], record["config"]):
            raise SonnetCollectError(f"{name} is stored under a request ID that is not its own")
        if record["refused"] != verdict.refused or record["usage"] != {
            "input_tokens": verdict.input_tokens, "output_tokens": verdict.output_tokens}:  # fmt: skip
            raise SonnetCollectError(f"{name}'s stored refusal flag or usage differs from what its result says")
        scores[(record["item_id"], record["config"])] = PaidScore(
            verdict.score, sonnet_gate.overhead_dollars(settings, verdict), verdict.refused
        )
    return scores


# ---- the command line -----------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("what", choices=["batch", "latency"])
    parser.add_argument("--split", choices=["train", "dev", "smoke"])
    parser.add_argument("--smoke-file", type=Path)
    parser.add_argument("--sample-file", type=Path, help="latency: a JSON list of {item_id, config} to call")
    parser.add_argument("--cost-per-check", type=float, required=True)
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--poll-s", type=float, default=30.0)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    split = args.split or ("train" if args.what == "latency" else None)
    if split is None or (split == "smoke") != (args.smoke_file is not None):
        parser.error("--split is needed for a batch, and --smoke-file goes with --split smoke, and only with it")
    from harness.client import make_client  # loads the project's .env; the key itself is never read here

    settings = sonnet_gate.load_settings()
    inputs = read_smoke_inputs(args.smoke_file) if split == "smoke" else gate_inputs(split)
    if args.what == "latency":
        if args.sample_file is None or split not in ("train", "smoke"):
            parser.error("latency needs --sample-file and a train or smoke split")
        wanted = {(e["item_id"], e["config"]) for e in json.loads(args.sample_file.read_text(encoding="utf-8"))}
        inputs = [i for i in inputs if (i.item_id, i.config) in wanted]
    jobs = build_jobs(inputs, read_free_records(split, args.results_dir), settings)
    client = make_client()
    if args.what == "latency":
        made = run_latency(jobs, split=split, client=client, settings=settings, cost_per_check=args.cost_per_check,
                           max_usd=args.max_usd, results_dir=args.results_dir)  # fmt: skip
        print(f"{made} calls made")
        return 0
    sent = submit(jobs, split=split, client=client, cost_per_check=args.cost_per_check, max_usd=args.max_usd,
                  results_dir=args.results_dir)  # fmt: skip
    print(f"{sent.sent} requests sent in {len(sent.batches)} batches; {sent.in_flight} in flight, {sent.stored} stored")
    wait(jobs, split=split, client=client, results_dir=args.results_dir, poll_s=args.poll_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
