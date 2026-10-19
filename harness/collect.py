"""Stage 2 collection run (PLAN_v4.md §6 and §14; Amendments A1 and A6).

Sends the benchmark through the model configs with the Message Batches API: train
through every config except the frontier, dev and test through all four, and the dev
noise subset twice more as runs 1 and 2. Each split goes in a batch of its own.

Run from the project root, in this order:
  uv run python -m harness.collect plan                  # request counts and projected cost; no API calls
  uv run python -m harness.collect submit --max-usd 45   # submit every request not stored or in flight
  uv run python -m harness.collect wait                  # poll until every batch ends, then store its results
  uv run python -m harness.collect score                 # score train/dev into results/dev_matrix.sqlite
  uv run python -m harness.collect seal                  # make the test store read-only and record its hash

`--smoke` runs the same commands on 20 pilot items on haiku (Amendment A1, item 8),
with every file under results/collection_smoke/.

Train and dev responses go to the response store (results/responses/collection/),
then scored rows go to results/dev_matrix.sqlite. Test responses go only to
results/test_sealed.sqlite, unscored. For test, this module reports request outcomes,
model-ID mismatches and total cost, nothing more (Amendment A6, items 3 and 4). It is
the only module that names the sealed store.

Every command is safe to re-run. A batch's ID is saved to the state file as soon as the
batch is created, `submit` skips requests that are stored or in a batch not yet stored,
so only errored or expired requests are sent again, and storing a batch twice changes
nothing.
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness import check_benchmark
from harness.build_benchmark import BENCHMARK_FILE, MANIFEST_FILE
from harness.client import PROJECT_ROOT, answer_text, build_request, cost_usd, load_models, make_client
from harness.evaluators import score
from harness.loaders import structured_variant
from harness.run_pilot import print_table
from harness.store import ResponseStore, request_hash

RESULTS = PROJECT_ROOT / "results"
SPLITS = ("train", "dev", "test")
FRONTIER = "opus"  # runs on dev and test only (Amendment A1, item 3)
NOISE_RUNS = (1, 2)  # the noise subset's extra runs; every analysis uses run 0 (Amendment A6, item 1)
CUSTOM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # what the Batches API accepts
MAX_BATCH_BYTES = 200_000_000  # under the Batches API's 256 MB per batch

# The cost projection uses the pilot's mean cost per (source, format, config).
PILOT_ROWS = RESULTS / "pilot" / "pilot.rows.jsonl"
PILOT_FILE = PROJECT_ROOT / "data" / "pilot.jsonl"
SMOKE_SOURCES = ("ifeval", "banking77", "mmlu_pro", "mbppplus")
SMOKE_PER_SOURCE = 5  # 20 items in all; the last of each source stands in for test


@dataclass(frozen=True)
class Paths:
    state: Path  # batch IDs and each batch's outcome
    store: Path  # response store root for train and dev
    matrix: Path  # scored train and dev rows
    sealed: Path  # raw test responses, unscored until Stage 8
    manifest: Path  # where `seal` records the sealed store's hash


MAIN = Paths(
    state=RESULTS / "collection" / "batches.json",
    store=RESULTS / "responses" / "collection",
    matrix=RESULTS / "dev_matrix.sqlite",
    sealed=RESULTS / "test_sealed.sqlite",
    manifest=PROJECT_ROOT / "data" / "collection_manifest.json",
)


def paths_under(root: Path) -> Paths:
    """Every file in one folder: the smoke run, and check_collect's throwaway runs."""
    return Paths(root / "batches.json", root / "responses", root / "dev_matrix.sqlite",
                 root / "test_sealed.sqlite", root / "collection_manifest.json")  # fmt: skip


SMOKE = paths_under(RESULTS / "collection_smoke")


@dataclass(frozen=True)
class Job:
    custom_id: str
    split: str
    item: dict
    config: str
    run: int
    request: dict


def make_custom_id(item_id: str, config: str, run: int) -> str:
    # Item IDs contain ":", which the API doesn't accept (Amendment A6, item 5).
    return f"{re.sub(r'[^A-Za-z0-9_-]', '-', item_id)}_{config}_r{run}"


def build_jobs(items: list[dict], models: dict, configs: list[str], noise_ids: set[str]) -> list[Job]:
    jobs = []
    for item in items:
        for name in configs:
            if item["split"] == "train" and name == FRONTIER:
                continue
            config = models["configs"][name]
            request = build_request(config, item["prompt"], config["max_tokens"], output_schema=item.get("output_schema"))
            for run in (0, *NOISE_RUNS) if item["item_id"] in noise_ids else (0,):
                jobs.append(Job(make_custom_id(item["item_id"], name, run), item["split"], item, name, run, request))

    ids = [job.custom_id for job in jobs]
    invalid = [i for i in ids if not CUSTOM_ID.match(i)]
    if invalid or len(set(ids)) != len(ids):
        raise ValueError(f"custom_ids must be valid and unique: {invalid[:3]}")
    # The response store keys by request and run, so two jobs sharing both would
    # silently share one stored answer.
    keys = [(request_hash(job.request), job.run) for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("two jobs send the same request as the same run")
    return jobs


def load_jobs(smoke: bool, models: dict) -> list[Job]:
    if not smoke:
        items = [json.loads(line) for line in BENCHMARK_FILE.open()]
        noise = set(json.loads(MANIFEST_FILE.read_text())["noise_subset"])
        return build_jobs(items, models, list(models["configs"]), noise)

    taken = Counter()
    items = []
    for line in PILOT_FILE.open():
        item = json.loads(line)
        if item["source"] not in SMOKE_SOURCES or taken[item["source"]] == SMOKE_PER_SOURCE:
            continue
        taken[item["source"]] += 1
        if item["source"] in ("banking77", "mmlu_pro"):  # answered as JSON (Amendment A3, item 2)
            item = structured_variant(item)
        items.append({**item, "split": "test" if taken[item["source"]] == SMOKE_PER_SOURCE else "dev"})
    return build_jobs(items, models, ["haiku"], set())


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def usd(dollars: float) -> str:
    # Cents hide a smoke run's cost, so small amounts get four decimals.
    return f"${dollars:.2f}" if dollars >= 1 else f"${dollars:.4f}"


# State file


def load_state(paths: Paths) -> dict:
    return json.loads(paths.state.read_text()) if paths.state.exists() else {"batches": []}


def save_state(paths: Paths, state: dict) -> None:
    paths.state.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.state.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, paths.state)


# Sealed test store

SEALED_COLUMNS = (
    "custom_id", "item_id", "config", "model", "effort", "run", "batch_id", "request_hash", "reported_model",
    "stop_reason", "input_tokens", "output_tokens", "thinking_tokens", "answer", "response_json", "created_at",
)  # fmt: skip


def sealed_ids(paths: Paths) -> set[str]:
    """Which test requests have a stored result. Reads no answers."""
    if not paths.sealed.exists():
        return set()
    with closing(sqlite3.connect(paths.sealed.as_uri() + "?mode=ro", uri=True)) as conn:
        return {row[0] for row in conn.execute("SELECT custom_id FROM answers")}


def open_sealed(paths: Paths) -> sqlite3.Connection:
    if paths.sealed.exists() and not os.access(paths.sealed, os.W_OK):
        raise RuntimeError(f"{paths.sealed.name} is sealed (read-only); refusing to write to it")
    paths.sealed.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(paths.sealed)
    conn.execute(f"CREATE TABLE IF NOT EXISTS answers ({', '.join(SEALED_COLUMNS)}, PRIMARY KEY (custom_id))")
    return conn


# Submit and store


def pending_jobs(paths: Paths, jobs: list[Job], state: dict) -> tuple[list[Job], set[str]]:
    in_flight = {cid for batch in state["batches"] if batch["ingested_at"] is None for cid in batch["custom_ids"]}
    store = ResponseStore(paths.store)
    sealed = sealed_ids(paths)

    def stored(job: Job) -> bool:
        return job.custom_id in sealed if job.split == "test" else store.get(job.request, job.run) is not None

    return [job for job in jobs if job.custom_id not in in_flight and not stored(job)], in_flight


def projected_cost(jobs: list[Job], models: dict) -> float:
    costs = defaultdict(list)
    for line in PILOT_ROWS.open():
        row = json.loads(line)
        costs[(row["source"], row["format"], row["config"])].append(row["cost_usd"])
    mean = {key: statistics.mean(values) for key, values in costs.items()}
    total = sum(
        mean[(job.item["source"], "structured" if "output_schema" in job.item else "free", job.config)] for job in jobs
    )
    return total * models["batch_discount"]


def batch_requests(jobs: list[Job]) -> list[dict]:
    requests = [{"custom_id": job.custom_id, "params": job.request} for job in jobs]
    size = len(json.dumps(requests).encode("utf-8"))
    if size > MAX_BATCH_BYTES:
        raise ValueError(f"batch of {len(jobs)} requests is {size:,} bytes, over {MAX_BATCH_BYTES:,}")
    return requests


def submit(paths: Paths, jobs: list[Job], models: dict, client, max_usd: float, check: bool = True) -> None:
    state = load_state(paths)
    pending, in_flight = pending_jobs(paths, jobs, state)
    print(f"{len(jobs)} requests: {len(pending)} to submit, {len(in_flight)} in flight, the rest stored")
    if not pending:
        return
    projected = projected_cost(pending, models)
    print(f"Projected cost at pilot rates, Batches API price: {usd(projected)} (cap {usd(max_usd)})")
    if projected > max_usd:
        raise SystemExit("Projected cost is over the cap; nothing submitted.")
    if check and check_benchmark.main() != 0:  # Amendment A5, item 7
        raise SystemExit("check_benchmark failed; nothing submitted.")

    for split in SPLITS:
        group = [job for job in pending if job.split == split]
        if not group:
            continue
        batch = client.messages.batches.create(requests=batch_requests(group))
        # Saved before anything else, so no crash from here on can lead to paying for it twice.
        state["batches"].append({
            "id": batch.id,
            "split": split,
            "created_at": batch.created_at.isoformat(),
            "ingested_at": None,
            "custom_ids": [job.custom_id for job in group],
        })  # fmt: skip
        save_state(paths, state)
        print(f"Submitted {split}: {len(group)} requests as {batch.id}", flush=True)


def ingest(paths: Paths, entry: dict, by_id: dict[str, Job], models: dict, client) -> None:
    """Store an ended batch's results. Test results go only to the sealed store, and
    what is printed about them is the same for every split: outcomes, cost, mismatches."""
    store = ResponseStore(paths.store)
    sealed = open_sealed(paths) if entry["split"] == "test" else None
    counts, failed, cost, mismatched = Counter(), {}, 0.0, 0
    stored_at = now()
    try:
        for result in client.messages.batches.results(entry["id"]):
            job = by_id[result.custom_id]
            assert job.split == entry["split"], result.custom_id
            counts[result.result.type] += 1
            if result.result.type != "succeeded":
                failed[result.custom_id] = result.result.to_dict(mode="json")
                continue
            response = result.result.message.to_dict(mode="json")
            record = {
                "run": job.run,
                "request": job.request,
                "response": response,
                "request_id": None,  # batch results carry none
                "batch_id": entry["id"],
                "custom_id": job.custom_id,
                "latency_s": None,  # no meaningful latency in a batch (Amendment A1, item 5)
                "created_at": stored_at,
            }
            cost += cost_usd(record, models["prices"], models["batch_discount"])
            mismatched += response["model"] != job.request["model"]
            if sealed is None:
                store.put(job.request, job.run, record)
                continue
            usage = response["usage"]
            sealed.execute(
                f"INSERT OR IGNORE INTO answers VALUES ({', '.join('?' * len(SEALED_COLUMNS))})",
                (job.custom_id, job.item["item_id"], job.config, job.request["model"],
                 job.request["output_config"]["effort"], job.run, entry["id"], request_hash(job.request),
                 response["model"], response["stop_reason"], usage["input_tokens"], usage["output_tokens"],
                 (usage.get("output_tokens_details") or {}).get("thinking_tokens"), answer_text(response),
                 json.dumps(response), stored_at),
            )  # fmt: skip
        if sealed is not None:
            sealed.commit()
    finally:
        if sealed is not None:
            sealed.close()

    entry.update(ingested_at=stored_at, counts=dict(counts), cost_usd=round(cost, 6), model_mismatches=mismatched,
                 failed=failed)  # fmt: skip
    outcomes = ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
    print(f"Stored {entry['split']} batch {entry['id']}: {outcomes}; {usd(cost)}; {mismatched} model mismatches",
          flush=True)  # fmt: skip


def wait(paths: Paths, jobs: list[Job], models: dict, client, poll_s: float) -> None:
    state = load_state(paths)
    by_id = {job.custom_id: job for job in jobs}
    while open_batches := [batch for batch in state["batches"] if batch["ingested_at"] is None]:
        for entry in open_batches:
            batch = client.messages.batches.retrieve(entry["id"])
            if batch.processing_status == "ended":
                ingest(paths, entry, by_id, models, client)
                save_state(paths, state)
            else:
                counts = batch.request_counts
                print(f"{now()} {entry['split']:5s} {entry['id']}: {batch.processing_status}, "
                      f"{counts.processing} processing, {counts.succeeded} succeeded, {counts.errored} errored",
                      flush=True)  # fmt: skip
        if any(batch["ingested_at"] is None for batch in state["batches"]):
            time.sleep(poll_s)

    pending, _ = pending_jobs(paths, jobs, state)
    total = sum(batch.get("cost_usd", 0) for batch in state["batches"])
    print(f"Every batch is stored. Cost so far: {usd(total)}.")
    if pending:
        print(f"{len(pending)} requests have no stored result; run `submit` again to resend them.")


# Score train and dev

MATRIX_COLUMNS = (
    "custom_id", "item_id", "split", "source", "task_type", "config", "model", "reported_model", "effort", "run",
    "stop_reason", "refusal_category", "status", "quality", "success", "parsed", "detail", "answer",
    "input_tokens", "cached_input_tokens", "output_tokens", "thinking_tokens", "cost_usd", "latency_s",
    "price_snapshot", "batch_id", "request_hash",
)  # fmt: skip


def make_row(job: Job, record: dict, models: dict) -> dict:
    response = record["response"]
    usage = response["usage"]
    result = score(job.item, response)
    return {
        "custom_id": job.custom_id,
        "item_id": job.item["item_id"],
        "split": job.split,
        "source": job.item["source"],
        "task_type": job.item["task_type"],
        "config": job.config,
        "model": job.request["model"],
        "reported_model": response["model"],
        "effort": job.request["output_config"]["effort"],
        "run": job.run,
        "stop_reason": response["stop_reason"],
        "refusal_category": (response.get("stop_details") or {}).get("category"),
        "status": result.status,
        "quality": result.quality,
        "success": int(result.success),
        "parsed": result.parsed,
        "detail": json.dumps(result.detail, default=str),
        "answer": answer_text(response),
        "input_tokens": usage["input_tokens"],
        "cached_input_tokens": usage.get("cache_read_input_tokens") or 0,
        "output_tokens": usage["output_tokens"],
        "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
        "cost_usd": cost_usd(record, models["prices"], models["batch_discount"]),
        "latency_s": record["latency_s"],
        "price_snapshot": models["price_snapshot"],
        "batch_id": record["batch_id"],
        "request_hash": request_hash(job.request),
    }


def score_all(paths: Paths, jobs: list[Job], models: dict, rescore: bool) -> None:
    # Scored here, on the main thread, one at a time (see evaluators.py).
    paths.matrix.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(paths.matrix)) as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS rows ({', '.join(MATRIX_COLUMNS)}, PRIMARY KEY (custom_id))")
        scored = set() if rescore else {row[0] for row in conn.execute("SELECT custom_id FROM rows")}
        store = ResponseStore(paths.store)
        todo = [job for job in jobs if job.split != "test" and job.custom_id not in scored]
        missing = 0
        for n, job in enumerate(todo, 1):
            record = store.get(job.request, job.run)
            if record is None:
                missing += 1
                continue
            row = make_row(job, record, models)
            conn.execute(f"INSERT OR REPLACE INTO rows VALUES ({', '.join('?' * len(MATRIX_COLUMNS))})",
                         [row[column] for column in MATRIX_COLUMNS])  # fmt: skip
            if n % 100 == 0:
                conn.commit()
                print(f"[{n}/{len(todo)}] scored", flush=True)
        conn.commit()
        print(f"Scored {len(todo) - missing} rows; {missing} train/dev requests have no stored response yet.")
        summarize(conn, list(models["configs"]))


def summarize(conn: sqlite3.Connection, configs: list[str]) -> None:
    rows = conn.execute("SELECT split, source, config, run, status, cost_usd, model != reported_model FROM rows")
    cells, cost, runs, mismatched = defaultdict(Counter), defaultdict(list), Counter(), 0
    for split, source, config, run, status, dollars, mismatch in rows:
        runs[run] += 1
        mismatched += mismatch
        if run == 0:
            cells[(split, source, config)][status] += 1
            cost[config].append(dollars)

    print("\nRun 0, correct / items. Brackets: no answer in the requested format, other failures (stops).")
    table = []
    for split in ("train", "dev"):
        for source in sorted({key[1] for key in cells}):
            line = [f"{split} {source}"]
            for config in configs:
                status = cells.get((split, source, config))
                if not status:
                    line.append("-")
                    continue
                other = sum(status.values()) - status["correct"] - status["wrong"] - status["no_answer"]
                extra = f" [{status['no_answer']} fmt]" if status["no_answer"] else ""
                extra += f" [{other} stop]" if other else ""
                line.append(f"{status['correct']}/{sum(status.values())}{extra}")
            if any(cell != "-" for cell in line[1:]):
                table.append(line)
    print_table(["split, task"] + configs, table)
    print("\nMean cost per request (run 0, Batches API price): "
          + ", ".join(f"{c} ${statistics.mean(cost[c]):.5f}" for c in configs if cost[c]))  # fmt: skip
    print(f"Rows by run: {dict(sorted(runs.items()))}. Model mismatches: {mismatched}.")


# Seal


def seal(paths: Paths, jobs: list[Job], benchmark_sha256: str | None) -> int:
    test_ids = {job.custom_id for job in jobs if job.split == "test"}
    present = sealed_ids(paths)
    if present - test_ids:
        raise SystemExit(f"The sealed store holds {len(present - test_ids)} rows outside the test jobs.")
    if test_ids - present:
        raise SystemExit(f"{len(test_ids - present)} of {len(test_ids)} test requests have no result yet; "
                         "nothing sealed.")  # fmt: skip
    digest = hashlib.sha256(paths.sealed.read_bytes()).hexdigest()

    if not os.access(paths.sealed, os.W_OK):  # sealed before: check it instead
        recorded = json.loads(paths.manifest.read_text())["test_sealed"]["sha256"]
        print(f"Already sealed. sha256 {digest[:16]}... {'matches' if digest == recorded else 'DOES NOT MATCH'} "
              f"the manifest.")  # fmt: skip
        return 0 if digest == recorded else 1

    state = load_state(paths)
    batch_fields = ("id", "split", "created_at", "ingested_at", "counts", "cost_usd", "model_mismatches")
    manifest = {
        "built_by": "harness/collect.py",
        "sealed_at": now(),
        "benchmark_sha256": benchmark_sha256,
        "price_snapshot": load_models()["price_snapshot"],
        "test_sealed": {"file": paths.sealed.name, "sha256": digest, "rows": len(present)},
        "batches": [{key: batch.get(key) for key in batch_fields} for batch in state["batches"]],
        "total_cost_usd": round(sum(batch.get("cost_usd", 0) for batch in state["batches"]), 2),
    }
    # The manifest is written first, so an interrupted seal leaves the store writable
    # and a re-run simply seals it again.
    paths.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.chmod(paths.sealed, 0o444)
    print(f"Sealed {len(present)} test responses. sha256 {digest}")
    print(f"Recorded in {paths.manifest}")
    return 0


# Plan


def plan(paths: Paths, jobs: list[Job], models: dict) -> None:
    configs = list(models["configs"])
    counts = Counter((job.split if job.run == 0 else f"{job.split} noise", job.config) for job in jobs)
    rows = [[group] + [counts[(group, c)] or "-" for c in configs] + [sum(counts[(group, c)] for c in configs)]
            for group in ("train", "dev", "dev noise", "test") if any(counts[(group, c)] for c in configs)]  # fmt: skip
    print_table(["requests"] + configs + ["total"], rows)
    for split in SPLITS:
        group = [job for job in jobs if job.split == split]
        if group:
            size = len(json.dumps(batch_requests(group)).encode("utf-8"))
            print(f"  {split} batch: {len(group)} requests, {size / 1e6:.1f} MB")
    pending, in_flight = pending_jobs(paths, jobs, load_state(paths))
    print(f"{len(jobs)} requests in all; projected {usd(projected_cost(jobs, models))} at pilot rates, Batches API price.")
    print(f"Not yet stored: {len(pending)} ({usd(projected_cost(pending, models))}); in flight: {len(in_flight)}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "submit", "wait", "score", "seal"))
    parser.add_argument("--smoke", action="store_true", help="20 pilot items on haiku, in results/collection_smoke/")
    parser.add_argument("--max-usd", type=float, help="submit: refuse if the projected cost is higher (required)")
    parser.add_argument("--poll", type=float, default=60, help="wait: seconds between status checks (default 60)")
    parser.add_argument("--rescore", action="store_true", help="score: rescore rows already in the matrix")
    args = parser.parse_args()
    if args.command == "submit" and args.max_usd is None:
        parser.error("submit needs --max-usd")

    models = load_models()
    paths = SMOKE if args.smoke else MAIN
    jobs = load_jobs(args.smoke, models)
    if args.command == "plan":
        plan(paths, jobs, models)
    elif args.command == "submit":
        submit(paths, jobs, models, make_client(), args.max_usd, check=not args.smoke)
    elif args.command == "wait":
        wait(paths, jobs, models, make_client(), args.poll)
    elif args.command == "score":
        score_all(paths, jobs, models, args.rescore)
    else:
        benchmark_sha256 = None if args.smoke else json.loads(MANIFEST_FILE.read_text())["sha256"]["benchmark.jsonl"]
        return seal(paths, jobs, benchmark_sha256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
