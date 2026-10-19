"""Test harness/collect.py end to end without calling the API.

Run from the project root:  uv run python -m harness.check_collect

A fake Batches client answers the smoke run's 20 requests (pilot items on haiku)
with the responses the pilot already stored for them, in shuffled order, and fails
one dev request (errored) and one test request (expired) the first time. Checks:

- the full benchmark's request counts: train without the frontier, dev and test
  with it, 600 noise-run requests;
- a request in flight is never submitted twice, and only failed requests are resent;
- storing the same batches twice changes nothing;
- scored rows match the pilot's grades for the same responses, at half the pilot's
  cost (Batches API price);
- seal refuses while a test result is missing, then records a hash that matches the
  file, makes it read-only, refuses later writes, and verifies itself when re-run;
- no other harness module names the sealed store.

Needs the pilot's stored responses (results/) and the Docker sandbox for MBPP+.
Everything else happens in a temporary folder.
"""

import hashlib
import json
import os
import random
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from anthropic.types.messages import MessageBatchIndividualResponse

from harness import run_pilot
from harness.client import PROJECT_ROOT, build_request, load_models
from harness.collect import (
    FRONTIER,
    MAIN,
    load_jobs,
    load_state,
    open_sealed,
    paths_under,
    save_state,
    score_all,
    seal,
    sealed_ids,
    submit,
    wait,
)  # fmt: skip
from harness.store import ResponseStore


class FakeBatches:
    """Stands in for client.messages.batches. Every batch has ended by the time it's checked."""

    def __init__(self, answers: dict[str, dict], fail_first_time: dict[str, str]):
        self.answers = answers
        self.fail_first_time = dict(fail_first_time)
        self.created: dict[str, list[dict]] = {}
        self.outcomes: dict[str, list] = {}

    def create(self, requests):
        batch_id = f"msgbatch_fake{len(self.created)}"
        self.created[batch_id] = list(requests)
        return SimpleNamespace(id=batch_id, created_at=datetime.now(timezone.utc))

    def retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def results(self, batch_id):
        # Fixed on first read, so reading a batch twice returns the same results.
        if batch_id not in self.outcomes:
            outcomes = []
            for request in self.created[batch_id]:
                custom_id = request["custom_id"]
                kind = self.fail_first_time.pop(custom_id, "succeeded")
                if kind == "succeeded":
                    result = {"type": "succeeded", "message": self.answers[custom_id]}
                elif kind == "errored":
                    result = {
                        "type": "errored",
                        "error": {"type": "error", "error": {"type": "api_error", "message": "fake"}},
                    }
                else:
                    result = {"type": kind}
                outcomes.append(
                    MessageBatchIndividualResponse.model_validate({"custom_id": custom_id, "result": result})
                )
            random.Random(len(self.outcomes)).shuffle(outcomes)  # results arrive in any order
            self.outcomes[batch_id] = outcomes
        return iter(self.outcomes[batch_id])


def store_snapshot(store_root: Path) -> dict[str, dict]:
    return {str(path): json.loads(path.read_text())["response"] for path in store_root.rglob("*.json")}


def main() -> int:
    problems = []

    def check(ok: bool, what: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            problems.append(what)

    models = load_models()

    jobs = load_jobs(False, models)
    counts = Counter(job.split for job in jobs)
    check(counts == {"train": 2973, "dev": 3576, "test": 2972}, f"benchmark requests per split {dict(counts)}")
    check(sum(job.run > 0 for job in jobs) == 600, "600 noise-run requests, all in dev")
    check(not any(job.split == "train" and job.config == FRONTIER for job in jobs), "no frontier requests on train")

    jobs = load_jobs(True, models)
    pilot_store = ResponseStore(run_pilot.STORE_ROOT)
    answers = {}
    for job in jobs:
        pilot_request = build_request(models["configs"][job.config], job.item["prompt"], run_pilot.PILOT_MAX_TOKENS,
                                      output_schema=job.item.get("output_schema"))  # fmt: skip
        record = pilot_store.get(pilot_request)
        assert record is not None, f"no pilot response for {job.custom_id}"
        answers[job.custom_id] = record["response"]
    dev_fail = next(job.custom_id for job in jobs if job.split == "dev")
    test_fail = next(job.custom_id for job in jobs if job.split == "test")
    batches = FakeBatches(answers, {dev_fail: "errored", test_fail: "expired"})
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))

    with tempfile.TemporaryDirectory() as tmp:
        paths = paths_under(Path(tmp))

        print("\n-- first round: submit twice, then store --")
        submit(paths, jobs, models, client, max_usd=1.0, check=False)
        check(sorted(len(r) for r in batches.created.values()) == [4, 16], "one batch per split (16 dev, 4 test)")
        submit(paths, jobs, models, client, max_usd=1.0, check=False)
        check(len(batches.created) == 2, "requests in flight are not submitted again")
        wait(paths, jobs, models, client, poll_s=0)
        state = load_state(paths)
        failed = {cid for batch in state["batches"] for cid in batch["failed"]}
        check(failed == {dev_fail, test_fail}, "the errored and expired requests are recorded as failed")
        check(len(sealed_ids(paths)) == 3, "3 of 4 test results stored in the sealed store")
        stored_ids = {json.loads(path.read_text())["custom_id"] for path in paths.store.rglob("*.json")}
        test_ids = {job.custom_id for job in jobs if job.split == "test"}
        check(len(stored_ids) == 15 and not stored_ids & test_ids,
              "the response store holds the 15 dev results and no test result")  # fmt: skip
        try:
            seal(paths, jobs, None)
            check(False, "seal refuses while a test result is missing")
        except SystemExit:
            check(not paths.manifest.exists(), "seal refuses while a test result is missing")

        print("\n-- second round: resend failures --")
        submit(paths, jobs, models, client, max_usd=1.0, check=False)
        resent = sorted(r["custom_id"] for batch_id in list(batches.created)[2:] for r in batches.created[batch_id])
        check(resent == sorted([dev_fail, test_fail]), "only the two failed requests are resent")
        wait(paths, jobs, models, client, poll_s=0)
        check(len(sealed_ids(paths)) == 4 and len(store_snapshot(paths.store)) == 16, "every result is stored")

        print("\n-- store every batch again --")
        before = (store_snapshot(paths.store), sealed_ids(paths))
        state = load_state(paths)
        for batch in state["batches"]:
            batch["ingested_at"] = None
        save_state(paths, state)
        wait(paths, jobs, models, client, poll_s=0)
        check(
            (store_snapshot(paths.store), sealed_ids(paths)) == before, "storing the same batches twice changes nothing"
        )

        print("\n-- score --")
        score_all(paths, jobs, models, rescore=False)
        pilot_rows = {(row["item_id"], row["format"], row["config"]): row for row in map(json.loads, run_pilot.ROWS_DIR.joinpath("pilot.rows.jsonl").open())}  # fmt: skip
        with closing(sqlite3.connect(paths.matrix)) as conn:
            rows = conn.execute("SELECT item_id, config, status, cost_usd, latency_s, split FROM rows").fetchall()
        mismatched = []
        for item_id, config, status, cost, latency, split in rows:
            fmt = "structured" if item_id.startswith(("banking77", "mmlu_pro")) else "free"
            pilot = pilot_rows[(item_id, fmt, config)]
            if (status, latency) != (pilot["status"], None) or abs(cost - pilot["cost_usd"] / 2) > 1e-12:
                mismatched.append((item_id, status, pilot["status"]))
        check(len(rows) == 16 and {row[5] for row in rows} == {"dev"}, "16 dev rows scored, no test rows")
        check(not mismatched, f"grades match the pilot's, at half its cost, latency empty {mismatched[:3]}")

        print("\n-- seal --")
        check(seal(paths, jobs, None) == 0, "seal succeeds once every test result is in")
        manifest = json.loads(paths.manifest.read_text())
        digest = hashlib.sha256(paths.sealed.read_bytes()).hexdigest()
        check(manifest["test_sealed"]["sha256"] == digest and manifest["test_sealed"]["rows"] == 4,
              "the manifest records the sealed file's sha256 and row count")  # fmt: skip
        check(not os.access(paths.sealed, os.W_OK), "the sealed store is read-only")
        try:
            open_sealed(paths)
            check(False, "writing to the sealed store is refused")
        except RuntimeError:
            check(True, "writing to the sealed store is refused")
        check(seal(paths, jobs, None) == 0, "sealing again verifies the hash")
        os.chmod(paths.sealed, 0o644)  # so the temporary folder can be removed

    harness = PROJECT_ROOT / "harness"
    naming = [p.name for p in harness.glob("*.py") if p.name != "collect.py" and MAIN.sealed.name in p.read_text()]
    check(not naming, f"no other harness module names the sealed store {naming}")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
