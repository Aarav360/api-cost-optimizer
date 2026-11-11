"""The Stage 7 smoke test, before the settings commit (PLAN_v4.md §9; Amendment A17, items 18 and 20.1; issue #68).

Made-up answers to real train prompts, from all four tasks: `analysis/gates_smoke_answers.json` holds each answer with
its request. The prompts are train prompts (a check holds each equal to its item's), the answers are invented here and
are none of the results matrix's: no gate sees a benchmark answer before the settings commit (item 20). The smoke test:

1. runs the free checks on them (Docker), and compares each verdict with what was written down beforehand: the code
   check passes made-up code that is right (the import rule included: one answer never imports `math`, one reaches
   `sys` without importing it) and fails wrong code, an empty block and a missing block; the stop check fails a cap stop;
2. measures the Jev gate's fixed overhead from minimal calls (the largest of three, as Stage 6's did), and the questions
   and the longest output schema in tokens, and from them the byte budget (A16 item 8, `jev_gate.gate_budget_bytes`);
   then makes the gate's calls on the answers that passed, which shows that the frozen `parse_answers` accepts the
   three statement IDs (no call refused);
3. sends the Sonnet gate's answers that passed through the Batches API (a tiny batch, which shows the batch path
   accepts the enum schema) and four through the standard API (one per task), and measures the cost per check per task;
4. projects Stage 7's gate spend from those measured costs against the $25 cap (item 18).

Nothing here changes a statement, a prompt, a setting or a rule because of a verdict (item 20.1): a mechanical failure
stops the run and the fix is made by hand and recorded in the record's `fixes`. The record, with every stored response,
goes to `analysis/gates_smoke_record.json`; the measured values it holds are then filled into
`routers/gate_settings.json` by hand (the byte budget, the image digest, the Dockerfile's hash).

Run from the project root (keys in `TYPESAFE_API_KEY` and `ANTHROPIC_API_KEY`, or in .env):
    uv run python -m analysis.smoke_gates
Refuses to run if the record already exists, and stops before any paid call if a free-check verdict is not the expected
one. Every stage is rerun-safe (stored records are never called again), and the Jev measurement calls are kept as they
come in `results/gates/smoke_measurements.json` (uncommitted). The whole smoke is capped at $2.
"""

import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence

from analysis import collect_gates, collect_jev_gate, gate_spend
from analysis import collect_sonnet_gate as sonnet_collect
from analysis.benchmark import read_rows
from analysis.collect_jev import RESULTS_DIR, from_row, load_settings, now_local
from analysis.noise_samples import task_of
from harness.client import PROJECT_ROOT, load_models, make_client
from routers import gates, jev, jev_call, jev_gate, sonnet_gate

ANSWERS_FILE = PROJECT_ROOT / "analysis" / "gates_smoke_answers.json"
RECORD_FILE = PROJECT_ROOT / "analysis" / "gates_smoke_record.json"
DOCKERFILE = PROJECT_ROOT / "sandbox" / "Dockerfile"
MEASUREMENTS_FILE = RESULTS_DIR / "gates" / "smoke_measurements.json"
SMOKE_CAP_USD = 2.0
MINIMAL_REPEATS = 3
ONE_QUESTION = {"q": {"type": "noul", "instructions": "Is this a greeting?"}}
PROVISIONAL_COST_PER_CHECK = 0.01  # dollars, a high guess to size the batch's cap check; the measured cost replaces it
STANDARD_CALLS = (  # one haiku answer per task through the standard API
    ("banking77:test-0011", "haiku"), ("mmlu_pro:10019", "haiku"), ("ifeval:1012", "haiku"), ("mbppplus:102", "haiku"),
)  # fmt: skip
NOISE_PER_TASK = 25  # the Jev gate's noise pass (A17 item 17)
LATENCY_PER_TASK = 20  # the Sonnet gate's latency sample (A17 item 16)
SONNET_CONFIGS = 3  # haiku, sonnet, opus_low
# Written down before the run: the answers whose free checks must fail, and the check that must fail first. Every
# other answer in the file must pass all three.
EXPECTED_FREE_FAILS = {
    ("banking77:test-0008", "haiku"): "stop",  # a cap stop
    ("mbppplus:139", "opus_low"): "code",  # wrong code
    ("mbppplus:85", "haiku"): "code",  # no code block
    ("mbppplus:124", "haiku"): "code",  # an empty code block
}


class SmokeError(RuntimeError):
    """A mechanical failure of the smoke test. The run stops; nothing is decided on its account."""


# ---- the pure parts (checked in analysis/check_gates_settings_commit.py) -------------------------------------


def task_means(samples: Sequence[tuple[str, float]]) -> dict[str, float]:
    """The mean of each task's values, from (item ID, value) pairs."""
    by_task: dict[str, list[float]] = {}
    for item, value in samples:
        by_task.setdefault(task_of(item), []).append(value)
    return {task: sum(values) / len(values) for task, values in by_task.items()}


def project(
    counts: Mapping[str, Mapping[str, int]],
    *,
    sonnet_batch: Mapping[str, float],
    sonnet_list: Mapping[str, float],
    jev: Mapping[str, float],
    cap: float = gate_spend.STAGE_CAP_USD,
) -> dict:
    """Stage 7's gate spend, in dollars, from the measured cost of one check per task (A17 item 18). `counts` is each
    split's items per task. An upper bound: every answer is taken to pass the free checks and be judged. The Sonnet
    gate judges three configs on train and dev at batch prices; the Jev gate judges three on train and four on dev;
    the Jev noise pass is 25 answers per task; the latency sample is 20 per task at list price."""
    tasks = counts["train"]
    sonnet_train = sum(counts["train"][t] * SONNET_CONFIGS * sonnet_batch[t] for t in tasks)
    sonnet_dev = sum(counts["dev"][t] * SONNET_CONFIGS * sonnet_batch[t] for t in tasks)
    jev_train = sum(counts["train"][t] * 3 * jev[t] for t in tasks) + sum(NOISE_PER_TASK * jev[t] for t in tasks)
    jev_dev = sum(counts["dev"][t] * 4 * jev[t] for t in tasks)
    latency = sum(LATENCY_PER_TASK * sonnet_list[t] for t in tasks)
    total = sonnet_train + sonnet_dev + jev_train + jev_dev + latency
    return {
        "sonnet_train_usd": sonnet_train, "sonnet_dev_usd": sonnet_dev, "jev_train_with_noise_usd": jev_train,
        "jev_dev_usd": jev_dev, "latency_sample_usd": latency, "total_usd": total, "cap_usd": cap,
        "headroom_usd": cap - total, "within_cap": total <= cap,
    }  # fmt: skip


def free_check_mismatches(records: Sequence[Mapping[str, object]]) -> list[str]:
    """Where the free-check records differ from EXPECTED_FREE_FAILS: an answer that failed that should have passed, one
    that passed that should have failed, or one that failed at another check than expected."""
    problems = []
    for record in records:
        key = (str(record["item_id"]), str(record["config"]))
        want = EXPECTED_FREE_FAILS.get(key)
        if want is None and not record["passed"]:
            problems.append(f"{key} failed {record['failed_first']} but should pass")
        elif want is not None and (record["passed"] or record["failed_first"] != want):
            problems.append(f"{key} should fail {want}, it {'passed' if record['passed'] else record['failed_first']}")
    return problems


# ---- the run ---------------------------------------------------------------------------------------------------


def longest_schema() -> dict | None:
    """The train or dev request output schema with the most serialized bytes (requests only, never an answer)."""
    schemas = [row["output_schema"] for row in read_rows() if row.get("output_schema")]
    return max(schemas, key=lambda schema: len(jev_call.dumps(schema)), default=None)


def check_prompts(inputs: Sequence[collect_gates.GateInput]) -> None:
    """Every smoke request is the real train request of its item (prompt and schema alike)."""

    rows = {row["item_id"]: from_row(row) for row in read_rows(split="train")}
    for item in inputs:
        if item.item_id not in rows or rows[item.item_id] != item.request:
            raise SmokeError(f"{item.item_id} is not a train item with this prompt and schema")


def measure_jev(settings: jev_gate.JevGateSettings, client: jev.JevClient, schema: dict | None) -> dict[str, dict]:
    """Send each measurement body once (resuming from the kept file) and return label -> what came back."""
    tiny = jev_gate.gate_state("Hi", None, "Hi")

    def body(state: dict, questions: dict[str, dict]) -> bytes:
        return jev_call.dumps({"model": settings.model, "state": state, "questions": questions})

    bodies: dict[str, bytes] = {f"minimal_{n}": body(tiny, ONE_QUESTION) for n in range(1, MINIMAL_REPEATS + 1)}
    bodies["gate_questions"] = body(tiny, jev_gate.gate_questions(settings))
    bodies["longest_schema"] = body(jev_gate.gate_state("Hi", schema, "Hi"), ONE_QUESTION)
    done = json.loads(MEASUREMENTS_FILE.read_text(encoding="utf-8")) if MEASUREMENTS_FILE.exists() else {}
    for label, payload in bodies.items():
        if label in done:
            continue
        result = client.call(payload, jev_call.body_sha256(payload), f"gates smoke measurement {label}")
        usage = result.usage or {}
        done[label] = {
            "body_bytes": len(payload), "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"), "request_id": result.request_id,
            "server_date": result.server_date, "seconds": result.seconds, "attempts": result.attempts,
        }  # fmt: skip
        MEASUREMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        MEASUREMENTS_FILE.write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
        print(f"  measured {label}: {done[label]['input_tokens']} input tokens", flush=True)
    return done


def main() -> int:
    if RECORD_FILE.exists():
        print(f"{RECORD_FILE.name} exists: the smoke test was run")
        return 1
    ran_at = now_local()
    inputs = collect_gates.read_smoke_inputs(ANSWERS_FILE)
    check_prompts(inputs)
    tasks = sorted({task_of(i.item_id) for i in inputs})
    print(f"{len(inputs)} made-up answers to train prompts of {tasks}")

    # 1. the free checks
    code_check = gates.read_gate_settings().code_check
    digest = gates.image_digest(code_check.image)
    summary = collect_gates.collect_free(inputs, split="smoke", settings=code_check, digest=digest)
    free = collect_gates.read_free_records("smoke")
    print(f"free checks: {summary.new} checked, {summary.skipped} already stored")
    if mismatches := free_check_mismatches(free):
        print(
            "a free-check verdict is not the expected one; stopping before any paid call:\n  " + "\n  ".join(mismatches)
        )
        return 2
    passed = sum(bool(r["passed"]) for r in free)
    print(f"  as expected: {len(free) - passed} fail, {passed} pass")

    # 2. the Jev gate
    loaded = load_settings()
    gate_settings = jev_gate.read_settings()
    schema = longest_schema()
    with jev.JevClient(loaded.client) as client:
        measurements = measure_jev(gate_settings, client, schema)
        overhead = max(measurements[f"minimal_{n}"]["input_tokens"] for n in range(1, MINIMAL_REPEATS + 1))
        budget = jev_gate.gate_budget_bytes(gate_settings, overhead_tokens=overhead, schema=schema)
        print(f"Jev: overhead {overhead} tokens, byte budget {budget}")
        jev_summary = collect_jev_gate.collect(
            inputs, split="smoke", settings=gate_settings, client=client, budget_bytes=budget,
            progress=collect_jev_gate.show_progress,
        )  # fmt: skip
    print(f"Jev gate: {jev_summary.new} new, {jev_summary.skipped} stored, {jev_summary.not_called} not called, "
          f"{jev_summary.refused} refused, {jev_summary.cut} cut")  # fmt: skip
    jev_records = collect_jev_gate.read_records("smoke")

    # 3. the Sonnet gate
    settings = sonnet_gate.load_settings()
    jobs = sonnet_collect.build_jobs(inputs, free, settings)
    standard_wanted = set(STANDARD_CALLS)
    anthropic_client = make_client()
    sent = sonnet_collect.submit(jobs, split="smoke", client=anthropic_client,
                                 cost_per_check=PROVISIONAL_COST_PER_CHECK, max_usd=SMOKE_CAP_USD)  # fmt: skip
    print(f"Sonnet batch: {sent.sent} requests in {len(sent.batches)} batches; waiting")
    sonnet_collect.wait(jobs, split="smoke", client=anthropic_client)
    made = sonnet_collect.run_latency([j for j in jobs if (j.item_id, j.config) in standard_wanted], split="smoke",
                                      client=anthropic_client, settings=settings, cost_per_check=0.05, max_usd=0.5)  # fmt: skip
    print(f"Sonnet standard API: {made} calls")
    batch_records = sonnet_collect.read_sonnet_records("smoke")
    standard_records = sonnet_collect.read_latency_records("smoke")
    return write_record(ran_at, free, digest, measurements, overhead, budget, schema, gate_settings,
                        jev_records, settings, batch_records, standard_records)  # fmt: skip


def write_record(
    ran_at: str,
    free: list[dict],
    digest: str,
    measurements: dict[str, dict],
    overhead: int,
    budget: int,
    schema: dict | None,
    gate_settings: jev_gate.JevGateSettings,
    jev_records: list[dict],
    settings: sonnet_gate.SonnetSettings,
    batch_records: list[dict],
    standard_records: list[dict],
) -> int:
    models = load_models()
    discount = float(models["batch_discount"])
    jev_price = jev.read_client_settings().price_per_million_input_tokens

    def list_cost(record: Mapping) -> float:
        return sonnet_gate.overhead_dollars(settings, sonnet_gate.parse_response(record["result"]["message"]))

    sonnet_list = task_means([(r["item_id"], list_cost(r)) for r in batch_records])
    sonnet_batch = {t: c * discount for t, c in sonnet_list.items()}
    jev_cost = task_means([(r["item_id"], jev.overhead_dollars(r["usage"]["input_tokens"], jev.read_client_settings()))
                           for r in jev_records])  # fmt: skip
    counts: dict[str, dict[str, int]] = {split: {} for split in ("train", "dev")}
    for split in counts:
        for row in read_rows(split=split):
            task = task_of(row["item_id"])
            counts[split][task] = counts[split].get(task, 0) + 1
    projection = project(counts, sonnet_batch=sonnet_batch, sonnet_list=sonnet_list, jev=jev_cost)
    measure_usd = sum(m["input_tokens"] for m in measurements.values()) * jev_price / 1e6
    recorded = gate_spend.recorded_spend()
    spend = {
        "jev_measurement_calls_usd": measure_usd, "jev_gate_calls_usd": recorded.jev,
        "sonnet_batch_usd_at_batch_price": recorded.sonnet_batch, "sonnet_standard_usd": recorded.sonnet_standard,
        "total_usd": measure_usd + recorded.total, "smoke_cap_usd": SMOKE_CAP_USD,
    }  # fmt: skip
    usage_in = sum(r["usage"]["input_tokens"] for r in [*batch_records, *standard_records])
    usage_out = sum(r["usage"]["output_tokens"] for r in [*batch_records, *standard_records])
    batch_ids = {r["batch_id"] for r in batch_records}
    by_id = {r["custom_id"]: r for r in batch_records}
    record = {
        "purpose": "A17 item 20.1's smoke test: made-up answers to real train prompts, before the settings commit",
        "ran_at": ran_at, "machine": platform.platform(),
        "answers_sha256": hashlib.sha256(ANSWERS_FILE.read_bytes()).hexdigest(),
        "tasks": sorted(counts["train"]),
        "image": {"name": gates.read_gate_settings().code_check.image, "digest": digest,
                  "dockerfile_sha256": hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest()},
        "measured": {"jev_budget_bytes": budget, "jev_fixed_overhead_tokens": overhead,
                     "jev_gate_questions_tokens": measurements["gate_questions"]["input_tokens"] - overhead,
                     "jev_longest_schema_tokens": measurements["longest_schema"]["input_tokens"] - overhead,
                     "jev_longest_schema_bytes": len(jev_call.dumps(schema)),
                     "sonnet_cost_per_check_list_usd": sonnet_list, "sonnet_cost_per_check_batch_usd": sonnet_batch,
                     "jev_cost_per_call_usd": jev_cost},
        "free_checks": {"expected_fails": [[*k, v] for k, v in EXPECTED_FREE_FAILS.items()],
                        "as_expected": not free_check_mismatches(free), "records": free},
        "jev": {"model": gate_settings.model, "measurements": measurements,
                "calls": len(jev_records), "refused": sum(r["refused"] for r in jev_records),
                "cut": sum(r["answer_cut_bytes"] > 0 or r["prompt_cut_bytes"] > 0 for r in jev_records),
                "records": jev_records},
        "sonnet": {"model": settings.model, "effort": settings.effort, "batch_discount": discount,
                   "batch_ids": sorted(batch_ids), "batch_log": [b for b in sonnet_collect.load_log(
                       sonnet_collect.batch_log_path())["batches"] if b["split"] == "smoke"],
                   "checks": len(batch_records), "refused": sum(r["refused"] for r in batch_records),
                   "tokens": {"input": usage_in, "output": usage_out},
                   "standard_calls": [{"custom_id": r["custom_id"], "seconds": r["seconds"],
                                       "first_call": r["first_call"],
                                       "standard_score": sonnet_gate.parse_response(r["result"]["message"]).score,
                                       "batch_score": sonnet_gate.parse_response(by_id[r["custom_id"]]["result"]["message"]).score}
                                      for r in standard_records],
                   "batch_records": batch_records, "standard_records": standard_records},
        "spend": spend, "projection": projection,
        "fixes": [],
    }  # fmt: skip
    RECORD_FILE.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"measured": record["measured"], "spend": spend, "projection": projection}, indent=1))
    return 0 if spend["total_usd"] <= SMOKE_CAP_USD else 3


if __name__ == "__main__":
    sys.exit(main())
