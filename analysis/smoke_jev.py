"""The Jev smoke test, before the settings commit (PLAN_v4.md §8 3c; Amendment A15 items 3, 8 and 13; issue #38).

Made-up requests only: `analysis/jev_smoke_requests.json` goes through G and G+ by the collection script into the
`smoke` split, with G+'s state carrying the real past requests so their tokens can be measured (item 13 allows that;
no train, dev or test request is sent as the request). Then a few measurement calls, each on a tiny made-up state, give
the fixed parts in Jev's reported tokens:

- the fixed overhead: a minimal call (a two-letter prompt, one short question), the largest of three. It counts the
  call's own few tokens too, which only errs safe (item 8);
- G's three questions, G+'s 18 questions, G+'s 16 past requests with their outcomes, and the longest output schema
  among train's and dev's requests (their requests, never an answer, are read for its size), each as that call's input
  tokens less the overhead.

`budget_bytes` for both routers then comes from `routers.jev_call`'s budget functions, with the overhead as measured
and the longest schema, and is written into the settings file. The smoke test's record, with the stored responses, goes
to `analysis/jev_smoke_record.json`. Nothing here changes a question, past request or setting because of an answer
(item 13): a mechanical failure stops the run, and the fix is made by hand and recorded in the record's `fixes`.

Run from the project root (the key in TYPESAFE_API_KEY, or in .env):  uv run python -m analysis.smoke_jev
Refuses to run if the record already exists. The measurement calls are kept as they come in
`results/jev/smoke_measurements.json` (uncommitted), so a stopped run resumes without repeating them.
"""

import hashlib
import json
import sys

from analysis import collect_jev
from analysis.benchmark import read_rows
from analysis.build_jev_settings import settings_text
from routers import jev, jev_call

PROJECT_ROOT = collect_jev.PROJECT_ROOT
REQUESTS_FILE = PROJECT_ROOT / "analysis" / "jev_smoke_requests.json"
RECORD_FILE = PROJECT_ROOT / "analysis" / "jev_smoke_record.json"
MEASUREMENTS_FILE = collect_jev.RESULTS_DIR / "jev" / "smoke_measurements.json"
PROVISIONAL_BUDGET = 2000  # bytes; far over the made-up prompts, so no cut fires, and replaced by the measured one
MINIMAL_REPEATS = 3
ONE_QUESTION = {"q": {"type": "noul", "instructions": "Is this a greeting?"}}


def longest_schema() -> dict | None:
    """The train or dev request output schema with the most serialized bytes (the reader refuses test)."""
    schemas = [row["output_schema"] for row in read_rows() if row.get("output_schema")]
    return max(schemas, key=lambda schema: len(jev_call.dumps(schema)), default=None)


def question_bytes(questions: dict[str, dict]) -> int:
    return sum(len(jev_call.dumps({key: question})) for key, question in questions.items())


def measurement_bodies(settings: jev_call.Settings, schema: dict | None) -> dict[str, tuple[bytes, dict]]:
    """Label -> (body, what its input tokens less the overhead stand for, in bytes). Tiny made-up states throughout."""

    def body(state: dict, questions: dict[str, dict]) -> bytes:
        return jev_call.dumps({"model": settings.model, "state": state, "questions": questions})

    tiny = jev_call.g_state("Hi", None)
    g_questions, plus_questions = jev_call.g_questions(settings), jev_call.g_plus_questions(settings)
    past_state = jev_call.g_plus_state("Hi", None, settings)
    schema_state = jev_call.g_state("Hi", schema)
    extra = {"state": len(jev_call.dumps(past_state)) - len(jev_call.dumps(tiny))}
    bodies: dict[str, tuple[bytes, dict]] = {
        f"minimal_{n}": (body(tiny, ONE_QUESTION), {}) for n in range(1, MINIMAL_REPEATS + 1)
    }
    bodies["g_questions"] = (
        body(tiny, g_questions),
        {"questions": question_bytes(g_questions) - question_bytes(ONE_QUESTION)},
    )
    bodies["g_plus_questions"] = (body(tiny, plus_questions), {"questions": question_bytes(plus_questions) - question_bytes(ONE_QUESTION)})  # fmt: skip
    bodies["g_plus_past_requests"] = (body(past_state, ONE_QUESTION), extra)
    bodies["longest_schema"] = (body(schema_state, ONE_QUESTION), {"state": len(jev_call.dumps(schema_state)) - len(jev_call.dumps(tiny))})  # fmt: skip
    return bodies


def measure(settings: jev_call.Settings, loaded: collect_jev.Loaded, schema: dict | None) -> dict[str, dict]:
    """Send each measurement body once (resuming from the kept file) and return label -> what came back."""
    done = json.loads(MEASUREMENTS_FILE.read_text(encoding="utf-8")) if MEASUREMENTS_FILE.exists() else {}
    with jev.JevClient(loaded.client) as client:
        for label, (body, grows) in measurement_bodies(settings, schema).items():
            if label in done:
                continue
            result = client.call(body, jev_call.body_sha256(body), f"smoke measurement {label}")
            usage = result.usage or {}
            done[label] = {
                "body_bytes": len(body), "body_added_bytes": grows, "input_tokens": usage.get("input_tokens"),
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
    loaded = collect_jev.load_settings()
    settings = loaded.call
    schema = longest_schema()
    items = collect_jev.read_smoke_requests(REQUESTS_FILE)
    print(f"{len(items)} made-up requests; longest schema {len(jev_call.dumps(schema))} bytes")
    with jev.JevClient(loaded.client) as client:
        for router in collect_jev.ROUTERS:
            summary = collect_jev.collect(
                items, router=router, split="smoke", loaded=loaded, client=client,
                budget_bytes=PROVISIONAL_BUDGET, progress=collect_jev.show_progress,
            )  # fmt: skip
            print(
                f"{router}: {summary.new} new, {summary.skipped} stored, {summary.refused} refused, {summary.cut} cut"
            )
    measurements = measure(settings, loaded, schema)

    overhead = max(measurements[f"minimal_{n}"]["input_tokens"] for n in range(1, MINIMAL_REPEATS + 1))
    parts = {
        label: measurements[label]["input_tokens"] - overhead
        for label in ("g_questions", "g_plus_questions", "g_plus_past_requests", "longest_schema")
    }
    budgets = {
        "jev": jev_call.g_budget_bytes(settings, overhead_tokens=overhead, schema=schema),
        "jev_bolstered": jev_call.g_plus_budget_bytes(settings, overhead_tokens=overhead, schema=schema),
    }
    stored = {router: collect_jev.read_records(router, "smoke") for router in collect_jev.ROUTERS}
    smoke_usage = [r["usage"] for records in stored.values() for r in records]
    all_usage = smoke_usage + [{"input_tokens": m["input_tokens"], "output_tokens": m["output_tokens"]} for m in measurements.values()]  # fmt: skip
    record = {
        "purpose": "A15 item 13's smoke test: made-up requests only, before the settings commit",
        "model": settings.model,
        "made_up_requests_sha256": hashlib.sha256(REQUESTS_FILE.read_bytes()).hexdigest(),
        "calls": {"smoke": len(smoke_usage), "measurement": len(measurements), "all": len(all_usage)},
        "tokens": {"input": sum(u["input_tokens"] for u in all_usage), "output": sum(u["output_tokens"] for u in all_usage)},
        "fixed_overhead_tokens": overhead,
        "fixed_parts_tokens": parts,
        "longest_schema_bytes": len(jev_call.dumps(schema)),
        "budget_bytes": budgets,
        "measurements": measurements,
        "smoke_records": stored,
        "fixes": [],
    }  # fmt: skip
    RECORD_FILE.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    data = json.loads(jev_call.SETTINGS_FILE.read_text(encoding="utf-8"))
    data["budget_bytes"] = budgets
    jev_call.SETTINGS_FILE.write_text(settings_text(data), encoding="utf-8")
    print(f"overhead {overhead} tokens, parts {parts}, budgets {budgets}")
    print(f"{record['calls']['all']} calls, {record['tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
