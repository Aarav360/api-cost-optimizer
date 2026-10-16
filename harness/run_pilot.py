"""Stage 0 pilot: run the pilot items through the model configs, score every answer, summarize.

Run from the project root:
  uv run python -m harness.run_pilot --configs haiku --per-source 3   # trial (Amendment A1, item 8)
  uv run python -m harness.run_pilot                                  # full pilot, both answer formats
  uv run python -m harness.run_pilot --pilot-file data/pilot_math_level5.jsonl --formats structured

Every item runs in free text, as its prompt was written. Banking77, MMLU-Pro,
MATH and CRUXEval also run as structured outputs (loaders.structured_variant),
so the two answer formats can be compared on the same items before one is
chosen for the collection run.

Calls use the standard API, because the pilot is where per-config latency is
measured, and go through the response store: a re-run only pays for calls not
already stored, and re-scoring after an evaluator fix costs nothing.

Writes one row per (item, format, config) to results/pilot/<pilot file
name>.rows.jsonl (e.g. pilot.rows.jsonl), answer text included for hand review, and prints what Stage 0 needs: accuracy and
format failures per task and config, the format comparison, token counts and
output lengths (for the max_tokens caps), latency, cost, and the projected
collection-run cost.
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from harness.client import PROJECT_ROOT, answer_text, build_request, call, cost_usd, load_models, make_client
from harness.evaluators import score
from harness.loaders import STRUCTURED_INSTRUCTIONS, structured_variant
from harness.store import ResponseStore

PILOT_FILE = PROJECT_ROOT / "data" / "pilot.jsonl"
STORE_ROOT = PROJECT_ROOT / "results" / "responses" / "pilot"
ROWS_DIR = PROJECT_ROOT / "results" / "pilot"
FORMATS = ("free", "structured")

# Generous, so no pilot answer is cut short: the collection run's per-config
# caps are set from these output lengths (Amendment A1, item 7). Any max_tokens
# stop is listed in the summary and means this cap was too low for that item.
PILOT_MAX_TOKENS = 32_000

# Collection-run shape (Amendment A1): train runs every config except the
# frontier, dev and test run all of them, all at the Batches API price.
FRONTIER = "opus"
TRAIN_SHARE, DEV_TEST_SHARE = 0.4, 0.6


def run_calls(jobs: list[tuple], store: ResponseStore, workers: int) -> tuple[dict, list]:
    """Return ({(item_id, format, config): (record, from_cache)}, [(item_id, format, config, error)])."""
    client = make_client()
    records, errors = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, client, store, request): (item, fmt, name) for item, fmt, name, request in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            item, fmt, name = futures[future]
            label = f"[{done}/{len(jobs)}] {name:8s} {fmt:10s} {item['item_id']:40s}"
            try:
                record, from_cache = future.result()
            except Exception as e:  # noqa: BLE001  (not stored, so a re-run retries it)
                errors.append((item["item_id"], fmt, name, repr(e)))
                print(f"{label} ERROR {e!r}", flush=True)
                continue
            records[(item["item_id"], fmt, name)] = (record, from_cache)
            response = record["response"]
            source = "stored" if from_cache else f"{record['latency_s']:5.1f}s"
            print(f"{label} {source:>7s} {response['stop_reason']:10s} out {response['usage']['output_tokens']:6d}",
                  flush=True)  # fmt: skip
    return records, errors


def make_row(item: dict, fmt: str, name: str, config: dict, record: dict, prices: dict) -> dict:
    response = record["response"]
    usage = response["usage"]
    result = score(item, response)
    return {
        "item_id": item["item_id"],
        "source": item["source"],
        "task_type": item["task_type"],
        "format": fmt,
        "config": name,
        "model": config["model"],
        "reported_model": response["model"],
        "effort": config["effort"],
        "stop_reason": response["stop_reason"],
        "refusal_category": (response.get("stop_details") or {}).get("category"),
        "status": result.status,
        "success": result.success,
        "quality": result.quality,
        "parsed": result.parsed,
        "detail": result.detail,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
        "cost_usd": cost_usd(record, prices),
        "latency_s": record["latency_s"],
        "request_id": record["request_id"],
        "answer": answer_text(response),
    }


def task_label(row: dict) -> str:
    return row["source"] if row["format"] == "free" else f"{row['source']} json"


def nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))]


def print_table(header: list[str], rows: list[list]) -> None:
    cells = [header] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    for i, row in enumerate(cells):
        print("  " + "  ".join(c.ljust(w) if j == 0 else c.rjust(w) for j, (c, w) in enumerate(zip(row, widths))))
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))


def summarize(rows: list[dict], configs: list[str], models: dict, rows_file) -> None:
    tasks = sorted(dict.fromkeys(task_label(row) for row in rows))
    cell = defaultdict(list)
    for row in rows:
        cell[(task_label(row), row["config"])].append(row)
        if row["format"] == "free":
            cell[("ALL free text", row["config"])].append(row)

    print("\nAccuracy (correct / items). Brackets: no answer in the requested format, other failures.")
    table = []
    for task in tasks + ["ALL free text"]:
        line = [task]
        for name in configs:
            group = cell[(task, name)]
            status = Counter(row["status"] for row in group)
            other = len(group) - status["correct"] - status["wrong"] - status["no_answer"]
            extra = f" [{status['no_answer']} fmt]" if status["no_answer"] else ""
            extra += f" [{other} stop]" if other else ""
            line.append(f"{status['correct']}/{len(group)}{extra}" if group else "-")
        table.append(line)
    print_table(["task"] + configs, table)

    print_format_comparison(rows, configs)

    print("\nPer config and format (cost at standard prices; latency from the original calls)")
    table = []
    for name in configs:
        for fmt in FORMATS:
            group = [row for row in rows if row["config"] == name and row["format"] == fmt]
            if not group:
                continue
            out = [row["output_tokens"] for row in group]
            thinking = [row["thinking_tokens"] or 0 for row in group]
            latency = [row["latency_s"] for row in group]
            mismatched = sum(row["reported_model"] != row["model"] for row in group)
            table.append([
                name,
                fmt,
                f"{statistics.mean(row['input_tokens'] for row in group):.0f}",
                f"{statistics.mean(out):.0f}",
                f"{statistics.median(out):.0f}",
                f"{max(out)}",
                f"{sum(thinking) / max(1, sum(out)):.0%}",
                f"{statistics.median(latency):.1f}s",
                f"{nearest_rank(latency, 0.9):.1f}s",
                f"${statistics.mean(row['cost_usd'] for row in group):.5f}",
                f"{mismatched}",
            ])  # fmt: skip
    print_table(["config", "format", "in mean", "out mean", "out med", "out max", "thinking", "lat med", "lat p90",
                 "$/item", "model mismatch"], table)  # fmt: skip

    print("\nOutput tokens per task, max (p95), for setting max_tokens caps")
    table = []
    for task in tasks:
        line = [task]
        for name in configs:
            out = [row["output_tokens"] for row in cell[(task, name)]]
            line.append(f"{max(out)} ({nearest_rank(out, 0.95)})" if out else "-")
        table.append(line)
    print_table(["task"] + configs, table)

    if set(configs) == set(models["configs"]):
        print_projection(cell, tasks, configs, models)

    stops = [row for row in rows if row["status"] not in ("correct", "wrong", "no_answer")]
    if stops:
        print("\nStops other than end_turn (scored as failures):")
        for row in stops:
            category = f" ({row['refusal_category']})" if row["refusal_category"] else ""
            print(f"  {row['config']:8s} {task_label(row):16s} {row['item_id']}: {row['status']}{category} "
                  f"at {row['output_tokens']} output tokens")  # fmt: skip

    spent = sum(row["cost_usd"] for row in rows)
    print(f"\nPilot calls in these rows cost ${spent:.2f} at standard prices (stored calls included).")
    print(f"Rows, with answers for hand review: {rows_file.relative_to(PROJECT_ROOT)}")


def print_format_comparison(rows: list[dict], configs: list[str]) -> None:
    by_key = {(row["item_id"], row["format"], row["config"]): row for row in rows}
    sources = sorted({row["source"] for row in rows if row["format"] == "structured"})
    if not sources or not any(row["format"] == "free" for row in rows):
        return
    print("\nFormat comparison on the same items: correct in free text -> correct as JSON.")
    print("Brackets: items right only in free text / only as JSON, then format failures free / JSON.")
    table = []
    for source in sources:
        line = [source]
        for name in configs:
            pairs = [
                (by_key[(item_id, "free", name)], by_key[(item_id, "structured", name)])
                for (item_id, fmt, config) in by_key
                if fmt == "structured" and config == name and by_key[(item_id, fmt, config)]["source"] == source
                and (item_id, "free", name) in by_key
            ]  # fmt: skip
            if not pairs:
                line.append("-")
                continue
            free_ok = sum(f["success"] for f, _ in pairs)
            json_ok = sum(s["success"] for _, s in pairs)
            only_free = sum(f["success"] and not s["success"] for f, s in pairs)
            only_json = sum(s["success"] and not f["success"] for f, s in pairs)
            free_fmt = sum(f["status"] == "no_answer" for f, _ in pairs)
            json_fmt = sum(s["status"] == "no_answer" for _, s in pairs)
            line.append(f"{free_ok}->{json_ok}/{len(pairs)} [{only_free}/{only_json}, fmt {free_fmt}/{json_fmt}]")
        table.append(line)
    print_table(["task"] + configs, table)


def print_projection(cell: dict, tasks: list[str], configs: list[str], models: dict) -> None:
    per_item = {}
    for task in tasks:
        mean = {name: statistics.mean(row["cost_usd"] for row in cell[(task, name)]) for name in configs}
        train = sum(cost for name, cost in mean.items() if name != FRONTIER)
        per_item[task] = models["batch_discount"] * (TRAIN_SHARE * train + DEV_TEST_SHARE * sum(mean.values()))

    print("\nProjected collection-run cost (Batches API; train without the frontier config, dev and test with it)")
    print_table(
        ["task", "$ per item", "600 items", "300 items"],
        [[t, f"${c:.5f}", f"${600 * c:.2f}", f"${300 * c:.2f}"] for t, c in per_item.items()],
    )
    free = sorted(c for t, c in per_item.items() if not t.endswith(" json"))
    if len(free) >= 4:
        print(f"  Plan size, 4 tasks x 600, free text: ${600 * sum(free[:4]):.2f} (4 cheapest) to "
              f"${600 * sum(free[-4:]):.2f} (4 most expensive)")  # fmt: skip
    if len(free) >= 3:
        print(f"  Small size, 3 tasks x 300, free text: ${300 * sum(free[:3]):.2f} to ${300 * sum(free[-3:]):.2f}")
    print("  Swap in a task's json row if that format is chosen. Model calls for the results matrix only; not")
    print("  included: the Sonnet verifier gate on dev answers, the noise-subset re-runs, Jev calls, and the")
    print("  Stage 8 live replay. Working ceiling: $150.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot-file", default=str(PILOT_FILE), help="pilot items (default: data/pilot.jsonl)")
    parser.add_argument("--configs", default=None, help="comma-separated config names (default: all)")
    parser.add_argument("--per-source", type=int, default=20, help="pilot items per dataset (default: all 20)")
    parser.add_argument("--formats", default=",".join(FORMATS), help="free, structured, or both (default)")
    parser.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    args = parser.parse_args()

    models = load_models()
    configs = args.configs.split(",") if args.configs else list(models["configs"])
    formats = args.formats.split(",")
    unknown = [name for name in configs if name not in models["configs"]] + [f for f in formats if f not in FORMATS]
    if unknown:
        parser.error(f"unknown config(s) or format(s): {', '.join(unknown)}")

    taken = Counter()
    items = []
    pilot_file = PROJECT_ROOT / args.pilot_file
    for line in pilot_file.open():
        item = json.loads(line)
        if taken[item["source"]] < args.per_source:
            taken[item["source"]] += 1
            items.append(item)

    variants = []  # (item as sent, format)
    for item in items:
        if "free" in formats:
            variants.append((item, "free"))
        if "structured" in formats and item["source"] in STRUCTURED_INSTRUCTIONS:
            variants.append((structured_variant(item), "structured"))

    store = ResponseStore(STORE_ROOT)
    jobs = [
        (item, fmt, name,
         build_request(models["configs"][name], item["prompt"], PILOT_MAX_TOKENS, output_schema=item.get("output_schema")))
        for name in configs
        for item, fmt in variants
    ]  # fmt: skip
    print(f"{len(variants)} item variants x {len(configs)} configs = {len(jobs)} calls (stored ones are free)")
    records, errors = run_calls(jobs, store, args.workers)

    # Scored here, on the main thread, after all calls finish (see evaluators.py).
    print("\nScoring...", flush=True)
    rows = [
        make_row(item, fmt, name, models["configs"][name], records[(item["item_id"], fmt, name)][0], models["prices"])
        for item, fmt in variants
        for name in configs
        if (item["item_id"], fmt, name) in records
    ]
    rows_file = ROWS_DIR / f"{pilot_file.stem}.rows.jsonl"
    rows_file.parent.mkdir(parents=True, exist_ok=True)
    rows_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    new_calls = sum(not from_cache for _, from_cache in records.values())
    print(f"{new_calls} new API calls this run, {len(records) - new_calls} served from the store")
    summarize(rows, configs, models, rows_file)
    if errors:
        print(f"\n{len(errors)} calls failed and were not stored; re-run to retry them:")
        for item_id, fmt, name, error in errors:
            print(f"  {name:8s} {fmt:10s} {item_id}: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
