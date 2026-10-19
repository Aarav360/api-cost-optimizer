"""Stage 2 review of the scored train/dev matrix (PLAN_v4.md §14; Amendment A6).

Run from the project root, after `harness.collect score`:
  uv run python -m harness.review_collection           # the collection run
  uv run python -m harness.review_collection --smoke   # the smoke run (no noise subset)

Reads only results/dev_matrix.sqlite (train and dev), never test. Prints:

- stops other than end_turn (refusals with their category, output-cap hits) and
  model-ID mismatches, per split and config;
- the noise measure fixed in Amendment A6, item 2: the share of noise-subset
  (item, config) pairs whose success differs across runs 0-2, per config and task,
  and whether any config reaches the 5% that marks the oracle as an optimistic
  upper bound.

It also writes the hand-check sample for the evaluator review (§14): every run-0
answer with no answer in the requested format, plus 20 seeded random run-0 wrong
answers per task, with the expected answer and the grader's reading, to
results/collection/review_sample.jsonl (or the smoke folder's).
"""

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing

from harness.build_benchmark import BENCHMARK_FILE, MANIFEST_FILE
from harness.client import load_models
from harness.collect import MAIN, SMOKE, PILOT_FILE
from harness.run_pilot import print_table

FLIP_LIMIT = 0.05  # Amendment A6, item 2
WRONG_PER_TASK = 20
SEED = 20261018


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="review the smoke run instead")
    args = parser.parse_args()
    paths = SMOKE if args.smoke else MAIN
    configs = list(load_models()["configs"])

    with closing(sqlite3.connect(paths.matrix.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute("SELECT * FROM rows")]
    if any(row["split"] not in ("train", "dev") for row in rows):
        raise SystemExit("the matrix holds rows outside train and dev")
    print(f"{len(rows)} rows: " + ", ".join(f"{n} {s}" for s, n in sorted(Counter(r['split'] for r in rows).items())))

    print("\nStops other than end_turn, and model-ID mismatches (all runs)")
    stops = Counter()
    for row in rows:
        if row["stop_reason"] != "end_turn":
            category = f" ({row['refusal_category']})" if row["refusal_category"] else ""
            stops[(row["split"], row["config"], row["source"], f"{row['stop_reason']}{category}")] += 1
        if row["reported_model"] != row["model"]:
            stops[(row["split"], row["config"], row["source"], f"reported {row['reported_model']}")] += 1
    if stops:
        print_table(["split", "config", "task", "stop", "rows"], [[*key, n] for key, n in sorted(stops.items())])
    else:
        print("  none")

    noise = set(json.loads(MANIFEST_FILE.read_text())["noise_subset"]) if not args.smoke else set()
    if noise:
        print_noise(rows, noise, configs)
    write_sample(rows, paths.state.parent / "review_sample.jsonl", args.smoke)
    return 0


def print_noise(rows: list[dict], noise: set[str], configs: list[str]) -> None:
    runs = defaultdict(dict)  # (item, config) -> {run: success}
    for row in rows:
        if row["item_id"] in noise:
            runs[(row["item_id"], row["config"], row["source"])][row["run"]] = row["success"]
    complete = {key: r for key, r in runs.items() if set(r) == {0, 1, 2}}
    print(f"\nNoise subset: {len(complete)} of {len(noise) * len(configs)} (item, config) pairs have all three runs.")
    print("Flip rate: pairs whose success differs across runs 0-2 (Amendment A6, item 2).")

    flips, pairs = Counter(), Counter()
    for (_, config, source), r in complete.items():
        flipped = len(set(r.values())) > 1
        for key in ((config, source), (config, "ALL")):
            pairs[key] += 1
            flips[key] += flipped
    sources = sorted({source for _, _, source in complete}) + ["ALL"]
    table = [[source] + [f"{flips[(c, source)]}/{pairs[(c, source)]}" + (f" {flips[(c, source)] / pairs[(c, source)]:.0%}"
             if pairs[(c, source)] else "") for c in configs] for source in sources]  # fmt: skip
    print_table(["task"] + configs, table)
    over = [c for c in configs if pairs[(c, "ALL")] and flips[(c, "ALL")] / pairs[(c, "ALL")] >= FLIP_LIMIT]
    if over:
        print(f"At or over {FLIP_LIMIT:.0%}: {', '.join(over)}. The oracle curve is reported as an optimistic upper bound.")
    else:
        print(f"Every config is under {FLIP_LIMIT:.0%}.")


def write_sample(rows: list[dict], path, smoke: bool) -> None:
    source_file = PILOT_FILE if smoke else BENCHMARK_FILE
    expected = {item["item_id"]: item["expected_answer"] for item in map(json.loads, source_file.open())}
    first = [row for row in rows if row["run"] == 0]
    sample = [row for row in first if row["status"] == "no_answer"]
    rng = random.Random(SEED)
    for source in sorted({row["source"] for row in first}):
        wrong = sorted((row for row in first if row["source"] == source and row["status"] == "wrong"),
                       key=lambda row: row["custom_id"])  # fmt: skip
        sample += rng.sample(wrong, min(WRONG_PER_TASK, len(wrong)))
    fields = ("custom_id", "split", "source", "config", "status", "parsed", "detail", "answer")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for row in sample:
            out.write(json.dumps({**{k: row[k] for k in fields}, "expected": expected[row["item_id"]]}) + "\n")
    kinds = Counter((row["source"], row["status"]) for row in sample)
    print(f"\nHand-check sample: {len(sample)} answers in {path}")
    print("  " + ", ".join(f"{source} {status} {n}" for (source, status), n in sorted(kinds.items())))


if __name__ == "__main__":
    sys.exit(main())
