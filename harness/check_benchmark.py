"""Check the frozen benchmark (Amendment A5) before it is used.

Run from the project root:  uv run python -m harness.check_benchmark

Run before the Stage 2 collection run, and in Stage 8 before the test split is
scored. Exits nonzero on any problem:

- data/benchmark.jsonl, its split ID lists and configs/task_mix.yaml match the
  hashes in data/benchmark_manifest.json, and the pilot files are unchanged;
- no pilot item is in the benchmark, item IDs are unique, and no source_group
  straddles splits;
- Banking77 and MMLU-Pro items carry their JSON schema and the others don't;
- the noise subset lies within dev;
- every Banking77 and MMLU-Pro reference answer scores correct, and every IFEval
  item's instructions build. MBPP+ reference solutions are covered by
  check_sandbox (and mbppplus:255 by check_evaluators' memory override).
"""

import collections
import hashlib
import json
import sys

from harness.build_benchmark import BENCHMARK_FILE, DATA_DIR, MANIFEST_FILE, SPLIT_SHARES, TASK_MIX_FILE
from harness.client import PROJECT_ROOT
from harness.evaluators import score


def reply(text: str) -> dict:
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


def main() -> int:
    manifest = json.loads(MANIFEST_FILE.read_text())
    items = [json.loads(line) for line in BENCHMARK_FILE.open()]
    problems = []

    hashes = manifest["sha256"]
    if hashlib.sha256(BENCHMARK_FILE.read_bytes()).hexdigest() != hashes["benchmark.jsonl"]:
        problems.append("benchmark.jsonl does not match its manifest hash")
    if hashlib.sha256(TASK_MIX_FILE.read_bytes()).hexdigest() != hashes["configs/task_mix.yaml"]:
        problems.append("configs/task_mix.yaml does not match its manifest hash")
    for name in SPLIT_SHARES:
        ids = "\n".join(sorted(x["item_id"] for x in items if x["split"] == name))
        if hashlib.sha256(ids.encode("utf-8")).hexdigest() != hashes[f"{name}_ids"]:
            problems.append(f"{name} split IDs do not match their manifest hash")
    for name, digest in manifest["pilot_files"].items():
        if hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() != digest:
            problems.append(f"pilot file {name} changed since the benchmark was built")

    ids = [x["item_id"] for x in items]
    pilot = {json.loads(line)["item_id"] for name in manifest["pilot_files"] for line in (DATA_DIR / name).open()}
    if len(ids) != len(set(ids)):
        problems.append("duplicate item IDs")
    if pilot & set(ids):
        problems.append(f"pilot items in the benchmark: {sorted(pilot & set(ids))[:5]}")
    splits_by_group = collections.defaultdict(set)
    for x in items:
        splits_by_group[x["source_group"]].add(x["split"])
    straddling = [g for g, s in splits_by_group.items() if len(s) > 1]
    if straddling:
        problems.append(f"groups across splits: {straddling[:5]}")
    for x in items:
        if (x["source"] in ("banking77", "mmlu_pro")) != ("output_schema" in x):
            problems.append(f"{x['item_id']}: wrong answer format")
    dev = {x["item_id"] for x in items if x["split"] == "dev"}
    if not set(manifest["noise_subset"]) <= dev:
        problems.append("noise subset reaches outside dev")

    for x in items:
        if x["source"] == "banking77":
            result = score(x, reply(json.dumps({"answer": x["expected_answer"]})))
        elif x["source"] == "mmlu_pro":
            result = score(x, reply(json.dumps({"explanation": "-", "answer": x["expected_answer"]})))
        elif x["source"] == "ifeval":
            score(x, reply("x"))  # raises if an instruction can't be built
            continue
        else:
            continue
        if not result.success:
            problems.append(f"{x['item_id']}: reference answer scores {result.status}")

    counts = collections.Counter((x["source"], x["split"]) for x in items)
    for task in manifest["tasks"]:
        print(f"{task:10s} " + "  ".join(f"{s} {counts[(task, s)]:4d}" for s in SPLIT_SHARES))
    print(f"{len(items)} items, {len(manifest['noise_subset'])} in the noise subset, "
          f"benchmark sha256 {hashes['benchmark.jsonl'][:16]}...")  # fmt: skip
    for problem in problems[:20]:
        print(f"PROBLEM  {problem}")
    print("all checks passed" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
