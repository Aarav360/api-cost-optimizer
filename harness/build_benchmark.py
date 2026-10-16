"""Stage 1: draw the benchmark items and split them 40/30/30 (PLAN_v4.md §5, Amendment A5).

Run from the project root:  uv run python -m harness.build_benchmark

Writes data/benchmark.jsonl (every item, with its split) and
data/benchmark_manifest.json (counts, groups, the noise subset and hashes). The
draw is seeded, so a re-run reproduces both files byte for byte, and the
manifest's hashes are what Stage 8 checks before scoring the test split.

- Tasks (Amendment A3): IFEval and MBPP+ use every item outside the pilots;
  Banking77 and MMLU-Pro draw 800 each. Banking77 and MMLU-Pro items are stored
  in their structured-output form (loaders.structured_variant), as sent.
- Items in any data/pilot*.jsonl file are excluded.
- Items whose question text is the same after normalizing case, punctuation and
  spacing share a source_group, and a group never straddles splits.
- Each task is split 40/30/30 into train, dev and test by group, stratified by
  Banking77 intent and MMLU-Pro subject so every split has the same mix. In
  Stage 5 the trained router cross-fits over dev instead of taking a larger
  train share.
- About 10% of each task's dev items form the noise subset, re-run three times
  in Stage 2 to measure run-to-run variation (§6).
- The manifest also records the hash of configs/task_mix.yaml, the two task
  mixes fixed alongside the splits.
"""

import hashlib
import json
import random
import re
from collections import defaultdict

import yaml

from harness.client import PROJECT_ROOT
from harness.loaders import EXCLUDED, load_banking77, load_ifeval, load_mbppplus, load_mmlu_pro, structured_variant

SEED = 20261016
DATA_DIR = PROJECT_ROOT / "data"
BENCHMARK_FILE = DATA_DIR / "benchmark.jsonl"
MANIFEST_FILE = DATA_DIR / "benchmark_manifest.json"
TASK_MIX_FILE = PROJECT_ROOT / "configs" / "task_mix.yaml"

# Task: (loader, items to draw or None for every available item, answer format)
TASKS = {
    "ifeval": (load_ifeval, None, "free"),
    "banking77": (load_banking77, 800, "structured"),
    "mmlu_pro": (load_mmlu_pro, 800, "structured"),
    "mbppplus": (load_mbppplus, None, "free"),
}
SPLIT_SHARES = {"train": 0.4, "dev": 0.3, "test": 0.3}
NOISE_SHARE = 0.10

# What each task's splits are balanced on. Tasks not listed are split as one stratum.
STRATA = {
    "banking77": lambda item: item["expected_answer"],
    "mmlu_pro": lambda item: item["meta"]["category"],
}


def question_text(item: dict) -> str:
    """The part of the prompt that identifies the question, without the template around it."""
    prompt = item["prompt"]
    if item["source"] == "banking77":
        return prompt.split("\n\nQuery: ", 1)[1].split("\n\n", 1)[0]
    if item["source"] == "mmlu_pro":
        return prompt.split("\n\n", 1)[1].split("\n\nA. ", 1)[0]
    if item["source"] == "mbppplus":
        return prompt.split("\n\nTask: ", 1)[1].split("\n\n", 1)[0]
    return prompt


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def group_items(items: list[dict]) -> list[list[dict]]:
    """Groups of items with the same normalized question, each sorted, in item-ID order."""
    by_key = defaultdict(list)
    for item in sorted(items, key=lambda x: x["item_id"]):
        by_key[normalize(question_text(item))].append(item)
    groups = sorted(by_key.values(), key=lambda g: g[0]["item_id"])
    for group in groups:
        for item in group:
            item["source_group"] = group[0]["item_id"]
    return groups


def draw(groups: list[list[dict]], target: int, rng: random.Random) -> list[list[dict]]:
    """Whole groups, in seeded random order, until exactly `target` items (or as close as groups allow)."""
    order = groups[:]
    rng.shuffle(order)
    drawn, total = [], 0
    for group in order:
        if total + len(group) <= target:
            drawn.append(group)
            total += len(group)
        if total == target:
            break
    return sorted(drawn, key=lambda g: g[0]["item_id"])


def split(groups: list[list[dict]], rng: random.Random, stratum=None) -> dict[str, list[dict]]:
    """Assign whole groups to train / dev / test, 40/30/30 within each stratum.

    Quotas use cumulative rounding across strata (taken in sorted order), so each
    stratum is split as close to 40/30/30 as its size allows and the task's totals
    still come out at the overall shares.
    """
    strata = defaultdict(list)
    for group in groups:
        strata[stratum(group[0]) if stratum else ""].append(group)
    names = list(SPLIT_SHARES)
    splits = {name: [] for name in names}
    seen = 0
    for key in sorted(strata):
        order = strata[key][:]
        rng.shuffle(order)
        before = {name: len(splits[name]) for name in names}
        seen += sum(len(g) for g in order)
        # Cumulative boundaries after this stratum: train ends at 40%, dev at 70%.
        bounds = {"train": round(SPLIT_SHARES["train"] * seen), "dev": round((SPLIT_SHARES["train"] + SPLIT_SHARES["dev"]) * seen)}
        quota = {"train": bounds["train"] - before["train"], "dev": bounds["dev"] - bounds["train"] - before["dev"]}
        quota["test"] = sum(len(g) for g in order) - quota["train"] - quota["dev"]
        taken = dict.fromkeys(names, 0)
        for group in order:
            # The first split with room for the whole group, else the one with the most room left.
            name = next((s for s in names if quota[s] - taken[s] >= len(group)), None)
            name = name or max(names, key=lambda s: quota[s] - taken[s])
            taken[name] += len(group)
            for item in group:
                splits[name].append({**item, "split": name})
    return {name: sorted(items, key=lambda x: x["item_id"]) for name, items in splits.items()}


def main() -> None:
    mixes = yaml.safe_load(TASK_MIX_FILE.read_text())
    for mix, weights in mixes.items():
        if set(weights) != set(TASKS) or abs(sum(weights.values()) - 1) > 1e-9:
            raise ValueError(f"task mix {mix!r} must weight exactly {sorted(TASKS)} and sum to 1")

    pilot_files = sorted(DATA_DIR.glob("pilot*.jsonl"))
    pilot_ids = {json.loads(line)["item_id"] for path in pilot_files for line in path.open()}

    benchmark, tasks_manifest, noise_subset = [], {}, []
    for task, (loader, target, answer_format) in TASKS.items():
        pool = [x for x in loader() if x["item_id"] not in pilot_ids and x["item_id"] not in EXCLUDED]
        groups = group_items(pool)
        if target is not None:
            groups = draw(groups, target, random.Random(f"{SEED}:draw:{task}"))
        splits = split(groups, random.Random(f"{SEED}:split:{task}"), STRATA.get(task))

        dev_ids = [x["item_id"] for x in splits["dev"]]
        noise = sorted(random.Random(f"{SEED}:noise:{task}").sample(dev_ids, round(NOISE_SHARE * len(dev_ids))))
        noise_subset += noise

        for name in SPLIT_SHARES:
            for item in splits[name]:
                benchmark.append(structured_variant(item) if answer_format == "structured" else item)
        duplicates = [g for g in groups if len(g) > 1]
        tasks_manifest[task] = {
            "available": len(pool),
            "drawn": sum(len(g) for g in groups),
            "answer_format": answer_format,
            "splits": {name: len(items) for name, items in splits.items()},
            "noise_subset": len(noise),
            "duplicate_groups": len(duplicates),
            "items_in_duplicate_groups": sum(len(g) for g in duplicates),
            "stratified_by": {"banking77": "intent", "mmlu_pro": "subject"}.get(task),
        }

    BENCHMARK_FILE.write_text("".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in benchmark))
    ids_by_split = {name: sorted(x["item_id"] for x in benchmark if x["split"] == name) for name in SPLIT_SHARES}
    manifest = {
        "built_by": "harness/build_benchmark.py",
        "seed": SEED,
        "split_shares": SPLIT_SHARES,
        "excluded_pilot_items": len(pilot_ids),
        "pilot_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in pilot_files},
        "tasks": tasks_manifest,
        "noise_subset": noise_subset,
        "task_mixes": mixes,
        "sha256": {
            "benchmark.jsonl": hashlib.sha256(BENCHMARK_FILE.read_bytes()).hexdigest(),
            "configs/task_mix.yaml": hashlib.sha256(TASK_MIX_FILE.read_bytes()).hexdigest(),
            **{f"{name}_ids": sha256_text("\n".join(ids)) for name, ids in ids_by_split.items()},
        },
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"{'task':10s} {'available':>9s} {'drawn':>6s} {'train':>6s} {'dev':>5s} {'test':>5s} {'noise':>6s} {'dup groups':>11s}")
    for task, t in tasks_manifest.items():
        s = t["splits"]
        print(f"{task:10s} {t['available']:9d} {t['drawn']:6d} {s['train']:6d} {s['dev']:5d} {s['test']:5d} "
              f"{t['noise_subset']:6d} {t['duplicate_groups']:11d}")  # fmt: skip
    print(f"\n{len(benchmark)} items -> {BENCHMARK_FILE.relative_to(PROJECT_ROOT)} "
          f"({BENCHMARK_FILE.stat().st_size / 1e6:.1f} MB, sha256 {manifest['sha256']['benchmark.jsonl'][:16]}...)")  # fmt: skip
    print(f"manifest -> {MANIFEST_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
