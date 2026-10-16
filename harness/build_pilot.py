"""Draw the Stage 0 pilot items and write them to data/pilot.jsonl.

Run from the project root:  uv run python -m harness.build_pilot

Pilot items are kept out of train / dev / test (PLAN_v4.md §5), so Stage 1
reads both pilot files and excludes their IDs. Each source gets its own seeded
generator, so adding or dropping a source never changes another source's draw.
"""

import hashlib
import json
import random
import statistics

from harness.client import PROJECT_ROOT
from harness.loaders import load_all

PILOT_SEED = 20261015
ITEMS_PER_SOURCE = 20
OUTPUT = PROJECT_ROOT / "data" / "pilot.jsonl"

# Supplement (Amendment A3): the main draw's MATH items were mostly Levels 2-4,
# and every config solved all 20, so 20 Level 5 items test whether hard MATH
# still separates the configs. Its own file, so pilot.jsonl stays as it was.
LEVEL5_OUTPUT = PROJECT_ROOT / "data" / "pilot_math_level5.jsonl"


def write_items(path, items) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in items))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {len(items)} items to {path.relative_to(PROJECT_ROOT)} (sha256 {digest[:16]}...)")


def main() -> None:
    pools, dropped = load_all()
    pilot = []
    for source in sorted(pools):
        pool = sorted(pools[source], key=lambda x: x["item_id"])
        rng = random.Random(f"{PILOT_SEED}:{source}")
        for item in rng.sample(pool, ITEMS_PER_SOURCE):
            pilot.append({**item, "split": "pilot"})
        lengths = [len(x["prompt"]) for x in pilot[-ITEMS_PER_SOURCE:]]
        print(
            f"{source:10s} pool {len(pool):6d} | drew {ITEMS_PER_SOURCE} | "
            f"prompt chars median {statistics.median(lengths):5.0f}, max {max(lengths):5d}"
        )

    print()
    write_items(OUTPUT, pilot)

    drawn = {item["item_id"] for item in pilot}
    level5 = sorted(
        (x for x in pools["math"] if x["meta"]["level"] == "Level 5" and x["item_id"] not in drawn),
        key=lambda x: x["item_id"],
    )
    rng = random.Random(f"{PILOT_SEED}:math-level5")
    write_items(LEVEL5_OUTPUT, [{**item, "split": "pilot"} for item in rng.sample(level5, ITEMS_PER_SOURCE)])
    for item_id, reason in dropped.items():
        print(f"dropped from pools: {item_id}: {reason}")


if __name__ == "__main__":
    main()
