"""Draw the Stage 0 pilot items and write them to data/pilot.jsonl.

Run from the project root:  uv run python -m harness.build_pilot

Pilot items are kept out of train / dev / test (PLAN_v4.md §5), so Stage 1
reads this file and excludes these IDs. Each source gets its own seeded
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

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in pilot))
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"\nwrote {len(pilot)} items to {OUTPUT.relative_to(PROJECT_ROOT)} (sha256 {digest[:16]}...)")
    for item_id, reason in dropped.items():
        print(f"dropped from pools: {item_id}: {reason}")


if __name__ == "__main__":
    main()
