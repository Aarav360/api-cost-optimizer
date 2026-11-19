"""Dev's flip rates, which carry over to test's references (PLAN_v4.md Amendment A19, item 8; A6 item 2; A8 item 4; issue #88).

A **flip rate** is the share of (item, config) pairs in dev's noise subset whose success differs across the three runs (A6
item 2), per config and per task. Test has run 0 only, so it has no flip rates of its own: dev's carry over, and so do A6
item 2's words. If any config's rate is 5% or more, the oracle curve is described as an optimistic upper bound (the oracle
sends each item to the cheapest config that succeeded once, so a lucky draw counts as savings no deployable policy could rely
on). On dev that holds for `haiku` and `opus_low` (A6), so the test oracle is described the same way.

Dev's rates are counted from dev's stored runs once, by the command below, into
`analysis/dev_flip_rates.json`, and the offline half's `flip_rates` part reads that file on either split: nothing about test
goes into it, and a test build needs no dev results. `committed` recounts them from dev's matrix, so the dev reproduction holds
the file to dev's runs (hook only), and `analysis/check_target_met.py` holds the file to A6's numbers in the plan (CI).

Run from the project root:  uv run python -m analysis.flip_rates --write
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from analysis.matrix import read_store
from analysis.pareto import ORACLE_LABEL
from harness.client import PROJECT_ROOT, load_models
from harness.collect import MAIN

FLIP_LIMIT = 0.05  # A6 item 2: at or over this, the oracle curve is an optimistic upper bound
DEV_RATES = PROJECT_ROOT / "analysis" / "dev_flip_rates.json"
MANIFEST = PROJECT_ROOT / "data" / "benchmark_manifest.json"
RUNS = {0, 1, 2}
WORDS = "optimistic upper bound"


class FlipRatesError(ValueError):
    """Stored runs or a carried-over file that don't make flip rates."""


def counts(rows: Iterable[Mapping[str, Any]], noise: set[str], configs: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Flips and pairs per config and per task, from stored rows: a pair is an (item, config) of the noise subset that has all
    three runs (A6 item 1), and it flipped if its three successes are not all one. Tasks in the order they first appear."""
    runs: defaultdict[tuple[str, str, str], dict[int, int]] = defaultdict(dict)
    for row in rows:
        if row["item_id"] in noise:
            runs[(row["item_id"], row["config"], row["source"])][row["run"]] = int(row["success"])
    found: dict[str, dict[str, Any]] = {config: {"flips": 0, "pairs": 0, "tasks": {}} for config in configs}
    for (_, config, source), by_run in sorted(runs.items()):
        if set(by_run) != RUNS or config not in found:
            continue
        flipped = len(set(by_run.values())) > 1
        for place in (found[config], found[config]["tasks"].setdefault(source, {"flips": 0, "pairs": 0})):
            place["pairs"] += 1
            place["flips"] += flipped
    return found


def with_rates(found: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The counts with each rate (flips over pairs) beside them."""
    out: dict[str, dict[str, Any]] = {}
    for config, held in found.items():
        if not held["pairs"]:
            raise FlipRatesError(f"{config} has no (item, config) pair with all three runs")
        tasks = {task: {**t, "rate": t["flips"] / t["pairs"]} for task, t in sorted(held["tasks"].items())}
        out[config] = {
            "flips": held["flips"],
            "pairs": held["pairs"],
            "rate": held["flips"] / held["pairs"],
            "tasks": tasks,
        }
    return out


def dev_rates() -> dict[str, dict[str, Any]]:
    """Dev's flip rates, counted from dev's stored runs (the dev results matrix, which holds runs 0 to 2 of the noise subset)."""
    noise = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["noise_subset"])
    rows = [row for row in read_store(MAIN.matrix) if row["split"] == "dev"]
    return with_rates(counts(rows, noise, list(load_models()["configs"])))


def carried() -> dict[str, dict[str, Any]]:
    """The flip rates the file holds, for the part."""
    held: dict[str, dict[str, Any]] = json.loads(DEV_RATES.read_text(encoding="utf-8"))["configs"]
    return held


def words(rates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A6 item 2's reading of the rates: the configs at or over the limit, and the oracle's description if any is."""
    over = [config for config, held in rates.items() if held["rate"] >= FLIP_LIMIT]
    return {
        "limit": FLIP_LIMIT,
        "over_limit": over,
        "oracle": ORACLE_LABEL,
        "oracle_described_as": WORDS if over else None,
    }


def part_of(rates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "from": "dev's noise subset, runs 0 to 2 (A6 item 2); test has run 0 only, so these carry over",
        "configs": dict(rates),
        **words(rates),
    }


def part(inputs: Any) -> dict[str, Any]:
    """The flip-rates part: dev's rates from the committed file, on either split."""
    return part_of(carried())


def committed() -> dict[str, Any]:
    """What dev's own stored runs give, for the dev reproduction to hold the file to."""
    return part_of(dev_rates())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help=f"write {DEV_RATES.name}")
    args = parser.parse_args()
    record: dict[str, Any] = {"purpose": "Dev's flip rates (A6 item 2), counted from dev's noise subset; test's references carry them over.",
              "built_by": "analysis/flip_rates.py", "limit": FLIP_LIMIT, "configs": dev_rates()}  # fmt: skip
    text = json.dumps(record, indent=2) + "\n"
    if args.write:
        DEV_RATES.write_text(text, encoding="utf-8")
        print(f"wrote {DEV_RATES}")
    for config, held in record["configs"].items():
        print(f"{config}: {held['flips']}/{held['pairs']} {held['rate']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
