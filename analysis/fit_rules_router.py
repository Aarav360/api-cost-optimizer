"""Fit the rules router on train and write its rates file (PLAN_v4.md §8 3a; Amendment A9, items 4-6 and 13).

Run from the project root:  uv run python -m analysis.fit_rules_router

Reads the benchmark's train requests through the reader and the train cells of the results matrix (run 0, `haiku`,
`sonnet` and `opus_low`: train has no `opus` cells, Amendment A1), fits the rates and writes `routers/rules_rates.json`.
Then it prints the train task-inference diagnostic, the router's latency over the dev requests and the freeze hashes.

Nothing here computes an E number on dev (item 13): the dev requests are only timed, and the router scores that
timing produces are thrown away. The dev router-score table, the dev diagnostic and the walk come after the freeze.
"""

import json
import platform
import subprocess
import time

import numpy as np

from analysis import matrix as results_matrix
from analysis.benchmark import read_rows
from analysis.freeze import RATES, freeze_hashes
from harness.client import PROJECT_ROOT
from routers import rules
from routers.ladder import SCORED
from routers.request import Request, from_row

# Each task's own type. This is the benchmark's knowledge, so it lives here and never in the router.
TASK_TYPE = {"banking77": rules.CLASSIFICATION, "mmlu_pro": rules.MULTIPLE_CHOICE, "mbppplus": rules.CODE,
             "ifeval": rules.OPEN_ENDED}  # fmt: skip
WARM_UP = 50  # calls made before the timing starts, so the first call's setup isn't timed


def train_inputs() -> tuple[dict[str, Request], dict[str, dict[str, float]], dict[str, str]]:
    """Train requests, their run-0 successes for the scored configs, and each item's task (for the diagnostic only)."""
    rows = [row for row in read_rows() if row["split"] == "train"]
    train = results_matrix.load().split("train")
    successes = {row["item_id"]: {c: train.cells[(row["item_id"], c)].success for c in SCORED} for row in rows}
    return ({row["item_id"]: from_row(row) for row in rows}, successes, {row["item_id"]: row["source"] for row in rows})


def rates_text(rates: dict) -> str:
    return json.dumps(rates, indent=2, sort_keys=True) + "\n"


def diagnostic(requests: dict[str, Request], tasks: dict[str, str], split: str = "train") -> str:
    """How often the inferred type is the item's task, with the output schema and with it withheld."""
    n = len(requests)
    full = sum(rules.infer_type(request) == TASK_TYPE[tasks[i]] for i, request in requests.items())
    bare = sum(rules.infer_type(Request(request.prompt)) == TASK_TYPE[tasks[i]] for i, request in requests.items())
    return (f"On this benchmark's templates the inferred type is the item's task for {full} of {n} {split} items "
            f"({100 * full / n:.1f}%), and for {bare} of {n} ({100 * bare / n:.1f}%) with the output schema withheld")  # fmt: skip


def machine() -> str:
    cpu = ""
    if platform.system() == "Darwin":
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
        )
        cpu = f"{brand.stdout.strip()}, "
    return f"{cpu}{platform.system()} {platform.release()} {platform.machine()}, Python {platform.python_version()}"


def latency(dev_requests: list[Request]) -> tuple[float, float]:
    """Median and p90 of one `score(request)` call in microseconds, over the dev requests. The router scores it
    returns are dropped on the spot (item 13)."""
    for request in dev_requests[:WARM_UP]:
        rules.score(request)
    times = []
    for request in dev_requests:
        start = time.perf_counter_ns()
        rules.score(request)
        times.append(time.perf_counter_ns() - start)
    micros = np.array(times) / 1000
    return float(np.median(micros)), float(np.percentile(micros, 90))


def main() -> None:
    requests, successes, tasks = train_inputs()
    rates = rules.fit(requests, successes)
    (PROJECT_ROOT / RATES).write_text(rates_text(rates))
    print(f"wrote {RATES} from {len(requests)} train items")
    print(f"cut points (train medians): {rates['cut_points']}")
    for key, bucket in rates["buckets"].items():
        said = ", ".join(f"{c} {bucket['rates'][c]:.3f}" for c in SCORED)
        print(f"  {key:24} n = {bucket['count']:3}{'  folded into its type' if bucket['folded'] else ''}: {said}")
    print(f"\n{diagnostic(requests, tasks)}")
    dev_requests = [from_row(row) for row in read_rows() if row["split"] == "dev"]
    median, p90 = latency(dev_requests)
    print(
        f"\nlatency of score(request) over {len(dev_requests)} dev requests: median {median:.1f} us, p90 {p90:.1f} us"
    )
    print(f"machine: {machine()}")
    print("\nfreeze hashes (sha256):")
    for relative, digest in freeze_hashes().items():
        print(f"  `{relative}`: `{digest}`")


if __name__ == "__main__":
    main()
