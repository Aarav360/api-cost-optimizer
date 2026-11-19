"""Check the real go's whole-run spend projection against both caps, from non-test inputs (PLAN_v4.md Amendment A19, items 9 and 13;
A17 item 18; issue #77; the Stage 8 review's O3).

Run from the project root:  uv run python -m analysis.check_final_projection

The ledger's projection (`analysis/final_ledger.py`) runs at the go's liveness step and refuses over a cap before the first call,
which would stop the go after the unlock. This check runs the same projectors, `final_ledger.projections`, on the real world's
settings, dev's stored records, and a stand-in for the handle that holds **no test item**: test's counts come from
`data/benchmark_manifest.json` (743 items, 156 + 240 + 240 + 107), and every input a projector reads of a test item is a worst
case taken from train and dev:

- a request is as long as it can be: a prompt past every byte budget and the longest output schema of train and dev, so each Jev
  body is cut to its budget (G's, G+'s and the Jev gate's), the most a call can carry;
- an answer is as long as the Jev gate's budget, and any of the 2,972 answers (every config) may pass the free checks, so the
  gate is projected on all of them (the free checks have not run yet at the go);
- the Sonnet gate is projected at the projection's own rate (twice dev's list price per verdict, `final_sonnet.PROJECTION_FACTOR`)
  on all 2,229 answers of the three gated configs, the resends at list price and the batch at the discount;
- the live replay's 100 items are drawn from dev's items by the replay's own rule (`unlock_facts.draw_replay`: 25 per task, in
  sha256 order), and each config's stored usage on them is dev's largest over its runs. The real draw is from test, so the check
  also draws 2,000 other samples of 25 per task from dev (seeded) and counts how many would be refused, and runs the dearest 25
  of each task as a known positive: the replay's projection is a bound that doubles each request's cost and counts an
  escalation to `opus` for every one (`final_replay.SAFETY`), so it is far above what the replay will cost, and it can be over
  the $10 pool for a draw of expensive items.

Needs `results/` (dev's stored records, not committed), so the hook runs it and CI doesn't (`HOOK_ONLY` in
`analysis/check_guardrails.py`). It reads no test item, no answer and no verdict, opens no store, and calls nothing: no Jev or
Anthropic call, no router, no gate. Nothing writes to `results/`: the world's output folder is a temporary one.

- each pool's projection, with nothing recorded at the go's start, is under its cap: the $25 calls pool (G's and G+'s drift
  resends and test calls, the Jev gate's, the Sonnet gate's, each part printed, every input at its worst) and the $10 live pool (the
  liveness calls and the replay, on a draw from dev), and at most 1% of dev draws are over the live pool's cap;
- the stand-in is no handle: `require_issued` refuses it, so nothing can read a store through it;
- known positives: a cap just below a pool's total refuses it and one just above lets it through, the dearest 25 dev items of each
  task as the replay's items are refused by the live pool's cap, the worst-case request's body is at least every dev request's,
  and a stand-in with no test items holds only the drift parts, smaller than the whole.
"""

import functools
import inspect
import json
import random
import statistics
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from analysis import collect_jev, final_ledger, final_replay
from analysis.benchmark import read_rows
from analysis.checking import refused
from analysis.final_ledger import CALLS, CAPS, LIVE, Projection, Spend, projections, refusal, totals
from analysis.final_record import RECORD_FILE, Record
from analysis.final_run import Run, now_utc
from analysis.gate_answers import Answer
from analysis.gate_spend import list_dollars
from analysis.matrix import read_store
from analysis.unlock import HandleError, StoredResult, require_issued
from analysis.unlock_facts import draw_replay
from analysis.world import World
from harness.client import PROJECT_ROOT, load_models
from routers import jev, jev_call, jev_gate
from routers.ladder import LADDER
from routers.request import Request

problems: list[str] = []
MANIFEST = PROJECT_ROOT / "data" / "benchmark_manifest.json"
RESULTS = PROJECT_ROOT / "results"
BENCHMARK = (
    inspect.signature(read_rows).parameters["path"].default
)  # the benchmark reader's own default, as the go's entry point takes it
TASKS = ("ifeval", "banking77", "mmlu_pro", "mbppplus")
REPLAY_PER_TASK = 25  # A17 item 26.5; `analysis/unlock_facts.REPLAY_PER_TASK`
DRAWS = 2000


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(a - b) <= tolerance


@dataclass(frozen=True)
class WorstCase:
    """What a projector reads of test, with no test item in it: counts per task and the worst each input can be. It is no
    handle (`require_issued` refuses it) and opens nothing. The reads are the handle's own names, so a projector takes it as it
    takes the real one."""

    counts: Mapping[str, int]  # task -> test items, from the benchmark manifest
    request: Request  # as long as a request can be
    answer: str  # as long as an answer a gate is given can be
    replay: Mapping[str, list[str]]  # task -> the dev items that stand in for the replay's items
    usage: Mapping[
        tuple[str, str], tuple[int, int]
    ]  # (dev item, config) -> the most input and output tokens a run was billed

    def ids(self) -> list[str]:
        return [f"{task}:worst-{k}" for task in TASKS for k in range(self.counts[task])]

    def requests(self) -> dict[str, Request]:
        return {item: self.request for item in self.ids()}

    def answers(self) -> list[Answer]:
        return [Answer(item, config, 0, self.answer, "end_turn") for item in self.ids() for config in LADDER]

    def stored_results(self) -> list[StoredResult]:
        return [
            StoredResult(f"{item}_{config}_r0", item, config, 0, task, item, *self.usage[item, config])
            for task, items in self.replay.items()
            for item in items
            for config in LADDER
        ]

    def replay_items(self) -> tuple[str, ...]:
        return tuple(item for task in TASKS for item in self.replay[task])


def test_counts() -> dict[str, int]:
    """Each task's test items, counted in the benchmark manifest: counts only, no item."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {task: int(manifest["tasks"][task]["splits"]["test"]) for task in TASKS}


def worst_request() -> Request:
    """A prompt far past every byte budget (a cut to the budget is then the most a body can carry) with the longest output schema
    of train's and dev's items."""
    schemas = [row["output_schema"] for row in read_rows() if row.get("output_schema")]
    longest = max(schemas, key=lambda schema: len(json.dumps(schema)))
    return Request("x" * 80_000, longest)


def dev_usage() -> tuple[dict[str, str], dict[tuple[str, str], tuple[int, int]], dict[str, float]]:
    """Each dev item's task, each (item, config)'s most input and output tokens over dev's runs, and each item's list-price cost
    over the four configs at those tokens."""
    prices = load_models()["prices"]
    usage: dict[tuple[str, str], tuple[int, int]] = defaultdict(lambda: (0, 0))
    task_of: dict[str, str] = {}
    model_of: dict[str, str] = {}
    for row in read_store(split="dev"):
        key = (row["item_id"], row["config"])
        usage[key] = (max(usage[key][0], row["input_tokens"]), max(usage[key][1], row["output_tokens"]))
        task_of[row["item_id"]] = row["source"]
        model_of[row["config"]] = row["model"]
    cost: dict[str, float] = defaultdict(float)
    for (item, config), (tokens_in, tokens_out) in usage.items():
        price = prices[model_of[config]]
        cost[item] += list_dollars(tokens_in, tokens_out, float(price["input"]), float(price["output"]))
    return task_of, dict(usage), dict(cost)


def dearest_items(task_of: Mapping[str, str], cost: Mapping[str, float]) -> dict[str, list[str]]:
    """The 25 dev items of each task that cost most: the worst replay draw there could be from dev."""
    return {
        task: sorted((i for i in cost if task_of[i] == task), key=lambda i: (-cost[i], i))[:REPLAY_PER_TASK]
        for task in TASKS
    }


def drawn_items(task_of: Mapping[str, str]) -> dict[str, list[str]]:
    """Dev's items drawn as the replay draws test's: 25 per task, in the seeded sha256 order (`unlock_facts.draw_replay`)."""
    rows = [{"source": task_of[row["item_id"]], "item_id": row["item_id"]} for row in read_rows(split="dev")]
    drawn = draw_replay(rows, REPLAY_PER_TASK)
    return {task: [i for i in drawn if task_of[i] == task] for task in TASKS}


def run_on(stand_in: object, output: Path) -> Run:
    """A `Run` on the real world's settings and stored records, the handle replaced by `stand_in`. Its output folder is
    `output`, empty: no Stage 8 record exists at the go's start, so nothing is recorded and nothing is projected as stored."""
    world = World(root=PROJECT_ROOT, sealed_store=None, benchmark=BENCHMARK, results=RESULTS, output=output)
    run = Run(world, Record(output / RECORD_FILE), now_utc)
    run.handle = stand_in  # type: ignore[assignment]
    return run


def parts(run: Run) -> dict[str, Projection]:
    """What every built projector says, as the liveness step adds them up. The cut of a prompt to its byte budget sizes it a
    character at a time, which for the same worst-case prompt and answer on thousands of calls is minutes of the same answer,
    so the two cutting functions are memoised for the run: the same arguments give the same value."""
    with (
        mock.patch.object(jev_gate, "fit_to_budget", functools.cache(jev_gate.fit_to_budget)),
        mock.patch.object(jev_call, "cut_prompt", functools.cache(jev_call.cut_prompt)),
    ):
        return {p.name: p for p in projections(run)}


def stage_8_files() -> set[str]:
    return {p.name for p in RESULTS.rglob("*") if p.name.endswith(("_test.jsonl", "_drift.jsonl"))}


def main() -> int:
    print("-- the real go's projection, from non-test inputs --")
    counts = test_counts()
    total = sum(counts.values())
    check(counts == {"ifeval": 156, "banking77": 240, "mmlu_pro": 240, "mbppplus": 107} and total == 743,
          f"the benchmark manifest counts {total} test items: {counts}")  # fmt: skip
    task_of, usage, cost = dev_usage()
    replay = drawn_items(task_of)
    worst = WorstCase(counts, worst_request(), "y" * 80_000, replay, usage)
    check(len(worst.answers()) == 4 * total and len(worst.replay_items()) == 4 * REPLAY_PER_TASK,
          f"the stand-in holds {4 * total} answers (every config) and the replay's {4 * REPLAY_PER_TASK} items")  # fmt: skip

    with tempfile.TemporaryDirectory() as folder:  # the world's output folder: empty, and not the project's
        before = stage_8_files()
        whole = parts(run_on(worst, Path(folder)))
        empty = WorstCase({task: 0 for task in TASKS}, worst.request, worst.answer, replay, usage)
        drifts = parts(run_on(empty, Path(folder)))
        dearest = WorstCase(counts, worst.request, worst.answer, dearest_items(task_of, cost), usage)
        worst_draw = parts(run_on(dearest, Path(folder)))
        check(
            before == stage_8_files() == set(),
            "no Stage 8 record file exists before or after: the projection wrote nothing",
        )

    pooled = totals(list(whole.values()))
    print(f"\n   {'projector':<12} {'pool':<6} {'whole run':>10}   {'its drift part':>14}")
    for name, projection in whole.items():
        print(f"   {name:<12} {projection.pool:<6} ${projection.dollars:>9.4f}   ${drifts[name].dollars:>13.4f}")
    for pool, cap in CAPS.items():
        print(f"   the {pool} pool: ${pooled[pool]:.4f} projected of its ${cap:.2f} cap")
    check(set(whole) == {p.name for p in final_ledger.PROJECTORS if p.entry}
          and {p.pool for p in whole.values()} == {CALLS, LIVE},
          "every built projector of the ledger ran, in both pools")  # fmt: skip
    for pool in (CALLS, LIVE):
        check(
            0 < pooled[pool] < CAPS[pool],
            f"the {pool} pool's projection is under its cap: ${pooled[pool]:.2f} of ${CAPS[pool]:.0f}",
        )
    check(refusal(Spend(0.0, 0.0, 0), pooled) is None, "and the ledger's own refusal test lets the go through")

    print("\n-- the live pool's projection over other draws of 25 items per task from dev --")
    plan = final_replay.plan_of()
    models = load_models()
    price = jev.read_client_settings().price_per_million_input_tokens
    budget = jev_gate.read_settings().budget_bytes or 0
    bound = {
        i: sum(final_replay.request_bound(p, i, usage, models, price, budget) for p in plan.policies) for i in cost
    }
    check(close(sum(bound[i] for i in worst.replay_items()), whole["replay"].dollars),
          "the replay projector's total on the drawn items is the sum of its per-request bounds over them")  # fmt: skip
    by_task = {task: sorted(i for i in cost if task_of[i] == task) for task in TASKS}
    rng = random.Random(2026)
    draws = sorted(sum(bound[i] for t in TASKS for i in rng.sample(by_task[t], REPLAY_PER_TASK)) for _ in range(DRAWS))
    room = CAPS[LIVE] - whole["liveness"].dollars  # what the replay's projection may be, beside the liveness calls
    over = sum(d > room for d in draws) / DRAWS
    print(f"   {DRAWS} draws: mean ${statistics.fmean(draws):.2f}, median ${draws[DRAWS // 2]:.2f}, 99th percentile "
          f"${draws[int(DRAWS * 0.99)]:.2f}, largest ${draws[-1]:.2f}; the replay may be ${room:.2f}; {over:.2%} are over")  # fmt: skip
    check(
        over <= 0.01,
        f"at most 1% of dev draws would make the live pool's projection refuse at the liveness step ({over:.2%})",
    )
    check(refusal(Spend(0.0, 0.0, 0), totals(list(worst_draw.values()))) is not None,
          f"known positive: the dearest 25 dev items of each task as the replay's are refused by the live cap (${worst_draw['replay'].dollars:.2f})")  # fmt: skip

    print("\n-- known positives: the comparison bites, and the worst case is the worst --")
    for pool in (CALLS, LIVE):
        alone = {pool: pooled[pool]}
        with mock.patch.dict(CAPS, {pool: pooled[pool] * 0.99}):
            check(refusal(Spend(0.0, 0.0, 0), alone) is not None, f"a {pool} cap 1% below the projection refuses it")
        with mock.patch.dict(CAPS, {pool: pooled[pool] * 1.0001}):
            check(refusal(Spend(0.0, 0.0, 0), alone) is None, f"and a {pool} cap just above it lets it through")
    check(all(drifts[n].dollars < whole[n].dollars for n in ("jev_router", "jev_gate", "sonnet_gate")),
          "a stand-in with no test items projects only the drift resends: each call part is less than the whole run's")  # fmt: skip
    loaded = collect_jev.load_settings()
    dev = [Request(row["prompt"], row.get("output_schema")) for row in read_rows(split="dev")]
    for router in collect_jev.ROUTERS:
        limit = loaded.call.budget_bytes[router]
        build = collect_jev.CALLS[router].build
        longest = max(len(build(r, loaded.call, budget_bytes=limit).body) for r in dev)
        biggest = len(build(worst.request, loaded.call, budget_bytes=limit).body)
        check(
            longest <= biggest,
            f"{router}: the worst-case body ({biggest} bytes) is at least every dev request's ({longest})",
        )
    settings = jev_gate.read_settings()
    gate_budget = settings.budget_bytes or 0
    biggest = len(jev_gate.gate_call(worst.request, worst.answer, settings, budget_bytes=gate_budget).body)
    check(biggest > gate_budget, f"the Jev gate's worst-case body ({biggest} bytes) is past its prompt-and-answer budget ({gate_budget}), its wrapper included")  # fmt: skip
    check(refused(HandleError, lambda: require_issued(worst)), "the stand-in is no handle: require_issued refuses it")
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
