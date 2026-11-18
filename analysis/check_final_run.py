"""Check the final evaluation's runner, grading and offline half: the rehearsal (PLAN_v4.md Amendment A19, items 2, 5, 8
and 13; issue #79).

Run from the project root:  uv run python -m analysis.check_final_run

The whole runner on a made-up world (`analysis/rehearsal_world.py`: eight test items, two for each task, each with its four
configs' stored results, in a made-up sealed store written by the collection module's own writer), with a local stand-in
for the sandbox that grades MBPP+ code. No results/, no Docker, no Jev or Anthropic call, and nothing near the real store.
The hook runs `analysis/check_final_run_sandbox.py` too, the same world graded in the real sandbox. Expected values are
worked by hand below, from the world's token counts, the price sheet and the task mixes; the oracle's are the best of every
one of the 4^8 ways to pick a config for each item, on the upper hull of their points.

- the runner runs A19 item 2's steps in order, skips the ones not built, and records each step's start and end, request
  counts and the unlock's entry, with the injected clock's times in order;
- grading grades all 32 stored answers, one at a time on the main thread, no two overlapping (shown on a spy, which is
  shown to catch an overlap), through the collection run's own row building, into a results matrix with dev's 27 columns
  whose statuses, qualities and costs are the hand-worked ones;
- the offline half's tables for A to D, under both mixes and per task, equal the hand-worked numbers: config qualities,
  costs, differences and savings; the envelope's corners; the oracle's whole curve and the cost at each target against the
  brute-force hull; IFEval's and MBPP+'s entries marked indicative, in test's words; the statuses counted;
- four markers (an expected answer's, an answer's, a thinking block's and a grade's) appear in nothing printed or logged
  before the tables step, nor in the record, and the grade's does appear in the tables, so the search finds it (a step
  made to print a marker is the known positive);
- a crash injected mid-grading stops the run naming the item and the exception's class and showing no grade; the next go
  runs the unlock again, grades only the ungraded answers, and the record keeps both gos' unlock entries;
- a refused unlock stops the run at the unlock, a grading step with no handle is refused, a stored answer whose request
  hash isn't the config's request is refused naming it, and a tables step over ungraded answers is refused;
- a step or a part added by one line runs where it was put.
"""

import contextlib
import dataclasses
import io
import itertools
import json
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest import mock

import numpy as np

from analysis import final_grading, offline_half, rehearsal_trained
from analysis import rehearsal_world as rehearsal
from analysis.backends import Backends
from analysis.final_record import RECORD_FILE, Record
from analysis.final_run import LOG_FILE, STEPS, ItemFailure, Outcome, Run, Step, StepResult, describe, now_utc, run
from analysis.made_up_world import MadeUp, made_up_world
from analysis.offline_half import OfflineError, Part
from analysis.unlock import HandleError, unlock
from harness import collect, evaluators

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(a - b) <= tolerance


# ---- the hand-worked world ---------------------------------------------------------------------------------------

CONFIGS = ("haiku", "sonnet", "opus_low", "opus")
TASKS = ("ifeval", "banking77", "mmlu_pro", "mbppplus")
# The items in the world's order: banking 0, 1; mmlu 0, 1; ifeval 0, 1; mbpp 0, 1. Each item's success on each config.
SUCCESS = [
    (1, 1, 1, 1),
    (0, 1, 1, 1),
    (1, 1, 1, 1),
    (0, 0, 1, 0),
    (1, 1, 1, 1),
    (0, 0, 0, 1),
    (1, 1, 1, 1),
    (0, 0, 1, 1),
]
ITEM_TASKS = ["banking77", "banking77", "mmlu_pro", "mmlu_pro", "ifeval", "ifeval", "mbppplus", "mbppplus"]
# Dollars per request in millionths: 1,000 input tokens and 500 / 400 / 300 / 600 output tokens (MBPP+ doubles them) at
# per-million prices of haiku $0.10 in and $0.50 out, sonnet $2 and $10, opus (both configs) $4 and $20:
# haiku 100 + 250 = 350 (MBPP+ 100 + 500 = 600); sonnet 2000 + 4000 = 6000 (MBPP+ 2000 + 8000 = 10000);
# opus_low 4000 + 6000 = 10000 (MBPP+ 4000 + 12000 = 16000); opus 4000 + 12000 = 16000 (MBPP+ 4000 + 24000 = 28000).
MICRO = {"haiku": (350, 600), "sonnet": (6000, 10000), "opus_low": (10000, 16000), "opus": (16000, 28000)}
MIXES = {
    "uniform": {"ifeval": 0.25, "banking77": 0.25, "mmlu_pro": 0.25, "mbppplus": 0.25},
    "realistic": {"mbppplus": 0.65, "banking77": 0.20, "ifeval": 0.10, "mmlu_pro": 0.05},
}
# Each item's status on each config: the graders' own words, by hand (see rehearsal_world's answers).
STATUS = {
    ("banking77:made-test-1", "haiku"): "wrong", ("mmlu_pro:made-test-1", "haiku"): "wrong",
    ("mmlu_pro:made-test-1", "sonnet"): "refusal", ("mmlu_pro:made-test-1", "opus"): "wrong",
    ("ifeval:made-test-1", "haiku"): rehearsal.GRADE_MARKER, ("ifeval:made-test-1", "sonnet"): "wrong",
    ("ifeval:made-test-1", "opus_low"): "wrong", ("mbppplus:made-test-1", "haiku"): "wrong",
    ("mbppplus:made-test-1", "sonnet"): "wrong",
}  # fmt: skip
ITEMS = [made["item_id"] for made in rehearsal.ITEMS]
EXPECTED_STATUS_COUNTS = {
    "haiku": {"correct": 4, "wrong": 3, rehearsal.GRADE_MARKER: 1},
    "sonnet": {"correct": 5, "refusal": 1, "wrong": 2},
    "opus_low": {"correct": 7, "wrong": 1},
    "opus": {"correct": 7, "wrong": 1},
}
# The collection run's matrix columns, written out: the results matrix test writes has dev's.
MATRIX_COLUMNS = [
    "custom_id", "item_id", "split", "source", "task_type", "config", "model", "reported_model", "effort", "run",
    "stop_reason", "refusal_category", "status", "quality", "success", "parsed", "detail", "answer", "input_tokens",
    "cached_input_tokens", "output_tokens", "thinking_tokens", "cost_usd", "latency_s", "price_snapshot", "batch_id",
    "request_hash",
]  # fmt: skip
MARKERS = (rehearsal.EXPECTED_MARKER, rehearsal.ANSWER_MARKER, rehearsal.THINKING_MARKER, rehearsal.GRADE_MARKER)
# By hand: a per-request dollar cost in a task, then each mix's cost per 1,000 requests and quality of each config.
HAND_QUALITY = {
    "uniform": {"haiku": 0.5, "sonnet": 0.625, "opus_low": 0.875, "opus": 0.875},
    "realistic": {"haiku": 0.5, "sonnet": 0.6, "opus_low": 0.95, "opus": 0.975},
}
HAND_COST = {  # dollars per 1,000 requests: sum of mix weight times the task's micro-dollars, over 1,000
    "uniform": {"haiku": 0.4125, "sonnet": 7.0, "opus_low": 11.5, "opus": 19.0},
    "realistic": {"haiku": 0.5125, "sonnet": 8.6, "opus_low": 13.9, "opus": 23.8},
}
# Per task: quality of each config, and cost per 1,000 requests (the micro-dollars over 1,000).
HAND_TASK_QUALITY = {
    "banking77": {"haiku": 0.5, "sonnet": 1.0, "opus_low": 1.0, "opus": 1.0},
    "mmlu_pro": {"haiku": 0.5, "sonnet": 0.5, "opus_low": 1.0, "opus": 0.5},
    "ifeval": {"haiku": 0.5, "sonnet": 0.5, "opus_low": 0.5, "opus": 1.0},
    "mbppplus": {"haiku": 0.5, "sonnet": 0.5, "opus_low": 1.0, "opus": 1.0},
}
# Corners of each reference envelope, worked by hand from the points: the upper hull from the cheapest point.
# Under the uniform mix haiku (0.4125, 0.5) climbs to opus_low (11.5, 0.875) at 0.0338 a dollar, steeper than sonnet's
# 0.0190 and opus's 0.0202, and nothing is above opus_low; under the realistic mix opus_low is steeper than sonnet and
# opus from haiku too, and opus (23.8, 0.975) is above it.
HAND_ENVELOPE = {
    "uniform": {"haiku": (0.4125, 0.5), "opus_low": (11.5, 0.875)},
    "realistic": {"haiku": (0.5125, 0.5), "opus_low": (13.9, 0.95), "opus": (23.8, 0.975)},
}
HAND_TASK_ENVELOPE = {
    "banking77": {"haiku": (0.35, 0.5), "sonnet": (6.0, 1.0)},
    "mmlu_pro": {"haiku": (0.35, 0.5), "opus_low": (10.0, 1.0)},
    "ifeval": {"haiku": (0.35, 0.5), "opus": (16.0, 1.0)},
    "mbppplus": {"haiku": (0.6, 0.5), "opus_low": (16.0, 1.0)},
}
INDICATIVE = "Only 2 test items, below the plan's minimum of a few hundred: indicative"


def clock() -> Callable[[], datetime]:
    """A clock that moves one second at every look, so a step's start and end order is the order they were taken in."""
    now = [datetime(2026, 12, 1, tzinfo=timezone.utc)]

    def tick() -> datetime:
        now[0] += timedelta(seconds=1)
        return now[0]

    return tick


class Rehearsal:
    """A made-up world in a folder, a counting stand-in sandbox, and what a run printed."""

    def __init__(self, made: MadeUp):
        self.sandbox_calls = 0
        # F's stand-ins: a fake embedder that has not moved since dev's table was scored (`analysis/rehearsal_trained.py`)
        self.embedder = rehearsal_trained.stand_in(0.0)
        self.world = dataclasses.replace(made.world, backends=Backends(sandbox=self.sandbox, trained=self.embedder))
        self.tick = clock()

    def sandbox(self, source: str, timeout_s: float = 10.0, memory: str = "512m"):
        self.sandbox_calls += 1
        return rehearsal.local_run_python(source, timeout_s, memory)

    def go(self, steps: tuple[Step, ...] = STEPS) -> tuple[Outcome, str, str]:
        """One go: its outcome and what it printed to the screen and to the error stream."""
        out, err = io.StringIO(), io.StringIO()
        # The made-up instructions need no NLTK data, which would download in CI.
        with (
            mock.patch.object(evaluators, "_ensure_nltk_data", lambda: None),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            outcome = run(self.world, steps, self.tick)
        return outcome, out.getvalue(), err.getvalue()

    @property
    def record(self) -> dict:
        return json.loads((self.world.output / RECORD_FILE).read_text(encoding="utf-8"))

    @property
    def log(self) -> str:
        return (self.world.output / LOG_FILE).read_text(encoding="utf-8")

    @property
    def tables(self) -> dict:
        return json.loads((self.world.output / "tables_test.json").read_text(encoding="utf-8"))

    def rows(self) -> list[dict]:
        with closing(sqlite3.connect(final_grading.matrix_path(self.world.results))) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute("SELECT * FROM rows ORDER BY item_id, config, run")]


@contextlib.contextmanager
def rehearsed(content: Any = None):
    with made_up_world(content or rehearsal.content()) as made:
        yield Rehearsal(made)


class Spy:
    """Stands in the grader's place (`score`) and records every call: it passes each through to the real grader, and can
    raise on a chosen call. Two calls at once, or a call off the main thread, show in `overlapped` and `threads`."""

    def __init__(self, crash_at: int | None = None, delay: float = 0.0):
        self.real = collect.score
        self.crash_at, self.delay = crash_at, delay
        self.calls: list[str] = []
        self.active = self.overlapped = 0
        self.threads: set[str] = set()
        self.thread_counts: set[int] = set()
        self.lock = threading.Lock()

    def __call__(self, item: dict, response: dict):
        with self.lock:
            self.active += 1
            self.overlapped = max(self.overlapped, self.active)
            self.calls.append(item["item_id"])
            self.threads.add(threading.current_thread().name)
            self.thread_counts.add(threading.active_count())
            number = len(self.calls)
        try:
            time.sleep(self.delay)
            if number == self.crash_at:
                raise RuntimeError(f"boom {rehearsal.GRADE_MARKER}")  # a message that holds a grade
            return self.real(item, response)
        finally:
            with self.lock:
                self.active -= 1


def leaks(text: str) -> list[str]:
    return [marker for marker in MARKERS if marker in text]


# ---- the order, the record and the skeleton ----------------------------------------------------------------------


def first_go() -> None:
    print("-- the runner runs A19 item 2's steps in order and records them --")
    expected = ["unlock", "liveness", "drift_trained_router", "drift_jev_router", "drift_jev_gate", "drift_sonnet_gate",
                "calls_rules_trained", "calls_jev_router", "calls_free_checks", "calls_jev_gate", "calls_sonnet_gate",
                "grading", "tables", "replay", "record"]  # fmt: skip
    check([step.name for step in STEPS] == expected, "the steps' order is A19 item 2's: the unlock, the liveness calls, "
          "the drift checks, the calls, grading, the tables, the live replay, the record")  # fmt: skip
    built = {step.name for step in STEPS if step.entry is not None}
    check(
        built == {"unlock", "grading", "tables", "record", "drift_trained_router", "calls_rules_trained"},
        f"the steps built so far are the four of #79 and F's drift check and E's and F's calls (#83) {built}",
    )
    with rehearsed() as r:
        spy = Spy()
        with mock.patch.object(collect, "score", spy):
            outcome, out, err = r.go()
        record = r.record
        check(outcome == Outcome(), f"a go on the made-up world runs every step {outcome}")
        go = record["gos"][0]
        check(len(record["gos"]) == 1 and go["go"] == 1 and go["stopped"] is None and go["ended"] is not None,
              "the record holds one go, which ended without stopping")  # fmt: skip
        check([s["name"] for s in go["steps"]] == expected, "the record's steps are in the same order")
        status = {s["name"]: s["status"] for s in go["steps"]}
        check({n for n, v in status.items() if v == "done"} == built and set(status.values()) == {"done", "skipped"},
              "each built step is done, and each step not built is skipped")  # fmt: skip
        times = [t for s in go["steps"] for t in (s["started"], s["ended"])]
        check(times == sorted(times) and times[0] > go["started"] and go["ended"] > times[-1],
              "each step's start and end, and the go's, follow the clock in order")  # fmt: skip
        check(all(set(s["requests"]) <= {"sent", "stored"} for s in go["steps"] if s["status"] == "done")
              and go["steps"][11]["requests"] == {"sent": 0, "stored": 0},
              "a built step records its request counts: grading sent none")  # fmt: skip
        unlock_entry = go["unlock"]
        check([c["name"] for c in unlock_entry["checks"]] == ["sealed_store_path", "manifest_record", "sealed_store_sha256"]
              and unlock_entry["time"] <= go["steps"][0]["ended"],
              "the unlock's entry names its three checks and its time")  # fmt: skip
        check(r.log.count("go 1 starts") == 1 and "grading: 32 answers graded" in r.log and r.log.startswith("2026-12-01"),
              "the log holds the run's lines, each stamped")  # fmt: skip
        check(out.strip() == r.log.strip() and err == "", "everything the run printed is in its log, and nothing went to the error stream")  # fmt: skip
        grading(r, spy)
        tables(r)
        markers(r, out, err)


# ---- grading -----------------------------------------------------------------------------------------------------


def grading(r: Rehearsal, spy: Spy) -> None:
    print("\n-- grading: every answer, one at a time, into test's results matrix --")
    check(len(spy.calls) == 32, f"the grader ran on all 32 stored answers, {len(spy.calls)} times")
    check(spy.overlapped == 1 and spy.threads == {"MainThread"} and spy.thread_counts == {1},
          "one at a time: never two gradings at once, always on the main thread with no other thread running")  # fmt: skip
    probe = Spy(delay=0.05)
    pair = [threading.Thread(target=probe, args=({"item_id": "x"}, {})) for _ in range(2)]
    with mock.patch.object(probe, "real", lambda item, response: None):
        for thread in pair:
            thread.start()
        for thread in pair:
            thread.join()
    check(probe.overlapped == 2 and "MainThread" not in probe.threads and probe.thread_counts != {1},
          "the spy sees two gradings at once and a grading off the main thread, so its all-clear is earned")  # fmt: skip
    check(r.sandbox_calls == 8, f"MBPP+'s eight answers ran in the world's stand-in sandbox ({r.sandbox_calls} runs)")
    check(
        evaluators.run_python.__module__ != rehearsal.__name__, "the real sandbox is back in the grader after the step"
    )

    rows = r.rows()
    with closing(sqlite3.connect(final_grading.matrix_path(r.world.results))) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(rows)")]
    check(columns == MATRIX_COLUMNS, "test's matrix has the collection run's 27 columns, in its order")
    check(len(rows) == 32 and {row["split"] for row in rows} == {"test"} and {row["run"] for row in rows} == {0},
          "it holds 32 rows, all test and run 0")  # fmt: skip
    wrong_status = [(row["item_id"], row["config"]) for row in rows
                    if row["status"] != STATUS.get((row["item_id"], row["config"]), "correct")]  # fmt: skip
    check(not wrong_status, f"every row's status is the hand-worked one, MBPP+'s from running the code {wrong_status}")
    right = {(i, c): SUCCESS[p][k] for p, i in enumerate(ITEMS) for k, c in enumerate(CONFIGS)}
    check(all(row["success"] == right[(row["item_id"], row["config"])] and row["quality"] == float(row["success"])
              for row in rows), "every row's success and quality are the hand-worked ones")  # fmt: skip
    tokens = {"haiku": 500, "sonnet": 400, "opus_low": 300, "opus": 600}
    ok = True
    for row in rows:
        mbpp = row["source"] == "mbppplus"
        micro = MICRO[row["config"]][1 if mbpp else 0]
        ok &= close(row["cost_usd"], micro * 1e-6 / 2, 1e-12)  # the collection run's batch price: half the list price
        ok &= row["input_tokens"] == 1000 and row["output_tokens"] == tokens[row["config"]] * (2 if mbpp else 1)
    check(ok, "every row's tokens and batch cost are the hand-worked ones (half the list price of 1,000 in, n out)")
    check(all(row["price_snapshot"] == "2026-10-12" and row["batch_id"] == "msgbatch_made_up" and row["latency_s"] is None
              for row in rows), "rows carry the price snapshot, the batch and no latency, as the collection run's do")  # fmt: skip
    refusal = next(row for row in rows if row["status"] == "refusal")
    check(refusal["refusal_category"] == "made_up" and refusal["stop_reason"] == "refusal",
          "a refusal's category comes from the stored answer's stop details, through the collection run's row building")  # fmt: skip
    job, record = direct_job(r)
    direct = collect.make_row(job, record, final_grading.load_models())
    check(direct == {k: v for k, v in rows[0].items()} and direct["custom_id"] == rows[0]["custom_id"],
          "the first stored row equals the collection run's own `make_row` on the same answer: its row building, unchanged")  # fmt: skip


def direct_job(r: Rehearsal):
    """One stored answer's job and record, as the grading step rebuilds them, for a direct call of the row building."""
    handle = unlock(r.world)
    models = final_grading.load_models()
    job, record = final_grading.job_of(handle.graded_items()[0], models)
    return job, record


# ---- the tables --------------------------------------------------------------------------------------------------


def upper_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The upper hull of points (cost, quality) from the cheapest to the highest quality, by monotone chain."""
    hull: list[tuple[float, float]] = []
    for p in sorted(set(points)):
        while (
            len(hull) >= 2
            and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])
            >= 0
        ):
            hull.pop()
        hull.append(p)
    top = max(q for _, q in hull)
    return hull[: next(i for i, (_, q) in enumerate(hull) if q == top) + 1]


def brute_force_hull(mix: dict[str, float], positions: list[int]) -> list[tuple[float, float]]:
    """Every way to give each of the `positions`' items one config (4 to the number of items ways), each a point of
    (cost per 1,000 requests, quality) under the mix: the upper hull of them is the best any assignment reaches, which is
    what the hindsight oracle's curve is."""
    tasks = [ITEM_TASKS[p] for p in positions]
    weights = np.array([mix[t] / tasks.count(t) for t in tasks]) if len(set(tasks)) == len(mix) else None
    if weights is None:  # a single task's own oracle
        weights = np.array([1 / len(tasks)] * len(tasks))
    success = np.array([SUCCESS[p] for p in positions], dtype=float)
    cost = np.array([[MICRO[c][1 if ITEM_TASKS[p] == "mbppplus" else 0] / 1000 for c in CONFIGS] for p in positions])
    choices = np.array(list(itertools.product(range(4), repeat=len(positions))))
    rows = np.arange(len(positions))
    quality = (weights * success[rows, choices]).sum(axis=1)
    dollars = (weights * cost[rows, choices]).sum(axis=1)
    return upper_hull(list(zip(dollars.round(12).tolist(), quality.round(12).tolist())))


def curve_agrees(corners: list[dict], hull: list[tuple[float, float]]) -> bool:
    """The oracle's polyline through its corners and the brute-force hull give the same quality at every cost."""
    xs, ys = [c["cost_per_1k"] for c in corners], [c["quality"] for c in corners]
    hx, hy = [p[0] for p in hull], [p[1] for p in hull]
    grid = np.linspace(min(hx), max(hx), 400)
    return (
        close(xs[0], hx[0])
        and close(xs[-1], hx[-1])
        and close(ys[-1], hy[-1])
        and bool(np.allclose(np.interp(grid, xs, ys), np.interp(grid, hx, hy), atol=1e-9))
    )


def estimate_ok(estimate: dict) -> bool:
    return estimate["low"] <= estimate["value"] + 1e-12 and estimate["value"] <= estimate["high"] + 1e-12


def tables(r: Rehearsal) -> None:
    print("\n-- the offline half's tables: A to D under both mixes and per task, by hand --")
    file = r.tables
    check(file["split"] == "test" and file["n"] == 8 and file["run"] == 0 and file["price_snapshot"] == "2026-10-12"
          and file["bootstrap"] == {"resamples": 2000, "seed": 2026, "interval": "percentile 95%"},
          "the tables name the split, its 8 items, the snapshot and the seeded bootstrap")  # fmt: skip
    check(
        list(file["parts"]) == ["references", "stops", "rules", "trained"],
        "the tables hold the parts in the order PARTS gives",
    )
    mixes = file["parts"]["references"]["mixes"]
    for mix in ("realistic", "uniform"):
        panel = mixes[mix]
        configs = panel["configs"]
        a_cost = HAND_COST[mix]["opus"]
        ok = all(close(configs[c]["quality"]["value"], HAND_QUALITY[mix][c]) and close(configs[c]["cost_per_1k"]["value"], HAND_COST[mix][c])
                 for c in CONFIGS)  # fmt: skip
        check(ok, f"{mix}: each config's quality and cost per 1,000 requests are the hand-worked ones")
        check(all(close(configs[c]["difference_from_A"]["value"], HAND_QUALITY[mix][c] - HAND_QUALITY[mix]["opus"])
                  and close(configs[c]["savings_vs_A"]["per_1k"], a_cost - HAND_COST[mix][c])
                  and close(configs[c]["savings_vs_A"]["share"], (a_cost - HAND_COST[mix][c]) / a_cost) for c in CONFIGS),
              f"{mix}: each config's difference from A, and its savings against A in dollars and as a share, are hand-worked")  # fmt: skip
        check(all(estimate_ok(configs[c][k]) for c in CONFIGS for k in ("quality", "cost_per_1k", "difference_from_A")),
              f"{mix}: every interval holds its value")  # fmt: skip
        envelope = {c: (p["cost_per_1k"], p["quality"]) for c, p in panel["envelope"].items()}
        check(list(envelope) == list(HAND_ENVELOPE[mix]) and all(
            close(envelope[c][0], HAND_ENVELOPE[mix][c][0]) and close(envelope[c][1], HAND_ENVELOPE[mix][c][1]) for c in envelope),
              f"{mix}: C's corners are the hand-worked hull {list(envelope)}")  # fmt: skip
        hull = brute_force_hull(MIXES[mix], list(range(8)))
        check(curve_agrees(panel["oracle"], hull) and close(panel["oracle_max_quality"], 1.0),
              f"{mix}: D's whole curve is the upper hull of all 65,536 assignments, and its top quality is 1")  # fmt: skip
        check(panel["failed_by_every_config"] == 0, f"{mix}: no item fails on every config")
        hx, hy = [p[0] for p in hull], [p[1] for p in hull]
        a_quality = HAND_QUALITY[mix]["opus"]
        ok = True
        for share, key in ((1.0, "100%"), (0.99, "99%"), (0.97, "97%")):
            target = panel["targets"][key]
            wanted = share * a_quality
            ok &= close(target["quality"], wanted) and close(
                target["oracle"]["cost_per_1k"], float(np.interp(wanted, hy, hx))
            )
            ok &= close(sum(target["oracle"]["routing"].values()), 1.0)
            ok &= 0 <= target["envelope"]["captured"]["met"] <= 1
        check(ok, f"{mix}: each target's quality and D's cost to reach it (read off the brute-force hull), and routing, are hand-worked")  # fmt: skip
    tasks = file["parts"]["references"]["tasks"]
    check(list(tasks) == list(TASKS), "per task: the four tasks in A5's order")
    ok = True
    for task in TASKS:
        panel = tasks[task]
        ok &= panel["items"] == 2
        for config in CONFIGS:
            micro = MICRO[config][1 if task == "mbppplus" else 0]
            ok &= close(panel["configs"][config]["quality"]["value"], HAND_TASK_QUALITY[task][config])
            ok &= close(panel["configs"][config]["cost_per_1k"]["value"], micro / 1000)
        ok &= list(panel["envelope"]) == list(HAND_TASK_ENVELOPE[task]) and all(
            close(panel["envelope"][c]["cost_per_1k"], HAND_TASK_ENVELOPE[task][c][0])
            and close(panel["envelope"][c]["quality"], HAND_TASK_ENVELOPE[task][c][1]) for c in panel["envelope"])  # fmt: skip
        positions = [p for p, t in enumerate(ITEM_TASKS) if t == task]
        ok &= curve_agrees(panel["oracle"], brute_force_hull({task: 1.0}, positions)) and close(
            panel["oracle_max_quality"], 1.0
        )
    check(
        ok,
        "per task: item counts, each config's quality and cost, C's corners and D's curve (brute force) are hand-worked",
    )
    check({t: tasks[t]["warning"] for t in TASKS} == {"ifeval": INDICATIVE, "banking77": None, "mmlu_pro": None, "mbppplus": INDICATIVE},
          "IFEval's and MBPP+'s per-task entries are marked indicative, in test's words, and the others aren't")  # fmt: skip
    check(file["parts"]["stops"] == EXPECTED_STATUS_COUNTS,
          "the stops part counts each config's graded rows by status: truncations and refusals are reported with the tables")  # fmt: skip


# ---- the markers -------------------------------------------------------------------------------------------------


def leaky_step(run: Run) -> StepResult:
    run.say(f"the known positive: {rehearsal.GRADE_MARKER}")
    return StepResult()


def extra_step(run: Run) -> StepResult:
    run.say("an added step ran")
    return StepResult(requests={"sent": 3, "stored": 1}, detail={"note": "added by one line"})


def markers(r: Rehearsal, out: str, err: str) -> None:
    print("\n-- nothing shows an outcome before the tables step; the tables hold it --")
    before = r.log.split("tables: starts")[0]
    record = (r.world.output / RECORD_FILE).read_text(encoding="utf-8")
    check(
        "grading: 32 answers graded" in before and "tables: starts" in r.log,
        "the cut is after grading and before the tables",
    )
    check(not leaks(before) and not leaks(record) and not leaks(err),
          "no marker (an expected answer's, an answer's, a thinking block's or a grade's) is in what the run printed or logged "
          "before the tables step, or in the record")  # fmt: skip
    tables_text = (r.world.output / "tables_test.json").read_text(encoding="utf-8")
    check(rehearsal.GRADE_MARKER in tables_text, "the grade's marker is in the tables, so a search for it can find it")
    check(rehearsal.GRADE_MARKER not in tables_text.replace(f'"{rehearsal.GRADE_MARKER}"', ""),
          "it is there as the status key of the stops part, and nowhere else")  # fmt: skip
    with rehearsed() as other:
        steps = (STEPS[0], Step("leak", "analysis.check_final_run:leaky_step"))
        other.go(steps)
        check(leaks(other.log) == [rehearsal.GRADE_MARKER] and leaks((other.world.output / RECORD_FILE).read_text()) == [],
              "a step made to print a marker before the tables step is caught by the same search (the known positive)")  # fmt: skip


# ---- a crash and a resume ----------------------------------------------------------------------------------------


def crash_and_resume() -> None:
    print("\n-- a crash injected mid-grading, and the next go --")
    with rehearsed() as r:
        spy = Spy(crash_at=13)
        with mock.patch.object(collect, "score", spy):
            outcome, out, err = r.go()
        custom_id = "ifeval-made-test-1_haiku_r0"  # the 13th answer: three items' four configs graded, then this one
        check(
            outcome == Outcome("grading", "RuntimeError", custom_id),
            f"the run stops at grading, naming the answer and the exception {outcome}",
        )
        shown = out + err + r.log
        check(custom_id in r.log and "RuntimeError" in r.log and "tables: starts" not in r.log,
              "the log names the answer and the exception's class, and the run went no further")  # fmt: skip
        check(not leaks(shown + (r.world.output / RECORD_FILE).read_text()) and "boom" not in shown,
              "the error's message (which held a grade) is shown nowhere: no marker, no grade")  # fmt: skip
        stopped = r.record["gos"][0]["stopped"]
        check(
            stopped == {"step": "grading", "item": custom_id, "error": "RuntimeError"},
            f"the record says where it stopped {stopped}",
        )
        check(len(r.rows()) == 12 and not (r.world.output / "tables_test.json").exists(),
              "twelve answers are stored graded, and no tables were written")  # fmt: skip

        again = Spy()
        with mock.patch.object(collect, "score", again):
            resumed, _, _ = r.go()
        check(resumed == Outcome(), f"the next go runs every step {resumed}")
        check(
            len(again.calls) == 20 and len(r.rows()) == 32,
            "it graded only the 20 answers left, and the matrix holds all 32",
        )
        gos = r.record["gos"]
        check([g["go"] for g in gos] == [1, 2] and all(g["unlock"] is not None for g in gos)
              and gos[0]["unlock"]["time"] < gos[1]["unlock"]["time"],
              "the record keeps both gos' unlock entries: the second go ran the unlock again")  # fmt: skip
        check(gos[1]["steps"][11]["detail"]["already_graded"] == 12 and gos[1]["steps"][11]["detail"]["graded"] == 20,
              "its grading step records 12 already graded and 20 graded")  # fmt: skip
        resumed_tables = r.tables
        with rehearsed() as clean:
            with mock.patch.object(collect, "score", Spy()):
                clean.go()
            check(
                clean.tables == resumed_tables,
                "the tables after a crash and a resume equal those of a go with no crash",
            )


# ---- refusals ----------------------------------------------------------------------------------------------------


def refusals() -> None:
    print("\n-- refusals --")
    with rehearsed() as r:
        r.world = dataclasses.replace(r.world, sealed_store=None)
        outcome, _, _ = r.go()
        record = r.record["gos"][0]
        check(outcome.stopped_at == "unlock" and outcome.error == "UnlockRefusal"
              and record["stopped"]["check"] == "sealed_store_path" and record["unlock"] is None
              and [s["name"] for s in record["steps"]] == ["unlock"],
              "a refused unlock stops the run at the unlock, naming its check, with no entry and no later step")  # fmt: skip
        check(not final_grading.matrix_path(r.world.results).exists(), "and nothing was graded")
    with rehearsed() as r:
        outcome, _, _ = r.go((Step("grading", "analysis.final_grading:grade_step"),))
        check(outcome.stopped_at == "grading" and outcome.error == "HandleError" and "no unlock" in r.log,
              "a grading step run with no handle from the unlock is refused")  # fmt: skip
    content = rehearsal.content()
    sealed = list(content.sealed)
    row, item_id = sealed[5]
    sealed[5] = (row[:7] + ("not-the-hash",) + row[8:], item_id)
    with rehearsed(dataclasses.replace(content, sealed=sealed)) as r:
        outcome, _, _ = r.go()
        check(outcome == Outcome("grading", "ValueError", row[0]),
              "a stored answer whose request hash is not its config's request is refused, naming it")  # fmt: skip
    with rehearsed() as r:
        r.go()
        with closing(sqlite3.connect(final_grading.matrix_path(r.world.results))) as conn:
            conn.execute("DELETE FROM rows WHERE custom_id = ?", (row[0],))
            conn.commit()
        state = Run(r.world, Record(r.world.output / RECORD_FILE), now_utc)
        state.handle = unlock(r.world)
        try:
            offline_half.test_inputs(state)
            check(False, "the tables step refuses test's matrix when an answer is ungraded")
        except OfflineError as error:
            check(
                "1 stored answers are ungraded" in str(error),
                "the tables step refuses test's matrix when an answer is ungraded",
            )
        try:
            offline_half.test_inputs(Run(r.world, Record(r.world.output / RECORD_FILE), now_utc))
            check(False, "the tables step refuses a run with no handle")
        except HandleError:
            check(True, "the tables step refuses a run with no handle")


# ---- extending ---------------------------------------------------------------------------------------------------


def extending() -> None:
    print("\n-- a step, a part and a loader added by one line each --")
    steps = STEPS[:1] + (Step("added", "analysis.check_final_run:extra_step"),) + STEPS[1:12]
    with rehearsed() as r:
        outcome, _, _ = r.go(steps)
        names = [s["name"] for s in r.record["gos"][0]["steps"]]
        added = r.record["gos"][0]["steps"][1]
        check(outcome == Outcome() and names[:3] == ["unlock", "added", "liveness"] and names[-1] == "grading",
              "a step inserted by one line runs where it was put, between the steps around it")  # fmt: skip
        check(added["requests"] == {"sent": 3, "stored": 1} and added["detail"] == {"note": "added by one line"},
              "and its request counts and detail reach the record")  # fmt: skip
        state = Run(r.world, Record(r.world.output / RECORD_FILE), now_utc)
        state.handle = unlock(r.world)
        inputs = offline_half.test_inputs(state)
        extra = Part("extra", lambda i: {"items": len(i.matrix.items), "seen": sorted(i.extras)}, None)
        built = offline_half.build_tables(inputs, offline_half.PARTS + (extra,))
        check(list(built["parts"]) == ["references", "stops", "rules", "trained", "extra"] and built["parts"]["extra"] == {"items": 8, "seen": ["rules_table", "trained_facts", "trained_table"]},
              "a part added by one line is built after the others and reads its inputs")  # fmt: skip
        extras_in = dataclasses.replace(inputs, extras={"loaded": 1})
        check(extras_in.extras == {"loaded": 1} and offline_half.stops_part(extras_in)["haiku"]["correct"] == 4,
              "inputs carry what a loader adds beside the matrix")  # fmt: skip


def failure_names_item() -> None:
    print("\n-- an error names the item and the exception --")
    try:
        try:
            raise KeyError("an answer")
        except KeyError as cause:
            raise ItemFailure("some-item_haiku_r0") from cause
    except ItemFailure as error:
        check(
            describe(error) == {"item": "some-item_haiku_r0", "error": "KeyError"},
            "the description holds the item and the class only",
        )


def main() -> int:
    first_go()
    crash_and_resume()
    refusals()
    extending()
    failure_names_item()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
