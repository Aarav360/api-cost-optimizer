"""Grading test's answers: the final evaluation's step 5 (PLAN_v4.md Amendment A19, items 2, 5 and 13; issue #79).

Every stored test answer is graded by the collection run's own graders and row building (`harness.collect.make_row`,
unchanged), one answer at a time on the main thread with nothing else running, since math-verify's alarm needs the main
thread and MBPP+'s sandbox runs and the grader's limits are timing-sensitive under load (A18 item 1). The answers come
through the handle's graded items, the one read that holds the expected answer, and the grades go into test's results
matrix (`TEST_MATRIX` under the world's results folder), with dev's 27 columns, uncommitted.

Each row is stored the moment it is graded, so a crash keeps what was done and the next go (which runs the unlock again)
grades only the answers with no row. Nothing is shown of a grade here: the step says how many answers it graded and how
long it took, and an error names the answer's custom ID and the exception's class (`analysis/final_run.py`).

The sandbox the world's backends name stands in for the real one while grading: the grader calls the module-level
`run_python` of `harness/evaluators.py`, which stays unchanged, so the stand-in is put there for the step and taken out
after. A world with no stand-in grades in the real sandbox.
"""

import contextlib
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

from analysis.final_run import ItemFailure, Run, StepResult
from analysis.unlock import GradedItem
from harness import evaluators
from harness.client import build_request, load_models
from harness.collect import MATRIX_COLUMNS, Job, make_row
from harness.store import request_hash

TEST_MATRIX = "test_matrix.sqlite"  # test's results matrix, in the world's results folder
PROGRESS_EVERY = 250


class GradingError(RuntimeError):
    """Grading can't start safely: it isn't on the main thread, or something else is running."""


def matrix_path(results: Path) -> Path:
    return results / TEST_MATRIX


@contextlib.contextmanager
def sandbox_of(run: Run) -> Iterator[None]:
    """The world's stand-in sandbox in place of the grader's, for the step."""
    stand_in = run.world.backends.sandbox if run.world.backends is not None else None
    if stand_in is None:
        yield
        return
    real = evaluators.run_python
    evaluators.run_python = stand_in
    try:
        yield
    finally:
        evaluators.run_python = real


def job_of(graded: GradedItem, models: dict) -> tuple[Job, dict]:
    """The collection run's job and record for a stored result, rebuilt from the handle's graded item. The sealed store
    holds no request, so it is built as the collection run built it, from the config and the item, and must be the
    request the stored hash names."""
    config = models["configs"][graded.config]
    request = build_request(config, graded.item["prompt"], config["max_tokens"],
                            output_schema=graded.item.get("output_schema"))  # fmt: skip
    if request_hash(request) != graded.request_hash:
        raise ValueError("the stored result's request hash is not the hash of the request the config builds")
    job = Job(graded.custom_id, "test", graded.item, graded.config, graded.run, request)
    return job, {"run": graded.run, "request": request, "response": graded.response, "batch_id": graded.batch_id,
                 "latency_s": None}  # fmt: skip


def grade_step(run: Run) -> StepResult:
    """Grade every stored test answer with no row yet, one at a time."""
    handle = run.unlocked()
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise GradingError("grading runs on the main thread with no other thread running")
    models = load_models()
    started = time.monotonic()
    graded_items = handle.graded_items()
    path = matrix_path(run.world.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn, sandbox_of(run):
        conn.execute(f"CREATE TABLE IF NOT EXISTS rows ({', '.join(MATRIX_COLUMNS)}, PRIMARY KEY (custom_id))")
        done = {row[0] for row in conn.execute("SELECT custom_id FROM rows")}
        todo = [graded for graded in graded_items if graded.custom_id not in done]
        run.say(f"grading: {len(graded_items)} stored answers, {len(done)} already graded, {len(todo)} to grade")
        for number, graded in enumerate(todo, 1):
            try:
                job, record = job_of(graded, models)
                row = make_row(job, record, models)
                conn.execute(f"INSERT INTO rows VALUES ({', '.join('?' * len(MATRIX_COLUMNS))})",
                             [row[column] for column in MATRIX_COLUMNS])  # fmt: skip
                conn.commit()  # a crash on the next answer keeps this one
            except Exception as error:
                raise ItemFailure(graded.custom_id) from error
            if number % PROGRESS_EVERY == 0:
                run.say(f"grading: {number} of {len(todo)} graded, {time.monotonic() - started:.0f} s")
    seconds = round(time.monotonic() - started, 1)
    run.say(f"grading: {len(todo)} answers graded in {seconds} s")
    return StepResult(detail={"stored": len(graded_items), "already_graded": len(done), "graded": len(todo),
                              "seconds": seconds})  # fmt: skip
