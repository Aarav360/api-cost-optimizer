"""The final evaluation's runner: the go's entry point and the rehearsal's (PLAN_v4.md Amendment A19, items 2, 5 and 13;
issue #77, #79).

`run(world)` runs the steps below in A19 item 2's order, on the world it is given: the go's entry point
(`analysis/final_go.py`) builds the real one, a rehearsal a made-up one (`analysis/made_up_world.py`,
`analysis/rehearsal_world.py`). Nothing else differs between them.

- **It stops at the first crash or refusal** and returns what stopped it. A resume is a new go: `run` again, which runs
  the unlock again, and a step skips what an earlier go stored (a graded answer, a stored call).
- **Nothing shows test's outcomes before the tables step** (item 5). Until then a step says only what it did and how long
  it took, through `run.say`, and everything printed or logged goes to the log file beside the record. An error names the
  step, the item and the exception's class, never its message (a grader's message can quote an answer), except for the
  refusals whose message names a check (`MESSAGE_SHOWN`). The check searches all of it for planted markers.
- **The record** (`analysis/final_record.py`) keeps each go's unlock entry, each step's start and end and the request counts.

How to extend: a step is one line of `STEPS`, in A19's order. Replace a `None` entry with `"module:function"` for a
function `(run: Run) -> StepResult | None`, or insert a line where the step belongs. It takes the unlocked handle from
`run.handle`, the world from `run.world` (`run.world.backends` for a fake or a real backend) and says what it did with
`run.say`. It raises `ItemFailure(item)` from the error to name the item it was on, and returns a `StepResult` with its
request counts and what the record keeps. The module is imported when the step runs, so a line adds no import here, and
two tickets' lines don't touch each other. `analysis/check_final_run.py` runs every step that is built.
"""

import contextlib
import importlib
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from analysis.final_record import RECORD_FILE, Record
from analysis.unlock import Handle, HandleError, UnlockRefusal, unlock
from analysis.unlock_facts import environment
from analysis.world import World

LOG_FILE = "final_run.log"
TABLES = "tables"  # the step that writes the tables: from it on, outcomes may be shown (item 5)


class StepRefusal(Exception):
    """A refusal by a step before it calls or reads anything further: a model that isn't the pin, a drift limit, a cap.
    Its message names a model, a limit or a cap, never an answer, so the run shows it (`MESSAGE_SHOWN`)."""


MESSAGE_SHOWN = (
    UnlockRefusal,
    HandleError,
    StepRefusal,
)  # errors whose message names a check or a handle, never an answer


@dataclass(frozen=True)
class Step:
    name: str
    entry: str | None  # "module:function" of the step, or None while it isn't built: the step is skipped


# A19 item 2's order. A step that is None is one a later ticket builds, named above it. Each step sits between comment lines,
# so two tickets that each replace one line merge without a conflict.
STEPS = (
    # #78 built the unlock's checks; #80 and #81 add to them, in analysis/unlock.py
    Step("unlock", "analysis.final_run:unlock_step"),
    # #82
    Step("liveness", "analysis.final_liveness:liveness_step"),
    # #83
    Step("drift_trained_router", None),
    # #84
    Step("drift_jev_router", None),
    # #85
    Step("drift_jev_gate", None),
    # #86
    Step("drift_sonnet_gate", None),
    # #83
    Step("calls_rules_trained", None),
    # #84
    Step("calls_jev_router", None),
    # #85
    Step("calls_free_checks", None),
    # #85
    Step("calls_jev_gate", None),
    # #86
    Step("calls_sonnet_gate", None),
    # #79
    Step("grading", "analysis.final_grading:grade_step"),
    # #79; #83 to #90 add what the tables hold, in analysis/offline_half.py (PARTS, EXTRAS)
    Step("tables", "analysis.offline_half:tables_step"),
    # #91
    Step("replay", None),
    # #92 completes the record, in analysis/final_record.py and `record_step`
    Step("record", "analysis.final_run:record_step"),
)


class ItemFailure(Exception):
    """A step failed on one item. The error that caused it is `__cause__`; only its class is ever shown."""

    def __init__(self, item: str):
        super().__init__(f"failed on {item}")
        self.item = item


@dataclass(frozen=True)
class StepResult:
    # Requests sent and requests found already stored (a resume sends none of those): zero for a step that calls nothing.
    requests: Mapping[str, int] = field(default_factory=lambda: {"sent": 0, "stored": 0})
    detail: Mapping[str, Any] = field(default_factory=dict)  # what the record keeps: no grade before the tables step


@dataclass(frozen=True)
class Outcome:
    """How a go ended: `stopped_at` names the step that crashed or refused, None if every step ran."""

    stopped_at: str | None = None
    error: str | None = None  # the exception's class
    item: str | None = None


class Run:
    """What a step is given: the world, the unlocked handle (after the unlock step), the record and a way to say things."""

    def __init__(self, world: World, record: Record, clock: Callable[[], datetime]):
        self.world, self.record, self.clock = world, record, clock
        self.handle: Handle | None = None

    def stamp(self) -> str:
        return self.clock().isoformat(timespec="seconds")

    def say(self, text: str) -> None:
        """The one way a step shows anything: its line goes to the screen and the log. Before the tables step it says
        what was done, counts, spend and times, never a grade or an answer."""
        print(f"{self.stamp()} {text}", flush=True)

    def unlocked(self) -> Handle:
        """The handle the unlock step issued. A step before it, or a run whose unlock failed, has none."""
        if self.handle is None:
            raise HandleError("no unlock has issued a handle in this go")
        return self.handle


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def unlock_step(run: Run) -> StepResult:
    """Item 3: the unlock's checks, then the handle. Its entry in the record is this go's, kept beside every other go's: the
    commit, the time, the hook's result, the environment and each check's result (the replay's draw among them)."""
    handle = unlock(run.world)
    run.handle = handle
    passed = {check.name: check.detail for check in handle.results}
    run.record.set_unlock(
        {
            "time": run.stamp(),
            "commit": passed["git_facts"]["head"],  # the commit the go ran from (A19 item 3)
            "hook": dict(passed["hook"]),
            "environment": dict(environment()),
            "checks": [{"name": c.name, "detail": dict(c.detail)} for c in handle.results],
        }
    )
    run.say(f"unlock passed {len(handle.results)} checks")
    return StepResult()


def record_step(run: Run) -> StepResult:
    """The last step: the record's closing fields (#92 completes them). The go's end is written by the runner."""
    from analysis.final_ledger import spend_report  # here, since the ledger imports this module's `StepRefusal`

    return StepResult(detail={"file": RECORD_FILE, "spend": spend_report(run.world)})


def resolve(entry: str) -> Callable[[Run], StepResult | None]:
    module, _, function = entry.partition(":")
    return getattr(importlib.import_module(module), function)


def describe(error: BaseException) -> dict[str, str | None]:
    """What a stop shows and the record keeps: the item (if the step named one), the exception's class, and its message
    only for a refusal that names a check."""
    item = error.item if isinstance(error, ItemFailure) else None
    cause = (error.__cause__ or error) if isinstance(error, ItemFailure) else error
    shown: dict[str, str | None] = {"item": item, "error": type(cause).__qualname__}
    if isinstance(cause, MESSAGE_SHOWN):
        shown["message"] = str(cause)
    if isinstance(cause, UnlockRefusal):
        shown["check"] = cause.check
    return shown


class Tee:
    """A stream that writes to the screen's stream and the log, so anything printed in a run is logged."""

    def __init__(self, stream: TextIO, log: TextIO):
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()


@contextlib.contextmanager
def logged(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(out, log), Tee(err, log)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout, sys.stderr = out, err


def run(world: World, steps: tuple[Step, ...] = STEPS, clock: Callable[[], datetime] = now_utc) -> Outcome:
    """One go on `world`: every step in order, stopping at the first crash or refusal."""
    record = Record(world.output / RECORD_FILE)
    state = Run(world, record, clock)
    with logged(world.output / LOG_FILE):
        number = record.start_go(state.stamp())
        state.say(f"go {number} starts")
        for step in steps:
            record.start_step(step.name, state.stamp())
            if step.entry is None:
                record.end_step("skipped", state.stamp(), {}, {})
                state.say(f"{step.name}: skipped, not built yet")
                continue
            state.say(f"{step.name}: starts")
            try:
                result = resolve(step.entry)(state) or StepResult()
            except Exception as error:  # a crash or a refusal: the run stops, and a resume is a new go
                shown = describe(error)
                record.stop(step.name, shown, state.stamp())
                where = f" on {shown['item']}" if shown["item"] else ""
                said = f": {shown['message']}" if shown.get("message") else ""
                state.say(f"{step.name}: stopped{where} by {shown['error']}{said}")
                return Outcome(step.name, shown["error"], shown["item"])
            record.end_step("done", state.stamp(), result.requests, result.detail)
            state.say(f"{step.name}: done")
        record.finish(state.stamp())
        state.say(f"go {number} ran every step")
    return Outcome()
