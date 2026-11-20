"""What the final evaluation's record must hold, and the holds that tie test's tables to it (PLAN_v4.md Amendment A19, items
10 and 13; issue #77, #92).

`analysis/final_record.py` writes the record; this module says what it must contain and checks it. `verify` is the one entry
point: before the go the rehearsal's record goes through it in CI (`analysis/check_final_record.py`); after the go the
committed record does (#95), with the committed results folder, the final evaluation's folder and `tables_test.json`.

**The fields** (`FIELDS`, one per line, grouped under item 10's bullets). Each is a path into a go of the record, and the
record holds it only if the path resolves and the value is well formed. The unlock's fields are held for every go (a resume's
unlock sits beside the first), every other field for the last go, which must have run every step.

**The files.** `file_hashes(results, output)` is item 10's last bullet: the sha256 of every test record file, that is the
Stage 8 `*_test.jsonl` and `*_drift.jsonl` files in `results/gates/` and `results/jev/` (`analysis/record_files.py`), E's and F's
router scores, test's results matrix, and the final evaluation's own record files (the liveness and replay calls, the replay's
summaries, the Sonnet batch log and `tables_test.json`). It names every file the final evaluation writes, so a file that is
missing or not named is found. The closing step records them (`analysis/final_run.py`, `record_step`), and `file_problems`
holds each file on disk to its entry.

**The holds** (`held_problems`) tie `tables_test.json` to the record by a second source each: the item counts, the stops the
tables report (refusals and truncations), the spend against each cap recomputed from the stored records, the order of every
step's time, and the request counts against the stored files. They compare two things that were counted apart, so a table
that disagrees with the record, or a record with a number moved, is found.

The record's steps before `tables` hold no outcome (A19 item 5). The refusals and truncations (`stops`, the extras' counts) are
written by the tables step, which is the first allowed to show them.
"""

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from analysis import final_grading, final_replay, final_routers, final_sonnet, record_files, unlock
from analysis.drift_rule import CUTOFF, THRESHOLD
from analysis.final_ledger import CAPS, LIVE_FILES, recorded_spend, report
from analysis.final_run import STEPS
from analysis.offline_half import NOT_STOPS, TABLES_FILE
from analysis.record_files import read_jsonl
from analysis.router_names import G_PLUS, NAMED_ROUTERS, G
from routers.ladder import LADDER

CONFIGS = LADDER  # A's ladder, in price order
OUTPUT_FOLDER = record_files.RECORD_FOLDER  # the key prefix of the final evaluation's own files in `files`
# The unlock's checks, in the order `analysis/unlock.py` runs them, which is the order its entry keeps them.
UNLOCK_CHECKS = unlock.CHECK_NAMES
SHA256 = re.compile(r"[0-9a-f]{64}")


# ---- paths into the record --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StepNamed:
    """The step called `name` among a go's steps."""

    name: str


@dataclass(frozen=True)
class CheckNamed:
    """The entry called `name` in a list of unlock checks (the node before it in the path)."""

    name: str


FieldPath = tuple[Any, ...]  # keys, with a `StepNamed` or `CheckNamed` where a list holds named entries


def resolve(go: Mapping[str, Any], path: FieldPath) -> Any:
    """The value `path` names in a go; raises KeyError, IndexError or TypeError where the record doesn't hold it."""
    node: Any = go
    for key in path:
        if isinstance(key, StepNamed):
            node = next(s for s in node["steps"] if s["name"] == key.name)
        elif isinstance(key, CheckNamed):
            node = next(c for c in node if c["name"] == key.name)
        else:
            node = node[key]
    return node


def delete_at(go: dict[str, Any], path: FieldPath) -> None:
    """Remove the leaf `path` names (the known positive for a field gone)."""
    del resolve(go, path[:-1])[path[-1]]


# ---- what a well formed value looks like ------------------------------------------------------------------------------


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def text(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def moment(value: Any) -> bool:
    try:
        datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return True


def keyed(value: Any, keys: Sequence[str], valid: Callable[[Any], bool]) -> bool:
    """A mapping that holds each of `keys` with a value `valid` accepts."""
    return isinstance(value, Mapping) and all(key in value and valid(value[key]) for key in keys)


def measures_ok(value: Any) -> bool:
    """A drift check's D6, D8 and shift for each key it measured."""
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(keyed(m, ("d6", "d8", "shift"), number) for m in value.values())
    )


def flips_of(dial: str) -> Callable[[Any], bool]:
    """The decisions that flip at each operating dial, a router's `threshold` or a gate's `cutoff` (the key the entry holds it
    under): one entry each, even if none flipped."""

    def flips_ok(value: Any) -> bool:
        return (
            isinstance(value, list)
            and bool(value)
            and all(
                isinstance(f, Mapping)
                and dial in f
                and "points" in f
                and isinstance(f.get("count"), int)
                and "items" in f
                for f in value
            )
        )

    return flips_ok


def steps_ok(value: Any) -> bool:
    """Every step of the runner, in its order, finished, with its start, end and request counts."""
    return (
        isinstance(value, list)
        and [s.get("name") for s in value] == [step.name for step in STEPS]
        and all(
            moment(s.get("started")) and moment(s.get("ended")) and s.get("status") == "done"
            and keyed(s.get("requests"), ("sent", "stored"), lambda n: isinstance(n, int))
            for s in value
        )
    )  # fmt: skip


def spend_ok(value: Any) -> bool:
    return keyed(value, tuple(CAPS), lambda pool: keyed(pool, ("cap", "recorded", "projected"), number))


def files_ok(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(h, Mapping) and SHA256.fullmatch(str(h.get("sha256"))) and isinstance(h.get("bytes"), int)
        for h in value.values()
    )  # fmt: skip


def batches_ok(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(keyed(b, ("id", "created_at"), text) for b in value)


def shares_ok(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(number(v) for v in value.values())
        and abs(sum(value.values()) - 1) < 1e-9
    )


def stops_ok(value: Any) -> bool:
    return keyed(
        value,
        CONFIGS,
        lambda by_status: isinstance(by_status, Mapping) and all(isinstance(n, int) for n in by_status.values()),
    )


def replay_results_ok(value: Any) -> bool:
    """For each policy, the first call and any reconnects (and any request a crash interrupted) reported apart."""
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(r, Mapping) and keyed(r.get("live"), ("first_call_s", "reconnects_s", "interrupted_s"), lambda t: isinstance(t, list))
        for r in value.values()
    )  # fmt: skip


def check_list_ok(value: Any) -> bool:
    return isinstance(value, list) and [c.get("name") for c in value] == list(UNLOCK_CHECKS) and all(
        isinstance(c.get("detail"), Mapping) for c in value
    )  # fmt: skip


def items_ok(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(text(i) for i in value) and len(set(value)) == len(value)


def hook_ok(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("passed") is True
        and value.get("exit_code") == 0
        and "seconds" in value
        and "last_line" in value
    )


def environment_ok(value: Any) -> bool:
    return keyed(value, ("os", "chip", "math library", "python"), text) and bool(value.get("libraries"))


def present(value: Any) -> bool:
    return value is not None and value != {} and value != []


# ---- the fields -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    name: str
    path: FieldPath
    valid: Callable[[Any], bool] = present
    each_go: bool = False  # held for every go (the unlock's), not only the last
    damage: Callable[[dict[str, Any]], None] | None = (
        None  # a break that leaves every other field as it was; default: delete the leaf
    )


def cut_ended(go: dict[str, Any]) -> None:
    del go["steps"][3]["ended"]


TABLES, DETAIL = StepNamed("tables"), "detail"
EXTRAS = (TABLES, DETAIL, "extras")
UNLOCK = ("unlock",)
JEV_ROUTERS = (StepNamed("drift_jev_router"), DETAIL, "routers")
LIVE_REPLAY = (StepNamed("replay"), DETAIL)
CLOSING = (StepNamed("record"), DETAIL)

# One field per line, under the bullet of A19 item 10 it comes from.
FIELDS: tuple[Field, ...] = (
    # The unlock's commit, time and hook result, each check's result, and the environment (every go's)
    Field("unlock.commit", (*UNLOCK, "commit"), lambda v: isinstance(v, str) and re.fullmatch(r"[0-9a-f]{40}", v) is not None, True),
    Field("unlock.time", (*UNLOCK, "time"), moment, True),
    Field("unlock.hook", (*UNLOCK, "hook"), hook_ok, True),
    Field("unlock.checks", (*UNLOCK, "checks"), check_list_ok, True),
    Field("unlock.environment", (*UNLOCK, "environment"), environment_ok, True),
    # The liveness calls' returned model IDs
    Field("liveness.models", (StepNamed("liveness"), DETAIL, "models"), lambda v: keyed(v, ("haiku", "opus"), text)),
    # The drift checks' numbers: D6, D8 and the shifts, the decisions that flip at each operating threshold and cutoff, and F's largest difference
    Field("drift.F.largest_difference", (StepNamed("drift_trained_router"), DETAIL, "largest_difference"), number),
    Field("drift.F.flips", (StepNamed("drift_trained_router"), DETAIL, "flips"), flips_of(THRESHOLD)),
    Field("drift.G.measures", (*JEV_ROUTERS, G, "measures"), measures_ok),
    Field("drift.G.flips", (*JEV_ROUTERS, G, "flips"), flips_of(THRESHOLD)),
    Field("drift.G+.measures", (*JEV_ROUTERS, G_PLUS, "measures"), measures_ok),
    Field("drift.G+.flips", (*JEV_ROUTERS, G_PLUS, "flips"), flips_of(THRESHOLD)),
    Field("drift.jev_gate.measures", (StepNamed("drift_jev_gate"), DETAIL, "measures"), measures_ok),
    Field("drift.jev_gate.flips", (StepNamed("drift_jev_gate"), DETAIL, "flips"), flips_of(CUTOFF)),
    Field("drift.sonnet_gate.measures", (StepNamed("drift_sonnet_gate"), DETAIL, "measures"), measures_ok),
    Field("drift.sonnet_gate.flips", (StepNamed("drift_sonnet_gate"), DETAIL, "flips"), flips_of(CUTOFF)),
    # Each step's start and end and its request counts; the refusals and truncations; the spend against each cap
    Field("steps", ("steps",), steps_ok, damage=cut_ended),
    Field("stops", (*EXTRAS[:2], "stops"), stops_ok),
    Field("truncations.F", (*EXTRAS, "trained_facts", "truncated"), lambda v: isinstance(v, int)),
    Field("refusals.jev_routers", (*EXTRAS, "jev_facts"), lambda v: keyed(v, (G, G_PLUS), lambda r: keyed(r, ("calls", "refused", "cut"), lambda n: isinstance(n, int)))),
    Field("refusals.gates", (*EXTRAS, "gate_table"), lambda v: keyed(v, CONFIGS, lambda c: keyed(c, ("rejected", "jev_refused", "jev_cut", "sonnet_refused", "sonnet_cap_stops"), lambda n: isinstance(n, int)))),
    Field("spend", (*CLOSING, "spend"), spend_ok),
    # The order evidence: the first test Jev call's server Date, and the Sonnet test batch's ID and creation time
    Field("order.jev_date", (StepNamed("calls_jev_router"), DETAIL, "first_test_jev_date"), text),
    Field("order.sonnet_batches", (StepNamed("calls_sonnet_gate"), DETAIL, "batches"), batches_ok),
    # F's fold shares on test, and the live-replay items drawn at the unlock
    Field("folds.F", (*EXTRAS, "trained_facts", "fold_shares"), shares_ok),
    Field("replay.items", (*UNLOCK, "checks", CheckNamed("replay_draw"), DETAIL, "item_ids"), items_ok, True),
    # The live replay's conditions, with the first call and any reconnects apart
    Field("replay.conditions", (*LIVE_REPLAY, "conditions"), lambda v: keyed(v, ("machine", "network", "go_started", "go_ended"), text)),
    Field("replay.calls", (*LIVE_REPLAY, "results"), replay_results_ok),
    # The sha256 of every test record file
    Field("files", (*CLOSING, "files"), files_ok),
)  # fmt: skip


def last_go(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record["gos"][-1]


def field_problems(record: Mapping[str, Any]) -> list[str]:
    """The fields the record lacks or holds malformed, each named by its go: `go 2: unlock.commit`. An empty list means the
    record holds every field of A19 item 10."""
    gos = record.get("gos")
    if not isinstance(gos, list) or not gos:
        return ["gos"]
    found = []
    for number_, go in enumerate(gos, start=1):
        for field in FIELDS:
            if not field.each_go and go is not gos[-1]:
                continue
            try:
                ok = field.valid(resolve(go, field.path))
            except (KeyError, IndexError, TypeError, StopIteration):
                ok = False
            if not ok:
                found.append(f"go {number_}: {field.name}")
    return found


# ---- the files --------------------------------------------------------------------------------------------------------


def named_files(results: Path, output: Path) -> dict[str, Path]:
    """Every test record file the final evaluation writes, by the name the record keeps: the Stage 8 files of the gate and Jev
    folders (the ones on disk, and each one `STAGE_8_FILES` names), E's and F's scores and test's matrix beside them, and the
    final evaluation's own files in its folder. A name's file may not exist; `file_hashes` refuses that."""
    files = {
        f"{folder}/{name}": results / folder / name
        for folder, names in record_files.STAGE_8_FILES.items()
        for name in names
    }
    for path in record_files.stage_8_files(results):
        files[f"{path.parent.name}/{path.name}"] = path
    files[final_routers.SCORES_FILE] = results / final_routers.SCORES_FILE
    files[final_grading.TEST_MATRIX] = results / final_grading.TEST_MATRIX
    for name in (*LIVE_FILES, final_replay.REQUESTS_FILE, final_sonnet.BATCH_LOG, TABLES_FILE.format(split="test")):
        files[f"{OUTPUT_FOLDER}/{name}"] = output / name
    return dict(sorted(files.items()))


class FilesError(Exception):
    """A test record file the final evaluation should have written is not there."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(results: Path, output: Path) -> dict[str, dict[str, Any]]:
    """The sha256 and size of every test record file, by name. Refuses (FilesError) when one that should exist doesn't."""
    named = named_files(results, output)
    absent = [name for name, path in named.items() if not path.is_file()]
    if absent:
        raise FilesError(f"test record files missing: {', '.join(absent)}")
    return {name: {"sha256": sha256_of(path), "bytes": path.stat().st_size} for name, path in named.items()}


def file_problems(kept: Mapping[str, Mapping[str, Any]], results: Path, output: Path) -> list[str]:
    """The record's `files` against the files on disk: a file with no entry, an entry with no file, and a hash or size that
    differs."""
    named = named_files(results, output)
    problems = [f"{name} is a test record file with no entry in the record" for name in named if name not in kept]
    for name, entry in kept.items():
        path = named.get(name)
        if path is None:
            problems.append(f"the record names {name}, which is not a test record file")
        elif not path.is_file():
            problems.append(f"the record names {name}, which is not on disk")
        elif sha256_of(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            problems.append(f"{name} differs from its sha256 in the record")
    return problems


# ---- the holds --------------------------------------------------------------------------------------------------------


def stamps(go: Mapping[str, Any]) -> list[str]:
    """The go's times in the order they were taken: its start, each step's start and end, its end."""
    return [go["started"], *(t for s in go["steps"] for t in (s["started"], s["ended"])), go["ended"]]


def held_problems(record: Mapping[str, Any], tables: Mapping[str, Any], results: Path, output: Path) -> list[str]:
    """What disagrees between the tables, the record and the stored files. The record must hold every field (`field_problems`)."""
    go = last_go(record)
    step = {s["name"]: s for s in go["steps"]}
    detail = {name: s["detail"] for name, s in step.items()}
    extras = detail["tables"]["extras"]
    n = tables["n"]
    found: list[str] = []

    def expect(ok: bool, what: str) -> None:
        if not ok:
            found.append(what)

    # counts: every count that is the number of test items, from the tables, the steps and the stored files
    expect(go["stopped"] is None, f"the last go stopped at {go['stopped'] and go['stopped'].get('step')}")
    expect(detail["tables"]["items"] == n, f"the tables step records {detail['tables']['items']} items, the tables {n}")
    expect(
        extras["trained_facts"]["items"] == n,
        f"F's scores cover {extras['trained_facts']['items']} items, the tables {n}",
    )
    expect(all(extras["jev_facts"][r]["calls"] == n for r in (G, G_PLUS)), f"G's and G+'s calls are not {n} each")
    configs = tuple(tables["parts"]["stops"])
    expect(configs == CONFIGS, f"the tables' stops cover {configs}")
    expect(
        all(sum(tables["parts"]["stops"][c].values()) == n for c in configs),
        f"a config's graded answers are not {n} in the tables' stops",
    )
    expect(
        detail["grading"]["stored"] == n * len(configs),
        f"grading stored {detail['grading']['stored']}, not {n} x {len(configs)}",
    )
    expect(
        all(extras["gate_table"][c]["answers"] == n for c in configs), f"a config's free-checked answers are not {n}"
    )
    sent = {name: s["requests"]["sent"] + s["requests"]["stored"] for name, s in step.items()}
    free, jev_gate_calls = (
        read_jsonl(results / "gates" / "free_checks_test.jsonl"),
        read_jsonl(results / "gates" / "jev_gate_test.jsonl"),
    )
    sonnet_calls = read_jsonl(results / "gates" / "sonnet_gate_test.jsonl")
    routers = [read_jsonl(results / "jev" / f"{name}_test.jsonl") for name in (NAMED_ROUTERS[G], NAMED_ROUTERS[G_PLUS])]
    expect(
        sent["calls_free_checks"] == len(free) == n * len(configs),
        f"the free checks: {sent['calls_free_checks']} requests, {len(free)} stored, {n} x {len(configs)} answers",
    )
    first_pass = [r for r in jev_gate_calls if r["pass"] == 1]
    gate = extras["gate_table"]
    expect(
        sent["calls_jev_gate"] == len(first_pass) == sum(gate[c]["jev_calls"] for c in configs),
        f"the Jev gate: {sent['calls_jev_gate']} requests, {len(first_pass)} stored, {sum(gate[c]['jev_calls'] for c in configs)} in the tables",
    )
    expect(
        len(sonnet_calls) == sum(gate[c]["sonnet_calls"] for c in configs),
        f"the Sonnet gate: {len(sonnet_calls)} stored verdicts, {sum(gate[c]['sonnet_calls'] for c in configs)} in the tables",
    )
    expect(
        sent["calls_jev_router"] == sum(len(r) for r in routers) == 2 * n,
        f"G and G+: {sent['calls_jev_router']} requests, {[len(r) for r in routers]} stored, 2 x {n} requests",
    )
    replay_items = [c for c in go["unlock"]["checks"] if c["name"] == "replay_draw"][0]["detail"]["item_ids"]
    expect(
        detail["replay"]["items"] == len(replay_items),
        f"the replay used {detail['replay']['items']} items, the unlock drew {len(replay_items)}",
    )

    # refusals and truncations: the record's stops are the tables' non-finishing statuses, counted again here from the tables
    again = {
        c: {s: k for s, k in by_status.items() if s not in NOT_STOPS}
        for c, by_status in tables["parts"]["stops"].items()
    }
    expect(detail["tables"]["stops"] == again, "the record's refusals and truncations differ from the tables' stops")
    expect(
        sum(gate[c]["jev_refused"] for c in configs) == sum(bool(r["refused"]) for r in first_pass),
        "the Jev gate's refusals in the tables are not those in its stored calls",
    )
    expect(
        sum(gate[c]["sonnet_refused"] for c in configs) == sum(bool(r["refused"]) for r in sonnet_calls),
        "the Sonnet gate's refusals in the tables are not those in its stored verdicts",
    )
    expect(
        sum(gate[c]["sonnet_cap_stops"] for c in configs)
        == sum(r["stop_reason"] == "max_tokens" for r in sonnet_calls),
        "the Sonnet gate's cap stops in the tables are not those in its stored verdicts",
    )

    # spend: what the stored records say, against each cap, and never falling from step to step
    closing = detail["record"]["spend"]
    recomputed = report(recorded_spend(results, output))
    expect(closing == recomputed, f"the recorded spend {closing} is not what the stored records give, {recomputed}")
    expect(
        all(closing[pool]["recorded"] <= cap for pool, cap in CAPS.items()),
        f"the recorded spend is over a cap: {closing}",
    )
    seen = [s["detail"]["spend"] for s in go["steps"] if "spend" in s["detail"]]
    expect(
        all(a[pool]["recorded"] <= b[pool]["recorded"] for a, b in zip(seen, seen[1:]) for pool in CAPS),
        "a step's recorded spend is less than an earlier step's",
    )

    # step times: in the order they were taken, with the unlock and the replay's own stamps inside their steps
    times = [datetime.fromisoformat(t) for t in stamps(go)]
    expect(times == sorted(times), "the go's and the steps' times are not in order")
    expect(
        datetime.fromisoformat(step["unlock"]["started"])
        <= datetime.fromisoformat(go["unlock"]["time"])
        <= datetime.fromisoformat(step["unlock"]["ended"]),
        "the unlock's time is outside the unlock step",
    )
    conditions = detail["replay"]["conditions"]
    expect(
        datetime.fromisoformat(step["replay"]["started"]) <= datetime.fromisoformat(conditions["go_started"])
        <= datetime.fromisoformat(conditions["go_ended"]) <= datetime.fromisoformat(step["replay"]["ended"]),
        "the replay's own times are outside the replay step",
    )  # fmt: skip
    return found


def verify(record: Mapping[str, Any], tables: Mapping[str, Any], results: Path, output: Path) -> list[str]:
    """Every problem with a record against its tables and files: the fields it lacks (then nothing else is looked at), the
    files that don't match their hashes, and what disagrees with the tables. Empty if the record is whole."""
    lacking = field_problems(record)
    if lacking:
        return [f"the record lacks {name}" for name in lacking]
    return [*file_problems(last_go(record)["steps"][-1]["detail"]["files"], results, output),
            *held_problems(record, tables, results, output)]  # fmt: skip
