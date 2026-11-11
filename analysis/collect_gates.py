"""Collect the gates' verdicts and store them (PLAN_v4.md §9, §13; Amendment A17, items 14 and 20). Free checks first.

Run from the project root:  uv run python -m analysis.collect_gates free --split train|dev
                            uv run python -m analysis.collect_gates free --split smoke --smoke-file made_up.json

Storage is JSON Lines under `results/gates/`, uncommitted, like `results/jev/`: `free_checks_{split}.jsonl` holds one
record per answer, with the keys every gate record has (`GATE_KEYS`: item, config, run 0, split, gate, pass) and then
the free checks' fields (`routers.gates.FREE_FIELDS`). The paid gates' files (`jev_gate_{split}.jsonl`,
`sonnet_gate_{split}.jsonl`) go beside it, written by their own collection scripts through `append_record`,
`read_gate_records` and `gate_path` here. Smoke records, for made-up answers, go in `*_smoke.jsonl`; test records
belong to Stage 8 and nothing here writes or reads them. A rerun skips answers already stored. A run stops (SandboxError,
Docker failing after its retries) before a verdict is written, so no infrastructure trouble becomes a verdict.

Made-up answers are the only answers any gate sees before the settings commit (A17 item 20): the smoke file is a JSON
list of `{"item_id", "config", "prompt", "output_schema", "answer", "stop_reason"}` objects.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from analysis.collect_jev import RESULTS_DIR, append_record, now_local, trim_torn_line
from analysis.gate_answers import read_answers, requests_of
from harness.collect import MAIN
from routers.gates import (
    FREE_FIELDS,
    CodeCheckSettings,
    Runner,
    free_checks,
    image_digest,
    read_gate_settings,
    sandbox_runner,
)
from routers.request import Request

SPLITS = ("train", "dev", "smoke")  # test is Stage 8's, after its unlock
GATE_KEYS = ("item_id", "config", "run", "split", "gate", "pass")
FREE = "free"  # a free-check record's `gate`
FREE_RECORD_FIELDS = (*GATE_KEYS, *FREE_FIELDS)


class GateCollectError(ValueError):
    """A collection or a stored file that is refused. Nothing is added."""


@dataclass(frozen=True)
class GateInput:
    """One answer to be judged, with its request: what every gate receives."""

    item_id: str
    config: str
    request: Request
    answer: str
    stop_reason: str | None


@dataclass(frozen=True)
class Summary:
    new: int
    skipped: int


# ---- storage --------------------------------------------------------------------------------------------------


def gate_path(name: str, split: str, results_dir: Path = RESULTS_DIR) -> Path:
    """`results/gates/{name}_{split}.jsonl`, for `name` such as `free_checks`, `jev_gate` or `sonnet_gate`."""
    if split not in SPLITS:
        raise GateCollectError(f"unknown split {split!r}; the gates' files are for {SPLITS}")
    return results_dir / "gates" / f"{name}_{split}.jsonl"


def read_gate_records(
    path: Path, fields: tuple[str, ...], split: str, gate: str, passes: Collection[int] = (1,)
) -> list[dict]:
    """The records of one JSON Lines file (none if it doesn't exist), refused whole (GateCollectError) unless each is
    a record of exactly `fields`, is `split`'s and `gate`'s, is run 0, has a pass in `passes` and is the only record
    of its (item, config, run, pass). A last line with no newline is a write cut short, and is dropped."""
    if not path.exists():
        return []
    lines = path.read_bytes().decode("utf-8").split("\n")[:-1]
    records, seen = [], set()
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except ValueError as error:
            raise GateCollectError(f"{path.name} line {number} is not JSON") from error
        if not isinstance(record, dict) or tuple(record) != fields:
            raise GateCollectError(f"{path.name} line {number} is not a stored record: its fields differ")
        if record["split"] != split or record["gate"] != gate or record["run"] != 0:
            raise GateCollectError(f"{path.name} line {number} is not a {gate} record of run 0 for {split}")
        if record["pass"] not in passes:
            raise GateCollectError(f"{path.name} line {number} has pass {record['pass']!r}, not one of {tuple(passes)}")
        key = (record["item_id"], record["config"], record["pass"])
        if key in seen:
            raise GateCollectError(f"{path.name} line {number} repeats {key}")
        seen.add(key)
        records.append(record)
    return records


def read_free_records(split: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    """A split's stored free-check records, in the order stored: the one reader of them."""
    return read_gate_records(gate_path("free_checks", split, results_dir), FREE_RECORD_FIELDS, split, FREE)


# ---- the free checks ------------------------------------------------------------------------------------------


def gate_inputs(split: str, benchmark_file: Path | None = None, store: Path = MAIN.matrix) -> list[GateInput]:
    """Train's or dev's answers with their requests, through the gates' answer reader and the benchmark reader.
    Refuses the test split (SealedItemError). `benchmark_file` and `store` are for a check to feed in."""
    requests = requests_of(split, benchmark_file)
    return [GateInput(a.item_id, a.config, requests[a.item_id], a.answer, a.stop_reason)
            for a in read_answers(split, store, benchmark_file)]  # fmt: skip


def read_smoke_inputs(path: Path) -> list[GateInput]:
    """Made-up answers from a JSON file (see the module's docstring), refused if malformed."""
    keys = {"item_id", "config", "prompt", "output_schema", "answer", "stop_reason"}
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or any(not isinstance(e, dict) or set(e) != keys for e in entries):
        raise GateCollectError(f"{path} must be a list of objects with the keys {sorted(keys)}")
    return [GateInput(e["item_id"], e["config"], Request(e["prompt"], e["output_schema"]), e["answer"], e["stop_reason"])
            for e in entries]  # fmt: skip


def collect_free(
    inputs: Sequence[GateInput],
    *,
    split: str,
    settings: CodeCheckSettings,
    digest: str,
    results_dir: Path = RESULTS_DIR,
    run: Runner = sandbox_runner,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], str] = now_local,
    progress: Callable[[int, int], None] | None = None,
) -> Summary:
    """Run the three free checks on every input not yet stored for `split`, appending each record as it is made.

    Refuses (GateCollectError) a repeated (item, config), before the first check. A SandboxError, raised once the
    retries are spent, ends the run; the records stored before it stay valid and a rerun resumes after them."""
    path = gate_path("free_checks", split, results_dir)
    keys = [(i.item_id, i.config) for i in inputs]
    if len(set(keys)) != len(keys):
        raise GateCollectError("the same answer is in the list more than once")
    trim_torn_line(path)
    stored = {(r["item_id"], r["config"]) for r in read_free_records(split, results_dir)}
    todo = [i for i in inputs if (i.item_id, i.config) not in stored]
    for number, item in enumerate(todo, start=1):
        fields = free_checks(item.request, item.answer, item.stop_reason, settings, digest=digest, now=now, run=run,
                             sleep=sleep)  # fmt: skip
        key = {"item_id": item.item_id, "config": item.config, "run": 0, "split": split, "gate": FREE, "pass": 1}
        append_record(path, key | fields)
        if progress:
            progress(number, len(todo))
    return Summary(len(todo), len(inputs) - len(todo))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("gate", choices=["free"])
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--smoke-file", type=Path)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if (args.split == "smoke") != (args.smoke_file is not None):
        parser.error("--smoke-file goes with --split smoke, and only with it")
    inputs = read_smoke_inputs(args.smoke_file) if args.split == "smoke" else gate_inputs(args.split)
    settings = read_gate_settings().code_check
    summary = collect_free(inputs, split=args.split, settings=settings, digest=image_digest(settings.image),
                           results_dir=args.results_dir)  # fmt: skip
    print(f"{summary.new} answers checked, {summary.skipped} already stored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
