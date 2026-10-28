"""What the train collection left behind, measured from the stored responses (PLAN_v4.md §8 3c; Amendment A15, items 9,
10 and 19; issue #39): the call, refusal, truncation and retry counts, token usage, latency, the first call's server
`Date`, and whether every stored body hash is what the frozen call building gives.

Run from the project root:  uv run python -m analysis.jev_train_report

Train's stored records only (`results/jev/jev_train.jsonl` and `jev_bolstered_train.jsonl`, both passes), through the
collection script's own reader, and train's requests through the benchmark reader, which refuses test. It calls no Jev.
A15 item 19 is filled from this printout and `python -m analysis.fit_jev_router`, and `analysis/check_jev_freeze.py`
compares the record with both.

- **Counts:** refusals are the stored flag (read from the response when it was stored, item 9); retries are the attempts
  beyond each call's first; a reconnect is a call that opened a new connection, the first call apart.
- **Truncation** isn't stored: the cut is a pure function of the request and the frozen budget, so each call is built
  again, which also shows whether its body hashes as stored.
- **Latency** is the successful attempt's time alone (item 10). The first call and every reconnect are left out of the
  median and p90, and reported beside them. The p90 is numpy's percentile with linear interpolation.
"""

import email.utils
import hashlib
import ipaddress
import platform
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from analysis import collect_jev
from analysis.benchmark import read_rows
from analysis.router_names import LETTER_OF
from routers import jev_call
from routers.request import Request, from_row

ROUTERS = collect_jev.ROUTERS


class ReportError(ValueError):
    """Stored records or a request this report refuses whole: a Date that isn't one, an unknown item, no calls left."""


@dataclass(frozen=True)
class Counts:
    calls: int
    pass_two: int  # of them, the noise pass's
    refused: int
    retries: int  # attempts beyond each call's first
    failed_attempts: int
    reconnects: int  # calls that opened a new connection, the first call apart
    input_tokens: int  # in all
    median_tokens: float  # per call
    max_tokens: int
    models: tuple[str, ...]  # the model IDs seen, sorted


@dataclass(frozen=True)
class Latency:
    median: float  # seconds
    p90: float
    count: int  # calls these two come from
    first: float  # the first call's time
    reconnects: list[float]  # each later call that opened a new connection
    by_task: dict[str, tuple[float, float, int]]  # task -> median, p90, count


@dataclass(frozen=True)
class Rebuilt:
    mismatched: list[str]  # item IDs whose stored body hash isn't what the frozen code builds
    cut: int  # calls whose prompt the byte budget cut


def call_counts(records: Sequence[Mapping[str, Any]]) -> Counts:
    """The counts of one router's stored records."""
    tokens = [collect_jev.input_tokens(r) for r in records]
    return Counts(
        calls=len(records),
        pass_two=sum(r["pass"] == 2 for r in records),
        refused=sum(bool(r["refused"]) for r in records),
        retries=sum(int(r["attempts"]) - 1 for r in records),
        failed_attempts=sum(len(r["failed_attempts"]) for r in records),
        reconnects=sum(bool(r["new_connection"]) for r in records[1:]),
        input_tokens=sum(tokens),
        median_tokens=statistics.median(tokens),
        max_tokens=max(tokens),
        models=tuple(sorted({str(r["model"]) for r in records})),
    )


def summary(seconds: Sequence[float]) -> tuple[float, float]:
    return float(np.median(seconds)), float(np.percentile(seconds, 90))


def latency(records: Sequence[Mapping[str, Any]], task_of: Mapping[str, str]) -> Latency:
    """Median and p90 of the successful attempts' times, in the order stored: the first call and every call that opened
    a new connection are set apart, overall and per task. Refused whole if nothing is left or a task is unknown."""
    kept: list[tuple[str, float]] = []
    reconnects: list[float] = []
    for number, r in enumerate(records):
        if number > 0 and r["new_connection"]:
            reconnects.append(float(r["seconds"]))
        elif number > 0:
            if r["item_id"] not in task_of:
                raise ReportError(f"no task for {r['item_id']}")
            kept.append((task_of[str(r["item_id"])], float(r["seconds"])))
    if not kept:
        raise ReportError("no calls are left once the first call and the reconnects are set apart")
    median, p90 = summary([s for _, s in kept])
    tasks = sorted({t for t, _ in kept})
    by_task = {}
    for task in tasks:
        times = [s for t, s in kept if t == task]
        by_task[task] = (*summary(times), len(times))
    return Latency(median, p90, len(kept), float(records[0]["seconds"]), reconnects, by_task)


def first_date(records: Sequence[Mapping[str, Any]]) -> datetime:
    """The server `Date` of the first stored call, as a time."""
    if not records:
        raise ReportError("no stored calls")
    try:
        return email.utils.parsedate_to_datetime(str(records[0]["server_date"]))
    except (TypeError, ValueError) as error:
        raise ReportError(f"the first call's Date is not a date: {records[0]['server_date']!r}") from error


def rebuild(
    router: str, records: Sequence[Mapping[str, Any]], requests: Mapping[str, Request], settings: jev_call.Settings
) -> Rebuilt:
    """Build each stored call again with the frozen code and the settings' budget: name those whose body hash differs
    from the stored one, and count the cuts."""
    if router not in collect_jev.CALLS:
        raise ReportError(f"unknown router {router!r}")
    build = collect_jev.CALLS[router].build
    mismatched, cut = [], 0
    for r in records:
        item = str(r["item_id"])
        if item not in requests:
            raise ReportError(f"no request for {item}")
        call = build(requests[item], settings)
        cut += call.cut_bytes > 0
        if call.sha256 != r["body_sha256"]:
            mismatched.append(item)
    return Rebuilt(mismatched, cut)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def machine() -> str:
    """The machine as the environment records it (A11 item 19's wording): chip, OS, architecture, Python."""
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return f"{chip or platform.processor()}, {platform.system()} {platform.release()} {platform.machine()}, Python {platform.python_version()}"


def collected_span(records: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """The earliest and latest `collected_at` stamp among the records, as stored (item 10 records dates and times)."""
    if not records:
        raise ReportError("no stored calls")
    try:
        stamps = sorted((datetime.fromisoformat(str(r["collected_at"])), str(r["collected_at"])) for r in records)
    except ValueError as error:
        raise ReportError(f"a call's collected_at is not a time: {error}") from error
    return stamps[0][1], stamps[-1][1]


def describe_network(route: str, ports: str) -> str:
    """The network as A15 item 19 words it for train: the default route's hardware port and device, whether its gateway is
    a private address, and that no network name or address is recorded. `route` is `route -n get default`'s output and
    `ports` is `networksetup -listallhardwareports`'s. Neither the gateway's address nor a hardware address is kept."""
    fields = {k.strip(): v.strip() for k, _, v in (line.partition(":") for line in route.splitlines())}
    device = fields.get("interface")
    if not device:
        return "not determined"
    name = None
    for block in ports.split("\n\n"):
        lines = {k.strip(): v.strip() for k, _, v in (line.partition(":") for line in block.splitlines())}
        if lines.get("Device") == device:
            name = lines.get("Hardware Port")
    try:
        private = ipaddress.ip_address(fields.get("gateway", "")).is_private
    except ValueError:
        private = False
    where = f"{name} ({device})" if name else device
    return f"{where}{' behind a private gateway' if private else ''}, no network name or address recorded"


def network() -> str:
    """This machine's network, described by `describe_network` from the system's own tools ("not determined" where
    they are missing)."""
    try:
        route = subprocess.run(["route", "-n", "get", "default"], capture_output=True, text=True, check=False).stdout
        ports = subprocess.run(
            ["networksetup", "-listallhardwareports"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return "not determined"
    return describe_network(route, ports)


def display(router: str) -> str:
    return LETTER_OF[router]


def record_lines(router: str, counts: Counts, rebuilt: Rebuilt, timing: Latency) -> tuple[str, str]:
    """The calls entry and the latency entry of A15 item 19 for one router, in the words the record uses. The record
    is written from these lines, and the check looks for them in it."""
    models = " and ".join(f"`{m}`" for m in counts.models)
    calls = (
        f"{display(router)}: {counts.calls} calls ({counts.calls - counts.pass_two} in pass 1, {counts.pass_two} in "
        f"pass 2); returned model IDs {models}; {len(rebuilt.mismatched)} stored body hashes differ from the frozen "
        f"code; refused {counts.refused}, cut {rebuilt.cut}, retries {counts.retries} ({counts.failed_attempts} failed "
        f"attempts), reconnects {counts.reconnects}; input tokens {counts.input_tokens} in all, median "
        f"{counts.median_tokens:g} per call, largest {counts.max_tokens}."
    )
    reconnects = ", ".join(f"{s * 1000:.1f}" for s in timing.reconnects) or "none"
    tasks = "; ".join(f"{task} {m * 1000:.1f}, {p * 1000:.1f}, {n}" for task, (m, p, n) in timing.by_task.items())
    latency_line = (
        f"{display(router)}: over {timing.count} calls, median {timing.median * 1000:.1f} ms and p90 "
        f"{timing.p90 * 1000:.1f} ms; the first call {timing.first * 1000:.1f} ms; reconnects (ms): {reconnects}. Per "
        f"task (median, p90 in ms, calls): {tasks}."
    )
    return calls, latency_line


def main() -> int:
    loaded = collect_jev.load_settings()
    train = read_rows(split="train")
    task_of = {row["item_id"]: row["source"] for row in train}
    requests = {row["item_id"]: from_row(row) for row in train}
    for router in ROUTERS:
        path = collect_jev.records_path(router, "train")
        records = collect_jev.read_records(router, "train")
        counts = call_counts(records)
        rebuilt = rebuild(router, records, requests, loaded.call)
        calls, timing = record_lines(router, counts, rebuilt, latency(records, task_of))
        print(f"calls   {calls}\nlatency {timing}")
        print(f"  server Date of the first call {records[0]['server_date']}, of the last {records[-1]['server_date']}")
        print(f"  {path.relative_to(collect_jev.PROJECT_ROOT).as_posix()}: {file_sha256(path)}")
    both = [collect_jev.read_records(router, "train") for router in ROUTERS]
    print(f"first train call, server Date: {min(first_date(r) for r in both).strftime('%a, %d %b %Y %H:%M:%S GMT')}")
    print(f"machine: {machine()}")
    print(f"network: {network()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
