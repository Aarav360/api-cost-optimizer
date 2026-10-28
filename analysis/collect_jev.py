"""The Jev collection script: requests in, calls through the router's client, stored records out (PLAN_v4.md §8 3c
and 3c+; Amendment A15, items 9, 10, 12 and 13; GitHub issue #30's stored record).

Requests come from one of two sources:
- the benchmark reader, for train or dev. It refuses test whole, so a test request can't reach Jev from here
  (AGENTS.md); this module names neither the benchmark file nor the sealed store;
- a JSON list of made-up requests, stored apart under the `smoke` split (A15 item 13: the smoke test, before the
  settings commit). No table builder reads the smoke split: `read_records` is how a builder reads records, and the
  builders ask it for train and dev only.

Calls go through `routers.jev.JevClient`, built from `routers/jev_call.py`'s bodies, so the bodies, retries and
refusal flag are the router's own. G+ needs no weights to be collected: the weights are fitted on the stored train
responses afterwards. A stored record is turned into router scores by `record_scores`, which takes the same path a
live call takes (`routers.jev.g_response_scores` and `g_plus_response_scores`).

Records. One JSON Lines file per router and split, `results/jev/<router>_<split>.jsonl` (uncommitted), one record per
call, in the fields and order of `STORED_FIELDS`. A record is appended and synced before the next call is made, so a
stop, a crash or a power cut leaves every earlier record valid; a last line torn mid-write is dropped on reading and
trimmed before the next append.

- Resumable (A15 item 9): a rerun never calls a request that is stored. A request is stored under its item ID and
  pass, so the noise sample's second send (pass 2, from `noise`) sits beside its pass 1 and never replaces it; the
  file is append-only and a repeated (item ID, pass) is refused on reading.
- Stopping (A15 item 9): the run stops at the retry cap and on any 4xx the client doesn't retry, raising a `RunStop`
  that names the request, and the command line exits 1 with that message. Nothing is lost.
- The noise pass (A15 item 10): the settings file's noise sample, sent a second time after the train collection
  finishes. It is refused unless every sampled item has its pass 1 stored for that router.
- Refusals are counted from the stored responses (`refused_items`), not from router scores, since Jev can itself answer
  0.00. Truncation (a prompt the byte budget cut) is counted per run and printed, since the record has no field for it.
- The key comes from the environment (`TYPESAFE_API_KEY`) and is in no record.

Overhead and the byte budget (A15 items 8 and 10) come from stored usage: `input_tokens` and `overhead_dollars` read a
record's input tokens and charge them at the settings' price. `measured_overhead_tokens` is the largest input token
count among records: an upper bound on Jev's fixed overhead when they are minimal calls, and for a call that carries
past requests it counts their tokens twice, which only makes a budget safer. `measured_budget_bytes` feeds it to
`routers.jev_call`'s budget functions with the settings' questions and past requests as the fixed parts.

Run from the project root (the key in `TYPESAFE_API_KEY`):

    uv run python -m analysis.collect_jev collect --router jev --split smoke --requests made_up.json --budget-bytes 20000
    uv run python -m analysis.collect_jev collect --router both --split train
    uv run python -m analysis.collect_jev noise --router both
    uv run python -m analysis.collect_jev collect --router both --split dev
    uv run python -m analysis.collect_jev report --router jev --split train
    uv run python -m analysis.collect_jev budget --router jev --split smoke --schema-file longest_schema.json

`--budget-bytes` is for one router and replaces the settings file's (null before the smoke test fills it in). Dev is
collected only after the freeze commit (A15 item 13); nothing here enforces that order.
"""

import argparse
import json
import os
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from analysis.benchmark import BenchmarkError, read_rows
from harness.client import PROJECT_ROOT
from routers import jev, jev_call
from routers.request import Request, from_row

RESULTS_DIR = PROJECT_ROOT / "results"
ROUTERS = ("jev", "jev_bolstered")  # the section names in A15 item 14: G and G+
SPLITS = ("train", "dev", "smoke")
PASSES = (1, 2)  # 2 is the noise pass
STORED_FIELDS = (
    "router", "split", "item_id", "pass", "body_sha256", "response", "model", "usage", "request_id", "server_date",
    "seconds", "attempts", "failed_attempts", "new_connection", "refused", "collected_at",
)  # fmt: skip  # issue #30's stored record, in its order


class CollectError(ValueError):
    """A request, file or option this script refuses whole, before any call is made."""


@dataclass(frozen=True)
class Item:
    """One request to send, under the ID it is stored by."""

    item_id: str
    request: Request


@dataclass(frozen=True)
class Loaded:
    """A settings file read three ways: what call building uses, what the client uses, and the raw data."""

    call: jev_call.Settings
    client: jev.ClientSettings
    data: dict

    @property
    def noise_item_ids(self) -> list[str]:
        sample = self.data.get("noise_sample")
        ids = sample.get("item_ids") if isinstance(sample, dict) else None
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise CollectError("the settings file's noise_sample needs a non-empty list of distinct item IDs")
        return ids


@dataclass(frozen=True)
class Summary:
    new: int  # calls made and stored this run
    skipped: int  # requests already stored, never called again
    refused: int  # of the new calls, those Jev refused (the flag the router read from the response)
    cut: int  # of the new calls, those whose prompt the byte budget cut


def load_settings(path: Path = jev_call.SETTINGS_FILE) -> Loaded:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Loaded(jev_call.parse_settings(data), jev.parse_client_settings(data), data)


# ---- requests in ----------------------------------------------------------------------------------------------


def benchmark_items(split: str, benchmark_file: Path | None = None) -> list[Item]:
    """A split's requests through the benchmark reader, which refuses test (SealedItemError) and any other split."""
    rows = read_rows(split=split) if benchmark_file is None else read_rows(benchmark_file, split=split)
    return [Item(row["item_id"], from_row(row)) for row in rows]


def read_smoke_requests(path: Path) -> list[Item]:
    """Made-up requests from a JSON list of `{"prompt": ..., "item_id": ..., "output_schema": ...}`. Only the prompt is
    required: the item ID defaults to smoke-001, smoke-002, ..., the schema to null. Refused whole if any is malformed."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise CollectError("the made-up requests are a non-empty JSON list")
    items = []
    for number, entry in enumerate(data, start=1):
        if not isinstance(entry, dict) or not set(entry) <= {"prompt", "item_id", "output_schema"}:
            raise CollectError(f"made-up request {number} must be an object of prompt, item_id and output_schema only")
        prompt, schema = entry.get("prompt"), entry.get("output_schema")
        item_id = entry.get("item_id", f"smoke-{number:03d}")
        if not isinstance(prompt, str) or not prompt:
            raise CollectError(f"made-up request {number} needs a non-empty prompt")
        if not isinstance(item_id, str) or not item_id:
            raise CollectError(f"made-up request {number} needs a non-empty item_id")
        if not (schema is None or isinstance(schema, dict)):
            raise CollectError(f"made-up request {number}'s output_schema must be an object or null")
        items.append(Item(item_id, Request(prompt, schema)))
    return items


def requests_for(split: str, smoke_file: Path | None = None, benchmark_file: Path | None = None) -> list[Item]:
    """The requests of a split: made-up ones for smoke, the benchmark reader's for train and dev. Test goes to the
    reader and is refused there."""
    if split == "smoke":
        if smoke_file is None:
            raise CollectError("the smoke split needs a file of made-up requests")
        return read_smoke_requests(smoke_file)
    if smoke_file is not None:
        raise CollectError("made-up requests are stored under the smoke split only")
    return benchmark_items(split, benchmark_file)


def noise_items(loaded: Loaded, benchmark_file: Path | None = None) -> list[Item]:
    """The noise sample's requests, in the settings file's order. All must be train items: the reader refuses a test
    item whole, and a dev item is refused here."""
    ids = loaded.noise_item_ids
    rows = read_rows(item_ids=ids) if benchmark_file is None else read_rows(benchmark_file, item_ids=ids)
    if wrong := [row["item_id"] for row in rows if row["split"] != "train"]:
        raise CollectError(f"the noise sample holds non-train items, e.g. {wrong[0]}")
    return [Item(row["item_id"], from_row(row)) for row in rows]


# ---- storage --------------------------------------------------------------------------------------------------


def records_path(router: str, split: str, results_dir: Path = RESULTS_DIR) -> Path:
    if router not in ROUTERS or split not in SPLITS:
        raise CollectError(f"unknown router {router!r} or split {split!r}")
    return results_dir / "jev" / f"{router}_{split}.jsonl"


def read_records(router: str, split: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    """A router's stored records for a split, in the order stored: the one reader of the stored records, which the
    collection, the fit and the table builders share."""
    return read_records_file(records_path(router, split, results_dir), router, split)


def read_records_file(path: Path, router: str, split: str) -> list[dict]:
    """The records of one JSON Lines file (none if it doesn't exist), refused whole unless each is a record of
    exactly `STORED_FIELDS`, is `router`'s and `split`'s, has a pass of 1 or 2 and a refusal flag that is a bool, and
    is the only record of its item and pass (a stored response is never overwritten).

    A last line with no newline is a write that was cut short, and is dropped."""
    if not path.exists():
        return []
    lines = path.read_bytes().decode("utf-8").split("\n")[:-1]  # what follows the last newline is a torn write
    records, seen = [], set()
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except ValueError as error:
            raise CollectError(f"{path.name} line {number} is not JSON") from error
        if not isinstance(record, dict) or tuple(record) != STORED_FIELDS:
            raise CollectError(f"{path.name} line {number} is not a stored record: its fields differ from #30's")
        if record["router"] != router or record["split"] != split:
            raise CollectError(
                f"{path.name} line {number} is a {record['router']} {record['split']} record, not {router} {split}"
            )
        if record["pass"] not in PASSES or type(record["refused"]) is not bool:
            raise CollectError(
                f"{path.name} line {number} has a pass other than 1 or 2, or a refusal that isn't a bool"
            )
        key = (record["item_id"], record["pass"])
        if key in seen:
            raise CollectError(f"{path.name} line {number} repeats {key[0]} pass {key[1]}")
        seen.add(key)
        records.append(record)
    return records


def trim_torn_line(path: Path) -> None:
    """Cut a file back to its last newline, so an append never joins a torn write."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])


def append_record(path: Path, record: dict) -> None:
    """Add one record and sync it to disk before the next call, so a stop loses nothing already collected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---- the collection -------------------------------------------------------------------------------------------


def collect(
    items: Sequence[Item],
    *,
    router: str,
    split: str,
    loaded: Loaded,
    client: jev.JevClient,
    results_dir: Path = RESULTS_DIR,
    budget_bytes: int | None = None,
    pass_number: int = 1,
    now: Callable[[], str] = now_local,
    progress: Callable[[int, int], None] | None = None,
) -> Summary:
    """Send every item not yet stored for this router, split and pass, storing each record as it comes.

    Everything that can be refused is refused before the first call. `budget_bytes` replaces the settings file's for
    this router (null there until the smoke test fills it in). A `RunStop` from the client ends the run and
    names the request; the records stored before it stay valid."""
    if router not in ROUTERS or split not in SPLITS or pass_number not in PASSES:
        raise CollectError(f"unknown router {router!r}, split {split!r} or pass {pass_number!r}")
    if pass_number == 2 and split != "train":
        raise CollectError("the noise pass sends train items only")
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise CollectError("an item ID is repeated in the requests")
    budget = loaded.call.budget_bytes[router] if budget_bytes is None else budget_bytes
    if budget is None or budget < 1:
        raise CollectError(f"no byte budget for {router}: pass one, or fill budget_bytes in the settings file")
    path = records_path(router, split, results_dir)
    trim_torn_line(path)
    existing = read_records(router, split, results_dir)
    done = {(r["item_id"], r["pass"]) for r in existing}
    if pass_number == 2 and (missing := [i for i in ids if (i, 1) not in done]):
        raise CollectError(f"{len(missing)} items have no pass 1 stored for {router}, e.g. {missing[0]}")
    build = jev_call.g_call if router == "jev" else jev_call.g_plus_call
    new = skipped = refused = cut = 0
    for number, item in enumerate(items, start=1):
        if (item.item_id, pass_number) in done:
            skipped += 1
            continue
        call = build(item.request, loaded.call, budget_bytes=budget)
        result = client.call(call.body, call.sha256, f"{router} {split} {item.item_id} (pass {pass_number})")
        record = {"router": router, "split": split, "item_id": item.item_id, "pass": pass_number}
        append_record(path, record | result.to_record() | {"collected_at": now()})
        new += 1
        refused += result.refused
        cut += call.cut_bytes > 0
        if progress is not None:
            progress(number, len(items))
    return Summary(new, skipped, refused, cut)


# ---- stored records out ---------------------------------------------------------------------------------------


def record_scores(
    record: Mapping[str, object], request: Request, loaded: Loaded, weights: jev.GPlusWeights | None = None
) -> jev.GScores:
    """A stored record's router scores and refusal, by the path a live call takes. G+ needs its weights and the
    request (its schema flag is a feature)."""
    if record["router"] == "jev":
        return jev.g_response_scores(record["response"], loaded.call)
    if record["router"] == "jev_bolstered":
        if weights is None:
            raise CollectError("G+'s router scores need its weights")
        return jev.g_plus_response_scores(record["response"], request, loaded.call, weights)
    raise CollectError(f"unknown router {record['router']!r}")


def refused_items(records: Sequence[Mapping[str, object]], loaded: Loaded) -> list[str]:
    """The item IDs whose stored response is a refusal, read from the response itself (A15 item 9). A record of
    pass 2 is listed beside its pass 1."""
    questions = {"jev": jev_call.g_questions, "jev_bolstered": jev_call.g_plus_questions}
    return [
        str(r["item_id"])
        for r in records
        if jev.parse_answers(r["response"], questions[str(r["router"])](loaded.call)).refused
    ]


def input_tokens(record: Mapping[str, object]) -> int:
    """A record's input tokens from its stored usage; refused if the usage lacks them."""
    usage = record.get("usage")
    tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    if type(tokens) is not int or tokens < 0:
        raise CollectError(f"{record.get('item_id')} has no stored input token count")
    return tokens


def overhead_dollars(record: Mapping[str, object], loaded: Loaded) -> float:
    """A call's overhead in dollars: its stored input tokens at the settings' price (A15 item 10)."""
    return jev.overhead_dollars(input_tokens(record), loaded.client)


def measured_overhead_tokens(records: Sequence[Mapping[str, object]]) -> int:
    """The largest input token count among the records (see the module docstring for what it bounds)."""
    if not records:
        raise CollectError("no stored records to measure the overhead from")
    return max(input_tokens(r) for r in records)


def measured_budget_bytes(
    router: str, records: Sequence[Mapping[str, object]], loaded: Loaded, schema: dict | None
) -> int:
    """A router's byte budget from stored usage: the measured overhead, with the questions and (for G+) past requests
    as the fixed parts. `schema` is the longest output schema the router will see."""
    overhead = measured_overhead_tokens(records)
    if router == "jev":
        return jev_call.g_budget_bytes(loaded.call, overhead_tokens=overhead, schema=schema)
    if router == "jev_bolstered":
        return jev_call.g_plus_budget_bytes(loaded.call, overhead_tokens=overhead, schema=schema)
    raise CollectError(f"unknown router {router!r}")


# ---- the command line -----------------------------------------------------------------------------------------


def routers_of(choice: str, budget_bytes: int | None) -> list[str]:
    if choice == "both":
        if budget_bytes is not None:
            raise CollectError("--budget-bytes is for one router: G's and G+'s budgets differ")
        return list(ROUTERS)
    return [choice]


def show_progress(done: int, total: int) -> None:
    if done % 100 == 0 or done == total:
        print(f"  {done}/{total}", flush=True)


def command_collect(args: argparse.Namespace, environ: Mapping[str, str] | None) -> int:
    loaded = load_settings(args.settings)
    routers = routers_of(args.router, args.budget_bytes)
    if args.command == "noise":
        items, split, pass_number = noise_items(loaded, args.benchmark_file), "train", 2
    else:
        items, split, pass_number = requests_for(args.split, args.requests, args.benchmark_file), args.split, 1
    with jev.JevClient(loaded.client, environ) as client:
        for router in routers:
            summary = collect(
                items, router=router, split=split, loaded=loaded, client=client, results_dir=args.results_dir,
                budget_bytes=args.budget_bytes, pass_number=pass_number, progress=show_progress,
            )  # fmt: skip
            print(f"{router} {split} pass {pass_number}: {summary.new} new, {summary.skipped} already stored, "
                  f"{summary.refused} refused, {summary.cut} cut")  # fmt: skip
    return 0


def command_report(args: argparse.Namespace) -> int:
    loaded = load_settings(args.settings)
    records = read_records(args.router, args.split, args.results_dir)
    tokens = [input_tokens(r) for r in records]
    print(f"{args.router} {args.split}: {len(records)} records, {sum(r['pass'] == 2 for r in records)} of pass 2")
    print(f"  refused: {len(refused_items(records, loaded))}")
    if records:
        print(f"  input tokens: {sum(tokens)} in all, largest {max(tokens)}, "
              f"overhead ${sum(overhead_dollars(r, loaded) for r in records):.6f}")  # fmt: skip
        print(f"  attempts beyond the first: {sum(r['attempts'] - 1 for r in records)}, "
              f"median seconds {statistics.median(r['seconds'] for r in records):.3f}")  # fmt: skip
    return 0


def command_budget(args: argparse.Namespace) -> int:
    loaded = load_settings(args.settings)
    schema = json.loads(args.schema_file.read_text(encoding="utf-8")) if args.schema_file else None
    records = read_records(args.router, args.split, args.results_dir)
    overhead = measured_overhead_tokens(records)
    budget = measured_budget_bytes(args.router, records, loaded, schema)
    print(
        f"{args.router}: overhead {overhead} tokens, budget {budget} bytes (from {len(records)} {args.split} records)"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description="Collect Jev responses for G and G+, or report on what is stored.")
    sub = top.add_subparsers(dest="command", required=True)
    for name, text in (("collect", "send the requests of a split"), ("noise", "send the noise sample again as pass 2")):
        one = sub.add_parser(name, help=text)
        one.add_argument("--router", choices=(*ROUTERS, "both"), required=True)
        one.add_argument("--budget-bytes", type=int, help="the byte budget for one router, over the settings file's")
        if name == "collect":
            one.add_argument("--split", choices=SPLITS, required=True)
            one.add_argument("--requests", type=Path, help="a JSON list of made-up requests (smoke only)")
    for name, text in (("report", "count what is stored"), ("budget", "measure the byte budget from stored usage")):
        one = sub.add_parser(name, help=text)
        one.add_argument("--router", choices=ROUTERS, required=True)
        one.add_argument("--split", choices=SPLITS, required=True)
        if name == "budget":
            one.add_argument("--schema-file", type=Path, help="JSON of the longest output schema the router will see")
    for one in sub.choices.values():
        one.add_argument("--settings", type=Path, default=jev_call.SETTINGS_FILE)
        one.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
        one.add_argument("--benchmark-file", type=Path, help=argparse.SUPPRESS)  # a check's synthetic benchmark
    return top


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in ("collect", "noise"):
            return command_collect(args, environ)
        return command_report(args) if args.command == "report" else command_budget(args)
    except jev.RunStop as stop:
        print(f"stopped: {stop}\nwhat was stored before it is valid; rerun to resume", file=sys.stderr)
        return 1
    except (CollectError, BenchmarkError, jev_call.CallError, jev.ClientSettingsError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
