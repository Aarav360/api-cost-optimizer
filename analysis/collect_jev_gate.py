"""Collect the Jev gate's responses and turn them into the gate-score table's Jev column (PLAN_v4.md §9; Amendment A17,
items 5, 7, 13, 14 and 15; A15 item 9).

Run from the project root (the key in `TYPESAFE_API_KEY`):

    uv run python -m analysis.collect_jev_gate collect --split train|dev
    uv run python -m analysis.collect_jev_gate collect --split smoke --smoke-file made_up.json --budget-bytes 20000
    uv run python -m analysis.collect_jev_gate noise --pairs noise_list.json
    uv run python -m analysis.collect_jev_gate report --split train

Answers come in through `analysis.collect_gates` (the gates' answer reader and the benchmark reader; smoke answers
from a made-up file), and only those whose free checks passed are called: the free-check file is the filter, and an
answer that failed one gets no call (A17 item 7). A split's free checks must be stored first. Each response is stored,
whole, in `results/gates/jev_gate_{split}.jsonl` through the same append, sync and torn-line helpers as the free checks,
one record per call, keyed by item, config, run 0, split, gate and pass, then A15 item 9's fields and the bytes the cut
dropped from the answer and from the prompt. The key is read from the environment by the frozen client and is in no
record. A rerun never calls a stored answer again. The noise pass (A17 item 17) sends a given list again as pass 2, for
train answers whose pass 1 is stored; pass-2 records never enter the table. A run stop (a 4xx the client doesn't retry,
the retry cap) ends the run naming the answer, exit 1; what was stored before it is valid and a rerun resumes.

The byte budget stays empty in `routers/gate_settings.json` until the smoke test measures it (A17 item 20), so a run
passes it with `--budget-bytes` (or `budget_bytes=` to `collect`).

`paid_scores` is the stored records' way into the table: each response is read by the gate's own parser, the score and
refusal from the response, the overhead from the stored input tokens at Jev's price (A15 item 10). Pass it to
`analysis.gate_table.build_table` as `paid={JEV_GATE: ...}`. A pure cascade's walk then reads only the table.
"""

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from analysis.benchmark import BenchmarkError
from analysis.collect_gates import (
    GATE_KEYS,
    SPLITS,
    GateCollectError,
    GateInput,
    gate_inputs,
    gate_path,
    read_free_records,
    read_gate_records,
    read_smoke_inputs,
)
from analysis.collect_jev import RESULTS_DIR, append_record, now_local, trim_torn_line
from analysis.gate_table import PaidScore
from analysis.router_names import JEV_GATE
from routers import jev, jev_gate

PASSES = (1, 2)  # 2 is the noise pass
STORED_FIELDS = (
    *GATE_KEYS, "body_sha256", "response", "model", "usage", "request_id", "server_date", "seconds", "attempts",
    "failed_attempts", "new_connection", "refused", "answer_cut_bytes", "prompt_cut_bytes", "collected_at",
)  # fmt: skip  # A17 item 14: the key, A15 item 9's fields, and the bytes the cut dropped


@dataclass(frozen=True)
class Summary:
    new: int  # calls made and stored this run
    skipped: int  # answers already stored for this pass, never called again
    not_called: int  # answers that failed a free check and get no call
    refused: int  # of the new calls, those Jev refused
    cut: int  # of the new calls, those the byte budget cut


def jev_path(split: str, results_dir: Path = RESULTS_DIR) -> Path:
    return gate_path("jev_gate", split, results_dir)


def read_records(split: str, results_dir: Path = RESULTS_DIR) -> list[dict]:
    """A split's stored Jev gate records (passes 1 and 2), in the order stored: the one reader of them."""
    records = read_gate_records(jev_path(split, results_dir), STORED_FIELDS, split, JEV_GATE, passes=PASSES)
    if any(type(r["refused"]) is not bool for r in records):
        raise GateCollectError("a stored Jev gate record has a refusal flag that isn't a bool")
    return records


def collect(
    inputs: Sequence[GateInput],
    *,
    split: str,
    settings: jev_gate.JevGateSettings,
    client: jev.JevClient,
    results_dir: Path = RESULTS_DIR,
    budget_bytes: int | None = None,
    pass_number: int = 1,
    now: Callable[[], str] = now_local,
    progress: Callable[[int, int], None] | None = None,
) -> Summary:
    """Call the Jev gate on every input whose free checks passed and that is not yet stored for this pass, storing each
    record as it comes. Everything that can be refused (GateCollectError, GateCallError) is refused before the first call.

    `budget_bytes` replaces the settings' (null there until the smoke test). A `RunStop` from the client ends the run
    and names the answer; the records stored before it stay valid."""
    if split not in SPLITS or pass_number not in PASSES:
        raise GateCollectError(f"unknown split {split!r} or pass {pass_number!r}")
    if pass_number == 2 and split != "train":
        raise GateCollectError("the noise pass sends train answers only")
    keys = [(i.item_id, i.config) for i in inputs]
    if len(set(keys)) != len(keys):
        raise GateCollectError("the same answer is in the list more than once")
    budget = settings.budget_bytes if budget_bytes is None else budget_bytes
    if budget is None or budget < 1:
        raise GateCollectError("no byte budget: pass one, or fill budget_bytes in the settings file (smoke test)")
    free = {(r["item_id"], r["config"]): bool(r["passed"]) for r in read_free_records(split, results_dir)}
    if unchecked := [k for k in keys if k not in free]:
        raise GateCollectError(f"{len(unchecked)} answers have no free-check record, e.g. {unchecked[0]}")
    path = jev_path(split, results_dir)
    trim_torn_line(path)
    done = {(r["item_id"], r["config"], r["pass"]) for r in read_records(split, results_dir)}
    callable_inputs = [i for i in inputs if free[(i.item_id, i.config)]]
    if pass_number == 2 and (missing := [i for i in callable_inputs if (i.item_id, i.config, 1) not in done]):
        raise GateCollectError(
            f"{len(missing)} answers have no pass 1 stored, e.g. {missing[0].item_id} {missing[0].config}"
        )
    new = skipped = refused = cut = 0
    for number, item in enumerate(callable_inputs, start=1):
        if (item.item_id, item.config, pass_number) in done:
            skipped += 1
            continue
        call = jev_gate.gate_call(item.request, item.answer, settings, budget_bytes=budget)
        label = f"Jev gate {split} {item.item_id} {item.config} (pass {pass_number})"
        result = client.call(call.body, call.sha256, label)
        key = {"item_id": item.item_id, "config": item.config, "run": 0, "split": split, "gate": JEV_GATE,
               "pass": pass_number}  # fmt: skip
        extra = {"answer_cut_bytes": call.answer_cut_bytes, "prompt_cut_bytes": call.prompt_cut_bytes}
        append_record(path, key | result.to_record() | extra | {"collected_at": now()})
        new += 1
        refused += result.refused
        cut += call.answer_cut_bytes > 0 or call.prompt_cut_bytes > 0
        if progress is not None:
            progress(number, len(callable_inputs))
    return Summary(new, skipped, len(inputs) - len(callable_inputs), refused, cut)


# ---- stored records out: the table's Jev column ----------------------------------------------------------------


def input_tokens(record: Mapping[str, object]) -> int:
    usage = record.get("usage")
    tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    if type(tokens) is not int or tokens < 0:
        raise GateCollectError(f"{record.get('item_id')} {record.get('config')} has no stored input token count")
    return tokens


def paid_score(
    record: Mapping[str, object], settings: jev_gate.JevGateSettings, client: jev.ClientSettings
) -> PaidScore:
    """One stored record as a table entry, through the gate's own parser. The stored refusal flag must agree with the
    response (refused whole otherwise); a refused call has no score and is charged its input tokens."""
    read = jev_gate.read_response(record["response"], settings)
    if read.refused != record["refused"]:
        raise GateCollectError(
            f"{record['item_id']} {record['config']}: the stored refusal flag disagrees with the response"
        )
    return PaidScore(read.score, jev.overhead_dollars(input_tokens(record), client), read.refused)


def paid_scores(
    split: str,
    settings: jev_gate.JevGateSettings,
    client: jev.ClientSettings,
    results_dir: Path = RESULTS_DIR,
) -> dict[tuple[str, str], PaidScore]:
    """The Jev column of a split: pass 1's records only (the noise pass never enters), keyed by (item, config). Refused
    whole unless every answer whose free checks passed has its call, and no other answer has one."""
    passed = {(r["item_id"], r["config"]) for r in read_free_records(split, results_dir) if r["passed"]}
    scores = {(r["item_id"], r["config"]): paid_score(r, settings, client)
              for r in read_records(split, results_dir) if r["pass"] == 1}  # fmt: skip
    if missing := sorted(passed - set(scores)):
        raise GateCollectError(
            f"{len(missing)} answers that passed the free checks have no Jev call, e.g. {missing[0]}"
        )
    if stray := sorted(set(scores) - passed):
        raise GateCollectError(
            f"{len(stray)} Jev calls are on answers that did not pass the free checks, e.g. {stray[0]}"
        )
    return scores


# ---- the command line -------------------------------------------------------------------------------------------


def show_progress(done: int, total: int) -> None:
    if done % 100 == 0 or done == total:
        print(f"  {done}/{total}", flush=True)


def read_pairs(path: Path) -> list[tuple[str, str]]:
    """A noise list: a JSON list of [item_id, config] pairs."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, list)
        or not data
        or not all(isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) for x in p) for p in data)
    ):
        raise GateCollectError(f"{path} must be a non-empty JSON list of [item_id, config] pairs")
    return [(p[0], p[1]) for p in data]


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description="Collect the Jev gate's responses, or report on what is stored.")
    sub = top.add_subparsers(dest="command", required=True)
    one = sub.add_parser("collect", help="call the gate on a split's answers that passed the free checks")
    one.add_argument("--split", choices=SPLITS, required=True)
    one.add_argument("--smoke-file", type=Path, help="a JSON list of made-up answers (smoke only)")
    noise = sub.add_parser("noise", help="send a list of train answers again as pass 2")
    noise.add_argument("--pairs", type=Path, required=True, help="a JSON list of [item_id, config] pairs")
    report = sub.add_parser("report", help="count what is stored")
    report.add_argument("--split", choices=SPLITS, required=True)
    for each in (one, noise):
        each.add_argument("--budget-bytes", type=int, help="the byte budget, over the settings file's")
    for each in sub.choices.values():
        each.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
        each.add_argument("--gate-settings", type=Path, default=jev_gate.SETTINGS_FILE)
        each.add_argument("--client-settings", type=Path, default=jev.SETTINGS_FILE)
        each.add_argument("--benchmark-file", type=Path, help=argparse.SUPPRESS)  # a check's synthetic benchmark
    return top


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = jev_gate.read_settings(args.gate_settings)
        client_settings = jev.read_client_settings(args.client_settings)
        if args.command == "report":
            records = read_records(args.split, args.results_dir)
            print(f"{args.split}: {len(records)} records, {sum(r['pass'] == 2 for r in records)} of pass 2, "
                  f"{sum(r['refused'] for r in records)} refused")  # fmt: skip
            return 0
        if args.command == "noise":
            split, pass_number = "train", 2
            wanted = set(read_pairs(args.pairs))
            inputs = [i for i in gate_inputs(split, args.benchmark_file) if (i.item_id, i.config) in wanted]
            if len(inputs) != len(wanted):
                raise GateCollectError("the noise list names answers that are not train's")
        else:
            split, pass_number = args.split, 1
            if (split == "smoke") != (args.smoke_file is not None):
                raise GateCollectError("--smoke-file goes with --split smoke, and only with it")
            inputs = read_smoke_inputs(args.smoke_file) if split == "smoke" else gate_inputs(split, args.benchmark_file)
        with jev.JevClient(client_settings, environ) as client:
            summary = collect(inputs, split=split, settings=settings, client=client, results_dir=args.results_dir,
                              budget_bytes=args.budget_bytes, pass_number=pass_number, progress=show_progress)  # fmt: skip
        print(f"{split} pass {pass_number}: {summary.new} new, {summary.skipped} already stored, {summary.not_called} "
              f"failed a free check, {summary.refused} refused, {summary.cut} cut")  # fmt: skip
        return 0
    except jev.RunStop as stop:
        print(f"stopped: {stop}\nwhat was stored before it is valid; rerun to resume", file=sys.stderr)
        return 1
    except (GateCollectError, BenchmarkError, jev_gate.GateCallError, jev.ClientSettingsError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
