"""Stage 7's train work, measured from the stored records and written as A17 item 28 (PLAN_v4.md §9; Amendment A17,
items 12, 16, 17, 20.3 to 20.6 and 28; issue #69).

Run from the project root:

    uv run python -m analysis.gate_train_report                 # print item 28's text
    uv run python -m analysis.gate_train_report --write         # replace item 28 in PLAN_v4.md, and write the cutoffs file
    uv run python -m analysis.gate_train_report --noise-files DIR   # write the two noise lists as JSON for the collectors

It reads train's stored gate records (`results/gates/*_train.jsonl`, the batch log), train's requests and outcomes, and
the committed settings, and calls no gate. It reads no dev answer, outcome or verdict. `analysis/check_gates_freeze.py`
compares the written item with the same sources.

The cutoffs are also written to `routers/gate_cutoffs.json`, the file Stage 7's dev work and Stage 8 read them from; item
28 records its hash and the cutoffs themselves.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from analysis import collect_sonnet_gate, gate_train, jev_train_report, noise_samples
from analysis.benchmark import read_rows
from analysis.bootstrap import resample
from analysis.check_gates_settings_commit import COMMIT_FILES, SMOKE_RECORD
from analysis.check_jev_settings_commit import settings_commit
from analysis.checking import git
from analysis.collect_gates import read_free_records
from analysis.collect_jev import RESULTS_DIR
from analysis.collect_jev_gate import paid_scores as jev_paid_scores
from analysis.collect_jev_gate import read_records as read_jev_records
from analysis.collect_sonnet_gate import paid_scores as sonnet_paid_scores
from analysis.gate_answers import read_answers
from analysis.gate_cutoff import fixed_cutoff
from analysis.gate_metrics import signal_check
from analysis.gate_table import PAID_GATES, GateScoreTable, build_table
from analysis.matrix import Matrix, load_train
from analysis.plan import PLAN
from analysis.router_names import JEV_GATE, SONNET_GATE
from harness.client import PROJECT_ROOT, load_models
from routers import gates, jev, jev_gate, sonnet_gate
from routers.ladder import SCORED

CUTOFFS_FILE = "routers/gate_cutoffs.json"
BATCH_LOG = "sonnet_batches.json"
STORED = {  # the train records whose hashes item 28 holds, under results/
    "results/gates/free_checks_train.jsonl": "free_checks",
    "results/gates/jev_gate_train.jsonl": "jev_gate",
    "results/gates/sonnet_gate_train.jsonl": "sonnet_gate",
    "results/gates/sonnet_gate_latency.jsonl": "latency",
}
SIGNALS = (JEV_GATE, SONNET_GATE, gates.CODE)


@dataclass(frozen=True)
class Train:
    """Everything item 28 holds that comes from the stored train records and the train matrix."""

    free: list[dict]
    jev: list[dict]
    batch: list[dict]
    latency: list[dict]
    log: dict
    task_of: dict[str, str]
    lists: dict[str, list[tuple[str, str]]]
    counts: dict[str, gate_train.ConfigCounts]
    cutoffs: dict[str, float]
    reports: dict[str, gate_train.CutoffReport]
    signals: list
    noise: dict[str, gate_train.Difference]
    drift: dict[str, gate_train.Difference]
    sonnet_latency: gate_train.SampleLatency
    jev_latency: jev_train_report.Latency
    spend: gate_train.TrainSpend
    table: GateScoreTable
    matrix: Matrix


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(results_dir: Path = RESULTS_DIR) -> Train:
    """Read train's stored records and rebuild every measure item 28 holds. Refuses (the readers' errors) a store that
    is missing, torn or inconsistent, and a train table that does not rebuild."""
    free = read_free_records("train", results_dir)
    jev_records = read_jev_records("train", results_dir)
    batch = collect_sonnet_gate.read_sonnet_records("train", results_dir)
    latency = collect_sonnet_gate.read_latency_records("train", results_dir)
    log = collect_sonnet_gate.load_log(collect_sonnet_gate.batch_log_path(results_dir))
    rows = read_rows(split="train")
    task_of = {str(r["item_id"]): str(r["source"]) for r in rows}
    jev_settings = jev_gate.read_settings()
    sonnet_settings = sonnet_gate.load_settings()
    answers = {(a.item_id, a.config) for a in read_answers("train")}
    table = build_table(
        free, answers,
        paid={
            JEV_GATE: jev_paid_scores("train", jev_settings, jev.read_client_settings(), results_dir),
            SONNET_GATE: sonnet_paid_scores("train", sonnet_settings, results_dir),
        },
    )  # fmt: skip
    matrix = load_train()
    cutoffs = {gate: fixed_cutoff(table, matrix, gate) for gate in PAID_GATES}
    reports = {gate: gate_train.cutoff_report(train_answers(table, matrix, gate), cutoffs[gate]) for gate in PAID_GATES}
    noise_settings = noise_samples.read_noise_settings()
    models = load_models()
    client = jev.read_client_settings()
    by_pass_one = [r for r in jev_records if r["pass"] == 1]
    return Train(
        free, jev_records, batch, latency, log, task_of,
        noise_samples.noise_lists(free, noise_settings),
        gate_train.count_configs(free, jev_records, batch, SCORED),
        cutoffs, reports,
        signal_check(resample(matrix), matrix, table, free, SCORED),
        gate_train.jev_noise([r for r in jev_records if r["item_id"] in noise_ids(free, noise_settings)], jev_settings),
        gate_train.sonnet_drift(batch, latency),
        gate_train.sonnet_latency(latency, task_of),
        jev_train_report.latency(by_pass_one, task_of),
        gate_train.train_spend(
            jev=jev_records, batch=batch, latency=latency, jev_price=client.price_per_million_input_tokens,
            input_price=sonnet_settings.input_per_million, output_price=sonnet_settings.output_per_million,
            batch_discount=float(models["batch_discount"]),
        ),
        table, matrix,
    )  # fmt: skip


def noise_ids(free: Sequence[dict], settings: noise_samples.NoiseSettings) -> set[str]:
    return {item for item, _ in noise_samples.noise_lists(free, settings)["jev"]}


def train_answers(table: GateScoreTable, matrix: Matrix, gate: str) -> list[tuple[float | None, bool]]:
    """`(gate score, answer was right)` for haiku's, sonnet's and opus_low's train answers that passed the free checks:
    the answers item 12's Youden's J is taken over."""
    return [
        (row.paid(gate).score, matrix.cells[(row.item_id, row.config)].success == 1)  # type: ignore[union-attr]
        for row in table.rows
        if row.config in SCORED and row.free_passed
    ]


# ---- the text -----------------------------------------------------------------------------------------------


def f4(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f}"


def lists_lines(lists: dict[str, list[tuple[str, str]]]) -> list[str]:
    jev_ids = ", ".join(f"`{i}`" for i, _ in lists["jev"])
    sonnet_ids = ", ".join(f"`{i}`" for i, _ in lists["sonnet"])
    return [
        f"The Jev gate's noise list (item 17), {len(lists['jev'])} `haiku` answers, 25 per task: {jev_ids}.",
        f"The Sonnet gate's latency sample (item 16), {len(lists['sonnet'])} `haiku` answers, 20 per task: {sonnet_ids}.",
    ]


def count_lines(counts: dict[str, gate_train.ConfigCounts]) -> list[str]:
    lines = []
    for config, c in counts.items():
        first = ", ".join(f"{name} {c.failed_first[name]}" for name in gate_train.CHECKS)
        failing = ", ".join(f"{name} {c.failing[name]}" for name in gate_train.CHECKS)
        lines.append(
            f"{config}: {c.answers} answers checked; {c.rejected} free-check rejections (by the check that failed first: "
            f"{first}; answers failing each check: {failing}); Jev gate: {c.jev_calls} calls, {c.jev_refused} refused, "
            f"{c.jev_cut} cut, {c.jev_retries} retries ({c.jev_failed_attempts} failed attempts); Sonnet gate: "
            f"{c.sonnet_calls} verdicts, {c.sonnet_refused} refused, {c.sonnet_cap_stops} cap stops."
        )
    return lines


def cutoff_lines(train: Train) -> list[str]:
    lines = []
    for gate, report in train.reports.items():
        lines.append(
            f"{gate}: cutoff {train.cutoffs[gate]!r}, J {report.j:.4f}; {report.wrong_rejected} of {report.wrong} wrong "
            f"answers rejected ({report.wrong_rejected / report.wrong:.4f}) and {report.right_rejected} of {report.right} "
            f"right answers rejected ({report.right_rejected / report.right:.4f})."
        )
    return lines


def signal_lines(signals: list) -> list[str]:
    lines = []
    for s in signals:
        pooled = s.reading.pooled
        tasks = ", ".join(f"{task} {f4(est.value)}" for task, est in s.reading.per_task.items())
        lines.append(
            f"{s.gate} {s.config}: pooled {f4(pooled.value)} (95% interval {f4(pooled.low)} to {f4(pooled.high)}), "
            f"above 0.5: {'yes' if s.above_half else 'no'}, lower end above 0.5: {'yes' if s.interval_above_half else 'no'}; "
            f"per task {tasks}."
        )
    return lines


def noise_lines(train: Train) -> list[str]:
    def one(name: str, by_config: dict[str, gate_train.Difference]) -> str:
        parts = ", ".join(f"{c} D6 {d.d6:.4f} shift {d.shift:+.4f} over {d.pairs} answers ({d.unscored} left out)"
                          for c, d in by_config.items())  # fmt: skip
        return f"{name}: {parts}"

    return [
        one("Jev gate, pass 2 less pass 1", train.noise),
        one("Sonnet gate, standard API less batch (D6 floored at 0.01)", train.drift)
        + "; unfloored mean absolute difference "
        + ", ".join(f"{c} {d.mean_abs:.4f}" for c, d in train.drift.items()),
    ]


def latency_lines(train: Train, conditions: str) -> list[str]:
    s, j = train.sonnet_latency, train.jev_latency
    by_task = "; ".join(f"{task} {m:.2f}, {p:.2f}, {n}" for task, (m, p, n) in s.by_task.items())
    jev_tasks = "; ".join(f"{task} {ms(m)}, {ms(p)}, {n}" for task, (m, p, n) in j.by_task.items())
    reconnects = ", ".join(ms(x) for x in j.reconnects) or "none"
    stamps = sorted(r["collected_at"] for r in train.latency)
    jev_stamps = sorted(r["collected_at"] for r in train.jev)
    code = [float(r["code_duration_s"]) for r in train.free if r["code_applied"]]
    return [
        f"The Sonnet gate, {len(train.latency)} standard-API calls one at a time, the first left out: median "
        f"{s.median:.2f} s and p90 {s.p90:.2f} s over {s.count} calls; the first call {s.first:.2f} s. Per task (median, p90 "
        f"in s, calls): {by_task}.",
        f"The Jev gate, pass 1, one at a time, the first call and every reconnect left out: median {ms(j.median)} ms and "
        f"p90 {ms(j.p90)} ms over {j.count} calls; the first call {ms(j.first)} ms; reconnects (ms): {reconnects}. Per "
        f"task (median, p90 in ms, calls): {jev_tasks}.",
        f"The code check, timed locally, over {len(code)} answers it applied to: median {sorted(code)[len(code) // 2]:.2f} s "
        "(the sandbox's own duration).",
        f"Collected: the Sonnet latency calls from {stamps[0]} to {stamps[-1]}; the Jev calls (both passes) from "
        f"{jev_stamps[0]} to {jev_stamps[-1]}.",
        conditions,
    ]


def spend_lines(train: Train) -> list[str]:
    s = train.spend
    return [
        f"The Jev gate, both passes: {s.jev_input_tokens} input tokens, ${s.jev:.4f} at $0.042 per million.",
        f"The Sonnet gate's train batch: {s.batch_input_tokens} input and {s.batch_output_tokens} output tokens, "
        f"${s.batch_list:.4f} at list price, ${s.batch:.4f} as billed at the batch discount.",
        f"The Sonnet latency calls: {s.latency_input_tokens} input and {s.latency_output_tokens} output tokens, "
        f"${s.latency:.4f} at list price.",
        f"Train's gate calls in all, as billed: ${s.total:.4f}.",
    ]


def first_dates(train: Train) -> tuple[str, str, str]:
    """The first train Jev call's server Date, the train batch's created_at, and the Sonnet batch's id(s)."""
    first = jev_train_report.first_date([r for r in train.jev if r["pass"] == 1])
    batches = [b for b in train.log["batches"] if b["split"] == "train"]
    created = min(str(b["created_at"]) for b in batches)
    return first.strftime("%a, %d %b %Y %H:%M:%S GMT"), created, ", ".join(f"`{b['id']}`" for b in batches)


def render(train: Train, conditions: str, root: Path = PROJECT_ROOT) -> str:
    """Item 28's whole text, from the stored records and the committed files. `conditions` is the line that records the
    machine and network, `Conditions: ...`: it depends on where the collection ran, so it is quoted as measured."""
    commit = settings_commit(root, SMOKE_RECORD)
    if commit is None:
        raise gate_train.GateTrainError("the settings commit is not in this history")
    when = git("show", "-s", "--format=%cI", commit).stdout.strip()
    settings = json.loads((root / "routers/gate_settings.json").read_text(encoding="utf-8"))
    client = json.loads((root / "routers/jev_settings.json").read_text(encoding="utf-8"))["jev"]
    sonnet = settings["sonnet"]
    hashed = "\n".join(f"      - `{path}`: `{file_sha256(root / path)}`" for path in sorted(COMMIT_FILES))
    stored = "\n".join(f"      - `{path}`: `{file_sha256(root / path)}`" for path in STORED)
    date, created, batch_ids = first_dates(train)

    def bullets(lines: Sequence[str], indent: int = 6) -> str:
        return "\n".join(f"{' ' * indent}- {line}" for line in lines)

    return f"""28. **The freeze record** (item 20.4), written in the freeze commit, before any dev gate verdict. Nothing here reads a dev outcome or sends a dev answer to a gate. `analysis/check_gates_freeze.py` compares each entry with its source, and each hash with its file. sha256, as in A7 item 3. The measures come from `python -m analysis.gate_train_report`:
    - **The settings commit**: `{commit}`, committed {when}. Every file it froze is as that commit left it.
    - **Frozen files**, every file the settings commit froze (item 20.2), the sandbox Dockerfile among them:
{hashed}
    - **The sandbox image**: `{settings["code_check"]["image"]}`, digest `{settings["code_check"]["image_digest"]}`.
    - **The gates' identities**: the Jev gate calls `{client["model"]}` at `{client["endpoint"]}`, $0.042 per million input tokens, checked {client["price_checked"]}; the Sonnet gate calls `{sonnet["model"]}` at {sonnet["effort"]} effort with `max_tokens` {sonnet["max_tokens"]}, billed at {sonnet["list_prices"]["price_snapshot"]}'s list prices of ${sonnet["list_prices"]["input_per_million"]:g} per million input tokens and ${sonnet["list_prices"]["output_per_million"]:g} per million output tokens.
    - **The two noise lists**, rebuilt by the settings commit's drawing rule from train's free-check records:
{bullets(lists_lines(train.lists), 6)}
    - **Counts per config** (items 7 and 14), train, run 0. Retries are the attempts beyond each Jev call's first; the Sonnet batches resent {gate_train.batch_retries(train.log, "train")} requests (errored or expired); a Jev truncation is a call the byte budget cut, a Sonnet one a cap stop:
{bullets(count_lines(train.counts), 6)}
    - **The two fixed cutoffs** (item 12), Youden's J over `haiku`'s, `sonnet`'s and `opus_low`'s train answers that passed the free checks, ties to the stricter cutoff. The cutoffs are in the file `{CUTOFFS_FILE}`: `{cutoffs_text_sha256(train)}`, and are:
{bullets(cutoff_lines(train), 6)}
    - **The train signal check** (item 21), within-task concordance per config of each signal against 0.5, pooled over tasks, with the percentile 95% interval over 2,000 paired resamples of train:
{bullets(signal_lines(train.signals), 6)}
    - **D6 and the shift** (item 17), on the gate score as a probability:
{bullets(noise_lines(train), 6)}
    - **Train latency** (item 16), the successful call's time alone, as measured; it depends on the machine and the network, so it isn't re-checked:
{bullets(latency_lines(train, conditions), 6)}
    - **Tokens and cost of the train calls:**
{bullets(spend_lines(train), 6)}
    - **The stored train records**, kept under `results/gates/`, not committed:
{stored}
    - **The order** (item 20.6): the first train Jev call's server Date `{date}`; the train batch's `created_at` {created} (batch {batch_ids}). Both come after the settings commit's commit time above.
"""


# ---- the command line -------------------------------------------------------------------------------------------


def cutoffs_text(train: Train) -> str:
    return json.dumps(train.cutoffs, indent=2) + "\n"


def cutoffs_text_sha256(train: Train) -> str:
    return hashlib.sha256(cutoffs_text(train).encode()).hexdigest()


def write_cutoffs(train: Train, root: Path = PROJECT_ROOT) -> None:
    (root / CUTOFFS_FILE).write_text(cutoffs_text(train), encoding="utf-8")


def replace_item(plan: str, text: str) -> str:
    """The plan with A17 item 28 replaced by `text`."""
    start = plan.index("\n28. **The freeze record**") + 1
    end = plan.index("\n29. **Record.**", start) + 1
    return plan[:start] + text + plan[end:]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--write", action="store_true", help="write the cutoffs file and replace item 28 in the plan")
    parser.add_argument("--noise-files", type=Path, help="write the two noise lists as JSON files in this folder")
    args = parser.parse_args(argv)
    if args.noise_files:
        lists = noise_samples.noise_lists(read_free_records("train"), noise_samples.read_noise_settings())
        for name, pairs in lists.items():
            (args.noise_files / f"noise_{name}.json").write_text(json.dumps(pairs), encoding="utf-8")
        return 0
    train = load()
    if args.write:
        write_cutoffs(train)
    conditions = f"Conditions: {jev_train_report.machine()}; {jev_train_report.network()}."
    text = render(train, conditions)
    if args.write:
        PLAN.write_text(replace_item(PLAN.read_text(encoding="utf-8"), text), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
