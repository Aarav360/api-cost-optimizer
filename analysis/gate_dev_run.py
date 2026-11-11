"""Stage 7's dev run, read back from the stored records, and its committed record (PLAN_v4.md §9; Amendment A17, items 7,
14, 16 and 20.5 to 20.6; issue #70).

Run from the project root:

    uv run python -m analysis.gate_dev_run measures            # print what the dev records hold
    uv run python -m analysis.gate_dev_run record --write      # write analysis/gates_dev_record.json
    uv run python -m analysis.gate_dev_run walks               # print the fifteen walks' operating points
    uv run python -m analysis.gate_dev_run walks --write       # ... and write the fifteen sections (A17 item 20.5, step 6)

It reads dev's stored gate records (`results/gates/*_dev.jsonl`, the batch log, `dev_conditions.json`), dev's requests and
outcomes and the committed settings, and calls no gate. `analysis/check_gates_dev.py` rebuilds all of it.

`dev_conditions.json` is written by `python -m analysis.gate_dev_run conditions` just before the dev Jev collection starts,
so the machine and network are the ones the calls ran on (A16 item 3 found that dev's conditions in Stage 6 were not
recorded). The committed record copies it.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis import collect_sonnet_gate, gate_dev, gate_train, jev_train_report
from analysis.benchmark import read_rows
from analysis.bootstrap import resample
from analysis.collect_gates import read_free_records
from analysis.collect_jev import RESULTS_DIR, now_local
from analysis.collect_jev_gate import paid_scores as jev_paid_scores
from analysis.collect_jev_gate import read_records as read_jev_records
from analysis.collect_sonnet_gate import paid_scores as sonnet_paid_scores
from analysis.gate_answers import read_answers
from analysis.gate_table import GateScoreTable, build_table
from analysis.matrix import Matrix
from analysis.matrix import load as load_matrix
from analysis.operating_points import OPERATING_POINTS_FILE
from analysis.router_names import G_PLUS, JEV_GATE, NAMED_ROUTERS, SONNET_GATE, E, F, G
from analysis.router_table import RouterScoreTable
from harness.client import PROJECT_ROOT, load_models
from routers import jev, jev_gate, sonnet_gate
from routers.ladder import SCORED

RECORD_FILE = PROJECT_ROOT / "analysis" / "gates_dev_record.json"
CUTOFFS_FILE = PROJECT_ROOT / "routers" / "gate_cutoffs.json"
CONDITIONS_NAME = "dev_conditions.json"
STORED = (
    "free_checks_dev.jsonl",
    "jev_gate_dev.jsonl",
    "sonnet_gate_dev.jsonl",
)  # the dev records whose hashes the record holds
CONFIGS = (*SCORED, "opus")  # dev has all four configs' answers (train has three)


class DevRunError(ValueError):
    """Dev records that don't make a whole run: a gate with a call missing or on an answer it must not judge."""


def conditions_path(results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / "gates" / CONDITIONS_NAME


def capture_conditions(results_dir: Path = RESULTS_DIR) -> dict[str, str]:
    """The machine and network the dev gate calls run on, stamped, written to `dev_conditions.json` (once: a file that is
    there is left alone, since the conditions are those of the first call)."""
    path = conditions_path(results_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    found = {"captured_at": now_local(), "machine": jev_train_report.machine(), "network": jev_train_report.network()}
    path.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
    return found


def measures(
    free: Sequence[Mapping[str, Any]],
    jev_records: Sequence[Mapping[str, Any]],
    batch: Sequence[Mapping[str, Any]],
    log: Mapping[str, Any],
    task_of: Mapping[str, str],
    configs: Sequence[str],
    *,
    jev_price: float,
    input_price: float,
    output_price: float,
    batch_discount: float,
) -> dict[str, Any]:
    """Everything the record holds that the stored dev records give: counts per config (the free checks' rejections, both
    gates' calls, refusals and truncations), the Jev gate's latency with the first call and the reconnects set apart, the
    code check's median time, the first Jev call's server Date, the dev batches' creation times, and the spend. Refuses
    (ValueError) records with no Jev call, or a config with no free-check record."""
    counts = gate_train.count_configs(free, jev_records, batch, configs)
    pass_one = [r for r in jev_records if r["pass"] == 1]
    timing = jev_train_report.latency(pass_one, task_of)
    first = jev_train_report.first_date(pass_one)
    code = [float(r["code_duration_s"]) for r in free if r["code_applied"]]
    spend = gate_train.train_spend(
        jev=jev_records, batch=batch, latency=[], jev_price=jev_price, input_price=input_price,
        output_price=output_price, batch_discount=batch_discount,
    )  # fmt: skip
    return {
        "counts": {config: asdict(c) for config, c in counts.items()},
        "jev_latency": {
            "median_s": timing.median, "p90_s": timing.p90, "count": timing.count, "first_s": timing.first,
            "reconnects_s": timing.reconnects,
            "by_task": {t: {"median_s": m, "p90_s": p, "count": n} for t, (m, p, n) in timing.by_task.items()},
        },
        "code_checks": len(code),
        "code_check_median_s": float(np.median(code)) if code else None,
        "first_jev_date": str(pass_one[0]["server_date"]),
        "first_jev_date_parsed": first.isoformat(),
        "jev_collected": list(jev_train_report.collected_span(jev_records)),
        "batches": [{"id": b["id"], "created_at": b["created_at"]} for b in log["batches"] if b["split"] == "dev"],
        "batch_resent": gate_train.batch_retries(log, "dev"),
        "spend": {
            "jev_input_tokens": spend.jev_input_tokens, "jev": spend.jev,
            "batch_input_tokens": spend.batch_input_tokens, "batch_output_tokens": spend.batch_output_tokens,
            "batch_list": spend.batch_list, "batch": spend.batch, "total": spend.jev + spend.batch,
        },
    }  # fmt: skip


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stored_hashes(results_dir: Path = RESULTS_DIR) -> dict[str, str]:
    return {name: file_sha256(results_dir / "gates" / name) for name in STORED}


def read_cutoffs(path: Path = CUTOFFS_FILE) -> dict[str, float]:
    return {str(k): float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


# ---- the dev records in, the table out -------------------------------------------------------------------------


def dev_table(results_dir: Path = RESULTS_DIR) -> GateScoreTable:
    """Dev's gate-score table from the stored records, through each gate's own parser (A17 item 13). Refuses
    (GateTableError, GateCollectError) a table that misses or repeats an answer, a Jev call missing from a passing answer
    or on a failed one, and (DevRunError) a Sonnet verdict missing from a passing haiku, sonnet or opus_low answer or
    present on an opus one."""
    free = read_free_records("dev", results_dir)
    sonnet = sonnet_paid_scores("dev", sonnet_gate.load_settings(), results_dir)
    passed = {(r["item_id"], r["config"]) for r in free if r["passed"]}
    wanted = {key for key in passed if key[1] in SCORED}
    if missing := sorted(wanted - set(sonnet)):
        raise DevRunError(
            f"{len(missing)} answers that passed the free checks have no Sonnet verdict, e.g. {missing[0]}"
        )
    if stray := sorted(set(sonnet) - wanted):
        raise DevRunError(f"{len(stray)} Sonnet verdicts are on answers the gate must not judge, e.g. {stray[0]}")
    return build_table(
        free,
        {(a.item_id, a.config) for a in read_answers("dev")},
        paid={
            JEV_GATE: jev_paid_scores("dev", jev_gate.read_settings(), jev.read_client_settings(), results_dir),
            SONNET_GATE: sonnet,
        },
    )


def dev_measures(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    free = read_free_records("dev", results_dir)
    task_of = {str(r["item_id"]): str(r["source"]) for r in read_rows(split="dev")}
    client, sonnet = jev.read_client_settings(), sonnet_gate.load_settings()
    return measures(
        free, read_jev_records("dev", results_dir), collect_sonnet_gate.read_sonnet_records("dev", results_dir),
        collect_sonnet_gate.load_log(collect_sonnet_gate.batch_log_path(results_dir)), task_of, CONFIGS,
        jev_price=client.price_per_million_input_tokens, input_price=sonnet.input_per_million,
        output_price=sonnet.output_per_million, batch_discount=float(load_models()["batch_discount"]),
    )  # fmt: skip


def dev_record(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """The committed record of the dev run: the conditions it ran under, the order, the counts, the latency, the spend,
    the cutoffs it read and the hashes of the stored dev records."""
    path = conditions_path(results_dir)
    if not path.exists():
        raise DevRunError("no dev_conditions.json: capture the conditions before the dev collection starts")
    return {
        "purpose": "A17 item 20.5's dev run: the free checks, the Jev gate and the Sonnet gate's batch on dev's answers",
        "conditions": json.loads(path.read_text(encoding="utf-8")),
        "cutoffs": read_cutoffs(),
        **dev_measures(results_dir),
        "stored_sha256": stored_hashes(results_dir),
    }


def routers_dev_tables() -> dict[str, RouterScoreTable]:
    """The four routers' dev router-score tables, keyed by section name: E's and F's rebuilt from their routers, G's and
    G+'s from the stored Jev router responses."""
    from analysis import jev_table, rules_table, trained_table

    return {
        NAMED_ROUTERS[E]: rules_table.dev_table(),
        NAMED_ROUTERS[F]: trained_table.dev_table(),
        NAMED_ROUTERS[G]: jev_table.dev_table(NAMED_ROUTERS[G]),
        NAMED_ROUTERS[G_PLUS]: jev_table.dev_table(NAMED_ROUTERS[G_PLUS]),
    }


def dev_walks(dev: Matrix | None = None, results_dir: Path = RESULTS_DIR) -> dict[str, gate_dev.Walks]:
    """The fifteen walks on dev, from the stored records, the router tables and the fixed cutoffs."""
    dev = dev or load_matrix().split("dev")
    return gate_dev.cascade_walks(dev, dev_table(results_dir), routers_dev_tables(), read_cutoffs(), resample(dev))


# ---- the command line -------------------------------------------------------------------------------------------


def describe(key: str, walks: gate_dev.Walks) -> list[str]:
    from analysis.router_names import CASCADES

    dial = gate_dev.dial_of(key)
    lines = [f"{CASCADES[key]} ({key}):"]
    for mix, by_share in walks.items():
        for share, w in by_share.items():
            op = w.operating
            where = (
                "falls back to A"
                if w.fell_back
                else (f"{dial} {getattr(op, dial)!r}" if getattr(op, dial) is not None else "the free checks")
            )
            lines.append(f"  {mix:9} {share:4.0%}: {where}; {len(w.certified) - 1} of {len(w.steps) - 1} steps "
                         f"certified; quality {100 * op.quality:.1f}%, ${op.cost_per_1k:.2f} per 1k (target "
                         f"{100 * w.target:.1f}%, margin {100 * op.margin:.2f})")  # fmt: skip
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("what", choices=["conditions", "measures", "record", "walks"])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.what == "conditions":
        print(json.dumps(capture_conditions(), indent=2))
    elif args.what == "measures":
        print(json.dumps(dev_measures(), indent=2))
    elif args.what == "record":
        text = json.dumps(dev_record(), indent=2) + "\n"
        if args.write:
            RECORD_FILE.write_text(text, encoding="utf-8")
        print(text)
    else:
        dev = load_matrix().split("dev")
        walks = dev_walks(dev)
        sections = gate_dev.cascade_sections(walks)
        for key, w in walks.items():
            print("\n".join(describe(key, w)))
        if args.write:
            gate_dev.write_sections(sections)
            print(f"\nwrote {len(sections)} sections to {OPERATING_POINTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
