"""Whose record files are whose: Stage 7's and Stage 8's (PLAN_v4.md Amendment A19, items 9 and 13; A17 item 14).

Stage 7's collectors wrote `results/gates/` and `results/jev/` files labelled `train`, `dev` and `smoke`. Stage 8 writes
beside them, with the same stems and two new labels (A17 item 14 names test's `*_test.jsonl`):

- **`test`**: the calls on test items, `{free_checks,jev_gate,sonnet_gate}_test.jsonl` in `results/gates/` and
  `{jev,jev_bolstered}_test.jsonl` in `results/jev/`;
- **`drift`**: the drift checks' resends of train items (A11 item 17, A15 item 17, A17 item 26.2), apart from Stage 7's
  records, `{jev_gate,sonnet_gate}_drift.jsonl` and `{jev,jev_bolstered}_drift.jsonl`.

A file is Stage 8's if and only if the last `_` part of its stem is one of those labels. Everything else in the two
folders is Stage 7's, so a file with a name nobody planned for is counted in Stage 7's spend and held by its guards, never
left out of both. Stage 7's ledger (`analysis/gate_spend.py`), the numbers `analysis/check_final_design.py` quotes and the
hook's "no test record" guards (`analysis/check_gates_freeze.py`, `analysis/check_gates_dev.py`) skip Stage 8's files by this
rule, and Stage 8's ledger (`analysis/final_ledger.py`) reads only them. None of these names is in a freeze (A18 item 1
hashes train's and dev's record files by their own names).

The guard keeps its bite before the go: a Stage 8 file may exist only where the final evaluation's record (`final/` under
the results folder, written by `analysis/final_record.py`) shows an unlock, since nothing writes one before it.
"""

import json
from pathlib import Path

STAGE_7_LABELS = ("train", "dev", "smoke")
STAGE_8_LABELS = ("test", "drift")
FOLDERS = ("gates", "jev")

# How a Stage 8 file's records are priced (`analysis/final_ledger.py`): Jev's input tokens at its price, an Anthropic
# batch at list price less the batch discount, a standard-API call at list price, or not at all (free checks cost nothing).
JEV, BATCH, STANDARD, FREE = "jev", "batch", "standard", "free"
STAGE_8_FILES: dict[str, dict[str, str]] = {
    "gates": {
        "free_checks_test.jsonl": FREE,
        "jev_gate_test.jsonl": JEV,
        "jev_gate_drift.jsonl": JEV,
        "sonnet_gate_test.jsonl": BATCH,
        "sonnet_gate_drift.jsonl": STANDARD,  # the drift resend goes through the standard API (A17 item 26.2)
    },
    "jev": {
        "jev_test.jsonl": JEV,
        "jev_bolstered_test.jsonl": JEV,
        "jev_drift.jsonl": JEV,
        "jev_bolstered_drift.jsonl": JEV,
    },
}

RECORD_FOLDER = "final"  # under the results folder: where the go's entry point writes the record
RECORD_NAME = "final_record.json"  # `analysis.final_record.RECORD_FILE`; a check holds the two equal


def is_stage_8(name: str) -> bool:
    """True for a record file whose stem ends in a Stage 8 label (`jev_gate_test.jsonl`), whatever else it is called."""
    path = Path(name)
    return path.suffix == ".jsonl" and path.stem.rsplit("_", 1)[-1] in STAGE_8_LABELS


def files_in(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.jsonl")) if folder.is_dir() else []


def stage_7_files(folder: Path) -> list[Path]:
    """The `.jsonl` files of `folder` that are Stage 7's."""
    return [path for path in files_in(folder) if not is_stage_8(path.name)]


def stage_8_files(results_dir: Path) -> list[Path]:
    """Every Stage 8 record file in the gate and Jev folders, in name order."""
    return [path for folder in FOLDERS for path in files_in(results_dir / folder) if is_stage_8(path.name)]


def unlocked(results_dir: Path) -> bool:
    """True if the final evaluation's record holds an unlock: a go passed the unlock, so test's records may exist."""
    path = results_dir / RECORD_FOLDER / RECORD_NAME
    if not path.is_file():
        return False
    gos = json.loads(path.read_text(encoding="utf-8")).get("gos", [])
    return any(go.get("unlock") for go in gos)


def unopened(results_dir: Path) -> list[str]:
    """The Stage 8 record files that exist with no unlock in the record: none before the go. The hook's guard."""
    return [] if unlocked(results_dir) else [f"{path.parent.name}/{path.name}" for path in stage_8_files(results_dir)]
