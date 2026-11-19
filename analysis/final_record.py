"""The final evaluation's record: its skeleton (PLAN_v4.md Amendment A19, items 2, 5 and 10; issue #79).

One JSON file in the world's output folder, written after every event so a crash keeps what was done. It holds one entry
per go, in order, because a resume is a new go that runs the unlock again and the record keeps every go's unlock entry:

    {"record": "final evaluation", "gos": [{"go": 1, "started": ..., "ended": ..., "unlock": {...} | null,
                                           "stopped": {...} | null, "steps": [{"name", "started", "ended", "status",
                                                                              "requests", "detail"}]}]}

Before the tables step the record holds only what A19 item 5 lets be shown: times, call and request counts and spend. A
step's `detail` is for that kind of fact only. Item 10's fields sit in the steps' details and the unlock's entry, and
`analysis/final_record_fields.py` lists where, holds the record to the list and the tables to the record:

- the unlock's entry: commit, time, hook, the environment, each check's result, the 100 replay items (`checks`, `replay_draw`);
- the liveness step: the returned model IDs; the four drift steps: D6, D8, the shifts, the flips, F's largest difference;
- `calls_jev_router`: the first test Jev call's server `Date`; `calls_sonnet_gate`: the test batch's ID and creation time;
- the tables step, the first allowed to show outcomes: the stops (refusals and truncations), F's fold shares and the extras'
  counts; the replay step: its conditions, with the first call and any reconnects apart;
- the closing step: the spend against each cap and the sha256 of every test record file.

Nothing in it is a grade, and a check searches it for the rehearsal's markers (`analysis/check_final_run.py`).
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RECORD_FILE = "final_record.json"
TITLE = "final evaluation"


class Record:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"record": TITLE, "gos": []}
        )

    @property
    def go(self) -> dict[str, Any]:
        return self.data["gos"][-1]

    def start_go(self, now: str) -> int:
        """A new go: the next number, with nothing in it yet."""
        number = len(self.data["gos"]) + 1
        self.data["gos"].append(
            {"go": number, "started": now, "ended": None, "unlock": None, "stopped": None, "steps": []}
        )
        self.save()
        return number

    def set_unlock(self, entry: Mapping[str, Any]) -> None:
        self.go["unlock"] = dict(entry)
        self.save()

    def start_step(self, name: str, now: str) -> None:
        self.go["steps"].append(
            {"name": name, "started": now, "ended": None, "status": "running", "requests": {}, "detail": {}}
        )
        self.save()

    def end_step(self, status: str, now: str, requests: Mapping[str, int], detail: Mapping[str, Any]) -> None:
        step = self.go["steps"][-1]
        step.update(ended=now, status=status, requests=dict(requests), detail=dict(detail))
        self.save()

    def stop(self, step: str, error: Mapping[str, Any], now: str) -> None:
        """The go stopped at `step`: `error` names the exception's class and the item, never a grade."""
        if self.go["steps"] and self.go["steps"][-1]["status"] == "running":
            self.end_step("failed", now, {}, {})
        self.go["stopped"] = {"step": step, **error}
        self.go["ended"] = now
        self.save()

    def finish(self, now: str) -> None:
        """The go ran every step. Item 10's closing fields (the spend, the file hashes) are the closing step's own detail."""
        self.go["ended"] = now
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
