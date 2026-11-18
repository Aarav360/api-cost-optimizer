"""The final evaluation's record: its skeleton (PLAN_v4.md Amendment A19, items 2, 5 and 10; issue #79).

One JSON file in the world's output folder, written after every event so a crash keeps what was done. It holds one entry
per go, in order, because a resume is a new go that runs the unlock again and the record keeps every go's unlock entry:

    {"record": "final evaluation", "gos": [{"go": 1, "started": ..., "ended": ..., "unlock": {...} | null,
                                           "stopped": {...} | null, "steps": [{"name", "started", "ended", "status",
                                                                              "requests", "detail"}]}]}

Before the tables step the record holds only what A19 item 5 lets be shown: times, call and request counts and spend. A
step's `detail` is for that kind of fact only. #92 completes the record with item 10's other fields, by adding
to what a step's result carries and to `finish` (the environment, the file hashes, the spend against each cap).

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
        """The go ran every step. #92 adds item 10's closing fields here."""
        self.go["ended"] = now
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
