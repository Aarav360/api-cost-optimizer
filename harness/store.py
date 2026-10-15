"""On-disk response store (PLAN_v4.md Amendment A1, item 6).

Every API response is saved once and reused on re-runs, so re-running analysis
or fixing scoring code never pays for the same call twice.

Layout: <root>/<model>/<effort>/<request_hash>_r<run>.json

The hash covers the whole request (model, effort, max_tokens, system prompt,
messages), not just the prompt text. Changing any of those is a different
request and gets a fresh call instead of silently reusing an old answer.

Use a separate root per purpose (e.g. results/responses/pilot vs. a sealed root
for test answers) so cached test responses are never reachable from dev code.
"""

import hashlib
import json
import os
from pathlib import Path


def request_hash(request: dict) -> str:
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, request: dict, run: int) -> Path:
        effort = request.get("output_config", {}).get("effort", "default")
        name = f"{request_hash(request)[:16]}_r{run}.json"
        return self.root / request["model"] / effort / name

    def get(self, request: dict, run: int = 0) -> dict | None:
        path = self.path_for(request, run)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        # A 16-hex-char prefix collision is very unlikely, but a wrong cache hit
        # would corrupt results silently, so check the full hash too.
        if record["request_hash"] != request_hash(request):
            raise RuntimeError(f"Hash prefix collision at {path}")
        return record

    def put(self, request: dict, run: int, record: dict) -> Path:
        path = self.path_for(request, run)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file and rename, so a crash mid-write never leaves a
        # half-written record that a later run would trust.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"request_hash": request_hash(request), **record}, indent=2))
        os.replace(tmp, path)
        return path
