"""The liveness calls: the final evaluation's second step (PLAN_v4.md Amendment A19, items 2.2 and 9; A1 item 1; issue #77, #82).

One made-up prompt goes to `haiku` at high effort (A2) and one to `opus` at medium, through the standard API, built by the
live replay's own request builder (`analysis/live_calls.py`). Each returned model ID is checked against the pin, and a
mismatch refuses (`LivenessRefusal`) before anything else is sent, so a retired model is found before test is opened to a
request and no substitute is ever chosen with outcomes known.

**The whole run is projected first** (`analysis/final_ledger.py`): this step is the first to call, so it adds up every
call ticket's projection and refuses over a cap before its own first call. The calls are stored in `liveness.jsonl` in the
final evaluation's folder and count under the $10 cap with the live replay.

**A resume asks again.** A liveness call isn't an outcome that a second draw would change, it asks whether the model still
answers, so each go sends both. The cost is cents, and a model retired between two goes is found at once.
"""

import json
import os
from pathlib import Path

from analysis.final_ledger import LIVE, check_projection
from analysis.final_run import Run, StepRefusal, StepResult
from analysis.live_calls import live_request, send, worst_case_dollars
from harness.client import load_models, make_client

LIVENESS_FILE = "liveness.jsonl"
PROMPT = "Reply with the single word: ready."  # made up: no benchmark item, no test item
CONFIGS = ("haiku", "opus")  # haiku at high effort, opus at medium (configs/models.yaml)


class LivenessRefusal(StepRefusal):
    """A liveness call that came back from a model other than the pin, or with no model ID."""


def requests() -> dict[str, dict]:
    models = load_models()
    return {config: live_request(config, PROMPT, models=models) for config in CONFIGS}


def project(run: Run) -> float:
    """The liveness calls' projected cost, an upper bound: each request's prompt and its whole `max_tokens`."""
    return sum(worst_case_dollars(request) for request in requests().values())


def client_of(run: Run):
    """The real client for the go's world (no backends named), the world's own fake for a rehearsal's."""
    backends = run.world.backends
    if backends is None:
        return make_client()
    if backends.anthropic is None:
        raise LivenessRefusal("the world names backends but no Anthropic client: only the go's world, which names none, "
                            "gets the real one")  # fmt: skip
    return backends.anthropic


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def liveness_step(run: Run) -> StepResult:
    account = check_projection(run)  # a cap refuses here, before the first call
    client = client_of(run)
    asked = requests()
    returned: dict[str, str | None] = {}
    for config, request in asked.items():
        record = send(client, request)
        append(run.world.output / LIVENESS_FILE,
               {"go": run.record.go["go"], "config": config, **record, "usage": record["response"]["usage"]})  # fmt: skip
        returned[config] = record["response"].get("model")
        run.say(f"liveness: {config} answered in {record['latency_s']:.2f} s")
    wrong = {c: (asked[c]["model"], returned[c]) for c in asked if returned[c] != asked[c]["model"]}
    if wrong:
        raise LivenessRefusal("; ".join(f"{c} was asked as {pin} and answered as {got}" for c, (pin, got) in wrong.items())
                              + "; no test request was sent")  # fmt: skip
    run.say(f"liveness: both models answered as pinned; ${account[LIVE]['projected']:.4f} projected for the live pool")
    return StepResult(requests={"sent": len(asked), "stored": 0}, detail={"models": returned, "spend": account})
