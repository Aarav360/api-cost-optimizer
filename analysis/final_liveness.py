"""The liveness calls: the final evaluation's second step (PLAN_v4.md Amendment A19, items 2.2 and 9; A1 item 1; issue #77, #82).

One made-up prompt goes to `haiku` at high effort (A2) and one to `opus` at medium, through the standard API, built by the
live replay's own request builder (`analysis/live_calls.py`). Each returned model ID is checked against the pin, and a
mismatch refuses (`LivenessRefusal`) before anything else is sent, so a retired model is found before test is opened to a
request and no substitute is ever chosen with outcomes known.

**The whole run is projected first** (`analysis/final_ledger.py`): this step is the first to call, so it adds up every
call ticket's projection and refuses over a cap before its own first call. The calls are stored in `liveness.jsonl` in the
final evaluation's folder and count under the $10 cap with the live replay.

**A resume asks again.** A liveness call isn't an outcome that a second draw would change, it asks whether the model still
answers, so each go sends both. The cost is cents, and a model retired between two goes is found at once. This is the researcher's
decision (#82), and a departure from A19 item 2's "a resume calls only requests with no stored response", which holds for every
other step; the record shows it, each go's liveness step listing its own two requests as sent (`analysis/check_final_ledger.py`).
"""

from analysis.collect_jev import append_record
from analysis.final_ledger import LIVE, check_projection
from analysis.final_run import Run, StepRefusal, StepResult
from analysis.live_calls import client_of, live_request, send, worst_case_dollars
from harness.client import load_models

LIVENESS_FILE = "liveness.jsonl"
PROMPT = "Reply with the single word: ready."  # made up: no benchmark item, no test item
CONFIGS = ("haiku", "opus")  # haiku at high effort, opus at medium (configs/models.yaml)


class LivenessRefusal(StepRefusal):
    """A liveness call that came back from a model other than the pin, or with no model ID."""


def requests() -> dict[str, dict]:
    models = load_models()
    return {config: live_request(config, PROMPT, models=models) for config in CONFIGS}


def project(_run: Run) -> float:
    """The liveness calls' projected cost, an upper bound: each request's prompt and its whole `max_tokens`. It reads nothing
    of the run, but takes it, as every projector does (`final_ledger.PROJECTORS`: a projector is `(run) -> dollars`)."""
    return sum(worst_case_dollars(request) for request in requests().values())


def liveness_step(run: Run) -> StepResult:
    account = check_projection(run)  # a cap refuses here, before the first call
    client = client_of(run, LivenessRefusal)
    asked = requests()
    returned: dict[str, str | None] = {}
    for config, request in asked.items():
        record = send(client, request)
        append_record(run.world.output / LIVENESS_FILE,
               {"go": run.record.go["go"], "config": config, **record, "usage": record["response"]["usage"]})  # fmt: skip
        returned[config] = record["response"].get("model")
        run.say(f"liveness: {config} answered in {record['latency_s']:.2f} s")
    wrong = {c: (asked[c]["model"], returned[c]) for c in asked if returned[c] != asked[c]["model"]}
    if wrong:
        raise LivenessRefusal("; ".join(f"{c} was asked as {pin} and answered as {got}" for c, (pin, got) in wrong.items())
                              + "; no test request was sent")  # fmt: skip
    run.say(f"liveness: both models answered as pinned; ${account[LIVE]['projected']:.4f} projected for the live pool")
    return StepResult(requests={"sent": len(asked), "stored": 0}, detail={"models": returned, "spend": account})
