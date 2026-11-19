"""The live replay: the final evaluation's seventh step (PLAN_v4.md Amendment A17, item 26.5; A19, items 2.7, 8, 9 and 10;
issue #77, #91).

**What runs.** A, and the best cascade (item 24's rule, `analysis/replay_plan.py`) at its realistic-mix 99% and 97% operating
points, a point that falls back to A skipped. Each runs on the unlock's 100 items, 25 for each task (`handle.replay_items()`),
end to end through the standard API: the router if the cascade starts at one, the model, the free checks, the paid gate and
the escalation to `opus`. **The configs a request calls are the deployable cascade decision's** (`routers/cascade.py`, the
function the offline traces call too), from the gates' own modules and the liveness calls' request builder
(`analysis/live_calls.py`); nothing here decides anything a gate or a cutoff decides. Requests go one at a time, an item's
policies together (so API load touches them alike), and each is timed from the request to the answer returned, on one clock.

**What is stored.** Every model and gate call, as it is made, in `replay.jsonl` in the final evaluation's folder (it is the
live pool's, `analysis/final_ledger.py`), and one summary per finished request in `replay_requests.jsonl`. A crash stops the run
and the next go sends only what has no stored response: a request that was cut keeps its stored calls and finishes from them,
flagged `interrupted` so its time (which spans the crash) is set apart, not counted. A stored response is never drawn again.

**What is reported** (from the tables step on, which this step follows): per policy the median and p90 latency, the cost per
request and the escalation rate, beside the offline estimate for the same 100 items, and the answers graded with the same
graders (`harness.collect.make_row`, one at a time on the main thread after every call is made) and shown with their n, not
judged. The conditions (machine, network, the go's times) go in the record, with the first call and any reconnect apart.

**The cap.** The live pool counts the liveness calls with the replay. The projection (`project`) is each pending request's
stored cost on test, doubled (`SAFETY`), and the escalation counted as if every request needed it, summed and checked before
the first call of the run (the ledger's `check_projection`, and again here); before each call the pool's recorded spend plus
that call's worst case (its whole `max_tokens`) must fit, so a run that costs more than projected still stops at the cap.
"""

import contextlib
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from analysis import final_gates, final_grading
from analysis.collect_jev import append_record, trim_torn_line
from analysis.final_ledger import CAPS, LIVE, LedgerRefusal, Spend, call_dollars, live_spend, recorded_spend, refusal
from analysis.final_run import ItemFailure, Run, StepRefusal, StepResult
from analysis.gate_spend import jev_projected_cost, list_dollars
from analysis.live_calls import client_of
from analysis.record_files import read_jsonl
from analysis.replay_plan import (
    ROUTER_EXTRAS,
    Estimate,
    Plan,
    Policy,
    ReplayPlanError,
    best_cascade_section,
    code_items_of,
    latency_stats,
    make_plan,
    offline_estimate,
    read_medians,
)
from analysis.router_names import HAIKU, JEV_GATE, SONNET_GATE
from analysis.unlock import Handle
from harness.client import answer_text, load_models
from harness.collect import Job, make_row
from routers import jev, jev_gate, rules, sonnet_gate
from routers.cascade import configs_called, paid_gate_called
from routers.decision import choose
from routers.gates import free_checks, read_gate_settings
from routers.ladder import FRONTIER, SCORED
from routers.request import Request

# In the final evaluation's folder: the calls (priced by the ledger) and the finished requests.
CALLS_FILE = "replay.jsonl"
REQUESTS_FILE = "replay_requests.jsonl"
SAFETY = 2.0  # the projection's margin over a request's stored cost: a fresh draw can run longer than the stored one
SONNET_TEMPLATE_TOKENS = 2000  # a Sonnet gate call's prompt template, rounded up
# Its reply: dev's mean was about 290 tokens, and a call that runs to the cap is the call guard's to stop.
SONNET_OUTPUT_TOKENS = 500


class ReplayRefusal(StepRefusal):
    """A replay that can't go on: a model that isn't the pin, a stored record of another plan, or a cap. Its message
    names a model, a policy or a cap, never an answer."""


# ---- the plan and what a go holds ---------------------------------------------------------------------------------


def plan_of() -> Plan:
    """The policies: item 24's best cascade at its realistic 99% and 97% points, and A. Refuses (ReplayPlanError) what the
    plan module refuses, as a refusal of the step."""
    try:
        return make_plan(best_cascade_section())
    except ReplayPlanError as error:
        raise ReplayRefusal(str(error)) from error


def stored_state(
    output: Path, plan: Plan, items: Sequence[str]
) -> tuple[dict[tuple[str, str, str], dict], dict[tuple[str, str], dict]]:
    """The calls and the finished requests an earlier go stored, each by its policy, item and (for a call) role. Refuses
    (ReplayRefusal) a record of a policy or an item that isn't this plan's: the stored answers belong to another run."""
    keys, wanted = {p.key for p in plan.policies}, set(items)
    for name in (CALLS_FILE, REQUESTS_FILE):
        trim_torn_line(output / name)
    calls = {(r["policy"], r["item_id"], r["role"]): r for r in read_jsonl(output / CALLS_FILE, missing_ok=True)}
    done = {(r["policy"], r["item_id"]): r for r in read_jsonl(output / REQUESTS_FILE, missing_ok=True)}
    stray = [k for k in (*calls, *done) if k[0] not in keys or k[1] not in wanted]
    if stray:
        raise ReplayRefusal(
            f"the stored replay holds {len(stray)} records of another plan or other items, e.g. {stray[0][:2]}"
        )
    return calls, done


# ---- the projection ---------------------------------------------------------------------------------------------


def stored_usage(handle: Handle) -> dict[tuple[str, str], tuple[int, int]]:
    """The most input and output tokens any stored run of an (item, config) was billed: token counts only, from the handle's
    stored-results read, which holds no answer and no expected answer, for the projection."""
    found: dict[tuple[str, str], tuple[int, int]] = {}
    for stored in handle.stored_results():
        key = (stored.item_id, stored.config)
        before = found.get(key, (0, 0))
        found[key] = (max(before[0], stored.input_tokens), max(before[1], stored.output_tokens))
    return found


def request_bound(
    policy: Policy,
    item_id: str,
    usage: Mapping[tuple[str, str], tuple[int, int]],
    models: Mapping,
    jev_price: float,
    jev_budget: int,
) -> float:
    """An upper bound, in dollars, on one request of `policy` on `item_id`: SAFETY times the stored cost of each config it
    may call (the start at the dearest it can be, and `opus` in case of escalation) and of its paid gate."""

    def dollars(config: str, tokens: tuple[int, int]) -> float:
        price = models["prices"][models["configs"][config]["model"]]
        return list_dollars(tokens[0], tokens[1], float(price["input"]), float(price["output"]))

    opus = dollars(FRONTIER, usage[item_id, FRONTIER])
    if policy.section is None:
        return SAFETY * opus
    starts = (HAIKU,) if policy.start == HAIKU else SCORED
    start = max(starts, key=lambda c: dollars(c, usage[item_id, c]))
    gate = 0.0
    if policy.gate == JEV_GATE:
        gate = jev_projected_cost([jev_budget], jev_price)
    elif policy.gate == SONNET_GATE:
        settings = sonnet_gate.load_settings()
        tokens_in = usage[item_id, start][0] + usage[item_id, start][1] + SONNET_TEMPLATE_TOKENS
        gate = list_dollars(tokens_in, SONNET_OUTPUT_TOKENS, settings.input_per_million, settings.output_per_million)
    return SAFETY * (dollars(start, usage[item_id, start]) + gate + opus)


def project(run: Run) -> float:
    """The replay's projected spend for the ledger (an upper bound in the live pool): each request with no finished summary,
    by `request_bound`. Refuses (ReplayRefusal) a plan it can't build, so a cascade the replay can't run is found before any
    call is made."""
    handle = run.unlocked()
    plan = plan_of()
    items = handle.replay_items()
    _, done = stored_state(run.world.output, plan, items)
    usage = stored_usage(handle)
    models = load_models()
    price = jev.read_client_settings().price_per_million_input_tokens
    budget = jev_gate.read_settings().budget_bytes or 0
    return sum(
        request_bound(p, i, usage, models, price, budget)
        for i in items
        for p in plan.policies
        if (p.key, i) not in done
    )


# ---- one request ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """What the live phase of one go holds: the clients, the settings, the stored state and the go's first-call marks."""

    run: Run
    client: Any
    models: Mapping
    clock: Callable[[], float]
    jev_price: float
    code: Any
    runner: Any
    digest: str
    jev_client: Any | None
    jev_settings: jev_gate.JevGateSettings
    sonnet: sonnet_gate.SonnetSettings
    scorer: Callable[[Request], dict[str, float]] | None
    calls: dict[tuple[str, str, str], dict]
    go: int
    seen: set[str] = field(
        default_factory=set
    )  # services this go has already called once (the first call opens a connection)

    @property
    def path(self) -> Path:
        return self.run.world.output / CALLS_FILE

    def guard(self, worst_case: float) -> None:
        """Refuse (LedgerRefusal) a call whose worst case wouldn't fit under the live pool's cap with what is recorded."""
        words = refusal(
            Spend(0.0, live_spend(self.run.world.output, self.models, self.jev_price), 0), {LIVE: worst_case}
        )
        if words:
            raise LedgerRefusal(words)

    def store(self, policy: Policy, item_id: str, role: str, record: dict) -> dict:
        full = {"kind": "call", "go": self.go, "policy": policy.key, "item_id": item_id, "role": role, **record}
        append_record(self.path, full)
        self.calls[policy.key, item_id, role] = full
        return full


@dataclass
class Facts:
    """What one request noted on the way: whether it reused a stored call, opened a connection, reconnected or retried."""

    resumed: bool = False
    first: bool = False
    reconnect: bool = False
    retries: int = 0

    def first_use(self, ctx: Context, service: str) -> bool:
        new = service not in ctx.seen
        ctx.seen.add(service)
        self.first |= new
        return new


def check_pin(who: str, asked: str, answered: object) -> None:
    """Refuse (ReplayRefusal) a call answered by a model other than the one asked, naming both: a stored call is checked
    again when a resume reuses it, so a refusal that stopped one go stops the next."""
    if answered != asked:
        raise ReplayRefusal(f"{who} was asked as {asked} and answered as {answered}")


def model_call(
    ctx: Context, facts: Facts, policy: Policy, item_id: str, role: str, config: str, request: Request
) -> dict:
    """One standard-API model call, or the stored one. A returned model that isn't the one asked refuses."""
    if (stored := ctx.calls.get((policy.key, item_id, role))) is not None:
        facts.resumed = True
        check_pin(config, stored["request"]["model"], stored["response"].get("model"))
        return stored
    from analysis.live_calls import live_request, send, worst_case_dollars

    params = live_request(config, request.prompt, output_schema=request.output_schema, models=ctx.models)
    ctx.guard(worst_case_dollars(params, ctx.models))
    facts.first_use(ctx, "anthropic")
    sent = send(ctx.client, params)
    full = ctx.store(policy, item_id, role, {"service": "anthropic", **sent, "usage": sent["response"]["usage"]})
    check_pin(config, params["model"], sent["response"].get("model"))
    return full


def gate_call(
    ctx: Context, facts: Facts, policy: Policy, item_id: str, request: Request, answer: str
) -> tuple[float | None, bool]:
    """The paid gate's call on the answer, or the stored one: its score (None if refused) and whether it refused."""
    stored = ctx.calls.get((policy.key, item_id, "gate"))
    if stored is not None:
        facts.resumed = True
    if policy.gate == JEV_GATE:
        if stored is None:
            budget = ctx.jev_settings.budget_bytes
            assert budget is not None and ctx.jev_client is not None
            call = jev_gate.gate_call(request, answer, ctx.jev_settings, budget_bytes=budget)
            ctx.guard(jev_projected_cost([len(call.body)], ctx.jev_price))
            new = facts.first_use(ctx, "jev")
            result = ctx.jev_client.call(call.body, call.sha256, f"Jev gate replay {policy.key} {item_id}")
            facts.reconnect |= bool(result.new_connection) and not new
            facts.retries += len(result.failed_attempts)
            extra = {"answer_cut_bytes": call.answer_cut_bytes, "prompt_cut_bytes": call.prompt_cut_bytes}
            stored = ctx.store(policy, item_id, "gate", {"service": "jev", **result.to_record(), **extra})
        check_pin("the Jev gate", ctx.jev_settings.model, stored["model"])
        read = jev_gate.read_response(stored["response"], ctx.jev_settings)
        return read.score, read.refused
    if stored is None:
        started = ctx.clock()
        gate_request = sonnet_gate.build_request(ctx.sonnet, request, answer)
        from analysis.live_calls import worst_case_dollars

        ctx.guard(worst_case_dollars(gate_request, ctx.models))
        facts.first_use(ctx, "anthropic")
        checked = sonnet_gate.check_answer(ctx.client, ctx.sonnet, request, answer)
        stored = ctx.store(policy, item_id, "gate", {"service": "anthropic", "request": checked.params,
                                                     "response": checked.response, "usage": checked.response["usage"],
                                                     "request_id": checked.request_id, "latency_s": ctx.clock() - started})  # fmt: skip
    check_pin("the Sonnet gate", stored["request"]["model"], stored["response"].get("model"))
    reply = sonnet_gate.parse_response(stored["response"])
    return reply.score, reply.refused


def start_of(ctx: Context, policy: Policy, request: Request) -> str:
    """The config a request starts at: `opus` for A, `haiku` for a pure cascade, the router's pick at the point's threshold
    for a router-started one (E and F score locally, so a resumed request re-scores and gets the same pick)."""
    if policy.section is None:
        return FRONTIER
    if policy.start == HAIKU:
        return HAIKU
    assert ctx.scorer is not None and policy.threshold is not None
    return choose(ctx.scorer(request), policy.threshold)


def run_request(ctx: Context, policy: Policy, item_id: str, request: Request) -> dict:
    """One request of one policy end to end, timed from the request to the answer returned, and its summary record (not yet
    stored). The configs called are `routers.cascade.configs_called`'s, given whether the free checks passed and the gate's score."""
    facts = Facts()
    started_at = ctx.run.stamp()
    began = ctx.clock()
    start = start_of(ctx, policy, request)
    first = model_call(ctx, facts, policy, item_id, "start", start, request)
    free_passed: bool | None = None
    score: float | None = None
    refused: bool | None = None
    called: tuple[str, ...] = (start,)
    if start != FRONTIER:
        response = first["response"]
        fields = free_checks(request, answer_text(response), response["stop_reason"], ctx.code, digest=ctx.digest,
                             now=lambda: final_gates.stamp(ctx.run), run=ctx.runner, sleep=time.sleep)  # fmt: skip
        free_passed = bool(fields["passed"])
        if paid_gate_called(start, free_passed, policy.cutoff):
            score, refused = gate_call(ctx, facts, policy, item_id, request, answer_text(response))
        called = configs_called(start, free_passed, score, policy.cutoff)
        if called != (start,):
            model_call(ctx, facts, policy, item_id, "escalation", FRONTIER, request)
    seconds = ctx.clock() - began
    mine = [
        ctx.calls[key] for role in ("start", "gate", "escalation") if (key := (policy.key, item_id, role)) in ctx.calls
    ]
    cost = sum(call_dollars(call, "replay", ctx.models, ctx.jev_price) for call in mine)
    return {
        "kind": "request", "go": ctx.go, "policy": policy.key, "item_id": item_id, "start": start,
        "configs": list(called), "answer_config": called[-1], "free_passed": free_passed, "gate_score": score,
        "gate_refused": refused, "latency_s": seconds, "cost_usd": cost, "first_call": facts.first,
        "reconnect": facts.reconnect, "retries": facts.retries, "interrupted": facts.resumed, "started": started_at,
        "ended": ctx.run.stamp(),
    }  # fmt: skip


# ---- the live phase and grading -----------------------------------------------------------------------------------


def scorer_of(run: Run, plan: Plan) -> Callable[[Request], dict[str, float]] | None:
    """The local router's `score` for a cascade that starts at E or F, else None."""
    starts = {p.start for p in plan.policies if p.section is not None}
    if starts == {HAIKU} or not starts:
        return None
    if starts == {"rules"}:
        return rules.score
    from analysis.final_routers import trained_scorer

    return trained_scorer(run.world)[0]


def conditions() -> dict[str, str]:
    """The machine and the network, as the earlier records word them; "not determined" where the tools are missing."""
    from analysis import jev_train_report

    try:
        return {"machine": jev_train_report.machine(), "network": jev_train_report.network()}
    except OSError:
        return {"machine": "not determined", "network": "not determined"}


def live_phase(
    run: Run,
    plan: Plan,
    items: Sequence[str],
    requests: Mapping[str, Request],
    clock: Callable[[], float],
    calls: dict[tuple[str, str, str], dict],
    done: dict[tuple[str, str], dict],
) -> int:
    """Every request with no finished summary, one at a time, an item's policies together; each summary stored when its
    request ends. Returns how many were sent. A failure names the policy and the item."""
    pending = [(i, p) for i in items for p in plan.policies if (p.key, i) not in done]
    if not pending:
        return 0
    code = read_gate_settings().code_check
    runner, digest = final_gates.sandbox_of(run, code)
    needs_jev = any(p.gate == JEV_GATE for p in plan.policies)
    with contextlib.ExitStack() as stack:
        jev_client = stack.enter_context(final_gates.jev_client_of(run)) if needs_jev else None
        ctx = Context(run, client_of(run, ReplayRefusal), load_models(), clock, jev.read_client_settings().price_per_million_input_tokens,
                      code, runner, digest, jev_client, jev_gate.read_settings(), sonnet_gate.load_settings(),
                      scorer_of(run, plan), calls, run.record.go["go"])  # fmt: skip
        for number, (item_id, policy) in enumerate(pending, start=1):
            try:
                summary = run_request(ctx, policy, item_id, requests[item_id])
                append_record(run.world.output / REQUESTS_FILE, summary)
            except Exception as error:
                raise ItemFailure(f"{policy.key} {item_id}") from error
            done[policy.key, item_id] = summary
            if number % 25 == 0:
                run.say(f"replay: {number} of {len(pending)} requests sent")
    return len(pending)


def grade(
    run: Run,
    plan: Plan,
    items: Sequence[str],
    calls: Mapping[tuple[str, str, str], dict],
    done: Mapping[tuple[str, str], dict],
) -> dict[tuple[str, str], dict]:
    """Each finished request's final answer, graded by the collection run's own row building, one at a time on the main
    thread with nothing else running (the fake Jev server stops with its session before this). Returns `(policy, item)` to
    the row's status and success, and the item's task."""
    final_grading.require_alone()
    rows = final_grading.item_rows(run.unlocked(), set(items))
    models = load_models()
    grades: dict[tuple[str, str], dict] = {}
    with final_grading.sandbox_of(run):
        for (policy_key, item_id), summary in done.items():
            role = "escalation" if len(summary["configs"]) > 1 else "start"
            call = calls[policy_key, item_id, role]
            config = summary["answer_config"]
            custom_id = re.sub(r"[^A-Za-z0-9_-]", "-", f"replay_{policy_key}_{item_id}")
            try:
                job = Job(custom_id, "test", rows[item_id], config, 0, call["request"])
                row = make_row(job, {"run": 0, "request": call["request"], "response": call["response"], "batch_id": None,
                                     "latency_s": call["latency_s"]}, models)  # fmt: skip
            except Exception as error:
                raise ItemFailure(f"{policy_key} {item_id}") from error
            grades[policy_key, item_id] = {
                "status": row["status"],
                "success": row["success"],
                "task": rows[item_id]["source"],
            }
    return grades


# ---- the report ---------------------------------------------------------------------------------------------------


def live_numbers(
    policy: Policy, summaries: Sequence[Mapping[str, Any]], grades: Mapping[tuple[str, str], dict]
) -> dict[str, Any]:
    """A policy's measured numbers: latency (the first call, reconnects and cut requests apart), cost per request, the
    escalation rate and the graded quality with its n, overall and per task."""
    mine = [s for s in summaries if s["policy"] == policy.key]
    graded = [grades[policy.key, s["item_id"]] for s in mine]
    tasks = sorted({g["task"] for g in graded})
    return {
        **latency_stats(mine),
        "cost_per_request": statistics.fmean(s["cost_usd"] for s in mine),
        "escalation_rate": sum(len(s["configs"]) > 1 for s in mine) / len(mine),
        "quality": {"value": statistics.fmean(g["success"] for g in graded), "n": len(graded)},
        "quality_by_task": {
            t: {
                "value": statistics.fmean(g["success"] for g in graded if g["task"] == t),
                "n": sum(g["task"] == t for g in graded),
            }
            for t in tasks
        },  # fmt: skip
        "retries": sum(s["retries"] for s in mine),
        "configs_called": dict(Counter(c for s in mine for c in s["configs"])),
    }


def offline_numbers(run: Run, plan: Plan, items: Sequence[str], requests: Mapping[str, Request]) -> dict[str, Estimate]:
    """Each policy's offline estimate for the same items, from test's matrix and gate-score table (the tables step's own
    inputs, `analysis/offline_half.py`)."""
    from analysis import offline_half

    inputs = offline_half.test_inputs(run)
    table = inputs.extras[final_gates.GATE_TABLE].table
    routers = {s: inputs.extras[extra] for s, extra in ROUTER_EXTRAS.items() if extra in inputs.extras}
    medians, code_items = read_medians(), code_items_of(requests, items)
    return {
        p.key: offline_estimate(p, inputs.matrix, table, routers, items, code_items, medians) for p in plan.policies
    }


def say_policy(run: Run, policy: Policy, live: Mapping[str, Any], offline: Estimate) -> None:
    def seconds(value: float | None) -> str:
        return "none" if value is None else f"{value:.2f} s"

    run.say(f"replay {policy.label}: median {seconds(live['median_s'])} (estimated {offline.latency_median_s:.2f} s), p90 "
            f"{seconds(live['p90_s'])} (estimated {offline.latency_p90_s:.2f} s) over {live['kept']} requests; cost per request "
            f"${live['cost_per_request']:.5f} (estimated ${offline.cost_per_request:.5f}); escalation "
            f"{live['escalation_rate']:.1%} (estimated {offline.escalation_rate:.1%}); quality {live['quality']['value']:.1%} "
            f"of {live['quality']['n']} (estimated {offline.quality:.1%}), not judged; first call "
            f"{[round(s, 2) for s in live['first_call_s']]}, reconnects {[round(s, 2) for s in live['reconnects_s']]}, "
            f"cut {len(live['interrupted_s'])}")  # fmt: skip


def replay_step(run: Run) -> StepResult:
    """The live replay (see the module's docstring). Returns the record's detail: the plan, each policy's live and offline
    numbers, the conditions and the spend."""
    handle = run.unlocked()
    plan = plan_of()
    items, requests = handle.replay_items(), handle.requests()
    if missing := [i for i in items if i not in requests]:
        raise ReplayRefusal(f"{len(missing)} replay items are not test requests")
    output = run.world.output
    calls, done = stored_state(output, plan, items)
    already = len(done)
    words = refusal(recorded_spend(run.world.results, output), {LIVE: project(run)})
    if words:
        raise LedgerRefusal(words)
    began = run.stamp()
    sent = live_phase(run, plan, items, requests, time.perf_counter, calls, done)
    ended = run.stamp()
    run.say(f"replay: {sent} requests sent, {already} already stored, under the {CAPS[LIVE]:.0f} dollar live cap")
    grades = grade(run, plan, items, calls, done)
    summaries = list(done.values())
    offline = offline_numbers(run, plan, items, requests)
    results = {}
    for policy in plan.policies:
        live = live_numbers(policy, summaries, grades)
        say_policy(run, policy, live, offline[policy.key])
        results[policy.key] = {"label": policy.label, "live": live, "offline": asdict(offline[policy.key])}
    first = plan.policies[0].key
    per_task = dict(Counter(g["task"] for (key, _), g in grades.items() if key == first))
    detail = {
        "cascade": plan.section, "rule": "A17 item 24, applied to dev's table v4 by final_headline.best_cascade, the headline's own choice",
        "policies": [asdict(p) for p in plan.policies], "skipped": list(plan.skipped), "items": len(items),
        "per_task": per_task,
        "conditions": {**conditions(), "go_started": began, "go_ended": ended},
        "results": results,
    }  # fmt: skip
    return StepResult(requests={"sent": sent, "stored": already}, detail=detail)
