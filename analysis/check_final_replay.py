"""Check the live replay (PLAN_v4.md Amendment A17, items 24 and 26.5; A19, items 2.7, 8, 9 and 10; issue #77, #91).

Run from the project root:  uv run python -m analysis.check_final_replay

The replay on the final evaluation's made-up world (`analysis/rehearsal_world.py`: eight test items, so the replay's draw is
all eight), against the fake Anthropic client with injected delays, the fake Jev server and the stand-in sandbox. No results/,
no Docker, no Jev or Anthropic call, and nothing near the real store. Expected values are worked by hand below, from the fakes'
scripted answers, the plan's cutoffs and the price sheet, never from the code under test.

- the plan: item 24's rule on the committed dev savings names haiku/Jev (the plan's own sentence agrees), whose realistic 99% and
  97% points give two policies beside A; in a hand-made operating-points file where the 99% point falls back to A, that point
  is skipped; a cascade started by a Jev router is refused before any call; a tie the rule leaves open is refused;
- the configs called on each item are the deployable cascade decision's, worked by hand from whether each item's answer passes the free checks,
  its scripted gate score and the cutoff, for a Jev-gated cascade, a Sonnet-gated one (through the standard API, one point
  skipped as it falls back) and a cascade started by a router, in which the router's pick starts the item;
- calls go one at a time (no two fake calls overlap, shown on a pair of threads that do), and each request's time is the sum
  of the delays the fakes injected, within a margin; the median and p90 are over the requests with the first call, a reconnect
  and a cut request apart, and the hand-made list in the statistics' own case gives the hand-worked median and p90;
- cost per request, escalation rate and graded quality (with its n, overall and per task) are the hand-worked ones, beside the
  hand-worked offline estimate for the same items, and the ledger's live pool holds the hand-priced sum of the replay's calls;
- a projection over the $10 cap refuses before any call, a call that would pass the cap is refused, and a crash mid-replay
  stops the run naming the policy and the item with no answer shown; the next go sends only what has no stored response, and
  the stored calls and the finished requests equal those of a go with no crash;
- the record holds the conditions, and a stored record of another plan is refused.
"""

import contextlib
import copy
import dataclasses
import json
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any
from unittest import mock

import numpy as np

from analysis import final_headline, final_ledger, final_replay, replay_plan
from analysis import rehearsal_gates as gates_module
from analysis import rehearsal_world as rehearsal
from analysis.check_final_run import Rehearsal, rehearsed
from analysis.fake_anthropic import FakeAnthropic, message_dict
from analysis.final_record import RECORD_FILE, Record
from analysis.final_run import STEPS, Outcome, Run, Step, now_utc
from analysis.offline_half import STOPS_LABEL
from analysis.operating_points import read_section, write_section
from analysis.router_names import cascade_section
from analysis.unlock import unlock
from harness.client import PROJECT_ROOT, load_models
from routers import jev_gate, sonnet_gate

# The cascades' section keys, from the names module (nothing else may type one out).
HAIKU_JEV, HAIKU_SONNET = cascade_section("haiku", "jev"), cascade_section("haiku", "sonnet")
RULES_JEV, JEV_JEV = cascade_section("rules", "jev"), cascade_section("jev", "jev")
RULES_FREE, TRAINED_FREE = cascade_section("rules", "free"), cascade_section("trained", "free")

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(a - b) <= tolerance


# ---- the hand-worked world --------------------------------------------------------------------------------------------

POSITIONS = range(8)  # banking 0, 1; MMLU-Pro 0, 1; IFEval 0, 1; MBPP+ 0, 1 (the world's order)
IDS = [made["item_id"] for made in rehearsal.ITEMS]
TASKS = ["banking77", "banking77", "mmlu_pro", "mmlu_pro", "ifeval", "ifeval", "mbppplus", "mbppplus"]
CONFIG_OF = {("claude-haiku-5-5", "high"): "haiku", ("claude-sonnet-5-5", "high"): "sonnet",
             ("claude-opus-5-5", "low"): "opus_low", ("claude-opus-5-5", "medium"): "opus"}  # fmt: skip
CONFIGS = ("haiku", "sonnet", "opus_low", "opus")
DELAY = {"haiku": 0.03, "sonnet": 0.03, "opus_low": 0.05, "opus": 0.05, "sonnet gate": 0.04, "jev": 0.02}  # seconds
SLACK = 0.30  # a measured time may run this far over the delays it holds (a loaded machine, the client's own work)
# The cutoffs of haiku/Jev's realistic 99% and 97% points and of the Jev gate shared by the router-started cascades, as table v4
# prints them to four places (A17 item 29): a check holds the file's within 5e-5 of these.
CUTOFF = {"99%": 0.8149, "97%": 0.6334, "fixed": 0.8098}
JEV_SCORES = gates_module.TEST_SCORES  # by position: what the fake Jev gate gives each item's prompt
SONNET_SCORES = (
    90,
    40,
    70,
    20,
    50,
    49,
    85,
    80,
)  # the Sonnet gate's probabilities by position (item 6 never reaches it)
FREE_FAIL = {6}  # MBPP+ 0: the live `haiku` answer stops at the cap, so the free checks reject it
# Which items the Jev gate accepts at each cutoff by hand (score at least the cutoff, a tie counting), items 6 apart:
# 0.99, 0.60, 0.90, 0.30, 0.82, 0.75, 0.97 (free-check rejected), 0.50.
JEV_ACCEPT = {"99%": {0, 2, 4}, "97%": {0, 2, 4, 5}, "fixed": {0, 4}}
SONNET_ACCEPT = {0, 2, 4, 7}  # at 0.5: 90, 70, 50 (the tie) and 80; not 40, 20 and 49
# The live answers' correctness is the rehearsal world's stored table (haiku, sonnet, opus_low, opus), with IFEval 1's haiku
# stop as an ordinary stop here (its stored one is a marker).
RIGHT = rehearsal.CORRECT
TOKENS = {"haiku": (1000, 500), "sonnet": (1000, 500), "opus_low": (1000, 500), "opus": (1000, 500)}
# Dollars by hand at the price sheet (A1): haiku 0.10 / 0.50 per million in / out, opus 4 / 20, the Jev gate 0.042 per million in
# (1,000 tokens a call): a haiku call 0.00035, an opus call 0.014, a Jev call 0.000042.
HAIKU, OPUS, JEV = 0.00035, 0.014, 0.000042


def expected_configs(accept: set[int], start: Mapping[int, str] | None = None) -> dict[int, tuple[str, ...]]:
    """Each position's configs called: the start alone if the free checks pass and the gate accepts, else the start and `opus`."""
    found: dict[int, tuple[str, ...]] = {}
    for p in POSITIONS:
        first = (start or {}).get(p, "haiku")
        if first == "opus":
            found[p] = ("opus",)
        elif p in FREE_FAIL or p not in accept:
            found[p] = (first, "opus")
        else:
            found[p] = (first,)
    return found


def expected_dollars(configs: tuple[str, ...], p: int) -> float:
    """What a request cost by hand: each model call and the paid gate's call (made unless the start was `opus` or the free
    checks rejected the answer)."""
    model = sum(HAIKU if c == "haiku" else OPUS if c == "opus" else HAIKU for c in configs)
    gated = configs[0] != "opus" and p not in FREE_FAIL
    return model + (JEV if gated else 0.0)


def live_reply(params: Mapping[str, Any], sonnet_scores: tuple[int, ...] = SONNET_SCORES) -> dict:
    """The fake Anthropic API: an answer to an item's prompt as the config asked would give it (right or wrong as the world's
    table says), `max_tokens` where the hand-worked world says, a Sonnet gate reply by the item the gate is asked about, and
    "ready" for anything else (the liveness calls). It answers as the model it was asked."""
    content = params["messages"][0]["content"]
    model = params["model"]
    if "<answer>\n" in content:  # the Sonnet gate: its prompt holds the item's prompt and the answer
        position = next(p for p in POSITIONS if rehearsal.ITEMS[p]["prompt"] in content)
        return message_dict(json.dumps({"probability_correct": sonnet_scores[position]}), usage=(300, 20), model=model)
    positions = [p for p in POSITIONS if rehearsal.ITEMS[p]["prompt"] == content]
    if not positions:
        return message_dict("ready", usage=TOKENS["haiku"], model=model)
    p, config = positions[0], CONFIG_OF[model, params["output_config"]["effort"]]
    stop = "max_tokens" if (p in FREE_FAIL and config == "haiku") else "end_turn"
    text = rehearsal.answer_text(p, bool(RIGHT[p][CONFIGS.index(config)]))
    return message_dict(text, stop, TOKENS[config], model)


def delay_of(params: Mapping[str, Any]) -> float:
    if "<answer>\n" in params["messages"][0]["content"]:
        return DELAY["sonnet gate"]
    return DELAY[CONFIG_OF[params["model"], params["output_config"]["effort"]]]


@contextlib.contextmanager
def prepared(
    respond: Callable[[Mapping[str, Any]], dict] = live_reply, **kwargs: Any
) -> Iterator[tuple[Rehearsal, FakeAnthropic]]:
    """A rehearsal whose Anthropic client is the delaying fake, and whose fake Jev waits before it answers."""
    with rehearsed() as r:
        fake = FakeAnthropic(respond=respond, delay_s=delay_of, **kwargs)
        r.anthropic = fake
        r.world = dataclasses.replace(r.world, backends=dataclasses.replace(r.world.backends, anthropic=fake))
        r.gates.fake.delay_s = DELAY["jev"]
        yield r, fake


def named(*names: str) -> tuple[Step, ...]:
    return tuple(step for step in STEPS if step.name in names)


def through_tables(r: Rehearsal) -> None:
    """Everything before the replay: the unlock, the drift checks and the calls, grading and the tables."""
    outcome, _, _ = r.go(STEPS[: [s.name for s in STEPS].index("replay")])
    assert outcome == Outcome(), outcome


def replay_go(r: Rehearsal) -> Outcome:
    outcome, _, _ = r.go(named("unlock", "replay"))
    return outcome


def files(r: Rehearsal) -> tuple[list[dict], list[dict]]:
    out = r.world.output
    calls = [json.loads(line) for line in (out / final_replay.CALLS_FILE).read_text().splitlines()]
    requests = [json.loads(line) for line in (out / final_replay.REQUESTS_FILE).read_text().splitlines()]
    return calls, requests


def by_policy(requests: list[dict]) -> dict[str, dict[str, dict]]:
    found: dict[str, dict[str, dict]] = {}
    for request in requests:
        found.setdefault(request["policy"], {})[request["item_id"]] = request
    return found


def overlapping(intervals: list[tuple[float, float]]) -> bool:
    ordered = sorted(intervals)
    return any(later[0] < earlier[1] - 1e-6 for earlier, later in zip(ordered, ordered[1:], strict=False))


# ---- the plan -----------------------------------------------------------------------------------------------------------


def plan_cases() -> None:
    print("-- the plan: the best cascade by the rule, its points, and what falls back --")
    found = replay_plan.best_cascade_section()
    plan_text = (PROJECT_ROOT / "PLAN_v4.md").read_text(encoding="utf-8")
    check(found == HAIKU_JEV and "The best cascade** by item 24's rule is haiku/Jev" in plan_text,
          "item 24's rule on the committed dev savings gives haiku/Jev, as the plan's own sentence says")  # fmt: skip
    plan = replay_plan.make_plan(found)
    check([p.label for p in plan.policies] == ["A", "haiku/Jev realistic 99%", "haiku/Jev realistic 97%"] and not plan.skipped,
          "the policies are A and haiku/Jev at its realistic 99% and 97% points")  # fmt: skip
    check(plan.policies[0].section is None and all(p.start == "haiku" and p.gate == "jev" and p.threshold is None
                                                   for p in plan.policies[1:]),
          "A has no cascade, and the cascade is pure: it starts at haiku, ends in the Jev gate and has no router threshold")  # fmt: skip
    check(close(plan.policies[1].cutoff or 0, CUTOFF["99%"], 5e-5) and close(plan.policies[2].cutoff or 0, CUTOFF["97%"], 5e-5),
          "the cutoffs are the plan's: 0.8149 at the 99% point and 0.6334 at the 97% point")  # fmt: skip

    section = read_section(found)
    section["realistic"]["99%"] = {**section["realistic"]["99%"], "fell_back": True, "threshold": None, "cutoff": None}
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "points.json"
        write_section(found, section, path)
        made = replay_plan.make_plan(found, path)
        check([p.label for p in made.policies] == ["A", "haiku/Jev realistic 97%"] and made.skipped == ("haiku/Jev realistic 99%",),
              "in a hand-made operating-points file where the 99% point falls back to A, that point is skipped and the 97% one runs")  # fmt: skip
        section["realistic"]["97%"] = {
            **section["realistic"]["97%"],
            "fell_back": True,
            "threshold": None,
            "cutoff": None,
        }
        write_section(found, section, path)
        both = replay_plan.make_plan(found, path)
        check([p.label for p in both.policies] == ["A"] and len(both.skipped) == 2,
              "with both points fallen back the replay is A alone, and says what it skipped")  # fmt: skip
    sonnet = replay_plan.make_plan(HAIKU_SONNET)
    check([p.label for p in sonnet.policies] == ["A", "haiku/Sonnet realistic 97%"] and sonnet.skipped == ("haiku/Sonnet realistic 99%",)
          and close(sonnet.policies[1].cutoff or 0, 0.5),
          "the committed file's haiku/Sonnet falls back at its realistic 99% (skipped) and cuts at 0.5 at its 97%")  # fmt: skip
    refused = False
    try:
        replay_plan.make_plan(JEV_JEV)
    except replay_plan.ReplayPlanError as error:
        refused = "G" in str(error) and "nothing was called" in str(error)
    check(refused, "a cascade started by the Jev router is refused by the plan, before any call")

    whole = json.loads((PROJECT_ROOT / "figures" / "table_v4.json").read_text())
    # Two router-started cascades with the same gate tied at the top, and nothing above them: the rule is silent.
    tied = copy.deepcopy(whole)
    for key, cascade in tied["cascades"].items():
        for by_target in cascade["operating_points"].values():
            for row in by_target.values():
                row["savings_vs_A"] = (
                    {"share": 0.5, "per_1k": 0.5}
                    if key in (RULES_FREE, TRAINED_FREE)
                    else {"share": 0.0, "per_1k": 0.0}
                )
                row["fell_back"] = False
    wrong = {**whole, "best_cascade": {**whole["best_cascade"], "best": RULES_FREE}}
    with tempfile.TemporaryDirectory() as folder:
        for name, held, word in (("tie", tied, "tie"), ("wrong", wrong, "the rule gives")):
            path = Path(folder) / f"{name}.json"
            path.write_text(json.dumps(held))
            try:
                replay_plan.best_cascade_section(path)
                check(False, f"a table whose best cascade is a {name} is refused")
            except replay_plan.ReplayPlanError as error:
                check(word in str(error), f"a table whose best cascade is a {name} is refused: no choice is made here")
    check(replay_plan.best_cascade_section() == final_headline.best_cascade(),
          "the replay's best cascade is the headline's: one rule, called once for each")  # fmt: skip


def statistics_case() -> None:
    print("\n-- the latency figures: the first call, reconnects and a cut request apart --")

    def request(seconds: float, **flags: bool) -> dict:
        return {"latency_s": seconds, "first_call": False, "reconnect": False, "interrupted": False, **flags}

    listed = [request(s) for s in (1.0, 2.0, 3.0, 4.0, 5.0)] + [request(100.0, first_call=True),
                                                                request(50.0, reconnect=True),
                                                                request(70.0, interrupted=True)]  # fmt: skip
    found = replay_plan.latency_stats(listed)
    # By hand: five kept requests 1 to 5 give median 3; p90 sits at rank 0.9 x 4 = 3.6, so 4 + 0.6 x (5 - 4) = 4.6.
    check(close(found["median_s"], 3.0) and close(found["p90_s"], 4.6) and found["kept"] == 5,
          "the median and p90 are over the five ordinary requests: 3 s and 4.6 s")  # fmt: skip
    check(found["first_call_s"] == [100.0] and found["reconnects_s"] == [50.0] and found["interrupted_s"] == [70.0],
          "the first call, the reconnect and the cut request are listed apart, each as its own time")  # fmt: skip
    check(replay_plan.latency_stats([request(1.0, first_call=True)])["median_s"] is None,
          "a policy with nothing left once those are apart has no median")  # fmt: skip


# ---- a replay, by hand --------------------------------------------------------------------------------------------------


def hand_requests(order: list[str]) -> list[dict]:
    """Each request's expected configs, seconds and flags, in the order the replay sends them: items in the draw's order, each
    item's three policies together (A, the 99% point, the 97% point). The first Anthropic call and the first Jev call are the
    first calls; the Jev call after the server's hang-up (the 6th of the replay) is the reconnect."""
    table = {"99%": expected_configs(JEV_ACCEPT["99%"]), "97%": expected_configs(JEV_ACCEPT["97%"])}
    found: list[dict] = []
    jev_calls = 0
    seen_anthropic = seen_jev = False
    for item_id in order:
        p = IDS.index(item_id)
        for name in ("A", "99%", "97%"):
            configs = ("opus",) if name == "A" else table[name][p]
            gated = name != "A" and p not in FREE_FAIL
            jev_calls += gated
            first = (not seen_anthropic) or (gated and not seen_jev)
            seen_anthropic, seen_jev = True, seen_jev or gated
            found.append({"name": name, "item_id": item_id, "p": p, "configs": configs, "gated": gated, "first_call": first,
                          "seconds": sum(DELAY[c] for c in configs) + (DELAY["jev"] if gated else 0),
                          "reconnect": gated and jev_calls == RECONNECT_AT + 1})  # fmt: skip
    return found


def lookup(requests: list[dict]) -> dict[str, dict[str, dict]]:
    found: dict[str, dict[str, dict]] = {}
    for request in requests:
        found.setdefault(request["name"], {})[request["item_id"]] = request
    return found


RECONNECT_AT = 5  # the fake Jev hangs up after the 5th gate call of the replay, so the 6th reconnects


def main_run() -> None:
    print("\n-- a replay on the made-up world: Jev-gated haiku/Jev at its two points, beside A --")
    with prepared() as (r, fake):
        through_tables(r)
        order = [c["detail"]["item_ids"] for c in r.record["gos"][0]["unlock"]["checks"] if c["name"] == "replay_draw"][
            0
        ]
        before = len(fake.calls)
        jev_before = r.gates.fake.calls
        r.gates.fake.hang_up_after = {jev_before + RECONNECT_AT}
        outcome = replay_go(r)
        check(outcome == Outcome(), f"the replay step runs after the tables {outcome}")
        calls, requests = files(r)
        policy_keys = {"A": "A", "99%": f"{HAIKU_JEV}/realistic/99%", "97%": f"{HAIKU_JEV}/realistic/97%"}
        found = by_policy(requests)
        hand = lookup(hand_requests(order))
        check(sorted(found) == sorted(policy_keys.values()) and all(len(v) == 8 for v in found.values()),
              "each of the three policies ran on all eight items, one summary each")  # fmt: skip
        check([(q["policy"], q["item_id"]) for q in requests]
              == [(policy_keys[n], i) for i in order for n in ("A", "99%", "97%")],
              "the requests went in the draw's order, an item's three policies together")  # fmt: skip
        ok = all(tuple(found[policy_keys[n]][i]["configs"]) == hand[n][i]["configs"] for n in hand for i in order)
        check(ok, "the configs called on each item are the hand-worked decision's, for A and for both points")
        free = {(q["policy"], q["item_id"]): q["free_passed"] for q in requests}
        check(all(free[policy_keys[n], IDS[p]] is False for n in ("99%", "97%") for p in FREE_FAIL)
              and all(free[policy_keys[n], IDS[p]] for n in ("99%", "97%") for p in POSITIONS if p not in FREE_FAIL)
              and all(free[policy_keys["A"], i] is None for i in order),
              "the free checks ran on every cascade answer and rejected only the capped one; A has none")  # fmt: skip
        check(all(q["gate_score"] is None for q in requests if q["free_passed"] is not True), "no gate score for an answer a free check rejected or for A")  # fmt: skip
        scores = {(q["policy"], q["item_id"]): q["gate_score"] for q in requests if q["free_passed"]}
        check(all(close(scores[policy_keys["99%"], IDS[p]], JEV_SCORES[p], 1e-3) for p in POSITIONS if p not in FREE_FAIL),
              "each answer's Jev gate score is the scripted one (the gate's own parser read the stored response)")  # fmt: skip

        sent = fake.calls[before:]
        models = [c["model"] for c in sent]
        check(len(sent) == 8 + 8 + 5 + 8 + 4 and models.count("claude-opus-5-5") == 8 + 5 + 4,
              "the fake received 33 requests: A's 8 opus, each point's 8 haiku and the 9 escalations to opus")  # fmt: skip
        check(all(c["output_config"]["effort"] in ("high", "medium") for c in sent)
              and all(set(c) == {"model", "max_tokens", "output_config", "messages"} for c in sent),
              "each request is the collection harness's own: the pinned model, effort, max_tokens, one user message, no extras")  # fmt: skip
        check(
            len(r.gates.fake.bodies) - jev_before == 14,
            "the fake Jev received 14 gate calls: seven items under two points",
        )
        check(not overlapping(fake.intervals + r.gates.fake.intervals),
              "calls went one at a time: no two calls to the fakes overlap in time")  # fmt: skip
        with mock.patch.object(r.gates.fake, "delay_s", 0.05):
            clash = FakeAnthropic(delay_s=0.05)
            pair = [threading.Thread(target=lambda: clash.messages.create(**final_probe())) for _ in range(2)]
            for thread in pair:
                thread.start()
            for thread in pair:
                thread.join()
        check(overlapping(clash.intervals), "the overlap test sees two calls made at once, so its all-clear is earned")

        # Latency: each request's time is the delays the fakes injected, within the margin; the figures match.
        late = [
            (n, i)
            for n in hand
            for i in order
            if not within(hand[n][i]["seconds"], found[policy_keys[n]][i]["latency_s"])
        ]
        check(not late, f"each request's time is its injected delays, within {SLACK} s over {late[:2]}")
        check(not within(0.10, 0.10 + SLACK + 0.05) and not within(0.10, 0.05) and within(0.10, 0.12),
              "the time check can fail: a request too slow or too fast for its delays is outside it, one a little over is inside")  # fmt: skip
        check(all(found[policy_keys[n]][i]["first_call"] == hand[n][i]["first_call"] for n in hand for i in order),
              "the first call is the first request to reach each service, and no other")  # fmt: skip
        check(all(found[policy_keys[n]][i]["reconnect"] == hand[n][i]["reconnect"] for n in hand for i in order)
              and sum(q["reconnect"] for q in requests) == 1,
              "the one request whose Jev call came after the server's hang-up is the reconnect")  # fmt: skip
        detail = r.record["gos"][-1]["steps"][-1]["detail"]
        live = {k: v["live"] for k, v in detail["results"].items()}
        for n, key in policy_keys.items():
            kept = [
                hand[n][i]["seconds"] for i in order if not hand[n][i]["first_call"] and not hand[n][i]["reconnect"]
            ]
            median, p90 = float(np.median(kept)), float(np.percentile(kept, 90))
            check(median - 0.01 <= live[key]["median_s"] <= median + SLACK and p90 - 0.01 <= live[key]["p90_s"] <= p90 + SLACK
                  and live[key]["kept"] == len(kept),
                  f"{n}: the median and p90 over {len(kept)} requests match the injected delays ({median:.2f} s, {p90:.2f} s)")  # fmt: skip
        check(len(live[policy_keys["A"]]["first_call_s"]) == 1 and len(live[policy_keys["99%"]]["first_call_s"]) == 1
              and live[policy_keys["97%"]]["first_call_s"] == [] and len(live[policy_keys["97%"]]["reconnects_s"]) == 1,
              "the first calls and the reconnect are reported apart, per policy")  # fmt: skip

        # Cost, escalation and graded quality, by hand.
        cost = {n: sum(expected_dollars(hand[n][i]["configs"], hand[n][i]["p"]) for i in order) / 8 for n in hand}
        check(all(close(live[policy_keys[n]]["cost_per_request"], cost[n], 1e-9) for n in hand)
              and close(cost["A"], 0.014) and close(cost["99%"], 0.00913675) and close(cost["97%"], 0.00738675),
              "cost per request: A $0.014, the 99% point $0.00913675, the 97% point $0.00738675, from the billed tokens")  # fmt: skip
        check(close(live[policy_keys["A"]]["escalation_rate"], 0) and close(live[policy_keys["99%"]]["escalation_rate"], 5 / 8)
              and close(live[policy_keys["97%"]]["escalation_rate"], 4 / 8),
              "escalation rate: none for A, 5 of 8 at the 99% point, 4 of 8 at the 97% point")  # fmt: skip
        quality = {key: v["quality"] for key, v in live.items()}
        check(close(quality[policy_keys["A"]]["value"], 7 / 8) and close(quality[policy_keys["99%"]]["value"], 7 / 8)
              and close(quality[policy_keys["97%"]]["value"], 6 / 8) and all(q["n"] == 8 for q in quality.values()),
              "graded quality, each over its n of 8: A 7 of 8, the 99% point 7 of 8, the 97% point 6 of 8 (IFEval 1 kept its wrong haiku answer)")  # fmt: skip
        by_task = {key: {t: (v["value"], v["n"]) for t, v in live[key]["quality_by_task"].items()} for key in live}
        check(by_task[policy_keys["97%"]] == {"banking77": (1.0, 2), "ifeval": (0.5, 2), "mbppplus": (1.0, 2), "mmlu_pro": (0.5, 2)}
              and by_task[policy_keys["A"]]["mmlu_pro"] == (0.5, 2),
              "quality per task carries each task's n (2 here, 25 in the real replay)")  # fmt: skip
        check(detail["per_task"] == {"banking77": 2, "ifeval": 2, "mbppplus": 2, "mmlu_pro": 2},
              "the record counts the items per task")  # fmt: skip

        # The offline estimates for the same items, by hand (see the module's notes below).
        offline = {k: v["offline"] for k, v in detail["results"].items()}
        a, c99, c97 = (offline[policy_keys[n]] for n in ("A", "99%", "97%"))
        check(close(a["cost_per_request"], 0.019) and close(a["quality"], 7 / 8) and a["escalation_rate"] == 0
              and a["n"] == 8 and close(a["latency_median_s"], 3.95) and close(a["latency_p90_s"], 14.0),
              "A's offline estimate: $0.019 a request (six at $0.016, two MBPP+ at $0.028), 7 of 8, median 3.95 s and p90 14 s from the pilot")  # fmt: skip
        check(close(c99["cost_per_request"], 0.00994925, 1e-9) and close(c99["escalation_rate"], 0.5) and close(c99["quality"], 7 / 8)
              and close(c99["latency_median_s"], 4.663045, 2e-4) and close(c99["latency_p90_s"], 12.262518, 2e-4),
              "the cascade's offline estimate: $0.00994925 a request, 4 of 8 escalated, 7 of 8, median 4.663 s and p90 12.263 s")  # fmt: skip
        check(c97 == c99, "at both points the same four items escalate offline (IFEval 1's stored haiku answer fails a free check), so the two estimates are equal")  # fmt: skip

        # The ledger counts the replay's calls at the hand-priced sum, in the live pool.
        spent = final_ledger.recorded_spend(r.world.results, r.world.output).live
        by_hand = (
            8 * OPUS + 16 * HAIKU + 9 * OPUS + 14 * JEV + HAIKU + OPUS
        )  # A, haiku calls, escalations, Jev, liveness
        check(close(spent, by_hand, 1e-9) and spent < final_ledger.CAPS[final_ledger.LIVE],
              f"the live pool holds the replay's and the liveness calls' hand-priced ${by_hand:.6f}, under its cap ({spent:.6f})")  # fmt: skip
        conditions = detail["conditions"]
        check({"machine", "network", "go_started", "go_ended"} <= set(conditions) and conditions["machine"] != "",
              "the conditions are in the record: the machine, the network and the go's start and end")  # fmt: skip
        check(r.record["gos"][-1]["steps"][-1]["requests"] == {"sent": 24, "stored": 0}, "the step records 24 requests sent and none stored")  # fmt: skip

        # A resume with everything stored sends nothing and reports the same figures.
        sent_before = len(fake.calls)
        again = replay_go(r)
        _, requests_again = files(r)
        check(again == Outcome() and len(fake.calls) == sent_before and len(requests_again) == 24
              and r.record["gos"][-1]["steps"][-1]["requests"] == {"sent": 0, "stored": 24},
              "a next go with every request stored sends nothing")  # fmt: skip


def within(expected: float, measured: float) -> bool:
    """A measured time is the delays injected into it, a little over (sleeping never returns early, but the clock may differ by
    a hair): at most SLACK over."""
    return expected - 0.01 <= measured <= expected + SLACK


def final_probe() -> dict:
    return {"model": "claude-haiku-5-5", "max_tokens": 10, "messages": [{"role": "user", "content": "x"}]}


# ---- a crash, and the next go ---------------------------------------------------------------------------------------


def draw_of(r: Rehearsal) -> list[str]:
    return [c["detail"]["item_ids"] for c in r.record["gos"][0]["unlock"]["checks"] if c["name"] == "replay_draw"][0]


def crash_case() -> None:
    print("\n-- a crash mid-replay stops the run; the next go sends only what has no stored response --")
    with prepared() as (r, fake):
        through_tables(r)
        order = draw_of(r)
        hand = hand_requests(order)
        # The crash is on the first escalation to opus of a cascade request (which holds its haiku answer and its gate call).
        calls_before = 0
        for index, request in enumerate(hand):
            if request["name"] != "A" and len(request["configs"]) > 1:
                break
            calls_before += len(request["configs"])
        stored_gates = sum(q["gated"] for q in hand[: index + 1])
        crashed = hand[index]
        base = len(fake.calls)
        fake.message_errors = {base + calls_before + 2: RuntimeError(f"boom {rehearsal.GRADE_MARKER}")}
        outcome = replay_go(r)
        said = (r.world.output / "final_run.log").read_text()
        key = {"99%": f"{HAIKU_JEV}/realistic/99%", "97%": f"{HAIKU_JEV}/realistic/97%"}[crashed["name"]]
        wanted = Outcome("replay", "RuntimeError", f"{key} {crashed['item_id']}")
        check(
            outcome == wanted,
            f"the run stops at the replay, naming the policy and the item and the exception {outcome}",
        )
        # the tables step (#88) says every stop by config, and the made-up world's one marker is a stop reason, so it is said there
        rest = "\n".join(line for line in said.splitlines() if STOPS_LABEL not in line)
        check("boom" not in said and rehearsal.GRADE_MARKER not in rest, "the error's message is shown nowhere")
        calls_one, requests_one = files(r)
        check(len(requests_one) == index and len(calls_one) == calls_before + 1 + stored_gates,
              f"{index} requests are finished and stored, with the cut one's haiku answer and gate call beside their calls")  # fmt: skip
        stored_before = {(c["policy"], c["item_id"], c["role"]): c["response"] for c in calls_one}
        fake.message_errors = {}
        resumed = replay_go(r)
        calls_two, requests_two = files(r)
        check(resumed == Outcome(), f"the next go runs the replay to its end {resumed}")
        check(
            len(fake.calls) - base == 33 + 1,
            "in all the fake received 33 requests and the one that failed, sent again once",
        )
        keys = [(c["policy"], c["item_id"], c["role"]) for c in calls_two]
        check(len(keys) == len(set(keys)) and calls_two[: len(calls_one)] == calls_one,
              "no stored call was drawn again or changed: each (policy, item, role) is stored once, the earlier ones as they were")  # fmt: skip
        check(all(stored_before[k] == next(c["response"] for c in calls_two if (c["policy"], c["item_id"], c["role"]) == k)
                  for k in stored_before), "the answers stored before the crash are the ones the finished replay used")  # fmt: skip
        cut = [q for q in requests_two if q["interrupted"]]
        check(len(cut) == 1 and (cut[0]["policy"], cut[0]["item_id"]) == (key, crashed["item_id"]) and cut[0]["configs"] == ["haiku", "opus"],
              "the request the crash cut is flagged interrupted, it is the only one, and it finished from its stored calls: haiku, then opus")  # fmt: skip
        step = r.record["gos"][-1]["steps"][-1]
        check(len(step["detail"]["results"][key]["live"]["interrupted_s"]) == 1
              and step["requests"] == {"sent": 24 - index, "stored": index},
              f"its time is reported apart from the median, and the step says {24 - index} sent and {index} stored")  # fmt: skip
        with prepared() as (clean, _):
            through_tables(clean)
            replay_go(clean)
            clean_calls, clean_requests = files(clean)
            fields = ("policy", "item_id", "configs", "cost_usd")
            check([{k: q[k] for k in fields} for q in requests_two] == [{k: q[k] for k in fields} for q in clean_requests]
                  and len(calls_two) == len(clean_calls),
                  "the finished requests, their configs and costs equal those of a go with no crash")  # fmt: skip


# ---- the cap, and a stored record of another plan -------------------------------------------------------------------------


def refusal_cases() -> None:
    print("\n-- the cap and the stored state --")
    with prepared() as (r, fake):
        through_tables(r)
        base = len(fake.calls)
        first = draw_of(r)[0]
        with mock.patch.object(final_replay, "SAFETY", 1e6):
            outcome = replay_go(r)
        check(outcome == Outcome("replay", "LedgerRefusal", None) and len(fake.calls) == base,
              "a projection over the $10 cap refuses up front, with nothing sent")  # fmt: skip
        check("over Stage 8's cap" in (r.world.output / "final_run.log").read_text(), "and the refusal says which cap")
        # Recorded so far: the liveness calls, $0.01435. Opus's worst case is its whole 16,000 output tokens, $0.32.
        with (
            mock.patch.dict(final_ledger.CAPS, {final_ledger.LIVE: 0.30}),
            mock.patch.object(final_replay, "project", lambda run: 0.0),
        ):
            outcome = replay_go(r)
        check(outcome == Outcome("replay", "LedgerRefusal", f"A {first}") and len(fake.calls) == base
              and not (r.world.output / final_replay.CALLS_FILE).exists(),
              "a call whose worst case would pass the cap is refused before it is sent, though the projection let it through, and stores nothing")  # fmt: skip
        stray = {"kind": "request", "policy": f"{TRAINED_FREE}/realistic/99%", "item_id": IDS[0]}
        (r.world.output / final_replay.REQUESTS_FILE).write_text(json.dumps(stray) + "\n")
        outcome = replay_go(r)
        check(outcome.stopped_at == "replay" and outcome.error == "ReplayRefusal" and len(fake.calls) == base,
              "a stored request of another policy is refused before any call")  # fmt: skip
    with prepared(lambda params: message_dict("ready", usage=(10, 5), model="claude-sonnet-5-5")) as (r, fake):
        # The liveness step would refuse this fake, so the steps before the replay run without it.
        names = [s.name for s in STEPS]
        outcome, _, _ = r.go(tuple(s for s in STEPS[: names.index("replay")] if s.name != "liveness"))
        check(outcome == Outcome(), "(setup) the steps before the replay run without the liveness calls")
        outcome = replay_go(r)
        check(outcome.error == "ReplayRefusal" and "answered as claude-sonnet-5-5" in (r.world.output / "final_run.log").read_text(),
              "a call answered by a model other than the one asked refuses, naming the models")  # fmt: skip
        calls_before = len(fake.calls)
        outcome = replay_go(r)
        check(outcome.error == "ReplayRefusal" and len(fake.calls) == calls_before,
              "the next go refuses at the same stored call: a refusal that stopped one go stops the next, with nothing sent")  # fmt: skip


# ---- a Sonnet-gated cascade, and a cascade started by a router --------------------------------------------------------------


def estimate_stub(plan: replay_plan.Plan) -> dict[str, replay_plan.Estimate]:
    return {p.key: replay_plan.Estimate(8, 0.0, 0.0, 0.0, {}, 0.0, 0.0) for p in plan.policies}


def sonnet_case() -> None:
    print("\n-- a Sonnet-gated cascade, through the standard API; its 99% point falls back to A and is skipped --")
    with prepared() as (r, fake):
        through_tables(r)
        base = len(fake.calls)
        plan = replay_plan.make_plan(HAIKU_SONNET)
        with (mock.patch.object(final_replay, "plan_of", lambda: plan),
              mock.patch.object(final_replay, "offline_numbers", lambda run, p, items, requests: estimate_stub(p))):  # fmt: skip
            outcome = replay_go(r)
        check(outcome == Outcome(), f"the replay runs with the Sonnet gate {outcome}")
        _, requests = files(r)
        found = by_policy(requests)
        key = f"{HAIKU_SONNET}/realistic/97%"
        check(sorted(found) == sorted(["A", key]) and "skipped" not in found,
              "the policies are A and the one point that routes: the 99% point, which falls back to A, is not run")  # fmt: skip
        hand = expected_configs(SONNET_ACCEPT)
        check(all(tuple(found[key][IDS[p]]["configs"]) == hand[p] for p in POSITIONS),
              "the configs called follow the decision at cutoff 0.5: a score of 0.50 is kept (a tie counts), 0.49 escalates")  # fmt: skip
        sent = fake.calls[base:]
        gate = [c for c in sent if c["model"] == "claude-sonnet-5-5"]
        settings = sonnet_gate.load_settings()
        check(len(gate) == 7 and all(c["output_config"]["effort"] == settings.effort and "temperature" not in c
                                     and "thinking" not in c for c in gate),
              "the Sonnet gate was called for the seven answers that passed the free checks, as its own request builds it, on the standard API")  # fmt: skip
        check(close(sum(q["gate_score"] or 0 for q in requests if q["policy"] == key and q["gate_score"] is not None),
                    sum(SONNET_SCORES[p] for p in POSITIONS if p not in FREE_FAIL) / 100, 1e-9),
              "each gate score is the Sonnet gate's probability over 100")  # fmt: skip
        spent = final_ledger.live_spend(r.world.output)
        sonnet = (300 * 2 + 20 * 10) / 1e6
        by_hand = (
            (8 * OPUS + 8 * HAIKU + 4 * OPUS) + 7 * sonnet + HAIKU + OPUS
        )  # A, haiku calls, 4 escalations, gates, liveness
        check(
            close(spent, by_hand, 1e-9), f"the live pool holds the Sonnet gate's calls at list price too (${spent:.6f})"
        )


def router_case() -> None:
    print("\n-- a cascade started by a router: the router's pick starts each item --")
    plan = replay_plan.make_plan(RULES_JEV)
    check(all(0 < (p.threshold or 0) <= 1 and close(p.cutoff or 0, CUTOFF["fixed"], 5e-5) for p in plan.policies[1:]),
          "E's cascade with the Jev gate holds a router threshold and the fixed Jev cutoff 0.8098 at both its points")  # fmt: skip
    # Router scores for haiku, sonnet, opus_low, by position: 1.0 picks haiku; 0 then 1 picks sonnet; all 0 picks opus.
    picks = {1: {"haiku": 0.0, "sonnet": 1.0, "opus_low": 1.0}, 2: {"haiku": 0.0, "sonnet": 0.0, "opus_low": 0.0}}
    by_prompt = {
        rehearsal.ITEMS[p]["prompt"]: picks.get(p, {"haiku": 1.0, "sonnet": 1.0, "opus_low": 1.0}) for p in POSITIONS
    }
    scorer = lambda request: by_prompt[request.prompt]  # noqa: E731
    with prepared() as (r, fake):
        through_tables(r)
        base, jev_before = len(fake.calls), r.gates.fake.calls
        with (mock.patch.object(final_replay, "plan_of", lambda: plan),
              mock.patch.object(final_replay, "scorer_of", lambda run, p: scorer)):  # fmt: skip
            outcome = replay_go(r)
        check(outcome == Outcome(), f"the replay runs for a router-started cascade {outcome}")
        _, requests = files(r)
        key = plan.policies[1].key
        found = by_policy(requests)[key]
        # By hand at the fixed cutoff 0.8098 (scores .99 .60 .90 .30 .82 .75 .97 .50): item 1 starts at sonnet and its .60
        # escalates; item 2 starts at opus and is gated by nothing; item 6's haiku answer fails the free checks.
        hand = {0: ("haiku",), 1: ("sonnet", "opus"), 2: ("opus",), 3: ("haiku", "opus"), 4: ("haiku",), 5: ("haiku", "opus"),
                6: ("haiku", "opus"), 7: ("haiku", "opus")}  # fmt: skip
        check(all(tuple(found[IDS[p]]["configs"]) == hand[p] for p in POSITIONS),
              "the router's pick starts each item (haiku, sonnet, or opus alone with no gate) and the decision follows")  # fmt: skip
        sent = fake.calls[base:]
        check(sum(c["model"] == "claude-opus-5-5" for c in sent) == 8 + 6 + 6 and r.gates.fake.calls - jev_before == 12,
              "opus ran for A's eight items and six a point (one sent straight to it, five escalated), and the gate for six a point, none for the item sent to opus")  # fmt: skip


def projection_case() -> None:
    print("\n-- the projection: each pending request's stored cost, doubled, with its dearest escalation --")
    with prepared() as (r, fake):
        state = Run(r.world, Record(r.world.output / RECORD_FILE), now_utc)
        state.handle = unlock(r.world)
        # By hand. The world's stored answers cost, per request, haiku $0.00035 (MBPP+ $0.0006) and opus $0.016 (MBPP+ $0.028).
        # A: 2 x (6 x 0.016 + 2 x 0.028) = 0.304. A haiku/Jev point: 2 x (6 x (0.00035 + 0.016) + 2 x (0.0006 + 0.028) + 8 x the
        # Jev gate's bound), the bound being its 29,597-byte budget and 512 tokens at $0.042 a million: 8 x 0.001264578. Two
        # points: 0.304 + 2 x 0.330833248 = 0.965666496.
        check(close(final_replay.project(state), 0.965666496, 1e-9),
              "the projection for A and the two haiku/Jev points is the hand-worked $0.965666496, before any call")  # fmt: skip
        # The Sonnet gate: a call reads the prompt, the start's answer and 2,000 tokens of its own template, and answers in at
        # most 500 at $2 and $10 a million: 0.012 (MBPP+ 0.013). One point: 0.304 + 2 x (6 x 0.02835 + 2 x 0.0416) = 0.8106.
        with mock.patch.object(final_replay, "plan_of", lambda: replay_plan.make_plan(HAIKU_SONNET)):
            check(
                close(final_replay.project(state), 0.8106, 1e-9),
                "and for A and the Sonnet-gated point it is the hand-worked $0.8106",
            )
        pending = final_replay.project(state)
        outcome, _, _ = r.go(STEPS[:2])
        check(outcome == Outcome() and len(fake.calls) == 2 and close(pending, 0.965666496, 1e-9),
              "(setup) the liveness step, which projects the whole run first, passes with the replay's bound under the cap")  # fmt: skip
    with prepared() as (r, fake):
        with mock.patch.object(final_replay, "SAFETY", 1e6):
            outcome, _, _ = r.go(STEPS[:2])
        check(outcome == Outcome("liveness", "LedgerRefusal", None) and fake.calls == [],
              "a replay projected over the $10 cap stops the run at the liveness step, before the first call of any step")  # fmt: skip


def stray_case() -> None:
    print("\n-- a plan the replay can't run --")
    with prepared() as (r, fake):
        through_tables(r)
        base = len(fake.calls)
        plan_error = replay_plan.ReplayPlanError("made up: a router with no live path")
        with mock.patch.object(final_replay, "make_plan", side_effect=plan_error):
            outcome = replay_go(r)
        check(outcome.error == "ReplayRefusal" and len(fake.calls) == base,
              "a plan the replay refuses stops the step before any call, as a refusal that is shown")  # fmt: skip


def settings_case() -> None:
    print("\n-- the gates' own settings are what the replay uses --")
    check(jev_gate.read_settings().model == "jev-1.14.0" and load_models()["configs"]["opus"]["effort"] == "medium",
          "the Jev gate is the pinned jev-1.14.0 and A's opus is at medium effort")  # fmt: skip
    check(final_ledger.LIVE_FILES == ("liveness.jsonl", final_replay.CALLS_FILE) and final_replay.SAFETY >= 1,
          "the replay's calls are in the ledger's live files and the projection errs high")  # fmt: skip


def main() -> int:
    plan_cases()
    statistics_case()
    main_run()
    crash_case()
    refusal_cases()
    sonnet_case()
    router_case()
    projection_case()
    stray_case()
    settings_case()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
