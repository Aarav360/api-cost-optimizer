"""The rehearsal's Sonnet gate stand-ins: a made-up Stage 7 sample and a scripted fake Anthropic client (PLAN_v4.md Amendment A19,
items 2, 7 and 13; A15 item 17; A17 items 14 and 26; issue #77, #86).

Not a check: a module `analysis/check_final_run.py`'s rehearsal and `analysis/check_final_sonnet.py` import. It gives a made-up
results folder what the Sonnet gate's drift check reads from Stage 7, and a model that answers as told:

- **Stage 7's side**: `sonnet_gate_train.jsonl` (the batch's verdicts, pass 1: `BATCH`, ten made-up `haiku` train answers) and
  `sonnet_gate_latency.jsonl` (the standard-API calls of the latency sample, pass 2: `STANDARD`, the first six of them), each
  record built as the collector builds one (the frozen request and its sha256, the response, `RECORD_FIELDS`/`LATENCY_FIELDS` in
  order). Hand-worked: D6 is 0.02 for `haiku` (every standard score is 2 points from the batch's, above for even items and below
  for odd) and the shift 0. The train answers are `analysis/rehearsal_gates.py`'s.
- **The model** (`ScriptedSonnet`): `respond(params)` answers each request from `scores`, the whole-number score it should come
  out as, by the request's prompt; a prompt in `modes` is answered with a refusal, a cap stop or text that is no JSON instead. It
  answers as the pin unless `model` says otherwise. `responder(fallback)` is what the rehearsal's one fake Anthropic client
  takes: the Sonnet model's requests go to `respond`, any other model's to `fallback` (the liveness calls'). The resend
  scenarios `RESENDS` are what the drift check's resend scores are.
- **The test side**: `TEST_SCORES`, the score of each of the rehearsal world's eight test items, whatever config answered.

Billed tokens are 1,000 in and 50 out for a verdict (a cap stop is billed 16,000 out), so each cost is worked by hand in
`analysis/check_final_sonnet.py`. Nothing here touches `results/` (a made-up folder is passed in), the real store or the API.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from analysis import collect_sonnet_gate, rehearsal_gates
from analysis import rehearsal_world as rehearsal
from analysis.collect_jev import append_record
from analysis.fake_anthropic import message_dict
from routers import sonnet_gate

PIN = "claude-sonnet-5-5"
USAGE = (1000, 50)  # billed tokens of a verdict, in and out
CAP_USAGE = (1000, 16000)  # a cap stop is billed the whole output cap
CONFIG = "haiku"
# Stage 7's batch scores of ten train answers (whole numbers over 100), and the latency sample's standard-API scores of the first
# six: 2 points from the batch's, above for even items and below for odd. The values sit either side of the frozen cutoffs
# (0.77, 0.5, 0.3), so a resend 3 points off flips decisions that are worked by hand in the check.
BATCH = (95, 79, 74, 31, 49, 52, 60, 20, 88, 10)
SAMPLE = (0, 1, 2, 3, 4, 5)
STANDARD = tuple(BATCH[k] + (2 if k % 2 == 0 else -2) for k in SAMPLE)
D6, D8_LIMIT = 0.02, 0.06  # by hand: every standard score is 0.02 from the batch's; 3 x 0.02
# What the drift check's resend scores are, by scenario: the batch score moved by a fixed number of points, item by item.
RESENDS = {
    "quiet": tuple(3 if k % 2 == 0 else -3 for k in SAMPLE),  # D8 0.03, shift 0: within both limits
    "loud": tuple(-8 if k % 2 == 0 else 8 for k in SAMPLE),  # D8 0.08 > 3 x 0.02, shift 0
    "uniform": tuple(3 for _ in SAMPLE),  # D8 0.03 (well inside 3 x D6), shift +0.03 > 0.02: the shift rule alone
}
# The rehearsal world's eight test items' scores (banking 0, 1; MMLU-Pro 0, 1; IFEval 0, 1; MBPP+ 0, 1).
TEST_SCORES = (96, 55, 88, 35, 80, 72, 93, 45)
MODES = ("REFUSE", "CAP", "GARBAGE")


def verdict_message(score: int, model: str = PIN) -> dict:
    return message_dict(json.dumps({sonnet_gate.SCORE_KEY: score}), "end_turn", USAGE, model)


def reply(score: int | None, mode: str | None, model: str) -> dict:
    """The model's response: a score, or what a refusal, a cap stop or an unparseable text looks like."""
    if mode == "REFUSE":
        return message_dict("", "refusal", USAGE, model)
    if mode == "CAP":
        return message_dict("", "max_tokens", CAP_USAGE, model)
    if mode == "GARBAGE" or score is None:
        return message_dict("probably fine", "end_turn", USAGE, model)
    return verdict_message(score, model)


class ScriptedSonnet:
    """A model that gives each prompt its scripted score, as `model`, and logs the prompts it was asked about."""

    def __init__(self) -> None:
        self.scores: dict[str, int] = {}  # prompt -> score
        self.modes: dict[str, str] = {}  # prompt -> one of MODES
        self.model = PIN
        self.asked: list[str] = []  # the prompt of each request answered, in order
        self.unexpected: list[str] = []

    def respond(self, params: Mapping[str, Any]) -> dict:
        content = params["messages"][0]["content"]
        found = [p for p in {*self.scores, *self.modes} if p in content]
        if len(found) != 1:
            self.unexpected.append(f"{len(found)} scripted prompts in a request")
            return reply(None, None, self.model)
        self.asked.append(found[0])
        return reply(self.scores.get(found[0]), self.modes.get(found[0]), self.model)

    def responder(self, fallback: Callable[[Mapping[str, Any]], dict]) -> Callable[[Mapping[str, Any]], dict]:
        """What the rehearsal's fake client takes: this model for the Sonnet gate's requests, `fallback` for any other model's."""

        def respond(params: Mapping[str, Any]) -> dict:
            return self.respond(params) if params["model"] == PIN else fallback(params)

        return respond

    def resend(self, scenario: str) -> None:
        """Set the train prompts' scores to the scenario's resend: the batch score moved by the scenario's points."""
        self.move(RESENDS[scenario])

    def move(self, points: Sequence[int]) -> None:
        """Set the sample's train prompts' scores to the batch score moved by `points`, item by item."""
        for k, by in zip(SAMPLE, points, strict=True):
            self.scores[rehearsal_gates.train_prompt(k)] = BATCH[k] + by

    def test_scores(self) -> None:
        for made, score in zip(rehearsal.ITEMS, TEST_SCORES, strict=True):
            self.scores[made["prompt"]] = score


def stored(k: int, number: int, score: int, settings: sonnet_gate.SonnetSettings) -> dict:
    """One stored Stage 7 verdict of train item `k`: the batch's (pass 1) or the latency sample's (pass 2), as the collector
    stores it, with the frozen request's hash."""
    item = rehearsal_gates.train_input(k)
    [job] = collect_sonnet_gate.build_jobs(
        [item], [{"item_id": item.item_id, "config": CONFIG, "passed": True}], settings
    )
    message = verdict_message(score)
    result = {"type": "succeeded", "message": message}
    record = collect_sonnet_gate.build_record(job, "train", number, result, sonnet_gate.parse_response(message),
                                              "msgbatch_made_up" if number == 1 else None,
                                              lambda: "2026-11-10T12:00:00-08:00")  # fmt: skip
    if number == collect_sonnet_gate.STANDARD_PASS:
        record |= {
            "api": collect_sonnet_gate.STANDARD,
            "seconds": 2.0 + k,
            "first_call": k == 0,
            "request_id": f"req_{k:032x}",
        }
    return record


def write_train(
    results: Path, settings: sonnet_gate.SonnetSettings | None = None, standard: Sequence[int] = STANDARD
) -> None:
    """Stage 7's made-up Sonnet files under `results/gates/`: the batch's ten verdicts and the latency sample's six calls
    (`standard`'s scores)."""
    settings = settings or sonnet_gate.load_settings()
    for k, score in enumerate(BATCH):
        append_record(results / "gates" / "sonnet_gate_train.jsonl", stored(k, 1, score, settings))
    for k, score in zip(SAMPLE, standard, strict=True):
        append_record(results / "gates" / "sonnet_gate_latency.jsonl", stored(k, 2, score, settings))


def stand_in(results: Path, scenario: str = "quiet") -> ScriptedSonnet:
    """Stage 7's made-up files written under `results/gates/`, and a model that answers the `scenario`'s resends and the test
    items' scores. The rehearsal puts it behind its fake client (`responder`)."""
    write_train(results)
    model = ScriptedSonnet()
    for k, score in enumerate(BATCH):
        model.scores[rehearsal_gates.train_prompt(k)] = score
    model.resend(scenario)
    model.test_scores()
    return model


def prompt_of(params: Mapping[str, Any]) -> str:
    """Which scripted train or test prompt a request was about (for a check reading the fake's log)."""
    content = params["messages"][0]["content"]
    found = [p for p in [*(rehearsal_gates.train_prompt(k) for k in range(len(BATCH))), *(m["prompt"] for m in rehearsal.ITEMS)]
             if p in content]  # fmt: skip
    return found[0] if len(found) == 1 else ""
