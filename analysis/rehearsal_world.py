"""The rehearsal's made-up world: test items that the real graders can grade (PLAN_v4.md Amendment A19, items 2, 5, 13).

`analysis/made_up_world.py` builds the made-up world that the unlock's check uses, whose items no grader reads. The
rehearsal grades, so this module gives that builder different content: two test items for each of the four tasks, each
with its four configs' stored results, every item's expected answer one the task's own grader reads and every answer
written so that the grader's status is known:

- Banking77 and MMLU-Pro items are structured (their answer is JSON, as the collection run asked for), IFEval's is free
  text under one instruction (no commas), and MBPP+'s is code in a block, graded by running it, in the sandbox the
  world's backends name;
- a marker sits in an MBPP+ item's tests and an IFEval item's expected answer (`EXPECTED_MARKER`), in the answers to
  those two tasks' items (`ANSWER_MARKER`), in every thinking block (`THINKING_MARKER`) and as one stored result's stop
  reason (`GRADE_MARKER`), which the grader takes as that result's status: a marker that lands in the grades and so in
  the tables. Nothing printed or logged before the tables step may hold any of them.

Token counts are the same for every item of a task, so each config's cost and each mix's total are worked by hand in
`analysis/check_final_run.py`: 1,000 input tokens and an output of 500 (haiku), 400 (sonnet), 300 (opus_low) or 600
(opus) tokens, doubled for MBPP+.

Nothing here touches the real store, the real benchmark or `results/`.
"""

import json
import subprocess
import sys
import time

from analysis.made_up_world import Content, stored_row_values
from harness.client import build_request, load_models
from harness.loaders import BANKING77_PROMPT, MMLU_PRO_PROMPT, structured_variant
from harness.store import request_hash
from routers.sandbox import ExecResult

EXPECTED_MARKER = "REHEARSAL-EXPECTED-MARKER"  # in an MBPP+ item's tests and an IFEval item's expected answer
ANSWER_MARKER = "REHEARSAL-ANSWER-MARKER"  # in the stored answers to those two tasks' items
THINKING_MARKER = "REHEARSAL-THINKING-MARKER"  # in every stored thinking block
GRADE_MARKER = "REHEARSAL-GRADE-MARKER"  # one stored result's stop reason, so its status in the matrix and the tables
CONFIGS = ("haiku", "sonnet", "opus_low", "opus")
OUTPUT_TOKENS = {"haiku": 500, "sonnet": 400, "opus_low": 300, "opus": 600}
INPUT_TOKENS = 1000

LABELS = ["card_arrival", "lost_card"]
NO_COMMA = {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}


def banking(number: int, label: str, text: str) -> dict:
    prompt = BANKING77_PROMPT.format(labels="\n".join(LABELS), text=text)
    return structured_variant(item(f"banking77:made-test-{number}", "banking77", "exact_label", prompt, label))


def mmlu(number: int, letter: str, question: str) -> dict:
    prompt = MMLU_PRO_PROMPT.format(question=question, options="A) 3\nB) 4\nC) 5\nD) 6")
    return structured_variant({**item(f"mmlu_pro:made-test-{number}", "mmlu_pro", "choice_letter", prompt, letter),
                               "meta": {"num_options": 4}})  # fmt: skip


def item(item_id: str, source: str, method: str, prompt: str, expected: object) -> dict:
    return {"item_id": item_id, "split": "test", "source": source, "source_group": item_id, "task_type": "made up",
            "evaluation_method": method, "meta": {}, "prompt": prompt, "expected_answer": expected}  # fmt: skip


def code_item(number: int, tests: str) -> dict:
    expected = {"test_imports": [], "test": f"# {EXPECTED_MARKER}\n{tests}"}
    return item(f"mbppplus:made-test-{number}", "mbppplus", "code_tests", f"Write a function for case {number}.",
                expected)  # fmt: skip


ITEMS = [
    banking(0, "card_arrival", "Where is my card?"),
    banking(1, "lost_card", "I cannot find my card anywhere."),
    mmlu(0, "B", "What is 2 + 2?"),
    mmlu(1, "C", "What is 2 + 3?"),
    item(
        "ifeval:made-test-0",
        "ifeval",
        "ifeval_strict",
        "Write one sentence about rain, with no commas.",
        {**NO_COMMA, "note": EXPECTED_MARKER},
    ),  # fmt: skip
    item(
        "ifeval:made-test-1",
        "ifeval",
        "ifeval_strict",
        "Write one sentence about snow, with no commas.",
        {**NO_COMMA, "note": EXPECTED_MARKER},
    ),  # fmt: skip
    code_item(0, "assert add(1, 2) == 3\nassert add(2, 5) == 7"),
    code_item(1, "assert mul(2, 3) == 6\nassert mul(4, 5) == 20"),
]
TEST_IDS = [made["item_id"] for made in ITEMS]
BENCHMARK_ROWS = [
    {**made, "split": "train", "item_id": made["item_id"].replace("test", "train")} for made in ITEMS[:1]
] + ITEMS  # a train row, so the benchmark has the other split too

# Which configs get each item right, in the ladder's order (haiku, sonnet, opus_low, opus), by item position.
CORRECT = [
    (1, 1, 1, 1), (0, 1, 1, 1),  # Banking77: the second item is wrong on haiku
    (1, 1, 1, 1), (0, 0, 1, 0),  # MMLU-Pro: the second is wrong on haiku, sonnet (a refusal) and opus
    (1, 1, 1, 1), (0, 0, 0, 1),  # IFEval: the second is wrong on haiku (a stop the grader doesn't know), sonnet, opus_low
    (1, 1, 1, 1), (0, 0, 1, 1),  # MBPP+: the second is wrong on haiku and sonnet
]  # fmt: skip
STOPS = {(5, "haiku"): GRADE_MARKER, (3, "sonnet"): "refusal"}  # (item position, config): stop reason, else end_turn


def answer_text(position: int, right: bool) -> str:
    """What a config says for item `position`, right or wrong; the stop reasons above carry no answer worth reading."""
    source = ITEMS[position]["source"]
    if source == "banking77":
        label = (
            ITEMS[position]["expected_answer"]
            if right
            else next(other for other in LABELS if other != ITEMS[position]["expected_answer"])
        )
        return json.dumps({"answer": label})
    if source == "mmlu_pro":
        letter = ITEMS[position]["expected_answer"] if right else "A"
        return json.dumps({"explanation": "worked it out", "answer": letter})
    if source == "ifeval":
        return f"Rain fell all night {ANSWER_MARKER}" if right else f"Rain fell, all night {ANSWER_MARKER}"
    name, operator = ("add", "+") if position == 6 else ("mul", "*")
    wrong = "-" if operator == "+" else "+"
    return f"```python\n# {ANSWER_MARKER}\ndef {name}(a, b):\n    return a {operator if right else wrong} b\n```"


def sealed_rows() -> list[tuple[tuple, str]]:
    """The 32 stored results (each item's four configs), each with the ID of the test item it answers."""
    models = load_models()
    rows = []
    for position, made in enumerate(ITEMS):
        for index, config in enumerate(CONFIGS):
            settings = models["configs"][config]
            request = build_request(settings, made["prompt"], settings["max_tokens"],
                                    output_schema=made.get("output_schema"))  # fmt: skip
            stop = STOPS.get((position, config), "end_turn")
            output = OUTPUT_TOKENS[config] * (2 if made["source"] == "mbppplus" else 1)
            response = {
                "model": settings["model"],
                "stop_reason": stop,
                "content": [{"type": "thinking", "thinking": f"{THINKING_MARKER}-{position}"},
                            {"type": "text", "text": answer_text(position, bool(CORRECT[position][index]))}],
                "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": output},
            }  # fmt: skip
            if stop == "refusal":
                response["stop_details"] = {"category": "made_up"}
            rows.append((stored_row_values(made["item_id"], config, 0, settings, request_hash(request), response),
                         made["item_id"]))  # fmt: skip
    return rows


def content(train_items: int = 1) -> Content:
    """The rehearsal's benchmark and sealed rows. `train_items` is how many train rows the benchmark holds (the Jev routers'
    drift check, whose noise sample is a world's train items, wants several); each is the first test item's text again."""
    more = [{**ITEMS[0], "split": "train", "source_group": f"made-train-{n}", "item_id": f"banking77:made-train-{n}"}
            for n in range(1, train_items)]  # fmt: skip
    return Content(BENCHMARK_ROWS + more, sealed_rows())


def local_run_python(source: str, timeout_s: float = 10.0, memory: str = "512m") -> ExecResult:
    """The CI rehearsal's stand-in sandbox: the program on standard input of a fresh local interpreter, as the real
    sandbox gets it, with the same result type. It isolates nothing, and only made-up code is run through it."""
    started = time.perf_counter()
    try:
        done = subprocess.run([sys.executable, "-I", "-"], input=source, capture_output=True, text=True,
                              timeout=timeout_s)  # fmt: skip
    except subprocess.TimeoutExpired:
        return ExecResult(None, True, "", "", time.perf_counter() - started)
    return ExecResult(done.returncode, False, done.stdout, done.stderr, time.perf_counter() - started)
