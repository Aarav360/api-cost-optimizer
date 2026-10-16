"""Check the evaluators: reference answers must score correct, and wrong or malformed ones must not.

Run from the project root:  uv run python -m harness.check_evaluators

Re-run after changing an evaluator, a prompt template, or a dependency in uv.lock.

- Banking77, MMLU-Pro, MATH, CRUXEval: the reference answer of every item in
  the full candidate pool, written the way its prompt asks (plus formatting
  variants), must score correct.
- IFEval has no reference responses: every pool item's instructions must build
  from their stored arguments, and hand-written cases must pass or fail.
- MBPP+: the 20 pilot items' reference solutions, written as a model reply,
  must pass in the sandbox (check_sandbox covers the whole pool).
- IFBench: every item's instructions build, and hand-written cases pass or fail.
- LiveCodeBench: every problem's cached tests load and match their hash, and
  hand-written solutions to two problems (stdin and functional) pass, while
  wrong, crashing and looping ones fail.
- Every task: hand-written wrong and malformed replies, and stop reasons.
- Structured-output variants (Banking77, MMLU-Pro, MATH, CRUXEval): every pool
  item converts, its schema allows the reference answer, and a JSON reply with
  the reference answer scores correct.

Also reports MATH reference answers that math-verify can't parse; those items
can only be graded by string match (worth knowing before Stage 1).
"""

import json
import sys

from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse

from harness.client import PROJECT_ROOT
from harness.evaluators import score
from harness.loaders import REVISIONS, STRUCTURED_INSTRUCTIONS, livecodebench_tests, load_all, structured_variant


def reply(text: str, stop: str = "end_turn") -> dict:
    # Shaped like a stored response; the empty thinking block checks that only
    # text blocks are read as the answer.
    return {
        "stop_reason": stop,
        "content": [{"type": "thinking", "thinking": "", "signature": ""}, {"type": "text", "text": text}],
    }


class Results:
    def __init__(self):
        self.groups: dict[str, list] = {}

    def check(self, group: str, item: dict, response_text: str, expected: str, stop: str = "end_turn"):
        got = score(item, reply(response_text, stop))
        ok = got.status == expected
        self.groups.setdefault(group, []).append(
            None if ok else f"{item['item_id']}: expected {expected}, got {got.status} "
            f"(parsed {got.parsed!r:.80}) for reply {response_text!r:.120}"
        )

    def report(self) -> int:
        failures = 0
        for group, outcomes in self.groups.items():
            failed = [o for o in outcomes if o is not None]
            failures += len(failed)
            print(f"{'FAIL' if failed else 'ok  '} {group}: {len(outcomes) - len(failed)}/{len(outcomes)}")
            for line in failed[:10]:
                print(f"       {line}")
            if len(failed) > 10:
                print(f"       ... and {len(failed) - 10} more")
        return failures


def check_references(r: Results, pools: dict[str, list[dict]]) -> None:
    for item in pools["banking77"]:
        e = item["expected_answer"]
        r.check("banking77 reference", item, e, "correct")
        for variant in (f"**{e}**", f"`{e}`", f"{e}.", f'"{e}"', e.lower(), f"\n{e}\n"):
            r.check("banking77 reference variants", item, variant, "correct")

    for item in pools["mmlu_pro"]:
        e = item["expected_answer"]
        r.check("mmlu_pro reference", item, f"Working through the options.\n\nAnswer: {e}", "correct")
        for variant in (f"**Answer: ({e})**", f"**Answer:** {e}", f"Answer: {e}. That option", f"Answer: {e})"):
            r.check("mmlu_pro reference variants", item, f"Reasoning.\n{variant}", "correct")

    for item in pools["math"]:
        e = item["expected_answer"]
        r.check("math reference", item, f"Working.\n\nThe answer is $\\boxed{{{e}}}$.", "correct")

    for item in pools["cruxeval"]:
        e = item["expected_answer"]
        r.check("cruxeval reference", item, f"Tracing the function.\n\nAnswer: {e}", "correct")
        for variant in (f"**Answer:** `{e}`", f"Answer:\n```python\n{e}\n```", f"Answer: **{e}**"):
            r.check("cruxeval reference variants", item, f"Tracing.\n{variant}", "correct")


def check_bad_replies(r: Results, pools: dict[str, list[dict]]) -> None:
    bank = pools["banking77"][0]
    other_label = next(x["expected_answer"] for x in pools["banking77"] if x["expected_answer"] != bank["expected_answer"])
    r.check("banking77 bad replies", bank, other_label, "wrong")
    r.check("banking77 bad replies", bank, f"The intent is {bank['expected_answer']}", "no_answer")
    r.check("banking77 bad replies", bank, "", "no_answer")

    mmlu = pools["mmlu_pro"][0]
    wrong_letter = "A" if mmlu["expected_answer"] != "A" else "B"
    r.check("mmlu_pro bad replies", mmlu, f"Answer: {wrong_letter}", "wrong")
    r.check("mmlu_pro bad replies", mmlu, f"The answer is {mmlu['expected_answer']}.", "no_answer")
    r.check("mmlu_pro bad replies", mmlu, "Answer: I think it's B", "no_answer")
    r.check("mmlu_pro bad replies", mmlu, f"Answer: {mmlu['expected_answer']}\nAnswer: {wrong_letter}", "wrong")

    math = next(x for x in pools["math"] if x["expected_answer"] == "5")
    r.check("math bad replies", math, "So $\\boxed{6}$.", "wrong")
    r.check("math bad replies", math, "So $\\boxed{-5}$.", "wrong")
    r.check("math bad replies", math, "The answer is 5.", "no_answer")
    r.check("math bad replies", math, "So $\\boxed{}$.", "no_answer")

    crux = next(x for x in pools["cruxeval"] if x["expected_answer"].startswith("'"))
    r.check("cruxeval bad replies", crux, "Answer: 'definitely not this'", "wrong")
    r.check("cruxeval bad replies", crux, "Answer: f returns the reversed string", "no_answer")
    r.check("cruxeval bad replies", crux, f"The output is {crux['expected_answer']}", "no_answer")

    for item in (bank, mmlu, math, crux):
        right = {"banking77": bank["expected_answer"], "mmlu_pro": f"Answer: {mmlu['expected_answer']}",
                 "math": "$\\boxed{5}$", "cruxeval": f"Answer: {crux['expected_answer']}"}[item["source"]]  # fmt: skip
        r.check("stop reasons score as failures", item, right, "max_tokens", stop="max_tokens")
        r.check("stop reasons score as failures", item, right, "refusal", stop="refusal")


def check_structured(r: Results, pools: dict[str, list[dict]]) -> None:
    """Structured-output variants: every pool item converts, its schema allows the right answer,
    and a JSON reply with the reference answer scores correct."""
    for source in STRUCTURED_INSTRUCTIONS:
        for plain in pools[source]:
            group = f"{source} structured"
            try:
                item = structured_variant(plain)
            except ValueError as e:
                r.groups.setdefault(f"{group} conversion", []).append(str(e))
                continue
            r.groups.setdefault(f"{group} conversion", []).append(None)
            answer = item["output_schema"]["properties"]["answer"]
            e = item["expected_answer"]
            if "enum" in answer and e not in answer["enum"]:
                r.groups.setdefault(f"{group} schema allows the answer", []).append(f"{item['item_id']}: {e!r}")
            else:
                r.groups.setdefault(f"{group} schema allows the answer", []).append(None)
            fields = {} if source == "banking77" else {"explanation": "Because."}
            r.check(f"{group} reference", item, json.dumps({**fields, "answer": e}), "correct")
            variants = {"banking77": [e.lower()], "mmlu_pro": [e.lower()], "math": [f"${e}$", f"\\boxed{{{e}}}"],
                        "cruxeval": [f"`{e}`"]}[source]  # fmt: skip
            for variant in variants:
                r.check(f"{group} reference variants", item, json.dumps({**fields, "answer": variant}), "correct")

    for source in STRUCTURED_INSTRUCTIONS:
        item = structured_variant(pools[source][0])
        e = item["expected_answer"]
        wrong = {"banking77": next(x["expected_answer"] for x in pools[source] if x["expected_answer"] != e),
                 "mmlu_pro": "A" if e != "A" else "B", "math": f"{e}+1", "cruxeval": "'definitely not this'"}[source]  # fmt: skip
        r.check("structured bad replies", item, json.dumps({"explanation": "x", "answer": wrong}), "wrong")
        r.check("structured bad replies", item, f"The answer is {e}", "no_answer")  # not JSON
        r.check("structured bad replies", item, json.dumps({"explanation": e}), "no_answer")  # no answer field
        r.check("structured bad replies", item, json.dumps({"answer": e})[:-3], "max_tokens", stop="max_tokens")


def check_ifeval(r: Results, pool: list[dict]) -> None:
    # Every pool item's instructions must build from their stored arguments.
    for item in pool:
        try:
            score(item, reply("x"))
            r.groups.setdefault("ifeval instructions build", []).append(None)
        except Exception as e:  # noqa: BLE001
            r.groups.setdefault("ifeval instructions build", []).append(f"{item['item_id']}: {e!r}")

    def case(ids_and_kwargs, text, expected):
        item = {
            "item_id": f"ifeval-case:{','.join(i for i, _ in ids_and_kwargs)}",
            "prompt": "(hand-written case)",
            "evaluation_method": "ifeval_strict",
            "expected_answer": {
                "instruction_id_list": [i for i, _ in ids_and_kwargs],
                "kwargs": [k for _, k in ids_and_kwargs],
            },
        }
        r.check("ifeval hand-written cases", item, text, expected)

    case([("punctuation:no_comma", {})], "Hello world. Nothing else", "correct")
    case([("punctuation:no_comma", {})], "Hello, world", "wrong")
    case([("detectable_format:number_bullet_lists", {"num_bullets": 2})], "Intro\n* one\n* two", "correct")
    case([("detectable_format:number_bullet_lists", {"num_bullets": 2})], "* one\n* two\n* three", "wrong")
    # Sentence and word counts need nltk's punkt_tab data.
    case([("length_constraints:number_sentences", {"num_sentences": 3, "relation": "less than"})],
         "This is one. This is two.", "correct")  # fmt: skip
    case([("length_constraints:number_sentences", {"num_sentences": 3, "relation": "less than"})],
         "One here. Two here. Three here. Four here.", "wrong")  # fmt: skip
    case([("change_case:capital_word_frequency", {"capital_frequency": 2, "capital_relation": "at least"})],
         "HELLO there BIG world", "correct")  # fmt: skip
    case([("language:response_language", {"language": "ko"})],
         "안녕하세요. 오늘 날씨가 정말 좋네요. 산책하러 가고 싶어요.", "correct")  # fmt: skip
    case([("language:response_language", {"language": "ko"})],
         "Hello there, the weather is lovely today and I would like to go for a walk.", "wrong")  # fmt: skip
    case([("startend:quotation", {})], '"All wrapped up"', "correct")
    # Unused arguments stored as None must be dropped, not passed through.
    case([("detectable_format:title", {"num_bullets": None, "language": None})], "<<A Title>>\nBody", "correct")
    # Prompt level: one failed instruction fails the item.
    case([("punctuation:no_comma", {}), ("detectable_format:title", {})], "<<Title>>\nNo commas here", "correct")
    case([("punctuation:no_comma", {}), ("detectable_format:title", {})], "<<Title>>\nOne, two", "wrong")
    case([("punctuation:no_comma", {})], "   ", "wrong")


def check_ifbench(r: Results, pool: list[dict]) -> None:
    for item in pool:
        try:
            score(item, reply("x"))
            r.groups.setdefault("ifbench instructions build", []).append(None)
        except Exception as e:  # noqa: BLE001
            r.groups.setdefault("ifbench instructions build", []).append(f"{item['item_id']}: {e!r}")

    def case(instruction_id, kwargs, text, expected):
        item = {
            "item_id": f"ifbench-case:{instruction_id}",
            "prompt": "(hand-written case)",
            "evaluation_method": "ifbench_strict",
            "expected_answer": {"instruction_id_list": [instruction_id], "kwargs": [kwargs]},
        }
        r.check("ifbench hand-written cases", item, text, expected)

    case("count:numbers", {"N": 2}, "I have 3 cats and 4 dogs.", "correct")
    case("count:numbers", {"N": 2}, "I have 3 cats, 4 dogs and 5 fish.", "wrong")
    case("format:no_whitespace", {}, "Hello-world!", "correct")
    case("format:no_whitespace", {}, "Hello world", "wrong")
    # These three need nltk's tokenizer, stopword list and part-of-speech tagger.
    case("count:pronouns", {"N": 3}, "I told her that they would come with me.", "correct")
    case("count:pronouns", {"N": 3}, "The dog ran.", "wrong")
    case("ratio:stop_words", {"percentage": 30}, "Bright red apples taste sweet.", "correct")
    case("ratio:stop_words", {"percentage": 30}, "It is the one that is in the box.", "wrong")
    case("words:start_verb", {}, "Go to the store.", "correct")
    case("words:start_verb", {}, "The report is ready.", "wrong")
    keywords = {"keyword1": "a1", "keyword2": "b2", "keyword3": "c3", "keyword4": "d4", "keyword5": "e5"}
    case("count:keywords_multiple", keywords, "a1 b2 b2 c3 c3 c3 d4 d4 d4 d4 d4 e5 e5 e5 e5 e5 e5 e5", "correct")
    case("count:keywords_multiple", keywords, "a1 a1 b2 b2 c3 c3 c3 d4 d4 d4 d4 d4 e5 e5 e5 e5 e5 e5 e5", "wrong")


LCB_MULTIPLICATION = "X = int(input())\nprint(sum(i * j for i in range(1, 10) for j in range(1, 10) if i * j != X))\n"
LCB_ZIGZAG = """class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        seq = []
        for r, row in enumerate(grid):
            seq.extend(row if r % 2 == 0 else row[::-1])
        return seq[::2]
"""


def check_livecodebench(r: Results, pool: list[dict]) -> None:
    for item in pool:
        spec = item["expected_answer"]
        try:
            tests = livecodebench_tests(item)
            ok = len(tests) == spec["num_tests"] > 0
            if spec["kind"] == "functional":
                ok = ok and bool(spec["fn_name"]) and "class Solution" in item["prompt"]
            r.groups.setdefault("livecodebench tests load", []).append(None if ok else f"{item['item_id']}: bad spec")
        except Exception as e:  # noqa: BLE001
            r.groups.setdefault("livecodebench tests load", []).append(f"{item['item_id']}: {e!r}")

    by_id = {x["item_id"]: x for x in pool}
    stdin_item, functional_item = by_id["livecodebench:abc387_b"], by_id["livecodebench:3708"]
    block = "Here is my solution.\n\n```python\n{}```\n"
    r.check("livecodebench runner", stdin_item, block.format(LCB_MULTIPLICATION), "correct")
    r.check("livecodebench runner", stdin_item, block.format(LCB_MULTIPLICATION.replace(" if i * j != X", "")), "wrong")
    r.check("livecodebench runner", stdin_item, block.format("raise ValueError('boom')\n"), "wrong")
    r.check("livecodebench runner", stdin_item, LCB_MULTIPLICATION, "no_answer")  # no code block
    r.check("livecodebench runner", functional_item, block.format(LCB_ZIGZAG), "correct")
    r.check("livecodebench runner", functional_item, block.format(LCB_ZIGZAG.replace("seq[::2]", "seq[1::2]")), "wrong")
    r.check("livecodebench runner", functional_item, block.format(LCB_ZIGZAG.replace("return seq[::2]", "while True: pass")), "wrong")


def check_mbppplus(r: Results, pilot: list[dict]) -> None:
    rows = load_dataset("evalplus/mbppplus", split="test", revision=REVISIONS["evalplus/mbppplus"])
    reference = {f"mbppplus:{row['task_id']}": row["code"] for row in rows}
    items = [x for x in pilot if x["source"] == "mbppplus"]
    for item in items:
        code = reference[item["item_id"]]
        r.check("mbppplus pilot references", item, f"Here is the function.\n\n```python\n{code}\n```\n", "correct")

    item = items[0]
    code = reference[item["item_id"]]
    r.check("mbppplus bad replies", item, "```python\ndef unrelated():\n    return 1\n```", "wrong")
    r.check("mbppplus bad replies", item, "```python\nwhile True:\n    pass\n```", "wrong")  # timeout
    r.check("mbppplus bad replies", item, code, "no_answer")  # no code block
    # A draft block followed by the final one: the last python block is graded.
    r.check("mbppplus last block", item, f"```python\ndef draft(:\n```\nFixed:\n```python\n{code}\n```", "correct")
    r.check("mbppplus unlabeled block", item, f"```\n{code}\n```", "correct")


def report_unparsed_math(pool: list[dict]) -> None:
    unparsed = []
    for item in pool:
        parsed = parse(f"\\boxed{{{item['expected_answer']}}}", [LatexExtractionConfig()])
        if not parsed or isinstance(parsed[0], str):
            unparsed.append(item)
    pilot_note = sum(1 for x in unparsed if x["item_id"] in PILOT_IDS)
    print(f"note math reference answers math-verify can't parse (string match only): "
          f"{len(unparsed)}/{len(pool)}, {pilot_note} of them in the pilot")  # fmt: skip
    for item in unparsed[:8]:
        print(f"       {item['item_id']}: {item['expected_answer']!r}")


PILOT = [json.loads(line) for line in (PROJECT_ROOT / "data" / "pilot.jsonl").open()]
PILOT_IDS = {x["item_id"] for x in PILOT}


def main() -> int:
    pools, _ = load_all()
    r = Results()
    check_references(r, pools)
    check_bad_replies(r, pools)
    check_structured(r, pools)
    check_ifeval(r, pools["ifeval"])
    check_ifbench(r, pools["ifbench"])
    check_livecodebench(r, pools["livecodebench"])
    check_mbppplus(r, PILOT)
    failures = r.report()
    print()
    report_unparsed_math(pools["math"])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
