"""Load each candidate dataset into the plan's item format (PLAN_v4.md §5).

Dataset revisions are pinned, so an item ID always means the same item.
`expected_answer` is for evaluators only. `meta` is for reporting only.
Routers and gates never see either.

`split` stays None here; the pilot sampler sets "pilot", and Stage 1 assigns
train / dev / test to the remaining items.
"""

import base64
import hashlib
import io
import json
import pickle
import zlib
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

REVISIONS = {
    "mteb/banking77": "18072d2685ea682290f7b8924d94c62acc19c0b2",
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "cruxeval-org/cruxeval": "b96af0450242eb4da433032b90998f25588a5d0f",
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "evalplus/mbppplus": "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
    # Tested by Amendment A4 as additions to the instruction-following and coding
    # tasks, and not adopted. Kept so that pilot can be reproduced.
    "allenai/IFBench_test": "2e8a48de45ff3bf41242f927254ca81b59ca3ae2",
    "livecodebench/code_generation_lite": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
}

# LiveCodeBench releases v5 and v6 (problems from contests dated 2024-09 to
# 2025-04), the newest and so least likely to be in training data. The full
# dataset is about 4.5 GB; these two files are about 690 MB.
LIVECODEBENCH_FILES = {"test5.jsonl": "v5", "test6.jsonl": "v6"}

# Decoded LiveCodeBench tests (up to 30 MB per problem) live here, not in the
# items. Each item carries the sha256 of its tests, checked on every read.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
LIVECODEBENCH_TESTS_DIR = CACHE_DIR / "livecodebench"

# Items dropped from the pools, with the reason, so the write-up can list them.
# Adding an entry changes the pilot draw for that source, so after the pilot
# is frozen, new exclusions belong in the Stage 1 split builder instead.
EXCLUDED = {
    "mbppplus:590": "reference solution fails its own test on both macOS and Linux "
    "(last-digit float difference in cmath.polar output)",
}

# mbppplus:255's tests peak at ~2.2 GB, over the sandbox's default 512 MB, so
# its grader gets 3 GB (evaluators.CODE_MEMORY) instead of excluding the item.

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Prompt templates. Each one fixes the answer format its evaluator parses.

BANKING77_PROMPT = """Classify the customer's banking query into exactly one of these intents:

{labels}

Query: {text}

Reply with only the intent name, exactly as written in the list."""

MMLU_PRO_PROMPT = """Answer the following multiple-choice question.

{question}

{options}

End your response with a final line of the form "Answer: X", where X is the letter of the correct option."""

MATH_PROMPT = """Solve the following math problem. Put your final answer in \\boxed{{}}.

{problem}"""

CRUXEVAL_PROMPT = """Here is a Python function:

```python
{code}
```

What does f({input}) return?

End your response with a final line of the form "Answer: <value>", where <value> is the returned value written as a Python literal, for example 'abc', [1, 2] or {{'a': 1}}."""

MBPPPLUS_PROMPT = """Write a Python function for the task below. It must pass this example test:

{example_test}

Task: {description}

Return the complete function, with any imports it needs, in a single ```python code block."""


# Structured-output variants of the four short-answer tasks, for the pilot's
# format comparison: the same prompt with its format instruction swapped for one
# that asks for the answer as JSON, plus the schema sent as output_config.format.
# IFEval has no variant (its instructions are about the free-text shape) and
# neither does MBPP+ (code stays in a code block). The schema asks for a short
# explanation, never for reasoning: a `reasoning` field can draw a
# `reasoning_extraction` refusal.
STRUCTURED_INSTRUCTIONS = {
    # source: (free-text instruction, its structured replacement)
    "banking77": (
        "Reply with only the intent name, exactly as written in the list.",
        "Give the intent name as the answer.",
    ),
    "mmlu_pro": (
        'End your response with a final line of the form "Answer: X", where X is the letter of the correct option.',
        "Give a short explanation, then the letter of the correct option as the answer.",
    ),
    "math": (
        "Put your final answer in \\boxed{}.",
        "Give a short explanation, then the final answer in LaTeX, without \\boxed{}.",
    ),
    "cruxeval": (
        'End your response with a final line of the form "Answer: <value>", where <value> is the returned value '
        "written as a Python literal, for example 'abc', [1, 2] or {'a': 1}.",
        "Give a short explanation, then the returned value as the answer, written as a Python literal, "
        "for example 'abc', [1, 2] or {'a': 1}.",
    ),
}


def banking77_labels(prompt: str) -> list[str]:
    """The intent labels listed in a Banking77 prompt, in order."""
    head = BANKING77_PROMPT.split("{labels}")[0]
    if not prompt.startswith(head):
        raise ValueError("prompt does not follow BANKING77_PROMPT")
    return prompt[len(head) : prompt.index("\n\nQuery: ", len(head))].split("\n")


def structured_variant(item: dict) -> dict:
    """The item asking for its answer as JSON, with the schema in `output_schema`."""
    free_text, structured = STRUCTURED_INSTRUCTIONS[item["source"]]
    if item["prompt"].count(free_text) != 1:
        raise ValueError(f"{item['item_id']}: format instruction not found exactly once")
    if item["source"] == "banking77":
        answer = {"type": "string", "enum": banking77_labels(item["prompt"])}
    elif item["source"] == "mmlu_pro":
        answer = {"type": "string", "enum": list("ABCDEFGHIJ"[: item["meta"]["num_options"]])}
    else:
        answer = {"type": "string"}
    # Banking77's free-text prompt asks for the label alone, so its schema has no explanation either.
    properties: dict[str, dict] = {} if item["source"] == "banking77" else {"explanation": {"type": "string"}}
    properties["answer"] = answer
    schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    return {**item, "prompt": item["prompt"].replace(free_text, structured), "output_schema": schema}


LIVECODEBENCH_STDIN_PROMPT = """Solve the following programming problem in Python.

{question}

Your program must read the input from standard input and print the answer to standard output.

Return the complete program in a single ```python code block."""

LIVECODEBENCH_FUNCTIONAL_PROMPT = """Solve the following programming problem in Python.

{question}

Complete this starter code:

```python
{starter_code}
```

Return the complete solution, the class with its method and any imports it needs, in a single ```python code block."""


def _load(name: str, split: str, config: str | None = None):
    return load_dataset(name, config, split=split, revision=REVISIONS[name])


def _item(source, native_id, task_type, prompt, expected_answer, evaluation_method, meta=None):
    item_id = f"{source}:{native_id}"
    return {
        "item_id": item_id,
        "source": source,
        # Every item is independent in these six sets, so each is its own group.
        "source_group": item_id,
        "task_type": task_type,
        "split": None,
        "prompt": prompt,
        "expected_answer": expected_answer,
        "evaluation_method": evaluation_method,
        "meta": meta or {},
    }


def load_banking77() -> list[dict]:
    rows = _load("mteb/banking77", "test")
    names = {}
    for split in ("train", "test"):
        for row in _load("mteb/banking77", split):
            names[row["label"]] = row["label_text"]
    labels = "\n".join(names[i] for i in sorted(names))
    return [
        _item(
            "banking77",
            f"test-{i:04d}",
            "classification",
            BANKING77_PROMPT.format(labels=labels, text=row["text"]),
            row["label_text"],
            "exact_label",
        )
        for i, row in enumerate(rows)
    ]


def load_mmlu_pro() -> list[dict]:
    items = []
    for row in _load("TIGER-Lab/MMLU-Pro", "test"):
        letters = "ABCDEFGHIJ"[: len(row["options"])]
        if letters[row["answer_index"]] != row["answer"]:
            raise ValueError(f"MMLU-Pro {row['question_id']}: answer letter and index disagree")
        options = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, row["options"]))
        items.append(
            _item(
                "mmlu_pro",
                row["question_id"],
                "multiple_choice",
                MMLU_PRO_PROMPT.format(question=row["question"], options=options),
                row["answer"],
                "choice_letter",
                {"category": row["category"], "src": row["src"], "num_options": len(letters)},
            )
        )
    return items


def last_boxed(solution: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}) in a MATH solution, braces matched."""
    start = max(solution.rfind("\\boxed{"), solution.rfind("\\fbox{"))
    if start == -1:
        return None
    i = solution.index("{", start) + 1
    depth = 1
    for j in range(i, len(solution)):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[i:j]
    return None


def load_math() -> tuple[list[dict], list[str]]:
    """Return (items, skipped_ids). Items whose reference answer can't be extracted are skipped."""
    items, skipped = [], []
    for subject in MATH_SUBJECTS:
        for i, row in enumerate(_load("EleutherAI/hendrycks_math", "test", subject)):
            native_id = f"{subject}-test-{i:04d}"
            answer = last_boxed(row["solution"])
            if answer is None:
                skipped.append(f"math:{native_id}")
                continue
            items.append(
                _item(
                    "math",
                    native_id,
                    "math",
                    MATH_PROMPT.format(problem=row["problem"]),
                    answer,
                    "math_equivalence",
                    {"level": row["level"], "subject": row["type"]},
                )
            )
    return items, skipped


def load_cruxeval() -> list[dict]:
    return [
        _item(
            "cruxeval",
            row["id"],
            "code_reasoning",
            CRUXEVAL_PROMPT.format(code=row["code"], input=row["input"]),
            row["output"],
            "python_literal",
        )
        for row in _load("cruxeval-org/cruxeval", "test")
    ]


def load_ifeval() -> list[dict]:
    return [
        _item(
            "ifeval",
            row["key"],
            "instruction_following",
            row["prompt"],
            {"instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"]},
            "ifeval_strict",
        )
        for row in _load("google/IFEval", "train")
    ]


def load_mbppplus() -> list[dict]:
    return [
        _item(
            "mbppplus",
            row["task_id"],
            "coding",
            MBPPPLUS_PROMPT.format(example_test=row["test_list"][0], description=row["prompt"]),
            # The test program calls the function by name, so the model's code
            # must define it; the example test in the prompt shows the name.
            {"test_imports": row["test_imports"], "test": row["test"]},
            "code_tests",
        )
        for row in _load("evalplus/mbppplus", "test")
    ]


def _whole_numbers(value):
    # The Hugging Face copy stores integer arguments as floats (3.0); upstream's
    # own data file gives the same values. 3 and 3.0 compare equal, and an int
    # is also safe where a checker uses the value as an index.
    return int(value) if isinstance(value, float) and value.is_integer() else value


def load_ifbench() -> list[dict]:
    items = []
    for row in _load("allenai/IFBench_test", "train"):
        kwargs = [{k: _whole_numbers(v) for k, v in kw.items() if v is not None} for kw in row["kwargs"]]
        items.append(
            _item(
                "ifbench",
                row["key"],
                "instruction_following",
                row["prompt"],
                {"instruction_id_list": row["instruction_id_list"], "kwargs": kwargs},
                "ifbench_strict",
            )
        )
    return items


class _TextOnlyUnpickler(pickle.Unpickler):
    """Refuses every class and function lookup, so a pickle can't run code here.

    LiveCodeBench stores most hidden tests as base64(zlib(pickle(json text))).
    A pickled str needs no lookups, so only a malicious or unexpected file fails.
    """

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def _decode_livecodebench_tests(blob: str) -> list[dict]:
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        text = _TextOnlyUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(blob)))).load()
        if not isinstance(text, str):
            raise ValueError(f"expected pickled text, got {type(text).__name__}")
        return json.loads(text)


def _tests_path(question_id: str) -> Path:
    return LIVECODEBENCH_TESTS_DIR / f"{question_id}.json.zlib"


def load_livecodebench() -> list[dict]:
    """Problems from LIVECODEBENCH_FILES. Their tests are written to the cache on first load."""
    index_path = LIVECODEBENCH_TESTS_DIR / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    items = []
    for filename, release in LIVECODEBENCH_FILES.items():
        path = hf_hub_download(
            "livecodebench/code_generation_lite",
            filename,
            repo_type="dataset",
            revision=REVISIONS["livecodebench/code_generation_lite"],
        )
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                qid = row["question_id"]
                if qid not in index or not _tests_path(qid).exists():
                    # Public tests first, then hidden ones, as LiveCodeBench grades them.
                    cases = json.loads(row["public_test_cases"]) + _decode_livecodebench_tests(
                        row["private_test_cases"]
                    )
                    tests = [{"input": c["input"], "output": c["output"]} for c in cases]
                    kinds = {c["testtype"] for c in cases}
                    if len(kinds) != 1:
                        raise ValueError(f"livecodebench:{qid}: mixed test types {kinds}")
                    data = json.dumps(tests, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                    LIVECODEBENCH_TESTS_DIR.mkdir(parents=True, exist_ok=True)
                    _tests_path(qid).write_bytes(zlib.compress(data, 6))
                    index[qid] = {
                        "kind": kinds.pop(),
                        "num_tests": len(tests),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                kind = index[qid]["kind"]
                if kind == "functional":
                    prompt = LIVECODEBENCH_FUNCTIONAL_PROMPT.format(
                        question=row["question_content"].strip(), starter_code=row["starter_code"].rstrip()
                    )
                else:
                    prompt = LIVECODEBENCH_STDIN_PROMPT.format(question=row["question_content"].strip())
                items.append(
                    _item(
                        "livecodebench",
                        qid,
                        "coding",
                        prompt,
                        {
                            "kind": kind,
                            "fn_name": json.loads(row["metadata"]).get("func_name"),
                            "num_tests": index[qid]["num_tests"],
                            "tests_sha256": index[qid]["sha256"],
                        },
                        "lcb_tests",
                        {
                            "platform": row["platform"],
                            "difficulty": row["difficulty"],
                            "contest_date": row["contest_date"],
                            "release": release,
                            "title": row["question_title"],
                        },
                    )
                )
    LIVECODEBENCH_TESTS_DIR.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True))
    return items


def livecodebench_tests(item: dict) -> list[dict]:
    """The item's tests from the cache, checked against the hash the item carries."""
    qid = item["item_id"].removeprefix("livecodebench:")
    path = _tests_path(qid)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; load_livecodebench() rebuilds the test cache")
    data = zlib.decompress(path.read_bytes())
    if hashlib.sha256(data).hexdigest() != item["expected_answer"]["tests_sha256"]:
        raise ValueError(f"{path} does not match the item's tests_sha256")
    return json.loads(data)


def load_all() -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Return ({source: items}, {dropped item ID: reason})."""
    math_items, math_skipped = load_math()
    pools = {
        "banking77": load_banking77(),
        "mmlu_pro": load_mmlu_pro(),
        "math": math_items,
        "cruxeval": load_cruxeval(),
        "ifeval": load_ifeval(),
        "mbppplus": load_mbppplus(),
        "ifbench": load_ifbench(),
        "livecodebench": load_livecodebench(),
    }
    dropped = {item_id: "no \\boxed answer in the reference solution" for item_id in math_skipped}
    for source, items in pools.items():
        dropped.update({x["item_id"]: EXCLUDED[x["item_id"]] for x in items if x["item_id"] in EXCLUDED})
        pools[source] = [x for x in items if x["item_id"] not in EXCLUDED]
    return pools, dropped
