"""Score a stored model response against its item's expected answer (PLAN_v4.md §4, Quality).

This is the only module that reads `expected_answer`. Routers and gates never
import it (§9, the leakage rule).

Every core task is right or wrong, so `quality` is 1.0 or 0.0 and `success` is
the same value as a bool. IFEval is scored prompt-level strict: an item succeeds
only if every instruction is followed. The per-instruction results stay in
`detail` as a diagnostic.

Parsing is modest on purpose. It forgives whitespace, markdown emphasis, code
ticks, wrapping quotes and letter case where case carries no meaning, and it
never guesses what a malformed answer meant. A response with no answer in the
requested format scores `no_answer`, counted apart from `wrong`, so format
failures stay visible per config.

A structured-output item (one with `output_schema`, see
loaders.structured_variant) answers in JSON. Its `answer` field is graded by
the same comparison as the free-text answer it replaces.

Call `score` from the main thread: math-verify times out with SIGALRM, which
only works there.
"""

import ast
import base64
import functools
import json
import re
import zlib
from dataclasses import dataclass, field

import langdetect
import nltk
from math_verify import LatexExtractionConfig, parse, verify

from harness.client import PROJECT_ROOT, answer_text
from harness.ifbench import instructions_registry as ifbench_registry
from harness.ifeval import instructions_registry as ifeval_registry
from harness.loaders import banking77_labels, last_boxed, livecodebench_tests
from harness.sandbox import run_python

NLTK_DATA = PROJECT_ROOT / ".cache" / "nltk_data"
CODE_TIMEOUT_S = 20.0  # the limit the MBPP+ reference solutions were checked under
# MBPP+ items whose own tests need more than the sandbox's default 512 MB. The
# reference solution of mbppplus:255 peaks at about 2.2 GB and passes at 3 GB.
CODE_MEMORY = {"mbppplus:255": "3g"}
LCB_TEST_TIMEOUT_S = 6  # LiveCodeBench's per-test limit
LCB_MEMORY = "2g"  # LiveCodeBench allows 4 GB; some inputs are tens of MB
LCB_RUNNER = (PROJECT_ROOT / "harness" / "livecodebench_runner.py").read_text()

# Seeded, so a language check gives the same verdict on the same text every run.
langdetect.DetectorFactory.seed = 0


@dataclass
class Score:
    # correct | wrong | no_answer, or the stop reason when the model didn't finish
    status: str
    parsed: str | None = None  # the extracted answer, kept for hand review
    detail: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "correct"

    @property
    def quality(self) -> float:
        return 1.0 if self.success else 0.0


def score(item: dict, response: dict) -> Score:
    """Score one stored API response (the record's `response` dict) for one item."""
    stop = response["stop_reason"]
    if stop != "end_turn":
        # Refusal and max_tokens stops are failures (Amendment A1, item 1). Any
        # other stop reason is unexpected and fails the same visible way.
        return Score(stop)
    text = answer_text(response)
    if "output_schema" in item:
        return _score_structured(item, text)
    return EVALUATORS[item["evaluation_method"]](item, text)


def _score_structured(item: dict, text: str) -> Score:
    try:
        answer = json.loads(text)["answer"]
    except (ValueError, KeyError, TypeError):  # not JSON, or no answer field
        return Score("no_answer", text[:200])
    if not isinstance(answer, str):
        return Score("no_answer", repr(answer)[:200])
    return GRADERS[item["evaluation_method"]](item, answer)


# Shared parsing

# A line like "Answer: X", also "**Answer:** X", "> Final answer: X".
_ANSWER_LINE = re.compile(
    r"^[ \t*_#>]*(?:final[ \t]+)?answer[ \t*_]*:(?P<value>.*)$", re.IGNORECASE | re.MULTILINE
)


def _last_answer_line(text: str) -> tuple[str, str] | None:
    """(value, text after that line) for the last answer line, or None if there is none."""
    matches = list(_ANSWER_LINE.finditer(text))
    if not matches:
        return None
    return matches[-1]["value"], text[matches[-1].end() :]


def _strip_markup(value: str) -> str:
    """Remove surrounding whitespace, bold/italic markers and code ticks or fences."""
    value = value.strip().strip("*_").strip()
    fenced = re.fullmatch(r"```[\w+-]*[ \t]*\n?(.*?)\n?[ \t]*```", value, re.DOTALL)
    if fenced:
        return fenced[1].strip()
    if len(value) >= 2 and value[0] == value[-1] == "`":
        return value[1:-1].strip()
    return value


# exact_label (Banking77)


def exact_label(item: dict, text: str) -> Score:
    value, previous = text, None
    while value != previous:
        previous = value
        value = _strip_markup(value).rstrip(".").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
    # The whole reply must be a label. Labels can contain other labels, so a
    # label inside a longer reply is never picked out.
    return grade_label(item, value)


def grade_label(item: dict, value: str) -> Score:
    # Case is ignored: one label is capitalized (Refund_not_showing_up), and
    # structured outputs don't guarantee the case of enum values.
    labels = {label.casefold(): label for label in banking77_labels(item["prompt"])}
    label = labels.get(value.casefold())
    if label is None:
        return Score("no_answer", value[:200])
    return Score("correct" if label == item["expected_answer"] else "wrong", label)


# choice_letter (MMLU-Pro)

# "C", "C.", "C) text", "C: text", "(C)", "(C) text". A letter followed directly
# by words ("I think...") is not read as a choice.
_CHOICE = re.compile(r"\((?P<a>[A-J])\).*|(?P<b>[A-J])(?:[.:)].*)?", re.DOTALL)


def choice_letter(item: dict, text: str) -> Score:
    found = _last_answer_line(text)
    if found is None:
        return Score("no_answer")
    match = _CHOICE.fullmatch(_strip_markup(found[0]))
    if match is None:
        return Score("no_answer", found[0].strip()[:200])
    return grade_letter(item, match["a"] or match["b"])


def grade_letter(item: dict, letter: str) -> Score:
    # Upper-cased for structured outputs, which don't guarantee enum case. The
    # free-text parser only accepts capitals, so there this changes nothing.
    letter = letter.strip().upper()
    return Score("correct" if letter == item["expected_answer"] else "wrong", letter)


# math_equivalence (MATH)

_TEX_SPACING = re.compile(r"\\left(?![A-Za-z])|\\right(?![A-Za-z])|\\[!,;: ]|\s+|\$")


def _normalize_tex(tex: str) -> str:
    """Notation-only cleanup: spacing commands, \\left/\\right, \\dfrac, a trailing period."""
    tex = _TEX_SPACING.sub("", tex)
    return tex.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").rstrip(".")


def math_equivalence(item: dict, text: str) -> Score:
    boxed = last_boxed(text)
    if boxed is None or not boxed.strip():
        return Score("no_answer")
    return grade_math(item, boxed)


def grade_math(item: dict, boxed: str) -> Score:
    boxed = boxed.strip().strip("$").strip()
    if "\\boxed" in boxed:  # a JSON answer that boxed itself anyway
        boxed = last_boxed(boxed) or boxed
    if not boxed:
        return Score("no_answer")
    expected = item["expected_answer"]
    if _normalize_tex(boxed) == _normalize_tex(expected):
        return Score("correct", boxed, {"method": "string"})
    # LaTeX parsing only. math-verify's plain-expression parser runs sympy's
    # parse_expr, which evaluates Python, and model output is untrusted. Its
    # comparison defaults are used as is.
    config = [LatexExtractionConfig()]
    gold = parse(f"\\boxed{{{expected}}}", config)
    pred = parse(f"\\boxed{{{boxed}}}", config)
    return Score("correct" if verify(gold, pred) else "wrong", boxed, {"method": "math_verify"})


# python_literal (CRUXEval)


def python_literal(item: dict, text: str) -> Score:
    found = _last_answer_line(text)
    if found is None:
        return Score("no_answer")
    value, after = found
    # An empty answer line may be followed by the value, e.g. in a code block.
    return grade_literal(item, value if value.strip() else after)


def grade_literal(item: dict, value: str) -> Score:
    value = _strip_markup(value)
    try:
        predicted = ast.literal_eval(value)
    except Exception:  # SyntaxError, ValueError, RecursionError, ...
        return Score("no_answer", value[:200])
    # CRUXEval itself checks `f(input) == output`, so == is the comparison.
    correct = predicted == ast.literal_eval(item["expected_answer"])
    return Score("correct" if correct else "wrong", value)


# ifeval_strict (IFEval) and ifbench_strict (IFBench)

# nltk data the checkers use: sentence and word tokenizers (both), and for
# IFBench a stopword list and a part-of-speech tagger. Kept in the project
# (.cache/, not committed), downloaded on first use. None of these is a pickle.
NLTK_PACKAGES = {
    "punkt_tab": "tokenizers/punkt_tab/english/",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng/",
}


@functools.cache
def _ensure_nltk_data() -> None:
    nltk.data.path.insert(0, str(NLTK_DATA))
    for package, resource in NLTK_PACKAGES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, download_dir=str(NLTK_DATA), quiet=True)
            nltk.data.find(resource)


def _strict_instructions(registry, item: dict, text: str) -> Score:
    """Upstream's test_instruction_following_strict (both evaluation_lib.py files), per item."""
    _ensure_nltk_data()
    spec = item["expected_answer"]
    results = []
    for instruction_id, kwargs in zip(spec["instruction_id_list"], spec["kwargs"], strict=True):
        instruction = registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        # Some copies of the dataset fill every unused argument with None, which
        # build_description rejects; upstream's own data omits them.
        instruction.build_description(**{k: v for k, v in kwargs.items() if v is not None})
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=item["prompt"])
        try:
            followed = bool(text.strip() and instruction.check_following(text))
        except Exception as e:  # a checker crashing on odd text; upstream would stop the whole run
            followed = False
            results.append([instruction_id, followed, f"checker error: {e!r}"[:200]])
            continue
        results.append([instruction_id, followed])
    status = "correct" if all(result[1] for result in results) else "wrong"
    return Score(status, None, {"instructions": results})


def ifeval_strict(item: dict, text: str) -> Score:
    return _strict_instructions(ifeval_registry, item, text)


def ifbench_strict(item: dict, text: str) -> Score:
    return _strict_instructions(ifbench_registry, item, text)


# code_tests (MBPP+)

_CODE_BLOCK = re.compile(r"```(?P<lang>[\w+-]*)[ \t]*\n(?P<code>.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    """The last ```python block, else the last unlabeled ``` block, else None."""
    blocks = [(m["lang"].lower(), m["code"]) for m in _CODE_BLOCK.finditer(text)]
    python = [code for lang, code in blocks if lang in ("python", "py", "python3")]
    plain = [code for lang, code in blocks if not lang]
    return (python or plain or [None])[-1]


def mbppplus_program(code: str, tests: dict) -> str:
    return "\n".join(tests["test_imports"]) + "\n" + code + "\n\n" + tests["test"]


def code_tests(item: dict, text: str) -> Score:
    code = extract_code(text)
    if code is None or not code.strip():
        return Score("no_answer")
    # A SandboxError (Docker failed) propagates: it is never a model failure.
    memory = CODE_MEMORY.get(item["item_id"], "512m")
    result = run_python(mbppplus_program(code, item["expected_answer"]), timeout_s=CODE_TIMEOUT_S, memory=memory)
    passed = result.returncode == 0 and not result.timed_out
    detail = {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_s": round(result.duration_s, 2),
        "stderr_tail": result.stderr.strip().splitlines()[-1:],
    }
    return Score("correct" if passed else "wrong", code, detail)


# Free-text evaluators: find the answer in the reply, then grade it.
# lcb_tests (LiveCodeBench)


def lcb_tests(item: dict, text: str) -> Score:
    """Run the extracted code against every test in the sandbox (see livecodebench_runner.py)."""
    code = extract_code(text)
    if code is None or not code.strip():
        return Score("no_answer")
    spec = item["expected_answer"]
    tests = livecodebench_tests(item)
    packed = base64.b64encode(zlib.compress(json.dumps(tests).encode("utf-8"), 6)).decode("ascii")
    program = (
        f"KIND = {spec['kind']!r}\nFN_NAME = {spec['fn_name']!r}\nTIMEOUT = {LCB_TEST_TIMEOUT_S}\n"
        f"SOLUTION = {code!r}\nTESTS = {packed!r}\n\n" + LCB_RUNNER
    )
    # LiveCodeBench's own overall limit, (per-test limit + 1) per test plus 5 s,
    # plus 10 s for the container to start.
    budget_s = (LCB_TEST_TIMEOUT_S + 1) * len(tests) + 15
    # A SandboxError (Docker failed) propagates: it is never a model failure.
    result = run_python(program, timeout_s=budget_s, memory=LCB_MEMORY)
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):  # the runner itself died: out of memory, a crash, the overall limit
        detail = {"runner": "no report", "returncode": result.returncode, "timed_out": result.timed_out,
                  "stderr_tail": result.stderr.strip().splitlines()[-1:], "duration_s": round(result.duration_s, 2)}  # fmt: skip
        return Score("wrong", code, detail)
    passed = report["passed"] == report["total"] == len(tests)
    detail = {"passed": report["passed"], "total": report["total"], "failure": report["failure"],
              "duration_s": round(result.duration_s, 2)}  # fmt: skip
    return Score("correct" if passed else "wrong", code, detail)


EVALUATORS = {
    "exact_label": exact_label,
    "choice_letter": choice_letter,
    "math_equivalence": math_equivalence,
    "python_literal": python_literal,
    "ifeval_strict": ifeval_strict,
    "ifbench_strict": ifbench_strict,
    "code_tests": code_tests,
    "lcb_tests": lcb_tests,
}

# Graders for an answer already extracted, used for structured-output items.
GRADERS = {
    "exact_label": grade_label,
    "choice_letter": grade_letter,
    "math_equivalence": grade_math,
    "python_literal": grade_literal,
}
