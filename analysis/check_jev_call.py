"""Check the Jev router's call building (PLAN_v4.md §8 3c; Amendment A15, items 4 and 8).

Run from the project root:  uv run python -m analysis.check_jev_call

Never a test item and never a Jev call: every request is made-up text and the settings are synthetic.

- G's body: equal, byte for byte, to hand-written JSON, with a null schema, non-ASCII text kept, the fixed key
  order and a schema's own key order; the body's sha256;
- the cut, at hand-worked boundaries (escapes, 2- and 3-byte characters, exactly at and one over the budget), counted,
  with the schema left whole, and a call refused without a usable budget;
- the budget, from hand-worked parts: each of Jev's two limits binding in turn, G's from synthetic settings, and
  parts that leave no room refused;
- the settings: each malformed part refused, with its own check;
- the imports: the module's closure uses the standard library and numpy only, and importing it loads no analysis,
  harness, scikit-learn or torch module.
"""

import ast
import hashlib
import subprocess
import sys
from collections.abc import Callable

from analysis.freeze import closure
from harness.client import PROJECT_ROOT
from routers import jev_call
from routers.request import Request

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(error: type[Exception], call: Callable[[], object]) -> bool:
    try:
        call()
    except error:
        return True
    return False


def imported_packages(tree: ast.AST) -> set[str]:
    """The top-level packages a module imports (a relative import stays inside its package)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def synthetic_data() -> dict:
    """A settings file in #30's format, as far as call building reads it, with made-up statements."""
    return {
        "jev": {"model": "jev-test-0"},
        "display_names": {"haiku": "Small", "sonnet": "Middle", "opus_low": "Large"},
        "g": {
            "success_statements": [
                {"id": "success_haiku", "config": "haiku", "text": "Small text."},
                {"id": "success_sonnet", "config": "sonnet", "text": "Middle text."},
                {"id": "success_opus_low", "config": "opus_low", "text": "Large text."},
            ]
        },
        "budget_bytes": {"jev": None, "jev_bolstered": None},
    }


QUESTIONS = (
    r'"questions":{"success_haiku":{"type":"noul","instructions":"Small text."},'
    r'"success_sonnet":{"type":"noul","instructions":"Middle text."},'
    r'"success_opus_low":{"type":"noul","instructions":"Large text."}}'
)


def main() -> int:
    settings = jev_call.parse_settings(synthetic_data())

    print("-- G's body --")
    # The schema's keys are out of alphabetical order on purpose; the prompt has an accent, a quote, a newline and CJK.
    request = Request('Résumé "x"\n日本', {"type": "object", "z": 1, "a": [1, 2]})
    expected = (
        r'{"model":"jev-test-0","state":{"request":{"prompt":"Résumé \"x\"\n日本",'
        r'"output_schema":{"type":"object","z":1,"a":[1,2]}}},' + QUESTIONS + "}"
    )
    call = jev_call.g_call(request, settings, budget_bytes=1000)
    check(call.body == expected.encode("utf-8"), "G's body is the hand-written JSON, byte for byte")
    check("日本".encode() in call.body and b"\\u" not in call.body, "non-ASCII text is kept, not escaped")
    null_body = r'{"model":"jev-test-0","state":{"request":{"prompt":"Hi","output_schema":null}},' + QUESTIONS + "}"
    check(
        jev_call.g_call(Request("Hi"), settings, budget_bytes=1000).body == null_body.encode(),
        'a missing schema is `"output_schema": null`',
    )
    check(
        call.sha256 == hashlib.sha256(expected.encode("utf-8")).hexdigest()
        and jev_call.body_sha256(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "the sha256 is the body's, and sha256 of b'abc' is the known digest",
    )

    print("\n-- the cut --")
    # The budget counts the prompt's serialized bytes (A15 item 8): an escape costs its length, a CJK or accented
    # character its UTF-8 length. Each case below is worked by hand.
    alphabet = "abcdefghijklmnopqrst"  # 20 bytes
    check(
        jev_call.cut_prompt(alphabet, 10) == ("abcdepqrst", 10),
        "20 bytes over a budget of 10 keep 5 at the head and 5 at the tail, and 10 are counted as cut",
    )
    check(jev_call.cut_prompt(alphabet[:10], 10) == (alphabet[:10], 0), "a prompt exactly at the budget is not cut")
    check(
        jev_call.cut_prompt(alphabet[:11], 10) == ("abcdeghijk", 1),
        "a prompt one byte over the budget loses exactly one byte, from the middle",
    )
    check(
        jev_call.cut_prompt("é" * 7, 10) == ("éééé", 6),
        "a 2-byte character is never split: 14 bytes at a budget of 10 keep two characters each side",
    )
    check(
        jev_call.cut_prompt("\n" * 8, 10) == ("\n" * 4, 8),
        "an escape counts as its serialized length: 8 newlines are 16 bytes, and four are kept",
    )
    check(jev_call.cut_prompt("日" * 4, 6) == ("日日", 6), "a 3-byte character each side at a budget of 6")
    check(jev_call.cut_prompt("日" * 4, 5) == ("", 12), "a half too small for a character keeps none of it")
    cut = jev_call.g_call(Request(alphabet, {"type": "object", "pad": "x" * 50}), settings, budget_bytes=10)
    check(
        b'"prompt":"abcdepqrst","output_schema":{"type":"object","pad":"' + b"x" * 50 + b'"}}' in cut.body
        and cut.cut_bytes == 10,
        "the body carries the cut prompt and the whole schema, and the cut is counted",
    )
    check(
        jev_call.g_call(Request(alphabet), settings, budget_bytes=20).cut_bytes == 0,
        "a call within budget counts no cut",
    )
    check(
        refused(jev_call.CallError, lambda: jev_call.g_call(request, settings)),
        "a null budget in the settings and none given is refused: no call is built without a budget",
    )
    check(
        refused(jev_call.CallError, lambda: jev_call.g_call(request, settings, budget_bytes=0)),
        "a budget of zero is refused",
    )

    print("\n-- the budget --")
    # Jev's limits are 32K for the state plus any one question and 64K for the state plus all of them; the code reads
    # them as 32,000 and 64,000 tokens, the tighter reading, and a token is a byte.
    check(
        jev_call.compute_budget_bytes(overhead_tokens=270, state_bytes=1000, question_bytes=[30, 50, 40]) == 30680,
        "the one-question limit binds: 32,000 - 270 - 1,000 - 50 (the longest question) = 30,680",
    )
    check(
        jev_call.compute_budget_bytes(overhead_tokens=270, state_bytes=1000, question_bytes=[3000] * 20) == 2730,
        "the all-questions limit binds: 64,000 - 270 - 1,000 - 60,000 = 2,730, below the one-question 27,730",
    )
    check(
        jev_call.compute_budget_bytes(overhead_tokens=0, state_bytes=0, question_bytes=[1]) == 31999,
        "with nothing else, the budget is the 32,000 limit less the one question",
    )
    # The empty state is 46 bytes by hand count and a schema of 17 bytes replaces the 4 of `null`: 59. The questions are
    # 62, 64 and 65 bytes as one-question objects.
    check(
        jev_call.g_budget_bytes(settings, overhead_tokens=270, schema={"type": "object"}) == 31606,
        "G's budget from the settings: 32,000 - 270 - 59 - 65 = 31,606, the all-questions limit being looser",
    )
    check(
        jev_call.g_budget_bytes(settings, overhead_tokens=270, schema=None) == 31619,
        "with no schema the state is 46 bytes: 32,000 - 270 - 46 - 65 = 31,619",
    )
    check(
        refused(
            jev_call.CallError,
            lambda: jev_call.compute_budget_bytes(overhead_tokens=270, state_bytes=31700, question_bytes=[50]),
        ),
        "parts that leave no room (32,000 - 270 - 31,700 - 50 = -20) are refused, not a negative budget",
    )
    check(
        refused(jev_call.CallError, lambda: jev_call.compute_budget_bytes(overhead_tokens=1, state_bytes=1, question_bytes=[])),
        "a call with no question is refused",
    )  # fmt: skip

    print("\n-- the settings --")
    check(
        settings.model == "jev-test-0"
        and [(s.id, s.config) for s in settings.g_statements]
        == [("success_haiku", "haiku"), ("success_sonnet", "sonnet"), ("success_opus_low", "opus_low")]
        and settings.budget_bytes == {"jev": None, "jev_bolstered": None},
        "the parsed settings hold the model, G's statements in order and the null budgets",
    )
    check(
        jev_call.parse_settings({**synthetic_data(), "budget_bytes": {"jev": 5000, "jev_bolstered": 4000}}).budget_bytes
        == {"jev": 5000, "jev_bolstered": 4000},
        "filled budgets are read",
    )
    check(
        jev_call.g_call(Request("x" * 6000), jev_call.parse_settings({**synthetic_data(), "budget_bytes": {"jev": 5000, "jev_bolstered": None}})).cut_bytes
        == 1000,
        "G's call takes its budget from the settings when none is given",
    )  # fmt: skip

    def mutated(change: Callable[[dict], object]) -> Callable[[], object]:
        data = synthetic_data()
        change(data)
        return lambda: jev_call.parse_settings(data)

    refusals = {
        "a missing model": lambda d: d["jev"].pop("model"),
        "an empty model": lambda d: d["jev"].update(model=""),
        "a missing display name": lambda d: d["display_names"].pop("sonnet"),
        "an extra display name": lambda d: d["display_names"].update(opus="Largest"),
        "a missing statement": lambda d: d["g"]["success_statements"].pop(),
        "statements out of the ladder's order": lambda d: d["g"]["success_statements"].reverse(),
        "a statement ID that is not success_<config>": lambda d: d["g"]["success_statements"][0].update(id="haiku"),
        "an empty statement": lambda d: d["g"]["success_statements"][1].update(text=""),
        "a budget that is a string": lambda d: d["budget_bytes"].update(jev="9000"),
        "a budget that is a bool": lambda d: d["budget_bytes"].update(jev=True),
        "a budget below one": lambda d: d["budget_bytes"].update(jev=0),
        "a missing budget key": lambda d: d["budget_bytes"].pop("jev_bolstered"),
    }
    for what, change in refusals.items():
        check(refused(jev_call.CallError, mutated(change)), f"settings with {what} are refused")

    print("\n-- imports --")
    sources = {
        relative: ast.parse((PROJECT_ROOT / relative).read_text()) for relative in closure("routers/jev_call.py")
    }
    imported = set().union(*(imported_packages(tree) for tree in sources.values()))
    check(
        {"json", "hashlib"} <= imported and "routers" in imported,
        "the guard finds the imports it looks for: json, hashlib and routers",
    )
    check(
        imported_packages(ast.parse("import sklearn.linear_model\nfrom analysis import freeze\nimport harness"))
        == {"sklearn", "analysis", "harness"},
        "the guard finds an import of scikit-learn, analysis and harness in a module that has them",
    )
    allowed = set(sys.stdlib_module_names) | {"numpy", "routers"}
    check(
        not imported - allowed,
        f"the closure imports the standard library and numpy only ({sorted(imported - allowed)})",
    )
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, routers.jev_call; print(sorted(m for m in sys.modules if m.split('.')[0] in "
         "{'analysis', 'harness', 'sklearn', 'torch', 'transformers'}))"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    check(probe.returncode == 0 and probe.stdout.strip() == "[]",
          f"importing the module loads no analysis, harness, scikit-learn or torch module ({probe.stdout.strip()})")  # fmt: skip

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
