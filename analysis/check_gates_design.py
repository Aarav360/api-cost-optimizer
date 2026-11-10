"""Recompute the train and dev facts Amendment A17 quotes and compare them with the plan's text (PLAN_v4.md; Amendment
A17, item 27; CODING_STANDARDS.md, "Numbers in documents").

Run from the project root:  uv run python -m analysis.check_gates_design

Calls no gate, no Jev and no model, and reads no test item: it reads the stored train and dev rows, train's MBPP+ hidden
tests (through the benchmark reader, so a test item is refused) and the committed chart files. A17's design review read
those outcomes, answers and tests before any gate existed, so this check may read them; gate code may not.

Each fact is recomputed from its source and turned into the sentence A17's text should hold. The text is read through
the amendment-item reader. A claim is ok when the recomputed sentence is in the text, and wrong otherwise.

A known positive nudges the number in each ok claim's sentence, in a copy of the text, and expects every nudge to be
caught. The provenance claim (dev's success counts per task and config equal chart v0's per-task config qualities) and
the median of the grader's sandbox runs (issue #67; A17 does not quote it) are compared with their sources directly.

The sections that need the stored results run only where `results/` holds the matrix.
"""

import ast
import json
import math
import re
import statistics
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace

from analysis import matrix
from analysis.benchmark import read_rows
from analysis.check_jev_settings_commit import SETTINGS, SMOKE_RECORD
from analysis.checking import git, refused
from analysis.plan import PLAN, amendment_item
from analysis.router_names import NAMED_ROUTERS, G
from analysis.simulator import always, load_mixes, simulate, totals
from harness.client import PROJECT_ROOT
from harness.collect import MAIN
from routers import jev_call
from routers.ladder import LADDER

problems: list[str] = []
PARETO_TASKS = PROJECT_ROOT / "figures" / "pareto_tasks.json"
CHART_V0_COMMIT = "dc73ffb"  # "Finish chart v0 and record Stage 3 (Amendment A8, item 12)"
SANDBOX_MEDIAN_S = 0.34  # issue #67's acceptance criterion; A17 does not quote it
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
SCHEMA_KEYS = {"type", "properties", "enum", "required", "additionalProperties"}


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


# --- the recomputations, on plain data ---------------------------------------------------------------------------


def assert_lines(prompt: str) -> list[str]:
    """The lines of a prompt that start with `assert`."""
    return [line for line in prompt.split("\n") if line.startswith("assert")]


def module_calls(line: str) -> set[str]:
    """The `module.function` names a line calls on a standard-library module: what the gate's import rule must import."""
    return {
        f"{node.func.value.id}.{node.func.attr}"
        for node in ast.walk(ast.parse(line))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in sys.stdlib_module_names
    }


def hidden_inputs(test: str) -> tuple[str | None, set[tuple[str, ...]]]:
    """The function a hidden test calls (`name(*inp)`) and each hidden input as its tuple of arguments, each argument
    as its syntax tree, so a match is exact in value and in kind: 1 is not 1.0, a tuple is not a list."""
    module = ast.parse(test)
    inputs = [
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "inputs"
        and isinstance(node.value, ast.List)
    ]
    called = [
        node.func.id
        for node in ast.walk(module.body[-1])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id != "assertion"
        and node.args
        and isinstance(node.args[0], ast.Starred)
    ]
    if not inputs or not called:
        return None, set()
    return called[0], {
        tuple(ast.dump(arg) for arg in element.elts)
        for element in inputs[0].elts
        if isinstance(element, (ast.List, ast.Tuple))
    }


def example_among_hidden(prompt: str, test: str) -> bool:
    """Whether the prompt's example call has literal arguments equal to one of the hidden test's inputs."""
    function, inputs = hidden_inputs(test)
    calls = [
        node
        for line in assert_lines(prompt)
        for node in ast.walk(ast.parse(line))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function
    ]
    return len(calls) == 1 and not calls[0].keywords and tuple(ast.dump(arg) for arg in calls[0].args) in inputs


def example_verbatim(prompt: str, test: str) -> bool:
    return any(line in test for line in assert_lines(prompt))


def rescued(successes: Mapping[str, Mapping[str, float]], failing: str, rescuer: str) -> tuple[int, int]:
    """How many items `failing` fails, and how many of those `rescuer` answers correctly."""
    failed = [item for item, by_config in successes.items() if by_config[failing] == 0]
    return len(failed), sum(successes[item][rescuer] == 1 for item in failed)


def schema_violation(answer: str, schema: dict) -> str | None:
    """Why an answer does not match its output schema, or None. Understands the keywords the benchmark's schemas use
    and refuses any other."""
    if unknown := set(schema) - SCHEMA_KEYS:
        raise ValueError(f"schema keywords this check does not understand: {sorted(unknown)}")
    try:
        value = json.loads(answer)
    except ValueError:
        return "not JSON"
    if not isinstance(value, dict):
        return "not an object"
    properties = schema["properties"]
    if schema.get("additionalProperties") is False and (extra := set(value) - set(properties)):
        return f"extra keys {sorted(extra)}"
    if missing := set(schema.get("required", ())) - set(value):
        return f"missing keys {sorted(missing)}"
    for key, spec in properties.items():
        if key in value and "enum" in spec and value[key] not in spec["enum"]:
            return f"{key} is not in its enum"
        if key in value and spec.get("type") == "string" and not isinstance(value[key], str):
            return f"{key} is not a string"
    return None


def stop_sentences(stops: list[tuple[str, str, str]]) -> list[str]:
    """For (item, config, stop reason) rows: the sentence parts A17 item 3 words them in, dearest config first."""
    verbs = {"refusal": "refused", "max_tokens": "hit its cap on"}
    parts = []
    for (item, reason), group in sorted(_grouped(stops).items()):
        configs = sorted(group, key=LADDER.index, reverse=True)
        names = " and ".join(f"`{config}`" for config in configs)
        parts.append(f"{names} {verbs.get(reason, f'stopped ({reason}) on')} `{item}`")
    return parts


def _grouped(stops: list[tuple[str, str, str]]) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for item, config, reason in stops:
        grouped.setdefault((item, reason), []).append(config)
    return grouped


# --- claims ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    where: str  # "preamble" or an item number as text
    what: str
    fragment: str  # the sentence the recomputed numbers make; it must be in the text


def judge(claim: Claim, text: str) -> str:
    """ok or wrong."""
    return "ok" if claim.fragment in text else "wrong"


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def nudged(fragment: str) -> str:
    """The fragment with its first number one more (one in its last decimal place)."""
    match = NUMBER.search(fragment)
    assert match, fragment
    digits = match.group().replace(",", "")
    places = len(digits.split(".")[1]) if "." in digits else 0
    bumped = f"{float(digits) + 10**-places:,.{places}f}"
    return fragment[: match.start()] + bumped + fragment[match.end() :]


@dataclass(frozen=True)
class Facts:
    stop_rows: int  # run-0 rows of train and dev
    stops: list[tuple[str, str, str]]  # (item, config, stop reason) of the run-0 rows that did not end their turn
    structured_tasks: set[str]
    unstructured_tasks: set[str]
    schema_violations: int  # end_turn answers of the structured tasks that don't match their schema
    mbpp_prompts: int
    one_assert: int  # MBPP+ prompts with exactly one assert line
    other_with_assert: int
    calls: Counter  # module.function -> MBPP+ prompts whose example calls it
    train_tests: int
    verbatim: int
    inputs_found: int
    train_items: int
    failed: int
    sonnet_rescued: int
    opus_low_rescued: int
    longest: tuple[int, int, str, str, str]  # characters, serialized bytes, item, config, split
    longest_prompt: int
    schema_reserve: int
    g_budget: int
    limits: dict[str, dict[str, float]]  # config -> mix -> the share of opus's cost a gate may not exceed
    haiku_cost: float  # realistic mix, dollars per 1,000
    opus_cost: float
    mbpp_share: float
    train_configs: int
    banking_tokens: float
    token_ratio: float  # mean input tokens of Banking77's answers over MMLU-Pro's, `sonnet`, run 0
    banking_sonnet_cost: float


def claims(f: Facts) -> list[Claim]:
    kb = round((f.longest[1] + f.longest_prompt + f.schema_reserve) / 1000)
    pct = lambda n: round(100 * n / f.failed)  # noqa: E731
    cut = {c: {m: round(100 * v) for m, v in by_mix.items()} for c, by_mix in f.limits.items()}
    haiku_stops = sum(config == "haiku" for _, config, _ in f.stops)
    named_calls = " or ".join(f"`{name}` ({n})" for name, n in f.calls.most_common())
    claims_ = [
        Claim(
            "3",
            "item 3: the rows the stops are counted in",
            f"has {WORDS[len(f.stops)]} such stops in {f.stop_rows:,} rows",
        ),
        *(Claim("3", f"item 3: who stopped ({part})", part) for part in stop_sentences(f.stops)),
        Claim("3", "item 3: haiku's stops", f"`haiku` has {WORDS[haiku_stops]}"),
        Claim(
            "3",
            "item 3: no schema where there is none",
            "IFEval and MBPP+ have no schema" if f.unstructured_tasks == {"ifeval", "mbppplus"} else "(other tasks)",
        ),
        Claim(
            "3",
            "item 3: every end_turn answer fits its schema",
            "structured outputs constrain every `end_turn` answer to its schema"
            if f.schema_violations == 0 and f.structured_tasks == {"banking77", "mmlu_pro"}
            else f"({f.schema_violations} answers off their schema)",
        ),
        Claim(
            "4",
            "item 4: assert lines",
            f"Each of the {f.mbpp_prompts} MBPP+ train and dev prompts has exactly one such line, and no other prompt has any"
            if f.one_assert == f.mbpp_prompts and f.other_with_assert == 0
            else f"({f.one_assert} with one, {f.other_with_assert} others)",
        ),
        Claim(
            "4",
            "item 4: example tests that need an import",
            f"{sum(f.calls.values())} of the {f.mbpp_prompts} example tests call {named_calls}",
        ),
        Claim(
            "4",
            "item 4: example inputs among the hidden inputs",
            f"verbatim in {'none' if f.verbatim == 0 else f.verbatim} of the {f.train_tests} train items' hidden tests, "
            f"while its input is among the hidden inputs for {f.inputs_found} of them",
        ),
        Claim(
            "5",
            "item 5: the longest answer",
            f"{f.longest[0]:,} characters ({f.longest[1]:,} serialized bytes; `{f.longest[2]}`, `{f.longest[3]}`, {f.longest[4]})",
        ),
        Claim(
            "5", "item 5: the longest train prompt", f"the longest train prompt ({f.longest_prompt:,} serialized bytes)"
        ),
        Claim("5", "item 5: the schema reserve", f"A16 item 8's {f.schema_reserve:,}-byte schema reserve"),
        Claim("5", "item 5: the total", f"that comes to about {kb} KB, under G's budget of {f.g_budget:,} bytes"),
        Claim(
            "6",
            "item 6: Banking77's input",
            f"the input alone is about {round(f.banking_tokens, -3):,.0f} tokens, about ${f.banking_sonnet_cost:.3f}, "
            "already what `sonnet` charges to answer",
        ),
        Claim("8", "item 8: MBPP+'s share", f"MBPP+ carries {round(100 * f.mbpp_share)}% of the realistic mix"),
        Claim(
            "9",
            "item 9: the rescue rates",
            f"Of the {f.failed} train items `haiku` fails, `sonnet` answers {f.sonnet_rescued} correctly ({pct(f.sonnet_rescued)}%) "
            f"and `opus_low` {f.opus_low_rescued} ({pct(f.opus_low_rescued)}%)",
        ),
        Claim(
            "9",
            "item 9: the limits",
            f"about {cut['sonnet']['uniform']}% (uniform) and {cut['sonnet']['realistic']}% (realistic) for `sonnet`, "
            f"and about {cut['opus_low']['uniform']}% and {cut['opus_low']['realistic']}% for `opus_low`",
        ),
        Claim(
            "10",
            "item 10: the strictest step's cost",
            f"about ${f.haiku_cost:.2f} for `haiku`, plus about $0.04 for the gate, plus 95% of A's ${f.opus_cost:.2f}, "
            f"comes to about ${f.haiku_cost + 0.04 + 0.95 * f.opus_cost:.2f}",
        ),
        Claim(
            "20",
            "item 20: Banking77's prompts against MMLU-Pro's",
            f"Banking77's prompts are about {f.token_ratio:.1f}× as long as MMLU-Pro's, in mean input tokens",
        ),
        Claim("12", "item 12: the train gate runs", f"about {f.train_items} items × {f.train_configs} configs"),
    ]
    return claims_


def verdicts(all_claims: list[Claim], texts: dict[str, str]) -> list[tuple[str, Claim]]:
    return [(judge(claim, texts[claim.where]), claim) for claim in all_claims]


# --- reading the sources -----------------------------------------------------------------------------------------


def gather() -> Facts:
    import sqlite3
    from contextlib import closing

    columns = "item_id, split, source, config, run, stop_reason, status, answer, input_tokens, detail"
    with closing(sqlite3.connect(MAIN.matrix.as_uri() + "?mode=ro", uri=True)) as conn:
        rows = [
            dict(zip(columns.split(", "), row)) for row in conn.execute(f"SELECT {columns} FROM rows WHERE run = 0")
        ]
    benchmark = {row["item_id"]: row for row in read_rows()}  # train and dev only; a test item is refused here
    store = matrix.load()
    train, dev = store.split("train"), store.split("dev")

    stops = [(r["item_id"], r["config"], r["stop_reason"]) for r in rows if r["stop_reason"] != "end_turn"]
    structured = {r["source"] for r in benchmark.values() if "output_schema" in r}
    violations = sum(
        schema_violation(r["answer"], benchmark[r["item_id"]]["output_schema"]) is not None
        for r in rows
        if r["stop_reason"] == "end_turn" and "output_schema" in benchmark[r["item_id"]]
    )
    mbpp = [r for r in benchmark.values() if r["source"] == "mbppplus"]
    train_mbpp = [r for r in mbpp if r["split"] == "train"]
    successes: dict[str, dict[str, float]] = {}
    for (item, config), cell in train.cells.items():
        successes.setdefault(item, {})[config] = cell.success
    failed, sonnet_rescued = rescued(successes, "haiku", "sonnet")
    opus_low_rescued = rescued(successes, "haiku", "opus_low")[1]
    answered = [r for r in rows if r["answer"] is not None]
    longest_row = max(answered, key=lambda r: len(r["answer"]))
    mixes = load_mixes()
    dev_totals = {c: {m: totals(simulate(dev, always(dev, c)), mixes[m]) for m in mixes} for c in LADDER}
    banking = [r for r in rows if r["source"] == "banking77" and r["config"] == "sonnet"]
    banking_cost = [train.cells.get((r["item_id"], "sonnet")) or dev.cells[(r["item_id"], "sonnet")] for r in banking]
    settings = json.loads((PROJECT_ROOT / SETTINGS).read_text(encoding="utf-8"))
    smoke = json.loads((PROJECT_ROOT / SMOKE_RECORD).read_text(encoding="utf-8"))
    return Facts(
        stop_rows=len(rows),
        stops=stops,
        structured_tasks=structured,
        unstructured_tasks={r["source"] for r in benchmark.values()} - structured,
        schema_violations=violations,
        mbpp_prompts=len(mbpp),
        one_assert=sum(len(assert_lines(r["prompt"])) == 1 for r in mbpp),
        other_with_assert=sum(bool(assert_lines(r["prompt"])) for r in benchmark.values() if r["source"] != "mbppplus"),
        calls=Counter(name for r in mbpp for name in module_calls(assert_lines(r["prompt"])[0])),
        train_tests=len(train_mbpp),
        verbatim=sum(example_verbatim(r["prompt"], r["expected_answer"]["test"]) for r in train_mbpp),
        inputs_found=sum(example_among_hidden(r["prompt"], r["expected_answer"]["test"]) for r in train_mbpp),
        train_items=len(train.items),
        failed=failed,
        sonnet_rescued=sonnet_rescued,
        opus_low_rescued=opus_low_rescued,
        longest=(
            len(longest_row["answer"]),
            sum(jev_call.serialized_size(c) for c in longest_row["answer"]),
            longest_row["item_id"],
            longest_row["config"],
            longest_row["split"],
        ),
        longest_prompt=max(
            sum(jev_call.serialized_size(c) for c in r["prompt"]) for r in benchmark.values() if r["split"] == "train"
        ),
        schema_reserve=smoke["longest_schema_bytes"],
        g_budget=settings["budget_bytes"][NAMED_ROUTERS[G]],
        limits={
            c: {m: 1 - dev_totals[c][m].cost_per_1k / dev_totals["opus"][m].cost_per_1k for m in mixes}
            for c in ("sonnet", "opus_low")
        },
        haiku_cost=dev_totals["haiku"]["realistic"].cost_per_1k,
        opus_cost=dev_totals["opus"]["realistic"].cost_per_1k,
        mbpp_share=mixes["realistic"]["mbppplus"],
        train_configs=len({config for _, config in train.cells}),
        banking_tokens=statistics.mean(r["input_tokens"] for r in banking),
        token_ratio=statistics.mean(r["input_tokens"] for r in banking)
        / statistics.mean(r["input_tokens"] for r in rows if r["source"] == "mmlu_pro" and r["config"] == "sonnet"),
        banking_sonnet_cost=statistics.mean(c.cost for c in banking_cost),
    )


def a17_texts() -> dict[str, str]:
    plan = PLAN.read_text(encoding="utf-8")
    texts = {str(n): amendment_item(plan, "A17", n) for n in range(1, 30)}
    texts["preamble"] = plan[plan.index("## Amendment A17") :].split("\n1. **", 1)[0]
    return texts


# --- the checks --------------------------------------------------------------------------------------------------

PROMPT = 'Write f.\n\nassert f(1,"a")==3\n\nTask: say assert in prose\n assert not_a_line\nReturn it.'
HIDDEN = (
    "inputs = [[1, 'a'], [2, 'b']]\nresults = [3, 4]\n"
    "for i, (inp, exp) in enumerate(zip(inputs, results)):\n    assertion(f(*inp), exp, 0)\n"
)


def with_example(example: str) -> str:
    return f"Write f.\n\n{example}\n\nTask: x"


def recomputations() -> None:
    print("-- the recomputations, on made-up data --")
    check(assert_lines(PROMPT) == ['assert f(1,"a")==3'], "an assert line is a line that starts with `assert`, not a mention or an indent")  # fmt: skip
    check(assert_lines("Write f.\nTask: nothing") == [], "a prompt with no such line has none")
    check(module_calls("assert math.isclose(f(1), 2.0, rel_tol=0.1)") == {"math.isclose"}, "a call on a standard-library module is found")  # fmt: skip
    check(module_calls("assert sys.getsizeof(f(1)) == 56") == {"sys.getsizeof"}, "so is sys.getsizeof")
    check(module_calls("assert f(1) == 2") == set() and module_calls("assert np.allclose(f(1), 2)") == set(),
          "a plain call, and a call on a module outside the standard library, are not")  # fmt: skip
    check(
        example_among_hidden(PROMPT, HIDDEN),
        "an example whose arguments are a hidden input is found, quote style aside",
    )
    tupled = HIDDEN.replace("[[1, 'a'], [2, 'b']]", "[(1, 'a'), (2, 'b')]")
    check(example_among_hidden(PROMPT, tupled), "a hidden input written as a tuple of arguments is read like a list")
    check(example_among_hidden(with_example("assert f(2,'b')==4"), HIDDEN), "wherever it sits among the hidden inputs")
    for what, example in (
        ("a float for an int", "assert f(1.0,'a')==3"),
        ("a tuple for the two arguments", "assert f((1,'a'))==3"),
        ("a list for the two arguments", "assert f([1,'a'])==3"),
        ("an input not in the list", "assert f(3,'c')==3"),
        ("a keyword argument", "assert f(1,x='a')==3"),
        ("another function", "assert g(1,'a')==3"),
    ):
        check(not example_among_hidden(with_example(example), HIDDEN), f"known negative: {what} is not a match")
    check(example_among_hidden(with_example("assert f(1,'a')==3"), "x = 1\nassertion(f(*inp), 1, 0)\n") is False, "a hidden test with no `inputs` matches nothing")  # fmt: skip
    check(example_verbatim(with_example("assert f(1)==2"), "x\nassert f(1)==2\ny") and not example_verbatim(PROMPT, HIDDEN),
          "an example is verbatim in a test only if the line is in it")  # fmt: skip
    ladder = {"a": {"haiku": 0, "sonnet": 1, "opus_low": 1}, "b": {"haiku": 0, "sonnet": 0, "opus_low": 1},
              "c": {"haiku": 1, "sonnet": 1, "opus_low": 1}, "d": {"haiku": 0, "sonnet": 0, "opus_low": 0}}  # fmt: skip
    check(rescued(ladder, "haiku", "sonnet") == (3, 1) and rescued(ladder, "haiku", "opus_low") == (3, 2),
          "of 3 items haiku fails, sonnet answers 1 and opus_low 2")  # fmt: skip
    schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["A", "B"]}}, "required": ["answer"], "additionalProperties": False}  # fmt: skip
    check(schema_violation('{"answer":"A"}', schema) is None, "an answer in its enum fits its schema")
    for what, answer in (("outside its enum", '{"answer":"C"}'), ("missing a key", "{}"), ("with an extra key", '{"answer":"A","x":1}'),
                         ("not JSON", "A"), ("not an object", '"A"')):  # fmt: skip
        check(schema_violation(answer, schema) is not None, f"known negative: an answer {what} is a violation")
    check(refused(ValueError, lambda: schema_violation("{}", {"minimum": 0})), "a schema keyword the check doesn't understand is refused")  # fmt: skip
    stops = [("m:1", "opus", "refusal"), ("m:1", "opus_low", "refusal"), ("m:2", "sonnet", "max_tokens")]
    check(stop_sentences(stops) == ["`opus` and `opus_low` refused `m:1`", "`sonnet` hit its cap on `m:2`"],
          "stops are worded as A17 item 3 words them, the dearest config first")  # fmt: skip
    check(nudged("about 1,234 tokens") == "about 1,235 tokens" and nudged("$0.29 for") == "$0.30 for" and nudged("a 9 b") == "a 10 b",
          "a nudge adds one in the number's last place")  # fmt: skip


def claim_machinery() -> None:
    print("\n-- the claims, ok and wrong --")
    claim = Claim("3", "some claim", "has 3 stops")
    check(judge(claim, "it has 3 stops.") == "ok", "a claim whose sentence is in the text is ok")
    check(judge(claim, "it has 4 stops.") == "wrong", "known negative: a text with another number is wrong")


def sample_facts() -> Facts:
    return Facts(
        stop_rows=100, stops=[("m:1", "opus", "refusal")], structured_tasks={"banking77", "mmlu_pro"},
        unstructured_tasks={"ifeval", "mbppplus"}, schema_violations=0, mbpp_prompts=10, one_assert=10,
        other_with_assert=0, calls=Counter({"math.isclose": 2}), train_tests=6, verbatim=0, inputs_found=5,
        train_items=20, failed=8, sonnet_rescued=3, opus_low_rescued=4, longest=(1500, 1600, "ifeval:1", "haiku", "train"),
        longest_prompt=300, schema_reserve=100, g_budget=2000,
        limits={"sonnet": {"uniform": 0.5, "realistic": 0.6}, "opus_low": {"uniform": 0.2, "realistic": 0.1}},
        haiku_cost=0.3, opus_cost=8.0, mbpp_share=0.65, train_configs=3, banking_tokens=1954.0, token_ratio=3.45, banking_sonnet_cost=0.0041,
    )  # fmt: skip


def known_positive(all_claims: list[Claim], texts: dict[str, str], what: str) -> None:
    """Every ok claim's sentence, with its first number nudged in a copy of the text, is caught."""
    missed = []
    ok = [claim for status, claim in verdicts(all_claims, texts) if status == "ok" and NUMBER.search(claim.fragment)]
    for claim in ok:
        changed = {**texts, claim.where: texts[claim.where].replace(claim.fragment, nudged(claim.fragment), 1)}
        if judge(claim, changed[claim.where]) == "ok":
            missed.append(claim.what)
    check(bool(ok) and not missed, f"known positive: a number nudged in a copy of {what} fails each of its {len(ok)} claims" + (f" (missed: {missed})" if missed else ""))  # fmt: skip


def synthetic_amendment() -> None:
    print("\n-- the claims against a made-up amendment --")
    f = sample_facts()
    made = claims(f)
    check(len({c.what for c in made}) == len(made), "every claim has its own description")
    texts = {c.where: "" for c in made}
    for c in made:
        texts[c.where] += f"{c.fragment}.\n"
    texts["preamble"] = ""
    check(
        all(status == "ok" for status, _ in verdicts(made, texts)),
        "a text made from the recomputed sentences is all ok",
    )
    known_positive(made, texts, "a made-up amendment")
    check(claims(replace(f, verbatim=2))[7].fragment.startswith("verbatim in 2 of"), "an example verbatim in tests is worded by its count")  # fmt: skip
    check("(3 with one, 0 others)" in claims(replace(f, one_assert=3))[5].fragment, "a prompt with no assert line makes the assert claim unsayable")  # fmt: skip


def stored_facts() -> None:
    print("\n-- A17's facts against the stored train and dev rows --")
    texts = a17_texts()
    f = gather()
    all_claims = claims(f)
    for status, claim in verdicts(all_claims, texts):
        check(status == "ok", f"{claim.what}: {claim.fragment}")
    check(f.stop_rows == 5949 and f.stop_rows != 6549, "the run-0 rows are 5,949; 6,549 is the store's rows in all three runs")  # fmt: skip
    known_positive(all_claims, texts, "A17")


def sandbox_and_provenance() -> None:
    import sqlite3
    from contextlib import closing

    print("\n-- the sandbox's run times and the provenance claim --")
    with closing(sqlite3.connect(MAIN.matrix.as_uri() + "?mode=ro", uri=True)) as conn:
        details = [
            json.loads(d) for (d,) in conn.execute("SELECT detail FROM rows WHERE run = 0 AND source = 'mbppplus'")
        ]
    durations = [d["duration_s"] for d in details if "duration_s" in d]
    check(len(durations) == 857 and statistics.median(durations) == SANDBOX_MEDIAN_S,
          f"the grader's sandbox ran {len(durations)} times on run 0's MBPP+ answers, with a median of {statistics.median(durations)} s")  # fmt: skip
    dev = matrix.load().split("dev")
    counts: dict[str, dict[str, float]] = {}
    for (item, config), cell in dev.cells.items():
        by_config = counts.setdefault(dev.items[item].task, {})
        by_config[config] = by_config.get(config, 0) + cell.success
    sizes = Counter(item.task for item in dev.items.values())
    sources = {"the committed figures/pareto_tasks.json": PARETO_TASKS.read_text(encoding="utf-8")}
    shallow = git("rev-parse", "--is-shallow-repository").stdout.strip() != "false"
    shown = git("show", f"{CHART_V0_COMMIT}:figures/pareto_tasks.json")
    if shallow or shown.returncode != 0:
        print("      (chart v0's commit is not in this clone: its copy is skipped)")
    else:
        sources[f"chart v0 ({CHART_V0_COMMIT})"] = shown.stdout
    for name, source in sources.items():
        panels = json.loads(source)["panels"]
        same = set(panels) == set(sizes) and all(
            panels[task]["items"] == sizes[task]
            and all(
                math.isclose(panel["quality"]["value"], counts[task][config] / sizes[task], rel_tol=0, abs_tol=1e-12)
                for config, panel in panels[task]["configs"].items()
            )
            and set(panels[task]["configs"]) == set(counts[task])
            for task in panels
        )
        check(same, f"dev's success counts per task and config equal {name}'s per-task config qualities")


def provenance_wording() -> None:
    texts = a17_texts()
    check("chart v0 already published in `figures/pareto_tasks.json`" in texts["preamble"].replace("\n", " "),
          "the provenance paragraph says the counts are chart v0's published qualities")  # fmt: skip
    check("`analysis/check_gates_design.py` (item 27)" in texts["preamble"] and "analysis/check_gates_design.py" in texts["27"],
          "A17 names this check in its provenance and in item 27")  # fmt: skip


def main() -> int:
    recomputations()
    claim_machinery()
    synthetic_amendment()
    provenance_wording()
    if MAIN.matrix.exists():
        stored_facts()
        sandbox_and_provenance()
    else:
        print("\n      (no results matrix here: the sections that recompute from it are skipped)")
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
