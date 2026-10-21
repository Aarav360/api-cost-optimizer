"""Check the rules router, E (PLAN_v4.md §8 3a; Amendment A9, items 4-6 and 13).

Run from the project root:  uv run python -m analysis.check_rules_router

On synthetic requests, with answers worked out by hand, plus the committed rates file and the router's source.
Never a test item, and no E number on dev.

- type inference, in A9 item 5's order: schema, option list, classification request with labels, code request,
  else open-ended;
- each cue: the calculation flag, the constraint count, the length, and the bucket they give at a cut point;
- fitting on hand-worked buckets, including the fold below 30 and a type with no items, and its refusals;
- the router: its router scores are its bucket's rates, and the decision rule reads them;
- the committed rates file's shape, and that `score` reads it;
- no Banking77 label in the rules router's source, with the benchmark's own labels as the known positive;
- the freeze (Amendment A9 item 17): the recorded module list is the import closure worked out from the router's
  imports, each recorded hash is the file's, and the guards can fail: a module imported later and a changed byte
  both show;
- the hook only (it needs the results stores, which CI lacks): the committed rates equal a refit from the train
  matrix, and the train task-inference diagnostic equals item 17's.
"""

import json
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from analysis.benchmark import read_rows
from analysis.fit_rules_router import diagnostic, rates_text, train_inputs
from analysis.freeze import RATES, closure, sha256
from analysis.plan import PLAN, amendment_item
from harness.client import PROJECT_ROOT
from harness.loaders import banking77_labels
from routers import rules
from routers.decision import choose
from routers.ladder import SCORED
from routers.request import Request
from routers.rules import CLASSIFICATION, CODE, MULTIPLE_CHOICE, OPEN_ENDED

RECORD = re.compile(r"`(routers/[\w./]+)`: `([0-9a-f]{64})`")  # a path and its sha256, as item 17 writes them

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object]) -> bool:
    try:
        call()
    except rules.FitError:
        return True
    return False


LETTERS = {"properties": {"answer": {"enum": ["A", "B", "C", "D"]}}}
LABELS = {"properties": {"answer": {"enum": ["alpha_one", "beta_two"]}}}
NO_ENUM = {"properties": {"answer": {"type": "string"}}}
CALC = "A 5 kg mass accelerates at 2 m/s. Which force acts on it?\nA. 10 N\nB. 7 N"  # digits with units
PLAIN = "Which city is the capital?\nA. Paris\nB. Rome"

TYPE_CASES = [
    # (what, request, type)
    (
        "a schema whose answer is a set of letters is multiple choice, whatever the prompt says",
        Request("Write a function that returns 1.", LETTERS),
        MULTIPLE_CHOICE,
    ),
    ("a schema whose answer is a set of other labels is classification", Request("Pick one.", LABELS), CLASSIFICATION),
    ("the schema is checked before the prompt: labels beat an option list", Request(PLAIN, LABELS), CLASSIFICATION),
    ("a schema with no set of values falls through to the prompt", Request(PLAIN, NO_ENUM), MULTIPLE_CHOICE),
    (
        "a schema with no set of values and a plain prompt is open-ended",
        Request("Tell me a joke.", NO_ENUM),
        OPEN_ENDED,
    ),
    ("a lettered option list is multiple choice", Request(PLAIN), MULTIPLE_CHOICE),
    ("options in the form (A) and (B) are multiple choice", Request("Pick.\n(A) one\n(B) two"), MULTIPLE_CHOICE),
    (
        "a roman-numeral list before the options doesn't hide them",
        Request("True?\nI. x\nII. y\n\nA. I only\nB. II only"),
        MULTIPLE_CHOICE,
    ),
    (
        "a multiple-choice request that contains code is multiple choice",
        Request("```python\nprint(1)\n```\nWhat does this print?\nA. 1\nB. 2"),
        MULTIPLE_CHOICE,
    ),
    (
        "a multiple-choice request that asks for a function is multiple choice",
        Request("Write a function that returns 1. Which one runs?\nA. def f(): return 1\nB. def f(): return 2"),
        MULTIPLE_CHOICE,
    ),
    ("a lone A line is not an option list", Request("A. Introduction\nWrite about cats."), OPEN_ENDED),
    (
        "a request to classify with a list of labels, one per line, is classification",
        Request("Classify the ticket into one of these:\n\nbilling\nshipping\nsupport\n\nTicket: late"),
        CLASSIFICATION,
    ),
    (
        "a request to categorize with an inline list is classification",
        Request("Categorize this ticket into one of: billing, shipping, or support."),
        CLASSIFICATION,
    ),
    (
        "a request to classify with no list of labels is open-ended",
        Request("Classify the mood of this sentence: I am happy."),
        OPEN_ENDED,
    ),
    (
        "a list of labels with no request to classify is open-ended",
        Request("Here are the days:\nmonday\ntuesday\nwednesday\nWrite a poem."),
        OPEN_ENDED,
    ),
    ("a request for a function is code", Request("Write a Python function that reverses a string."), CODE),
    (
        "a request for a function, worded another way, is code",
        Request("Please implement a recursive Fibonacci function."),
        CODE,
    ),
    ("a request for a code block is code", Request("Return the answer in a single ```python code block."), CODE),
    ("writing about a program is not asking for one", Request("Write a review of a chess program."), OPEN_ENDED),
    ("anything unrecognised is open-ended", Request("Tell me a joke."), OPEN_ENDED),
    ("an empty prompt is open-ended", Request(""), OPEN_ENDED),
]


def write_package(root: Path, modules: dict[str, str]) -> None:
    """A small `routers` package under root, from module names to their source."""
    (root / "routers").mkdir()
    for name, source in modules.items():
        (root / "routers" / f"{name}.py").write_text(source)


def padded(start: str, length: int) -> str:
    return start + "x" * (length - len(start))


def requests_for(prefix: str, count: int, make: Callable[[], Request]) -> dict[str, Request]:
    return {f"{prefix}{i}": make() for i in range(count)}


def outcomes(ids: list[str], **successes: int) -> dict[str, dict[str, float]]:
    """The first k items of ids succeed on a config given k; the others fail."""
    return {i: {config: float(n < successes[config]) for config in SCORED} for n, i in enumerate(ids)}


def main() -> int:
    print("-- type inference --")
    for what, request, kind in TYPE_CASES:
        check(rules.infer_type(request) == kind, what)

    print("\n-- the cues --")
    for text in (CALC, "What is 3 + 4?", "Solve x = 5 for y.", "Evaluate $\\frac{1}{2}$.", "A car goes 20 m/s.",
                 "Heat to 30°C.", "It costs 5 kg of flour."):  # fmt: skip
        check(rules.needs_calculation(text), f"calculation flag is yes for {text!r}")
    for text in ("In 1992 the film won three awards.", "Which of the 3 branches of government is largest?",
                 "She had 5 more apples.", "The ticket costs $5 and the popcorn $10.",
                 "Which is true?\nA. 5 kg\nB. 6 kg", "Name the 1st president."):  # fmt: skip
        check(not rules.needs_calculation(text), f"calculation flag is no for {text!r}")
    check(rules.constraint_sentences("Write an essay. It must be long. Do not use commas. Use at least 3 "
                                     "paragraphs and end your response with Done.") == 3,
          "the constraint count: must, do not, and at least + end your response with are three sentences")  # fmt: skip
    check(rules.constraint_sentences("Rules:\n- must be short\n- exactly two lines") == 2,
          "a line break ends a sentence for the constraint count")  # fmt: skip
    check(rules.constraint_sentences("Tell me a story about a dog.") == 0, "no constraint marker counts 0")
    check(rules.constraint_sentences("You MUST not stop. MUST be kind.") == 2, "markers match whatever the case")
    check(rules.cue_value(CLASSIFICATION, "abcd") == 4.0 and rules.cue_value(CODE, "abc") == 3.0,
          "the cue for classification and code is the prompt's length in characters")  # fmt: skip
    check(rules.cue_value(MULTIPLE_CHOICE, CALC) == 1.0 and rules.cue_value(MULTIPLE_CHOICE, PLAIN) == 0.0,
          "the cue for multiple choice is the 0 or 1 flag")  # fmt: skip
    check(rules.cue_value(OPEN_ENDED, "It must be long. Do not stop.") == 2.0,
          "the cue for open-ended is the constraint count")  # fmt: skip
    cuts = {CLASSIFICATION: 100.0, CODE: 50.0, OPEN_ENDED: 1.0}
    classify = Request("x", LABELS)
    for length, bucket in ((100, "classification/short"), (101, "classification/long")):
        check(rules.bucket_key(Request(padded("p", length), LABELS), cuts) == bucket,
              f"a classification prompt of {length} characters, cut at 100, is {bucket}")  # fmt: skip
    check(
        rules.bucket_key(Request(padded("Write a function ", 50)), cuts) == "code/short",
        "a code prompt at its cut is short",
    )
    check(
        rules.bucket_key(Request(padded("Write a function ", 51)), cuts) == "code/long",
        "a code prompt above its cut is long",
    )
    check(rules.bucket_key(Request("It must be brief."), cuts) == "open_ended/few", "one constraint, cut at 1, is few")
    check(rules.bucket_key(Request("It must be brief. Do not ramble."), cuts) == "open_ended/many", "two are many")
    check(rules.bucket_key(Request(CALC), cuts) == "multiple_choice/calc", "a calculation request is calc")
    check(rules.bucket_key(Request(PLAIN), cuts) == "multiple_choice/plain", "a request with no calculation is plain")
    check(rules.bucket_key(classify, {**cuts, CLASSIFICATION: 0.0}) == "classification/long", "cut points are read")

    print("\n-- fitting on hand-worked buckets --")
    check(rules.MIN_BUCKET_SIZE == 30, "the minimum bucket size is 30 train items (Amendment A9 item 4)")
    requests: dict[str, Request] = {}
    successes: dict[str, dict[str, float]] = {}

    def add(prefix: str, count: int, make: Callable[[], Request], **wins: int) -> None:
        new = requests_for(prefix, count, make)
        requests.update(new)
        successes.update(outcomes(list(new), **wins))

    # 40 calc and 30 plain: 30 is enough to stand alone.
    add("mc-calc", 40, lambda: Request(CALC), haiku=10, sonnet=20, opus_low=30)
    add("mc-plain", 30, lambda: Request(PLAIN), haiku=30, sonnet=15, opus_low=30)
    # 30 prompts of 100 characters and 30 of 200: the median is 150, so 30 on each side.
    add("cl-100", 30, lambda: Request(padded("p", 100), LABELS), haiku=24, sonnet=27, opus_low=30)
    add("cl-200", 30, lambda: Request(padded("p", 200), LABELS), haiku=12, sonnet=18, opus_low=24)
    # 29 of 100, 1 of 200, 29 of 300 characters: the median (the 30th of 59) is 200, so short has 30 and long 29.
    add("co-100", 29, lambda: Request(padded("Write a function ", 100)), haiku=15, sonnet=24, opus_low=30)
    add("co-200", 1, lambda: Request(padded("Write a function ", 200)), haiku=0, sonnet=0, opus_low=0)
    add("co-300", 29, lambda: Request(padded("Write a function ", 300)), haiku=5, sonnet=14, opus_low=29)
    # 31 with no constraint and 29 with two: the median is 0, so few has 31 and many 29.
    add("op-0", 31, lambda: Request("Tell me a story."), haiku=25, sonnet=28, opus_low=31)
    add("op-2", 29, lambda: Request("It must be brief. Do not ramble."), haiku=10, sonnet=20, opus_low=29)
    # The 100-character code items are the first 29 and the 200-character one failed everywhere, so the short code
    # bucket's 30 items are those 30: haiku 15 of 29 + 0, sonnet 24, opus_low 29 + 0.
    # Hand-worked rates, as k wins out of n items:
    expected = {
        "multiple_choice/calc": (40, False, (10 / 40, 20 / 40, 30 / 40)),
        "multiple_choice/plain": (30, False, (30 / 30, 15 / 30, 30 / 30)),
        "classification/short": (30, False, (24 / 30, 27 / 30, 30 / 30)),
        "classification/long": (30, False, (12 / 30, 18 / 30, 24 / 30)),
        "code/short": (30, False, (15 / 30, 24 / 30, 29 / 30)),
        "code/long": (29, True, (20 / 59, 38 / 59, 58 / 59)),  # folded: all 59 code items
        "open_ended/few": (31, False, (25 / 31, 28 / 31, 31 / 31)),
        "open_ended/many": (29, True, (35 / 60, 48 / 60, 60 / 60)),  # folded: all 60 open-ended items
    }
    rates = rules.fit(requests, successes)
    check(rates["cut_points"] == {CLASSIFICATION: 150.0, CODE: 200.0, OPEN_ENDED: 0.0},
          "the cut points are the types' train medians: 150, 200 and 0")  # fmt: skip
    check(set(rates["buckets"]) == set(expected), "there are eight buckets")
    for key, (count, folded, wins) in expected.items():
        bucket = rates["buckets"][key]
        check(bucket["count"] == count and bucket["folded"] is folded,
              f"{key}: {count} train items{', folded into its type' if folded else ''}")  # fmt: skip
        check(tuple(bucket["rates"][c] for c in SCORED) == wins, f"{key}: rates {tuple(round(w, 4) for w in wins)}")
    check(rates["types"]["code"]["count"] == 59 and rates["types"]["multiple_choice"]["count"] == 70,
          "each type holds its own count: 59 code items and 70 multiple-choice items")  # fmt: skip
    check(rules.fit(requests, successes) == rates, "fitting twice on the same data gives the same contents")
    check(json.loads(json.dumps(rates)) == rates, "the contents are plain data that survive a round trip through JSON")

    few = rules.fit(requests_for("a", 35, lambda: Request(PLAIN)), outcomes([f"a{i}" for i in range(35)],
                                                                          haiku=7, sonnet=14, opus_low=35))  # fmt: skip
    check(few["types"]["classification"] == {"count": 0, "rates": {"haiku": 0.2, "sonnet": 0.4, "opus_low": 1.0}},
          "a type with no train items takes the rates of all items")  # fmt: skip
    check(few["buckets"]["classification/short"]["folded"] and few["buckets"]["classification/short"]["count"] == 0,
          "an empty bucket is folded")  # fmt: skip
    check(few["buckets"]["multiple_choice/plain"]["count"] == 35 and not few["buckets"]["multiple_choice/plain"]["folded"],
          "35 plain multiple-choice items stand alone")  # fmt: skip

    ids = list(requests)[:3]
    check(refused(lambda: rules.fit({}, {})), "no requests are refused")
    check(refused(lambda: rules.fit({i: requests[i] for i in ids}, {i: successes[i] for i in ids[:2]})),
          "requests and successes that name different items are refused")  # fmt: skip
    check(refused(lambda: rules.fit({ids[0]: requests[ids[0]]}, {ids[0]: {"haiku": 1.0, "sonnet": 1.0}})),
          "a missing config is refused")  # fmt: skip
    check(refused(lambda: rules.fit({ids[0]: requests[ids[0]]}, {ids[0]: {**successes[ids[0]], "haiku": 1.5}})),
          "a success outside 0 to 1 is refused")  # fmt: skip

    print("\n-- the router --")
    router = rules.RulesRouter(rates)
    calc_router_scores = router.score(Request(CALC))
    check(
        calc_router_scores == {"haiku": 0.25, "sonnet": 0.5, "opus_low": 0.75},
        "a calc request gets the calc bucket's rates",
    )
    check(
        set(calc_router_scores) == set(SCORED),
        "the router scores each config below the frontier and never the frontier",
    )
    calc_router_scores["haiku"] = 0.0
    check(
        router.score(Request(CALC))["haiku"] == 0.25,
        "the router scores returned are a copy: changing them leaves the rates",
    )
    check(
        choose(router.score(Request(CALC)), 0.7) == "opus_low",
        "the decision rule reads the router scores: 0.7 goes to opus_low",
    )
    check(router.score(Request(padded("p", 120), LABELS)) == rates["buckets"]["classification/short"]["rates"],
          "a classification prompt of 120 characters, under the 150 cut, gets the short bucket's rates")  # fmt: skip
    check(router.score(Request(padded("Write a function ", 250))) == {"haiku": 20 / 59, "sonnet": 38 / 59, "opus_low": 58 / 59},
          "a request in a folded bucket gets its type's rates")  # fmt: skip

    print("\n-- the committed rates file --")
    committed = rules.load_rates()
    check(set(committed) == {"min_bucket_size", "cut_points", "types", "buckets"}, "the file holds four sections")
    check(committed["min_bucket_size"] == 30, "it records the minimum bucket size, 30")
    check(
        set(committed["cut_points"]) == {CLASSIFICATION, CODE, OPEN_ENDED},
        "it has a cut point for each type but multiple choice",
    )
    check(set(committed["buckets"]) == set(expected), "it has the eight buckets")
    check(all(set(b["rates"]) == set(SCORED) and all(0 <= r <= 1 for r in b["rates"].values())
              for b in [*committed["buckets"].values(), *committed["types"].values()]),
          "every bucket and type has a rate from 0 to 1 for each config below the frontier")  # fmt: skip
    check(all(b["folded"] == (b["count"] < 30) for b in committed["buckets"].values()),
          "a bucket is folded exactly when it has fewer than 30 train items")  # fmt: skip
    check(all(sum(b["count"] for k, b in committed["buckets"].items() if k.startswith(kind)) == t["count"]
              for kind, t in committed["types"].items()),
          "each type's bucket counts add up to the type's count")  # fmt: skip
    check((PROJECT_ROOT / "routers" / "rules_rates.json").read_text() == rates_text(committed),
          "the file is written the way the fitting script writes it (sorted keys)")  # fmt: skip
    committed_router = rules.RulesRouter(committed)
    check(rules.score(Request(CALC)) == committed_router.score(Request(CALC))
          == committed["buckets"]["multiple_choice/calc"]["rates"],
          "score(request) reads the committed file: a calc request gets its calc bucket's rates")  # fmt: skip

    print("\n-- no Banking77 label in the rules router --")
    sample = next(row for row in read_rows(split="train") if row["source"] == "banking77")
    labels = banking77_labels(sample["prompt"])
    source = (PROJECT_ROOT / "routers" / "rules.py").read_text().casefold()

    def found_in(text: str) -> list[str]:
        return [label for label in labels if label.casefold() in text.casefold()]

    check(len(labels) == 77, "the benchmark's own prompt lists 77 labels")
    check(found_in("\n".join(labels)) == labels, "known positive: the guard finds every label in the label list")
    check(
        found_in(source + "\n# card_arrival") == ["card_arrival"],
        "known positive: the guard finds a label added to the source",
    )
    check(found_in(source + "\n# Refund_not_showing_up") == ["Refund_not_showing_up"], "labels match whatever the case")
    check(not found_in(source), "the rules router's source holds no label")

    print("\n-- the freeze (Amendment A9 item 17) --")
    record = amendment_item(PLAN.read_text(), "A9", 17)
    recorded = dict(RECORD.findall(record))
    frozen_now = closure()
    check(sorted(p for p in recorded if p.endswith(".py")) == frozen_now,
          f"the recorded modules are the import closure worked out from the rules router's imports {frozen_now}")  # fmt: skip
    check(RATES in recorded and len(recorded) == len(frozen_now) + 1, "the rates file is recorded too, once")
    check(
        "routers/__init__.py" in recorded and "routers/rules.py" in recorded, "__init__ and the rules router are in it"
    )
    for relative, digest in recorded.items():
        check(sha256(PROJECT_ROOT / relative) == digest, f"{relative} hashes as recorded")
    check(re.search(r"median [\d.]+ us, p90 [\d.]+ us, on .*Python", record) is not None,
          "item 17 records the latency's median and p90 and the machine")  # fmt: skip
    check(
        "On this benchmark's templates" in record,
        "item 17 words the diagnostic as holding on this benchmark's templates",
    )

    # A guard proves it can fail: a changed byte, a module imported later, and a record that leaves one out.
    with tempfile.TemporaryDirectory() as tmp:
        for relative, digest in recorded.items():
            copy = Path(tmp) / Path(relative).name
            copy.write_bytes((PROJECT_ROOT / relative).read_bytes() + b" ")
            check(sha256(copy) != digest, f"a byte added to a copy of {relative} changes its hash")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_package(root, {"__init__": "", "rules": "import json\nfrom routers.a import x\n", "a": "from .b import y\n",
                             "b": "from . import d\n", "c": "", "d": "from routers import e\n", "e": ""})  # fmt: skip
        listed = [
            "routers/__init__.py",
            "routers/a.py",
            "routers/b.py",
            "routers/d.py",
            "routers/e.py",
            "routers/rules.py",
        ]
        check(closure("routers/rules.py", root) == listed,
              "on a synthetic package the closure follows absolute, relative and from-package imports, and leaves c out")  # fmt: skip
        (root / "routers" / "rules.py").write_text("import json\nfrom routers.a import x\nimport routers.c\n")
        late = closure("routers/rules.py", root)
        check(late == sorted([*listed, "routers/c.py"]) and late != listed,
              "a module imported later joins the closure, so a record made before it differs")  # fmt: skip
    short_record = [p for p in recorded if p.endswith(".py") and p != "routers/request.py"]
    check(short_record != frozen_now, "a record that leaves a module out differs from the closure")

    if (PROJECT_ROOT / "results" / "dev_matrix.sqlite").exists():
        print("\n-- the train matrix against the committed rates and item 17 (the hook only) --")
        requests_, successes_, tasks_ = train_inputs()
        check(rules.fit(requests_, successes_) == committed, "a refit from the train matrix equals the committed rates")
        check(f"{diagnostic(requests_, tasks_)}." in record, "the train task-inference diagnostic is item 17's")

    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
