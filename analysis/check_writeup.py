"""Hold WRITEUP.md's skeleton: the declared parts present and in order against the skeleton's commit, and every number in
it against the committed tables (PLAN_v4.md; Amendment A19, items 6 and 11; CODING_STANDARDS.md, "Numbers in documents").

Run from the project root:  uv run python -m analysis.check_writeup

Reads no test item and calls nothing. Two holds:

- **The declared parts.** A19 item 11 fixes what the write-up declares before the go: the section order, the headline
  as a template, the declared questions in order, the replication summary's form, the expected misses, the limitations
  and a short Future work. Each is a heading of WRITEUP.md, and `DECLARED` lists them in A19's order. The skeleton's
  commit is the first commit that added `MARKER` to WRITEUP.md (a search of the history through
  `analysis/checking.py`, so no commit's hash is written down, and one that doesn't exist yet can't be named wrongly).
  That commit's headings must be `DECLARED`, and the file now must hold them in the same order. Sections added later
  pass, and a declared part removed, renamed or moved fails (item 11: "nothing declared is removed or reordered, and the
  run may add sections"). Before the commit exists, as when the hook runs for the commit that makes it, the working
  tree's file stands in for it; in a clone without history the history part is skipped and the file is held against
  `DECLARED` alone. CI's checkout fetches the whole history (`fetch-depth: 0` in `.github/workflows/check.yml`, which
  `ci_checkout_check` holds), so the history part runs there too: the Stage 8 review found it skipped in CI.
- **The numbers.** Each number-bearing sentence of the skeleton is a `Fragment`, built here from the committed tables
  (`analysis/final_design.py`, `figures/*.json`, `data/benchmark_manifest.json`, `configs/task_mix.yaml`), and must stand
  in its section. After the fragments are taken out, no digit may remain outside a citation (`A19 item 6`, `§5`), a code
  span, a comment or a {placeholder}: a number no check holds fails. The results PR fills in numbers, and adds a
  `Fragment` for each, so that this scan finds none of them unheld (item 11, "Every number"). After the go (#95) these are
  `results_fragments`: every line of every paragraph `analysis/writeup_results.py` builds from `figures/tables_test.json`
  (test's tables), `analysis/final_record.json` (the live replay) and `figures/table_v4.json` (dev's points beside the headline),
  plus the summary's headline and answers. The skeleton's own holds (`fragments`) stay apart, since the skeleton's commit
  can't hold the results' lines. The recounts of those tables from their own rows are in `analysis/check_final_results.py`;
  here are the words: the formats and the counts worked by hand on made-up cells, a digit changed in the headline, a
  comparison cell, a policy row, a replication count and a replay figure each found, and the README quoting the headline.
"""

import json
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from analysis import writeup_results as wr
from analysis.checking import full_history, git, git_bytes
from analysis.final_design import HEADLINE_MIX, HEADLINE_TARGET, MIXES, expected_misses, headline
from analysis.plan import PLAN, amendment_item
from analysis.references import TARGETS
from analysis.router_names import CASCADES, JEV_GATE, E, F, cascade_section
from analysis.simulator import load_mixes
from analysis.writeup_results import built, load
from harness.client import PROJECT_ROOT

WRITEUP = PROJECT_ROOT / "WRITEUP.md"
NOTES = PROJECT_ROOT / "NOTES.md"
PARETO = PROJECT_ROOT / "figures" / "pareto.json"
TABLE_V4 = PROJECT_ROOT / "figures" / "table_v4.json"
MANIFEST = PROJECT_ROOT / "data" / "benchmark_manifest.json"
MARKER = "<!-- skeleton:"  # the comment that opens the skeleton; the first commit adding it is the skeleton's commit

Heading = tuple[int, str]  # a heading's level (number of #) and its text

# A19 item 11's declared parts, in order. The list is written here, apart from WRITEUP.md, so that an edit of one is
# caught by the other, and the skeleton's commit holds both.
DECLARED: tuple[Heading, ...] = (
    (2, "Summary"),
    (3, "The headline"),
    (3, "The answers to the declared questions"),
    (2, "Design"),
    (2, "Expected misses"),
    (2, "Results"),
    (3, "The headline in full"),
    (3, "Misses against the expectation"),
    (3, "The declared questions"),
    (4, "haiku/Jev against G"),
    (4, "haiku/Jev against G+"),
    (4, "The other cascades against the routers"),
    (4, "What the gate adds to a router"),
    (4, "What the router adds to a cascade"),
    (4, "What a paid gate adds over the free checks"),
    (4, "The Jev gate against the Sonnet gate"),
    (3, "The replication summary"),
    (3, "Every policy at its operating points"),
    (3, "The hindsight oracle and the random-mixture envelope"),
    (3, "The live replay"),
    (2, "Found after test was seen"),
    (2, "Discussion"),
    (2, "Limitations"),
    (3, "Selection on dev"),
    (3, "IFEval and MBPP+ are only indicative"),
    (3, "One run of test"),
    (3, "The oracle is an optimistic upper bound"),
    (2, "Future work"),
    (3, "A benchmark where the configs are further apart"),
)

# Each declared question's heading and the comparison group in table v4 that holds its comparisons (A17 item 22). The
# first two are the headline pair, taken out of "cascades against routers"; the rest follow item 22's own order.
QUESTIONS = {
    "haiku/Jev against G": None,
    "haiku/Jev against G+": None,
    "The other cascades against the routers": "cascades_against_routers",
    "What the gate adds to a router": "gate_adds",
    "What the router adds to a cascade": "router_adds",
    "What a paid gate adds over the free checks": "paid_gate_over_free",
    "The Jev gate against the Sonnet gate": "jev_against_sonnet",
}

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


# --- reading a document ---------------------------------------------------------------------------------------------

COMMENT = re.compile(r"<!--.*?-->", re.S)
FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.S | re.M)
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)


def prose(text: str) -> str:
    """The text without comments and fenced code, which hold no declared heading and no number the write-up states."""
    return FENCE.sub("", COMMENT.sub("", text))


def headings(text: str) -> list[Heading]:
    """Every heading of `text` below the title, as (level, text), in order: the title is the researcher's to change, and
    no part is declared in it. A heading inside a comment or a fence is no heading."""
    return [(len(m.group(1)), m.group(2)) for m in HEADING.finditer(prose(text)) if len(m.group(1)) > 1]


def tidy(text: str) -> str:
    """`text` on one line, with its runs of whitespace as single spaces, so a sentence found wrapped is found."""
    return " ".join(text.split())


def section_body(text: str, heading: Heading) -> str | None:
    """The text under the first heading `heading`, up to the next heading of any level; None if there is none."""
    body = prose(text)
    for m in HEADING.finditer(body):
        if (len(m.group(1)), m.group(2)) == heading:
            rest = body[m.end() :]
            following = HEADING.search(rest)
            return rest[: following.start()] if following else rest
    return None


def held_in_order(declared: Sequence[Heading], text: str) -> list[str]:
    """What is wrong with `text` against `declared`: each declared heading that is missing, or is there but out of
    order. A heading `text` has besides the declared ones is no fault."""
    present = headings(text)
    wrong, after = [], -1
    for part in declared:
        try:
            after = present.index(part, after + 1)
        except ValueError:
            wrong.append(f"{'#' * part[0]} {part[1]}: " + ("out of order" if part in present else "missing"))
    return wrong


# --- the numbers ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Fragment:
    """A number-bearing phrase of the skeleton, built from the committed outputs, and the section it must stand in."""

    section: Heading
    text: str


# What a document may say with digits that is no quantity of the study: a citation of the plan's items, a section sign,
# an amendment, the plan's file name, a table's version, a stage's name, a unit and the null value an interval
# excludes. Code spans, comments and placeholders are taken out apart.
CITATION = re.compile(
    r"""A\d+                                      # an amendment
        | items?\s+\d+(?:\.\d+)?(?:(?:,\s*|\s+and\s+|\s+to\s+)\d+(?:\.\d+)?)*   # item 24, items 3 and 4, item 26.5
        | §\d+(?:\.\d+)?                          # §5
        | PLAN_v\d+\.md
        | \bv\d+\b                                # tables v3 and v4
        | \bStage\s+\d+\b                         # a stage's name
        | \bper\s+1k\b                            # a unit: dollars per 1k requests
        | \bexclud\w*\s+0\b                       # the null value an interval excludes
        | \bBanking77\b                           # a dataset's name""",
    re.X,
)
CODE_SPAN = re.compile(r"`[^`\n]*`")
PLACEHOLDER = re.compile(r"\{[^{}]*\}", re.S)
LIST_MARKER = re.compile(r"^[ \t]*\d+\.[ \t]", re.M)
NUMBER = re.compile(r"[$]?\d[\d,]*(?:\.\d+)?%?")


def missing_fragments(fragments: Sequence[Fragment], text: str) -> list[str]:
    """The fragments that don't stand in their section of `text`."""
    wrong = []
    bodies: dict[
        Heading, str | None
    ] = {}  # each section's tidied body, read once: the results hold about a thousand lines
    for fragment in fragments:
        if fragment.section not in bodies:
            body = section_body(text, fragment.section)
            bodies[fragment.section] = None if body is None else tidy(body)
        body = bodies[fragment.section]
        if body is None or tidy(fragment.text) not in body:
            wrong.append(f"{fragment.section[1]}: {fragment.text!r}")
    return wrong


def unheld_numbers(fragments: Sequence[Fragment], text: str) -> list[str]:
    """The numbers left in `text` once its held fragments, citations, code spans, comments and placeholders are taken
    out: each is a number no check holds."""
    left = prose(text)
    for stripped in (PLACEHOLDER, CODE_SPAN, LIST_MARKER):
        left = stripped.sub(" ", left)
    left = tidy(left)
    for fragment in sorted(fragments, key=lambda f: -len(f.text)):
        left = left.replace(tidy(fragment.text), " ")
    return NUMBER.findall(CITATION.sub(" ", left))


def share(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def target_list(targets: Sequence[str]) -> str:
    """'100%, 99% and 97%'."""
    return f"{', '.join(targets[:-1])} and {targets[-1]}"


def fragments() -> list[Fragment]:
    """Every number the skeleton states, in the words it states it, from the committed tables."""
    misses, head = expected_misses(), headline()
    v4 = json.loads(TABLE_V4.read_text(encoding="utf-8"))
    pareto = json.loads(PARETO.read_text(encoding="utf-8"))["panels"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]
    targets = [f"{t:.0%}" for t in TARGETS]
    mixes = load_mixes()
    design, expected = (2, "Design"), (2, "Expected misses")
    out = [
        Fragment((3, "The headline"), f"the {HEADLINE_MIX} mix at the {HEADLINE_TARGET} target"),
        Fragment(
            (3, "The headline"),
            f"On dev that is {CASCADES[head.best]}, and on dev that point saved {share(head.saves)} of A's cost and met "
            f"its target in {share(head.met, 2)} of dev's resamples",
        ),
        Fragment((3, "The headline"), f"The {MIXES[1]} mix and the {targets[-1]} target are reported beside it"),
        Fragment(
            (3, "The headline"),
            f"two-thirds MBPP+ ({share(mixes[HEADLINE_MIX]['mbppplus'], 0)}), the smallest test split",
        ),
        Fragment(design, f"quality targets of {target_list(targets)} of A's quality"),
    ]
    by_router = ", ".join(f"{letter} {n}" for letter, n in misses.routers.items())
    out += [
        Fragment(
            expected,
            f"{misses.total} operating points route: {misses.routed} of the cascades' {misses.cascades_every} and "
            f"{sum(misses.routers.values())} of the routers' {misses.routers_every} ({by_router})",
        ),
        Fragment(
            expected,
            f"Over the {misses.covered} that tables v3 and v4 cover (the cascades', G's and G+'s), dev's own resamples "
            f"give about {round(misses.dev_misses)} expected misses",
        ),
        Fragment(
            expected,
            f"gives about {round(misses.fresh_misses)}, since a point that cleared by exactly the margin misses "
            f"{share(misses.at_margin)} of dev's resamples but about {round(100 * misses.fresh_at_margin)}% of fresh "
            f"draws of test's items",
        ),
        Fragment(expected, f"E's and F's {misses.routers[E] + misses.routers[F]} points aren't counted"),
        Fragment(expected, f"(G/Jev's {targets[1]} and {targets[2]}, for one)"),
    ]
    groups = v4["comparisons"]["groups"]
    declared = (3, "The declared questions")
    out.append(Fragment(declared, f"The same-target comparisons are {v4['comparisons']['count']} in all"))
    out.append(Fragment((2, "Found after test was seen"), f"the {v4['comparisons']['count']} comparisons"))
    lead = len(v4["comparisons"]["headline"])
    for name, group in QUESTIONS.items():
        n = 1 if group is None else len(groups[group]["pairs"]) - (lead if group == "cascades_against_routers" else 0)
        out.append(Fragment((4, name), f"Comparisons: {n}."))
    tests = {task: manifest[task]["splits"]["test"] for task in ("ifeval", "mbppplus")}
    for label, task in (("IFEval", "ifeval"), ("MBPP+", "mbppplus")):
        out.append(
            Fragment(
                (3, "Every policy at its operating points"),
                f"{label}: Only {tests[task]} test items, below the plan's minimum of a few hundred: indicative.",
            )
        )
    out += [
        Fragment(
            (3, "Selection on dev"),
            f"(the best of {len(CASCADES)} by A17 item 24's rule)",
        ),
        Fragment(
            (3, "IFEval and MBPP+ are only indicative"),
            f"Test holds {tests['ifeval']} IFEval items and {tests['mbppplus']} MBPP+ items",
        ),
    ]
    quality = {mix: {c: pareto[mix]["configs"][c]["quality"]["value"] for c in ("haiku", "opus")} for mix in MIXES}
    r, u = quality["realistic"], quality["uniform"]
    out.append(
        Fragment(
            (3, "A benchmark where the configs are further apart"),
            f"on dev it scores {share(r['haiku'])} against the frontier's {share(r['opus'])} on the realistic mix and "
            f"{share(u['haiku'])} against {share(u['opus'])} on the uniform mix",
        )
    )
    return out


def results_fragments() -> list[Fragment]:
    """Every number the results state, in the words it states it: each line of each paragraph `analysis/writeup_results.py`
    builds from test's committed tables and the committed record, the headline sentence and the answers. They stand in
    WRITEUP.md after the skeleton's commit, so the skeleton's own holds (`fragments`) leave them out."""
    tables, record, dev = load()
    result = built(tables, record, dev, expected_misses())
    answers = (3, "The answers to the declared questions")
    out = [Fragment(heading, line) for heading, line in result.lines()]
    out.append(Fragment((3, "The headline"), result.headline))
    out += [Fragment(answers, f"{question}: {said}") for question, said in result.answers.items()]
    out.append(Fragment(answers, f"The replication summary: {result.replication_answer}"))
    return out


def number_checks(text: str, where: str) -> None:
    held = [*fragments(), *results_fragments()]
    wrong = missing_fragments(held, text)
    check(
        not wrong,
        f"{where}: each of the {len(held)} number-bearing lines stands in its section"
        + "".join(f"\n      {w}" for w in wrong[:10])
        + (f"\n      ... and {len(wrong) - 10} more" if len(wrong) > 10 else ""),
    )
    left = unheld_numbers(held, text)
    check(not left, f"{where}: no number is left that no fragment holds" + (f" ({left[:10]})" if left else ""))


def has_number(line: str) -> bool:
    """Whether `line` states a number that is no citation: a header row or a rule holds none."""
    return bool(NUMBER.findall(CITATION.sub(" ", CODE_SPAN.sub(" ", line))))


def real_guards(text: str) -> None:
    """The holds can fail on the real write-up: a fragment taken away, a digit changed or a heading dropped."""
    print("\n-- the guards on the real write-up --")
    skeleton, results = fragments(), results_fragments()
    held = [*skeleton, *results]
    texts = [g.text for g in held]
    # A sentence two sections share ("Comparisons: 12.") is taken out of the text wherever it stands, so its twin
    # fragment still holds it: the section check, not this scan, tells the two places apart.
    alone = [i for i, f in enumerate(held) if texts.count(f.text) == 1 and has_number(f.text)]
    # Taking each one away costs a scan of the whole text, so of the results' lines every 15th is taken, and the skeleton's all.
    sampled = [i for i in alone if i >= len(skeleton)][::15]
    tried = [i for i in alone if i < len(skeleton)] + sampled
    loose = [held[i].text[:70] for i in tried if not unheld_numbers(held[:i] + held[i + 1 :], text)]
    check(not loose and len(sampled) >= 20 and len(tried) >= len(skeleton) // 2 + 20,
          f"taking away any one of {len(tried)} fragments with a sentence of its own leaves a number unheld"
          + (f" ({loose})" if loose else ""))  # fmt: skip
    slipped = text.replace("72 operating points route", "73 operating points route")
    check(
        slipped != text and len(missing_fragments(skeleton, slipped)) == 1,
        "a slip of one digit in the routing count is found",
    )
    dropped = text.replace("\n## Expected misses\n", "\n")
    check(dropped != text and held_in_order(DECLARED, dropped) == ["## Expected misses: missing"],
          "dropping the expected misses' heading is found")  # fmt: skip
    answers = results[-8:]
    check(all(a.text in text.replace("\n", " ") or tidy(a.text) in tidy(text) for a in answers) and len(answers) == 8,
          "the headline's answers are in the write-up: seven questions and the replication summary")  # fmt: skip
    bumps = {
        "a headline digit": ("saves 47.3% of A's cost", "saves 47.4% of A's cost"),
        "a cell of a comparison table": ("| haiku/Jev less G | +$0.79 [+$0.59, +$1.00]*", "| haiku/Jev less G | +$0.80 [+$0.59, +$1.00]*"),
        "a row of the table of every policy": ("| haiku/Jev | realistic | 99% | met | 90.6% [86.8%, 94.1%]", "| haiku/Jev | realistic | 99% | met | 90.7% [86.8%, 94.1%]"),
        "a replication count": ("| Difference in cost per 1k | 185 |", "| Difference in cost per 1k | 186 |"),
        "a live replay figure": ("| A | 100 | 82% | 81% |", "| A | 100 | 83% | 81% |"),
    }  # fmt: skip
    for what, (old, new) in bumps.items():
        slip = text.replace(old, new)
        check(old in text and slip != text and bool(missing_fragments(held, slip)),
              f"known positive: {what} changed is found")  # fmt: skip
    gone = text.replace(
        "| haiku/Jev | realistic | 99% | met | 90.6% [86.8%, 94.1%]",
        "| haiku/Jev | realistic | 99% | met | 90.6% [86.8%, 94.1%]".replace("met", ""),
    )
    check(gone != text and bool(missing_fragments(held, gone)), "known positive: a changed word of a row is found")


# --- what the fragments' words rest on ------------------------------------------------------------------------------


def source_checks(plan: str) -> None:
    print("\n-- the fragments' words against their sources --")
    item6, item22 = amendment_item(plan, "A19", 6), amendment_item(plan, "A17", 22)
    v4 = json.loads(TABLE_V4.read_text(encoding="utf-8"))
    comparisons = v4["comparisons"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]
    mixes = load_mixes()
    targets = [f"{t:.0%}" for t in TARGETS]
    head = headline()

    check(f"on the {HEADLINE_MIX} mix at the {HEADLINE_TARGET} target" in item6 and HEADLINE_TARGET in targets,
          f"A19 item 6 declares the headline on the {HEADLINE_MIX} mix at the {HEADLINE_TARGET} target, one of §11's")  # fmt: skip
    check(f"The {MIXES[1]} mix and the {targets[-1]} target are reported beside it" in item6 and TARGETS[-1] == min(TARGETS),
          f"item 6 reports the {MIXES[1]} mix and the lowest target, {targets[-1]}, beside the headline")  # fmt: skip
    check(not head.fell_back and f"That is {CASCADES[head.best]} today" in item6,
          f"the rule's cascade on dev is {CASCADES[head.best]}, which does not fall back at the headline's point")  # fmt: skip
    weights = mixes[HEADLINE_MIX]
    test_n = {task: info["splits"]["test"] for task, info in manifest.items()}
    check(max(weights, key=lambda task: weights[task]) == "mbppplus" and abs(weights["mbppplus"] - 2 / 3) < 0.05
          and min(test_n, key=lambda task: test_n[task]) == "mbppplus",
          "MBPP+ is the realistic mix's two-thirds and the smallest test split, as the headline's note says")  # fmt: skip
    check(set(test_n) == {"ifeval", "banking77", "mmlu_pro", "mbppplus"}
          and all(v4["task_warnings"][t]["warning"] == f"Only {v4['task_warnings'][t]['items']} dev items, below the plan's minimum of a few hundred: indicative"
                  for t in ("ifeval", "mbppplus"))
          and all(v4["task_warnings"][t]["warning"] is None for t in ("banking77", "mmlu_pro")),
          "table v4 words the warning for IFEval and MBPP+ as the skeleton does for test, and warns of neither other task")  # fmt: skip

    groups = comparisons["groups"]
    sizes = {name: len(g["pairs"]) for name, g in groups.items()}
    check(comparisons["count"] == sum(sizes.values()) == 51 and "51 comparisons" in item22
          and all(pair in groups["cascades_against_routers"]["pairs"] for pair in
                  (f"{CASCADES[head.best]} less G", f"{CASCADES[head.best]} less G+")),
          "table v4's five groups hold the 51 comparisons, and haiku/Jev against G and G+ are two of the cascades' against routers")  # fmt: skip
    check(comparisons["headline"] == [f"{CASCADES[head.best]} less G", f"{CASCADES[head.best]} less G+"]
          and "`haiku/Jev` against G and against G+ matter most" in item22,
          "table v4's headline pair is haiku/Jev less G and less G+, which item 22 calls the questions that matter most")  # fmt: skip
    check(list(sizes.values()) == [12, 12, 10, 5, 12] and all(f"({n})" in groups[name]["question"] for name, n in sizes.items()),
          f"the groups' sizes are {list(sizes.values())}, as item 22 lists them")  # fmt: skip
    check("The write-up leads with the declared questions, haiku/Jev against G and G+ first" in amendment_item(plan, "A19", 8),
          "A19 item 8 has the write-up lead with the declared questions, haiku/Jev against G and G+ first")  # fmt: skip

    points = json.loads((PROJECT_ROOT / "routers" / "operating_points.json").read_text(encoding="utf-8"))
    jev_g = points[cascade_section("jev", JEV_GATE)]
    same_step = [mix for mix in MIXES if jev_g[mix]["99%"]["threshold"] == jev_g[mix]["97%"]["threshold"] is not None]
    check(same_step == list(MIXES), "G/Jev's 99% and 97% operating points are the same step, under both mixes")  # fmt: skip

    stage0 = plan[plan.index("## Amendment A3") : plan.index("## Amendment A5")]
    check(all(name in stage0 for name in ("MATH", "LiveCodeBench", "CRUXEval", "IFBench")),
          "A3 and A4 name the harder tasks Stage 0 tried: MATH, LiveCodeBench, CRUXEval and IFBench")  # fmt: skip
    notes = NOTES.read_text(encoding="utf-8")
    part2 = notes[notes.index("\n## Part 2") :]
    first_idea = re.search(r"^### \S+ · (.+)$", part2, re.M)
    lead = DECLARED[-1][1]
    check(first_idea is not None and first_idea.group(1).replace("models", "configs") == lead,
          "NOTES.md's Part 2 first idea is a benchmark where the models are further apart, which Future work leads with")  # fmt: skip
    check(all(name in part2 for name in ("GPQA Diamond", "Humanity's Last Exam", "SWE-bench Verified", "Terminal-Bench",
                                         "τ-bench", "LMArena", "AlpacaEval")),
          "Part 2 names the candidate benchmarks Future work lists")  # fmt: skip


README = PROJECT_ROOT / "README.md"


def readme_check() -> None:
    """The README leads with test's headline: the summary's sentence, word for word (A19 item 8; "Numbers in documents")."""
    print("\n-- the README quotes the headline --")
    sentence = tidy(built(*load(), expected_misses()).headline)
    readme = tidy(README.read_text(encoding="utf-8"))
    check(sentence in readme, "the README says the headline's sentence, as the write-up's summary does")
    check(
        sentence.replace("47.3%", "47.4%") not in readme, "known positive: a one-digit slip in it is not in the README"
    )


def results_guard_checks() -> None:
    """The words of `analysis/writeup_results.py` on made-up cells, each worked by hand: the number formats, which side of 0 an
    interval is on, a question's counts, the missed points that share a step and the replication table."""
    print("\n-- the results' words on made-up cells --")
    check(wr.pct(0.4727674) == "47.3%" and wr.pct(0.89654, 2) == "89.65%" and wr.usd(3.9594849) == "$3.96",
          "a share is one place of percent, a cost two places of dollars")  # fmt: skip
    check(wr.points(0.006378) == "+0.64" and wr.points(-0.0000001) == "0.00" and wr.points(-0.0114) == "-1.14",
          "a quality difference is in points to two places, signed, and a negative zero is no sign")  # fmt: skip
    check(wr.dollars(-3.99024) == "-$3.99" and wr.dollars(0.004) == "$0.00" and wr.dollars(0.0449) == "+$0.04",
          "a cost difference is in dollars to two places, signed")  # fmt: skip
    check([wr.excludes(r) for r in ({"low": 0.1, "high": 0.3}, {"low": -0.3, "high": -0.1}, {"low": 0.0, "high": 0.3},
                                    {"low": -0.1, "high": 0.0}, {"low": -0.1, "high": 0.1}, None)] == [1, -1, 0, 0, 0, 0],
          "an interval excludes 0 only when wholly on one side, and one that touches 0 or is missing does not")  # fmt: skip

    def interval(value: float, low: float, high: float) -> dict[str, float]:
        return {"value": value, "low": low, "high": high}

    both = {
        "reading": "both fall back to A",
        "first_fell_back": True,
        "second_fell_back": True,
        "cost_per_1k": None,
        "quality": None,
    }

    def cell(cost: dict[str, float], quality: dict[str, float], first: bool = False, second: bool = False) -> dict:
        return {
            "reading": "",
            "first_fell_back": first,
            "second_fell_back": second,
            "cost_per_1k": cost,
            "quality": quality,
        }

    cells = {
        ("realistic", "100%"): both,
        ("realistic", "99%"): both,
        ("realistic", "97%"): cell(interval(-1.0, -1.5, -0.5), interval(0.01, 0.005, 0.02)),  # lower*, higher*
        ("uniform", "100%"): cell(interval(-0.5, -1.0, 0.2), interval(-0.01, -0.03, 0.01), first=True),  # lower, lower
        ("uniform", "99%"): cell(interval(2.0, 1.0, 3.0), interval(0.0, -0.01, 0.01), second=True),  # higher*, equal
        ("uniform", "97%"): cell(interval(0.0, 0.0, 0.0), interval(0.02, 0.01, 0.03)),  # equal, higher*
    }
    pairs = {"x less y": {mix: {t: cells[(mix, t)] for m, t in cells if m == mix} for mix in ("realistic", "uniform")}}
    check(wr.reading(pairs) == "over 6 cells, 2 with both points at A and so no difference; in the other 4, the first policy's "
          "cost per 1k was lower in 2 (1 with an interval excluding 0), higher in 1 (1) and equal in 1, and its quality was "
          "lower in 1 (0 with an interval excluding 0), higher in 2 (2) and equal in 1.",
          "a question's cells are counted by hand: both at A, then lower, higher and equal, and those excluding 0")  # fmt: skip
    check(wr.cell_words(cells[("uniform", "97%")], "quality") == "+2.00 [+1.00, +3.00]*"
          and wr.cell_words(cells[("uniform", "100%")], "cost_per_1k") == "-$0.50 [-$1.00, +$0.20]\u2020"
          and wr.cell_words(cells[("uniform", "99%")], "cost_per_1k") == "+$2.00 [+$1.00, +$3.00]\u2021*"
          and wr.cell_words(both, "quality") == "both at A",
          "a cell says its difference with its interval, a dagger for a side at A, a star for an interval excluding 0")  # fmt: skip

    def row(step: float | None, cutoff: float | None) -> dict:
        return {"threshold": step, "cutoff": cutoff}

    toy = {"parts": {
        "final_table": {"per_policy": {"p": {"label": "P"}, "q": {"label": "Q"}}},
        "target_met": {"points": {
            "p": {"realistic": {"99%": {"reading": "missed", "difference": {"value": -0.01}, "met_share": 0.25},
                                "97%": {"reading": "met", "difference": {"value": 0.02}, "met_share": 0.9}},
                  "uniform": {"99%": {"reading": "missed", "difference": {"value": -0.02}, "met_share": 0.2},
                              "97%": {"reading": "falls back to A", "difference": None, "met_share": None}}},
            "q": {"realistic": {"99%": {"reading": "missed", "difference": {"value": -0.03}, "met_share": 0.1}}},
        }},
        "p": {"mixes": {"realistic": {"99%": row(0.9, None), "97%": row(0.8, None)},
                        "uniform": {"99%": row(0.9, None), "97%": row(None, None)}}},
        "q": {"mixes": {"realistic": {"99%": row(0.9, None)}}},
    }}  # fmt: skip
    found = wr.misses(toy)
    check([(m["label"], m["mix"], m["target"], m["shares"]) for m in found]
          == [("P", "realistic", "99%", ["uniform 99%"]), ("P", "uniform", "99%", ["realistic 99%"]), ("Q", "realistic", "99%", [])],
          "missed points share a step only with a missed point of the same policy on the same threshold and cutoff")  # fmt: skip
    block, answer = wr.replication_blocks(
        {
            "parts": {
                "replication": {
                    "counts": {
                        "cost_per_1k": {"dev_excluded": 5, "same_sign": 4, "same_sign_excluding_zero": 3},
                        "quality": {"dev_excluded": 2, "same_sign": 2, "same_sign_excluding_zero": 1},
                    }
                }
            }
        }
    )
    check("| Difference in cost per 1k | 5 | 4 | 3 |" in block[(3, "The replication summary")][0]
          and "| Difference in quality | 2 | 2 | 1 |" in block[(3, "The replication summary")][0]
          and answer.startswith("for cost per 1k, 4 of the 5 dev intervals that excluded 0 have the same sign on test and 3 of them")
          and answer.endswith("for quality, 2 of 2 and 1 of 2."),
          "the replication table and its answer take the counts as they stand")  # fmt: skip


def declared_checks() -> None:
    print("\n-- the declared parts follow A19 item 11 --")
    item11 = amendment_item(PLAN.read_text(encoding="utf-8"), "A19", 11)
    texts = [text for _, text in DECLARED]

    def index(heading: str) -> int:
        return texts.index(heading)

    questions = [t for _, t in DECLARED if t in QUESTIONS]
    check(questions == list(QUESTIONS), "the declared questions are in order")
    check(texts[index("The declared questions") + 1 : index("The declared questions") + 3] == ["haiku/Jev against G", "haiku/Jev against G+"],
          "haiku/Jev against G and G+ come first among them")  # fmt: skip
    check(index("The replication summary") > index(questions[-1]), "the replication summary comes after the questions")
    check(index("Expected misses") < index("Results"), "the expected misses come before the results (item 6)")
    check(index("Found after test was seen") > index("Results") and index("Summary") == 0,
          "the summary comes first, and the heading for what was found after test was seen after the results")  # fmt: skip
    limitations = texts[index("Limitations") + 1 : index("Future work")]
    check(limitations == ["Selection on dev", "IFEval and MBPP+ are only indicative", "One run of test",
                          "The oracle is an optimistic upper bound"]
          and "selection on dev, IFEval and MBPP+ only indicative, one run of test, and the oracle an optimistic upper bound" in item11,
          "the four limitations are item 11's, in its order")  # fmt: skip
    check(texts[-2:] == ["Future work", DECLARED[-1][1]] and "a short Future work section" in item11,
          "Future work comes last and holds its lead idea")  # fmt: skip
    check(len(set(DECLARED)) == len(DECLARED), "no declared heading appears twice")


# --- the guards: made-up texts --------------------------------------------------------------------------------------

TOY = (
    (2, "Summary"),
    (3, "The headline"),
    (2, "Results"),
    (3, "Misses"),
    (2, "Future work"),
)
TOY_TEXT = f"""# Title

{MARKER} the toy skeleton -->

## Summary

### The headline

It saves 40% of the cost.

## Results

### Misses

About 3 misses.

## Future work
"""


def guard_checks() -> None:
    print("\n-- the guards on made-up texts --")
    check(held_in_order(TOY, TOY_TEXT) == [], "the made-up skeleton holds its declared parts")
    added = TOY_TEXT.replace("## Future work", "## A reported drift\n\nText.\n\n### Detail\n\n## Future work")
    check(held_in_order(TOY, added) == [], "a section added between the declared parts passes")
    check(held_in_order(TOY, TOY_TEXT + "\n## Found after test was seen\n") == [], "a section added at the end passes")
    removed = TOY_TEXT.replace("### Misses\n\nAbout 3 misses.\n\n", "")
    check(held_in_order(TOY, removed) == ["### Misses: missing"], "a removed declared part fails, named")
    renamed = TOY_TEXT.replace("### Misses", "### Some misses")
    check(held_in_order(TOY, renamed) == ["### Misses: missing"], "a renamed declared part fails")
    leveled = TOY_TEXT.replace("### Misses", "## Misses")
    check(held_in_order(TOY, leveled) == ["### Misses: missing"], "a declared part moved to another level fails")
    swapped = (
        TOY_TEXT.replace("## Summary", "## TMP").replace("## Results", "## Summary").replace("## TMP", "## Results")
    )
    check(len(held_in_order(TOY, swapped)) >= 1 and any("out of order" in w for w in held_in_order(TOY, swapped)),
          "two declared parts swapped fail, as out of order")  # fmt: skip
    hidden = TOY_TEXT.replace("### Misses", "<!--\n### Misses\n-->").replace(
        "## Future work", "```\n## Future work\n```"
    )
    check(held_in_order(TOY, hidden) == ["### Misses: missing", "## Future work: missing"],
          "a heading inside a comment or a code fence doesn't count")  # fmt: skip
    check(section_body(TOY_TEXT, (3, "Misses")) is not None and "About 3 misses." in (section_body(TOY_TEXT, (3, "Misses")) or "")
          and "Future" not in (section_body(TOY_TEXT, (3, "Misses")) or ""),
          "a section's body runs to the next heading")  # fmt: skip

    toy = [Fragment((3, "The headline"), "It saves 40% of the cost."), Fragment((3, "Misses"), "About 3 misses.")]
    check(
        missing_fragments(toy, TOY_TEXT) == [] and unheld_numbers(toy, TOY_TEXT) == [],
        "held sentences leave no stray number",
    )
    check(len(missing_fragments(toy, TOY_TEXT.replace("About 3 misses.", "About 4 misses."))) == 1,
          "a sentence with one digit changed is missing from its section")  # fmt: skip
    check(len(missing_fragments(toy, TOY_TEXT.replace("It saves 40%", "It saves\n40%"))) == 0,
          "a sentence found wrapped over two lines is still found")  # fmt: skip
    moved = TOY_TEXT.replace("About 3 misses.", "").replace(
        "It saves 40% of the cost.", "It saves 40% of the cost.\n\nAbout 3 misses."
    )
    check(len(missing_fragments(toy, moved)) == 1, "a sentence moved to another section is missing from its own")
    check(unheld_numbers(toy, TOY_TEXT.replace("## Future work", "## Future work\n\nIt costs 12.5% more.")) == ["12.5%"],
          "a number in a sentence no fragment holds is found")  # fmt: skip
    check(unheld_numbers(toy[:1], TOY_TEXT) == ["3"], "taking a fragment away leaves its number unheld")
    cited = "## Future work\n\nSee A19 item 6, items 3 and 4, §5, item 26.5, PLAN_v4.md, tables v3 and v4, `table_v4.json`.\n\n1. one {n} <!-- 5 -->\n"
    check(
        unheld_numbers([], cited) == [], "citations, code spans, comments, placeholders and list markers hold no number"
    )
    check(unheld_numbers([], "A pass of 99% and $25 and 1,000 requests.") == ["99%", "$25", "1,000"],
          "a percent, a dollar amount and a thousands separator are each a number")  # fmt: skip


def commit_pair(repo: Path, texts: Sequence[str]) -> list[str]:
    """Commit `texts` as WRITEUP.md, one commit each in `repo`, and return the commits' hashes, oldest first."""
    shas = []
    for text in texts:
        (repo / "WRITEUP.md").write_text(text, encoding="utf-8")
        git("add", "-A", repo=repo)
        git("commit", "-q", "--allow-empty", "-m", "write-up", repo=repo)
        shas.append(git("rev-parse", "HEAD", repo=repo).stdout.strip())
    return shas


def throwaway_checks() -> None:
    print("\n-- the skeleton's commit, found in a throwaway repository --")
    with tempfile.TemporaryDirectory() as folder:
        repo = Path(folder) / "repo"
        repo.mkdir()
        broke = sum(git(*args, repo=repo).returncode != 0
                    for args in (("init", "-q"), ("config", "user.email", "check@example.com"),
                                 ("config", "user.name", "check"), ("config", "commit.gpgsign", "false")))  # fmt: skip
        (repo / "other.txt").write_text(f"{MARKER} in another file\n", encoding="utf-8")
        before = commit_pair(repo, ["# Write Up\n\nA stub.\n"])
        check(
            skeleton_commit(repo) is None,
            "no commit adds the marker to WRITEUP.md yet, and another file's doesn't count",
        )
        skeleton, grown = commit_pair(
            repo, [TOY_TEXT, TOY_TEXT.replace("## Future work", "## A reported drift\n\n## Future work")]
        )
        check(
            skeleton_commit(repo) == skeleton != before[0], "the commit that first added the marker is the skeleton's"
        )
        check(show_skeleton(skeleton, repo) == TOY_TEXT, "its file is read back from git exactly")
        check(
            headings(show_skeleton(skeleton, repo) or "") == list(TOY),
            "its headings, the title aside, are the declared parts",
        )
        check(held_in_order(headings(show_skeleton(skeleton, repo) or ""), (repo / "WRITEUP.md").read_text(encoding="utf-8")) == [],
              "after a section is added in a later commit, the skeleton's parts are still held")  # fmt: skip
        gone, doubled = commit_pair(repo, [TOY_TEXT.replace(MARKER, "<!-- gone:"), f"{MARKER} again -->\n{TOY_TEXT}"])
        check(
            skeleton_commit(repo) == skeleton,
            "the marker removed, then added again, leaves the first commit the skeleton's",
        )
        check(held_in_order(headings(TOY_TEXT), TOY_TEXT.replace("### Misses", "### Findings")) == ["### Misses: missing"],
              "a later edit that renames a declared part fails against the skeleton's headings")  # fmt: skip
        check(
            broke == 0 and len({before[0], skeleton, grown, gone, doubled}) == 5,
            "every git command that set up the throwaway repository succeeded",
        )
        clone = Path(folder) / "clone"
        cloned = git("clone", "-q", "--depth", "1", f"file://{repo}", str(clone), repo=Path(folder))
        check(cloned.returncode == 0 and not full_history(clone) and full_history(repo),
              "a shallow clone, as CI's checkout is, has no history to search, and the repository does")  # fmt: skip


# --- the skeleton's commit ------------------------------------------------------------------------------------------


def skeleton_commit(repo: Path = PROJECT_ROOT) -> str | None:
    """The commit that first added `MARKER` to WRITEUP.md: the oldest of those whose change alters the marker's count
    there, since git lists the newest first. None where no commit has, as before the skeleton's own commit exists."""
    found = git("log", "--format=%H", f"-S{MARKER}", "--", "WRITEUP.md", repo=repo).stdout.split()
    return found[-1] if found else None


def show_skeleton(commit: str, repo: Path = PROJECT_ROOT) -> str | None:
    """WRITEUP.md as `commit` has it, byte for byte; None if git can't give it."""
    shown = git_bytes("show", f"{commit}:WRITEUP.md", repo=repo)
    return shown.stdout.decode("utf-8") if shown.returncode == 0 else None


CHECKOUT = re.compile(r"uses:\s*actions/checkout@\S+\s*\n\s*with:\s*\n(?:\s+[^\n]*\n)*?\s+fetch-depth:\s*0\b")


def checks_out_history(workflow: str) -> bool:
    """Whether the workflow's checkout step fetches the whole history (`fetch-depth: 0`), as the history part needs."""
    return CHECKOUT.search(workflow) is not None


def ci_checkout_check() -> None:
    print("\n-- CI's checkout holds the history this check searches --")
    shallow = "steps:\n      - uses: actions/checkout@v4\n      - uses: astral-sh/setup-uv@v6\n"
    deep = "steps:\n      - uses: actions/checkout@v4\n        with:\n          fetch-depth: 0\n      - uses: astral-sh/setup-uv@v6\n"
    elsewhere = "steps:\n      - uses: actions/checkout@v4\n      - uses: other/step@v1\n        with:\n          fetch-depth: 0\n"
    check(
        not checks_out_history(shallow),
        "known positive: a workflow whose checkout takes the default (one commit) is found",
    )
    check(not checks_out_history(elsewhere), "and a fetch-depth set on another step doesn't count")
    check(checks_out_history(deep), "a checkout with fetch-depth 0 passes")
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "check.yml").read_text(encoding="utf-8")
    check(checks_out_history(workflow), "check.yml's checkout fetches the whole history, so CI runs the history part")


def history_checks(current: str) -> None:
    print("\n-- WRITEUP.md against the skeleton's commit --")
    if not full_history():
        print("SKIP  this clone has no history: WRITEUP.md is held against DECLARED alone")
        return
    commit = skeleton_commit()
    if commit is None:
        check(MARKER in current,
              "no commit holds the skeleton yet, so the working file is it: it opens with the skeleton's marker")  # fmt: skip
        print("      (the commit that adds it pins the declared parts from then on)")
        return
    skeleton = show_skeleton(commit)
    check(skeleton is not None and MARKER in skeleton, f"commit {commit[:9]} adds the skeleton, and git gives its file")
    if skeleton is None:
        return
    check(headings(skeleton) == list(DECLARED), "that commit's headings are exactly the declared parts")
    wrong = missing_fragments(fragments(), skeleton)
    check(not wrong, "its number-bearing sentences stand in their sections" + "".join(f"\n      {w}" for w in wrong))
    wrong = held_in_order(headings(skeleton), current)
    check(
        not wrong,
        "every heading it declared is in WRITEUP.md now, in its order" + "".join(f"\n      {w}" for w in wrong),
    )


def main() -> int:
    text = WRITEUP.read_text(encoding="utf-8")
    plan = PLAN.read_text(encoding="utf-8")
    print("-- WRITEUP.md's declared parts --")
    check(
        MARKER in text and text.index(MARKER) < text.index("\n## "),
        "the skeleton's marker opens the file, before its first section",
    )
    wrong = held_in_order(DECLARED, text)
    check(not wrong, "every declared part is there, in order" + "".join(f"\n      {w}" for w in wrong))
    print("\n-- the numbers --")
    number_checks(text, "WRITEUP.md")
    real_guards(text)
    source_checks(plan)
    readme_check()
    declared_checks()
    guard_checks()
    results_guard_checks()
    throwaway_checks()
    ci_checkout_check()
    history_checks(text)
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
