"""Check G's and G+'s dev router-score tables, quantiles and walks (PLAN_v4.md §8 3c and 3c+; Amendment A15, items 9,
10, 13 and 14; Amendment A9, items 9 and 18; Amendment A11, item 13; issue #40).

Run from the project root:  uv run python -m analysis.check_jev_dev

No Jev call, no test item and no outcome of a test request: the pieces run on synthetic stored responses built from the
committed settings' own question IDs, and the real dev records are read only where `results/jev/` holds them.

- the table, hand-worked: each item's router scores come from the stored response through the router's own parser
  (G's unchanged, G+'s through the frozen weights), each overhead is the stored input tokens at the settings' price, a
  refused response scores 0 for every config, and the table passes the existing refusal;
- the table's refusals, each fed its bad case: a dev item with no record, a record that isn't a dev request, a stored
  refusal flag that disagrees with the response, a request stored twice;
- the quantiles, hand-worked;
- the pieces of the walk checks, each with a known positive: a certified step that misses its target by its margin, a
  captured fraction above 1, a first call that comes before the freeze commit;
- where `results/` holds the dev matrix and the stored dev responses (the hook, not CI): the stored records (744 per
  router, pass 1, the router's model ID, every body hash as the frozen code builds it, every call after the freeze
  commit), the tables rebuilt from them by a second path and compared, the quantiles, the walks reproducing the sections
  of `routers/operating_points.json` exactly, every certified step clearing its target by its margin, every captured
  fraction at most 1, E's and F's sections as A9 item 18 and A11 item 21 hash them, and the allowlist asking for all four
  sections (E, F, G and G+).
"""

import copy
import email.utils
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np

from analysis import collect_jev, jev_table
from analysis import jev_train_report as report
from analysis.benchmark import read_rows
from analysis.check_jev_settings_commit import settings_commit
from analysis.operating_points import (
    OPERATING_POINTS_FILE,
    OperatingPointsError,
    check_sections,
    read_section,
    section_hash,
)
from analysis.plan import PLAN, amendment_item
from analysis.references import ROUNDING, TARGETS
from analysis.router_names import NAMED_ROUTERS, E, F
from harness.client import PROJECT_ROOT
from routers import jev, jev_call
from routers.ladder import SCORED
from routers.request import Request

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def refused(call: Callable[[], object], error: type[Exception] = jev_table.JevTableError) -> bool:
    try:
        call()
    except error:
        return True
    return False


def near(got: float, want: float, tolerance: float = 1e-9) -> bool:
    return abs(got - want) <= tolerance


def g_record(item: str, answers: dict[str, float] | None, tokens: int = 1000, flag: bool | None = None) -> dict:
    """A stored G record: a 200 whose answers are the three success statements, or none (a refusal)."""
    body = {} if answers is None else {f"success_{c}": {"type": "noul", "noul": a} for c, a in answers.items()}
    return {"router": "jev", "split": "dev", "item_id": item, "pass": 1, "response": {"answers": body},
            "usage": {"input_tokens": tokens}, "refused": answers is None if flag is None else flag}  # fmt: skip


def table_checks() -> None:
    loaded = collect_jev.load_settings()
    ids = ["a:1", "a:2", "a:3"]
    requests = {i: Request(f"prompt {i}") for i in ids}
    records = [
        g_record("a:1", {"haiku": 0.9, "sonnet": 0.95, "opus_low": 0.99}, tokens=1000),
        g_record("a:2", None, tokens=2000),
        g_record("a:3", {"haiku": 0.0, "sonnet": 0.5, "opus_low": 1.0}, tokens=500),
    ]
    table = jev_table.build_table("jev", records, requests, ids, loaded)
    by_id = {e.item_id: e for e in table.items}
    check([e.item_id for e in table.items] == ids, "the table holds the dev items in the order asked")
    check(
        by_id["a:1"].router_scores == {"haiku": 0.9, "sonnet": 0.95, "opus_low": 0.99},
        "G's router scores are Jev's answers",
    )
    check(by_id["a:2"].router_scores == dict.fromkeys(SCORED, 0.0), "a refused response scores 0 for every config")
    price = jev.overhead_dollars(1, loaded.client)
    check(
        near(by_id["a:1"].overhead, 1000 * 0.042 / 1_000_000) and near(price, 0.042 / 1_000_000),
        "overhead is input tokens at $0.042 per million",
    )
    check(near(by_id["a:2"].overhead, 2000 * 0.042 / 1_000_000), "a refused call is still charged its input tokens")
    check(
        refused(lambda: jev_table.build_table("jev", records[:2], requests, ids, loaded)),
        "a dev item with no record is refused",
    )
    check(
        refused(lambda: jev_table.build_table("jev", records + [records[0]], requests, ids, loaded)),
        "a request stored twice is refused",
    )
    other = [{**records[0], "split": "train"}, *records[1:]]
    check(
        refused(lambda: jev_table.build_table("jev", other, requests, ids, loaded)),
        "a record of another split is refused",
    )
    pass_two = [{**records[0], "pass": 2}, *records[1:]]
    check(
        refused(lambda: jev_table.build_table("jev", pass_two, requests, ids, loaded)),
        "a pass 2 record is not a dev call and is refused",
    )
    lying = [g_record("a:1", {"haiku": 0.9, "sonnet": 0.95, "opus_low": 0.99}, flag=True), *records[1:]]
    check(
        refused(lambda: jev_table.build_table("jev", lying, requests, ids, loaded)),
        "a refusal flag the response disagrees with is refused",
    )
    check(
        refused(lambda: jev_table.build_table("nope", records, requests, ids, loaded)), "an unknown router is refused"
    )


def plus_record(item: str, schema_free: bool = True) -> dict:
    """A stored G+ record whose 17 Noul answers are all 0.5 (log-odds 0) and whose request type is (.1, .2, .3, .4)."""
    settings = json.loads(jev_settings_text())["g_plus"]
    ids = [s["id"] for s in settings["success_statements"] + settings["difficulty_statements"]]
    options = settings["request_type"]["options"]
    answers: dict = {i: {"type": "noul", "noul": 0.5} for i in ids}
    answers["request_type"] = {"type": "choice", "probabilities": dict(zip(options, (0.1, 0.2, 0.3, 0.4), strict=True))}
    return {"router": "jev_bolstered", "split": "dev", "item_id": item, "pass": 1, "response": {"answers": answers},
            "usage": {"input_tokens": 9000}, "refused": False}  # fmt: skip


def jev_settings_text() -> str:
    return jev.SETTINGS_FILE.read_text(encoding="utf-8")


def plus_checks() -> None:
    loaded = collect_jev.load_settings()
    weights = json.loads(jev.WEIGHTS_FILE.read_text(encoding="utf-8"))
    mean, scale = weights["standardizer"]["mean"], weights["standardizer"]["scale"]
    # The feature vector worked by hand: the 11 success and difficulty features are 0, the four request-type answers
    # are .1 to .4 as returned, and the schema flag is 0 for a request with no output schema.
    features = [0.0] * 11 + [0.1, 0.2, 0.3, 0.4] + [0.0]
    ids = ["b:1", "b:2"]
    requests = {"b:1": Request("p"), "b:2": Request("q", {"type": "object"})}
    records = [plus_record("b:1"), plus_record("b:2")]
    table = jev_table.build_table("jev_bolstered", records, requests, ids, loaded, jev.read_weights(loaded.call))
    for item, flag in (("b:1", 0.0), ("b:2", 1.0)):
        x = [*features[:-1], flag]
        want = {
            c: 1
            / (
                1
                + math.exp(
                    -(
                        sum(w * (v - m) / s for w, v, m, s in zip(h["coef"], x, mean, scale, strict=True))
                        + h["intercept"]
                    )
                )
            )
            for c, h in weights["heads"].items()
        }
        got = next(e for e in table.items if e.item_id == item).router_scores
        check(
            all(near(got[c], want[c]) for c in SCORED),
            f"G+'s router scores for {item} are the heads on the hand-worked features",
        )
    check(
        near(table.items[0].overhead, 9000 * 0.042 / 1_000_000),
        "G+'s overhead is its input tokens at $0.042 per million",
    )
    check(
        refused(lambda: jev_table.build_table("jev_bolstered", records, requests, ids, loaded)),
        "G+'s table without its weights is refused",
    )


def quantile_checks() -> None:
    loaded = collect_jev.load_settings()
    ids = [f"a:{n}" for n in range(1, 6)]
    records = [g_record(f"a:{n}", {"haiku": n / 10, "sonnet": 0.5, "opus_low": 1.0 - n / 10}) for n in range(1, 6)]
    table = jev_table.build_table("jev", records, dict.fromkeys(ids, Request("p")), ids, loaded)
    got = jev_table.quantiles(table)
    # haiku's scores are .1 .2 .3 .4 .5: numpy's linear quantile at q is the value at position q * 4.
    want = {0.0: 0.1, 0.05: 0.12, 0.25: 0.2, 0.5: 0.3, 0.75: 0.4, 0.95: 0.48, 1.0: 0.5}
    check(all(near(got["haiku"][q], v) for q, v in want.items()), "the quantiles are numpy's linear ones, by hand")
    check(
        set(got) == set(SCORED) and all(near(v, 0.5) for v in got["sonnet"].values()),
        "a constant score has constant quantiles",
    )
    check(near(jev_table.share_at_or_above(table, "haiku", 0.3), 0.6), "the share at or above a value, by hand")


SHARE_HASH = re.compile(r"sha256 `([0-9a-f]{64})`")  # how A9 item 18 and A11 item 21 write a section's hash


def step_problems(section: Mapping[str, Any]) -> list[str]:
    """What is wrong with a router's section, by the rules of A9 item 9: the walk starts at A, runs from the highest
    threshold down, every certified step (A included) clears its target by its margin, the walk ends at the first
    step that misses (or tried them all), and the operating point is the cheapest certified step, ties to the earlier."""
    found = []
    for mix, by_target in section.items():
        for key, point in by_target.items():
            steps, target, where = point["steps"], point["target"], f"{mix} {key}"
            stop = next((i for i, step in enumerate(steps) if not step["clears"]), len(steps))
            certified, missed = steps[:stop], steps[stop:]
            if steps[0]["threshold"] is not None or steps[0]["margin"] != 0:
                found.append(f"{where}: the walk does not start at A")
            if any(step["quality"] - target < step["margin"] - ROUNDING for step in certified):
                found.append(f"{where}: a certified step misses its target by its margin")
            if missed and (len(missed) > 1 or missed[0]["quality"] - target >= missed[0]["margin"] - ROUNDING):
                found.append(f"{where}: the walk goes on past the first step that misses")
            thresholds = [step["threshold"] for step in steps[1:]]
            if thresholds != sorted(thresholds, reverse=True):
                found.append(f"{where}: the steps do not run from the highest threshold down")
            cheapest = certified[0]
            for step in certified[1:]:
                if step["cost_per_1k"] < cheapest["cost_per_1k"] - ROUNDING:
                    cheapest = step
            if point["threshold"] != cheapest["threshold"] or point["fell_back"] != (point["threshold"] is None):
                found.append(f"{where}: the operating point is not the cheapest certified step")
    return found


def captured_problems(values: Sequence[float | None]) -> list[str]:
    """Captured fractions that are not at most 1 (and at least 0); n/a (None) is allowed."""
    return [f"{v}" for v in values if v is not None and not 0 <= v <= 1 + ROUNDING]


def first_call_problems(records: Sequence[Mapping[str, Any]], freeze: datetime) -> list[str]:
    """The first stored call's server `Date`, and every call's own collection time, must come after the freeze."""
    if not records:
        return ["no stored calls"]
    found = []
    if report.first_date(records) <= freeze:
        found.append(f"the first call's Date {records[0]['server_date']} is not after the freeze commit")
    if any(datetime.fromisoformat(r["collected_at"]) <= freeze for r in records):
        found.append("a call was collected before the freeze commit")
    return found


def rule_checks() -> None:
    steps = [
        {"threshold": None, "quality": 0.9, "cost_per_1k": 10.0, "margin": 0.0, "clears": True},
        {"threshold": 0.9, "quality": 0.89, "cost_per_1k": 8.0, "margin": 0.005, "clears": True},
        {"threshold": 0.8, "quality": 0.7, "cost_per_1k": 5.0, "margin": 0.005, "clears": False},
    ]
    point = {"threshold": 0.9, "margin": 0.005, "target": 0.88, "fell_back": False, "steps": steps}
    section: dict[str, Any] = {"realistic": {"99%": point}}
    check(
        step_problems(section) == [],
        "a hand-built walk passes: A, one certified step, the first miss, the cheapest chosen",
    )
    worse = copy.deepcopy(section)
    worse["realistic"]["99%"]["steps"][1]["quality"] = 0.884  # 0.884 - 0.88 = 0.004 < the margin 0.005
    check(step_problems(worse) != [], "known positive: a certified step short of its margin is found")
    wrong = copy.deepcopy(section)
    wrong["realistic"]["99%"]["threshold"] = None
    wrong["realistic"]["99%"]["fell_back"] = True
    check(
        step_problems(wrong) != [], "known positive: an operating point that isn't the cheapest certified step is found"
    )
    check(captured_problems([0.0, 0.5, 1.0, None]) == [] and captured_problems([1.2]) != [] and captured_problems([-0.1]) != [],
          "a captured fraction must lie in [0, 1] (n/a allowed); 1.2 and -0.1 are found")  # fmt: skip
    freeze = datetime.fromisoformat("2026-10-27T20:20:00-07:00")
    after = [{"server_date": "Wed, 28 Oct 2026 03:26:00 GMT", "collected_at": "2026-10-27T20:26:00-07:00"}]
    before = [{"server_date": "Wed, 28 Oct 2026 03:10:00 GMT", "collected_at": "2026-10-27T20:26:00-07:00"}]
    check(first_call_problems(after, freeze) == [], "a first call at 03:26 GMT is after a freeze at 03:20 GMT")
    check(
        first_call_problems(before, freeze) != [],
        "known positive: a first call at 03:10 GMT, before the freeze, is found",
    )
    check(email.utils.parsedate_to_datetime(after[0]["server_date"]) > freeze, "the Date header is read as GMT")


def stored_checks(freeze: str) -> None:
    """The stored dev records, against the settings and the frozen call building (no Jev call)."""
    loaded = collect_jev.load_settings()
    dev_rows = read_rows(split="dev")
    requests = {r["item_id"]: collect_jev.from_row(r) for r in dev_rows}
    when = datetime.fromisoformat(
        subprocess.run(
            ["git", "show", "-s", "--format=%cI", freeze], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
    )
    for router in collect_jev.ROUTERS:
        records = collect_jev.read_records(router, "dev")
        counts = report.call_counts(records)
        rebuilt = report.rebuild(router, records, requests, loaded.call)
        check(counts.calls == len(dev_rows) == 744 and counts.pass_two == 0, f"{router}: 744 dev calls, all pass 1")
        check(
            [r["item_id"] for r in records] == [r["item_id"] for r in dev_rows],
            f"{router}: the calls are in the reader's order, dev items only",
        )
        check(counts.models == (loaded.call.model,), f"{router}: every response names {loaded.call.model}")
        check(rebuilt.mismatched == [], f"{router}: every stored body hash is what the frozen code builds")
        check(
            first_call_problems(records, when) == [],
            f"{router}: the first call's server Date and every collection time are after the freeze commit",
        )


def table_checks_real() -> None:
    """Each table rebuilt by a second path from the stored records (the parser's own functions, with the price as the
    plan states it) and compared with `jev_table`'s."""
    loaded = collect_jev.load_settings()
    weights = jev.read_weights(loaded.call)
    dev_rows = read_rows(split="dev")
    requests = {r["item_id"]: collect_jev.from_row(r) for r in dev_rows}
    for router in collect_jev.ROUTERS:
        records = collect_jev.read_records(router, "dev")
        table = jev_table.dev_table(router)
        check(
            [e.item_id for e in table.items] == [r["item_id"] for r in dev_rows],
            f"{router}'s table is dev's items in the reader's order",
        )
        same = True
        for entry, record in zip(table.items, records, strict=True):
            if router == "jev":
                scored = jev.g_scores(record["response"], jev_call.g_questions(loaded.call))
                scores = scored.scores
            else:
                parsed = jev.parse_answers(record["response"], jev_call.g_plus_questions(loaded.call))
                scores = jev.g_plus_scores(
                    parsed.answers, parsed.refused, requests[entry.item_id], loaded.call, weights
                )
            tokens = record["usage"]["input_tokens"]
            same &= scores == entry.router_scores and abs(entry.overhead - tokens * 0.042 / 1_000_000) < 1e-15
        check(
            same,
            f"{router}'s table rebuilds from the stored records: scores through the parser, overhead at $0.042 per million",
        )
        by_hand = {
            c: np.quantile([e.router_scores[c] for e in table.items], jev_table.QUANTILES).tolist()
            for c in jev_table.SCORED
        }
        got = jev_table.quantiles(table)
        check(
            all(list(got[c].values()) == by_hand[c] for c in by_hand),
            f"{router}'s quantiles are numpy's on the table's scores, from no outcome",
        )


def walk_checks() -> None:
    from analysis import jev_walk, matrix
    from analysis.bootstrap import resample

    dev = matrix.load().split("dev")
    draws = resample(dev)
    plan_hashes = {
        E: SHARE_HASH.search(amendment_item(PLAN.read_text(), "A9", 18)),
        F: SHARE_HASH.search(amendment_item(PLAN.read_text(), "A11", 21)),
    }
    for letter in (E, F):
        found = plan_hashes[letter]
        check(
            found is not None and section_hash(read_section(NAMED_ROUTERS[letter])) == found[1],
            f"{letter}'s section hashes as its record says",
        )
    for router in jev_walk.ROUTERS:
        name = NAMED_ROUTERS[router]
        section = read_section(name)
        walks = jev_walk.walks_of(router, draws)
        check(
            list(section) == ["realistic", "uniform"]
            and all(set(v) == {"100%", "99%", "97%"} for v in section.values()),
            f"{name}: both mixes and all three targets, six operating points",
        )
        rebuilt = jev_walk.rebuilt_section(router)
        check(
            rebuilt == section and section_hash(rebuilt) == section_hash(section),
            f"the walk on dev reproduces {name}'s section exactly, its hash too",
        )
        check(
            step_problems(section) == [],
            f"{name}: every certified step clears its target by its margin, and the walk's other rules hold",
        )
        values = [r.captured for by in jev_walk.readings(router, walks, draws).values() for r in by.values()]
        check(len(values) == 6 and captured_problems(values) == [], f"{name}: every captured fraction is at most 1")
        check(
            all(share in TARGETS for by in walks.values() for share in by),
            f"{name}: the walks are at A's quality shares of section 11",
        )
    try:
        check_sections(OPERATING_POINTS_FILE, required=tuple(NAMED_ROUTERS.values()))
        ok = True
    except OperatingPointsError:
        ok = False
    check(ok, "the allowlist passes with all four sections required: E, F, G and G+")
    check(
        set(json.loads(OPERATING_POINTS_FILE.read_text())) == set(NAMED_ROUTERS.values()),
        "and the file holds those four sections and no other",
    )


def main() -> int:
    print("-- the table --")
    table_checks()
    print("\n-- G+'s table --")
    plus_checks()
    print("\n-- the quantiles --")
    quantile_checks()
    print("\n-- the walk checks' own rules, with known positives --")
    rule_checks()
    freeze = settings_commit(PROJECT_ROOT, "routers/jev_weights.json")
    have_dev = all(collect_jev.records_path(r, "dev").exists() for r in collect_jev.ROUTERS)
    if freeze is None or not have_dev or not (PROJECT_ROOT / "results" / "dev_matrix.sqlite").exists():
        print(
            "\n(the freeze commit, the stored dev responses or the dev matrix are not here: the stored-record sections are skipped)"
        )
    else:
        print("\n-- the stored dev records --")
        stored_checks(freeze)
        print("\n-- the tables rebuilt from them --")
        table_checks_real()
        print("\n-- the walks and the operating-points file --")
        walk_checks()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
