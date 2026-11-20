"""The sentences and tables of WRITEUP.md's results, built from test's committed tables and the committed record (PLAN_v4.md,
Amendment A19, items 8, 10 and 11; issue #95).

Run from the project root to print the text each slot takes:  uv run python -m analysis.writeup_results

A19 item 11 has an agent fill in the write-up's numbers and a check hold each one. This module is the one place the words
are made: `built(tables, record, dev)` gives, for each heading of WRITEUP.md that holds numbers, its paragraphs (a sentence, or
a table of lines), and `analysis/check_writeup.py` takes each line of them as a fragment that must stand in its section. The
write-up's text was written from here, and a number changed in either place fails the check. Every number comes from
`figures/tables_test.json` (test's tables, whose sha256 the committed record holds), `analysis/final_record.json` (the live
replay) or `figures/table_v4.json` (dev, beside test's headline). Nothing here is computed from a test item: the readings
are counts of cells and operating points the tables already hold, and no sentence says more than the cells do.

The check recomputes the headline, the replication counts and the readings' counts apart from this module, from the same
files (`analysis/check_writeup.py`, `results_checks`).
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from analysis.final_comparisons import QUESTION_HEADINGS
from analysis.final_design import MIXES
from analysis.references import TARGETS
from harness.client import PROJECT_ROOT

TABLES = PROJECT_ROOT / "figures" / "tables_test.json"
RECORD = PROJECT_ROOT / "analysis" / "final_record.json"
DEV = PROJECT_ROOT / "figures" / "table_v4.json"

Heading = tuple[int, str]
Json = Mapping[str, Any]

TARGET_KEYS = tuple(f"{t:.0%}" for t in TARGETS)  # 100%, 99%, 97%
CELLS = tuple((mix, target) for mix in MIXES for target in TARGET_KEYS)
MARKS = {"first": "†", "second": "‡"}  # a dagger: the first policy fell back to A; a double dagger: the second
EXCLUDES = "*"  # a cell whose interval excludes 0
BOTH_AT_A = "both at A"
TASKS = {"ifeval": "IFEval", "banking77": "Banking77", "mmlu_pro": "MMLU-Pro", "mbppplus": "MBPP+"}


# ---- words for numbers ------------------------------------------------------------------------------------------------


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def interval_pct(row: Json, digits: int = 1) -> str:
    low, high = (("n/a" if row[k] is None else pct(row[k], digits)) for k in ("low", "high"))
    return f"{pct(row['value'], digits)} [{low}, {high}]"


def usd(value: float, digits: int = 2) -> str:
    return f"${value:.{digits}f}"


def interval_usd(row: Json) -> str:
    return f"{usd(row['value'])} [{usd(row['low'])}, {usd(row['high'])}]"


def signed(value: float, scale: float, unit: str = "") -> str:
    """`value` times `scale` to two places with its sign (none for 0, never a negative zero), `unit` after the sign."""
    rounded = round(value * scale, 2) + 0.0
    return ("+" if rounded > 0 else "-" if rounded < 0 else "") + f"{unit}{abs(rounded):.2f}"


def points(value: float) -> str:
    """A difference in quality, in percentage points."""
    return signed(value, 100)


def dollars(value: float) -> str:
    """A difference in cost per 1k, in dollars."""
    return signed(value, 1, "$")


def interval_signed(row: Json, how: Any) -> str:
    return f"{how(row['value'])} [{how(row['low'])}, {how(row['high'])}]"


def excludes(row: Json | None) -> int:
    """+1 where a 95% interval sits wholly above 0, -1 wholly below, 0 otherwise (and for a cell with none)."""
    if row is None:
        return 0
    return 1 if row["low"] > 0 else -1 if row["high"] < 0 else 0


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A Markdown table: its header, its rule and one line a row."""
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


# ---- the headline ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Built:
    """What the write-up's slots take: paragraphs under a heading, and the answers to the declared questions by heading."""

    blocks: dict[Heading, list[str]]
    answers: dict[str, str]
    headline: str
    replication_answer: str

    def lines(self) -> list[tuple[Heading, str]]:
        """Every line the check holds, with its heading: a sentence is a line, and so is each line of a table."""
        found: list[tuple[Heading, str]] = []
        for heading, paragraphs in self.blocks.items():
            for paragraph in paragraphs:
                found += [(heading, line) for line in paragraph.split("\n")]
        return found


def headline_sentence(tables: Json) -> str:
    """The summary's headline, the skeleton's sentence with its slots filled."""
    part = tables["parts"]["headline"]
    point = part["headline"]
    cost, a = point["cost_per_1k"], point["A"]["cost_per_1k"]
    mix, target = point["mix"], point["target"]
    return (
        f"On the {mix} mix at the {target} target, {part['label']}, the best cascade by A17 item 24's rule, saves "
        f"{pct(point['savings_vs_A']['share'])} of A's cost, at {usd(cost['value'])} per 1k {_bracket(cost, usd)} "
        f"against A's {usd(a)}, at a quality of {interval_pct(point['quality'])}, "
        f"{points(point['difference']['value'])} points {_bracket(point['difference'], points)} from the target's share "
        f"of A's quality on the same items (target {point['reading']}), and meets that target in "
        f"{pct(point['met_share'])} of test's resamples."
    )


def _bracket(row: Json, how: Any) -> str:
    return f"[{how(row['low'])}, {how(row['high'])}]"


def headline_blocks(tables: Json, dev: Json) -> dict[Heading, list[str]]:
    parts = tables["parts"]
    section = parts["headline"]["cascade"]
    label = parts["headline"]["label"]
    by_mix = tables["parts"]["final_table"]["by_mix"]
    best = [(mix, target, next(r for r in by_mix[mix][target] if r["policy"] == "I")) for mix, target in CELLS]
    met = parts["target_met"]["points"][section]
    dev_points = dev["cascades"][section]["operating_points"]
    test_rows = [
        [mix, target, row["reading"], interval_pct(row["quality"]), interval_usd(row["cost_per_1k"]),
         pct(row["savings_vs_A"]), pct(met[mix][target]["met_share"])]
        for mix, target, row in best
    ]  # fmt: skip
    dev_rows = []
    for mix, target in CELLS:
        row = dev_points[mix][target]
        dev_rows.append([mix, target, interval_pct(row["quality"]), interval_usd(row["cost_per_1k"]),
                         pct(row["savings_vs_A"]["share"]), pct(row["captured"]["met"])])  # fmt: skip
    a_rows = {mix: next(r for r in by_mix[mix][TARGET_KEYS[0]] if r["policy"] == "A") for mix in MIXES}
    a_words = " and ".join(
        f"{interval_pct(a_rows[mix]['quality'])} at {interval_usd(a_rows[mix]['cost_per_1k'])} per 1k on the {mix} mix"
        for mix in MIXES
    )
    head = parts["headline"]["headline"]
    return {
        (3, "The headline in full"): [
            f"The headline is {label} on the {head['mix']} mix at the {head['target']} target, against A. On test A scores "
            f"{a_words}. Test's six points for {label} follow, each with its quality and cost per 1k (95% intervals), the "
            f"share of A's cost saved, and the share of test's resamples that meet the target (A19 item 6).",
            table(
                ["Mix", "Target", "Reading", "Quality", "Cost per 1k", "Saved against A", "Meets the target in"],
                test_rows,
            ),  # fmt: skip
            f"Dev's points for {label}, beside them, with the share of dev's resamples that meet the target.",
            table(["Mix", "Target", "Quality", "Cost per 1k", "Saved against A", "Meets the target in"], dev_rows),
        ]
    }


# ---- the operating points that missed --------------------------------------------------------------------------------


def step_of(row: Json) -> tuple[Any, Any]:
    return row.get("threshold"), row.get("cutoff")


def misses(tables: Json) -> list[dict[str, Any]]:
    """Each operating point that routes and missed its target on test, with the points of its own policy that sit on the same
    step (the same threshold and cutoff): `section`, `label`, `mix`, `target`, `difference`, `met_share`, `shares`."""
    parts = tables["parts"]
    labels = {s: p["label"] for s, p in parts["final_table"]["per_policy"].items()}
    found: list[dict[str, Any]] = []
    for section, by_mix in parts["target_met"]["points"].items():
        for mix, by_target in by_mix.items():
            for target, point in by_target.items():
                if point["reading"] == "missed":
                    found.append({"section": section, "label": labels[section], "mix": mix, "target": target,
                                  "difference": point["difference"], "met_share": point["met_share"],
                                  "step": step_of(parts[section]["mixes"][mix][target])})  # fmt: skip
    for point in found:
        point["shares"] = [
            f"{o['mix']} {o['target']}"
            for o in found
            if o["section"] == point["section"] and o["step"] == point["step"] and o is not point
        ]
    return found


def misses_blocks(tables: Json, expected: Any) -> dict[Heading, list[str]]:
    counts = tables["parts"]["target_met"]["counts"]
    total = counts["all"]
    found = misses(tables)
    covered_missed = sum(counts[s]["missed"] for s in counts if s not in ("all", "rules", "trained"))
    router_missed = counts["rules"]["missed"] + counts["trained"]["missed"]
    router_routes = counts["rules"]["routes"] + counts["trained"]["routes"]
    sharing = [m for m in found if m["shares"]]
    steps = len({(m["section"], m["step"]) for m in sharing})
    rows = [
        [m["label"], m["mix"], m["target"], points(m["difference"]["value"]), pct(m["met_share"]),
         ", ".join(m["shares"]) or "none"]
        for m in found
    ]  # fmt: skip
    return {
        (3, "Misses against the expectation"): [
            f"{total['missed']} of the {total['routes']} operating points that route missed their target on test, and "
            f"{total['met']} met it; the other {total['falls_back']} of the {total['points']} points fall back to A, "
            f"which is counted as neither.",
            f"The expectation above was about {round(expected.dev_misses)} misses over the {expected.covered} points the "
            f"tables v3 and v4 cover (the cascades', G's and G+'s), and about {round(expected.fresh_misses)} allowing for "
            f"test's own sampling. On test {covered_missed} of those {expected.covered} missed. E's and F's "
            f"{router_routes} points that route, which the expectation leaves out, had {router_missed} miss.",
            f"{len(sharing)} of the {total['missed']} misses sit on a step that another missed point of the same policy "
            f"also uses, {steps} steps in all; the other {total['missed'] - len(sharing)} have a step of their own. "
            f"Each missed point follows, with test's quality less the target's share of A's quality (in points), the "
            f"share of test's resamples that meet the target, and the missed points on its step.",
            table(["Policy", "Mix", "Target", "Quality less the target", "Meets the target in", "Same step as"], rows),
        ]
    }


# ---- the comparisons -------------------------------------------------------------------------------------------------


def question_cells(tables: Json) -> dict[str, dict[str, Any]]:
    """Each declared question's pairs, in order, each with its six cells (the comparison parts' own cells)."""
    comparisons = tables["parts"]["comparisons"]["groups"]
    out: dict[str, dict[str, Any]] = {}
    for entry in tables["parts"]["declared"]["declared"]:
        group = comparisons[entry["group"]]["pairs"]
        out[entry["question"]] = {pair: group[pair] for pair in entry["pairs"]}
    return out


def cell_words(cell: Json, measure: str) -> str:
    """One cell of a comparison table: the difference with its interval, a dagger for each side that fell back to A, a star
    where the interval excludes 0, and `both at A` where both did (no difference is held)."""
    if cell["reading"] == "both fall back to A":
        return BOTH_AT_A
    row = cell[measure]
    how = dollars if measure == "cost_per_1k" else points
    words = interval_signed(row, how)
    return (
        words
        + (MARKS["first"] if cell["first_fell_back"] else "")
        + (MARKS["second"] if cell["second_fell_back"] else "")
        + (EXCLUDES if excludes(row) else "")
    )


def comparison_tables(pairs: Mapping[str, Any]) -> list[str]:
    header = ["First less second", *(f"{mix} {target}" for mix, target in CELLS)]
    out = []
    for measure, heading in (("cost_per_1k", "Difference in cost per 1k, in dollars:"),
                             ("quality", "Difference in quality, in percentage points:")):  # fmt: skip
        rows = [
            [pair, *(cell_words(by_mix[mix][target], measure) for mix, target in CELLS)]
            for pair, by_mix in pairs.items()
        ]
        out += [heading, table(header, rows)]
    return out


def reading(pairs: Mapping[str, Any]) -> str:
    """The cells of a question counted: how many hold no difference, and in the rest the sign of each difference and how many
    of those have an interval excluding 0. A count of what the cells hold, not a conclusion."""
    cells = [by_mix[mix][target] for by_mix in pairs.values() for mix, target in CELLS]
    defined = [c for c in cells if c["reading"] != "both fall back to A"]

    def count(measure: str) -> str:
        rows = [c[measure] for c in defined]
        lower, higher = [r for r in rows if r["value"] < 0], [r for r in rows if r["value"] > 0]
        equal = len(rows) - len(lower) - len(higher)
        return (
            f"lower in {len(lower)} ({sum(excludes(r) != 0 for r in lower)} with an interval excluding 0), higher in "
            f"{len(higher)} ({sum(excludes(r) != 0 for r in higher)}) and equal in {equal}"
        )

    return (
        f"over {len(cells)} cells, {len(cells) - len(defined)} with both points at A and so no difference; in the other "
        f"{len(defined)}, the first policy's cost per 1k was {count('cost_per_1k')}, and its quality was "
        f"{count('quality')}."
    )


def comparison_blocks(tables: Json) -> tuple[dict[Heading, list[str]], dict[str, str]]:
    blocks: dict[Heading, list[str]] = {}
    answers: dict[str, str] = {}
    for question, pairs in question_cells(tables).items():
        said = reading(pairs)
        answers[question] = said
        blocks[(4, question)] = [
            f"Each pair is the first policy's point less the second's at one mix and one target, paired over the same "
            f"resamples, in each of the six mix-and-target cells. {said[0].upper() + said[1:]}",
            *comparison_tables(pairs),
        ]
    return blocks, answers


# ---- the replication summary -----------------------------------------------------------------------------------------


def replication_blocks(tables: Json) -> tuple[dict[Heading, list[str]], str]:
    counts = tables["parts"]["replication"]["counts"]
    cost, quality = counts["cost_per_1k"], counts["quality"]
    rows = [
        [name, str(c["dev_excluded"]), str(c["same_sign"]), str(c["same_sign_excluding_zero"])]
        for name, c in (("Difference in cost per 1k", cost), ("Difference in quality", quality))
    ]
    answer = (
        f"for cost per 1k, {cost['same_sign']} of the {cost['dev_excluded']} dev intervals that excluded 0 have the same "
        f"sign on test and {cost['same_sign_excluding_zero']} of them also exclude 0 there; for quality, "
        f"{quality['same_sign']} of {quality['dev_excluded']} and {quality['same_sign_excluding_zero']} of "
        f"{quality['dev_excluded']}."
    )
    header = (
        "| | Dev intervals excluding 0 | Same sign on test | Same sign, and excluding 0 on test |\n|---|---|---|---|"
    )
    return {
        (3, "The replication summary"): [header + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)]
    }, answer


# ---- every policy ----------------------------------------------------------------------------------------------------


def policy_blocks(tables: Json) -> dict[Heading, list[str]]:
    final = tables["parts"]["final_table"]
    per = final["per_policy"]
    rows = []
    for section, body in per.items():
        for mix, target in CELLS:
            r = body["mixes"][mix][target]
            rows.append([body["label"], mix, target, r["reading"], interval_pct(r["quality"]),
                         interval_usd(r["cost_per_1k"]), pct(r["savings_vs_A"]),
                         "none" if r["captured"] is None else pct(r["captured"])])  # fmt: skip
    blocks: dict[Heading, list[str]] = {
        (4, "Under both mixes"): [
            "Each row is a policy at one mix and one target: whether the target is met (A19 item 6), quality and cost per 1k "
            "with 95% intervals, the share of A's cost saved, and the fraction of the hindsight oracle's savings captured "
            "at that target (none where the tables hold none).",
            table(
                ["Policy", "Mix", "Target", "Reading", "Quality", "Cost per 1k", "Saved against A", "Captured"], rows
            ),
        ]
    }
    for task, name in TASKS.items():
        by_task = final["by_task"][task]
        header = ["Policy", *(f"{mix} {target}" for mix, target in CELLS)]
        a_row = next(r for r in by_task["mixes"]["realistic"]["100%"] if r["policy"] == "A")
        task_rows = [
            [a_row["label"]] + [f"{pct(a_row['quality']['value'])}, {usd(a_row['cost_per_1k']['value'])}"] * len(CELLS)
        ]
        for section, body in per.items():
            cells = []
            for mix, target in CELLS:
                r = body["tasks"][task][mix][target]
                cells.append(f"{pct(r['quality']['value'])}, {usd(r['cost_per_1k']['value'])}")
            task_rows.append([body["label"], *cells])
        warning = by_task["warning"]
        blocks[(4, f"{name}, per task")] = [
            (
                f"{name} has {by_task['items']} test items."
                + (f" {warning}." if warning else "")
                + " Each cell is the policy's "
                "quality and cost per 1k on this task alone, at the operating point the mix and target froze; intervals are "
                "in figures/tables_test.json."
            ),
            table(header, task_rows),
        ]
    return blocks


# ---- the oracle and the envelope --------------------------------------------------------------------------------------


def oracle_blocks(tables: Json) -> dict[Heading, list[str]]:
    refs = tables["parts"]["references"]["mixes"]
    rows = []
    for mix, target in CELLS:
        point = refs[mix]["targets"][target]
        captured = point["envelope"]["captured"]
        rows.append([mix, target, pct(point["quality"]), usd(point["oracle"]["cost_per_1k"]),
                     pct(point["oracle"]["savings_vs_A"]["share"]), usd(point["envelope"]["cost_per_1k"]),
                     interval_pct(captured)])  # fmt: skip
    ends = {mix: refs[mix]["configs"] for mix in MIXES}
    envelope = " ".join(
        f"On the {mix} mix the envelope runs from B (haiku) at {pct(ends[mix]['haiku']['quality']['value'])} and "
        f"{usd(ends[mix]['haiku']['cost_per_1k']['value'])} per 1k to A at {pct(ends[mix]['opus']['quality']['value'])} and "
        f"{usd(ends[mix]['opus']['cost_per_1k']['value'])} per 1k, and the oracle reaches at most "
        f"{pct(refs[mix]['oracle_max_quality'])}; {refs[mix]['failed_by_every_config']} items fail on every config."
        for mix in MIXES
    )
    return {
        (3, "The hindsight oracle and the random-mixture envelope"): [
            envelope,
            "At each quality target the oracle's cost per 1k and the share of A's cost it saves, and the cost per 1k of "
            "the random mixture that reaches the target with the envelope's fraction of the oracle's savings it captures "
            "(95% interval). Each policy's captured fraction at its operating points is in the last column of the table of "
            "every policy above.",
            table(
                [
                    "Mix",
                    "Target",
                    "Target quality",
                    "Oracle cost per 1k",
                    "Oracle saved against A",
                    "Envelope cost per 1k",
                    "Envelope captured",
                ],
                rows,
            ),  # fmt: skip
        ]
    }


# ---- the live replay -------------------------------------------------------------------------------------------------


def replay_of(record: Json) -> Json:
    go = record["gos"][-1]
    return next(s for s in go["steps"] if s["name"] == "replay")["detail"]


def _and(words: Sequence[str]) -> str:
    return f"{', '.join(words[:-1])} and {words[-1]}"


def seconds(values: Sequence[float]) -> str:
    return "none" if not values else ", ".join(f"{v:.2f} s" for v in values)


def replay_blocks(record: Json) -> dict[Heading, list[str]]:
    detail = replay_of(record)
    results = detail["results"]
    cond = detail["conditions"]
    per_task = ", ".join(f"{n} {TASKS[t]}" for t, n in sorted(detail["per_task"].items()))
    quality, latency, by_task = [], [], []
    for key, entry in results.items():
        live, off = entry["live"], entry["offline"]
        label = entry["label"]
        quality.append([label, str(live["requests"]), pct(live["quality"]["value"], 0), pct(off["quality"], 0),
                        usd(live["cost_per_request"], 5), usd(off["cost_per_request"], 5),
                        pct(live["escalation_rate"], 0), pct(off["escalation_rate"], 0)])  # fmt: skip
        latency.append([label, str(live["kept"]), f"{live['median_s']:.2f}", f"{off['latency_median_s']:.2f}",
                        f"{live['p90_s']:.2f}", f"{off['latency_p90_s']:.2f}"])  # fmt: skip
        for task in sorted(TASKS):
            q = live["quality_by_task"][task]
            offline_q, offline_n = off["quality_by_task"][task]
            by_task.append([label, TASKS[task], str(q["n"]), pct(q["value"], 0), str(offline_n), pct(offline_q, 0)])
    apart = "; ".join(
        f"{e['label']}: first call {seconds(e['live']['first_call_s'])}, reconnects {seconds(e['live']['reconnects_s'])}, "
        f"cut by a crash {len(e['live']['interrupted_s'])}"
        for e in results.values()
    )
    return {
        (3, "The live replay"): [
            f"The replay ran {_and([e['label'] for e in results.values()])} end to end through the standard API on "
            f"{detail['items']} items ({per_task}), drawn at the unlock. The machine was {cond['machine']}; the network was "
            f"{cond['network']}. It started at {cond['go_started']} and ended at {cond['go_ended']}.",
            "Quality is the share of the replay's answers graded correct, and the offline estimate is the same policy on "
            "the same items from test's matrix. Cost per request is at list price.",
            table(
                [
                    "Policy",
                    "n",
                    "Quality, live",
                    "Quality, offline",
                    "Cost per request, live",
                    "Cost per request, offline",
                    "Escalation, live",
                    "Escalation, offline",
                ],
                quality,
            ),  # fmt: skip
            "Latency is from the request to the answer returned. A policy's first call and any request a crash cut are "
            "set apart and not timed, so the requests timed are fewer than n where there is one; the offline estimate "
            "is from dev's committed medians (A17 item 16).",
            table(
                [
                    "Policy",
                    "Requests timed",
                    "Median, live (s)",
                    "Median, offline (s)",
                    "p90, live (s)",
                    "p90, offline (s)",
                ],
                latency,
            ),  # fmt: skip
            f"The first call, the reconnects and the requests a crash cut, reported apart: {apart}.",
            "Quality by task, with each side's n.",
            table(["Policy", "Task", "n, live", "Quality, live", "n, offline", "Quality, offline"], by_task),
        ]
    }


# ---- the whole ---------------------------------------------------------------------------------------------------------


def built(tables: Json, record: Json, dev: Json, expected: Any) -> Built:
    """Every slot of the write-up that holds numbers. `expected` is `final_design.expected_misses()`."""
    blocks: dict[Heading, list[str]] = {}
    blocks |= headline_blocks(tables, dev)
    blocks |= misses_blocks(tables, expected)
    comparisons, answers = comparison_blocks(tables)
    blocks |= comparisons
    replication, replication_answer = replication_blocks(tables)
    blocks |= replication
    blocks |= policy_blocks(tables)
    blocks |= oracle_blocks(tables)
    blocks |= replay_blocks(record)
    return Built(blocks, answers, headline_sentence(tables), replication_answer)


def load() -> tuple[Json, Json, Json]:
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    return read(TABLES), read(RECORD), read(DEV)


if __name__ == "__main__":
    from analysis.final_design import expected_misses

    result = built(*load(), expected_misses())
    print(result.headline, "\n")
    for question, said in result.answers.items():
        print(question + ":", said)
    print("replication:", result.replication_answer, "\n")
    for heading, paragraphs in result.blocks.items():
        print("#" * heading[0], heading[1])
        print("\n\n".join(paragraphs), "\n")
    assert [q for q, _ in QUESTION_HEADINGS] == list(result.answers)
