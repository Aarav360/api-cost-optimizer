"""Check Stage 7's record and the README's numbers about it (PLAN_v4.md §10, §12; Amendment A17, items 24 and 29; issue #72).

Run from the project root:  uv run python -m analysis.check_stage7
Print the record's lines as the committed files give them:  uv run python -m analysis.check_stage7 --print
Print the README's numeric sentences:  uv run python -m analysis.check_stage7 --print-readme

Reads `figures/table_v4.json`, `routers/operating_points.json`, `analysis/gates_dev_record.json` and PLAN_v4.md, so CI runs
it (the git order needs the freeze commit and is skipped where the clone lacks it). Never a test item, no gate and no API
call.

- Amendment A17 item 29's record, read through the plan's amendment-item reader: every number in it is the one the committed
  files give, at the precision it prints: the table of the fifteen cascades (savings against A and the share of resamples
  meeting the target, at three targets under both mixes, and the mean over the six points), the best cascade by item 24's
  rule, each cascade section's sha256 and the whole operating-points file's, dev's first Jev call's `Date` and the dev
  batch's creation time, and dev's latency conditions. Known positives: a one-digit slip in a table cell, in a hash and in
  the `Date` each fail, and so do a number the check does not compare and a line missing from the record;
- what the record's numbers rest on, recomputed from the sections and not from table v4's own claims: the best cascade
  from the six savings of each cascade, the savings from each section's operating-point step and A's cost, the hash of
  each section and of the whole file from the file's bytes;
- the order (item 20.6), from git where the freeze commit is present: dev's first call and the dev batch come after it;
- the README's numbers about Stage 7: each sentence that quotes one is the one the committed files give, with a guard that
  a one-digit slip is not in the README.
"""

import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from email.utils import parsedate_to_datetime
from itertools import zip_longest
from typing import Any

from analysis.check_stage5 import numeric
from analysis.operating_points import OPERATING_POINTS_FILE, read_section, section_hash, target_key
from analysis.plan import PLAN, amendment_item
from analysis.references import ROUNDING, TARGETS
from analysis.router_names import CASCADES, FREE_GATE, HAIKU, JEV_GATE, SONNET_GATE, parse_cascade
from analysis.rules_walk import MIXES
from analysis.table_v2 import DESCRIPTIVE
from analysis.table_v4 import TABLE_V4
from harness.client import PROJECT_ROOT

DEV_RECORD = PROJECT_ROOT / "analysis" / "gates_dev_record.json"
README = PROJECT_ROOT / "README.md"
FREEZE_COMMIT = "8d5dfaadd85137b060eb35d5593105ce0f7234a9"  # A17 item 28's freeze commit; dev's calls come after it
POINTS = [
    (mix, share) for mix in MIXES for share in TARGETS
]  # item 24's order: the realistic mix's targets, then the uniform's
problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def whole_file_sha256() -> str:
    return hashlib.sha256(OPERATING_POINTS_FILE.read_bytes()).hexdigest()


def cell(row: Mapping[str, Any]) -> str:
    """One table cell: savings against A as a share of A's cost, and the share of resamples meeting the target. A fall back
    to A saves 0.0% and A meets every target in every resample, so it reads 0.0% / 100.0%."""
    return f"{row['savings_vs_A']['share']:.1%} / {row['captured']['met']:.1%}"


def rows_of(table: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    """A cascade's six operating-point rows in item 24's order."""
    return [table["cascades"][key]["operating_points"][mix][target_key(share)] for mix, share in POINTS]


def means_of(table: Mapping[str, Any]) -> dict[str, float]:
    """Each cascade's mean savings share over its six operating points, from the rows."""
    return {key: sum(row["savings_vs_A"]["share"] for row in rows_of(table, key)) / len(POINTS) for key in CASCADES}


def best_of(means: Mapping[str, float]) -> str | None:
    """Item 24's rule, from the means alone: the highest mean, ties (within rounding) to a pure cascade, then to the
    cheaper gate (the free checks, then Jev, then Sonnet); None where that leaves more than one."""
    top = max(means.values())
    tied = [key for key in CASCADES if means[key] >= top - ROUNDING]
    pure = [key for key in tied if parse_cascade(key)[0] == HAIKU] or tied
    order = [FREE_GATE, JEV_GATE, SONNET_GATE]  # item 24's "cheaper gate", typed out here as the rule states it
    cheapest = min(order.index(parse_cascade(key)[1]) for key in pure)
    left = [key for key in pure if order.index(parse_cascade(key)[1]) == cheapest]
    return left[0] if len(left) == 1 else None


def pt(value: float) -> str:
    """Seconds as milliseconds, to one decimal."""
    return f"{1000 * value:.1f} ms"


def record_lines(table: Mapping[str, Any], dev: Mapping[str, Any], file_hash: str) -> list[str]:
    """Amendment A17 item 29's lines, as the committed files give them, at the precision the record prints."""
    bootstrap = table["bootstrap"]
    header = (
        f"29. **Record.** The cascades' operating points on {table['split']} (run {table['run']}, n = {table['n']}, list "
        f"prices of {table['price_snapshot']}), from `figures/table_v4.json`, and the hash of each cascade's section of "
        f"`routers/operating_points.json` and of the whole file, as A9 item 18, A11 item 21 and A15 item 20 record the "
        f"routers'. Intervals are percentile 95% from {bootstrap['resamples']:,} paired resamples, and "
        f"`analysis/check_stage7.py` compares every number here with those files."
    )
    means = means_of(table)
    best = table["best_cascade"]
    heads = " | ".join(f"{mix.capitalize()} {target_key(share)}" for mix, share in POINTS)
    lines = [
        header,
        "    - **The fifteen cascades.** Each cell is the savings against A, as a share of A's cost, then the share of "
        "resamples in which the point meets its target. A fall back to A reads 0.0% / 100.0%: the point is A itself, "
        "which saves nothing and meets every target. The last column is item 24's mean savings over the six points.",
        "",
        f"      | Cascade | {heads} | Mean savings |",
        "      |---|---|---|---|---|---|---|---|",
    ]
    for key, label in CASCADES.items():
        cells = " | ".join(cell(row) for row in rows_of(table, key))
        lines.append(f"      | {label} | {cells} | {means[key]:.1%} |")
    ranked = sorted(means.items(), key=lambda kv: -kv[1])
    runner_up = ranked[1] if best["best"] == ranked[0][0] else ranked[0]
    lines += [
        "",
        f"    - **The best cascade** by item 24's rule is {best['label']}, with mean savings of {means[best['best']]:.1%} "
        f"over its six operating points. The next is {CASCADES[runner_up[0]]} at {runner_up[1]:.1%}. The rule left "
        f"{'a tie open' if best['open_case'] else 'no tie open'}, and ranking the cascades by dollars saved per 1,000 "
        f"requests rather than by share gives {'the same' if best['same_best_by_dollars'] else 'another'} cascade.",
        "    - **Section hashes.** Each cascade's section of `routers/operating_points.json`, written with sorted keys, has "
        "the 64-digit sha256:",
    ]
    for key, label in CASCADES.items():
        lines.append(f"      - {label}: `{section_hash(read_section(key))}`")
    jev = dev["first_jev_date"]
    created = dev["batches"][0]["created_at"]
    freeze_time = freeze_commit_time()
    lines += [
        f"    - **The whole file.** `routers/operating_points.json` as committed has the 64-digit sha256 `{file_hash}`, "
        f"which Stage 8's unlock verifies (item 26).",
        f"    - **Order** (item 20.6). Dev's first Jev call's server `Date` is `{jev}` and the dev Sonnet batch "
        f"`{dev['batches'][0]['id']}` was created at {created}. Both come after the freeze commit `{FREEZE_COMMIT}`, "
        f"committed {freeze_time}. The `Date` is in GMT.",
    ]
    lat, conditions = dev["jev_latency"], dev["conditions"]
    inputs = table["latency"]["inputs"]
    sonnet = inputs["sonnet_gate_s"].values()
    first, last = dev["jev_collected"]
    lines += [
        f"    - **Dev's latency conditions** (item 16). The Jev gate's {lat['count']} dev calls, one at a time, on "
        f"{conditions['machine']}, over {conditions['network']}; collected from {first} to {last}, the conditions captured "
        f"at {conditions['captured_at']}, before the first call. Median {pt(lat['median_s'])} and p90 "
        f"{pt(lat['p90_s'])}; the first call {pt(lat['first_s'])} and the reconnects "
        f"{' and '.join(pt(s) for s in lat['reconnects_s'])}, which the median leaves out.",
        f"    - The Sonnet gate has no dev latency: the estimate uses the 80-call train sample's per-task medians, "
        f"{min(sonnet):.2f} to {max(sonnet):.2f} s, and the code check's dev median is {inputs['code_check_s']:.2f} s over "
        f"{dev['code_checks']} checks. Every policy's estimated p50 latency is in `figures/table_v4.json`.",
        f"    - The comparisons in `figures/table_v4.json` are {DESCRIPTIVE}.",
    ]
    return lines


def freeze_commit_time() -> str:
    """The freeze commit's commit time as git gives it, or the one item 28 records where the clone lacks the commit."""
    shown = subprocess.run(["git", "show", "-s", "--format=%cI", FREEZE_COMMIT], cwd=PROJECT_ROOT,
                           capture_output=True, text=True, check=False)  # fmt: skip
    return shown.stdout.strip() if shown.returncode == 0 and shown.stdout.strip() else "2026-11-10T21:51:04-08:00"


def record_mismatches(record: Sequence[str], good: Sequence[str]) -> list[tuple[str, str]]:
    """Each numeric line of the record that differs from the line the files give, paired with it; a line missing on either
    side is paired with ''."""
    return [(says, given) for says, given in zip_longest(numeric(list(record)), numeric(list(good)), fillvalue="")
            if says != given]  # fmt: skip


def slip_checks(good: Sequence[str]) -> None:
    """The record check catches a one-digit slip in a table cell, in a hash and in the first dev call's Date. A line the guard
    needs that the files' lines lack is a FAIL, not a crash."""

    def slipped(find: Callable[[str], bool], change: Callable[[str], str], what: str) -> None:
        index = next((i for i, line in enumerate(good) if find(line)), None)
        if index is None:
            check(False, f"the lines the files give hold the line this guard slips ({what})")
            return
        slip = change(good[index])
        check(slip != good[index] and len(record_mismatches([*good[:index], slip, *good[index + 1 :]], good)) == 1,
              f"the record check catches a slip in {what}")  # fmt: skip

    slipped(lambda line: line.startswith("      | ") and re.search(r"\d\.\d%", line) is not None,
            lambda line: re.sub(r"(\d+\.)(\d)%", lambda m: f"{m[1]}{(int(m[2]) + 1) % 10}%", line, count=1),
            "a table cell")  # fmt: skip
    slipped(lambda line: re.search(r"`[0-9a-f]{64}`", line) is not None,
            lambda line: re.sub(r"`([0-9a-f])", lambda m: f"`{'0' if m[1] != '0' else '1'}", line, count=1),
            "a hash")  # fmt: skip
    slipped(lambda line: "server `Date`" in line, lambda line: line.replace(":56 GMT", ":57 GMT"),
            "the first dev call's Date")  # fmt: skip


def own_guards() -> None:
    print("\n-- the check's own guards print FAIL, they do not raise, on what they can't read --")
    before = len(problems)
    with contextlib.redirect_stdout(io.StringIO()):
        slip_checks(["      | a line none of the guards looks for |"])
    failed = len(problems) - before
    del problems[before:]
    check(failed == 3, "known positive: lines without the three the guards slip are three FAILs, not a crash")
    check(best_of({k: 0.0 for k in CASCADES}) == next(iter(CASCADES)),
          "known positive: all fifteen tied at 0 go to the first pure cascade, the cheapest gate")  # fmt: skip


def record_checks(table: Mapping[str, Any], dev: Mapping[str, Any], file_hash: str) -> None:
    print("\n-- the record, Amendment A17 item 29 --")
    record = amendment_item(PLAN.read_text(), "A17", 29).splitlines()
    good = record_lines(table, dev, file_hash)
    mismatches = record_mismatches(record, good)
    check(not mismatches,
          "every number in Amendment A17 item 29's record is the one the committed files give, at the precision the record "
          "prints, and the hashes are the sections' and the file's"
          + "".join(f"\n      record: {says!r}\n      files:  {gives!r}" for says, gives in mismatches))  # fmt: skip
    text = "".join(record)
    check("It will hold" not in text and len(numeric(record)) == len(numeric(good)),
          "the record is filled in: every line the files give is in it")  # fmt: skip
    check(DESCRIPTIVE in text and "no multiplicity correction" in text,
          "the record calls the comparisons descriptive, with no multiplicity correction")  # fmt: skip
    check(not record_mismatches(good, good), "known positive: the lines the files give match themselves")
    slip_checks(good)
    check(len(record_mismatches([*good, "    - One number more: 1.0 points."], good)) == 1,
          "the record check catches a number it doesn't compare")  # fmt: skip
    check(len(record_mismatches(good[:-1], good)) == 1, "and a line missing from the record")


def sources_checks(table: Mapping[str, Any], dev: Mapping[str, Any], file_hash: str) -> None:
    print("\n-- what the record rests on, recomputed --")
    means = means_of(table)
    best = table["best_cascade"]
    check(best["best"] == best_of(means) and best["best"] is not None and not best["open_case"]
          and all(abs(best["mean_savings_share"][k] - means[k]) < ROUNDING for k in CASCADES),
          f"the best cascade is {CASCADES[best['best']]}, and the table's means are the rows' savings averaged over six points")  # fmt: skip
    # Each operating point's cost is the section's cheapest certified step, and its savings are A's cost less that over A's.
    wrong = []
    for key in CASCADES:
        section = read_section(key)
        for mix, share in POINTS:
            row, point = (
                table["cascades"][key]["operating_points"][mix][target_key(share)],
                section[mix][target_key(share)],
            )
            steps = point["steps"]
            certified = steps[: next((i for i, s in enumerate(steps) if not s["clears"]), len(steps))]
            cost = min((s["cost_per_1k"] for s in certified), default=steps[0]["cost_per_1k"])
            a_cost = steps[0]["cost_per_1k"]
            ok = abs(row["cost_per_1k"]["value"] - cost) < ROUNDING and row["fell_back"] == point["fell_back"]
            ok &= abs(row["savings_vs_A"]["share"] - (1 - cost / a_cost)) < ROUNDING
            if not ok:
                wrong.append(f"{CASCADES[key]}, {mix} {target_key(share)}")
    rows = [
        row
        for key in CASCADES
        for mix, share in POINTS
        for row in [table["cascades"][key]["operating_points"][mix][target_key(share)]]
    ]
    check(all(row["savings_vs_A"]["share"] == 0.0 and row["captured"]["met"] == 1.0 for row in rows if row["fell_back"])
          and sum(row["fell_back"] for row in rows) > 0,
          "a fall back to A saves 0.0% and meets its target in every resample, as the record's table says")  # fmt: skip
    check(not wrong, "every operating point's cost and savings in table v4 are the section's cheapest certified step and "
          "A's cost" + "".join(f"\n      differs: {w}" for w in wrong))  # fmt: skip
    check(all(section_hash(read_section(key)) == hashlib.sha256(json.dumps(read_section(key), sort_keys=True, indent=2)
                                                                .encode()).hexdigest() for key in CASCADES),
          "each section's hash is the sha256 of the section written with sorted keys")  # fmt: skip
    check(file_hash == whole_file_sha256() and len(file_hash) == 64 and whole_file_sha256().startswith("d183dd1c"),
          "the whole operating-points file's sha256 is the committed bytes', the one #70 recorded")  # fmt: skip
    check(set(json.loads(OPERATING_POINTS_FILE.read_text())) >= set(CASCADES),
          "the file holds all fifteen cascade sections")  # fmt: skip
    # The order, as git has it where the freeze commit is present.
    present = subprocess.run(["git", "cat-file", "-e", f"{FREEZE_COMMIT}^{{commit}}"], cwd=PROJECT_ROOT,
                             capture_output=True, check=False).returncode == 0  # fmt: skip
    if not present:
        print(f"(commit {FREEZE_COMMIT[:7]} is not in this clone: the order against it is skipped)")
        return
    frozen = datetime.fromisoformat(freeze_commit_time())
    called = parsedate_to_datetime(dev["first_jev_date"])
    created = datetime.fromisoformat(dev["batches"][0]["created_at"])
    check(called > frozen and created > frozen and dev["first_jev_date_parsed"] == called.isoformat(),
          "dev's first Jev call and the dev batch come after the freeze commit (item 20.6)")  # fmt: skip


# ---- the README's numbers -------------------------------------------------------------------------------------------


def readme_facts(table: Mapping[str, Any]) -> list[str]:
    """The README's sentences that quote a number about Stage 7, as the committed files give it."""
    best = table["best_cascade"]
    means = means_of(table)
    realistic = {key: table["cascades"][key]["operating_points"]["realistic"]["97%"] for key in CASCADES}

    def cost(key: str) -> float:
        return float(realistic[key]["cost_per_1k"]["value"])

    frontier = next(iter(realistic.values()))
    a_dollars = frontier["cost_per_1k"]["value"] + frontier["savings_vs_A"]["per_1k"]
    cheaper = [CASCADES[key] for key in CASCADES if key != best["best"] and cost(key) < cost(best["best"])]
    fell = sum(row["fell_back"] for key in CASCADES for rows in table["cascades"][key]["operating_points"].values()
               for row in rows.values())  # fmt: skip
    return [
        f"By Amendment A17 item 24's rule the best cascade on dev is {best['label']}, with mean savings against A of "
        f"{means[best['best']]:.1%} over its six operating points.",
        f"At the 97% target on the realistic mix {best['label']} costs ${cost(best['best']):.2f} per 1,000 requests against "
        f"A's ${a_dollars:.2f}, and {' and '.join(cheaper)} "
        f"{'costs' if len(cheaper) == 1 else 'cost'} less there.",
        f"{fell} of the 90 operating points fall back to A.",
    ]


def readme_checks(table: Mapping[str, Any]) -> None:
    print("\n-- the README's numbers about Stage 7 --")
    text = " ".join(README.read_text().split())
    facts = readme_facts(table)
    for fact in facts:
        check(fact in text, f"the README says: {fact}")
    slip = re.sub(r"(\d)(?=[ ,.)%]|$)", lambda m: str((int(m[1]) + 1) % 10), facts[0], count=1)
    check(slip != facts[0] and slip not in text,
          "known positive: a one-digit slip in the first sentence is not in the README")  # fmt: skip


def main() -> int:
    table = json.loads(TABLE_V4.read_text())
    dev = json.loads(DEV_RECORD.read_text())
    file_hash = whole_file_sha256()
    if "--print" in sys.argv[1:]:
        print("\n".join(record_lines(table, dev, file_hash)))
        return 0
    if "--print-readme" in sys.argv[1:]:
        print("\n".join(readme_facts(table)))
        return 0
    record_checks(table, dev, file_hash)
    sources_checks(table, dev, file_hash)
    readme_checks(table)
    own_guards()
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
