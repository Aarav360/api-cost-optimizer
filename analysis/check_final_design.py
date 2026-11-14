"""Recompute the numbers Amendment A19 quotes and compare them with its text (PLAN_v4.md; Amendment A19, items 6, 9
and 13; CODING_STANDARDS.md, "Numbers in documents").

Run from the project root:  uv run python -m analysis.check_final_design

Reads no test item and calls nothing. Item 6's numbers come from the committed tables (`figures/table_v3.json`,
`figures/table_v4.json`) and the operating-points file; item 9's Sonnet cost from `analysis/gates_dev_record.json`, the
caps from A17 item 18 and the spend before Stage 6 from A6's record. Where results/ holds them, the spend since is
recomputed from the stored Jev router records, at `analysis/collect_jev.py`'s price, and the stored gate records,
through `recorded_spend` in `analysis/gate_spend.py`. CI has no results/, so that part prints a SKIP line there.
"""

import json
import sys
from collections.abc import Mapping
from statistics import NormalDist

from analysis import collect_jev
from analysis.gate_spend import recorded_spend
from analysis.operating_points import OPERATING_POINTS_FILE
from analysis.plan import PLAN, amendment_item
from analysis.router_names import CASCADES, G_PLUS, HAIKU, JEV_GATE, NAMED_ROUTERS, G, cascade_section
from harness.client import PROJECT_ROOT

TABLE_V3 = PROJECT_ROOT / "figures" / "table_v3.json"
TABLE_V4 = PROJECT_ROOT / "figures" / "table_v4.json"
DEV_RECORD = PROJECT_ROOT / "analysis" / "gates_dev_record.json"
MIXES = ("realistic", "uniform")
BEFORE_STAGE_6 = 38  # A6's record: "That brings spending to about $38 of the $150 ceiling."
NORMAL = NormalDist()

problems: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        problems.append(what)


def points(by_mix: Mapping[str, Mapping[str, dict]]) -> list[dict]:
    """A policy's six operating points: two mixes, three quality targets."""
    return [point for mix in MIXES for point in by_mix[mix].values()]


def fresh_miss(dev_miss: float) -> float:
    """A point's chance of missing its target on a fresh sample, from the share of dev's resamples that miss it. Dev's
    estimate and the fresh one each carry the same sampling error, so their gap's is √2 times one (a normal
    approximation; it ignores the bias that choosing on dev adds)."""
    if dev_miss <= 0.0 or dev_miss >= 1.0:
        return dev_miss
    return NORMAL.cdf(NORMAL.inv_cdf(dev_miss) / 2**0.5)


def section(plan: str, heading: str, after: str) -> str:
    """The plan's text from one amendment heading to the next."""
    start = plan.index(heading)
    return plan[start : plan.index(after, start)]


def headline_and_misses(plan: str) -> None:
    item = amendment_item(plan, "A19", 6)
    print("-- item 6: the headline's dev numbers, the routing points and the expected misses --")
    v3 = json.loads(TABLE_V3.read_text(encoding="utf-8"))["operating_points"]
    v4 = json.loads(TABLE_V4.read_text(encoding="utf-8"))
    best = v4["best_cascade"]["best"]
    check(best == cascade_section(HAIKU, JEV_GATE) and f"That is {CASCADES[best]} today" in item,
          f"A17 item 24's rule picks {best} in table v4, as item 6 says")  # fmt: skip
    headline = v4["cascades"][best]["operating_points"]["realistic"]["99%"]
    saves, met = headline["savings_vs_A"]["share"], headline["captured"]["met"]
    check(not headline["fell_back"]
          and f"saves {saves:.1%} of A's cost and meets its target in {met:.2%} of dev's resamples" in item,
          f"the headline's point on dev saves {saves:.1%} and meets its target in {met:.2%} of resamples, as quoted")  # fmt: skip

    sections = json.loads(OPERATING_POINTS_FILE.read_text(encoding="utf-8"))
    routing = {name: sum(not p["fell_back"] for p in points(s)) for name, s in sections.items()}
    cascades = [name for name in sections if name in CASCADES]
    routed = sum(routing[name] for name in cascades)
    every = 6 * len(cascades)
    routers = {letter: routing[name] for letter, name in NAMED_ROUTERS.items()}
    by_router = ", ".join(f"{letter} {n}" for letter, n in routers.items())
    total = routed + sum(routers.values())
    check(f"{total} operating points route: {routed} of the cascades' {every} and {sum(routers.values())} of the "
          f"routers' {6 * len(routers)} ({by_router})" in item,
          f"{total} operating points route ({routed} cascades', {by_router}), as quoted")  # fmt: skip

    covered = [p for letter in (G, G_PLUS) for p in points(v3[letter]) if not p["fell_back"]]
    covered += [p for c in v4["cascades"].values() for p in points(c["operating_points"]) if not p["fell_back"]]
    check(len(covered) == routed + routers[G] + routers[G_PLUS],
          f"tables v3 and v4 hold the {len(covered)} routing points of the cascades, G and G+, as the file does")  # fmt: skip
    dev_misses = sum(1 - p["captured"]["met"] for p in covered)
    fresh_misses = sum(fresh_miss(1 - p["captured"]["met"]) for p in covered)
    check(f"Over the {len(covered)} that tables v3 and v4 cover" in item
          and f"about {round(dev_misses)} expected misses" in item and f"gives about {round(fresh_misses)}," in item
          and f"about {round(dev_misses) + 1} expected misses" not in item,
          f"the expected misses are about {round(dev_misses)} by dev's resamples ({dev_misses:.2f}) and about "
          f"{round(fresh_misses)} on a fresh sample ({fresh_misses:.2f}), as quoted")  # fmt: skip
    at_margin, fresh = NORMAL.cdf(-1.0), fresh_miss(NORMAL.cdf(-1.0))
    check(f"misses {at_margin:.1%} of dev's resamples but about {round(100 * fresh)}% of fresh samples" in item,
          f"a point exactly at the margin misses {at_margin:.1%} of dev's resamples and {fresh:.1%} of fresh samples")  # fmt: skip
    check(abs(fresh_miss(0.5) - 0.5) < 1e-12 and fresh_miss(0.0) == 0.0 and fresh_miss(1.0) == 1.0
          and 0.239 < fresh < 0.240,
          "the fresh-sample miss keeps a coin flip a coin flip and 0 and 1 as they are, and takes 15.9% to 24.0%")  # fmt: skip


def spend(plan: str) -> None:
    item = amendment_item(plan, "A19", 9)
    print("\n-- item 9: the caps, the Sonnet gate's dev cost and the spend so far --")
    caps = amendment_item(plan, "A17", 18)
    check(all(words in caps for words in ("$25 for its gate calls", "about $5", "$10 for the live replay"))
          and "$25" in item and "$10" in item and "A17 item 18's \"about $5\"" in item,
          "A17 item 18's $25 and $10 caps and its \"about $5\" are the ones item 9 quotes")  # fmt: skip
    batch = json.loads(DEV_RECORD.read_text(encoding="utf-8"))["spend"]["batch"]
    check(f"The Sonnet gate on dev cost ${batch:.2f}" in item and f"nearer ${round(batch)}" in item,
          f"the dev run's record prices the Sonnet gate's dev batch at ${batch:.2f}, as quoted")  # fmt: skip
    record = section(plan, "## Amendment A6", "## Amendment A7")
    ceiling = amendment_item(plan, "A1", 9)
    check(f"spending to about ${BEFORE_STAGE_6} of the $150 ceiling" in record and "$150" in ceiling
          and f"A6's \"about ${BEFORE_STAGE_6}\"" in item and "$150 ceiling" in item,
          f"A6's record puts the spend before Stage 6 at about ${BEFORE_STAGE_6}, and A1 item 9 the ceiling at $150")  # fmt: skip
    if not any(collect_jev.records_path(router, "train").is_file() for router in collect_jev.ROUTERS):
        print("SKIP  the spend since A6, from the stored Jev and gate records: results/ isn't here (CI)")
        return
    loaded = collect_jev.load_settings()
    jev = sum(
        collect_jev.overhead_dollars(record, loaded)
        for router in collect_jev.ROUTERS
        for split in collect_jev.SPLITS
        if collect_jev.records_path(router, split).is_file()
        for record in collect_jev.read_records(router, split)
    )
    gates = recorded_spend().total
    so_far = round(BEFORE_STAGE_6 + jev + gates)
    check(f"${jev:.2f} for the Jev routers' stored calls" in item and f"${gates:.2f} for Stage 7's stored gate records" in item,
          f"the stored records price the Jev routers' calls at ${jev:.2f} and Stage 7's gates at ${gates:.2f}, as quoted")  # fmt: skip
    check(f"About ${so_far} is spent so far" in item and f"about ${so_far + 25 + 10} of the $150 ceiling at worst" in item,
          f"so about ${so_far} is spent, and the caps leave about ${so_far + 35} at worst, as quoted")  # fmt: skip


def main() -> int:
    plan = PLAN.read_text(encoding="utf-8")
    headline_and_misses(plan)
    spend(plan)
    print("\nall checks passed" if not problems else f"\n{len(problems)} checks failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
