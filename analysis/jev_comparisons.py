"""G's and G+'s comparisons and diagnostics on dev (PLAN_v4.md §8 3c and 3c+; Amendment A15, items 10, 14 and 16;
Amendment A11, items 12 and 13; Amendment A7, item 2; issue #41).

Run from the project root:  uv run python -m analysis.jev_comparisons

Declared in A15 item 14 and run after the walks (`analysis/jev_walk.py`). None selects a setting or threshold, and
nothing here writes to the operating-points file: it reads the walks, never the section writer. Stage 6 · 12's chart
and table v3 call these functions; `main` prints what they return. No Jev call and no test item: the stored dev
records and the dev matrix only.

- **Same-target comparisons** (`same_target_cells`): G against E and F; G+ against E, F-train, F and G, at each mix and
  target, with `analysis.same_target`'s fallback rules. G against F-train is rejected (G uses no labels).
- **Matched cost** (`matched_cost_rows`): each operating point's paired quality difference from the envelope at the
  point's own cost, as E's and F's rows in table v1 (`Bootstrap.matched_cost`).
- **McNemar rows** (`mcnemar_rows`): per task, each operating point against A, null where it falls back to A.
- **Within-task concordance** (`concordance_readings`): per config, task cells only, since G and G+ are each one
  model as F-train is; and the difference G+ less G (`concordance_gap`). Per-task entries carry A7 item 2's warning.
- **Dev latency** (`dev_latency`, `dev_counts`): the successful attempt's median and p90, overall and per task, from the
  stored records, with the first call and the reconnects apart and retries and failed attempts as counts (item 10).
- **Descriptive**, with no multiplicity correction: at 95%, about one interval in twenty excludes 0 by chance.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from analysis import collect_jev
from analysis import jev_train_report as report
from analysis import trained_diagnostics as diagnostics
from analysis.benchmark import read_rows
from analysis.bootstrap import Bootstrap, Estimate, MatchedCost
from analysis.jev_table import dev_table, quantiles
from analysis.jev_walk import walks_of
from analysis.matrix import Matrix
from analysis.router_metrics import Difference, Reading
from analysis.router_names import F_TRAIN, G_PLUS, LETTER_OF, NAMED_ROUTERS, E, F, G
from analysis.router_panels import mcnemar_row
from analysis.router_table import RouterScoreTable
from analysis.same_target import SameTarget, compare_same_target
from analysis.simulator import always, load_mixes, simulate
from analysis.task_warnings import task_warning
from analysis.walk import Walk
from routers.ladder import FRONTIER, SCORED

JEV = (G, G_PLUS)  # the routers that get matched-cost, McNemar and concordance readings, in print order
# A15 item 14's same-target comparisons, first less second. G against F-train is rejected: G uses no labels.
COMPARISONS = ((G, E), (G, F), (G_PLUS, E), (G_PLUS, F_TRAIN), (G_PLUS, F), (G_PLUS, G))
DESCRIPTIVE = "descriptive, with no multiplicity correction: at 95%, about one interval in twenty excludes 0 by chance"

Walks = Mapping[str, Mapping[float, Walk]]  # one router's walks, by mix and then target share


class JevComparisonsError(ValueError):
    """Inputs the comparisons refuse whole: a router without walks or a table, or walks over different mixes."""


def excludes_zero(estimate: Estimate | None) -> bool:
    """Whether a 95% interval sits wholly above or wholly below 0 (None, a cell where both fall back, never does)."""
    return estimate is not None and (estimate.low > 0 or estimate.high < 0)


def same_target_cells(
    draws: Bootstrap,
    mixes: Mapping[str, dict[str, float]],
    walks: Mapping[str, Walks],
    pairs: tuple[tuple[str, str], ...] = COMPARISONS,
) -> dict[tuple[str, str], dict[str, dict[float, SameTarget]]]:
    """Each pair's operating points compared at every mix and target of its first router's walks: first less second,
    paired over `draws`, with `analysis.same_target`'s fallback rules (a side that falls back is A, both reads "both
    fall back to A")."""
    return {
        (a, b): {
            mix: {share: compare_same_target(draws, mixes[mix], walks[a][mix][share], walks[b][mix][share])
                  for share in walks[a][mix]}
            for mix in mixes}
        for a, b in pairs}  # fmt: skip


def matched_cost_rows(
    draws: Bootstrap, dev: Matrix, mixes: Mapping[str, dict[str, float]], walks: Walks
) -> dict[str, dict[float, MatchedCost]]:
    """One router's operating points' paired quality difference from the envelope at their own cost, by mix and
    target (A9 item 15's row, `Bootstrap.matched_cost`). A point that falls back to A is A, a corner of the envelope."""
    fixed = {config: simulate(dev, always(dev, config)) for config in (*SCORED, FRONTIER)}
    return {
        mix: {share: draws.matched_cost(list(walk.operating.outcomes), fixed, mixes[mix])
              for share, walk in walks[mix].items()}
        for mix in mixes}  # fmt: skip


def tasks_of(dev: Matrix) -> list[str]:
    """The dev tasks, in the order they first appear."""
    return list(dict.fromkeys(item.task for item in dev.items.values()))


def mcnemar_rows(
    dev: Matrix, mixes: Mapping[str, dict[str, float]], walks: Walks
) -> dict[str, dict[str, dict[float, dict[str, float] | None]]]:
    """McNemar's b, c and p of each operating point against A within each task: task, then mix, then target. A point
    that falls back to A is A and has nothing to test against: None."""
    return {
        task: {
            mix: {share: None if walk.fell_back else mcnemar_row(list(walk.operating.outcomes),
                                                                  list(walk.steps[0].outcomes), task)
                  for share, walk in walks[mix].items()}
            for mix in mixes}
        for task in tasks_of(dev)}  # fmt: skip


def concordance_readings(dev: Matrix, draws: Bootstrap, table: RouterScoreTable) -> dict[str, Reading]:
    """Within-task concordance per config for a single model: cells are the task alone (A15 item 14), pooled over
    pairs within a task and per task. Ties count half, and a task with no pair reads n/a."""
    return diagnostics.concordance_readings(dev, draws, table, diagnostics.cells_by_task(dev))


@dataclass(frozen=True)
class JevComparisons:
    """Every number of Stage 6 · 11, computed once, in memory. Nothing here is written to any file."""

    dev: Matrix
    draws: Bootstrap
    walks: Mapping[str, Walks]  # E, F, F-train, G, G+
    tables: Mapping[str, RouterScoreTable]  # G and G+
    same_target: dict[tuple[str, str], dict[str, dict[float, SameTarget]]]
    matched_cost: dict[str, dict[str, dict[float, MatchedCost]]]  # router, mix, target
    mcnemar: dict[str, dict[str, dict[str, dict[float, dict[str, float] | None]]]]  # router, task, mix, target
    concordance: dict[str, dict[str, Reading]]  # router, config
    concordance_gap: dict[str, Difference]  # G+ less G, per config
    latency: dict[str, report.Latency]  # dev, by collect_jev router name; empty where not given
    counts: dict[str, report.Counts]
    conditions: dict[str, object] = field(
        default_factory=dict
    )  # where and when dev's calls were timed; `compute` fills it


def compare(
    dev: Matrix,
    draws: Bootstrap,
    mixes: Mapping[str, dict[str, float]],
    walks: Mapping[str, Walks],
    tables: Mapping[str, RouterScoreTable],
) -> JevComparisons:
    """The comparisons and diagnostics from dev's matrix, the resamples, each router's walks (E, F, F-train, G and G+)
    and G's and G+'s tables. Refuses (JevComparisonsError) a router without walks or, for G and G+, a table, or walks
    that don't cover exactly `mixes`. Latency is left empty: `compute` fills it from the stored records."""
    needed = {name for pair in COMPARISONS for name in pair}
    if missing := sorted(needed - set(walks)):
        raise JevComparisonsError(f"no walks for {', '.join(missing)}")
    if missing := sorted(set(JEV) - set(tables)):
        raise JevComparisonsError(f"no router-score table for {', '.join(missing)}")
    if bad := sorted(name for name in needed if set(walks[name]) != set(mixes)):
        raise JevComparisonsError(f"{bad[0]}'s walks are not over exactly the mixes {sorted(mixes)}")
    concordance = {name: concordance_readings(dev, draws, tables[name]) for name in JEV}
    return JevComparisons(
        dev, draws, walks, tables,
        same_target_cells(draws, mixes, walks),
        {name: matched_cost_rows(draws, dev, mixes, walks[name]) for name in JEV},
        {name: mcnemar_rows(dev, mixes, walks[name]) for name in JEV},
        concordance, diagnostics.gaps(concordance[G_PLUS], concordance[G]), {}, {},
    )  # fmt: skip


def dev_latency() -> dict[str, report.Latency]:
    """Each router's dev latency from the stored records: the successful attempt's median and p90, overall and per
    task, the first call and every reconnect apart (item 10), keyed by `collect_jev.ROUTERS`' names."""
    task_of = {row["item_id"]: row["source"] for row in read_rows(split="dev")}
    return {r: report.latency(collect_jev.read_records(r, "dev"), task_of) for r in collect_jev.ROUTERS}


def dev_counts() -> dict[str, report.Counts]:
    """Each router's dev call counts, retries and failed attempts among them, from the stored records."""
    return {r: report.call_counts(collect_jev.read_records(r, "dev")) for r in collect_jev.ROUTERS}


def latency_conditions() -> dict[str, object]:
    """What a remote service's latency depends on (item 10: "machine, network, dates and times"): the machine and
    network this runs on, which hold the stored records (uncommitted `results/` lives on the collecting machine only),
    and the earliest and latest `collected_at` stamp over both routers' dev calls, from the records themselves."""
    first, last = report.collected_span(
        [r for router in collect_jev.ROUTERS for r in collect_jev.read_records(router, "dev")]
    )
    return {"machine": report.machine(), "network": report.network(), "collected_from": first, "collected_to": last}


def compute() -> JevComparisons:
    """All of Stage 6 · 11's numbers, from the dev matrix, the stored dev records and the committed files. E, F and
    F-train come from Stage 5's diagnostics (the same resamples), G and G+ from `jev_walk` over their dev tables."""
    base = diagnostics.compute()
    walks = {name: base.walks[name] for name in (E, F, F_TRAIN)}
    tables = {}
    for name in JEV:
        walks[name] = walks_of(name, base.draws)
        tables[name] = dev_table(NAMED_ROUTERS[name])
    found = compare(base.dev, base.draws, load_mixes(), walks, tables)
    return JevComparisons(
        **{**found.__dict__, "latency": dev_latency(), "counts": dev_counts(), "conditions": latency_conditions()}
    )


# ---------------------------------------------------------------------------------------------------------------
# JSON-ready data, for table v3 and the chart script


def _matched_data(row: MatchedCost) -> dict[str, float | None]:
    return {"envelope": row.envelope, "difference": row.difference, "low": row.low, "high": row.high,
            "unread": row.unread}  # fmt: skip


def latency_data(latency: Mapping[str, report.Latency], counts: Mapping[str, report.Counts]) -> dict[str, object]:
    """Dev latency in seconds, by router (G, G+): median and p90 overall and per task, the first call and the
    reconnects apart, and the retries and failed attempts as counts beside them."""
    return {
        LETTER_OF[r]: {
            "median": t.median, "p90": t.p90, "calls": t.count, "first_call": t.first, "reconnects": t.reconnects,
            "retries": counts[r].retries, "failed_attempts": counts[r].failed_attempts,
            "per_task": {task: {"median": m, "p90": p, "calls": n} for task, (m, p, n) in t.by_task.items()},
        }
        for r, t in latency.items()}  # fmt: skip


def to_data(result: JevComparisons) -> dict[str, object]:
    """The numbers as plain JSON-ready data. Same-target cells are keyed `first less second`, then mix, then target,
    as table v2's are; per-task entries carry A7 item 2's warning (null where there is none)."""
    dev = result.dev
    items = {t: sum(1 for i in dev.items.values() if i.task == t) for t in tasks_of(dev)}
    return {
        "descriptive": DESCRIPTIVE,
        "same_target": diagnostics.same_target_data(result.same_target),
        "matched_cost": {n: {mix: {f"{share:.0%}": _matched_data(row) for share, row in by.items()}
                             for mix, by in by_mix.items()} for n, by_mix in result.matched_cost.items()},
        "mcnemar": {n: {task: {mix: {f"{share:.0%}": row for share, row in by.items()} for mix, by in by_mix.items()}
                        for task, by_mix in by_task.items()} for n, by_task in result.mcnemar.items()},
        "within_task_concordance": {n: {c: diagnostics.reading_data(r, items) for c, r in by.items()}
                                    for n, by in result.concordance.items()},
        "concordance_differences": {f"{G_PLUS} less {G}": {c: diagnostics.reading_data(d, items)
                                                            for c, d in result.concordance_gap.items()}},
        "router_score_quantiles": {n: {c: {f"{q:g}": v for q, v in by.items()} for c, by in quantiles(t).items()}
                                   for n, t in result.tables.items()},
        "latency": latency_data(result.latency, result.counts) if result.latency else {},
        "latency_conditions": result.conditions,
    }  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------
# Printing


def _spread(estimate: Estimate | None, scale: float = 1.0, digits: int = 2) -> str:
    assert estimate is not None
    flag = "  (interval excludes 0)" if excludes_zero(estimate) else ""
    return (f"{scale * estimate.value:+.{digits}f} [{scale * estimate.low:+.{digits}f}, "
            f"{scale * estimate.high:+.{digits}f}]{flag}")  # fmt: skip


def print_same_target(result: JevComparisons) -> None:
    print(f"\nSame-target comparisons, first less second (cost in dollars per 1k, quality in points; 95% intervals; "
          f"{DESCRIPTIVE})")  # fmt: skip
    for (a, b), by_mix in result.same_target.items():
        print(f"  {a} against {b}")
        for mix, by_share in by_mix.items():
            for share, cell in by_share.items():
                if cell.both_fall_back:
                    print(f"    {mix:9} {share:4.0%}: both fall back to A")
                    continue
                sides = ("first" if cell.first_fell_back else "") + ("second" if cell.second_fell_back else "")
                print(f"    {mix:9} {share:4.0%}: cost {_spread(cell.cost)}; quality {_spread(cell.quality, 100, 1)}"
                      + (f"   ({sides} side falls back to A)" if sides else ""))  # fmt: skip


def print_matched_cost(result: JevComparisons) -> None:
    print("\nMatched-cost difference from the envelope (quality in points, at the operating point's own cost)")
    for name, by_mix in result.matched_cost.items():
        print(f"  {name}")
        for mix, by_share in by_mix.items():
            for share, row in by_share.items():
                if row.difference is None or row.low is None or row.high is None:
                    print(f"    {mix:9} {share:4.0%}: n/a (below the envelope's cheapest corner)")
                    continue
                print(f"    {mix:9} {share:4.0%}: envelope {100 * (row.envelope or 0):.1f}%, difference "
                      f"{100 * row.difference:+.1f} [{100 * row.low:+.1f}, {100 * row.high:+.1f}]"
                      + ("  (interval excludes 0)" if row.low > 0 or row.high < 0 else ""))  # fmt: skip


def print_mcnemar(result: JevComparisons) -> None:
    print("\nPer-task McNemar rows against A: b = only the router right, c = only A right, exact two-sided p")
    for name, by_task in result.mcnemar.items():
        print(f"  {name}")
        for task, by_mix in by_task.items():
            warn = task_warning(task, sum(1 for i in result.dev.items.values() if i.task == task))
            for mix, by_share in by_mix.items():
                cells = ", ".join("falls back to A" if row is None else f"b {row['b']:.0f} c {row['c']:.0f} p {row['p']:.3f}"
                                  for row in by_share.values())  # fmt: skip
                print(f"    {task:9} {mix:9} (100%, 99%, 97%): {cells}" + (f"   [{warn}]" if warn else ""))


def print_concordance(result: JevComparisons) -> None:
    print("\nWithin-task concordance, pairs within a task only (95% paired intervals)")
    items = {t: sum(1 for i in result.dev.items.values() if i.task == t) for t in tasks_of(result.dev)}

    def line(estimate) -> str:  # type: ignore[no-untyped-def]
        if estimate.value is None:
            return "n/a (no pair)"
        flag = "" if estimate.low is None else f" [{estimate.low:.3f}, {estimate.high:.3f}]"
        return f"{estimate.value:.3f}{flag}"

    for name, by_config in result.concordance.items():
        for config in SCORED:
            reading = by_config[config]
            print(f"  {name:3} {config:9} pooled {line(reading.pooled)} ({reading.size[None]} pairs)")
            for task, estimate in reading.per_task.items():
                warn = task_warning(task, items[task])
                print(f"      {'':9}   {task:9} {line(estimate)}" + (f"   [{warn}]" if warn else ""))
    print(f"\nDifference in concordance, {G_PLUS} less {G}")
    for config, diff in result.concordance_gap.items():
        print(f"  {config:9} pooled {line(diff.pooled)}")
        for task, estimate in diff.per_task.items():
            print(f"  {'':9}   {task:9} {line(estimate)}")


def print_latency(result: JevComparisons) -> None:
    print("\nLatency on dev (successful attempt's time alone, ms; the first call and reconnects apart)")
    where = result.conditions
    print(f"  machine: {where['machine']}; network: {where['network']}; dev calls collected from "
          f"{where['collected_from']} to {where['collected_to']}")  # fmt: skip
    for r, t in result.latency.items():
        name = LETTER_OF[r]
        c = result.counts[r]
        print(f"  {name:3} {t.count} calls: median {1000 * t.median:.1f}, p90 {1000 * t.p90:.1f}; retries {c.retries}, "
              f"failed attempts {c.failed_attempts}; first call {1000 * t.first:.1f}; "
              f"{len(t.reconnects)} reconnects ({', '.join(f'{1000 * s:.1f}' for s in t.reconnects) or 'none'})")  # fmt: skip
        for task, (m, p, n) in t.by_task.items():
            print(f"      {task:9} median {1000 * m:.1f}, p90 {1000 * p:.1f} ({n} calls)")


def main() -> None:
    result = compute()
    print_same_target(result)
    print_matched_cost(result)
    print_mcnemar(result)
    print_concordance(result)
    print_latency(result)
    print("\nNothing is written to the operating-points file; every comparison above is " + DESCRIPTIVE + ".")


if __name__ == "__main__":
    main()
